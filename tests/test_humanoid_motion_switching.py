from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "realtime" / "humanoid_robot" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_music_humanoid_matcher import (
    MotionEntryFeatures,
    MotionLoader,
    OneShotPhaseTracker,
    apply_terminal_safe_idle,
    choose_ready_transition,
    compute_transition_duration,
    frame_root_yaw,
    forced_transition_blend,
    initial_motion_candidates,
    sample_actuator_pose,
    select_initial_motion_id,
    select_motion_entry,
    wrist_motion_diagnostics,
)
from robot_motion import (
    JointDynamicsLimiter,
    JointDynamicsLimits,
    RobotMotionFrame,
    RootMotionContinuity,
    load_joint_dynamics_limits,
)


def entry_features(
    positions: np.ndarray,
    *,
    candidates: tuple[int, ...],
    salience: tuple[float, ...],
    contacts: np.ndarray | None = None,
) -> MotionEntryFeatures:
    matrix = np.asarray(positions, dtype=np.float64)
    frame_count, joint_count = matrix.shape
    velocities = np.zeros_like(matrix)
    if frame_count > 1:
        velocities[1:] = np.diff(matrix, axis=0) * 10.0
        velocities[0] = velocities[1]
    return MotionEntryFeatures(
        joint_names=tuple(f"joint_{index}" for index in range(joint_count)),
        joint_positions=matrix,
        joint_velocities=velocities,
        joint_ranges=np.full(joint_count, 2.0),
        root_positions=np.zeros((frame_count, 3)),
        root_tilt=np.zeros(frame_count),
        root_linear_velocities=np.zeros((frame_count, 3)),
        root_angular_speeds=np.zeros(frame_count),
        foot_contacts=(
            np.zeros((frame_count, 2), dtype=bool)
            if contacts is None
            else np.asarray(contacts, dtype=bool)
        ),
        candidate_indices=np.asarray(candidates, dtype=np.int32),
        candidate_salience=np.asarray(salience, dtype=np.float64),
        fps=10.0,
        duration=frame_count / 10.0,
    )


def startup_profile(
    motion_id: str,
    *,
    velocity_p90: float,
    duration_seconds: float = 12.0,
    preflight_passed: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        motion_id=motion_id,
        velocity_p90=velocity_p90,
        duration_seconds=duration_seconds,
        preflight_passed=preflight_passed,
        preflight_reason="ok" if preflight_passed else "failed",
    )


class InitialMotionSelectionTests(unittest.TestCase):
    def test_random_pool_uses_low_activity_long_motions_only(self) -> None:
        catalog = SimpleNamespace(
            motions={
                "quiet_short": startup_profile(
                    "quiet_short", velocity_p90=0.1, duration_seconds=4.0
                ),
                "quiet_long": startup_profile("quiet_long", velocity_p90=0.2),
                "medium_long": startup_profile("medium_long", velocity_p90=0.8),
                "loud_long": startup_profile("loud_long", velocity_p90=2.0),
                "failed": startup_profile(
                    "failed", velocity_p90=0.05, preflight_passed=False
                ),
            }
        )

        candidates = initial_motion_candidates(
            catalog,
            minimum_duration_seconds=9.0,
            low_activity_quantile=2.0 / 3.0,
        )

        self.assertEqual(("quiet_long", "medium_long"), candidates)

    def test_seeded_random_selection_is_reproducible(self) -> None:
        catalog = SimpleNamespace(
            motions={
                motion_id: startup_profile(motion_id, velocity_p90=velocity)
                for motion_id, velocity in (
                    ("quiet_a", 0.1),
                    ("quiet_b", 0.2),
                    ("quiet_c", 0.3),
                    ("loud", 2.0),
                )
            }
        )
        args = SimpleNamespace(
            initial_motion_id=None,
            initial_motion_low_activity_quantile=0.75,
            initial_motion_seed=1234,
            match_window_seconds=6.0,
            match_interval_seconds=1.0,
            switch_required_wins=3,
        )

        first = select_initial_motion_id(args, catalog)
        second = select_initial_motion_id(args, catalog)

        self.assertEqual(first, second)
        self.assertIn(first, {"quiet_a", "quiet_b", "quiet_c"})

    def test_explicit_initial_motion_overrides_random_pool(self) -> None:
        catalog = SimpleNamespace(
            motions={
                "quiet": startup_profile("quiet", velocity_p90=0.1),
                "requested": startup_profile("requested", velocity_p90=3.0),
            }
        )
        args = SimpleNamespace(initial_motion_id="requested")

        self.assertEqual("requested", select_initial_motion_id(args, catalog))

    def test_startup_duration_covers_speed_and_preparation_reserve(self) -> None:
        catalog = SimpleNamespace(
            motions={
                "too_short": startup_profile(
                    "too_short",
                    velocity_p90=0.1,
                    duration_seconds=20.0,
                ),
                "safe_length": startup_profile(
                    "safe_length",
                    velocity_p90=0.2,
                    duration_seconds=21.0,
                ),
            }
        )
        args = SimpleNamespace(
            initial_motion_id=None,
            initial_motion_low_activity_quantile=1.0,
            initial_motion_seed=1,
            match_window_seconds=6.0,
            match_interval_seconds=1.0,
            switch_required_wins=3,
            startup_ready_reserve_seconds=4.0,
            speed_max=1.6,
        )

        self.assertEqual("safe_length", select_initial_motion_id(args, catalog))


class OneShotMotionTests(unittest.TestCase):
    def test_phase_wrap_holds_terminal_frame(self) -> None:
        tracker = OneShotPhaseTracker(0.8)

        phase, ended = tracker.clamp(0.95)
        self.assertAlmostEqual(0.95, phase)
        self.assertFalse(ended)

        phase, ended = tracker.clamp(0.02)
        self.assertEqual(OneShotPhaseTracker.terminal_phase, phase)
        self.assertTrue(ended)
        self.assertEqual((phase, True), tracker.clamp(0.25))

    def test_phase_never_moves_backward_before_wrap(self) -> None:
        tracker = OneShotPhaseTracker(0.4)
        phase, ended = tracker.clamp(0.39)
        self.assertEqual(0.4, phase)
        self.assertFalse(ended)

    def test_matcher_never_samples_loop_interpolation_interval(self) -> None:
        sampler = SimpleNamespace(frames=[0, 1, 2, 3])
        controller = SimpleNamespace(update=lambda _now: (0.999, 1.0, 0.0, 0.0))
        sampled_phases: list[float] = []

        def sample_frame(_sampler: object, **kwargs: object) -> RobotMotionFrame:
            sampled_phases.append(float(kwargs["phase"]))
            return RobotMotionFrame({"joint": 0.0})

        with patch(
            "realtime_music_humanoid_matcher.base.sample_robot_motion_frame",
            side_effect=sample_frame,
        ):
            _frame, reported_phase = sample_actuator_pose(
                sampler,
                controller,
                0.0,
                SimpleNamespace(),
                None,
                None,
                OneShotPhaseTracker(),
            )

        self.assertAlmostEqual(0.999, reported_phase)
        self.assertLess(sampled_phases[0], 0.75)
        self.assertAlmostEqual(0.999 * 3.0 / 4.0, sampled_phases[0])

    def test_forced_blend_reaches_one_exactly_at_terminal_phase(self) -> None:
        start = 0.8
        self.assertLess(forced_transition_blend(0.9, start), 1.0)
        self.assertEqual(
            1.0,
            forced_transition_blend(OneShotPhaseTracker.terminal_phase, start),
        )

    def test_forced_blend_uses_elapsed_time_when_source_already_ended(self) -> None:
        terminal = OneShotPhaseTracker.terminal_phase
        self.assertEqual(
            0.0,
            forced_transition_blend(
                terminal,
                terminal,
                elapsed_seconds=0.0,
                duration_seconds=0.5,
            ),
        )
        self.assertAlmostEqual(
            0.5,
            forced_transition_blend(
                terminal,
                terminal,
                elapsed_seconds=0.25,
                duration_seconds=0.5,
            ),
        )
        self.assertEqual(
            1.0,
            forced_transition_blend(
                terminal,
                terminal,
                elapsed_seconds=0.5,
                duration_seconds=0.5,
            ),
        )

    def test_terminal_safe_idle_moves_joints_without_moving_root(self) -> None:
        frame = RobotMotionFrame(
            {
                "waist_pitch_joint": 0.1,
                "left_shoulder_pitch_joint": -0.2,
                "unrelated_joint": 0.3,
            },
            root_position=np.asarray([1.0, 2.0, 3.0]),
            root_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        )

        idle = apply_terminal_safe_idle(frame, elapsed_seconds=1.25, amplitude=0.02)

        self.assertAlmostEqual(0.12, idle.joint_positions["waist_pitch_joint"])
        self.assertAlmostEqual(-0.191, idle.joint_positions["left_shoulder_pitch_joint"])
        self.assertEqual(0.3, idle.joint_positions["unrelated_joint"])
        np.testing.assert_allclose(idle.root_position, frame.root_position)
        np.testing.assert_allclose(
            idle.root_quaternion_wxyz,
            frame.root_quaternion_wxyz,
        )

    def test_no_ready_candidate_holds_terminal_without_wrap(self) -> None:
        tracker = OneShotPhaseTracker(0.95)
        terminal, ended = tracker.clamp(0.01)
        self.assertTrue(ended)
        self.assertEqual(OneShotPhaseTracker.terminal_phase, terminal)
        for cyclic_phase in (0.1, 0.5, 0.9):
            self.assertEqual((terminal, True), tracker.clamp(cyclic_phase))

    def test_wrist_trace_diagnostics_separate_angle_and_speed(self) -> None:
        previous = {
            "left_wrist_roll_joint": 0.2,
            "right_wrist_yaw_joint": -0.1,
        }
        frame = RobotMotionFrame(
            {
                "left_wrist_roll_joint": 0.5,
                "right_wrist_yaw_joint": -0.9,
                "left_elbow_joint": 2.0,
            }
        )

        diagnostics, positions = wrist_motion_diagnostics(frame, previous, 0.1)

        self.assertEqual("right_wrist_yaw_joint", diagnostics.max_abs_joint)
        self.assertAlmostEqual(0.9, diagnostics.max_abs_rad)
        self.assertEqual("right_wrist_yaw_joint", diagnostics.max_speed_joint)
        self.assertAlmostEqual(8.0, diagnostics.max_speed_rad_s)
        self.assertNotIn("left_elbow_joint", positions)


class MotionEntrySelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = {
            "joint_0": JointDynamicsLimits(16.0, 2000.0),
            "joint_1": JointDynamicsLimits(16.0, 2000.0),
        }

    def test_pose_velocity_and_contact_compatibility_outweigh_salience(self) -> None:
        source = entry_features(
            np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            candidates=(0,),
            salience=(1.0,),
            contacts=np.asarray([[True, False]] * 3),
        )
        target_positions = np.zeros((20, 2))
        target_positions[4] = (0.05, -0.05)
        target_positions[10] = (1.0, -1.0)
        contacts = np.zeros((20, 2), dtype=bool)
        contacts[4] = (True, False)
        contacts[10] = (False, True)
        target = entry_features(
            target_positions,
            candidates=(4, 10),
            salience=(0.2, 1.0),
            contacts=contacts,
        )

        selected = select_motion_entry(
            source,
            1.0,
            target,
            self.limits,
            beat_period=0.1,
            beats_per_bar=1,
            minimum_remaining_bars=0.0,
            speed_max=1.0,
            transition_minimum=0.05,
            transition_maximum=0.5,
        )

        self.assertEqual(4, selected.frame_index)

    def test_rejects_entry_without_required_remaining_bar(self) -> None:
        source = entry_features(
            np.zeros((3, 2)),
            candidates=(0,),
            salience=(1.0,),
        )
        target = entry_features(
            np.zeros((40, 2)),
            candidates=(4, 38),
            salience=(0.4, 1.0),
        )

        selected = select_motion_entry(
            source,
            1.0,
            target,
            self.limits,
            beat_period=0.5,
            beats_per_bar=4,
            minimum_remaining_bars=1.0,
            speed_max=1.0,
            transition_minimum=0.1,
            transition_maximum=0.5,
        )

        self.assertEqual(4, selected.frame_index)

    def test_transition_duration_is_half_beat_quantized_and_bounded(self) -> None:
        duration = compute_transition_duration(
            np.asarray([0.0]),
            np.asarray([1.0]),
            np.asarray([2.0]),
            np.asarray([10.0]),
            beat_period=0.5,
            minimum=0.35,
            maximum=1.2,
        )
        self.assertAlmostEqual(1.0, duration)
        bounded = compute_transition_duration(
            np.asarray([0.0]),
            np.asarray([100.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
            beat_period=0.5,
            minimum=0.35,
            maximum=1.2,
        )
        self.assertAlmostEqual(1.2, bounded)

    def test_ready_pool_uses_prepared_fallback_when_preferred_is_not_ready(self) -> None:
        source_features = entry_features(
            np.zeros((20, 2)),
            candidates=(0,),
            salience=(1.0,),
        )
        fallback_features = entry_features(
            np.zeros((20, 2)),
            candidates=(2,),
            salience=(0.8,),
        )
        source_sampler = SimpleNamespace(entry_features=source_features)
        fallback_sampler = SimpleNamespace(entry_features=fallback_features)

        class Loader:
            score_seconds: list[float] = []

            @staticmethod
            def take_ready(motion_id: str) -> object | None:
                return fallback_sampler if motion_id == "fallback" else None

        args = SimpleNamespace(
            switch_beats_per_bar=1,
            switch_min_remaining_bars=0.0,
            speed_max=1.0,
            transition_min_seconds=0.05,
            transition_max_seconds=0.5,
        )
        selected = choose_ready_transition(
            Loader(),
            source_sampler,
            0.5,
            ("preferred", "fallback"),
            self.limits,
            args,
            beat_period=0.1,
            preferred_id="preferred",
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual("fallback", selected[0])
        self.assertTrue(selected[3])

    def test_failed_preparation_is_quarantined_for_fallback(self) -> None:
        loader = object.__new__(MotionLoader)
        failed_future: Future[object] = Future()
        failed_future.set_exception(RuntimeError("broken motion"))
        loader.cache = {}
        loader.prepare_futures = {"broken": failed_future}
        loader.futures = {"broken": Future()}
        loader.submitted_at = {"broken": 1.0}
        loader.failed_motions = {}

        self.assertIsNone(loader.take_ready("broken"))
        self.assertIn("broken", loader.failed_motions)
        self.assertNotIn("broken", loader.prepare_futures)
        self.assertIsNone(loader.take_ready("broken"))


class JointDynamicsLimiterTests(unittest.TestCase):
    def test_static_step_respects_velocity_and_acceleration_and_does_not_overshoot(self) -> None:
        limits = {"joint": JointDynamicsLimits(1.0, 10.0)}
        limiter = JointDynamicsLimiter(limits)
        limiter.reset(RobotMotionFrame({"joint": 0.0}))
        previous_position = 0.0
        previous_velocity = 0.0
        for _ in range(300):
            frame = limiter.apply(RobotMotionFrame({"joint": 1.0}), 0.01)
            position = frame.joint_positions["joint"]
            velocity = (position - previous_position) / 0.01
            acceleration = (velocity - previous_velocity) / 0.01
            self.assertLessEqual(abs(velocity), 1.0 + 1e-9)
            self.assertLessEqual(abs(acceleration), 10.0 + 1e-7)
            self.assertLessEqual(position, 1.0 + 1e-9)
            previous_position = position
            previous_velocity = velocity
        self.assertGreater(limiter.activation_rate, 0.0)

    def test_invalid_dt_holds_previous_output(self) -> None:
        limiter = JointDynamicsLimiter({"joint": JointDynamicsLimits(1.0, 10.0)})
        limiter.reset(RobotMotionFrame({"joint": 0.25}))
        output = limiter.apply(RobotMotionFrame({"joint": 1.0}), math.nan)
        self.assertEqual(0.25, output.joint_positions["joint"])

    def test_reverse_target_respects_dynamics_and_reaches_target(self) -> None:
        limiter = JointDynamicsLimiter(
            {"joint": JointDynamicsLimits(2.0, 20.0)}
        )
        limiter.reset(RobotMotionFrame({"joint": 0.0}))
        previous_position = 0.0
        previous_velocity = 0.0
        targets = [1.0] * 100 + [-1.0] * 200
        for target in targets:
            output = limiter.apply(RobotMotionFrame({"joint": target}), 0.01)
            position = output.joint_positions["joint"]
            velocity = (position - previous_position) / 0.01
            acceleration = (velocity - previous_velocity) / 0.01
            self.assertLessEqual(abs(velocity), 2.0 + 1e-9)
            self.assertLessEqual(abs(acceleration), 20.0 + 1e-7)
            previous_position = position
            previous_velocity = velocity
        self.assertAlmostEqual(-1.0, previous_position, places=6)

    def test_missing_and_unlimited_joints_do_not_break_state(self) -> None:
        limiter = JointDynamicsLimiter(
            {"limited": JointDynamicsLimits(1.0, 10.0)}
        )
        limiter.reset(RobotMotionFrame({"limited": 0.0}))
        output = limiter.apply(RobotMotionFrame({"unlimited": 3.0}), 0.01)
        self.assertEqual({"unlimited": 3.0}, output.joint_positions)
        resumed = limiter.apply(
            RobotMotionFrame({"limited": 0.5, "unlimited": 4.0}),
            0.01,
        )
        self.assertLessEqual(resumed.joint_positions["limited"], 0.001 + 1e-9)
        self.assertEqual(4.0, resumed.joint_positions["unlimited"])

    def test_json_supports_default_and_per_joint_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "limits.json"
            path.write_text(
                json.dumps(
                    {
                        "default": {
                            "max_speed_rad_s": 4.0,
                            "max_acceleration_rad_s2": 40.0,
                        },
                        "joints": {
                            "knee": {"max_speed_rad_s": 2.0},
                        },
                    }
                ),
                encoding="utf-8",
            )
            limits = load_joint_dynamics_limits(
                path,
                {"hip", "knee"},
                default_speed=16.0,
                default_acceleration=2000.0,
            )
        self.assertEqual(JointDynamicsLimits(4.0, 40.0), limits["hip"])
        self.assertEqual(JointDynamicsLimits(2.0, 40.0), limits["knee"])

    def test_json_rejects_unknown_joint_and_nonpositive_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "limits.json"
            path.write_text(
                json.dumps({"joints": {"unknown": {"max_speed_rad_s": 1.0}}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unknown joints"):
                load_joint_dynamics_limits(
                    path,
                    {"known"},
                    default_speed=16.0,
                    default_acceleration=2000.0,
                )
            path.write_text(
                json.dumps({"default": {"max_speed_rad_s": 0.0}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                load_joint_dynamics_limits(
                    path,
                    {"known"},
                    default_speed=16.0,
                    default_acceleration=2000.0,
                )


class WorldRootContinuityTests(unittest.TestCase):
    def test_source_switch_keeps_nonzero_world_anchor_and_accumulates_motion(self) -> None:
        continuity = RootMotionContinuity(
            np.asarray([5.0, -3.0, 0.8]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            mode="continuous",
        )
        first = continuity.apply(
            RobotMotionFrame(
                {},
                np.asarray([0.0, 0.0, 0.0]),
                np.asarray([1.0, 0.0, 0.0, 0.0]),
            ),
            phase=0.5,
            source_id="first",
        )
        switched = continuity.apply(
            RobotMotionFrame(
                {},
                np.asarray([0.0, 0.0, 0.0]),
                np.asarray([1.0, 0.0, 0.0, 0.0]),
            ),
            phase=0.2,
            source_id="second",
        )
        moved = continuity.apply(
            RobotMotionFrame(
                {},
                np.asarray([1.0, 0.0, 0.0]),
                np.asarray([1.0, 0.0, 0.0, 0.0]),
            ),
            phase=0.3,
            source_id="second",
        )

        np.testing.assert_allclose(first.root_position[:2], [5.0, -3.0])
        np.testing.assert_allclose(switched.root_position[:2], [5.0, -3.0])
        np.testing.assert_allclose(moved.root_position[:2], [6.0, -3.0])

    def test_multiple_switches_preserve_nonzero_xy_and_yaw(self) -> None:
        yaw = 0.7
        anchor_quaternion = np.asarray(
            [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
        )
        continuity = RootMotionContinuity(
            np.asarray([2.0, 4.0, 0.8]),
            anchor_quaternion,
            mode="continuous",
        )
        identity = np.asarray([1.0, 0.0, 0.0, 0.0])
        outputs = []
        for source_id, phase, x in (
            ("first", 0.1, 0.0),
            ("first", 0.2, 0.5),
            ("second", 0.3, 0.0),
            ("second", 0.4, 0.5),
            ("third", 0.2, 0.0),
        ):
            outputs.append(
                continuity.apply(
                    RobotMotionFrame(
                        {},
                        np.asarray([x, 0.0, 0.0]),
                        identity,
                    ),
                    phase=phase,
                    source_id=source_id,
                )
            )

        for output in outputs:
            self.assertGreater(np.linalg.norm(output.root_position[:2]), 1.0)
            self.assertAlmostEqual(yaw, frame_root_yaw(output), places=6)
        np.testing.assert_allclose(
            outputs[2].root_position,
            outputs[1].root_position,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            outputs[4].root_position,
            outputs[3].root_position,
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
