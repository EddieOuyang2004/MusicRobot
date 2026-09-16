"""Numerical and lifecycle checks for authored-state v2 bridges."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import Future
from unittest.mock import patch
import importlib.util
import math
import sys
import unittest
import tempfile
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_bridges import (AuthoredTrajectory, AuthoredClock, HermiteBridge,
    JointState, make_bridge, InfeasibleBridge, boundary_weight, load_jerk_limits)
import realtime_music_humanoid_matcher_v2 as v2


def state(q, v=0., a=0.):
    return JointState(*(np.atleast_1d(np.asarray(x, dtype=float)) for x in (q, v, a)))


def bridge(backend="hermite", start=None, end=None, **kwargs):
    values = dict(lower=np.array([-10.]), upper=np.array([10.]),
                  speed=np.array([10.]), acceleration=np.array([50.]),
                  jerk=np.array([500.]), minimum=.35)
    values.update(kwargs)
    return make_bridge(backend, start or state(0., .2, .1), end or state(1., -.1, -.2), **values)


class PolynomialTests(unittest.TestCase):
    def test_search_below_displacement_estimate(self):
        # Moving endpoints can connect in about .5 s although the old
        # rest-to-rest displacement estimate is .893 s (> maximum).
        curve = bridge(start=state(0., 2.), end=state(1., 2.),
                       lower=np.array([0.]), upper=np.array([1.]),
                       speed=np.array([2.1]), minimum=.35, maximum=.6)
        self.assertLessEqual(curve.duration, .6)
        self.assertTrue(curve.within_limits(0, np.array([0.]), np.array([1.])))
        self.assertTrue(curve.within_limits(1, np.array([-2.1]), np.array([2.1])))

    def test_refines_between_coarse_durations(self):
        # A narrow feasible interval lies between coarse .35 and .42 trials.
        with patch.object(HermiteBridge, 'within_limits',
                          autospec=True, side_effect=lambda curve, *a: .39 < curve.duration < .40):
            curve = bridge(start=state(0.), end=state(.01), maximum=.5)
        self.assertGreater(curve.duration, .39)
        self.assertLess(curve.duration, .40)

    def test_failure_reports_joint_constraint_and_duration(self):
        with self.assertRaisesRegex(InfeasibleBridge, 'duration=.*joint_index=.*constraint='):
            bridge(start=state(1., 1.), end=state(0.),
                   lower=np.array([-1.]), upper=np.array([1.]))

    def test_nonzero_boundary_derivatives(self):
        a, b = state([0., .1], [.2, -.3], [.1, .4]), state([1., -.2], [-.1, .4], [-.2, .1])
        for duration in (.001, .35, 10.):
            curve = HermiteBridge.between(a, b, duration)
            for t, target in ((0., a), (duration, b)):
                actual = curve.at_time(t)
                for field in ("position", "velocity", "acceleration"):
                    np.testing.assert_allclose(getattr(actual, field), getattr(target, field), atol=1e-7)

    def test_extrema_find_peaks_between_samples(self):
        curve = HermiteBridge.between(state(0.), state(1.), 1.)
        self.assertAlmostEqual(curve.extrema(1)[1][0], 1.875)
        self.assertAlmostEqual(curve.extrema(2)[1][0], 10/math.sqrt(3))
        self.assertAlmostEqual(curve.extrema(3)[1][0], 60.)

    def test_limits_and_jerk_drive_duration(self):
        curve = bridge(start=state(0.), end=state(1.), jerk=np.array([5.]))
        for derivative, limit in ((1, 10.), (2, 50.), (3, 5.)):
            low, high = curve.extrema(derivative)
            self.assertLessEqual(max(abs(low[0]), abs(high[0])), limit+1e-7)

    def test_infeasible_endpoints_and_duration(self):
        for values in (dict(start=state(11.)), dict(end=state(1., 11.)),
                       dict(end=state(1., 0., 51.)), dict(maximum=.01)):
            with self.assertRaises(InfeasibleBridge):
                bridge(**values)

    def test_internal_overshoot_rejected(self):
        # Both endpoint positions fit; outward velocity at upper bound cannot.
        with self.assertRaises(InfeasibleBridge):
            bridge(start=state(1., 1.), end=state(0.), lower=np.array([-1.]), upper=np.array([1.]))

    def test_ruckig_requires_explicit_jerk(self):
        with self.assertRaisesRegex(ValueError, "every joint"):
            bridge("ruckig", jerk=np.array([math.inf]))

    def test_ruckig_dependency_failure_is_clear(self):
        with patch("motion_bridges.importlib.import_module", side_effect=ImportError):
            with self.assertRaisesRegex(ValueError, "pip install ruckig"):
                bridge("ruckig")

    def test_jerk_configuration(self):
        self.assertTrue(math.isinf(load_jerk_limits(None, {"j"})["j"]))
        self.assertEqual(load_jerk_limits(None, {"j"}, 42)["j"], 42)
        for value in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                load_jerk_limits(None, {"j"}, value)

    def test_per_joint_jerk_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"limits.json"
            path.write_text(json.dumps({"default": {"max_jerk_rad_s3": 10},
                                         "joints": {"a": {"max_jerk_rad_s3": 20}}}))
            self.assertEqual(load_jerk_limits(path, {"a", "b"}, 5), {"a": 20, "b": 10})
            with self.assertRaisesRegex(ValueError, "Unknown joint"):
                load_jerk_limits(path, {"b"})


class AuthoredTests(unittest.TestCase):
    def test_constant_linear_quadratic_and_endpoint_sampling(self):
        one = AuthoredTrajectory([[2.]], 10)
        np.testing.assert_equal(one.at_time(.05).velocity, [0.])
        two = AuthoredTrajectory([[0.], [1.]], 10)
        np.testing.assert_allclose(two.at_time(.025).position, [.25])
        np.testing.assert_allclose(two.at_time(.025).velocity, [10.])
        np.testing.assert_allclose(two.at_time(.025).acceleration, [0.])
        t = np.arange(11)/10
        curve = AuthoredTrajectory((t*t)[:, None], 10)
        for seconds in (0., .037, .1, .45, 1.):
            actual = curve.at_time(seconds)
            np.testing.assert_allclose(actual.position, [seconds*seconds], atol=1e-12)
            np.testing.assert_allclose(actual.velocity, [2*seconds], atol=1e-12)
            np.testing.assert_allclose(actual.acceleration, [2.], atol=1e-10)
        for seconds in t[1:-1]:
            np.testing.assert_allclose(curve.at_time(seconds-1e-9).acceleration,
                                       curve.at_time(seconds+1e-9).acceleration, atol=1e-7)

    def test_fades_do_not_overlap_short_segments(self):
        self.assertEqual(boundary_weight(.8, .8, 1., .5), 0.)
        self.assertAlmostEqual(boundary_weight(.9, .8, 1., .5), 1.)
        self.assertEqual(boundary_weight(1., .8, 1., .5), 0.)

    def test_clock_carries_excess_wall_time(self):
        clock = AuthoredClock(1., .2, .1, 5.)
        self.assertAlmostEqual(clock.advance(6., 1.), .2)
        self.assertEqual(clock.seconds, 1.)

    def test_rate_returns_to_authored_at_exit(self):
        clock = AuthoredClock(2., 0., .5, 0.)
        previous = 0.
        for tick in range(1, 3001):
            clock.advance(tick/1000, 2.)
            if 1.99 < clock.seconds < 2.:
                self.assertAlmostEqual((clock.seconds-previous)/.001, 1., delta=.001)
            previous = clock.seconds
        self.assertEqual(clock.weight, 0.)

    def test_selected_exit_rate_and_acceleration_converge_at_both_rates(self):
        for hz in (60, 120):
            for rate in (0.8, 1.3):
                clock = AuthoredClock(6., 0., .5, 0.)
                clock.stop = 4.
                previous, speeds = 0., []
                for tick in range(1, 1000):
                    carry = clock.advance(tick/hz, rate)
                    if clock.seconds == clock.stop:
                        self.assertGreaterEqual(carry, 0.)
                        break
                    speeds.append((clock.seconds-previous)*hz)
                    previous = clock.seconds
                self.assertAlmostEqual(speeds[-1], 1., delta=.002)
                self.assertAlmostEqual((speeds[-1]-speeds[-2])*hz, 0., delta=.2)
                self.assertEqual(clock.duration, 6.)
                self.assertEqual(clock.weight, 0.)


def fake_sampler(offset=0.):
    q = (np.linspace(0., .2, 61)+offset)[:, None]
    authored = AuthoredTrajectory(q, 10)
    features = v2.MotionEntryFeatures(
        ("joint",), q, authored.velocities, np.array([20.]), np.zeros((61, 3)),
        np.zeros(61), np.zeros((61, 3)), np.zeros(61), np.zeros((61, 2), bool),
        np.array([0]), np.array([1.]), 10., 6., authored)
    def sample_frame(phase, amplitude, accent, features):
        return v2.RobotMotionFrame({"joint": float(np.interp(phase*61, np.arange(61), q[:, 0]))},
                                  np.zeros(3), np.array([1., 0., 0., 0.]))
    return SimpleNamespace(entry_features=features, frames=q, sample_frame=sample_frame)


class ImmediateExecutor:
    calls = 0
    def submit(self, fn, *args):
        self.calls += 1
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


def fake_controller():
    return SimpleNamespace(update=lambda now: (0., 1., 0., 0.), speed_multiplier=1.,
                           phase_correction_remaining=0., phase=0.)


class LifecycleTests(unittest.TestCase):
    def test_expanded_retry_is_bounded_and_uses_new_candidates(self):
        engine = self.engine()
        engine.status = 'failed'
        engine.frozen = (('old', (1., 1.)),)
        match = SimpleNamespace(accepted=True, motions=[
            SimpleNamespace(motion_id='new', final_score=.95, music_score=.95)])
        with patch.object(v2, 'musical_shortlist', return_value=('new',)) as shortlist, \
                patch.object(v2, 'prepare_state_bridge', side_effect=InfeasibleBridge('infeasible')):
            engine.prepare(match)
            engine.prepare(match)
            engine.prepare(match)
        self.assertEqual(engine.loader.score_executor.calls, 1)
        self.assertTrue(engine.expansion_attempted)
        self.assertEqual(engine.frozen[0][0], 'new')
        self.assertEqual(engine.status, 'failed')
        self.assertTrue(shortlist.call_args.kwargs['expanded'])

    def engine(self):
        sampler = fake_sampler()
        executor = ImmediateExecutor()
        loader = SimpleNamespace(score_executor=executor, failed_motions={},
            preload=lambda ids: None, prepare_pool=lambda ids: None, take_ready=lambda mid: fake_sampler(.1))
        args = SimpleNamespace(no_mic=True, audio_input=None, transition_boundary_seconds=.5,
            transition_exit_window_seconds=0.,
            motion_timing="authored", speed_max=1., transition_min_seconds=.35,
            transition_backend="hermite", entry_max_phase=.2)
        return v2.AuthoredPlayback("a", sampler, fake_controller(), loader,
            SimpleNamespace(motions={"a": object()}), args,
            {"joint": v2.JointDynamicsLimits(10., 50.)}, {"joint": (-10., 10.)},
            {"joint": math.inf}, 0.)

    def test_prepare_once_and_ignore_later_shortlists(self):
        engine = self.engine()
        engine.prepare(None)
        engine.prepare(None)
        first = engine.plan
        for _ in range(4):
            engine.prepare(SimpleNamespace(motions=[]))
        self.assertIs(engine.plan, first)
        self.assertEqual(engine.loader.score_executor.calls, 1)

    def test_startup_freezes_first_retrieval(self):
        engine = self.engine()
        engine.frozen = None
        engine.prepare(None)
        self.assertEqual(engine.loader.score_executor.calls, 0)
        match = SimpleNamespace(motions=[])
        with patch.object(v2, "musical_shortlist", return_value=("b",)) as shortlist:
            engine.prepare(match)
            engine.prepare(match)
            engine.prepare(SimpleNamespace(motions=[]))
        self.assertEqual(shortlist.call_count, 1)
        self.assertEqual(engine.plan.motion_id, "b")
        self.assertFalse(engine.plan.replay)

    def test_infeasible_candidates_then_replay(self):
        engine = self.engine()
        plan = v2.prepare_state_bridge(0, "a", engine.sampler,
            (("b", fake_sampler(20.), (1., 1.)),), engine.limits, engine.ranges, engine.jerks, engine.args)
        self.assertTrue(plan.replay)

    def test_scoring_configuration_reaches_candidates_and_replay(self):
        engine = self.engine()
        engine.args.entry_pose_tolerance_fraction = .123
        with patch.object(v2, "select_motion_entry", wraps=v2.select_motion_entry) as select:
            v2.prepare_state_bridge(0, "a", engine.sampler,
                (("b", fake_sampler(.1), (1., 1.)),), engine.limits,
                engine.ranges, engine.jerks, engine.args)
            v2.prepare_state_bridge(0, "a", engine.sampler, (), engine.limits,
                engine.ranges, engine.jerks, engine.args)
        self.assertEqual(select.call_count, 2)
        for call in select.call_args_list:
            self.assertEqual(call.kwargs["scoring_config"].pose_tolerance_fraction, .123)

    def test_retry_next_ranked_entry(self):
        engine = self.engine()
        original = v2.make_bridge
        count = 0
        def reject_first(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                raise InfeasibleBridge("first candidate overshoots")
            return original(*args, **kwargs)
        with patch.object(v2, "make_bridge", side_effect=reject_first):
            plan = v2.prepare_state_bridge(0, "a", engine.sampler,
                (("b", fake_sampler(.1), (1., 1.)),), engine.limits, engine.ranges, engine.jerks, engine.args)
        self.assertEqual(count, 2)
        self.assertFalse(plan.replay)

    def test_earlier_exit_avoids_invalid_terminal(self):
        engine = self.engine()
        engine.args.transition_exit_window_seconds = 2.
        engine.sampler.entry_features.authored.velocities[-1] = 100.
        plan = v2.prepare_state_bridge(0, "a", engine.sampler,
            (("b", fake_sampler(.1), (1., 1.)),), engine.limits,
            engine.ranges, engine.jerks, engine.args)
        self.assertGreaterEqual(plan.exit_seconds, 4.)
        self.assertLess(plan.exit_seconds, 6.)
        self.assertFalse(plan.replay)
        self.assertLess(max(plan.residuals), 1e-8)
        engine.args.transition_exit_window_seconds = 0.
        with self.assertRaises(InfeasibleBridge):
            v2.prepare_state_bridge(0, "a", engine.sampler, (), engine.limits,
                engine.ranges, engine.jerks, engine.args)

    def test_pair_ranking_matches_exhaustive_existing_scores(self):
        engine = self.engine()
        engine.args.transition_exit_window_seconds = 2.
        target = fake_sampler(.1)
        source_features = v2.authored_features(engine.sampler)
        expected = []
        for index in range(40, 61):
            scores = v2.select_motion_entry(source_features, index/60,
                v2.authored_features(target), engine.limits, beat_period=.5, beats_per_bar=4,
                minimum_remaining_bars=0, speed_max=1., transition_minimum=.35,
                transition_maximum=10., entry_max_phase=.2, return_all=True)
            expected.extend((score.total, -index, score.frame_index) for score in scores)
        plan = v2.prepare_state_bridge(0, 'a', engine.sampler,
            (('b', target, (1., 1.)),), engine.limits, engine.ranges, engine.jerks, engine.args)
        self.assertEqual((plan.score.total, -plan.exit_frame_index, plan.score.frame_index), min(expected))

    def test_selected_exit_preserves_clip_timeline_and_carry(self):
        engine = self.engine()
        engine.args.transition_exit_window_seconds = 2.
        engine.prepare(None)
        engine.prepare(None)
        exit_time = engine.plan.exit_seconds
        duration = engine.plan.trajectory.duration
        self.assertLess(exit_time, 6.)
        self.assertEqual(engine.clock.duration, 6.)
        engine.sample(exit_time+.003, v2.base.FeatureState(), None, None)
        self.assertEqual(engine.event, 'switch_start')
        self.assertAlmostEqual(engine.bridge_start, exit_time)
        self.assertEqual(engine.clock.weight, 0.)
        with patch.object(v2, 'make_controller', return_value=fake_controller()):
            engine.sample(exit_time+duration+.004, v2.base.FeatureState(), None, None)
        self.assertEqual(engine.event, 'switch_complete')
        self.assertAlmostEqual(engine.clock.seconds, .004, places=8)

    def test_late_exit_plan_is_discarded_then_replanned(self):
        engine = self.engine()
        engine.args.transition_exit_window_seconds = 2.
        engine.prepare(None)
        engine.clock.seconds = 5.8
        engine.prepare(None)
        self.assertIsNone(engine.plan)
        self.assertEqual(engine.status, 'late_plan_discarded')
        engine.prepare(None)
        engine.prepare(None)
        self.assertEqual(engine.plan.exit_seconds, 6.)

    def test_late_poll_accounts_for_time_since_last_sample(self):
        engine = self.engine()
        engine.args.transition_exit_window_seconds = 2.
        engine.prepare(None)
        engine.prepare(None, now=5.8)
        self.assertIsNone(engine.plan)
        self.assertEqual(engine.status, 'late_plan_discarded')
        self.assertEqual(engine.clock.seconds, 0.)

    def test_cancelled_search_and_expired_solver(self):
        engine = self.engine()
        engine.search_cancel.set()
        with self.assertRaisesRegex(InfeasibleBridge, 'cancelled'):
            v2.prepare_state_bridge(0, 'a', engine.sampler, (), engine.limits,
                engine.ranges, engine.jerks, engine.args, cancel=engine.search_cancel)
        with self.assertRaisesRegex(InfeasibleBridge, 'deadline'):
            bridge('hermite', deadline=0.)

    def test_pose_effects_fade_to_authored_terminal(self):
        engine = self.engine()
        features = v2.base.FeatureState()
        engine.prepare(None)
        engine.prepare(None)
        modulator = SimpleNamespace(modulate=lambda pose, features, **kwargs:
                                    {n: q+.1 for n, q in pose.items()})
        middle = engine.sample(3., features, modulator, None)
        self.assertAlmostEqual(middle.joint_positions['joint'], .2)
        terminal = engine.sample(6., features, modulator, None)
        self.assertAlmostEqual(terminal.joint_positions['joint'], .2)

    def test_boundary_suppresses_phase_corrections_only_near_joins(self):
        engine = self.engine()
        features = v2.base.FeatureState()
        engine.controller.phase_correction_remaining = .1
        engine.sample(.01, features, None, None)
        self.assertEqual(engine.controller.phase_correction_remaining, 0.)
        engine.sample(3., features, None, None)
        engine.controller.phase_correction_remaining = .1
        engine.sample(3.01, features, None, None)
        self.assertEqual(engine.controller.phase_correction_remaining, .1)

    def test_stale_generation_ignored(self):
        engine = self.engine()
        engine.prepare(None)
        engine.generation += 1
        engine.prepare(None)
        self.assertIsNone(engine.plan)

    def test_late_preparation_holds_without_claiming_continuity(self):
        engine = self.engine()
        engine.sample(6.01, v2.base.FeatureState(), None, None)
        self.assertEqual(engine.status, "terminal_hold")
        engine.prepare(None)
        self.assertEqual(engine.loader.score_executor.calls, 0)
        self.assertIn("did not finish", engine.failure)

    def test_failure_has_reason(self):
        engine = self.engine()
        with patch.object(v2, "prepare_state_bridge", side_effect=InfeasibleBridge("impossible")):
            engine.prepare(None)
            engine.prepare(None)
        self.assertEqual(engine.status, "failed")
        self.assertEqual(engine.failure, "impossible")

    def test_continuous_handoff_carries_time_into_next_clip(self):
        engine = self.engine()
        engine.prepare(None)
        engine.prepare(None)
        duration = engine.plan.trajectory.duration
        features = v2.base.FeatureState()
        frame = engine.sample(6.003, features, None, None)
        self.assertEqual(engine.event, "switch_start")
        self.assertGreater(engine.blend, 0.)
        self.assertAlmostEqual(engine.bridge_start, 6.)
        with patch.object(v2, "make_controller", return_value=fake_controller()):
            engine.sample(6.+duration+.004, features, None, None)
        self.assertEqual(engine.event, "switch_complete")
        self.assertAlmostEqual(engine.clock.seconds, .004, places=8)
        self.assertEqual(engine.generation, 1)
        self.assertLess(max(engine.last_residuals), 1e-8)


@unittest.skipUnless(importlib.util.find_spec("ruckig"), "Optional Ruckig dependency not installed")
class RuckigTests(unittest.TestCase):
    def test_selected_exit_uses_same_ranking_with_ruckig(self):
        engine = LifecycleTests().engine()
        engine.args.transition_backend = 'ruckig'
        engine.args.transition_exit_window_seconds = 2.
        engine.jerks['joint'] = 500.
        plan = v2.prepare_state_bridge(0, 'a', engine.sampler,
            (('b', fake_sampler(.1), (1., 1.)),), engine.limits,
            engine.ranges, engine.jerks, engine.args)
        self.assertGreaterEqual(plan.exit_seconds, 4.)
        self.assertLessEqual(plan.exit_seconds, 6.)
        self.assertFalse(plan.replay)
        self.assertLess(max(plan.residuals), 1e-6)

    def test_real_solver_matches_moving_states(self):
        curve = bridge("ruckig")
        for t, expected in ((0., state(0., .2, .1)), (curve.duration, state(1., -.1, -.2))):
            actual = curve.at_time(t)
            for field in ("position", "velocity", "acceleration"):
                np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=1e-8)

    def test_real_solver_checks_positions(self):
        with self.assertRaises(InfeasibleBridge):
            bridge("ruckig", start=state(1., 1.), end=state(0.),
                   lower=np.array([-1.]), upper=np.array([1.]))


if __name__ == "__main__":
    unittest.main()
