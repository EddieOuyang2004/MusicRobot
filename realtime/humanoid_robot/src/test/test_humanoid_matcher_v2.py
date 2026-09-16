"""Deterministic checks for v2 selection, audio history, and smooth bridges.

Run: python -m unittest discover -s realtime/humanoid_robot/src/test -p test_humanoid_matcher_v2.py
"""
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import unittest
from unittest.mock import patch, Mock
from contextlib import redirect_stderr
import io

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import realtime_music_humanoid_matcher_v2 as v2


def motion_features(positions, duration=10.0):
    q = np.asarray(positions, dtype=float).reshape(-1, 1)
    n = len(q)
    return v2.MotionEntryFeatures(
        joint_names=("joint",), joint_positions=q, joint_velocities=np.zeros_like(q),
        joint_ranges=np.array([2.0]), root_positions=np.zeros((n, 3)),
        root_tilt=np.zeros(n), root_linear_velocities=np.zeros((n, 3)),
        root_angular_speeds=np.zeros(n), foot_contacts=np.zeros((n, 2), dtype=bool),
        candidate_indices=np.array([0]), candidate_salience=np.array([1.0]),
        fps=(n - 1) / duration, duration=duration,
    )


def args():
    return SimpleNamespace(
        speed_max=1.0, motion_timing="beat-sync", transition_min_seconds=0.35,
        entry_max_phase=0.2, switch_diversity_score_drop=0.05,
        switch_diversity_music_score_drop=0.08, switch_diversity_top_k=5,
    )


class EntryTests(unittest.TestCase):
    def test_penalty_curve_and_unit_normalization(self):
        penalty = v2.entry_difference_penalty
        np.testing.assert_allclose(penalty([0, .2, .5, 1, 2], 1), [0, .04, .25, 1, 8])
        self.assertTrue(np.all(np.diff(penalty(np.linspace(0, 3, 301), 1)) > 0))
        self.assertAlmostEqual(float(penalty(1 - 1e-9, 1)), 1, places=8)
        self.assertAlmostEqual(float(penalty(1 + 1e-9, 1)), 1, places=8)
        np.testing.assert_allclose(penalty([.05, .15, 1.0], [.025, .075, .5]), [8, 8, 8])
        self.assertEqual(float(penalty(2, 1, 0)), 4)

    def test_config_validation_and_immutability(self):
        config = v2.EntryScoringConfig()
        with self.assertRaises(FrozenInstanceError):
            config.pose_tolerance_fraction = .2
        for field in fields(config):
            invalid = [-1, math.inf, math.nan]
            if field.name != "excess_penalty_weight":
                invalid.append(0)
            for value in invalid:
                with self.subTest(field=field.name, value=value), self.assertRaises(ValueError):
                    replace(config, **{field.name: value})

    def test_contact_and_root_penalties(self):
        target = motion_features([0] * 11, duration=20)
        target.foot_contacts[1, 0] = True
        target.foot_contacts[2] = True
        target.root_positions[3, 2] = .05
        scores = {s.frame_index: s for s in self.score(target, return_all=True, entry_max_phase=.4)}
        self.assertEqual([scores[i].contact for i in range(3)], [0, 1, 8])
        self.assertAlmostEqual(scores[3].root, 8 / 4)

    def test_joint_penalties_precede_averaging(self):
        source = replace(self.source, joint_names=("joint", "other"),
                         joint_positions=np.zeros((2, 2)), joint_velocities=np.zeros((2, 2)),
                         joint_ranges=np.array([2., 2.]))
        target = replace(source, joint_positions=np.array([[.2, 0.], [0., 0.]]))
        score = v2.select_motion_entry(
            source, 1, target, {"joint": self.limits["joint"], "other": self.limits["joint"]},
            beat_period=.5, beats_per_bar=4, minimum_remaining_bars=0, speed_max=1,
            transition_minimum=.35, transition_maximum=1.2,
        )
        self.assertAlmostEqual(score.pose, 8 / 2)

    def test_large_mismatch_loses_to_balanced_entry(self):
        target = motion_features([.2, .11] + [2.] * 9, duration=20)
        target.joint_velocities[1] = .22
        # Old linear cost prefers frame 0: .045 < .04675 (before / .95).
        self.assertLess(.45 * .2 / 2, .45 * .11 / 2 + .2 * .22 / 2)
        self.assertEqual(self.score(target).frame_index, 1)
        # Looser pose tolerance reverses that choice, including worker selection.
        config = v2.EntryScoringConfig(pose_tolerance_fraction=.5)
        self.assertEqual(self.score(target, scoring_config=config).frame_index, 0)
        options = args()
        options.entry_pose_tolerance_fraction = .5
        sampler = SimpleNamespace(entry_features=target)
        choice = v2.score_ready_candidates(self.source, (("target", sampler),), self.limits, options)
        self.assertEqual(choice[2].frame_index, 0)
        replay = v2.replay_choice("target", sampler, self.source, self.limits, options)
        self.assertAlmostEqual(replay[2].pose, .04)

    def setUp(self):
        self.source = motion_features([10, 0])
        self.limits = {"joint": v2.JointDynamicsLimits(2.0, 4.0)}

    def score(self, target, **kw):
        return v2.select_motion_entry(
            self.source, 1.0, target, self.limits, beat_period=0.5,
            beats_per_bar=4, minimum_remaining_bars=0, speed_max=1.0,
            transition_minimum=0.35, transition_maximum=1.2, **kw,
        )

    def test_searches_non_keypoint_frames_using_final_source(self):
        score = self.score(motion_features([5, 4, 3, 0, 2, 1, 1, 1, 1, 1, 1], duration=20.0), entry_max_phase=.4)
        self.assertEqual(score.frame_index, 3)
        self.assertEqual(score.total, 0.0)

    def test_entry_window_excludes_late_minimum(self):
        score = self.score(motion_features([4, 3, 2, 1, 2, 2, 2, 0, 0, 0, 0]))
        self.assertEqual(score.frame_index, 2)
        self.assertGreaterEqual(score.remaining_seconds, 8.0)

    def test_exact_twenty_percent_boundary_and_tie(self):
        score = self.score(motion_features([5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(score.frame_index, 2)
        self.assertEqual(score.remaining_seconds, 8.0)
        self.assertEqual(self.score(motion_features([0] * 11)).frame_index, 0)

    def test_short_clip_is_eligible_and_replay_starts_at_zero(self):
        short = motion_features([2, 0], duration=2.0)
        self.assertEqual(self.score(short).frame_index, 0)
        self.assertIsNone(self.score(short, eligible_frame_mask=np.zeros(2, dtype=bool)))
        replay = v2.replay_choice("same", SimpleNamespace(entry_features=short),
                                  self.source, self.limits, args())
        self.assertEqual(replay[0], "same")
        self.assertEqual(replay[2].frame_index, 0)
        self.assertTrue(replay[3])

    def test_house_clip_frame_boundaries(self):
        for frames, last_entry in ((426, 85), (480, 95)):
            target = motion_features([0] * frames, duration=(frames-1)/60)
            scores = self.score(target, return_all=True)
            self.assertEqual(sorted(s.frame_index for s in scores), list(range(last_entry+1)))
            self.assertTrue(all(s.remaining_seconds >= .8*target.duration-1e-9 for s in scores))

    def test_entry_window_scales_with_duration_independent_of_speed(self):
        target = motion_features([5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        for duration, speed in ((2., .8), (7.083333333, 1.), (10., 1.3), (47.95, 1.6)):
            target = replace(target, duration=duration, fps=10/duration)
            with self.subTest(duration=duration, speed=speed):
                score = v2.select_motion_entry(
                    self.source, 1.0, target, self.limits, beat_period=.5,
                    beats_per_bar=4, minimum_remaining_bars=0, speed_max=speed,
                    transition_minimum=.35, transition_maximum=1.2,
                    entry_max_phase=0.2,
                )
                self.assertEqual(score.frame_index, 2)
                self.assertAlmostEqual(score.remaining_seconds, .8*duration)
                self.assertEqual([s.frame_index for s in sorted(self.score(target, return_all=True), key=lambda s: s.frame_index)], [0, 1, 2])

    def test_velocity_and_contact_break_pose_tie(self):
        target = motion_features([0] * 11, duration=20.0)
        target.joint_velocities[:3] = 5
        target.foot_contacts[3] = True
        self.assertEqual(self.score(target, entry_max_phase=.4).frame_index, 4)

    def test_matching_fast_frames_keep_zero_velocity_penalty(self):
        target = motion_features([0] * 11, duration=20.0)
        self.source.joint_velocities[:] = 1.5
        target.joint_velocities[:] = 1.5
        self.assertEqual(self.score(target).velocity, 0.)

    def test_matches_independent_scalar_reference(self):
        rng = np.random.default_rng(8)
        target = motion_features(rng.normal(size=101), duration=20)
        target.joint_velocities[:] = rng.normal(size=(101, 1))
        target.foot_contacts[:] = rng.random((101, 2)) > 0.5
        target.root_positions[:, 2] = rng.normal(size=101) * 0.1
        costs = []
        for i in range(101):
            if i / 100 > .2 + 1e-9:
                continue
            def penalty(r):
                return r * r + 4 * max(r - 1, 0) ** 2
            pose = penalty(abs(target.joint_positions[i, 0]) / .1)
            velocity = penalty(abs(target.joint_velocities[i, 0]) / .2)
            contact = penalty(sum(bool(x) for x in target.foot_contacts[i]))
            root = penalty(abs(target.root_positions[i, 2]) / .025) / 4
            costs.append(((0.45 * pose + 0.2 * velocity + 0.2 * contact + 0.1 * root) / 0.95, i))
        expected_cost, expected_index = min(costs)
        actual = self.score(target)
        self.assertEqual(actual.frame_index, expected_index)
        self.assertAlmostEqual(actual.total, expected_cost)

    def test_cross_clip_effort_then_music_rank(self):
        bad = SimpleNamespace(entry_features=motion_features([4] * 11))
        good = SimpleNamespace(entry_features=motion_features([0] * 11))
        chosen = v2.score_ready_candidates(self.source, (("music_best", bad), ("smooth", good)), self.limits, args())
        self.assertEqual(chosen[0], "smooth")
        chosen = v2.score_ready_candidates(self.source, (("music_best", good), ("smooth", good)), self.limits, args())
        self.assertEqual(chosen[0], "music_best")
        later = SimpleNamespace(entry_features=motion_features([1, 1, 0] + [1] * 8))
        chosen = v2.score_ready_candidates(self.source, (("a", later, (1., 1.)), ("z", good, (1., 1.))), self.limits, args())
        self.assertEqual(chosen[0], "z")  # Equal relevance: earlier frame precedes motion ID.
        self.assertIsNone(v2.score_ready_candidates(self.source, (), self.limits, args()))

    def test_terminal_recheck_uses_command_without_mutating_cache(self):
        actual = v2.terminal_features(self.source, v2.RobotMotionFrame({"joint": 3.0}))
        self.assertEqual(actual.joint_positions[0, 0], 3.0)
        self.assertEqual(self.source.joint_positions[-1, 0], 0.0)


class TransitionTests(unittest.TestCase):
    def test_endpoints_midpoint_and_quaternion(self):
        a = v2.RobotMotionFrame({"j": 0.0}, np.zeros(3), np.array([1., 0., 0., 0.]))
        b = v2.RobotMotionFrame({"j": 2.0}, np.ones(3), np.array([0., 0., 0., 1.]))
        self.assertEqual(v2.quintic_blend(a, b, -1).joint_positions, a.joint_positions)
        self.assertEqual(v2.quintic_blend(a, b, 2).joint_positions, b.joint_positions)
        mid = v2.quintic_blend(a, b, 0.5)
        self.assertEqual(mid.joint_positions["j"], 1.0)
        np.testing.assert_allclose(mid.root_position, 0.5)
        np.testing.assert_allclose(mid.root_quaternion_wxyz, [math.sqrt(0.5), 0, 0, math.sqrt(0.5)])

    def test_duration_respects_quintic_dynamics_without_maximum_clamp(self):
        duration = v2.compute_transition_duration(np.array([0.]), np.array([4.]),
            np.array([0.5]), np.array([0.75]), beat_period=0.5, minimum=0.35, maximum=1.2)
        self.assertGreater(duration, 1.2)
        u = np.linspace(0, 1, 10001)
        speeds = 4 / duration * 30 * u ** 2 * (1 - u) ** 2
        accelerations = 4 / duration ** 2 * (60*u - 180*u*u + 120*u*u*u)
        self.assertLessEqual(max(speeds), 0.5 + 1e-9)
        self.assertLessEqual(max(abs(accelerations)), 0.75 + 1e-9)

    def test_one_shot_does_not_wrap_before_bridge(self):
        tracker = v2.OneShotPhaseTracker(0.3)
        self.assertFalse(tracker.clamp(0.9)[1])
        self.assertTrue(tracker.clamp(0.01)[1])
        self.assertGreater(tracker.clamp(0.4)[0], 0.99999)


class HistoryAndShortlistTests(unittest.TestCase):
    def test_worker_preserves_source_sample_rate(self):
        result = v2.MatchResult((), (), 120)
        extractor = SimpleNamespace(describe=Mock(return_value=object()), last_timing_ms={})
        matcher = SimpleNamespace(match=Mock(return_value=result), last_timing_ms={})
        worker = v2.RetrievalWorker(extractor, matcher, 5, 10)
        try:
            self.assertTrue(worker.submit(np.zeros(48000), 48000))
            worker.future.result(timeout=5)
            self.assertIs(worker.poll(), result)
            self.assertEqual(extractor.describe.call_args.kwargs['source_sample_rate'], 48000)
        finally:
            worker.close()
        with patch.object(v2, '_RETRIEVAL_PROCESS_EXTRACTOR', extractor), patch.object(v2, '_RETRIEVAL_PROCESS_MATCHER', matcher):
            self.assertIs(v2._run_retrieval_process(np.zeros(48000), 5, 10, 48000).result, result)
            self.assertEqual(extractor.describe.call_args.kwargs['source_sample_rate'], 48000)

    def test_cli_defaults_and_validation(self):
        with patch.object(sys, 'argv', ['matcher_v2', '--headless', '--no-mic']):
            parsed = v2.parse_args()
        self.assertEqual(parsed.history_max_seconds, 30)
        self.assertEqual(parsed.analysis_min_seconds, 2)
        self.assertEqual(parsed.entry_max_phase, .2)
        for value in ('0', '0.2', '1'):
            with patch.object(sys, 'argv', ['matcher_v2', '--entry-max-phase', value]):
                self.assertEqual(v2.parse_args().entry_max_phase, float(value))
        for value in ('-0.1', '1.1', 'nan', 'inf'):
            with patch.object(sys, 'argv', ['matcher_v2', '--entry-max-phase', value]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    v2.parse_args()
        self.assertEqual(parsed.transition_exit_window_seconds, 2.)
        for value in ('0', '1.5'):
            with patch.object(sys, 'argv', ['matcher_v2', '--transition-exit-window-seconds', value]):
                self.assertEqual(v2.parse_args().transition_exit_window_seconds, float(value))
        for value in ('-1', 'nan', 'inf'):
            with patch.object(sys, 'argv', ['matcher_v2', '--transition-exit-window-seconds', value]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    v2.parse_args()
        self.assertEqual(parsed.speed_max, 1.3)
        self.assertEqual(parsed.control_rate_hz, 60.)
        for option in (['--control-rate-hz', '120'], ['--control-rate-hz=90']):
            with patch.object(sys, 'argv', ['matcher_v2', *option]):
                self.assertEqual(v2.parse_args().control_rate_hz, 120. if len(option) == 2 else 90.)
        with patch.object(sys, 'argv', ['dancer', '--headless', '--no-mic']):
            self.assertEqual(v2.base.parse_args().control_rate_hz, 120.)
        for option in (['--speed-max', '1.1'], ['--speed-max=1.1']):
            with patch.object(sys, 'argv', ['matcher_v2', *option]):
                self.assertEqual(v2.parse_args().speed_max, 1.1)
        self.assertGreaterEqual(parsed.switch_ready_pool_size, 5)
        self.assertEqual(parsed.match_top_motions, 20)
        self.assertEqual(parsed.switch_diversity_top_k, 10)
        self.assertEqual(v2.EntryScoringConfig.from_args(parsed), v2.EntryScoringConfig())
        for field in fields(v2.EntryScoringConfig):
            flag = '--entry-' + field.name.replace('_', '-')
            with patch.object(sys, 'argv', ['matcher_v2', flag, '0.3']):
                self.assertEqual(getattr(v2.EntryScoringConfig.from_args(v2.parse_args()), field.name), .3)
            invalid = ['-1', 'nan', 'inf']
            if field.name != 'excess_penalty_weight':
                invalid.append('0')
            for value in invalid:
                with self.subTest(flag=flag, value=value), patch.object(sys, 'argv', ['matcher_v2', flag, value]), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        v2.parse_args()
        for option in (['--match-window-seconds', '6'], ['--history-max-seconds', '31'],
                       ['--history-max-seconds', '1'], ['--shortlist-size', '0']):
            with self.subTest(option=option), patch.object(sys, 'argv', ['matcher_v2'] + option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    v2.parse_args()

    def test_history_grows_caps_and_does_not_change_beat_buffer(self):
        forwarded = []
        analyzer = SimpleNamespace(sample_rate=10, plp_history_sec=6,
                                   _append_audio_chunk=lambda t, a: forwarded.append(len(a)))
        history = v2.RetrievalAudioHistory(analyzer, 30, 2)
        analyzer._append_audio_chunk(0, np.arange(10))
        self.assertIsNone(history.recent_audio())
        analyzer._append_audio_chunk(1, np.arange(10, 25))
        np.testing.assert_array_equal(history.recent_audio(), np.arange(25))
        analyzer._append_audio_chunk(2.5, np.arange(25, 410))
        np.testing.assert_array_equal(history.recent_audio(), np.arange(110, 410))
        self.assertEqual(analyzer.plp_history_sec, 6)
        self.assertEqual(forwarded, [10, 15, 385])

    def test_file_history_is_causal_and_matches_live_buffer(self):
        source = object.__new__(v2.MatcherFileMicrophoneSource)
        source.sample_rate, source.startup_samples = 10, 10
        source.audio = np.arange(500, dtype=np.float32)
        source.window_seconds, source.minimum_seconds = 30, 2
        source.cursor = 25
        self.assertIsNone(source.recent_audio())
        source.cursor = 35
        np.testing.assert_array_equal(source.recent_audio(), np.arange(25))
        source.cursor = 410
        np.testing.assert_array_equal(source.recent_audio(), np.arange(100, 400))

    def test_shortlist_rejection_margins_and_current_exclusion(self):
        def match(mid, final, music):
            return SimpleNamespace(motion_id=mid, final_score=final, music_score=music)
        result = v2.MatchResult((), (match("current", 1, 1), match("yes", .97, .95),
                                    match("bad_music", .99, .8), match("bad_total", .8, 1)), 120)
        self.assertEqual(v2.musical_shortlist(result, "current", args()), ("yes",))
        self.assertEqual(v2.musical_shortlist(replace(result, accepted=False), "current", args()), ())
        self.assertEqual(v2.musical_shortlist(None, "current", args()), ())
        expanded = replace(result, motions=result.motions + (
            match('expanded', .92, .87), match('too_far', .89, .99)))
        self.assertEqual(v2.musical_shortlist(expanded, 'current', args()), ('yes',))
        self.assertEqual(v2.musical_shortlist(expanded, 'current', args(), expanded=True),
                         ('yes', 'expanded'))
        self.assertEqual(v2.musical_shortlist(expanded, 'current', args(), expanded=True,
                                             exclude={'yes'}), ('expanded',))


if __name__ == "__main__":
    unittest.main()
