from __future__ import annotations

import csv
from concurrent.futures import Future
import json
import shutil
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "realtime/humanoid_robot/src"
sys.path[:0] = [str(SRC), str(SRC / "test")]
import run_thesis_supplement as supplement
import run_humanoid_matcher_experiments as runner
import reanalyse_thesis_experiments as reanalysis
import realtime_music_humanoid_matcher as matcher
from build_aistpp_fact_feature_bundle import interpolate_trace_row
from humanoid_matcher_experiment_metrics import timed_derivatives, edge_pfc_g1_adapted


class SupplementTests(unittest.TestCase):
    def test_matrix_exact_and_seed_complete(self):
        protocol = supplement.read_json(runner.DEFAULT_PROTOCOL)
        # Source discovery is read-only; no model inference or experiment starts.
        matrix = supplement.supplement_matrix(None, protocol)
        self.assertEqual({k: len(v) for k, v in matrix.items()}, supplement.EXPECTED)
        for name, rows in matrix.items():
            self.assertEqual(len(rows), len({(r['condition'], r['seed'], r['source_id']) for r in rows}))
            self.assertTrue(all('--experiment-causal-file-input' in r['condition_arguments'] for r in rows))
        self.assertEqual({0, 1, 2, 3, 4}, {r['seed'] for r in matrix['long']})
        self.assertEqual({606.0}, {r['duration'] for r in matrix['long']})

    def test_pilot_matrix_is_bounded_and_balanced(self) -> None:
        protocol = supplement.read_json(runner.DEFAULT_PROTOCOL)
        matrix = supplement.supplement_matrix(None, protocol, pilot=True)
        self.assertEqual(
            {key: len(value) for key, value in matrix.items()},
            supplement.PILOT_EXPECTED,
        )
        self.assertEqual(
            {"full", "authored_timing"},
            {item["condition"] for item in matrix["short"]},
        )
        self.assertEqual({0, 1, 2}, {item["seed"] for item in matrix["long"]})
        self.assertEqual({306.0}, {item["duration"] for item in matrix["long"]})

    def test_windows_power_preflight_rejects_balanced_plan(self) -> None:
        balanced = "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e"
        high = f"Power Scheme GUID: {supplement.WINDOWS_HIGH_PERFORMANCE_GUID}"
        with mock.patch.object(supplement.os, "name", "nt"), mock.patch.object(
            runner, "command_output", return_value=balanced
        ):
            with self.assertRaisesRegex(RuntimeError, "High performance"):
                supplement.require_experiment_power_scheme()
        with mock.patch.object(supplement.os, "name", "nt"), mock.patch.object(
            runner, "command_output", return_value=high
        ):
            self.assertEqual(high, supplement.require_experiment_power_scheme())

    def test_frozen_manifest_rejects_changes_without_overwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.json'
            first = supplement.freeze_manifest(path, {'schema': 2, 'inputs': 'a'})
            original = path.read_bytes()
            self.assertEqual(first, supplement.freeze_manifest(path, {'schema': 2, 'inputs': 'a'}))
            with self.assertRaises(ValueError):
                supplement.freeze_manifest(path, {'schema': 2, 'inputs': 'b'})
            self.assertEqual(original, path.read_bytes())

    def fake_artifacts(self, folder, *, bad_clock=False, fail_safety=False):
        folder = Path(folder)
        trace, timing, pose, log = [folder / name for name in ('trace.csv', 'timing.json', 'pose.npz', 'run.log')]
        times = np.array([0., .01, .03, .06, .1])
        if bad_clock:
            times[2] = times[1]
        safety = np.zeros((5, 4), dtype=np.int32)
        if fail_safety:
            safety[2] = [1, 1, 1, 1]
        trace.write_text('audio_time_seconds,event,current_motion_id\n0,start,a\n.1,end,a\n', encoding='utf-8')
        timing.write_text(json.dumps({'iterations': 5, 'beat_input_mode': 'online_causal',
                                     'onnxruntime_providers': ['CPUExecutionProvider'],
                                      'final_residual_clearance_violations': int(safety[:, 2].sum()),
                                      'final_joint_limit_violations': int(safety[:, 3].sum()),
                                      'control_rate_hz': 120,
                                      'deadline_miss_ratio': .5}), encoding='utf-8')
        log.write_text('Synthetic test artifact, not a real experiment.\n', encoding='utf-8')
        joints = np.zeros((5, 2))
        np.savez(pose, schema_version=2, beat_input_mode='online_causal',
                 time_seconds=np.array([0, 0, .02, .04, .08]), wall_time_seconds=times,
                 frame_index=np.arange(5), audio_sample_rate_hz=100, audio_received_sample_index=[0, 0, 2, 4, 8],
                 final_qpos=np.zeros((5, 9)), joint_positions=joints, joint_qpos_indices=[7, 8],
                 model_nq=9, model_xml_sha256='a' * 64, safety_counts=safety,
                 beat_events=np.array([[.01, .04, .02, 1.]]),
                 body_positions=np.zeros((5, 1, 3)), control_rate_hz=120,
                 center_of_mass=np.zeros((5, 3)), left_foot_position=np.zeros((5, 3)),
                 right_foot_position=np.zeros((5, 3)), left_foot_support_height=np.zeros(5),
                 right_foot_support_height=np.zeros(5), motion_ids=['a'] * 5,
                 accepted_causal_beat_times_seconds=np.array([.04]))
        return trace, timing, pose, log

    def test_validity_does_not_gate_performance_or_safety(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fake_artifacts(folder, fail_safety=True)
            valid, errors = supplement.validate_causal_artifacts(*paths)
            self.assertTrue(valid, errors)

    def test_corrupt_clock_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fake_artifacts(folder, bad_clock=True)
            valid, errors = supplement.validate_causal_artifacts(*paths)
            self.assertFalse(valid)
            self.assertTrue(any('strictly increasing' in e for e in errors))

    def test_nonfinite_pose_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fake_artifacts(folder)
            with np.load(paths[2]) as archive:
                data = dict(archive)
            data['final_qpos'][0, 0] = np.nan
            np.savez(paths[2], **data)
            valid, _ = supplement.validate_causal_artifacts(*paths)
            self.assertFalse(valid)

    def test_actual_time_derivatives(self):
        times = np.array([0, .01, .04, .08, .11])
        velocity, acceleration, jerk = timed_derivatives((2 * times)[:, None], times)
        np.testing.assert_allclose(velocity, 2)
        np.testing.assert_allclose(acceleration, 0, atol=1e-10)
        np.testing.assert_allclose(jerk, 0, atol=1e-8)
        with self.assertRaises(ValueError):
            timed_derivatives(np.zeros((4, 1)), np.zeros(4))

    def test_resume_and_corrupt_artifact_retry_preserve_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixtures = root / 'fixtures'
            fixtures.mkdir()
            artifacts = self.fake_artifacts(fixtures)
            matrix = [dict(condition='full', seed=0, source_id='synthetic', audio=root / 'unused.wav',
                           condition_arguments=['--experiment-causal-file-input'], cohort_identity='frozen')]
            def fake_process(command, **kwargs):
                for flag, source in zip(('--trace-csv', '--timing-report', '--experiment-pose-npz'), artifacts[:3]):
                    shutil.copy2(source, command[command.index(flag) + 1])
                kwargs['stdout'].write('Synthetic subprocess stub only.\n')
                return SimpleNamespace(returncode=0)
            kwargs = dict(max_seconds=.1, dry_run=False, resume=True, suite='supplement_test', max_attempts=2,
                          artifact_validator=supplement.validate_causal_artifacts, require_completed_status=True,
                          stop_on_failure=True)
            with mock.patch.object(runner.subprocess, 'run', side_effect=fake_process) as process:
                records, failures = runner.execute_runs(matrix, root / 'runs', root / 'catalog', **kwargs)
                self.assertFalse(failures)
                self.assertEqual(1, process.call_count)
                runner.execute_runs(matrix, root / 'runs', root / 'catalog', **kwargs)
                self.assertEqual(1, process.call_count)
                Path(records[0]['pose']).write_bytes(b'corrupt zip')
                records, failures = runner.execute_runs(matrix, root / 'runs', root / 'catalog', **kwargs)
                self.assertFalse(failures)
                self.assertEqual(2, process.call_count)
                status = supplement.read_json(root / 'runs/run_status.json')
                self.assertTrue(status['attempt_history'])
                self.assertTrue(Path(records[0]['archived_previous_artifacts']).exists())

    def test_interruption_during_resume_retains_later_completed_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = root / 'fixture'
            fixture.mkdir()
            artifacts = self.fake_artifacts(fixture)
            matrix = [dict(condition='full', seed=seed, source_id='synthetic', audio=root / 'unused.wav',
                           condition_arguments=[], cohort_identity='frozen') for seed in (0, 1)]
            def fake_process(command, **kwargs):
                for flag, source in zip(('--trace-csv', '--timing-report', '--experiment-pose-npz'), artifacts[:3]):
                    shutil.copy2(source, command[command.index(flag) + 1])
                kwargs['stdout'].write('Synthetic subprocess stub.\n')
                return SimpleNamespace(returncode=0)
            kwargs = dict(max_seconds=.1, dry_run=False, resume=True, suite='supplement_test', max_attempts=2,
                          artifact_validator=supplement.validate_causal_artifacts, require_completed_status=True)
            with mock.patch.object(runner.subprocess, 'run', side_effect=fake_process) as process:
                runner.execute_runs(matrix, root / 'runs', root / 'catalog', **kwargs)
                self.assertEqual(2, process.call_count)
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    runner.execute_runs(matrix, root / 'runs', root / 'catalog', **kwargs,
                                        before_run=mock.Mock(side_effect=[None, RuntimeError('interruption')]))
                status = supplement.read_json(root / 'runs/run_status.json')
                self.assertEqual(2, len(status['runs']))
                runner.execute_runs(matrix, root / 'runs', root / 'catalog', **kwargs)
                self.assertEqual(2, process.call_count, 'Resume must not rerun intact later records')

    def test_reanalysis_exports_actual_safety_and_undefined_bas(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.fake_artifacts(folder)
            with np.load(paths[2]) as archive:
                data = dict(archive)
            data['wall_time_seconds'] += 6
            data['time_seconds'] += 6
            np.savez(paths[2], **data)
            paths[0].write_text('audio_time_seconds,wall_time_seconds,event,current_motion_id\n6,6,start,a\n6.1,6.1,end,a\n', encoding='utf-8')
            run = dict(zip(('trace', 'timing', 'pose', 'log'), map(str, paths)))
            run.update(condition='full', source_id='synthetic', seed=0, run_key='synthetic', duration=6.1)
            result = reanalysis.analyse_run(run, True, {})
            self.assertIsNone(result['metrics']['bas_harmonic'])
            self.assertTrue(result['safety']['actual_final_output_derivatives_recoverable'])
            self.assertEqual(0, result['safety']['final_residual_clearance_violations'])
            supplement.atomic_json(Path(folder) / 'export.json', reanalysis.clean_json(result))

    def test_phase_interpolation_wrap_and_boundary(self):
        first = dict(current_motion_id='a', transition_motion_id='', current_phase='.9', transition_blend='0')
        last = dict(first, current_phase='.1')
        row = interpolate_trace_row(first, last, .25)
        self.assertAlmostEqual(.95, float(row['current_phase']))
        self.assertEqual(first, interpolate_trace_row(first, dict(last, current_motion_id='b'), .5))
        ordinary = interpolate_trace_row(dict(first, current_phase='.1'), dict(first, current_phase='.3'), .5)
        self.assertAlmostEqual(.2, float(ordinary['current_phase']))

    def test_longrun_includes_all_repetition_boundaries(self):
        times, genres = runner.repeated_stitched_changes(606)
        self.assertIn(39., times)
        self.assertIn(585., times)
        self.assertTrue(all(a < b for a, b in zip(times, times[1:])))
        self.assertEqual(len(times), len(genres))
        self.assertEqual(77, len(times))

    def test_statistics_use_sources_not_65_independent_seeds(self):
        rows = [dict(condition=c, source_id=str(source), seed=seed, metrics={'bas': float(source) + (c != 'full')})
                for c in ('full', 'authored_timing') for source in range(13) for seed in range(5)]
        output = reanalysis.source_statistics(rows)
        self.assertEqual(13, output['paired_tests']['authored_timing:bas']['pairs'])
        self.assertEqual(13, output['summaries']['full:bas']['bootstrap_95_ci']['samples'])
        self.assertIn('holm_adjusted_p_value', output['paired_tests']['authored_timing:bas'])

    def test_missing_beats_not_zeroed_in_statistics(self):
        rows = [dict(condition=c, source_id='silent', seed=seed, metrics={'bas': None})
                for c in ('full', 'authored_timing') for seed in range(5)]
        self.assertEqual({}, reanalysis.source_statistics(rows)['paired_tests'])

    def test_per_seed_bundle_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'features.npz'
            np.savez(path, real_item_ids=[f'm{i}:motion' for i in range(40)],
                     output_item_ids=[f'm{i}:seed={seed}' for i in range(40) for seed in range(5)],
                     real_kinetic=np.zeros((40, 72)), output_kinetic=np.zeros((200, 72)),
                     real_geometric=np.zeros((40, 32)), output_geometric=np.zeros((200, 32)), extractor_identity='test')
            with mock.patch.object(reanalysis.metrics, 'literature_feature_metrics', return_value={'fid_k': 0}):
                result = reanalysis.per_seed_features(path)
            self.assertEqual(5, len(result['per_seed_40_items']))
            self.assertEqual(780, result['diversity_pairs_per_seed'])

    def test_pfc_adapted_equation_and_zero_acceleration(self):
        t = np.arange(5, dtype=float)
        root = np.column_stack((t * t, t * 0, t * 0))
        foot = np.column_stack((t * .01, t * 0, t * 0))
        self.assertAlmostEqual(1., edge_pfc_g1_adapted(root, foot, foot))
        self.assertEqual(0., edge_pfc_g1_adapted(root * 0, foot, foot))

    def test_causal_prefix_ignores_future_audio_and_never_preanalyses(self):
        class PrefixAnalyzer:
            sample_rate = 100
            block_size = 10
            plp_history_sec = 6
            def __init__(self):
                self.blocks = []
            def _callback(self, block, *args):
                self.blocks.append(block.copy())
            def drain(self):
                # Deterministic detector makes the received-data causality contract observable.
                return [float(block.sum()) for block in self.blocks]
        args = SimpleNamespace(audio_input_delay_sec=0, realtime=True, play_audio=False,
                               experiment_causal_file_input=True)
        outcomes = []
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.wav'
            path.touch()
            for suffix in (np.zeros(70), np.ones(70) * 999):
                analyzer = PrefixAnalyzer()
                with mock.patch.object(matcher.base, 'make_analyzer', return_value=analyzer), \
                     mock.patch.object(matcher.base.librosa, 'load', return_value=(np.r_[np.arange(30), suffix], 100)), \
                     mock.patch.object(matcher.MatcherFileMicrophoneSource, '_analyze_beats', side_effect=AssertionError('future scan')):
                    source = matcher.MatcherFileMicrophoneSource(path, args, 6)
                source.stream_start_wall = 100.
                with mock.patch.object(matcher.time, 'perf_counter', return_value=100.3):
                    source.advance(10.)  # Huge nominal dt must not expose future samples.
                    outcomes.append(source.drain())
                self.assertLessEqual(source.cursor, 30)
        self.assertEqual(outcomes[0], outcomes[1])

    def test_real_online_analyser_prefix_invariance_without_worker_process(self):
        with mock.patch.object(sys, 'argv', ['matcher', '--experiment-causal-file-input']):
            args = matcher.parse_args()
        args.mic_sample_rate, args.mic_block_size = 16000, 512
        args.audio_input_delay_sec, args.startup_calibration_sec = 0, 0
        args.realtime, args.play_audio = True, False
        prefix = np.zeros(32000, dtype=np.float32)
        for index in (2000, 10000, 18000, 26000):
            prefix[index:index + 160] = np.hanning(160)
        results = []
        submitted = []
        class InlineExecutor:
            def submit(self, function, *values):
                submitted.append(len(values[2]))
                future = Future()
                future.set_result(function(*values))
                return future
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.wav'
            path.touch()
            for suffix in (np.zeros(16000), np.ones(16000)):
                analyzer = matcher.base.make_analyzer(args)
                analyzer.analysis_executor = InlineExecutor()
                with mock.patch.object(matcher.base, 'make_analyzer', return_value=analyzer), \
                     mock.patch.object(matcher.base.librosa, 'load', return_value=(np.r_[prefix, suffix], 16000)):
                    source = matcher.MatcherFileMicrophoneSource(path, args, 6)
                source.stream_start_wall = 100.
                observed = []
                for step in range(1, 63):
                    with mock.patch.object(matcher.time, 'perf_counter', return_value=100 + step * .032):
                        source.advance(.032)
                        observed.extend((f.timestamp, f.is_beat, f.rms, f.beat_period) for f in source.drain())
                # Only compare before the shared prefix ends.
                results.append([f for f in observed if f[0] <= 102.])
        self.assertTrue(submitted, 'Synthetic test must exercise the real rolling-window analyser')
        self.assertEqual(results[0], results[1])


if __name__ == '__main__':
    unittest.main()
