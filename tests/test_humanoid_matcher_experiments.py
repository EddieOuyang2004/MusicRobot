from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEST_TOOL_DIR = ROOT / "realtime" / "humanoid_robot" / "src" / "test"
SRC_DIR = TEST_TOOL_DIR.parent
for directory in (str(TEST_TOOL_DIR), str(SRC_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from humanoid_matcher_experiment_metrics import (  # noqa: E402
    aggregate_pose_npz,
    aggregate_trace_csv,
    beat_alignment_score,
    bidirectional_beat_metrics,
    binary_rejection_metrics,
    diversity,
    foot_skating_metrics,
    frechet_distance,
    holm_adjust,
    literature_feature_metrics,
    paired_permutation_test,
    ranking_metrics,
    response_latency_from_trace,
    selection_diversity,
)
from music_motion_catalog import AudioFeatureExtractor  # noqa: E402
from music_motion_catalog import MusicCatalog  # noqa: E402
from aistplusplus_features import (  # noqa: E402
    extract_kinetic_features,
    extract_manual_features,
)
from build_aistpp_fact_feature_bundle import reconstruct_output_positions  # noqa: E402
import consolidate_thesis_experiments as consolidate_module  # noqa: E402
from run_humanoid_matcher_experiments import (  # noqa: E402
    archive_incomplete_run_artifacts,
    execute_runs,
    experiment_subprocess_env,
    literature_items,
    prepare_short_audio_inputs,
    run_matrix,
    validate_run_artifacts,
)


class HumanoidMatcherExperimentMetricTests(unittest.TestCase):
    def test_redirected_child_log_preserves_unicode_under_gbk_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "unicode.log"
            with mock.patch.dict("os.environ", {"PYTHONIOENCODING": "gbk"}):
                env = experiment_subprocess_env()
                with log.open("w", encoding="utf-8") as handle:
                    result = subprocess.run(
                        [sys.executable, "-c", "import sys; print('\\u00c9 / \\u00ef'); print('\\u00c9', file=sys.stderr)"],
                        stdout=handle, stderr=subprocess.STDOUT, env=env, check=False,
                    )
            self.assertEqual(result.returncode, 0)
            self.assertCountEqual(log.read_text(encoding="utf-8").splitlines(), ["\u00c9 / \u00ef", "\u00c9"])

    def test_ranking_metrics_are_music_level_compatible(self) -> None:
        result = ranking_metrics(
            ["BR", "HO", "BR"],
            [["BR", "HO"], ["BR", "HO"], ["HO", "BR"]],
        )
        self.assertAlmostEqual(1.0 / 3.0, result["recall_at_1"])
        self.assertEqual(1.0, result["recall_at_3"])
        self.assertGreater(result["mean_reciprocal_rank"], 0.6)
        self.assertIn("BR", result["per_genre"])

    def test_rejection_metrics_and_auroc(self) -> None:
        result = binary_rejection_metrics(
            [False, False, True, True],
            [False, True, True, True],
            [0.1, 0.2, 0.8, 0.9],
        )
        self.assertAlmostEqual(2.0 / 3.0, result["precision"])
        self.assertEqual(1.0, result["recall"])
        self.assertEqual(1.0, result["auroc"])
        self.assertAlmostEqual(0.5, result["false_reject_rate"])

    def test_bidirectional_bas_exposes_missing_beats(self) -> None:
        music = [0.0, 0.5, 1.0, 1.5]
        dance = [0.0, 1.0]
        result = bidirectional_beat_metrics(music, dance)
        self.assertLess(result["bas_music_to_dance"], result["bas_dance_to_music"])
        self.assertLess(result["bas_harmonic"], result["bas_dance_to_music"])
        self.assertEqual(1.0, beat_alignment_score(dance, dance))

    def test_fid_and_diversity_are_zero_for_identical_features(self) -> None:
        features = np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
        self.assertAlmostEqual(0.0, frechet_distance(features, features), places=8)
        self.assertGreater(diversity(features), 0.0)
        result = literature_feature_metrics(
            features,
            features,
            features,
            features,
            extractor_identity="official-aist++-fairmotion@commit",
        )
        self.assertEqual("official-aist++-fairmotion@commit", result["feature_extractor"])
        with self.assertRaises(ValueError):
            literature_feature_metrics(features, features, features, features, extractor_identity="")

    def test_selection_diversity_reports_concentration(self) -> None:
        result = selection_diversity(
            ["a", "a", "b", "a"],
            {"a": "c1", "b": "c2"},
            [0.9, 0.8, 0.7, 0.9],
        )
        self.assertEqual(2, result["unique_motions"])
        self.assertEqual(2, result["unique_clusters"])
        self.assertEqual(1.0, result["visual_cluster_coverage"])
        self.assertAlmostEqual(0.75, result["dominant_motion_share"])
        self.assertGreater(result["normalized_selection_entropy"], 0.0)

    def test_foot_skating_detects_contact_motion(self) -> None:
        left = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        right = np.asarray([[0.0, 0.2, 0.0], [0.0, 0.2, 0.0], [0.0, 0.2, 0.0]])
        result = foot_skating_metrics(left, right, fps=10.0)
        self.assertGreater(result["fsr"], 0.0)
        self.assertGreater(result["contact_speed_mean_m_s"], 0.0)

    def test_trace_aggregation_uses_switch_rows(self) -> None:
        fieldnames = [
            "audio_time_seconds",
            "event",
            "current_motion_id",
            "output_max_joint_speed_rad_s",
            "output_max_joint_acceleration_rad_s2",
            "switch_anchor_xy_delta_m",
            "collision_override",
            "limiter_activation_rate",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "audio_time_seconds": "0",
                        "event": "match",
                        "current_motion_id": "a",
                        "output_max_joint_speed_rad_s": "1",
                        "output_max_joint_acceleration_rad_s2": "2",
                        "collision_override": "0",
                        "limiter_activation_rate": "0",
                    }
                )
                writer.writerow(
                    {
                        "audio_time_seconds": "1",
                        "event": "switch_start",
                        "current_motion_id": "b",
                        "output_max_joint_speed_rad_s": "3",
                        "output_max_joint_acceleration_rad_s2": "4",
                        "switch_anchor_xy_delta_m": "0.01",
                        "collision_override": "1",
                        "limiter_activation_rate": "0.1",
                    }
                )
            result = aggregate_trace_csv(path)
        self.assertEqual(1, result["switch_starts"])
        self.assertEqual(1, result["collision_overrides"])
        self.assertAlmostEqual(0.01, result["transition"]["anchor_xy_delta_m"]["max"])

    def test_trace_aggregation_excludes_registered_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text(
                "audio_time_seconds,event,current_motion_id\n"
                "1,match,warmup\n"
                "6,match,stable\n",
                encoding="utf-8",
            )
            result = aggregate_trace_csv(path, start_seconds=6.0)
        self.assertEqual(1, result["rows"])
        self.assertEqual({"stable": 1}, result["selection"]["motion_counts"])

    def test_holm_adjustment_is_monotonic(self) -> None:
        result = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
        self.assertLessEqual(result["a"], result["b"])
        self.assertLessEqual(result["b"], result["c"])

    def test_paired_permutation_reports_ablation_minus_full(self) -> None:
        result = paired_permutation_test([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
        self.assertEqual(3, result["pairs"])
        self.assertAlmostEqual(1.0, result["mean_difference"])
        self.assertGreater(result["p_value"], 0.0)

    def test_response_latency_uses_expected_genre_and_milestones(self) -> None:
        fieldnames = [
            "audio_time_seconds",
            "event",
            "top_genre",
            "pending_motion_id",
            "transition_blend",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stitched.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    [
                        {"audio_time_seconds": "8.1", "event": "match", "top_genre": "JS"},
                        {"audio_time_seconds": "8.2", "pending_motion_id": "motion"},
                        {"audio_time_seconds": "8.3", "event": "switch_start"},
                        {"audio_time_seconds": "8.5", "transition_blend": "0.5"},
                        {"audio_time_seconds": "8.7", "event": "switch_complete"},
                    ]
                )
            result = response_latency_from_trace(path, [8.0], ["JS"])
        change = result["changes"][0]
        self.assertAlmostEqual(0.1, change["first_match_seconds"])
        self.assertAlmostEqual(0.7, change["stable_switch_seconds"])

    def test_pose_aggregation_uses_support_height_for_contact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.npz"
            frames = 12
            times = np.arange(frames, dtype=float) / 120.0
            bodies = np.zeros((frames, 3, 3), dtype=np.float32)
            bodies[:, 1, 0] = np.linspace(0.0, 0.1, frames)
            left = np.column_stack(
                (np.linspace(0.0, 0.1, frames), np.zeros(frames), np.full(frames, 0.1))
            )
            right = np.column_stack(
                (np.zeros(frames), np.ones(frames), np.full(frames, 0.1))
            )
            np.savez_compressed(
                path,
                time_seconds=times,
                joint_positions=np.zeros((frames, 2), dtype=np.float32),
                body_positions=bodies,
                center_of_mass=np.zeros((frames, 3), dtype=np.float32),
                left_foot_position=left,
                right_foot_position=right,
                left_foot_support_height=np.zeros(frames),
                right_foot_support_height=np.zeros(frames),
                motion_ids=np.asarray(["a"] * frames),
                accepted_causal_beat_times_seconds=np.asarray([0.0]),
                control_rate_hz=np.asarray(120.0),
            )
            result = aggregate_pose_npz(path)
        self.assertGreater(result["physical"]["contact_frames"], 0)
        self.assertGreater(result["physical"]["fsr"], 0.0)

    def test_audio_extractor_exposes_stage_timing(self) -> None:
        extractor = AudioFeatureExtractor(sample_rate=16_000)
        extractor.describe(np.zeros(16_000, dtype=np.float32))
        self.assertIn("descriptor_total_ms", extractor.last_timing_ms)
        self.assertIn("audio_preprocess_ms", extractor.last_timing_ms)

    def test_report_schema_is_closed_and_self_consistent(self) -> None:
        schema = json.loads(
            (TEST_TOOL_DIR / "humanoid_matcher_experiment.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertLessEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(1, schema["properties"]["schema_version"]["const"])

    def test_registered_run_counts_and_literature_ground_truth_are_frozen(self) -> None:
        catalog = MusicCatalog.load(
            ROOT / "realtime" / "humanoid_robot" / "data" / "music_catalog" / "catalog.json"
        )
        protocol = json.loads(
            (TEST_TOOL_DIR / "humanoid_matcher_experiment_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        items = literature_items(catalog, 4)
        self.assertEqual(40, len(items))
        for music_id, _audio, motion_id in items:
            expected = sorted(
                profile.motion_id
                for profile in catalog.motions.values()
                if profile.music_id == music_id and profile.preflight_passed
            )[0]
            self.assertEqual(expected, motion_id)
        cc0 = ROOT / "realtime" / "humanoid_robot" / "data" / "test_audio" / "cc0_matcher_set"
        self.assertEqual(200, len(run_matrix("literature", catalog, protocol, cc0)))
        self.assertEqual(350, len(run_matrix("full", catalog, protocol, cc0)))
        self.assertEqual(520, len(run_matrix("ablation", catalog, protocol, cc0)))
        self.assertEqual(40, len(run_matrix("longrun", catalog, protocol, cc0)))
        self.assertEqual(
            606.0,
            protocol["longrun"]["stable_seconds"] + protocol["longrun"]["warmup_seconds"],
        )

    def test_run_artifact_validation_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.csv"
            timing = root / "timing.json"
            pose = root / "pose.npz"
            log = root / "run.log"
            trace.write_text("audio_time_seconds,current_motion_id\n0,a\n", encoding="utf-8")
            timing.write_text('{"iterations": 2}\n', encoding="utf-8")
            np.savez_compressed(
                pose,
                time_seconds=np.asarray([0.0, 0.1]),
                joint_positions=np.zeros((2, 2)),
                motion_ids=np.asarray(["a", "a"]),
            )
            log.write_text("ok\n", encoding="utf-8")
            self.assertEqual((True, []), validate_run_artifacts(trace, timing, pose, log))
            timing.write_text('{"iterations": 0}\n', encoding="utf-8")
            valid, errors = validate_run_artifacts(trace, timing, pose, log)
            self.assertFalse(valid)
            self.assertTrue(any("iterations" in value for value in errors))
            timing.write_text('{"iterations": 2}\n', encoding="utf-8")
            valid, errors = validate_run_artifacts(
                trace,
                timing,
                pose,
                log,
                minimum_duration_seconds=1.0,
            )
            self.assertFalse(valid)
            self.assertTrue(any("ends at" in value for value in errors))

    def test_short_audio_is_repeated_without_modifying_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "short.wav"
            samples = np.arange(8_000, dtype=np.int16)
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(8_000)
                handle.writeframes(samples.tobytes())
            original_hash = source.read_bytes()
            matrix = [{"audio": source}]
            prepare_short_audio_inputs(
                matrix,
                root / "output",
                required_seconds=3.0,
                sample_rate=8_000,
                dry_run=False,
            )
            prepared = Path(matrix[0]["audio"])
            with wave.open(str(prepared), "rb") as handle:
                duration = handle.getnframes() / handle.getframerate()
            self.assertEqual(4.0, duration)
            self.assertTrue(matrix[0]["audio_was_repeated"])
            self.assertEqual(original_hash, source.read_bytes())

    def test_archive_long_run_name_preserves_artifacts_and_retry_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / ("r" * max(1, 105 - len(str(root)) - 1))
            runs.mkdir()
            stem = "0390_fixed_entry_simple_transition_s0_aistpp_stitched_test"
            paths = [runs / f"{stem}{suffix}" for suffix in (".csv", "_timing.json", "_poses.npz", ".log")]
            old_destination = runs / "failed_attempts" / stem / "20260905T023025_229720Z" / paths[2].name
            self.assertGreaterEqual(len(str(old_destination)), 260)
            for path in paths:
                path.write_bytes(b"original result")
            archive = archive_incomplete_run_artifacts(stem, paths, runs)
            for path in paths:
                self.assertFalse(path.exists())
                self.assertLess(len(str(archive / path.name)), 260)
                self.assertEqual((archive / path.name).read_bytes(), b"original result")
            paths[0].write_bytes(b"retry result")
            retry_archive = archive_incomplete_run_artifacts(stem, paths, runs)
            self.assertNotEqual(archive, retry_archive)
            self.assertEqual((retry_archive / paths[0].name).read_bytes(), b"retry result")
            self.assertEqual((archive / paths[0].name).read_bytes(), b"original result")

    def test_resume_reuses_only_a_complete_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runs = output / "runs"
            runs.mkdir()
            stem = "0000_full_s0_source"
            trace = runs / f"{stem}.csv"
            timing = runs / f"{stem}_timing.json"
            pose = runs / f"{stem}_poses.npz"
            log = runs / f"{stem}.log"
            trace.write_text(
                "audio_time_seconds,current_motion_id\n0,a\n0.95,a\n",
                encoding="utf-8",
            )
            timing.write_text('{"iterations": 1}', encoding="utf-8")
            np.savez_compressed(
                pose,
                time_seconds=np.asarray([0.0, 0.95]),
                joint_positions=np.zeros((2, 1)),
                motion_ids=np.asarray(["a", "a"]),
            )
            log.write_text("complete", encoding="utf-8")
            matrix = [
                {
                    "condition": "full",
                    "condition_arguments": [],
                    "seed": 0,
                    "source_id": "source",
                    "audio": output / "unused.wav",
                    "ground_truth_motion_id": None,
                }
            ]
            results, failures = execute_runs(
                matrix,
                output,
                ROOT / "realtime" / "humanoid_robot" / "data" / "music_catalog" / "catalog.json",
                max_seconds=1.0,
                dry_run=False,
                resume=True,
                suite="smoke",
                max_attempts=2,
            )
        self.assertEqual("reused", results[0]["status"])
        self.assertEqual([], failures)

    def test_pinned_official_feature_shapes(self) -> None:
        rng = np.random.default_rng(7)
        positions = rng.normal(size=(12, 24, 3))
        self.assertEqual((72,), extract_kinetic_features(positions).shape)
        self.assertEqual((32,), extract_manual_features(positions).shape)

    def test_trace_reconstruction_uses_phase_and_transition_blend(self) -> None:
        class FakeCache:
            @staticmethod
            def sample_phase(motion_id: str, phase: float) -> np.ndarray:
                offset = 10.0 if motion_id == "b" else 0.0
                return np.full((24, 3), offset + phase)

        rows = [
            {
                "audio_time_seconds": "0.0",
                "current_motion_id": "a",
                "current_phase": "0.0",
                "transition_motion_id": "",
                "transition_phase": "",
                "transition_blend": "0",
            },
            {
                "audio_time_seconds": "1.0",
                "current_motion_id": "a",
                "current_phase": "0.5",
                "transition_motion_id": "b",
                "transition_phase": "0.25",
                "transition_blend": "0.5",
            },
            {
                "audio_time_seconds": "2.0",
                "current_motion_id": "b",
                "current_phase": "0.5",
                "transition_motion_id": "",
                "transition_phase": "",
                "transition_blend": "0",
            },
        ]
        result = reconstruct_output_positions(
            rows, FakeCache(), start_seconds=0.0, clip_seconds=2.0, fps=1.0
        )
        self.assertEqual((2, 24, 3), result.shape)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_consolidator_writes_ready_only_for_complete_valid_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            output = root / "results"
            identity = {
                "git_commit": "abc",
                "catalog": {"sha256": "catalog", "arrays_sha256": "arrays"},
                "models": {"embedding_model": {"sha256": "model"}},
                "protocol": {"sha256": "protocol"},
                "output_schema": {"sha256": "schema"},
            }
            offline = {
                "manifest": {**identity, "onnxruntime_providers": ["CPUExecutionProvider"]},
                "protocol": {},
                "quality": {},
                "audio_evaluation": {},
                "failures": [],
            }
            (raw / "offline").mkdir(parents=True)
            (raw / "offline" / "experiment_report.json").write_text(
                json.dumps(offline), encoding="utf-8"
            )
            suite = raw / "preflight"
            runs = suite / "runs"
            runs.mkdir(parents=True)
            trace, timing, pose, log = (
                runs / "run.csv",
                runs / "timing.json",
                runs / "pose.npz",
                runs / "run.log",
            )
            trace.write_text(
                "audio_time_seconds,current_motion_id\n0,a\n9.95,a\n",
                encoding="utf-8",
            )
            timing.write_text('{"iterations": 1}', encoding="utf-8")
            np.savez_compressed(
                pose,
                time_seconds=np.asarray([0.0, 9.95]),
                joint_positions=np.zeros((2, 1)),
                motion_ids=np.asarray(["a", "a"]),
            )
            log.write_text("ok", encoding="utf-8")
            run = {
                "run_key": "run",
                "status": "completed",
                "trace": str(trace),
                "timing": str(timing),
                "pose": str(pose),
                "log": str(log),
            }
            (suite / "run_status.json").write_text(
                json.dumps({"suite": "smoke", "expected_runs": 1, "runs": [run]}),
                encoding="utf-8",
            )
            (suite / "experiment_report.json").write_text(
                json.dumps({"manifest": identity, "failures": [], "acceptance": {"all_evaluated_passed": False}}),
                encoding="utf-8",
            )
            (suite / "preflight_validation.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            features = raw / "features" / "aistpp_fact_features.npz"
            features.parent.mkdir(parents=True)
            np.savez_compressed(
                features,
                real_kinetic=np.zeros((40, 72), dtype=np.float32),
                output_kinetic=np.zeros((200, 72), dtype=np.float32),
                real_geometric=np.zeros((40, 32), dtype=np.float32),
                output_geometric=np.zeros((200, 32), dtype=np.float32),
                extractor_identity=np.asarray("official@test"),
            )
            (features.with_suffix(".manifest.json")).write_text(
                json.dumps({"catalog_sha256": "catalog"}), encoding="utf-8"
            )
            with mock.patch.object(consolidate_module, "EXPECTED_RUNS", {"preflight": 1}):
                marker = consolidate_module.consolidate(raw, output)
            self.assertTrue(marker["ready"])
            self.assertTrue((output / "READY_FOR_THESIS").is_file())
            self.assertTrue((output / "consolidated_results.json").is_file())
            self.assertTrue((output / "consolidated_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
