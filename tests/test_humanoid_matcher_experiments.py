from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

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


class HumanoidMatcherExperimentMetricTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
