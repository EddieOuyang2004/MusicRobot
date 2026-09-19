"""Tests for statistical accounting and sample exclusion in the v2 audit."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realtime/humanoid_robot/src/test"))
from run_v2_selection_audit import concentration, trace_counts, without_music


class AuditTests(unittest.TestCase):
    def test_songs_are_equally_weighted_despite_different_window_counts(self):
        report = concentration([{"a": 100}, {"b": 1}, {}], 3)
        self.assertEqual(report["songs_with_observations"], 2)
        self.assertEqual(report["songs_total"], 3)
        self.assertAlmostEqual(report["top1_share"], 0.5)
        self.assertAlmostEqual(report["effective_ids"], 2.0)

    def test_no_observations_is_not_zero_concentration(self):
        self.assertIsNone(concentration([{}, {}], 2)["top1_share"])

    def test_exclude_initial_motion_and_count_replays_once_not_per_frame(self):
        def row(t, event, motion):
            return {"audio_time_seconds": str(t), "event": event, "current_motion_id": motion}
        rows = [row(0, "", "startup"), row(5, "match", "startup"),
                row(10, "switch_complete", "a"), row(11, "match", "a"),
                row(15, "switch_complete", "a"), row(18, "", "a"),
                row(20, "switch_complete", "b"), row(25, "", "b")]
        plays, dwell, replays = trace_counts(rows)
        self.assertEqual(dict(plays), {"a": 2, "b": 1})
        self.assertEqual(dict(dwell), {"a": 10, "b": 5})
        self.assertEqual(replays, 1)
        self.assertEqual(trace_counts(rows[:2])[0], {})

    def test_leave_music_out_removes_all_variants_and_preserves_normalization(self):
        catalog = SimpleNamespace(
            catalog_path=Path("catalog.json"),
            segment_metadata=[{"music_id": "A", "variant": 1}, {"music_id": "B"}, {"music_id": "A", "variant": 2}],
            tracks={"A": {}, "B": {}},
            embeddings=np.array([[1, 2], [3, 4], [5, 6]]),
            rhythm_timbre=np.array([[1], [2], [3]]), tags=np.zeros((3, 0)),
            embedding_mean=np.array([2, 3]), embedding_std=np.array([1, 1]),
            rhythm_mean=np.array([2]), rhythm_std=np.array([1]),
            metadata={"motions": {"motion_a": {"music_id": "A"}},
                      "motion_stats": {"velocity_median": 1, "velocity_scale": 1}})
        filtered = without_music(catalog, "A")
        self.assertEqual(filtered.segment_metadata, [{"music_id": "B"}])
        self.assertEqual(set(filtered.tracks), {"B"})
        self.assertEqual(filtered.motions, {})
        np.testing.assert_array_equal(filtered.embeddings, [[3, 4]])
        np.testing.assert_array_equal(filtered.embedding_mean, catalog.embedding_mean)


if __name__ == "__main__":
    unittest.main()
