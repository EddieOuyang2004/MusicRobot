from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEST_TOOL_DIR = ROOT / "realtime" / "humanoid_robot" / "src" / "test"
if str(TEST_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_TOOL_DIR))

from analyze_aistpp_keypoint_bpm import (  # noqa: E402
    canonical_clip_id,
    nearest_beat_offsets,
    safe_correlation,
)


class AistppKeypointBpmAnalysisTests(unittest.TestCase):
    def test_camera_id_maps_to_camera_independent_asset(self) -> None:
        self.assertEqual(
            "gLH_sBM_cAll_d17_mLH0_ch02",
            canonical_clip_id("gLH_sBM_c01_d17_mLH0_ch02"),
        )

    def test_constant_tempo_does_not_produce_spurious_correlation(self) -> None:
        result = safe_correlation(
            np.full(8, 159.0),
            np.asarray([70.0, 90.0, 80.0, 100.0, 60.0, 85.0, 75.0, 95.0]),
        )
        self.assertEqual("insufficient_variation", result["status"])
        self.assertIsNone(result["pearson_r"])

    def test_nearest_beat_offsets_keep_sign(self) -> None:
        offsets = nearest_beat_offsets(
            np.asarray([0.9, 2.1]), np.asarray([0.0, 1.0, 2.0, 3.0])
        )
        np.testing.assert_allclose(offsets, [-0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
