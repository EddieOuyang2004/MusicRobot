from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEST_TOOL_DIR = ROOT / "realtime" / "humanoid_robot" / "src" / "test"
if str(TEST_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_TOOL_DIR))

from analyze_motion_audio_phase import (  # noqa: E402
    build_phase_chart,
    circular_mean_cycles,
    phase_error_metrics,
    wrapped_phase_error,
)


class MotionAudioPhaseAnalysisTests(unittest.TestCase):
    def test_wrapped_error_uses_short_path_across_cycle_boundary(self) -> None:
        self.assertAlmostEqual(0.02, wrapped_phase_error(0.01, 0.99))
        self.assertAlmostEqual(-0.02, wrapped_phase_error(0.99, 0.01))

    def test_circular_mean_handles_phase_boundary(self) -> None:
        mean = circular_mean_cycles(np.asarray([0.49, -0.49]))
        self.assertAlmostEqual(0.5, abs(mean), places=6)

    def test_metrics_separate_constant_offset_from_drift(self) -> None:
        errors = np.full(120, 0.125)
        metrics = phase_error_metrics(errors, motion_duration=8.0)

        self.assertAlmostEqual(0.125, metrics["mean_absolute_error_cycles"])
        self.assertAlmostEqual(45.0, metrics["mean_absolute_error_degrees"])
        self.assertAlmostEqual(1.0, metrics["mean_equivalent_motion_time_error_sec"])
        self.assertAlmostEqual(0.125, metrics["circular_bias_cycles"])
        self.assertAlmostEqual(0.0, metrics["bias_corrected_mean_absolute_error_cycles"])
        self.assertAlmostEqual(1.0, metrics["phase_lock_value"])

    def test_metrics_report_known_mixed_errors(self) -> None:
        metrics = phase_error_metrics([-0.10, 0.0, 0.10], motion_duration=6.0)
        self.assertAlmostEqual(2.0 / 30.0, metrics["mean_absolute_error_cycles"])
        self.assertAlmostEqual(24.0, metrics["mean_absolute_error_degrees"])
        self.assertAlmostEqual(0.4, metrics["mean_equivalent_motion_time_error_sec"])

    def test_chart_contains_phase_error_and_accepted_beat_layers(self) -> None:
        rows = [
            {
                "time_sec": 0.0,
                "original_phase": 0.0,
                "driven_phase": 0.1,
                "signed_phase_error_cycles": 0.1,
            },
            {
                "time_sec": 1.0,
                "original_phase": 0.5,
                "driven_phase": 0.6,
                "signed_phase_error_cycles": 0.1,
            },
        ]
        report = {
            "motion_path": "motion.pkl",
            "analyzed_duration_sec": 1.0,
            "conclusion_zh": "测试结论",
            "phase_error_after_first_accepted_beat": phase_error_metrics([0.1], 2.0),
            "phase_error_full_audio": phase_error_metrics([0.1], 2.0),
            "accepted_beat_diagnostics": [{"beat_audio_time_sec": 0.5}],
        }

        chart = build_phase_chart(report, rows)

        self.assertIn("Original phase", chart)
        self.assertIn("Microphone-driven phase", chart)
        self.assertIn("Signed phase error", chart)
        self.assertIn('class="beat"', chart)


if __name__ == "__main__":
    unittest.main()
