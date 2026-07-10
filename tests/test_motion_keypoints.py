from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "realtime" / "humanoid_robot" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motion_keypoints import DEFAULT_FALLBACK_PHASES, detect_motion_keypoints


def phase_distance(a: float, b: float) -> float:
    delta = abs((a - b) % 1.0)
    return min(delta, 1.0 - delta)


def make_motion(frame_count: int, build_pose) -> tuple[np.ndarray, list[dict[str, float]]]:
    phases = np.arange(frame_count, dtype=float) / frame_count
    poses = [build_pose(float(phase)) for phase in phases]
    return phases, poses


class MotionKeypointTests(unittest.TestCase):
    def test_sine_arm_swing_detects_extrema(self) -> None:
        phases, poses = make_motion(
            240,
            lambda phase: {
                "left_shoulder_pitch": math.sin(2.0 * math.pi * phase),
                "right_shoulder_pitch": -math.sin(2.0 * math.pi * phase),
            },
        )

        result = detect_motion_keypoints(phases, poses, duration=2.0, min_spacing_sec=0.35, prominence=0.12)

        self.assertFalse(result.fallback_used)
        self.assertTrue(any(phase_distance(phase, 0.25) < 0.04 for phase in result.phases), result.phases)
        self.assertTrue(any(phase_distance(phase, 0.75) < 0.04 for phase in result.phases), result.phases)

    def test_wrist_direction_change_detects_turn(self) -> None:
        def pose(phase: float) -> dict[str, float]:
            if phase < 0.5:
                value = phase / 0.5
            else:
                value = 1.0 - (phase - 0.5) / 0.5
            return {"left_wrist_yaw": value}

        phases, poses = make_motion(180, pose)

        result = detect_motion_keypoints(phases, poses, duration=1.8, min_spacing_sec=0.30, prominence=0.12)

        self.assertFalse(result.fallback_used)
        self.assertTrue(any(phase_distance(phase, 0.5) < 0.04 for phase in result.phases), result.phases)

    def test_close_duplicate_peaks_are_suppressed(self) -> None:
        phases, poses = make_motion(
            240,
            lambda phase: {
                "left_wrist_pitch": math.exp(-((phase - 0.30) / 0.018) ** 2)
                + 0.95 * math.exp(-((phase - 0.34) / 0.018) ** 2)
                + 0.75 * math.exp(-((phase - 0.78) / 0.020) ** 2)
            },
        )

        result = detect_motion_keypoints(phases, poses, duration=2.4, min_spacing_sec=0.20, prominence=0.10)

        near_first_group = [phase for phase in result.phases if 0.25 <= phase <= 0.40]
        self.assertEqual(1, len(near_first_group), result.phases)

    def test_boundary_near_peaks_are_merged(self) -> None:
        phases, poses = make_motion(
            240,
            lambda phase: {
                "right_wrist_roll": math.exp(-((phase - 0.02) / 0.018) ** 2)
                + 0.98 * math.exp(-((phase - 0.98) / 0.018) ** 2)
                + 0.90 * math.exp(-((phase - 0.52) / 0.024) ** 2)
            },
        )

        result = detect_motion_keypoints(phases, poses, duration=2.4, min_spacing_sec=0.18, prominence=0.10)

        near_boundary = [phase for phase in result.phases if phase < 0.08 or phase > 0.92]
        self.assertEqual(1, len(near_boundary), result.phases)

    def test_flat_motion_falls_back(self) -> None:
        phases, poses = make_motion(120, lambda _phase: {"left_shoulder_pitch": 0.25})

        result = detect_motion_keypoints(phases, poses, duration=1.0)

        self.assertTrue(result.fallback_used)
        self.assertEqual(DEFAULT_FALLBACK_PHASES, result.phases)


if __name__ == "__main__":
    unittest.main()
