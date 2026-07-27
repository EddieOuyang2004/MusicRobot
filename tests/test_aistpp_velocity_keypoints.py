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

from aistpp_velocity_keypoints import (
    detect_aistpp_velocity_keypoints,
    detect_velocity_valleys,
    joint_position_velocity,
    smpl_angular_velocity,
)


def phase_distance(a: float, b: float) -> float:
    return abs(a - b)


class AistppVelocityKeypointTests(unittest.TestCase):
    def test_joint_position_velocity_finds_sine_extrema(self) -> None:
        fps = 120.0
        frame_count = 240
        phases = np.arange(frame_count, dtype=float) / frame_count
        joints = np.zeros((frame_count, 2, 3), dtype=float)
        joints[:, 0, 0] = np.sin(2.0 * math.pi * phases)
        joints[:, 1, 1] = -0.6 * np.sin(2.0 * math.pi * phases)
        poses = np.zeros((frame_count, 72), dtype=float)

        result = detect_aistpp_velocity_keypoints(
            poses,
            joints3d=joints,
            fps=fps,
            smoothing_sec=0.02,
            min_spacing_sec=0.30,
            prominence=0.20,
            boundary_sec=0.02,
        )

        self.assertEqual("3d-joint-position", result.signal_source)
        self.assertTrue(any(phase_distance(phase, 0.25) < 0.02 for phase in result.phases), result.phases)
        self.assertTrue(any(phase_distance(phase, 0.75) < 0.02 for phase in result.phases), result.phases)

    def test_smpl_geodesic_velocity_finds_rotation_turns(self) -> None:
        fps = 120.0
        frame_count = 240
        phases = np.arange(frame_count, dtype=float) / frame_count
        poses = np.zeros((frame_count, 24, 3), dtype=float)
        poses[:, 16, 0] = 1.1 * np.sin(2.0 * math.pi * phases)
        poses[:, 17, 1] = -0.8 * np.sin(2.0 * math.pi * phases)

        velocity = smpl_angular_velocity(poses, fps)
        result = detect_velocity_valleys(
            velocity,
            fps=fps,
            smoothing_sec=0.02,
            min_spacing_sec=0.30,
            prominence=0.20,
            boundary_sec=0.02,
        )

        self.assertTrue(any(phase_distance(phase, 0.25) < 0.02 for phase in result.phases), result.phases)
        self.assertTrue(any(phase_distance(phase, 0.75) < 0.02 for phase in result.phases), result.phases)

    def test_translation_scaling_is_applied(self) -> None:
        fps = 100.0
        frame_count = 200
        phases = np.arange(frame_count, dtype=float) / frame_count
        poses = np.zeros((frame_count, 72), dtype=float)
        translations = np.zeros((frame_count, 3), dtype=float)
        translations[:, 0] = 80.0 * np.sin(2.0 * math.pi * phases)

        result = detect_aistpp_velocity_keypoints(
            poses,
            smpl_trans=translations,
            smpl_scaling=80.0,
            fps=fps,
            smoothing_sec=0.02,
            min_spacing_sec=0.30,
            prominence=0.20,
            boundary_sec=0.02,
        )

        self.assertEqual("smpl-angular+root-translation", result.signal_source)
        self.assertTrue(any(phase_distance(phase, 0.25) < 0.02 for phase in result.phases), result.phases)
        self.assertTrue(any(phase_distance(phase, 0.75) < 0.02 for phase in result.phases), result.phases)

    def test_close_valleys_obey_minimum_spacing(self) -> None:
        fps = 100.0
        frames = np.arange(200, dtype=float)
        velocity = np.ones(200, dtype=float)
        velocity -= 0.9 * np.exp(-0.5 * ((frames - 60.0) / 2.0) ** 2)
        velocity -= 0.8 * np.exp(-0.5 * ((frames - 72.0) / 2.0) ** 2)
        velocity -= 0.9 * np.exp(-0.5 * ((frames - 150.0) / 2.0) ** 2)

        result = detect_velocity_valleys(
            velocity,
            fps=fps,
            smoothing_sec=0.01,
            min_spacing_sec=0.20,
            prominence=0.20,
            boundary_sec=0.0,
        )

        first_group = [frame for frame in result.frame_indices if 50 <= frame <= 80]
        self.assertEqual(1, len(first_group), result.frame_indices)
        self.assertTrue(any(abs(frame - 150) <= 2 for frame in result.frame_indices), result.frame_indices)

    def test_clip_boundary_valley_is_rejected(self) -> None:
        fps = 100.0
        frames = np.arange(200, dtype=float)
        velocity = np.ones(200, dtype=float)
        velocity -= 0.9 * np.exp(-0.5 * ((frames - 2.0) / 1.0) ** 2)
        velocity -= 0.8 * np.exp(-0.5 * ((frames - 100.0) / 2.0) ** 2)

        result = detect_velocity_valleys(
            velocity,
            fps=fps,
            smoothing_sec=0.0,
            min_spacing_sec=0.10,
            prominence=0.20,
            boundary_sec=0.05,
        )

        self.assertFalse(any(frame < 5 for frame in result.frame_indices), result.frame_indices)
        self.assertTrue(any(abs(frame - 100) <= 1 for frame in result.frame_indices), result.frame_indices)

    def test_flat_velocity_returns_no_keypoints(self) -> None:
        result = detect_velocity_valleys(np.ones(120), fps=60.0)

        self.assertEqual((), result.frame_indices)
        self.assertIn("flat", result.reason)

    def test_joint_position_velocity_has_one_value_per_frame(self) -> None:
        positions = np.zeros((10, 3, 3), dtype=float)
        positions[:, :, 0] = np.arange(10, dtype=float)[:, None]

        velocity = joint_position_velocity(positions, fps=2.0)

        np.testing.assert_allclose(velocity, 2.0)


if __name__ == "__main__":
    unittest.main()
