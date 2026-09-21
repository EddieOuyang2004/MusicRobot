from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realtime" / "humanoid_robot" / "src"))

from aistpp_smpl import load_aistpp_motion
from segment_finedance import (
    build, load_finedance, parse_args, resample_motion, rotation_6d_to_matrix,
    segment_bounds, source_splits,
)


class FineDanceSegmentationTests(unittest.TestCase):
    def test_rotation_rows_and_invalid_vectors(self):
        matrices = Rotation.from_euler("xyz", [[23, 47, -61], [-11, 99, 137]], degrees=True).as_matrix()
        encoded = matrices[:, :2, :].reshape(-1, 6)
        np.testing.assert_allclose(rotation_6d_to_matrix(encoded), matrices, atol=1e-12)
        for invalid in (np.zeros((1, 6)), [[1, 0, 0, 2, 0, 0]], [[np.nan] * 6]):
            with self.assertRaises(ValueError):
                rotation_6d_to_matrix(invalid)

    def test_all_frame_counts_have_legal_nonoverlapping_segments(self):
        for count in range(1, 10_001):
            bounds = segment_bounds(count)
            if count < 480:
                self.assertEqual(bounds, [])
                continue
            self.assertEqual(bounds[0][0], 0)
            self.assertLessEqual(bounds[-1][1], count)
            for index, (start, end) in enumerate(bounds):
                self.assertTrue(480 <= end - start <= 720)
                if index:
                    self.assertEqual(start, bounds[index - 1][1])
            if (count + 719) // 720 <= count // 480:
                self.assertEqual(bounds[-1][1], count)
        self.assertEqual(segment_bounds(15 * 60), [(0, 720)])
        self.assertEqual(segment_bounds(20 * 60), [(0, 600), (600, 1200)])

    def test_slerp_short_arc_real_timestamps_and_neutral_hands(self):
        matrices = np.tile(np.eye(3), (2, 52, 1, 1))
        matrices[:, 0] = Rotation.from_euler("z", [170, -170], degrees=True).as_matrix()
        matrices[:, 22:] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        trans = np.array([[0, 0, 0], [2, 4, 6]], dtype=float)
        poses, moved = resample_motion(trans, matrices, 4, 30, 1.3)
        poses = poses.reshape(4, 24, 3)
        mid = Rotation.from_rotvec(poses[1, 0]).as_matrix()
        np.testing.assert_allclose(mid, Rotation.from_euler("z", 180, degrees=True).as_matrix(), atol=2e-7)
        np.testing.assert_allclose(moved[1], [1, 3.3, 3], atol=2e-7)
        np.testing.assert_allclose(moved[2], moved[3])
        np.testing.assert_array_equal(poses[:, 22:], 0)

    def make_source(self, root: Path, name: str = "001", seconds: int = 20,
                    audio_seconds: int | None = None) -> None:
        for folder in ("motion", "music_wav", "label_json"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        frames = seconds * 30
        motion = np.zeros((frames, 315), dtype=np.float32)
        motion[:, 0] = np.arange(frames) / 30
        motion[:, 3:] = np.tile([1, 0, 0, 0, 1, 0], 52)
        np.save(root / "motion" / f"{name}.npy", motion)
        audio_seconds = seconds if audio_seconds is None else audio_seconds
        t = np.arange(audio_seconds * 48000) / 48000
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
        # A timing marker in the second segment detects independently reset or
        # incorrectly converted audio/motion start indices.
        audio[int(12 * 48000):int(12.01 * 48000)] = 0.9
        sf.write(root / "music_wav" / f"{name}.wav", audio, 48000)
        (root / "label_json" / f"{name}.json").write_text(json.dumps({"style2": "Jazz"}))

    def test_end_to_end_export_loader_audio_alignment_and_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            self.make_source(root)
            report = build(parse_args(["--input-root", str(root), "--output-root", str(output)]))
            self.assertEqual(report["clip_count"], 2)
            self.assertTrue(report["export_complete"])
            self.assertEqual(len((output / "train.txt").read_text().splitlines()), 2)
            second = report["clips"][1]
            poses, trans = load_aistpp_motion(output / second["motion_path"])
            self.assertEqual(poses.shape, (600, 24, 3))
            np.testing.assert_allclose(trans[0], [10, 1.3, 0], atol=1e-6)
            audio, rate = sf.read(output / second["audio_path"], always_2d=True)
            self.assertEqual((len(audio), rate, audio.shape[1]), (441000, 44100, 2))
            self.assertGreater(audio[int(2.003 * rate), 0], 0.85)
            with (output / second["motion_path"]).open("rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(payload["fps"], 60)
            self.assertEqual(payload["smpl_scaling"].tolist(), [1.0])
            self.assertEqual(second["label"]["style2"], "Jazz")
            with self.assertRaises(FileExistsError):
                build(parse_args(["--input-root", str(root), "--output-root", str(output)]))

    def test_mismatches_ignored_sources_and_explicit_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "source"
            self.make_source(root, seconds=20, audio_seconds=18)
            self.make_source(root, name="130")
            args = ["--input-root", str(root), "--dry-run"]
            report = build(parse_args(args + ["--output-root", str(base / "audit")]))
            self.assertEqual(report["clip_count"], 0)
            self.assertEqual({x["reason"] for x in report["excluded"]}, {"duration_mismatch", "split_ignore"})
            self.assertFalse((base / "audit" / "motions").exists())
            report = build(parse_args(args + ["--output-root", str(base / "allowed"), "--allow-duration-mismatch"]))
            self.assertEqual(report["sources"][0]["bounds"][-1][1], 18 * 60)
            self.assertTrue(report["sources"][0]["duration_mismatch_allowed"])
            self.assertEqual(report["sources"][0]["dropped_motion_seconds"], 2)

    def test_custom_splits_reject_leakage_and_preserve_official_test(self):
        self.assertEqual(source_splits(["001", "063", "130"], None),
                         {"001": "train", "063": "test", "130": "ignore"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({"train": ["001"], "test": [1]}))
            with self.assertRaises(ValueError):
                source_splits(["001"], path)
            path.write_text(json.dumps({"train": ["001"]}))
            with self.assertRaises(ValueError):
                source_splits(["001", "002"], path)

    def test_preprocessed_and_axis_angle_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = np.zeros((2, 315))
            data[:, 3:] = np.tile([1, 0, 0, 0, 1, 0], 52)
            path = root / "motion.npy"
            np.save(path, np.column_stack((np.ones((2, 4)), data)))
            trans, matrices = load_finedance(path)
            np.testing.assert_array_equal(trans, 0)
            np.testing.assert_allclose(matrices, np.tile(np.eye(3), (2, 52, 1, 1)))
            np.save(path, np.zeros((2, 159)))
            _, axis_matrices = load_finedance(path)
            np.testing.assert_allclose(matrices, axis_matrices)
            np.save(path, np.zeros((2, 312)))
            with self.assertRaises(ValueError):
                load_finedance(path)


if __name__ == "__main__":
    unittest.main()
