from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "realtime" / "humanoid_robot" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import music_motion_catalog as catalog_module
from music_motion_catalog import (
    AudioDescriptor,
    CandidateStabilizer,
    MatchResult,
    MotionMatch,
    MotionProfile,
    MusicCatalog,
    MusicMotionMatcher,
    TrackMatch,
    _representative_audio_variants,
    robust_audio_normalize,
)
from realtime_music_humanoid_matcher import blend_poses


def descriptor(embedding: tuple[float, ...], rhythm: tuple[float, ...]) -> AudioDescriptor:
    return AudioDescriptor(
        embedding=np.asarray(embedding, dtype=np.float32),
        rhythm_timbre=np.asarray(rhythm, dtype=np.float32),
        tag_probabilities=np.empty(0, dtype=np.float32),
        bpm=120.0,
        beat_strength=0.8,
        onset_density=2.0,
        offbeat_ratio=0.25,
        tempo_stability=0.9,
        spectral_flux=0.6,
        percussive_ratio=0.7,
    )


def motion_profile(motion_id: str, passed: bool) -> MotionProfile:
    return MotionProfile(
        motion_id=motion_id,
        music_id="BR0",
        genre="BR",
        situation="BM",
        motion_path=f"motions/{motion_id}.pkl",
        duration_seconds=8.0,
        keypoint_phases=(0.0, 0.5),
        keypoint_scores=(0.8, 0.8),
        keypoint_density_hz=0.25,
        weighted_keypoint_density=0.2,
        keypoint_interval_cv=0.0,
        velocity_median=1.0,
        velocity_p90=1.2,
        original_bpm=120.0,
        preflight_passed=passed,
        preflight_reason="ok" if passed else "failed",
    )


class MusicMotionCatalogTests(unittest.TestCase):
    def test_robust_normalization_is_gain_invariant(self) -> None:
        time = np.linspace(0.0, 2.0, 32_000, endpoint=False)
        audio = 0.2 * np.sin(2.0 * np.pi * 220.0 * time)

        reference = robust_audio_normalize(audio)
        for gain in (0.1, 0.5, 2.0):
            np.testing.assert_allclose(
                reference,
                robust_audio_normalize(gain * audio),
                atol=2e-6,
            )

    def test_audio_variants_deduplicate_repeated_motion_audio(self) -> None:
        rows = []
        with tempfile.TemporaryDirectory() as directory:
            audio_dir = Path(directory)
            sample_rate = 8_000
            time = np.arange(sample_rate * 6) / sample_rate
            first = np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
            second = np.sin(2.0 * np.pi * 330.0 * time).astype(np.float32)
            names = (
                "gBR_sBM_cAll_d04_mBR0_ch01",
                "gBR_sBM_cAll_d04_mBR0_ch02",
                "gBR_sFM_cAll_d04_mBR0_ch03",
            )
            for index, name in enumerate(names):
                path = audio_dir / f"{name}.wav"
                sf.write(path, first if index < 2 else second, sample_rate)
                rows.append(
                    {
                        "motion_name": name,
                        "audio_path": path.name,
                        "duration_seconds": "6.0",
                    }
                )

            variants = _representative_audio_variants(rows, audio_dir)

        self.assertEqual(2, len(variants))
        basic = next(item for item in variants if item["situation"] == "BM")
        self.assertEqual(2, basic["duplicate_count"])

    def test_failed_preflight_motion_never_matches(self) -> None:
        metadata = {
            "segments": [
                {
                    "music_id": "BR0",
                    "genre": "BR",
                }
            ],
            "tracks": {
                "BR0": {
                    "music_id": "BR0",
                    "genre": "BR",
                    "motion_ids": ["passed", "failed"],
                }
            },
            "motions": {
                "passed": motion_profile("passed", True).to_dict(),
                "failed": motion_profile("failed", False).to_dict(),
            },
            "motion_stats": {
                "velocity_median": 1.0,
                "velocity_scale": 0.2,
            },
        }
        arrays = {
            "embeddings": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "rhythm_timbre": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "tags": np.empty((1, 0), dtype=np.float32),
            "embedding_mean": np.asarray([0.0, 0.0], dtype=np.float32),
            "embedding_std": np.asarray([1.0, 1.0], dtype=np.float32),
            "rhythm_mean": np.asarray([0.0, 0.0], dtype=np.float32),
            "rhythm_std": np.asarray([1.0, 1.0], dtype=np.float32),
        }
        catalog = MusicCatalog(Path("catalog.json"), metadata, arrays)

        result = MusicMotionMatcher(catalog).match(descriptor((1.0, 0.0), (1.0, 0.0)))

        self.assertEqual(["passed"], [item.motion_id for item in result.motions])

    def test_candidate_requires_consecutive_wins_and_margin(self) -> None:
        stabilizer = CandidateStabilizer(required_wins=3, margin=0.08)
        result = MatchResult(
            tracks=(TrackMatch("BR0", "BR", 0.9, 0.9, 0.9, None),),
            motions=(
                MotionMatch("new", "BR0", 0.80, 0.9, 0.7, 1.0, 0.7, 0.7, 1.0),
                MotionMatch("current", "BR0", 0.70, 0.9, 0.5, 1.0, 0.5, 0.5, 1.0),
            ),
            query_bpm=120.0,
        )

        self.assertIsNone(stabilizer.observe(result, "current"))
        self.assertIsNone(stabilizer.observe(result, "current"))
        self.assertEqual("new", stabilizer.observe(result, "current"))

    def test_candidate_rejects_ambiguous_runner_up(self) -> None:
        stabilizer = CandidateStabilizer(required_wins=1, margin=0.08)
        result = MatchResult(
            tracks=(TrackMatch("BR0", "BR", 0.9, 0.9, 0.9, None),),
            motions=(
                MotionMatch("best", "BR0", 0.80, 0.9, 0.7, 1.0, 0.7, 0.7, 1.0),
                MotionMatch("close", "BR0", 0.75, 0.9, 0.6, 1.0, 0.6, 0.6, 1.0),
            ),
            query_bpm=120.0,
        )

        self.assertIsNone(stabilizer.observe(result, None))

    def test_motion_switch_blend_is_continuous_at_endpoints(self) -> None:
        first = {"hip": -1.0, "knee": 0.5}
        second = {"hip": 1.0, "knee": -0.5}

        self.assertEqual(first, blend_poses(first, second, 0.0))
        self.assertEqual(second, blend_poses(first, second, 1.0))
        self.assertAlmostEqual(0.0, blend_poses(first, second, 0.5)["hip"])

    def test_catalog_reuses_aistpp_keypoint_detector(self) -> None:
        self.assertEqual(
            "aistpp_velocity_keypoints",
            catalog_module.detect_aistpp_file.__module__,
        )
        source = Path(catalog_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("def detect_velocity_valleys", source)
        self.assertNotIn("find_peaks(", source)


if __name__ == "__main__":
    unittest.main()
