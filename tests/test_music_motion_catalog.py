from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "realtime" / "humanoid_robot" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import music_motion_catalog as catalog_module
from music_motion_catalog import (
    AudioDescriptor,
    AudioFeatureExtractor,
    CandidateStabilizer,
    DiversityCandidateSelector,
    MatchResult,
    MotionMatch,
    MotionProfile,
    MotionSelectionPolicy,
    MusicCatalog,
    MusicMotionMatcher,
    TrackMatch,
    _representative_audio_variants,
    robust_audio_normalize,
)
from realtime_music_humanoid_matcher import blend_poses, make_controller


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


def motion_match(
    motion_id: str,
    final_score: float,
    music_score: float = 0.9,
) -> MotionMatch:
    return MotionMatch(
        motion_id,
        "BR0",
        final_score,
        music_score,
        0.7,
        1.0,
        0.7,
        0.7,
        1.0,
    )


def match_result(*motions: MotionMatch) -> MatchResult:
    return MatchResult(
        tracks=(TrackMatch("BR0", "BR", 0.9, 0.9, 0.9, None),),
        motions=motions,
        query_bpm=120.0,
    )


class MusicMotionCatalogTests(unittest.TestCase):
    def test_effnet_extraction_skips_dsp_and_hpss_feature_work(self) -> None:
        class FakeEffnetBackend:
            output_dimension = 3

            @staticmethod
            def encode(_samples: np.ndarray, _sample_rate: int) -> np.ndarray:
                return np.asarray([0.6, 0.8, 0.0], dtype=np.float32)

        extractor = AudioFeatureExtractor.__new__(AudioFeatureExtractor)
        extractor.sample_rate = 16_000
        extractor.embedding_backend = FakeEffnetBackend()
        extractor.tag_backend = None
        samples = np.linspace(-0.5, 0.5, 1_024, dtype=np.float32)
        mfcc = np.arange(26, dtype=np.float32).reshape(13, 2)
        contrast = np.arange(12, dtype=np.float32).reshape(6, 2)

        with (
            patch.object(
                catalog_module,
                "robust_audio_normalize",
                return_value=samples,
            ),
            patch.object(catalog_module.np, "percentile", return_value=1.0),
            patch.object(
                catalog_module.librosa.onset,
                "onset_strength",
                return_value=np.asarray([0.5, 1.0], dtype=np.float32),
            ),
            patch.object(
                catalog_module.librosa.beat,
                "beat_track",
                return_value=(120.0, np.empty(0, dtype=int)),
            ),
            patch.object(
                catalog_module.librosa.onset,
                "onset_detect",
                return_value=np.empty(0, dtype=int),
            ),
            patch.object(
                catalog_module.librosa.feature,
                "melspectrogram",
                return_value=np.ones((64, 2), dtype=np.float32),
            ),
            patch.object(
                catalog_module.librosa,
                "power_to_db",
                return_value=np.zeros((64, 2), dtype=np.float32),
            ),
            patch.object(catalog_module.librosa.feature, "mfcc", return_value=mfcc),
            patch.object(
                catalog_module.librosa.feature,
                "spectral_contrast",
                return_value=contrast,
            ),
            patch.object(
                catalog_module.librosa.feature,
                "chroma_stft",
                side_effect=AssertionError("EffNet must not calculate chroma"),
            ) as chroma_mock,
            patch.object(
                catalog_module.librosa.effects,
                "hpss",
                side_effect=AssertionError("Matcher must not calculate HPSS"),
            ) as hpss_mock,
            patch.object(
                catalog_module.np,
                "std",
                side_effect=AssertionError("EffNet must not calculate DSP statistics"),
            ) as std_mock,
            patch.object(
                catalog_module.np,
                "concatenate",
                side_effect=AssertionError("EffNet must not build a DSP embedding"),
            ) as concatenate_mock,
        ):
            result = extractor.describe(samples)

        np.testing.assert_allclose(result.embedding, [0.6, 0.8, 0.0])
        self.assertEqual((24,), result.rhythm_timbre.shape)
        self.assertEqual(0.0, result.percussive_ratio)
        chroma_mock.assert_not_called()
        hpss_mock.assert_not_called()
        std_mock.assert_not_called()
        concatenate_mock.assert_not_called()

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

    def test_candidate_compares_best_with_current_not_runner_up(self) -> None:
        stabilizer = CandidateStabilizer(required_wins=1, margin=0.08)
        result = MatchResult(
            tracks=(TrackMatch("BR0", "BR", 0.9, 0.9, 0.9, None),),
            motions=(
                MotionMatch("best", "BR0", 0.80, 0.9, 0.7, 1.0, 0.7, 0.7, 1.0),
                MotionMatch("close", "BR0", 0.75, 0.9, 0.6, 1.0, 0.6, 0.6, 1.0),
            ),
            query_bpm=120.0,
        )

        self.assertEqual("best", stabilizer.observe(result, None))

    def test_diversity_selector_requires_consecutive_eligible_results(self) -> None:
        selector = DiversityCandidateSelector(required_wins=3)
        result = match_result(
            motion_match("current", 0.90),
            motion_match("alternative", 0.87),
        )

        selector.observe(result)
        selector.observe(result)
        self.assertIsNone(selector.select("current"))

        selector.observe(match_result(motion_match("current", 0.90)))
        selector.observe(result)
        self.assertIsNone(selector.select("current"))
        selector.observe(result)
        selector.observe(result)
        self.assertEqual("alternative", selector.select("current").motion_id)

    def test_diversity_selector_applies_score_and_music_quality_gates(self) -> None:
        selector = DiversityCandidateSelector(
            required_wins=1,
            top_k=5,
            score_drop=0.05,
            music_score_drop=0.08,
        )
        selector.observe(
            match_result(
                motion_match("current", 0.90, 0.90),
                motion_match("good", 0.86, 0.84),
                motion_match("low-final", 0.84, 0.90),
                motion_match("low-music", 0.87, 0.81),
            )
        )

        self.assertEqual(
            ["good"],
            [item.motion_id for item in selector.stable_candidates("current")],
        )

    def test_diversity_selector_rotates_deterministically_by_recency(self) -> None:
        selector = DiversityCandidateSelector(required_wins=1, recent_history=3)
        result = match_result(
            motion_match("first", 0.90),
            motion_match("second", 0.89),
            motion_match("third", 0.88),
            motion_match("fourth", 0.87),
        )
        selector.record_played("first")
        selector.observe(result)

        self.assertEqual("second", selector.select("first").motion_id)
        selector.record_played("second")
        self.assertEqual("third", selector.select("second").motion_id)
        selector.record_played("third")
        self.assertEqual("fourth", selector.select("third").motion_id)
        selector.record_played("fourth")
        self.assertEqual("first", selector.select("fourth").motion_id)

    def test_diversity_selector_holds_when_only_current_is_eligible(self) -> None:
        selector = DiversityCandidateSelector(required_wins=1)
        selector.observe(match_result(motion_match("current", 0.90)))

        self.assertIsNone(selector.select("current"))

    def test_selection_policy_latches_diversity_until_fourth_bar(self) -> None:
        policy = MotionSelectionPolicy(
            current_motion_id="current",
            required_wins=1,
            max_hold_bars=4,
        )
        result = match_result(
            motion_match("current", 0.90),
            motion_match("alternative", 0.87),
        )
        policy.observe(result)
        self.assertIsNone(policy.pending)
        for _ in range(3):
            self.assertIsNone(policy.on_bar_boundary())

        pending = policy.observe(result)
        self.assertIsNotNone(pending)
        self.assertEqual("diversity", pending.reason)
        changed_result = match_result(
            motion_match("current", 0.90),
            motion_match("different", 0.89),
        )
        self.assertEqual(pending, policy.observe(changed_result))
        self.assertEqual(pending, policy.on_bar_boundary())

    def test_selection_policy_relevance_switch_preempts_hold_limit(self) -> None:
        policy = MotionSelectionPolicy(
            current_motion_id="current",
            required_wins=3,
            score_margin=0.08,
            max_hold_bars=4,
        )
        result = match_result(
            motion_match("new", 0.85),
            motion_match("runner-up", 0.84),
        )

        self.assertIsNone(policy.observe(result))
        self.assertIsNone(policy.observe(result))
        pending = policy.observe(result)
        self.assertEqual("new", pending.motion_id)
        self.assertEqual("relevance", pending.reason)
        self.assertEqual(pending, policy.on_bar_boundary())

    def test_diversity_can_become_ready_on_the_hold_limit_boundary(self) -> None:
        policy = MotionSelectionPolicy(
            current_motion_id="current",
            required_wins=3,
            max_hold_bars=4,
        )
        result = match_result(
            motion_match("current", 0.90),
            motion_match("alternative", 0.87),
        )
        policy.observe(result)
        policy.observe(result)
        for _ in range(4):
            policy.on_bar_boundary()

        pending = policy.observe(result)
        self.assertEqual("diversity", pending.reason)
        self.assertEqual(pending, policy.ready_selection())

    def test_pending_motion_is_first_in_preload_order(self) -> None:
        policy = MotionSelectionPolicy(
            current_motion_id="current",
            required_wins=1,
        )
        result = match_result(
            motion_match("new", 0.90),
            motion_match("other", 0.80),
        )
        policy.observe(result)

        self.assertEqual(("new", "other"), policy.preload_motion_ids(result))

    def test_motion_switch_blend_is_continuous_at_endpoints(self) -> None:
        first = {"hip": -1.0, "knee": 0.5}
        second = {"hip": 1.0, "knee": -0.5}

        self.assertEqual(first, blend_poses(first, second, 0.0))
        self.assertEqual(second, blend_poses(first, second, 1.0))
        self.assertAlmostEqual(0.0, blend_poses(first, second, 0.5)["hip"])

    def test_motion_switch_continues_after_strongest_keypoint(self) -> None:
        args = Namespace(
            smoothing_tau=0.1,
            speed_min=0.5,
            speed_max=2.0,
            amp_min=0.3,
            amp_max=1.0,
            accent_duration=0.1,
            tempo_timeout=1.0,
            beat_confidence_threshold=0.4,
            beat_keypoint_interval_ratio=1.0,
            beat_selection_mode="adaptive",
            beat_contrast_weight=0.5,
        )
        source_profile = motion_profile("source", True)
        target_profile = replace(
            motion_profile("target", True),
            keypoint_phases=(0.1, 0.4, 0.8),
            keypoint_scores=(0.2, 0.9, 0.3),
        )
        previous = make_controller(args, source_profile)
        previous.last_beat_wall = 12.0
        previous.last_candidate_beat_wall = 12.2
        previous.candidate_scores.extend((0.4, 0.6, 0.8))

        controller = make_controller(args, target_profile, previous=previous)

        self.assertAlmostEqual(0.4, controller.phase)
        self.assertEqual(2, controller.beat_index)
        self.assertEqual(12.0, controller.last_beat_wall)
        self.assertEqual([0.4, 0.6, 0.8], list(controller.candidate_scores))

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
