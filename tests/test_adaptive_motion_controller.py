from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "realtime" / "robot_arm" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from realtime_music_adaptive_player import AdaptiveMotionController, MusicFrame, RealtimeMusicAnalyzer
except ModuleNotFoundError as exc:
    if exc.name != "librosa":
        raise
    AdaptiveMotionController = None
    MusicFrame = None
    RealtimeMusicAnalyzer = None
    _IMPORT_SKIP_REASON = "librosa is not installed"
else:
    _IMPORT_SKIP_REASON = ""


def make_frame(timestamp: float, confidence: float, is_beat: bool = True) -> MusicFrame:
    return MusicFrame(
        timestamp=timestamp,
        rms=0.1,
        gate_rms=0.01,
        rms_norm=0.8,
        onset_strength=0.5,
        brightness=0.4,
        low_energy=0.3,
        mid_energy=0.4,
        high_energy=0.3,
        spectral_centroid=0.4,
        spectral_rolloff=0.4,
        zero_crossing_rate=0.1,
        energy_delta=0.2,
        spectral_contrast=0.0,
        mfcc_1=0.0,
        mfcc_2=0.0,
        rhythm_density=0.3,
        tempo_stability=0.8,
        offbeat_ratio=0.0,
        beat_period=0.5,
        beat_confidence=confidence,
        is_beat=is_beat,
        is_active=True,
    )


def make_controller() -> AdaptiveMotionController:
    return AdaptiveMotionController(
        authored_cycle_duration=2.0,
        beats_per_cycle=4,
        keypoint_phases=(0.0, 0.25, 0.5, 0.75),
        use_keypoints=True,
        smoothing_tau=0.1,
        speed_min=0.5,
        speed_max=2.0,
        amp_min=0.3,
        amp_max=1.0,
        accent_duration=0.1,
        tempo_timeout=1.0,
        beat_confidence_threshold=0.4,
        beat_keypoint_interval_ratio=1.0,
    )


class AdaptiveMotionControllerBeatFilterTests(unittest.TestCase):
    @unittest.skipIf(AdaptiveMotionController is None, _IMPORT_SKIP_REASON)
    def test_rejects_low_confidence_beats(self) -> None:
        controller = make_controller()

        controller.observe(make_frame(timestamp=1.0, confidence=0.39))

        self.assertEqual(0, controller.beat_index)
        self.assertIsNone(controller.last_beat_wall)

    @unittest.skipIf(AdaptiveMotionController is None, _IMPORT_SKIP_REASON)
    def test_rejects_beats_closer_than_keypoint_spacing_at_speed_limit(self) -> None:
        controller = make_controller()

        controller.observe(make_frame(timestamp=1.0, confidence=0.9))
        controller.observe(make_frame(timestamp=1.2, confidence=0.9))
        controller.observe(make_frame(timestamp=1.26, confidence=0.9))

        self.assertEqual(2, controller.beat_index)
        self.assertEqual(1.26, controller.last_beat_wall)


class RealtimeMusicAnalyzerBeatFilterTests(unittest.TestCase):
    @unittest.skipIf(RealtimeMusicAnalyzer is None, _IMPORT_SKIP_REASON)
    def test_minimum_next_beat_gap_tracks_estimated_period(self) -> None:
        analyzer = RealtimeMusicAnalyzer(
            sample_rate=16000,
            block_size=512,
            onset_threshold_scale=3.0,
            min_beat_period=0.25,
            max_beat_period=2.0,
            refractory_sec=0.18,
            noise_gate_rms=0.002,
            noise_gate_ratio=1.8,
            startup_calibration_sec=1.0,
            plp_history_sec=8.0,
            plp_analysis_interval_sec=0.10,
            plp_hop_length=256,
            plp_peak_prominence=0.15,
        )

        analyzer.estimator.add_beat(1.0)
        analyzer.estimator.add_beat(1.5)
        analyzer.estimator.add_beat(2.0)

        self.assertAlmostEqual(0.275, analyzer._minimum_next_beat_gap())


if __name__ == "__main__":
    unittest.main()
