from __future__ import annotations

import sys
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "realtime" / "robot_arm" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import realtime_music_adaptive_player as player_module
    from realtime_music_adaptive_player import (
        AdaptiveMotionController,
        MusicFrame,
        RealtimeMusicAnalyzer,
        compute_beat_contrasts,
    )
except ModuleNotFoundError as exc:
    if exc.name != "librosa":
        raise
    AdaptiveMotionController = None
    MusicFrame = None
    RealtimeMusicAnalyzer = None
    compute_beat_contrasts = None
    _IMPORT_SKIP_REASON = "librosa is not installed"
else:
    _IMPORT_SKIP_REASON = ""


def make_frame(
    timestamp: float,
    confidence: float,
    is_beat: bool = True,
    contrast: float = 0.5,
    beat_period: float | None = 0.5,
) -> MusicFrame:
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
        beat_period=beat_period,
        beat_confidence=confidence,
        beat_contrast=contrast,
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


def make_adaptive_controller(
    *,
    duration: float = 4.0,
    phases: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
    speed_min: float = 0.5,
    speed_max: float = 2.0,
) -> AdaptiveMotionController:
    return AdaptiveMotionController(
        authored_cycle_duration=duration,
        beats_per_cycle=len(phases),
        keypoint_phases=phases,
        use_keypoints=True,
        smoothing_tau=0.1,
        speed_min=speed_min,
        speed_max=speed_max,
        amp_min=0.3,
        amp_max=1.0,
        accent_duration=0.1,
        tempo_timeout=1.0,
        beat_confidence_threshold=0.4,
        beat_keypoint_interval_ratio=1.0,
        beat_selection_mode="adaptive",
        beat_contrast_weight=0.5,
    )


class AdaptiveMotionControllerBeatFilterTests(unittest.TestCase):
    @unittest.skipIf(AdaptiveMotionController is None, _IMPORT_SKIP_REASON)
    def test_beat_alignment_is_slewed_without_phase_snap_and_stays_speed_bounded(self) -> None:
        controller = make_adaptive_controller(
            duration=4.0,
            phases=(0.25, 0.5, 0.75, 0.0),
            speed_min=0.5,
            speed_max=2.0,
        )
        self.assertTrue(controller.observe(make_frame(1.0, 0.9)))
        self.assertEqual(0.0, controller.phase)
        controller.update(1.0)
        phase, *_ = controller.update(1.1)
        self.assertGreater(phase, 0.0)
        maximum_delta = controller.authored_phase_rate * controller.speed_max * 0.1
        self.assertLessEqual(phase, maximum_delta + 1e-12)
        self.assertGreater(controller.phase_correction_remaining, 0.0)

    @unittest.skipIf(AdaptiveMotionController is None, _IMPORT_SKIP_REASON)
    def test_effective_speed_change_is_rate_limited(self) -> None:
        controller = make_adaptive_controller(speed_min=0.5, speed_max=2.0)
        controller.max_speed_change_per_sec = 2.0
        controller.update(1.0)
        controller.target_phase_rate = controller.authored_phase_rate * 2.0
        controller.phase_correction_remaining = 0.4

        previous = controller.speed_multiplier
        controller.update(1.1)

        self.assertLessEqual(controller.speed_multiplier - previous, 0.2 + 1e-12)
        self.assertGreater(controller.phase_correction_remaining, 0.0)

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

    @unittest.skipIf(AdaptiveMotionController is None, _IMPORT_SKIP_REASON)
    def test_keypoint_intervals_include_nonuniform_wraparound(self) -> None:
        controller = make_adaptive_controller(
            duration=4.0,
            phases=(0.1, 0.3, 0.8),
        )

        np.testing.assert_allclose(controller.keypoint_intervals, (1.2, 0.8, 2.0))

    @unittest.skipIf(AdaptiveMotionController is None, _IMPORT_SKIP_REASON)
    def test_within_speed_capacity_accepts_weak_beats(self) -> None:
        controller = make_adaptive_controller(duration=2.0)

        first = controller.observe(make_frame(1.0, 0.8, contrast=0.0, beat_period=0.30))
        second = controller.observe(make_frame(1.3, 0.8, contrast=0.0, beat_period=0.30))

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(2, controller.beat_index)

    @unittest.skipIf(AdaptiveMotionController is None, _IMPORT_SKIP_REASON)
    def test_over_capacity_prefers_strong_eligible_beat(self) -> None:
        controller = make_adaptive_controller()

        self.assertTrue(controller.observe(make_frame(1.0, 0.9, contrast=0.9, beat_period=0.10)))
        for timestamp in (1.1, 1.2, 1.3, 1.4):
            self.assertFalse(
                controller.observe(make_frame(timestamp, 0.9, contrast=0.7, beat_period=0.10))
            )
        self.assertFalse(controller.observe(make_frame(1.5, 0.8, contrast=0.0, beat_period=0.10)))
        self.assertTrue(controller.observe(make_frame(1.6, 0.9, contrast=1.0, beat_period=0.10)))

        self.assertEqual(2, controller.beat_index)
        self.assertEqual(1.6, controller.last_beat_wall)

    @unittest.skipIf(AdaptiveMotionController is None, _IMPORT_SKIP_REASON)
    def test_deadline_fallback_never_accepts_low_confidence_noise(self) -> None:
        controller = make_adaptive_controller()
        self.assertTrue(controller.observe(make_frame(1.0, 0.9, contrast=0.9, beat_period=0.10)))
        controller.candidate_scores = deque((0.9, 0.9, 0.9, 0.9), maxlen=16)

        self.assertFalse(controller.observe(make_frame(1.5, 0.8, contrast=0.0, beat_period=0.10)))
        self.assertFalse(controller.observe(make_frame(3.0, 0.39, contrast=1.0, beat_period=0.10)))
        self.assertTrue(controller.observe(make_frame(3.1, 0.8, contrast=0.0, beat_period=0.10)))

        self.assertEqual(2, controller.beat_index)
        self.assertEqual("accepted", controller.last_beat_rejection_reason)

    @unittest.skipIf(AdaptiveMotionController is None, _IMPORT_SKIP_REASON)
    def test_speed_uses_accepted_interval_and_keeps_raw_bpm(self) -> None:
        controller = make_adaptive_controller(
            duration=4.0,
            phases=(0.0, 0.25, 0.75),
        )

        self.assertTrue(controller.observe(make_frame(1.0, 0.9, beat_period=0.25)))
        self.assertTrue(controller.observe(make_frame(2.0, 0.9, beat_period=0.25)))

        self.assertAlmostEqual(0.25, controller.target_phase_rate)
        self.assertAlmostEqual(1.0, controller.last_accepted_interval)
        self.assertAlmostEqual(240.0, controller.estimated_bpm)


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

    @unittest.skipIf(RealtimeMusicAnalyzer is None, _IMPORT_SKIP_REASON)
    def test_heavy_window_analysis_never_blocks_drain_or_builds_a_backlog(self) -> None:
        analyzer = RealtimeMusicAnalyzer(
            sample_rate=16000,
            block_size=512,
            onset_threshold_scale=3.0,
            min_beat_period=0.25,
            max_beat_period=2.0,
            refractory_sec=0.18,
            noise_gate_rms=0.002,
            noise_gate_ratio=1.8,
            startup_calibration_sec=0.0,
            plp_history_sec=8.0,
            plp_analysis_interval_sec=0.02,
            plp_hop_length=256,
            plp_peak_prominence=0.15,
        )
        analyzer.latest_status_frame = make_frame(1.0, 0.9, is_beat=False)
        analyzer._append_audio_chunk(time.perf_counter() - 1.0, np.ones(16000))

        def slow_analysis(*_args: object) -> None:
            time.sleep(0.05)
            return None

        analyzer.analysis_executor = player_module.ThreadPoolExecutor(max_workers=1)
        with patch.object(player_module, "compute_window_analysis_job", side_effect=slow_analysis):
            started = time.perf_counter()
            analyzer.drain()
            first_drain = time.perf_counter() - started
            analyzer.last_plp_analysis_time = 0.0
            analyzer.drain()
            self.assertLess(first_drain, 0.02)
            self.assertEqual(1, analyzer.analysis_submitted)
            self.assertEqual(1, analyzer.analysis_skipped_busy)
        analyzer.stop()

    @unittest.skipIf(RealtimeMusicAnalyzer is None, _IMPORT_SKIP_REASON)
    def test_reset_discards_completed_analysis_from_an_older_generation(self) -> None:
        analyzer = RealtimeMusicAnalyzer(
            sample_rate=16000,
            block_size=512,
            onset_threshold_scale=3.0,
            min_beat_period=0.25,
            max_beat_period=2.0,
            refractory_sec=0.18,
            noise_gate_rms=0.002,
            noise_gate_ratio=1.8,
            startup_calibration_sec=0.0,
            plp_history_sec=8.0,
            plp_analysis_interval_sec=0.02,
            plp_hop_length=256,
            plp_peak_prominence=0.15,
        )
        analyzer.latest_status_frame = make_frame(1.0, 0.9, is_beat=False)
        analyzer._append_audio_chunk(time.perf_counter() - 1.0, np.ones(16000))
        analyzer.analysis_executor = player_module.ThreadPoolExecutor(max_workers=1)
        with patch.object(player_module, "compute_window_analysis_job", return_value=None):
            analyzer.drain()
            analyzer.reset()
            while analyzer.analysis_future is not None and not analyzer.analysis_future.done():
                time.sleep(0.001)
            analyzer.drain()
        self.assertEqual(0, analyzer.analysis_completed)
        analyzer.stop()

    @unittest.skipIf(compute_beat_contrasts is None, _IMPORT_SKIP_REASON)
    def test_beat_contrast_separates_alternating_strong_and_weak_beats(self) -> None:
        sample_rate = 1000
        hop_length = 100
        beat_frames = np.arange(2, 10, dtype=int)
        audio = np.zeros(1200, dtype=float)
        onset = np.zeros(12, dtype=float)
        for index, frame in enumerate(beat_frames):
            amplitude = 1.0 if index % 2 == 0 else 0.2
            center = frame * hop_length
            audio[center - 40 : center + 40] = amplitude
            onset[frame] = amplitude

        contrasts = compute_beat_contrasts(
            audio,
            onset,
            beat_frames,
            sample_rate,
            hop_length,
        )

        self.assertTrue(np.allclose(contrasts[:3], 0.5))
        self.assertGreater(float(np.mean(contrasts[4::2])), float(np.mean(contrasts[3::2])))

    @unittest.skipIf(compute_beat_contrasts is None, _IMPORT_SKIP_REASON)
    def test_equal_beats_keep_neutral_contrast(self) -> None:
        sample_rate = 1000
        hop_length = 100
        beat_frames = np.arange(2, 10, dtype=int)
        audio = np.zeros(1200, dtype=float)
        onset = np.zeros(12, dtype=float)
        for frame in beat_frames:
            center = frame * hop_length
            audio[center - 40 : center + 40] = 0.6
            onset[frame] = 0.6

        contrasts = compute_beat_contrasts(
            audio,
            onset,
            beat_frames,
            sample_rate,
            hop_length,
        )

        self.assertTrue(np.allclose(contrasts, 0.5))


if __name__ == "__main__":
    unittest.main()
