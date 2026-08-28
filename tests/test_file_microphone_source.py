from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "realtime" / "humanoid_robot" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_music_humanoid_dancer import FileMicrophoneSource
from realtime_music_humanoid_matcher import MatcherFileMicrophoneSource


class RecordingAnalyzer:
    def __init__(self, sample_rate: int = 100, block_size: int = 10) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.reset_count = 0
        self.blocks: list[np.ndarray] = []
        self.callback_times: list[float] = []

    def reset(self) -> None:
        self.reset_count += 1

    def _callback(self, data: np.ndarray, frames: int, callback_time: dict, _status: object) -> None:
        self.blocks.append(np.asarray(data[:frames, 0], dtype=np.float32).copy())
        self.callback_times.append(float(callback_time["callback_time"]))

    def drain(self) -> list[object]:
        return []


class RecordingOutputStream:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        self.blocks: list[np.ndarray] = []

    def start(self) -> None:
        self.started = True

    def write(self, block: np.ndarray) -> None:
        self.blocks.append(np.asarray(block, dtype=np.float32).copy())

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FileMicrophoneSourceTests(unittest.TestCase):
    def test_resets_then_feeds_one_second_of_silence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "input.wav"
            audio = np.full(20, 0.5, dtype=np.float32)
            sf.write(audio_path, audio, 100, subtype="FLOAT")
            analyzer = RecordingAnalyzer()
            source = FileMicrophoneSource(audio_path, analyzer, startup_delay_sec=1.0, throttle=False)

            source.start()
            for _ in range(10):
                source.advance(0.1)

            self.assertEqual(analyzer.reset_count, 1)
            self.assertEqual(source.playback_seconds, 0.0)
            self.assertTrue(np.allclose(np.concatenate(analyzer.blocks), 0.0))
            np.testing.assert_allclose(np.diff(analyzer.callback_times), 0.1, atol=1e-6)

            source.advance(0.1)
            self.assertEqual(source.playback_seconds, 0.1)
            self.assertTrue(np.allclose(analyzer.blocks[-1], 0.5))

    def test_recent_audio_excludes_startup_silence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "input.wav"
            audio = np.linspace(-0.5, 0.5, 20, dtype=np.float32)
            sf.write(audio_path, audio, 100, subtype="FLOAT")
            source = FileMicrophoneSource(
                audio_path,
                RecordingAnalyzer(),
                startup_delay_sec=1.0,
                throttle=False,
            )

            source.start()
            for _ in range(12):
                source.advance(0.1)

            recent = source.recent_audio(0.2)
            self.assertIsNotNone(recent)
            self.assertTrue(np.allclose(recent, audio, atol=1e-6))

    def test_play_audio_outputs_the_same_startup_silence_and_audio_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "input.wav"
            audio = np.full(20, 0.5, dtype=np.float32)
            sf.write(audio_path, audio, 100, subtype="FLOAT")
            analyzer = RecordingAnalyzer()
            output = RecordingOutputStream()
            with patch("sounddevice.OutputStream", return_value=output) as output_stream:
                source = FileMicrophoneSource(
                    audio_path,
                    analyzer,
                    startup_delay_sec=0.1,
                    throttle=False,
                    play_audio=True,
                )
                source.start()
                source.advance(0.1)
                source.advance(0.1)
                source.stop()

            output_stream.assert_called_once_with(
                samplerate=100,
                blocksize=10,
                channels=1,
                dtype="float32",
            )
            self.assertTrue(output.started)
            self.assertTrue(output.stopped)
            self.assertTrue(output.closed)
            rendered = np.concatenate([block[:, 0] for block in output.blocks])
            self.assertTrue(np.allclose(rendered[:10], 0.0))
            self.assertTrue(np.allclose(rendered[10:20], 0.5))

    def test_matcher_file_beats_include_relative_contrast(self) -> None:
        sample_rate = 1000
        hop_length = 100
        beat_frames = np.arange(2, 10, dtype=int)
        audio = np.zeros(1200, dtype=np.float32)
        onset = np.zeros(12, dtype=float)
        for index, frame in enumerate(beat_frames):
            amplitude = 1.0 if index % 2 == 0 else 0.2
            center = frame * hop_length
            audio[center - 40 : center + 40] = amplitude
            onset[frame] = amplitude

        args = Namespace(
            mic_sample_rate=sample_rate,
            mic_block_size=100,
            onset_threshold_scale=3.0,
            min_beat_period=0.05,
            max_beat_period=2.0,
            mic_refractory_sec=0.05,
            noise_gate_rms=0.0,
            noise_gate_ratio=1.0,
            startup_calibration_sec=0.0,
            plp_history_sec=2.0,
            plp_analysis_interval_sec=0.1,
            plp_hop_length=hop_length,
            plp_peak_prominence=0.15,
            audio_input_delay_sec=0.0,
            realtime=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "beats.wav"
            sf.write(audio_path, audio, sample_rate, subtype="FLOAT")
            with (
                patch("librosa.onset.onset_strength", return_value=onset),
                patch("librosa.beat.beat_track", return_value=(np.asarray([120.0]), beat_frames)),
            ):
                source = MatcherFileMicrophoneSource(audio_path, args, window_seconds=1.0)

        contrasts = np.asarray(source._pending_beat_contrasts)
        self.assertTrue(np.allclose(contrasts[:3], 0.5))
        self.assertGreater(float(np.mean(contrasts[4::2])), float(np.mean(contrasts[3::2])))


if __name__ == "__main__":
    unittest.main()
