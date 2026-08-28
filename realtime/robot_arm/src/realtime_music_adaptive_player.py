from __future__ import annotations

import argparse
import csv
import math
import os
import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Lock
from typing import Optional

NUMBA_CACHE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "numba_cache"
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

import librosa
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

p = None
pybullet_data = None


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def wrap_phase_error(delta: float) -> float:
    return ((delta + 0.5) % 1.0) - 0.5


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "beat", "keypoint", "primary"}


def find_pulse_peaks(pulse: np.ndarray, distance: int, prominence: float) -> tuple[np.ndarray, np.ndarray]:
    if pulse.size < 3:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    candidates = np.flatnonzero((pulse[1:-1] > pulse[:-2]) & (pulse[1:-1] >= pulse[2:])) + 1
    if candidates.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    kept: list[tuple[int, float, float]] = []
    radius = max(distance, 1)
    for idx in candidates:
        left = max(0, idx - radius)
        right = min(pulse.size, idx + radius + 1)
        local_floor = max(float(np.min(pulse[left : idx + 1])), float(np.min(pulse[idx:right])))
        peak_prominence = float(pulse[idx] - local_floor)
        if peak_prominence >= prominence:
            kept.append((int(idx), float(pulse[idx]), peak_prominence))

    kept.sort(key=lambda item: item[1], reverse=True)
    selected: list[tuple[int, float]] = []
    for idx, _height, peak_prominence in kept:
        if all(abs(idx - selected_idx) >= distance for selected_idx, _ in selected):
            selected.append((idx, peak_prominence))

    selected.sort(key=lambda item: item[0])
    if not selected:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    return (
        np.asarray([idx for idx, _ in selected], dtype=int),
        np.asarray([peak_prominence for _, peak_prominence in selected], dtype=float),
    )


def _beat_measurement(
    audio: np.ndarray,
    onset_envelope: np.ndarray,
    beat_frame: int,
    sample_rate: int,
    hop_length: int,
    window_sec: float = 0.10,
) -> tuple[float, float]:
    """Measure local loudness and onset impact around one beat frame."""
    center_sample = int(librosa.frames_to_samples(int(beat_frame), hop_length=hop_length))
    radius_samples = max(1, int(round(0.5 * window_sec * sample_rate)))
    start_sample = max(0, center_sample - radius_samples)
    end_sample = min(audio.size, center_sample + radius_samples)
    segment = np.asarray(audio[start_sample:end_sample], dtype=float)
    loudness = float(np.sqrt(np.mean(segment * segment))) if segment.size else 0.0

    radius_frames = max(1, int(round(0.5 * window_sec * sample_rate / hop_length)))
    start_frame = max(0, int(beat_frame) - radius_frames)
    end_frame = min(onset_envelope.size, int(beat_frame) + radius_frames + 1)
    onset_region = np.asarray(onset_envelope[start_frame:end_frame], dtype=float)
    onset_impact = float(np.max(onset_region)) if onset_region.size else 0.0
    return loudness, onset_impact


def _midrank_percentile(value: float, history: deque[float]) -> float:
    values = np.asarray([*history, float(value)], dtype=float)
    if values.size < 4:
        return 0.5
    scale = max(float(np.max(np.abs(values))), 1.0)
    if float(np.ptp(values)) <= 1e-8 * scale:
        return 0.5
    tolerance = 1e-8 * scale
    below = int(np.count_nonzero(values < value - tolerance))
    equal = int(np.count_nonzero(np.abs(values - value) <= tolerance))
    return float(np.clip((below + 0.5 * equal) / values.size, 0.0, 1.0))


def _relative_beat_contrast(
    loudness: float,
    onset_impact: float,
    loudness_history: deque[float],
    onset_history: deque[float],
) -> float:
    loudness_rank = _midrank_percentile(loudness, loudness_history)
    onset_rank = _midrank_percentile(onset_impact, onset_history)
    loudness_history.append(float(loudness))
    onset_history.append(float(onset_impact))
    return float(np.clip(0.5 * loudness_rank + 0.5 * onset_rank, 0.0, 1.0))


def compute_beat_contrasts(
    audio: np.ndarray,
    onset_envelope: np.ndarray,
    beat_frames: np.ndarray,
    sample_rate: int,
    hop_length: int,
    history_size: int = 16,
) -> np.ndarray:
    """Return causal loudness/onset contrast scores for beat-aligned frames."""
    loudness_history: deque[float] = deque(maxlen=max(int(history_size), 1))
    onset_history: deque[float] = deque(maxlen=max(int(history_size), 1))
    contrasts: list[float] = []
    for beat_frame in np.asarray(beat_frames, dtype=int):
        loudness, onset_impact = _beat_measurement(
            np.asarray(audio, dtype=float),
            np.asarray(onset_envelope, dtype=float),
            int(beat_frame),
            int(sample_rate),
            int(hop_length),
        )
        contrasts.append(
            _relative_beat_contrast(
                loudness,
                onset_impact,
                loudness_history,
                onset_history,
            )
        )
    return np.asarray(contrasts, dtype=float)


@dataclass
class Trajectory:
    t: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    phase: np.ndarray
    keypoint_phases: tuple[float, ...]

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 1e-6)


@dataclass
class MusicFrame:
    timestamp: float
    rms: float
    gate_rms: float
    rms_norm: float
    onset_strength: float
    brightness: float
    low_energy: float
    mid_energy: float
    high_energy: float
    spectral_centroid: float
    spectral_rolloff: float
    zero_crossing_rate: float
    energy_delta: float
    spectral_contrast: float
    mfcc_1: float
    mfcc_2: float
    rhythm_density: float
    tempo_stability: float
    offbeat_ratio: float
    beat_period: Optional[float]
    beat_confidence: float
    beat_contrast: float
    is_beat: bool
    is_active: bool


class BeatIntervalEstimator:
    def __init__(
        self,
        min_intervals: int = 3,
        max_intervals: int = 5,
        min_period: float = 0.25,
        max_period: float = 2.0,
    ) -> None:
        self.min_intervals = min_intervals
        self.max_intervals = max_intervals
        self.min_period = min_period
        self.max_period = max_period
        self.beat_times: deque[float] = deque(maxlen=max_intervals + 1)

    def reset(self) -> None:
        self.beat_times.clear()

    def add_beat(self, beat_time: float) -> tuple[bool, Optional[float]]:
        if self.beat_times:
            period = beat_time - self.beat_times[-1]
            if period < self.min_period:
                return False, None
            if period > self.max_period:
                self.beat_times.clear()

        self.beat_times.append(beat_time)
        if len(self.beat_times) < self.min_intervals + 1:
            return True, None

        intervals = np.diff(np.asarray(self.beat_times, dtype=float))
        recent = intervals[-self.max_intervals :]
        valid = recent[(recent >= self.min_period) & (recent <= self.max_period)]
        if valid.size < self.min_intervals:
            return True, None
        return True, float(np.median(valid))


class RealtimeMusicAnalyzer:
    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        onset_threshold_scale: float,
        min_beat_period: float,
        max_beat_period: float,
        refractory_sec: float,
        noise_gate_rms: float,
        noise_gate_ratio: float,
        startup_calibration_sec: float,
        plp_history_sec: float,
        plp_analysis_interval_sec: float,
        plp_hop_length: int,
        plp_peak_prominence: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.onset_threshold_scale = onset_threshold_scale
        self.min_beat_period = min_beat_period
        self.max_beat_period = max_beat_period
        self.refractory_sec = refractory_sec
        self.noise_gate_rms = max(noise_gate_rms, 0.0)
        self.noise_gate_ratio = max(noise_gate_ratio, 1.0)
        self.startup_calibration_sec = max(startup_calibration_sec, 0.0)
        self.plp_history_sec = max(plp_history_sec, 1.0)
        self.plp_analysis_interval_sec = max(plp_analysis_interval_sec, 0.02)
        self.plp_hop_length = max(plp_hop_length, 64)
        self.plp_peak_prominence = max(plp_peak_prominence, 0.0)
        self.frames: SimpleQueue[MusicFrame] = SimpleQueue()
        self.estimator = BeatIntervalEstimator(min_period=min_beat_period, max_period=max_beat_period)

        self.audio_chunks: deque[tuple[float, np.ndarray]] = deque()
        self.audio_lock = Lock()
        self.prev_sample = 0.0
        self.prev_energy = 0.0
        self.rms_floor = 0.0
        self.rms_peak = 1e-6
        self.onset_floor = 0.0
        self.last_beat_time = 0.0
        self.last_plp_analysis_time = 0.0
        self.latest_status_frame: Optional[MusicFrame] = None
        self.inactive_since: Optional[float] = None
        self.activity_release_sec = 0.5
        self.recent_onset_times: deque[float] = deque(maxlen=128)
        self.cached_spectral_contrast = 0.0
        self.cached_mfcc_1 = 0.0
        self.cached_mfcc_2 = 0.0
        self.cached_rhythm_density = 0.0
        self.cached_offbeat_ratio = 0.0
        self.start_wall: Optional[float] = None
        self.calibration_rms_values: deque[float] = deque(maxlen=256)
        self.calibration_onset_values: deque[float] = deque(maxlen=256)
        self.beat_loudness_history: deque[float] = deque(maxlen=16)
        self.beat_onset_history: deque[float] = deque(maxlen=16)
        self.calibrated_noise_rms = 0.0
        self.calibrated_noise_onset_delta = 0.0
        self.stream = None

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed. Install requirements.txt to use realtime music input.")

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def drain(self) -> list[MusicFrame]:
        self._analyze_plp_if_due()
        frames: list[MusicFrame] = []
        while True:
            try:
                frames.append(self.frames.get_nowait())
            except Empty:
                return frames

    def reset(self) -> None:
        self.estimator.reset()
        self.prev_energy = 0.0
        self.last_beat_time = 0.0
        self.last_plp_analysis_time = 0.0
        self.latest_status_frame = None
        self.inactive_since = None
        self.recent_onset_times.clear()
        self.cached_spectral_contrast = 0.0
        self.cached_mfcc_1 = 0.0
        self.cached_mfcc_2 = 0.0
        self.cached_rhythm_density = 0.0
        self.cached_offbeat_ratio = 0.0
        self.rms_floor = 0.0
        self.rms_peak = 1e-6
        self.onset_floor = 0.0
        self.start_wall = time.perf_counter()
        with self.audio_lock:
            self.audio_chunks.clear()
        self.calibration_rms_values.clear()
        self.calibration_onset_values.clear()
        self.beat_loudness_history.clear()
        self.beat_onset_history.clear()
        self.calibrated_noise_rms = 0.0
        self.calibrated_noise_onset_delta = 0.0

    def _callback(self, indata: np.ndarray, frames: int, time_info: dict, status: object) -> None:
        del frames, status

        mono = np.asarray(indata[:, 0], dtype=np.float64)
        now = time.perf_counter()
        if isinstance(time_info, dict) and "callback_time" in time_info:
            now = float(time_info["callback_time"])
        if self.start_wall is None:
            self.start_wall = now
        chunk_start = now - (mono.size / self.sample_rate)
        self._append_audio_chunk(chunk_start, mono)

        rms = float(np.sqrt(np.mean(mono * mono)))
        highpass = np.diff(mono, prepend=self.prev_sample)
        self.prev_sample = float(mono[-1])
        energy = float(np.sqrt(np.mean(highpass * highpass)))

        onset_delta = max(energy - self.prev_energy, 0.0)
        calibrating = (now - self.start_wall) < self.startup_calibration_sec
        if calibrating:
            self.calibration_rms_values.append(rms)
            self.calibration_onset_values.append(onset_delta)
        elif self.calibrated_noise_rms <= 0.0:
            if self.calibration_rms_values:
                rms_values = np.asarray(self.calibration_rms_values, dtype=float)
                self.calibrated_noise_rms = float(np.percentile(rms_values, 90.0))
            if self.calibration_onset_values:
                onset_values = np.asarray(self.calibration_onset_values, dtype=float)
                self.calibrated_noise_onset_delta = float(np.percentile(onset_values, 90.0))
            self.rms_floor = self.calibrated_noise_rms
            self.onset_floor = self.calibrated_noise_onset_delta

        noise_reference = self.calibrated_noise_rms
        active_threshold = max(self.noise_gate_rms, noise_reference * self.noise_gate_ratio)
        raw_active = (not calibrating) and rms >= active_threshold
        if raw_active:
            self.inactive_since = None
        elif self.inactive_since is None:
            self.inactive_since = now
        recently_active = (
            not calibrating
            and self.latest_status_frame is not None
            and self.latest_status_frame.is_active
            and self.inactive_since is not None
            and (now - self.inactive_since) < self.activity_release_sec
        )
        is_active = raw_active or recently_active

        if rms > self.rms_peak:
            self.rms_peak = rms
        else:
            self.rms_peak = max(active_threshold + 1e-6, 0.999 * self.rms_peak)
        rms_norm = float(np.clip((rms - active_threshold) / max(self.rms_peak - active_threshold, 1e-6), 0.0, 1.0))
        if not is_active:
            rms_norm = 0.0
            self.estimator.reset()
            self.last_beat_time = 0.0
            self.beat_loudness_history.clear()
            self.beat_onset_history.clear()

        onset_reference = max(self.onset_floor, self.calibrated_noise_onset_delta)
        threshold = max(1e-5, self.onset_threshold_scale * onset_reference)
        onset_strength = float(onset_delta / threshold) if is_active and threshold > 0 else 0.0

        features = self._block_features(mono, rms, onset_delta) if is_active else self._empty_block_features()
        if is_active and onset_delta > threshold:
            self.recent_onset_times.append(now)

        tempo_stability = self._tempo_stability()
        is_beat = False
        beat_period = None
        beat_confidence = 0.0

        self.prev_energy = energy
        self.frames.put(
            MusicFrame(
                timestamp=now,
                rms=rms,
                gate_rms=active_threshold,
                rms_norm=rms_norm,
                onset_strength=onset_strength,
                brightness=features["brightness"],
                low_energy=features["low_energy"],
                mid_energy=features["mid_energy"],
                high_energy=features["high_energy"],
                spectral_centroid=features["spectral_centroid"],
                spectral_rolloff=features["spectral_rolloff"],
                zero_crossing_rate=features["zero_crossing_rate"],
                energy_delta=features["energy_delta"],
                spectral_contrast=self.cached_spectral_contrast if is_active else 0.0,
                mfcc_1=self.cached_mfcc_1 if is_active else 0.0,
                mfcc_2=self.cached_mfcc_2 if is_active else 0.0,
                rhythm_density=self.cached_rhythm_density if is_active else 0.0,
                tempo_stability=tempo_stability if is_active else 0.0,
                offbeat_ratio=self.cached_offbeat_ratio if is_active else 0.0,
                beat_period=beat_period,
                beat_confidence=beat_confidence,
                beat_contrast=0.0,
                is_beat=is_beat,
                is_active=is_active,
            )
        )
        self.latest_status_frame = MusicFrame(
            timestamp=now,
            rms=rms,
            gate_rms=active_threshold,
            rms_norm=rms_norm,
            onset_strength=onset_strength,
            brightness=features["brightness"],
            low_energy=features["low_energy"],
            mid_energy=features["mid_energy"],
            high_energy=features["high_energy"],
            spectral_centroid=features["spectral_centroid"],
            spectral_rolloff=features["spectral_rolloff"],
            zero_crossing_rate=features["zero_crossing_rate"],
            energy_delta=features["energy_delta"],
            spectral_contrast=self.cached_spectral_contrast if is_active else 0.0,
            mfcc_1=self.cached_mfcc_1 if is_active else 0.0,
            mfcc_2=self.cached_mfcc_2 if is_active else 0.0,
            rhythm_density=self.cached_rhythm_density if is_active else 0.0,
            tempo_stability=tempo_stability if is_active else 0.0,
            offbeat_ratio=self.cached_offbeat_ratio if is_active else 0.0,
            beat_period=None,
            beat_confidence=0.0,
            beat_contrast=0.0,
            is_beat=False,
            is_active=is_active,
        )

    def _append_audio_chunk(self, chunk_start: float, mono: np.ndarray) -> None:
        keep_after = chunk_start - self.plp_history_sec - 1.0
        with self.audio_lock:
            self.audio_chunks.append((chunk_start, mono.copy()))
            while self.audio_chunks and self.audio_chunks[0][0] < keep_after:
                self.audio_chunks.popleft()

    def _audio_window(self) -> tuple[Optional[float], np.ndarray]:
        with self.audio_lock:
            if not self.audio_chunks:
                return None, np.empty(0, dtype=np.float64)
            chunks = list(self.audio_chunks)

        end_time = chunks[-1][0] + chunks[-1][1].size / self.sample_rate
        start_time = max(chunks[0][0], end_time - self.plp_history_sec)
        pieces: list[np.ndarray] = []
        for chunk_start, chunk in chunks:
            chunk_end = chunk_start + chunk.size / self.sample_rate
            if chunk_end <= start_time:
                continue
            offset = max(0, int(round((start_time - chunk_start) * self.sample_rate)))
            pieces.append(chunk[offset:])
        if not pieces:
            return None, np.empty(0, dtype=np.float64)
        return start_time, np.concatenate(pieces)

    def _analyze_plp_if_due(self) -> None:
        now = time.perf_counter()
        if now - self.last_plp_analysis_time < self.plp_analysis_interval_sec:
            return
        self.last_plp_analysis_time = now

        status = self.latest_status_frame
        if status is None or not status.is_active:
            return

        window_start, audio = self._audio_window()
        min_samples = max(self.sample_rate, self.plp_hop_length * 8)
        if window_start is None or audio.size < min_samples:
            return

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="n_fft=.*too large.*")
                onset_env = librosa.onset.onset_strength(
                    y=audio,
                    sr=self.sample_rate,
                    hop_length=self.plp_hop_length,
                )
        except Exception:
            return

        self._update_window_features(audio, onset_env, window_start)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="n_fft=.*too large.*")
                pulse = librosa.beat.plp(
                    onset_envelope=onset_env,
                    sr=self.sample_rate,
                    hop_length=self.plp_hop_length,
                    tempo_min=60.0 / max(self.max_beat_period, 1e-6),
                    tempo_max=60.0 / max(self.min_beat_period, 1e-6),
                )
        except Exception:
            return

        if pulse.size == 0 or float(np.max(pulse)) <= 1e-9:
            return

        pulse = np.asarray(pulse, dtype=float)
        pulse_norm = pulse / max(float(np.max(pulse)), 1e-9)
        distance = max(1, int(round(self.refractory_sec * self.sample_rate / self.plp_hop_length)))
        peaks, prominences = find_pulse_peaks(
            pulse_norm,
            distance=distance,
            prominence=self.plp_peak_prominence,
        )
        if peaks.size == 0:
            return

        peak_times = window_start + librosa.frames_to_time(
            peaks,
            sr=self.sample_rate,
            hop_length=self.plp_hop_length,
        )
        min_next_gap = self._minimum_next_beat_gap()
        recent_mask = (peak_times > self.last_beat_time + min_next_gap) & (
            peak_times >= now - max(0.35, 2.0 * self.plp_analysis_interval_sec)
        )
        if not np.any(recent_mask):
            return

        peak_idx = int(np.flatnonzero(recent_mask)[0])
        beat_time = float(peak_times[peak_idx])
        accepted, beat_period = self.estimator.add_beat(beat_time)
        if not accepted:
            return

        self.last_beat_time = beat_time
        prominence = float(prominences[peak_idx]) if len(prominences) > peak_idx else 0.0
        confidence = float(np.clip(0.5 * pulse_norm[peaks[peak_idx]] + 0.5 * prominence, 0.0, 1.0))
        loudness, onset_impact = _beat_measurement(
            audio,
            onset_env,
            int(peaks[peak_idx]),
            self.sample_rate,
            self.plp_hop_length,
        )
        contrast = _relative_beat_contrast(
            loudness,
            onset_impact,
            self.beat_loudness_history,
            self.beat_onset_history,
        )
        self.frames.put(
            MusicFrame(
                timestamp=beat_time,
                rms=status.rms,
                gate_rms=status.gate_rms,
                rms_norm=status.rms_norm,
                onset_strength=max(status.onset_strength, confidence),
                brightness=status.brightness,
                low_energy=status.low_energy,
                mid_energy=status.mid_energy,
                high_energy=status.high_energy,
                spectral_centroid=status.spectral_centroid,
                spectral_rolloff=status.spectral_rolloff,
                zero_crossing_rate=status.zero_crossing_rate,
                energy_delta=status.energy_delta,
                spectral_contrast=self.cached_spectral_contrast,
                mfcc_1=self.cached_mfcc_1,
                mfcc_2=self.cached_mfcc_2,
                rhythm_density=self.cached_rhythm_density,
                tempo_stability=self._tempo_stability(),
                offbeat_ratio=self.cached_offbeat_ratio,
                beat_period=beat_period,
                beat_confidence=confidence,
                beat_contrast=contrast,
                is_beat=True,
                is_active=True,
            )
        )

    def _minimum_next_beat_gap(self) -> float:
        gap = self.refractory_sec
        if len(self.estimator.beat_times) >= 2:
            intervals = np.diff(np.asarray(self.estimator.beat_times, dtype=float))
            valid = intervals[(intervals >= self.min_beat_period) & (intervals <= self.max_beat_period)]
            if valid.size:
                gap = max(gap, 0.55 * float(np.median(valid)))
        return float(min(max(gap, 0.0), self.max_beat_period))

    @staticmethod
    def _empty_block_features() -> dict[str, float]:
        return {
            "brightness": 0.0,
            "low_energy": 0.0,
            "mid_energy": 0.0,
            "high_energy": 0.0,
            "spectral_centroid": 0.0,
            "spectral_rolloff": 0.0,
            "zero_crossing_rate": 0.0,
            "energy_delta": 0.0,
        }

    def _block_features(self, mono: np.ndarray, rms: float, onset_delta: float) -> dict[str, float]:
        if rms < 1e-5 or mono.size < 8:
            features = self._empty_block_features()
            features["energy_delta"] = float(onset_delta)
            return features

        windowed = mono * np.hanning(mono.size)
        spectrum = np.abs(np.fft.rfft(windowed))
        total = float(np.sum(spectrum))
        if total <= 1e-9:
            features = self._empty_block_features()
            features["energy_delta"] = float(onset_delta)
            return features

        freqs = np.fft.rfftfreq(mono.size, d=1.0 / self.sample_rate)
        centroid = float(np.sum(freqs * spectrum) / total)
        nyquist = max(0.5 * self.sample_rate, 1e-6)

        cumulative = np.cumsum(spectrum)
        rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1], side="left"))
        rolloff_hz = float(freqs[min(rolloff_idx, freqs.size - 1)])

        power = spectrum * spectrum
        power_total = float(np.sum(power))
        if power_total <= 1e-12:
            low_energy = mid_energy = high_energy = 0.0
        else:
            low_energy = float(np.sum(power[(freqs >= 20.0) & (freqs < 250.0)]) / power_total)
            mid_energy = float(np.sum(power[(freqs >= 250.0) & (freqs < 4000.0)]) / power_total)
            high_energy = float(np.sum(power[freqs >= 4000.0]) / power_total)

        zero_crossings = np.count_nonzero(np.diff(np.signbit(mono)))
        zcr = float(zero_crossings / max(mono.size - 1, 1))

        return {
            "brightness": float(np.clip(centroid / nyquist, 0.0, 1.0)),
            "low_energy": float(np.clip(low_energy, 0.0, 1.0)),
            "mid_energy": float(np.clip(mid_energy, 0.0, 1.0)),
            "high_energy": float(np.clip(high_energy, 0.0, 1.0)),
            "spectral_centroid": float(np.clip(centroid / nyquist, 0.0, 1.0)),
            "spectral_rolloff": float(np.clip(rolloff_hz / nyquist, 0.0, 1.0)),
            "zero_crossing_rate": float(np.clip(zcr, 0.0, 1.0)),
            "energy_delta": float(max(onset_delta, 0.0)),
        }

    def _update_window_features(self, audio: np.ndarray, onset_env: np.ndarray, window_start: float) -> None:
        if audio.size < self.plp_hop_length * 8 or onset_env.size == 0:
            return

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="n_fft=.*too large.*")
                contrast = librosa.feature.spectral_contrast(
                    y=audio,
                    sr=self.sample_rate,
                    hop_length=self.plp_hop_length,
                )
                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=self.sample_rate,
                    n_mfcc=2,
                    hop_length=self.plp_hop_length,
                )
        except Exception:
            return

        self.cached_spectral_contrast = self._finite_float(np.mean(contrast) / 80.0, 0.0, 1.0)
        self.cached_mfcc_1 = self._finite_float(np.mean(mfcc[0]) / 100.0, -2.0, 2.0) if mfcc.shape[0] > 0 else 0.0
        self.cached_mfcc_2 = self._finite_float(np.mean(mfcc[1]) / 100.0, -2.0, 2.0) if mfcc.shape[0] > 1 else 0.0

        onset = np.asarray(onset_env, dtype=float)
        onset_max = float(np.max(onset)) if onset.size else 0.0
        if onset.size < 3 or onset_max <= 1e-9:
            self.cached_rhythm_density = 0.0
            self.cached_offbeat_ratio = 0.0
            return

        onset_norm = onset / onset_max
        threshold = max(0.2, float(np.median(onset_norm) + 0.5 * np.std(onset_norm)))
        onset_peaks, _prom = find_pulse_peaks(
            onset_norm,
            distance=max(1, int(round(0.08 * self.sample_rate / self.plp_hop_length))),
            prominence=threshold,
        )
        duration = max(audio.size / self.sample_rate, 1e-6)
        self.cached_rhythm_density = float(np.clip((onset_peaks.size / duration) / 8.0, 0.0, 1.0))
        self.cached_offbeat_ratio = self._offbeat_ratio(window_start, onset_peaks)

    def _offbeat_ratio(self, window_start: float, onset_peaks: np.ndarray) -> float:
        if onset_peaks.size == 0 or len(self.estimator.beat_times) < 2:
            return 0.0

        beat_times = np.asarray(self.estimator.beat_times, dtype=float)
        onset_times = window_start + librosa.frames_to_time(
            onset_peaks,
            sr=self.sample_rate,
            hop_length=self.plp_hop_length,
        )
        counted = 0
        offbeat = 0
        for onset_time in onset_times:
            idx = int(np.searchsorted(beat_times, onset_time, side="right") - 1)
            if idx < 0 or idx >= beat_times.size - 1:
                continue
            period = beat_times[idx + 1] - beat_times[idx]
            if period <= 1e-6:
                continue
            phase = (onset_time - beat_times[idx]) / period
            counted += 1
            if 0.25 <= phase <= 0.75:
                offbeat += 1
        if counted == 0:
            return 0.0
        return float(np.clip(offbeat / counted, 0.0, 1.0))

    def _tempo_stability(self) -> float:
        if len(self.estimator.beat_times) < 4:
            return 0.0
        intervals = np.diff(np.asarray(self.estimator.beat_times, dtype=float))
        valid = intervals[(intervals >= self.min_beat_period) & (intervals <= self.max_beat_period)]
        if valid.size < 3:
            return 0.0
        median = float(np.median(valid))
        if median <= 1e-6:
            return 0.0
        variation = float(np.std(valid) / median)
        return float(np.clip(1.0 - variation * 4.0, 0.0, 1.0))

    @staticmethod
    def _finite_float(value: float, lower: float, upper: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, lower, upper))


class AdaptiveMotionController:
    def __init__(
        self,
        authored_cycle_duration: float,
        beats_per_cycle: int,
        keypoint_phases: tuple[float, ...],
        use_keypoints: bool,
        smoothing_tau: float,
        speed_min: float,
        speed_max: float,
        amp_min: float,
        amp_max: float,
        accent_duration: float,
        tempo_timeout: float,
        beat_confidence_threshold: float = 0.25,
        beat_keypoint_interval_ratio: float = 0.85,
        beat_selection_mode: str = "every",
        beat_contrast_weight: float = 0.5,
        phase_correction_tau: float = 0.25,
    ) -> None:
        self.authored_cycle_duration = max(authored_cycle_duration, 1e-6)
        self.authored_phase_rate = 1.0 / max(authored_cycle_duration, 1e-6)
        self.default_phase_rate = self.authored_phase_rate
        self.beats_per_cycle = max(beats_per_cycle, 1)
        self.keypoint_phases = keypoint_phases if use_keypoints else ()
        self.smoothing_tau = max(smoothing_tau, 1e-6)
        self.speed_min = max(speed_min, 1e-3)
        self.speed_max = max(speed_max, self.speed_min)
        self.amp_min = max(amp_min, 0.0)
        self.amp_max = max(amp_max, self.amp_min)
        self.accent_duration = max(accent_duration, 1e-6)
        self.tempo_timeout = max(tempo_timeout, 0.0)
        self.beat_confidence_threshold = float(np.clip(beat_confidence_threshold, 0.0, 1.0))
        self.beat_keypoint_interval_ratio = max(beat_keypoint_interval_ratio, 0.0)
        if beat_selection_mode not in {"every", "adaptive"}:
            raise ValueError("beat_selection_mode must be 'every' or 'adaptive'")
        self.beat_selection_mode = beat_selection_mode
        self.beat_contrast_weight = float(np.clip(beat_contrast_weight, 0.0, 1.0))
        self.phase_correction_tau = max(float(phase_correction_tau), 1e-6)
        self.keypoint_intervals = self._keypoint_intervals()
        self.min_accepted_beat_interval = self._min_accepted_beat_interval()

        self.phase = 0.0
        self.phase_rate = self.default_phase_rate
        self.effective_phase_rate = self.default_phase_rate
        self.target_phase_rate = self.default_phase_rate
        self.phase_correction_remaining = 0.0
        self.amplitude_scale = 0.0
        self.target_amplitude_scale = 0.0
        self.last_update_wall: Optional[float] = None
        self.last_beat_wall: Optional[float] = None
        self.last_candidate_beat_wall: Optional[float] = None
        self.beat_index = 0
        self.last_period: Optional[float] = None
        self.last_accepted_interval: Optional[float] = None
        self.last_brightness = 0.0
        self.accent_started_wall: Optional[float] = None
        self.accent_strength = 0.0
        self.music_active = False
        self.candidate_scores: deque[float] = deque(maxlen=16)
        self.last_selection_score = 0.0
        self.last_beat_accepted = False
        self.last_beat_rejection_reason = "none"

    def _keypoint_intervals(self) -> tuple[float, ...]:
        if self.keypoint_phases:
            phases = tuple(float(phase) % 1.0 for phase in self.keypoint_phases)
            if len(phases) == 1:
                return (self.authored_cycle_duration,)
            intervals = []
            for index, phase in enumerate(phases):
                previous = phases[index - 1]
                phase_gap = (phase - previous) % 1.0
                intervals.append(max(phase_gap * self.authored_cycle_duration, 1e-6))
            return tuple(intervals)
        uniform = self.authored_cycle_duration / self.beats_per_cycle
        return tuple(uniform for _ in range(self.beats_per_cycle))

    def _min_accepted_beat_interval(self) -> float:
        fastest_authored_gap = min(self.keypoint_intervals) / self.speed_max
        return max(0.0, fastest_authored_gap * self.beat_keypoint_interval_ratio)

    def reset(self) -> None:
        self.phase = 0.0
        self.phase_rate = self.default_phase_rate
        self.effective_phase_rate = self.default_phase_rate
        self.target_phase_rate = self.default_phase_rate
        self.phase_correction_remaining = 0.0
        self.amplitude_scale = 0.0
        self.target_amplitude_scale = 0.0
        self.last_update_wall = None
        self.last_beat_wall = None
        self.last_candidate_beat_wall = None
        self.beat_index = 0
        self.last_period = None
        self.last_accepted_interval = None
        self.last_brightness = 0.0
        self.accent_started_wall = None
        self.accent_strength = 0.0
        self.music_active = False
        self.candidate_scores.clear()
        self.last_selection_score = 0.0
        self.last_beat_accepted = False
        self.last_beat_rejection_reason = "none"

    def observe(self, frame: MusicFrame) -> bool:
        self.last_beat_accepted = False
        self.music_active = frame.is_active
        if not frame.is_active:
            self.target_amplitude_scale = 0.0
            self.last_brightness = 0.0
            self.accent_started_wall = None
            self.accent_strength = 0.0
            return False

        self.target_amplitude_scale = float(
            np.clip(self.amp_min + frame.rms_norm * (self.amp_max - self.amp_min), self.amp_min, self.amp_max)
        )
        self.last_brightness = frame.brightness

        if not frame.is_beat:
            return False

        candidate_period = frame.beat_period
        if candidate_period is None and self.last_candidate_beat_wall is not None:
            measured = frame.timestamp - self.last_candidate_beat_wall
            if measured > 0.0:
                candidate_period = measured
        self.last_candidate_beat_wall = frame.timestamp
        if frame.beat_period is not None:
            self.last_period = frame.beat_period

        confidence = float(np.clip(frame.beat_confidence, 0.0, 1.0))
        contrast = float(np.clip(frame.beat_contrast, 0.0, 1.0))
        selection_score = float(
            (1.0 - self.beat_contrast_weight) * confidence
            + self.beat_contrast_weight * contrast
        )
        self.last_selection_score = selection_score

        if frame.beat_confidence < self.beat_confidence_threshold:
            self.last_beat_rejection_reason = "low-confidence"
            return False

        previous_beat_wall = self.last_beat_wall
        target_interval = self._target_keypoint_interval()
        if self.beat_selection_mode == "adaptive":
            accepted, reason = self._accept_adaptive_beat(
                frame.timestamp,
                candidate_period,
                target_interval,
                selection_score,
            )
        else:
            accepted = not (
                previous_beat_wall is not None
                and frame.timestamp - previous_beat_wall < self.min_accepted_beat_interval
            )
            reason = "accepted" if accepted else "too-early"
        self.candidate_scores.append(selection_score)
        if not accepted:
            self.last_beat_rejection_reason = reason
            return False

        self.last_beat_wall = frame.timestamp
        expected_phase = self._expected_beat_phase()
        beat_alignment_error = wrap_phase_error(expected_phase - self.phase)
        correction_source = selection_score if self.beat_selection_mode == "adaptive" else confidence
        correction_gain = 1.0 if self.beat_index == 0 else 0.25 + 0.35 * correction_source
        self.phase_correction_remaining = wrap_phase_error(
            self.phase_correction_remaining + correction_gain * beat_alignment_error
        )
        self.beat_index += 1

        if self.beat_selection_mode == "adaptive" and previous_beat_wall is not None:
            accepted_interval = frame.timestamp - previous_beat_wall
            self.last_accepted_interval = accepted_interval
            phase_gap = target_interval / self.authored_cycle_duration
            unclamped_rate = phase_gap / max(accepted_interval, 1e-6)
            min_rate = self.authored_phase_rate * self.speed_min
            max_rate = self.authored_phase_rate * self.speed_max
            self.target_phase_rate = float(np.clip(unclamped_rate, min_rate, max_rate))
        elif frame.beat_period is not None:
            unclamped_rate = 1.0 / max(frame.beat_period * self.beats_per_cycle, 1e-6)
            min_rate = self.authored_phase_rate * self.speed_min
            max_rate = self.authored_phase_rate * self.speed_max
            self.target_phase_rate = float(np.clip(unclamped_rate, min_rate, max_rate))

        self.accent_started_wall = frame.timestamp
        accent_score = selection_score if self.beat_selection_mode == "adaptive" else confidence
        self.accent_strength = float(np.clip(0.45 + accent_score + 0.25 * frame.onset_strength, 0.0, 1.5))
        self.last_beat_accepted = True
        self.last_beat_rejection_reason = "accepted"
        return True

    def _target_keypoint_interval(self) -> float:
        return self.keypoint_intervals[self.beat_index % len(self.keypoint_intervals)]

    def _accept_adaptive_beat(
        self,
        timestamp: float,
        candidate_period: Optional[float],
        target_interval: float,
        selection_score: float,
    ) -> tuple[bool, str]:
        if self.last_beat_wall is None:
            return True, "accepted"

        elapsed = timestamp - self.last_beat_wall
        latest = target_interval / self.speed_min
        earliest = min(
            (target_interval / self.speed_max) * self.beat_keypoint_interval_ratio,
            latest,
        )
        if elapsed < earliest:
            return False, "too-early"

        if candidate_period is None or candidate_period >= earliest:
            return True, "accepted"

        if elapsed >= latest:
            return True, "deadline"
        if len(self.candidate_scores) < 4:
            return True, "startup"

        overload_quantile = float(np.clip(1.0 - candidate_period / max(earliest, 1e-6), 0.0, 0.90))
        base_threshold = float(np.quantile(np.asarray(self.candidate_scores, dtype=float), overload_quantile))
        available_window = max(latest - earliest, 1e-6)
        deadline_progress = float(np.clip((elapsed - earliest) / available_window, 0.0, 1.0))
        threshold = (1.0 - deadline_progress) * base_threshold
        if selection_score < threshold:
            return False, "weak-overload-beat"
        return True, "accepted"

    def update(self, now_wall: float) -> tuple[float, float, float, float]:
        if self.last_update_wall is None:
            self.last_update_wall = now_wall
            return self.phase, self.amplitude_scale, 0.0, self.last_brightness

        dt = max(now_wall - self.last_update_wall, 0.0)
        self.last_update_wall = now_wall

        tempo_reference = (
            self.last_candidate_beat_wall
            if self.beat_selection_mode == "adaptive"
            else self.last_beat_wall
        )
        if tempo_reference is not None and (now_wall - tempo_reference) > self.tempo_timeout:
            self.target_phase_rate = self.default_phase_rate
            if not self.music_active:
                self.target_amplitude_scale = 0.0
                self.last_period = None
                self.last_beat_wall = None
                self.last_candidate_beat_wall = None
                self.last_accepted_interval = None
                self.candidate_scores.clear()

        alpha = 1.0 - math.exp(-dt / self.smoothing_tau)
        self.phase_rate += alpha * (self.target_phase_rate - self.phase_rate)
        minimum_rate = self.authored_phase_rate * self.speed_min
        maximum_rate = self.authored_phase_rate * self.speed_max
        self.phase_rate = float(np.clip(self.phase_rate, minimum_rate, maximum_rate))
        self.amplitude_scale += alpha * (self.target_amplitude_scale - self.amplitude_scale)
        base_delta = self.phase_rate * dt
        correction_alpha = 1.0 - math.exp(-dt / self.phase_correction_tau)
        desired_correction = self.phase_correction_remaining * correction_alpha
        correction = float(
            np.clip(
                desired_correction,
                (minimum_rate - self.phase_rate) * dt,
                (maximum_rate - self.phase_rate) * dt,
            )
        )
        self.phase_correction_remaining -= correction
        total_delta = base_delta + correction
        self.effective_phase_rate = self.phase_rate if dt <= 0.0 else total_delta / dt
        self.phase = (self.phase + total_delta) % 1.0
        return self.phase, self.amplitude_scale, self._accent(now_wall), self.last_brightness

    def _expected_beat_phase(self) -> float:
        if self.keypoint_phases:
            return self.keypoint_phases[self.beat_index % len(self.keypoint_phases)]
        return (self.beat_index % self.beats_per_cycle) / self.beats_per_cycle

    def _accent(self, now_wall: float) -> float:
        if self.accent_started_wall is None:
            return 0.0
        age = now_wall - self.accent_started_wall
        if age < 0.0 or age > self.accent_duration:
            return 0.0
        env = math.sin(math.pi * age / self.accent_duration)
        return self.accent_strength * env

    @property
    def speed_multiplier(self) -> float:
        return self.effective_phase_rate / max(self.authored_phase_rate, 1e-6)

    @property
    def estimated_bpm(self) -> Optional[float]:
        if self.last_period is None:
            return None
        return 60.0 / self.last_period


class TrajectorySampler:
    def __init__(
        self,
        trajectory: Trajectory,
        motion: str,
        neutral_yaw: Optional[float],
        neutral_pitch: Optional[float],
        accent_gain_deg: float,
        clap_yaw_amp_deg: float,
        clap_open_pitch_deg: float,
        clap_pitch_deg: float,
        clap_bob_deg: float,
        figure_yaw_amp_deg: float,
        figure_pitch_amp_deg: float,
    ) -> None:
        self.trajectory = trajectory
        self.motion = motion
        default_neutral_yaw = 0.0 if motion in {"clap", "figure-eight"} else float(np.mean(trajectory.yaw))
        default_neutral_pitch = math.radians(clap_open_pitch_deg) if motion == "clap" else float(np.mean(trajectory.pitch))
        self.neutral_yaw = default_neutral_yaw if neutral_yaw is None else neutral_yaw
        self.neutral_pitch = default_neutral_pitch if neutral_pitch is None else neutral_pitch
        self.accent_gain_rad = math.radians(accent_gain_deg)
        self.clap_yaw_amp_rad = math.radians(clap_yaw_amp_deg)
        self.clap_open_pitch_rad = math.radians(clap_open_pitch_deg)
        self.clap_pitch_rad = math.radians(clap_pitch_deg)
        self.clap_bob_rad = math.radians(clap_bob_deg)
        self.figure_yaw_amp_rad = math.radians(figure_yaw_amp_deg)
        self.figure_pitch_amp_rad = math.radians(figure_pitch_amp_deg)

    def sample(self, phase: float, amplitude_scale: float, accent: float, brightness: float) -> tuple[float, float]:
        phase = phase % 1.0
        if self.motion == "clap":
            yaw_base, pitch_base = self._sample_clap_base(phase)
        elif self.motion == "figure-eight":
            yaw_base, pitch_base = self._sample_figure_eight_base(phase)
        else:
            yaw_base = self._interp_loop(phase, self.trajectory.phase, self.trajectory.yaw)
            pitch_base = self._interp_loop(phase, self.trajectory.phase, self.trajectory.pitch)

        yaw = self.neutral_yaw + (yaw_base - self.neutral_yaw) * amplitude_scale
        pitch = self.neutral_pitch + (pitch_base - self.neutral_pitch) * amplitude_scale

        brightness_scale = 0.75 + 0.5 * brightness
        accent_rad = self.accent_gain_rad * accent * brightness_scale
        yaw += 0.25 * accent_rad
        pitch += accent_rad
        return yaw, pitch

    def _sample_clap_base(self, phase: float) -> tuple[float, float]:
        close_env = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
        yaw = self.neutral_yaw + self.clap_yaw_amp_rad * math.sin(2.0 * math.pi * phase)
        pitch = self.clap_open_pitch_rad + (self.clap_pitch_rad - self.clap_open_pitch_rad) * close_env
        pitch += self.clap_bob_rad * math.sin(4.0 * math.pi * phase)
        return yaw, pitch

    def _sample_figure_eight_base(self, phase: float) -> tuple[float, float]:
        yaw = self.neutral_yaw + self.figure_yaw_amp_rad * math.sin(2.0 * math.pi * phase)
        pitch = self.neutral_pitch + self.figure_pitch_amp_rad * math.sin(4.0 * math.pi * phase + math.pi / 2.0)
        return yaw, pitch

    @staticmethod
    def _interp_loop(phase: float, phase_track: np.ndarray, values: np.ndarray) -> float:
        phases = phase_track
        vals = values
        if phases[0] > 0.0 or phases[-1] < 1.0:
            phases = np.concatenate(([0.0], phases, [1.0]))
            vals = np.concatenate(([values[-1]], values, [values[0]]))
        return float(np.interp(phase, phases, vals))


class PyBulletRobotPlayer:
    def __init__(
        self,
        urdf: str,
        joint_yaw: int,
        joint_pitch: int,
        fixed_base: bool,
        sim_dt: float,
    ) -> None:
        self.urdf = urdf
        self.joint_yaw = joint_yaw
        self.joint_pitch = joint_pitch
        self.fixed_base = fixed_base
        self.sim_dt = sim_dt
        self.client: Optional[int] = None
        self.robot_id: Optional[int] = None

    def start(self, initial_yaw: float, initial_pitch: float) -> None:
        global p, pybullet_data
        if p is None or pybullet_data is None:
            import pybullet as bullet
            import pybullet_data as bullet_data

            p = bullet
            pybullet_data = bullet_data

        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf, useFixedBase=self.fixed_base)

        n_joints = p.getNumJoints(self.robot_id)
        if self.joint_yaw >= n_joints or self.joint_pitch >= n_joints:
            raise IndexError(
                f"Joint index out of range. Robot has {n_joints} joints, "
                f"got yaw={self.joint_yaw}, pitch={self.joint_pitch}"
            )

        p.resetJointState(self.robot_id, self.joint_yaw, initial_yaw)
        p.resetJointState(self.robot_id, self.joint_pitch, initial_pitch)
        p.setTimeStep(self.sim_dt)

    def is_connected(self) -> bool:
        return p.isConnected()

    def keyboard_events(self) -> dict[int, int]:
        return p.getKeyboardEvents()

    def set_targets(self, yaw_target: float, pitch_target: float) -> None:
        assert self.robot_id is not None
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.joint_yaw,
            controlMode=p.POSITION_CONTROL,
            targetPosition=yaw_target,
            force=200.0,
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.joint_pitch,
            controlMode=p.POSITION_CONTROL,
            targetPosition=pitch_target,
            force=200.0,
        )

    def step(self) -> None:
        p.stepSimulation()

    def stop(self) -> None:
        if p.isConnected():
            if self.client is None:
                p.disconnect()
            else:
                p.disconnect(self.client)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime music-adaptive trajectory playback using microphone audio features."
    )
    parser.add_argument("--csv", type=Path, default=Path("adaptive_clap/outputs/clap_trajectory.csv"))
    parser.add_argument("--urdf", type=str, default="kuka_iiwa/model.urdf")
    parser.add_argument("--joint-yaw", type=int, default=0)
    parser.add_argument("--joint-pitch", type=int, default=1)
    parser.add_argument("--fixed-base", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--motion",
        choices=("clap", "figure-eight", "csv"),
        default="clap",
        help="Motion style used by the realtime player.",
    )
    parser.add_argument("--motion-cycle-duration", type=float, default=1.0)
    parser.add_argument("--beats-per-cycle", type=int, default=2)
    parser.add_argument("--speed-min", type=float, default=0.5)
    parser.add_argument("--speed-max", type=float, default=1.8)
    parser.add_argument("--amp-min", type=float, default=0.65)
    parser.add_argument("--amp-max", type=float, default=1.45)
    parser.add_argument("--accent-gain-deg", type=float, default=10.0)
    parser.add_argument("--accent-duration", type=float, default=0.16)
    parser.add_argument("--clap-yaw-amp-deg", type=float, default=0.0)
    parser.add_argument("--clap-open-pitch-deg", type=float, default=22.0)
    parser.add_argument("--clap-pitch-deg", type=float, default=-24.0)
    parser.add_argument("--clap-bob-deg", type=float, default=5.0)
    parser.add_argument("--figure-yaw-amp-deg", type=float, default=28.0)
    parser.add_argument("--figure-pitch-amp-deg", type=float, default=22.0)
    parser.add_argument("--mic-sample-rate", type=int, default=16000)
    parser.add_argument("--mic-block-size", type=int, default=512)
    parser.add_argument("--plp-history-sec", type=float, default=8.0)
    parser.add_argument("--plp-analysis-interval-sec", type=float, default=0.10)
    parser.add_argument("--plp-hop-length", type=int, default=256)
    parser.add_argument("--plp-peak-prominence", type=float, default=0.15)
    parser.add_argument("--onset-threshold-scale", type=float, default=3.0)
    parser.add_argument(
        "--noise-gate-rms",
        type=float,
        default=0.002,
        help="Minimum RMS required before microphone input is treated as music.",
    )
    parser.add_argument(
        "--noise-gate-ratio",
        type=float,
        default=1.8,
        help="Input RMS must exceed estimated noise by this multiplier to be active.",
    )
    parser.add_argument(
        "--startup-calibration-sec",
        type=float,
        default=1.0,
        help="Seconds used at startup to estimate the room/microphone noise floor.",
    )
    parser.add_argument("--smoothing-tau", type=float, default=0.25)
    parser.add_argument("--tempo-timeout", type=float, default=2.0)
    parser.add_argument("--min-beat-period", type=float, default=0.25)
    parser.add_argument("--max-beat-period", type=float, default=2.0)
    parser.add_argument("--mic-refractory-sec", type=float, default=0.18)
    parser.add_argument(
        "--beat-confidence-threshold",
        type=float,
        default=0.25,
        help="Ignore PLP beats below this confidence before applying trajectory alignment.",
    )
    parser.add_argument(
        "--beat-keypoint-interval-ratio",
        type=float,
        default=0.85,
        help=(
            "Reject beats closer than this fraction of the fastest allowed authored keypoint spacing. "
            "The spacing is derived from keypoint phases and --speed-max."
        ),
    )
    parser.add_argument("--neutral-yaw-rad", type=float, default=None)
    parser.add_argument("--neutral-pitch-rad", type=float, default=None)
    parser.add_argument("--disable-keypoint-alignment", action="store_true")
    parser.add_argument("--status-interval", type=float, default=1.0)
    return parser.parse_args()


def load_trajectory(csv_path: Path) -> Trajectory:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"t", "yaw_rad", "pitch_rad"}
        if reader.fieldnames is None:
            raise ValueError("CSV is missing a header row.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        t_vals: list[float] = []
        yaw_vals: list[float] = []
        pitch_vals: list[float] = []
        phase_vals: list[Optional[float]] = []
        keypoint_candidates: list[tuple[float, float]] = []

        has_phase = "phase" in reader.fieldnames
        has_keypoint = "keypoint" in reader.fieldnames
        has_weight = "beat_weight" in reader.fieldnames
        for row in reader:
            t_vals.append(float(row["t"]))
            yaw_vals.append(float(row["yaw_rad"]))
            pitch_vals.append(float(row["pitch_rad"]))
            phase_vals.append(float(row["phase"]) % 1.0 if has_phase and row["phase"] != "" else None)

            weight = float(row["beat_weight"]) if has_weight and row["beat_weight"] != "" else 0.0
            marked = has_keypoint and is_truthy(row["keypoint"])
            if marked or weight > 0.0:
                phase_value = phase_vals[-1]
                keypoint_candidates.append((0.0 if phase_value is None else phase_value, weight if weight > 0.0 else 1.0))

    t = np.asarray(t_vals, dtype=float)
    yaw = np.asarray(yaw_vals, dtype=float)
    pitch = np.asarray(pitch_vals, dtype=float)

    if t.size < 2:
        raise ValueError("Trajectory requires at least 2 rows.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("Column 't' must be strictly increasing.")

    if any(phase is None for phase in phase_vals):
        phase = (t - t[0]) / max(t[-1] - t[0], 1e-6)
    else:
        phase = np.asarray([float(value) for value in phase_vals], dtype=float)

    keypoint_phases = tuple(
        phase for phase, _weight in sorted(keypoint_candidates, key=lambda item: (-item[1], item[0]))
    )
    keypoint_phases = tuple(dict.fromkeys(round(phase, 6) for phase in keypoint_phases))
    return Trajectory(t=t, yaw=yaw, pitch=pitch, phase=phase, keypoint_phases=keypoint_phases)


def make_builtin_trajectory(cycle_duration: float, keypoint_phases: tuple[float, ...]) -> Trajectory:
    duration = max(cycle_duration, 1e-6)
    t = np.asarray([0.0, duration], dtype=float)
    return Trajectory(
        t=t,
        yaw=np.zeros_like(t),
        pitch=np.zeros_like(t),
        phase=np.asarray([0.0, 1.0], dtype=float),
        keypoint_phases=keypoint_phases,
    )


def any_key_triggered(events: dict[int, int], keys: list[int]) -> bool:
    for key in keys:
        if key in events and events[key] & p.KEY_WAS_TRIGGERED:
            return True
    return False


def print_status(controller: AdaptiveMotionController, last_frame: Optional[MusicFrame]) -> None:
    bpm = controller.estimated_bpm
    bpm_text = f"{bpm:.1f}" if bpm is not None else "--"
    rms_text = f"{last_frame.rms:.4f}/{last_frame.gate_rms:.4f}" if last_frame is not None else "--"
    rms_norm_text = f"{last_frame.rms_norm:.2f}" if last_frame is not None else "--"
    bright_text = f"{last_frame.brightness:.2f}" if last_frame is not None else "--"
    band_text = (
        f"{last_frame.low_energy:.2f}/{last_frame.mid_energy:.2f}/{last_frame.high_energy:.2f}"
        if last_frame is not None
        else "--"
    )
    rolloff_text = f"{last_frame.spectral_rolloff:.2f}" if last_frame is not None else "--"
    zcr_text = f"{last_frame.zero_crossing_rate:.2f}" if last_frame is not None else "--"
    rhythm_text = f"{last_frame.rhythm_density:.2f}" if last_frame is not None else "--"
    stable_text = f"{last_frame.tempo_stability:.2f}" if last_frame is not None else "--"
    offbeat_text = f"{last_frame.offbeat_ratio:.2f}" if last_frame is not None else "--"
    state_text = "music" if last_frame is not None and last_frame.is_active else "silent"
    print(
        f"{state_text} | "
        f"bpm={bpm_text} | speed={controller.speed_multiplier:.2f}x | "
        f"amp={controller.amplitude_scale:.2f}x | rms/gate={rms_text} | "
        f"level={rms_norm_text} | brightness={bright_text} | "
        f"bands(L/M/H)={band_text} | rolloff={rolloff_text} | zcr={zcr_text} | "
        f"rhythm={rhythm_text} | stable={stable_text} | offbeat={offbeat_text}"
    )


def main() -> None:
    args = parse_args()
    if args.motion == "clap":
        keypoint_phases = (0.0, 0.5)
    elif args.motion == "csv":
        trajectory = load_trajectory(args.csv)
        keypoint_phases = trajectory.keypoint_phases
    else:
        keypoint_phases = ()
    if args.motion != "csv":
        trajectory = make_builtin_trajectory(args.motion_cycle_duration, keypoint_phases)
    use_keypoints = bool(keypoint_phases) and not args.disable_keypoint_alignment

    analyzer = RealtimeMusicAnalyzer(
        sample_rate=args.mic_sample_rate,
        block_size=args.mic_block_size,
        onset_threshold_scale=args.onset_threshold_scale,
        min_beat_period=args.min_beat_period,
        max_beat_period=args.max_beat_period,
        refractory_sec=args.mic_refractory_sec,
        noise_gate_rms=args.noise_gate_rms,
        noise_gate_ratio=args.noise_gate_ratio,
        startup_calibration_sec=args.startup_calibration_sec,
        plp_history_sec=args.plp_history_sec,
        plp_analysis_interval_sec=args.plp_analysis_interval_sec,
        plp_hop_length=args.plp_hop_length,
        plp_peak_prominence=args.plp_peak_prominence,
    )
    controller = AdaptiveMotionController(
        authored_cycle_duration=trajectory.duration,
        beats_per_cycle=args.beats_per_cycle,
        keypoint_phases=keypoint_phases,
        use_keypoints=use_keypoints,
        smoothing_tau=args.smoothing_tau,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        amp_min=args.amp_min,
        amp_max=args.amp_max,
        accent_duration=args.accent_duration,
        tempo_timeout=args.tempo_timeout,
        beat_confidence_threshold=args.beat_confidence_threshold,
        beat_keypoint_interval_ratio=args.beat_keypoint_interval_ratio,
    )
    sampler = TrajectorySampler(
        trajectory=trajectory,
        motion=args.motion,
        neutral_yaw=args.neutral_yaw_rad,
        neutral_pitch=args.neutral_pitch_rad,
        accent_gain_deg=args.accent_gain_deg,
        clap_yaw_amp_deg=args.clap_yaw_amp_deg,
        clap_open_pitch_deg=args.clap_open_pitch_deg,
        clap_pitch_deg=args.clap_pitch_deg,
        clap_bob_deg=args.clap_bob_deg,
        figure_yaw_amp_deg=args.figure_yaw_amp_deg,
        figure_pitch_amp_deg=args.figure_pitch_amp_deg,
    )
    player = PyBulletRobotPlayer(
        urdf=args.urdf,
        joint_yaw=args.joint_yaw,
        joint_pitch=args.joint_pitch,
        fixed_base=args.fixed_base,
        sim_dt=1.0 / 240.0,
    )

    initial_yaw, initial_pitch = sampler.sample(0.0, 0.0, 0.0, 0.0)
    player.start(initial_yaw, initial_pitch)
    print(f"Calibrating microphone noise from the first {args.startup_calibration_sec:.2f}s of live input.")
    analyzer.start()
    print(
        f"Realtime music adaptive player started with PLP rhythm tracking and {args.motion} motion. "
        "Press R to reset, Q or Esc to quit."
    )
    if use_keypoints:
        print(f"Using {len(keypoint_phases)} motion keypoint phase(s) for beat alignment.")
    else:
        print("Using evenly spaced beat phases.")

    exit_keys = [ord("q"), ord("Q")]
    escape_key = getattr(p, "B3G_ESCAPE", None)
    if escape_key is not None:
        exit_keys.append(escape_key)

    last_frame: Optional[MusicFrame] = None
    last_status = time.perf_counter()
    try:
        while player.is_connected():
            now = time.perf_counter()
            events = player.keyboard_events()
            if any_key_triggered(events, exit_keys):
                break
            if ord("r") in events and events[ord("r")] & p.KEY_WAS_TRIGGERED:
                analyzer.reset()
                controller.reset()
                print("Realtime music controller reset.")

            for frame in analyzer.drain():
                last_frame = frame
                controller.observe(frame)

            phase, amplitude, accent, brightness = controller.update(now)
            yaw_target, pitch_target = sampler.sample(phase, amplitude, accent, brightness)
            player.set_targets(yaw_target, pitch_target)
            player.step()

            if now - last_status >= args.status_interval:
                print_status(controller, last_frame)
                last_status = now

            if args.realtime:
                time.sleep(1.0 / 240.0)
    finally:
        analyzer.stop()
        player.stop()


if __name__ == "__main__":
    main()
