from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import librosa
import numpy as np
import soundfile as sf

from aistpp_velocity_keypoints import detect_aistpp_file


CATALOG_SCHEMA_VERSION = 2
SUPPORTED_CATALOG_SCHEMA_VERSIONS = (1, CATALOG_SCHEMA_VERSION)
DEFAULT_ANALYSIS_SAMPLE_RATE = 16_000
DEFAULT_WINDOW_SECONDS = 6.0
DEFAULT_HOP_SECONDS = 2.0
DEFAULT_MOTION_FPS = 60.0
MOTION_NAME_PATTERN = re.compile(
    r"^g(?P<genre>[A-Z]{2})_s(?P<situation>[A-Z]{2})_c(?P<camera>[^_]+)_"
    r"d(?P<dancer>\d+)_m(?P<music>[A-Z]{2}\d+)_ch(?P<choreo>\d+)$"
)


def _array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _unit_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = _array(values).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return default
    return result if math.isfinite(result) else default


def parse_motion_name(name: str) -> dict[str, str]:
    match = MOTION_NAME_PATTERN.match(Path(name).stem)
    if match is None:
        raise ValueError(f"Unexpected AIST++ motion name: {name}")
    return match.groupdict()


def robust_audio_normalize(audio: np.ndarray) -> np.ndarray:
    """Remove DC and recording gain without using absolute loudness as a feature."""

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    samples = samples - float(np.median(samples))
    scale = float(np.percentile(np.abs(samples), 95.0))
    if not math.isfinite(scale) or scale <= 1e-7:
        return np.zeros_like(samples)
    return np.clip(0.8 * samples / scale, -1.0, 1.0).astype(np.float32)


def load_audio_mono(path: Path, sample_rate: int = DEFAULT_ANALYSIS_SAMPLE_RATE) -> np.ndarray:
    audio, source_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if int(source_rate) != int(sample_rate):
        mono = librosa.resample(
            mono,
            orig_sr=int(source_rate),
            target_sr=int(sample_rate),
            res_type="soxr_hq",
        ).astype(np.float32)
    return mono


def iter_audio_windows(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
) -> Iterable[tuple[float, float, np.ndarray]]:
    window_samples = max(1, int(round(window_seconds * sample_rate)))
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    if audio.size < window_samples:
        padded = np.pad(audio, (0, window_samples - audio.size))
        yield 0.0, float(audio.size / sample_rate), padded.astype(np.float32)
        return
    starts = list(range(0, audio.size - window_samples + 1, hop_samples))
    final_start = audio.size - window_samples
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    for start in starts:
        end = start + window_samples
        yield (
            float(start / sample_rate),
            float(end / sample_rate),
            np.asarray(audio[start:end], dtype=np.float32),
        )


@dataclass(frozen=True)
class AudioDescriptor:
    embedding: np.ndarray
    rhythm_timbre: np.ndarray
    tag_probabilities: np.ndarray
    bpm: float
    beat_strength: float
    onset_density: float
    offbeat_ratio: float
    tempo_stability: float
    spectral_flux: float
    percussive_ratio: float
    signal_rms: float = 0.0
    dynamic_range_db: float = 0.0

    @property
    def activity(self) -> float:
        onset = min(max(self.onset_density / 4.0, 0.0), 1.0)
        return float(
            np.clip(
                0.30 * self.beat_strength
                + 0.30 * onset
                + 0.20 * self.spectral_flux
                + 0.20 * self.percussive_ratio,
                0.0,
                1.0,
            )
        )

    def scalar_metadata(self) -> dict[str, float]:
        return {
            "bpm": self.bpm,
            "beat_strength": self.beat_strength,
            "onset_density": self.onset_density,
            "offbeat_ratio": self.offbeat_ratio,
            "tempo_stability": self.tempo_stability,
            "spectral_flux": self.spectral_flux,
            "percussive_ratio": self.percussive_ratio,
            "signal_rms": self.signal_rms,
            "dynamic_range_db": self.dynamic_range_db,
            "activity": self.activity,
        }

    @property
    def beat_quality(self) -> float:
        onset = float(np.clip(self.onset_density / 4.0, 0.0, 1.0))
        return float(
            np.clip(
                0.40 * self.beat_strength
                + 0.35 * self.tempo_stability
                + 0.25 * onset,
                0.0,
                1.0,
            )
        )


class OnnxEffnetBackend:
    """Optional Windows-native ONNX backend for Discogs EffNet models."""

    def __init__(
        self,
        model_path: Path,
        preferred_dimension: int | None = None,
        normalize_output: bool = True,
        intra_op_threads: int | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required when an ONNX embedding/tag model is configured."
            ) from exc

        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        session_options = ort.SessionOptions()
        if intra_op_threads is not None:
            session_options.intra_op_num_threads = max(int(intra_op_threads), 1)
            session_options.inter_op_num_threads = 1
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input = self.session.get_inputs()[0]
        self.preferred_dimension = preferred_dimension
        self.normalize_output = bool(normalize_output)
        candidates = [
            output
            for output in self.session.get_outputs()
            if len(output.shape) == 2 and isinstance(output.shape[-1], int)
        ]
        self.output_dimensions = tuple(int(output.shape[-1]) for output in candidates)
        preferred = [
            output
            for output in candidates
            if preferred_dimension is not None
            and output.shape[-1] == preferred_dimension
        ]
        selected = preferred[0] if preferred else (candidates[0] if candidates else None)
        if selected is None:
            raise RuntimeError(
                f"ONNX model exposes no fixed-width two-dimensional output: {self.model_path}"
            )
        self.output_dimension = int(selected.shape[-1])
        self.labels_by_dimension: dict[int, tuple[str, ...]] = {}
        self.sidecar_metadata: dict[str, Any] = {}
        self.sidecar_sha256: str | None = None
        metadata_path = self.model_path.with_suffix(".json")
        if metadata_path.exists():
            try:
                model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.sidecar_metadata = dict(model_metadata)
                self.sidecar_sha256 = self._file_sha256(metadata_path)
                labels = tuple(str(label) for label in model_metadata.get("classes", ()))
                if labels:
                    self.labels_by_dimension[len(labels)] = labels
            except (OSError, ValueError, TypeError):
                self.labels_by_dimension = {}
                self.sidecar_metadata = {}
                self.sidecar_sha256 = None
        self.labels = self.labels_by_dimension.get(self.output_dimension, ())

    @property
    def sha256(self) -> str:
        return self._file_sha256(self.model_path)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def identity_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "path": str(self.model_path),
            "sha256": self.sha256,
        }
        for key in ("name", "version", "release_date", "framework"):
            if key in self.sidecar_metadata:
                metadata[key] = self.sidecar_metadata[key]
        if self.sidecar_sha256 is not None:
            metadata["sidecar_sha256"] = self.sidecar_sha256
        return metadata

    @staticmethod
    def patches(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != DEFAULT_ANALYSIS_SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=DEFAULT_ANALYSIS_SAMPLE_RATE,
                res_type="soxr_hq",
            )
        # Match Essentia TensorflowInputMusiCNN: unnormalised Hann window,
        # power spectrum, Slaney/unit-area mel bands, then
        # log10(1 + 10000 * bands).
        frames = librosa.util.frame(
            np.pad(np.asarray(audio, dtype=np.float32), (256, 256)),
            frame_length=512,
            hop_length=256,
        ).T
        windowed = frames * np.hanning(512).astype(np.float32)
        # Essentia's TensorflowInputMusiCNN feeds power mel bands into the
        # Discogs EffNet models.  Feeding magnitude here pushes otherwise
        # unrelated audio toward the same Electronic activations.
        power = np.square(
            np.abs(np.fft.rfft(windowed, n=512, axis=1)).astype(np.float32)
        )
        mel_basis = librosa.filters.mel(
            sr=DEFAULT_ANALYSIS_SAMPLE_RATE,
            n_fft=512,
            n_mels=96,
            fmin=0.0,
            fmax=8_000.0,
            htk=False,
            norm="slaney",
        ).astype(np.float32)
        mel = power @ mel_basis.T
        log_mel = np.log10(1.0 + 10_000.0 * np.maximum(mel, 0.0))
        patch_size = 128
        patch_hop = 62
        if log_mel.shape[0] < patch_size:
            log_mel = np.pad(log_mel, ((0, patch_size - log_mel.shape[0]), (0, 0)))
        starts = list(range(0, log_mel.shape[0] - patch_size + 1, patch_hop))
        if not starts:
            starts = [0]
        return np.stack([log_mel[start : start + patch_size] for start in starts]).astype(
            np.float32
        )

    def encode_outputs(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict[int, np.ndarray]:
        patches = self.patches(audio, sample_rate)
        expected_shape = self.input.shape
        fixed_batch = (
            expected_shape
            and isinstance(expected_shape[0], int)
            and expected_shape[0] > patches.shape[0]
        )
        actual_count = patches.shape[0]
        if fixed_batch:
            patches = np.pad(
                patches,
                ((0, int(expected_shape[0]) - actual_count), (0, 0), (0, 0)),
            )
        outputs = self.session.run(None, {self.input.name: patches})
        candidates = [
            np.asarray(output, dtype=np.float32)
            for output in outputs
            if np.asarray(output).ndim == 2
        ]
        if not candidates:
            raise RuntimeError(f"ONNX model produced no two-dimensional output: {self.model_path}")
        return {
            int(candidate.shape[-1]): np.mean(
                candidate[:actual_count],
                axis=0,
            ).astype(np.float32)
            for candidate in candidates
        }

    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        outputs = self.encode_outputs(audio, sample_rate)
        if self.preferred_dimension is not None:
            selected = outputs.get(self.preferred_dimension)
        else:
            selected = None
        if selected is None:
            selected = outputs[next(iter(outputs))]
        averaged = np.asarray(selected, dtype=np.float32)
        return _unit_vector(averaged) if self.normalize_output else averaged


class AudioFeatureExtractor:
    def __init__(
        self,
        sample_rate: int = DEFAULT_ANALYSIS_SAMPLE_RATE,
        embedding_model: Path | None = None,
        tag_model: Path | None = None,
        onnx_intra_op_threads: int | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.last_timing_ms: dict[str, float] = {}
        resolved_embedding = Path(embedding_model).resolve() if embedding_model else None
        resolved_tags = Path(tag_model).resolve() if tag_model else None
        self.embedding_backend = (
            OnnxEffnetBackend(
                resolved_embedding,
                preferred_dimension=1280,
                intra_op_threads=onnx_intra_op_threads,
            )
            if embedding_model is not None
            else None
        )
        if (
            self.embedding_backend is not None
            and resolved_tags is not None
            and resolved_tags == resolved_embedding
            and 400 in self.embedding_backend.output_dimensions
        ):
            self.tag_backend = self.embedding_backend
        else:
            self.tag_backend = (
                OnnxEffnetBackend(
                    resolved_tags,
                    preferred_dimension=400,
                    normalize_output=False,
                    intra_op_threads=onnx_intra_op_threads,
                )
            if tag_model is not None
            else None
            )

    @property
    def backend_name(self) -> str:
        return "discogs-effnet-onnx" if self.embedding_backend is not None else "dsp"

    def model_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {"embedding_backend": self.backend_name}
        if self.embedding_backend is not None:
            metadata["embedding_model"] = self.embedding_backend.identity_metadata()
        if self.tag_backend is not None:
            metadata["tag_model"] = {
                **self.tag_backend.identity_metadata(),
                "labels": list(self.tag_backend.labels_by_dimension.get(400, ())),
            }
        return metadata

    def describe(self, audio: np.ndarray, source_sample_rate: int | None = None) -> AudioDescriptor:
        total_started = time.perf_counter_ns()
        source_rate = int(source_sample_rate or self.sample_rate)
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if source_rate != self.sample_rate:
            samples = librosa.resample(
                samples,
                orig_sr=source_rate,
                target_sr=self.sample_rate,
                res_type="soxr_hq",
            ).astype(np.float32)
        raw_rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
        raw_abs = np.abs(samples)
        low_level = float(np.percentile(raw_abs, 10.0)) if raw_abs.size else 0.0
        high_level = float(np.percentile(raw_abs, 95.0)) if raw_abs.size else 0.0
        dynamic_range_db = float(
            20.0 * math.log10(max(high_level, 1e-8) / max(low_level, 1e-8))
        )
        samples = robust_audio_normalize(samples)
        preprocess_finished = time.perf_counter_ns()
        if samples.size < 512 or not np.any(samples):
            descriptor = self._empty_descriptor()
            total_finished = time.perf_counter_ns()
            self.last_timing_ms = {
                "audio_preprocess_ms": (preprocess_finished - total_started) / 1_000_000.0,
                "handcrafted_features_ms": 0.0,
                "onnx_inference_ms": 0.0,
                "descriptor_total_ms": (total_finished - total_started) / 1_000_000.0,
            }
            return descriptor

        use_dsp_embedding = self.embedding_backend is None
        hop = 256
        duration = max(samples.size / self.sample_rate, 1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onset_env = librosa.onset.onset_strength(
                y=samples,
                sr=self.sample_rate,
                hop_length=hop,
            )
            tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                hop_length=hop,
            )
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                hop_length=hop,
                units="frames",
            )
            mel = librosa.feature.melspectrogram(
                y=samples,
                sr=self.sample_rate,
                n_fft=1024,
                hop_length=hop,
                n_mels=64,
                fmax=self.sample_rate / 2.0,
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=13)
            chroma = (
                librosa.feature.chroma_stft(
                    y=samples,
                    sr=self.sample_rate,
                    n_fft=1024,
                    hop_length=hop,
                )
                if use_dsp_embedding
                else None
            )
            contrast = librosa.feature.spectral_contrast(
                y=samples,
                sr=self.sample_rate,
                n_fft=1024,
                hop_length=hop,
                n_bands=5,
            )

        bpm = _safe_float(tempo)
        onset_scale = max(float(np.percentile(onset_env, 95.0)), 1e-8)
        normalized_onsets = np.clip(onset_env / onset_scale, 0.0, 1.0)
        beat_strength = (
            float(np.mean(normalized_onsets[np.asarray(beat_frames, dtype=int)]))
            if len(beat_frames)
            else 0.0
        )
        onset_density = float(len(onset_frames) / duration)
        tempo_stability = self._tempo_stability(beat_frames, hop)
        offbeat_ratio = self._offbeat_ratio(onset_frames, beat_frames)
        spectral_flux = (
            float(np.mean(np.maximum(np.diff(log_mel, axis=1), 0.0)))
            if log_mel.shape[1] > 1
            else 0.0
        )
        spectral_flux = float(np.clip(spectral_flux / 6.0, 0.0, 1.0))
        percussive_ratio = 0.0
        if use_dsp_embedding and mel.size:
            harmonic_mel, percussive_mel = librosa.decompose.hpss(mel)
            total_mel_energy = max(float(np.sum(harmonic_mel + percussive_mel)), 1e-12)
            percussive_ratio = float(
                np.clip(np.sum(percussive_mel) / total_mel_energy, 0.0, 1.0)
            )

        band_ratios = self._band_ratios(samples)
        mfcc_mean = np.mean(mfcc, axis=1)
        contrast_mean = np.mean(contrast, axis=1)
        rhythm = np.asarray(
            [
                math.log2(max(bpm, 1.0) / 120.0),
                beat_strength,
                onset_density / 4.0,
                offbeat_ratio,
                tempo_stability,
                spectral_flux,
                percussive_ratio,
                *band_ratios,
                *mfcc_mean[:8],
                *contrast_mean[:6],
            ],
            dtype=np.float32,
        )
        handcrafted_finished = time.perf_counter_ns()
        model_started = handcrafted_finished
        if (
            self.embedding_backend is not None
            and self.tag_backend is self.embedding_backend
        ):
            model_outputs = self.embedding_backend.encode_outputs(
                samples,
                self.sample_rate,
            )
            embedding = _unit_vector(
                model_outputs[self.embedding_backend.output_dimension]
            )
            tags = model_outputs[400]
        else:
            if self.embedding_backend is not None:
                embedding = self.embedding_backend.encode(samples, self.sample_rate)
            else:
                assert chroma is not None
                dsp_embedding = np.concatenate(
                    (
                        mfcc_mean,
                        np.std(mfcc, axis=1),
                        np.mean(chroma, axis=1),
                        np.std(chroma, axis=1),
                        contrast_mean,
                        np.std(contrast, axis=1),
                        band_ratios,
                        np.asarray(
                            [
                                beat_strength,
                                onset_density / 4.0,
                                offbeat_ratio,
                                tempo_stability,
                                spectral_flux,
                                percussive_ratio,
                            ],
                            dtype=np.float32,
                        ),
                    )
                )
                embedding = _unit_vector(dsp_embedding)
            tags = (
                self.tag_backend.encode(samples, self.sample_rate)
                if self.tag_backend is not None
                else np.empty(0, dtype=np.float32)
            )
        model_finished = time.perf_counter_ns()
        descriptor = AudioDescriptor(
            embedding=_array(embedding),
            rhythm_timbre=_array(rhythm),
            tag_probabilities=_array(tags),
            bpm=bpm,
            beat_strength=beat_strength,
            onset_density=onset_density,
            offbeat_ratio=offbeat_ratio,
            tempo_stability=tempo_stability,
            spectral_flux=spectral_flux,
            percussive_ratio=percussive_ratio,
            signal_rms=raw_rms,
            dynamic_range_db=dynamic_range_db,
        )
        total_finished = time.perf_counter_ns()
        self.last_timing_ms = {
            "audio_preprocess_ms": (preprocess_finished - total_started) / 1_000_000.0,
            "handcrafted_features_ms": (
                handcrafted_finished - preprocess_finished
            ) / 1_000_000.0,
            "onnx_inference_ms": (
                model_finished - model_started
            ) / 1_000_000.0 if (
                self.embedding_backend is not None or self.tag_backend is not None
            ) else 0.0,
            "descriptor_finalize_ms": (total_finished - model_finished) / 1_000_000.0,
            "descriptor_total_ms": (total_finished - total_started) / 1_000_000.0,
        }
        return descriptor

    def _empty_descriptor(self) -> AudioDescriptor:
        embedding_size = (
            self.embedding_backend.output_dimension
            if self.embedding_backend is not None
            else 71
        )
        tag_size = 0
        if self.tag_backend is not None:
            tag_size = (
                400
                if self.tag_backend is self.embedding_backend
                else self.tag_backend.output_dimension
            )
        return AudioDescriptor(
            embedding=np.zeros(embedding_size, dtype=np.float32),
            rhythm_timbre=np.zeros(24, dtype=np.float32),
            tag_probabilities=np.zeros(tag_size, dtype=np.float32),
            bpm=0.0,
            beat_strength=0.0,
            onset_density=0.0,
            offbeat_ratio=0.0,
            tempo_stability=0.0,
            spectral_flux=0.0,
            percussive_ratio=0.0,
            signal_rms=0.0,
            dynamic_range_db=0.0,
        )

    def _band_ratios(self, samples: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(samples)) ** 2
        frequencies = np.fft.rfftfreq(samples.size, d=1.0 / self.sample_rate)
        total = max(float(np.sum(spectrum)), 1e-12)
        bands = ((20.0, 250.0), (250.0, 2_000.0), (2_000.0, 8_000.0))
        return np.asarray(
            [
                float(np.sum(spectrum[(frequencies >= low) & (frequencies < high)]))
                / total
                for low, high in bands
            ],
            dtype=np.float32,
        )

    def _tempo_stability(self, beat_frames: Sequence[int], hop: int) -> float:
        if len(beat_frames) < 4:
            return 0.0
        times = librosa.frames_to_time(
            np.asarray(beat_frames),
            sr=self.sample_rate,
            hop_length=hop,
        )
        intervals = np.diff(times)
        mean = float(np.mean(intervals))
        if mean <= 1e-8:
            return 0.0
        coefficient = float(np.std(intervals) / mean)
        return float(np.clip(math.exp(-4.0 * coefficient), 0.0, 1.0))

    @staticmethod
    def _offbeat_ratio(onset_frames: Sequence[int], beat_frames: Sequence[int]) -> float:
        if len(onset_frames) == 0 or len(beat_frames) < 2:
            return 0.0
        beats = np.asarray(beat_frames, dtype=float)
        offbeat = 0
        considered = 0
        for onset in np.asarray(onset_frames, dtype=float):
            right = int(np.searchsorted(beats, onset))
            if right == 0 or right >= len(beats):
                continue
            left_beat, right_beat = beats[right - 1], beats[right]
            midpoint = 0.5 * (left_beat + right_beat)
            beat_distance = min(abs(onset - left_beat), abs(onset - right_beat))
            if abs(onset - midpoint) < beat_distance:
                offbeat += 1
            considered += 1
        return float(offbeat / considered) if considered else 0.0


@dataclass(frozen=True)
class MotionProfile:
    motion_id: str
    music_id: str
    genre: str
    situation: str
    motion_path: str
    duration_seconds: float
    keypoint_phases: tuple[float, ...]
    keypoint_scores: tuple[float, ...]
    keypoint_density_hz: float
    weighted_keypoint_density: float
    keypoint_interval_cv: float
    velocity_median: float
    velocity_p90: float
    original_bpm: float
    preflight_passed: bool
    preflight_reason: str
    motion_cluster_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["keypoint_phases"] = list(self.keypoint_phases)
        result["keypoint_scores"] = list(self.keypoint_scores)
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "MotionProfile":
        data = dict(values)
        data["keypoint_phases"] = tuple(float(v) for v in data["keypoint_phases"])
        data["keypoint_scores"] = tuple(float(v) for v in data["keypoint_scores"])
        if not data.get("motion_cluster_id"):
            velocity_tier = int(round(float(data["velocity_p90"]) * 2.0))
            density_tier = int(round(float(data["weighted_keypoint_density"]) * 4.0))
            data["motion_cluster_id"] = (
                f"{data.get('genre', 'unknown')}:v{velocity_tier}:k{density_tier}"
            )
        return cls(**data)


@dataclass(frozen=True)
class TrackMatch:
    music_id: str
    genre: str
    score: float
    embedding_score: float
    rhythm_timbre_score: float
    tag_score: float | None


@dataclass(frozen=True)
class GenreMatch:
    genre: str
    score: float
    track_ids: tuple[str, ...]


@dataclass(frozen=True)
class MotionMatch:
    motion_id: str
    music_id: str
    final_score: float
    music_score: float
    motion_score: float
    tempo_score: float
    keypoint_score: float
    activity_score: float
    speed_ratio: float


@dataclass(frozen=True)
class MatchResult:
    tracks: tuple[TrackMatch, ...]
    motions: tuple[MotionMatch, ...]
    query_bpm: float
    query_tags: tuple[tuple[str, float], ...] = ()
    genres: tuple[GenreMatch, ...] = ()
    accepted: bool = True
    confidence: float = 1.0
    rejection_reason: str = ""
    track_margin: float = 0.0
    motion_margin: float = 0.0
    beat_quality: float = 0.0
    signal_rms: float = 0.0
    dynamic_range_db: float = 0.0
    negative_style_score: float = 0.0


class MusicCatalog:
    def __init__(
        self,
        catalog_path: Path,
        metadata: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.metadata = dict(metadata)
        self.segment_metadata = list(self.metadata["segments"])
        self.tracks = dict(self.metadata["tracks"])
        self.motions = {
            key: MotionProfile.from_dict(value)
            for key, value in self.metadata["motions"].items()
        }
        self.embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
        self.rhythm_timbre = np.asarray(arrays["rhythm_timbre"], dtype=np.float32)
        self.tags = np.asarray(arrays["tags"], dtype=np.float32)
        self.embedding_mean = np.asarray(arrays["embedding_mean"], dtype=np.float32)
        self.embedding_std = np.asarray(arrays["embedding_std"], dtype=np.float32)
        self.rhythm_mean = np.asarray(arrays["rhythm_mean"], dtype=np.float32)
        self.rhythm_std = np.asarray(arrays["rhythm_std"], dtype=np.float32)
        self.motion_velocity_median = float(self.metadata["motion_stats"]["velocity_median"])
        self.motion_velocity_scale = max(
            float(self.metadata["motion_stats"]["velocity_scale"]),
            1e-6,
        )

    @classmethod
    def load(cls, catalog_path: Path | str) -> "MusicCatalog":
        path = Path(catalog_path)
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if int(metadata.get("schema_version", -1)) not in SUPPORTED_CATALOG_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported catalog schema {metadata.get('schema_version')}; "
                f"expected one of {SUPPORTED_CATALOG_SCHEMA_VERSIONS}."
            )
        arrays_path = path.parent / metadata["arrays_file"]
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        return cls(path, metadata, arrays)


class MusicMotionMatcher:
    def __init__(
        self,
        catalog: MusicCatalog,
        speed_min: float = 0.55,
        speed_max: float = 1.6,
        style_first: bool = True,
        weak_music_threshold: float = 0.17,
        weak_music_consecutive_windows: int = 2,
        motion_compatibility: bool = True,
    ) -> None:
        self.catalog = catalog
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.style_first = bool(style_first)
        self.weak_music_threshold = max(float(weak_music_threshold), 0.0)
        self.weak_music_consecutive_windows = max(
            int(weak_music_consecutive_windows),
            1,
        )
        self.motion_compatibility = bool(motion_compatibility)
        self.last_timing_ms: dict[str, float] = {}
        self._weak_music_streak = 0
        self._catalog_embeddings = self._standardized_unit_rows(
            catalog.embeddings,
            catalog.embedding_mean,
            catalog.embedding_std,
        )
        self._catalog_rhythm = self._standardized_unit_rows(
            catalog.rhythm_timbre,
            catalog.rhythm_mean,
            catalog.rhythm_std,
        )
        self._catalog_tags = self._unit_rows(catalog.tags)

    def _weak_music_is_persistent(self, negative_style_score: float) -> bool:
        if negative_style_score >= self.weak_music_threshold:
            self._weak_music_streak += 1
        else:
            self._weak_music_streak = 0
        return self._weak_music_streak >= self.weak_music_consecutive_windows

    @staticmethod
    def _tag_genre_priors(
        labels: Sequence[str],
        probabilities: np.ndarray,
    ) -> dict[str, float]:
        """Map broad audio tags onto the ten motion genres in AIST++."""

        scores = defaultdict(float)
        for label, raw_score in zip(labels, probabilities, strict=False):
            score = max(float(raw_score), 0.0)
            parent, _, style = label.partition("---")
            style_lower = style.lower()
            if parent == "Hip Hop" or "hip hop" in style_lower:
                scores["MH"] += score
                scores["LH"] += 0.85 * score
            if parent == "Funk / Soul" or any(
                token in style_lower for token in ("funk", "boogie", "disco")
            ):
                scores["LO"] += score
                scores["PO"] += 0.45 * score
                scores["WA"] += 0.35 * score
            if parent == "Jazz":
                scores["JS"] += score
                scores["JB"] += 0.70 * score
            if parent in {"Classical", "Stage & Screen"}:
                scores["JB"] += score
                scores["JS"] += 0.35 * score
            if parent in {"Latin", "Folk, World, & Country", "Reggae"}:
                scores["WA"] += score
                scores["LO"] += 0.30 * score
            if parent == "Rock":
                scores["BR"] += score
                if any(
                    token in style_lower
                    for token in ("hardcore", "metal", "industrial", "punk")
                ):
                    scores["KR"] += 0.80 * score
            if parent == "Pop":
                scores["PO"] += score
                scores["JS"] += 0.25 * score
            if parent == "Electronic":
                if any(
                    token in style_lower
                    for token in ("house", "techno", "trance", "garage")
                ):
                    scores["HO"] += score
                elif any(
                    token in style_lower
                    for token in ("break", "drum n bass", "jungle", "electro")
                ):
                    scores["BR"] += score
                elif any(
                    token in style_lower
                    for token in ("experimental", "glitch", "idm", "downtempo")
                ):
                    scores["JS"] += score
                else:
                    scores["PO"] += 0.60 * score
        return dict(scores)

    def match(
        self,
        descriptor: AudioDescriptor,
        top_k_tracks: int = 5,
        top_k_motions: int = 10,
    ) -> MatchResult:
        match_started = time.perf_counter_ns()
        query_embedding = self._standardized_unit(
            descriptor.embedding,
            self.catalog.embedding_mean,
            self.catalog.embedding_std,
        )
        query_rhythm = self._standardized_unit(
            descriptor.rhythm_timbre,
            self.catalog.rhythm_mean,
            self.catalog.rhythm_std,
        )
        embedding_scores = self._similarities(self._catalog_embeddings, query_embedding)
        rhythm_scores = self._similarities(self._catalog_rhythm, query_rhythm)

        tags_available = (
            descriptor.tag_probabilities.size > 0
            and self._catalog_tags.shape[1] == descriptor.tag_probabilities.size
        )
        if tags_available:
            query_tags = _unit_vector(descriptor.tag_probabilities)
            tag_scores = self._similarities(self._catalog_tags, query_tags)
            segment_scores = 0.70 * embedding_scores + 0.20 * rhythm_scores + 0.10 * tag_scores
        else:
            tag_scores = np.zeros_like(embedding_scores)
            segment_scores = (0.70 / 0.90) * embedding_scores + (0.20 / 0.90) * rhythm_scores
        similarity_finished = time.perf_counter_ns()

        by_track: dict[str, list[int]] = defaultdict(list)
        for index, segment in enumerate(self.catalog.segment_metadata):
            by_track[str(segment["music_id"])].append(index)

        track_matches: list[TrackMatch] = []
        for music_id, indices in by_track.items():
            ranked = sorted(indices, key=lambda index: float(segment_scores[index]), reverse=True)
            selected = ranked[: min(3, len(ranked))]
            track_matches.append(
                TrackMatch(
                    music_id=music_id,
                    genre=str(self.catalog.tracks[music_id]["genre"]),
                    score=float(np.mean(segment_scores[selected])),
                    embedding_score=float(np.mean(embedding_scores[selected])),
                    rhythm_timbre_score=float(np.mean(rhythm_scores[selected])),
                    tag_score=float(np.mean(tag_scores[selected])) if tags_available else None,
                )
            )
        track_matches.sort(key=lambda item: item.score, reverse=True)
        all_track_matches = tuple(track_matches)

        tracks_by_genre: dict[str, list[TrackMatch]] = defaultdict(list)
        for item in all_track_matches:
            tracks_by_genre[item.genre].append(item)
        genre_matches = [
            GenreMatch(
                genre=genre,
                score=float(np.mean([item.score for item in items[:3]])),
                track_ids=tuple(item.music_id for item in items),
            )
            for genre, items in tracks_by_genre.items()
        ]
        genre_matches.sort(key=lambda item: item.score, reverse=True)

        labels: tuple[str, ...] = ()
        tag_metadata = self.catalog.metadata.get("extractor", {}).get("tag_model")
        if tags_available and tag_metadata:
            labels = tuple(str(label) for label in tag_metadata.get("labels", ()))
        tag_genre_priors = (
            self._tag_genre_priors(labels, descriptor.tag_probabilities)
            if len(labels) == descriptor.tag_probabilities.size
            else {}
        )
        if genre_matches and tag_genre_priors:
            base_values = np.asarray([item.score for item in genre_matches], dtype=float)
            base_min = float(np.min(base_values))
            base_span = max(float(np.max(base_values) - base_min), 1e-9)
            prior_max = max(tag_genre_priors.values(), default=0.0)
            if prior_max >= 0.08:
                genre_matches = [
                    GenreMatch(
                        genre=item.genre,
                        score=float(
                            0.35 * ((item.score - base_min) / base_span)
                            + 0.65
                            * (tag_genre_priors.get(item.genre, 0.0) / prior_max)
                        ),
                        track_ids=item.track_ids,
                    )
                    for item in genre_matches
                ]
                genre_matches.sort(key=lambda item: item.score, reverse=True)
        negative_style_score = 0.0
        if len(labels) == descriptor.tag_probabilities.size:
            weak_markers = (
                "---Ambient",
                "---Dark Ambient",
                "---Drone",
                "---Field Recording",
                "---Noise",
                "---Power Electronics",
                "---Sound Collage",
            )
            weak_indices = [
                index
                for index, label in enumerate(labels)
                if label.startswith("Non-Music---")
                or any(label.endswith(marker) for marker in weak_markers)
            ]
            if weak_indices:
                negative_style_score = float(
                    np.max(descriptor.tag_probabilities[np.asarray(weak_indices, dtype=int)])
                )

        genre_margin = (
            genre_matches[0].score - genre_matches[1].score
            if len(genre_matches) >= 2
            else (genre_matches[0].score if genre_matches else 0.0)
        )
        confidence = float(
            np.clip(
                0.55 * descriptor.beat_quality
                + 0.45 * np.clip(genre_margin / 0.10, 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        rejection_reason = ""
        if self._weak_music_is_persistent(negative_style_score):
            rejection_reason = "non_dance_or_ambient"
        elif descriptor.beat_quality < 0.35:
            rejection_reason = "weak_beat_evidence"
        accepted = not rejection_reason
        if not self.style_first:
            accepted = True
            rejection_reason = ""

        accepted_genres: set[str] = set()
        if genre_matches:
            accepted_genres.add(genre_matches[0].genre)
            if len(genre_matches) > 1 and genre_margin <= 0.025:
                accepted_genres.add(genre_matches[1].genre)

        returned_tracks = all_track_matches[: max(int(top_k_tracks), 1)]
        style_tracks = [
            item
            for item in all_track_matches
            if not self.style_first or item.genre in accepted_genres
        ][: max(int(top_k_tracks), 1)]
        track_scores = {item.music_id: item.score for item in style_tracks}
        retrieval_finished = time.perf_counter_ns()

        motion_matches: list[MotionMatch] = []
        for profile in self.catalog.motions.values():
            if (
                not accepted
                or not profile.preflight_passed
                or profile.music_id not in track_scores
                or (self.style_first and profile.genre not in accepted_genres)
            ):
                continue
            if self.motion_compatibility:
                tempo_score, speed_ratio = self._tempo_compatibility(
                    descriptor.bpm,
                    profile.original_bpm,
                )
                if tempo_score is None:
                    continue
                keypoint_score = self._keypoint_compatibility(
                    descriptor,
                    profile,
                    speed_ratio,
                )
                activity_score = self._activity_compatibility(descriptor, profile)
                motion_score = (
                    0.50 * tempo_score
                    + 0.35 * keypoint_score
                    + 0.15 * activity_score
                )
                final_score = 0.65 * track_scores[profile.music_id] + 0.35 * motion_score
            else:
                tempo_score = keypoint_score = activity_score = motion_score = 1.0
                speed_ratio = 1.0
                final_score = track_scores[profile.music_id]
            music_score = track_scores[profile.music_id]
            motion_matches.append(
                MotionMatch(
                    motion_id=profile.motion_id,
                    music_id=profile.music_id,
                    final_score=float(final_score),
                    music_score=float(music_score),
                    motion_score=float(motion_score),
                    tempo_score=float(tempo_score),
                    keypoint_score=float(keypoint_score),
                    activity_score=float(activity_score),
                    speed_ratio=float(speed_ratio),
                )
            )
        motion_matches.sort(key=lambda item: item.final_score, reverse=True)
        ranking_finished = time.perf_counter_ns()
        query_tags: tuple[tuple[str, float], ...] = ()
        if tags_available and tag_metadata:
            if len(labels) == descriptor.tag_probabilities.size:
                top_indices = np.argsort(descriptor.tag_probabilities)[-5:][::-1]
                query_tags = tuple(
                    (labels[int(index)], float(descriptor.tag_probabilities[int(index)]))
                    for index in top_indices
                )
        result = MatchResult(
            tracks=tuple(returned_tracks),
            motions=tuple(motion_matches[: max(int(top_k_motions), 1)]),
            query_bpm=descriptor.bpm,
            query_tags=query_tags,
            genres=tuple(genre_matches),
            accepted=accepted,
            confidence=confidence,
            rejection_reason=rejection_reason,
            track_margin=(
                returned_tracks[0].score - returned_tracks[1].score
                if len(returned_tracks) >= 2
                else 0.0
            ),
            motion_margin=(
                motion_matches[0].final_score - motion_matches[1].final_score
                if len(motion_matches) >= 2
                else 0.0
            ),
            beat_quality=descriptor.beat_quality,
            signal_rms=descriptor.signal_rms,
            dynamic_range_db=descriptor.dynamic_range_db,
            negative_style_score=negative_style_score,
        )
        match_finished = time.perf_counter_ns()
        scale = 1e-6
        self.last_timing_ms = {
            "similarity_ms": (similarity_finished - match_started) * scale,
            "track_retrieval_ms": (retrieval_finished - similarity_finished) * scale,
            "motion_ranking_ms": (ranking_finished - retrieval_finished) * scale,
            "result_finalize_ms": (match_finished - ranking_finished) * scale,
            "matcher_total_ms": (match_finished - match_started) * scale,
        }
        return result

    def _tempo_compatibility(
        self,
        query_bpm: float,
        original_bpm: float,
    ) -> tuple[float | None, float]:
        if query_bpm <= 0.0 or original_bpm <= 0.0:
            return 0.25, 1.0
        base = query_bpm / original_bpm
        ratios = (base, 0.5 * base, 2.0 * base)
        valid = [ratio for ratio in ratios if self.speed_min <= ratio <= self.speed_max]
        if not valid:
            return None, 1.0
        ratio = min(valid, key=lambda value: abs(math.log(max(value, 1e-8))))
        score = math.exp(-0.5 * (math.log(max(ratio, 1e-8)) / 0.28) ** 2)
        return float(score), float(ratio)

    @staticmethod
    def _keypoint_compatibility(
        descriptor: AudioDescriptor,
        profile: MotionProfile,
        speed_ratio: float,
    ) -> float:
        query_rate = 0.70 * max(descriptor.bpm / 60.0, 0.0) + 0.30 * descriptor.onset_density
        motion_rate = profile.weighted_keypoint_density * speed_ratio
        if query_rate <= 1e-6 or motion_rate <= 1e-6:
            return 0.25
        log_distance = abs(math.log(motion_rate / query_rate))
        density_score = math.exp(-0.5 * (log_distance / 0.65) ** 2)
        regularity_score = math.exp(-min(profile.keypoint_interval_cv, 2.0))
        return float(np.clip(0.80 * density_score + 0.20 * regularity_score, 0.0, 1.0))

    def _activity_compatibility(
        self,
        descriptor: AudioDescriptor,
        profile: MotionProfile,
    ) -> float:
        z_score = (
            profile.velocity_p90 - self.catalog.motion_velocity_median
        ) / self.catalog.motion_velocity_scale
        motion_activity = 1.0 / (1.0 + math.exp(-z_score))
        return float(np.clip(1.0 - abs(descriptor.activity - motion_activity), 0.0, 1.0))

    @staticmethod
    def _similarities(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if matrix.shape[1] == 0 or vector.size == 0:
            return np.zeros(matrix.shape[0], dtype=np.float32)
        cosine = np.clip(matrix @ vector, -1.0, 1.0)
        return (0.5 + 0.5 * cosine).astype(np.float32)

    @staticmethod
    def _unit_rows(matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[1] == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    @classmethod
    def _standardized_unit_rows(
        cls,
        matrix: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        return cls._unit_rows((matrix - mean) / np.maximum(std, 1e-6))

    @staticmethod
    def _standardized_unit(
        vector: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        if vector.size != mean.size:
            raise ValueError(
                f"Query feature dimension {vector.size} does not match catalog dimension {mean.size}."
            )
        return _unit_vector((vector - mean) / np.maximum(std, 1e-6))


class CandidateStabilizer:
    def __init__(self, required_wins: int = 3, margin: float = 0.08) -> None:
        self.required_wins = max(int(required_wins), 1)
        self.margin = max(float(margin), 0.0)
        self.candidate: str | None = None
        self.wins = 0

    def reset(self) -> None:
        self.candidate = None
        self.wins = 0

    def observe(self, result: MatchResult, current_motion_id: str | None) -> str | None:
        if not result.motions:
            self.reset()
            return None
        best = result.motions[0]
        if best.motion_id == current_motion_id:
            self.reset()
            return None
        current_score = next(
            (
                motion.final_score
                for motion in result.motions
                if motion.motion_id == current_motion_id
            ),
            0.0,
        )
        if best.final_score < current_score + self.margin:
            self.reset()
            return None
        if best.motion_id != self.candidate:
            self.candidate = best.motion_id
            self.wins = 1
        else:
            self.wins += 1
        if self.wins < self.required_wins:
            return None
        pending = self.candidate
        self.reset()
        return pending


@dataclass(frozen=True)
class MotionSelection:
    motion_id: str
    reason: str
    final_score: float
    music_score: float
    cluster_id: str = ""


class DiversityCandidateSelector:
    """Track stable, musically acceptable alternatives and rotate them by recency."""

    def __init__(
        self,
        required_wins: int = 3,
        top_k: int = 5,
        score_drop: float = 0.05,
        music_score_drop: float = 0.08,
        recent_history: int = 3,
        motion_clusters: Mapping[str, str] | None = None,
    ) -> None:
        self.required_wins = max(int(required_wins), 1)
        self.top_k = max(int(top_k), 1)
        self.score_drop = max(float(score_drop), 0.0)
        self.music_score_drop = max(float(music_score_drop), 0.0)
        self.recent_history = max(int(recent_history), 0)
        self.motion_clusters = dict(motion_clusters or {})
        self._streaks: dict[str, int] = {}
        self._scores: dict[str, deque[tuple[float, float]]] = {}
        self._latest_order: tuple[str, ...] = ()
        self._last_played: dict[str, int] = {}
        self._play_counter = 0
        self._recent: deque[str] = deque(maxlen=self.recent_history)
        self._recent_clusters: deque[str] = deque(maxlen=self.recent_history)

    def _cluster(self, motion_id: str) -> str:
        return self.motion_clusters.get(motion_id, motion_id)

    def observe(self, result: MatchResult) -> None:
        if not result.motions:
            self._streaks.clear()
            self._scores.clear()
            self._latest_order = ()
            return

        best = result.motions[0]
        candidates = result.motions[: self.top_k]
        eligible = {
            motion.motion_id: motion
            for motion in candidates
            if motion.final_score >= best.final_score - self.score_drop
            and motion.music_score >= best.music_score - self.music_score_drop
        }

        for motion_id in tuple(self._streaks):
            if motion_id not in eligible:
                del self._streaks[motion_id]
                self._scores.pop(motion_id, None)

        for motion_id, motion in eligible.items():
            self._streaks[motion_id] = self._streaks.get(motion_id, 0) + 1
            history = self._scores.setdefault(
                motion_id,
                deque(maxlen=self.required_wins),
            )
            history.append((motion.final_score, motion.music_score))
        self._latest_order = tuple(
            motion.motion_id
            for motion in candidates
            if motion.motion_id in eligible
        )

    def record_played(self, motion_id: str) -> None:
        self._play_counter += 1
        self._last_played[motion_id] = self._play_counter
        self._recent.append(motion_id)
        self._recent_clusters.append(self._cluster(motion_id))

    def stable_candidates(
        self,
        current_motion_id: str | None,
    ) -> tuple[MotionSelection, ...]:
        selections = []
        for motion_id in self._latest_order:
            if motion_id == current_motion_id:
                continue
            history = self._scores.get(motion_id)
            if (
                self._streaks.get(motion_id, 0) < self.required_wins
                or history is None
                or len(history) < self.required_wins
            ):
                continue
            selections.append(
                MotionSelection(
                    motion_id=motion_id,
                    reason="diversity",
                    final_score=float(np.mean([score[0] for score in history])),
                    music_score=float(np.mean([score[1] for score in history])),
                    cluster_id=self._cluster(motion_id),
                )
            )
        return tuple(selections)

    def select(self, current_motion_id: str | None) -> MotionSelection | None:
        candidates = list(self.stable_candidates(current_motion_id))
        if not candidates:
            return None

        recent = set(self._recent)
        recent_clusters = set(self._recent_clusters)
        fresh_clusters = [
            item for item in candidates if item.cluster_id not in recent_clusters
        ]
        fresh_ids = [item for item in candidates if item.motion_id not in recent]
        available = fresh_clusters or fresh_ids or candidates
        return min(
            available,
            key=lambda item: (
                self._last_played.get(item.motion_id, -1),
                -item.final_score,
                item.motion_id,
            ),
        )


class MotionSelectionPolicy:
    """Combine relevance-driven switches with bounded-hold diversity rotation."""

    def __init__(
        self,
        current_motion_id: str,
        required_wins: int = 3,
        score_margin: float = 0.08,
        max_hold_bars: int = 4,
        diversity_top_k: int = 5,
        diversity_score_drop: float = 0.05,
        diversity_music_score_drop: float = 0.08,
        recent_history: int = 3,
        motion_clusters: Mapping[str, str] | None = None,
    ) -> None:
        self.current_motion_id = current_motion_id
        self.max_hold_bars = max(int(max_hold_bars), 0)
        self.bars_held = 0
        self.pending: MotionSelection | None = None
        self.relevance = CandidateStabilizer(required_wins, score_margin)
        self.diversity = DiversityCandidateSelector(
            required_wins=required_wins,
            top_k=diversity_top_k,
            score_drop=diversity_score_drop,
            music_score_drop=diversity_music_score_drop,
            recent_history=recent_history,
            motion_clusters=motion_clusters,
        )
        self.diversity.record_played(current_motion_id)

    def observe(self, result: MatchResult) -> MotionSelection | None:
        if not result.accepted:
            self.pending = None
            self.relevance.reset()
            self.diversity.observe(result)
            return None
        self.diversity.observe(result)
        if self.pending is not None:
            return self.pending

        relevant_motion_id = self.relevance.observe(
            result,
            self.current_motion_id,
        )
        if relevant_motion_id is not None:
            match = next(
                motion
                for motion in result.motions
                if motion.motion_id == relevant_motion_id
            )
            self.pending = MotionSelection(
                motion_id=relevant_motion_id,
                reason="relevance",
                final_score=match.final_score,
                music_score=match.music_score,
                cluster_id=self.diversity._cluster(relevant_motion_id),
            )
            return self.pending

        diversity_armed = (
            self.max_hold_bars > 0
            and self.bars_held >= self.max_hold_bars - 1
        )
        if diversity_armed:
            self.pending = self.diversity.select(self.current_motion_id)
        return self.pending

    def on_bar_boundary(self) -> MotionSelection | None:
        self.bars_held += 1
        return self.ready_selection()

    def ready_selection(self) -> MotionSelection | None:
        if self.pending is None:
            return None
        if self.pending.reason == "relevance":
            return self.pending
        if self.max_hold_bars > 0 and self.bars_held >= self.max_hold_bars:
            return self.pending
        return None

    def complete_switch(self, motion_id: str) -> None:
        self.current_motion_id = motion_id
        self.bars_held = 0
        self.pending = None
        self.relevance.reset()
        self.diversity.record_played(motion_id)

    def preload_motion_ids(self, result: MatchResult) -> tuple[str, ...]:
        motion_ids = []
        if self.pending is not None:
            motion_ids.append(self.pending.motion_id)
        motion_ids.extend(motion.motion_id for motion in result.motions)
        return tuple(dict.fromkeys(motion_ids))

    def diversity_pool(self) -> tuple[MotionSelection, ...]:
        return self.diversity.stable_candidates(self.current_motion_id)


class MotionPreflightValidator:
    """Validate the final canonical GMR-G1 trajectory in headless MuJoCo."""

    def __init__(self, model_path: Path, gmr_motion_root: Path, simulation_stride: int = 4) -> None:
        from realtime_music_humanoid_dancer import MujocoHumanoidPlayer
        from unitree_g1_dance_adapter import UnitreeG1JointPoseAdapter

        self.player = MujocoHumanoidPlayer(model_path, realtime=False, headless=True)
        self.adapter = UnitreeG1JointPoseAdapter(self.player.actuator_names)
        self.gmr_motion_root = Path(gmr_motion_root)
        self.simulation_stride = max(int(simulation_stride), 1)
        if self.adapter.resolved_count != 29:
            raise ValueError(
                f"MuJoCo model resolves {self.adapter.resolved_count}/29 Unitree G1 joints."
            )

    def validate(self, motion_path: Path, fps: float = DEFAULT_MOTION_FPS) -> tuple[bool, str]:
        import mujoco
        from scipy.spatial.transform import Rotation

        from realtime_music_humanoid_dancer import FeatureState, GmrUnitreeG1MotionSampler

        artifact_path = self.gmr_motion_root / f"{motion_path.stem}.pkl"
        try:
            sampler = GmrUnitreeG1MotionSampler(artifact_path, None, 1.0, 0.0, False)
            if (
                sampler.format_version != 1
                or sampler.pipeline_version != 4
                or sampler.source_format != "aistpp_smpl_direct"
            ):
                return False, "GMR artifact is not canonical SMPL-direct pipeline version 4"
            if sampler.retargeter != "GMR":
                return False, f"unexpected retargeter: {sampler.retargeter!r}"
            if sampler.collision_avoidance.get("enabled") is not True:
                return False, "GMR artifact was generated without collision avoidance"
            if sampler.source_motion_id != motion_path.stem:
                return False, (
                    f"GMR source id mismatch: expected {motion_path.stem}, "
                    f"got {sampler.source_motion_id!r}"
                )
            source_digest = hashlib.sha256()
            with motion_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    source_digest.update(chunk)
            if sampler.source_sha256 != source_digest.hexdigest():
                return False, "GMR source hash does not match the AIST++ motion"
            if not isinstance(sampler.retargeter_version, str) or len(sampler.retargeter_version) != 40:
                return False, "GMR artifact does not record a full retargeter commit"
            if not math.isclose(sampler.fps, fps, rel_tol=0.0, abs_tol=1e-6):
                return False, f"GMR fps mismatch: expected {fps:g}, got {sampler.fps:g}"
            features = FeatureState(is_active=True)

            joint_velocity = np.abs(np.diff(sampler.frames.astype(np.float64), axis=0)) * sampler.fps
            if joint_velocity.size and float(np.max(joint_velocity)) > 20.0:
                index = np.unravel_index(int(np.argmax(joint_velocity)), joint_velocity.shape)
                return False, (
                    f"G1 joint velocity spike at frame {index[0] + 1}, "
                    f"{sampler.dof_names[index[1]]}={joint_velocity[index]:.3f} rad/s"
                )
            root_velocity = np.linalg.norm(np.diff(sampler.root_positions, axis=0), axis=1) * sampler.fps
            if root_velocity.size and float(np.max(root_velocity)) > 10.0:
                return False, f"G1 root velocity spike: {float(np.max(root_velocity)):.3f} m/s"
            rotations = Rotation.from_quat(sampler.root_quaternions[:, [1, 2, 3, 0]])
            angular_velocity = (rotations[:-1].inv() * rotations[1:]).magnitude() * sampler.fps
            if angular_velocity.size and float(np.max(angular_velocity)) > 20.0:
                return False, f"G1 root angular velocity spike: {float(np.max(angular_velocity)):.3f} rad/s"

            mujoco.mj_resetData(self.player.model, self.player.data)
            self.player.ground_sampler(sampler, self.adapter)
            minimum_foot_height = math.inf
            for frame_index in range(len(sampler.frames)):
                phase = frame_index / len(sampler.frames)
                frame = sampler.sample_frame(phase, 1.0, 0.0, features)
                joints = self.adapter.adapt_pose(frame.joint_positions, features)
                for name, value in joints.items():
                    joint_range = self.player.actuator_joint_ranges.get(name)
                    if not math.isfinite(value):
                        return False, f"non-finite joint: {name}"
                    if joint_range is not None and not (
                        joint_range[0] - 1e-6 <= value <= joint_range[1] + 1e-6
                    ):
                        return False, f"joint outside qpos range: {name}={value:.5f}"
                if frame_index % self.simulation_stride == 0:
                    self.player.set_frame(frame.with_joint_positions(joints))
                    self.player.step()
                    minimum_foot_height = min(minimum_foot_height, self.player.support_height())
                    if not (
                        np.all(np.isfinite(self.player.data.qpos))
                        and np.all(np.isfinite(self.player.data.qvel))
                        and np.all(np.isfinite(self.player.data.ctrl))
                    ):
                        return False, "MuJoCo state became non-finite"
                    if np.any(self.player.data.ctrl != 0.0):
                        return False, "kinematic preview unexpectedly wrote actuator controls"
            if not math.isfinite(minimum_foot_height) or minimum_foot_height < -1e-5:
                return False, f"foot penetrates floor by {-minimum_foot_height:.6f} m"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return True, "ok"


def _audio_prefix_sha256(path: Path, seconds: float = DEFAULT_WINDOW_SECONDS) -> str:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="int16")
    frame_count = min(len(audio), int(round(seconds * sample_rate)))
    return hashlib.sha256(np.asarray(audio[:frame_count]).tobytes()).hexdigest()


def _representative_audio_variants(
    rows: Sequence[Mapping[str, str]],
    audio_dir: Path,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        parsed = parse_motion_name(row["motion_name"])
        audio_path = audio_dir / row["audio_path"]
        signature = _audio_prefix_sha256(audio_path)
        grouped[(parsed["music"], parsed["situation"], signature)].append(
            {
                "row": dict(row),
                "parsed": parsed,
                "audio_path": audio_path,
            }
        )

    representatives: list[dict[str, Any]] = []
    for (music_id, situation, signature), items in sorted(grouped.items()):
        representative = max(
            items,
            key=lambda item: float(item["row"]["duration_seconds"]),
        )
        representative.update(
            {
                "music_id": music_id,
                "situation": situation,
                "signature": signature,
                "duplicate_count": len(items),
            }
        )
        representatives.append(representative)
    return representatives


def _keypoint_interval_cv(phases: Sequence[float]) -> float:
    if len(phases) < 2:
        return 1.0
    ordered = np.sort(np.mod(np.asarray(phases, dtype=float), 1.0))
    wrapped = np.concatenate((ordered, [ordered[0] + 1.0]))
    intervals = np.diff(wrapped)
    mean = float(np.mean(intervals))
    return float(np.std(intervals) / mean) if mean > 1e-8 else 1.0


def build_music_catalog(
    aistpp_root: Path,
    output_dir: Path,
    *,
    embedding_model: Path | None = None,
    tag_model: Path | None = None,
    mujoco_model: Path | None = None,
    gmr_motion_root: Path | None = None,
    run_preflight: bool = True,
    limit_motions: int | None = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
) -> Path:
    root = Path(aistpp_root).resolve()
    output = Path(output_dir).resolve()
    manifest_path = root / "audio" / "manifest.csv"
    audio_dir = root / "audio"
    motions_dir = root / "motions"
    if not manifest_path.exists():
        raise FileNotFoundError(f"AIST++ audio manifest not found: {manifest_path}")
    if not motions_dir.is_dir():
        raise FileNotFoundError(f"AIST++ motions directory not found: {motions_dir}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit_motions is not None:
        rows = rows[: max(int(limit_motions), 0)]
    if not rows:
        raise ValueError("AIST++ manifest contains no selected rows.")

    failures: list[dict[str, str]] = []
    valid_rows: list[dict[str, str]] = []
    for row in rows:
        audio_path = audio_dir / row["audio_path"]
        motion_path = motions_dir / f"{row['motion_name']}.pkl"
        if not audio_path.exists() or not motion_path.exists():
            failures.append(
                {
                    "motion_id": row["motion_name"],
                    "reason": "missing audio or motion file",
                }
            )
            continue
        valid_rows.append(row)
    if not valid_rows:
        raise RuntimeError("No manifest rows have both an audio file and a motion file.")

    extractor = AudioFeatureExtractor(
        embedding_model=embedding_model,
        tag_model=tag_model,
    )
    variants = _representative_audio_variants(valid_rows, audio_dir)
    descriptors: list[AudioDescriptor] = []
    segment_metadata: list[dict[str, Any]] = []
    tracks: dict[str, dict[str, Any]] = {}

    for variant_index, variant in enumerate(variants):
        music_id = variant["music_id"]
        parsed = variant["parsed"]
        audio_path = Path(variant["audio_path"])
        audio = load_audio_mono(audio_path, extractor.sample_rate)
        variant_id = f"{music_id}:{variant['situation']}:{variant['signature'][:12]}"
        track = tracks.setdefault(
            music_id,
            {
                "music_id": music_id,
                "genre": parsed["genre"],
                "variants": [],
                "motion_ids": [],
            },
        )
        track["variants"].append(
            {
                "variant_id": variant_id,
                "situation": variant["situation"],
                "source_audio": str(audio_path.relative_to(root)),
                "duplicate_count": int(variant["duplicate_count"]),
            }
        )
        for start, end, window in iter_audio_windows(
            audio,
            extractor.sample_rate,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        ):
            descriptor = extractor.describe(window)
            descriptors.append(descriptor)
            segment_metadata.append(
                {
                    "music_id": music_id,
                    "genre": parsed["genre"],
                    "variant_id": variant_id,
                    "variant_index": variant_index,
                    "source_audio": str(audio_path.relative_to(root)),
                    "start_seconds": start,
                    "end_seconds": end,
                    **descriptor.scalar_metadata(),
                }
            )

    bpm_by_music: dict[str, float] = {}
    for music_id in tracks:
        bpms = [
            float(segment["bpm"])
            for segment in segment_metadata
            if segment["music_id"] == music_id and float(segment["bpm"]) > 0.0
        ]
        bpm_by_music[music_id] = float(np.median(bpms)) if bpms else 0.0

    validator = None
    if run_preflight:
        if mujoco_model is None:
            raise ValueError("mujoco_model is required when run_preflight=True.")
        if gmr_motion_root is None:
            gmr_motion_root = root.parent / "aistpp_gmr"
        validator = MotionPreflightValidator(Path(mujoco_model), Path(gmr_motion_root))

    motions: dict[str, MotionProfile] = {}
    for index, row in enumerate(valid_rows, start=1):
        motion_id = row["motion_name"]
        parsed = parse_motion_name(motion_id)
        motion_path = motions_dir / f"{motion_id}.pkl"
        try:
            keypoints = detect_aistpp_file(motion_path, fps=DEFAULT_MOTION_FPS)
            preflight_passed, preflight_reason = (
                validator.validate(motion_path, fps=DEFAULT_MOTION_FPS)
                if validator is not None
                else (True, "skipped")
            )
            duration = float(row["duration_seconds"])
            smoothed = np.asarray(keypoints.smoothed_velocity, dtype=float)
            profile = MotionProfile(
                motion_id=motion_id,
                music_id=parsed["music"],
                genre=parsed["genre"],
                situation=parsed["situation"],
                motion_path=str(motion_path.relative_to(root)),
                duration_seconds=duration,
                keypoint_phases=tuple(float(v) for v in keypoints.phases),
                keypoint_scores=tuple(float(v) for v in keypoints.scores),
                keypoint_density_hz=float(len(keypoints.phases) / max(duration, 1e-6)),
                weighted_keypoint_density=float(
                    sum(keypoints.scores) / max(duration, 1e-6)
                ),
                keypoint_interval_cv=_keypoint_interval_cv(keypoints.phases),
                velocity_median=float(np.median(smoothed)),
                velocity_p90=float(np.percentile(smoothed, 90.0)),
                original_bpm=bpm_by_music.get(parsed["music"], 0.0),
                preflight_passed=preflight_passed,
                preflight_reason=preflight_reason,
            )
            if not preflight_passed:
                failures.append({"motion_id": motion_id, "reason": preflight_reason})
                continue
            motions[motion_id] = profile
            tracks[parsed["music"]]["motion_ids"].append(motion_id)
        except Exception as exc:
            failures.append(
                {
                    "motion_id": motion_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        if index == 1 or index % 25 == 0 or index == len(valid_rows):
            print(f"Motion profiles: {index}/{len(valid_rows)}", flush=True)

    if not motions:
        raise RuntimeError("No motion passed profiling and preflight.")

    embeddings = np.stack([descriptor.embedding for descriptor in descriptors])
    rhythm = np.stack([descriptor.rhythm_timbre for descriptor in descriptors])
    tag_dimension = max((descriptor.tag_probabilities.size for descriptor in descriptors), default=0)
    tags = (
        np.stack([descriptor.tag_probabilities for descriptor in descriptors])
        if tag_dimension
        else np.empty((len(descriptors), 0), dtype=np.float32)
    )
    velocity_values = np.asarray(
        [profile.velocity_p90 for profile in motions.values()],
        dtype=float,
    )
    velocity_median = float(np.median(velocity_values))
    velocity_scale = float(
        max(
            np.percentile(velocity_values, 75.0)
            - np.percentile(velocity_values, 25.0),
            np.std(velocity_values),
            1e-6,
        )
    )
    density_values = np.asarray(
        [profile.weighted_keypoint_density for profile in motions.values()],
        dtype=float,
    )
    regularity_values = np.asarray(
        [profile.keypoint_interval_cv for profile in motions.values()],
        dtype=float,
    )
    velocity_cutoffs = np.quantile(velocity_values, (1.0 / 3.0, 2.0 / 3.0))
    density_cutoffs = np.quantile(density_values, (1.0 / 3.0, 2.0 / 3.0))
    regularity_cutoff = float(np.median(regularity_values))
    for motion_id, profile in tuple(motions.items()):
        velocity_tier = int(np.searchsorted(velocity_cutoffs, profile.velocity_p90))
        density_tier = int(
            np.searchsorted(density_cutoffs, profile.weighted_keypoint_density)
        )
        regularity_tier = int(profile.keypoint_interval_cv > regularity_cutoff)
        motions[motion_id] = replace(
            profile,
            motion_cluster_id=(
                f"{profile.genre}:v{velocity_tier}:k{density_tier}:r{regularity_tier}"
            ),
        )

    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / "catalog_features.npz"
    np.savez_compressed(
        arrays_path,
        embeddings=embeddings.astype(np.float32),
        rhythm_timbre=rhythm.astype(np.float32),
        tags=tags.astype(np.float32),
        embedding_mean=np.mean(embeddings, axis=0).astype(np.float32),
        embedding_std=np.std(embeddings, axis=0).astype(np.float32),
        rhythm_mean=np.mean(rhythm, axis=0).astype(np.float32),
        rhythm_std=np.std(rhythm, axis=0).astype(np.float32),
    )
    catalog_path = output / "catalog.json"
    model_metadata = extractor.model_metadata()
    for key in ("embedding_model", "tag_model"):
        model = model_metadata.get(key)
        if isinstance(model, dict) and model.get("path"):
            model["path"] = os.path.relpath(Path(model["path"]), output).replace("\\", "/")

    metadata = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "aistpp_root": os.path.relpath(root, output).replace("\\", "/"),
        "manifest": str(manifest_path.relative_to(root)),
        "arrays_file": arrays_path.name,
        "extractor": {
            "sample_rate": extractor.sample_rate,
            "window_seconds": float(window_seconds),
            "hop_seconds": float(hop_seconds),
            **model_metadata,
        },
        "counts": {
            "manifest_rows": len(valid_rows),
            "music_ids": len(tracks),
            "audio_variants": len(variants),
            "segments": len(segment_metadata),
            "motions_included": len(motions),
            "motions_excluded": len(failures),
        },
        "motion_stats": {
            "velocity_median": velocity_median,
            "velocity_scale": velocity_scale,
            "cluster_policy": "genre_velocity_density_regularity_v1",
            "velocity_cutoffs": [float(value) for value in velocity_cutoffs],
            "density_cutoffs": [float(value) for value in density_cutoffs],
            "regularity_cutoff": regularity_cutoff,
        },
        "tracks": tracks,
        "motions": {key: value.to_dict() for key, value in motions.items()},
        "segments": segment_metadata,
    }
    catalog_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    excluded_path = output / "excluded_motions.csv"
    with excluded_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("motion_id", "reason"))
        writer.writeheader()
        writer.writerows(failures)
    return catalog_path
