from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


DEFAULT_FALLBACK_PHASES = (0.0, 0.5)
HIGH_PRIORITY_TERMS = ("shoulder", "elbow", "wrist", "waist", "ankle", "knee")
MEDIUM_PRIORITY_TERMS = ("hand", "arm", "torso", "hip", "neck", "head")


def default_keypoint_count(duration: float) -> int:
    """Return one keypoint per rounded second of authored motion."""
    return max(1, int(max(duration, 0.0) + 0.5))


@dataclass(frozen=True)
class KeypointDetectionResult:
    phases: tuple[float, ...]
    scores: tuple[float, ...]
    fallback_used: bool
    reason: str


def parse_keypoint_phases(value: str) -> tuple[float, ...]:
    phases = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        phase = float(item) % 1.0
        phases.append(phase)
    return tuple(sorted(_dedupe_phases(phases)))


def sample_pose_sequence(
    sampler: object,
    duration: float,
    features: object,
    sample_rate: float = 120.0,
    min_samples: int = 96,
    max_samples: int = 1024,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    sample_count = int(np.clip(round(max(duration, 1e-6) * sample_rate), min_samples, max_samples))
    phases = np.arange(sample_count, dtype=float) / sample_count
    poses = [
        _finite_pose(sampler.sample(float(phase), amplitude=1.0, accent=0.0, features=features))
        for phase in phases
    ]
    return phases, poses


def detect_motion_keypoints(
    phases: Sequence[float],
    poses: Sequence[Mapping[str, float]],
    duration: float,
    min_spacing_sec: float = 0.28,
    prominence: float = 0.18,
    max_count: int = 32,
    fallback_phases: tuple[float, ...] = DEFAULT_FALLBACK_PHASES,
) -> KeypointDetectionResult:
    phases_array = np.asarray(phases, dtype=float)
    if phases_array.ndim != 1 or phases_array.size < 4 or len(poses) != phases_array.size:
        return _fallback(fallback_phases, "not enough sampled poses")

    matrix, joint_names = pose_matrix(poses)
    if matrix.shape[0] < 4 or matrix.shape[1] == 0:
        return _fallback(fallback_phases, "no varying pose channels")

    salience = motion_salience(matrix, joint_names)
    if salience.size == 0 or float(np.max(salience)) <= 1e-9:
        return _fallback(fallback_phases, "motion is too flat")

    min_spacing_frames = _spacing_frames(
        frame_count=phases_array.size,
        duration=duration,
        min_spacing_sec=min_spacing_sec,
    )
    peaks = select_salient_peaks(
        salience=salience,
        phases=phases_array,
        min_spacing_frames=min_spacing_frames,
        prominence=prominence,
        max_count=max_count,
    )
    if len(peaks) < 2:
        return _fallback(fallback_phases, "fewer than two motion keypoints detected")

    peaks.sort(key=lambda item: item[0])
    return KeypointDetectionResult(
        phases=tuple(float(phase) for phase, _score in peaks),
        scores=tuple(float(score) for _phase, score in peaks),
        fallback_used=False,
        reason="ok",
    )


def pose_matrix(poses: Sequence[Mapping[str, float]]) -> tuple[np.ndarray, tuple[str, ...]]:
    names = sorted({name for pose in poses for name in pose})
    if not names:
        return np.empty((len(poses), 0), dtype=float), ()

    matrix = np.zeros((len(poses), len(names)), dtype=float)
    for row, pose in enumerate(poses):
        for col, name in enumerate(names):
            matrix[row, col] = _finite_float(pose.get(name, 0.0))

    scale = np.std(matrix, axis=0)
    active = scale > 1e-6
    if not np.any(active):
        return np.empty((len(poses), 0), dtype=float), ()

    matrix = matrix[:, active]
    active_names = tuple(name for name, keep in zip(names, active) if keep)
    center = np.median(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    normalized = (matrix - center) / np.maximum(scale, 1e-6)
    weights = np.asarray([joint_weight(name) for name in active_names], dtype=float)
    return normalized * weights, active_names


def motion_salience(matrix: np.ndarray, joint_names: Sequence[str]) -> np.ndarray:
    del joint_names
    if matrix.shape[0] < 4 or matrix.shape[1] == 0:
        return np.empty(0, dtype=float)

    prev_frame = np.roll(matrix, 1, axis=0)
    next_frame = np.roll(matrix, -1, axis=0)
    incoming = matrix - prev_frame
    outgoing = next_frame - matrix
    acceleration = next_frame - 2.0 * matrix + prev_frame

    accel_score = np.linalg.norm(acceleration, axis=1)
    extremity_score = np.linalg.norm(matrix, axis=1)
    incoming_norm = np.linalg.norm(incoming, axis=1)
    outgoing_norm = np.linalg.norm(outgoing, axis=1)
    dot = np.sum(incoming * outgoing, axis=1)
    cosine = dot / np.maximum(incoming_norm * outgoing_norm, 1e-9)
    turn_score = np.maximum(-cosine, 0.0) * np.minimum(incoming_norm, outgoing_norm)

    salience = (
        0.45 * _unit_range(accel_score)
        + 0.35 * _unit_range(turn_score)
        + 0.20 * _unit_range(extremity_score)
    )
    return _smooth_circular(salience)


def select_salient_peaks(
    salience: np.ndarray,
    phases: np.ndarray,
    min_spacing_frames: int,
    prominence: float,
    max_count: int,
) -> list[tuple[float, float]]:
    local_maxima = np.flatnonzero(
        (salience > np.roll(salience, 1))
        & (salience >= np.roll(salience, -1))
        & (salience >= prominence)
    )
    if local_maxima.size == 0:
        return []

    candidates = []
    radius = max(min_spacing_frames, 1)
    for idx in local_maxima:
        offsets = (np.arange(idx - radius, idx + radius + 1) % salience.size).astype(int)
        local_floor = float(np.min(salience[offsets]))
        peak_prominence = float(salience[idx] - local_floor)
        if peak_prominence >= prominence:
            candidates.append((int(idx), float(salience[idx])))

    candidates.sort(key=lambda item: item[1], reverse=True)
    selected: list[tuple[int, float]] = []
    for idx, score in candidates:
        if len(selected) >= max(1, max_count):
            break
        if all(_circular_frame_distance(idx, other_idx, salience.size) >= min_spacing_frames for other_idx, _ in selected):
            selected.append((idx, score))

    selected.sort(key=lambda item: item[0])
    return [(float(phases[idx] % 1.0), score) for idx, score in selected]


def joint_weight(name: str) -> float:
    lowered = name.lower()
    if any(term in lowered for term in HIGH_PRIORITY_TERMS):
        return 1.45
    if any(term in lowered for term in MEDIUM_PRIORITY_TERMS):
        return 1.15
    return 1.0


def _spacing_frames(frame_count: int, duration: float, min_spacing_sec: float) -> int:
    if duration <= 1e-6:
        return max(1, int(round(frame_count * 0.08)))
    return max(1, int(round(frame_count * max(min_spacing_sec, 0.0) / duration)))


def _fallback(fallback_phases: tuple[float, ...], reason: str) -> KeypointDetectionResult:
    phases = tuple(sorted(_dedupe_phases(fallback_phases)))
    scores = tuple(0.0 for _phase in phases)
    return KeypointDetectionResult(phases=phases, scores=scores, fallback_used=True, reason=reason)


def _dedupe_phases(phases: Sequence[float], epsilon: float = 1e-5) -> list[float]:
    result: list[float] = []
    for phase in sorted(float(phase) % 1.0 for phase in phases):
        if not result or abs(phase - result[-1]) > epsilon:
            result.append(phase)
    if len(result) > 1 and _phase_distance(result[0], result[-1]) <= epsilon:
        result.pop()
    return result


def _phase_distance(a: float, b: float) -> float:
    delta = abs((a - b) % 1.0)
    return min(delta, 1.0 - delta)


def _circular_frame_distance(a: int, b: int, frame_count: int) -> int:
    delta = abs(a - b)
    return min(delta, frame_count - delta)


def _smooth_circular(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values
    smoothed = 0.25 * np.roll(values, 1) + 0.50 * values + 0.25 * np.roll(values, -1)
    return _unit_range(smoothed)


def _unit_range(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.min(values)) if values.size else 0.0
    high = float(np.max(values)) if values.size else 0.0
    if not math.isfinite(low) or not math.isfinite(high) or high - low <= 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _finite_pose(pose: Mapping[str, float]) -> dict[str, float]:
    return {name: _finite_float(value) for name, value in pose.items()}


def _finite_float(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0
