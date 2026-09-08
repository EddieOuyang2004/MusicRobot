"""Reusable metrics for the Humanoid Matcher experiment suite.

The functions in this module deliberately separate literature-domain motion
features (SMPL/AIST++) from robot-domain diagnostics (Unitree G1 traces).  In
particular, callers must supply the kinetic/geometric features produced by the
chosen feature extractor; this module never pretends that G1 joint vectors are
directly comparable with published AIST++ FID numbers.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPERIMENT_SCHEMA_VERSION = 1


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    return array[np.isfinite(array)]


def distribution_summary(values: Iterable[float]) -> dict[str, float | int]:
    """Return the common latency/quality distribution summary."""

    array = _finite(values)
    if not array.size:
        return {
            "samples": 0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "iqr": 0.0,
        }
    q25, q50, q75, q95, q99 = np.percentile(array, (25, 50, 75, 95, 99))
    return {
        "samples": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p50": float(q50),
        "p95": float(q95),
        "p99": float(q99),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "iqr": float(q75 - q25),
    }


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    iterations: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap a confidence interval over independent run/music aggregates."""

    array = _finite(values)
    if not array.size:
        return {"samples": 0, "mean": 0.0, "low": 0.0, "high": 0.0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(max(int(iterations), 1), array.size), replace=True)
    means = np.mean(draws, axis=1)
    tail = (1.0 - float(np.clip(confidence, 0.0, 1.0))) / 2.0
    return {
        "samples": int(array.size),
        "mean": float(np.mean(array)),
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1.0 - tail)),
    }


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def ranking_metrics(
    expected_genres: Sequence[str],
    ranked_genres: Sequence[Sequence[str]],
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Compute genre retrieval/classification metrics per independent query."""

    if len(expected_genres) != len(ranked_genres):
        raise ValueError("expected_genres and ranked_genres must have equal length")
    hits = {int(k): 0 for k in ks}
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_genre_hits: dict[str, dict[int, int]] = defaultdict(
        lambda: {int(k): 0 for k in ks}
    )
    per_genre_count: Counter[str] = Counter()

    for expected, ranking in zip(expected_genres, ranked_genres, strict=True):
        expected = str(expected)
        unique_ranking = _unique(ranking)
        per_genre_count[expected] += 1
        prediction = unique_ranking[0] if unique_ranking else "__missing__"
        confusion[expected][prediction] += 1
        rank = (
            unique_ranking.index(expected) + 1
            if expected in unique_ranking
            else None
        )
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
        ndcg_values.append(
            1.0 / math.log2(rank + 1.0)
            if rank is not None and rank <= 5
            else 0.0
        )
        for k in hits:
            hit = expected in unique_ranking[:k]
            hits[k] += int(hit)
            per_genre_hits[expected][k] += int(hit)

    labels = sorted(set(expected_genres))
    f1_values: list[float] = []
    per_genre: dict[str, Any] = {}
    for label in labels:
        true_positive = confusion[label][label]
        false_negative = sum(confusion[label].values()) - true_positive
        false_positive = sum(
            predictions[label]
            for actual, predictions in confusion.items()
            if actual != label
        )
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        count = max(per_genre_count[label], 1)
        per_genre[label] = {
            "queries": int(per_genre_count[label]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            **{
                f"recall_at_{k}": per_genre_hits[label][k] / count
                for k in hits
            },
        }

    count = max(len(expected_genres), 1)
    return {
        "queries": len(expected_genres),
        **{f"recall_at_{k}": hits[k] / count for k in hits},
        "macro_f1": float(statistics.fmean(f1_values)) if f1_values else 0.0,
        "mean_reciprocal_rank": (
            float(statistics.fmean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        ),
        "ndcg_at_5": float(statistics.fmean(ndcg_values)) if ndcg_values else 0.0,
        "per_genre": per_genre,
        "confusion": {
            actual: dict(predictions)
            for actual, predictions in sorted(confusion.items())
        },
    }


def binary_rejection_metrics(
    expected_reject: Sequence[bool],
    predicted_reject: Sequence[bool],
    negative_style_scores: Sequence[float] | None = None,
) -> dict[str, float | int | None]:
    """Compute rejection precision/recall/F1 and score AUROC."""

    if len(expected_reject) != len(predicted_reject):
        raise ValueError("rejection label and prediction lengths differ")
    tp = sum(bool(y) and bool(p) for y, p in zip(expected_reject, predicted_reject))
    fp = sum(not bool(y) and bool(p) for y, p in zip(expected_reject, predicted_reject))
    fn = sum(bool(y) and not bool(p) for y, p in zip(expected_reject, predicted_reject))
    tn = len(expected_reject) - tp - fp - fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    auroc: float | None = None
    if negative_style_scores is not None:
        if len(negative_style_scores) != len(expected_reject):
            raise ValueError("negative_style_scores length differs from labels")
        positives = [
            float(score)
            for score, label in zip(negative_style_scores, expected_reject)
            if label
        ]
        negatives = [
            float(score)
            for score, label in zip(negative_style_scores, expected_reject)
            if not label
        ]
        if positives and negatives:
            wins = sum(
                1.0 if positive > negative else 0.5 if positive == negative else 0.0
                for positive in positives
                for negative in negatives
            )
            auroc = wins / (len(positives) * len(negatives))
    return {
        "samples": len(expected_reject),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc,
        "false_reject_rate": fp / max(fp + tn, 1),
        "missed_reject_rate": fn / max(fn + tp, 1),
    }


def beat_alignment_score(
    source_beats_seconds: Sequence[float],
    target_beats_seconds: Sequence[float],
    *,
    sigma_seconds: float = 3.0 / 60.0,
) -> float:
    """Gaussian nearest-beat alignment, directed source -> target."""

    source = _finite(source_beats_seconds)
    target = _finite(target_beats_seconds)
    if not source.size or not target.size:
        return 0.0
    distances = np.min(np.abs(source[:, None] - target[None, :]), axis=1)
    sigma = max(float(sigma_seconds), 1e-9)
    return float(np.mean(np.exp(-np.square(distances) / (2.0 * sigma * sigma))))


def bidirectional_beat_metrics(
    music_beats_seconds: Sequence[float],
    dance_beats_seconds: Sequence[float],
    *,
    sigma_seconds: float = 3.0 / 60.0,
) -> dict[str, float | int]:
    """Return Bailando-direction BAS, reverse BAS, harmonic BAS and errors."""

    music = _finite(music_beats_seconds)
    dance = _finite(dance_beats_seconds)
    forward = beat_alignment_score(music, dance, sigma_seconds=sigma_seconds)
    reverse = beat_alignment_score(dance, music, sigma_seconds=sigma_seconds)
    harmonic = 2.0 * forward * reverse / max(forward + reverse, 1e-12)
    if music.size and dance.size:
        music_error = np.min(np.abs(music[:, None] - dance[None, :]), axis=1)
        dance_error = np.min(np.abs(dance[:, None] - music[None, :]), axis=1)
    else:
        music_error = np.asarray([], dtype=np.float64)
        dance_error = np.asarray([], dtype=np.float64)
    tolerance = 2.0 * max(float(sigma_seconds), 1e-9)
    return {
        "music_beats": int(music.size),
        "dance_beats": int(dance.size),
        "bas_music_to_dance": forward,
        "bas_dance_to_music": reverse,
        "bas_harmonic": harmonic,
        "beat_timing_mae_seconds": float(np.mean(music_error)) if music_error.size else 0.0,
        "beat_timing_p95_seconds": (
            float(np.percentile(music_error, 95.0)) if music_error.size else 0.0
        ),
        "missed_music_beat_rate": (
            float(np.mean(music_error > tolerance)) if music_error.size else 0.0
        ),
        "extra_dance_beat_rate": (
            float(np.mean(dance_error > tolerance)) if dance_error.size else 0.0
        ),
    }


def detect_kinematic_beats(
    joint_positions: np.ndarray,
    *,
    fps: float,
    minimum_separation_seconds: float = 0.20,
) -> np.ndarray:
    """Detect local minima of root-relative whole-body kinetic velocity."""

    joints = np.asarray(joint_positions, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[0] < 3 or joints.shape[-1] != 3:
        return np.asarray([], dtype=np.float64)
    root_relative = joints - joints[:, :1, :]
    velocity = np.linalg.norm(np.diff(root_relative, axis=0) * float(fps), axis=-1)
    kinetic = np.mean(velocity, axis=1)
    minima = np.flatnonzero(
        (kinetic[1:-1] <= kinetic[:-2]) & (kinetic[1:-1] < kinetic[2:])
    ) + 1
    if not minima.size:
        return np.asarray([], dtype=np.float64)
    separation = max(int(round(minimum_separation_seconds * fps)), 1)
    selected: list[int] = []
    for index in minima:
        if not selected or index - selected[-1] >= separation:
            selected.append(int(index))
        elif kinetic[index] < kinetic[selected[-1]]:
            selected[-1] = int(index)
    return np.asarray(selected, dtype=np.float64) / float(fps)


def diversity(features: np.ndarray) -> float:
    """Mean distance over all unordered feature-vector pairs."""

    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        return 0.0
    delta = array[:, None, :] - array[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    upper = np.triu_indices(array.shape[0], 1)
    return float(np.mean(distances[upper]))


def frechet_distance(real_features: np.ndarray, output_features: np.ndarray) -> float:
    """Stable Fr\u00e9chet distance between two feature distributions."""

    real = np.asarray(real_features, dtype=np.float64)
    output = np.asarray(output_features, dtype=np.float64)
    if real.ndim != 2 or output.ndim != 2 or real.shape[1:] != output.shape[1:]:
        raise ValueError("feature arrays must be 2-D with matching widths")
    if real.shape[0] < 2 or output.shape[0] < 2:
        return 0.0
    mean_real = np.mean(real, axis=0)
    mean_output = np.mean(output, axis=0)
    covariance_real = np.cov(real, rowvar=False)
    covariance_output = np.cov(output, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_real)
    sqrt_real = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    middle = sqrt_real @ covariance_output @ sqrt_real
    middle = 0.5 * (middle + middle.T)
    middle_eigenvalues = np.linalg.eigvalsh(middle)
    covariance_trace = float(np.sum(np.sqrt(np.maximum(middle_eigenvalues, 0.0))))
    mean_term = float(np.dot(mean_real - mean_output, mean_real - mean_output))
    return max(
        mean_term
        + float(np.trace(covariance_real))
        + float(np.trace(covariance_output))
        - 2.0 * covariance_trace,
        0.0,
    )


def literature_feature_metrics(
    real_kinetic: np.ndarray,
    output_kinetic: np.ndarray,
    real_geometric: np.ndarray,
    output_geometric: np.ndarray,
    *,
    extractor_identity: str,
) -> dict[str, Any]:
    """Compute FID/Div while retaining the extractor identity in the result."""

    if not extractor_identity.strip():
        raise ValueError("extractor_identity is required for comparable FID/Div")
    return {
        "feature_extractor": extractor_identity,
        "fid_k": frechet_distance(real_kinetic, output_kinetic),
        "fid_g": frechet_distance(real_geometric, output_geometric),
        "div_k": diversity(output_kinetic),
        "div_g": diversity(output_geometric),
        "ground_truth_div_k": diversity(real_kinetic),
        "ground_truth_div_g": diversity(real_geometric),
    }


def physical_foot_contact(
    center_of_mass: np.ndarray,
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    *,
    fps: float,
) -> float:
    """Legacy COM/foot-speed proxy; NOT the official EDGE PFC definition."""

    com = np.asarray(center_of_mass, dtype=np.float64)
    feet = np.stack((left_foot, right_foot), axis=1).astype(np.float64)
    if com.ndim != 2 or com.shape[0] < 3 or feet.shape[0] != com.shape[0]:
        return 0.0
    acceleration = np.diff(com, n=2, axis=0) * float(fps) ** 2
    foot_velocity = np.linalg.norm(np.diff(feet, axis=0) * float(fps), axis=-1)
    foot_velocity = foot_velocity[1:]
    minimum_foot_speed = np.min(foot_velocity, axis=1)
    horizontal_acceleration = np.linalg.norm(acceleration[:, :2], axis=1)
    positive_vertical = np.maximum(acceleration[:, 2], 0.0)
    acceleration_scale = np.linalg.norm(acceleration, axis=1)
    weight = (horizontal_acceleration + positive_vertical) / np.maximum(
        acceleration_scale, 1e-8
    )
    return float(np.mean(weight * minimum_foot_speed))


def edge_pfc_g1_adapted(root: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """EDGE eval_pfc.py algebra on a 30 FPS grid, with one G1 anchor per foot.

    Reference: https://github.com/Stanford-TML/EDGE/blob/main/eval/eval_pfc.py
    Unlike the SMPL evaluator's ankle/toe minima, G1 provides one body origin
    per side. This is explicitly an adaptation, not directly comparable PFC.
    """
    acceleration = np.diff(root, n=2, axis=0).copy()
    acceleration[:, 2] = np.maximum(acceleration[:, 2], 0)
    magnitude = np.linalg.norm(acceleration, axis=1)
    if not magnitude.size or np.max(magnitude) <= 1e-12:
        return 0.0  # Explicit extension for the official script's 0/0 case.
    left_step = np.linalg.norm(np.diff(left[:, :2], axis=0)[1:], axis=1)
    right_step = np.linalg.norm(np.diff(right[:, :2], axis=0)[1:], axis=1)
    return float(np.mean(left_step * right_step * magnitude / np.max(magnitude)) * 10000)


def timed_derivatives(values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, ...]:
    """Forward differences at midpoint clocks, with no nominal-rate substitution."""
    if len(times) < 4 or np.any(np.diff(times) <= 0):
        raise ValueError("At least four strictly increasing actual timestamps required")
    result = []
    for _ in range(3):
        values = np.diff(values, axis=0) / np.diff(times)[:, None]
        times = (times[1:] + times[:-1]) / 2
        result.append(values)
    return tuple(result)


def foot_skating_metrics(
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    *,
    fps: float,
    ground_height: float = 0.0,
    contact_height_m: float = 0.05,
    skating_speed_m_s: float = 0.05,
) -> dict[str, float | int]:
    """Measure horizontal foot motion while either foot is near the ground."""

    feet = np.stack((left_foot, right_foot), axis=1).astype(np.float64)
    if feet.ndim != 3 or feet.shape[0] < 2:
        return {"contact_frames": 0, "fsr": 0.0, "contact_speed_mean_m_s": 0.0}
    horizontal_speed = np.linalg.norm(
        np.diff(feet[..., :2], axis=0) * float(fps), axis=-1
    )
    contact = feet[1:, :, 2] <= float(ground_height + contact_height_m)
    contact_speeds = horizontal_speed[contact]
    return {
        "contact_frames": int(np.sum(contact)),
        "fsr": (
            float(np.mean(contact_speeds > skating_speed_m_s))
            if contact_speeds.size
            else 0.0
        ),
        "contact_speed_mean_m_s": (
            float(np.mean(contact_speeds)) if contact_speeds.size else 0.0
        ),
        "contact_speed_p95_m_s": (
            float(np.percentile(contact_speeds, 95.0)) if contact_speeds.size else 0.0
        ),
    }


def selection_diversity(
    motion_ids: Sequence[str],
    cluster_by_motion: Mapping[str, str] | None = None,
    scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Coverage, entropy, repetition and optional relevance for selections."""

    ids = [str(value) for value in motion_ids if str(value)]
    counts = Counter(ids)
    total = max(len(ids), 1)
    probabilities = np.asarray([count / total for count in counts.values()], dtype=float)
    entropy = float(-np.sum(probabilities * np.log(probabilities))) if counts else 0.0
    normalized_entropy = entropy / max(math.log(len(counts)), 1e-12) if len(counts) > 1 else 0.0
    repeats = sum(left == right for left, right in zip(ids, ids[1:]))
    clusters = {
        cluster_by_motion.get(motion_id, "")
        for motion_id in ids
        if cluster_by_motion and cluster_by_motion.get(motion_id, "")
    }
    cluster_universe = {
        str(value) for value in (cluster_by_motion or {}).values() if str(value)
    }
    return {
        "selections": len(ids),
        "unique_motions": len(counts),
        "unique_clusters": len(clusters),
        "visual_cluster_coverage": (
            len(clusters) / len(cluster_universe) if cluster_universe else None
        ),
        "normalized_selection_entropy": normalized_entropy,
        "dominant_motion_share": max(counts.values(), default=0) / total,
        "consecutive_repetition_rate": repeats / max(len(ids) - 1, 1),
        "mean_match_score": (
            float(np.mean(_finite(scores))) if scores is not None and _finite(scores).size else None
        ),
        "motion_counts": dict(counts),
    }


def _float(row: Mapping[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def aggregate_trace_csv(
    path: Path | str,
    cluster_by_motion: Mapping[str, str] | None = None,
    *,
    start_seconds: float = 0.0,
) -> dict[str, Any]:
    """Aggregate a production matcher trace without treating frames as IID."""

    trace_path = Path(path)
    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in all_rows
        if (_float(row, "audio_time_seconds") or 0.0) >= float(start_seconds)
    ]
    switches = [row for row in rows if row.get("event") == "switch_start"]
    completed = [row for row in rows if row.get("event") == "switch_complete"]
    motions = [row.get("current_motion_id", "") for row in rows]
    collapsed_motions = [
        motion
        for index, motion in enumerate(motions)
        if motion and (index == 0 or motion != motions[index - 1])
    ]

    def values(key: str, source: Sequence[Mapping[str, str]] = rows) -> list[float]:
        return [value for row in source if (value := _float(row, key)) is not None]

    return {
        "trace": str(trace_path),
        "evaluation_start_seconds": float(start_seconds),
        "rows": len(rows),
        "duration_seconds": max(values("audio_time_seconds"), default=0.0),
        "matches": sum(row.get("event") == "match" for row in rows),
        "switch_starts": len(switches),
        "switch_completions": len(completed),
        "hold_last_events": sum(row.get("event") == "hold_last" for row in rows),
        "collision_overrides": sum(row.get("collision_override") == "1" for row in rows),
        "selection": selection_diversity(collapsed_motions, cluster_by_motion),
        "match_score": distribution_summary(values("top_motion_score")),
        "transition": {
            "joint_speed_rad_s": distribution_summary(values("output_max_joint_speed_rad_s")),
            "joint_acceleration_rad_s2": distribution_summary(
                values("output_max_joint_acceleration_rad_s2")
            ),
            "anchor_xy_delta_m": distribution_summary(
                values("switch_anchor_xy_delta_m", switches)
            ),
            "anchor_z_delta_m": distribution_summary(
                values("switch_anchor_z_delta_m", switches)
            ),
            "anchor_yaw_delta_rad": distribution_summary(
                values("switch_anchor_yaw_delta_rad", switches)
            ),
            "post_switch_xy_delta_m": distribution_summary(
                values("post_switch_xy_delta_m")
            ),
            "post_switch_z_delta_m": distribution_summary(
                values("post_switch_z_delta_m")
            ),
            "post_switch_yaw_delta_rad": distribution_summary(
                values("post_switch_yaw_delta_rad")
            ),
            "entry_total_score": distribution_summary(values("entry_total_score", switches)),
            "entry_pose_score": distribution_summary(values("entry_pose_score", switches)),
            "entry_velocity_score": distribution_summary(
                values("entry_velocity_score", switches)
            ),
            "entry_contact_score": distribution_summary(
                values("entry_contact_score", switches)
            ),
            "entry_root_score": distribution_summary(values("entry_root_score", switches)),
            "entry_music_score": distribution_summary(values("entry_music_score", switches)),
        },
        "beat": {
            "accepted_beat_pose_error_rad": distribution_summary(
                values("accepted_beat_pose_error_rad")
            ),
            "accepted_beat_delay_seconds": distribution_summary(
                values("accepted_beat_delay_seconds")
            ),
        },
        "limiter": {
            "activation_rate": distribution_summary(values("limiter_activation_rate")),
            "activated_row_rate": (
                sum(bool(row.get("limiter_activated_joints")) for row in rows)
                / max(len(rows), 1)
            ),
        },
    }


def aggregate_pose_npz(
    path: Path | str,
    *,
    evaluation_fps: float = 60.0,
    cluster_by_motion: Mapping[str, str] | None = None,
    start_seconds: float = 0.0,
) -> dict[str, Any]:
    """Compute robot-domain rhythm, contact, continuity and diversity metrics."""

    pose_path = Path(path)
    with np.load(pose_path, allow_pickle=False) as archive:
        required = {
            "time_seconds",
            "joint_positions",
            "body_positions",
            "center_of_mass",
            "left_foot_position",
            "right_foot_position",
            "left_foot_support_height",
            "right_foot_support_height",
            "motion_ids",
            "accepted_causal_beat_times_seconds",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Pose NPZ is missing fields: {missing}")
        times = np.asarray(archive["time_seconds"], dtype=np.float64)
        joints = np.asarray(archive["joint_positions"], dtype=np.float64)
        bodies = np.asarray(archive["body_positions"], dtype=np.float64)
        # MuJoCo body 0 is the stationary world, not the robot root.
        # The detector subtracts body 0, so pass pelvis-first model bodies.
        if "body_names" in archive and str(archive["body_names"][0]) == "world":
            bodies = bodies[:, 1:]
        com = np.asarray(archive["center_of_mass"], dtype=np.float64)
        left = np.asarray(archive["left_foot_position"], dtype=np.float64)
        right = np.asarray(archive["right_foot_position"], dtype=np.float64)
        left_height = np.asarray(archive["left_foot_support_height"], dtype=np.float64)
        right_height = np.asarray(archive["right_foot_support_height"], dtype=np.float64)
        motion_ids = [str(value) for value in archive["motion_ids"]]
        music_beats = np.asarray(
            archive["accepted_causal_beat_times_seconds"], dtype=np.float64
        )
        source_rate = float(archive["control_rate_hz"].item())
        actual_clock = int(archive["schema_version"].item()) >= 2 if "schema_version" in archive else False
        root = com.copy()
        if actual_clock:
            times = np.asarray(archive["wall_time_seconds"], dtype=np.float64)
            if np.any(np.diff(times) <= 0):
                raise ValueError("Non-monotonic output wall clock")
            events = np.asarray(archive["beat_events"], dtype=np.float64).reshape(-1, 4)
            music_beats = events[:, 1]  # Actual delivery, not backdated beat peaks.
            root = np.asarray(archive["final_qpos"], dtype=np.float64)[:, :3]

    keep = times >= float(start_seconds)
    times = times[keep]
    joints = joints[keep]
    bodies = bodies[keep]
    com = com[keep]
    root = root[keep]
    left = left[keep]
    right = right[keep]
    left_height = left_height[keep]
    right_height = right_height[keep]
    motion_ids = [motion_id for motion_id, retained in zip(motion_ids, keep) if retained]
    music_beats = music_beats[music_beats >= float(start_seconds)]

    if times.size < 2:
        return {
            "pose_npz": str(pose_path),
            "evaluation_start_seconds": float(start_seconds),
            "frames": int(times.size),
            "duration_seconds": 0.0,
            "beat": bidirectional_beat_metrics([], []),
            "selection": selection_diversity(motion_ids, cluster_by_motion),
        }

    # Literature BAS is evaluated at 60 FPS.  Resample every continuous
    # channel onto that grid rather than relying on a 120->60 stride.
    output_times = np.arange(times[0], times[-1] + 1e-9, 1.0 / evaluation_fps)

    def resample(array: np.ndarray) -> np.ndarray:
        flat = np.asarray(array, dtype=np.float64).reshape(len(times), -1)
        output = np.column_stack(
            [np.interp(output_times, times, flat[:, column]) for column in range(flat.shape[1])]
        )
        return output.reshape((len(output_times), *array.shape[1:]))

    bodies_60 = resample(bodies)
    dance_beats = detect_kinematic_beats(bodies_60, fps=evaluation_fps)
    # Convert detected relative times back to the original audio clock.
    dance_beats = dance_beats + output_times[0]
    dt = 1.0 / max(source_rate, 1e-9)
    joint_velocity = np.diff(joints, axis=0) / dt
    joint_acceleration = np.diff(joint_velocity, axis=0) / dt
    joint_jerk = np.diff(joint_acceleration, axis=0) / dt
    if actual_clock:
        joint_velocity, joint_acceleration, joint_jerk = timed_derivatives(joints, times)
    # The recorded body origins sit above the sole.  Contact and skating must
    # therefore use the model-derived support heights rather than body-origin Z.
    contact_left = left.copy()
    contact_right = right.copy()
    contact_left[:, 2] = left_height
    contact_right[:, 2] = right_height
    skating = foot_skating_metrics(
        resample(contact_left) if actual_clock else contact_left,
        resample(contact_right) if actual_clock else contact_right,
        fps=evaluation_fps if actual_clock else source_rate)
    pfc_grid = np.arange(times[0], times[-1], 1 / 30)
    def at_30(array: np.ndarray) -> np.ndarray:
        return np.column_stack([np.interp(pfc_grid, times, array[:, j]) for j in range(3)])
    beat_metrics = bidirectional_beat_metrics(music_beats, dance_beats)
    if not len(music_beats):
        for key in ("bas_music_to_dance", "bas_dance_to_music", "bas_harmonic",
                    "beat_timing_mae_seconds", "beat_timing_p95_seconds",
                    "missed_music_beat_rate", "extra_dance_beat_rate"):
            beat_metrics[key] = None
    support_height = np.minimum(left_height, right_height)
    collapsed_motion_ids = [
        motion_id
        for index, motion_id in enumerate(motion_ids)
        if motion_id and (index == 0 or motion_id != motion_ids[index - 1])
    ]
    return {
        "pose_npz": str(pose_path),
        "evaluation_start_seconds": float(start_seconds),
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "kinematic_preview_only": True,
        "frames": int(times.size),
        "duration_seconds": float(times[-1] - times[0]),
        "source_control_rate_hz": source_rate,
        "evaluation_fps": float(evaluation_fps),
        "time_basis": "actual_monotonic_wall" if actual_clock else "audio_blocks; derivatives_nominal_only; actual_time_unrecoverable",
        "beat_input_mode": "causal_delivery" if actual_clock else "preanalysed_replay",
        "beat": beat_metrics,
        "selection": selection_diversity(collapsed_motion_ids, cluster_by_motion),
        "physical": {
            "pfc_legacy_proxy": physical_foot_contact(com, left, right, fps=source_rate) if not actual_clock else None,
            "pfc_edge_g1_adapted_30fps": edge_pfc_g1_adapted(at_30(root), at_30(left), at_30(right)) if actual_clock else None,
            **skating,
            "minimum_foot_support_height_m": float(np.min(support_height)),
            "ground_penetration_max_m": float(max(-np.min(support_height), 0.0)),
            "ground_penetration_frame_rate": float(np.mean(support_height < -1e-6)),
        },
        "continuity": {
            "joint_speed_rad_s": distribution_summary(
                np.max(np.abs(joint_velocity), axis=1)
            ),
            "joint_acceleration_rad_s2": distribution_summary(
                np.max(np.abs(joint_acceleration), axis=1)
            ),
            "joint_jerk_rad_s3": distribution_summary(
                np.max(np.abs(joint_jerk), axis=1)
            ),
        },
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm family-wise adjustment for named p-values."""

    ordered = sorted((float(value), key) for key, value in p_values.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (value, key) in enumerate(ordered):
        running = max(running, min((count - rank) * value, 1.0))
        adjusted[key] = running
    return adjusted


def paired_permutation_test(
    first: Sequence[float],
    second: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Two-sided paired sign-flip permutation test over independent runs."""

    if len(first) != len(second):
        raise ValueError("paired samples must have equal length")
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    differences = left[valid] - right[valid]
    if not differences.size:
        return {"pairs": 0, "mean_difference": 0.0, "p_value": 1.0}
    observed = abs(float(np.mean(differences)))
    if differences.size <= 16:
        masks = np.arange(1 << differences.size, dtype=np.uint64)[:, None]
        bits = (masks >> np.arange(differences.size, dtype=np.uint64)) & 1
        signs = np.where(bits, 1.0, -1.0)
        permuted = np.abs(np.mean(signs * differences[None, :], axis=1))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(
            (-1.0, 1.0),
            size=(max(int(iterations), 1), differences.size),
        )
        permuted = np.abs(np.mean(signs * differences[None, :], axis=1))
    p_value = (float(np.sum(permuted >= observed - 1e-15)) + 1.0) / (
        len(permuted) + 1.0
    )
    return {
        "pairs": int(differences.size),
        "mean_difference": float(np.mean(differences)),
        "p_value": min(p_value, 1.0),
    }


def response_latency_from_trace(
    path: Path | str,
    change_times_seconds: Sequence[float],
    expected_genres: Sequence[str] | None = None,
    *, clock_field: str = "audio_time_seconds",
) -> dict[str, Any]:
    """Measure audio-change -> match/pending/switch/blend/stable milestones."""

    trace_path = Path(path)
    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if expected_genres is not None and len(expected_genres) != len(change_times_seconds):
        raise ValueError("expected_genres must match change_times_seconds")

    def time_of(row: Mapping[str, str]) -> float:
        return float(row.get(clock_field, "nan"))

    changes = []
    for index, change_time in enumerate(change_times_seconds):
        expected = expected_genres[index] if expected_genres is not None else None
        next_change = change_times_seconds[index + 1] if index + 1 < len(change_times_seconds) else math.inf
        eligible = [row for row in rows if float(change_time) <= time_of(row) < next_change]
        if expected:
            matching_results = [
                row
                for row in eligible
                if row.get("event") == "match" and row.get("top_genre") == expected
            ]
        else:
            matching_results = [row for row in eligible if row.get("event") == "match"]
        first_match = matching_results[0] if matching_results else None
        match_time = time_of(first_match) if first_match is not None else None
        after_match = [
            row
            for row in eligible
            if match_time is not None and time_of(row) >= match_time
        ]
        pending = next(
            (row for row in after_match if bool(row.get("pending_motion_id"))),
            None,
        )
        switch_start = next(
            (row for row in after_match if row.get("event") == "switch_start"),
            None,
        )
        blend_half = next(
            (
                row
                for row in after_match
                if (_float(row, "transition_blend") or 0.0) >= 0.5
            ),
            None,
        )
        switch_complete = next(
            (row for row in after_match if row.get("event") == "switch_complete"),
            None,
        )

        def latency(row: Mapping[str, str] | None) -> float | None:
            return time_of(row) - float(change_time) if row is not None else None

        changes.append(
            {
                "change_time_seconds": float(change_time),
                "expected_genre": expected,
                "first_match_seconds": latency(first_match),
                "pending_seconds": latency(pending),
                "switch_start_seconds": latency(switch_start),
                "blend_50_seconds": latency(blend_half),
                "stable_switch_seconds": latency(switch_complete),
            }
        )
    return {
        "trace": str(trace_path),
        "changes": changes,
        "summary": {
            key: distribution_summary(
                item[key]
                for item in changes
                if item[key] is not None
            )
            for key in (
                "first_match_seconds",
                "pending_seconds",
                "switch_start_seconds",
                "blend_50_seconds",
                "stable_switch_seconds",
            )
        },
    }
