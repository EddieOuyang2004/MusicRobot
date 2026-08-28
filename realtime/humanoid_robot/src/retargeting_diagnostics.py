from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from aistpp_smpl import Y_UP_TO_Z_UP, smpl_world_kinematics


@dataclass(frozen=True)
class DiagnosticThresholds:
    raw_axis_angle_jump_rad: float = float(np.pi)
    smpl_joint_speed_m_s: float = 15.0
    bvh_joint_speed_m_s: float = 15.0
    bvh_position_error_m: float = 1e-4
    g1_root_speed_m_s: float = 3.0 + 1e-6
    g1_root_angular_speed_rad_s: float = 4.0 * float(np.pi) + 1e-6
    g1_joint_speed_rad_s: float = 3.0 * float(np.pi) + 1e-6


@dataclass(frozen=True)
class PoseBoneSpec:
    name: str
    source_start: str
    source_end: str
    target_start: str
    target_end: str


def diagnose_pose_similarity(
    *,
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    source_names: Sequence[str],
    target_names: Sequence[str],
    bones: Sequence[PoseBoneSpec],
    median_threshold: float = 0.85,
    negative_fraction_threshold: float = 0.5,
) -> dict[str, object]:
    """Compare source/target bone directions without assuming equal limb lengths."""
    source = np.asarray(source_positions, dtype=np.float64)
    target = np.asarray(target_positions, dtype=np.float64)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"Expected source_positions[N,J,3], got {source.shape}.")
    if target.ndim != 3 or target.shape[2] != 3 or len(target) != len(source):
        raise ValueError(f"Expected target_positions[{len(source)},J,3], got {target.shape}.")
    source_index = {str(name): index for index, name in enumerate(source_names)}
    target_index = {str(name): index for index, name in enumerate(target_names)}
    if len(source_index) != len(source_names) or len(target_index) != len(target_names):
        raise ValueError("Pose similarity joint names must be unique.")

    metrics: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    events: list[dict[str, object]] = []
    for bone in bones:
        try:
            source_vector = (
                source[:, source_index[bone.source_end]] - source[:, source_index[bone.source_start]]
            )
            target_vector = (
                target[:, target_index[bone.target_end]] - target[:, target_index[bone.target_start]]
            )
        except KeyError as exc:
            raise ValueError(f"Pose bone {bone.name!r} references unknown joint {exc.args[0]!r}.") from exc
        source_norm = np.linalg.norm(source_vector, axis=1)
        target_norm = np.linalg.norm(target_vector, axis=1)
        degenerate = (source_norm <= 1e-9) | (target_norm <= 1e-9)
        denominators = np.maximum(source_norm * target_norm, 1e-18)
        cosine = np.sum(source_vector * target_vector, axis=1) / denominators
        cosine = np.clip(cosine, -1.0, 1.0)
        cosine[degenerate] = -1.0
        worst_frame = int(np.argmin(cosine))
        median = float(np.median(cosine))
        negative_fraction = float(np.mean(cosine < 0.0))
        failed = median <= median_threshold or negative_fraction >= negative_fraction_threshold
        metrics[bone.name] = {
            "median_cosine": median,
            "minimum_cosine": float(cosine[worst_frame]),
            "negative_frame_fraction": negative_fraction,
            "degenerate_frame_fraction": float(np.mean(degenerate)),
            "worst_frame": worst_frame,
            "failed": bool(failed),
        }
        if failed:
            failures.append(bone.name)
            events.append(
                {
                    "frame": worst_frame,
                    "classification": "pose_similarity",
                    "bone": bone.name,
                    "cosine": float(cosine[worst_frame]),
                    "median_cosine": median,
                    "negative_frame_fraction": negative_fraction,
                }
            )

    symmetry_pairs = {
        "thigh": ("left_thigh", "right_thigh"),
        "shin": ("left_shin", "right_shin"),
        "upper_arm": ("left_upper_arm", "right_upper_arm"),
        "forearm": ("left_forearm", "right_forearm"),
    }
    symmetry_quality_gap = {
        name: abs(
            float(metrics[left]["median_cosine"]) - float(metrics[right]["median_cosine"])
        )
        for name, (left, right) in symmetry_pairs.items()
        if left in metrics and right in metrics
    }
    worst_bone = min(metrics, key=lambda name: float(metrics[name]["median_cosine"]))
    return {
        "status": "failed" if failures else "ok",
        "thresholds": {
            "median_cosine_exclusive_minimum": float(median_threshold),
            "negative_frame_fraction_exclusive_maximum": float(negative_fraction_threshold),
        },
        "bones": metrics,
        "failed_bones": failures,
        "worst_bone": worst_bone,
        "worst_frame": int(metrics[worst_bone]["worst_frame"]),
        "left_right_quality_gap": symmetry_quality_gap,
        "events": events,
    }


def quaternion_angular_speed_wxyz(quaternions: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected quaternions[N,4], got {values.shape}.")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("Quaternion trajectory contains a zero quaternion.")
    normalized = values / norms[:, None]
    dots = np.abs(np.sum(normalized[1:] * normalized[:-1], axis=1))
    return 2.0 * np.arccos(np.clip(dots, -1.0, 1.0)) * float(fps)


def _max_vector_speed(values: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    deltas = np.linalg.norm(np.diff(np.asarray(values, dtype=np.float64), axis=0), axis=-1)
    if deltas.ndim == 1:
        return deltas * fps, np.zeros(len(deltas), dtype=np.int64)
    indices = np.argmax(deltas, axis=1)
    return deltas[np.arange(len(deltas)), indices] * fps, indices


def _max_scalar_speed(values: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    deltas = np.abs(np.diff(np.asarray(values, dtype=np.float64), axis=0))
    indices = np.argmax(deltas, axis=1)
    return deltas[np.arange(len(deltas)), indices] * fps, indices


def diagnose_continuity(
    *,
    raw_axis_angles: np.ndarray,
    smpl_positions: np.ndarray,
    bvh_positions: np.ndarray,
    g1_root_positions: np.ndarray,
    g1_root_quaternions_wxyz: np.ndarray,
    g1_dof_positions: np.ndarray,
    fps: float,
    smpl_joint_names: Sequence[str],
    dof_names: Sequence[str],
    thresholds: DiagnosticThresholds = DiagnosticThresholds(),
) -> dict[str, object]:
    """Classify representation-only changes separately from real layer discontinuities."""
    raw = np.asarray(raw_axis_angles, dtype=np.float64)
    smpl = np.asarray(smpl_positions, dtype=np.float64)
    bvh = np.asarray(bvh_positions, dtype=np.float64)
    root = np.asarray(g1_root_positions, dtype=np.float64)
    dofs = np.asarray(g1_dof_positions, dtype=np.float64)
    frame_count = len(raw)
    if frame_count < 2:
        raise ValueError("A diagnostic trajectory needs at least two frames.")
    expected = (frame_count, len(smpl_joint_names), 3)
    if smpl.shape != expected or bvh.shape != expected:
        raise ValueError(f"Expected selected skeleton arrays {expected}, got {smpl.shape}/{bvh.shape}.")
    if root.shape != (frame_count, 3) or dofs.shape != (frame_count, len(dof_names)):
        raise ValueError("G1 arrays do not match the source frame count.")

    raw_speed, raw_indices = _max_vector_speed(raw, fps)
    smpl_speed, smpl_indices = _max_vector_speed(smpl, fps)
    bvh_speed, bvh_indices = _max_vector_speed(bvh, fps)
    bvh_errors = np.linalg.norm(bvh - smpl, axis=2)
    bvh_error_indices = np.argmax(bvh_errors, axis=1)
    bvh_error = bvh_errors[np.arange(frame_count), bvh_error_indices]
    root_speed, _ = _max_vector_speed(root, fps)
    root_angular_speed = quaternion_angular_speed_wxyz(g1_root_quaternions_wxyz, fps)
    dof_speed, dof_indices = _max_scalar_speed(dofs, fps)

    candidate_frames: set[int] = set(
        (np.flatnonzero(raw_speed > thresholds.raw_axis_angle_jump_rad * fps) + 1).tolist()
    )
    candidate_frames.update((np.flatnonzero(smpl_speed > thresholds.smpl_joint_speed_m_s) + 1).tolist())
    candidate_frames.update((np.flatnonzero(bvh_speed > thresholds.bvh_joint_speed_m_s) + 1).tolist())
    candidate_frames.update(np.flatnonzero(bvh_error > thresholds.bvh_position_error_m).tolist())
    candidate_frames.update((np.flatnonzero(root_speed > thresholds.g1_root_speed_m_s) + 1).tolist())
    candidate_frames.update(
        (np.flatnonzero(root_angular_speed > thresholds.g1_root_angular_speed_rad_s) + 1).tolist()
    )
    candidate_frames.update((np.flatnonzero(dof_speed > thresholds.g1_joint_speed_rad_s) + 1).tolist())

    events: list[dict[str, object]] = []
    failure_counts = {"source_smpl": 0, "bvh": 0, "gmr_g1": 0}
    for frame in sorted(candidate_frames):
        step = max(frame - 1, 0)
        raw_jump = frame > 0 and raw_speed[step] > thresholds.raw_axis_angle_jump_rad * fps
        source_failure = frame > 0 and smpl_speed[step] > thresholds.smpl_joint_speed_m_s
        bvh_failure = bool(
            bvh_error[frame] > thresholds.bvh_position_error_m
            or (frame > 0 and bvh_speed[step] > thresholds.bvh_joint_speed_m_s and not source_failure)
        )
        g1_failure = bool(
            frame > 0
            and (
                root_speed[step] > thresholds.g1_root_speed_m_s
                or root_angular_speed[step] > thresholds.g1_root_angular_speed_rad_s
                or dof_speed[step] > thresholds.g1_joint_speed_rad_s
            )
        )
        if source_failure:
            classification = "source_smpl"
        elif bvh_failure:
            classification = "bvh"
        elif g1_failure:
            classification = "gmr_g1"
        else:
            classification = "parameterization_only"
        if classification in failure_counts:
            failure_counts[classification] += 1
        events.append(
            {
                "frame": int(frame),
                "time_seconds": float(frame / fps),
                "classification": classification,
                "raw_axis_angle_jump": bool(raw_jump),
                "raw_joint": str(smpl_joint_names[int(raw_indices[step])]) if frame > 0 else None,
                "raw_delta_rad": float(raw_speed[step] / fps) if frame > 0 else 0.0,
                "smpl_joint": str(smpl_joint_names[int(smpl_indices[step])]) if frame > 0 else None,
                "smpl_speed_m_s": float(smpl_speed[step]) if frame > 0 else 0.0,
                "bvh_joint": str(smpl_joint_names[int(bvh_indices[step])]) if frame > 0 else None,
                "bvh_speed_m_s": float(bvh_speed[step]) if frame > 0 else 0.0,
                "bvh_error_joint": str(smpl_joint_names[int(bvh_error_indices[frame])]),
                "bvh_position_error_m": float(bvh_error[frame]),
                "g1_root_speed_m_s": float(root_speed[step]) if frame > 0 else 0.0,
                "g1_root_angular_speed_rad_s": (
                    float(root_angular_speed[step]) if frame > 0 else 0.0
                ),
                "g1_dof": str(dof_names[int(dof_indices[step])]) if frame > 0 else None,
                "g1_dof_speed_rad_s": float(dof_speed[step]) if frame > 0 else 0.0,
            }
        )

    return {
        "status": "failed" if any(failure_counts.values()) else "ok",
        "failure_counts": failure_counts,
        "parameterization_only_count": sum(
            event["classification"] == "parameterization_only" for event in events
        ),
        "events": events,
        "maxima": {
            "raw_axis_angle_delta_rad": float(np.max(raw_speed) / fps),
            "smpl_joint_speed_m_s": float(np.max(smpl_speed)),
            "bvh_joint_speed_m_s": float(np.max(bvh_speed)),
            "bvh_position_error_m": float(np.max(bvh_error)),
            "g1_root_speed_m_s": float(np.max(root_speed)),
            "g1_root_angular_speed_rad_s": float(np.max(root_angular_speed)),
            "g1_joint_speed_rad_s": float(np.max(dof_speed)),
        },
    }
