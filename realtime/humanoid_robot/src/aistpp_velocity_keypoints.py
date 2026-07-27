from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation


DEFAULT_AISTPP_MOTION = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "aistpp"
    / "motions"
    / "gWA_sBM_cAll_d26_mWA0_ch07.pkl"
)
JOINT_POSITION_KEYS = ("keypoints3d", "joints3d", "smpl_joints", "joint_positions")


@dataclass(frozen=True)
class AistppMotionData:
    smpl_poses: np.ndarray
    smpl_trans: np.ndarray | None
    smpl_scaling: float | None
    joints3d: np.ndarray | None


@dataclass(frozen=True)
class VelocityValleyResult:
    frame_indices: tuple[int, ...]
    times: tuple[float, ...]
    phases: tuple[float, ...]
    scores: tuple[float, ...]
    prominences: tuple[float, ...]
    velocities: tuple[float, ...]
    raw_velocity: np.ndarray
    smoothed_velocity: np.ndarray
    fps: float
    signal_source: str
    reason: str


def load_aistpp_motion(path: Path | str) -> AistppMotionData:
    motion_path = Path(path)
    with motion_path.open("rb") as motion_file:
        motion = pickle.load(motion_file)
    if not isinstance(motion, Mapping):
        raise ValueError(f"AIST++ motion must contain a mapping, got {type(motion).__name__}.")
    if "smpl_poses" not in motion:
        raise ValueError("AIST++ motion is missing smpl_poses.")

    poses = np.asarray(motion["smpl_poses"], dtype=np.float64)
    if poses.ndim == 3 and poses.shape[1:] == (24, 3):
        poses = poses.reshape(poses.shape[0], 72)
    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError(f"Expected smpl_poses with shape (N, 72), got {poses.shape}.")
    if poses.shape[0] < 3:
        raise ValueError("AIST++ motion must contain at least three frames.")
    if not np.all(np.isfinite(poses)):
        raise ValueError("smpl_poses contains non-finite values.")

    translations = _optional_frame_array(motion.get("smpl_trans"), poses.shape[0], 3, "smpl_trans")
    scaling = _optional_positive_scalar(motion.get("smpl_scaling"), "smpl_scaling")
    joints3d = _find_joint_positions(motion, poses.shape[0])
    return AistppMotionData(
        smpl_poses=poses,
        smpl_trans=translations,
        smpl_scaling=scaling,
        joints3d=joints3d,
    )


def detect_aistpp_velocity_keypoints(
    smpl_poses: np.ndarray,
    fps: float = 60.0,
    smpl_trans: np.ndarray | None = None,
    smpl_scaling: float | Sequence[float] | np.ndarray | None = None,
    joints3d: np.ndarray | None = None,
    smoothing_sec: float = 0.08,
    min_spacing_sec: float = 0.28,
    prominence: float = 0.08,
    max_count: int | None = None,
    boundary_sec: float = 0.10,
    angular_radius_m: float = 0.35,
    root_translation_weight: float = 1.0,
) -> VelocityValleyResult:
    fps = _positive_float(fps, "fps")
    poses = _as_smpl_poses(smpl_poses)

    if joints3d is not None:
        positions = _as_joint_positions(joints3d, poses.shape[0])
        velocity = joint_position_velocity(positions, fps)
        signal_source = "3d-joint-position"
    else:
        angular_velocity = smpl_angular_velocity(poses, fps)
        velocity = max(float(angular_radius_m), 0.0) * angular_velocity
        signal_source = "smpl-angular"

        if smpl_trans is not None and root_translation_weight > 0.0:
            translations = _as_translations(smpl_trans, poses.shape[0])
            scale = _optional_positive_scalar(smpl_scaling, "smpl_scaling")
            if scale is not None:
                translations = translations / scale
            root_velocity = root_linear_velocity(translations, fps)
            velocity = np.hypot(velocity, float(root_translation_weight) * root_velocity)
            signal_source += "+root-translation"

    return detect_velocity_valleys(
        velocity=velocity,
        fps=fps,
        smoothing_sec=smoothing_sec,
        min_spacing_sec=min_spacing_sec,
        prominence=prominence,
        max_count=max_count,
        boundary_sec=boundary_sec,
        signal_source=signal_source,
    )


def detect_aistpp_file(
    path: Path | str,
    fps: float = 60.0,
    **kwargs: Any,
) -> VelocityValleyResult:
    motion = load_aistpp_motion(path)
    return detect_aistpp_velocity_keypoints(
        smpl_poses=motion.smpl_poses,
        smpl_trans=motion.smpl_trans,
        smpl_scaling=motion.smpl_scaling,
        joints3d=motion.joints3d,
        fps=fps,
        **kwargs,
    )


def joint_position_velocity(joints3d: np.ndarray, fps: float) -> np.ndarray:
    """Compute the AIST++ whole-body kinetic-velocity signal from 3D joints."""
    positions = np.asarray(joints3d, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] < 3:
        raise ValueError(f"Expected joints3d with shape (N, J, 3), got {positions.shape}.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("joints3d contains non-finite values.")

    derivative = _central_difference(positions, _positive_float(fps, "fps"))
    joint_speeds_squared = np.sum(derivative * derivative, axis=2)
    return np.sqrt(np.mean(joint_speeds_squared, axis=1))


def smpl_angular_velocity(smpl_poses: np.ndarray, fps: float) -> np.ndarray:
    """Return RMS geodesic angular velocity for all 24 SMPL joints."""
    poses = _as_smpl_poses(smpl_poses).reshape(-1, 24, 3)
    frame_count = poses.shape[0]
    matrices = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix()
    matrices = matrices.reshape(frame_count, 24, 3, 3)

    angular_speed = np.empty((frame_count, 24), dtype=np.float64)
    angular_speed[0] = _relative_rotation_angles(matrices[0], matrices[1]) * fps
    angular_speed[-1] = _relative_rotation_angles(matrices[-2], matrices[-1]) * fps
    angular_speed[1:-1] = (
        _relative_rotation_angles(matrices[:-2], matrices[2:]) * (0.5 * fps)
    )
    return np.sqrt(np.mean(angular_speed * angular_speed, axis=1))


def root_linear_velocity(translations: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(translations, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 3:
        raise ValueError(f"Expected translations with shape (N, 3), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("translations contains non-finite values.")
    derivative = _central_difference(values, _positive_float(fps, "fps"))
    return np.linalg.norm(derivative, axis=1)


def detect_velocity_valleys(
    velocity: Sequence[float] | np.ndarray,
    fps: float,
    smoothing_sec: float = 0.08,
    min_spacing_sec: float = 0.28,
    prominence: float = 0.08,
    max_count: int | None = None,
    boundary_sec: float = 0.10,
    signal_source: str = "velocity",
) -> VelocityValleyResult:
    fps = _positive_float(fps, "fps")
    raw = np.asarray(velocity, dtype=np.float64)
    if raw.ndim != 1 or raw.size < 3:
        raise ValueError(f"Expected a one-dimensional velocity curve with at least 3 frames, got {raw.shape}.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("velocity contains non-finite values.")

    sigma_frames = max(float(smoothing_sec), 0.0) * fps
    smoothed = (
        gaussian_filter1d(raw, sigma=sigma_frames, mode="nearest")
        if sigma_frames > 1e-6
        else raw.copy()
    )
    normalized, dynamic_range = _robust_unit_range(smoothed)
    if dynamic_range <= 1e-9:
        return _empty_result(raw, smoothed, fps, signal_source, "velocity curve is too flat")

    min_spacing_frames = max(1, int(round(max(float(min_spacing_sec), 0.0) * fps)))
    peak_prominence = max(float(prominence), 0.0)
    candidates, properties = find_peaks(
        -normalized,
        distance=min_spacing_frames,
        prominence=peak_prominence,
    )
    if candidates.size == 0:
        return _empty_result(raw, smoothed, fps, signal_source, "no prominent velocity valleys found")

    boundary_frames = max(0, int(math.ceil(max(float(boundary_sec), 0.0) * fps)))
    if boundary_frames:
        keep = (candidates >= boundary_frames) & (candidates < raw.size - boundary_frames)
        candidates = candidates[keep]
        valley_prominences = np.asarray(properties["prominences"], dtype=float)[keep]
    else:
        valley_prominences = np.asarray(properties["prominences"], dtype=float)
    if candidates.size == 0:
        return _empty_result(raw, smoothed, fps, signal_source, "all valleys were at clip boundaries")

    valley_depth = 1.0 - normalized[candidates]
    scores = np.clip(0.65 * valley_prominences + 0.35 * valley_depth, 0.0, 1.0)
    if max_count is not None:
        count = max(int(max_count), 0)
        if count == 0:
            return _empty_result(raw, smoothed, fps, signal_source, "max_count is zero")
        if candidates.size > count:
            chosen = np.argsort(scores)[-count:]
            candidates = candidates[chosen]
            valley_prominences = valley_prominences[chosen]
            scores = scores[chosen]

    order = np.argsort(candidates)
    candidates = candidates[order]
    valley_prominences = valley_prominences[order]
    scores = scores[order]
    frame_count = raw.size
    return VelocityValleyResult(
        frame_indices=tuple(int(index) for index in candidates),
        times=tuple(float(index / fps) for index in candidates),
        phases=tuple(float(index / frame_count) for index in candidates),
        scores=tuple(float(value) for value in scores),
        prominences=tuple(float(value) for value in valley_prominences),
        velocities=tuple(float(smoothed[index]) for index in candidates),
        raw_velocity=raw,
        smoothed_velocity=smoothed,
        fps=fps,
        signal_source=signal_source,
        reason="ok",
    )


def result_as_dict(result: VelocityValleyResult, motion_path: Path | str | None = None) -> dict[str, Any]:
    keypoints = [
        {
            "frame": frame,
            "time_sec": time_sec,
            "phase": phase,
            "score": score,
            "prominence": prominence,
            "velocity": velocity,
        }
        for frame, time_sec, phase, score, prominence, velocity in zip(
            result.frame_indices,
            result.times,
            result.phases,
            result.scores,
            result.prominences,
            result.velocities,
        )
    ]
    output: dict[str, Any] = {
        "fps": result.fps,
        "frame_count": int(result.raw_velocity.size),
        "duration_sec": float(result.raw_velocity.size / result.fps),
        "signal_source": result.signal_source,
        "reason": result.reason,
        "keypoint_count": len(keypoints),
        "keypoints": keypoints,
    }
    if motion_path is not None:
        output["motion"] = str(Path(motion_path))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect dance keypoints as prominent valleys of an AIST++ whole-body velocity curve."
    )
    parser.add_argument("motion", nargs="?", type=Path, default=DEFAULT_AISTPP_MOTION)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--smoothing-sec", type=float, default=0.08)
    parser.add_argument("--min-spacing-sec", type=float, default=0.28)
    parser.add_argument("--prominence", type=float, default=0.08)
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--boundary-sec", type=float, default=0.10)
    parser.add_argument("--angular-radius-m", type=float, default=0.35)
    parser.add_argument("--root-translation-weight", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optionally save the result as JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = detect_aistpp_file(
        path=args.motion,
        fps=args.fps,
        smoothing_sec=args.smoothing_sec,
        min_spacing_sec=args.min_spacing_sec,
        prominence=args.prominence,
        max_count=args.max_count,
        boundary_sec=args.boundary_sec,
        angular_radius_m=args.angular_radius_m,
        root_translation_weight=args.root_translation_weight,
    )
    output = result_as_dict(result, args.motion)
    json_text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text + "\n", encoding="utf-8")

    if args.json:
        print(json_text)
        return

    print(
        f"Velocity signal: {result.signal_source}; "
        f"{len(result.frame_indices)} keypoints; {result.reason}."
    )
    print("  #   frame    time(s)    phase    score    velocity")
    for number, (frame, time_sec, phase, score, velocity) in enumerate(
        zip(result.frame_indices, result.times, result.phases, result.scores, result.velocities),
        start=1,
    ):
        print(f"{number:3d} {frame:7d} {time_sec:10.3f} {phase:8.4f} {score:8.3f} {velocity:11.5f}")
    if args.output_json is not None:
        print(f"Saved JSON: {args.output_json}")


def _as_smpl_poses(value: np.ndarray) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim == 3 and poses.shape[1:] == (24, 3):
        poses = poses.reshape(poses.shape[0], 72)
    if poses.ndim != 2 or poses.shape[1] != 72 or poses.shape[0] < 3:
        raise ValueError(f"Expected smpl_poses with shape (N, 72), got {poses.shape}.")
    if not np.all(np.isfinite(poses)):
        raise ValueError("smpl_poses contains non-finite values.")
    return poses


def _as_joint_positions(value: np.ndarray, frame_count: int) -> np.ndarray:
    positions = np.asarray(value, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[0] != frame_count or positions.shape[2] != 3:
        raise ValueError(f"Expected joints3d with shape ({frame_count}, J, 3), got {positions.shape}.")
    if positions.shape[1] == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("joints3d must contain finite joint positions.")
    return positions


def _as_translations(value: np.ndarray, frame_count: int) -> np.ndarray:
    translations = np.asarray(value, dtype=np.float64)
    if translations.shape != (frame_count, 3):
        raise ValueError(f"Expected smpl_trans with shape ({frame_count}, 3), got {translations.shape}.")
    if not np.all(np.isfinite(translations)):
        raise ValueError("smpl_trans contains non-finite values.")
    return translations


def _optional_frame_array(
    value: Any,
    frame_count: int,
    width: int,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (frame_count, width):
        raise ValueError(f"Expected {name} with shape ({frame_count}, {width}), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _optional_positive_scalar(value: Any, name: str) -> float | None:
    if value is None:
        return None
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{name} must contain one scalar value.")
    scalar = float(values[0])
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return scalar


def _find_joint_positions(motion: Mapping[str, Any], frame_count: int) -> np.ndarray | None:
    for key in JOINT_POSITION_KEYS:
        if key not in motion:
            continue
        positions = np.asarray(motion[key], dtype=np.float64)
        if positions.ndim == 2 and positions.shape[0] == frame_count and positions.shape[1] % 3 == 0:
            positions = positions.reshape(frame_count, -1, 3)
        return _as_joint_positions(positions, frame_count)
    return None


def _central_difference(values: np.ndarray, fps: float) -> np.ndarray:
    derivative = np.empty_like(values, dtype=np.float64)
    derivative[0] = (values[1] - values[0]) * fps
    derivative[-1] = (values[-1] - values[-2]) * fps
    derivative[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    return derivative


def _relative_rotation_angles(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = np.matmul(np.swapaxes(first, -1, -2), second)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def _robust_unit_range(values: np.ndarray) -> tuple[np.ndarray, float]:
    low, high = np.percentile(values, (5.0, 95.0))
    dynamic_range = float(high - low)
    if not math.isfinite(dynamic_range) or dynamic_range <= 1e-9:
        low = float(np.min(values))
        high = float(np.max(values))
        dynamic_range = high - low
        if not math.isfinite(dynamic_range) or dynamic_range <= 1e-9:
            return np.zeros_like(values), 0.0
    return np.clip((values - low) / dynamic_range, 0.0, 1.0), dynamic_range


def _empty_result(
    raw: np.ndarray,
    smoothed: np.ndarray,
    fps: float,
    signal_source: str,
    reason: str,
) -> VelocityValleyResult:
    return VelocityValleyResult(
        frame_indices=(),
        times=(),
        phases=(),
        scores=(),
        prominences=(),
        velocities=(),
        raw_velocity=raw,
        smoothed_velocity=smoothed,
        fps=fps,
        signal_source=signal_source,
        reason=reason,
    )


def _positive_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return result


if __name__ == "__main__":
    main()
