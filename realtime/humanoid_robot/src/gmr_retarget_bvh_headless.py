from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np


FORMAT_VERSION = 1
PIPELINE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retarget a LaFAN1 BVH to Unitree G1 with GMR without opening a viewer."
    )
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument("--bvh-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-fps", type=float, required=True)
    parser.add_argument("--source-motion-id", required=True)
    parser.add_argument("--retargeter-version", required=True)
    parser.add_argument("--source-sha256")
    parser.add_argument("--smpl-model-sha256")
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--max-joint-speed", type=float, default=3.0 * np.pi)
    parser.add_argument("--max-root-speed", type=float, default=3.0)
    parser.add_argument("--max-root-angular-speed", type=float, default=4.0 * np.pi)
    parser.add_argument("--loop-closure-seconds", type=float, default=0.5)
    parser.add_argument(
        "--velocity-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply GMR/Mink joint velocity limits (enabled by default).",
    )
    return parser.parse_args()


def _slerp_wxyz(first: np.ndarray, second: np.ndarray, amount: float) -> np.ndarray:
    a = first / np.linalg.norm(first)
    b = second / np.linalg.norm(second)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = a + amount * (b - a)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    result = (
        np.sin((1.0 - amount) * angle) / np.sin(angle) * a
        + np.sin(amount * angle) / np.sin(angle) * b
    )
    return result / np.linalg.norm(result)


def limit_qpos_velocity(
    qpos: np.ndarray,
    fps: float,
    *,
    max_joint_speed: float,
    max_root_speed: float,
    max_root_angular_speed: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Enforce per-source-frame limits after GMR's multi-iteration IK solve."""
    if min(max_joint_speed, max_root_speed, max_root_angular_speed) <= 0.0:
        raise ValueError("All trajectory speed limits must be positive.")
    result = qpos.copy()
    counts = {"root_position": 0, "root_rotation": 0, "joints": 0}
    max_joint_step = max_joint_speed / fps
    max_root_step = max_root_speed / fps
    max_angular_step = max_root_angular_speed / fps
    result[0, 3:7] /= np.linalg.norm(result[0, 3:7])
    for frame_index in range(1, len(result)):
        root_delta = result[frame_index, :3] - result[frame_index - 1, :3]
        root_distance = float(np.linalg.norm(root_delta))
        if root_distance > max_root_step:
            result[frame_index, :3] = (
                result[frame_index - 1, :3] + root_delta * (max_root_step / root_distance)
            )
            counts["root_position"] += 1

        previous_quat = result[frame_index - 1, 3:7]
        current_quat = result[frame_index, 3:7]
        current_quat /= np.linalg.norm(current_quat)
        dot = abs(float(np.clip(np.dot(previous_quat, current_quat), -1.0, 1.0)))
        angular_distance = 2.0 * float(np.arccos(dot))
        if angular_distance > max_angular_step:
            current_quat = _slerp_wxyz(
                previous_quat,
                current_quat,
                max_angular_step / angular_distance,
            )
            counts["root_rotation"] += 1
        elif np.dot(previous_quat, current_quat) < 0.0:
            current_quat = -current_quat
        result[frame_index, 3:7] = current_quat

        joint_delta = result[frame_index, 7:] - result[frame_index - 1, 7:]
        clipped_delta = np.clip(joint_delta, -max_joint_step, max_joint_step)
        counts["joints"] += int(np.count_nonzero(np.abs(clipped_delta - joint_delta) > 1e-12))
        result[frame_index, 7:] = result[frame_index - 1, 7:] + clipped_delta
    return result, counts


def apply_loop_closure(qpos: np.ndarray, fps: float, duration: float) -> tuple[np.ndarray, int]:
    """Close joint/Z/tilt seams while preserving the authored terminal XY/yaw path."""
    frame_count = min(max(int(round(duration * fps)), 0), len(qpos) - 1)
    if frame_count == 0:
        return qpos.copy(), 0
    result = qpos.copy()
    start = len(result) - frame_count
    for frame_index in range(start, len(result)):
        linear = (frame_index - start + 1) / frame_count
        amount = linear * linear * (3.0 - 2.0 * linear)
        result[frame_index, 2] = (
            (1.0 - amount) * qpos[frame_index, 2] + amount * qpos[0, 2]
        )
        result[frame_index, 7:] = (
            (1.0 - amount) * qpos[frame_index, 7:] + amount * qpos[0, 7:]
        )
        quat = qpos[frame_index, 3:7] / np.linalg.norm(qpos[frame_index, 3:7])
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_only = np.asarray([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
        result[frame_index, 3:7] = _slerp_wxyz(quat, yaw_only, amount)
    return result, frame_count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qpos_joint_names(model: object) -> tuple[str, ...]:
    """Read the scalar-joint order from MuJoCo qpos addresses, not actuator order."""
    import mujoco

    addressed: list[tuple[int, str]] = []
    for joint_id in range(model.njnt):
        address = int(model.jnt_qposadr[joint_id])
        joint_type = int(model.jnt_type[joint_id])
        if address < 7:
            continue
        if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError(f"G1 joint at qpos[{address}] is not scalar.")
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            raise ValueError(f"Unnamed G1 joint at qpos[{address}].")
        addressed.append((address, name.removesuffix("_joint")))
    addressed.sort()
    expected_addresses = list(range(7, 7 + len(addressed)))
    actual_addresses = [address for address, _ in addressed]
    if actual_addresses != expected_addresses:
        raise ValueError(
            f"G1 qpos joints are not contiguous after the free joint: {actual_addresses}."
        )
    return tuple(name for _, name in addressed)


def retarget(args: argparse.Namespace) -> dict[str, object]:
    gmr_root = args.gmr_root.resolve()
    bvh_path = args.bvh_file.resolve()
    if not (gmr_root / "general_motion_retargeting").is_dir():
        raise FileNotFoundError(f"GMR package not found under {gmr_root}.")
    if not bvh_path.is_file():
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")
    sys.path.insert(0, str(gmr_root))

    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    human_frames, actual_human_height = load_bvh_file(str(bvh_path), format="lafan1")
    if not human_frames:
        raise ValueError(f"BVH contains no frames: {bvh_path}")
    retargeter = GeneralMotionRetargeting(
        src_human="bvh_lafan1",
        tgt_robot="unitree_g1",
        actual_human_height=actual_human_height,
        solver=args.solver,
        verbose=False,
        use_velocity_limit=args.velocity_limit,
    )
    names = qpos_joint_names(retargeter.model)
    if len(names) != 29 or retargeter.model.nq != 36:
        raise ValueError(
            f"Expected free root + 29 G1 joints, got nq={retargeter.model.nq}, names={len(names)}."
        )

    qpos = np.empty((len(human_frames), retargeter.model.nq), dtype=np.float64)
    for frame_index, human_frame in enumerate(human_frames):
        qpos[frame_index] = retargeter.retarget(human_frame)
    if not np.all(np.isfinite(qpos)):
        indices = np.argwhere(~np.isfinite(qpos))[0].tolist()
        raise ValueError(f"GMR generated a non-finite qpos at index {indices}.")

    qpos, loop_closure_frames = apply_loop_closure(
        qpos,
        float(args.motion_fps),
        float(args.loop_closure_seconds),
    )
    qpos, limited_counts = limit_qpos_velocity(
        qpos,
        float(args.motion_fps),
        max_joint_speed=float(args.max_joint_speed),
        max_root_speed=float(args.max_root_speed),
        max_root_angular_speed=float(args.max_root_angular_speed),
    )
    root_quats = qpos[:, 3:7]
    norms = np.linalg.norm(root_quats, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("GMR generated a zero-length root quaternion.")
    root_quats = root_quats / norms[:, None]
    return {
        "format_version": FORMAT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "fps": float(args.motion_fps),
        "root_pos": qpos[:, :3],
        # MuJoCo free-joint qpos is scalar-first. Keep that exact order on disk.
        "root_rot": root_quats,
        "root_rot_order": "wxyz",
        "dof_pos": qpos[:, 7:],
        "dof_names": list(names),
        "source_motion_id": args.source_motion_id,
        "source_sha256": args.source_sha256,
        "bvh_sha256": sha256(bvh_path),
        "smpl_model_sha256": args.smpl_model_sha256,
        "retargeter": "GMR",
        "retargeter_version": args.retargeter_version,
        "continuity_limits": {
            "max_joint_speed_rad_s": float(args.max_joint_speed),
            "max_root_speed_m_s": float(args.max_root_speed),
            "max_root_angular_speed_rad_s": float(args.max_root_angular_speed),
            "limited_values": limited_counts,
            "loop_closure_frames": loop_closure_frames,
        },
    }


def main() -> int:
    args = parse_args()
    if args.motion_fps <= 0:
        raise ValueError("--motion-fps must be positive.")
    payload = retarget(args)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)
    print(f"Wrote {output_path} ({len(payload['root_pos'])} frames, 29 DoF).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
