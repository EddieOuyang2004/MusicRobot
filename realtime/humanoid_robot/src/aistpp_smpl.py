from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SMPL_JOINT_COUNT = 24
Y_UP_TO_Z_UP = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# Names are the exact lower-case keys consumed by GMR's smplx_to_g1 config.
GMR_SMPL_JOINTS: tuple[tuple[str, int], ...] = (
    ("pelvis", 0),
    ("spine3", 9),
    ("left_hip", 1),
    ("right_hip", 2),
    ("left_knee", 4),
    ("right_knee", 5),
    ("left_foot", 10),
    ("right_foot", 11),
    ("left_shoulder", 16),
    ("right_shoulder", 17),
    ("left_elbow", 18),
    ("right_elbow", 19),
    ("left_wrist", 20),
    ("right_wrist", 21),
)
GMR_SMPL_NAMES = tuple(name for name, _index in GMR_SMPL_JOINTS)
GMR_SMPL_INDICES = np.asarray([index for _name, index in GMR_SMPL_JOINTS], dtype=np.int64)


def import_smpl_dependencies() -> tuple[object, object]:
    try:
        import smplx  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "AIST++ SMPL processing needs optional dependencies: install torch and "
            "smplx[all] in the GMR environment."
        ) from exc
    return smplx, torch


def load_smpl_rest_pose(model_path: Path, gender: str = "NEUTRAL") -> tuple[np.ndarray, np.ndarray]:
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"SMPL model path not found: {model_path}")
    smplx, torch = import_smpl_dependencies()
    model = smplx.create(
        model_path=str(model_path.parent),
        model_type="smpl",
        gender=gender,
        batch_size=1,
    )
    with torch.no_grad():
        rest = model()
    joints = rest.joints.detach().cpu().numpy().squeeze()[:SMPL_JOINT_COUNT]
    parents = model.parents.detach().cpu().numpy()[:SMPL_JOINT_COUNT]
    return joints.astype(np.float64), parents.astype(np.int64)


def load_aistpp_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an AIST++ SMPL motion as axis angles and translations in metres."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"AIST++ motion file not found: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"AIST++ motion must contain a dict, got {type(payload).__name__}: {path}")
    missing = {"smpl_poses", "smpl_trans"} - set(payload)
    if missing:
        raise ValueError(f"AIST++ motion is missing keys {sorted(missing)}: {path}")

    poses = np.asarray(payload["smpl_poses"], dtype=np.float64)
    if poses.ndim == 2 and poses.shape[1] == SMPL_JOINT_COUNT * 3:
        poses = poses.reshape(-1, SMPL_JOINT_COUNT, 3)
    elif poses.ndim != 3 or poses.shape[1:] != (SMPL_JOINT_COUNT, 3):
        raise ValueError(
            f"Expected smpl_poses with shape (N,72) or (N,24,3), got {poses.shape}: {path}"
        )
    translations = np.asarray(payload["smpl_trans"], dtype=np.float64)
    if translations.shape != (len(poses), 3):
        raise ValueError(
            f"Expected smpl_trans with shape {(len(poses), 3)}, got {translations.shape}: {path}"
        )
    scaling = payload.get("smpl_scaling")
    if scaling is not None:
        scale = float(np.asarray(scaling, dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError(f"smpl_scaling must be finite and nonzero: {path}")
        translations = translations / scale
    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(translations)):
        raise ValueError(f"AIST++ motion contains non-finite values: {path}")
    return poses, translations


def smpl_world_kinematics(
    axis_angles: np.ndarray,
    translations_m: np.ndarray,
    rest_joints_m: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Z-up world positions and scalar-first global rotations for all SMPL joints."""
    poses = np.asarray(axis_angles, dtype=np.float64)
    translations = np.asarray(translations_m, dtype=np.float64)
    rest = np.asarray(rest_joints_m, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    if poses.ndim != 3 or poses.shape[2] != 3:
        raise ValueError(f"Expected axis_angles[N,J,3], got {poses.shape}.")
    frame_count, joint_count, _ = poses.shape
    if translations.shape != (frame_count, 3):
        raise ValueError(f"Expected translations[{frame_count},3], got {translations.shape}.")
    if rest.shape != (joint_count, 3) or parents.shape != (joint_count,):
        raise ValueError("SMPL rest joints/parents do not match the pose joint count.")

    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(
        frame_count, joint_count, 3, 3
    )
    global_matrices = np.empty_like(local)
    positions = np.empty((frame_count, joint_count, 3), dtype=np.float64)
    for joint_index in range(joint_count):
        parent_index = int(parents[joint_index])
        if parent_index < 0:
            global_matrices[:, joint_index] = local[:, joint_index]
            positions[:, joint_index] = translations + rest[joint_index]
        else:
            global_matrices[:, joint_index] = global_matrices[:, parent_index] @ local[:, joint_index]
            offset = rest[joint_index] - rest[parent_index]
            positions[:, joint_index] = positions[:, parent_index] + np.einsum(
                "nij,j->ni", global_matrices[:, parent_index], offset
            )

    positions = positions @ Y_UP_TO_Z_UP.T
    rotations_z_up = Y_UP_TO_Z_UP[None, None, :, :] @ global_matrices
    quaternions_xyzw = Rotation.from_matrix(rotations_z_up.reshape(-1, 3, 3)).as_quat()
    quaternions_wxyz = quaternions_xyzw[:, [3, 0, 1, 2]].reshape(frame_count, joint_count, 4)
    # Quaternion signs do not change orientations, but stable signs make debugging and
    # serialized intermediate layers deterministic.
    for frame_index in range(1, frame_count):
        dots = np.sum(quaternions_wxyz[frame_index - 1] * quaternions_wxyz[frame_index], axis=1)
        quaternions_wxyz[frame_index, dots < 0.0] *= -1.0
    return positions, quaternions_wxyz


def gmr_smpl_frames(
    positions_m_zup: np.ndarray,
    rotations_wxyz_zup: np.ndarray,
) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    positions = np.asarray(positions_m_zup, dtype=np.float64)
    rotations = np.asarray(rotations_wxyz_zup, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (SMPL_JOINT_COUNT, 3):
        raise ValueError(f"Expected SMPL positions[N,24,3], got {positions.shape}.")
    if rotations.shape != (len(positions), SMPL_JOINT_COUNT, 4):
        raise ValueError(f"Expected SMPL rotations[{len(positions)},24,4], got {rotations.shape}.")
    frames: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    for frame_index in range(len(positions)):
        frames.append(
            {
                name: (
                    positions[frame_index, joint_index].copy(),
                    rotations[frame_index, joint_index].copy(),
                )
                for name, joint_index in GMR_SMPL_JOINTS
            }
        )
    return frames

