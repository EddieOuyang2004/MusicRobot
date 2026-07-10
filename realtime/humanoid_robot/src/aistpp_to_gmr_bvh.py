from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AISTPP_ROOT = Path(__file__).resolve().parents[1] / "data" / "aistpp"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "aistpp_bvh"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "body_models" / "smpl"
SMPL_JOINT_COUNT = 24
ROTATION_ORDER = "ZYX"


@dataclass(frozen=True)
class JointSpec:
    name: str
    smpl_index: int
    parent: str | None


GMR_LAFAN_JOINTS: tuple[JointSpec, ...] = (
    JointSpec("Hips", 0, None),
    JointSpec("Spine", 3, "Hips"),
    JointSpec("Spine1", 6, "Spine"),
    JointSpec("Spine2", 9, "Spine1"),
    JointSpec("LeftUpLeg", 1, "Hips"),
    JointSpec("LeftLeg", 4, "LeftUpLeg"),
    JointSpec("LeftFoot", 7, "LeftLeg"),
    JointSpec("LeftToe", 10, "LeftFoot"),
    JointSpec("RightUpLeg", 2, "Hips"),
    JointSpec("RightLeg", 5, "RightUpLeg"),
    JointSpec("RightFoot", 8, "RightLeg"),
    JointSpec("RightToe", 11, "RightFoot"),
    JointSpec("LeftArm", 16, "Spine2"),
    JointSpec("LeftForeArm", 18, "LeftArm"),
    JointSpec("LeftHand", 20, "LeftForeArm"),
    JointSpec("RightArm", 17, "Spine2"),
    JointSpec("RightForeArm", 19, "RightArm"),
    JointSpec("RightHand", 21, "RightForeArm"),
)
JOINT_BY_NAME = {joint.name: joint for joint in GMR_LAFAN_JOINTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AIST++ SMPL pickle motions into LaFAN-style BVH files that "
            "GMR can retarget with --format lafan1."
        )
    )
    parser.add_argument("--aistpp-root", type=Path, default=DEFAULT_AISTPP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help="AIST++ split to convert. 'all' converts train, val, and test.",
    )
    parser.add_argument(
        "--motion",
        type=Path,
        default=None,
        help="Optional single .pkl motion. When set, split files are ignored.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output .bvh path for --motion.",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--gender", choices=("MALE", "FEMALE", "NEUTRAL"), default="NEUTRAL")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of motions to convert, useful for smoke tests.",
    )
    return parser.parse_args()


def resolve_workspace_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def import_smpl_dependencies() -> tuple[object, object]:
    try:
        import smplx  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "AIST++ -> BVH conversion needs optional dependencies: "
            "install torch and smplx[all] in the environment used to run this script."
        ) from exc
    return smplx, torch


def load_smpl_rest_pose(model_path: Path, gender: str) -> tuple[np.ndarray, np.ndarray]:
    model_path = resolve_workspace_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"SMPL model path not found: {model_path}. "
            "Place licensed SMPL model files there before converting."
        )

    smplx, torch = import_smpl_dependencies()
    model = smplx.create(
        model_path=str(model_path),
        model_type="smpl",
        gender=gender,
        batch_size=1,
    )
    with torch.no_grad():
        rest = model()

    joints = rest.joints.detach().cpu().numpy().squeeze()[:SMPL_JOINT_COUNT]
    parents = model.parents.detach().cpu().numpy()[:SMPL_JOINT_COUNT]
    return joints.astype(np.float64), parents.astype(np.int64)


def collect_motion_jobs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    output_root = resolve_workspace_path(args.output_root)
    if args.motion is not None:
        motion_path = resolve_workspace_path(args.motion)
        output_file = args.output_file
        if output_file is None:
            output_path = output_root / f"{motion_path.stem}.bvh"
        else:
            output_path = resolve_workspace_path(output_file)
        return [(motion_path, output_path)]

    aistpp_root = resolve_workspace_path(args.aistpp_root)
    split_names = ("train", "val", "test") if args.split == "all" else (args.split,)
    jobs: list[tuple[Path, Path]] = []
    for split_name in split_names:
        split_file = aistpp_root / f"{split_name}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"AIST++ split file not found: {split_file}")

        for line in split_file.read_text(encoding="utf-8").splitlines():
            motion_name = line.strip()
            if not motion_name:
                continue
            filename = motion_name if motion_name.endswith(".pkl") else f"{motion_name}.pkl"
            motion_path = aistpp_root / "motions" / filename
            if not motion_path.exists():
                alternate_path = aistpp_root / split_name / filename
                motion_path = alternate_path if alternate_path.exists() else motion_path
            output_path = output_root / split_name / f"{Path(filename).stem}.bvh"
            jobs.append((motion_path, output_path))

    if args.limit is not None:
        jobs = jobs[: max(args.limit, 0)]
    return jobs


def offsets_from_rest_pose(rest_joints: np.ndarray) -> dict[str, np.ndarray]:
    offsets = {}
    for joint in GMR_LAFAN_JOINTS:
        position = rest_joints[joint.smpl_index]
        if joint.parent is None:
            offset = position
        else:
            parent = JOINT_BY_NAME[joint.parent]
            offset = position - rest_joints[parent.smpl_index]
        offsets[joint.name] = offset * 100.0
    return offsets


def global_smpl_rotations(axis_angles: np.ndarray, parents: np.ndarray) -> np.ndarray:
    frame_count = axis_angles.shape[0]
    local_quats = Rotation.from_rotvec(axis_angles.reshape(-1, 3)).as_quat().reshape(
        frame_count, SMPL_JOINT_COUNT, 4
    )
    global_quats = np.empty_like(local_quats)
    for joint_index in range(SMPL_JOINT_COUNT):
        parent_index = int(parents[joint_index])
        local_rotation = Rotation.from_quat(local_quats[:, joint_index])
        if parent_index < 0:
            global_quats[:, joint_index] = local_rotation.as_quat()
        else:
            parent_rotation = Rotation.from_quat(global_quats[:, parent_index])
            global_quats[:, joint_index] = (parent_rotation * local_rotation).as_quat()
    return global_quats


def output_joint_eulers(global_quats: np.ndarray) -> dict[str, np.ndarray]:
    rotations = {}
    for joint in GMR_LAFAN_JOINTS:
        joint_rotation = Rotation.from_quat(global_quats[:, joint.smpl_index])
        if joint.parent is not None:
            parent = JOINT_BY_NAME[joint.parent]
            parent_rotation = Rotation.from_quat(global_quats[:, parent.smpl_index])
            joint_rotation = parent_rotation.inv() * joint_rotation
        rotations[joint.name] = joint_rotation.as_euler(ROTATION_ORDER, degrees=True)
    return rotations


def load_aistpp_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"AIST++ motion file not found: {path}")

    with path.open("rb") as motion_file:
        motion = pickle.load(motion_file)
    if not isinstance(motion, dict):
        raise ValueError(f"AIST++ motion must contain a dict, got {type(motion).__name__}: {path}")

    missing = {"smpl_poses", "smpl_trans"} - set(motion)
    if missing:
        raise ValueError(f"AIST++ motion is missing keys {sorted(missing)}: {path}")

    poses = np.asarray(motion["smpl_poses"], dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != SMPL_JOINT_COUNT * 3:
        raise ValueError(f"Expected smpl_poses with shape (N, 72), got {poses.shape}: {path}")

    translations = np.asarray(motion["smpl_trans"], dtype=np.float64)
    if translations.shape != (poses.shape[0], 3):
        raise ValueError(
            f"Expected smpl_trans with shape {(poses.shape[0], 3)}, got {translations.shape}: {path}"
        )

    scaling = motion.get("smpl_scaling")
    if scaling is not None:
        scale_value = float(np.asarray(scaling, dtype=np.float64).reshape(-1)[0])
        if scale_value == 0.0:
            raise ValueError(f"smpl_scaling must be nonzero: {path}")
        translations = translations / scale_value

    return poses.reshape(poses.shape[0], SMPL_JOINT_COUNT, 3), translations * 100.0


def hierarchy_children() -> dict[str | None, list[JointSpec]]:
    children: dict[str | None, list[JointSpec]] = {}
    for joint in GMR_LAFAN_JOINTS:
        children.setdefault(joint.parent, []).append(joint)
    return children


def write_joint_hierarchy(
    lines: list[str],
    joint: JointSpec,
    children: dict[str | None, list[JointSpec]],
    offsets: dict[str, np.ndarray],
    indent: int,
) -> None:
    prefix = "\t" * indent
    keyword = "ROOT" if joint.parent is None else "JOINT"
    offset = offsets[joint.name]
    lines.append(f"{prefix}{keyword} {joint.name}")
    lines.append(f"{prefix}{{")
    lines.append(f"{prefix}\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
    if joint.parent is None:
        lines.append(f"{prefix}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
    else:
        lines.append(f"{prefix}\tCHANNELS 3 Zrotation Yrotation Xrotation")

    joint_children = children.get(joint.name, [])
    if joint_children:
        for child in joint_children:
            write_joint_hierarchy(lines, child, children, offsets, indent + 1)
    else:
        lines.append(f"{prefix}\tEnd Site")
        lines.append(f"{prefix}\t{{")
        lines.append(f"{prefix}\t\tOFFSET 0.000000 0.000000 0.000000")
        lines.append(f"{prefix}\t}}")
    lines.append(f"{prefix}}}")


def channel_order(children: dict[str | None, list[JointSpec]]) -> list[JointSpec]:
    ordered: list[JointSpec] = []

    def visit(joint: JointSpec) -> None:
        ordered.append(joint)
        for child in children.get(joint.name, []):
            visit(child)

    visit(children[None][0])
    return ordered


def write_bvh(
    output_path: Path,
    offsets: dict[str, np.ndarray],
    root_positions: np.ndarray,
    rotations: dict[str, np.ndarray],
    fps: int,
) -> None:
    children = hierarchy_children()
    root_joint = children[None][0]
    motion_joints = channel_order(children)
    lines = ["HIERARCHY"]
    write_joint_hierarchy(lines, root_joint, children, offsets, indent=0)
    lines.append("MOTION")
    lines.append(f"Frames: {len(root_positions)}")
    lines.append(f"Frame Time: {1.0 / fps:.7f}")

    for frame_index in range(len(root_positions)):
        values: list[float] = []
        for joint in motion_joints:
            if joint.parent is None:
                values.extend(root_positions[frame_index].tolist())
            values.extend(rotations[joint.name][frame_index].tolist())
        lines.append(" ".join(f"{value:.6f}" for value in values))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_motion(
    motion_path: Path,
    output_path: Path,
    offsets: dict[str, np.ndarray],
    smpl_parents: np.ndarray,
    fps: int,
    overwrite: bool,
) -> tuple[bool, str]:
    if output_path.exists() and not overwrite:
        return False, f"skip existing {output_path}"

    poses, root_positions = load_aistpp_motion(motion_path)
    global_quats = global_smpl_rotations(poses, smpl_parents)
    rotations = output_joint_eulers(global_quats)
    root_positions = root_positions + offsets["Hips"][None, :]
    write_bvh(output_path, offsets, root_positions, rotations, fps)
    return True, f"wrote {output_path} ({len(root_positions)} frames)"


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")

    try:
        jobs = collect_motion_jobs(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not jobs:
        print("No AIST++ motions matched the requested input.")
        return 0

    try:
        rest_joints, smpl_parents = load_smpl_rest_pose(args.model_path, args.gender)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    offsets = offsets_from_rest_pose(rest_joints)

    converted = 0
    for motion_path, output_path in jobs:
        motion_path = resolve_workspace_path(motion_path)
        output_path = resolve_workspace_path(output_path)
        try:
            did_convert, message = convert_motion(
                motion_path=motion_path,
                output_path=output_path,
                offsets=offsets,
                smpl_parents=smpl_parents,
                fps=args.fps,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            print(f"error {motion_path}: {exc}", file=sys.stderr)
            continue
        converted += int(did_convert)
        print(message)

    print(f"Finished. Converted {converted} of {len(jobs)} requested motion(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
