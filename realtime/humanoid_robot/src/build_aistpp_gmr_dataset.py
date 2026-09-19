from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from unitree_g1_dance_adapter import UnitreeG1DanceAdapter


ROOT = Path(__file__).resolve().parents[3]
HUMANOID_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AISTPP_ROOT = HUMANOID_ROOT / "data" / "aistpp"
DEFAULT_OUTPUT_ROOT = HUMANOID_ROOT / "data" / "aistpp_gmr"
DEFAULT_SMPL_ROOT = HUMANOID_ROOT / "assets" / "body_models" / "smpl"
MANIFEST_NAME = "manifest.json"
FAILURES_NAME = "failures.json"
PIPELINE_VERSION = 4
SOURCE_FORMAT = "aistpp_smpl_direct"
DEFAULT_COLLISION_MIN_DISTANCE_M = 0.005
DEFAULT_COLLISION_DETECTION_DISTANCE_M = 0.08
DEFAULT_COLLISION_GAIN = 0.85


@dataclass(frozen=True)
class MotionBuildResult:
    source_path: Path
    artifact_path: Path
    message: str
    entry: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch AIST++ SMPL -> GMR Unitree G1 preparation.")
    parser.add_argument("--aistpp-root", type=Path, default=DEFAULT_AISTPP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smpl-model-path", type=Path, default=DEFAULT_SMPL_ROOT)
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument("--gmr-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--numeric-threads",
        type=int,
        default=1,
        help="Maximum BLAS/OpenMP threads in each GMR subprocess.",
    )
    parser.add_argument(
        "--child-priority",
        choices=("idle", "below-normal", "normal"),
        default="below-normal",
        help="Windows scheduling priority for GMR subprocesses.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip valid current-version artifacts and replace stale artifacts in place "
            "without creating .invalid backups."
        ),
    )
    parser.add_argument("--solver", default="daqp")
    parser.add_argument(
        "--velocity-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply GMR/Mink joint velocity limits (enabled by default).",
    )
    parser.add_argument(
        "--collision-avoidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Mink self-collision avoidance constraints (enabled by default).",
    )
    parser.add_argument(
        "--collision-min-distance",
        type=float,
        default=DEFAULT_COLLISION_MIN_DISTANCE_M,
    )
    parser.add_argument(
        "--collision-detection-distance",
        type=float,
        default=DEFAULT_COLLISION_DETECTION_DISTANCE_M,
    )
    parser.add_argument("--collision-gain", type=float, default=DEFAULT_COLLISION_GAIN)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def collect_motion_paths(args: argparse.Namespace) -> list[Path]:
    aistpp_root = resolve(args.aistpp_root)
    if args.motion is not None:
        return [resolve(args.motion)]
    split_names = ("train", "val", "test") if args.split == "all" else (args.split,)
    paths: list[Path] = []
    for split_name in split_names:
        split_path = aistpp_root / f"{split_name}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"AIST++ split not found: {split_path}")
        for name in split_path.read_text(encoding="utf-8").splitlines():
            name = name.strip()
            if name:
                paths.append(aistpp_root / "motions" / f"{Path(name).stem}.pkl")
    if args.limit is not None:
        paths = paths[: max(args.limit, 0)]
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gmr_artifact(
    path: Path,
    *,
    source_hash: str | None = None,
    model_hash: str | None = None,
    retargeter_version: str | None = None,
    expected_fps: float | None = None,
    expected_pipeline_version: int = PIPELINE_VERSION,
    expected_collision_avoidance: bool | None = None,
    expected_collision_min_distance: float | None = None,
    expected_collision_detection_distance: float | None = None,
    expected_collision_gain: float | None = None,
) -> None:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {
        "fps",
        "root_pos",
        "root_rot",
        "root_rot_order",
        "dof_pos",
        "dof_names",
        "source_motion_id",
        "source_sha256",
        "smpl_model_sha256",
        "retargeter",
        "retargeter_version",
        "collision_avoidance",
        "mink_limits_api",
        "continuity_limits",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"GMR artifact is missing fields {sorted(missing)}: {path}")
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError(f"Unsupported or missing GMR artifact format_version: {path}")
    if int(payload.get("pipeline_version", -1)) != expected_pipeline_version:
        raise ValueError(f"Unsupported or missing GMR artifact pipeline_version: {path}")
    if payload.get("source_format") != SOURCE_FORMAT:
        raise ValueError(f"Unsupported or missing GMR artifact source_format: {path}")
    if payload.get("root_rot_order") != "wxyz" or payload.get("retargeter") != "GMR":
        raise ValueError(f"Non-canonical GMR artifact metadata: {path}")
    if "bvh_sha256" in payload:
        raise ValueError(f"SMPL-direct GMR artifact unexpectedly records BVH provenance: {path}")
    if source_hash is not None and payload.get("source_sha256") != source_hash:
        raise ValueError(f"Source hash changed for {path.name}.")
    if model_hash is not None and payload.get("smpl_model_sha256") != model_hash:
        raise ValueError(f"SMPL model hash changed for {path.name}.")
    if retargeter_version is not None and payload.get("retargeter_version") != retargeter_version:
        raise ValueError(f"GMR commit changed for {path.name}.")
    if expected_fps is not None and not np.isclose(float(payload["fps"]), expected_fps):
        raise ValueError(f"FPS changed for {path.name}.")
    dof_pos = np.asarray(payload["dof_pos"], dtype=np.float64)
    root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
    root_rot = np.asarray(payload["root_rot"], dtype=np.float64)
    names = tuple(str(name) for name in payload["dof_names"])
    if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
        raise ValueError(f"GMR artifact must contain dof_pos[N,29], got {dof_pos.shape}.")
    if root_pos.shape != (len(dof_pos), 3) or root_rot.shape != (len(dof_pos), 4):
        raise ValueError("GMR root arrays do not match the DoF frame count.")
    if names != tuple(UnitreeG1DanceAdapter.GMR_DOF_NAMES):
        raise ValueError("GMR artifact DoF names/order do not match the MuJoCo qpos order.")
    if not all(np.all(np.isfinite(values)) for values in (dof_pos, root_pos, root_rot)):
        raise ValueError("GMR artifact contains non-finite values.")
    quaternion_norms = np.linalg.norm(root_rot, axis=1)
    if not np.allclose(quaternion_norms, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("GMR artifact contains non-unit root quaternions.")
    if len(root_rot) > 1 and np.any(np.sum(root_rot[1:] * root_rot[:-1], axis=1) < -1e-8):
        raise ValueError("GMR artifact root quaternion signs are discontinuous.")
    continuity = payload["continuity_limits"]
    if not isinstance(continuity, dict) or continuity.get("loop_closure_frames") != 0:
        raise ValueError("SMPL-direct GMR artifact must not apply generation-time loop closure.")
    collision = payload["collision_avoidance"]
    if not isinstance(collision, dict) or collision.get("preset") != "g1_self_collision_v2":
        raise ValueError("GMR artifact has missing or unsupported collision-avoidance metadata.")
    if expected_collision_avoidance is not None and bool(collision.get("enabled")) != bool(
        expected_collision_avoidance
    ):
        raise ValueError(f"Collision-avoidance mode changed for {path.name}.")
    expected_collision_values = (
        ("minimum_distance_m", expected_collision_min_distance),
        ("detection_distance_m", expected_collision_detection_distance),
        ("gain", expected_collision_gain),
    )
    if bool(collision.get("enabled")):
        for key, expected in expected_collision_values:
            if expected is not None and not np.isclose(float(collision.get(key, np.nan)), expected):
                raise ValueError(f"Collision-avoidance {key} changed for {path.name}.")


def artifact_entry(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {
        "source_motion_id": str(payload["source_motion_id"]),
        "artifact": path.name,
        "frames": int(np.asarray(payload["dof_pos"]).shape[0]),
        "fps": float(payload["fps"]),
        "source_sha256": str(payload["source_sha256"]),
        "smpl_model_sha256": str(payload["smpl_model_sha256"]),
        "gmr_commit": str(payload["retargeter_version"]),
        "pipeline_version": PIPELINE_VERSION,
        "source_format": SOURCE_FORMAT,
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path, default: object) -> object:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def quarantine_artifact(path: Path) -> Path | None:
    """Move a stale canonical artifact aside so matchers cannot load it by name."""
    if not path.exists():
        return None
    candidate = path.with_suffix(path.suffix + ".invalid")
    counter = 1
    while candidate.exists():
        candidate = path.with_suffix(path.suffix + f".invalid.{counter}")
        counter += 1
    path.replace(candidate)
    return candidate


def gmr_version(gmr_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(gmr_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError(f"Could not resolve a full GMR git commit from {gmr_root}: {commit!r}")
    return commit


def process_motion(args: argparse.Namespace, motion_path: Path) -> MotionBuildResult:
    motion_path = resolve(motion_path)
    if not motion_path.exists():
        raise FileNotFoundError(f"AIST++ motion not found: {motion_path}")
    output_path = resolve(args.output_root) / f"{motion_path.stem}.pkl"
    source_hash = sha256(motion_path)
    model_path = resolve(args.smpl_model_path) / "SMPL_NEUTRAL.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Prepared neutral SMPL model not found: {model_path}")
    model_hash = sha256(model_path)
    if output_path.exists() and not args.overwrite:
        try:
            validate_gmr_artifact(
                output_path,
                source_hash=source_hash,
                model_hash=model_hash,
                retargeter_version=args.gmr_version,
                expected_fps=float(args.fps),
                expected_collision_avoidance=bool(args.collision_avoidance),
                expected_collision_min_distance=float(args.collision_min_distance),
                expected_collision_detection_distance=float(args.collision_detection_distance),
                expected_collision_gain=float(args.collision_gain),
            )
            return MotionBuildResult(
                motion_path,
                output_path,
                f"valid cached {output_path.name}",
                artifact_entry(output_path),
            )
        except (AttributeError, KeyError, OSError, pickle.UnpicklingError, TypeError, ValueError):
            if not args.resume:
                quarantined = quarantine_artifact(output_path)
                if quarantined is not None:
                    print(f"{motion_path.stem}: quarantined stale {quarantined.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmr_script = Path(__file__).with_name("gmr_retarget_smpl_headless.py")
    build_path = output_path.with_suffix(output_path.suffix + ".build")
    if build_path.exists():
        build_path.unlink()
    gmr_command = [
        str(resolve(args.gmr_python)),
        str(gmr_script),
        "--gmr-root",
        str(resolve(args.gmr_root)),
        "--motion",
        str(motion_path),
        "--smpl-model-path",
        str(resolve(args.smpl_model_path)),
        "--output",
        str(build_path),
        "--motion-fps",
        str(args.fps),
        "--source-motion-id",
        motion_path.stem,
        "--source-sha256",
        source_hash,
        "--smpl-model-sha256",
        model_hash,
        "--retargeter-version",
        args.gmr_version,
        "--solver",
        args.solver,
    ]
    gmr_command.append("--velocity-limit" if args.velocity_limit else "--no-velocity-limit")
    gmr_command.append(
        "--collision-avoidance" if args.collision_avoidance else "--no-collision-avoidance"
    )
    gmr_command.extend(
        [
            "--collision-min-distance",
            str(args.collision_min_distance),
            "--collision-detection-distance",
            str(args.collision_detection_distance),
            "--collision-gain",
            str(args.collision_gain),
        ]
    )
    try:
        child_environment = os.environ.copy()
        numeric_threads = str(args.numeric_threads)
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            child_environment[variable] = numeric_threads
        child_environment["OMP_DYNAMIC"] = "FALSE"
        child_environment["MKL_DYNAMIC"] = "FALSE"
        creation_flags = 0
        if sys.platform == "win32":
            if args.child_priority == "idle":
                creation_flags = subprocess.IDLE_PRIORITY_CLASS
            elif args.child_priority == "below-normal":
                creation_flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS
        subprocess.run(
            gmr_command,
            check=True,
            cwd=resolve(args.gmr_root),
            capture_output=True,
            text=True,
            env=child_environment,
            creationflags=creation_flags,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"SMPL-direct GMR subprocess failed: {detail}") from exc
    validate_gmr_artifact(
        build_path,
        source_hash=source_hash,
        model_hash=model_hash,
        retargeter_version=args.gmr_version,
        expected_fps=float(args.fps),
    )
    build_path.replace(output_path)
    return MotionBuildResult(
        motion_path,
        output_path,
        f"wrote {output_path.name}",
        artifact_entry(output_path),
    )


def write_indexes(
    args: argparse.Namespace,
    results: list[MotionBuildResult],
    failures: list[dict[str, object]],
) -> None:
    output_root = resolve(args.output_root)
    manifest_path = output_root / MANIFEST_NAME
    failures_path = output_root / FAILURES_NAME
    replace_all = args.motion is None and args.split == "all" and args.limit is None

    previous_manifest = _load_json(manifest_path, {})
    previous_entries = [] if replace_all else list(
        previous_manifest.get("motions", []) if isinstance(previous_manifest, dict) else []
    )
    entries_by_id = {
        str(entry.get("source_motion_id")): entry
        for entry in previous_entries
        if isinstance(entry, dict) and entry.get("source_motion_id")
    }
    for result in results:
        entries_by_id[str(result.entry["source_motion_id"])] = result.entry
    failed_ids = {str(item["source_motion_id"]) for item in failures}
    for motion_id in failed_ids:
        entries_by_id.pop(motion_id, None)
        # A failed replacement must not leave a same-named old v2 artifact in
        # the canonical lookup path. Keep it recoverable, but make runtime
        # name-based discovery unable to select it after this index update.
        quarantine_artifact(output_root / f"{motion_id}.pkl")

    manifest = {
        "format_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "source_format": SOURCE_FORMAT,
        "retargeter": "GMR",
        "retargeter_version": args.gmr_version,
        "motions": sorted(entries_by_id.values(), key=lambda entry: str(entry["source_motion_id"])),
    }

    previous_failures = _load_json(failures_path, {})
    previous_failure_items = [] if replace_all else list(
        previous_failures.get("failures", []) if isinstance(previous_failures, dict) else []
    )
    failures_by_id = {
        str(item.get("source_motion_id")): item
        for item in previous_failure_items
        if isinstance(item, dict) and item.get("source_motion_id")
    }
    for result in results:
        failures_by_id.pop(str(result.entry["source_motion_id"]), None)
    for failure in failures:
        failures_by_id[str(failure["source_motion_id"])] = failure
    failure_report = {
        "format_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "source_format": SOURCE_FORMAT,
        "retargeter": "GMR",
        "retargeter_version": args.gmr_version,
        "failures": sorted(failures_by_id.values(), key=lambda item: str(item["source_motion_id"])),
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(failures_path, failure_report)


def main() -> int:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")
    if args.fps <= 0 or args.jobs <= 0 or args.numeric_threads <= 0:
        raise ValueError("--fps, --jobs, and --numeric-threads must be positive.")
    if args.collision_min_distance < 0:
        raise ValueError("--collision-min-distance must be non-negative.")
    if args.collision_detection_distance <= args.collision_min_distance:
        raise ValueError("--collision-detection-distance must exceed --collision-min-distance.")
    if not 0.0 < args.collision_gain <= 1.0:
        raise ValueError("--collision-gain must be in (0, 1].")
    args.gmr_root = resolve(args.gmr_root)
    args.gmr_version = gmr_version(args.gmr_root)
    motion_paths = collect_motion_paths(args)
    if not motion_paths:
        print("No AIST++ motions selected.")
        return 0

    results: list[MotionBuildResult] = []
    failures: list[dict[str, object]] = []
    model_hash = sha256(resolve(args.smpl_model_path) / "SMPL_NEUTRAL.pkl")
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(process_motion, args, path): path for path in motion_paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"{path.stem}: {result.message}")
            except Exception as exc:
                failures.append(
                    {
                        "source_motion_id": path.stem,
                        "source_path": str(resolve(path)),
                        "source_sha256": sha256(resolve(path)) if resolve(path).is_file() else None,
                        "smpl_model_sha256": model_hash,
                        "gmr_commit": args.gmr_version,
                        "stage": "AIST++ SMPL direct -> GMR",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                print(f"{path.stem}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
    write_indexes(args, results, failures)
    print(f"Finished {len(motion_paths) - len(failures)}/{len(motion_paths)} motion(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
