"""Regenerate selected existing GMR clips from SMPL into a separate v2 dataset."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from build_aistpp_gmr_dataset import (
    _atomic_json, _load_json, gmr_version, sha256, validate_gmr_artifact,
    quarantine_artifact,
)
from gmr_collision_projection import (
    PIPELINE_VERSION, project_trajectory, projection_settings, validate_v2_metadata, state_trajectory_digest,
)

from gmr_batch_lock import BatchAlreadyRunning, batch_lock

BASE = Path(__file__).resolve().parents[1]

# Exact predecessor retained only for cache compatibility after the lock-only fix.
LEGACY_BUILDER_SHA256 = "2822e03c9828986487df1712d2b477aa8a3940f0825e9b1d77ff09f8e2c6aa58"
LEGACY_WORKER_SHA256 = "f252de36ce7a7b8afa31302b6179982bb4cf2163a6fbdcbe196645e6e8d73c9a"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=BASE/"data/aistpp_gmr")
    parser.add_argument("--source-root", type=Path, default=BASE/"data/aistpp/motions")
    parser.add_argument("--output-root", type=Path, default=BASE/"data/aistpp_gmr_v2")
    parser.add_argument("--gmr-root", type=Path, default=BASE/".deps/GMR")
    parser.add_argument("--gmr-python", type=Path, default=BASE/".venv-gmr/Scripts/python.exe")
    parser.add_argument("--smpl-model-path", type=Path, default=BASE/"assets/body_models/smpl")
    parser.add_argument("--motion-id", action="append", help="Select one ID; repeat to select several.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--smoothing-weight", type=float, default=2.)
    parser.add_argument("--planning-clearance", type=float, default=.008, help="Metres (default: 8 mm).")
    parser.add_argument("--dry-run", action="store_true", help="List selection without writing files.")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true", help="Reuse matching v2 files; rebuild stale ones.")
    policy.add_argument("--overwrite", action="store_true", help="Rebuild selected v2 files.")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def check_roots(input_root: Path, source_root: Path, output_root: Path):
    """Reject overlapping roots, including paths resolved through links."""
    output_root = output_root.resolve()
    for protected in (input_root.resolve(), source_root.resolve(), (BASE/"data/aistpp_gmr").resolve()):
        if output_root == protected or protected in output_root.parents or output_root in protected.parents:
            raise ValueError("V2 output must be a separate folder, outside the original GMR/source roots.")


def collect_inputs(args):
    available = {p.stem: p for p in args.input_root.glob("*.pkl")}
    selected = sorted(set(args.motion_id)) if args.motion_id else sorted(available)
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(f"Motion IDs absent from --input-root: {missing}")
    if args.limit is not None:
        selected = selected[:args.limit]
    return [available[name] for name in selected]


def implementation_hash(gmr_root, *, legacy_lock_version=False):
    if legacy_lock_version and hashlib.sha256(inspect.getsource(worker).encode()).hexdigest() != LEGACY_WORKER_SHA256:
        return None  # Generation changed: do not accept the predecessor cache.
    files = [Path(__file__), *[Path(__file__).with_name(name) for name in (
        "gmr_collision_projection.py", "gmr_state_trajectory.py", "motion_bridges.py",
        "hermite_bounds.py", "gmr_retarget_smpl_headless.py",
        "gmr_retarget_bvh_headless.py", "aistpp_smpl.py")]]
    for folder in (gmr_root/"general_motion_retargeting", gmr_root/"assets/unitree_g1", BASE/"assets"):
        # Exclude the licensed body-model directory: SMPL is hashed separately.
        files.extend(p for p in folder.rglob("*") if p.is_file()
                     and p.suffix.lower() in (".py", ".json", ".xml", ".stl", ".obj")
                     and "body_models" not in p.parts)
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path).encode())
        file_hash = LEGACY_BUILDER_SHA256 if legacy_lock_version and path == Path(__file__) else sha256(path)
        digest.update(file_hash.encode())
    return digest.hexdigest()


def fingerprint(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def cached_payload(path, identity, *, compatible_implementation_sha256=None):
    validate_gmr_artifact(path, source_hash=identity["source_sha256"],
                          model_hash=identity["smpl_model_sha256"],
                          retargeter_version=identity["gmr_commit"],
                          expected_fps=identity["fps"], expected_pipeline_version=PIPELINE_VERSION,
                          expected_collision_avoidance=True)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    validate_v2_metadata(payload)
    if payload.get("v2_build_fingerprint") != fingerprint(identity):
        predecessor = dict(identity, implementation_sha256=compatible_implementation_sha256)
        if (compatible_implementation_sha256 is None
                or payload.get("v2_build") != predecessor
                or payload.get("v2_build_fingerprint") != fingerprint(predecessor)):
            raise ValueError("Source, settings, models, code or dependency versions changed.")
    return payload


def worker(job_path):
    import gmr_retarget_smpl_headless as runner
    job = json.loads(job_path.read_text(encoding="utf-8"))
    identity = job["identity"]
    settings = identity["projection"]
    report = {}
    def projection(models, qpos, fps, **kwargs):
        result, counts, validation = project_trajectory(models, qpos, fps, settings=settings, **kwargs)
        report.update(validation)
        return result, counts
    args = argparse.Namespace(
        gmr_root=Path(job["gmr_root"]), motion=Path(job["source"]),
        smpl_model_path=Path(job["smpl_model_path"]), solver="daqp", velocity_limit=True,
        collision_avoidance=True, collision_min_distance=.005,
        collision_detection_distance=.08, collision_gain=.85,
        collision_validation_model=runner.DEFAULT_COLLISION_VALIDATION_MODEL,
        motion_fps=identity["fps"], max_joint_speed=settings["max_joint_speed_rad_s"],
        max_root_speed=settings["max_root_speed_m_s"],
        max_root_angular_speed=settings["max_root_angular_speed_rad_s"],
        source_motion_id=identity["motion_id"], retargeter_version=identity["gmr_commit"],
        source_sha256=identity["source_sha256"], smpl_model_sha256=identity["smpl_model_sha256"])
    start = time.perf_counter()
    payload = runner.retarget(args, trajectory_filter=projection)
    payload["dof_vel"] = report.pop("_dof_velocity")
    payload["dof_acc"] = report.pop("_dof_acceleration")
    payload.update(pipeline_version=PIPELINE_VERSION, motion_version="gmr_v2",
                   projection=settings, projection_validation=report,
                   v2_build_fingerprint=fingerprint(identity), v2_build=identity,
                   generation_seconds=time.perf_counter()-start)
    payload["state_trajectory_sha256"] = state_trajectory_digest(payload)
    validate_v2_metadata(payload)
    output = Path(job["output"])
    temporary = output.with_suffix(".pkl.build")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    cached_payload(temporary, identity)
    temporary.replace(output)


def process_one(args, original, common):
    source = args.source_root/f"{original.stem}.pkl"
    if not source.is_file():
        raise FileNotFoundError(f"Original SMPL source required: {source}")
    with original.open("rb") as handle:
        old = pickle.load(handle)
    fps = float(old["fps"])
    if not np.isfinite(fps) or fps <= 0 or old.get("source_format") != "aistpp_smpl_direct":
        raise ValueError("Input must be an SMPL-direct GMR artifact with positive FPS.")
    if old.get("source_motion_id") != original.stem:
        raise ValueError("Input source_motion_id does not match its filename.")
    identity = dict(common, motion_id=original.stem, fps=fps,
                    source_sha256=sha256(source), original_gmr_sha256=sha256(original))
    output = args.output_root/original.name
    payload = None
    if output.exists() and args.overwrite:
        quarantine_artifact(output)
    if output.exists() and not args.overwrite:
        if not args.resume:
            raise FileExistsError(f"{output.name} exists; use --resume or --overwrite.")
        try:
            payload = cached_payload(output, identity, compatible_implementation_sha256=args.compatible_implementation_sha256)
        except (ValueError, TypeError, KeyError, AttributeError, EOFError, OSError, pickle.UnpicklingError):
            # A failed rebuild must not leave a stale file at the runtime lookup name.
            quarantine_artifact(output)
    reused = payload is not None
    if not reused:
        job = dict(identity=identity, source=str(source), output=str(output),
                   gmr_root=str(args.gmr_root), smpl_model_path=str(args.smpl_model_path))
        job_path = args.output_root/"logs"/f"{original.stem}.job.json"
        _atomic_json(job_path, job)
        environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        result = subprocess.run([str(args.gmr_python), str(Path(__file__).resolve()),
                                 "--worker", str(job_path)], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", env=environment,
                                creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0)
        log = args.output_root/"logs"/f"{original.stem}.log"
        log.write_text(result.stdout+"\n"+result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Generation/validation failed; see {log}. {(result.stderr or result.stdout)[-1200:]}")
        payload = cached_payload(output, identity)
    return dict(source_motion_id=original.stem, artifact=output.name, status="cached" if reused else "written",
                artifact_sha256=sha256(output), fps=fps, frames=len(payload["dof_pos"]),
                generation_seconds=payload["generation_seconds"],
                metrics=payload["projection_validation"]["metrics"],
                solver_failures=payload["continuity_limits"]["limited_values"]["solver_failures"],
                v2_build_fingerprint=payload["v2_build_fingerprint"])


def main():
    args = parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    for name in ("input_root", "source_root", "output_root", "gmr_root", "gmr_python", "smpl_model_path"):
        setattr(args, name, getattr(args, name).resolve())
    check_roots(args.input_root, args.source_root, args.output_root)
    settings = projection_settings(args.smoothing_weight, args.planning_clearance)
    if args.jobs < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("--jobs and --limit must be positive.")
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)
    originals = collect_inputs(args)
    print(f"Selected {len(originals)} existing GMR clips; source: {args.source_root}", flush=True)
    print(f"V2 output: {args.output_root}", flush=True)
    if args.dry_run:
        missing = [p.stem for p in originals if not (args.source_root/p.name).is_file()]
        print(f"Missing SMPL sources: {len(missing)}; examples: {missing[:5]}")
        print("First selected IDs: " + ", ".join(p.stem for p in originals[:5]))
        return 1 if missing else 0
    if not originals:
        raise ValueError("No input GMR files selected.")
    if not (args.resume or args.overwrite) and any((args.output_root/p.name).exists() for p in originals):
        raise FileExistsError("Selected v2 files already exist; use --resume or --overwrite.")
    versions = subprocess.check_output([str(args.gmr_python), "-c",
        "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ['numpy','mujoco','mink','qpsolvers','daqp','scipy','torch','smplx']}))"], text=True)
    common = dict(pipeline_version=PIPELINE_VERSION, projection=settings,
                  smpl_model_sha256=sha256(args.smpl_model_path/"SMPL_NEUTRAL.pkl"),
                  gmr_commit=gmr_version(args.gmr_root), implementation_sha256=implementation_hash(args.gmr_root),
                  dependencies=json.loads(versions))
    args.compatible_implementation_sha256 = implementation_hash(args.gmr_root, legacy_lock_version=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = args.output_root/".batch.lock"
    with batch_lock(lock):
        prior = _load_json(args.output_root/"manifest.json", {})
        entries = {x["source_motion_id"]: x for x in prior.get("motions", [])}
        prior_failures = _load_json(args.output_root/"failures.json", {})
        failures = {x["source_motion_id"]: x for x in prior_failures.get("failures", [])}
        selected_failures = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            pending = {executor.submit(process_one, args, path, common): path for path in originals}
            for index, future in enumerate(as_completed(pending), 1):
                path = pending[future]
                try:
                    entry = future.result()
                    entries[path.stem] = entry
                    failures.pop(path.stem, None)
                    print(f"[{index}/{len(originals)}] {path.stem}: {entry['status']}", flush=True)
                except Exception as exc:
                    selected_failures += 1
                    entries.pop(path.stem, None)
                    failures[path.stem] = dict(source_motion_id=path.stem, error=str(exc))
                    print(f"[{index}/{len(originals)}] {path.stem}: FAILED {exc}", flush=True)
                # Checkpoint after each completion so an interrupted batch can resume.
                _atomic_json(args.output_root/"manifest.json", dict(motion_version="gmr_v2",
                    pipeline_version=PIPELINE_VERSION, motions=[entries[k] for k in sorted(entries)]))
                _atomic_json(args.output_root/"failures.json", dict(failures=[failures[k] for k in sorted(failures)]))
        print(f"Finished {len(originals)-selected_failures}/{len(originals)} selected clips.", flush=True)
        return int(selected_failures > 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BatchAlreadyRunning as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
