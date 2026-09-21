"""Convert separated FineDance SMPL clips directly to validated GMR v2 motions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

from aistpp_smpl import load_aistpp_motion
from build_aistpp_gmr_dataset import sha256, gmr_version, quarantine_artifact
from finedance_batch_state import atomic_json as _atomic_json, atomic_text, read_checkpoint, cache_builder_hash
import build_gmr_v2 as v2
from gmr_batch_lock import BatchAlreadyRunning, batch_lock
from gmr_collision_projection import PIPELINE_VERSION, projection_settings

BASE = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=BASE / "data/finedance_aistpp")
    parser.add_argument("--output-root", type=Path, default=BASE / "data/finedance_gmr_v2")
    parser.add_argument("--gmr-root", type=Path, default=BASE / ".deps/GMR")
    parser.add_argument("--gmr-python", type=Path, default=BASE / ".venv-gmr/Scripts/python.exe")
    parser.add_argument("--smpl-model-path", type=Path, default=BASE / "assets/body_models/smpl")
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--motion-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--smoothing-weight", type=float, default=2.)
    parser.add_argument("--planning-clearance", type=float, default=.008)
    parser.add_argument("--dry-run", action="store_true", help="Validate selection without writing files.")
    parser.add_argument("--repair-only", action="store_true", help="Rebuild output metadata from valid saved motions; do not generate motions.")
    parser.add_argument("--audit-only", action="store_true", help="Audit selected saved outputs without rebuilding.")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--jobs and --limit must be positive")
    if sum((args.dry_run, args.audit_only, args.repair_only)) > 1:
        parser.error("--dry-run, --audit-only and --repair-only are mutually exclusive")
    for name in ("input_root", "output_root", "gmr_root", "gmr_python", "smpl_model_path"):
        setattr(args, name, getattr(args, name).resolve())
    projection_settings(args.smoothing_weight, args.planning_clearance)
    return args


def check_roots(args):
    for protected in (args.input_root, BASE / "data/finedance", BASE / "data/aistpp",
                      BASE / "data/aistpp_gmr", BASE / "data/aistpp_gmr_v2"):
        v2.check_roots(protected, protected, args.output_root)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes the dataset: {relative}")
    return path


def collect_inputs(args):
    manifest = read_json(args.input_root / "manifest.json")
    if (manifest.get("export_complete") is not True or manifest.get("dry_run") is not False
            or manifest.get("format_version") != 1 or manifest.get("fps") != 60):
        raise ValueError("Expected a completed 60 FPS FineDance segmentation export.")
    clips = manifest["clips"]
    available = {}
    for clip in clips:
        name = clip["clip_id"]
        if not re.fullmatch(r"finedance_\d+_\d{7,}_\d{7,}", name) or name in available:
            raise ValueError(f"Invalid or duplicate clip ID: {name}")
        if clip["split"] not in SPLITS:
            raise ValueError(f"Invalid split for {name}")
        available[name] = clip
    if manifest.get("clip_count") != len(available):
        raise ValueError("Manifest clip_count does not match clips.")
    membership = {}
    for split in SPLITS:
        names = (args.input_root / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        for name in (line.strip() for line in names if line.strip()):
            if name in membership or name not in available or available[name]["split"] != split:
                raise ValueError(f"Duplicate or inconsistent split membership: {name}")
            membership[name] = split
    if set(membership) != set(available):
        raise ValueError("Split files and manifest have different clip membership.")
    selected = sorted(set(args.motion_id)) if args.motion_id else sorted(available)
    for name in selected:
        if name not in available:
            raise ValueError(f"Unknown motion ID: {name}")
        if args.motion_id and args.split != "all" and membership[name] != args.split:
            raise ValueError(f"Motion {name} is outside --split {args.split}")
    selected = [name for name in selected if args.split == "all" or membership[name] == args.split]
    if args.limit is not None:
        selected = selected[:args.limit]
    return manifest, [available[name] for name in selected]


def validate_source(root, manifest, clip):
    name = clip["clip_id"]
    start, end, frames = clip["start_frame"], clip["end_frame_exclusive"], clip["frames"]
    if (any(type(x) is not int for x in (start, end, frames)) or start < 0
            or not 480 <= frames <= 720 or end - start != frames
            or name != f"finedance_{clip['source_id']}_{start:07d}_{end:07d}"):
        raise ValueError(f"Inconsistent FineDance segment boundaries: {name}")
    for key, expected in (("start_seconds", start / 60), ("end_seconds", end / 60),
                          ("duration_seconds", frames / 60), ("authored_duration_seconds", (frames - 1) / 60)):
        if not np.isclose(float(clip[key]), expected, rtol=0, atol=1e-8):
            raise ValueError(f"Inconsistent {key}: {name}")
    if clip["motion_path"] != f"motions/{name}.pkl" or clip["audio_path"] != f"audio/{name}.wav":
        raise ValueError(f"Noncanonical source paths: {name}")
    source = contained(root, clip["motion_path"])
    if not contained(root, clip["audio_path"]).is_file():
        raise FileNotFoundError(clip["audio_path"])
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    expected = dict(fps=60, source_dataset="FineDance", source_id=clip["source_id"],
                    source_start_seconds=start / 60, source_fps=manifest["source_fps"],
                    root_y_offset_m=manifest["root_y_offset_m"], body_only_approximation=True)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Inconsistent source provenance {key}: {name}")
    poses, translations = load_aistpp_motion(source)
    if len(poses) != frames or len(translations) != frames:
        raise ValueError(f"Source frame count differs from manifest: {name}")
    return source


def cached_payload(path, identity):
    payload = v2.cached_payload(path, identity)
    if (payload.get("v2_build") != identity or payload.get("source_motion_id") != identity["motion_id"]
            or len(payload["dof_pos"]) != identity["clip"]["frames"]):
        raise ValueError("Output timing or FineDance provenance does not match source.")
    return payload


@lru_cache(maxsize=1)
def audit_function():
    # Explicit path avoids collision with Python's own "test" package.
    spec = importlib.util.spec_from_file_location("finedance_smoothness_audit",
                                                Path(__file__).parent / "test/audit_gmr_smoothness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit


def audit_motion(path, payload):
    # Use the per-file audit, avoiding its CLI's AIST++ catalog assumptions.
    ranges = {e.attrib["name"].removesuffix("_joint"): tuple(map(float, e.attrib["range"].split()))
              for e in ET.parse(BASE / "assets/g1_29dof.xml").iter("joint")
              if "range" in e.attrib and "name" in e.attrib}
    row = audit_function()(path, ranges)
    settings = payload["projection"]
    if (row["v2_speed_rad_s"] > settings["max_joint_speed_rad_s"] + 1e-5
            or row["v2_acceleration_rad_s2"] > settings["max_joint_acceleration_rad_s2"] + 1e-5
            or row["v2_position_excess_rad"] > 1e-5
            or row["root_speed_violation"] or row["root_angular_speed_violation"]):
        raise ValueError("Saved-trajectory smoothness/kinematic audit failed.")
    row.update(payload["projection_validation"]["metrics"])
    return row


def process_one(args, manifest, clip, common):
    name = clip["clip_id"]
    output = args.output_root / f"{name}.pkl"
    stage = args.output_root / "logs" / f"{name}.pending.pkl"
    log = args.output_root / "logs" / f"{name}.log"
    try:
        source = validate_source(args.input_root, manifest, clip)
        identity = dict(common, motion_id=name, fps=60., source_sha256=sha256(source), clip=clip)
        payload = None
        if output.exists():
            if args.overwrite:
                quarantine_artifact(output)
            elif not args.resume:
                raise FileExistsError(f"{output.name} exists; use --resume or --overwrite.")
            else:
                try:
                    payload = cached_payload(output, identity)
                except (ValueError, TypeError, KeyError, AttributeError, EOFError, OSError, pickle.UnpicklingError):
                    quarantine_artifact(output)
        reused = payload is not None
        if not reused:
            # The shared worker publishes only to a staging path. The final path
            # is published after FineDance timing/provenance and audit checks too.
            job = dict(identity=identity, source=str(source), output=str(stage),
                       gmr_root=str(args.gmr_root), smpl_model_path=str(args.smpl_model_path))
            job_path = args.output_root / "logs" / f"{name}.job.json"
            _atomic_json(job_path, job)
            environment = os.environ.copy()
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = "1"
            with log.open("w", encoding="utf-8") as handle:
                result = subprocess.run([str(args.gmr_python), str(Path(v2.__file__).resolve()),
                                         "--worker", str(job_path)], stdout=handle, stderr=subprocess.STDOUT,
                                        env=environment, creationflags=(subprocess.BELOW_NORMAL_PRIORITY_CLASS
                                                                      if os.name == "nt" else 0))
            if result.returncode:
                raise RuntimeError(f"Generation/validation failed; see {log}")
            payload = cached_payload(stage, identity)
        row = audit_motion(output if reused else stage, payload)
        row["motion_id"] = name
        if sha256(source) != identity["source_sha256"]:
            raise ValueError("Source changed during generation.")
        if not reused:
            stage.replace(output)
        return dict(source_motion_id=name, artifact=output.name, status="cached" if reused else "written",
                    artifact_sha256=sha256(output), fps=60., frames=clip["frames"], clip=clip,
                    generation_seconds=payload["generation_seconds"], audit=row,
                    metrics=payload["projection_validation"]["metrics"],
                    solver_failures=payload["continuity_limits"]["limited_values"]["solver_failures"],
                    v2_build_fingerprint=payload["v2_build_fingerprint"], v2_build=identity)
    except Exception as exc:
        if output.exists() and (args.resume or args.overwrite):
            quarantine_artifact(output)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\nFineDance conversion failed: {exc}\n")
        raise


def checkpoint(root, entries, failures, input_root):
    motions = [entries[name] for name in sorted(entries)]
    _atomic_json(root / "manifest.json", dict(source_dataset="FineDance", input_root=str(input_root),
                 motion_version="gmr_v2", pipeline_version=PIPELINE_VERSION, motions=motions))
    _atomic_json(root / "failures.json", dict(failures=[failures[name] for name in sorted(failures)]))
    for split in SPLITS:
        atomic_text(root / f"{split}.txt", "".join(
            f"{e['source_motion_id']}\n" for e in motions if e["clip"]["split"] == split))


def load_output_state(args, manifest):
    """Called under the batch lock; recover metadata without trusting stale PKLs."""
    recover = args.resume or args.overwrite or args.repair_only
    prior, damaged = read_checkpoint(args.output_root / "manifest.json", "motions", recover=recover)
    if prior and (prior.get("source_dataset") != "FineDance" or prior.get("input_root") != str(args.input_root)):
        raise ValueError("Output directory belongs to another dataset.")
    report, _ = read_checkpoint(args.output_root / "failures.json", "failures", recover=recover)
    failures = {e["source_motion_id"]: e for e in report.get("failures", [])}
    entries = {e["source_motion_id"]: e for e in prior.get("motions", [])
               if (args.output_root / e["artifact"]).is_file()}
    if not args.repair_only and not (damaged and recover):
        return entries, failures
    entries = {}
    clips = {c["clip_id"]: c for c in manifest["clips"]}
    manifest_hash = sha256(args.input_root / "manifest.json")
    paths = sorted(args.output_root.glob("*.pkl"))
    for index, path in enumerate(paths, 1):
        try:
            clip = clips[path.stem]
            source = validate_source(args.input_root, manifest, clip)
            with path.open("rb") as handle:
                saved = pickle.load(handle)
            identity = saved["v2_build"]
            if (identity.get("source_dataset") != "FineDance" or identity.get("motion_id") != path.stem
                    or identity.get("clip") != clip or identity.get("source_sha256") != sha256(source)
                    or identity.get("segmentation_manifest_sha256") != manifest_hash):
                raise ValueError("Saved output does not match current FineDance source/provenance.")
            payload = cached_payload(path, identity)
            row = audit_motion(path, payload)
            entries[path.stem] = dict(source_motion_id=path.stem, artifact=path.name, status="recovered",
                artifact_sha256=sha256(path), fps=payload["fps"], frames=clip["frames"], clip=clip,
                generation_seconds=payload["generation_seconds"], audit=row,
                metrics=payload["projection_validation"]["metrics"],
                solver_failures=payload["continuity_limits"]["limited_values"]["solver_failures"],
                v2_build_fingerprint=payload["v2_build_fingerprint"], v2_build=identity)
            failures.pop(path.stem, None)
        except Exception as exc:
            failures[path.stem] = dict(source_motion_id=path.stem, error=f"Recovery validation failed: {exc}")
        if index % 50 == 0:
            print(f"Checked {index}/{len(paths)} saved motions for metadata recovery.", flush=True)
    checkpoint(args.output_root, entries, failures, args.input_root)
    print(f"Recovered {len(entries)}/{len(paths)} saved motions; {len(paths) - len(entries)} need rebuilding.", flush=True)
    return entries, failures


def audit_existing(args, manifest, clips):
    rows, errors = [], []
    for clip in clips:
        try:
            source = validate_source(args.input_root, manifest, clip)
            path = args.output_root / f"{clip['clip_id']}.pkl"
            with path.open("rb") as handle:
                identity = pickle.load(handle)["v2_build"]
            if (identity["source_sha256"] != sha256(source) or identity["clip"] != clip
                    or identity["segmentation_manifest_sha256"] != sha256(args.input_root / "manifest.json")):
                raise ValueError("Saved motion source/provenance is stale.")
            rows.append(audit_motion(path, cached_payload(path, identity)))
        except Exception as exc:
            errors.append(dict(source_motion_id=clip["clip_id"], error=str(exc)))
    _atomic_json(args.output_root / "audit.json", dict(motions=rows, errors=errors))
    print(f"Audited {len(rows)}/{len(clips)} clips; {len(errors)} failures.", flush=True)
    return int(bool(errors))


def main(argv=None):
    args = parse_args(argv)
    check_roots(args)
    manifest, clips = collect_inputs(args)
    print(f"Selected {len(clips)} FineDance clips at 60 FPS; output: {args.output_root}", flush=True)
    if args.dry_run:
        failures = []
        for clip in clips:
            try:
                validate_source(args.input_root, manifest, clip)
            except Exception as exc:
                failures.append(f"{clip['clip_id']}: {exc}")
        print("First selected IDs: " + ", ".join(c["clip_id"] for c in clips[:5]))
        print(f"Validated {len(clips) - len(failures)}/{len(clips)} sources.")
        for error in failures:
            print(error, file=sys.stderr)
        return int(bool(failures))
    if not clips:
        raise ValueError("No clips selected.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with batch_lock(args.output_root / ".batch.lock"):
        if args.audit_only:
            return audit_existing(args, manifest, clips)
        entries, failures = load_output_state(args, manifest)
        if args.repair_only:
            return int(bool(failures))
        if not (args.resume or args.overwrite) and any((args.output_root / f"{c['clip_id']}.pkl").exists() for c in clips):
            raise FileExistsError("Selected outputs already exist; use --resume or --overwrite.")
        versions = subprocess.check_output([str(args.gmr_python), "-c",
            "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in "
            "['numpy','mujoco','mink','qpsolvers','daqp','scipy','torch','smplx']}))"], text=True)
        common = dict(source_dataset="FineDance", pipeline_version=PIPELINE_VERSION,
                      projection=projection_settings(args.smoothing_weight, args.planning_clearance),
                      smpl_model_sha256=sha256(args.smpl_model_path / "SMPL_NEUTRAL.pkl"),
                      gmr_commit=gmr_version(args.gmr_root), dependencies=json.loads(versions),
                      implementation_sha256=v2.fingerprint(dict(shared=v2.implementation_hash(args.gmr_root),
                          finedance=cache_builder_hash(Path(__file__)), validator=sha256(Path(v2.__file__).with_name("build_aistpp_gmr_dataset.py")),
                          audit=sha256(Path(__file__).parent / "test/audit_gmr_smoothness.py"))),
                      segmentation_manifest_sha256=sha256(args.input_root / "manifest.json"),
                      segmentation={key: value for key, value in manifest.items() if key not in ("clips", "sources", "excluded")})
        (args.output_root / "logs").mkdir(exist_ok=True)
        for clip in clips:
            entries.pop(clip["clip_id"], None)
        checkpoint(args.output_root, entries, failures, args.input_root)
        failed = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            pending = {executor.submit(process_one, args, manifest, clip, common): clip for clip in clips}
            for index, future in enumerate(as_completed(pending), 1):
                name = pending[future]["clip_id"]
                try:
                    entries[name] = future.result()
                    failures.pop(name, None)
                    status = entries[name]["status"]
                except Exception as exc:
                    failed += 1
                    entries.pop(name, None)
                    failures[name] = dict(source_motion_id=name, error=str(exc))
                    status = f"FAILED: {exc}"
                checkpoint(args.output_root, entries, failures, args.input_root)
                print(f"[{index}/{len(clips)}] {name}: {status}", flush=True)
        return int(failed > 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BatchAlreadyRunning as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
