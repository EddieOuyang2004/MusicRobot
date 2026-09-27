"""Split >16 s AIST++ pairs; optionally rebuild, validate and activate the catalogue.

Original files are never edited. No retargeting is performed. Resume is automatic.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import uuid

import numpy as np
import soundfile as sf

import build_combined_music_catalog as catalog_builder
from build_combined_music_catalog import atomic_json, digest, fingerprint, read_json, relative
from gmr_batch_lock import batch_lock
from gmr_collision_projection import state_trajectory_digest, validate_v2_metadata

BASE = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = BASE / "data/music_catalog_combined/catalog.json"
DEFAULT_OUTPUT = BASE / "data/music_catalog_segmented"
FPS = 60
ROBOT_ARRAYS = ("root_pos", "root_rot", "dof_pos", "dof_vel", "dof_acc")


def segment_bounds(frames: int) -> list[tuple[int, int]]:
    if frames < 2:
        raise ValueError("A motion must contain at least two frames")
    if frames <= 16 * FPS:
        return [(0, frames)]
    count = min(range(2, math.ceil(frames / (8 * FPS)) + 1),
                key=lambda n: (max(8 * FPS - frames / n, 0, frames / n - 10 * FPS),
                               abs(frames / n - 9 * FPS), n))
    size, remainder = divmod(frames, count)
    bounds, start = [], 0
    for i in range(count):
        end = start + size + (i < remainder)
        bounds.append((start, end))
        start = end
    return bounds


def child_id(parent, start, end):
    return f"{parent}__f{start:07d}_{end:07d}"


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def rebase_metadata(metadata, source_dir, destination_dir):
    result = copy.deepcopy(metadata)
    for roots in result["datasets"].values():
        for key in ("source_root", "gmr_root"):
            roots[key] = relative(source_dir / roots[key], destination_dir)
    for key in ("embedding_model", "tag_model"):
        model = result.get("extractor", {}).get(key)
        if model and model.get("path"):
            model["path"] = relative(source_dir / model["path"], destination_dir)
    result["arrays_file"] = relative(source_dir / result["arrays_file"], destination_dir)
    return result


def inventory(args):
    """Read-only, including under --dry-run; take membership from the active library."""
    snapshot = args.output_dir / "input_catalog.json"
    origin = snapshot if snapshot.exists() else args.catalog.resolve()
    metadata = read_json(origin)
    if metadata.get("segmentation_input_catalog", str(args.catalog)) != str(args.catalog):
        raise ValueError("This output folder belongs to another input catalogue; choose a new --output-dir")
    if metadata.get("schema_version") != 3 or set(metadata["datasets"]) != {"aistpp", "finedance"}:
        raise ValueError("Expected a combined schema-3 catalogue with both datasets")
    roots = {ds: {k: (origin.parent / v).resolve() for k, v in values.items()}
             for ds, values in metadata["datasets"].items()}
    for values in roots.values():
        for path in values.values():
            if args.output_dir == path or args.output_dir in path.parents or path in args.output_dir.parents:
                raise ValueError("Output must not overlap any input dataset")
    if args.catalog.resolve().is_relative_to(args.output_dir):
        raise ValueError("Input catalogue must be outside output directory")
    originals = {e["motion_id"]: e for e in catalog_builder.dataset_entries("aistpp", **roots["aistpp"])}
    selected = sorted(k for k, v in metadata["motions"].items() if v["dataset_id"] == "aistpp")
    if args.limit is not None:
        selected = [k for k in selected if metadata["motions"][k]["duration_seconds"] > 16][:args.limit]
    parents = []
    for name in selected:
        entry = originals[name]
        hashes = catalog_builder.validate_inputs(entry, **roots["aistpp"])
        frames = entry["frames"]
        bounds = segment_bounds(frames)
        info = sf.info(roots["aistpp"]["source_root"] / entry["audio_path"])
        # Use the frame clock, never stretch or pad audio to hide a mismatch.
        if abs(info.frames - round(frames / FPS * info.samplerate)) > 1:
            raise ValueError(f"{name}: paired audio differs from frame clock by more than one sample")
        children = [dict(motion_id=child_id(name, a, b) if len(bounds) > 1 else name,
                         start_frame=a, end_frame_exclusive=b, frames=b-a,
                         duration_seconds=(b-a)/FPS) for a, b in bounds]
        parents.append(dict(entry=entry, hashes=hashes, children=children))
    if not parents:
        raise ValueError("No eligible AIST++ motions")
    return origin, metadata, roots, parents


class SliceValidator:
    """Recheck stored Hermite states using the same two collision models as GMR v2."""
    def __init__(self, model, gmr_model):
        import mujoco
        from gmr_collision_projection import collision_states
        from gmr_retarget_smpl_headless import activate_required_collision_geoms
        from gmr_retarget_bvh_headless import qpos_joint_names
        self.models = [mujoco.MjModel.from_xml_path(str(path)) for path in (gmr_model, model)]
        self.low, self.high = np.full(29, -np.inf), np.full(29, np.inf)
        self.names = qpos_joint_names(self.models[0])
        for item in self.models:
            activate_required_collision_geoms(item)
            if item.nq != 36 or qpos_joint_names(item) != self.names:
                raise ValueError("Collision model joint order mismatch")
            for joint in range(item.njnt):
                address = int(item.jnt_qposadr[joint]) - 7
                if address >= 0 and item.jnt_limited[joint]:
                    self.low[address] = max(self.low[address], item.jnt_range[joint, 0])
                    self.high[address] = min(self.high[address], item.jnt_range[joint, 1])
        self.states = collision_states(self.models)

    def validate(self, payload):
        from motion_bridges import AuthoredTrajectory, HermiteBridge
        from gmr_state_trajectory import _curve_safe
        from gmr_retarget_smpl_headless import validate_joint_limits, _violates_configured_clearance
        if tuple(payload["dof_names"]) != self.names:
            raise ValueError("Artifact joint order mismatch")
        settings = payload["projection"]
        qpos = np.column_stack([payload["root_pos"], payload["root_rot"], payload["dof_pos"]])
        for model in self.models:
            validate_joint_limits(model, qpos, self.names)
        trajectory = AuthoredTrajectory(payload["dof_pos"], FPS, velocities=payload["dof_vel"],
                                        accelerations=payload["dof_acc"])
        peaks = dict(speed_rad_s=0., acceleration_rad_s2=0., jerk_rad_s3=0.)
        for frame in range(len(qpos)-1):
            bridge = HermiteBridge(1/FPS, trajectory.segments[:, frame])
            if not bridge.within_limits(0, self.low, self.high):
                raise ValueError(f"Child joint-range violation at frame {frame}")
            for derivative, label in enumerate(peaks, 1):
                low, high = bridge.extrema(derivative)
                peaks[label] = max(peaks[label], float(np.max(np.maximum(np.abs(low), np.abs(high)))))
            if not _curve_safe(bridge, qpos[frame], self.states, settings["validation_clearance_m"],
                               settings["validation_samples"]):
                raise ValueError(f"Child collision violation at frame {frame}")
        if _violates_configured_clearance(self.states, qpos[-1], settings["validation_clearance_m"]):
            raise ValueError("Child final-frame collision violation")
        return dict(passed=True, clearance_m=settings["validation_clearance_m"], violating_samples=0,
                    samples_per_interval=settings["validation_samples"], models=len(self.models),
                    interpolation="quintic_hermite_states", continuous_peaks=peaks,
                    validation_method="revalidated_stored_state_slice", frames=len(qpos))


def slice_payloads(source, robot, start, end, name, provenance, validator):
    validate_v2_metadata(robot)
    if any(len(robot[key]) != len(source["smpl_poses"]) for key in ROBOT_ARRAYS):
        raise ValueError("Parent state array lengths differ")
    human = copy.deepcopy(source)
    for key in ("smpl_poses", "smpl_trans"):
        human[key] = source[key][start:end].copy()
    human_bytes = pickle.dumps(human, protocol=pickle.HIGHEST_PROTOCOL)
    result = copy.deepcopy(robot)
    for key in ROBOT_ARRAYS:
        result[key] = robot[key][start:end].copy()
    import hashlib
    result.update(source_motion_id=name, source_sha256=hashlib.sha256(human_bytes).hexdigest(),
                  segmentation=provenance)
    result["parent_generation"] = {k: result.pop(k) for k in
        ("v2_build", "v2_build_fingerprint", "projection_validation", "generation_seconds") if k in result}
    result["v2_build"] = dict(operation="state_slice", provenance=provenance,
                               source_sha256=result["source_sha256"])
    result["v2_build_fingerprint"] = fingerprint(result["v2_build"])
    result["projection_validation"] = validator.validate(result)
    result["state_trajectory_sha256"] = state_trajectory_digest(result)
    validate_v2_metadata(result)
    return human_bytes, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)


def validation_identity(args):
    files = [Path(__file__), *[Path(__file__).with_name(n) for n in (
        "gmr_collision_projection.py", "gmr_state_trajectory.py", "motion_bridges.py", "hermite_bounds.py",
        "gmr_retarget_smpl_headless.py", "gmr_retarget_bvh_headless.py", "music_motion_catalog.py",
        "realtime_music_humanoid_dancer.py", "unitree_g1_dance_adapter.py")]]
    for root in (args.model.parent, args.gmr_model.parent):
        files.extend(p for p in root.rglob("*") if p.suffix.lower() in (".xml", ".stl", ".obj"))
    return dict(files={str(p.resolve()): digest(p) for p in sorted(set(files))},
                dependencies={n: importlib.metadata.version(n) for n in ("numpy", "scipy", "mujoco", "soundfile")})


def export(args, roots, parents, identity):
    generation = fingerprint(dict(parents=parents, validation=identity))[:24]
    root = args.output_dir / "assets" / generation
    source_root, gmr_root = root / "aistpp", root / "aistpp_gmr_v2"
    for folder in (source_root / "motions", source_root / "audio", gmr_root, root / "cache"):
        folder.mkdir(parents=True, exist_ok=True)
    clips, records, failures = [], [], []
    validator = None
    preflight = catalog_builder.MotionPreflightValidator(args.model, gmr_root)
    total = sum(len(p["children"]) for p in parents)
    cached_count = 0
    for parent in parents:
        entry = parent["entry"]
        old_source = roots["aistpp"]["source_root"] / entry["motion_path"]
        old_audio = roots["aistpp"]["source_root"] / entry["audio_path"]
        old_robot = roots["aistpp"]["gmr_root"] / entry["gmr_record"]["artifact"]
        source, robot, audio = None, None, None
        for child in parent["children"]:
            name = child["motion_id"]
            start, end = child["start_frame"], child["end_frame_exclusive"]
            split = name != entry["motion_id"]
            provenance = dict(parent_motion_id=entry["motion_id"], parent_hashes=parent["hashes"],
                              start_frame=start, end_frame_exclusive=end, fps=FPS)
            clip = {k: v for k, v in entry.items() if k not in ("gmr_record", "dataset_id", "segmentation_hash")}
            clip.update(motion_id=name, music_id=name if split else entry["music_id"],
                        parent_motion_id=entry["motion_id"], frames=end-start,
                        duration_seconds=(end-start)/FPS, authored_duration_seconds=(end-start-1)/FPS,
                        start_seconds=start/FPS, end_seconds=end/FPS,
                        motion_path=f"motions/{name}.pkl", audio_path=f"audio/{name}.wav", provenance=provenance)
            paths = dict(source=source_root / clip["motion_path"], audio=source_root / clip["audio_path"],
                         artifact=gmr_root / (name + ".pkl"))
            cache_path = root / "cache" / (name + ".json")
            key = fingerprint(dict(clip=clip, validation=identity))
            cached = catalog_builder.read_cache(cache_path, key)
            try:
                if cached and all(p.is_file() and digest(p) == cached["hashes"][k] for k, p in paths.items()):
                    record = cached
                    cached_count += 1
                else:
                    print(json.dumps(dict(status="validating", motion_id=name, completed=len(clips), total=total)), flush=True)
                    if split:
                        if source is None:
                            source, robot = pickle.loads(old_source.read_bytes()), pickle.loads(old_robot.read_bytes())
                            audio, rate = sf.read(old_audio, dtype="float64", always_2d=True)
                            info = sf.info(old_audio)
                        if validator is None:
                            validator = SliceValidator(args.model, args.gmr_model)
                        human_bytes, robot_bytes = slice_payloads(source, robot, start, end, name, provenance, validator)
                        atomic_bytes(paths["source"], human_bytes)
                        atomic_bytes(paths["artifact"], robot_bytes)
                        first, last = round(start/FPS*rate), round(end/FPS*rate)
                        if end == entry["frames"]:
                            last = len(audio)  # retain a permissible one-sample rounding remainder
                        temporary = paths["audio"].with_suffix(".tmp.wav")
                        sf.write(temporary, audio[first:last], rate, subtype=info.subtype, format="WAV")
                        os.replace(temporary, paths["audio"])
                    else:
                        for old, new in zip((old_source, old_audio, old_robot), paths.values()):
                            atomic_copy(old, new)
                    passed, reason = preflight.validate(paths["source"], fps=FPS)
                    if not passed:
                        raise ValueError("Preflight: " + reason)
                    record = dict(source_motion_id=name, artifact=paths["artifact"].name,
                                  artifact_sha256=digest(paths["artifact"]), status="written", fps=FPS,
                                  frames=end-start, hashes={k: digest(p) for k, p in paths.items()},
                                  clip=clip, preflight_reason=reason)
                    atomic_json(cache_path, dict(key=key, value=record))
                clips.append(clip)
                records.append(record)
            except (OSError, ValueError, KeyError, TypeError, pickle.UnpicklingError, EOFError) as exc:
                failures.append(dict(motion_id=name, reason=f"{type(exc).__name__}: {exc}"))
            atomic_json(args.output_dir / "progress.json", dict(status="exporting", completed=len(clips),
                        total=total, failures=failures, cached=cached_count))
    manifest = dict(format_version=1, export_complete=not failures, pilot=args.limit is not None,
                    expected_count=total, clips=clips, failures=failures, generation=generation)
    atomic_json(source_root / "segments_manifest.json", manifest)
    manifest_hash = digest(source_root / "segments_manifest.json")
    for record in records:
        record["segmentation_manifest_sha256"] = manifest_hash
    atomic_json(gmr_root / "manifest.json", dict(pipeline_version=5, motion_version="gmr_v2", motions=records))
    for split in ("train", "val", "test"):
        content = "".join(c["motion_id"] + "\n" for c in clips if c["split"] == split).encode()
        atomic_bytes(source_root / (split + ".txt"), content)
        atomic_bytes(gmr_root / (split + ".txt"), content)
    if failures:
        raise ValueError(f"{len(failures)} export failures; inspect progress.json; catalogue remains unchanged")
    return source_root, gmr_root, manifest


def validate_membership(candidate, baseline, manifest, pilot=False):
    expected_aist = {c["motion_id"] for c in manifest["clips"]}
    actual_aist = {k for k, v in candidate["motions"].items() if v["dataset_id"] == "aistpp"}
    if expected_aist != actual_aist:
        raise ValueError("Candidate does not contain exactly the exported AIST++ clips")
    if not pilot:
        expected_fine = {k for k, v in baseline["motions"].items() if v["dataset_id"] == "finedance"}
        actual_fine = {k for k, v in candidate["motions"].items() if v["dataset_id"] == "finedance"}
        if expected_fine != actual_fine:
            raise ValueError("FineDance membership changed; candidate will not be activated")
    if any(v["duration_seconds"] > 16 for v in candidate["motions"].values()):
        raise ValueError("Candidate still contains motions over 16 seconds")


def publish(candidate_path, target, expected_target_hash):
    """Caller must hold the target builder lock. Only catalog.json activates a build."""
    from music_motion_catalog import MusicCatalog
    MusicCatalog.load(candidate_path)
    if digest(target) != expected_target_hash:
        raise ValueError("Active catalogue changed during this run; refusing to overwrite it")
    metadata = rebase_metadata(read_json(candidate_path), candidate_path.parent, target.parent)
    arrays = (candidate_path.parent / read_json(candidate_path)["arrays_file"]).resolve()
    local_arrays = target.parent / arrays.name
    atomic_copy(arrays, local_arrays)
    metadata["arrays_file"] = local_arrays.name
    backup = target.with_name("catalog.backup." + uuid.uuid4().hex + ".json")
    atomic_copy(target, backup)
    # Constructor checks the rebased paths and arrays before the activation pointer changes.
    pending = target.with_name("catalog.pending.json")
    atomic_json(pending, metadata)
    MusicCatalog.load(pending)
    os.replace(pending, target)
    return backup


def build_catalog(args, baseline, roots, source, gmr, manifest):
    atomic_json(args.output_dir / "progress.json", dict(status="building_catalog", pilot=args.limit is not None))
    candidate_dir = args.output_dir / "candidate"
    options = ["--aistpp-root", str(source), "--aistpp-gmr-root", str(gmr),
               "--finedance-root", str(roots["finedance"]["source_root"]),
               "--finedance-gmr-root", str(roots["finedance"]["gmr_root"]),
               "--output-dir", str(candidate_dir), "--model", str(args.model)]
    if args.limit is not None:
        # Keep every exported AIST child and select the same small number of FineDance clips.
        options.extend(["--limit-per-dataset", str(len(manifest["clips"]))])
    for key in ("embedding_model", "tag_model"):
        model = baseline.get("extractor", {}).get(key)
        if model and model.get("path"):
            options.extend(["--" + key.replace("_", "-"), str((args.output_dir / model["path"]).resolve())])
    candidate = catalog_builder.build(catalog_builder.parse_args(options))
    validate_membership(candidate, baseline, manifest, pilot=args.limit is not None)
    candidate_path = candidate_dir / "catalog.json"
    child = next((c["motion_id"] for c in manifest["clips"] if c["motion_id"] != c["parent_motion_id"]),
                 manifest["clips"][0]["motion_id"])
    atomic_json(args.output_dir / "progress.json", dict(status="validating_playback", candidate=str(candidate_path)))
    command = [sys.executable, str(BASE / "src/test/validate_combined_catalog.py"),
               "--catalog", str(candidate_path), "--output-dir", str(args.output_dir / "playback" / candidate["build_id"]),
               "--aistpp-motion-id", child, "--short-startup-check"]
    subprocess.run(command, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return candidate_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Read and print proposed cuts; write nothing")
    parser.add_argument("--limit", type=int, help="Pilot: select this many long parents; never activate")
    parser.add_argument("--build-catalog", action="store_true", help="Build and test; activate only a complete non-pilot run")
    parser.add_argument("--model", type=Path, default=BASE / "assets/open_humanoid_dancer.xml")
    parser.add_argument("--gmr-model", type=Path, default=BASE / ".deps/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    for key in ("catalog", "output_dir", "model", "gmr_model"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def run(args):
    origin, metadata, roots, parents = inventory(args)
    total = sum(len(p["children"]) for p in parents)
    summary = dict(parents=len(parents), split_parents=sum(len(p["children"]) > 1 for p in parents),
                   output_aistpp_motions=total, pilot=args.limit is not None,
                   output_dir=str(args.output_dir))
    if args.dry_run:
        for parent in parents:
            if len(parent["children"]) > 1:
                print(json.dumps(dict(parent=parent["entry"]["motion_id"], clips=parent["children"])))
        print(json.dumps(summary))
        return summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with batch_lock(args.output_dir / ".segment.lock"):
        snapshot = args.output_dir / "input_catalog.json"
        if not snapshot.exists():
            baseline = rebase_metadata(metadata, origin.parent, args.output_dir)
            baseline["segmentation_input_catalog"] = str(args.catalog)
            atomic_json(snapshot, baseline)
        baseline = read_json(snapshot)
        target_hash = digest(args.catalog)
        identity = validation_identity(args)
        source, gmr, manifest = export(args, roots, parents, identity)
        summary.update(source_root=str(source), gmr_root=str(gmr), status="export_complete")
        if args.build_catalog:
            candidate = build_catalog(args, baseline, roots, source, gmr, manifest)
            summary.update(candidate=str(candidate), status="pilot_validated" if args.limit else "validated")
            if args.limit is None:
                with batch_lock(args.catalog.parent / ".build.lock"):
                    backup = publish(candidate, args.catalog, target_hash)
                summary.update(status="activated", catalog=str(args.catalog), backup=str(backup))
        atomic_json(args.output_dir / "progress.json", summary)
        print(json.dumps(summary), flush=True)
        return summary


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(arguments)
    except Exception as exc:
        if not arguments.dry_run and arguments.output_dir.is_dir():
            atomic_json(arguments.output_dir / "last_failed_run.json",
                        dict(error=f"{type(exc).__name__}: {exc}"))
        print(f"Segmentation stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
