"""Build a resumable, validated AIST++/FineDance retrieval library (schema 3)."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import importlib.metadata
import json
import pickle
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from gmr_batch_lock import batch_lock
from music_motion_catalog import (
    AudioFeatureExtractor, MotionPreflightValidator, MotionProfile, MusicCatalog,
    COMBINED_CATALOG_SCHEMA_VERSION, detect_aistpp_file, iter_audio_windows,
    load_audio_mono, parse_motion_name, _keypoint_interval_cv,
)

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "realtime/humanoid_robot"
DATA = BASE / "data"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path, output):
    return os.path.relpath(Path(path).resolve(), Path(output).resolve()).replace("\\", "/")


def split_membership(root):
    result = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.txt"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    motion_id = Path(line.strip()).stem
                    if motion_id in result and result[motion_id] != split:
                        raise ValueError(f"Conflicting split membership: {motion_id}")
                    result[motion_id] = split
    return result


def dataset_entries(dataset_id, source_root, gmr_root, split="all", limit=None):
    """Return source entries, including missing GMR outputs for explicit exclusion."""
    successes = read_json(gmr_root / "manifest.json")["motions"]
    successes = {item["source_motion_id"]: item for item in successes}
    entries = []
    if dataset_id == "finedance":
        manifest = read_json(source_root / "manifest.json")
        if manifest.get("export_complete") is not True:
            raise ValueError("FineDance segmentation export is incomplete")
        segmentation_hash = digest(source_root / "manifest.json")
        for clip in manifest["clips"]:
            motion_id = clip["clip_id"]
            entries.append(dict(
                motion_id=motion_id, music_id=motion_id, recording_id=clip["source_id"],
                genre="finedance:" + clip["label"]["style2"], situation="finedance",
                split=clip["split"], motion_path=clip["motion_path"], audio_path=clip["audio_path"],
                frames=clip["frames"], duration_seconds=clip["duration_seconds"],
                song_title=clip["label"].get("name", ""), labels=clip["label"],
                start_seconds=clip["start_seconds"], end_seconds=clip["end_seconds"],
                segmentation_hash=segmentation_hash,
            ))
    elif (source_root / "segments_manifest.json").exists():
        manifest = read_json(source_root / "segments_manifest.json")
        if manifest.get("export_complete") is not True or manifest.get("format_version") != 1:
            raise ValueError("AIST++ segmentation export is incomplete or unsupported")
        segmentation_hash = digest(source_root / "segments_manifest.json")
        for clip in manifest["clips"]:
            entries.append(dict(clip, segmentation_hash=segmentation_hash))
    else:
        splits = split_membership(source_root)
        with (source_root / "audio/manifest.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                motion_id = row["motion_name"]
                parsed = parse_motion_name(motion_id)
                entries.append(dict(
                    motion_id=motion_id, music_id=parsed["music"], recording_id=parsed["music"],
                    genre=parsed["genre"], situation=parsed["situation"],
                    split=splits.get(motion_id, "unknown"), motion_path=f"motions/{motion_id}.pkl",
                    audio_path="audio/" + row["audio_path"], frames=int(row["motion_frames"]),
                    duration_seconds=float(row["duration_seconds"]), labels=parsed,
                    song_title=parsed["music"], start_seconds=0.0,
                    end_seconds=float(row["duration_seconds"]),
                ))
    entries.sort(key=lambda entry: entry["motion_id"])
    if split != "all":
        entries = [entry for entry in entries if entry["split"] == split]
    if limit is not None:
        # Round-robin genres makes a small real pilot cover multiple styles.
        groups = {}
        for entry in entries:
            groups.setdefault(entry["genre"], []).append(entry)
        selected = []
        while len(selected) < limit and any(groups.values()):
            for group in groups.values():
                if group and len(selected) < limit:
                    selected.append(group.pop(0))
        entries = selected
    for entry in entries:
        entry["dataset_id"] = dataset_id
        entry["gmr_record"] = successes.get(entry["motion_id"])
    return entries


def validate_inputs(entry, source_root, gmr_root):
    record = entry["gmr_record"]
    if not record or record.get("status") not in ("ok", "success", "cached", "generated", "written"):
        raise ValueError("No successful GMR conversion in manifest")
    motion = source_root / entry["motion_path"]
    audio = source_root / entry["audio_path"]
    artifact = gmr_root / record["artifact"]
    if artifact.name != entry["motion_id"] + ".pkl":
        raise ValueError("GMR artifact name does not match clip ID")
    hashes = dict(source=digest(motion), audio=digest(audio), artifact=digest(artifact))
    if hashes["artifact"] != record["artifact_sha256"]:
        raise ValueError("GMR artifact checksum mismatch")
    with artifact.open("rb") as handle:
        saved = pickle.load(handle)
    if not isinstance(saved, dict):
        raise ValueError("GMR artifact must contain a dictionary")
    if saved.get("source_sha256") != hashes["source"]:
        raise ValueError("Stale GMR source checksum")
    if saved.get("source_motion_id") != entry["motion_id"] or saved.get("pipeline_version") != 5:
        raise ValueError("Expected matching GMR v2 artifact")
    with motion.open("rb") as handle:
        source = pickle.load(handle)
    if not isinstance(source, dict):
        raise ValueError("Source motion must contain a dictionary")
    frames = len(source["smpl_poses"])
    if frames != entry["frames"] or len(saved["dof_pos"]) != frames:
        raise ValueError("Motion frame count mismatch")
    if not np.isclose(saved["fps"], 60.0) or not np.isclose(entry["duration_seconds"], frames / 60, atol=1e-6):
        raise ValueError("Motion timing mismatch")
    info = sf.info(audio)
    tolerance = 1 / info.samplerate + 1e-6 if entry["dataset_id"] == "finedance" else 0.1
    if abs(info.duration - frames / 60) > tolerance:
        raise ValueError("Paired audio duration mismatch")
    if entry["dataset_id"] == "finedance":
        if record["v2_build"]["segmentation_manifest_sha256"] != entry["segmentation_hash"]:
            raise ValueError("Stale FineDance segmentation provenance")
        clip = record["clip"]
        for key in ("motion_path", "audio_path", "frames", "split", "start_seconds", "end_seconds"):
            if clip[key] != entry[key]:
                raise ValueError(f"FineDance provenance mismatch: {key}")
        if clip["label"] != entry["labels"] or clip["source_id"] != entry["recording_id"]:
            raise ValueError("FineDance label/recording mismatch")
    elif "segmentation_hash" in entry:
        if record.get("segmentation_manifest_sha256") != entry["segmentation_hash"]:
            raise ValueError("Stale AIST++ segmentation provenance")
        if record.get("clip") != {k: v for k, v in entry.items()
                                   if k not in ("dataset_id", "gmr_record", "segmentation_hash")}:
            raise ValueError("AIST++ segment manifest mismatch")
        if entry.get("parent_motion_id") != entry["motion_id"]:
            if saved.get("segmentation") != entry.get("provenance"):
                raise ValueError("AIST++ child provenance mismatch")
        if hashes != record.get("hashes"):
            raise ValueError("Segment source/audio/artifact checksum mismatch")
    return hashes


def profile_motion(entry, source_root, validator):
    path = source_root / entry["motion_path"]
    passed, reason = validator.validate(path, fps=60.0)
    if not passed:
        raise ValueError("Preflight: " + reason)
    keypoints = detect_aistpp_file(path, fps=60.0)
    velocity = np.asarray(keypoints.smoothed_velocity)
    duration = entry["duration_seconds"]
    return MotionProfile(
        motion_id=entry["motion_id"], music_id=entry["music_id"], genre=entry["genre"],
        situation=entry["situation"], motion_path=entry["motion_path"], duration_seconds=duration,
        keypoint_phases=tuple(keypoints.phases), keypoint_scores=tuple(keypoints.scores),
        keypoint_density_hz=len(keypoints.phases) / duration,
        weighted_keypoint_density=sum(keypoints.scores) / duration,
        keypoint_interval_cv=_keypoint_interval_cv(keypoints.phases),
        velocity_median=float(np.median(velocity)), velocity_p90=float(np.percentile(velocity, 90)),
        original_bpm=0.0, preflight_passed=True, preflight_reason="ok",
        dataset_id=entry["dataset_id"], recording_id=entry["recording_id"], split=entry["split"],
    ).to_dict()


def describe_audio(path, extractor, window_seconds, hop_seconds):
    audio = load_audio_mono(path, extractor.sample_rate)
    result = []
    for start, end, window in iter_audio_windows(audio, extractor.sample_rate, window_seconds, hop_seconds):
        descriptor = extractor.describe(window)
        result.append(dict(start_seconds=start, end_seconds=end, **descriptor.scalar_metadata(),
                           embedding=descriptor.embedding.tolist(), rhythm_timbre=descriptor.rhythm_timbre.tolist(),
                           tags=descriptor.tag_probabilities.tolist()))
    return result


def read_cache(path, key):
    try:
        cached = read_json(path)
        return cached["value"] if cached["key"] == key else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def assemble(accepted, datasets, extractor_metadata, output, failures, selected_count):
    tracks, profiles, segments = {}, {}, []
    embeddings, rhythm, tags = [], [], []
    variant_keys = set()
    for entry, profile, windows, audio_hash in accepted:
        music_id = entry["music_id"]
        track = tracks.setdefault(music_id, dict(
            music_id=music_id, genre=entry["genre"], dataset_id=entry["dataset_id"],
            recording_id=entry["recording_id"], song_title=entry["song_title"],
            labels=entry["labels"], variants=[], motion_ids=[],
        ))
        if track["genre"] != entry["genre"] or track["dataset_id"] != entry["dataset_id"]:
            raise ValueError(f"Conflicting retrieval identity: {music_id}")
        track["motion_ids"].append(entry["motion_id"])
        profiles[entry["motion_id"]] = profile
        variant_id = f"{music_id}:{entry['situation']}:{audio_hash[:16]}"
        if variant_id in variant_keys:
            continue
        variant_keys.add(variant_id)
        association = dict(dataset_id=entry["dataset_id"], recording_id=entry["recording_id"],
                           source_audio=entry["audio_path"], split=entry["split"],
                           source_start_seconds=entry["start_seconds"], source_end_seconds=entry["end_seconds"])
        track["variants"].append(dict(variant_id=variant_id, situation=entry["situation"], **association))
        for window in windows:
            window = dict(window)
            embeddings.append(window.pop("embedding"))
            rhythm.append(window.pop("rhythm_timbre"))
            tags.append(window.pop("tags"))
            segments.append(dict(music_id=music_id, genre=entry["genre"], variant_id=variant_id,
                                 variant_index=len(variant_keys) - 1, **association, **window))
    if not profiles:
        raise ValueError("No motion passed validation; previous catalog has not been replaced")
    for music_id, track in tracks.items():
        bpms = [s["bpm"] for s in segments if s["music_id"] == music_id and s["bpm"] > 0]
        bpm = float(np.median(bpms)) if bpms else 0.0
        for motion_id in track["motion_ids"]:
            profiles[motion_id]["original_bpm"] = bpm
    velocities = np.asarray([p["velocity_p90"] for p in profiles.values()])
    densities = np.asarray([p["weighted_keypoint_density"] for p in profiles.values()])
    regularities = np.asarray([p["keypoint_interval_cv"] for p in profiles.values()])
    velocity_cuts, density_cuts = np.quantile(velocities, (1/3, 2/3)), np.quantile(densities, (1/3, 2/3))
    regularity_cut = float(np.median(regularities))
    for p in profiles.values():
        p["motion_cluster_id"] = (f"{p['genre']}:v{np.searchsorted(velocity_cuts, p['velocity_p90'])}"
            f":k{np.searchsorted(density_cuts, p['weighted_keypoint_density'])}"
            f":r{int(p['keypoint_interval_cv'] > regularity_cut)}")
    embeddings, rhythm, tags = (np.asarray(values, dtype=np.float32) for values in (embeddings, rhythm, tags))
    build_id = uuid.uuid4().hex
    arrays_name = f"catalog_features_{build_id}.npz"
    arrays = dict(embeddings=embeddings, rhythm_timbre=rhythm, tags=tags,
                  embedding_mean=embeddings.mean(axis=0), embedding_std=embeddings.std(axis=0),
                  rhythm_mean=rhythm.mean(axis=0), rhythm_std=rhythm.std(axis=0))
    with (output / arrays_name).open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    counts = {dataset_id: sum(p["dataset_id"] == dataset_id for p in profiles.values()) for dataset_id in datasets}
    metadata = dict(schema_version=COMBINED_CATALOG_SCHEMA_VERSION, build_id=build_id,
                    datasets=datasets, arrays_file=arrays_name, extractor=extractor_metadata,
                    counts=dict(selected=selected_count, motions_included=len(profiles), motions_excluded=len(failures),
                                music_ids=len(tracks), segments=len(segments), by_dataset=counts),
                    motion_stats=dict(velocity_median=float(np.median(velocities)),
                                      velocity_scale=float(max(np.ptp(np.quantile(velocities, (.25, .75))), np.std(velocities), 1e-6)),
                                      cluster_policy="genre_velocity_density_regularity_v1"),
                    tracks=tracks, motions=profiles, segments=segments, exclusions=failures)
    # Immutable arrays first; atomically replace the sole metadata pointer last.
    MusicCatalog(output / "catalog.json", metadata, arrays)
    atomic_json(output / "catalog.json", metadata)
    atomic_json(output / "excluded_motions.json", failures)
    return metadata


def build(args):
    output = args.output_dir.resolve()
    if output == (DATA / "music_catalog").resolve():
        raise ValueError("The original AIST++ catalog is preserved; choose a separate --output-dir")
    for dataset_id in args.datasets:
        for suffix in ("_root", "_gmr_root"):
            source = getattr(args, dataset_id + suffix).resolve()
            if output == source or output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError("Catalog output must not overlap dataset source or GMR directories")
    output.mkdir(parents=True, exist_ok=True)
    with batch_lock(output / ".build.lock"):
        return build_locked(args, output)


def build_locked(args, output):
    started = time.perf_counter()
    cache = output / ".cache"
    cache.mkdir(exist_ok=True)
    extractor = AudioFeatureExtractor(embedding_model=args.embedding_model, tag_model=args.tag_model,
                                      onnx_intra_op_threads=1)
    model_metadata = extractor.model_metadata()
    source_dir = Path(__file__).parent
    implementation_names = ("build_combined_music_catalog.py", "music_motion_catalog.py", "aistpp_velocity_keypoints.py",
                            "aistpp_smpl.py", "realtime_music_humanoid_dancer.py", "unitree_g1_dance_adapter.py",
                            "gmr_collision_projection.py", "gmr_state_trajectory.py", "robot_motion.py", "hermite_bounds.py")
    implementation = {name: digest(source_dir / name) for name in implementation_names}
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "librosa", "soundfile", "onnxruntime", "mujoco")}
    # Include local model assets as well as its top-level XML in validation cache identity.
    assets = {str(path.relative_to(args.model.parent)): digest(path)
              for path in args.model.parent.rglob("*")
              if path.is_file() and path.suffix.lower() in (".xml", ".stl", ".obj")}
    settings = dict(implementation=implementation, dependencies=versions, model=digest(args.model), assets=assets,
                    extractor=model_metadata, window=args.window_seconds, hop=args.hop_seconds)
    settings_key = fingerprint(settings)
    accepted, failures, datasets = [], [], {}
    selected_count, cached_count = 0, 0
    for dataset_id in args.datasets:
        source_root = getattr(args, dataset_id + "_root").resolve()
        gmr_root = getattr(args, dataset_id + "_gmr_root").resolve()
        datasets[dataset_id] = dict(source_root=relative(source_root, output), gmr_root=relative(gmr_root, output))
        entries = dataset_entries(dataset_id, source_root, gmr_root, args.split, args.limit_per_dataset)
        selected_count += len(entries)
        validator = MotionPreflightValidator(args.model, gmr_root)
        for index, entry in enumerate(entries, 1):
            motion_id = entry["motion_id"]
            try:
                hashes = validate_inputs(entry, source_root, gmr_root)
                key = fingerprint(dict(settings=settings_key, hashes=hashes, entry=entry))
                cached_path = cache / (motion_id + ".json")
                value = read_cache(cached_path, key)
                if value is None:
                    profile_cache = cache / (motion_id + ".preflight.json")
                    profile = read_cache(profile_cache, key)
                    if profile is None:
                        profile = profile_motion(entry, source_root, validator)
                        atomic_json(profile_cache, dict(key=key, value=profile))
                    audio_key = fingerprint(dict(settings=settings_key, audio=hashes["audio"]))
                    audio_cache = cache / ("audio_" + audio_key + ".json")
                    windows = read_cache(audio_cache, audio_key)
                    if windows is None:
                        windows = describe_audio(source_root / entry["audio_path"], extractor, args.window_seconds, args.hop_seconds)
                        atomic_json(audio_cache, dict(key=audio_key, value=windows))
                    value = dict(profile=profile, windows=windows)
                    atomic_json(cached_path, dict(key=key, value=value))
                else:
                    cached_count += 1
                accepted.append((entry, value["profile"], value["windows"], hashes["audio"]))
            except (OSError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as exc:
                failures.append(dict(dataset_id=dataset_id, motion_id=motion_id, reason=f"{type(exc).__name__}: {exc}"))
            if index == 1 or index % 10 == 0 or index == len(entries):
                progress = dict(status="running", dataset=dataset_id, processed=index, selected=len(entries),
                                accepted=len(accepted), excluded=len(failures), cached=cached_count,
                                elapsed_seconds=time.perf_counter() - started)
                atomic_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    for name in ("embedding_model", "tag_model"):
        if model_metadata.get(name):
            model_metadata[name]["path"] = relative(model_metadata[name]["path"], output)
    extractor_metadata = dict(sample_rate=extractor.sample_rate, window_seconds=args.window_seconds,
                              hop_seconds=args.hop_seconds, **model_metadata)
    missing_datasets = set(args.datasets) - {entry["dataset_id"] for entry, *_ in accepted}
    if missing_datasets:
        atomic_json(output / "last_failed_build.json", dict(missing_datasets=sorted(missing_datasets), exclusions=failures))
        raise ValueError(f"No validated motions for selected datasets: {sorted(missing_datasets)}; previous catalog retained")
    metadata = assemble(accepted, datasets, extractor_metadata, output, failures, selected_count)
    result = dict(status="complete", **metadata["counts"], cached=cached_count,
                  elapsed_seconds=time.perf_counter() - started)
    atomic_json(output / "progress.json", result)
    print(json.dumps(result), flush=True)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("aistpp", "finedance"), default=["aistpp", "finedance"])
    parser.add_argument("--aistpp-root", type=Path, default=DATA / "aistpp")
    parser.add_argument("--finedance-root", type=Path, default=DATA / "finedance_aistpp")
    parser.add_argument("--aistpp-gmr-root", type=Path, default=DATA / "aistpp_gmr_v2")
    parser.add_argument("--finedance-gmr-root", type=Path, default=DATA / "finedance_gmr_v2")
    parser.add_argument("--output-dir", type=Path, default=DATA / "music_catalog_combined")
    parser.add_argument("--model", type=Path, default=BASE / "assets/open_humanoid_dancer.xml")
    parser.add_argument("--embedding-model", type=Path, default=BASE / "models/discogs-effnet-bsdynamic-1.onnx")
    parser.add_argument("--tag-model", type=Path, default=BASE / "models/discogs-effnet-bsdynamic-1.onnx")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--window-seconds", type=float, default=6.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.limit_per_dataset is not None and args.limit_per_dataset < 1:
        parser.error("--limit-per-dataset must be positive")
    if args.window_seconds <= 0 or args.hop_seconds <= 0:
        parser.error("Window and hop must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Dataset names must be unique")
    return args


if __name__ == "__main__":
    build(parse_args())
