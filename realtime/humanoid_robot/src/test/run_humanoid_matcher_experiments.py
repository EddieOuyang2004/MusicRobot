"""Run and aggregate the objective Humanoid Matcher experiment protocol.

The default invocation performs repeatable offline catalog evaluation and
aggregates any supplied production traces/timing reports.  Expensive wall-clock
system runs are opt-in via ``--execute-suite``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import shlex
import subprocess
import sys
import time
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent
HUMANOID_DIR = SRC_DIR.parent
ROOT = HUMANOID_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from humanoid_matcher_experiment_metrics import (  # noqa: E402
    EXPERIMENT_SCHEMA_VERSION,
    aggregate_trace_csv,
    aggregate_pose_npz,
    binary_rejection_metrics,
    bootstrap_mean_ci,
    diversity,
    distribution_summary,
    literature_feature_metrics,
    holm_adjust,
    paired_permutation_test,
    ranking_metrics,
    response_latency_from_trace,
    selection_diversity,
)
from music_motion_catalog import (  # noqa: E402
    AudioDescriptor,
    AudioFeatureExtractor,
    MusicCatalog,
    MusicMotionMatcher,
    iter_audio_windows,
    load_audio_mono,
)


DEFAULT_CATALOG = HUMANOID_DIR / "data" / "music_catalog" / "catalog.json"
DEFAULT_CC0_ROOT = HUMANOID_DIR / "data" / "test_audio" / "cc0_matcher_set"
DEFAULT_NEGATIVE_ROOT = HUMANOID_DIR / "data" / "test_audio" / "matcher_negative_set"
DEFAULT_PROTOCOL = TEST_DIR / "humanoid_matcher_experiment_protocol.json"
DEFAULT_SCHEMA = TEST_DIR / "humanoid_matcher_experiment.schema.json"
DEFAULT_OUTPUT = TEST_DIR / "output" / "humanoid_matcher_experiment"
MATCHER_ENTRYPOINT = SRC_DIR / "realtime_music_humanoid_matcher.py"
STITCHED_MANIFEST = HUMANOID_DIR / "data" / "test_audio" / "aistpp_stitched_test.json"
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
TRACE_DURATION_TOLERANCE_SECONDS = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cc0-root", type=Path, default=DEFAULT_CC0_ROOT)
    parser.add_argument("--negative-audio-root", type=Path, default=DEFAULT_NEGATIVE_ROOT)
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--timing", type=Path, action="append", default=[])
    parser.add_argument("--pose", type=Path, action="append", default=[])
    parser.add_argument("--feature-bundle", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument(
        "--execute-suite",
        choices=("none", "smoke", "literature", "full", "ablation", "longrun"),
        default="none",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete, validated run artifacts and retry everything else.",
    )
    parser.add_argument("--max-runs", type=int)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum attempts for each incomplete run (default: 2).",
    )
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--no-audio-evaluation", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_replace(source: Path, destination: Path, attempts: int = 20) -> None:
    """Replace a file atomically, tolerating short Windows scanner locks."""

    for attempt in range(max(attempts, 1)):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.05)


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text if result.returncode == 0 and text else None


def _model_path(catalog: MusicCatalog, key: str) -> Path | None:
    value = catalog.metadata.get("extractor", {}).get(key)
    if not isinstance(value, dict) or not value.get("path"):
        return None
    path = Path(str(value["path"]))
    return path if path.is_absolute() else (catalog.catalog_path.parent / path).resolve()


def make_extractor(catalog: MusicCatalog) -> AudioFeatureExtractor:
    return AudioFeatureExtractor(
        sample_rate=int(catalog.metadata["extractor"]["sample_rate"]),
        embedding_model=_model_path(catalog, "embedding_model"),
        tag_model=_model_path(catalog, "tag_model"),
        onnx_intra_op_threads=1,
    )


def environment_manifest(
    catalog: MusicCatalog,
    protocol_path: Path,
    extractor: AudioFeatureExtractor | None,
) -> dict[str, Any]:
    providers: list[str] = []
    if extractor is not None and extractor.embedding_backend is not None:
        providers = list(extractor.embedding_backend.session.get_providers())
    models = {}
    for key in ("embedding_model", "tag_model"):
        path = _model_path(catalog, key)
        if path is not None:
            models[key] = {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": file_sha256(path) if path.is_file() else None,
            }
    git_status = command_output(["git", "status", "--short"])
    implementation_paths = {
        "catalog_matcher": SRC_DIR / "music_motion_catalog.py",
        "realtime_matcher": MATCHER_ENTRYPOINT,
        "realtime_dancer": SRC_DIR / "realtime_music_humanoid_dancer.py",
        "robot_motion": SRC_DIR / "robot_motion.py",
        "experiment_metrics": TEST_DIR / "humanoid_matcher_experiment_metrics.py",
        "experiment_runner": Path(__file__).resolve(),
    }
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(git_status),
        "git_status": git_status or "",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "windows_power_scheme": (
            command_output(["powercfg", "/getactivescheme"])
            if os.name == "nt"
            else None
        ),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "packages": {
            name: package_version(name)
            for name in ("numpy", "librosa", "onnxruntime", "mujoco", "scipy")
        },
        "onnxruntime_providers": providers,
        "catalog": {
            "path": str(catalog.catalog_path),
            "sha256": file_sha256(catalog.catalog_path),
            "arrays_sha256": file_sha256(
                catalog.catalog_path.parent / catalog.metadata["arrays_file"]
            ),
        },
        "models": models,
        "implementation_sha256": {
            name: file_sha256(path) for name, path in implementation_paths.items()
        },
        "protocol": {"path": str(protocol_path), "sha256": file_sha256(protocol_path)},
        "output_schema": {
            "path": str(DEFAULT_SCHEMA),
            "sha256": file_sha256(DEFAULT_SCHEMA),
        },
    }


def descriptor_from_segment(catalog: MusicCatalog, index: int) -> AudioDescriptor:
    metadata = catalog.segment_metadata[index]
    return AudioDescriptor(
        embedding=catalog.embeddings[index],
        rhythm_timbre=catalog.rhythm_timbre[index],
        tag_probabilities=catalog.tags[index],
        bpm=float(metadata["bpm"]),
        beat_strength=float(metadata["beat_strength"]),
        onset_density=float(metadata["onset_density"]),
        offbeat_ratio=float(metadata["offbeat_ratio"]),
        tempo_stability=float(metadata["tempo_stability"]),
        spectral_flux=float(metadata["spectral_flux"]),
        percussive_ratio=float(metadata["percussive_ratio"]),
    )


def catalog_without_music(catalog: MusicCatalog, music_id: str) -> MusicCatalog:
    keep = np.asarray(
        [str(item["music_id"]) != music_id for item in catalog.segment_metadata],
        dtype=bool,
    )
    metadata = copy.deepcopy(catalog.metadata)
    metadata["segments"] = [
        item for item in catalog.segment_metadata if str(item["music_id"]) != music_id
    ]
    metadata["tracks"].pop(music_id, None)
    metadata["motions"] = {
        motion_key: values
        for motion_key, values in metadata["motions"].items()
        if str(values["music_id"]) != music_id
    }
    arrays = {
        "embeddings": catalog.embeddings[keep],
        "rhythm_timbre": catalog.rhythm_timbre[keep],
        "tags": catalog.tags[keep],
        "embedding_mean": catalog.embedding_mean,
        "embedding_std": catalog.embedding_std,
        "rhythm_mean": catalog.rhythm_mean,
        "rhythm_std": catalog.rhythm_std,
    }
    return MusicCatalog(catalog.catalog_path, metadata, arrays)


def strict_leave_one_music_evaluation(
    catalog: MusicCatalog,
    *,
    bootstrap_iterations: int = 2_000,
) -> dict[str, Any]:
    fold_matchers = {
        music_id: MusicMotionMatcher(catalog_without_music(catalog, music_id))
        for music_id in sorted(catalog.tracks)
    }
    expected_genres: list[str] = []
    ranked_genres: list[list[str]] = []
    query_rows: list[dict[str, Any]] = []
    bpm_errors: list[float] = []
    activity_errors: list[float] = []
    margins: list[float] = []
    latencies: list[float] = []
    genre_rank_scores_by_music: dict[str, Counter[str]] = defaultdict(Counter)

    for index, metadata in enumerate(catalog.segment_metadata):
        music_id = str(metadata["music_id"])
        expected_genre = str(metadata["genre"])
        matcher = fold_matchers[music_id]
        matcher._weak_music_streak = 0
        started = time.perf_counter_ns()
        result = matcher.match(
            descriptor_from_segment(catalog, index), top_k_tracks=60, top_k_motions=10
        )
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
        ranking = [genre.genre for genre in result.genres]
        expected_genres.append(expected_genre)
        ranked_genres.append(ranking)
        for rank, genre in enumerate(dict.fromkeys(ranking), start=1):
            genre_rank_scores_by_music[music_id][genre] += 1.0 / rank
        top_motion = result.motions[0] if result.motions else None
        if top_motion is not None:
            profile = matcher.catalog.motions[top_motion.motion_id]
            query_bpm = max(result.query_bpm, 1e-8)
            effective_bpm = profile.original_bpm * top_motion.speed_ratio
            bpm_errors.append(abs(1200.0 * math.log2(effective_bpm / query_bpm)))
            query_activity = float(metadata.get("activity", 0.0))
            activity_errors.append(abs(query_activity - profile.velocity_median))
        margins.append(result.track_margin)
        query_rows.append(
            {
                "query_index": index,
                "music_id": music_id,
                "expected_genre": expected_genre,
                "ranked_genres": ranking,
                "accepted": result.accepted,
                "rejection_reason": result.rejection_reason,
                "top_motion_id": top_motion.motion_id if top_motion else None,
                "track_margin": result.track_margin,
                "motion_margin": result.motion_margin,
            }
        )
    music_ids = sorted(genre_rank_scores_by_music)
    music_expected = [str(catalog.tracks[music_id]["genre"]) for music_id in music_ids]
    music_ranked = [
        [
            genre
            for genre, _score in genre_rank_scores_by_music[music_id].most_common()
        ]
        for music_id in music_ids
    ]
    music_ranking = ranking_metrics(music_expected, music_ranked)
    music_level_ci = {}
    for k in (1, 3, 5):
        observations = [
            float(expected in ranking[:k])
            for expected, ranking in zip(music_expected, music_ranked, strict=True)
        ]
        music_level_ci[f"recall_at_{k}"] = bootstrap_mean_ci(
            observations,
            iterations=bootstrap_iterations,
            seed=k,
        )
    return {
        "definition": (
            "All segments, track metadata, and motions sharing the query music_id "
            "are removed before ranking. Queries are aggregated by music in statistics."
        ),
        "ranking": music_ranking,
        "music_level_bootstrap_95_ci": music_level_ci,
        "window_level_diagnostic": ranking_metrics(expected_genres, ranked_genres),
        "latency_ms": distribution_summary(latencies),
        "bpm_error_cents": distribution_summary(bpm_errors),
        "activity_absolute_error": distribution_summary(activity_errors),
        "track_margin": distribution_summary(margins),
        "queries": query_rows,
    }


def source_motion_sanity(catalog: MusicCatalog) -> dict[str, Any]:
    matcher = MusicMotionMatcher(catalog)
    hits = Counter({1: 0, 3: 0, 10: 0})
    available = 0
    for index, metadata in enumerate(catalog.segment_metadata):
        matcher._weak_music_streak = 0
        result = matcher.match(
            descriptor_from_segment(catalog, index), top_k_tracks=5, top_k_motions=10
        )
        source = str(metadata["music_id"])
        ranked = [item.music_id for item in result.motions]
        available += int(bool(ranked))
        for k in hits:
            hits[k] += int(source in ranked[:k])
    count = max(len(catalog.segment_metadata), 1)
    return {
        "note": "Same-catalog source-track sanity check; not an unseen-music result.",
        "queries": len(catalog.segment_metadata),
        "queries_with_motion_candidates_rate": available / count,
        **{f"motion_recall_at_{k}": hits[k] / count for k in hits},
    }


def _audio_files(root: Path | None) -> list[Path]:
    if root is None or not root.is_dir():
        return []
    return [path for path in sorted(root.iterdir()) if path.suffix.lower() in AUDIO_SUFFIXES]


def _label_for_audio(path: Path, labels: Mapping[str, Any]) -> Mapping[str, Any]:
    stem = path.stem.lower().replace(" ", "_")
    for key, value in labels.items():
        if str(key).lower() in stem:
            return value
    return {"expected_genres": [], "reject": True}


def audio_set_evaluation(
    catalog: MusicCatalog,
    extractor: AudioFeatureExtractor,
    protocol: Mapping[str, Any],
    cc0_root: Path,
    negative_root: Path | None,
) -> dict[str, Any]:
    matcher = MusicMotionMatcher(catalog)
    labels = dict(protocol.get("cc0_labels", {}))
    files = [(path, _label_for_audio(path, labels)) for path in _audio_files(cc0_root)]
    files.extend(
        (path, {"expected_genres": [], "reject": True})
        for path in _audio_files(negative_root)
    )
    expected_reject: list[bool] = []
    predicted_reject: list[bool] = []
    scores: list[float] = []
    file_reports: dict[str, Any] = {}
    all_motions: list[str] = []
    all_scores: list[float] = []
    cluster_by_motion = {
        motion_id: profile.motion_cluster_id
        for motion_id, profile in catalog.motions.items()
    }

    for audio_path, label in files:
        matcher._weak_music_streak = 0
        audio = load_audio_mono(audio_path, extractor.sample_rate)
        windows = list(
            iter_audio_windows(
                audio[: int(30.0 * extractor.sample_rate)],
                extractor.sample_rate,
                float(protocol["match_window_seconds"]),
                float(protocol["match_interval_seconds"]),
            )
        )
        results = [matcher.match(extractor.describe(window)) for _, _, window in windows]
        rejected = [not result.accepted or not result.motions for result in results]
        genres = [result.genres[0].genre if result.genres else "" for result in results]
        motions = [result.motions[0].motion_id for result in results if result.motions]
        motion_scores = [result.motions[0].final_score for result in results if result.motions]
        expected = bool(label.get("reject", False))
        expected_reject.extend([expected] * len(results))
        predicted_reject.extend(rejected)
        scores.extend(result.negative_style_score for result in results)
        all_motions.extend(motions)
        all_scores.extend(motion_scores)
        expected_genres = set(str(value) for value in label.get("expected_genres", ()))
        genre_hit_rate = (
            sum(genre in expected_genres for genre in genres) / max(len(genres), 1)
            if expected_genres
            else None
        )
        file_reports[audio_path.name] = {
            "label": dict(label),
            "windows": len(results),
            "rejected": sum(rejected),
            "rejection_rate": sum(rejected) / max(len(results), 1),
            "genre_hit_rate": genre_hit_rate,
            "genre_counts": dict(Counter(genres)),
            "motion_counts": dict(Counter(motions)),
            "negative_style_score": distribution_summary(
                result.negative_style_score for result in results
            ),
        }
    return {
        "files": file_reports,
        "rejection": binary_rejection_metrics(
            expected_reject, predicted_reject, scores
        ),
        "selection": selection_diversity(all_motions, cluster_by_motion, all_scores),
    }


def aggregate_timing_reports(paths: Iterable[Path]) -> dict[str, Any]:
    reports = []
    failures = []
    for path in paths:
        try:
            reports.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError) as error:
            failures.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
    numeric: dict[str, list[float]] = defaultdict(list)
    per_run = []
    for path, report in reports:
        per_run.append({"path": str(path), **report})
        for key, value in report.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                numeric[key].append(float(value))
    return {
        "runs": per_run,
        "aggregate": {key: distribution_summary(values) for key, values in sorted(numeric.items())},
        "failures": failures,
    }


def load_feature_bundle(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as archive:
        required = {"real_kinetic", "output_kinetic", "real_geometric", "output_geometric"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Feature bundle is missing: {missing}")
        identity = (
            str(archive["extractor_identity"].item())
            if "extractor_identity" in archive.files
            else ""
        )
        result = literature_feature_metrics(
            archive["real_kinetic"],
            archive["output_kinetic"],
            archive["real_geometric"],
            archive["output_geometric"],
            extractor_identity=identity,
        )
        if {"g1_library_features", "g1_output_features"} <= set(archive.files):
            library_distance = diversity(archive["g1_library_features"])
            output_distance = diversity(archive["g1_output_features"])
            result["g1_output_domain"] = {
                "library_pairwise_distance": library_distance,
                "output_pairwise_distance": output_distance,
                "output_to_library_distance_ratio": (
                    output_distance / library_distance if library_distance > 0.0 else None
                ),
            }
        return result


def track_audio_path(catalog: MusicCatalog, music_id: str) -> Path:
    track = catalog.tracks[music_id]
    variant = track["variants"][0]
    root = (catalog.catalog_path.parent / str(catalog.metadata["aistpp_root"])).resolve()
    return (root / str(variant["source_audio"])).resolve()


def literature_music_ids(catalog: MusicCatalog, clips_per_genre: int) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for music_id, track in sorted(catalog.tracks.items()):
        grouped[str(track["genre"])].append(str(music_id))
    return [
        music_id
        for genre in sorted(grouped)
        for music_id in grouped[genre][: max(int(clips_per_genre), 0)]
    ]


def literature_items(
    catalog: MusicCatalog,
    clips_per_genre: int,
) -> list[tuple[str, Path, str]]:
    """Freeze each literature music ID to one deterministic ground-truth motion."""

    items = []
    for music_id in literature_music_ids(catalog, clips_per_genre):
        motions = sorted(
            profile.motion_id
            for profile in catalog.motions.values()
            if profile.music_id == music_id and profile.preflight_passed
        )
        if not motions:
            raise ValueError(f"No preflight-passed motion for literature item {music_id}")
        items.append((music_id, track_audio_path(catalog, music_id), motions[0]))
    return items


def run_matrix(
    suite: str,
    catalog: MusicCatalog,
    protocol: Mapping[str, Any],
    cc0_root: Path,
) -> list[dict[str, Any]]:
    seeds = list(protocol["seeds"])
    full_condition = {"name": "full", "arguments": []}
    ground_truth_by_source: dict[str, str] = {}
    if suite == "smoke":
        music_ids = literature_music_ids(catalog, 1)[:1]
        sources = [(music_ids[0], track_audio_path(catalog, music_ids[0]))]
        conditions = [full_condition]
        seeds = seeds[:1]
    elif suite == "literature":
        fixed_items = literature_items(
            catalog,
            protocol["literature"]["clips_per_genre"],
        )
        sources = [(music_id, audio) for music_id, audio, _motion_id in fixed_items]
        ground_truth_by_source = {
            music_id: motion_id for music_id, _audio, motion_id in fixed_items
        }
        conditions = [full_condition]
    elif suite == "full":
        sources = [
            (music_id, track_audio_path(catalog, music_id))
            for music_id in sorted(catalog.tracks)
        ] + [(path.stem, path) for path in _audio_files(cc0_root)]
        conditions = [full_condition]
    elif suite in {"ablation", "longrun"}:
        stitched = HUMANOID_DIR / "data" / "test_audio" / "aistpp_stitched_test.wav"
        sources = [(stitched.stem, stitched)]
        if suite == "ablation":
            sources += [(path.stem, path) for path in _audio_files(cc0_root)]
            change_root = HUMANOID_DIR / "data" / "test_audio" / "matcher_change_streams"
            sources += [
                (path.stem, path)
                for path in _audio_files(change_root)
                if path.stem != "genre_jump_aistpp"
            ]
        conditions = [
            {"name": name, "arguments": list(arguments)}
            for name, arguments in protocol["ablations"].items()
        ]
    else:
        return []
    return [
        {
            "condition": condition["name"],
            "condition_arguments": condition["arguments"],
            "seed": seed,
            "source_id": source_id,
            "audio": audio,
            "ground_truth_motion_id": ground_truth_by_source.get(source_id),
        }
        for condition in conditions
        for seed in seeds
        for source_id, audio in sources
    ]


def write_repeated_audio(
    source: Path,
    destination: Path,
    *,
    seconds: float,
    sample_rate: int,
) -> Path:
    """Materialize a deterministic long-running WAV without changing production playback."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    required = max(int(round(float(seconds) * sample_rate)), 1)
    samples = load_audio_mono(source, sample_rate)
    if not samples.size:
        raise ValueError(f"Cannot repeat empty audio: {source}")
    repeated = np.resize(np.asarray(samples, dtype=np.float32), required)
    pcm = np.round(np.clip(repeated, -1.0, 1.0) * 32767.0).astype("<i2")
    temporary = destination.with_suffix(".tmp.wav")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    atomic_replace(temporary, destination)
    return destination


def prepare_short_audio_inputs(
    matrix: list[dict[str, Any]],
    output_dir: Path,
    *,
    required_seconds: float,
    sample_rate: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Repeat only source files that cannot cover the registered run duration."""

    duration_cache: dict[Path, float] = {}
    repeated_cache: dict[Path, Path] = {}
    generated_root = output_dir / "generated_streams"
    for item in matrix:
        source = Path(item["audio"]).resolve()
        if source not in duration_cache:
            samples = load_audio_mono(source, sample_rate)
            if not samples.size:
                raise ValueError(f"Cannot evaluate empty audio: {source}")
            duration_cache[source] = len(samples) / float(sample_rate)
        source_duration = duration_cache[source]
        item["required_duration_seconds"] = float(required_seconds)
        item["source_audio_duration_seconds"] = float(source_duration)
        item["original_audio"] = source
        if source_duration + TRACE_DURATION_TOLERANCE_SECONDS >= required_seconds:
            item["audio_was_repeated"] = False
            continue
        if source not in repeated_cache:
            digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:10]
            destination = generated_root / (
                f"{source.stem}_{digest}_{required_seconds:g}s_repeated.wav"
            )
            if not dry_run:
                write_repeated_audio(
                    source,
                    destination,
                    seconds=required_seconds + 1.0,
                    sample_rate=sample_rate,
                )
            repeated_cache[source] = destination
        item["audio"] = repeated_cache[source]
        item["audio_was_repeated"] = True
    return matrix


def _finite_csv_float(row: Mapping[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def validate_run_artifacts(
    trace: Path,
    timing: Path,
    pose: Path,
    log: Path,
    *,
    minimum_duration_seconds: float | None = None,
) -> tuple[bool, list[str]]:
    """Return whether a run has every non-empty, parseable canonical artifact."""

    errors = []
    try:
        with trace.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            first = next(reader, None)
            if first is None or "audio_time_seconds" not in (reader.fieldnames or []):
                errors.append("trace has no data rows or required header")
            elif minimum_duration_seconds is not None:
                trace_times = [_finite_csv_float(first, "audio_time_seconds")]
                trace_times.extend(
                    _finite_csv_float(row, "audio_time_seconds") for row in reader
                )
                finite_times = [value for value in trace_times if value is not None]
                maximum_time = max(finite_times, default=0.0)
                if (
                    maximum_time + TRACE_DURATION_TOLERANCE_SECONDS
                    < float(minimum_duration_seconds)
                ):
                    errors.append(
                        f"trace ends at {maximum_time:.3f}s; expected at least "
                        f"{float(minimum_duration_seconds):.3f}s"
                    )
    except (OSError, UnicodeError) as error:
        errors.append(f"trace: {error}")
    try:
        timing_data = json.loads(timing.read_text(encoding="utf-8"))
        if int(timing_data.get("iterations", 0)) <= 0:
            errors.append("timing iterations is not positive")
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        errors.append(f"timing: {error}")
    try:
        with np.load(pose, allow_pickle=False) as archive:
            required = {"time_seconds", "joint_positions", "motion_ids"}
            if not required <= set(archive.files):
                errors.append("pose archive is missing required arrays")
            elif len(archive["time_seconds"]) < 2:
                errors.append("pose archive has fewer than two frames")
            elif minimum_duration_seconds is not None:
                maximum_time = float(np.max(archive["time_seconds"]))
                if (
                    maximum_time + TRACE_DURATION_TOLERANCE_SECONDS
                    < float(minimum_duration_seconds)
                ):
                    errors.append(
                        f"pose ends at {maximum_time:.3f}s; expected at least "
                        f"{float(minimum_duration_seconds):.3f}s"
                    )
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"pose: {error}")
    if not log.is_file():
        errors.append("log file is missing")
    return not errors, errors


def archive_incomplete_run_artifacts(stem: str, paths: Iterable[Path], runs_root: Path) -> Path:
    """Move a previous failed/incomplete attempt aside before retrying it."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    # Keep the full run key in filenames, not twice in the archive path:
    # long ablation names otherwise exceed Windows' legacy MAX_PATH limit.
    archive_key = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:16]
    destination = runs_root / "failed_attempts" / archive_key / timestamp
    destination.mkdir(parents=True, exist_ok=False)
    for path in paths:
        if path.exists():
            shutil.move(str(path), str(destination / path.name))
    return destination


def experiment_subprocess_env() -> dict[str, str]:
    """Match redirected Python stdout/stderr to the UTF-8 experiment logs."""
    env = os.environ.copy()
    # The parent's TextIO encoding does not configure a child's inherited fd.
    env["PYTHONIOENCODING"] = "utf-8"
    # Realtime runs already parallelise PLP and retrieval at the process level.
    # Prevent BLAS/OpenMP inside either worker from oversubscribing the machine
    # and pre-empting the latency-critical control process.
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    return env


def execute_runs(
    matrix: list[dict[str, Any]],
    output_dir: Path,
    catalog_path: Path,
    *,
    max_seconds: float,
    dry_run: bool,
    resume: bool,
    suite: str,
    max_attempts: int,
    artifact_validator=None,
    before_run=None,
    require_completed_status: bool = False,
    stop_on_failure: bool = False,
    control_rate_hz: float = 120.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validator = artifact_validator or validate_run_artifacts
    results = []
    failures = []
    runs_root = output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    previous_status_path = output_dir / "run_status.json"
    attempt_history: list[dict[str, Any]] = []
    previous_run_by_key: dict[str, dict[str, Any]] = {}
    if previous_status_path.is_file():
        try:
            previous_status = json.loads(previous_status_path.read_text(encoding="utf-8"))
            if previous_status.get("suite") == suite:
                attempt_history.extend(previous_status.get("attempt_history", []))
                previous_run_by_key = {
                    str(run.get("run_key")): run
                    for run in previous_status.get("runs", [])
                    if run.get("run_key")
                }
        except (OSError, ValueError, TypeError):
            pass

    def atomic_status_write() -> None:
        status_path = output_dir / "run_status.json"
        temporary = status_path.with_suffix(".json.tmp")
        persisted_results = results
        if require_completed_status:
            # An interruption while replaying the completed prefix must not
            # erase status/provenance for valid later runs in the same suite.
            visited = {record['run_key'] for record in results}
            persisted_results = [*results, *(record for key, record in previous_run_by_key.items() if key not in visited)]
        payload = {
            "schema_version": 1,
            "suite": suite,
            "expected_runs": len(matrix),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "runs": persisted_results,
            "failures": failures,
            "attempt_history": attempt_history,
        }
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        atomic_replace(temporary, status_path)

    for index, item in enumerate(matrix):
        if before_run is not None:
            before_run()
        if require_completed_status:
            print(f"[{suite} {index + 1}/{len(matrix)}] {item['condition']} seed={item['seed']} {item['source_id']}", flush=True)
        stem = f"{index:04d}_{item['condition']}_s{item['seed']}_{item['source_id']}"
        trace = runs_root / f"{stem}.csv"
        timing = runs_root / f"{stem}_timing.json"
        log = runs_root / f"{stem}.log"
        pose = runs_root / f"{stem}_poses.npz"
        command = [
            sys.executable,
            str(MATCHER_ENTRYPOINT),
            "--catalog",
            str(catalog_path),
            "--audio-input",
            str(item["audio"]),
            "--headless",
            "--realtime",
            "--max-seconds",
            str(max_seconds),
            "--control-rate-hz",
            str(float(control_rate_hz)),
            "--runtime-collision-check",
            "always",
            "--match-window-seconds",
            "6",
            "--match-interval-seconds",
            "1",
            "--experiment-warmup-seconds",
            "6",
            "--initial-motion-seed",
            str(item["seed"]),
            "--trace-csv",
            str(trace),
            "--timing-report",
            str(timing),
            "--experiment-pose-npz",
            str(pose),
            *item["condition_arguments"],
        ]
        record = {
            **{key: str(value) if isinstance(value, Path) else value for key, value in item.items()},
            "command": subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command),
            "trace": str(trace),
            "timing": str(timing),
            "log": str(log),
            "pose": str(pose),
            "run_key": stem,
        }
        if dry_run:
            record.update({"status": "dry_run", "returncode": None})
            results.append(record)
            atomic_status_write()
            continue
        if resume:
            valid, validation_errors = validator(
                trace,
                timing,
                pose,
                log,
                minimum_duration_seconds=float(
                    item.get("required_duration_seconds", max_seconds)
                ),
            )
            previous_record = previous_run_by_key.get(stem, {})
            if require_completed_status and (
                previous_record.get("status") not in {"completed", "reused"}
                or previous_record.get("cohort_identity") != item.get("cohort_identity")
                or previous_record.get("prepared_audio_sha256") != item.get("prepared_audio_sha256")
                or previous_record.get("command") != record["command"]
            ):
                valid = False
                validation_errors.append("No completed status with matching cohort and command")
            if require_completed_status and valid:
                actual_hashes = {key: file_sha256(Path(record[key])) for key in ("trace", "timing", "pose", "log")}
                if previous_record.get("artifact_sha256") != actual_hashes:
                    valid = False
                    validation_errors.append("Artifact hashes differ from completed status")
                else:
                    record["artifact_sha256"] = actual_hashes
            if valid:
                record.update(
                    {
                        "status": "reused",
                        "returncode": 0,
                        "validation": "complete",
                    }
                )
                results.append(record)
                atomic_status_write()
                continue
            if any(path.exists() for path in (trace, timing, pose, log)):
                record["previous_artifact_errors"] = validation_errors
                previous_record = previous_run_by_key.get(stem)
                if previous_record and previous_record.get("status") != "dry_run":
                    attempt_history.append(copy.deepcopy(previous_record))
                archive_path = archive_incomplete_run_artifacts(
                    stem,
                    (trace, timing, pose, log),
                    runs_root,
                )
                record["archived_previous_artifacts"] = str(archive_path)
                attempt_history.append(
                    {
                        "run_key": stem,
                        "status": "incomplete_previous_attempt",
                        "artifact_errors": validation_errors,
                        "archive": str(archive_path),
                        "recorded_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
        record["status"] = "running"
        record["started_utc"] = datetime.now(timezone.utc).isoformat()
        results.append(record)
        atomic_status_write()
        for attempt in range(1, max(int(max_attempts), 1) + 1):
            record["attempt"] = attempt
            record["status"] = "running"
            record["started_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_status_write()
            started = time.perf_counter()
            with log.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                    env=experiment_subprocess_env(),
                )
            record.update(
                {
                    "status": "completed" if completed.returncode == 0 else "failed",
                    "returncode": completed.returncode,
                    "wall_seconds": time.perf_counter() - started,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            valid, validation_errors = validator(
                trace,
                timing,
                pose,
                log,
                minimum_duration_seconds=float(
                    item.get("required_duration_seconds", max_seconds)
                ),
            )
            if completed.returncode == 0 and not valid:
                record["status"] = "invalid"
                record["artifact_errors"] = validation_errors
            if record["status"] == "completed" and require_completed_status:
                record["artifact_sha256"] = {
                    key: file_sha256(Path(record[key])) for key in ("trace", "timing", "pose", "log")}
            atomic_status_write()
            if record["status"] == "completed":
                break
            attempt_record = copy.deepcopy(record)
            if attempt < max(int(max_attempts), 1):
                archive_path = archive_incomplete_run_artifacts(
                    stem,
                    (trace, timing, pose, log),
                    runs_root,
                )
                attempt_record["archive"] = str(archive_path)
            attempt_history.append(attempt_record)
        if record["status"] not in {"completed", "reused"}:
            failures.append(record)
        atomic_status_write()
        if failures and stop_on_failure:
            break
    return results, failures


def acceptance_report(timing: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    constraints = protocol["realtime_constraints"]
    runs = timing.get("runs", [])
    checks: list[dict[str, Any]] = []

    def maximum(key: str) -> float | None:
        values = [
            float(run[key])
            for run in runs
            if isinstance(run.get(key), (int, float)) and math.isfinite(float(run[key]))
        ]
        return max(values, default=None)

    def add_check(
        name: str,
        observed: float | None,
        *,
        relation: str = "max",
        limit_override: float | None = None,
    ) -> None:
        limit = (
            float(constraints[name])
            if limit_override is None
            else float(limit_override)
        )
        tolerance = max(abs(limit) * 1e-9, 1e-9)
        if observed is None:
            passed = False
        elif relation == "min":
            passed = observed + tolerance >= limit
        else:
            passed = observed <= limit + tolerance
        checks.append(
            {
                "name": name,
                "observed": observed,
                "limit": limit,
                "relation": relation,
                "evaluated": observed is not None,
                "passed": passed,
            }
        )

    add_check("query_p95_ms", maximum("retrieval_waveform_to_match_ms_p95"))
    add_check("control_work_p99_ms", maximum("work_ms_p99"))
    add_check("deadline_miss_ratio", maximum("deadline_miss_ratio"))

    attempts = sum(float(run.get("retrieval_submit_attempts", 0)) for run in runs)
    completed = sum(float(run.get("retrieval_completed", 0)) for run in runs)
    busy = sum(float(run.get("retrieval_busy_submissions", 0)) for run in runs)
    add_check(
        "completed_query_slot_rate",
        completed / attempts if attempts > 0 else None,
        relation="min",
    )
    add_check("busy_submission_rate", busy / attempts if attempts > 0 else None)
    add_check("ready_pool_underruns", maximum("ready_pool_underruns"))
    add_check("preparation_hold_last_events", maximum("hold_last_events"))
    add_check("max_joint_speed_rad_s", maximum("limiter_max_output_speed_rad_s"))
    add_check(
        "max_joint_acceleration_rad_s2",
        maximum("limiter_max_output_acceleration_rad_s2"),
    )
    add_check("joint_limit_violations", maximum("joint_limit_violations"))
    collision_checks = sum(float(run.get("collision_checks", 0)) for run in runs)
    add_check(
        "self_collision_violations",
        maximum("self_collision_violations") if collision_checks > 0 else None,
    )
    return {
        "checks": checks,
        "intervention_diagnostics": {
            "collision_dynamics_infeasible_max": maximum(
                "collision_dynamics_infeasible"
            ),
            "collision_projection_fallbacks_max": maximum(
                "collision_projection_fallbacks"
            ),
            "interpretation": (
                "Intervention counters describe safe fallback use; the safety "
                "contract is evaluated on final emitted pose clearance, joint "
                "limits, speed and acceleration."
            ),
        },
        "all_evaluated": bool(checks) and all(check["evaluated"] for check in checks),
        "all_evaluated_passed": bool(checks) and all(check["passed"] for check in checks),
        "note": "Unevaluated constraints are not silently counted as passes.",
    }


def stitched_change_spec(path: Path = STITCHED_MANIFEST) -> tuple[list[float], list[str]]:
    """Return exact crossfade-aware change clocks and target AIST++ genres."""

    if not path.is_file():
        return [], []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    segments = list(manifest.get("segments", []))
    crossfade = float(manifest.get("crossfade_seconds", 0.0))
    change_times: list[float] = []
    genres: list[str] = []
    clock = 0.0
    for index, segment in enumerate(segments):
        if index:
            change_times.append(clock - crossfade)
            stem = Path(str(segment.get("source", ""))).stem
            genres.append(stem[1:3] if len(stem) >= 3 and stem.startswith("g") else "")
        clock += float(segment.get("duration_seconds", 0.0)) - (
            crossfade if index else 0.0
        )
    return change_times, genres


def repeated_stitched_changes(duration: float, path: Path = STITCHED_MANIFEST):
    times, genres = stitched_change_spec(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    segments = manifest["segments"]
    period = sum(float(s["duration_seconds"]) for s in segments) - float(manifest.get("crossfade_seconds", 0)) * (len(segments) - 1)
    if period <= 0:
        raise ValueError("Invalid stitched period")
    first = Path(segments[0]["source"]).stem[1:3]
    events = []
    for cycle in range(int(duration / period) + 1):
        if cycle:
            events.append((cycle * period, first))
        events.extend((cycle * period + t, genre) for t, genre in zip(times, genres))
    events = sorted((t, g) for t, g in events if 0 < t < duration)
    return [t for t, _ in events], [g for _, g in events]


def response_reports(
    run_results: Iterable[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reports = []
    for run in run_results:
        trace = Path(str(run.get("trace", "")))
        source_id = str(run.get("source_id", ""))
        expected_genres: list[str] | None = None
        if source_id in {"aistpp_stitched_test", "aistpp_stitched_longrun"}:
            change_times, expected_genres = repeated_stitched_changes(
                float(run.get("required_duration_seconds", run.get("duration", 40))))
        else:
            change_times = []
            change_manifest = (
                HUMANOID_DIR
                / "data"
                / "test_audio"
                / "matcher_change_streams"
                / "manifest.json"
            )
            if change_manifest.is_file():
                data = json.loads(change_manifest.read_text(encoding="utf-8"))
                stream = next(
                    (
                        item
                        for item in data.get("streams", [])
                        if Path(str(item.get("path", ""))).stem == source_id
                    ),
                    None,
                )
                if stream is not None:
                    change_times = [
                        float(value.get("time_seconds", 0.0))
                        if isinstance(value, dict)
                        else float(value)
                        for value in stream.get("changes", [])
                    ]
        if not change_times or not trace.is_file():
            continue
        result = response_latency_from_trace(trace, change_times, expected_genres)
        timing_path = Path(str(run.get("timing", "")))
        timing_data = (
            json.loads(timing_path.read_text(encoding="utf-8"))
            if timing_path.is_file()
            else {}
        )

        def gaps(first: str, second: str) -> list[float]:
            return [
                float(item[second]) - float(item[first])
                for item in result["changes"]
                if item[first] is not None and item[second] is not None
            ]

        result["decomposition"] = {
            "causal_perception_window_seconds": float(protocol["match_window_seconds"]),
            "query_schedule_upper_bound_seconds": float(protocol["match_interval_seconds"]),
            "query_compute_p95_seconds": (
                float(timing_data["retrieval_waveform_to_match_ms_p95"]) / 1_000.0
                if "retrieval_waveform_to_match_ms_p95" in timing_data
                else None
            ),
            "observed_match_to_pending_seconds": distribution_summary(
                gaps("first_match_seconds", "pending_seconds")
            ),
            "observed_pending_to_switch_seconds": distribution_summary(
                gaps("pending_seconds", "switch_start_seconds")
            ),
            "observed_switch_to_stable_seconds": distribution_summary(
                gaps("switch_start_seconds", "stable_switch_seconds")
            ),
        }
        result.update(
            {
                "condition": run.get("condition"),
                "seed": run.get("seed"),
                "change_times_include_crossfade": True,
            }
        )
        reports.append(result)
    return reports


def ablation_report(
    run_results: Iterable[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    cluster_by_motion: Mapping[str, str],
) -> dict[str, Any]:
    """Pair ablations with full runs by source and seed, then Holm-correct tests."""

    records: dict[tuple[str, int, str], dict[str, float]] = {}
    for run in run_results:
        if run.get("status") not in {"completed", "reused"}:
            continue
        trace_path = Path(str(run.get("trace", "")))
        timing_path = Path(str(run.get("timing", "")))
        pose_path = Path(str(run.get("pose", "")))
        if not (trace_path.is_file() and timing_path.is_file() and pose_path.is_file()):
            continue
        trace = aggregate_trace_csv(
            trace_path,
            cluster_by_motion,
            start_seconds=float(protocol["match_window_seconds"]),
        )
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        pose = aggregate_pose_npz(
            pose_path,
            evaluation_fps=float(protocol["literature"]["evaluation_fps"]),
            cluster_by_motion=cluster_by_motion,
            start_seconds=float(protocol["match_window_seconds"]),
        )
        expected_genres = set(
            str(value)
            for value in protocol.get("cc0_labels", {})
            .get(str(run["source_id"]), {})
            .get("expected_genres", [])
        )
        with trace_path.open("r", encoding="utf-8", newline="") as handle:
            match_genres = [
                row.get("top_genre", "")
                for row in csv.DictReader(handle)
                if row.get("event") == "match"
            ]
        genre_recall_at_1 = (
            sum(value in expected_genres for value in match_genres) / len(match_genres)
            if expected_genres and match_genres
            else math.nan
        )
        records[(str(run["condition"]), int(run["seed"]), str(run["source_id"]))] = {
            "genre_recall_at_1": float(genre_recall_at_1),
            "bas_harmonic": float(pose["beat"]["bas_harmonic"]) if pose["beat"]["bas_harmonic"] is not None else math.nan,
            "normalized_selection_entropy": float(
                trace["selection"]["normalized_selection_entropy"]
            ),
            "unique_motions": float(trace["selection"]["unique_motions"]),
            "mean_match_score": float(trace["match_score"]["mean"]),
            "joint_jerk_p95_rad_s3": float(
                pose["continuity"]["joint_jerk_rad_s3"]["p95"]
            ),
            "deadline_miss_ratio": float(timing.get("deadline_miss_ratio", math.nan)),
            "query_p95_ms": float(
                timing.get("retrieval_waveform_to_match_ms_p95", math.nan)
            ),
        }

    conditions = sorted({key[0] for key in records if key[0] != "full"})
    metrics = sorted(next(iter(records.values()))) if records else []
    comparisons: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for condition in conditions:
        paired_keys = sorted(
            (seed, source)
            for name, seed, source in records
            if name == condition and ("full", seed, source) in records
        )
        condition_report: dict[str, Any] = {"pairs": len(paired_keys), "metrics": {}}
        for metric in metrics:
            # Seeds are repeated measurements, not independent music samples.
            ablated, full = [], []
            for source in sorted({source for _, source in paired_keys}):
                pairs = [(records[(condition, seed, source)][metric], records[("full", seed, source)][metric])
                         for seed, candidate in paired_keys if candidate == source]
                pairs = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
                if pairs:
                    ablated.append(float(np.mean([a for a, _ in pairs])))
                    full.append(float(np.mean([b for _, b in pairs])))
            test = paired_permutation_test(ablated, full)
            key = f"{condition}:{metric}"
            raw_p_values[key] = float(test["p_value"])
            condition_report["metrics"][metric] = test
        comparisons[condition] = condition_report
    adjusted = holm_adjust(raw_p_values)
    for condition, condition_report in comparisons.items():
        for metric, test in condition_report["metrics"].items():
            test["holm_adjusted_p_value"] = adjusted[f"{condition}:{metric}"]

    quality = protocol["quality_constraints"]
    no_diversity = comparisons.get("no_diversity", {}).get("metrics", {})
    authored = comparisons.get("authored_timing", {}).get("metrics", {})
    score_drop = float(
        no_diversity.get("mean_match_score", {}).get("mean_difference", math.nan)
    )
    no_diversity_score = np.mean(
        [
            values["mean_match_score"]
            for (name, _, _), values in records.items()
            if name == "no_diversity"
        ]
    ) if any(name == "no_diversity" for name, _, _ in records) else math.nan
    score_loss_fraction = score_drop / max(abs(float(no_diversity_score)), 1e-12)
    deadline_increase = -float(
        authored.get("deadline_miss_ratio", {}).get("mean_difference", math.nan)
    )
    def directional_check(condition: str, metric: str, relation: str) -> dict[str, Any]:
        comparison = comparisons.get(condition, {}).get("metrics", {}).get(metric, {})
        difference = float(comparison.get("mean_difference", math.nan))
        passed = math.isfinite(difference) and (
            difference < 0.0 if relation == "ablation_lower" else difference > 0.0
        ) and float(comparison.get("holm_adjusted_p_value", 1.0)) < 0.05
        return {
            "condition": condition,
            "metric": metric,
            "difference": difference if math.isfinite(difference) else None,
            "difference_direction": "ablation_minus_full",
            "expected": relation,
            "passed": passed,
        }
    return {
        "evaluated": bool(comparisons),
        "paired_unit": "source_id, averaging matched seeds before testing",
        "difference_direction": "ablation_minus_full",
        "comparisons": comparisons,
        "quality_checks": {
            "diversity_mean_score_loss_fraction": {
                "observed": score_loss_fraction if math.isfinite(score_loss_fraction) else None,
                "limit": float(quality["diversity_max_mean_score_loss_fraction"]),
                "passed": bool(math.isfinite(score_loss_fraction)) and score_loss_fraction <= float(quality["diversity_max_mean_score_loss_fraction"]),
            },
            "beat_sync_deadline_miss_increase": {
                "observed": deadline_increase if math.isfinite(deadline_increase) else None,
                "limit": float(quality["beat_sync_max_deadline_miss_increase"]),
                "passed": bool(math.isfinite(deadline_increase)) and deadline_increase <= float(quality["beat_sync_max_deadline_miss_increase"]),
            },
            "full_recall_exceeds_legacy": directional_check(
                "legacy_retrieval", "genre_recall_at_1", "ablation_lower"
            ),
            "beat_sync_bas_exceeds_authored": directional_check(
                "authored_timing", "bas_harmonic", "ablation_lower"
            ),
            "full_transition_jerk_below_fixed_entry": directional_check(
                "fixed_entry_simple_transition", "joint_jerk_p95_rad_s3", "ablation_higher"
            ),
            "diversity_entropy_exceeds_no_diversity": directional_check(
                "no_diversity", "normalized_selection_entropy", "ablation_lower"
            ),
        },
    }


def write_csv_summary(path: Path, report: Mapping[str, Any]) -> None:
    rows = []
    ranking = report["quality"]["strict_leave_one_music"]["ranking"]
    rows.extend(
        {"section": "retrieval", "metric": key, "value": value}
        for key, value in ranking.items()
        if isinstance(value, (int, float))
    )
    for check in report["acceptance"]["checks"]:
        rows.append(
            {
                "section": "acceptance",
                "metric": check["name"],
                "value": check["observed"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or [{"status": "not_evaluated"}])


def write_result_artifacts(output_dir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Emit the preregistered four main tables, ablation table and plots."""

    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    retrieval_rows = [
        {"section": "strict_leave_one_music", "metric": key, "value": value}
        for key, value in report["quality"]["strict_leave_one_music"]["ranking"].items()
        if isinstance(value, (int, float))
    ]
    retrieval_rows += [
        {"section": "rejection", "metric": key, "value": value}
        for key, value in report.get("audio_evaluation", {}).get("rejection", {}).items()
        if isinstance(value, (int, float))
    ]
    table_paths = {
        "retrieval_rejection": tables / "table_retrieval_rejection.csv",
        "rhythm_diversity_physical": tables / "table_rhythm_diversity_physical.csv",
        "continuity_safety": tables / "table_continuity_safety.csv",
        "latency": tables / "table_latency.csv",
        "ablation": tables / "table_ablation.csv",
    }
    _write_rows(table_paths["retrieval_rejection"], retrieval_rows)

    rhythm_rows = []
    for item in report.get("pose_aggregates", []):
        rhythm_rows.append(
            {
                "run": item.get("pose_npz"),
                **item.get("beat", {}),
                **item.get("physical", {}),
                **{
                    f"selection_{key}": value
                    for key, value in item.get("selection", {}).items()
                    if isinstance(value, (int, float))
                },
            }
        )
    _write_rows(table_paths["rhythm_diversity_physical"], rhythm_rows)

    continuity_rows = []
    for item in report.get("pose_aggregates", []):
        continuity_rows.append(
            {
                "run": item.get("pose_npz"),
                "speed_p95": item.get("continuity", {}).get("joint_speed_rad_s", {}).get("p95"),
                "acceleration_p95": item.get("continuity", {}).get("joint_acceleration_rad_s2", {}).get("p95"),
                "jerk_p95": item.get("continuity", {}).get("joint_jerk_rad_s3", {}).get("p95"),
                "ground_penetration_max_m": item.get("physical", {}).get("ground_penetration_max_m"),
            }
        )
    for check in report.get("acceptance", {}).get("checks", []):
        continuity_rows.append(
            {
                "run": "acceptance",
                "safety_metric": check.get("name"),
                "observed": check.get("observed"),
                "limit": check.get("limit"),
                "passed": check.get("passed"),
            }
        )
    _write_rows(table_paths["continuity_safety"], continuity_rows)

    latency_rows = []
    for run in report.get("timing", {}).get("runs", []):
        for key, value in run.items():
            if isinstance(value, (int, float)) and (
                "_ms" in key or "deadline" in key or key.startswith("retrieval_")
            ):
                latency_rows.append({"run": run.get("path"), "metric": key, "value": value})
    _write_rows(table_paths["latency"], latency_rows)

    ablation_rows = []
    for condition, item in report.get("ablation", {}).get("comparisons", {}).items():
        for metric, values in item.get("metrics", {}).items():
            ablation_rows.append({"condition": condition, "metric": metric, **values})
    _write_rows(table_paths["ablation"], ablation_rows)

    figure_paths: dict[str, str] = {}
    plot_error = None
    plot_backend = "matplotlib"
    work = sorted(
        float(run["work_ms_p99"])
        for run in report.get("timing", {}).get("runs", [])
        if "work_ms_p99" in run
    )
    bas = [
        float(item["beat"]["bas_harmonic"])
        for item in report.get("pose_aggregates", [])
    ]
    traces = report.get("trace_aggregates", [])
    poses = report.get("pose_aggregates", [])
    responses = report.get("response_latency", [])
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def save(name: str) -> None:
            path = figures / f"{name}.png"
            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()
            figure_paths[name] = str(path)

        if work:
            plt.step(work, np.arange(1, len(work) + 1) / len(work), where="post")
            plt.xlabel("Per-run control work p99 (ms)")
            plt.ylabel("Empirical CDF")
            save("latency_cdf")

        if bas:
            plt.hist(bas, bins=min(max(len(bas), 1), 20))
            plt.xlabel("Bidirectional BAS harmonic mean")
            plt.ylabel("Runs")
            save("bas_distribution")

        if traces:
            labels = [Path(str(item["trace"])).stem for item in traces]
            coverage = [item["selection"]["unique_motions"] for item in traces]
            plt.bar(np.arange(len(labels)), coverage)
            plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
            plt.ylabel("Unique selected motions")
            save("selection_coverage")

        if poses:
            speed = [item["continuity"]["joint_speed_rad_s"]["p95"] for item in poses]
            jerk = [item["continuity"]["joint_jerk_rad_s3"]["p95"] for item in poses]
            positions = np.arange(len(poses))
            plt.semilogy(positions, np.maximum(speed, 1e-12), "o-", label="speed p95")
            plt.semilogy(positions, np.maximum(jerk, 1e-12), "o-", label="jerk p95")
            plt.xlabel("Run")
            plt.ylabel("Magnitude (log scale)")
            plt.legend()
            save("switch_speed_jerk")

        if responses:
            milestone_keys = [
                "first_match_seconds",
                "pending_seconds",
                "switch_start_seconds",
                "blend_50_seconds",
                "stable_switch_seconds",
            ]
            for index, item in enumerate(responses):
                values = [item["summary"][key]["p50"] for key in milestone_keys]
                plt.plot(milestone_keys, values, "o-", label=f"run {index}")
            plt.xticks(rotation=25, ha="right")
            plt.ylabel("Latency from change (s), p50")
            if len(responses) <= 10:
                plt.legend()
            save("change_response_timeline")
    except (ImportError, OSError, ValueError) as error:
        plot_error = f"{type(error).__name__}: {error}"
        plot_backend = "svg_fallback"

        def svg_plot(name: str, title: str, series: Mapping[str, list[float]]) -> None:
            clean = {
                label: [float(value) for value in values if math.isfinite(float(value))]
                for label, values in series.items()
            }
            clean = {label: values for label, values in clean.items() if values}
            if not clean:
                return
            width, height, margin = 800, 440, 55
            all_values = [value for values in clean.values() for value in values]
            low, high = min(all_values), max(all_values)
            span = max(high - low, 1e-12)
            colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c")
            elements = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{width / 2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>',
                f'<path d="M {margin} {margin} V {height-margin} H {width-margin}" stroke="#333" fill="none"/>',
                f'<text x="8" y="{margin}" font-family="sans-serif" font-size="11">{high:.4g}</text>',
                f'<text x="8" y="{height-margin}" font-family="sans-serif" font-size="11">{low:.4g}</text>',
            ]
            for series_index, (label, values) in enumerate(clean.items()):
                x_span = max(len(values) - 1, 1)
                points = " ".join(
                    f"{margin + index * (width - 2 * margin) / x_span:.2f},"
                    f"{height - margin - (value - low) * (height - 2 * margin) / span:.2f}"
                    for index, value in enumerate(values)
                )
                color = colors[series_index % len(colors)]
                elements.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
                )
                elements.append(
                    f'<text x="{margin + series_index * 150}" y="{height-15}" font-family="sans-serif" font-size="11" fill="{color}">{label}</text>'
                )
            elements.append("</svg>")
            path = figures / f"{name}.svg"
            path.write_text("\n".join(elements) + "\n", encoding="utf-8")
            figure_paths[name] = str(path)

        svg_plot(
            "latency_cdf",
            "Sorted control-work p99 by run",
            {"work p99 ms": work},
        )
        svg_plot("bas_distribution", "BAS distribution by run", {"BAS": bas})
        svg_plot(
            "selection_coverage",
            "Unique motion coverage by run",
            {"motions": [item["selection"]["unique_motions"] for item in traces]},
        )
        svg_plot(
            "switch_speed_jerk",
            "Joint speed and jerk p95 by run",
            {
                "speed": [item["continuity"]["joint_speed_rad_s"]["p95"] for item in poses],
                "jerk": [item["continuity"]["joint_jerk_rad_s3"]["p95"] for item in poses],
            },
        )
        milestone_keys = [
            "first_match_seconds",
            "pending_seconds",
            "switch_start_seconds",
            "blend_50_seconds",
            "stable_switch_seconds",
        ]
        svg_plot(
            "change_response_timeline",
            "Music-change response milestones",
            {
                f"run {index}": [item["summary"][key]["p50"] for key in milestone_keys]
                for index, item in enumerate(responses)
            },
        )
        if figure_paths:
            plot_error = None

    return {
        "tables": {key: str(value) for key, value in table_paths.items()},
        "figures": figure_paths,
        "plot_backend": plot_backend,
        "plot_error": plot_error,
    }


def main() -> int:
    args = parse_args()
    catalog_path = args.catalog.resolve()
    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    catalog = MusicCatalog.load(catalog_path)
    extractor = None if args.no_audio_evaluation else make_extractor(catalog)
    cluster_by_motion = {
        motion_id: profile.motion_cluster_id
        for motion_id, profile in catalog.motions.items()
        if profile.preflight_passed
    }

    quality = {
        "strict_leave_one_music": strict_leave_one_music_evaluation(
            catalog,
            bootstrap_iterations=args.bootstrap_iterations,
        ),
        "source_motion_sanity": source_motion_sanity(catalog),
    }
    audio_report = (
        audio_set_evaluation(
            catalog,
            extractor,
            protocol,
            args.cc0_root.resolve(),
            args.negative_audio_root.resolve() if args.negative_audio_root else None,
        )
        if extractor is not None
        else {"evaluated": False, "reason": "--no-audio-evaluation"}
    )
    trace_reports = []
    failures = []
    for path in args.trace:
        try:
            trace_reports.append(
                aggregate_trace_csv(
                    path.resolve(),
                    cluster_by_motion,
                    start_seconds=float(protocol["match_window_seconds"]),
                )
            )
        except (OSError, ValueError) as error:
            failures.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
    timing = aggregate_timing_reports(path.resolve() for path in args.timing)
    failures.extend(timing["failures"])
    pose_reports = []
    for path in args.pose:
        try:
            pose_reports.append(
                aggregate_pose_npz(
                    path.resolve(),
                    evaluation_fps=float(protocol["literature"]["evaluation_fps"]),
                    cluster_by_motion=cluster_by_motion,
                    start_seconds=float(protocol["match_window_seconds"]),
                )
            )
        except (OSError, ValueError) as error:
            failures.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})

    matrix = run_matrix(
        args.execute_suite,
        catalog,
        protocol,
        args.cc0_root.resolve(),
    )
    if args.max_runs is not None:
        matrix = matrix[: max(args.max_runs, 0)]
    default_seconds = {
        "smoke": 10.0,
        "literature": protocol["literature"]["clip_seconds"] + protocol["match_window_seconds"],
        "full": 30.0,
        "ablation": 40.0,
        "longrun": (
            protocol["longrun"]["stable_seconds"]
            + protocol["longrun"]["warmup_seconds"]
        ),
        "none": 0.0,
    }[args.execute_suite]
    effective_seconds = float(args.max_seconds or default_seconds)
    if args.execute_suite == "longrun" and matrix:
        long_audio = output_dir / "generated_streams" / "aistpp_stitched_longrun.wav"
        if not args.dry_run:
            write_repeated_audio(
                Path(matrix[0]["audio"]),
                long_audio,
                seconds=effective_seconds + 1.0,
                sample_rate=int(catalog.metadata["extractor"]["sample_rate"]),
            )
        for item in matrix:
            item["original_audio"] = item["audio"]
            item["audio"] = long_audio
            item["source_id"] = "aistpp_stitched_longrun"
            item["audio_was_repeated"] = True
            item["required_duration_seconds"] = effective_seconds
    elif matrix:
        prepare_short_audio_inputs(
            matrix,
            output_dir,
            required_seconds=effective_seconds,
            sample_rate=int(catalog.metadata["extractor"]["sample_rate"]),
            dry_run=args.dry_run,
        )
    run_results, run_failures = execute_runs(
        matrix,
        output_dir,
        catalog_path,
        max_seconds=effective_seconds,
        dry_run=args.dry_run,
        resume=args.resume,
        suite=args.execute_suite,
        max_attempts=args.max_attempts,
    ) if matrix else ([], [])
    failures.extend(run_failures)

    if run_results and not args.dry_run:
        new_traces = [Path(item["trace"]) for item in run_results if Path(item["trace"]).is_file()]
        new_timings = [Path(item["timing"]) for item in run_results if Path(item["timing"]).is_file()]
        new_poses = [Path(item["pose"]) for item in run_results if Path(item["pose"]).is_file()]
        trace_reports.extend(
            aggregate_trace_csv(
                path,
                cluster_by_motion,
                start_seconds=float(protocol["match_window_seconds"]),
            )
            for path in new_traces
        )
        pose_reports.extend(
            aggregate_pose_npz(
                path,
                evaluation_fps=float(protocol["literature"]["evaluation_fps"]),
                cluster_by_motion=cluster_by_motion,
                start_seconds=float(protocol["match_window_seconds"]),
            )
            for path in new_poses
        )
        timing = aggregate_timing_reports([*args.timing, *new_timings])

    report = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": environment_manifest(catalog, protocol_path, extractor),
        "protocol": protocol,
        "catalog": {
            "tracks": len(catalog.tracks),
            "segments": len(catalog.segment_metadata),
            "motions": len(catalog.motions),
            "preflight_passed_motions": sum(
                profile.preflight_passed for profile in catalog.motions.values()
            ),
            "gmr_files": len(list((HUMANOID_DIR / "data" / "aistpp_gmr").glob("*.pkl"))),
            "literature_music_ids": literature_music_ids(
                catalog, protocol["literature"]["clips_per_genre"]
            ),
        },
        "quality": quality,
        "audio_evaluation": audio_report,
        "literature_motion_metrics": load_feature_bundle(
            args.feature_bundle.resolve() if args.feature_bundle else None
        ),
        "runs": run_results,
        "trace_aggregates": trace_reports,
        "pose_aggregates": pose_reports,
        "response_latency": response_reports(run_results, protocol),
        "ablation": ablation_report(run_results, protocol, cluster_by_motion),
        "timing": timing,
        "acceptance": acceptance_report(timing, protocol),
        "failures": failures,
        "limitations": [
            "MuJoCo results are kinematic playback diagnostics, not torque-controlled balance or real-robot evidence.",
            "FID/Div are emitted only when an explicitly identified SMPL/AIST++ feature bundle is supplied.",
            "No objective metric fully replaces human judgement of naturalness or choreography aesthetics.",
        ],
    }
    report["artifacts"] = write_result_artifacts(output_dir, report)
    json_path = output_dir / "experiment_report.json"
    csv_path = output_dir / "experiment_summary.csv"
    manifest_path = output_dir / "experiment_manifest.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(report["manifest"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv_summary(csv_path, report)
    print(f"Wrote {json_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {csv_path}")
    return 1 if run_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
