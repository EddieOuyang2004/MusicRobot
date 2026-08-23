"""Benchmark retrieval quality and runtime performance of the humanoid matcher.

The benchmark uses the descriptors already stored in the production catalog for
repeatable retrieval measurements.  When the catalog's feature model and runtime
are available, it also measures waveform-to-match latency and gain invariance.

Run from the repository root::

    python realtime/humanoid_robot/src/test/benchmark_humanoid_matcher.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent
HUMANOID_DIR = SRC_DIR.parent
ROOT = HUMANOID_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from music_motion_catalog import (  # noqa: E402
    AudioDescriptor,
    AudioFeatureExtractor,
    MusicCatalog,
    MusicMotionMatcher,
    iter_audio_windows,
    load_audio_mono,
)


DEFAULT_CATALOG = HUMANOID_DIR / "data" / "music_catalog" / "catalog.json"
DEFAULT_OUTPUT_DIR = TEST_DIR / "output"
DEFAULT_RESULT_STEM = "humanoid_matcher_performance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure humanoid music-motion matcher retrieval quality, latency, "
            "throughput, and (when available) end-to-end audio performance."
        )
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-stem", default=DEFAULT_RESULT_STEM)
    parser.add_argument(
        "--latency-iterations",
        type=int,
        default=1_000,
        help="Number of timed catalog-descriptor match calls (default: 1000).",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=50,
        help="Untimed match calls before the latency benchmark (default: 50).",
    )
    parser.add_argument(
        "--gain-check-limit",
        type=int,
        default=3,
        help="Tracks used for waveform gain checks when the model is available.",
    )
    parser.add_argument(
        "--feature-iterations",
        type=int,
        default=5,
        help="Timed waveform-to-match calls when the model is available.",
    )
    return parser.parse_args()


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def process_rss_bytes() -> int | None:
    """Return current process RSS without requiring the optional psutil package."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_info.restype = wintypes.BOOL
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if get_info(handle, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError):
            return None
        return None

    try:
        import resource

        maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS and BSD report bytes.
        return maximum_rss * 1024 if sys.platform.startswith("linux") else maximum_rss
    except (ImportError, OSError):
        return None


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


def percentile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(np.percentile(array, quantile)) if array.size else 0.0


def latency_summary(latencies_ms: list[float]) -> dict[str, float | int]:
    total_seconds = sum(latencies_ms) / 1_000.0
    return {
        "samples": len(latencies_ms),
        "mean_ms": float(statistics.fmean(latencies_ms)) if latencies_ms else 0.0,
        "median_ms": float(statistics.median(latencies_ms)) if latencies_ms else 0.0,
        "p95_ms": percentile(latencies_ms, 95.0),
        "p99_ms": percentile(latencies_ms, 99.0),
        "min_ms": min(latencies_ms, default=0.0),
        "max_ms": max(latencies_ms, default=0.0),
        "queries_per_second": (
            float(len(latencies_ms) / total_seconds) if total_seconds > 0.0 else 0.0
        ),
    }


def random_recall_probability(candidate_count: int, relevant_count: int, k: int) -> float:
    if candidate_count <= 0 or relevant_count <= 0:
        return 0.0
    draw_count = min(max(int(k), 0), candidate_count)
    irrelevant_count = candidate_count - relevant_count
    if draw_count > irrelevant_count:
        return 1.0
    return 1.0 - (
        math.comb(irrelevant_count, draw_count)
        / math.comb(candidate_count, draw_count)
    )


def ranked_tracks_excluding_query(
    catalog: MusicCatalog,
    matcher: MusicMotionMatcher,
    query_index: int,
) -> list[str]:
    query_metadata = catalog.segment_metadata[query_index]
    query_music = str(query_metadata["music_id"])
    query_embedding = matcher._standardized_unit(
        catalog.embeddings[query_index],
        catalog.embedding_mean,
        catalog.embedding_std,
    )
    query_rhythm = matcher._standardized_unit(
        catalog.rhythm_timbre[query_index],
        catalog.rhythm_mean,
        catalog.rhythm_std,
    )
    embedding_scores = matcher._similarities(
        matcher._catalog_embeddings,
        query_embedding,
    )
    rhythm_scores = matcher._similarities(
        matcher._catalog_rhythm,
        query_rhythm,
    )
    if catalog.tags.shape[1]:
        query_tags = catalog.tags[query_index]
        query_tags = query_tags / max(float(np.linalg.norm(query_tags)), 1e-12)
        tag_scores = matcher._similarities(matcher._catalog_tags, query_tags)
        segment_scores = (
            0.70 * embedding_scores + 0.20 * rhythm_scores + 0.10 * tag_scores
        )
    else:
        segment_scores = (
            (0.70 / 0.90) * embedding_scores + (0.20 / 0.90) * rhythm_scores
        )

    grouped: dict[str, list[float]] = defaultdict(list)
    for index, segment in enumerate(catalog.segment_metadata):
        music_id = str(segment["music_id"])
        if music_id != query_music:
            grouped[music_id].append(float(segment_scores[index]))

    return sorted(
        grouped,
        key=lambda music_id: float(
            np.mean(sorted(grouped[music_id], reverse=True)[:3])
        ),
        reverse=True,
    )


def evaluate_quality(
    catalog: MusicCatalog,
    matcher: MusicMotionMatcher,
) -> tuple[dict[str, Any], list[AudioDescriptor]]:
    descriptors = [
        descriptor_from_segment(catalog, index)
        for index in range(len(catalog.segment_metadata))
    ]
    self_track_hits = 0
    motion_source_hits = {1: 0, 3: 0, 10: 0}
    queries_with_motions = 0

    genre_hits = {1: 0, 3: 0, 5: 0}
    per_genre: dict[str, dict[str, int]] = defaultdict(
        lambda: {"queries": 0, "hits_at_1": 0, "hits_at_3": 0, "hits_at_5": 0}
    )
    reciprocal_ranks: list[float] = []
    random_recall = {1: [], 3: [], 5: []}

    for index, descriptor in enumerate(descriptors):
        query_metadata = catalog.segment_metadata[index]
        source_music_id = str(query_metadata["music_id"])
        source_genre = str(query_metadata["genre"])
        result = matcher.match(descriptor, top_k_tracks=5, top_k_motions=10)
        if result.tracks and result.tracks[0].music_id == source_music_id:
            self_track_hits += 1
        if result.motions:
            queries_with_motions += 1
        ranked_motion_music = [item.music_id for item in result.motions]
        for k in motion_source_hits:
            if source_music_id in ranked_motion_music[:k]:
                motion_source_hits[k] += 1

        ranked_tracks = ranked_tracks_excluding_query(catalog, matcher, index)
        ranked_genres = [str(catalog.tracks[item]["genre"]) for item in ranked_tracks]
        matching_ranks = [
            rank
            for rank, genre in enumerate(ranked_genres, start=1)
            if genre == source_genre
        ]
        reciprocal_ranks.append(1.0 / matching_ranks[0] if matching_ranks else 0.0)
        per_genre[source_genre]["queries"] += 1
        relevant_tracks = sum(
            str(track["genre"]) == source_genre
            and str(music_id) != source_music_id
            for music_id, track in catalog.tracks.items()
        )
        for k in genre_hits:
            hit = source_genre in ranked_genres[:k]
            genre_hits[k] += int(hit)
            per_genre[source_genre][f"hits_at_{k}"] += int(hit)
            random_recall[k].append(
                random_recall_probability(len(ranked_tracks), relevant_tracks, k)
            )

    query_count = max(len(descriptors), 1)
    per_genre_rates = {
        genre: {
            "queries": values["queries"],
            "recall_at_1": values["hits_at_1"] / max(values["queries"], 1),
            "recall_at_3": values["hits_at_3"] / max(values["queries"], 1),
            "recall_at_5": values["hits_at_5"] / max(values["queries"], 1),
        }
        for genre, values in sorted(per_genre.items())
    }
    quality = {
        "query_count": len(descriptors),
        "same_segment_descriptor_track_recall_at_1": self_track_hits / query_count,
        "same_segment_descriptor_note": (
            "Sanity check only: each exact stored segment descriptor is queried "
            "against a catalog that contains that segment."
        ),
        "source_track_motion_recall_at_1": motion_source_hits[1] / query_count,
        "source_track_motion_recall_at_3": motion_source_hits[3] / query_count,
        "source_track_motion_recall_at_10": motion_source_hits[10] / query_count,
        "queries_with_motion_candidates_rate": queries_with_motions / query_count,
        "leave_one_music_genre_recall_at_1": genre_hits[1] / query_count,
        "leave_one_music_genre_recall_at_3": genre_hits[3] / query_count,
        "leave_one_music_genre_recall_at_5": genre_hits[5] / query_count,
        "leave_one_music_genre_mean_reciprocal_rank": float(
            statistics.fmean(reciprocal_ranks)
        ),
        "random_genre_recall_baseline_at_1": float(statistics.fmean(random_recall[1])),
        "random_genre_recall_baseline_at_3": float(statistics.fmean(random_recall[3])),
        "random_genre_recall_baseline_at_5": float(statistics.fmean(random_recall[5])),
        "per_genre": per_genre_rates,
    }
    return quality, descriptors


def benchmark_match_latency(
    matcher: MusicMotionMatcher,
    descriptors: list[AudioDescriptor],
    warmup_iterations: int,
    latency_iterations: int,
) -> dict[str, float | int]:
    if not descriptors:
        return latency_summary([])
    for index in range(max(warmup_iterations, 0)):
        matcher.match(descriptors[index % len(descriptors)])
    latencies_ms: list[float] = []
    for index in range(max(latency_iterations, 1)):
        started = time.perf_counter_ns()
        matcher.match(descriptors[index % len(descriptors)])
        elapsed_ns = time.perf_counter_ns() - started
        latencies_ms.append(elapsed_ns / 1_000_000.0)
    return latency_summary(latencies_ms)


def resolve_model_path(catalog: MusicCatalog, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return (
        path
        if path.is_absolute()
        else (catalog.catalog_path.parent / path).resolve()
    )


def first_audio_window(
    catalog: MusicCatalog,
    extractor: AudioFeatureExtractor,
    music_id: str,
) -> np.ndarray:
    track = catalog.tracks[music_id]
    variant = track["variants"][0]
    root = (
        catalog.catalog_path.parent / str(catalog.metadata["aistpp_root"])
    ).resolve()
    audio = load_audio_mono(root / variant["source_audio"], extractor.sample_rate)
    window_seconds = float(catalog.metadata["extractor"]["window_seconds"])
    hop_seconds = float(catalog.metadata["extractor"]["hop_seconds"])
    return next(
        iter_audio_windows(
            audio,
            extractor.sample_rate,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )
    )[2]


def evaluate_audio_pipeline(
    catalog: MusicCatalog,
    matcher: MusicMotionMatcher,
    gain_check_limit: int,
    feature_iterations: int,
) -> dict[str, Any]:
    extractor_metadata = catalog.metadata.get("extractor", {})
    embedding_metadata = extractor_metadata.get("embedding_model")
    tag_metadata = extractor_metadata.get("tag_model")
    embedding_model = resolve_model_path(
        catalog,
        embedding_metadata.get("path") if embedding_metadata else None,
    )
    tag_model = resolve_model_path(
        catalog,
        tag_metadata.get("path") if tag_metadata else None,
    )
    model_paths = [path for path in (embedding_model, tag_model) if path is not None]
    missing_models = sorted({relative_path(path) for path in model_paths if not path.is_file()})
    onnx_required = bool(model_paths)
    onnx_available = importlib.util.find_spec("onnxruntime") is not None
    status: dict[str, Any] = {
        "evaluated": False,
        "backend": extractor_metadata.get("embedding_backend", "unknown"),
        "onnxruntime_available": onnx_available,
        "model_files": sorted({relative_path(path) for path in model_paths}),
        "missing_model_files": missing_models,
    }
    if missing_models:
        status["skip_reason"] = "Catalog feature model file(s) are not present."
        return status
    if onnx_required and not onnx_available:
        status["skip_reason"] = "onnxruntime is not installed in the active environment."
        return status

    try:
        extractor = AudioFeatureExtractor(
            sample_rate=int(extractor_metadata["sample_rate"]),
            embedding_model=embedding_model,
            tag_model=tag_model,
        )
        music_ids = sorted(catalog.tracks)
        benchmark_window = first_audio_window(catalog, extractor, music_ids[0])
        extractor.describe(benchmark_window)
        latencies_ms: list[float] = []
        for _ in range(max(feature_iterations, 1)):
            started = time.perf_counter_ns()
            matcher.match(extractor.describe(benchmark_window))
            latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

        checked = 0
        unchanged = 0
        details: list[dict[str, Any]] = []
        for music_id in music_ids[: max(gain_check_limit, 0)]:
            window = first_audio_window(catalog, extractor, music_id)
            predictions: list[str | None] = []
            for gain in (0.1, 0.5, 1.0, 2.0):
                result = matcher.match(extractor.describe(gain * window))
                predictions.append(result.tracks[0].music_id if result.tracks else None)
            invariant = len(set(predictions)) == 1
            checked += 1
            unchanged += int(invariant)
            details.append(
                {
                    "music_id": music_id,
                    "top_1_predictions": predictions,
                    "invariant": invariant,
                }
            )
        status.update(
            {
                "evaluated": True,
                "waveform_to_match_latency": latency_summary(latencies_ms),
                "gain_invariance": {
                    "checked_tracks": checked,
                    "unchanged_tracks": unchanged,
                    "rate": unchanged / max(checked, 1),
                    "gains": [0.1, 0.5, 1.0, 2.0],
                    "details": details,
                },
            }
        )
    except (FileNotFoundError, KeyError, RuntimeError, StopIteration, ValueError) as error:
        status["skip_reason"] = f"Audio pipeline could not be evaluated: {error}"
        status["error_type"] = type(error).__name__
    return status


def markdown_report(report: dict[str, Any]) -> str:
    catalog = report["catalog"]
    quality = report["quality"]
    runtime = report["runtime"]
    audio = report["audio_pipeline"]
    latency = runtime["catalog_descriptor_match_latency"]

    lines = [
        "# Humanoid matcher performance result",
        "",
        f"Generated: `{report['generated_utc']}`",
        "",
        "## Summary",
        "",
        f"- Catalog: {catalog['segments']} segments, {catalog['tracks']} tracks, "
        f"{catalog['motions']} motions ({catalog['preflight_passed_motions']} preflight-passed).",
        f"- Core match latency: median **{latency['median_ms']:.3f} ms**, "
        f"p95 **{latency['p95_ms']:.3f} ms**, p99 **{latency['p99_ms']:.3f} ms**.",
        f"- Core throughput: **{latency['queries_per_second']:.1f} queries/s** "
        f"over {latency['samples']} timed calls.",
        f"- Leave-one-music genre Recall@1/3/5: "
        f"**{quality['leave_one_music_genre_recall_at_1']:.1%} / "
        f"{quality['leave_one_music_genre_recall_at_3']:.1%} / "
        f"{quality['leave_one_music_genre_recall_at_5']:.1%}**.",
        f"- Random genre baselines @1/3/5: "
        f"{quality['random_genre_recall_baseline_at_1']:.1%} / "
        f"{quality['random_genre_recall_baseline_at_3']:.1%} / "
        f"{quality['random_genre_recall_baseline_at_5']:.1%}.",
        f"- Source-track motion Recall@1/3/10: "
        f"**{quality['source_track_motion_recall_at_1']:.1%} / "
        f"{quality['source_track_motion_recall_at_3']:.1%} / "
        f"{quality['source_track_motion_recall_at_10']:.1%}**.",
        "",
        "The exact stored-segment track Recall@1 is a pipeline sanity check, not an "
        "out-of-sample accuracy estimate, because the queried descriptor is present in "
        "the catalog. Leave-one-music genre retrieval excludes every segment belonging "
        "to the query music ID and is the more informative quality measure.",
        "",
        "## Startup and memory",
        "",
        f"- Catalog load: {runtime['catalog_load_ms']:.3f} ms.",
        f"- Matcher initialization: {runtime['matcher_initialization_ms']:.3f} ms.",
        f"- Catalog feature arrays: {catalog['feature_array_bytes'] / 1024 / 1024:.2f} MiB.",
        f"- Matcher normalized indexes: {catalog['matcher_index_bytes'] / 1024 / 1024:.2f} MiB.",
    ]
    if runtime["rss_after_initialization_bytes"] is not None:
        lines.append(
            f"- Process RSS after initialization: "
            f"{runtime['rss_after_initialization_bytes'] / 1024 / 1024:.2f} MiB "
            f"(delta {runtime['rss_initialization_delta_bytes'] / 1024 / 1024:.2f} MiB)."
        )

    lines.extend(["", "## Per-genre leave-one-music recall", ""])
    lines.extend(
        [
            "| Genre | Queries | R@1 | R@3 | R@5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for genre, metrics in quality["per_genre"].items():
        lines.append(
            f"| {genre} | {metrics['queries']} | {metrics['recall_at_1']:.1%} | "
            f"{metrics['recall_at_3']:.1%} | {metrics['recall_at_5']:.1%} |"
        )

    lines.extend(["", "## Audio pipeline", ""])
    if audio["evaluated"]:
        end_to_end = audio["waveform_to_match_latency"]
        gain = audio["gain_invariance"]
        lines.extend(
            [
                f"- Waveform-to-match latency: median {end_to_end['median_ms']:.3f} ms, "
                f"p95 {end_to_end['p95_ms']:.3f} ms.",
                f"- Gain-invariant top-1: {gain['unchanged_tracks']}/{gain['checked_tracks']} "
                f"({gain['rate']:.1%}) for 0.1x/0.5x/1x/2x.",
            ]
        )
    else:
        lines.extend(
            [
                f"Skipped: {audio.get('skip_reason', 'unavailable')}",
                "",
                "This does not invalidate the core matcher timings above, but those "
                "timings exclude audio decoding and feature extraction.",
            ]
        )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "From the repository root:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe "
            "realtime/humanoid_robot/src/test/benchmark_humanoid_matcher.py",
            "```",
            "",
            f"Raw machine-readable results: `{report['result_files']['json']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    catalog_path = args.catalog.resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog does not exist: {catalog_path}")

    rss_before = process_rss_bytes()
    started = time.perf_counter_ns()
    catalog = MusicCatalog.load(catalog_path)
    catalog_load_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    started = time.perf_counter_ns()
    matcher = MusicMotionMatcher(catalog)
    matcher_initialization_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    rss_after = process_rss_bytes()

    quality, descriptors = evaluate_quality(catalog, matcher)
    match_latency = benchmark_match_latency(
        matcher,
        descriptors,
        args.warmup_iterations,
        args.latency_iterations,
    )
    audio_pipeline = evaluate_audio_pipeline(
        catalog,
        matcher,
        args.gain_check_limit,
        args.feature_iterations,
    )

    feature_arrays = (
        catalog.embeddings,
        catalog.rhythm_timbre,
        catalog.tags,
        catalog.embedding_mean,
        catalog.embedding_std,
        catalog.rhythm_mean,
        catalog.rhythm_std,
    )
    matcher_indexes = (
        matcher._catalog_embeddings,
        matcher._catalog_rhythm,
        matcher._catalog_tags,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.result_stem}.json"
    markdown_path = output_dir / f"{args.result_stem}.md"

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "librosa": package_version("librosa"),
            "onnxruntime": package_version("onnxruntime"),
        },
        "configuration": {
            "catalog": relative_path(catalog_path),
            "latency_iterations": max(args.latency_iterations, 1),
            "warmup_iterations": max(args.warmup_iterations, 0),
            "gain_check_limit": max(args.gain_check_limit, 0),
            "feature_iterations": max(args.feature_iterations, 1),
            "top_k_tracks": 5,
            "top_k_motions": 10,
        },
        "catalog": {
            "segments": len(catalog.segment_metadata),
            "tracks": len(catalog.tracks),
            "motions": len(catalog.motions),
            "preflight_passed_motions": sum(
                profile.preflight_passed for profile in catalog.motions.values()
            ),
            "embedding_dimensions": int(catalog.embeddings.shape[1]),
            "rhythm_timbre_dimensions": int(catalog.rhythm_timbre.shape[1]),
            "tag_dimensions": int(catalog.tags.shape[1]),
            "feature_array_bytes": sum(array.nbytes for array in feature_arrays),
            "matcher_index_bytes": sum(array.nbytes for array in matcher_indexes),
            "catalog_json_bytes": catalog_path.stat().st_size,
            "catalog_npz_bytes": (catalog_path.parent / catalog.metadata["arrays_file"]).stat().st_size,
        },
        "quality": quality,
        "runtime": {
            "catalog_load_ms": catalog_load_ms,
            "matcher_initialization_ms": matcher_initialization_ms,
            "catalog_descriptor_match_latency": match_latency,
            "rss_before_catalog_load_bytes": rss_before,
            "rss_after_initialization_bytes": rss_after,
            "rss_initialization_delta_bytes": (
                rss_after - rss_before
                if rss_after is not None and rss_before is not None
                else None
            ),
        },
        "audio_pipeline": audio_pipeline,
        "result_files": {
            "json": relative_path(json_path),
            "markdown": relative_path(markdown_path),
        },
    }

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown_report(report), encoding="utf-8")

    print(markdown_report(report))
    print(f"Wrote {relative_path(json_path)}")
    print(f"Wrote {relative_path(markdown_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
