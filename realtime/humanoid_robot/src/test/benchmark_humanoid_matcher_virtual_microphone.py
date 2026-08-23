"""End-to-end benchmark using original WAV files and the virtual microphone.

This exercises the production ``MatcherFileMicrophoneSource`` block-feeding
path, realtime analyzer, Discogs EffNet ONNX extractor, music matcher, and
motion ranking.  One representative original AIST++ WAV is selected per genre.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent
HUMANOID_DIR = SRC_DIR.parent
ROOT = HUMANOID_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import realtime_music_humanoid_matcher as matcher_app  # noqa: E402
from music_motion_catalog import (  # noqa: E402
    AudioFeatureExtractor,
    MusicCatalog,
    MusicMotionMatcher,
)


DEFAULT_CATALOG = HUMANOID_DIR / "data" / "music_catalog" / "catalog.json"
DEFAULT_MODEL = HUMANOID_DIR / "models" / "discogs-effnet-bsdynamic-1.onnx"
DEFAULT_OUTPUT_DIR = TEST_DIR / "output"
RESULT_STEM = "humanoid_matcher_virtual_microphone"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the humanoid matcher with realtime virtual-microphone WAV input."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seconds-per-file",
        type=float,
        default=10.0,
        help="Original-audio seconds played for each selected WAV (default: 10).",
    )
    parser.add_argument(
        "--genre-limit",
        type=int,
        default=0,
        help="Limit the number of genres; zero tests every genre.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(values, quantile)) if values else 0.0


def latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "median_ms": statistics.median(values) if values else 0.0,
        "p95_ms": percentile(values, 95.0),
        "p99_ms": percentile(values, 99.0),
        "min_ms": min(values, default=0.0),
        "max_ms": max(values, default=0.0),
    }


def production_defaults() -> argparse.Namespace:
    original_argv = sys.argv
    try:
        # Use the real parsers so the source/analyzer configuration stays aligned
        # with the production entrypoint.  realtime=False makes the file source
        # itself throttle block delivery to wall-clock time in this headless test.
        sys.argv = [original_argv[0], "--headless"]
        args = matcher_app.parse_args()
    finally:
        sys.argv = original_argv
    args.realtime = False
    return args


def select_wavs(catalog: MusicCatalog, genre_limit: int) -> list[dict[str, str]]:
    aistpp_root = (
        catalog.catalog_path.parent / str(catalog.metadata["aistpp_root"])
    ).resolve()
    selected: list[dict[str, str]] = []
    seen_genres: set[str] = set()
    for music_id, track in sorted(catalog.tracks.items()):
        genre = str(track["genre"])
        if genre in seen_genres:
            continue
        variant = track["variants"][0]
        path = (aistpp_root / str(variant["source_audio"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Original catalog WAV is missing: {path}")
        selected.append(
            {
                "music_id": str(music_id),
                "genre": genre,
                "path": str(path),
            }
        )
        seen_genres.add(genre)
        if genre_limit > 0 and len(selected) >= genre_limit:
            break
    return selected


def wait_for_result(
    worker: matcher_app.RetrievalWorker,
    pending: dict[str, Any],
) -> tuple[Any, float]:
    while True:
        result = worker.poll()
        if result is not None:
            return result, (time.perf_counter() - pending["wall_started"]) * 1_000.0
        time.sleep(0.005)


def run_one_file(
    item: dict[str, str],
    runtime_args: argparse.Namespace,
    worker: matcher_app.RetrievalWorker,
    seconds_per_file: float,
) -> dict[str, Any]:
    setup_started = time.perf_counter()
    source = matcher_app.MatcherFileMicrophoneSource(
        Path(item["path"]),
        runtime_args,
        runtime_args.match_window_seconds,
    )
    setup_ms = (time.perf_counter() - setup_started) * 1_000.0
    source.start()
    loop_started = time.perf_counter()
    step_seconds = source.block_size / source.sample_rate
    last_match_clock = -float("inf")
    pending: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    submit_attempts = 0
    busy_attempts = 0
    music_frames = 0
    beat_frames = 0

    try:
        while not source.done and source.playback_seconds < seconds_per_file:
            source.advance(step_seconds)
            # Match the production main loop: --max-seconds is checked directly
            # after advancing the virtual microphone and before draining/submitting.
            if source.playback_seconds >= seconds_per_file:
                break
            frames = source.drain()
            music_frames += len(frames)
            beat_frames += sum(frame.is_beat for frame in frames)

            match_clock = source.playback_seconds
            if match_clock - last_match_clock >= runtime_args.match_interval_seconds:
                audio = source.recent_audio()
                if audio is not None:
                    submit_attempts += 1
                    submitted = worker.submit(audio)
                    if submitted:
                        last_match_clock = match_clock
                        pending = {
                            "audio_time": match_clock,
                            "wall_started": time.perf_counter(),
                        }
                    else:
                        busy_attempts += 1

            result = worker.poll()
            if result is not None:
                assert pending is not None
                latency_ms = (time.perf_counter() - pending["wall_started"]) * 1_000.0
                results.append(
                    {
                        "query_audio_time_seconds": pending["audio_time"],
                        "latency_ms": latency_ms,
                        "top_tracks": [track.music_id for track in result.tracks],
                        "top_track_genres": [track.genre for track in result.tracks],
                        "top_motion_id": result.motions[0].motion_id if result.motions else None,
                        "query_bpm": result.query_bpm,
                    }
                )
                pending = None

        playback_wall_seconds = time.perf_counter() - loop_started
        if pending is not None:
            result, latency_ms = wait_for_result(worker, pending)
            results.append(
                {
                    "query_audio_time_seconds": pending["audio_time"],
                    "latency_ms": latency_ms,
                    "top_tracks": [track.music_id for track in result.tracks],
                    "top_track_genres": [track.genre for track in result.tracks],
                    "top_motion_id": result.motions[0].motion_id if result.motions else None,
                    "query_bpm": result.query_bpm,
                }
            )
    finally:
        source.stop()

    total_wall_seconds = time.perf_counter() - loop_started
    expected_music = item["music_id"]
    expected_genre = item["genre"]
    active_audio_wall_seconds = max(
        playback_wall_seconds - source.startup_delay_sec,
        1e-9,
    )
    schedulable_seconds = max(
        seconds_per_file - runtime_args.match_window_seconds,
        0.0,
    )
    scheduled_slots = (
        int(
            np.ceil(
                schedulable_seconds / runtime_args.match_interval_seconds
                - 1e-9
            )
        )
        if seconds_per_file > runtime_args.match_window_seconds
        else 0
    )
    return {
        "music_id": expected_music,
        "genre": expected_genre,
        "wav": relative_path(Path(item["path"])),
        "source_setup_ms": setup_ms,
        "audio_played_seconds": source.playback_seconds,
        "playback_loop_wall_seconds": playback_wall_seconds,
        "total_wall_seconds_including_final_result": total_wall_seconds,
        "realtime_factor": source.playback_seconds
        / active_audio_wall_seconds,
        "stream_realtime_factor_including_startup": (
            source.playback_seconds + source.startup_delay_sec
        )
        / max(playback_wall_seconds, 1e-9),
        "music_frames": music_frames,
        "beat_frames": beat_frames,
        "submit_attempts": submit_attempts,
        "busy_submit_attempts": busy_attempts,
        "scheduled_match_slots": scheduled_slots,
        "completed_matches": len(results),
        "top_1_track_hits": sum(
            bool(result["top_tracks"] and result["top_tracks"][0] == expected_music)
            for result in results
        ),
        "top_3_track_hits": sum(
            expected_music in result["top_tracks"][:3] for result in results
        ),
        "top_1_genre_hits": sum(
            bool(result["top_track_genres"] and result["top_track_genres"][0] == expected_genre)
            for result in results
        ),
        "top_3_genre_hits": sum(
            expected_genre in result["top_track_genres"][:3] for result in results
        ),
        "matches": results,
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    latency = summary["waveform_to_match_latency"]
    lines = [
        "# Humanoid matcher virtual-microphone result",
        "",
        f"Generated: `{report['generated_utc']}`",
        "",
        "## Scope",
        "",
        "Original AIST++ WAV files were delivered block-by-block through the production "
        "`MatcherFileMicrophoneSource` at wall-clock speed. Each completed query includes "
        "the rolling audio window, EffNet ONNX inference, track retrieval, and motion ranking.",
        "The WAVs belong to the catalog, so exact-track accuracy checks reconstruction and "
        "realtime plumbing; it is not an unseen-music generalization score.",
        "",
        "## Summary",
        "",
        f"- Files/genres: {summary['files']} / {summary['genres']}.",
        f"- Completed matches: {summary['completed_matches']}.",
        f"- Waveform-to-match latency: median **{latency['median_ms']:.1f} ms**, "
        f"p95 **{latency['p95_ms']:.1f} ms**, max **{latency['max_ms']:.1f} ms**.",
        f"- First (cold) match: **{summary['cold_match_latency_ms']:.1f} ms**; "
        f"steady-state median: **{summary['steady_state_latency']['median_ms']:.1f} ms**.",
        f"- WAV source setup/beat analysis: median "
        f"**{summary['source_setup_latency']['median_ms']:.1f} ms**, max "
        f"**{summary['source_setup_latency']['max_ms']:.1f} ms**.",
        f"- Results within the {summary['match_interval_ms']:.0f} ms match interval: "
        f"**{summary['within_interval_rate']:.1%}**.",
        f"- Exact music Top-1/Top-3: **{summary['track_top_1_rate']:.1%} / "
        f"{summary['track_top_3_rate']:.1%}**.",
        f"- Genre Top-1/Top-3: **{summary['genre_top_1_rate']:.1%} / "
        f"{summary['genre_top_3_rate']:.1%}**.",
        f"- Mean realtime factor: **{summary['mean_realtime_factor']:.3f}x** "
        "(1.0x means wall-clock playback).",
        f"- Busy submission attempts: {summary['busy_submit_attempts']} / "
        f"{summary['submit_attempts']} ({summary['busy_submit_rate']:.1%}).",
        f"- Completed scheduled match slots: {summary['completed_matches']} / "
        f"{summary['scheduled_match_slots']} ({summary['scheduled_slot_completion_rate']:.1%}).",
        "",
        "## Per-file results",
        "",
        "| Genre | Music | Matches | Track Top-1 | Genre Top-1 | Median latency | Realtime |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["files"]:
        completed = max(item["completed_matches"], 1)
        item_latencies = [match["latency_ms"] for match in item["matches"]]
        lines.append(
            f"| {item['genre']} | {item['music_id']} | {item['completed_matches']} | "
            f"{item['top_1_track_hits'] / completed:.1%} | "
            f"{item['top_1_genre_hits'] / completed:.1%} | "
            f"{statistics.median(item_latencies) if item_latencies else 0.0:.1f} ms | "
            f"{item['realtime_factor']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe "
            "realtime/humanoid_robot/src/test/benchmark_humanoid_matcher_virtual_microphone.py",
            "```",
            "",
            f"Raw results: `{report['result_files']['json']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    catalog_path = args.catalog.resolve()
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model is missing: {model_path}")

    catalog = MusicCatalog.load(catalog_path)
    catalog_model = catalog.metadata["extractor"]["embedding_model"]
    actual_sha256 = sha256(model_path)
    expected_sha256 = str(catalog_model["sha256"])
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Model SHA-256 does not match catalog: {actual_sha256} != {expected_sha256}"
        )

    runtime_args = production_defaults()
    extractor = AudioFeatureExtractor(
        sample_rate=int(catalog.metadata["extractor"]["sample_rate"]),
        embedding_model=model_path,
        tag_model=model_path,
    )
    matcher = MusicMotionMatcher(
        catalog,
        speed_min=runtime_args.speed_min,
        speed_max=runtime_args.speed_max,
    )
    worker = matcher_app.RetrievalWorker(
        extractor,
        matcher,
        runtime_args.match_top_tracks,
        runtime_args.match_top_motions,
    )

    selected = select_wavs(catalog, max(args.genre_limit, 0))
    file_results: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(selected, start=1):
            print(
                f"[{index}/{len(selected)}] realtime WAV: "
                f"{item['genre']} {item['music_id']}"
            )
            file_results.append(
                run_one_file(
                    item,
                    runtime_args,
                    worker,
                    max(args.seconds_per_file, runtime_args.match_window_seconds),
                )
            )
    finally:
        worker.close()

    matches = [match for item in file_results for match in item["matches"]]
    latencies = [match["latency_ms"] for match in matches]
    completed = len(matches)
    submit_attempts = sum(item["submit_attempts"] for item in file_results)
    busy_attempts = sum(item["busy_submit_attempts"] for item in file_results)
    interval_ms = runtime_args.match_interval_seconds * 1_000.0
    scheduled_slots = sum(item["scheduled_match_slots"] for item in file_results)
    summary = {
        "files": len(file_results),
        "genres": len({item["genre"] for item in file_results}),
        "completed_matches": completed,
        "waveform_to_match_latency": latency_summary(latencies),
        "cold_match_latency_ms": latencies[0] if latencies else 0.0,
        "steady_state_latency": latency_summary(latencies[1:]),
        "source_setup_latency": latency_summary(
            [item["source_setup_ms"] for item in file_results]
        ),
        "match_interval_ms": interval_ms,
        "within_interval_rate": sum(value <= interval_ms for value in latencies)
        / max(completed, 1),
        "track_top_1_rate": sum(item["top_1_track_hits"] for item in file_results)
        / max(completed, 1),
        "track_top_3_rate": sum(item["top_3_track_hits"] for item in file_results)
        / max(completed, 1),
        "genre_top_1_rate": sum(item["top_1_genre_hits"] for item in file_results)
        / max(completed, 1),
        "genre_top_3_rate": sum(item["top_3_genre_hits"] for item in file_results)
        / max(completed, 1),
        "mean_realtime_factor": statistics.fmean(
            item["realtime_factor"] for item in file_results
        ),
        "submit_attempts": submit_attempts,
        "busy_submit_attempts": busy_attempts,
        "busy_submit_rate": busy_attempts / max(submit_attempts, 1),
        "scheduled_match_slots": scheduled_slots,
        "scheduled_slot_completion_rate": completed / max(scheduled_slots, 1),
        "music_frames": sum(item["music_frames"] for item in file_results),
        "beat_frames": sum(item["beat_frames"] for item in file_results),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{RESULT_STEM}.json"
    markdown_path = output_dir / f"{RESULT_STEM}.md"
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "onnxruntime_providers": extractor.embedding_backend.session.get_providers(),
        },
        "configuration": {
            "catalog": relative_path(catalog_path),
            "model": relative_path(model_path),
            "model_sha256": actual_sha256,
            "seconds_per_file": max(
                args.seconds_per_file,
                runtime_args.match_window_seconds,
            ),
            "match_window_seconds": runtime_args.match_window_seconds,
            "match_interval_seconds": runtime_args.match_interval_seconds,
            "microphone_sample_rate": runtime_args.mic_sample_rate,
            "microphone_block_size": runtime_args.mic_block_size,
            "startup_delay_seconds": runtime_args.audio_input_delay_sec,
            "top_tracks": runtime_args.match_top_tracks,
            "top_motions": runtime_args.match_top_motions,
        },
        "summary": summary,
        "files": file_results,
        "result_files": {
            "json": relative_path(json_path),
            "markdown": relative_path(markdown_path),
        },
    }
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rendered = markdown_report(report)
    markdown_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"Wrote {relative_path(json_path)}")
    print(f"Wrote {relative_path(markdown_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
