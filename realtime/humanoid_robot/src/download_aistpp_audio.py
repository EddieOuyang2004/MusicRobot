from __future__ import annotations

import argparse
import csv
import pickle
import subprocess
import threading
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg


DEFAULT_AISTPP_ROOT = Path(__file__).resolve().parents[1] / "data" / "aistpp"
VIDEO_URL = (
    "https://storage.repository.aist.go.jp/1095/v1.0.0/video/2M/{video_name}.mp4"
)
MOTION_FPS = 60.0
AUDIO_RATE = 44_100
AUDIO_CHANNELS = 2


@dataclass(frozen=True)
class AudioResult:
    motion_name: str
    video_name: str
    frames: int
    duration_seconds: float
    audio_path: Path
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the c01 AIST refined video corresponding to each AIST++ "
            "SMPL motion and extract a sequence-named, synchronized WAV file."
        )
    )
    parser.add_argument("--aistpp-root", type=Path, default=DEFAULT_AISTPP_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Keep newly downloaded 2 Mbps MP4 files in the cache.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N motions (useful for a smoke test).",
    )
    return parser.parse_args()


def motion_frame_count(motion_path: Path) -> int:
    with motion_path.open("rb") as handle:
        data = pickle.load(handle)
    poses = data.get("smpl_poses")
    if poses is None:
        raise KeyError(f"{motion_path} has no smpl_poses field")
    return len(poses)


def motion_to_video_name(motion_name: str) -> str:
    parts = motion_name.split("_")
    try:
        camera_index = parts.index("cAll")
    except ValueError as exc:
        raise ValueError(f"Unexpected AIST++ sequence name: {motion_name}") from exc
    parts[camera_index] = "c01"
    return "_".join(parts)


def wav_is_complete(path: Path, expected_seconds: float) -> bool:
    if not path.exists():
        return False
    try:
        with wave.open(str(path), "rb") as wav:
            actual_seconds = wav.getnframes() / wav.getframerate()
            return (
                wav.getframerate() == AUDIO_RATE
                and wav.getnchannels() == AUDIO_CHANNELS
                and abs(actual_seconds - expected_seconds) <= 1.0 / AUDIO_RATE
            )
    except (EOFError, wave.Error):
        return False


def download_video(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl.exe",
        "-L",
        "--fail",
        "--show-error",
        "--silent",
        "--retry",
        "10",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "15",
        "--max-time",
        "180",
        "--http1.1",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    subprocess.run(command, check=True)
    partial.replace(destination)


def extract_audio(
    ffmpeg: str, video_path: Path, output_path: Path, duration_seconds: float
) -> None:
    temporary = output_path.with_suffix(".wav.part")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(AUDIO_RATE),
        "-ac",
        str(AUDIO_CHANNELS),
        "-af",
        f"apad,atrim=duration={duration_seconds:.9f}",
        "-f",
        "wav",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(output_path)


def process_motion(
    motion_path: Path,
    videos_dir: Path,
    cache_dir: Path,
    audio_dir: Path,
    ffmpeg: str,
    keep_videos: bool,
) -> AudioResult:
    motion_name = motion_path.stem
    frames = motion_frame_count(motion_path)
    duration_seconds = frames / MOTION_FPS
    video_name = motion_to_video_name(motion_name)
    audio_path = audio_dir / f"{motion_name}.wav"

    if wav_is_complete(audio_path, duration_seconds):
        return AudioResult(
            motion_name,
            video_name,
            frames,
            duration_seconds,
            audio_path,
            "existing",
        )

    existing_video = videos_dir / f"{video_name}.mp4"
    cached_video = cache_dir / f"{video_name}.mp4"
    downloaded = False
    if existing_video.exists():
        video_path = existing_video
    else:
        video_path = cached_video
        if not video_path.exists():
            download_video(VIDEO_URL.format(video_name=video_name), video_path)
            downloaded = True

    try:
        extract_audio(ffmpeg, video_path, audio_path, duration_seconds)
        if not wav_is_complete(audio_path, duration_seconds):
            raise RuntimeError(f"Extracted WAV failed duration validation: {audio_path}")
    finally:
        if downloaded and not keep_videos and cached_video.exists():
            cached_video.unlink()

    return AudioResult(
        motion_name,
        video_name,
        frames,
        duration_seconds,
        audio_path,
        "downloaded" if downloaded else "extracted",
    )


def write_manifest(results: list[AudioResult], manifest_path: Path) -> None:
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "motion_name",
                "video_name",
                "motion_frames",
                "duration_seconds",
                "audio_path",
                "status",
            ]
        )
        for result in sorted(results, key=lambda item: item.motion_name):
            writer.writerow(
                [
                    result.motion_name,
                    result.video_name,
                    result.frames,
                    f"{result.duration_seconds:.9f}",
                    result.audio_path.name,
                    result.status,
                ]
            )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    root = args.aistpp_root.resolve()
    motions_dir = root / "motions"
    videos_dir = root / "videos"
    audio_dir = root / "audio"
    cache_dir = root / "_audio_video_cache"
    if not motions_dir.is_dir():
        raise FileNotFoundError(f"AIST++ motions directory not found: {motions_dir}")

    motion_paths = sorted(motions_dir.glob("*.pkl"))
    if args.limit is not None:
        motion_paths = motion_paths[: args.limit]
    if not motion_paths:
        raise RuntimeError(f"No motion PKL files found under {motions_dir}")

    audio_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    results: list[AudioResult] = []
    failures: list[tuple[str, str]] = []
    print_lock = threading.Lock()

    print(
        f"Processing {len(motion_paths)} motions with {args.workers} workers; "
        f"audio output: {audio_dir}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_motion,
                motion_path,
                videos_dir,
                cache_dir,
                audio_dir,
                ffmpeg,
                args.keep_videos,
            ): motion_path
            for motion_path in motion_paths
        }
        for index, future in enumerate(as_completed(futures), start=1):
            motion_path = futures[future]
            try:
                result = future.result()
                results.append(result)
                message = f"[{index}/{len(futures)}] {result.status}: {result.motion_name}"
            except Exception as exc:
                failures.append((motion_path.stem, str(exc)))
                message = f"[{index}/{len(futures)}] FAILED: {motion_path.stem}: {exc}"
            with print_lock:
                print(message, flush=True)

    manifest_path = audio_dir / "manifest.csv"
    write_manifest(results, manifest_path)
    if not args.keep_videos and cache_dir.exists() and not any(cache_dir.iterdir()):
        cache_dir.rmdir()

    print(
        f"Completed: {len(results)}; failed: {len(failures)}; "
        f"manifest: {manifest_path}",
        flush=True,
    )
    if failures:
        failure_path = audio_dir / "failures.csv"
        with failure_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["motion_name", "error"])
            writer.writerows(failures)
        print(f"Failure report: {failure_path}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
