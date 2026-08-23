"""Analyse whether AIST++ motion keypoint frequency follows local music BPM.

The camera token in an AIST++ video name (for example ``c01``) is mapped to
``cAll`` because the released SMPL motion and audio are camera-independent.

Run from the repository root::

    python realtime/humanoid_robot/src/test/analyze_aistpp_keypoint_bpm.py

The default report is written to ``src/test/output`` as JSON and CSV.  The
JSON distinguishes tempo-following evidence from simple beat synchronisation:
a nearly constant-tempo clip cannot support a meaningful correlation test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import pearsonr, spearmanr


TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent
DATA_DIR = SRC_DIR.parent / "data" / "aistpp"
DEFAULT_CLIP_ID = "gLH_sBM_c01_d17_mLH0_ch02"
DEFAULT_OUTPUT_DIR = TEST_DIR / "output"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aistpp_velocity_keypoints import detect_aistpp_file  # noqa: E402


@dataclass(frozen=True)
class TempoTrack:
    times: np.ndarray
    bpm: np.ndarray
    beat_times: np.ndarray
    global_bpm: float
    sample_rate: int


def canonical_clip_id(clip_id: str) -> str:
    """Map a camera-specific AIST++ name to its cAll motion/audio name."""
    name = Path(clip_id).stem
    tokens = name.split("_")
    if len(tokens) != 6:
        raise ValueError(f"Expected a 6-token AIST++ clip id, got: {clip_id}")
    if not re.fullmatch(r"c(?:All|\d+)", tokens[2]):
        raise ValueError(f"Invalid AIST++ camera token: {tokens[2]}")
    tokens[2] = "cAll"
    return "_".join(tokens)


def resolve_inputs(
    clip_id: str,
    motion: Path | None = None,
    audio: Path | None = None,
) -> tuple[str, Path, Path]:
    canonical = canonical_clip_id(clip_id)
    motion_path = motion or DATA_DIR / "motions" / f"{canonical}.pkl"
    audio_path = audio or DATA_DIR / "audio" / f"{canonical}.wav"
    missing = [str(path) for path in (motion_path, audio_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing analysis input(s): " + ", ".join(missing))
    return canonical, motion_path.resolve(), audio_path.resolve()


def extract_tempo_track(
    audio_path: Path,
    hop_length: int = 256,
    tempo_window_sec: float = 8.0,
) -> TempoTrack:
    """Return local tempogram BPM plus beat times for a music clip."""
    samples, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    onset = librosa.onset.onset_strength(
        y=samples,
        sr=sample_rate,
        hop_length=hop_length,
    )
    global_tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=sample_rate,
        hop_length=hop_length,
        units="frames",
    )
    global_bpm = float(np.asarray(global_tempo).reshape(-1)[0])
    local_bpm = librosa.feature.tempo(
        onset_envelope=onset,
        sr=sample_rate,
        hop_length=hop_length,
        ac_size=tempo_window_sec,
        aggregate=None,
        start_bpm=global_bpm,
    ).astype(float)
    # Remove isolated octave errors without erasing genuine gradual tempo change.
    if local_bpm.size >= 5:
        local_bpm = median_filter(local_bpm, size=5, mode="nearest")
    times = librosa.frames_to_time(
        np.arange(local_bpm.size), sr=sample_rate, hop_length=hop_length
    )
    beat_times = librosa.frames_to_time(
        beat_frames, sr=sample_rate, hop_length=hop_length
    ).astype(float)
    return TempoTrack(times, local_bpm, beat_times, global_bpm, int(sample_rate))


def nearest_beat_offsets(event_times: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    if event_times.size == 0 or beat_times.size == 0:
        return np.empty(0, dtype=float)
    positions = np.searchsorted(beat_times, event_times)
    right = np.clip(positions, 0, beat_times.size - 1)
    left = np.clip(positions - 1, 0, beat_times.size - 1)
    use_right = np.abs(beat_times[right] - event_times) < np.abs(
        beat_times[left] - event_times
    )
    nearest = np.where(use_right, beat_times[right], beat_times[left])
    return event_times - nearest


def safe_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Compute correlations only when both signals contain usable variation."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    result: dict[str, Any] = {"sample_count": int(x.size)}
    if x.size < 4:
        result.update(status="insufficient_samples", pearson_r=None, spearman_rho=None)
        return result
    x_mean = max(abs(float(np.mean(x))), 1e-12)
    y_mean = max(abs(float(np.mean(y))), 1e-12)
    x_cv = float(np.std(x) / x_mean)
    y_cv = float(np.std(y) / y_mean)
    result.update(x_cv=x_cv, y_cv=y_cv)
    if x_cv < 0.01 or y_cv < 0.01:
        result.update(
            status="insufficient_variation",
            pearson_r=None,
            pearson_p=None,
            spearman_rho=None,
            spearman_p=None,
        )
        return result
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    result.update(
        status="ok",
        pearson_r=float(pearson.statistic),
        pearson_p=float(pearson.pvalue),
        spearman_rho=float(spearman.statistic),
        spearman_p=float(spearman.pvalue),
    )
    return result


def analyse(
    requested_clip_id: str,
    canonical_id: str,
    motion_path: Path,
    audio_path: Path,
    fps: float = 60.0,
    smoothing_sec: float = 0.08,
    min_spacing_sec: float = 0.28,
    prominence: float = 0.08,
    hop_length: int = 256,
    tempo_window_sec: float = 8.0,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    keypoints = detect_aistpp_file(
        motion_path,
        fps=fps,
        smoothing_sec=smoothing_sec,
        min_spacing_sec=min_spacing_sec,
        prominence=prominence,
    )
    tempo = extract_tempo_track(audio_path, hop_length, tempo_window_sec)
    event_times = np.asarray(keypoints.times, dtype=float)
    if event_times.size < 2:
        raise ValueError("At least two motion keypoints are required for frequency analysis.")

    intervals = np.diff(event_times)
    interval_midpoints = (event_times[:-1] + event_times[1:]) * 0.5
    motion_frequency = 60.0 / intervals
    local_bpm = np.interp(interval_midpoints, tempo.times, tempo.bpm)
    ratios = motion_frequency / local_bpm
    correlation = safe_correlation(local_bpm, motion_frequency)

    tempo_mean = float(np.mean(local_bpm))
    tempo_std = float(np.std(local_bpm))
    tempo_cv = tempo_std / max(abs(tempo_mean), 1e-12)
    median_motion_frequency = float(np.median(motion_frequency))
    candidate_ratios = np.asarray([0.25, 0.5, 1.0, 2.0], dtype=float)
    median_ratio = float(np.median(ratios))
    best_ratio = float(candidate_ratios[np.argmin(np.abs(np.log(median_ratio / candidate_ratios)))])
    ratio_error = abs(median_ratio - best_ratio) / best_ratio

    offsets = nearest_beat_offsets(event_times, tempo.beat_times)
    abs_offsets = np.abs(offsets)
    median_beat_period = (
        float(np.median(np.diff(tempo.beat_times)))
        if tempo.beat_times.size >= 2
        else 60.0 / tempo.global_bpm
    )
    beat_interval_counts = intervals / median_beat_period

    if correlation["status"] == "ok":
        follows_tempo_change: bool | None = bool(
            correlation["pearson_r"] > 0.5 and correlation["pearson_p"] < 0.05
        )
        conclusion = (
            "局部动作频率与局部 BPM 存在显著正相关。"
            if follows_tempo_change
            else "检测到足够的 BPM 变化，但没有发现可靠的正相关。"
        )
    else:
        follows_tempo_change = None
        conclusion = "音乐局部 BPM 变化不足，无法从该片段识别动作是否会随实时 BPM 改变。"
    if ratio_error <= 0.10:
        conclusion += f" 不过动作 keypoint 频率约为音乐 BPM 的 {best_ratio:g} 倍，表现出节拍比例同步。"

    rows = [
        {
            "midpoint_sec": float(time),
            "keypoint_interval_sec": float(interval),
            "motion_frequency_per_min": float(frequency),
            "local_music_bpm": float(bpm),
            "frequency_to_bpm_ratio": float(ratio),
            "interval_in_beats": float(beats),
        }
        for time, interval, frequency, bpm, ratio, beats in zip(
            interval_midpoints,
            intervals,
            motion_frequency,
            local_bpm,
            ratios,
            beat_interval_counts,
        )
    ]
    report: dict[str, Any] = {
        "requested_clip_id": requested_clip_id,
        "canonical_clip_id": canonical_id,
        "camera_mapping": "camera-specific cNN -> camera-independent cAll",
        "motion_path": str(motion_path),
        "audio_path": str(audio_path),
        "duration_sec": float(keypoints.raw_velocity.size / keypoints.fps),
        "method": {
            "motion": "prominent valleys of whole-body SMPL angular + root translation velocity",
            "music": "librosa onset strength, beat tracking, and local tempogram tempo",
            "comparison": "instantaneous keypoint interval frequency sampled against local BPM",
            "fps": fps,
            "tempo_window_sec": tempo_window_sec,
        },
        "music": {
            "global_bpm": tempo.global_bpm,
            "local_bpm_mean": tempo_mean,
            "local_bpm_std": tempo_std,
            "local_bpm_cv": tempo_cv,
            "beat_count": int(tempo.beat_times.size),
        },
        "motion": {
            "keypoint_count": int(event_times.size),
            "keypoint_times_sec": event_times.tolist(),
            "frequency_median_per_min": median_motion_frequency,
            "frequency_mean_per_min": float(np.mean(motion_frequency)),
            "frequency_std_per_min": float(np.std(motion_frequency)),
            "signal_source": keypoints.signal_source,
        },
        "relationship": {
            "tempo_change_identifiable": correlation["status"] == "ok",
            "follows_realtime_bpm_change": follows_tempo_change,
            "correlation": correlation,
            "median_frequency_to_bpm_ratio": median_ratio,
            "nearest_simple_ratio": best_ratio,
            "simple_ratio_relative_error": ratio_error,
            "median_keypoint_interval_in_beats": float(np.median(beat_interval_counts)),
            "median_nearest_beat_offset_ms": float(np.median(abs_offsets) * 1000.0),
            "p90_nearest_beat_offset_ms": float(np.percentile(abs_offsets, 90.0) * 1000.0),
        },
        "conclusion_zh": conclusion,
    }
    return report, rows


def write_csv(path: Path, rows: Sequence[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare AIST++ keypoint frequency with the music's local BPM."
    )
    parser.add_argument("--clip-id", default=DEFAULT_CLIP_ID)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--smoothing-sec", type=float, default=0.08)
    parser.add_argument("--min-spacing-sec", type=float, default=0.28)
    parser.add_argument("--prominence", type=float, default=0.08)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--tempo-window-sec", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.hop_length <= 0 or args.tempo_window_sec <= 0:
        raise ValueError("fps, hop length, and tempo window must be positive")
    canonical, motion_path, audio_path = resolve_inputs(
        args.clip_id, args.motion, args.audio
    )
    report, rows = analyse(
        args.clip_id,
        canonical,
        motion_path,
        audio_path,
        fps=args.fps,
        smoothing_sec=args.smoothing_sec,
        min_spacing_sec=args.min_spacing_sec,
        prominence=args.prominence,
        hop_length=args.hop_length,
        tempo_window_sec=args.tempo_window_sec,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.clip_id}_keypoint_bpm_analysis.json"
    csv_path = output_dir / f"{args.clip_id}_keypoint_bpm_intervals.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(csv_path, rows)

    print(report["conclusion_zh"])
    print(
        f"Music: {report['music']['global_bpm']:.2f} BPM; "
        f"local CV={report['music']['local_bpm_cv']:.4f}."
    )
    print(
        f"Motion: {report['motion']['keypoint_count']} keypoints; median frequency="
        f"{report['motion']['frequency_median_per_min']:.2f}/min."
    )
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
