"""Replay synchronized music through the realtime microphone path and measure phase error.

This tool answers a different question from ``analyze_aistpp_keypoint_bpm.py``:
instead of comparing event frequencies, it runs the production
``RealtimeMusicAnalyzer`` and ``AdaptiveMotionController`` offline, then compares
the controller's driven motion phase with the original authored phase over the
whole motion.

Run from the repository root::

    python realtime/humanoid_robot/src/test/analyze_motion_audio_phase.py \
      --motion realtime/humanoid_robot/data/aistpp/motions/gWA_sBM_cAll_d26_mWA0_ch07.pkl \
      --audio realtime/humanoid_robot/data/aistpp/audio/gWA_sBM_cAll_d26_mWA0_ch07.wav

If ``--audio`` is omitted, the matching AIST++ audio path is inferred from the
motion filename. JSON contains aggregate metrics and accepted-beat diagnostics;
CSV contains the time series used to calculate them.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

import librosa
import numpy as np


TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent
HUMANOID_DIR = SRC_DIR.parent
ROOT = HUMANOID_DIR.parents[1]
ROBOT_ARM_SRC = ROOT / "realtime" / "robot_arm" / "src"
DEFAULT_MOTION = (
    HUMANOID_DIR / "data" / "aistpp" / "motions" / "gWA_sBM_cAll_d26_mWA0_ch07.pkl"
)
DEFAULT_OUTPUT_DIR = TEST_DIR / "output"

for import_dir in (SRC_DIR, ROBOT_ARM_SRC):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import realtime_music_adaptive_player as music_runtime  # noqa: E402
from aistpp_velocity_keypoints import detect_aistpp_file  # noqa: E402
from motion_keypoints import default_keypoint_count  # noqa: E402
from realtime_music_adaptive_player import (  # noqa: E402
    AdaptiveMotionController,
    RealtimeMusicAnalyzer,
)


@dataclass(frozen=True)
class PhaseAnalysisResult:
    report: dict[str, Any]
    samples: list[dict[str, Any]]


def wrapped_phase_error(actual: float | np.ndarray, reference: float | np.ndarray) -> Any:
    """Return signed circular error in cycles, in the interval [-0.5, 0.5)."""
    error = (np.asarray(actual) - np.asarray(reference) + 0.5) % 1.0 - 0.5
    if error.ndim == 0:
        return float(error)
    return error


def circular_mean_cycles(values: Sequence[float] | np.ndarray) -> float:
    values_array = np.asarray(values, dtype=float)
    if values_array.size == 0:
        return 0.0
    vector = np.mean(np.exp(2j * np.pi * values_array))
    if abs(vector) <= 1e-12:
        return 0.0
    return float(np.angle(vector) / (2.0 * np.pi))


def phase_error_metrics(errors: Sequence[float] | np.ndarray, motion_duration: float) -> dict[str, Any]:
    values = np.asarray(errors, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"sample_count": 0, "status": "no_samples"}

    absolute = np.abs(values)
    bias = circular_mean_cycles(values)
    corrected = wrapped_phase_error(values, bias)
    phase_lock_value = abs(np.mean(np.exp(2j * np.pi * values)))
    return {
        "status": "ok",
        "sample_count": int(values.size),
        "mean_absolute_error_cycles": float(np.mean(absolute)),
        "root_mean_square_error_cycles": float(np.sqrt(np.mean(values * values))),
        "p95_absolute_error_cycles": float(np.percentile(absolute, 95.0)),
        "maximum_absolute_error_cycles": float(np.max(absolute)),
        "mean_absolute_error_degrees": float(np.mean(absolute) * 360.0),
        "p95_absolute_error_degrees": float(np.percentile(absolute, 95.0) * 360.0),
        "mean_equivalent_motion_time_error_sec": float(np.mean(absolute) * motion_duration),
        "p95_equivalent_motion_time_error_sec": float(
            np.percentile(absolute, 95.0) * motion_duration
        ),
        "circular_bias_cycles": bias,
        "circular_bias_degrees": bias * 360.0,
        "bias_corrected_mean_absolute_error_cycles": float(np.mean(np.abs(corrected))),
        "bias_corrected_mean_absolute_error_degrees": float(
            np.mean(np.abs(corrected)) * 360.0
        ),
        "phase_lock_value": float(phase_lock_value),
        "fraction_within_5_percent_cycle": float(np.mean(absolute <= 0.05)),
        "fraction_within_10_percent_cycle": float(np.mean(absolute <= 0.10)),
    }


def infer_audio_path(motion_path: Path) -> Path:
    if motion_path.parent.name == "motions":
        candidate = motion_path.parent.parent / "audio" / f"{motion_path.stem}.wav"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not infer synchronized audio. Pass --audio explicitly; expected an AIST++ "
        f"audio file matching {motion_path.name}."
    )


def make_analyzer(args: argparse.Namespace) -> RealtimeMusicAnalyzer:
    return RealtimeMusicAnalyzer(
        sample_rate=args.mic_sample_rate,
        block_size=args.mic_block_size,
        onset_threshold_scale=args.onset_threshold_scale,
        min_beat_period=args.min_beat_period,
        max_beat_period=args.max_beat_period,
        refractory_sec=args.mic_refractory_sec,
        noise_gate_rms=args.noise_gate_rms,
        noise_gate_ratio=args.noise_gate_ratio,
        startup_calibration_sec=args.startup_calibration_sec,
        plp_history_sec=args.plp_history_sec,
        plp_analysis_interval_sec=args.plp_analysis_interval_sec,
        plp_hop_length=args.plp_hop_length,
        plp_peak_prominence=args.plp_peak_prominence,
    )


def _feed_analyzer_block(
    analyzer: RealtimeMusicAnalyzer,
    stream: np.ndarray,
    cursor: int,
    base_time: float,
) -> int:
    end = min(cursor + analyzer.block_size, stream.size)
    block = np.asarray(stream[cursor:end], dtype=np.float32)
    actual_size = block.size
    if actual_size < analyzer.block_size:
        block = np.pad(block, (0, analyzer.block_size - actual_size))
    callback_time = base_time + end / analyzer.sample_rate
    analyzer._callback(
        np.asarray(block[:, None], dtype=np.float32),
        actual_size,
        {"callback_time": callback_time},
        None,
    )
    return end


def _conclusion_zh(metrics: dict[str, Any]) -> str:
    if metrics.get("status") != "ok":
        return "没有足够的有效采样点来评估相位差。"
    mae = float(metrics["mean_absolute_error_cycles"])
    corrected = float(metrics["bias_corrected_mean_absolute_error_cycles"])
    bias = abs(float(metrics["circular_bias_cycles"]))
    if mae <= 0.05:
        quality = "整体相位跟踪良好"
    elif mae <= 0.10:
        quality = "整体相位存在中等误差"
    else:
        quality = "整体相位误差较大"
    if bias >= 0.05 and corrected <= 0.6 * mae:
        diagnosis = "；主要问题更像是固定的起始相位偏置，而不是持续漂移"
    elif corrected >= 0.8 * mae:
        diagnosis = "；去除固定偏置后改善有限，误差主要来自运行中的跟踪或漂移"
    else:
        diagnosis = "；误差同时包含起始偏置和运行中漂移"
    return quality + diagnosis + "。"


def run_analysis(args: argparse.Namespace) -> PhaseAnalysisResult:
    motion_path = args.motion.resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    audio_path = (args.audio.resolve() if args.audio is not None else infer_audio_path(motion_path))
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    max_keypoints = args.keypoint_max_count
    keypoints_unlimited = detect_aistpp_file(
        motion_path,
        fps=args.motion_fps,
        smoothing_sec=args.keypoint_smoothing_sec,
        min_spacing_sec=args.keypoint_min_spacing_sec,
        prominence=args.keypoint_prominence,
        boundary_sec=args.keypoint_boundary_sec,
    )
    motion_duration = float(keypoints_unlimited.raw_velocity.size / keypoints_unlimited.fps)
    if max_keypoints is None:
        max_keypoints = default_keypoint_count(motion_duration)
    keypoints = detect_aistpp_file(
        motion_path,
        fps=args.motion_fps,
        smoothing_sec=args.keypoint_smoothing_sec,
        min_spacing_sec=args.keypoint_min_spacing_sec,
        prominence=args.keypoint_prominence,
        max_count=max_keypoints,
        boundary_sec=args.keypoint_boundary_sec,
    )
    if len(keypoints.phases) < 2:
        raise ValueError(
            "Fewer than two motion keypoints were detected; lower --keypoint-prominence "
            "or inspect the motion file."
        )

    audio, _ = librosa.load(audio_path, sr=args.mic_sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio_duration = float(audio.size / args.mic_sample_rate)
    analysis_duration = min(motion_duration, audio_duration)
    if args.max_seconds is not None:
        analysis_duration = min(analysis_duration, args.max_seconds)
    if analysis_duration <= 0.0:
        raise ValueError("Motion/audio overlap duration must be positive.")

    analyzer = make_analyzer(args)
    controller = AdaptiveMotionController(
        authored_cycle_duration=motion_duration,
        beats_per_cycle=len(keypoints.phases),
        keypoint_phases=keypoints.phases,
        use_keypoints=True,
        smoothing_tau=args.smoothing_tau,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        amp_min=args.amp_min,
        amp_max=args.amp_max,
        accent_duration=args.accent_duration,
        tempo_timeout=args.tempo_timeout,
        beat_confidence_threshold=args.beat_confidence_threshold,
        beat_keypoint_interval_ratio=args.beat_keypoint_interval_ratio,
        beat_selection_mode=args.beat_selection_mode,
        beat_contrast_weight=args.beat_contrast_weight,
    )

    startup_samples = int(round(args.audio_input_delay_sec * args.mic_sample_rate))
    audio_samples = int(math.ceil(analysis_duration * args.mic_sample_rate))
    stream = np.concatenate((np.zeros(startup_samples, dtype=np.float32), audio[:audio_samples]))
    base_time = 1000.0
    control_dt = 1.0 / args.analysis_fps
    total_elapsed = args.audio_input_delay_sec + analysis_duration
    cursor = 0
    detected_beats = 0
    accepted_beats = 0
    first_accepted_audio_time: float | None = None
    accepted_events: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    with patch.object(music_runtime.time, "perf_counter", return_value=base_time):
        analyzer.reset()
    controller.update(base_time)

    step_count = int(math.ceil(total_elapsed * args.analysis_fps))
    for step_index in range(1, step_count + 1):
        elapsed = min(step_index * control_dt, total_elapsed)
        target_cursor = min(int(math.floor(elapsed * args.mic_sample_rate)), stream.size)
        while cursor + analyzer.block_size <= target_cursor:
            cursor = _feed_analyzer_block(analyzer, stream, cursor, base_time)

        now = base_time + elapsed
        with patch.object(music_runtime.time, "perf_counter", return_value=now):
            frames = analyzer.drain()

        accepted_this_step = False
        for frame in frames:
            if frame.is_beat:
                detected_beats += 1
                target_phase = keypoints.phases[controller.beat_index % len(keypoints.phases)]
                beat_audio_time = frame.timestamp - base_time - args.audio_input_delay_sec
                was_accepted = controller.observe(frame)
                if was_accepted:
                    accepted_this_step = True
                    accepted_beats += 1
                    if first_accepted_audio_time is None:
                        first_accepted_audio_time = max(beat_audio_time, 0.0)
                    original_phase = (beat_audio_time / motion_duration) % 1.0
                    accepted_events.append(
                        {
                            "beat_audio_time_sec": float(beat_audio_time),
                            "target_keypoint_phase": float(target_phase),
                            "original_phase_at_beat": float(original_phase),
                            "driven_phase_after_alignment": float(controller.phase),
                            "original_to_target_error_cycles": wrapped_phase_error(
                                original_phase, target_phase
                            ),
                            "driven_to_target_error_cycles": wrapped_phase_error(
                                controller.phase, target_phase
                            ),
                            "beat_confidence": float(frame.beat_confidence),
                            "beat_contrast": float(frame.beat_contrast),
                            "estimated_beat_period_sec": (
                                None if frame.beat_period is None else float(frame.beat_period)
                            ),
                        }
                    )
            else:
                controller.observe(frame)

        driven_phase, _amplitude, _accent, _brightness = controller.update(now)
        audio_time = elapsed - args.audio_input_delay_sec
        if 0.0 <= audio_time <= analysis_duration:
            original_phase = (audio_time / motion_duration) % 1.0
            error = wrapped_phase_error(driven_phase, original_phase)
            samples.append(
                {
                    "time_sec": float(audio_time),
                    "original_phase": float(original_phase),
                    "driven_phase": float(driven_phase),
                    "signed_phase_error_cycles": float(error),
                    "absolute_phase_error_degrees": float(abs(error) * 360.0),
                    "equivalent_motion_time_error_sec": float(abs(error) * motion_duration),
                    "phase_rate_cycles_per_sec": float(controller.phase_rate),
                    "speed_multiplier": float(controller.speed_multiplier),
                    "estimated_bpm": (
                        None if controller.estimated_bpm is None else float(controller.estimated_bpm)
                    ),
                    "music_active": bool(controller.music_active),
                    "detected_beats": detected_beats,
                    "accepted_beats": accepted_beats,
                    "beat_accepted_this_step": accepted_this_step,
                }
            )

    all_errors = np.asarray([row["signed_phase_error_cycles"] for row in samples])
    after_first_errors = np.asarray(
        [
            row["signed_phase_error_cycles"]
            for row in samples
            if first_accepted_audio_time is not None and row["time_sec"] >= first_accepted_audio_time
        ]
    )
    full_metrics = phase_error_metrics(all_errors, motion_duration)
    tracking_metrics = phase_error_metrics(after_first_errors, motion_duration)
    primary_metrics = tracking_metrics if tracking_metrics.get("status") == "ok" else full_metrics
    report: dict[str, Any] = {
        "motion_path": str(motion_path),
        "audio_path": str(audio_path),
        "motion_duration_sec": motion_duration,
        "audio_duration_sec": audio_duration,
        "analyzed_duration_sec": analysis_duration,
        "motion_coverage_fraction": float(analysis_duration / motion_duration),
        "reference_phase_definition": "original_phase = audio_time / authored_motion_duration",
        "phase_error_definition": "wrap(driven_phase - original_phase) into [-0.5, 0.5) cycles",
        "motion_keypoints": {
            "count": len(keypoints.phases),
            "phases": list(keypoints.phases),
            "times_sec": list(keypoints.times),
            "frame_indices": list(keypoints.frame_indices),
            "signal_source": keypoints.signal_source,
        },
        "microphone_and_controller": {
            "sample_rate": args.mic_sample_rate,
            "block_size": args.mic_block_size,
            "analysis_fps": args.analysis_fps,
            "startup_silence_sec": args.audio_input_delay_sec,
            "detected_beat_count": detected_beats,
            "accepted_beat_count": accepted_beats,
            "first_accepted_beat_audio_time_sec": first_accepted_audio_time,
            "beat_selection_mode": args.beat_selection_mode,
        },
        "phase_error_full_audio": full_metrics,
        "phase_error_after_first_accepted_beat": tracking_metrics,
        "accepted_beat_diagnostics": accepted_events,
        "conclusion_zh": _conclusion_zh(primary_metrics),
    }
    if analysis_duration + 1e-6 < motion_duration:
        report["coverage_warning"] = (
            "Audio or --max-seconds ended before the authored motion; whole-motion phase was not measured."
        )
    return PhaseAnalysisResult(report=report, samples=samples)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty phase time series.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _svg_paths(
    rows: Sequence[dict[str, Any]],
    value_key: str,
    left: float,
    top: float,
    width: float,
    height: float,
    value_min: float,
    value_max: float,
    split_wraps: bool = False,
) -> str:
    if not rows:
        return ""
    end_time = max(float(rows[-1]["time_sec"]), 1e-9)
    span = max(value_max - value_min, 1e-9)
    paths: list[str] = []
    points: list[str] = []
    previous: float | None = None
    for row in rows:
        value = float(row[value_key])
        if split_wraps and previous is not None and abs(value - previous) > 0.5:
            if points:
                paths.append("M " + " L ".join(points))
            points = []
        x = left + width * float(row["time_sec"]) / end_time
        y = top + height * (1.0 - (value - value_min) / span)
        points.append(f"{x:.2f},{y:.2f}")
        previous = value
    if points:
        paths.append("M " + " L ".join(points))
    return " ".join(paths)


def build_phase_chart(report: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    """Return a self-contained HTML phase comparison chart."""
    if not rows:
        raise ValueError("Cannot draw an empty phase time series.")
    width, height = 1200, 720
    left, plot_width = 82.0, 1078.0
    phase_top, phase_height = 45.0, 270.0
    error_top, error_height = 405.0, 210.0
    end_time = max(float(rows[-1]["time_sec"]), 1e-9)
    original_path = _svg_paths(
        rows, "original_phase", left, phase_top, plot_width, phase_height, 0.0, 1.0, True
    )
    driven_path = _svg_paths(
        rows, "driven_phase", left, phase_top, plot_width, phase_height, 0.0, 1.0, True
    )
    error_path = _svg_paths(
        rows,
        "signed_phase_error_cycles",
        left,
        error_top,
        plot_width,
        error_height,
        -0.5,
        0.5,
    )

    horizontal_grid: list[str] = []
    for tick in np.linspace(0.0, 1.0, 5):
        y = phase_top + phase_height * (1.0 - tick)
        horizontal_grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" class="grid"/>'
            f'<text x="{left - 12}" y="{y + 4:.2f}" class="tick ytick">{tick:.2f}</text>'
        )
    for tick in (-0.5, -0.25, 0.0, 0.25, 0.5):
        y = error_top + error_height * (1.0 - (tick + 0.5))
        line_class = "zero" if tick == 0.0 else "grid"
        horizontal_grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" class="{line_class}"/>'
            f'<text x="{left - 12}" y="{y + 4:.2f}" class="tick ytick">{tick:+.2f}</text>'
        )

    x_grid: list[str] = []
    for tick in np.linspace(0.0, end_time, 7):
        x = left + plot_width * tick / end_time
        x_grid.append(
            f'<line x1="{x:.2f}" y1="{phase_top}" x2="{x:.2f}" y2="{phase_top + phase_height}" class="grid"/>'
            f'<line x1="{x:.2f}" y1="{error_top}" x2="{x:.2f}" y2="{error_top + error_height}" class="grid"/>'
            f'<text x="{x:.2f}" y="{error_top + error_height + 27}" class="tick xtick">{tick:.1f}s</text>'
        )

    beat_lines: list[str] = []
    for event in report.get("accepted_beat_diagnostics", []):
        beat_time = float(event["beat_audio_time_sec"])
        if beat_time < 0.0 or beat_time > end_time:
            continue
        x = left + plot_width * beat_time / end_time
        beat_lines.append(
            f'<line x1="{x:.2f}" y1="{phase_top}" x2="{x:.2f}" y2="{phase_top + phase_height}" class="beat"/>'
            f'<line x1="{x:.2f}" y1="{error_top}" x2="{x:.2f}" y2="{error_top + error_height}" class="beat"/>'
        )

    metrics = report["phase_error_after_first_accepted_beat"]
    if metrics.get("status") != "ok":
        metrics = report["phase_error_full_audio"]
    mae = float(metrics.get("mean_absolute_error_cycles", 0.0))
    p95 = float(metrics.get("p95_absolute_error_cycles", 0.0))
    bias = float(metrics.get("circular_bias_cycles", 0.0))
    conclusion = html.escape(str(report.get("conclusion_zh", "")))
    source = html.escape(Path(str(report["motion_path"])).name)

    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Humanoid motion/audio phase comparison</title>
<style>
:root{{--bg:#0d141c;--panel:#151f2b;--text:#eef5fb;--muted:#9eb0c2;--grid:#334354;--original:#70d7ff;--driven:#ffb65c;--error:#ed6f91;--beat:#9f88ff;--zero:#d2dde8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1240px;margin:auto;padding:28px 20px 46px}} h1{{font-size:25px;margin:0 0 5px}} .subtitle{{color:var(--muted);margin-bottom:18px}}
.summary{{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0 18px}} .summary b{{font-size:18px}} .summary span{{color:var(--muted);margin-right:6px}}
.legend{{display:flex;gap:20px;flex-wrap:wrap;margin:0 0 8px}} .legend i{{display:inline-block;width:24px;height:3px;margin-right:7px;vertical-align:middle}}
.chart{{background:var(--panel);border:1px solid #273648;border-radius:12px;padding:12px;overflow-x:auto}}
svg{{display:block;width:100%;min-width:820px;height:auto}} .grid{{stroke:var(--grid);stroke-width:1}} .zero{{stroke:var(--zero);stroke-width:1.5}}
.original{{fill:none;stroke:var(--original);stroke-width:2.5}} .driven{{fill:none;stroke:var(--driven);stroke-width:2.5}}
.error{{fill:none;stroke:var(--error);stroke-width:2.2}} .beat{{stroke:var(--beat);stroke-width:1;stroke-dasharray:5 5;opacity:.65}}
.tick{{fill:var(--muted);font-size:12px}} .ytick{{text-anchor:end}} .xtick{{text-anchor:middle}} .label{{fill:var(--text);font-size:14px;font-weight:600}}
.note{{color:var(--muted);margin:14px 2px 0}} code{{color:var(--text)}}
</style></head><body><main>
<h1>Motion / microphone phase comparison</h1><div class="subtitle"><code>{source}</code> · {report['analyzed_duration_sec']:.2f}s · accepted beats shown as dashed lines</div>
<div class="summary"><div><span>MAE</span><b>{mae:.4f} cycle / {mae * 360.0:.1f}°</b></div><div><span>P95</span><b>{p95:.4f} cycle</b></div><div><span>Bias</span><b>{bias:+.4f} cycle</b></div></div>
<div class="legend"><span><i style="background:var(--original)"></i>Original phase</span><span><i style="background:var(--driven)"></i>Microphone-driven phase</span><span><i style="background:var(--error)"></i>Signed phase error</span><span><i style="border-top:2px dashed var(--beat);height:0"></i>Accepted beat</span></div>
<div class="chart"><svg viewBox="0 0 {width} {height}" role="img" aria-label="Original and microphone-driven motion phase with signed phase error over time">
{''.join(x_grid)}{''.join(horizontal_grid)}{''.join(beat_lines)}
<text x="{left}" y="25" class="label">Phase (cycle)</text><path d="{original_path}" class="original"/><path d="{driven_path}" class="driven"/>
<text x="{left}" y="385" class="label">Driven − original (cycle, circular)</text><path d="{error_path}" class="error"/>
<text x="{left + plot_width / 2}" y="{error_top + error_height + 58}" class="label" text-anchor="middle">Audio time</text>
</svg></div><p class="note">{conclusion}</p>
</main></body></html>'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare microphone-driven humanoid motion phase with the original authored phase."
    )
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--audio", type=Path, help="Synchronized audio; inferred for AIST++ when omitted.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--open-chart", action="store_true", help="Open the generated phase chart.")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--motion-fps", type=float, default=60.0)
    parser.add_argument("--analysis-fps", type=float, default=60.0)
    parser.add_argument("--keypoint-min-spacing-sec", type=float, default=0.28)
    parser.add_argument("--keypoint-prominence", type=float, default=0.18)
    parser.add_argument("--keypoint-smoothing-sec", type=float, default=0.08)
    parser.add_argument("--keypoint-boundary-sec", type=float, default=0.10)
    parser.add_argument("--keypoint-max-count", type=int)
    parser.add_argument("--speed-min", type=float, default=0.55)
    parser.add_argument("--speed-max", type=float, default=1.9)
    parser.add_argument("--amp-min", type=float, default=0.35)
    parser.add_argument("--amp-max", type=float, default=1.25)
    parser.add_argument("--smoothing-tau", type=float, default=0.22)
    parser.add_argument("--tempo-timeout", type=float, default=2.0)
    parser.add_argument("--accent-duration", type=float, default=0.16)
    parser.add_argument("--mic-sample-rate", type=int, default=16000)
    parser.add_argument("--mic-block-size", type=int, default=512)
    parser.add_argument("--plp-history-sec", type=float, default=8.0)
    parser.add_argument("--plp-analysis-interval-sec", type=float, default=0.10)
    parser.add_argument("--plp-hop-length", type=int, default=256)
    parser.add_argument("--plp-peak-prominence", type=float, default=0.15)
    parser.add_argument("--onset-threshold-scale", type=float, default=3.0)
    parser.add_argument("--noise-gate-rms", type=float, default=0.002)
    parser.add_argument("--noise-gate-ratio", type=float, default=1.8)
    parser.add_argument("--startup-calibration-sec", type=float, default=1.0)
    parser.add_argument("--audio-input-delay-sec", type=float, default=1.0)
    parser.add_argument("--min-beat-period", type=float, default=0.25)
    parser.add_argument("--max-beat-period", type=float, default=2.0)
    parser.add_argument("--mic-refractory-sec", type=float, default=0.18)
    parser.add_argument("--beat-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--beat-keypoint-interval-ratio", type=float, default=1.0)
    parser.add_argument("--beat-selection-mode", choices=("adaptive", "every"), default="adaptive")
    parser.add_argument("--beat-contrast-weight", type=float, default=0.5)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "motion_fps": args.motion_fps,
        "analysis_fps": args.analysis_fps,
        "mic_sample_rate": args.mic_sample_rate,
        "mic_block_size": args.mic_block_size,
        "plp_history_sec": args.plp_history_sec,
        "plp_analysis_interval_sec": args.plp_analysis_interval_sec,
        "plp_hop_length": args.plp_hop_length,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError("These arguments must be positive: " + ", ".join(invalid))
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive.")
    if args.audio_input_delay_sec < args.startup_calibration_sec:
        raise ValueError(
            "--audio-input-delay-sec must be at least --startup-calibration-sec so calibration "
            "does not consume the synchronized music."
        )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    result = run_analysis(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.motion.stem
    json_path = output_dir / f"{stem}_motion_audio_phase.json"
    csv_path = output_dir / f"{stem}_motion_audio_phase_timeseries.csv"
    chart_path = output_dir / f"{stem}_motion_audio_phase_chart.html"
    json_path.write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(csv_path, result.samples)
    chart_path.write_text(build_phase_chart(result.report, result.samples), encoding="utf-8")

    metrics = result.report["phase_error_after_first_accepted_beat"]
    if metrics.get("status") != "ok":
        metrics = result.report["phase_error_full_audio"]
    print(result.report["conclusion_zh"])
    print(
        f"Analyzed {result.report['analyzed_duration_sec']:.2f}s "
        f"({100.0 * result.report['motion_coverage_fraction']:.1f}% of motion); "
        f"beats detected/accepted={result.report['microphone_and_controller']['detected_beat_count']}/"
        f"{result.report['microphone_and_controller']['accepted_beat_count']}."
    )
    if metrics.get("status") == "ok":
        print(
            f"Phase MAE={metrics['mean_absolute_error_cycles']:.4f} cycles "
            f"({metrics['mean_absolute_error_degrees']:.2f} deg); "
            f"p95={metrics['p95_absolute_error_cycles']:.4f} cycles; "
            f"bias-corrected MAE={metrics['bias_corrected_mean_absolute_error_cycles']:.4f} cycles."
        )
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"Chart: {chart_path}")
    if args.open_chart:
        webbrowser.open(chart_path.as_uri())


if __name__ == "__main__":
    main()
