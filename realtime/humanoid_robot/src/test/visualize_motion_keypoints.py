"""Visualize the keypoints selected by ``motion_keypoints``.

This is an isolated diagnostic CLI: it only reads a motion file and writes one
self-contained HTML report.  It does not change the realtime dancer or require
matplotlib.

Run from the repository root, for example::

    python realtime/humanoid_robot/src/test/visualize_motion_keypoints.py
    python realtime/humanoid_robot/src/test/visualize_motion_keypoints.py \
        --motion-source procedural --open
"""

from __future__ import annotations

import argparse
import html
import sys
import webbrowser
from pathlib import Path
from typing import Sequence

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent
REPOSITORY_ROOT = TEST_DIR.parents[3]
DEFAULT_OUTPUT = TEST_DIR / "output" / "motion_keypoints.html"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from motion_keypoints import (  # noqa: E402
    DEFAULT_FALLBACK_PHASES,
    default_keypoint_count,
    detect_motion_keypoints,
    motion_salience,
    pose_matrix,
    sample_pose_sequence,
)
from realtime_music_humanoid_dancer import (  # noqa: E402
    DEFAULT_AISTPP_MOTION,
    DEFAULT_BVH_MOTION,
    SMPL_FPS,
    AistppMotionSampler,
    BvhUnitreeG1MotionSampler,
    FeatureState,
    GmrUnitreeG1MotionSampler,
    HumanoidDanceSampler,
)


COLORS = ("#69d2e7", "#f7b267", "#a8e063", "#c79bf2", "#ff6b8a", "#ffd166", "#73a9ff", "#55d6be")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML report for motion_keypoints detection."
    )
    parser.add_argument(
        "--motion-source",
        choices=("aistpp", "gmr-pkl", "bvh", "procedural"),
        default="aistpp",
    )
    parser.add_argument("--aistpp-motion", type=Path, default=DEFAULT_AISTPP_MOTION)
    parser.add_argument("--aistpp-fps", type=float, default=SMPL_FPS)
    parser.add_argument("--gmr-motion", type=Path)
    parser.add_argument("--gmr-fps", type=float)
    parser.add_argument("--bvh-motion", type=Path, default=DEFAULT_BVH_MOTION)
    parser.add_argument("--bvh-fps", type=float)
    parser.add_argument("--bvh-neutral-frame", type=int, default=0)
    parser.add_argument("--duration", type=float, help="Override the motion cycle duration in seconds.")
    parser.add_argument("--sample-rate", type=float, default=120.0)
    parser.add_argument("--min-spacing-sec", type=float, default=0.28)
    parser.add_argument("--prominence", type=float, default=0.18)
    parser.add_argument("--max-count", type=int)
    parser.add_argument("--channels", type=int, default=8, help="Number of joint channels to plot.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--open", action="store_true", help="Open the generated report in the default browser.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPOSITORY_ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Motion file not found: {resolved}")
    return resolved


def make_sampler(args: argparse.Namespace) -> object:
    if args.motion_source == "procedural":
        return HumanoidDanceSampler(pose_gain=1.0, accent_gain=0.0)
    if args.motion_source == "aistpp":
        return AistppMotionSampler(resolve_path(args.aistpp_motion), args.aistpp_fps, 1.0, 0.0)
    if args.motion_source == "gmr-pkl":
        if args.gmr_motion is None:
            raise ValueError("--motion-source gmr-pkl requires --gmr-motion")
        return GmrUnitreeG1MotionSampler(resolve_path(args.gmr_motion), args.gmr_fps, 1.0, 0.0, False)
    return BvhUnitreeG1MotionSampler(
        resolve_path(args.bvh_motion), args.bvh_fps, 1.0, 0.0, False, args.bvh_neutral_frame
    )


def svg_path(values: np.ndarray, x: float, y: float, width: float, height: float) -> str:
    if values.size == 0:
        return ""
    finite = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(np.min(finite)), float(np.max(finite))
    span = max(high - low, 1e-9)
    points = []
    denominator = max(len(finite) - 1, 1)
    for index, value in enumerate(finite):
        px = x + width * index / denominator
        py = y + height * (1.0 - (float(value) - low) / span)
        points.append(f"{px:.2f},{py:.2f}")
    return "M " + " L ".join(points)


def salience_svg(
    phases: np.ndarray,
    salience: np.ndarray,
    keypoint_phases: Sequence[float],
    prominence: float,
    duration: float,
) -> str:
    width, height = 1120, 360
    left, top, plot_width, plot_height = 72, 30, 1018, 270
    path = svg_path(salience, left, top, plot_width, plot_height)
    threshold_y = top + plot_height * (1.0 - float(np.clip(prominence, 0.0, 1.0)))
    markers = []
    for number, phase in enumerate(keypoint_phases, start=1):
        index = int(np.argmin(np.abs(phases - phase)))
        score = float(salience[index]) if salience.size else 0.0
        marker_x = left + plot_width * float(phase)
        marker_y = top + plot_height * (1.0 - float(np.clip(score, 0.0, 1.0)))
        markers.append(
            f'<line x1="{marker_x:.2f}" y1="{top}" x2="{marker_x:.2f}" y2="{top + plot_height}" class="key-line"/>'
            f'<circle cx="{marker_x:.2f}" cy="{marker_y:.2f}" r="6" class="key-dot"/>'
            f'<text x="{marker_x:.2f}" y="{max(marker_y - 11, 17):.2f}" class="key-label">K{number}</text>'
        )
    ticks = []
    for tick in range(6):
        fraction = tick / 5
        tick_x = left + plot_width * fraction
        ticks.append(
            f'<line x1="{tick_x:.2f}" y1="{top + plot_height}" x2="{tick_x:.2f}" y2="{top + plot_height + 6}" class="axis"/>'
            f'<text x="{tick_x:.2f}" y="{top + plot_height + 25}" class="tick">{fraction * duration:.2f}s</text>'
        )
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Motion salience and detected keypoints">
      <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>
      <line x1="{left}" y1="{threshold_y:.2f}" x2="{left + plot_width}" y2="{threshold_y:.2f}" class="threshold"/>
      <text x="{left + 8}" y="{threshold_y - 7:.2f}" class="threshold-label">prominence {prominence:.2f}</text>
      <path d="{path}" class="salience"/>
      {''.join(markers)}{''.join(ticks)}
      <text x="16" y="{top + plot_height / 2}" class="axis-label" transform="rotate(-90 16 {top + plot_height / 2})">salience</text>
    </svg>'''


def channels_svg(matrix: np.ndarray, names: Sequence[str], limit: int) -> tuple[str, tuple[str, ...]]:
    if matrix.shape[1] == 0 or limit <= 0:
        return "<p>No varying joint channels.</p>", ()
    ranges = np.ptp(matrix, axis=0)
    selected = np.argsort(ranges)[::-1][: min(limit, matrix.shape[1])]
    selected_names = tuple(names[index] for index in selected)
    row_height, left, plot_width = 62, 210, 880
    height = 24 + row_height * len(selected)
    rows = []
    for row, column in enumerate(selected):
        y = 16 + row * row_height
        color = COLORS[row % len(COLORS)]
        path = svg_path(matrix[:, column], left, y + 7, plot_width, 40)
        rows.append(
            f'<text x="{left - 12}" y="{y + 32}" class="channel-label">{html.escape(names[column])}</text>'
            f'<line x1="{left}" y1="{y + 27}" x2="{left + plot_width}" y2="{y + 27}" class="zero"/>'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
    return f'<svg viewBox="0 0 1120 {height}" role="img" aria-label="Normalized joint trajectories">{"".join(rows)}</svg>', selected_names


def build_report(
    source: str,
    duration: float,
    phases: np.ndarray,
    matrix: np.ndarray,
    names: Sequence[str],
    salience: np.ndarray,
    result: object,
    prominence: float,
    min_spacing_sec: float,
    channel_count: int,
) -> str:
    trajectory_chart, selected_names = channels_svg(matrix, names, channel_count)
    keypoint_rows = []
    for index, (phase, score) in enumerate(zip(result.phases, result.scores), start=1):
        keypoint_rows.append(
            f"<tr><td>K{index}</td><td>{phase:.4f}</td><td>{phase * duration:.3f} s</td><td>{score:.3f}</td></tr>"
        )
    status = "fallback" if result.fallback_used else "detected"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Motion keypoint visualization</title>
<style>
:root{{--bg:#10151d;--panel:#18212c;--text:#edf4fb;--muted:#9eb0c1;--grid:#435366;--accent:#4de3c1;--warn:#ffcc66}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:32px 24px 60px}} h1{{margin:0 0 6px;font-size:28px}} h2{{margin:0 0 12px;font-size:18px}}
.subtitle,.note{{color:var(--muted)}} .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}
.card,.panel{{background:var(--panel);border:1px solid #263545;border-radius:12px}} .card{{padding:14px 16px}} .card b{{display:block;font-size:21px}}
.panel{{padding:18px;margin:14px 0;overflow:auto}} svg{{display:block;width:100%;min-width:760px;height:auto}}
.axis,.zero{{stroke:var(--grid);stroke-width:1}} .threshold{{stroke:var(--warn);stroke-width:1;stroke-dasharray:7 5}}
.threshold-label{{fill:var(--warn);font-size:12px}} .salience{{fill:none;stroke:var(--accent);stroke-width:3}}
.key-line{{stroke:#ff6b8a;stroke-width:1;stroke-dasharray:4 4}} .key-dot{{fill:#ff6b8a;stroke:#fff;stroke-width:2}}
.key-label{{fill:#fff;font-size:12px;text-anchor:middle;font-weight:700}} .tick{{fill:var(--muted);font-size:12px;text-anchor:middle}}
.axis-label{{fill:var(--muted);font-size:12px;text-anchor:middle}} .channel-label{{fill:var(--text);font-size:12px;text-anchor:end}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #2a3949}} th{{color:var(--muted)}}
code{{color:#9fe8d7}} @media(max-width:760px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<h1>Motion keypoint visualization</h1><div class="subtitle">source: <code>{html.escape(source)}</code></div>
<div class="cards"><div class="card"><span>duration</span><b>{duration:.2f} s</b></div>
<div class="card"><span>samples</span><b>{len(phases)}</b></div><div class="card"><span>active channels</span><b>{len(names)}</b></div>
<div class="card"><span>result</span><b>{len(result.phases)} {status}</b></div></div>
<section class="panel"><h2>Salience and selected keypoints</h2>
{salience_svg(phases, salience, result.phases, prominence, duration)}</section>
<section class="panel"><h2>Keypoint details</h2><table><thead><tr><th>keypoint</th><th>phase</th><th>time</th><th>score</th></tr></thead>
<tbody>{''.join(keypoint_rows)}</tbody></table><p class="note">reason: {html.escape(result.reason)}; minimum spacing: {min_spacing_sec:.3f} s</p></section>
<section class="panel"><h2>Most dynamic normalized channels</h2>{trajectory_chart}
<p class="note">Channels shown: {html.escape(', '.join(selected_names))}. Values are normalized and weighted exactly as used by the detector.</p></section>
</main></body></html>'''


def main() -> None:
    args = parse_args()
    if args.sample_rate <= 0 or args.min_spacing_sec < 0 or args.prominence < 0:
        raise ValueError("sample rate must be positive; spacing and prominence must be non-negative")
    sampler = make_sampler(args)
    duration = max(args.duration if args.duration is not None else float(sampler.duration), 1e-6)
    phases, poses = sample_pose_sequence(
        sampler, duration, FeatureState(is_active=True), sample_rate=args.sample_rate
    )
    matrix, names = pose_matrix(poses)
    salience = motion_salience(matrix, names)
    max_count = args.max_count if args.max_count is not None else default_keypoint_count(duration)
    result = detect_motion_keypoints(
        phases,
        poses,
        duration,
        min_spacing_sec=args.min_spacing_sec,
        prominence=args.prominence,
        max_count=max_count,
        fallback_phases=DEFAULT_FALLBACK_PHASES,
    )
    report = build_report(
        args.motion_source,
        duration,
        phases,
        matrix,
        names,
        salience,
        result,
        args.prominence,
        args.min_spacing_sec,
        args.channels,
    )
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote keypoint visualization: {output.resolve()}")
    print(f"Keypoints ({'fallback' if result.fallback_used else 'detected'}): " + ", ".join(f"{p:.4f}" for p in result.phases))
    if args.open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
