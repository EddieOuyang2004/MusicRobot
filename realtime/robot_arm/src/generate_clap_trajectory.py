import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a simple looping clap-like trajectory for adaptive tempo tests."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("adaptive_clap/outputs/clap_trajectory.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--traj-hz", type=float, default=100.0, help="Output trajectory sample rate.")
    parser.add_argument(
        "--cycle-duration",
        type=float,
        default=1.0,
        help="Duration of one authored trajectory cycle in seconds.",
    )
    parser.add_argument("--yaw-amp-deg", type=float, default=18.0, help="Side sway amplitude.")
    parser.add_argument("--open-pitch-deg", type=float, default=22.0, help="Open pose pitch.")
    parser.add_argument("--clap-pitch-deg", type=float, default=-24.0, help="Closed clap pose pitch.")
    parser.add_argument("--bob-deg", type=float, default=5.0, help="Small vertical accent amplitude.")
    return parser.parse_args()


def generate_cycle(
    traj_hz: float,
    cycle_duration: float,
    yaw_amp_deg: float,
    open_pitch_deg: float,
    clap_pitch_deg: float,
    bob_deg: float,
) -> dict[str, np.ndarray]:
    n = int(np.floor(cycle_duration * traj_hz)) + 1
    t = np.linspace(0.0, cycle_duration, n)
    phase = np.clip(t / max(cycle_duration, 1e-6), 0.0, 1.0)

    close_env = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
    yaw = np.deg2rad(yaw_amp_deg) * np.sin(2.0 * np.pi * phase)
    pitch = np.deg2rad(open_pitch_deg + (clap_pitch_deg - open_pitch_deg) * close_env)
    pitch += np.deg2rad(bob_deg) * np.sin(4.0 * np.pi * phase)

    keypoint = np.zeros_like(phase, dtype=int)
    keypoint_label = np.full(phase.shape, "", dtype=object)
    beat_weight = np.zeros_like(phase)

    open_idx = int(np.argmin(np.abs(phase - 0.0)))
    clap_idx = int(np.argmin(np.abs(phase - 0.5)))
    keypoint[open_idx] = 1
    keypoint_label[open_idx] = "open"
    beat_weight[open_idx] = 0.5
    keypoint[clap_idx] = 1
    keypoint_label[clap_idx] = "clap"
    beat_weight[clap_idx] = 1.0

    return {
        "t": t,
        "yaw_rad": yaw,
        "pitch_rad": pitch,
        "yaw_deg": np.rad2deg(yaw),
        "pitch_deg": np.rad2deg(pitch),
        "phase": phase,
        "close_env": close_env,
        "keypoint": keypoint,
        "keypoint_label": keypoint_label,
        "beat_weight": beat_weight,
    }


def write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    fieldnames = list(columns.keys())
    row_count = len(next(iter(columns.values())))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(row_count):
            writer.writerow({name: values[i] for name, values in columns.items()})


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    columns = generate_cycle(
        traj_hz=args.traj_hz,
        cycle_duration=args.cycle_duration,
        yaw_amp_deg=args.yaw_amp_deg,
        open_pitch_deg=args.open_pitch_deg,
        clap_pitch_deg=args.clap_pitch_deg,
        bob_deg=args.bob_deg,
    )
    write_csv(args.out, columns)
    print(f"Saved adaptive clap trajectory: {args.out} ({len(next(iter(columns.values())))} rows)")


if __name__ == "__main__":
    main()
