"""Deterministic, microphone-free GMR action transition preview.

Compare a stationary quintic bridge (v2's blend shape) with a moving
crossfade. This is a kinematic diagnostic, not the live matcher pipeline.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

import numpy as np

import realtime_music_humanoid_dancer as base
from robot_motion import align_motion_frame_root, blend_motion_frames


MOTION_ROOT = Path(__file__).resolve().parents[1] / "data" / "aistpp_gmr"


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def quintic(u):
    u = float(np.clip(u, 0, 1))
    return u ** 3 * (10 + u * (-15 + 6 * u))


class Transition:
    """Samples in seconds, clamped to the last authored frame (never wraps)."""

    def __init__(self, sample_a, sample_b, end_a, end_b, entry, duration, mode):
        self.a, self.b = sample_a, sample_b
        self.end_a, self.end_b = end_a, end_b
        self.entry, self.duration, self.mode = entry, duration, mode
        self.start = end_a if mode == "bridge" else end_a - duration
        self.source = self.a(self.start)
        self.target = self.b(entry)
        self.finish = self.start + duration
        self.total = self.finish + end_b - entry - (duration if mode == "overlap" else 0)

    def aligned_b(self, t):
        return align_motion_frame_root(self.b(t), source_reference=self.target,
                                       target_reference=self.source)

    def sample(self, t):
        if t < self.start:
            return self.a(t), "action_a", 0.0
        if t < self.finish:
            u = (t - self.start) / self.duration
            if self.mode == "bridge":
                a, b = self.source, self.aligned_b(self.entry)
            else:
                a, b = self.a(t), self.aligned_b(self.entry + t - self.start)
            return blend_motion_frames(a, b, quintic(u), smoothstep=False), "transition", quintic(u)
        b_time = self.entry + t - self.finish
        if self.mode == "overlap":
            b_time += self.duration
        return self.aligned_b(b_time), "action_b", 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-a", type=Path, help="GMR pickle path, or ID in --motion-root")
    parser.add_argument("--motion-b", type=Path, help="GMR pickle path, or ID in --motion-root")
    parser.add_argument("--motion-root", type=Path, default=MOTION_ROOT)
    parser.add_argument("--list-motions", action="store_true")
    parser.add_argument("--mode", choices=("bridge", "overlap"), default="overlap")
    parser.add_argument("--duration", type=positive, default=1.2)
    parser.add_argument("--entry-seconds", type=float, default=0.0)
    parser.add_argument("--lead-seconds", type=positive, default=3.0,
                        help="Seconds of action A shown before the transition")
    parser.add_argument("--tail-seconds", type=positive, default=3.0)
    parser.add_argument("--rate", type=positive, default=120.0)
    parser.add_argument("--slow-motion", type=positive, default=1.0,
                        help="Wall seconds per motion second; 2 means half speed")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer or wall-clock delays")
    parser.add_argument("--csv", type=Path, help="Save raw requested joint positions and finite differences")
    parser.add_argument("--model", type=Path, default=base.DEFAULT_MODEL)
    args = parser.parse_args()
    available = sorted(args.motion_root.glob("*.pkl"))
    if args.list_motions:
        print("\n".join(p.stem for p in available))
        return
    if len(available) < 2 and (args.motion_a is None or args.motion_b is None):
        parser.error("Provide two GMR clips or a motion root containing at least two clips")

    def resolve(value, index):
        if value is None:
            return available[index]
        if value.is_file():
            return value
        return args.motion_root / (str(value) if value.suffix else f"{value}.pkl")

    paths = [resolve(args.motion_a, 0), resolve(args.motion_b, 1)]
    samplers = [base.GmrUnitreeG1MotionSampler(p, None, 1.0, 0.0, False) for p in paths]
    ends = [(len(s.frames) - 1) / s.fps for s in samplers]
    if not math.isfinite(args.entry_seconds) or not 0 <= args.entry_seconds < ends[1]:
        parser.error("--entry-seconds must be within action B before its final frame")
    if args.mode == "overlap" and args.duration >= min(ends[0], ends[1] - args.entry_seconds):
        parser.error("Overlap duration must fit inside both clips")
    player = base.MujocoHumanoidPlayer(args.model, realtime=not args.headless,
                                     headless=args.headless, viewer_rate_hz=60.0)
    output = None
    try:
        adapters = [base.make_pose_adapter(args, player, s) for s in samplers]
        for sampler, adapter in zip(samplers, adapters):
            player.ground_sampler(sampler, adapter)
        features = base.FeatureState()

        def sampler_fn(sampler, adapter, end):
            return lambda t: base.sample_robot_motion_frame(
                sampler, phase=float(np.clip(t, 0, end)) / sampler.duration,
                amplitude=1.0, accent=0.0, features=features,
                pose_adapter=adapter, modulator=None)

        samples = [sampler_fn(s, a, e) for s, a, e in zip(samplers, adapters, ends)]
        transition = Transition(*samples, *ends, args.entry_seconds, args.duration, args.mode)
        start = max(0.0, transition.start - args.lead_seconds)
        stop = min(transition.total, transition.finish + args.tail_seconds)
        names = sorted(samples[0](0).joint_positions)
        writer = None
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            output = args.csv.open("w", newline="", encoding="utf-8")
            writer = csv.writer(output)
            writer.writerow(["time", "stage", "weight", "max_speed_rad_s", "max_acceleration_rad_s2", *names])
        print(f"A: {paths[0].stem}\nB: {paths[1].stem}")
        print(f"{args.mode}: {args.duration:.3f}s; B entry={args.entry_seconds:.3f}s")
        player.start()
        wall_start = time.perf_counter()
        previous = velocity = None
        peak_speed = peak_accel = 0.0
        last_stage = None
        for i in range(math.ceil((stop - start) * args.rate) + 1):
            if not player.is_running():
                break
            t = start + i / args.rate
            frame, stage, weight = transition.sample(t)
            q = np.array([frame.joint_positions[n] for n in names])
            v = None if previous is None else (q - previous) * args.rate
            speed = 0.0 if v is None else float(np.max(np.abs(v)))
            accel = 0.0 if velocity is None else float(np.max(np.abs(v - velocity))) * args.rate
            previous, velocity = q, v
            peak_speed, peak_accel = max(peak_speed, speed), max(peak_accel, accel)
            if writer:
                writer.writerow([t - start, stage, weight, speed, accel, *q])
            if stage != last_stage:
                print(f"{t - start:.3f}s: {stage}")
                last_stage = stage
            if not args.headless:
                delay = wall_start + (t - start) * args.slow_motion - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            player.set_frame(frame)
            player.step(1 / args.rate)
        print(f"Raw target peaks: speed={peak_speed:.3f} rad/s; acceleration={peak_accel:.3f} rad/s^2")
    finally:
        if output:
            output.close()
        player.stop()


if __name__ == "__main__":
    main()
