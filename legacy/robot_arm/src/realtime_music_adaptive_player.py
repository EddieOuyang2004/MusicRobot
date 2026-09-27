from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SHARED_SRC = Path(__file__).resolve().parents[3] / "realtime" / "shared"
if str(SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(SHARED_SRC))

from music_runtime import (
    MusicFrame,
    RealtimeMusicAnalyzer,
    AdaptiveMotionController,
)

p = None
pybullet_data = None


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "beat", "keypoint", "primary"}


@dataclass
class Trajectory:
    t: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    phase: np.ndarray
    keypoint_phases: tuple[float, ...]

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 1e-6)


class TrajectorySampler:
    def __init__(
        self,
        trajectory: Trajectory,
        motion: str,
        neutral_yaw: Optional[float],
        neutral_pitch: Optional[float],
        accent_gain_deg: float,
        clap_yaw_amp_deg: float,
        clap_open_pitch_deg: float,
        clap_pitch_deg: float,
        clap_bob_deg: float,
        figure_yaw_amp_deg: float,
        figure_pitch_amp_deg: float,
    ) -> None:
        self.trajectory = trajectory
        self.motion = motion
        default_neutral_yaw = 0.0 if motion in {"clap", "figure-eight"} else float(np.mean(trajectory.yaw))
        default_neutral_pitch = math.radians(clap_open_pitch_deg) if motion == "clap" else float(np.mean(trajectory.pitch))
        self.neutral_yaw = default_neutral_yaw if neutral_yaw is None else neutral_yaw
        self.neutral_pitch = default_neutral_pitch if neutral_pitch is None else neutral_pitch
        self.accent_gain_rad = math.radians(accent_gain_deg)
        self.clap_yaw_amp_rad = math.radians(clap_yaw_amp_deg)
        self.clap_open_pitch_rad = math.radians(clap_open_pitch_deg)
        self.clap_pitch_rad = math.radians(clap_pitch_deg)
        self.clap_bob_rad = math.radians(clap_bob_deg)
        self.figure_yaw_amp_rad = math.radians(figure_yaw_amp_deg)
        self.figure_pitch_amp_rad = math.radians(figure_pitch_amp_deg)

    def sample(self, phase: float, amplitude_scale: float, accent: float, brightness: float) -> tuple[float, float]:
        phase = phase % 1.0
        if self.motion == "clap":
            yaw_base, pitch_base = self._sample_clap_base(phase)
        elif self.motion == "figure-eight":
            yaw_base, pitch_base = self._sample_figure_eight_base(phase)
        else:
            yaw_base = self._interp_loop(phase, self.trajectory.phase, self.trajectory.yaw)
            pitch_base = self._interp_loop(phase, self.trajectory.phase, self.trajectory.pitch)

        yaw = self.neutral_yaw + (yaw_base - self.neutral_yaw) * amplitude_scale
        pitch = self.neutral_pitch + (pitch_base - self.neutral_pitch) * amplitude_scale

        brightness_scale = 0.75 + 0.5 * brightness
        accent_rad = self.accent_gain_rad * accent * brightness_scale
        yaw += 0.25 * accent_rad
        pitch += accent_rad
        return yaw, pitch

    def _sample_clap_base(self, phase: float) -> tuple[float, float]:
        close_env = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
        yaw = self.neutral_yaw + self.clap_yaw_amp_rad * math.sin(2.0 * math.pi * phase)
        pitch = self.clap_open_pitch_rad + (self.clap_pitch_rad - self.clap_open_pitch_rad) * close_env
        pitch += self.clap_bob_rad * math.sin(4.0 * math.pi * phase)
        return yaw, pitch

    def _sample_figure_eight_base(self, phase: float) -> tuple[float, float]:
        yaw = self.neutral_yaw + self.figure_yaw_amp_rad * math.sin(2.0 * math.pi * phase)
        pitch = self.neutral_pitch + self.figure_pitch_amp_rad * math.sin(4.0 * math.pi * phase + math.pi / 2.0)
        return yaw, pitch

    @staticmethod
    def _interp_loop(phase: float, phase_track: np.ndarray, values: np.ndarray) -> float:
        phases = phase_track
        vals = values
        if phases[0] > 0.0 or phases[-1] < 1.0:
            phases = np.concatenate(([0.0], phases, [1.0]))
            vals = np.concatenate(([values[-1]], values, [values[0]]))
        return float(np.interp(phase, phases, vals))


class PyBulletRobotPlayer:
    def __init__(
        self,
        urdf: str,
        joint_yaw: int,
        joint_pitch: int,
        fixed_base: bool,
        sim_dt: float,
    ) -> None:
        self.urdf = urdf
        self.joint_yaw = joint_yaw
        self.joint_pitch = joint_pitch
        self.fixed_base = fixed_base
        self.sim_dt = sim_dt
        self.client: Optional[int] = None
        self.robot_id: Optional[int] = None

    def start(self, initial_yaw: float, initial_pitch: float) -> None:
        global p, pybullet_data
        if p is None or pybullet_data is None:
            import pybullet as bullet
            import pybullet_data as bullet_data

            p = bullet
            pybullet_data = bullet_data

        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf, useFixedBase=self.fixed_base)

        n_joints = p.getNumJoints(self.robot_id)
        if self.joint_yaw >= n_joints or self.joint_pitch >= n_joints:
            raise IndexError(
                f"Joint index out of range. Robot has {n_joints} joints, "
                f"got yaw={self.joint_yaw}, pitch={self.joint_pitch}"
            )

        p.resetJointState(self.robot_id, self.joint_yaw, initial_yaw)
        p.resetJointState(self.robot_id, self.joint_pitch, initial_pitch)
        p.setTimeStep(self.sim_dt)

    def is_connected(self) -> bool:
        return p.isConnected()

    def keyboard_events(self) -> dict[int, int]:
        return p.getKeyboardEvents()

    def set_targets(self, yaw_target: float, pitch_target: float) -> None:
        assert self.robot_id is not None
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.joint_yaw,
            controlMode=p.POSITION_CONTROL,
            targetPosition=yaw_target,
            force=200.0,
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.joint_pitch,
            controlMode=p.POSITION_CONTROL,
            targetPosition=pitch_target,
            force=200.0,
        )

    def step(self) -> None:
        p.stepSimulation()

    def stop(self) -> None:
        if p.isConnected():
            if self.client is None:
                p.disconnect()
            else:
                p.disconnect(self.client)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime music-adaptive trajectory playback using microphone audio features."
    )
    parser.add_argument("--csv", type=Path, default=Path("legacy/robot_arm/outputs/clap_trajectory.csv"))
    parser.add_argument("--urdf", type=str, default="kuka_iiwa/model.urdf")
    parser.add_argument("--joint-yaw", type=int, default=0)
    parser.add_argument("--joint-pitch", type=int, default=1)
    parser.add_argument("--fixed-base", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--motion",
        choices=("clap", "figure-eight", "csv"),
        default="clap",
        help="Motion style used by the realtime player.",
    )
    parser.add_argument("--motion-cycle-duration", type=float, default=1.0)
    parser.add_argument("--beats-per-cycle", type=int, default=2)
    parser.add_argument("--speed-min", type=float, default=0.5)
    parser.add_argument("--speed-max", type=float, default=1.8)
    parser.add_argument("--amp-min", type=float, default=0.65)
    parser.add_argument("--amp-max", type=float, default=1.45)
    parser.add_argument("--accent-gain-deg", type=float, default=10.0)
    parser.add_argument("--accent-duration", type=float, default=0.16)
    parser.add_argument("--clap-yaw-amp-deg", type=float, default=0.0)
    parser.add_argument("--clap-open-pitch-deg", type=float, default=22.0)
    parser.add_argument("--clap-pitch-deg", type=float, default=-24.0)
    parser.add_argument("--clap-bob-deg", type=float, default=5.0)
    parser.add_argument("--figure-yaw-amp-deg", type=float, default=28.0)
    parser.add_argument("--figure-pitch-amp-deg", type=float, default=22.0)
    parser.add_argument("--mic-sample-rate", type=int, default=16000)
    parser.add_argument(
        "--mic-device",
        default=None,
        help="Optional sounddevice input index or exact device name.",
    )
    parser.add_argument("--mic-block-size", type=int, default=512)
    parser.add_argument("--plp-history-sec", type=float, default=8.0)
    parser.add_argument("--plp-analysis-interval-sec", type=float, default=0.10)
    parser.add_argument("--plp-hop-length", type=int, default=256)
    parser.add_argument("--plp-peak-prominence", type=float, default=0.15)
    parser.add_argument("--onset-threshold-scale", type=float, default=3.0)
    parser.add_argument(
        "--noise-gate-rms",
        type=float,
        default=0.002,
        help="Minimum RMS required before microphone input is treated as music.",
    )
    parser.add_argument(
        "--noise-gate-ratio",
        type=float,
        default=1.8,
        help="Input RMS must exceed estimated noise by this multiplier to be active.",
    )
    parser.add_argument(
        "--startup-calibration-sec",
        type=float,
        default=1.0,
        help="Seconds used at startup to estimate the room/microphone noise floor.",
    )
    parser.add_argument("--smoothing-tau", type=float, default=0.25)
    parser.add_argument("--tempo-timeout", type=float, default=2.0)
    parser.add_argument("--min-beat-period", type=float, default=0.25)
    parser.add_argument("--max-beat-period", type=float, default=2.0)
    parser.add_argument("--mic-refractory-sec", type=float, default=0.18)
    parser.add_argument(
        "--beat-confidence-threshold",
        type=float,
        default=0.25,
        help="Ignore PLP beats below this confidence before applying trajectory alignment.",
    )
    parser.add_argument(
        "--beat-keypoint-interval-ratio",
        type=float,
        default=0.85,
        help=(
            "Reject beats closer than this fraction of the fastest allowed authored keypoint spacing. "
            "The spacing is derived from keypoint phases and --speed-max."
        ),
    )
    parser.add_argument("--neutral-yaw-rad", type=float, default=None)
    parser.add_argument("--neutral-pitch-rad", type=float, default=None)
    parser.add_argument("--disable-keypoint-alignment", action="store_true")
    parser.add_argument("--status-interval", type=float, default=1.0)
    return parser.parse_args()


def load_trajectory(csv_path: Path) -> Trajectory:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"t", "yaw_rad", "pitch_rad"}
        if reader.fieldnames is None:
            raise ValueError("CSV is missing a header row.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        t_vals: list[float] = []
        yaw_vals: list[float] = []
        pitch_vals: list[float] = []
        phase_vals: list[Optional[float]] = []
        keypoint_candidates: list[tuple[float, float]] = []

        has_phase = "phase" in reader.fieldnames
        has_keypoint = "keypoint" in reader.fieldnames
        has_weight = "beat_weight" in reader.fieldnames
        for row in reader:
            t_vals.append(float(row["t"]))
            yaw_vals.append(float(row["yaw_rad"]))
            pitch_vals.append(float(row["pitch_rad"]))
            phase_vals.append(float(row["phase"]) % 1.0 if has_phase and row["phase"] != "" else None)

            weight = float(row["beat_weight"]) if has_weight and row["beat_weight"] != "" else 0.0
            marked = has_keypoint and is_truthy(row["keypoint"])
            if marked or weight > 0.0:
                phase_value = phase_vals[-1]
                keypoint_candidates.append((0.0 if phase_value is None else phase_value, weight if weight > 0.0 else 1.0))

    t = np.asarray(t_vals, dtype=float)
    yaw = np.asarray(yaw_vals, dtype=float)
    pitch = np.asarray(pitch_vals, dtype=float)

    if t.size < 2:
        raise ValueError("Trajectory requires at least 2 rows.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("Column 't' must be strictly increasing.")

    if any(phase is None for phase in phase_vals):
        phase = (t - t[0]) / max(t[-1] - t[0], 1e-6)
    else:
        phase = np.asarray([float(value) for value in phase_vals], dtype=float)

    keypoint_phases = tuple(
        phase for phase, _weight in sorted(keypoint_candidates, key=lambda item: (-item[1], item[0]))
    )
    keypoint_phases = tuple(dict.fromkeys(round(phase, 6) for phase in keypoint_phases))
    return Trajectory(t=t, yaw=yaw, pitch=pitch, phase=phase, keypoint_phases=keypoint_phases)


def make_builtin_trajectory(cycle_duration: float, keypoint_phases: tuple[float, ...]) -> Trajectory:
    duration = max(cycle_duration, 1e-6)
    t = np.asarray([0.0, duration], dtype=float)
    return Trajectory(
        t=t,
        yaw=np.zeros_like(t),
        pitch=np.zeros_like(t),
        phase=np.asarray([0.0, 1.0], dtype=float),
        keypoint_phases=keypoint_phases,
    )


def any_key_triggered(events: dict[int, int], keys: list[int]) -> bool:
    for key in keys:
        if key in events and events[key] & p.KEY_WAS_TRIGGERED:
            return True
    return False


def print_status(controller: AdaptiveMotionController, last_frame: Optional[MusicFrame]) -> None:
    bpm = controller.estimated_bpm
    bpm_text = f"{bpm:.1f}" if bpm is not None else "--"
    rms_text = f"{last_frame.rms:.4f}/{last_frame.gate_rms:.4f}" if last_frame is not None else "--"
    rms_norm_text = f"{last_frame.rms_norm:.2f}" if last_frame is not None else "--"
    bright_text = f"{last_frame.brightness:.2f}" if last_frame is not None else "--"
    band_text = (
        f"{last_frame.low_energy:.2f}/{last_frame.mid_energy:.2f}/{last_frame.high_energy:.2f}"
        if last_frame is not None
        else "--"
    )
    rolloff_text = f"{last_frame.spectral_rolloff:.2f}" if last_frame is not None else "--"
    zcr_text = f"{last_frame.zero_crossing_rate:.2f}" if last_frame is not None else "--"
    rhythm_text = f"{last_frame.rhythm_density:.2f}" if last_frame is not None else "--"
    stable_text = f"{last_frame.tempo_stability:.2f}" if last_frame is not None else "--"
    offbeat_text = f"{last_frame.offbeat_ratio:.2f}" if last_frame is not None else "--"
    state_text = "music" if last_frame is not None and last_frame.is_active else "silent"
    print(
        f"{state_text} | "
        f"bpm={bpm_text} | speed={controller.speed_multiplier:.2f}x | "
        f"amp={controller.amplitude_scale:.2f}x | rms/gate={rms_text} | "
        f"level={rms_norm_text} | brightness={bright_text} | "
        f"bands(L/M/H)={band_text} | rolloff={rolloff_text} | zcr={zcr_text} | "
        f"rhythm={rhythm_text} | stable={stable_text} | offbeat={offbeat_text}"
    )


def main() -> None:
    args = parse_args()
    if args.motion == "clap":
        keypoint_phases = (0.0, 0.5)
    elif args.motion == "csv":
        trajectory = load_trajectory(args.csv)
        keypoint_phases = trajectory.keypoint_phases
    else:
        keypoint_phases = ()
    if args.motion != "csv":
        trajectory = make_builtin_trajectory(args.motion_cycle_duration, keypoint_phases)
    use_keypoints = bool(keypoint_phases) and not args.disable_keypoint_alignment

    analyzer = RealtimeMusicAnalyzer(
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
        input_device=(
            int(args.mic_device)
            if args.mic_device is not None and args.mic_device.lstrip("-").isdigit()
            else args.mic_device
        ),
    )
    controller = AdaptiveMotionController(
        authored_cycle_duration=trajectory.duration,
        beats_per_cycle=args.beats_per_cycle,
        keypoint_phases=keypoint_phases,
        use_keypoints=use_keypoints,
        smoothing_tau=args.smoothing_tau,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        amp_min=args.amp_min,
        amp_max=args.amp_max,
        accent_duration=args.accent_duration,
        tempo_timeout=args.tempo_timeout,
        beat_confidence_threshold=args.beat_confidence_threshold,
        beat_keypoint_interval_ratio=args.beat_keypoint_interval_ratio,
    )
    sampler = TrajectorySampler(
        trajectory=trajectory,
        motion=args.motion,
        neutral_yaw=args.neutral_yaw_rad,
        neutral_pitch=args.neutral_pitch_rad,
        accent_gain_deg=args.accent_gain_deg,
        clap_yaw_amp_deg=args.clap_yaw_amp_deg,
        clap_open_pitch_deg=args.clap_open_pitch_deg,
        clap_pitch_deg=args.clap_pitch_deg,
        clap_bob_deg=args.clap_bob_deg,
        figure_yaw_amp_deg=args.figure_yaw_amp_deg,
        figure_pitch_amp_deg=args.figure_pitch_amp_deg,
    )
    player = PyBulletRobotPlayer(
        urdf=args.urdf,
        joint_yaw=args.joint_yaw,
        joint_pitch=args.joint_pitch,
        fixed_base=args.fixed_base,
        sim_dt=1.0 / 240.0,
    )

    initial_yaw, initial_pitch = sampler.sample(0.0, 0.0, 0.0, 0.0)
    player.start(initial_yaw, initial_pitch)
    print(f"Calibrating microphone noise from the first {args.startup_calibration_sec:.2f}s of live input.")
    analyzer.start()
    print(
        f"Realtime music adaptive player started with PLP rhythm tracking and {args.motion} motion. "
        "Press R to reset, Q or Esc to quit."
    )
    if use_keypoints:
        print(f"Using {len(keypoint_phases)} motion keypoint phase(s) for beat alignment.")
    else:
        print("Using evenly spaced beat phases.")

    exit_keys = [ord("q"), ord("Q")]
    escape_key = getattr(p, "B3G_ESCAPE", None)
    if escape_key is not None:
        exit_keys.append(escape_key)

    last_frame: Optional[MusicFrame] = None
    last_status = time.perf_counter()
    try:
        while player.is_connected():
            now = time.perf_counter()
            events = player.keyboard_events()
            if any_key_triggered(events, exit_keys):
                break
            if ord("r") in events and events[ord("r")] & p.KEY_WAS_TRIGGERED:
                analyzer.reset()
                controller.reset()
                print("Realtime music controller reset.")

            for frame in analyzer.drain():
                last_frame = frame
                controller.observe(frame)

            phase, amplitude, accent, brightness = controller.update(now)
            yaw_target, pitch_target = sampler.sample(phase, amplitude, accent, brightness)
            player.set_targets(yaw_target, pitch_target)
            player.step()

            if now - last_status >= args.status_interval:
                print_status(controller, last_frame)
                last_status = now

            if args.realtime:
                time.sleep(1.0 / 240.0)
    finally:
        analyzer.stop()
        player.stop()


if __name__ == "__main__":
    main()
