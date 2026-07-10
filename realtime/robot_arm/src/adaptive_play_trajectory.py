import argparse
import csv
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data

try:
    import sounddevice as sd
except ImportError:
    sd = None


def wrap_phase_error(delta: float) -> float:
    return ((delta + 0.5) % 1.0) - 0.5


@dataclass
class Trajectory:
    t: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    @property
    def duration(self) -> float:
        return self.end - self.start

    def sample_phase(self, phase: float) -> tuple[float, float]:
        play_t = self.start + (phase % 1.0) * self.duration
        yaw_target = float(np.interp(play_t, self.t, self.yaw))
        pitch_target = float(np.interp(play_t, self.t, self.pitch))
        return yaw_target, pitch_target


class BeatIntervalEstimator:
    def __init__(
        self,
        min_intervals: int = 3,
        max_intervals: int = 5,
        min_period: float = 0.2,
        max_period: float = 2.0,
    ) -> None:
        self.min_intervals = min_intervals
        self.max_intervals = max_intervals
        self.min_period = min_period
        self.max_period = max_period
        self.beat_times: deque[float] = deque(maxlen=max_intervals + 1)

    def reset(self) -> None:
        self.beat_times.clear()

    def add_beat(self, beat_time: float) -> tuple[bool, Optional[float]]:
        if self.beat_times:
            period = beat_time - self.beat_times[-1]
            if period < self.min_period:
                return False, None
            if period > self.max_period:
                self.beat_times.clear()

        self.beat_times.append(beat_time)
        if len(self.beat_times) < self.min_intervals + 1:
            return True, None

        intervals = np.diff(np.asarray(self.beat_times, dtype=float))
        recent = intervals[-self.max_intervals :]
        valid = recent[(recent >= self.min_period) & (recent <= self.max_period)]
        if valid.size < self.min_intervals:
            return True, None
        return True, float(np.median(valid))


class PhaseTempoController:
    def __init__(
        self,
        authored_cycle_duration: float,
        beats_per_cycle: int,
        smoothing_tau: float,
        phase_correction_gain: float,
        speed_min: float,
        speed_max: float,
        tempo_timeout: float,
        default_bpm: Optional[float] = None,
    ) -> None:
        self.authored_phase_rate = 1.0 / max(authored_cycle_duration, 1e-6)
        if default_bpm is None:
            self.default_phase_rate = self.authored_phase_rate
        else:
            self.default_phase_rate = default_bpm / 60.0 / max(beats_per_cycle, 1)

        self.beats_per_cycle = max(beats_per_cycle, 1)
        self.smoothing_tau = max(smoothing_tau, 1e-6)
        self.phase_correction_gain = np.clip(phase_correction_gain, 0.0, 1.0)
        self.speed_min = max(speed_min, 1e-3)
        self.speed_max = max(speed_max, self.speed_min)
        self.tempo_timeout = max(tempo_timeout, 0.0)

        self.phase = 0.0
        self.phase_rate = self.default_phase_rate
        self.target_phase_rate = self.default_phase_rate
        self.last_update_wall: Optional[float] = None
        self.last_beat_wall: Optional[float] = None
        self.beat_index = 0
        self.last_period: Optional[float] = None

    def reset(self) -> None:
        self.phase = 0.0
        self.phase_rate = self.default_phase_rate
        self.target_phase_rate = self.default_phase_rate
        self.last_update_wall = None
        self.last_beat_wall = None
        self.beat_index = 0
        self.last_period = None

    def register_beat(self, beat_wall: float, beat_period: Optional[float]) -> None:
        self.last_beat_wall = beat_wall
        if self.beat_index == 0:
            self.phase = 0.0
        else:
            expected_phase = (self.beat_index % self.beats_per_cycle) / self.beats_per_cycle
            phase_error = wrap_phase_error(expected_phase - self.phase)
            self.phase = (self.phase + self.phase_correction_gain * phase_error) % 1.0
        self.beat_index += 1

        if beat_period is None:
            return

        self.last_period = beat_period
        unclamped_rate = 1.0 / max(beat_period * self.beats_per_cycle, 1e-6)
        min_rate = self.authored_phase_rate * self.speed_min
        max_rate = self.authored_phase_rate * self.speed_max
        self.target_phase_rate = float(np.clip(unclamped_rate, min_rate, max_rate))

    def update(self, now_wall: float) -> float:
        if self.last_update_wall is None:
            self.last_update_wall = now_wall
            return self.phase

        dt = max(now_wall - self.last_update_wall, 0.0)
        self.last_update_wall = now_wall

        if self.last_beat_wall is not None and (now_wall - self.last_beat_wall) > self.tempo_timeout:
            self.target_phase_rate = self.default_phase_rate

        alpha = 1.0 - math.exp(-dt / self.smoothing_tau)
        self.phase_rate += alpha * (self.target_phase_rate - self.phase_rate)
        self.phase = (self.phase + self.phase_rate * dt) % 1.0
        return self.phase

    @property
    def speed_multiplier(self) -> float:
        return self.phase_rate / max(self.authored_phase_rate, 1e-6)

    @property
    def estimated_bpm(self) -> Optional[float]:
        if self.last_period is None:
            return None
        return 60.0 / self.last_period


class MicClapDetector:
    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        threshold_scale: float,
        refractory_sec: float,
        absolute_threshold: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold_scale = threshold_scale
        self.refractory_sec = refractory_sec
        self.absolute_threshold = absolute_threshold
        self.event_queue: SimpleQueue[float] = SimpleQueue()
        self.noise_floor = 0.0
        self.prev_energy = 0.0
        self.prev_sample = 0.0
        self.last_event_time = 0.0
        self.stream = None

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed. Install requirements.txt to use --input-source mic.")

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def pop_event(self) -> Optional[float]:
        try:
            return self.event_queue.get_nowait()
        except Empty:
            return None

    def _callback(self, indata: np.ndarray, frames: int, time_info: dict, status: object) -> None:
        del frames, time_info, status

        mono = np.asarray(indata[:, 0], dtype=np.float64)
        highpass = np.diff(mono, prepend=self.prev_sample)
        self.prev_sample = float(mono[-1])
        energy = float(np.sqrt(np.mean(highpass * highpass)))
        self.noise_floor = 0.97 * self.noise_floor + 0.03 * energy if self.noise_floor > 0 else energy

        threshold = max(self.absolute_threshold, self.threshold_scale * self.noise_floor)
        now = time.perf_counter()
        rising = energy > max(self.prev_energy * 1.35, threshold)
        if rising and (now - self.last_event_time) >= self.refractory_sec:
            self.last_event_time = now
            self.event_queue.put(now)

        self.prev_energy = energy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive trajectory playback using tap tempo or microphone clap onsets."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("adaptive_clap/outputs/clap_trajectory.csv"),
        help="Path to trajectory CSV.",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="kuka_iiwa/model.urdf",
        help="URDF path. If relative, it is resolved by pybullet_data search path.",
    )
    parser.add_argument("--joint-yaw", type=int, default=0, help="Joint index for yaw.")
    parser.add_argument("--joint-pitch", type=int, default=1, help="Joint index for pitch.")
    parser.add_argument("--fixed-base", action="store_true", help="Load robot with fixed base.")
    parser.add_argument("--realtime", action="store_true", help="Sleep at the simulation time step.")
    parser.add_argument(
        "--input-source",
        choices=("tap", "mic"),
        default="tap",
        help="Use space-bar tapping or microphone clap onset detection.",
    )
    parser.add_argument(
        "--beats-per-cycle",
        type=int,
        default=2,
        help="How many beats correspond to one full trajectory cycle.",
    )
    parser.add_argument(
        "--default-bpm",
        type=float,
        default=None,
        help="Fallback BPM before enough taps/claps are observed. Defaults to authored cycle speed.",
    )
    parser.add_argument(
        "--smoothing-tau",
        type=float,
        default=0.25,
        help="Speed smoothing time constant in seconds.",
    )
    parser.add_argument(
        "--phase-correction-gain",
        type=float,
        default=0.35,
        help="Small phase correction applied when a beat is detected.",
    )
    parser.add_argument("--speed-min", type=float, default=0.5, help="Minimum speed multiplier.")
    parser.add_argument("--speed-max", type=float, default=1.8, help="Maximum speed multiplier.")
    parser.add_argument(
        "--tempo-timeout",
        type=float,
        default=2.0,
        help="Seconds without beats before drifting back to the default speed.",
    )
    parser.add_argument(
        "--min-beat-period",
        type=float,
        default=0.2,
        help="Reject beat intervals shorter than this.",
    )
    parser.add_argument(
        "--max-beat-period",
        type=float,
        default=2.0,
        help="Reset estimator if a beat interval exceeds this.",
    )
    parser.add_argument("--mic-sample-rate", type=int, default=16000, help="Microphone sample rate.")
    parser.add_argument("--mic-block-size", type=int, default=512, help="Microphone block size.")
    parser.add_argument(
        "--mic-threshold-scale",
        type=float,
        default=3.5,
        help="Dynamic threshold multiplier for clap onset detection.",
    )
    parser.add_argument(
        "--mic-absolute-threshold",
        type=float,
        default=0.02,
        help="Absolute lower bound on the clap energy threshold.",
    )
    parser.add_argument(
        "--mic-refractory-sec",
        type=float,
        default=0.18,
        help="Minimum spacing between accepted clap onsets.",
    )
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
        for row in reader:
            t_vals.append(float(row["t"]))
            yaw_vals.append(float(row["yaw_rad"]))
            pitch_vals.append(float(row["pitch_rad"]))

    t = np.asarray(t_vals, dtype=float)
    yaw = np.asarray(yaw_vals, dtype=float)
    pitch = np.asarray(pitch_vals, dtype=float)

    if t.size < 2:
        raise ValueError("Trajectory requires at least 2 rows.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("Column 't' must be strictly increasing.")

    return Trajectory(t=t, yaw=yaw, pitch=pitch)


def print_status(controller: PhaseTempoController, beat_period: Optional[float], source: str) -> None:
    bpm = controller.estimated_bpm
    bpm_text = f"{bpm:.1f}" if bpm is not None else "--"
    period_text = f"{beat_period:.3f}s" if beat_period is not None else "--"
    print(
        f"[{source}] beat accepted | period={period_text} | bpm={bpm_text} | speed={controller.speed_multiplier:.2f}x"
    )


def any_key_triggered(events: dict[int, int], keys: list[int]) -> bool:
    for key in keys:
        if key in events and events[key] & p.KEY_WAS_TRIGGERED:
            return True
    return False


def main() -> None:
    args = parse_args()
    trajectory = load_trajectory(args.csv)
    estimator = BeatIntervalEstimator(
        min_intervals=3,
        max_intervals=5,
        min_period=args.min_beat_period,
        max_period=args.max_beat_period,
    )
    controller = PhaseTempoController(
        authored_cycle_duration=trajectory.duration,
        beats_per_cycle=args.beats_per_cycle,
        smoothing_tau=args.smoothing_tau,
        phase_correction_gain=args.phase_correction_gain,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        tempo_timeout=args.tempo_timeout,
        default_bpm=args.default_bpm,
    )

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(args.urdf, useFixedBase=args.fixed_base)

    n_joints = p.getNumJoints(robot_id)
    if args.joint_yaw >= n_joints or args.joint_pitch >= n_joints:
        raise IndexError(
            f"Joint index out of range. Robot has {n_joints} joints, "
            f"got yaw={args.joint_yaw}, pitch={args.joint_pitch}"
        )

    yaw0, pitch0 = trajectory.sample_phase(0.0)
    p.resetJointState(robot_id, args.joint_yaw, yaw0)
    p.resetJointState(robot_id, args.joint_pitch, pitch0)

    sim_dt = 1.0 / 240.0
    p.setTimeStep(sim_dt)

    mic_detector = None
    if args.input_source == "mic":
        mic_detector = MicClapDetector(
            sample_rate=args.mic_sample_rate,
            block_size=args.mic_block_size,
            threshold_scale=args.mic_threshold_scale,
            refractory_sec=args.mic_refractory_sec,
            absolute_threshold=args.mic_absolute_threshold,
        )
        mic_detector.start()
        print("Microphone clap detection started.")
    else:
        print("Tap mode started. Focus the PyBullet window and press Space on each beat.")

    exit_keys = [ord("q"), ord("Q")]
    escape_key = getattr(p, "B3G_ESCAPE", None)
    if escape_key is not None:
        exit_keys.append(escape_key)

    try:
        while p.isConnected():
            now = time.perf_counter()
            events = p.getKeyboardEvents()

            if any_key_triggered(events, exit_keys):
                break
            if ord("r") in events and events[ord("r")] & p.KEY_WAS_TRIGGERED:
                estimator.reset()
                controller.reset()
                print("Tempo controller reset.")

            if args.input_source == "tap":
                if ord(" ") in events and events[ord(" ")] & p.KEY_WAS_TRIGGERED:
                    accepted, beat_period = estimator.add_beat(now)
                    if accepted:
                        controller.register_beat(now, beat_period)
                        print_status(controller, beat_period, "tap")
            else:
                assert mic_detector is not None
                while True:
                    beat_time = mic_detector.pop_event()
                    if beat_time is None:
                        break
                    accepted, beat_period = estimator.add_beat(beat_time)
                    if accepted:
                        controller.register_beat(beat_time, beat_period)
                        print_status(controller, beat_period, "mic")

            phase = controller.update(now)
            yaw_target, pitch_target = trajectory.sample_phase(phase)

            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=args.joint_yaw,
                controlMode=p.POSITION_CONTROL,
                targetPosition=yaw_target,
                force=200.0,
            )
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=args.joint_pitch,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pitch_target,
                force=200.0,
            )

            p.stepSimulation()
            if args.realtime:
                time.sleep(sim_dt)
    finally:
        if mic_detector is not None:
            mic_detector.stop()
        if p.isConnected():
            p.disconnect(client)


if __name__ == "__main__":
    main()
