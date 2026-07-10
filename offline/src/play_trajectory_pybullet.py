import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play trajectory CSV (t, yaw_rad, pitch_rad) in PyBullet."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/trajectory.csv"),
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
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Play in real time. If disabled, simulation runs as fast as possible.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback when reaching the end of trajectory.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. Example: 0.5 is half speed, 2.0 is double speed.",
    )
    parser.add_argument(
        "--fixed-base",
        action="store_true",
        help="Load robot with fixed base.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional audio file to play in sync (Windows WAV only).",
    )
    parser.add_argument(
        "--audio-offset",
        type=float,
        default=0.0,
        help="Audio start offset in seconds. Positive means audio starts later.",
    )
    return parser.parse_args()


def load_trajectory(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return t, yaw, pitch


def target_from_time(
    play_t: float,
    t: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray,
) -> tuple[float, float]:
    yaw_target = float(np.interp(play_t, t, yaw))
    pitch_target = float(np.interp(play_t, t, pitch))
    return yaw_target, pitch_target


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be > 0")

    t, yaw, pitch = load_trajectory(args.csv)
    duration = float(t[-1] - t[0])

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

    p.resetJointState(robot_id, args.joint_yaw, yaw[0])
    p.resetJointState(robot_id, args.joint_pitch, pitch[0])

    sim_dt = 1.0 / 240.0
    p.setTimeStep(sim_dt)

    start_wall = time.perf_counter()
    if args.audio is not None:
        play_audio(args.audio, start_wall, args.audio_offset)

    while p.isConnected():
        elapsed = (time.perf_counter() - start_wall) * args.speed
        play_t = elapsed + t[0]

        if play_t > t[-1]:
            if args.loop:
                elapsed = elapsed % duration
                play_t = elapsed + t[0]
            else:
                break

        yaw_target, pitch_target = target_from_time(play_t, t, yaw, pitch)

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

    p.disconnect(client)
    if sys.platform == "win32":
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_ASYNC)
        except Exception:
            pass


def play_audio(audio_path: Path, start_wall: float, offset: float) -> None:
    if sys.platform != "win32":
        print("Audio sync currently supports Windows only (wav).")
        return
    if audio_path.suffix.lower() != ".wav":
        print("Audio sync currently supports .wav only. Skipping audio playback.")
        return
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}. Skipping audio playback.")
        return

    # winsound is standard on Windows and can play wav asynchronously.
    import winsound

    delay = offset
    if delay > 0:
        now = time.perf_counter()
        remain = (start_wall + delay) - now
        if remain > 0:
            time.sleep(remain)
    winsound.PlaySound(str(audio_path), winsound.SND_FILENAME | winsound.SND_ASYNC)


if __name__ == "__main__":
    main()
