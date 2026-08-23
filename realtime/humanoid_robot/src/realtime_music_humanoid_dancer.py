from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import mujoco.viewer
except ImportError:
    mujoco.viewer = None

ROOT = Path(__file__).resolve().parents[3]
ROBOT_ARM_SRC = ROOT / "realtime" / "robot_arm" / "src"
if str(ROBOT_ARM_SRC) not in sys.path:
    sys.path.insert(0, str(ROBOT_ARM_SRC))

from realtime_music_adaptive_player import (
    AdaptiveMotionController,
    MusicFrame,
    RealtimeMusicAnalyzer,
    compute_beat_contrasts,
)
from aistpp_velocity_keypoints import detect_aistpp_velocity_keypoints
from music_pose_modulator import MusicPoseModulator
from motion_keypoints import (
    DEFAULT_FALLBACK_PHASES,
    default_keypoint_count,
    detect_motion_keypoints,
    parse_keypoint_phases,
    sample_pose_sequence,
)
from unitree_g1_dance_adapter import UnitreeG1DanceAdapter, UnitreeG1JointPoseAdapter


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "assets" / "open_humanoid_dancer.xml"
DEFAULT_AISTPP_ROOT = Path(__file__).resolve().parents[1] / "data" / "aistpp"
DEFAULT_AISTPP_MOTION = DEFAULT_AISTPP_ROOT / "motions" / "gWA_sBM_cAll_d26_mWA0_ch07.pkl"
DEFAULT_BVH_MOTION = Path(__file__).resolve().parents[1] / "data" / "lafan1_dance" / "dance1_subject1.bvh"
SMPL_FPS = 60.0


@dataclass
class FeatureState:
    rms_norm: float = 0.0
    onset_strength: float = 0.0
    brightness: float = 0.0
    low_energy: float = 0.0
    mid_energy: float = 0.0
    high_energy: float = 0.0
    rhythm_density: float = 0.0
    tempo_stability: float = 0.0
    offbeat_ratio: float = 0.0
    is_active: bool = False

    def update(self, frame: MusicFrame, alpha: float) -> None:
        self.is_active = frame.is_active
        target = self.from_frame(frame) if frame.is_active else FeatureState()
        self.rms_norm += alpha * (target.rms_norm - self.rms_norm)
        self.onset_strength += alpha * (target.onset_strength - self.onset_strength)
        self.brightness += alpha * (target.brightness - self.brightness)
        self.low_energy += alpha * (target.low_energy - self.low_energy)
        self.mid_energy += alpha * (target.mid_energy - self.mid_energy)
        self.high_energy += alpha * (target.high_energy - self.high_energy)
        self.rhythm_density += alpha * (target.rhythm_density - self.rhythm_density)
        self.tempo_stability += alpha * (target.tempo_stability - self.tempo_stability)
        self.offbeat_ratio += alpha * (target.offbeat_ratio - self.offbeat_ratio)

    @classmethod
    def from_frame(cls, frame: MusicFrame) -> "FeatureState":
        return cls(
            rms_norm=frame.rms_norm,
            onset_strength=min(frame.onset_strength, 2.0) / 2.0,
            brightness=frame.brightness,
            low_energy=frame.low_energy,
            mid_energy=frame.mid_energy,
            high_energy=frame.high_energy,
            rhythm_density=frame.rhythm_density,
            tempo_stability=frame.tempo_stability,
            offbeat_ratio=frame.offbeat_ratio,
            is_active=frame.is_active,
        )


class HumanoidDanceSampler:
    """Maps music features to a looping side-step dance pose."""

    pose_space = "dancer"

    def __init__(self, pose_gain: float, accent_gain: float) -> None:
        self.pose_gain = max(pose_gain, 0.0)
        self.accent_gain = max(accent_gain, 0.0)
        self.duration = 1.0

    def sample(self, phase: float, amplitude: float, accent: float, features: FeatureState) -> dict[str, float]:
        del features
        phase = phase % 1.0
        beat = 2.0 * math.pi * phase
        half = 4.0 * math.pi * phase
        side = math.sin(beat)
        cross = math.sin(beat + math.pi)
        bounce = 0.5 * (1.0 - math.cos(half))

        amp = self.pose_gain

        hip_sway = amp * 0.20 * side
        torso_twist = amp * 0.25 * math.sin(beat + 0.25 * math.pi)
        torso_roll = amp * 0.18 * side
        torso_pitch = amp * (-0.08 - 0.10 * bounce)
        head_nod = amp * (-0.10 * bounce)

        arm_lift = 0.55
        arm_swing = amp * 0.65
        left_arm = arm_swing * math.sin(beat + 0.10 * math.pi)
        right_arm = arm_swing * math.sin(beat + 1.10 * math.pi)
        left_roll = 0.45 + arm_lift + amp * 0.25 * side
        right_roll = -0.45 - arm_lift + amp * 0.25 * side
        elbow_pulse = -0.55 - 0.35 * bounce

        step = amp * 0.30
        knee_base = 0.18 + 0.35 * amp * bounce
        left_step = max(side, 0.0)
        right_step = max(-side, 0.0)

        pose = {
            "torso_yaw": torso_twist,
            "torso_roll": torso_roll,
            "torso_pitch": torso_pitch,
            "neck_pitch": head_nod,
            "left_shoulder_pitch": -0.35 + left_arm,
            "left_shoulder_roll": left_roll,
            "left_elbow": elbow_pulse,
            "right_shoulder_pitch": -0.35 + right_arm,
            "right_shoulder_roll": right_roll,
            "right_elbow": elbow_pulse + 0.15 * math.sin(half + math.pi),
            "left_hip_yaw": 0.18 * amp * side,
            "left_hip_roll": hip_sway,
            "left_hip_pitch": -0.10 - step * left_step,
            "left_knee": knee_base + 0.30 * amp * left_step,
            "left_ankle_pitch": -0.10 * bounce + 0.10 * left_step,
            "right_hip_yaw": -0.18 * amp * side,
            "right_hip_roll": hip_sway,
            "right_hip_pitch": -0.10 - step * right_step,
            "right_knee": knee_base + 0.30 * amp * right_step,
            "right_ankle_pitch": -0.10 * bounce + 0.10 * right_step,
        }
        return {name: float(value) for name, value in pose.items()}


class AistppMotionSampler:
    """Samples the first AIST++ train motion and retargets it to this MJCF."""

    pose_space = "dancer"

    def __init__(self, motion_path: Path, fps: float, pose_gain: float, accent_gain: float) -> None:
        self.motion_path = motion_path
        self.fps = max(fps, 1e-6)
        self.pose_gain = max(pose_gain, 0.0)
        self.accent_gain = max(accent_gain, 0.0)

        with motion_path.open("rb") as motion_file:
            motion = pickle.load(motion_file)
        poses = np.asarray(motion["smpl_poses"], dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] != 72:
            raise ValueError(f"Expected smpl_poses with shape (N, 72), got {poses.shape}.")
        if len(poses) < 2:
            raise ValueError("AIST++ motion must contain at least two frames.")
        self.frames = poses.reshape((-1, 24, 3))
        translations = motion.get("smpl_trans")
        self.translations = None if translations is None else np.asarray(translations, dtype=np.float64)
        if self.translations is not None and self.translations.shape != (len(self.frames), 3):
            raise ValueError(
                f"Expected smpl_trans with shape {(len(self.frames), 3)}, got {self.translations.shape}."
            )
        scaling = motion.get("smpl_scaling")
        self.scaling = None if scaling is None else float(np.asarray(scaling, dtype=np.float64).reshape(-1)[0])
        self.duration = len(self.frames) / self.fps

    def sample(self, phase: float, amplitude: float, accent: float, features: FeatureState) -> dict[str, float]:
        del amplitude, accent, features
        phase = phase % 1.0
        frame_pos = phase * len(self.frames)
        frame_a = int(math.floor(frame_pos)) % len(self.frames)
        frame_b = (frame_a + 1) % len(self.frames)
        blend = frame_pos - math.floor(frame_pos)
        frame = (1.0 - blend) * self.frames[frame_a] + blend * self.frames[frame_b]
        pose = self._retarget(frame)

        return {name: float(self.pose_gain * value) for name, value in pose.items()}

    @staticmethod
    def _euler_xyz(frame: np.ndarray, joint_index: int) -> np.ndarray:
        return Rotation.from_rotvec(frame[joint_index]).as_euler("xyz", degrees=False)

    def _retarget(self, frame: np.ndarray) -> dict[str, float]:
        root = self._euler_xyz(frame, 0)
        left_hip = self._euler_xyz(frame, 1)
        right_hip = self._euler_xyz(frame, 2)
        spine1 = self._euler_xyz(frame, 3)
        left_knee = self._euler_xyz(frame, 4)
        right_knee = self._euler_xyz(frame, 5)
        spine2 = self._euler_xyz(frame, 6)
        left_ankle = self._euler_xyz(frame, 7)
        right_ankle = self._euler_xyz(frame, 8)
        spine3 = self._euler_xyz(frame, 9)
        neck = self._euler_xyz(frame, 12)
        head = self._euler_xyz(frame, 15)
        left_shoulder = self._euler_xyz(frame, 16)
        right_shoulder = self._euler_xyz(frame, 17)
        left_elbow = self._euler_xyz(frame, 18)
        right_elbow = self._euler_xyz(frame, 19)

        spine_yaw = 0.34 * spine1[2] + 0.33 * spine2[2] + 0.33 * spine3[2]
        spine_roll = 0.34 * spine1[0] + 0.33 * spine2[0] + 0.33 * spine3[0]
        spine_pitch = 0.34 * spine1[1] + 0.33 * spine2[1] + 0.33 * spine3[1]

        left_elbow_flex = -0.18 - 0.65 * abs(left_elbow[1]) - 0.35 * abs(left_elbow[2])
        right_elbow_flex = -0.18 - 0.65 * abs(right_elbow[1]) - 0.35 * abs(right_elbow[2])
        left_knee_flex = 0.15 + 0.72 * abs(left_knee[0]) + 0.18 * abs(left_knee[2])
        right_knee_flex = 0.15 + 0.72 * abs(right_knee[0]) + 0.18 * abs(right_knee[2])

        return {
            "torso_yaw": 0.22 * root[2] + 0.50 * spine_yaw,
            "torso_roll": 0.18 * root[0] + 0.42 * spine_roll,
            "torso_pitch": 0.16 * root[1] + 0.48 * spine_pitch,
            "neck_pitch": 0.35 * neck[1] + 0.30 * head[1],
            "left_shoulder_pitch": 0.65 * left_shoulder[1] - 0.18 * left_shoulder[0],
            "left_shoulder_roll": 0.62 + 0.52 * left_shoulder[2] + 0.18 * left_shoulder[0],
            "left_elbow": left_elbow_flex,
            "right_shoulder_pitch": 0.65 * right_shoulder[1] - 0.18 * right_shoulder[0],
            "right_shoulder_roll": -0.62 + 0.52 * right_shoulder[2] - 0.18 * right_shoulder[0],
            "right_elbow": right_elbow_flex,
            "left_hip_yaw": 0.42 * left_hip[2],
            "left_hip_roll": 0.42 * left_hip[0],
            "left_hip_pitch": 0.52 * left_hip[1] - 0.10 * left_knee_flex,
            "left_knee": left_knee_flex,
            "left_ankle_pitch": 0.35 * left_ankle[1],
            "right_hip_yaw": 0.42 * right_hip[2],
            "right_hip_roll": 0.42 * right_hip[0],
            "right_hip_pitch": 0.52 * right_hip[1] - 0.10 * right_knee_flex,
            "right_knee": right_knee_flex,
            "right_ankle_pitch": 0.35 * right_ankle[1],
        }


class GmrUnitreeG1MotionSampler:
    """Samples a GMR-retargeted Unitree G1 pickle file."""

    pose_space = "unitree-g1"

    def __init__(
        self,
        motion_path: Path,
        fps_override: Optional[float],
        pose_gain: float,
        accent_gain: float,
        use_music_amplitude: bool,
    ) -> None:
        self.motion_path = motion_path
        self.pose_gain = max(pose_gain, 0.0)
        self.accent_gain = max(accent_gain, 0.0)
        self.use_music_amplitude = use_music_amplitude

        with motion_path.open("rb") as motion_file:
            motion = pickle.load(motion_file)
        if not isinstance(motion, dict):
            raise ValueError(f"GMR motion pickle must contain a dict, got {type(motion).__name__}.")

        dof_pos = np.asarray(motion.get("dof_pos"), dtype=np.float32)
        if dof_pos.ndim != 2:
            raise ValueError(f"Expected GMR dof_pos with shape (N, D), got {dof_pos.shape}.")
        if len(dof_pos) < 2:
            raise ValueError("GMR motion must contain at least two frames.")

        g1_dof_count = len(UnitreeG1DanceAdapter.GMR_DOF_NAMES)
        if dof_pos.shape[1] < g1_dof_count:
            raise ValueError(f"Expected at least {g1_dof_count} G1 DoF columns, got {dof_pos.shape[1]}.")

        dof_names = motion.get("dof_names") or motion.get("joint_names")
        if dof_names is None:
            self.dof_names = UnitreeG1DanceAdapter.GMR_DOF_NAMES
            self.frames = dof_pos[:, :g1_dof_count]
        else:
            self.dof_names = tuple(_logical_g1_name(str(name)) for name in dof_names)
            if len(self.dof_names) != dof_pos.shape[1]:
                raise ValueError(
                    f"GMR dof_names length ({len(self.dof_names)}) does not match "
                    f"dof_pos columns ({dof_pos.shape[1]})."
                )
            self.frames = dof_pos

        fps = fps_override if fps_override is not None else float(motion.get("fps", 30.0))
        self.fps = max(fps, 1e-6)
        self.duration = len(self.frames) / self.fps

    def sample(self, phase: float, amplitude: float, accent: float, features: FeatureState) -> dict[str, float]:
        del amplitude, accent, features
        phase = phase % 1.0
        frame_pos = phase * len(self.frames)
        frame_a = int(math.floor(frame_pos)) % len(self.frames)
        frame_b = (frame_a + 1) % len(self.frames)
        blend = frame_pos - math.floor(frame_pos)
        frame = (1.0 - blend) * self.frames[frame_a] + blend * self.frames[frame_b]

        pose = {}
        gain = self.pose_gain
        accent_gain = 0.0
        if self.use_music_amplitude:
            gain *= max(amplitude, 0.0)
            accent_gain = self.accent_gain * max(accent, 0.0)
        for index, name in enumerate(self.dof_names):
            if not name:
                continue
            value = float(gain * frame[index])
            if name in {"left_knee", "right_knee", "left_elbow", "right_elbow"}:
                value += math.copysign(0.08 * accent_gain, value if value != 0.0 else 1.0)
            pose[name] = value
        return pose


@dataclass(frozen=True)
class BvhJoint:
    name: str
    channels: tuple[str, ...]
    channel_start: int


class BvhUnitreeG1MotionSampler:
    """Samples a BVH clip and retargets it into GMR's Unitree G1 29-DoF order."""

    pose_space = "unitree-g1"

    def __init__(
        self,
        motion_path: Path,
        fps_override: Optional[float],
        pose_gain: float,
        accent_gain: float,
        use_music_amplitude: bool,
        neutral_frame: int,
    ) -> None:
        self.motion_path = motion_path
        self.pose_gain = max(pose_gain, 0.0)
        self.accent_gain = max(accent_gain, 0.0)
        self.use_music_amplitude = use_music_amplitude
        self.joints, values, frame_time = self._load_bvh(motion_path)
        if len(values) < 2:
            raise ValueError("BVH motion must contain at least two frames.")

        self.fps = max(fps_override if fps_override is not None else 1.0 / frame_time, 1e-6)
        self.neutral_frame = int(np.clip(neutral_frame, 0, len(values) - 1))
        self.joint_rotations = self._joint_rotations(values)
        self.neutral_rotations = {
            name: self.joint_rotations[name][self.neutral_frame]
            for name in self.joint_rotations
        }
        self.dof_names = UnitreeG1DanceAdapter.GMR_DOF_NAMES
        self.frames = self._retarget_all()
        self.duration = len(self.frames) / self.fps

    def sample(self, phase: float, amplitude: float, accent: float, features: FeatureState) -> dict[str, float]:
        del features
        phase = phase % 1.0
        frame_pos = phase * len(self.frames)
        frame_a = int(math.floor(frame_pos)) % len(self.frames)
        frame_b = (frame_a + 1) % len(self.frames)
        blend = frame_pos - math.floor(frame_pos)
        frame = (1.0 - blend) * self.frames[frame_a] + blend * self.frames[frame_b]

        pose = {}
        gain = self.pose_gain
        accent_gain = 0.0
        if self.use_music_amplitude:
            gain *= max(amplitude, 0.0)
            accent_gain = self.accent_gain * max(accent, 0.0)
        for index, name in enumerate(self.dof_names):
            value = float(gain * frame[index])
            if name in {"left_knee", "right_knee", "left_elbow", "right_elbow"}:
                value += math.copysign(0.08 * accent_gain, value if value != 0.0 else 1.0)
            pose[name] = value
        return pose

    @staticmethod
    def _load_bvh(path: Path) -> tuple[list[BvhJoint], np.ndarray, float]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        joints: list[BvhJoint] = []
        channel_cursor = 0
        motion_index = None

        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if line == "MOTION":
                motion_index = index
                break
            parts = line.split()
            if len(parts) >= 2 and parts[0] in {"ROOT", "JOINT"}:
                joints.append(BvhJoint(name=parts[1], channels=(), channel_start=channel_cursor))
            elif parts and parts[0] == "CHANNELS":
                if not joints:
                    raise ValueError(f"BVH CHANNELS appeared before any joint in {path}.")
                channel_count = int(parts[1])
                channels = tuple(parts[2 : 2 + channel_count])
                joints[-1] = BvhJoint(joints[-1].name, channels, channel_cursor)
                channel_cursor += channel_count

        if motion_index is None:
            raise ValueError(f"BVH MOTION section not found: {path}")

        frame_count = None
        frame_time = None
        data_start = None
        for index in range(motion_index + 1, len(lines)):
            line = lines[index].strip()
            if not line:
                continue
            if line.startswith("Frames:"):
                frame_count = int(line.split(":", 1)[1].strip())
            elif line.startswith("Frame Time:"):
                frame_time = float(line.split(":", 1)[1].strip())
                data_start = index + 1
                break

        if frame_count is None or frame_time is None or data_start is None:
            raise ValueError(f"BVH motion header is incomplete: {path}")

        rows = []
        for raw_line in lines[data_start:]:
            line = raw_line.strip()
            if line:
                rows.append([float(value) for value in line.split()])
        values = np.asarray(rows, dtype=np.float32)
        if values.shape != (frame_count, channel_cursor):
            raise ValueError(
                f"BVH data shape mismatch in {path}: expected {(frame_count, channel_cursor)}, got {values.shape}."
            )
        return joints, values, frame_time

    def _joint_rotations(self, values: np.ndarray) -> dict[str, Rotation]:
        rotations = {}
        for joint in self.joints:
            rotation_axes = []
            rotation_columns = []
            for channel_offset, channel in enumerate(joint.channels):
                if channel.endswith("rotation"):
                    rotation_axes.append(channel[0].upper())
                    rotation_columns.append(joint.channel_start + channel_offset)
            if not rotation_axes:
                continue
            angles = values[:, rotation_columns]
            rotations[joint.name] = Rotation.from_euler("".join(rotation_axes), angles, degrees=True)
        return rotations

    def _joint_delta_euler(self, joint_name: str, frame_index: int) -> np.ndarray:
        rotations = self.joint_rotations.get(joint_name)
        if rotations is None:
            return np.zeros(3, dtype=np.float32)
        delta = self.neutral_rotations[joint_name].inv() * rotations[frame_index]
        return delta.as_euler("xyz", degrees=False)

    def _retarget_all(self) -> np.ndarray:
        frames = np.zeros((len(next(iter(self.joint_rotations.values()))), len(self.dof_names)), dtype=np.float32)
        for frame_index in range(len(frames)):
            pose = self._retarget_frame(frame_index)
            frames[frame_index] = [pose.get(name, 0.0) for name in self.dof_names]
        return frames

    def _retarget_frame(self, frame_index: int) -> dict[str, float]:
        hips = self._joint_delta_euler("Hips", frame_index)
        spine = self._average_euler(("Spine", "Spine1", "Spine2"), frame_index)
        neck = self._average_euler(("Neck", "Head"), frame_index)
        left_hip = self._joint_delta_euler("LeftUpLeg", frame_index)
        right_hip = self._joint_delta_euler("RightUpLeg", frame_index)
        left_knee = self._joint_delta_euler("LeftLeg", frame_index)
        right_knee = self._joint_delta_euler("RightLeg", frame_index)
        left_ankle = self._joint_delta_euler("LeftFoot", frame_index)
        right_ankle = self._joint_delta_euler("RightFoot", frame_index)
        left_shoulder = self._average_euler(("LeftShoulder", "LeftArm"), frame_index)
        right_shoulder = self._average_euler(("RightShoulder", "RightArm"), frame_index)
        left_elbow = self._joint_delta_euler("LeftForeArm", frame_index)
        right_elbow = self._joint_delta_euler("RightForeArm", frame_index)
        left_wrist = self._joint_delta_euler("LeftHand", frame_index)
        right_wrist = self._joint_delta_euler("RightHand", frame_index)

        left_knee_flex = 0.10 + 0.78 * abs(left_knee[1]) + 0.20 * abs(left_knee[2])
        right_knee_flex = 0.10 + 0.78 * abs(right_knee[1]) + 0.20 * abs(right_knee[2])
        left_elbow_flex = -0.12 - 0.68 * abs(left_elbow[1]) - 0.30 * abs(left_elbow[2])
        right_elbow_flex = -0.12 - 0.68 * abs(right_elbow[1]) - 0.30 * abs(right_elbow[2])

        return {
            "left_hip_pitch": 0.58 * left_hip[1] - 0.08 * left_knee_flex,
            "left_hip_roll": 0.48 * left_hip[0],
            "left_hip_yaw": 0.42 * left_hip[2],
            "left_knee": left_knee_flex,
            "left_ankle_pitch": 0.42 * left_ankle[1],
            "left_ankle_roll": 0.32 * left_ankle[0],
            "right_hip_pitch": 0.58 * right_hip[1] - 0.08 * right_knee_flex,
            "right_hip_roll": 0.48 * right_hip[0],
            "right_hip_yaw": 0.42 * right_hip[2],
            "right_knee": right_knee_flex,
            "right_ankle_pitch": 0.42 * right_ankle[1],
            "right_ankle_roll": 0.32 * right_ankle[0],
            "waist_yaw": 0.18 * hips[2] + 0.52 * spine[2],
            "waist_roll": 0.16 * hips[0] + 0.48 * spine[0],
            "waist_pitch": 0.16 * hips[1] + 0.48 * spine[1],
            "left_shoulder_pitch": 0.58 * left_shoulder[1] - 0.16 * left_shoulder[0],
            "left_shoulder_roll": 0.46 * left_shoulder[2] + 0.22 * left_shoulder[0],
            "left_shoulder_yaw": 0.38 * left_shoulder[2] + 0.18 * spine[2],
            "left_elbow": left_elbow_flex,
            "left_wrist_roll": 0.22 * left_wrist[0],
            "left_wrist_pitch": 0.22 * left_wrist[1] - 0.08 * neck[1],
            "left_wrist_yaw": 0.22 * left_wrist[2],
            "right_shoulder_pitch": 0.58 * right_shoulder[1] - 0.16 * right_shoulder[0],
            "right_shoulder_roll": 0.46 * right_shoulder[2] - 0.22 * right_shoulder[0],
            "right_shoulder_yaw": 0.38 * right_shoulder[2] + 0.18 * spine[2],
            "right_elbow": right_elbow_flex,
            "right_wrist_roll": 0.22 * right_wrist[0],
            "right_wrist_pitch": 0.22 * right_wrist[1] - 0.08 * neck[1],
            "right_wrist_yaw": 0.22 * right_wrist[2],
        }

    def _average_euler(self, joint_names: tuple[str, ...], frame_index: int) -> np.ndarray:
        values = [self._joint_delta_euler(name, frame_index) for name in joint_names if name in self.joint_rotations]
        if not values:
            return np.zeros(3, dtype=np.float32)
        return np.mean(values, axis=0)


class MujocoHumanoidPlayer:
    def __init__(self, model_path: Path, realtime: bool, headless: bool) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.realtime = realtime
        self.headless = headless
        self.actuator_ids = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(self.model.nu)
        }
        self.joint_qpos_ids = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i): int(self.model.jnt_qposadr[i])
            for i in range(self.model.njnt)
        }
        self.actuator_joint_qpos_ids: dict[str, int] = {}
        self.actuator_joint_ranges: dict[str, tuple[float, float] | None] = {}
        for name, actuator_id in self.actuator_ids.items():
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            self.actuator_joint_qpos_ids[name] = int(self.model.jnt_qposadr[joint_id])
            if bool(self.model.jnt_limited[joint_id]):
                low, high = self.model.jnt_range[joint_id]
                self.actuator_joint_ranges[name] = (float(low), float(high))
            else:
                self.actuator_joint_ranges[name] = None
        self.viewer = None

    @property
    def actuator_names(self) -> set[str]:
        return set(self.actuator_ids)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def start(self) -> None:
        if not self.headless:
            if mujoco.viewer is None:
                raise RuntimeError("mujoco.viewer is unavailable; rerun with --headless for a non-GUI smoke test.")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 4.2
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -12

    def is_running(self) -> bool:
        return self.viewer.is_running() if self.viewer is not None else True

    def set_pose(self, pose: dict[str, float]) -> None:
        for name, value in pose.items():
            actuator_id = self.actuator_ids.get(name)
            qpos_id = self.actuator_joint_qpos_ids.get(name, self.joint_qpos_ids.get(name))
            if actuator_id is None:
                continue
            low, high = self.model.actuator_ctrlrange[actuator_id]
            target = float(np.clip(value, low, high))
            self.data.ctrl[actuator_id] = target
            if qpos_id is not None:
                joint_range = self.actuator_joint_ranges.get(name)
                qpos_target = target if joint_range is None else float(np.clip(target, joint_range[0], joint_range[1]))
                self.data.qpos[qpos_id] = qpos_target

    def step(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        self.data.time += self.dt
        if self.viewer is not None:
            self.viewer.sync()
        if self.realtime:
            time.sleep(self.dt)

    def stop(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class FileMicrophoneSource:
    """Feed an audio file to the microphone analyzer at realtime speed."""

    def __init__(
        self,
        path: Path,
        analyzer: RealtimeMusicAnalyzer,
        startup_delay_sec: float = 1.0,
        throttle: bool = True,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Audio input file not found: {path}")
        self.path = path
        self.analyzer = analyzer
        self.sample_rate = int(analyzer.sample_rate)
        self.block_size = int(analyzer.block_size)
        self.startup_delay_sec = max(float(startup_delay_sec), 0.0)
        self.startup_samples = int(round(self.startup_delay_sec * self.sample_rate))
        self.throttle = bool(throttle)
        audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        self.audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self.audio.size == 0:
            raise ValueError(f"Audio input file contains no samples: {path}")
        self.cursor = 0
        self.target_cursor = 0.0
        self.stream_start_wall: float | None = None
        self.last_advance_wall: float | None = None

    @property
    def playback_seconds(self) -> float:
        audio_samples = max(self.cursor - self.startup_samples, 0)
        return float(min(audio_samples, self.audio.size) / self.sample_rate)

    @property
    def done(self) -> bool:
        return self.cursor >= self.startup_samples + self.audio.size

    def start(self) -> None:
        # This is the virtual equivalent of opening a microphone. Resetting here
        # makes the following silent delay available to startup noise calibration.
        self.analyzer.reset()
        self.stream_start_wall = time.perf_counter()
        self.last_advance_wall = self.stream_start_wall

    def stop(self) -> None:
        return

    def advance(self, dt: float) -> None:
        dt = max(float(dt), 0.0)
        if self.throttle and dt > 0.0:
            time.sleep(dt)
        now = time.perf_counter()
        elapsed = dt
        if self.last_advance_wall is not None:
            elapsed = max(dt, now - self.last_advance_wall)
        self.last_advance_wall = now
        total_samples = self.startup_samples + self.audio.size
        self.target_cursor = min(
            self.target_cursor + elapsed * self.sample_rate,
            float(total_samples),
        )
        target = int(self.target_cursor)
        while target - self.cursor >= self.block_size:
            self._feed_block(self.block_size)
        if target >= total_samples and self.cursor < total_samples:
            self._feed_block(total_samples - self.cursor)

    def drain(self) -> list[MusicFrame]:
        return self.analyzer.drain()

    def recent_audio(self, window_seconds: float) -> np.ndarray | None:
        required = int(round(max(float(window_seconds), 0.0) * self.sample_rate))
        audio_cursor = int(np.clip(self.cursor - self.startup_samples, 0, self.audio.size))
        if required <= 0 or audio_cursor < required:
            return None
        return np.asarray(self.audio[audio_cursor - required : audio_cursor], dtype=np.float32)

    def _feed_block(self, size: int) -> None:
        end = self.cursor + max(int(size), 0)
        positions = np.arange(self.cursor, end)
        block = np.zeros(positions.size, dtype=np.float32)
        audio_mask = positions >= self.startup_samples
        audio_indices = positions[audio_mask] - self.startup_samples
        valid = audio_indices < self.audio.size
        if np.any(valid):
            block_indices = np.flatnonzero(audio_mask)[valid]
            block[block_indices] = self.audio[audio_indices[valid]]
        callback_block = block
        if block.size < self.block_size:
            callback_block = np.pad(block, (0, self.block_size - block.size))
        callback_time = time.perf_counter()
        if self.stream_start_wall is not None:
            callback_time = self.stream_start_wall + end / self.sample_rate
        self.analyzer._callback(
            np.asarray(callback_block[:, None], dtype=np.float32),
            block.size,
            {"callback_time": callback_time},
            None,
        )
        self.cursor = end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime music-adaptive MuJoCo humanoid dancer.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--motion-source",
        choices=("aistpp", "gmr-pkl", "bvh", "procedural"),
        default="aistpp",
        help="Dance motion source. Defaults to the checked-in AIST++ motion clip.",
    )
    parser.add_argument("--aistpp-root", type=Path, default=DEFAULT_AISTPP_ROOT)
    parser.add_argument(
        "--aistpp-motion",
        type=Path,
        default=DEFAULT_AISTPP_MOTION,
        help="Optional AIST++ .pkl override.",
    )
    parser.add_argument("--aistpp-fps", type=float, default=SMPL_FPS)
    parser.add_argument(
        "--gmr-motion",
        type=Path,
        default=None,
        help="GMR-retargeted Unitree G1 .pkl containing dof_pos/fps/root_pos/root_rot.",
    )
    parser.add_argument(
        "--gmr-fps",
        type=float,
        default=None,
        help="Optional FPS override for --motion-source gmr-pkl.",
    )
    parser.add_argument(
        "--gmr-use-music-amplitude",
        action="store_true",
        help="Legacy GMR amplitude scaling used only with --disable-music-modulation.",
    )
    parser.add_argument(
        "--bvh-motion",
        type=Path,
        default=DEFAULT_BVH_MOTION,
        help="BVH motion file to retarget into GMR's Unitree G1 29-DoF order.",
    )
    parser.add_argument(
        "--bvh-fps",
        type=float,
        default=None,
        help="Optional FPS override for --motion-source bvh.",
    )
    parser.add_argument(
        "--bvh-neutral-frame",
        type=int,
        default=0,
        help="BVH frame used as the neutral pose before retargeting.",
    )
    parser.add_argument(
        "--bvh-use-music-amplitude",
        action="store_true",
        help="Legacy BVH amplitude scaling used only with --disable-music-modulation.",
    )
    parser.add_argument(
        "--target-robot",
        choices=("auto", "dancer", "unitree-g1"),
        default="auto",
        help="Pose target adapter. auto enables Unitree G1 mapping when G1-like actuators are detected.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument(
        "--preview-trajectory",
        action="store_true",
        help="Play the authored dance trajectory at normal speed without microphone or adaptive music mapping.",
    )
    parser.add_argument("--no-mic", action="store_true", help="Use the fallback idle/demo feature state instead of microphone input.")
    parser.add_argument(
        "--audio-input",
        type=Path,
        default=None,
        help="Use an audio file (including data/test_audio MP3/WAV files) as a realtime virtual microphone.",
    )
    parser.add_argument(
        "--audio-input-delay-sec",
        type=float,
        default=1.0,
        help="Silent virtual-microphone time before file playback, reserved for denoiser reset/calibration.",
    )
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional duration limit for smoke tests.")
    parser.add_argument("--realtime", action="store_true", help="Sleep at the MuJoCo timestep.")
    parser.add_argument("--motion-cycle-duration", type=float, default=None)
    parser.add_argument("--preview-amplitude", type=float, default=0.85, help="Fixed dance amplitude for --preview-trajectory.")
    parser.add_argument("--beats-per-cycle", type=int, default=2)
    parser.add_argument(
        "--keypoint-mode",
        choices=("auto", "aist-velocity", "fixed", "off"),
        default="auto",
        help=(
            "How music beats choose dance phases. auto uses AIST++ velocity valleys for AIST++ motions "
            "and pose salience otherwise; aist-velocity requires an AIST++ source."
        ),
    )
    parser.add_argument(
        "--keypoint-phases",
        type=str,
        default="0.0,0.5",
        help="Comma-separated phases used by --keypoint-mode fixed.",
    )
    parser.add_argument(
        "--keypoint-min-spacing-sec",
        type=float,
        default=0.28,
        help="Minimum authored-motion time between auto-detected keypoints.",
    )
    parser.add_argument(
        "--keypoint-prominence",
        type=float,
        default=0.18,
        help="Minimum normalized prominence for auto-detected keypoints.",
    )
    parser.add_argument(
        "--keypoint-smoothing-sec",
        type=float,
        default=0.08,
        help="Gaussian smoothing width for AIST++ velocity-valley detection.",
    )
    parser.add_argument(
        "--keypoint-boundary-sec",
        type=float,
        default=0.10,
        help="Ignore velocity valleys this close to an AIST++ clip boundary.",
    )
    parser.add_argument(
        "--keypoint-max-count",
        type=int,
        default=None,
        help="Maximum number of auto-detected keypoints per dance cycle. Defaults to the rounded motion duration in seconds.",
    )
    parser.add_argument("--speed-min", type=float, default=0.55)
    parser.add_argument("--speed-max", type=float, default=1.9)
    parser.add_argument("--amp-min", type=float, default=0.35)
    parser.add_argument("--amp-max", type=float, default=1.25)
    parser.add_argument("--pose-gain", type=float, default=1.0)
    parser.add_argument("--accent-gain", type=float, default=0.55)
    parser.add_argument("--music-modulation-strength", type=float, default=1.0)
    parser.add_argument(
        "--disable-music-modulation",
        action="store_true",
        help="Disable the unified music-to-pose modulation layer.",
    )
    parser.add_argument("--feature-smoothing-tau", type=float, default=0.18)
    parser.add_argument("--smoothing-tau", type=float, default=0.22)
    parser.add_argument("--tempo-timeout", type=float, default=2.0)
    parser.add_argument("--accent-duration", type=float, default=0.16)
    parser.add_argument("--status-interval", type=float, default=1.0)
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
    parser.add_argument("--min-beat-period", type=float, default=0.25)
    parser.add_argument("--max-beat-period", type=float, default=2.0)
    parser.add_argument("--mic-refractory-sec", type=float, default=0.18)
    parser.add_argument(
        "--beat-confidence-threshold",
        type=float,
        default=0.35,
        help="Ignore PLP beats below this confidence before applying pose/keypoint alignment.",
    )
    parser.add_argument(
        "--beat-keypoint-interval-ratio",
        type=float,
        default=1.0,
        help=(
            "Reject beats closer than this fraction of the next authored keypoint interval at --speed-max."
        ),
    )
    parser.add_argument(
        "--beat-selection-mode",
        choices=("adaptive", "every"),
        default="adaptive",
        help="Prefer strong beats only when raw beat timing would exceed the motion speed limit.",
    )
    parser.add_argument(
        "--beat-contrast-weight",
        type=float,
        default=0.5,
        help="Weight of local beat loudness/onset contrast versus PLP confidence during adaptive selection.",
    )
    return parser.parse_args()


def resolve_aistpp_motion(aistpp_root: Path, motion_override: Optional[Path]) -> Path:
    if motion_override is not None:
        motion_path = motion_override
        if not motion_path.is_absolute():
            motion_path = ROOT / motion_path
        if not motion_path.exists():
            raise FileNotFoundError(f"AIST++ motion file not found: {motion_path}")
        return motion_path

    split_file = aistpp_root / "train.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"AIST++ train split not found: {split_file}")

    first_motion_name = None
    for line in split_file.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            first_motion_name = name
            break
    if first_motion_name is None:
        raise ValueError(f"AIST++ train split is empty: {split_file}")

    filename = first_motion_name if first_motion_name.endswith(".pkl") else f"{first_motion_name}.pkl"
    motion_path = aistpp_root / "motions" / filename
    if not motion_path.exists():
        alternate_path = aistpp_root / "train" / filename
        if alternate_path.exists():
            return alternate_path
        raise FileNotFoundError(f"AIST++ train motion not found: {motion_path}")
    return motion_path


def resolve_workspace_path(path: Path, description: str) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _logical_g1_name(name: str) -> str:
    return name.removesuffix("_joint")


def make_sampler(
    args: argparse.Namespace,
) -> HumanoidDanceSampler | AistppMotionSampler | GmrUnitreeG1MotionSampler | BvhUnitreeG1MotionSampler:
    if args.motion_source == "procedural":
        return HumanoidDanceSampler(pose_gain=args.pose_gain, accent_gain=args.accent_gain)

    if args.motion_source == "gmr-pkl":
        if args.gmr_motion is None:
            raise ValueError("--motion-source gmr-pkl requires --gmr-motion.")
        motion_path = resolve_workspace_path(args.gmr_motion, "GMR Unitree G1 motion file")
        sampler = GmrUnitreeG1MotionSampler(
            motion_path=motion_path,
            fps_override=args.gmr_fps,
            pose_gain=args.pose_gain,
            accent_gain=args.accent_gain,
            use_music_amplitude=args.disable_music_modulation and args.gmr_use_music_amplitude,
        )
        print(
            f"Loaded GMR Unitree G1 motion: {motion_path.name} "
            f"({len(sampler.frames)} frames, {sampler.duration:.2f}s, {len(sampler.dof_names)} DoF)."
        )
        return sampler

    if args.motion_source == "bvh":
        motion_path = resolve_workspace_path(args.bvh_motion, "BVH motion file")
        sampler = BvhUnitreeG1MotionSampler(
            motion_path=motion_path,
            fps_override=args.bvh_fps,
            pose_gain=args.pose_gain,
            accent_gain=args.accent_gain,
            use_music_amplitude=args.disable_music_modulation and args.bvh_use_music_amplitude,
            neutral_frame=args.bvh_neutral_frame,
        )
        print(
            f"Loaded BVH motion as GMR Unitree G1 pose stream: {motion_path.name} "
            f"({len(sampler.frames)} frames, {sampler.duration:.2f}s, {len(sampler.dof_names)} DoF)."
        )
        return sampler

    motion_path = resolve_aistpp_motion(args.aistpp_root, args.aistpp_motion)
    sampler = AistppMotionSampler(
        motion_path=motion_path,
        fps=args.aistpp_fps,
        pose_gain=args.pose_gain,
        accent_gain=args.accent_gain,
    )
    print(f"Loaded AIST++ train motion: {motion_path.name} ({len(sampler.frames)} frames, {sampler.duration:.2f}s).")
    return sampler


def make_pose_adapter(
    args: argparse.Namespace,
    player: MujocoHumanoidPlayer,
    sampler: HumanoidDanceSampler | AistppMotionSampler | GmrUnitreeG1MotionSampler | BvhUnitreeG1MotionSampler,
) -> UnitreeG1DanceAdapter | UnitreeG1JointPoseAdapter | None:
    if getattr(sampler, "pose_space", "dancer") == "unitree-g1":
        adapter = UnitreeG1JointPoseAdapter(player.actuator_names)
        if adapter.resolved_count == 0:
            print("Warning: GMR Unitree G1 motion loaded, but no compatible actuators were found.")
        else:
            print(f"Using GMR Unitree G1 joint adapter ({adapter.resolved_count} actuator names resolved).")
        return adapter

    if args.target_robot == "dancer":
        return None

    looks_like_g1 = UnitreeG1DanceAdapter.looks_like_unitree_g1(player.actuator_names)
    if args.target_robot == "auto" and not looks_like_g1:
        return None

    adapter = UnitreeG1DanceAdapter(player.actuator_names)
    if adapter.resolved_count == 0:
        print("Warning: --target-robot unitree-g1 selected, but no Unitree G1-style actuators were found.")
        return adapter

    print(f"Using Unitree G1 dance adapter ({adapter.resolved_count} actuator names resolved).")
    return adapter


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


def resolve_controller_keypoints(
    args: argparse.Namespace,
    sampler: HumanoidDanceSampler | AistppMotionSampler | GmrUnitreeG1MotionSampler | BvhUnitreeG1MotionSampler,
    motion_cycle_duration: float,
) -> tuple[tuple[float, ...], int, bool]:
    if args.keypoint_mode == "off":
        beats_per_cycle = max(args.beats_per_cycle, 1)
        print(f"Motion keypoints disabled; using {beats_per_cycle} uniform beat phases per cycle.")
        return (), beats_per_cycle, False

    if args.keypoint_mode == "fixed":
        keypoint_phases = parse_keypoint_phases(args.keypoint_phases)
        if not keypoint_phases:
            keypoint_phases = DEFAULT_FALLBACK_PHASES
            print("Warning: --keypoint-phases was empty; falling back to phases 0.000, 0.500.")
        print(
            f"Using fixed motion keypoints ({len(keypoint_phases)} beats/cycle): "
            f"{_format_phases(keypoint_phases)}"
        )
        return keypoint_phases, max(len(keypoint_phases), 1), True

    keypoint_max_count = (
        args.keypoint_max_count
        if args.keypoint_max_count is not None
        else default_keypoint_count(motion_cycle_duration)
    )
    use_aist_velocity = isinstance(sampler, AistppMotionSampler) and args.keypoint_mode in (
        "auto",
        "aist-velocity",
    )
    if args.keypoint_mode == "aist-velocity" and not isinstance(sampler, AistppMotionSampler):
        raise ValueError("--keypoint-mode aist-velocity requires --motion-source aistpp.")

    if use_aist_velocity:
        result = detect_aistpp_velocity_keypoints(
            smpl_poses=sampler.frames,
            smpl_trans=sampler.translations,
            smpl_scaling=sampler.scaling,
            fps=sampler.fps,
            smoothing_sec=args.keypoint_smoothing_sec,
            min_spacing_sec=args.keypoint_min_spacing_sec,
            prominence=args.keypoint_prominence,
            max_count=keypoint_max_count,
            boundary_sec=args.keypoint_boundary_sec,
        )
        if len(result.phases) < 2:
            print(
                "Warning: AIST++ velocity-valley detection found fewer than two keypoints; "
                f"falling back to phases {_format_phases(DEFAULT_FALLBACK_PHASES)} ({result.reason})."
            )
            return DEFAULT_FALLBACK_PHASES, len(DEFAULT_FALLBACK_PHASES), True

        frame_text = ", ".join(str(frame) for frame in result.frame_indices)
        print(
            f"AIST++ velocity-valley keypoints ({len(result.phases)} beats/cycle): "
            f"{_format_phases(result.phases)}; frames: {frame_text}"
        )
        return result.phases, len(result.phases), True

    neutral_features = FeatureState(is_active=True)
    phases, poses = sample_pose_sequence(
        sampler=sampler,
        duration=motion_cycle_duration,
        features=neutral_features,
    )
    result = detect_motion_keypoints(
        phases=phases,
        poses=poses,
        duration=motion_cycle_duration,
        min_spacing_sec=args.keypoint_min_spacing_sec,
        prominence=args.keypoint_prominence,
        max_count=keypoint_max_count,
        fallback_phases=DEFAULT_FALLBACK_PHASES,
    )
    if result.fallback_used:
        print(
            f"Warning: auto keypoint detection fell back to default phases "
            f"({_format_phases(result.phases)}): {result.reason}."
        )
    else:
        print(
            f"Auto-detected motion keypoints ({len(result.phases)} beats/cycle): "
            f"{_format_phases(result.phases)}"
        )
    return result.phases, max(len(result.phases), 1), True


def _format_phases(phases: tuple[float, ...]) -> str:
    return ", ".join(f"{phase:.3f}" for phase in phases)


def print_status(
    controller: AdaptiveMotionController,
    features: FeatureState,
    last_frame: MusicFrame | None,
    recent_detected_beat_frame: MusicFrame | None = None,
    recent_accepted_beat_frame: MusicFrame | None = None,
) -> None:
    bpm = controller.estimated_bpm
    bpm_text = f"{bpm:.1f}" if bpm is not None else "--"
    state = "music" if features.is_active else "demo"
    rms_text = f"{last_frame.rms:.4f}/{last_frame.gate_rms:.4f}" if last_frame is not None else "--"
    detected_text = "1" if recent_detected_beat_frame is not None else "0"
    accepted_text = "1" if recent_accepted_beat_frame is not None else "0"
    confidence_text = (
        f"{recent_detected_beat_frame.beat_confidence:.2f}"
        if recent_detected_beat_frame is not None
        else "0.00"
    )
    contrast_text = (
        f"{recent_detected_beat_frame.beat_contrast:.2f}"
        if recent_detected_beat_frame is not None
        else "0.00"
    )
    print(
        f"{state} | bpm={bpm_text} | speed={controller.speed_multiplier:.2f}x | "
        f"amp={controller.amplitude_scale:.2f} | rms/gate={rms_text} | "
        f"beat={detected_text} accepted={accepted_text} conf={confidence_text} "
        f"contrast={contrast_text} result={controller.last_beat_rejection_reason} | "
        f"bright={features.brightness:.2f} | "
        f"bands={features.low_energy:.2f}/{features.mid_energy:.2f}/{features.high_energy:.2f} | "
        f"rhythm={features.rhythm_density:.2f} | offbeat={features.offbeat_ratio:.2f}"
    )


def run_trajectory_preview(
    args: argparse.Namespace,
    sampler: HumanoidDanceSampler | AistppMotionSampler | GmrUnitreeG1MotionSampler | BvhUnitreeG1MotionSampler,
    player: MujocoHumanoidPlayer,
    pose_adapter: UnitreeG1DanceAdapter | UnitreeG1JointPoseAdapter | None,
    modulator: MusicPoseModulator | None,
) -> None:
    features = FeatureState(
        rms_norm=args.preview_amplitude,
        brightness=0.35,
        low_energy=0.55,
        mid_energy=0.45,
        high_energy=0.25,
        rhythm_density=0.35,
        is_active=True,
    )
    cycle_duration = max(args.motion_cycle_duration or sampler.duration, 1e-6)
    amplitude = max(args.preview_amplitude, 0.0)

    player.start()
    print(
        f"Previewing authored dance trajectory at normal speed "
        f"({cycle_duration:.2f}s per cycle, amplitude={amplitude:.2f})."
    )

    start = time.perf_counter()
    last_status = start
    try:
        while player.is_running():
            now = time.perf_counter()
            elapsed = now - start
            if args.max_seconds is not None and elapsed >= args.max_seconds:
                break

            phase = (elapsed / cycle_duration) % 1.0
            pose = sampler.sample(phase, amplitude, accent=0.0, features=features)
            if modulator is not None:
                pose = modulator.modulate(pose, features, phase=phase, amplitude=amplitude, accent=0.0)
            if pose_adapter is not None:
                pose = pose_adapter.adapt_pose(pose, features)
            player.set_pose(pose)
            player.step()

            if now - last_status >= args.status_interval:
                print(f"preview | phase={phase:.2f} | amp={amplitude:.2f} | speed=1.00x")
                last_status = now
    finally:
        player.stop()


def main() -> None:
    args = parse_args()
    sampler = make_sampler(args)
    motion_cycle_duration = max(args.motion_cycle_duration or sampler.duration, 1e-6)
    player = MujocoHumanoidPlayer(args.model, realtime=args.realtime, headless=args.headless)
    pose_adapter = make_pose_adapter(args, player, sampler)
    modulator = None if args.disable_music_modulation else MusicPoseModulator(args.music_modulation_strength)
    if args.preview_trajectory:
        run_trajectory_preview(args, sampler, player, pose_adapter, modulator)
        return

    keypoint_phases, controller_beats_per_cycle, use_keypoints = resolve_controller_keypoints(
        args,
        sampler,
        motion_cycle_duration,
    )
    controller = AdaptiveMotionController(
        authored_cycle_duration=motion_cycle_duration,
        beats_per_cycle=controller_beats_per_cycle,
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
        beat_selection_mode=args.beat_selection_mode,
        beat_contrast_weight=args.beat_contrast_weight,
    )
    file_source: FileMicrophoneSource | None = None
    analyzer = None if args.no_mic and args.audio_input is None else make_analyzer(args)
    if args.audio_input is not None:
        audio_path = args.audio_input if args.audio_input.is_absolute() else ROOT / args.audio_input
        assert analyzer is not None
        file_source = FileMicrophoneSource(
            audio_path,
            analyzer,
            startup_delay_sec=args.audio_input_delay_sec,
            throttle=not args.realtime,
        )
    features = FeatureState(rms_norm=0.45, brightness=0.35, low_energy=0.55, mid_energy=0.45, high_energy=0.25)

    player.start()
    if file_source is not None:
        file_source.start()
        print(
            f"Using realtime virtual microphone: {file_source.path} "
            f"(audio starts after {file_source.startup_delay_sec:.2f}s denoiser reset time)."
        )
    elif analyzer is not None:
        print(f"Calibrating microphone noise from the first {args.startup_calibration_sec:.2f}s of live input.")
        analyzer.start()
    else:
        controller.target_amplitude_scale = 0.75
        print("Running in --no-mic demo mode.")

    start = time.perf_counter()
    last_status = start
    last_feature_update = start
    last_frame: MusicFrame | None = None
    recent_detected_beat_frame: MusicFrame | None = None
    recent_accepted_beat_frame: MusicFrame | None = None
    try:
        while player.is_running():
            now = time.perf_counter()
            if file_source is not None:
                file_source.advance(player.dt)
                elapsed = file_source.playback_seconds
            else:
                elapsed = now - start
            if args.max_seconds is not None and elapsed >= args.max_seconds:
                break

            if analyzer is not None:
                frames = file_source.drain() if file_source is not None else analyzer.drain()
                for frame in frames:
                    last_frame = frame
                    if frame.is_beat:
                        recent_detected_beat_frame = frame
                    if controller.observe(frame):
                        recent_accepted_beat_frame = frame
                    dt = max(now - last_feature_update, player.dt)
                    alpha = 1.0 - math.exp(-dt / max(args.feature_smoothing_tau, 1e-6))
                    features.update(frame, alpha)
                    last_feature_update = now
            else:
                controller.target_amplitude_scale = 0.75

            phase, amplitude, accent, _brightness = controller.update(now)
            pose = sampler.sample(phase, amplitude, accent, features)
            if modulator is not None:
                pose = modulator.modulate(pose, features, phase=phase, amplitude=amplitude, accent=accent)
            if pose_adapter is not None:
                pose = pose_adapter.adapt_pose(pose, features)
            player.set_pose(pose)
            player.step()

            if now - last_status >= args.status_interval:
                print_status(
                    controller,
                    features,
                    last_frame,
                    recent_detected_beat_frame,
                    recent_accepted_beat_frame,
                )
                recent_detected_beat_frame = None
                recent_accepted_beat_frame = None
                last_status = now
            if file_source is not None and file_source.done:
                break
    finally:
        if file_source is not None:
            file_source.stop()
        elif analyzer is not None:
            analyzer.stop()
        player.stop()


if __name__ == "__main__":
    main()
