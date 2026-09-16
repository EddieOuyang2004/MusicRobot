from __future__ import annotations

import argparse
import json
import math
import pickle
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
from gmr_retarget_smpl_headless import (
    COLLISION_CLEARANCE_BUFFER_M,
    DEFAULT_COLLISION_MIN_DISTANCE_M,
    activate_required_collision_geoms,
    collision_geom_pairs,
)
from robot_motion import (
    RobotMotionFrame,
    RootMotionContinuity,
    normalize_wxyz,
    rotation_to_wxyz,
    slerp_wxyz,
    wxyz_to_rotation,
)
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
DEFAULT_GMR_MOTION_ROOT = Path(__file__).resolve().parents[1] / "data" / "aistpp_gmr"
DEFAULT_GMR_ROOT = Path(__file__).resolve().parents[1] / ".deps" / "GMR"
DEFAULT_GMR_PYTHON = Path(__file__).resolve().parents[1] / ".venv-gmr" / "Scripts" / "python.exe"
DEFAULT_BVH_MOTION = Path(__file__).resolve().parents[1] / "data" / "lafan1_dance" / "dance1_subject1.bvh"
SMPL_FPS = 60.0
AIST_TO_MUJOCO = np.asarray(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=np.float64,
)
AIST_TO_MUJOCO_ROTATION = Rotation.from_matrix(AIST_TO_MUJOCO)


def gmr_generation_hint(motion_path: Path) -> str:
    builder = Path(__file__).with_name("build_aistpp_gmr_dataset.py")
    return (
        f'"{DEFAULT_GMR_PYTHON}" "{builder}" '
        f'--gmr-root "{DEFAULT_GMR_ROOT}" --gmr-python "{DEFAULT_GMR_PYTHON}" '
        f'--motion "{motion_path}"'
    )


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


class RealtimeLoopScheduler:
    """Absolute-deadline scheduler that never accumulates catch-up iterations."""

    def __init__(self, rate_hz: float, enabled: bool) -> None:
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("--control-rate-hz must be a positive finite value.")
        self.period = 1.0 / float(rate_hz)
        self.enabled = bool(enabled)
        self.next_deadline: float | None = None
        self.work_seconds: deque[float] = deque(maxlen=200_000)
        self.output_interval_seconds: deque[float] = deque(maxlen=200_000)
        self.deadline_misses = 0
        self.skipped_periods = 0
        self.iterations = 0
        self.last_output_time: float | None = None

    def time_since_last_output(self, now: float | None = None) -> float:
        """Return the real output interval available to the next limiter step."""

        current = time.perf_counter() if now is None else float(now)
        if self.last_output_time is None:
            return self.period
        return max(current - self.last_output_time, 1e-9)

    def record_output(self, emitted_at: float | None = None) -> float:
        """Record an emitted command and return its actual wall-clock interval."""

        current = time.perf_counter() if emitted_at is None else float(emitted_at)
        interval = self.time_since_last_output(current)
        if self.last_output_time is not None:
            self.output_interval_seconds.append(interval)
        self.last_output_time = current
        return interval

    def wait(self, work_started: float) -> None:
        now = time.perf_counter()
        self.work_seconds.append(max(now - work_started, 0.0))
        self.iterations += 1
        if not self.enabled:
            return
        if self.next_deadline is None:
            self.next_deadline = work_started + self.period
        else:
            self.next_deadline += self.period
        if now < self.next_deadline:
            time.sleep(self.next_deadline - now)
            return
        self.deadline_misses += 1
        missed = math.floor((now - self.next_deadline) / self.period) + 1
        self.skipped_periods += missed
        # Do not emit a catch-up frame immediately after an overrun. Restart the
        # release clock so the next cycle has a full period available.
        self.next_deadline = now + self.period
        time.sleep(self.period)

    def reset_statistics(self) -> None:
        """Start a new measurement interval without disturbing wall-clock scheduling."""

        self.work_seconds.clear()
        self.output_interval_seconds.clear()
        self.deadline_misses = 0
        self.skipped_periods = 0
        self.iterations = 0

    def summary(self) -> dict[str, float | int]:
        samples_ms = np.asarray(self.work_seconds, dtype=np.float64) * 1_000.0

        def percentile(quantile: float) -> float:
            return float(np.percentile(samples_ms, quantile)) if samples_ms.size else 0.0

        intervals_ms = np.asarray(
            self.output_interval_seconds, dtype=np.float64
        ) * 1_000.0

        def interval_percentile(quantile: float) -> float:
            return (
                float(np.percentile(intervals_ms, quantile))
                if intervals_ms.size
                else 0.0
            )

        return {
            "control_rate_hz": 1.0 / self.period,
            "iterations": self.iterations,
            "deadline_misses": self.deadline_misses,
            "deadline_miss_ratio": self.deadline_misses / max(self.iterations, 1),
            "skipped_control_periods": self.skipped_periods,
            "work_ms_p50": percentile(50.0),
            "work_ms_p95": percentile(95.0),
            "work_ms_p99": percentile(99.0),
            "work_ms_max": float(np.max(samples_ms)) if samples_ms.size else 0.0,
            "output_interval_samples": int(intervals_ms.size),
            "output_interval_ms_min": (
                float(np.min(intervals_ms)) if intervals_ms.size else 0.0
            ),
            "output_interval_ms_p50": interval_percentile(50.0),
            "output_interval_ms_p95": interval_percentile(95.0),
            "output_interval_ms_p99": interval_percentile(99.0),
            "output_interval_ms_max": (
                float(np.max(intervals_ms)) if intervals_ms.size else 0.0
            ),
        }


def write_timing_report(
    path: Path | None,
    scheduler: RealtimeLoopScheduler,
    analyzer: RealtimeMusicAnalyzer | None,
    player: "MujocoHumanoidPlayer",
    extra: dict[str, Any] | None = None,
) -> None:
    if path is None:
        return
    report = scheduler.summary()
    report.update(
        {
            "collision_checks": player.collision_check_count,
            "collision_projections": player.collision_projection_count,
            "collision_anchor_recoveries": player.collision_anchor_recovery_count,
        }
    )
    viewer_samples_ms = np.asarray(player.viewer_render_seconds, dtype=np.float64) * 1_000.0
    report.update(
        {
            "viewer_render_samples": int(viewer_samples_ms.size),
            "viewer_render_ms_p50": (
                float(np.percentile(viewer_samples_ms, 50.0))
                if viewer_samples_ms.size
                else 0.0
            ),
            "viewer_render_ms_p95": (
                float(np.percentile(viewer_samples_ms, 95.0))
                if viewer_samples_ms.size
                else 0.0
            ),
            "viewer_render_ms_p99": (
                float(np.percentile(viewer_samples_ms, 99.0))
                if viewer_samples_ms.size
                else 0.0
            ),
            "viewer_render_ms_max": (
                float(np.max(viewer_samples_ms)) if viewer_samples_ms.size else 0.0
            ),
        }
    )
    if analyzer is not None:
        report.update(
            {
                "plp_submitted": analyzer.analysis_submitted,
                "plp_completed": analyzer.analysis_completed,
                "plp_skipped_busy": analyzer.analysis_skipped_busy,
                "plp_errors": analyzer.analysis_errors,
                "plp_last_ms": analyzer.analysis_last_latency_sec * 1_000.0,
                "plp_max_ms": analyzer.analysis_max_latency_sec * 1_000.0,
            }
        )
    if extra:
        report.update(extra)
    report_path = path if path.is_absolute() else ROOT / path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


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

    def sample_frame(
        self,
        phase: float,
        amplitude: float,
        accent: float,
        features: FeatureState,
    ) -> RobotMotionFrame:
        return RobotMotionFrame(self.sample(phase, amplitude, accent, features))


class AistppMotionSampler:
    """Direct, dependency-free AIST++ fallback mapped into canonical G1 joints."""

    pose_space = "unitree-g1"
    SMPL_PARENTS = (
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
        9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
    )
    DIRECT_LIMITS: dict[str, tuple[float, float]] = {
        "left_hip_pitch": (-2.5307, 2.8798),
        "left_hip_roll": (-0.5236, 2.9671),
        "left_hip_yaw": (-2.7576, 2.7576),
        "left_knee": (-0.087267, 2.8798),
        "left_ankle_pitch": (-0.87267, 0.5236),
        "left_ankle_roll": (-0.2618, 0.2618),
        "right_hip_pitch": (-2.5307, 2.8798),
        "right_hip_roll": (-2.9671, 0.5236),
        "right_hip_yaw": (-2.7576, 2.7576),
        "right_knee": (-0.087267, 2.8798),
        "right_ankle_pitch": (-0.87267, 0.5236),
        "right_ankle_roll": (-0.2618, 0.2618),
        "waist_yaw": (-2.618, 2.618),
        "waist_roll": (-0.52, 0.52),
        "waist_pitch": (-0.52, 0.52),
        "left_shoulder_pitch": (-3.0892, 2.6704),
        "left_shoulder_roll": (-1.5882, 2.2515),
        "left_shoulder_yaw": (-2.618, 2.618),
        "left_elbow": (-1.0472, 2.0944),
        "left_wrist_roll": (-1.97222, 1.97222),
        "left_wrist_pitch": (-1.61443, 1.61443),
        "left_wrist_yaw": (-1.61443, 1.61443),
        "right_shoulder_pitch": (-3.0892, 2.6704),
        "right_shoulder_roll": (-2.2515, 1.5882),
        "right_shoulder_yaw": (-2.618, 2.618),
        "right_elbow": (-1.0472, 2.0944),
        "right_wrist_roll": (-1.97222, 1.97222),
        "right_wrist_pitch": (-1.61443, 1.61443),
        "right_wrist_yaw": (-1.61443, 1.61443),
    }

    def __init__(self, motion_path: Path, fps: float, pose_gain: float, accent_gain: float) -> None:
        self.motion_path = motion_path
        self.fps = max(fps, 1e-6)
        self.pose_gain = max(pose_gain, 0.0)
        self.accent_gain = max(accent_gain, 0.0)

        with motion_path.open("rb") as motion_file:
            motion = pickle.load(motion_file)
        poses = np.asarray(motion["smpl_poses"], dtype=np.float32)
        if poses.ndim == 2 and poses.shape[1] == 72:
            frames = poses.reshape((-1, 24, 3))
        elif poses.ndim == 3 and poses.shape[1:] == (24, 3):
            frames = poses
        else:
            raise ValueError(f"Expected smpl_poses with shape (N, 72) or (N, 24, 3), got {poses.shape}.")
        if len(poses) < 2:
            raise ValueError("AIST++ motion must contain at least two frames.")
        self.frames = frames
        xyzw = Rotation.from_rotvec(self.frames.reshape(-1, 3)).as_quat().reshape(-1, 24, 4)
        self.local_quaternions_wxyz = xyzw[..., [3, 0, 1, 2]]
        translations = motion.get("smpl_trans")
        self.translations = None if translations is None else np.asarray(translations, dtype=np.float64)
        if self.translations is not None and self.translations.shape != (len(self.frames), 3):
            raise ValueError(
                f"Expected smpl_trans with shape {(len(self.frames), 3)}, got {self.translations.shape}."
            )
        scaling = motion.get("smpl_scaling")
        self.scaling = None if scaling is None else float(np.asarray(scaling, dtype=np.float64).reshape(-1)[0])
        if self.scaling is not None and (not math.isfinite(self.scaling) or abs(self.scaling) <= 1e-8):
            raise ValueError("AIST++ smpl_scaling must be finite and nonzero.")
        self.root_positions, self.root_quaternions = self._build_root_trajectory()
        self.ground_offset_z = 0.0
        self.is_grounded = False
        self.duration = len(self.frames) / self.fps

    def sample(self, phase: float, amplitude: float, accent: float, features: FeatureState) -> dict[str, float]:
        return self.sample_frame(phase, amplitude, accent, features).joint_positions

    def sample_frame(
        self,
        phase: float,
        amplitude: float,
        accent: float,
        features: FeatureState,
    ) -> RobotMotionFrame:
        del amplitude, accent, features
        phase = phase % 1.0
        frame_pos = phase * len(self.frames)
        frame_a = int(math.floor(frame_pos)) % len(self.frames)
        frame_b = (frame_a + 1) % len(self.frames)
        blend = frame_pos - math.floor(frame_pos)
        frame = self._interpolate_smpl_frame(frame_a, frame_b, blend)
        pose = self._retarget(frame)

        if frame_b == 0:
            root_position = self.root_positions[frame_a].copy()
            root_quaternion = self.root_quaternions[frame_a].copy()
        else:
            root_position = (1.0 - blend) * self.root_positions[frame_a] + blend * self.root_positions[frame_b]
            root_quaternion = slerp_wxyz(
                self.root_quaternions[frame_a],
                self.root_quaternions[frame_b],
                blend,
            )

        return RobotMotionFrame(
            joint_positions={name: float(self.pose_gain * value) for name, value in pose.items()},
            root_position=root_position + np.asarray([0.0, 0.0, self.ground_offset_z]),
            root_quaternion_wxyz=root_quaternion,
        )

    def _build_root_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        if self.translations is None:
            positions = np.zeros((len(self.frames), 3), dtype=np.float64)
        else:
            scale = self.scaling if self.scaling is not None else 1.0
            relative = (self.translations - self.translations[0]) / scale
            positions = relative @ AIST_TO_MUJOCO.T

        quaternions = np.empty((len(self.frames), 4), dtype=np.float64)
        basis = AIST_TO_MUJOCO_ROTATION
        for index, root_rotvec in enumerate(self.frames[:, 0]):
            converted = basis * Rotation.from_rotvec(root_rotvec) * basis.inv()
            quaternions[index] = rotation_to_wxyz(converted)
        origin_rotation = wxyz_to_rotation(quaternions[0])
        # Rebase translation and orientation into the same initial-heading frame.
        positions = origin_rotation.inv().apply(positions)
        quaternions = np.stack(
            [rotation_to_wxyz(origin_rotation.inv() * wxyz_to_rotation(value)) for value in quaternions]
        )
        return positions, quaternions

    def _interpolate_smpl_frame(self, first: int, second: int, amount: float) -> np.ndarray:
        a = self.local_quaternions_wxyz[first].astype(np.float64, copy=True)
        b = self.local_quaternions_wxyz[second].astype(np.float64, copy=True)
        t = float(np.clip(amount, 0.0, 1.0))
        dots = np.sum(a * b, axis=1)
        negative = dots < 0.0
        b[negative] *= -1.0
        dots = np.clip(np.abs(dots), -1.0, 1.0)
        close = dots > 0.9995
        theta = np.arccos(dots)
        sin_theta = np.sin(theta)
        first_weight = np.empty_like(theta)
        second_weight = np.empty_like(theta)
        first_weight[close] = 1.0 - t
        second_weight[close] = t
        first_weight[~close] = np.sin((1.0 - t) * theta[~close]) / sin_theta[~close]
        second_weight[~close] = np.sin(t * theta[~close]) / sin_theta[~close]
        quaternions = first_weight[:, None] * a + second_weight[:, None] * b
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
        return Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_rotvec()

    @staticmethod
    def _rotation(frame: np.ndarray, joint_index: int) -> Rotation:
        return Rotation.from_rotvec(frame[joint_index])

    @classmethod
    def _euler_xyz(cls, frame: np.ndarray, joint_index: int) -> np.ndarray:
        return cls._rotation(frame, joint_index).as_euler("xyz", degrees=False)

    @staticmethod
    def _converted_rotation(rotation: Rotation) -> Rotation:
        return AIST_TO_MUJOCO_ROTATION * rotation * AIST_TO_MUJOCO_ROTATION.inv()

    @classmethod
    def _global_rotations(cls, frame: np.ndarray) -> tuple[Rotation, ...]:
        rotations: list[Rotation] = []
        for joint_index, parent_index in enumerate(cls.SMPL_PARENTS):
            local = cls._rotation(frame, joint_index)
            rotations.append(local if parent_index < 0 else rotations[parent_index] * local)
        return tuple(rotations)

    @staticmethod
    def _bone_flexion(
        global_rotations: tuple[Rotation, ...],
        proximal_joint: int,
        distal_joint: int,
        rest_direction: np.ndarray,
    ) -> float:
        direction = np.asarray(rest_direction, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        proximal = global_rotations[proximal_joint].apply(direction)
        distal = global_rotations[distal_joint].apply(direction)
        cosine = float(np.clip(np.dot(proximal, distal), -1.0, 1.0))
        return float(math.acos(cosine))

    @staticmethod
    def _morphology_compressed_flexion(angle: float, soft_limit: float, hard_limit: float) -> float:
        """Preserve ordinary human flexion and smoothly fit deep bends into G1 range."""

        value = max(float(angle), 0.0)
        if value <= soft_limit:
            return value
        span = hard_limit - soft_limit
        return float(soft_limit + span * (1.0 - math.exp(-(value - soft_limit) / span)))

    @classmethod
    def _g1_euler(cls, rotation: Rotation) -> tuple[float, float, float]:
        pitch, roll, yaw = cls._converted_rotation(rotation).as_euler("YXZ", degrees=False)
        return float(pitch), float(roll), float(yaw)

    @classmethod
    def _shoulder_angles(cls, frame: np.ndarray, side: str) -> tuple[float, float, float]:
        if side == "left":
            collar_index, shoulder_index, neutral_roll = 13, 16, -0.5 * math.pi
        else:
            collar_index, shoulder_index, neutral_roll = 14, 17, 0.5 * math.pi
        source = cls._rotation(frame, collar_index) * cls._rotation(frame, shoulder_index)
        neutral = cls._converted_rotation(Rotation.from_euler("z", neutral_roll))
        converted_source = cls._converted_rotation(source)
        pitch, roll, yaw = (neutral.inv() * converted_source).as_euler("YXZ", degrees=False)
        return float(pitch), float(roll), float(yaw)

    def _retarget(self, frame: np.ndarray) -> dict[str, float]:
        pose, _compression = self._retarget_with_diagnostics(frame)
        return pose

    def _retarget_with_diagnostics(
        self,
        frame: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        left_hip = self._g1_euler(self._rotation(frame, 1))
        right_hip = self._g1_euler(self._rotation(frame, 2))
        spine_rotation = self._rotation(frame, 3) * self._rotation(frame, 6) * self._rotation(frame, 9)
        spine = self._g1_euler(spine_rotation)
        left_ankle = self._g1_euler(self._rotation(frame, 7))
        right_ankle = self._g1_euler(self._rotation(frame, 8))
        left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw = self._shoulder_angles(frame, "left")
        right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw = self._shoulder_angles(frame, "right")
        global_rotations = self._global_rotations(frame)
        source_flexion = {
            "left_elbow": self._bone_flexion(global_rotations, 16, 18, np.asarray([1.0, 0.0, 0.0])),
            "right_elbow": self._bone_flexion(global_rotations, 17, 19, np.asarray([-1.0, 0.0, 0.0])),
            "left_knee": self._bone_flexion(global_rotations, 1, 4, np.asarray([0.0, -1.0, 0.0])),
            "right_knee": self._bone_flexion(global_rotations, 2, 5, np.asarray([0.0, -1.0, 0.0])),
        }
        mapped_flexion = {
            "left_elbow": self._morphology_compressed_flexion(source_flexion["left_elbow"], 1.75, 2.05),
            "right_elbow": self._morphology_compressed_flexion(source_flexion["right_elbow"], 1.75, 2.05),
            "left_knee": self._morphology_compressed_flexion(source_flexion["left_knee"], 2.50, 2.80),
            "right_knee": self._morphology_compressed_flexion(source_flexion["right_knee"], 2.50, 2.80),
        }
        left_wrist = self._g1_euler(self._rotation(frame, 20))
        right_wrist = self._g1_euler(self._rotation(frame, 21))

        pose = {
            "waist_yaw": float(0.70 * spine[2]),
            "waist_roll": float(0.40 * spine[1]),
            "waist_pitch": float(0.40 * spine[0]),
            "left_shoulder_pitch": left_shoulder_pitch,
            "left_shoulder_roll": left_shoulder_roll,
            "left_shoulder_yaw": left_shoulder_yaw,
            "left_elbow": mapped_flexion["left_elbow"],
            "left_wrist_roll": float(left_wrist[1]),
            "left_wrist_pitch": float(left_wrist[0]),
            "left_wrist_yaw": float(left_wrist[2]),
            "right_shoulder_pitch": right_shoulder_pitch,
            "right_shoulder_roll": right_shoulder_roll,
            "right_shoulder_yaw": right_shoulder_yaw,
            "right_elbow": mapped_flexion["right_elbow"],
            "right_wrist_roll": float(right_wrist[1]),
            "right_wrist_pitch": float(right_wrist[0]),
            "right_wrist_yaw": float(right_wrist[2]),
            "left_hip_yaw": float(left_hip[2]),
            "left_hip_roll": float(-left_hip[1]),
            "left_hip_pitch": float(left_hip[0]),
            "left_knee": mapped_flexion["left_knee"],
            "left_ankle_pitch": float(left_ankle[0]),
            "left_ankle_roll": float(left_ankle[1]),
            "right_hip_yaw": float(right_hip[2]),
            "right_hip_roll": float(right_hip[1]),
            "right_hip_pitch": float(right_hip[0]),
            "right_knee": mapped_flexion["right_knee"],
            "right_ankle_pitch": float(right_ankle[0]),
            "right_ankle_roll": float(right_ankle[1]),
        }
        compression = {
            name: (source_flexion[name], mapped_flexion[name])
            for name in source_flexion
        }
        return {name: float(value) for name, value in pose.items()}, compression


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
        if not math.isclose(float(pose_gain), 1.0, rel_tol=0.0, abs_tol=1e-12):
            print(
                f"Ignoring pose_gain={pose_gain:g} for GMR motion {motion_path.name}; "
                "GMR dof_pos is always played at its authored scale."
            )
        self.accent_gain = max(accent_gain, 0.0)
        if use_music_amplitude:
            print("Ignoring legacy GMR music-amplitude scaling; authored GMR joint values remain unchanged.")
        self.use_music_amplitude = False

        with motion_path.open("rb") as motion_file:
            motion = pickle.load(motion_file)
        if not isinstance(motion, dict):
            raise ValueError(f"GMR motion pickle must contain a dict, got {type(motion).__name__}.")
        self.motion_path = motion_path
        self.format_version = motion.get("format_version")
        self.pipeline_version = motion.get("pipeline_version")
        self.source_format = motion.get("source_format")
        self.source_motion_id = motion.get("source_motion_id")
        self.source_sha256 = motion.get("source_sha256")
        self.smpl_model_sha256 = motion.get("smpl_model_sha256")
        self.retargeter = motion.get("retargeter")
        self.retargeter_version = motion.get("retargeter_version")
        self.collision_avoidance = motion.get("collision_avoidance")
        self.mink_limits_api = motion.get("mink_limits_api")

        if self.format_version != 1 or self.pipeline_version != 4:
            raise ValueError(
                "GMR motion must use canonical format_version=1 and pipeline_version=4."
            )
        if self.source_format != "aistpp_smpl_direct":
            raise ValueError(
                "GMR motion must set source_format='aistpp_smpl_direct'; legacy BVH artifacts are refused."
            )
        if self.retargeter != "GMR":
            raise ValueError(f"GMR motion has unexpected retargeter: {self.retargeter!r}.")

        missing = {
            "fps",
            "root_pos",
            "root_rot",
            "root_rot_order",
            "dof_pos",
            "dof_names",
            "source_motion_id",
            "source_sha256",
            "smpl_model_sha256",
            "retargeter_version",
            "collision_avoidance",
            "mink_limits_api",
            "continuity_limits",
        } - set(motion)
        if missing:
            raise ValueError(f"GMR motion is missing required fields: {sorted(missing)}.")
        if (
            not isinstance(self.collision_avoidance, dict)
            or self.collision_avoidance.get("preset") != "g1_self_collision_v2"
        ):
            raise ValueError("GMR motion has missing or unsupported collision-avoidance metadata.")

        dof_pos = np.asarray(motion.get("dof_pos"), dtype=np.float32)
        if dof_pos.ndim != 2:
            raise ValueError(f"Expected GMR dof_pos with shape (N, D), got {dof_pos.shape}.")
        if len(dof_pos) < 2:
            raise ValueError("GMR motion must contain at least two frames.")
        if not np.all(np.isfinite(dof_pos)):
            raise ValueError("GMR dof_pos contains non-finite values.")

        g1_dof_count = len(UnitreeG1DanceAdapter.GMR_DOF_NAMES)
        if dof_pos.shape[1] != g1_dof_count:
            raise ValueError(f"Expected exactly {g1_dof_count} G1 DoF columns, got {dof_pos.shape[1]}.")

        dof_names = motion["dof_names"]
        loaded_names = tuple(str(name) for name in dof_names)
        if len(loaded_names) != dof_pos.shape[1]:
            raise ValueError(
                f"GMR dof_names length ({len(loaded_names)}) does not match "
                f"dof_pos columns ({dof_pos.shape[1]})."
            )
        if len(set(loaded_names)) != len(loaded_names):
            raise ValueError("GMR dof_names contains duplicate joint names.")
        canonical_names = UnitreeG1DanceAdapter.GMR_DOF_NAMES
        canonical_name_set = set(canonical_names)
        unknown_dofs = set(loaded_names) - canonical_name_set
        missing_dofs = canonical_name_set - set(loaded_names)
        if unknown_dofs:
            raise ValueError(f"GMR motion contains unknown G1 DoFs: {sorted(unknown_dofs)}.")
        if missing_dofs:
            raise ValueError(f"GMR motion is missing G1 DoFs: {sorted(missing_dofs)}.")
        if loaded_names != canonical_names:
            raise ValueError(
                "GMR dof_names must exactly match the canonical MuJoCo qpos address order."
            )
        self.dof_names = UnitreeG1DanceAdapter.GMR_DOF_NAMES
        self.frames = dof_pos

        self.root_positions = np.asarray(motion["root_pos"], dtype=np.float64)
        self.root_quaternions = np.asarray(motion["root_rot"], dtype=np.float64)
        root_rot_order = motion.get("root_rot_order")
        if root_rot_order != "wxyz":
            raise ValueError(
                "Canonical v3 GMR motion must explicitly set root_rot_order to 'wxyz'."
            )
        expected_root_position_shape = (len(self.frames), 3)
        expected_root_rotation_shape = (len(self.frames), 4)
        if self.root_positions.shape != expected_root_position_shape:
            raise ValueError(
                f"Expected GMR root_pos with shape {expected_root_position_shape}, "
                f"got {self.root_positions.shape}."
            )
        if self.root_quaternions.shape != expected_root_rotation_shape:
            raise ValueError(
                f"Expected GMR root_rot with shape {expected_root_rotation_shape}, "
                f"got {self.root_quaternions.shape}."
            )
        if not np.all(np.isfinite(self.root_positions)) or not np.all(np.isfinite(self.root_quaternions)):
            raise ValueError("GMR root trajectory contains non-finite values.")
        self.root_quaternions = np.stack(
            [normalize_wxyz(value, description=f"root_rot[{index}]") for index, value in enumerate(self.root_quaternions)]
        )
        if len(self.root_quaternions) > 1:
            signs = np.sum(self.root_quaternions[1:] * self.root_quaternions[:-1], axis=1)
            if np.any(signs < -1e-8):
                frame = int(np.flatnonzero(signs < -1e-8)[0] + 1)
                raise ValueError(f"GMR root quaternion sign discontinuity at frame {frame}.")
        self.root_positions = self.root_positions - self.root_positions[0]
        root_origin = wxyz_to_rotation(self.root_quaternions[0])
        self.root_positions = root_origin.inv().apply(self.root_positions)
        self.root_quaternions = np.stack(
            [rotation_to_wxyz(root_origin.inv() * wxyz_to_rotation(value)) for value in self.root_quaternions]
        )
        self.ground_offset_z = 0.0
        self.is_grounded = False

        fps = fps_override if fps_override is not None else float(motion.get("fps", 30.0))
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"GMR fps must be finite and positive, got {fps}.")
        self.fps = fps
        self.duration = len(self.frames) / self.fps

    def sample(self, phase: float, amplitude: float, accent: float, features: FeatureState) -> dict[str, float]:
        return self.sample_frame(phase, amplitude, accent, features).joint_positions

    def sample_frame(
        self,
        phase: float,
        amplitude: float,
        accent: float,
        features: FeatureState,
    ) -> RobotMotionFrame:
        del features
        phase = phase % 1.0
        frame_pos = phase * len(self.frames)
        frame_a = int(math.floor(frame_pos)) % len(self.frames)
        frame_b = (frame_a + 1) % len(self.frames)
        blend = frame_pos - math.floor(frame_pos)
        frame = (1.0 - blend) * self.frames[frame_a] + blend * self.frames[frame_b]

        pose = {}
        for index, name in enumerate(self.dof_names):
            if not name:
                continue
            value = float(frame[index])
            pose[name] = value
        if frame_b == 0:
            root_position = self.root_positions[frame_a].copy()
            root_quaternion = self.root_quaternions[frame_a].copy()
        else:
            root_position = (1.0 - blend) * self.root_positions[frame_a] + blend * self.root_positions[frame_b]
            root_quaternion = slerp_wxyz(
                self.root_quaternions[frame_a],
                self.root_quaternions[frame_b],
                blend,
            )
        return RobotMotionFrame(
            joint_positions=pose,
            root_position=root_position + np.asarray([0.0, 0.0, self.ground_offset_z]),
            root_quaternion_wxyz=root_quaternion,
        )


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

    def sample_frame(
        self,
        phase: float,
        amplitude: float,
        accent: float,
        features: FeatureState,
    ) -> RobotMotionFrame:
        return RobotMotionFrame(self.sample(phase, amplitude, accent, features))

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
    def __init__(
        self,
        model_path: Path,
        realtime: bool,
        headless: bool,
        viewer_rate_hz: float = 60.0,
    ) -> None:
        if not math.isfinite(viewer_rate_hz) or viewer_rate_hz <= 0.0:
            raise ValueError("--viewer-rate-hz must be a positive finite value.")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        activate_required_collision_geoms(self.model)
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
        self.floating_base_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "floating_base_joint",
        )
        self.floating_base_qpos_id = (
            None
            if self.floating_base_joint_id < 0
            else int(self.model.jnt_qposadr[self.floating_base_joint_id])
        )
        self.limit_clip_counts = {name: 0 for name in self.actuator_ids}
        self.collision_check_count = 0
        self.collision_projection_count = 0
        self.collision_anchor_recovery_count = 0
        self.collision_projection_fallback_count = 0
        self.collision_dynamics_infeasible_count = 0
        self.last_collision_projection_scale = 1.0
        self.last_collision_safe_qpos: np.ndarray | None = None
        self.collision_candidate_violation_count = 0
        expanded_collision_pairs = {
            tuple(sorted((first, second)))
            for first_group, second_group in collision_geom_pairs(self.model)
            for first in first_group
            for second in second_group
            if first != second
        }
        self.self_collision_geom_pairs = np.asarray(
            sorted(expanded_collision_pairs), dtype=np.int32
        ).reshape((-1, 2))
        self.collision_scratch = mujoco.MjData(self.model)
        self.left_foot_support_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "left_ankle_roll_link",
        )
        self.right_foot_support_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_ankle_roll_link",
        )
        support_bodies = {
            self.left_foot_support_body_id,
            self.right_foot_support_body_id,
        }
        self.foot_support_geom_ids = tuple(
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in support_bodies
            and int(self.model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        )
        self.left_foot_support_geom_ids = tuple(
            geom_id
            for geom_id in self.foot_support_geom_ids
            if int(self.model.geom_bodyid[geom_id]) == self.left_foot_support_body_id
        )
        self.right_foot_support_geom_ids = tuple(
            geom_id
            for geom_id in self.foot_support_geom_ids
            if int(self.model.geom_bodyid[geom_id]) == self.right_foot_support_body_id
        )
        self.viewer = None
        self.viewer_data: mujoco.MjData | None = None
        self.viewer_rate_hz = float(viewer_rate_hz)
        self.viewer_stop = threading.Event()
        self.viewer_thread: threading.Thread | None = None
        self.viewer_snapshot_lock = threading.Lock()
        self.viewer_qpos_snapshot: np.ndarray | None = None
        self.viewer_time_snapshot = 0.0
        self.viewer_render_seconds: list[float] = []

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
            self.viewer_data = mujoco.MjData(self.model)
            self.viewer_data.qpos[:] = self.data.qpos
            mujoco.mj_forward(self.model, self.viewer_data)
            self.viewer = mujoco.viewer.launch_passive(self.model, self.viewer_data)
            self.viewer.cam.distance = 4.2
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -12
            pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            if pelvis_id >= 0:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = pelvis_id
                self.viewer.cam.fixedcamid = -1
            self.viewer_stop.clear()
            self._publish_viewer_snapshot()
            self.viewer_thread = threading.Thread(
                target=self._viewer_sync_loop,
                name="mujoco-viewer-sync",
                daemon=True,
            )
            self.viewer_thread.start()

    def _publish_viewer_snapshot(self) -> None:
        if self.viewer is None:
            return
        with self.viewer_snapshot_lock:
            if self.viewer_qpos_snapshot is None:
                self.viewer_qpos_snapshot = np.empty_like(self.data.qpos)
            np.copyto(self.viewer_qpos_snapshot, self.data.qpos)
            self.viewer_time_snapshot = float(self.data.time)

    def _viewer_sync_loop(self) -> None:
        viewer = self.viewer
        viewer_data = self.viewer_data
        if viewer is None or viewer_data is None:
            return
        period = 1.0 / self.viewer_rate_hz
        next_sync = time.perf_counter()
        while not self.viewer_stop.is_set() and viewer.is_running():
            render_started = time.perf_counter()
            with self.viewer_snapshot_lock:
                snapshot = (
                    None
                    if self.viewer_qpos_snapshot is None
                    else self.viewer_qpos_snapshot.copy()
                )
                snapshot_time = self.viewer_time_snapshot
            if snapshot is not None:
                viewer_data.qpos[:] = snapshot
                viewer_data.time = snapshot_time
                mujoco.mj_forward(self.model, viewer_data)
            viewer.sync()
            self.viewer_render_seconds.append(
                max(time.perf_counter() - render_started, 0.0)
            )
            next_sync += period
            delay = next_sync - time.perf_counter()
            if delay <= 0.0:
                next_sync = time.perf_counter()
                continue
            self.viewer_stop.wait(delay)

    def is_running(self) -> bool:
        return self.viewer.is_running() if self.viewer is not None else True

    def root_frame(self) -> RobotMotionFrame:
        if self.floating_base_qpos_id is None:
            return RobotMotionFrame({})
        start = self.floating_base_qpos_id
        return RobotMotionFrame(
            {},
            self.data.qpos[start : start + 3].copy(),
            normalize_wxyz(self.data.qpos[start + 3 : start + 7].copy()),
        )

    def _write_frame(self, data: mujoco.MjData, frame: RobotMotionFrame, *, count_limits: bool) -> None:
        data.ctrl.fill(0.0)
        data.qvel.fill(0.0)
        if (
            self.floating_base_qpos_id is not None
            and frame.root_position is not None
            and frame.root_quaternion_wxyz is not None
        ):
            start = self.floating_base_qpos_id
            root_position = np.asarray(frame.root_position, dtype=np.float64)
            if root_position.shape != (3,) or not np.all(np.isfinite(root_position)):
                raise ValueError(f"Root position must contain three finite values, got {root_position}.")
            data.qpos[start : start + 3] = root_position
            data.qpos[start + 3 : start + 7] = normalize_wxyz(frame.root_quaternion_wxyz)

        for name, value in frame.joint_positions.items():
            actuator_id = self.actuator_ids.get(name)
            qpos_id = self.actuator_joint_qpos_ids.get(name, self.joint_qpos_ids.get(name))
            if actuator_id is None:
                continue
            if qpos_id is not None:
                joint_range = self.actuator_joint_ranges.get(name)
                target = float(value)
                qpos_target = target if joint_range is None else float(np.clip(target, joint_range[0], joint_range[1]))
                if count_limits and abs(qpos_target - target) > 1e-9:
                    self.limit_clip_counts[name] += 1
                data.qpos[qpos_id] = qpos_target

    def set_frame(self, frame: RobotMotionFrame) -> None:
        self._write_frame(self.data, frame, count_limits=True)

    def _has_self_clearance_violation(
        self,
        data: mujoco.MjData,
        minimum_distance: float,
        tolerance: float = 1e-7,
    ) -> bool:
        mujoco.mj_forward(self.model, data)
        return self._has_self_clearance_violation_after_forward(
            data, minimum_distance, tolerance
        )

    def _has_self_clearance_violation_after_forward(
        self,
        data: mujoco.MjData,
        minimum_distance: float,
        tolerance: float = 1e-7,
    ) -> bool:
        """Check clearance when ``mj_forward`` has already populated geometry."""

        pairs = self.self_collision_geom_pairs
        if not len(pairs):
            return False
        first = pairs[:, 0]
        second = pairs[:, 1]
        center_distance = np.linalg.norm(data.geom_xpos[first] - data.geom_xpos[second], axis=1)
        sphere_clearance = center_distance - (
            self.model.geom_rbound[first] + self.model.geom_rbound[second]
        )
        distance_limit = max(float(minimum_distance) + tolerance, 1e-4)
        from_to = np.empty(6, dtype=np.float64)
        for pair_index in np.flatnonzero(sphere_clearance < distance_limit):
            geom_a, geom_b = (int(value) for value in pairs[pair_index])
            distance = mujoco.mj_geomDistance(
                self.model,
                data,
                geom_a,
                geom_b,
                distance_limit,
                from_to,
            )
            if float(distance) < float(minimum_distance) - tolerance:
                return True
        return False

    def project_self_collision_safe(
        self,
        frame: RobotMotionFrame,
        minimum_distance: float = (
            DEFAULT_COLLISION_MIN_DISTANCE_M
            + COLLISION_CLEARANCE_BUFFER_M
            + 0.0005
        ),
        iterations: int = 8,
        minimum_dynamic_scale: float = 0.0,
    ) -> RobotMotionFrame:
        """Scale a pose toward the last safe output and verify the exact result.

        ``minimum_distance`` includes a small enforcement margin.  The margin is
        deliberately soft: when preserving it would require a discontinuous jump
        or violate the output dynamics bounds, the exact emitted-pose contract is
        still enforced without the extra 0.5 mm headroom.
        """
        self.collision_check_count += 1
        self.last_collision_projection_scale = 1.0
        audit_distance = max(float(minimum_distance) - 0.0005, 0.0)
        scratch = self.collision_scratch
        scratch.qpos[:] = self.data.qpos
        self._write_frame(scratch, frame, count_limits=False)
        candidate_qpos = scratch.qpos.copy()
        if not self._has_self_clearance_violation(scratch, minimum_distance):
            self.last_collision_safe_qpos = candidate_qpos
            return frame

        self.collision_candidate_violation_count += 1
        candidate_contract_safe = not self._has_self_clearance_violation(
            scratch, audit_distance
        )
        anchor_qpos = self.data.qpos.copy()
        projection_distance = float(minimum_distance)
        scratch.qpos[:] = anchor_qpos
        if self._has_self_clearance_violation(scratch, minimum_distance):
            if not self._has_self_clearance_violation(scratch, audit_distance):
                # The current command is contract-safe but has used some of the
                # optional clearance headroom.  Keep it as the interpolation
                # anchor; replacing it with an older margin-safe pose creates an
                # unbounded joint discontinuity.
                projection_distance = audit_distance
            else:
                fallback = self.last_collision_safe_qpos
                if fallback is not None:
                    anchor_qpos = fallback.copy()
                    scratch.qpos[:] = anchor_qpos
                if fallback is None or self._has_self_clearance_violation(
                    scratch,
                    audit_distance,
                ):
                    anchor_qpos = self.data.qpos.copy()
                    anchor_qpos[7:] = 0.0
                    scratch.qpos[:] = anchor_qpos
                if not self._has_self_clearance_violation(
                    scratch, audit_distance
                ) and self._has_self_clearance_violation(
                    scratch, minimum_distance
                ):
                    projection_distance = audit_distance
            if self._has_self_clearance_violation(scratch, audit_distance):
                # Keep the realtime loop alive even when an incompatible model
                # provides no safe neutral anchor.  The policy benchmark will
                # force the default mode to off if this path is observed.
                self.collision_anchor_recovery_count += 1
                return frame
            self.collision_anchor_recovery_count += 1

        safe_amount = 0.0
        unsafe_amount = 1.0
        probe = candidate_qpos.copy()
        for _iteration in range(max(int(iterations), 1)):
            amount = 0.5 * (safe_amount + unsafe_amount)
            probe[:] = candidate_qpos
            probe[:3] = anchor_qpos[:3] + amount * (
                candidate_qpos[:3] - anchor_qpos[:3]
            )
            probe[3:7] = slerp_wxyz(
                anchor_qpos[3:7], candidate_qpos[3:7], amount
            )
            probe[7:] = anchor_qpos[7:] + amount * (candidate_qpos[7:] - anchor_qpos[7:])
            scratch.qpos[:] = probe
            if self._has_self_clearance_violation(scratch, projection_distance):
                unsafe_amount = amount
            else:
                safe_amount = amount
        probe[:] = candidate_qpos
        probe[:3] = anchor_qpos[:3] + safe_amount * (
            candidate_qpos[:3] - anchor_qpos[:3]
        )
        probe[3:7] = slerp_wxyz(
            anchor_qpos[3:7], candidate_qpos[3:7], safe_amount
        )
        probe[7:] = anchor_qpos[7:] + safe_amount * (candidate_qpos[7:] - anchor_qpos[7:])

        minimum_dynamic_scale = float(np.clip(minimum_dynamic_scale, 0.0, 1.0))
        if safe_amount + 1e-12 < minimum_dynamic_scale:
            if candidate_contract_safe:
                # Preserve the already-limited command when only the optional
                # margin conflicts with dynamics.  This keeps both hard safety
                # contracts (clearance and joint derivatives) satisfiable.
                probe[:] = candidate_qpos
                safe_amount = 1.0
            else:
                # No point on the contract-safe segment also satisfies the
                # acceleration interval. Clearance takes precedence; telemetry
                # records the infeasibility rather than hiding it.
                probe[:] = anchor_qpos
                safe_amount = 0.0
                self.collision_dynamics_infeasible_count += 1

        scratch.qpos[:] = probe
        if self._has_self_clearance_violation(scratch, audit_distance):
            fallback_candidates = [anchor_qpos]
            if self.last_collision_safe_qpos is not None:
                fallback_candidates.append(self.last_collision_safe_qpos)
            neutral_qpos = self.data.qpos.copy()
            neutral_qpos[7:] = 0.0
            fallback_candidates.append(neutral_qpos)
            recovered = False
            for fallback in fallback_candidates:
                scratch.qpos[:] = fallback
                if not self._has_self_clearance_violation(
                    scratch, audit_distance
                ):
                    probe[:] = fallback
                    safe_amount = 0.0
                    recovered = True
                    break
            if not recovered:
                # No compatible hard-clearance pose exists for this model/root.
                # Keep the loop alive and expose the residual in final telemetry.
                self.collision_projection_fallback_count += 1
                return frame
            # Validate the exact selected fallback rather than trusting a stale
            # cached pose.  This is intentionally redundant: the final command
            # safety contract must not depend on cache history.
            scratch.qpos[:] = probe
            if self._has_self_clearance_violation(scratch, audit_distance):
                self.collision_projection_fallback_count += 1
                return frame
            self.collision_projection_fallback_count += 1
        projected_joints = {
            name: float(probe[qpos_id])
            for name, qpos_id in self.actuator_joint_qpos_ids.items()
        }
        self.last_collision_safe_qpos = probe.copy()
        self.last_collision_projection_scale = safe_amount
        self.collision_projection_count += 1
        return RobotMotionFrame(
            joint_positions=projected_joints,
            root_position=probe[:3].copy(),
            root_quaternion_wxyz=normalize_wxyz(probe[3:7]),
        )

    def apply_collision_policy(
        self,
        frame: RobotMotionFrame,
        mode: str,
        *,
        runtime_modified: bool,
        minimum_dynamic_scale: float = 0.0,
    ) -> RobotMotionFrame:
        if mode == "off" or (mode == "auto" and not runtime_modified):
            return frame
        return self.project_self_collision_safe(
            frame, minimum_dynamic_scale=minimum_dynamic_scale
        )

    def support_height(self, data: mujoco.MjData | None = None) -> float:
        target = self.data if data is None else data
        return min(
            float(target.geom_xpos[geom_id, 2] - self.model.geom_size[geom_id, 0])
            for geom_id in self.foot_support_geom_ids
        )

    def foot_support_height(
        self,
        geom_ids: tuple[int, ...],
        data: mujoco.MjData | None = None,
    ) -> float:
        target = self.data if data is None else data
        return min(
            float(target.geom_xpos[geom_id, 2] - self.model.geom_size[geom_id, 0])
            for geom_id in geom_ids
        )

    def ground_sampler(
        self,
        sampler: object,
        pose_adapter: UnitreeG1DanceAdapter | UnitreeG1JointPoseAdapter | None,
        *,
        cooperative_yield: bool = False,
    ) -> float:
        if not hasattr(sampler, "ground_offset_z") or not hasattr(sampler, "frames"):
            return 0.0
        if len(self.foot_support_geom_ids) != 8:
            raise ValueError(
                "Expected eight spherical G1 foot support geoms under the ankle-roll bodies, "
                f"found {len(self.foot_support_geom_ids)}."
            )
        if bool(getattr(sampler, "is_grounded", False)):
            return float(getattr(sampler, "ground_offset_z"))
        setattr(sampler, "ground_offset_z", 0.0)
        scratch = mujoco.MjData(self.model)
        features = FeatureState()
        frame_count = len(getattr(sampler, "frames"))
        minimum = math.inf
        support_heights = np.empty((frame_count, 2), dtype=np.float64)
        adapted_joint_frames: list[dict[str, float]] = []
        for frame_index in range(frame_count):
            frame = sampler.sample_frame(frame_index / frame_count, 1.0, 0.0, features)
            joints = frame.joint_positions
            if pose_adapter is not None:
                joints = pose_adapter.adapt_pose(joints, features)
            adapted_joint_frames.append(
                {name: float(value) for name, value in joints.items()}
            )
            self._write_frame(scratch, frame.with_joint_positions(joints), count_limits=False)
            mujoco.mj_forward(self.model, scratch)
            left_height = self.foot_support_height(self.left_foot_support_geom_ids, scratch)
            right_height = self.foot_support_height(self.right_foot_support_geom_ids, scratch)
            support_heights[frame_index] = (left_height, right_height)
            minimum = min(minimum, left_height, right_height)
            if cooperative_yield:
                # Preparation is intentionally lower priority than the control
                # loop. Yielding here prevents a long Python/MuJoCo scan from
                # monopolising the interpreter on large clips.
                time.sleep(0.001)
        if not math.isfinite(minimum):
            raise ValueError("Could not compute a finite foot support height for the motion.")
        offset = -minimum
        setattr(sampler, "ground_offset_z", offset)
        setattr(sampler, "is_grounded", True)
        grounded_heights = support_heights + offset
        fps = max(float(getattr(sampler, "fps", 60.0)), 1e-6)
        vertical_speed = np.zeros_like(grounded_heights)
        if frame_count > 1:
            vertical_speed[1:] = np.diff(grounded_heights, axis=0) * fps
            vertical_speed[0] = vertical_speed[1]
        foot_contacts = (
            (grounded_heights <= 0.015)
            & (np.abs(vertical_speed) <= 0.20)
        )
        setattr(sampler, "foot_support_heights", grounded_heights)
        setattr(sampler, "foot_contacts", foot_contacts)
        setattr(sampler, "adapted_joint_frames", tuple(adapted_joint_frames))
        print(f"Grounded {sampler_source_id(sampler)} with constant root Z offset {offset:+.6f} m.")
        return offset

    def set_pose(self, pose: dict[str, float]) -> None:
        """Backward-compatible joint-only kinematic target."""

        self.set_frame(RobotMotionFrame(pose))

    def step(self, elapsed: float | None = None) -> None:
        mujoco.mj_forward(self.model, self.data)
        self.data.time += self.dt if elapsed is None else max(float(elapsed), 0.0)
        self._publish_viewer_snapshot()

    def stop(self) -> None:
        self.viewer_stop.set()
        if self.viewer_thread is not None:
            self.viewer_thread.join(timeout=1.0)
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.viewer_thread is not None and self.viewer_thread.is_alive():
            self.viewer_thread.join(timeout=1.0)
        self.viewer_thread = None
        self.viewer_data = None
        self.viewer_qpos_snapshot = None


class _AsyncAudioOutput:
    """Write speaker blocks on a worker so PortAudio cannot stall control."""

    def __init__(self, stream: Any, max_pending_blocks: int = 8) -> None:
        self.stream = stream
        self.blocks: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=max(int(max_pending_blocks), 1)
        )
        self.closing = threading.Event()
        self.error: BaseException | None = None
        self.dropped_blocks = 0
        self.thread = threading.Thread(
            target=self._run,
            name="audio-output",
            daemon=True,
        )
        self.thread.start()

    def submit(self, block: np.ndarray) -> None:
        self.raise_if_failed()
        rendered = np.ascontiguousarray(block, dtype=np.float32)
        try:
            self.blocks.put_nowait(rendered)
            return
        except queue.Full:
            # Keeping stale audio would increase A/V latency indefinitely. Drop
            # the oldest block and preserve the newest point on the timeline.
            try:
                self.blocks.get_nowait()
                self.blocks.task_done()
                self.dropped_blocks += 1
            except queue.Empty:
                pass
        try:
            self.blocks.put_nowait(rendered)
        except queue.Full:
            self.dropped_blocks += 1

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError("Audio output worker failed.") from self.error

    def stop(self) -> None:
        self.closing.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            abort = getattr(self.stream, "abort", None)
            if callable(abort):
                abort()
            self.thread.join(timeout=1.0)
        self.raise_if_failed()

    def _run(self) -> None:
        try:
            while not self.closing.is_set() or not self.blocks.empty():
                try:
                    block = self.blocks.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    self.stream.write(block)
                finally:
                    self.blocks.task_done()
        except BaseException as exc:
            self.error = exc
        finally:
            while True:
                try:
                    self.blocks.get_nowait()
                    self.blocks.task_done()
                except queue.Empty:
                    break
            try:
                self.stream.stop()
            finally:
                self.stream.close()


class FileMicrophoneSource:
    """Feed an audio file to the microphone analyzer at realtime speed."""

    def __init__(
        self,
        path: Path,
        analyzer: RealtimeMusicAnalyzer,
        startup_delay_sec: float = 1.0,
        throttle: bool = True,
        play_audio: bool = False,
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
        self.play_audio = bool(play_audio)
        self.output_stream: Any | None = None
        self.audio_output: _AsyncAudioOutput | None = None
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
        if self.play_audio:
            stream = None
            try:
                import sounddevice as sd

                stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    channels=1,
                    dtype="float32",
                )
                stream.start()
            except Exception as exc:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                raise RuntimeError(
                    "Could not open the default audio output for --play-audio. "
                    "Check the Windows output device and sounddevice/PortAudio setup."
                ) from exc
            self.output_stream = stream
            self.audio_output = _AsyncAudioOutput(stream)
        self.stream_start_wall = time.perf_counter()
        self.last_advance_wall = self.stream_start_wall

    def stop(self) -> None:
        analyzer_stop = getattr(self.analyzer, "stop", None)
        if callable(analyzer_stop):
            analyzer_stop()
        audio_output = self.audio_output
        self.audio_output = None
        stream = self.output_stream
        self.output_stream = None
        if audio_output is not None:
            audio_output.stop()
        elif stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

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
        if self.audio_output is not None:
            output_block = np.ascontiguousarray(block.reshape(-1, 1), dtype=np.float32)
            self.audio_output.submit(output_block)
        self.cursor = end


def parse_args(*, control_rate_default: float = 120.0) -> argparse.Namespace:
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
        "--retarget-policy",
        choices=("prefer-gmr", "require-gmr", "direct"),
        default="require-gmr",
        help="Require pre-retargeted GMR artifacts for AIST++ motions, or explicitly select a diagnostic fallback.",
    )
    parser.add_argument(
        "--gmr-motion-root",
        type=Path,
        default=DEFAULT_GMR_MOTION_ROOT,
        help="Directory containing <AIST motion id>.pkl GMR artifacts.",
    )
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
        help="Deprecated compatibility option; ignored because GMR values are always preserved.",
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
    parser.add_argument(
        "--play-audio",
        action="store_true",
        help="Play --audio-input through the default output device in sync with virtual-microphone analysis.",
    )
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional duration limit for smoke tests.")
    parser.add_argument("--realtime", action="store_true", help="Throttle the control loop to wall-clock time.")
    parser.add_argument(
        "--control-rate-hz",
        type=float,
        default=control_rate_default,
        help="Realtime kinematic control frequency (default: %(default)s Hz).",
    )
    parser.add_argument(
        "--viewer-rate-hz",
        type=float,
        default=60.0,
        help="MuJoCo Viewer refresh frequency, decoupled from control (default: 60 Hz).",
    )
    parser.add_argument(
        "--timing-report",
        type=Path,
        default=None,
        help="Optional JSON report containing loop, PLP, and collision timing counters.",
    )
    parser.add_argument(
        "--runtime-collision-check",
        choices=("auto", "always", "off"),
        default="off",
        help="Check modified/blended poses, every pose, or no realtime poses.",
    )
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
    parser.add_argument("--speed-max", type=float, default=1.6)
    parser.add_argument(
        "--max-speed-change-per-sec",
        type=float,
        default=2.0,
        help="Maximum change in motion speed multiplier per second (default: 2.0x/s).",
    )
    parser.add_argument("--amp-min", type=float, default=0.35)
    parser.add_argument("--amp-max", type=float, default=1.25)
    parser.add_argument("--pose-gain", type=float, default=1.0)
    parser.add_argument("--accent-gain", type=float, default=0.55)
    parser.add_argument("--music-modulation-strength", type=float, default=1.0)
    parser.add_argument(
        "--pose-modulation-mode",
        choices=("off", "subtle", "expressive"),
        default="subtle",
        help="Subtle preserves authored amplitudes; expressive retains the legacy full-pose scaling.",
    )
    parser.add_argument(
        "--root-motion",
        choices=("continuous", "in-place", "reset"),
        default="continuous",
        help="How floating-base motion is anchored across clip loops and switches.",
    )
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
    if args.retarget_policy != "direct":
        gmr_root = args.gmr_motion_root if args.gmr_motion_root.is_absolute() else ROOT / args.gmr_motion_root
        gmr_candidate = gmr_root / f"{motion_path.stem}.pkl"
        if gmr_candidate.exists():
            sampler = GmrUnitreeG1MotionSampler(
                motion_path=gmr_candidate,
                fps_override=args.gmr_fps,
                pose_gain=args.pose_gain,
                accent_gain=args.accent_gain,
                use_music_amplitude=False,
            )
            print(
                f"Loaded preferred GMR artifact for AIST++ motion: {gmr_candidate.name} "
                f"({len(sampler.frames)} frames, {sampler.duration:.2f}s)."
            )
            return sampler
        if args.retarget_policy == "require-gmr":
            raise FileNotFoundError(
                f"Required GMR artifact not found: {gmr_candidate}\n"
                f"Generate it with:\n{gmr_generation_hint(motion_path)}"
            )
        print(f"GMR artifact not found for {motion_path.stem}; using direct AIST++ fallback.")
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
        input_device=(
            int(getattr(args, "mic_device", ""))
            if str(getattr(args, "mic_device", "")).lstrip("-").isdigit()
            else getattr(args, "mic_device", None)
        ),
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
    aist_source_sampler: AistppMotionSampler | None = (
        sampler if isinstance(sampler, AistppMotionSampler) else None
    )
    if (
        aist_source_sampler is None
        and args.motion_source == "aistpp"
        and isinstance(sampler, GmrUnitreeG1MotionSampler)
    ):
        source_path = resolve_aistpp_motion(args.aistpp_root, args.aistpp_motion)
        aist_source_sampler = AistppMotionSampler(source_path, args.aistpp_fps, 1.0, 0.0)
    use_aist_velocity = aist_source_sampler is not None and args.keypoint_mode in (
        "auto",
        "aist-velocity",
    )
    if args.keypoint_mode == "aist-velocity" and aist_source_sampler is None:
        raise ValueError("--keypoint-mode aist-velocity requires --motion-source aistpp.")

    if use_aist_velocity:
        assert aist_source_sampler is not None
        result = detect_aistpp_velocity_keypoints(
            smpl_poses=aist_source_sampler.frames,
            smpl_trans=aist_source_sampler.translations,
            smpl_scaling=aist_source_sampler.scaling,
            fps=aist_source_sampler.fps,
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


def sample_robot_motion_frame(
    sampler: HumanoidDanceSampler | AistppMotionSampler | GmrUnitreeG1MotionSampler | BvhUnitreeG1MotionSampler,
    *,
    phase: float,
    amplitude: float,
    accent: float,
    features: FeatureState,
    pose_adapter: UnitreeG1DanceAdapter | UnitreeG1JointPoseAdapter | None,
    modulator: MusicPoseModulator | None,
) -> RobotMotionFrame:
    frame = sampler.sample_frame(phase, amplitude, accent, features)
    joints = frame.joint_positions
    if modulator is not None:
        joints = modulator.modulate(
            joints,
            features,
            phase=phase,
            amplitude=amplitude,
            accent=accent,
        )
    if pose_adapter is not None:
        joints = pose_adapter.adapt_pose(joints, features)
    return frame.with_joint_positions(joints)


def sampler_source_id(sampler: object) -> str:
    motion_path = getattr(sampler, "motion_path", None)
    return str(motion_path) if motion_path is not None else type(sampler).__name__


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
    initial_root = player.root_frame()
    root_motion = RootMotionContinuity(
        initial_root.root_position if initial_root.root_position is not None else np.zeros(3),
        initial_root.root_quaternion_wxyz
        if initial_root.root_quaternion_wxyz is not None
        else np.asarray([1.0, 0.0, 0.0, 0.0]),
        mode=args.root_motion,
        grounded_z=True,
    )
    print(
        f"Previewing authored dance trajectory at normal speed "
        f"({cycle_duration:.2f}s per cycle, amplitude={amplitude:.2f})."
    )

    start = time.perf_counter()
    scheduler = RealtimeLoopScheduler(
        getattr(args, "control_rate_hz", 120.0),
        getattr(args, "realtime", False),
    )
    last_status = start
    try:
        while player.is_running():
            work_started = time.perf_counter()
            now = work_started
            elapsed = now - start
            if args.max_seconds is not None and elapsed >= args.max_seconds:
                break

            phase = (elapsed / cycle_duration) % 1.0
            frame = sample_robot_motion_frame(
                sampler,
                phase=phase,
                amplitude=amplitude,
                accent=0.0,
                features=features,
                pose_adapter=pose_adapter,
                modulator=None,
            )
            frame = root_motion.apply(frame, phase=phase, source_id=sampler_source_id(sampler))
            player.set_frame(frame)
            player.step()

            if now - last_status >= args.status_interval:
                print(f"preview | phase={phase:.2f} | amp={amplitude:.2f} | speed=1.00x")
                last_status = now
            scheduler.wait(work_started)
    finally:
        player.stop()


def main() -> None:
    args = parse_args()
    sampler = make_sampler(args)
    motion_cycle_duration = max(args.motion_cycle_duration or sampler.duration, 1e-6)
    player = MujocoHumanoidPlayer(
        args.model,
        realtime=args.realtime,
        headless=args.headless,
        viewer_rate_hz=args.viewer_rate_hz,
    )
    pose_adapter = make_pose_adapter(args, player, sampler)
    player.ground_sampler(sampler, pose_adapter)
    modulation_off = args.disable_music_modulation or args.pose_modulation_mode == "off"
    modulator = (
        None
        if modulation_off
        else MusicPoseModulator(args.music_modulation_strength, mode=args.pose_modulation_mode)
    )
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
        max_speed_change_per_sec=args.max_speed_change_per_sec,
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
            play_audio=args.play_audio,
        )
    features = FeatureState(rms_norm=0.45, brightness=0.35, low_energy=0.55, mid_energy=0.45, high_energy=0.25)

    if analyzer is not None:
        # Spawning a numerical worker after MuJoCo creates its OpenGL viewer
        # can deadlock on Windows.  Finish worker startup first.
        analyzer.prepare_background_analysis()
    live_microphone_started = analyzer is not None and file_source is None
    if live_microphone_started:
        print(f"Calibrating microphone noise from the first {args.startup_calibration_sec:.2f}s of live input.")
        analyzer.start()
    player.start()
    if live_microphone_started:
        analyzer.reset()
    initial_root = player.root_frame()
    root_motion = RootMotionContinuity(
        initial_root.root_position if initial_root.root_position is not None else np.zeros(3),
        initial_root.root_quaternion_wxyz
        if initial_root.root_quaternion_wxyz is not None
        else np.asarray([1.0, 0.0, 0.0, 0.0]),
        mode=args.root_motion,
        grounded_z=True,
    )
    if file_source is not None:
        file_source.start()
        print(
            f"Using realtime virtual microphone: {file_source.path} "
            f"(audio starts after {file_source.startup_delay_sec:.2f}s denoiser reset time; "
            f"speaker output={'on' if file_source.play_audio else 'off'})."
        )
    elif analyzer is not None:
        pass
    else:
        controller.target_amplitude_scale = 0.75
        print("Running in --no-mic demo mode.")

    start = time.perf_counter()
    scheduler = RealtimeLoopScheduler(args.control_rate_hz, args.realtime)
    last_status = start
    last_feature_update = start
    last_frame: MusicFrame | None = None
    recent_detected_beat_frame: MusicFrame | None = None
    recent_accepted_beat_frame: MusicFrame | None = None
    try:
        while player.is_running():
            work_started = time.perf_counter()
            now = work_started
            if file_source is not None:
                file_source.advance(scheduler.period)
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
                    dt = max(now - last_feature_update, scheduler.period)
                    alpha = 1.0 - math.exp(-dt / max(args.feature_smoothing_tau, 1e-6))
                    features.update(frame, alpha)
                    last_feature_update = now
            else:
                controller.target_amplitude_scale = 0.75

            phase, amplitude, accent, _brightness = controller.update(now)
            motion_frame = sample_robot_motion_frame(
                sampler,
                phase=phase,
                amplitude=amplitude,
                accent=accent,
                features=features,
                pose_adapter=pose_adapter,
                modulator=modulator,
            )
            motion_frame = root_motion.apply(
                motion_frame,
                phase=phase,
                source_id=sampler_source_id(sampler),
            )
            motion_frame = player.apply_collision_policy(
                motion_frame,
                args.runtime_collision_check,
                runtime_modified=modulator is not None and features.is_active,
            )
            player.set_frame(motion_frame)
            player.step(scheduler.period)

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
            scheduler.wait(work_started)
    finally:
        if file_source is not None:
            file_source.stop()
        elif analyzer is not None:
            analyzer.stop()
        player.stop()
        write_timing_report(args.timing_report, scheduler, analyzer, player)


if __name__ == "__main__":
    main()
