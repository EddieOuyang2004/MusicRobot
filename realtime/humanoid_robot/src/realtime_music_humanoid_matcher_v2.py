"""V2: early motion selection and authored-state bridges for one-shot playback."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import math
import random
import sys
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterator

import librosa
import numpy as np

import realtime_music_humanoid_dancer as base
from music_motion_catalog import (
    AudioFeatureExtractor,
    MatchResult,
    MotionSelection,
    MotionSelectionPolicy,
    MotionProfile,
    MusicCatalog,
    MusicMotionMatcher,
)
from music_pose_modulator import MusicPoseModulator
from motion_bridges import (
    AuthoredTrajectory, AuthoredClock, JointState, InfeasibleBridge,
    make_bridge, load_jerk_limits, require_ruckig, warmup_hermite,
)
from robot_motion import (
    JointDynamicsLimiter,
    JointDynamicsLimits,
    RobotMotionFrame,
    RootMotionContinuity,
    align_motion_frame_root,
    blend_motion_frames,
    load_joint_dynamics_limits,
    wxyz_to_rotation,
    yaw_only,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = (
    ROOT / "realtime" / "humanoid_robot" / "data" / "music_catalog" / "catalog.json"
)


def _set_windows_thread_priority(priority: int) -> None:
    """Best-effort per-thread priority without changing process-wide policy."""

    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), int(priority))
    except Exception:
        pass


def _set_background_worker_priority() -> None:
    # THREAD_PRIORITY_BELOW_NORMAL. The control thread remains responsive when
    # a Python/MuJoCo preparation scan becomes runnable on Windows.
    _set_windows_thread_priority(-1)


def _set_retrieval_process_priority() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # BELOW_NORMAL_PRIORITY_CLASS
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


@dataclass(frozen=True)
class EntryScoringConfig:
    """Fixed soft tolerances; initial tuning values, not calibrated thresholds."""

    pose_tolerance_fraction: float = 0.05
    velocity_tolerance_fraction: float = 0.10
    contact_tolerance_feet: float = 1.0
    root_height_tolerance_m: float = 0.025
    root_tilt_tolerance_rad: float = 0.075
    root_linear_velocity_tolerance_m_s: float = 0.25
    root_angular_speed_tolerance_rad_s: float = 0.5
    excess_penalty_weight: float = 4.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            allow_zero = field.name == "excess_penalty_weight"
            if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
                requirement = "nonnegative" if allow_zero else "positive"
                raise ValueError(f"--entry-{field.name.replace('_', '-')} must be finite and {requirement}")

    @classmethod
    def from_args(cls, args):
        defaults = cls()
        return cls(**{field.name: getattr(args, f"entry_{field.name}", getattr(defaults, field.name))
                      for field in fields(cls)})


def entry_difference_penalty(difference, tolerance, excess_weight=4.0):
    """Vectorized unbounded penalty for nonnegative differences; no dead zone."""
    ratio = np.asarray(difference, dtype=np.float64) / tolerance
    return np.square(ratio) + excess_weight * np.square(np.maximum(ratio - 1.0, 0.0))


def parse_args() -> argparse.Namespace:
    """Parse matcher-specific flags, then delegate all legacy flags to the frozen entrypoint."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--embedding-model", type=Path, default=None)
    parser.add_argument("--tag-model", type=Path, default=None)
    obsolete = {"--match-window-seconds", "--switch-required-wins", "--switch-score-margin",
                "--switch-max-hold-bars", "--switch-min-remaining-bars", "--transition-max-seconds",
                "--diagnostic-instant-top1", "--diagnostic-fixed-entry-transition",
                "--terminal-safe-idle-amplitude", "--switch-recent-history", "--switch-beats-per-bar"}
    for argument in sys.argv[1:]:
        if argument.split("=", 1)[0] in obsolete:
            parser.error(f"{argument.split('=', 1)[0]} is a v1 switching option; v2 selects authored exit and entry frames.")
    parser.add_argument("--history-max-seconds", type=float, default=30.0)
    parser.add_argument("--analysis-min-seconds", type=float, default=2.0)
    parser.add_argument("--entry-max-phase", type=float, default=0.2,
                        help="Latest eligible entry as a fraction of each motion (default: 0.2, first 20%%).")
    parser.add_argument("--transition-exit-window-seconds", type=float, default=2.0,
                        help="Search this many authored seconds before the source endpoint; zero uses only the endpoint.")
    scoring_defaults = EntryScoringConfig()
    for field in fields(scoring_defaults):
        parser.add_argument(
            f"--entry-{field.name.replace('_', '-')}", type=float,
            default=getattr(scoring_defaults, field.name),
            help=("Extra squared penalty above tolerance (default: %(default)s)."
                  if field.name == "excess_penalty_weight" else
                  "Soft scoring tolerance; smaller is more sensitive (default: %(default)s)."),
        )
    parser.set_defaults(match_window_seconds=30.0)
    parser.add_argument("--match-interval-seconds", type=float, default=1.0)
    parser.add_argument("--match-top-tracks", type=int, default=5)
    parser.add_argument("--match-top-motions", type=int, default=20)
    parser.add_argument(
        "--motion-timing",
        choices=("beat-sync", "authored"),
        default="beat-sync",
        help=(
            "How motion phase advances. beat-sync follows detected beats and "
            "keypoints; authored keeps every motion at its original 1.0x rate "
            "while preserving matching, accents, and switching."
        ),
    )
    parser.add_argument(
        "--match-policy",
        choices=("style-first", "legacy"),
        default="style-first",
        help="Prefer a confident genre family before ranking motions (default: style-first).",
    )
    parser.add_argument(
        "--match-weak-music-threshold",
        type=float,
        default=0.17,
        help="Ambient/non-music tag threshold used by the consecutive-window gate.",
    )
    parser.add_argument(
        "--match-weak-music-consecutive-windows",
        type=int,
        default=2,
        help=(
            "Consecutive analysis windows at or above the weak-music threshold "
            "required before holding the current motion (default: 2)."
        ),
    )
    parser.add_argument(
        "--initial-motion-id",
        type=str,
        default=None,
        help=(
            "Optional catalog motion ID to use at startup. By default the matcher "
            "randomly chooses a sufficiently long motion from the lowest-activity quartile."
        ),
    )
    parser.add_argument(
        "--initial-motion-low-activity-quantile",
        type=float,
        default=0.25,
        help="Fraction of the lowest-velocity eligible motions used for random startup selection.",
    )
    parser.add_argument(
        "--initial-motion-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible random startup motion selection.",
    )
    parser.add_argument("--switch-required-wins", type=int, default=3)
    parser.add_argument("--switch-score-margin", type=float, default=0.08)
    parser.add_argument("--switch-beats-per-bar", type=int, default=4)
    parser.add_argument("--switch-max-hold-bars", type=int, default=4)
    parser.add_argument("--shortlist-size", "--switch-diversity-top-k", dest="switch_diversity_top_k", type=int, default=10)
    parser.add_argument("--shortlist-score-drop", "--switch-diversity-score-drop", dest="switch_diversity_score_drop", type=float, default=0.05)
    parser.add_argument(
        "--shortlist-music-score-drop", "--switch-diversity-music-score-drop", dest="switch_diversity_music_score_drop",
        type=float,
        default=0.08,
    )
    parser.add_argument("--switch-recent-history", type=int, default=3)
    parser.add_argument("--switch-ready-pool-size", type=int, default=3)
    parser.add_argument(
        "--motion-cache-size",
        type=int,
        default=96,
        help="Maximum prepared motions retained by the realtime loader.",
    )
    parser.add_argument(
        "--python-thread-switch-interval-ms",
        type=float,
        default=1.0,
        help=(
            "Python GIL scheduling interval used in realtime mode. A 1 ms interval "
            "reduces control-loop stalls while background feature work is active."
        ),
    )
    parser.add_argument(
        "--startup-ready-reserve-seconds",
        type=float,
        default=4.0,
        help=(
            "Extra wall-clock time reserved for background motion preparation "
            "when choosing the initial one-shot motion."
        ),
    )
    parser.add_argument("--switch-min-remaining-bars", type=float, default=1.0)
    parser.add_argument("--transition-min-seconds", type=float, default=0.35)
    parser.add_argument("--transition-backend", choices=("hermite", "ruckig", "quintic"), default="hermite")
    parser.add_argument("--transition-boundary-seconds", type=float, default=0.5)
    parser.add_argument("--output-max-joint-jerk", type=float, default=None,
                        help="Optional rad/s^3 bound; Ruckig requires limits for every joint.")
    parser.add_argument("--transition-max-seconds", type=float, default=1.20)
    parser.add_argument(
        "--terminal-safe-idle-amplitude",
        type=float,
        default=0.018,
        help=(
            "Maximum procedural joint offset in radians while a one-shot motion "
            "waits at its terminal pose; set to 0 to disable."
        ),
    )
    parser.add_argument("--output-max-joint-speed", type=float, default=16.0)
    # Keep 20% implementation headroom below the 2000 rad/s^2 experiment
    # acceptance boundary.  The limiter runs shortly before emission, so Windows
    # scheduling and final collision auditing make the measured wall-clock dt
    # slightly different from the dt available at the limiter call.
    parser.add_argument("--output-max-joint-acceleration", type=float, default=1600.0)
    parser.add_argument("--output-joint-limits-json", type=Path, default=None)
    parser.add_argument(
        "--diagnostic-disable-output-limiter",
        action="store_true",
        help="Headless-only A/B diagnostic; never disables collision handling.",
    )
    parser.add_argument(
        "--diagnostic-disable-motion-compatibility",
        action="store_true",
        help=(
            "Headless-only ablation: rank motions by retrieved music score without "
            "tempo, keypoint-density, or activity compatibility."
        ),
    )
    parser.add_argument(
        "--diagnostic-fixed-entry-transition",
        action="store_true",
        help=(
            "Headless-only ablation: enter at phase zero, skip root alignment, "
            "and use a fixed-duration linear blend."
        ),
    )
    parser.add_argument(
        "--diagnostic-instant-top1",
        action="store_true",
        help=(
            "Headless-only ablation: request the current Top-1 as soon as it is "
            "prepared, without waiting for a beat/bar boundary."
        ),
    )
    parser.add_argument(
        "--trace-csv",
        type=Path,
        default=None,
        help="Optional CSV timeline containing matcher rankings, motion switches, and phase progress.",
    )
    parser.add_argument(
        "--trace-interval-seconds",
        type=float,
        default=0.05,
        help="Audio-time interval between --trace-csv samples (default: 0.05).",
    )
    parser.add_argument(
        "--experiment-causal-file-input", action="store_true",
        help="Experiment only: online beats, wall-clock-paced file input and final-output telemetry.",
    )
    parser.add_argument(
        "--experiment-pose-npz",
        type=Path,
        default=None,
        help=(
            "Optional experiment-only NPZ containing every output pose, body "
            "position, COM, feet, motion ID, and accepted causal beat time."
        ),
    )
    parser.add_argument(
        "--experiment-warmup-seconds",
        type=float,
        default=0.0,
        help=(
            "Experiment-only interval excluded from control-stage/deadline timing; "
            "the default zero preserves normal CLI behaviour."
        ),
    )
    parser.add_argument(
        "--matcher-help",
        action="store_true",
        help="Show the matcher-specific options; use --help for inherited dancer options.",
    )
    matcher_args, remaining = parser.parse_known_args()
    explicit_aistpp_motion = any(
        argument == "--aistpp-motion" or argument.startswith("--aistpp-motion=")
        for argument in remaining
    )
    if matcher_args.matcher_help:
        help_parser = argparse.ArgumentParser(
            description=(
                "Humanoid matcher v2: authored exit/entry transitions (60 Hz control by default). "
                "All realtime_music_humanoid_dancer.py options are also accepted."
            )
        )
        for action in parser._actions:
            if action.dest != "help" and not any(option in obsolete for option in action.option_strings):
                help_parser._add_action(action)
        help_parser.print_help()
        raise SystemExit(0)

    original_argv = sys.argv
    try:
        # Override only v2's inherited default; keep explicit values and the
        # base parser's validation, without changing v1 or the dancer.
        speed_default = [] if any(
            value == "--speed-max" or value.startswith("--speed-max=")
            for value in remaining
        ) else ["--speed-max", "1.3"]
        sys.argv = [original_argv[0], *remaining, *speed_default]
        args = base.parse_args(control_rate_default=60.0)
    finally:
        sys.argv = original_argv
    for name, value in vars(matcher_args).items():
        setattr(args, name, value)
    try:
        EntryScoringConfig.from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if explicit_aistpp_motion and args.initial_motion_id is None:
        args.initial_motion_id = Path(args.aistpp_motion).stem
    for name in ("history_max_seconds", "analysis_min_seconds", "match_interval_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.entry_max_phase) or not 0 <= args.entry_max_phase <= 1:
        parser.error("--entry-max-phase must be finite and between 0 and 1")
    if not math.isfinite(args.transition_exit_window_seconds) or args.transition_exit_window_seconds < 0:
        parser.error("--transition-exit-window-seconds must be finite and non-negative")
    if args.analysis_min_seconds > args.history_max_seconds:
        parser.error("--analysis-min-seconds must not exceed --history-max-seconds")
    if args.history_max_seconds > 30.0:
        parser.error("--history-max-seconds must not exceed 30 seconds")
    if args.switch_diversity_top_k <= 0 or args.match_top_motions <= 0:
        parser.error("shortlist and retrieval sizes must be positive")
    for name in ("switch_diversity_score_drop", "switch_diversity_music_score_drop"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.match_policy != "style-first":
        parser.error("v2 requires --match-policy style-first")
    if not math.isfinite(args.transition_boundary_seconds) or args.transition_boundary_seconds <= 0:
        parser.error("--transition-boundary-seconds must be finite and positive")
    if args.output_max_joint_jerk is not None and (
        not math.isfinite(args.output_max_joint_jerk) or args.output_max_joint_jerk <= 0
    ):
        parser.error("--output-max-joint-jerk must be finite and positive")
    if args.transition_backend != "quintic" and args.transition_min_seconds > 10:
        parser.error("State bridge minimum duration cannot exceed 10 seconds")
    args.match_window_seconds = args.history_max_seconds
    args.switch_required_wins = 1
    args.switch_ready_pool_size = max(args.switch_ready_pool_size, args.switch_diversity_top_k)
    args.terminal_safe_idle_amplitude = 0.0
    return args


@dataclass(frozen=True)
class MotionEntryFeatures:
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_ranges: np.ndarray
    root_positions: np.ndarray
    root_tilt: np.ndarray
    root_linear_velocities: np.ndarray
    root_angular_speeds: np.ndarray
    foot_contacts: np.ndarray
    candidate_indices: np.ndarray
    candidate_salience: np.ndarray
    fps: float
    duration: float
    authored: AuthoredTrajectory | None = None

    def index_for_phase(self, phase: float) -> int:
        return int(
            np.clip(
                round(float(np.clip(phase, 0.0, 1.0)) * (len(self.joint_positions) - 1)),
                0,
                len(self.joint_positions) - 1,
            )
        )


@dataclass(frozen=True)
class MotionEntryScore:
    """Unbounded nonlinear costs, lower is better (also in CSV entry_*_score).

    Components replace the old linear differences. Music remains zero here.
    """
    phase: float
    frame_index: int
    total: float
    pose: float
    velocity: float
    contact: float
    root: float
    music: float
    remaining_seconds: float
    transition_seconds: float


@dataclass
class PendingBeatDiagnostic:
    accepted_time_seconds: float
    target_pose: dict[str, float]
    errors_rad: list[float]


@dataclass(frozen=True)
class WristMotionDiagnostics:
    max_abs_joint: str = ""
    max_abs_rad: float = 0.0
    max_speed_joint: str = ""
    max_speed_rad_s: float = 0.0


def wrist_motion_diagnostics(
    frame: RobotMotionFrame,
    previous_positions: dict[str, float],
    dt: float,
) -> tuple[WristMotionDiagnostics, dict[str, float]]:
    positions = {
        name: float(value)
        for name, value in frame.joint_positions.items()
        if "wrist" in name.lower()
    }
    if not positions:
        return WristMotionDiagnostics(), positions

    max_abs_joint = max(positions, key=lambda name: abs(positions[name]))
    common = positions.keys() & previous_positions.keys()
    speeds = (
        {
            name: abs(positions[name] - previous_positions[name]) / dt
            for name in common
        }
        if math.isfinite(dt) and dt > 0.0
        else {}
    )
    max_speed_joint = max(speeds, key=speeds.get) if speeds else ""
    return (
        WristMotionDiagnostics(
            max_abs_joint=max_abs_joint,
            max_abs_rad=abs(positions[max_abs_joint]),
            max_speed_joint=max_speed_joint,
            max_speed_rad_s=speeds.get(max_speed_joint, 0.0),
        ),
        positions,
    )


def frame_root_yaw(frame: RobotMotionFrame) -> float | None:
    if frame.root_quaternion_wxyz is None:
        return None
    matrix = wxyz_to_rotation(frame.root_quaternion_wxyz).as_matrix()
    return float(math.atan2(matrix[1, 0], matrix[0, 0]))


def wrapped_angle_difference(first: float, second: float) -> float:
    return float(math.atan2(math.sin(first - second), math.cos(first - second)))


class OneShotPhaseTracker:
    """Clamp a controller's cyclic phase before it can sample a loop seam."""

    terminal_phase = float(np.nextafter(1.0, 0.0))

    def __init__(self, entry_phase: float = 0.0) -> None:
        self.last_phase = float(np.clip(entry_phase, 0.0, self.terminal_phase))
        self.ended = False

    def clamp(self, phase: float) -> tuple[float, bool]:
        value = float(phase % 1.0)
        wrapped = value + 0.5 < self.last_phase
        if self.ended or wrapped:
            self.ended = True
            self.last_phase = self.terminal_phase
            return self.terminal_phase, True
        self.last_phase = min(max(value, self.last_phase), self.terminal_phase)
        return self.last_phase, False

    def minimum_time_to_end(self, maximum_phase_rate: float) -> float:
        if self.ended:
            return 0.0
        return max(self.terminal_phase - self.last_phase, 0.0) / max(
            float(maximum_phase_rate),
            1e-9,
        )


def forced_transition_blend(
    current_phase: float,
    start_phase: float,
    *,
    elapsed_seconds: float = 0.0,
    duration_seconds: float = 1.0,
) -> float:
    if start_phase >= OneShotPhaseTracker.terminal_phase - 1e-9:
        return float(
            np.clip(
                elapsed_seconds / max(float(duration_seconds), 1e-9),
                0.0,
                1.0,
            )
        )
    return float(
        np.clip(
            (current_phase - start_phase)
            / max(OneShotPhaseTracker.terminal_phase - start_phase, 1e-9),
            0.0,
            1.0,
        )
    )


def apply_terminal_safe_idle(
    frame: RobotMotionFrame,
    elapsed_seconds: float,
    amplitude: float,
) -> RobotMotionFrame:
    """Add a tiny stationary breathing/sway signal without moving the root."""

    strength = max(float(amplitude), 0.0)
    if strength <= 0.0:
        return frame
    breath = strength * math.sin(2.0 * math.pi * 0.20 * elapsed_seconds)
    sway = 0.35 * strength * math.sin(2.0 * math.pi * 0.11 * elapsed_seconds)
    joints = dict(frame.joint_positions)

    def add_existing(names: tuple[str, ...], offset: float) -> None:
        for name in names:
            if name in joints:
                joints[name] = float(joints[name] + offset)
                return

    add_existing(("waist_pitch_joint", "waist_pitch", "torso_pitch"), breath)
    add_existing(("waist_roll_joint", "waist_roll", "torso_roll"), sway)
    add_existing(
        ("left_shoulder_pitch_joint", "left_shoulder_pitch"),
        0.45 * breath,
    )
    add_existing(
        ("right_shoulder_pitch_joint", "right_shoulder_pitch"),
        0.45 * breath,
    )
    return frame.with_joint_positions(joints)


def build_motion_entry_features(
    sampler: MotionSampler,
    profile: MotionProfile,
    adapter: Any,
    joint_ranges: dict[str, tuple[float, float] | None],
) -> MotionEntryFeatures:
    frame_count = len(sampler.frames)
    sampled_joints = list(getattr(sampler, "adapted_joint_frames", ()))
    if len(sampled_joints) != frame_count:
        features = base.FeatureState()
        sampled_joints = []
        for frame_index in range(frame_count):
            frame = sampler.sample_frame(frame_index / frame_count, 1.0, 0.0, features)
            joints = frame.joint_positions
            if adapter is not None:
                joints = adapter.adapt_pose(joints, features)
            sampled_joints.append(
                {name: float(value) for name, value in joints.items()}
            )
    joint_names = tuple(sorted(set().union(*(item.keys() for item in sampled_joints))))
    positions = np.asarray(
        [[item.get(name, 0.0) for name in joint_names] for item in sampled_joints],
        dtype=np.float64,
    )
    fps = max(float(sampler.fps), 1e-6)
    velocities = np.zeros_like(positions)
    if frame_count > 1:
        velocities[1:] = np.diff(positions, axis=0) * fps
        velocities[0] = velocities[1]
    ranges = np.asarray(
        [
            max(values[1] - values[0], 1e-6)
            if (values := joint_ranges.get(name)) is not None
            else 2.0 * math.pi
            for name in joint_names
        ],
        dtype=np.float64,
    )
    root_positions = np.asarray(sampler.root_positions, dtype=np.float64).copy()
    root_positions[:, 2] += float(getattr(sampler, "ground_offset_z", 0.0))
    root_tilt = np.empty(frame_count, dtype=np.float64)
    rotations = []
    for index, quaternion in enumerate(np.asarray(sampler.root_quaternions, dtype=np.float64)):
        rotation = wxyz_to_rotation(quaternion)
        rotations.append(rotation)
        root_tilt[index] = float((yaw_only(rotation).inv() * rotation).magnitude())
    root_linear_velocities = np.zeros_like(root_positions)
    root_angular_speeds = np.zeros(frame_count, dtype=np.float64)
    if frame_count > 1:
        root_linear_velocities[1:] = np.diff(root_positions, axis=0) * fps
        root_linear_velocities[0] = root_linear_velocities[1]
        root_angular_speeds[1:] = np.asarray(
            [(rotations[index - 1].inv() * rotations[index]).magnitude() * fps for index in range(1, frame_count)]
        )
        root_angular_speeds[0] = root_angular_speeds[1]
    contacts = np.asarray(
        getattr(sampler, "foot_contacts", np.zeros((frame_count, 2), dtype=bool)),
        dtype=bool,
    )
    candidates: dict[int, float] = {}
    keypoints = profile.keypoint_phases or base.DEFAULT_FALLBACK_PHASES
    scores = profile.keypoint_scores or tuple(0.5 for _ in keypoints)
    for keypoint_index, keypoint_phase in enumerate(keypoints):
        center = int(round(float(keypoint_phase) * (frame_count - 1)))
        salience = float(scores[keypoint_index]) if keypoint_index < len(scores) else 0.5
        for offset in range(-3, 4):
            index = int(np.clip(center + offset, 0, frame_count - 1))
            phase_compatibility = 1.0 - abs(offset) / 4.0
            candidates[index] = max(
                candidates.get(index, 0.0),
                salience * phase_compatibility,
            )
    candidate_indices = np.asarray(sorted(candidates), dtype=np.int32)
    candidate_salience = np.asarray(
        [candidates[int(index)] for index in candidate_indices],
        dtype=np.float64,
    )
    return MotionEntryFeatures(
        joint_names=joint_names,
        joint_positions=positions,
        joint_velocities=velocities,
        joint_ranges=ranges,
        root_positions=root_positions,
        root_tilt=root_tilt,
        root_linear_velocities=root_linear_velocities,
        root_angular_speeds=root_angular_speeds,
        foot_contacts=contacts,
        candidate_indices=candidate_indices,
        candidate_salience=candidate_salience,
        fps=fps,
        duration=float(sampler.duration),
        authored=AuthoredTrajectory(positions, fps),
    )


def compute_transition_duration(
    first: np.ndarray,
    second: np.ndarray,
    speed_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    *,
    beat_period: float,
    minimum: float,
    maximum: float,
) -> float:
    delta = np.abs(np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64))
    velocity_time = float(np.max(1.875 * delta / np.maximum(speed_limits, 1e-9)))
    acceleration_time = float(
        np.max(np.sqrt((10.0 / math.sqrt(3.0)) * delta / np.maximum(acceleration_limits, 1e-9)))
    )
    raw = max(float(minimum), velocity_time, acceleration_time)
    return raw  # Never shorten a transition below its dynamics-derived duration.


def select_motion_entry(
    source: MotionEntryFeatures,
    source_phase: float,
    target: MotionEntryFeatures,
    limits: dict[str, JointDynamicsLimits],
    *,
    beat_period: float,
    beats_per_bar: int,
    minimum_remaining_bars: float,
    speed_max: float,
    transition_minimum: float,
    transition_maximum: float,
    enforce_remaining: bool = True,
    entry_max_phase: float = 0.2,
    return_all: bool | str = False,
    scoring_config: EntryScoringConfig | None = None,
    eligible_frame_mask: np.ndarray | None = None,
) -> MotionEntryScore | list[MotionEntryScore] | Iterator[MotionEntryScore] | None:
    config = scoring_config if scoring_config is not None else EntryScoringConfig()
    source_index = source.index_for_phase(source_phase)
    source_lookup = {name: index for index, name in enumerate(source.joint_names)}
    target_lookup = {name: index for index, name in enumerate(target.joint_names)}
    names = tuple(sorted(set(source_lookup) & set(target_lookup) & set(limits)))
    if not names:
        raise ValueError("Source and target motions have no common limited joints.")
    source_columns = np.asarray([source_lookup[name] for name in names], dtype=np.int32)
    target_columns = np.asarray([target_lookup[name] for name in names], dtype=np.int32)
    source_pose = source.joint_positions[source_index, source_columns]
    source_velocity = source.joint_velocities[source_index, source_columns]
    joint_ranges = source.joint_ranges[source_columns]
    speed_limits = np.asarray([limits[name].max_speed_rad_s for name in names])
    acceleration_limits = np.asarray(
        [limits[name].max_acceleration_rad_s2 for name in names]
    )
    candidate_indices = np.arange(len(target.joint_positions), dtype=np.int32)
    candidate_pose = target.joint_positions[candidate_indices][:, target_columns]
    candidate_velocity = target.joint_velocities[candidate_indices][:, target_columns]
    phases = candidate_indices.astype(np.float64) / max(len(target.joint_positions) - 1, 1)
    deltas = np.abs(candidate_pose - source_pose[None, :])
    velocity_times = np.max(
        1.875 * deltas / np.maximum(speed_limits[None, :], 1e-9),
        axis=1,
    )
    acceleration_times = np.max(
        np.sqrt(
            (10.0 / math.sqrt(3.0)) * deltas / np.maximum(acceleration_limits[None, :], 1e-9)
        ),
        axis=1,
    )
    raw_times = np.maximum.reduce(
        (
            np.full(len(candidate_indices), float(transition_minimum)),
            velocity_times,
            acceleration_times,
        )
    )
    transition_times = raw_times
    # Search the configured prefix of each clip regardless of duration or speed.
    # Keep authored remaining seconds for trace diagnostics.
    remaining_times = (1.0 - phases) * target.duration
    eligible = phases <= entry_max_phase + 1e-9
    if not enforce_remaining:
        eligible = candidate_indices == 0
    if eligible_frame_mask is not None:
        eligible &= eligible_frame_mask
    if not np.any(eligible):
        return None
    pose_scores = np.mean(
        entry_difference_penalty(deltas, joint_ranges[None, :] * config.pose_tolerance_fraction,
                                 config.excess_penalty_weight), axis=1,
    )
    velocity_scores = np.mean(
        entry_difference_penalty(
            np.abs(candidate_velocity - source_velocity[None, :]),
            np.maximum(speed_limits[None, :], 1e-9) * config.velocity_tolerance_fraction,
            config.excess_penalty_weight), axis=1,
    )
    contact_mismatches = np.sum(
        target.foot_contacts[candidate_indices]
        != source.foot_contacts[source_index][None, :],
        axis=1,
    )
    contact_scores = entry_difference_penalty(
        contact_mismatches, config.contact_tolerance_feet, config.excess_penalty_weight,
    )
    height_errors = np.abs(
        target.root_positions[candidate_indices, 2]
        - source.root_positions[source_index, 2]
    ) / config.root_height_tolerance_m
    tilt_errors = np.abs(
        target.root_tilt[candidate_indices] - source.root_tilt[source_index]
    ) / config.root_tilt_tolerance_rad
    linear_errors = np.linalg.norm(
        target.root_linear_velocities[candidate_indices]
        - source.root_linear_velocities[source_index][None, :],
        axis=1,
    ) / config.root_linear_velocity_tolerance_m_s
    angular_errors = np.abs(
        target.root_angular_speeds[candidate_indices]
        - source.root_angular_speeds[source_index]
    ) / config.root_angular_speed_tolerance_rad_s
    root_scores = np.mean(
        entry_difference_penalty(
            np.stack((height_errors, tilt_errors, linear_errors, angular_errors), axis=1),
            1.0, config.excess_penalty_weight),
        axis=1,
    )
    music_scores = np.zeros(len(candidate_indices))
    totals = (
        0.45 * pose_scores
        + 0.20 * velocity_scores
        + 0.20 * contact_scores
        + 0.10 * root_scores
    ) / 0.95
    eligible_indices = np.flatnonzero(eligible)
    order = np.lexsort((phases[eligible_indices], totals[eligible_indices]))
    def entry(selected):
        return MotionEntryScore(
            phase=float(phases[selected]),
            frame_index=int(candidate_indices[selected]),
            total=float(totals[selected]),
            pose=float(pose_scores[selected]),
            velocity=float(velocity_scores[selected]),
            contact=float(contact_scores[selected]),
            root=float(root_scores[selected]),
            music=float(music_scores[selected]),
            remaining_seconds=float(remaining_times[selected]),
            transition_seconds=float(transition_times[selected]),
        )
    if return_all == "iterator":
        return (entry(int(eligible_indices[i])) for i in order)
    if return_all:
        return [entry(int(eligible_indices[i])) for i in order]
    return entry(int(eligible_indices[int(order[0])]))


def available_history(audio, sample_rate, maximum, minimum):
    required = int(round(minimum * sample_rate))
    if len(audio) < required:
        return None
    return np.asarray(audio[-int(round(maximum * sample_rate)):], dtype=np.float32).copy()


class RetrievalAudioHistory:
    """Own bounded buffer, independent of the beat analyser's history length."""
    def __init__(self, analyzer, maximum, minimum):
        self.sample_rate = analyzer.sample_rate
        self.maximum, self.minimum = maximum, minimum
        self.chunks = deque()
        self.count = 0
        self.lock = threading.Lock()
        original = analyzer._append_audio_chunk

        def append(start, mono):
            original(start, mono)
            chunk = np.asarray(mono, dtype=np.float32).copy()
            with self.lock:
                self.chunks.append(chunk)
                self.count += len(chunk)
                excess = self.count - int(round(self.maximum * self.sample_rate))
                while excess > 0 and self.chunks:
                    first = self.chunks.popleft()
                    removed = min(excess, len(first))
                    if removed < len(first):
                        self.chunks.appendleft(first[removed:])
                    excess -= removed
                    self.count -= removed
        analyzer._append_audio_chunk = append

    def recent_audio(self):
        with self.lock:
            if self.count < int(round(self.minimum * self.sample_rate)):
                return None
            return np.concatenate(list(self.chunks)).copy()


def musical_shortlist(result, current_id, args, *, expanded=False, exclude=()):
    if result is None or not result.accepted or not result.motions:
        return ()
    best = max(result.motions, key=lambda item: item.final_score)
    eligible = [item for item in result.motions
                if item.motion_id != current_id and item.motion_id not in exclude
                and item.final_score >= best.final_score - max(args.switch_diversity_score_drop, .10 if expanded else 0.)
                and item.music_score >= best.music_score - max(args.switch_diversity_music_score_drop, .15 if expanded else 0.)]
    return tuple(item.motion_id for item in sorted(
        eligible, key=lambda item: (-item.final_score, -item.music_score, item.motion_id)
    )[:args.switch_diversity_top_k])


def quintic_blend(first, second, amount):
    u = float(np.clip(amount, 0.0, 1.0))
    weight = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    return blend_motion_frames(first, second, weight, smoothstep=False)


def terminal_features(features, frame):
    """Replace the cached final pose with the terminal command pose for rechecking."""
    positions = features.joint_positions[-1:].copy()
    for column, name in enumerate(features.joint_names):
        positions[0, column] = frame.joint_positions.get(name, positions[0, column])
    roots = features.root_positions[-1:].copy()
    tilt = features.root_tilt[-1:].copy()
    if frame.root_position is not None:
        roots[0] = frame.root_position
    if frame.root_quaternion_wxyz is not None:
        rotation = wxyz_to_rotation(frame.root_quaternion_wxyz)
        tilt[0] = (yaw_only(rotation).inv() * rotation).magnitude()
    return replace(features, joint_positions=positions,
                   joint_velocities=features.joint_velocities[-1:].copy(),
                   root_positions=roots, root_tilt=tilt,
                   root_linear_velocities=features.root_linear_velocities[-1:].copy(),
                   root_angular_speeds=features.root_angular_speeds[-1:].copy(),
                   foot_contacts=features.foot_contacts[-1:].copy())


def score_ready_candidates(source_features, ready, limits, args, timing_samples=None):
    """Worker input is an immutable snapshot; never access the loader from this thread."""
    started = time.perf_counter()
    choices = []
    for rank, item in enumerate(ready):
        motion_id, sampler = item[:2]
        music_relevance = item[2] if len(item) > 2 else (-rank, 0.0)
        score = select_motion_entry(
            source_features, 1.0, sampler.entry_features, limits,
            beat_period=0.5, beats_per_bar=4, minimum_remaining_bars=0,
            speed_max=1.0 if args.motion_timing == "authored" else args.speed_max,
            transition_minimum=args.transition_min_seconds, transition_maximum=math.inf,
            entry_max_phase=args.entry_max_phase,
            scoring_config=EntryScoringConfig.from_args(args),
        )
        if score is not None:
            choices.append((score.total, -music_relevance[0], -music_relevance[1],
                            score.frame_index, motion_id, sampler, score))
    if timing_samples is not None:
        timing_samples.append(time.perf_counter() - started)
    if not choices:
        return None
    chosen = min(choices, key=lambda item: item[:5])
    return chosen[4], chosen[5], chosen[6], False


def replay_choice(current_id, sampler, source_features, limits, args):
    score = select_motion_entry(
        source_features, 1.0, sampler.entry_features, limits,
        beat_period=0.5, beats_per_bar=4, minimum_remaining_bars=0,
        speed_max=args.speed_max, transition_minimum=args.transition_min_seconds,
        transition_maximum=math.inf, enforce_remaining=False,
        scoring_config=EntryScoringConfig.from_args(args),
    )
    assert score is not None
    return current_id, sampler, score, True


@dataclass(frozen=True)
class PreparedStateBridge:
    generation: int
    motion_id: str
    sampler: Any
    score: MotionEntryScore
    trajectory: Any
    names: tuple[str, ...]
    replay: bool
    residuals: tuple[float, float, float]
    exit_frame_index: int = -1
    exit_seconds: float = -1.0
    pair_count: int = 0
    bridge_attempts: int = 0
    preparation_seconds: float = 0.0
    source_duration: float = 0.0


def authored_features(sampler):
    feature = sampler.entry_features
    authored = feature.authored
    return replace(feature, joint_velocities=authored.velocities, duration=authored.duration)


def prepare_state_bridge(generation, current_id, source_sampler, ready, limits, ranges, jerks, args,
                         source_seconds=0.0, source_entry=0.0, cancel=None):
    """Merge sorted exit/entry rows without constructing pair-sized Python objects."""
    started = time.perf_counter()
    source = authored_features(source_sampler)
    rate = max(1.0, args.speed_max) if args.motion_timing == "beat-sync" else 1.0
    deadline = started + max(0.0, source.duration-source_seconds)/rate
    window = getattr(args, "transition_exit_window_seconds", 2.0)
    boundary = args.transition_boundary_seconds
    names = tuple(sorted(limits))
    lower = np.array([ranges[n][0] if ranges.get(n) is not None else -math.inf for n in names])
    upper = np.array([ranges[n][1] if ranges.get(n) is not None else math.inf for n in names])
    speed = np.array([limits[n].max_speed_rad_s for n in names])
    acceleration = np.array([limits[n].max_acceleration_rad_s2 for n in names])
    jerk = np.array([jerks[n] for n in names])
    pair_count = attempts = 0

    def check():
        if (cancel is not None and cancel.is_set()) or time.perf_counter() >= deadline:
            raise InfeasibleBridge("Transition preparation deadline expired or cancelled")

    def reachable(index):
        seconds = exit_time(index)
        progress = source_seconds + rate*(time.perf_counter()-started)
        if index == len(source.joint_positions)-1:
            return progress < source.duration
        return (seconds >= source_entry+2*boundary and
                progress <= min(seconds-boundary, source.duration-boundary))

    def exit_time(index):
        return source.duration if index == len(source.joint_positions)-1 else index/source.fps

    def state_at(feature, index):
        if not set(names).issubset(feature.joint_names):
            raise InfeasibleBridge("Motion is missing limited joints")
        columns = [feature.joint_names.index(n) for n in names]
        state = feature.authored.state(index)
        return JointState(*(getattr(state, f)[columns] for f in ("position", "velocity", "acceleration")))

    def valid(state):
        return (all(np.all(np.isfinite(v)) for v in (state.position, state.velocity, state.acceleration))
                and np.all(state.position >= lower-1e-8) and np.all(state.position <= upper+1e-8)
                and np.all(np.abs(state.velocity) <= speed+1e-7)
                and np.all(np.abs(state.acceleration) <= acceleration+1e-7))

    def boundary_mask(feature):
        if not set(names).issubset(feature.joint_names):
            return np.zeros(len(feature.joint_positions), dtype=bool)
        columns = [feature.joint_names.index(n) for n in names]
        q = feature.authored.positions[:, columns]
        v = feature.authored.velocities[:, columns]
        a = feature.authored.accelerations[:, columns]
        return (np.all(np.isfinite(q) & np.isfinite(v) & np.isfinite(a), axis=1)
                & np.all((q >= lower-1e-8) & (q <= upper+1e-8), axis=1)
                & np.all(np.abs(v) <= speed+1e-7, axis=1)
                & np.all(np.abs(a) <= acceleration+1e-7, axis=1))

    exits = [i for i in range(len(source.joint_positions))
             if exit_time(i) >= source.duration-window-1e-9 and reachable(i)]
    starts = {i: state_at(source, i) for i in exits}
    exits = [i for i in exits if valid(starts[i])]
    config = EntryScoringConfig.from_args(args)

    def row(scores, exit_index, motion_id, sampler, relevance, replay):
        for score in scores:
            yield (score.total, -relevance[0], -relevance[1], -exit_index,
                   score.frame_index, motion_id, sampler, score, replay)

    last_reason = "No eligible exit/entry pair"
    for replay, candidates in ((False, ready), (True, ((current_id, source_sampler, (0., 0.)),))):
        rows = []
        targets = {}
        for motion_id, sampler, relevance in candidates:
            target = authored_features(sampler)
            targets[motion_id] = target
            mask = boundary_mask(target)
            mask &= (np.arange(len(mask)) == 0 if replay else
                     np.arange(len(mask))/max(len(mask)-1, 1) <= args.entry_max_phase+1e-9)
            eligible_count = int(np.count_nonzero(mask))
            if not eligible_count:
                continue
            for index in exits:
                check()
                if not reachable(index):
                    continue
                scores = select_motion_entry(
                    source, index/max(len(source.joint_positions)-1, 1), target, limits,
                    beat_period=.5, beats_per_bar=4, minimum_remaining_bars=0, speed_max=rate,
                    transition_minimum=args.transition_min_seconds, transition_maximum=10,
                    entry_max_phase=args.entry_max_phase,
                    enforce_remaining=not replay, return_all="iterator", scoring_config=config,
                    eligible_frame_mask=mask,
                )
                if scores is not None:
                    pair_count += eligible_count
                    rows.append(row(scores, index, motion_id, sampler, relevance, replay))
        # Each row is sorted by total then entry. Merge compares only stable scalar keys.
        for item in heapq.merge(*rows, key=lambda item: item[:6]):
            check()
            _, _, _, negative_exit, _, motion_id, sampler, score, replay = item
            index = -negative_exit
            if not reachable(index):
                continue
            start = starts[index]
            try:
                end = state_at(targets[motion_id], score.frame_index)
                if not valid(end):
                    last_reason = "Boundary state exceeds joint limits"
                    continue
                attempts += 1
                bridge = make_bridge(args.transition_backend, start, end, lower, upper,
                                     speed, acceleration, jerk, args.transition_min_seconds,
                                     deadline=deadline, cancel=cancel)
            except InfeasibleBridge as exc:
                last_reason = (f"motion={motion_id} exit_frame={index} entry_frame={score.frame_index}: {exc}; "
                               f"joint_order={','.join(names)}")
                if attempts <= 3:
                    print(f"Bridge rejection: {last_reason}")
                continue
            check()
            if not reachable(index):
                continue
            residuals = tuple(max(float(np.max(np.abs(getattr(bridge.at_time(t), field)-getattr(state, field))))
                              for t, state in ((0, start), (bridge.duration, end)))
                              for field in ("position", "velocity", "acceleration"))
            return PreparedStateBridge(generation, motion_id, sampler,
                replace(score, transition_seconds=bridge.duration), bridge, names, replay, residuals,
                index, exit_time(index), pair_count, attempts, time.perf_counter()-started, source.duration)
    raise InfeasibleBridge(f"Candidates and replay infeasible: {last_reason}")


def neutral_authored_frame(sampler, seconds):
    authored = sampler.entry_features.authored
    phase = min(max(seconds / authored.duration, 0.0), 1.0)
    count = len(sampler.frames)
    frame = sampler.sample_frame(phase * (count-1)/max(count, 1), 1.0, 0.0, base.FeatureState())
    return frame.with_joint_positions(dict(zip(sampler.entry_features.joint_names,
                                              authored.at_time(seconds).position)))


class AuthoredPlayback:
    """V2 state bridge lifecycle. Legacy quintic playback remains separate."""

    def __init__(self, current_id, sampler, controller, loader, catalog, args, limits, ranges, jerks, now):
        self.current_id, self.sampler, self.controller = current_id, sampler, controller
        self.loader, self.catalog, self.args = loader, catalog, args
        self.limits, self.ranges, self.jerks = limits, ranges, jerks
        self.generation = 0
        self.clock = AuthoredClock(sampler.entry_features.authored.duration, 0,
                                   args.transition_boundary_seconds, now)
        self.frozen = () if args.no_mic and args.audio_input is None else None
        self.future = self.plan = self.active = None
        self.future_generation = None
        self.root_source = self.root_target = None
        self.bridge_source = self.bridge_target = self.bridge_target_raw = None
        self.bridge_start = None
        self.status = "awaiting_retrieval"
        self.failure = ""
        self.event = ""
        self.reason = ""
        self.score = None
        self.last_residuals = None
        self.blend = 0.
        self.duration = 0.
        self.last_terminal = None
        self.held = False
        self.latest_match = None
        self.search_cancel = threading.Event()
        self.last_plan = None
        self.preparation_started = None
        self.expansion_attempted = False

    def prepare(self, match, now=None):
        self.latest_match = match
        rate = max(1., self.args.speed_max) if self.args.motion_timing == "beat-sync" else 1.
        progress = self.clock.seconds + (max(0., now-self.clock.last_wall)*rate if now is not None else 0.)
        if self.held or self.active is not None:
            return
        if self.frozen is None and (match is not None or
                self.clock.duration-self.clock.seconds <= self.args.transition_boundary_seconds):
            ids = musical_shortlist(match, self.current_id, self.args) if match is not None else ()
            relevance = {m.motion_id: (m.final_score, m.music_score) for m in match.motions} if match else {}
            self.frozen = tuple((mid, relevance.get(mid, (0., 0.))) for mid in ids)
        if self.frozen is None:
            return
        if self.future is not None and self.future.done():
            future, generation = self.future, self.future_generation
            self.future = None
            try:
                plan = future.result()
                if generation == self.generation and plan.generation == self.generation:
                    exit_time = plan.exit_seconds if plan.exit_seconds >= 0 else self.clock.duration
                    if exit_time < self.clock.duration and (
                            progress > min(exit_time-self.clock.window,
                                                     self.clock.duration-self.clock.window)
                            or exit_time < self.clock.entry+2*self.clock.window):
                        self.status = "late_plan_discarded"
                        return
                    self.clock.stop = exit_time
                    self.plan = plan
                    self.last_plan = plan
                    self.status = "ready"
            except Exception as exc:
                if generation == self.generation:
                    self.status, self.failure = "failed", str(exc)
            finally:
                if hasattr(self.loader, "score_seconds") and self.preparation_started is not None:
                    self.loader.score_seconds.append(time.perf_counter()-self.preparation_started)
            return
        if self.status == "failed" and not self.expansion_attempted and match is not None:
            if progress < self.clock.duration-self.args.transition_boundary_seconds and match.accepted:
                self.expansion_attempted = True
                ids = musical_shortlist(match, self.current_id, self.args, expanded=True,
                                        exclude={mid for mid, _ in self.frozen})
                relevance = {m.motion_id: (m.final_score, m.music_score) for m in match.motions}
                if set(ids) - {mid for mid, _ in self.frozen}:
                    self.frozen = tuple((mid, relevance[mid]) for mid in ids)
                    self.status, self.failure = "expanded_retry", ""
                    print(f"Retrying bridge preparation with expanded shortlist: {', '.join(ids)}")
        if self.plan is not None or self.future is not None or self.status == "failed":
            return
        ids = [mid for mid, _ in self.frozen]
        self.loader.preload(ids)
        self.loader.prepare_pool(ids)
        ready = []
        for mid, relevance in self.frozen:
            sampler = self.loader.take_ready(mid)
            if sampler is not None:
                ready.append((mid, sampler, relevance))
            elif mid not in self.loader.failed_motions:
                self.status = "loading"
                return
        self.future_generation = self.generation
        self.preparation_started = time.perf_counter()
        self.future = self.loader.score_executor.submit(
            prepare_state_bridge, self.generation, self.current_id, self.sampler,
            tuple(ready), self.limits, self.ranges, self.jerks, self.args,
            progress, self.clock.entry, self.search_cancel,
        )
        self.status = "preparing"

    def aligned(self, frame):
        if self.root_source is not None:
            return align_motion_frame_root(frame, source_reference=self.root_source,
                                           target_reference=self.root_target)
        return frame

    def sample(self, now, features, modulator, adapter):
        self.event, self.blend = "", 0.
        if self.held:
            return self.aligned(neutral_authored_frame(self.sampler, self.clock.duration))
        if self.active is None:
            if self.clock.weight < 1.0:
                self.controller.phase_correction_remaining = 0.0
            _, amplitude, accent, _ = self.controller.update(now)
            rate = self.controller.speed_multiplier if self.args.motion_timing == "beat-sync" else 1.
            carry = self.clock.advance(now, rate)
            phase = self.clock.seconds/self.clock.duration
            self.controller.phase = min(phase, OneShotPhaseTracker.terminal_phase)
            neutral = neutral_authored_frame(self.sampler, self.clock.seconds)
            if self.clock.seconds < self.clock.stop:
                count = len(self.sampler.frames)
                effected = base.sample_robot_motion_frame(
                    self.sampler, phase=phase*(count-1)/max(count, 1), amplitude=amplitude,
                    accent=accent, features=features, pose_adapter=adapter, modulator=modulator)
                # Apply deviations from the original neutral sampler to the C2
                # authored trajectory, rather than replacing its interpolation.
                raw = self.sampler.sample_frame(phase*(count-1)/max(count, 1), 1., 0., base.FeatureState())
                raw_joints = adapter.adapt_pose(raw.joint_positions, base.FeatureState()) if adapter else raw.joint_positions
                weight = self.clock.weight
                joints = {n: q + weight*(effected.joint_positions.get(n, q)-raw_joints.get(n, q))
                          for n, q in neutral.joint_positions.items()}
                return self.aligned(neutral.with_joint_positions(joints))
            self.last_terminal = now-carry
            if self.plan is None:
                self.held = True
                self.search_cancel.set()
                self.last_residuals = None
                self.failure = self.failure or "Bridge preparation did not finish before the terminal state"
                self.status, self.event = "terminal_hold", "bridge_hold"
                return self.aligned(neutral)
            self.active, self.plan = self.plan, None
            self.bridge_start = now-carry
            self.bridge_source = self.aligned(neutral)
            self.bridge_target_raw = neutral_authored_frame(self.active.sampler, self.active.score.frame_index/self.active.sampler.entry_features.fps)
            self.bridge_target = align_motion_frame_root(self.bridge_target_raw,
                source_reference=self.bridge_target_raw, target_reference=self.bridge_source)
            self.score = self.active.score
            self.duration = self.active.trajectory.duration
            self.last_residuals = self.active.residuals
            self.reason = "replay_no_feasible_candidate" if self.active.replay else (
                "selected_exit" if self.clock.stop < self.clock.duration else "motion_end")
            self.status, self.event = "executing", "switch_start"
        seconds = now-self.bridge_start
        self.blend = min(seconds/self.duration, 1.)
        frame = quintic_blend(self.bridge_source, self.bridge_target, self.blend)
        frame = frame.with_joint_positions({**frame.joint_positions,
                    **dict(zip(self.active.names, self.active.trajectory.at_time(seconds).position))})
        if seconds < self.duration:
            return frame
        plan = self.active
        self.current_id, self.sampler = plan.motion_id, plan.sampler
        self.controller = make_controller(self.args, self.catalog.motions[self.current_id],
                                         previous=self.controller, entry_phase=plan.score.phase)
        self.controller.phase_correction_remaining = 0.
        authored = self.sampler.entry_features.authored
        entry = plan.score.frame_index/authored.fps
        end_wall = self.bridge_start+self.duration
        self.clock = AuthoredClock(authored.duration, entry, self.args.transition_boundary_seconds, end_wall)
        self.controller.last_update_wall = end_wall
        self.root_source, self.root_target = self.bridge_target_raw, self.bridge_source
        self.active = self.future = None
        self.search_cancel.set()
        self.search_cancel = threading.Event()
        self.generation += 1
        self.expansion_attempted = False
        self.last_plan = None
        ids = musical_shortlist(self.latest_match, self.current_id, self.args) if self.latest_match else ()
        relevance = {m.motion_id: (m.final_score, m.music_score) for m in self.latest_match.motions} if self.latest_match else {}
        self.frozen = tuple((mid, relevance.get(mid, (0., 0.))) for mid in ids)
        self.status, self.failure = "awaiting_preparation", ""
        # Reuse normal playback evaluation with wall-time carryover at entry.
        result = self.sample(now, features, modulator, adapter)
        self.event, self.blend = "switch_complete", 1.
        return result


class MicrophoneSource:
    def __init__(self, analyzer: base.RealtimeMusicAnalyzer, window_seconds: float, minimum_seconds: float = 2.0) -> None:
        self.analyzer = analyzer
        self.sample_rate = analyzer.sample_rate
        self.window_seconds = float(window_seconds)
        self.history = RetrievalAudioHistory(analyzer, window_seconds, minimum_seconds)

    @property
    def playback_seconds(self) -> float:
        return time.perf_counter()

    @property
    def done(self) -> bool:
        return False

    def start(self) -> None:
        self.analyzer.start()

    def stop(self) -> None:
        self.analyzer.stop()

    def advance(self, dt: float) -> None:
        del dt

    def drain(self) -> list[base.MusicFrame]:
        return self.analyzer.drain()

    def recent_audio(self) -> np.ndarray | None:
        return self.history.recent_audio()


class SilentSource:
    """No-input source used for headless and hardware-free smoke tests."""

    @property
    def playback_seconds(self) -> float:
        return time.perf_counter()

    @property
    def done(self) -> bool:
        return False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def advance(self, dt: float) -> None:
        del dt

    def drain(self) -> list[base.MusicFrame]:
        return []

    def recent_audio(self) -> None:
        return None


class MatcherFileMicrophoneSource(base.FileMicrophoneSource):
    """Virtual file microphone with matcher-specific beat and window handling."""

    def __init__(
        self,
        path: Path,
        args: argparse.Namespace,
        window_seconds: float,
    ) -> None:
        self.window_seconds = float(window_seconds)
        analyzer = base.make_analyzer(args)
        self.minimum_seconds = args.analysis_min_seconds
        super().__init__(
            path,
            analyzer,
            startup_delay_sec=args.audio_input_delay_sec,
            throttle=not args.realtime,
            play_audio=getattr(args, "play_audio", False),
        )
        self._pending_beats: list[float] = []
        self._pending_beat_contrasts: list[float] = []
        self._next_beat = 0
        self._beat_period: float | None = None
        self.causal_input = getattr(args, "experiment_causal_file_input", False)
        self._feeder_stop = threading.Event()
        self._feeder_thread: threading.Thread | None = None
        self._feeder_error: BaseException | None = None
        if not self.causal_input:
            self._analyze_beats()

    def start(self) -> None:
        if not self.causal_input:
            return super().start()
        # Warm the analyzer's process worker before starting the experiment clock.
        # Otherwise its first spawn/import can pre-empt the control loop seconds later.
        prepare = getattr(self.analyzer, "prepare_background_analysis", None)
        if callable(prepare):
            prepare()
        super().start()
        self._feeder_stop.clear()
        self._feeder_error = None
        self._feeder_thread = threading.Thread(
            target=self._run_causal_feeder,
            name="causal-file-microphone",
            daemon=True,
        )
        self._feeder_thread.start()

    def stop(self) -> None:
        feeder = self._feeder_thread
        self._feeder_thread = None
        if feeder is not None:
            self._feeder_stop.set()
            feeder.join(timeout=2.0)
        super().stop()
        self._raise_feeder_error()

    def _raise_feeder_error(self) -> None:
        if self._feeder_error is not None:
            raise RuntimeError("Causal file microphone feeder failed.") from self._feeder_error

    def _run_causal_feeder(self) -> None:
        try:
            _set_background_worker_priority()
            total = self.startup_samples + self.audio.size
            while self.cursor < total and not self._feeder_stop.is_set():
                block_size = min(self.block_size, total - self.cursor)
                assert self.stream_start_wall is not None
                deadline = self.stream_start_wall + (self.cursor + block_size) / self.sample_rate
                remaining = deadline - time.perf_counter()
                if remaining > 0.0:
                    self._feeder_stop.wait(remaining)
                    if self._feeder_stop.is_set():
                        break
                self._feed_block(block_size)
        except BaseException as exc:
            self._feeder_error = exc

    def advance(self, dt: float) -> None:
        if not self.causal_input:
            return super().advance(dt)
        if self._feeder_thread is not None:
            self._raise_feeder_error()
            return
        # Absolute receive clock: never accumulate nominal dt ahead of real audio.
        now = time.perf_counter()
        total = self.startup_samples + self.audio.size
        target = min(int(max(now - self.stream_start_wall, 0.0) * self.sample_rate), total)
        while target - self.cursor >= self.block_size:
            self._feed_block(self.block_size)
        if target == total and self.cursor < total:
            self._feed_block(total - self.cursor)

    def _analyze_beats(self) -> None:
        hop_length = self.analyzer.plp_hop_length
        onset = librosa.onset.onset_strength(
            y=self.audio,
            sr=self.sample_rate,
            hop_length=hop_length,
        )
        _tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset,
            sr=self.sample_rate,
            hop_length=hop_length,
        )
        self._pending_beats = [
            float(value)
            for value in librosa.frames_to_time(
                beat_frames,
                sr=self.sample_rate,
                hop_length=hop_length,
            )
        ]
        self._pending_beat_contrasts = base.compute_beat_contrasts(
            self.audio,
            onset,
            np.asarray(beat_frames, dtype=int),
            self.sample_rate,
            hop_length,
        ).tolist()
        if len(self._pending_beats) >= 2:
            self._beat_period = float(np.median(np.diff(self._pending_beats)))

    def drain(self) -> list[base.MusicFrame]:
        if self.causal_input:
            return super().drain()
        frames = [frame for frame in super().drain() if not frame.is_beat]
        playback = self.playback_seconds
        now = time.perf_counter()
        template = frames[-1] if frames else self.analyzer.latest_status_frame
        while (
            template is not None
            and self._next_beat < len(self._pending_beats)
            and self._pending_beats[self._next_beat] <= playback
        ):
            frames.append(
                replace(
                    template,
                    timestamp=now - max(playback - self._pending_beats[self._next_beat], 0.0),
                    beat_period=self._beat_period,
                    beat_confidence=float(
                        np.clip(
                            self._pending_beat_contrasts[self._next_beat],
                            0.0,
                            1.0,
                        )
                    ),
                    beat_contrast=self._pending_beat_contrasts[self._next_beat],
                    is_beat=True,
                    is_active=True,
                )
            )
            self._next_beat += 1
        frames.sort(key=lambda frame: frame.timestamp)
        return frames

    def recent_audio(self) -> np.ndarray | None:
        cursor = int(np.clip(self.cursor - self.startup_samples, 0, self.audio.size))
        start = max(0, cursor - int(round(self.window_seconds * self.sample_rate)))
        return available_history(self.audio[start:cursor], self.sample_rate,
                                 self.window_seconds, self.minimum_seconds)


@dataclass(frozen=True)
class RetrievalProcessConfig:
    catalog_path: Path
    sample_rate: int
    embedding_model: Path | None
    tag_model: Path | None
    speed_min: float
    speed_max: float
    style_first: bool
    weak_music_threshold: float
    weak_music_consecutive_windows: int
    motion_compatibility: bool


@dataclass(frozen=True)
class RetrievalJobResult:
    result: MatchResult
    feature_seconds: float
    match_seconds: float
    feature_stage_ms: dict[str, float]
    matcher_stage_ms: dict[str, float]


_RETRIEVAL_PROCESS_EXTRACTOR: AudioFeatureExtractor | None = None
_RETRIEVAL_PROCESS_MATCHER: MusicMotionMatcher | None = None


def _initialize_retrieval_process(config: RetrievalProcessConfig) -> None:
    global _RETRIEVAL_PROCESS_EXTRACTOR, _RETRIEVAL_PROCESS_MATCHER
    _set_retrieval_process_priority()
    catalog = MusicCatalog.load(config.catalog_path)
    _RETRIEVAL_PROCESS_EXTRACTOR = AudioFeatureExtractor(
        sample_rate=config.sample_rate,
        embedding_model=config.embedding_model,
        tag_model=config.tag_model,
        onnx_intra_op_threads=1,
    )
    _RETRIEVAL_PROCESS_MATCHER = MusicMotionMatcher(
        catalog,
        speed_min=config.speed_min,
        speed_max=config.speed_max,
        style_first=config.style_first,
        weak_music_threshold=config.weak_music_threshold,
        weak_music_consecutive_windows=config.weak_music_consecutive_windows,
        motion_compatibility=config.motion_compatibility,
    )


def _warm_retrieval_process() -> bool:
    extractor = _RETRIEVAL_PROCESS_EXTRACTOR
    matcher = _RETRIEVAL_PROCESS_MATCHER
    if extractor is None or matcher is None:
        return False
    # Exercise the cold librosa/Numba and ONNX paths before the experiment
    # clock begins. Silence takes a fast path and would not actually warm them.
    duration = 6.0
    samples = np.arange(int(round(duration * extractor.sample_rate)), dtype=np.float32)
    samples = (0.02 * np.sin(samples * (2.0 * np.pi * 220.0 / extractor.sample_rate))).astype(
        np.float32
    )
    descriptor = extractor.describe(samples)
    matcher.match(descriptor, top_k_tracks=1, top_k_motions=1)
    return True


def _run_retrieval_process(
    samples: np.ndarray,
    top_tracks: int,
    top_motions: int,
    source_sample_rate: int | None = None,
) -> RetrievalJobResult:
    extractor = _RETRIEVAL_PROCESS_EXTRACTOR
    matcher = _RETRIEVAL_PROCESS_MATCHER
    if extractor is None or matcher is None:
        raise RuntimeError("Retrieval process was not initialized.")
    feature_started = time.perf_counter()
    descriptor = extractor.describe(samples, source_sample_rate=source_sample_rate)
    feature_seconds = max(time.perf_counter() - feature_started, 0.0)
    match_started = time.perf_counter()
    result = matcher.match(
        descriptor,
        top_k_tracks=top_tracks,
        top_k_motions=top_motions,
    )
    match_seconds = max(time.perf_counter() - match_started, 0.0)
    return RetrievalJobResult(
        result=result,
        feature_seconds=feature_seconds,
        match_seconds=match_seconds,
        feature_stage_ms=dict(extractor.last_timing_ms),
        matcher_stage_ms=dict(matcher.last_timing_ms),
    )


class RetrievalWorker:
    def __init__(
        self,
        extractor: AudioFeatureExtractor,
        matcher: MusicMotionMatcher,
        top_tracks: int,
        top_motions: int,
        process_config: RetrievalProcessConfig | None = None,
    ) -> None:
        self.extractor = extractor
        self.matcher = matcher
        self.top_tracks = max(int(top_tracks), 1)
        self.top_motions = max(int(top_motions), 1)
        self.process_isolated = process_config is not None
        if process_config is None:
            self.executor: ThreadPoolExecutor | ProcessPoolExecutor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="music-match",
                initializer=_set_background_worker_priority,
            )
        else:
            self.executor = ProcessPoolExecutor(
                max_workers=1,
                initializer=_initialize_retrieval_process,
                initargs=(process_config,),
            )
            if not self.executor.submit(_warm_retrieval_process).result():
                raise RuntimeError("Could not initialize isolated retrieval worker.")
        self.future: Future[MatchResult | RetrievalJobResult] | None = None
        self.submitted_at: float | None = None
        self.submit_attempts = 0
        self.accepted_submissions = 0
        self.completed = 0
        self.copy_seconds: list[float] = []
        self.queue_seconds: list[float] = []
        self.feature_seconds: list[float] = []
        self.match_seconds: list[float] = []
        self.total_seconds: list[float] = []
        self.feature_stage_ms: dict[str, list[float]] = {}
        self.matcher_stage_ms: dict[str, list[float]] = {}
        self.selection_seconds: list[float] = []

    def submit(self, audio: np.ndarray, source_sample_rate: int | None = None) -> bool:
        self.submit_attempts += 1
        # A completed result still belongs to poll(); replacing that Future here
        # would silently discard the query immediately before it is consumed.
        if self.future is not None:
            return False
        copy_started = time.perf_counter()
        samples = np.asarray(audio, dtype=np.float32).copy()
        self.copy_seconds.append(max(time.perf_counter() - copy_started, 0.0))
        self.submitted_at = time.perf_counter()
        self.accepted_submissions += 1
        if self.process_isolated:
            self.future = self.executor.submit(
                _run_retrieval_process,
                samples,
                self.top_tracks,
                self.top_motions,
                source_sample_rate,
            )
        else:
            self.future = self.executor.submit(self._run, samples, source_sample_rate)
        return True

    def _run(self, samples: np.ndarray, source_sample_rate: int | None = None) -> MatchResult:
        run_started = time.perf_counter()
        if self.submitted_at is not None:
            self.queue_seconds.append(max(run_started - self.submitted_at, 0.0))
        feature_started = time.perf_counter()
        descriptor = self.extractor.describe(samples, source_sample_rate=source_sample_rate)
        self.feature_seconds.append(max(time.perf_counter() - feature_started, 0.0))
        for key, value in self.extractor.last_timing_ms.items():
            self.feature_stage_ms.setdefault(key, []).append(float(value))
        match_started = time.perf_counter()
        result = self.matcher.match(
            descriptor,
            top_k_tracks=self.top_tracks,
            top_k_motions=self.top_motions,
        )
        self.match_seconds.append(max(time.perf_counter() - match_started, 0.0))
        for key, value in self.matcher.last_timing_ms.items():
            self.matcher_stage_ms.setdefault(key, []).append(float(value))
        submitted = self.submitted_at if self.submitted_at is not None else run_started
        self.total_seconds.append(max(time.perf_counter() - submitted, 0.0))
        return result

    def poll(self) -> MatchResult | None:
        if self.future is None or not self.future.done():
            return None
        future = self.future
        self.future = None
        outcome = future.result()
        if isinstance(outcome, RetrievalJobResult):
            self.feature_seconds.append(outcome.feature_seconds)
            self.match_seconds.append(outcome.match_seconds)
            for key, value in outcome.feature_stage_ms.items():
                self.feature_stage_ms.setdefault(key, []).append(float(value))
            for key, value in outcome.matcher_stage_ms.items():
                self.matcher_stage_ms.setdefault(key, []).append(float(value))
            submitted = (
                self.submitted_at
                if self.submitted_at is not None
                else time.perf_counter()
            )
            self.total_seconds.append(max(time.perf_counter() - submitted, 0.0))
            result = outcome.result
        else:
            result = outcome
        self.completed += 1
        return result

    def record_selection_time(self, seconds: float) -> None:
        self.selection_seconds.append(max(float(seconds), 0.0))

    def timing_summary(self) -> dict[str, float | int]:
        def metric(prefix: str, values: list[float], *, milliseconds: bool = True) -> dict[str, float]:
            samples = np.asarray(values, dtype=np.float64)
            if milliseconds:
                samples = samples * 1_000.0
            result: dict[str, float] = {}
            for label, subset in (
                ("", samples),
                ("_cold", samples[:1]),
                ("_warm", samples[1:]),
            ):
                for quantile_label, quantile in (("p50", 50.0), ("p95", 95.0), ("p99", 99.0)):
                    result[f"{prefix}{label}_{quantile_label}"] = (
                        float(np.percentile(subset, quantile)) if subset.size else 0.0
                    )
                result[f"{prefix}{label}_max"] = (
                    float(np.max(subset)) if subset.size else 0.0
                )
            return result

        result: dict[str, float | int] = {
            "retrieval_submit_attempts": self.submit_attempts,
            "retrieval_submitted": self.accepted_submissions,
            "retrieval_completed": self.completed,
            "retrieval_busy_submissions": self.submit_attempts - self.accepted_submissions,
            "retrieval_process_isolated": int(self.process_isolated),
        }
        result.update(metric("retrieval_audio_copy_ms", self.copy_seconds))
        result.update(metric("retrieval_queue_ms", self.queue_seconds))
        result.update(metric("retrieval_feature_ms", self.feature_seconds))
        result.update(metric("retrieval_match_ms", self.match_seconds))
        result.update(metric("selection_policy_ms", self.selection_seconds))
        result.update(metric("retrieval_waveform_to_match_ms", self.total_seconds))
        for key, values in sorted(self.feature_stage_ms.items()):
            result.update(metric(f"retrieval_{key}", values, milliseconds=False))
        for key, values in sorted(self.matcher_stage_ms.items()):
            result.update(metric(f"retrieval_{key}", values, milliseconds=False))
        return result

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


MotionSampler = base.AistppMotionSampler | base.GmrUnitreeG1MotionSampler


@dataclass(frozen=True)
class MotionPreparationResult:
    sampler: MotionSampler
    load_seconds: float
    prepare_seconds: float
    feature_seconds: float


_MOTION_PROCESS_ARGS: argparse.Namespace | None = None
_MOTION_PROCESS_CATALOG: MusicCatalog | None = None
_MOTION_PROCESS_ADAPTER: Any = None
_MOTION_PROCESS_PLAYER: base.MujocoHumanoidPlayer | None = None


def _initialize_motion_prepare_process(
    args: argparse.Namespace,
    catalog: MusicCatalog,
    adapter: Any,
) -> None:
    global _MOTION_PROCESS_ARGS, _MOTION_PROCESS_CATALOG
    global _MOTION_PROCESS_ADAPTER, _MOTION_PROCESS_PLAYER
    _set_retrieval_process_priority()
    _MOTION_PROCESS_ARGS = args
    _MOTION_PROCESS_CATALOG = catalog
    _MOTION_PROCESS_ADAPTER = adapter
    _MOTION_PROCESS_PLAYER = base.MujocoHumanoidPlayer(
        args.model,
        realtime=False,
        headless=True,
    )


def _motion_prepare_process_ready() -> bool:
    return _MOTION_PROCESS_PLAYER is not None


def _run_motion_prepare_process(motion_id: str) -> MotionPreparationResult:
    args = _MOTION_PROCESS_ARGS
    catalog = _MOTION_PROCESS_CATALOG
    player = _MOTION_PROCESS_PLAYER
    if args is None or catalog is None or player is None:
        raise RuntimeError("Motion preparation process was not initialized.")
    load_started = time.perf_counter()
    sampler = load_motion_sampler(args, catalog, catalog.motions[motion_id])
    load_seconds = max(time.perf_counter() - load_started, 0.0)
    prepare_started = time.perf_counter()
    player.ground_sampler(sampler, _MOTION_PROCESS_ADAPTER, cooperative_yield=False)
    prepare_seconds = max(time.perf_counter() - prepare_started, 0.0)
    feature_started = time.perf_counter()
    setattr(
        sampler,
        "entry_features",
        build_motion_entry_features(
            sampler,
            catalog.motions[motion_id],
            _MOTION_PROCESS_ADAPTER,
            player.actuator_joint_ranges,
        ),
    )
    feature_seconds = max(time.perf_counter() - feature_started, 0.0)
    return MotionPreparationResult(
        sampler=sampler,
        load_seconds=load_seconds,
        prepare_seconds=prepare_seconds,
        feature_seconds=feature_seconds,
    )


class MotionLoader:
    def __init__(self, catalog: MusicCatalog, args: argparse.Namespace, adapter: Any) -> None:
        self.catalog = catalog
        self.args = args
        self.adapter = adapter
        self.ready_pool_size = max(int(args.switch_ready_pool_size), 1)
        self.max_cached = max(int(args.motion_cache_size), self.ready_pool_size + 1)
        self.process_isolated = bool(getattr(args, "realtime", False))
        self.executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="motion-load",
            initializer=_set_background_worker_priority,
        )
        if self.process_isolated:
            self.prepare_executor: ThreadPoolExecutor | ProcessPoolExecutor = ProcessPoolExecutor(
                max_workers=1,
                initializer=_initialize_motion_prepare_process,
                initargs=(args, catalog, adapter),
            )
            if not self.prepare_executor.submit(_motion_prepare_process_ready).result():
                raise RuntimeError("Could not initialize isolated motion preparation worker.")
        else:
            self.prepare_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="motion-prepare",
                initializer=_set_background_worker_priority,
            )
        self.score_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="motion-score",
            initializer=_set_background_worker_priority,
        )
        self.grounding_player = base.MujocoHumanoidPlayer(
            args.model,
            realtime=False,
            headless=True,
        )
        self.futures: dict[str, Future[MotionSampler]] = {}
        self.prepare_futures: dict[str, Future[MotionSampler]] = {}
        self.cache: OrderedDict[str, MotionSampler] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.load_seconds: list[float] = []
        self.prepare_seconds: list[float] = []
        self.feature_seconds: list[float] = []
        self.ready_seconds: list[float] = []
        self.score_seconds: list[float] = []
        self.submitted_at: dict[str, float] = {}
        self.ready_pool_underruns = 0
        self.desired_ids: tuple[str, ...] = ()
        self.failed_motions: dict[str, str] = {}

    def submit(self, motion_id: str) -> None:
        self.preload((motion_id,))

    def preload(self, motion_ids: tuple[str, ...] | list[str]) -> None:
        desired = tuple(dict.fromkeys(motion_ids))[: self.max_cached]
        self.desired_ids = desired
        desired_set = set(desired)
        for motion_id, future in list(self.futures.items()):
            if motion_id not in desired_set and future.cancel():
                del self.futures[motion_id]
                self.submitted_at.pop(motion_id, None)
        for motion_id in desired:
            if (
                motion_id in self.cache
                or motion_id in self.futures
                or motion_id in self.prepare_futures
                or motion_id in self.failed_motions
            ):
                continue
            if self.process_isolated:
                continue
            self.submitted_at[motion_id] = time.perf_counter()
            self.futures[motion_id] = self.executor.submit(
                self._load_sampler,
                motion_id,
            )

    def take_ready(self, motion_id: str) -> MotionSampler | None:
        if motion_id in self.cache:
            self.cache_hits = getattr(self, "cache_hits", 0) + 1
            if hasattr(self.cache, "move_to_end"):
                self.cache.move_to_end(motion_id)
            return self.cache[motion_id]
        self.cache_misses = getattr(self, "cache_misses", 0) + 1
        future = self.prepare_futures.get(motion_id)
        if future is None or not future.done():
            return None
        try:
            sampler = self._record_preparation_outcome(motion_id, future.result())
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self.failed_motions[motion_id] = reason
            print(f"Warning: motion preparation failed for {motion_id}: {reason}")
            del self.prepare_futures[motion_id]
            self.futures.pop(motion_id, None)
            self.submitted_at.pop(motion_id, None)
            return None
        del self.prepare_futures[motion_id]
        self.futures.pop(motion_id, None)
        self.cache[motion_id] = sampler
        if hasattr(self.cache, "move_to_end"):
            self.cache.move_to_end(motion_id)
        while len(self.cache) > self.max_cached:
            removable = next(
                (
                    cached_id
                    for cached_id in self.cache
                    if cached_id not in self.desired_ids and cached_id != motion_id
                ),
                None,
            )
            if removable is None:
                break
            del self.cache[removable]
            self.cache_evictions = getattr(self, "cache_evictions", 0) + 1
        return sampler

    def prepare(self, motion_id: str) -> None:
        if (
            motion_id in self.cache
            or motion_id in self.prepare_futures
            or motion_id in self.failed_motions
        ):
            return
        if self.process_isolated:
            self.submitted_at.setdefault(motion_id, time.perf_counter())
            self.prepare_futures[motion_id] = self.prepare_executor.submit(
                _run_motion_prepare_process,
                motion_id,
            )
            return
        load_future = self.futures.get(motion_id)
        if load_future is None:
            self.submitted_at.setdefault(motion_id, time.perf_counter())
            sampler = self.cache.get(motion_id)
            load_future = (
                self.executor.submit(lambda value=sampler: value)
                if sampler is not None
                else self.executor.submit(
                    self._load_sampler,
                    motion_id,
                )
            )
            self.futures[motion_id] = load_future
        self.prepare_futures[motion_id] = self.prepare_executor.submit(
            self._prepare_sampler,
            motion_id,
            load_future,
        )

    def prepare_pool(self, motion_ids: tuple[str, ...] | list[str]) -> None:
        desired = tuple(dict.fromkeys(motion_ids))[: self.ready_pool_size]
        desired_set = set(desired)
        for motion_id, future in list(self.prepare_futures.items()):
            if motion_id not in desired_set and future.cancel():
                del self.prepare_futures[motion_id]
                self.futures.pop(motion_id, None)
                self.submitted_at.pop(motion_id, None)
        for motion_id in desired:
            self.prepare(motion_id)

    def wait_for_pool(
        self,
        motion_ids: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Finish the initial pool before the realtime scheduler starts."""

        desired = tuple(dict.fromkeys(motion_ids))[: self.ready_pool_size]
        for motion_id in desired:
            future = self.prepare_futures.get(motion_id)
            if future is None:
                continue
            try:
                sampler = self._record_preparation_outcome(motion_id, future.result())
            except Exception as exc:
                print(
                    f"Warning: initial motion preparation failed for {motion_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                self.prepare_futures.pop(motion_id, None)
                self.futures.pop(motion_id, None)
                self.failed_motions[motion_id] = f"{type(exc).__name__}: {exc}"
                continue
            self.prepare_futures.pop(motion_id, None)
            self.futures.pop(motion_id, None)
            self.cache[motion_id] = sampler
            if hasattr(self.cache, "move_to_end"):
                self.cache.move_to_end(motion_id)
        return tuple(motion_id for motion_id in desired if motion_id in self.cache)

    def _record_preparation_outcome(
        self,
        motion_id: str,
        outcome: MotionSampler | MotionPreparationResult,
    ) -> MotionSampler:
        if isinstance(outcome, MotionPreparationResult):
            self.load_seconds.append(outcome.load_seconds)
            self.prepare_seconds.append(outcome.prepare_seconds)
            self.feature_seconds.append(outcome.feature_seconds)
            submitted = self.submitted_at.pop(motion_id, time.perf_counter())
            self.ready_seconds.append(max(time.perf_counter() - submitted, 0.0))
            return outcome.sampler
        return outcome

    def ready_motion_ids(self, motion_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        ready = []
        for motion_id in tuple(dict.fromkeys(motion_ids)):
            sampler = self.take_ready(motion_id)
            if sampler is not None:
                ready.append(motion_id)
        if len(ready) < min(self.ready_pool_size, len(tuple(dict.fromkeys(motion_ids)))):
            self.ready_pool_underruns += 1
        return tuple(ready)

    def _prepare_sampler(
        self,
        motion_id: str,
        load_future: Future[MotionSampler],
    ) -> MotionSampler:
        sampler = load_future.result()
        prepare_started = time.perf_counter()
        self.grounding_player.ground_sampler(
            sampler,
            self.adapter,
            cooperative_yield=True,
        )
        self.prepare_seconds.append(max(time.perf_counter() - prepare_started, 0.0))
        feature_started = time.perf_counter()
        profile = self.catalog.motions[motion_id]
        setattr(
            sampler,
            "entry_features",
            build_motion_entry_features(
                sampler,
                profile,
                self.adapter,
                self.grounding_player.actuator_joint_ranges,
            ),
        )
        self.feature_seconds.append(max(time.perf_counter() - feature_started, 0.0))
        submitted = self.submitted_at.pop(motion_id, prepare_started)
        self.ready_seconds.append(max(time.perf_counter() - submitted, 0.0))
        return sampler

    def _load_sampler(self, motion_id: str) -> MotionSampler:
        started = time.perf_counter()
        sampler = load_motion_sampler(
            self.args,
            self.catalog,
            self.catalog.motions[motion_id],
        )
        self.load_seconds.append(max(time.perf_counter() - started, 0.0))
        return sampler

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.prepare_executor.shutdown(wait=True, cancel_futures=True)
        self.score_executor.shutdown(wait=True, cancel_futures=True)

    def timing_summary(self) -> dict[str, float | int]:
        def metric(prefix: str, values: list[float]) -> dict[str, float]:
            milliseconds = np.asarray(values, dtype=np.float64) * 1_000.0
            result: dict[str, float] = {}
            for label, subset in (
                ("", milliseconds),
                ("_cold", milliseconds[:1]),
                ("_warm", milliseconds[1:]),
            ):
                for quantile_label, quantile in (("p50", 50.0), ("p95", 95.0), ("p99", 99.0)):
                    result[f"{prefix}_ms{label}_{quantile_label}"] = (
                        float(np.percentile(subset, quantile)) if subset.size else 0.0
                    )
                result[f"{prefix}_ms{label}_max"] = (
                    float(np.max(subset)) if subset.size else 0.0
                )
            return result

        result: dict[str, float | int] = {
            "ready_pool_underruns": self.ready_pool_underruns,
            "motion_prepare_failures": len(self.failed_motions),
            "motion_cache_capacity": self.max_cached,
            "motion_cache_entries": len(self.cache),
            "motion_cache_hits": self.cache_hits,
            "motion_cache_misses": self.cache_misses,
            "motion_cache_evictions": self.cache_evictions,
            "motion_preparation_process_isolated": int(
                getattr(self, "process_isolated", False)
            ),
        }
        result.update(metric("motion_load", self.load_seconds))
        result.update(metric("motion_grounding", self.prepare_seconds))
        result.update(metric("motion_prepare", self.prepare_seconds))
        result.update(metric("entry_feature", self.feature_seconds))
        result.update(metric("motion_fully_ready", self.ready_seconds))
        result.update(metric("entry_score", self.score_seconds))
        return result


def motion_path(catalog: MusicCatalog, profile: MotionProfile) -> Path:
    root = Path(catalog.metadata["aistpp_root"])
    return root / Path(profile.motion_path)


def gmr_motion_path(args: argparse.Namespace, profile: MotionProfile) -> Path:
    root = args.gmr_motion_root if args.gmr_motion_root.is_absolute() else ROOT / args.gmr_motion_root
    return root / f"{profile.motion_id}.pkl"


def load_motion_sampler(
    args: argparse.Namespace,
    catalog: MusicCatalog,
    profile: MotionProfile,
) -> MotionSampler:
    gmr_path = gmr_motion_path(args, profile)
    if args.retarget_policy != "direct" and gmr_path.exists():
        return base.GmrUnitreeG1MotionSampler(
            gmr_path,
            args.gmr_fps,
            args.pose_gain,
            args.accent_gain,
            False,
        )
    if args.retarget_policy == "require-gmr":
        source_path = motion_path(catalog, profile)
        raise FileNotFoundError(
            f"Required GMR artifact not found: {gmr_path}\n"
            f"Generate it with:\n{base.gmr_generation_hint(source_path)}"
        )
    return base.AistppMotionSampler(
        motion_path(catalog, profile),
        args.aistpp_fps,
        args.pose_gain,
        args.accent_gain,
    )


def make_extractor(args: argparse.Namespace, catalog: MusicCatalog) -> AudioFeatureExtractor:
    metadata = catalog.metadata["extractor"]
    embedding_path = args.embedding_model
    tag_path = args.tag_model
    if embedding_path is None and metadata.get("embedding_model"):
        embedding_path = Path(metadata["embedding_model"]["path"])
    if tag_path is None and metadata.get("tag_model"):
        tag_path = Path(metadata["tag_model"]["path"])
    if embedding_path is not None and not embedding_path.is_absolute():
        embedding_path = catalog.catalog_path.parent / embedding_path
    if tag_path is not None and not tag_path.is_absolute():
        tag_path = catalog.catalog_path.parent / tag_path
    extractor = AudioFeatureExtractor(
        sample_rate=int(metadata["sample_rate"]),
        embedding_model=embedding_path,
        tag_model=tag_path,
        onnx_intra_op_threads=1,
    )
    expected = str(metadata["embedding_backend"])
    if extractor.backend_name != expected:
        raise RuntimeError(
            f"Catalog uses {expected}, but realtime extractor resolved to "
            f"{extractor.backend_name}. Supply the catalog's ONNX model."
        )
    return extractor


def make_controller(
    args: argparse.Namespace,
    profile: MotionProfile,
    previous: base.AdaptiveMotionController | None = None,
    entry_phase: float | None = None,
) -> base.AdaptiveMotionController:
    keypoints = profile.keypoint_phases
    if len(keypoints) < 2:
        keypoints = base.DEFAULT_FALLBACK_PHASES
    controller = base.AdaptiveMotionController(
        authored_cycle_duration=max(profile.duration_seconds, 1e-6),
        beats_per_cycle=max(len(keypoints), 1),
        keypoint_phases=keypoints,
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
        max_speed_change_per_sec=getattr(args, "max_speed_change_per_sec", 2.0),
        sync_to_beats=getattr(args, "motion_timing", "beat-sync") == "beat-sync",
    )
    if previous is not None:
        if entry_phase is None:
            entry_index = (
                int(np.argmax(profile.keypoint_scores))
                if profile.keypoint_scores
                else 0
            )
            entry_phase = float(keypoints[entry_index % len(keypoints)])
        else:
            entry_index = int(
                np.argmin(
                    np.abs(
                        np.asarray(keypoints, dtype=np.float64)
                        - float(entry_phase)
                    )
                )
            )
        controller.phase = float(np.clip(entry_phase, 0.0, np.nextafter(1.0, 0.0)))
        controller.beat_index = entry_index + 1
        controller.amplitude_scale = previous.amplitude_scale
        controller.target_amplitude_scale = previous.target_amplitude_scale
        controller.last_period = previous.last_period
        controller.last_beat_wall = previous.last_beat_wall
        controller.last_candidate_beat_wall = previous.last_candidate_beat_wall
        controller.last_accepted_interval = previous.last_accepted_interval
        controller.candidate_scores.extend(previous.candidate_scores)
        controller.last_brightness = previous.last_brightness
        controller.music_active = previous.music_active
        controller.last_update_wall = previous.last_update_wall
        if controller.sync_to_beats:
            inherited_rate = controller.authored_phase_rate * previous.speed_multiplier
            controller.target_phase_rate = float(
                np.clip(
                    inherited_rate,
                    controller.authored_phase_rate * controller.speed_min,
                    controller.authored_phase_rate * controller.speed_max,
                )
            )
            controller.phase_rate = controller.target_phase_rate
            controller.effective_phase_rate = controller.target_phase_rate
        else:
            controller.target_phase_rate = controller.authored_phase_rate
            controller.phase_rate = controller.authored_phase_rate
            controller.effective_phase_rate = controller.authored_phase_rate
            controller.phase_correction_remaining = 0.0
    return controller


def sample_actuator_pose(
    sampler: MotionSampler,
    controller: base.AdaptiveMotionController,
    now: float,
    features: base.FeatureState,
    modulator: MusicPoseModulator | None,
    adapter: Any,
    phase_tracker: OneShotPhaseTracker | None = None,
    stage_seconds: dict[str, list[float]] | None = None,
) -> tuple[RobotMotionFrame, float]:
    stage_started = time.perf_counter()
    phase, amplitude, accent, _brightness = controller.update(now)
    if phase_tracker is not None:
        phase, _ended = phase_tracker.clamp(phase)
        frame_count = len(getattr(sampler, "frames", ()))
        if frame_count > 1:
            # The shared samplers intentionally interpolate frame N-1 back to
            # frame 0 for preview looping. Matcher one-shot playback maps its
            # full phase range onto frame 0..N-1, never entering that seam.
            sampling_phase = phase * (frame_count - 1) / frame_count
        else:
            sampling_phase = 0.0
    else:
        sampling_phase = phase
    if stage_seconds is not None:
        stage_seconds.setdefault("phase_controller", []).append(
            max(time.perf_counter() - stage_started, 0.0)
        )
    stage_started = time.perf_counter()
    frame = base.sample_robot_motion_frame(
        sampler,
        phase=sampling_phase,
        amplitude=amplitude,
        accent=accent,
        features=features,
        pose_adapter=adapter,
        modulator=modulator,
    )
    if stage_seconds is not None:
        stage_seconds.setdefault("pose_sampling", []).append(
            max(time.perf_counter() - stage_started, 0.0)
        )
    return frame, phase


def control_stage_timing_summary(
    stage_seconds: dict[str, list[float]],
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name, values in sorted(stage_seconds.items()):
        samples = np.asarray(values, dtype=np.float64) * 1_000.0
        result[f"control_{name}_samples"] = int(samples.size)
        for temperature, subset in (
            ("", samples),
            ("_cold", samples[:1]),
            ("_warm", samples[1:]),
        ):
            for label, quantile in (("p50", 50.0), ("p95", 95.0), ("p99", 99.0)):
                result[f"control_{name}_ms{temperature}_{label}"] = (
                    float(np.percentile(subset, quantile)) if subset.size else 0.0
                )
            result[f"control_{name}_ms{temperature}_max"] = (
                float(np.max(subset)) if subset.size else 0.0
            )
    return result


def blend_poses(
    first: dict[str, float],
    second: dict[str, float],
    amount: float,
) -> dict[str, float]:
    t = float(np.clip(amount, 0.0, 1.0))
    smooth = t * t * (3.0 - 2.0 * t)
    names = set(first) | set(second)
    return {
        name: float((1.0 - smooth) * first.get(name, 0.0) + smooth * second.get(name, 0.0))
        for name in names
    }


def initial_motion_candidates(
    catalog: MusicCatalog,
    *,
    minimum_duration_seconds: float,
    low_activity_quantile: float,
) -> tuple[str, ...]:
    eligible = [
        profile
        for profile in catalog.motions.values()
        if profile.preflight_passed
        and profile.duration_seconds + 1e-9 >= minimum_duration_seconds
    ]
    if not eligible:
        eligible = [
            profile for profile in catalog.motions.values() if profile.preflight_passed
        ]
    if not eligible:
        raise ValueError("The catalog contains no preflight-passed startup motions.")

    quantile = float(np.clip(low_activity_quantile, 1e-9, 1.0))
    pool_size = max(1, int(math.ceil(len(eligible) * quantile)))
    ranked = sorted(
        eligible,
        key=lambda profile: (profile.velocity_p90, profile.motion_id),
    )
    return tuple(profile.motion_id for profile in ranked[:pool_size])


def select_initial_motion_id(
    args: argparse.Namespace,
    catalog: MusicCatalog,
) -> str:
    requested = getattr(args, "initial_motion_id", None)
    if requested:
        requested_id = Path(str(requested)).stem
        if requested_id not in catalog.motions:
            raise ValueError(f"Unknown --initial-motion-id: {requested_id}")
        if not catalog.motions[requested_id].preflight_passed:
            raise ValueError(
                f"Initial motion {requested_id} failed catalog preflight: "
                f"{catalog.motions[requested_id].preflight_reason}"
            )
        return requested_id

    wall_clock_budget = (
        max(float(args.analysis_min_seconds), 0.0)
        + max(float(args.match_interval_seconds), 0.0)
        * max(int(args.switch_required_wins), 1)
        + max(float(getattr(args, "startup_ready_reserve_seconds", 0.0)), 0.0)
    )
    minimum_duration = wall_clock_budget * max(
        float(getattr(args, "speed_max", 1.0)),
        1.0,
    )
    candidates = initial_motion_candidates(
        catalog,
        minimum_duration_seconds=minimum_duration,
        low_activity_quantile=args.initial_motion_low_activity_quantile,
    )
    return random.Random(args.initial_motion_seed).choice(candidates)


def initial_motion(
    args: argparse.Namespace,
    catalog: MusicCatalog,
) -> tuple[str, MotionProfile, MotionSampler]:
    requested = select_initial_motion_id(args, catalog)
    profile = catalog.motions[requested]
    sampler = load_motion_sampler(args, catalog, profile)
    return requested, profile, sampler


def print_match_status(result: MatchResult) -> None:
    tracks = ", ".join(
        f"{track.music_id}:{track.score:.2f}" for track in result.tracks[:3]
    )
    motion = result.motions[0] if result.motions else None
    motion_text = (
        f"{motion.motion_id}:{motion.final_score:.2f}"
        if motion is not None
        else "--"
    )
    tags = ", ".join(label for label, _ in result.query_tags[:3])
    tag_text = f" | tags=[{tags}]" if tags else ""
    genre = result.genres[0] if result.genres else None
    genre_text = f"{genre.genre}:{genre.score:.2f}" if genre is not None else "--"
    state = "accepted" if result.accepted else f"held:{result.rejection_reason}"
    print(
        f"match | {state} | confidence={result.confidence:.2f} | "
        f"genre={genre_text} | bpm={result.query_bpm:.1f} | tracks=[{tracks}] | "
        f"motion={motion_text}{tag_text}"
    )


def main() -> int:
    process_started = time.perf_counter()
    startup_timings_ms: dict[str, float] = {}
    args = parse_args()
    if args.transition_backend == "ruckig":
        require_ruckig()
    if not math.isfinite(args.transition_min_seconds) or args.transition_min_seconds <= 0.0:
        raise ValueError("--transition-min-seconds must be finite and positive.")
    if args.switch_ready_pool_size <= 0:
        raise ValueError("--switch-ready-pool-size must be positive.")
    if args.motion_cache_size <= 0:
        raise ValueError("--motion-cache-size must be positive.")
    if (
        not math.isfinite(args.python_thread_switch_interval_ms)
        or args.python_thread_switch_interval_ms <= 0.0
    ):
        raise ValueError("--python-thread-switch-interval-ms must be finite and positive.")
    if args.realtime:
        sys.setswitchinterval(args.python_thread_switch_interval_ms / 1_000.0)
        # THREAD_PRIORITY_ABOVE_NORMAL; only the latency-critical main thread is
        # raised, while process workers are explicitly lowered on creation.
        _set_windows_thread_priority(1)
    if (
        not math.isfinite(args.startup_ready_reserve_seconds)
        or args.startup_ready_reserve_seconds < 0.0
    ):
        raise ValueError("--startup-ready-reserve-seconds must be finite and non-negative.")
    if (
        not math.isfinite(args.terminal_safe_idle_amplitude)
        or args.terminal_safe_idle_amplitude < 0.0
    ):
        raise ValueError("--terminal-safe-idle-amplitude must be finite and non-negative.")
    if (
        not math.isfinite(args.match_weak_music_threshold)
        or args.match_weak_music_threshold < 0.0
    ):
        raise ValueError("--match-weak-music-threshold must be finite and non-negative.")
    if args.match_weak_music_consecutive_windows <= 0:
        raise ValueError("--match-weak-music-consecutive-windows must be positive.")
    if not 0.0 < args.initial_motion_low_activity_quantile <= 1.0:
        raise ValueError(
            "--initial-motion-low-activity-quantile must be greater than 0 and at most 1."
        )
    if args.diagnostic_disable_output_limiter and not args.headless:
        raise ValueError(
            "--diagnostic-disable-output-limiter is restricted to --headless runs."
        )
    if args.experiment_causal_file_input and (
        args.audio_input is None or not args.headless or args.experiment_pose_npz is None
    ):
        raise ValueError("Causal experiment mode requires --audio-input, --headless and --experiment-pose-npz.")
    if (
        args.diagnostic_disable_motion_compatibility
        or args.diagnostic_fixed_entry_transition
        or args.diagnostic_instant_top1
    ) and not args.headless:
        raise ValueError("Matcher ablation flags are restricted to --headless runs.")
    if args.motion_source != "aistpp":
        raise ValueError("The matcher currently selects AIST++ motions; use --motion-source aistpp.")
    catalog_path = args.catalog if args.catalog.is_absolute() else ROOT / args.catalog
    stage_started = time.perf_counter()
    catalog = MusicCatalog.load(catalog_path)
    startup_timings_ms["startup_catalog_load_ms"] = (
        time.perf_counter() - stage_started
    ) * 1_000.0
    stage_started = time.perf_counter()
    extractor = make_extractor(args, catalog)
    startup_timings_ms["startup_onnx_session_ms"] = (
        time.perf_counter() - stage_started
    ) * 1_000.0
    stage_started = time.perf_counter()
    matcher = MusicMotionMatcher(
        catalog,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        style_first=args.match_policy == "style-first",
        weak_music_threshold=args.match_weak_music_threshold,
        weak_music_consecutive_windows=args.match_weak_music_consecutive_windows,
        motion_compatibility=not args.diagnostic_disable_motion_compatibility,
    )
    startup_timings_ms["startup_matcher_init_ms"] = (
        time.perf_counter() - stage_started
    ) * 1_000.0
    stage_started = time.perf_counter()
    current_id, current_profile, current_sampler = initial_motion(args, catalog)
    current_controller = make_controller(args, current_profile)
    startup_timings_ms["startup_initial_motion_load_ms"] = (
        time.perf_counter() - stage_started
    ) * 1_000.0
    print(
        f"Initial retarget source: "
        f"{'GMR artifact' if isinstance(current_sampler, base.GmrUnitreeG1MotionSampler) else 'direct AIST++ fallback'}"
    )

    stage_started = time.perf_counter()
    player = base.MujocoHumanoidPlayer(
        args.model,
        realtime=args.realtime,
        headless=args.headless,
        viewer_rate_hz=args.viewer_rate_hz,
    )
    startup_timings_ms["startup_mujoco_init_ms"] = (
        time.perf_counter() - stage_started
    ) * 1_000.0
    adapter = base.make_pose_adapter(args, player, current_sampler)
    stage_started = time.perf_counter()
    player.ground_sampler(current_sampler, adapter)
    startup_timings_ms["startup_initial_grounding_ms"] = (
        time.perf_counter() - stage_started
    ) * 1_000.0
    stage_started = time.perf_counter()
    setattr(
        current_sampler,
        "entry_features",
        build_motion_entry_features(
            current_sampler,
            current_profile,
            adapter,
            player.actuator_joint_ranges,
        ),
    )
    startup_timings_ms["startup_initial_entry_features_ms"] = (
        time.perf_counter() - stage_started
    ) * 1_000.0
    limits_path = args.output_joint_limits_json
    if limits_path is not None and not limits_path.is_absolute():
        limits_path = ROOT / limits_path
    joint_limits = load_joint_dynamics_limits(
        limits_path,
        player.actuator_names,
        default_speed=args.output_max_joint_speed,
        default_acceleration=args.output_max_joint_acceleration,
    )
    jerk_limits = load_jerk_limits(limits_path, player.actuator_names, args.output_max_joint_jerk)
    if args.transition_backend == "hermite":
        warmup_hermite()
    if args.transition_backend == "ruckig":
        if any(not math.isfinite(value) for value in jerk_limits.values()):
            raise ValueError("Ruckig requires --output-max-joint-jerk or JSON jerk limits for every joint.")
    dynamics_limiter = JointDynamicsLimiter(joint_limits)
    modulation_off = args.disable_music_modulation or args.pose_modulation_mode == "off"
    modulator = (
        None
        if modulation_off
        else MusicPoseModulator(args.music_modulation_strength, mode=args.pose_modulation_mode)
    )
    if args.preview_trajectory:
        base.run_trajectory_preview(args, current_sampler, player, adapter, modulator)
        return 0

    if args.audio_input is not None:
        audio_path = args.audio_input if args.audio_input.is_absolute() else ROOT / args.audio_input
        source: MicrophoneSource | SilentSource | MatcherFileMicrophoneSource = MatcherFileMicrophoneSource(
            audio_path,
            args,
            args.match_window_seconds,
        )
        print(
            f"Using realtime virtual microphone: {audio_path} "
            f"(audio starts after {source.startup_delay_sec:.2f}s denoiser reset time; "
            f"speaker output={'on' if source.play_audio else 'off'})."
        )
    else:
        if args.no_mic:
            source = SilentSource()
            print("No microphone input: smoothly replaying the current motion at each end.")
        else:
            source = MicrophoneSource(
                base.make_analyzer(args),
                args.match_window_seconds,
                args.analysis_min_seconds,
            )
            print(
                f"Calibrating microphone noise from the first "
                f"{args.startup_calibration_sec:.2f}s of live input."
            )

    retrieval = RetrievalWorker(
        extractor,
        matcher,
        args.match_top_tracks,
        args.match_top_motions,
        process_config=(
            RetrievalProcessConfig(
                catalog_path=catalog.catalog_path.resolve(),
                sample_rate=extractor.sample_rate,
                embedding_model=(
                    extractor.embedding_backend.model_path
                    if extractor.embedding_backend is not None
                    else None
                ),
                tag_model=(
                    extractor.tag_backend.model_path
                    if extractor.tag_backend is not None
                    else None
                ),
                speed_min=matcher.speed_min,
                speed_max=matcher.speed_max,
                style_first=matcher.style_first,
                weak_music_threshold=matcher.weak_music_threshold,
                weak_music_consecutive_windows=matcher.weak_music_consecutive_windows,
                motion_compatibility=matcher.motion_compatibility,
            )
            if args.realtime
            else None
        ),
    )
    loader = MotionLoader(catalog, args, adapter)
    # Do not seed the pool from catalog insertion order: those motions are not
    # evidence-backed and can otherwise become arbitrary startup fallbacks.
    bootstrap_ids: tuple[str, ...] = ()
    selection_policy = MotionSelectionPolicy(
        current_motion_id=current_id,
        required_wins=args.switch_required_wins,
        score_margin=args.switch_score_margin,
        max_hold_bars=args.switch_max_hold_bars,
        diversity_top_k=args.switch_diversity_top_k,
        diversity_score_drop=args.switch_diversity_score_drop,
        diversity_music_score_drop=args.switch_diversity_music_score_drop,
        recent_history=args.switch_recent_history,
        motion_clusters={
            motion_id: profile.motion_cluster_id
            for motion_id, profile in catalog.motions.items()
        },
    )
    features = base.FeatureState()
    terminal_frame = None
    terminal_time = None
    terminal_recheck = None
    terminal_ready = ()
    terminal_source_features = None
    background_score_future = None
    background_score_key = None
    pending_plan = None
    bridge_source = None
    bridge_target = None
    bridge_target_raw = None
    bridge_features = None
    entry_features_override = None
    terminal_recheck_key = None
    audio_history_seconds = 0.0
    scored_history_seconds = 0.0
    shortlist_relevance = {}
    source_generation = 0
    last_terminal_time = None
    transition_sampler: MotionSampler | None = None
    transition_controller: base.AdaptiveMotionController | None = None
    current_phase_tracker = OneShotPhaseTracker(current_controller.phase)
    transition_phase_tracker: OneShotPhaseTracker | None = None
    transition_selection: MotionSelection | None = None
    transition_start: float | None = None
    transition_source_start_phase: float | None = None
    transition_forced_end = False
    active_transition_reason = ""
    transition_entry_score: MotionEntryScore | None = None
    transition_duration = 0.5
    transition_root_source_reference: RobotMotionFrame | None = None
    transition_root_target_reference: RobotMotionFrame | None = None
    current_root_source_reference: RobotMotionFrame | None = None
    current_root_target_reference: RobotMotionFrame | None = None
    candidate_motion_ids: tuple[str, ...] = bootstrap_ids
    forced_fallback_count = 0
    hold_last_count = 0
    collision_override_count = 0
    joint_limit_violations = 0
    control_stage_seconds: dict[str, list[float]] = {}
    holding_last = False
    terminal_safe_idle_started: float | None = None
    terminal_safe_idle_active = False
    forced_transition_plan: tuple[str, MotionSampler, MotionEntryScore, bool] | None = None
    bar_transition_plan: tuple[str, MotionSampler, MotionEntryScore, bool] | None = None
    bar_score_future: Future[
        tuple[str, MotionSampler, MotionEntryScore, bool] | None
    ] | None = None
    bar_score_generation = 0
    submitted_bar_score_generation = -1
    accepted_beat_count = 0
    accepted_beat_times_seconds: list[float] = []
    pending_beat_diagnostics: list[PendingBeatDiagnostic] = []
    beat_pose_errors_rad: list[float] = []
    beat_delays_seconds: list[float] = []
    limiter_impacted_beats = 0
    latest_accepted_beat_time: float | None = None
    latest_beat_pose_error = 0.0
    latest_beat_delay_seconds: float | None = None
    latest_beat_activated_joints = ""
    completed_root_reference: RobotMotionFrame | None = None
    latest_post_switch_xy_delta = 0.0
    latest_post_switch_z_delta = 0.0
    latest_post_switch_yaw_delta = 0.0
    previous_raw_wrist_positions: dict[str, float] = {}
    previous_output_wrist_positions: dict[str, float] = {}
    max_raw_wrist_abs_rad = 0.0
    max_raw_wrist_speed_rad_s = 0.0
    max_output_wrist_abs_rad = 0.0
    max_output_wrist_speed_rad_s = 0.0
    last_match_clock = -math.inf
    last_feature_update = time.perf_counter()
    last_status = time.perf_counter()
    last_frame: base.MusicFrame | None = None
    recent_detected_beat_frame: base.MusicFrame | None = None
    recent_accepted_beat_frame: base.MusicFrame | None = None
    last_match_result: MatchResult | None = None
    trace_handle = None
    trace_writer: csv.DictWriter | None = None
    trace_next_audio_time = 0.0
    trace_interval = max(float(args.trace_interval_seconds), 0.001)
    if args.trace_csv is not None:
        trace_path = args.trace_csv if args.trace_csv.is_absolute() else ROOT / args.trace_csv
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = trace_path.open("w", encoding="utf-8", newline="")
        trace_writer = csv.DictWriter(
            trace_handle,
            fieldnames=(
                "audio_time_seconds",
                "wall_time_seconds",
                "event",
                "transition_backend", "bridge_preparation_status", "bridge_failure_reason",
                "bridge_position_residual", "bridge_velocity_residual", "bridge_acceleration_residual",
                "bridge_output_modified", "boundary_effect_weight", "bridge_generation",
                "current_motion_id",
                "pending_motion_id",
                "preferred_motion_id",
                "audio_history_seconds", "shortlist", "terminal_time_seconds",
                "entry_frame_index", "preparation_delay_seconds",
                "exit_frame_index", "exit_seconds", "exit_phase", "trimmed_seconds",
                "transition_pair_count", "bridge_attempts", "transition_preparation_seconds",
                "transition_motion_id",
                "fallback_reason",
                "current_phase",
                "transition_phase",
                "transition_blend",
                "transition_reason",
                "entry_phase",
                "entry_total_score",
                "entry_pose_score",
                "entry_velocity_score",
                "entry_contact_score",
                "entry_root_score",
                "entry_music_score",
                "entry_remaining_seconds",
                "transition_duration_seconds",
                "motion_timing",
                "speed_multiplier",
                "target_speed_multiplier",
                "transition_speed_multiplier",
                "phase_correction_remaining",
                "raw_max_wrist_abs_rad",
                "raw_max_wrist_angle_joint",
                "raw_max_wrist_speed_rad_s",
                "raw_max_wrist_speed_joint",
                "output_max_wrist_abs_rad",
                "output_max_wrist_angle_joint",
                "output_max_wrist_speed_rad_s",
                "output_max_wrist_speed_joint",
                "root_x",
                "root_y",
                "root_z",
                "root_yaw",
                "switch_anchor_xy_delta_m",
                "switch_anchor_z_delta_m",
                "switch_anchor_yaw_delta_rad",
                "post_switch_xy_delta_m",
                "post_switch_z_delta_m",
                "post_switch_yaw_delta_rad",
                "accepted_beat_time_seconds",
                "accepted_beat_pose_error_rad",
                "accepted_beat_delay_seconds",
                "accepted_beat_limiter_joints",
                "target_max_joint_speed_rad_s",
                "output_max_joint_speed_rad_s",
                "output_max_joint_acceleration_rad_s2",
                "limiter_activated_joints",
                "limiter_activation_rate",
                "collision_override",
                "limiter_enabled",
                "terminal_safe_idle_active",
                "ready_pool_size",
                "ready_motion_ids",
                "failed_motion_ids",
                "motion_load_ms",
                "motion_grounding_ms",
                "entry_feature_ms",
                "motion_fully_ready_ms",
                "match_state",
                "match_confidence",
                "match_rejection_reason",
                "beat_quality",
                "signal_rms",
                "dynamic_range_db",
                "negative_style_score",
                "top_genre",
                "top_genre_score",
                "track_margin",
                "motion_margin",
                "ranked_tracks",
                "ranked_motions",
                "query_tags",
                "current_motion_cluster_id",
                "pending_motion_cluster_id",
                "top_track_id",
                "top_track_score",
                "top_motion_id",
                "top_motion_score",
                "query_bpm",
            ),
        )
        trace_writer.writeheader()

    pose_times: list[float] = []
    causal_experiment = getattr(args, "experiment_causal_file_input", False)
    pose_wall_times: list[float] = []
    pose_qpos: list[np.ndarray] = []
    pose_safety: list[list[int]] = []
    beat_events: list[list[float]] = []
    final_residual_count = 0
    final_limit_count = 0
    pose_joint_positions: list[list[float]] = []
    pose_body_positions: list[np.ndarray] = []
    pose_center_of_mass: list[np.ndarray] = []
    pose_left_foot: list[np.ndarray] = []
    pose_right_foot: list[np.ndarray] = []
    pose_left_support_height: list[float] = []
    pose_right_support_height: list[float] = []
    pose_motion_ids: list[str] = []
    pose_events: list[str] = []
    experiment_joint_names = tuple(player.actuator_names)
    experiment_qpos_ids = [player.actuator_joint_qpos_ids[name] for name in experiment_joint_names]
    experiment_joint_ranges = [player.actuator_joint_ranges.get(name, (-np.inf, np.inf))
                               for name in experiment_joint_names]
    experiment_body_names = tuple(
        base.mujoco.mj_id2name(player.model, base.mujoco.mjtObj.mjOBJ_BODY, index)
        or f"body_{index}"
        for index in range(player.model.nbody)
    )

    source_analyzer = getattr(source, "analyzer", None)
    if source_analyzer is not None:
        # Windows process spawning must happen before the viewer owns an
        # OpenGL context; source.start() remains responsible for the stream.
        source_analyzer.prepare_background_analysis()
    live_microphone_started = isinstance(source, MicrophoneSource)
    if live_microphone_started:
        # Some Windows PortAudio backends can block when opened after an
        # OpenGL viewer.  Open hardware first, then restart calibration once
        # the viewer is ready.
        source.start()
    player.start()
    if live_microphone_started:
        source_analyzer.reset()
    initial_root = player.root_frame()
    root_motion = RootMotionContinuity(
        initial_root.root_position if initial_root.root_position is not None else np.zeros(3),
        initial_root.root_quaternion_wxyz
        if initial_root.root_quaternion_wxyz is not None
        else np.asarray([1.0, 0.0, 0.0, 0.0]),
        mode=args.root_motion,
        grounded_z=True,
    )
    if not live_microphone_started:
        source.start()
    startup_timings_ms["startup_until_control_loop_ms"] = (
        time.perf_counter() - process_started
    ) * 1_000.0
    wall_start = time.perf_counter()
    audio_wall_origin = (
        source.stream_start_wall + source.startup_delay_sec
        if isinstance(source, MatcherFileMicrophoneSource) else wall_start
    )
    scheduler = base.RealtimeLoopScheduler(args.control_rate_hz, args.realtime)
    timing_warmup_reset = args.experiment_warmup_seconds <= 0.0
    authored_playback = (AuthoredPlayback(current_id, current_sampler, current_controller,
        loader, catalog, args, joint_limits, player.actuator_joint_ranges, jerk_limits, wall_start)
        if args.transition_backend != "quintic" else None)
    print(f"Initial motion: {current_id}")
    try:
        while player.is_running() and not source.done:
            trace_event = ""
            trace_transition_blend = 0.0
            trace_anchor_xy_delta = 0.0
            trace_anchor_z_delta = 0.0
            trace_anchor_yaw_delta = 0.0
            accepted_times_this_iteration: list[float] = []
            work_started = time.perf_counter()
            audio_stage_started = work_started
            now = work_started
            source.advance(scheduler.period)
            elapsed = (
                source.playback_seconds
                if isinstance(source, MatcherFileMicrophoneSource)
                else now - wall_start
            )
            if args.max_seconds is not None and elapsed >= args.max_seconds:
                break
            evaluation_elapsed = now - audio_wall_origin if causal_experiment else elapsed
            if not timing_warmup_reset and evaluation_elapsed >= args.experiment_warmup_seconds:
                scheduler.reset_statistics()
                control_stage_seconds.clear()
                timing_warmup_reset = True

            switch_boundary = False
            ready_selection: MotionSelection | None = None
            for frame in source.drain():
                received_wall = time.perf_counter()
                last_frame = frame
                accepted = current_controller.observe(frame)
                if causal_experiment and frame.is_beat:
                    beat_events.append([frame.timestamp - audio_wall_origin,
                                        received_wall - audio_wall_origin,
                                        source.playback_seconds, float(accepted)])
                # Incoming controller is held fixed throughout interpolation.
                dt = max(now - last_feature_update, scheduler.period)
                alpha = 1.0 - math.exp(
                    -dt / max(args.feature_smoothing_tau, 1e-6)
                )
                features.update(frame, alpha)
                last_feature_update = now
                if frame.is_beat:
                    recent_detected_beat_frame = frame
                if accepted:
                    recent_accepted_beat_frame = frame
                    accepted_times_this_iteration.append(elapsed)
                    accepted_beat_times_seconds.append(elapsed)
                    accepted_beat_count += 1
                    is_switch_boundary = (
                        accepted_beat_count % max(args.switch_beats_per_bar, 1) == 0
                    )
                    if is_switch_boundary:
                        switch_boundary = True
                        pass  # V2 switches only after the terminal frame.
            control_stage_seconds.setdefault("audio_beat_analysis", []).append(
                max(time.perf_counter() - audio_stage_started, 0.0)
            )

            match_clock = source.playback_seconds
            if match_clock - last_match_clock >= args.match_interval_seconds:
                audio = source.recent_audio()
                if audio is not None and retrieval.submit(audio, source.sample_rate):
                    last_match_clock = match_clock
                    audio_history_seconds = len(audio) / source.sample_rate

            try:
                result = retrieval.poll()
            except Exception as exc:
                print(f"Warning: realtime retrieval failed: {type(exc).__name__}: {exc}")
                result = None
                last_match_result = None
                candidate_motion_ids = ()
            if result is not None:
                last_match_result = result
                scored_history_seconds = audio_history_seconds
                trace_event = "match"
                print_match_status(result)
                candidate_motion_ids = musical_shortlist(result, current_id, args)
                shortlist_relevance = {item.motion_id: (item.final_score, item.music_score)
                                       for item in result.motions}
                if authored_playback is None:
                    loader.preload(list(candidate_motion_ids))
                    loader.prepare_pool(list(candidate_motion_ids))

            if authored_playback is not None:
                transition_stage_started = time.perf_counter()
                authored_playback.prepare(last_match_result, now)
                motion_frame = authored_playback.sample(now, features, modulator, adapter)
                current_id = authored_playback.current_id
                current_sampler = authored_playback.sampler
                current_controller = authored_playback.controller
                current_profile = catalog.motions[current_id]
                current_phase = authored_playback.clock.seconds / authored_playback.clock.duration
                current_phase_tracker = OneShotPhaseTracker(current_phase)
                transition_sampler = authored_playback.active.sampler if authored_playback.active else None
                transition_controller = None
                transition_phase = authored_playback.score.phase if authored_playback.score else None
                transition_entry_score = authored_playback.score
                transition_duration = authored_playback.duration
                transition_start = authored_playback.bridge_start
                transition_selection = (MotionSelection(authored_playback.active.motion_id,
                    authored_playback.reason, 0., 0.) if authored_playback.active else None)
                active_transition_reason = authored_playback.reason
                trace_transition_blend = authored_playback.blend
                last_terminal_time = authored_playback.last_terminal
                if authored_playback.event:
                    trace_event = authored_playback.event
                    if trace_event == "switch_start":
                        forced_fallback_count += int(authored_playback.active.replay)
                        print(f"Switching ({active_transition_reason}, {args.transition_backend}): "
                              f"{current_id} -> {authored_playback.active.motion_id} "
                              f"entry={transition_phase:.4f} blend={transition_duration:.3f}s")
                    elif trace_event == "switch_complete":
                        selection_policy.complete_switch(current_id)
                        print(f"Motion switch complete: {current_id}")
                    elif trace_event == "bridge_hold":
                        hold_last_count += 1
                        print(f"Warning: terminal hold: {authored_playback.failure}")
                selection_policy.pending = (MotionSelection(authored_playback.plan.motion_id,
                    "prepared_motion_end", 0., 0.) if authored_playback.plan else None)
                candidate_motion_ids = tuple(mid for mid, _ in (authored_playback.frozen or ()))
                root_source_id = "v2_continuous"
            else:
                ready_snapshot = tuple(
                    (motion_id, ready, shortlist_relevance.get(motion_id, (0.0, 0.0))) for motion_id in candidate_motion_ids
                    if (ready := loader.take_ready(motion_id)) is not None
                )
                score_key = (source_generation, tuple((item[0], item[2]) for item in ready_snapshot))
                if background_score_future is not None and background_score_future.done():
                    try:
                        completed = background_score_future.result()
                        pending_plan = completed if background_score_key == score_key else None
                        selection_policy.pending = (
                            MotionSelection(pending_plan[0], "planned_motion_end", 0.0, 0.0)
                            if pending_plan is not None else None
                        )
                    except Exception as exc:
                        print(f"Warning: background entry scoring failed: {exc}")
                        pending_plan = None
                    background_score_future = None
                if (terminal_frame is None and transition_sampler is None
                        and background_score_future is None and background_score_key != score_key):
                    background_score_key = score_key
                    background_score_future = loader.score_executor.submit(
                        score_ready_candidates, current_sampler.entry_features,
                        ready_snapshot, joint_limits, args, loader.score_seconds,
                    )

                transition_stage_started = time.perf_counter()
                if terminal_frame is None and transition_sampler is None:
                    # Freeze feature values for the first sample after a bridge, so its
                    # target pose and the resumed clip agree exactly at the entry frame.
                    if entry_features_override is not None:
                        current_frame = bridge_target
                        current_phase = current_phase_tracker.last_phase
                        current_controller.last_update_wall = now
                        entry_features_override = None
                    else:
                        current_frame, current_phase = sample_actuator_pose(
                            current_sampler, current_controller, now,
                            features, modulator, adapter,
                            current_phase_tracker, control_stage_seconds,
                        )
                        if current_root_source_reference is not None:
                            current_frame = align_motion_frame_root(
                                current_frame, source_reference=current_root_source_reference,
                                target_reference=current_root_target_reference,
                            )
                    if current_phase_tracker.ended:
                        terminal_frame = current_frame
                        terminal_time = last_terminal_time = now
                        terminal_ready = ready_snapshot
                        terminal_source_features = None
                        terminal_recheck_key = None
                        trace_event = "terminal_frame"
                else:
                    current_frame = terminal_frame if terminal_frame is not None else bridge_source
                    current_phase = OneShotPhaseTracker.terminal_phase

                # Recheck with the final command captured after limiting, on a worker.
                # New music can still replace the shortlist until the bridge starts.
                recheck_key = tuple((item[0], item[2]) for item in ready_snapshot)
                if (terminal_source_features is not None and terminal_frame is not None
                        and transition_sampler is None and terminal_recheck_key != recheck_key):
                    if terminal_recheck is not None:
                        terminal_recheck.cancel()
                    terminal_recheck_key = recheck_key
                    terminal_recheck = loader.score_executor.submit(
                        score_ready_candidates, terminal_source_features,
                        ready_snapshot, joint_limits, args, loader.score_seconds,
                    ) if ready_snapshot else None

                # Emit the final frame for at least one control tick before a bridge.
                if (terminal_frame is not None and transition_sampler is None
                        and now > terminal_time and terminal_source_features is not None
                        and (terminal_recheck is None or terminal_recheck.done())):
                    choice = None
                    if terminal_recheck is not None:
                        try:
                            choice = terminal_recheck.result()
                        except Exception as exc:
                            print(f"Warning: terminal recheck failed; replaying current motion: {exc}")
                    if choice is None:
                        choice = replay_choice(current_id, current_sampler, terminal_source_features, joint_limits, args)
                    chosen_id, transition_sampler, entry_score, used_fallback = choice
                    active_transition_reason = "replay_no_suitable_ready_motion" if used_fallback else "motion_end"
                    transition_entry_score = entry_score
                    transition_selection = MotionSelection(chosen_id, active_transition_reason, 0.0, 0.0)
                    transition_controller = make_controller(
                        args, catalog.motions[chosen_id], previous=current_controller,
                        entry_phase=entry_score.phase,
                    )
                    transition_controller.last_update_wall = now
                    transition_controller.phase_correction_remaining = 0.0
                    transition_phase_tracker = OneShotPhaseTracker(entry_score.phase)
                    bridge_features = replace(features)
                    bridge_target_raw, _ = sample_actuator_pose(
                        transition_sampler, transition_controller, now, bridge_features,
                        modulator, adapter, transition_phase_tracker,
                    )
                    bridge_source = terminal_frame
                    bridge_target = align_motion_frame_root(
                        bridge_target_raw, source_reference=bridge_target_raw,
                        target_reference=bridge_source,
                    )
                    names = sorted(set(bridge_source.joint_positions) & set(bridge_target.joint_positions) & set(joint_limits))
                    transition_duration = compute_transition_duration(
                        np.array([bridge_source.joint_positions[name] for name in names]),
                        np.array([bridge_target.joint_positions[name] for name in names]),
                        np.array([joint_limits[name].max_speed_rad_s for name in names]),
                        np.array([joint_limits[name].max_acceleration_rad_s2 for name in names]),
                        beat_period=0.5, minimum=args.transition_min_seconds, maximum=math.inf,
                    )
                    transition_start = now
                    forced_fallback_count += int(used_fallback)
                    print(f"Switching ({active_transition_reason}): {current_id} -> {chosen_id} "
                          f"entry={entry_score.phase:.4f} frame={entry_score.frame_index} "
                          f"blend={transition_duration:.3f}s cost={entry_score.total:.4f}")
                    trace_event = "switch_start"

                transition_phase = None
                root_source_id = "v2_continuous"  # Already aligned across clips, including replay.
                if transition_sampler is not None:
                    transition_phase = transition_entry_score.phase
                    blend = (now - transition_start) / transition_duration
                    trace_transition_blend = float(np.clip(blend, 0.0, 1.0))
                    motion_frame = quintic_blend(bridge_source, bridge_target, blend)
                    if blend >= 1.0:
                        current_id = transition_selection.motion_id
                        current_profile = catalog.motions[current_id]
                        current_sampler = transition_sampler
                        current_controller = transition_controller
                        current_controller.last_update_wall = now
                        current_controller.phase = transition_entry_score.phase
                        current_controller.phase_correction_remaining = 0.0
                        current_phase_tracker = OneShotPhaseTracker(transition_entry_score.phase)
                        current_root_source_reference = bridge_target_raw
                        current_root_target_reference = bridge_source
                        entry_features_override = bridge_features
                        terminal_frame = None
                        terminal_time = None
                        terminal_recheck = None
                        terminal_source_features = None
                        transition_sampler = None
                        transition_controller = None
                        transition_selection = None
                        transition_start = None
                        selection_policy.complete_switch(current_id)
                        candidate_motion_ids = musical_shortlist(last_match_result, current_id, args)
                        loader.preload(list(candidate_motion_ids))
                        loader.prepare_pool(list(candidate_motion_ids))
                        source_generation += 1
                        pending_plan = None
                        trace_event = "switch_complete"
                        print(f"Motion switch complete: {current_id}")
                else:
                    motion_frame = current_frame

            motion_frame = root_motion.apply(
                motion_frame,
                phase=0.0,  # V2 aligns roots itself; clip phases must not trigger loop reanchoring.
                source_id=root_source_id,
            )
            if completed_root_reference is not None:
                if (
                    completed_root_reference.root_position is not None
                    and motion_frame.root_position is not None
                ):
                    post_delta = (
                        motion_frame.root_position
                        - completed_root_reference.root_position
                    )
                    latest_post_switch_xy_delta = float(
                        np.linalg.norm(post_delta[:2])
                    )
                    latest_post_switch_z_delta = float(abs(post_delta[2]))
                previous_yaw = frame_root_yaw(completed_root_reference)
                current_yaw = frame_root_yaw(motion_frame)
                if previous_yaw is not None and current_yaw is not None:
                    latest_post_switch_yaw_delta = abs(
                        wrapped_angle_difference(current_yaw, previous_yaw)
                    )
                completed_root_reference = None
                if not trace_event:
                    trace_event = "post_switch_continuity"
            if trace_event == "switch_complete":
                completed_root_reference = motion_frame

            control_stage_seconds.setdefault("transition_root_blend", []).append(
                max(time.perf_counter() - transition_stage_started, 0.0)
            )

            for accepted_time in accepted_times_this_iteration:
                pending_beat_diagnostics.append(
                    PendingBeatDiagnostic(
                        accepted_time_seconds=accepted_time,
                        target_pose=dict(motion_frame.joint_positions),
                        errors_rad=[],
                    )
                )

            raw_wrist_diagnostics, previous_raw_wrist_positions = (
                wrist_motion_diagnostics(
                    motion_frame,
                    previous_raw_wrist_positions,
                    scheduler.period,
                )
            )
            max_raw_wrist_abs_rad = max(
                max_raw_wrist_abs_rad,
                raw_wrist_diagnostics.max_abs_rad,
            )
            max_raw_wrist_speed_rad_s = max(
                max_raw_wrist_speed_rad_s,
                raw_wrist_diagnostics.max_speed_rad_s,
            )

            safety_stage_started = time.perf_counter()
            planned_joint_positions = dict(motion_frame.joint_positions)
            limiter_dt = scheduler.time_since_last_output(safety_stage_started)
            if args.diagnostic_disable_output_limiter:
                limited_frame = motion_frame
                dynamics_limiter.reset(motion_frame)
                minimum_dynamic_scale = 0.0
            else:
                limited_frame = dynamics_limiter.apply(
                    motion_frame,
                    limiter_dt,
                )
                minimum_dynamic_scale = dynamics_limiter.minimum_feasible_output_scale(
                    limited_frame, limiter_dt
                )

            resolved_beat = False
            for diagnostic in list(pending_beat_diagnostics):
                common_names = set(diagnostic.target_pose) & set(
                    limited_frame.joint_positions
                )
                error = max(
                    (
                        abs(
                            limited_frame.joint_positions[name]
                            - diagnostic.target_pose[name]
                        )
                        for name in common_names
                    ),
                    default=0.0,
                )
                diagnostic.errors_rad.append(float(error))
                if len(diagnostic.errors_rad) < 9:
                    continue
                errors = np.asarray(diagnostic.errors_rad, dtype=np.float64)
                best = float(np.min(errors))
                best_indices = np.flatnonzero(errors <= best + 1e-10)
                delay_cycles = int(best_indices[0]) if best_indices.size else 8
                beat_delays_seconds.append(delay_cycles * scheduler.period)
                latest_accepted_beat_time = diagnostic.accepted_time_seconds
                latest_beat_delay_seconds = delay_cycles * scheduler.period
                pending_beat_diagnostics.remove(diagnostic)
                resolved_beat = True
            if resolved_beat and not trace_event:
                trace_event = "beat_limiter_result"

            if accepted_times_this_iteration:
                accepted_error = max(
                    (
                        diagnostic.errors_rad[0]
                        for diagnostic in pending_beat_diagnostics
                        if diagnostic.accepted_time_seconds
                        in accepted_times_this_iteration
                    ),
                    default=0.0,
                )
                latest_accepted_beat_time = accepted_times_this_iteration[-1]
                latest_beat_pose_error = float(accepted_error)
                latest_beat_activated_joints = ";".join(
                    dynamics_limiter.last_activated
                )
                beat_pose_errors_rad.extend(
                    [float(accepted_error)] * len(accepted_times_this_iteration)
                )
                if dynamics_limiter.last_activated:
                    limiter_impacted_beats += len(accepted_times_this_iteration)
            detections_before = getattr(player, "collision_candidate_violation_count", 0)
            corrections_before = player.collision_projection_count
            motion_frame = player.apply_collision_policy(
                limited_frame,
                args.runtime_collision_check,
                runtime_modified=(
                    transition_sampler is not None
                    or (modulator is not None and features.is_active)
                ),
                minimum_dynamic_scale=minimum_dynamic_scale,
            )
            joint_limit_violations += sum(
                1
                for name, value in motion_frame.joint_positions.items()
                if (
                    (joint_range := player.actuator_joint_ranges.get(name)) is not None
                    and (
                        float(value) < joint_range[0] - 1e-8
                        or float(value) > joint_range[1] + 1e-8
                    )
                )
            )
            collision_override = any(
                abs(
                    motion_frame.joint_positions.get(name, 0.0)
                    - limited_frame.joint_positions.get(name, 0.0)
                )
                > 1e-10
                for name in set(motion_frame.joint_positions)
                | set(limited_frame.joint_positions)
            )
            if collision_override:
                collision_override_count += 1
            control_stage_seconds.setdefault("limiter_collision", []).append(
                max(time.perf_counter() - safety_stage_started, 0.0)
            )
            output_wrist_diagnostics, previous_output_wrist_positions = (
                wrist_motion_diagnostics(
                    motion_frame,
                    previous_output_wrist_positions,
                    limiter_dt,
                )
            )
            max_output_wrist_abs_rad = max(
                max_output_wrist_abs_rad,
                output_wrist_diagnostics.max_abs_rad,
            )
            max_output_wrist_speed_rad_s = max(
                max_output_wrist_speed_rad_s,
                output_wrist_diagnostics.max_speed_rad_s,
            )
            mujoco_stage_started = time.perf_counter()
            player.set_frame(motion_frame)
            bridge_output_modified = any(
                abs(float(player.data.qpos[player.actuator_joint_qpos_ids[name]]) - value) > 1e-8
                for name, value in planned_joint_positions.items()
                if name in player.actuator_joint_qpos_ids
            )
            if trace_event == "terminal_frame":
                # Keep the local root reference, but use the final joint command
                # after the output limiter and collision policy have run.
                terminal_frame = replace(terminal_frame, joint_positions=dict(motion_frame.joint_positions))
                terminal_source_features = terminal_features(current_sampler.entry_features, terminal_frame)
            emitted_at = time.perf_counter()
            actual_output_dt = scheduler.record_output(emitted_at)
            if not args.diagnostic_disable_output_limiter:
                dynamics_limiter.sync_output(motion_frame, actual_output_dt)
            player.step(actual_output_dt)
            control_stage_seconds.setdefault("mujoco_forward", []).append(
                max(time.perf_counter() - mujoco_stage_started, 0.0)
            )
            if args.experiment_pose_npz is not None:
                pose_times.append(float(elapsed))
                pose_joint_positions.append(
                    [float(player.data.qpos[index]) for index in experiment_qpos_ids]
                    if causal_experiment else
                    [float(motion_frame.joint_positions.get(name, 0.0)) for name in experiment_joint_names]
                )
                if causal_experiment:
                    # Snapshot the final displayed qpos, after clipping and collision policy.
                    pose_wall_times.append(emitted_at - audio_wall_origin)
                    pose_qpos.append(player.data.qpos.copy())
                    audit_started = time.perf_counter()
                    residual = int(player._has_self_clearance_violation_after_forward(
                        player.data, base.DEFAULT_COLLISION_MIN_DISTANCE_M
                        + base.COLLISION_CLEARANCE_BUFFER_M))
                    limits = sum(int(value < low - 1e-8 or value > high + 1e-8)
                                 for value, (low, high) in zip(pose_joint_positions[-1], experiment_joint_ranges))
                    final_residual_count += residual
                    final_limit_count += limits
                    pose_safety.append([
                        player.collision_candidate_violation_count - detections_before,
                        player.collision_projection_count - corrections_before, residual, limits])
                    control_stage_seconds.setdefault("experiment_final_safety_audit", []).append(
                        time.perf_counter() - audit_started)
                pose_body_positions.append(np.asarray(player.data.xpos, dtype=np.float64).copy())
                pose_center_of_mass.append(
                    np.asarray(player.data.subtree_com[0], dtype=np.float64).copy()
                )
                pose_left_foot.append(
                    np.asarray(
                        player.data.xpos[player.left_foot_support_body_id],
                        dtype=np.float64,
                    ).copy()
                )
                pose_right_foot.append(
                    np.asarray(
                        player.data.xpos[player.right_foot_support_body_id],
                        dtype=np.float64,
                    ).copy()
                )
                pose_left_support_height.append(
                    player.foot_support_height(player.left_foot_support_geom_ids, player.data)
                )
                pose_right_support_height.append(
                    player.foot_support_height(player.right_foot_support_geom_ids, player.data)
                )
                pose_motion_ids.append(current_id)
                pose_events.append(trace_event)

            if trace_writer is not None and (
                trace_event or elapsed + 1e-9 >= trace_next_audio_time
            ):
                top_track = (
                    last_match_result.tracks[0]
                    if last_match_result is not None and last_match_result.tracks
                    else None
                )
                top_motion = (
                    last_match_result.motions[0]
                    if last_match_result is not None and last_match_result.motions
                    else None
                )
                pending_selection = selection_policy.pending
                reported_current_phase = (
                    transition_phase
                    if authored_playback is None and trace_event == "switch_complete"
                    and transition_phase is not None
                    else current_phase
                )
                trace_writer.writerow(
                    {
                        "audio_time_seconds": f"{elapsed:.6f}",
                        "wall_time_seconds": f"{time.perf_counter() - audio_wall_origin:.9f}",
                        "event": trace_event,
                        "transition_backend": args.transition_backend,
                        "bridge_preparation_status": authored_playback.status if authored_playback else "legacy",
                        "bridge_failure_reason": authored_playback.failure if authored_playback else "",
                        "bridge_position_residual": authored_playback.last_residuals[0] if authored_playback and authored_playback.last_residuals is not None else "",
                        "bridge_velocity_residual": authored_playback.last_residuals[1] if authored_playback and authored_playback.last_residuals is not None else "",
                        "bridge_acceleration_residual": authored_playback.last_residuals[2] if authored_playback and authored_playback.last_residuals is not None else "",
                        "bridge_output_modified": int(bridge_output_modified),
                        "boundary_effect_weight": authored_playback.clock.weight if authored_playback and not authored_playback.active else 0.,
                        "bridge_generation": authored_playback.generation if authored_playback else "",

                        "current_motion_id": current_id,
                        "pending_motion_id": (
                            pending_selection.motion_id if pending_selection is not None else ""
                        ),
                        "preferred_motion_id": (
                            pending_selection.motion_id
                            if pending_selection is not None
                            else ""
                        ),
                        "audio_history_seconds": f"{scored_history_seconds:.6f}",
                        "shortlist": ";".join(candidate_motion_ids),
                        "terminal_time_seconds": "" if last_terminal_time is None else f"{last_terminal_time - wall_start:.6f}",
                        "entry_frame_index": "" if transition_entry_score is None else transition_entry_score.frame_index,
                        **({
                            "exit_frame_index": authored_playback.last_plan.exit_frame_index,
                            "exit_seconds": authored_playback.last_plan.exit_seconds,
                            "exit_phase": authored_playback.last_plan.exit_seconds / authored_playback.last_plan.source_duration,
                            "trimmed_seconds": authored_playback.last_plan.source_duration-authored_playback.last_plan.exit_seconds,
                            "transition_pair_count": authored_playback.last_plan.pair_count,
                            "bridge_attempts": authored_playback.last_plan.bridge_attempts,
                            "transition_preparation_seconds": authored_playback.last_plan.preparation_seconds,
                        } if authored_playback is not None and authored_playback.last_plan is not None else {}),
                        "preparation_delay_seconds": "" if terminal_time is None else f"{max(0.0, (transition_start or now) - terminal_time):.6f}",
                        "transition_motion_id": (
                            transition_selection.motion_id
                            if transition_selection is not None
                            else ""
                        ),
                        "fallback_reason": (
                            active_transition_reason if active_transition_reason.startswith("replay_") else ""
                        ),
                        "current_phase": f"{reported_current_phase:.9f}",
                        "transition_phase": (
                            f"{transition_phase:.9f}"
                            if transition_phase is not None
                            else ""
                        ),
                        "transition_blend": f"{trace_transition_blend:.6f}",
                        "transition_reason": active_transition_reason,
                        "entry_phase": (
                            f"{transition_entry_score.phase:.9f}"
                            if transition_entry_score is not None
                            else ""
                        ),
                        "entry_total_score": (
                            f"{transition_entry_score.total:.6f}"
                            if transition_entry_score is not None
                            else ""
                        ),
                        "entry_pose_score": (
                            f"{transition_entry_score.pose:.6f}"
                            if transition_entry_score is not None
                            else ""
                        ),
                        "entry_velocity_score": (
                            f"{transition_entry_score.velocity:.6f}"
                            if transition_entry_score is not None
                            else ""
                        ),
                        "entry_contact_score": (
                            f"{transition_entry_score.contact:.6f}"
                            if transition_entry_score is not None
                            else ""
                        ),
                        "entry_root_score": (
                            f"{transition_entry_score.root:.6f}"
                            if transition_entry_score is not None
                            else ""
                        ),
                        "entry_music_score": (
                            f"{transition_entry_score.music:.6f}"
                            if transition_entry_score is not None
                            else ""
                        ),
                        "entry_remaining_seconds": (
                            f"{transition_entry_score.remaining_seconds:.6f}"
                            if transition_entry_score is not None
                            else ""
                        ),
                        "transition_duration_seconds": f"{transition_duration:.6f}",
                        "motion_timing": args.motion_timing,
                        "speed_multiplier": f"{current_controller.speed_multiplier:.6f}",
                        "target_speed_multiplier": (
                            f"{current_controller.target_phase_rate / max(current_controller.authored_phase_rate, 1e-9):.6f}"
                        ),
                        "transition_speed_multiplier": (
                            f"{transition_controller.speed_multiplier:.6f}"
                            if transition_controller is not None
                            else ""
                        ),
                        "phase_correction_remaining": (
                            f"{current_controller.phase_correction_remaining:.9f}"
                        ),
                        "raw_max_wrist_abs_rad": (
                            f"{raw_wrist_diagnostics.max_abs_rad:.9f}"
                        ),
                        "raw_max_wrist_angle_joint": (
                            raw_wrist_diagnostics.max_abs_joint
                        ),
                        "raw_max_wrist_speed_rad_s": (
                            f"{raw_wrist_diagnostics.max_speed_rad_s:.9f}"
                        ),
                        "raw_max_wrist_speed_joint": (
                            raw_wrist_diagnostics.max_speed_joint
                        ),
                        "output_max_wrist_abs_rad": (
                            f"{output_wrist_diagnostics.max_abs_rad:.9f}"
                        ),
                        "output_max_wrist_angle_joint": (
                            output_wrist_diagnostics.max_abs_joint
                        ),
                        "output_max_wrist_speed_rad_s": (
                            f"{output_wrist_diagnostics.max_speed_rad_s:.9f}"
                        ),
                        "output_max_wrist_speed_joint": (
                            output_wrist_diagnostics.max_speed_joint
                        ),
                        "root_x": (
                            f"{motion_frame.root_position[0]:.6f}"
                            if motion_frame.root_position is not None
                            else ""
                        ),
                        "root_y": (
                            f"{motion_frame.root_position[1]:.6f}"
                            if motion_frame.root_position is not None
                            else ""
                        ),
                        "root_z": (
                            f"{motion_frame.root_position[2]:.6f}"
                            if motion_frame.root_position is not None
                            else ""
                        ),
                        "root_yaw": (
                            f"{root_yaw:.6f}"
                            if (root_yaw := frame_root_yaw(motion_frame)) is not None
                            else ""
                        ),
                        "switch_anchor_xy_delta_m": f"{trace_anchor_xy_delta:.9f}",
                        "switch_anchor_z_delta_m": f"{trace_anchor_z_delta:.9f}",
                        "switch_anchor_yaw_delta_rad": f"{trace_anchor_yaw_delta:.9f}",
                        "post_switch_xy_delta_m": f"{latest_post_switch_xy_delta:.9f}",
                        "post_switch_z_delta_m": f"{latest_post_switch_z_delta:.9f}",
                        "post_switch_yaw_delta_rad": f"{latest_post_switch_yaw_delta:.9f}",
                        "accepted_beat_time_seconds": (
                            f"{latest_accepted_beat_time:.6f}"
                            if latest_accepted_beat_time is not None
                            else ""
                        ),
                        "accepted_beat_pose_error_rad": f"{latest_beat_pose_error:.9f}",
                        "accepted_beat_delay_seconds": (
                            f"{latest_beat_delay_seconds:.9f}"
                            if latest_beat_delay_seconds is not None
                            else ""
                        ),
                        "accepted_beat_limiter_joints": latest_beat_activated_joints,
                        "target_max_joint_speed_rad_s": f"{dynamics_limiter.last_target_speed_rad_s:.6f}",
                        "output_max_joint_speed_rad_s": f"{dynamics_limiter.last_output_speed_rad_s:.6f}",
                        "output_max_joint_acceleration_rad_s2": (
                            f"{dynamics_limiter.last_output_acceleration_rad_s2:.6f}"
                        ),
                        "limiter_activated_joints": ";".join(dynamics_limiter.last_activated),
                        "limiter_activation_rate": f"{dynamics_limiter.activation_rate:.9f}",
                        "collision_override": "1" if collision_override else "0",
                        "limiter_enabled": (
                            "0" if args.diagnostic_disable_output_limiter else "1"
                        ),
                        "terminal_safe_idle_active": (
                            "1" if terminal_safe_idle_active else "0"
                        ),
                        "ready_pool_size": str(len(loader.cache)),
                        "ready_motion_ids": ";".join(loader.cache),
                        "failed_motion_ids": ";".join(loader.failed_motions),
                        "motion_load_ms": (
                            f"{loader.load_seconds[-1] * 1000.0:.6f}"
                            if loader.load_seconds
                            else ""
                        ),
                        "motion_grounding_ms": (
                            f"{loader.prepare_seconds[-1] * 1000.0:.6f}"
                            if loader.prepare_seconds
                            else ""
                        ),
                        "entry_feature_ms": (
                            f"{loader.feature_seconds[-1] * 1000.0:.6f}"
                            if loader.feature_seconds
                            else ""
                        ),
                        "motion_fully_ready_ms": (
                            f"{loader.ready_seconds[-1] * 1000.0:.6f}"
                            if loader.ready_seconds
                            else ""
                        ),
                        "match_state": (
                            "warming_up"
                            if last_match_result is None
                            else ("accepted" if last_match_result.accepted else "rejected")
                        ),
                        "match_confidence": (
                            f"{last_match_result.confidence:.6f}"
                            if last_match_result is not None
                            else ""
                        ),
                        "match_rejection_reason": (
                            last_match_result.rejection_reason
                            if last_match_result is not None
                            else ""
                        ),
                        "beat_quality": (
                            f"{last_match_result.beat_quality:.6f}"
                            if last_match_result is not None
                            else ""
                        ),
                        "signal_rms": (
                            f"{last_match_result.signal_rms:.9f}"
                            if last_match_result is not None
                            else ""
                        ),
                        "dynamic_range_db": (
                            f"{last_match_result.dynamic_range_db:.6f}"
                            if last_match_result is not None
                            else ""
                        ),
                        "negative_style_score": (
                            f"{last_match_result.negative_style_score:.6f}"
                            if last_match_result is not None
                            else ""
                        ),
                        "top_genre": (
                            last_match_result.genres[0].genre
                            if last_match_result is not None and last_match_result.genres
                            else ""
                        ),
                        "top_genre_score": (
                            f"{last_match_result.genres[0].score:.6f}"
                            if last_match_result is not None and last_match_result.genres
                            else ""
                        ),
                        "track_margin": (
                            f"{last_match_result.track_margin:.6f}"
                            if last_match_result is not None
                            else ""
                        ),
                        "motion_margin": (
                            f"{last_match_result.motion_margin:.6f}"
                            if last_match_result is not None
                            else ""
                        ),
                        "ranked_tracks": (
                            ";".join(
                                f"{item.music_id}:{item.score:.6f}"
                                for item in last_match_result.tracks
                            )
                            if last_match_result is not None
                            else ""
                        ),
                        "ranked_motions": (
                            ";".join(
                                f"{item.motion_id}:{item.final_score:.6f}"
                                for item in last_match_result.motions
                            )
                            if last_match_result is not None
                            else ""
                        ),
                        "query_tags": (
                            ";".join(
                                f"{label}:{score:.6f}"
                                for label, score in last_match_result.query_tags
                            )
                            if last_match_result is not None
                            else ""
                        ),
                        "current_motion_cluster_id": catalog.motions[
                            current_id
                        ].motion_cluster_id,
                        "pending_motion_cluster_id": (
                            pending_selection.cluster_id
                            if pending_selection is not None
                            else ""
                        ),
                        "top_track_id": top_track.music_id if top_track is not None else "",
                        "top_track_score": (
                            f"{top_track.score:.6f}" if top_track is not None else ""
                        ),
                        "top_motion_id": top_motion.motion_id if top_motion is not None else "",
                        "top_motion_score": (
                            f"{top_motion.final_score:.6f}" if top_motion is not None else ""
                        ),
                        "query_bpm": (
                            f"{last_match_result.query_bpm:.6f}"
                            if last_match_result is not None
                            else ""
                        ),
                    }
                )
                if elapsed + 1e-9 >= trace_next_audio_time:
                    trace_next_audio_time = (
                        math.floor(elapsed / trace_interval + 1.0) * trace_interval
                    )

            if now - last_status >= args.status_interval:
                base.print_status(
                    current_controller,
                    features,
                    last_frame,
                    recent_detected_beat_frame,
                    recent_accepted_beat_frame,
                )
                recent_detected_beat_frame = None
                recent_accepted_beat_frame = None
                last_status = now
            scheduler.wait(work_started)
    finally:
        source.stop()
        retrieval.close()
        if authored_playback is not None:
            authored_playback.search_cancel.set()
        loader.close()
        if args.experiment_pose_npz is not None:
            pose_path = (
                args.experiment_pose_npz
                if args.experiment_pose_npz.is_absolute()
                else ROOT / args.experiment_pose_npz
            )
            pose_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pose_path,
                schema_version=np.asarray(2 if causal_experiment else 1, dtype=np.int32),
                beat_input_mode=np.asarray("online_causal" if causal_experiment else "preanalysed_replay"),
                wall_time_seconds=np.asarray(pose_wall_times, dtype=np.float64),
                wall_clock_origin_perf_counter=np.asarray(audio_wall_origin),
                final_qpos=np.asarray(pose_qpos, dtype=np.float64),
                joint_qpos_indices=np.asarray(experiment_qpos_ids, dtype=np.int32),
                joint_ranges=np.asarray(experiment_joint_ranges, dtype=np.float64),
                model_xml_sha256=np.asarray(hashlib.sha256(Path(args.model).read_bytes()).hexdigest()),
                model_nq=np.asarray(player.model.nq),
                safety_counts=np.asarray(pose_safety, dtype=np.int32).reshape(-1, 4),
                safety_columns=np.asarray(["candidate_clearance_detection", "collision_projection",
                                           "final_residual_clearance", "final_joint_limit"]),
                beat_events=np.asarray(beat_events, dtype=np.float64).reshape(-1, 4),
                beat_event_columns=np.asarray(["event_wall_seconds", "delivery_wall_seconds",
                                               "audio_received_seconds", "accepted"]),
                control_rate_hz=np.asarray(args.control_rate_hz, dtype=np.float64),
                time_seconds=np.asarray(pose_times, dtype=np.float64),
                frame_index=np.arange(len(pose_times), dtype=np.int64),
                audio_sample_rate_hz=np.asarray(getattr(source, "sample_rate", 0), dtype=np.int32),
                audio_received_sample_index=np.rint(np.asarray(pose_times) * getattr(source, "sample_rate", 0)).astype(np.int64),
                joint_names=np.asarray(experiment_joint_names, dtype=np.str_),
                joint_positions=np.asarray(pose_joint_positions, dtype=np.float32),
                body_names=np.asarray(experiment_body_names, dtype=np.str_),
                body_positions=np.asarray(pose_body_positions, dtype=np.float32),
                center_of_mass=np.asarray(pose_center_of_mass, dtype=np.float32),
                left_foot_position=np.asarray(pose_left_foot, dtype=np.float32),
                right_foot_position=np.asarray(pose_right_foot, dtype=np.float32),
                left_foot_support_height=np.asarray(
                    pose_left_support_height,
                    dtype=np.float32,
                ),
                right_foot_support_height=np.asarray(
                    pose_right_support_height,
                    dtype=np.float32,
                ),
                motion_ids=np.asarray(pose_motion_ids, dtype=np.str_),
                events=np.asarray(pose_events, dtype=np.str_),
                accepted_causal_beat_times_seconds=np.asarray(
                    accepted_beat_times_seconds,
                    dtype=np.float64,
                ),
                kinematic_preview_only=np.asarray(True, dtype=np.bool_),
            )
        player.stop()
        if trace_handle is not None:
            trace_handle.close()
        base.write_timing_report(
            args.timing_report,
            scheduler,
            getattr(source, "analyzer", None),
            player,
            extra={
                **startup_timings_ms,
                **retrieval.timing_summary(),
                **loader.timing_summary(),
                **control_stage_timing_summary(control_stage_seconds),
                "evaluation_warmup_seconds": args.experiment_warmup_seconds,
                "beat_input_mode": "online_causal" if causal_experiment else "preanalysed_replay",
                "pose_schema_version": 2 if causal_experiment else 1,
                "onnxruntime_providers": extractor.embedding_backend.session.get_providers()
                if extractor.embedding_backend is not None else [],
                "collision_candidate_detections": getattr(player, "collision_candidate_violation_count", 0),
                "collision_corrections": player.collision_projection_count,
                "collision_projection_fallbacks": player.collision_projection_fallback_count,
                "collision_dynamics_infeasible": player.collision_dynamics_infeasible_count,
                "final_residual_clearance_violations": final_residual_count if causal_experiment else None,
                "final_joint_limit_violations": final_limit_count if causal_experiment else None,
                "motion_timing": args.motion_timing,
                "max_raw_wrist_abs_rad": max_raw_wrist_abs_rad,
                "max_raw_wrist_speed_rad_s": max_raw_wrist_speed_rad_s,
                "max_output_wrist_abs_rad": max_output_wrist_abs_rad,
                "max_output_wrist_speed_rad_s": max_output_wrist_speed_rad_s,
                "forced_fallbacks": forced_fallback_count,
                "hold_last_events": hold_last_count,
                "collision_dynamics_overrides": collision_override_count,
                "joint_limit_violations": joint_limit_violations,
                "self_collision_violations": final_residual_count if causal_experiment else None,
                "limiter_enabled": not args.diagnostic_disable_output_limiter,
                "limiter_activation_rate": dynamics_limiter.activation_rate,
                "limiter_max_target_speed_rad_s": dynamics_limiter.max_target_speed_rad_s,
                "limiter_max_output_speed_rad_s": dynamics_limiter.max_output_speed_rad_s,
                "limiter_max_output_acceleration_rad_s2": (
                    dynamics_limiter.max_output_acceleration_rad_s2
                ),
                "accepted_beats": accepted_beat_count,
                "limiter_impacted_beats": limiter_impacted_beats,
                "limiter_impacted_beat_ratio": (
                    limiter_impacted_beats / max(accepted_beat_count, 1)
                ),
                "accepted_beat_pose_error_rad_p50": (
                    float(np.percentile(beat_pose_errors_rad, 50.0))
                    if beat_pose_errors_rad
                    else 0.0
                ),
                "accepted_beat_pose_error_rad_p95": (
                    float(np.percentile(beat_pose_errors_rad, 95.0))
                    if beat_pose_errors_rad
                    else 0.0
                ),
                "accepted_beat_pose_error_rad_max": (
                    float(np.max(beat_pose_errors_rad))
                    if beat_pose_errors_rad
                    else 0.0
                ),
                "accepted_beat_delay_ms_p50": (
                    float(np.percentile(beat_delays_seconds, 50.0) * 1000.0)
                    if beat_delays_seconds
                    else 0.0
                ),
                "accepted_beat_delay_ms_p95": (
                    float(np.percentile(beat_delays_seconds, 95.0) * 1000.0)
                    if beat_delays_seconds
                    else 0.0
                ),
                "accepted_beat_delay_ms_max": (
                    float(np.max(beat_delays_seconds) * 1000.0)
                    if beat_delays_seconds
                    else 0.0
                ),
                "accepted_beat_delay_warning": bool(
                    beat_delays_seconds
                    and (
                        np.percentile(beat_delays_seconds, 95.0)
                        > scheduler.period + 1e-9
                        or np.max(beat_delays_seconds)
                        > 2.0 * scheduler.period + 1e-9
                    )
                ),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
