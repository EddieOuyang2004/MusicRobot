from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RobotMotionFrame:
    """One kinematic humanoid target, including the floating base when available."""

    joint_positions: dict[str, float]
    root_position: np.ndarray | None = None
    root_quaternion_wxyz: np.ndarray | None = None

    def with_joint_positions(self, values: Mapping[str, float]) -> "RobotMotionFrame":
        return RobotMotionFrame(
            joint_positions={name: float(value) for name, value in values.items()},
            root_position=None if self.root_position is None else self.root_position.copy(),
            root_quaternion_wxyz=(
                None if self.root_quaternion_wxyz is None else self.root_quaternion_wxyz.copy()
            ),
        )


@dataclass(frozen=True)
class JointDynamicsLimits:
    max_speed_rad_s: float
    max_acceleration_rad_s2: float


def load_joint_dynamics_limits(
    path: Path | None,
    joint_names: set[str],
    *,
    default_speed: float,
    default_acceleration: float,
) -> dict[str, JointDynamicsLimits]:
    """Load optional per-joint output limits with strict joint-name validation."""

    def validated(speed: object, acceleration: object, description: str) -> JointDynamicsLimits:
        try:
            speed_value = float(speed)
            acceleration_value = float(acceleration)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{description} must contain numeric speed and acceleration limits.") from exc
        if (
            not math.isfinite(speed_value)
            or speed_value <= 0.0
            or not math.isfinite(acceleration_value)
            or acceleration_value <= 0.0
        ):
            raise ValueError(f"{description} limits must be positive and finite.")
        return JointDynamicsLimits(speed_value, acceleration_value)

    fallback = validated(
        default_speed,
        default_acceleration,
        "default joint dynamics",
    )
    overrides: dict[str, JointDynamicsLimits] = {}
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Joint dynamics limits JSON must contain an object.")
        default_payload = payload.get("default", {})
        if not isinstance(default_payload, dict):
            raise ValueError("Joint dynamics limits 'default' must contain an object.")
        fallback = validated(
            default_payload.get("max_speed_rad_s", fallback.max_speed_rad_s),
            default_payload.get(
                "max_acceleration_rad_s2",
                fallback.max_acceleration_rad_s2,
            ),
            "default joint dynamics",
        )
        joints_payload = payload.get("joints", {})
        if not isinstance(joints_payload, dict):
            raise ValueError("Joint dynamics limits 'joints' must contain an object.")
        unknown = set(joints_payload) - set(joint_names)
        if unknown:
            raise ValueError(f"Unknown joints in dynamics limits: {sorted(unknown)}")
        for name, values in joints_payload.items():
            if not isinstance(values, dict):
                raise ValueError(f"Joint dynamics for {name} must contain an object.")
            overrides[name] = validated(
                values.get("max_speed_rad_s", fallback.max_speed_rad_s),
                values.get(
                    "max_acceleration_rad_s2",
                    fallback.max_acceleration_rad_s2,
                ),
                f"joint dynamics for {name}",
            )
    return {name: overrides.get(name, fallback) for name in joint_names}


class JointDynamicsLimiter:
    """Stateful joint target tracker with velocity and acceleration bounds."""

    def __init__(self, limits: Mapping[str, JointDynamicsLimits]) -> None:
        self.limits = dict(limits)
        self.positions: dict[str, float] = {}
        self.velocities: dict[str, float] = {}
        self.targets: dict[str, float] = {}
        self.target_velocities: dict[str, float] = {}
        self.activation_counts = {name: 0 for name in self.limits}
        self.samples = 0
        self.joint_samples = 0
        self.max_target_speed_rad_s = 0.0
        self.max_output_speed_rad_s = 0.0
        self.max_output_acceleration_rad_s2 = 0.0
        self.last_activated: tuple[str, ...] = ()
        self.last_target_speed_rad_s = 0.0
        self.last_output_speed_rad_s = 0.0
        self.last_output_acceleration_rad_s2 = 0.0

    def reset(self, frame: RobotMotionFrame | None = None) -> None:
        self.positions.clear()
        self.velocities.clear()
        self.targets.clear()
        self.target_velocities.clear()
        if frame is not None:
            for name, value in frame.joint_positions.items():
                if name in self.limits:
                    self.positions[name] = float(value)
                    self.velocities[name] = 0.0
                    self.targets[name] = float(value)
                    self.target_velocities[name] = 0.0
        self.last_activated = ()

    def apply(self, frame: RobotMotionFrame, dt: float) -> RobotMotionFrame:
        if not math.isfinite(dt) or dt <= 0.0:
            joints = dict(frame.joint_positions)
            for name, previous in self.positions.items():
                if name in joints:
                    joints[name] = previous
            self.last_activated = tuple(
                sorted(name for name in self.positions if name in frame.joint_positions)
            )
            return frame.with_joint_positions(joints)

        output = dict(frame.joint_positions)
        activated: list[str] = []
        target_speed_max = 0.0
        output_speed_max = 0.0
        output_acceleration_max = 0.0
        for name, target_value in frame.joint_positions.items():
            limits = self.limits.get(name)
            if limits is None:
                continue
            target = float(target_value)
            if name not in self.positions:
                self.positions[name] = target
                self.velocities[name] = 0.0
                self.targets[name] = target
                self.target_velocities[name] = 0.0
                continue
            previous = self.positions[name]
            previous_velocity = self.velocities.get(name, 0.0)
            previous_target = self.targets.get(name, previous)
            previous_target_velocity = self.target_velocities.get(name, 0.0)
            requested_velocity = (target - previous_target) / dt
            requested_acceleration = (
                requested_velocity - previous_target_velocity
            ) / dt
            constraint_engaged = (
                abs(requested_velocity) > limits.max_speed_rad_s + 1e-10
                or abs(requested_acceleration)
                > limits.max_acceleration_rad_s2 + 1e-10
            )
            error = target - previous
            target_velocity = error / dt
            target_speed_max = max(target_speed_max, abs(target_velocity))

            target_is_moving = abs(target - previous_target) > 1e-10
            if target_is_moving:
                desired_velocity = float(
                    np.clip(
                        target_velocity,
                        -limits.max_speed_rad_s,
                        limits.max_speed_rad_s,
                    )
                )
            elif abs(error) <= 1e-12:
                desired_velocity = 0.0
            else:
                direction = math.copysign(1.0, error)
                moving_toward_target = previous_velocity * direction > 0.0
                stopping_distance = (
                    previous_velocity * previous_velocity
                    / (2.0 * limits.max_acceleration_rad_s2)
                    + abs(previous_velocity) * dt
                )
                if moving_toward_target and stopping_distance >= abs(error):
                    desired_velocity = 0.0
                else:
                    acceleration_step = limits.max_acceleration_rad_s2 * dt
                    braking_speed = max(
                        math.sqrt(
                            max(
                                2.0 * limits.max_acceleration_rad_s2 * abs(error)
                                + acceleration_step * acceleration_step,
                                0.0,
                            )
                        )
                        - acceleration_step,
                        0.0,
                    )
                    desired_velocity = direction * min(
                        abs(target_velocity),
                        limits.max_speed_rad_s,
                        braking_speed,
                    )
            acceleration_delta = limits.max_acceleration_rad_s2 * dt
            velocity = float(
                np.clip(
                    desired_velocity,
                    previous_velocity - acceleration_delta,
                    previous_velocity + acceleration_delta,
                )
            )
            velocity = float(
                np.clip(velocity, -limits.max_speed_rad_s, limits.max_speed_rad_s)
            )
            position = previous + velocity * dt
            if error != 0.0 and (target - position) * error < 0.0:
                feasible_velocity = error / dt
                if abs(feasible_velocity - previous_velocity) <= acceleration_delta + 1e-12:
                    position = target
                    velocity = feasible_velocity

            acceleration = (velocity - previous_velocity) / dt
            output[name] = float(position)
            self.positions[name] = float(position)
            self.velocities[name] = float(velocity)
            self.targets[name] = target
            self.target_velocities[name] = requested_velocity
            output_speed_max = max(output_speed_max, abs(velocity))
            output_acceleration_max = max(output_acceleration_max, abs(acceleration))
            self.joint_samples += 1
            if constraint_engaged:
                activated.append(name)
                self.activation_counts[name] += 1

        self.samples += 1
        self.last_activated = tuple(sorted(activated))
        self.last_target_speed_rad_s = target_speed_max
        self.last_output_speed_rad_s = output_speed_max
        self.last_output_acceleration_rad_s2 = output_acceleration_max
        self.max_target_speed_rad_s = max(self.max_target_speed_rad_s, target_speed_max)
        self.max_output_speed_rad_s = max(self.max_output_speed_rad_s, output_speed_max)
        self.max_output_acceleration_rad_s2 = max(
            self.max_output_acceleration_rad_s2,
            output_acceleration_max,
        )
        return frame.with_joint_positions(output)

    @property
    def activation_rate(self) -> float:
        return sum(self.activation_counts.values()) / max(self.joint_samples, 1)


def normalize_wxyz(quaternion: np.ndarray, *, description: str = "quaternion") -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{description} must contain four finite wxyz values, got {value}.")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8:
        raise ValueError(f"{description} has zero length.")
    return value / norm


def wxyz_to_rotation(quaternion: np.ndarray) -> Rotation:
    w, x, y, z = normalize_wxyz(quaternion)
    return Rotation.from_quat([x, y, z, w])


def rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    x, y, z, w = rotation.as_quat()
    return normalize_wxyz(np.asarray([w, x, y, z], dtype=np.float64))


def yaw_only(rotation: Rotation) -> Rotation:
    """Return only the world-Z heading component of a Z-up rotation."""

    matrix = rotation.as_matrix()
    return Rotation.from_euler("z", math.atan2(matrix[1, 0], matrix[0, 0]))


def slerp_wxyz(first: np.ndarray, second: np.ndarray, amount: float) -> np.ndarray:
    """Shortest-path quaternion interpolation without constructing a Slerp object."""

    a = normalize_wxyz(first)
    b = normalize_wxyz(second)
    t = float(np.clip(amount, 0.0, 1.0))
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_wxyz((1.0 - t) * a + t * b)
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    return normalize_wxyz(
        np.sin((1.0 - t) * theta) / sin_theta * a
        + np.sin(t * theta) / sin_theta * b
    )


def blend_motion_frames(
    first: RobotMotionFrame,
    second: RobotMotionFrame,
    amount: float,
    *,
    smoothstep: bool = True,
) -> RobotMotionFrame:
    t = float(np.clip(amount, 0.0, 1.0))
    smooth = t * t * (3.0 - 2.0 * t) if smoothstep else t
    names = set(first.joint_positions) | set(second.joint_positions)
    joints = {
        name: float(
            (1.0 - smooth) * first.joint_positions.get(name, 0.0)
            + smooth * second.joint_positions.get(name, 0.0)
        )
        for name in names
    }

    if first.root_position is None:
        root_position = None if second.root_position is None else second.root_position.copy()
    elif second.root_position is None:
        root_position = first.root_position.copy()
    else:
        root_position = (1.0 - smooth) * first.root_position + smooth * second.root_position

    if first.root_quaternion_wxyz is None:
        root_quaternion = (
            None if second.root_quaternion_wxyz is None else second.root_quaternion_wxyz.copy()
        )
    elif second.root_quaternion_wxyz is None:
        root_quaternion = first.root_quaternion_wxyz.copy()
    else:
        root_quaternion = slerp_wxyz(
            first.root_quaternion_wxyz,
            second.root_quaternion_wxyz,
            smooth,
        )
    return RobotMotionFrame(joints, root_position, root_quaternion)


def align_motion_frame_root(
    frame: RobotMotionFrame,
    *,
    source_reference: RobotMotionFrame,
    target_reference: RobotMotionFrame,
) -> RobotMotionFrame:
    """Yaw/XY-align a source root while preserving its grounded Z, roll and pitch."""

    if (
        frame.root_position is None
        or frame.root_quaternion_wxyz is None
        or source_reference.root_position is None
        or source_reference.root_quaternion_wxyz is None
        or target_reference.root_position is None
        or target_reference.root_quaternion_wxyz is None
    ):
        return frame
    source_rotation = wxyz_to_rotation(source_reference.root_quaternion_wxyz)
    target_rotation = wxyz_to_rotation(target_reference.root_quaternion_wxyz)
    alignment_rotation = yaw_only(target_rotation) * yaw_only(source_rotation).inv()
    relative_position = np.asarray(frame.root_position) - source_reference.root_position
    aligned_relative = alignment_rotation.apply(relative_position)
    root_position = np.asarray(frame.root_position, dtype=np.float64).copy()
    root_position[:2] = target_reference.root_position[:2] + aligned_relative[:2]
    return RobotMotionFrame(
        joint_positions=dict(frame.joint_positions),
        root_position=root_position,
        root_quaternion_wxyz=rotation_to_wxyz(
            alignment_rotation * wxyz_to_rotation(frame.root_quaternion_wxyz)
        ),
    )


class RootMotionContinuity:
    """Anchor clip-local roots into one continuous MuJoCo world trajectory."""

    def __init__(
        self,
        anchor_position: np.ndarray,
        anchor_quaternion_wxyz: np.ndarray,
        mode: str = "continuous",
        grounded_z: bool = False,
    ) -> None:
        if mode not in {"continuous", "in-place", "reset"}:
            raise ValueError(f"Unknown root motion mode: {mode}")
        self.mode = mode
        self.grounded_z = bool(grounded_z)
        self.initial_anchor_position = np.asarray(anchor_position, dtype=np.float64).copy()
        self.initial_anchor_rotation = wxyz_to_rotation(anchor_quaternion_wxyz)
        self.anchor_position = self.initial_anchor_position.copy()
        self.anchor_rotation = self.initial_anchor_rotation
        self.source_id: str | None = None
        self.source_origin_position: np.ndarray | None = None
        self.source_origin_rotation: Rotation | None = None
        self.last_phase: float | None = None
        self.last_world_position = self.anchor_position.copy()
        self.last_world_rotation = self.anchor_rotation

    def apply(self, frame: RobotMotionFrame, *, phase: float, source_id: str) -> RobotMotionFrame:
        if frame.root_position is None or frame.root_quaternion_wxyz is None:
            return frame

        raw_position = np.asarray(frame.root_position, dtype=np.float64)
        raw_rotation = wxyz_to_rotation(frame.root_quaternion_wxyz)
        phase = float(phase % 1.0)
        source_changed = source_id != self.source_id
        wrapped = (
            not source_changed
            and self.last_phase is not None
            and phase + 0.5 < self.last_phase
        )

        if source_changed:
            if self.source_id is not None:
                self.anchor_position = self.last_world_position.copy()
                self.anchor_rotation = self.last_world_rotation
            self.source_id = source_id
            self.source_origin_position = raw_position.copy()
            self.source_origin_rotation = raw_rotation
        elif wrapped and self.mode == "continuous":
            self.anchor_position[:2] = self.last_world_position[:2]
            self.anchor_rotation = yaw_only(self.last_world_rotation)
            self.source_origin_position = raw_position.copy()
            self.source_origin_rotation = raw_rotation
        elif wrapped and self.mode == "reset":
            self.anchor_position = self.initial_anchor_position.copy()
            self.anchor_rotation = self.initial_anchor_rotation
            self.source_origin_position = raw_position.copy()
            self.source_origin_rotation = raw_rotation
        elif self.source_origin_position is None or self.source_origin_rotation is None:
            self.source_id = source_id
            self.source_origin_position = raw_position.copy()
            self.source_origin_rotation = raw_rotation

        assert self.source_origin_position is not None
        assert self.source_origin_rotation is not None
        relative_position = raw_position - self.source_origin_position
        relative_rotation = self.source_origin_rotation.inv() * raw_rotation

        if self.mode == "in-place":
            relative_position = relative_position.copy()
            relative_position[:2] = 0.0
        horizontal = relative_position.copy()
        horizontal[2] = 0.0
        world_position = self.anchor_position.copy()
        world_position[:2] += self.anchor_rotation.apply(horizontal)[:2]
        world_position[2] = raw_position[2] if self.grounded_z else world_position[2] + relative_position[2]
        world_rotation = self.anchor_rotation * relative_rotation
        self.last_phase = phase
        self.last_world_position = world_position
        self.last_world_rotation = world_rotation
        return RobotMotionFrame(
            joint_positions=dict(frame.joint_positions),
            root_position=world_position,
            root_quaternion_wxyz=rotation_to_wxyz(world_rotation),
        )
