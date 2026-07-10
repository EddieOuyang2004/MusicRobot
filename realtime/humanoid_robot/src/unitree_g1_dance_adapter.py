from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


class UnitreeG1DanceAdapter:
    """Expand the simple dancer pose into a Unitree G1-style joint target.

    The current dancer sampler emits a compact 20-DoF pose with project-local
    names such as ``left_hip_pitch``. Official Unitree humanoid MJCF/URDF files
    normally expose hardware-oriented names such as ``left_hip_pitch_joint`` and
    include extra axes for waist, ankle roll, shoulder yaw, and wrists. This
    adapter keeps the sampler independent from a specific Unitree model file.
    """

    LOGICAL_ALIASES: dict[str, tuple[str, ...]] = {
        "left_hip_pitch": ("left_hip_pitch_joint", "left_hip_pitch"),
        "left_hip_roll": ("left_hip_roll_joint", "left_hip_roll"),
        "left_hip_yaw": ("left_hip_yaw_joint", "left_hip_yaw"),
        "left_knee": ("left_knee_joint", "left_knee"),
        "left_ankle_pitch": ("left_ankle_pitch_joint", "left_ankle_pitch"),
        "left_ankle_roll": ("left_ankle_roll_joint", "left_ankle_roll"),
        "right_hip_pitch": ("right_hip_pitch_joint", "right_hip_pitch"),
        "right_hip_roll": ("right_hip_roll_joint", "right_hip_roll"),
        "right_hip_yaw": ("right_hip_yaw_joint", "right_hip_yaw"),
        "right_knee": ("right_knee_joint", "right_knee"),
        "right_ankle_pitch": ("right_ankle_pitch_joint", "right_ankle_pitch"),
        "right_ankle_roll": ("right_ankle_roll_joint", "right_ankle_roll"),
        "waist_yaw": ("waist_yaw_joint", "waist_yaw", "torso_yaw"),
        "waist_roll": ("waist_roll_joint", "waist_roll", "torso_roll"),
        "waist_pitch": ("waist_pitch_joint", "waist_pitch", "torso_pitch"),
        "left_shoulder_pitch": ("left_shoulder_pitch_joint", "left_shoulder_pitch"),
        "left_shoulder_roll": ("left_shoulder_roll_joint", "left_shoulder_roll"),
        "left_shoulder_yaw": ("left_shoulder_yaw_joint", "left_shoulder_yaw"),
        "left_elbow": ("left_elbow_joint", "left_elbow"),
        "left_wrist_roll": ("left_wrist_roll_joint", "left_wrist_roll"),
        "left_wrist_pitch": ("left_wrist_pitch_joint", "left_wrist_pitch"),
        "left_wrist_yaw": ("left_wrist_yaw_joint", "left_wrist_yaw"),
        "right_shoulder_pitch": ("right_shoulder_pitch_joint", "right_shoulder_pitch"),
        "right_shoulder_roll": ("right_shoulder_roll_joint", "right_shoulder_roll"),
        "right_shoulder_yaw": ("right_shoulder_yaw_joint", "right_shoulder_yaw"),
        "right_elbow": ("right_elbow_joint", "right_elbow"),
        "right_wrist_roll": ("right_wrist_roll_joint", "right_wrist_roll"),
        "right_wrist_pitch": ("right_wrist_pitch_joint", "right_wrist_pitch"),
        "right_wrist_yaw": ("right_wrist_yaw_joint", "right_wrist_yaw"),
    }

    G1_HINT_NAMES = {
        "waist_yaw_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
    }

    # GMR's Unitree G1 29-DoF order follows the motor order in
    # assets/unitree_g1/g1_mocap_29dof.xml.
    GMR_DOF_NAMES: tuple[str, ...] = (
        "left_hip_pitch",
        "left_hip_roll",
        "left_hip_yaw",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_hip_pitch",
        "right_hip_roll",
        "right_hip_yaw",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll",
        "waist_yaw",
        "waist_roll",
        "waist_pitch",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "left_wrist_roll",
        "left_wrist_pitch",
        "left_wrist_yaw",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        "right_wrist_roll",
        "right_wrist_pitch",
        "right_wrist_yaw",
    )

    def __init__(
        self,
        available_actuators: Iterable[str] | None = None,
        *,
        lower_body_gain: float = 0.65,
        upper_body_gain: float = 0.85,
        waist_gain: float = 0.70,
        wrist_gain: float = 0.25,
    ) -> None:
        self.available_actuators = set(available_actuators or ())
        self.lower_body_gain = max(lower_body_gain, 0.0)
        self.upper_body_gain = max(upper_body_gain, 0.0)
        self.waist_gain = max(waist_gain, 0.0)
        self.wrist_gain = max(wrist_gain, 0.0)
        self.name_map = self._resolve_names()

    @classmethod
    def looks_like_unitree_g1(cls, actuator_names: Iterable[str]) -> bool:
        names = set(actuator_names)
        return bool(cls.G1_HINT_NAMES & names)

    @property
    def resolved_count(self) -> int:
        return len(self.name_map)

    def adapt_pose(self, dancer_pose: Mapping[str, float], features: object | None = None) -> dict[str, float]:
        """Return a pose dictionary keyed by the loaded Unitree actuator names."""

        low_energy = _feature_value(features, "low_energy")
        high_energy = _feature_value(features, "high_energy")
        rhythm_density = _feature_value(features, "rhythm_density")

        torso_yaw = dancer_pose.get("torso_yaw", 0.0)
        torso_roll = dancer_pose.get("torso_roll", 0.0)
        torso_pitch = dancer_pose.get("torso_pitch", 0.0)
        left_shoulder_roll = dancer_pose.get("left_shoulder_roll", 0.0)
        right_shoulder_roll = dancer_pose.get("right_shoulder_roll", 0.0)
        left_knee = dancer_pose.get("left_knee", 0.0)
        right_knee = dancer_pose.get("right_knee", 0.0)

        logical_pose = {
            "waist_yaw": self.waist_gain * torso_yaw,
            "waist_roll": self.waist_gain * torso_roll,
            "waist_pitch": self.waist_gain * torso_pitch,
            "left_hip_yaw": self.lower_body_gain * dancer_pose.get("left_hip_yaw", 0.0),
            "left_hip_roll": self.lower_body_gain * dancer_pose.get("left_hip_roll", 0.0),
            "left_hip_pitch": self.lower_body_gain * dancer_pose.get("left_hip_pitch", 0.0),
            "left_knee": self.lower_body_gain * left_knee,
            "left_ankle_pitch": self.lower_body_gain * dancer_pose.get("left_ankle_pitch", 0.0),
            "left_ankle_roll": -0.18 * self.lower_body_gain * dancer_pose.get("left_hip_roll", 0.0),
            "right_hip_yaw": self.lower_body_gain * dancer_pose.get("right_hip_yaw", 0.0),
            "right_hip_roll": self.lower_body_gain * dancer_pose.get("right_hip_roll", 0.0),
            "right_hip_pitch": self.lower_body_gain * dancer_pose.get("right_hip_pitch", 0.0),
            "right_knee": self.lower_body_gain * right_knee,
            "right_ankle_pitch": self.lower_body_gain * dancer_pose.get("right_ankle_pitch", 0.0),
            "right_ankle_roll": -0.18 * self.lower_body_gain * dancer_pose.get("right_hip_roll", 0.0),
            "left_shoulder_pitch": self.upper_body_gain * dancer_pose.get("left_shoulder_pitch", 0.0),
            "left_shoulder_roll": self.upper_body_gain * left_shoulder_roll,
            "left_shoulder_yaw": 0.20 * self.upper_body_gain * torso_yaw + 0.12 * math.sin(left_shoulder_roll),
            "left_elbow": self.upper_body_gain * dancer_pose.get("left_elbow", 0.0),
            "left_wrist_roll": dancer_pose.get(
                "left_wrist_roll",
                self.wrist_gain * (0.65 * left_shoulder_roll + 0.25 * high_energy),
            ),
            "left_wrist_pitch": dancer_pose.get(
                "left_wrist_pitch",
                -self.wrist_gain * (0.35 * left_knee + 0.20 * rhythm_density),
            ),
            "left_wrist_yaw": dancer_pose.get(
                "left_wrist_yaw",
                self.wrist_gain * (0.45 * torso_yaw + 0.20 * low_energy),
            ),
            "right_shoulder_pitch": self.upper_body_gain * dancer_pose.get("right_shoulder_pitch", 0.0),
            "right_shoulder_roll": self.upper_body_gain * right_shoulder_roll,
            "right_shoulder_yaw": 0.20 * self.upper_body_gain * torso_yaw + 0.12 * math.sin(right_shoulder_roll),
            "right_elbow": self.upper_body_gain * dancer_pose.get("right_elbow", 0.0),
            "right_wrist_roll": dancer_pose.get(
                "right_wrist_roll",
                self.wrist_gain * (0.65 * right_shoulder_roll - 0.25 * high_energy),
            ),
            "right_wrist_pitch": dancer_pose.get(
                "right_wrist_pitch",
                -self.wrist_gain * (0.35 * right_knee + 0.20 * rhythm_density),
            ),
            "right_wrist_yaw": dancer_pose.get(
                "right_wrist_yaw",
                self.wrist_gain * (0.45 * torso_yaw - 0.20 * low_energy),
            ),
        }

        return {
            resolved_name: float(logical_pose[logical_name])
            for logical_name, resolved_name in self.name_map.items()
            if logical_name in logical_pose
        }

    def _resolve_names(self) -> dict[str, str]:
        if not self.available_actuators:
            return {logical_name: aliases[0] for logical_name, aliases in self.LOGICAL_ALIASES.items()}

        resolved = {}
        for logical_name, aliases in self.LOGICAL_ALIASES.items():
            for candidate in aliases:
                if candidate in self.available_actuators:
                    resolved[logical_name] = candidate
                    break
        return resolved


class UnitreeG1JointPoseAdapter:
    """Map GMR Unitree G1 joint poses onto the loaded actuator names."""

    def __init__(self, available_actuators: Iterable[str] | None = None) -> None:
        self.available_actuators = set(available_actuators or ())
        self.name_map = self._resolve_names()

    @property
    def resolved_count(self) -> int:
        return len(self.name_map)

    def adapt_pose(self, g1_pose: Mapping[str, float], features: object | None = None) -> dict[str, float]:
        del features
        return {
            resolved_name: float(g1_pose[logical_name])
            for logical_name, resolved_name in self.name_map.items()
            if logical_name in g1_pose
        }

    def pose_from_dof_frame(self, dof_frame: Sequence[float]) -> dict[str, float]:
        if len(dof_frame) < len(UnitreeG1DanceAdapter.GMR_DOF_NAMES):
            raise ValueError(
                "GMR Unitree G1 dof_pos frame must contain at least "
                f"{len(UnitreeG1DanceAdapter.GMR_DOF_NAMES)} values, got {len(dof_frame)}."
            )
        return {
            name: float(dof_frame[index])
            for index, name in enumerate(UnitreeG1DanceAdapter.GMR_DOF_NAMES)
        }

    def _resolve_names(self) -> dict[str, str]:
        if not self.available_actuators:
            return {
                logical_name: aliases[0]
                for logical_name, aliases in UnitreeG1DanceAdapter.LOGICAL_ALIASES.items()
            }

        resolved = {}
        for logical_name, aliases in UnitreeG1DanceAdapter.LOGICAL_ALIASES.items():
            for candidate in aliases:
                if candidate in self.available_actuators:
                    resolved[logical_name] = candidate
                    break
        return resolved


def _feature_value(features: object | None, name: str) -> float:
    if features is None:
        return 0.0
    return float(getattr(features, name, 0.0))
