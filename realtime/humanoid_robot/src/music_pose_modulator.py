from __future__ import annotations

import math
from collections.abc import Mapping


class MusicPoseModulator:
    """Apply one music-to-motion mapping to dancer and Unitree G1 pose streams."""

    def __init__(self, strength: float = 1.0) -> None:
        self.strength = max(strength, 0.0)

    def modulate(
        self,
        pose: Mapping[str, float],
        features: object,
        *,
        phase: float,
        amplitude: float,
        accent: float,
    ) -> dict[str, float]:
        phase = phase % 1.0
        beat = 2.0 * math.pi * phase
        half = 4.0 * math.pi * phase
        side = math.sin(beat)
        counter = math.sin(beat + 0.25 * math.pi)
        bounce = 0.5 * (1.0 - math.cos(half))
        offbeat_wave = math.sin(half + 0.5 * math.pi)

        low = self._feature(features, "low_energy")
        mid = self._feature(features, "mid_energy")
        high = self._feature(features, "high_energy")
        bright = self._feature(features, "brightness")
        rhythm = self._feature(features, "rhythm_density")
        offbeat = self._feature(features, "offbeat_ratio")

        amp = max(amplitude, 0.0)
        hit = self.strength * max(accent, 0.0)
        texture = self.strength * max(high, rhythm)

        modulated = {name: float(value) * amp for name, value in pose.items()}

        # Bass makes the body feel grounded: hips/waist sway more and knees bounce deeper.
        self._add_any(modulated, ("torso_roll", "waist_roll"), self.strength * 0.10 * low * side)
        self._add_pair(modulated, "left_hip_roll", "right_hip_roll", self.strength * 0.08 * low * side)
        self._add_pair(modulated, "left_knee", "right_knee", self.strength * 0.12 * low * bounce)
        self._add_pair(modulated, "left_ankle_pitch", "right_ankle_pitch", -self.strength * 0.04 * low * bounce)

        # Mid-band energy reads as torso/shoulder phrasing.
        self._add_any(modulated, ("torso_yaw", "waist_yaw"), self.strength * 0.10 * mid * counter)
        self._add_pair(modulated, "left_shoulder_yaw", "right_shoulder_yaw", self.strength * 0.06 * mid * counter)

        # Bright/high-frequency material lifts the arms and adds faster upper-body texture.
        self._add(modulated, "left_shoulder_roll", self.strength * (0.10 * bright + 0.05 * hit))
        self._add(modulated, "right_shoulder_roll", -self.strength * (0.10 * bright + 0.05 * hit))
        self._add(modulated, "left_shoulder_pitch", self.strength * 0.06 * high * math.sin(half))
        self._add(modulated, "right_shoulder_pitch", -self.strength * 0.06 * high * math.sin(half))
        self._bend(modulated, "left_elbow", -self.strength * (0.08 * texture * math.sin(half) + 0.10 * hit))
        self._bend(modulated, "right_elbow", self.strength * (0.08 * texture * math.sin(half) - 0.10 * hit))

        # Offbeat/rhythm density gives small subdivision motion without changing the main beat lock.
        self._add_any(modulated, ("neck_pitch",), self.strength * 0.05 * offbeat * offbeat_wave)
        self._add_any(modulated, ("torso_pitch", "waist_pitch"), self.strength * (0.05 * hit - 0.04 * rhythm * bounce))

        # Wrist keys are present on G1 streams and are also consumed by the G1 dance adapter.
        self._add(modulated, "left_wrist_roll", self.strength * (0.06 * high * math.sin(half) + 0.04 * hit))
        self._add(modulated, "right_wrist_roll", -self.strength * (0.06 * high * math.sin(half) + 0.04 * hit))
        self._add(modulated, "left_wrist_pitch", -self.strength * 0.05 * rhythm * bounce)
        self._add(modulated, "right_wrist_pitch", -self.strength * 0.05 * rhythm * bounce)
        self._add(modulated, "left_wrist_yaw", self.strength * 0.05 * low * side)
        self._add(modulated, "right_wrist_yaw", -self.strength * 0.05 * low * side)

        return modulated

    @staticmethod
    def _feature(features: object, name: str) -> float:
        return float(max(0.0, min(1.0, getattr(features, name, 0.0))))

    @staticmethod
    def _add(pose: dict[str, float], name: str, delta: float) -> None:
        pose[name] = float(pose.get(name, 0.0) + delta)

    @classmethod
    def _add_any(cls, pose: dict[str, float], names: tuple[str, ...], delta: float) -> None:
        for name in names:
            if name in pose:
                cls._add(pose, name, delta)
                return
        if names:
            cls._add(pose, names[0], delta)

    @classmethod
    def _add_pair(cls, pose: dict[str, float], left_name: str, right_name: str, delta: float) -> None:
        cls._add(pose, left_name, delta)
        cls._add(pose, right_name, delta)

    @classmethod
    def _bend(cls, pose: dict[str, float], name: str, delta: float) -> None:
        current = pose.get(name, 0.0)
        if current < 0.0:
            cls._add(pose, name, -abs(delta))
        elif current > 0.0:
            cls._add(pose, name, abs(delta))
        else:
            cls._add(pose, name, delta)
