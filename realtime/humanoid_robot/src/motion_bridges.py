"""Authored joint trajectories and checked, state-to-state motion bridges.

All times are seconds, joint angles radians. This module does not model contacts
or floating-base dynamics. Ruckig is imported only by its optional backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
import time
from typing import Protocol

import numpy as np
from numpy.polynomial import polynomial as poly
from hermite_bounds import classify, warmup as warmup_hermite


@dataclass(frozen=True)
class JointState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class Bridge(Protocol):
    duration: float

    def at_time(self, seconds: float) -> JointState: ...


class InfeasibleBridge(ValueError):
    pass


def smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u ** 3 * (10.0 + u * (-15.0 + 6.0 * u))


def coefficients(start: JointState, end: JointState, duration: float) -> np.ndarray:
    """Coefficients in normalized time; leading axis is polynomial power."""
    c0 = start.position
    c1 = duration * start.velocity
    c2 = duration ** 2 * start.acceleration / 2.0
    d = end.position - c0 - c1 - c2
    v = duration * end.velocity - c1 - 2.0 * c2
    a = duration ** 2 * end.acceleration - 2.0 * c2
    return np.stack((c0, c1, c2, 10*d-4*v+a/2, -15*d+7*v-a, 6*d-3*v+a/2))


@dataclass(frozen=True)
class HermiteBridge:
    duration: float
    coefficients: np.ndarray

    @classmethod
    def between(cls, start: JointState, end: JointState, duration: float):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Bridge duration must be finite and positive.")
        return cls(duration, coefficients(start, end, duration))

    def at_time(self, seconds: float) -> JointState:
        u = float(np.clip(seconds / self.duration, 0.0, 1.0))
        return JointState(*(poly.polyval(u, poly.polyder(self.coefficients, m=k)) /
                            self.duration ** k for k in range(3)))

    def extrema(self, derivative: int = 0) -> tuple[np.ndarray, np.ndarray]:
        c = poly.polyder(self.coefficients, m=derivative) / self.duration ** derivative
        lower, upper = [], []
        for column in c.T:
            roots = poly.polyroots(poly.polyder(column))
            points = [0., 1.] + [float(r.real) for r in roots
                                  if abs(r.imag) < 1e-8 and 0 < r.real < 1]
            values = poly.polyval(points, column)
            lower.append(float(np.min(values)))
            upper.append(float(np.max(values)))
        return np.asarray(lower), np.asarray(upper)

    def within_limits(self, derivative, lower, upper):
        """Fast bounds with the original root calculation for uncertain joints."""
        c = np.ascontiguousarray(poly.polyder(self.coefficients, m=derivative) /
                                 self.duration ** derivative)
        status = classify(c, lower, upper)
        if np.any(status < 0):
            return False
        for joint in np.flatnonzero(status == 0):
            column = c[:, joint]
            roots = poly.polyroots(poly.polyder(column))
            points = [0., 1.] + [float(r.real) for r in roots
                                  if abs(r.imag) < 1e-8 and 0 < r.real < 1]
            values = poly.polyval(points, column)
            if np.min(values) < lower[joint]-1e-7 or np.max(values) > upper[joint]+1e-7:
                return False
        return True


class AuthoredTrajectory:
    """C2 interpolation through cached samples, with derivatives in authored time."""

    def __init__(self, positions: np.ndarray, fps: float, *, velocities=None, accelerations=None):
        q = np.asarray(positions, dtype=float)
        if q.ndim != 2 or not len(q) or not np.all(np.isfinite(q)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Authored samples must be finite, nonempty and have positive fps.")
        self.positions = q.copy()
        self.fps = float(fps)
        self.duration = max(len(q) - 1, 1) / self.fps
        if (velocities is None) != (accelerations is None):
            raise ValueError("Provide both authored velocities and accelerations.")
        if velocities is not None:
            self.velocities = np.asarray(velocities, dtype=float).copy()
            self.accelerations = np.asarray(accelerations, dtype=float).copy()
            if any(x.shape != q.shape or not np.all(np.isfinite(x))
                   for x in (self.velocities, self.accelerations)):
                raise ValueError("Authored derivatives must be finite and match the position array.")
        elif len(q) == 1:
            self.velocities = np.zeros_like(q)
            self.accelerations = np.zeros_like(q)
        else:
            edge = 2 if len(q) > 2 else 1
            self.velocities = np.gradient(q, 1/self.fps, axis=0, edge_order=edge)
            self.accelerations = np.gradient(self.velocities, 1/self.fps, axis=0, edge_order=edge)
        self.segments = coefficients(
            JointState(q[:-1], self.velocities[:-1], self.accelerations[:-1]),
            JointState(q[1:], self.velocities[1:], self.accelerations[1:]), 1/self.fps,
        )

    def state(self, index: int) -> JointState:
        return JointState(self.positions[index], self.velocities[index], self.accelerations[index])

    def at_time(self, seconds: float) -> JointState:
        if len(self.positions) == 1:
            return self.state(0)
        frame = float(np.clip(seconds * self.fps, 0, len(self.positions)-1))
        index = min(int(frame), len(self.positions)-2)
        return HermiteBridge(1/self.fps, self.segments[:, index]).at_time((frame-index)/self.fps)


def load_jerk_limits(path, names, default=None) -> dict[str, float]:
    """Read the existing dynamics JSON without changing v1's shared limit type."""
    def positive(value):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Joint jerk limits must be finite and positive (rad/s^3).")
        return value
    fallback = math.inf if default is None else positive(default)
    overrides = {}
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "max_jerk_rad_s3" in payload.get("default", {}):
            fallback = positive(payload["default"]["max_jerk_rad_s3"])
        for name, values in payload.get("joints", {}).items():
            if name not in names:
                raise ValueError(f"Unknown joint in jerk limits: {name}")
            if "max_jerk_rad_s3" in values:
                overrides[name] = positive(values["max_jerk_rad_s3"])
    return {name: overrides.get(name, fallback) for name in names}


def require_ruckig():
    try:
        return importlib.import_module("ruckig")
    except ImportError as exc:
        raise ValueError("Ruckig backend requires the optional dependency: pip install ruckig") from exc


def _within(lower, upper, bounds):
    return bool(np.all(lower >= -bounds - 1e-7) and np.all(upper <= bounds + 1e-7))


@dataclass
class RuckigBridge:
    trajectory: object
    duration: float

    def at_time(self, seconds: float) -> JointState:
        values = self.trajectory.at_time(float(np.clip(seconds, 0, self.duration)))
        return JointState(*(np.asarray(v, dtype=float) for v in values[:3]))


def make_bridge(backend, start, end, lower, upper, speed, acceleration, jerk,
                minimum=0.35, maximum=10.0, *, deadline=None, cancel=None) -> Bridge:
    def check_deadline():
        if ((deadline is not None and time.perf_counter() >= deadline)
                or (cancel is not None and cancel.is_set())):
            raise InfeasibleBridge("Transition preparation deadline expired or cancelled")
    check_deadline()
    arrays = [np.asarray(x, dtype=float) for x in (lower, upper, speed, acceleration, jerk)]
    lower, upper, speed, acceleration, jerk = arrays
    for state in (start, end):
        if any(not np.all(np.isfinite(v)) for v in (state.position, state.velocity, state.acceleration)):
            raise InfeasibleBridge("Non-finite boundary state")
        if (np.any(state.position < lower-1e-8) or np.any(state.position > upper+1e-8)
                or not _within(state.velocity, state.velocity, speed)
                or not _within(state.acceleration, state.acceleration, acceleration)):
            raise InfeasibleBridge("Boundary state exceeds joint limits")
    delta = np.abs(end.position-start.position)
    initial = max(minimum, float(np.max(1.875*delta/speed)),
                  float(np.max(np.sqrt((10/math.sqrt(3))*delta/acceleration))))
    if backend == "ruckig":
        if not np.all(np.isfinite(jerk)):
            raise ValueError("Ruckig requires max_jerk_rad_s3 for every joint.")
        r = require_ruckig()
        inp = r.InputParameter(len(delta))
        for prefix, state in (("current", start), ("target", end)):
            for field in ("position", "velocity", "acceleration"):
                setattr(inp, f"{prefix}_{field}", getattr(state, field).tolist())
        inp.max_velocity, inp.max_acceleration, inp.max_jerk = speed.tolist(), acceleration.tolist(), jerk.tolist()
        inp.minimum_duration = minimum
        inp.synchronization = r.Synchronization.Time
        generator, trajectory = r.Ruckig(len(delta)), r.Trajectory(len(delta))
        try:
            valid = generator.validate_input(inp, True, True)
            result = generator.calculate(inp, trajectory) if valid else -100
        except Exception as exc:
            raise InfeasibleBridge(f"Ruckig rejected boundary states: {exc}") from exc
        if int(result) < 0 or trajectory.duration > maximum:
            raise InfeasibleBridge(f"Ruckig failed or exceeded {maximum}s: {result}")
        extrema = trajectory.position_extrema
        if any(e.min < lo-1e-7 or e.max > hi+1e-7 for e, lo, hi in zip(extrema, lower, upper)):
            raise InfeasibleBridge("Ruckig trajectory exceeds joint position limits")
        return RuckigBridge(trajectory, trajectory.duration)
    if backend != "hermite":
        raise ValueError(f"Unknown state bridge backend: {backend}")
    if minimum > maximum:
        raise InfeasibleBridge("Initial duration exceeds search limit")
    coarse = np.unique(np.r_[initial * 1.2 ** np.arange(40), maximum])
    coarse = coarse[(coarse >= minimum) & (coarse <= maximum)]
    # Preserve the fast path. On failure search below the displacement estimate
    # and between coarse samples; that estimate is not a bound for nonzero v/a.
    fine = np.geomspace(minimum, maximum, max(2, int(math.ceil(math.log(maximum/minimum)/math.log(1.02)))+1))
    for duration in np.r_[coarse, fine[~np.isin(fine, coarse)]]:
        check_deadline()
        bridge = HermiteBridge.between(start, end, float(duration))
        for k, lo, hi in ((0, lower, upper), (1, -speed, speed),
                          (2, -acceleration, acceleration), (3, -jerk, jerk)):
            if bridge.within_limits(k, lo, hi):
                continue
            break
        else:
            return bridge
    # Only compute detailed extrema once, after the entire search fails.
    lows, highs = bridge.extrema(k)
    indices = np.flatnonzero((lows < lo-1e-7) | (highs > hi+1e-7))
    joint = int(indices[0]) if len(indices) else int(np.argmax(np.maximum(lo-lows, highs-hi)))
    last_violation = (f"duration={duration:.6g}s joint_index={joint} "
                      f"constraint={('position', 'speed', 'acceleration', 'jerk')[k]} "
                      f"range=[{lows[joint]:.6g},{highs[joint]:.6g}] "
                      f"limits=[{lo[joint]:.6g},{hi[joint]:.6g}]")
    raise InfeasibleBridge(f"No feasible Hermite duration in the search interval; {last_violation}")


def boundary_weight(seconds, entry, end, window):
    window = min(window, max(0., end-entry)/2)
    if window <= 0:
        return 0.0
    return min(smoothstep((seconds-entry)/window), smoothstep((end-seconds)/window))


class AuthoredClock:
    """Continuous time map, with authored rate at both joins and no phase jumps."""
    def __init__(self, duration, entry, window, now):
        self.duration, self.entry, self.window = duration, entry, window
        self.stop = duration
        self.seconds, self.last_wall = entry, now

    @property
    def weight(self):
        return boundary_weight(self.seconds, self.entry, self.stop, self.window)

    def advance(self, now, music_rate):
        remaining_wall = max(0., now-self.last_wall)
        self.last_wall = now
        # Small bounded integration steps make the rate fade insensitive to a
        # delayed control sample. Split at the exact end and return wall overshoot.
        while remaining_wall > 1e-12 and self.seconds < self.stop:
            dt = min(remaining_wall, 1/240)
            rate = 1 + self.weight * (max(float(music_rate), 1e-6)-1)
            needed = (self.stop-self.seconds)/rate
            if needed <= dt:
                self.seconds = self.stop
                remaining_wall -= needed
                break
            self.seconds += dt*rate
            remaining_wall -= dt
        return remaining_wall
