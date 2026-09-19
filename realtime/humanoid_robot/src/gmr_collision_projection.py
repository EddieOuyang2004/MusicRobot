"""Source-rate collision-constrained projection for the separate GMR v2 dataset.

A displacement QP builds a guide; a state-aware Hermite planner generates the
validated output. This is not a dynamic feasibility certificate.
"""
from __future__ import annotations

import numpy as np

PIPELINE_VERSION = 5  # User-facing dataset name is GMR v2; v4 is the legacy pipeline.
METHOD = "collision_projection_v2"


def projection_settings(weight: float = 2.0, planning_clearance: float = .008) -> dict:
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Smoothing weight must be finite and non-negative.")
    if not np.isfinite(planning_clearance) or not .006 < planning_clearance < .08:
        raise ValueError("Planning clearance must be strictly between .006 and .08 m.")
    return dict(method=METHOD, revision=2, smoothing_weight=float(weight),
                planning_clearance_m=float(planning_clearance), gain=.5,
                detection_distance_m=.08, segment_clearance_m=.006,
                validation_clearance_m=.005, segment_samples=40, validation_samples=81,
                solver="daqp", lookahead_frames=12, planning_iterations=5,
                max_joint_acceleration_rad_s2=1600., max_joint_speed_rad_s=float(3*np.pi),
                max_root_speed_m_s=3., max_root_angular_speed_rad_s=float(4*np.pi))


def state_trajectory_digest(payload: dict) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name in ("dof_pos", "dof_vel", "dof_acc"):
        values = np.ascontiguousarray(payload[name], dtype="<f8")
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    digest.update(np.float64(payload["fps"]).tobytes())
    return digest.hexdigest()


def validate_v2_metadata(payload: dict) -> None:
    settings = payload.get("projection", {})
    check = payload.get("projection_validation", {})
    if (payload.get("motion_version") != "gmr_v2"
            or not isinstance(settings, dict) or settings.get("method") != METHOD
            or settings.get("revision") != 2
            or not isinstance(check, dict) or check.get("passed") is not True
            or check.get("interpolation") != "quintic_hermite_states"
            or check.get("violating_samples") != 0
            or check.get("samples_per_interval", 0) < 81
            or check.get("clearance_m", 0) < .005):
        raise ValueError("GMR v2 requires revision-2 Hermite state validation; regenerate with --resume.")
    q = np.asarray(payload.get("dof_pos"), dtype=float)
    if q.ndim != 2 or q.shape[1] != 29 or len(q) < 2 or not np.all(np.isfinite(q)):
        raise ValueError("GMR v2 requires finite joint positions with shape [N>=2,29].")
    for name in ("dof_vel", "dof_acc"):
        values = np.asarray(payload.get(name), dtype=float)
        if values.shape != q.shape or not np.all(np.isfinite(values)):
            raise ValueError("GMR v2 requires finite stored velocity/acceleration matching its positions.")
    if payload.get("state_trajectory_sha256") != state_trajectory_digest(payload):
        raise ValueError("GMR v2 Hermite state checksum mismatch.")
    peaks = check.get("continuous_peaks", {})
    for measured, limit in (("speed_rad_s", "max_joint_speed_rad_s"),
                            ("acceleration_rad_s2", "max_joint_acceleration_rad_s2")):
        value = float(peaks.get(measured, np.nan))
        bound = float(settings.get(limit, np.nan))
        if not np.isfinite(value) or not np.isfinite(bound) or bound <= 0 or value > bound+1e-5:
            raise ValueError("GMR v2 has missing or failed continuous derivative validation.")


def collision_states(models):
    import mujoco
    from gmr_retarget_smpl_headless import collision_geom_pairs
    states = []
    for model in models:
        pairs = {tuple(sorted((a, b))) for first, second in collision_geom_pairs(model)
                 for a in first for b in second}
        states.append((model, mujoco.MjData(model), pairs,
                       np.asarray(sorted(pairs), dtype=np.int32)))
    return tuple(states)


def distance_constraints(states, previous, clearance, detection=.08, gain=.5, *, allow_penetration=False):
    """Linearized normal displacement limits for the exact validation pair set."""
    import mujoco
    rows, bounds = [], []
    for model, data, _, pairs in states:
        data.qpos[:] = previous
        mujoco.mj_forward(model, data)
        a, b = pairs.T
        broad_distance = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=1)
        broad_distance -= model.geom_rbound[a] + model.geom_rbound[b]
        fromto = np.empty(6)
        jac_a, jac_b = np.empty((3, model.nv)), np.empty((3, model.nv))
        for index in np.flatnonzero(broad_distance < detection):
            ga, gb = pairs[index]
            distance = mujoco.mj_geomDistance(model, data, int(ga), int(gb), detection, fromto)
            if distance >= detection - 1e-12:
                continue
            normal = fromto[3:] - fromto[:3]
            norm = np.linalg.norm(normal)
            if norm < 1e-12 or (distance < 0 and not allow_penetration):
                raise ValueError("Undefined contact normal or penetrating starting pose.")
            normal /= norm
            if distance < 0:
                normal *= -1.0
            mujoco.mj_jac(model, data, jac_a, None, fromto[:3], model.geom_bodyid[ga])
            mujoco.mj_jac(model, data, jac_b, None, fromto[3:], model.geom_bodyid[gb])
            rows.append(-(normal @ (jac_b - jac_a))[6:])
            # QP variable is displacement: do NOT divide this bound by dt.
            bounds.append(gain * (distance - clearance))
    return (np.asarray(rows), np.asarray(bounds)) if rows else (None, None)


def trajectory_metrics(qpos, raw, fps):
    v = np.diff(qpos[:, 7:], axis=0)*fps
    a = np.diff(v, axis=0)*fps
    j = np.diff(a, axis=0)*fps
    rms = lambda x: float(np.sqrt(np.mean(x*x))) if x.size else 0.
    return dict(all_joint_hold_intervals=int(np.sum(np.max(np.abs(v), axis=1)<1e-6)),
                joint_target_rmse_rad=rms(qpos[:, 7:]-raw[:, 7:]),
                peak_speed_rad_s=float(np.max(np.abs(v))),
                acceleration_rms_rad_s2=rms(a), jerk_rms_rad_s3=rms(j))


def project_trajectory(models, qpos, fps, *, settings, max_joint_speed,
                       max_root_speed, max_root_angular_speed,
                       minimum_collision_distance):
    from qpsolvers import solve_qp
    from gmr_retarget_bvh_headless import limit_qpos_velocity, qpos_joint_names
    from gmr_retarget_smpl_headless import (
        limit_qpos_velocity_collision_aware, _segment_violates_configured_clearance,
        _violates_configured_clearance, validate_joint_limits,
    )
    if (not models or qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2
            or not np.all(np.isfinite(qpos)) or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Projection requires finite G1 qpos[N>=2,36], models and positive FPS.")
    speeds = dict(max_joint_speed=max_joint_speed, max_root_speed=max_root_speed,
                  max_root_angular_speed=max_root_angular_speed)
    names = qpos_joint_names(models[0])
    for model in models:
        if model.nq != 36 or model.nv != 35 or qpos_joint_names(model) != names:
            raise ValueError("Projection models must have the same free-root + 29-joint order.")
    states = collision_states(models)
    result = qpos.copy()
    result[:1], initial = limit_qpos_velocity_collision_aware(
        models, qpos[:1], fps, **speeds,
        minimum_collision_distance=minimum_collision_distance, stabilization_passes=1)
    low, high = np.full(29, -np.inf), np.full(29, np.inf)
    for model in models:
        for jid in range(model.njnt):
            address = int(model.jnt_qposadr[jid]) - 7
            if address >= 0 and model.jnt_limited[jid]:
                low[address] = max(low[address], model.jnt_range[jid, 0])
                high[address] = min(high[address], model.jnt_range[jid, 1])
    previous_step = np.zeros(29)
    weight = settings["smoothing_weight"]
    H = (1+weight)*np.eye(29)
    counts = dict(root_position=0, root_rotation=0, joints=0,
                  collision_frames_adjusted=0, solver_failures=0,
                  initial_frame_adjusted=initial["initial_frame_adjusted"])
    for frame in range(1, len(qpos)):
        previous = result[frame-1]
        limited, local = limit_qpos_velocity(np.stack((previous, qpos[frame])), fps, **speeds)
        for key in ("root_position", "root_rotation", "joints"):
            counts[key] += local[key]
        reference = limited[1, 7:] - previous[7:]
        G, h = distance_constraints(states, previous, settings["planning_clearance_m"],
                                    settings["detection_distance_m"], settings["gain"])
        lb = np.maximum(-max_joint_speed/fps, low-previous[7:])
        ub = np.minimum(max_joint_speed/fps, high-previous[7:])
        step = solve_qp(H, -(reference+weight*previous_step), G, h,
                        lb=lb, ub=ub, solver=settings["solver"])
        if step is None or not np.all(np.isfinite(step)):
            counts["solver_failures"] += 1
            step = np.zeros(29)
        # Remove solver-scale box-bound overshoot, then recheck nonlinear geometry.
        step = np.clip(step, lb, ub)
        candidate = limited[1].copy()
        candidate[7:] = previous[7:] + step
        def unsafe(probe):
            return _segment_violates_configured_clearance(
                states, previous, probe, minimum_collision_distance,
                samples=settings["segment_samples"])
        if unsafe(candidate):
            counts["collision_frames_adjusted"] += 1
            safe, blocked = 0., 1.
            for _ in range(14):
                amount = (safe+blocked)/2
                candidate[7:] = previous[7:] + amount*step
                if unsafe(candidate):
                    blocked = amount
                else:
                    safe = amount
            candidate[7:] = previous[7:] + safe*step
        result[frame] = candidate
        previous_step = candidate[7:]-previous[7:]

    # Validation uses a denser, offset grid. Reject, rather than publish, failures.
    for model in models:
        validate_joint_limits(model, result, names)
    clearance = settings["validation_clearance_m"]
    if _violates_configured_clearance(states, result[0], clearance):
        raise ValueError("Projection fails initial-frame collision validation.")
    for frame in range(1, len(result)):
        if _segment_violates_configured_clearance(
                states, result[frame-1], result[frame], clearance,
                samples=settings["validation_samples"]):
            raise ValueError(f"Projection fails final collision validation at frame {frame}.")
    from gmr_state_trajectory import plan_state_trajectory
    result, velocity, acceleration, state_report = plan_state_trajectory(
        result, fps, states, low, high, settings)
    counts["state_planning_fallbacks"] = state_report["planning_fallbacks"]
    metrics = trajectory_metrics(result, qpos, fps)
    if metrics["peak_speed_rad_s"] > max_joint_speed+1e-7:
        raise ValueError("Projection exceeds final joint speed limit.")
    report = dict(passed=True, clearance_m=clearance, violating_samples=0,
                  samples_per_interval=settings["validation_samples"], models=len(models),
                  interpolation="quintic_hermite_states", metrics=metrics, **state_report,
                  _dof_velocity=velocity, _dof_acceleration=acceleration)
    return result, counts, report
