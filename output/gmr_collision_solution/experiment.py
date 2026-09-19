"""Offline comparison of scalar collision backtracking and joint-space projection.

Run with realtime/humanoid_robot/.venv-gmr/Scripts/python.exe. Production
retargeting and dataset files are not modified. This is a kinematic experiment.
"""
from pathlib import Path
import argparse
import json
import pickle
import sys
import time

import mujoco
import numpy as np
from qpsolvers import solve_qp

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "realtime/humanoid_robot"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
import gmr_retarget_smpl_headless as runner
from gmr_retarget_bvh_headless import limit_qpos_velocity

MOTION = "gWA_sBM_cAll_d26_mWA0_ch07"


def states_for(models):
    states = []
    for model in models:
        pairs = {tuple(sorted((a, b)))
                 for first, second in runner.collision_geom_pairs(model)
                 for a in first for b in second}
        states.append((model, mujoco.MjData(model), pairs,
                       np.asarray(sorted(pairs), dtype=np.int32)))
    return tuple(states)


def constraints(states, previous, clearance, detection=0.08, gain=0.5):
    """Linearized signed-distance constraints on joint displacement (not velocity).

    Use the exact same configured pair set as the nonlinear validation filter.
    All pairs are self-collisions; the common free-root transform cancels out.
    """
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
            if norm < 1e-12:
                raise ValueError("Undefined contact normal at an unsafe starting pose")
            normal /= norm
            mujoco.mj_jac(model, data, jac_a, None, fromto[:3], model.geom_bodyid[ga])
            mujoco.mj_jac(model, data, jac_b, None, fromto[3:], model.geom_bodyid[gb])
            gradient = (normal @ (jac_b - jac_a))[6:]
            rows.append(-gradient)
            bounds.append(gain * (distance - clearance))
    return (np.asarray(rows), np.asarray(bounds)) if rows else (None, None)


def project(models, raw, fps, weight, avoidance_clearance=.008):
    states = states_for(models)
    result = raw.copy()
    # Reuse the established neutral-anchor recovery for the first pose only.
    result[:1], _ = runner.limit_qpos_velocity_collision_aware(
        models, raw[:1], fps, max_joint_speed=3*np.pi, max_root_speed=3.,
        max_root_angular_speed=4*np.pi, minimum_collision_distance=0.006,
        stabilization_passes=1)
    n = raw.shape[1] - 7
    low, high = np.full(n, -np.inf), np.full(n, np.inf)
    for model in models:
        for jid in range(model.njnt):
            address = int(model.jnt_qposadr[jid]) - 7
            if address >= 0 and model.jnt_limited[jid]:
                low[address] = max(low[address], model.jnt_range[jid, 0])
                high[address] = min(high[address], model.jnt_range[jid, 1])
    previous_step = np.zeros(n)
    stats = dict(backtracked_frames=0, solver_failures=0)
    for frame in range(1, len(raw)):
        previous = result[frame-1]
        limited, _ = limit_qpos_velocity(
            np.stack((previous, raw[frame])), fps, max_joint_speed=3*np.pi,
            max_root_speed=3., max_root_angular_speed=4*np.pi)
        target_step = limited[1, 7:] - previous[7:]
        G, h = constraints(states, previous, avoidance_clearance)
        step = solve_qp(
            (1 + weight) * np.eye(n), -(target_step + weight * previous_step),
            G, h, lb=np.maximum(-3*np.pi/fps, low-previous[7:]),
            ub=np.minimum(3*np.pi/fps, high-previous[7:]), solver="daqp")
        if step is None or not np.all(np.isfinite(step)):
            stats["solver_failures"] += 1
            step = np.zeros(n)
        candidate = limited[1].copy()
        candidate[7:] = previous[7:] + step
        if runner._segment_violates_configured_clearance(states, previous, candidate, 0.006):
            stats["backtracked_frames"] += 1
            safe, unsafe = 0., 1.
            for _ in range(14):
                amount = (safe + unsafe)/2
                candidate[7:] = previous[7:] + amount*step
                if runner._segment_violates_configured_clearance(states, previous, candidate, 0.006):
                    unsafe = amount
                else:
                    safe = amount
            candidate[7:] = previous[7:] + safe*step
        result[frame] = candidate
        previous_step = candidate[7:] - previous[7:]
        if frame % 120 == 0:
            print(f"projection weight={weight:g}: {frame}/{len(raw)}", flush=True)
    return result, stats


def metrics(qpos, raw, fps):
    v = np.diff(qpos[:, 7:], axis=0)*fps
    a = np.diff(v, axis=0)*fps
    j = np.diff(a, axis=0)*fps
    return dict(all_joint_hold_intervals=int(np.sum(np.max(np.abs(v), axis=1)<1e-6)),
                joint_target_rmse_rad=float(np.sqrt(np.mean((qpos[:, 7:]-raw[:, 7:])**2))),
                peak_speed_rad_s=float(np.max(np.abs(v))),
                acceleration_rms_rad_s2=float(np.sqrt(np.mean(a*a))),
                peak_acceleration_rad_s2=float(np.max(np.abs(a))),
                jerk_rms_rad_s3=float(np.sqrt(np.mean(j*j))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, default=2.)
    parser.add_argument('--avoidance-clearance', type=float, default=.008)
    args = parser.parse_args()
    cached = OUT / "raw.npz"
    with (BASE / "data/aistpp_gmr" / f"{MOTION}.pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    if not cached.exists():
        original_filter = runner.limit_qpos_velocity_collision_aware
        def capture(models, qpos, fps, **kwargs):
            np.savez(cached, qpos=qpos, fps=fps)
            print("Captured raw IK trajectory", flush=True)
            return qpos, {}
        runner.limit_qpos_velocity_collision_aware = capture
        settings = argparse.Namespace(
            gmr_root=BASE / ".deps/GMR", motion=BASE / "data/aistpp/motions" / f"{MOTION}.pkl",
            smpl_model_path=BASE / "assets/body_models/smpl", solver="daqp", velocity_limit=True,
            collision_avoidance=True, collision_min_distance=.005,
            collision_detection_distance=.08, collision_gain=.85,
            collision_validation_model=runner.DEFAULT_COLLISION_VALIDATION_MODEL,
            motion_fps=float(metadata["fps"]), max_joint_speed=3*np.pi,
            max_root_speed=3., max_root_angular_speed=4*np.pi,
            source_motion_id=MOTION, retargeter_version=metadata["retargeter_version"],
            source_sha256=metadata["source_sha256"], smpl_model_sha256=metadata["smpl_model_sha256"])
        runner.retarget(settings)
        runner.limit_qpos_velocity_collision_aware = original_filter
    source = np.load(cached)
    raw, fps = source["qpos"], float(source["fps"])
    models = (mujoco.MjModel.from_xml_path(str(BASE / ".deps/GMR/assets/unitree_g1/g1_mocap_29dof.xml")),
              mujoco.MjModel.from_xml_path(str(runner.DEFAULT_COLLISION_VALIDATION_MODEL)))
    runner.activate_required_collision_geoms(models[0])
    baseline_path = OUT / "baseline.npz"
    if not baseline_path.exists():
        print("Computing current filter baseline from identical raw input", flush=True)
        baseline, counts = runner.limit_qpos_velocity_collision_aware(
            models, raw, fps, max_joint_speed=3*np.pi, max_root_speed=3.,
            max_root_angular_speed=4*np.pi, minimum_collision_distance=.006)
        np.savez(baseline_path, qpos=baseline)
        (OUT / "baseline_counts.json").write_text(json.dumps(counts, indent=2))
    baseline = np.load(baseline_path)["qpos"]
    start = time.perf_counter()
    projected, stats = project(models, raw, fps, args.weight, args.avoidance_clearance)
    stats["elapsed_seconds"] = time.perf_counter()-start
    tag = f"projected_w{args.weight:g}_m{args.avoidance_clearance*1000:g}"
    np.savez(OUT / f"{tag}.npz", qpos=projected, fps=fps)
    # Independent denser and offset interpolation validation, both exact models.
    print("Validating both trajectories at 81 interior/end samples per interval", flush=True)
    validation = {}
    for name, trajectory in (("baseline", baseline), (tag, projected)):
        states = states_for(models)
        failures = 0
        for frame in range(1, len(trajectory)):
            for amount in np.linspace(0., 1., 82)[1:]:
                probe = trajectory[frame].copy()
                probe[7:] = trajectory[frame-1, 7:] + amount*(trajectory[frame, 7:]-trajectory[frame-1, 7:])
                if runner._violates_configured_clearance(states, probe, .005):
                    failures += 1
        validation[name] = dict(samples_per_interval=81, clearance_m=.005, violating_samples=failures)
    report = dict(motion=MOTION, fps=fps, frames=len(raw), smoothness_weight=args.weight,
                  avoidance_clearance_m=args.avoidance_clearance, baseline=metrics(baseline, raw, fps), projected=metrics(projected, raw, fps),
                  projection_stats=stats, validation=validation,
                  caveat="Sampled kinematic self-collision validation, not continuous collision or dynamic feasibility certification.")
    (OUT / f"{tag}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
