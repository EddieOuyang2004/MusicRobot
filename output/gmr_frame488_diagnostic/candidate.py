"""Receding-horizon, state-to-state quintic planning on a collision-free guide.

The QP optimizes terminal q, T*v and T^2*a. Initial q/v/a are fixed. Bernstein
control bounds constrain the whole polynomial; nonlinear collision checks use
the same polynomial later serialized as Hermite knot states.
"""
from __future__ import annotations

import math
import numpy as np
from numpy.polynomial import polynomial as poly

from motion_bridges import AuthoredTrajectory, HermiteBridge, JointState, coefficients


def bernstein_matrix(degree):
    return np.array([[math.comb(i, k)/math.comb(degree, k) if k <= i else 0.
                      for k in range(degree+1)] for i in range(degree+1)])


# Columns are terminal position, duration*velocity, duration^2*acceleration.
END_MAP = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                    [10., -4., .5], [-15., 7., -1.], [6., -3., .5]])


def subdivided_bernstein_matrix(degree, subdivisions=4):
    """Bound each subinterval independently without loosening physical limits."""
    blocks = []
    for segment in range(subdivisions):
        origin = segment / subdivisions
        width = 1. / subdivisions
        restriction = np.array([[math.comb(k, j) * origin**(k-j) * width**j
                                 if k >= j else 0. for k in range(degree+1)]
                                for j in range(degree+1)])
        blocks.append(bernstein_matrix(degree) @ restriction)
    return np.vstack(blocks)


def _affine(start, duration):
    zero = np.zeros_like(start.position)
    offset = coefficients(start, JointState(zero, zero, zero), duration)
    return offset, END_MAP


def _matrix_rows(mapping, offset, low, high):
    n = offset.shape[1]
    A = np.kron(mapping, np.eye(n))
    b = offset.ravel()
    return np.vstack((A, -A)), np.r_[np.broadcast_to(high, offset.shape).ravel()-b,
                                    b-np.broadcast_to(low, offset.shape).ravel()]


def _curve_safe(bridge, root, states, clearance, samples):
    from gmr_retarget_smpl_headless import _violates_configured_clearance
    pose = root.copy()
    for u in np.linspace(0., 1., samples+1):
        pose[7:] = bridge.at_time(u*bridge.duration).position
        if _violates_configured_clearance(states, pose, clearance):
            return False
    return True


def solve_horizon(start, references, duration, low, high, states, root, settings, guess=None):
    """Return a checked quintic with fixed initial state, or None if infeasible.

    References are q/v/a at future source timestamps. Their terminal state is a
    tracking objective, rather than an infeasible hard collision constraint.
    """
    from qpsolvers import solve_qp
    from gmr_collision_projection import distance_constraints
    def fail(reason):
        if settings.get("_debug"):
            print("horizon failed:", reason, "duration", duration, flush=True)
        return None
    n = len(start.position)
    offset, mapping = _affine(start, duration)
    H = np.eye(3*n)*1e-8
    f = np.zeros(3*n)
    grid = np.linspace(0., 1., len(references)+1)[1:]
    for u, reference in zip(grid, references):
        for derivative, weight in ((0, 1.), (1, .1), (2, .005)):
            # Derivatives scaled to normalized time make the QP well conditioned.
            a = poly.polyval(u, poly.polyder(mapping, m=derivative))
            b = poly.polyval(u, poly.polyder(offset, m=derivative))
            target = (reference.position, duration*reference.velocity,
                      duration**2*reference.acceleration)[derivative]
            A = np.kron(a.reshape(1, 3), np.eye(n))
            H += weight*A.T@A
            f += weight*A.T@(b-target)
    # Integrated squared jerk is a smoothness cost; no arbitrary jerk bound.
    for u in np.linspace(0., 1., 5):
        a = poly.polyval(u, poly.polyder(mapping, m=3))
        b = poly.polyval(u, poly.polyder(offset, m=3))
        A = np.kron(a.reshape(1, 3), np.eye(n))
        weight = settings["smoothing_weight"]*1e-9/(5*duration**6)
        H += weight*A.T@A
        f += weight*A.T@b
    rows, bounds = [], []
    for derivative, lower, upper in (
            (0, low, high), (1, -settings["max_joint_speed_rad_s"], settings["max_joint_speed_rad_s"]),
            (2, -settings["max_joint_acceleration_rad_s2"], settings["max_joint_acceleration_rad_s2"])):
        transform = subdivided_bernstein_matrix(5-derivative)
        A = transform@poly.polyder(mapping, m=derivative)/duration**derivative
        b = transform@poly.polyder(offset, m=derivative)/duration**derivative
        G, h = _matrix_rows(A, b, lower, upper)
        rows.append(G); bounds.append(h)
    # Leave braking room beyond the horizon instead of arriving at a joint
    # boundary with outward velocity. A short constant-acceleration continuation
    # has affine Bernstein controls and protects the next planning problem.
    reserve = .05
    continuation = np.array([[1., 0., 0.], [1., reserve/(2*duration), 0.],
                             [1., reserve/duration, reserve**2/(2*duration**2)]])
    G, h = _matrix_rows(continuation, np.zeros((3, n)), low, high)
    rows.append(G); bounds.append(h)
    continuation_v = np.array([[0., 1/duration, 0.], [0., 1/duration, reserve/duration**2]])
    G, h = _matrix_rows(continuation_v, np.zeros((2, n)),
                        -settings["max_joint_speed_rad_s"], settings["max_joint_speed_rad_s"])
    rows.append(G); bounds.append(h)
    fixed_G, fixed_h = np.vstack(rows), np.concatenate(bounds)
    # Fixed start state creates constant rows. Detect incompatible boundary states.
    nonzero = np.max(np.abs(fixed_G), axis=1) > 1e-12
    if np.any(fixed_h[~nonzero] < -1e-7):
        return fail("fixed boundary")
    fixed_G, fixed_h = fixed_G[nonzero], fixed_h[nonzero]
    end = references[-1]
    z = np.r_[end.position, duration*end.velocity, duration**2*end.acceleration]
    if guess is not None:
        end = guess.at_time(min(duration, guess.duration))
        z = np.r_[end.position, duration*end.velocity, duration**2*end.acceleration]
    collision_grid = np.linspace(0., 1., max(12, 2*len(references))+1)[1:]
    pose = root.copy()
    for iteration in range(settings["planning_iterations"]):
        rows, bounds = [fixed_G], [fixed_h]
        for u in collision_grid:
            a = poly.polyval(u, mapping)
            b = poly.polyval(u, offset)
            A = np.kron(a.reshape(1, 3), np.eye(n))
            pose[7:] = b + A@z
            # Ramp the buffer from the already-accepted clearance, without
            # demanding an instantaneous separation jump at the initial knot.
            ramp = u*u*(3-2*u)
            margin = settings["segment_clearance_m"] + ramp*(
                settings["planning_clearance_m"]-settings["segment_clearance_m"])
            try:
                G, h = distance_constraints(states, pose, margin,
                                            settings["detection_distance_m"], 1., allow_penetration=True)
            except ValueError:
                # A colliding initial guess needs a usable signed-distance normal.
                return fail("contact normal")
            if G is not None:
                rows.append(G@A)
                bounds.append(h + G@(pose[7:]-b))
        # Also constrain the terminal continuation: a safe terminal pose alone
        # can still have velocity/acceleration directed into the body.
        for future in (reserve/2, reserve):
            terminal_map = np.array([1., future/duration, .5*future**2/duration**2])
            A = np.kron(terminal_map.reshape(1, 3), np.eye(n))
            pose[7:] = A@z
            G, h = distance_constraints(states, pose, settings["planning_clearance_m"],
                                        settings["detection_distance_m"], 1., allow_penetration=True)
            if G is not None:
                rows.append(G@A)
                bounds.append(h + G@pose[7:])
        # DAQP defaults permit residuals larger than the exact curve validator.
        solver_options = {"primal_tol": 1e-9, "dual_tol": 1e-9} if settings["solver"] == "daqp" else {}
        solution = solve_qp(H, f, np.vstack(rows), np.concatenate(bounds), solver=settings["solver"], **solver_options)
        if solution is None or not np.all(np.isfinite(solution)):
            return fail("QP infeasible")
        z = solution
        end = JointState(z[:n], z[n:2*n]/duration, z[2*n:]/duration**2)
        bridge = HermiteBridge.between(start, end, duration)
        # Independent nonlinear checks use a finer grid than the QP linearization.
        if _curve_safe(bridge, root, states, settings["segment_clearance_m"],
                       max(40, 4*len(references))):
            # Bernstein bounds already constrain the entire curve, but verify
            # actual extrema too to catch solver-scale errors before execution.
            checks = ((0, low, high),
                      (1, -settings["max_joint_speed_rad_s"], settings["max_joint_speed_rad_s"]),
                      (2, -settings["max_joint_acceleration_rad_s2"], settings["max_joint_acceleration_rad_s2"]))
            if all(bridge.within_limits(d, np.broadcast_to(lo, (n,)), np.broadcast_to(hi, (n,)))
                   for d, lo, hi in checks):
                return bridge
    return fail("nonlinear check or exact bound")


def plan_state_trajectory(guide, fps, states, low, high, settings, *, initial_state=None):
    """Carry accepted q/v/a between horizons; never insert a zero-velocity hold."""
    n = guide.shape[1]-7
    target = AuthoredTrajectory(guide[:, 7:], fps)
    q, v, a = (np.empty((len(guide), n)) for _ in range(3))
    start = initial_state or target.state(0)
    q[0], v[0], a[0] = start.position, start.velocity, start.acceleration
    dt = 1/fps
    previous_plan = None
    fallbacks = 0
    initial_scale = 1.
    for frame in range(len(guide)-1):
        count = min(settings["lookahead_frames"], len(guide)-1-frame)
        references = [target.state(frame+i) for i in range(1, count+1)]
        start = JointState(q[frame], v[frame], a[frame])
        plan = None
        for horizon in sorted({count, max(1, count//2), max(1, count//4), 1}, reverse=True):
            plan = solve_horizon(start, references[:horizon], horizon*dt, low, high,
                                 states, guide[frame], settings, guess=previous_plan)
            if plan is not None:
                break
        # At clip initialization only, no incoming trajectory exists. Reduce an
        # infeasible estimated initial derivative, recording the adjustment.
        if frame == 0 and initial_state is None:
            for scale in (.5, .25, .125, 0.):
                if plan is not None:
                    break
                initial_scale = scale
                initial = target.state(0)
                start = JointState(initial.position, scale*initial.velocity, scale*initial.acceleration)
                plan = solve_horizon(start, references, count*dt, low, high,
                                     states, guide[frame], settings)
            q[0], v[0], a[0] = start.position, start.velocity, start.acceleration
        if plan is None and previous_plan is not None and previous_plan.duration >= dt-1e-10:
            plan = previous_plan
            fallbacks += 1
        if plan is None:
            raise ValueError(f"No feasible state-continuous collision trajectory at frame {frame}; clip not published.")
        accepted = plan.at_time(dt)
        q[frame+1], v[frame+1], a[frame+1] = accepted.position, accepted.velocity, accepted.acceleration
        remaining = plan.duration-dt
        previous_plan = (HermiteBridge.between(accepted, plan.at_time(plan.duration), remaining)
                         if remaining > 1e-10 else None)
    output = guide.copy()
    output[:, 7:] = q
    curve = AuthoredTrajectory(q, fps, velocities=v, accelerations=a)
    peaks = dict(speed_rad_s=0., acceleration_rad_s2=0., jerk_rad_s3=0.)
    for frame in range(len(q)-1):
        bridge = HermiteBridge(dt, curve.segments[:, frame])
        for derivative, label in ((1, "speed_rad_s"), (2, "acceleration_rad_s2"), (3, "jerk_rad_s3")):
            lower, upper = bridge.extrema(derivative)
            peaks[label] = max(peaks[label], float(np.max(np.maximum(np.abs(lower), np.abs(upper)))))
        if not bridge.within_limits(0, low, high):
            raise ValueError(f"Hermite joint-range violation at frame {frame}.")
        if not _curve_safe(bridge, output[frame], states, settings["validation_clearance_m"],
                           settings["validation_samples"]):
            raise ValueError(f"Hermite collision validation failed at frame {frame}.")
    if (peaks["speed_rad_s"] > settings["max_joint_speed_rad_s"]+1e-6
            or peaks["acceleration_rad_s2"] > settings["max_joint_acceleration_rad_s2"]+1e-5):
        raise ValueError("Hermite trajectory exceeds a continuous derivative limit.")
    return output, v, a, dict(planning_fallbacks=fallbacks, initial_derivative_scale=initial_scale,
                              continuous_peaks=peaks)
