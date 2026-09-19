# Collision trajectory state audit

The user's concern is valid, with one distinction: the GMR v2 collision projection is a per-frame displacement optimizer, not a rest-to-rest polynomial. It retains the last accepted displacement as a velocity proxy and softly penalizes the change in displacement. It does not carry previous acceleration as a state, match terminal velocity/acceleration, or impose acceleration/jerk limits.

At timestep h, its smoothness term is ||s_k - s_(k-1)||^2 = h^4 ||a_k||^2 using backward finite differences. That discourages acceleration, but it is not the six boundary conditions of a state-to-state quintic. The reference step is the gap to the next raw target position, clipped by speed; it is not the desired motion's velocity. previous_step starts at zero once at clip initialization, not at every collision. Solver fallback can choose zero and backtracking can shorten a step abruptly.

The transition implementation in motion_bridges.py is a quintic Hermite polynomial matching position, velocity and acceleration at both ends. The matcher passes these three quantities from AuthoredTrajectory states into make_bridge. Rest-to-rest also uses a polynomial, but assumes zero velocity and acceleration at both boundaries; these assumptions can conflict with adjoining moving clips. Matching actual derivatives explains the improvement. Quintic Hermite assures C2 joins when boundary states agree; it does not by itself bound derivatives or avoid collisions.

## Measured playback interpolation

Audited the two v2 pilots with the actual AuthoredTrajectory quintic and the sampler's float32 input conversion. Derivative and joint-range extrema were calculated from polynomial roots; self-collision clearance was sampled at 81 points per interval on both models using MuJoCo 3.12.0.

| Clip | Saved finite-difference peak speed | Hermite peak speed | Hermite peak acceleration | Joint-range excess | 5 mm clearance violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standing gWA...ch07 | 9.425 rad/s | 11.554 rad/s | 1060.95 rad/s² | 0.01409 rad, waist_pitch | 0 sampled points |
| Breakdance gBR...ch08 | 9.425 rad/s | 12.003 rad/s | 1349.48 rad/s² | 0.01308 rad, waist_roll | 29 sampled points in 2 intervals |

The declared export speed cap is 9.42478 rad/s. Very small finite-difference excess (~4e-6) comes from float32 conversion; the Hermite overshoot is much larger. Breakdance violating intervals begin at zero-based frame indices 22 and 557. The worst checked clearance is 4.80276 mm at 0.37222 seconds, between left_knee_link and left_wrist_yaw_link. This is a positive gap below the required margin, not detected penetration. Sampled collision results are not a continuous-path proof.

The earlier v2 generation validation was for linear joint interpolation. It did not certify the Hermite curve used by the matcher. AuthoredTrajectory estimates derivatives from the filtered samples using numpy.gradient, then constructs Hermite segments. The two paths differ between samples.

## Recommended correction

1. Plan before reaching the clearance boundary, over multiple future frames.
2. Start from the currently accepted/executed q, velocity and acceleration, in the actual playback time scale.
3. Optimize collision-avoiding intermediate states and a feasible rejoin state in the future target motion. Match the target q, velocity and acceleration where feasible; do not force an infeasible raw target state.
4. Connect those states with quintic Hermite segments, with collision, joint-position, speed and acceleration constraints applied to the actual curves. Penalize jerk or bound it when an appropriate limit is configured. Include nonlinear checks between knots and adequate braking clearance; smoothing alone cannot ensure collision avoidance.
5. Use the same trajectory representation in generation validation and playback. Revalidate changes caused by time warping and downstream output limiting. Preserve the music timeline by changing the path/choosing a later rejoin time rather than inserting arbitrary pauses.

If the present velocity cannot be stopped inside the available gap, smooth continuation and collision clearance may be mutually infeasible. Anticipation is essential; Hermite interpolation cannot resolve that by itself.

No generator, player, or motion file was modified during this audit. Diagnostic code and numeric results are in check.py and report.json in this folder. The second focused distance check augmented report.json with the minimum gap in the flagged intervals.
