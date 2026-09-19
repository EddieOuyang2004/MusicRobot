# GMR collision holds: diagnosis and tested solution

The user's understanding is substantially correct for this repository: the final
collision filter can freeze all 29 joints. It stops a proposed unsafe movement
before accepting it; this does not mean a physical collision has already happened.
The floating root can continue moving during a joint hold.

## Why it happens

`realtime/humanoid_robot/src/gmr_retarget_smpl_headless.py`, function
`limit_qpos_velocity_collision_aware`, computes a speed-limited next pose, then
checks interpolated self-collision clearance. If the step is unsafe, it finds one
scalar `safe_amount` and applies it to the entire joint vector:

```python
candidate[7:] = previous_joints + safe_amount * joint_delta
```

One blocked wrist can therefore suppress otherwise feasible leg and arm motion.
At the clearance boundary, repeated straight-line attempts can give
`safe_amount == 0`. The algorithm shortens a direction; it cannot choose a new
direction around the obstruction. This is the repository's final filter, not
an inherent requirement of GMR or collision avoidance.

Speed limits alone also permit sudden velocity reversals and stops. The existing
pipeline does not impose export-time acceleration or jerk bounds.

## Tested alternative

`experiment.py` solves a small quadratic program for the 29 joint displacements
at each source frame. Let `s` be the proposed displacement, `r` the speed-limited
step toward the current raw GMR target, and `s_previous` the last accepted step.
The objective is:

```text
minimize  0.5 * ||s - r||^2 + 0.5 * lambda * ||s - s_previous||^2
subject to joint ranges, per-frame speed limits,
           grad(distance_i) * s >= gain * (planning_clearance - distance_i)
```

Collision constraints restrict approach along the collision normal, allowing
feasible tangential motion and movement of unrelated joints. Changes of joint
velocity are penalized inside the constrained solve. This is consistent with the
normal-motion constraint approach documented by
[Mink](https://kevinzakka.github.io/mink/api/limits.html#collision-avoidance-limits).

The tested settings are lambda=2, collision gain=0.5 and planning clearance=8 mm.
The extra planning space proved important: using 6 mm for both planning and the
final segment check reduced holds to 40, but could still become stuck. At 8 mm,
the tested clip required no backtracking and had no solver failures.

Every proposed segment still passes the existing nonlinear check at 40 samples
with 6 mm clearance. An unsafe segment is backtracked as a fallback. Both the GMR
model and the final playback model use the full configured self-collision pairs.
Frame zero uses the existing neutral-anchor recovery.

## Measured result

Clip: `gWA_sBM_cAll_d26_mWA0_ch07`, 720 frames, 60 Hz, 12 seconds. Both methods
receive the **same captured raw IK trajectory**. Dataset files were not replaced.

| Measurement | Existing filter | Prototype |
| --- | ---: | ---: |
| All-joint hold intervals | 234 | 0 |
| Joint target RMSE, radians | 0.29363 | 0.07273 |
| RMS acceleration, rad/s² | 81.03 | 44.74 |
| RMS jerk, rad/s³ | 7186.74 | 1907.19 |
| Peak joint speed, rad/s | 9.42478 | 9.42478 |
| Peak joint acceleration, rad/s² | 1130.97 | 562.80 |
| Sampled 5 mm clearance violations | 0 | 0 |

Target RMSE is relative to the raw GMR **joint targets**, not human task-space
error. Derivatives are finite differences at 60 Hz. A hold means all 29 joint
speeds are below 1e-6 rad/s; root motion is excluded. The older diagnostic had
236 confirmed holds; the matched-input fresh baseline here has 234, so the older
count was not used in this comparison.

Independent validation used 81 samples per interval, on a different sampling grid
from the filter, for both models. Three synthetic regression tests also passed:
the contact gradient matches finite differences, an unrelated joint can move when
another is blocked, and a blocked joint can move away from contact. Playback-model
joint limits were checked before rendering. An additional check under the playback environment's MuJoCo 3.10.0 found zero 5 mm clearance violations at 42 samples per interval (including endpoints); see `playback_environment_validation.json`.

## Recommended integration

Replace the final scalar-only limiter with a source-rate constrained projection,
keeping nonlinear validation and a conservative fallback. Keep source timestamps
fixed for music synchronization. Regenerate from raw IK or source SMPL: smoothing
already-frozen files cannot recover the suppressed target movement.

For a production implementation, add explicit acceleration/jerk requirements and
a short optimization window that can anticipate collisions and braking. A local
QP cannot guarantee escape from every local minimum. If the desired pose is
geometrically impossible, approach a feasible pose and accept tracking error; a
hold is still appropriate when no safe useful direction exists. Soft smoothing
here reduces acceleration but does not guarantee an acceleration or jerk bound.

Validate the full dataset and the actual runtime interpolation/time-warping path
before changing the default. This experiment validates linearly interpolated
joint paths, not the runtime's quintic curves. It does not address floating-root
smoothness, foot contact, balance or dynamic controller feasibility. Sampled
collision checks are not a proof of continuous collision freedom.

An additional dependency issue deserves a separate fix: the installed Mink
`collision_avoidance_limit.py` divides its positive separation allowance by `dt`,
while `solve_ik.py` builds the QP in displacement coordinates. At this model's
0.002 s timestep, a local check found identical constraint matrices but positive
allowances 500 times those at dt=1. Correct displacement constraints use
`gain * (distance - clearance)`; velocity constraints divide that by dt.
The prototype explicitly uses displacement units. Pin/test a corrected dependency
or provide a local adapter before relying on upstream IK collision constraints.

## Artifacts and reproduction

- `comparison.mp4`: matched-timestamp side-by-side kinematic preview.
- `projected_motion.pkl`: experimental result, deliberately marked with an
  experimental pipeline version so it cannot masquerade as a production cache.
- `projected_w2_m8_report.json`: complete numeric comparison and validation.
- `raw.npz`, `baseline.npz`, `projected_w2_m8.npz`: reproducible numeric inputs/results.
- `provenance.json`: SHA-256 hashes of the inputs/results and video.

From the repository root:

```powershell
& realtime/humanoid_robot/.venv-gmr/Scripts/python.exe output/gmr_collision_solution/test_projection.py
& realtime/humanoid_robot/.venv-gmr/Scripts/python.exe output/gmr_collision_solution/experiment.py
& .venv/Scripts/python.exe output/gmr_collision_solution/preview.py
```

The experiment is intentionally separate from the production pipeline. Its cached
raw/baseline files belong to this clip and this local dependency setup; remove or
use a fresh output directory before testing a different source or dependency.
