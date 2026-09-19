# Completed state-aware collision trajectory correction

The interrupted batch ended cleanly. Its lock was idle. Breakdance completed, while
standing was rejected at frame 630 due to an infeasible continuation. The available
logs do not establish a power outage; the earlier tool interruption was an approval
service usage-limit error.

## Changes

- Added gmr_state_trajectory.py: receding-horizon quintic planning with accepted
  position, velocity and acceleration fixed at each new segment start. Future
  guide states contribute position, velocity and acceleration tracking objectives.
- Plans up to 12 source frames ahead and tries shorter horizons when needed.
  Both joint-bound and self-collision constraints include a short terminal
  continuation to avoid approaching an unrecoverable state at the horizon end.
- Bernstein control bounds constrain position, speed and acceleration over each
  entire polynomial. Exact extrema and dense nonlinear geometry checks validate
  the resulting curves. Physical-time jerk is penalized, not hard bounded.
- If replanning fails, the already checked curve can continue for its remaining
  duration. Otherwise the clip fails generation; no sudden zero-velocity fallback
  is inserted in the output trajectory.
- The old displacement filter now supplies a guide only. Published poses and
  derivatives come from the state planner, and final collision checks evaluate
  that actual Hermite representation.
- Motion files store float64 dof_pos/dof_vel/dof_acc, validated interpolation
  metadata and a checksum binding the state arrays to FPS. Sampler, matcher
  entry/exit features, Hermite transitions and smoothness audit consume those
  derivatives directly. They do not estimate new derivatives from sampled poses.
- Projection revision 2 invalidates older linear-only v2 artifacts. The original
  pipeline-v4 files still load. Rerun build_gmr_v2.py --jobs 2 --resume to rebuild
  older v2 files; backups are retained and originals are untouched.

## Validated pilot results

| Metric | Standing | Breakdance |
| --- | ---: | ---: |
| Continuous peak speed | 9.42478 rad/s | 9.42478 rad/s |
| Continuous peak acceleration | 267.06 rad/s² | 412.08 rad/s² |
| Joint range violations | 0 | 0 |
| Sampled 5 mm clearance violations, both generation models | 0 | 0 |
| Playback MuJoCo 3.10 clearance violations | 0 | 0 |
| All-joint hold intervals | 0 | 0 |
| Planning fallbacks | 0 | 0 |
| Initial derivative scale | 1 | 0 |

Generation validates 81 samples per source interval on both models (MuJoCo 3.12).
An independent playback check uses 42 samples per interval under MuJoCo 3.10.
Sampler evaluation matches the saved curve to approximately 2e-14 rad. Matcher
velocity and acceleration arrays exactly match the saved arrays after name mapping.

The breakdance clip's initial derivative estimate was infeasible, so it starts
with zero velocity/acceleration at frame zero. This is an explicit initialization
choice when no incoming state exists; subsequent collision replans preserve their
incoming states. Tests also verify that a supplied nonzero initial state is kept,
and an infeasible supplied state is rejected rather than silently reset.

Smoothing has a tracking tradeoff. Joint target RMSE is 0.0823 rad for standing and
0.2361 rad for breakdance, compared with 0.0727 and 0.1900 for the earlier positional
projection. The corrected curve satisfies the speed and position limits that the
previous post-fit Hermite curves exceeded.

## Scope

Two pilot clips have been regenerated, not the entire corrected 411-clip dataset.
The fixes cover authored within-clip joint trajectories. Dense collision sampling
is not a continuous collision proof. Separate inter-clip bridges, loop seams,
time warping, downstream limiters, root dynamics and physical balance are not
certified by these checks. The acceleration limit is the existing 1600 rad/s²
software limit; jerk has a soft cost only.

Tests cover stored state validation, C2 joins, nonzero boundaries, infeasible state
rejection, revision/cache invalidation, checksum tampering, FPS overrides and
sampler/matcher mapping, plus existing transition/playback/catalog/batch suites.
Numeric pilot details are in fixed_playback_report.json. Reproduction commands and
migration guidance are in docs/gmr_v2_batch.md.
