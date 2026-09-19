# Frame 488 planner failure: tested interval-bound fix

Reported clip: gBR_sBM_cAll_d04_mBR3_ch02 (524 frames).
Current installed solver already has DAQP primal_tol=dual_tol=1e-9.

Reproduction: frame 488 rejects the 12-frame horizon through fixed Bernstein boundary controls; 6-, 3-, and 1-frame horizons return QP infeasible. This differs from the previous frame-69 numerical residual.

Candidate: bound the same quintic and its derivatives with Bernstein controls on four equal subintervals. This reduces conservative rejection without relaxing joint, velocity, acceleration, or collision limits. The terminal continuation constraints, saved q/v/a states, and independent exact extrema and sampled collision validation remain unchanged.

Validation:
- Reported clip: all 524 frames pass; zero planning fallbacks; initial derivative scale 1. Speed peak 9.424569891 rad/s; acceleration peak 214.595364 rad/s^2.
- Previous frame-69 clip: all 720 frames pass; zero planning fallbacks; initial derivative scale 1. Speed peak 9.424725718 rad/s; acceleration peak 190.891264 rad/s^2.
- Seven staged regression tests pass, including a feasible turnaround rejected by global controls and exact subinterval polynomial reconstruction.
- Tests used the same saved positional guide and generation geometry setup with both validation models. They exercised full state planning and final validation, not full artifact publication.

The active user batch holds its lock. No production planner changes were applied. apply_interval_fix.py checks the lock, expected source/test hashes, staged hashes, compiles the replacements, saves backups, and installs the tested candidate and regression tests after the batch finishes. It retains the earlier solver precision fix.

Do not delete the v2 folder. After installation, --resume checks the changed implementation fingerprint and rebuilds stale artifacts. The full 411-clip dataset has not been validated with this candidate; the planner may still reject other motions. Collision checks are sampled, not a proof of continuous separation.
