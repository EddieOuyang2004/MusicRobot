# V2 nonlinear entry scoring

State-bridge selection compares reachable authored frames in the current motion's
final two seconds with eligible frames of shortlisted motions. Set
`--transition-exit-window-seconds 0` for terminal-only selection. The legacy
quintic path remains terminal-only. Each nonnegative difference is divided by a
fixed soft tolerance, then scored as:

```text
r = difference / tolerance
penalty = r² + excess_weight × max(r − 1, 0)²
```

The default excess weight is 4. Ratios 0, 0.5, 1, and 2 score 0, 0.25, 1,
and 8. There is no dead zone, clipping, or rejection at the tolerance. These
are initial tuning defaults, not empirically calibrated thresholds.

| CLI option | Default | Meaning |
| --- | ---: | --- |
| `--entry-pose-tolerance-fraction` | 0.05 | Fraction of each joint's range |
| `--entry-velocity-tolerance-fraction` | 0.10 | Fraction of each joint's speed limit |
| `--entry-contact-tolerance-feet` | 1 | Number of mismatched feet |
| `--entry-root-height-tolerance-m` | 0.025 | Root height difference in metres |
| `--entry-root-tilt-tolerance-rad` | 0.075 | Root tilt difference in radians |
| `--entry-root-linear-velocity-tolerance-m-s` | 0.25 | Norm of root velocity difference in m/s |
| `--entry-root-angular-speed-tolerance-rad-s` | 0.5 | Root angular speed difference in rad/s |
| `--entry-excess-penalty-weight` | 4 | Additional squared penalty above tolerance |

Tolerances must be finite and positive; excess weight must be finite and
nonnegative. Smaller tolerances increase sensitivity, including below the
threshold. Normalization does not depend on the other candidate motions.
Existing joint-range fallbacks and root-feature definitions are retained.

Joint pose and velocity penalties are calculated per joint before averaging.
Root height, tilt, linear velocity, and angular speed are penalized separately
before averaging. Contact uses the mismatch count directly: zero, one, and
two mismatched feet score 0, 1, and 8 with default settings.

```text
total = (0.45 × pose + 0.20 × velocity + 0.20 × contact + 0.10 × root) / 0.95
```

`MotionEntryScore` and the existing CSV `entry_pose_score`,
`entry_velocity_score`, `entry_contact_score`, `entry_root_score`, and
`entry_total_score` fields now contain unbounded nonlinear penalties. Lower
is better; their magnitudes are not comparable with earlier linear scores.
`entry_music_score` remains zero. Music shortlisting and tie-breaking,
remaining-duration eligibility, and bridge feasibility checks are unchanged.
All v2 scoring paths, including replay, use the same configuration.
Pair search adds no new scoring term or weight. Acceleration remains part of
bridge feasibility validation, not the nonlinear score. Equal pair costs use
music relevance, later exit, earlier entry, and motion ID as tie-breakers.
Replay compares eligible exits against frame zero of the current motion.

V2 defaults to 60 Hz control; `--control-rate-hz 120` overrides it. V1 and the
standalone dancer retain their 120 Hz default. Pair search runs once per
preparation in the background, not once per control tick.

An early exit must be committed before its boundary fade begins, before the
original terminal fade begins, and leave room for both entry and exit fades.
Playback keeps the original authored phase denominator and stops at the selected
frame. Preparation checks its conservative remaining-time deadline between score
rows, bridge attempts, and Hermite duration trials; native solver calls are not
preemptible. If no feasible bridge is ready, the existing terminal hold remains.

CSV additions: `exit_frame_index`, `exit_seconds`, `exit_phase`, `trimmed_seconds`,
`transition_pair_count`, `bridge_attempts`, and `transition_preparation_seconds`.
Pair counts describe ranked pairs after boundary-state and remaining-duration
filtering. The phase/exit fields identify the prepared bridge.

Validation commands:

```powershell
.venv/Scripts/python.exe -m unittest discover -s realtime/humanoid_robot/src/test -p test_humanoid_matcher_v2.py
.venv/Scripts/python.exe -m unittest discover -s realtime/humanoid_robot/src/test -p test_motion_bridges.py
```
