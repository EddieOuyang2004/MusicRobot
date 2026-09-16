# V2 entry window — 2026-09-16

Entry selection now searches the first 20% of each candidate motion, including
the 20% boundary when it falls on a frame. Frame phase is i / (N - 1).
This replaces the eight-second minimum remaining requirement and preserves at
least 80% of the target's authored duration at entry, regardless of playback
speed. Existing source-exit trimming can still end that playback early.

The CLI option is now `--entry-max-phase` (default 0.2, accepted range 0–1),
replacing `--entry-min-remaining-seconds`. Both entry scoring and the bridge
worker apply the same window. Replay still enters at frame zero. Joint bounds,
bridge feasibility checks, and source-exit behavior are unchanged.

## Validation

- 24 matcher tests and 38 bridge tests passed (62 total).
- Tests cover the inclusive boundary, exclusion of later better-scoring frames,
  duration/speed independence, short-clip eligibility, CLI validation, and
  exhaustive exit/entry ranking. For real House clip lengths, 426-frame clips
  permit frames 0–85 and 480-frame clips permit frames 0–95.
- A 60-second headless causal Chronos run with initial seed 42 completed seven
  transitions, with zero replay fallbacks and zero terminal holds. The full
  trace verifier passed using the new entry-phase constraint. Observed entry
  phases ranged from 0.068833652 to 0.197496523.
- The previously failing PO0 source transitioned to
  `gHO_sBM_cAll_d20_mHO5_ch04` at entry frame 80 (phase 0.188235294).
- Artifacts: `tmp/v2_entry20_20260916.{log,csv,json}` and
  `tmp/v2_entry20_20260916_poses.npz`.

This single run verifies the requested entry policy and recovery of the tested
sequence; it does not establish indefinite continuous playback.
