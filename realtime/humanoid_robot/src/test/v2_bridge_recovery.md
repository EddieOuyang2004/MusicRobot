# Expanded motion search and refined Hermite durations

V2 retrieves 20 motions and shortlists up to 10 by default. Existing explicit
`--match-top-motions` and `--shortlist-size` overrides still apply. The normal
music margins remain 0.05 for final score and 0.08 for music score.

After preparation fails, if an accepted match and playback time remain, the
playback controller can make one expanded retry per source-motion generation.
It uses the latest match with margins at least 0.10 and 0.15 respectively and
excludes already attempted motions, allowing the next eligible results into
the fallback pool. It requires at least one new motion and preserves the shortlist size limit and
the existing cancellation, exit-window reachability, and preparation deadline.
Successful plans are not replaced by later retrievals. A second failure still
leads to terminal hold; recovery is not guaranteed.

For each Hermite boundary pair, the solver first runs its original 20% duration
steps. If those fail, it searches the full minimum-to-10-second interval with
geometric spacing no larger than 2%, including durations below the displacement
estimate. The estimate assumes rest-to-rest motion and is not a lower bound
for nonzero endpoint derivatives. The refinement is a bounded discrete search,
not a proof that no feasible continuous duration exists.

The first three rejected bridge attempts per preparation print motion ID, exit
and entry frame, and the rejection reason. Exhausted Hermite searches include
the last rejected duration, joint index, constraint, actual extrema, and limits.
The supplied joint ordering maps the index to the actuator name. These are
bounded diagnostic samples, not an exhaustive per-duration log. The final
rejection remains in the failure message/trace. Detailed extrema are computed
only after a search fails, preserving the fast path.

No joint limits, remaining-playback threshold, or maximum bridge duration are
relaxed. Existing exit-window behavior is retained. Refinement and broader
shortlists can increase background work, but still obey preparation deadlines.

## Validation — 2026-09-16

- 38 bridge tests and 23 matcher tests pass. Coverage includes a feasible bridge
  below the old duration estimate, refinement between coarse trials, diagnostic
  details, shortlist widening/exclusion, and one-time retry lifecycle.
- A 40-second headless causal Chronos run starting from
  `gLO_sBM_cAll_d13_mLO2_ch02` completed three transitions, with zero terminal
  holds and zero reported control deadline misses. The expanded retry selected
  `gHO_sFM_cAll_d21_mHO5_ch20` after `gPO_sBM_cAll_d11_mPO0_ch02`, completing
  at wall time 25.837 s. Maximum recorded preparation time was 1.234 s.
- Artifacts: `tmp/v2_ab_final_20260916.{log,csv,json}` and the corresponding
  `_poses.npz`. This is a short regression run, not a guarantee of indefinite
  playback. Current exit-window logic differs from the older retest2 code.
