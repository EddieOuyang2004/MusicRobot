# V2 exit/entry pair validation — 2026-09-16

## Implementation

- Search all reachable, valid authored source frames in the final two seconds against eligible target entries. Zero exit window selects the original terminal-only behavior.
- Reuse the existing nonlinear pose/velocity/contact/root score without new weights or acceleration penalties. Full bridge validation still enforces dynamics limits.
- Filter invalid boundary states once per motion, score one exit row at a time with NumPy, and merge sorted rows lazily. No exit × entry × joint tensor or full collection of pair score objects.
- Commit early exits before either affected fade begins; keep the full clip duration for phase sampling and use a separate playback stop time. Discard late/stale plans.
- Check cancellation/deadlines between ranking rows, bridge attempts, and Hermite duration trials. A native solver call is not preemptible.
- V2 now defaults to 60 Hz. Explicit control-rate overrides work; v1 and the standalone dancer keep 120 Hz.

## Tests

64 tests passed: 34 bridge/lifecycle (including real Ruckig), 23 matcher/scoring, 4 Hermite bounds, and 3 transition demo tests. Coverage includes exhaustive pair-ranking equivalence, invalid terminal/valid earlier exit, matching fast frames, late-result rejection, cancellation, terminal-only mode, replay, and rate convergence/carryover at 60 and 120 Hz.

## Full runtime comparison

Seven sequential headless causal Chronos runs, 60 seconds each, seed 42, default Hermite and beat-sync, unchanged limits and eight-second target-entry minimum. The terminal-only run uses the new default control rate. Startup is outside scheduler timing. These are shared-Windows-host measurements; online retrieval and preparation timing can change the selected sequence even with the same seed.

| Run | Hz | Work p99 (ms) | Misses (%) | Completed | Different motion | Holds | Max successful preparation (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exit_pairs_final_120_1 | 120 | 9.454 | 4.422 | 2 | 2 | 1 | 1036.5 |
| exit_pairs_final_120_2 | 120 | 11.063 | 6.568 | 5 | 1 | 0 | 1426.3 |
| exit_pairs_final_120_3 | 120 | 11.250 | 5.378 | 2 | 2 | 1 | 1340.2 |
| exit_pairs_final_60_1 | 60 | 11.642 | 0.577 | 2 | 2 | 1 | 812.8 |
| exit_pairs_final_60_2 | 60 | 11.650 | 0.578 | 2 | 2 | 1 | 830.7 |
| exit_pairs_final_60_3 | 60 | 12.064 | 0.523 | 2 | 2 | 1 | 820.5 |
| exit_pairs_terminal_60 | 60 | 10.982 | 0.550 | 2 | 2 | 1 | 14.9 |

All three 60 Hz runs meet the loop timing criteria (p99 below 16.67 ms and fewer than 1% deadline misses). None of the 120 Hz runs meets its 8.33 ms / 1% criteria. Final recorded joint-limit violations are zero in every run.

**Continuous playback is not fully solved:** all three 60 Hz pair-search runs eventually hold because candidate bridges and replay are infeasible, not because the recorded preparation deadline expired. Lowering the control rate does not remove this motion-specific limitation. The strict full trace verifier rejects these runs; their completed-transition prefixes pass. One 120 Hz run passes the full trace verifier, but mostly replays.

Pair-search time is background preparation time, not per-control-tick work. It increases with exits × candidate entries × joints; bridge solving stops at the first feasible ranked pair. The recorded successful bridges each required one solver attempt. Motion loading is separately reported in timing JSON and is not included in the successful-search times above. Failed preparation and poll latency are included in the existing `entry_score_ms_*` metrics.

At 60 Hz, successful pair searches took 360–831 ms for 36,470–86,999 eligible pairs, versus 12–15 ms for 719–1,002 pairs in the terminal-only run. This is a substantial preparation-cost increase, despite the acceptable control-loop timing. Start preparing while the current clip is still playing; these costs would not fit inside one control tick at either rate.

For the same second motion pair (`gPO_sBM_cAll_d10_mPO0_ch02` to `gPO_sBM_cAll_d11_mPO0_ch02`), terminal-only selection scored 2.685721 at target frame 123. All three 60 Hz pair searches selected source frame 641 (1.3 seconds before the end) and target frame 117, scoring 0.307327 under the unchanged scoring formula. Both approaches produced a 0.35-second bridge. This demonstrates a better matching boundary score for this pair; both approaches completed two different-motion transitions before the later infeasible hold.

Detailed pair counts, trimmed durations, attempts, failure reasons, trace verification results, and loading times are in `tmp/exit_pairs_validation_summary.json`. Per-run CSV, JSON, logs, and pose NPZ files share the run names above. Initial prefilter-development runs (`exit_pairs_120_*`) are excluded from this table.

## Reproduce

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/realtime_music_humanoid_matcher_v2.py --headless --realtime --audio-input realtime/humanoid_robot/data/test_audio/chronos.mp3 --experiment-causal-file-input --experiment-pose-npz tmp/exit_pairs_repeat.npz --max-seconds 60 --initial-motion-seed 42 --control-rate-hz 60 --trace-csv tmp/exit_pairs_repeat.csv --timing-report tmp/exit_pairs_repeat.json
```

Repeat three times with distinct output names at 120 and 60 Hz. Add `--transition-exit-window-seconds 0` for the terminal-only baseline. Use the existing unit test files listed above with `python -m unittest discover -s realtime/humanoid_robot/src/test -p <filename>`. These runs validate software timing and joint bridge continuity, not physical balance or collision-free locomotion.
