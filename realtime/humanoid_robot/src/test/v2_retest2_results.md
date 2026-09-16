# Current v2 Chronos retest

Tested working-tree v2 after its 2026-09-15 03:13 update. No playback code changed
by this test. All 26 bridge and 21 matcher tests passed.

## 60-second headless causal run

- Default Hermite, beat-sync, shortlist 5, initial-motion seed 42, 120 Hz.
- No terminal holds or failed preparation states.
- Four completed replay bridges; fifth began at 59.768 s and remained in progress
  when the run ended. All bridges were 0.35 s.
- All five selected the original `gWA_sBM_cAll_d26_mWA0_ch05` with reason
  `replay_no_feasible_candidate`. No different-motion transition was exercised.
- Zero control deadline misses across 7324 iterations; control work p95 2.422 ms,
  p99 3.071 ms, maximum 5.727 ms. Maximum output interval 11.740 ms.
- Zero final recorded joint-limit violations. The pre-player counter recorded
  12 joint/sample exceedances. No bridge modifications flagged in trace samples.
- Trace verifier passed with minimum remaining duration set to 8 seconds.

## Why it keeps replaying

The current default entry minimum is 8 seconds. Beat-sync entry eligibility
divides remaining authored duration by maximum speed (1.6x), so even a frame-zero
entry needs about 12.8 seconds of authored motion. The actual PO shortlist clips
are 12.0 or 8.733 seconds, and the later HO shortlist clips are 7.1 seconds.
All are therefore too short for this eligibility rule, before checking their
bridges. Replay bypasses the remaining-duration restriction.

Continuous playback worked in this run, but this is not evidence that the earlier
different-motion bridge infeasibility is fixed: those candidates were excluded
by duration. The generic replay reason does not distinguish this case from
solver rejection.

## Reproduce

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/realtime_music_humanoid_matcher_v2.py --headless --realtime --audio-input realtime/humanoid_robot/data/test_audio/chronos.mp3 --experiment-causal-file-input --experiment-pose-npz tmp/v2_chronos_retest2_poses.npz --initial-motion-seed 42 --max-seconds 60 --trace-csv tmp/v2_chronos_retest2.csv --timing-report tmp/v2_chronos_retest2.json
```

Artifacts: `tmp/v2_chronos_retest2.csv`, `.json`, `.log`, and `_poses.npz`.
No speaker output or GUI; no physical balance or collision-clearance validation.
