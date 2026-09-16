# V2 transition timing review — 2026-09-14

## Conclusion

Yes: the default Hermite path can run out of **preparation time**, and a missed
deadline leaves playback permanently holding the terminal pose. A separate
failure is that all candidate bridges and the replay can be infeasible. Making
the shortlist smaller improves some cases but did not prevent a stop here.

This review leaves the playback algorithm unchanged. Added a new test track and
an opt-in profiling runner to make the expensive background search measurable.

## New track and measurements

Chronos, from [FreePD's electronic category](https://en.freepd.cn/music/electronic),
is listed by the source as CC0/public domain. The downloaded track is
`realtime/humanoid_robot/data/test_audio/chronos.mp3`; adjacent `chronos.json`
records its source and checksum. Duration: 129.22 s. Full-track librosa tempo
estimate: 117.19 BPM, not ground truth. This is outside the AIST++ music catalog
and the existing ten-track test set. In these runs the online beat estimate
eventually settled around 121 BPM after unstable initial estimates, making the
intro useful for testing acquisition as well as repeated transitions.

Both runs used headless MuJoCo, causal file input, 120 Hz, beat-sync, default
Hermite limits, and initial-motion seed 42. The initial motion was
`gWA_sBM_cAll_d26_mWA0_ch05`. Speaker playback was off.

| Measurement | Default shortlist (5), 60 s | Shortlist 1, 40 s, profiled |
|---|---:|---:|
| Completed transitions | 1 | 3 |
| First terminal hold, audio time | 17.368 s | 35.224 s |
| Control iterations | 5504 | 4313 |
| Control deadline misses | 822 (14.93%) | 336 (7.79%) |
| Control work p95 | 13.21 ms | 8.32 ms |
| Control work p99 | 29.68 ms | 21.11 ms |
| Maximum output interval | 147.87 ms | 70.74 ms |

The 120 Hz period is 8.33 ms. These are exploratory runs on this Windows host,
not repeated controlled benchmarks: duration, selected motions, causal beat
timing and profiling differ. Full-track audio analysis was also run during the
first baseline, so its timing is not an isolated-machine benchmark. Do not
interpret the table as an isolated causal
estimate of shortlist size. No GUI, physical robot, balance validation or runtime
collision checking was exercised.

## Findings

1. **Preparation waits for the entire frozen shortlist.**
   `AuthoredPlayback.prepare` returns `loading` if any nonfailed candidate is not
   ready. `MotionLoader` prepares motions on one process worker. The default
   run's first shortlist waited about 5.54 s, although its eventual successful
   bridge itself lasted just 0.35 s. After the first transition, loading took
   about 3.30 s, leaving only about 1.48 s of wall time between the observed
   `preparing` state and terminal hold. Source locations: matcher v2
   `prepare` around lines 998–1040 and loader construction around 1678–1706.

2. **Bridge search has no computation deadline or attempt limit.**
   `prepare_state_bridge` ranks every eligible entry across candidates and tries
   them sequentially; each Hermite attempt may search many durations and solve
   polynomial extrema for every joint. The 10 s maximum in `make_bridge` bounds
   the *generated trajectory's duration*, not CPU time. In the shortlist-1 run,
   the first three preparations took 35.4, 33.8 and 31.8 ms, but the fourth took
   5.700 s: 43 attempts, all rejected. Its final reported reason was
   `Boundary state exceeds joint limits`; this is the last rejection, not proof
   that all 43 failed for the same reason. Source: matcher lines 909–960 and
   `motion_bridges.py`, `make_bridge`.

   A separate 25-second repeat with the default shortlist and profiling confirmed
   the expensive case: the second search attempted **211 bridges, rejected all
   211, and took 21.108 s**, including 21.051 s inside the solver. The first
   search took 59.3 ms. The failed search continued beyond the playback deadline;
   increasing the preparation reserve slightly would not solve this case.
   Its profile is `tmp/v2_chronos_default_profile.jsonl`; trace, timing report,
   log and poses use the `tmp/v2_chronos_profiled` prefix. Reproduce with the
   profiling command below, omit `--shortlist-size 1`, use `--max-seconds 25`,
   and choose those distinct output paths.

3. **A missed deadline is permanent, and search is not cancelled.**
   `sample` sets `held=True` at the endpoint without a plan; both `prepare` and
   `sample` subsequently return early. The default run held for the remaining
   roughly 42.6 seconds. A late result cannot restore playback. Simply clearing
   `held` would be incorrect: the planned bridge starts with the authored
   endpoint velocity/acceleration, while a held pose is at rest. Recovery needs
   a newly planned state-consistent transition. An in-progress score worker can
   continue after the hold, and shutdown waits for it. Source: matcher lines
   1002, 1051–1080, 1925–1927.

4. **The scoring thread can compete with the control loop.**
   Motion loading is process-isolated, but bridge scoring uses a thread in the
   controller's process. Long Python/extrema searches can contend for the GIL
   and CPU. The observed misses demonstrate inadequate timing in these runs;
   the precise share attributable to the GIL versus other scheduling costs is
   not established by these measurements.

5. **Existing validation can miss a failed performance run.**
   All 25 bridge tests and 16 matcher tests passed. The existing trace verifier
   also passed both runs because at least one transition completed; it does not
   reject a later `bridge_hold`. The default Hermite path also does not append
   to `loader.score_seconds`, so `entry_score_ms_max` was misleadingly zero.
   The new profiling runner captures the actual search time independently.

The bridge's motion duration does not need to fit inside the remaining source
clip: it executes after the source ends, and the target entry is held in phase
until handoff. The deadline concerns constructing the bridge before that join.
Feasibility still depends on endpoint derivatives, limits and the finite
duration search; simply increasing duration is not guaranteed to help.

## Recommended implementation work

1. Give search a wall-time/attempt budget, and check cancellation between
   entries and duration trials; move expensive scoring into a process worker.
2. Prepare and validate a fallback early, and retain it while better candidates
   are searched. Replay itself must be checked; it is not always feasible.
3. Allow planning from candidates already ready instead of blocking on all five,
   with an explicit policy for whether later candidates can replace the plan.
4. Use measured loading/search time and the maximum playback rate to reserve
   enough time at each entry. The current four-second entry rule only checks
   remaining playback, not the next preparation budget.
5. Add terminal-hold counts, submit/start/finish/slack telemetry and a strict
   continuous-playback acceptance check. Design rest-state recovery separately
   if automatic recovery is required.

## Reproduce and artifacts

From the repository root, the first run used:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/realtime_music_humanoid_matcher_v2.py --headless --realtime --audio-input realtime/humanoid_robot/data/test_audio/chronos.mp3 --experiment-causal-file-input --experiment-pose-npz tmp/v2_chronos_default_poses.npz --max-seconds 60 --initial-motion-seed 42 --trace-csv tmp/v2_chronos_default.csv --timing-report tmp/v2_chronos_default.json
```

The profiled shortlist-1 run used:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/test/profile_humanoid_v2_bridges.py --bridge-profile tmp/v2_chronos_profile.jsonl --headless --realtime --audio-input realtime/humanoid_robot/data/test_audio/chronos.mp3 --experiment-causal-file-input --experiment-pose-npz tmp/v2_chronos_one_poses.npz --max-seconds 40 --initial-motion-seed 42 --shortlist-size 1 --trace-csv tmp/v2_chronos_one.csv --timing-report tmp/v2_chronos_one.json
```

Logs, CSV traces, timing JSON and final-pose NPZ files share the corresponding
`tmp/v2_chronos_default` and `tmp/v2_chronos_one` prefixes. The profiling runner
wraps the existing functions without changing selection, limits or playback.
