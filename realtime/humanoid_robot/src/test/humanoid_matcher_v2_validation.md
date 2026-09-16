# Humanoid matcher v2 validation — 2026-09-12

## Checks

- 16 deterministic unit tests cover exhaustive search against a scalar reference,
  terminal-source selection, pose/velocity/contact ranking, musical shortlist
  margins, ties, remaining duration, replay fallback, quintic interpolation and
  duration bounds, causal audio history, sample-rate forwarding, and CLI validation.
- An 18-second no-input run with the real `gHO_sBM_cAll_d20_mHO5_ch01` motion
  completed two smooth replays. Trace verification confirmed terminal-first
  switching, frame-zero fallback, held entry phase, and completed interpolation.
- A 40-second causal file-input run completed three different-clip transitions
  and one replay. The same trace checks passed, including four-second entry
  eligibility and the 30-second history cap.
- A subsequent 16-second run after terminal-command capture and timing telemetry
  changes exercised a different-clip transition again.

The integration runs used the existing catalog and GMR artifacts, a headless
MuJoCo player, default beat-sync and modulation for music input, and a 120 Hz
control rate. No physical robot or live microphone was used. Runtime collision
checks were off under the inherited default; these runs are continuity and
timing checks, not validation of physical balance or collision-free transitions.

## Initial v1 comparison

Both versions used the first 40 seconds of `data/test_audio/aistpp_stitched_test.wav`,
the same explicit initial motion, causal file input, final-pose NPZ recording,
and 120 Hz headless realtime execution. Runs were performed sequentially.

| Measurement | V1 | V2 |
| --- | ---: | ---: |
| Control iterations | 4801 | 4801 |
| Recorded deadline misses | 0 | 0 |
| Control work p95 | 1.89 ms | 1.81 ms |
| Control work p99 | 2.42 ms | 2.59 ms |
| Background audio feature extraction p95 | 45.59 ms | 254.06 ms |
| Completed transitions | 8 | 4 |
| Largest joint RMS frame step at transition start | 0.03713 rad | 0 rad |
| Largest joint RMS frame step at transition completion | 0.02353 rad | 0.000047 rad |

The frame-step figures compare the final recorded joint vector with the immediately
preceding vector at each boundary event. They are continuity observations from one
run, not a claim of universally better motion quality. V1 blends moving clips;
v2 holds its entry pose during interpolation. V1 began these transitions between
phase 0.91 and 0.99; all v2 transitions began after phase 1.0.

The longer history has a measurable responsiveness cost. The first retrieval
arrived at audio time 2.04 seconds in v2 versus 6.04 seconds in v1. Both initially
retrieved BR0. After the first stitched change, JS3 became the top retrieved track
at 11.16 seconds in v1 and 16.47 seconds in v2. KR2 became top at 19.35 versus
23.74 seconds. V2 still ranked KR2 highest through its last completed query around
38 seconds, whereas v1 had subsequently recognised PO1 and WA0. This is consistent
with equally weighted 30-second history averaging across multiple eight-second
segments; it does not establish retrieval accuracy on a broader music set.

Local diagnostic artifacts are under `tmp/matcher_v2_replay.*`,
`tmp/matcher_v2_music*`, `tmp/matcher_v2_final*`, and `tmp/matcher_v1_comparison*`.

## Reproduce

```powershell
python -m unittest discover -s realtime/humanoid_robot/src/test -p test_humanoid_matcher_v2.py
python realtime/humanoid_robot/src/realtime_music_humanoid_matcher_v2.py --headless --realtime --audio-input realtime/humanoid_robot/data/test_audio/aistpp_stitched_test.wav --experiment-causal-file-input --experiment-pose-npz tmp/v2_poses.npz --max-seconds 40 --initial-motion-id gHO_sBM_cAll_d20_mHO5_ch01 --trace-csv tmp/v2.csv --timing-report tmp/v2_timing.json
python realtime/humanoid_robot/src/test/verify_humanoid_matcher_v2_trace.py tmp/v2.csv
```

Substitute `realtime_music_humanoid_matcher.py` and distinct output filenames for
the comparison run. The v2 trace verifier intentionally rejects v1 early switches.
