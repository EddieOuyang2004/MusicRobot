# Action transition review and isolated demo

The requested command was absent from the message, so this review covers both
matcher versions. No live matcher code was changed.

## Findings

- In `realtime_music_humanoid_matcher_v2.py`, the terminal block around line 2600
  finishes A and holds its final command while a worker rechecks entry scores.
  Preparation/recheck latency can extend this hold. The bridge around line 2697
  interpolates two fixed poses with quintic easing. B's playback clock starts
  after the bridge, with another exact entry-pose sample on the next tick.
  This is position-continuous but the bridge has zero endpoint velocity:
  nonzero terminal velocity in A or entry velocity in B creates a stop/restart.
  Output limiting can soften this at the cost of tracking lag. A quintic weight
  alone does not make the entire concatenated trajectory velocity-continuous.
- In `realtime_music_humanoid_matcher.py`, `compute_transition_duration` clips
  the calculated duration to `maximum`, even when dynamics require longer.
  For example, a 2-radian stationary displacement at 1 rad/s needs at least
  3 seconds with the cubic smoothstep used there; a 1.2-second cap implies a
  2.5 rad/s peak. Moving endpoints add further velocity terms. This is a
  potential cause of limiter intervention, not proof of the user's exact run.
- V2 already removes that duration cap. Its duration calculation bounds joint
  motion for a stationary bridge; it does not establish root dynamics or
  continuity with the incoming/outgoing moving clips. Independently grounded
  clips and joint-space blending also do not guarantee planted-foot continuity.

## Run the demo

From the repository root, compare the same two automatically selected GMR clips:

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/demo_action_transition.py --mode bridge
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/demo_action_transition.py --mode overlap
```

The viewer shows 3 seconds of A, the transition, then 3 seconds of B. Each run
exits automatically. `bridge` demonstrates the fixed-pose approach; `overlap`
starts the transition before A finishes and advances both clips throughout it.
Both use quintic weights and root alignment. The duration defaults to 1.2 seconds.

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/demo_action_transition.py --list-motions
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/demo_action_transition.py --motion-a gBR_sBM_cAll_d04_mBR0_ch01 --motion-b gBR_sBM_cAll_d04_mBR0_ch02 --mode overlap --duration 1.2 --slow-motion 2 --csv tmp/transition.csv
```

Use `--entry-seconds` to select where B begins, `--lead-seconds` and
`--tail-seconds` to change preview length, and `--headless` for deterministic
fast execution. Paths to GMR pickles are also accepted. Sampling clamps at the
last authored frame so the demo never blends the end of a clip back to frame 0.

This isolates kinematic blending: it does not run microphone retrieval,
automatic entry selection, music modulation, the output dynamics limiter,
or the live collision policy. CSV values are raw requested joint targets,
not measured robot motion. The stationary mode uses a user-set duration and
entry, so it is not an exact replay of the complete V2 pipeline.

## Validation

Three deterministic tests verify overlap pose/velocity continuity (joints and
root translation), expose the stationary bridge's velocity discontinuities,
and ensure B's time continues after overlap:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s realtime/humanoid_robot/src/test -p test_action_transition_demo.py
```

Both modes completed headless with the default clips at 120 Hz and 1.2 seconds.
Raw target peaks over the whole preview, including authored motion:

| Mode | Speed (rad/s) | Acceleration (rad/s²) |
| --- | ---: | ---: |
| Bridge | 9.425 | 2045.470 |
| Overlap | 9.732 | 1395.895 |

These whole-preview numbers are not a general smoothness guarantee: the modes
show different source/target time intervals. Boundary tests provide the direct
continuity evidence. GUI appearance has not been visually verified.

For live adoption, first reproduce the exact command and clip pair. If complete
playback of A is required, decelerate into the final frame and accelerate out
of B's entry, or use a bridge matching endpoint velocities with dynamics checks.
If overlapping clips is acceptable, evaluate the demo's moving crossfade with
entry selection, contact compatibility, and the existing limiter enabled.
