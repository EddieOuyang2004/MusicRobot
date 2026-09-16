# Authored-state bridge validation

## Implementation

Matcher v2 defaults to general quintic Hermite bridges. `--transition-backend
ruckig` selects the optional Ruckig backend; `quintic` selects the previous
rest-to-rest implementation. V1 and shared joint-limit types are unchanged.

Both new backends use cached, adapted authored joint states and a matching
piecewise quintic playback representation. One-sided finite differences estimate
endpoint derivatives. Music is not predicted during planning. Playback deviations
and speed fade to neutral/1x at the boundaries, with no extra held entry sample.
Root blending remains kinematic and is not covered by joint derivative guarantees.

Jerk is optional for Hermite and required for every joint for Ruckig. The CLI
`--output-max-joint-jerk` supplies a common value. The existing
`--output-joint-limits-json` accepts `max_jerk_rad_s3` under `default` and individual
`joints` entries; JSON defaults override the CLI fallback and per-joint values
override defaults, consistently with existing velocity/acceleration settings.
No joint jerk limits are assumed by default.

Hermite checks polynomial extrema, trying the initial duration, successive 1.2x
durations, and 10 seconds. This finite search is not a proof that no other
polynomial duration could work. Ruckig checks its reported position extrema and
rejects solver failures or durations above 10 seconds. Neither backend relaxes
boundary derivatives or changes backend on failure.

## Checks performed

- 25 numerical/lifecycle tests cover nonzero boundary derivatives, analytic
  extrema, overshoot, optional jerk configuration, Ruckig dependency errors and
  the actual Ruckig 0.19.4 solver, short clips, consistent authored interpolation,
  fade envelopes, clock carryover, first-retrieval commitment, stale generations,
  candidate retries, replay and terminal holds.
- The 16 existing matcher-v2 tests pass.
- Two completed headless replays per backend using
  `gJS_sBM_cAll_d03_mJS5_ch02`; both traces pass the updated verifier. The Ruckig
  run used 10000 rad/s^3 solely as an explicit simulation test setting.
- A 26-second causal file-input Hermite run with `--shortlist-size 1` completed
  two different-clip transitions: JS5 -> MH0 (entry frame 35), then MH0 -> BR0
  (entry frame 3). Its trace passes the verifier. Music effects and beat timing
  remained enabled outside boundary fades.
- A replay of `gHO_sBM_cAll_d20_mHO5_ch01` correctly rejected internal joint
  position overshoot. Increasing duration did not fix its authored derivatives.
- A causal music run with that short initial clip and the default five-entry
  shortlist exercised the preparation-deadline hold. A missed bridge remains
  held until restart; a late moving-endpoint trajectory is not resumed from rest.

All these runs were headless. Runtime collision checks used the inherited off
default. These are functional checks, not physical balance, collision clearance,
hardware tracking, or worst-case 120 Hz timing validation.

## Reproduce

Run from the repository root using the project virtual environment:

```powershell
.venv/Scripts/python.exe -m pip install -r realtime/humanoid_robot/requirements-ruckig.txt
.venv/Scripts/python.exe -m unittest discover -s realtime/humanoid_robot/src/test -p test_motion_bridges.py
.venv/Scripts/python.exe -m unittest discover -s realtime/humanoid_robot/src/test -p test_humanoid_matcher_v2.py
.venv/Scripts/python.exe realtime/humanoid_robot/src/realtime_music_humanoid_matcher_v2.py --headless --realtime --no-mic --max-seconds 16 --initial-motion-id gJS_sBM_cAll_d03_mJS5_ch02 --transition-backend hermite --trace-csv tmp/authored_hermite_success.csv
.venv/Scripts/python.exe realtime/humanoid_robot/src/realtime_music_humanoid_matcher_v2.py --headless --realtime --no-mic --max-seconds 16 --initial-motion-id gJS_sBM_cAll_d03_mJS5_ch02 --transition-backend ruckig --output-max-joint-jerk 10000 --trace-csv tmp/authored_ruckig_success.csv
.venv/Scripts/python.exe realtime/humanoid_robot/src/realtime_music_humanoid_matcher_v2.py --headless --realtime --audio-input realtime/humanoid_robot/data/test_audio/aistpp_stitched_test.wav --experiment-causal-file-input --experiment-pose-npz tmp/authored_hermite_shortlist_poses.npz --max-seconds 26 --initial-motion-id gJS_sBM_cAll_d03_mJS5_ch02 --transition-backend hermite --shortlist-size 1 --trace-csv tmp/authored_hermite_shortlist.csv
.venv/Scripts/python.exe realtime/humanoid_robot/src/test/verify_humanoid_matcher_v2_trace.py tmp/authored_hermite_shortlist.csv
```

`bridge_position_residual`, `bridge_velocity_residual` and
`bridge_acceleration_residual` describe the planned bridge's two endpoints.
`bridge_output_modified` compares each planned joint sample with final displayed
joint positions after limiting, collision policy and player clipping. Inspect it
separately; successful polynomial joins do not imply unmodified emitted motion.
