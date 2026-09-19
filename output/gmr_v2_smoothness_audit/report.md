# GMR v2 smoothness assessment

All 411 v2 files passed the checked joint speed, acceleration, reversal and sampled joint-range criteria at authored speed. This does not mean every aspect of whole-body playback is fully smooth.

Coverage: 275,270 frames, 76.35 minutes. No files changed between the primary and supplemental audits.

| Check | Original GMR | GMR v2 |
|---|---:|---:|
| Clips exceeding playback speed threshold (16 rad/s) | 127 | 0 |
| Clips exceeding acceleration threshold (1600 rad/s²) | 347 | 0 |
| Clips with fast joint reversals | 309 | 0 |
| Clips with sampled joint-range overshoot > 1e-5 rad | 346 | 0 |

V2 also passes each file's stricter export speed bound (3*pi = 9.42478 rad/s). Maximum continuous joint acceleration is 343.47 rad/s².
True whole-joint holds, using the continuous speed polynomial: 0 intervals across 0 clips; longest 0.0000 s. A hold is not automatically a defect or proof of collision blocking.
Maximum within-clip C2 join residuals (position/velocity/acceleration): {'position_rad': 5.551115123125783e-17, 'velocity_rad_s': 7.105427357601002e-15, 'acceleration_rad_s2': 1.3073986337985843e-12}. These are floating-point-scale differences.

## Remaining limitations

- Jerk is not constrained to be continuous and no acceptance threshold is configured. Peak joint jerk is 95537.4 rad/s³; the largest adjacent-segment jerk jump is 82172.7 rad/s³. These are measurements, not a pass/fail judgment of perceptual smoothness.
- Root translation/rotation are unchanged within 1e-7 tolerance in 394/411 clips. Root acceleration reaches 359.65 m/s² and angular acceleration reaches 1507.83 rad/s². Joint smoothing does not resolve those whole-body spikes.
- 411 clips have a raw last-to-first joint-pose difference over 0.1 rad (maximum 3.434 rad). This flags direct-wrap seams, not the output of the runtime loop/transition bridge.
- Music-driven time warping, runtime transitions, contact/dynamic tracking and human-perceived smoothness were not certified. No motion files were modified.
- Stored revision-2 state checksums and validation metadata passed. Collision tests were recorded by generation; this audit did not rerun full collision geometry for every frame.
- Joint derivative extrema were recomputed analytically from the saved Hermite curves. Joint-position bounds were sampled at 33 points per segment. Root derivatives use finite differences.

## Highest measured joint jerk

| Motion | Joint | Time (s) | Jerk (rad/s³) |
|---|---|---:|---:|
| gJB_sBM_cAll_d08_mJB5_ch04 | right_hip_roll | 0.0000 | 95537.4 |
| gBR_sBM_cAll_d05_mBR0_ch09 | left_knee | 8.8000 | 82363.6 |
| gKR_sBM_cAll_d29_mKR1_ch01 | left_shoulder_pitch | 6.5333 | 29652.6 |
| gJB_sFM_cAll_d07_mJB5_ch06 | left_shoulder_roll | 7.0500 | 26499.3 |
| gBR_sBM_cAll_d05_mBR0_ch05 | right_shoulder_yaw | 0.4667 | 20201.9 |

Recommendation: retain the current joint-state planner; next address root translation/orientation smoothness and explicit jerk targets, then check loop/transition and music-warped playback. Avoid claiming that all motion is perceptually or dynamically smooth solely from the joint-limit pass.

Files: `audit.json` and `motions.csv` contain all primary results and source hashes; `supplement.json` contains holds, joins, jerk locations and loop gaps; `baseline/` contains the freshly rerun original-dataset comparison. The existing auditor's four regression tests passed.
