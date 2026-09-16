# AIST++ post-GMR smoothness audit

Scope: all local exported clips, authored speed, within each clip only. No transitions or loop seams.

Raw derivatives use first/second/third forward differences with the artifact FPS. V2 joint interpolation uses the actual AuthoredTrajectory implementation. Speed, acceleration and jerk peaks are exact: every stationary point of each quintic segment is solved in closed form. Position overshoot is sampled at 33 points per segment, so that metric alone is a lower bound. V2 reference thresholds are the defaults of its output limiter, --output-max-joint-speed 16 rad/s and --output-max-joint-acceleration 1600 rad/s². These are software limits, not perceptual smoothness or hardware certification. No jerk acceptance threshold is configured in V2. Joint ranges come from g1_29dof.xml. Root orientation differences use rotations, not quaternion component subtraction. Music modulation, time warping, runtime collision corrections and final output limiting are excluded.

## Coverage and findings

```json
{
  "artifact_count": 411,
  "valid_count": 411,
  "errors": [],
  "catalog_count": 411,
  "source_count": 411,
  "missing_catalog_artifacts": [],
  "missing_source_artifacts": [],
  "total_frames": 275270,
  "clips_with_raw_speed_violating_intervals": 0,
  "clips_with_v2_speed_bad_segments": 127,
  "clips_with_v2_acceleration_bad_segments": 347,
  "clips_with_root_speed_violation": 0,
  "clips_with_root_angular_speed_violation": 0,
  "clips_with_fast_reversal_frames": 309,
  "clean_clips": 60,
  "fast_reversal_frames": 4761,
  "export_speed_clamp_fraction": 0.02928774389778032,
  "clips_with_raw_position_excess_rad": 0,
  "clips_with_v2_position_excess_rad": 346,
  "acceleration_bad_interval_fraction": 0.02925499983627969,
  "acceleration_bad_interval_share_per_clip": {
    "p50": 0.013908205841446454,
    "p90": 0.07074569789674952,
    "p100": 0.4298642533936652
  },
  "maxima": {
    "raw_acceleration_rad_s2": 1130.973355292326,
    "raw_jerk_rad_s3": 135716.80263507908,
    "v2_speed_rad_s": 18.849560022354126,
    "v2_acceleration_rad_s2": 3229.557292651826,
    "v2_jerk_rad_s3": 1995038.4027063847,
    "v2_position_excess_rad": 0.019761862530231378,
    "root_acceleration_m_s2": 359.6539123855292,
    "root_angular_acceleration_rad_s2": 1507.8270668951145
  }
}
```

## Largest interpolated acceleration peaks

| Motion | Joint | Time (s) | Speed peak (rad/s) | Acceleration peak (rad/s²) | Jerk peak (rad/s³) | Saturated reversal frames |
|---|---|---:|---:|---:|---:|---:|
| gJB_sBM_cAll_d09_mJB5_ch07 | right_knee | 6.5132 | 17.52 | 3229.56 | 1995038 | 124 |
| gWA_sBM_cAll_d26_mWA0_ch02 | left_shoulder_roll | 6.1299 | 18.85 | 3213.09 | 1984858 | 41 |
| gJB_sFM_cAll_d07_mJB5_ch06 | right_ankle_pitch | 26.2868 | 17.52 | 3209.63 | 1933196 | 47 |
| gJB_sBM_cAll_d07_mJB0_ch02 | left_shoulder_pitch | 8.7966 | 17.49 | 3199.17 | 1946697 | 46 |
| gMH_sFM_cAll_d23_mMH3_ch11 | right_hip_yaw | 12.5965 | 17.49 | 3198.16 | 1921436 | 32 |
| gJB_sBM_cAll_d09_mJB5_ch03 | waist_pitch | 3.1132 | 17.40 | 3170.19 | 1907846 | 60 |
| gJB_sFM_cAll_d09_mJB5_ch20 | right_hip_pitch | 25.0466 | 17.39 | 3169.13 | 1882061 | 37 |
| gWA_sBM_cAll_d26_mWA0_ch03 | left_hip_pitch | 11.0798 | 17.37 | 3164.37 | 1902676 | 12 |
| gWA_sBM_cAll_d26_mWA0_ch01 | right_elbow | 1.2868 | 17.38 | 3157.49 | 1832177 | 17 |
| gJB_sBM_cAll_d09_mJB5_ch10 | right_elbow | 2.1966 | 17.38 | 3157.49 | 1832177 | 56 |
| gWA_sBM_cAll_d27_mWA3_ch02 | left_shoulder_yaw | 5.2868 | 18.85 | 3157.49 | 1832177 | 61 |
| gMH_sFM_cAll_d24_mMH3_ch18 | left_shoulder_roll | 33.7799 | 18.85 | 3094.45 | 1814343 | 50 |
| gPO_sFM_cAll_d11_mPO1_ch09 | left_wrist_roll | 21.7202 | 17.18 | 3088.12 | 1866644 | 29 |
| gWA_sBM_cAll_d25_mWA2_ch02 | left_shoulder_pitch | 8.4632 | 16.68 | 3034.40 | 1740462 | 30 |
| gJB_sFM_cAll_d08_mJB5_ch13 | left_elbow | 1.5535 | 16.85 | 3026.07 | 1749760 | 23 |
| gPO_sFM_cAll_d10_mPO1_ch02 | right_hip_roll | 23.9868 | 16.81 | 3019.87 | 1764309 | 13 |
| gJB_sFM_cAll_d08_mJB5_ch14 | right_shoulder_yaw | 9.3132 | 16.71 | 2986.35 | 1736820 | 83 |
| gJB_sBM_cAll_d09_mJB3_ch01 | right_elbow | 0.1201 | 16.65 | 2984.10 | 1745682 | 13 |
| gJB_sBM_cAll_d09_mJB4_ch02 | left_shoulder_yaw | 4.7465 | 18.85 | 2980.54 | 1743421 | 26 |
| gWA_sBM_cAll_d25_mWA0_ch02 | right_shoulder_yaw | 7.4701 | 16.60 | 2961.66 | 1680389 | 18 |

## Interpretation

C2 interpolation makes position, velocity and acceleration continuous, but does not bound their magnitudes or make jerk continuous. The dominant source of the peaks is the exported samples themselves: GMR clamps per-frame joint speed, and many clips hold that clamp while reversing direction from one frame to the next, which is a full-amplitude oscillation at the frame Nyquist rate. Central-difference velocities and accelerations at those frames feed the quintic endpoint states, and the interpolation then overshoots to roughly twice the export clamp. Filtering the interpolator alone would not remove chatter that is already present in the samples.

Exceedances are exact, so they disprove compliance outright. A clip reported clean is clean for these three metrics at authored speed only; music-driven time warping raises both speed and acceleration further. Do not apply an unchecked low-pass filter to collision-constrained motions: any revised trajectory needs joint-range and collision revalidation.

The CSV and JSON include every clip, root metrics, source artifact SHA-256 hashes, export clamp statistics and exact locations of acceleration peaks.
