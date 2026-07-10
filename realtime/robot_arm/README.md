# Realtime Robot Arm

This folder contains the realtime PyBullet robot-arm workflow:

1. Generate or provide a looping trajectory CSV
2. Estimate beat period from recent taps or microphone clap onsets
3. Retime playback with a phase integrator instead of wall-clock `elapsed * speed`

## 1) Generate a Simple Clap Trajectory

```powershell
python realtime/robot_arm/src/generate_clap_trajectory.py --out realtime/robot_arm/outputs/clap_trajectory.csv
```

## 2) Tap Tempo with Space Bar

This is the first development step. Start the player and tap the space bar in the PyBullet window.

```powershell
python realtime/robot_arm/src/adaptive_play_trajectory.py --csv realtime/robot_arm/outputs/clap_trajectory.csv --input-source tap --realtime
```

Controls:

- `Space`: register a beat
- `R`: reset phase and tempo controller
- `Q`: quit
- `Esc`: also quits on PyBullet builds that expose `B3G_ESCAPE`

The tempo estimator uses the median of the most recent `3..5` beat intervals.

## 3) Microphone Clap Onset Input

Install dependencies from `requirements.txt`, then run:

```powershell
python realtime/robot_arm/src/adaptive_play_trajectory.py --csv realtime/robot_arm/outputs/clap_trajectory.csv --input-source mic --realtime
```

Useful options:

```powershell
python realtime/robot_arm/src/adaptive_play_trajectory.py `
  --csv realtime/robot_arm/outputs/clap_trajectory.csv `
  --input-source mic `
  --beats-per-cycle 2 `
  --speed-min 0.5 `
  --speed-max 1.8 `
  --mic-threshold-scale 3.5
```

## Notes

- The adaptive player assumes the input trajectory is loopable.
- `beats-per-cycle` defines how many detected beats map to one full trajectory cycle.
- In `mic` mode, `sounddevice` must be available and a microphone must be accessible.

## 4) Realtime Music-Adaptive Playback

Use this player when the robot should clap along with continuous live music rather than only manual taps or isolated claps:

```powershell
python realtime/robot_arm/src/realtime_music_adaptive_player.py --csv realtime/robot_arm/outputs/clap_trajectory.csv --realtime
```

The realtime player defaults to built-in clap motion and does not require a trajectory CSV:

```powershell
python realtime/robot_arm/src/realtime_music_adaptive_player.py --motion clap --realtime
```

To use the older figure-eight motion instead:

```powershell
python realtime/robot_arm/src/realtime_music_adaptive_player.py --motion figure-eight --realtime
```

To drive motion directly from a trajectory CSV, use:

```powershell
python realtime/robot_arm/src/realtime_music_adaptive_player.py --motion csv --csv realtime/robot_arm/outputs/clap_trajectory.csv --realtime
```

It listens through the microphone and adapts several motion elements at once:

- librosa PLP beat timing controls playback speed and phase alignment
- RMS loudness controls motion amplitude
- strong onsets trigger short pitch/yaw accent motions
- spectral brightness slightly changes accent sensitivity
- silence/noise keeps amplitude at zero so the robot does not move

Useful options:

```powershell
python realtime/robot_arm/src/realtime_music_adaptive_player.py `
  --csv realtime/robot_arm/outputs/clap_trajectory.csv `
  --beats-per-cycle 2 `
  --speed-min 0.5 `
  --speed-max 1.8 `
  --amp-min 0.65 `
  --amp-max 1.45 `
  --motion clap `
  --accent-gain-deg 10 `
  --rhythm-method plp `
  --plp-history-sec 8.0 `
  --plp-analysis-interval-sec 0.10 `
  --plp-hop-length 256 `
  --plp-peak-prominence 0.15 `
  --onset-threshold-scale 3.0 `
  --noise-gate-rms 0.002 `
  --noise-gate-ratio 1.8 `
  --startup-calibration-sec 1.0 `
  --realtime
```

Noise handling:

- The first `startup-calibration-sec` seconds estimate the room/microphone noise floor from live input.
- Keep the room quiet during that startup window, then start music after calibration finishes.
- Input is treated as music only when RMS is above both `noise-gate-rms` and `noise-gate-ratio * startup_noise_floor`.
- When input is below the gate, BPM is cleared, onset detection is ignored, and motion amplitude smooths to `0`.

Rhythm tracking:

- `--rhythm-method plp` is the default. It analyzes a rolling microphone window with `librosa.onset.onset_strength` and `librosa.beat.plp`, then finds recent pulse peaks for beat events.
- `--rhythm-method energy` keeps the older high-pass energy onset detector as a lightweight fallback.
- Heavier future options include madmom RNN+DBN, Essentia `RhythmExtractor2013` for offline files, and BeatNet for neural real-time beat/downbeat/meter tracking.

Controls:

- `R`: reset realtime analyzer and motion controller
- `Q`: quit
- `Esc`: also quits on PyBullet builds that expose `B3G_ESCAPE`

In `--motion csv` mode, the trajectory CSV still only requires `t`, `yaw_rad`, and `pitch_rad`. If available, the realtime player also reads:

- `phase`: normalized loop phase for each row
- `keypoint`: marks poses that should land on musical beats
- `keypoint_label`: optional readable label such as `open` or `clap`
- `beat_weight`: optional importance weight for beat-aligned poses

Run `generate_clap_trajectory.py` again to create a sample CSV with these optional keypoint columns.
