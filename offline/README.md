# Offline Pipeline

This folder keeps the completed offline music-to-trajectory pipeline unchanged:

1. Extract `beat`, `RMS`, and `onset` features with `librosa`
2. Generate a continuous clap trajectory
3. Save trajectory targets to `CSV`
4. Play the trajectory in `PyBullet`, with optional synchronized audio playback

## Generate Trajectory CSV

```powershell
python offline/src/generate_trajectory.py offline/audio/your_song.mp3 --out offline/outputs/trajectory.csv
```

Optional tuning:

```powershell
python offline/src/generate_trajectory.py offline/audio/your_song.mp3 `
  --out offline/outputs/trajectory.csv `
  --traj-hz 100 `
  --yaw-amp-deg 18 `
  --open-pitch-deg 22 `
  --clap-pitch-deg -24 `
  --bob-deg 5 `
  --accent-duration 0.15 `
  --accent-gain-deg 14 `
  --onset-quantile 0.9
```

The generated motion is an open-clap-open loop. RMS loudness scales the clap amplitude, strong onsets add a short accent, and the CSV includes `open` / `clap` keypoint metadata for beat-aligned playback.

## Convert MP3 to WAV

```powershell
ffmpeg -i offline/audio/your_song.mp3 -ac 2 -ar 44100 offline/audio/your_song.wav
```

## Play Trajectory in PyBullet

```powershell
python offline/src/play_trajectory_pybullet.py --csv offline/outputs/trajectory.csv --realtime
```

Common options:

```powershell
python offline/src/play_trajectory_pybullet.py --csv offline/outputs/trajectory.csv --realtime --loop --speed 1.2
python offline/src/play_trajectory_pybullet.py --csv offline/outputs/trajectory.csv --urdf your_robot.urdf --joint-yaw 0 --joint-pitch 1 --fixed-base
```

## Play Trajectory with Audio Sync

```powershell
python offline/src/play_trajectory_pybullet.py --csv offline/outputs/trajectory.csv --audio offline/audio/your_song.wav --realtime
```
