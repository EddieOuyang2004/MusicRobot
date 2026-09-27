# Offline Pipeline

Archived workflow. Run commands from the repository root and install
`python -m pip install -r legacy/requirements.txt`. Active work is documented in
the [humanoid guide](../../realtime/humanoid_robot/README.md).

This folder keeps the completed offline music-to-trajectory pipeline unchanged:

1. Extract `beat`, `RMS`, and `onset` features with `librosa`
2. Generate a continuous clap trajectory
3. Save trajectory targets to `CSV`
4. Play the trajectory in `PyBullet`, with optional synchronized audio playback

## Generate Trajectory CSV

```powershell
python legacy/offline/src/generate_trajectory.py legacy/offline/audio/your_song.mp3 --out legacy/offline/outputs/trajectory.csv
```

Optional tuning:

```powershell
python legacy/offline/src/generate_trajectory.py legacy/offline/audio/your_song.mp3 `
  --out legacy/offline/outputs/trajectory.csv `
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
ffmpeg -i legacy/offline/audio/your_song.mp3 -ac 2 -ar 44100 legacy/offline/audio/your_song.wav
```

## Play Trajectory in PyBullet

```powershell
python legacy/offline/src/play_trajectory_pybullet.py --csv legacy/offline/outputs/trajectory.csv --realtime
```

Common options:

```powershell
python legacy/offline/src/play_trajectory_pybullet.py --csv legacy/offline/outputs/trajectory.csv --realtime --loop --speed 1.2
python legacy/offline/src/play_trajectory_pybullet.py --csv legacy/offline/outputs/trajectory.csv --urdf your_robot.urdf --joint-yaw 0 --joint-pitch 1 --fixed-base
```

## Play Trajectory with Audio Sync

```powershell
python legacy/offline/src/play_trajectory_pybullet.py --csv legacy/offline/outputs/trajectory.csv --audio legacy/offline/audio/your_song.wav --realtime
```
