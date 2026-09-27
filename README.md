# Beats2Trajectory

Realtime music-to-motion matching for the Unitree G1 humanoid in MuJoCo.
The supported matcher uses the authored-transition implementation formerly called v2,
with a combined FineDance/AIST++ music catalog.

## Start here

Run commands from the repository root, using the project Python environment:

```powershell
python -m pip install -r requirements.txt
```

Prepare the source datasets, validated GMR motions, and local embedding model using
the [combined catalog guide](docs/finedance_matcher_integration.md). Then build or
resume the catalog and start microphone-driven matching:

```powershell
python realtime/humanoid_robot/src/build_combined_music_catalog.py
python realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py --realtime
```

To split motions longer than 16 seconds into paired clips around 8–10 seconds,
see the [segmentation and catalogue organization guide](docs/segmented_music_catalog.md).
Its resumable command validates a candidate before activating it as the default.

For file input:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py `
  --audio-input "realtime/humanoid_robot/data/test_audio/Metronome 120 BPM - QuickSounds.com.mp3" `
  --play-audio --realtime
```

Use `--matcher-help` for matching/transition options and `--help` for audio,
controller, and viewer options. See the [humanoid guide](realtime/humanoid_robot/README.md)
for retargeting, standalone motion playback, and diagnostics. MuJoCo playback is a
kinematic preview; it does not send commands to physical hardware.

## Project layout

```text
realtime/
  humanoid_robot/   # Matcher, retargeting tools, assets, catalogs, and runtime tests
  shared/          # Audio analysis, beat estimation, and adaptive motion timing
tests/             # Cross-module regression tests
docs/              # Dataset guides, paper, and thesis
output/            # Retained research results and reference media
legacy/            # Archived robot-arm and offline workflows
tmp/               # Local scratch work, tools, and caches
requirements.txt   # Humanoid/runtime dependencies
```

## Validation

```powershell
python -m unittest discover -s tests
python -m unittest discover -s realtime/humanoid_robot/src/test -p "test_*.py"
python realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py --headless --no-mic --max-seconds 1
```

The smoke run requires the local catalog and referenced assets. Experiment protocol 2
uses the canonical matcher; start a new output directory when migrating from v1.
[Experiment tools](realtime/humanoid_robot/src/test/README.md) retain historical result readers.

## Historical workflows

The [legacy archive](legacy/README.md) contains the frozen offline pipeline and
PyBullet robot-arm examples. Their extra dependencies are in `legacy/requirements.txt`.
