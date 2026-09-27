# Archived workflows

These examples are retained for reference and archival fixes. Active development
uses the [humanoid matcher](../realtime/humanoid_robot/README.md).

Run all commands from the repository root. Install the additional PyBullet dependency:

```powershell
python -m pip install -r legacy/requirements.txt
```

- [Robot arm](robot_arm/README.md): PyBullet clap/trajectory examples. The music-adaptive player uses the shared realtime audio/controller module.
- [Offline pipeline](offline/README.md): frozen music-to-trajectory generation and playback, with its original audio and feature-mapping workbook.

Generated arm and offline outputs remain ignored. The workbook and its generator
remain versioned. No humanoid code depends on this directory.
