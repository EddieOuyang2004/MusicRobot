# Beats2Trajectory

This repository is organized around two project modes:

- `offline/`: the completed offline music-to-trajectory pipeline. This area is frozen and should only be changed for archival fixes.
- `realtime/`: active realtime robot-control work, split by robot target.

## Project Layout

```text
Beats2Trajectory/
  offline/
    audio/
    outputs/
    src/
    README.md
  realtime/
    robot_arm/
      noise_profiles/
      outputs/
      src/
      README.md
    humanoid_robot/
      README.md
  requirements.txt
```

## Start Here

All commands in the subproject READMEs are intended to be run from the repository root.

- Offline pipeline: [offline/README.md](offline/README.md)
- Realtime PyBullet robot arm: [realtime/robot_arm/README.md](realtime/robot_arm/README.md)
- Realtime MuJoCo humanoid robot: [realtime/humanoid_robot/README.md](realtime/humanoid_robot/README.md)
