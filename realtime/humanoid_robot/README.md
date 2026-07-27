# Realtime Humanoid Robot

MuJoCo realtime humanoid dancer driven by the same microphone music features used by
`realtime/robot_arm/`.

The default motion is the checked-in AIST++ pickle at
`realtime/humanoid_robot/data/aistpp/motions/gWA_sBM_cAll_d26_mWA0_ch07.pkl`.
It plays on the included G1 MJCF scene in `assets/open_humanoid_dancer.xml`,
with robot meshes under `assets/meshes/`.

## Run

From the repository root:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --realtime
```

For a non-GUI smoke test:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --headless --no-mic --max-seconds 2
```

To preview the authored dance trajectory at normal speed without microphone or
music adaptation:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --preview-trajectory --realtime
```

To choose a different downloaded BVH clip:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --motion-source bvh --bvh-motion realtime/humanoid_robot/data/lafan1_dance/dance2_subject5.bvh --preview-trajectory --realtime
```

The BVH path is retargeted into GMR's Unitree G1 29-DoF order internally. Use
`--bvh-neutral-frame` if the first frame is not a good neutral pose, and
`--bvh-use-music-amplitude` if you want live music amplitude and beat accents to
scale the retargeted motion.

To choose a different AIST++ pickle:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --motion-source aistpp --aistpp-motion path/to/motion.pkl --preview-trajectory --realtime
```

## AIST++ Velocity-Valley Keypoints

The standalone detector finds dance key poses at prominent local minima of the
whole-body velocity curve:

```powershell
python realtime/humanoid_robot/src/aistpp_velocity_keypoints.py realtime/humanoid_robot/data/aistpp/motions/gWA_sBM_cAll_d26_mWA0_ch07.pkl --max-count 12
```

Use `--json` for machine-readable stdout, or save the frame, timestamp, phase,
score, prominence, and velocity of every keypoint:

```powershell
python realtime/humanoid_robot/src/aistpp_velocity_keypoints.py path/to/motion.pkl --output-json outputs/keypoints.json
```

If a pickle contains 3D joint positions under `keypoints3d`, `joints3d`,
`smpl_joints`, or `joint_positions`, the detector uses their whole-body kinetic
velocity directly. Standard AIST++ SMPL pickles in this repository do not
contain those positions, so it uses geodesic angular velocity from all 24 SMPL
joints plus scale-corrected root translation.

The realtime dancer's `--keypoint-mode auto` selects this detector for AIST++
motions. It can also be requested explicitly:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --keypoint-mode aist-velocity --realtime
```

The main tuning controls are `--keypoint-prominence`,
`--keypoint-min-spacing-sec`, `--keypoint-smoothing-sec`, and
`--keypoint-max-count`.

For higher quality Unitree G1 retargeting, convert AIST++ SMPL pickles to a
GMR-readable LaFAN-style BVH first, then let GMR do the robot retargeting. The
converter needs optional dependencies and licensed SMPL model files that are not
committed to this repository:

```powershell
pip install torch smplx[all]
```

Place the SMPL body model files under:

```text
realtime/humanoid_robot/assets/body_models/smpl/
```

For a one-file smoke conversion:

```powershell
python realtime/humanoid_robot/src/aistpp_to_gmr_bvh.py --split train --limit 1 --overwrite
```

To convert all AIST++ split files:

```powershell
python realtime/humanoid_robot/src/aistpp_to_gmr_bvh.py --split all --overwrite
```

Then, from a GMR checkout/environment, retarget one generated BVH to Unitree G1:

```powershell
python scripts/bvh_to_robot.py --bvh_file path/to/converted.bvh --robot unitree_g1 --save_path path/to/gmr_unitree_g1.pkl --format lafan1 --motion_fps 60
```

Load the GMR output with the existing GMR pickle path:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --motion-source gmr-pkl --gmr-motion path/to/gmr_unitree_g1.pkl --preview-trajectory --realtime
```

To compare against the old procedural side-step motion:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --motion-source procedural --preview-trajectory --realtime
```

To play a Unitree G1 motion pickle exported by
[GMR: General Motion Retargeting](https://github.com/YanjieZe/GMR), first use GMR
to retarget your source motion with `--robot unitree_g1 --save_path ...`, then
load the resulting pickle here:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --motion-source gmr-pkl --gmr-motion path/to/gmr_unitree_g1.pkl --preview-trajectory --realtime
```

GMR pickles are expected to contain `fps`, `root_pos`, `root_rot`, and `dof_pos`.
The `dof_pos` array is read in GMR's Unitree G1 29-DoF order and mapped onto the
actuator names present in the loaded MuJoCo model. By default, GMR joint angles
are preserved exactly apart from `--pose-gain`; add `--gmr-use-music-amplitude`
if you want live music amplitude and beat accents to scale the GMR motion.

The default model is already the official 29-DoF G1 scene, so the compact AIST++
and procedural samplers are automatically expanded into G1 actuator targets:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --target-robot unitree-g1 --realtime
```

If you are using the checked-in virtual environment:

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --realtime
```

## Music Retrieval and Automatic Motion Switching

`realtime_music_humanoid_matcher.py` is a separate entrypoint that keeps
`realtime_music_humanoid_dancer.py` unchanged. It analyzes a six-second rolling
music window, retrieves matching AIST++ music, ranks only motions that passed
catalog preflight, and changes motion on a stable four-beat boundary.

Build the catalog after downloading the synchronized AIST++ audio:

```powershell
python realtime/humanoid_robot/src/build_aistpp_music_catalog.py
```

The default catalog uses a local, gain-invariant DSP embedding. To build with
Windows CPU ONNX Runtime, place the official dynamic-batch Discogs EffNet model
and its same-named JSON metadata under `realtime/humanoid_robot/models/`. Passing
the same model for both flags runs it once per window and reads both its
1280-dimensional embedding and 400 explainable style activations:

```powershell
python realtime/humanoid_robot/src/build_aistpp_music_catalog.py `
  --embedding-model realtime/humanoid_robot/models/discogs-effnet-bsdynamic-1.onnx `
  --tag-model realtime/humanoid_robot/models/discogs-effnet-bsdynamic-1.onnx
```

Use the microphone:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py --realtime
```

Use `--matcher-help` for retrieval/switching options and `--help` for the
inherited dancer/controller options. A hardware-free silent smoke test keeps the
current motion without attempting retrieval:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py `
  --headless --no-mic --max-seconds 1
```

Evaluate segment retrieval, leave-one-music genre retrieval, and gain
invariance:

```powershell
python realtime/humanoid_robot/src/evaluate_music_catalog.py
```

Use a synchronized WAV for a deterministic test:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py `
  --audio-input realtime/humanoid_robot/data/aistpp/audio/gBR_sBM_cAll_d04_mBR0_ch01.wav `
  --headless --realtime --max-seconds 10
```

Absolute microphone RMS and LUFS are not retrieval features. RMS is retained
only for the noise gate, silence handling, and live pose amplitude. Catalog and
query audio are DC-removed and robustly gain-normalized before extracting
embedding, rhythm, timbre, and optional tag probabilities.

## Current Feature Mapping

The mapping follows `offline/outputs/music_feature_motion_mapping.xlsx`:

- Beat period and PLP beat events retime the dance cycle and align AIST++ motions
  to detected velocity-valley key poses (or configured fixed phases).
- RMS loudness controls full-body amplitude and fades motion toward silence.
- Onsets and beat confidence produce short pose accents.
- Brightness raises arm height and sharpens beat accents.
- Low-frequency energy increases hip sway, knee bounce, and grounded weight shifts.
- Mid-frequency energy increases torso twist.
- High-frequency energy adds faster arm/elbow texture.
- Rhythm density and offbeat ratio add extra upper-body texture on subdivisions.

## Next Extension Points

- Improve the BVH-to-G1 retargeting signs and gains for additional skeleton
  conventions beyond the checked-in LaFAN-style clips.
- Swap `assets/open_humanoid_dancer.xml` for a target humanoid MJCF and keep the
  feature-to-pose layer if actuator names are mapped.
- Add balance control by moving from fixed-torso dancing to foot contacts and
  root motion.
