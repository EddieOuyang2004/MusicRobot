# Realtime Humanoid Robot

MuJoCo realtime humanoid dancer driven by the same microphone music features used by
`realtime/robot_arm/`.

The default motion is the checked-in AIST++ pickle at
`realtime/humanoid_robot/data/aistpp/motions/gWA_sBM_cAll_d26_mWA0_ch07.pkl`.
It plays on the included G1 MJCF scene in `assets/open_humanoid_dancer.xml`,
with robot meshes under `assets/meshes/`.

## Generate GMR v2 motions

The separate collision-projection batch builder regenerates existing GMR clips from
SMPL into `data/aistpp_gmr_v2`, with resume support and per-clip validation.
See [batch generation and playback instructions](../../docs/gmr_v2_batch.md).

## Run

From the repository root:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --realtime
```

Use an MP3 or WAV under `data/test_audio` as a realtime virtual microphone:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py `
  --audio-input "realtime/humanoid_robot/data/test_audio/Metronome 120 BPM - QuickSounds.com.mp3" `
  --play-audio `
  --realtime
```

The virtual microphone resets the analyzer when it opens and sends one second
of silence before the file starts, allowing startup denoising/calibration to
settle. Change this with `--audio-input-delay-sec` if needed. `--play-audio`
sends the same blocks to the default output device so heard audio remains
synchronized with virtual-microphone analysis.

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

The AIST++ path now requires a same-named pre-retargeted GMR artifact under
`data/aistpp_gmr/` by default. If it is absent, the program stops and prints the
exact generation command. The handwritten mapper remains available only as the
explicit diagnostic mode `--retarget-policy direct`; `prefer-gmr` is retained
for comparison experiments. Override the artifact directory with
`--gmr-motion-root`.

Floating-base motion is enabled by default. `--root-motion continuous` anchors
each clip to the current pelvis pose and preserves displacement across loops and
matcher switches. Use `in-place` to keep horizontal position fixed or `reset`
to re-anchor each loop. Music uses bounded `subtle` pose modulation by default;
it does not scale the authored pose or change the root trajectory. Use
`--pose-modulation-mode expressive` for the legacy amplitude-driven behavior,
or `off` for an unmodified retargeted trajectory.

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

By default, the humanoid entrypoints use adaptive strong-beat alignment. Each
detected beat receives a causal contrast score from its local RMS loudness and
onset impact relative to the most recent beats. When the raw beat rate is fast
enough to exceed `--speed-max` for the next authored keypoint interval, weak
beats are skipped and stronger beats are preferred; at normal tempos, every
confidence-qualified beat remains eligible. The controller derives motion speed
from accepted alignment beats rather than from skipped subdivisions.

Use `--beat-selection-mode every` to restore confidence-only beat alignment, or
adjust the confidence/contrast balance with `--beat-contrast-weight` (default
`0.5`). `--beat-confidence-threshold` remains a hard noise-rejection gate.

## Formal AIST++ -> GMR -> G1 preparation

The licensed archive is expected at
`assets/SMPL_python_v.1.1.0.zip`. Its recorded SHA-256 is
`87C9E6CB1DDDAD79CF3B8B01760E240A9CD0C29D7FF14EDA30FB07E3BD430C27`.
Only the female, neutral, and male model pickles are read from the archive. The
preparation script replaces the legacy Chumpy `shapedirs` object with a NumPy
array, writes the three filenames expected by `smplx`, validates 6890 vertices,
24 joints and the kinematic tree, then performs a neutral T-pose forward pass:

```powershell
.\realtime\humanoid_robot\.venv-gmr\Scripts\python.exe `
  realtime/humanoid_robot/src/prepare_smpl_models.py
```

The resulting ignored files are:

```text
assets/body_models/smpl/SMPL_FEMALE.pkl
assets/body_models/smpl/SMPL_NEUTRAL.pkl
assets/body_models/smpl/SMPL_MALE.pkl
```

This workspace uses the ignored official GMR checkout at `.deps/GMR`, commit
`bb1bbe40774794fceb2a7c579a3464a28e68c844`, and an isolated Python environment.
To recreate the environment, use Python 3.11 and the pinned dependency file.
GMR defaults to DAQP; this avoids the official `proxqp` extra attempting a local
C++/NMake build on Windows:

```powershell
python -m venv realtime/humanoid_robot/.venv-gmr
.\realtime\humanoid_robot\.venv-gmr\Scripts\python.exe -m pip install `
  -r realtime/humanoid_robot/requirements-gmr.txt
git clone https://github.com/YanjieZe/GMR realtime/humanoid_robot/.deps/GMR
```

Generate one artifact with the headless runner:

```powershell
.\realtime\humanoid_robot\.venv-gmr\Scripts\python.exe `
  realtime/humanoid_robot/src/build_aistpp_gmr_dataset.py `
  --gmr-root realtime/humanoid_robot/.deps/GMR `
  --gmr-python realtime/humanoid_robot/.venv-gmr/Scripts/python.exe `
  --motion realtime/humanoid_robot/data/aistpp/motions/gWA_sBM_cAll_d26_mWA0_ch07.pkl
```

Generate every split artifact and `data/aistpp_gmr/manifest.json` with:

```powershell
.\realtime\humanoid_robot\.venv-gmr\Scripts\python.exe `
  realtime/humanoid_robot/src/build_aistpp_gmr_dataset.py `
  --gmr-root realtime/humanoid_robot/.deps/GMR `
  --gmr-python realtime/humanoid_robot/.venv-gmr/Scripts/python.exe `
  --split all --jobs 1 --overwrite
```

An existing artifact is reused only when its source hash, neutral SMPL model
hash, schema, and full GMR commit still match. `--overwrite` forces regeneration.
A single-motion run merges its result into the existing manifest; it never
removes unrelated valid entries. A complete, unlimited `--split all` run
rebuilds the manifest from the selected 411 source motions. Failed motions are
removed from the valid manifest and recorded in `data/aistpp_gmr/failures.json`.
If a same-named artifact is stale or malformed, it is moved to a recoverable
`.invalid` filename before regeneration so the Matcher cannot load it by name.
The official GUI script is deliberately not used: the repository's headless
runner calls the GMR API once for the whole sequence and never opens one MuJoCo
window per file.

### What happens to every frame

1. The AIST++ pickle supplies `smpl_poses[N,72]` axis-angle rotations,
   `smpl_trans[N,3]`, and optional `smpl_scaling`. Rotations remain the local
   24-joint SMPL skeleton; root translation becomes meters with
   `smpl_trans / smpl_scaling`.
2. The neutral licensed model supplies the rest joint locations and parents.
   Local rotations are composed through the complete 24-joint kinematic tree,
   then world positions and rotations are converted from SMPL Y-up to MuJoCo
   Z-up. The production path sends the required pelvis, spine, leg and arm
   world transforms directly to GMR; it does not serialize or reload BVH.
3. GMR runs with `src_human="smplx"`, the official `smplx_to_g1` configuration,
   and a fixed `1.75 m` human-height assumption. This is important because that
   configuration uses distinct left/right shoulder, elbow and wrist offsets.
   The runner also appends `mink.CollisionAvoidanceLimit` for non-adjacent G1
   arms, torso, pelvis and legs. It maintains `5 mm` separation once a pair is
   within `8 cm`; adjacent mechanical links are excluded because their meshes
   overlap at the joint by design. Override these defaults with
   `--collision-min-distance`, `--collision-detection-distance`, and
   `--collision-gain`. `--no-collision-avoidance` is intended only for A/B
   diagnostics.
4. The headless runner reads joint names in actual MuJoCo `jnt_qposadr` order,
   rather than guessing an actuator/column order. Since GMR may do several IK
   iterations for one source frame, final qpos continuity is also constrained
   at the source FPS (joint `3π rad/s`, root `3 m/s`, root rotation `4π rad/s`).
   The final trajectory is checked twice with 40 samples per source-frame
   segment in both the GMR and Preview models. A bounding-sphere-prefiltered
   `mj_geomDistance` fallback keeps shallow-contact results consistent between
   the MuJoCo versions used by those two environments; an initially colliding
   frame is backed off toward the neutral joint pose.
5. The canonical v3 pickle stores `format_version=1`, `pipeline_version=3`,
   `source_format="aistpp_smpl_direct"`, `fps`, `root_pos[N,3]` metres,
   `root_rot[N,4]`, explicit `root_rot_order="wxyz"`, `dof_pos[N,29]` radians,
   `dof_names`, hashes, source ID, and the exact GMR commit. Root quaternion
   signs are continuous and no generation-time loop closure is applied. The
   exact collision preset, distances, gain and expanded geom-pair count are
   recorded under `collision_avoidance`. `mink_limits_api` records the adapter
   used to pass GMR's legacy positional limit list into Mink 1.3's keyword-only
   `limits` slot; without this adapter, upstream GMR silently passes that list
   as `safety_break` instead. Legacy v1/v2/BVH artifacts and ambiguous
   quaternion arrays are rejected.

`aistpp_to_gmr_bvh.py` and `gmr_retarget_bvh_headless.py` remain available for
FK diagnostics and regression comparisons only. Their v1 output is not a
canonical runtime artifact, and `.deps/GMR` is never modified.

### How MuJoCo playback works

At playback time the sampler converts controller phase to a fractional source
frame. It linearly interpolates root position and all 29 joint angles, and uses
quaternion SLERP for the root rotation. Initial heading removal rotates both
orientation and displacement into the same frame. `root-motion=continuous`
accumulates the complete horizontal path across loops; action switches blend
root position, root quaternion, and all joints.

`MujocoHumanoidPlayer` then writes root position plus scalar-first quaternion to
the free joint's seven qpos values, and writes the 29 named hinge values to their
joint qpos addresses. It clears `qvel` and `ctrl`, calls `mj_forward` (not
`mj_step`), then synchronizes the viewer. This is therefore a kinematic preview,
not torque control or balance simulation. A constant root-Z grounding offset is
computed from all eight foot support spheres, and the viewer camera tracks the
`pelvis` body so full multi-metre choreography stays in view.

Beat alignment no longer assigns a new phase on a beat. The controller holds a
phase error and consumes it gradually while the effective speed remains between
`speed_min` and `speed_max`, preventing the root and all joints from teleporting
together.

Run the retargeting audit for every locally available source-video sample with
the isolated GMR environment:

```powershell
.\realtime\humanoid_robot\.venv-gmr\Scripts\python.exe `
  realtime/humanoid_robot/src/audit_aistpp_retargeting.py
```

For each sample this writes `report.json`, compressed `layers.npz`, and a
self-contained synchronized `comparison.html` under
`data/aistpp_gmr/diagnostics/<motion-id>/`. Raw axis-angle jumps are candidates,
not failures: the report labels them `parameterization_only` when SMPL world FK
remains continuous. Actual discontinuities are attributed to `source_smpl`,
`bvh`, or `gmr_g1` with the exact frame, joint, velocity, and cross-layer error.
BVH is an optional diagnostic layer: a missing same-named BVH does not fail the
canonical audit. The report also compares nine SMPL/G1 bone directions on every
frame, records median/minimum cosine, negative-frame fraction, worst bone/frame,
and left/right quality gaps, and fails continuous but directionally wrong poses.
Pass one or more `--motion path/to/motion.pkl` arguments to audit motions that do
not have a local source video; add `--no-video-samples --limit N` to audit a
bounded dataset subset without generating comparison pages.

Load the GMR output with the existing GMR pickle path:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --motion-source gmr-pkl --gmr-motion path/to/gmr_unitree_g1.pkl --preview-trajectory --realtime
```

To compare against the old procedural side-step motion:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py --motion-source procedural --preview-trajectory --realtime
```

Canonical GMR pickles must use the v2 SMPL-direct schema described above and
contain explicit `dof_names`. The names must cover the canonical Unitree G1
29-DoF set exactly; generic upstream exports, old BVH artifacts, unnamed files,
and ambiguous files are rejected. GMR joint angles
are always preserved at authored scale, so `--pose-gain` and the legacy
`--gmr-use-music-amplitude` option are ignored for GMR sources. The player scans
all authored frames once and adds one constant root-Z offset so the lowest of
the eight spherical foot supports touches the MuJoCo ground plane.

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
At startup it randomly chooses a sufficiently long motion from the lowest-activity
quartile of the catalog. Use `--initial-motion-seed` for a reproducible choice,
`--initial-motion-low-activity-quantile` to tune the pool, or
`--initial-motion-id` to request an exact catalog motion.

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

For playback-speed diagnosis, `--motion-timing authored` keeps music retrieval,
beat accents, bar-boundary switching, and motion blending enabled while advancing
every source and target motion at its authored `1.0x` rate. It prevents detected
beats and motion keypoints from correcting phase or changing speed. Use the same
`--initial-motion-seed` in authored and default `beat-sync` runs for a repeatable
A/B comparison. Add `--disable-music-modulation` only for a second, stricter pass
that also removes music-driven pose modulation.

The default `--match-policy style-first` first selects a confident AIST++ genre
family, then ranks only that family's motions. Weak beat evidence and strong
ambient/non-music tags reject the retrieval and hold the current motion instead
of forcing a dance. `--match-policy legacy` is available for A/B diagnosis, and
`--match-weak-music-threshold` tunes the ambient/non-music gate. Trace CSV files
include the decision state, confidence, rejection reason, genre ranking, motion
ranking, audio evidence, and visual motion-cluster IDs.

The matcher separates relevance-driven changes from diversity rotation. A
clearly better motion still wins after the configured consecutive retrievals;
when music remains stable, the default policy changes after four held bars to a
stable, least-recently-used motion within `0.05` of the best total score and
`0.08` of the best music score. Tune this with
`--switch-max-hold-bars`, `--switch-diversity-top-k`,
`--switch-diversity-score-drop`, `--switch-diversity-music-score-drop`, and
`--switch-recent-history`. Set `--switch-max-hold-bars 0` to disable forced
diversity rotation.
The recency policy prefers a motion from a visual activity/density/regularity
cluster that has not recently played before considering another motion from the
same cluster.

Startup motion selection reserves enough authored duration for the rolling
match window, consecutive wins, maximum playback speed, and background motion
preparation. Tune the preparation allowance with
`--startup-ready-reserve-seconds` (default `4`). If a one-shot motion still
reaches its terminal frame before a candidate is ready—or the current audio is
rejected as ambient/non-music—the matcher applies a small stationary breathing
idle instead of displaying a completely frozen pose. Tune it with
`--terminal-safe-idle-amplitude` or set that value to `0` to disable it.

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py `
  --headless --no-mic --max-seconds 1
```

Evaluate segment retrieval, leave-one-music genre retrieval, and gain
invariance:

```powershell
python realtime/humanoid_robot/src/evaluate_music_catalog.py
```

Use an MP3 or WAV as a realtime virtual microphone for the matcher:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py `
  --audio-input "realtime/humanoid_robot/data/test_audio/Metronome 120 BPM - QuickSounds.com.mp3" `
  --play-audio `
  --headless --realtime --max-seconds 10
```

Run the same matcher and switching path at authored speed:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py `
  --audio-input "realtime/humanoid_robot/data/test_audio/Metronome 120 BPM - QuickSounds.com.mp3" `
  --motion-timing authored `
  --initial-motion-seed 0 `
  --trace-csv realtime/humanoid_robot/src/test/output/authored_timing.csv `
  --play-audio --realtime --max-seconds 30
```

Matcher traces include the timing mode, source and transition speed multipliers,
remaining phase correction, and raw/output wrist angle and speed maxima. These
fields distinguish source-motion saturation from beat-driven changes and
transition-blend spikes.

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

## Humanoid Matcher V2

`src/realtime_music_humanoid_matcher_v2.py` is a separate copy of the matcher
with end-only switching. The original matcher remains available for comparison.

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_matcher_v2.py --realtime
```

V2 starts retrieval after two seconds of audio, requests updates approximately
once per second, and uses equally weighted available history up to 30 seconds.
The live retrieval buffer is separate from the beat analyser's shorter buffer.
It keeps the existing genre-first matching and confidence gates, then shortlists
up to five musically suitable clips. Kinematic cost chooses the clip and entry
frame jointly from every eligible frame in that shortlist. Cost combines joint
position, velocity, foot-contact, and root continuity, without music salience.
Entries must leave at least four seconds at maximum playback speed (1x in
authored timing mode).

Hermite limit checking uses sampled rejection, Bernstein bounds and at most
three subdivision levels, followed by the original polynomial extrema check for
uncertain joints. It keeps the same duration trials and boundary states. When
Numba is installed, these bounds run as cached native code with the GIL released;
v2 warms the kernel before playback. A Python fallback works without Numba.
This improves computation time but does not make infeasible transitions feasible.
Use `src/test/benchmark_hermite_checks.py` with captured JSONL inputs to compare
the optimized checker against the original extrema checks.

The default `--transition-backend hermite` prepares a general quintic bridge
matching authored joint position, velocity and acceleration at both endpoints.
The shortlist is frozen at clip entry (the first retrieval at startup), and the
first feasible bridge is committed on a background worker. Later music affects
the following selection. Planning does not predict music or include modulation.
Cached piecewise quintics provide the same authored states during playback.

Music effects and playback speed fade to neutral and 1x over the final
`--transition-boundary-seconds` (default 0.5 authored seconds), and fade back in
after entry to the next clip. The current clip reaches its last frame without
an extra terminal hold; elapsed time carries across both bridge boundaries.
Root translation and quaternion interpolation remain separate from joint-state
generation. Bridges last at least 0.35 seconds; Hermite searches up to 10 seconds
and checks analytic position, velocity and acceleration extrema.

`--transition-backend ruckig` uses optional local Ruckig state-to-state generation.
Install `requirements-ruckig.txt` and supply `--output-max-joint-jerk` in rad/s^3
or `max_jerk_rad_s3` values in the existing dynamics JSON's `default` / `joints`
objects. Every limited joint needs a jerk bound for Ruckig. Hermite also checks
jerk when configured. No hardware jerk defaults are assumed.

If ranked candidates are infeasible, the same backend tries replay to frame zero.
If replay is infeasible or preparation misses the terminal state, v2 holds the
terminal target through the existing output limiter and reports the reason.
That failure path remains held until restart; late results cannot splice a
moving-endpoint bridge onto a stationary hold. No limits are silently relaxed.
`--transition-backend quintic` retains the previous terminal recheck and
rest-to-rest interpolation for comparison.

Use `--history-max-seconds` (at most 30), `--analysis-min-seconds`,
`--entry-min-remaining-seconds`, `--shortlist-size`, `--shortlist-score-drop`,
`--shortlist-music-score-drop`, and `--transition-min-seconds` to tune v2.
V2 defaults to `--speed-max 1.3`. Its default entry minimum is eight seconds
of **authored motion** after the chosen entry, independent of playback speed.
At 1.3x, eight authored seconds take about 6.15 seconds to play; the entry
remains eligible. The trace's `entry_remaining_seconds` also uses authored time.
V1 early-switch, fixed-window, and maximum-transition-duration options are
rejected with an explanation. Use `--matcher-help` for the full option list.
Existing microphone/file input, beat-sync/authored timing, pose modulation,
root alignment, output limiting, and diagnostic recording remain available.

`--trace-csv` includes audio-history duration, shortlist IDs, terminal timestamps,
entry frame and component costs, interpolation progress/duration, preparation
delay, and replay reasons. New fields include backend, preparation status/failure,
clip generation, boundary-effect weight, planned q/v/a residuals and whether the
final emitted joint positions differ from the planned sample. Audio analysis
continues during interpolation. Output limiting or collision correction can
change the reference; planned continuity is not a hardware tracking guarantee.

```powershell
python -m unittest discover -s realtime/humanoid_robot/src/test -p test_humanoid_matcher_v2.py
python -m unittest discover -s realtime/humanoid_robot/src/test -p test_motion_bridges.py
python realtime/humanoid_robot/src/test/verify_humanoid_matcher_v2_trace.py path/to/v2_trace.csv
```

Long equal-weight history deliberately responds more slowly to musical changes.
The existing catalog still contains six-second reference segments. The initial
comparison and its limitations are recorded in
[`src/test/humanoid_matcher_v2_validation.md`](src/test/humanoid_matcher_v2_validation.md).
Kinematic cost measures continuity rather than physical energy or dynamic balance.
New backend checks and reproduction commands are in
[`src/test/authored_motion_bridges_validation.md`](src/test/authored_motion_bridges_validation.md).

## Next Extension Points

- Improve the BVH-to-G1 retargeting signs and gains for additional skeleton
  conventions beyond the checked-in LaFAN-style clips.
- Swap `assets/open_humanoid_dancer.xml` for a target humanoid MJCF and keep the
  feature-to-pose layer if actuator names are mapped.
- Add balance control by moving from fixed-torso dancing to foot contacts and
  root motion.
