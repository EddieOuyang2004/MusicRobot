# FineDance segmentation for the AIST++ motion pipeline

`realtime/humanoid_robot/src/segment_finedance.py` exports paired motion/audio
clips of **8–12 seconds at 60 FPS**, with a preferred duration of 10 seconds.
It uses the existing NumPy, SciPy, and SoundFile dependencies; no GPU or body
model download is needed for segmentation.

## Run

From the repository root, audit first if desired:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/segment_finedance.py `
  --dry-run --output-root tmp/finedance_plan
```

Export the dataset:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/segment_finedance.py
```

Defaults read `realtime/humanoid_robot/data/finedance` and write
`realtime/humanoid_robot/data/finedance_aistpp`. Override these with
`--input-root` and `--output-root`. The output directory must be absent or empty,
and separate from the input tree. Use different directories for an audit and an
export. Source files are never modified. `--limit 1` exports the first recording.
An interrupted export requires a new output directory; only `manifest.json`
with `export_complete: true` signifies successful completion.

To prefer 12-second clips while retaining the 8–12-second limits:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/segment_finedance.py `
  --target-seconds 12 --output-root realtime/humanoid_robot/data/finedance_12s
```

Segments are contiguous, nonoverlapping, and balanced around the target.
For example, 25 seconds becomes three clips of about 8.33 seconds.
Whenever possible, all frames in the common audio/motion interval are used.
An interval too short to partition legally retains its longest legal prefix:
15 seconds becomes one 12-second clip with a reported 3-second tail. Recordings
shorter than 8 seconds are excluded; there is no padding, looping, or speed change.

The 8–12-second rule is this export's requirement, not a universal property of
AIST++. The local AIST++ collection also contains shorter clips and long sequences.

## Compatibility checked

| Property | FineDance release / local data | Export contract |
|---|---|---|
| Motion rate | 30 FPS | Always 60 FPS; `--source-fps` describes the input only |
| Raw motion | `(N,315)`: XYZ translation + 52 × 6D rotations | AIST++ pickle with `smpl_poses: (N,72)` axis angles in radians |
| Other inputs | `(N,319)` with four contact flags; `(N,159)` with axis angles | Explicit layouts; other shapes rejected |
| Skeleton | SMPL-H: 22 body joints + 30 finger joints | Body joints 0–21 retained; terminal SMPL hand joints 22–23 neutral |
| Translation / scale | SMPL-family metre convention | `smpl_trans: (N,3)` metres, `smpl_scaling: (1,) = 1` |
| Coordinates | Y-up in the upstream renderer | Y-up retained; downstream `aistpp_smpl.py` performs its existing Y-to-Z conversion |
| Vertical offset | Upstream renderer adds 1.3 m to Y | Same default; `--root-y-offset 0` preserves source translation exactly |
| Audio | Paired WAVs, including 48 kHz stereo | 44,100 Hz stereo PCM-16, matching this repository's AIST++ downloader |
| Duration | Source timeline | 480–720 frames; duration is `N / 60`; WAV duration matches exactly |
| Splits | Source recording IDs | All clips of a source retain its split |

Rotations use FineDance/PyTorch3D's **first two matrix rows** convention.
Gram–Schmidt reconstruction rejects degenerate vectors. Conversion uses
quaternion SLERP for rotations and linear interpolation for translation at
`t = frame / 60`. It does not interpolate rotation-vector components or 6D values.
The final pose is held during the last source-frame interval: N frames at 30 FPS
occupy N/30 seconds although the last measured pose is at (N−1)/30.
The manifest also records `(N−1)/60` as each clip's authored duration,
matching the distinction used by the robot playback code.

Audio is resampled once per recording and cut at the same start/end times as
motion. At 44.1 kHz there are exactly 735 samples per 60-FPS motion frame.
Mono is duplicated to stereo. Horizontal position and heading are preserved;
segments are not independently recentered.

**This is an approximate body-pose transfer, not an exact SMPL-H-to-SMPL mesh
conversion.** Finger articulation is omitted; source files retain it. SMPL-H
joints 22 and 23 are fingers and must not be treated as SMPL hand terminals.
The neutral SMPL body shape and root/rest-joint differences can alter geometry.
The 1.3 m offset follows the upstream visualization convention and is not a
fitted floor correction. Validate floor contact and retargeted motion before
robot playback. No fabricated `smpl_loss`, cameras, or 3D keypoints are exported.

Every exported pickle is read back through this project's
`aistpp_smpl.load_aistpp_motion`, and each saved WAV's length, rate, and channels
are verified. The three motion keys also match the official AIST++ `load_motion`
contract. This does not supply the cameras/environment files needed to construct
the full official multi-view dataset loader.

## Timing checks and dataset splits

Paired motion and audio are assumed to start at the same time, following the
upstream slicing code. Duration checks detect truncation but cannot prove beat
alignment or detect equal-duration offsets. No automatic time shift is inferred.

By default, audio/motion duration differences exceeding **0.1 seconds** are
excluded and listed in `plan.json` / `manifest.json`. Smaller differences use
only the common interval, rounded down to a whole motion frame. All discarded
durations are recorded. After inspecting problematic recordings, enable explicit
truncation with:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/segment_finedance.py `
  --allow-duration-mismatch --output-root realtime/humanoid_robot/data/finedance_allow_mismatch
```

This option does not repair synchronization. It accepts the shorter shared
timeline and marks the mismatch. Alternatively adjust
`--max-duration-mismatch-seconds` for an inspected dataset version.

Default membership follows upstream `cross_genre`, with **ignore taking
precedence** over test where the upstream lists overlap (120 and 130).
No validation set is invented; `val.txt` is empty. Override using
`--split-json path/to/splits.json` with `train`, `val`, `test`, and optionally
`ignore` lists of original source IDs such as `"001"`. All local source IDs must
be assigned exactly once; duplicates and missing assignments are rejected.
This preserves recording-level membership, not dancer-disjoint evaluation or
a guarantee against songs duplicated under different IDs. Supply your own
grouped splits for those requirements.

Precomputed `music_npy` and contact features are not copied: their frame rate
and windowing must be reconciled with the new clips. Recompute features needed
by the consuming model from the exported audio/motion.

## Output and downstream use

```text
finedance_aistpp/
  motions/finedance_001_0000000_0000586.pkl
  audio/finedance_001_0000000_0000586.wav
  audio/manifest.csv
  labels/finedance_001_0000000_0000586.json
  train.txt
  val.txt
  test.txt
  plan.json
  manifest.json
```

Names contain the source recording and start/end indices on the 60-FPS timeline;
the end is exclusive. Original song/genre labels are stored in the manifests and
per-clip JSON. The original label's `frames` value is metadata, not clip length.

For corrected GMR v2 output with the AIST++ smoothness fix, use the dedicated
[FineDance GMR converter](finedance_gmr_conversion.md). The example below is the
legacy pipeline without the v2 Hermite correction.

The existing GMR batch builder accepts the exported motion schema and split
files. For example, test one clip with the installed GMR environment:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/build_aistpp_gmr_dataset.py `
  --aistpp-root realtime/humanoid_robot/data/finedance_aistpp `
  --output-root realtime/humanoid_robot/data/finedance_gmr `
  --gmr-root realtime/humanoid_robot/.deps/GMR `
  --gmr-python realtime/humanoid_robot/.venv-gmr/Scripts/python.exe `
  --fps 60 --limit 1
```

The existing **music retrieval catalog is not yet a drop-in consumer**:
`music_motion_catalog.parse_motion_name` requires AIST++ filenames and its genre
priors assume AIST++'s ten genres. Adapt that parser and genre handling to the
exported labels before building a mixed catalog. The export does not disguise
FineDance as AIST++ filenames or assign invented AIST genres.

## Local verification (2026-09-20)

- Inspected all 203 local motion files: all use the 315D raw layout.
- Default audit: 185 accepted recordings, **2,530 planned clips** (2,380 train,
  150 test), approximately 7.027 hours. Durations: 8.667–11.567 seconds.
- Excluded 16 recordings for duration mismatch and 2 by the ignore list
  (130, 202). Notable mismatches: 043 has 20 seconds less audio, 059 has 18.033
  seconds less audio, and 131 has 17.200 seconds more audio.
- Exported source 001 as 10 real clips and verified every pickle/WAV. A real
  exported clip also passed the existing SMPL forward-kinematics calculation.
  The full dataset export and GMR retargeting were not run.
- Seven focused tests cover rotation conventions, shortest-path interpolation,
  all segment lengths up to 10,000 frames, audio timing markers, split leakage,
  mismatches, and supported layouts:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_segment_finedance.py -v
```

## Primary references

- [FineDance dataset description](https://github.com/li-ronghui/FineDance)
- [FineDance preprocessing: FPS and layout](https://github.com/li-ronghui/FineDance/blob/main/data/code/pre_motion.py)
- [FineDance rotation utilities](https://github.com/li-ronghui/FineDance/blob/main/dataset/quaternion.py)
- [FineDance renderer and Y offset](https://github.com/li-ronghui/FineDance/blob/main/render.py)
- [FineDance source splits](https://github.com/li-ronghui/FineDance/blob/main/dataset/FineDance_dataset.py)
- [PyTorch3D rotation representation](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py)
- [Official AIST++ loader](https://github.com/google/aistplusplus_api/blob/main/aist_plusplus/loader.py)

## Reproduce the published GMR release

The [release reproduction guide](finedance_reproduction/README.md) includes the
frozen full segmentation manifest, exact clip-to-audio mapping, source/output
checksums, and an audio-only reconstruction command. The historical pilot status
above predates the completed 2,530-clip export and 2,522 successful GMR motions.
