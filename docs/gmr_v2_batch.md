# Batch generation of GMR v2 motions

Use `realtime/humanoid_robot/src/build_gmr_v2.py`. It selects the existing GMR
filenames, regenerates their original AIST++ SMPL motions through GMR, and uses
a collision-free guide followed by state-aware Hermite trajectory optimization.
Filtering the old saved poses alone would not restore movements already removed
by collision holds.

The original 411 GMR files stay in `realtime/humanoid_robot/data/aistpp_gmr`.
The same filenames are saved under `realtime/humanoid_robot/data/aistpp_gmr_v2`.
All 411 currently have matching SMPL source files. The SMPL body model and local
GMR installation are also required; they are present in this workspace.

## Generate the complete v2 set

Run in PowerShell from the MusicRobot repository:

```powershell
cd C:\Users\Eddie\Desktop\MusicRobot

.\realtime\humanoid_robot\.venv-gmr\Scripts\python.exe `
  .\realtime\humanoid_robot\src\build_gmr_v2.py --jobs 2 --resume
```

This selects **the existing GMR set**, not every entry in the original AIST++
train/validation/test lists. It preserves each clip's original FPS, source frame
count, filename and source timeline. Each generation worker uses one numeric
thread. `--jobs 1` reduces simultaneous CPU/memory usage.

Two pilot files already exist in the v2 folder:

- `gWA_sBM_cAll_d26_mWA0_ch07.pkl`
- `gBR_sBM_cAll_d05_mBR0_ch08.pkl`

Both passed dense collision checks and had zero whole-joint holds and zero solver
failures. The current Hermite correction has been validated on these two pilot clips. The
command above rebuilds older v2 revisions and reuses matching corrected files.

## Inspect selection or run a small batch

```powershell
# Inspect selection and missing SMPL sources; writes nothing.
.\realtime\humanoid_robot\.venv-gmr\Scripts\python.exe `
  .\realtime\humanoid_robot\src\build_gmr_v2.py --dry-run

# First five IDs in alphabetical order.
.\realtime\humanoid_robot\.venv-gmr\Scripts\python.exe `
  .\realtime\humanoid_robot\src\build_gmr_v2.py --limit 5 --jobs 2 --resume

# One specific clip; repeat --motion-id to select several.
.\realtime\humanoid_robot\.venv-gmr\Scripts\python.exe `
  .\realtime\humanoid_robot\src\build_gmr_v2.py `
  --motion-id gWA_sBM_cAll_d26_mWA0_ch07 --resume
```

`--input-root`, `--source-root`, `--output-root`, `--gmr-root`, `--gmr-python`
and `--smpl-model-path` override the workspace defaults. Relative override paths
are resolved from the current working directory. The tool rejects output folders
that overlap the original GMR or SMPL source folders.

## Resume and inspect results

Rerun the same command with `--resume`. A saved v2 artifact is reused only when its
source, original GMR file, FPS, SMPL model, projection settings, implementation,
robot assets, GMR commit and recorded dependency versions match. Stale selected
v2 files are renamed with an `.invalid` suffix before rebuilding. `--overwrite`
forces rebuilding and also retains the previous selected v2 files this way.
Neither option edits the original GMR folder.

The output folder contains:

| File | Purpose |
| --- | --- |
| `<motion-id>.pkl` | Validated GMR v2 motion, in the existing pose-array format |
| `manifest.json` | Successful clips, hashes, generation time, hold/smoothness metrics and solver failures |
| `failures.json` | Failed clips and explanations; resolved failures disappear on resume |
| `logs/<motion-id>.log` | Full worker output for diagnosis |
| `logs/<motion-id>.job.json` | Exact generation settings and source provenance |

Results and failures are checkpointed after each completion. Exit status 1 means
at least one selected clip failed. A collision-validation failure is not published
as a canonical `.pkl`; inspect its log and fix the cause before resuming.

Only one batch may use a given output folder at a time. The builder now holds an
operating-system lock that is released automatically on exit or a crash. The
`.batch.lock` file remains as metadata; its existence does not block resume. Old
PID-only locks are recovered automatically when their owner no longer exists.
A live owner is reported clearly rather than overwritten. Do not delete the lock
while a batch is running. If a parent process was killed while workers were active,
wait for those workers to finish before restarting. The lock-only fix preserves
cache compatibility only for identical generation code. The Hermite correction
changes generation and requires rebuilding older v2 files.

## Preview or use v2

The player and matcher still default to the original folder. Select v2 explicitly:

```powershell
.\.venv\Scripts\python.exe `
  .\realtime\humanoid_robot\src\realtime_music_humanoid_dancer.py `
  --preview-trajectory --realtime `
  --gmr-motion-root .\realtime\humanoid_robot\data\aistpp_gmr_v2
```

The matcher also accepts `--gmr-motion-root`; add the same argument to its usual
launch command after the desired v2 clips have been generated. The GMR **dataset**
version here is independent of the matcher's own v2 name.

Artifacts use `motion_version="gmr_v2"` and internal `pipeline_version=5`.
The old dataset uses pipeline version 4. Internal version 2 already referred to
an older pipeline, so it is deliberately not reused. Playback and auditing now
accept validated version 5 while continuing to accept version 4.

## State-aware Hermite correction (projection revision 2)

The previous projection revision checked linear interpolation. Revision 2 plans
and validates the quintic Hermite curves used by playback. Older revision-1 v2
files are rejected by playback/auditing with a regeneration message. Run the same
batch command with `--resume`: old selected files are retained as `.invalid`
backups and rebuilt. Original pipeline-v4 files remain supported and unchanged.

The optimizer uses a collision-free positional guide, then looks up to 12 source
frames ahead. Each new curve begins at the accepted position, velocity and
acceleration and tracks future guide states. Shorter horizons are tried when
needed. Joint ranges, speed and acceleration are bounded across the polynomial
using Bernstein controls and checked with exact extrema. Terminal continuation
constraints leave room beyond the horizon for joint motion and collision avoidance.
Jerk is penalized in physical time, but has no hard maximum.

Defaults: 8 mm planning clearance, 6 mm planning segment clearance, 5 mm final
clearance, 3*pi rad/s joint speed, 1600 rad/s² joint acceleration (the existing
software playback acceleration limit), 3 m/s root speed, and 4*pi rad/s root
angular speed. These are software settings, not robot-certified limits.
`--smoothing-weight` and `--planning-clearance` still control smoothing/clearance.

An unsuccessful replan may consume the remaining validated curve. It does not
replace the current motion with an instantaneous zero-velocity hold. If no safe
continuation remains, generation fails and the clip is not published. Only at the
first frame, where no incoming clip is supplied, an infeasible derivative estimate
may be reduced; `initial_derivative_scale` records this explicitly.

Files store float64 `dof_pos`, `dof_vel`, and `dof_acc`, plus a checksum binding
those arrays to FPS. Player, matcher entry/exit state selection and smoothness
audits use these exact states rather than recalculating derivatives. Generation
checks 81 samples per Hermite interval on both models, and exact continuous
joint-range/speed/acceleration extrema. Validation results and continuous peaks
are stored in `projection_validation`. Revision 2 preserves fixed source timing.

The correction covers within-clip joint curves at authored speed. Collision checks
are dense samples, not a proof of continuous clearance. It does not certify
separate inter-clip bridges, loop joins, time warping, downstream limiting, balance,
floor contact or physical robot tracking. FPS overrides for these files are refused;
changing timing requires revalidation. Review full-dataset failures and metrics
before selecting the corrected dataset for an entire playback workflow.
