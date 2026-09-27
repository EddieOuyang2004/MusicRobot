# Splitting long music–motion pairs

Run these commands in PowerShell from the MusicRobot repository root. Use the
existing `.venv`; no retargeting, dataset download, or new model is required.

## Preview, pilot, then full build

Preview the inventory and every proposed frame boundary without generating files:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/segment_music_catalog.py --dry-run
```

The current library contains 411 AIST++ motions and 2,522 FineDance motions.
Only the 28 AIST++ motions longer than 16 seconds are split. The expected output
is 496 AIST++ motions: 383 unchanged clips plus 113 children. Including FineDance,
the expected combined membership is 3,018 motions. These counts are derived from
the input catalogue, not hard-coded admission rules.

Run a small, separate pilot (one long parent, all its children, and a small
FineDance selection):

```powershell
.venv/Scripts/python.exe -u realtime/humanoid_robot/src/segment_music_catalog.py `
  --limit 1 --build-catalog --output-dir tmp/segmented_pilot
```

**Run the complete workflow yourself with this command:**

```powershell
.venv/Scripts/python.exe -u realtime/humanoid_robot/src/segment_music_catalog.py --build-catalog
```

This exports segments, validates stored robot trajectories and headless playback,
rebuilds retrieval features, checks membership, and runs cross-dataset transition
tests. Only after success does it replace the catalogue used by the default
matcher. Pilot runs never activate. Stop with Ctrl+C and rerun the same command
to resume. Validation checks collision geometry densely; the full build can take
substantial time even though it does not repeat retargeting.

To generate assets only, omit `--build-catalog`. Add it on a later run to finish.
Resume is automatic: source hashes, implementation, dependencies, models, settings,
and output hashes must match before cached validation is reused.

## Understand the three layers

Keep original datasets, derived assets, and retrieval metadata separate:

```text
realtime/humanoid_robot/data/
  aistpp/                          Original SMPL and paired audio
  aistpp_gmr_v2/                   Original validated robot trajectories
  finedance_aistpp/                Existing FineDance source/audio clips
  finedance_gmr_v2/                Existing FineDance robot trajectories
  music_catalog_segmented/
    input_catalog.json            Frozen membership and paths for this workflow
    assets/<generation>/
      aistpp/motions/              Segmented SMPL plus unchanged short clips
      aistpp/audio/                Matching WAV files
      aistpp/segments_manifest.json
      aistpp/train.txt, val.txt, test.txt
      aistpp_gmr_v2/               Matching GMR files and manifest.json
      cache/                      Per-clip validation and output checksums
    candidate/                    Catalogue, immutable feature arrays, build cache
    playback/<build-id>/           Transition traces, logs, timing, summary.json
    progress.json
    last_failed_run.json           Most recent failure, if any
  music_catalog_combined/
    catalog.json                  Active default catalogue
    catalog_features_<id>.npz     Immutable features referenced by catalog.json
    catalog.backup.<id>.json      Previous catalogue; retained for rollback
```

Generation directories are selected by input and validation fingerprints. Changes
create another directory so the active catalogue can keep referencing its assets.
Keep every asset directory and feature array referenced by an active or backup
catalogue. Do not manually rename files that manifests or catalogues reference.

The input snapshot keeps a resumed run tied to its original membership, even after
activation. To reorganize a different input catalogue or include newly admitted
material, use a fresh `--output-dir` and pass that catalogue with `--catalog`.
That explicit catalogue is also the activation target for a successful full run.

## How music and motion stay paired

Child names include the parent ID and half-open frame range, for example:

```text
gBR_sFM_cAll_d04_mBR0_ch01__f0000576_0001152
```

The name identifies frames 576 through 1151. At 60 FPS this starts at 9.6 seconds
and ends at 19.2 seconds. The SMPL pickle, robot pickle, and WAV use the same stem.
Boundaries use motion frames; audio sample indices are rounded from those times.
All frames and audio samples are retained, with at most one sample of endpoint
rounding tolerance. Larger audio/motion mismatches are rejected.

Cuts are balanced and contiguous, with no overlap, fades, tempo changes, or
beat snapping. The algorithm prioritizes 8–10 seconds and then proximity to
9 seconds. A small exception is allowed when an exact partition is impossible;
the current children are approximately 7.98–9.85 seconds. Exactly 16 seconds is
unchanged. Robot sample duration (`frames/60`) and interpolation duration
(`(frames-1)/60`) are recorded separately.

Joint positions, velocities, accelerations, and root transforms are copied exactly
from the same parent frame slice. The child receives fresh source, artifact, and
state hashes, continuous derivative/collision validation, and MuJoCo preflight.
Parent generation details remain under `parent_generation`; they are not presented
as a new retargeting result.

Each child has its own retrieval music ID, so its audio score stays attached to
its paired motion. Its original recording ID, genre, labels, and train/validation/
test split are retained. Siblings remain grouped during recording-held-out
evaluation. Six-second retrieval windows with a two-second hop are unchanged;
these are feature windows inside clips, not the physical segmentation length.

## Monitor and diagnose

```powershell
Get-Content realtime/humanoid_robot/data/music_catalog_segmented/progress.json
Get-Content realtime/humanoid_robot/data/music_catalog_segmented/candidate/progress.json
```

Export failures appear in the generation's `segments_manifest.json` and progress
report. Catalogue admission failures appear in `candidate/excluded_motions.json`.
The eight previously missing FineDance conversions can remain excluded; losing an
existing member or adding unexpected FineDance membership blocks activation.
Every expected AIST++ child must be admitted.

Playback must complete a cross-dataset switch in each direction without loading
failures, with the existing one-second retrieval limit. Reports also count
`terminal_hold_rows`. An additional shortest-FineDance startup run diagnoses the
known preparation-delay problem; its failure to switch is reported but is not an
activation gate. Transition algorithms and constraints are unchanged. A candidate
that fails the two required directions remains available for inspection while the
default catalogue stays intact. A subprocess error or timeout also stops activation.

## Launch, compare, and roll back

After successful activation, restart the matcher:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py --realtime
```

To inspect a candidate before activation:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py `
  --catalog realtime/humanoid_robot/data/music_catalog_segmented/candidate/catalog.json --realtime
```

The successful command prints the exact backup path. Run the matcher with
`--catalog "that-backup-path"` to compare immediately. To restore that catalogue as
the default, stop catalogue builders, choose the backup printed by the successful
run, then replace only the metadata pointer:

```powershell
$catalogFolder = "realtime/humanoid_robot/data/music_catalog_combined"
Get-ChildItem $catalogFolder -Filter "catalog.backup.*.json" | Sort-Object LastWriteTime -Descending
# Replace <id> with the backup you want to restore.
Copy-Item "$catalogFolder/catalog.backup.<id>.json" "$catalogFolder/catalog.restore.json"
[System.IO.File]::Replace(
  (Resolve-Path "$catalogFolder/catalog.restore.json"),
  (Resolve-Path "$catalogFolder/catalog.json"),
  (Join-Path (Resolve-Path $catalogFolder) "catalog.before-rollback.json"))
```

Restart the matcher after restoring. Do not delete feature arrays: each backup
references its own immutable array file.

## Add and organize future recordings

1. Add paired source motion and audio to the appropriate original dataset through
   its existing import/export workflow. Ensure duration and alignment match.
2. Assign genre/labels, a stable recording identity, and a source split before
   creating children. Keep siblings in their parent's split. Preserve existing
   official split assignments when reorganizing this library.
3. Generate and validate the original GMR trajectory, updating its conversion
   manifest. Rebuild an original combined catalogue into a separate output folder
   using `build_combined_music_catalog.py --output-dir ...`.
4. Run this splitter against that catalogue with a fresh output directory. Inspect
   `--dry-run`, then run `--build-catalog`. It activates the explicit input catalogue;
   launch it with `--catalog` if it is not the default folder.

Use manifest labels and catalogue recording/genre fields to organize the library.
Do not rearrange files into genre folders by hand or edit `catalog_features_*.npz`:
the builder maintains references, cached features, and normalization together.
