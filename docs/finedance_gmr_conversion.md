# FineDance to corrected GMR v2 motions

`realtime/humanoid_robot/src/build_finedance_gmr_dataset.py` converts the
completed separated FineDance export directly to Unitree G1 motions. It reuses
the AIST++ GMR v2 worker and its revision-2 Hermite smoothness correction. No
preliminary legacy GMR conversion is needed.

## Commands

Run from the MusicRobot repository root. The default source is
`realtime/humanoid_robot/data/finedance_aistpp`; the default destination is
`realtime/humanoid_robot/data/finedance_gmr_v2`. The installed GMR Python,
GMR checkout, neutral SMPL body model and robot assets are required for generation.

Validate source selection without creating any output:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/build_finedance_gmr_dataset.py --dry-run
```

Generate or resume the three representative pilot clips:

```powershell
realtime/humanoid_robot/.venv-gmr/Scripts/python.exe `
  realtime/humanoid_robot/src/build_finedance_gmr_dataset.py `
  --motion-id finedance_001_0000000_0000586 `
  --motion-id finedance_037_0000000_0000596 `
  --motion-id finedance_187_0000000_0000694 --jobs 1 --resume
```

Generate the complete collection, reusing matching pilot outputs:

```powershell
realtime/humanoid_robot/.venv-gmr/Scripts/python.exe `
  realtime/humanoid_robot/src/build_finedance_gmr_dataset.py --jobs 1 --resume
```

Audit the saved pilot trajectories without rebuilding:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/build_finedance_gmr_dataset.py `
  --audit-only --motion-id finedance_001_0000000_0000586 `
  --motion-id finedance_037_0000000_0000596 `
  --motion-id finedance_187_0000000_0000694
```

After full generation, omit the motion IDs to audit all input clips. Missing
outputs are reported as errors. `--audit-only` writes `audit.json`; it checks
saved validation metadata, checksums, source provenance and trajectory extrema.
It does not repeat the expensive geometric collision checks performed during
generation. It uses the existing audit's per-file function without its
AIST++-specific catalog summary.

`--split train|val|test|all` filters source membership. `--motion-id` is repeatable
and IDs must belong to the selected split. `--limit N` applies after sorting and
deduplication. An empty split is valid for a dry run; generation requires at
least one clip. `--jobs 1` is the conservative default; `--jobs 2` is the maximum. Each worker uses one
numeric thread. Relative path overrides resolve from the current working directory.

Override paths with `--input-root`, `--output-root`, `--gmr-root`, `--gmr-python`
and `--smpl-model-path`. `--smoothing-weight` defaults to 2 and
`--planning-clearance` defaults to 0.008 metres. Changing these invalidates
selected cached outputs. Timing overrides and disabling collision checks are
not supported by this converter.

## Constraints and timing

The converter enables the existing GMR/Mink joint speed and self-collision
constraints, then applies the same collision-free guide and state-aware quintic
Hermite projection as the AIST++ fix. The robot target has a free root and 29 joints.

| Setting | Default |
| --- | --- |
| Planning / segment / final clearance | 8 / 6 / 5 mm |
| Joint speed | 3*pi rad/s |
| Joint acceleration | 1600 rad/s^2 |
| Root speed | 3 m/s |
| Root angular speed | 4*pi rad/s |
| Final collision sampling | 81 samples per Hermite interval on both existing robot models |

Joint position, speed and acceleration bounds are checked over the polynomial,
including continuous extrema. Jerk is penalized and reported, with no hard jerk
limit. Failed safe continuation rejects the clip rather than publishing an
instantaneous hold. Naturally stationary intervals are reported, not forbidden.

Source filenames, all frames and 60 FPS timing are retained. Nominal clip/audio
duration is N/60; the last authored pose is at (N-1)/60. The source audio,
segmentation, horizontal translation and heading are not independently edited;
robot root motion still passes through the existing retargeter and speed limits.
The segmented SMPL data already includes the FineDance vertical offset and
body-only conversion; neither is applied a second time.

This matches AIST++'s existing scope. Dense collision samples are not a continuous
collision proof. Floor contact, balance, hardware tracking, time warping, loop
joins and inter-clip transitions are not certified by this conversion.

## Output, failures and resume

The output root contains canonical `<clip-id>.pkl` files, `manifest.json`,
`failures.json`, `train.txt`, `val.txt`, `test.txt`, and per-clip logs/job settings
under `logs/`. Only successfully published clips appear in the manifest and split
lists. Each success gets a small per-clip result checkpoint; aggregate metadata is saved
every 30 seconds and on exit. The process returns nonzero if any
selected clip fails. Inspect `failures.json` and the named log before retrying.

A shared OS batch lock prevents concurrent builds in one output directory and
releases on process exit. Do not delete a live lock. If the parent was forcibly
terminated, let any remaining workers finish before restarting.

Without `--resume` or `--overwrite`, existing selected outputs are refused.
Resume reuses an output only when source data, segmentation manifest, clip labels,
settings, implementation, robot assets, SMPL model, GMR commit and recorded
runtime dependency versions match. Stale outputs are retained with an `.invalid`
suffix before rebuilding. `--overwrite` forces the same rebuild policy. Failed
staging files remain under `logs/` and are never canonical playback outputs.
Unselected successful outputs remain in the manifest; resume validates the
selected set, so rerun the full selection after changing shared settings.

Each motion retains `source_format="aistpp_smpl_direct"` for loader compatibility,
`motion_version="gmr_v2"`, `pipeline_version=5`, and projection revision 2.
Float64 `dof_pos`, `dof_vel`, `dof_acc` arrays and FPS are bound by the existing
state checksum. Root rotation uses `wxyz`. FineDance provenance is stored in
`v2_build` in the pickle and manifest: source recording, segment bounds, split,
labels, source hash, audio reference, segmentation settings and manifest hash.
Audio references are relative to the input dataset recorded by the output
manifest; audio is not copied. The segmentation manifest hash covers its full
recording plans and exclusions without duplicating those lists per output.

Existing motion samplers can load the files directly. Register successful outputs
with the [combined music retrieval catalog](finedance_matcher_integration.md),
which reads FineDance labels and resolves each dataset's GMR directory.
See [segmentation details](finedance_segmentation.md) and
[the AIST++ correction](gmr_v2_batch.md).

## Regression checks

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_finedance_gmr.py -v
realtime/humanoid_robot/.venv-gmr/Scripts/python.exe -m unittest discover -s tests -p test_gmr_v2.py -v
.venv/Scripts/python.exe -m unittest discover -s tests -p test_gmr_v2.py -v
realtime/humanoid_robot/.venv-gmr/Scripts/python.exe -m unittest discover -s tests -p test_gmr_state_trajectory.py -v
.venv/Scripts/python.exe -m unittest discover -s tests -p test_gmr_batch_lock.py -v
.venv/Scripts/python.exe -m unittest discover -s realtime/humanoid_robot/src/test -p test_audit_gmr_smoothness.py -v
```

The two GMR v2 test invocations cover the optional generation and playback
dependencies in their respective installed environments.

## Local pilot results (2026-09-20)

All 2,530 separated source clips passed the non-mutating dry run. The pilot
selected the first training clip, first test clip, and longest remaining clip
(alphabetical tie-break). All three generated successfully and were reused
as cached artifacts on a subsequent resume. The full collection was not converted.

| Clip | Frames | Peak joint speed (rad/s) | Peak joint acceleration (rad/s^2) | Peak jerk (rad/s^3) | Joint target RMSE (rad) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `finedance_001_0000000_0000586` | 586 | 2.4655 | 68.88 | 4286.16 | 0.002063 |
| `finedance_037_0000000_0000596` | 596 | 9.1357 | 157.56 | 7347.13 | 0.008489 |
| `finedance_187_0000000_0000694` | 694 | 9.4231 | 203.20 | 10718.43 | 0.134359 |

Peak derivatives above are continuous Hermite extrema. All three had zero
violating collision samples, zero whole-joint hold intervals and zero solver
failures. Both `g1_mocap_29dof.xml` and `open_humanoid_dancer.xml` were checked.
The saved-output audit passed all three with no joint position excess or
root speed/angular-speed violation. Original 60 FPS and frame counts were
preserved. The existing playback sampler loaded all three and preserved
stored position, velocity and acceleration at both authored endpoints.

Clip 187 needed substantially more correction (0.134 rad joint target RMSE)
than clips 001 and 037. This measures deviation from the raw robot IK target,
not SMPL reconstruction error; review its motion fidelity before wider use.

Detailed results are in `data/finedance_gmr_v2/manifest.json`, `audit.json`,
and `pilot_validation.json`. Generation took approximately 31, 45 and 68
seconds per clip in the local two-worker run; these are not full-batch estimates.

The focused checks passed: 11 FineDance converter tests, 13 unique shared
GMR v2 tests (across the two environments), seven state-trajectory tests,
five batch-lock tests and four smoothness-audit tests: 40 distinct tests.

## Recovering a damaged output manifest

If an interrupted run leaves `manifest.json` or `failures.json` empty, truncated,
zero-filled, or otherwise unreadable, `--resume` now preserves it as
`<name>.corrupt.<timestamp>` and reconstructs missing motion metadata from saved
canonical `.pkl` files. Input segmentation manifests still fail strictly if damaged.
A missing output manifest is also recoverable. Recovery scans all saved outputs,
even when the next generation command selects only a subset.

To repair metadata without generating any new motions:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/build_finedance_gmr_dataset.py --repair-only
```

Repair rechecks source hashes, segmentation provenance, state checksums, existing
collision-validation metadata and the saved-trajectory smoothness audit. Only
passing files enter the reconstructed manifest and split lists. Invalid motion
files remain untouched and are listed in `failures.json`; a subsequent `--resume`
will quarantine and rebuild them. Missing outputs are generated normally on resume.
A nonzero repair exit code means failure entries remain, not that every recovered
motion failed. Historical failure messages cannot be recovered from zero-filled
metadata; the retained worker logs remain available for diagnosis.

Checkpoint JSON and split lists are flushed and synced to disk before atomic
replacement. This improves interruption resilience but cannot guarantee survival
of every storage/device failure. The precise cause of zero-filled files is not
established by the JSON error alone.

This metadata-only correction preserves cache compatibility with the exact previous
builder revision. Source, settings, dependency, model and other implementation
changes still invalidate caches normally; no collision or smoothness limits have
been relaxed.

Recovery verified locally on 2026-09-21: the manifest, failure report and split
lists were zero-filled. Of 683 canonical motion files, 682 passed recovery; one
zero-filled motion (`finedance_065_0003125_0003750.pkl`) was preserved as `.invalid`
and regenerated successfully with the existing constraints. The rebuilt manifest
and split lists contain 683 motions. A known-good clip resumed as `cached`, and
all 17 converter tests plus five batch-lock tests passed. The remaining full
collection was not launched during this repair.

## Desktop stability safeguards (2026-09-22)

The local System log records unexpected restarts (events 41/6008), but does not
establish an out-of-memory, thermal, driver or hardware root cause. These changes
bound the converter's resource use and fix batch failure propagation; they do not
claim to repair an unidentified OS/hardware problem.

Resume conservatively from the repository root:

```powershell
realtime/humanoid_robot/.venv-gmr/Scripts/python.exe `
  realtime/humanoid_robot/src/build_finedance_gmr_dataset.py --jobs 1 --resume
```

- Default one worker; maximum two. Only that many tasks are submitted at once.
  A progress/checkpoint failure cannot leave thousands of queued jobs running.
- Windows workers run at idle priority in kill-on-close Job Objects. Each job
  includes the virtual-environment launcher and its descendants. Default limits:
  3 GiB committed memory per worker tree and 20% of total machine CPU per tree.
  The combined configured CPU budget cannot exceed 50%.
- Before launching a worker, require its configured memory budget plus a 4 GiB
  free-system-memory reserve. While running, check the reserve every 0.5 seconds;
  stop the batch if it is breached. This monitor is not an instantaneous guarantee.
- Default per-clip timeout: 900 seconds. A timeout fails that clip and terminates
  its worker tree; other clips may continue. Memory pressure stops the whole run.
- Ctrl+C, coordinator failure and parent-process exit clean up owned Windows
  worker trees. Completed motion files remain available for resume.
- Parent and child numeric libraries use one thread. Child logs are unbuffered.
- Console output uses a bounded nonblocking queue. A frozen/selected console
  cannot block batch progress. Some console lines may be dropped if it stalls.
  `progress.json` reports current active IDs, completed count and running/stopped/
  complete status, updating at most about five seconds apart during generation.
- Successful results also get `logs/<clip-id>.result.json`; the larger manifest
  and splits are synced every 30 seconds and on exit, reducing repeated full-file
  writes. Abrupt termination can leave aggregate metadata behind saved artifacts;
  `--resume` checks selected artifacts again, or use `--repair-only` to reconcile.

The resource options are `--worker-timeout`, `--worker-memory-gb`,
`--worker-cpu-percent`, and `--min-free-memory-gb`. Defaults are intended for this
Windows desktop. Windows CPU and committed-memory hard caps are not provided on
other operating systems. The generation algorithm, collision checks, source timing
and smoothness limits are unchanged; the exact stability revision retains cache
compatibility with prior known-compatible builder revisions.

Tests exercise real lightweight Windows children for memory-limit failure,
timeout cleanup of descendants, cancellation and normal exit, plus bounded queue,
low-memory preflight and heartbeat/checkpoint failure behavior.

Verified locally: all 24 converter/runtime tests passed. A fresh real clip
`finedance_001_0000000_0000586` completed in 35.4 seconds under the default
resource limits in `tmp/finedance_stability_smoke`, passed collision validation
with zero violating samples, and resumed as `cached`. Existing dataset cache
implementation compatibility was checked separately. No full batch was started.
