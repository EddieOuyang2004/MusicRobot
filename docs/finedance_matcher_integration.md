# FineDance and AIST++ combined matcher catalog

The humanoid matcher defaults to a combined, style-based library with separate dataset genres.
The original `data/music_catalog` remains available through an explicit `--catalog`;
the matcher uses the former v2 implementation under its unversioned name.
The combined catalog uses existing validated robot trajectories; it does not
regenerate motions or provide exact song-position/choreography synchronization.

## Build and resume

From the MusicRobot repository root:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/build_combined_music_catalog.py
```

Defaults:

- AIST++ source: `data/aistpp`; GMR output: `data/aistpp_gmr_v2`.
- FineDance source: `data/finedance_aistpp`; GMR output: `data/finedance_gmr_v2`.
- Output: `data/music_catalog_combined` (all paths above are under `realtime/humanoid_robot`).
- Both datasets, all train/validation/test splits, six-second reference windows,
  two-second hop, 16 kHz analysis, existing Discogs EffNet embedding/tag model.
- Sequential processing with one numeric/ONNX thread. An OS lock excludes a
  second build in the same output directory. Do not delete an active lock.

Rerun the same command to resume. Successful preflight and feature results are
cached using source/audio/artifact hashes, source metadata, processing settings,
model identity, relevant implementation, dependencies, and local robot assets.
Changed inputs invalidate the affected results. A truncated cache is rebuilt.
Source and artifact hashes are checked even when expensive work is cached.

Inspect `progress.json` during a build. A completed build writes `catalog.json`,
its uniquely named `catalog_features_<build-id>.npz`, and `excluded_motions.json`.
The JSON catalog also embeds its exclusion report. Arrays are written first and
the JSON pointer is atomically replaced last; an interruption cannot combine new
metadata with old arrays. Previous arrays remain available for already-running
readers. Restart the matcher to adopt a new catalog.

A small pilot covers several genres by selecting clips round-robin by genre:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/build_combined_music_catalog.py `
  --limit-per-dataset 4 --output-dir tmp/combined_catalog_pilot
```

Options include `--datasets aistpp finedance`, `--split all|train|val|test`,
`--aistpp-root`, `--finedance-root`, `--aistpp-gmr-root`, `--finedance-gmr-root`,
`--embedding-model`, `--tag-model`, `--model`, and `--output-dir`.
A selected dataset must have at least one passing motion to publish a catalog.
FineDance has no validation split, so use `--datasets aistpp --split val` when
building an AIST++ validation-only catalog. The original AIST++ catalog directory
and directories overlapping dataset inputs cannot be used as the output.

## Admission and retrieval behavior

FineDance source export must be complete. The builder selects source entries and
admits only successful GMR manifest records with matching artifact and source
hashes, 60 FPS/frame counts, paired audio duration, and segmentation provenance.
It runs the existing headless MuJoCo preflight on every admitted motion (or reuses
an identical successful cached result). Failed/missing/stale/malformed motions
are listed as exclusions; no audio belonging solely to rejected motions enters
the library. Repair retargeting failures separately with the FineDance converter.
No collision or joint-limit checks are disabled by this builder.

Each FineDance clip has a distinct retrieval music ID equal to its clip ID.
Its source recording, source time bounds, song title, original `style1`/`style2`,
and split remain in the catalog. Audio scores therefore stay attached to the
paired clip rather than propagating to every dance segment from a song. FineDance
BPM is computed from that clip's windows. AIST++ retains its music IDs and motion
IDs.

AIST++ genre codes remain unchanged. FineDance uses `finedance:<style2>`, e.g.
`finedance:Jazz` and `finedance:Hiphop`; overlapping labels are deliberately not
merged. A recording contributes only its strongest track score to its genre;
genre similarity averages the top three distinct recordings. Existing matching
weights, confidence gates, motion compatibility, and transition constraints remain
unchanged. Combined feature normalization and motion statistics are recomputed
from the accepted library.

## Playback and compatibility

After building, run the matcher with `--realtime`; it selects the combined catalog
by default. The equivalent explicit command is:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py `
  --catalog realtime/humanoid_robot/data/music_catalog_combined/catalog.json `
  --realtime
```

For file-driven headless playback, add `--headless --audio-input path/to/music.wav`.
The inherited `--motion-source aistpp` spelling also covers converted FineDance
SMPL files; it remains the default. Playback loads each dataset's GMR directory
from catalog schema v3. No motion files need copying into a shared directory.

Override roots individually with `--aistpp-gmr-motion-root` and
`--finedance-gmr-motion-root`. An explicit global `--gmr-motion-root` is rejected
for mixed catalogs because it cannot identify which dataset it overrides.
Schema v1/v2 catalogs retain the legacy global override behavior. Dataset source
and GMR roots are resolved relative to the catalog; source audio/motion paths
are relative to their own dataset root. Stored model paths are catalog-relative.

Rollback/comparison remains explicit:

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py `
  --catalog realtime/humanoid_robot/data/music_catalog/catalog.json `
  --gmr-motion-root realtime/humanoid_robot/data/aistpp_gmr_v2 --realtime
```

The original catalog, original source files, and existing experiment outputs are
not rewritten. Generated combined catalogs and local smoke artifacts are ignored
by Git; rebuild them after acquiring the source datasets and model assets.

## Evaluation

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/evaluate_music_catalog.py `
  --catalog realtime/humanoid_robot/data/music_catalog_combined/catalog.json `
  --output-json tmp/combined_catalog_evaluation.json
```

The report separates library self-retrieval from recording-held-out genre
retrieval, with results per dataset. Held-out evaluation excludes all sibling
clips from the recording and refits feature normalization using only references.
By default it samples one window per recording; use `--queries-per-recording 0`
for all windows or a positive number for evenly spaced samples per recording.
Library self-retrieval is not a generalization result. Recording grouping does
not guarantee distinct songs across different recording IDs or across datasets.

The report also includes catalog/model startup, matching latency, waveform
extraction-plus-matching latency, and gain-invariance checks. The default tests
four recordings per dataset at gains 0.1, 0.5, 1, and 2; change this with
`--gain-check-limit`. Cold first extraction is reported separately because the
runtime warms its isolated retrieval worker before control begins.

Focused regression checks:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_combined_music_catalog.py -v
.venv/Scripts/python.exe -m unittest discover -s tests -p test_music_motion_catalog.py -v
.venv/Scripts/python.exe -m unittest discover -s realtime/humanoid_robot/src/test -p test_humanoid_matcher_v2.py -v
```

Repeat the two-direction headless playback acceptance check (including a different
working directory and the spawned retrieval worker):

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/test/validate_combined_catalog.py `
  --catalog realtime/humanoid_robot/data/music_catalog_combined/catalog.json `
  --output-dir tmp/combined_full_acceptance
```

This uses fixed reference audio, an AIST++ reference startup motion, and the
longest accepted FineDance startup motion. It preserves authored timing,
checks that a cross-dataset switch completes in each direction, and requires the
observed feature-extraction plus matching maximum to remain below one second.
It writes traces, timing reports, and `summary.json`.

Historical thesis and AIST++ benchmark runners keep their original catalog
and single-dataset assumptions. Use the updated evaluator and combined acceptance
runner above for schema v3; do not present results from the legacy recording-group
logic as a combined-library evaluation.

## Startup preparation limit

With the full library, forcing the 9.77-second `finedance_001_0000000_0000586`
startup motion while retrieving AIST++ Breaking produced a terminal hold: the
existing planner waits for all ten shortlisted motions to finish preparation,
and the final candidate was not ready before the source ended. This reproduced
without concurrent evaluation. Retrieval itself continued within one second.
Starting with `finedance_187_0000000_0000694` (11.57 seconds) completed the reverse
transition with the same shortlist and constraints. The acceptance runner uses
the longest FineDance clip for that reason. The terminal-hold policy and
transition algorithm have not been changed; successful catalog admission does
not guarantee that every pair can transition before its deadline.

## Local validation (2026-09-25)

The full build completed in 1,906 seconds and admitted all 2,933 successfully
converted motions: 411 AIST++ and 2,522 FineDance. Exactly eight missing FineDance
conversions were excluded. It contains 9,133 audio windows, 2,582 retrieval IDs,
245 source recordings, and 26 separate genre groups. Membership includes AIST++
252 train / 22 validation / 137 test motions and FineDance 2,374 train / 148 test
motions. The eight-clip real pilot resumed with all eight cached in 0.93 seconds.

Evaluation sampled up to three evenly spaced windows per recording (716 queries):

| Dataset | Library self-retrieval R@1 / R@3 | Recording-held-out genre R@1 / R@3 |
|---|---:|---:|
| AIST++ | 100.0% / 100.0% | 26.1% / 42.9% |
| FineDance | 88.5% / 99.3% | 51.0% / 76.0% |
| Combined | 91.1% / 99.4% | 45.4% / 68.6% |

Catalog load took 415 ms and model initialization 110 ms. Matcher-only latency
was 63.6 ms median / 79.9 ms p95; sampled warm waveform extraction plus matching
was 111.6 ms median / 131.7 ms p95 / 149.8 ms maximum. First cold extraction was
1.35 seconds, measured separately; the realtime worker warms before playback.
All eight tested recordings retained their top prediction across four gains.

Full-library headless playback passed both dataset directions with the startup
selection described above. Startup to the control loop took 11.35–11.63 seconds;
the conservative maximum feature-plus-match timing was 391.5 ms and 415.4 ms.
Each run completed 29 retrieval jobs with zero busy submissions and no motion
loading failures. These are local observations, not hard realtime guarantees.
The short-startup terminal hold described above remains a known limitation.

All 87 focused regression tests passed: 13 combined-catalog tests, 27 existing
catalog tests, 24 matcher-v2 tests, and 23 legacy experiment tests. The original
catalog and v1 default were preserved. Detailed metrics and artifact paths are
saved in [the validation report](../output/combined_catalog_validation/report.json).
