# Humanoid dancer analysis tools

## Motion/audio phase comparison

`analyze_motion_audio_phase.py` feeds the motion's synchronized music through
the same microphone BPM/beat analyzer and adaptive phase controller used by the
humanoid dancer. It compares the resulting phase with the authored phase
`audio_time / motion_duration` over the complete overlapping duration.

From the repository root:

```powershell
python realtime/humanoid_robot/src/test/analyze_motion_audio_phase.py `
  --motion realtime/humanoid_robot/data/aistpp/motions/gWA_sBM_cAll_d26_mWA0_ch07.pkl `
  --audio realtime/humanoid_robot/data/aistpp/audio/gWA_sBM_cAll_d26_mWA0_ch07.wav
```

For AIST++ files, `--audio` can be omitted when the `.wav` has the same stem in
the sibling `audio` directory. The default output directory is
`realtime/humanoid_robot/src/test/output`.

The JSON report contains:

- full-audio and post-first-accepted-beat phase MAE, RMSE, p95, and maximum;
- errors in cycles, degrees, and equivalent authored-motion seconds;
- circular bias and bias-corrected error to separate start offset from drift;
- detected/accepted beat counts and per-beat keypoint alignment diagnostics;
- motion/audio coverage, so a short audio file cannot be mistaken for a
  whole-motion result.

The CSV is the complete phase time series. The command also writes an HTML chart
showing original phase, microphone-driven phase, circular phase error, and every
accepted beat on one time axis. Add `--open-chart` to open it automatically.
The analysis defaults mirror `realtime_music_humanoid_dancer.py`; its
controller, keypoint, microphone, and PLP options can be overridden from the
command line.

Use `--max-seconds 3` for a quick pipeline check. Omit it for the requested
whole-motion comparison.

## Matcher performance benchmark

`benchmark_humanoid_matcher.py` measures the production music-motion catalog's
retrieval quality, matcher latency/throughput, startup cost, and memory use. Its
leave-one-music genre test removes all segments of the query music ID before
ranking, while the same-segment result is reported only as a pipeline sanity
check. If the catalog's feature model and `onnxruntime` are available, the tool
also measures waveform-to-match latency and gain invariance.

From the repository root:

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/benchmark_humanoid_matcher.py
```

The human-readable Markdown and raw JSON reports are written to
`realtime/humanoid_robot/src/test/output` by default. Use
`--latency-iterations`, `--feature-iterations`, and `--gain-check-limit` to
change the workload.

### Original WAV virtual-microphone benchmark

After installing `onnxruntime` and placing the catalog's EffNet model under
`realtime/humanoid_robot/models`, run the complete block-fed audio path at
wall-clock speed:

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/benchmark_humanoid_matcher_virtual_microphone.py
```

The benchmark selects one original catalog WAV per genre and exercises
`MatcherFileMicrophoneSource`, the realtime analyzer, rolling audio windows,
ONNX feature extraction, track retrieval, and motion ranking. Its JSON and
Markdown reports are also written to the test `output` directory.

### CC0 style and diversity benchmark

`benchmark_cc0_matcher.py` compares 3, 6, 9, and 12 second retrieval windows
over the heterogeneous CC0 matcher set. It reports accepted/rejected windows,
genre and motion concentration, and per-file rankings. With a rebuilt EffNet
catalog, `--assert-style-first` enforces the regression limits (no genre above
35%, no motion above 20%, and no ambient switch candidates):

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/benchmark_cc0_matcher.py `
    --assert-style-first `
    --output-json realtime/humanoid_robot/src/test/output/cc0_matcher_benchmark.json
```

## Complete Humanoid Matcher experiment protocol

`run_humanoid_matcher_experiments.py` is the preregistered runner for retrieval,
rejection, BAS, robot-domain diversity, PFC/FSR, transition continuity, staged
latency, response latency, safety constraints, and paired ablations. The fixed
configuration and output contract live in
`humanoid_matcher_experiment_protocol.json` and
`humanoid_matcher_experiment.schema.json`.

Run the deterministic offline evaluation first. This removes every segment and
motion associated with the query music ID before retrieval and reports both
music-level primary results and window-level diagnostics:

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/run_humanoid_matcher_experiments.py `
  --output-dir realtime/humanoid_robot/src/test/output/humanoid_matcher_experiment
```

Create the balanced 20-file negative set, then include it in rejection tests:

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/build_matcher_negative_set.py `
  --output-dir realtime/humanoid_robot/data/test_audio/matcher_negative_set

.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/run_humanoid_matcher_experiments.py `
  --negative-audio-root realtime/humanoid_robot/data/test_audio/matcher_negative_set
```

Build the registered tempo jump, five-style genre jump, and silence/recovery
streams with:

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/build_matcher_change_streams.py
```

Preview the exact subprocess matrix without running MuJoCo, or execute the
10-second end-to-end smoke suite:

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/run_humanoid_matcher_experiments.py `
  --execute-suite ablation --dry-run

.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/run_humanoid_matcher_experiments.py `
  --execute-suite smoke
```

The other suites are `literature` (fixed 40 clips), `full` (all 60 AIST++ plus
CC0), `ablation` (stitched stream plus CC0, every registered condition and five
seeds), and `longrun` (every condition and seed over a deterministically repeated
10-minute stitched stream). Every executed run enables the runtime collision checker and
writes a trace CSV, timing JSON, log, and pose NPZ. The report aggregates frames
to runs before statistics; stitched traces also receive crossfade-aware
music-change-to-match/pending/switch/blend/stable response times.

FID_k/FID_g and Div_k/Div_g are intentionally omitted unless `--feature-bundle`
is supplied. The NPZ bundle must contain `real_kinetic`, `output_kinetic`,
`real_geometric`, `output_geometric`, and a non-empty `extractor_identity` so
results from incompatible feature extractors cannot be mixed. Robot pose
bundles may additionally provide `g1_library_features` and
`g1_output_features`; their within-set pairwise-distance ratio is then reported
as the G1 output-to-preflight-library diversity ratio. Robot pose
outputs are explicitly marked as kinematic MuJoCo previews, not evidence of
closed-loop balance, torque feasibility, or real-hardware success.

Each report directory contains the immutable manifest/report schema, five CSV
tables (the four main tables plus ablations), and—when the corresponding runs
exist—PNG figures for latency CDF, BAS distribution, selection coverage,
speed/jerk, and change-response milestones.
