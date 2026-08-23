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
