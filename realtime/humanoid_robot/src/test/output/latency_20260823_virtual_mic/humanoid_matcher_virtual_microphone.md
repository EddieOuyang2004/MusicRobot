# Humanoid matcher virtual-microphone result

Generated: `2026-08-23T01:36:22.601037+00:00`

## Scope

Original AIST++ WAV files were delivered block-by-block through the production `MatcherFileMicrophoneSource` at wall-clock speed. Each completed query includes the rolling audio window, EffNet ONNX inference, track retrieval, and motion ranking.
The WAVs belong to the catalog, so exact-track accuracy checks reconstruction and realtime plumbing; it is not an unseen-music generalization score.

## Summary

- Files/genres: 1 / 1.
- Completed matches: 4.
- Waveform-to-match latency: median **62.5 ms**, p95 **87.3 ms**, max **87.3 ms**.
- First (cold) match: **87.2 ms**; steady-state median: **37.9 ms**.
- WAV source setup/beat analysis: median **2447.3 ms**, max **2447.3 ms**.
- Results within the 1000 ms match interval: **100.0%**.
- Exact music Top-1/Top-3: **100.0% / 100.0%**.
- Genre Top-1/Top-3: **100.0% / 100.0%**.
- Mean realtime factor: **0.998x** (1.0x means wall-clock playback).
- Busy submission attempts: 0 / 4 (0.0%).
- Completed scheduled match slots: 4 / 4 (100.0%).

## Per-file results

| Genre | Music | Matches | Track Top-1 | Genre Top-1 | Median latency | Realtime |
|---|---|---:|---:|---:|---:|---:|
| BR | BR0 | 4 | 100.0% | 100.0% | 62.5 ms | 0.998x |

## Reproduce

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/benchmark_humanoid_matcher_virtual_microphone.py
```

Raw results: `realtime/humanoid_robot/src/test/output/latency_20260823_virtual_mic/humanoid_matcher_virtual_microphone.json`.
