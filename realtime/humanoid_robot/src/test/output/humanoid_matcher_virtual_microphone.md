# Humanoid matcher virtual-microphone result

Generated: `2026-08-12T00:21:25.160228+00:00`

## Scope

Original AIST++ WAV files were delivered block-by-block through the production `MatcherFileMicrophoneSource` at wall-clock speed. Each completed query includes the rolling audio window, EffNet ONNX inference, track retrieval, and motion ranking.
The WAVs belong to the catalog, so exact-track accuracy checks reconstruction and realtime plumbing; it is not an unseen-music generalization score.

## Summary

- Files/genres: 10 / 10.
- Completed matches: 36.
- Waveform-to-match latency: median **732.8 ms**, p95 **935.3 ms**, max **966.1 ms**.
- First (cold) match: **966.1 ms**; steady-state median: **730.0 ms**.
- WAV source setup/beat analysis: median **294.7 ms**, max **2617.0 ms**.
- Results within the 1000 ms match interval: **100.0%**.
- Exact music Top-1/Top-3: **100.0% / 100.0%**.
- Genre Top-1/Top-3: **100.0% / 100.0%**.
- Mean realtime factor: **0.990x** (1.0x means wall-clock playback).
- Busy submission attempts: 8 / 46 (17.4%).
- Completed scheduled match slots: 36 / 40 (90.0%).

## Per-file results

| Genre | Music | Matches | Track Top-1 | Genre Top-1 | Median latency | Realtime |
|---|---|---:|---:|---:|---:|---:|
| BR | BR0 | 1 | 100.0% | 100.0% | 966.1 ms | 0.995x |
| HO | HO0 | 3 | 100.0% | 100.0% | 674.9 ms | 0.940x |
| JB | JB0 | 4 | 100.0% | 100.0% | 783.5 ms | 0.993x |
| JS | JS0 | 4 | 100.0% | 100.0% | 698.6 ms | 0.997x |
| KR | KR0 | 4 | 100.0% | 100.0% | 709.1 ms | 0.994x |
| LH | LH0 | 4 | 100.0% | 100.0% | 718.4 ms | 0.996x |
| LO | LO0 | 4 | 100.0% | 100.0% | 713.3 ms | 0.996x |
| MH | MH0 | 4 | 100.0% | 100.0% | 761.7 ms | 0.996x |
| PO | PO0 | 4 | 100.0% | 100.0% | 760.8 ms | 0.995x |
| WA | WA0 | 4 | 100.0% | 100.0% | 768.9 ms | 0.995x |

## Reproduce

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/benchmark_humanoid_matcher_virtual_microphone.py
```

Raw results: `realtime/humanoid_robot/src/test/output/humanoid_matcher_virtual_microphone.json`.
