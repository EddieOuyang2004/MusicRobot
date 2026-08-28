# Humanoid matcher performance result

Generated: `2026-08-23T01:35:24.673478+00:00`

## Summary

- Catalog: 348 segments, 60 tracks, 411 motions (411 preflight-passed).
- Core match latency: median **2.158 ms**, p95 **2.465 ms**, p99 **2.625 ms**.
- Core throughput: **470.2 queries/s** over 200 timed calls.
- Leave-one-music genre Recall@1/3/5: **25.9% / 50.6% / 64.7%**.
- Random genre baselines @1/3/5: 8.5% / 23.7% / 36.8%.
- Source-track motion Recall@1/3/10: **97.1% / 98.6% / 100.0%**.

The exact stored-segment track Recall@1 is a pipeline sanity check, not an out-of-sample accuracy estimate, because the queried descriptor is present in the catalog. Leave-one-music genre retrieval excludes every segment belonging to the query music ID and is the more informative quality measure.

## Startup and memory

- Catalog load: 31.287 ms.
- Matcher initialization: 1.962 ms.
- Catalog feature arrays: 2.27 MiB.
- Matcher normalized indexes: 2.26 MiB.
- Process RSS after initialization: 86.80 MiB (delta 7.01 MiB).

## Per-genre leave-one-music recall

| Genre | Queries | R@1 | R@3 | R@5 |
|---|---:|---:|---:|---:|
| BR | 40 | 10.0% | 82.5% | 87.5% |
| HO | 27 | 85.2% | 100.0% | 100.0% |
| JB | 31 | 87.1% | 100.0% | 100.0% |
| JS | 34 | 0.0% | 5.9% | 8.8% |
| KR | 36 | 11.1% | 38.9% | 58.3% |
| LH | 32 | 9.4% | 40.6% | 65.6% |
| LO | 36 | 8.3% | 38.9% | 80.6% |
| MH | 34 | 67.6% | 100.0% | 100.0% |
| PO | 38 | 7.9% | 21.1% | 21.1% |
| WA | 40 | 0.0% | 0.0% | 40.0% |

## Audio pipeline

- Waveform-to-match latency: median 44.243 ms, p95 62.806 ms.
- Gain-invariant top-1: 0/0 (0.0%) for 0.1x/0.5x/1x/2x.

## Reproduce

From the repository root:

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/benchmark_humanoid_matcher.py
```

Raw machine-readable results: `realtime/humanoid_robot/src/test/output/humanoid_matcher_6s_latency_20260823.json`.
