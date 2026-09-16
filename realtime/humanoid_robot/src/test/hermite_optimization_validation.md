# Hermite optimization validation — 2026-09-15

Implemented sampled rejection, Bernstein convex-hull acceptance, bounded
de Casteljau subdivision (depth three), and original polynomial extrema fallback
for uncertain joints. Float64 roundoff margins leave near-bound decisions to the
fallback. No fastmath. Native compilation is optional, cached, releases the GIL,
and is warmed before v2 playback. Original extrema remains available unchanged.
Backend, boundary states, limit settings, candidate ordering and duration search
are unchanged. No CUDA dependency or lifecycle/recovery policy change.

## Validation

- 4 new differential tests pass: 120 random 8-joint polynomials, four derivative
  orders and three margins around computed extrema; forced fallback; constants
  and infinite bounds; native/Python agreement.
- Existing 25 motion bridge and 16 v2 matcher tests pass (45 tests total).
- 212 captured Chronos attempts and 64 successful synthetic cases keep identical
  acceptance decisions and selected durations versus original extrema checking.

Three-repeat paired checker benchmark, alternating execution order and using
the captured limits unchanged (no jerk override):

| Corpus | Original mean | Optimized mean | Speedup |
|---|---:|---:|---:|
| Chronos, 212 cases | 80.97 ms | 1.873 ms | 43.2× |
| Synthetic, 64 accepted cases | 7.017 ms | 0.306 ms | 22.9× |

These are exploratory shared-desktop solver timings, not hard real-time
guarantees. Different runs showed substantial host timing variation. Compilation
is excluded. The reproducible benchmark replaces only the checker, leaving the
same surrounding solver for both variants. Reports:
`tmp/v2_hermite_chronos_paired.json`, `tmp/v2_hermite_synthetic_paired.json`.

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/test/benchmark_hermite_checks.py --cases tmp/v2_backend_cases_chronos.jsonl --report tmp/v2_hermite_chronos_paired.json --repeats 3
```

## Full matcher smoke run

25-second headless causal Chronos run, seed 42, 120 Hz, default Hermite and
shortlist. Completed the first transition. The next search finished with
`Candidates and replay infeasible`, resulting in terminal hold; the earlier
capture reported unfinished preparation at that join. Optimization does not
resolve infeasibility or add hold recovery. No physical robot or collision
validation was performed.

Control work p95 5.68 ms, p99 8.48 ms; 67 deadline misses out of 3028 iterations.
Thus the run is not deadline-miss-free. Artifacts use
`tmp/v2_hermite_optimized` (CSV, log, poses) and
`tmp/v2_hermite_optimized_timing.json`.
