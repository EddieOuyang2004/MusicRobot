# V2 transition acceleration review — 2026-09-14

## Recommendation

Optimize CPU rejection of infeasible Hermite durations first if the existing
trajectory shape must be preserved. Evaluate the already-integrated Ruckig CPU
backend if changing trajectory construction is acceptable. A CUDA port is a
later option for large batches, not the first step for this single humanoid.

The two state-to-state backends are `hermite` and `ruckig`. V2 also exposes the
legacy `quintic` mode; it is not an equivalent replacement for the checked
authored-position/velocity/acceleration bridges compared here.

## Measurements on this machine

Environment: Windows, Python 3.11.14, NumPy 1.26.4, Ruckig 0.19.4. NVIDIA-SMI
detects an RTX 3080 with 10 GB VRAM, driver 591.86, advertising CUDA compatibility
13.1. This does not establish that the CUDA toolkit is installed. Torch and CuPy
are absent from the project environment; no GPU implementation was benchmarked.

Captured 212 actual `make_bridge` inputs from a 25-second, headless Chronos run,
seed 42, default five-motion shortlist, 120 Hz, causal file input. Each input has
29 joints. The capture reproduced one completed transition followed by a terminal
hold. Its disk instrumentation perturbs live timing; use it as an input capture,
not an uninstrumented performance baseline.

Replayed the inputs sequentially, outside playback/audio processing, three times
per backend, after a first-case warmup. Both backends received the same finite
jerk override. Profiling ran separately from the wall-time measurements.

### Captured cases, experimental jerk = 50,000 rad/s³

This jerk value is a benchmark parameter, **not a recommended physical limit**.

| Solver | Mean per attempt | p95 | Accepted unique cases |
|---|---:|---:|---:|
| Current Hermite | 24.96 ms | 26.72 ms | 1 / 212 |
| Hermite with experimental rejection prefilter | 1.20 ms | 1.56 ms | 1 / 212 |
| Existing Ruckig wrapper | 0.109 ms | 0.163 ms | 194 / 212 |

The Hermite prototype is **20.8× faster on this failure-heavy corpus**. It matched
all 212 original decisions and accepted durations. One pass over the corpus
averaged 5.291 s for Hermite, 0.254 s with the prefilter, and 0.0232 s for Ruckig.
These are exhaustive solver replays, not end-to-end search times: live search
stops at its first success, and candidate ranking/loading are excluded.

Ruckig's acceptance differs because it constructs a different jerk-limited
trajectory, rather than searching durations for a single quintic Hermite family.
At experimental jerk = 500 rad/s³, both backends rejected all 212 inputs;
Ruckig averaged 0.079 ms primarily through early boundary-state rejection, while
Hermite averaged 25.60 ms. A speed comparison without feasibility counts would
therefore be misleading. Finite jerk settings must be chosen independently from
the robot's actual requirements, then evaluated across motions.

### Additional accepted-case check

On 64 seeded synthetic 29-joint cases, all accepted, with jerk = 500 rad/s³:

| Solver | Mean per attempt |
|---|---:|
| Current Hermite | 6.33 ms |
| Hermite prefilter | 7.07 ms |
| Ruckig | 0.108 ms |

The prototype matched every Hermite decision and duration here too, but incurred
about 12% overhead on easy successful cases. This supports targeting expensive
rejections, not claiming a universal 21× speedup. Ruckig's median motion duration
was 0.464 s versus Hermite's 0.556 s; computation time and motion duration are
separate measurements. The synthetic cases used seed 42, positions in
[-0.5, 0.5], velocities/accelerations in [-0.2, 0.2], position limits ±2,
speed limits 3, and acceleration limits 20, with all units as in motion_bridges.py.

## Bottlenecks and concrete changes

1. **Hermite extrema dominate.** A separate profile of the 212-case jerk-500
   replay spent 11.018 of 11.264 instrumented seconds in `HermiteBridge.extrema`,
   calling `polyroots` 122,235 times. The method loops over every joint and solves
   small companion-matrix eigenproblems, repeatedly across trial durations.
   See `motion_bridges.py:67` and `motion_bridges.py:155`.
2. **Add a rejection-only prefilter.** The experiment evaluates nine normalized
   times across all joints before solving roots. If any point already violates a
   limit, reject that duration; otherwise retain the original exact-extrema
   calculation. Sampled values never authorize a bridge. This prototype only
   exists in the offline benchmark; production integration needs boundary and
   numerical regression coverage. Testing position samples first and only using
   further filters when beneficial may reduce successful-case overhead.
3. **Reduce repeated work.** Hoist source state, sorted joint names, index maps,
   and limit arrays outside the entry loop in `prepare_state_bridge`. Precompute
   target endpoint validity masks, including acceleration. Validate the common
   source once before trying targets. Stop exact per-joint checking as soon as a
   violating joint is found. Consider compiled CPU polynomial evaluation/root
   routines after these algorithmic savings; preserve float64/tolerance behavior.
4. **Use Ruckig's existing CPU backend as the alternative.** V2 already supports
   `--transition-backend ruckig`, but requires finite jerk limits for every joint
   through configuration or `--output-max-joint-jerk`. Keep the existing position
   extrema rejection. The installed wrapper constructs generator/input/trajectory
   objects each call; reuse worker-local buffers if profiling justifies it.
   Ruckig is a C++ library with Python bindings, and this state-to-state use is
   local. It does not become a CUDA solver by changing a flag.
5. **Bound preparation latency independently of solver speed.** Search currently
   waits for the whole frozen shortlist, tries entries without a wall-time budget,
   and uses a scoring thread in the control process. Prepare a verified fallback
   early, allow ready candidates to be considered, impose a deadline/cancellation
   policy, and use a persistent process worker with compact arrays/cached data.
   Process isolation reduces Python contention; it does not itself guarantee a
   shorter solve or real-time deadlines. Terminal hold needs explicit recovery
   planning from the actual held state. A faster rejection still leaves no plan.

The earlier `v2_transition_timing_review.md` documents a live search taking
21.108 s, with 21.051 s inside 211 rejected bridge attempts, and several seconds
of candidate loading. Those measurements involved different runtime contention
and must not be compared directly to the isolated replay as a measured speedup.

## GPU/CUDA assessment

This implementation makes many sequential small numerical calls. Simply
replacing NumPy with CuPy would not remove Python branching and per-joint calls,
and would introduce device launches, transfers, and synchronization. NVIDIA
recommends batching small transfers; CuPy documents startup/compilation overhead
and the need for synchronized GPU timing. The assessment that a naive port is
unlikely to help is an engineering inference, not a measured RTX 3080 result.

A useful GPU design would batch many candidate × duration × joint checks in
resident arrays, use fused polynomial evaluation/rejection kernels, and return a
small result set for continuous validation. This is more attractive for offline
transition-library generation or many robots than one 29-joint bridge. Respect
non-monotonic Hermite feasibility as duration increases; a binary search for
duration is not generally valid with nonzero endpoint derivatives.

Precomputing grounded trajectories and pairwise bridges is also useful because
these state-bridge endpoints use authored derivatives. Cache keys must include
motion content, retarget/grounding configuration, joint ordering and limits,
backend, and duration settings. Full all-pairs caching may be too large; cache
frequent pairs or validated fallbacks first.

## Reproduction and artifacts

Added `benchmark_v2_bridge_backends.py`. Production matcher/solver files were
not modified. Capture writes exclusively to a new JSONL file. It records inputs
and leaves solver choices unchanged; replay applies explicit experimental jerk
limits and tests both original backends plus the temporary prefilter.

```powershell
.venv/Scripts/python.exe realtime/humanoid_robot/src/test/benchmark_v2_bridge_backends.py capture --cases tmp/v2_backend_cases_chronos.jsonl --headless --realtime --audio-input realtime/humanoid_robot/data/test_audio/chronos.mp3 --experiment-causal-file-input --experiment-pose-npz tmp/v2_backend_capture_poses.npz --max-seconds 25 --initial-motion-seed 42 --trace-csv tmp/v2_backend_capture.csv --timing-report tmp/v2_backend_capture.json
.venv/Scripts/python.exe realtime/humanoid_robot/src/test/benchmark_v2_bridge_backends.py replay --cases tmp/v2_backend_cases_chronos.jsonl --report tmp/v2_backend_benchmark_50000.json --jerk 50000 --repeats 3
```

Use a fresh cases filename to repeat capture. Raw cases, reports and separate
`.profile.txt` outputs are under `tmp/v2_backend_*`. The additional accepted-case
inputs/report use `tmp/v2_backend_synthetic*`. These are exploratory benchmarks
on a shared desktop, with fixed backend order and a limited corpus; no claim of
full-track continuity or physical robot validation follows from them.

## External references

- [NumPy polyroots: companion-matrix eigenvalues and numerical caveats](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyroots.html)
- [Ruckig source and C++/Python API](https://github.com/pantor/ruckig)
- [CuPy performance and GPU timing guidance](https://docs.cupy.dev/en/stable/user_guide/performance.html)
- [NVIDIA CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
