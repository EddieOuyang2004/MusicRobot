"""Capture real v2 solver inputs, then compare backends off the control loop.

Capture: python benchmark_v2_bridge_backends.py capture --cases PATH [v2 flags]
Replay:  python benchmark_v2_bridge_backends.py replay --cases PATH --report PATH
         --jerk 500 --repeats 3
The finite jerk override is an experiment parameter, not a robot limit recommendation.
Hermite uses the current production implementation. For an original-extrema vs
optimized comparison, use benchmark_hermite_checks.py instead. The historical
hermite_prefilter variant only wraps extrema and may be redundant after optimization.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_bridges import JointState, HermiteBridge, InfeasibleBridge, make_bridge, require_ruckig


def capture(path, remaining):
    import realtime_music_humanoid_matcher_v2 as v2
    original = v2.make_bridge
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids overwriting an earlier experiment.
    with path.open("x", encoding="utf-8") as output:
        def recorded(backend, start, end, lower, upper, speed, acceleration, jerk,
                     minimum=.35, maximum=10., **search_options):
            record = dict(backend=backend, minimum=minimum, maximum=maximum)
            for label, state in (("start", start), ("end", end)):
                record[label] = [getattr(state, f).tolist() for f in
                                 ("position", "velocity", "acceleration")]
            for label, value in (("lower", lower), ("upper", upper), ("speed", speed),
                                 ("acceleration", acceleration), ("jerk", jerk)):
                record[label] = [float(x) if np.isfinite(x) else str(float(x)) for x in value]
            output.write(json.dumps(record) + "\n")
            output.flush()
            return original(backend, start, end, lower, upper, speed, acceleration,
                            jerk, minimum, maximum, **search_options)
        v2.make_bridge = recorded
        sys.argv = [sys.argv[0], *remaining]
        try:
            v2.main()
        finally:
            v2.make_bridge = original


def replay(args):
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line]
    if not cases:
        raise ValueError("No captured cases")
    require_ruckig()  # Exclude import cost from warm measurements.
    results = {}
    reference_outcomes = None
    for backend in ("hermite", "hermite_prefilter", "ruckig"):
        timings, outcomes, durations = [], Counter(), []
        first_pass = []
        inputs = []
        for case in cases:
            inputs.append(dict(start=JointState(*map(np.asarray, case["start"])),
                               end=JointState(*map(np.asarray, case["end"])),
                               **{k: np.asarray(case[k], dtype=float) for k in
                                  ("lower", "upper", "speed", "acceleration")},
                               jerk=np.full(len(case["speed"]), args.jerk),
                               minimum=case["minimum"], maximum=case["maximum"]))
        def solve(values):
            original_extrema = HermiteBridge.extrema
            def prefiltered(curve, derivative=0):
                # Rejection-only experiment: every accepted duration still goes
                # through the original continuous polynomial extrema checks.
                coeff = np.polynomial.polynomial.polyder(curve.coefficients, m=derivative)
                sampled = np.polynomial.polynomial.polyval(np.linspace(0., 1., 9), coeff) / curve.duration**derivative
                lo, hi = sampled.min(axis=1), sampled.max(axis=1)
                if derivative == 0:
                    lower, upper = values["lower"], values["upper"]
                else:
                    upper = values[("speed", "acceleration", "jerk")[derivative-1]]
                    lower = -upper
                if np.any(lo < lower-1e-7) or np.any(hi > upper+1e-7):
                    return lo, hi
                return original_extrema(curve, derivative)
            try:
                with (patch.object(HermiteBridge, "extrema", prefiltered)
                      if backend == "hermite_prefilter" else nullcontext()):
                    curve = make_bridge("hermite" if backend == "hermite_prefilter" else backend, **values)
                return "accepted", curve.duration
            except InfeasibleBridge as exc:
                return str(exc), None
        solve(inputs[0])
        for repeat in range(args.repeats):
            for values in inputs:
                started = time.perf_counter()
                outcome, duration = solve(values)
                timings.append((time.perf_counter()-started)*1000)
                outcomes[outcome] += 1
                if repeat == 0:
                    first_pass.append((outcome, duration))
                if duration is not None:
                    durations.append(duration)
        results[backend] = dict(calls=len(timings), total_ms=sum(timings),
            mean_ms=float(np.mean(timings)), median_ms=float(np.median(timings)),
            p95_ms=float(np.percentile(timings, 95)), max_ms=max(timings),
            outcomes=dict(outcomes),
            accepted_duration_median=float(np.median(durations)) if durations else None)
        if backend == "hermite":
            reference_outcomes = first_pass
        elif backend == "hermite_prefilter":
            results[backend]["matches_hermite_outcomes_and_durations"] = first_pass == reference_outcomes
        print(backend, json.dumps(results[backend]), flush=True)
        # Profiling is separate from wall-time samples.
        profile = cProfile.Profile()
        profile.enable()
        for values in inputs:
            solve(values)
        profile.disable()
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(25)
        args.report.with_suffix(f".{backend}.profile.txt").write_text(stream.getvalue())
    report = dict(cases=len(cases), repeats=args.repeats, jerk_override=args.jerk,
                  joint_counts=sorted(set(len(c["speed"]) for c in cases)),
                  results=results,
                  note="Same captured inputs and finite jerk override for both backends; solver-only, not live playback.")
    args.report.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "replay"))
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--jerk", type=float)
    parser.add_argument("--repeats", type=int, default=3)
    args, remaining = parser.parse_known_args()
    if args.mode == "capture":
        capture(args.cases, remaining)
    else:
        if remaining or args.report is None or args.jerk is None or not np.isfinite(args.jerk) or args.jerk <= 0 or args.repeats < 1:
            parser.error("Replay requires --report, positive finite --jerk, positive --repeats, and no v2 flags")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        replay(args)
