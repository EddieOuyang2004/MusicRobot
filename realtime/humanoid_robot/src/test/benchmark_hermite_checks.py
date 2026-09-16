"""Compare original extrema checking with optimized checks on captured JSONL.

python benchmark_hermite_checks.py --cases PATH --report PATH
Uses captured limits unchanged; warms JIT before timing. No playback required.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_bridges import HermiteBridge, JointState, InfeasibleBridge, make_bridge, warmup_hermite


def original_check(curve, derivative, lower, upper):
    lo, hi = curve.extrema(derivative)
    return bool(np.all(lo >= lower-1e-7) and np.all(hi <= upper+1e-7))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    inputs = []
    for line in args.cases.read_text().splitlines():
        c = json.loads(line)
        values = {k: np.asarray(c[k], dtype=float) for k in
                  ('lower', 'upper', 'speed', 'acceleration', 'jerk')}
        values.update(start=JointState(*map(np.asarray, c['start'])),
                      end=JointState(*map(np.asarray, c['end'])),
                      minimum=c['minimum'], maximum=c['maximum'])
        inputs.append(values)
    if not inputs:
        parser.error('Empty cases file')
    warmup_hermite()
    optimized = HermiteBridge.within_limits
    timings = {'original': [], 'optimized': []}
    reference = None
    for repeat in range(args.repeats):
        order = ('original', 'optimized') if repeat % 2 == 0 else ('optimized', 'original')
        for label in order:
            decisions = []
            with patch.object(HermiteBridge, 'within_limits', original_check if label == 'original' else optimized):
                for values in inputs:
                    started = time.perf_counter()
                    try:
                        duration = make_bridge('hermite', **values).duration
                    except InfeasibleBridge:
                        duration = None
                    timings[label].append((time.perf_counter()-started)*1000)
                    decisions.append(duration)
            if reference is None:
                reference = decisions
            if reference != decisions:
                raise AssertionError('Acceptance or selected duration differs')
    report = dict(cases=len(inputs), repeats=args.repeats,
                  identical_decisions_and_durations=True,
                  accepted=sum(x is not None for x in reference),
                  results={label: dict(mean_ms=float(np.mean(t)), p95_ms=float(np.percentile(t,95)))
                           for label, t in timings.items()})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
