"""Run v2 unchanged, recording background bridge search cost as JSON lines.

Usage: python profile_humanoid_v2_bridges.py --bridge-profile PATH [v2 flags]
Run headless for timing comparisons; instrumentation adds two clocks per attempt.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import realtime_music_humanoid_matcher_v2 as v2


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bridge-profile", type=Path, required=True)
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    original_prepare = v2.prepare_state_bridge
    original_make = v2.make_bridge
    record = {}

    def measured_make(*positional, **keywords):
        started = time.perf_counter()
        record["attempts"] += 1
        try:
            return original_make(*positional, **keywords)
        except v2.InfeasibleBridge:
            record["rejected"] += 1
            raise
        finally:
            record["solver_seconds"] += time.perf_counter() - started

    def measured_prepare(generation, current_id, source, ready, *other):
        # V2 uses one bridge-scoring worker, so these records cannot overlap.
        record.clear()
        record.update(generation=generation, source=current_id,
                      candidates=[item[0] for item in ready], attempts=0,
                      rejected=0, solver_seconds=0.)
        started = time.perf_counter()
        try:
            plan = original_prepare(generation, current_id, source, ready, *other)
            record.update(target=plan.motion_id, replay=plan.replay,
                          bridge_seconds=plan.trajectory.duration)
            return plan
        except Exception as exc:
            record["error"] = str(exc)
            raise
        finally:
            record["preparation_seconds"] = time.perf_counter() - started
            with args.bridge_profile.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    args.bridge_profile.parent.mkdir(parents=True, exist_ok=True)
    args.bridge_profile.write_text("", encoding="utf-8")
    v2.make_bridge = measured_make
    v2.prepare_state_bridge = measured_prepare
    try:
        v2.main()
    finally:
        v2.make_bridge = original_make
        v2.prepare_state_bridge = original_prepare


if __name__ == "__main__":
    main()
