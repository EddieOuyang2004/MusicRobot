"""Experiment-only wrapper: log each delivered retrieval exactly once."""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import realtime_music_humanoid_matcher_v2 as v2

    original_poll = v2.RetrievalWorker.poll
    path = Path(os.environ["MATCHER_AUDIT_JSONL"])
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        def audited_poll(worker):
            nonlocal count
            result = original_poll(worker)
            if result is not None:
                count += 1
                handle.write(json.dumps({"index": count, "monotonic_seconds": time.perf_counter(),
                                         "result": asdict(result)}) + "\n")
                handle.flush()
            return result

        v2.RetrievalWorker.poll = audited_poll
        try:
            return v2.main()
        finally:
            v2.RetrievalWorker.poll = original_poll


if __name__ == "__main__":
    raise SystemExit(main())
