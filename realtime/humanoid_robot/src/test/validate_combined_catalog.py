"""Run real headless transitions in both directions for a combined catalog."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from music_motion_catalog import MusicCatalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=32.)
    parser.add_argument("--aistpp-motion-id", help="Use a segmented child as reference and startup motion")
    parser.add_argument("--short-startup-check", action="store_true",
                        help="Also report preparation holds from the shortest FineDance startup")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalog = MusicCatalog.load(args.catalog.resolve())
    # Stable reference pair previously validated with the project's bridge planner.
    ids = dict(aistpp="gBR_sBM_cAll_d04_mBR0_ch01", finedance="finedance_001_0000000_0000586")
    if args.aistpp_motion_id:
        ids["aistpp"] = args.aistpp_motion_id
        if ids["finedance"] not in catalog.motions:
            ids["finedance"] = next(p.motion_id for p in catalog.motions.values() if p.dataset_id == "finedance")
    for motion_id in ids.values():
        if motion_id not in catalog.motions:
            parser.error(f"Reference motion absent from catalog: {motion_id}")
    # The full AIST++ candidate pool contains long clips whose grounding is slower.
    # Keep the default ten-candidate policy, but give reverse-start preparation
    # the longest authored FineDance clip rather than forcing a 9.77-second start.
    fine_start = max((p for p in catalog.motions.values() if p.dataset_id == "finedance"),
                     key=lambda p: (p.duration_seconds, -p.velocity_p90, p.motion_id)).motion_id
    initial_ids = dict(ids, finedance=fine_start)
    report = dict(catalog=str(catalog.catalog_path), build_id=catalog.metadata.get("build_id"), runs=[])
    runs = [("aistpp", "finedance", initial_ids["aistpp"], False),
            ("finedance", "aistpp", initial_ids["finedance"], False)]
    if args.short_startup_check:
        shortest = min((p for p in catalog.motions.values() if p.dataset_id == "finedance"),
                       key=lambda p: (p.duration_seconds, p.motion_id)).motion_id
        runs.append(("finedance", "aistpp", shortest, True))
    required_failures = []
    for initial, target, startup_id, diagnostic in runs:
        name = initial + "_to_" + target + ("_short_startup" if diagnostic else "")
        profile = catalog.motions[ids[target]]
        variant = catalog.tracks[profile.music_id]["variants"][0]
        audio, sample_rate = sf.read(catalog.audio_file(variant), always_2d=True)
        repeats = int(np.ceil((args.seconds + 5) * sample_rate / len(audio)))
        audio_path = output / (name + ".wav")
        sf.write(audio_path, np.tile(audio, (repeats, 1)), sample_rate)
        trace_path, timing_path = output / (name + ".csv"), output / (name + "_timing.json")
        command = [sys.executable, "-u", str(SRC / "realtime_music_humanoid_matcher.py"),
                   "--catalog", str(catalog.catalog_path), "--initial-motion-id", startup_id,
                   "--audio-input", str(audio_path), "--headless", "--realtime",
                   "--audio-input-delay-sec", "1", "--max-seconds", str(args.seconds),
                   "--motion-timing", "authored", "--trace-csv", str(trace_path),
                   "--timing-report", str(timing_path)]
        environment = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = "1"
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            # A different CWD exercises catalog-relative dataset/model paths,
            # including the spawned retrieval worker.
            subprocess.run(command, cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=180, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        with trace_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        dataset_sequence = []
        for row in rows:
            dataset = catalog.motions[row["current_motion_id"]].dataset_id
            if not dataset_sequence or dataset_sequence[-1] != dataset:
                dataset_sequence.append(dataset)
        crossed = (initial, target) in list(zip(dataset_sequence, dataset_sequence[1:]))
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        retrieval_max = timing["retrieval_feature_ms_max"] + timing["retrieval_match_ms_max"]
        item = dict(direction=name, initial_motion_id=startup_id, dataset_sequence=dataset_sequence, crossed=crossed,
                    diagnostic=diagnostic,
                    terminal_hold_rows=sum(row.get("bridge_preparation_status") == "terminal_hold" for row in rows),
                    startup_ms=timing["startup_until_control_loop_ms"],
                    retrieval_completed=timing["retrieval_completed"],
                    retrieval_busy_submissions=timing["retrieval_busy_submissions"],
                    retrieval_max_ms_upper_bound=retrieval_max,
                    failed_motion_ids=sorted({row["failed_motion_ids"] for row in rows
                                              if row["failed_motion_ids"] not in ("", "[]")}),
                    trace=str(trace_path), timing=str(timing_path))
        report["runs"].append(item)
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(item), flush=True)
        if (not crossed or item["failed_motion_ids"] or retrieval_max >= 1000) and not diagnostic:
            required_failures.append(name)
    report["passed"] = not required_failures
    report["required_failures"] = required_failures
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if required_failures:
        raise RuntimeError(f"Combined playback acceptance failed: {required_failures}; inspect summary and traces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
