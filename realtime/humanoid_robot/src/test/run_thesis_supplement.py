"""Version-isolated, serial causal supplement. Never runs on import or dry-run."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

import run_humanoid_matcher_experiments as runner

RAW_ROOT = runner.TEST_DIR / "output" / "thesis_supplement"
SCHEMA_VERSION = 2
FORMAL_EXPECTED = {"smoke": 1, "short": 130, "long": 5}
PILOT_EXPECTED = {"smoke": 1, "short": 18, "long": 3}
# Backward-compatible name used by the original supplement tests and tooling.
EXPECTED = FORMAL_EXPECTED
WINDOWS_HIGH_PERFORMANCE_GUID = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    runner.atomic_replace(temporary, path)


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def active_windows_power_scheme() -> str | None:
    if os.name != "nt":
        return None
    return runner.command_output(["powercfg", "/getactivescheme"])


def require_experiment_power_scheme() -> str | None:
    scheme = active_windows_power_scheme()
    if os.name == "nt" and (
        scheme is None or WINDOWS_HIGH_PERFORMANCE_GUID not in scheme.lower()
    ):
        raise RuntimeError(
            "Realtime thesis experiments require the Windows High performance "
            "power plan. Run 'powercfg /setactive "
            f"{WINDOWS_HIGH_PERFORMANCE_GUID}' and start a new cohort. "
            f"Active scheme: {scheme!r}"
        )
    return scheme


def supplement_matrix(
    catalog, protocol, cc0_root=runner.DEFAULT_CC0_ROOT, *, pilot=False
):
    ablation = runner.run_matrix("ablation", catalog, protocol, cc0_root)
    short = [dict(item, duration=40.0) for item in ablation
             if item["condition"] in {"full", "authored_timing"}]
    if pilot:
        pilot_sources = {
            "aistpp_stitched_test",
            "tempo_jump_90_to_150",
            "silence_recovery",
        }
        short = [
            dict(item, duration=60.0)
            for item in short
            if item["source_id"] in pilot_sources and int(item["seed"]) < 3
        ]
    sources = sorted({item["source_id"] for item in short})
    expected = PILOT_EXPECTED if pilot else FORMAL_EXPECTED
    expected_sources = 3 if pilot else 13
    if len(sources) != expected_sources or len(short) != expected["short"]:
        raise ValueError(
            f"Expected {expected_sources} sources and {expected['short']} short runs; "
            f"got {len(sources)} / {len(short)}"
        )
    long_duration = 306.0 if pilot else 606.0
    long = [dict(item, duration=long_duration) for item in short
            if item["condition"] == "full" and item["source_id"] == "aistpp_stitched_test"]
    smoke = [dict(short[0], duration=10.0)]
    suites = {"smoke": smoke, "short": short, "long": long}
    for name, matrix in suites.items():
        if len(matrix) != expected[name]:
            raise ValueError(f"Wrong {name} matrix count")
        for item in matrix:
            item["condition_arguments"] = [*item["condition_arguments"], "--experiment-causal-file-input"]
    return suites


def validate_causal_artifacts(trace, timing, pose, log, *, minimum_duration_seconds=None):
    valid, errors = runner.validate_run_artifacts(
        trace, timing, pose, log, minimum_duration_seconds=minimum_duration_seconds)
    if not valid:
        return valid, errors
    try:
        timing_data = read_json(timing)
        if timing_data.get("beat_input_mode") != "online_causal":
            errors.append("Timing is not online causal")
        if timing_data.get("onnxruntime_providers") != ["CPUExecutionProvider"]:
            errors.append("Runtime did not use CPUExecutionProvider only")
        with np.load(pose, allow_pickle=False) as data:
            if int(data["schema_version"]) != SCHEMA_VERSION or str(data["beat_input_mode"]) != "online_causal":
                errors.append("Wrong pose schema / beat input mode")
            times = data["wall_time_seconds"]
            audio = data["time_seconds"]
            qpos = data["final_qpos"]
            joints = data["joint_positions"]
            safety = data["safety_counts"]
            events = data["beat_events"]
            count = len(times)
            if not np.array_equal(data["frame_index"], np.arange(count)):
                errors.append("Missing or inconsistent frame indices")
            if not np.allclose(data["audio_received_sample_index"] / float(data["audio_sample_rate_hz"]), audio, atol=1e-7):
                errors.append("Audio cursor metadata differs from receive clock")
            if count < 4 or np.any(np.diff(times) <= 0):
                errors.append("Output wall timestamps must be strictly increasing")
            if len(audio) != count or qpos.shape != (count, int(data["model_nq"])) or safety.shape != (count, 4):
                errors.append("Frame-array length or shape mismatch")
            if np.any(audio > np.maximum(times, 0) + 1e-6):
                errors.append("Received audio is ahead of the actual wall clock")
            if minimum_duration_seconds and (times[-1] < minimum_duration_seconds - .1 or times[0] > .1):
                errors.append("Incomplete actual wall-clock evaluation interval")
            if not np.allclose(qpos[:, data["joint_qpos_indices"]], joints, atol=2e-7):
                errors.append("Joint values differ from final output qpos")
            if len(str(data["model_xml_sha256"])) != 64:
                errors.append("Missing model XML identity")
            if events.ndim != 2 or events.shape[1] != 4:
                errors.append("Malformed beat events")
            elif len(events):
                if np.any(np.diff(events[:, 1]) < 0) or np.any(events[:, 0] > events[:, 1] + 1e-6):
                    errors.append("Beat event/delivery clocks are inconsistent")
                if np.any(events[:, 2] > np.maximum(events[:, 1], 0) + 1e-6):
                    errors.append("Beat analyser received future audio")
            for key in data.files:
                array = data[key]
                # Unlimited joint ranges can legitimately contain infinity.
                if array.dtype.kind == "f" and key != "joint_ranges" and not np.all(np.isfinite(array)):
                    errors.append(f"Non-finite array: {key}")
            if np.any(safety < 0) or np.any(safety[:, :3] > 1):
                errors.append("Invalid safety counters")
            if int(safety[:, 2].sum()) != timing_data["final_residual_clearance_violations"]:
                errors.append("Residual collision counter mismatch")
            if int(safety[:, 3].sum()) != timing_data["final_joint_limit_violations"]:
                errors.append("Final joint-limit counter mismatch")
            if not np.isclose(
                float(data["control_rate_hz"]),
                float(timing_data["control_rate_hz"]),
            ):
                errors.append("Pose and timing control rates differ")
        text = Path(log).read_text(encoding="utf-8", errors="replace")
        if "Traceback (most recent call last)" in text:
            errors.append("Log contains an unhandled Python exception")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        errors.append(f"Causal artifact validation: {exc}")
    # Deliberately do not reject latency/safety failures: they are valid outcomes.
    return not errors, errors


def input_paths(catalog, matrices):
    paths = {catalog.catalog_path, runner.DEFAULT_PROTOCOL, runner.DEFAULT_SCHEMA,
             runner.TEST_DIR / "thesis_supplement.schema.json", runner.TEST_DIR / "run_thesis_experiments.ps1",
             catalog.catalog_path.parent / catalog.metadata["arrays_file"]}
    paths.update(Path(item["audio"]).resolve() for matrix in matrices.values() for item in matrix)
    paths.update(path for key in ("embedding_model", "tag_model")
                 if (path := runner._model_path(catalog, key)) is not None)
    # Include transitive runtime helpers and robot assets, not just entrypoints.
    for folder in (runner.SRC_DIR, runner.ROOT / "realtime" / "robot_arm" / "src"):
        paths.update(path for path in folder.rglob("*.py") if "output" not in path.parts)
    paths.update((runner.HUMANOID_DIR / "data" / "aistpp_gmr").glob("*.pkl"))
    paths.update((runner.HUMANOID_DIR / "data" / "aistpp_gmr").glob("*.json"))
    for path in (runner.HUMANOID_DIR / "assets").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".xml", ".stl", ".obj", ".urdf", ".json"}:
            paths.add(path)
    paths.update((runner.HUMANOID_DIR / "data" / "test_audio").glob("*.json"))
    paths.update((runner.HUMANOID_DIR / "data" / "test_audio" / "matcher_change_streams").glob("*.json"))
    return sorted({path.resolve() for path in paths})


def freeze_manifest(path: Path, payload):
    identity = digest_json(payload)
    document = {"identity": identity, "payload": payload}
    if path.exists():
        old = read_json(path)
        if old != document:
            raise ValueError("Supplement version/input mismatch. Old results are untouched. Restore the frozen version or ask for a new cohort; do not delete the manifest.")
    else:
        atomic_json(path, document)
    return document


@contextmanager
def cohort_lock(root: Path):
    """OS-owned lock releases on crash; no stale-lock deletion is necessary."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another supplement/reanalysis process owns this cohort") from exc
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def validate_cohort(root: Path):
    manifest = read_json(root / "manifest.json")
    if digest_json(manifest["payload"]) != manifest["identity"]:
        raise ValueError("Corrupt frozen manifest")
    runs = {}
    model_hashes = set()
    for suite, expected_matrix in manifest["payload"]["matrix"].items():
        expected = len(expected_matrix)
        status = read_json(root / suite / "run_status.json")
        records = status["runs"]
        keys = {(r["condition"], r["seed"], r["source_id"]) for r in records}
        wanted = {(r["condition"], r["seed"], r["source_id"]) for r in expected_matrix}
        if len(records) != expected or keys != wanted or status["expected_runs"] != expected:
            raise ValueError(f"Incomplete or duplicate {suite} matrix")
        for record in records:
            if record["status"] not in {"completed", "reused"} or record.get("returncode") != 0:
                raise ValueError(f"Unfinished run: {record['run_key']}")
            if record.get("cohort_identity") != manifest["identity"]:
                raise ValueError(f"Cross-version run: {record['run_key']}")
            item = next(item for item in expected_matrix if (item["condition"], item["seed"], item["source_id"])
                        == (record["condition"], record["seed"], record["source_id"]))
            if record["duration"] != item["duration"] or record["condition_arguments"] != item["condition_arguments"]:
                raise ValueError(f"Run configuration differs from manifest: {record['run_key']}")
            if record.get("prepared_audio_sha256") != runner.file_sha256(Path(record["audio"])):
                raise ValueError(f"Prepared audio checksum mismatch: {record['run_key']}")
            if record.get("artifact_sha256") != {
                key: runner.file_sha256(Path(record[key])) for key in ("trace", "timing", "pose", "log")}:
                raise ValueError(f"Artifact checksum mismatch: {record['run_key']}")
            valid, errors = validate_causal_artifacts(
                *(Path(record[key]) for key in ("trace", "timing", "pose", "log")),
                minimum_duration_seconds=float(record["duration"]))
            if not valid:
                raise ValueError(f"Invalid {record['run_key']}: {errors}")
            with np.load(record['pose'], allow_pickle=False) as pose:
                model_hashes.add(str(pose['model_xml_sha256']))
        runs[suite] = records
    known_xml_hashes = {value for name, value in manifest['payload']['input_sha256'].items() if name.lower().endswith('.xml')}
    if len(model_hashes) != 1 or not model_hashes <= known_xml_hashes:
        raise ValueError('Runtime model XML differs across runs or is absent from frozen input manifest')
    return manifest, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RAW_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--control-rate-hz", type=float, default=120.0)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    catalog = runner.MusicCatalog.load(runner.DEFAULT_CATALOG)
    protocol = read_json(runner.DEFAULT_PROTOCOL)
    if not np.isfinite(args.control_rate_hz) or args.control_rate_hz <= 0.0:
        raise ValueError("--control-rate-hz must be positive and finite")
    matrices = supplement_matrix(catalog, protocol, pilot=args.pilot)
    expected = PILOT_EXPECTED if args.pilot else FORMAL_EXPECTED
    if args.dry_run:
        scheme = active_windows_power_scheme()
        print(json.dumps({"dry_run": True, "processes": expected,
                          "formal_runs": expected["short"] + expected["long"],
                          "pilot": args.pilot,
                          "control_rate_hz": args.control_rate_hz,
                          "duration_seconds": {key: matrix[0]["duration"] for key, matrix in matrices.items()},
                          "root": str(args.output_dir), "serial": True,
                          "windows_power_scheme": scheme,
                          "power_scheme_ready": (
                              os.name != "nt"
                              or scheme is not None
                              and WINDOWS_HIGH_PERFORMANCE_GUID in scheme.lower()
                          )}, indent=2))
        return 0
    require_experiment_power_scheme()
    with cohort_lock(args.output_dir):
        if not args.resume and (args.output_dir / "manifest.json").exists():
            raise ValueError("Existing cohort: use --resume; refusing overwrite")
        if shutil.disk_usage(args.output_dir).free < 10 * 1024**3:
            raise ValueError("At least 10 GB free space required")
        print("Checking dependencies, providers and immutable source hashes...", flush=True)
        extractor = runner.make_extractor(catalog)
        environment = runner.environment_manifest(catalog, runner.DEFAULT_PROTOCOL, extractor)
        if environment["onnxruntime_providers"] != ["CPUExecutionProvider"]:
            raise ValueError("Expected CPUExecutionProvider only")
        del extractor
        paths = input_paths(catalog, matrices)
        hashes = {str(path): runner.file_sha256(path) for path in paths}
        stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
        payload = {
            "schema_version": SCHEMA_VERSION, "input_sha256": hashes,
            "environment": {k: v for k, v in environment.items() if k not in {"generated_utc", "git_status", "git_dirty"}},
            "matrix": json.loads(json.dumps(matrices, default=str)),
            "protocol": (
                f"causal wall-paced; {args.control_rate_hz:g} Hz; headless; "
                "collision always; warmup 6 s; safety audit overhead included"
            ),
            "pilot": args.pilot,
            "control_rate_hz": args.control_rate_hz,
        }
        frozen = freeze_manifest(args.output_dir / "manifest.json", payload)
        completed_marker = args.output_dir / "SUPPLEMENT_COMPLETE"
        if completed_marker.exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            runner.atomic_replace(completed_marker, completed_marker.with_name(f'SUPPLEMENT_COMPLETE.{stamp}.previous'))

        def check_unchanged():
            require_experiment_power_scheme()
            for filename, stamp in stats.items():
                stat = Path(filename).stat()
                if (stat.st_size, stat.st_mtime_ns) != stamp:
                    raise ValueError(f"Input changed during experiment: {filename}; stop and preserve results")

        for suite, matrix in matrices.items():
            print(f"Stage {suite}: {len(matrix)} serial runs", flush=True)
            seconds = matrix[0]["duration"]
            root = args.output_dir / suite
            runner.prepare_short_audio_inputs(matrix, root, required_seconds=seconds,
                                              sample_rate=int(catalog.metadata["extractor"]["sample_rate"]), dry_run=False)
            prepared_hashes = {str(path): runner.file_sha256(path) for path in {Path(item['audio']) for item in matrix}}
            for item in matrix:
                item["cohort_identity"] = frozen["identity"]
                item["prepared_audio_sha256"] = prepared_hashes[str(item['audio'])]
            records, failures = runner.execute_runs(
                matrix, root, catalog.catalog_path, max_seconds=seconds, dry_run=False,
                resume=args.resume, suite=f"supplement_{suite}", max_attempts=2,
                artifact_validator=validate_causal_artifacts, before_run=check_unchanged,
                require_completed_status=True, stop_on_failure=True,
                control_rate_hz=args.control_rate_hz)
            if failures:
                raise RuntimeError(f"{suite} stopped after retries. See {root / 'run_status.json'}; rerun with --resume.")
            print(f"Validated {len(records)} {suite} runs (performance failures remain included).", flush=True)
        check_unchanged()
        if any(runner.file_sha256(Path(name)) != value for name, value in hashes.items()):
            raise ValueError("Input hash changed during supplement; completion marker withheld")
        manifest, runs = validate_cohort(args.output_dir)
        atomic_json(args.output_dir / "SUPPLEMENT_COMPLETE", {
            "identity": manifest["identity"],
            "formal_runs": len(runs["short"]) + len(runs["long"]),
            "pilot": args.pilot,
            "control_rate_hz": args.control_rate_hz,
            "integrity_complete": True, "performance_passed": "not implied; evaluate offline"})
        print(f"Supplement complete: {args.output_dir / 'SUPPLEMENT_COMPLETE'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
