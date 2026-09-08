"""Validate and consolidate the immutable Humanoid Matcher thesis results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent
HUMANOID_DIR = SRC_DIR.parent
ROOT = HUMANOID_DIR.parents[1]
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from run_humanoid_matcher_experiments import (  # noqa: E402
    atomic_replace,
    file_sha256,
    load_feature_bundle,
    validate_run_artifacts,
)
from humanoid_matcher_experiment_metrics import (  # noqa: E402
    bootstrap_mean_ci,
    distribution_summary,
)


DEFAULT_RAW_ROOT = TEST_DIR / "output" / "thesis_final"
DEFAULT_OUTPUT = ROOT / "docs" / "thesis" / "experiment_results"
EXPECTED_RUNS = {
    "preflight": 1,
    "literature": 200,
    "full": 350,
    "ablation": 520,
    "longrun": 40,
}
EXPECTED_DURATIONS_SECONDS = {
    "preflight": 10.0,
    "literature": 26.0,
    "full": 30.0,
    "ablation": 40.0,
    "longrun": 606.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    atomic_replace(temporary, path)


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"{path}: {type(error).__name__}: {error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path}: expected a JSON object")
        return {}
    return value


def _identity(manifest: dict[str, Any]) -> dict[str, Any]:
    runtime_keys = {
        "catalog_matcher",
        "realtime_matcher",
        "realtime_dancer",
        "robot_motion",
    }
    return {
        "git_commit": manifest.get("git_commit"),
        "catalog_sha256": manifest.get("catalog", {}).get("sha256"),
        "catalog_arrays_sha256": manifest.get("catalog", {}).get("arrays_sha256"),
        "models": {
            key: value.get("sha256")
            for key, value in sorted(manifest.get("models", {}).items())
        },
        "runtime_implementation_sha256": {
            key: value
            for key, value in manifest.get("implementation_sha256", {}).items()
            if key in runtime_keys
        },
        "protocol_sha256": manifest.get("protocol", {}).get("sha256"),
        "schema_sha256": manifest.get("output_schema", {}).get("sha256"),
    }


def validate_suite(
    name: str,
    suite_root: Path,
    expected: int,
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    status = _load_json(suite_root / "run_status.json", errors)
    report = _load_json(suite_root / "experiment_report.json", errors)
    runs = status.get("runs", [])
    if status.get("suite") not in ({"smoke"} if name == "preflight" else {name}):
        errors.append(f"{name}: incorrect suite identity {status.get('suite')!r}")
    if status.get("expected_runs") != expected or len(runs) != expected:
        errors.append(
            f"{name}: expected {expected} runs, status declares {status.get('expected_runs')} "
            f"and contains {len(runs)}"
        )
    run_keys = [run.get("run_key") for run in runs]
    if len(set(run_keys)) != len(run_keys) or any(not key for key in run_keys):
        errors.append(f"{name}: run keys are missing or non-unique")
    for run in runs:
        if run.get("status") not in {"completed", "reused"}:
            errors.append(f"{name}/{run.get('run_key')}: terminal state is {run.get('status')!r}")
            continue
        paths = [Path(run[key]) for key in ("trace", "timing", "pose", "log")]
        valid, artifact_errors = validate_run_artifacts(
            *paths,
            minimum_duration_seconds=EXPECTED_DURATIONS_SECONDS[name],
        )
        if not valid:
            errors.append(f"{name}/{run.get('run_key')}: {'; '.join(artifact_errors)}")
    if report.get("failures"):
        errors.append(f"{name}: report contains {len(report['failures'])} unresolved failures")
    return status, report


def validate_feature_bundle(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected = {
                "real_kinetic": (40, 72),
                "output_kinetic": (200, 72),
                "real_geometric": (40, 32),
                "output_geometric": (200, 32),
            }
            for key, shape in expected.items():
                if key not in archive.files or archive[key].shape != shape:
                    actual = archive[key].shape if key in archive.files else None
                    errors.append(f"feature bundle {key}: expected {shape}, got {actual}")
            if "extractor_identity" not in archive.files:
                errors.append("feature bundle has no extractor_identity")
        return load_feature_bundle(path)
    except (OSError, ValueError, TypeError) as error:
        errors.append(f"{path}: {type(error).__name__}: {error}")
        return None


def write_summary_csv(path: Path, suites: dict[str, Any], feature_metrics: Any) -> None:
    rows = []
    for name, payload in suites.items():
        status = payload["status"]
        report = payload["report"]
        rows.extend(
            [
                {"suite": name, "section": "completion", "metric": "expected_runs", "value": status["expected_runs"]},
                {"suite": name, "section": "completion", "metric": "completed_runs", "value": len(status["runs"])},
                {"suite": name, "section": "history", "metric": "failed_attempts_retained", "value": len(status.get("attempt_history", []))},
                {"suite": name, "section": "acceptance", "metric": "all_evaluated_passed", "value": report.get("acceptance", {}).get("all_evaluated_passed")},
            ]
        )
    for key, value in (feature_metrics or {}).items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            rows.append({"suite": "literature", "section": "fid_div", "metric": key, "value": value})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("suite", "section", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)


def copy_report_artifacts(suites: dict[str, Any], output_dir: Path) -> list[str]:
    copied = []
    destination_root = output_dir / "suite_artifacts"
    for name, payload in suites.items():
        report = payload["report"]
        for group in ("tables", "figures"):
            for label, source_value in report.get("artifacts", {}).get(group, {}).items():
                source = Path(source_value)
                if not source.is_file():
                    continue
                destination = destination_root / group / f"{name}_{label}{source.suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied.append(str(destination.resolve()))
    return copied


def statistical_summary(status: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Aggregate frames to runs, seeds to sources, then bootstrap music units."""

    run_by_artifact: dict[str, dict[str, Any]] = {}
    for run in status.get("runs", []):
        for key in ("trace", "pose", "timing"):
            run_by_artifact[str(Path(run[key]).resolve())] = run
    grouped: dict[tuple[str, str, str], list[float]] = {}

    def add(artifact: str, metric: str, value: Any) -> None:
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            return
        run = run_by_artifact.get(str(Path(artifact).resolve()))
        if run is None:
            return
        key = (str(run["condition"]), str(run["source_id"]), metric)
        grouped.setdefault(key, []).append(float(value))

    for pose in report.get("pose_aggregates", []):
        artifact = pose.get("pose_npz", "")
        add(artifact, "bas_harmonic", pose.get("beat", {}).get("bas_harmonic"))
        add(artifact, "pfc", pose.get("physical", {}).get("pfc"))
        add(artifact, "fsr", pose.get("physical", {}).get("fsr"))
        add(
            artifact,
            "joint_jerk_p95_rad_s3",
            pose.get("continuity", {}).get("joint_jerk_rad_s3", {}).get("p95"),
        )
    for trace in report.get("trace_aggregates", []):
        artifact = trace.get("trace", "")
        add(
            artifact,
            "normalized_selection_entropy",
            trace.get("selection", {}).get("normalized_selection_entropy"),
        )
        add(artifact, "unique_motions", trace.get("selection", {}).get("unique_motions"))
        add(artifact, "mean_match_score", trace.get("match_score", {}).get("mean"))
    for timing in report.get("timing", {}).get("runs", []):
        artifact = timing.get("path", "")
        for metric in (
            "retrieval_waveform_to_match_ms_p95",
            "work_ms_p99",
            "deadline_miss_ratio",
        ):
            add(artifact, metric, timing.get(metric))

    conditions = sorted({condition for condition, _source, _metric in grouped})
    metrics = sorted({metric for _condition, _source, metric in grouped})
    result: dict[str, Any] = {}
    for condition in conditions:
        result[condition] = {}
        for metric in metrics:
            source_means = [
                float(np.mean(values))
                for (item_condition, _source, item_metric), values in grouped.items()
                if item_condition == condition and item_metric == metric
            ]
            if not source_means:
                continue
            result[condition][metric] = {
                "run_distribution": distribution_summary(
                    value
                    for (item_condition, _source, item_metric), values in grouped.items()
                    if item_condition == condition and item_metric == metric
                    for value in values
                ),
                "music_bootstrap_95_ci": bootstrap_mean_ci(
                    source_means, iterations=2_000, seed=0
                ),
            }
    return result


def consolidate(raw_root: Path, output_dir: Path) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ready = output_dir / "READY_FOR_THESIS"
    if ready.exists():
        atomic_replace(ready, output_dir / "READY_FOR_THESIS.previous")

    errors: list[str] = []
    offline = _load_json(raw_root / "offline" / "experiment_report.json", errors)
    if offline.get("failures"):
        errors.append("offline: report contains unresolved failures")
    providers = offline.get("manifest", {}).get("onnxruntime_providers", [])
    if providers != ["CPUExecutionProvider"]:
        errors.append(f"offline: expected only CPUExecutionProvider, got {providers!r}")

    suites: dict[str, Any] = {}
    for name, expected in EXPECTED_RUNS.items():
        status, report = validate_suite(name, raw_root / name, expected, errors)
        suites[name] = {"status": status, "report": report}
    preflight_validation = _load_json(
        raw_root / "preflight" / "preflight_validation.json", errors
    )
    if not preflight_validation.get("passed", False):
        errors.append("preflight: dependency/input validation did not pass")

    identities = {"offline": _identity(offline.get("manifest", {}))}
    identities.update(
        {name: _identity(payload["report"].get("manifest", {})) for name, payload in suites.items()}
    )
    canonical = identities["offline"]
    for name, identity in identities.items():
        if identity != canonical:
            errors.append(f"{name}: catalog/model/protocol/schema/commit hashes differ from offline")

    feature_path = raw_root / "features" / "aistpp_fact_features.npz"
    feature_metrics = validate_feature_bundle(feature_path, errors)
    feature_manifest = _load_json(feature_path.with_suffix(".manifest.json"), errors)
    if feature_manifest.get("catalog_sha256") != canonical.get("catalog_sha256"):
        errors.append("feature bundle catalog hash differs from experiment suites")

    validation = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "expected_runs": EXPECTED_RUNS,
        "total_expected_process_runs": sum(EXPECTED_RUNS.values()),
        "identities": identities,
        "errors": errors,
        "ready": not errors,
    }
    atomic_json(output_dir / "validation_report.json", validation)
    if errors:
        raise ValueError("Thesis result validation failed:\n- " + "\n- ".join(errors))

    consolidated = {
        "schema_version": 1,
        "generated_utc": validation["generated_utc"],
        "manifest_identity": canonical,
        "protocol": offline.get("protocol"),
        "offline": {
            "quality": offline.get("quality"),
            "audio_evaluation": offline.get("audio_evaluation"),
        },
        "literature_feature_metrics": feature_metrics,
        "feature_manifest": feature_manifest,
        "suites": {
            name: {
                "completion": {
                    "expected_runs": payload["status"]["expected_runs"],
                    "completed_runs": len(payload["status"]["runs"]),
                    "attempt_history": payload["status"].get("attempt_history", []),
                },
                "trace_aggregates": payload["report"].get("trace_aggregates"),
                "pose_aggregates": payload["report"].get("pose_aggregates"),
                "response_latency": payload["report"].get("response_latency"),
                "timing": payload["report"].get("timing"),
                "acceptance": payload["report"].get("acceptance"),
                "ablation": payload["report"].get("ablation"),
            }
            for name, payload in suites.items()
        },
        "statistical_summaries": {
            name: statistical_summary(payload["status"], payload["report"])
            for name, payload in suites.items()
        },
        "all_acceptance_passed": all(
            payload["report"].get("acceptance", {}).get("all_evaluated_passed", False)
            for name, payload in suites.items()
            if name != "preflight"
        ),
        "all_ablation_quality_checks_passed": all(
            check.get("passed", False)
            for check in suites.get("ablation", {})
            .get("report", {})
            .get("ablation", {})
            .get("quality_checks", {})
            .values()
        ),
        "interpretation_guardrails": [
            "A failed real-time or safety threshold remains a failed result in Chapter 6.",
            "FID/Div reuse real retrieved motions and are not a fully fair comparison to unconstrained generation.",
            "MuJoCo playback is not evidence of closed-loop balance, torque feasibility, or hardware success.",
        ],
    }
    consolidated["copied_artifacts"] = copy_report_artifacts(suites, output_dir)
    result_path = output_dir / "consolidated_results.json"
    atomic_json(result_path, consolidated)
    write_summary_csv(output_dir / "consolidated_summary.csv", suites, feature_metrics)
    marker = {
        "ready": True,
        "generated_utc": validation["generated_utc"],
        "consolidated_results_sha256": file_sha256(result_path),
        "feature_bundle_sha256": file_sha256(feature_path),
        "total_process_runs": sum(EXPECTED_RUNS.values()),
        "acceptance_all_passed": consolidated["all_acceptance_passed"],
        "ablation_quality_checks_all_passed": consolidated[
            "all_ablation_quality_checks_passed"
        ],
    }
    temporary = output_dir / "READY_FOR_THESIS.tmp"
    temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    atomic_replace(temporary, ready)
    return marker


def main() -> int:
    args = parse_args()
    try:
        marker = consolidate(args.raw_root, args.output_dir)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(marker, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
