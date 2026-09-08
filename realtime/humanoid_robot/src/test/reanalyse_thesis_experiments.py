"""Offline-only revision; never changes thesis_final or its original completion marker."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path

import numpy as np

import run_humanoid_matcher_experiments as runner
import humanoid_matcher_experiment_metrics as metrics
import build_aistpp_fact_feature_bundle as features
import consolidate_thesis_experiments as original
from run_thesis_supplement import (RAW_ROOT, atomic_json, cohort_lock, read_json, validate_cohort)

OUTPUT = runner.ROOT / "docs" / "thesis" / "experiment_results" / "reanalysis_v2"
OLD_ROOT = runner.TEST_DIR / "output" / "thesis_final"


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def source_statistics(records):
    """Run summaries -> seed means per source -> source bootstrap and paired tests."""
    grouped = defaultdict(list)
    for record in records:
        for metric, value in record["metrics"].items():
            if finite(value):
                grouped[(record["condition"], record["source_id"], metric)].append(float(value))
    source_means = {key: float(np.mean(values)) for key, values in grouped.items()}
    summaries, tests = {}, {}
    for condition, _, metric in sorted(source_means):
        key = f"{condition}:{metric}"
        if key in summaries:
            continue
        values = [v for (c, _, m), v in source_means.items() if (c, m) == (condition, metric)]
        summaries[key] = {"source_distribution": metrics.distribution_summary(values),
                          "bootstrap_95_ci": metrics.bootstrap_mean_ci(values),
                          "source_seed_counts": {s: len(grouped[(c, s, m)]) for c, s, m in grouped
                                                 if (c, m) == (condition, metric)}}
        if condition == "full":
            continue
        # Match seeds before averaging, excluding undefined values in both arms.
        pairs_by_source = defaultdict(list)
        full = {(r["source_id"], r["seed"]): r for r in records if r["condition"] == "full"}
        for record in records:
            partner = full.get((record["source_id"], record["seed"]))
            if record["condition"] != condition or partner is None:
                continue
            a, b = record["metrics"].get(metric), partner["metrics"].get(metric)
            if finite(a) and finite(b):
                pairs_by_source[record["source_id"]].append((a, b))
        means = [np.mean(values, axis=0) for _, values in sorted(pairs_by_source.items())]
        if means:
            test = metrics.paired_permutation_test([v[0] for v in means], [v[1] for v in means])
            test["difference_bootstrap_95_ci"] = metrics.bootstrap_mean_ci([v[0] - v[1] for v in means])
            test["sources"] = sorted(pairs_by_source)
            tests[key] = test
    corrected = metrics.holm_adjust({key: value["p_value"] for key, value in tests.items()})
    for key, test in tests.items():
        test["holm_adjusted_p_value"] = corrected[key]
        test["significant"] = corrected[key] < .05
    return {"unit": "source; matched seeds averaged first", "difference": "ablation_minus_full",
            "holm_family": "all reported condition-by-metric tests within this suite",
            "summaries": summaries, "paired_tests": tests}


def per_seed_features(path: Path):
    reports = {}
    with np.load(path, allow_pickle=False) as data:
        real_ids = [str(x).split(":")[0] for x in data["real_item_ids"]]
        output_ids = [str(x) for x in data["output_item_ids"]]
        if len(set(real_ids)) != 40:
            raise ValueError("Expected 40 fixed real items")
        expected_ids = {f"{music}:seed={seed}" for music in real_ids for seed in range(5)}
        if len(output_ids) != 200 or set(output_ids) != expected_ids:
            raise ValueError("Missing, duplicate or unexpected output feature IDs")
        for seed in range(5):
            indices = [output_ids.index(f"{music}:seed={seed}") for music in real_ids]
            # Both raw feature scale and real-set normalisation are explicit.
            # No pooled 200-output vs 40-real estimate is used.
            reports[str(seed)] = {}
            for scaling in ("raw", "real_standardised"):
                arrays = []
                for kind in ("kinetic", "geometric"):
                    real = np.asarray(data[f"real_{kind}"], dtype=np.float64)
                    output = np.asarray(data[f"output_{kind}"][indices], dtype=np.float64)
                    if real.shape != (40, 72 if kind == "kinetic" else 32) or output.shape != real.shape:
                        raise ValueError("Invalid per-seed feature dimensions")
                    if scaling == "real_standardised":
                        mean, scale = real.mean(axis=0), np.maximum(real.std(axis=0), 1e-8)
                        real, output = (real - mean) / scale, (output - mean) / scale
                    arrays.extend((real, output))
                reports[str(seed)][scaling] = metrics.literature_feature_metrics(
                    *arrays, extractor_identity=str(data["extractor_identity"]))
    seed_summary = {scaling: {metric: metrics.distribution_summary(
        reports[str(seed)][scaling][metric] for seed in range(5))
        for metric, value in reports['0'][scaling].items() if finite(value)}
        for scaling in ('raw', 'real_standardised')}
    return {"per_seed_40_items": reports, "seed_summary": seed_summary, "diversity_pairs_per_seed": 780,
            "scaling_note": "raw and real-set population-std normalised; no unqualified cross-paper comparison",
            "limitations": "cycled short ground truth, low-rate trace reconstruction, translation-only alignment, real-motion retrieval reuse"}


def response(run, causal):
    source = run["source_id"]
    duration = float(run.get("duration", run.get("required_duration_seconds", 40)))
    genres = None
    event_metadata = []
    if source.startswith("aistpp_stitched"):
        times, genres = runner.repeated_stitched_changes(duration)
    else:
        manifest = read_json(runner.HUMANOID_DIR / "data" / "test_audio" / "matcher_change_streams" / "manifest.json")
        stream = next((s for s in manifest["streams"] if Path(s["path"]).stem == source), None)
        event_metadata = stream["changes"] if stream else []
        times = [float(item["time_seconds"] if isinstance(item, dict) else item) for item in event_metadata]
    result = metrics.response_latency_from_trace(
        run["trace"], times, genres, clock_field="wall_time_seconds" if causal else "audio_time_seconds")
    result["event_metadata"] = event_metadata
    result["time_basis"] = "actual_wall; audio-origin aligned" if causal else "audio-block proxy; actual wall response unrecoverable"
    result["interpretation"] = ("censored at next change; genre-targeted only for stitched. Tempo/silence milestones are scheduling diagnostics, not proof of correct tempo or rejection.")
    return result


def analyse_run(run, causal, clusters):
    pose = metrics.aggregate_pose_npz(run["pose"], start_seconds=6, cluster_by_motion=clusters)
    trace = metrics.aggregate_trace_csv(run["trace"], clusters, start_seconds=6)
    timing = read_json(Path(run["timing"]))
    safety = {"final_joint_limit_violations": None, "final_residual_clearance_violations": None,
              "collision_corrections": timing.get("collision_corrections", timing.get("self_collision_violations")),
              "pre_output_joint_limit_count": timing.get("joint_limit_violations"),
              "actual_final_output_derivatives_recoverable": causal}
    if causal:
        with np.load(run["pose"], allow_pickle=False) as data:
            keep = data["wall_time_seconds"] >= 6
            counts = data["safety_counts"][keep].sum(axis=0)
        safety.update(dict(zip(("candidate_clearance_detections", "collision_corrections",
                                "final_residual_clearance_violations", "final_joint_limit_violations"), map(int, counts))))
        speed = pose["continuity"]["joint_speed_rad_s"]["max"]
        acceleration = pose["continuity"]["joint_acceleration_rad_s2"]["max"]
        safety.update(final_speed_max_rad_s=speed, final_acceleration_max_rad_s2=acceleration,
                      speed_passed=speed <= 16, acceleration_passed=acceleration <= 2000,
                      residual_clearance_passed=counts[2] == 0, final_limits_passed=counts[3] == 0)
    values = {
        "bas_harmonic": pose["beat"]["bas_harmonic"],
        "bas_music_to_dance": pose["beat"]["bas_music_to_dance"],
        "normalized_selection_entropy": trace["selection"]["normalized_selection_entropy"],
        "unique_motions": trace["selection"]["unique_motions"],
        "mean_match_score": trace["match_score"]["mean"],
        "joint_jerk_p95_rad_s3": pose["continuity"]["joint_jerk_rad_s3"]["p95"],
        "deadline_miss_ratio": timing.get("deadline_miss_ratio"),
        "query_p95_ms": timing.get("retrieval_waveform_to_match_ms_p95"),
        "control_work_p99_ms": timing.get("work_ms_p99"),
        "pfc_edge_g1_adapted_30fps": pose["physical"]["pfc_edge_g1_adapted_30fps"],
    }
    expected = set(read_json(runner.DEFAULT_PROTOCOL)["cc0_labels"].get(run["source_id"], {}).get("expected_genres", []))
    with Path(run["trace"]).open(encoding="utf-8", newline="") as handle:
        genres = [row.get("top_genre") for row in csv.DictReader(handle)
                  if row.get("event") == "match" and float(row["audio_time_seconds"]) >= 6]
    values["genre_recall_at_1"] = sum(g in expected for g in genres) / len(genres) if expected and genres else None
    adjusted_timing = dict(timing)
    # Reuse timing acceptance checks, replacing misleading safety aliases.
    adjusted_timing.update(
        limiter_max_output_speed_rad_s=safety.get("final_speed_max_rad_s"),
        limiter_max_output_acceleration_rad_s2=safety.get("final_acceleration_max_rad_s2"),
        joint_limit_violations=safety["final_joint_limit_violations"],
        self_collision_violations=safety["final_residual_clearance_violations"])
    acceptance_protocol = read_json(runner.DEFAULT_PROTOCOL)
    rate_hz = float(timing.get("control_rate_hz", acceptance_protocol["control_rate_hz"]))
    acceptance_protocol["realtime_constraints"]["control_work_p99_ms"] = 1000.0 / rate_hz
    return {"condition": run["condition"], "source_id": run["source_id"], "seed": run["seed"],
            "run_key": run["run_key"], "metrics": values, "pose": pose, "trace": trace,
            "stage_timing": timing, "safety": safety, "response": response(run, causal),
            "acceptance": runner.acceptance_report({"runs": [adjusted_timing]}, acceptance_protocol)}


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"Offline reanalysis: {args.old_root} + {args.supplement_root} -> {args.output_dir}; no experiments launched")
        return 0
    if args.output_dir.resolve() in {args.old_root.resolve(), (runner.ROOT / "docs/thesis/experiment_results").resolve()}:
        raise ValueError("Refusing to overwrite original outputs")
    with cohort_lock(args.supplement_root):
        frozen, new_runs = validate_cohort(args.supplement_root)
        is_pilot = bool(frozen.get("payload", {}).get("pilot", False))
        if is_pilot:
            catalog = runner.MusicCatalog.load(runner.DEFAULT_CATALOG)
            clusters = {
                key: motion.motion_cluster_id
                for key, motion in catalog.motions.items()
                if motion.preflight_passed
            }
            output = {
                "schema_version": 2,
                "pilot": True,
                "supplement_identity": frozen["identity"],
                "control_rate_hz": frozen["payload"]["control_rate_hz"],
                "cohorts": {"online_causal": {}},
            }
            all_results = []
            for suite, runs in new_runs.items():
                if suite == "smoke":
                    continue
                print(f"Offline analysing pilot/{suite}: {len(runs)} runs", flush=True)
                results = [analyse_run(run, True, clusters) for run in runs]
                all_results.extend(results)
                output["cohorts"]["online_causal"][suite] = {
                    "runs": results,
                    "statistics": source_statistics(results),
                }
            passed = sum(
                bool(result["acceptance"]["all_evaluated_passed"])
                for result in all_results
            )
            output["pilot_gate"] = {
                "evaluated_runs": len(all_results),
                "all_constraints_passed_runs": passed,
                "all_runs_passed": passed == len(all_results),
                "note": (
                    "A failed pilot is retained evidence. Diagnose it before a "
                    "formal cohort; do not repeat runs selectively."
                ),
            }
            atomic_json(
                args.output_dir / "consolidated_results.json",
                clean_json(output),
            )
            marker = args.output_dir / "PILOT_ANALYSIS_COMPLETE"
            atomic_json(
                marker,
                {
                    "schema_version": 2,
                    "supplement_identity": frozen["identity"],
                    "analysed_runs": len(all_results),
                    "all_runs_passed": passed == len(all_results),
                    "ready_means": "integrity and pilot analysis complete",
                },
            )
            print(f"Pilot reanalysis complete: {marker}")
            return 0
        errors = []
        old_suites = {}
        identities = []
        for suite, count in original.EXPECTED_RUNS.items():
            status, report = original.validate_suite(suite, args.old_root / suite, count, errors)
            old_suites[suite] = status["runs"]
            identities.append(original._identity(report["manifest"]))
        if errors or any(identity != identities[0] for identity in identities):
            raise ValueError(f"Original cohort invalid or mixed-version: {errors}")
        catalog = runner.MusicCatalog.load(runner.DEFAULT_CATALOG)
        if runner.file_sha256(catalog.catalog_path) != identities[0]["catalog_sha256"]:
            raise ValueError("Original feature reanalysis requires original catalog")
        if runner.file_sha256(catalog.catalog_path.parent / catalog.metadata['arrays_file']) != identities[0]['catalog_arrays_sha256']:
            raise ValueError("Original catalog descriptor arrays changed")
        old_feature_manifest = read_json(args.old_root / 'features/aistpp_fact_features.manifest.json')
        if runner.file_sha256(features.DEFAULT_SMPL_MODEL) != old_feature_manifest['smpl_model_sha256']:
            raise ValueError("Original SMPL feature model changed")
        clusters = {key: motion.motion_cluster_id for key, motion in catalog.motions.items() if motion.preflight_passed}
        # A new revision directory is authoritative only after this marker is recreated.
        marker = args.output_dir / "READY_FOR_THESIS"
        if marker.exists():
            runner.atomic_replace(marker, marker.with_name("READY_FOR_THESIS.previous"))
        output = {"schema_version": 2, "original_identity": identities[0],
                  "supplement_identity": frozen["identity"], "cohorts": {},
                  "limitations": ["Old actual frame and response clocks are unrecoverable.",
                                  "Old PFC is a proxy; G1 PFC is explicitly adapted, not SMPL PFC.",
                                  "Negative inputs are synthetic noise/drone/speech-like stress tests, not field recordings.",
                                  "Kinematic replay is not closed-loop balance or hardware success."]}
        for cohort, suites in (("preanalysed_replay", old_suites), ("online_causal", new_runs)):
            output["cohorts"][cohort] = {}
            for suite, runs in suites.items():
                if suite in {"preflight", "smoke"}:
                    continue
                print(f"Offline analysing {cohort}/{suite}: {len(runs)} runs", flush=True)
                results = [analyse_run(run, cohort == "online_causal", clusters) for run in runs]
                output["cohorts"][cohort][suite] = {"runs": results, "statistics": source_statistics(results)}
        feature_path = args.output_dir / "features" / "aistpp_fact_features_v2.npz"
        features.build_bundle(catalog.catalog_path, args.old_root / "literature/run_status.json",
                              feature_path, features.DEFAULT_SMPL_MODEL, warmup_seconds=6, clip_seconds=20, fps=60)
        output["feature_metrics"] = per_seed_features(feature_path)
        output["feature_manifest"] = read_json(feature_path.with_suffix(".manifest.json"))
        output["offline_original"] = read_json(args.old_root / "offline/experiment_report.json")
        output["postprocessor_sha256"] = {str(path): runner.file_sha256(path) for path in
                                          (Path(__file__), Path(metrics.__file__), Path(features.__file__), Path(runner.__file__))}
        atomic_json(args.output_dir / "consolidated_results.json", clean_json(output))
        rows = [{"cohort": cohort, "suite": suite, "condition": run["condition"],
                 "source_id": run["source_id"], "seed": run["seed"], **run["metrics"]}
                for cohort, suites in output["cohorts"].items() for suite, data in suites.items() for run in data["runs"]]
        csv_path = args.output_dir / "consolidated_summary.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        runner.atomic_replace(temporary, csv_path)
        atomic_json(marker, {"schema_version": 2, "supplement_identity": frozen["identity"],
                             "formal_supplement_runs": sum(
                                 len(runs) for suite, runs in new_runs.items()
                                 if suite != "smoke"
                             ),
                             "results_sha256": runner.file_sha256(args.output_dir / "consolidated_results.json"),
                             "ready_means": "integrity and analysis complete, NOT all constraints passed"})
        print(f"Reanalysis ready: {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
