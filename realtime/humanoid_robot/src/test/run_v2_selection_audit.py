"""Prepare/run/analyse a frozen 20-song concentration audit of humanoid matcher v2.

Preparation does not run inference or the runtime experiment. No production files
are modified. See matcher_v2_selection_audit.md for the protocol and limitations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
ROOT = HERE.parents[3]
DATA = ROOT / "realtime/humanoid_robot/data"
DEFAULT_OUTPUT = HERE / "output/v2_selection_audit_20"
sys.path.insert(0, str(SRC))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def prepare(output):
    import importlib.metadata
    import numpy as np
    import soundfile as sf
    from music_motion_catalog import load_audio_mono

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        verify(output, read_json(manifest_path))
        print(f"Existing frozen preparation verified: {manifest_path}")
        return
    catalog_path = DATA / "music_catalog/catalog.json"
    catalog = read_json(catalog_path)
    aist_root = (catalog_path.parent / catalog["aistpp_root"]).resolve()
    cc0 = DATA / "test_audio/cc0_matcher_set"
    songs = []
    for track in read_json(cc0 / "manifest.json")["tracks"]:
        songs.append({"title": track["title"], "group": "cc0_ambient" if "ambient" in track["file"] else "cc0_music",
                      "style": track["style"], "music_id": None, "source": relative(cc0 / track["file"])})
    # Select the longest available original per genre without looking at results.
    for genre in sorted({t["genre"] for t in catalog["tracks"].values()}):
        candidates = []
        for music_id, track in sorted(catalog["tracks"].items()):
            if track["genre"] == genre:
                for variant in track["variants"]:
                    path = aist_root / variant["source_audio"]
                    candidates.append((sf.info(path).duration, music_id, path))
        _, music_id, path = max(candidates, key=lambda item: (item[0], item[1], str(item[2])))
        songs.append({"title": f"AIST++ {music_id}", "group": "aist_catalog", "style": genre,
                      "music_id": music_id, "source": relative(path)})
    if len(songs) != 20 or len({s["source"] for s in songs}) != 20:
        raise ValueError("Expected exactly 20 distinct source tracks")
    rate, seconds = 16000, 60
    (output / "audio").mkdir(parents=True, exist_ok=True)
    fingerprints = {}
    for index, song in enumerate(songs, 1):
        song["id"] = f"song_{index:02d}"
        source = ROOT / song["source"]
        audio = load_audio_mono(source, rate)
        if len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError(f"Empty/non-finite audio: {source}")
        song["source_seconds"] = len(audio) / rate
        song["looped"] = len(audio) < seconds * rate
        song["loop_boundaries_seconds"] = [i * len(audio) / rate for i in range(1, math.ceil(seconds * rate / len(audio)))]
        samples = np.tile(audio, math.ceil(seconds * rate / len(audio)))[:seconds * rate]
        wav = output / "audio" / f"{song['id']}.wav"
        sf.write(wav, samples, rate, subtype="FLOAT")
        song["prepared_audio"] = str(wav.resolve())
        song["prepared_sha256"] = digest(wav)
        fingerprints[song["source"]] = digest(source)
    if len(set(fingerprints[s["source"]] for s in songs)) != 20:
        raise ValueError("Duplicate source file content in 20-song sample")
    frozen = [catalog_path, catalog_path.parent / catalog["arrays_file"], Path(__file__),
              HERE / "matcher_v2_audited_entrypoint.py"]
    frozen += list(SRC.glob("*.py"))
    frozen += list((ROOT / "realtime/shared").glob("*.py"))
    for kind in ("embedding_model", "tag_model"):
        model = (catalog_path.parent / catalog["extractor"][kind]["path"]).resolve()
        frozen.append(model)
        frozen.extend(model.parent.glob(model.stem + "*.json"))
        if digest(model) != catalog["extractor"][kind]["sha256"]:
            raise ValueError(f"Model differs from catalog identity: {model}")
    gmr_root = DATA / "aistpp_gmr"
    for motion in catalog["motions"].values():
        if motion["preflight_passed"]:
            frozen.append(gmr_root / (motion["motion_id"] + ".pkl"))
    for path in sorted(set(frozen)):
        fingerprints[relative(path)] = digest(path)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "librosa", "soundfile", "onnxruntime", "mujoco")}
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "packages": versions, "catalog": relative(catalog_path),
                "gmr_root": relative(gmr_root), "duration_seconds": seconds, "sample_rate": rate,
                "seed": 20260917, "history_seconds": 30, "minimum_history_seconds": 2,
                "diagnostic_hop_seconds": 2, "songs": songs, "sha256": fingerprints,
                "interpretation": "Descriptive concentration, not a test against uniform/random motion selection."}
    write_json(manifest_path, manifest)
    write_csv(output / "songs.csv", [{k: s[k] for k in ("id", "title", "group", "style", "music_id", "source", "source_seconds", "looped")} for s in songs])
    write_csv(output / "catalog_inventory.csv", [
        {"motion_id": m["motion_id"], "music_id": m["music_id"], "genre": m["genre"],
         "preflight_passed": m["preflight_passed"]} for m in catalog["motions"].values()])
    print(f"Prepared 20 songs; no experiment run. Manifest: {manifest_path}")


def verify(output, manifest):
    import importlib.metadata
    for name, expected in manifest["sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}. Use a new --output-dir and prepare again.")
    for song in manifest["songs"]:
        if digest(song["prepared_audio"]) != song["prepared_sha256"]:
            raise ValueError(f"Prepared audio changed: {song['id']}")
    for package, version in manifest["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Package changed: {package}. Prepare a new experiment directory.")


def without_music(catalog, music_id):
    """Remove every variant/segment and associated motion; retain training normalization."""
    import numpy as np
    from music_motion_catalog import MusicCatalog
    metadata = dict(catalog.metadata)
    mask = np.array([s["music_id"] != music_id for s in catalog.segment_metadata])
    metadata["segments"] = [s for s in catalog.segment_metadata if s["music_id"] != music_id]
    metadata["tracks"] = {k: v for k, v in catalog.tracks.items() if k != music_id}
    metadata["motions"] = {k: v for k, v in metadata["motions"].items() if v["music_id"] != music_id}
    arrays = {k: getattr(catalog, k)[mask] for k in ("embeddings", "rhythm_timbre", "tags")}
    arrays.update({k: getattr(catalog, k) for k in ("embedding_mean", "embedding_std", "rhythm_mean", "rhythm_std")})
    return MusicCatalog(catalog.catalog_path, metadata, arrays)


def embedding_tracks(matcher, descriptor):
    import numpy as np
    catalog = matcher.catalog
    query = matcher._standardized_unit(descriptor.embedding, catalog.embedding_mean, catalog.embedding_std)
    similarities = matcher._similarities(matcher._catalog_embeddings, query)
    scores = defaultdict(list)
    for segment, value in zip(catalog.segment_metadata, similarities):
        scores[segment["music_id"]].append(float(value))
    ranked = [{"music_id": key, "score": float(np.mean(sorted(values, reverse=True)[:3]))}
              for key, values in scores.items()]
    return sorted(ranked, key=lambda item: (-item["score"], item["music_id"]))[:5]


def diagnostic(manifest, song, destination, limit=None):
    from music_motion_catalog import MusicCatalog, MusicMotionMatcher, load_audio_mono
    from realtime_music_humanoid_matcher_v2 import make_extractor
    catalog = MusicCatalog.load(ROOT / manifest["catalog"])
    extractor = make_extractor(argparse.Namespace(embedding_model=None, tag_model=None), catalog)
    matchers = {"production": MusicMotionMatcher(catalog, speed_min=0.55, speed_max=1.3)}
    if song["music_id"]:
        matchers["leave_music_out"] = MusicMotionMatcher(without_music(catalog, song["music_id"]), speed_min=0.55, speed_max=1.3)
    audio = load_audio_mono(Path(song["prepared_audio"]), extractor.sample_rate)
    rows = []
    stop = int(limit or manifest["duration_seconds"])
    for end in range(2, stop + 1, manifest["diagnostic_hop_seconds"]):
        start = max(0, end - manifest["history_seconds"])
        descriptor = extractor.describe(audio[start * extractor.sample_rate:end * extractor.sample_rate])
        for condition, matcher in matchers.items():
            result = matcher.match(descriptor, top_k_tracks=5, top_k_motions=20)
            rows.append({"end_seconds": end, "start_seconds": start, "condition": condition,
                         "embedding_tracks": embedding_tracks(matcher, descriptor), "result": asdict(result)})
    write_json(destination, rows)


def runtime_command(manifest, song, run_dir, seconds=None):
    return [sys.executable, "-u", str(HERE / "matcher_v2_audited_entrypoint.py"),
            "--catalog", str(ROOT / manifest["catalog"]),
            "--gmr-motion-root", str(ROOT / manifest["gmr_root"]), "--retarget-policy", "require-gmr",
            "--audio-input", song["prepared_audio"], "--headless", "--realtime",
            "--audio-input-delay-sec", "1", "--max-seconds", str(seconds or manifest["duration_seconds"]),
            "--initial-motion-seed", str(manifest["seed"]), "--control-rate-hz", "60",
            "--speed-min", "0.55", "--speed-max", "1.3", "--history-max-seconds", "30",
            "--analysis-min-seconds", "2", "--match-interval-seconds", "1",
            "--match-top-tracks", "5", "--match-top-motions", "20", "--match-policy", "style-first",
            "--trace-csv", str(run_dir / "trace.csv"), "--timing-report", str(run_dir / "timing.json")]


def complete(run_dir):
    marker = run_dir / "complete.json"
    if not marker.exists():
        return False
    try:
        return all(digest(run_dir / name) == expected for name, expected in read_json(marker)["sha256"].items())
    except (OSError, ValueError, KeyError):
        return False


def run(manifest, output, smoke=False):
    verify(output, manifest)
    songs = manifest["songs"][:1] if smoke else manifest["songs"]
    base = output / ("smoke" if smoke else "runs")
    for index, song in enumerate(songs, 1):
        run_dir = base / song["id"]
        if complete(run_dir):
            print(f"[{index}/{len(songs)}] Verified, skipping {song['title']}", flush=True)
            continue
        if run_dir.exists():
            archive = base / "failed_attempts"
            archive.mkdir(exist_ok=True)
            run_dir.rename(archive / f"{song['id']}_{time.time_ns()}")
        run_dir.mkdir(parents=True)
        print(f"[{index}/{len(songs)}] {song['title']}: deterministic ONNX diagnostic", flush=True)
        diagnostic(manifest, song, run_dir / "diagnostic.json", limit=8 if smoke else None)
        command = runtime_command(manifest, song, run_dir, seconds=8 if smoke else None)
        write_json(run_dir / "command.json", command)
        env = os.environ.copy()
        env.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        env.update(PYTHONIOENCODING="utf-8", MATCHER_AUDIT_JSONL=str(run_dir / "retrieval.jsonl"))
        print(f"[{index}/{len(songs)}] {song['title']}: v2 runtime (log: {run_dir / 'runtime.log'})", flush=True)
        with (run_dir / "runtime.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        with (run_dir / "trace.csv").open(encoding="utf-8-sig", newline="") as handle:
            trace = list(csv.DictReader(handle))
        expected = 8 if smoke else manifest["duration_seconds"]
        if not trace or float(trace[-1]["audio_time_seconds"]) < expected - 0.25:
            raise ValueError(f"Incomplete audio coverage: {run_dir}")
        retrievals = [json.loads(line) for line in (run_dir / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()]
        if not retrievals or any(row["index"] != i for i, row in enumerate(retrievals, 1)):
            raise ValueError(f"Missing or invalid retrieval log: {run_dir}")
        read_json(run_dir / "timing.json")
        if "realtime retrieval failed" in (run_dir / "runtime.log").read_text(encoding="utf-8"):
            raise ValueError(f"Runtime retrieval failure: {run_dir}")
        names = ("diagnostic.json", "command.json", "runtime.log", "trace.csv", "timing.json", "retrieval.jsonl")
        write_json(run_dir / "complete.json", {"sha256": {name: digest(run_dir / name) for name in names}})
    if not smoke:
        analyse(manifest, output)


def trace_counts(rows):
    plays, dwell = Counter(), Counter()
    switched = False
    replay_count = 0
    previous_motion = rows[0]["current_motion_id"] if rows else ""
    for index, row in enumerate(rows):
        motion = row["current_motion_id"]
        if row["event"] == "switch_complete":
            switched = True
            plays[motion] += 1
            replay_count += int(motion == previous_motion)
            previous_motion = motion
        if switched and index + 1 < len(rows):
            dt = max(0.0, float(rows[index + 1]["audio_time_seconds"]) - float(row["audio_time_seconds"]))
            dwell[motion] += dt
    return plays, dwell, replay_count


def concentration(per_song, all_song_count):
    """Equal song weighting; no pretending correlated windows are independent songs."""
    totals = Counter()
    presence = Counter()
    informative = 0
    for counts in per_song:
        total = sum(counts.values())
        if not total:
            continue
        informative += 1
        for key, value in counts.items():
            totals[key] += value / total
            presence[key] += 1
    shares = {key: value / informative for key, value in totals.items()} if informative else {}
    ordered = sorted(shares, key=lambda key: (-shares[key], key))
    return {"songs_total": all_song_count, "songs_with_observations": informative,
            "unique_ids": len(shares), "top1_share": shares[ordered[0]] if ordered else None,
            "top5_share": sum(shares[key] for key in ordered[:5]) if ordered else None,
            "effective_ids": 1 / sum(s * s for s in shares.values()) if shares else None,
            "rankings": [{"id": key, "song_balanced_share": shares[key], "songs_present": presence[key],
                          "review_flag": shares[key] >= 0.20 and presence[key] >= min(5, all_song_count)} for key in ordered]}


def analyse(manifest, output):
    records = []
    runtime_records = []
    distributions = defaultdict(dict)
    complete_songs = []
    for song in manifest["songs"]:
        directory = output / "runs" / song["id"]
        if not complete(directory):
            continue
        complete_songs.append(song)
        diag = read_json(directory / "diagnostic.json")
        live = [json.loads(line) for line in (directory / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()]
        conditions = {"live": live, "diagnostic": [r for r in diag if r["condition"] == "production"],
                      "diagnostic_first_pass": [r for r in diag if r["condition"] == "production" and r["end_seconds"] <= song["source_seconds"]]}
        if song["music_id"]:
            conditions["leave_music_out"] = [r for r in diag if r["condition"] == "leave_music_out"]
            conditions["leave_music_out_first_pass"] = [r for r in diag if r["condition"] == "leave_music_out" and r["end_seconds"] <= song["source_seconds"]]
        for condition, rows in conditions.items():
            counters = defaultdict(Counter)
            accepted = 0
            for row in rows:
                result = row["result"]
                if row.get("embedding_tracks"):
                    counters["embedding_track_all"][row["embedding_tracks"][0]["music_id"]] += 1
                if result["tracks"]:
                    counters["combined_track_all"][result["tracks"][0]["music_id"]] += 1
                if result["accepted"] and result["motions"]:
                    accepted += 1
                    counters["motion_top1_accepted"][result["motions"][0]["motion_id"]] += 1
                    counters["genre_accepted"][result["genres"][0]["genre"]] += 1
                    for motion in result["motions"][:5]:
                        counters["motion_top5_slots"][motion["motion_id"]] += 1
                else:
                    counters["rejections"][result["rejection_reason"] or "no_compatible_motion"] += 1
            for metric, counts in counters.items():
                distributions[f"{condition}.{metric}"][song["id"]] = counts
            top = counters["motion_top1_accepted"].most_common(1)
            records.append({"song": song["id"], "title": song["title"], "group": song["group"],
                            "condition": condition, "observations": len(rows), "accepted_with_motion": accepted,
                            "rejected_or_no_motion": len(rows) - accepted,
                            "top_motion": top[0][0] if top else "", "top_motion_share": top[0][1] / accepted if top else None})
        with (directory / "trace.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        plays, dwell, replays = trace_counts(rows)
        distributions["runtime.completed_plays_excluding_startup"][song["id"]] = plays
        distributions["runtime.dwell_after_first_switch"][song["id"]] = dwell
        runtime_records.append({"song": song["id"], "title": song["title"], "group": song["group"],
                                "completed_plays_excluding_startup": sum(plays.values()),
                                "motion_changes": sum(plays.values()) - replays, "same_motion_replays": replays,
                                "observed_seconds_after_first_switch": sum(dwell.values()),
                                "top_motion": plays.most_common(1)[0][0] if plays else "",
                                "top_motion_share": plays.most_common(1)[0][1] / sum(plays.values()) if plays else None})
    summaries = {}
    for group in ("all", "cc0_music", "cc0_ambient", "aist_catalog"):
        subset = [s for s in complete_songs if group == "all" or s["group"] == group]
        summaries[group] = {}
        for metric, counts in distributions.items():
            eligible = [s for s in subset if not metric.startswith("leave_music_out") or s["music_id"]]
            summaries[group][metric] = concentration([counts.get(s["id"], {}) for s in eligible], len(eligible))
    write_json(output / "report.json", {"complete": len(complete_songs) == 20, "completed_songs": len(complete_songs),
                                       "groups": summaries, "per_song": records, "runtime": runtime_records})
    write_csv(output / "per_song.csv", records)
    write_csv(output / "runtime_per_song.csv", runtime_records)
    write_csv(output / "selection_frequencies.csv", [
        {"group": group, "metric": metric, **item} for group, metrics in summaries.items()
        for metric, summary in metrics.items() for item in summary["rankings"]])
    lines = ["# Matcher v2: 20-song selection audit", "", f"Completed songs: {len(complete_songs)}/20.", "",
             "Shares give equal weight to each song with observations; missing/rejected songs are reported separately.",
             "Startup motion is excluded from play counts. Dwell includes holds and attributes blends to the current source motion.",
             "The embedding-only ranking is a diagnostic track ranking, not the full motion policy. AIST catalog songs are in-sample; use leave-music-out and CC0 separately.", "",
             "| Group | Metric | Songs with observations | Unique IDs | Top share | Leading ID |", "|---|---|---:|---:|---:|---|"]
    for group, metrics in summaries.items():
        for metric, summary in metrics.items():
            if "rejections" in metric or "top5_slots" in metric:
                continue
            leading = summary["rankings"][:1]
            share = f"{summary['top1_share']:.1%}" if leading else "n/a"
            identifier = leading[0]["id"] if leading else "none"
            lines.append(f"| {group} | {metric} | {summary['songs_with_observations']}/{summary['songs_total']} | {summary['unique_ids']} | {share} | {identifier} |")
    lines += ["", "A review flag means >=20% song-balanced share and presence in >=5 songs (or every song in a smaller group). It is a descriptive flag, not statistical significance or proof of a model defect.",
              "Inspect selection_frequencies.csv and per_song.csv, then compare diagnostic embedding tracks, combined tracks, motion recommendations and completed plays. Nonuniform catalog sizes, style/tempo compatibility, loop boundaries and transition feasibility can all affect concentration.",
              "Runtime uses one fixed startup seed and the normal file-input path; it is not a live-microphone or hardware trial. A held startup with no switches is not evidence that ONNX selected that motion."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {output / 'report.md'} ({len(complete_songs)}/20 songs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "analyse", "smoke", "dry-run"), default="prepare")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.stage == "prepare":
        prepare(output)
        return 0
    manifest = read_json(output / "manifest.json")
    if args.stage == "analyse":
        analyse(manifest, output)
    elif args.stage == "dry-run":
        verify(output, manifest)
        write_json(output / "commands.json", [runtime_command(manifest, s, output / "runs" / s["id"]) for s in manifest["songs"]])
        print("Verified frozen inputs; wrote commands.json. No inference or runtime experiment executed.")
    else:
        run(manifest, output, smoke=args.stage == "smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
