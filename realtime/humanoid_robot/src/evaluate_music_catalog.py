"""Evaluate library retrieval, recording-held-out genres and runtime latency."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

# Evaluation should not compete with the realtime matcher for all CPU cores.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from music_motion_catalog import AudioDescriptor, AudioFeatureExtractor, MusicCatalog, MusicMotionMatcher, iter_audio_windows, load_audio_mono

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = ROOT / "realtime/humanoid_robot/data/music_catalog/catalog.json"


def descriptor_from_segment(catalog, index):
    item = catalog.segment_metadata[index]
    return AudioDescriptor(
        embedding=catalog.embeddings[index], rhythm_timbre=catalog.rhythm_timbre[index],
        tag_probabilities=catalog.tags[index],
        **{key: float(item.get(key, 0.0)) for key in (
            "bpm", "beat_strength", "onset_density", "offbeat_ratio", "tempo_stability",
            "spectral_flux", "percussive_ratio", "signal_rms", "dynamic_range_db")},
    )


def query_groups(catalog, queries_per_recording=1):
    groups = defaultdict(list)
    for index, segment in enumerate(catalog.segment_metadata):
        groups[catalog.recording_key(segment["music_id"])].append(index)
    if queries_per_recording:
        for key, indices in groups.items():
            picks = np.linspace(0, len(indices)-1, min(queries_per_recording, len(indices)), dtype=int)
            groups[key] = [indices[i] for i in picks]
    return groups


def reference_without_recording(catalog, recording):
    """Exclude all sibling clips, and fit feature normalization on references only."""
    tracks = {key: value for key, value in catalog.tracks.items() if catalog.recording_key(key) != recording}
    indices = [i for i, segment in enumerate(catalog.segment_metadata) if segment["music_id"] in tracks]
    if not indices:
        return None
    motions = {key: profile.to_dict() for key, profile in catalog.motions.items() if profile.music_id in tracks}
    metadata = dict(catalog.metadata, tracks=tracks, motions=motions,
                    segments=[catalog.segment_metadata[i] for i in indices])
    velocity = np.asarray([profile["velocity_p90"] for profile in motions.values()])
    if velocity.size:
        metadata["motion_stats"] = dict(velocity_median=float(np.median(velocity)),
                                       velocity_scale=float(max(np.ptp(np.quantile(velocity, (.25, .75))), np.std(velocity), 1e-6)))
    embeddings, rhythm = catalog.embeddings[indices], catalog.rhythm_timbre[indices]
    arrays = dict(embeddings=embeddings, rhythm_timbre=rhythm, tags=catalog.tags[indices],
                  embedding_mean=embeddings.mean(axis=0), embedding_std=embeddings.std(axis=0),
                  rhythm_mean=rhythm.mean(axis=0), rhythm_std=rhythm.std(axis=0))
    return MusicCatalog(catalog.catalog_path, metadata, arrays)


def summarize(rows):
    result = {}
    for dataset in ["all", *sorted({row["dataset_id"] for row in rows})]:
        selected = rows if dataset == "all" else [row for row in rows if row["dataset_id"] == dataset]
        result[dataset] = dict(queries=len(selected),
                              recall_at_1=float(np.mean([r["hit1"] for r in selected])) if selected else None,
                              recall_at_3=float(np.mean([r["hit3"] for r in selected])) if selected else None,
                              accepted_rate=float(np.mean([r["accepted"] for r in selected])) if selected else None)
    return result


def retrieval_evaluation(catalog, queries_per_recording=1):
    matcher = MusicMotionMatcher(catalog)
    library_rows, heldout_rows, times = [], [], []
    for recording, indices in query_groups(catalog, queries_per_recording).items():
        reference = reference_without_recording(catalog, recording)
        heldout = MusicMotionMatcher(reference) if reference is not None else None
        for index in indices:
            descriptor = descriptor_from_segment(catalog, index)
            segment = catalog.segment_metadata[index]
            started = time.perf_counter()
            result = matcher.match(descriptor, top_k_tracks=3)
            times.append(1000 * (time.perf_counter() - started))
            ids = [track.music_id for track in result.tracks]
            library_rows.append(dict(dataset_id=recording[0], hit1=bool(ids and ids[0] == segment["music_id"]),
                                     hit3=segment["music_id"] in ids[:3], accepted=result.accepted))
            if heldout is not None:
                result = heldout.match(descriptor)
                genres = [genre.genre for genre in result.genres]
                heldout_rows.append(dict(dataset_id=recording[0], hit1=bool(genres and genres[0] == segment["genre"]),
                                         hit3=segment["genre"] in genres[:3], accepted=result.accepted))
    return dict(library_self_retrieval=summarize(library_rows),
                recording_held_out_genre_retrieval=summarize(heldout_rows),
                matcher_latency_ms=latency_summary(times),
                queries_per_recording=queries_per_recording,
                evaluation_note="Library self-retrieval is not generalization. Held-out normalization uses reference recordings only; genre labels remain dataset-specific.")


def latency_summary(values):
    if not values:
        return dict(count=0)
    return dict(count=len(values), median=float(np.median(values)), p95=float(np.percentile(values, 95)), max=float(max(values)))


def waveform_checks(catalog, extractor, limit_per_dataset=4):
    groups = defaultdict(list)
    for recording, indices in query_groups(catalog, 1).items():
        groups[recording[0]].append(indices[0])
    matcher = MusicMotionMatcher(catalog)
    rows, elapsed = [], []
    first_ms = None
    for dataset, indices in sorted(groups.items()):
        for index in indices[:limit_per_dataset]:
            item = catalog.segment_metadata[index]
            audio = load_audio_mono(catalog.audio_file(item), extractor.sample_rate)
            start, end = (int(round(item[key] * extractor.sample_rate)) for key in ("start_seconds", "end_seconds"))
            window = audio[start:end]
            predictions = []
            for gain in (.1, .5, 1., 2.):
                started = time.perf_counter()
                descriptor = extractor.describe(window * gain)
                result = matcher.match(descriptor)
                duration = 1000 * (time.perf_counter() - started)
                if first_ms is None:
                    first_ms = duration
                else:
                    elapsed.append(duration)
                predictions.append(result.tracks[0].music_id if result.tracks else None)
            rows.append(dict(dataset_id=dataset, music_id=item["music_id"],
                             unchanged=len(set(predictions)) == 1 and predictions[0] is not None))
    return dict(gain_checks=rows, extraction_and_match_ms=latency_summary(elapsed),
                first_extraction_and_match_ms=first_ms,
                within_one_second=bool(elapsed) and max(elapsed) < 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--queries-per-recording", type=int, default=1, help="Evenly spaced windows per recording; 0 evaluates all windows.")
    parser.add_argument("--gain-check-limit", type=int, default=4, help="Recordings per dataset for waveform gain and latency checks.")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.queries_per_recording < 0 or args.gain_check_limit < 0:
        parser.error("Query and gain-check limits must be nonnegative")
    started = time.perf_counter()
    catalog = MusicCatalog.load(args.catalog.resolve())
    loaded = time.perf_counter()
    metadata = catalog.metadata["extractor"]
    extractor = AudioFeatureExtractor(
        sample_rate=int(metadata["sample_rate"]), onnx_intra_op_threads=1,
        embedding_model=catalog.resolve_path(metadata["embedding_model"]["path"]) if metadata.get("embedding_model") else None,
        tag_model=catalog.resolve_path(metadata["tag_model"]["path"]) if metadata.get("tag_model") else None,
    )
    initialized = time.perf_counter()
    report = dict(catalog=str(catalog.catalog_path), build_id=catalog.metadata.get("build_id"),
                  counts=catalog.metadata.get("counts"), catalog_load_ms=1000*(loaded-started),
                  extractor_startup_ms=1000*(initialized-loaded),
                  **retrieval_evaluation(catalog, args.queries_per_recording),
                  **waveform_checks(catalog, extractor, args.gain_check_limit))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
