from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from music_motion_catalog import (
    AudioFeatureExtractor,
    MusicCatalog,
    MusicMotionMatcher,
    iter_audio_windows,
    load_audio_mono,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = (
    ROOT / "realtime" / "humanoid_robot" / "data" / "music_catalog" / "catalog.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AIST++ catalog retrieval and recording-gain invariance."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--gain-check-limit",
        type=int,
        default=10,
        help="Number of representative music IDs used for waveform gain checks.",
    )
    return parser.parse_args()


def descriptor_from_segment(
    catalog: MusicCatalog,
    index: int,
):
    from music_motion_catalog import AudioDescriptor

    metadata = catalog.segment_metadata[index]
    return AudioDescriptor(
        embedding=catalog.embeddings[index],
        rhythm_timbre=catalog.rhythm_timbre[index],
        tag_probabilities=catalog.tags[index],
        bpm=float(metadata["bpm"]),
        beat_strength=float(metadata["beat_strength"]),
        onset_density=float(metadata["onset_density"]),
        offbeat_ratio=float(metadata["offbeat_ratio"]),
        tempo_stability=float(metadata["tempo_stability"]),
        spectral_flux=float(metadata["spectral_flux"]),
        percussive_ratio=float(metadata["percussive_ratio"]),
    )


def segment_recall(catalog: MusicCatalog, matcher: MusicMotionMatcher) -> float:
    correct = 0
    for index, segment in enumerate(catalog.segment_metadata):
        result = matcher.match(
            descriptor_from_segment(catalog, index),
            top_k_tracks=1,
            top_k_motions=1,
        )
        if result.tracks and result.tracks[0].music_id == segment["music_id"]:
            correct += 1
    return correct / max(len(catalog.segment_metadata), 1)


def leave_one_music_genre_recall_at_3(
    catalog: MusicCatalog,
    matcher: MusicMotionMatcher,
) -> float:
    successes = 0
    total = 0
    for query_index, query_metadata in enumerate(catalog.segment_metadata):
        query_music = str(query_metadata["music_id"])
        query_genre = str(query_metadata["genre"])
        query_embedding = matcher._standardized_unit(
            catalog.embeddings[query_index],
            catalog.embedding_mean,
            catalog.embedding_std,
        )
        query_rhythm = matcher._standardized_unit(
            catalog.rhythm_timbre[query_index],
            catalog.rhythm_mean,
            catalog.rhythm_std,
        )
        embedding_scores = matcher._similarities(
            matcher._catalog_embeddings,
            query_embedding,
        )
        rhythm_scores = matcher._similarities(
            matcher._catalog_rhythm,
            query_rhythm,
        )
        if catalog.tags.shape[1]:
            query_tags = catalog.tags[query_index]
            tag_scores = matcher._similarities(
                matcher._catalog_tags,
                query_tags / max(float(np.linalg.norm(query_tags)), 1e-12),
            )
            scores = (
                0.70 * embedding_scores
                + 0.20 * rhythm_scores
                + 0.10 * tag_scores
            )
        else:
            scores = (0.70 / 0.90) * embedding_scores + (0.20 / 0.90) * rhythm_scores
        grouped: dict[str, list[float]] = {}
        for index, segment in enumerate(catalog.segment_metadata):
            music_id = str(segment["music_id"])
            if music_id == query_music:
                continue
            grouped.setdefault(music_id, []).append(float(scores[index]))
        ranked = sorted(
            grouped,
            key=lambda music_id: float(np.mean(sorted(grouped[music_id], reverse=True)[:3])),
            reverse=True,
        )[:3]
        if any(str(catalog.tracks[music_id]["genre"]) == query_genre for music_id in ranked):
            successes += 1
        total += 1
    return successes / max(total, 1)


def gain_invariance(
    catalog: MusicCatalog,
    matcher: MusicMotionMatcher,
    extractor: AudioFeatureExtractor,
    limit: int,
) -> tuple[int, int]:
    checked = 0
    unchanged = 0
    root = Path(catalog.metadata["aistpp_root"])
    for music_id in sorted(catalog.tracks)[: max(limit, 0)]:
        variant = catalog.tracks[music_id]["variants"][0]
        audio = load_audio_mono(root / variant["source_audio"], extractor.sample_rate)
        window = next(
            iter(
                iter_audio_windows(
                    audio,
                    extractor.sample_rate,
                    window_seconds=float(catalog.metadata["extractor"]["window_seconds"]),
                    hop_seconds=float(catalog.metadata["extractor"]["hop_seconds"]),
                )
            )
        )[2]
        predictions = []
        for gain in (0.1, 0.5, 1.0, 2.0):
            result = matcher.match(
                extractor.describe(gain * window),
                top_k_tracks=1,
                top_k_motions=1,
            )
            predictions.append(result.tracks[0].music_id if result.tracks else None)
        checked += 1
        if len(set(predictions)) == 1:
            unchanged += 1
    return unchanged, checked


def main() -> int:
    args = parse_args()
    catalog = MusicCatalog.load(args.catalog)
    matcher = MusicMotionMatcher(catalog)
    extractor_metadata = catalog.metadata["extractor"]
    embedding_model = (
        Path(extractor_metadata["embedding_model"]["path"])
        if extractor_metadata.get("embedding_model")
        else None
    )
    tag_model = (
        Path(extractor_metadata["tag_model"]["path"])
        if extractor_metadata.get("tag_model")
        else None
    )
    if embedding_model is not None and not embedding_model.is_absolute():
        embedding_model = ROOT / embedding_model
    if tag_model is not None and not tag_model.is_absolute():
        tag_model = ROOT / tag_model
    extractor = AudioFeatureExtractor(
        sample_rate=int(extractor_metadata["sample_rate"]),
        embedding_model=embedding_model,
        tag_model=tag_model,
    )

    recall_at_1 = segment_recall(catalog, matcher)
    genre_recall_at_3 = leave_one_music_genre_recall_at_3(catalog, matcher)
    gain_unchanged, gain_checked = gain_invariance(
        catalog,
        matcher,
        extractor,
        args.gain_check_limit,
    )
    gain_rate = gain_unchanged / max(gain_checked, 1)
    print(f"segments={len(catalog.segment_metadata)} music_ids={len(catalog.tracks)}")
    print(f"same-music segment Recall@1={recall_at_1:.4f}")
    print(f"leave-one-music genre Recall@3={genre_recall_at_3:.4f}")
    print(
        f"gain-invariant top-1={gain_unchanged}/{gain_checked} "
        f"({gain_rate:.4f}) for gains 0.1x/0.5x/1x/2x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
