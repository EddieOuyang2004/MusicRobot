from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SRC = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from music_motion_catalog import (  # noqa: E402
    AudioFeatureExtractor,
    MusicCatalog,
    MusicMotionMatcher,
    iter_audio_windows,
    load_audio_mono,
)


DEFAULT_CATALOG = ROOT / "realtime" / "humanoid_robot" / "data" / "music_catalog" / "catalog.json"
DEFAULT_AUDIO_ROOT = ROOT / "realtime" / "humanoid_robot" / "data" / "test_audio" / "cc0_matcher_set"


def _model_path(catalog: MusicCatalog, key: str) -> Path | None:
    model = catalog.metadata.get("extractor", {}).get(key)
    if not isinstance(model, dict) or not model.get("path"):
        return None
    path = Path(model["path"])
    return path if path.is_absolute() else catalog.catalog_path.parent / path


def run_benchmark(
    catalog_path: Path,
    audio_root: Path,
    window_seconds: float,
    *,
    hop_seconds: float = 1.0,
    max_seconds: float = 30.0,
) -> dict[str, Any]:
    catalog = MusicCatalog.load(catalog_path)
    extractor = AudioFeatureExtractor(
        sample_rate=int(catalog.metadata["extractor"]["sample_rate"]),
        embedding_model=_model_path(catalog, "embedding_model"),
        tag_model=_model_path(catalog, "tag_model"),
        onnx_intra_op_threads=1,
    )
    matcher = MusicMotionMatcher(catalog, style_first=True)
    genre_counts: Counter[str] = Counter()
    motion_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    files: dict[str, Any] = {}
    total = accepted = 0

    for audio_path in sorted(audio_root.glob("*.mp3")):
        audio = load_audio_mono(audio_path, extractor.sample_rate)
        audio = audio[: int(round(max_seconds * extractor.sample_rate))]
        file_genres: Counter[str] = Counter()
        file_motions: Counter[str] = Counter()
        file_rejections: Counter[str] = Counter()
        file_total = file_accepted = 0
        for _, _, window in iter_audio_windows(
            audio,
            extractor.sample_rate,
            window_seconds,
            hop_seconds,
        ):
            result = matcher.match(extractor.describe(window))
            total += 1
            file_total += 1
            if not result.accepted or not result.motions:
                reason = result.rejection_reason or "no_compatible_motion"
                rejection_counts[reason] += 1
                file_rejections[reason] += 1
                continue
            accepted += 1
            file_accepted += 1
            genre = result.genres[0].genre if result.genres else "unknown"
            motion_id = result.motions[0].motion_id
            genre_counts[genre] += 1
            motion_counts[motion_id] += 1
            file_genres[genre] += 1
            file_motions[motion_id] += 1
        files[audio_path.name] = {
            "windows": file_total,
            "accepted": file_accepted,
            "top_genres": file_genres.most_common(3),
            "top_motions": file_motions.most_common(3),
            "rejections": dict(file_rejections),
        }

    dominant_genre = genre_counts.most_common(1)[0] if genre_counts else ("", 0)
    dominant_motion = motion_counts.most_common(1)[0] if motion_counts else ("", 0)
    ambient_candidates = sum(
        item["accepted"] for name, item in files.items() if "ambient" in name.lower()
    )
    return {
        "catalog": str(catalog_path),
        "window_seconds": window_seconds,
        "hop_seconds": hop_seconds,
        "total_windows": total,
        "accepted_windows": accepted,
        "rejected_windows": total - accepted,
        "unique_genres": len(genre_counts),
        "unique_motions": len(motion_counts),
        "dominant_genre": dominant_genre[0],
        "dominant_genre_share": dominant_genre[1] / max(accepted, 1),
        "dominant_motion": dominant_motion[0],
        "dominant_motion_share": dominant_motion[1] / max(accepted, 1),
        "ambient_switch_candidates": ambient_candidates,
        "genre_counts": dict(genre_counts),
        "motion_counts": dict(motion_counts),
        "rejection_counts": dict(rejection_counts),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CC0 matcher diversity by window size.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--windows", default="3,6,9,12")
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--assert-style-first", action="store_true")
    args = parser.parse_args()
    reports = [
        run_benchmark(
            args.catalog.resolve(),
            args.audio_root.resolve(),
            float(window),
            hop_seconds=args.hop_seconds,
            max_seconds=args.max_seconds,
        )
        for window in args.windows.split(",")
        if window.strip()
    ]
    payload = {"reports": reports}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if args.assert_style_first:
        failed = [
            report
            for report in reports
            if report["dominant_genre_share"] > 0.35
            or report["dominant_motion_share"] > 0.20
            or report["ambient_switch_candidates"] > 0
        ]
        if failed:
            print("Style-first acceptance thresholds failed.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
