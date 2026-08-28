from __future__ import annotations

import argparse
from pathlib import Path

from music_motion_catalog import (
    DEFAULT_HOP_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    build_music_catalog,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AISTPP_ROOT = (
    PROJECT_ROOT / "realtime" / "humanoid_robot" / "data" / "aistpp"
)
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "realtime"
    / "humanoid_robot"
    / "assets"
    / "open_humanoid_dancer.xml"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "realtime" / "humanoid_robot" / "data" / "music_catalog"
)
DEFAULT_GMR_MOTION_ROOT = (
    PROJECT_ROOT / "realtime" / "humanoid_robot" / "data" / "aistpp_gmr"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a gain-invariant AIST++ music-to-motion retrieval catalog."
    )
    parser.add_argument("--aistpp-root", type=Path, default=DEFAULT_AISTPP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gmr-motion-root", type=Path, default=DEFAULT_GMR_MOTION_ROOT)
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=None,
        help="Optional Discogs multi-similarity EffNet ONNX model. DSP embedding is used when omitted.",
    )
    parser.add_argument(
        "--tag-model",
        type=Path,
        default=None,
        help="Optional Discogs 400-style ONNX model.",
    )
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--hop-seconds", type=float, default=DEFAULT_HOP_SECONDS)
    parser.add_argument("--limit-motions", type=int, default=None)
    parser.add_argument(
        "--skip-mujoco-preflight",
        action="store_true",
        help="Skip MuJoCo validation for fast feature experiments. Production catalogs should not use this.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog_path = build_music_catalog(
        aistpp_root=args.aistpp_root,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        tag_model=args.tag_model,
        mujoco_model=args.model,
        gmr_motion_root=args.gmr_motion_root,
        run_preflight=not args.skip_mujoco_preflight,
        limit_motions=args.limit_motions,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
    )
    print(f"Catalog written: {catalog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
