"""Fail fast on formal thesis experiment dependencies and immutable inputs."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent
HUMANOID_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from music_motion_catalog import MusicCatalog  # noqa: E402
from run_humanoid_matcher_experiments import (  # noqa: E402
    DEFAULT_CATALOG,
    DEFAULT_PROTOCOL,
    atomic_replace,
    environment_manifest,
    make_extractor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = MusicCatalog.load(args.catalog.resolve())
    errors = []
    for module in ("numpy", "scipy", "librosa", "onnxruntime", "mujoco", "soundfile"):
        try:
            importlib.import_module(module)
        except ImportError as error:
            errors.append(f"dependency {module}: {error}")
    extractor = make_extractor(catalog)
    manifest = environment_manifest(catalog, args.protocol.resolve(), extractor)
    providers = manifest["onnxruntime_providers"]
    if providers != ["CPUExecutionProvider"]:
        errors.append(f"expected only CPUExecutionProvider, got {providers!r}")
    for name, model in manifest["models"].items():
        if not model["exists"] or not model["sha256"]:
            errors.append(f"model {name} is missing or unhashed: {model['path']}")
    if len(catalog.tracks) != 60 or len(catalog.motions) != 411:
        errors.append(f"catalog counts are tracks={len(catalog.tracks)}, motions={len(catalog.motions)}")
    gmr_root = HUMANOID_DIR / "data" / "aistpp_gmr"
    gmr_files = [path for path in gmr_root.iterdir() if path.is_file()]
    if len(list(gmr_root.glob("*.pkl"))) != 411 or len(gmr_files) != 476:
        errors.append("GMR directory does not contain 411 valid PKL and 476 total records")
    cc0_count = len(list((HUMANOID_DIR / "data" / "test_audio" / "cc0_matcher_set").glob("*.mp3")))
    negative_count = len(list((HUMANOID_DIR / "data" / "test_audio" / "matcher_negative_set").glob("*.wav")))
    if cc0_count != 10 or negative_count != 20:
        errors.append(f"audio counts are CC0={cc0_count}, negative={negative_count}")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "counts": {
            "tracks": len(catalog.tracks),
            "motions": len(catalog.motions),
            "gmr_total": len(gmr_files),
            "gmr_valid": len(list(gmr_root.glob("*.pkl"))),
            "cc0": cc0_count,
            "negative": negative_count,
        },
        "manifest": manifest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    atomic_replace(temporary, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
