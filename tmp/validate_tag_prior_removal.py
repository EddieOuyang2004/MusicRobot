"""Replay the saved 20-song no-tag-prior reference without changing audit inputs."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "realtime/humanoid_robot/src"
sys.path.insert(0, str(SRC))
from music_motion_catalog import MusicCatalog, MusicMotionMatcher, load_audio_mono
from realtime_music_humanoid_matcher_v2 import make_extractor


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare(actual, expected, location):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), location
        for key in expected:
            compare(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), location
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{location}[{index}]")
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7, err_msg=location)
    else:
        assert actual == expected, f"{location}: {actual!r} != {expected!r}"


def main():
    base = SRC / "test/output/v2_selection_audit_20"
    reference_path = base / "improvement_probe/probe.json"
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    catalog_path = ROOT / manifest["catalog"]
    catalog = MusicCatalog.load(catalog_path)
    inputs = [catalog_path, catalog_path.parent / catalog.metadata["arrays_file"]]
    for kind in ("embedding_model", "tag_model"):
        inputs.append((catalog_path.parent / catalog.metadata["extractor"][kind]["path"]).resolve())
    for path in inputs:
        assert sha(path) == manifest["sha256"][path.relative_to(ROOT).as_posix()], path
    extractor = make_extractor(argparse.Namespace(embedding_model=None, tag_model=None), catalog)
    songs = {s["id"]: s for s in manifest["songs"]}
    rows = []
    for entry in reference["rows"]:
        song = songs[entry["song"]]
        audio_path = Path(song["prepared_audio"])
        assert sha(audio_path) == song["prepared_sha256"], audio_path
        assert song["source_seconds"] >= 26, song["id"]
        audio = load_audio_mono(audio_path, extractor.sample_rate)[:26 * extractor.sample_rate]
        descriptor = extractor.describe(audio)
        matcher = MusicMotionMatcher(catalog, speed_max=1.3)
        matcher._weak_music_streak = entry["previous_weak_streak"]
        result = matcher.match(descriptor, top_k_tracks=5, top_k_motions=20)
        actual = json.loads(json.dumps(asdict(result)))
        compare(actual, entry["results"]["no_tag_genre_prior"], song["id"])
        rows.append({"song": song["id"], "title": song["title"], "matched_reference": True, "result": actual})
        print(f"{song['id']} {song['title']}: all result fields match", flush=True)
    assert len(rows) == 20 and len({row['song'] for row in rows}) == 20
    output = SRC / "test/output/tag_prior_removed_validation"
    output.mkdir(parents=True, exist_ok=True)
    payload = {"passed": True, "songs": len(rows), "window_seconds": 26,
               "reference": str(reference_path), "reference_sha256": sha(reference_path),
               "matcher_sha256": sha(SRC / "music_motion_catalog.py"),
               "comparison": "All MatchResult fields; float rtol=1e-5, atol=1e-7; exact IDs/order/acceptance/reasons.",
               "rows": rows}
    (output / "replay26.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: 20/20 songs. Report: {output / 'replay26.json'}")


if __name__ == "__main__":
    main()
