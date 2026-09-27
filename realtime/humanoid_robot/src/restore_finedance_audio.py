"""Restore release-aligned FineDance audio from original WAVs, without GMR."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "realtime/humanoid_robot/data"
RECIPE = ROOT / "docs/finedance_reproduction"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def wav_bytes(audio):
    buffer = io.BytesIO()
    sf.write(buffer, audio, 44100, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def restore(input_root, output_root, recipe=RECIPE, source_ids=(), verify_only=False,
            all_clips=False):
    input_root, output_root, recipe = map(lambda p: Path(p).resolve(),
                                         (input_root, output_root, recipe))
    if (input_root == output_root or input_root in output_root.parents
            or output_root in input_root.parents):
        raise ValueError("Output must be separate from the source dataset tree.")
    manifest_path = recipe / "segmentation_manifest.json"
    checks = json.loads((recipe / "checksums.json").read_text(encoding="utf-8"))
    if sha256(manifest_path) != checks["segmentation_manifest_sha256"]:
        raise ValueError("Segmentation manifest checksum mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["fps"] != 60 or manifest["audio_rate"] != 44100:
        raise ValueError("Expected the published 60 FPS / 44100 Hz recipe.")
    selected = set(source_ids)
    unknown = selected - set(checks["sources"])
    if unknown:
        raise ValueError(f"Unknown source IDs: {sorted(unknown)}")
    groups = {}
    for clip in manifest["clips"]:
        if selected and clip["source_id"] not in selected:
            continue
        if not all_clips and checks["clips"][clip["clip_id"]]["gmr_sha256"] is None:
            continue
        start, end = clip["start_frame"], clip["end_frame_exclusive"]
        sid = clip["source_id"]
        name = f"finedance_{sid}_{start:07d}_{end:07d}"
        if (not sid.isdigit() or not 0 <= start < end
                or name != clip["clip_id"] or clip["audio_path"] != f"audio/{name}.wav"
                or clip["audio_samples"] != (end - start) * 735):
            raise ValueError(f"Inconsistent clip timing/path: {clip['clip_id']}")
        groups.setdefault(sid, []).append(clip)
    if not groups:
        raise ValueError("No clips selected.")
    count = 0
    for sid, clips in sorted(groups.items()):
        source = input_root / "music_wav" / f"{sid}.wav"
        if sha256(source) != checks["sources"][sid]["audio_sha256"]:
            raise ValueError(f"Original WAV differs from the released source: {source}")
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        if audio.shape[1] not in (1, 2) or not np.isfinite(audio).all():
            raise ValueError(f"Invalid source audio: {source}")
        # Resample the WHOLE recording before cutting, matching segment_finedance.
        if rate != 44100:
            divisor = math.gcd(rate, 44100)
            audio = resample_poly(audio, 44100 // divisor, rate // divisor, axis=0)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        for clip in clips:
            start, end = clip["start_frame"] * 735, clip["end_frame_exclusive"] * 735
            if end > len(audio):
                raise ValueError(f"Source too short: {clip['clip_id']}")
            payload = wav_bytes(audio[start:end])
            expected = checks["clips"][clip["clip_id"]]["audio_sha256"]
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Restored WAV checksum mismatch: {clip['clip_id']}; "
                                 "check the documented NumPy/SciPy/SoundFile versions.")
            target = (output_root / clip["audio_path"]).resolve()
            if not target.is_relative_to(output_root):
                raise ValueError("Output audio path escapes output root.")
            if not verify_only:
                if target.exists():
                    if sha256(target) != expected:
                        raise FileExistsError(f"Refusing to overwrite different audio: {target}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as handle:
                        handle.write(payload)
            count += 1
        print(f"{sid}: verified {len(clips)} clips", flush=True)
    print(f"{count} byte-identical WAVs {'verified in memory' if verify_only else 'restored/verified'}.")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DATA / "finedance")
    parser.add_argument("--output-root", type=Path, default=DATA / "finedance_aistpp")
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument("--source-id", action="append", default=[], help="Repeatable, e.g. 001.")
    parser.add_argument("--verify-only", action="store_true", help="Regenerate/check in memory; write no audio.")
    parser.add_argument("--all-clips", action="store_true", help="Include the eight clips without successful GMR.")
    args = parser.parse_args()
    restore(args.input_root, args.output_root, args.recipe, args.source_id,
            args.verify_only, args.all_clips)


if __name__ == "__main__":
    main()
