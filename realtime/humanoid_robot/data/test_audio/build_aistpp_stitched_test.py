"""Build the short AIST++ audio montage described by the adjacent JSON file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf


MANIFEST_PATH = Path(__file__).with_name("aistpp_stitched_test.json")
TARGET_RMS = 0.20


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    sample_rate = int(manifest["sample_rate_hz"])
    clips: list[np.ndarray] = []

    for item in manifest["segments"]:
        source = (MANIFEST_PATH.parent / item["source"]).resolve()
        audio, source_rate = sf.read(source, always_2d=True, dtype="float32")
        if source_rate != sample_rate:
            raise ValueError(f"Unexpected sample rate in {source}: {source_rate}")

        start = round(float(item["start_seconds"]) * sample_rate)
        frame_count = round(float(item["duration_seconds"]) * sample_rate)
        clip = audio[start : start + frame_count].copy()
        if len(clip) != frame_count:
            raise ValueError(f"Requested segment exceeds {source}")

        clip -= np.mean(clip, axis=0, keepdims=True)
        rms = float(np.sqrt(np.mean(clip * clip)))
        peak = float(np.max(np.abs(clip)))
        gain = min(TARGET_RMS / max(rms, 1e-9), 0.95 / max(peak, 1e-9))
        clips.append(clip * gain)

    fade_frames = round(float(manifest["crossfade_seconds"]) * sample_rate)
    phase = np.linspace(0.0, np.pi / 2.0, fade_frames, dtype=np.float32)[:, None]
    fade_out = np.cos(phase)
    fade_in = np.sin(phase)
    output = clips[0]
    for clip in clips[1:]:
        overlap = output[-fade_frames:] * fade_out + clip[:fade_frames] * fade_in
        output = np.concatenate((output[:-fade_frames], overlap, clip[fade_frames:]), axis=0)

    peak = float(np.max(np.abs(output)))
    if peak > 0.98:
        output *= 0.98 / peak

    output_path = MANIFEST_PATH.with_name(manifest["output"])
    sf.write(output_path, output, sample_rate, subtype="PCM_16")
    print(f"created={output_path.resolve()}")
    print(
        f"duration={len(output) / sample_rate:.3f}s "
        f"frames={len(output)} peak={float(np.max(np.abs(output))):.4f}"
    )


if __name__ == "__main__":
    main()
