"""Build deterministic tempo-jump, genre-jump and silence/recovery streams."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import wave
from pathlib import Path

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent
HUMANOID_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from music_motion_catalog import load_audio_mono  # noqa: E402


def write_wave(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def click_segment(bpm: float, seconds: float, sample_rate: int) -> np.ndarray:
    count = int(round(seconds * sample_rate))
    time_axis = np.arange(count, dtype=np.float64) / sample_rate
    samples = 0.06 * np.sin(2.0 * np.pi * 110.0 * time_axis)
    click_length = max(int(round(0.045 * sample_rate)), 2)
    envelope = np.hanning(click_length * 2)[:click_length]
    click = 0.75 * envelope * np.sin(
        2.0 * np.pi * 1_200.0 * np.arange(click_length) / sample_rate
    )
    for beat in np.arange(0.0, seconds, 60.0 / bpm):
        start = int(round(beat * sample_rate))
        end = min(start + click_length, count)
        samples[start:end] += click[: end - start]
    return np.asarray(samples, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HUMANOID_DIR / "data" / "test_audio" / "matcher_change_streams",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    first = click_segment(90.0, 20.0, args.sample_rate)
    second = click_segment(150.0, 20.0, args.sample_rate)
    tempo_path = output / "tempo_jump_90_to_150.wav"
    write_wave(tempo_path, np.concatenate((first, second)), args.sample_rate)

    stitched_source = HUMANOID_DIR / "data" / "test_audio" / "aistpp_stitched_test.wav"
    genre_path = output / "genre_jump_aistpp.wav"
    shutil.copyfile(stitched_source, genre_path)

    stitched = load_audio_mono(stitched_source, args.sample_rate)
    before = np.resize(stitched, int(12.0 * args.sample_rate))
    silence = np.zeros(int(8.0 * args.sample_rate), dtype=np.float32)
    after_source = stitched[int(12.0 * args.sample_rate) :]
    after = np.resize(after_source if after_source.size else stitched, int(20.0 * args.sample_rate))
    silence_path = output / "silence_recovery.wav"
    write_wave(silence_path, np.concatenate((before, silence, after)), args.sample_rate)

    manifest = {
        "schema_version": 1,
        "sample_rate_hz": args.sample_rate,
        "streams": [
            {
                "id": "tempo_jump",
                "path": tempo_path.name,
                "duration_seconds": 40.0,
                "changes": [{"time_seconds": 20.0, "from_bpm": 90, "to_bpm": 150}],
            },
            {
                "id": "genre_jump",
                "path": genre_path.name,
                "duration_seconds": 39.0,
                "changes": [7.75, 15.5, 23.25, 31.0],
            },
            {
                "id": "silence_recovery",
                "path": silence_path.name,
                "duration_seconds": 40.0,
                "changes": [
                    {"time_seconds": 12.0, "state": "silence"},
                    {"time_seconds": 20.0, "state": "music"},
                ],
            },
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
