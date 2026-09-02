"""Build the fixed 20-clip non-dance rejection stress set.

The speech-like clips are synthetic formant/noise signals, not recordings of
human speech.  This keeps the set redistributable and deterministic; a real
speech directory can additionally be supplied to the experiment runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


TEST_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = TEST_DIR.parent.parent / "data" / "test_audio" / "matcher_negative_set"


def normalize(audio: np.ndarray, peak: float = 0.20) -> np.ndarray:
    maximum = max(float(np.max(np.abs(audio))), 1e-12)
    return np.asarray(audio * (peak / maximum), dtype=np.float32)


def pink_noise(rng: np.random.Generator, samples: int) -> np.ndarray:
    white = rng.normal(size=samples)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(samples)
    scale = np.ones_like(frequencies)
    scale[1:] = 1.0 / np.sqrt(frequencies[1:])
    return normalize(np.fft.irfft(spectrum * scale, n=samples))


def ambient_drone(
    rng: np.random.Generator,
    samples: int,
    sample_rate: int,
    index: int,
) -> np.ndarray:
    time = np.arange(samples, dtype=np.float64) / sample_rate
    fundamental = 45.0 + 7.0 * index
    signal = sum(
        (1.0 / harmonic) * np.sin(2.0 * np.pi * fundamental * harmonic * time + rng.uniform(0, 2 * np.pi))
        for harmonic in range(1, 6)
    )
    envelope = 0.65 + 0.35 * np.sin(2.0 * np.pi * (0.03 + 0.004 * index) * time)
    return normalize(signal * envelope + 0.05 * rng.normal(size=samples))


def speech_like(
    rng: np.random.Generator,
    samples: int,
    sample_rate: int,
    index: int,
) -> np.ndarray:
    time = np.arange(samples, dtype=np.float64) / sample_rate
    pitch = 95.0 + 14.0 * index + 8.0 * np.sin(2.0 * np.pi * 0.35 * time)
    phase = 2.0 * np.pi * np.cumsum(pitch) / sample_rate
    source = np.sin(phase) + 0.35 * np.sin(2.0 * phase) + 0.18 * np.sin(3.0 * phase)
    syllable_rate = 2.7 + 0.23 * index
    syllables = np.maximum(np.sin(2.0 * np.pi * syllable_rate * time), 0.0) ** 1.7
    pauses = (np.sin(2.0 * np.pi * 0.19 * time + index) > -0.35).astype(float)
    aspiration = rng.normal(size=samples) * 0.08
    return normalize((source + aspiration) * syllables * pauses)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples = int(round(args.seconds * args.sample_rate))
    manifest = {
        "schema_version": 1,
        "sample_rate_hz": args.sample_rate,
        "duration_seconds": args.seconds,
        "clips": [],
        "note": "Synthetic objective rejection stress data; speech_like is not real speech.",
    }
    for category in ("silence", "pink_noise", "ambient_drone", "speech_like"):
        for index in range(5):
            rng = np.random.default_rng(10_000 * (index + 1) + len(category))
            if category == "silence":
                audio = np.zeros(samples, dtype=np.float32)
            elif category == "pink_noise":
                audio = pink_noise(rng, samples)
            elif category == "ambient_drone":
                audio = ambient_drone(rng, samples, args.sample_rate, index)
            else:
                audio = speech_like(rng, samples, args.sample_rate, index)
            path = output / f"{category}_{index + 1:02d}.wav"
            sf.write(path, audio, args.sample_rate, subtype="PCM_16")
            manifest["clips"].append(
                {"file": path.name, "category": category, "expected_reject": True}
            )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest['clips'])} clips and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
