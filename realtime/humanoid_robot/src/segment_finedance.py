"""Segment released FineDance motion/audio into AIST++ body-motion files.

See docs/finedance_segmentation.md for the lossy skeleton mapping and timing
contract. Requires only the project's numpy, scipy, and soundfile dependencies.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation, Slerp

from aistpp_smpl import load_aistpp_motion


DATA = Path(__file__).resolve().parents[1] / "data"
FPS = 60
AUDIO_RATE = 44_100
# Upstream FineDance_dataset.get_train_test_list("cross_genre"). Ignore wins
# where upstream lists overlap (120 and 130); no invented validation split.
GENRE_TEST = set("063 132 143 036 098 198 130 012 211 193 179 065 137 161 092 120 037 109 204 144".split())
GENRE_IGNORE = set("116 117 118 119 120 121 122 123 202 130".split())


def rotation_6d_to_matrix(values: np.ndarray) -> np.ndarray:
    """Decode PyTorch3D's first-two-ROWS representation, not columns."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError("Rotations must be finite 6D vectors.")
    first, second = values[..., :3], values[..., 3:]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate first rotation vector.")
    first = first / norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Collinear rotation vectors.")
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=-2)


def load_finedance(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return metre translations and 52 local rotation matrices.

    Accept raw 315D, preprocessed 319D (four contact flags), or decoded 159D
    axis angles. No guesswork for other layouts; contacts are never poses.
    """
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or values.shape[1] not in (315, 319, 159):
        raise ValueError(f"Expected (N,315/319/159) FineDance motion: {path}, {values.shape}")
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"Need at least two finite motion frames: {path}")
    if values.shape[1] == 319:
        values = values[:, 4:]
    trans = np.asarray(values[:, :3], dtype=np.float64)
    if values.shape[1] == 315:
        matrices = rotation_6d_to_matrix(values[:, 3:].reshape(-1, 52, 6))
    else:
        matrices = Rotation.from_rotvec(values[:, 3:].reshape(-1, 3)).as_matrix()
        matrices = matrices.reshape(-1, 52, 3, 3)
    return trans, matrices


def segment_bounds(total_frames: int, min_frames: int = 480,
                   max_frames: int = 720, target_frames: int = 600) -> list[tuple[int, int]]:
    """Balanced, contiguous, nonoverlapping clips near the preferred duration.

    Cover every frame if a legal partition exists. Otherwise keep the longest
    legal prefix (e.g. 15 seconds -> 12 seconds + 3 seconds reported as dropped).
    """
    if not 0 < min_frames <= target_frames <= max_frames:
        raise ValueError("Require 0 < minimum <= target <= maximum frames.")
    if total_frames < min_frames:
        return []
    fewest = math.ceil(total_frames / max_frames)
    most = total_frames // min_frames
    if fewest <= most:
        count = min(most, max(fewest, int(total_frames / target_frames + 0.5)))
        usable = total_frames
    else:
        count, usable = most, most * max_frames
    base, remainder = divmod(usable, count)
    bounds, start = [], 0
    for index in range(count):
        end = start + base + int(index < remainder)
        bounds.append((start, end))
        start = end
    return bounds


def resample_motion(trans: np.ndarray, matrices: np.ndarray, frame_count: int,
                    source_fps: float, root_y_offset: float) -> tuple[np.ndarray, np.ndarray]:
    """SLERP body rotations; linearly interpolate translation on real timestamps."""
    source_times = np.arange(len(trans), dtype=np.float64) / source_fps
    # N source frames occupy N/source_fps seconds. Hold the last pose for the
    # final source-frame interval instead of stretching the entire recording.
    times = np.minimum(np.arange(frame_count, dtype=np.float64) / FPS, source_times[-1])
    poses = np.zeros((frame_count, 24, 3), dtype=np.float32)
    for joint in range(22):
        poses[:, joint] = Slerp(source_times, Rotation.from_matrix(matrices[:, joint]))(times).as_rotvec()
    # SMPL-H 22..51 are fingers, NOT SMPL's two terminal hand joints. Neutral
    # hand joints make this an explicit body-only approximation, not a refit.
    translations = np.column_stack([np.interp(times, source_times, trans[:, axis]) for axis in range(3)])
    translations[:, 1] += root_y_offset
    return poses.reshape(-1, 72), translations.astype(np.float32)


def source_splits(ids: list[str], split_json: Path | None) -> dict[str, str]:
    if split_json is None:
        return {name: "ignore" if name in GENRE_IGNORE else "test" if name in GENRE_TEST else "train"
                for name in ids}
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) - {"train", "val", "test", "ignore"}:
        raise ValueError("Split JSON must map train/val/test/ignore to source-ID lists.")
    result: dict[str, str] = {}
    for split, names in payload.items():
        if not isinstance(names, list):
            raise ValueError(f"Split {split} must be a list.")
        for name in names:
            if not isinstance(name, (str, int)):
                raise ValueError("Source IDs must be strings or integers.")
            name = Path(str(name)).stem.zfill(3)
            if name in result:
                raise ValueError(f"Source {name} occurs more than once in split JSON.")
            result[name] = split
    missing = set(ids) - result.keys()
    if missing:
        raise ValueError(f"Split JSON does not assign sources: {sorted(missing)}")
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict:
    root, output = args.input_root.resolve(), args.output_root.resolve()
    paths = sorted((root / "motion").glob("*.npy"))
    if not paths:
        raise ValueError(f"No .npy motions found in {root / 'motion'}")
    splits = source_splits([p.stem for p in paths], args.split_json)
    if args.limit is not None:
        paths = paths[:args.limit]
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Output must be separate from the source dataset tree.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty; choose a new --output-root: {output}")
    # Validate before writing anything. Do not silently swallow malformed files.
    frame_limits = [int(round(s * FPS)) for s in (args.min_seconds, args.max_seconds, args.target_seconds)]
    plans, excluded = [], []
    for path in paths:
        name = path.stem
        if splits[name] == "ignore":
            excluded.append({"source_id": name, "reason": "split_ignore"})
            continue
        trans, _ = load_finedance(path)
        wav_path = root / "music_wav" / f"{name}.wav"
        if not wav_path.is_file():
            raise FileNotFoundError(f"Missing paired audio: {wav_path}")
        info = sf.info(wav_path)
        if info.channels not in (1, 2):
            raise ValueError(f"Expected mono or stereo audio: {wav_path}")
        motion_seconds = len(trans) / args.source_fps
        delta = info.duration - motion_seconds
        mismatch = abs(delta) > args.max_duration_mismatch_seconds + 1e-9
        if mismatch and not args.allow_duration_mismatch:
            excluded.append({"source_id": name, "reason": "duration_mismatch",
                             "audio_minus_motion_seconds": delta})
            continue
        total = math.floor(min(motion_seconds, info.duration) * FPS + 1e-9)
        bounds = segment_bounds(total, *frame_limits)
        if not bounds:
            excluded.append({"source_id": name, "reason": "shorter_than_minimum",
                             "available_seconds": total / FPS})
            continue
        label_path = root / "label_json" / f"{name}.json"
        label = json.loads(label_path.read_text(encoding="utf-8")) if label_path.exists() else None
        plans.append({"source_id": name, "split": splits[name], "source_frames": len(trans),
                      "source_audio_rate": info.samplerate, "source_audio_channels": info.channels,
                      "source_audio_seconds": info.duration, "source_motion_seconds": motion_seconds,
                      "audio_minus_motion_seconds": delta, "duration_mismatch_allowed": mismatch,
                      "dropped_motion_seconds": motion_seconds - bounds[-1][1] / FPS,
                      "dropped_audio_seconds": info.duration - bounds[-1][1] / FPS,
                      "bounds": bounds, "label": label})
    report = {
        "format_version": 1, "dry_run": args.dry_run, "input_root": str(root),
        "fps": FPS, "source_fps": args.source_fps, "audio_rate": AUDIO_RATE,
        "min_seconds": args.min_seconds, "max_seconds": args.max_seconds,
        "target_seconds": args.target_seconds, "root_y_offset_m": args.root_y_offset,
        "max_duration_mismatch_seconds": args.max_duration_mismatch_seconds,
        "allow_duration_mismatch": args.allow_duration_mismatch,
        "split_policy": str(args.split_json.resolve()) if args.split_json else "FineDance cross_genre, ignore precedence",
        "sources_selected": len(paths), "sources_accepted": len(plans),
        "clip_count": sum(len(p["bounds"]) for p in plans), "excluded": excluded,
        "compatibility": {
            "motion_schema": "smpl_poses[N,72] radians; smpl_trans[N,3] metres; smpl_scaling[1]=1",
            "coordinates": "Y-up; original heading and horizontal translation retained",
            "skeleton": "SMPL-H body joints 0..21 copied; SMPL joints 22..23 neutral; fingers omitted",
            "geometry": "Body-pose transfer, not a mesh/shape refit; floor contact is not guaranteed",
            "timing": "N/fps nominal duration; SLERP, linear translation, endpoint hold; common audio interval",
            "sync": "Paired recordings assumed aligned at t=0; duration checks cannot verify semantic synchronization",
            "catalog": "AIST++-specific name/genre parser needs adaptation; names remain finedance_*",
            "features": "Source music_npy/contact features are not exported; recompute from clips",
        },
        "sources": plans,
    }
    output.mkdir(parents=True, exist_ok=True)
    # Keep a reviewable plan even if the export is interrupted. Only the final
    # manifest marks a completed export; a partial output cannot be reused.
    write_json(output / "plan.json", report)
    if args.dry_run:
        return report
    if not plans:
        raise ValueError(f"No eligible sources. See {output / 'plan.json'}")
    for folder in ("motions", "audio", "labels"):
        (output / folder).mkdir()
    clips, split_clips = [], {split: [] for split in ("train", "val", "test")}
    for plan_index, plan in enumerate(plans):
        name = plan["source_id"]
        trans, matrices = load_finedance(root / "motion" / f"{name}.npy")
        count = plan["bounds"][-1][1]
        poses, translations = resample_motion(trans, matrices, count, args.source_fps, args.root_y_offset)
        audio, rate = sf.read(root / "music_wav" / f"{name}.wav", dtype="float32", always_2d=True)
        if not np.isfinite(audio).all():
            raise ValueError(f"Non-finite audio samples: {name}")
        if rate != AUDIO_RATE:
            divisor = math.gcd(rate, AUDIO_RATE)
            audio = resample_poly(audio, AUDIO_RATE // divisor, rate // divisor, axis=0)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        for start, end in plan["bounds"]:
            clip_id = f"finedance_{name}_{start:07d}_{end:07d}"
            sample_start, sample_end = start * (AUDIO_RATE // FPS), end * (AUDIO_RATE // FPS)
            if sample_end > len(audio):
                raise ValueError(f"Audio shorter than planned clip: {clip_id}")
            entry = {"clip_id": clip_id, "source_id": name, "split": plan["split"],
                     "start_frame": start, "end_frame_exclusive": end, "frames": end - start,
                     "start_seconds": start / FPS, "end_seconds": end / FPS,
                     "duration_seconds": (end - start) / FPS,
                     "authored_duration_seconds": (end - start - 1) / FPS,
                     "motion_path": f"motions/{clip_id}.pkl", "audio_path": f"audio/{clip_id}.wav",
                     "audio_samples": sample_end - sample_start, "label": plan["label"]}
            payload = {"smpl_poses": poses[start:end], "smpl_trans": translations[start:end],
                       "smpl_scaling": np.ones(1, dtype=np.float32), "fps": FPS,
                       "source_dataset": "FineDance", "source_id": name,
                       "source_start_seconds": start / FPS, "source_fps": args.source_fps,
                       "root_y_offset_m": args.root_y_offset, "body_only_approximation": True}
            motion_path, audio_path = output / entry["motion_path"], output / entry["audio_path"]
            with motion_path.open("xb") as handle:
                pickle.dump(payload, handle, protocol=4)
            sf.write(audio_path, audio[sample_start:sample_end], AUDIO_RATE, subtype="PCM_16")
            # Verify through the actual project loader and the on-disk WAV header.
            loaded_poses, loaded_trans = load_aistpp_motion(motion_path)
            saved_audio = sf.info(audio_path)
            if (loaded_poses.shape != (end - start, 24, 3)
                    or loaded_trans.shape != (end - start, 3)
                    or saved_audio.frames != sample_end - sample_start
                    or saved_audio.samplerate != AUDIO_RATE or saved_audio.channels != 2):
                raise ValueError(f"Export verification failed: {clip_id}")
            write_json(output / "labels" / f"{clip_id}.json", entry)
            clips.append(entry)
            split_clips[plan["split"]].append(clip_id)
        print(f"[{plan_index + 1}/{len(plans)}] {name}: {len(plan['bounds'])} clips", flush=True)
    for split, names in split_clips.items():
        (output / f"{split}.txt").write_text("".join(name + "\n" for name in names), encoding="utf-8")
    with (output / "audio" / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["motion_name", "frames", "duration_seconds", "audio_path", "source_id", "split"])
        writer.writeheader()
        for entry in clips:
            writer.writerow({"motion_name": entry["clip_id"], "frames": entry["frames"],
                             "duration_seconds": entry["duration_seconds"],
                             "audio_path": Path(entry["audio_path"]).name,
                             "source_id": entry["source_id"], "split": entry["split"]})
    report["clips"] = clips
    report["export_complete"] = True
    write_json(output / "manifest.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DATA / "finedance")
    parser.add_argument("--output-root", type=Path, default=DATA / "finedance_aistpp")
    parser.add_argument("--source-fps", type=float, default=30.0, help="Actual source FPS; output is always 60 FPS.")
    parser.add_argument("--min-seconds", type=float, default=8.0)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--target-seconds", type=float, default=10.0)
    parser.add_argument("--root-y-offset", type=float, default=1.3, help="Metres added to Y, matching upstream FineDance render.py; use 0 for untouched translation.")
    parser.add_argument("--max-duration-mismatch-seconds", type=float, default=0.1)
    parser.add_argument("--allow-duration-mismatch", action="store_true", help="Explicitly allow mismatches; truncate to common interval and record them.")
    parser.add_argument("--split-json", type=Path, help="Optional train/val/test/ignore lists of source IDs; all local sources must be assigned.")
    parser.add_argument("--limit", type=int, help="Inspect/export only the first N source files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate sources and write only plan.json to an empty output directory.")
    args = parser.parse_args(argv)
    numbers = (args.source_fps, args.min_seconds, args.max_seconds, args.target_seconds,
               args.root_y_offset, args.max_duration_mismatch_seconds)
    if not all(math.isfinite(x) for x in numbers):
        parser.error("Numeric options must be finite.")
    if args.source_fps <= 0 or args.max_duration_mismatch_seconds < 0:
        parser.error("Source FPS must be positive and mismatch tolerance nonnegative.")
    if not 8 <= args.min_seconds <= args.target_seconds <= args.max_seconds <= 12:
        parser.error("Require 8 <= min-seconds <= target-seconds <= max-seconds <= 12.")
    if any(abs(s * FPS - round(s * FPS)) > 1e-7 for s in numbers[1:4]):
        parser.error("Segment durations must be exact multiples of 1/60 second.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    return args


def main() -> int:
    args = parse_args()
    report = build(args)
    print(f"{'Planned' if args.dry_run else 'Exported'} {report['clip_count']} clips at {FPS} FPS "
          f"from {report['sources_accepted']} sources; {len(report['excluded'])} excluded. "
          f"Output: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
