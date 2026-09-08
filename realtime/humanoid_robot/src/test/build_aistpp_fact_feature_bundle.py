"""Build the pinned AIST++/FACT FID and diversity feature bundle.

The output is reconstructed in the original 24-joint SMPL domain from the
motion IDs, phases and transition blend stored by the production trace.  The
first six seconds are excluded and the following 20 seconds are sampled at
60 FPS.  This script is intentionally separate from the online runner so that
feature extraction cannot contaminate latency measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent
HUMANOID_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aistpp_smpl import (  # noqa: E402
    Y_UP_TO_Z_UP,
    load_aistpp_motion,
    load_smpl_rest_pose,
    smpl_world_kinematics,
)
from aistplusplus_features import (  # noqa: E402
    extract_kinetic_features,
    extract_manual_features,
)
from music_motion_catalog import MusicCatalog  # noqa: E402
from run_humanoid_matcher_experiments import atomic_replace  # noqa: E402


UPSTREAM_COMMIT = "2dd7b3e946b794fd0081c98e2e2433545abf8b87"
DEFAULT_CATALOG = HUMANOID_DIR / "data" / "music_catalog" / "catalog.json"
DEFAULT_STATUS = TEST_DIR / "output" / "thesis_final" / "literature" / "run_status.json"
DEFAULT_OUTPUT = TEST_DIR / "output" / "thesis_final" / "features" / "aistpp_fact_features.npz"
DEFAULT_SMPL_MODEL = HUMANOID_DIR / "assets" / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--run-status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smpl-model", type=Path, default=DEFAULT_SMPL_MODEL)
    parser.add_argument("--warmup-seconds", type=float, default=6.0)
    parser.add_argument("--clip-seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=60.0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extractor_identity() -> tuple[str, dict[str, Any]]:
    source_paths = [
        TEST_DIR / "aistplusplus_features" / name
        for name in ("kinetic.py", "manual.py", "utils.py")
    ]
    hashes = {path.name: file_sha256(path) for path in source_paths}
    aggregate = hashlib.sha256(
        "".join(f"{name}:{value}\n" for name, value in sorted(hashes.items())).encode()
    ).hexdigest()
    identity = f"google/aistplusplus_api@{UPSTREAM_COMMIT};local_sha256={aggregate}"
    return identity, {
        "upstream_repository": "https://github.com/google/aistplusplus_api",
        "upstream_commit": UPSTREAM_COMMIT,
        "source_hashes": hashes,
        "aggregate_sha256": aggregate,
        "kinetic_dimensions": 72,
        "geometric_dimensions": 32,
    }


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "audio_time_seconds" not in rows[0]:
        raise ValueError(f"Trace is empty or malformed: {path}")
    return rows


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


class SmplPositionCache:
    """Lazily convert source AIST++ axis angles into official Y-up joints."""

    def __init__(self, catalog: MusicCatalog, smpl_model: Path):
        self.catalog = catalog
        self.aist_root = (
            catalog.catalog_path.parent / str(catalog.metadata["aistpp_root"])
        ).resolve()
        rest, parents = load_smpl_rest_pose(smpl_model.resolve())
        self.rest = rest
        self.parents = parents
        self.cache: dict[str, np.ndarray] = {}

    def full_motion(self, motion_id: str) -> np.ndarray:
        if motion_id not in self.cache:
            profile = self.catalog.motions.get(motion_id)
            if profile is None:
                raise KeyError(f"Trace references unknown motion ID: {motion_id}")
            motion_path = (self.aist_root / profile.motion_path).resolve()
            poses, translations = load_aistpp_motion(motion_path)
            z_up, _rotations = smpl_world_kinematics(
                poses, translations, self.rest, self.parents
            )
            # Invert aistpp_smpl.Y_UP_TO_Z_UP. The official feature extractor
            # assumes the original AIST++ Y-up coordinate convention.
            self.cache[motion_id] = np.asarray(z_up @ Y_UP_TO_Z_UP, dtype=np.float64)
        return self.cache[motion_id]

    def sample_phase(self, motion_id: str, phase: float) -> np.ndarray:
        positions = self.full_motion(motion_id)
        coordinate = float(np.clip(phase, 0.0, 1.0)) * (len(positions) - 1)
        left = int(np.floor(coordinate))
        right = min(left + 1, len(positions) - 1)
        fraction = coordinate - left
        return (1.0 - fraction) * positions[left] + fraction * positions[right]


def interpolate_trace_row(left: dict[str, str], right: dict[str, str], amount: float) -> dict[str, str]:
    """Interpolate only inside the same categorical motion/transition segment.

    A phase falling by over half a cycle is unwrapped; other resets are held,
    since low-rate traces cannot identify their exact sub-frame boundary.
    """
    row = dict(left)
    if any(left.get(key, "") != right.get(key, "")
           for key in ("current_motion_id", "transition_motion_id")):
        return row
    for key in ("current_phase", "transition_phase", "transition_blend"):
        first, last = _float(left, key), _float(right, key)
        delta = last - first
        if key.endswith("phase") and delta < -0.5:
            delta += 1.0
        elif delta < 0:
            continue
        value = first + float(np.clip(amount, 0, 1)) * delta
        row[key] = str(value % 1.0 if key.endswith("phase") else value)
    return row


def reconstruct_output_positions(
    rows: list[dict[str, str]],
    cache: SmplPositionCache,
    *,
    start_seconds: float,
    clip_seconds: float,
    fps: float,
) -> np.ndarray:
    """Reconstruct a continuous SMPL sequence from categorical trace samples."""

    trace_times = np.asarray([_float(row, "audio_time_seconds") for row in rows])
    sample_times = start_seconds + np.arange(int(round(clip_seconds * fps))) / fps
    trace_end_tolerance = max(0.1, 1.0 / fps)
    if trace_times[-1] + trace_end_tolerance < sample_times[-1]:
        raise ValueError(
            f"Trace ends at {trace_times[-1]:.3f}s, before required {sample_times[-1]:.3f}s"
        )
    indices = np.searchsorted(trace_times, sample_times, side="right") - 1
    indices = np.clip(indices, 0, len(rows) - 1)

    output = []
    active_motion = ""
    active_offset = np.zeros(3, dtype=np.float64)
    transition_motion = ""
    transition_offset = np.zeros(3, dtype=np.float64)
    previous_root = np.zeros(3, dtype=np.float64)
    for sample_index, row_index in enumerate(indices):
        row = rows[int(row_index)]
        next_index = int(row_index) + 1
        if next_index < len(rows) and trace_times[next_index] > trace_times[row_index]:
            amount = (sample_times[sample_index] - trace_times[row_index]) / (trace_times[next_index] - trace_times[row_index])
            row = interpolate_trace_row(row, rows[next_index], amount)
        current_id = row.get("current_motion_id", "").strip()
        if not current_id:
            raise ValueError(f"Trace has no current motion at {sample_times[sample_index]:.3f}s")
        current_raw = cache.sample_phase(current_id, _float(row, "current_phase"))
        if current_id != active_motion:
            if current_id == transition_motion:
                active_offset = transition_offset.copy()
            elif sample_index:
                active_offset = previous_root - current_raw[0]
            else:
                active_offset = -current_raw[0]
            active_motion = current_id
        current = current_raw + active_offset

        target_id = row.get("transition_motion_id", "").strip()
        blend = float(np.clip(_float(row, "transition_blend"), 0.0, 1.0))
        if target_id and blend > 0.0:
            target_raw = cache.sample_phase(target_id, _float(row, "transition_phase"))
            if target_id != transition_motion:
                transition_motion = target_id
                transition_offset = current[0] - target_raw[0]
            target = target_raw + transition_offset
            smooth_blend = blend * blend * (3.0 - 2.0 * blend)
            pose = (1.0 - smooth_blend) * current + smooth_blend * target
        else:
            if not target_id:
                transition_motion = ""
            pose = current
        previous_root = pose[0].copy()
        output.append(pose)
    return np.asarray(output, dtype=np.float64)


def ground_truth_positions(
    motion_id: str,
    cache: SmplPositionCache,
    *,
    start_seconds: float,
    clip_seconds: float,
    fps: float,
) -> np.ndarray:
    profile = cache.catalog.motions[motion_id]
    times = start_seconds + np.arange(int(round(clip_seconds * fps))) / fps
    duration = max(float(profile.duration_seconds), 1.0 / fps)
    phases = np.mod(times, duration) / duration
    positions = np.asarray([cache.sample_phase(motion_id, phase) for phase in phases])
    return positions - positions[0, 0]


def features(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kinetic = extract_kinetic_features(positions)
    geometric = extract_manual_features(positions)
    if kinetic.shape != (72,) or geometric.shape != (32,):
        raise ValueError(f"Unexpected feature shapes: {kinetic.shape}, {geometric.shape}")
    if not np.all(np.isfinite(kinetic)) or not np.all(np.isfinite(geometric)):
        raise ValueError("Official feature extractor emitted non-finite values")
    return kinetic, geometric


def build_bundle(
    catalog_path: Path,
    status_path: Path,
    output_path: Path,
    smpl_model: Path,
    *,
    warmup_seconds: float,
    clip_seconds: float,
    fps: float,
) -> dict[str, Any]:
    catalog = MusicCatalog.load(catalog_path.resolve())
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("suite") != "literature":
        raise ValueError(f"Expected literature status, got {status.get('suite')!r}")
    valid = [run for run in status.get("runs", []) if run.get("status") in {"completed", "reused"}]
    if len(valid) != int(status.get("expected_runs", -1)):
        raise ValueError("Literature suite is incomplete; refusing to build a partial bundle")
    per_music = Counter(str(run["source_id"]) for run in valid)
    if len(per_music) != 40 or set(per_music.values()) != {5}:
        raise ValueError(f"Expected 40 music IDs with five seeds each, got {dict(per_music)}")

    cache = SmplPositionCache(catalog, smpl_model)
    real_kinetic = []
    real_geometric = []
    real_ids = []
    for music_id in sorted(per_music):
        candidates = {str(run.get("ground_truth_motion_id", "")) for run in valid if run["source_id"] == music_id}
        if len(candidates) != 1 or not next(iter(candidates)):
            raise ValueError(f"Ground-truth motion is not uniquely frozen for {music_id}")
        motion_id = next(iter(candidates))
        kinetic, geometric = features(
            ground_truth_positions(
                motion_id,
                cache,
                start_seconds=warmup_seconds,
                clip_seconds=clip_seconds,
                fps=fps,
            )
        )
        real_kinetic.append(kinetic)
        real_geometric.append(geometric)
        real_ids.append(f"{music_id}:{motion_id}")

    output_kinetic = []
    output_geometric = []
    output_ids = []
    for run in sorted(valid, key=lambda value: (str(value["source_id"]), int(value["seed"]))):
        positions = reconstruct_output_positions(
            read_trace(Path(run["trace"])),
            cache,
            start_seconds=warmup_seconds,
            clip_seconds=clip_seconds,
            fps=fps,
        )
        kinetic, geometric = features(positions)
        output_kinetic.append(kinetic)
        output_geometric.append(geometric)
        output_ids.append(f"{run['source_id']}:seed={run['seed']}")

    identity, identity_data = extractor_identity()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        real_kinetic=np.asarray(real_kinetic, dtype=np.float32),
        output_kinetic=np.asarray(output_kinetic, dtype=np.float32),
        real_geometric=np.asarray(real_geometric, dtype=np.float32),
        output_geometric=np.asarray(output_geometric, dtype=np.float32),
        real_item_ids=np.asarray(real_ids),
        output_item_ids=np.asarray(output_ids),
        extractor_identity=np.asarray(identity),
        reconstruction_protocol=np.asarray(
            "v2: within-segment phase/blend interpolation, cyclic unwrap, boundary hold; "
            "smoothstep blend; translation-only root alignment (not exact G1 yaw/limiter reconstruction); 60 FPS"
        ),
        evaluation_window_seconds=np.asarray([warmup_seconds, warmup_seconds + clip_seconds]),
    )
    atomic_replace(temporary, output_path)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": str(output_path.resolve()),
        "bundle_sha256": file_sha256(output_path),
        "extractor_identity": identity,
        "extractor": identity_data,
        "reconstruction_source_sha256": {
            "feature_builder": file_sha256(Path(__file__).resolve()),
            "aistpp_smpl": file_sha256(SRC_DIR / "aistpp_smpl.py"),
        },
        "catalog_sha256": file_sha256(catalog_path),
        "run_status_sha256": file_sha256(status_path),
        "smpl_model_sha256": file_sha256(smpl_model),
        "real_items": len(real_ids),
        "output_items": len(output_ids),
        "warmup_seconds": warmup_seconds,
        "clip_seconds": clip_seconds,
        "fps": fps,
        "warning": (
            "Retrieval reuses real source motions, so FID/Div may be optimistic and must not "
            "be presented as a fully fair comparison with unconstrained generation models. "
            "Short ground-truth motions are cycled with possible root seams, not natural 20 s clips. "
            "Low-rate trace boundary timing and yaw alignment cannot be recovered exactly."
        ),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    atomic_replace(temp_manifest, manifest_path)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = build_bundle(
        args.catalog,
        args.run_status,
        args.output,
        args.smpl_model,
        warmup_seconds=args.warmup_seconds,
        clip_seconds=args.clip_seconds,
        fps=args.fps,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
