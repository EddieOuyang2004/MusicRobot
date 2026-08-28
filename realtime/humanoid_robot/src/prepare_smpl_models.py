from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import types
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy import sparse


HUMANOID_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = HUMANOID_ROOT / "assets" / "SMPL_python_v.1.1.0.zip"
DEFAULT_OUTPUT_ROOT = HUMANOID_ROOT / "assets" / "body_models" / "smpl"
EXPECTED_ARCHIVE_SHA256 = "87c9e6cb1dddad79cf3b8b01760e240a9cd0c29d7ff14eda30fb07e3bd430c27"
MODEL_MEMBERS = {
    "SMPL_FEMALE.pkl": (
        "SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl"
    ),
    "SMPL_NEUTRAL.pkl": (
        "SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    ),
    "SMPL_MALE.pkl": (
        "SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl"
    ),
}


class _LegacyChumpyArray:
    """Minimal pickle target for SMPL v1.1.0's chumpy.ch.Ch shapedirs."""

    def __setstate__(self, state: dict[str, object]) -> None:
        self.state = state

    def as_array(self) -> np.ndarray:
        value = self.state.get("x")
        if value is None:
            raise ValueError("Legacy Chumpy object is missing its numeric 'x' value.")
        return np.asarray(value)


@contextmanager
def _legacy_chumpy_modules() -> Iterator[None]:
    previous_chumpy = sys.modules.get("chumpy")
    previous_chumpy_ch = sys.modules.get("chumpy.ch")
    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")
    chumpy_ch_module.Ch = _LegacyChumpyArray
    chumpy_module.ch = chumpy_ch_module
    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module
    try:
        yield
    finally:
        if previous_chumpy is None:
            sys.modules.pop("chumpy", None)
        else:
            sys.modules["chumpy"] = previous_chumpy
        if previous_chumpy_ch is None:
            sys.modules.pop("chumpy.ch", None)
        else:
            sys.modules["chumpy.ch"] = previous_chumpy_ch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_clean_model(archive: zipfile.ZipFile, member: str) -> dict[str, object]:
    if member not in archive.namelist():
        raise FileNotFoundError(f"SMPL archive is missing required member: {member}")
    with _legacy_chumpy_modules(), archive.open(member) as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected SMPL model dict in {member}, got {type(payload).__name__}.")
    cleaned = {
        key: value.as_array() if isinstance(value, _LegacyChumpyArray) else value
        for key, value in payload.items()
    }
    validate_model_payload(cleaned, member)
    return cleaned


def validate_model_payload(payload: dict[str, object], description: str) -> None:
    required = {"f", "J_regressor", "kintree_table", "weights", "posedirs", "v_template", "shapedirs"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{description} is missing SMPL fields: {sorted(missing)}")
    vertices = np.asarray(payload["v_template"])
    weights = np.asarray(payload["weights"])
    kintree = np.asarray(payload["kintree_table"])
    shapedirs = np.asarray(payload["shapedirs"])
    regressor = payload["J_regressor"]
    regressor_shape = regressor.shape if sparse.issparse(regressor) else np.asarray(regressor).shape
    if vertices.shape != (6890, 3):
        raise ValueError(f"{description} has invalid v_template shape {vertices.shape}.")
    if weights.shape != (6890, 24) or regressor_shape != (24, 6890):
        raise ValueError(
            f"{description} has invalid weights/J_regressor shapes {weights.shape}/{regressor_shape}."
        )
    if kintree.shape != (2, 24):
        raise ValueError(f"{description} has invalid kintree_table shape {kintree.shape}.")
    if shapedirs.shape[:2] != (6890, 3) or shapedirs.ndim != 3:
        raise ValueError(f"{description} has invalid shapedirs shape {shapedirs.shape}.")
    for key in ("v_template", "weights", "posedirs", "shapedirs"):
        if not np.all(np.isfinite(np.asarray(payload[key]))):
            raise ValueError(f"{description} contains non-finite {key} values.")


def validate_with_smplx(output_root: Path) -> None:
    try:
        import smplx  # type: ignore
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "smplx and torch are required for forward validation; run this script with the GMR Python."
        ) from exc
    model = smplx.create(
        # smplx.create() appends the model type directory ("smpl") itself.
        model_path=str(output_root.parent),
        model_type="smpl",
        gender="neutral",
        batch_size=1,
    )
    with torch.no_grad():
        result = model()
    joints = result.joints.detach().cpu().numpy()
    vertices = result.vertices.detach().cpu().numpy()
    if joints.shape[0] != 1 or joints.shape[1] < 24 or vertices.shape != (1, 6890, 3):
        raise ValueError(
            f"Unexpected smplx forward shapes: joints={joints.shape}, vertices={vertices.shape}."
        )
    if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(vertices)):
        raise ValueError("smplx neutral T-pose forward pass returned non-finite values.")


def prepare_models(
    archive_path: Path,
    output_root: Path,
    *,
    expected_sha256: str | None = EXPECTED_ARCHIVE_SHA256,
    validate_smplx: bool = True,
) -> None:
    archive_path = archive_path.resolve()
    output_root = output_root.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"SMPL archive not found: {archive_path}")
    actual_hash = sha256(archive_path)
    if expected_sha256 is not None and actual_hash.lower() != expected_sha256.lower():
        raise ValueError(
            f"SMPL archive SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for output_name, member in MODEL_MEMBERS.items():
            payload = load_and_clean_model(archive, member)
            destination = output_root / output_name
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(destination)
            print(f"Prepared {destination.name}")
    if validate_smplx:
        validate_with_smplx(output_root)
        print("Validated neutral SMPL T-pose with smplx.")
    print(f"SMPL archive SHA-256: {actual_hash}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare licensed SMPL v1.1.0 models for smplx.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--skip-smplx-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepare_models(
        args.archive,
        args.output_root,
        expected_sha256=None if args.skip_hash_check else EXPECTED_ARCHIVE_SHA256,
        validate_smplx=not args.skip_smplx_validation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
