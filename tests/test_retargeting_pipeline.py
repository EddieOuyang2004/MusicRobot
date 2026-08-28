from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HUMANOID = ROOT / "realtime" / "humanoid_robot"
SRC = HUMANOID / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_aistpp_gmr_dataset import MotionBuildResult, validate_gmr_artifact, write_indexes
from aistpp_to_gmr_bvh import load_aistpp_motion
from prepare_smpl_models import MODEL_MEMBERS, prepare_models
from retargeting_diagnostics import (
    DiagnosticThresholds,
    PoseBoneSpec,
    Y_UP_TO_Z_UP,
    diagnose_continuity,
    diagnose_pose_similarity,
    smpl_world_kinematics,
)
from unitree_g1_dance_adapter import UnitreeG1DanceAdapter


GMR_PYTHON = HUMANOID / ".venv-gmr" / "Scripts" / "python.exe"
SMPL_ROOT = HUMANOID / "assets" / "body_models" / "smpl"


class ModelPreparationTests(unittest.TestCase):
    def test_prepare_uses_only_fixed_archive_members_and_output_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "models.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for member in MODEL_MEMBERS.values():
                    handle.writestr(member, b"placeholder")
                handle.writestr("../../escaped.pkl", b"must not be extracted")
            output = root / "body_models" / "smpl"
            payload = {"marker": np.asarray([1.0])}
            with (
                patch("prepare_smpl_models.load_and_clean_model", return_value=payload),
                patch("prepare_smpl_models.validate_with_smplx"),
            ):
                prepare_models(archive, output, expected_sha256=None, validate_smplx=True)
            self.assertEqual(set(MODEL_MEMBERS), {path.name for path in output.iterdir()})
            self.assertFalse((root / "escaped.pkl").exists())

    @unittest.skipUnless(
        GMR_PYTHON.is_file() and (SMPL_ROOT / "SMPL_NEUTRAL.pkl").is_file(),
        "prepared licensed models and isolated GMR Python are required",
    )
    def test_prepared_models_are_chumpy_free_and_smplx_forward_succeeds(self) -> None:
        script = """
import pickle, pathlib, numpy as np, smplx, torch
root = pathlib.Path(r'%s')
for path in root.glob('SMPL_*.pkl'):
    with path.open('rb') as handle:
        payload = pickle.load(handle, encoding='latin1')
    assert isinstance(payload['shapedirs'], np.ndarray)
    assert payload['v_template'].shape == (6890, 3)
    assert payload['kintree_table'].shape == (2, 24)
model = smplx.create(str(root.parent), model_type='smpl', gender='neutral', batch_size=1)
with torch.no_grad():
    result = model()
assert tuple(result.vertices.shape) == (1, 6890, 3)
assert bool(torch.isfinite(result.vertices).all())
assert bool(torch.isfinite(result.joints).all())
""" % str(SMPL_ROOT)
        subprocess.run([str(GMR_PYTHON), "-c", script], check=True, cwd=ROOT)


class CoordinateAndDiagnosticTests(unittest.TestCase):
    def test_aistpp_translation_scaling_becomes_bvh_centimetres(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.pkl"
            with path.open("wb") as handle:
                pickle.dump(
                    {
                        "smpl_poses": np.zeros((2, 72)),
                        "smpl_trans": np.asarray([[80.0, 160.0, -40.0], [0.0, 0.0, 0.0]]),
                        "smpl_scaling": np.asarray([80.0]),
                    },
                    handle,
                )
            poses, translations_cm = load_aistpp_motion(path)
            self.assertEqual((2, 24, 3), poses.shape)
            np.testing.assert_allclose([100.0, 200.0, -50.0], translations_cm[0])

    def test_smpl_fk_and_y_up_to_z_up_basis(self) -> None:
        poses = np.zeros((2, 2, 3), dtype=np.float64)
        translations = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        rest = np.asarray([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
        parents = np.asarray([-1, 0])
        positions, rotations = smpl_world_kinematics(poses, translations, rest, parents)
        np.testing.assert_allclose(Y_UP_TO_Z_UP @ np.asarray([0.0, 1.5, 2.0]), positions[0, 0])
        np.testing.assert_allclose(Y_UP_TO_Z_UP @ np.asarray([1.0, 1.5, 2.0]), positions[0, 1])
        self.assertEqual((2, 2, 4), rotations.shape)

    def _diagnose(
        self,
        raw: np.ndarray,
        smpl: np.ndarray,
        bvh: np.ndarray,
        root: np.ndarray,
        dofs: np.ndarray,
    ) -> dict[str, object]:
        return diagnose_continuity(
            raw_axis_angles=raw,
            smpl_positions=smpl,
            bvh_positions=bvh,
            g1_root_positions=root,
            g1_root_quaternions_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
            g1_dof_positions=dofs,
            fps=60.0,
            smpl_joint_names=("root", "child"),
            dof_names=("joint",),
            thresholds=DiagnosticThresholds(),
        )

    def test_axis_angle_wrap_is_parameterization_only(self) -> None:
        raw = np.zeros((3, 2, 3))
        raw[0, 1, 0] = np.pi - 0.01
        raw[1:, 1, 0] = -np.pi + 0.01
        skeleton = np.zeros((3, 2, 3))
        result = self._diagnose(raw, skeleton, skeleton, np.zeros((3, 3)), np.zeros((3, 1)))
        self.assertEqual("ok", result["status"])
        self.assertEqual("parameterization_only", result["events"][0]["classification"])

    def test_discontinuities_are_attributed_to_the_first_real_layer(self) -> None:
        raw = np.zeros((3, 2, 3))
        base = np.zeros((3, 2, 3))
        root = np.zeros((3, 3))
        dofs = np.zeros((3, 1))

        source = base.copy()
        source[1:, 1, 0] = 1.0
        result = self._diagnose(raw, source, source, root, dofs)
        self.assertEqual("source_smpl", result["events"][0]["classification"])

        bvh = base.copy()
        bvh[1:, 1, 0] = 0.01
        result = self._diagnose(raw, base, bvh, root, dofs)
        self.assertEqual("bvh", result["events"][0]["classification"])

        dofs[1:, 0] = 0.5
        result = self._diagnose(raw, base, base, root, dofs)
        self.assertEqual("gmr_g1", result["events"][0]["classification"])

    def test_continuous_right_arm_flip_fails_pose_similarity(self) -> None:
        source = np.zeros((3, 3, 3), dtype=np.float64)
        source[:, 1, 0] = 1.0
        source[:, 2, 0] = 2.0
        target = source.copy()
        target[:, 2, 0] = 0.0
        result = diagnose_pose_similarity(
            source_positions=source,
            target_positions=target,
            source_names=("shoulder", "elbow", "wrist"),
            target_names=("shoulder", "elbow", "wrist"),
            bones=(PoseBoneSpec("right_forearm", "elbow", "wrist", "elbow", "wrist"),),
        )
        self.assertEqual("failed", result["status"])
        self.assertEqual(["right_forearm"], result["failed_bones"])
        self.assertEqual("pose_similarity", result["events"][0]["classification"])


class ArtifactSchemaTests(unittest.TestCase):
    def _payload(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "pipeline_version": 4,
            "source_format": "aistpp_smpl_direct",
            "fps": 60.0,
            "root_pos": np.zeros((2, 3)),
            "root_rot": np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
            "root_rot_order": "wxyz",
            "dof_pos": np.zeros((2, 29)),
            "dof_names": list(UnitreeG1DanceAdapter.GMR_DOF_NAMES),
            "source_motion_id": "test",
            "source_sha256": "s" * 64,
            "smpl_model_sha256": "m" * 64,
            "retargeter": "GMR",
            "retargeter_version": "a" * 40,
            "collision_avoidance": {
                "enabled": True,
                "preset": "g1_self_collision_v2",
            },
            "mink_limits_api": "mink_1_3_keyword_limits_adapter",
            "continuity_limits": {"loop_closure_frames": 0},
        }

    def test_cache_accepts_only_collision_aware_smpl_direct_v4_without_bvh_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.pkl"
            with valid_path.open("wb") as handle:
                pickle.dump(self._payload(), handle)
            validate_gmr_artifact(valid_path)

            cases = (
                ("v1", "pipeline_version", 1, "pipeline_version"),
                ("v3", "pipeline_version", 3, "pipeline_version"),
                ("wrong_source", "source_format", "bvh_lafan1", "source_format"),
                ("bvh_hash", "bvh_sha256", "b" * 64, "BVH provenance"),
            )
            for name, key, value, message in cases:
                payload = self._payload()
                payload[key] = value
                path = root / f"{name}.pkl"
                with path.open("wb") as handle:
                    pickle.dump(payload, handle)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    validate_gmr_artifact(path)

    def test_cache_rejects_missing_wxyz_and_wrong_dof_names_or_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases: list[tuple[str, dict[str, object], str]] = []

            missing_order = self._payload()
            missing_order.pop("root_rot_order")
            cases.append(("missing_wxyz", missing_order, "missing fields"))

            wrong_name = self._payload()
            wrong_name["dof_names"] = list(wrong_name["dof_names"])
            wrong_name["dof_names"][0] = "mystery_joint"
            cases.append(("wrong_name", wrong_name, "names/order"))

            wrong_order = self._payload()
            wrong_order["dof_names"] = list(reversed(wrong_order["dof_names"]))
            cases.append(("wrong_order", wrong_order, "names/order"))

            for name, payload, message in cases:
                path = root / f"{name}.pkl"
                with path.open("wb") as handle:
                    pickle.dump(payload, handle)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    validate_gmr_artifact(path)


class BatchIndexTests(unittest.TestCase):
    def _args(self, output: Path, *, full: bool = False) -> argparse.Namespace:
        return argparse.Namespace(
            output_root=output,
            motion=None if full else Path("selected.pkl"),
            split="all",
            limit=None,
            gmr_version="a" * 40,
            resume=False,
        )

    @staticmethod
    def _result(output: Path, motion_id: str) -> MotionBuildResult:
        entry = {
            "source_motion_id": motion_id,
            "artifact": f"{motion_id}.pkl",
            "frames": 2,
            "fps": 60.0,
            "source_sha256": motion_id * 4,
            "smpl_model_sha256": "m" * 64,
            "gmr_commit": "a" * 40,
        }
        return MotionBuildResult(Path(f"{motion_id}.pkl"), output / f"{motion_id}.pkl", "ok", entry)

    def test_incremental_manifest_merges_and_failure_removes_selected_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_indexes(self._args(output), [self._result(output, "first")], [])
            write_indexes(self._args(output), [self._result(output, "second")], [])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(["first", "second"], [item["source_motion_id"] for item in manifest["motions"]])
            self.assertEqual(4, manifest["pipeline_version"])
            self.assertEqual("aistpp_smpl_direct", manifest["source_format"])

            failure = {"source_motion_id": "second", "message": "broken"}
            stale_artifact = output / "second.pkl"
            stale_artifact.write_bytes(b"old-v4")
            write_indexes(self._args(output), [], [failure])
            manifest = json.loads((output / "manifest.json").read_text())
            failures = json.loads((output / "failures.json").read_text())
            self.assertEqual(["first"], [item["source_motion_id"] for item in manifest["motions"]])
            self.assertEqual(["second"], [item["source_motion_id"] for item in failures["failures"]])
            self.assertFalse(stale_artifact.exists())
            self.assertTrue((output / "second.pkl.invalid").exists())

    def test_full_manifest_rebuild_drops_unselected_previous_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_indexes(self._args(output), [self._result(output, "old")], [])
            write_indexes(self._args(output, full=True), [self._result(output, "new")], [])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(["new"], [item["source_motion_id"] for item in manifest["motions"]])


if __name__ == "__main__":
    unittest.main()
