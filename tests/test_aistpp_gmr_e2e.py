from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import mujoco


ROOT = Path(__file__).resolve().parents[1]
HUMANOID = ROOT / "realtime" / "humanoid_robot"
SRC = HUMANOID / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from music_motion_catalog import MotionPreflightValidator
from gmr_retarget_smpl_headless import _violates_configured_clearance, collision_geom_pairs


GMR_PYTHON = HUMANOID / ".venv-gmr" / "Scripts" / "python.exe"
GMR_ROOT = HUMANOID / ".deps" / "GMR"
SMPL_ROOT = HUMANOID / "assets" / "body_models" / "smpl"
SOURCE_MOTION = HUMANOID / "data" / "aistpp" / "motions" / "gWA_sBM_cAll_d26_mWA0_ch07.pkl"
MODEL = HUMANOID / "assets" / "open_humanoid_dancer.xml"
E2E_AVAILABLE = all(
    path.exists()
    for path in (GMR_PYTHON, GMR_ROOT, SMPL_ROOT / "SMPL_NEUTRAL.pkl", SOURCE_MOTION, MODEL)
)


@unittest.skipUnless(E2E_AVAILABLE, "licensed SMPL model and isolated GMR environment are required")
class AistppGmrEndToEndTests(unittest.TestCase):
    def test_real_aistpp_to_gmr_to_headless_mujoco(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            output_root = temporary / "gmr"
            command = [
                str(GMR_PYTHON),
                str(SRC / "build_aistpp_gmr_dataset.py"),
                "--gmr-root",
                str(GMR_ROOT),
                "--gmr-python",
                str(GMR_PYTHON),
                "--smpl-model-path",
                str(SMPL_ROOT),
                "--motion",
                str(SOURCE_MOTION),
                "--output-root",
                str(output_root),
            ]
            subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True)

            artifact = output_root / f"{SOURCE_MOTION.stem}.pkl"
            with artifact.open("rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(1, payload["format_version"])
            self.assertEqual(4, payload["pipeline_version"])
            self.assertEqual("aistpp_smpl_direct", payload["source_format"])
            self.assertEqual("wxyz", payload["root_rot_order"])
            self.assertNotIn("bvh_sha256", payload)
            self.assertTrue(payload["collision_avoidance"]["enabled"])
            self.assertGreater(payload["collision_avoidance"]["geom_pairs"], 0)
            self.assertEqual(0, payload["continuity_limits"]["loop_closure_frames"])
            self.assertEqual((720, 29), np.asarray(payload["dof_pos"]).shape)
            self.assertEqual(29, len(payload["dof_names"]))
            self.assertTrue(np.all(np.isfinite(np.asarray(payload["root_pos"]))))

            model = mujoco.MjModel.from_xml_path(str(MODEL))
            data = mujoco.MjData(model)
            configured_pairs = {
                tuple(sorted((first, second)))
                for first_group, second_group in collision_geom_pairs(model)
                for first in first_group
                for second in second_group
            }
            pair_array = np.asarray(sorted(configured_pairs), dtype=np.int32)
            collision_states = ((model, data, configured_pairs, pair_array),)
            minimum_contact_distance = 0.0
            qpos = np.column_stack(
                (payload["root_pos"], payload["root_rot"], payload["dof_pos"])
            )
            for frame_index, frame in enumerate(qpos):
                previous = qpos[max(frame_index - 1, 0)]
                for amount in np.linspace(0.25, 1.0, 4):
                    probe = frame.copy()
                    probe[7:] = previous[7:] + amount * (frame[7:] - previous[7:])
                    self.assertFalse(
                        _violates_configured_clearance(
                            collision_states,
                            probe,
                            payload["collision_avoidance"]["minimum_distance_m"],
                        ),
                        f"clearance violation at frame {frame_index}, interpolation {amount:g}",
                    )
                    data.qpos[:] = probe
                    mujoco.mj_forward(model, data)
                    for contact_index in range(data.ncon):
                        contact = data.contact[contact_index]
                        pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                        if pair in configured_pairs:
                            minimum_contact_distance = min(
                                minimum_contact_distance, float(contact.dist)
                            )
            self.assertGreaterEqual(minimum_contact_distance, -1e-6)
            max_joint_step = float(np.max(np.abs(np.diff(payload["dof_pos"], axis=0))))
            self.assertLessEqual(max_joint_step, 3.0 * np.pi / 60.0 + 1e-12)

            validator = MotionPreflightValidator(MODEL, output_root)
            self.assertEqual((True, "ok"), validator.validate(SOURCE_MOTION, 60.0))

    def test_symmetric_down_arm_pose_preserves_both_arm_directions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            motion_path = temporary / "symmetric.pkl"
            poses = np.zeros((2, 24, 3), dtype=np.float64)
            poses[:, 16, 2] = -0.5 * np.pi
            poses[:, 17, 2] = 0.5 * np.pi
            with motion_path.open("wb") as handle:
                pickle.dump(
                    {"smpl_poses": poses.reshape(2, 72), "smpl_trans": np.zeros((2, 3))},
                    handle,
                )
            output_root = temporary / "gmr"
            subprocess.run(
                [
                    str(GMR_PYTHON),
                    str(SRC / "build_aistpp_gmr_dataset.py"),
                    "--gmr-root", str(GMR_ROOT),
                    "--gmr-python", str(GMR_PYTHON),
                    "--smpl-model-path", str(SMPL_ROOT),
                    "--motion", str(motion_path),
                    "--output-root", str(output_root),
                    "--overwrite",
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            audit_root = temporary / "audit"
            subprocess.run(
                [
                    str(GMR_PYTHON),
                    str(SRC / "audit_aistpp_retargeting.py"),
                    "--motion", str(motion_path),
                    "--gmr-motion-root", str(output_root),
                    "--smpl-model-path", str(SMPL_ROOT),
                    "--mujoco-model", str(MODEL),
                    "--output-root", str(audit_root),
                    "--no-html",
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            report = __import__("json").loads(
                (audit_root / motion_path.stem / "report.json").read_text(encoding="utf-8")
            )
            self.assertEqual("ok", report["status"])

            # The audit runs under the dedicated GMR environment where torch and
            # smplx are installed. Assert its serialized bone metrics here rather
            # than importing those optional dependencies into the main test venv.
            for bone in (
                "left_upper_arm",
                "right_upper_arm",
                "left_forearm",
                "right_forearm",
            ):
                median_cosine = report["pose_similarity"]["bones"][bone]["median_cosine"]
                self.assertGreater(float(median_cosine), 0.9, bone)


if __name__ == "__main__":
    unittest.main()
