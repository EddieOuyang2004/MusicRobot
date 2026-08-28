from __future__ import annotations

import math
import pickle
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "realtime" / "humanoid_robot" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from music_pose_modulator import MusicPoseModulator
from gmr_retarget_bvh_headless import apply_loop_closure, limit_qpos_velocity
from gmr_retarget_smpl_headless import collision_geom_pairs, install_gmr_mink_limits_compatibility
from realtime_music_humanoid_dancer import (
    AistppMotionSampler,
    FeatureState,
    GmrUnitreeG1MotionSampler,
    MujocoHumanoidPlayer,
)
from realtime_music_humanoid_matcher import load_motion_sampler
from robot_motion import RobotMotionFrame, RootMotionContinuity, align_motion_frame_root, blend_motion_frames
from unitree_g1_dance_adapter import UnitreeG1DanceAdapter, UnitreeG1JointPoseAdapter


class HumanoidRetargetingTests(unittest.TestCase):
    def test_realtime_projection_scales_colliding_crossed_hands_to_safe_pose(self) -> None:
        model_path = ROOT / "realtime" / "humanoid_robot" / "assets" / "open_humanoid_dancer.xml"
        player = MujocoHumanoidPlayer(model_path, realtime=False, headless=True)
        crossed_arms = {
            "left_shoulder_pitch": -1.397803572925763,
            "left_shoulder_roll": 0.19197895788978014,
            "left_shoulder_yaw": -0.7298091788876543,
            "left_elbow": 0.6708478365713926,
            "left_wrist_roll": 0.14865423896211932,
            "left_wrist_pitch": 0.09302005079270192,
            "left_wrist_yaw": 0.3409800583164294,
            "right_shoulder_pitch": -1.3957220447029512,
            "right_shoulder_roll": -0.3026502352493593,
            "right_shoulder_yaw": 0.7186641141781159,
            "right_elbow": 1.0114740943633653,
            "right_wrist_roll": -0.07763248042729448,
            "right_wrist_pitch": -0.11344634339866903,
            "right_wrist_yaw": -0.7440059338849634,
        }
        candidate = RobotMotionFrame(crossed_arms)
        player.collision_scratch.qpos[:] = player.data.qpos
        player._write_frame(player.collision_scratch, candidate, count_limits=False)
        self.assertTrue(player._has_self_clearance_violation(player.collision_scratch, 0.006))

        projected = player.project_self_collision_safe(candidate)
        player.collision_scratch.qpos[:] = player.data.qpos
        player._write_frame(player.collision_scratch, projected, count_limits=False)
        self.assertFalse(player._has_self_clearance_violation(player.collision_scratch, 0.006))
        self.assertEqual(1, player.collision_projection_count)

    def test_playback_collision_pairs_include_rubber_hands_and_head(self) -> None:
        model_path = ROOT / "realtime" / "humanoid_robot" / "assets" / "open_humanoid_dancer.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))

        def active_mesh_geom(mesh_name: str) -> int:
            mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
            matches = [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
                and int(model.geom_dataid[geom_id]) == mesh_id
                and (int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0)
            ]
            self.assertEqual(1, len(matches), mesh_name)
            return matches[0]

        left_hand = active_mesh_geom("left_rubber_hand")
        right_hand = active_mesh_geom("right_rubber_hand")
        head = active_mesh_geom("head_link")
        expanded = {
            tuple(sorted((first, second)))
            for first_group, second_group in collision_geom_pairs(model)
            for first in first_group
            for second in second_group
        }
        self.assertIn(tuple(sorted((left_hand, right_hand))), expanded)
        self.assertIn(tuple(sorted((left_hand, head))), expanded)
        self.assertIn(tuple(sorted((right_hand, head))), expanded)

    def test_gmr_mink_13_adapter_routes_positional_list_to_limits_keyword(self) -> None:
        observed: dict[str, object] = {}

        def mink_13_solve_ik(
            configuration: object,
            tasks: object,
            dt: float,
            solver: str,
            damping: float = 1e-12,
            safety_break: bool = False,
            limits: object = None,
        ) -> np.ndarray:
            observed["safety_break"] = safety_break
            observed["limits"] = limits
            return np.zeros(1)

        module = SimpleNamespace(mink=SimpleNamespace(solve_ik=mink_13_solve_ik))
        mode = install_gmr_mink_limits_compatibility(module)
        configured_limits = [object(), object()]
        module.mink.solve_ik(object(), [], 0.002, "daqp", 0.5, configured_limits)
        self.assertEqual("mink_1_3_keyword_limits_adapter", mode)
        self.assertFalse(observed["safety_break"])
        self.assertIs(configured_limits, observed["limits"])

    def _write_aistpp(self, path: Path) -> None:
        poses = np.zeros((2, 24, 3), dtype=np.float32)
        poses[:, 16, 2] = -0.5 * math.pi
        poses[:, 17, 2] = 0.5 * math.pi
        poses[1, 18, 1] = 1.0
        poses[1, 20, 0] = 0.4
        payload = {
            "smpl_poses": poses.reshape(2, 72),
            "smpl_trans": np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32),
            "smpl_scaling": np.asarray([2.0], dtype=np.float32),
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def _write_gmr(
        self,
        path: Path,
        *,
        zero_quaternion: bool = False,
        reverse_names: bool = False,
    ) -> None:
        names = UnitreeG1DanceAdapter.GMR_DOF_NAMES
        if reverse_names:
            names = tuple(reversed(names))
        first = np.arange(29, dtype=np.float32)
        second = first + 2.0
        quaternions = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        if zero_quaternion:
            quaternions[0] = 0.0
        payload = {
            "format_version": 1,
            "pipeline_version": 4,
            "source_format": "aistpp_smpl_direct",
            "fps": 60.0,
            "root_pos": np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "root_rot": quaternions,
            "root_rot_order": "wxyz",
            "dof_pos": np.stack([first, second]),
            "dof_names": names,
            "source_motion_id": path.stem,
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
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def test_direct_fallback_maps_down_arms_to_neutral_and_uses_wrist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.pkl"
            self._write_aistpp(path)
            sampler = AistppMotionSampler(path, 60.0, 1.0, 0.0)

            neutral = sampler._retarget(sampler.frames[0])
            bent = sampler._retarget(sampler.frames[1])
            self.assertLess(abs(neutral["left_shoulder_roll"]), 1e-6)
            self.assertLess(abs(neutral["right_shoulder_roll"]), 1e-6)
            self.assertGreater(bent["left_elbow"], 0.9)
            self.assertAlmostEqual(0.4, bent["left_wrist_roll"], places=5)
            midpoint = sampler.sample_frame(0.25, 1.0, 0.0, FeatureState())
            self.assertAlmostEqual(0.5, midpoint.joint_positions["left_elbow"], places=5)
            self.assertAlmostEqual(0.2, midpoint.joint_positions["left_wrist_roll"], places=5)
            np.testing.assert_allclose([0.25, -0.75, 0.5], midpoint.root_position, atol=1e-6)
            frame = sampler.sample_frame(0.5, 1.0, 0.0, FeatureState())
            np.testing.assert_allclose([0.5, -1.5, 1.0], frame.root_position, atol=1e-6)

    def test_direct_fallback_uses_global_bone_angles_and_converted_axes(self) -> None:
        sampler = object.__new__(AistppMotionSampler)
        t_pose = np.zeros((24, 3), dtype=np.float64)
        mapped_t = sampler._retarget(t_pose)
        self.assertAlmostEqual(-0.5 * math.pi, mapped_t["left_shoulder_pitch"], places=6)
        self.assertAlmostEqual(0.5 * math.pi, mapped_t["right_shoulder_pitch"], places=6)
        self.assertAlmostEqual(0.0, mapped_t["left_shoulder_roll"], places=6)

        crouch = t_pose.copy()
        crouch[4, 0] = 0.8
        crouch[5, 0] = -0.6
        mapped_crouch = sampler._retarget(crouch)
        self.assertAlmostEqual(0.8, mapped_crouch["left_knee"], places=6)
        self.assertAlmostEqual(0.6, mapped_crouch["right_knee"], places=6)

        wrist_turn = t_pose.copy()
        wrist_turn[20, 2] = 0.3
        mapped_wrist = sampler._retarget(wrist_turn)
        self.assertAlmostEqual(-0.3, mapped_wrist["left_wrist_pitch"], places=6)
        self.assertAlmostEqual(0.0, mapped_wrist["left_wrist_yaw"], places=6)

        hip_roll = t_pose.copy()
        hip_roll[1, 0] = 0.3
        hip_roll[2, 0] = 0.3
        mapped_hip = sampler._retarget(hip_roll)
        self.assertAlmostEqual(-0.3, mapped_hip["left_hip_roll"], places=6)
        self.assertAlmostEqual(0.3, mapped_hip["right_hip_roll"], places=6)

        self.assertLess(sampler._morphology_compressed_flexion(math.pi, 1.75, 2.05), 2.05)
        deep_elbow = t_pose.copy()
        deep_elbow[18, 1] = 2.4
        mapped_deep, diagnostics = sampler._retarget_with_diagnostics(deep_elbow)
        self.assertAlmostEqual(2.4, diagnostics["left_elbow"][0], places=6)
        self.assertLess(mapped_deep["left_elbow"], diagnostics["left_elbow"][0])

    def test_gmr_loader_preserves_qpos_order_and_slerps_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.pkl"
            self._write_gmr(path)
            sampler = GmrUnitreeG1MotionSampler(path, None, 1.0, 0.0, False)
            frame = sampler.sample_frame(0.25, 1.0, 0.0, FeatureState())

            self.assertAlmostEqual(1.0, frame.joint_positions["left_hip_pitch"])
            np.testing.assert_allclose([1.0, 0.0, 0.0], frame.root_position, atol=1e-7)
            rotation = Rotation.from_quat(
                [
                    frame.root_quaternion_wxyz[1],
                    frame.root_quaternion_wxyz[2],
                    frame.root_quaternion_wxyz[3],
                    frame.root_quaternion_wxyz[0],
                ]
            )
            self.assertAlmostEqual(0.5 * math.pi, rotation.magnitude(), places=6)
            end_frame = sampler.sample_frame(0.75, 1.0, 0.0, FeatureState())
            np.testing.assert_allclose([2.0, 0.0, 0.0], end_frame.root_position, atol=1e-7)

    def test_gmr_loader_rejects_zero_root_quaternion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.pkl"
            self._write_gmr(path, zero_quaternion=True)
            with self.assertRaisesRegex(ValueError, "zero length"):
                GmrUnitreeG1MotionSampler(path, None, 1.0, 0.0, False)

    def test_gmr_loader_rejects_noncanonical_xyzw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.pkl"
            self._write_gmr(path)
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            wxyz = payload["root_rot"].copy()
            payload["root_rot"] = wxyz[:, [1, 2, 3, 0]]
            payload["root_rot_order"] = "xyzw"
            with path.open("wb") as handle:
                pickle.dump(payload, handle)
            with self.assertRaisesRegex(ValueError, "root_rot_order"):
                GmrUnitreeG1MotionSampler(path, None, 1.0, 0.0, False)

    def test_gmr_loader_refuses_ambiguous_quaternion_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.pkl"
            self._write_gmr(path)
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            payload.pop("root_rot_order")
            with path.open("wb") as handle:
                pickle.dump(payload, handle)
            with self.assertRaisesRegex(ValueError, "root_rot_order"):
                GmrUnitreeG1MotionSampler(path, None, 1.0, 0.0, False)

    def test_gmr_loader_rejects_v1_and_wrong_source_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "motion.pkl"
            self._write_gmr(path)
            with path.open("rb") as handle:
                valid = pickle.load(handle)
            for name, key, value, message in (
                ("v1", "pipeline_version", 1, "pipeline_version=4"),
                ("v3", "pipeline_version", 3, "pipeline_version=4"),
                ("bvh", "source_format", "bvh_lafan1", "legacy BVH"),
            ):
                payload = dict(valid)
                payload[key] = value
                case_path = root / f"{name}.pkl"
                with case_path.open("wb") as handle:
                    pickle.dump(payload, handle)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    GmrUnitreeG1MotionSampler(case_path, None, 1.0, 0.0, False)

    def test_final_gmr_velocity_limiter_bounds_every_qpos_step(self) -> None:
        qpos = np.zeros((3, 36), dtype=np.float64)
        qpos[:, 3] = 1.0
        qpos[1, :3] = [2.0, 0.0, 0.0]
        qpos[1, 3:7] = [0.0, 0.0, 0.0, 1.0]
        qpos[1, 7:] = 2.0
        qpos[2] = qpos[1]
        limited, counts = limit_qpos_velocity(
            qpos,
            10.0,
            max_joint_speed=1.0,
            max_root_speed=2.0,
            max_root_angular_speed=3.0,
        )
        self.assertLessEqual(float(np.max(np.abs(np.diff(limited[:, 7:], axis=0)))), 0.1 + 1e-12)
        self.assertLessEqual(
            float(np.max(np.linalg.norm(np.diff(limited[:, :3], axis=0), axis=1))),
            0.2 + 1e-12,
        )
        self.assertGreater(counts["joints"], 0)

    def test_velocity_finalization_does_not_pull_authored_tail_to_first_frame(self) -> None:
        qpos = np.zeros((31, 36), dtype=np.float64)
        qpos[:, 3] = 1.0
        qpos[:, 7] = np.linspace(0.0, 0.25, len(qpos))
        finalized, counts = limit_qpos_velocity(
            qpos,
            60.0,
            max_joint_speed=3.0 * np.pi,
            max_root_speed=3.0,
            max_root_angular_speed=4.0 * np.pi,
        )
        np.testing.assert_allclose(finalized, qpos)
        self.assertAlmostEqual(0.25, finalized[-1, 7])
        self.assertNotEqual(finalized[0, 7], finalized[-1, 7])
        self.assertEqual({"root_position": 0, "root_rotation": 0, "joints": 0}, counts)

    def test_loop_closure_preserves_xy_path_and_closes_joint_z_tilt_seams(self) -> None:
        qpos = np.zeros((5, 36), dtype=np.float64)
        qpos[:, 3] = 1.0
        qpos[:, 0] = np.linspace(0.0, 2.0, 5)
        qpos[-1, 1] = 3.0
        qpos[-1, 2] = 1.0
        qpos[-1, 3:7] = Rotation.from_euler("xyz", [0.4, 0.2, 0.7]).as_quat()[[3, 0, 1, 2]]
        qpos[-1, 7:] = 2.0
        closed, frame_count = apply_loop_closure(qpos, fps=4.0, duration=0.5)
        self.assertEqual(2, frame_count)
        np.testing.assert_allclose(closed[:, :2], qpos[:, :2])
        self.assertAlmostEqual(closed[0, 2], closed[-1, 2])
        np.testing.assert_allclose(closed[0, 7:], closed[-1, 7:])
        final_rotation = Rotation.from_quat(closed[-1, 3:7][[1, 2, 3, 0]])
        roll, pitch, _yaw = final_rotation.as_euler("xyz")
        self.assertAlmostEqual(0.0, roll, places=7)
        self.assertAlmostEqual(0.0, pitch, places=7)

    def test_gmr_loader_requires_exact_explicit_canonical_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "motion.pkl"
            self._write_gmr(path)
            with path.open("rb") as handle:
                valid = pickle.load(handle)

            cases = []
            wrong_order = dict(valid)
            wrong_order["dof_names"] = list(reversed(valid["dof_names"]))
            cases.append((wrong_order, "qpos address order"))

            missing_field = dict(valid)
            missing_field.pop("root_pos")
            cases.append((missing_field, "missing required fields"))

            missing_names = dict(valid)
            missing_names.pop("dof_names")
            cases.append((missing_names, "missing required fields"))

            duplicate = dict(valid)
            duplicate["dof_names"] = list(valid["dof_names"])
            duplicate["dof_names"][0] = duplicate["dof_names"][1]
            cases.append((duplicate, "duplicate"))

            unknown = dict(valid)
            unknown["dof_names"] = list(valid["dof_names"])
            unknown["dof_names"][0] = "mystery_joint"
            cases.append((unknown, "unknown"))

            extra = dict(valid)
            extra["dof_names"] = [*valid["dof_names"], "mystery_joint"]
            extra["dof_pos"] = np.pad(valid["dof_pos"], ((0, 0), (0, 1)))
            cases.append((extra, "exactly 29"))

            for index, (payload, message) in enumerate(cases):
                case_path = root / f"invalid_{index}.pkl"
                with case_path.open("wb") as handle:
                    pickle.dump(payload, handle)
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        GmrUnitreeG1MotionSampler(case_path, None, 1.0, 0.0, False)

    def test_gmr_pose_gain_is_ignored_and_v3_metadata_is_traceable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.pkl"
            self._write_gmr(path)
            sampler = GmrUnitreeG1MotionSampler(path, None, 0.25, 1.0, True)
            frame = sampler.sample_frame(0.0, 0.1, 1.0, FeatureState())
            self.assertAlmostEqual(0.0, frame.joint_positions["left_hip_pitch"])

            self.assertEqual(4, sampler.pipeline_version)
            self.assertEqual("aistpp_smpl_direct", sampler.source_format)
            self.assertEqual("GMR", sampler.retargeter)
            self.assertTrue(sampler.collision_avoidance["enabled"])

    def test_matcher_prefers_same_named_gmr_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gmr_path = root / "clip.pkl"
            self._write_gmr(gmr_path)
            args = Namespace(
                gmr_motion_root=root,
                retarget_policy="prefer-gmr",
                gmr_fps=None,
                pose_gain=1.0,
                accent_gain=0.0,
                aistpp_fps=60.0,
            )
            catalog = SimpleNamespace(metadata={"aistpp_root": str(root)})
            profile = SimpleNamespace(motion_id="clip", motion_path="missing.pkl")
            sampler = load_motion_sampler(args, catalog, profile)
            self.assertIsInstance(sampler, GmrUnitreeG1MotionSampler)

    def test_continuous_root_reanchors_without_wrap_teleport(self) -> None:
        continuity = RootMotionContinuity(
            np.asarray([0.0, 0.0, 0.8]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            mode="continuous",
        )
        identity = np.asarray([1.0, 0.0, 0.0, 0.0])
        first = continuity.apply(
            RobotMotionFrame({}, np.asarray([0.0, 0.0, 0.0]), identity),
            phase=0.9,
            source_id="clip",
        )
        end = continuity.apply(
            RobotMotionFrame({}, np.asarray([1.0, 0.0, 0.0]), identity),
            phase=0.99,
            source_id="clip",
        )
        wrapped = continuity.apply(
            RobotMotionFrame({}, np.asarray([0.1, 0.0, 0.0]), identity),
            phase=0.01,
            source_id="clip",
        )
        np.testing.assert_allclose([0.0, 0.0, 0.8], first.root_position)
        np.testing.assert_allclose(end.root_position, wrapped.root_position, atol=1e-8)

    def test_transition_root_alignment_matches_first_target(self) -> None:
        identity = np.asarray([1.0, 0.0, 0.0, 0.0])
        source_reference = RobotMotionFrame({}, np.asarray([8.0, -3.0, 0.2]), identity)
        target_reference = RobotMotionFrame({}, np.asarray([1.0, 2.0, 0.8]), identity)
        aligned_reference = align_motion_frame_root(
            source_reference,
            source_reference=source_reference,
            target_reference=target_reference,
        )
        np.testing.assert_allclose([1.0, 2.0, 0.2], aligned_reference.root_position)
        later = align_motion_frame_root(
            RobotMotionFrame({}, np.asarray([9.0, -3.0, 0.2]), identity),
            source_reference=source_reference,
            target_reference=target_reference,
        )
        np.testing.assert_allclose([2.0, 2.0, 0.2], later.root_position)

    def test_continuous_root_accumulates_only_xy_and_yaw(self) -> None:
        identity = np.asarray([1.0, 0.0, 0.0, 0.0])
        continuity = RootMotionContinuity(np.asarray([0.0, 0.0, 0.8]), identity, mode="continuous")
        continuity.apply(RobotMotionFrame({}, np.zeros(3), identity), phase=0.0, source_id="clip")
        tilted = Rotation.from_euler("z", 0.5) * Rotation.from_euler("x", 0.3)
        x, y, z, w = tilted.as_quat()
        continuity.apply(
            RobotMotionFrame({}, np.asarray([1.0, 0.0, 0.2]), np.asarray([w, x, y, z])),
            phase=0.99,
            source_id="clip",
        )
        wrapped = continuity.apply(
            RobotMotionFrame({}, np.zeros(3), identity),
            phase=0.0,
            source_id="clip",
        )
        np.testing.assert_allclose([1.0, 0.0, 0.8], wrapped.root_position, atol=1e-8)
        wrapped_rotation = Rotation.from_quat(
            [wrapped.root_quaternion_wxyz[1], wrapped.root_quaternion_wxyz[2], wrapped.root_quaternion_wxyz[3], wrapped.root_quaternion_wxyz[0]]
        )
        yaw, pitch, roll = wrapped_rotation.as_euler("zyx")
        self.assertAlmostEqual(0.5, yaw, places=6)
        self.assertAlmostEqual(0.0, pitch, places=6)
        self.assertAlmostEqual(0.0, roll, places=6)

    def test_in_place_and_reset_root_modes_have_distinct_wrap_semantics(self) -> None:
        identity = np.asarray([1.0, 0.0, 0.0, 0.0])
        moved_rotation = Rotation.from_euler("z", 0.4)
        x, y, z, w = moved_rotation.as_quat()
        moved = RobotMotionFrame({}, np.asarray([2.0, 3.0, 0.2]), np.asarray([w, x, y, z]))

        in_place = RootMotionContinuity(np.asarray([3.0, 4.0, 0.8]), identity, mode="in-place")
        in_place.apply(RobotMotionFrame({}, np.zeros(3), identity), phase=0.0, source_id="clip")
        held = in_place.apply(moved, phase=0.9, source_id="clip")
        np.testing.assert_allclose([3.0, 4.0, 1.0], held.root_position, atol=1e-8)

        reset = RootMotionContinuity(np.asarray([3.0, 4.0, 0.8]), identity, mode="reset")
        reset.apply(RobotMotionFrame({}, np.zeros(3), identity), phase=0.0, source_id="clip")
        reset.apply(moved, phase=0.9, source_id="clip")
        restarted = reset.apply(RobotMotionFrame({}, np.zeros(3), identity), phase=0.0, source_id="clip")
        np.testing.assert_allclose([3.0, 4.0, 0.8], restarted.root_position, atol=1e-8)
        np.testing.assert_allclose(identity, restarted.root_quaternion_wxyz, atol=1e-8)

    def test_motion_frame_blend_preserves_both_endpoints(self) -> None:
        first = RobotMotionFrame({"left_elbow": 0.2}, np.asarray([1.0, 2.0, 0.8]), np.asarray([1.0, 0.0, 0.0, 0.0]))
        second_rotation = Rotation.from_euler("z", 0.7)
        x, y, z, w = second_rotation.as_quat()
        second = RobotMotionFrame({"left_elbow": 1.2}, np.asarray([4.0, 3.0, 1.0]), np.asarray([w, x, y, z]))
        start = blend_motion_frames(first, second, 0.0)
        end = blend_motion_frames(first, second, 1.0)
        self.assertEqual(first.joint_positions, start.joint_positions)
        self.assertEqual(second.joint_positions, end.joint_positions)
        np.testing.assert_allclose(first.root_position, start.root_position)
        np.testing.assert_allclose(second.root_position, end.root_position)
        np.testing.assert_allclose(first.root_quaternion_wxyz, start.root_quaternion_wxyz)
        np.testing.assert_allclose(second.root_quaternion_wxyz, end.root_quaternion_wxyz)

    def test_root_source_change_preserves_current_roll_and_pitch(self) -> None:
        identity = np.asarray([1.0, 0.0, 0.0, 0.0])
        continuity = RootMotionContinuity(np.asarray([0.0, 0.0, 0.8]), identity, mode="continuous")
        continuity.apply(RobotMotionFrame({}, np.zeros(3), identity), phase=0.0, source_id="old")
        tilted = Rotation.from_euler("xyz", [0.2, -0.15, 0.4])
        x, y, z, w = tilted.as_quat()
        previous = continuity.apply(
            RobotMotionFrame({}, np.asarray([0.5, 0.1, 0.2]), np.asarray([w, x, y, z])),
            phase=0.8,
            source_id="old",
        )
        switched = continuity.apply(
            RobotMotionFrame({}, np.asarray([9.0, -2.0, 0.7]), np.asarray([1.0, 0.0, 0.0, 0.0])),
            phase=0.1,
            source_id="new",
        )
        np.testing.assert_allclose(previous.root_position[:2], switched.root_position[:2], atol=1e-8)
        np.testing.assert_allclose(previous.root_quaternion_wxyz, switched.root_quaternion_wxyz, atol=1e-8)

    def test_subtle_modulation_preserves_base_amplitude_and_bounds(self) -> None:
        pose = {
            "left_shoulder_yaw": 1.2,
            "right_shoulder_yaw": -1.1,
            "left_wrist_roll": 0.7,
            "right_wrist_roll": -0.6,
            "left_knee": 0.5,
            "right_knee": 0.5,
        }
        features = FeatureState(low_energy=1.0, mid_energy=1.0, high_energy=1.0, rhythm_density=1.0)
        result = MusicPoseModulator(1.0, mode="subtle").modulate(
            pose,
            features,
            phase=0.125,
            amplitude=0.1,
            accent=1.0,
        )
        self.assertLessEqual(abs(result["left_shoulder_yaw"] - pose["left_shoulder_yaw"]), 0.03 + 1e-9)
        self.assertLessEqual(abs(result["left_wrist_roll"] - pose["left_wrist_roll"]), 0.04 + 1e-9)
        self.assertLessEqual(abs(result["left_knee"] - pose["left_knee"]), 0.02 + 1e-9)
        self.assertGreater(result["left_shoulder_yaw"], 1.0)

    def test_mujoco_player_writes_free_root_and_joint_qpos_not_torque(self) -> None:
        model = ROOT / "realtime" / "humanoid_robot" / "assets" / "open_humanoid_dancer.xml"
        player = MujocoHumanoidPlayer(model, realtime=False, headless=True)
        joints = {
            name: 0.01 * (index + 1)
            for index, name in enumerate(UnitreeG1DanceAdapter.GMR_DOF_NAMES)
        }
        joints["left_shoulder_roll"] = 0.25
        frame = RobotMotionFrame(
            joints,
            np.asarray([0.2, -0.1, 0.9]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
        player.set_frame(frame)
        start = player.floating_base_qpos_id
        assert start is not None
        np.testing.assert_allclose([0.2, -0.1, 0.9], player.data.qpos[start : start + 3])
        qpos_id = player.actuator_joint_qpos_ids["left_shoulder_roll"]
        self.assertAlmostEqual(0.25, player.data.qpos[qpos_id])
        for name, target in joints.items():
            self.assertAlmostEqual(target, player.data.qpos[player.actuator_joint_qpos_ids[name]])
        np.testing.assert_allclose(0.0, player.data.ctrl)

    def test_viewer_uses_pelvis_tracking_camera(self) -> None:
        model = ROOT / "realtime" / "humanoid_robot" / "assets" / "open_humanoid_dancer.xml"
        player = MujocoHumanoidPlayer(model, realtime=False, headless=False)
        viewer = SimpleNamespace(
            cam=SimpleNamespace(
                distance=0.0,
                azimuth=0.0,
                elevation=0.0,
                type=None,
                trackbodyid=-1,
                fixedcamid=0,
            )
        )
        with patch(
            "realtime_music_humanoid_dancer.mujoco.viewer.launch_passive",
            return_value=viewer,
        ):
            player.start()
        pelvis_id = mujoco.mj_name2id(player.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.assertEqual(int(mujoco.mjtCamera.mjCAMERA_TRACKING), int(viewer.cam.type))
        self.assertEqual(pelvis_id, viewer.cam.trackbodyid)
        self.assertEqual(-1, viewer.cam.fixedcamid)

    def test_whole_clip_grounding_preserves_root_vertical_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.pkl"
            self._write_aistpp(path)
            sampler = AistppMotionSampler(path, 60.0, 1.0, 0.0)
            model = ROOT / "realtime" / "humanoid_robot" / "assets" / "open_humanoid_dancer.xml"
            player = MujocoHumanoidPlayer(model, realtime=False, headless=True)
            joint_adapter = UnitreeG1JointPoseAdapter(player.actuator_names)
            before_range = float(np.ptp(sampler.root_positions[:, 2]))
            player.ground_sampler(sampler, joint_adapter)
            minimum = math.inf
            for index in range(len(sampler.frames)):
                frame = sampler.sample_frame(index / len(sampler.frames), 1.0, 0.0, FeatureState())
                frame = frame.with_joint_positions(joint_adapter.adapt_pose(frame.joint_positions))
                player.set_frame(frame)
                mujoco.mj_forward(player.model, player.data)
                minimum = min(minimum, player.support_height())
            self.assertAlmostEqual(0.0, minimum, places=5)
            grounded = sampler.root_positions[:, 2] + sampler.ground_offset_z
            self.assertAlmostEqual(before_range, float(np.ptp(grounded)), places=10)


if __name__ == "__main__":
    unittest.main()
