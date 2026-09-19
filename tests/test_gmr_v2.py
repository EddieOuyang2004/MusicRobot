"""Regression checks for projected motion and separate, resumable v2 artifacts."""
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
import importlib.util

import mujoco
import numpy as np
try:
    from qpsolvers import solve_qp
except ImportError:
    solve_qp = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"realtime/humanoid_robot/src"))
from build_gmr_v2 import cached_payload, check_roots, collect_inputs, fingerprint
from gmr_collision_projection import distance_constraints, projection_settings, validate_v2_metadata, state_trajectory_digest
from unitree_g1_dance_adapter import UnitreeG1DanceAdapter


@unittest.skipIf(solve_qp is None, "Run geometry tests with .venv-gmr")
class ProjectionGeometryTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''
          <mujoco><worldbody><body><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
            <geom name="obstacle" type="sphere" size=".1" pos=".3 0 0"/>
            <body><joint type="slide" axis="1 0 0"/><geom name="moving" type="sphere" size=".1"/></body>
            <body pos="0 2 0"><joint type="slide" axis="0 1 0"/><geom type="sphere" size=".1"/></body>
          </body></worldbody></mujoco>''')
        self.q = self.model.qpos0.copy()
        self.pair = tuple(sorted((self.model.geom('obstacle').id, self.model.geom('moving').id)))
        self.data = mujoco.MjData(self.model)
        self.states = ((self.model, self.data, {self.pair}, np.array([self.pair])),)

    def test_gradient_matches_finite_difference(self):
        G, _ = distance_constraints(self.states, self.q, .05, detection=.2)
        initial = mujoco.mj_geomDistance(self.model, self.data, *self.pair, .2, None)
        self.data.qpos[7] += 1e-6
        mujoco.mj_forward(self.model, self.data)
        changed = mujoco.mj_geomDistance(self.model, self.data, *self.pair, .2, None)
        self.assertAlmostEqual(-G[0, 0], (changed-initial)/1e-6, places=6)

    def test_blocked_joint_does_not_freeze_other_joint(self):
        self.q[7] = .05
        G, h = distance_constraints(self.states, self.q, .05, detection=.2)
        step = solve_qp(np.eye(2), -np.array([.1, .1]), G, h, solver='daqp')
        np.testing.assert_allclose(step, [0., .1], atol=1e-7)

    def test_can_move_away_from_contact(self):
        self.q[7] = .05
        G, h = distance_constraints(self.states, self.q, .05, detection=.2)
        step = solve_qp(np.eye(2), -np.array([-.1, .1]), G, h, solver='daqp')
        np.testing.assert_allclose(step, [-.1, .1], atol=1e-7)

    def test_planning_margin_requests_retreat_in_displacement_units(self):
        self.q[7] = .05
        G, h = distance_constraints(self.states, self.q, .06, detection=.2)
        step = solve_qp(np.eye(2), -np.array([.1, .1]), G, h, solver='daqp')
        np.testing.assert_allclose(step, [-.005, .1], atol=1e-7)


class BatchTests(unittest.TestCase):
    def test_protects_original_and_source_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for output in (root/'old', root/'old/nested', root, root/'source'):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    check_roots(root/'old', root/'source', output)
            check_roots(root/'old', root/'source', root/'v2')

    def test_settings_reject_invalid_or_nonfinite_values(self):
        for weight, clearance in ((-1,.008), (float('nan'),.008), (2,.006), (2,float('inf'))):
            with self.subTest(weight=weight, clearance=clearance), self.assertRaises(ValueError):
                projection_settings(weight, clearance)

    def test_selection_is_deduplicated_and_checks_ids(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'one.pkl').touch()
            args = Namespace(input_root=root, motion_id=['one', 'one'], limit=None)
            self.assertEqual(collect_inputs(args), [root/'one.pkl'])
            args.motion_id = ['../unknown']
            with self.assertRaises(ValueError):
                collect_inputs(args)

    def payload(self):
        payload = dict(format_version=1, pipeline_version=5, motion_version='gmr_v2',
                    source_format='aistpp_smpl_direct', fps=60.,
                    root_pos=np.zeros((2,3)), root_rot=np.tile([1.,0.,0.,0.], (2,1)),
                    root_rot_order='wxyz', dof_pos=np.zeros((2,29)),
                    dof_names=list(UnitreeG1DanceAdapter.GMR_DOF_NAMES), source_motion_id='one',
                    source_sha256='s'*64, smpl_model_sha256='m'*64, retargeter='GMR',
                    retargeter_version='a'*40, mink_limits_api='test',
                    collision_avoidance=dict(enabled=True,preset='g1_self_collision_v2'),
                    continuity_limits=dict(loop_closure_frames=0), projection=projection_settings(),
                    projection_validation=dict(passed=True,clearance_m=.005,violating_samples=0,
                                               samples_per_interval=81,
                                               interpolation="quintic_hermite_states",
                                               continuous_peaks=dict(speed_rad_s=0., acceleration_rad_s2=0.)))
        payload['dof_vel'] = np.zeros((2,29))
        payload['dof_acc'] = np.zeros((2,29))
        payload['state_trajectory_sha256'] = state_trajectory_digest(payload)
        return payload

    def test_lock_only_cache_compatibility_is_exact(self):
        previous = dict(source_sha256='s'*64, smpl_model_sha256='m'*64,
                        gmr_commit='a'*40, fps=60., projection=projection_settings(),
                        implementation_sha256='old-lock-code')
        current = dict(previous, implementation_sha256='new-lock-code')
        payload = self.payload()
        payload.update(v2_build=previous, v2_build_fingerprint=fingerprint(previous))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'one.pkl'
            path.write_bytes(pickle.dumps(payload))
            cached_payload(path, current, compatible_implementation_sha256='old-lock-code')
            with self.assertRaisesRegex(ValueError, 'changed'):
                cached_payload(path, current, compatible_implementation_sha256='other-code')
            with self.assertRaisesRegex(ValueError, 'changed'):
                cached_payload(path, dict(current, projection=projection_settings(3.)),
                               compatible_implementation_sha256='old-lock-code')

    def test_resume_rejects_changed_settings_and_legacy_files(self):
        identity = dict(source_sha256='s'*64,smpl_model_sha256='m'*64,gmr_commit='a'*40,
                        fps=60., projection=projection_settings())
        payload = self.payload()
        payload['v2_build_fingerprint'] = fingerprint(identity)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'one.pkl'
            path.write_bytes(pickle.dumps(payload))
            cached_payload(path, identity)
            changed = dict(identity, projection=projection_settings(3.))
            with self.assertRaisesRegex(ValueError, 'changed'):
                cached_payload(path, changed)
            payload['pipeline_version'] = 4
            path.write_bytes(pickle.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'pipeline_version'):
                cached_payload(path, identity)

    @unittest.skipUnless(importlib.util.find_spec("librosa"), "Run playback test with project .venv")
    def test_runtime_accepts_validated_v2_and_rejects_missing_validation(self):
        from realtime_music_humanoid_dancer import GmrUnitreeG1MotionSampler
        payload = self.payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'one.pkl'
            path.write_bytes(pickle.dumps(payload))
            sampler = GmrUnitreeG1MotionSampler(path, None, 1., 0., False)
            self.assertEqual(sampler.pipeline_version, 5)
            payload.pop('projection_validation')
            path.write_bytes(pickle.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'validation'):
                GmrUnitreeG1MotionSampler(path, None, 1., 0., False)

    def test_rejects_old_linear_revision_and_tampered_states(self):
        payload = self.payload()
        payload['projection']['revision'] = 1
        with self.assertRaisesRegex(ValueError, 'regenerate'):
            validate_v2_metadata(payload)
        payload = self.payload()
        payload['dof_vel'][0,0] = 1.
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_v2_metadata(payload)

    @unittest.skipUnless(importlib.util.find_spec("librosa"), "Run playback test with project .venv")
    def test_sampler_preserves_saved_derivatives_and_refuses_fps_override(self):
        from realtime_music_humanoid_dancer import GmrUnitreeG1MotionSampler, FeatureState
        from realtime_music_humanoid_matcher_v2 import build_motion_entry_features
        from unitree_g1_dance_adapter import UnitreeG1JointPoseAdapter
        from types import SimpleNamespace
        payload = self.payload()
        payload['dof_pos'][1] = .01
        payload['dof_vel'][:] = .6
        payload['state_trajectory_sha256'] = state_trajectory_digest(payload)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'one.pkl'
            path.write_bytes(pickle.dumps(payload))
            sampler = GmrUnitreeG1MotionSampler(path,None,1.,0.,False)
            state = sampler.authored_trajectory.at_time(.5/60)
            frame = sampler.sample_frame(.25,1.,0.,FeatureState())
            np.testing.assert_allclose([frame.joint_positions[n] for n in sampler.dof_names],state.position,atol=1e-14)
            adapter = UnitreeG1JointPoseAdapter()
            features = build_motion_entry_features(sampler,SimpleNamespace(keypoint_phases=(),keypoint_scores=()),adapter,{})
            np.testing.assert_allclose(features.authored.velocities,.6,atol=1e-14)
            with self.assertRaisesRegex(ValueError, 'FPS'):
                GmrUnitreeG1MotionSampler(path,30.,1.,0.,False)

    def test_failed_clearance_cannot_claim_validated_v2(self):
        payload = self.payload()
        payload['projection_validation']['violating_samples'] = 1
        with self.assertRaises(ValueError):
            validate_v2_metadata(payload)


if __name__ == '__main__':
    unittest.main()
