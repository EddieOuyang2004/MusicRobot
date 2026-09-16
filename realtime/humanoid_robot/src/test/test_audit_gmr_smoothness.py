"""Synthetic checks for audit units, quaternion signs, and spike detection."""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from audit_gmr_smoothness import audit, derivatives, exact_peaks, UnitreeG1DanceAdapter
from motion_bridges import HermiteBridge, JointState, coefficients


class SmoothnessAuditTests(unittest.TestCase):
    def test_derivative_units(self):
        t = np.arange(20)/60.
        v, a, j = derivatives((3*t*t)[:, None], 60.)
        np.testing.assert_allclose(a, 6., atol=1e-10)
        np.testing.assert_allclose(j, 0., atol=1e-9)

    def test_closed_form_extrema_match_root_finding(self):
        rng = np.random.default_rng(7)
        fps, joints = 60., 8
        states = [JointState(*(rng.normal(size=(2, joints))*scale
                               for scale in (1., 12., 900.))) for _ in range(2)]
        segments = coefficients(states[0], states[1], 1/fps)
        peaks = exact_peaks(segments, fps)
        for derivative, key in ((1, 'speed_rad_s'), (2, 'acceleration_rad_s2'), (3, 'jerk_rad_s3')):
            for segment in range(2):
                bridge = HermiteBridge(1/fps, segments[:, segment])
                low, high = bridge.extrema(derivative)
                expected = np.maximum(np.abs(low), np.abs(high))
                np.testing.assert_allclose(peaks[key][0][segment], expected, rtol=1e-7)

    def check_motion(self, q):
        names = UnitreeG1DanceAdapter.GMR_DOF_NAMES
        quat = np.tile([1., 0., 0., 0.], (len(q), 1))
        quat[::2] *= -1
        payload = dict(dof_pos=q, root_pos=np.zeros((len(q), 3)), root_rot=quat,
                       fps=60., dof_names=names, root_rot_order='wxyz',
                       continuity_limits=dict(max_joint_speed_rad_s=10.,
                                              max_root_speed_m_s=3.,
                                              max_root_angular_speed_rad_s=12.))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'synthetic.pkl'
            path.write_bytes(pickle.dumps(payload))
            return audit(path, {n: (-3., 3.) for n in names})

    def test_constant_and_quaternion_sign(self):
        row = self.check_motion(np.zeros((20, 29)))
        for key in ('raw_speed_rad_s', 'v2_acceleration_rad_s2', 'v2_jerk_rad_s3',
                    'root_angular_speed_rad_s', 'v2_position_excess_rad'):
            self.assertEqual(row[key], 0.)

    def test_velocity_limited_reversal_is_not_smooth(self):
        q = np.zeros((20, 29))
        q[1::2, 0] = .15
        row = self.check_motion(q)
        self.assertEqual(row['raw_speed_violating_intervals'], 0)
        self.assertGreater(row['v2_acceleration_bad_segments'], 0)
        self.assertGreater(row['v2_jerk_jump_rad_s3'], 0)
        self.assertGreater(row['fast_reversal_frames'], 0)
        self.assertEqual(row['export_speed_clamp_fraction'], 0.)


if __name__ == '__main__':
    unittest.main()
