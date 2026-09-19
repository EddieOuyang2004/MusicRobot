"""Small independent geometry regression tests for the offline prototype."""
import unittest
import mujoco
import numpy as np
from qpsolvers import solve_qp
from experiment import constraints


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''
          <mujoco><worldbody><body><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
            <geom name="obstacle" type="sphere" size=".1" pos=".3 0 0"/>
            <body><joint type="slide" axis="1 0 0"/>
              <geom name="moving" type="sphere" size=".1"/></body>
            <body pos="0 2 0"><joint type="slide" axis="0 1 0"/>
              <geom type="sphere" size=".1"/></body>
          </body></worldbody></mujoco>''')
        self.q = self.model.qpos0.copy()
        self.pair = tuple(sorted((self.model.geom('obstacle').id, self.model.geom('moving').id)))
        self.data = mujoco.MjData(self.model)
        self.states = ((self.model, self.data, {self.pair}, np.array([self.pair])),)

    def test_distance_jacobian_matches_finite_difference(self):
        G, _ = constraints(self.states, self.q, .05, detection=.2)
        self.data.qpos[:] = self.q
        mujoco.mj_forward(self.model, self.data)
        initial = mujoco.mj_geomDistance(self.model, self.data, *self.pair, .2, None)
        self.data.qpos[7] += 1e-6
        mujoco.mj_forward(self.model, self.data)
        changed = mujoco.mj_geomDistance(self.model, self.data, *self.pair, .2, None)
        self.assertAlmostEqual(-G[0, 0], (changed-initial)/1e-6, places=6)

    def test_blocked_joint_does_not_freeze_independent_joint(self):
        self.q[7] = .05  # Exactly at the configured .05 m clearance.
        G, h = constraints(self.states, self.q, .05, detection=.2)
        step = solve_qp(np.eye(2), -np.array([.1, .1]), G, h,
                        lb=np.full(2, -.1), ub=np.full(2, .1), solver='daqp')
        np.testing.assert_allclose(step, [0., .1], atol=1e-7)

    def test_motion_away_from_contact_remains_possible(self):
        self.q[7] = .05
        G, h = constraints(self.states, self.q, .05, detection=.2)
        step = solve_qp(np.eye(2), -np.array([-.1, .1]), G, h, solver='daqp')
        np.testing.assert_allclose(step, [-.1, .1], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
