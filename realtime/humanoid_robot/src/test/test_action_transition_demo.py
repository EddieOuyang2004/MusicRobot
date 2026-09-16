"""Boundary regressions for the isolated transition demo."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_action_transition import Transition
from robot_motion import RobotMotionFrame


def action(offset, speed):
    def sample(t):
        return RobotMotionFrame({"joint": offset + speed * t},
                                np.array([offset + speed * t, 0., 1.]),
                                np.array([1., 0., 0., 0.]))
    return sample


class TransitionTests(unittest.TestCase):
    def test_overlap_preserves_boundary_pose_and_velocity(self):
        transition = Transition(action(0, 1), action(10, 2), 5, 5, 0, 1, "overlap")
        h = 1e-5
        for boundary in (transition.start, transition.finish):
            frames = [transition.sample(boundary + dt)[0] for dt in (-h, 0, h)]
            q = [f.joint_positions["joint"] for f in frames]
            self.assertAlmostEqual((q[1] - q[0]) / h, (q[2] - q[1]) / h, places=3)
            left = (frames[1].root_position - frames[0].root_position) / h
            right = (frames[2].root_position - frames[1].root_position) / h
            np.testing.assert_allclose(left, right, atol=1e-3)

    def test_bridge_exposes_stop_restart_velocity_change(self):
        transition = Transition(action(0, 1), action(10, 2), 5, 5, 0, 1, "bridge")
        h = 1e-5
        for boundary, expected in ((transition.start, 1), (transition.finish, 2)):
            q = [transition.sample(boundary + dt)[0].joint_positions["joint"] for dt in (-h, 0, h)]
            self.assertAlmostEqual(abs((q[1] - q[0]) / h - (q[2] - q[1]) / h), expected, places=3)

    def test_overlap_continues_target_clock(self):
        transition = Transition(action(0, 1), action(10, 2), 5, 5, .5, 1, "overlap")
        frame, stage, weight = transition.sample(transition.finish + .2)
        self.assertAlmostEqual(frame.joint_positions["joint"], 13.4)
        self.assertEqual(stage, "action_b")
        self.assertEqual(weight, 1)


if __name__ == "__main__":
    unittest.main()
