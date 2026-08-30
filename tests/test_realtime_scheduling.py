from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "realtime" / "humanoid_robot" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import realtime_music_humanoid_dancer as dancer
from robot_motion import RobotMotionFrame


class RealtimeLoopSchedulerTests(unittest.TestCase):
    def test_sleeps_only_until_the_absolute_deadline(self) -> None:
        scheduler = dancer.RealtimeLoopScheduler(100.0, enabled=True)
        with (
            patch.object(dancer.time, "perf_counter", return_value=10.004),
            patch.object(dancer.time, "sleep") as sleep,
        ):
            scheduler.wait(10.0)
        sleep.assert_called_once()
        self.assertAlmostEqual(0.006, sleep.call_args.args[0])
        self.assertEqual(0, scheduler.deadline_misses)

    def test_late_frame_skips_deadlines_without_sleeping_or_catching_up(self) -> None:
        scheduler = dancer.RealtimeLoopScheduler(100.0, enabled=True)
        scheduler.next_deadline = 10.0
        with (
            patch.object(dancer.time, "perf_counter", return_value=10.035),
            patch.object(dancer.time, "sleep") as sleep,
        ):
            scheduler.wait(10.02)
        sleep.assert_not_called()
        self.assertEqual(1, scheduler.deadline_misses)
        self.assertAlmostEqual(10.04, scheduler.next_deadline)


class ViewerDecouplingTests(unittest.TestCase):
    def test_control_step_publishes_without_syncing_viewer(self) -> None:
        player = object.__new__(dancer.MujocoHumanoidPlayer)
        player.model = object()
        player.data = SimpleNamespace(time=0.0)
        player.viewer = Mock()
        player._publish_viewer_snapshot = Mock()
        with patch.object(dancer.mujoco, "mj_forward"):
            dancer.MujocoHumanoidPlayer.step(player, 1.0 / 120.0)
        player._publish_viewer_snapshot.assert_called_once_with()
        player.viewer.sync.assert_not_called()

    def test_viewer_snapshot_is_copied(self) -> None:
        player = object.__new__(dancer.MujocoHumanoidPlayer)
        player.viewer = Mock()
        player.data = SimpleNamespace(qpos=np.asarray([1.0, 2.0]), time=3.0)
        player.viewer_snapshot_lock = threading.Lock()
        player.viewer_qpos_snapshot = None
        player.viewer_time_snapshot = 0.0
        player._publish_viewer_snapshot()
        np.testing.assert_array_equal(np.asarray([1.0, 2.0]), player.viewer_qpos_snapshot)
        self.assertEqual(3.0, player.viewer_time_snapshot)


class CollisionPolicyTests(unittest.TestCase):
    def test_off_and_unmodified_auto_skip_geometry_checks(self) -> None:
        player = object.__new__(dancer.MujocoHumanoidPlayer)
        frame = RobotMotionFrame({})
        with patch.object(
            dancer.MujocoHumanoidPlayer,
            "project_self_collision_safe",
            return_value=frame,
        ) as project:
            self.assertIs(frame, player.apply_collision_policy(frame, "off", runtime_modified=True))
            self.assertIs(frame, player.apply_collision_policy(frame, "auto", runtime_modified=False))
        project.assert_not_called()

    def test_modified_auto_and_always_run_geometry_checks(self) -> None:
        player = object.__new__(dancer.MujocoHumanoidPlayer)
        frame = RobotMotionFrame({})
        with patch.object(
            dancer.MujocoHumanoidPlayer,
            "project_self_collision_safe",
            return_value=frame,
        ) as project:
            player.apply_collision_policy(frame, "auto", runtime_modified=True)
            player.apply_collision_policy(frame, "always", runtime_modified=False)
        self.assertEqual(2, project.call_count)


if __name__ == "__main__":
    unittest.main()
