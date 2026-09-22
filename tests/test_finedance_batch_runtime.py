"""Exercise bounded dispatch and real Windows job lifecycle without GMR load."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realtime/humanoid_robot/src"))
import finedance_batch_runtime as rt

class RuntimeTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(worker_memory_gb=.25, worker_cpu_percent=10., min_free_memory_gb=.1,
                      worker_timeout=10., stop_event=threading.Event())
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_no_unbounded_queue_and_caller_abort_stops_submission(self):
        started = []
        stop = threading.Event()
        def work(clip):
            started.append(clip)
            return clip
        results = rt.bounded_results(range(10000), 2, work, stop, lambda active: None)
        clip, future = next(results)
        self.assertEqual(future.result(), clip)
        results.close()
        self.assertLessEqual(len(started), 2)
        self.assertTrue(stop.is_set())

    def test_heartbeat_error_stops_queue(self):
        stop = threading.Event()
        started = []
        def heartbeat(active):
            raise OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            list(rt.bounded_results(range(100), 1, lambda x: started.append(x), stop, heartbeat))
        self.assertEqual(started, [0])
        self.assertTrue(stop.is_set())

    def test_low_memory_stops_before_launch(self):
        args = self.args()
        with patch.object(rt, "available_memory_gb", return_value=0), patch.object(rt.subprocess, "Popen") as launch:
            with self.assertRaises(rt.BatchStopped):
                rt.run_worker([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, env=os.environ.copy(), args=args)
        launch.assert_not_called()
        self.assertTrue(args.stop_event.is_set())

    @unittest.skipUnless(os.name == "nt", "Windows job containment")
    def test_real_worker_and_job_configuration(self):
        result = rt.run_worker([sys.executable, "-c", "print('ok')"], stdout=subprocess.DEVNULL,
                               env=os.environ.copy(), args=self.args())
        self.assertEqual(result.returncode, 0)

    @unittest.skipUnless(os.name == "nt", "Windows job containment")
    def test_timeout_terminates_worker_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "escaped.txt"
            child = "import time; from pathlib import Path; time.sleep(2); Path(" + repr(str(marker)) + ").write_text('escaped')"
            parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); time.sleep(60)"
            with self.assertRaises(TimeoutError):
                rt.run_worker([sys.executable, "-c", parent], stdout=subprocess.DEVNULL,
                              env=os.environ.copy(), args=self.args(worker_timeout=.7))
            time.sleep(2.2)
            self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "nt", "Windows job containment")
    def test_memory_cap_fails_worker_without_harming_parent(self):
        result = rt.run_worker([sys.executable, "-c", "x=bytearray(512*1024*1024)"],
                              stdout=subprocess.DEVNULL, env=os.environ.copy(), args=self.args())
        self.assertNotEqual(result.returncode, 0)

    def test_cancel_stops_running_worker(self):
        args = self.args()
        timer = threading.Timer(.5, args.stop_event.set)
        timer.start()
        try:
            with self.assertRaises(rt.BatchStopped):
                rt.run_worker([sys.executable, "-c", "import time; time.sleep(60)"], stdout=subprocess.DEVNULL,
                              env=os.environ.copy(), args=args)
        finally:
            timer.cancel()

if __name__ == "__main__":
    unittest.main()
