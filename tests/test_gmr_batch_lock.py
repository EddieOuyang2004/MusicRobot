from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1]/"realtime/humanoid_robot/src"
sys.path.insert(0, str(SRC))
from gmr_batch_lock import BatchAlreadyRunning, batch_lock, process_alive


class BatchLockTests(unittest.TestCase):
    def test_current_process_is_alive(self):
        self.assertTrue(process_alive(os.getpid()))

    def test_recovers_dead_legacy_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'.batch.lock'
            path.write_text('27052')
            with patch('gmr_batch_lock.process_alive', return_value=False), batch_lock(path):
                self.assertTrue(path.exists())  # Windows byte locks also exclude other readers.
            self.assertEqual(json.loads(path.read_text())['status'], 'idle')
            with batch_lock(path):
                pass

    def test_live_legacy_owner_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'.batch.lock'
            path.write_text(str(os.getpid()))
            with self.assertRaises(BatchAlreadyRunning), batch_lock(path):
                self.fail('Acquired live legacy lock')
            self.assertEqual(path.read_text(), str(os.getpid()))

    def test_exception_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'.batch.lock'
            with self.assertRaisesRegex(RuntimeError, 'test'), batch_lock(path):
                raise RuntimeError('test')
            with batch_lock(path):
                pass

    def test_active_process_blocks_and_crash_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'.batch.lock'
            code = ('import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); '
                    'from gmr_batch_lock import batch_lock\n'
                    'with batch_lock(Path(sys.argv[2])):\n'
                    ' print("locked",flush=True)\n'
                    ' sys.stdin.read()\n')
            child = subprocess.Popen([sys.executable, '-c', code, str(SRC), str(path)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), 'locked')
                with self.assertRaises(BatchAlreadyRunning), batch_lock(path):
                    self.fail('Acquired active lock')
                child.kill()  # Deliberately skip cleanup, simulating the reported interruption.
                child.communicate(timeout=10)
                with batch_lock(path):
                    pass
            finally:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=10)


if __name__ == '__main__':
    unittest.main()
