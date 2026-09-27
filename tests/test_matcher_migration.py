"""Compatibility boundaries for the single authored-transition runtime."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "realtime/humanoid_robot/src"
sys.path.insert(0, str(SRC / "test"))
sys.path.insert(0, str(SRC))

import run_humanoid_matcher_experiments as runner
import consolidate_thesis_experiments as consolidator


class MatcherMigrationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(runner.DEFAULT_PROTOCOL.read_text(encoding="utf-8"))

    @staticmethod
    def matrix(root):
        return [dict(condition="full", condition_arguments=[], seed=0,
                     source_id="fixture", audio=root / "audio.wav")]

    def test_humanoid_imports_without_arm_or_pybullet_in_fresh_process(self):
        code = '''import importlib.abc, sys
class BlockArm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"pybullet", "pybullet_data", "realtime_music_adaptive_player"}:
            raise AssertionError("Humanoid tried to import " + fullname)
sys.meta_path.insert(0, BlockArm())
sys.path.insert(0, sys.argv[1])
import realtime_music_humanoid_matcher as matcher
assert matcher.base.MusicFrame.__module__ == "music_runtime"
assert matcher.base.AdaptiveMotionController.__module__ == "music_runtime"
assert matcher.DEFAULT_CATALOG.parent.name == "music_catalog_combined"
'''
        result = subprocess.run([sys.executable, "-c", code, str(SRC)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_protocol_conditions_and_historical_readers(self):
        self.assertEqual(2, self.protocol["protocol_version"])
        self.assertEqual(runner.SUPPORTED_CONDITIONS, set(self.protocol["ablations"]))
        for retired in runner.RETIRED_CONDITIONS:
            old = copy.deepcopy(self.protocol)
            old["ablations"][retired] = []
            with self.subTest(retired=retired), self.assertRaisesRegex(ValueError, "retired v1"):
                runner.run_matrix("ablation", None, old, ROOT)
        historical = {**self.protocol, "protocol_version": 1}
        old_report = runner.ablation_report([], historical, {})
        current_report = runner.ablation_report([], self.protocol, {})
        self.assertIn("full_recall_exceeds_legacy", old_report["quality_checks"])
        self.assertNotIn("full_recall_exceeds_legacy", current_report["quality_checks"])
        self.assertEqual(520, consolidator.expected_runs(historical)["ablation"])
        self.assertEqual(260, consolidator.expected_runs(self.protocol)["ablation"])

    def test_commands_use_growing_history_and_changed_runtime_cannot_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = root / "catalog.json"
            catalog.write_text("{}", encoding="utf-8")
            output = root / "output"
            kwargs = dict(max_seconds=1, dry_run=True, resume=False,
                          suite="smoke", max_attempts=1, protocol=self.protocol)
            records, failures = runner.execute_runs(self.matrix(root), output, catalog, **kwargs)
            self.assertFalse(failures)
            command = records[0]["command"]
            self.assertIn("realtime_music_humanoid_matcher.py", command)
            self.assertIn("--history-max-seconds 30.0", command)
            self.assertIn("--analysis-min-seconds 2.0", command)
            self.assertNotIn("--match-window-seconds", command)
            status = output / "run_status.json"
            original = status.read_bytes()
            old_hash = runner.file_sha256
            def changed_shared_runtime(path):
                return "changed" if path.name == "music_runtime.py" else old_hash(path)
            with patch.object(runner, "file_sha256", side_effect=changed_shared_runtime):
                with self.assertRaisesRegex(RuntimeError, "new output directory"):
                    runner.execute_runs(self.matrix(root), output, catalog, **kwargs)
            self.assertEqual(original, status.read_bytes())

    def test_unversioned_artifacts_are_preserved_and_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runs = root / "runs"
            runs.mkdir()
            evidence = runs / "old.csv"
            evidence.write_bytes(b"historical research evidence")
            with self.assertRaisesRegex(RuntimeError, "no runtime/protocol provenance"):
                runner.execute_runs(self.matrix(root), root, root / "catalog.json",
                                    max_seconds=1, dry_run=False, resume=True,
                                    suite="smoke", max_attempts=1)
            self.assertEqual(b"historical research evidence", evidence.read_bytes())

    def test_cli_rejects_historical_output_before_preparing_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "experiment_report.json"
            old.write_bytes(b"historical report")
            args = SimpleNamespace(catalog=root / "catalog.json", protocol=runner.DEFAULT_PROTOCOL,
                                   output_dir=root, execute_suite="smoke")
            with patch.object(runner, "parse_args", return_value=args), \
                 patch.object(runner, "prepare_short_audio_inputs") as prepare, \
                 patch.object(runner.MusicCatalog, "load") as load:
                with self.assertRaisesRegex(RuntimeError, "new output directory"):
                    runner.main()
            prepare.assert_not_called()
            load.assert_not_called()
            self.assertEqual(b"historical report", old.read_bytes())


if __name__ == "__main__":
    unittest.main()
