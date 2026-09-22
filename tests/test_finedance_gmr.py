"""FineDance selection, provenance, publication and resume regression tests."""
import json
import hashlib
import inspect
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realtime/humanoid_robot/src"))
import build_finedance_gmr_dataset as fd
import finedance_batch_state as batch_state
from gmr_collision_projection import projection_settings, state_trajectory_digest
from unitree_g1_dance_adapter import UnitreeG1DanceAdapter


class FineDanceGmrTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.output = self.root / "output"
        (self.source / "motions").mkdir(parents=True)
        (self.source / "audio").mkdir()
        (self.output / "logs").mkdir(parents=True)
        self.manifest = dict(format_version=1, export_complete=True, dry_run=False, fps=60,
                             source_fps=30., root_y_offset_m=1.3, clip_count=3, clips=[])
        for index, split in enumerate(("train", "train", "test"), 1):
            source_id = f"{index:03d}"
            name = f"finedance_{source_id}_0000000_0000480"
            clip = dict(clip_id=name, source_id=source_id, split=split, start_frame=0,
                        end_frame_exclusive=480, frames=480, start_seconds=0., end_seconds=8.,
                        duration_seconds=8., authored_duration_seconds=479 / 60,
                        motion_path=f"motions/{name}.pkl", audio_path=f"audio/{name}.wav",
                        label={"style2": "Jazz"})
            self.manifest["clips"].append(clip)
            payload = dict(smpl_poses=np.zeros((480, 72)), smpl_trans=np.zeros((480, 3)),
                           smpl_scaling=np.ones(1), fps=60, source_dataset="FineDance", source_id=source_id,
                           source_start_seconds=0., source_fps=30., root_y_offset_m=1.3,
                           body_only_approximation=True)
            (self.source / clip["motion_path"]).write_bytes(pickle.dumps(payload))
            (self.source / clip["audio_path"]).touch()
        self.save_manifest()
        for split in fd.SPLITS:
            (self.source / f"{split}.txt").write_text("".join(
                c["clip_id"] + "\n" for c in self.manifest["clips"] if c["split"] == split))
        self.args = fd.parse_args(["--input-root", str(self.source), "--output-root", str(self.output), "--resume"])
        self.common = dict(source_dataset="FineDance", smpl_model_sha256="m" * 64, gmr_commit="a" * 40,
                           projection=projection_settings(), segmentation_manifest_sha256=fd.sha256(self.source / "manifest.json"))

    def save_manifest(self):
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))

    def identity(self, clip):
        return dict(self.common, motion_id=clip["clip_id"], fps=60., clip=clip,
                    source_sha256=fd.sha256(self.source / clip["motion_path"]))

    def payload(self, identity):
        n = identity["clip"]["frames"]
        payload = dict(format_version=1, pipeline_version=5, motion_version="gmr_v2",
                       source_format="aistpp_smpl_direct", fps=60., root_pos=np.zeros((n, 3)),
                       root_rot=np.tile([1., 0., 0., 0.], (n, 1)), root_rot_order="wxyz",
                       dof_pos=np.zeros((n, 29)), dof_vel=np.zeros((n, 29)), dof_acc=np.zeros((n, 29)),
                       dof_names=list(UnitreeG1DanceAdapter.GMR_DOF_NAMES), source_motion_id=identity["motion_id"],
                       source_sha256=identity["source_sha256"], smpl_model_sha256=identity["smpl_model_sha256"],
                       retargeter="GMR", retargeter_version=identity["gmr_commit"], mink_limits_api="test",
                       collision_avoidance=dict(enabled=True, preset="g1_self_collision_v2"),
                       continuity_limits=dict(loop_closure_frames=0, limited_values=dict(solver_failures=0),
                                              max_joint_speed_rad_s=3 * np.pi, max_root_speed_m_s=3.,
                                              max_root_angular_speed_rad_s=4 * np.pi),
                       projection=projection_settings(), generation_seconds=1.,
                       projection_validation=dict(passed=True, clearance_m=.005, violating_samples=0,
                           samples_per_interval=81, interpolation="quintic_hermite_states",
                           continuous_peaks=dict(speed_rad_s=0., acceleration_rad_s2=0.),
                           metrics=dict(all_joint_hold_intervals=n - 1, joint_target_rmse_rad=0.)),
                       v2_build=identity, v2_build_fingerprint=fd.v2.fingerprint(identity))
        payload["state_trajectory_sha256"] = state_trajectory_digest(payload)
        return payload

    def fake_worker(self, command, **kwargs):
        job = fd.read_json(Path(command[-1]))
        Path(job["output"]).write_bytes(pickle.dumps(self.payload(job["identity"])))
        self.assertFalse((self.output / f"{job['identity']['motion_id']}.pkl").exists())
        return SimpleNamespace(returncode=0)

    def test_selection_is_sorted_deduplicated_and_split_filtered(self):
        ids = [c["clip_id"] for c in self.manifest["clips"]]
        self.args.motion_id = [ids[1], ids[0], ids[1]]
        _, clips = fd.collect_inputs(self.args)
        self.assertEqual([c["clip_id"] for c in clips], ids[:2])
        self.args.motion_id = None
        self.args.split = "test"
        self.assertEqual(fd.collect_inputs(self.args)[1], self.manifest["clips"][2:])
        self.args.motion_id = [ids[0]]
        with self.assertRaisesRegex(ValueError, "outside"):
            fd.collect_inputs(self.args)
        self.args.motion_id = ["../escape"]
        with self.assertRaisesRegex(ValueError, "Unknown"):
            fd.collect_inputs(self.args)

    def test_incomplete_export_and_duplicate_split_rejected(self):
        self.manifest["export_complete"] = False
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "completed"):
            fd.collect_inputs(self.args)
        self.manifest["export_complete"] = True
        self.save_manifest()
        with (self.source / "test.txt").open("a") as handle:
            handle.write(self.manifest["clips"][0]["clip_id"] + "\n")
        with self.assertRaisesRegex(ValueError, "membership"):
            fd.collect_inputs(self.args)

    def test_source_shapes_provenance_timing_and_finite_values(self):
        clip = self.manifest["clips"][0]
        path = fd.validate_source(self.source, self.manifest, clip)
        original = pickle.loads(path.read_bytes())
        for key, bad in (("fps", 30), ("source_dataset", "AIST++"), ("source_id", "999"),
                         ("smpl_poses", np.zeros((479, 72))), ("smpl_trans", np.full((480, 3), np.nan))):
            with self.subTest(key=key):
                payload = dict(original, **{key: bad})
                path.write_bytes(pickle.dumps(payload))
                with self.assertRaises(ValueError):
                    fd.validate_source(self.source, self.manifest, clip)
        path.write_bytes(pickle.dumps(original))
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            fd.validate_source(self.source, self.manifest, dict(clip, duration_seconds=9.))
        with self.assertRaises(ValueError):
            fd.contained(self.source, "../outside.pkl")

    def test_roots_and_dry_run_do_not_write(self):
        self.args.output_root = self.source / "nested"
        with self.assertRaises(ValueError):
            fd.check_roots(self.args)
        unused = self.root / "unused"
        self.assertEqual(fd.main(["--input-root", str(self.source), "--output-root", str(unused), "--dry-run"]), 0)
        self.assertFalse(unused.exists())

    def test_worker_publication_preserves_timing_and_resume_avoids_worker(self):
        clip = self.manifest["clips"][0]
        with patch.object(fd, "run_worker", side_effect=self.fake_worker) as worker:
            result = fd.process_one(self.args, self.manifest, clip, self.common)
            self.assertEqual(result["frames"], 480)
            self.assertEqual(result["audit"]["duration_s"], 479 / 60)
            self.assertEqual(result["clip"], clip)
            result = fd.process_one(self.args, self.manifest, clip, self.common)
            self.assertEqual(result["status"], "cached")
            self.assertEqual(sum("--worker" in call.args[0] for call in worker.call_args_list), 1)

    def test_changed_source_settings_or_states_rejected(self):
        clip = self.manifest["clips"][0]
        identity = self.identity(clip)
        payload = self.payload(identity)
        path = self.output / f"{clip['clip_id']}.pkl"
        path.write_bytes(pickle.dumps(payload))
        for changed in (dict(identity, source_sha256="x" * 64),
                        dict(identity, projection=projection_settings(3.)),
                        dict(identity, clip=dict(clip, label={"style2": "changed"}))):
            with self.assertRaises(ValueError):
                fd.cached_payload(path, changed)
        payload["dof_vel"][0, 0] = 1.
        path.write_bytes(pickle.dumps(payload))
        with self.assertRaisesRegex(ValueError, "checksum"):
            fd.cached_payload(path, identity)

    def test_failed_worker_quarantines_stale_output_and_does_not_publish(self):
        clip = self.manifest["clips"][0]
        output = self.output / f"{clip['clip_id']}.pkl"
        output.write_bytes(b"stale")
        with patch.object(fd, "run_worker", return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                fd.process_one(self.args, self.manifest, clip, self.common)
        self.assertFalse(output.exists())
        self.assertTrue(list(self.output.glob("*.invalid*")))

    def test_failed_final_audit_does_not_publish(self):
        clip = self.manifest["clips"][0]
        with patch.object(fd, "run_worker", side_effect=self.fake_worker), \
                patch.object(fd, "audit_motion", side_effect=ValueError("audit failed")):
            with self.assertRaisesRegex(ValueError, "audit failed"):
                fd.process_one(self.args, self.manifest, clip, self.common)
        self.assertFalse((self.output / f"{clip['clip_id']}.pkl").exists())

    def test_rejects_collision_failure_and_changed_frame_count(self):
        clip = self.manifest["clips"][0]
        identity = self.identity(clip)
        path = self.output / f"{clip['clip_id']}.pkl"
        payload = self.payload(identity)
        payload["projection_validation"]["violating_samples"] = 1
        path.write_bytes(pickle.dumps(payload))
        with self.assertRaises(ValueError):
            fd.cached_payload(path, identity)
        payload = self.payload(identity)
        for key in ("root_pos", "root_rot", "dof_pos", "dof_vel", "dof_acc"):
            payload[key] = payload[key][:-1]
        payload["state_trajectory_sha256"] = state_trajectory_digest(payload)
        path.write_bytes(pickle.dumps(payload))
        with self.assertRaisesRegex(ValueError, "timing"):
            fd.cached_payload(path, identity)

    def test_audit_only_reports_saved_success_and_missing_output(self):
        clips = self.manifest["clips"][:2]
        clip = clips[0]
        payload = self.payload(self.identity(clip))
        (self.output / f"{clip['clip_id']}.pkl").write_bytes(pickle.dumps(payload))
        self.assertEqual(fd.audit_existing(self.args, self.manifest, clips), 1)
        report = fd.read_json(self.output / "audit.json")
        self.assertEqual(len(report["motions"]), 1)
        self.assertEqual(len(report["errors"]), 1)

    def test_corrupt_checkpoints_are_preserved_and_require_explicit_recovery(self):
        path = self.output / "manifest.json"
        for contents in (b"", b"\x00" * 100, b'{"motions":', b"[]", b"\xff"):
            with self.subTest(contents=contents[:10]):
                path.write_bytes(contents)
                with self.assertRaisesRegex(ValueError, "--resume"):
                    batch_state.read_checkpoint(path, "motions", recover=False)
                self.assertEqual(path.read_bytes(), contents)
                before = set(self.output.glob("manifest.json.corrupt.*"))
                value, damaged = batch_state.read_checkpoint(path, "motions", recover=True)
                self.assertEqual(value, {})
                self.assertTrue(damaged)
                backups = set(self.output.glob("manifest.json.corrupt.*")) - before
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups.pop().read_bytes(), contents)

    def test_repair_recovers_valid_unselected_outputs_and_keeps_damaged_files(self):
        good, bad, stale = self.manifest["clips"]
        for clip in (good, stale):
            payload = self.payload(self.identity(clip))
            if clip is stale:
                payload["v2_build"]["source_sha256"] = "changed"
            (self.output / f"{clip['clip_id']}.pkl").write_bytes(pickle.dumps(payload))
        bad_path = self.output / f"{bad['clip_id']}.pkl"
        bad_path.write_bytes(b"\x00" * 10)
        (self.output / "manifest.json").write_bytes(b"\x00" * 50)
        (self.output / "failures.json").write_text('{"failures":')
        self.args.motion_id = [bad["clip_id"]]
        with patch.object(fd.subprocess, "run", side_effect=AssertionError("Must not generate")):
            entries, failures = fd.load_output_state(self.args, self.manifest)
        self.assertEqual(list(entries), [good["clip_id"]])
        self.assertEqual(set(failures), {bad["clip_id"], stale["clip_id"]})
        self.assertEqual(bad_path.read_bytes(), b"\x00" * 10)
        self.assertEqual((self.output / "train.txt").read_text().strip(), good["clip_id"])
        self.assertEqual((self.output / "test.txt").read_text(), "")
        self.assertEqual(len(fd.read_json(self.output / "manifest.json")["motions"]), 1)

    def test_repair_only_cli_never_starts_generation(self):
        clip = self.manifest["clips"][0]
        payload = self.payload(self.identity(clip))
        (self.output / f"{clip['clip_id']}.pkl").write_bytes(pickle.dumps(payload))
        with patch.object(fd, "process_one", side_effect=AssertionError("Must not generate")):
            status = fd.main(["--input-root", str(self.source), "--output-root", str(self.output), "--repair-only"])
        self.assertEqual(status, 0)
        self.assertEqual(len(fd.read_json(self.output / "manifest.json")["motions"]), 1)

    def test_checkpoint_write_flushes_and_preserves_previous_file_on_error(self):
        path = self.output / "manifest.json"
        path.write_text('{"motions": []}')
        original = path.read_bytes()
        with patch.object(batch_state.os, "fsync", side_effect=OSError("flush failed")) as sync:
            with self.assertRaisesRegex(OSError, "flush failed"):
                batch_state.atomic_json(path, {"motions": ["new"]})
        sync.assert_called_once()
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(self.output.glob("*.tmp")))
        with patch.object(batch_state.os, "fsync", wraps=batch_state.os.fsync) as sync:
            batch_state.atomic_json(path, {"motions": []})
        sync.assert_called_once()
        self.assertEqual(fd.read_json(path), {"motions": []})

    def test_metadata_fix_cache_compatibility_is_exact_and_generation_is_unchanged(self):
        builder = Path(fd.__file__)
        self.assertEqual(batch_state.cache_builder_hash(builder), batch_state.LEGACY_BUILDER_SHA256)
        changed = self.root / "changed.py"
        changed.write_bytes(builder.read_bytes() + b"\n# later change\n")
        self.assertEqual(batch_state.cache_builder_hash(changed), hashlib.sha256(changed.read_bytes()).hexdigest())
        names = ("validate_source", "cached_payload", "audit_function", "audit_motion")
        code = "".join(inspect.getsource(getattr(fd, name)) for name in names)
        self.assertEqual(hashlib.sha256(code.encode()).hexdigest(),
                         "6a4e0bd4f13071341fc699940cb2d8d0408ee828949f061b72a5a2a177b6a12c")

    def test_corrupt_input_manifest_is_not_treated_as_output_recovery(self):
        path = self.source / "manifest.json"
        path.write_bytes(b"\x00" * 20)
        with self.assertRaises(ValueError):
            fd.collect_inputs(self.args)
        self.assertEqual(path.read_bytes(), b"\x00" * 20)
        self.assertFalse(list(self.source.glob("*.corrupt.*")))

    def test_batch_continues_failure_and_writes_successful_splits_only(self):
        model_root = self.root / "model"
        model_root.mkdir()
        (model_root / "SMPL_NEUTRAL.pkl").write_bytes(b"model")
        def result(args, manifest, clip, common):
            if clip["source_id"] == "002":
                raise ValueError("unsafe trajectory")
            return dict(source_motion_id=clip["clip_id"], clip=clip, artifact=clip["clip_id"] + ".pkl", status="written")
        with patch.object(fd, "process_one", side_effect=result), \
                patch.object(fd.subprocess, "check_output", return_value="{}"), \
                patch.object(fd, "gmr_version", return_value="test"), \
                patch.object(fd.v2, "implementation_hash", return_value="test"):
            status = fd.main(["--input-root", str(self.source), "--output-root", str(self.output),
                              "--smpl-model-path", str(model_root), "--resume"])
        self.assertEqual(status, 1)
        report = fd.read_json(self.output / "manifest.json")
        self.assertEqual(len(report["motions"]), 2)
        self.assertEqual((self.output / "train.txt").read_text().strip(), self.manifest["clips"][0]["clip_id"])
        self.assertEqual(len(fd.read_json(self.output / "failures.json")["failures"]), 1)


if __name__ == "__main__":
    unittest.main()
