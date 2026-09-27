from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realtime/humanoid_robot/src"))
sys.path.insert(0, str(Path(__file__).parent))
import segment_music_catalog as splitter
import build_combined_music_catalog as builder
from gmr_collision_projection import projection_settings, state_trajectory_digest, validate_v2_metadata
from test_music_motion_catalog import similarity_catalog


def valid_report():
    return dict(passed=True, clearance_m=.005, violating_samples=0, samples_per_interval=81,
                models=2, interpolation="quintic_hermite_states",
                continuous_peaks=dict(speed_rad_s=0., acceleration_rad_s2=0., jerk_rad_s3=0.))


class SegmentedCatalogTests(unittest.TestCase):
    def fixture(self, root, frames=1200):
        source, gmr = root / "source", root / "gmr"
        (source / "motions").mkdir(parents=True)
        (source / "audio").mkdir()
        gmr.mkdir()
        name = "gBR_sFM_cAll_d04_mBR0_ch01"
        human = dict(smpl_poses=np.arange(frames * 72, dtype=np.float64).reshape(frames, 72),
                     smpl_trans=np.arange(frames * 3, dtype=np.float64).reshape(frames, 3),
                     smpl_scaling=np.array([1.]))
        motion = source / "motions" / (name + ".pkl")
        motion.write_bytes(pickle.dumps(human))
        robot = dict(source_motion_id=name, source_sha256=builder.digest(motion), fps=60.,
                     pipeline_version=5, motion_version="gmr_v2", projection=projection_settings(),
                     projection_validation=valid_report(), root_pos=human["smpl_trans"].copy(),
                     root_rot=np.tile([1., 0., 0., 0.], (frames, 1)),
                     dof_pos=np.zeros((frames, 29)), dof_vel=np.zeros((frames, 29)),
                     dof_acc=np.zeros((frames, 29)), v2_build=dict(parent=True), generation_seconds=123.)
        robot["state_trajectory_sha256"] = state_trajectory_digest(robot)
        artifact = gmr / (name + ".pkl")
        artifact.write_bytes(pickle.dumps(robot))
        audio = np.zeros((round(frames/60*48000), 2))
        audio[12*48000:12*48000+50] = [.75, -.25]
        sf.write(source / "audio" / (name + ".wav"), audio, 48000, subtype="PCM_24")
        (source / "train.txt").write_text(name + "\n")
        (source / "audio/manifest.csv").write_text(
            "motion_name,motion_frames,duration_seconds,audio_path\n"
            f"{name},{frames},{frames/60},{name}.wav\n")
        builder.atomic_json(gmr / "manifest.json", dict(motions=[dict(source_motion_id=name,
            artifact=artifact.name, artifact_sha256=builder.digest(artifact), status="written")]))
        entry = builder.dataset_entries("aistpp", source, gmr)[0]
        parent = dict(entry=entry, hashes=builder.validate_inputs(entry, source, gmr), children=[
            dict(motion_id=splitter.child_id(name, a, b), start_frame=a, end_frame_exclusive=b)
            for a, b in splitter.segment_bounds(frames)])
        args = splitter.parse_args(["--output-dir", str(root / "output"), "--limit", "1"])
        args.output_dir.mkdir()
        roots = dict(aistpp=dict(source_root=source, gmr_root=gmr))
        return args, roots, parent, human, robot

    def test_bounds_threshold_balancing_and_complete_coverage(self):
        self.assertEqual(splitter.segment_bounds(960), [(0, 960)])
        self.assertEqual(splitter.segment_bounds(961), [(0, 481), (481, 961)])
        self.assertEqual(splitter.segment_bounds(1200), [(0, 600), (600, 1200)])
        for frames in range(961, 10001):
            bounds = splitter.segment_bounds(frames)
            sizes = [b-a for a, b in bounds]
            self.assertEqual(bounds[0][0], 0)
            self.assertEqual(bounds[-1][1], frames)
            self.assertTrue(all(a[1] == b[0] for a, b in zip(bounds, bounds[1:])))
            self.assertLessEqual(max(sizes)-min(sizes), 1)
        self.assertEqual(splitter.child_id("parent", 576, 1152), "parent__f0000576_0001152")
        with self.assertRaises(ValueError):
            splitter.segment_bounds(1)

    def test_slice_preserves_states_and_refreshes_hashes_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            args, roots, parent, human, robot = self.fixture(Path(directory))
            robot["dof_vel"][600:] = .01
            robot["dof_acc"][600:] = -.03
            robot["state_trajectory_sha256"] = state_trajectory_digest(robot)
            validator = Mock()
            validator.validate.return_value = valid_report()
            provenance = dict(parent_motion_id=parent["entry"]["motion_id"], start_frame=600, end_frame_exclusive=1200)
            a, b = splitter.slice_payloads(human, robot, 600, 1200, "child", provenance, validator)
            smpl, child = pickle.loads(a), pickle.loads(b)
            for key in splitter.ROBOT_ARRAYS:
                np.testing.assert_array_equal(child[key], robot[key][600:])
            np.testing.assert_array_equal(smpl["smpl_poses"], human["smpl_poses"][600:])
            self.assertEqual(child["source_sha256"], hashlib.sha256(a).hexdigest())
            self.assertEqual(child["source_motion_id"], "child")
            self.assertEqual(child["segmentation"], provenance)
            self.assertEqual(child["parent_generation"]["v2_build"], dict(parent=True))
            self.assertNotIn("generation_seconds", child)
            validate_v2_metadata(child)

    def test_export_audio_alignment_resume_corruption_and_manifest_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            args, roots, parent, _, _ = self.fixture(Path(directory))
            with patch.object(splitter, "SliceValidator") as validator, \
                 patch.object(builder, "MotionPreflightValidator") as preflight:
                validator.return_value.validate.return_value = valid_report()
                preflight.return_value.validate.return_value = (True, "ok")
                source, gmr, manifest = splitter.export(args, roots, [parent], dict(revision=1))
                self.assertEqual(len(manifest["clips"]), 2)
                entries = builder.dataset_entries("aistpp", source, gmr)
                for entry in entries:
                    builder.validate_inputs(entry, source, gmr)
                    self.assertEqual(entry["recording_id"], "BR0")
                    self.assertEqual(entry["music_id"], entry["motion_id"])
                    self.assertEqual(entry["split"], "train")
                audio, rate = sf.read(source / entries[1]["audio_path"], always_2d=True)
                self.assertEqual((rate, audio.shape), (48000, (480000, 2)))
                np.testing.assert_allclose(audio[2*rate], [.75, -.25])
                self.assertEqual(sf.info(source / entries[1]["audio_path"]).subtype, "PCM_24")
                splitter.export(args, roots, [parent], dict(revision=1))
                self.assertEqual(validator.return_value.validate.call_count, 2)
                damaged = gmr / entries[0]["gmr_record"]["artifact"]
                damaged.write_bytes(b"interrupted")
                splitter.export(args, roots, [parent], dict(revision=1))
                self.assertEqual(validator.return_value.validate.call_count, 3)
                new_source, _, _ = splitter.export(args, roots, [parent], dict(revision=2))
                self.assertNotEqual(source, new_source)
                self.assertEqual(validator.return_value.validate.call_count, 5)
                changed = dict(entries[0], provenance=dict(tampered=True))
                with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                    builder.validate_inputs(changed, source, gmr)
                # Same duration and path, different waveform must invalidate admission.
                path = source / entries[0]["audio_path"]
                sf.write(path, np.ones((480000, 2))*.1, 48000)
                with self.assertRaisesRegex(ValueError, "checksum"):
                    builder.validate_inputs(entries[0], source, gmr)

    def test_failed_child_blocks_manifest_admission_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, roots, parent, _, _ = self.fixture(Path(directory))
            with patch.object(splitter, "SliceValidator") as validator, \
                 patch.object(builder, "MotionPreflightValidator") as preflight:
                validator.return_value.validate.return_value = valid_report()
                preflight.return_value.validate.side_effect = [(False, "collision"), (True, "ok")]
                with self.assertRaisesRegex(ValueError, "export failures"):
                    splitter.export(args, roots, [parent], {})
                manifest = next(args.output_dir.glob("assets/*/aistpp/segments_manifest.json"))
                self.assertFalse(builder.read_json(manifest)["export_complete"])
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    builder.dataset_entries("aistpp", manifest.parent, manifest.parent.parent / "aistpp_gmr_v2")
                preflight.return_value.validate.side_effect = None
                preflight.return_value.validate.return_value = (True, "ok")
                _, _, result = splitter.export(args, roots, [parent], {})
                self.assertTrue(result["export_complete"])
                self.assertEqual(validator.return_value.validate.call_count, 3)

    def test_inventory_dry_run_does_not_create_output_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, roots, parent, _, _ = self.fixture(root)
            args.output_dir = root / "not-created"
            args.catalog = root / "catalog.json"
            metadata = dict(schema_version=3, datasets=dict(roots, finedance=dict(source_root="fine", gmr_root="fine_gmr")),
                            motions={parent["entry"]["motion_id"]: dict(dataset_id="aistpp", duration_seconds=20)})
            metadata["datasets"]["aistpp"] = {k: str(v) for k, v in roots["aistpp"].items()}
            builder.atomic_json(args.catalog, metadata)
            args.dry_run = True
            result = splitter.run(args)
            self.assertEqual(result["output_aistpp_motions"], 2)
            self.assertFalse(args.output_dir.exists())
            audio_path = roots["aistpp"]["source_root"] / parent["entry"]["audio_path"]
            sf.write(audio_path, np.zeros(959000), 48000)  # smaller than builder's legacy 0.1s tolerance
            with self.assertRaisesRegex(ValueError, "one sample"):
                splitter.inventory(args)

    def test_membership_rejects_missing_child_and_changed_finedance(self):
        baseline = dict(motions=dict(long=dict(dataset_id="aistpp"), fine=dict(dataset_id="finedance")))
        candidate = dict(motions=dict(child=dict(dataset_id="aistpp", duration_seconds=9),
                                     fine=dict(dataset_id="finedance", duration_seconds=10)))
        manifest = dict(clips=[dict(motion_id="child")])
        splitter.validate_membership(candidate, baseline, manifest)
        bad = copy.deepcopy(candidate)
        bad["motions"].pop("child")
        with self.assertRaises(ValueError):
            splitter.validate_membership(bad, baseline, manifest)
        bad = copy.deepcopy(candidate)
        bad["motions"]["new_fine"] = bad["motions"].pop("fine")
        with self.assertRaises(ValueError):
            splitter.validate_membership(bad, baseline, manifest)

    def test_publication_rebases_backs_up_and_preserves_previous_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_dir, active_dir = root / "candidate", root / "active"
            candidate_dir.mkdir()
            active_dir.mkdir()
            sample = similarity_catalog({"BR": [.8]})
            metadata = copy.deepcopy(sample.metadata)
            metadata.update(schema_version=3, datasets=dict(aistpp=dict(source_root="../assets", gmr_root="../robot")),
                            arrays_file="catalog_features_new.npz", extractor={})
            np.savez(candidate_dir / metadata["arrays_file"], embeddings=sample.embeddings,
                     rhythm_timbre=sample.rhythm_timbre, tags=sample.tags, embedding_mean=sample.embedding_mean,
                     embedding_std=sample.embedding_std, rhythm_mean=sample.rhythm_mean, rhythm_std=sample.rhythm_std)
            candidate, target = candidate_dir / "catalog.json", active_dir / "catalog.json"
            builder.atomic_json(candidate, metadata)
            target.write_text('{"previous": true}')
            original = target.read_bytes()
            with patch.object(splitter.os, "replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    splitter.publish(candidate, target, builder.digest(target))
            self.assertEqual(target.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "changed"):
                splitter.publish(candidate, target, "wrong")
            replace = splitter.os.replace
            def fail_at_activation(source, destination):
                if Path(destination) == target:
                    raise OSError("activation interrupted")
                return replace(source, destination)
            with patch.object(splitter.os, "replace", side_effect=fail_at_activation):
                with self.assertRaisesRegex(OSError, "activation interrupted"):
                    splitter.publish(candidate, target, builder.digest(target))
            self.assertEqual(target.read_bytes(), original)
            backup = splitter.publish(candidate, target, builder.digest(target))
            self.assertEqual(backup.read_bytes(), original)
            result = builder.read_json(target)
            self.assertEqual(result["datasets"]["aistpp"]["source_root"], "../assets")
            self.assertTrue((active_dir / result["arrays_file"]).exists())

    def test_orchestrator_never_activates_pilot_or_failed_playback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, roots, parent, _, _ = self.fixture(root)
            args.catalog = root / "catalog.json"
            args.catalog.write_text('{"active": true}')
            baseline = dict(datasets={k: {n: str(p) for n, p in v.items()} for k, v in roots.items()},
                            arrays_file="features.npz", extractor={})
            args.build_catalog = True
            active = args.catalog.read_bytes()
            with patch.object(splitter, "inventory", return_value=(args.catalog, baseline, roots, [parent])), \
                 patch.object(splitter, "validation_identity", return_value={}), \
                 patch.object(splitter, "export", return_value=(root / "derived", root / "robot", {})), \
                 patch.object(splitter, "build_catalog", return_value=root / "candidate.json") as build, \
                 patch.object(splitter, "publish") as publish:
                self.assertEqual(splitter.run(args)["status"], "pilot_validated")
                publish.assert_not_called()
                args.limit = None
                build.side_effect = RuntimeError("playback failed")
                with self.assertRaisesRegex(RuntimeError, "playback failed"):
                    splitter.run(args)
                publish.assert_not_called()
                self.assertEqual(args.catalog.read_bytes(), active)

    def test_unchanged_short_clip_is_copied_and_admitted_without_slicing(self):
        with tempfile.TemporaryDirectory() as directory:
            args, roots, parent, _, _ = self.fixture(Path(directory), frames=960)
            parent["children"][0]["motion_id"] = parent["entry"]["motion_id"]
            with patch.object(splitter, "SliceValidator") as validator, \
                 patch.object(builder, "MotionPreflightValidator") as preflight:
                preflight.return_value.validate.return_value = (True, "ok")
                source, gmr, _ = splitter.export(args, roots, [parent], {})
                validator.assert_not_called()
                entry = builder.dataset_entries("aistpp", source, gmr)[0]
                hashes = builder.validate_inputs(entry, source, gmr)
                self.assertEqual(hashes, parent["hashes"])
                self.assertEqual(entry["music_id"], "BR0")


if __name__ == "__main__":
    unittest.main()
