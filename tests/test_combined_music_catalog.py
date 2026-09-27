from __future__ import annotations
import json
import os
import pickle
import sys
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realtime/humanoid_robot/src"))
sys.path.insert(0, str(Path(__file__).parent))
import build_combined_music_catalog as builder
import evaluate_music_catalog as evaluate
import realtime_music_humanoid_matcher as v2
from music_motion_catalog import MusicCatalog, MusicMotionMatcher, MotionProfile
from test_music_motion_catalog import similarity_catalog, descriptor, motion_profile


class CombinedCatalogTests(unittest.TestCase):
    def test_recording_has_only_one_genre_vote_and_clip_scores_stay_local(self):
        catalog = similarity_catalog({"finedance:Jazz": [.95, .95, .95, .10], "BR": [.8, .8, .8]})
        for i in range(3):
            catalog.tracks[f"finedance:Jazz{i}"]["recording_id"] = "001"
            catalog.tracks[f"finedance:Jazz{i}"]["dataset_id"] = "finedance"
        catalog.tracks["finedance:Jazz3"]["dataset_id"] = "finedance"
        query = replace(descriptor((1, 0), (1, 0)), tag_probabilities=np.array([1., 0.]))
        result = MusicMotionMatcher(catalog, style_first=False, motion_compatibility=False).match(query, top_k_tracks=10, top_k_motions=10)
        self.assertEqual(result.genres[0].genre, "BR")
        scores = {track.music_id: track.score for track in result.tracks}
        genre = next(g for g in result.genres if g.genre == "finedance:Jazz")
        self.assertAlmostEqual(genre.score, (scores["finedance:Jazz0"] + scores["finedance:Jazz3"])/2)
        motions = {m.music_id: m.music_score for m in result.motions}
        self.assertGreater(motions["finedance:Jazz0"], motions["finedance:Jazz3"])

    def test_heldout_excludes_sibling_clips_and_refits_normalization(self):
        catalog = similarity_catalog({"finedance:Jazz": [.99, .98, .5], "BR": [.2]})
        for i in (0, 1):
            catalog.tracks[f"finedance:Jazz{i}"].update(dataset_id="finedance", recording_id="001")
        reference = evaluate.reference_without_recording(catalog, ("finedance", "001"))
        self.assertNotIn("finedance:Jazz0", reference.tracks)
        self.assertNotIn("finedance:Jazz1", reference.tracks)
        np.testing.assert_allclose(reference.embedding_mean, catalog.embeddings[2:].mean(axis=0))
        np.testing.assert_allclose(reference.embedding_std, catalog.embeddings[2:].std(axis=0))
        self.assertEqual(len(reference.motions), 2)

    def test_catalog_paths_and_overrides_from_another_working_directory(self):
        catalog = similarity_catalog({"BR": [.8]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog.catalog_path = root / "catalog/catalog.json"
            catalog.metadata["datasets"] = {
                "aistpp": dict(source_root="../aistpp", gmr_root="../aistpp_gmr"),
                "finedance": dict(source_root="../fine", gmr_root="../fine_gmr"),
            }
            profile = replace(motion_profile("clip", True), dataset_id="finedance")
            args = Namespace(gmr_motion_root=Path("legacy"), explicit_gmr_motion_root=False)
            previous = Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(catalog.motion_file(profile), root / "fine/motions/clip.pkl")
                self.assertEqual(catalog.audio_file(dict(dataset_id="finedance", source_audio="audio/a.wav")), root / "fine/audio/a.wav")
                self.assertEqual(v2.gmr_motion_path(args, profile, catalog), root / "fine_gmr/clip.pkl")
                args.finedance_gmr_motion_root = root / "override"
                self.assertEqual(v2.gmr_motion_path(args, profile, catalog), root / "override/clip.pkl")
                args.explicit_gmr_motion_root = True
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    v2.validate_catalog_overrides(args, catalog)
            finally:
                os.chdir(previous)

    def test_legacy_schema_files_and_paths_remain_supported(self):
        catalog = similarity_catalog({"BR": [.8]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = dict(embeddings=catalog.embeddings, rhythm_timbre=catalog.rhythm_timbre, tags=catalog.tags,
                          embedding_mean=catalog.embedding_mean, embedding_std=catalog.embedding_std,
                          rhythm_mean=catalog.rhythm_mean, rhythm_std=catalog.rhythm_std)
            np.savez(root / "features.npz", **arrays)
            for schema in (1, 2):
                metadata = dict(catalog.metadata, schema_version=schema, arrays_file="features.npz", aistpp_root="source")
                for profile in metadata["motions"].values():
                    for field in ("dataset_id", "recording_id", "split"):
                        profile.pop(field, None)
                (root / "catalog.json").write_text(json.dumps(metadata))
                loaded = MusicCatalog.load(root / "catalog.json")
                profile = next(iter(loaded.motions.values()))
                self.assertEqual(profile.dataset_id, "aistpp")
                self.assertEqual(loaded.motion_file(profile), root / "source" / profile.motion_path)

    def fixture(self, root):
        source, gmr = root / "source", root / "gmr"
        (source / "motions").mkdir(parents=True)
        (source / "audio").mkdir()
        gmr.mkdir()
        motion_id = "finedance_001_0000000_0000600"
        motion_path = source / "motions" / (motion_id + ".pkl")
        artifact_path = gmr / (motion_id + ".pkl")
        motion_path.write_bytes(pickle.dumps(dict(smpl_poses=np.zeros((600, 72)))))
        artifact_path.write_bytes(pickle.dumps(dict(source_sha256=builder.digest(motion_path),
                                                  source_motion_id=motion_id, pipeline_version=5,
                                                  dof_pos=np.zeros((600, 29)), fps=60.)))
        sf.write(source / "audio" / (motion_id + ".wav"), np.zeros(441000), 44100)
        clip = dict(clip_id=motion_id, source_id="001", split="train", frames=600,
                    duration_seconds=10., start_seconds=0., end_seconds=10.,
                    motion_path="motions/" + motion_id + ".pkl", audio_path="audio/" + motion_id + ".wav",
                    label=dict(name="Song", style1="Street", style2="Jazz"))
        failed_clip = dict(clip, clip_id="finedance_002_missing", source_id="002", split="test")
        builder.atomic_json(source / "manifest.json", dict(export_complete=True, clips=[clip, failed_clip]))
        record = dict(source_motion_id=motion_id, artifact=artifact_path.name, status="written",
                      artifact_sha256=builder.digest(artifact_path), clip=clip,
                      v2_build=dict(segmentation_manifest_sha256=builder.digest(source / "manifest.json")))
        builder.atomic_json(gmr / "manifest.json", dict(motions=[record]))
        return source, gmr, motion_id

    def test_manifest_mapping_all_splits_and_missing_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            source, gmr, motion_id = self.fixture(Path(directory))
            entries = builder.dataset_entries("finedance", source, gmr)
            self.assertEqual(len(entries), 2)
            entry = entries[0]
            self.assertEqual(entry["music_id"], motion_id)
            self.assertEqual(entry["genre"], "finedance:Jazz")
            self.assertEqual(entry["recording_id"], "001")
            self.assertEqual(entry["labels"]["style1"], "Street")
            self.assertEqual(len(builder.dataset_entries("finedance", source, gmr, "train")), 1)
            builder.validate_inputs(entry, source, gmr)
            with self.assertRaisesRegex(ValueError, "successful GMR"):
                builder.validate_inputs(entries[1], source, gmr)
            with self.assertRaises(FileNotFoundError):
                builder.validate_inputs(dict(entry, audio_path="audio/missing.wav"), source, gmr)

    def test_changed_artifact_source_timing_and_provenance_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, gmr, _ = self.fixture(Path(directory))
            entry = builder.dataset_entries("finedance", source, gmr)[0]
            for changed, pattern in ((dict(entry, segmentation_hash="stale"), "provenance"),
                                     (dict(entry, frames=599), "frame count"),
                                     (dict(entry, duration_seconds=9), "timing")):
                with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                    builder.validate_inputs(changed, source, gmr)
            audio = source / entry["audio_path"]
            sf.write(audio, np.zeros(44100), 44100)
            with self.assertRaisesRegex(ValueError, "audio duration"):
                builder.validate_inputs(entry, source, gmr)
            sf.write(audio, np.zeros(441000), 44100)
            artifact = gmr / entry["gmr_record"]["artifact"]
            artifact.write_bytes(artifact.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "artifact checksum"):
                builder.validate_inputs(entry, source, gmr)
            entry["gmr_record"]["artifact_sha256"] = builder.digest(artifact)
            (source / entry["motion_path"]).write_bytes(pickle.dumps(dict(smpl_poses=np.ones((600, 72)))))
            with self.assertRaisesRegex(ValueError, "source checksum"):
                builder.validate_inputs(entry, source, gmr)

    def test_cache_identity_and_truncated_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            builder.atomic_json(path, dict(key="old", value=dict(passed=True)))
            self.assertEqual(builder.read_cache(path, "old"), dict(passed=True))
            self.assertIsNone(builder.read_cache(path, "new"))
            path.write_text('{"key":')
            self.assertIsNone(builder.read_cache(path, "old"))

    def test_atomic_publication_keeps_previous_catalog_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text('{"old": true}')
            with patch.object(builder.os, "replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    builder.atomic_json(path, dict(new=True))
            self.assertEqual(json.loads(path.read_text()), dict(old=True))

    def test_malformed_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, gmr, _ = self.fixture(Path(directory))
            entry = builder.dataset_entries("finedance", source, gmr)[0]
            artifact = gmr / entry["gmr_record"]["artifact"]
            artifact.write_bytes(pickle.dumps([1, 2, 3]))
            entry["gmr_record"]["artifact_sha256"] = builder.digest(artifact)
            with self.assertRaisesRegex(ValueError, "dictionary"):
                builder.validate_inputs(entry, source, gmr)

    def test_resume_reuses_features_and_preflight_and_empty_build_preserves_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, gmr, motion_id = self.fixture(root)
            model = root / "model" / "test.xml"
            model.parent.mkdir()
            model.write_text("<mujoco/>")
            args = Namespace(output_dir=root / "catalog", datasets=["finedance"],
                             finedance_root=source, finedance_gmr_root=gmr, model=model,
                             embedding_model=None, tag_model=None, window_seconds=6., hop_seconds=2.,
                             split="all", limit_per_dataset=None)
            profile = replace(motion_profile(motion_id, True), music_id=motion_id,
                              genre="finedance:Jazz", dataset_id="finedance", recording_id="001", split="train").to_dict()
            windows = [dict(start_seconds=0., end_seconds=6., bpm=120., embedding=[1., 2.],
                            rhythm_timbre=[2., 1.], tags=[], beat_strength=.8, onset_density=2.,
                            offbeat_ratio=.2, tempo_stability=.8, spectral_flux=.5, percussive_ratio=.6)]
            with patch.object(builder, "profile_motion", return_value=profile) as preflight, \
                 patch.object(builder, "describe_audio", return_value=windows) as describe, \
                 patch.object(builder, "MotionPreflightValidator"):
                first = builder.build(args)
                self.assertEqual(first["counts"]["motions_included"], 1)
                self.assertEqual(first["counts"]["motions_excluded"], 1)
                self.assertEqual(len(first["tracks"]), 1)
                self.assertEqual(len(first["segments"]), 1)
                builder.build(args)
                self.assertEqual(preflight.call_count, 1)
                self.assertEqual(describe.call_count, 1)
                previous = (args.output_dir / "catalog.json").read_bytes()
                (gmr / (motion_id + ".pkl")).write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "No validated motions"):
                    builder.build(args)
                self.assertEqual((args.output_dir / "catalog.json").read_bytes(), previous)

    def test_cli_tracks_global_override_and_dataset_overrides(self):
        with patch.object(sys, "argv", ["matcher", "--gmr-motion-root", "legacy", "--finedance-gmr-motion-root", "fine"]):
            args = v2.parse_args()
        self.assertTrue(args.explicit_gmr_motion_root)
        self.assertEqual(args.finedance_gmr_motion_root, Path("fine"))
        with patch.object(sys, "argv", ["matcher"]):
            self.assertFalse(v2.parse_args().explicit_gmr_motion_root)

    def test_model_paths_resolve_against_catalog_for_runtime(self):
        catalog = similarity_catalog({"BR": [.8]})
        catalog.catalog_path = ROOT / "tmp" / "example" / "catalog.json"
        catalog.metadata["extractor"] = dict(sample_rate=16000, embedding_backend="test",
                                             embedding_model=dict(path="../model.onnx"), tag_model=None)
        with patch.object(v2, "AudioFeatureExtractor") as extractor:
            extractor.return_value.backend_name = "test"
            v2.make_extractor(Namespace(embedding_model=None, tag_model=None), catalog)
            expected = catalog.catalog_path.parent / "../model.onnx"
            self.assertEqual(extractor.call_args.kwargs["embedding_model"], expected)


    def test_canonical_matcher_defaults_to_combined_catalog(self):
        self.assertEqual(v2.DEFAULT_CATALOG.parent.name, "music_catalog_combined")
        with patch.object(sys, "argv", ["matcher"]):
            self.assertEqual(v2.parse_args().catalog, v2.DEFAULT_CATALOG)



if __name__ == "__main__":
    unittest.main()
