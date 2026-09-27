from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realtime/humanoid_robot/src"))
from restore_finedance_audio import restore, sha256


class RestoreFineDanceAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.recipe = self.root / "recipe"
        self.output = self.root / "output"
        (self.source / "music_wav").mkdir(parents=True)
        self.recipe.mkdir()
        wav = self.source / "music_wav/001.wav"
        sf.write(wav, np.sin(np.arange(3 * 44100) * 0.01) * 0.5, 44100, subtype="PCM_16")
        mono, _ = sf.read(wav, dtype="float32")
        clips, hashes = [], {}
        self.expected = {}
        for i, (start, end) in enumerate([(10, 70), (70, 130)]):
            name = f"finedance_001_{start:07d}_{end:07d}"
            audio = np.column_stack([mono[start * 735:end * 735]] * 2)
            buffer = io.BytesIO()
            sf.write(buffer, audio, 44100, format="WAV", subtype="PCM_16")
            self.expected[name] = buffer.getvalue()
            clips.append(dict(source_id="001", clip_id=name, start_frame=start,
                              end_frame_exclusive=end, audio_path=f"audio/{name}.wav",
                              audio_samples=(end - start) * 735))
            hashes[name] = dict(audio_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
                                gmr_sha256="a" * 64 if i == 0 else None)
        manifest = self.recipe / "segmentation_manifest.json"
        manifest.write_text(json.dumps(dict(fps=60, audio_rate=44100, clips=clips)), encoding="utf-8")
        (self.recipe / "checksums.json").write_text(json.dumps(dict(
            segmentation_manifest_sha256=sha256(manifest),
            sources={"001": dict(audio_sha256=sha256(wav))}, clips=hashes)), encoding="utf-8")

    def run_restore(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return restore(self.source, self.output, self.recipe, **kwargs)

    def test_successful_subset_full_set_and_idempotent_write(self):
        self.assertEqual(self.run_restore(), 1)
        self.assertEqual(len(list(self.output.rglob("*.wav"))), 1)
        self.assertEqual(self.run_restore(all_clips=True), 2)
        for name, payload in self.expected.items():
            self.assertEqual((self.output / "audio" / (name + ".wav")).read_bytes(), payload)
        self.assertEqual(self.run_restore(all_clips=True), 2)

    def test_verify_only_writes_nothing_and_unknown_source_rejected(self):
        self.assertEqual(self.run_restore(verify_only=True, all_clips=True), 2)
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "Unknown source"):
            self.run_restore(source_ids=["999"])

    def test_changed_source_and_manifest_are_rejected(self):
        wav = self.source / "music_wav/001.wav"
        original = wav.read_bytes()
        wav.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "Original WAV differs"):
            self.run_restore()
        self.assertFalse(self.output.exists())
        wav.write_bytes(original)
        manifest = self.recipe / "segmentation_manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            self.run_restore()

    def test_different_existing_output_is_preserved(self):
        self.run_restore()
        target = next(self.output.rglob("*.wav"))
        target.write_bytes(b"keep existing user file")
        with self.assertRaises(FileExistsError):
            self.run_restore()
        self.assertEqual(target.read_bytes(), b"keep existing user file")


if __name__ == "__main__":
    unittest.main()
