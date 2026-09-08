"""Regression checks for the final-data thesis export; no experiments run."""
from pathlib import Path
import hashlib
import json
import re
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'realtime/humanoid_robot/src/test'), str(ROOT/'realtime/humanoid_robot/src')]
from humanoid_matcher_experiment_metrics import aggregate_pose_npz


class BeatReferenceTests(unittest.TestCase):
    def test_robot_pelvis_not_world_is_detector_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'poses.npz'
            t=np.arange(10)/60
            root=np.column_stack((t*t, t*0, t*0))
            bodies=np.stack((root*0,root,root+[1,0,0]),axis=1)
            np.savez(path,time_seconds=t,joint_positions=np.zeros((10,1)),body_positions=bodies,
                     body_names=['world','pelvis','limb'],center_of_mass=root,
                     left_foot_position=root,right_foot_position=root,
                     left_foot_support_height=t*0,right_foot_support_height=t*0,
                     motion_ids=['a']*10,accepted_causal_beat_times_seconds=[.1],control_rate_hz=60)
            with mock.patch('humanoid_matcher_experiment_metrics.detect_kinematic_beats',return_value=np.array([])) as detector:
                aggregate_pose_npz(path)
            sampled=detector.call_args.args[0]
            self.assertEqual(2,sampled.shape[1])
            np.testing.assert_allclose(sampled[:,0],root)


@unittest.skipUnless((ROOT/'docs/thesis/experiment_results/reanalysis_v2/READY_FOR_THESIS').exists(),
                     'Formal consolidated results are local artifacts')
class ChapterExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thesis=ROOT/'docs/thesis'
        cls.results=cls.thesis/'experiment_results/reanalysis_v2'

    def test_ready_and_export_hashes_match_results(self):
        digest=hashlib.sha256((self.results/'consolidated_results.json').read_bytes()).hexdigest()
        marker=json.loads((self.results/'READY_FOR_THESIS').read_text())
        export=json.loads((self.results/'chapter6_export_manifest.json').read_text())
        self.assertEqual(marker['results_sha256'],digest)
        self.assertEqual(export['consolidated_sha256'],digest)
        self.assertEqual('pelvis_excluding_world_v1',marker['beat_reference_revision'])

    def test_chapter_inputs_figures_citations_and_numbers_resolve(self):
        chapter=(self.thesis/'chapter6_experimental_evaluation.tex').read_text()
        bibliography=(self.thesis/'references.bib').read_text()
        for relative in re.findall(r'\\input\{([^}]+)\}',chapter):
            self.assertTrue((self.thesis/relative).is_file(),relative)
        for relative in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',chapter):
            self.assertTrue((self.thesis/relative).is_file(),relative)
        keys=set(re.findall(r'@\w+\{([^,]+)',bibliography))
        for group in re.findall(r'\\cite\{([^}]+)\}',chapter):
            self.assertTrue(set(group.split(',')) <= keys)
        macros=(self.results/'chapter6_numbers.tex').read_text()
        used=set(re.findall(r'\\(ChSix\w+)',chapter))
        defined=set(re.findall(r'\\newcommand\{\\(ChSix\w+)\}',macros))
        self.assertTrue(used<=defined, used-defined)
        data=json.loads((self.results/'consolidated_results.json').read_text())
        expected=data['cohorts']['online_causal']['short']['statistics']['summaries']['full:bas_harmonic']['bootstrap_95_ci']['mean']
        observed=float(re.search(r'\\ChSixFullBas\}\{([^}]+)',macros).group(1))
        self.assertAlmostEqual(expected,observed,delta=.0005)


if __name__=='__main__':
    unittest.main()
