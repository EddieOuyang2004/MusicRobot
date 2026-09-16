"""Read-only comparison: original versus fold-only audio normalisation."""
import sys
import json
import hashlib
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'realtime/humanoid_robot/src/test'))
import run_humanoid_matcher_experiments as runner

catalog_path = ROOT / 'realtime/humanoid_robot/data/music_catalog/catalog.json'
catalog = runner.MusicCatalog.load(catalog_path)
original_builder = runner.catalog_without_music
results = {'catalog_sha256': hashlib.sha256(catalog_path.read_bytes()).hexdigest()}
start = time.perf_counter()
for mode in ('existing_global_statistics', 'fold_only_audio_statistics'):
    if mode == 'fold_only_audio_statistics':
        def fold_only(source, music_id):
            fold = original_builder(source, music_id)
            fold.embedding_mean = np.mean(fold.embeddings, axis=0).astype(np.float32)
            fold.embedding_std = np.std(fold.embeddings, axis=0).astype(np.float32)
            fold.rhythm_mean = np.mean(fold.rhythm_timbre, axis=0).astype(np.float32)
            fold.rhythm_std = np.std(fold.rhythm_timbre, axis=0).astype(np.float32)
            return fold
        runner.catalog_without_music = fold_only
    result = runner.strict_leave_one_music_evaluation(catalog)
    results[mode] = {key: result[key] for key in ('ranking', 'music_level_bootstrap_95_ci')}
    print(mode, json.dumps(results[mode]), flush=True)
results['elapsed_seconds'] = time.perf_counter() - start
results['scope'] = 'Diagnostic only; audio normalisation changed in memory. No saved catalogue, source code, thesis, or frozen result modified. Motion statistics and manually selected settings were not re-estimated.'
(Path(__file__).parent / 'retrieval_normalisation_check.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
