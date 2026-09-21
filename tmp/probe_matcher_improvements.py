"""Exploratory ranking ablation: one original 26-second window per audit song.

Never modifies production code or launches the robot/runtime. Keeps the saved
baseline rejection decision for the unrestricted-style ranking comparison.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'realtime/humanoid_robot/src'
sys.path.insert(0, str(SRC))
from music_motion_catalog import MusicCatalog, MusicMotionMatcher, load_audio_mono
from realtime_music_humanoid_matcher_v2 import make_extractor

BASE = SRC / 'test/output/v2_selection_audit_20'
OUT = BASE / 'improvement_probe'
OUT.mkdir(exist_ok=True)
manifest = json.loads((BASE / 'manifest.json').read_text())
catalog = MusicCatalog.load(ROOT / manifest['catalog'])
extractor = make_extractor(argparse.Namespace(embedding_model=None, tag_model=None), catalog)


class NoTagPrior(MusicMotionMatcher):
    @staticmethod
    def _tag_genre_priors(labels, probabilities):
        return {}


rows = []
for song in manifest['songs']:
    saved = json.loads((BASE / 'runs' / song['id'] / 'diagnostic.json').read_text())
    saved = [row for row in saved if row['condition'] == 'production']
    reference = next(row['result'] for row in saved if row['end_seconds'] == 26)
    streak = 0
    for row in saved:
        if row['end_seconds'] >= 26:
            break
        streak = streak + 1 if row['result']['negative_style_score'] >= .17 else 0
    audio = load_audio_mono(Path(song['prepared_audio']), extractor.sample_rate)[:26 * extractor.sample_rate]
    descriptor = extractor.describe(audio)
    models = {
        'baseline': MusicMotionMatcher(catalog, speed_max=1.3),
        'no_tag_genre_prior': NoTagPrior(catalog, speed_max=1.3),
        'no_hard_genre_gate': MusicMotionMatcher(catalog, speed_max=1.3, style_first=False),
    }
    results = {}
    for name, model in models.items():
        model._weak_music_streak = streak
        result = model.match(descriptor, top_k_tracks=5, top_k_motions=20)
        if name == 'no_hard_genre_gate':
            base = results['baseline']
            result = replace(result, accepted=base.accepted, rejection_reason=base.rejection_reason,
                             motions=result.motions if base.accepted else ())
        results[name] = result
    baseline = asdict(results['baseline'])
    assert baseline['accepted'] == reference['accepted'], song['id']
    assert [x['motion_id'] for x in baseline['motions']] == [x['motion_id'] for x in reference['motions']], song['id']
    assert [x['music_id'] for x in baseline['tracks']] == [x['music_id'] for x in reference['tracks']], song['id']
    rows.append({'song': song['id'], 'title': song['title'], 'group': song['group'],
                 'music_id': song['music_id'], 'source_seconds': song['source_seconds'],
                 'previous_weak_streak': streak, 'results': {k: asdict(v) for k, v in results.items()}})
    print(song['title'], {k: v.motions[0].motion_id if v.motions else 'rejected' for k, v in results.items()}, flush=True)

summary = {}
for condition in models:
    summary[condition] = {}
    for group in ('all', 'aist_catalog', 'cc0_music', 'cc0_ambient'):
        selected = [row for row in rows if group == 'all' or row['group'] == group]
        valid = [row for row in selected if row['results'][condition]['accepted'] and row['results'][condition]['motions']]
        ids = Counter(row['results'][condition]['motions'][0]['motion_id'] for row in valid)
        genres = Counter(catalog.motions[row['results'][condition]['motions'][0]['motion_id']].genre for row in valid)
        expected = [row for row in selected if row['music_id']]
        same = sum(bool(row['results'][condition]['motions']) and
                   row['results'][condition]['motions'][0]['music_id'] == row['music_id'] for row in expected)
        summary[condition][group] = {'songs': len(selected), 'accepted_with_motions': len(valid),
                                   'same_music_top_motion': same if expected else None,
                                   'same_music_denominator': len(expected), 'top_motion_counts': ids,
                                   'motion_genre_counts': genres}

output = {'protocol': 'Exploratory one-window-per-song comparison. First 26 seconds, before any loop; catalog/speed limits unchanged; baseline replay validated against saved rankings. No inference about full-run quality.',
          'baseline_reproduction': 'All 20 accepted flags, track lists and motion lists matched saved diagnostic window 26.',
          'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (SRC / 'music_motion_catalog.py', Path(__file__))},
          'summary': summary, 'rows': rows}
(OUT / 'probe.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
