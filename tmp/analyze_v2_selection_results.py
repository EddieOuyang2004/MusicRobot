"""Read-only reanalysis of the completed 20-song experiment; writes supplements."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'realtime/humanoid_robot/src/test/output/v2_selection_audit_20'
OUT = BASE / 'analysis'
OUT.mkdir(exist_ok=True)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


manifest = read(BASE / 'manifest.json')
report = read(BASE / 'report.json')
catalog = read(ROOT / manifest['catalog'])
changed_inputs = [name for name, expected in manifest['sha256'].items() if sha(ROOT / name) != expected]
assert not changed_inputs, changed_inputs
assert len(manifest['songs']) == 20
data = []
for song in manifest['songs']:
    folder = BASE / 'runs' / song['id']
    marker = read(folder / 'complete.json')
    assert all(sha(folder / name) == expected for name, expected in marker['sha256'].items())
    assert sha(Path(song['prepared_audio'])) == song['prepared_sha256']
    live = [json.loads(line)['result'] for line in (folder / 'retrieval.jsonl').read_text().splitlines()]
    with (folder / 'trace.csv').open(encoding='utf-8-sig', newline='') as f:
        trace = list(csv.DictReader(f))
    timing = read(folder / 'timing.json')
    assert len(live) == timing['retrieval_completed']
    assert float(trace[-1]['audio_time_seconds']) >= 59.75
    accepted = [r for r in live if r['accepted'] and r['motions']]
    recommendations = Counter(r['motions'][0]['motion_id'] for r in accepted)
    tracks = Counter(r['tracks'][0]['music_id'] for r in live if r['tracks'])
    plays = Counter(r['current_motion_id'] for r in trace if r['event'] == 'switch_complete')
    held_seconds = sum(float(b['audio_time_seconds']) - float(a['audio_time_seconds'])
                       for a, b in zip(trace, trace[1:]) if a['bridge_preparation_status'] == 'terminal_hold')
    failures = list(dict.fromkeys(r['bridge_failure_reason'] for r in trace if r['bridge_failure_reason']))
    diag = [r for r in read(folder / 'diagnostic.json') if r['condition'] == 'production']
    data.append({**song, 'retrievals': len(live), 'accepted': len(accepted),
                 'rejections': dict(Counter(r['rejection_reason'] for r in live if not r['accepted'])),
                 'top1_recommendations': dict(recommendations), 'top1_tracks': dict(tracks),
                 'completed_plays': dict(plays), 'terminal_hold_seconds': held_seconds,
                 'bridge_failures': failures, 'trace_end_seconds': float(trace[-1]['audio_time_seconds']),
                 'diagnostic_accepted': sum(r['result']['accepted'] and bool(r['result']['motions']) for r in diag)})


def shares(rows, key, transform=lambda value: value):
    total = Counter()
    informative = 0
    for row in rows:
        counts = row[key]
        denominator = sum(counts.values())
        if denominator:
            informative += 1
            for identifier, count in counts.items():
                total[transform(identifier)] += count / denominator
    return {k: v / informative for k, v in total.most_common()}, informative


genre = lambda mid: catalog['motions'][mid]['genre']
recommendation_genres, n_recommendations = shares(data, 'top1_recommendations', genre)
play_genres, n_plays = shares(data, 'completed_plays', genre)
summary = {'integrity': 'All 20 completion markers, raw artifact hashes, prepared audio hashes and frozen inputs verified.',
           'total_retrievals': sum(s['retrievals'] for s in data),
           'accepted_with_motions': sum(s['accepted'] for s in data),
           'completed_plays': sum(sum(s['completed_plays'].values()) for s in data),
           'actual_recommended_motion_genre_shares': recommendation_genres,
           'actual_played_motion_genre_shares': play_genres,
           'songs_with_recommendations': n_recommendations, 'songs_with_plays': n_plays,
           'songs': data}
(OUT / 'analysis.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

sys.path.append(str(ROOT / 'tmp/thesis_plot_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': '#fafbfc', 'axes.facecolor': '#fafbfc'})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={'width_ratios': [1, 1.35]})
metrics = [report['groups']['all'][key] for key in ['live.motion_top1_accepted', 'runtime.completed_plays_excluding_startup']]
x = np.arange(2)
for offset, key, label, color in [(-0.18, 'top1_share', 'Most frequent motion ID', '#3a65a5'),
                                (0.18, 'top5_share', 'Five most frequent motion IDs', '#d68936')]:
    values = [100 * s[key] for s in metrics]
    bars = axes[0].bar(x + offset, values, width=.34, label=label, color=color)
    axes[0].bar_label(bars, labels=[f'{v:.1f}%' for v in values], padding=4)
axes[0].set_xticks(x, ['Top recommendations\n19 contributing songs', 'Completed plays\n17 contributing songs'])
axes[0].set_ylim(0, 76)
axes[0].set_ylabel('Share, giving equal weight to each contributing song (%)')
axes[0].set_title('Repetition is stronger in recommendations', loc='left', weight='bold')
axes[0].legend(fontsize=8, frameon=False, loc='upper right')
held = sorted([s for s in data if s['terminal_hold_seconds'] > 0], key=lambda s: s['terminal_hold_seconds'])
bars = axes[1].barh([s['title'] for s in held], [s['terminal_hold_seconds'] for s in held], color='#af5266')
axes[1].bar_label(bars, fmt='%.1f s', padding=4)
axes[1].set_xlim(0, 62)
axes[1].set_xlabel('Seconds in terminal hold during each 60-second trial')
axes[1].set_title('Six trials entered a persistent terminal hold', loc='left', weight='bold')
fig.suptitle('Humanoid matcher v2 - completed 20-song audit', fontsize=15, weight='bold', y=1.02)
fig.text(.01, -.025, 'Playback excludes the initial motion. Rejected/no-switch songs are excluded from the respective share denominators; holds remain visible at right.', fontsize=9)
fig.tight_layout()
fig.savefig(OUT / 'concentration_and_holds.png', dpi=170, bbox_inches='tight')
plt.close(fig)

top = [x['id'] for x in report['groups']['all']['live.motion_top1_accepted']['rankings'][:8]]
matrix = np.array([[s['top1_recommendations'].get(mid, 0) / s['accepted'] if s['accepted'] else np.nan
                    for mid in top] for s in data]) * 100
fig, ax = plt.subplots(figsize=(11, 9.3))
cmap = plt.colormaps['Blues'].copy()
cmap.set_bad('#e5e7eb')
im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=100)
ax.set_yticks(range(20), [s['title'] + (f" ({s['accepted']}/57 accepted)" if s['accepted'] < 57 else '') for s in data])
ax.set_xticks(range(len(top)), [mid.replace('_cAll', '').replace('g', '', 1) for mid in top], rotation=45, ha='right', fontsize=8)
for i in range(20):
    for j in range(len(top)):
        v = matrix[i, j]
        if np.isfinite(v) and v > 0:
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', color='white' if v > 55 else '#243347', fontsize=9)
ax.axhline(9.5, color='#af5266', lw=1.5)
ax.set_title('Which songs repeatedly recommend the same motions?\nPercent of accepted recommendations within each song; eight most frequent IDs', loc='left', weight='bold', pad=14)
fig.colorbar(im, ax=ax, fraction=.026, pad=.025, label='Percent of accepted recommendations')
fig.tight_layout()
fig.savefig(OUT / 'recommendation_by_song.png', dpi=170, bbox_inches='tight')
plt.close(fig)

lines = [
    '# Interpretation of the completed 20-song matcher v2 experiment', '',
    '**Your impression is partly correct: a small set of motions repeatedly tops the rankings. Actual playback uses more distinct motion IDs, but style concentration and persistent terminal holds remain. The evidence does not justify blaming the ONNX embedding model alone.**', '',
    '## Validation and denominators', '',
    'All 20 trials completed. All raw artifact hashes, prepared audio hashes and frozen input hashes match. Every trial contains 57 delivered retrievals (1,140 total), and traces reach 59.96 seconds. There are 994 accepted results with motions, 83 completed motion switches and 40 distinct played motion IDs. No same-ID consecutive replay completed.', '',
    'Shares below first normalize within a song and then average across contributing songs. Recommendations use 19 songs with at least one accepted motion; playback uses 17 songs with at least one completed switch. Therefore these stages have different denominators. They describe concentration, not a statistically significant defect. No motion crosses both the preregistered 20% share and five-song prevalence thresholds in the full-sample recommendation table.', '',
    '## Repeated recommendations', '',
    '| Motion ID | Share of top recommendations | Songs where it ranks first at least once |',
    '|---|---:|---:|']
for row in report['groups']['all']['live.motion_top1_accepted']['rankings'][:5]:
    lines.append(f"| `{row['id']}` | {row['song_balanced_share']:.1%} | {row['songs_present']}/19 |")
lines += [
    '', 'The leading three account for 47.6% and the leading five for 62.6% of recommendations. Ten of the 19 contributing songs have one motion at the top for at least 90% of accepted retrievals. This within-song stability is real, but overlapping windows and consistent musical style naturally produce correlated rankings.', '',
    'The leading played motion accounts for 7.9%, and the five leading played motions account for 25.5% of completed plays. The exact top recommendation is not always the motion that passes transition selection. Diversity across IDs also does not guarantee visual diversity: Popping motions account for 43.8% of song-balanced plays (56.4% among catalog-song trials). These are descriptive style shares, not an expectation that every style should be equally frequent.', '',
    '## Where the concentration appears', '',
    '**External audio:** among the nine rhythmic CC0 tracks, 83.7% of embedding-only top track choices fall in the Jazz Ballet (JB) family. The combined runtime ranking assigns 81.5% to that family. Individual JB5 is the top combined track in five of the nine songs and has a 29.0% share. Thus concentration already exists in feature/catalog retrieval for this external sample. It could reflect feature geometry, catalog representation or the limited sample; this run cannot isolate encoder training as the cause.', '',
    '**Catalog controls:** all 570 live retrievals across the ten AIST++ songs correctly retrieve their own music ID as the top combined track. All 300 deterministic embedding-only observations also retrieve their own music ID. Yet the following accepted motion rankings occur:', '',
    '| Input song | Top combined track | Dominant motion recommendation | Frequency |',
    '|---|---|---|---:|',
    '| KR2 | KR2 in 57/57 | `gPO_sFM_cAll_d12_mPO1_ch16` | 57/57 |',
    '| PO1 | PO1 in 57/57 | `gPO_sFM_cAll_d12_mPO1_ch16` | 57/57 |',
    '| WA0 | WA0 in 57/57 | `gPO_sFM_cAll_d12_mPO1_ch16` | 57/57 |',
    '| MH3 | MH3 in 57/57 | `gPO_sBM_cAll_d12_mPO4_ch02` | 52/57 |',
    '| LH4 | LH4 in 57/57 | `gHO_sBM_cAll_d20_mHO5_ch10` | 43/57 |', '',
    'This locates a major effect after track recognition. In `music_motion_catalog.py:1006`, the genre score gives 65% weight to a normalized tag-derived prior and 35% to normalized genre similarity. Lines 1060-1081 retain only the winning genre (plus a near-tied second genre) before ranking compatible motions. The manually defined Electronic-tag fallback routes many styles into PO. These rules can exclude the genre of a very strong individual track match. The top reported track list is computed before this gate, so its first entry need not supply the selected motion.', '',
    'For example, the 29th live retrieval of KR2 scores its own track at 0.970 versus 0.621 for PO1, but the winning genre is PO and the chosen motion comes from PO1. This shows a strong match being overridden; it does not by itself prove the dance is aesthetically wrong, since AIST dance styles and broad audio tags are not identical concepts. A tag-prior ablation is needed to measure the causal contribution of that rule.', '',
    '**Library size is not a sufficient explanation:** the library contains 411 preflight-passed motions and 60 music IDs. Per-genre motion counts range from 36 to 43, with PO having 43/411 (10.5%). The library does not contain a comparably overwhelming proportion of PO motions. Uniform selection is still not the appropriate expected distribution.', '',
    '## Rejection and transition problems', '',
    '| Song | Accepted retrievals | Completed switches | Terminal hold |',
    '|---|---:|---:|---:|']
for s in data:
    if s['terminal_hold_seconds']:
        lines.append(f"| {s['title']} | {s['accepted']}/57 | {sum(s['completed_plays'].values())} | {s['terminal_hold_seconds']:.1f} s |")
lines += [
    '', 'Bollywood Groove and Battle Ready have 57/57 valid recommendations but never finish a transition. The trace remains in loading until the first terminal state; the logged reason is that bridge preparation did not finish in time. Later recommendations cannot recover them: `prepare()` returns immediately when `self.held` is set (`realtime_music_humanoid_matcher_v2.py:1181`), and the sample path retains the terminal pose. This is a separate transition/recovery issue, not repeated ONNX selection.', '',
    'Hippety Hop also ends in a preparation-timeout hold. Isolation Waltz, the ambient control and JB5 log infeasible candidate/replay bridges. Their logs identify Hermite position-limit violations; this analysis does not establish whether the underlying cause is endpoint derivatives, available trajectories or the bridge search.', '',
    'The ambient control is rejected in all 57 runtime retrievals, as intended. Its deterministic diagnostic accepts only the first of 30 windows before the consecutive-window gate takes effect; the earlier report\'s 100% conditional motion share for this control refers to that single accepted window. JB5 is rejected as non-dance/ambient in 55/57 live retrievals despite its own track being correctly recognized. Its conditional recommendation percentages are based on only two observations and must not be treated as robust evidence. Hippety Hop and Isolation Waltz also lose 10/57 and 24/57 retrievals to that gate.', '',
    '## Sensitivity and limits', '',
    'The first-pass diagnostic (before any loop) keeps the leading external recommendation at 20.3%, and the leading catalog recommendation at 30.0%. Removing each catalog song and all its associated motions still gives `gPO_sBM_cAll_d12_mPO4_ch02` a 23.7% leading recommendation share (22.6% before looping); however, that motion occurs in only three of the ten leave-music-out songs. Repetition is therefore not explained solely by loops or exact self-matches.', '',
    'The earlier `genre_accepted` field measures the highest-scored style, which can differ from the winning motion\'s genre when a second genre is admitted. Recomputing from actual motion IDs gives PO 34.3% of top recommendations, versus 31.8% for the highest-scored-style field. The accompanying JSON records the motion-derived values.', '',
    'There are only nine external rhythmic songs, one ambient control, ten in-sample controls and one fixed startup seed. All trials share the seeded startup motion by design. The aggregate excludes that initial play, but holds and sparse accepted sets still affect interpretation. Windows are correlated and are not independent sample replicates. There is no human rating of visual similarity or dance quality, and no model/genre-prior ablation has yet been run.', '',
    '## Suggested next changes to test', '',
    '1. Test tag-prior strength and hard genre gating with the same frozen songs. Preserve strong track evidence and measure genre/motion concentration together with compatibility, rather than adding randomness immediately.',
    '2. Investigate loading deadlines and add a controlled recovery path after terminal hold. Recheck Bollywood Groove and Battle Ready, then the full set.',
    '3. Recalibrate the ambient rejection gate using the successful ambient rejection alongside JB5, Hippety Hop and Isolation Waltz, where valid music is often rejected.',
    '4. After those targeted checks, assess feature/catalog similarity on more unseen songs and multiple startup seeds. Compare actual choreography families as well as individual motion IDs.', '',
    'No matcher, model or experiment inputs were modified during this analysis. The raw report is preserved; these files are an interpretive supplement.', '',
    '![Concentration and holds](concentration_and_holds.png)', '',
    '![Per-song recommendations](recommendation_by_song.png)', '']
(OUT / 'interpretation.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'songs'}, indent=2))
print('Analysis saved:', OUT)
