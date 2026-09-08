"""Offline-only BAS reference-frame correction after the expensive feature pass.

All non-BAS metrics and the SMPL feature bundle are preserved. Revision hashes
make this selective recalculation explicit; no experiment is launched.
"""
from pathlib import Path
import csv
import numpy as np
import humanoid_matcher_experiment_metrics as metrics
import reanalyse_thesis_experiments as reanalysis
from run_thesis_supplement import read_json, atomic_json
from run_humanoid_matcher_experiments import file_sha256, atomic_replace


def corrected_beats(path: Path, causal: bool):
    with np.load(path, allow_pickle=False) as data:
        times = data['wall_time_seconds' if causal else 'time_seconds']
        keep = times >= 6
        times = times[keep]
        bodies = data['body_positions'][keep]
        if 'body_names' in data and str(data['body_names'][0]) == 'world':
            bodies = bodies[:, 1:]
        music = data['beat_events'][:, 1] if causal else data['accepted_causal_beat_times_seconds']
        music = music[music >= 6]
    grid = np.arange(times[0], times[-1] + 1e-9, 1 / 60)
    flat = bodies.reshape(len(times), -1)
    sampled = np.column_stack([np.interp(grid, times, flat[:, j]) for j in range(flat.shape[1])])
    dance = metrics.detect_kinematic_beats(sampled.reshape(len(grid), -1, 3), fps=60) + grid[0]
    result = metrics.bidirectional_beat_metrics(music, dance)
    if not len(music):
        for key in ('bas_music_to_dance','bas_dance_to_music','bas_harmonic',
                    'beat_timing_mae_seconds','beat_timing_p95_seconds',
                    'missed_music_beat_rate','extra_dance_beat_rate'):
            result[key] = None
    return result


def main():
    root = reanalysis.OUTPUT
    path, marker = root/'consolidated_results.json', root/'READY_FOR_THESIS'
    previous = read_json(marker if marker.exists() else marker.with_name('READY_FOR_THESIS.before_beat_reference'))
    if file_sha256(path) != previous['results_sha256']:
        raise ValueError('Input consolidated checksum mismatch')
    data = read_json(path)
    if data.get('beat_reference_revision') == 'pelvis_excluding_world_v1':
        print('BAS root-reference revision already complete')
        return
    if marker.exists():
        atomic_replace(marker, marker.with_name('READY_FOR_THESIS.before_beat_reference'))
    for cohort, suites in data['cohorts'].items():
        for suite, group in suites.items():
            print(f'BAS reference refresh: {cohort}/{suite}', flush=True)
            for run in group['runs']:
                beat = corrected_beats(Path(run['pose']['pose_npz']), cohort=='online_causal')
                run['pose']['beat'] = beat
                run['pose']['beat_reference'] = 'pelvis-relative model-body speeds; world body excluded'
                run['metrics']['bas_harmonic'] = beat['bas_harmonic']
                run['metrics']['bas_music_to_dance'] = beat['bas_music_to_dance']
            group['statistics'] = reanalysis.source_statistics(group['runs'])
    data['beat_reference_revision'] = 'pelvis_excluding_world_v1'
    data['analysis_revision'] = {'base_consolidated_sha256':previous['results_sha256'],
                               'refresh_script_sha256':file_sha256(Path(__file__)),
                               'metrics_sha256':file_sha256(Path(metrics.__file__)),
                               'scope':'BAS and its source-level statistics only; raw data and SMPL features unchanged'}
    atomic_json(path, reanalysis.clean_json(data))
    rows = [{'cohort':cohort, 'suite':suite,'condition':run['condition'],'source_id':run['source_id'],
             'seed':run['seed'],**run['metrics']}
            for cohort,suites in data['cohorts'].items() for suite,group in suites.items() for run in group['runs']]
    target = root/'consolidated_summary.csv'
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    atomic_replace(temporary,target)
    atomic_json(marker,{**previous,'results_sha256':file_sha256(path),'beat_reference_revision':data['beat_reference_revision']})
    print('BAS reference refresh complete')


if __name__=='__main__':
    main()
