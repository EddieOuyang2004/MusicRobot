"""Export Chapter 6 numbers/tables/figures from the verified consolidated result.

Run with an environment providing NumPy and Matplotlib; experiments are never
launched. Figure distributions use runs or source means, never IID frames.
"""
from pathlib import Path
import hashlib
import json
import csv

ROOT = Path(__file__).resolve().parents[3]
THESIS = ROOT / 'docs/thesis'
RESULT = THESIS / 'experiment_results/reanalysis_rescue_60_v5'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def number(value, places=3):
    if value is None or not np.isfinite(value):
        return '--'
    if abs(value) >= 10000:
        return f'{value:.2e}'
    return f'{value:.{places}f}'


def table(name, columns, header, rows):
    text = '\\begin{tabular}{' + columns + '}\n\\toprule\n'
    text += ' & '.join(header) + ' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str, row)) + ' \\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    (TABLES / (name + '.tex')).write_text(text, encoding='utf-8')


def subset(cohort, suite, condition='full'):
    return [r for r in DATA['cohorts'][cohort][suite]['runs'] if r['condition'] == condition]


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def timing(rows, key):
    return median([r['stage_timing'].get(key) for r in rows])


def save(name):
    plt.savefig(FIGURES / (name + '.png'), dpi=240, bbox_inches='tight')
    plt.close()


def main():
    global DATA, TABLES, FIGURES
    data_path = RESULT / 'consolidated_results.json'
    marker = load(RESULT / 'READY_FOR_THESIS')
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if marker['results_sha256'] != digest:
        raise ValueError('Consolidated result checksum mismatch')
    DATA = load(data_path)
    TABLES = RESULT / 'chapter6_tables'
    FIGURES = THESIS / 'figures'
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    full = subset('online_causal', 'short')
    authored = subset('online_causal', 'short', 'authored_timing')
    long = subset('online_causal', 'long')
    groups = [('Causal full', full), ('Causal authored', authored), ('Causal long', long)]
    control_hz = float(full[0]['stage_timing']['control_rate_hz'])
    control_budget_ms = 1000.0 / control_hz
    macros = {}
    def macro(name, value, places=3):
        macros[name] = number(value, places)
    offline = DATA['offline_original']
    rank = offline['quality']['strict_leave_one_music']['ranking']
    reject = offline['audio_evaluation']['rejection']
    metrics = [('Recall@1', 'recall_at_1', 'Precision', 'precision'),
               ('Recall@3', 'recall_at_3', 'Recall', 'recall'),
               ('Recall@5', 'recall_at_5', '$F_1$', 'f1'),
               ('Macro-$F_1$', 'macro_f1', 'AUROC', 'auroc'),
               ('MRR', 'mean_reciprocal_rank', 'False rejection', 'false_reject_rate'),
               ('NDCG@5', 'ndcg_at_5', 'Missed rejection', 'missed_reject_rate')]
    table('retrieval', 'lr|lr', ['Retrieval metric', 'Value', 'Rejection metric', 'Value'],
          [[a, number(rank[b]), c, number(reject[d])] for a,b,c,d in metrics])
    macro('RecallOne',rank['recall_at_1']); macro('RecallFive',rank['recall_at_5'])
    macro('RejectAuroc',reject['auroc'])
    table('genre', 'lrrr', ['Genre', '$N$', 'Recall@1', 'Recall@5'],
          [[key, value['queries'], number(value['recall_at_1']), number(value['recall_at_5'])]
           for key,value in rank['per_genre'].items()])
    table('cczero', 'lrr', ['CC0 input', 'Genre hit rate', 'Rejection rate'],
          [[name.split('.')[0].replace('_', r'\_'), number(value.get('genre_hit_rate')), number(value['rejection_rate'])]
           for name,value in offline['audio_evaluation']['files'].items() if name[:2].isdigit()])
    rows=[]
    for label, runs in groups:
        key = 'online_causal'
        suite = 'long' if label.endswith('long') else 'short'
        condition = 'authored_timing' if 'authored' in label else 'full'
        summaries = DATA['cohorts'][key][suite]['statistics']['summaries']
        def ci(metric):
            item = summaries.get(f'{condition}:{metric}',{}).get('bootstrap_95_ci')
            if not item: return '--'
            return f"{number(item['mean'])} [{number(item['low'])}, {number(item['high'])}]"
        rows.append([label, len(runs), ci('bas_harmonic'), ci('pfc_edge_g1_adapted_30fps')])
    table('rhythm', 'lrlr', ['Condition', '$N$', r'Harmonic BAS [95\% CI]', r'G1 PFC [95\% CI]'], rows)
    f = DATA['feature_metrics']['seed_summary']
    table('features', 'lrrrr', ['Scale', 'FID$_k$', 'FID$_g$', 'Div$_k$', 'Div$_g$'],
          [[scale.replace('_',' ')] + [number(f[scale][metric]['mean']) + r' $\pm$ ' + number(f[scale][metric]['std'])
                                    for metric in ('fid_k','fid_g','div_k','div_g')]
           for scale in ('raw','real_standardised')])
    table('groundtruth_div', 'lrr', ['Scale', 'Ground-truth Div$_k$', 'Ground-truth Div$_g$'],
          [[scale.replace('_',' '),number(f[scale]['ground_truth_div_k']['mean']),number(f[scale]['ground_truth_div_g']['mean'])]
           for scale in ('raw','real_standardised')])
    table('safety', 'lrrrrr', ['Causal condition', '$N$', 'Max speed', 'Max accel.', 'Residual frames', 'Limit frames'],
          [[label,len(runs),number(max(r['safety']['final_speed_max_rad_s'] for r in runs),1),
            number(max(r['safety']['final_acceleration_max_rad_s2'] for r in runs),1),
            sum(r['safety']['final_residual_clearance_violations'] for r in runs),
            sum(r['safety']['final_joint_limit_violations'] for r in runs)]
           for label,runs in groups])
    table('timing', 'lrrrrr', ['Condition', 'Query p95', 'Work p99', 'Miss ratio', 'p99 fails', 'Miss fails'],
          [[label,number(timing(runs,'retrieval_waveform_to_match_ms_p95'),1),
            number(timing(runs,'work_ms_p99'),2),number(timing(runs,'deadline_miss_ratio')),
            f"{sum(r['stage_timing']['work_ms_p99'] > 1000/float(r['stage_timing']['control_rate_hz']) for r in runs)}/{len(runs)}",
            f"{sum(r['stage_timing']['deadline_miss_ratio'] >= .01 for r in runs)}/{len(runs)}"] for label,runs in groups])
    stages = [('Audio copy','retrieval_audio_copy'),('Audio preprocessing','retrieval_audio_preprocess'),
              ('Hand-crafted features','retrieval_handcrafted_features'),('ONNX inference','retrieval_onnx_inference'),
              ('Track retrieval','retrieval_track_retrieval'),('Motion ranking','retrieval_motion_ranking'),
              ('Selection policy','selection_policy'),('Waveform to match','retrieval_waveform_to_match'),
              ('Motion file load','motion_load'),('Grounding','motion_grounding'),('Entry features','entry_feature'),
              ('Entry scoring','entry_score'),('Submission to ready','motion_fully_ready'),
              ('Audio/beat control stage','control_audio_beat_analysis'),('Phase controller','control_phase_controller'),
              ('Pose sampling','control_pose_sampling'),('Transition/root blending','control_transition_root_blend'),
              ('Limiter/collision','control_limiter_collision'),('MuJoCo forward','control_mujoco_forward'),
              ('Final-output safety audit','control_experiment_final_safety_audit')]
    table('stages', 'lrrrr', ['Stage (causal full short)', 'p50', 'p95', 'p99', 'Worst max'],
          [[label]+[number(timing(full,key+'_ms_'+percentile),3) for percentile in ('p50','p95','p99')]
           + [number(max((r['stage_timing'][key+'_ms_max'] for r in full if key+'_ms_max' in r['stage_timing']), default=None),3)]
           for label,key in stages])
    table('readiness', 'lrrr', ['Causal condition', 'Underruns', 'Runs with hold', 'Hold events'],
          [[label,sum(r['stage_timing']['ready_pool_underruns'] for r in runs),
            sum(r['stage_timing']['hold_last_events'] > 0 for r in runs),
            sum(r['stage_timing']['hold_last_events'] for r in runs)] for label,runs in groups])
    table('startup', 'lr', ['Startup stage', 'Median ms'],
          [[label,number(timing(full,key),1)] for label,key in [
              ('ONNX session','startup_onnx_session_ms'),('Catalogue','startup_catalog_load_ms'),
              ('Matcher initialisation','startup_matcher_init_ms'),('MuJoCo initialisation','startup_mujoco_init_ms'),
              ('Initial grounding','startup_initial_grounding_ms'),('Until control loop','startup_until_control_loop_ms')]])
    comparisons=[]
    for cohort,suite,condition,metric,label in [
        ('online_causal','short','authored_timing','bas_harmonic','Causal authored: BAS'),
        ('online_causal','short','authored_timing','deadline_miss_ratio','Causal authored: miss ratio')]:
        test=DATA['cohorts'][cohort][suite]['statistics']['paired_tests'].get(condition+':'+metric)
        comparisons.append([label,test['pairs'],number(test['mean_difference']),number(test['p_value'],4),number(test['holm_adjusted_p_value'],4)])
    table('ablation', 'lrrrr', ['Ablation minus full', 'Sources', r'$\Delta$', '$p$', r'$p_{\rm Holm}$'], comparisons)
    tests=DATA['cohorts']['online_causal']['short']['statistics']['paired_tests']
    for name,condition in [('Full','full'),('Authored','authored_timing')]:
        summary=DATA['cohorts']['online_causal']['short']['statistics']['summaries'][condition+':bas_harmonic']['bootstrap_95_ci']
        macro(name+'Bas',summary['mean'])
    test=tests['authored_timing:bas_harmonic']
    macro('BasGain',-test['mean_difference'],4)
    macro('BasGainLow',-test['difference_bootstrap_95_ci']['high'],4)
    macro('BasGainHigh',-test['difference_bootstrap_95_ci']['low'],4)
    macro('BasHolm',test['holm_adjusted_p_value'],4)
    macro('MissDifference',tests['authored_timing:deadline_miss_ratio']['mean_difference'],4)
    macro('MissHolm',tests['authored_timing:deadline_miss_ratio']['holm_adjusted_p_value'],4)
    macro('GeomDivRatio',f['raw']['div_g']['mean']/f['raw']['ground_truth_div_g']['mean'])
    for label,runs in groups:
        prefix='Long' if label.endswith('long') else ('Authored' if 'authored' in label else 'Full')
        for suffix,key in [('Work','work_ms_p99'),('Miss','deadline_miss_ratio'),('Query','retrieval_waveform_to_match_ms_p95')]:
            macro(prefix+suffix,timing(runs,key))
        macro(prefix+'Speed',max(r['safety']['final_speed_max_rad_s'] for r in runs),1)
        macro(prefix+'Residual',sum(r['safety']['final_residual_clearance_violations'] for r in runs),0)
    (RESULT/'chapter6_numbers.tex').write_text('% Generated from consolidated result SHA-256 '+digest+'\n'+
        '\n'.join('\\newcommand{\\ChSix'+name+'}{'+value+'}' for name,value in macros.items())+'\n',encoding='utf-8')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for label,runs in groups:
        for ax,key in zip(axes,['retrieval_waveform_to_match_ms_p95','work_ms_p99']):
            values=np.sort([r['stage_timing'][key] for r in runs])
            ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=label)
    axes[0].set(xlabel='Per-run query p95 (ms)',ylabel='Empirical CDF')
    axes[1].axvline(control_budget_ms,color='black',ls='--',lw=1,
                    label=f'{control_budget_ms:.2f} ms target')
    axes[1].set(xlabel='Per-run control-work p99 (ms)',ylabel='Empirical CDF')
    axes[1].legend(fontsize=8,loc='lower right')
    save('chapter6_latency_cdf')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for i,(label,runs) in enumerate(groups):
        for ax,key in zip(axes,['bas_harmonic','normalized_selection_entropy']):
            values=[r['metrics'][key] for r in runs if r['metrics'].get(key) is not None]
            ax.boxplot(values,positions=[i+1],widths=.5,showfliers=False)
    for ax,title in zip(axes,['Harmonic BAS','Selection-episode entropy']):
        ax.set_xticks(range(1,4),['Causal\nfull','Causal\nauthored','Causal\nlong'])
        ax.set_ylabel(title); ax.set_ylim(-.03,1.03)
    save('chapter6_rhythm_diversity')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for label,runs in groups[:2]:
        means={}
        for r in runs:
            means.setdefault(r['source_id'],[]).append(r['trace']['selection']['visual_cluster_coverage'])
        axes[0].plot(range(len(means)),[np.mean(means[key]) for key in sorted(means)],'o-',label=label)
    axes[0].set(xlabel='Input index (sorted source ID)',ylabel='Catalogue cluster coverage')
    axes[0].legend(fontsize=8)
    for label,runs in groups:
        speeds=np.sort([r['safety']['final_speed_max_rad_s'] for r in runs])
        axes[1].step(speeds,np.arange(1,len(speeds)+1)/len(speeds),where='post',label=label)
    axes[1].axvline(16,color='black',ls='--',lw=1)
    axes[1].set(xlabel='Per-run final-output maximum speed (rad/s)',ylabel='Empirical CDF')
    axes[1].legend(fontsize=8)
    save('chapter6_coverage_safety')
    # Fixed illustration: lexical first stitched full run, seed 0, never cherry-picked.
    run=next(r for r in full if r['source_id']=='aistpp_stitched_test' and r['seed']==0)
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
    stages=['first_match_seconds','pending_seconds','switch_start_seconds','blend_50_seconds','stable_switch_seconds']
    for event in run['response']['changes']:
        change=event['change_time_seconds']
        ax.axvline(change,color='.8',lw=1)
        for i,key in enumerate(stages):
            delay=event[key]
            if delay is not None: ax.scatter(change+delay,i,s=35,color=f'C{i}')
    ax.set_yticks(range(5),['New genre match','Pending','Switch start','50% blend','Switch complete'])
    ax.set(xlabel='Actual wall time from audio origin (s)',xlim=(0,40),ylim=(-.5,4.5))
    save('chapter6_response_timeline')
    with Path(run['trace']['trace']).open(encoding='utf-8',newline='') as handle:
        rows=list(csv.DictReader(handle))
    switches=[float(r['wall_time_seconds']) for r in rows if r['event']=='switch_start' and float(r['wall_time_seconds'])>=6]
    if switches:
        onset=switches[0]
        with np.load(run['pose']['pose_npz'],allow_pickle=False) as pose:
            times=pose['wall_time_seconds']; joints=pose['joint_positions'].astype(float)
        velocity=np.diff(joints,axis=0)/np.diff(times)[:,None]
        vt=(times[1:]+times[:-1])/2
        acceleration=np.diff(velocity,axis=0)/np.diff(vt)[:,None]
        at=(vt[1:]+vt[:-1])/2
        jerk=np.diff(acceleration,axis=0)/np.diff(at)[:,None]
        jt=(at[1:]+at[:-1])/2
        fig,axes=plt.subplots(2,1,figsize=(8.5,4.5),sharex=True,layout='constrained')
        for ax,times,values,label in [(axes[0],vt,velocity,'Max joint speed (rad/s)'),
                                      (axes[1],jt,jerk,'Max joint jerk (rad/s³)')]:
            mask=np.abs(times-onset)<=1
            ax.plot(times[mask]-onset,np.maximum(np.max(np.abs(values[mask]),axis=1),1e-6),lw=1)
            ax.axvline(0,color='black',ls='--',lw=1)
            ax.set(yscale='log',ylabel=label)
        axes[0].axhline(16,color='C3',ls=':',label='16 rad/s target');axes[0].legend(fontsize=8)
        axes[1].set_xlabel('Actual wall seconds relative to first post-warm-up switch')
        save('chapter6_switch_dynamics')
    atomic={"consolidated_sha256":digest,"exporter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tables":[p.name for p in TABLES.glob('*.tex')],"figures":[p.name for p in FIGURES.glob('chapter6_*.png')]}
    (RESULT/'chapter6_export_manifest.json').write_text(json.dumps(atomic,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(atomic,indent=2))


if __name__=='__main__':
    main()
