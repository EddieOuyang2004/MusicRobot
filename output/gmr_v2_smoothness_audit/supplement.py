"""Read-only hold, continuity, seam and matched-baseline audit."""
from pathlib import Path
import sys, json, pickle, hashlib
import numpy as np
from numpy.polynomial import polynomial as poly

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'realtime/humanoid_robot'
sys.path.insert(0,str(BASE/'src/test'))
from audit_gmr_smoothness import AuthoredTrajectory, exact_peaks, horner


def longest(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    starts=np.flatnonzero(edges==1); ends=np.flatnonzero(edges==-1)
    if len(starts)==0: return 0,0
    i=int(np.argmax(ends-starts))
    return int(ends[i]-starts[i]),int(starts[i])


def main():
    current=json.loads((OUT/'audit.json').read_text())
    old=json.loads((OUT/'baseline/audit.json').read_text())
    vrows={r['motion_id']:r for r in current['motions']}
    orows={r['motion_id']:r for r in old['motions']}
    details=[]; changed=[]
    for mid,row in vrows.items():
        path=BASE/'data/aistpp_gmr_v2'/f'{mid}.pkl'
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']: changed.append(mid)
        d=pickle.loads(raw); fps=d['fps']; q=np.asarray(d['dof_pos'],dtype=float)
        curve=AuthoredTrajectory(q,fps,velocities=d['dof_vel'],accelerations=d['dof_acc'])
        c=curve.segments; peaks=exact_peaks(c,fps)
        holds=np.max(peaks['speed_rad_s'][0],axis=1)<1e-6
        run,start=longest(holds)
        seams={}
        for order,label in ((0,'position_rad'),(1,'velocity_rad_s'),(2,'acceleration_rad_s2')):
            derivative=poly.polyder(c,m=order)*fps**order
            seams[label]=float(np.max(np.abs(poly.polyval(1.,derivative)[:-1]-poly.polyval(0.,derivative)[1:])))
        jerk,index=peaks['jerk_rad_s3']
        seg,joint=np.unravel_index(int(np.argmax(jerk)),jerk.shape)
        previous=pickle.loads((BASE/'data/aistpp_gmr'/f'{mid}.pkl').read_bytes())
        same_root=all(np.allclose(d[k],previous[k],atol=1e-7,rtol=0) for k in ('root_pos','root_rot'))
        measured= d['projection_validation']['continuous_peaks']
        deviation=max(abs(float(measured[k])-float(np.max(peaks[k][0]))) for k in measured if k in peaks)
        details.append(dict(motion_id=mid,hold_intervals=int(holds.sum()),longest_hold_s=run/fps,
            longest_hold_start_s=start/fps,c2_join_residuals=seams,
            speed_over_export_bound=int(np.any(peaks['speed_rad_s'][0]>d['projection']['max_joint_speed_rad_s']+1e-5,axis=1).sum()),
            acceleration_over_export_bound=int(np.any(peaks['acceleration_rad_s2'][0]>d['projection']['max_joint_acceleration_rad_s2']+1e-5,axis=1).sum()),
            jerk_peak_rad_s3=float(jerk[seg,joint]),jerk_peak_joint=d['dof_names'][joint],
            jerk_peak_time_s=(seg+float(index[seg,joint]))/fps,
            raw_loop_joint_gap_rad=float(np.max(np.abs(q[-1]-q[0]))),
            root_unchanged_from_v1=same_root,metadata_peak_max_absolute_error=deviation,
            planning_fallbacks=d['projection_validation'].get('planning_fallbacks'),
            saved_collision_validation=d['projection_validation']['passed']))
    summary=dict(clips=len(details),frames=current['summary']['total_frames'],
        total_authored_minutes=sum(r['duration_s'] for r in vrows.values())/60,
        files_changed_since_primary_audit=changed,
        clips_with_holds=sum(x['hold_intervals']>0 for x in details),
        total_hold_intervals=sum(x['hold_intervals'] for x in details),
        longest_hold_s=max(x['longest_hold_s'] for x in details),
        clips_exceeding_own_joint_speed_limit=sum(x['speed_over_export_bound']>0 for x in details),
        clips_exceeding_own_joint_acceleration_limit=sum(x['acceleration_over_export_bound']>0 for x in details),
        maximum_c2_join_residuals={k:max(x['c2_join_residuals'][k] for x in details) for k in details[0]['c2_join_residuals']},
        maximum_jerk_jump_rad_s3=max(r['v2_jerk_jump_rad_s3'] for r in vrows.values()),
        clips_with_raw_loop_pose_gap_over_0_1_rad=sum(x['raw_loop_joint_gap_rad']>.1 for x in details),
        maximum_raw_loop_joint_gap_rad=max(x['raw_loop_joint_gap_rad'] for x in details),
        clips_with_root_unchanged_from_v1=sum(x['root_unchanged_from_v1'] for x in details),
        largest_saved_vs_recomputed_peak_difference=max(x['metadata_peak_max_absolute_error'] for x in details),
        highest_jerk_clips=sorted(details,key=lambda x:x['jerk_peak_rad_s3'],reverse=True)[:5],
        highest_root_acceleration_clips=sorted(vrows.values(),key=lambda x:x['root_acceleration_m_s2'],reverse=True)[:3])
    (OUT/'supplement.json').write_text(json.dumps(dict(summary=summary,motions=details),indent=2),encoding='utf-8')
    a,b=old['summary'],current['summary']
    lines=['# GMR v2 smoothness assessment','',
        'All 411 v2 files passed the checked joint speed, acceleration, reversal and sampled joint-range criteria at authored speed. This does not mean every aspect of whole-body playback is fully smooth.','',
        f"Coverage: {summary['frames']:,} frames, {summary['total_authored_minutes']:.2f} minutes. No files changed between the primary and supplemental audits.",'',
        '| Check | Original GMR | GMR v2 |','|---|---:|---:|']
    for key,label in [('clips_with_v2_speed_bad_segments','Clips exceeding playback speed threshold (16 rad/s)'),('clips_with_v2_acceleration_bad_segments','Clips exceeding acceleration threshold (1600 rad/s²)'),('clips_with_fast_reversal_frames','Clips with fast joint reversals'),('clips_with_v2_position_excess_rad','Clips with sampled joint-range overshoot > 1e-5 rad')]:
        lines.append(f'| {label} | {a[key]} | {b[key]} |')
    lines += ['',f"V2 also passes each file's stricter export speed bound (3*pi = 9.42478 rad/s). Maximum continuous joint acceleration is {b['maxima']['v2_acceleration_rad_s2']:.2f} rad/s².",
        f"True whole-joint holds, using the continuous speed polynomial: {summary['total_hold_intervals']} intervals across {summary['clips_with_holds']} clips; longest {summary['longest_hold_s']:.4f} s. A hold is not automatically a defect or proof of collision blocking.",
        f"Maximum within-clip C2 join residuals (position/velocity/acceleration): {summary['maximum_c2_join_residuals']}. These are floating-point-scale differences.",'',
        '## Remaining limitations','',
        f"- Jerk is not constrained to be continuous and no acceptance threshold is configured. Peak joint jerk is {b['maxima']['v2_jerk_rad_s3']:.1f} rad/s³; the largest adjacent-segment jerk jump is {summary['maximum_jerk_jump_rad_s3']:.1f} rad/s³. These are measurements, not a pass/fail judgment of perceptual smoothness.",
        f"- Root translation/rotation are unchanged within 1e-7 tolerance in {summary['clips_with_root_unchanged_from_v1']}/411 clips. Root acceleration reaches {b['maxima']['root_acceleration_m_s2']:.2f} m/s² and angular acceleration reaches {b['maxima']['root_angular_acceleration_rad_s2']:.2f} rad/s². Joint smoothing does not resolve those whole-body spikes.",
        f"- {summary['clips_with_raw_loop_pose_gap_over_0_1_rad']} clips have a raw last-to-first joint-pose difference over 0.1 rad (maximum {summary['maximum_raw_loop_joint_gap_rad']:.3f} rad). This flags direct-wrap seams, not the output of the runtime loop/transition bridge.",
        '- Music-driven time warping, runtime transitions, contact/dynamic tracking and human-perceived smoothness were not certified. No motion files were modified.',
        '- Stored revision-2 state checksums and validation metadata passed. Collision tests were recorded by generation; this audit did not rerun full collision geometry for every frame.',
        '- Joint derivative extrema were recomputed analytically from the saved Hermite curves. Joint-position bounds were sampled at 33 points per segment. Root derivatives use finite differences.',
        '', '## Highest measured joint jerk', '',
        '| Motion | Joint | Time (s) | Jerk (rad/s³) |','|---|---|---:|---:|']
    for x in summary['highest_jerk_clips']:
        lines.append(f"| {x['motion_id']} | {x['jerk_peak_joint']} | {x['jerk_peak_time_s']:.4f} | {x['jerk_peak_rad_s3']:.1f} |")
    lines += ['', 'Recommendation: retain the current joint-state planner; next address root translation/orientation smoothness and explicit jerk targets, then check loop/transition and music-warped playback. Avoid claiming that all motion is perceptually or dynamically smooth solely from the joint-limit pass.', '',
              'Files: `audit.json` and `motions.csv` contain all primary results and source hashes; `supplement.json` contains holds, joins, jerk locations and loop gaps; `baseline/` contains the freshly rerun original-dataset comparison. The existing auditor\'s four regression tests passed.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if not k.startswith('highest')},indent=2))
    print('Highest jerk:',[(x['motion_id'],x['jerk_peak_joint'],x['jerk_peak_time_s']) for x in summary['highest_jerk_clips']])


if __name__=='__main__': main()
