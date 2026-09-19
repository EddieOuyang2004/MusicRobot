from pathlib import Path
import sys, pickle, json, hashlib, argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'realtime/humanoid_robot'
sys.path.insert(0,str(BASE/'src'))
import gmr_retarget_smpl_headless as runner
source=Path(runner.__file__).read_text(encoding='utf-8')
needle='        candidate[7:] = previous_joints + safe_amount * joint_delta'
replacement="""        TRACE_EVENTS.append(dict(frame=int(frame_index), pass_number=int(stabilization_passes),
            safe_amount=float(safe_amount), requested_max_joint_step=float(np.max(np.abs(joint_delta))),
            resulting_max_joint_step=float(safe_amount*np.max(np.abs(joint_delta)))))
"""+needle
assert source.count(needle)==1
exec(compile(source.replace(needle,replacement),str(runner.__file__),'exec'),runner.__dict__)
runner.TRACE_EVENTS=[]
mid=sys.argv[1]
old_path=BASE/'data/aistpp_gmr'/f'{mid}.pkl'
old=pickle.load(old_path.open('rb'))
out=ROOT/'output/gmr_limit_analysis'/mid
out.mkdir(parents=True,exist_ok=True)
sys.argv=['trace','--gmr-root',str(BASE/'.deps/GMR'),'--motion',str(BASE/'data/aistpp/motions'/f'{mid}.pkl'),
 '--smpl-model-path',str(BASE/'assets/body_models/smpl'),
 '--output',str(out/'traced.pkl'),'--motion-fps',str(old['fps']),'--source-motion-id',mid,
 '--retargeter-version',old['retargeter_version'],'--source-sha256',old['source_sha256'],
 '--smpl-model-sha256',old['smpl_model_sha256']]
args=runner.parse_args()
print('Starting instrumented retarget: '+mid,flush=True)
new=runner.retarget(args)
errors={k:float(np.max(np.abs(new[k]-old[k]))) for k in ['root_pos','root_rot','dof_pos']}
matched=all(v<1e-8 for v in errors.values())
# Only confirm holds when a nonzero proposed step was suppressed by collision
# and the final saved trajectory also has all joints stationary.
with (out/'traced.pkl').open('wb') as handle: pickle.dump(new,handle)
v=np.diff(new['dof_pos'],axis=0)*new['fps']
pauses=sorted({e['frame'] for e in runner.TRACE_EVENTS if e['requested_max_joint_step']>1e-8
 and e['resulting_max_joint_step']*old['fps']<1e-6
 and np.max(np.abs(v[e['frame']-1]))<1e-6})
report=dict(motion_id=mid,target_sha256=hashlib.sha256((out/'traced.pkl').read_bytes()).hexdigest(),
 replay_matches_saved=True,original_matches_replay=matched,max_abs_errors_vs_original=errors,fps=old['fps'],
 confirmed_pause_frames=pauses,events=runner.TRACE_EVENTS,
 definition='Collision filter suppressed a nonzero proposed joint step; all final joint speeds <1e-6 rad/s. Root excluded.')
(out/'collision_trace.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(motion=mid,matched=matched,errors=errors,confirmed_pause_frames=len(pauses),events=len(runner.TRACE_EVENTS))),flush=True)
print('Trace applies to traced.pkl, not the original dataset file.',flush=True)
