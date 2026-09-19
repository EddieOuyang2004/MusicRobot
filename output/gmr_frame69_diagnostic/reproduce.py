from pathlib import Path
import sys,json,numpy as np
root=Path.cwd(); src=root/'realtime/humanoid_robot/src';sys.path.insert(0,str(src))
import gmr_state_trajectory as planner
import build_gmr_v2 as builder
out=root/'output/gmr_frame69_diagnostic'
original=planner.plan_state_trajectory
job=json.loads((src.parent/'data/aistpp_gmr_v2/logs/gBR_sBM_cAll_d04_mBR0_ch09.job.json').read_text())
job['output']=str(out/'diagnostic.pkl')
(out/'job.json').write_text(json.dumps(job))
def capture(guide,fps,states,low,high,settings,**kwargs):
    np.savez(out/'guide.npz',qpos=guide,fps=fps,low=low,high=high)
    # Run only far enough to reproduce the reported failure.
    settings=dict(settings,_debug=True)
    return original(guide[:100],fps,states,low,high,settings,**kwargs)
planner.plan_state_trajectory=capture
# No artifact is allowed to be written from this truncated diagnostic run.
try:
    builder.worker(out/'job.json')
except Exception as exc:
    print(type(exc).__name__+': '+str(exc),flush=True)
