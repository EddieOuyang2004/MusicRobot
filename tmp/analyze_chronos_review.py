"""Analyze captured output and source motion, without modifying playback."""
import csv,json,pickle,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'realtime/humanoid_robot/src'
sys.path[:0]=[str(SRC),str(SRC/'test')]
from audit_gmr_smoothness import audit
from verify_humanoid_matcher_v2_trace import verify_trace
prefix=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'tmp/chronos_review')
motion_root=ROOT/(sys.argv[2] if len(sys.argv)>2 else 'realtime/humanoid_robot/data/aistpp_gmr')
trace=list(csv.DictReader(prefix.with_suffix('.csv').open()))
rows=list(csv.DictReader(Path(str(prefix)+'_frames.csv').open()))
z=np.load(str(prefix)+'_display.npz')
t=z['perf_time'];t=t-t[0];q=z['qpos'][:,z['qpos_ids']]
names=list(z['joint_names']);dt=np.diff(t)
v=np.diff(q,axis=0)/dt[:,None]
a=np.diff(v,axis=0)/((dt[1:]+dt[:-1])/2)[:,None]
step=np.max(abs(np.diff(q,axis=0)),axis=1);amax=np.max(abs(a),axis=1)
clipped=abs(q-z['raw_joints'])>1e-8
assert len(rows)==len(q)
ranges={e.attrib['name'].removesuffix('_joint'):tuple(map(float,e.attrib['range'].split())) for e in ET.parse(SRC.parent/'assets/g1_29dof.xml').iter('joint') if 'range' in e.attrib and 'name' in e.attrib}
result={'frames':len(q),'capture_duration_s':float(t[-1]),'verification':verify_trace(trace),'clip_frames':int(clipped.any(axis=1).sum()),'clipped_joint_samples':int(clipped.sum()),'clip_joints':{n:int(clipped[:,i].sum()) for i,n in enumerate(names) if clipped[:,i].any()},'max_clip_correction_rad':float(np.max(abs(q-z['raw_joints']))),'interval_ms':{str(p):float(np.percentile(dt*1000,p)) for p in (50,95,99,100)},'large_gaps':[(float(t[i+1]),float(x*1000)) for i,x in enumerate(dt) if x>.03],'max_joint_step_rad':float(step.max()),'max_speed_rad_s':float(np.max(abs(v))),'max_acceleration_rad_s2':float(amax.max()),'sources':{},'peaks':[],'transitions':[]}
def windows(indices):
    out=[]
    for i in indices:
        if not out or i>out[-1][-1]+1:out.append([int(i)])
        else:out[-1].append(int(i))
    return out
sources={}
for mid in dict.fromkeys(r['motion_id'] for r in rows):
    p=motion_root/(mid+'.pkl')
    d=pickle.loads(p.read_bytes());sources[mid]=d
    sq=np.asarray(d['dof_pos']);fps=d['fps'];plateaus=[]
    for w in windows(np.flatnonzero(np.max(abs(np.diff(sq,axis=0)),axis=1)<1e-4)):
        if len(w)<4:continue
        lo,hi=w[0]/fps,(w[-1]+1)/fps
        selected=[i for i,r in enumerate(rows) if r['motion_id']==mid and r['status']!='executing' and lo<=float(r['authored_seconds'])<=hi]
        plateaus.append({'frames':[w[0],w[-1]+1],'source_seconds':[lo,hi],'capture_windows':[[float(t[w[0]]),float(t[w[-1]])] for w in windows(selected)]})
    result['sources'][mid]={'audit':audit(p,ranges),'plateaus':plateaus,'pipeline_version':d.get('pipeline_version'),'continuity_limits':d.get('continuity_limits')}
for k in np.argsort(amax)[-12:][::-1]:
    i=int(k+2);r=rows[i];j=int(np.argmax(abs(a[k])));d=sources[r['motion_id']]
    fi=int(round(float(r['authored_seconds'])*d['fps']));sj=list(d['dof_names']).index(names[j])
    result['peaks'].append(dict(capture_seconds=float(t[i]),motion_id=r['motion_id'],status=r['status'],joint=names[j],acceleration=float(a[k,j]),dt_ms=float(dt[i-1]*1000),source_frame=fi,source_samples=np.asarray(d['dof_pos'])[max(fi-2,0):fi+3,sj].tolist()))
for i,r in enumerate(rows):
    if r['event'] not in ('switch_start','switch_complete'):continue
    sl=slice(max(i-2,0),min(i+3,len(amax)))
    result['transitions'].append({'capture_seconds':float(t[i]),'event':r['event'],'motion_id':r['motion_id'],'nearby_max_acceleration_rad_s2':float(amax[sl].max()),'nearby_max_step_rad':float(step[max(0,i-2):i+2].max()),'nearby_clipped_frames':int(clipped[max(0,i-2):i+3].any(axis=1).sum())})
executing=np.array([r['status']=='executing' for r in rows])
result['bridge_clipped_frames']=int(np.sum(clipped.any(axis=1)&executing))
result['bridge_max_acceleration_rad_s2']=float(amax[executing[2:]&executing[1:-1]&executing[:-2]].max())
result['trace_events']={e:sum(r['event']==e for r in trace) for e in ('switch_start','switch_complete','bridge_hold')}
Path(str(prefix)+'_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:val for k,val in result.items() if k not in ('sources','transitions')},indent=2))
print('SOURCE_SUMMARY')
for mid,s in result['sources'].items():
    au=s['audit']
    print(mid,json.dumps({k:au[k] for k in ('raw_step_rad','v2_acceleration_rad_s2','v2_position_excess_rad','fast_reversal_frames')}),json.dumps(s['plateaus']))
sys.path.insert(0,str(ROOT/'tmp/thesis_plot_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(4,1,figsize=(13,9),sharex=True,layout='constrained')
axes[0].plot(t[1:],np.rad2deg(step),lw=.65);axes[0].set_ylabel('Max joint step\n(degrees)')
axes[1].plot(t[2:],amax,lw=.65);axes[1].set_ylabel('Joint acceleration\n(rad/s²)')
axes[2].plot(t,clipped.sum(axis=1),lw=.8);axes[2].set_ylabel('Clipped joints')
axes[3].plot(t[1:],dt*1000,lw=.65);axes[3].axhline(1000/60,ls='--',color='gray');axes[3].set_ylabel('Output interval\n(ms)')
for ax in axes:
    for w in windows(np.flatnonzero(executing)):ax.axvspan(t[w[0]],t[w[-1]],alpha=.15,color='green')
    ax.grid(alpha=.2)
axes[3].set_xlabel('Seconds since first captured output (green: transition bridges)')
fig.suptitle('Chronos | authored timing | seed 42 | GUI + audio\nActual player poses; finite differences use measured output intervals')
fig.savefig(str(prefix)+'_continuity.png',dpi=160)
