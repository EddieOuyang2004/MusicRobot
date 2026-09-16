import csv,json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('realtime/humanoid_robot/src').resolve()))
import realtime_music_humanoid_dancer as base
from motion_bridges import AuthoredTrajectory
orig=list(csv.DictReader(open('tmp/v2_chronos_gui.csv')))
player=base.MujocoHumanoidPlayer(base.DEFAULT_MODEL,False,True)
print('ORIGINAL_SPIKES_SOURCE')
cache={}
for row in sorted(orig,key=lambda r:float(r['output_max_joint_acceleration_rad_s2'] or 0),reverse=True)[:12]:
 mid=row['current_motion_id']
 if mid not in cache:
  p=pickle.load(open(Path('realtime/humanoid_robot/data/aistpp_gmr')/(mid+'.pkl'),'rb'))
  cache[mid]=(p,AuthoredTrajectory(p['dof_pos'],p['fps']))
 p,tr=cache[mid];i=round(float(row['current_phase'])*(len(p['dof_pos'])-1));q=np.asarray(p['dof_pos']);j=int(np.argmax(abs(q[min(i+1,len(q)-1)]-2*q[i]+q[max(i-1,0)])))
 s=tr.at_time(float(row['current_phase'])*tr.duration)
 print(row['audio_time_seconds'],mid,'frame',i,p['dof_names'][j],np.round(q[max(0,i-2):i+3,j],5).tolist(),'C2_max_accel',round(float(np.max(abs(s.acceleration))),1),'rate',row['speed_multiplier'])
print('SAVED_TRACE_BRIDGES')
print('max_accel_executing',max(float(r['output_max_joint_acceleration_rad_s2']) for r in orig if r['bridge_preparation_status']=='executing'))
print('modified_executing',sum(int(r['bridge_output_modified']) for r in orig if r['bridge_preparation_status']=='executing'))
print('modified_all',sum(int(r['bridge_output_modified']) for r in orig))
pth=Path('tmp/chronos_smoothness_display.npz')
if not pth.exists(): sys.exit(0)
z=np.load(pth);rows=list(csv.DictReader(open('tmp/chronos_smoothness_replay.csv')))
print('CAPTURE',len(rows),z['qpos'].shape)
# set_frame is called once per control iteration; dense trace records every iteration.
rt=np.array([float(r['wall_time_seconds']) for r in rows]); wt=z['perf_time']-z['perf_time'][0]+rt[0]
nearest=np.clip(np.searchsorted(rt,wt),0,len(rt)-1)
nearest=np.where((nearest>0)&(abs(rt[np.maximum(nearest-1,0)]-wt)<abs(rt[nearest]-wt)),nearest-1,nearest)
rows=[rows[i] for i in nearest]
print('TRACE_ASSOCIATION_MAX_MS',float(np.max(abs(rt[nearest]-wt))*1000))
t=z['perf_time'];q=z['qpos'];names=z['joint_names'];ix=[player.actuator_joint_qpos_ids[str(n)] for n in names]
jq=q[:,ix];dt=np.diff(t);v=np.diff(jq,axis=0)/dt[:,None];acc=np.diff(v,axis=0)/((dt[1:]+dt[:-1])/2)[:,None]
for k in np.argsort(np.max(abs(acc),axis=1))[-12:][::-1]:
 j=int(np.argmax(abs(acc[k])));r=rows[k+2]
 print('FINAL_PEAK',r['audio_time_seconds'],r['current_motion_id'],r['current_phase'],str(names[j]),round(float(acc[k,j]),2),'dt',round(dt[k+1]*1000,2),'status',r['bridge_preparation_status'],'clip',r['bridge_output_modified'])
clipped=abs(jq-z['raw_joints'])>1e-8
print('CLIPPING',int(clipped.sum()),'frames',int(clipped.any(axis=1).sum()),'joints',[(str(n),int(clipped[:,j].sum())) for j,n in enumerate(names) if clipped[:,j].any()])
print('LARGE_GAPS',[(rows[i+1]['audio_time_seconds'],round(float(x*1000),2)) for i,x in enumerate(dt) if x>.03])
flags=[]
for i,qp in enumerate(q):
 player.data.qpos[:]=qp
 if player._has_self_clearance_violation(player.data,0.006): flags.append(i)
print('COLLISION_FLAGS',len(flags),'bridge',sum(rows[i]['bridge_preparation_status']=='executing' for i in flags))
print('COLLISION_WINDOWS')
windows=[]
for i in flags:
 if not windows or i>windows[-1][-1]+1: windows.append([i])
 else: windows[-1].append(i)
for w in windows:
 print(rows[w[0]]['audio_time_seconds'],rows[w[-1]]['audio_time_seconds'],len(w),rows[w[0]]['current_motion_id'],rows[w[0]]['bridge_preparation_status'])
np.savez_compressed('tmp/chronos_smoothness_audit.npz',collision_indices=flags,clip_mask=clipped,velocity=v,acceleration=acc)
