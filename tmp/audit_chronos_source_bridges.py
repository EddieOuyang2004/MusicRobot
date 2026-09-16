"""Reconstruct saved-run bridges and inspect GMR plateaus offline."""
import sys,csv,json,pickle
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('realtime/humanoid_robot/src').resolve()))
import realtime_music_humanoid_matcher_v2 as m
from motion_bridges import HermiteBridge
args=m.parse_args(); catalog=m.MusicCatalog.load(args.catalog)
player=m.base.MujocoHumanoidPlayer(args.model,False,True)
rows=list(csv.DictReader(open('tmp/v2_chronos_gui.csv')));cache={};result={'clips':{},'bridges':[]}
def get(mid):
 if mid in cache:return cache[mid]
 s=m.load_motion_sampler(args,catalog,catalog.motions[mid]);a=m.base.make_pose_adapter(args,player,s);player.ground_sampler(s,a)
 s.entry_features=m.build_motion_entry_features(s,catalog.motions[mid],a,player.actuator_joint_ranges);cache[mid]=s;return s
for mid in dict.fromkeys(r['current_motion_id'] for r in rows):
 s=get(mid); q=s.entry_features.authored.positions;delta=np.max(abs(np.diff(q,axis=0)),axis=1)
 windows=[]
 for i in np.flatnonzero(delta<0.0001):
  if not windows or i>windows[-1][-1]+1:windows.append([int(i)])
  else:windows[-1].append(int(i))
 plateaus=[]
 for w in windows:
  if len(w)<4:continue
  lo,hi=w[0],w[-1]+1
  rr=[r for r in rows if r['current_motion_id']==mid and lo<=float(r['current_phase'])*(len(q)-1)<=hi and r['bridge_preparation_status']!='executing']
  plateaus.append({'frames':[lo,hi],'duration_s':len(w)/s.fps,'audio_window': [rr[0]['audio_time_seconds'],rr[-1]['audio_time_seconds']] if rr else None})
 result['clips'][mid]={'plateaus':plateaus}
for r in rows:
 if r['event']!='switch_start':continue
 src=get(r['current_motion_id']);dst=get(r['transition_motion_id']); sf=src.entry_features;df=dst.entry_features
 i=int(r['exit_frame_index']);j=int(r['entry_frame_index']);duration=float(r['transition_duration_seconds']);bridge=HermiteBridge.between(sf.authored.state(i),df.authored.state(j),duration)
 assert sf.joint_names==df.joint_names
 source=m.neutral_authored_frame(src,i/sf.fps);rawtarget=m.neutral_authored_frame(dst,j/df.fps)
 target=m.align_motion_frame_root(rawtarget,source_reference=rawtarget,target_reference=source)
 flags=[];minimum=1.;pair=None
 for t in np.linspace(0,duration,351):
  f=m.quintic_blend(source,target,t/duration).with_joint_positions(dict(zip(sf.joint_names,bridge.at_time(t).position)))
  player.set_frame(f);m.base.mujoco.mj_forward(player.model,player.data)
  if player._has_self_clearance_violation_after_forward(player.data,.006):flags.append(float(t))
  for a,b in player.self_collision_geom_pairs:
   if np.linalg.norm(player.data.geom_xpos[a]-player.data.geom_xpos[b])-player.model.geom_rbound[a]-player.model.geom_rbound[b]>=.0060001:continue
   d=m.base.mujoco.mj_geomDistance(player.model,player.data,int(a),int(b),.006,None)
   if d<minimum:minimum=float(d);pair=[m.base.mujoco.mj_id2name(player.model,m.base.mujoco.mjtObj.mjOBJ_BODY,int(player.model.geom_bodyid[g])) for g in (a,b)]
 result['bridges'].append({'audio_start':float(r['audio_time_seconds']),'source':r['current_motion_id'],'target':r['transition_motion_id'],'clearance_violating_samples':len(flags),'samples':351,'minimum_distance_capped_at_6mm':minimum,'pair':pair,'violation_bridge_time_range': [min(flags),max(flags)] if flags else None})
Path('tmp/chronos_smoothness_offline.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
