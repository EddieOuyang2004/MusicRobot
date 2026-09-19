from pathlib import Path
import json,pickle,sys
import numpy as np
import mujoco
root=Path.cwd(); src=root/'realtime/humanoid_robot/src'; sys.path.insert(0,str(src))
from motion_bridges import AuthoredTrajectory,HermiteBridge,JointState
from gmr_collision_projection import collision_states
from gmr_retarget_smpl_headless import activate_required_collision_geoms,_violates_configured_clearance
base=src.parent
models=(mujoco.MjModel.from_xml_path(str(base/'.deps/GMR/assets/unitree_g1/g1_mocap_29dof.xml')),mujoco.MjModel.from_xml_path(str(base/'assets/open_humanoid_dancer.xml')))
activate_required_collision_geoms(models[0]); states=collision_states(models)
reports=[]
for name in ['gWA_sBM_cAll_d26_mWA0_ch07','gBR_sBM_cAll_d05_mBR0_ch08']:
    with (base/'data/aistpp_gmr_v2'/f'{name}.pkl').open('rb') as f: p=pickle.load(f)
    q=np.asarray(p['dof_pos'],dtype=np.float32).astype(float); fps=p['fps']; trajectory=AuthoredTrajectory(q,fps)
    report=dict(motion=name,frames=len(q),fps=fps,interpolation='actual AuthoredTrajectory quintic; float32 sampler positions',mujoco_version=mujoco.__version__,model_count=len(models),samples_per_interval=81,
        finite_difference_peak_speed=float(np.max(np.abs(np.diff(q,axis=0)*fps))),speed_limit=p['continuity_limits']['max_joint_speed_rad_s'])
    max_v=max_a=0.; violations=0; violating_intervals=[]; position_excess=0.
    joint_ids=[j for j in range(models[1].njnt) if models[1].jnt_qposadr[j]>=7]
    lo=models[1].jnt_range[joint_ids,0]; hi=models[1].jnt_range[joint_ids,1]
    for k in range(len(q)-1):
        bridge=HermiteBridge(1/fps,trajectory.segments[:,k])
        for derivative in [1,2]:
            lower,upper=bridge.extrema(derivative)
            peak=float(np.max(np.maximum(np.abs(lower),np.abs(upper))))
            if derivative==1:max_v=max(max_v,peak)
            else:max_a=max(max_a,peak)
        lower,upper=bridge.extrema(0); position_excess=max(position_excess,float(np.max(np.maximum(lo-lower,upper-hi))))
        bad=False
        for u in np.linspace(0,1,82)[1:]:
            pose=np.r_[p['root_pos'][k+1],p['root_rot'][k+1],bridge.at_time(u/fps).position]
            if _violates_configured_clearance(states,pose,.005):
                violations+=1;bad=True
        if bad:violating_intervals.append(k)
    report.update(quintic_peak_speed=max_v,quintic_peak_acceleration=max_a,quintic_joint_range_excess_rad=position_excess,
                  clearance_m=.005,violating_samples=violations,violating_intervals=violating_intervals)
    reports.append(report); print(json.dumps(report),flush=True)
# Nonzero-state example: rest-to-rest does not match the surrounding motion.
start=JointState(np.array([0.]),np.array([1.]),np.array([.4]))
end=JointState(np.array([.5]),np.array([.6]),np.array([-.2]))
b=HermiteBridge.between(start,end,1.)
rest=HermiteBridge.between(JointState(start.position,np.zeros(1),np.zeros(1)),JointState(end.position,np.zeros(1),np.zeros(1)),1.)
example={label:{side:{key:getattr(bridge.at_time(t),key).tolist() for key in ['position','velocity','acceleration']} for side,t in [('start',0.),('end',1.)]} for label,bridge in [('state_matched',b),('rest_to_rest',rest)]}
(root/'output/gmr_collision_state_audit/report.json').write_text(json.dumps(dict(clips=reports,boundary_example=example),indent=2),encoding='utf-8')
