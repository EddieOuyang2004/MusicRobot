from pathlib import Path
import sys,pickle,json,hashlib
import numpy as np,mujoco
from types import SimpleNamespace
src=Path('realtime/humanoid_robot/src').resolve();sys.path.insert(0,str(src))
from realtime_music_humanoid_dancer import GmrUnitreeG1MotionSampler,FeatureState
from realtime_music_humanoid_matcher_v2 import build_motion_entry_features
from unitree_g1_dance_adapter import UnitreeG1JointPoseAdapter
from motion_bridges import HermiteBridge
from gmr_collision_projection import collision_states
from gmr_retarget_smpl_headless import _violates_configured_clearance
model=mujoco.MjModel.from_xml_path(str(src.parent/'assets/open_humanoid_dancer.xml')); states=collision_states((model,))
results=[]
for name in ['gWA_sBM_cAll_d26_mWA0_ch07','gBR_sBM_cAll_d05_mBR0_ch08']:
    path=src.parent/'data/aistpp_gmr_v2'/f'{name}.pkl';p=pickle.load(path.open('rb'))
    sampler=GmrUnitreeG1MotionSampler(path,None,1.,0.,False)
    adapter=UnitreeG1JointPoseAdapter([model.actuator(i).name for i in range(model.nu)])
    feature=build_motion_entry_features(sampler,SimpleNamespace(keypoint_phases=(),keypoint_scores=()),adapter,{})
    columns=[sampler.dof_names.index({v:k for k,v in adapter.name_map.items()}[name]) for name in feature.joint_names]
    np.testing.assert_array_equal(feature.authored.velocities,p['dof_vel'][:,columns])
    np.testing.assert_array_equal(feature.authored.accelerations,p['dof_acc'][:,columns])
    bad=0; sampler_error=0.
    for k in range(len(p['dof_pos'])-1):
        for u in np.linspace(0.,1.,43)[1:]:
            time=(k+u)/p['fps'];joint=sampler.authored_trajectory.at_time(time).position
            frame=sampler.sample_frame((k+u)/len(sampler.frames),1.,0.,FeatureState())
            sampler_error=max(sampler_error,float(np.max(np.abs(joint-np.array([frame.joint_positions[n] for n in sampler.dof_names])))))
            pose=np.r_[p['root_pos'][k+1],p['root_rot'][k+1],joint]
            bad+=int(_violates_configured_clearance(states,pose,.005))
    original=src.parent/'data/aistpp_gmr'/f'{name}.pkl'
    assert hashlib.sha256(original.read_bytes()).hexdigest()==p['v2_build']['original_gmr_sha256']
    row=dict(motion=name,report=p['projection_validation'],sampler_max_error_rad=sampler_error,playback_mujoco=mujoco.__version__,playback_clearance_violations=bad,playback_samples_per_interval=42)
    assert bad==0 and sampler_error<1e-10
    results.append(row);print(json.dumps(row),flush=True)
Path('output/gmr_collision_state_audit/fixed_playback_report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
