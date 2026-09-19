from pathlib import Path
import sys,json,numpy as np
root=Path.cwd();base=root/'realtime/humanoid_robot';sys.path.insert(0,str(base/'src'))
import mujoco,gmr_state_trajectory as p
from gmr_collision_projection import collision_states
out=root/'output/gmr_frame488_diagnostic'
d=np.load(out/'guide.npz');settings=json.loads((out/'job.json').read_text())['identity']['projection'];settings['_debug']=True
models=tuple(mujoco.MjModel.from_xml_path(str(x)) for x in (base/'.deps/GMR/assets/unitree_g1/g1_mocap_29dof.xml',base/'assets/open_humanoid_dancer.xml'))
from gmr_retarget_smpl_headless import activate_required_collision_geoms
activate_required_collision_geoms(models[0])
states=collision_states(models)
source=(base/'src/gmr_state_trajectory.py').read_text()
source=source.replace('return fail("nonlinear check or exact bound")', '''print("JOINT VIOLATION", float(np.max(bridge.extrema(0)[1]-high)),float(np.max(low-bridge.extrema(0)[0])),flush=True); print("FINAL CHECK", "clearance", _curve_safe(bridge, root, states, settings["segment_clearance_m"], max(40,4*len(references))), "bounds", [(d, bridge.within_limits(d,np.broadcast_to(lo,(n,)),np.broadcast_to(hi,(n,)))) for d,lo,hi in ((0,low,high),(1,-settings["max_joint_speed_rad_s"],settings["max_joint_speed_rad_s"]),(2,-settings["max_joint_acceleration_rad_s2"],settings["max_joint_acceleration_rad_s2"]))], flush=True)
    return fail("nonlinear check or exact bound")''')
source=source.replace('start = JointState(q[frame], v[frame], a[frame])', 'start = JointState(q[frame], v[frame], a[frame])\n        print("FRAME",frame,flush=True)')
exec(compile(source,str(out/'instrumented.py'),'exec'),p.__dict__)
try:
 result=p.plan_state_trajectory(d['qpos'],float(d['fps']),states,d['low'],d['high'],settings)
 print('SUCCESS',result[3],flush=True)
except Exception as exc: print(exc,flush=True)



