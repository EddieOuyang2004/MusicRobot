"""Render actual paired SMPL/G1 frames using NumPy LBS and MuJoCo.

Run with the project .venv Python. No model parameters are exported.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'realtime/humanoid_robot'
sys.path.insert(0, str(BASE / 'src'))
from aistpp_smpl import load_aistpp_motion, smpl_world_kinematics, Y_UP_TO_Z_UP

TMP = ROOT / 'tmp/pdfs/smpl_g1'
ID = 'gBR_sBM_cAll_d04_mBR0_ch01'
FRAMES = [269, 358, 422]


def render_body(renderer, model, data, camera, opt):
    renderer.update_scene(data, camera=camera, scene_option=opt)
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    segments = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    body_ids = np.flatnonzero(model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
    keep = (segments[:,:,1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segments[:,:,0],body_ids)
    rgb[~keep] = [255,255,255]
    return rgb


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    source = BASE / 'data/aistpp/motions' / (ID + '.pkl')
    target = BASE / 'data/aistpp_gmr' / (ID + '.pkl')
    modelpath = BASE / 'assets/body_models/smpl/SMPL_NEUTRAL.pkl'
    with target.open('rb') as f: robot = pickle.load(f)
    with modelpath.open('rb') as f: smpl = pickle.load(f)
    assert digest(source) == robot['source_sha256']
    assert digest(modelpath) == robot['smpl_model_sha256']
    poses, trans = load_aistpp_motion(source)
    assert len(poses) == len(robot['root_pos'])
    J = np.asarray(smpl['J_regressor'].dot(smpl['v_template']))
    parents = np.asarray(smpl['kintree_table'][0], dtype=np.int64)
    parents[0] = -1
    positions, _ = smpl_world_kinematics(poses[FRAMES], trans[FRAMES], J, parents)
    robotmodel = mujoco.MjModel.from_xml_path(str(BASE / 'assets/open_humanoid_dancer.xml'))
    robotmodel.vis.global_.offwidth = 700
    robotmodel.vis.global_.offheight = 800
    # Neutral studio background; the physical robot mesh and joint values stay intact.
    for g in range(robotmodel.ngeom):
        if robotmodel.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            robotmodel.geom_matid[g] = -1
            robotmodel.geom_rgba[g] = [.94, .95, .96, 1]
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = False
    # A common metric camera distance preserves body-size differences.
    camera = mujoco.MjvCamera()
    camera.distance = 2.5
    camera.azimuth = 90
    camera.elevation = -12
    camera.lookat[:] = [0, 0, .9]
    for slot, frame in enumerate(FRAMES):
        R = Rotation.from_rotvec(poses[frame]).as_matrix()
        v = np.asarray(smpl['v_template']) + np.einsum('vcp,p->vc', smpl['posedirs'], (R[1:]-np.eye(3)).ravel())
        G = np.zeros((24,4,4))
        for j in range(24):
            local = np.eye(4); local[:3,:3] = R[j]
            local[:3,3] = J[j] if j == 0 else J[j]-J[parents[j]]
            G[j] = local if j == 0 else G[parents[j]] @ local
        A = G.copy()
        A[:,:3,3] -= np.einsum('jab,jb->ja', G[:,:3,:3], J)
        skin = np.einsum('vj,jab->vab', smpl['weights'], A)
        vertices = np.einsum('vab,vb->va', skin, np.c_[v,np.ones(len(v))])[:,:3] + trans[frame]
        vertices = vertices @ Y_UP_TO_Z_UP.T
        # Verify the FK used by LBS matches the retargeting pipeline.
        fk = (G[:,:3,3] + trans[frame]) @ Y_UP_TO_Z_UP.T
        assert np.max(np.abs(fk - positions[slot])) < 1e-6
        vertices[:,:2] -= positions[slot,0,:2]
        vertices[:,2] -= vertices[:,2].min()
        obj = TMP / f'smpl_{frame}.obj'
        with obj.open('w') as f:
            for x,y,z in vertices: f.write(f'v {x:.7f} {y:.7f} {z:.7f}\n')
            for a,b,d in smpl['f']+1: f.write(f'f {a} {b} {d}\n')
        xml = f'''<mujoco><visual><global offwidth="700" offheight="800"/><headlight diffuse=".8 .8 .8" ambient=".35 .35 .35"/></visual>
        <asset><mesh name="body" file="{obj.as_posix()}"/></asset>
        <worldbody><light pos="1 -2 4"/><geom type="plane" size="5 5 .1" rgba=".94 .95 .96 1"/>
        <geom type="mesh" mesh="body" rgba=".25 .53 .73 1"/></worldbody></mujoco>'''
        humanmodel = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(humanmodel); mujoco.mj_forward(humanmodel,data)
        with mujoco.Renderer(humanmodel,height=800,width=700) as renderer:
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
            np.save(TMP/f'smpl_{frame}.npy',render_body(renderer,humanmodel,data,camera,opt))
        data = mujoco.MjData(robotmodel)
        data.qpos[:3] = robot['root_pos'][frame]
        data.qpos[:2] = 0
        data.qpos[3:7] = robot['root_rot'][frame]
        for name,value in zip(robot['dof_names'],robot['dof_pos'][frame]):
            jid = mujoco.mj_name2id(robotmodel,mujoco.mjtObj.mjOBJ_JOINT,name)
            if jid < 0:
                jid = mujoco.mj_name2id(robotmodel,mujoco.mjtObj.mjOBJ_JOINT,name+'_joint')
            assert jid >= 0
            data.qpos[robotmodel.jnt_qposadr[jid]] = value
        mujoco.mj_forward(robotmodel,data)
        lowest = float('inf')
        for g in range(robotmodel.ngeom):
            if robotmodel.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = robotmodel.geom_dataid[g]
            start = robotmodel.mesh_vertadr[mesh]
            count = robotmodel.mesh_vertnum[mesh]
            verts = robotmodel.mesh_vert[start:start+count]
            world = verts @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g]
            lowest = min(lowest,float(world[:,2].min()))
        data.qpos[2] -= lowest
        mujoco.mj_forward(robotmodel,data)
        with mujoco.Renderer(robotmodel,height=800,width=700) as renderer:
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
            np.save(TMP/f'g1_{frame}.npy',render_body(renderer,robotmodel,data,camera,opt))
    report = dict(source_motion_id=ID, source_sha256=digest(source), target_sha256=digest(target),
                  smpl_model_sha256=digest(modelpath), frames_zero_based=FRAMES, fps=robot['fps'],
                  times_seconds=[i/robot['fps'] for i in FRAMES],
                  frame_selection='Within this clip: maximum wrist separation, maximum mean wrist height relative to pelvis, minimum mean wrist height relative to pelvis. Selected from source kinematics, not target fit.',
                  source_frames=len(poses), target_frames=len(robot['root_pos']),
                  display='Common camera and metric scale; independent horizontal root centering and vertical mesh grounding; joint rotations unchanged. Background excluded using renderer segmentation.',
                  smpl_shape='Neutral template; zero shape coefficients, matching the retargeting rest skeleton')
    (TMP/'provenance.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__': main()
