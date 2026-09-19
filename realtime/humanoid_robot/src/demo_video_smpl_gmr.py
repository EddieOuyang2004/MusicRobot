"""Render synchronized original / neutral SMPL surface / saved GMR comparison.

Requires the project environment plus pillow and imageio-ffmpeg. This is
kinematic playback of saved poses, not a dynamics/controller tracking test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from aistpp_smpl import Y_UP_TO_Z_UP, load_aistpp_motion

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'realtime/humanoid_robot'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-id', default='gBR_sBM_cAll_d05_mBR0_ch08')
    parser.add_argument('--video', type=Path, default=ROOT/'gBR_sBM_c01_d05_mBR0_ch08.mp4')
    parser.add_argument('--output', type=Path, default=ROOT/'output/video_smpl_gmr_demo')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--retargeted', type=Path, help='Alternate traced GMR artifact')
    parser.add_argument('--collision-trace', type=Path, help='Verified replay trace for collision-caused holds')
    parser.add_argument('--camera-distance', type=float, default=3.1)
    parser.add_argument('--camera-height', type=float, default=.9)
    parser.add_argument('--max-frames', type=int, default=0, help='Limit output for render checks')
    parser.add_argument('--video-offset', type=float, default=0, help='Video seconds corresponding to motion t=0')
    args = parser.parse_args()
    if args.fps <= 0 or args.video_offset < 0:
        parser.error('fps must be positive and video offset nonnegative')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = BASE/'data/aistpp/motions'/f'{args.motion_id}.pkl'
    target = BASE/'data/aistpp_gmr'/f'{args.motion_id}.pkl'
    if args.retargeted:
        target = args.retargeted.resolve()
    body_path = BASE/'assets/body_models/smpl/SMPL_NEUTRAL.pkl'
    with target.open('rb') as f:
        robot = pickle.load(f)
    with body_path.open('rb') as f:
        smpl = pickle.load(f)
    if digest(source) != robot['source_sha256'] or digest(body_path) != robot['smpl_model_sha256']:
        raise ValueError('Saved retargeting provenance does not match the source/model')
    poses, trans = load_aistpp_motion(source)
    if len(poses) != len(robot['root_pos']) or robot['root_rot_order'] != 'wxyz':
        raise ValueError('Incompatible saved target timeline/quaternion convention')
    collision_pauses = set()
    if args.collision_trace:
        trace = json.loads(args.collision_trace.read_text(encoding='utf-8'))
        if not trace['replay_matches_saved'] or trace['target_sha256'] != digest(target):
            raise ValueError('Collision trace does not verify this saved trajectory')
        collision_pauses = set(trace['confirmed_pause_frames'])
        (out/'collision_trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
    collision_preview_saved = False
    source_fps = float(robot['fps'])
    count = int(np.floor(len(poses)/source_fps*args.fps))
    if args.max_frames:
        count = min(count, args.max_frames)
    frame_ids = np.minimum(np.rint(np.arange(count)*source_fps/args.fps).astype(int), len(poses)-1)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Timestamp-based video resampling; do not assume the camera's 59.94 fps is 60.
    decoder = subprocess.Popen([ffmpeg, '-v', 'error', '-ss', str(args.video_offset),
        '-i', str(args.video), '-vf', f'fps={args.fps},scale=640:360',
        '-frames:v', str(count), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], stdout=subprocess.PIPE)
    width, height = 480, 540
    model = mujoco.MjModel.from_xml_path(str(BASE/'assets/open_humanoid_dancer.xml'))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_matid[g] = -1
            model.geom_rgba[g] = [.88, .91, .94, 1]
    data = mujoco.MjData(model)
    addresses = []
    joint_ids = []
    for name in robot['dof_names']:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name+'_joint')
        if jid < 0:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(name)
        addresses.append(model.jnt_qposadr[jid])
        joint_ids.append(jid)
    camera = mujoco.MjvCamera()
    camera.distance = args.camera_distance
    camera.azimuth = 90
    camera.elevation = -10
    camera.lookat[:] = [0, 0, args.camera_height]
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = False
    J = np.asarray(smpl['J_regressor'].dot(smpl['v_template']))
    parents = np.asarray(smpl['kintree_table'][0], dtype=np.int64)
    parents[0] = -1
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23)
    annotation_font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    video_silent = out/'comparison_silent.mp4'
    writer = imageio_ffmpeg.write_frames(str(video_silent), (1600, 720), fps=args.fps,
        codec='libx264', quality=8, macro_block_size=1)
    writer.send(None)
    q = np.asarray(robot['dof_pos'])
    velocity = np.diff(q, axis=0)*source_fps
    speed_limit = float(robot['continuity_limits']['max_joint_speed_rad_s'])
    speed_hits = np.isclose(np.abs(velocity), speed_limit, atol=1e-5, rtol=0)
    speed_events = [dict(start_seconds=i/source_fps, end_seconds=(i+1)/source_fps,
        ending_frame=i+1, joints=[robot['dof_names'][j] for j in np.flatnonzero(row)])
        for i,row in enumerate(speed_hits) if row.any()]
    (out/'speed_limit_events.json').write_text(json.dumps(dict(
        limit_rad_s=speed_limit, tolerance_rad_s=1e-5,
        measurement='Absolute backward finite difference of saved GMR joint positions; not acceleration or angle limits.',
        intervals=speed_events),indent=2),encoding='utf-8')
    acceleration = np.diff(q, n=2, axis=0)*source_fps**2
    limits = model.jnt_range[joint_ids]
    limited = model.jnt_limited[joint_ids].astype(bool)
    violations = ((q < limits[:,0]-1e-6) | (q > limits[:,1]+1e-6)) & limited
    face_text = ''.join(f'f {a+1} {b+1} {c+1}\n' for a,b,c in smpl['f'])
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
            for index, frame in enumerate(frame_ids):
                raw = decoder.stdout.read(640*360*3)
                if len(raw) != 640*360*3:
                    raise RuntimeError(f'Original video ended before output frame {index}')
                original = Image.frombytes('RGB', (640,360), raw)
                R = Rotation.from_rotvec(poses[frame]).as_matrix()
                v = np.asarray(smpl['v_template']) + np.einsum('vcp,p->vc', smpl['posedirs'], (R[1:]-np.eye(3)).ravel())
                G = np.zeros((24,4,4))
                for j in range(24):
                    local = np.eye(4)
                    local[:3,:3] = R[j]
                    local[:3,3] = J[j] if j == 0 else J[j]-J[parents[j]]
                    G[j] = local if j == 0 else G[parents[j]] @ local
                A = G.copy()
                A[:,:3,3] -= np.einsum('jab,jb->ja', G[:,:3,:3], J)
                skin = np.einsum('vj,jab->vab', smpl['weights'], A)
                vertices = np.einsum('vab,vb->va', skin, np.c_[v,np.ones(len(v))])[:,:3] + trans[frame]
                vertices = vertices @ Y_UP_TO_Z_UP.T
                root = (G[0,:3,3]+trans[frame]) @ Y_UP_TO_Z_UP.T
                vertices[:,:2] -= root[:2]
                mesh_bytes = (''.join(f'v {x:.7f} {y:.7f} {z:.7f}\n' for x,y,z in vertices)+face_text).encode()
                human = mujoco.MjModel.from_xml_string(f'''<mujoco><visual><global offwidth="480" offheight="540"/>
                    <headlight diffuse=".8 .8 .8" ambient=".35 .35 .35"/></visual>
                    <asset><mesh name="body" file="body.obj"/></asset><worldbody>
                    <light pos="1 -2 4"/><geom type="plane" size="5 5 .1" rgba=".88 .91 .94 1"/>
                    <geom type="mesh" mesh="body" rgba=".23 .54 .77 1"/></worldbody></mujoco>''', assets={'body.obj': mesh_bytes})
                hd = mujoco.MjData(human)
                mujoco.mj_forward(human, hd)
                with mujoco.Renderer(human, height=height, width=width) as hr:
                    hr.update_scene(hd, camera=camera)
                    human_image = Image.fromarray(hr.render())
                data.qpos[:3] = robot['root_pos'][frame]
                data.qpos[:2] = 0
                data.qpos[3:7] = robot['root_rot'][frame]
                data.qpos[addresses] = q[frame]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                hit_joints = np.flatnonzero(speed_hits[frame-1]) if frame > 0 else np.array([],dtype=int)
                for j in hit_joints:
                    marker = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([.045,.045,.045]), data.xanchor[joint_ids[j]],
                        np.eye(3).ravel(), np.array([1.,.18,.03,.9]))
                    renderer.scene.ngeom += 1
                canvas = Image.new('RGB', (1600,720), '#101b2b')
                canvas.paste(original, (0,180))
                canvas.paste(human_image, (640,90))
                canvas.paste(Image.fromarray(renderer.render()), (1120,90))
                draw = ImageDraw.Draw(canvas)
                for x,title in [(20,'Original Motion Video'),
                    (660,'SMPL Motion'),
                    (1140,'GMR → G1 (MuJoCo)')]:
                    draw.text((x,14),title,font=font,fill='white')
                draw.text((20,570),f't = {index/args.fps:05.2f} s   |   SMPL / GMR frame {frame:03d}',font=font,fill='white')
                collision_hold = int(frame) in collision_pauses
                active = len(hit_joints) > 0
                color = '#ff743d' if active else '#a8bbd0'
                status = f'SPEED LIMIT REACHED | {len(hit_joints)} joints' if active else 'Below speed limit'
                if frame == 0:
                    status = 'Speed: N/A at initial frame'
                if collision_hold:
                    status = 'COLLISION HOLD | joints paused'
                    color = '#df83ff'
                    draw.rectangle((1122,92,1597,627),outline=color,width=5)
                draw.text((1140,53),status,font=annotation_font,fill=color)
                if frame > 0:
                    peak_joint = int(np.argmax(np.abs(velocity[frame-1])))
                    peak_speed = float(abs(velocity[frame-1,peak_joint]))
                    draw.text((660,643),f'Max joint speed: {peak_speed:.3f} / {speed_limit:.3f} rad/s',font=font,fill=color)
                    detail = 'Collision filter blocked joint motion | root may still move' if collision_hold else f'{robot["dof_names"][peak_joint]} | orange markers = joints at limit'
                    draw.text((660,680),detail,font=annotation_font,fill=color)
                for k,row in enumerate(speed_hits):
                    if row.any():
                        x = 20 + int(k / len(speed_hits)*580)
                        draw.line((x,646,x,657),fill='#ff743d',width=2)
                for k in collision_pauses:
                    x = 20 + int((k-1) / len(speed_hits)*580)
                    draw.line((x,665,x,675),fill='#df83ff',width=2)
                cursor = 20 + int(frame / (len(q)-1)*580)
                draw.line((cursor,640,cursor,677),fill='white',width=3)
                if args.collision_trace:
                    draw.text((20,680),'Orange: speed cap',font=annotation_font,fill='#ff743d')
                    draw.text((260,680),'Purple: collision hold',font=annotation_font,fill='#df83ff')
                else:
                    draw.text((20,680),f'Speed-limit timeline | 0 - {len(q)/source_fps:g} s',font=annotation_font,fill='#a8bbd0')
                if collision_hold and not collision_preview_saved:
                    canvas.save(out/'collision_hold_preview.jpg')
                    collision_preview_saved = True
                writer.send(np.asarray(canvas))
                if index in {0,count//3,2*count//3}:
                    canvas.save(out/f'preview_{index:04d}.jpg')
                if index % 30 == 0:
                    print(f'Rendered {index+1}/{count}', flush=True)
    finally:
        writer.close()
        decoder.stdout.close()
        decoder.wait()
    result = out/'original_smpl_gmr.mp4'
    subprocess.run([ffmpeg,'-v','error','-y','-i',str(video_silent),'-ss',str(args.video_offset),
        '-i',str(args.video),'-map','0:v:0','-map','1:a:0?','-c:v','copy','-c:a','aac',
        '-t',str(count/args.fps),'-movflags','+faststart',str(result)],check=True)
    video_silent.unlink()
    report = dict(motion_id=args.motion_id, source=str(source), target=str(target), video=str(args.video.resolve()),
        source_sha256=digest(source), target_sha256=digest(target), video_sha256=digest(args.video),
        smpl_model_sha256=digest(body_path), source_fps=source_fps, output_fps=args.fps,
        output_frames=count, source_frame_indices=frame_ids.tolist(), video_offset_seconds=args.video_offset,
        synchronization='Shared t=0 by matching clip ID; video timestamps resampled, SMPL/GMR nearest frame. Camera calibration/offset not independently verified.',
        display='Neutral SMPL shape as in retargeting; shared 3D camera and metric scale; XY root centered independently; original Z and rotations preserved.',
        camera_distance=args.camera_distance, camera_height=args.camera_height,
        playback='Kinematic mj_forward; no physics tracking; saved GMR includes pipeline postprocessing.',
        continuity_limits=robot.get('continuity_limits'), collision_avoidance=robot.get('collision_avoidance'),
        joint_angle_violating_frames=int(violations.any(axis=1).sum()),
        max_joint_speed_rad_s=float(np.abs(velocity).max()),
        max_joint_acceleration_rad_s2=float(np.abs(acceleration).max()),
        acceleration_note='Finite difference diagnostic only; no acceleration threshold assumed.')
    (out/'provenance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__':
    main()
