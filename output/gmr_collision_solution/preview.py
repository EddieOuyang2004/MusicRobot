"""Render baseline and projected poses at identical timestamps and camera angles."""
import hashlib
import json
import pickle

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pathlib import Path
import sys
OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1] / 'realtime/humanoid_robot'
MOTION = 'gWA_sBM_cAll_d26_mWA0_ch07'
sys.path.insert(0, str(BASE/'src'))
import gmr_retarget_smpl_headless as runner


def main():
    baseline = np.load(OUT / "baseline.npz")["qpos"]
    projected = np.load(OUT / "projected_w2_m8.npz")["qpos"]
    report = json.loads((OUT / "projected_w2_m8_report.json").read_text())
    assert all(v["violating_samples"] == 0 for v in report["validation"].values())
    model = mujoco.MjModel.from_xml_path(str(runner.DEFAULT_COLLISION_VALIDATION_MODEL))
    with (BASE / "data/aistpp_gmr" / f"{MOTION}.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    runner.validate_joint_limits(model, projected, tuple(payload["dof_names"]))
    # Deliberately not a production cache version: this must not silently replace
    # approved dataset files or inherit the baseline's correction counts.
    payload.update(pipeline_version="experimental_collision_projection_v1",
                   root_pos=projected[:, :3], root_rot=projected[:, 3:7],
                   dof_pos=projected[:, 7:], experiment=report)
    payload["continuity_limits"]["limited_values"] = report["projection_stats"]
    payload["collision_avoidance"]["final_projection"] = dict(
        method="joint_displacement_qp", planning_distance_m=.008,
        segment_check_distance_m=.006, validation_distance_m=.005,
        gain=.5, previous_step_weight=2.)
    with (OUT / "projected_motion.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    model.vis.global_.offwidth = 480
    model.vis.global_.offheight = 540
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_matid[geom] = -1
            model.geom_rgba[geom] = [.88, .91, .94, 1]
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.1, 90, -10
    camera.lookat[:] = [0, 0, .9]
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = False
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    holds = [np.r_[False, np.max(np.abs(np.diff(q[:, 7:], axis=0)*60), axis=1)<1e-6]
             for q in (baseline, projected)]
    writer = imageio_ffmpeg.write_frames(str(OUT/'comparison.mp4'), (960, 720),
                                        fps=30, codec='libx264', quality=8, macro_block_size=1)
    writer.send(None)
    saved = False
    try:
        with mujoco.Renderer(model, height=540, width=480) as renderer:
            for frame in range(0, len(projected), 2):
                canvas = Image.new('RGB', (960, 720), '#101b2b')
                draw = ImageDraw.Draw(canvas)
                for side, (name, trajectory) in enumerate((('Current filter', baseline),
                                                           ('Target-seeking projection', projected))):
                    data.qpos[:] = trajectory[frame]
                    data.qpos[:2] = 0
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    canvas.paste(Image.fromarray(renderer.render()), (480*side, 80))
                    color = '#df83ff' if holds[side][frame] else '#b6ded7'
                    draw.text((20+480*side, 12), name, font=font, fill='white')
                    status = 'ALL JOINTS HELD' if holds[side][frame] else 'Joints moving'
                    if frame == 0:
                        status = 'Initial pose'
                    draw.text((20+480*side, 47), status, font=small, fill=color)
                    if holds[side][frame]:
                        draw.rectangle((480*side+2, 82, 480*side+477, 618), outline=color, width=4)
                    for k in np.flatnonzero(holds[side]):
                        x = 20+480*side+int(k/(len(projected)-1)*440)
                        draw.line((x, 643, x, 655), fill='#df83ff', width=2)
                    x = 20+480*side+int(frame/(len(projected)-1)*440)
                    draw.line((x, 640, x, 658), fill='white', width=2)
                draw.text((20, 669), f't = {frame/60:05.2f} s  |  Same source timing  |  Purple: all-joint holds', font=small, fill='white')
                draw.text((20, 696), 'Kinematic preview; XY centered; sampled self-collision validation on both models.', font=small, fill='#a8bbd0')
                writer.send(np.asarray(canvas))
                if holds[0][frame] and not saved:
                    canvas.save(OUT / 'comparison.jpg')
                    saved = True
    finally:
        writer.close()
    files = ['raw.npz', 'baseline.npz', 'projected_w2_m8.npz', 'projected_motion.pkl', 'comparison.mp4']
    (OUT / 'provenance.json').write_text(json.dumps(
        {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in files}, indent=2))
    print('Saved comparison.mp4 and experimental projected_motion.pkl', flush=True)


if __name__ == '__main__':
    main()
