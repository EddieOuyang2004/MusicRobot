from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

from aistpp_smpl import (
    load_aistpp_motion,
    load_smpl_rest_pose,
)
from aistpp_to_gmr_bvh import GMR_LAFAN_JOINTS
from retargeting_diagnostics import (
    DiagnosticThresholds,
    PoseBoneSpec,
    diagnose_continuity,
    diagnose_pose_similarity,
    smpl_world_kinematics,
)
from unitree_g1_dance_adapter import UnitreeG1DanceAdapter


HUMANOID_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AISTPP_ROOT = HUMANOID_ROOT / "data" / "aistpp"
DEFAULT_BVH_ROOT = HUMANOID_ROOT / "data" / "aistpp_bvh"
DEFAULT_GMR_ROOT = HUMANOID_ROOT / "data" / "aistpp_gmr"
DEFAULT_GMR_CHECKOUT = HUMANOID_ROOT / ".deps" / "GMR"
DEFAULT_SMPL_ROOT = HUMANOID_ROOT / "assets" / "body_models" / "smpl"
DEFAULT_MUJOCO_MODEL = HUMANOID_ROOT / "assets" / "open_humanoid_dancer.xml"
DEFAULT_OUTPUT_ROOT = DEFAULT_GMR_ROOT / "diagnostics"

BVH_NAMES = tuple(joint.name for joint in GMR_LAFAN_JOINTS)
BVH_SMPL_INDICES = np.asarray([joint.smpl_index for joint in GMR_LAFAN_JOINTS], dtype=np.int64)
# Use attachment-to-attachment vectors for cross-morphology direction audits:
# SMPL collars anchor the full upper arm and SMPL hand joints extend the full forearm.
AUDIT_SMPL_NAMES = (
    "pelvis", "spine3",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_foot", "right_foot",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
AUDIT_SMPL_INDICES = np.asarray(
    [0, 9, 1, 2, 4, 5, 10, 11, 13, 14, 18, 19, 22, 23], dtype=np.int64
)
SMPL_EDGES = (
    (0, 1),
    (0, 2), (2, 4), (4, 6),
    (0, 3), (3, 5), (5, 7),
    (1, 8), (8, 10), (10, 12),
    (1, 9), (9, 11), (11, 13),
)
G1_BODY_NAMES = (
    "pelvis", "torso_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    # Exact GMR orientation-task bodies are retained for static regression data,
    # while the visual/audit upper-arm segment starts at the full shoulder assembly.
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
)
G1_EDGES = (
    (0, 1),
    (0, 2), (2, 4), (4, 6),
    (0, 3), (3, 5), (5, 7),
    (1, 8), (8, 10), (10, 12),
    (1, 9), (9, 11), (11, 13),
)
POSE_BONES = (
    PoseBoneSpec("torso", "pelvis", "spine3", "pelvis", "torso_link"),
    PoseBoneSpec("left_thigh", "left_hip", "left_knee", "left_hip_roll_link", "left_knee_link"),
    PoseBoneSpec("right_thigh", "right_hip", "right_knee", "right_hip_roll_link", "right_knee_link"),
    PoseBoneSpec("left_shin", "left_knee", "left_foot", "left_knee_link", "left_ankle_roll_link"),
    PoseBoneSpec("right_shin", "right_knee", "right_foot", "right_knee_link", "right_ankle_roll_link"),
    PoseBoneSpec("left_upper_arm", "left_shoulder", "left_elbow", "left_shoulder_pitch_link", "left_elbow_link"),
    PoseBoneSpec("right_upper_arm", "right_shoulder", "right_elbow", "right_shoulder_pitch_link", "right_elbow_link"),
    PoseBoneSpec("left_forearm", "left_elbow", "left_wrist", "left_elbow_link", "left_wrist_yaw_link"),
    PoseBoneSpec("right_forearm", "right_elbow", "right_wrist", "right_elbow_link", "right_wrist_yaw_link"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit AIST++ SMPL FK, optional BVH diagnostics, and final G1 pose similarity."
    )
    parser.add_argument("--motion", type=Path, action="append", help="AIST++ motion; repeatable.")
    parser.add_argument(
        "--video-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When no --motion is supplied, audit every locally available source-video sample.",
    )
    parser.add_argument("--aistpp-root", type=Path, default=DEFAULT_AISTPP_ROOT)
    parser.add_argument("--bvh-root", type=Path, default=DEFAULT_BVH_ROOT)
    parser.add_argument("--gmr-motion-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--gmr-checkout", type=Path, default=DEFAULT_GMR_CHECKOUT)
    parser.add_argument("--smpl-model-path", type=Path, default=DEFAULT_SMPL_ROOT)
    parser.add_argument("--mujoco-model", type=Path, default=DEFAULT_MUJOCO_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-json", type=Path, help="Optional combined summary JSON.")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--raw-axis-angle-jump", type=float, default=float(np.pi))
    parser.add_argument("--smpl-joint-speed", type=float, default=15.0)
    parser.add_argument("--bvh-joint-speed", type=float, default=15.0)
    parser.add_argument("--bvh-position-error", type=float, default=1e-4)
    parser.add_argument("--pose-median-cosine", type=float, default=0.85)
    parser.add_argument("--pose-negative-frame-fraction", type=float, default=0.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_to_motion_id(video_path: Path) -> str:
    parts = video_path.stem.split("_")
    if len(parts) < 3 or not parts[2].startswith("c"):
        raise ValueError(f"Unexpected AIST++ video name: {video_path.name}")
    parts[2] = "cAll"
    return "_".join(parts)


def collect_jobs(args: argparse.Namespace) -> list[tuple[Path, Path | None]]:
    aistpp_root = args.aistpp_root.resolve()
    if args.motion:
        jobs = [
            ((path if path.is_absolute() else Path.cwd() / path).resolve(), None)
            for path in args.motion
        ]
    elif args.video_samples:
        jobs = []
        for video in sorted((aistpp_root / "videos").glob("*.mp4")):
            motion = aistpp_root / "motions" / f"{video_to_motion_id(video)}.pkl"
            jobs.append((motion, video))
    else:
        jobs = [(path, None) for path in sorted((aistpp_root / "motions").glob("*.pkl"))]
    if args.limit is not None:
        jobs = jobs[: max(int(args.limit), 0)]
    return jobs


def load_source_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return load_aistpp_motion(path)


def load_bvh_layer(gmr_checkout: Path, bvh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    sys.path.insert(0, str(gmr_checkout.resolve()))
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, _height = load_bvh_file(str(bvh_path.resolve()), format="lafan1")
    positions = np.asarray(
        [[frame[name][0] for name in BVH_NAMES] for frame in frames], dtype=np.float64
    )
    rotations = np.asarray(
        [[frame[name][1] for name in BVH_NAMES] for frame in frames], dtype=np.float64
    )
    return positions, rotations


def load_g1_artifact(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"GMR artifact must contain a dict: {path}")
    required = {
        "fps",
        "root_pos",
        "root_rot",
        "root_rot_order",
        "dof_pos",
        "dof_names",
        "source_motion_id",
        "source_sha256",
        "smpl_model_sha256",
        "retargeter",
        "retargeter_version",
        "collision_avoidance",
        "mink_limits_api",
        "continuity_limits",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"GMR artifact is missing fields {sorted(missing)}: {path}")
    if payload.get("format_version") != 1 or payload.get("pipeline_version") != 4:
        raise ValueError(f"Non-canonical GMR artifact: {path}")
    if payload.get("source_format") != "aistpp_smpl_direct":
        raise ValueError(f"Non-SMPL-direct GMR artifact: {path}")
    if payload.get("root_rot_order") != "wxyz" or payload.get("retargeter") != "GMR":
        raise ValueError(f"Ambiguous GMR root rotation or retargeter: {path}")
    if "bvh_sha256" in payload:
        raise ValueError(f"SMPL-direct artifact unexpectedly records BVH provenance: {path}")
    collision = payload["collision_avoidance"]
    if not isinstance(collision, dict) or collision.get("preset") != "g1_self_collision_v2":
        raise ValueError(f"Unsupported collision-avoidance metadata: {path}")
    names = tuple(str(name) for name in payload["dof_names"])
    if names != UnitreeG1DanceAdapter.GMR_DOF_NAMES:
        raise ValueError(f"GMR artifact DoF names/order are not canonical: {path}")
    arrays = {
        "root_pos": np.asarray(payload["root_pos"], dtype=np.float64),
        "root_rot": np.asarray(payload["root_rot"], dtype=np.float64),
        "dof_pos": np.asarray(payload["dof_pos"], dtype=np.float64),
    }
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError(f"GMR artifact contains non-finite values: {path}")
    frame_count = len(arrays["dof_pos"])
    if arrays["dof_pos"].ndim != 2 or arrays["dof_pos"].shape[1] != 29:
        raise ValueError(f"GMR artifact must contain dof_pos[N,29]: {path}")
    if arrays["root_pos"].shape != (frame_count, 3) or arrays["root_rot"].shape != (
        frame_count,
        4,
    ):
        raise ValueError(f"GMR artifact root arrays do not match DoF frames: {path}")
    norms = np.linalg.norm(arrays["root_rot"], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"GMR artifact contains non-unit root rotations: {path}")
    if frame_count > 1 and np.any(
        np.sum(arrays["root_rot"][1:] * arrays["root_rot"][:-1], axis=1) < -1e-8
    ):
        raise ValueError(f"GMR artifact contains quaternion sign discontinuities: {path}")
    continuity = payload["continuity_limits"]
    if not isinstance(continuity, dict) or continuity.get("loop_closure_frames") != 0:
        raise ValueError(f"GMR artifact contains generation-time loop closure: {path}")
    return {**payload, **arrays}


def g1_body_positions(
    model_path: Path,
    root_positions: np.ndarray,
    root_rotations: np.ndarray,
    dof_positions: np.ndarray,
    dof_names: tuple[str, ...],
) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    free_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if free_joint < 0:
        raise ValueError("MuJoCo model is missing floating_base_joint.")
    free_address = int(model.jnt_qposadr[free_joint])
    dof_addresses = []
    for name in dof_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
        if joint_id < 0:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing G1 joint {name!r}.")
        dof_addresses.append(int(model.jnt_qposadr[joint_id]))
    body_ids = []
    for body_name in G1_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model is missing diagnostic body {body_name!r}.")
        body_ids.append(body_id)

    positions = np.empty((len(dof_positions), len(body_ids), 3), dtype=np.float64)
    for frame_index in range(len(dof_positions)):
        data.qpos[free_address : free_address + 3] = root_positions[frame_index]
        data.qpos[free_address + 3 : free_address + 7] = root_rotations[frame_index]
        data.qpos[dof_addresses] = dof_positions[frame_index]
        data.qvel.fill(0.0)
        data.ctrl.fill(0.0)
        mujoco.mj_forward(model, data)
        positions[frame_index] = data.xpos[body_ids]
    return positions


def write_comparison_html(
    path: Path,
    *,
    video_path: Path,
    fps: float,
    smpl_positions: np.ndarray,
    g1_positions: np.ndarray,
    events: list[dict[str, object]],
) -> None:
    relative_video = Path(os.path.relpath(video_path, path.parent)).as_posix()
    payload = {
        "fps": fps,
        "frames": len(smpl_positions),
        "smpl": np.round(smpl_positions, 5).tolist(),
        "g1": np.round(g1_positions, 5).tolist(),
        "smplEdges": SMPL_EDGES,
        "g1Edges": G1_EDGES,
        "events": events,
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(path.parent.name)
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title} retargeting audit</title>
<style>
body{{margin:0;background:#101318;color:#e8edf4;font:14px system-ui}}header{{padding:14px 18px}}
.grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;padding:0 12px}}video,canvas{{width:100%;aspect-ratio:16/9;background:#080a0e;border:1px solid #343b47}}
.label{{font-weight:600;margin:4px}}#timeline{{margin:14px;height:18px;background:#242a34;position:relative}}.tick{{position:absolute;top:0;width:2px;height:18px;background:#ff5964}}
#events{{padding:0 18px 20px;max-height:180px;overflow:auto}}input{{width:calc(100% - 36px);margin:0 18px}}
</style></head><body><header><h2>{title}: AIST++ SMPL → GMR smplx → G1</h2><span id="status"></span></header>
<div class="grid"><div><div class="label">Source video</div><video id="video" controls src="{html.escape(relative_video)}"></video></div>
<div><div class="label">SMPL world skeleton (Z-up)</div><canvas id="smpl" width="640" height="360"></canvas></div>
<div><div class="label">G1 MuJoCo FK</div><canvas id="g1" width="640" height="360"></canvas></div></div>
<input id="frame" type="range" min="0" max="1" value="0"><div id="timeline"></div><div id="events"></div>
<script id="audit-data" type="application/json">{data}</script><script>
const d=JSON.parse(document.getElementById('audit-data').textContent), v=document.getElementById('video'), s=document.getElementById('frame');
s.max=d.frames-1; const colors={{source_smpl:'#ff5964',bvh:'#ffb347',gmr_g1:'#a66cff',pose_similarity:'#ff2d95',parameterization_only:'#55c2ff'}};
function draw(id,pts,edges){{const c=document.getElementById(id),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);if(!pts)return;
 const projected=pts.map(p=>[p[0]+.3*p[1],p[2]+.12*p[1]]), xs=projected.map(p=>p[0]), ys=projected.map(p=>p[1]);
 const span=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),.1),scale=280/span,cx=(Math.min(...xs)+Math.max(...xs))/2,cy=(Math.min(...ys)+Math.max(...ys))/2;
 const q=projected.map(p=>[c.width/2+(p[0]-cx)*scale,c.height/2-(p[1]-cy)*scale]);x.strokeStyle='#76d6ff';x.lineWidth=3;x.beginPath();for(const e of edges){{x.moveTo(...q[e[0]]);x.lineTo(...q[e[1]])}}x.stroke();x.fillStyle='#f4f7fb';for(const p of q){{x.beginPath();x.arc(p[0],p[1],4,0,7);x.fill()}}}}
function render(frame){{frame=Math.max(0,Math.min(d.frames-1,frame|0));s.value=frame;draw('smpl',d.smpl[frame],d.smplEdges);draw('g1',d.g1[frame],d.g1Edges);document.getElementById('status').textContent=`frame ${{frame}} / ${{d.frames-1}} · ${{(frame/d.fps).toFixed(3)}} s`;}}
s.oninput=()=>{{v.currentTime=+s.value/d.fps;render(+s.value)}};v.ontimeupdate=()=>render(Math.round(v.currentTime*d.fps));
const t=document.getElementById('timeline'),e=document.getElementById('events');for(const item of d.events){{const m=document.createElement('span');m.className='tick';m.style.left=(100*item.frame/(d.frames-1))+'%';m.style.background=colors[item.classification];m.title=`frame ${{item.frame}}: ${{item.classification}}`;t.appendChild(m)}}
e.innerHTML=d.events.length?d.events.map(x=>`<div style="color:${{colors[x.classification]}}">frame ${{x.frame}} (${{x.time_seconds.toFixed(3)}}s): ${{x.classification}}</div>`).join(''):'No discontinuity candidates.';render(0);
</script></body></html>"""
    path.write_text(document, encoding="utf-8", newline="\n")


def audit_motion(
    args: argparse.Namespace,
    motion_path: Path,
    video_path: Path | None,
    rest_joints: np.ndarray,
    parents: np.ndarray,
) -> dict[str, object]:
    motion_id = motion_path.stem
    bvh_path = args.bvh_root.resolve() / f"{motion_id}.bvh"
    artifact_path = args.gmr_motion_root.resolve() / f"{motion_id}.pkl"
    for required in (motion_path, artifact_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    poses, translations = load_source_motion(motion_path)
    smpl_positions_all, smpl_rotations_all = smpl_world_kinematics(
        poses, translations, rest_joints, parents
    )
    smpl_positions = smpl_positions_all[:, AUDIT_SMPL_INDICES]
    smpl_rotations = smpl_rotations_all[:, AUDIT_SMPL_INDICES]
    smpl_bvh_positions = smpl_positions_all[:, BVH_SMPL_INDICES]
    bvh_available = bvh_path.is_file()
    if bvh_available:
        bvh_positions, bvh_rotations = load_bvh_layer(args.gmr_checkout, bvh_path)
    else:
        # Continuity classification remains usable when the optional diagnostic
        # layer is absent; an identity comparison cannot create a BVH failure.
        bvh_positions = smpl_bvh_positions.copy()
        bvh_rotations = None
    artifact = load_g1_artifact(artifact_path)
    if artifact["source_motion_id"] != motion_id:
        raise ValueError(
            f"GMR source id mismatch: expected {motion_id}, got {artifact['source_motion_id']!r}"
        )
    if artifact["source_sha256"] != sha256(motion_path):
        raise ValueError(f"GMR source hash does not match the audited SMPL motion: {motion_id}")
    root_positions = artifact["root_pos"]
    root_rotations = artifact["root_rot"]
    dof_positions = artifact["dof_pos"]
    dof_names = tuple(str(name) for name in artifact["dof_names"])
    frame_counts = {len(poses), len(root_positions), len(dof_positions)}
    if bvh_available:
        frame_counts.add(len(bvh_positions))
    if len(frame_counts) != 1:
        raise ValueError(f"Layer frame counts disagree for {motion_id}: {sorted(frame_counts)}")

    thresholds = DiagnosticThresholds(
        raw_axis_angle_jump_rad=float(args.raw_axis_angle_jump),
        smpl_joint_speed_m_s=float(args.smpl_joint_speed),
        bvh_joint_speed_m_s=float(args.bvh_joint_speed),
        bvh_position_error_m=float(args.bvh_position_error),
    )
    continuity = diagnose_continuity(
        raw_axis_angles=poses[:, BVH_SMPL_INDICES],
        smpl_positions=smpl_bvh_positions,
        bvh_positions=bvh_positions,
        g1_root_positions=root_positions,
        g1_root_quaternions_wxyz=root_rotations,
        g1_dof_positions=dof_positions,
        fps=float(args.fps),
        smpl_joint_names=BVH_NAMES,
        dof_names=dof_names,
        thresholds=thresholds,
    )
    g1_positions = g1_body_positions(
        args.mujoco_model, root_positions, root_rotations, dof_positions, dof_names
    )
    pose_similarity = diagnose_pose_similarity(
        source_positions=smpl_positions,
        target_positions=g1_positions,
        source_names=AUDIT_SMPL_NAMES,
        target_names=G1_BODY_NAMES,
        bones=POSE_BONES,
        median_threshold=float(args.pose_median_cosine),
        negative_fraction_threshold=float(args.pose_negative_frame_fraction),
    )
    pose_events = []
    for event in pose_similarity["events"]:
        pose_events.append({**event, "time_seconds": float(event["frame"] / args.fps)})
    events = sorted(
        [*continuity["events"], *pose_events],
        key=lambda event: (int(event["frame"]), str(event["classification"])),
    )
    failure_counts = dict(continuity["failure_counts"])
    failure_counts["pose_similarity"] = len(pose_similarity["failed_bones"])
    status = (
        "failed"
        if continuity["status"] == "failed" or pose_similarity["status"] == "failed"
        else "ok"
    )

    output_dir = args.output_root.resolve() / motion_id
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "layers.npz"
    layer_arrays = {
        "raw_axis_angles": poses,
        "root_translation_m": translations,
        "smpl_positions_m_zup": smpl_positions,
        "smpl_global_rot_wxyz_zup": smpl_rotations,
        "g1_root_pos_m": root_positions,
        "g1_root_rot_wxyz": root_rotations,
        "g1_dof_pos_rad": dof_positions,
        "g1_body_positions_m_zup": g1_positions,
    }
    if bvh_available:
        layer_arrays["bvh_positions_m_zup"] = bvh_positions
        layer_arrays["bvh_global_rot_wxyz_zup"] = bvh_rotations
    np.savez_compressed(arrays_path, **layer_arrays)
    report = {
        "format_version": 1,
        "source_motion_id": motion_id,
        "fps": float(args.fps),
        "frames": len(poses),
        "source_sha256": sha256(motion_path),
        "pipeline_version": 4,
        "source_format": "aistpp_smpl_direct",
        "bvh_diagnostic": {
            "available": bvh_available,
            "sha256": sha256(bvh_path) if bvh_available else None,
        },
        "artifact_sha256": sha256(artifact_path),
        "retargeter_version": artifact.get("retargeter_version"),
        "arrays": arrays_path.name,
        "video": str(video_path.resolve()) if video_path is not None else None,
        "joint_names": list(AUDIT_SMPL_NAMES),
        "dof_names": list(dof_names),
        "thresholds": {
            "continuity": thresholds.__dict__,
            "pose_similarity": pose_similarity["thresholds"],
        },
        "status": status,
        "failure_counts": failure_counts,
        "parameterization_only_count": continuity["parameterization_only_count"],
        "events": events,
        "maxima": continuity["maxima"],
        "pose_similarity": pose_similarity,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if video_path is not None and not args.no_html:
        comparison_path = output_dir / "comparison.html"
        write_comparison_html(
            comparison_path,
            video_path=video_path,
            fps=float(args.fps),
            smpl_positions=smpl_positions,
            g1_positions=g1_positions,
            events=events,
        )
        report["comparison"] = comparison_path.name
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if not -1.0 <= args.pose_median_cosine <= 1.0:
        raise ValueError("--pose-median-cosine must be between -1 and 1.")
    if not 0.0 <= args.pose_negative_frame_fraction <= 1.0:
        raise ValueError("--pose-negative-frame-fraction must be between 0 and 1.")
    jobs = collect_jobs(args)
    if not jobs:
        print("No AIST++ motions selected.")
        return 0
    rest_joints, parents = load_smpl_rest_pose(args.smpl_model_path, "NEUTRAL")
    reports = []
    failures = []
    for motion_path, video_path in jobs:
        try:
            report = audit_motion(args, motion_path, video_path, rest_joints, parents)
            reports.append(report)
            print(
                f"{motion_path.stem}: {report['status']} "
                f"({len(report['events'])} candidate frame(s))"
            )
        except Exception as exc:
            failure = {
                "source_motion_id": motion_path.stem,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            failures.append(failure)
            print(f"{motion_path.stem}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
    summary = {
        "format_version": 1,
        "reports": [
            {
                "source_motion_id": report["source_motion_id"],
                "status": report["status"],
                "failure_counts": report["failure_counts"],
                "events": len(report["events"]),
                "worst_bone": report["pose_similarity"]["worst_bone"],
                "worst_frame": report["pose_similarity"]["worst_frame"],
            }
            for report in reports
        ],
        "failures": failures,
    }
    summary_path = (
        args.output_json.resolve()
        if args.output_json is not None
        else args.output_root.resolve() / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 1 if failures or any(report["status"] == "failed" for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
