from __future__ import annotations

import argparse
import inspect
import pickle
import sys
from pathlib import Path

import numpy as np

from aistpp_smpl import (
    gmr_smpl_frames,
    load_aistpp_motion,
    load_smpl_rest_pose,
    smpl_world_kinematics,
)
from gmr_retarget_bvh_headless import limit_qpos_velocity, qpos_joint_names


FORMAT_VERSION = 1
PIPELINE_VERSION = 4
SOURCE_FORMAT = "aistpp_smpl_direct"
HUMAN_HEIGHT_M = 1.75
HUMANOID_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLISION_VALIDATION_MODEL = HUMANOID_ROOT / "assets" / "open_humanoid_dancer.xml"
DEFAULT_COLLISION_MIN_DISTANCE_M = 0.005
DEFAULT_COLLISION_DETECTION_DISTANCE_M = 0.08
DEFAULT_COLLISION_GAIN = 0.85
COLLISION_CLEARANCE_BUFFER_M = 0.001


LEFT_ARM = (
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
)
RIGHT_ARM = tuple(name.replace("left_", "right_", 1) for name in LEFT_ARM)
LEFT_DISTAL_ARM = LEFT_ARM[1:]
RIGHT_DISTAL_ARM = RIGHT_ARM[1:]
LEFT_HAND = ("left_rubber_hand",)
RIGHT_HAND = ("right_rubber_hand",)
HEAD = ("head_link",)
CORE = ("torso_link", "pelvis", *HEAD)
LEFT_LEG = (
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
)
RIGHT_LEG = tuple(name.replace("left_", "right_", 1) for name in LEFT_LEG)
LOWER_BODY = ("pelvis", *LEFT_LEG, *RIGHT_LEG)

# Each entry is expanded by Mink into every collision-geom cross product. The
# groups deliberately omit adjacent links, whose meshes overlap at mechanical
# joints by design and should not be treated as self-collisions.
G1_SELF_COLLISION_BODY_PAIRS = (
    (LEFT_ARM, CORE),
    (RIGHT_ARM, CORE),
    (LEFT_DISTAL_ARM, LOWER_BODY),
    (RIGHT_DISTAL_ARM, LOWER_BODY),
    (LEFT_DISTAL_ARM, RIGHT_DISTAL_ARM),
    (LEFT_HAND, RIGHT_HAND),
    (LEFT_HAND, CORE),
    (RIGHT_HAND, CORE),
    (LEFT_LEG, RIGHT_LEG),
)

OPTIONAL_COLLISION_BODIES = {*LEFT_HAND, *RIGHT_HAND, *HEAD}


def activate_required_collision_geoms(model: object) -> int:
    """Enable hand meshes that upstream GMR ships as visual-only geoms."""
    import mujoco

    activated = 0
    required_meshes = {*LEFT_HAND, *RIGHT_HAND}
    by_mesh: dict[str, list[int]] = {name: [] for name in required_meshes}
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom_id])
            )
            if mesh_name in by_mesh:
                by_mesh[mesh_name].append(geom_id)
    for mesh_name, geom_ids in by_mesh.items():
        if not geom_ids:
            raise ValueError(f"Required G1 collision mesh is missing: {mesh_name}")
        if any(int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0 for g in geom_ids):
            continue
        geom_id = geom_ids[-1]
        model.geom_contype[geom_id] = 1
        model.geom_conaffinity[geom_id] = 1
        activated += 1
    return activated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retarget AIST++ SMPL directly to Unitree G1 with GMR's smplx config."
    )
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--smpl-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-fps", type=float, required=True)
    parser.add_argument("--source-motion-id", required=True)
    parser.add_argument("--retargeter-version", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--smpl-model-sha256", required=True)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--max-joint-speed", type=float, default=3.0 * np.pi)
    parser.add_argument("--max-root-speed", type=float, default=3.0)
    parser.add_argument("--max-root-angular-speed", type=float, default=4.0 * np.pi)
    parser.add_argument(
        "--velocity-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply GMR/Mink velocity limits in addition to final trajectory limits.",
    )
    parser.add_argument(
        "--collision-avoidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Mink self-collision avoidance constraints (enabled by default).",
    )
    parser.add_argument(
        "--collision-min-distance",
        type=float,
        default=DEFAULT_COLLISION_MIN_DISTANCE_M,
        help="Minimum separation maintained between configured collision geoms, in metres.",
    )
    parser.add_argument(
        "--collision-detection-distance",
        type=float,
        default=DEFAULT_COLLISION_DETECTION_DISTANCE_M,
        help="Distance at which collision constraints become active, in metres.",
    )
    parser.add_argument(
        "--collision-gain",
        type=float,
        default=DEFAULT_COLLISION_GAIN,
        help="Mink collision approach gain in (0, 1].",
    )
    parser.add_argument(
        "--collision-validation-model",
        type=Path,
        default=DEFAULT_COLLISION_VALIDATION_MODEL,
        help="Final playback MJCF checked by the collision-aware trajectory limiter.",
    )
    return parser.parse_args()


def collision_geom_pairs(model: object) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Resolve named G1 body groups to active MuJoCo collision geom IDs."""
    import mujoco

    def geoms_for(body_names: tuple[str, ...]) -> tuple[int, ...]:
        body_ids: set[int] = set()
        optional_mesh_names = set(body_names) & OPTIONAL_COLLISION_BODIES
        for body_name in body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                if body_name in OPTIONAL_COLLISION_BODIES:
                    continue
                raise ValueError(f"G1 collision body is missing from the GMR model: {body_name}")
            body_ids.add(int(body_id))
        geom_ids: list[int] = []
        for geom_id in range(model.ngeom):
            mesh_name = None
            if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh_name = mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_MESH,
                    int(model.geom_dataid[geom_id]),
                )
            belongs = (
                int(model.geom_bodyid[geom_id]) in body_ids
                or mesh_name in optional_mesh_names
            )
            active = int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0
            if belongs and active:
                geom_ids.append(geom_id)
        geom_ids = tuple(geom_ids)
        if not geom_ids:
            raise ValueError(f"No active collision geoms found for G1 bodies: {body_names}")
        return geom_ids

    return [(geoms_for(first), geoms_for(second)) for first, second in G1_SELF_COLLISION_BODY_PAIRS]


def add_collision_avoidance_limit(retargeter: object, args: argparse.Namespace) -> dict[str, object]:
    """Attach Mink's self-collision QP limit and return traceable settings."""
    if not args.collision_avoidance:
        return {
            "enabled": False,
            "preset": "g1_self_collision_v2",
            "body_pair_groups": len(G1_SELF_COLLISION_BODY_PAIRS),
            "geom_pairs": 0,
        }

    import mink

    activated_hand_geoms = activate_required_collision_geoms(retargeter.model)
    geom_pairs = collision_geom_pairs(retargeter.model)
    limit = mink.CollisionAvoidanceLimit(
        retargeter.model,
        geom_pairs=geom_pairs,
        gain=float(args.collision_gain),
        minimum_distance_from_collisions=float(args.collision_min_distance),
        collision_detection_distance=float(args.collision_detection_distance),
        bound_relaxation=0.0,
        broadphase=True,
    )
    retargeter.ik_limits.append(limit)
    return {
        "enabled": True,
        "preset": "g1_self_collision_v2",
        "minimum_distance_m": float(args.collision_min_distance),
        "validation_distance_m": float(
            args.collision_min_distance + COLLISION_CLEARANCE_BUFFER_M
        ),
        "detection_distance_m": float(args.collision_detection_distance),
        "gain": float(args.collision_gain),
        "bound_relaxation": 0.0,
        "body_pair_groups": len(G1_SELF_COLLISION_BODY_PAIRS),
        "geom_pairs": int(limit.max_num_contacts),
        "activated_hand_geoms": activated_hand_geoms,
    }


def install_gmr_mink_limits_compatibility(gmr_motion_retarget: object) -> str:
    """Make GMR's legacy positional solve_ik call reach Mink's `limits` argument.

    GMR commit bb1bbe4 calls solve_ik(..., damping, self.ik_limits). In Mink
    1.3, the sixth positional parameter is `safety_break` and `limits` is the
    seventh parameter. Without this adapter every configured GMR/Mink limit is
    silently bypassed.
    """
    solve_ik = gmr_motion_retarget.mink.solve_ik
    parameters = tuple(inspect.signature(solve_ik).parameters)
    if len(parameters) < 7 or parameters[5] != "safety_break" or parameters[6] != "limits":
        return "native_limits_position"

    def solve_ik_with_limits(
        configuration: object,
        tasks: object,
        dt: float,
        solver: str,
        damping: float = 1e-12,
        limits: object = None,
    ) -> np.ndarray:
        return solve_ik(
            configuration,
            tasks,
            dt,
            solver,
            damping=damping,
            limits=limits,
        )

    gmr_motion_retarget.mink.solve_ik = solve_ik_with_limits
    return "mink_1_3_keyword_limits_adapter"


def validate_joint_limits(model: object, qpos: np.ndarray, names: tuple[str, ...]) -> None:
    import mujoco

    violations: list[tuple[float, int, str, float, float, float]] = []
    for joint_id in range(model.njnt):
        address = int(model.jnt_qposadr[joint_id])
        if address < 7 or not bool(model.jnt_limited[joint_id]):
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        low, high = (float(value) for value in model.jnt_range[joint_id])
        values = qpos[:, address]
        below = low - values
        above = values - high
        excess = np.maximum(below, above)
        worst_frame = int(np.argmax(excess))
        if float(excess[worst_frame]) > 1e-7:
            logical_index = address - 7
            name = names[logical_index] if 0 <= logical_index < len(names) else f"qpos[{address}]"
            violations.append(
                (float(excess[worst_frame]), worst_frame, name, float(values[worst_frame]), low, high)
            )
    if violations:
        _excess, frame, name, value, low, high = max(violations)
        raise ValueError(
            f"GMR joint limit violation at frame {frame}: {name}={value:.8g}, "
            f"expected [{low:.8g}, {high:.8g}]."
        )


def clamp_joint_limits(model: object, qpos: np.ndarray) -> int:
    """Remove solver-scale boundary overshoot before validating/serializing qpos."""
    import mujoco

    changed = 0
    for joint_id in range(model.njnt):
        address = int(model.jnt_qposadr[joint_id])
        if address < 7 or not bool(model.jnt_limited[joint_id]):
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        low, high = (float(value) for value in model.jnt_range[joint_id])
        clipped = np.clip(qpos[:, address], low, high)
        changed += int(np.count_nonzero(clipped != qpos[:, address]))
        qpos[:, address] = clipped
    return changed


def normalize_root_quaternions(qpos: np.ndarray) -> None:
    root_quats = qpos[:, 3:7]
    norms = np.linalg.norm(root_quats, axis=1)
    if np.any(norms < 1e-8):
        frame = int(np.flatnonzero(norms < 1e-8)[0])
        raise ValueError(f"GMR generated a zero-length root quaternion at frame {frame}.")
    root_quats /= norms[:, None]
    for frame_index in range(1, len(root_quats)):
        if float(np.dot(root_quats[frame_index - 1], root_quats[frame_index])) < 0.0:
            root_quats[frame_index] *= -1.0


def _violates_configured_clearance(
    collision_states: tuple[
        tuple[object, object, set[tuple[int, int]], np.ndarray], ...
    ],
    qpos: np.ndarray,
    minimum_distance: float,
    tolerance: float = 1e-7,
) -> bool:
    """Check configured pairs against the requested positive separation."""
    import mujoco

    for model, data, geom_pair_ids, geom_pairs in collision_states:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            if pair in geom_pair_ids and float(contact.dist) < minimum_distance - tolerance:
                return True
        # MuJoCo 3.10 and 3.12 differ in which shallow self-contacts they add
        # to data.contact. mj_geomDistance is stable across both versions, so
        # use it as a narrow-phase fallback for bounding-sphere-near pairs.
        first_ids = geom_pairs[:, 0]
        second_ids = geom_pairs[:, 1]
        center_distance = np.linalg.norm(
            data.geom_xpos[first_ids] - data.geom_xpos[second_ids], axis=1
        )
        sphere_clearance = center_distance - (
            model.geom_rbound[first_ids] + model.geom_rbound[second_ids]
        )
        from_to = np.empty(6, dtype=np.float64)
        distance_limit = max(minimum_distance + tolerance, 1e-4)
        for pair_index in np.flatnonzero(sphere_clearance < distance_limit):
            first = int(first_ids[pair_index])
            second = int(second_ids[pair_index])
            distance = mujoco.mj_geomDistance(
                model, data, first, second, distance_limit, from_to
            )
            if float(distance) < minimum_distance - tolerance:
                return True
    return False


def _segment_violates_configured_clearance(
    collision_states: tuple[
        tuple[object, object, set[tuple[int, int]], np.ndarray], ...
    ],
    previous: np.ndarray,
    candidate: np.ndarray,
    minimum_distance: float,
    samples: int = 40,
) -> bool:
    """Sample the same linear joint interpolation used by trajectory preview."""
    joint_delta = candidate[7:] - previous[7:]
    for sample_index in range(1, samples + 1):
        amount = sample_index / samples
        probe = candidate.copy()
        probe[7:] = previous[7:] + amount * joint_delta
        if _violates_configured_clearance(collision_states, probe, minimum_distance):
            return True
    return False


def limit_qpos_velocity_collision_aware(
    models: tuple[object, ...],
    qpos: np.ndarray,
    fps: float,
    *,
    max_joint_speed: float,
    max_root_speed: float,
    max_root_angular_speed: float,
    minimum_collision_distance: float = DEFAULT_COLLISION_MIN_DISTANCE_M,
    line_search_iterations: int = 14,
    stabilization_passes: int = 2,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply source-rate limits without interpolating through the robot body."""
    import mujoco

    if line_search_iterations < 1:
        raise ValueError("line_search_iterations must be positive.")
    if stabilization_passes < 1:
        raise ValueError("stabilization_passes must be positive.")
    result = qpos.copy()
    counts = {
        "root_position": 0,
        "root_rotation": 0,
        "joints": 0,
        "collision_frames_adjusted": 0,
        "collision_line_search_evaluations": 0,
        "initial_frame_penetrating": 0,
        "initial_frame_adjusted": 0,
        "collision_filter_passes": 1,
    }
    if not models:
        raise ValueError("At least one collision-validation model is required.")
    collision_states = []
    for model in models:
        if int(model.nq) != int(qpos.shape[1]):
            raise ValueError(
                f"Collision-validation model nq={model.nq} does not match qpos width {qpos.shape[1]}."
            )
        expanded_pairs = {
            tuple(sorted((first, second)))
            for first_group, second_group in collision_geom_pairs(model)
            for first in first_group
            for second in second_group
        }
        pair_array = np.asarray(sorted(expanded_pairs), dtype=np.int32)
        collision_states.append(
            (model, mujoco.MjData(model), expanded_pairs, pair_array)
        )
    states = tuple(collision_states)
    counts["initial_frame_penetrating"] = int(
        _violates_configured_clearance(states, result[0], minimum_collision_distance)
    )
    if counts["initial_frame_penetrating"]:
        # There is no preceding trajectory pose for frame zero. Use the
        # robot's neutral joint configuration at the requested floating-base
        # pose as a known-safe anchor, then retain as much of the requested
        # first pose as collision clearance permits.
        requested = result[0].copy()
        anchor = requested.copy()
        anchor[7:] = 0.0
        if _violates_configured_clearance(states, anchor, minimum_collision_distance):
            raise ValueError(
                "Neutral joint configuration penetrates a configured collision pair."
            )
        joint_delta = requested[7:] - anchor[7:]
        safe_amount = 0.0
        colliding_amount = 1.0
        for _iteration in range(line_search_iterations):
            amount = 0.5 * (safe_amount + colliding_amount)
            probe = requested.copy()
            probe[7:] = anchor[7:] + amount * joint_delta
            counts["collision_line_search_evaluations"] += 1
            if _segment_violates_configured_clearance(
                states, anchor, probe, minimum_collision_distance
            ):
                colliding_amount = amount
            else:
                safe_amount = amount
        result[0, 7:] = anchor[7:] + safe_amount * joint_delta
        counts["initial_frame_adjusted"] = 1

    for frame_index in range(1, len(result)):
        limited_pair, local_counts = limit_qpos_velocity(
            np.stack((result[frame_index - 1], qpos[frame_index])),
            fps,
            max_joint_speed=max_joint_speed,
            max_root_speed=max_root_speed,
            max_root_angular_speed=max_root_angular_speed,
        )
        for key in ("root_position", "root_rotation", "joints"):
            counts[key] += int(local_counts[key])
        candidate = limited_pair[1]
        if not _segment_violates_configured_clearance(
            states, result[frame_index - 1], candidate, minimum_collision_distance
        ):
            result[frame_index] = candidate
            continue

        # The preceding pose is collision-free and the speed-limited candidate
        # is not. Find the furthest safe point on that continuous joint-space
        # segment. Root motion does not affect self-collision and stays at its
        # already speed-limited candidate value.
        previous_joints = result[frame_index - 1, 7:]
        joint_delta = candidate[7:] - previous_joints
        safe_amount = 0.0
        colliding_amount = 1.0
        for _iteration in range(line_search_iterations):
            amount = 0.5 * (safe_amount + colliding_amount)
            probe = candidate.copy()
            probe[7:] = previous_joints + amount * joint_delta
            counts["collision_line_search_evaluations"] += 1
            if _segment_violates_configured_clearance(
                states, result[frame_index - 1], probe, minimum_collision_distance
            ):
                colliding_amount = amount
            else:
                safe_amount = amount
        candidate[7:] = previous_joints + safe_amount * joint_delta
        result[frame_index] = candidate
        counts["collision_frames_adjusted"] += 1

    # A second pass removes rare sampling aliases around very narrow contact
    # intervals. It operates on the already-safe trajectory, so it normally
    # changes nothing and otherwise makes only a sub-step correction.
    if stabilization_passes > 1:
        result, extra_counts = limit_qpos_velocity_collision_aware(
            models,
            result,
            fps,
            max_joint_speed=max_joint_speed,
            max_root_speed=max_root_speed,
            max_root_angular_speed=max_root_angular_speed,
            minimum_collision_distance=minimum_collision_distance,
            line_search_iterations=line_search_iterations,
            stabilization_passes=stabilization_passes - 1,
        )
        for key in (
            "root_position",
            "root_rotation",
            "joints",
            "collision_frames_adjusted",
            "collision_line_search_evaluations",
            "initial_frame_adjusted",
            "collision_filter_passes",
        ):
            counts[key] += int(extra_counts[key])
    return result, counts


def retarget(args: argparse.Namespace) -> dict[str, object]:
    gmr_root = args.gmr_root.resolve()
    motion_path = args.motion.resolve()
    model_path = args.smpl_model_path.resolve()
    if not (gmr_root / "general_motion_retargeting").is_dir():
        raise FileNotFoundError(f"GMR package not found under {gmr_root}.")
    if not motion_path.is_file():
        raise FileNotFoundError(f"AIST++ motion not found: {motion_path}")
    if not (model_path / "SMPL_NEUTRAL.pkl").is_file():
        raise FileNotFoundError(f"Prepared neutral SMPL model not found under {model_path}.")
    sys.path.insert(0, str(gmr_root))
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting import motion_retarget as gmr_motion_retarget

    limits_api = install_gmr_mink_limits_compatibility(gmr_motion_retarget)

    poses, translations = load_aistpp_motion(motion_path)
    rest_joints, parents = load_smpl_rest_pose(model_path, "NEUTRAL")
    world_positions, world_rotations = smpl_world_kinematics(
        poses, translations, rest_joints, parents
    )
    human_frames = gmr_smpl_frames(world_positions, world_rotations)
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="unitree_g1",
        actual_human_height=HUMAN_HEIGHT_M,
        solver=args.solver,
        verbose=False,
        use_velocity_limit=args.velocity_limit,
    )
    collision_avoidance = add_collision_avoidance_limit(retargeter, args)
    names = qpos_joint_names(retargeter.model)
    if len(names) != 29 or retargeter.model.nq != 36:
        raise ValueError(
            f"Expected free root + 29 G1 joints, got nq={retargeter.model.nq}, names={len(names)}."
        )

    # Seed the nonlinear IK on the intended symmetric arm branch. Some AIST++
    # clips begin in an extreme pose; solving those directly from zero qpos can
    # converge to the mirrored right-arm local minimum even though the target is
    # reachable. The seed is not serialized and does not change source timing.
    seed_poses = np.zeros((1, 24, 3), dtype=np.float64)
    seed_poses[:, 16, 2] = -0.5 * np.pi
    seed_poses[:, 17, 2] = 0.5 * np.pi
    seed_positions, seed_rotations = smpl_world_kinematics(
        seed_poses, translations[:1], rest_joints, parents
    )
    seed_frame = gmr_smpl_frames(seed_positions, seed_rotations)[0]
    for _iteration in range(2):
        retargeter.retarget(seed_frame)

    qpos = np.empty((len(human_frames), retargeter.model.nq), dtype=np.float64)
    for frame_index, human_frame in enumerate(human_frames):
        qpos[frame_index] = retargeter.retarget(human_frame)
    if not np.all(np.isfinite(qpos)):
        indices = np.argwhere(~np.isfinite(qpos))[0].tolist()
        raise ValueError(f"GMR generated a non-finite qpos at index {indices}.")

    joint_limit_count = clamp_joint_limits(retargeter.model, qpos)
    # Collision geometry must be evaluated using the exact normalized root
    # quaternions that will be serialized and replayed.
    normalize_root_quaternions(qpos)
    if args.collision_avoidance:
        import mujoco

        validation_model_path = args.collision_validation_model.resolve()
        if not validation_model_path.is_file():
            raise FileNotFoundError(
                f"Collision-validation MuJoCo model not found: {validation_model_path}"
            )
        validation_model = mujoco.MjModel.from_xml_path(str(validation_model_path))
        qpos, limited_counts = limit_qpos_velocity_collision_aware(
            (retargeter.model, validation_model),
            qpos,
            float(args.motion_fps),
            max_joint_speed=float(args.max_joint_speed),
            max_root_speed=float(args.max_root_speed),
            max_root_angular_speed=float(args.max_root_angular_speed),
            minimum_collision_distance=float(
                args.collision_min_distance + COLLISION_CLEARANCE_BUFFER_M
            ),
        )
        collision_avoidance["validation_models"] = [
            Path(retargeter.xml_file).name,
            validation_model_path.name,
        ]
    else:
        qpos, limited_counts = limit_qpos_velocity(
            qpos,
            float(args.motion_fps),
            max_joint_speed=float(args.max_joint_speed),
            max_root_speed=float(args.max_root_speed),
            max_root_angular_speed=float(args.max_root_angular_speed),
        )
    normalize_root_quaternions(qpos)
    limited_counts["joint_limits"] = joint_limit_count
    validate_joint_limits(retargeter.model, qpos, names)
    return {
        "format_version": FORMAT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "source_format": SOURCE_FORMAT,
        "fps": float(args.motion_fps),
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7],
        "root_rot_order": "wxyz",
        "dof_pos": qpos[:, 7:],
        "dof_names": list(names),
        "source_motion_id": args.source_motion_id,
        "source_sha256": args.source_sha256,
        "smpl_model_sha256": args.smpl_model_sha256,
        "retargeter": "GMR",
        "retargeter_version": args.retargeter_version,
        "collision_avoidance": collision_avoidance,
        "mink_limits_api": limits_api,
        "continuity_limits": {
            "max_joint_speed_rad_s": float(args.max_joint_speed),
            "max_root_speed_m_s": float(args.max_root_speed),
            "max_root_angular_speed_rad_s": float(args.max_root_angular_speed),
            "limited_values": limited_counts,
            "loop_closure_frames": 0,
        },
    }


def main() -> int:
    args = parse_args()
    if args.motion_fps <= 0:
        raise ValueError("--motion-fps must be positive.")
    if args.collision_min_distance < 0:
        raise ValueError("--collision-min-distance must be non-negative.")
    if args.collision_detection_distance <= args.collision_min_distance:
        raise ValueError("--collision-detection-distance must exceed --collision-min-distance.")
    if not 0.0 < args.collision_gain <= 1.0:
        raise ValueError("--collision-gain must be in (0, 1].")
    payload = retarget(args)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)
    print(f"Wrote {output_path} ({len(payload['root_pos'])} frames, 29 DoF, SMPL direct).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
