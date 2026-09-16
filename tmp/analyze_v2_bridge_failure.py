import sys, csv, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'realtime/humanoid_robot/src'))
import realtime_music_humanoid_matcher_v2 as v

sys.argv = [sys.argv[0], '--headless']
args = v.parse_args()
catalog = v.MusicCatalog.load(args.catalog if args.catalog.is_absolute() else v.ROOT / args.catalog)
rows = list(csv.DictReader(open('tmp/v2_chronos_retest_20260916.csv')))
pools = list(dict.fromkeys(r['shortlist'] for r in rows if r['bridge_generation']=='2' and r['bridge_preparation_status']=='preparing'))
source_id = 'gPO_sBM_cAll_d11_mPO0_ch02'
ids = [source_id] + list(dict.fromkeys(';'.join(pools).split(';'))) + ['gHO_sFM_cAll_d21_mHO5_ch20']
player = v.base.MujocoHumanoidPlayer(args.model, realtime=False, headless=True)
limits_path = args.output_joint_limits_json
if limits_path is not None and not limits_path.is_absolute(): limits_path = v.ROOT / limits_path
limits = v.load_joint_dynamics_limits(limits_path, player.actuator_names, default_speed=args.output_max_joint_speed, default_acceleration=args.output_max_joint_acceleration)
names = sorted(limits)
ranges = player.actuator_joint_ranges
lo, hi = np.array([ranges[n][0] for n in names]), np.array([ranges[n][1] for n in names])
speed = np.array([limits[n].max_speed_rad_s for n in names])
acc = np.array([limits[n].max_acceleration_rad_s2 for n in names])
samplers = {}
for mid in ids:
    s = v.load_motion_sampler(args, catalog, catalog.motions[mid])
    adapter = v.base.make_pose_adapter(args, player, s)
    player.ground_sampler(s, adapter, cooperative_yield=False)
    f = v.build_motion_entry_features(s, catalog.motions[mid], adapter, ranges)
    s.entry_features = f
    samplers[mid] = s
    cols = [f.joint_names.index(n) for n in names]
    q, vel, a = [getattr(f.authored, attr)[:,cols] for attr in ('positions','velocities','accelerations')]
    masks = [np.all((q>=lo-1e-8)&(q<=hi+1e-8),axis=1), np.all(abs(vel)<=speed+1e-7,axis=1),np.all(abs(a)<=acc+1e-7,axis=1)]
    remaining = f.authored.duration-np.arange(len(q))/f.fps >= args.entry_min_remaining_seconds-1e-9
    record = dict(id=mid, duration=f.authored.duration, frames=len(q), remaining_ok=int(remaining.sum()), position_ok=int((remaining&masks[0]).sum()), velocity_ok=int((remaining&masks[1]).sum()), acceleration_ok=int((remaining&masks[2]).sum()), eligible=int((remaining & masks[0]&masks[1]&masks[2]).sum()))
    if mid==source_id:
        tail = np.arange(len(q))/f.fps >= f.authored.duration-args.transition_exit_window_seconds-1e-9
        record['valid_exits'] = np.flatnonzero(tail&masks[0]&masks[1]&masks[2]).tolist()
        record['ankle_605'] = [float(x[605,0]) for x in (q,vel,a)]
        record['ankle_0'] = [float(x[0,0]) for x in (q,vel,a)]
    print('AUDIT '+json.dumps(record), flush=True)
jerks = v.load_jerk_limits(limits_path, player.actuator_names, args.output_max_joint_jerk)
v.warmup_hermite()
for label, pool in [('initial', pools[0].split(';')), ('expanded', pools[1].split(';')), ('known_alternative', [ids[-1]])]:
    try:
        plan = v.prepare_state_bridge(0, source_id, samplers[source_id], [(mid,samplers[mid],(0.,0.)) for mid in pool], limits, ranges, jerks, args)
        print('SEARCH', label, plan.motion_id, plan.exit_frame_index, plan.score.frame_index, plan.trajectory.duration, flush=True)
    except Exception as exc: print('SEARCH',label,str(exc),flush=True)
args.entry_min_remaining_seconds = 6.0
try:
    plan = v.prepare_state_bridge(0, source_id, samplers[source_id], [(mid,samplers[mid],(0.,0.)) for mid in pools[0].split(';')], limits, ranges, jerks, args)
    print('SEARCH minimum_6_seconds', plan.motion_id, plan.exit_frame_index, plan.score.frame_index, plan.trajectory.duration, flush=True)
except Exception as exc: print('SEARCH minimum_6_seconds',str(exc),flush=True)
