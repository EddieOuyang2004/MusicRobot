"""Audit within-clip GMR motion at authored speed; never modifies motion files.

Run with the project .venv Python. Speed, acceleration and jerk peaks of the V2
quintic are exact: every stationary point is found in closed form. Only the
position-overshoot metric is sampled, so it alone is a lower bound.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.spatial.transform import Rotation

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from motion_bridges import AuthoredTrajectory
from unitree_g1_dance_adapter import UnitreeG1DanceAdapter

HUMANOID = SRC.parent


def derivatives(q, fps):
    return [np.diff(q, n=k, axis=0) * fps**k for k in (1, 2, 3)]


def peak(a):
    return float(np.max(np.abs(a))) if a.size else 0.0


def horner(c, u):
    """Evaluate ascending coefficients c at a per-element normalized time u."""
    value = np.broadcast_to(c[-1], u.shape).astype(float)
    for k in range(len(c)-2, -1, -1):
        value = value*u + c[k]
    return value


def unit(u):
    # Endpoints are always candidates, so clamping maps useless roots onto them.
    return np.clip(np.where(np.isfinite(u), u, 0.), 0., 1.)


def linear_root(a, b):
    return unit(np.divide(-b, a, out=np.zeros_like(np.asarray(b, dtype=float)), where=a != 0))


def quadratic_roots(a, b, c):
    degenerate = linear_root(b, c)
    discriminant = b*b - 4*a*c
    usable = (a != 0) & (discriminant >= 0)
    root = np.sqrt(np.where(usable, discriminant, 0.))
    denominator = np.where(a != 0, 2*a, 1.)
    return tuple(unit(np.where(usable, (-b+sign*root)/denominator, degenerate))
                 for sign in (1., -1.))


def cubic_roots(a, b, c, d):
    """Real roots of a cubic in [0, 1]; a == 0 falls back to the quadratic."""
    degenerate = quadratic_roots(b, c, d)
    scale = np.where(a != 0, a, 1.)
    b, c, d = b/scale, c/scale, d/scale
    shift = b/3.
    p = c - b*b/3.
    q = 2*b**3/27. - b*c/3. + d
    discriminant = q*q/4. + p**3/27.
    three = (discriminant <= 0) & (p < 0)
    safe = np.where(three, p, -1.)
    radius = 2*np.sqrt(-safe/3.)
    angle = np.arccos(np.clip(3*q/(safe*radius), -1., 1.))/3.
    offset = np.sqrt(np.maximum(discriminant, 0.))
    single = np.cbrt(-q/2.+offset) + np.cbrt(-q/2.-offset) - shift
    return tuple(unit(np.where(a != 0, np.where(three, radius*np.cos(angle-2*np.pi*k/3.)-shift, single),
                               degenerate[min(k, 1)])) for k in range(3))


def exact_peaks(segments, fps):
    """Exact |derivative| maxima per segment and joint, with their locations."""
    v, a, j, s = (poly.polyder(segments, m=m)*fps**m for m in (1, 2, 3, 4))
    ends = (np.zeros(segments.shape[1:]), np.ones(segments.shape[1:]))
    peaks = {}
    for key, coefficients, candidates in (
            ('speed_rad_s', v, ends + cubic_roots(a[3], a[2], a[1], a[0])),
            ('acceleration_rad_s2', a, ends + quadratic_roots(j[2], j[1], j[0])),
            ('jerk_rad_s3', j, ends + (linear_root(s[1], s[0]),))):
        values = np.stack([np.abs(horner(coefficients, u)) for u in candidates])
        best = values.argmax(axis=0)
        peaks[key] = (np.take_along_axis(values, best[None], 0)[0],
                      np.take_along_axis(np.stack(candidates), best[None], 0)[0])
    return peaks


def audit(path, ranges):
    raw = path.read_bytes()
    d = pickle.loads(raw)
    q = np.asarray(d['dof_pos'], dtype=float)
    p = np.asarray(d['root_pos'], dtype=float)
    quat = np.asarray(d['root_rot'], dtype=float)
    fps = float(d['fps'])
    names = list(d['dof_names'])
    if (q.ndim != 2 or q.shape[1] != 29 or len(q) < 4 or
            p.shape != (len(q), 3) or quat.shape != (len(q), 4) or
            not all(np.all(np.isfinite(a)) for a in (q, p, quat)) or
            not np.isfinite(fps) or fps <= 0 or
            tuple(names) != UnitreeG1DanceAdapter.GMR_DOF_NAMES or
            d['root_rot_order'] != 'wxyz' or
            not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError('Invalid shape, names, quaternion, FPS, or nonfinite data')
    v, a, j = derivatives(q, fps)
    lower = np.array([ranges[n][0] for n in names])
    upper = np.array([ranges[n][1] for n in names])
    overshoot = np.maximum(lower-q, q-upper)
    row = dict(motion_id=path.stem, sha256=hashlib.sha256(raw).hexdigest(),
               frames=len(q), fps=fps, duration_s=(len(q)-1)/fps,
               raw_step_rad=peak(np.diff(q, axis=0)), raw_speed_rad_s=peak(v),
               raw_acceleration_rad_s2=peak(a), raw_jerk_rad_s3=peak(j),
               raw_acceleration_p99_rad_s2=float(np.percentile(np.abs(a), 99)),
               raw_position_excess_rad=max(0., float(overshoot.max())))
    lim = d['continuity_limits']
    clamp = float(lim['max_joint_speed_rad_s'])
    row['raw_speed_violating_intervals'] = int(np.sum(np.any(np.abs(v) > clamp+1e-5, axis=1)))
    # GMR clamps per-frame joint speed, so saturated frames that alternate in
    # sign are chatter the exporter could not remove, not interpolation error.
    saturated = np.abs(v) > clamp-1e-6
    reversal = (v[:-1]*v[1:] < 0) & (np.minimum(np.abs(v[:-1]), np.abs(v[1:])) > .5*clamp)
    row['export_speed_clamp_fraction'] = float(saturated.mean())
    row['fast_reversal_samples'] = int(reversal.sum())
    row['fast_reversal_frames'] = int(np.any(reversal, axis=1).sum())
    counts = lim.get('limited_values', {})
    for key in ('joints', 'joint_limits', 'collision_frames_adjusted'):
        row['gmr_limited_'+key] = int(counts.get(key, 0))
    # V2 loads the artifact joint positions as float32 before interpolation.
    trajectory = AuthoredTrajectory(q.astype(np.float32).astype(float), fps)
    c = trajectory.segments
    peaks = exact_peaks(c, fps)
    for key, (values, _) in peaks.items():
        row['v2_'+key] = peak(values)
    row['v2_speed_bad_segments'] = int(np.any(peaks['speed_rad_s'][0] > 16.+1e-5, axis=1).sum())
    accel, location = peaks['acceleration_rad_s2']
    row['v2_acceleration_bad_segments'] = int(np.any(accel > 1600.+1e-5, axis=1).sum())
    segment, joint = np.unravel_index(int(np.argmax(accel)), accel.shape)
    row['worst_acceleration_joint'] = names[joint]
    row['worst_acceleration_time_s'] = (segment+float(location[segment, joint]))/fps
    # Position overshoot is the one sampled metric, so it is a lower bound.
    row['v2_position_excess_rad'] = max(
        float(np.max(np.maximum(lower-values, values-upper)))
        for values in (poly.polyval(u, c) for u in np.linspace(0., 1., 33)))
    row['v2_position_excess_rad'] = max(0., row['v2_position_excess_rad'])
    dc = poly.polyder(c, m=3)*fps**3
    row['v2_jerk_jump_rad_s3'] = peak(poly.polyval(1., dc)[:-1] - poly.polyval(0., dc)[1:])
    pv, pa, pj = derivatives(p, fps)
    rot = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    # World-frame rotation increments avoid quaternion sign/wrap artifacts.
    omega = (rot[1:] * rot[:-1].inv()).as_rotvec()*fps
    for key, values in [('root_speed_m_s', pv), ('root_acceleration_m_s2', pa),
                        ('root_jerk_m_s3', pj), ('root_angular_speed_rad_s', omega),
                        ('root_angular_acceleration_rad_s2', np.diff(omega, axis=0)*fps)]:
        row[key] = float(np.max(np.linalg.norm(values, axis=1)))
    row['root_speed_violation'] = row['root_speed_m_s'] > lim['max_root_speed_m_s']+1e-5
    row['root_angular_speed_violation'] = row['root_angular_speed_rad_s'] > lim['max_root_angular_speed_rad_s']+1e-5
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-root', type=Path, default=HUMANOID/'data/aistpp_gmr')
    parser.add_argument('--output', type=Path, default=SRC/'test/output/gmr_smoothness')
    args = parser.parse_args()
    ranges = {e.attrib['name'].removesuffix('_joint'): tuple(map(float, e.attrib['range'].split()))
              for e in ET.parse(HUMANOID/'assets/g1_29dof.xml').iter('joint') if 'range' in e.attrib and 'name' in e.attrib}
    files = sorted(args.motion_root.glob('*.pkl'))
    rows, errors = [], []
    for i, path in enumerate(files):
        try:
            rows.append(audit(path, ranges))
        except Exception as exc:
            errors.append(dict(motion_id=path.stem, error=str(exc)))
        if (i+1) % 50 == 0:
            print(f'Audited {i+1}/{len(files)}', flush=True)
    catalog = json.loads((HUMANOID/'data/music_catalog/catalog.json').read_text())
    source_ids = {p.stem for p in (HUMANOID/'data/aistpp/motions').glob('*.pkl')}
    ids = {p.stem for p in files}
    summary = dict(artifact_count=len(files), valid_count=len(rows), errors=errors,
                   catalog_count=len(catalog['motions']), source_count=len(source_ids),
                   missing_catalog_artifacts=sorted(set(catalog['motions'])-ids),
                   missing_source_artifacts=sorted(source_ids-ids),
                   total_frames=sum(r['frames'] for r in rows))
    for key in ('raw_speed_violating_intervals', 'v2_speed_bad_segments', 'v2_acceleration_bad_segments',
                'root_speed_violation', 'root_angular_speed_violation', 'fast_reversal_frames'):
        summary['clips_with_'+key] = sum(r[key] > 0 for r in rows)
    summary['clean_clips'] = sum(r['v2_speed_bad_segments'] == 0 and r['v2_acceleration_bad_segments'] == 0
                                 and r['fast_reversal_frames'] == 0 for r in rows)
    summary['fast_reversal_frames'] = sum(r['fast_reversal_frames'] for r in rows)
    summary['export_speed_clamp_fraction'] = (
        sum(r['export_speed_clamp_fraction']*(r['frames']-1) for r in rows) /
        max(sum(r['frames']-1 for r in rows), 1))
    for key in ('raw_position_excess_rad', 'v2_position_excess_rad'):
        summary['clips_with_'+key] = sum(r[key] > 1e-5 for r in rows)
    intervals = sum(r['frames']-1 for r in rows)
    summary['acceleration_bad_interval_fraction'] = (
        sum(r['v2_acceleration_bad_segments'] for r in rows)/max(intervals, 1))
    shares = sorted(r['v2_acceleration_bad_segments']/max(r['frames']-1, 1) for r in rows)
    summary['acceleration_bad_interval_share_per_clip'] = {
        f'p{p}': float(np.percentile(shares, p)) for p in (50, 90, 100)} if shares else {}
    summary['maxima'] = {key: max((r[key] for r in rows), default=None) for key in (
        'raw_acceleration_rad_s2', 'raw_jerk_rad_s3', 'v2_speed_rad_s',
        'v2_acceleration_rad_s2', 'v2_jerk_rad_s3', 'v2_position_excess_rad',
        'root_acceleration_m_s2', 'root_angular_acceleration_rad_s2')}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'audit.json').write_text(json.dumps(dict(summary=summary, motions=rows), indent=2)+'\n')
    if rows:
        with (args.output/'motions.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# AIST++ post-GMR smoothness audit', '',
             'Scope: all local exported clips, authored speed, within each clip only. No transitions or loop seams.', '',
             'Raw derivatives use first/second/third forward differences with the artifact FPS. '
             'V2 joint interpolation uses the actual AuthoredTrajectory implementation. '
             'Speed, acceleration and jerk peaks are exact: every stationary point of each quintic segment is solved in closed form. '
             'Position overshoot is sampled at 33 points per segment, so that metric alone is a lower bound. '
             'V2 reference thresholds are the defaults of its output limiter, --output-max-joint-speed 16 rad/s and '
             '--output-max-joint-acceleration 1600 rad/s². These are software limits, not perceptual smoothness or hardware certification. '
             'No jerk acceptance threshold is configured in V2. Joint ranges come from g1_29dof.xml. '
             'Root orientation differences use rotations, not quaternion component subtraction. '
             'Music modulation, time warping, runtime collision corrections and final output limiting are excluded.', '',
             '## Coverage and findings', '', '```json', json.dumps(summary, indent=2), '```', '',
             '## Largest interpolated acceleration peaks', '',
             '| Motion | Joint | Time (s) | Speed peak (rad/s) | Acceleration peak (rad/s²) | Jerk peak (rad/s³) | Saturated reversal frames |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in sorted(rows, key=lambda r:r['v2_acceleration_rad_s2'], reverse=True)[:20]:
        lines.append(f"| {r['motion_id']} | {r['worst_acceleration_joint']} | {r['worst_acceleration_time_s']:.4f} | "
                     f"{r['v2_speed_rad_s']:.2f} | {r['v2_acceleration_rad_s2']:.2f} | {r['v2_jerk_rad_s3']:.0f} | {r['fast_reversal_frames']} |")
    lines += ['', '## Interpretation', '',
              'C2 interpolation makes position, velocity and acceleration continuous, but does not bound their magnitudes or make jerk continuous. '
              'The dominant source of the peaks is the exported samples themselves: GMR clamps per-frame joint speed, and many clips hold that clamp '
              'while reversing direction from one frame to the next, which is a full-amplitude oscillation at the frame Nyquist rate. '
              'Central-difference velocities and accelerations at those frames feed the quintic endpoint states, and the interpolation then overshoots '
              'to roughly twice the export clamp. Filtering the interpolator alone would not remove chatter that is already present in the samples.', '',
              'Exceedances are exact, so they disprove compliance outright. A clip reported clean is clean for these three metrics at authored speed only; '
              'music-driven time warping raises both speed and acceleration further. '
              'Do not apply an unchecked low-pass filter to collision-constrained motions: any revised trajectory needs joint-range and collision revalidation.', '',
              'The CSV and JSON include every clip, root metrics, source artifact SHA-256 hashes, export clamp statistics and exact locations of acceleration peaks.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
