import argparse
from dataclasses import dataclass
import os
from pathlib import Path

NUMBA_CACHE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "numba_cache"
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

import librosa
import numpy as np
import pandas as pd


@dataclass
class Config:
    audio_path: Path
    output_csv: Path
    traj_hz: float
    sr: int
    hop_length: int
    yaw_amp_deg: float
    open_pitch_deg: float
    clap_pitch_deg: float
    bob_deg: float
    accent_duration: float
    accent_gain_deg: float
    onset_quantile: float


def smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


def normalize_rms(rms: np.ndarray) -> np.ndarray:
    if np.allclose(rms.max(), rms.min()):
        return np.ones_like(rms)
    n = (rms - rms.min()) / (rms.max() - rms.min())
    # map to about [0.5, 1.5] as amplitude modulator
    return np.clip(0.5 + n, 0.5, 1.5)


def compute_phase(t: np.ndarray, beats: np.ndarray, fallback_bpm: float) -> np.ndarray:
    if beats.size < 2:
        period = 60.0 / max(fallback_bpm, 1e-6)
        return np.mod(t / period, 1.0)

    idx = np.searchsorted(beats, t, side="right") - 1
    idx = np.clip(idx, 0, beats.size - 2)
    t0 = beats[idx]
    t1 = beats[idx + 1]
    denom = np.maximum(t1 - t0, 1e-6)
    phase = (t - t0) / denom
    return np.clip(phase, 0.0, 0.999999)


def build_pose_track(t: np.ndarray, beats: np.ndarray, amp_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Beat-to-beat clap keyframes: open -> clap -> open.
    poses_deg = np.array([
        [0.0, 22.0],
        [0.0, -24.0],
    ])
    poses = np.deg2rad(poses_deg)

    if beats.size < 2:
        yaw = np.zeros_like(t)
        pitch = np.zeros_like(t)
        return yaw, pitch

    idx = np.searchsorted(beats, t, side="right") - 1
    idx = np.clip(idx, 0, beats.size - 2)

    k0 = idx % len(poses)
    k1 = (idx + 1) % len(poses)

    t0 = beats[idx]
    t1 = beats[idx + 1]
    phase = np.clip((t - t0) / np.maximum(t1 - t0, 1e-6), 0.0, 1.0)
    s = smoothstep(phase)

    q0 = poses[k0]
    q1 = poses[k1]
    q = (1.0 - s)[:, None] * q0 + s[:, None] * q1

    # Loudness-dependent scaling
    q = q * amp_t[:, None]
    return q[:, 0], q[:, 1]


def build_clap_motion(
    phase: np.ndarray,
    amp_t: np.ndarray,
    yaw_amp_deg: float,
    open_pitch_deg: float,
    clap_pitch_deg: float,
    bob_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close_env = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
    yaw = np.deg2rad(yaw_amp_deg) * np.sin(2.0 * np.pi * phase)
    pitch = np.deg2rad(open_pitch_deg + (clap_pitch_deg - open_pitch_deg) * close_env)
    pitch += np.deg2rad(bob_deg) * np.sin(4.0 * np.pi * phase)

    neutral_pitch = np.deg2rad(open_pitch_deg)
    yaw = yaw * amp_t
    pitch = neutral_pitch + (pitch - neutral_pitch) * amp_t
    return yaw, pitch, close_env


def build_keypoints(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keypoint = np.zeros_like(phase, dtype=int)
    keypoint_label = np.full(phase.shape, "", dtype=object)
    beat_weight = np.zeros_like(phase)

    if phase.size == 0:
        return keypoint, keypoint_label, beat_weight

    open_idx = int(np.argmin(np.abs(phase - 0.0)))
    clap_idx = int(np.argmin(np.abs(phase - 0.5)))

    keypoint[open_idx] = 1
    keypoint_label[open_idx] = "open"
    beat_weight[open_idx] = 0.5
    keypoint[clap_idx] = 1
    keypoint_label[clap_idx] = "clap"
    beat_weight[clap_idx] = 1.0
    return keypoint, keypoint_label, beat_weight


def build_onset_pulse(
    t: np.ndarray,
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    duration: float,
    gain_rad: float,
    quantile: float,
) -> np.ndarray:
    if onset_times.size == 0 or onset_strengths.size == 0:
        return np.zeros_like(t)

    thr = float(np.quantile(onset_strengths, quantile))
    strong = onset_strengths >= thr
    selected = onset_times[strong]
    if selected.size == 0:
        return np.zeros_like(t)

    pulse = np.zeros_like(t)
    for i, ts in enumerate(selected):
        dt = t - ts
        mask = (dt >= 0.0) & (dt <= duration)
        if not np.any(mask):
            continue
        env = np.sin(np.pi * dt[mask] / duration)
        # alternate direction to look less mechanical
        sign = 1.0 if (i % 2 == 0) else -1.0
        pulse[mask] += sign * gain_rad * env

    return pulse


def generate(cfg: Config) -> pd.DataFrame:
    y, sr = librosa.load(cfg.audio_path, sr=cfg.sr, mono=True)
    dur = librosa.get_duration(y=y, sr=sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=cfg.hop_length)
    bpm = float(np.atleast_1d(tempo)[0])
    if bpm <= 0.0 or not np.isfinite(bpm):
        bpm = 120.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=cfg.hop_length)

    rms = librosa.feature.rms(y=y, hop_length=cfg.hop_length)[0]
    rms_times = librosa.times_like(rms, sr=sr, hop_length=cfg.hop_length)
    amp_norm = normalize_rms(rms)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=cfg.hop_length)
    onset_times = librosa.times_like(onset_env, sr=sr, hop_length=cfg.hop_length)

    n = int(np.floor(dur * cfg.traj_hz)) + 1
    t = np.linspace(0.0, dur, n)

    amp_t = np.interp(t, rms_times, amp_norm, left=amp_norm[0], right=amp_norm[-1])
    phase = compute_phase(t, beat_times, bpm)

    yaw_clap, pitch_clap, close_env = build_clap_motion(
        phase=phase,
        amp_t=amp_t,
        yaw_amp_deg=cfg.yaw_amp_deg,
        open_pitch_deg=cfg.open_pitch_deg,
        clap_pitch_deg=cfg.clap_pitch_deg,
        bob_deg=cfg.bob_deg,
    )

    # Beat keyframe interpolation (smoothstep)
    yaw_pose, pitch_pose = build_pose_track(t, beat_times, amp_t)

    # Onset accent pulse
    accent = build_onset_pulse(
        t=t,
        onset_times=onset_times,
        onset_strengths=onset_env,
        duration=cfg.accent_duration,
        gain_rad=np.deg2rad(cfg.accent_gain_deg),
        quantile=cfg.onset_quantile,
    )

    yaw = yaw_clap + 0.15 * yaw_pose + 0.20 * accent
    pitch = pitch_clap + 0.15 * pitch_pose + 0.85 * accent
    keypoint, keypoint_label, beat_weight = build_keypoints(phase)

    df = pd.DataFrame(
        {
            "t": t,
            "yaw_rad": yaw,
            "pitch_rad": pitch,
            "yaw_deg": np.rad2deg(yaw),
            "pitch_deg": np.rad2deg(pitch),
            "phase": phase,
            "close_env": close_env,
            "amp_scale": amp_t,
            "accent": accent,
            "tempo_bpm": bpm,
            "keypoint": keypoint,
            "keypoint_label": keypoint_label,
            "beat_weight": beat_weight,
        }
    )
    return df


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Generate beat-synced robot trajectory CSV from audio (librosa)."
    )
    parser.add_argument("audio", type=Path, help="Input audio file path (wav/mp3/ogg/...) ")
    parser.add_argument("--out", type=Path, default=Path("outputs/trajectory.csv"), help="Output CSV path")
    parser.add_argument("--traj-hz", type=float, default=100.0, help="Output trajectory sample rate")
    parser.add_argument("--sr", type=int, default=22050, help="Audio load sample rate")
    parser.add_argument("--hop", type=int, default=512, help="Analysis hop length")
    parser.add_argument("--yaw-amp-deg", type=float, default=18.0, help="Side sway amplitude in degrees")
    parser.add_argument("--open-pitch-deg", type=float, default=22.0, help="Open clap pose pitch in degrees")
    parser.add_argument("--clap-pitch-deg", type=float, default=-24.0, help="Closed clap pose pitch in degrees")
    parser.add_argument("--bob-deg", type=float, default=5.0, help="Small clap bob amplitude in degrees")
    parser.add_argument("--accent-duration", type=float, default=0.15, help="Accent pulse length in seconds")
    parser.add_argument("--accent-gain-deg", type=float, default=12.0, help="Accent pulse gain in degree")
    parser.add_argument("--onset-quantile", type=float, default=0.90, help="Quantile threshold for strong onsets")

    a = parser.parse_args()
    return Config(
        audio_path=a.audio,
        output_csv=a.out,
        traj_hz=a.traj_hz,
        sr=a.sr,
        hop_length=a.hop,
        yaw_amp_deg=a.yaw_amp_deg,
        open_pitch_deg=a.open_pitch_deg,
        clap_pitch_deg=a.clap_pitch_deg,
        bob_deg=a.bob_deg,
        accent_duration=a.accent_duration,
        accent_gain_deg=a.accent_gain_deg,
        onset_quantile=a.onset_quantile,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = generate(cfg)
    df.to_csv(cfg.output_csv, index=False)
    print(f"Saved trajectory: {cfg.output_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
