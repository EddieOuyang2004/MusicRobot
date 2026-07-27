from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import librosa
import numpy as np

import realtime_music_humanoid_dancer as base
from music_motion_catalog import (
    AudioFeatureExtractor,
    CandidateStabilizer,
    MatchResult,
    MotionProfile,
    MusicCatalog,
    MusicMotionMatcher,
    load_audio_mono,
)
from music_pose_modulator import MusicPoseModulator


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = (
    ROOT / "realtime" / "humanoid_robot" / "data" / "music_catalog" / "catalog.json"
)


def parse_args() -> argparse.Namespace:
    """Parse matcher-specific flags, then delegate all legacy flags to the frozen entrypoint."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--audio-input",
        type=Path,
        default=None,
        help="Use a WAV/audio file as a deterministic realtime source instead of the microphone.",
    )
    parser.add_argument("--embedding-model", type=Path, default=None)
    parser.add_argument("--tag-model", type=Path, default=None)
    parser.add_argument("--match-window-seconds", type=float, default=6.0)
    parser.add_argument("--match-interval-seconds", type=float, default=1.0)
    parser.add_argument("--match-top-tracks", type=int, default=5)
    parser.add_argument("--match-top-motions", type=int, default=10)
    parser.add_argument("--switch-required-wins", type=int, default=3)
    parser.add_argument("--switch-score-margin", type=float, default=0.08)
    parser.add_argument("--switch-beats-per-bar", type=int, default=4)
    parser.add_argument(
        "--matcher-help",
        action="store_true",
        help="Show the matcher-specific options; use --help for inherited dancer options.",
    )
    matcher_args, remaining = parser.parse_known_args()
    if matcher_args.matcher_help:
        help_parser = argparse.ArgumentParser(
            description=(
                "Realtime AIST++ music retrieval and motion switching. "
                "All realtime_music_humanoid_dancer.py options are also accepted."
            )
        )
        for action in parser._actions:
            if action.dest != "help":
                help_parser._add_action(action)
        help_parser.print_help()
        raise SystemExit(0)

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = base.parse_args()
    finally:
        sys.argv = original_argv
    for name, value in vars(matcher_args).items():
        setattr(args, name, value)
    return args


class MicrophoneSource:
    def __init__(self, analyzer: base.RealtimeMusicAnalyzer, window_seconds: float) -> None:
        self.analyzer = analyzer
        self.window_seconds = float(window_seconds)

    @property
    def playback_seconds(self) -> float:
        return time.perf_counter()

    @property
    def done(self) -> bool:
        return False

    def start(self) -> None:
        self.analyzer.start()

    def stop(self) -> None:
        self.analyzer.stop()

    def advance(self, dt: float) -> None:
        del dt

    def drain(self) -> list[base.MusicFrame]:
        return self.analyzer.drain()

    def recent_audio(self) -> np.ndarray | None:
        _start, audio = self.analyzer._audio_window()
        required = int(round(self.window_seconds * self.analyzer.sample_rate))
        if audio.size < required:
            return None
        return np.asarray(audio[-required:], dtype=np.float32)


class SilentSource:
    """No-input source used for headless and hardware-free smoke tests."""

    @property
    def playback_seconds(self) -> float:
        return time.perf_counter()

    @property
    def done(self) -> bool:
        return False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def advance(self, dt: float) -> None:
        del dt

    def drain(self) -> list[base.MusicFrame]:
        return []

    def recent_audio(self) -> None:
        return None


class WavSource:
    """Feed the existing realtime analyzer from a file using simulated audio time."""

    def __init__(
        self,
        path: Path,
        args: argparse.Namespace,
        window_seconds: float,
    ) -> None:
        self.path = path
        self.sample_rate = int(args.mic_sample_rate)
        self.block_size = int(args.mic_block_size)
        self.window_seconds = float(window_seconds)
        self.audio = load_audio_mono(path, self.sample_rate)
        self.cursor = 0
        self.target_cursor = 0.0
        self.analyzer = base.RealtimeMusicAnalyzer(
            sample_rate=self.sample_rate,
            block_size=self.block_size,
            onset_threshold_scale=args.onset_threshold_scale,
            min_beat_period=args.min_beat_period,
            max_beat_period=args.max_beat_period,
            refractory_sec=args.mic_refractory_sec,
            noise_gate_rms=0.0,
            noise_gate_ratio=1.0,
            startup_calibration_sec=0.0,
            plp_history_sec=max(args.plp_history_sec, window_seconds),
            plp_analysis_interval_sec=args.plp_analysis_interval_sec,
            plp_hop_length=args.plp_hop_length,
            plp_peak_prominence=args.plp_peak_prominence,
        )
        self.analyzer.start_wall = time.perf_counter() - 1.0
        self.analyzer.calibrated_noise_rms = 1e-7
        self._pending_beats: list[float] = []
        self._next_beat = 0
        self._beat_period: float | None = None
        self._analyze_beats()

    @property
    def playback_seconds(self) -> float:
        return float(self.cursor / self.sample_rate)

    @property
    def done(self) -> bool:
        return self.cursor >= self.audio.size

    def _analyze_beats(self) -> None:
        onset = librosa.onset.onset_strength(
            y=self.audio,
            sr=self.sample_rate,
            hop_length=256,
        )
        _tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset,
            sr=self.sample_rate,
            hop_length=256,
        )
        self._pending_beats = [
            float(value)
            for value in librosa.frames_to_time(
                beat_frames,
                sr=self.sample_rate,
                hop_length=256,
            )
        ]
        if len(self._pending_beats) >= 2:
            self._beat_period = float(np.median(np.diff(self._pending_beats)))

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def advance(self, dt: float) -> None:
        self.target_cursor = min(
            self.target_cursor + max(float(dt), 0.0) * self.sample_rate,
            float(self.audio.size),
        )
        target = int(self.target_cursor)
        while target - self.cursor >= self.block_size:
            end = min(self.cursor + self.block_size, self.audio.size)
            block = self.audio[self.cursor:end]
            self.analyzer._callback(
                np.asarray(block[:, None], dtype=np.float32),
                block.size,
                {},
                None,
            )
            self.cursor = end
        if target >= self.audio.size and self.cursor < self.audio.size:
            end = self.audio.size
            block = self.audio[self.cursor:end]
            padded = np.pad(block, (0, self.block_size - block.size))
            self.analyzer._callback(
                np.asarray(padded[:, None], dtype=np.float32),
                block.size,
                {},
                None,
            )
            self.cursor = end

    def drain(self) -> list[base.MusicFrame]:
        frames = [frame for frame in self.analyzer.drain() if not frame.is_beat]
        playback = self.playback_seconds
        template = frames[-1] if frames else self.analyzer.latest_status_frame
        while (
            template is not None
            and self._next_beat < len(self._pending_beats)
            and self._pending_beats[self._next_beat] <= playback
        ):
            frames.append(
                replace(
                    template,
                    timestamp=time.perf_counter(),
                    beat_period=self._beat_period,
                    beat_confidence=1.0,
                    is_beat=True,
                    is_active=True,
                )
            )
            self._next_beat += 1
        frames.sort(key=lambda frame: frame.timestamp)
        return frames

    def recent_audio(self) -> np.ndarray | None:
        required = int(round(self.window_seconds * self.sample_rate))
        if self.cursor < required:
            return None
        return np.asarray(self.audio[self.cursor - required : self.cursor], dtype=np.float32)


class RetrievalWorker:
    def __init__(
        self,
        extractor: AudioFeatureExtractor,
        matcher: MusicMotionMatcher,
        top_tracks: int,
        top_motions: int,
    ) -> None:
        self.extractor = extractor
        self.matcher = matcher
        self.top_tracks = max(int(top_tracks), 1)
        self.top_motions = max(int(top_motions), 1)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="music-match")
        self.future: Future[MatchResult] | None = None

    def submit(self, audio: np.ndarray) -> bool:
        if self.future is not None and not self.future.done():
            return False
        samples = np.asarray(audio, dtype=np.float32).copy()
        self.future = self.executor.submit(self._run, samples)
        return True

    def _run(self, samples: np.ndarray) -> MatchResult:
        descriptor = self.extractor.describe(samples)
        return self.matcher.match(
            descriptor,
            top_k_tracks=self.top_tracks,
            top_k_motions=self.top_motions,
        )

    def poll(self) -> MatchResult | None:
        if self.future is None or not self.future.done():
            return None
        future = self.future
        self.future = None
        return future.result()

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


class MotionLoader:
    def __init__(self, catalog: MusicCatalog, args: argparse.Namespace) -> None:
        self.catalog = catalog
        self.args = args
        self.max_cached = max(int(args.match_top_motions), 1)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="motion-load")
        self.futures: dict[str, Future[base.AistppMotionSampler]] = {}
        self.cache: dict[str, base.AistppMotionSampler] = {}

    def submit(self, motion_id: str) -> None:
        self.preload((motion_id,))

    def preload(self, motion_ids: tuple[str, ...] | list[str]) -> None:
        desired = tuple(dict.fromkeys(motion_ids))[: self.max_cached]
        desired_set = set(desired)
        for motion_id, future in list(self.futures.items()):
            if motion_id not in desired_set and future.cancel():
                del self.futures[motion_id]
        for motion_id in list(self.cache):
            if motion_id not in desired_set:
                del self.cache[motion_id]
        for motion_id in desired:
            if motion_id in self.cache or motion_id in self.futures:
                continue
            profile = self.catalog.motions[motion_id]
            path = motion_path(self.catalog, profile)
            self.futures[motion_id] = self.executor.submit(
                base.AistppMotionSampler,
                path,
                self.args.aistpp_fps,
                self.args.pose_gain,
                self.args.accent_gain,
            )

    def take_ready(self, motion_id: str) -> base.AistppMotionSampler | None:
        if motion_id in self.cache:
            return self.cache[motion_id]
        future = self.futures.get(motion_id)
        if future is None or not future.done():
            return None
        sampler = future.result()
        del self.futures[motion_id]
        self.cache[motion_id] = sampler
        return sampler

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def motion_path(catalog: MusicCatalog, profile: MotionProfile) -> Path:
    root = Path(catalog.metadata["aistpp_root"])
    return root / Path(profile.motion_path)


def make_extractor(args: argparse.Namespace, catalog: MusicCatalog) -> AudioFeatureExtractor:
    metadata = catalog.metadata["extractor"]
    embedding_path = args.embedding_model
    tag_path = args.tag_model
    if embedding_path is None and metadata.get("embedding_model"):
        embedding_path = Path(metadata["embedding_model"]["path"])
    if tag_path is None and metadata.get("tag_model"):
        tag_path = Path(metadata["tag_model"]["path"])
    if embedding_path is not None and not embedding_path.is_absolute():
        embedding_path = ROOT / embedding_path
    if tag_path is not None and not tag_path.is_absolute():
        tag_path = ROOT / tag_path
    extractor = AudioFeatureExtractor(
        sample_rate=int(metadata["sample_rate"]),
        embedding_model=embedding_path,
        tag_model=tag_path,
    )
    expected = str(metadata["embedding_backend"])
    if extractor.backend_name != expected:
        raise RuntimeError(
            f"Catalog uses {expected}, but realtime extractor resolved to "
            f"{extractor.backend_name}. Supply the catalog's ONNX model."
        )
    return extractor


def make_controller(
    args: argparse.Namespace,
    profile: MotionProfile,
    previous: base.AdaptiveMotionController | None = None,
) -> base.AdaptiveMotionController:
    keypoints = profile.keypoint_phases
    if len(keypoints) < 2:
        keypoints = base.DEFAULT_FALLBACK_PHASES
    controller = base.AdaptiveMotionController(
        authored_cycle_duration=max(profile.duration_seconds, 1e-6),
        beats_per_cycle=max(len(keypoints), 1),
        keypoint_phases=keypoints,
        use_keypoints=True,
        smoothing_tau=args.smoothing_tau,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        amp_min=args.amp_min,
        amp_max=args.amp_max,
        accent_duration=args.accent_duration,
        tempo_timeout=args.tempo_timeout,
        beat_confidence_threshold=args.beat_confidence_threshold,
        beat_keypoint_interval_ratio=args.beat_keypoint_interval_ratio,
    )
    if previous is not None:
        strongest = (
            int(np.argmax(profile.keypoint_scores))
            if profile.keypoint_scores
            else 0
        )
        controller.phase = float(keypoints[strongest % len(keypoints)])
        controller.amplitude_scale = previous.amplitude_scale
        controller.target_amplitude_scale = previous.target_amplitude_scale
        controller.last_period = previous.last_period
        controller.last_brightness = previous.last_brightness
        controller.music_active = previous.music_active
        controller.last_update_wall = previous.last_update_wall
        if previous.last_period is not None:
            unclamped = 1.0 / max(
                previous.last_period * controller.beats_per_cycle,
                1e-6,
            )
            controller.target_phase_rate = float(
                np.clip(
                    unclamped,
                    controller.authored_phase_rate * controller.speed_min,
                    controller.authored_phase_rate * controller.speed_max,
                )
            )
            controller.phase_rate = controller.target_phase_rate
    return controller


def sample_actuator_pose(
    sampler: base.AistppMotionSampler,
    controller: base.AdaptiveMotionController,
    now: float,
    features: base.FeatureState,
    modulator: MusicPoseModulator | None,
    adapter: Any,
) -> dict[str, float]:
    phase, amplitude, accent, _brightness = controller.update(now)
    pose = sampler.sample(phase, amplitude, accent, features)
    if modulator is not None:
        pose = modulator.modulate(
            pose,
            features,
            phase=phase,
            amplitude=amplitude,
            accent=accent,
        )
    if adapter is not None:
        pose = adapter.adapt_pose(pose, features)
    return pose


def blend_poses(
    first: dict[str, float],
    second: dict[str, float],
    amount: float,
) -> dict[str, float]:
    t = float(np.clip(amount, 0.0, 1.0))
    smooth = t * t * (3.0 - 2.0 * t)
    names = set(first) | set(second)
    return {
        name: float((1.0 - smooth) * first.get(name, 0.0) + smooth * second.get(name, 0.0))
        for name in names
    }


def initial_motion(
    args: argparse.Namespace,
    catalog: MusicCatalog,
) -> tuple[str, MotionProfile, base.AistppMotionSampler]:
    requested = Path(args.aistpp_motion).stem if args.aistpp_motion is not None else ""
    if requested not in catalog.motions:
        requested = next(iter(catalog.motions))
    profile = catalog.motions[requested]
    sampler = base.AistppMotionSampler(
        motion_path(catalog, profile),
        args.aistpp_fps,
        args.pose_gain,
        args.accent_gain,
    )
    return requested, profile, sampler


def print_match_status(result: MatchResult) -> None:
    tracks = ", ".join(
        f"{track.music_id}:{track.score:.2f}" for track in result.tracks[:3]
    )
    motion = result.motions[0] if result.motions else None
    motion_text = (
        f"{motion.motion_id}:{motion.final_score:.2f}"
        if motion is not None
        else "--"
    )
    tags = ", ".join(label for label, _ in result.query_tags[:3])
    tag_text = f" | tags=[{tags}]" if tags else ""
    print(
        f"match | bpm={result.query_bpm:.1f} | tracks=[{tracks}] | "
        f"motion={motion_text}{tag_text}"
    )


def main() -> int:
    args = parse_args()
    if args.motion_source != "aistpp":
        raise ValueError("The matcher currently selects AIST++ motions; use --motion-source aistpp.")
    catalog_path = args.catalog if args.catalog.is_absolute() else ROOT / args.catalog
    catalog = MusicCatalog.load(catalog_path)
    extractor = make_extractor(args, catalog)
    matcher = MusicMotionMatcher(
        catalog,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
    )
    current_id, current_profile, current_sampler = initial_motion(args, catalog)
    current_controller = make_controller(args, current_profile)

    player = base.MujocoHumanoidPlayer(
        args.model,
        realtime=args.realtime,
        headless=args.headless,
    )
    adapter = base.make_pose_adapter(args, player, current_sampler)
    modulator = (
        None
        if args.disable_music_modulation
        else MusicPoseModulator(args.music_modulation_strength)
    )
    if args.preview_trajectory:
        base.run_trajectory_preview(args, current_sampler, player, adapter, modulator)
        return 0

    if args.audio_input is not None:
        audio_path = args.audio_input if args.audio_input.is_absolute() else ROOT / args.audio_input
        source: MicrophoneSource | SilentSource | WavSource = WavSource(
            audio_path,
            args,
            args.match_window_seconds,
        )
        print(f"Using deterministic audio input: {audio_path}")
    else:
        if args.no_mic:
            source = SilentSource()
            print("No microphone input: holding the current motion.")
        else:
            source = MicrophoneSource(
                base.make_analyzer(args),
                args.match_window_seconds,
            )
            print(
                f"Calibrating microphone noise from the first "
                f"{args.startup_calibration_sec:.2f}s of live input."
            )

    retrieval = RetrievalWorker(
        extractor,
        matcher,
        args.match_top_tracks,
        args.match_top_motions,
    )
    loader = MotionLoader(catalog, args)
    stabilizer = CandidateStabilizer(
        args.switch_required_wins,
        args.switch_score_margin,
    )
    features = base.FeatureState()
    pending_motion_id: str | None = None
    transition_sampler: base.AistppMotionSampler | None = None
    transition_controller: base.AdaptiveMotionController | None = None
    transition_start: float | None = None
    transition_duration = 0.5
    accepted_beat_count = 0
    last_match_clock = -math.inf
    last_feature_update = time.perf_counter()
    last_status = time.perf_counter()
    last_frame: base.MusicFrame | None = None
    recent_beat_frame: base.MusicFrame | None = None

    player.start()
    source.start()
    wall_start = time.perf_counter()
    print(f"Initial motion: {current_id}")
    try:
        while player.is_running() and not source.done:
            now = time.perf_counter()
            source.advance(player.dt)
            elapsed = (
                source.playback_seconds
                if isinstance(source, WavSource)
                else now - wall_start
            )
            if args.max_seconds is not None and elapsed >= args.max_seconds:
                break

            switch_boundary = False
            for frame in source.drain():
                last_frame = frame
                current_controller.observe(frame)
                if transition_controller is not None:
                    transition_controller.observe(frame)
                dt = max(now - last_feature_update, player.dt)
                alpha = 1.0 - math.exp(
                    -dt / max(args.feature_smoothing_tau, 1e-6)
                )
                features.update(frame, alpha)
                last_feature_update = now
                if (
                    frame.is_beat
                    and frame.beat_confidence >= args.beat_confidence_threshold
                ):
                    recent_beat_frame = frame
                    accepted_beat_count += 1
                    switch_boundary = (
                        accepted_beat_count % max(args.switch_beats_per_bar, 1) == 0
                    )

            match_clock = source.playback_seconds
            if match_clock - last_match_clock >= args.match_interval_seconds:
                audio = source.recent_audio()
                if audio is not None and retrieval.submit(audio):
                    last_match_clock = match_clock

            try:
                result = retrieval.poll()
            except Exception as exc:
                print(f"Warning: realtime retrieval failed: {type(exc).__name__}: {exc}")
                result = None
            if result is not None:
                print_match_status(result)
                loader.preload([motion.motion_id for motion in result.motions])
                stable = stabilizer.observe(result, current_id)
                if stable is not None and stable in catalog.motions:
                    pending_motion_id = stable
                    print(f"Pending motion after stable retrieval: {stable}")

            if (
                pending_motion_id is not None
                and transition_sampler is None
                and switch_boundary
            ):
                ready = loader.take_ready(pending_motion_id)
                if ready is not None:
                    transition_sampler = ready
                    transition_profile = catalog.motions[pending_motion_id]
                    transition_controller = make_controller(
                        args,
                        transition_profile,
                        previous=current_controller,
                    )
                    transition_start = now
                    transition_duration = (
                        current_controller.last_period
                        if current_controller.last_period is not None
                        else 0.5
                    )
                    print(
                        f"Switching on bar boundary: {current_id} -> {pending_motion_id} "
                        f"({transition_duration:.3f}s blend)"
                    )

            current_pose = sample_actuator_pose(
                current_sampler,
                current_controller,
                now,
                features,
                modulator,
                adapter,
            )
            if (
                transition_sampler is not None
                and transition_controller is not None
                and transition_start is not None
            ):
                next_pose = sample_actuator_pose(
                    transition_sampler,
                    transition_controller,
                    now,
                    features,
                    modulator,
                    adapter,
                )
                blend = (now - transition_start) / max(transition_duration, 1e-6)
                pose = blend_poses(current_pose, next_pose, blend)
                if blend >= 1.0:
                    current_id = pending_motion_id or current_id
                    current_profile = catalog.motions[current_id]
                    current_sampler = transition_sampler
                    current_controller = transition_controller
                    pending_motion_id = None
                    transition_sampler = None
                    transition_controller = None
                    transition_start = None
                    stabilizer.reset()
                    print(f"Motion switch complete: {current_id}")
            else:
                pose = current_pose

            player.set_pose(pose)
            player.step()

            if now - last_status >= args.status_interval:
                base.print_status(
                    current_controller,
                    features,
                    last_frame,
                    recent_beat_frame,
                )
                recent_beat_frame = None
                last_status = now
    finally:
        source.stop()
        retrieval.close()
        loader.close()
        player.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
