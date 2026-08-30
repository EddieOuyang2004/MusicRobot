from __future__ import annotations

import argparse
import csv
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
    MatchResult,
    MotionSelection,
    MotionSelectionPolicy,
    MotionProfile,
    MusicCatalog,
    MusicMotionMatcher,
)
from music_pose_modulator import MusicPoseModulator
from robot_motion import (
    RobotMotionFrame,
    RootMotionContinuity,
    align_motion_frame_root,
    blend_motion_frames,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = (
    ROOT / "realtime" / "humanoid_robot" / "data" / "music_catalog" / "catalog.json"
)


def parse_args() -> argparse.Namespace:
    """Parse matcher-specific flags, then delegate all legacy flags to the frozen entrypoint."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--embedding-model", type=Path, default=None)
    parser.add_argument("--tag-model", type=Path, default=None)
    parser.add_argument("--match-window-seconds", type=float, default=6.0)
    parser.add_argument("--match-interval-seconds", type=float, default=1.0)
    parser.add_argument("--match-top-tracks", type=int, default=5)
    parser.add_argument("--match-top-motions", type=int, default=10)
    parser.add_argument("--switch-required-wins", type=int, default=3)
    parser.add_argument("--switch-score-margin", type=float, default=0.08)
    parser.add_argument("--switch-beats-per-bar", type=int, default=4)
    parser.add_argument("--switch-max-hold-bars", type=int, default=4)
    parser.add_argument("--switch-diversity-top-k", type=int, default=5)
    parser.add_argument("--switch-diversity-score-drop", type=float, default=0.05)
    parser.add_argument(
        "--switch-diversity-music-score-drop",
        type=float,
        default=0.08,
    )
    parser.add_argument("--switch-recent-history", type=int, default=3)
    parser.add_argument(
        "--trace-csv",
        type=Path,
        default=None,
        help="Optional CSV timeline containing matcher rankings, motion switches, and phase progress.",
    )
    parser.add_argument(
        "--trace-interval-seconds",
        type=float,
        default=0.05,
        help="Audio-time interval between --trace-csv samples (default: 0.05).",
    )
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


class MatcherFileMicrophoneSource(base.FileMicrophoneSource):
    """Virtual file microphone with matcher-specific beat and window handling."""

    def __init__(
        self,
        path: Path,
        args: argparse.Namespace,
        window_seconds: float,
    ) -> None:
        self.window_seconds = float(window_seconds)
        analyzer = base.make_analyzer(args)
        analyzer.plp_history_sec = max(analyzer.plp_history_sec, self.window_seconds)
        super().__init__(
            path,
            analyzer,
            startup_delay_sec=args.audio_input_delay_sec,
            throttle=not args.realtime,
            play_audio=getattr(args, "play_audio", False),
        )
        self._pending_beats: list[float] = []
        self._pending_beat_contrasts: list[float] = []
        self._next_beat = 0
        self._beat_period: float | None = None
        self._analyze_beats()

    def _analyze_beats(self) -> None:
        hop_length = self.analyzer.plp_hop_length
        onset = librosa.onset.onset_strength(
            y=self.audio,
            sr=self.sample_rate,
            hop_length=hop_length,
        )
        _tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset,
            sr=self.sample_rate,
            hop_length=hop_length,
        )
        self._pending_beats = [
            float(value)
            for value in librosa.frames_to_time(
                beat_frames,
                sr=self.sample_rate,
                hop_length=hop_length,
            )
        ]
        self._pending_beat_contrasts = base.compute_beat_contrasts(
            self.audio,
            onset,
            np.asarray(beat_frames, dtype=int),
            self.sample_rate,
            hop_length,
        ).tolist()
        if len(self._pending_beats) >= 2:
            self._beat_period = float(np.median(np.diff(self._pending_beats)))

    def drain(self) -> list[base.MusicFrame]:
        frames = [frame for frame in super().drain() if not frame.is_beat]
        playback = self.playback_seconds
        now = time.perf_counter()
        template = frames[-1] if frames else self.analyzer.latest_status_frame
        while (
            template is not None
            and self._next_beat < len(self._pending_beats)
            and self._pending_beats[self._next_beat] <= playback
        ):
            frames.append(
                replace(
                    template,
                    timestamp=now - max(playback - self._pending_beats[self._next_beat], 0.0),
                    beat_period=self._beat_period,
                    beat_confidence=1.0,
                    beat_contrast=self._pending_beat_contrasts[self._next_beat],
                    is_beat=True,
                    is_active=True,
                )
            )
            self._next_beat += 1
        frames.sort(key=lambda frame: frame.timestamp)
        return frames

    def recent_audio(self) -> np.ndarray | None:
        return super().recent_audio(self.window_seconds)


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


MotionSampler = base.AistppMotionSampler | base.GmrUnitreeG1MotionSampler


class MotionLoader:
    def __init__(self, catalog: MusicCatalog, args: argparse.Namespace, adapter: Any) -> None:
        self.catalog = catalog
        self.args = args
        self.adapter = adapter
        self.max_cached = max(int(args.match_top_motions), 1)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="motion-load")
        self.prepare_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="motion-prepare",
        )
        self.grounding_player = base.MujocoHumanoidPlayer(
            args.model,
            realtime=False,
            headless=True,
        )
        self.futures: dict[str, Future[MotionSampler]] = {}
        self.prepare_futures: dict[str, Future[MotionSampler]] = {}
        self.cache: dict[str, MotionSampler] = {}

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
            self.futures[motion_id] = self.executor.submit(
                load_motion_sampler,
                self.args,
                self.catalog,
                profile,
            )

    def take_ready(self, motion_id: str) -> MotionSampler | None:
        if motion_id in self.cache:
            return self.cache[motion_id]
        future = self.prepare_futures.get(motion_id)
        if future is None or not future.done():
            return None
        sampler = future.result()
        del self.prepare_futures[motion_id]
        self.futures.pop(motion_id, None)
        self.cache[motion_id] = sampler
        return sampler

    def prepare(self, motion_id: str) -> None:
        if motion_id in self.cache or motion_id in self.prepare_futures:
            return
        load_future = self.futures.get(motion_id)
        if load_future is None:
            sampler = self.cache.get(motion_id)
            load_future = (
                self.executor.submit(lambda value=sampler: value)
                if sampler is not None
                else self.executor.submit(
                    load_motion_sampler,
                    self.args,
                    self.catalog,
                    self.catalog.motions[motion_id],
                )
            )
            self.futures[motion_id] = load_future
        self.prepare_futures[motion_id] = self.prepare_executor.submit(
            self._prepare_sampler,
            motion_id,
            load_future,
        )

    def _prepare_sampler(
        self,
        motion_id: str,
        load_future: Future[MotionSampler],
    ) -> MotionSampler:
        sampler = load_future.result()
        self.grounding_player.ground_sampler(sampler, self.adapter)
        return sampler

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.prepare_executor.shutdown(wait=False, cancel_futures=True)


def motion_path(catalog: MusicCatalog, profile: MotionProfile) -> Path:
    root = Path(catalog.metadata["aistpp_root"])
    return root / Path(profile.motion_path)


def gmr_motion_path(args: argparse.Namespace, profile: MotionProfile) -> Path:
    root = args.gmr_motion_root if args.gmr_motion_root.is_absolute() else ROOT / args.gmr_motion_root
    return root / f"{profile.motion_id}.pkl"


def load_motion_sampler(
    args: argparse.Namespace,
    catalog: MusicCatalog,
    profile: MotionProfile,
) -> MotionSampler:
    gmr_path = gmr_motion_path(args, profile)
    if args.retarget_policy != "direct" and gmr_path.exists():
        return base.GmrUnitreeG1MotionSampler(
            gmr_path,
            args.gmr_fps,
            args.pose_gain,
            args.accent_gain,
            False,
        )
    if args.retarget_policy == "require-gmr":
        source_path = motion_path(catalog, profile)
        raise FileNotFoundError(
            f"Required GMR artifact not found: {gmr_path}\n"
            f"Generate it with:\n{base.gmr_generation_hint(source_path)}"
        )
    return base.AistppMotionSampler(
        motion_path(catalog, profile),
        args.aistpp_fps,
        args.pose_gain,
        args.accent_gain,
    )


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
        onnx_intra_op_threads=1,
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
        beat_selection_mode=args.beat_selection_mode,
        beat_contrast_weight=args.beat_contrast_weight,
        max_speed_change_per_sec=getattr(args, "max_speed_change_per_sec", 2.0),
    )
    if previous is not None:
        strongest = (
            int(np.argmax(profile.keypoint_scores))
            if profile.keypoint_scores
            else 0
        )
        controller.phase = float(keypoints[strongest % len(keypoints)])
        controller.beat_index = strongest + 1
        controller.amplitude_scale = previous.amplitude_scale
        controller.target_amplitude_scale = previous.target_amplitude_scale
        controller.last_period = previous.last_period
        controller.last_beat_wall = previous.last_beat_wall
        controller.last_candidate_beat_wall = previous.last_candidate_beat_wall
        controller.last_accepted_interval = previous.last_accepted_interval
        controller.candidate_scores.extend(previous.candidate_scores)
        controller.last_brightness = previous.last_brightness
        controller.music_active = previous.music_active
        controller.last_update_wall = previous.last_update_wall
        inherited_rate = controller.authored_phase_rate * previous.speed_multiplier
        controller.target_phase_rate = float(
            np.clip(
                inherited_rate,
                controller.authored_phase_rate * controller.speed_min,
                controller.authored_phase_rate * controller.speed_max,
            )
        )
        controller.phase_rate = controller.target_phase_rate
    return controller


def sample_actuator_pose(
    sampler: MotionSampler,
    controller: base.AdaptiveMotionController,
    now: float,
    features: base.FeatureState,
    modulator: MusicPoseModulator | None,
    adapter: Any,
) -> tuple[RobotMotionFrame, float]:
    phase, amplitude, accent, _brightness = controller.update(now)
    frame = base.sample_robot_motion_frame(
        sampler,
        phase=phase,
        amplitude=amplitude,
        accent=accent,
        features=features,
        pose_adapter=adapter,
        modulator=modulator,
    )
    return frame, phase


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
) -> tuple[str, MotionProfile, MotionSampler]:
    requested = Path(args.aistpp_motion).stem if args.aistpp_motion is not None else ""
    if requested not in catalog.motions:
        requested = next(iter(catalog.motions))
    profile = catalog.motions[requested]
    sampler = load_motion_sampler(args, catalog, profile)
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
    print(
        f"Initial retarget source: "
        f"{'GMR artifact' if isinstance(current_sampler, base.GmrUnitreeG1MotionSampler) else 'direct AIST++ fallback'}"
    )

    player = base.MujocoHumanoidPlayer(
        args.model,
        realtime=args.realtime,
        headless=args.headless,
        viewer_rate_hz=args.viewer_rate_hz,
    )
    adapter = base.make_pose_adapter(args, player, current_sampler)
    player.ground_sampler(current_sampler, adapter)
    modulation_off = args.disable_music_modulation or args.pose_modulation_mode == "off"
    modulator = (
        None
        if modulation_off
        else MusicPoseModulator(args.music_modulation_strength, mode=args.pose_modulation_mode)
    )
    if args.preview_trajectory:
        base.run_trajectory_preview(args, current_sampler, player, adapter, modulator)
        return 0

    if args.audio_input is not None:
        audio_path = args.audio_input if args.audio_input.is_absolute() else ROOT / args.audio_input
        source: MicrophoneSource | SilentSource | MatcherFileMicrophoneSource = MatcherFileMicrophoneSource(
            audio_path,
            args,
            args.match_window_seconds,
        )
        print(
            f"Using realtime virtual microphone: {audio_path} "
            f"(audio starts after {source.startup_delay_sec:.2f}s denoiser reset time; "
            f"speaker output={'on' if source.play_audio else 'off'})."
        )
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
    loader = MotionLoader(catalog, args, adapter)
    selection_policy = MotionSelectionPolicy(
        current_motion_id=current_id,
        required_wins=args.switch_required_wins,
        score_margin=args.switch_score_margin,
        max_hold_bars=args.switch_max_hold_bars,
        diversity_top_k=args.switch_diversity_top_k,
        diversity_score_drop=args.switch_diversity_score_drop,
        diversity_music_score_drop=args.switch_diversity_music_score_drop,
        recent_history=args.switch_recent_history,
    )
    features = base.FeatureState()
    transition_sampler: MotionSampler | None = None
    transition_controller: base.AdaptiveMotionController | None = None
    transition_selection: MotionSelection | None = None
    transition_start: float | None = None
    transition_duration = 0.5
    transition_root_source_reference: RobotMotionFrame | None = None
    transition_root_target_reference: RobotMotionFrame | None = None
    accepted_beat_count = 0
    last_match_clock = -math.inf
    last_feature_update = time.perf_counter()
    last_status = time.perf_counter()
    last_frame: base.MusicFrame | None = None
    recent_detected_beat_frame: base.MusicFrame | None = None
    recent_accepted_beat_frame: base.MusicFrame | None = None
    last_match_result: MatchResult | None = None
    trace_handle = None
    trace_writer: csv.DictWriter | None = None
    trace_next_audio_time = 0.0
    trace_interval = max(float(args.trace_interval_seconds), 0.001)
    if args.trace_csv is not None:
        trace_path = args.trace_csv if args.trace_csv.is_absolute() else ROOT / args.trace_csv
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_handle = trace_path.open("w", encoding="utf-8", newline="")
        trace_writer = csv.DictWriter(
            trace_handle,
            fieldnames=(
                "audio_time_seconds",
                "event",
                "current_motion_id",
                "pending_motion_id",
                "transition_motion_id",
                "current_phase",
                "transition_phase",
                "transition_blend",
                "speed_multiplier",
                "top_track_id",
                "top_track_score",
                "top_motion_id",
                "top_motion_score",
                "query_bpm",
            ),
        )
        trace_writer.writeheader()

    source_analyzer = getattr(source, "analyzer", None)
    if source_analyzer is not None:
        # Windows process spawning must happen before the viewer owns an
        # OpenGL context; source.start() remains responsible for the stream.
        source_analyzer.prepare_background_analysis()
    live_microphone_started = isinstance(source, MicrophoneSource)
    if live_microphone_started:
        # Some Windows PortAudio backends can block when opened after an
        # OpenGL viewer.  Open hardware first, then restart calibration once
        # the viewer is ready.
        source.start()
    player.start()
    if live_microphone_started:
        source_analyzer.reset()
    initial_root = player.root_frame()
    root_motion = RootMotionContinuity(
        initial_root.root_position if initial_root.root_position is not None else np.zeros(3),
        initial_root.root_quaternion_wxyz
        if initial_root.root_quaternion_wxyz is not None
        else np.asarray([1.0, 0.0, 0.0, 0.0]),
        mode=args.root_motion,
        grounded_z=True,
    )
    if not live_microphone_started:
        source.start()
    wall_start = time.perf_counter()
    scheduler = base.RealtimeLoopScheduler(args.control_rate_hz, args.realtime)
    print(f"Initial motion: {current_id}")
    try:
        while player.is_running() and not source.done:
            trace_event = ""
            trace_transition_blend = 0.0
            work_started = time.perf_counter()
            now = work_started
            source.advance(scheduler.period)
            elapsed = (
                source.playback_seconds
                if isinstance(source, MatcherFileMicrophoneSource)
                else now - wall_start
            )
            if args.max_seconds is not None and elapsed >= args.max_seconds:
                break

            switch_boundary = False
            ready_selection: MotionSelection | None = None
            for frame in source.drain():
                last_frame = frame
                accepted = current_controller.observe(frame)
                if transition_controller is not None:
                    transition_controller.observe(frame)
                dt = max(now - last_feature_update, scheduler.period)
                alpha = 1.0 - math.exp(
                    -dt / max(args.feature_smoothing_tau, 1e-6)
                )
                features.update(frame, alpha)
                last_feature_update = now
                if frame.is_beat:
                    recent_detected_beat_frame = frame
                if accepted:
                    recent_accepted_beat_frame = frame
                    accepted_beat_count += 1
                    is_switch_boundary = (
                        accepted_beat_count % max(args.switch_beats_per_bar, 1) == 0
                    )
                    if is_switch_boundary:
                        switch_boundary = True
                        ready_selection = selection_policy.on_bar_boundary()

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
                last_match_result = result
                trace_event = "match"
                print_match_status(result)
                previous_pending = selection_policy.pending
                pending = selection_policy.observe(result)
                loader.preload(selection_policy.preload_motion_ids(result))
                if pending is not None:
                    loader.prepare(pending.motion_id)
                if previous_pending is None and pending is not None:
                    pool = ", ".join(
                        f"{item.motion_id}:{item.final_score:.3f}/{item.music_score:.3f}"
                        for item in selection_policy.diversity_pool()
                    )
                    print(
                        f"Pending motion ({pending.reason}): {pending.motion_id} "
                        f"score={pending.final_score:.3f} "
                        f"music={pending.music_score:.3f} "
                        f"held={selection_policy.bars_held} bar(s) "
                        f"pool=[{pool}]"
                    )
                if switch_boundary and ready_selection is None:
                    ready_selection = selection_policy.ready_selection()

            if (
                ready_selection is not None
                and transition_sampler is None
            ):
                loader.prepare(ready_selection.motion_id)
                ready = loader.take_ready(ready_selection.motion_id)
                if ready is not None:
                    transition_sampler = ready
                    transition_selection = ready_selection
                    transition_profile = catalog.motions[ready_selection.motion_id]
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
                        f"Switching on bar boundary ({ready_selection.reason}, "
                        f"held={selection_policy.bars_held} bars): "
                        f"{current_id} -> {ready_selection.motion_id} "
                        f"({transition_duration:.3f}s blend, "
                        f"score={ready_selection.final_score:.3f}, "
                        f"music={ready_selection.music_score:.3f})"
                    )
                    trace_event = "switch_start"

            current_frame, current_phase = sample_actuator_pose(
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
                next_frame, transition_phase = sample_actuator_pose(
                    transition_sampler,
                    transition_controller,
                    now,
                    features,
                    modulator,
                    adapter,
                )
                if transition_root_source_reference is None:
                    transition_root_source_reference = next_frame
                    transition_root_target_reference = current_frame
                assert transition_root_target_reference is not None
                next_frame = align_motion_frame_root(
                    next_frame,
                    source_reference=transition_root_source_reference,
                    target_reference=transition_root_target_reference,
                )
                blend = (now - transition_start) / max(transition_duration, 1e-6)
                trace_transition_blend = float(np.clip(blend, 0.0, 1.0))
                motion_frame = blend_motion_frames(current_frame, next_frame, blend)
                root_source_id = f"{current_id}->{transition_selection.motion_id if transition_selection else 'pending'}"
                if blend >= 1.0:
                    if transition_selection is None:
                        raise RuntimeError("Motion transition lost its selection state.")
                    current_id = transition_selection.motion_id
                    current_profile = catalog.motions[current_id]
                    current_sampler = transition_sampler
                    current_controller = transition_controller
                    selection_policy.complete_switch(current_id)
                    transition_sampler = None
                    transition_controller = None
                    transition_selection = None
                    transition_start = None
                    transition_root_source_reference = None
                    transition_root_target_reference = None
                    print(
                        f"Motion switch complete: {current_id} "
                        f"({'GMR artifact' if isinstance(current_sampler, base.GmrUnitreeG1MotionSampler) else 'direct AIST++ fallback'})"
                    )
                    trace_event = "switch_complete"
            else:
                motion_frame = current_frame
                transition_phase = None
                root_source_id = current_id

            motion_frame = root_motion.apply(
                motion_frame,
                phase=current_phase if transition_phase is None else transition_phase,
                source_id=root_source_id,
            )
            motion_frame = player.apply_collision_policy(
                motion_frame,
                args.runtime_collision_check,
                runtime_modified=(
                    transition_sampler is not None
                    or (modulator is not None and features.is_active)
                ),
            )
            player.set_frame(motion_frame)
            player.step(scheduler.period)

            if trace_writer is not None and (
                trace_event or elapsed + 1e-9 >= trace_next_audio_time
            ):
                top_track = (
                    last_match_result.tracks[0]
                    if last_match_result is not None and last_match_result.tracks
                    else None
                )
                top_motion = (
                    last_match_result.motions[0]
                    if last_match_result is not None and last_match_result.motions
                    else None
                )
                pending_selection = selection_policy.pending
                trace_writer.writerow(
                    {
                        "audio_time_seconds": f"{elapsed:.6f}",
                        "event": trace_event,
                        "current_motion_id": current_id,
                        "pending_motion_id": (
                            pending_selection.motion_id if pending_selection is not None else ""
                        ),
                        "transition_motion_id": (
                            transition_selection.motion_id
                            if transition_selection is not None
                            else ""
                        ),
                        "current_phase": f"{current_controller.phase:.9f}",
                        "transition_phase": (
                            f"{transition_controller.phase:.9f}"
                            if transition_controller is not None
                            else ""
                        ),
                        "transition_blend": f"{trace_transition_blend:.6f}",
                        "speed_multiplier": f"{current_controller.speed_multiplier:.6f}",
                        "top_track_id": top_track.music_id if top_track is not None else "",
                        "top_track_score": (
                            f"{top_track.score:.6f}" if top_track is not None else ""
                        ),
                        "top_motion_id": top_motion.motion_id if top_motion is not None else "",
                        "top_motion_score": (
                            f"{top_motion.final_score:.6f}" if top_motion is not None else ""
                        ),
                        "query_bpm": (
                            f"{last_match_result.query_bpm:.6f}"
                            if last_match_result is not None
                            else ""
                        ),
                    }
                )
                if elapsed + 1e-9 >= trace_next_audio_time:
                    trace_next_audio_time = (
                        math.floor(elapsed / trace_interval + 1.0) * trace_interval
                    )

            if now - last_status >= args.status_interval:
                base.print_status(
                    current_controller,
                    features,
                    last_frame,
                    recent_detected_beat_frame,
                    recent_accepted_beat_frame,
                )
                recent_detected_beat_frame = None
                recent_accepted_beat_frame = None
                last_status = now
            scheduler.wait(work_started)
    finally:
        source.stop()
        retrieval.close()
        loader.close()
        player.stop()
        if trace_handle is not None:
            trace_handle.close()
        base.write_timing_report(
            args.timing_report,
            scheduler,
            getattr(source, "analyzer", None),
            player,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
