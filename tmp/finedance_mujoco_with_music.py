from pathlib import Path
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'realtime/humanoid_robot/src'))
import realtime_music_humanoid_dancer as dancer


def preview_with_audio(args, sampler, player, pose_adapter, modulator):
    audio_path = ROOT / 'realtime/humanoid_robot/data/finedance_aistpp/audio/finedance_037_0000000_0000596.wav'
    audio, sample_rate = sf.read(audio_path, dtype='float32', always_2d=True)
    assert abs(len(audio) / sample_rate - sampler.duration) < 1 / sample_rate
    clock = {'cursor': 0, 'start': None, 'underflows': 0}

    def callback(outdata, frames, timing, status):
        if status:
            clock['underflows'] += 1
        if clock['start'] is None:
            clock['start'] = timing.outputBufferDacTime
        indices = (np.arange(frames) + clock['cursor']) % len(audio)
        outdata[:] = audio[indices]
        clock['cursor'] += frames

    features = dancer.FeatureState(rms_norm=0.85, brightness=0.35, low_energy=0.55,
                                  mid_energy=0.45, high_energy=0.25, rhythm_density=0.35, is_active=True)
    player.start()
    initial = player.root_frame()
    root_motion = dancer.RootMotionContinuity(initial.root_position, initial.root_quaternion_wxyz,
                                             mode=args.root_motion, grounded_z=True)
    scheduler = dancer.RealtimeLoopScheduler(args.control_rate_hz, True)
    try:
        with sd.OutputStream(samplerate=sample_rate, channels=audio.shape[1], dtype='float32', callback=callback) as stream:
            print(f'Playing synchronized music: {audio_path.name}, {sampler.duration:.3f}s, looping at 1x.', flush=True)
            last_report = -2.0
            while player.is_running():
                work_started = time.perf_counter()
                elapsed = max(0.0, stream.time - clock['start']) if clock['start'] is not None else 0.0
                phase = (elapsed / sampler.duration) % 1.0
                frame = dancer.sample_robot_motion_frame(sampler, phase=phase, amplitude=1.0,
                            accent=0.0, features=features, pose_adapter=pose_adapter, modulator=None)
                frame = root_motion.apply(frame, phase=phase, source_id=dancer.sampler_source_id(sampler))
                player.set_frame(frame)
                player.step()
                if elapsed - last_report >= 2:
                    print(f'Audio-clock playback: {elapsed:.2f}s; phase={phase:.3f}; audio status events={clock["underflows"]}', flush=True)
                    last_report = elapsed
                scheduler.wait(work_started)
    finally:
        player.stop()


dancer.run_trajectory_preview = preview_with_audio
if __name__ == '__main__':
    dancer.main()
