"""Capture final player poses and exact playback metadata without changing motion."""
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'realtime/humanoid_robot/src'))
import realtime_music_humanoid_matcher_v2 as matcher

records = []
metadata = {}
original_frame = matcher.base.MujocoHumanoidPlayer.set_frame
original_sample = matcher.AuthoredPlayback.sample


def sample(self, *args, **kwargs):
    result = original_sample(self, *args, **kwargs)
    metadata.update(motion_id=self.current_id, authored_seconds=self.clock.seconds,
                    duration=self.clock.duration, status=self.status, event=self.event,
                    weight=self.clock.weight)
    return result


def capture(self, frame):
    result = original_frame(self, frame)
    if metadata:
        names = tuple(self.actuator_joint_qpos_ids)
        records.append((time.perf_counter(), self.data.qpos.copy(),
                        [frame.joint_positions.get(n, 0.) for n in names], names,
                        [self.actuator_joint_qpos_ids[n] for n in names], dict(metadata)))
    return result


if __name__ == '__main__':
    matcher.AuthoredPlayback.sample = sample
    matcher.base.MujocoHumanoidPlayer.set_frame = capture
    try:
        matcher.main()
    finally:
        if records:
            prefix = Path(sys.argv[sys.argv.index('--trace-csv') + 1]).with_suffix('')
            np.savez_compressed(str(prefix) + '_display.npz',
                                perf_time=[r[0] for r in records], qpos=[r[1] for r in records],
                                raw_joints=[r[2] for r in records], joint_names=records[0][3],
                                qpos_ids=records[0][4])
            with Path(str(prefix) + '_frames.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['capture_index'] + list(metadata))
                writer.writeheader()
                for i, record in enumerate(records):
                    writer.writerow(dict(capture_index=i, **record[-1]))

