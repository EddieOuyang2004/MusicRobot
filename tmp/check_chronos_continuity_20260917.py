"""Capture actual GUI poses; leave matcher behavior and source unchanged."""
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'realtime/humanoid_robot/src'))
import realtime_music_humanoid_matcher_v2 as matcher

records = []
original = matcher.base.MujocoHumanoidPlayer.set_frame
original_write = csv.DictWriter.writerow


def capture(self, frame):
    result = original(self, frame)
    names = tuple(self.actuator_joint_qpos_ids)
    records.append((time.perf_counter(), self.data.qpos.copy(),
                    [frame.joint_positions.get(n, 0.) for n in names], names,
                    [self.actuator_joint_qpos_ids[n] for n in names], {}))
    return result


def capture_trace(self, row):
    if records and 'current_motion_id' in row:
        records[-1][-1].update(row)
    return original_write(self, row)


if __name__ == '__main__':
    matcher.base.MujocoHumanoidPlayer.set_frame = capture
    csv.DictWriter.writerow = capture_trace
    try:
        matcher.main()
    finally:
        if records:
            prefix = Path(sys.argv[sys.argv.index('--trace-csv') + 1]).with_suffix('')
            np.savez_compressed(str(prefix) + '_display.npz',
                                perf_time=[r[0] for r in records],
                                qpos=[r[1] for r in records], raw_joints=[r[2] for r in records],
                                joint_names=records[0][3], qpos_ids=records[0][4])
            with Path(str(prefix) + '_frames.csv').open('w', newline='', encoding='utf-8') as f:
                keys = sorted(set().union(*(r[-1].keys() for r in records)))
                writer = csv.DictWriter(f, fieldnames=['frame_index'] + keys)
                writer.writeheader()
                for i, record in enumerate(records):
                    original_write(writer, dict(frame_index=i, **record[-1]))
