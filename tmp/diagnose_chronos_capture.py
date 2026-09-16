"""Diagnostic wrapper: capture final displayed poses without changing playback."""
import sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'realtime/humanoid_robot/src'))
import realtime_music_humanoid_matcher_v2 as matcher
records=[]
original=matcher.base.MujocoHumanoidPlayer.set_frame
def capture(self,frame):
    result=original(self,frame)
    names=tuple(self.actuator_joint_qpos_ids)
    records.append((time.perf_counter(),self.data.qpos.copy(),[frame.joint_positions.get(n,0.) for n in names],names))
    return result
if __name__=='__main__':
    matcher.base.MujocoHumanoidPlayer.set_frame=capture
    try:
        matcher.main()
    finally:
        if records:
            np.savez_compressed('tmp/chronos_smoothness_display.npz',perf_time=[r[0] for r in records],qpos=[r[1] for r in records],raw_joints=[r[2] for r in records],joint_names=records[0][3])
