from pathlib import Path
import hashlib,json
root=Path.cwd();out=root/'output/gmr_frame488_diagnostic'
files=[('realtime/humanoid_robot/src/gmr_state_trajectory.py','candidate.py'),('tests/test_gmr_state_trajectory.py','test_gmr_state_trajectory.py')]
manifest=[]
for target,staged in files:
    manifest.append(dict(target=target,staged=staged,before=hashlib.sha256((root/target).read_bytes()).hexdigest(),after=hashlib.sha256((out/staged).read_bytes()).hexdigest()))
(out/'patch_manifest.json').write_text(json.dumps(manifest,indent=2))
