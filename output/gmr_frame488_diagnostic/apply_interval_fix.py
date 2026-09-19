"""Install the validated interval-bound fix only while the default batch is idle."""
from pathlib import Path
import hashlib,json,sys
out=Path(__file__).resolve().parent
root=out.parents[1]
base=root/'realtime/humanoid_robot'
sys.path.insert(0,str(base/'src'))
from gmr_batch_lock import batch_lock,BatchAlreadyRunning
try:
    with batch_lock(base/'data/aistpp_gmr_v2/.batch.lock'):
        pending=[]
        for item in json.loads((out/'patch_manifest.json').read_text()):
            target=root/item['target'];data=target.read_bytes();replacement=(out/item['staged']).read_bytes()
            if hashlib.sha256(replacement).hexdigest()!=item['after']:
                raise RuntimeError('Staged fix changed; refusing installation.')
            digest=hashlib.sha256(data).hexdigest()
            if digest==item['after']: continue
            if digest!=item['before']:
                raise RuntimeError(f'{target} changed since testing; refusing to overwrite it.')
            compile(replacement,str(target),'exec')
            backup=out/(target.name+'.before_interval_fix')
            if backup.exists() and backup.read_bytes()!=data:
                raise RuntimeError(f'Unexpected backup: {backup}')
            pending.append((target,data,replacement,backup))
        for target,data,replacement,backup in pending:
            backup.write_bytes(data)
            temporary=target.with_name(target.name+'.interval_fix_tmp')
            temporary.write_bytes(replacement)
            temporary.replace(target)
        print('Interval-bound fix and regression tests installed.' if pending else 'Fix already installed.')
        print('Validation limits are unchanged. Use build_gmr_v2.py --resume to rebuild.')
except BatchAlreadyRunning as exc:
    print(exc)
    print('No generator source changed. Run this installer after the current batch finishes.')
    sys.exit(2)
