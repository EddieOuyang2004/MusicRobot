import sys, json, time, importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path('realtime/humanoid_robot/src').resolve()))
import motion_bridges as new
spec=importlib.util.spec_from_file_location('original_bridges', 'tmp/motion_bridges_before_optimization.py')
old=importlib.util.module_from_spec(spec); sys.modules[spec.name]=old; spec.loader.exec_module(old)
new.warmup_hermite()
report={}
for corpus in ('chronos','synthetic'):
    path='tmp/v2_backend_cases_chronos.jsonl' if corpus=='chronos' else 'tmp/v2_backend_synthetic.jsonl'
    cases=[json.loads(x) for x in Path(path).read_text().splitlines()]
    data=[]
    for c in cases:
        v={k:np.asarray(c[k],dtype=float) for k in ('lower','upper','speed','acceleration','jerk')}
        v.update(start=new.JointState(*map(np.asarray,c['start'])),end=new.JointState(*map(np.asarray,c['end'])),minimum=c['minimum'],maximum=c['maximum'])
        data.append(v)
    result={}; outcomes=[]
    for label,module in [('original',old),('optimized',new)]:
        times=[]; decisions=[]
        for v in data:
            t=time.perf_counter()
            try: d=module.make_bridge('hermite',**v).duration
            except module.InfeasibleBridge: d=None
            times.append((time.perf_counter()-t)*1000); decisions.append(d)
        result[label]={'mean_ms':float(np.mean(times)),'p95_ms':float(np.percentile(times,95)),'total_ms':sum(times),'accepted':sum(x is not None for x in decisions)}
        outcomes.append(decisions)
    assert outcomes[0]==outcomes[1], corpus
    result['identical_decisions_and_durations']=True
    report[corpus]=result
    print(corpus,result,flush=True)
Path('tmp/v2_hermite_optimized_benchmark.json').write_text(json.dumps(report,indent=2))
