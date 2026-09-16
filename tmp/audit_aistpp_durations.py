import json
import pickle
from pathlib import Path
from collections import Counter

base = Path('realtime/humanoid_robot/data')
catalog = json.loads((base/'music_catalog/catalog.json').read_text())['motions']
records = []
errors = []
for root, kind in [(base/'aistpp', 'source'), (base/'aistpp_gmr', 'gmr')]:
    for path in sorted(root.rglob('*.pkl')):
        if '__MACOSX' in path.parts or path.name.startswith('._'):
            continue
        try:
            with path.open('rb') as h: data = pickle.load(h)
            frames = len(data['smpl_poses'] if kind == 'source' else data['dof_pos'])
            fps = float(data.get('fps', 60.0))
            records.append(dict(path=str(path.resolve()), kind=kind, folder=str(path.parent.relative_to(base)), motion_id=path.stem,
                frames=frames, fps=fps, authored_seconds=(frames-1)/fps, nominal_seconds=frames/fps,
                catalog_seconds=catalog.get(path.stem, {}).get('duration_seconds'), genre=path.stem[1:3]))
        except Exception as exc: errors.append(dict(path=str(path), error=str(exc)))

source = [r for r in records if r['folder']=='aistpp\\motions' or r['folder']=='aistpp/motions']
def stats(rows):
    values=sorted(r['authored_seconds'] for r in rows)
    return dict(count=len(rows), minimum=values[0], maximum=values[-1], under_8=sum(v<8-1e-9 for v in values), at_least_8=sum(v>=8-1e-9 for v in values))
summary = dict(folders=dict(Counter(r['folder'] for r in records)), source=stats(source),
    genres={g:stats([r for r in source if r['genre']==g]) for g in sorted({r['genre'] for r in source})},
    shortest=[r['motion_id'] for r in source if r['authored_seconds']==min(x['authored_seconds'] for x in source)],
    longest=[r['motion_id'] for r in source if r['authored_seconds']==max(x['authored_seconds'] for x in source)], errors=errors)
by_id={}
for r in records: by_id.setdefault(r['motion_id'],set()).add((r['frames'],r['fps']))
summary['inconsistent_copies']={k:sorted(v) for k,v in by_id.items() if len(v)>1}
summary['nominal_8_but_authored_under_8']=[r['motion_id'] for r in source if r['nominal_seconds']>=8 and r['authored_seconds']<8]
summary['catalog_nominal_mismatches']=[r['motion_id'] for r in source if r['catalog_seconds'] is not None and abs(r['catalog_seconds']-r['nominal_seconds'])>1e-6]
Path('tmp/aistpp_duration_audit.json').write_text(json.dumps(dict(summary=summary,files=records),indent=2))
lines=['# Local AIST++ motion duration audit', '', 'Authored duration is (frames - 1) / fps, matching v2. Nominal duration is frames / fps. Original source files use the project\'s 60 fps convention; GMR files supply fps metadata.', '', '| Genre | Motions | Minimum (s) | Maximum (s) | Below 8 s | At least 8 s |', '|---|---:|---:|---:|---:|---:|']
for g,s in summary['genres'].items(): lines.append(f"| {g} | {s['count']} | {s['minimum']:.3f} | {s['maximum']:.3f} | {s['under_8']} | {s['at_least_8']} |")
lines += ['', '## Every canonical source motion', '', '| Motion | Frames | Nominal (s) | Authored (s) |', '|---|---:|---:|---:|']
for r in source: lines.append(f"| {r['motion_id']} | {r['frames']} | {r['nominal_seconds']:.3f} | {r['authored_seconds']:.3f} |")
Path('tmp/aistpp_duration_audit.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
