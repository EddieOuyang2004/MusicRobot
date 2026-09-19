"""Reproducible all-pairs robot joint trajectory audit; does not modify motions."""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
import csv
import hashlib
import json
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
from numba import njit, prange
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/aistpp_similarity_audit'
N = 96
BAND = 15

@njit(parallel=True)
def distances(x, pairs):
    result = np.empty(len(pairs))
    for p in prange(len(pairs)):
        a, b = pairs[p]
        cost = np.full((N+1, N+1), np.inf)
        length = np.zeros((N+1, N+1), np.int32)
        cost[0, 0] = 0.
        for i in range(1, N+1):
            for j in range(max(1, i-BAND), min(N, i+BAND)+1):
                v = 0.
                for k in range(x.shape[2]):
                    delta = x[a,i-1,k]-x[b,j-1,k]
                    v += delta*delta
                pi, pj = i-1, j-1
                if cost[i-1,j] < cost[pi,pj]: pi,pj = i-1,j
                if cost[i,j-1] < cost[pi,pj]: pi,pj = i,j-1
                cost[i,j] = cost[pi,pj]+v/x.shape[2]
                length[i,j] = length[pi,pj]+1
        result[p] = np.sqrt(cost[N,N]/length[N,N])*180/np.pi
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT/'realtime/humanoid_robot/data/aistpp_gmr_v2').glob('*.pkl'))
    arrays, records, hashes = [], [], []
    names = None
    for path in files:
        with path.open('rb') as f: d = pickle.load(f)
        if names is None: names = list(d['dof_names'])
        assert set(names) == set(d['dof_names'])
        q = np.asarray(d['dof_pos'], dtype=np.float64)[:,[d['dof_names'].index(k) for k in names]]
        assert np.isfinite(q).all() and q.shape[0] > 1
        hashes.append(hashlib.sha256(q.tobytes()).hexdigest())
        t = np.linspace(0, 1, len(q))
        arrays.append(np.stack([np.interp(np.linspace(0,1,N),t,q[:,j]) for j in range(q.shape[1])], axis=1))
        records.append(dict(motion_id=path.stem, genre=path.stem[1:3], situation=path.stem.split('_')[1][1:], duration=(len(q)-1)/float(d['fps'])))
    # Wrist joints can dilute visible body differences; primary metric excludes them.
    keep = [i for i,n in enumerate(names) if 'wrist' not in n]
    x = np.ascontiguousarray(np.stack(arrays)[:,:,keep])
    pairs = np.array([(i,j) for i in range(len(x)) for j in range(i+1,len(x))])
    print(f'Loaded {len(x)} motions, {len(keep)} body joints; comparing {len(pairs)} pairs.', flush=True)
    vals = distances(x,pairs)
    matrix = np.zeros((len(x),len(x)))
    matrix[pairs[:,0],pairs[:,1]] = vals
    matrix += matrix.T
    aligned = squareform(pdist(x.reshape(len(x),-1),metric='euclidean'))/np.sqrt(N*len(keep))*180/np.pi
    np.savez_compressed(OUT/'distance_matrices.npz', dtw_degrees=matrix, aligned_rms_degrees=aligned, motion_ids=np.array([r['motion_id'] for r in records]))
    np.fill_diagonal(matrix,np.inf)
    nearest = matrix.argmin(axis=1)
    nv = matrix[np.arange(len(x)),nearest]
    thresholds = [5,10,15,20,25,30]
    def stats(indices):
        sub = matrix[np.ix_(indices,indices)]
        near = sub.min(axis=1)
        pv = sub[np.triu_indices(len(indices),1)]
        return dict(count=len(indices), nearest_median_deg=float(np.median(near)), pair_median_deg=float(np.median(pv)), thresholds={str(t):dict(motions_with_neighbor=int((near<=t).sum()),pair_count=int((pv<=t).sum()), connected_components=int(connected_components(sub<=t,directed=False,return_labels=False))) for t in thresholds})
    summary = dict(motions=len(x),pairs=len(pairs),samples=N,dtw_band=BAND,joints=[names[i] for i in keep],exact_duplicate_excess=len(hashes)-len(set(hashes)),situations=dict(Counter(r['situation'] for r in records)),overall=stats(list(range(len(x)))),genres={g:stats([i for i,r in enumerate(records) if r['genre']==g]) for g in sorted({r['genre'] for r in records})},nearest_quantiles_deg={str(p):float(np.percentile(nv,p)) for p in [0,25,50,75,90,100]},nearest_same_genre=int(sum(records[i]['genre']==records[j]['genre'] for i,j in enumerate(nearest))))
    with (OUT/'nearest_neighbors.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['motion_id','genre','situation','duration','nearest_motion','dtw_deg','aligned_deg','same_genre'])
        w.writeheader()
        for i,r in enumerate(records):
            j=nearest[i]
            w.writerow(dict(**r,nearest_motion=records[j]['motion_id'],dtw_deg=round(nv[i],4),aligned_deg=round(aligned[i,j],4),same_genre=r['genre']==records[j]['genre']))
    order=np.argsort(vals)
    with (OUT/'all_pairs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['motion_a','motion_b','dtw_deg','aligned_deg','same_genre'])
        for p in order:
            i,j=pairs[p]; w.writerow([records[i]['motion_id'],records[j]['motion_id'],round(vals[p],4),round(aligned[i,j],4),records[i]['genre']==records[j]['genre']])
    summary['closest_pairs']=[dict(a=records[pairs[p,0]]['motion_id'],b=records[pairs[p,1]]['motion_id'],dtw_deg=float(vals[p]),aligned_deg=float(aligned[tuple(pairs[p])])) for p in order[:20]]
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# AIST++ GMR v2 关节轨迹相似度测量','',f'比较 {len(x)} 条动作的全部 {len(pairs):,} 对；以最终保存的 GMR v2 关节轨迹为对象。','', '## 方法与限制','',f'- 统一关节顺序；排除 6 个手腕关节，保留 {len(keep)} 个身体关节，等权计算。',f'- 每条动作按完整时长线性重采样为 {N} 帧；不归一化动作幅度，不镜像，不移除关节均值。',f'- 动态时间规整 DTW：起终点固定，允许偏离对角线 {BAND} 帧，约为完整时长的 {100*BAND/(N-1):.1f}%；最小化累计均方关节误差，输出该路径的关节均方根误差（度），越小越相似。','- 同时保存不做 DTW 的时间归一化 RMS，供检查对齐的影响。','- 5/10/15/20/25/30 度仅是工程参考阈值，未用人工视觉标签标定；不能解释成相似百分比。','- 这是整段身体姿态测量，不包含根节点平移/朝向、手腕、接触状态或播放过程的额外处理；不能发现所有局部短片段重复；不同起始相位可能使结果偏高。','- 时间归一化会忽略原始速度差异；DTW 也允许局部节奏拉伸，因此衡量的是动作形状接近程度。','- 连通分组具有传递性，组内并非任意两条都满足阈值，不能称为独立编舞数量。','', '## 总体结果','',f'- 完全一致的原始关节数组重复余量：{summary["exact_duplicate_excess"]}。',f'- 每条动作最近邻距离中位数：{np.median(nv):.2f}°。',f'- 全部动作对距离中位数：{np.median(vals):.2f}°。',f'- 最近邻同舞种：{summary["nearest_same_genre"]}/{len(x)}。','', '| 阈值 | 至少有一个邻居的动作 | 全库比例 | 满足阈值的动作对 | 连通分组数 |','|---|---:|---:|---:|---:|']
    for t,s in summary['overall']['thresholds'].items(): lines.append(f'| ≤{t}° | {s["motions_with_neighbor"]} | {100*s["motions_with_neighbor"]/len(x):.1f}% | {s["pair_count"]} | {s["connected_components"]} |')
    lines += ['', '## 各舞种（最近邻仅在本舞种内寻找）','','| 舞种 | 数量 | 最近邻中位数 | ≤10° 有邻居 | ≤15° 有邻居 |','|---|---:|---:|---:|---:|']
    for g,s in summary['genres'].items(): lines.append(f'| {g} | {s["count"]} | {s["nearest_median_deg"]:.2f}° | {s["thresholds"]["10"]["motions_with_neighbor"]} | {s["thresholds"]["15"]["motions_with_neighbor"]} |')
    lines += ['', '## 最接近的 20 对','','| 动作 A | 动作 B | DTW RMS | 未规整 RMS |','|---|---|---:|---:|']
    for r in summary['closest_pairs']: lines.append(f'| {r["a"]} | {r["b"]} | {r["dtw_deg"]:.2f}° | {r["aligned_deg"]:.2f}° |')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
