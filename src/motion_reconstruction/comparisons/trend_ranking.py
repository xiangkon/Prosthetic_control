"""Balanced descriptive rankings from saved reconstruction trend metrics."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from ..legacy_g99.viewer import TASK_LABELS
from ..paths import COMPARISON_ROOT
from ..tum_native.schema import Q_NAMES, sha256, write_json
from .trend_analysis import write_csv

METRICS=('pearson','spearman','velocity_pearson','direction_agreement')


def average(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def compute(rows, angles=Q_NAMES):
    # IK/EKF must describe exactly the same eligible trial-angle observations.
    grouped=defaultdict(dict)
    for r in rows:
        if r['angle'] in angles and r['policy']=='quality' and int(r['task'])!=0 and r['eligible']=='True':
            key=tuple(int(r[k]) for k in ('subject','task','trial'))+(r['angle'],)
            if r['method'] in grouped[key]: raise ValueError('Duplicate metric row')
            grouped[key][r['method']]=r
    observations=[]
    for (subject,task,trial,angle),methods in grouped.items():
        if set(methods)!={'ik','ekf'}: continue
        for method in ('combined','ik','ekf'):
            chosen=list(methods.values()) if method=='combined' else [methods[method]]
            r=dict(subject=subject,task=task,trial=trial,angle=angle,method=method)
            for m in METRICS:
                r[m]=average([float(x[m]) if x[m] else None for x in chosen])
            observations.append(r)
    # Repeated trials are summarized first, so they do not determine group weights.
    groups=defaultdict(list)
    for r in observations: groups[(r['subject'],r['task'],r['angle'],r['method'])].append(r)
    cells=[]
    for (subject,task,angle,method),rr in sorted(groups.items()):
        cell=dict(subject=subject,task=task,angle=angle,method=method,trials=len(rr))
        for m in METRICS:
            values=[r[m] for r in rr if r[m] is not None]
            cell[m]=float(np.median(values)) if values else None
        cells.append(cell)
    results={}
    for target,other,total in (('subject','task',30),('task','subject',6)):
        result=[]
        for method in ('combined','ik','ekf'):
            identifiers=sorted(set(c[target] for c in cells))
            for ident in identifiers:
                cc=[c for c in cells if c[target]==ident and c['method']==method]
                by_angle={a:[c for c in cc if c['angle']==a] for a in angles}
                if any(not values for values in by_angle.values()):
                    raise ValueError(f'{target} {ident} has an entirely missing angle')
                record=dict(method=method,identifier=ident,
                    name=f'subj_{ident:03d}' if target=='subject' else TASK_LABELS[ident],
                    available_cells=len(cc),expected_cells=total*len(angles),
                    minimum_groups_per_angle=min(len(v) for v in by_angle.values()),
                    trial_angle_pairs=sum(c['trials'] for c in cc))
                for metric in METRICS:
                    record[metric]=average([average([c[metric] for c in by_angle[a]]) for a in angles])
                for a in angles: record['r_'+a]=average([c['pearson'] for c in by_angle[a]])
                result.append(record)
            for metric in METRICS:
                ranked=sorted([r for r in result if r['method']==method],key=lambda r:(-r[metric],r['identifier']))
                for rank,r in enumerate(ranked,1): r[metric+'_rank']=rank
        results[target]=sorted(result,key=lambda r:(r['method'],r['pearson_rank']))
    return results,cells


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=COMPARISON_ROOT/'legacy_g99_vs_v002_trends_v001'
    p.add_argument('--input',type=Path,default=root/'trial_metrics.csv')
    p.add_argument('--output',type=Path,default=root/'rankings')
    p.add_argument('--angles',nargs='+',choices=Q_NAMES,default=Q_NAMES)
    args=p.parse_args()
    if len(set(args.angles))!=len(args.angles): p.error('Angles must be unique')
    angles=args.angles
    with args.input.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    results,cells=compute(rows,angles)
    sensitivity=[]
    combined=[c for c in cells if c['method']=='combined']
    for target,other in (('subject','task'),('task','subject')):
        ids=sorted(set(c[target] for c in combined))
        shared=set.intersection(*[{(c[other],c['angle']) for c in combined if c[target]==i} for i in ids])
        scores=[]
        for ident in ids:
            chosen=[c for c in combined if c[target]==ident and (c[other],c['angle']) in shared]
            if any(not any(c['angle']==a for c in chosen) for a in angles):
                raise ValueError('Shared-cell sensitivity lost an entire angle')
            score=average([average([c['pearson'] for c in chosen if c['angle']==a]) for a in angles])
            primary=next(r for r in results[target] if r['method']=='combined' and r['identifier']==ident)
            scores.append(dict(group=target,identifier=ident,name=primary['name'],
                primary_rank=primary['pearson_rank'],shared_cells=len(shared),pearson=score))
        for rank,r in enumerate(sorted(scores,key=lambda r:(-r['pearson'],r['identifier'])),1):
            r['shared_cell_rank']=rank
            sensitivity.append(r)
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/'subject_task_angle_cells.csv',cells)
    write_csv(args.output/'shared_cell_sensitivity.csv',sensitivity)
    for target,ranked in results.items(): write_csv(args.output/f'{target}_ranking.csv',ranked)
    lines=['# 按新老方法运动趋势一致性排序','',
        '比较角度：'+', '.join(angles)+'。FA_TWIST_PROXY为前臂旋转／扭转代理，WR_RD_FUNC为桡尺偏。', '',
        f'主排名为所选{len(angles)}个比较坐标等权的Pearson相关系数，越大表示整体曲线趋势越一致；不使用绝对值，不加入角度误差或幅度误差。增减方向一致率、Spearman和角速度相关作为补充，均有各自独立排名。',
        '', '仅用主分析quality、非Task 0、符合原有活动范围和有效时长门槛的数据。每个试验—角度必须在老IK和EKF下都符合条件；先将IK/EKF指标取算术平均，再在同一受试者×任务×角度内对重复试验取中位数。IK和EKF不是独立重复样本。',
        '', '受试者排名：每个角度先对可用任务等权平均，再对所选角度等权平均。动作排名：每个角度先对可用受试者等权平均，再对所选角度等权平均。相关系数直接做描述性算术平均，未使用Fisher变换。单独IK/EKF排名也采用同一批试验—角度数据。',
        '', f'缺失或低活动单元不填零、不插补；因此排名是当前可评价运动的一致性排序，不能当作重建精度或统计显著性排序。每人最多30任务×{len(angles)}角={30*len(angles)}单元，每动作最多6人×{len(angles)}角={6*len(angles)}单元。前臂旋转仍是扭转代理，不与旧桡尺旋转严格等价。']
    for target,title in (('subject','受试者'),('task','动作')):
        lines.extend(['',f'## {title}排名','',
            '|排名|编号|名称|趋势相关得分|方向一致率|可用单元|IK排名|EKF排名|'+''.join(a+'相关|' for a in angles),
            '|---:|---:|---|---:|---:|---:|---:|---:|'+''.join('---:|' for a in angles)])
        index={(r['method'],r['identifier']):r for r in results[target]}
        combined=sorted([r for r in results[target] if r['method']=='combined'],key=lambda r:r['pearson_rank'])
        for r in combined:
            ident=r['identifier']
            lines.append(f"|{r['pearson_rank']}|{ident:03d}|{r['name']}|{r['pearson']:.4f}|{r['direction_agreement']:.1%}|{r['available_cells']}/{r['expected_cells']}|{index['ik',ident]['pearson_rank']}|{index['ekf',ident]['pearson_rank']}|"+''.join(f"{r['r_'+a]:.4f}|" for a in angles))
    lines.extend(['','CSV保留所选角度各自得分、四种指标及排名；subject_task_angle_cells.csv可追溯每个汇总单元。方向一致率沿用±5°/s阈值，两者都静止的帧不进入分母，一动一静计为不一致。排名接近的项目不宜过度区分。',
        '', '## 共同单元敏感性检查', '',
        '额外仅保留所有受试者（或所有动作）均可评价的单元，重新按角度等权计算，详见shared_cell_sensitivity.csv。该检查改变了评价范围，不能简单当作纠正后的真排名。动作排名受低活动排除和可用人员范围影响，尤其不应将相邻名次解释为确定差异。'])
    for target,title in (('subject','受试者'),('task','动作')):
        rr=[r for r in sensitivity if r['group']==target]
        lines.extend(['',f'### {title}共同单元对照（保留{rr[0]["shared_cells"]}单元）','',
            '|名称|主排名|共同单元排名|共同单元得分|','|---|---:|---:|---:|'])
        for r in rr: lines.append(f"|{r['name']}|{r['primary_rank']}|{r['shared_cell_rank']}|{r['pearson']:.4f}|")
    (args.output/'RANKING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(args.output/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input=str(args.input),angles=angles,input_sha256=sha256(args.input),implementation_sha256=sha256(Path(__file__)),
        common_trial_angle_pairs=sum(c['trials'] for c in cells if c['method']=='combined'),
        grouped_cells=sum(c['method']=='combined' for c in cells)))
    print(args.output)


if __name__=='__main__': main()
