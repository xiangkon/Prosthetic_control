"""Read-only seven-coordinate trend comparison of legacy IK/EKF and native v002."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import spearmanr

from .viewer_data import (V002_ROOT, LEGACY_REJECT, paired_trials, common_frames,
    _check_sources, _reference_mapping, _old_thoracohumeral,
    _native_thoracohumeral, _relative_rotvec)
from ..legacy_g99.model import Model
from ..legacy_g99.kinematics import forward
from ..paths import LEGACY_ROOT, COMPARISON_ROOT, PROJECT
from ..tum_native.schema import Q_NAMES, canonical_hash, sha256, write_json


def correlation(x, y):
    if len(x) < 3 or min(np.std(x), np.std(y)) < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def trend_metrics(t, x, y, valid, smoothing_s=.1, speed_deg_s=5., min_rom_deg=5.):
    """Smooth only contiguous usable runs; never differentiate across gaps."""
    valid = valid & np.isfinite(x) & np.isfinite(y)
    dt = float(np.median(np.diff(t)))
    indices = np.flatnonzero(valid)
    cuts = (np.diff(indices) != 1) | (np.diff(t[indices]) > 1.5 * dt)
    runs = np.split(indices, np.flatnonzero(cuts) + 1)
    window = max(5, int(round(smoothing_s / dt)) | 1)
    xs, ys, vx, vy = [], [], [], []
    for ix in runs:
        if len(ix) < window:
            continue
        # Guard against irregular sampling rather than silently resampling it.
        if not np.allclose(np.diff(t[ix]), dt, rtol=.01, atol=1e-8):
            raise ValueError('Nonuniform within-run timestamps')
        a, b = (savgol_filter(z[ix], window, 2) for z in (x, y))
        da, db = (savgol_filter(z[ix], window, 2, deriv=1, delta=dt) for z in (x, y))
        trim = window // 2
        xs.extend(a[trim:-trim]); ys.extend(b[trim:-trim])
        vx.extend(da[trim:-trim]); vy.extend(db[trim:-trim])
    a, b, da, db = map(np.asarray, (xs, ys, vx, vy))
    result = dict(frames=len(t), paired_frames=int(valid.sum()),
        paired_coverage=float(valid.mean()), analyzed_frames=len(a),
        analyzed_duration_s=len(a)*dt, pearson=None, spearman=None,
        velocity_pearson=None, direction_agreement=None, moving_frames=0,
        old_rom_deg=None, new_rom_deg=None, rom_ratio=None,
        eligible=False, status='insufficient_contiguous_data')
    if len(a) < max(100, int(round(1 / dt))):
        return result
    ro, rn = (float(np.percentile(z, 95)-np.percentile(z, 5)) for z in (a, b))
    moving = (abs(da) > speed_deg_s) | (abs(db) > speed_deg_s)
    sa = np.where(abs(da) > speed_deg_s, np.sign(da), 0)
    sb = np.where(abs(db) > speed_deg_s, np.sign(db), 0)
    result.update(pearson=correlation(a,b),
        spearman=float(spearmanr(a,b).statistic) if min(np.std(a),np.std(b))>1e-9 else None,
        velocity_pearson=correlation(da,db), moving_frames=int(moving.sum()),
        direction_agreement=float(np.mean(sa[moving]==sb[moving])) if moving.any() else None,
        old_rom_deg=ro, new_rom_deg=rn, rom_ratio=rn/ro if ro>1e-9 else None,
        eligible=min(ro,rn)>=min_rom_deg,
        status='ok' if min(ro,rn)>=min_rom_deg else 'low_rom')
    return result


def load_pair(key, method, old_model, native, mapping):
    old_dir, new_dir = LEGACY_ROOT/key.relative, V002_ROOT/key.relative
    old_report = json.loads((old_dir/'report.json').read_text(encoding='utf-8'))
    new_report = json.loads((new_dir/'report.json').read_text(encoding='utf-8'))
    _check_sources(old_report, new_report)
    if (new_report['schema'] != 'tum-native-functional7-v2'
            or new_report['model_hash'] != canonical_hash(native)
            or old_report['model_sha256'] != canonical_hash(old_model.config)):
        raise ValueError('Model/schema mismatch')
    op, npth = old_dir/f'{method}_angles.npz', new_dir/'reconstruction.npz'
    with np.load(op, allow_pickle=False) as o, np.load(npth, allow_pickle=False) as n:
        oi, ni = common_frames(o['t'], n['t'])
        if o['angle_unit'].item()!='rad' or n['angle_unit'].item()!='rad' or n['q_names'].tolist()!=Q_NAMES:
            raise ValueError('Angle units/names mismatch')
        oq, nq = o['q'][oi].astype(float), n['q'][ni].astype(float)
        flags, nv = o['angle_quality_flag'][oi], n['relative_valid'][ni].astype(bool)
        t = n['t'][ni]
        a, b = np.full_like(nq,np.nan), np.full_like(nq,np.nan)
        a[:,3:], b[:,3:] = oq[:,8:], nq[:,3:]*mapping.signs+mapping.offset
        finite = np.isfinite(oq).all(1)
        if mapping.old_shoulder_zero is not None and mapping.native_shoulder_zero is not None:
            a[finite,:3] = _relative_rotvec(_old_thoracohumeral(forward(old_model,oq[finite])[2],mapping.bridge),mapping.old_shoulder_zero)
            b[:,:3] = _relative_rotvec(_native_thoracohumeral(nq,native),mapping.native_shoulder_zero)
        sf = np.bitwise_or.reduce(flags[:,:8],axis=1)
        of = np.column_stack([np.repeat(sf[:,None],3,axis=1), flags[:,8:]])
        ov = (of & LEGACY_REJECT)==0
        for q, mask in ((a,ov),(b,nv)):
            mask &= np.isfinite(q)
            mask[:,:3] &= (np.linalg.norm(q[:,:3],axis=1)<=np.deg2rad(170))[:,None]
    provenance = dict(subject=key.subject, trial=key.trial, method=method,
        task=int(new_report['job']['task']), old_sha256=sha256(op), new_sha256=sha256(npth),
        source_sha256=new_report['source_sha256'],
        historical_old_source_hash_verified=bool(old_report.get('source_sha256')),
        old_model_hash=canonical_hash(old_model.config), new_model_hash=canonical_hash(native))
    return t, np.rad2deg(a), np.rad2deg(b), ov, nv, provenance


def summarize(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    output=[]
    for group, rr in sorted(groups.items()):
        used=[r for r in rr if r['eligible'] and r['task']!=0]
        record=dict(zip(keys,group)); record.update(trials=len(rr),eligible_nonrest_trials=len(used))
        for name in ('paired_coverage','pearson','spearman','velocity_pearson','direction_agreement','rom_ratio'):
            values=[r[name] for r in (rr if name=='paired_coverage' else used) if r[name] is not None]
            for suffix,p in (('q25',25),('median',50),('q75',75)):
                record[name+'_'+suffix]=float(np.percentile(values,p)) if values else None
        output.append(record)
    return output


def write_csv(path, rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subjects',nargs='+',type=int,default=list(range(1,7)))
    p.add_argument('--trials',nargs='+',type=int)
    p.add_argument('--output',type=Path,default=COMPARISON_ROOT/'legacy_g99_vs_v002_trends_v001')
    p.add_argument('--smoothing-s',type=float,default=.1)
    p.add_argument('--speed-deg-s',type=float,default=5.)
    p.add_argument('--min-rom-deg',type=float,default=5.)
    args=p.parse_args()
    if min(args.smoothing_s,args.speed_deg_s,args.min_rom_deg)<=0: p.error('Thresholds must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]; sources=[]; mappings=[]; failures=[]; cache={}
    keys=[k for k in paired_trials() if k.subject in args.subjects and (not args.trials or k.trial in args.trials)]
    for count,key in enumerate(keys,1):
        for method in ('ik','ekf'):
            if not (LEGACY_ROOT/key.relative/f'{method}_angles.npz').exists(): continue
            try:
                ck=(key.subject,method)
                if ck not in cache:
                    old=Model.load(LEGACY_ROOT/f'subj_{key.subject:03d}'/'estimated_model.json',allow_estimated=True)
                    native=json.loads((V002_ROOT/f'subj_{key.subject:03d}'/'model.json').read_text(encoding='utf-8'))
                    mapping=_reference_mapping(old,native,LEGACY_ROOT,V002_ROOT,key.subject,method)
                    cache[ck]=(old,native,mapping)
                    mappings.append(dict(subject=key.subject,method=method,signs=mapping.signs.tolist(),
                        offsets_deg=np.rad2deg(mapping.offset).tolist(),reference=mapping.reference,
                        notes=mapping.notes,old_shoulder_zero=mapping.old_shoulder_zero.tolist() if mapping.old_shoulder_zero is not None else None,
                        native_shoulder_zero=mapping.native_shoulder_zero.tolist() if mapping.native_shoulder_zero is not None else None,
                        bridge=[v.tolist() for v in mapping.bridge] if mapping.bridge is not None else None))
                t,a,b,ov,nv,source=load_pair(key,method,*cache[ck]); sources.append(source)
                for j,name in enumerate(Q_NAMES):
                    for policy,mask in (('quality',ov[:,j]&nv[:,j]),('finite_sensitivity',np.isfinite(a[:,j])&np.isfinite(b[:,j]))):
                        row=dict(subject=key.subject,trial=key.trial,task=source['task'],method=method,angle=name,policy=policy,
                            old_coverage=float(ov[:,j].mean()),new_coverage=float(nv[:,j].mean()))
                        row.update(trend_metrics(t,a[:,j],b[:,j],mask,args.smoothing_s,args.speed_deg_s,args.min_rom_deg)); rows.append(row)
            except Exception as exc:
                failures.append(dict(subject=key.subject,trial=key.trial,method=method,error=str(exc)))
        if count%10==0 or count==len(keys): print(f'{count}/{len(keys)} trials; failures={len(failures)}',flush=True)
    write_csv(args.output/'trial_metrics.csv',rows)
    summary=summarize(rows,['method','policy','angle'])
    write_csv(args.output/'summary_by_angle.csv',summary)
    write_csv(args.output/'summary_by_subject.csv',summarize(rows,['subject','method','policy','angle']))
    write_csv(args.output/'summary_by_task.csv',summarize(rows,['task','method','policy','angle']))
    write_json(args.output/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        paired_trials=len(keys),sources=sources,reference_mappings=mappings,failures=failures,
        implementation_sha256=sha256(Path(__file__)),reconstruction_performed=False))
    lines=['# 老方法与新方法002：七坐标趋势分析','',
        '肩部为统一胸廓—上臂坐标下、各自Task 0参考归零的旋转向量；不是旧GH三个欧拉角。远端四角仅校正方向和参考偏移，不拟合幅度。',
        '',f'配对试验 {len(keys)}；成功读取方法—试验 {len(sources)}；失败 {len(failures)}。',
        '', '下表为质量筛选、非Task 0、两条曲线稳健活动范围均≥5°且可分析时长≥1 s的试验中位数；试验等权。',
        '', '|旧方法|角度|纳入/全部试验|Pearson|Spearman|方向一致率|角速度相关|幅度比 新/旧|共同覆盖率*|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v): return '—' if v is None else f'{v:.3f}'
    for r in summary:
        if r['policy']=='quality':
            lines.append('|'+ '|'.join([r['method'],r['angle'],f"{r['eligible_nonrest_trials']}/{r['trials']}"]+[fmt(r[n+'_median']) for n in ('pearson','spearman','direction_agreement','velocity_pearson','rom_ratio','paired_coverage')])+'|')
    lines.extend(['','*共同覆盖率在全部配对试验上取中位数，未排除静止或低活动试验。',
        '',f'平滑窗 {args.smoothing_s} s，二阶Savitzky–Golay；仅在连续可用段内计算并去掉半窗边缘。增减方向以±{args.speed_deg_s}°/s为阈值；任一方法运动时纳入，一动一静计为不一致，两者静止不进入分母。',
        '', '同时输出finite_sensitivity（有限数值、不按质量剔除）以检查质量筛选的选择效应。主汇总排除低活动试验，因此不代表全部动作的表现；结合trial_metrics.csv逐试验覆盖率、活动范围及summary_by_subject/task.csv判读。',
        '', '未平移时间、未做DTW、未跨缺口插值。零位和方向不从动态曲线拟合。前臂为扭转代理，肩部包含肩带贡献；一致性不能当作解剖精度。老结果缺少历史源哈希时，即使当前来源核对通过，也不能证明历史原始数据未变化。',
        '', '所有原生结果保持不变。输入NPZ哈希、002参考参数及失败记录见manifest.json。'])
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(str(args.output),flush=True)


if __name__=='__main__': main()
