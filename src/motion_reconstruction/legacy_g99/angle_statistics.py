"""Descriptive distal-angle coverage for selecting reconstruction trials."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..paths import OUTPUT_ROOT
from .marker_analysis import write_json
from .viewer import TASK_LABELS

ANGLE_NAMES = ['肘屈伸', '前臂旋前旋后', '腕屈伸', '腕桡尺偏']
PLOT_NAMES = ['q9 Elbow F/E', 'q10 Forearm P/S', 'q11 Wrist F/E', 'q12 Wrist R/U']
STRATA = ['finite', 'conditional']
REJECT_BITS = np.uint16(0xFFFF ^ 64)


def write_csv(path, rows, fieldnames=None):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames if fieldnames is not None else list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    keys = ['min', 'p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99', 'max', 'mean', 'std', 'range', 'rom_p95_p05']
    if not len(values):
        return dict.fromkeys(keys, np.nan)
    quantiles = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    result = dict(zip(keys[:9], map(float, quantiles)))
    result.update(mean=float(np.mean(values)), std=float(np.std(values)),
                  range=float(quantiles[-1] - quantiles[0]), rom_p95_p05=float(quantiles[6] - quantiles[2]))
    return result


def velocities(t, q, mask):
    """Differentiate only consecutive accepted samples, with no gap bridging."""
    result = np.full_like(q, np.nan, dtype=float)
    dt = np.diff(t)
    interval_ok = (dt > 0) & (dt <= 1.5 * np.median(dt))
    accepted = mask[1:] & mask[:-1] & interval_ok[:, None]
    difference = np.diff(q, axis=0) / dt[:, None]
    result[1:] = np.where(accepted, difference, np.nan)
    return result


def read_trials(root, subjects, method, baseline_window):
    trials = []
    for subject in subjects:
        folder = root / f'subj_{subject:03d}'
        manifest = json.loads((folder / 'trial_manifest.json').read_text(encoding='utf-8'))
        for entry in manifest:
            if entry['status'] != 'computed_estimates':
                continue
            directory = folder / 'trials' / f'trial_{entry["trial"]:03d}'
            report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
            with np.load(directory / f'{method}_angles.npz', allow_pickle=False) as archive:
                if str(archive['angle_unit'].item()) != 'rad':
                    raise ValueError(f'Unexpected angle unit in {directory}')
                t = archive['t'].astype(float)
                q = np.rad2deg(archive['q'][:, 8:].astype(float))
                flags = archive['angle_quality_flag'][:, 8:].copy()
            if len(t) < 2 or np.any(np.diff(t) <= 0):
                raise ValueError(f'Invalid time axis in {directory}')
            finite = np.isfinite(q)
            conditional = finite & ((flags & REJECT_BITS) == 0)
            baseline = np.full(4, np.nan)
            baseline_time = np.full(4, np.nan)
            for angle in range(4):
                indices = np.flatnonzero(finite[:, angle])
                if len(indices):
                    first = indices[0]
                    window = finite[:, angle] & (t >= t[first]) & (t <= t[first] + baseline_window)
                    baseline[angle] = np.median(q[window, angle])
                    baseline_time[angle] = t[first]
            masks = {'finite': finite, 'conditional': conditional}
            trials.append(dict(subject=folder.name, trial=report['trial'], task=report['task'],
                description=report['description'], synced=report['sync']['status'] == 'duration_validated_pulse_candidate',
                t=t, q=q, flags=flags, dt=float(np.median(np.diff(t))), baseline=baseline,
                baseline_time=baseline_time, delta=q-baseline, masks=masks,
                velocity={name:velocities(t,q,mask) for name,mask in masks.items()}))
    if not trials:
        raise ValueError('No computed reconstruction trials found')
    return trials


def angle_values(trials, angle, stratum, coordinate='q'):
    return np.concatenate([trial[coordinate][:, angle][trial['masks'][stratum][:, angle]] for trial in trials])


def summarize(trials, angle, stratum, activity_threshold):
    values = angle_values(trials, angle, stratum)
    delta = angle_values(trials, angle, stratum, 'delta')
    speed = np.concatenate([trial['velocity'][stratum][:, angle] for trial in trials])
    finite_speed = speed[np.isfinite(speed)]
    accepted = sum(int(trial['masks'][stratum][:, angle].sum()) for trial in trials)
    all_frames = sum(len(trial['t']) for trial in trials)
    prior = sum(int(np.sum(trial['masks'][stratum][:, angle] & ((trial['flags'][:, angle] & 64) != 0))) for trial in trials)
    strict = sum(int(np.sum(trial['masks'][stratum][:, angle] & (trial['flags'][:, angle] == 0))) for trial in trials)
    duration = sum(float(trial['masks'][stratum][:, angle].sum()) * trial['dt'] for trial in trials)
    trial_ranges = []
    trial_robust_ranges = []
    per_subject = {}
    for trial in trials:
        samples = trial['q'][:, angle][trial['masks'][stratum][:, angle]]
        if len(samples) >= 20:
            statistics = distribution(samples)
            trial_ranges.append(statistics['range'])
            trial_robust_ranges.append(statistics['rom_p95_p05'])
            per_subject.setdefault(trial['subject'], []).append(statistics['rom_p95_p05'])
    balanced = [np.median(rom) for rom in per_subject.values()]
    denominator = len(finite_speed)
    result = dict(angle=f'q{angle+9}', name=ANGLE_NAMES[angle], stratum=stratum,
        trials=len(trials), subjects=len({trial['subject'] for trial in trials}),
        synced_trials=sum(trial['synced'] for trial in trials), frames=accepted,
        sample_duration_s=duration, retained_fraction=accepted/max(all_frames,1),
        prior_fraction=prior/max(accepted,1), strict_fraction=strict/max(accepted,1),
        positive_velocity_fraction=float(np.sum(finite_speed > activity_threshold))/max(denominator,1),
        negative_velocity_fraction=float(np.sum(finite_speed < -activity_threshold))/max(denominator,1),
        active_fraction=float(np.sum(abs(finite_speed) > activity_threshold))/max(denominator,1),
        stationary_fraction=float(np.sum(abs(finite_speed) <= activity_threshold))/max(denominator,1),
        speed_abs_p50=float(np.median(abs(finite_speed))) if denominator else np.nan,
        speed_abs_p95=float(np.percentile(abs(finite_speed),95)) if denominator else np.nan,
        speed_abs_p99=float(np.percentile(abs(finite_speed),99)) if denominator else np.nan,
        speed_abs_max=float(np.max(abs(finite_speed))) if denominator else np.nan,
        valid_velocity_samples=denominator, trials_with_rom=len(trial_ranges),
        trial_range_max=float(np.max(trial_ranges)) if trial_ranges else np.nan,
        trial_robust_rom_p50=float(np.median(trial_robust_ranges)) if trial_ranges else np.nan,
        trial_robust_rom_p90=float(np.percentile(trial_robust_ranges,90)) if trial_ranges else np.nan,
        trial_robust_rom_max=float(np.max(trial_robust_ranges)) if trial_ranges else np.nan,
        subject_balanced_robust_rom=float(np.median(balanced)) if balanced else np.nan)
    result.update(distribution(values))
    result['range_to_robust_rom_ratio'] = result['range']/max(result['rom_p95_p05'],1e-6)
    result.update({'delta_'+key:value for key,value in distribution(delta).items()})
    return result


def joint_activity(trials, threshold):
    rows = []
    patterns = {}
    for trial in trials:
        velocity = trial['velocity']['conditional']
        usable = np.isfinite(velocity).all(axis=1)
        selected = velocity[usable]
        active_count = (abs(selected) > threshold).sum(axis=1)
        count = len(selected)
        row = dict(subject=trial['subject'], trial=trial['trial'], task=trial['task'],
                   description=trial['description'], synced=trial['synced'],
                   nominal_duration_s=len(trial['t'])*trial['dt'],
                   joint_conditional_duration_s=count*trial['dt'])
        for number in range(5):
            row[f'active_{number}_dof_fraction'] = float(np.sum(active_count == number))/max(count,1)
        for first in range(4):
            for second in range(first+1,4):
                correlation = np.nan
                if count >= 20 and np.std(selected[:,first]) > 1e-8 and np.std(selected[:,second]) > 1e-8:
                    correlation = float(np.corrcoef(selected[:,first],selected[:,second])[0,1])
                row[f'velocity_corr_q{first+9}_q{second+9}'] = correlation
        rows.append(row)
        signs = np.where(selected > threshold,1,np.where(selected < -threshold,-1,0))
        states, counts = np.unique(signs,axis=0,return_counts=True)
        for state, samples in zip(states,counts):
            key = tuple(int(value) for value in state)
            patterns[key] = patterns.get(key,0.) + int(samples)*trial['dt']
    total = sum(patterns.values())
    pattern_rows = [dict(q9=state[0],q10=state[1],q11=state[2],q12=state[3],
                         duration_s=duration,fraction=duration/max(total,1e-12))
                    for state,duration in sorted(patterns.items(),key=lambda item:-item[1])]
    return rows,pattern_rows


def plots_and_histograms(trials, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    histogram_rows = []
    figure, axes = plt.subplots(4,2,figsize=(12,12),layout='constrained')
    for angle in range(4):
        for column, coordinate in enumerate(['q','delta']):
            all_values = angle_values(trials,angle,'finite',coordinate)
            if not len(all_values):
                continue
            low = np.floor(all_values.min()/5)*5
            high = np.ceil(all_values.max()/5)*5
            edges = np.arange(low, max(high,low+5)+5,5)
            for stratum,color in [('finite','#b7bdc5'),('conditional','#287a9b')]:
                values = angle_values(trials,angle,stratum,coordinate)
                counts,_ = np.histogram(values,bins=edges)
                percent = counts/max(len(values),1)*100
                axes[angle,column].stairs(percent,edges,color=color,fill=True,alpha=.45,label=stratum)
                for left,right,count,percentage in zip(edges[:-1],edges[1:],counts,percent):
                    histogram_rows.append(dict(angle=f'q{angle+9}',coordinate=coordinate,stratum=stratum,
                        bin_left_deg=float(left),bin_right_deg=float(right),frames=int(count),percent=float(percentage)))
            axes[angle,column].set_title(PLOT_NAMES[angle] + (' | absolute' if coordinate=='q' else ' | relative to start'))
            axes[angle,column].set(xlabel='Angle (deg)',ylabel='Samples per bin (%)')
            axes[angle,column].grid(alpha=.15)
    axes[0,0].legend()
    figure.suptitle('Model-based distal-angle distributions | conditional keeps prior flag 64',fontsize=12)
    figure.savefig(output/'angle_distributions.png',dpi=160)
    plt.close(figure)
    tasks = sorted({row['task'] for row in summary})
    matrix = np.full((len(tasks),4),np.nan)
    for row in summary:
        if row['stratum']=='conditional':
            matrix[tasks.index(row['task']),int(row['angle'][1:])-9] = row['subject_balanced_robust_rom']
    figure, ax = plt.subplots(figsize=(10,13),layout='constrained')
    image = ax.imshow(matrix,aspect='auto',cmap='YlGnBu',vmin=0)
    ax.set_xticks(range(4),PLOT_NAMES)
    descriptions = {trial['task']:trial['description'] for trial in trials}
    ax.set_yticks(range(len(tasks)),[f'{task:02d} {descriptions[task]}' for task in tasks],fontsize=8)
    for row in range(len(tasks)):
        for column in range(4):
            if np.isfinite(matrix[row,column]):
                ax.text(column,row,f'{matrix[row,column]:.1f}',ha='center',va='center',fontsize=8,
                        color='white' if matrix[row,column] > np.nanmax(matrix)*.6 else '#203747')
    ax.set_title('Within-trial P95-P05 ROM | median over repeats, then subjects')
    figure.colorbar(image,ax=ax,label='Angle range (deg)',shrink=.65)
    figure.savefig(output/'task_rom.png',dpi=160)
    plt.close(figure)
    write_csv(output/'angle_histograms.csv',histogram_rows,
              ['angle','coordinate','stratum','bin_left_deg','bin_right_deg','frames','percent'])


def report(trials, global_rows, task_rows, trial_rows, overview, output, method, threshold, baseline_window):
    descriptions = {trial['task']:trial['description'] for trial in trials}
    total_duration = sum(len(trial['t'])*trial['dt'] for trial in trials)
    lines = ['# TUM肘腕四自由度角度覆盖统计','',
        f'统计采用 `{method}_angles.npz`，覆盖{len(trials)}条试验、{len({trial["subject"] for trial in trials})}位受试者，'
        f'共{sum(len(trial["t"]) for trial in trials):,}帧，名义时长{total_duration/60:.2f}分钟。角度单位为度，角速度为度/秒。','',
        '## 统计口径','',
        '- `finite`包含全部有限角度；`conditional`排除缺失、未收敛、残差、弱观测、接触、边界和标定失败等质量位，保留位64的先验依赖。',
        '- 前臂q10全部具有绝对相位先验标记，subj_005的q9也有先验标记。conditional用于固定模型下的数值覆盖分析，绝对准确性仍需独立验证。',
        f'- 相对角以每条试验各角首个有限采样起{baseline_window:.2f}秒内的中位数为基准。该窗口属于分析约定，任务起始姿态和皮肤标记误差会影响比较。',
        '- 单试验最大范围采用max−min，稳健范围采用P95−P05。每条试验至少20个可用采样才纳入试验范围汇总。',
        '- 全局角度分位数按采样点汇总；动作热图先对每位受试者的重复试验稳健范围取中位数，再在受试者间取中位数。',
        f'- 运动比例按相邻合格角度求差，绝对角速度超过{threshold:g}度/秒记为运动；不跨越无效帧。该阈值属于分析约定。','',
        '## 全局绝对角度分布','',
        '| 自由度 | 全部有限最小–最大 | conditional最小–最大 | conditional P5–P95 | 保留比例 | 位64比例 |',
        '|---|---|---|---|---:|---:|']
    for angle in range(4):
        raw = next(row for row in global_rows if row['angle']==f'q{angle+9}' and row['stratum']=='finite')
        row = next(row for row in global_rows if row['angle']==f'q{angle+9}' and row['stratum']=='conditional')
        lines.append(f'| q{angle+9} {ANGLE_NAMES[angle]} | {raw["min"]:.1f}–{raw["max"]:.1f} | '
            f'{row["min"]:.1f}–{row["max"]:.1f} | {row["p05"]:.1f}–{row["p95"]:.1f} | '
            f'{row["retained_fraction"]:.1%} | {row["prior_fraction"]:.1%} |')
    lines += ['', '## 单试验活动范围与速度','',
        '| 自由度 | 最大峰峰范围 | 典型稳健范围中位数 | 最大稳健范围 | 角速度绝对值P95 | 运动采样比例 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in global_rows:
        if row['stratum']=='conditional':
            lines.append(f'| {row["angle"]} {row["name"]} | {row["trial_range_max"]:.1f} | '
                f'{row["trial_robust_rom_p50"]:.1f} | {row["trial_robust_rom_max"]:.1f} | '
                f'{row["speed_abs_p95"]:.1f} | {row["active_fraction"]:.1%} |')
    lines += ['', '## 最大范围对应试验的核验','',
        '| 自由度 | 受试者/试验 | 峰峰范围 | 同一试验P95−P5 | 该试验保留比例 | 角速度绝对值P99 |',
        '|---|---|---:|---:|---:|---:|']
    extremes = []
    for angle in range(4):
        selected = [row for row in trial_rows if row['stratum']=='conditional' and row['angle']==f'q{angle+9}'
                    and row['trials_with_rom'] and np.isfinite(row['range'])]
        if selected:
            extreme = max(selected,key=lambda row:row['range'])
            extremes.append(extreme)
            lines.append(f'| q{angle+9} | {extreme["subject"]}/trial_{extreme["trial"]:03d} | '
                f'{extreme["range"]:.1f} | {extreme["rom_p95_p05"]:.1f} | '
                f'{extreme["retained_fraction"]:.1%} | {extreme["speed_abs_p99"]:.1f} |')
    suspicious = [row for row in extremes if row['range']>=20 and row['range_to_robust_rom_ratio']>3]
    if suspicious:
        example = max(suspicious,key=lambda row:row['range_to_robust_rom_ratio'])
        lines += ['',f'{example["angle"]}的最大峰峰范围来自{example["subject"]}的trial_{example["trial"]:03d}，'
            f'其峰峰范围为{example["range"]:.1f}°，而conditional样本P95−P5仅{example["rom_p95_p05"]:.1f}°，'
            f'角速度绝对值P99为{example["speed_abs_p99"]:.1f}°/s。'
            '少量极端值显著拉大了范围，需核验原始标记、轨迹和解的分支。当前质量位仍可能保留短时异常，'
            '训练覆盖应结合稳健范围、有效时长与速度判断。这里的极值/稳健范围比大于3仅用于提示人工检查。','']
    lines += ['', '## 各自由度覆盖较大的动作','',
        '下表按受试者平衡的P95−P05范围排序，列出的范围是每条试验内的活动幅度。范围较大只是候选依据，训练筛选还需结合质量、有效时长和协同运动。','',
        '| 自由度 | 排名 | 动作 | 平衡稳健范围 | 受试者数 | 试验数 | conditional保留比例 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for angle in range(4):
        selected = [row for row in task_rows if row['angle']==f'q{angle+9}' and row['stratum']=='conditional'
                    and np.isfinite(row['subject_balanced_robust_rom'])]
        for rank,row in enumerate(sorted(selected,key=lambda row:-row['subject_balanced_robust_rom'])[:5],1):
            task = row['task']
            label = TASK_LABELS[task] if task < len(TASK_LABELS) else descriptions[task]
            lines.append(f'| q{angle+9} | {rank} | {task} {label} | {row["subject_balanced_robust_rom"]:.1f} | '
                f'{row["subjects"]} | {row["trials"]} | {row["retained_fraction"]:.1%} |')
    durations = np.array([row['joint_conditional_duration_s'] for row in overview])
    denominator = max(durations.sum(),1e-12)
    fractions = [float(sum(row[f'active_{count}_dof_fraction']*duration for row,duration in zip(overview,durations))/denominator)
                 for count in range(5)]
    lines += ['', '## 多自由度运动与训练筛选','',
        f'四角同时满足conditional条件的相邻采样区间共{durations.sum()/60:.2f}分钟。'
        f'其中0/1/2/3/4个自由度运动的时长比例依次为'+ '、'.join(f'{value:.1%}' for value in fractions)+'。','',
        '选择训练数据时还需关注以下信息。','',
        '1. 角度覆盖密度与持续时长，检查极端角度是否只有很少采样，兼顾常见工作区和边缘工作区。',
        '2. 角速度、正反向运动和静止比例，覆盖启动、停止、快速运动及回程，保留抑制非预期输出所需的静止段。',
        '3. 单自由度与多自由度协同，结合速度相关性和运动状态组合，避免训练只学到任务内固定耦合。',
        '4. 跨受试者与重复一致性，检查覆盖是否由少数人或单条试验提供，先按受试者/试验划分数据，再切窗。',
        '5. 角度质量与同步，结合逐角质量、位64、时间缺口和EMG对齐状态；相对角能减少常量零位差，仍带有模型误差。',
        '6. 肌电质量和负载条件，另行检查噪声、饱和、掉线、通道有效性、幅值/频谱及肌电与运动的时间关系。持物或等长收缩时也可能角度静止。','',
        '## 文件','',
        '- [全局统计](global_angle_summary.csv)、[逐人统计](subject_angle_summary.csv)、[逐动作统计](task_angle_summary.csv)。',
        '- [逐试验逐角统计](trial_angle_summary.csv)包含绝对/相对角度分位数、范围、质量、速度和基准时间。',
        '- [逐试验协同统计](trial_activity_summary.csv)包含有效时长、同步、0–4自由度运动比例和两两速度相关性。',
        '- [运动方向组合](activity_patterns.csv)按四角负向/静止/正向组合统计持续时间。',
        '- [分布直方图数据](angle_histograms.csv)、[角度分布图](angle_distributions.png)、[动作范围热图](task_rom.png)。','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    metadata = dict(method=method,trials=len(trials),subjects=len({trial['subject'] for trial in trials}),
        frames=sum(len(trial['t']) for trial in trials),nominal_duration_s=total_duration,
        activity_threshold_deg_s=threshold,baseline_window_s=baseline_window,
        conditional_reject_bits=int(REJECT_BITS),activity_count_fractions=fractions,
        global_summary=global_rows)
    write_json(output/'summary.json',metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=OUTPUT_ROOT)
    parser.add_argument('--output',type=Path,default=OUTPUT_ROOT/'analysis/angle_statistics')
    parser.add_argument('--subjects',default='1,2,3,4,5,6')
    parser.add_argument('--method',choices=['ik','ekf'],default='ekf')
    args = parser.parse_args(argv)
    threshold, baseline_window = 5., .2
    trials = read_trials(args.results,[int(value) for value in args.subjects.split(',')],args.method,baseline_window)
    args.output.mkdir(parents=True,exist_ok=True)
    global_rows, subject_rows, task_rows, trial_rows = [], [], [], []
    for stratum in STRATA:
        for angle in range(4):
            global_rows.append(summarize(trials,angle,stratum,threshold))
            for subject in sorted({trial['subject'] for trial in trials}):
                selected = [trial for trial in trials if trial['subject']==subject]
                subject_rows.append(dict(subject=subject,**summarize(selected,angle,stratum,threshold)))
            for task in sorted({trial['task'] for trial in trials}):
                selected = [trial for trial in trials if trial['task']==task]
                task_rows.append(dict(task=task,description=selected[0]['description'],
                                      **summarize(selected,angle,stratum,threshold)))
            for trial in trials:
                trial_rows.append(dict(subject=trial['subject'],trial=trial['trial'],task=trial['task'],
                    description=trial['description'],baseline_deg=float(trial['baseline'][angle]),
                    baseline_time_s=float(trial['baseline_time'][angle]),
                    **summarize([trial],angle,stratum,threshold)))
    overview,patterns = joint_activity(trials,threshold)
    for filename,rows in [('global_angle_summary.csv',global_rows),('subject_angle_summary.csv',subject_rows),
                          ('task_angle_summary.csv',task_rows),('trial_angle_summary.csv',trial_rows),
                          ('trial_activity_summary.csv',overview)]:
        write_csv(args.output/filename,rows)
    write_csv(args.output/'activity_patterns.csv',patterns,['q9','q10','q11','q12','duration_s','fraction'])
    plots_and_histograms(trials,task_rows,args.output)
    report(trials,global_rows,task_rows,trial_rows,overview,args.output,args.method,threshold,baseline_window)
    print(json.dumps({'trials':len(trials),'output':str(args.output),'method':args.method}))
