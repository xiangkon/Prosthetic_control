"""Full-cohort q10 screening; thresholds flag estimates, not anatomical truth."""
from pathlib import Path
import argparse
import csv
import json
from datetime import datetime

import numpy as np

from ..paths import PROJECT, OUTPUT_ROOT, DATA_ROOT, resolve_historical_path
from .data_io import load_motion
from .pronation_audit import baseline, fingerprint, marker_twist, write_json
from src.task_metadata import tum_task_description_zh

THRESHOLDS = dict(warning_step_deg=5., severe_step_deg=20., max_step_interval_s=.01,
    boundary_tolerance_deg=.1, boundary_duration_s=.1,
    high_speed_deg_s=1500., high_speed_duration_s=.02,
    marker_stride=10, marker_support_window_s=.04)


def export_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows)


def runs(mask, t, interval=False):
    """Do not add a sampling interval to isolated point hits or bridge gaps."""
    indices = np.flatnonzero(mask)
    result = []
    for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
        if len(run):
            first, last = int(run[0]), int(run[-1])
            stop = last + 1 if interval else last
            result.append((float(t[first]), float(t[stop]), float(t[stop] - t[first]), first, stop))
    return result


def scan(args):
    if (args.output / 'scan.json').exists():
        raise RuntimeError('Scan exists; use --stage verify or a different --output')
    records, events, errors = [], [], []
    for subject in sorted(args.results.glob('subj_*')):
        model_path = subject / 'estimated_model.json'
        model = json.loads(model_path.read_text(encoding='utf-8'))
        lower, upper = np.rad2deg(model['lower_rad'][9]), np.rad2deg(model['upper_rad'][9])
        manifest_path = subject / 'trial_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for job in manifest:
            row = dict(subject=subject.name, trial=job['trial'], task=job['task'], task_zh=tum_task_description_zh(job['task']), description=job['description'])
            try:
                folder = subject / 'trials' / f'trial_{job["trial"]:03d}'
                methods = {}
                hashes = {}
                candidate_reasons = []
                local_events = []
                for method in ['ik', 'ekf']:
                    path = folder / f'{method}_angles.npz'
                    hashes[str(path.relative_to(PROJECT))] = fingerprint(path)
                    with np.load(path, allow_pickle=False) as z:
                        t = z['t'].astype(float)
                        q = np.rad2deg(z['q'].astype(float))
                        flags = z['angle_quality_flag'][:, 9]
                    assert len(t) > 1 and np.all(np.diff(t) > 0)
                    if method == 'ik':
                        original_t = t.copy()
                    else:
                        assert np.array_equal(t, original_t)
                    dt = np.diff(t)
                    valid_pair = np.isfinite(q[:-1, 9]) & np.isfinite(q[1:, 9]) & (dt <= THRESHOLDS['max_step_interval_s'])
                    step = np.abs(np.diff(q[:, 9]))
                    speed = step / dt
                    biggest = int(np.nanargmax(np.where(valid_pair, step, np.nan))) if valid_pair.any() else None
                    metrics = dict(max_step_deg=float(step[biggest]) if biggest is not None else None,
                        max_step_time_s=float(t[biggest + 1]) if biggest is not None else None,
                        max_speed_deg_s=float(np.nanmax(np.where(valid_pair, speed, np.nan))) if valid_pair.any() else None,
                        finite_fraction=float(np.isfinite(q[:, 9]).mean()), prior_fraction=float(np.mean((flags & 64) != 0)),
                        nonprior_clean_fraction=float(np.mean((flags & np.uint16(0xFFFF ^ 64)) == 0)),
                        min_deg=float(np.nanmin(q[:, 9])), max_deg=float(np.nanmax(q[:, 9])))
                    severe = valid_pair & (step >= THRESHOLDS['severe_step_deg'])
                    warning = valid_pair & (step >= THRESHOLDS['warning_step_deg'])
                    metrics['severe_jump_count'] = int(severe.sum())
                    metrics['warning_jump_count'] = int(warning.sum())
                    if warning.any():
                        candidate_reasons.append(f'{method}_step_ge_5deg')
                    # Every warning step has its exact time, magnitude and severity.
                    for index in np.flatnonzero(warning):
                        local_events.append(dict(row, method=method, kind='severe_jump' if severe[index] else 'warning_jump',
                            start_s=float(t[index]), end_s=float(t[index + 1]), duration_s=float(dt[index]),
                            before_deg=float(q[index, 9]), after_deg=float(q[index + 1, 9]),
                            magnitude_deg=float(step[index]), speed_deg_s=float(speed[index])))
                    for label, mask in [('lower_bound', abs(q[:, 9] - lower) <= THRESHOLDS['boundary_tolerance_deg']),
                                        ('upper_bound', abs(q[:, 9] - upper) <= THRESHOLDS['boundary_tolerance_deg'])]:
                        spans = runs(mask & np.isfinite(q[:, 9]), t)
                        metrics[f'{label}_max_duration_s'] = max((span[2] for span in spans), default=0.)
                        for start, end, duration, _, _ in spans:
                            if duration + 1e-9 >= THRESHOLDS['boundary_duration_s']:
                                candidate_reasons.append(f'{method}_{label}_ge_0.1s')
                                local_events.append(dict(row, method=method, kind=label, start_s=start, end_s=end, duration_s=duration))
                    speed_spans = runs(valid_pair & (speed >= THRESHOLDS['high_speed_deg_s']), t, interval=True)
                    metrics['high_speed_max_duration_s'] = max((span[2] for span in speed_spans), default=0.)
                    for start, end, duration, _, _ in speed_spans:
                        if duration + 1e-9 >= THRESHOLDS['high_speed_duration_s']:
                            candidate_reasons.append(f'{method}_high_speed_ge_0.02s')
                            local_events.append(dict(row, method=method, kind='sustained_high_speed', start_s=start, end_s=end, duration_s=duration))
                    # Adjacent wrist/elbow diagnostics are supplementary, not q10 triggers.
                    metrics['distal_other_max_steps_deg'] = {f'q{k+1}': float(np.nanmax(abs(np.diff(q[:, k])))) for k in [8, 10, 11]}
                    methods[method] = metrics
                row.update(duration_s=float(t[-1] - t[0]), methods=methods, reasons=sorted(set(candidate_reasons)), sources=hashes,
                    model_sha256=fingerprint(model_path), manifest_sha256=fingerprint(manifest_path), candidate=bool(candidate_reasons))
                records.append(row)
                events.extend(local_events)
            except Exception as exc:
                errors.append(dict(row, error=f'{type(exc).__name__}: {exc}'))
        print(f'{subject.name}: {len(manifest)} trials scanned', flush=True)
    result = dict(created_at=datetime.now().astimezone().isoformat(), thresholds=THRESHOLDS, records=records, events=events, errors=errors)
    write_json(args.output / 'scan.json', result)
    export_csv(args.output / 'events.csv', events)
    export_csv(args.output / 'all_trials.csv', [flatten(row) for row in records])
    candidates = [row for row in records if row['candidate']]
    export_csv(args.output / 'candidates.csv', [flatten(row) for row in candidates])
    print(f'Scanned {len(records)} trials; {len(candidates)} candidates; {len(errors)} read errors.', flush=True)


def flatten(row):
    result = {key: row[key] for key in ['subject', 'trial', 'task', 'task_zh', 'description', 'candidate', 'reasons']}
    for method, metrics in row['methods'].items():
        result.update({f'{method}_{key}': value for key, value in metrics.items()})
    for key in ['classification', 'proxy_coverage', 'proxy_relative_rmse_deg', 'proxy_corr', 'marker_supported_severe_jumps', 'marker_unsupported_severe_jumps', 'marker_unassessable_severe_jumps']:
        if key in row:
            result[key] = row[key]
    return result


def verify(args):
    scan_result = json.loads((args.output / 'scan.json').read_text(encoding='utf-8'))
    assert scan_result['thresholds'] == THRESHOLDS, 'Do not mix different scan thresholds'
    records = scan_result['records']
    verified_events = []
    errors = []
    for row in records:
        if not row['candidate']:
            row['classification'] = 'not_triggered'
            continue
        identifier = f'{row["subject"]}_trial_{row["trial"]:03d}'
        cache = args.output / 'trial_checks' / f'{identifier}.json'
        if cache.exists():
            saved = json.loads(cache.read_text(encoding='utf-8'))
            assert saved['sources'] == row['sources'] and saved['thresholds'] == THRESHOLDS
            for relative, expected in row['sources'].items():
                assert fingerprint(resolve_historical_path(relative)) == expected, relative
            assert fingerprint(args.results / row['subject'] / 'estimated_model.json') == row['model_sha256']
            assert fingerprint(PROJECT / saved['verification']['marker_source']) == saved['verification']['marker_source_sha256']
            row.update(saved['verification'])
            verified_events.extend(saved['events'])
            continue
        try:
            folder = args.results / row['subject'] / 'trials' / f'trial_{row["trial"]:03d}'
            for relative, expected in row['sources'].items():
                assert fingerprint(resolve_historical_path(relative)) == expected, relative
            model_path = args.results / row['subject'] / 'estimated_model.json'
            assert fingerprint(model_path) == row['model_sha256']
            model = json.loads(model_path.read_text(encoding='utf-8'))
            raw_path = args.data / row['subject'] / 'MoCap' / f'trial_{row["trial"]:03d}.mat'
            raw = load_motion(raw_path)
            with np.load(folder / 'ik_angles.npz', allow_pickle=False) as z:
                full_t, full_ik = z['t'], np.rad2deg(z['q'][:, 9].astype(float))
            with np.load(folder / 'ekf_angles.npz', allow_pickle=False) as z:
                full_ekf = np.rad2deg(z['q'][:, 9].astype(float))
            assert len(raw.t) == len(full_t) and np.allclose(raw.t, full_t, atol=1e-8)
            stride = THRESHOLDS['marker_stride']
            proxy, cluster_rms, lever = marker_twist(raw.select(stride), model['calibration']['selected_groups']['humerus']['markers'], True)
            t, ik, ekf = full_t[::stride], full_ik[::stride], full_ekf[::stride]
            ir, er = ik - baseline(full_ik, full_t), ekf - baseline(full_ekf, full_t)
            pair = np.isfinite(proxy) & np.isfinite(er)
            correlation = float(np.corrcoef(proxy[pair], er[pair])[0, 1]) if pair.sum() > 20 and np.std(proxy[pair]) > 2 and np.std(er[pair]) > 2 else None
            local_events = []
            counters = dict(marker_supported_severe_jumps=0, marker_unsupported_severe_jumps=0, marker_unassessable_severe_jumps=0)
            for event in scan_result['events']:
                if event['subject'] != row['subject'] or event['trial'] != row['trial'] or event['kind'] != 'severe_jump':
                    continue
                event = event.copy()
                left = (t >= event['start_s'] - .04) & (t <= event['start_s']) & np.isfinite(proxy)
                right = (t >= event['end_s']) & (t <= event['end_s'] + .04) & np.isfinite(proxy)
                if not left.any() or not right.any():
                    event['marker_assessment'] = 'unassessable'
                    counters['marker_unassessable_severe_jumps'] += 1
                else:
                    change = float(np.median(proxy[right]) - np.median(proxy[left]))
                    event['marker_change_over_bracketing_window_deg'] = change
                    event['marker_window_start_s'] = float(t[left][0])
                    event['marker_window_end_s'] = float(t[right][-1])
                    # Conservative contradiction criterion, not a truth/accuracy test.
                    unsupported = abs(change) < event['magnitude_deg'] * .5
                    event['marker_assessment'] = 'no_comparable_marker_rotation' if unsupported else 'marker_change_present_needs_review'
                    counters['marker_unsupported_severe_jumps' if unsupported else 'marker_supported_severe_jumps'] += 1
                local_events.append(event)
            severe_ik = row['methods']['ik']['severe_jump_count'] > 0
            severe_ekf = row['methods']['ekf']['severe_jump_count'] > 0
            ekf_boundary = max(row['methods']['ekf']['lower_bound_max_duration_s'], row['methods']['ekf']['upper_bound_max_duration_s']) + 1e-9 >= THRESHOLDS['boundary_duration_s']
            if severe_ekf or (severe_ik and ekf_boundary):
                classification = 'clear_discontinuity_filtered_affected'
            elif severe_ik:
                classification = 'clear_ik_discontinuity_filter_requires_review'
            elif ekf_boundary:
                classification = 'boundary_suspect'
            else:
                classification = 'warning_requires_review'
            info = dict(classification=classification, proxy_coverage=float(pair.mean()),
                proxy_relative_rmse_deg=float(np.sqrt(np.mean((er[pair] - proxy[pair]) ** 2))) if pair.any() else None,
                proxy_corr=correlation, upper_arm_cluster_rms_p95_mm=float(np.nanquantile(cluster_rms, .95)),
                marker_source=str(raw_path.relative_to(PROJECT)), marker_source_sha256=fingerprint(raw_path), **counters)
            row.update(info)
            np.savetxt(args.output / 'trial_checks' / f'{identifier}.csv', np.column_stack([t, ik, ekf, ir, er, proxy, cluster_rms, lever]),
                delimiter=',', header='time_s,ik_deg,ekf_deg,ik_relative_deg,ekf_relative_deg,marker_twist_relative_deg,upperarm_cluster_rms_mm,wrist_lever_mm', comments='')
            write_json(cache, dict(thresholds=THRESHOLDS, sources=row['sources'], verification=info, events=local_events))
            verified_events.extend(local_events)
            print(f'{identifier}: {classification}, marker RMSE={info["proxy_relative_rmse_deg"]:.1f}', flush=True)
        except Exception as exc:
            row['classification'] = 'marker_verification_failed'
            errors.append(dict(subject=row['subject'], trial=row['trial'], error=f'{type(exc).__name__}: {exc}'))
    result = dict(created_at=datetime.now().astimezone().isoformat(), thresholds=THRESHOLDS, records=records, events=verified_events,
        scan_errors=scan_result['errors'], verification_errors=errors)
    write_json(args.output / 'verified.json', result)
    export_csv(args.output / 'all_trials.csv', [flatten(row) for row in records])
    export_csv(args.output / 'candidates.csv', [flatten(row) for row in records if row['candidate']])
    export_csv(args.output / 'verified_jump_events.csv', verified_events)
    counts = {name: sum(row['classification'] == name for row in records) for name in sorted({r['classification'] for r in records})}
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    result = json.loads((args.output / 'verified.json').read_text(encoding='utf-8'))
    records = result['records']
    severe = [row for row in records if row['classification'].startswith('clear_')]
    warnings = [row for row in records if row['classification'] in ['warning_requires_review', 'boundary_suspect']]
    boundary_rows = [row for row in records if max(row['methods']['ekf']['lower_bound_max_duration_s'], row['methods']['ekf']['upper_bound_max_duration_s']) + 1e-9 >= THRESHOLDS['boundary_duration_s']]
    severe_names = '、'.join(f'{row["subject"]}/trial_{row["trial"]:03d}' for row in severe) or '无'
    boundary_names = '、'.join(f'{row["subject"]}/trial_{row["trial"]:03d}' for row in boundary_rows) or '无'
    candidate_count = sum(row['candidate'] for row in records)
    not_triggered = sum(row['classification'] == 'not_triggered' for row in records)
    pages = [('severe', severe)] + [(f'warnings_{i//6+1:02d}', warnings[i:i+6]) for i in range(0, len(warnings), 6)]
    for name, group in pages:
        if not group:
            continue
        fig, axes = plt.subplots(len(group), 2, figsize=(11, 2.2 * len(group)), squeeze=False, constrained_layout=True)
        for i, row in enumerate(group):
            folder = args.results / row['subject'] / 'trials' / f'trial_{row["trial"]:03d}'
            with np.load(folder / 'ik_angles.npz', allow_pickle=False) as z:
                t, ik = z['t'], np.rad2deg(z['q'][:, 9].astype(float))
            with np.load(folder / 'ekf_angles.npz', allow_pickle=False) as z:
                ekf = np.rad2deg(z['q'][:, 9].astype(float))
            proxy = np.genfromtxt(args.output / 'trial_checks' / f'{row["subject"]}_trial_{row["trial"]:03d}.csv', delimiter=',', names=True)
            a, b = axes[i]
            a.plot(t, ik, color='#999999', lw=.7, label='IK (500 Hz)')
            a.plot(t, ekf, color='#386A9A', lw=1, label='EKF (500 Hz)')
            b.plot(t, ik - baseline(ik, t), color='#999999', lw=.7, label='IK - initial')
            b.plot(t, ekf - baseline(ekf, t), color='#386A9A', lw=1, label='EKF - initial')
            b.plot(proxy['time_s'], proxy['marker_twist_relative_deg'], color='#B77942', lw=1, label='Raw marker twist (50 Hz)')
            a.set(title=f'{row["subject"]} / trial_{row["trial"]:03d} / Task {row["task"]}: {row["description"]}', ylabel='q10 (deg)', xlabel='Time (s)')
            b.set(title=f'IK max step {row["methods"]["ik"]["max_step_deg"]:.1f} deg; marker discrepancy RMSE {row["proxy_relative_rmse_deg"]:.1f} deg', ylabel='Relative rotation (deg)', xlabel='Time (s)')
            for ax in (a, b):
                ax.grid(alpha=.15)
                ax.margins(x=.01)
            if i == 0:
                a.legend(loc='upper center', bbox_to_anchor=(.5, 1.4), ncol=2, frameon=False)
                b.legend(loc='upper center', bbox_to_anchor=(.5, 1.4), ncol=3, frameon=False, fontsize=7)
        fig.savefig(args.output / f'{name}.png', dpi=150)
        fig.savefig(args.output / f'{name}.svg')
        plt.close(fig)
    summary_rows = []
    for subject in sorted({row['subject'] for row in records}):
        subset = [row for row in records if row['subject'] == subject]
        summary_rows.append(dict(subject=subject, scanned=len(subset),
            severe_ik_trials=sum(row['methods']['ik']['severe_jump_count'] > 0 for row in subset),
            severe_ekf_trials=sum(row['methods']['ekf']['severe_jump_count'] > 0 for row in subset),
            ekf_boundary_trials=sum(max(row['methods']['ekf']['lower_bound_max_duration_s'], row['methods']['ekf']['upper_bound_max_duration_s']) + 1e-9 >= THRESHOLDS['boundary_duration_s'] for row in subset),
            warning_trials=sum(row['classification'] == 'warning_requires_review' for row in subset)))
    export_csv(args.output / 'subject_summary.csv', summary_rows)
    lines = ['# 全量旋前旋后异常筛查', '',
        f'共扫描{len(records)}条正式重建试验，原始角度读取错误{len(result["scan_errors"])}条，候选标记复核错误{len(result["verification_errors"])}条。', '',
        '## 主要结论', '',
        f'严重q10数值跳变试验共{len(severe)}条：{severe_names}。严重跳变事件共{len(result["events"])}个，其中{sum(event.get("marker_assessment") == "no_comparable_marker_rotation" for event in result["events"])}个没有相应的大幅原始标记旋转。当前正式EKF持续边界停留试验：{boundary_names}。', '',
        f'另有{len(warnings)}条预警/可疑候选，{not_triggered}条未触发本次规则。未触发不代表解剖角度正确、绝对零位正确或不存在其他类型异常。修复前的扫描应在编号备份中查看，不能将历史异常列表当作当前输出状态。', '',
        '## 全量范围与规则', '',
        '- 范围来自6位受试者的正式trial_manifest.json，包含当前全部532条有效重建，不扫描filter_validation候选或历史备份。',
        '- 本次主检q10（旋前旋后），同时保留q9/q11/q12最大步长作伴随诊断，不宣称已经全面审计其他自由度。',
        '- 短间隔为相邻时间差≤0.01秒；本批均为原始500 Hz，帧间隔约2 ms。IK/EKF相邻有限q10变化≥20°为严重数值跳变，≥5°为预警。阈值是明确的工程筛查规则，不是论文公布的临床界限。',
        '- 连续与模型上下限相差≤0.1°且至少持续0.1秒，为边界停留候选。持续时间按首末采样时刻之差计算，不给孤立采样额外加一个间隔。边界停留单独存在不能认定错误。',
        '- 速度≥1500°/秒且连续≥0.02秒也触发候选。速度计算不跨缺失角度或长时间间隙。',
        f'- 对全部{candidate_count}条候选读取原始MoCap，使用上臂点簇参照的腕部横向标记扭转作对照（50 Hz）；不用IK、EKF或标定标记偏置。严重跳变左右各0.04秒内取标记角中位数，记录实际覆盖的窗口。',
        '- 该原始标记代理仍受皮肤运动、骨段假设和近似旋转轴影响，不是独立骨骼角度真值。代理差异RMSE不是角度真实误差；较长窗口用于判断有无相应大转动，不能宣称是2 ms真值测量。', '',
        '## 严重异常列表', '',
        '| 受试者 / 试验 | 任务 | IK最大单步变化 | IK严重跳变次数 | IK最长下限停留 | 当前EKF最长下限停留 | 当前状态 |',
        '|---|---|---:|---:|---:|---:|---|']
    for row in severe:
        ik, ekf = row['methods']['ik'], row['methods']['ekf']
        status = 'IK及当前EKF异常' if row['classification'] == 'clear_discontinuity_filtered_affected' else 'IK仍异常；EKF未触发本次严重规则'
        lines.append(f'| {row["subject"]} / trial_{row["trial"]:03d} | {row["task_zh"]} | {ik["max_step_deg"]:.2f}° | {ik["severe_jump_count"]} | {ik["lower_bound_max_duration_s"]:.3f} s | {ekf["lower_bound_max_duration_s"]:.3f} s | {status} |')
    lines += ['', 'IK跳变的全部精确时刻、前后角度、速度及标记窗口见 `verified_jump_events.csv`。下限长停留的起止见 `events.csv`。`severe.png` 使用全500 Hz角度展示严重候选；原始标记以50 Hz叠加。', '',
        '当前009的EKF q10范围约138.6–142.6°，010约134.7–141.3°，虽无边界锁定，但相对标记的变化幅度仍有差异（分别约14.8°、10.6°的相对差异RMSE）；不能把“已替换”和“未触发跳变”解释为绝对正确。', '',
        '## 其余预警列表', '',
        '| 受试者 / 试验 | 任务 | IK最大单步变化 | EKF最大单步变化 |',
        '|---|---|---:|---:|']
    for row in warnings:
        lines.append(f'| {row["subject"]} / trial_{row["trial"]:03d} | {row["task_zh"]} | {row["methods"]["ik"]["max_step_deg"]:.2f}° | {row["methods"]["ekf"]["max_step_deg"]:.2f}° |')
    lines += ['', '这些预警不与约170°错误分支跳变等同。其IK最大步长约5.02–8.85°；相邻2 ms内的尖峰仍提示独立逐帧IK噪声或不稳定，但需要与原始采集质量、任务过程共同判断。当前EKF最大单步变化均小于1°，未触发本次EKF边界/持续高速规则。曲线见 `warnings_01.png` 至 `warnings_03.png`。', '',
        '## 受试者分布', '', '| 受试者 | 扫描试验 | 严重IK跳变试验 | 当前EKF持续贴边试验 | 其他预警 |', '|---|---:|---:|---:|---:|']
    for row in summary_rows:
        lines.append(f'| {row["subject"]} | {row["scanned"]} | {row["severe_ik_trials"]} | {row["ekf_boundary_trials"]} | {row["warning_trials"]} |')
    lines += ['', '## 使用建议与文件', '',
        '- 严重候选中的IK不应直接作为可信监督标签；当前EKF是否受影响应单独检查。已修复的011结果见 `../trial_repairs/subj_005/trial_011/README.md`，历史异常已归档。',
        '- 预警先保留为IK不稳定清单；正式训练若使用EKF仍需结合原有质量标记及其他标签筛选，不能自动全部删除或自动认可。',
        '- 本次未修改任何正式角度、模型、同步标签或既有质量位。没有触发重建或全量重算。',
        f'- `all_trials.csv`：全部{len(records)}条试验，包括未触发项；`candidates.csv`：{candidate_count}条候选的完整指标。',
        f'- `events.csv`：全部预警/严重跳变及边界时间区间；`verified_jump_events.csv`：{len(result["events"])}个严重跳变的标记复核。',
        '- `scan.json` / `verified.json`：完整阈值、模型/角度来源哈希、处理记录及错误清单；`trial_checks/`：逐候选曲线数据及原始标记文件哈希。',
        '- `subject_summary.csv`：受试者统计。使用入口 `python scripts/analysis/screen_pronation.py`，新扫描应另指定空的 `--output`。已有扫描可用 `--stage verify` 复核、`--stage report` 重绘；重用候选缓存前核对输入哈希。', '']
    (args.output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Report and {len(pages)} figure pages written.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--data', type=Path, default=DATA_ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT_ROOT / '_checks/pronation_screening')
    parser.add_argument('--stage', choices=['scan', 'verify', 'report', 'all'], default='all')
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if not args.output.is_relative_to((PROJECT / 'outputs').resolve()):
        parser.error('Output must stay inside project outputs/')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'trial_checks').mkdir(exist_ok=True)
    if args.stage in ['scan', 'all']:
        scan(args)
    if args.stage in ['verify', 'all']:
        verify(args)
    if args.stage in ['report', 'all']:
        report(args)


if __name__ == '__main__':
    main()
