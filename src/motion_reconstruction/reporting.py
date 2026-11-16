"""Summarize estimated full-rate results without conflating estimates and truth."""
from pathlib import Path
import argparse
import json
import numpy as np
from .model import Q_NAMES
from .pipeline import jobs
from .marker_analysis import write_json,quantile
from .paths import DATA_ROOT, OUTPUT_ROOT


def nfmt(v,digits=2):
    return f'{v:.{digits}f}' if v is not None and np.isfinite(v) else '—'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(OUTPUT_ROOT))
    parser.add_argument('--data',default=str(DATA_ROOT))
    parser.add_argument('--subjects',default='1,2,3,4,5,6')
    args=parser.parse_args(argv); root=Path(args.root)
    subject_reports=[]; missing=[]; expected=0
    subject_ids=[int(v) for v in args.subjects.split(',')]
    for subject in subject_ids:
        name=f'subj_{subject:03d}'; source=Path(args.data)/name; out=root/name
        expected_jobs=jobs(source); expected+=len(expected_jobs)
        reports=[]
        for job in expected_jobs:
            p=out/'trials'/f'trial_{job["trial"]:03d}'/'report.json'
            if p.exists(): reports.append(json.loads(p.read_text(encoding='utf-8')))
            else: missing.append(dict(subject=name,trial=job['trial']))
        if not (out/'estimated_model.json').exists(): continue
        model=json.loads((out/'estimated_model.json').read_text(encoding='utf-8'))
        frames=sum(r['ekf']['frames'] for r in reports)
        accepted=np.zeros(12); weak=np.zeros(12)
        for r in reports:
            accepted+=np.array(r['ekf']['per_angle_clean_fraction'])*r['ekf']['frames']
            weak+=np.array(r['ekf']['per_angle_weak_fraction'])*r['ekf']['frames']
        sync=[r['sync']['duration_error_s'] for r in reports if r['sync']['status']=='duration_validated_pulse_candidate']
        row=dict(subject=name,expected_trials=len(expected_jobs),computed_trials=len(reports),output_frames=frames,
            calibration_marker_count=len(model['markers']),geometry_validation=model['calibration_validation'],
            empirical_sync_matched=len(sync),sync_duration_error_max_ms=quantile(abs(np.array(sync))*1000,1),
            trial_marker_rms_p90_median_mm=quantile([r['ekf']['marker_rms_p90_mm'] for r in reports],.5),
            trial_marker_rms_p90_worst_mm=quantile([r['ekf']['marker_rms_p90_mm'] for r in reports],1),
            per_angle_conditional_clean_fraction=(accepted/max(frames,1)).tolist(),
            per_angle_weak_fraction=(weak/max(frames,1)).tolist(),
            prior_dominated_angles=model['calibration']['prior_dominated_angles'])
        subject_reports.append(row)
        write_json(out/'subject_summary.json',row)
        lines=[f'# {name} 自动标定与重建','',
            f'已计算 {len(reports)}/{len(expected_jobs)} 条有效 MoCap 试验，共 {frames} 个输出采样点。','',
            '## 标定来源','',
            '- 胸廓系由解剖标记初始化，再使用逐受试者稳定点组跟踪。','- 功能性肘轴和中心由多姿态点簇相对运动估计；沿轴不可辨识方向保留有界解剖先验。',
            '- 未输入已知12角。G99尺寸初值、标记局部偏置与逐帧角度经过带先验的稀疏多帧优化。',
            '- 肩胛参考姿态、锁骨固定值、桡尺旋转绝对相位等仍有先验依赖，均保存在模型与质量位中。','',
            f'几何拟合留出帧位置RMS的P90：{nfmt(model["calibration_validation"]["marker_rms_p90_m"]*1000)} mm。该数值不是关节角精度。','',
            '## 逐角度条件质量','',
            '| 角度 | 固定标定模型下通过质量检查的帧比例 | 弱观测比例 | 绝对坐标依赖先验 |','|---|---:|---:|---|']
        for k,nameq in enumerate(Q_NAMES):
            lines.append(f'| q{k+1} {nameq} | {accepted[k]/max(frames,1):.1%} | {weak[k]/max(frames,1):.1%} | {"是" if row["prior_dominated_angles"][k] else "未触发该标记；仍非外部角度真值"} |')
        lines+=['','## 输出','',
            '- `estimated_model.json`：标定模型、点组选择、工程先验及训练记录。','- `calibration_report.json`、`validation_report.json`：标定与条件验证。',
            '- `trials/trial_XXX/ik_angles.npz`、`ekf_angles.npz`：原时间轴角度与逐角度质量。','- `trials/trial_XXX/emg_aligned_distal.npz`：成功匹配同步窗口时生成；保留窗口外无效值。',
            '- `trial_manifest.json`：逐试验运行状态。','',
            '质量位64表示绝对角依赖先验，不应作为经过独立验证的监督真值。缺少外部角度测量，因此所有结果均为模型约束估计。']
        (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    complete=not missing and len(subject_reports)==len(subject_ids)
    summary=dict(workflow_status='completed_estimated_outputs' if complete else 'incomplete',
        independent_12_angle_truth_validated=False,expected_trials=expected,
        computed_trials=sum(r['computed_trials'] for r in subject_reports),
        full_rate_frames=sum(r['output_frames'] for r in subject_reports),
        empirical_sync_matched=sum(r['empirical_sync_matched'] for r in subject_reports),
        missing=missing,subjects=subject_reports)
    write_json(root/'summary.json',summary)
    lines=['# 原始TUM数据到12角估计：端到端运行结果','',
        '**这些结果是模型约束下的12个关节坐标估计，尚未建立全部12角的独立测量准确性。** 程序已将先验依赖、弱观测、拟合异常和同步证据分别记录。','',
        f'运行状态：`{summary["workflow_status"]}`。有效试验 {summary["computed_trials"]}/{expected}，输出采样点 {summary["full_rate_frames"]:,}，脉冲持续时间匹配 {summary["empirical_sync_matched"]} 条。','',
        '| 受试者 | 已计算/预期试验 | 点数 | 标定留出位置RMS P90 (mm) | 逐试验位置RMS P90的中位数 (mm) | 同步匹配 | 详细结果 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in subject_reports:
        lines.append(f'| {r["subject"]} | {r["computed_trials"]}/{r["expected_trials"]} | {r["calibration_marker_count"]} | {nfmt(r["geometry_validation"]["marker_rms_p90_m"]*1000)} | {nfmt(r["trial_marker_rms_p90_median_mm"])} | {r["empirical_sync_matched"]} | [报告]({r["subject"]}/README.md) |')
    lines+=['','## 一次执行','',
        '从项目根目录运行（默认原始500 Hz，不降采样）：','',
        '```powershell','python reconstruct_angles.py --workers 3','```','',
        '该命令自动标定、必要时重试有限优化预算、断点续算、导出及检查完整性。缓存核对程序兼容版本、模型哈希与处理选项；`--subjects 1` 可指定受试者，`--rate-hz` 可降低计算频率。','',
        '## 已完成的处理链','',
        '1. 从刚性分析选择逐受试者点组，保留骨段归属假设。','2. 从解剖胸廓系、功能性肘轴和G99几何初始化，在未知角度条件下联合优化角度、有限尺寸尺度和标记偏置。',
        '3. 在未参与几何拟合的重复试验上检查重投影、多初值和标记留出误差，再进行全序列有界鲁棒IK。','4. 使用原始标记观测进行以IK解为线性化点的运动学EKF，按逐角度质量导出，并核验EMG同步脉冲持续时间。','',
        '原始数据未修改；未引用/已剔除的MoCap记录不强制生成标签。','',
        '## 结果边界','',
        '- 不将肩胛皮肤点簇方向直接等同于已测得的骨坐标系；肩部1–8坐标及桡尺旋转绝对相位保留先验标记。多初值不稳定的其他角也会被标记。',
        '- 无先验标记只表示固定标定模型下的数值质量通过，不代表独立角度准确性。标定误差、皮肤滑移和躯干位姿误差未被完整纳入滤波协方差。',
        '- 刚性筛选用过各重复试验，几何拟合留出仅针对几何优化步骤；这不是完全未接触数据的最终评估集。',
        '- EMG第73列254/255低电平窗口通过时长与连续采样计数核验；通道语义/硬件延迟仍缺少独立文档确认。同步说明保留在各试验报告中。',
        '- `strict_valid` 按逐角度全部质量位为0生成。前臂旋转绝对角被标记时，不会强制把四个远端角一起认作可信监督真值。','',
        '## 文件和质量位','',
        '`ik_angles.npz/ekf_angles.npz`：`t`、`q[N,12]`、`q_distal[N,4]`、逐角度与逐帧质量位、位置/接触残差、观测奇异值；角度rad，时间s。滤波结果另含条件标准差。',
        '`emg_aligned_distal.npz`：EMG时间、远端角估计、逐角度质量、`strict_valid`、同步说明和原始EMG路径；不复制原始肌电幅值。',
        '质量位：1缺失，2IK未收敛，4位置RMS>20mm，8弱观测，16接触误差，32边界/裁剪，64绝对坐标依赖先验，128标定留出失败。','',
        '全量文件、时间轴、模型版本及质量标记检查见 [output_integrity.json](output_integrity.json)；该检查不验证外部角度准确性。',
        '程序：[自动标定](../../../src/motion_reconstruction/calibration.py)、[批量数学计算](../../../src/motion_reconstruction/kinematics.py)、[轨迹求解](../../../src/motion_reconstruction/filtering.py)、[EMG同步](../../../src/motion_reconstruction/emg_sync.py)、[执行入口](../../../reconstruct_angles.py)。']
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('subjects','missing')},indent=2))


if __name__=='__main__': main()
