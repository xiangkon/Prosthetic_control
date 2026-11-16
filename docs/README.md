# 数据准备工作区

当前流程从开源数据集读取三维标记，恢复上肢12个关节坐标，并保存逐角质量和肌电时间对齐结果。

## 两个终端入口

在项目根目录运行。

```powershell
python -m pip install -r requirements.txt
python reconstruct_angles.py
python visualize_reconstruction.py
```

重建默认处理TUM全部6位受试者、532条有效运动试验，复用匹配的模型与结果。候选点组缺失时自动分析。只处理第1位可使用 `--subjects 1`，通过 `--root`、`--audit` 和 `--output` 指定其他数据、候选组及结果路径。

```powershell
python reconstruct_angles.py --self-check
python visualize_reconstruction.py --self-check
python visualize_reconstruction.py --results outputs/TUM/reconstruction/subj_002/trials/trial_011
```

## 目录

| 路径 | 内容 |
|---|---|
| `datasets/` | 下载的原始数据，TUM输入为 `datasets/TUM/TUM` |
| `src/motion_reconstruction/` | 数据读取、点组筛选、标定、运动学、滤波、质量、同步、批处理与界面 |
| `configs/motion_reconstruction/garner1999_tables.json` | G99原始几何参数及来源 |
| `outputs/TUM/marker_analysis/` | 每位受试者的必要候选点组JSON |
| `outputs/TUM/reconstruction/` | 最终模型、IK/滤波角度、同步标签及报告 |
| `references/` | 建立模型使用的原论文 |
| `docs/` | 数据分析结论、实施指南、查看器说明和研究记录 |

每条试验输出 `ik_angles.npz`、`ekf_angles.npz` 和 `report.json`，同步成功时另有 `emg_aligned_distal.npz`。角度单位rad，时间单位s。原始标记单位按官方mm说明转换为m。

结果是固定模型与工程先验下的估计，质量位和同步证据随标签保存。制作训练数据时，需确定肌电预处理、逐角质量筛选、窗口与受试者级划分。

## 详细说明

- [实际重建流程和参数](TUM_Hu2018_12DoF_Inverse_Kinematics_Reproduction_Guide_v2.md)
- [重建可视化操作](RECONSTRUCTION_VIEWER.md)
- [数据集结构与质量](dataset_analysis.md)
- [TUM标记名称与筛选结论](TUM_marker_analysis.md)
- [TUM试验与动作对应](TUM_trial_tasks.md)
