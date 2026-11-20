# 假肢控制数据准备工作区

当前流程为“原始数据 → 运动重建 → sEMG预处理与对齐 → 训练数据准备”。老方法输出12个关节坐标，新方法输出7个功能性旋转坐标；角度均为模型估计，须结合逐角质量、同步证据和方法限制使用。

## 安装与运行约定

```powershell
python -m pip install -r requirements.txt
```

以下命令默认在项目根目录运行。入口按自身位置定位项目根目录，也可从其他目录通过脚本绝对路径启动；默认数据和结果路径仍相对项目根目录，显式传入的相对路径相对当前工作目录。完整参数使用对应入口的 `--help` 查看。

## 运动重建

| 方法 | 主要入口 | 正式结果（位于outputs/TUM/reconstruction/） |
|---|---|---|
| 老方法12坐标 | `scripts/reconstruct_angles.py` | `legacy_g99/` |
| 新方法v002姿态约束 | `scripts/reconstruct_tum_native.py` | `tum_native/v002_anchored7/` |

输入为 `datasets/TUM/TUM/`。老方法532条正式结果为冻结基线，重跑必须指定独立 `--output`；先做自检和小范围验证。新方法须显式选择试验或 `--all`，输出冲突时使用独立 `--output-root`。新方法仅维护v002，不支持 `--replace`，冲突时须另选输出目录。

```powershell
# 老方法自检，不计算真实试验
python scripts/reconstruct_angles.py --self-check
# 老方法：独立目录，仅处理第1人（当前入口不支持按试验选择）
python scripts/reconstruct_angles.py --algorithm legacy_g99 --subjects 1 --output outputs/TUM/_checks/legacy_g99_reruns/experiment_001
# 新方法v002：小范围处理
python scripts/reconstruct_tum_native.py --subjects 1 --trials 43 45 --workers 2 --emg
# 新方法v002：明确选择全量处理
python scripts/reconstruct_tum_native.py --all --workers 2 --emg
```

老方法输出 `ik_angles.npz`、`ekf_angles.npz`、`report.json`，同步成功时另有 `emg_aligned_distal.npz`；新方法输出 `reconstruction.npz`、`report.json`，可用时另有 `emg_aligned_native.npz`。内部单位为m、rad、s，原始标记按官方mm说明转换为m。

配置分别位于 `configs/motion_reconstruction/legacy_g99/` 和 `configs/motion_reconstruction/tum_native/v002.json`。模型、参数、质量及版本差异见[老方法运动重建](docs/老方法运动重建.md)和[新方法运动重建](docs/新方法运动重建.md)。比较规则与数据选择证据见[新老方法比较](docs/新老方法比较.md)。

## sEMG预处理

默认版本为 `v002_hu2018_with_rest`，导出十个主要肘、前臂和腕肌肉通道的Task 0放松姿态与Task 1–30运动记录；归一化峰值只由每人的运动记录计算。v001继续保留。

```powershell
python scripts/preprocess_emg.py --self-check
# 小范围验证：仅所选试验参与归一化，不能当作正式全量结果
python scripts/preprocess_emg.py --subjects 1 --limit-trials 2 --output outputs/TUM/_checks/emg_preprocessed
# 正式处理全部有效EMG任务记录
python scripts/preprocess_emg.py
# 显式选择旧版本
python scripts/preprocess_emg.py --config configs/emg_preprocessing/v001_hu2018.json
```

输入为TUM原始EMG与任务表，配置在 `configs/emg_preprocessing/`，结果固定保存到 `outputs/TUM/emg_preprocessed/<版本>/`，下接受试者和试验目录，不增加时间戳层；替换时保留编号备份。滤波、通道映射、输出字段和归一化泄漏边界见[肌电预处理](docs/肌电预处理.md)。

## 可视化

四个日常入口统一位于 `scripts/visualize/`，读取现成结果，无须重建。统一重建查看入口已覆盖新方法，因此删除重复的 `visualize_tum_native.py` 快捷入口；原位置参数、`--frame` 和 `--save` 可在统一入口加 `--algorithm tum_native` 后使用。

```powershell
# 老方法：标记点、骨架和角度
python scripts/visualize/visualize_reconstruction.py --algorithm legacy_g99
# 新方法：指定版本与试验
python scripts/visualize/visualize_reconstruction.py --algorithm tum_native --results outputs/TUM/reconstruction/tum_native/v002_anchored7/subj_001/trials/trial_043
# 新老方法对照
python scripts/visualize/visualize_reconstruction_comparison.py --subject 1 --trial 43 --method ekf
# 肌电波形、包络、热图与频谱
python scripts/visualize/visualize_emg.py --data outputs/TUM/emg_preprocessed/v002_hu2018_with_rest/subj_001/trial_001/emg_preprocessed.npz
# 新方法v002运动与肌电同步查看（默认）
python scripts/visualize/visualize_emg_kinematics.py
# 切换老方法
python scripts/visualize/visualize_emg_kinematics.py --algorithm legacy_g99
```

肌电及同步查看器默认示例仍使用 `v001_hu2018/subj_001/trial_005`；可在界面切换版本和试验，或用 `--data` 指定。同步查看器默认使用v002，界面“运动重建”可切换新老方法；肌电版本独立选择。操作、静态导出、自检与帧号约定见[可视化使用说明](docs/可视化使用说明.md)。

## 分析工具与归档

`scripts/analysis/` 保存仍可复用的统计、转换、审计和报告入口；`scripts/archive/` 保存老方法单条试验修复工具，不作为日常处理入口。归档工具可能写入清单或正式结果，使用前须确认具体任务范围；本次整理未执行这些工具。

```powershell
# 仅读取已有结果；默认目标已存在时拒绝覆盖，重跑须另指定独立--output
python scripts/analysis/analyze_reconstruction_trends.py --help
python scripts/analysis/rank_reconstruction_consistency.py --help
# 老方法角度统计沿用重建入口的分析模式
python scripts/reconstruct_angles.py --angle-statistics
```

分析运行参数见[新老方法比较](docs/新老方法比较.md#v002趋势分析流程)，前臂审计见[老方法运动重建](docs/老方法运动重建.md#前臂旋转质量检查)。其他分析入口包括 `audit_pronation.py`、`screen_pronation.py`。

## 目录与结果管理

| 路径 | 用途 |
|---|---|
| `datasets/`、`references/` | 原始数据与文献，保持原样 |
| `scripts/` | 新老方法重建入口与肌电预处理入口 |
| `scripts/visualize/` | 四个日常可视化入口 |
| `scripts/analysis/` | 可复用分析、统计、审计和报告工具 |
| `scripts/archive/` | 历史专项任务脚本 |
| `src/`、`configs/` | 算法实现与版本参数，重建方法分开管理 |
| `docs/` | 按主题持续维护的中文说明与核心结论 |
| `outputs/TUM/reconstruction/` | 老方法、新方法及独立比较结果 |
| `outputs/TUM/emg_preprocessed/` | 按方法版本保存的肌电结果 |
| `outputs/TUM/analysis/` | 数据集层面的独立图表 |
| `outputs/TUM/_checks/` | 数据集层面的试跑与整理快照 |

各重建方法的分析、验证和备份保存在各方法结果目录内部。逐次运行报告、CSV和图片保存在 `outputs/`，`docs/` 维护稳定说明和核心结论。输出导航见[运动重建输出目录](outputs/TUM/reconstruction/README.md)。

## 中文文档导航

- [数据集说明](docs/数据集说明.md)：MHH/TUM结构、配对、标记与质量。
- [TUM试验任务对照](docs/TUM试验任务对照.md)：受试者、试验与动作查表。
- [老方法运动重建](docs/老方法运动重建.md)：12坐标模型、实现与前臂质量检查。
- [新方法运动重建](docs/新方法运动重建.md)：v002模型、质量、同步与运行说明。
- [新老方法比较](docs/新老方法比较.md#v002全量对比核心结果与数据选择)：532条配对趋势、受试者／动作排名、异常和选择限制。
- [肌电预处理](docs/肌电预处理.md)：版本、滤波、归一化、输出和训练边界。
- [可视化使用说明](docs/可视化使用说明.md)：四个查看器的操作与导出。
- [研究报告](docs/研究报告.md)：保留现有内容与文件名。

训练集制作前须确定逐角质量、时间对齐、覆盖时长和活动范围，按受试者／任务／重复冻结划分；归一化等拟合参数只使用训练或独立校准数据。跨方法高相关不能替代独立角度精度验证。
