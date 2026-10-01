# 示例与论文 demo

[English](README.md) | **简体中文** | [工具包概览](../README.zh-CN.md)

本目录包含合成数据 BRB 回归示例、保险频率—强度工作流模板，以及三个介绍论文建模、推理和学习方法的 notebook。

## 环境准备

激活 Python 3.10+ 环境，在工具包根目录运行：

```bash
python -m pip install -r requirements-paper.txt
python -m pip install -e .
```

Notebook 使用本地数据，执行时不下载论文、不安装软件包。请选择已安装上述依赖的 kernel。文件包含已保存输出，重新运行会计算新的结果。Notebook 使用中文说明，API 名称及数学记号保留原有形式。

## 建议顺序

| Notebook | 重点 | 论文依据 |
|---|---|---|
| [BRB 建模与训练](notebooks/01_brb_modelling_paper.ipynb) | 专家规则、层次推理及约束学习 | Yang 等（2007）；39 条规则、12 个观测 |
| [RIMER 推理](notebooks/02_rimer_inference_paper.ipynb) | 匹配、激活、ER 及未知质量传播 | Yang 等（2006）；36 条规则的石墨容器系统 |
| [MAKER 建模与训练](notebooks/03_maker_paper.ipynb) | 证据获取、依赖、可靠度及两阶段学习 | Yang 与 Xu（2025）；6300 条聚合记录、21 个参数 |

每份 notebook 包含十个步骤，提供计算值与参考值对照及可执行检查。

### BRB 建模与训练

使用 2007 年论文的 Expert 1 初始化构建层次规则库，预测十二个观测，并联合学习信念、权重及有序效用。一个完整单节点示例同时展示可微 BRB 接口。

主要流程采用约束 SLSQP，输出训练前后 MSE、求解状态、约束残差、学习后的效用及预测图。Table II 观测标记为 `paper_reference`，不将本地拟合参数描述为原论文求解器的输出。

### RIMER 推理

沿用 2006 年论文案例，显示输入信念、激活规则、中间节点，以及保留 `Unknown` 的完整与不完整结论。36 条规则的模型与 2007 年训练模型不同。

论文正文与 Table II 对第 13 条规则的信念描述不一致，转录采用 Table II。中间节点与不完整输出使用绝对容差 `5e-5`；完整最终结论使用 `1e-4`，以反映纸面数值的累积舍入。这是推理演示，不是训练实验。

### MAKER 建模与训练

从聚合计数重建似然证据，计算依赖指数，用论文公开参数复现 Table 18，再通过两个按优先级求解的 minimax 目标训练 21 个参数。

- 证据获取及训练采用不同计数组。训练使用 2100 条仅唾液检测、3100 条仅血液检测和 1100 条联合检测记录。
- Table 18 不含先验；训练目标包含独立先验证据。
- 第二阶段要求主目标不超过第一阶段结果加 `1e-5`。
- `{H1,H2}` 上的模糊质量与不可信质量分别保留。

**复现范围：** Table 18 概率及可信度在 `1.5e-4` 容差内吻合。论文报告训练目标的精确复现尚未解决：公开舍入参数代入正文公式后得到不同数值。Notebook 比较正文报告、参数重算及本地 SLSQP 结果；拟合参数不构成全局最优结论。

## 执行方式

在支持 Jupyter 的编辑器中打开 notebook，选择对应环境并运行全部单元格。请保留 notebook 在仓库中运行，以便定位工具包。

在工具包根目录执行全部 notebook：

```bash
python scripts/execute_paper_notebooks.py
```

执行单份 notebook：

```bash
python scripts/execute_paper_notebooks.py examples/notebooks/03_maker_paper.ipynb
```

执行器创建新 kernel、保存输出，遇到错误即停止。BRB 训练耗时随机器变化，可能需要几分钟。

## 脚本示例

```bash
python -m examples.quickstart_regression
python -m examples.rimer_training
python -m examples.maker_training
```

| 文件 | 功能 |
|---|---|
| [quickstart_regression.py](quickstart_regression.py) | 生成含缺失前件的合成数据，划分数据、训练可微 BRB，并报告测试指标及规则 |
| [rimer_training.py](rimer_training.py) | 运行十二个观测的层次训练案例，显示观测、初始及拟合输出 |
| [maker_training.py](maker_training.py) | 运行诊断案例两阶段学习，报告目标、求解状态及推理对照 |
| [insurance_frequency_severity_skeleton.py](insurance_frequency_severity_skeleton.py) | 提供分别训练频率与强度模型的函数；需自行提供配置及数据，不是可直接运行的数据集 demo |

上述命令在工具包根目录运行。RIMER/MAKER 脚本还会将参数及结果记录保存至 `outputs/`。

## 案例数据

[paper_cases/](paper_cases/) 保存规则、计数、参数及独立转录的参考结果：

- [graphite.json](paper_cases/graphite.json)：2006 年推理规则、观测及参考结论。
- [graphite_training.json](paper_cases/graphite_training.json)：2007 年 Expert 1 初始化。
- [diagnosis.json](paper_cases/diagnosis.json)：诊断计数及论文公开的权重、可靠度、参考结果。
- `graphite.py` 和 `diagnosis.py`：notebook 及脚本使用的模型构建函数。

诊断数据是论文中的示意性聚合案例，并非本项目收集的临床数据。这些 demo 验证所展示的计算路径，不据此推断未见数据集上的性能。

## 参考文献

1. Yang 等（2006），*Belief Rule-Base Inference Methodology Using the Evidential Reasoning Approach—RIMER*。[DOI](https://doi.org/10.1109/TSMCA.2005.851270)。
2. Yang 等（2007），*Optimization Models for Training Belief-Rule-Based Systems*。[DOI](https://doi.org/10.1109/TSMCA.2007.897606)。
3. Yang 与 Xu（2025），*Maximum Likelihood Evidential Reasoning*。[DOI](https://doi.org/10.1016/j.artint.2025.104289)。
