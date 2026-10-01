# BRB Toolkit

[English](README.md) | **简体中文**

用于信念规则库建模、证据推理及参数学习的 Python 工具包。工具包包含可微 BRB 核心、以论文为依据的 RIMER 推理与约束训练，以及 MAKER 证据组合和学习方法。

## 概览

| 模块 | 用途 | 学习方法 |
|---|---|---|
| `brb` | 用于回归与分类的完整笛卡尔规则库 | PyTorch 梯度优化 |
| `brb.rimer` | 专家规则、不完整信念及层次 BRB | 数值、信念或混合监督下的约束优化 |
| `brb.maker` | 集合证据、可靠度及依赖组合 | 两阶段优先级约束 minimax 优化 |

BRB 是知识表示，RIMER 是针对该表示的推理方法。MAKER 通过状态相关权重、证据可靠度及显式依赖信息，描述证据获取与组合过程。

## 主要功能

- **BRB：** 数值与类别前件、笛卡尔规则、ER 聚合、回归与分类、数据驱动初始化、缺失输入规则边缘化、内部数值参考点自适应及可解释参数报告。
- **梯度学习：** 可学习信念、规则及属性权重；MSE、MAE、Huber、Poisson NLL、Gamma deviance、分类 NLL、Brier 目标；Adam、AdamW、SGD、L-BFGS 及验证集早停。
- **RIMER：** 显式专家规则、AND/OR 匹配、部分前件、原始或信念输入、完整与不完整 ER、效用区间、层次信念传播、参数冻结、上下界及专家约束。
- **MAKER：** 非空焦元、独立保留的模糊与不可信质量、计数及数据框证据获取、显式平滑、一般依赖组合、前缀递归、独立与互斥模式及两阶段 minimax 学习。
- **数据：** CSV、TSV/TXT、Excel、Parquet、JSON 读取，以及可微 BRB 核心的 YAML 配置。

## 安装

需要 **Python 3.10+**。在包含 `pyproject.toml` 的工具包根目录运行：

```bash
python -m venv .venv
```

激活环境：

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

安装完整工具包及论文示例依赖：

```bash
python -m pip install -r requirements-paper.txt
python -m pip install -e .
```

仅使用可微 BRB 核心时，执行 `python -m pip install -e .` 即可。RIMER/MAKER 约束训练需要 SciPy；完整依赖文件还包含 notebook 执行、绘图、测试及 PDF 工具。

## 快速开始

### 可微 BRB

```python
import torch
from brb import AttributeConfig, BRBConfig, BRBModel, Trainer
from brb import initialize_model_from_data

torch.manual_seed(7)
config = BRBConfig(
    target="y",
    attributes=[AttributeConfig("x", "numeric", [0.0, 0.5, 1.0])],
    consequent_refs=[0.0, 0.5, 1.0],
    task="regression",
)
model = BRBModel(config)
x = torch.tensor([[0.0], [0.25], [0.75], [1.0]], dtype=torch.float64)
y = torch.tensor([0.0, 0.2, 0.8, 1.0], dtype=torch.float64)
x_val = torch.tensor([[0.4], [0.6]], dtype=torch.float64)
y_val = torch.tensor([0.35, 0.65], dtype=torch.float64)
initialize_model_from_data(model, x, y, shrinkage=1.0)
Trainer(loss="mse", max_epochs=100, verbose=False).fit(model, x, y, x_val, y_val)
with torch.no_grad():
    prediction, beliefs = model(x)
```

处理表格数据时，可使用 `load_table`、`split_data` 和 `prepare_tensors`。由数据确定的参考点及初始化参数应仅使用训练集计算。[回归示例](examples/quickstart_regression.py) 包含训练、验证、测试及规则报告。

### RIMER

```python
import numpy as np
from brb.rimer import RIMERAttribute, RIMERConfig, RIMERModel
from brb.rimer import RIMERTrainer, RIMERSupervision

model = RIMERModel(RIMERConfig(
    attributes=[RIMERAttribute("x", [0.0, 1.0])],
    consequents=["Low", "High"],
    rule_index=None,
    rule_beliefs=[[0.9, 0.1], [0.1, 0.9]],
    rule_weights=[1.0, 1.0], attribute_weights=[1.0],
    utilities=[0.0, 1.0],
))
result = model.predict([[0.25], [0.75]])
fit = RIMERTrainer().fit(
    model, [np.eye(2)],
    RIMERSupervision(numeric=np.array([0.05, 0.95])),
    trainable={"beta": True, "theta": False, "delta": False},
)
print(fit.success, fit.constraint_violation)
```

`predict` 转换原始输入，`infer` 接受每个前件对应的信念矩阵。不完整输出保留 `ignorance` 和 `utility_interval`；单一数值决策需显式选择下界、上界或中点策略。`HierarchicalRIMER` 在节点之间直接传递信念分布。

### MAKER

```python
from brb.maker import StateSpace, Evidence, combine_pair

space = StateSpace(["H1", "H2"])
# 焦元顺序：H1、H2、{H1, H2}。
a = Evidence([0.7, 0.2, 0.1], [0.8, 0.7, 0.9], 0.85, space)
b = Evidence([0.3, 0.5, 0.2], [0.7, 0.9, 0.8], 0.90, space)
result = combine_pair(a, b, mode="independent")
print(result.probabilities)
print(result.ambiguity, result.untrustworthiness)
```

该示例明确假设证据独立。依赖型组合需要 `JointEvidence` 和联合支持信息。多变量依赖模型需要指定顺序下各前缀的联合获取信息，仅有两两相关系数不足以确定一般联合结构。

`EvidenceAcquisition` 从计数或数据框建立似然表。`MAKERTrainer` 通过两个按优先级求解的 minimax 目标学习权重与可靠度；第二阶段以显式容差保持第一阶段目标。

## 示例

[示例指南](examples/README.zh-CN.md) 介绍环境、运行方式、数据及复现范围：

1. [BRB 建模与训练](examples/notebooks/01_brb_modelling_paper.ipynb)：39 条规则的层次模型及十二个观测。
2. [RIMER 推理](examples/notebooks/02_rimer_inference_paper.ipynb)：完整与不完整输入及未知质量传播。
3. [MAKER 建模与训练](examples/notebooks/03_maker_paper.ipynb)：诊断证据、依赖及 21 参数约束学习。

在工具包根目录运行脚本示例：

```bash
python -m examples.quickstart_regression
python -m examples.rimer_training
python -m examples.maker_training
```

在新 kernel 中执行全部论文 notebook 并保存输出：

```bash
python scripts/execute_paper_notebooks.py
```

## 验证与限制

RIMER 石墨容器推理案例和 MAKER 诊断 Table 18 使用独立转录的纸面数值进行对照，并注明舍入容差。训练示例报告求解状态、目标值和约束残差。

MAKER 复现了论文推理表，但**论文报告训练目标的精确复现尚未解决**。将公开的舍入参数重新代入目标函数，所得目标值与正文报告不同；notebook 同时展示这些数值及本地 SLSQP 结果。局部优化不保证全局最优，也不保证返回与原求解器相同的参数。

运行代表性的核心与推理测试：

```bash
python -m pytest tests/test_core.py tests/test_rimer_inference.py tests/test_maker_combination.py -q
```

工具包面向研究及可复现实验。PyTorch 核心与约束训练器采用不同参数化及缺失输入语义：BRB 的规则边缘化与 RIMER 的显式不完整信念处理应分别理解。

规则及焦元空间会指数增长，RIMER 对生成规则数设限，MAKER 对焦元数设限。目前不提供自动规则结构搜索、通用图形编辑器或任意联合依赖结构的自动恢复。

## 目录结构

```text
brb/
  config.py, model.py, trainer.py   可微 BRB 核心
  data.py, initialization.py       数据准备与初始化
  explain.py, evaluate.py          参数报告与评估
  rimer/                          RIMER 推理及约束学习
  maker/                          MAKER 获取、组合及学习
  paper_common/                   共用验证及训练结果
examples/
  paper_cases/                    论文规则、计数及参考值
  notebooks/                      三个论文示例
scripts/                          执行及验证工具
tests/                            数学及训练检查
requirements.txt                  核心依赖
requirements-paper.txt            完整示例及约束训练依赖
```

## 参考文献

- Yang, J.-B., Liu, J., Wang, J., Sii, H.-S., and Wang, H.-W. (2006). *Belief Rule-Base Inference Methodology Using the Evidential Reasoning Approach—RIMER*. [DOI: 10.1109/TSMCA.2005.851270](https://doi.org/10.1109/TSMCA.2005.851270)。
- Yang, J.-B., Liu, J., Xu, D.-L., Wang, J., and Wang, H.-W. (2007). *Optimization Models for Training Belief-Rule-Based Systems*. [DOI: 10.1109/TSMCA.2007.897606](https://doi.org/10.1109/TSMCA.2007.897606)。
- Yang, J.-B., and Xu, D.-L. (2025). *Maximum Likelihood Evidential Reasoning*. [DOI: 10.1016/j.artint.2025.104289](https://doi.org/10.1016/j.artint.2025.104289)。

报告实验时，应说明使用的模块、目标、依赖假设及求解器设置，并引用相应方法论文。

公式映射与数值验证: [formula_map.md](docs/formula_map.md), [validation.md](docs/validation.md).
