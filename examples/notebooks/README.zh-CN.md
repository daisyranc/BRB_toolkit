# 论文 notebook

[English](README.md) | **简体中文**

环境准备、执行方式、论文来源及复现范围见[示例指南](../README.zh-CN.md)。

1. [BRB 建模与训练](01_brb_modelling_paper.ipynb)：39 条规则的层次模型及十二个观测。
2. [RIMER 推理](02_rimer_inference_paper.ipynb)：完整与不完整输入及未知质量传播。
3. [MAKER 建模与训练](03_maker_paper.ipynb)：依赖诊断证据及两阶段学习。

在工具包根目录运行 `python scripts/execute_paper_notebooks.py`，可在新 kernel 中执行全部 notebook 并保存输出。交互运行时，请选择已安装 `requirements-paper.txt` 依赖的环境。
