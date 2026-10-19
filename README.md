# BRB Toolkit

A Python toolkit for belief rule base modelling, evidential reasoning, and parameter learning. It provides a differentiable BRB core, paper-based RIMER inference and constrained training, and MAKER evidence combination and learning.

## Overview

| Module | Purpose | Learning method |
|---|---|---|
| `brb` | Complete Cartesian BRBs for regression and classification | PyTorch gradient optimisation |
| `brb.rimer` | Expert rules, incomplete beliefs, and hierarchical BRBs | Constrained numerical, belief, or mixed-supervision optimisation |
| `brb.maker` | Set-valued evidence, reliability, and dependent combination | Two-stage priority-constrained minimax optimisation |

BRB is a knowledge representation; RIMER is an inference methodology for that representation. MAKER models evidence acquisition and combination through state-specific weights, evidence reliability, and explicit dependence information.

## Features

- **BRB:** numeric/categorical antecedents, Cartesian rules, ER aggregation, regression/classification, data-driven initialisation, missing-input rule marginalisation, adaptive internal numeric reference points, and interpretable parameter reports.
- **Gradient learning:** trainable beliefs and rule/attribute weights; MSE, MAE, Huber, Poisson NLL, Gamma deviance, classification NLL, and Brier objectives; Adam, AdamW, SGD, L-BFGS, and validation early stopping.
- **RIMER:** explicit expert rules, AND/OR matching, partial antecedents, raw or belief inputs, complete/incomplete ER, utility intervals, hierarchical belief propagation, parameter freezing, bounds, and expert constraints.
- **MAKER:** nonempty focal sets, separate ambiguity and untrustworthiness, count/data-frame acquisition, explicit smoothing, general dependent combination, ordered prefix recursion, independent/exclusive modes, and two-stage minimax learning.
- **Data:** CSV, TSV/TXT, Excel, Parquet, and JSON loading; YAML configuration for the differentiable BRB core.

## Installation

Requires **Python 3.10+**. Run commands from the toolkit root, containing `pyproject.toml`:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install the full toolkit and paper-demo dependencies:

```bash
python -m pip install -r requirements-paper.txt
python -m pip install -e .
```

For the differentiable BRB core alone, `python -m pip install -e .` is sufficient. Constrained RIMER/MAKER training requires SciPy. The full dependency file also includes notebook execution, plotting, testing, and PDF utilities.

## Quick start

### Differentiable BRB

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

For tabular data, use `load_table`, `split_data`, and `prepare_tensors`. Determine data-derived reference points and initialise the model using the training split only. The [regression example](examples/quickstart_regression.py) includes training, validation, testing, and rule reports.

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

`predict` transforms raw inputs; `infer` accepts one belief matrix per antecedent. Incomplete results retain `ignorance` and `utility_interval`. Scalar decisions require an explicit lower, upper, or midpoint policy. `HierarchicalRIMER` passes belief distributions directly between nodes.

### MAKER

```python
from brb.maker import StateSpace, Evidence, combine_pair

space = StateSpace(["H1", "H2"])
# Focal-set order: H1, H2, {H1, H2}.
a = Evidence([0.7, 0.2, 0.1], [0.8, 0.7, 0.9], 0.85, space)
b = Evidence([0.3, 0.5, 0.2], [0.7, 0.9, 0.8], 0.90, space)
result = combine_pair(a, b, mode="independent")
print(result.probabilities)
print(result.ambiguity, result.untrustworthiness)
```

This example explicitly assumes independence. Dependent combination requires `JointEvidence` and joint support information. Multi-variable dependent models require the joint acquisitions for each ordered prefix; pairwise correlations alone are insufficient.

`EvidenceAcquisition` builds likelihood tables from counts or data frames. `MAKERTrainer` learns weights and reliabilities through two successive minimax objectives, maintaining the first-stage objective within an explicit tolerance during the second stage.

## Demos

The [demo guide](examples/README.md) covers setup, execution, data, and reproduction scope:

1. [BRB modelling and training](examples/notebooks/01_brb_modelling_paper.ipynb): 39-rule hierarchy and twelve observations.
2. [RIMER inference](examples/notebooks/02_rimer_inference_paper.ipynb): complete/incomplete inputs and ignorance propagation.
3. [MAKER modelling and training](examples/notebooks/03_maker_paper.ipynb): diagnosis evidence, dependence, and 21-parameter constrained learning.

Run script examples from the toolkit root:

```bash
python -m examples.quickstart_regression
python -m examples.rimer_training
python -m examples.maker_training
```

Execute all paper notebooks in fresh kernels and save outputs:

```bash
python scripts/execute_paper_notebooks.py
```


## Project layout

```text
brb/
  config.py, model.py, trainer.py   Differentiable BRB core
  data.py, initialization.py       Data preparation and initialisation
  explain.py, evaluate.py          Reports and evaluation
  rimer/                          RIMER inference and constrained learning
  maker/                          MAKER acquisition, combination, and learning
  paper_common/                   Shared validation and training results
examples/
  paper_cases/                    Paper rules, counts, and reference values
  notebooks/                      Three paper demonstrations
scripts/                          Execution and verification tools
tests/                            Mathematical and training checks
requirements.txt                  Core dependencies
requirements-paper.txt            Full demo and constrained-training dependencies
```

## References

- Yang, J.-B., Liu, J., Wang, J., Sii, H.-S., and Wang, H.-W. (2006). *Belief Rule-Base Inference Methodology Using the Evidential Reasoning Approach—RIMER*. [DOI: 10.1109/TSMCA.2005.851270](https://doi.org/10.1109/TSMCA.2005.851270).
- Yang, J.-B., Liu, J., Xu, D.-L., Wang, J., and Wang, H.-W. (2007). *Optimization Models for Training Belief-Rule-Based Systems*. [DOI: 10.1109/TSMCA.2007.897606](https://doi.org/10.1109/TSMCA.2007.897606).
- Yang, J.-B., and Xu, D.-L. (2025). *Maximum Likelihood Evidential Reasoning*. [DOI: 10.1016/j.artint.2025.104289](https://doi.org/10.1016/j.artint.2025.104289).

When reporting experiments, identify the module, objective, dependence assumptions, and solver settings, and cite the relevant methodological papers.

Formula mapping and numerical validation: [formula_map.md](docs/formula_map.md)

## Development Status

This toolkit is **under active development**. The implementations of BRB, RIMER, and MAKER, including PyTorch-based CPU and GPU training, are currently being reviewed, tested, and refined. The repository will be updated regularly as improvements are made.

We also plan to add one or two end-to-end examples using publicly available datasets, covering data preprocessing, model configuration, training, testing, and evaluation. These examples will be included in upcoming updates to demonstrate the practical use of the toolkit.

Please note that the current release is preliminary, and further improvements and documentation updates are ongoing.

## License

This project is released under the [MIT License](LICENSE).
