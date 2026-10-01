# Examples and paper demos

**English** | [简体中文](README.zh-CN.md) | [Toolkit overview](../README.md)

This directory contains a synthetic BRB regression example, an insurance workflow template, and three notebooks illustrating paper-based modelling, inference, and learning.

## Setup

Activate a Python 3.10+ environment and run from the toolkit root:

```bash
python -m pip install -r requirements-paper.txt
python -m pip install -e .
```

Notebooks use local data and do not download papers or install packages during execution. Select a kernel with these dependencies installed. Saved outputs are included; rerunning computes fresh results. Explanatory notebook text is in Chinese; API names and mathematical notation retain their original form.

## Recommended order

| Notebook | Focus | Reference |
|---|---|---|
| [BRB modelling and training](notebooks/01_brb_modelling_paper.ipynb) | Expert rules, hierarchical inference, and constrained learning | Yang et al. (2007); 39 rules and 12 observations |
| [RIMER inference](notebooks/02_rimer_inference_paper.ipynb) | Matching, activation, ER, and ignorance propagation | Yang et al. (2006); 36-rule graphite-container system |
| [MAKER modelling and training](notebooks/03_maker_paper.ipynb) | Evidence acquisition, dependence, reliability, and two-stage learning | Yang and Xu (2025); 6,300 grouped records and 21 parameters |

Each notebook has ten steps with computed/reference comparisons and executable checks.

### BRB modelling and training

Constructs the hierarchical rule base using the 2007 paper's Expert 1 initialisation, predicts twelve observations, and jointly trains beliefs, weights, and ordered utilities. A complete single-node example also demonstrates the differentiable BRB API.

The main procedure uses constrained SLSQP. Outputs include before/after MSE, solver status, constraint residuals, learned utilities, and a prediction plot. Table II observations are labelled `paper_reference`; local fitted parameters are not presented as the original paper's optimisation output.

### RIMER inference

Follows the 2006 paper's example, showing input beliefs, activated rules, intermediate nodes, and complete/incomplete conclusions retaining `Unknown` mass. The 36-rule model differs from the 2007 training model.

Rule 13 follows Table II where the running text gives a different belief. Intermediate and incomplete-output comparisons use absolute tolerance `5e-5`; the complete final conclusion uses `1e-4` for accumulated printed rounding. This is an inference demonstration, not a training experiment.

### MAKER modelling and training

Reconstructs likelihood evidence from grouped counts, computes dependence indices, reproduces Table 18 using published parameters, and trains 21 parameters through two successive minimax objectives.

- Acquisition and training use different count groups. Training uses 2,100 saliva-only, 3,100 blood-only, and 1,100 joint-test records.
- Table 18 excludes the prior; the training objective includes independent prior evidence.
- Stage two retains the primary objective within `1e-5` of the first-stage result.
- Ambiguity on `{H1,H2}` and untrustworthiness remain separate.

**Reproduction scope:** Table 18 probabilities and trustworthiness agree within `1.5e-4`. Exact reproduction of reported training objectives remains unresolved: evaluating published rounded parameters with the stated equations gives different values. The notebook compares reported objectives, recomputed objectives, and local SLSQP results. Fitted parameters do not establish global optimality.

## Execution

Open a notebook in a Jupyter-compatible editor and run all cells using the configured environment. Keep notebooks within the repository so they can locate the toolkit.

Execute all notebooks from the toolkit root:

```bash
python scripts/execute_paper_notebooks.py
```

Execute one notebook:

```bash
python scripts/execute_paper_notebooks.py examples/notebooks/03_maker_paper.ipynb
```

The runner creates fresh kernels, saves outputs, and stops on errors. BRB training may take several minutes, depending on the machine.

## Script examples

```bash
python -m examples.quickstart_regression
python -m examples.rimer_training
python -m examples.maker_training
```

| File | Behaviour |
|---|---|
| [quickstart_regression.py](quickstart_regression.py) | Generates synthetic data with missing antecedents, splits data, trains a differentiable BRB, and reports test metrics and rules |
| [rimer_training.py](rimer_training.py) | Runs hierarchical training with twelve observations; prints observed, initial, and fitted outputs |
| [maker_training.py](maker_training.py) | Runs two-stage diagnosis learning and reports objectives, solver status, and inference comparisons |
| [insurance_frequency_severity_skeleton.py](insurance_frequency_severity_skeleton.py) | Functions for separate frequency/severity models; requires user-supplied configurations and data, rather than running as a standalone dataset demo |

Run these from the toolkit root. RIMER/MAKER training scripts also save parameter/result records under `outputs/`.

## Case data

The [paper_cases/](paper_cases/) directory contains rules, counts, parameters, and independently transcribed reference results:

- [graphite.json](paper_cases/graphite.json): 2006 inference rules, observations, and reference conclusions.
- [graphite_training.json](paper_cases/graphite_training.json): 2007 Expert 1 initialisation.
- [diagnosis.json](paper_cases/diagnosis.json): diagnosis counts and published weights, reliabilities, and reference results.
- `graphite.py` and `diagnosis.py`: model builders used by the notebooks and scripts.

The diagnosis data is an illustrative aggregated example from the paper, not a clinical dataset collected by this project. These demos verify the illustrated computational paths, not performance on unseen datasets.

## References

1. Yang et al. (2006), *Belief Rule-Base Inference Methodology Using the Evidential Reasoning Approach—RIMER*. [DOI](https://doi.org/10.1109/TSMCA.2005.851270).
2. Yang et al. (2007), *Optimization Models for Training Belief-Rule-Based Systems*. [DOI](https://doi.org/10.1109/TSMCA.2007.897606).
3. Yang and Xu (2025), *Maximum Likelihood Evidential Reasoning*. [DOI](https://doi.org/10.1016/j.artint.2025.104289).
