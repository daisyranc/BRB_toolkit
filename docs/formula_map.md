# Formula-to-implementation map

[中文](formula_map.zh-CN.md) · [Documentation index](README.md)

R06, R07 and M25 denote the papers listed in the [toolkit references](../README.md). RIMER modules are under `brb/rimer/`; MAKER modules are under `brb/maker/`. Case data and adapters are under `examples/paper_cases/`.

| Source | Formula or assumption | Implementation | Validation coverage |
|---|---|---|---|
| R06 §IV | (16a,b), numeric interpolation and explicit belief inputs | `rimer/transform.py` | Endpoints, interpolation and missing inputs |
| R06 §III | (6a) power-product AND; (6b) weighted product-sum OR; (7) activation | `rimer/inference.py`: `matching_and_activation` | Hand-calculated weights and participation masks |
| R06 §III | (8), mean completeness of participating antecedents updates beta | `rimer/inference.py`, `rimer/model.py` | Incomplete rules/inputs and X6/X9 Unknown |
| R06 §III | (9–11), recursive ER masses distinguish input ignorance from rule-weight discounting | `rimer/inference.py`: `er_aggregate` | Incomplete single rule and conflicting complete rules |
| R06 §III | (12a–c), utility bounds | `rimer/model.py`, `rimer/results.py` | Explicit decisions for incomplete outputs |
| R06 §IV–V | Direct belief propagation between layers | `rimer/hierarchy.py`, `paper_cases/graphite.py` | 36-rule intermediate node, explicit mapping and cycle detection |
| R07 §III | (12) numerical MSE; (15,16,19) consequent-wise and mixed minimax | `rimer/objectives.py`, `rimer/trainer.py` | Numerical, subjective and mixed targets; frozen/expert constraints |
| R07 §V | Table II observations; Table IV Expert 1 initialization | `paper_cases/graphite_training.json` | 39 rules, 12 observations, complete-row equalities and ordered utilities |
| M25 §2–3 | Normalized state-conditional likelihoods; nonempty-set focal elements | `maker/evidence.py`, `maker/state_space.py` | Tables 8–10; counts distinguished from posteriors |
| Input adapter | Univariate/prefix counts from available DataFrame rows; explicit smoothing | `maker/acquisition.py` | Missing patterns, discretization, zero support and unknown states |
| M25 §3 | (21,23), projection=1/(w·p+1−r), m=w·p | `maker/evidence.py`: `Evidence` | Separate storage of untrustworthy residual mass |
| M25 §4 | (49), alpha_AB=joint_AB/(p_A·p_B) | `maker/combination.py`: `dependency_indices` | Table 11 first entry: 1.7018 |
| M25 §4 | (51,54), joint-weight and projection exponents | `maker/combination.py`: `_dependent_products` | All-one coefficients reduce to independence |
| M25 Theorem 1 | (59–61), bounded sum and exponent-weighted intersection sum | `maker/combination.py`: `combine_pair` | Tables 16–18 probabilities and trustworthiness |
| M25 Corollaries 1.1/1.3 | (62,66), exclusivity and differential limits | `combine_pair`: `exclusive`, `derivative_rates` | Hand calculation with unreliability growth rates (1,2) |
| M25 Corollaries 1.4/1.5 | (67), independence and state-constant ER weights | `combine_pair`: `independent` | Set intersections and singleton Bayes products |
| M25 Corollary 1.6 | (68), augmented Dempster combination with tau/delta | `combine_pair`: `augment` | Ordinary Dempster and differential limits under total conflict |
| M25 Lemma 2.1 | (70–72), prefix ERPS recursion and prefix acquisition exponents | `maker/combination.py`: `extend_result`; `maker/model.py` | Three-variable prefix tables and independent recursion |
| M25 §5 | (86–90), frequency/total × half squared deviation | `maker/objectives.py`: `paper_objectives` | Independently calculated observation terms |
| M25 §5 | (91–95), maximum of count-weighted ESD and reliability/reference deviations | `maker/objectives.py` | Frozen references and hand-calculated weight deviations |
| M25 §5 | (97–100), two-stage prioritized minimax | `maker/trainer.py` | Stage-one failure stops; stage-two failure rolls back; primary objective bound |
| M25 §6 | 21-parameter diagnosis, separate/joint training groups and independent prior | `paper_cases/diagnosis.py`, `diagnosis.json` | 6,300 counts; Table 18 excludes prior, training includes prior |

OR expands to `1 − product(1 − normalized_delta × matching)`, equivalent to the recursion in (6b). Fully unknown outputs without activation, endpoint clipping outside input ranges, errors for unseen inputs, resource limits and SLSQP trial points outside the constraint surface are implementation policies.

Zero marginal support with positive joint support, or undefined weight exponents, raises an error by default. Smoothing cannot supply missing joint information. Independence and exclusivity are explicit user choices. Bayes/Dempster reductions apply only when their stated assumptions hold.

