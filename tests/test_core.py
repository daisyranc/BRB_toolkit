import math

import torch

from brb.config import AttributeConfig, BRBConfig
from brb.er import er_aggregate
from brb.model import BRBModel
from brb.transform import numeric_matching, scalar_to_belief


def test_numeric_matching():
    """Check linear interpolation. / 检查线性插值匹配。"""
    x = torch.tensor([35.0], dtype=torch.float64)
    refs = torch.tensor([20.0, 40.0, 60.0], dtype=torch.float64)
    m = numeric_matching(x, refs)
    assert torch.allclose(m, torch.tensor([[0.25, 0.75, 0.0]], dtype=torch.float64))


def test_scalar_to_belief():
    """Check expected-value preserving consequent interpolation. / 检查保持期望值的后件插值。"""
    beta = scalar_to_belief(0.35, [0.0, 0.2, 0.5, 1.0])
    value = float((beta * torch.tensor([0.0, 0.2, 0.5, 1.0], dtype=torch.float64)).sum())
    assert math.isclose(value, 0.35, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(float(beta.sum()), 1.0, rel_tol=1e-12)


def test_er_single_rule():
    """One fully activated rule should return its own belief. / 单条完全激活规则应返回其自身信念分布。"""
    activation = torch.tensor([[1.0]], dtype=torch.float64)
    beliefs = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    out = er_aggregate(activation, beliefs)
    assert torch.allclose(out, beliefs)


def test_model_forward_and_missing():
    """Check forward inference and missing antecedent handling. / 检查前向推理和缺失前件处理。"""
    config = BRBConfig(
        target="y",
        task="regression",
        attributes=[
            AttributeConfig("x1", "numeric", [0.0, 0.5, 1.0], adaptive=True),
            AttributeConfig("x2", "numeric", [0.0, 1.0], adaptive=False),
        ],
        consequent_refs=[0.0, 0.5, 1.0],
    )
    model = BRBModel(config)
    x = torch.tensor([[0.25, 0.7], [float("nan"), 0.3]], dtype=torch.float64)
    pred, belief = model(x)
    assert pred.shape == (2,)
    assert belief.shape == (2, 3)
    assert torch.all(torch.isfinite(pred))
    assert torch.allclose(belief.sum(dim=1), torch.ones(2, dtype=torch.float64), atol=1e-10)
