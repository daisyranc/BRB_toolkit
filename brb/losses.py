from __future__ import annotations

from collections.abc import Callable

import torch


LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


def _weighted_mean(values: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    """Return an optional sample-weighted mean. / 返回可选样本权重下的均值。"""
    if sample_weight is None:
        return values.mean()
    w = sample_weight.to(dtype=values.dtype, device=values.device)
    return (values * w).sum() / w.sum().clamp_min(1e-15)


def mse_loss(pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Mean squared error for regression. / 回归任务均方误差。"""
    return _weighted_mean((pred - target.to(pred.dtype)) ** 2, sample_weight)


def mae_loss(pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Mean absolute error for regression. / 回归任务平均绝对误差。"""
    return _weighted_mean(torch.abs(pred - target.to(pred.dtype)), sample_weight)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None = None, delta: float = 1.0) -> torch.Tensor:
    """Huber loss for robust regression. / 用于稳健回归的 Huber 损失。"""
    diff = torch.abs(pred - target.to(pred.dtype))
    values = torch.where(diff <= delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    return _weighted_mean(values, sample_weight)


def poisson_nll_loss(pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Poisson negative log-likelihood up to target-only constants. / 忽略仅与真实值有关常数项的 Poisson 负对数似然。"""
    lam = torch.clamp(pred, min=1e-12)
    y = target.to(pred.dtype)
    values = lam - y * torch.log(lam)
    return _weighted_mean(values, sample_weight)


def gamma_deviance_loss(pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Gamma deviance for positive severity-style regression. / 用于正值赔付强度回归的 Gamma deviance。"""
    mu = torch.clamp(pred, min=1e-12)
    y = torch.clamp(target.to(pred.dtype), min=1e-12)
    values = 2.0 * ((y - mu) / mu - torch.log(y / mu))
    return _weighted_mean(values, sample_weight)


def classification_nll_loss(prob: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Negative log-likelihood from BRB class probabilities. / 基于 BRB 类别信念概率的负对数似然。"""
    p = torch.clamp(prob, min=1e-12)
    rows = torch.arange(p.shape[0], device=p.device)
    values = -torch.log(p[rows, target.long()])
    return _weighted_mean(values, sample_weight)


def brier_loss(prob: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Multiclass Brier score used as a differentiable loss. / 将多分类 Brier score 作为可微损失。"""
    one_hot = torch.nn.functional.one_hot(target.long(), num_classes=prob.shape[1]).to(prob.dtype)
    values = ((prob - one_hot) ** 2).sum(dim=1)
    return _weighted_mean(values, sample_weight)


def get_loss(name: str) -> LossFn:
    """Return a named objective function. / 根据名称返回目标函数。"""
    key = name.lower()
    mapping: dict[str, LossFn] = {
        "mse": mse_loss,
        "mae": mae_loss,
        "huber": huber_loss,
        "poisson_nll": poisson_nll_loss,
        "gamma_deviance": gamma_deviance_loss,
        "cross_entropy": classification_nll_loss,
        "classification_nll": classification_nll_loss,
        "brier": brier_loss,
    }
    if key not in mapping:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(mapping)}")
    return mapping[key]
