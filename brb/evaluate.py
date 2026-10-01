from __future__ import annotations

import math

import torch

from .losses import gamma_deviance_loss, poisson_nll_loss


def evaluate_regression(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute common regression metrics. / 计算常用回归评价指标。"""
    pred = prediction.detach()
    y = target.to(pred.dtype)
    if sample_weight is None:
        w = torch.ones_like(y)
    else:
        w = sample_weight.to(y.dtype)
    w = w / w.sum().clamp_min(1e-15)
    error = pred - y
    mae = (w * torch.abs(error)).sum()
    mse = (w * error**2).sum()
    return {
        "mae": float(mae.item()),
        "rmse": float(math.sqrt(max(float(mse.item()), 0.0))),
        "poisson_nll": float(poisson_nll_loss(pred, y, sample_weight).item()),
        "gamma_deviance": float(gamma_deviance_loss(pred, y, sample_weight).item())
        if torch.all(y > 0) else float("nan"),
    }


def evaluate_classification(
    probability: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute accuracy and multiclass Brier score. / 计算分类准确率和多分类 Brier score。"""
    pred_class = probability.argmax(dim=1)
    correct = (pred_class == target.long()).to(probability.dtype)
    one_hot = torch.nn.functional.one_hot(target.long(), num_classes=probability.shape[1]).to(probability.dtype)
    brier = ((probability - one_hot) ** 2).sum(dim=1)
    if sample_weight is None:
        accuracy = correct.mean()
        brier_value = brier.mean()
    else:
        w = sample_weight.to(probability.dtype)
        w = w / w.sum().clamp_min(1e-15)
        accuracy = (w * correct).sum()
        brier_value = (w * brier).sum()
    return {"accuracy": float(accuracy.item()), "brier": float(brier_value.item())}


def evaluate_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Run BRB prediction and return task-appropriate test metrics. / 执行 BRB 预测并返回对应任务的测试指标。"""
    device = torch.device(device)
    model.to(device)
    x, y = x.to(device), y.to(device)
    if sample_weight is not None:
        sample_weight = sample_weight.to(device)
    model.eval()
    with torch.no_grad():
        pred, _ = model(x)
    if model.config.task == "regression":
        return evaluate_regression(pred, y, sample_weight)
    return evaluate_classification(pred, y, sample_weight)
