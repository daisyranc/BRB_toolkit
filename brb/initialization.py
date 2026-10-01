from __future__ import annotations

import torch

from .model import BRBModel
from .transform import scalar_to_belief


def _normalize_rows(beta: torch.Tensor, floor: float) -> torch.Tensor:
    """Apply a belief floor and renormalize rows. / 对信念应用下限并按行重新归一化。"""
    if floor > 0:
        beta = torch.clamp(beta, min=floor)
    return beta / beta.sum(dim=1, keepdim=True)


def initialize_regression_consequents(
    model: BRBModel,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    shrinkage: float = 0.0,
    belief_floor: float = 1e-3,
) -> torch.Tensor:
    """Initialize rule consequents from local weighted target means. / 根据局部加权目标均值初始化规则后件信念。"""
    if model.config.task != "regression":
        raise ValueError("This initializer is only for regression BRBs.")
    with torch.no_grad():
        # W_ik is the initial antecedent matching contribution.
        # W_ik 表示样本 i 对规则 k 的初始前件匹配贡献。
        w_rule = model.compute_rule_matching(x_train)
        if sample_weight is None:
            obs_weight = torch.ones_like(y_train, dtype=x_train.dtype)
        else:
            obs_weight = sample_weight.to(dtype=x_train.dtype)
        effective = w_rule * obs_weight.unsqueeze(1)

        global_mean = (obs_weight * y_train.to(x_train.dtype)).sum() / obs_weight.sum().clamp_min(1e-15)
        support = effective.sum(dim=0)
        local_sum = (effective * y_train.to(x_train.dtype).unsqueeze(1)).sum(dim=0)
        mu0 = (local_sum + shrinkage * global_mean) / (support + shrinkage).clamp_min(1e-15)

        refs = [float(v) for v in model.config.consequent_refs]
        rows = [scalar_to_belief(float(v.item()), refs, floor=0.0) for v in mu0]
        beta0 = torch.stack(rows).to(dtype=model.belief_logits.dtype, device=model.belief_logits.device)
        beta0 = _normalize_rows(beta0, belief_floor)
        model.belief_logits.copy_(torch.log(beta0))
        return beta0


def initialize_classification_consequents(
    model: BRBModel,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    shrinkage: float = 0.0,
    belief_floor: float = 1e-3,
) -> torch.Tensor:
    """Initialize rule consequents from local weighted class frequencies. / 根据局部加权类别频率初始化规则后件信念。"""
    if model.config.task != "classification":
        raise ValueError("This initializer is only for classification BRBs.")
    with torch.no_grad():
        w_rule = model.compute_rule_matching(x_train)
        n_classes = len(model.config.consequent_refs)
        one_hot = torch.nn.functional.one_hot(y_train.long(), num_classes=n_classes).to(x_train.dtype)
        if sample_weight is None:
            obs_weight = torch.ones(x_train.shape[0], dtype=x_train.dtype, device=x_train.device)
        else:
            obs_weight = sample_weight.to(dtype=x_train.dtype, device=x_train.device)

        weighted_one_hot = one_hot * obs_weight.unsqueeze(1)
        global_dist = weighted_one_hot.sum(dim=0)
        global_dist = global_dist / global_dist.sum().clamp_min(1e-15)

        effective = w_rule * obs_weight.unsqueeze(1)
        support = effective.sum(dim=0)
        counts = torch.einsum("bl,bn->ln", effective, one_hot)
        beta0 = (counts + shrinkage * global_dist.unsqueeze(0)) / (support.unsqueeze(1) + shrinkage).clamp_min(1e-15)
        beta0 = _normalize_rows(beta0, belief_floor)
        model.belief_logits.copy_(torch.log(beta0))
        return beta0


def initialize_model_from_data(
    model: BRBModel,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    shrinkage: float = 0.0,
    belief_floor: float = 1e-3,
) -> torch.Tensor:
    """Initialize BRB consequents from training data while keeping neutral rule/attribute weights. / 用训练数据初始化 BRB 后件，同时保持中性的规则与属性权重。"""
    with torch.no_grad():
        model.rule_logits.zero_()
        # Equal positive attribute weights are already encoded by the constructor.
        # 构造函数已经将属性权重初始化为相等的正值。
    if model.config.task == "regression":
        return initialize_regression_consequents(
            model, x_train, y_train, sample_weight, shrinkage, belief_floor
        )
    return initialize_classification_consequents(
        model, x_train, y_train, sample_weight, shrinkage, belief_floor
    )
