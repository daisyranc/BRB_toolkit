from __future__ import annotations

import torch


def combine_two_evidences(
    mass_a: torch.Tensor,
    theta_a: torch.Tensor,
    mass_b: torch.Tensor,
    theta_b: torch.Tensor,
    eps: float = 1e-15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two ER mass assignments with conflict normalization. / 使用冲突归一化合并两条 ER 质量分配。"""
    # Conflict K = sum_{i != j} m_i^A m_j^B.
    # 冲突量 K 为两条证据对不同结果的交叉支持之和。
    total_a = mass_a.sum(dim=1)
    total_b = mass_b.sum(dim=1)
    same = (mass_a * mass_b).sum(dim=1)
    conflict = total_a * total_b - same
    denom = torch.clamp(1.0 - conflict, min=eps)

    new_mass = (
        mass_a * mass_b
        + mass_a * theta_b.unsqueeze(1)
        + theta_a.unsqueeze(1) * mass_b
    ) / denom.unsqueeze(1)
    new_theta = (theta_a * theta_b) / denom
    return new_mass, new_theta


def er_aggregate(
    activation: torch.Tensor,
    rule_beliefs: torch.Tensor,
    eps: float = 1e-15,
) -> torch.Tensor:
    """Aggregate activated complete belief rules using recursive ER combination. / 使用递归 ER 组合聚合已激活的完整信念规则。"""
    if activation.ndim != 2 or rule_beliefs.ndim != 2:
        raise ValueError("activation must be [B,L] and rule_beliefs must be [L,N].")
    if activation.shape[1] != rule_beliefs.shape[0]:
        raise ValueError("Rule dimension mismatch between activation and rule beliefs.")

    batch, num_rules = activation.shape
    num_outcomes = rule_beliefs.shape[1]
    if num_rules == 0:
        raise ValueError("At least one rule is required.")

    w0 = activation[:, 0]
    mass = w0.unsqueeze(1) * rule_beliefs[0].unsqueeze(0)
    residual = 1.0 - w0

    for k in range(1, num_rules):
        wk = activation[:, k]
        mass_k = wk.unsqueeze(1) * rule_beliefs[k].unsqueeze(0)
        residual_k = 1.0 - wk
        mass, residual = combine_two_evidences(
            mass,
            residual,
            mass_k,
            residual_k,
            eps=eps,
        )

    committed = mass.sum(dim=1, keepdim=True)
    if torch.any(committed <= eps):
        # This is only expected in a degenerate fully-uncommitted case.
        # 仅在退化的完全未分配情形下会进入此分支。
        uniform = torch.full(
            (batch, num_outcomes),
            1.0 / num_outcomes,
            dtype=mass.dtype,
            device=mass.device,
        )
        return torch.where(committed > eps, mass / torch.clamp(committed, min=eps), uniform)
    return mass / committed
