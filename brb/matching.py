from __future__ import annotations

import torch


def normalize_attribute_weights(
    delta: torch.Tensor,
    available_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalize attribute weights by the maximum available weight. / 按当前可用属性中的最大权重归一化属性权重。"""
    if delta.ndim == 1:
        d = delta.unsqueeze(0).expand(available_mask.shape[0], -1)
        masked = torch.where(available_mask, d, torch.zeros_like(d))
        max_d = masked.max(dim=1, keepdim=True).values
        max_d = torch.where(max_d > 0, max_d, torch.ones_like(max_d))
        return torch.where(available_mask, d / max_d, torch.zeros_like(d))
    if delta.ndim == 2:
        # rule-specific weights [L, T] -> [B, L, T]
        d = delta.unsqueeze(0).expand(available_mask.shape[0], -1, -1)
        mask = available_mask.unsqueeze(1).expand(-1, d.shape[1], -1)
        masked = torch.where(mask, d, torch.zeros_like(d))
        max_d = masked.max(dim=2, keepdim=True).values
        max_d = torch.where(max_d > 0, max_d, torch.ones_like(max_d))
        return torch.where(mask, d / max_d, torch.zeros_like(d))
    raise ValueError("delta must have shape [T] or [L, T].")


def compute_rule_matching(
    attribute_matching: list[torch.Tensor],
    rule_index: torch.Tensor,
    delta: torch.Tensor,
    available_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute conjunctive weighted matching for every sample-rule pair. / 计算每个样本与每条规则的加权合取匹配度。"""
    batch = available_mask.shape[0]
    num_rules, num_attrs = rule_index.shape
    gathered: list[torch.Tensor] = []
    for t in range(num_attrs):
        # matching_t: [B, K_t], rule_index[:, t]: [L] -> [B, L]
        matching_t = attribute_matching[t][:, rule_index[:, t]]
        gathered.append(matching_t)
    m = torch.stack(gathered, dim=2)  # [B, L, T]

    if delta.ndim == 1:
        exponents = normalize_attribute_weights(delta, available_mask).unsqueeze(1)
        exponents = exponents.expand(batch, num_rules, num_attrs)
    else:
        exponents = normalize_attribute_weights(delta, available_mask)

    # Missing attributes have exponent 0 and therefore contribute a neutral factor 1.
    # 缺失属性的指数为 0，因此在乘积中贡献中性因子 1。
    safe_m = torch.clamp(m, min=1e-15)
    log_alpha = (exponents * torch.log(safe_m)).sum(dim=2)
    alpha = torch.exp(log_alpha)

    # Preserve exact zero when an observed antecedent has zero matching.
    # 若已观测前件存在精确零匹配，则保持该规则匹配度为精确零。
    zero_observed = ((m == 0) & (exponents > 0)).any(dim=2)
    alpha = torch.where(zero_observed, torch.zeros_like(alpha), alpha)
    return alpha


def compute_rule_activation(alpha: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Normalize weighted rule matching into rule activation weights. / 将规则匹配度与规则权重归一化为规则激活权重。"""
    weighted = alpha * theta.unsqueeze(0)
    denom = weighted.sum(dim=1, keepdim=True)
    fallback = theta.unsqueeze(0).expand_as(weighted)
    return torch.where(
        denom > 1e-15,
        weighted / torch.clamp(denom, min=1e-15),
        fallback,
    )
