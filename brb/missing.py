from __future__ import annotations

import torch


def available_attribute_mask(x: torch.Tensor) -> torch.Tensor:
    """Return True where antecedent values are observed. / 返回前件属性是否已观测的布尔掩码。"""
    return ~torch.isnan(x)


def count_missing_by_sample(x: torch.Tensor) -> torch.Tensor:
    """Count missing antecedent values for each sample. / 统计每个样本的缺失前件数量。"""
    return torch.isnan(x).sum(dim=1)
