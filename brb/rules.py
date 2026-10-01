from __future__ import annotations

from itertools import product

import torch

from .config import BRBConfig


def count_rules(config: BRBConfig) -> int:
    """Return the number of Cartesian-product rules. / 返回笛卡尔积规则数量。"""
    return config.num_rules


def build_rule_index(config: BRBConfig) -> torch.Tensor:
    """Build rule antecedents as reference-value indices. / 用参考值索引构建完整规则前件。"""
    ranges = [range(len(attr.reference_points)) for attr in config.attributes]
    rows = list(product(*ranges))
    return torch.tensor(rows, dtype=torch.long)


def rule_count_warning(config: BRBConfig, threshold: int = 5000) -> str | None:
    """Return a warning when the rule base becomes large. / 当规则数量较大时返回提示信息。"""
    n = config.num_rules
    if n <= threshold:
        return None
    return (
        f"The Cartesian rule base contains {n:,} rules. "
        "Consider fewer attributes/reference levels or a hierarchical BRB."
    )
