from __future__ import annotations

import math

import torch
from torch import nn

from .config import BRBConfig
from .er import er_aggregate
from .matching import compute_rule_activation, compute_rule_matching
from .missing import available_attribute_mask
from .rules import build_rule_index
from .transform import categorical_matching, numeric_matching


def inverse_softplus(x: float) -> float:
    """Return the inverse softplus for a positive scalar. / 返回正数的 softplus 逆变换。"""
    if x <= 0:
        raise ValueError("inverse_softplus requires x > 0.")
    return math.log(math.expm1(x))


class BRBModel(nn.Module):
    """Trainable complete BRB with ER aggregation. / 使用 ER 聚合的可训练完整 BRB。"""

    def __init__(self, config: BRBConfig):
        """Create a BRB model from configuration. / 根据配置创建 BRB 模型。"""
        super().__init__()
        config.validate()
        self.config = config
        dtype = torch.float64 if config.dtype == "float64" else torch.float32

        rule_index = build_rule_index(config)
        self.register_buffer("rule_index", rule_index)

        num_rules = config.num_rules
        num_consequents = len(config.consequent_refs)

        # Raw consequent parameters b; softmax(b) gives valid belief rows.
        # 原始后件参数 b；softmax(b) 生成合法的信念分布。
        self.belief_logits = nn.Parameter(torch.zeros(num_rules, num_consequents, dtype=dtype))

        # Raw rule parameters q; softmax(q) gives positive normalized rule weights.
        # 原始规则参数 q；softmax(q) 生成正且归一化的规则权重。
        self.rule_logits = nn.Parameter(torch.zeros(num_rules, dtype=dtype))

        init_a = inverse_softplus(1.0)
        if config.attribute_weight_mode == "global":
            shape = (config.num_attributes,)
        else:
            shape = (num_rules, config.num_attributes)
        self.attribute_raw = nn.Parameter(torch.full(shape, init_a, dtype=dtype))

        # Adaptive numeric reference points are represented by positive interval proportions.
        # 自适应数值参考点通过正的区间比例进行参数化。
        self.reference_gap_logits = nn.ParameterDict()
        for i, attr in enumerate(config.attributes):
            if attr.kind == "numeric" and attr.adaptive and len(attr.reference_points) > 2:
                refs = torch.tensor(attr.reference_points, dtype=dtype)
                gaps = refs[1:] - refs[:-1]
                proportions = gaps / gaps.sum()
                self.reference_gap_logits[str(i)] = nn.Parameter(torch.log(proportions))

        if config.task == "regression":
            self.register_buffer(
                "consequent_values",
                torch.tensor([float(v) for v in config.consequent_refs], dtype=dtype),
            )

    def consequent_beliefs(self) -> torch.Tensor:
        """Return normalized consequent-belief entries beta. / 返回归一化后的后件信念参数 beta。"""
        return torch.softmax(self.belief_logits, dim=1)

    def rule_weights(self) -> torch.Tensor:
        """Return normalized rule-weight entries theta. / 返回归一化后的规则权重 theta。"""
        return torch.softmax(self.rule_logits, dim=0)

    def attribute_weights(self) -> torch.Tensor:
        """Return positive attribute-weight entries delta. / 返回正的属性权重 delta。"""
        return torch.nn.functional.softplus(self.attribute_raw)

    def get_reference_points(self, attr_index: int) -> torch.Tensor:
        """Return current reference points for one antecedent attribute. / 返回某个前件属性当前的参考点。"""
        attr = self.config.attributes[attr_index]
        dtype = self.belief_logits.dtype
        device = self.belief_logits.device
        if attr.kind == "categorical":
            return torch.arange(len(attr.reference_points), dtype=dtype, device=device)

        base = torch.tensor(attr.reference_points, dtype=dtype, device=device)
        key = str(attr_index)
        if key not in self.reference_gap_logits:
            return base

        lower = base[0]
        upper = base[-1]
        proportions = torch.softmax(self.reference_gap_logits[key], dim=0)
        gaps = proportions * (upper - lower)
        refs = torch.cat([lower.reshape(1), lower + torch.cumsum(gaps, dim=0)])
        # The final point is mathematically equal to upper; assigning upper avoids tiny drift.
        # 最后一个点理论上等于上界；显式使用上界可避免微小数值漂移。
        refs = torch.cat([refs[:-1], upper.reshape(1)])
        return refs

    def compute_attribute_matching(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Transform raw antecedents into per-attribute matching degrees. / 将原始前件输入转换为各属性的匹配度。"""
        results: list[torch.Tensor] = []
        for i, attr in enumerate(self.config.attributes):
            if attr.kind == "numeric":
                refs = self.get_reference_points(i)
                results.append(numeric_matching(x[:, i], refs))
            else:
                results.append(categorical_matching(x[:, i], len(attr.reference_points)))
        return results

    def compute_rule_matching(self, x: torch.Tensor) -> torch.Tensor:
        """Compute matching degree alpha for every sample and rule. / 计算每个样本与每条规则的匹配度 alpha。"""
        attr_matching = self.compute_attribute_matching(x)
        available = available_attribute_mask(x)
        delta = self.attribute_weights()
        return compute_rule_matching(attr_matching, self.rule_index, delta, available)

    def compute_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized rule activation weights w. / 计算归一化后的规则激活权重 w。"""
        alpha = self.compute_rule_matching(x)
        theta = self.rule_weights()
        return compute_rule_activation(alpha, theta)

    def aggregate(self, activation: torch.Tensor) -> torch.Tensor:
        """Aggregate activated rules into an output belief distribution. / 将激活规则聚合为输出信念分布。"""
        beta = self.consequent_beliefs()
        return er_aggregate(activation, beta)

    def belief_to_value(self, belief: torch.Tensor) -> torch.Tensor:
        """Convert regression belief distributions to expected numeric values. / 将回归任务的信念分布转换为数值期望。"""
        if self.config.task != "regression":
            raise ValueError("belief_to_value is only defined for regression models.")
        return belief @ self.consequent_values

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run BRB inference and return prediction plus belief distribution. / 执行 BRB 推理并返回预测值与信念分布。"""
        activation = self.compute_activation(x)
        belief = self.aggregate(activation)
        if self.config.task == "regression":
            prediction = self.belief_to_value(belief)
        else:
            prediction = belief
        return prediction, belief
