from __future__ import annotations

from typing import Any

import torch


def generate_report(model: torch.nn.Module, top_rules: int = 10) -> dict[str, Any]:
    """Export interpretable BRB parameters in a plain Python dictionary. / 以普通 Python 字典导出可解释的 BRB 参数。"""
    with torch.no_grad():
        theta = model.rule_weights().detach().cpu()
        beta = model.consequent_beliefs().detach().cpu()
        delta = model.attribute_weights().detach().cpu()

        refs: dict[str, list[Any]] = {}
        for i, attr in enumerate(model.config.attributes):
            if attr.kind == "numeric":
                refs[attr.name] = [float(v) for v in model.get_reference_points(i).detach().cpu().tolist()]
            else:
                refs[attr.name] = list(attr.reference_points)

        order = torch.argsort(theta, descending=True)[: min(top_rules, len(theta))]
        rules = []
        for idx in order.tolist():
            antecedent = {}
            for t, attr in enumerate(model.config.attributes):
                ref_idx = int(model.rule_index[idx, t].item())
                antecedent[attr.name] = refs[attr.name][ref_idx]
            rules.append({
                "rule_index": idx,
                "antecedent": antecedent,
                "rule_weight": float(theta[idx].item()),
                "consequent_belief": [float(v) for v in beta[idx].tolist()],
            })

        return {
            "task": model.config.task,
            "num_rules": model.config.num_rules,
            "reference_points": refs,
            "attribute_weights": delta.tolist(),
            "top_rules": rules,
        }
