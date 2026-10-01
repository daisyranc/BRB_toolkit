from __future__ import annotations

import torch


def create_optimizer(
    model: torch.nn.Module,
    name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Create a common PyTorch optimizer for BRB training. / 创建常用的 PyTorch 优化器用于 BRB 训练。"""
    key = name.lower()
    params = model.parameters()
    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if key == "lbfgs":
        return torch.optim.LBFGS(params, lr=lr, max_iter=20, line_search_fn="strong_wolfe")
    raise ValueError("Optimizer must be one of: adam, adamw, sgd, lbfgs.")
