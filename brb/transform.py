from __future__ import annotations

import torch


def numeric_matching(value: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
    """Linearly map numeric values to neighboring reference points; missing values match all points. / 将数值线性映射到相邻参考点；缺失值对所有参考点保持中性匹配。"""
    x = value.reshape(-1)
    batch = x.shape[0]
    k = refs.numel()
    out = torch.zeros((batch, k), dtype=x.dtype, device=x.device)
    missing = torch.isnan(x)
    if missing.any():
        out[missing] = 1.0
    valid = ~missing
    if not valid.any():
        return out

    xv = x[valid]
    low_mask = xv <= refs[0]
    high_mask = xv >= refs[-1]
    valid_rows = torch.where(valid)[0]
    if low_mask.any():
        out[valid_rows[low_mask], 0] = 1.0
    if high_mask.any():
        out[valid_rows[high_mask], -1] = 1.0

    mid_mask = ~(low_mask | high_mask)
    if mid_mask.any():
        xm = xv[mid_mask]
        rows = valid_rows[mid_mask]
        # Interval lookup is discrete; interpolation inside the selected interval remains differentiable.
        # 区间索引本身是离散的；选定区间内的线性插值仍可对参考点求梯度。
        idx_hi = torch.searchsorted(refs.detach(), xm, right=False)
        idx_lo = idx_hi - 1
        lo = refs[idx_lo]
        hi = refs[idx_hi]
        upper = (xm - lo) / (hi - lo)
        lower = 1.0 - upper
        out[rows, idx_lo] = lower
        out[rows, idx_hi] = upper
    return out


def categorical_matching(value: torch.Tensor, num_categories: int) -> torch.Tensor:
    """Create one-hot categorical matching; missing values match all categories. / 创建类别型 one-hot 匹配；缺失值对所有类别保持中性匹配。"""
    x = value.reshape(-1)
    out = torch.zeros((x.shape[0], num_categories), dtype=x.dtype, device=x.device)
    missing = torch.isnan(x)
    if missing.any():
        out[missing] = 1.0
    valid = ~missing
    if valid.any():
        idx = x[valid].long()
        if torch.any(idx < 0) or torch.any(idx >= num_categories):
            raise ValueError("Categorical code is outside the configured reference range.")
        out[torch.where(valid)[0], idx] = 1.0
    return out


def scalar_to_belief(value: float, consequent_refs: list[float], floor: float = 0.0) -> torch.Tensor:
    """Represent a scalar by belief on two adjacent consequent references. / 将标量表示为两个相邻后件参考点上的信念分布。"""
    refs = torch.tensor(consequent_refs, dtype=torch.float64)
    y = torch.tensor(float(value), dtype=torch.float64)
    beta = torch.zeros_like(refs)
    if y <= refs[0]:
        beta[0] = 1.0
    elif y >= refs[-1]:
        beta[-1] = 1.0
    else:
        hi = int(torch.searchsorted(refs, y, right=False).item())
        lo = hi - 1
        upper = (y - refs[lo]) / (refs[hi] - refs[lo])
        beta[lo] = 1.0 - upper
        beta[hi] = upper
    if floor > 0:
        beta = torch.clamp(beta, min=floor)
        beta = beta / beta.sum()
    return beta
