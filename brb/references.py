from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def create_numeric_reference_points(
    series: pd.Series,
    n_points: int,
    method: str = "quantile",
) -> list[float]:
    """Create numeric reference points from training data only. / 仅基于训练数据生成数值型参考点。"""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("No valid numeric values are available.")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    if method == "quantile":
        refs = np.quantile(values, np.linspace(0.0, 1.0, n_points))
    elif method == "uniform":
        refs = np.linspace(values.min(), values.max(), n_points)
    else:
        raise ValueError("method must be 'quantile' or 'uniform'.")
    refs = np.unique(refs)
    if len(refs) < 2:
        raise ValueError("Generated reference points are not distinct.")
    return [float(x) for x in refs]


def validate_numeric_reference_points(reference_points: Sequence[float]) -> None:
    """Check that numeric reference points are strictly increasing. / 检查数值型参考点是否严格递增。"""
    refs = [float(x) for x in reference_points]
    if len(refs) < 2:
        raise ValueError("At least two reference points are required.")
    if any(b <= a for a, b in zip(refs[:-1], refs[1:])):
        raise ValueError("Reference points must be strictly increasing.")
