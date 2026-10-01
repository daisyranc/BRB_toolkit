from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import BRBConfig


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a table from CSV/TSV/Excel/Parquet/JSON. / 从 CSV、TSV、Excel、Parquet 或 JSON 读取表格。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table format: {suffix}")


def inspect_data(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize data types, missingness, and basic statistics. / 汇总数据类型、缺失率和基础统计量。"""
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        row: dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "unique_count": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            row.update({
                "min": float(s.min()) if s.notna().any() else np.nan,
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
            })
        else:
            row.update({"min": None, "mean": None, "max": None})
        rows.append(row)
    return pd.DataFrame(rows)


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into train/validation/test sets. / 将数据划分为训练、验证和测试集。"""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.")
    n = len(df)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def _torch_dtype(config: BRBConfig) -> torch.dtype:
    """Map config dtype to a PyTorch dtype. / 将配置中的 dtype 转为 PyTorch dtype。"""
    return torch.float64 if config.dtype == "float64" else torch.float32


def encode_features(df: pd.DataFrame, config: BRBConfig, strict_categories: bool = True) -> torch.Tensor:
    """Encode configured antecedent columns into a numeric tensor. / 将前件属性编码为数值张量。"""
    columns: list[np.ndarray] = []
    for attr in config.attributes:
        s = df[attr.name]
        if attr.kind == "numeric":
            values = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        else:
            mapping = {str(v): i for i, v in enumerate(attr.reference_points)}
            values = np.full(len(s), np.nan, dtype=float)
            for row_idx, value in enumerate(s):
                if pd.isna(value):
                    continue
                key = str(value)
                if key not in mapping:
                    if strict_categories:
                        raise ValueError(f"Unknown category '{value}' in attribute '{attr.name}'.")
                    continue
                values[row_idx] = float(mapping[key])
        columns.append(values)
    x = np.stack(columns, axis=1)
    return torch.tensor(x, dtype=_torch_dtype(config))


def encode_target(df: pd.DataFrame, config: BRBConfig) -> torch.Tensor:
    """Encode the target column for regression or classification. / 按回归或分类任务编码目标列。"""
    s = df[config.target]
    if config.task == "regression":
        y = pd.to_numeric(s, errors="raise").to_numpy(dtype=float)
        return torch.tensor(y, dtype=_torch_dtype(config))
    mapping = {str(v): i for i, v in enumerate(config.consequent_refs)}
    encoded = []
    for value in s:
        key = str(value)
        if key not in mapping:
            raise ValueError(f"Unknown target class '{value}'.")
        encoded.append(mapping[key])
    return torch.tensor(encoded, dtype=torch.long)


def prepare_tensors(
    df: pd.DataFrame,
    config: BRBConfig,
    sample_weight_column: str | None = None,
    strict_categories: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Create feature, target, and optional sample-weight tensors. / 创建特征、目标和可选样本权重张量。"""
    x = encode_features(df, config, strict_categories=strict_categories)
    y = encode_target(df, config)
    w = None
    if sample_weight_column is not None:
        values = pd.to_numeric(df[sample_weight_column], errors="raise").to_numpy(dtype=float)
        w = torch.tensor(values, dtype=_torch_dtype(config))
    return x, y, w
