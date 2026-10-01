from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class AttributeConfig:
    """Describe one antecedent attribute. / 描述一个前件属性。"""

    name: str
    kind: Literal["numeric", "categorical"]
    reference_points: list[Any]
    adaptive: bool = False

    def validate(self) -> None:
        """Validate one attribute configuration. / 检查单个属性配置是否合法。"""
        if len(self.reference_points) < 2:
            raise ValueError(f"{self.name}: at least two reference points are required.")
        if self.kind == "numeric":
            refs = [float(x) for x in self.reference_points]
            if any(b <= a for a, b in zip(refs[:-1], refs[1:])):
                raise ValueError(f"{self.name}: numeric reference points must be strictly increasing.")
        elif self.kind == "categorical":
            if len(set(map(str, self.reference_points))) != len(self.reference_points):
                raise ValueError(f"{self.name}: categorical reference values must be unique.")
            if self.adaptive:
                raise ValueError(f"{self.name}: categorical reference values cannot be adaptive.")
        else:
            raise ValueError(f"{self.name}: unsupported kind '{self.kind}'.")


@dataclass
class BRBConfig:
    """Configuration of a BRB model. / BRB 模型配置。"""

    target: str
    attributes: list[AttributeConfig]
    consequent_refs: list[Any]
    task: Literal["regression", "classification"] = "regression"
    attribute_weight_mode: Literal["global", "rule_specific"] = "global"
    missing_strategy: Literal["rule_marginalization"] = "rule_marginalization"
    dtype: Literal["float32", "float64"] = "float64"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the full BRB configuration. / 检查完整 BRB 配置。"""
        if not self.attributes:
            raise ValueError("At least one antecedent attribute is required.")
        names = [a.name for a in self.attributes]
        if len(names) != len(set(names)):
            raise ValueError("Attribute names must be unique.")
        for attr in self.attributes:
            attr.validate()
        if len(self.consequent_refs) < 2:
            raise ValueError("At least two consequent reference values are required.")
        if self.task == "regression":
            vals = [float(x) for x in self.consequent_refs]
            if any(b <= a for a, b in zip(vals[:-1], vals[1:])):
                raise ValueError("Regression consequent reference values must be strictly increasing.")
        elif self.task == "classification":
            if len(set(map(str, self.consequent_refs))) != len(self.consequent_refs):
                raise ValueError("Classification consequent labels must be unique.")
        else:
            raise ValueError(f"Unsupported task '{self.task}'.")

    @property
    def num_attributes(self) -> int:
        """Return the number of antecedent attributes. / 返回前件属性数量。"""
        return len(self.attributes)

    @property
    def num_rules(self) -> int:
        """Return the Cartesian-product rule count. / 返回笛卡尔积规则数量。"""
        n = 1
        for attr in self.attributes:
            n *= len(attr.reference_points)
        return n

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a plain dictionary. / 将配置转换为普通字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BRBConfig":
        """Build a configuration from a dictionary. / 从字典创建配置。"""
        attrs = [AttributeConfig(**item) for item in data["attributes"]]
        obj = cls(
            target=data["target"],
            attributes=attrs,
            consequent_refs=data["consequent_refs"],
            task=data.get("task", "regression"),
            attribute_weight_mode=data.get("attribute_weight_mode", "global"),
            missing_strategy=data.get("missing_strategy", "rule_marginalization"),
            dtype=data.get("dtype", "float64"),
            metadata=data.get("metadata", {}),
        )
        obj.validate()
        return obj


def save_config(config: BRBConfig, path: str | Path) -> None:
    """Save a BRB configuration as YAML. / 将 BRB 配置保存为 YAML。"""
    config.validate()
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False, allow_unicode=True)


def load_config(path: str | Path) -> BRBConfig:
    """Load a BRB configuration from YAML. / 从 YAML 读取 BRB 配置。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return BRBConfig.from_dict(data)
