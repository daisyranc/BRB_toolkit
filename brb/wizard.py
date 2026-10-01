from __future__ import annotations

import ast
from pathlib import Path

from .config import AttributeConfig, BRBConfig, save_config
from .data import inspect_data, load_table


def _parse_list(text: str) -> list:
    """Parse a comma-separated or Python-list user input. / 解析逗号分隔或 Python 列表形式的用户输入。"""
    text = text.strip()
    if text.startswith("["):
        value = ast.literal_eval(text)
        if not isinstance(value, list):
            raise ValueError("Please enter a list.")
        return value
    return [item.strip() for item in text.split(",") if item.strip()]


def run_wizard(output_path: str | Path = "brb_config.yaml") -> None:
    """Interactively create a simple BRB YAML configuration. / 通过交互式步骤创建简单的 BRB YAML 配置。"""
    print("BRB Toolkit Wizard / BRB 建模向导")
    path = input("Data file / 数据文件路径: ").strip()
    df = load_table(path)
    print("\nData summary / 数据概览:")
    print(inspect_data(df).to_string(index=False))

    target = input("\nTarget column / 目标列: ").strip()
    task = input("Task [regression/classification] / 任务类型 [regression/classification]: ").strip() or "regression"
    names = _parse_list(input("Antecedent columns, comma-separated / 前件属性列，用逗号分隔: "))

    attrs: list[AttributeConfig] = []
    for name in names:
        kind = input(f"{name} kind [numeric/categorical] / 类型: ").strip() or "numeric"
        raw_refs = _parse_list(input(f"{name} reference points / 参考点: "))
        if kind == "numeric":
            refs = [float(v) for v in raw_refs]
            adaptive = (input(f"Adaptive internal points for {name}? [y/N] / 是否训练内部参考点? ").strip().lower() == "y")
        else:
            refs = raw_refs
            adaptive = False
        attrs.append(AttributeConfig(name=name, kind=kind, reference_points=refs, adaptive=adaptive))

    raw_consequents = _parse_list(input("Consequent references / 后件参考值: "))
    consequents = [float(v) for v in raw_consequents] if task == "regression" else raw_consequents

    config = BRBConfig(
        target=target,
        attributes=attrs,
        consequent_refs=consequents,
        task=task,
    )
    config.validate()
    save_config(config, output_path)
    print(f"\nSaved / 已保存: {output_path}")
    print(f"Rule count / 规则数量: {config.num_rules}")


def main() -> None:
    """Command-line entry point for the interactive wizard. / BRB 交互式向导的命令行入口。"""
    run_wizard()


if __name__ == "__main__":
    main()
