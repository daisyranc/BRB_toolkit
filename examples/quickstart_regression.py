"""Minimal regression example / 最小回归示例。"""

import numpy as np
import pandas as pd

from brb import (
    AttributeConfig,
    BRBConfig,
    BRBModel,
    Trainer,
    evaluate_model,
    generate_report,
    initialize_model_from_data,
    prepare_tensors,
    split_data,
)


def make_demo_data(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Create a small synthetic regression table. / 创建一个小型合成回归表格。"""
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 80, n)
    power = rng.uniform(4, 15, n)
    # Positive mean-like target; replace this with your real target column.
    # 正的均值型目标；实际使用时替换成真实目标列。
    y = 0.05 + 0.006 * (age - 18) + 0.04 * (power - 4)
    y += rng.normal(0, 0.05, n)
    y = np.clip(y, 0.001, None)
    df = pd.DataFrame({"Age": age, "VehiclePower": power, "Target": y})
    # Add a few missing antecedents to demonstrate rule marginalization.
    # 加入少量前件缺失值以演示规则边缘化。
    missing_idx = rng.choice(n, size=20, replace=False)
    df.loc[missing_idx, "Age"] = np.nan
    return df


def main() -> None:
    """Run the complete train/validation/test workflow. / 运行完整的训练、验证和测试流程。"""
    df = make_demo_data()
    train_df, val_df, test_df = split_data(df, seed=42)

    config = BRBConfig(
        target="Target",
        task="regression",
        attributes=[
            AttributeConfig("Age", "numeric", [18, 35, 55, 80], adaptive=True),
            AttributeConfig("VehiclePower", "numeric", [4, 8, 12, 15], adaptive=True),
        ],
        consequent_refs=[0.0, 0.2, 0.4, 0.7, 1.0],
    )

    x_train, y_train, _ = prepare_tensors(train_df, config)
    x_val, y_val, _ = prepare_tensors(val_df, config)
    x_test, y_test, _ = prepare_tensors(test_df, config)

    model = BRBModel(config)
    initialize_model_from_data(model, x_train, y_train, shrinkage=20, belief_floor=1e-3)

    trainer = Trainer(
        loss="mse",
        optimizer="adam",
        lr=2e-3,
        max_epochs=120,
        patience=20,
    )
    trainer.fit(model, x_train, y_train, x_val, y_val)

    print("Test metrics:", evaluate_model(model, x_test, y_test))
    print("Report:", generate_report(model, top_rules=5))


if __name__ == "__main__":
    main()
