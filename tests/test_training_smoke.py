import torch

from brb.config import AttributeConfig, BRBConfig
from brb.initialization import initialize_model_from_data
from brb.model import BRBModel
from brb.trainer import Trainer


def test_training_smoke():
    """Run a tiny differentiable training smoke test. / 运行一个最小可微训练冒烟测试。"""
    torch.manual_seed(0)
    x = torch.rand(80, 2, dtype=torch.float64)
    y = 0.2 + 0.4 * x[:, 0] + 0.2 * x[:, 1]

    config = BRBConfig(
        target="y",
        task="regression",
        attributes=[
            AttributeConfig("x1", "numeric", [0.0, 0.5, 1.0], adaptive=True),
            AttributeConfig("x2", "numeric", [0.0, 0.5, 1.0], adaptive=True),
        ],
        consequent_refs=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    model = BRBModel(config)
    initialize_model_from_data(model, x[:50], y[:50], shrinkage=5)
    trainer = Trainer(loss="mse", optimizer="adam", lr=0.01, max_epochs=8, patience=8, verbose=False)
    history = trainer.fit(model, x[:50], y[:50], x[50:65], y[50:65])
    assert len(history) >= 1
    pred, belief = model(x[65:])
    assert torch.all(torch.isfinite(pred))
    assert torch.allclose(belief.sum(dim=1), torch.ones(len(pred), dtype=torch.float64), atol=1e-8)
