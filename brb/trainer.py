from __future__ import annotations

from copy import deepcopy

import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import get_loss
from .optimizers import create_optimizer


class Trainer:
    """Small training loop for differentiable BRB models. / 用于可微 BRB 模型的简洁训练器。"""

    def __init__(
        self,
        loss: str = "mse",
        optimizer: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        max_epochs: int = 500,
        patience: int = 50,
        min_delta: float = 1e-7,
        verbose: bool = True,
    ):
        """Store optimization and early-stopping settings. / 保存优化器和早停设置。"""
        self.loss_name = loss
        self.optimizer_name = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.history: list[dict[str, float]] = []
        self.best_epoch: int | None = None
        self.best_val_loss: float | None = None

    def _loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Evaluate the configured training objective. / 计算当前配置的训练目标函数。"""
        return get_loss(self.loss_name)(pred, target, sample_weight)

    def _evaluate_loss(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> float:
        """Evaluate loss without parameter updates. / 在不更新参数的情况下计算损失。"""
        model.eval()
        with torch.no_grad():
            pred, _ = model(x)
            loss = self._loss(pred, y, sample_weight)
        return float(loss.item())

    def _train_epoch_standard(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> float:
        """Train one epoch with Adam/AdamW/SGD. / 使用 Adam、AdamW 或 SGD 训练一个 epoch。"""
        if sample_weight is None:
            dataset = TensorDataset(x, y)
        else:
            dataset = TensorDataset(x, y, sample_weight)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in loader:
            xb, yb = batch[0], batch[1]
            wb = batch[2] if len(batch) == 3 else None
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = self._loss(pred, yb, wb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            total_n += len(xb)
        return total_loss / max(total_n, 1)

    def _train_epoch_lbfgs(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> float:
        """Train one full-batch epoch with L-BFGS. / 使用 L-BFGS 进行一次全批量训练。"""
        model.train()

        def closure() -> torch.Tensor:
            """Recompute loss and gradients for L-BFGS. / 为 L-BFGS 重新计算损失与梯度。"""
            optimizer.zero_grad()
            pred, _ = model(x)
            loss = self._loss(pred, y, sample_weight)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        return float(loss.item())

    def fit(
        self,
        model: torch.nn.Module,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        train_weight: torch.Tensor | None = None,
        val_weight: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> list[dict[str, float]]:
        """Train the BRB and restore the best validation checkpoint. / 训练 BRB 并恢复验证集表现最佳的参数。"""
        device = torch.device(device)
        model.to(device)
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_val, y_val = x_val.to(device), y_val.to(device)
        if train_weight is not None:
            train_weight = train_weight.to(device)
        if val_weight is not None:
            val_weight = val_weight.to(device)

        optimizer = create_optimizer(
            model,
            name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_state = deepcopy(model.state_dict())
        best_val = float("inf")
        bad_epochs = 0
        self.history = []

        for epoch in range(1, self.max_epochs + 1):
            if self.optimizer_name.lower() == "lbfgs":
                train_loss = self._train_epoch_lbfgs(
                    model, optimizer, x_train, y_train, train_weight
                )
            else:
                train_loss = self._train_epoch_standard(
                    model, optimizer, x_train, y_train, train_weight
                )

            val_loss = self._evaluate_loss(model, x_val, y_val, val_weight)
            self.history.append({
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            })

            improved = val_loss < best_val - self.min_delta
            if improved:
                best_val = val_loss
                best_state = deepcopy(model.state_dict())
                self.best_epoch = epoch
                self.best_val_loss = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            if self.verbose and (epoch == 1 or epoch % 20 == 0 or improved):
                print(f"Epoch {epoch:4d} | train={train_loss:.6f} | val={val_loss:.6f}")

            if bad_epochs >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}; best epoch = {self.best_epoch}.")
                break

        model.load_state_dict(best_state)
        return self.history
