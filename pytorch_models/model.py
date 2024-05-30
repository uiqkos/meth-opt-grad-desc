import abc
from typing import List, Callable, Iterator

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class _LinearRegressionModel(nn.Module):
    def __init__(self, f: Callable, parameters: int, in_size=1, out_size=1):
        super(_LinearRegressionModel, self).__init__()
        self.layers = []
        self.in_size = in_size
        self.out_size = out_size
        self.params = nn.ParameterList([
            Parameter(torch.randn(in_size, out_size))
            for _ in range(parameters)
        ])
        self.expr = f

    def forward(self, x):
        return self.expr(*self.params, x)

    def __call__(self, x, *args, **kwargs):
        return self.expr(*self.params, x)


class LinearRegressionModel(abc.ABC):
    def __init__(
        self,
        f: Callable,
        parameters: int,
        optimizer: Callable[[Iterator[Parameter]], Optimizer],
        criterion: _Loss,
        in_size: int = 1,
        out_size: int = 1,
    ):
        self.criterion = criterion
        self.model = _LinearRegressionModel(f, parameters, in_size, out_size)
        self.optimizer = optimizer(self.model.parameters())

    def fit(
        self,
        train_loader:
        DataLoader,
        epochs: int = 50,
        callback: Callable[[int, float], None] = None,
    ) -> List[float]:
        losses = []
        for epoch in range(epochs):
            self.model.train()
            loss = 0
            loss_counter = 0

            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                l = self.criterion(outputs, y_batch)
                l.backward()
                loss += l.item()
                loss_counter += 1
                self.optimizer.step()

            losses.append(loss / loss_counter)

            if callback is not None:
                callback(epoch, losses[-1])

        return losses

    def score(self, test_loader: DataLoader) -> float:
        loss = 0
        self.model.eval()

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = self.model(X_batch)
                loss += self.criterion(outputs, y_batch).item()

        return loss / len(test_loader)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(X)
