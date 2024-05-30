import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Iterator, List
from torch.nn.parameter import Parameter
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
import abc


class _DistanceEmbeddingModel(nn.Module):
    def __init__(self, num_points=5, embedding_dim=2):
        super(_DistanceEmbeddingModel, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(num_points, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self):
        return self.embeddings + self.bias


def gradd(X, D):
    N = X.shape[0]
    grad = torch.zeros_like(X)
    indexes = random.choices(list(range(N)), k=64) if True else list(range(N))

    for i in indexes:
        for j in indexes:
            difference = X[i] - X[j]
            squared_distance = (difference.T @ difference)
            grad[i] += (squared_distance - D[i, j] ** 2) * difference

    return grad


class DistanceEmbeddingModel(abc.ABC):
    def __init__(
        self,
        optimizer: Callable[[Iterator[Parameter]], Optimizer],
        criterion: _Loss,
        num_points: int = 5,
        embedding_dim: int = 2,
    ):
        self.criterion = criterion
        self.model = _DistanceEmbeddingModel(num_points, embedding_dim)
        self.optimizer = optimizer(self.model.parameters())

    def fit(
        self,
        D: torch.Tensor,
        epochs: int = 50,
        callback: Callable[[int, float, torch.Tensor], None] = None,
    ) -> List[float]:
        losses = []
        self.model.train()

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            embeddings = self.model.embeddings
            # print(embeddings)
            # loss = self.criterion(embeddings, D)
            # loss.backward()
            # print(embeddings)
            embeddings.grad = gradd(embeddings, D)
            # print()
            self.optimizer.step()

            # embeddings += 0.01 * grad(embeddings, D)
            # print(loss.item())
            loss = torch.zeros(1)

            losses.append(loss.item())

            if callback is not None:
                callback(epoch, loss.item(), embeddings.detach().numpy())

        return losses

    def predict(self) -> torch.Tensor:
        with torch.no_grad():
            return self.model().detach()


def loss_fn(embeddings, D, batched=False):
    N = embeddings.shape[0]
    loss = torch.zeros(1)
    loss_count = 0

    indexes = random.choices(list(range(N)), k=64) if batched else list(range(N))

    for i in indexes:
        for j in indexes:
            if i != j:
                diff = embeddings[i] - embeddings[j]
                # dist_squared = diff.T @ diff
                distance = torch.norm(embeddings[i] - embeddings[j])
                loss += (distance - D[i, j]) ** 2
                loss_count += 1

    return loss / loss_count

# def loss_fn(embeddings, D):
#     N = embeddings.shape[0]
#     loss = torch.zeros(1)
#     loss_count = 0
#
#     for i in range(N):
#         for j in range(N):
#             if i != j:
#                 diff = embeddings[i] - embeddings[j]
#                 dist_squared = torch.sum(diff ** 2)
#                 loss += (dist_squared - D[i, j] ** 2) ** 2
#                 loss_count += 1
#
#     return loss

