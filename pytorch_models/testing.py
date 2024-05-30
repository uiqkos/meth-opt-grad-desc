from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split as sk_train_test_split


def generate_linear_data(dim: int, size: int = 1000) -> (torch.Tensor, torch.Tensor):
    X = torch.randn(size, dim)
    coefs = torch.randn(dim, 1)
    y = X @ coefs + torch.randn(size, 1) * 0.5

    return X, y


def generate_data(signature, params, dim, size=1000):
    X = torch.randn(size, dim) * 2
    coefs = torch.randn(params, 1)
    y = signature(*coefs, X) + torch.randn(size, 1) * 0.5

    return X, y


def train_test_split(
    X: torch.Tensor, y: torch.Tensor,
    test_size: int = 200
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    X_train, X_test, y_train, y_test = sk_train_test_split(X, y, test_size=test_size)

    return X_train, y_train, X_test, y_test


def loaders(X_train, y_train, X_test, y_test) -> (DataLoader, DataLoader):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def plot_train_and_test(train_losses, X_test, y_test, *predictions):
    import matplotlib.pyplot as plt

    # График обучения
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    for i, pr in enumerate(predictions):
        plt.scatter(X_test.numpy(), pr.numpy(), label='Predictions: ' + str(i+1))

    plt.scatter(X_test.numpy(), y_test.numpy(), label='True Data')

    plt.legend()
    plt.show()


def _generate_distance_matrix(points):
    n = len(points)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def load_pic(path: str):
    df = pd.read_csv(path, header=None)

    y = _generate_distance_matrix(df.to_numpy())
    X = df.to_numpy()

    # return (
    #     torch.tensor(X, dtype=torch.float32),
    #     torch.tensor(y, dtype=torch.float32)
    # )

    return X, y
