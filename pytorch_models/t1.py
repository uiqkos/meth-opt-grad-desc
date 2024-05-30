import torch

from pytorch_models.testing import load_pic
import torch

import torch


def loss(X, D):
    N = X.shape[0]
    total_loss = 0

    for i in range(N):
        for j in range(N):
            difference = X[i] - X[j]
            squared_distance = torch.dot(difference.T, difference)
            total_loss += (squared_distance - D[i, j] ** 2) ** 2

    return total_loss


def gradient_descent(X, D, learning_rate=0.0001, num_iterations=1000, batch_size=2):
    N = X.shape[0]
    X = X.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([X], lr=learning_rate)

    for iteration in range(num_iterations):
        perm = torch.randperm(N)
        total_loss = 0

        for batch_start in range(0, N, batch_size):
            batch_indices = perm[batch_start:batch_start + batch_size]
            X_batch = X[batch_indices]
            D_batch = D[batch_indices][:, batch_indices]

            optimizer.zero_grad()
            l = loss(X_batch, D_batch)
            l.backward()
            optimizer.step()
            total_loss += l.item()

        print(f'Iteration {iteration + 1}, Loss: {total_loss}')

    return X


def generate_distance_matrix(points):
    n = len(points)
    distance_matrix = torch.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = torch.norm(points[i] - points[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


# Пример использования
# points = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# D = generate_distance_matrix(points)
#
# # Начальные координаты
# X = torch.randn_like(points, requires_grad=True)
#
# # Градиентный спуск
# X_optimized = gradient_descent(X, D, learning_rate=0.01, num_iterations=1000, batch_size=2)
# print(X_optimized)

X, y = load_pic('../Modular shapes.csv')
num_points = len(X)

D = y

# Пример использования
# points = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# D = generate_distance_matrix(points)

# Начальные координаты
X = torch.zeros_like(X, requires_grad=True)

# Градиентный спуск
X_optimized = gradient_descent(X, D)
print(X_optimized)
