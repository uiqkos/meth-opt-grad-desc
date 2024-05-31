import time

import numpy as np
import torch
from torch import optim
from distance_numpy import gradient_descent

from pytorch_models.distance_model import DistanceEmbeddingModel, loss_fn
from pytorch_models.testing import load_pic, plot_train_and_test
from IPython.display import display, clear_output

embedding_dim = 2
learning_rate = 0.001
num_iterations = 20

X, y = load_pic('../Modular shapes.csv')
num_points = len(X)

D = y

import matplotlib.pyplot as plt

ll = []

def callback(epoch, loss, points):
    print(points[:1])
    print(f"Epoch [{epoch + 1}/{num_iterations}], Loss: {loss:.4f}")
    ll.append(points[:1])

    clear_output(wait=True)

    plt.scatter(X[:, 0], X[:, 1], color='r')
    plt.scatter(points[:, 0], points[:, 1], alpha=epoch/100, color='g')

    plt.legend()
    plt.savefig(f'../images/1/epoch_{epoch}.png')

    time.sleep(0.1)
#
#
# model = DistanceEmbeddingModel(
#     optimizer=lambda params: optim.SGD(params, lr=0.001),
#     criterion=loss_fn,
#     num_points=num_points, embedding_dim=embedding_dim)
#
# losses = model.fit(D, epochs=num_iterations, callback=callback)
#
# print("Final Embeddings:")
# print(final_embeddings)
X_ = np.random.randn(*X.shape)
# plt.scatter(X_[:, 0], X_[:, 1], label='Predictions')
final_embeddings = gradient_descent(X_, D, learning_rate, 20)
plt.scatter(X[:, 0], X[:, 1], label='True Data', color='red')
plt.scatter(final_embeddings[:, 0], final_embeddings[:, 1], label='Predictions')

# for i, x in enumerate(ll):
#     plt.scatter(x[:, 0], x[:, 1], alpha=i / len(ll))
#     print(i, x)


plt.legend()
plt.show()
