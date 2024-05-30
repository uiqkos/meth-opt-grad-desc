import torch
from torch import optim, nn

from pytorch_models.model import LinearRegressionModel
from pytorch_models.testing import generate_linear_data, plot_train_and_test, loaders, load_pic, train_test_split, \
    generate_data

f = lambda a, b, c, d, e, x: (a * x + b * x ** 3 + c) + d * x ** 2 + 20 * e * torch.sin(x)
params = 5

X, y = generate_data(f, params, 1, 1000)

models = [
    LinearRegressionModel(
        f=f,
        parameters=params,
        optimizer=lambda x: optim.SGD(x, lr=0.0001, weight_decay=1e-5),
        criterion=nn.MSELoss(),
    ),
    LinearRegressionModel(
        f=f,
        parameters=params,
        optimizer=lambda x: optim.Adam(x, lr=0.0001, weight_decay=1e-5),
        criterion=nn.MSELoss(),
    ),
    LinearRegressionModel(
        f=f,
        parameters=params,
        optimizer=lambda x: optim.RMSprop(x, lr=0.0001, weight_decay=1e-5),
        criterion=nn.MSELoss(),
    ),
    LinearRegressionModel(
        f=f,
        parameters=params,
        optimizer=lambda x: optim.Adagrad(x, lr=0.0001, weight_decay=1e-5),
        criterion=nn.MSELoss(),
    ),

]

X_train, y_train, X_test, y_test = train_test_split(X, y)
train_loader, test_loader = loaders(X_train, y_train, X_test, y_test)

import matplotlib.pyplot as plt

for model in models:
    callback = lambda epoch, loss: print(f"Epoch [{epoch + 1}/{50}], Loss: {loss:.4f}")
    train_losses = model.fit(train_loader, epochs=100, callback=callback)
    test_loss = model.score(test_loader)

    plt.scatter(X_test.numpy(), model.predict(X_test).numpy(), label='Predictions: ' + str(model.optimizer.__class__.__name__))

plt.scatter(X_test.numpy(), y_test.numpy(), label='True Data')

plt.legend()
plt.show()
