import torch
from torch import optim, nn

from pytorch_models.model import LinearRegressionModel
from pytorch_models.testing import generate_linear_data, plot_train_and_test, loaders, load_pic, train_test_split, generate_data

f = lambda a, b, c, d, e, x: (a*x + b * x ** 3 + c) + d * x ** 2 + 20 * e * torch.sin(x)
params = 5

# X, y = generate_linear_data(1, 1000)
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
        optimizer=lambda x: optim.SGD(x, lr=0.0001, weight_decay=1e-3),
        criterion=nn.MSELoss(),
    ),
    LinearRegressionModel(
        f=f,
        parameters=params,
        optimizer=lambda x: optim.SGD(x, lr=0.001, weight_decay=1e-3),
        criterion=nn.MSELoss(),
    ),
    # LinearRegressionModel(
    #     optimizer=lambda x: optim.SGD(x, lr=0.01, momentum=0.9, dampening=0, nesterov=True),
    #     criterion=nn.MSELoss(),
    # ),

]

X_train, y_train, X_test, y_test = train_test_split(X, y)
train_loader, test_loader = loaders(X_train, y_train, X_test, y_test)

prs = []

for model in models:
    callback = lambda epoch, loss: print(f"Epoch [{epoch + 1}/{50}], Loss: {loss:.4f}")
    train_losses = model.fit(train_loader, epochs=3, callback=callback)
    test_loss = model.score(test_loader)
    prs.append(model.predict(X_test))

plot_train_and_test([], X_test, y_test, *prs)
