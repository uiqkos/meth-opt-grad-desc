import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Генерация набора данных
def generate_data(num_samples=1000):
    X = torch.randn(num_samples, 1)
    y = 3 * X + 2 + torch.randn(num_samples, 1) * 0.5
    return X, y


# Создание модели
class LinearRegressionModel(nn.Module):
    def __init__(self, in_size=1, out_size=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        return self.linear(x)


# Подготовка данных
X_train, y_train = generate_data()
X_test, y_test = generate_data(200)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Инициализация модели, критерия и оптимизатора
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Обучение модели
num_epochs = 50
train_losses = []

for epoch in range(num_epochs):
    model.train()
    loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    train_losses.append(loss.item())
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Тестирование модели
model.eval()
test_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

# График обучения
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Визуализация предсказаний
with torch.no_grad():
    predictions = model(X_test)

plt.scatter(X_test.numpy(), y_test.numpy(), label='True Data')
plt.scatter(X_test.numpy(), predictions.numpy(), label='Predictions', color='r')
plt.legend()
plt.show()
