import torch


# Define the custom gradient computation function
def compute_gradient(X, D):
    N = X.shape[0]
    X_expanded = X.unsqueeze(1) - X.unsqueeze(0)
    squared_distances = torch.sum(X_expanded ** 2, dim=2)

    difference = X_expanded
    grad = torch.sum(4 * (squared_distances.unsqueeze(2) - D ** 2).unsqueeze(2) * difference, dim=1)

    return grad


# Initialize data
N, D_in = 5, 3  # Number of points and dimensionality
X = torch.randn(N, D_in, requires_grad=True)  # Example input tensor
D = torch.rand(N, N)  # Example distance matrix

# Initialize optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD([X], lr=learning_rate)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Compute the custom gradient
    grad = compute_gradient(X, D)

    # Manually set the gradients
    X.grad = grad

    # Step the optimizer
    optimizer.step()

    # Optional: print loss or any other metrics to monitor training
    loss = torch.sum((torch.cdist(X, X) - D) ** 2)
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

print("Training completed.")
