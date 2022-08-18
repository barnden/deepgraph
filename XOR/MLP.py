# XOR with Multilayer Perceptron

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randrange

num_iter = 10000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Create linear layer
        self.linear_stack = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits


# Create an instance of the Net class
net = Net()

# Define the input and the target
XOR_input = torch.tensor(
    [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    requires_grad=True,
    dtype=torch.float)
XOR_target = torch.tensor([[-1], [1], [1], [-1]], dtype=torch.float)

# Define the loss and create your optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=5e-3, momentum=0.9)

# Main training loop
for i in range(num_iter):
    idx = randrange(4)
    X, y = XOR_input[idx], XOR_target[idx]

    # Prediction error
    pred = net(X)
    loss = loss_fn(pred, y)

    # Backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Printing results every 50 iterations
    if i % 50 == 0:
        loss = loss.item()
        print(f"loss: {loss:.5e} [{i} / {num_iter}]")

# Testing the network
test_loss, correct = 0, 0

with torch.no_grad():
    for i in range(num_iter):
        idx = randrange(4)
        X, y = XOR_input[idx], XOR_target[idx]

        pred = net(X)
        test_loss += loss_fn(pred, y).item()
        correct += torch.sign(pred)[0] == y

pct_correct = 100 * correct[0] / num_iter

# Testing the network
correct = 0

with torch.no_grad():
    for i in range(4):
        X, y = XOR_input[i], XOR_target[i]

        pred = net(X)
        correct += torch.sign(pred) == y

print()
print(f"Test Results:\n\t# Correct: {correct[0]}")
