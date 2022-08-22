import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

### 1. Working with Data

# Retrieve training & test data
fashion_train = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

fashion_test = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

batch_size = 64

# Create iterator for Dataset by creating a DataLoader
train_loader = DataLoader(fashion_train, batch_size=batch_size)
test_loader = DataLoader(fashion_test, batch_size=batch_size)

if True:
    # Print dimensions of data in test dataset

    for X, y in test_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")

        break  # All data in dataset should have same shape

### 2. Creating Models

# Choose Nvidia GPU if available, CPU otherwise.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device.upper()} device.")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()

        # In PyTorch, define layers in __init__() and specify how data is passed in forward()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

### 3. Optimizing the Model Parameters

# In order to perform training, we need to define a loss function and an optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader: DataLoader, model: nn.Module, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n\tAccuracy: {100. * correct : >0.1f}%\n\tAvg. Loss: {test_loss : >8f}\n"
    )


epochs = 5

for t in range(epochs):
    print("-" * 32)
    print(f"Epoch {t + 1}")
    print("-" * 32)

    train(train_loader, model, loss_fn, optimizer)
    test(train_loader, model, loss_fn)

print("Done.")

### 4. Saving Models

# Most common way is to serialise the internal state dictionary that contains
# the model's parameters.

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

### 5. Loading Models

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()

x, y = fashion_test[0][0], fashion_test[0][1]

with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
