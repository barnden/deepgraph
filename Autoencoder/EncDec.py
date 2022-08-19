# Autoencoder
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Using CIFAR10, H_in=W_in=32

        # Assignment states: each layer has square stride, kernel, and padding.

        # Conv2d: H_out=W_out=floor(1 + (32 + stride * padding - kernel))
        # ConvTranspose2d: H_out=W_out=31 * stride - 2 * padding + kernel + output_padding
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, 2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, 2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, 4, 2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode

### Create an instance of the Net class

# Choose Nvidia GPU if available, CPU otherwise.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device.upper()} device.")

net = Net().to(device)
print(net)

## Loading the training and test sets

# Converting the images for PILImage to tensor, so they can be accepted as the input to the network
transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)

if True:
    # Print dimensions of data in test dataset

    for X, y in testloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")

        break # All data in dataset should have same shape

### Define the loss and create your optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(net.parameters())

### Main training loop
size = len(trainloader.dataset)

for epoch in range(6):
    print('-' * 32)
    print(f"Epoch {epoch + 1}")
    print('-' * 32)

    for batch, (X, _) in enumerate(trainloader):
        ## Getting the input and the target from the training set
        X, y = X.to(device), X.to(device)

        # Compute prediction error
        pred = net(X)
        loss = loss_fn(pred, y)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:.7e} [{current:>5d}/{size:>5d}]")


### Testing the network on 10,000 test images and computing the loss

test_loss = 0
with torch.no_grad():
    for (X, _) in testloader:
        X, y = X.to(device), X.to(device)

        pred = net(X)
        test_loss += loss_fn(pred, y).item()

test_loss /= len(testloader)

print(f"Test Loss: {test_loss:>8f}\n")

### Displaying or saving the results as well as the ground truth images for the first five images in the test set

with torch.no_grad():
    test_images = []
    out_images = []

    for i in range(5):
        x, _ = testset[i]

        test_img = x.cpu()
        test_images.append(test_img)

        # plt.imsave(f"./EncDec/input_{i}.png", test_img.permute(1, 2, 0).numpy())

        X = x.to(device)
        pred = net(X)

        pred_img = pred.cpu()
        out_images.append(pred.cpu())

        # plt.imsave(f"./EncDec/output_{i}.png", pred_img.permute(1, 2, 0).numpy())

    images = test_images + out_images

    grid = torchvision.utils.make_grid(images, nrow=5).permute(1, 2, 0).numpy()
    plt.imsave(f"./EncDec/result_grid.png", grid)
