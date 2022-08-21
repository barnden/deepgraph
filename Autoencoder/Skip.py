# Autoencoder w/skip connections
from operator import concat
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.activation = {}

        def save_activation_map_as(name):
            def hook(_model, _inp, out):
                self.activation[name] = out.detach()

            return hook

        def concat_activation_with(name):
            def hook(_model, _inp, out):
                decoder_ftrs = out.detach()
                encoder_ftrs = self.activation[name]

                concat_ftrs = torch.cat((decoder_ftrs, encoder_ftrs), dim=1)

                return concat_ftrs

            return hook

        self.hooks = [
            self.encoder[1].register_forward_hook(save_activation_map_as('conv_1')),
            self.encoder[3].register_forward_hook(save_activation_map_as('conv_2')),
            self.decoder[1].register_forward_hook(concat_activation_with('conv_2')),
            self.decoder[7].register_forward_hook(concat_activation_with('conv_1')),
        ]

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
loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

### Main training loop
size = len(trainloader.dataset)

for epoch in range(2):
    print('-' * 32)
    print(f"Epoch {epoch + 1}")
    print('-' * 32)

    for batch, (X, _) in enumerate(trainloader, 1):
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

        X = x.to(device)[None]
        pred = net(X)

        pred_img = torch.squeeze(pred.cpu())
        out_images.append(pred_img)

        # plt.imsave(f"./EncDec/output_{i}.png", pred_img.permute(1, 2, 0).numpy())

    images = test_images + out_images

    grid = torchvision.utils.make_grid(images, nrow=5).permute(1, 2, 0).numpy()
    plt.imsave(f"./EncDec/skip_result_grid.png", grid)
