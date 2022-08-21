# Super-Resolution in RGB Space
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.skip = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, x):
        return self.skip(x)


class Net(nn.Module):
    def __init__(self, in_channels=3):
        super(Net, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            SkipConnection(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            SkipConnection(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            SkipConnection(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(8, in_channels, 4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.activation = {}

        def save_activation_map_as(name):
            def hook(_module, _inp, out):
                self.activation[name] = out.detach()

            return hook

        def concat_activation_with(name):
            def hook(_module, _inp, out):
                decoder_ftrs = out.detach()
                encoder_ftrs = self.activation[name]

                concat_ftrs = torch.cat((decoder_ftrs, encoder_ftrs), dim=1)

                return concat_ftrs

            return hook

        self.hooks = [
            self.encoder[1].register_forward_hook(save_activation_map_as('conv_1')),
            self.encoder[3].register_forward_hook(save_activation_map_as('conv_2')),
            self.encoder[5].register_forward_hook(save_activation_map_as('conv_3')),
            self.decoder[1].register_forward_hook(concat_activation_with('conv_3')),
            self.decoder[5].register_forward_hook(concat_activation_with('conv_2')),
            self.decoder[9].register_forward_hook(concat_activation_with('conv_1')),
        ]

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode

# Choose Nvidia GPU if available, CPU otherwise.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device.upper()} device.")

net = Net().to(device)
print(net)

transform = transforms.ToTensor()

trainset = torchvision.datasets.STL10(root='./data', split='unlabeled', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True)

testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)

for X, y in testloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")

    break

loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

size = len(trainloader.dataset)

# Set the below to False to skip training and use saved model "RGB_model.pth"
if True:
    for epoch in range(6):
        print('-' * 32)
        print(f"Epoch {epoch + 1}")
        print('-' * 32)

        for batch, (X, _) in enumerate(trainloader, 1):
            y = X.to(device)
            X = F.interpolate(X, (48, 48), mode="bilinear").to(device)

            pred = net(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:.7e} [{current:>5d}/{size:>5d}]")

    torch.save(net.state_dict(), "RGB_model.pth")
else:
    net.load_state_dict(torch.load("RGB_model.pth"))

test_loss = 0
with torch.no_grad():
    for (X, _) in testloader:
        y = X.to(device)
        X = F.interpolate(X, (48, 48), mode="bilinear").to(device)

        pred = net(X)
        test_loss += loss_fn(pred, y).item()

test_loss /= len(testloader)

print(f"Test Loss: {test_loss:>8f}\n")

with torch.no_grad():
    test_images = []
    out_images = []

    for i in range(5):
        x, _ = testset[i]

        X = F.interpolate(x[None], (48, 48), mode="bilinear")

        test_img = torch.squeeze(X)

        test_img_pad = F.pad(test_img, (24, 24, 24, 24), "constant", 0)
        test_images.append(test_img_pad)

        plt.imsave(f"./SR/input_{i}.png", test_img.permute(1, 2, 0).numpy())

        X = X.to(device)
        pred = net(X)

        pred_img = torch.squeeze(pred.cpu())
        out_images.append(pred_img)

        plt.imsave(f"./SR/output_{i}.png", pred_img.permute(1, 2, 0).numpy())

    images = test_images + out_images

    grid = torchvision.utils.make_grid(images, nrow=5).permute(1, 2, 0).numpy()
    plt.imsave(f"./SR/result_grid.png", grid)
