# Super-Resolution in LAB Space
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, lab2rgb


class Net(nn.Module):
    def __init__(self, in_channels=3):
        super(Net, self).__init__()

        # fmt: off
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

            nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 8, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        # fmt: on

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
            self.encoder[1].register_forward_hook(save_activation_map_as("conv_1")),
            self.encoder[3].register_forward_hook(save_activation_map_as("conv_2")),
            self.encoder[5].register_forward_hook(save_activation_map_as("conv_3")),
            self.decoder[1].register_forward_hook(concat_activation_with("conv_3")),
            self.decoder[3].register_forward_hook(concat_activation_with("conv_2")),
            self.decoder[5].register_forward_hook(concat_activation_with("conv_1")),
        ]

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode


def to_rgb(img_tensor: torch.Tensor) -> torch.Tensor:
    img_tensor = img_tensor.cpu()
    img = img_tensor.numpy()

    if img.ndim == 4:
        for b in range(img.shape[0]):
            img[b, 0, :, :] *= 100
            img[b] = lab2rgb(img[b], channel_axis=0)
    else:
        img[0, :, :] *= 100
        img = lab2rgb(img, channel_axis=0)

    img_tensor = torch.Tensor(img)

    return img_tensor


# Choose Nvidia GPU if available, CPU otherwise.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device.upper()} device.")

net = Net(in_channels=1).to(device)
print(net)


def ToCIELABTensor(pic):
    pic = np.array(pic)
    pic = rgb2lab(pic)

    # We divide lightness by 100 to get values in [0, 1] so that its representable
    # by the network; we must multiply by 100 before converting back to RGB
    pic[:, :, 0] /= 100

    tensor = torch.Tensor(pic)
    tensor = tensor.permute(2, 0, 1)

    return tensor


trainset = torchvision.datasets.STL10(
    root="./data", split="unlabeled", download=True, transform=ToCIELABTensor
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True)

testset = torchvision.datasets.STL10(
    root="./data", split="test", download=True, transform=ToCIELABTensor
)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)

for X, y in testloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")

    break

loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

size = len(trainloader.dataset)

# Set the below to False to skip training and use saved model "LAB_model.pth"
if False:
    for epoch in range(5):
        print("-" * 32)
        print(f"Epoch {epoch + 1}")
        print("-" * 32)

        for batch, (X, _) in enumerate(trainloader, 1):
            # Work only on lightness
            X = X[:, :1]
            y = X.to(device)

            X = F.interpolate(X, (48, 48), mode="bilinear")
            X = F.interpolate(X, (96, 96), mode="bilinear")
            X = X.to(device)

            pred = net(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:.7e} [{current:>6d}/{size:>6d}]")

    torch.save(net.state_dict(), "LAB_model.pth")
else:
    net.load_state_dict(torch.load("LAB_model.pth"))

test_loss = 0
with torch.no_grad():
    for (X, _) in testloader:
        X = X[:, :1]
        y = X.to(device)

        X = F.interpolate(X, (48, 48), mode="bilinear")
        X = F.interpolate(X, (96, 96), mode="bilinear")
        X = X.to(device)

        pred = net(X)
        test_loss += loss_fn(pred, y).item()

test_loss /= len(testloader)

print(f"Test Loss: {test_loss:>8f}\n")

with torch.no_grad():
    test_images = []
    out_images = []
    target_images = []

    for i in range(5):
        x, _ = testset[i]

        # FIXME: Here, we process the input to the SRNet as:
        #   RGB -> Upsample -> Downsample -> CIELAB -> Strip a*/b* -> SRNet
        # and in training, we perform:
        #   RGB -> CIELAB -> Strip a*/b* -> Upsample -> Downsample -> SRNet
        # Are these necessarily the same?

        X = F.interpolate(x[None], (48, 48), mode="bilinear")
        X = F.interpolate(X, (96, 96), mode="bilinear")

        test_img = torch.squeeze(to_rgb(X), dim=0)
        test_images.append(test_img)
        plt.imsave(f"./SR/LAB/input_{i}.png", test_img.permute(1, 2, 0).numpy())

        # Get rid of a* and b* channels
        X = X[:, :1]
        X = X.to(device)
        pred = net(X)

        pred_lightness = torch.squeeze(pred.cpu(), dim=0)

        # Concat predicted lightness with a* and b* channels from original input
        pred_img = torch.cat((pred_lightness[:1], x[1:]))
        pred_img = to_rgb(pred_img)

        out_images.append(pred_img)
        plt.imsave(f"./SR/LAB/output_{i}.png", pred_img.permute(1, 2, 0).numpy())

        target_images.append(to_rgb(x))

    images = test_images + out_images + target_images

    grid = torchvision.utils.make_grid(images, nrow=5).permute(1, 2, 0).numpy()
    plt.imsave(f"./SR/LAB/result_grid.png", grid)
