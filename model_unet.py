import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import cv2 as cv


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(  # nn.Sequential 有序容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace 是否进行覆盖运算
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# kernel = [[0, -1, 0],
#           [-1, 5, -1],
#           [0, -1, 0]]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# kernel = torch.FloatTensor(kernel).expand(8,512,3,3)
# weight = torch.nn.Parameter(data=kernel, requires_grad=False).to(device=DEVICE)


# class GaussianBlurConv(nn.Module):
#     def __init__(self, channels):
#         super(GaussianBlurConv, self).__init__()
#         self.channels = channels
#         kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
#                   [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
#                   [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
#                   [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
#                   [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
#         kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
#         kernel = np.repeat(kernel, self.channels, axis=0)
#         self.weight = nn.Parameter(data=kernel, requires_grad=False)
#
#     def __call__(self, x):
#         x = nn.Conv2d(x.unsqueeze(0), self.weight, padding=2, groups=self.channels)
#         return x


def get_kernel():
    """
    See https://setosa.io/ev/image-kernels/
    """

    # k1:blur k2:outline k3:sharpen

    k1 = np.array([[0.0625, 0.125, 0.0625],
                   [0.125, 0.25, 0.125],
                   [0.0625, 0.125, 0.0625]])

    # Sharpening Spatial Kernel, used in paper
    k2 = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

    k3 = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

    return k1, k2, k3


def build_sharp_blocks(layer):
    """
    Sharp Blocks
    """
    # Get number of channels in the feature
    in_channels = layer.shape[1]
    # Get kernel
    _, w, _ = get_kernel()
    # Change dimension
    w = np.expand_dims(w, axis=0)
    # Repeat filter by in_channels times to get (H, W, in_channels)
    w = np.repeat(w, in_channels, axis=0)
    # Expand dimension
    w = np.expand_dims(w, axis=0)
    return torch.FloatTensor(w)


class UNET(nn.Module):
    def __init__(
            self, in_channels, out_channels, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()  # 将多个Module加入list，但不存在实质性顺序，参考python的list
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # decoder part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # encoder part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
