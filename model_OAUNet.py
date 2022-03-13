from torch.nn import functional as F
import torch.nn as nn
import torch
import torch.nn as nn
from module_attention import Attention_Module
import torchvision.transforms.functional as TF

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

batchNorm_momentum = 0.1


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(  # nn.Sequential 有序容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
            nn.Conv2d(in_channels, out_channels, 3, 1, 2, bias=False, dilation=2),
            nn.BatchNorm2d(out_channels, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),  # inplace 是否进行覆盖运算
            nn.Conv2d(out_channels, out_channels, 3, 1, 2, bias=False, dilation=2),
            nn.BatchNorm2d(out_channels, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class batchnorm_relu(nn.Module):
    def __init__(self, in_channels):
        super(batchnorm_relu, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels, momentum=batchNorm_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.bn(input)
        x = self.relu(x)

        return x


class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        """ Convolutional layer """
        self.b1 = batchnorm_relu(in_channels)
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_channels)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

        """ Shortcut Connection (Identity Mapping) """
        self.s = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):  # 这里交换了BatchNormalizationReLu和卷积的位置
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip


class Res2NetBottleneck(nn.Module):
    expansion = 1  # 残差块的输出通道数=输入通道数*exp+ansion

    def __init__(self, in_channels, out_channels, downsample=None, stride=1, scales=4, groups=1, se=False, norm_layer=True):
        # scales为残差块中使用分层的特征组数，groups表示其中3*3卷积层数量，SE模块和BN层
        super(Res2NetBottleneck, self).__init__()

        if out_channels % scales != 0:  # 输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')
        if norm_layer:  # BN层
            norm_layer = nn.BatchNorm2d

        bottleneck_out_channels = groups * out_channels
        self.scales = scales
        self.stride = stride
        self.downsample = downsample
        # 1*1的卷积层,在第二个layer时缩小图片尺寸
        self.iden = nn.Conv2d(in_channels, bottleneck_out_channels, kernel_size=1, stride=stride)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_out_channels, kernel_size=1, stride=stride)
        self.bn1 = norm_layer(bottleneck_out_channels)
        # 3*3的卷积层，一共有3个卷积层和3个BN层
        self.conv2 = nn.ModuleList([nn.Conv2d(bottleneck_out_channels // scales, bottleneck_out_channels // scales,
                                              kernel_size=3, stride=1, padding=1, groups=groups) for _ in
                                    range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_out_channels // scales) for _ in range(scales - 1)])
        # 1*1的卷积层，经过这个卷积层之后输出的通道数变成
        self.conv3 = nn.Conv2d(bottleneck_out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # SE模块
        self.se = SEModule(out_channels * self.expansion) if se else None

    def forward(self, x):
        identity = self.iden(x)

        # 1*1的卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # scales个(3x3)的残差分层架构
        xs = torch.chunk(out, self.scales, 1)  # 将x分割成scales块
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        # 1*1的卷积层
        out = self.conv3(out)
        out = self.bn3(out)

        # 加入SE模块
        if self.se is not None:
            out = self.se(out)
        # 下采样
        # if self.downsample:
        #     identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        # self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        up = nn.Conv2d(up.shape[1], up.shape[1] // 2, 1, 1)

        return up


class conv1(nn.Module):
    def __init__(self, feature):
        super(conv1, self).__init__()

        self.conv = nn.Conv2d(feature * 2, feature, 1, 1)

    def forward(self, x):

        return self.conv(x)

class oaunet(nn.Module):
    def __init__(
            self, in_channels, out_channels, features=[64, 128, 256, 512],
    ):
        super(oaunet, self).__init__()
        self.ups = nn.ModuleList()  # 将多个Module加入list，但不存在实质性顺序，参考python的list
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)


        # Down part of UNET
        for feature in features:
            if feature <= 128:
                self.downs.append(residual_block(in_channels, feature))
                in_channels = feature

            else:
                self.downs.append(Res2NetBottleneck(in_channels, feature))
                in_channels = feature


        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(conv1(feature))
            self.ups.append(Attention_Module(feature))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
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
        for idx in range(0, len(self.ups), 3):
            x = self.upsample(x)
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//3]

            # if x.shape != skip_connection.shape:
            #     x = TF.resize(x, size=skip_connection.shape[2:])
            if idx <= 6:
                skip_connection = self.ups[idx+1](skip_connection)
            else:

                kernel = [[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]]  # outline

                kernel = torch.FloatTensor(kernel).expand(len(skip_connection[0]), len(skip_connection[0]), 3, 3)
                # kernel = torch.FloatTensor(kernel).expand(len(skip_connection[1]), len(skip_connection[1]), 3, 3)
                weight = torch.nn.Parameter(data=kernel, requires_grad=False).to(device=DEVICE)
                skip_connection = torch.nn.functional.conv2d(skip_connection, weight, padding=1)
                nn.BatchNorm2d(skip_connection.shape[1], momentum=batchNorm_momentum)
                nn.ReLU(inplace=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+2](concat_skip)

        return self.final_conv(x)

