import torch

from speechbrain.nnet.pooling import Pooling2d, AdaptivePool
from speechbrain.nnet.CNN import Conv2d
from speechbrain.nnet.normalization import BatchNorm2d


class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()

        self.pool = AdaptivePool((1, 1))
        self.fc1 = Conv2d(in_channels=in_channels, out_channels=in_channels // reduction, kernel_size=1)
        self.fc2 = Conv2d(in_channels=in_channels // reduction, out_channels=in_channels, kernel_size=1)
        self.act = torch.nn.ReLU(inplace=True)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x, length=None):
        # NFTC -> N11C
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.sig(self.fc2(w))
        return w * x


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=stride, bias=False)
        self.bn1 = BatchNorm2d(input_size=channels)

        self.conv2 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, bias=False)
        self.bn2 = BatchNorm2d(input_size=channels)

        self.relu = torch.nn.ReLU(inplace=True)

        self.se = SEBlock(channels)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = torch.nn.Sequential(
                Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(input_size=channels)
            )

    def forward(self, x):
        """Basic block convolution

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.
        """

        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))

        # squeeze and excitation
        h = self.se(h)

        # shortcut
        x = h + self.shortcut(x)
        x = self.relu(x)

        return x


class AttentiveStatisticsPooling(torch.nn.Module):
    def __init__(self, in_channels, attention_channels=128):
        super(AttentiveStatisticsPooling, self).__init__()

        self.fc1 = Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.bn1 = BatchNorm2d(input_size=128)
        self.fc2 = Conv2d(in_channels=128, out_channels=in_channels, kernel_size=1)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        w = self.relu(self.bn1(self.fc1(x)))
        w = self.softmax(self.fc2(w))

        print(x.size())

        mu = torch.mean(w * x, dim=1, keepdim=True)
        sigma = torch.sum(w * (x ** 2) - mu ** 2, dim=1, keepdim=True).clamp(1e-8).sqrt()

        return torch.cat([mu, sigma], dim=-1)


class ResNet(torch.nn.Module):
    def __init__(self, block, blocks, freq_bins=80, feature_dim=512, skip_pooling=True):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, bias=False)
        self.bn1 = BatchNorm2d(input_size=64)
        self.relu = torch.nn.ReLU(inplace=True)

        self.pool = torch.nn.Identity() if skip_pooling else Pooling2d("max", (3, 3), stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, blocks[3], stride=2)

        flatten_feat_dim = 512 * (freq_bins // 16)

        self.asp = AttentiveStatisticsPooling(flatten_feat_dim)
        self.bn = BatchNorm2d(input_size=flatten_feat_dim * 2)
        self.fc = Conv2d(in_channels=flatten_feat_dim * 2, out_channels=feature_dim, kernel_size=1)

    def make_layer(self, block, channels, blocks, stride):
        layers = [block(self.in_channels, channels, stride=stride)]

        self.in_channels = channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(-1)

        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size(0), x.size(1), 1, x.size(2) * x.size(3))

        x = self.asp(x)

        x = self.fc(self.bn(x))

        return x


if __name__ == "__main__":
    from torchinfo import summary
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    summary(model, input_size=(4, 200, 80))
