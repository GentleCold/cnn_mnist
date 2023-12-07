import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.stride != 1:
        #     identity = nn.functional.avg_pool2d(identity, 2)

        if self.stride != 1 or x.shape[1] != out.shape[1]:
            identity = nn.functional.conv2d(
                identity, self.conv1.weight, stride=self.stride, padding=1)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, dropout_rate):
        super(ResNet, self).__init__()

        blocks = [64, 128, 256, 512]
        self.features = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(1, blocks[0], kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(blocks[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            BasicBlock(blocks[0], blocks[0]),
            BasicBlock(blocks[0], blocks[0]),

            BasicBlock(blocks[0], blocks[1], stride=2),
            BasicBlock(blocks[1], blocks[1]),

            BasicBlock(blocks[1], blocks[2], stride=2),
            BasicBlock(blocks[2], blocks[2]),

            BasicBlock(blocks[2], blocks[3], stride=2),
            BasicBlock(blocks[3], blocks[3]),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(blocks[3], 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        output = F.log_softmax(x, dim=1)  # log(softmax(x))
        return output
