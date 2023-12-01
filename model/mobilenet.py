import torch.nn as nn


def depthwise_separable_conv(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        ),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


# 构建MobileNet模型
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            depthwise_separable_conv(32, 64, 1),
            depthwise_separable_conv(64, 128, 2),
            depthwise_separable_conv(128, 128, 1),
            depthwise_separable_conv(128, 256, 2),
            depthwise_separable_conv(256, 256, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 10),  # 将MobileNet的输出调整为MNIST数据集的类别数
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
