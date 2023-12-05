import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNetLarge(nn.Module):
    def __init__(self, dropout_rate):
        super(AlexNetLarge, self).__init__()
        self.features = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),

            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        output = F.log_softmax(x, dim=1)  # log(softmax(x))
        return output


class AlexNet(nn.Module):
    def __init__(self, dropout_rate):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(1, 24, kernel_size=5, padding=2),  # 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 28

            nn.Conv2d(24, 64, kernel_size=5, padding=2),  # 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 14

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 7
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        output = F.log_softmax(x, dim=1)  # log(softmax(x))
        return output
