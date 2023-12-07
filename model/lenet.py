import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # self.resize = nn.Upsample(scale_factor=2, mode='bilinear')
        self.active = nn.Sigmoid()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = self.resize(x)
        x = self.active(self.conv1(x))
        x = self.maxpool1(x)
        x = self.active(self.conv2(x))
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)  # flatten
        x = self.active(self.fc1(x))
        x = self.active(self.fc2(x))
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)  # log(softmax(x))
        return output
