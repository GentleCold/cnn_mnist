import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 激活函数
        self.active = nn.ReLU()
        # 第一个卷积层，1 -> 6, 卷积核大小5 * 5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 第一个池化层，窗口大小2 * 2
        self.maxpool1 = nn.MaxPool2d(2, 2)
        # 第二个卷积层，6 -> 16, 5 * 5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 第二个池化层，2 * 2
        self.maxpool2 = nn.MaxPool2d(2, 2)
        # 全连接层
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
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
