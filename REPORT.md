[TOC]

# 图像分类及经典CNN实现实验报告

## 项目文件

```shell
.
├── data
│   ├── MNIST           // MNIST数据集
│   └── data_handler.py // 数据处理文件
├── main.py             // 主运行文件
├── model               // 模型文件
│   ├── alexnet.py
│   ├── googlenet.py
│   ├── lenet.py
│   ├── mobilenet.py
│   └── resnet.py
├── README.md           // 项目用法介绍
├── REPORT.md           // 项目实验报告
├── REPORT.pdf          // 项目实验报告
├── analyse.ipynb       // 实验分析文件
└── train.py            // 模型训练文件
```

## 项目概述

介绍：图像分类是计算机视觉领域中的重要任务之一。它涉及将图像分为不同的类别，例如识别手写数字、识别物体或人脸等。本实验旨在训练经典的卷积神经网络（CNN）模型，使其能够准确识别手写数字。

数据集：本实验使用的数据集是MNIST，包含60000张训练图像和10000张测试图像，每张图像都是一个手写数字（0-9）的灰度图像，大小为28x28像素。

## 实验过程

### 所遇bug在此处统一标出

[bug1: 数据集维度不匹配](#bug1)

### 一、处理数据集

#### 1. 加载MNIST数据集

按照8：2的比例固定划分训练集为训练集和验证集

数据比例为 `train : val : test = 48000 : 12000 : 10000`

由于MNIST数据集的经典性，直接利用`torchvision.datasets`中的`MNIST`类加载即可

```python
# 下载数据集并转为Tensor
self.train_data = datasets.MNIST(
    root="./data", train=True, download=True,
    transform=transforms.ToTensor()
)

self.test_data = datasets.MNIST(
    root="./data", train=False, download=True,
    transform=transforms.ToTensor()
)

# 划分训练集和验证集(8 : 2)
train_size = int(0.8 * len(self.train_data))
val_size = len(self.train_data) - train_size

self.train_data, self.val_data = torch.utils.data.random_split(
    self.train_data, [train_size, val_size])

# 按batch_size加载数据集
self.train_loader = torch.utils.data.DataLoader(
    self.train_data, batch_size=self.batch_size, shuffle=True
)

self.val_loader = torch.utils.data.DataLoader(
    self.val_data, batch_size=self.batch_size, shuffle=True
)

self.test_loader = torch.utils.data.DataLoader(
    self.test_data, batch_size=self.batch_size, shuffle=False
)
```

MNIST的部分图像如下，可见图像经过较好的降噪处理

![](./imgs/image-20231205001236.png =50%x50%) 

### 二、模型实现

#### 1. LeNet

##### 1.1 模型结构

LeNet是一种经典的CNN模型，最早由Yann LeCun等人于1998年提出，用于手写数字识别任务。

原始模型结构如下：

![](./imgs/image-20231205220733.png =50%x50%)

1. 输入层：接受输入图像的像素值，此处为灰度图像，输入大小为32x32像素。

2. 卷积层1：包含6个卷积核，每个卷积核的大小为5x5，步长为1，输入通道为1（灰度图像）。

3. 池化层1：使用2x2的最大池化操作，步长为2。

4. 卷积层2：包含16个卷积核，每个卷积核的大小为5x5，步长为1。

5. 池化层2：使用2x2的最大池化操作，步长为2。

6. 全连接层1：包含120个神经元。

7. 全连接层2：包含84个神经元。

8. 输出层: 用于输出最终的分类结果，包含10个神经元（对应10个类别）。

* 激活函数：在每个卷积层和全连接层之后，通常使用非线性激活函数来引入非线性性质。在LeNet中，常用的激活函数是sigmoid函数和tanh函数

* 不完全连接：池化层1与卷积层2采用不完全连接，而pytorch本身的随机化策略即可实现这个特点

##### 1.2 模型实现

使用pytorch实现模型

<span id="bug1">*BUG1:*</span>

    传统的lenet使用32*32的图像输入，而MNIST数据集是28*28的图像输入

    解决办法：修改全连接层的输入维度16*5*5为16*4*4

    原因：根据模型架构可以推算特征维度

    28 * 28 --卷积--> 24 * 24 --池化--> 12 * 12 --卷积--> 8 * 8 --池化--> 4 * 4

    另一解决办法是resize图像大小，见alexnet部分

代码实现如下，使用Sigmoid激活函数，使用最大池化，最终使用softmax(映射到概率)输出:

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.active = nn.Sigmoid()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
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
```

打印出的模型信息如下：

![](./imgs/image-20231205222141.png =50%x50%)

#### 2. AlexNet

##### 2.1 模型结构

AlexNet是另一种经典的卷积神经网络模型，于2012年在ImageNet图像分类竞赛上取得了巨大成功。

原始模型结构如下：

![](./imgs/image-20231205234646.png =50%x50%)

论文使用两个CPU并行化训练，实际维度需要相加

1. 输入层：接受输入图像数据，大小为224x224x3。

2. 卷积层1：使用96个11x11大小的卷积核，步幅为4

3. 池化层1：使用3x3大小的最大池化窗口，步幅为2。

4. 卷积层2：使用256个5x5大小的卷积核，步幅为1。

5. 池化层2：使用3x3大小的最大池化窗口，步幅为2。

6. 卷积层3：使用384个3x3大小的卷积核，步幅为1。

7. 卷积层4：同样使用384个3x3大小的卷积核，步幅为1。

8. 卷积层5：使用256个3x3大小的卷积核，步幅为1。

9. 池化层3：使用3x3大小的最大池化窗口，步幅为2。

10. 全连接层1：包含4096个神经元

11. 全连接层2：同样包含4096个神经元

12. 输出层：包含1000个神经元(1000类别)

* 使用Relu作为激活函数，并使用dropout来防止过拟合

##### 2.2 模型实现

AlexNet使用更深的网络，训练需要花费更长的时间，另一方面MNIST仅为28x28的尺寸，此处使用`nn.Upsample`线性缩放图片为224x224

模型参数如下，参数数量达到了千万级别

![](./imgs/image-20231206004531.png =50%x50%)

(训练一步需要约40分钟，详细实现见model.alexnet.AlexNetLarge)

为方便后续实验对比，缩减网络结构

具体改动为，将原模型的卷积核从11x11改为5x5，步长改为1，将卷积核数均除以8，神经元除以8，图片尺寸缩放为2倍

代码如下：

```python
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
```

模型信息如下，参数数量减小为百万级别

![](./imgs/image-20231206004937.png =50%x50%)

#### 3. ResNet

##### 3.1 模型结构

##### 3.2 模型实现

