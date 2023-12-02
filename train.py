import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchsummary import summary
from matplotlib import pyplot as plt
from tqdm import tqdm

from model.lenet import LeNet
from model.alexnet import AlexNet
from model.resnet import ResNet
from model.mobilenet import MobileNet
from model.googlenet import GoogleNet
from data.data_handler import DataHandler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Model:
    def __init__(
        self,
        batch_size,
        epochs,
        learning_rate,
        dropout_rate,
        model_type,
        optimizer_type,
    ):
        # 固定种子
        set_seed(42)

        # 模型类型
        self.model_type = model_type

        # 超参数
        self.batch_size = batch_size
        self.eopchs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # 数据集
        self.data = DataHandler(self.batch_size)

        # 记录训练过程
        self.train_loss = []
        self.val_loss = []

        self.train_accuracy = []
        self.val_accuracy = []

        self.model: nn.Module

        self._set_model()

        if optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.learning_rate)
        elif optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.learning_rate)

    def train(self):
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []

        print("===== Traning Info =====")
        print("Device:", DEVICE)
        print(f"Model: {self.model_type}")
        print(f"Model Info:\n")
        summary(self.model)
        print("\n==== Starting Train ====")

        for epoch in range(1, self.eopchs + 1):
            self._one_epoch_train(epoch)
            self._eval_model(self.data.val_loader)

        self._eval_model(self.data.test_loader)

    def draw_plt(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.train_loss, label="Train Loss")
        plt.plot(self.val_loss, label="Test Loss")
        plt.title("Loss")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.train_accuracy, label="Train Accuracy")
        plt.plot(self.val_accuracy, label="Test Accuracy")
        plt.title("Accuracy")
        plt.legend()

        plt.show()

    def _set_model(self):
        if self.model_type == "lenet":
            self.model = LeNet().to(DEVICE)
        elif self.model_type == "alexnet":
            self.model = AlexNet(self.dropout_rate).to(DEVICE)
        elif self.model_type == "resnet":
            self.model = ResNet().to(DEVICE)
        elif self.model_type == "mobilenet":
            self.model = MobileNet().to(DEVICE)
        elif self.model_type == "googlenet":
            self.model = GoogleNet().to(DEVICE)

    def _one_epoch_train(self, epoch):
        self.model.train()

        correct = 0.0
        loss = 0.0

        for inputs, labels in tqdm(self.data.train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # 交叉熵损失函数
            loss_func = F.cross_entropy(outputs, labels)
            loss += loss_func.item()

            # 获取最大概率的预测结果
            predict = outputs.argmax(dim=1)
            correct += (predict == labels).sum().item()

            # 反向传播
            loss_func.backward()

            # 更新参数
            self.optimizer.step()

        accuracy = round(
            correct / len(self.data.train_loader.dataset), 6)
        loss /= len(self.data.train_loader.dataset)

        print(f"Train Epoch {epoch}")
        print(
            "Train set: \nLoss: {}, Accuracy: {}/{}({} %)".format(
                loss, correct, len(self.data.train_loader.dataset),
                100 * accuracy)
        )

        self.train_loss.append(loss)
        self.train_accuracy.append(accuracy)

    def _eval_model(self, target):
        self.model.eval()

        correct = 0.0
        loss = 0.0

        with torch.no_grad():
            for data, label in target:
                data, label = data.to(DEVICE), label.to(DEVICE)
                output = self.model(data)

                loss += F.cross_entropy(output, label).item()
                predict = output.argmax(dim=1)
                correct += (predict == label).sum().item()

        accuracy = round(
            correct / len(target.dataset), 6)
        loss /= len(target.dataset)

        if target == self.data.test_loader:
            print("Test set: \nLoss: {}, Accuracy: {}/{}({} %)\n".format(
                loss, correct, len(target.dataset),
                100 * accuracy
            ))
        else:
            print("Val set: \nLoss: {}, Accuracy: {}/{}({} %)\n".format(
                loss, correct, len(target.dataset),
                100 * accuracy
            ))
            self.val_loss.append(loss)
            self.val_accuracy.append(accuracy)
