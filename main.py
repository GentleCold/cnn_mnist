from model.lenet import LeNet
from model.alexnet import AlexNet
from model.resnet import ResNet
from model.mobilenet import MobileNet
from data.data_handler import DataHandler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.test_loss = []

        self.train_accuracy = []
        self.test_accuracy = []

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
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        for epoch in range(1, self.eopchs + 1):
            self._one_epoch_train(epoch)
            self._test_model()

    def draw_plt(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.train_loss, label="Train Loss")
        plt.plot(self.test_loss, label="Test Loss")
        plt.title("Loss")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.train_accuracy, label="Train Accuracy")
        plt.plot(self.test_accuracy, label="Test Accuracy")
        plt.title("Accuracy")
        plt.legend()

        plt.show()

    def cross_validation(self, folds):
        kf = KFold(n_splits=folds, shuffle=True)

        for fold, (train_index, test_index) in enumerate(
            kf.split(self.data.train_data)
        ):
            print(f"Fold {fold+1}/{folds}")

            self._set_model()

            #  FIXME:: dataset error
            train_dataset = torch.utils.data.Subset(
                self.data.train_data, train_index)
            test_dataset = torch.utils.data.Subset(
                self.data.train_data, test_index)

            self.data.current_train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )

            self.data.current_test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )

            self.train()

    def _set_model(self):
        if self.model_type == "lenet":
            self.model = LeNet().to(DEVICE)
        elif self.model_type == "alexnet":
            self.model = AlexNet(self.dropout_rate).to(DEVICE)
        elif self.model_type == "resnet":
            self.model = ResNet().to(DEVICE)
        elif self.model_type == "mobilenet":
            self.model = MobileNet().to(DEVICE)

    def _one_epoch_train(self, epoch):
        self.model.train()

        correct = 0.0
        loss = 0.0

        for inputs, labels in tqdm(self.data.current_train_loader):
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
            correct / len(self.data.current_train_loader.dataset), 6)
        loss /= len(self.data.current_train_loader.dataset)

        print(f"Train Epoch {epoch}")
        print(
            "Train set: \nLoss: {}, Accuracy: {}/{}({} %)\n".format(
                loss, correct, len(self.data.current_train_loader.dataset),
                100 * accuracy)
        )

        self.train_loss.append(loss)
        self.train_accuracy.append(accuracy)

    def _test_model(self):
        self.model.eval()

        correct = 0.0
        loss = 0.0

        with torch.no_grad():
            for data, label in self.data.current_test_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                output = self.model(data)

                loss += F.cross_entropy(output, label).item()
                predict = output.argmax(dim=1)
                correct += (predict == label).sum().item()

        accuracy = round(
            correct / len(self.data.current_test_loader.dataset), 6)
        loss /= len(self.data.current_test_loader.dataset)

        print(
            "Test set: \nLoss: {}, Accuracy: {}/{}({} %)\n".format(
                loss, correct, len(self.data.current_test_loader.dataset),
                100 * accuracy)
        )

        self.test_loss.append(loss)
        self.test_accuracy.append(accuracy)


if __name__ == "__main__":
    print("device:", DEVICE)
    start_time = time.time()
    model = Model(64, 5, 0.001, 0.5, "mobilenet", "adamw")

    # model.cross_validation(5)

    model.train()
    end_time = time.time()

    print("training time: ", end_time - start_time)
    model.draw_plt()
