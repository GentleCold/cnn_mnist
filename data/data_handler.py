import torch
from torchvision import datasets, transforms


class DataHandler:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.handle_data()

    def handle_data(self):
        # 数据集处理
        pipline_train = transforms.Compose(
            [
                transforms.ToTensor(),
                # 标准化，有利于加快训练速度
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        pipline_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.train_data = datasets.MNIST(
            root="./data", train=True, download=True, transform=pipline_train
        )

        self.test_data = datasets.MNIST(
            root="./data", train=False, download=True, transform=pipline_test
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False
        )

        self.current_train_loader = self.train_loader
        self.current_test_loader = self.test_loader
