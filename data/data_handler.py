import torch
from torchvision import datasets, transforms


class DataHandler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.handle_data()

    def handle_data(self):
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

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False
        )
