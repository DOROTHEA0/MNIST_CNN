import torch
import torch.nn as nn
from mnist_dataset import MnistDataSet
from torch.utils.data import DataLoader


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 7), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.fc_layer(x)
        return x


def train_epoch(train_data, val_data):
    pass


def train(args):
    model = MnistCNN()
    train_data_set = MnistDataSet('data/kmnist-train-imgs.npz', 'data/kmnist-train-labels.npz')
    train_data_loader = DataLoader(dataset=train_data_set, batch_size=32, shuffle=True, num_workers=6)
    for i, (x, y) in enumerate(train_data_loader):
        print(model(x))
