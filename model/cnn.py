import torch
import torch.nn as nn
from mnist_dataset import MnistDataSet, val_transform
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import cv2
import time


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
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.fc_layer(x)
        return x


class CNNTrainer:
    def __init__(self, args, model):
        self.args = args
        try:
            train_dataset = MnistDataSet(self.args.train_data, self.args.train_label, data_training=True)
            self.train_data = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                         num_workers=self.args.num_works)
        except AttributeError:
            self.train_data = None
        try:
            val_data = MnistDataSet(self.args.val_data, self.args.val_label, data_training=False)
        except AttributeError:
            val_data = MnistDataSet(self.args.test_data, self.args.test_label, data_training=False)
        self.val_data = DataLoader(dataset=val_data, batch_size=self.args.batch_size, shuffle=True,
                                   num_workers=self.args.num_works)

        self.cuda_available = torch.cuda.is_available() and 'cuda' in self.args.device
        self.model = model.cuda() if self.cuda_available else model

        try:
            lr = self.args.lr
        except AttributeError:
            lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        self.criterion = nn.CrossEntropyLoss()

        self.best_acc = 0.0
        self.cur_acc = 0.0

    def __call__(self, x):
        self.model.eval()
        out = self.model(x)
        _, pred = torch.max(out, 1)
        return pred, out

    def train(self):
        print('------------------------------------------------------------------------------------------------------'
              '----------')
        print('|                                              start training                                         '
              '         |')
        print('|                   use commend: "tensorboard --logdir runs" to start tensorboard service             '
              '         |')
        print('|                                      url: "http://localhost:6006/"                                  '
              '         |')
        print('------------------------------------------------------------------------------------------------------'
              '----------')
        epoch_size = self.args.epochs
        writer = SummaryWriter(log_dir='./runs')
        for epoch in range(1, epoch_size + 1):
            print("\n-------------------------------------------------- epoch",
                  epoch, "-----------------------------------------------------")
            train_loss, train_acc = self.train_epoch(self.train_data)
            val_loss, val_acc = self.validate()
            self.cur_acc = val_acc
            writer.add_scalar('accuracy/train', train_acc, epoch)
            writer.add_scalar('accuracy/validate', val_acc, epoch)
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/validate', val_loss, epoch)
            if self.cur_acc > self.best_acc:
                self.best_acc = self.cur_acc
                self.save("saved_models/best_model.pth")
        self.save('saved_models/last_model.pth')

    def train_epoch(self, train_data):
        self.model.train()
        data_count, train_loss, train_acc, batch_size = 0, 0.0, 0.0, train_data.batch_size
        print("batch size: %d" % batch_size)
        start_time = time.time()
        time_elapsed = 0.0
        for batch, (inputs, targets) in enumerate(train_data):
            if self.cuda_available:
                inputs, targets = inputs.to(torch.device("cuda:0")), targets.to(torch.device("cuda:0"))
            outputs = self.model(inputs)
            _, pred = torch.max(outputs, 1)
            loss = self.criterion(outputs, targets)

            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(pred == targets.data)
            data_count += inputs.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            time_elapsed = time.time() - start_time
            print('\r' + "training:" + ' %d/%d\t' % (batch + 1, len(train_data)), end='')
            print('[' + '#' * int(batch / len(train_data) * 30) + '-' * int(30 - (batch+1) / len(train_data) * 30)
                  + '] ' + "time elapsed: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60) +
                  ' loss: %.4lf' % (train_loss / data_count) + ' accuracy: %.4lf' %(train_acc / data_count), end='')
        print("\ntrain accuracy: %.4lf"%(train_acc / len(train_data.dataset)) +
              " time elapsed: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        return train_loss / data_count, train_acc / len(train_data.dataset)

    def validate(self):
        self.model.eval()
        data_count, val_loss, val_acc, batch_size = 0, 0.0, 0.0, self.val_data.batch_size
        print("batch size: %d" % batch_size)
        start_time = time.time()
        time_elapsed = 0.0
        for batch, (inputs, targets) in enumerate(self.val_data):
            if self.cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.model(inputs)
            _, pred = torch.max(outputs, 1)
            loss = self.criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            val_acc += torch.sum(pred == targets.data)
            data_count += inputs.size(0)

            time_elapsed = time.time() - start_time
            print('\r' + "validating:" + ' %d/%d\t' % (batch + 1, len(self.val_data)), end='')
            print('[' + '#' * int(batch / len(self.val_data) * 30) + '-' * int(30 - (batch + 1) / len(self.val_data) * 30)
                  + '] ' + "time elapsed: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
                  + ' loss: %.4lf' % (val_loss / data_count) + ' accuracy: %.4lf' % (val_acc / data_count), end='')
        print("\nvalidate accuracy: %.4lf" % (val_acc / len(self.val_data.dataset)) +
              " time elapsed: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        return val_loss / data_count, val_acc / len(self.val_data.dataset)

    def save(self, save_name):
        state = {
            'state_dict': self.model.state_dict(),
            'val_acc': self.cur_acc,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, save_name)

    def load(self, load_name):
        state = torch.load(load_name)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.cur_acc = state['val_acc']
        if self.cur_acc > self.best_acc:
            self.best_acc = self.cur_acc


def detect_image(img: str, model: MnistCNN, device: str) -> int:

    img = cv2.imread(img, 0)
    img = val_transform(img)
    img = torch.unsqueeze(img, 0)
    if torch.cuda.is_available() and 'cuda' in device:
        model = model.cuda()
        img = img.cuda()
    model.eval()
    out = model(img)
    _, pred = torch.max(out, 1)
    return int(pred)
