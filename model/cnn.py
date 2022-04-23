import torch
import torch.nn as nn
from mnist_dataset import MnistDataSet
from torch.utils.data import DataLoader
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
    def __init__(self, args):
        self.args = args
        self.cuda_available = torch.cuda.is_available() and self.args.device == 'cuda0'
        self.model = MnistCNN()

        if self.cuda_available:
            self.model.to(torch.device("cuda:0"))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
        self.criterion = nn.CrossEntropyLoss()

        self.best_acc = 0.0
        self.cur_acc = 0.0


    # def __call__(self, x):
    #     self.model.eval()
    #     out = self.model(x)
    #     _, pred = torch.max(out, 1)
    #     return pred, out

    def train(self):
        train_dataset = MnistDataSet(self.args.train_data, self.args.train_label)
        train_data = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_works)
        validate_dataset = MnistDataSet(self.args.val_data, self.args.val_label)
        validate_data = DataLoader(dataset=validate_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_works)
        epoch_size = self.args.epochs
        for epoch in range(1, epoch_size + 1):
            print("\n-------------------------------------------------- epoch",
                  epoch, "-----------------------------------------------------")
            self.train_epoch(train_data)
            self.cur_acc = self.validate(validate_data)
            if self.cur_acc > self.best_acc:
                self.best_acc = self.cur_acc
                self.save("best_model.pth")

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

    def validate(self, validate_data):
        self.model.eval()
        data_count, val_loss, val_acc, batch_size = 0, 0.0, 0.0, validate_data.batch_size
        print("batch size: %d" % batch_size)
        start_time = time.time()
        time_elapsed = 0.0
        for batch, (inputs, targets) in enumerate(validate_data):
            if self.cuda_available:
                inputs, targets = inputs.to(torch.device("cuda:0")), targets.to(torch.device("cuda:0"))
            outputs = self.model(inputs)
            _, pred = torch.max(outputs, 1)
            loss = self.criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            val_acc += torch.sum(pred == targets.data)
            data_count += inputs.size(0)

            time_elapsed = time.time() - start_time
            print('\r' + "validating:" + ' %d/%d\t' % (batch + 1, len(validate_data)), end='')
            print('[' + '#' * int(batch / len(validate_data) * 30) + '-' * int(30 - (batch + 1) / len(validate_data) * 30)
                  + '] ' + "time elapsed: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
                  + ' loss: %.4lf' % (val_loss / data_count) + ' accuracy: %.4lf' % (val_acc / data_count), end='')
        print("\nvalidate accuracy: %.4lf" % (val_acc / len(validate_data.dataset)) +
              " time elapsed: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        return val_acc / len(validate_data.dataset)

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
