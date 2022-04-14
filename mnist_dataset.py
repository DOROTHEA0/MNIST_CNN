import torch
from torch.utils.data import Dataset
import numpy as np


class MnistDataSet(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)['arr_0']
        self.label = np.load(label_path)['arr_0']

    def __getitem__(self, item):
        data = torch.unsqueeze(torch.from_numpy(self.data[item]).float() / 255, 0)
        label = self.label[item]
        return data, label

    def __len__(self):
        return len(self.label)
