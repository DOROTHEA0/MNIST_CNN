import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import random


class MnistDataSet(Dataset):
    """

    """
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)['arr_0']
        self.label = np.load(label_path)['arr_0']

    def __getitem__(self, item):
        data = transform(self.data[item])
        label = self.label[item]
        return data, label

    def __len__(self):
        return len(self.label)


def transform(img: np.ndarray) -> torch.Tensor:
    """

    :param img: input image type was ndarray
    :return: tensor training data
    """
    img_transform = cv2.flip(img, 1) if random.random() > 0.5 else img
    img_transform = torch.from_numpy(img_transform).float() / 255
    img_transform = F.pad(img_transform, [4, 4, 4, 4])
    img_transform = (img_transform - 0.5).true_divide(0.5)
    random_h, random_w = random.randint(0, 7), random.randint(0, 7)
    img_transform = img_transform[random_h: random_h + 28, random_w: random_w + 28]
    img_transform = torch.unsqueeze(img_transform, 0)
    return img_transform
