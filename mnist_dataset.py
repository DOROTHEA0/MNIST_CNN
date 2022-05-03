import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import random


class MnistDataSet(Dataset):
    """
    npz data set
    """
    def __init__(self, data_path, label_path, data_training=True):
        self.data = np.load(data_path)['arr_0']
        self.label = np.load(label_path)['arr_0']
        self.data_training = data_training

    def __getitem__(self, item):
        data = train_transform(self.data[item]) if self.data_training else val_transform(self.data[item])
        label = self.label[item]
        return data, label

    def __len__(self):
        return len(self.label)


def train_transform(img: np.ndarray) -> torch.Tensor:
    """
    image transform for train data
    random flip -> normalize 1 - -1 -> random crop -> toTensor
    :param img: input image type was ndarray
    :return: tensor training data
    """
    # random flip
    img_transform = cv2.flip(img, 1) if random.random() > 0.5 else img
    # normalize 0 - 1
    img_transform = torch.from_numpy(img_transform).float() / 255
    # add zero padding, padding size 4
    img_transform = F.pad(img_transform, [4, 4, 4, 4])
    # normalize -1 - 1
    img_transform = (img_transform - 0.5).true_divide(0.5)
    # random crop (28, 28)
    random_h, random_w = random.randint(0, 7), random.randint(0, 7)
    img_transform = img_transform[random_h: random_h + 28, random_w: random_w + 28]
    # un squeeze tensor (28, 28) H W -> (1, 28, 28) C H W
    img_transform = torch.unsqueeze(img_transform, 0)
    return img_transform


def val_transform(img: np.ndarray) -> torch.Tensor:
    """
    image transform for validate data and test data
    normalize 1 - -1 -> toTensor
    :param img: input image type was ndarray
    :return: tensor training data
    """
    # normalize 0 - 1
    img_transform = torch.from_numpy(img).float() / 255
    # normalize -1 - 1
    img_transform = (img_transform - 0.5).true_divide(0.5)
    # un squeeze tensor (28, 28) H W -> (1, 28, 28) C H W
    img_transform = torch.unsqueeze(img_transform, 0)
    return img_transform
