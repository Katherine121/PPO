import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UAVdataset(Dataset):
    def __init__(self, path_list):
        """
        train dataset, form a sequence every five frames with an end point frame.
        :param transform: torchvision.transforms.
        :param input_len: input sequence length (not containing the end point).
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.imgs = path_list

    def __len__(self):
        """
        return the length of the dataset.
        :return:
        """
        return len(self.imgs)

    def __getitem__(self, index):
        """
        read the image sequence, angle sequence and label corresponding to the index in the dataset.
        :param index: index of self.imgs.
        :return: frame sequence, angle sequence, the direction angle.
        """
        cur_img = Image.open(self.imgs[index][0])
        cur_img = self.val_transform(cur_img)
        cur_img = cur_img.unsqueeze(dim=0)
        end_img = Image.open(self.imgs[index][1])
        end_img = self.val_transform(end_img)
        end_img = end_img.unsqueeze(dim=0)
        new_state = torch.cat((cur_img, end_img), dim=0)
        return new_state
