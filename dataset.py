import torch
from PIL import Image
from torch.utils.data import Dataset


class UAVdataset(Dataset):
    def __init__(self, train_list, transform=None):
        """
        train dataset, form a sequence every five frames with an end point frame.
        :param transform: torchvision.transforms.
        :param input_len: input sequence length (not containing the end point).
        """
        self.imgs = train_list
        self.transform = transform

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
        cur_img = cur_img.convert('RGB')
        cur_img = self.transform(cur_img)
        cur_img = cur_img.unsqueeze(dim=0)
        end_img = Image.open(self.imgs[index][1])
        end_img = end_img.convert('RGB')
        end_img = self.transform(end_img)
        end_img = end_img.unsqueeze(dim=0)
        input = torch.cat((cur_img, end_img), dim=0)
        return input
