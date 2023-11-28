# import os
#
# import numpy as np
# import torch
# from PIL import Image
# from torch.utils.data import Dataset
#
#
# class UAVdataset(Dataset):
#     def __init__(self, train_path):
#         """
#         train dataset, form a sequence every five frames with an end point frame.
#         :param transform: torchvision.transforms.
#         :param input_len: input sequence length (not containing the end point).
#         """
#         batch_list = os.listdir(train_path)
#         batch_list.sort(key=lambda x: int(x[:-3]))
#         self.imgs = batch_list
#
#     def __len__(self):
#         """
#         return the length of the dataset.
#         :return:
#         """
#         return len(self.imgs)
#
#     def __getitem__(self, index):
#         """
#         read the image sequence, angle sequence and label corresponding to the index in the dataset.
#         :param index: index of self.imgs.
#         :return: frame sequence, angle sequence, the direction angle.
#         """
#         cur_img = torch.load(self.imgs[index][0])
#         end_img = torch.load(self.imgs[index][1])
#         new_state = torch.cat((cur_img, end_img), dim=0)
#         return new_state
