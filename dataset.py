import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from cor import IMG_SIZE


class UAVdataset(Dataset):
    def __init__(self, path_list, buffer_actions, buffer_logprobs, buffer_advantages, rewards):
        """
        train dataset.
        :param path_list:
        :param buffer_actions:
        :param buffer_logprobs:
        :param buffer_advantages:
        :param rewards:
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.val_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])
        self.imgs = path_list
        self.buffer_actions = buffer_actions
        self.buffer_logprobs = buffer_logprobs
        self.buffer_advantages = buffer_advantages
        self.rewards = rewards

    def __len__(self):
        """
        return the length of the dataset.
        :return:
        """
        return len(self.imgs)

    def __getitem__(self, index):
        """
        get a state_t, action_t, log(p(a|s)), advantage_t, reward_t
        :param index:
        :return:
        """
        cur_img = Image.open(self.imgs[index][0])
        cur_img = self.val_transform(cur_img)
        cur_img = cur_img.unsqueeze(dim=0)
        end_img = Image.open(self.imgs[index][1])
        end_img = self.val_transform(end_img)
        end_img = end_img.unsqueeze(dim=0)
        new_state = torch.cat((cur_img, end_img), dim=0)
        return new_state, torch.tensor(self.buffer_actions[index], dtype=torch.float32), \
               torch.tensor(self.buffer_logprobs[index], dtype=torch.float32), \
               self.buffer_advantages[index], \
               self.rewards[index]


class DDPGTD3dataset(Dataset):
    def __init__(self, path_list, buffer_actions, buffer_next_states, buffer_rewards, buffer_not_dones):
        """
        train dataset, form a sequence every five frames with an end point frame.
        :param transform: torchvision.transforms.
        :param input_len: input sequence length (not containing the end point).
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.val_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])
        self.imgs = path_list
        self.buffer_actions = buffer_actions
        self.buffer_next_states = buffer_next_states
        self.buffer_rewards = buffer_rewards
        self.buffer_not_dones = buffer_not_dones

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

        cur_img = Image.open(self.buffer_next_states[index][0])
        cur_img = self.val_transform(cur_img)
        cur_img = cur_img.unsqueeze(dim=0)
        end_img = Image.open(self.buffer_next_states[index][1])
        end_img = self.val_transform(end_img)
        end_img = end_img.unsqueeze(dim=0)
        next_state = torch.cat((cur_img, end_img), dim=0)

        return new_state, torch.tensor(self.buffer_actions[index], dtype=torch.float32), \
               next_state, \
               torch.tensor(self.buffer_rewards[index], dtype=torch.float32), \
               torch.tensor(self.buffer_not_dones[index], dtype=torch.int64),


class SURFdataset(Dataset):
    def __init__(self, path_list, buffer_actions, buffer_logprobs, buffer_advantages, rewards):
        """
        train dataset, form a sequence every five frames with an end point frame.
        :param transform: torchvision.transforms.
        :param input_len: input sequence length (not containing the end point).
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.val_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])

        self.surf = cv2.xfeatures2d.SURF_create()
        self.imgs = path_list
        self.buffer_actions = buffer_actions
        self.buffer_logprobs = buffer_logprobs
        self.buffer_advantages = buffer_advantages
        self.rewards = rewards

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
        # 读取图像
        cur_image = cv2.imread(self.imgs[index][0], cv2.IMREAD_GRAYSCALE)
        # 检测关键点和计算SURF特征
        cur_keypoints, cur_descriptors = self.surf.detectAndCompute(cur_image, None)
        # 将SURF特征转换为PyTorch张量
        if cur_descriptors is None:
            cur_surf_features = torch.zeros((10, 64), dtype=torch.float32)
        else:
            cur_surf_features = torch.from_numpy(cur_descriptors).float()

        # 读取图像
        end_image = cv2.imread(self.imgs[index][1], cv2.IMREAD_GRAYSCALE)
        # 检测关键点和计算SURF特征
        end_keypoints, end_descriptors = self.surf.detectAndCompute(end_image, None)
        # 将SURF特征转换为PyTorch张量
        if end_descriptors is None:
            end_surf_features = torch.zeros((10, 64), dtype=torch.float32)
        else:
            end_surf_features = torch.from_numpy(end_descriptors).float()

        if cur_surf_features.size(0) >= 10:
            random_indices1 = torch.randperm(cur_surf_features.size(0))[:10]
            cur_surf_features = cur_surf_features[random_indices1, :].flatten().unsqueeze(dim=0)
        else:
            x = cur_surf_features
            while cur_surf_features.size(0) < 10:
                cur_surf_features = torch.cat((cur_surf_features, x), dim=0)
            cur_surf_features = cur_surf_features[:10, :].flatten().unsqueeze(dim=0)

        if end_surf_features.size(0) >= 10:
            random_indices2 = torch.randperm(end_surf_features.size(0))[:10]
            end_surf_features = end_surf_features[random_indices2, :].flatten().unsqueeze(dim=0)
        else:
            y = end_surf_features
            while end_surf_features.size(0) < 10:
                end_surf_features = torch.cat((end_surf_features, y), dim=0)
            end_surf_features = end_surf_features[:10, :].flatten().unsqueeze(dim=0)

        new_state = torch.cat((cur_surf_features, end_surf_features), dim=0)
        return new_state, torch.tensor(self.buffer_actions[index], dtype=torch.float32), \
               torch.tensor(self.buffer_logprobs[index], dtype=torch.float32), \
               self.buffer_advantages[index], \
               self.rewards[index]


def transform_surf_features(state):
    surf = cv2.xfeatures2d.SURF_create()
    # 读取图像
    cur_image = cv2.imread(state[0], cv2.IMREAD_GRAYSCALE)
    # 检测关键点和计算SURF特征
    cur_keypoints, cur_descriptors = surf.detectAndCompute(cur_image, None)
    # 将SURF特征转换为PyTorch张量
    if cur_descriptors is None:
        cur_surf_features = torch.zeros((10, 64), dtype=torch.float32)
    else:
        cur_surf_features = torch.from_numpy(cur_descriptors).float()
    # print(cur_surf_features.size())

    # 读取图像
    end_image = cv2.imread(state[1], cv2.IMREAD_GRAYSCALE)
    # 检测关键点和计算SURF特征
    end_keypoints, end_descriptors = surf.detectAndCompute(end_image, None)
    # 将SURF特征转换为PyTorch张量
    if end_descriptors is None:
        end_surf_features = torch.zeros((10, 64), dtype=torch.float32)
    else:
        end_surf_features = torch.from_numpy(end_descriptors).float()
    # print(end_surf_features.size())

    if cur_surf_features.size(0) >= 10:
        random_indices1 = torch.randperm(cur_surf_features.size(0))[:10]
        cur_surf_features = cur_surf_features[random_indices1, :].flatten().unsqueeze(dim=0)
    else:
        x = cur_surf_features
        while cur_surf_features.size(0) < 10:
            cur_surf_features = torch.cat((cur_surf_features, x), dim=0)
        cur_surf_features = cur_surf_features[:10, :].flatten().unsqueeze(dim=0)

    if end_surf_features.size(0) >= 10:
        random_indices2 = torch.randperm(end_surf_features.size(0))[:10]
        end_surf_features = end_surf_features[random_indices2, :].flatten().unsqueeze(dim=0)
    else:
        y = end_surf_features
        while end_surf_features.size(0) < 10:
            end_surf_features = torch.cat((end_surf_features, y), dim=0)
        end_surf_features = end_surf_features[:10, :].flatten().unsqueeze(dim=0)

    new_state = torch.cat((cur_surf_features, end_surf_features), dim=0)
    return new_state
