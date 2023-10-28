import os
import random

import gym
import math
import numpy as np
import torch
from PIL import Image
from gym import spaces
from torchvision import transforms

from data_augment import ImageAugment


def get_start_end(center_lat=23.4, center_lon=120.3,
                  num_nodes=10,
                  radius=5000, ):
    points = []
    # 计算每个点之间的角度间隔
    angle_step = 2 * math.pi / num_nodes

    # 生成坐标点
    for i in range(0, num_nodes):
        angle = i * angle_step
        lat = round(center_lat + radius * math.sin(angle) / 111000, 8)
        lon = round(center_lon + radius * math.cos(angle) / 111000 / math.cos(lat / 180 * math.pi), 8)
        # 添加到坐标集合
        points.append((lat, lon))
    return points


def get_big_map(path):
    """
    get the big map stored on the UAV locally
    :param path: stored path
    :return: big map paths and center coordinates
    """
    paths = []
    labels = []

    file_path = os.listdir(path)
    file_path.sort()

    for file in file_path:
        full_file_path = os.path.join(path, file)
        paths.append(full_file_path)
        file = file[:-4]
        file = file.split(',')
        labels.append(list(map(eval, [file[0], file[1]])))

    return paths, labels


def screenshot(paths, labels, new_lat, new_lon, cur_height, img_aug):
    # new frame path
    min_dis = math.inf
    idx = -1

    for i in range(0, len(labels)):
        lat_dis = (new_lat - labels[i][0]) * 111000
        lon_dis = (new_lon - labels[i][1]) * 111000 * math.cos(labels[i][0] / 180 * math.pi)

        dis = math.sqrt(lat_dis * lat_dis + lon_dis * lon_dis)
        if dis < min_dis:
            min_dis = dis
            idx = i

    # latitude: 1000 m
    # longitude: 1000 m
    # 1400, 1400
    # 0.7143 lat m / pixel
    # 0.7143 lpn m / pixel
    # 1.93 alt m / lat m
    # 1.93 alt m / lon m
    # 1.38 alt m / pixel
    # 1.38 alt m / pixel
    # find the most match big map and screenshot
    lat_dis = (new_lat - labels[idx][0]) * 111000
    lon_dis = (new_lon - labels[idx][1]) * 111000 * math.cos(labels[idx][0] / 180 * math.pi)
    # 300的真实距离对应420的像素距离对应580米高度
    lat_pixel_dis = lat_dis * 1.4
    lon_pixel_dis = lon_dis * 1.4
    center = [1400 // 2, 1400 // 2]
    new_lat_pixel = center[0] - lat_pixel_dis
    new_lon_pixel = center[1] + lon_pixel_dis

    # altitude changing
    # 300的真实距离对应420的像素距离对应580米高度
    pixel_h = cur_height / 580 * 420
    pixel_w = pixel_h
    # # If the center of the new image is out of bounds
    # if new_lon_pixel - pixel_w // 2 > 1400:
    #     return -1
    # if new_lat_pixel - pixel_h // 2 > 1400:
    #     return -1
    # if new_lon_pixel + pixel_w // 2 < 0:
    #     return -1
    # if new_lat_pixel + pixel_h // 2 < 0:
    #     return -1

    pic = Image.open(paths[idx])
    pic = pic.crop((new_lon_pixel - pixel_w // 2, new_lat_pixel - pixel_h // 2,
                    new_lon_pixel + pixel_w // 2, new_lat_pixel + pixel_h // 2))
    pic = pic.resize((224, 224))
    # pic = pic.rotate(90 - ang)

    # add style noise to the new image
    pic = np.array(pic)
    pic = img_aug(pic)
    pic = Image.fromarray(pic)
    # pic.save(path)

    # # new screenshot image
    # return pic
    # If the center of the new image is out of bounds
    if new_lon_pixel - pixel_w // 2 > 1400:
        return [pic, -1]
    if new_lat_pixel - pixel_h // 2 > 1400:
        return [pic, -1]
    if new_lon_pixel + pixel_w // 2 < 0:
        return [pic, -1]
    if new_lat_pixel + pixel_h // 2 < 0:
        return [pic, -1]
    return pic


class MyUAVgym(gym.Env):
    def __init__(self, len, bigmap_dir, num_nodes, dis, done_thresh, max_step_num):
        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=+1.0, shape=(2, 3, 224, 224), dtype=np.float32)
        self.start_pos = []
        self.end_pos = []
        self.cur_pos = []
        self.start_pic = None
        self.end_pic = None
        self.cur_pic = None
        self.last_diff = 0
        self.next_angles = []
        self.len = len

        self.bigmap_dir = bigmap_dir
        self.num_nodes = num_nodes
        self.dis = dis
        self.done_thresh = done_thresh
        self.max_step_num = max_step_num
        self.points = get_start_end()
        self.paths, self.labels = get_big_map(path=self.bigmap_dir)

        self.HEIGHT_NOISE = {100: 25, 150: 25, 200: 25, 250: 50, 300: 50}
        self.height = 100
        self.cur_height = 100
        self.NOISE_DB = [["ori", "ori"], ["ori", "random"],
                         # ["cutout", "ori"], ["rain", "ori"],
                         # ["snow", "ori"], ["fog", "ori"], ["bright", "ori"],
                         # ["cutout", "random"], ["rain", "random"],
                         # ["snow", "random"], ["fog", "random"], ["bright", "random"]
                         ]
        self.step_num = 0
        self.image_augment = None
        self.angle = None

    def get_input(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        # slice to avoid modifying the original list
        cur_pic = self.cur_pic[:]
        next_angles = self.next_angles[:]

        next_imgs = None

        # original images and angles
        for i in range(0, len(cur_pic)):
            img = cur_pic[i]
            # img = Image.open(img)
            img = img.convert('RGB')
            img = val_transform(img).unsqueeze(dim=0)
            if next_imgs is None:
                next_imgs = img
            else:
                next_imgs = torch.cat((next_imgs, img), dim=0)
        next_angles.append([0, 0])

        # append the end point frame as part of model input
        # dest_img = Image.open(self.end_pic)
        dest_img = self.end_pic.convert('RGB')
        dest_img = val_transform(dest_img).unsqueeze(dim=0)
        next_imgs = torch.cat((next_imgs, dest_img), dim=0)

        dest_angle = [0, 0]
        next_angles.append(dest_angle)
        next_angles = torch.tensor(next_angles, dtype=torch.float)

        # if there are not enough input frames
        for i in range(0, self.len - 1 - len(cur_pic)):
            next_imgs = torch.cat((next_imgs, torch.zeros((1, 3, 224, 224))), dim=0)
            next_angles = torch.cat((next_angles, torch.zeros((1, 2))), dim=0)

        # add a batch dimension
        return next_imgs.unsqueeze(dim=0), next_angles.unsqueeze(dim=0)

    def reset(self):
        # p = random.randint(a=0, b=self.num_nodes - 2)
        # q = random.randint(a=p + 1, b=self.num_nodes - 1)
        p = 0
        q = 1
        self.start_pos = list(self.points[p])
        self.end_pos = list(self.points[q])
        self.cur_pos = self.start_pos

        keys = list(self.HEIGHT_NOISE.keys())
        # k_index = random.randint(a=0, b=len(keys) - 1)
        k_index = 1
        self.height = keys[k_index]

        # noise_db_index = random.randint(a=0, b=len(self.NOISE_DB) - 1)
        noise_db_index = 0
        # choose a kind of noise
        self.image_augment = ImageAugment(style_idx=self.NOISE_DB[noise_db_index][0],
                                          shift_idx=self.NOISE_DB[noise_db_index][1])
        shift = self.image_augment.forward_shift(self.HEIGHT_NOISE[self.height])
        self.start_pos[0] += shift[0]
        self.start_pos[1] += shift[1]
        self.cur_height = self.height + shift[2]
        self.cur_pos = self.start_pos
        lat_diff = (self.end_pos[0] - self.cur_pos[0]) * 111000
        lon_diff = (self.end_pos[1] - self.cur_pos[1]) * 111000 * math.cos(self.cur_pos[0] / 180 * math.pi)
        self.last_diff = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)

        # print("start_pos:")
        # print(self.start_pos)
        # print("end_pos:")
        # print(self.end_pos)
        # print("height:")
        # print(self.cur_height)
        # screenshot the start point image
        self.start_pic = screenshot(self.paths, self.labels, self.start_pos[0], self.start_pos[1],
                                    self.cur_height, self.image_augment)
        self.end_pic = screenshot(self.paths, self.labels, self.end_pos[0], self.end_pos[1],
                                  self.cur_height, self.image_augment)
        self.cur_pic = [self.start_pic]

        self.next_angles = []

        self.step_num = 0

        state = self.get_input()
        return state

    def step(self, action):
        # # 根据cur_pos, end计算下一步的angle
        # lat_diff = (self.end_pos[0] - self.cur_pos[0]) * 111000
        # lon_diff = (self.end_pos[1] - self.cur_pos[1]) * 111000 * math.cos(self.cur_pos[0] / 180 * math.pi)
        # # tan = lat_diff / lon_diff
        # # self.angle = math.atan(tan)
        # # if lat_diff > 0 and lon_diff < 0:
        # #     self.angle += 180
        # # elif lat_diff < 0 and lon_diff < 0:
        # #     self.angle -= 180
        # # assert -180 <= self.angle <= 180
        # diff = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
        # # print(diff)
        # self.angle = torch.tensor([lat_diff / diff, lon_diff / diff], dtype=torch.float32)

        # 根据cur_pos, action计算下一步的cur_pos
        # action *= 180
        # sin = math.sin(action)
        # cos = math.cos(action)
        # lat_delta = self.dis * sin
        # lon_delta = self.dis * cos
        lat_delta = self.dis * action[0]
        lon_delta = self.dis * action[1]
        lat_delta = float(lat_delta / 111000)
        lon_delta = float(lon_delta / 111000 / math.cos(self.cur_pos[0] / 180 * math.pi))
        self.cur_pos[0] += lat_delta
        self.cur_pos[1] += lon_delta

        # add shift noise to the origin position
        shift = self.image_augment.forward_shift(self.HEIGHT_NOISE[self.height])
        self.cur_pos[0] += shift[0]
        self.cur_pos[1] += shift[1]
        # print(self.cur_pos)
        self.cur_height = self.height + shift[2]

        # 根据self.angle, action计算reward
        # 相差angle_thresh度以内就增加概率
        # 每走一步减一
        # reward = 0
        # res = math.fabs(self.angle - action)
        # if res > 180:
        #     res = 360 - res
        # print(math.fabs(res))
        # if math.fabs(res) > self.angle_thresh:
        #     reward -= 0.1
        # else:
        #     reward += 1 / math.fabs(res)
        # reward = -math.fabs(res) + self.angle_thresh - 1
        # 130

        # reward = 1 / torch.norm(self.angle - action, p=2, dim=0) 294
        # reward = 1 / torch.norm(self.angle - action, p=2, dim=0) + 10 / diff - 0.5 195
        # reward = 1 / torch.norm(self.angle - action, p=2, dim=0) - 0.5 194
        # reward = 1 / torch.norm(self.angle - action, p=2, dim=0) - 0.5 - 0.2 154
        # reward = 1 / torch.norm(self.angle - action, p=2, dim=0) + 10 / diff - 1 95
        # reward = 1 / torch.norm(self.angle - action, p=2, dim=0) - 1 94
        # reward = 1 / torch.norm(self.angle - action, p=2, dim=0) + 1 / diff - 1 94

        # reward = 1 / torch.norm(self.angle - action, p=1, dim=0)
        # reward = 1 / torch.norm(self.angle - action, p=1, dim=0) - 0.1 - 0.2 166
        # reward = 1 / torch.norm(self.angle - action, p=1, dim=0) - 0.5 126

        #
        # 129
        # reward = 1 / torch.norm(self.angle - action, p=1, dim=0) + 10 / diff - 0.5
        # print(torch.norm(self.angle - action, p=2, dim=0))
        # reward = -math.pow(min(1, (torch.norm(self.angle - action, p=2, dim=0) / (2 * math.sqrt(2)) +
        #                             diff / 3000) / 2), 2.8)

        # 根据下一步cur_pos, end计算done
        lat_diff = (self.end_pos[0] - self.cur_pos[0]) * 111000
        lon_diff = (self.end_pos[1] - self.cur_pos[1]) * 111000 * math.cos(self.cur_pos[0] / 180 * math.pi)
        diff = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
        done = diff <= self.done_thresh or self.step_num == self.max_step_num - 1
        # 根据下一步cur_pos, end计算reward
        # reward1 = -math.pow(min(1, diff / 3000), 2.8)
        # -1~+1
        reward1 = -(diff - self.last_diff) / self.dis
        self.last_diff = diff
        reward = reward1
        # 成功到达终点额外加100奖励
        if diff <= self.done_thresh and self.step_num <= self.max_step_num - 1:
            print("successfully arrived! in " + str(diff) + " m, by total_step_num: " + str(self.step_num))
            print("origin end point pos:" + str(self.end_pos[0]) + "," + str(self.end_pos[1]))
            print("actual end point pos:" + str(self.cur_pos[0]) + "," + str(self.cur_pos[1]))
            reward = reward1 + 100
        # # 超时结束
        # elif self.step_num == self.max_step_num - 1:
        #     reward = reward1 - 100

        # 根据下一步cur_pos计算下一步state
        new_img = screenshot(self.paths, self.labels, self.cur_pos[0], self.cur_pos[1],
                             self.cur_height, self.image_augment)
        if type(new_img) is list:
            self.cur_pic.append(new_img[0])
            self.next_angles.append(action.tolist())
            if len(self.cur_pic) > self.len - 1:
                assert len(self.cur_pic) == len(self.next_angles) + 1
                self.cur_pic.pop(0)
                self.next_angles.pop(0)
            state = self.get_input()
            return state, reward1 - 100, True, {}
        else:
            self.cur_pic.append(new_img)
            self.next_angles.append(action.tolist())
            if len(self.cur_pic) > self.len - 1:
                assert len(self.cur_pic) == len(self.next_angles) + 1
                self.cur_pic.pop(0)
                self.next_angles.pop(0)
            state = self.get_input()

        self.step_num += 1
        info = {}
        return state, reward, done, info

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass
