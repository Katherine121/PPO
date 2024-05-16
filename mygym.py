import os
import random
from io import BytesIO
import cv2
import gym
import math
import numpy as np
import torch
from PIL import Image
from gym import spaces
from torchvision import transforms
from data_augment import ImageAugment
from cor import *


def dfs_compress(pic_path, out_path, target_size=199, quality=90, step=5, pic_type='.jpg'):
    # read images bytes
    with open(pic_path, 'rb') as f:
        pic_byte = f.read()

    img_np = np.frombuffer(pic_byte, np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_ANYCOLOR)

    current_size = len(pic_byte) / 1024
    # print("image size before compression (KB): ", current_size)
    while current_size > target_size:
        pic_byte = cv2.imencode(pic_type, img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
        if quality - step < 0:
            break
        quality -= step
        current_size = len(pic_byte) / 1024

    # save image
    with open(out_path, 'wb') as f:
        f.write(BytesIO(pic_byte).getvalue())

    return len(pic_byte) / 1024


def screenshot(paths, labels, new_lat, new_lon, cur_height, img_aug, path):
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
    # 300 distance : 420 pixel distance : 580 height
    lat_pixel_dis = lat_dis * 1.4
    lon_pixel_dis = lon_dis * 1.4
    # center = [1400 // 2, 1400 // 2]
    # pixel center of the cropped image
    new_lat_pixel = 700 - lat_pixel_dis
    new_lon_pixel = 700 + lon_pixel_dis

    # altitude changing
    # 300 distance : 420 pixel distance : 580 height
    pixel_h = cur_height / 580 * 420
    pixel_w = pixel_h
    # # If the center of the new image is out of bounds
    # if new_lat_pixel + pixel_h // 2 < 0:
    #     return -1
    # if new_lat_pixel - pixel_h // 2 > 1400:
    #     return -1
    # if new_lon_pixel + pixel_w // 2 < 0:
    #     return -1
    # if new_lon_pixel - pixel_w // 2 > 1400:
    #     return -1

    pic = Image.open(paths[idx])
    pic = pic.crop((new_lon_pixel - pixel_w // 2, new_lat_pixel - pixel_h // 2,
                    new_lon_pixel + pixel_w // 2, new_lat_pixel + pixel_h // 2))
    # pic = pic.resize((IMG_SIZE, IMG_SIZE))
    # pic = pic.rotate(90 - ang)

    # add style noise to the new image
    pic = pic.convert('RGB')
    # if img_aug is not None and img_aug.style_idx != "ori":
    #     pic = np.array(pic)
    #     pic = img_aug(pic)
    #     pic = Image.fromarray(pic)
    pic.save(path)
    dfs_compress(path, path, target_size=10)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    pic = val_transform(pic)
    pic = pic.unsqueeze(dim=0)

    # # new screenshot image
    # return pic
    # If the center of the new image is out of bounds
    if new_lat_pixel + pixel_h // 2 < 0:
        return [pic, -1]
    if new_lat_pixel - pixel_h // 2 > 1400:
        return [pic, -1]
    if new_lon_pixel + pixel_w // 2 < 0:
        return [pic, -1]
    if new_lon_pixel - pixel_w // 2 > 1400:
        return [pic, -1]
    return pic


class MyUAVgym(gym.Env):
    def __init__(self, dis, done_thresh, max_step_num, points, paths, path_labels):
        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=+1.0, shape=(2, 3, IMG_SIZE, IMG_SIZE,), dtype=np.float32)
        self.start_pos_i = 0
        self.start_pos = []
        self.end_pos = []
        self.cur_pos = []
        self.end_pic = None
        self.cur_pic = None
        self.end_path = ""
        self.cur_path = ""
        self.last_diff = 0
        self.last_angle = []
        self.test_num = 0

        self.dis = dis
        self.done_thresh = done_thresh
        self.max_step_num = max_step_num
        self.points = points
        self.paths = paths
        self.path_labels = path_labels

        self.HEIGHT_NOISE = HEIGHT_NOISE
        self.height = 200
        self.cur_height = 200
        self.NOISE_DB = NOISE_DB
        self.noise_index = 0
        self.step_num = 0
        self.image_augment = None
        self.episode_dir = ""

    def get_img_input(self):
        # return [self.cur_path, self.end_path]
        return torch.cat((self.cur_pic, self.end_pic), dim=0)

    def reset(self, dataset_dir, index):
        i = index % len(self.points)
        # initial data
        self.step_num = 0
        self.test_num = self.points[i][0]
        self.start_pos_i = self.points[i][1]
        self.end_pos_i = self.points[i][2]
        self.start_pos = self.points[i][3]
        self.end_pos = self.points[i][4]
        self.height = self.points[i][5]
        self.noise_index = self.points[i][6]

        self.episode_dir = dataset_dir + "path" + str(self.test_num) + str(",") \
                           + str(self.start_pos_i) + "," + str(self.end_pos_i) + "," \
                           + "height" + str(self.height) + "," \
                           + self.NOISE_DB[self.noise_index][0] + "-" + self.NOISE_DB[self.noise_index][1]
        if os.path.exists(self.episode_dir) is False:
            os.makedirs(self.episode_dir, exist_ok=True)

        # 1
        # 起点位置加噪声，高度随机选择档位，根据环境调整噪声
        # 中间位置根据环境调整噪声，高度随机选择档位，根据环境调整噪声
        # 终点位置不加噪声，高度不加噪声
        last_lat = self.start_pos[0] + random.uniform(a=-1, b=1) * NOISE_STEP / 111000
        last_lon = self.start_pos[1] + random.uniform(a=-1, b=1) * NOISE_STEP / 111000 \
                   / math.cos(last_lat / 180 * math.pi)
        self.cur_pos = [last_lat, last_lon]
        # last_lat = self.start_pos[0]
        # last_lon = self.start_pos[1]
        # self.cur_pos = [last_lat, last_lon]

        # 2
        self.image_augment = ImageAugment(style_idx=self.NOISE_DB[self.noise_index][0],
                                          shift_idx=self.NOISE_DB[self.noise_index][1])
        shift = self.image_augment.forward_shift(NOISE_STEP)
        self.cur_height = h_list[random.randint(a=0, b=4)] + shift[2]

        # 3
        lat_diff = (self.end_pos[0] - self.cur_pos[0]) * 111000
        lon_diff = (self.end_pos[1] - self.cur_pos[1]) * 111000 * math.cos(self.cur_pos[0] / 180 * math.pi)
        self.last_diff = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
        self.last_angle = [lat_diff / self.last_diff, lon_diff / self.last_diff]

        # screenshot the start point image
        # 4
        self.cur_path = self.episode_dir + '/' + str(self.step_num) + "," \
                        + str(self.cur_pos[0]) + "," + str(self.cur_pos[1]) \
                        + "," + str(self.cur_height) + '.png'
        self.cur_pic = screenshot(self.paths, self.path_labels, self.cur_pos[0], self.cur_pos[1],
                                  self.cur_height, self.image_augment, self.cur_path)
        self.end_path = self.episode_dir + '/' + str(10000) + "," \
                        + str(self.end_pos[0]) + "," + str(self.end_pos[1]) \
                        + "," + str(self.height) + '.png'
        self.end_pic = screenshot(self.paths, self.path_labels, self.end_pos[0], self.end_pos[1],
                                  self.height, None, self.end_path)

        self.step_num = 1

        state = self.get_img_input()
        return state, [self.cur_path, self.end_path], self.episode_dir

    def step(self, action):
        # calculate the next cur_pos, based on cur_pos, action
        # 1
        lat_delta = float(self.dis * action[0] / 111000)
        lon_delta = float(self.dis * action[1] / 111000 / math.cos(self.cur_pos[0] / 180 * math.pi))
        self.cur_pos[0] += lat_delta
        self.cur_pos[1] += lon_delta

        # add shift noise to the origin position
        # 2
        shift = self.image_augment.forward_shift(NOISE_STEP)
        self.cur_pos[0] += shift[0]
        self.cur_pos[1] += shift[1]
        self.cur_height = h_list[random.randint(a=0, b=4)] + shift[2]

        # calculate done based on the next cur_pos, destination
        # 3
        lat_diff = (self.end_pos[0] - self.cur_pos[0]) * 111000
        lon_diff = (self.end_pos[1] - self.cur_pos[1]) * 111000 * math.cos(self.cur_pos[0] / 180 * math.pi)
        diff = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
        angle = [lat_diff / self.last_diff, lon_diff / self.last_diff]
        done = diff <= self.done_thresh or self.step_num == self.max_step_num - 1

        # calculate reward based on the next cur_pos, destination
        # reward1 = -math.pow(min(1, diff / 3000), 2.8)
        # -1~+1
        reward1 = (self.last_diff - diff) / self.dis + \
                  torch.cosine_similarity(torch.tensor(action), torch.tensor(self.last_angle), dim=-1).item()
        # reward1 = (self.last_diff - diff) / self.dis + \
        #           torch.cosine_similarity(torch.tensor(action), torch.tensor(self.last_angle), dim=-1).item() + \
        #           1 / self.step_num
        # reward1 = (self.last_diff - diff) / self.dis + \
        #           torch.cosine_similarity(torch.tensor(action), torch.tensor(self.last_angle), dim=-1).item() - \
        #           math.log10(self.step_num)
        self.last_diff = diff
        self.last_angle = angle
        reward = reward1
        # successfully arrive
        success = 0
        success_diff = 0
        if diff <= self.done_thresh and self.step_num <= self.max_step_num - 1:
            print(self.episode_dir + ": successfully arrived in " + str(diff) + " m, by total_step_num " + str(
                self.step_num))
            # print("origin end point pos:" + str(self.end_pos[0]) + "," + str(self.end_pos[1]))
            # print("actual end point pos:" + str(self.cur_pos[0]) + "," + str(self.cur_pos[1]))
            reward = reward1 + 10
            success = 1
            success_diff = diff

        # calculate next state based on the next cur_pos
        # 4
        self.cur_path = self.episode_dir + '/' + str(self.step_num) + "," \
                        + str(self.cur_pos[0]) + "," + str(self.cur_pos[1]) \
                        + "," + str(self.cur_height) + '.png'
        new_img = screenshot(self.paths, self.path_labels, self.cur_pos[0], self.cur_pos[1],
                             self.cur_height, self.image_augment, self.cur_path)
        if type(new_img) is list:
            self.cur_pic = new_img[0]
            state = self.get_img_input()

            reward = reward1 - 10
            self.step_num += 1
            return state, [self.cur_path, self.end_path], reward, True, {}, success, success_diff, self.step_num
        # elif diff >= 2000:
        #     self.cur_pic = new_img
        #     state = self.get_img_input()
        #
        #     self.step_num += 1
        #     return state, [self.cur_path, self.end_path], reward1 - 10, True, {}, success, success_diff
        else:
            self.cur_pic = new_img
            state = self.get_img_input()

            self.step_num += 1
            return state, [self.cur_path, self.end_path], reward, done, {}, success, success_diff, self.step_num

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass
