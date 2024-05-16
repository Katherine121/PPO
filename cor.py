import os
import random
import shutil
import math

# 图像
IMG_SIZE = 256
PATCH_SIZE = 32
# the random position deviation
# HEIGHT_NOISE = {100: 25}
HEIGHT_NOISE = {200: 25}
# HEIGHT_NOISE = {300: 25}
# HEIGHT_NOISE = {400: 25}
# HEIGHT_NOISE = {500: 25}
# HEIGHT_NOISE = {100: 25, 200: 25, 300: 25, 400: 25, 500: 25}

h_list = [100, 200, 300, 400, 500]
NOISE_STEP = 25

# the kinds of disturbances
NOISE_DB = [
    ["ori", "ori"], ["ori", "random"],
    # ["cutout", "random"], ["rain", "random"],
    # ["snow", "random"], ["fog", "random"],
    # ["bright", "random"]
]

# 消融实验
dis_list = [50, 75, 100, 125, 150]
height_list = [160, 180, 200, 220, 240]
noise_list = [15, 20, 25, 30, 35]
max_ep_len_list = [128, 192, 256, 320, 384]
validate_num_nodes_list = [60, 80, 100, 120, 140]
radius_list = [800, 900, 1000, 1100, 1200]


def get_start_end(test_num, num_nodes, radius, center_lat=23.4, center_lon=120.3, ):
    points = []
    res = []
    # 计算每个点之间的角度间隔
    angle_step = 2 * math.pi / num_nodes

    # num_nodes个起点
    for i in range(0, num_nodes):
        angle = i * angle_step
        lat = round(center_lat + radius * math.sin(angle) / 111000, 8)
        lon = round(center_lon + radius * math.cos(angle) / 111000 / math.cos(lat / 180 * math.pi), 8)
        # 添加到坐标集合
        points.append((lat, lon))

    # 重复测试次数
    for index in range(0, test_num):

        # num_nodes
        for i in range(0, int(num_nodes)):
            start_point = points[i]
            # end_point = points[(i + num_nodes // 2) % num_nodes]
            end_point = [center_lat, center_lon]

            # 测试高度
            for h_key in HEIGHT_NOISE.keys():
                # 测试噪声
                for noise_index in range(0, len(NOISE_DB)):

                    res.append((index,
                                i, 0,
                                start_point, end_point,
                                h_list[random.randint(a=0, b=4)], noise_index, ))

    return res


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


def del_file(path):
    if not os.listdir(path):
        print('目录为空！')
    else:
        for i in os.listdir(path):
            path_file = os.path.join(path, i)  # 取文件绝对路径
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                del_file(path_file)
                shutil.rmtree(path_file)
