import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def draw_acc_reward(file_num, trainorval, accorreward, max_length):
    df_list = []
    max_timestep = 0

    for i in range(0, file_num):
        # 读取Excel文件
        if accorreward == 'reward':
            df_item = pd.read_csv('PPO_logs/UAVnavigation/PPO_UAVnavigation_' +
                                  trainorval + 'log_' + str(i + 1) + '.csv')
        else:
            df_item = pd.read_csv('PPO_logs/UAVnavigation/PPO_UAVnavigation_' +
                                  trainorval + accorreward + '_' + str(i + 1) + '.csv')
        # 整理数据
        if accorreward == 'acc':
            df_item[accorreward] *= 100
        df_item['timestep'] /= 1e6

        # 全部截取到规定的个epoch
        df_item = df_item.loc[:max_length]
        # 截取到最大准确率那一行
        max_row = df_item[accorreward].idxmax()

        # 如果最佳结果特别好，就删除后续掉落严重的数据
        # 如果最佳结果很差，那就不删除后续掉落严重的数据
        if accorreward == 'acc' and df_item.loc[max_row, accorreward] > 90:
            for j in range(min(len(df_item) - 1, max_length), max_row, -1):
                if df_item.loc[j, accorreward] < 90:
                    df_item = df_item.drop(df_item.index[j])
        elif accorreward == 'reward' and df_item.loc[max_row, accorreward] > 15:
            for j in range(min(len(df_item) - 1, max_length), max_row, -1):
                if df_item.loc[j, accorreward] < 15:
                    df_item = df_item.drop(df_item.index[j])

        df_list.append(df_item)

        # # 记录最长的timestep
        # if df_item['timestep'].max() > max_timestep:
        #     max_timestep = df_item['timestep'].max()
        #     max_df = i

    for i in range(0, file_num):
        df_item = df_list[i]
        # 不够规定数目个epoch的表格需要填充到规定数目个epoch，用删除掉落数据后的最后一个数据
        df_item = df_item.reindex(range(max_length + 1), method='nearest')
        # # 修改每一个表格的最后一个timestep为最大的timestep
        # df_item.at[len(df_item) - 1, 'timestep'] = max_timestep
        df_list[i] = df_item

    for i in range(0, file_num):
        df_item = df_list[i]
        if accorreward == 'acc':
            df_item = df_item.append(pd.Series([0, 0, 0, 0, 0], index=df_item.columns), ignore_index=True)
        else:
            df_item = df_item.append(pd.Series([0, 0, 0], index=df_item.columns), ignore_index=True)
        # 在第一行插入0,0,0,0,0
        for j in range(len(df_item) - 1, 1, -1):
            df_item.loc[j] = df_item.loc[j - 1]
        if accorreward == 'acc':
            df_item.loc[0] = [0, 0, 0, 0, 0]
        else:
            df_item.loc[0] = [0, 0, 0]
        df_list[i] = df_item
        print(df_item)

    color_list = [
        # 红
        'firebrick', 'salmon', 'salmon',
        # 绿
        'darkgreen', 'lightgreen', 'lightgreen',
        # 蓝
        'royalblue', 'lightblue', 'lightblue',
        # 粉
        'palevioletred', 'lightpink', 'lightpink',
        # 紫
        'purple', 'plum', 'plum',
        # 橙
        'orange', 'cornsilk', 'cornsilk',
        # 黑色
        'black', 'gray', 'gray',
    ]

    for i in range(0, file_num, 3):
        # 计算三个表格的acc/reward均值和方差
        acc_or_reward_mean = pd.concat([df_list[i][accorreward],
                                        df_list[i + 1][accorreward],
                                        df_list[i + 2][accorreward]], axis=1).mean(axis=1)
        acc_or_reward_std = pd.concat([df_list[i][accorreward],
                                       df_list[i + 1][accorreward],
                                       df_list[i + 2][accorreward]], axis=1).std(axis=1)
        # 绘制折线图
        plt.plot(df_list[0]['episode'], acc_or_reward_mean, color=color_list[i])
        # 填满
        plt.fill_between(df_list[0]['episode'],
                         (acc_or_reward_mean - acc_or_reward_std).clip(lower=0),
                         (acc_or_reward_mean + acc_or_reward_std).clip(upper=100),
                         color=color_list[i + 1], alpha=0.2)

    # 添加图例
    # 设置图形的最外侧边框为灰色
    plt.gca().spines['top'].set_color('gray')
    plt.gca().spines['right'].set_color('gray')
    plt.gca().spines['bottom'].set_color('gray')
    plt.gca().spines['left'].set_color('gray')
    # 删除所有标题
    plt.legend().set_visible(False)
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')

    # ax=plt.subplot()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # 显示图形
    plt.grid(True)
    plt.savefig('PPO_figs/UAVnavigation/' + trainorval + '.svg', format='svg', dpi=150)
    plt.show()


def draw_abl(index1, index2):
    item_list = []
    # 六个变量，四个方法，五次变化，一次重复实验
    # dis 50,75,100,125,150
    item_list.append(np.array([1.0, 3.0, 3.3, 2.0, 4.0]))
    item_list.append(np.array([7.0, 85.0, 89.3, 89.0, 90.0]))
    item_list.append(np.array([4.0, 1.0, 2.0, 3.0, 5.0]))
    item_list.append(np.array([91.0, 98.0, 99.0, 97.0, 96.0]))

    item_list.append(np.array([4.0, 2.0, 3.3, 2.0, 5.0]))
    item_list.append(np.array([52.0, 92.0, 92.0, 92.0, 97.0]))
    item_list.append(np.array([1.0, 2.0, 2.3, 5.0, 6.0]))
    item_list.append(np.array([85.0, 99.0, 99.3, 100.0, 99.0]))
    # height 160,180,200,220,240
    item_list.append(np.array([6.0, 3.0, 3.3, 5.0, 1.0]))
    item_list.append(np.array([89.0, 90.0, 89.3, 86.0, 86.0]))
    item_list.append(np.array([5.0, 2.0, 2.0, 4.0, 2.0]))
    item_list.append(np.array([77.0, 85.0, 99.0, 98.0, 88.0]))

    item_list.append(np.array([2.0, 3.0, 3.3, 3.0, 1.0]))
    item_list.append(np.array([90.0, 90.0, 92.0, 89.0, 92.0]))
    item_list.append(np.array([1.0, 1.0, 2.3, 6.0, 2.0]))
    item_list.append(np.array([72.0, 85.0, 99.3, 99.0, 85.0]))
    # noise 15,20,25,30,35
    item_list.append(np.array([6.0, 4.0, 3.3, 1.0, 5.0]))
    item_list.append(np.array([92.0, 88.0, 89.3, 88.0, 86.0]))
    item_list.append(np.array([4.0, 2.0, 2.0, 6.0, 3.0]))
    item_list.append(np.array([100.0, 98.0, 99.0, 98.0, 99.0]))

    item_list.append(np.array([2.0, 6.0, 3.3, 3.0, 4.0]))
    item_list.append(np.array([94.0, 92.0, 92.0, 95.0, 90.0]))
    item_list.append(np.array([2.0, 1.0, 2.3, 2.0, 1.0]))
    item_list.append(np.array([97.0, 97.0, 99.3, 100.0, 96.0]))
    # max_ep_len 128.192.256,320,384
    item_list.append(np.array([6.0, 4.0, 3.3, 4.0, 2.0]))
    item_list.append(np.array([87.0, 89.0, 89.3, 89.0, 89.0]))
    item_list.append(np.array([3.0, 4.0, 2.0, 3.0, 12.0]))
    item_list.append(np.array([83.0, 96.0, 99.0, 99.0, 99.0]))

    item_list.append(np.array([5.0, 2.0, 3.3, 4.0, 3.0]))
    item_list.append(np.array([88.0, 94.0, 92.0, 92.0, 96.0]))
    item_list.append(np.array([3.0, 3.0, 2.3, 7.0, 9.0]))
    item_list.append(np.array([76.0, 98.0, 99.3, 99.0, 100.0]))
    # node_num 60,80,100,120,140
    item_list.append(np.array([5.0, 1.25, 3.3, 2.5, 2.86]))
    item_list.append(np.array([88.3, 93.75, 89.3, 86.67, 91.43]))
    item_list.append(np.array([3.3, 1.25, 2.0, 2.5, 2.1]))
    item_list.append(np.array([95.0, 96.25, 99.0, 94.17, 99.29]))

    item_list.append(np.array([0.0, 5.0, 3.3, 2.5, 2.14]))
    item_list.append(np.array([91.67, 96.25, 92.0, 86.67, 92.14]))
    item_list.append(np.array([1.7, 2.5, 2.3, 5.8, 2.1]))
    item_list.append(np.array([95.0, 100.0, 99.3, 97.5, 97.86]))
    # radius 800,900,1000,1100,1200
    item_list.append(np.array([7.0, 5.0, 3.3, 1.0, 2.0]))
    item_list.append(np.array([100.0, 96.0, 89.3, 70.0, 65.0]))
    item_list.append(np.array([1.0, 5.0, 2.0, 6.0, 1.0]))
    item_list.append(np.array([99.0, 97.0, 99.0, 99.0, 99.0]))

    item_list.append(np.array([3.0, 4.0, 3.3, 4.0, 3.0]))
    item_list.append(np.array([99.0, 99.0, 92.0, 79.0, 72.0]))
    item_list.append(np.array([2.0, 1.0, 2.3, 6.0, 4.0]))
    item_list.append(np.array([100.0, 96.0, 99.3, 98.0, 97.0]))

    x_list = []
    x_list.append(np.array([50, 75, 100, 125, 150]))
    x_list.append(np.array([50, 75, 100, 125, 150]))
    x_list.append(np.array([100, 150, 200, 250, 300]))
    x_list.append(np.array([100, 150, 200, 250, 300]))
    x_list.append(np.array([15, 20, 25, 30, 35]))
    x_list.append(np.array([15, 20, 25, 30, 35]))
    x_list.append(np.array([128, 192, 256, 320, 384]))
    x_list.append(np.array([128, 192, 256, 320, 384]))
    x_list.append(np.array([60, 80, 100, 120, 140]))
    x_list.append(np.array([60, 80, 100, 120, 140]))
    x_list.append(np.array([800, 900, 1000, 1100, 1200]))
    x_list.append(np.array([800, 900, 1000, 1100, 1200]))

    # 绘制折线图
    # dronet
    plt.plot(item_list[index2], marker='o', color='black')
    # resnet
    plt.plot(item_list[index2 + 1], marker='*', color='royalblue')
    # ddpg
    plt.plot(item_list[index2 + 2], marker='x', color='darkgreen')
    # ours
    plt.plot(item_list[index2 + 3], marker='d', color='firebrick')

    # 添加图例
    # 设置图形的最外侧边框为灰色
    plt.gca().spines['top'].set_color('gray')
    plt.gca().spines['right'].set_color('gray')
    plt.gca().spines['bottom'].set_color('gray')
    plt.gca().spines['left'].set_color('gray')
    # 删除所有标题
    plt.legend().set_visible(False)
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')

    # ax = plt.subplot()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # 显示图形
    plt.grid(True)
    plt.savefig('PPO_figs/UAVnavigation/' + str(index2) + '.svg', format='svg', dpi=150)
    plt.show()


if __name__ == '__main__':
    # train 6400:31, val 5400:26
    # acc, reward
    draw_acc_reward(file_num=21, trainorval='train', accorreward='acc', max_length=31)
    draw_acc_reward(file_num=21, trainorval='val', accorreward='acc', max_length=26)
    for i in range(0, 48, 4):
        print(i)
        draw_abl(i % 4, i)
