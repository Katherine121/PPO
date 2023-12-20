import os
import random
import shutil
import math
import torch
from torch.backends import cudnn
import numpy as np
import threading
from datetime import datetime
from PPO import PPO
from mygym import MyUAVgym
from cor import *


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


def get_start_end(num_nodes, radius, test_num, center_lat=23.4, center_lon=120.3,):
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

    for i in range(int(0 * len(points)), int(1 * len(points))):
        start_point = points[i]
        end_point = [center_lat, center_lon]

        # 5个高度
        for h_key in HEIGHT_NOISE.keys():
            for index in range(0, test_num):
                # 原始
                res.append((start_point, end_point, h_key, 0, index))
            for index in range(0, test_num):
                # 位置噪声
                res.append((start_point, end_point, h_key, 1, index))
            for index in range(0, test_num):
                # cutout+位置噪声
                res.append((start_point, end_point, h_key, 2, index))
            for index in range(0, test_num):
                # rain+位置噪声
                res.append((start_point, end_point, h_key, 3, index))
            for index in range(0, test_num):
                # snow+位置噪声
                res.append((start_point, end_point, h_key, 4, index))
            for index in range(0, test_num):
                # fog+位置噪声
                res.append((start_point, end_point, h_key, 5, index))
            for index in range(0, test_num):
                # bright+位置噪声
                res.append((start_point, end_point, h_key, 6, index))

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


def train_thread(start_episode, end_episode):
    # 声明全局变量
    global print_running_reward
    global print_running_episodes
    global log_running_reward
    global log_running_episodes
    global total_success_num
    global total_success_diff
    global time_step
    global i_episode
    global ppo_agent

    env = MyUAVgym(dis=200, done_thresh=100, max_step_num=max_ep_len,
                   points=points, paths=paths, path_labels=path_labels)
    env.seed(random_seed)

    # training loop
    for index in range(start_episode, end_episode):
        if os.path.exists(dataset_dir) is False:
            os.makedirs(dataset_dir, exist_ok=True)

        # episode_dir = datasets_path + "/" + str(index)
        # if os.path.exists(episode_dir) is False:
        #     os.makedirs(episode_dir)

        state, path_list, episode_dir = env.reset(dataset_dir=dataset_dir, index=index)
        current_ep_reward = 0

        for t in range(1, max_ep_len):
            # select action with policy
            with ppo_agent_lock:
                action = ppo_agent.select_action(state, path_list, episode_dir)
            state, path_list, reward, done, _, success, success_diff = env.step(action)
            with total_success_num_lock:
                total_success_num += success
            with total_success_diff_lock:
                if success == 1:
                    total_success_diff += success_diff

            # # saving reward and is_terminals
            # ppo_agent.buffer.rewards.append(reward)
            # ppo_agent.buffer.is_terminals.append(done)
            # 写到文件里，这样多线程顺序才不会错
            reward_file = os.path.join(episode_dir, "reward.txt")
            with open(reward_file, "a") as file1:
                done_int = 1 if done else 0
                file1.write(str(reward) + " " + str(done_int) + "\n")
            file1.close()

            with time_step_lock:
                time_step += 1
            current_ep_reward += reward

            # break; if the episode is over
            if done:
                break

        with print_running_reward_lock:
            print_running_reward += current_ep_reward
        with print_running_episodes_lock:
            print_running_episodes += 1

        with log_running_reward_lock:
            log_running_reward += current_ep_reward
        with log_running_episodes_lock:
            log_running_episodes += 1

        with i_episode_lock:
            i_episode += 1

    env.close()


# ==============================================================================
# 需要修改的参数
run_num = 16
num_nodes = 100
radius = 3000
max_ep_len = 512  # max timesteps in one episode

test_num = 1  # 某一种起点、终点、高度、噪声对应的测试数目
update_timestep = num_nodes * len(HEIGHT_NOISE) * len(NOISE_DB) * test_num  # update policy every n timesteps

batch_size = 128
max_thread_num = 8
random_seed = 0  # set random seed if required (0 = no random seed)
# ==============================================================================

# ==============================================================================
# 不需要修改的参数
env_name = "UAVnavigation"
has_continuous_action_space = True  # continuous action space; else discrete
bigmap_dir = "../../../mnt/nfs/wyx/bigmap"

max_training_timesteps = 100 * update_timestep  # break training loop if timeteps > max_training_timesteps
save_model_freq = 500  # save model frequency (in num timesteps)

action_std = 0.6  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5 / max_ep_len)  # action_std decay frequency (in num timesteps)

K_epochs = 30  # update policy for K epochs in one PPO update
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network
# ==============================================================================

#### log files for multiple runs are NOT overwritten
log_dir = "PPO_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

#### create new log file for each run
log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"
#### create new log file for each run
diff_f_name = log_dir + '/PPO_' + env_name + "_diff_" + str(run_num) + ".csv"
#### create new log file for each run
acc_f_name = log_dir + '/PPO_' + env_name + "_acc_" + str(run_num) + ".csv"

print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + log_f_name)
#####################################################

directory = "PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)

checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num)
best_path = directory + "PPO_{}_{}_{}_best.pth".format(env_name, random_seed, run_num)
print("save checkpoint path : " + checkpoint_path)
#####################################################

############# print all hyperparameters #############
print("--------------------------------------------------------------------------------------------")
print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)
print("model saving frequency : " + str(save_model_freq) + " timesteps")
if has_continuous_action_space:
    print("Initializing a continuous action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("starting std of action distribution : ", action_std)
    print("decay rate of std of action distribution : ", action_std_decay_rate)
    print("minimum std of action distribution : ", min_action_std)
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
else:
    print("Initializing a discrete action space policy")
print("--------------------------------------------------------------------------------------------")
print("PPO update frequency : " + str(update_timestep) + " timesteps")
print("PPO K epochs : ", K_epochs)
print("update batch size : ", batch_size)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)
print("--------------------------------------------------------------------------------------------")
print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)
if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
#####################################################

print("============================================================================================")
torch.autograd.set_detect_anomaly(True)
################# training procedure ################

# initialize a PPO agent
ppo_agent = PPO(lr_actor, lr_critic, batch_size, gamma, K_epochs,
                eps_clip, has_continuous_action_space, action_std)

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")
# reduce CPU usage, use it after the model is loaded onto the GPU
torch.set_num_threads(1)
cudnn.benchmark = True

# logging file
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,reward\n')
# logging file
diff_f = open(diff_f_name, "w+")
diff_f.write('episode,timestep,diff\n')
# logging file
acc_f = open(acc_f_name, "w+")
acc_f.write('episode,timestep,acc\n')

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

total_success_num = 0
total_success_diff = 0

time_step = 0
i_episode = 0
best_acc = 0

print_running_reward_lock = threading.Lock()
print_running_episodes_lock = threading.Lock()
log_running_reward_lock = threading.Lock()
log_running_episodes_lock = threading.Lock()
total_success_num_lock = threading.Lock()
total_success_diff_lock = threading.Lock()
time_step_lock = threading.Lock()
i_episode_lock = threading.Lock()
ppo_agent_lock = threading.Lock()

dataset_dir = "../../../mnt/nfs/wyx/ppo/datasets" + str(run_num) + "/"
if os.path.exists(dataset_dir) is False:
    os.makedirs(dataset_dir, exist_ok=True)
else:
    del_file(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

points = get_start_end(num_nodes=num_nodes, radius=radius, test_num=test_num)
paths, path_labels = get_big_map(path=bigmap_dir)

while i_episode < max_training_timesteps:
    # 收集数据
    # 初始化线程
    episode_of_one_thread = update_timestep // max_thread_num
    remain = update_timestep % max_thread_num
    threads = []
    end_episode = 0
    for thread_index in range(0, max_thread_num):
        start_episode = end_episode
        end_episode = start_episode + episode_of_one_thread
        if remain > 0:
            end_episode += 1
            remain -= 1
        thread1 = threading.Thread(target=train_thread, args=(start_episode, end_episode))
        threads.append(thread1)

    # 启动线程
    for thread in threads:
        thread.start()

    # 等待所有线程结束
    for thread in threads:
        thread.join()

    # 记录打印结果
    if total_success_num > 0:
        print("total success num is: " + str(total_success_num / print_running_episodes))
        acc_f.write('{},{},{}\n'.format(i_episode, time_step, total_success_num / print_running_episodes))
        acc_f.flush()

        print("total success diff is: " + str(total_success_diff / total_success_num))
        diff_f.write('{},{},{}\n'.format(i_episode, time_step, total_success_diff / total_success_num))
        diff_f.flush()

        if total_success_num / print_running_episodes >= best_acc:
            best_acc = total_success_num / print_running_episodes
            print("------------------------------------------------------")
            print("saving best model at : " + best_path)
            ppo_agent.save(time_step, best_acc, best_path)
            print("best model saved")
            print("------------------------------------------------------")
    else:
        print("total success num is: " + str(total_success_num))
        acc_f.write('{},{},{}\n'.format(i_episode, time_step, total_success_num))
        acc_f.flush()

        print("total success diff is: " + str(total_success_diff))
        diff_f.write('{},{},{}\n'.format(i_episode, time_step, total_success_diff))
        diff_f.flush()
    total_success_num = 0
    total_success_diff = 0

    # log average reward till last episode
    log_avg_reward = log_running_reward / log_running_episodes
    log_avg_reward = round(log_avg_reward, 4)
    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
    log_f.flush()
    log_running_reward = 0
    log_running_episodes = 0

    # print average reward till last episode
    print_avg_reward = print_running_reward / print_running_episodes
    print_avg_reward = round(print_avg_reward, 2)
    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                            print_avg_reward))
    print_running_reward = 0
    print_running_episodes = 0

    # 保存agent
    print("--------------------------------------------------------------------------------------------")
    print("saving model at : " + checkpoint_path)
    ppo_agent.save(time_step, -1, checkpoint_path)
    print("model saved")
    print("--------------------------------------------------------------------------------------------")

    # 更新agent
    ppo_agent.update(dataset_dir)
    del_file(dataset_dir)

    # if continuous action space; then decay action std of ouput action distribution
    if has_continuous_action_space and i_episode % action_std_decay_freq == 0:
        ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

log_f.close()

# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")
