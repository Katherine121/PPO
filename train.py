import os
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


def get_start_end(num_nodes, radius, test_num, center_lat=23.4, center_lon=120.3, ):
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

    # num_nodes
    for i in range(int(0 * len(points)), int(1 * len(points))):
        start_point = points[i]
        end_point = [center_lat, center_lon]

        # 高度
        for h_key in HEIGHT_NOISE.keys():
            # 噪声数目
            for noise_index in range(0, len(NOISE_DB)):
                # 每一种噪声的测试路径数目
                for index in range(0, test_num):
                    res.append((start_point, end_point, h_key, noise_index, index))

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
    # global time_step
    # global print_running_episodes
    # global log_running_reward
    # global log_running_episodes
    global ppo_agent
    global time_step
    global total_success_num
    global total_success_diff
    global print_running_reward
    global i_episode

    env = MyUAVgym(dis=200, done_thresh=100, max_step_num=max_ep_len,
                   points=points, paths=paths, path_labels=path_labels)
    if random_seed:
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
            action = ppo_agent.select_action(state, path_list, episode_dir, ppo_agent_lock)
            state, path_list, reward, done, _, success, success_diff = env.step(action)

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

        with variance_lock:
            total_success_num += success
            total_success_diff += success_diff
            print_running_reward += current_ep_reward
            i_episode += 1

    env.close()


# ==============================================================================
# 需要修改的参数
run_num = 17
num_nodes = 100

test_num = 1  # 某一种起点、终点、高度、噪声对应的测试数目
update_timestep = num_nodes * len(HEIGHT_NOISE) * len(NOISE_DB) * test_num  # update policy every n timesteps

data = "../../../mnt/nfs/"

batch_size = 128
K_epochs = 30  # update policy for K epochs in one PPO update
lr_actor = 0.0003  # learning rate for actor network

radius = 3000
max_ep_len = 512  # max timesteps in one episode
max_thread_num = 8

random_seed = 0  # set random seed if required (0 = no random seed)
# ==============================================================================

# ==============================================================================
# 不需要修改的参数
env_name = "UAVnavigation"
has_continuous_action_space = True  # continuous action space; else discrete
bigmap_dir = data + "wyx/bigmap"

max_training_timesteps = 100 * update_timestep  # break training loop if timeteps > max_training_timesteps

action_std = 0.6  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5 / max_ep_len)  # action_std decay frequency (in num timesteps)

eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor
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
diff_f_name = log_dir + '/PPO_' + env_name + "_diff_" + str(run_num) + ".csv"
acc_f_name = log_dir + '/PPO_' + env_name + "_acc_" + str(run_num) + ".csv"
print("--------------------------------------------------------------------------------------------")
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
print("num_nodes : ", num_nodes)
print("test_num: ", test_num)
print("PPO update frequency : " + str(update_timestep) + " timesteps")

print("update batch size : ", batch_size)
print("PPO K epochs : ", K_epochs)
print("optimizer learning rate actor : ", lr_actor)

print("radius : ", radius)
print("max timesteps per episode : ", max_ep_len)
print("max thread num : ", max_thread_num)

if random_seed:
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", max_training_timesteps)

if has_continuous_action_space:
    print("Initializing a continuous action space policy")
    print("starting std of action distribution : ", action_std)
    print("decay rate of std of action distribution : ", action_std_decay_rate)
    print("minimum std of action distribution : ", min_action_std)
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
else:
    print("Initializing a discrete action space policy")

print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)
print("optimizer learning rate critic : ", lr_critic)
#####################################################

torch.autograd.set_detect_anomaly(True)
# reduce CPU usage, use it after the model is loaded onto the GPU
torch.set_num_threads(1)
cudnn.benchmark = True

################# training procedure ################

# logging file
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,reward\n')
# logging file
diff_f = open(diff_f_name, "w+")
diff_f.write('episode,timestep,diff\n')
# logging file
acc_f = open(acc_f_name, "w+")
acc_f.write('episode,timestep,acc\n')

# initialize a PPO agent
ppo_agent = PPO(lr_actor, lr_critic, batch_size, gamma, K_epochs,
                eps_clip, has_continuous_action_space, action_std)
# time_step = 2650888
time_step = 0
total_success_num = 0
total_success_diff = 0
print_running_reward = 0
# i_episode = 5600
i_episode = 0

ppo_agent_lock = threading.Lock()
time_step_lock = threading.Lock()
variance_lock = threading.Lock()

dataset_dir = data + "wyx/ppo/datasets" + str(run_num) + "/"
if os.path.exists(dataset_dir) is False:
    os.makedirs(dataset_dir, exist_ok=True)
else:
    del_file(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

points = get_start_end(num_nodes=num_nodes, radius=radius, test_num=test_num)
paths, path_labels = get_big_map(path=bigmap_dir)

# best_acc = 0.22142857142857142
# best_diff = 65.63956115429576
best_acc = 0
best_diff = 10000

print("============================================================================================")
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("============================================================================================")

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
        cur_acc = total_success_num / update_timestep
        cur_diff = total_success_diff / total_success_num

        print("total success num is: ", cur_acc)
        acc_f.write('{},{},{}\n'.format(i_episode, time_step, cur_acc))
        acc_f.flush()

        print("total success diff is: ", cur_diff)
        diff_f.write('{},{},{}\n'.format(i_episode, time_step, cur_diff))
        diff_f.flush()

        if cur_acc > best_acc or (cur_acc == best_acc and cur_diff <= best_diff):
            best_acc = cur_acc
            best_diff = cur_diff
            print("saving best model at : " + best_path)
            ppo_agent.save(time_step, best_acc, best_path)
            print("best model saved")
    else:
        print("total success num is: 0")
        acc_f.write('{},{},{}\n'.format(i_episode, time_step, 0))
        acc_f.flush()

        print("total success diff is: 0")
        diff_f.write('{},{},{}\n'.format(i_episode, time_step, 0))
        diff_f.flush()
    total_success_num = 0
    total_success_diff = 0

    # print average reward till last episode
    print_avg_reward = print_running_reward / update_timestep
    print_avg_reward = round(print_avg_reward, 4)
    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                            print_avg_reward))
    log_f.write('{},{},{}\n'.format(i_episode, time_step, print_avg_reward))
    log_f.flush()
    print_running_reward = 0

    # 保存agent
    print("saving model at : " + checkpoint_path)
    ppo_agent.save(time_step, -1, checkpoint_path)
    print("model saved")
    end_time = datetime.now().replace(microsecond=0)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    # 更新agent
    ppo_agent.update(dataset_dir)
    del_file(dataset_dir)
    print("============================================================================================")

    # if continuous action space; then decay action std of ouput action distribution
    if has_continuous_action_space and i_episode % action_std_decay_freq == 0:
        ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

log_f.close()
diff_f.close()
acc_f.close()

# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")
