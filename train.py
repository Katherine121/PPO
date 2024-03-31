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
from DDPG import DDPG
from td3 import TD3
from mygym import MyUAVgym
from cor import *


def del_file(path):
    if not os.listdir(path):
        print('empty directory!')
    else:
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                del_file(path_file)
                shutil.rmtree(path_file)


def get_start_end(test_num, num_nodes, radius, center_lat=23.4, center_lon=120.3, ):
    points = []
    res = []
    angle_step = 2 * math.pi / num_nodes

    # num_nodes starting points
    for i in range(0, num_nodes):
        angle = i * angle_step
        lat = round(center_lat + radius * math.sin(angle) / 111000, 8)
        lon = round(center_lon + radius * math.cos(angle) / 111000 / math.cos(lat / 180 * math.pi), 8)
        points.append((lat, lon))

    for index in range(0, test_num):
        for i in range(int(0 * len(points)), int(1 * len(points))):
            start_point = points[i]
            end_point = [center_lat, center_lon]

            for h_key in HEIGHT_NOISE.keys():
                for noise_index in range(0, len(NOISE_DB)):
                    res.append((index, i, start_point, end_point, h_key, noise_index))

    return res


def get_big_map(path):
    """
    get the big map gallery
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
    global ppo_agent
    global time_step
    global ideal_total_success_num
    global ideal_total_success_diff
    global noisy_total_success_num
    global noisy_total_success_diff
    global print_running_reward
    global i_episode

    env = MyUAVgym(dis=dis, done_thresh=done_thresh, max_step_num=max_ep_len,
                   points=train_points, paths=paths, path_labels=path_labels)
    if random_seed:
        env.seed(random_seed)

    # training loop
    for index in range(start_episode, end_episode):
        if os.path.exists(dataset_dir) is False:
            os.makedirs(dataset_dir, exist_ok=True)

        state, path_list, episode_dir = env.reset(dataset_dir=dataset_dir, index=index)
        current_ep_reward = 0

        for t in range(1, max_ep_len):
            # select action with policy
            action = ppo_agent.select_action(state, path_list, episode_dir, ppo_agent_lock)
            state, path_list, reward, done, _, success, success_diff = env.step(action)

            reward_file = os.path.join(episode_dir, "reward.txt")
            with open(reward_file, "a") as file1:
                done_int = 1 if done else 0
                file1.write(str(reward) + " " + str(done_int) + "\n")
            file1.close()

            newstate_file = os.path.join(episode_dir, "newstate.txt")
            with open(newstate_file, "a") as file1:
                file1.write(path_list[0] + " " + path_list[1] + "\n")
            file1.close()

            with time_step_lock:
                time_step += 1
            current_ep_reward += reward

            # break; if the episode is over
            if done:
                break

        with variance_lock:
            if index % 2 == 0:
                ideal_total_success_num += success
                ideal_total_success_diff += success_diff
            else:
                noisy_total_success_num += success
                noisy_total_success_diff += success_diff
            print_running_reward += current_ep_reward
            i_episode += 1

    env.close()


def train():
    global ppo_agent
    global time_step
    global ideal_total_success_num
    global ideal_total_success_diff
    global noisy_total_success_num
    global noisy_total_success_diff
    global print_running_reward
    global i_episode

    best_acc = 0
    best_diff = done_thresh

    while i_episode < max_training_timesteps:
        if i_episode >= 1000:
            cur_acc, cur_diff = validate()
            if cur_acc > best_acc or (cur_acc == best_acc and cur_diff <= best_diff):
                best_acc = cur_acc
                best_diff = cur_diff
                print("saving best model at : " + best_path)
                ppo_agent.save(time_step, best_acc, best_path)
                print("best model saved")
            print("============================================================================================")
            del_file(dataset_dir)

        # collect data
        # initialize threads
        episode_of_one_thread = update_timestep // max_thread_num
        remain = update_timestep % max_thread_num
        threads = []
        end_episode = i_episode
        for thread_index in range(0, max_thread_num):
            start_episode = end_episode
            end_episode = start_episode + episode_of_one_thread
            if remain > 0:
                end_episode += 1
                remain -= 1
            thread1 = threading.Thread(target=train_thread, args=(start_episode, end_episode))
            threads.append(thread1)

        # start threads
        for thread in threads:
            thread.start()

        # wait
        for thread in threads:
            thread.join()
        print("============================================================================================")

        # print results
        total_success_num = ideal_total_success_num + noisy_total_success_num
        total_success_diff = ideal_total_success_diff + noisy_total_success_diff
        if total_success_num > 0:
            cur_acc = total_success_num / update_timestep
            cur_diff = total_success_diff / total_success_num

            ideal_acc = ideal_total_success_num / (update_timestep / 2)
            noisy_acc = noisy_total_success_num / (update_timestep / 2)

            if ideal_total_success_num > 0:
                ideal_diff = ideal_total_success_diff / ideal_total_success_num
            else:
                ideal_diff = 0
            if noisy_total_success_num > 0:
                noisy_diff = noisy_total_success_diff / noisy_total_success_num
            else:
                noisy_diff = 0

            print("total success num is: ", cur_acc)
            train_acc_f.write('{},{},{},{},{}\n'.format(i_episode, time_step, cur_acc, ideal_acc, noisy_acc))
            train_acc_f.flush()

            print("total success diff is: ", cur_diff)
            train_diff_f.write('{},{},{},{},{}\n'.format(i_episode, time_step, cur_diff, ideal_diff, noisy_diff))
            train_diff_f.flush()
        else:
            cur_acc = 0
            cur_diff = done_thresh

            print("total success num is: 0")
            train_acc_f.write('{},{},{},{},{}\n'.format(i_episode, time_step, cur_acc, cur_acc, cur_acc))
            train_acc_f.flush()

            print("total success diff is: 0")
            train_diff_f.write('{},{},{},{},{}\n'.format(i_episode, time_step, cur_diff, cur_diff, cur_diff))
            train_diff_f.flush()
        ideal_total_success_num = 0
        ideal_total_success_diff = 0
        noisy_total_success_num = 0
        noisy_total_success_diff = 0

        # print average reward till last episode
        print_avg_reward = print_running_reward / update_timestep
        print_avg_reward = round(print_avg_reward, 4)
        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                print_avg_reward))
        train_log_f.write('{},{},{}\n'.format(i_episode, time_step, print_avg_reward))
        train_log_f.flush()
        print_running_reward = 0
        print("============================================================================================")

        # save agent
        print("saving model at : " + checkpoint_path)
        ppo_agent.save(time_step, -1, checkpoint_path)
        print("model saved")
        end_time = datetime.now().replace(microsecond=0)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")

        # update agent
        ppo_agent.update(dataset_dir)
        del_file(dataset_dir)
        print("============================================================================================")

        # if continuous action space; then decay action std of ouput action distribution
        if has_continuous_action_space and i_episode % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)


def validate_thread(start_episode, end_episode):
    global ppo_agent
    global validate_time_step
    global ideal_validate_total_success_num
    global ideal_validate_total_success_diff
    global noisy_validate_total_success_num
    global noisy_validate_total_success_diff
    global validate_print_running_reward
    global validate_i_episode

    env = MyUAVgym(dis=dis, done_thresh=done_thresh, max_step_num=max_ep_len,
                   points=validate_points, paths=paths, path_labels=path_labels)
    if random_seed:
        env.seed(random_seed)

    # training loop
    for index in range(start_episode, end_episode):
        if os.path.exists(dataset_dir) is False:
            os.makedirs(dataset_dir, exist_ok=True)

        state, path_list, episode_dir = env.reset(dataset_dir=dataset_dir, index=index)
        current_ep_reward = 0

        for t in range(1, max_ep_len):
            # select action with policy
            action = ppo_agent.select_action(state, path_list, episode_dir, ppo_agent_lock)
            state, path_list, reward, done, _, success, success_diff = env.step(action)

            reward_file = os.path.join(episode_dir, "reward.txt")
            with open(reward_file, "a") as file1:
                done_int = 1 if done else 0
                file1.write(str(reward) + " " + str(done_int) + "\n")
            file1.close()

            newstate_file = os.path.join(episode_dir, "newstate.txt")
            with open(newstate_file, "a") as file1:
                file1.write(path_list[0] + " " + path_list[1] + "\n")
            file1.close()

            with validate_time_step_lock:
                validate_time_step += 1
            current_ep_reward += reward

            # break; if the episode is over
            if done:
                break

        with validate_variance_lock:
            if index % 2 == 0:
                ideal_validate_total_success_num += success
                ideal_validate_total_success_diff += success_diff
            else:
                noisy_validate_total_success_num += success
                noisy_validate_total_success_diff += success_diff
            validate_print_running_reward += current_ep_reward
            validate_i_episode += 1

    env.close()


def validate():
    global ppo_agent
    global validate_time_step
    global ideal_validate_total_success_num
    global noisy_validate_total_success_num
    global ideal_validate_total_success_diff
    global noisy_validate_total_success_diff
    global validate_print_running_reward
    global validate_i_episode

    # collect data
    # initialize threads
    episode_of_one_thread = validate_episode_num // max_thread_num
    remain = validate_episode_num % max_thread_num
    threads = []
    end_episode = validate_i_episode
    for thread_index in range(0, max_thread_num):
        start_episode = end_episode
        end_episode = start_episode + episode_of_one_thread
        if remain > 0:
            end_episode += 1
            remain -= 1
        thread1 = threading.Thread(target=validate_thread, args=(start_episode, end_episode))
        threads.append(thread1)

    # start threads
    for thread in threads:
        thread.start()

    # wait
    for thread in threads:
        thread.join()
    print("============================================================================================")

    # print results
    validate_total_success_num = ideal_validate_total_success_num + noisy_validate_total_success_num
    validate_total_success_diff = ideal_validate_total_success_diff + noisy_validate_total_success_diff
    if validate_total_success_num > 0:
        cur_acc = validate_total_success_num / validate_episode_num
        cur_diff = validate_total_success_diff / validate_total_success_num

        ideal_acc = ideal_validate_total_success_num / (validate_episode_num / 2)
        noisy_acc = noisy_validate_total_success_num / (validate_episode_num / 2)

        if ideal_validate_total_success_num > 0:
            ideal_diff = ideal_validate_total_success_diff / ideal_validate_total_success_num
        else:
            ideal_diff = 0
        if noisy_validate_total_success_num > 0:
            noisy_diff = noisy_validate_total_success_diff / noisy_validate_total_success_num
        else:
            noisy_diff = 0

        print("total success num is: ", cur_acc)
        val_acc_f.write('{},{},{},{},{}\n'.format(validate_i_episode, validate_time_step, cur_acc, ideal_acc, noisy_acc))
        val_acc_f.flush()

        print("total success diff is: ", cur_diff)
        val_diff_f.write('{},{},{},{},{}\n'.format(validate_i_episode, validate_time_step, cur_diff, ideal_diff, noisy_diff))
        val_diff_f.flush()
    else:
        cur_acc = 0
        cur_diff = done_thresh

        print("total success num is: 0")
        val_acc_f.write('{},{},{},{},{}\n'.format(validate_i_episode, validate_time_step, cur_acc, cur_acc, cur_acc))
        val_acc_f.flush()

        print("total success diff is: 0")
        val_diff_f.write('{},{},{},{},{}\n'.format(validate_i_episode, validate_time_step, cur_diff, cur_diff, cur_diff))
        val_diff_f.flush()
    ideal_validate_total_success_num = 0
    ideal_validate_total_success_diff = 0
    noisy_validate_total_success_num = 0
    noisy_validate_total_success_diff = 0

    # print average reward till last episode
    print_avg_reward = validate_print_running_reward / validate_episode_num
    print_avg_reward = round(print_avg_reward, 4)
    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(validate_i_episode, validate_time_step,
                                                                            print_avg_reward))
    val_log_f.write('{},{},{}\n'.format(validate_i_episode, validate_time_step, print_avg_reward))
    val_log_f.flush()
    validate_print_running_reward = 0
    print("============================================================================================")

    return cur_acc, cur_diff


# ==============================================================================
env_name = "UAVnavigation"
has_continuous_action_space = True  # continuous action space; else discrete

run_num = 22
data = "../../../xxx/xxx/"
bigmap_dir = data + "xxx/bigmap"
ppo_supervised_dataset_dir = data + "xxx/ppo_supervised_data" + str(run_num) + "/"
if os.path.exists(ppo_supervised_dataset_dir) is False:
    os.makedirs(ppo_supervised_dataset_dir, exist_ok=True)

max_ep_len = 256  # max timesteps in one episode
train_test_num = 5
train_num_nodes = 20
validate_test_num = 1
validate_num_nodes = 100
radius = 1000
update_timestep = train_test_num * train_num_nodes * len(HEIGHT_NOISE) * len(NOISE_DB) # update policy every n timesteps
validate_episode_num = validate_test_num * validate_num_nodes * len(HEIGHT_NOISE) * len(NOISE_DB)
K_epochs = 30  # update policy for K epochs in one PPO update
batch_size = 128
lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network
max_training_timesteps = 200 * update_timestep  # break training loop if timeteps > max_training_timesteps

dis = 100
done_thresh = 50
max_thread_num = 8

random_seed = 0  # set random seed if required (0 = no random seed)
# ==============================================================================

# ==============================================================================
action_std = 0.6  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5 / max_ep_len)  # action_std decay frequency (in num timesteps)

eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

# ==============================================================================

#### log files for multiple runs are NOT overwritten
log_dir = "PPO_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

#### create new log file for each run

train_log_f_name = log_dir + '/PPO_' + env_name + "_trainlog_" + str(run_num) + ".csv"
train_diff_f_name = log_dir + '/PPO_' + env_name + "_traindiff_" + str(run_num) + ".csv"
train_acc_f_name = log_dir + '/PPO_' + env_name + "_trainacc_" + str(run_num) + ".csv"
val_log_f_name = log_dir + '/PPO_' + env_name + "_vallog_" + str(run_num) + ".csv"
val_diff_f_name = log_dir + '/PPO_' + env_name + "_valdiff_" + str(run_num) + ".csv"
val_acc_f_name = log_dir + '/PPO_' + env_name + "_valacc_" + str(run_num) + ".csv"
print("--------------------------------------------------------------------------------------------")
print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + train_log_f_name)
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
print("max timesteps per episode : ", max_ep_len)
print("train_test_num : ", train_test_num)
print("train_num_nodes : ", train_num_nodes)
print("validate_test_num : ", validate_test_num)
print("validate_num_nodes : ", validate_num_nodes)
print("radius : ", radius)
print("update_timestep : " + str(update_timestep) + " timesteps")
print("validate_episode_num : " + str(validate_episode_num) + " timesteps")
print("PPO K epochs : ", K_epochs)
print("update batch size : ", batch_size)
print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate actor : ", lr_critic)
print("max training timesteps : ", max_training_timesteps)
print("dis of one step : ", dis)
print("done thresh of one episode : ", done_thresh)
print("max thread num : ", max_thread_num)

if random_seed:
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
print("--------------------------------------------------------------------------------------------")

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
train_log_f = open(train_log_f_name, "w+")
train_log_f.write('episode,timestep,reward\n')
# logging file
train_diff_f = open(train_diff_f_name, "w+")
train_diff_f.write('episode,timestep,diff,ideal_diff,noisy_diff\n')
# logging file
train_acc_f = open(train_acc_f_name, "w+")
train_acc_f.write('episode,timestep,acc,ideal_acc,noisy_acc\n')

# logging file
val_log_f = open(val_log_f_name, "w+")
val_log_f.write('episode,timestep,reward\n')
# logging file
val_diff_f = open(val_diff_f_name, "w+")
val_diff_f.write('episode,timestep,diff,ideal_diff,noisy_diff\n')
# logging file
val_acc_f = open(val_acc_f_name, "w+")
val_acc_f.write('episode,timestep,acc,ideal_acc,noisy_acc\n')

# initialize a PPO agent
ppo_agent = PPO(lr_actor, lr_critic, batch_size, gamma, K_epochs,
                eps_clip, has_continuous_action_space, action_std)
time_step = 0
ideal_total_success_num = 0
ideal_total_success_diff = 0
noisy_total_success_num = 0
noisy_total_success_diff = 0
print_running_reward = 0
i_episode = 0

validate_time_step = 0
ideal_validate_total_success_num = 0
ideal_validate_total_success_diff = 0
noisy_validate_total_success_num = 0
noisy_validate_total_success_diff = 0
validate_print_running_reward = 0
validate_i_episode = 0

ppo_agent_lock = threading.Lock()
time_step_lock = threading.Lock()
variance_lock = threading.Lock()

validate_time_step_lock = threading.Lock()
validate_variance_lock = threading.Lock()

dataset_dir = data + "xxx/ppo/datasets" + str(run_num) + "/"
if os.path.exists(dataset_dir) is False:
    os.makedirs(dataset_dir, exist_ok=True)
else:
    del_file(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

train_points = get_start_end(test_num=train_test_num, num_nodes=train_num_nodes, radius=radius)
validate_points = get_start_end(test_num=validate_test_num, num_nodes=validate_num_nodes, radius=radius)

paths, path_labels = get_big_map(path=bigmap_dir)

print("============================================================================================")
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("============================================================================================")

train()

train_log_f.close()
train_diff_f.close()
train_acc_f.close()
val_log_f.close()
val_diff_f.close()
val_acc_f.close()

# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")
