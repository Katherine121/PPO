import argparse
import random
import torch
from torch.backends import cudnn
import numpy as np
import threading
from datetime import datetime

from PPO import PPO, MultiExpertPPO
from DDPG import DDPG
from td3 import TD3
from mygym import MyUAVgym
from cor import *


def train_thread(start_episode, end_episode):
    global ppo_agent
    global time_step
    global ideal_total_success_num
    global ideal_total_success_diff
    global noisy_total_success_num
    global noisy_total_success_diff
    global train_spl
    global print_running_reward
    global i_episode

    env = MyUAVgym(dis=args.dis, done_thresh=args.done_thresh, max_step_num=args.max_ep_len,
                   points=train_points, paths=paths, path_labels=path_labels)
    if args.random_seed:
        env.seed(args.random_seed)

    # training loop
    for index in range(start_episode, end_episode):
        if os.path.exists(args.dataset_dir) is False:
            os.makedirs(args.dataset_dir, exist_ok=True)

        state, path_list, episode_dir = env.reset(dataset_dir=args.dataset_dir, index=index)
        current_ep_reward = 0

        for t in range(1, args.max_ep_len):
            # select action with policy
            action = ppo_agent.select_action(state, path_list, episode_dir, ppo_agent_lock)
            state, path_list, reward, done, _, success, success_diff, step_num = env.step(action)

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
            train_spl += success * args.best_step_num / max(step_num, args.best_step_num)
            i_episode += 1
            print(episode_dir)

    env.close()


def train():
    global ppo_agent
    global time_step
    global ideal_total_success_num
    global ideal_total_success_diff
    global noisy_total_success_num
    global noisy_total_success_diff
    global train_spl
    global print_running_reward
    global i_episode

    best_acc = 0
    best_spl = 0
    best_diff = args.done_thresh

    while i_episode < args.max_training_timesteps:
        if i_episode >= 1000:
            cur_acc, cur_diff, cur_spl = validate()
            if cur_acc > best_acc:
                best_acc = cur_acc
                print("saving acc best model at : " + acc_best_path)
                ppo_agent.save(time_step, best_acc, cur_diff, cur_spl, acc_best_path)
                print("acc best model saved")
            if cur_spl > best_spl:
                best_spl = cur_spl
                print("saving spl best model at : " + spl_best_path)
                ppo_agent.save(time_step, cur_acc, cur_diff, best_spl, spl_best_path)
                print("spl best model saved")
            if cur_diff < best_diff:
                best_diff = cur_diff
                print("saving diff best model at : " + diff_best_path)
                ppo_agent.save(time_step, cur_acc, best_diff, cur_spl, diff_best_path)
                print("diff best model saved")

            print("============================================================================================")
            del_file(args.dataset_dir)

        # collect data
        # initialize threads
        episode_of_one_thread = args.collect_freq // args.max_thread_num
        remain = args.collect_freq % args.max_thread_num
        threads = []
        end_episode = i_episode
        for thread_index in range(0, args.max_thread_num):
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
        cur_spl = train_spl / args.collect_freq
        if total_success_num > 0:
            cur_acc = total_success_num / args.collect_freq
            cur_diff = total_success_diff / total_success_num

            ideal_acc = ideal_total_success_num / (args.collect_freq / 2)
            noisy_acc = noisy_total_success_num / (args.collect_freq / 2)

            if ideal_total_success_num > 0:
                ideal_diff = ideal_total_success_diff / ideal_total_success_num
            else:
                ideal_diff = 0
            if noisy_total_success_num > 0:
                noisy_diff = noisy_total_success_diff / noisy_total_success_num
            else:
                noisy_diff = 0

            print("total success acc is: ", cur_acc)
            train_acc_f.write('{},{},{},{},{},{}\n'.format(i_episode, time_step,
                                                           cur_acc, ideal_acc, noisy_acc, cur_spl))
            train_acc_f.flush()

            print("total success diff is: ", cur_diff)
            train_diff_f.write('{},{},{},{},{}\n'.format(i_episode, time_step,
                                                         cur_diff, ideal_diff, noisy_diff))
            train_diff_f.flush()
        else:
            cur_acc = 0
            cur_diff = args.done_thresh

            print("total success acc is: 0")
            train_acc_f.write('{},{},{},{},{},{}\n'.format(i_episode, time_step,
                                                        cur_acc, cur_acc, cur_acc, cur_spl))
            train_acc_f.flush()

            print("total success diff is: 0")
            train_diff_f.write('{},{},{},{},{}\n'.format(i_episode, time_step,
                                                         cur_diff, cur_diff, cur_diff))
            train_diff_f.flush()
        ideal_total_success_num = 0
        ideal_total_success_diff = 0
        noisy_total_success_num = 0
        noisy_total_success_diff = 0
        train_spl = 0

        # print average reward till last episode
        print_avg_reward = print_running_reward / args.collect_freq
        print_avg_reward = round(print_avg_reward, 4)
        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                print_avg_reward))
        train_log_f.write('{},{},{}\n'.format(i_episode, time_step, print_avg_reward))
        train_log_f.flush()
        print_running_reward = 0
        print("============================================================================================")

        # save agent
        print("saving model at : " + checkpoint_path)
        ppo_agent.save(time_step, cur_acc, cur_diff, cur_spl, checkpoint_path)
        print("model saved")
        end_time = datetime.now().replace(microsecond=0)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")

        # update agent
        ppo_agent.update(args.dataset_dir)
        del_file(args.dataset_dir)
        print("============================================================================================")

        # if continuous action space; then decay action std of ouput action distribution
        if has_continuous_action_space and i_episode % args.action_std_decay_freq == 0:
            ppo_agent.decay_action_std(args.action_std_decay_rate, args.min_action_std)


def validate_thread(start_episode, end_episode):
    global ppo_agent
    global validate_time_step
    global ideal_validate_total_success_num
    global ideal_validate_total_success_diff
    global noisy_validate_total_success_num
    global noisy_validate_total_success_diff
    global validate_spl
    global validate_print_running_reward
    global validate_i_episode

    env = MyUAVgym(dis=args.dis, done_thresh=args.done_thresh, max_step_num=args.max_ep_len,
                   points=validate_points, paths=paths, path_labels=path_labels)
    if args.random_seed:
        env.seed(args.random_seed)

    # training loop
    for index in range(start_episode, end_episode):
        if os.path.exists(args.dataset_dir) is False:
            os.makedirs(args.dataset_dir, exist_ok=True)

        state, path_list, episode_dir = env.reset(dataset_dir=args.dataset_dir, index=index)
        current_ep_reward = 0

        for t in range(1, args.max_ep_len):
            # select action with policy
            action = ppo_agent.select_action(state, path_list, episode_dir, ppo_agent_lock)
            state, path_list, reward, done, _, success, success_diff, step_num = env.step(action)

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
            validate_spl += success * args.best_step_num / max(step_num, args.best_step_num)
            validate_i_episode += 1

    env.close()


def validate():
    global ppo_agent
    global validate_time_step
    global ideal_validate_total_success_num
    global noisy_validate_total_success_num
    global ideal_validate_total_success_diff
    global noisy_validate_total_success_diff
    global validate_spl
    global validate_print_running_reward
    global validate_i_episode

    # collect data
    # initialize threads
    episode_of_one_thread = args.validate_episode_num // args.max_thread_num
    remain = args.validate_episode_num % args.max_thread_num
    threads = []
    end_episode = validate_i_episode
    for thread_index in range(0, args.max_thread_num):
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
    cur_spl = validate_spl / args.validate_episode_num
    if validate_total_success_num > 0:
        cur_acc = validate_total_success_num / args.validate_episode_num
        cur_diff = validate_total_success_diff / validate_total_success_num

        ideal_acc = ideal_validate_total_success_num / (args.validate_episode_num / 2)
        noisy_acc = noisy_validate_total_success_num / (args.validate_episode_num / 2)

        if ideal_validate_total_success_num > 0:
            ideal_diff = ideal_validate_total_success_diff / ideal_validate_total_success_num
        else:
            ideal_diff = 0
        if noisy_validate_total_success_num > 0:
            noisy_diff = noisy_validate_total_success_diff / noisy_validate_total_success_num
        else:
            noisy_diff = 0

        print("total success acc is: ", cur_acc)
        val_acc_f.write(
            '{},{},{},{},{},{}\n'.format(validate_i_episode, validate_time_step,
                                         cur_acc, ideal_acc, noisy_acc, cur_spl))
        val_acc_f.flush()

        print("total success diff is: ", cur_diff)
        val_diff_f.write(
            '{},{},{},{},{}\n'.format(validate_i_episode, validate_time_step,
                                         cur_diff, ideal_diff, noisy_diff))
        val_diff_f.flush()
    else:
        cur_acc = 0
        cur_diff = args.done_thresh

        print("total success acc is: 0")
        val_acc_f.write('{},{},{},{},{},{}\n'.format(validate_i_episode, validate_time_step,
                                                     cur_acc, cur_acc, cur_acc, cur_spl))
        val_acc_f.flush()

        print("total success diff is: 0")
        val_diff_f.write(
            '{},{},{},{},{}\n'.format(validate_i_episode, validate_time_step,
                                         cur_diff, cur_diff, cur_diff))
        val_diff_f.flush()
    ideal_validate_total_success_num = 0
    ideal_validate_total_success_diff = 0
    noisy_validate_total_success_num = 0
    noisy_validate_total_success_diff = 0
    validate_spl = 0

    # print average reward till last episode
    print_avg_reward = validate_print_running_reward / args.validate_episode_num
    print_avg_reward = round(print_avg_reward, 4)
    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(validate_i_episode, validate_time_step,
                                                                            print_avg_reward))
    val_log_f.write('{},{},{}\n'.format(validate_i_episode, validate_time_step, print_avg_reward))
    val_log_f.flush()
    validate_print_running_reward = 0
    print("============================================================================================")

    return cur_acc, cur_diff, cur_spl


# def test_multi_expert():
#     global ppo_agent
#     global time_step
#     global ideal_total_success_num
#     global ideal_total_success_diff
#     global noisy_total_success_num
#     global noisy_total_success_diff
#     global print_running_reward
#     global i_episode
#
#     best_acc = 0
#     best_diff = args.done_thresh
#
#     cur_acc, cur_diff = validate()
#     if cur_acc > best_acc or (cur_acc == best_acc and cur_diff <= best_diff):
#         best_acc = cur_acc
#         best_diff = cur_diff
#         print("saving best model at : " + best_path)
#         ppo_agent.save(time_step, best_acc, best_path)
#         print("best model saved")
#     print("============================================================================================")
#     del_file(args.dataset_dir)


parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--run_num', type=int)
parser.add_argument('--server_path', type=str, default="../../../mnt/nfs/")
parser.add_argument('--bigmap_dir', type=str, default="wyx/bigmap")
parser.add_argument('--dataset_dir', type=str, default="wyx/ppo/datasets")
parser.add_argument('--log_dir', type=str, default="PPO_logs")
parser.add_argument('--save_dir', type=str, default="PPO_preTrained")
parser.add_argument('--max_ep_len', type=int, default=256)
parser.add_argument('--collect_freq', type=int, default=0)
parser.add_argument('--train_test_num', type=int, default=5)
parser.add_argument('--train_num_nodes', type=int, default=20)
parser.add_argument('--validate_test_num', type=int, default=1)
parser.add_argument('--validate_num_nodes', type=int, default=100)
parser.add_argument('--radius', type=int, default=1000)
parser.add_argument('--update_episode_num', type=int)
parser.add_argument('--validate_episode_num', type=int)

parser.add_argument('--K_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr_actor', type=int, default=0.0003)
parser.add_argument('--lr_critic', type=int, default=0.001)
parser.add_argument('--max_training_timesteps', type=int, default=200)
parser.add_argument('--dis', type=int, default=100)
parser.add_argument('--done_thresh', type=int, default=50)
parser.add_argument('--max_thread_num', type=int, default=8)
parser.add_argument('--random_seed', type=int, default=0)
# moe
parser.add_argument('--num_experts', type=int, default=5)
parser.add_argument('--noisy_gating', type=bool, default=True)
parser.add_argument('--k', type=int, default=1)

parser.add_argument('--action_std', type=int, default=0.6)
parser.add_argument('--action_std_decay_rate', type=int, default=0.05)
parser.add_argument('--min_action_std', type=int, default=0.1)
parser.add_argument('--action_std_decay_freq', type=int, default=2.5e5)
parser.add_argument('--eps_clip', type=int, default=0.2)
parser.add_argument('--gamma', type=int, default=0.99)
parser.add_argument('--best_step_num', type=int)

args = parser.parse_args()
print(args)

env_name = "UAVnavigation"
has_continuous_action_space = True  # continuous action space; else discrete

args.bigmap_dir = args.server_path + args.bigmap_dir
args.update_episode_num = args.train_test_num * args.train_num_nodes * \
                          len(HEIGHT_NOISE) * len(NOISE_DB)
args.collect_freq = args.update_episode_num
args.validate_episode_num = args.validate_test_num * args.validate_num_nodes * \
                            len(HEIGHT_NOISE) * len(NOISE_DB)

args.max_training_timesteps = args.max_training_timesteps * args.update_episode_num  # break training loop if timeteps > args.max_training_timesteps
args.action_std_decay_freq = int(
    args.action_std_decay_freq / args.max_ep_len)  # action_std decay frequency (in num timesteps)

args.best_step_num = int(args.radius / args.dis)

# bigmap
paths, path_labels = get_big_map(path=args.bigmap_dir)

# captured paths
args.dataset_dir = args.server_path + "wyx/ppo/datasets" + str(args.run_num) + "/"
if os.path.exists(args.dataset_dir) is False:
    os.makedirs(args.dataset_dir, exist_ok=True)
else:
    del_file(args.dataset_dir)
    os.makedirs(args.dataset_dir, exist_ok=True)

# sampled starting and end points
train_points = get_start_end(test_num=args.train_test_num, num_nodes=args.train_num_nodes, radius=args.radius)
validate_points = get_start_end(test_num=args.validate_test_num, num_nodes=args.validate_num_nodes, radius=args.radius)
print("train_points" + str(len(train_points)))

# log results
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir, exist_ok=True)
args.log_dir = args.log_dir + '/' + env_name
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir, exist_ok=True)

train_log_f_name = args.log_dir + '/PPO_' + env_name + "_trainlog_" + str(args.run_num) + ".csv"
train_diff_f_name = args.log_dir + '/PPO_' + env_name + "_traindiff_" + str(args.run_num) + ".csv"
train_acc_f_name = args.log_dir + '/PPO_' + env_name + "_trainacc_" + str(args.run_num) + ".csv"
val_log_f_name = args.log_dir + '/PPO_' + env_name + "_vallog_" + str(args.run_num) + ".csv"
val_diff_f_name = args.log_dir + '/PPO_' + env_name + "_valdiff_" + str(args.run_num) + ".csv"
val_acc_f_name = args.log_dir + '/PPO_' + env_name + "_valacc_" + str(args.run_num) + ".csv"
print("logging at : " + train_log_f_name)

# logging file
train_log_f = open(train_log_f_name, "w+")
train_log_f.write('episode,timestep,reward\n')
train_diff_f = open(train_diff_f_name, "w+")
train_diff_f.write('episode,timestep,diff,ideal_diff,noisy_diff\n')
train_acc_f = open(train_acc_f_name, "w+")
train_acc_f.write('episode,timestep,acc,ideal_acc,noisy_acc,spl\n')

val_log_f = open(val_log_f_name, "w+")
val_log_f.write('episode,timestep,reward\n')
val_diff_f = open(val_diff_f_name, "w+")
val_diff_f.write('episode,timestep,diff,ideal_diff,noisy_diff\n')
val_acc_f = open(val_acc_f_name, "w+")
val_acc_f.write('episode,timestep,acc,ideal_acc,noisy_acc,spl\n')

# save checkpoints
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)
args.save_dir = args.save_dir + '/' + env_name + '/'
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)

checkpoint_path = args.save_dir + "PPO_{}_{}_{}.pth".format(env_name, args.random_seed, args.run_num)
acc_best_path = args.save_dir + "PPO_{}_{}_{}_acc_best.pth".format(env_name, args.random_seed, args.run_num)
diff_best_path = args.save_dir + "PPO_{}_{}_{}_diff_best.pth".format(env_name, args.random_seed, args.run_num)
spl_best_path = args.save_dir + "PPO_{}_{}_{}_spl_best.pth".format(env_name, args.random_seed, args.run_num)
print("save checkpoint path : " + checkpoint_path)

if args.random_seed:
    print("setting random seed to ", args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
print("--------------------------------------------------------------------------------------------")

torch.autograd.set_detect_anomaly(True)
# reduce CPU usage, use it after the model is loaded onto the GPU
torch.set_num_threads(1)
cudnn.benchmark = True

# initialize a PPO agent
ppo_agent = PPO(args.lr_actor, args.lr_critic, args.batch_size, args.gamma, args.K_epochs,
                args.eps_clip, has_continuous_action_space, args.action_std,
                args.num_experts, args.noisy_gating, args.k)
# ppo_agent = MultiExpertPPO([6, 7, 8, 9, 10], True,
#                            args.lr_actor, args.lr_critic, args.batch_size, args.gamma, args.K_epochs,
#                            args.eps_clip, has_continuous_action_space, args.action_std)
time_step = 0
ideal_total_success_num = 0
ideal_total_success_diff = 0
noisy_total_success_num = 0
noisy_total_success_diff = 0
train_spl = 0
print_running_reward = 0
i_episode = 0

validate_time_step = 0
ideal_validate_total_success_num = 0
ideal_validate_total_success_diff = 0
noisy_validate_total_success_num = 0
noisy_validate_total_success_diff = 0
validate_spl = 0
validate_print_running_reward = 0
validate_i_episode = 0

ppo_agent_lock = threading.Lock()
time_step_lock = threading.Lock()
variance_lock = threading.Lock()

validate_time_step_lock = threading.Lock()
validate_variance_lock = threading.Lock()

print("============================================================================================")
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("============================================================================================")

train()
# test_multi_expert()

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
