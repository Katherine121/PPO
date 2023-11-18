import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from torch.backends import cudnn

from PPO import PPO
from mygym import MyUAVgym


################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "UAVnavigation"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 64  # max timesteps in one episode
    max_training_timesteps = int(6e6)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 100  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 100  # log avg reward in the interval (in num timesteps)
    save_model_freq = max_ep_len * 500  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    # 4 episode, update 1 policy
    update_timestep = max_ep_len * 100  # update policy every n timesteps
    K_epochs = 30  # update policy for K epochs in one PPO update
    batch_size = 128

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor
    # gamma = 0

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network
    wd = 0.1

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = MyUAVgym(len=6, bigmap_dir="../../../mnt/nfs/wyx/bigmap", num_nodes=10, dis=250,
                   done_thresh=125, max_step_num=max_ep_len)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    current_num_files = next(os.walk(log_dir))[2]
    run_num = 6

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"
    #### create new log file for each run
    diff_f_name = log_dir + '/PPO_' + env_name + "_diff_" + str(run_num) + ".csv"
    #### create new log file for each run
    acc_f_name = log_dir + '/PPO_' + env_name + "_acc_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num)
    best_path = directory + "PPO_{}_{}_{}_best.pth".format(env_name, random_seed, run_num)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
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
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")
    torch.autograd.set_detect_anomaly(True)
    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, batch_size, wd, gamma, K_epochs,
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
    complete_episode_num = 0
    datasets_path = "datasets" + str(run_num)
    if os.path.exists(datasets_path) is True:
        del_file(datasets_path)

    # training loop
    while time_step <= max_training_timesteps:
        if os.path.exists(datasets_path) is False:
            os.mkdir(datasets_path)
        episode_files_dir = datasets_path + "/" + str(complete_episode_num) + "/"
        if os.path.exists(episode_files_dir) is False:
            os.mkdir(episode_files_dir)

        state, labels = env.reset(episode_files_dir)
        current_ep_reward = 0

        for t in range(1, max_ep_len):

            # select action with policy
            action = ppo_agent.select_action(state, labels)
            state, labels, reward, done, _, success, success_diff = env.step(action)
            total_success_num += success
            if success == 1:
                total_success_diff += success_diff

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        complete_episode_num += 1

        # update PPO agent
        if complete_episode_num % 100 == 0:

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
                    print("model saved")
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

        # save model weights
        if complete_episode_num % 500 == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(time_step, -1, checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

        if complete_episode_num % 100 == 0:
            ppo_agent.update()
            del_file(datasets_path)

        # if continuous action space; then decay action std of ouput action distribution
        if has_continuous_action_space and complete_episode_num % 1000 == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


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


if __name__ == '__main__':
    train()
