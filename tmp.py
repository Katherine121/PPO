
# ==============================================================================
# run_num = 22
# data = "../../../mnt/nfs/"

# max_ep_len = 256  # max timesteps in one episode
# train_test_num = 5
# train_num_nodes = 20
# validate_test_num = 1
# validate_num_nodes = 100
# radius = 1000

# K_epochs = 30  # update policy for K epochs in one PPO update
# batch_size = 128
# lr_actor = 0.0003  # learning rate for actor network
# lr_critic = 0.001  # learning rate for critic network

# dis = 100
# done_thresh = 50
# max_thread_num = 8

# random_seed = 0  # set random seed if required (0 = no random seed)

# ==============================================================================
# 不需要修改的参数
# action_std = 0.6  # starting std for action distribution (Multivariate Normal)
# action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
# min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
#
# eps_clip = 0.2  # clip parameter for PPO
# gamma = 0.99  # discount factor
# ==============================================================================

# log_dir = "PPO_logs"
# save_dir = "PPO_preTrained"
# args.ppo_supervised_dataset_dir = args.data + args.ppo_supervised_dataset_dir + str(args.run_num) + "/"
# if os.path.exists(args.ppo_supervised_dataset_dir) is False:
#     os.makedirs(args.ppo_supervised_dataset_dir, exist_ok=True)

############# print all hyperparameters #############
# print("max timesteps per episode : ", max_ep_len)
# print("train_test_num : ", train_test_num)
# print("train_num_nodes : ", train_num_nodes)
# print("validate_test_num : ", validate_test_num)
# print("validate_num_nodes : ", validate_num_nodes)
# print("radius : ", radius)
# print("update_timestep : " + str(update_episode_num) + " timesteps")
# print("validate_episode_num : " + str(validate_episode_num) + " timesteps")
# print("PPO K epochs : ", K_epochs)
# print("update batch size : ", batch_size)
# print("optimizer learning rate actor : ", lr_actor)
# print("optimizer learning rate actor : ", lr_critic)
# print("max training timesteps : ", max_training_timesteps)
# print("dis of one step : ", dis)
# print("done thresh of one episode : ", done_thresh)
# print("max thread num : ", max_thread_num)

# if has_continuous_action_space:
#     print("Initializing a continuous action space policy")
#     print("starting std of action distribution : ", action_std)
#     print("decay rate of std of action distribution : ", action_std_decay_rate)
#     print("minimum std of action distribution : ", min_action_std)
#     print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
# else:
#     print("Initializing a discrete action space policy")
#
# print("PPO epsilon clip : ", eps_clip)
# print("discount factor (gamma) : ", gamma)
# print("optimizer learning rate critic : ", lr_critic)
#####################################################
