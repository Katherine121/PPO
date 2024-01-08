import os
import random

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from dataset import UAVdataset
from vit import Actor, Critic

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
# class RolloutBuffer:
#     def __init__(self):
#         self.states = []
#         self.actions = []
#         self.logprobs = []
#         self.state_values = []
#         self.rewards = []
#         self.is_terminals = []
#
#     def clear(self):
#         del self.states[:]
#         del self.actions[:]
#         del self.logprobs[:]
#         del self.state_values[:]
#         del self.rewards[:]
#         del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = 2
            # 用init*init填充形状为(action_dim,)的tensor
            self.action_var = torch.full((2,), action_std_init * action_std_init).to(device)

        self.actor = Actor()
        self.actor = self.actor.cuda()
        self.critic = Critic()
        self.critic = self.critic.cuda()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    # old policy
    def act(self, state, ppo_agent_lock):
        if self.has_continuous_action_space:
            # 输入状态，经过神经网络，输出动作（连续值）
            action_mean = self.actor(state)
            # 取self.action_var的对角线元素
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            # 输入：分布的平均值、正定协方差矩阵
            # 输出：由均值向量和协方差矩阵参数化的多元正态(也称为高斯)分布
            with ppo_agent_lock:
                dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        # 离散动作：根据概率进行采样
        # 连续动作：不采样，确定性动作
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        # 输入状态，经过神经网络，输出奖励
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    # new policy
    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            # 将self.action_var扩展到和action_mean一样的维度
            action_var = self.action_var.expand_as(action_mean)
            # 将action_var中的值作为对角，形成对角矩阵
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            # (all_old_state_num, action_dim)
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, lr_actor, lr_critic, batch_size, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # self.buffer = RolloutBuffer()

        self.policy = ActorCritic(has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        # state_dict1 = torch.load("PPO_preTrained/UAVnavigation/PPO_UAVnavigation_0_18_best.pth")
        # print(state_dict1["best_acc"])
        # state_dict1 = state_dict1["state_dict"]
        # self.policy.load_state_dict(state_dict1)

        self.policy_old = ActorCritic(has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, path_list, episode_dir, ppo_agent_lock):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state, ppo_agent_lock)

            # self.buffer.states.append(path_list)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)
            # self.buffer.state_values.append(state_val)
            # 写到文件里，这样多线程顺序才不会错
            record_file = os.path.join(episode_dir, "record.txt")
            with open(record_file, "a") as file1:
                file1.write(path_list[0] + " " + path_list[1]
                            + " " + str(action[0, 0].item()) + " " + str(action[0, 1].item())
                            + " " + str(action_logprob.item())
                            + " " + str(state_val.item())
                            + "\n")
            file1.close()

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)
            # self.buffer.state_values.append(state_val)

            return action.item()

    def update(self, datasets_path):
        # Monte Carlo estimate of returns
        # 从文本中读取记录
        buffer_states = []
        buffer_actions = []
        buffer_logprobs = []
        buffer_state_values = []
        buffer_rewards = []
        buffer_is_terminals = []

        rewards = []
        discounted_reward = 0

        episode_dirs = os.listdir(datasets_path)
        random.shuffle(episode_dirs)
        for episode_dir in episode_dirs:
            full_episode_dir = os.path.join(datasets_path, episode_dir)
            full_record_file = os.path.join(full_episode_dir, "record.txt")
            full_reward_file = os.path.join(full_episode_dir, "reward.txt")
            f1 = open(full_record_file, 'rt')
            for line in f1:
                line = line.strip('\n')
                line = line.split(' ')
                buffer_states.append((line[0], line[1]))
                buffer_actions.append((float(line[2]), float(line[3])))
                buffer_logprobs.append(float(line[4]))
                buffer_state_values.append(float(line[5]))
            f1.close()

            f2 = open(full_reward_file, 'rt')
            for line in f2:
                line = line.strip('\n')
                line = line.split(' ')
                buffer_rewards.append(float(line[0]))
                is_terminal = True if int(line[1]) == 1 else False
                buffer_is_terminals.append(is_terminal)
            f2.close()
        buffer_actions = torch.tensor(buffer_actions, dtype=torch.float32)
        buffer_logprobs = torch.tensor(buffer_logprobs, dtype=torch.float32)
        buffer_state_values = torch.tensor(buffer_state_values, dtype=torch.float32)

        for reward, is_terminal in zip(reversed(buffer_rewards), reversed(buffer_is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_actions = buffer_actions.detach().to(device)
        old_logprobs = buffer_logprobs.detach().to(device)
        old_state_values = buffer_state_values.detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # batch_list = os.listdir(datasets_path)
        # batch_list.sort(key=lambda x: int(x[:-3]))
        train_dataset = UAVdataset(buffer_states)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=8, pin_memory=True, drop_last=False)

        # Optimize policy for K epochs
        for epoch_i in range(self.K_epochs):

            # # Evaluating old actions and values
            # logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            #
            # # match state_values tensor dimensions with rewards tensor
            # state_values = torch.squeeze(state_values)
            #
            # # Finding the ratio (pi_theta / pi_theta__old)
            # ratios = torch.exp(logprobs - old_logprobs.detach())
            #
            # # Finding Surrogate Loss
            # surr1 = ratios * advantages
            # surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            #
            # # # final loss of clipped objective PPO
            # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            #
            # # take gradient step
            # self.optimizer.zero_grad()
            # loss.mean().backward()
            # self.optimizer.step()
            total_loss = 0
            index = 0
            for i, old_states in enumerate(train_loader):
                old_states = old_states.to(device).to(dtype=torch.float32)
                start = index
                end = index + old_states.size(0)
                index = end
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions[start: end])

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                # e^(logp - logq) = e^(log(p/q)) = p/q
                ratios = torch.exp(logprobs - old_logprobs[start: end].detach())

                # Finding Surrogate Loss
                surr1 = ratios * advantages[start: end]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[start: end]

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards[start: end]) \
                       - 0.01 * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                total_loss += loss.mean().item()

            if epoch_i % 5 == 0:
                print("Epoch" + str(epoch_i) + ", Loss: " + str(total_loss / (i + 1)))

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # # clear buffer
        # self.buffer.clear()

    def save(self, time_step, best_acc, checkpoint_path):
        if best_acc == -1:
            torch.save(self.policy_old.state_dict(), checkpoint_path)
        else:
            state = {
                    'time_step': time_step,
                    'state_dict': self.policy_old.state_dict(),
                    'best_acc': best_acc,
                }
            torch.save(state, checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


if __name__ == '__main__':
    state_dict = torch.load("PPO_preTrained/UAVnavigation/PPO_UAVnavigation_0_15_best.pth")
    print(state_dict)
