import os

import torch
import torch.nn as nn
import torchvision
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torchvision.models import MobileNet_V3_Small_Weights, AlexNet

from model import ARTransformer
from utils import UncertaintyLoss
from vit import ViT, Actor, Critic

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
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.labels = []
        self.actions = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.labels[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        del self.is_terminals[:]


def load_weight(backbone1, checkpoint_path):
    backbone1_dict = backbone1.state_dict()
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(checkpoint['best_acc1'])
    state_dict = checkpoint['state_dict']
    for key in backbone1_dict:
        backbone1_dict[key] = state_dict["module." + key]
    backbone1.load_state_dict(backbone1_dict)
    print("=> loaded pre-trained model '{}'".format(checkpoint_path))


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            # 用init*init填充形状为(action_dim,)的tensor
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # if has_continuous_action_space:
        #     self.actor = nn.Sequential(
        #         nn.Linear(state_dim, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, action_dim),
        #         nn.Tanh()
        #     )
        # else:
        #     self.actor = nn.Sequential(
        #         nn.Linear(state_dim, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, action_dim),
        #         nn.Softmax(dim=-1)
        #     )
        # # critic
        # self.critic = nn.Sequential(
        #     nn.Linear(state_dim, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 1),
        # )

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
    def act(self, state):
        if self.has_continuous_action_space:
            # 输入状态，经过神经网络，输出动作（连续值）
            action_mean = self.actor(state)
            # 取self.action_var的对角线元素
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            # 输入：分布的平均值、正定协方差矩阵
            # 输出：由均值向量和协方差矩阵参数化的多元正态(也称为高斯)分布
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

        return action.detach(), action_logprob.detach(), state_val.detach(), \
               action.detach(), state_val.detach()

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

        return action_logprobs, state_values, dist_entropy, action_logprobs, state_values


# def normalization(state):
#     data = state.clone().detach()
#     if len(data.shape) == 1:
#         data[0] -= 23.55
#         data[1] -= 120.3
#         data[2] -= 23.55
#         data[3] -= 120.3
#     elif len(data.shape) == 2:
#         data[:, 0] -= 23.55
#         data[:, 1] -= 120.3
#         data[:, 2] -= 23.55
#         data[:, 3] -= 120.3
#     mean = torch.mean(data, dim=-1, keepdim=True)
#     std = torch.std(data, dim=-1, keepdim=True)
#     normData = (data - mean) / std
#     return normData


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, batch_size, wd, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # state_dict = torch.load("PPO_preTrained/UAVnavigation/PPO_UAVnavigation_0_0_best.pth")
        # print(state_dict["best_acc"])
        # self.policy.load_state_dict(state_dict["state_dict"])
        # for name, param in self.policy.named_parameters():
        #     print(param)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
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

    def select_action(self, state, labels):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val, actor_label, critic_label = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.labels.append(labels)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val, actor_label, critic_label = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

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
            size = len(self.buffer.actions)
            for i in range(0, size // self.batch_size):
                start = i * self.batch_size
                end = start + self.batch_size
                logprobs, state_values, dist_entropy, actor_label, critic_label = \
                    self.policy.evaluate(old_states[start: end], old_actions[start: end])

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

            if size % self.batch_size > 0:
                start = i * self.batch_size
                end = size - 1
                logprobs, state_values, dist_entropy, actor_label, critic_label = \
                    self.policy.evaluate(old_states[start: end], old_actions[start: end])

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

        # clear buffer
        self.buffer.clear()

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
    state_dict = torch.load("PPO_preTrained/UAVnavigation/PPO_UAVnavigation_0_5_best.pth")
    print(state_dict)
    state_dict = torch.load("PPO_preTrained/UAVnavigation/PPO_UAVnavigation_0_6_best.pth")
    print(state_dict)