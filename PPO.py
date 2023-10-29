import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torchvision.models import MobileNet_V3_Small_Weights

from model import ARTransformer

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
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            # 用init*init填充形状为(action_dim,)的tensor
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # # actor
        # backbone = torchvision.models.mobilenet_v3_small(weights=(MobileNet_V3_Small_Weights.IMAGENET1K_V1))
        # if has_continuous_action_space:
        #     self.actor = ARTransformer(
        #                 backbone=backbone,
        #                 backbone_pretrained="PPO_preTrained/mobilenet_pretrain.pth.tar",
        #                 extractor_dim=576,
        #                 action_dim=action_dim,
        #                 len=6,
        #                 dim=512,
        #                 depth=4,
        #                 heads=8,
        #                 dim_head=64,
        #                 mlp_dim=1024,
        #                 dropout=0.1,
        #                 emb_dropout=0.1,
        #                 is_actor=True
        #             )
        #     self.actor = self.actor.cuda()
        #     for name, param in self.actor.named_parameters():
        #         if param.requires_grad:
        #             print(name)
        # else:
        #     self.actor = ARTransformer(
        #                 backbone=backbone,
        #                 backbone_pretrained="PPO_preTrained/mobilenet_pretrain.pth.tar",
        #                 extractor_dim=576,
        #                 action_dim=action_dim,
        #                 len=6,
        #                 dim=512,
        #                 depth=4,
        #                 heads=8,
        #                 dim_head=64,
        #                 mlp_dim=1024,
        #                 dropout=0.1,
        #                 emb_dropout=0.1,
        #                 is_actor=True
        #             )
        #     self.actor = self.actor.cuda()
        # # critic
        # self.critic = ARTransformer(
        #                 backbone=backbone,
        #                 backbone_pretrained="PPO_preTrained/mobilenet_pretrain.pth.tar",
        #                 extractor_dim=576,
        #                 action_dim=action_dim,
        #                 len=6,
        #                 dim=512,
        #                 depth=4,
        #                 heads=8,
        #                 dim_head=64,
        #                 mlp_dim=1024,
        #                 dropout=0.1,
        #                 emb_dropout=0.1,
        #                 is_actor=False
        #             )
        # self.critic = self.critic.cuda()
            # actor
            if has_continuous_action_space:
                self.actor = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, action_dim),
                    nn.Tanh()
                )
            else:
                self.actor = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, action_dim),
                    nn.Softmax(dim=-1)
                )
            # critic
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

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
        input = normalization(state)
        if self.has_continuous_action_space:
            # 输入状态，经过神经网络，输出动作（连续值）
            action_mean = self.actor(input)
            # 取self.action_var的对角线元素
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            # 输入：分布的平均值、正定协方差矩阵
            # 输出：由均值向量和协方差矩阵参数化的多元正态(也称为高斯)分布
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(input)
            dist = Categorical(action_probs)

        # 离散动作：根据概率进行采样
        # 连续动作：不采样，确定性动作
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        # 输入状态，经过神经网络，输出奖励
        state_val = self.critic(input)

        return action.detach(), action_logprob.detach(), state_val.detach()

    # new policy
    def evaluate(self, state, action):
        input = normalization(state)
        if self.has_continuous_action_space:
            action_mean = self.actor(input)

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
            action_probs = self.actor(input)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(input)

        return action_logprobs, state_values, dist_entropy


def normalization(state):
    data = state.clone().detach()
    if len(data.shape) == 1:
        data[0] -= 23.55
        data[1] -= 120.3
        data[2] -= 23.55
        data[3] -= 120.3
    elif len(data.shape) == 2:
        data[:, 0] -= 23.55
        data[:, 1] -= 120.3
        data[:, 2] -= 23.55
        data[:, 3] -= 120.3
    mean = torch.mean(data, dim=-1, keepdim=True)
    std = torch.std(data, dim=-1, keepdim=True)
    normData = (data - mean)/std
    return normData


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, wd, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
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

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

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
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # with open("loss.txt", "a") as file1:
        #     file1.write(str(loss.mean()) + "\n")

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
