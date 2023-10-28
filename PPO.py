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
        # self.max_size = 8000

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    # def check(self):
    #     if len(self.actions) > self.max_size:
    #         self.actions.pop(0)
    #         self.states.pop(0)
    #         self.logprobs.pop(0)
    #         self.rewards.pop(0)
    #         self.state_values.pop(0)
    #         self.is_terminals.pop(0)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            # 用init*init填充形状为(action_dim,)的tensor
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        backbone = torchvision.models.mobilenet_v3_small(weights=(MobileNet_V3_Small_Weights.IMAGENET1K_V1))
        if has_continuous_action_space:
            self.actor = ARTransformer(
                        backbone=backbone,
                        backbone_pretrained="PPO_preTrained/mobilenet_pretrain.pth.tar",
                        extractor_dim=576,
                        action_dim=action_dim,
                        len=6,
                        dim=512,
                        depth=4,
                        heads=8,
                        dim_head=64,
                        mlp_dim=1024,
                        dropout=0.1,
                        emb_dropout=0.1,
                        is_actor=True
                    )
            self.actor = self.actor.cuda()
            for name, param in self.actor.named_parameters():
                if param.requires_grad:
                    print(name)
        else:
            self.actor = ARTransformer(
                        backbone=backbone,
                        backbone_pretrained="PPO_preTrained/mobilenet_pretrain.pth.tar",
                        extractor_dim=576,
                        action_dim=action_dim,
                        len=6,
                        dim=512,
                        depth=4,
                        heads=8,
                        dim_head=64,
                        mlp_dim=1024,
                        dropout=0.1,
                        emb_dropout=0.1,
                        is_actor=True
                    )
            self.actor = self.actor.cuda()
        # critic
        self.critic = ARTransformer(
                        backbone=backbone,
                        backbone_pretrained="PPO_preTrained/mobilenet_pretrain.pth.tar",
                        extractor_dim=576,
                        action_dim=action_dim,
                        len=6,
                        dim=512,
                        depth=4,
                        heads=8,
                        dim_head=64,
                        mlp_dim=1024,
                        dropout=0.1,
                        emb_dropout=0.1,
                        is_actor=False
                    )
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
    def __init__(self, state_dim, action_dim, lr_actor, wd, gamma, K_epochs, batch_size, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        #     {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        # ])
        parameters = list(filter(lambda p: p.requires_grad, self.policy.actor.parameters())) + \
                     list(filter(lambda p: p.requires_grad, self.policy.critic.parameters()))
        self.optimizer = torch.optim.Adam(parameters, lr=lr_actor, weight_decay=wd)
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

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
                # state = torch.FloatTensor(state).to(device)
                state = [state[0].to(dtype=torch.float32).to(device), state[1].to(dtype=torch.float32).to(device)]
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
        # read reward as timestamp decreasing
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            # store reward: ri + λ*A(i+1) as timestamp increasing
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # state, action, logprobs, state_vals都是old policy的输入输出
        old_states = self.buffer.states
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        # 环境给出的奖励 - old policy神经网络输出的奖励
        # print(rewards.size())
        # print(old_state_values.size())
        advantages = rewards.detach() - old_state_values.detach()
        # advantages = rewards.detach()

        # Optimize policy for K epochs
        # old policy sample enough trajectories, and policy updates for K epochs
        for epoch_i in range(self.K_epochs):
            if epoch_i % 5 == 0:
                print("Epoch: " + str(epoch_i))
            # Evaluating old actions and values
            # policy input sampled states and output state_vals

            size = len(self.buffer.actions)
            for i in range(0, len(self.buffer.rewards) // self.batch_size):
                start = size - self.batch_size * (i + 1)
                end = size - self.batch_size * i
                # batch_size个[img, ang]组成的list
                img = [x[0] for x in old_states[start: end]]
                ang = [x[1] for x in old_states[start: end]]
                # batch_size个img组成的tensor
                img = torch.squeeze(torch.stack(img, dim=0)).detach().to(device)
                ang = torch.squeeze(torch.stack(ang, dim=0)).detach().to(device)
                logprobs, state_values, dist_entropy =\
                    self.policy.evaluate([img, ang], old_actions[start: end])

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
                # loss = -torch.min(surr1, surr2)

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                # self.lr_scheduler.step()

            if size - self.batch_size * i > 0:
                start = 0
                end = size - self.batch_size * i
                # batch_size个[img, ang]组成的list
                img = [x[0] for x in old_states[start: end]]
                ang = [x[1] for x in old_states[start: end]]
                # batch_size个img组成的tensor
                img = torch.squeeze(torch.stack(img, dim=0)).detach().to(device)
                ang = torch.squeeze(torch.stack(ang, dim=0)).detach().to(device)
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate([img, ang], old_actions[start: end])

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
                # loss = -torch.min(surr1, surr2)

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                # self.lr_scheduler.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
