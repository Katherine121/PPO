import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from actor_critic import Actor, Critic
from dataset import UAVdataset, SURFdataset, transform_surf_features

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


class ActorCritic(nn.Module):
    def __init__(self, has_continuous_action_space, action_std_init, num_experts, noisy_gating, k):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = 2
            # Fill the tensor in the shape of (action_dim,) with init * init
            self.action_var = torch.full((self.action_dim,), action_std_init * action_std_init).to(device)

        self.actor = Actor(num_experts=num_experts,
                           noisy_gating=noisy_gating,
                           k=k)
        self.actor = self.actor.cuda()
        self.critic = Critic(num_experts=num_experts,
                             noisy_gating=noisy_gating,
                             k=k)
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
            action_mean, moe_actor_loss = self.actor(state)
            # Take the diagonal element of self.action_var
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            # Input: mean of distribution, positive definite covariance matrix
            # Output: Multivariate normal (also known as Gaussian) distribution
            # parameterized by mean vector and covariance matrix
            with ppo_agent_lock:
                dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        state_val, moe_critic_loss = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    # new policy
    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean, moe_actor_loss = self.actor(state)

            # Extend self.action_var to the same dimension as action_mean
            action_var = self.action_var.expand_as(action_mean)
            # Take the value in action_var as a diagonal to form a diagonal matrix
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
        state_values, moe_critic_loss = self.critic(state)
        dist_entropy = dist.entropy()

        return action_logprobs, state_values, dist_entropy, moe_actor_loss, moe_critic_loss


class PPO:
    def __init__(self, lr_actor, lr_critic, batch_size, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init, num_experts, noisy_gating, k):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # self.buffer = RolloutBuffer()

        self.policy = ActorCritic(has_continuous_action_space, action_std_init,
                                  num_experts, noisy_gating, k).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        # state_dict1 = torch.load("PPO_preTrained/UAVnavigation/PPO_UAVnavigation_0_28_best.pth")
        # print(state_dict1["best_acc"])
        # state_dict1 = state_dict1["state_dict"]
        # self.policy.load_state_dict(state_dict1, strict=False)

        self.policy_old = ActorCritic(has_continuous_action_space, action_std_init,
                                      num_experts, noisy_gating, k).to(device)
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
                # state = transform_surf_features(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state, ppo_agent_lock)

            # self.buffer.states.append(path_list)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)
            # self.buffer.state_values.append(state_val)
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
        buffer_states = []
        buffer_actions = []
        buffer_logprobs = []
        buffer_state_values = []
        buffer_rewards = []
        buffer_is_terminals = []

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
        buffer_state_values = torch.tensor(buffer_state_values, dtype=torch.float32).to(device)

        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(buffer_rewards), reversed(buffer_is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # calculate advantages
        # Q_t - V_t = r_t + V_t+1 - V_t
        buffer_advantages = rewards.detach() - buffer_state_values.detach()

        train_dataset = UAVdataset(buffer_states, buffer_actions, buffer_logprobs, buffer_advantages, rewards)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        # Optimize policy for K epochs
        for epoch_i in range(self.K_epochs):

            total_loss = 0

            for i, (old_states, old_actions, old_logprobs, old_advantages, old_rewards) in enumerate(train_loader):
                old_states = old_states.to(device).to(dtype=torch.float32)
                old_actions = old_actions.to(device).to(dtype=torch.float32)
                old_logprobs = old_logprobs.to(device).to(dtype=torch.float32)
                old_advantages = old_advantages.to(device).to(dtype=torch.float32)
                old_rewards = old_rewards.to(device).to(dtype=torch.float32)

                # The probability of taking the old action in the new distribution,
                # the reward predicted by the new strategy,
                # and the probability distribution entropy of the new strategy
                logprobs, state_values, dist_entropy, moe_actor_loss, moe_critic_loss = \
                    self.policy.evaluate(old_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                # e^(logp - logq) = e^(log(p/q)) = p/q
                # Probability of taking the old action in the new distribution
                # /
                # probability of taking the old action in the old distribution
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                surr1 = ratios * old_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * old_advantages

                # final loss of clipped objective PPO
                # Don't let the gap between old and new strategies be too large;
                # Let the V predicted by the new strategy be equal to the cumulative reward as much as possible;
                # Increase probability distribution entropy as much as possible
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, old_rewards) \
                       - 0.01 * dist_entropy + moe_actor_loss + moe_critic_loss

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                total_loss += loss.mean().item()

            if epoch_i % 5 == 0:
                print("Epoch" + str(epoch_i) + ", Loss: " + str(total_loss / (i + 1)))

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, time_step, best_acc, best_diff, best_spl, checkpoint_path):
        if best_acc == -1:
            torch.save(self.policy_old.state_dict(), checkpoint_path)
        else:
            state = {
                'time_step': time_step,
                'state_dict': self.policy_old.state_dict(),
                'best_acc': best_acc,
                'best_diff': best_diff,
                'best_spl': best_spl
            }
            torch.save(state, checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class MultiExpertPPO:
    def __init__(self, expert_list, avg_or_random, lr_actor, lr_critic, batch_size, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # self.buffer = RolloutBuffer()
        self.experts = []
        self.avg_or_random = avg_or_random
        for expert_idx in expert_list:
            self.policy_idx = ActorCritic(has_continuous_action_space, action_std_init).to(device)
            self.optimizer = torch.optim.Adam([
                {'params': self.policy_idx.actor.parameters(), 'lr': lr_actor},
                {'params': self.policy_idx.critic.parameters(), 'lr': lr_critic}
            ])
            state_dict1 = torch.load("PPO_preTrained/UAVnavigation/PPO_UAVnavigation_0_" +
                                     str(expert_idx) + "_best.pth")
            print(state_dict1["best_acc"])
            state_dict1 = state_dict1["state_dict"]
            self.policy_idx.load_state_dict(state_dict1, strict=False)

            self.policy_old_idx = ActorCritic(has_continuous_action_space, action_std_init).to(device)
            self.policy_old_idx.load_state_dict(self.policy_idx.state_dict())
            self.experts.append(self.policy_old_idx)

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, path_list, episode_dir, ppo_agent_lock):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                # state = transform_surf_features(state).to(device)
                lat_action = 0
                lon_action = 0

                if self.avg_or_random:
                    for expert_policy_old in self.experts:
                        action, action_logprob, state_val = expert_policy_old.act(state, ppo_agent_lock)
                        lat_action += action[0, 0].item()
                        lon_action += action[0, 1].item()
                    lat_action /= len(self.experts)
                    lon_action /= len(self.experts)
                else:
                    random_idx = random.randint(0, len(self.experts) - 1)
                    action, action_logprob, state_val = self.experts[random_idx].act(state, ppo_agent_lock)
                    lat_action = action[0, 0].item()
                    lon_action = action[0, 1].item()

            record_file = os.path.join(episode_dir, "record.txt")
            with open(record_file, "a") as file1:
                file1.write(path_list[0] + " " + path_list[1]
                            + " " + str(lat_action) + " " + str(lon_action)
                            + " " + str(action_logprob.item())
                            + " " + str(state_val.item())
                            + "\n")
            file1.close()

            return np.array((lat_action, lon_action), dtype=np.float32)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)
            # self.buffer.state_values.append(state_val)

            return action.item()

    def save(self, time_step, best_acc, checkpoint_path):
        if best_acc == -1:
            torch.save(self.experts[0].state_dict(), checkpoint_path)
        else:
            state = {
                'time_step': time_step,
                'state_dict': self.experts[0].state_dict(),
                'best_acc': best_acc,
            }
            torch.save(state, checkpoint_path)


if __name__ == '__main__':
    state_dict = torch.load("PPO_preTrained/UAVnavigation/PPO_UAVnavigation_0_3_best.pth")
    print(state_dict)
