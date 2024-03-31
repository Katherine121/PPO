import copy
import os
import random
import torch.nn.functional as F

from vit import *
from dataset import DDPGTD3dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


class DDPGCritic(nn.Module):
    def __init__(self):
        super(DDPGCritic, self).__init__()

        self.critic = ViT(image_size=IMG_SIZE,
                          patch_size=PATCH_SIZE,
                          dim=64,
                          depth=4,
                          heads=2,
                          dim_head=32,
                          mlp_dim=128,
                          dropout=0.,
                          emb_dropout=0.)
        self.critic.mlp_head = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(2 * self.critic.extractor_dim + 2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        feature = self.critic(state)
        sa = torch.cat((feature, action), dim=-1)
        return self.fc(sa)


class DDPG(object):
    def __init__(self, lr_actor, lr_critic, batch_size, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, tau=0.001):
        self.actor = Actor().to(device)
        self.actor_target = Actor().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = DDPGCritic().to(device)
        self.critic_target = DDPGCritic().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.tau = tau

    def select_action(self, state, path_list, episode_dir):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action = self.actor(state)

        record_file = os.path.join(episode_dir, "record.txt")
        with open(record_file, "a") as file1:
            file1.write(path_list[0] + " " + path_list[1]
                        + " " + str(action[0, 0].item()) + " " + str(action[0, 1].item())
                        + "\n")
        file1.close()
        return action.detach().cpu().numpy().flatten()

    def update(self, datasets_path):
        # Sample replay buffer
        buffer_states = []
        buffer_actions = []
        buffer_next_states = []
        buffer_rewards = []
        buffer_not_dones = []

        episode_dirs = os.listdir(datasets_path)
        random.shuffle(episode_dirs)
        for episode_dir in episode_dirs:
            full_episode_dir = os.path.join(datasets_path, episode_dir)
            full_record_file = os.path.join(full_episode_dir, "record.txt")
            full_reward_file = os.path.join(full_episode_dir, "reward.txt")
            full_newstate_file = os.path.join(full_episode_dir, "newstate.txt")

            f1 = open(full_record_file, 'rt')
            for line in f1:
                line = line.strip('\n')
                line = line.split(' ')
                buffer_states.append((line[0], line[1]))
                buffer_actions.append((float(line[2]), float(line[3])))
            f1.close()

            f2 = open(full_reward_file, 'rt')
            for line in f2:
                line = line.strip('\n')
                line = line.split(' ')
                buffer_rewards.append(float(line[0]))
                is_terminal = 0 if int(line[1]) == 1 else 1
                buffer_not_dones.append(is_terminal)
            f2.close()

            f3 = open(full_newstate_file, 'rt')
            for line in f3:
                line = line.strip('\n')
                line = line.split(' ')
                buffer_next_states.append((line[0], line[1]))
            f3.close()

        train_dataset = DDPGTD3dataset(buffer_states, buffer_actions, buffer_next_states, buffer_rewards, buffer_not_dones)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        # Optimize policy for K epochs
        for epoch_i in range(self.K_epochs):

            total_loss = 0

            # state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            for i, (state, action, next_state, reward, not_done) in enumerate(train_loader):
                state = state.to(device).to(dtype=torch.float32)
                action = action.to(device).to(dtype=torch.float32)
                next_state = next_state.to(device).to(dtype=torch.float32)
                reward = reward.to(device).to(dtype=torch.float32)
                not_done = not_done.to(device).to(dtype=torch.float32)

                # Compute the target Q value
                target_Q = self.critic_target(next_state, self.actor_target(next_state)).view(-1)
                target_Q = reward + (not_done * self.gamma * target_Q).detach()

                # Get current Q estimate
                current_Q = self.critic(state, action).view(-1)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q, target_Q)

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Compute actor loss
                actor_loss = -self.critic(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                total_loss += critic_loss.item() + actor_loss.item()

            if epoch_i % 5 == 0:
                print("Epoch" + str(epoch_i) + ", Loss: " + str(total_loss / (i + 1)))

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, time_step, best_acc, filename):
        torch.save(self.critic.state_dict(), filename.replace(".pth", "_critic.pth"))
        torch.save(self.critic_optimizer.state_dict(), filename.replace(".pth", "_critic_optimizer.pth"))

        torch.save(self.actor.state_dict(), filename.replace(".pth", "_actor.pth"))
        torch.save(self.actor_optimizer.state_dict(), filename.replace(".pth", "_actor_optimizer.pth"))

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename.replace(".pth", "_critic.pth")))
        self.critic_optimizer.load_state_dict(torch.load(filename.replace(".pth", "_critic_optimizer.pth")))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename.replace(".pth", "_actor.pth")))
        self.actor_optimizer.load_state_dict(torch.load(filename.replace(".pth", "_critic_optimizer.pth")))
        self.actor_target = copy.deepcopy(self.actor)
