import torch
from torch import nn
from thop import profile
from moe import MoE
from cor import *


class Actor(nn.Module):
    def __init__(self, num_experts, noisy_gating, k, checkpoint_path=None):
        super().__init__()
        self.actor = MoE(num_experts=num_experts,
                         noisy_gating=noisy_gating,
                         k=k)
        # self.actor = MobileNet()
        # print(self.actor)
        # flops 51.21M, params 0.41M
        flops, params = profile(self.actor, (torch.randn(2, 3, IMG_SIZE, IMG_SIZE),))
        print('flops: ', flops, 'params: ', params)
        print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
        if checkpoint_path is not None:
            state_dict1 = torch.load(checkpoint_path)
            state_dict1 = state_dict1["state_dict"]
            state_dict2 = {}
            for name, param in state_dict1.items():
                state_dict2[name[len("module:"):]] = param
            self.actor.load_state_dict(state_dict2)

    def forward(self, img):
        return self.actor(img)


class Critic(nn.Module):
    def __init__(self, num_experts, noisy_gating, k):
        super().__init__()
        self.critic = MoE(num_experts=num_experts,
                          noisy_gating=noisy_gating,
                          k=k)
        # self.critic = MobileNet()
        for critic_i in self.critic.experts:
            critic_i.mlp_head = nn.Sequential(
                nn.Linear(2 * critic_i.extractor_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

    def forward(self, img):
        return self.critic(img)
