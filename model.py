import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor_Critic(nn.Module):

    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed):
        super(Actor_Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        self.seed = torch.manual_seed(seed)
        
        #self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        #self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        #self.bn2 = nn.BatchNorm1d(fc2_units)

        self.fc_actor_mean = nn.Linear(fc2_units, self.action_size)
        self.fc_actor_std = nn.Linear(fc2_units, self.action_size)
        self.fc_critic = nn.Linear(fc2_units, 1)

        self.std = nn.Parameter(torch.zeros(1, action_size))

    def forward(self, x, action=None):
        #x = self.bn0(x)
        x = F.relu(self.fc1(x))
        #x = self.bn1(x)
        x = F.relu(self.fc2(x))
        #x = self.bn2(x)

        # Actor
        mean = torch.tanh(self.fc_actor_mean(x))
        std = F.softplus(self.fc_actor_std(x))
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        # Critic
        v = self.fc_critic(x)

        return action, log_prob, dist.entropy(), v