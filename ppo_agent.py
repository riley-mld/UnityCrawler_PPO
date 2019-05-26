import numpy as np
import random
import copy
from collections import namedtuple, deque

from config import Configuration
from model import Actor_Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


# Set up an instance of the config class
config = Configuration()


class PPO_AGENT():
    """A class to create PPO Agents."""
    
    def __init__(self, state_size, action_size, num_agents, seed):
        """Initialize the Agent."""
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        
        self.policy = Actor_Critic(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.policy.parameters(), config.lr, weight_decay=lr.weight_decay)
        
        self.trajectory = []
        
    def act(self, state):
        """Act."""
        states = torch.from_numpy(states).float().to(device)
        
        self.policy.eval()
        
        with torch.no_grad():
            actions, log_prob, _, value = self.policy(state)
            
        self.policy.train()
        
        log_prob = log_prob.detach().cpu().numpy()
        value = value.detach().squeeze(1).cpu().numpy()
        actions = actions.detach().cpu().numpy()
        
        return actions, log_prob, _, value
    
    def save_step(self, trajectory):
        """Save the step to the trajectory."""
        self.trajectory.append(trajectory)
        
    def process_trajectory(self, states):
        """Process Trajectory."""
        returns = self.act(states)[-1]
        self.trajectory.append([states, returns, None, None, None, None])

        processed_trajectory = [None] * (len(self.trajectory) - 1)
        advantages = torch.Tensor(np.zeros((self.n_agent, 1))).to(config.device)
        #returns = p_value.detach()
        
        for i in reversed(range(len(self.trajectory) - 1)):
            states, actions, rewards, log_prob, value, terminals = self.trajectory[i]
            terminals = torch.Tensor(terminals).unsqueeze(1).to(config.device)
            rewards = torch.Tensor(rewards).unsqueeze(1).to(config.device)
            actions = torch.Tensor(actions).to(config.device)
            states = torch.Tensor(states).to(config.device)
            #log_probs =
            #value =
            next_value = self.trajectory[i + 1][1]
            returns = rewards + config.gamma * terminals * returns
            td_error = rewards + config.gamma * terminals * next_value.detach() - value.detach()
            advantages = advantages * config.gae_tau * config.gamma * terminals + td_error
            processed_trajectory[i] = [states, actions, log_probs, returns, advantages]

        # reset trajectory
        self.trajectory = []
        
        return processed_trajectory
        
    def step(self, states):
        """STEP."""
        
        processed_trajectory = self.process_trajectory(states)
        
        states, actions, old_log_probs, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_trajectory))

        # Normalize advantages estimate
        advantages = (advantages - advantages.mean())  / (advantages.std() + 1.0e-10)
                
        for _ in range(config.epochs):

            for states_batch, actions_batch, old_log_probs_batch, returns_batch, advantages_batch in \
                self.prepare_batch(states, actions, old_log_probs, returns, advantages):

                # Get updated values from policy
                _, new_log_probs_batch, entropy_batch, values_batch = self.policy(states_batch, actions_batch)

                # Calculate ratio for clipping
                ratio = (new_log_probs_batch - old_log_probs_batch).exp()

                # Clipped Surrogate function
                clip = torch.clamp(ratio, 1-config.epsilon, 1+config.epsilon)
                clipped_surrogate = torch.min(ratio*advantages_batch.unsqueeze(1), clip*advantages_batch.unsqueeze(1))
                
                # Calculate Actor Loss
                actor_loss = -torch.mean(clipped_surrogate) - config.beta * entropy_batch.mean()
                # Calculate Critic Loss
                critic_loss = F.smooth_l1_loss(values_batch, returns_batch.unsqueeze(1))
                
                # Reset the gradient
                self.optimizer.zero_grad()
                # Calculate Gradient
                (actor_loss + critic_loss).backward()
                # Clip the gradient
                nn.utils.clip_grad_norm_(self.policy.parameters(), config.gradient_clip)
                # Gradient Descent
                self.optimizer.step()

            
    def prepare_batch(self, states, actions, old_log_probs, returns, advantages):
        """Prepare the batches."""
        # nsteps * num_agents
        length = states.shape[0]
        batch_size = int(length / config.num_batches)
        idx = np.random.permutation(length)
        for i in range(config.num_batches):
            rge = idx[i*batch_size:(i+1)*batch_size]
            yield (states[rge], actions[rge], old_log_probs[rge], returns[rge], advs[rge].squeeze(1))
            
    def save(self):
        """Save the trained model."""
        torch.save(self.policy.state_dict(), 
                   str(config.fc1_units)+'_'+str(config.fc2_units) + '_model.pth')
        
    def load(self, file):
        """Load the trained model."""
        self.policy.load_state_dict(torch.load(file))
        