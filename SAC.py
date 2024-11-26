import random
from collections import deque
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
from SalePredictor import SalePredictor

class ReplayBuffer:
    """Replay buffer storing states and features_2_results for reward calculation."""

    def __init__(self, buffer_size=100000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, data):
        """Store state and features_2_result."""
        state = data[:-2]  # Ensure state is a numeric array
        issue_result = data[-2:]  # Ensure result is numeric
        # Convert to float if needed
        state = np.array(state, dtype=np.float32)
        issue_result = np.array(issue_result)
        issue_result[0] = issue_result[0].replace(" ", "")  # 去掉结果中的空格
        self.buffer.append([state, issue_result])

    def sample(self, batch_size):
        """Sample a batch of data."""
        batch = random.sample(self.buffer, batch_size)
        states, issue_results = zip(*batch)  # Unpack into two lists

        # Convert states to torch.Tensor
        states = torch.tensor(np.array(states, dtype=np.float32), dtype=torch.float32)
        return states, issue_results

    def __len__(self):
        return len(self.buffer)

class SharedBlock(nn.Module):
    """Shared block for each feature group."""

    def __init__(self, input_dim, hidden_dim=16, output_dim=4):
        super(SharedBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class FinalBlock(nn.Module):
    """Final block to combine shared outputs and sale_predictions."""

    def __init__(self, final_new_input_dim, output_dim, shared_output_dim=4, hidden_dim=128):
        super(FinalBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(shared_output_dim * 14 + final_new_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, shared_outputs, final_new_input):
        combined = torch.cat((shared_outputs, final_new_input), dim=1)
        return self.block(combined)


class Actor(nn.Module):
    """Actor network with shared blocks, outputting actions based on a probability distribution."""

    def __init__(self, num_groups=14, group_input_dim=9, shared_hidden_dim=16, shared_output_dim=4,
                 sale_pred_dim=1, final_hidden_dim=128, action_dim=15):
        super(Actor, self).__init__()

        # Shared block: Single instance reused across all groups in the Actor
        self.shared_block = SharedBlock(group_input_dim, shared_hidden_dim, shared_output_dim)

        # Final block for generating action mean and log_std
        self.final_block = FinalBlock(final_new_input_dim=sale_pred_dim,
                                      output_dim=action_dim * 2,  # Output both mean and log_std
                                      shared_output_dim=shared_output_dim,
                                      hidden_dim=final_hidden_dim)

    def forward(self, features_2, sale_predictions):
        group_outputs = []
        for i in range(14):  # Loop through each group
            group_input = features_2[:, i * 9:(i + 1) * 9]  # Extract group-specific features
            group_outputs.append(self.shared_block(group_input))  # Reuse the same shared_block

        # Concatenate outputs from all shared blocks
        shared_outputs = torch.cat(group_outputs, dim=1)

        # Pass through final block to get action mean and log_std
        output = self.final_block(shared_outputs, sale_predictions)

        # Split output into mean and log_std
        action_mean, action_log_std = torch.chunk(output, 2, dim=1)
        action_std = torch.exp(action_log_std)  # Ensure std is positive

        # Limit action_mean to [-1, 1]
        action_mean = torch.tanh(action_mean)

        return action_mean, action_std

    def sample_action(self, features_2, sale_predictions):
        """Sample an action from the probability distribution."""
        action_mean, action_std = self(features_2, sale_predictions)
        dist = Normal(action_mean, action_std)  # Create normal distribution
        action = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action).sum(dim=-1)  # Log probability of the sampled action
        return action, log_prob


class Critic(nn.Module):
    """Critic network with shared blocks."""

    def __init__(self, num_groups=14, group_input_dim=9, shared_hidden_dim=16, shared_output_dim=4,
                 sale_pred_dim=1, action_dim=15, final_hidden_dim=128):
        super(Critic, self).__init__()

        # Shared block: Single instance reused across all groups in the Critic
        self.shared_block = SharedBlock(group_input_dim, shared_hidden_dim, shared_output_dim)

        # Final block for combining shared outputs, sale_predictions, and actions
        self.final_block = FinalBlock(final_new_input_dim=sale_pred_dim + action_dim,
                                      output_dim=1,
                                      shared_output_dim=shared_output_dim,
                                      hidden_dim=final_hidden_dim)

    def forward(self, features_2, sale_predictions, actions):
        group_outputs = []
        for i in range(14):  # Loop through each group
            group_input = features_2[:, i * 9:(i + 1) * 9]  # Extract group-specific features
            group_outputs.append(self.shared_block(group_input))  # Reuse the same shared_block

        # Concatenate outputs from all shared blocks
        shared_outputs = torch.cat(group_outputs, dim=1)

        # Combine with actions and sale_predictions
        combined_input = torch.cat((sale_predictions, actions), dim=1)

        # Pass through final block
        return self.final_block(shared_outputs, combined_input)


class SACAgent:
    """Soft Actor-Critic Agent."""

    def __init__(self, state_dim, action_dim, num_groups=14, group_input_dim=9,
                 shared_hidden_dim=16, shared_output_dim=4, sale_pred_dim=1,
                 final_hidden_dim=128, lr=1e-4, alpha_lr=1e-4, target_entropy=None, initial_alpha=0.5):
        """
        :param state_dim: Total state dimensions (features_2 + sale_predictions)
        :param action_dim: Number of actions (e.g., 15 in your setup)
        :param num_groups: Number of feature groups in features_2 (e.g., 14)
        :param group_input_dim: Number of features per group (e.g., 9)
        :param shared_hidden_dim: Hidden dimensions in the shared block
        :param shared_output_dim: Output dimensions of each shared block
        :param sale_pred_dim: Dimensions of sale_predictions (e.g., 1)
        :param final_hidden_dim: Hidden dimensions in the final block
        :param lr: Learning rate for optimizers
        """

        # Actor network
        self.actor = Actor(num_groups=num_groups,
                           group_input_dim=group_input_dim,
                           shared_hidden_dim=shared_hidden_dim,
                           shared_output_dim=shared_output_dim,
                           sale_pred_dim=sale_pred_dim,
                           final_hidden_dim=final_hidden_dim,
                           action_dim=action_dim)

        # Critic networks
        self.critic1 = Critic(num_groups=num_groups,
                              group_input_dim=group_input_dim,
                              shared_hidden_dim=shared_hidden_dim,
                              shared_output_dim=shared_output_dim,
                              sale_pred_dim=sale_pred_dim,
                              action_dim=action_dim,
                              final_hidden_dim=final_hidden_dim)

        self.critic2 = Critic(num_groups=num_groups,
                              group_input_dim=group_input_dim,
                              shared_hidden_dim=shared_hidden_dim,
                              shared_output_dim=shared_output_dim,
                              sale_pred_dim=sale_pred_dim,
                              action_dim=action_dim,
                              final_hidden_dim=final_hidden_dim)

        # Target critics for stable updates
        self.target_critic1 = Critic(num_groups=num_groups,
                                     group_input_dim=group_input_dim,
                                     shared_hidden_dim=shared_hidden_dim,
                                     shared_output_dim=shared_output_dim,
                                     sale_pred_dim=sale_pred_dim,
                                     action_dim=action_dim,
                                     final_hidden_dim=final_hidden_dim)

        self.target_critic2 = Critic(num_groups=num_groups,
                                     group_input_dim=group_input_dim,
                                     shared_hidden_dim=shared_hidden_dim,
                                     shared_output_dim=shared_output_dim,
                                     sale_pred_dim=sale_pred_dim,
                                     action_dim=action_dim,
                                     final_hidden_dim=final_hidden_dim)

        # Copy parameters from critics to target critics
        self._update_target_networks(tau=0.01)



        # Target entropy: defaults to -action_dim
        self.target_entropy = target_entropy if target_entropy is not None else -0.5*action_dim

        # Log alpha for stability and optimizer
        self.log_alpha = torch.tensor(np.log(initial_alpha), requires_grad=True)


        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=5e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=5e-4)

    @property
    def alpha(self):
        # Exponentiate log_alpha to get alpha
        return self.log_alpha.exp()

    def _update_target_networks(self, tau=0.01):
        """
        Soft update target networks: target = tau * online + (1 - tau) * target
        """
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
