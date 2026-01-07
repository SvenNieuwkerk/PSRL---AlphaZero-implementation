import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

class SeekerAlphaZeroNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.body = nn.Sequential(*layers)

        # Policy head: mean + log_std for Gaussian over actions
        self.mu_head = nn.Linear(last_dim, action_dim)
        self.log_std_head = nn.Linear(last_dim, action_dim)

        # Value head: scalar
        self.v_head = nn.Linear(last_dim, 1)

    def forward(self, obs: torch.Tensor):
        # obs: [batch, obs_dim]
        x = self.body(obs)
        mu = self.mu_head(x)                         # [batch, action_dim]
        log_std = self.log_std_head(x).clamp(-5, 2)  # keep std in a sane range
        v = self.v_head(x).squeeze(-1)               # [batch]
        return mu, log_std, v

    @staticmethod
    def policy_dist(mu: torch.Tensor, log_std: torch.Tensor):
        """
        Return a diagonal Gaussian policy distribution.
        Returns a distribution with event_dim=1 (action_dim).
        """
        std = log_std.exp()
        return Independent(Normal(mu, std), 1)