from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import independent, normal

try:
    from pwm_isaaclab.modules import networks as net
except ImportError:
    import modules.networks as net

from .cost_utils import disable_optimizer_dynamo_wrappers, ensure_optimizer_step_no_grad, unwrap_optimizer_step


Normal = lambda mean, std: independent.Independent(normal.Normal(mean, std), 1)


class DualPolicyV4(nn.Module):
    """Dreamer-style Gaussian dual actor for FDPI-Regime high-risk data collection."""

    def __init__(
        self,
        action_dim,
        feat_dim,
        hidden,
        min_std,
        max_std,
        lr,
        eps,
        use_amp,
        act,
        device,
        max_grad_norm=100.0,
    ):
        super().__init__()
        self.device = device
        self.action_dim = int(action_dim)
        self.std_offset = float(min_std)
        self.std_scale = float(max_std) - float(min_std)
        self.max_grad_norm = max_grad_norm
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = bool(use_amp)

        self.actor = net.AgentLayer(feat_dim, 2 * action_dim, hidden, act)
        disable_optimizer_dynamo_wrappers()
        self.optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr, eps=eps)
        unwrap_optimizer_step(self.optimizer)
        ensure_optimizer_step_no_grad(self.optimizer)
        self.to(device)

    @torch.no_grad()
    def initialize_from_main_actor(self, main_agent):
        self.actor.load_state_dict(main_agent.actor.state_dict())

    def distribution(self, feat):
        mean, std = self.actor(feat).chunk(2, dim=-1)
        std = self.std_scale * torch.sigmoid(std + 2) + self.std_offset
        return Normal(torch.tanh(mean), std)

    def sample(self, feat, greedy=False, reparameterize=False, return_log_prob=False):
        dist = self.distribution(feat)
        if greedy:
            action = dist.base_dist.loc
        elif reparameterize:
            action = dist.rsample()
        else:
            action = dist.sample()
        if return_log_prob:
            return action, dist.log_prob(action)[..., None]
        return action

    def rsample(self, feat, return_log_prob=False):
        return self.sample(feat, reparameterize=True, return_log_prob=return_log_prob)

    def log_prob(self, feat, action):
        return self.distribution(feat).log_prob(action)[..., None]

    def entropy(self, feat):
        return self.distribution(feat).entropy()[..., None]
