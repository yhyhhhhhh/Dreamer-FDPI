from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import independent, normal

try:
    from pwm_isaaclab.modules import networks as net
except ImportError:
    import modules.networks as net

from .utils import (
    disable_optimizer_dynamo_wrappers,
    dreamer_agent_distribution,
    temporarily_disable_grads,
    unwrap_optimizer_step,
)


Normal = lambda mean, std: independent.Independent(normal.Normal(mean, std), 1)


class DualPolicy(nn.Module):
    """A Dreamer-style Gaussian dual actor with bidirectional KL multipliers."""

    def __init__(
        self,
        action_dim,
        feat_dim,
        hidden,
        min_std,
        max_std,
        lr,
        eps,
        lambda_kl_init,
        lambda_lr,
        use_amp,
        act,
        device,
        gd_objective="max",
        dual_g_scale=3.0,
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
        self.gd_objective = str(gd_objective).lower()
        self.dual_g_scale = float(dual_g_scale)
        if self.gd_objective not in ("min", "mean", "max"):
            raise ValueError(f"Unsupported dual gd_objective: {gd_objective}")

        self.actor = net.AgentLayer(feat_dim, 2 * action_dim, hidden, act)
        self.lambda_dual_main = nn.Parameter(torch.tensor(float(lambda_kl_init), dtype=torch.float32))
        self.lambda_main_dual = nn.Parameter(torch.tensor(float(lambda_kl_init), dtype=torch.float32))

        disable_optimizer_dynamo_wrappers()
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr, eps=eps)
        self.lambda_optimizer = torch.optim.AdamW(
            [self.lambda_dual_main, self.lambda_main_dual],
            lr=lambda_lr,
            eps=eps,
        )
        unwrap_optimizer_step(self.actor_optimizer)
        unwrap_optimizer_step(self.lambda_optimizer)
        self.to(device)

    @torch.no_grad()
    def initialize_from_main_actor(self, main_agent):
        self.actor.load_state_dict(main_agent.actor.state_dict())

    def distribution(self, feat):
        mean, std = self.actor(feat).chunk(2, dim=-1)
        std = self.std_scale * torch.sigmoid(std + 2) + self.std_offset
        return Normal(torch.tanh(mean), std)

    def _combine_gd(self, gd1, gd2):
        if self.gd_objective == "max":
            return torch.maximum(gd1, gd2)
        if self.gd_objective == "mean":
            return 0.5 * (gd1 + gd2)
        return torch.minimum(gd1, gd2)

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

    @torch.no_grad()
    def sample_as_env_action(self, feat, greedy=False):
        self.eval()
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            action = self.sample(feat, greedy=greedy, reparameterize=False)
        return action.detach().cpu().numpy(), action

    def log_prob(self, feat, action):
        return self.distribution(feat).log_prob(action)[..., None]

    def update(
        self,
        feat,
        main_agent,
        feasibility,
        *,
        target_kl,
        logger=None,
        step=None,
    ):
        self.train()
        feat = feat.detach().flatten(0, -2)
        if feat.numel() == 0:
            return {}

        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            dual_dist = self.distribution(feat)
            dual_action = dual_dist.rsample()
            dual_log_prob = dual_dist.log_prob(dual_action)[..., None]

            with torch.no_grad():
                main_dist = dreamer_agent_distribution(main_agent, feat)
                main_log_on_dual = main_dist.log_prob(dual_action)[..., None]
                main_action = main_dist.sample()
                main_log_prob = main_dist.log_prob(main_action)[..., None]

            dual_log_on_main = dual_dist.log_prob(main_action)[..., None]
            kl_dual_main = (dual_log_prob - main_log_on_dual).mean()
            kl_main_dual = (main_log_prob - dual_log_on_main).mean()

            with temporarily_disable_grads(feasibility):
                gd1 = feasibility.gd1(feat, dual_action)
                gd2 = feasibility.gd2(feat, dual_action)
                dual_g = self._combine_gd(gd1, gd2).clamp(0.0, 1.0)

            dual_loss = (
                -self.dual_g_scale * dual_g.mean()
                + self.lambda_dual_main.detach().clamp_min(0.0) * kl_dual_main
                + self.lambda_main_dual.detach().clamp_min(0.0) * kl_main_dual
            )

        self.actor_optimizer.zero_grad(set_to_none=True)
        dual_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        with torch.no_grad():
            self.actor_optimizer.step()

        lambda_loss = (
            self.lambda_dual_main * (float(target_kl) - kl_dual_main.detach())
            + self.lambda_main_dual * (float(target_kl) - kl_main_dual.detach())
        )
        self.lambda_optimizer.zero_grad(set_to_none=True)
        lambda_loss.backward()
        with torch.no_grad():
            self.lambda_optimizer.step()
        self.lambda_dual_main.data.clamp_(min=0.0)
        self.lambda_main_dual.data.clamp_(min=0.0)

        info = {
            "dual_loss": float(dual_loss.detach().float().item()),
            "dual_g": float(dual_g.detach().float().mean().item()),
            "dual_g_scale": self.dual_g_scale,
            "kl_dual_main": float(kl_dual_main.detach().float().item()),
            "kl_main_dual": float(kl_main_dual.detach().float().item()),
            "lambda_dual_main": float(self.lambda_dual_main.detach().float().item()),
            "lambda_main_dual": float(self.lambda_main_dual.detach().float().item()),
        }
        if logger is not None:
            for key, value in info.items():
                logger.log(f"Dual/{key}", value, step)
        return info
