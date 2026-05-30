from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pwm_isaaclab.modules import networks as net
except ImportError:
    import modules.networks as net

from .utils import disable_optimizer_dynamo_wrappers, posterior_features, unwrap_optimizer_step


class LatentRiskCritic(nn.Module):
    def __init__(self, feat_dim, action_dim, hidden_dim, num_layers, act):
        super().__init__()
        layers = []
        last_dim = int(feat_dim) + int(action_dim)
        for _ in range(int(num_layers)):
            layers.append(net.FeedForwardLayer(last_dim, int(hidden_dim), act()))
            last_dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, 1)
        self.head.apply(net.uniform_weight_init(0.0))

    def forward(self, feat, action):
        x = torch.cat((feat, action), dim=-1)
        if len(self.backbone) > 0:
            x = self.backbone(x)
        return self.head(x)


class LatentFeasibilityModule(nn.Module):
    """Bootstrapped latent risk critics for main-policy and dual-policy risk."""

    def __init__(
        self,
        feat_dim,
        action_dim,
        hidden_dim,
        num_layers,
        cost_gamma,
        target_tau,
        lr,
        eps,
        use_amp,
        act,
        device,
        max_grad_norm=100.0,
    ):
        super().__init__()
        self.device = device
        self.cost_gamma = float(cost_gamma)
        self.target_tau = float(target_tau)
        self.max_grad_norm = max_grad_norm
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = bool(use_amp)

        self.gp1 = LatentRiskCritic(feat_dim, action_dim, hidden_dim, num_layers, act)
        self.gp2 = LatentRiskCritic(feat_dim, action_dim, hidden_dim, num_layers, act)
        self.gd1 = LatentRiskCritic(feat_dim, action_dim, hidden_dim, num_layers, act)
        self.gd2 = LatentRiskCritic(feat_dim, action_dim, hidden_dim, num_layers, act)
        self.target_gp1 = copy.deepcopy(self.gp1)
        self.target_gp2 = copy.deepcopy(self.gp2)
        self.target_gd1 = copy.deepcopy(self.gd1)
        self.target_gd2 = copy.deepcopy(self.gd2)
        self.to(device)
        self._disable_target_grads()

        disable_optimizer_dynamo_wrappers()
        self.gp_optimizer = torch.optim.AdamW(
            list(self.gp1.parameters()) + list(self.gp2.parameters()),
            lr=lr,
            eps=eps,
        )
        self.gd_optimizer = torch.optim.AdamW(
            list(self.gd1.parameters()) + list(self.gd2.parameters()),
            lr=lr,
            eps=eps,
        )
        unwrap_optimizer_step(self.gp_optimizer)
        unwrap_optimizer_step(self.gd_optimizer)

    def _disable_target_grads(self):
        for module in (self.target_gp1, self.target_gp2, self.target_gd1, self.target_gd2):
            for param in module.parameters():
                param.requires_grad_(False)

    @torch.no_grad()
    def soft_update_targets(self):
        pairs = (
            (self.gp1, self.target_gp1),
            (self.gp2, self.target_gp2),
            (self.gd1, self.target_gd1),
            (self.gd2, self.target_gd2),
        )
        for source, target in pairs:
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - self.target_tau).add_(source_param.data, alpha=self.target_tau)
            for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
                target_buffer.copy_(source_buffer)

    @torch.no_grad()
    def gp_risk(self, feat, action):
        return torch.maximum(self.gp1(feat, action), self.gp2(feat, action)).clamp(0.0, 1.0)

    @torch.no_grad()
    def gd_risk(self, feat, action):
        return torch.minimum(self.gd1(feat, action), self.gd2(feat, action)).clamp(0.0, 1.0)

    def _aligned_latent_transition_batch(self, batch, world_model):
        obs = batch["obs"].to(world_model.device)
        action = batch["action"].to(world_model.device)
        cost = batch["cost"].to(world_model.device)
        done = batch["done"].to(world_model.device)
        is_first = batch["is_first"].to(world_model.device)

        feat = posterior_features(world_model, obs, action, is_first)
        if feat.shape[1] < 2:
            raise ValueError("Feasibility update needs at least two posterior latent steps.")

        # parallel_observe posterior states start after the first replay action.
        z = feat[:, :-1]
        z_next = feat[:, 1:]
        replay_action = action[:, 1 : 1 + z.shape[1]]
        replay_cost = cost[:, 1 : 1 + z.shape[1]]
        replay_done = done[:, 1 : 1 + z.shape[1]]
        return (
            z.flatten(0, 1).detach(),
            replay_action.flatten(0, 1).detach(),
            z_next.flatten(0, 1).detach(),
            replay_cost.flatten(0, 1).detach().clamp(0.0, 1.0),
            replay_done.flatten(0, 1).detach().clamp(0.0, 1.0),
        )

    def update(self, batch, world_model, main_agent, dual_policy, *, logger=None, step=None):
        self.train()
        world_model.eval()
        main_agent.eval()
        dual_policy.eval()
        z, action, z_next, cost, done = self._aligned_latent_transition_batch(batch, world_model)

        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            with torch.no_grad():
                next_action_main = main_agent.sample(z_next)
                target_gp = torch.maximum(
                    self.target_gp1(z_next, next_action_main),
                    self.target_gp2(z_next, next_action_main),
                ).clamp(0.0, 1.0)
                y_gp = (
                    cost
                    + (1.0 - done)
                    * (1.0 - cost)
                    * self.cost_gamma
                    * target_gp
                ).clamp(0.0, 1.0)

            pred_gp1 = self.gp1(z, action)
            pred_gp2 = self.gp2(z, action)
            gp1_loss = F.mse_loss(pred_gp1, y_gp)
            gp2_loss = F.mse_loss(pred_gp2, y_gp)
            gp_loss = gp1_loss + gp2_loss

        self.gp_optimizer.zero_grad(set_to_none=True)
        gp_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.gp1.parameters()) + list(self.gp2.parameters()),
                self.max_grad_norm,
            )
        with torch.no_grad():
            self.gp_optimizer.step()

        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            with torch.no_grad():
                next_action_dual = dual_policy.sample(z_next)
                target_gd = torch.minimum(
                    self.target_gd1(z_next, next_action_dual),
                    self.target_gd2(z_next, next_action_dual),
                ).clamp(0.0, 1.0)
                y_gd = (
                    cost
                    + (1.0 - done)
                    * (1.0 - cost)
                    * self.cost_gamma
                    * target_gd
                ).clamp(0.0, 1.0)

            pred_gd1 = self.gd1(z, action)
            pred_gd2 = self.gd2(z, action)
            gd1_loss = F.mse_loss(pred_gd1, y_gd)
            gd2_loss = F.mse_loss(pred_gd2, y_gd)
            gd_loss = gd1_loss + gd2_loss

        self.gd_optimizer.zero_grad(set_to_none=True)
        gd_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.gd1.parameters()) + list(self.gd2.parameters()),
                self.max_grad_norm,
            )
        with torch.no_grad():
            self.gd_optimizer.step()
        self.soft_update_targets()

        with torch.no_grad():
            gp = torch.maximum(pred_gp1, pred_gp2).clamp(0.0, 1.0)
            gd = torch.minimum(pred_gd1, pred_gd2).clamp(0.0, 1.0)
        info = {
            "gp_loss": float(gp_loss.detach().float().item()),
            "gd_loss": float(gd_loss.detach().float().item()),
            "gp_mean": float(gp.detach().float().mean().item()),
            "gd_mean": float(gd.detach().float().mean().item()),
            "target_gp_mean": float(y_gp.detach().float().mean().item()),
            "target_gd_mean": float(y_gd.detach().float().mean().item()),
            "cost_rate": float(cost.detach().float().mean().item()),
        }
        if logger is not None:
            for key, value in info.items():
                logger.log(f"Feasibility/{key}", value, step)
        return info
