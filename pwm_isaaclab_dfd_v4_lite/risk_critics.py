from __future__ import annotations

import copy

import torch
import torch.nn as nn

try:
    from pwm_isaaclab.modules import networks as net
except ImportError:
    import modules.networks as net

from .cost_utils import (
    SOURCE_DUAL,
    SOURCE_MAIN,
    cfg_get,
    disable_optimizer_dynamo_wrappers,
    ensure_optimizer_step_no_grad,
    posterior_features,
    unwrap_optimizer_step,
)
from .sampling_utils import source_cost_weight


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


class _DoubleRiskCritic(nn.Module):
    prefix = "Risk"
    critic_prefix = "risk"
    target_reduce = staticmethod(torch.minimum)
    policy_reduce = staticmethod(torch.minimum)

    def __init__(
        self,
        feat_dim,
        action_dim,
        hidden_dim,
        num_layers,
        gamma_cost,
        target_tau,
        risk_max,
        lr,
        eps,
        use_amp,
        act,
        device,
        max_grad_norm=100.0,
        source_aware_weight=True,
        dual_source_weight=1.0,
        high_cost_weight=1.0,
        boundary_weight=1.0,
        high_cost_threshold=0.1,
        boundary_low=0.05,
        boundary_high=0.4,
    ):
        super().__init__()
        self.device = device
        self.gamma_cost = float(gamma_cost)
        self.target_tau = float(target_tau)
        self.risk_max = float(risk_max)
        self.max_grad_norm = max_grad_norm
        self.source_aware_weight = bool(source_aware_weight)
        self.dual_source_weight = float(dual_source_weight)
        self.high_cost_weight = float(high_cost_weight)
        self.boundary_weight = float(boundary_weight)
        self.high_cost_threshold = float(high_cost_threshold)
        self.boundary_low = float(boundary_low)
        self.boundary_high = float(boundary_high)
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = bool(use_amp)

        self.critic1 = LatentRiskCritic(feat_dim, action_dim, hidden_dim, num_layers, act)
        self.critic2 = LatentRiskCritic(feat_dim, action_dim, hidden_dim, num_layers, act)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.to(device)
        for module in (self.target_critic1, self.target_critic2):
            for param in module.parameters():
                param.requires_grad_(False)

        disable_optimizer_dynamo_wrappers()
        self.optimizer = torch.optim.AdamW(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr,
            eps=eps,
        )
        unwrap_optimizer_step(self.optimizer)
        ensure_optimizer_step_no_grad(self.optimizer)

    @classmethod
    def from_config(cls, feat_dim, action_dim, cfg, *, use_amp, act, device, default_lr=1.0e-4, default_eps=1.0e-8):
        return cls(
            feat_dim=feat_dim,
            action_dim=action_dim,
            hidden_dim=int(cfg_get(cfg, "HiddenDim", cfg_get(cfg, "hidden_dim", 256))),
            num_layers=int(cfg_get(cfg, "NumLayers", cfg_get(cfg, "num_layers", 2))),
            gamma_cost=float(cfg_get(cfg, "GammaCost", 0.97)),
            target_tau=float(cfg_get(cfg, "TargetTau", 0.005)),
            risk_max=float(cfg_get(cfg, "RiskMax", 1.0)),
            lr=float(cfg_get(cfg, "LR", default_lr)),
            eps=float(cfg_get(cfg, "Eps", default_eps)),
            use_amp=use_amp,
            act=act,
            device=device,
            max_grad_norm=float(cfg_get(cfg, "GradClipNorm", 100.0)),
            source_aware_weight=bool(cfg_get(cfg, "SourceAwareWeight", True)),
            dual_source_weight=float(cfg_get(cfg, "DualSourceWeight", 1.0)),
            high_cost_weight=float(cfg_get(cfg, "HighCostWeight", 1.0)),
            boundary_weight=float(cfg_get(cfg, "BoundaryWeight", 1.0)),
            high_cost_threshold=float(cfg_get(cfg, "HighCostThreshold", 0.1)),
            boundary_low=float(cfg_get(cfg, "BoundaryLow", 0.05)),
            boundary_high=float(cfg_get(cfg, "BoundaryHigh", 0.4)),
        ).to(device)

    @torch.no_grad()
    def soft_update_targets(self):
        for source, target in ((self.critic1, self.target_critic1), (self.critic2, self.target_critic2)):
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - self.target_tau).add_(source_param.data, alpha=self.target_tau)
            for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
                target_buffer.copy_(source_buffer)

    def risk(self, feat, action, clamp=True):
        value = self.policy_reduce(self.critic1(feat, action), self.critic2(feat, action))
        if clamp:
            value = value.clamp(0.0, self.risk_max)
        return value

    @torch.no_grad()
    def risk_no_grad(self, feat, action, clamp=True):
        return self.risk(feat, action, clamp=clamp)

    def _aligned_latent_transition_batch(self, batch, world_model):
        obs = batch["obs"].to(world_model.device)
        action = batch["action"].to(world_model.device)
        cost = batch.get("continuous_cost", batch.get("cost")).to(world_model.device)
        done = batch["done"].to(world_model.device)
        source = batch["source"].to(world_model.device)
        is_first = batch["is_first"].to(world_model.device)

        feat = posterior_features(world_model, obs, action, is_first)
        if feat.shape[1] < 2:
            raise ValueError(f"{self.prefix} update needs at least two posterior latent steps.")

        z = feat[:, :-1]
        z_next = feat[:, 1:]
        replay_action = action[:, 1 : 1 + z.shape[1]]
        replay_cost = cost[:, 1 : 1 + z.shape[1]]
        replay_done = done[:, 1 : 1 + z.shape[1]]
        replay_source = source[:, 1 : 1 + z.shape[1]]
        return (
            z.flatten(0, 1).detach(),
            replay_action.flatten(0, 1).detach(),
            z_next.flatten(0, 1).detach(),
            replay_cost.flatten(0, 1).detach().clamp(0.0, self.risk_max),
            replay_done.flatten(0, 1).detach().clamp(0.0, 1.0),
            replay_source.flatten(0, 1).detach().to(torch.int64),
        )

    def _weights(self, cost, source):
        if not self.source_aware_weight:
            return torch.ones_like(cost)
        return source_cost_weight(
            cost,
            source,
            high_cost_weight=self.high_cost_weight,
            dual_source_weight=self.dual_source_weight,
            boundary_weight=self.boundary_weight,
            high_cost_threshold=self.high_cost_threshold,
            boundary_low=self.boundary_low,
            boundary_high=self.boundary_high,
        )

    def _update_impl(self, batch, world_model, next_policy, *, logger=None, step=None, main_policy=None, dual_policy=None):
        self.train()
        world_model.eval()
        if hasattr(next_policy, "eval"):
            next_policy.eval()
        z, action, z_next, cost, done, source = self._aligned_latent_transition_batch(batch, world_model)
        weight = self._weights(cost, source)

        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            with torch.no_grad():
                next_action = next_policy.sample(z_next)
                target_risk = self.target_reduce(
                    self.target_critic1(z_next, next_action),
                    self.target_critic2(z_next, next_action),
                )
                y = cost + (1.0 - done) * self.gamma_cost * target_risk
                y = y.clamp(0.0, self.risk_max)

            pred1 = self.critic1(z, action)
            pred2 = self.critic2(z, action)
            loss1_per = weight * (pred1 - y).pow(2)
            loss2_per = weight * (pred2 - y).pow(2)
            loss = loss1_per.mean() + loss2_per.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), self.max_grad_norm)
        with torch.no_grad():
            self.optimizer.step()
        self.soft_update_targets()

        with torch.no_grad():
            risk = self.policy_reduce(pred1, pred2).clamp(0.0, self.risk_max)
            high_mask = cost > self.high_cost_threshold
            low_mask = ~high_mask
            dual_mask = source == SOURCE_DUAL
            main_mask = source == SOURCE_MAIN
            high_mean = risk[high_mask].mean() if high_mask.any() else risk.new_tensor(0.0)
            low_mean = risk[low_mask].mean() if low_mask.any() else risk.new_tensor(0.0)
            source_dual_loss = (loss1_per[dual_mask] + loss2_per[dual_mask]).mean() if dual_mask.any() else risk.new_tensor(0.0)
            source_main_loss = (loss1_per[main_mask] + loss2_per[main_mask]).mean() if main_mask.any() else risk.new_tensor(0.0)
            main_action_mean = risk.new_tensor(0.0)
            dual_action_mean = risk.new_tensor(0.0)
            if main_policy is not None:
                main_action = main_policy.sample(z)
                main_action_mean = self.risk(z, main_action).mean()
            if dual_policy is not None:
                dual_action = dual_policy.sample(z)
                dual_action_mean = self.risk(z, dual_action).mean()
        info = {
            "loss": float(loss.detach().float().item()),
            "mean": float(risk.detach().float().mean().item()),
            "high_cost_mean": float(high_mean.detach().float().item()),
            "low_cost_mean": float(low_mean.detach().float().item()),
            "separation": float((high_mean - low_mean).detach().float().item()),
            "target_mean": float(y.detach().float().mean().item()),
            "source_dual_loss": float(source_dual_loss.detach().float().item()),
            "source_main_loss": float(source_main_loss.detach().float().item()),
            "main_action_mean": float(main_action_mean.detach().float().item()),
            "dual_action_mean": float(dual_action_mean.detach().float().item()),
        }
        if logger is not None:
            for key, value in info.items():
                logger.log(f"{self.prefix}/{key}", value, step)
        return info


class GpRiskCritic(_DoubleRiskCritic):
    """Double continuous-risk critic for main-policy continuation risk."""

    prefix = "Gp"
    critic_prefix = "gp"
    target_reduce = staticmethod(torch.maximum)
    policy_reduce = staticmethod(torch.maximum)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gp1 = self.critic1
        self.gp2 = self.critic2
        self.target_gp1 = self.target_critic1
        self.target_gp2 = self.target_critic2

    def update(self, batch, world_model, main_policy, dual_policy=None, *, logger=None, step=None):
        return self._update_impl(
            batch,
            world_model,
            main_policy,
            logger=logger,
            step=step,
            main_policy=main_policy,
            dual_policy=dual_policy,
        )


class GdRiskCriticV4(_DoubleRiskCritic):
    """Double continuous-risk critic for dual-policy continuation risk."""

    prefix = "Gd"
    critic_prefix = "gd"
    target_reduce = staticmethod(torch.minimum)
    policy_reduce = staticmethod(torch.minimum)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gd1 = self.critic1
        self.gd2 = self.critic2
        self.target_gd1 = self.target_critic1
        self.target_gd2 = self.target_critic2

    def update(self, batch, world_model, dual_policy, *, logger=None, step=None):
        return self._update_impl(
            batch,
            world_model,
            dual_policy,
            logger=logger,
            step=step,
            dual_policy=dual_policy,
        )
