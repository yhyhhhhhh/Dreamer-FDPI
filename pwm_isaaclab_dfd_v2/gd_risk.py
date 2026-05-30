from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class GdRiskCritic(nn.Module):
    """Double continuous-risk critic for dual-policy continuation risk."""

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
        dual_source_weight=2.0,
        high_cost_weight=3.0,
        high_cost_threshold=0.1,
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
        self.high_cost_threshold = float(high_cost_threshold)
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = bool(use_amp)

        self.gd1 = LatentRiskCritic(feat_dim, action_dim, hidden_dim, num_layers, act)
        self.gd2 = LatentRiskCritic(feat_dim, action_dim, hidden_dim, num_layers, act)
        self.target_gd1 = copy.deepcopy(self.gd1)
        self.target_gd2 = copy.deepcopy(self.gd2)
        self.to(device)
        for module in (self.target_gd1, self.target_gd2):
            for param in module.parameters():
                param.requires_grad_(False)

        disable_optimizer_dynamo_wrappers()
        self.optimizer = torch.optim.AdamW(
            list(self.gd1.parameters()) + list(self.gd2.parameters()),
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
            dual_source_weight=float(cfg_get(cfg, "DualSourceWeight", 2.0)),
            high_cost_weight=float(cfg_get(cfg, "HighCostWeight", 3.0)),
            high_cost_threshold=float(cfg_get(cfg, "HighCostThreshold", 0.1)),
        ).to(device)

    @torch.no_grad()
    def soft_update_targets(self):
        for source, target in ((self.gd1, self.target_gd1), (self.gd2, self.target_gd2)):
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - self.target_tau).add_(source_param.data, alpha=self.target_tau)
            for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
                target_buffer.copy_(source_buffer)

    def risk(self, feat, action, clamp=True):
        value = torch.minimum(self.gd1(feat, action), self.gd2(feat, action))
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
        extreme_cost = batch.get("extreme_cost")
        if extreme_cost is not None:
            extreme_cost = extreme_cost.to(world_model.device)
        is_first = batch["is_first"].to(world_model.device)

        feat = posterior_features(world_model, obs, action, is_first)
        if feat.shape[1] < 2:
            raise ValueError("Gd update needs at least two posterior latent steps.")

        z = feat[:, :-1]
        z_next = feat[:, 1:]
        replay_action = action[:, 1 : 1 + z.shape[1]]
        replay_cost = cost[:, 1 : 1 + z.shape[1]]
        replay_done = done[:, 1 : 1 + z.shape[1]]
        replay_source = source[:, 1 : 1 + z.shape[1]]
        replay_extreme = None
        if extreme_cost is not None:
            replay_extreme = extreme_cost[:, 1 : 1 + z.shape[1]]
        else:
            replay_extreme = torch.zeros_like(replay_cost)
        return (
            z.flatten(0, 1).detach(),
            replay_action.flatten(0, 1).detach(),
            z_next.flatten(0, 1).detach(),
            replay_cost.flatten(0, 1).detach().clamp(0.0, self.risk_max),
            replay_done.flatten(0, 1).detach().clamp(0.0, 1.0),
            replay_source.flatten(0, 1).detach(),
            replay_extreme.flatten(0, 1).detach().clamp(0.0, 1.0),
        )

    def _weights(self, cost, source):
        weight = torch.ones_like(cost)
        if self.source_aware_weight:
            weight = torch.where(source == SOURCE_DUAL, weight * self.dual_source_weight, weight)
            weight = torch.where(cost > self.high_cost_threshold, weight * self.high_cost_weight, weight)
        return weight

    def update(self, batch, world_model, dual_policy, *, logger=None, step=None):
        self.train()
        world_model.eval()
        if hasattr(dual_policy, "eval"):
            dual_policy.eval()
        z, action, z_next, cost, done, source, extreme_cost = self._aligned_latent_transition_batch(batch, world_model)
        source = source.to(torch.int64)
        extreme_mask = extreme_cost > 0.5
        weight = self._weights(cost, source)

        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            with torch.no_grad():
                next_action_dual = dual_policy.sample(z_next)
                target_risk = torch.minimum(
                    self.target_gd1(z_next, next_action_dual),
                    self.target_gd2(z_next, next_action_dual),
                )
                y_gd = cost + (1.0 - done) * self.gamma_cost * target_risk
                y_gd = y_gd.clamp(0.0, self.risk_max)

            pred_gd1 = self.gd1(z, action)
            pred_gd2 = self.gd2(z, action)
            loss1_per = weight * (pred_gd1 - y_gd).pow(2)
            loss2_per = weight * (pred_gd2 - y_gd).pow(2)
            gd_loss = loss1_per.mean() + loss2_per.mean()

        self.optimizer.zero_grad(set_to_none=True)
        gd_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(self.gd1.parameters()) + list(self.gd2.parameters()), self.max_grad_norm)
        with torch.no_grad():
            self.optimizer.step()
        self.soft_update_targets()

        with torch.no_grad():
            gd = torch.minimum(pred_gd1, pred_gd2).clamp(0.0, self.risk_max)
            high_mask = cost > self.high_cost_threshold
            low_mask = ~high_mask
            dual_mask = source == SOURCE_DUAL
            main_mask = source == SOURCE_MAIN
            source_dual_loss = (loss1_per[dual_mask] + loss2_per[dual_mask]).mean() if dual_mask.any() else gd.new_tensor(0.0)
            source_main_loss = (loss1_per[main_mask] + loss2_per[main_mask]).mean() if main_mask.any() else gd.new_tensor(0.0)
            high_mean = gd[high_mask].mean() if high_mask.any() else gd.new_tensor(0.0)
            low_mean = gd[low_mask].mean() if low_mask.any() else gd.new_tensor(0.0)
            extreme_mean = gd[extreme_mask].mean() if extreme_mask.any() else gd.new_tensor(0.0)
            non_extreme_mean = gd[~extreme_mask].mean() if (~extreme_mask).any() else gd.new_tensor(0.0)
            extreme_target_mean = y_gd[extreme_mask].mean() if extreme_mask.any() else gd.new_tensor(0.0)
        info = {
            "loss": float(gd_loss.detach().float().item()),
            "mean": float(gd.detach().float().mean().item()),
            "high_cost_mean": float(high_mean.detach().float().item()),
            "low_cost_mean": float(low_mean.detach().float().item()),
            "separation": float((high_mean - low_mean).detach().float().item()),
            "extreme_mean": float(extreme_mean.detach().float().item()),
            "non_extreme_mean": float(non_extreme_mean.detach().float().item()),
            "extreme_target_mean": float(extreme_target_mean.detach().float().item()),
            "extreme_ratio": float(extreme_mask.float().mean().detach().float().item()),
            "target_mean": float(y_gd.detach().float().mean().item()),
            "source_dual_loss": float(source_dual_loss.detach().float().item()),
            "source_main_loss": float(source_main_loss.detach().float().item()),
        }
        if logger is not None:
            for key, value in info.items():
                logger.log(f"Gd/{key}", value, step)
        return info
