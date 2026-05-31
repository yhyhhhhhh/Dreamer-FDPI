from __future__ import annotations

import torch
from torch.distributions import independent, normal

try:
    from pwm_isaaclab import scan
except ImportError:
    import scan

try:
    from pwm_isaaclab_dfd_v2.agent_dfd_v2 import CostAwareActorCriticAgent
except ImportError:
    from agent_dfd_v2 import CostAwareActorCriticAgent

from .cost_utils import cfg_get, temporarily_disable_grads


swap = lambda tensor: torch.transpose(tensor, 0, 1)
Normal = lambda mean, std: independent.Independent(normal.Normal(mean, std), 1)


def _weighted_masked_mean(value: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype, device=value.device)
    weight = weight.to(dtype=value.dtype, device=value.device)
    denom = (mask * weight).sum().clamp_min(1.0)
    return (value * mask * weight).sum() / denom


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(dtype=value.dtype, device=value.device)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def fdpi_regime_loss_components(
    *,
    log_prob: torch.Tensor,
    entropy: torch.Tensor,
    norm_adv: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    pf: float,
    cg: float,
    lambda_cri: float,
    lambda_inf: float,
    risk_max: float,
    entropy_coef: float,
    min_reward_weight_cri: float = 0.80,
    min_reward_weight_inf: float = 0.80,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    g_mask = g.detach()
    fea = g_mask < (float(pf) - float(cg))
    cri = (g_mask >= (float(pf) - float(cg))) & (g_mask < float(pf))
    inf = g_mask >= float(pf)

    alpha = ((g - (float(pf) - float(cg))) / (float(cg) + eps)).clamp(0.0, 1.0)
    risk_margin = torch.relu(g - (float(pf) - float(cg))) / (float(cg) + eps)
    risk_excess = torch.relu(g - float(pf)) / max(float(risk_max) - float(pf), eps)

    reward_adv = norm_adv.detach()
    reward_pg = -log_prob * reward_adv
    min_reward_weight_cri = float(min_reward_weight_cri)
    min_reward_weight_inf = float(min_reward_weight_inf)
    cri_reward_weight = torch.clamp(1.0 - alpha, min=min_reward_weight_cri, max=1.0)
    inf_reward_weight = g.new_full(g.shape, min_reward_weight_inf)
    loss_fea_per = reward_pg
    loss_cri_per = cri_reward_weight * reward_pg + alpha * float(lambda_cri) * risk_margin
    loss_inf_per = inf_reward_weight * reward_pg + float(lambda_inf) * risk_excess
    reward_per = torch.where(fea, reward_pg, torch.where(cri, cri_reward_weight * reward_pg, inf_reward_weight * reward_pg))
    risk_per = torch.where(
        fea,
        torch.zeros_like(reward_pg),
        torch.where(cri, alpha * float(lambda_cri) * risk_margin, float(lambda_inf) * risk_excess),
    )
    loss_per = torch.where(fea, loss_fea_per, torch.where(cri, loss_cri_per, loss_inf_per))

    loss_fea = _weighted_masked_mean(loss_fea_per, fea, weight)
    loss_cri = _weighted_masked_mean(loss_cri_per, cri, weight)
    loss_inf = _weighted_masked_mean(loss_inf_per, inf, weight)
    reward_loss_fea = _weighted_masked_mean(reward_pg, fea, weight)
    reward_loss_cri = _weighted_masked_mean(cri_reward_weight * reward_pg, cri, weight)
    reward_loss_inf = _weighted_masked_mean(inf_reward_weight * reward_pg, inf, weight)
    risk_loss_cri = _weighted_masked_mean(alpha * float(lambda_cri) * risk_margin, cri, weight)
    risk_loss_inf = _weighted_masked_mean(float(lambda_inf) * risk_excess, inf, weight)
    reward_loss_total = _weighted_mean(reward_per, weight)
    risk_loss_total = _weighted_mean(risk_per, weight)
    entropy_loss = _weighted_mean(entropy, weight)
    total = _weighted_mean(loss_per, weight) - float(entropy_coef) * entropy_loss

    metrics = {
        "loss_fea": loss_fea.detach(),
        "loss_cri": loss_cri.detach(),
        "loss_inf": loss_inf.detach(),
        "reward_loss_total": reward_loss_total.detach(),
        "reward_loss_fea": reward_loss_fea.detach(),
        "reward_loss_cri": reward_loss_cri.detach(),
        "reward_loss_inf": reward_loss_inf.detach(),
        "risk_loss_total": risk_loss_total.detach(),
        "risk_loss_cri": risk_loss_cri.detach(),
        "risk_loss_inf": risk_loss_inf.detach(),
        "cri_reward_weight": cri_reward_weight.detach()[cri].mean() if cri.any() else g.new_tensor(0.0),
        "inf_reward_weight": inf_reward_weight.detach()[inf].mean() if inf.any() else g.new_tensor(0.0),
        "entropy": entropy_loss.detach(),
        "fea_ratio": fea.float().mean().detach(),
        "cri_ratio": cri.float().mean().detach(),
        "inf_ratio": inf.float().mean().detach(),
        "gp_mean": g.detach().float().mean(),
        "reward_adv_mean": reward_adv.detach().float().mean(),
    }
    return total, metrics


class FDPIRegimeActorCriticAgent(CostAwareActorCriticAgent):
    """Dreamer actor-critic with an FDPI-Regime actor branch."""

    def update_fdpi_regime(
        self,
        feat,
        action,
        discount,
        reward,
        weight,
        gp_critic,
        cfg,
        logger=None,
        step=None,
    ):
        self.train()
        self.slow_critic.eval()
        pf = float(cfg_get(cfg, "Pf", 0.40))
        cg = float(cfg_get(cfg, "Cg", 0.10))
        lambda_cri = float(cfg_get(cfg, "LambdaCri", 0.001))
        lambda_inf = float(cfg_get(cfg, "LambdaInf", 0.002))
        risk_max = float(cfg_get(cfg, "RiskMax", getattr(gp_critic, "risk_max", 1.0)))
        entropy_coef = float(cfg_get(cfg, "EntropyCoef", self.entropy_coef))
        min_reward_weight_cri = float(cfg_get(cfg, "MinRewardWeightCri", 0.80))
        min_reward_weight_inf = float(cfg_get(cfg, "MinRewardWeightInf", 0.80))
        action_anchor_coef = float(cfg_get(cfg, "ActionAnchorCoef", 0.0))
        detach_action_logprob = bool(cfg_get(cfg, "DetachActionForLogProb", False))

        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            means, stds, raw_value = self.get_logits_raw_value(feat)
            stds = self.std_scale * torch.sigmoid(stds + 2) + self.std_offset
            dist = Normal(torch.tanh(means[:, :-1]), stds[:, :-1])
            main_action = dist.rsample()
            log_prob_action = main_action.detach() if detach_action_logprob else main_action
            log_prob = dist.log_prob(log_prob_action)[..., None]
            entropy = dist.entropy()[..., None]

            with torch.no_grad():
                value = self.twohot_loss.decode(raw_value)
                if self.use_slow_critic:
                    target_value = self.twohot_loss.decode(self.slow_critic(feat))
                else:
                    target_value = value
                swap_rew, swap_val, swap_disc = swap(reward), swap(target_value), swap(discount)
                lambda_return = swap(
                    scan.parallel_lambda_return(swap_rew, swap_val[:-1], swap_val[1:], swap_disc, self.lambd)
                )
                norm_adv = (lambda_return - value[:, :-1]) / self.get_scale(lambda_return)

            critic_loss = torch.mean(self.twohot_loss(raw_value[:, :-1], lambda_return, reduce=False) * weight)
            with temporarily_disable_grads(gp_critic):
                g = gp_critic.risk(feat[:, :-1], main_action, clamp=True)
            fdpi_actor_loss, metrics = fdpi_regime_loss_components(
                log_prob=log_prob,
                entropy=entropy,
                norm_adv=norm_adv,
                g=g,
                weight=weight,
                pf=pf,
                cg=cg,
                lambda_cri=lambda_cri,
                lambda_inf=lambda_inf,
                risk_max=risk_max,
                entropy_coef=entropy_coef,
                min_reward_weight_cri=min_reward_weight_cri,
                min_reward_weight_inf=min_reward_weight_inf,
            )
            action_anchor_loss = torch.zeros((), dtype=fdpi_actor_loss.dtype, device=fdpi_actor_loss.device)
            if action_anchor_coef > 0.0 and action is not None:
                anchor_len = min(int(main_action.shape[1]), int(action.shape[1]), int(weight.shape[1]))
                anchor_action = action[:, :anchor_len].detach().to(dtype=main_action.dtype, device=main_action.device)
                anchor_error = (main_action[:, :anchor_len] - anchor_action).square().mean(dim=-1, keepdim=True)
                action_anchor_loss = _weighted_mean(anchor_error, weight[:, :anchor_len])
                fdpi_actor_loss = fdpi_actor_loss + action_anchor_coef * action_anchor_loss
            total_loss = critic_loss + fdpi_actor_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.update_slow_critic()

        if logger is not None:
            logger.log("MainFDPI/enabled", 1.0, step)
            logger.log("ActorCritic/critic_loss", critic_loss.detach().float().item(), step)
            logger.log("ActorCritic/fdpi_actor_loss", fdpi_actor_loss.detach().float().item(), step)
            logger.log("ActorCritic/scale", self.get_scale(), step)
            logger.log("ActorCritic/lambda_return", lambda_return.mean().detach().float().item(), step)
            logger.log("ActorCritic/norm_adv", norm_adv.mean().detach().float().item(), step)
            logger.log("MainFDPI/action_anchor_loss", action_anchor_loss.detach().float().item(), step)
            logger.log("MainFDPI/action_anchor_coef", action_anchor_coef, step)
            logger.log("MainFDPI/detach_action_logprob", float(detach_action_logprob), step)
            for key, value in metrics.items():
                logger.log(f"MainFDPI/{key}", value.detach().float().item(), step)
        return {
            "critic_loss": float(critic_loss.detach().float().item()),
            "fdpi_actor_loss": float(fdpi_actor_loss.detach().float().item()),
            "action_anchor_loss": float(action_anchor_loss.detach().float().item()),
            **{key: float(value.detach().float().item()) for key, value in metrics.items()},
        }
