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
    loss_fea_per = -log_prob * reward_adv
    loss_cri_per = -(1.0 - alpha) * log_prob * reward_adv + alpha * float(lambda_cri) * risk_margin
    loss_inf_per = float(lambda_inf) * risk_excess

    loss_fea = _weighted_masked_mean(loss_fea_per, fea, weight)
    loss_cri = _weighted_masked_mean(loss_cri_per, cri, weight)
    loss_inf = _weighted_masked_mean(loss_inf_per, inf, weight)
    entropy_loss = (entropy * weight).mean()
    total = loss_fea + loss_cri + loss_inf - float(entropy_coef) * entropy_loss

    metrics = {
        "loss_fea": loss_fea.detach(),
        "loss_cri": loss_cri.detach(),
        "loss_inf": loss_inf.detach(),
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
        del action
        self.train()
        self.slow_critic.eval()
        pf = float(cfg_get(cfg, "Pf", 0.10))
        cg = float(cfg_get(cfg, "Cg", 0.03))
        lambda_cri = float(cfg_get(cfg, "LambdaCri", 0.02))
        lambda_inf = float(cfg_get(cfg, "LambdaInf", 0.05))
        risk_max = float(cfg_get(cfg, "RiskMax", getattr(gp_critic, "risk_max", 1.0)))
        entropy_coef = float(cfg_get(cfg, "EntropyCoef", self.entropy_coef))

        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            means, stds, raw_value = self.get_logits_raw_value(feat)
            stds = self.std_scale * torch.sigmoid(stds + 2) + self.std_offset
            dist = Normal(torch.tanh(means[:, :-1]), stds[:, :-1])
            main_action = dist.rsample()
            log_prob = dist.log_prob(main_action)[..., None]
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
            )
            total_loss = critic_loss + fdpi_actor_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.update_slow_critic()

        if logger is not None:
            logger.log("ActorCritic/critic_loss", critic_loss.detach().float().item(), step)
            logger.log("ActorCritic/fdpi_actor_loss", fdpi_actor_loss.detach().float().item(), step)
            logger.log("ActorCritic/scale", self.get_scale(), step)
            logger.log("ActorCritic/lambda_return", lambda_return.mean().detach().float().item(), step)
            logger.log("ActorCritic/norm_adv", norm_adv.mean().detach().float().item(), step)
            for key, value in metrics.items():
                logger.log(f"MainFDPI/{key}", value.detach().float().item(), step)
        return {
            "critic_loss": float(critic_loss.detach().float().item()),
            "fdpi_actor_loss": float(fdpi_actor_loss.detach().float().item()),
            **{key: float(value.detach().float().item()) for key, value in metrics.items()},
        }
