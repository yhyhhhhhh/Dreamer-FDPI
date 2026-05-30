from __future__ import annotations

import torch

try:
    from pwm_isaaclab import scan
    from pwm_isaaclab.agents import ActorCriticAgent, Normal, swap
except ImportError:
    import scan
    from agents import ActorCriticAgent, Normal, swap

from .utils import disable_optimizer_dynamo_wrappers, unwrap_optimizer_step


class RiskConditionedActorCriticAgent(ActorCriticAgent):
    """ActorCriticAgent copy with an optional advantage modifier hook."""

    def __init__(self, *args, **kwargs):
        disable_optimizer_dynamo_wrappers()
        super().__init__(*args, **kwargs)
        unwrap_optimizer_step(self.optimizer)

    def update(
        self,
        feat,
        action,
        discount,
        reward,
        weight,
        logger=None,
        step=None,
        advantage_modifier_fn=None,
    ):
        self.train()
        self.slow_critic.eval()
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            means, stds, raw_value = self.get_logits_raw_value(feat)
            stds = self.std_scale * torch.sigmoid(stds + 2) + self.std_offset
            dist = Normal(torch.tanh(means[:, :-1]), stds[:, :-1])
            log_prob = dist.log_prob(action)[..., None]
            entropy = dist.entropy()[:, None]

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

            safe_adv = norm_adv
            if advantage_modifier_fn is not None:
                safe_adv = advantage_modifier_fn(
                    feat=feat[:, :-1],
                    action=action,
                    norm_adv=norm_adv,
                    weight=weight,
                    logger=logger,
                    step=step,
                )

            critic_loss = torch.mean(
                self.twohot_loss(raw_value[:, :-1], lambda_return, reduce=False) * weight
            )
            policy_loss = torch.mean(log_prob * safe_adv.detach() * weight)
            entropy_loss = torch.mean(entropy * weight)
            total_loss = critic_loss - policy_loss - self.entropy_coef * entropy_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        with torch.no_grad():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.update_slow_critic()

        if logger is not None:
            logger.log("ActorCritic/critic_loss", critic_loss.mean().item(), step)
            logger.log("ActorCritic/policy_loss", policy_loss.mean().item(), step)
            logger.log("ActorCritic/entropy", entropy_loss.mean().item(), step)
            logger.log("ActorCritic/scale", self.get_scale(), step)
            logger.log("ActorCritic/lambda_return", lambda_return.mean().item(), step)
            logger.log("ActorCritic/norm_adv", norm_adv.mean().item(), step)
            if advantage_modifier_fn is not None:
                logger.log("ActorCritic/safe_adv_delta", (safe_adv - norm_adv).mean().item(), step)
            logger.log("ActorCritic/use_slow_critic", float(self.use_slow_critic), step)
