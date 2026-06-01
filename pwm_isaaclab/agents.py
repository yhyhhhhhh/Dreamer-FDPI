import copy
import torch
import torch.nn as nn
from torch.distributions import independent, normal

try:
    from pwm_isaaclab import scan, utils
    from pwm_isaaclab.modules import functions_losses as func
    from pwm_isaaclab.modules import networks as net
except ImportError:
    import scan
    import utils
    import modules.functions_losses as func
    import modules.networks as net


params = lambda module: list(module.parameters())
swap = lambda tensor: torch.transpose(tensor, 0, 1)
percentile = lambda tensor, per: torch.quantile(tensor, per)
Normal = lambda mean, std: independent.Independent(normal.Normal(mean, std), 1)


class ActorCriticAgent(nn.Module):
    def __init__(
        self,
        action_dim,
        feat,
        hidden,
        entropy_coef,
        num_bin,
        max_bin,
        min_per,
        max_per,
        min_std,
        max_std,
        ema_decay,
        gamma,
        lambd,
        tau,
        use_slow_critic,
        lr,
        eps,
        use_amp,
        act,
        device,
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.entropy_coef = entropy_coef
        self.min_per = min_per
        self.max_per = max_per
        self.std_offset = min_std
        self.std_scale = max_std - min_std
        self.gamma = gamma
        self.lambd = lambd
        self.tau = tau
        self.use_slow_critic = bool(use_slow_critic)

        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = use_amp

        self.actor = net.AgentLayer(feat, 2 * action_dim, hidden, act)
        self.critic = net.AgentLayer(feat, num_bin, hidden, act)
        self.slow_critic = copy.deepcopy(self.critic)
        for param in self.slow_critic.parameters():
            param.requires_grad_(False)

        self.lower_ema = utils.EMAScalar(ema_decay)
        self.upper_ema = utils.EMAScalar(ema_decay)

        agent_params = params(self.actor) + params(self.critic)
        self.twohot_loss = func.SymLogTwoHotLoss(num_bin, -max_bin, max_bin)
        self.optimizer = torch.optim.AdamW(agent_params, lr=lr, eps=eps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @torch.no_grad()
    def sync_slow_critic(self):
        self.slow_critic.load_state_dict(self.critic.state_dict())

    @torch.no_grad()
    def update_slow_critic(self):
        if not self.use_slow_critic:
            return
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
        for slow_buffer, buffer in zip(self.slow_critic.buffers(), self.critic.buffers()):
            slow_buffer.copy_(buffer)

    @torch.no_grad()
    def sample(self, feat, greedy=False):
        self.eval()
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            mean, std = self.actor(feat).chunk(2, dim=-1)
            std = self.std_scale * torch.sigmoid(std + 2) + self.std_offset
            dist = Normal(torch.tanh(mean), std)
            action = torch.tanh(mean) if greedy else dist.sample()
        return action

    def sample_as_env_action(self, feat, greedy=False):
        action = self.sample(feat, greedy)
        env_action = action.detach().cpu().numpy()
        return env_action, action

    @torch.no_grad()
    def get_scale(self, samples=None):
        if samples is None:
            lower_bound = self.lower_ema.get()
            upper_bound = self.upper_ema.get()
        else:
            lower_bound = self.lower_ema(percentile(samples, self.min_per))
            upper_bound = self.upper_ema(percentile(samples, self.max_per))
        return max(upper_bound - lower_bound, 1.0)

    def get_logits_raw_value(self, feat):
        mean, std = self.actor(feat).chunk(2, dim=-1)
        raw_value = self.critic(feat)
        return mean, std, raw_value

    def update(self, feat, action, discount, reward, weight, logger=None, step=None):
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

            critic_loss = torch.mean(
                self.twohot_loss(raw_value[:, :-1], lambda_return, reduce=False) * weight
            )
            policy_loss = torch.mean(log_prob * norm_adv * weight)
            entropy_loss = torch.mean(entropy * weight)
            total_loss = critic_loss - policy_loss - self.entropy_coef * entropy_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
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
            logger.log("ActorCritic/grad_norm", float(torch.as_tensor(grad_norm).detach().float().item()), step)
            logger.log("ActorCritic/use_slow_critic", float(self.use_slow_critic), step)
