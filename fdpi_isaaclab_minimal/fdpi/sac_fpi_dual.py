from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .networks import QNet, SquashedGaussianActor
from .replay_buffer import ExperienceBatch


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    return (x * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def _as_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu())
    return float(value)


class TorchSACFPIDual(nn.Module):
    """PyTorch port of the JAX SACFPIDual update used by this repository."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        *,
        device: torch.device | str,
        gamma: float = 0.99,
        cost_gamma: float = 0.97,
        lr: float = 1e-4,
        max_grad_norm: float | None = 40.0,
        tau: float = 0.005,
        auto_alpha: bool = True,
        target_entropy: float | None = None,
        pf: float = 0.1,
        target_kl: float = 5.0,
        alpha: float = 1.0,
        cg: float = 0.01,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.cost_gamma = float(cost_gamma)
        self.max_grad_norm = max_grad_norm
        self.tau = float(tau)
        self.auto_alpha = bool(auto_alpha)
        self.target_entropy = -float(act_dim) if target_entropy is None else float(target_entropy)
        self.pf = float(pf)
        self.target_kl = float(target_kl)
        self.min_log_weight = math.log(min_weight)
        self.max_log_weight = math.log(max_weight)

        self.q1 = QNet(obs_dim, act_dim, hidden_sizes)
        self.q2 = QNet(obs_dim, act_dim, hidden_sizes)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)

        self.g1 = QNet(obs_dim, act_dim, hidden_sizes)
        self.g2 = QNet(obs_dim, act_dim, hidden_sizes)
        self.target_g1 = copy.deepcopy(self.g1)
        self.target_g2 = copy.deepcopy(self.g2)

        self.gr1 = QNet(obs_dim, act_dim, hidden_sizes)
        self.gr2 = QNet(obs_dim, act_dim, hidden_sizes)
        self.target_gr1 = copy.deepcopy(self.gr1)
        self.target_gr2 = copy.deepcopy(self.gr2)

        self.pi = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes)
        self.dual_pi = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes)

        self.dual_g1 = QNet(obs_dim, act_dim, hidden_sizes)
        self.dual_g2 = QNet(obs_dim, act_dim, hidden_sizes)
        self.dual_target_g1 = copy.deepcopy(self.dual_g1)
        self.dual_target_g2 = copy.deepcopy(self.dual_g2)

        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha), dtype=torch.float32))
        self.log_cg = nn.Parameter(torch.tensor(math.log(cg), dtype=torch.float32))
        self.lam1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.lam2 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.lam3 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.lam4 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.to(self.device)
        self._disable_target_grads()

        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.g1_optim = torch.optim.Adam(self.g1.parameters(), lr=lr)
        self.g2_optim = torch.optim.Adam(self.g2.parameters(), lr=lr)
        self.gr1_optim = torch.optim.Adam(self.gr1.parameters(), lr=lr)
        self.gr2_optim = torch.optim.Adam(self.gr2.parameters(), lr=lr)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.log_alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)
        self.log_cg_optim = torch.optim.Adam([self.log_cg], lr=lr)
        self.lam1_optim = torch.optim.Adam([self.lam1], lr=lr)
        self.lam2_optim = torch.optim.Adam([self.lam2], lr=lr)
        self.dual_g1_optim = torch.optim.Adam(self.dual_g1.parameters(), lr=lr)
        self.dual_g2_optim = torch.optim.Adam(self.dual_g2.parameters(), lr=lr)
        self.dual_pi_optim = torch.optim.Adam(self.dual_pi.parameters(), lr=lr)
        self.lam3_optim = torch.optim.Adam([self.lam3], lr=lr)
        self.lam4_optim = torch.optim.Adam([self.lam4], lr=lr)

    def _disable_target_grads(self) -> None:
        for module in (
            self.target_q1,
            self.target_q2,
            self.target_g1,
            self.target_g2,
            self.target_gr1,
            self.target_gr2,
            self.dual_target_g1,
            self.dual_target_g2,
        ):
            for param in module.parameters():
                param.requires_grad_(False)

    def _step_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        params: Iterable[torch.nn.Parameter],
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(params)
        if self.max_grad_norm is not None and len(params) > 0:
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, dual: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device, dtype=torch.float32)
        pi_dist = self.pi.distribution(obs)
        z = pi_dist.rsample()
        act = torch.tanh(z)
        if not dual:
            zeros = torch.zeros(obs.shape[0], dtype=torch.float32, device=self.device)
            return act, act, zeros, zeros

        dual_dist = self.dual_pi.distribution(obs)
        dual_z = dual_dist.rsample()
        dual_act = torch.tanh(dual_z)

        logp = self.pi.raw_log_prob(pi_dist, z)
        dual_logp = self.dual_pi.raw_log_prob(dual_dist, z)
        logp_dual = self.pi.raw_log_prob(pi_dist, dual_z)
        dual_logp_dual = self.dual_pi.raw_log_prob(dual_dist, dual_z)
        log_weight = torch.clamp(dual_logp - logp, self.min_log_weight, self.max_log_weight)
        log_weight_dual = torch.clamp(logp_dual - dual_logp_dual, self.min_log_weight, self.max_log_weight)
        return act, dual_act, log_weight, log_weight_dual

    @torch.no_grad()
    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.pi.deterministic_action(obs.to(self.device, dtype=torch.float32))

    def update(self, data: ExperienceBatch) -> dict[str, float]:
        obs = data.obs.to(self.device, dtype=torch.float32)
        action = data.action.to(self.device, dtype=torch.float32)
        next_obs = data.next_obs.to(self.device, dtype=torch.float32)
        reward = data.reward.to(self.device, dtype=torch.float32)
        cost = data.cost.to(self.device, dtype=torch.float32)
        done = data.done.to(self.device, dtype=torch.float32)
        weight = torch.exp(data.log_weight_dual.to(self.device, dtype=torch.float32))
        dual_weight = torch.exp(data.log_weight.to(self.device, dtype=torch.float32))

        with torch.no_grad():
            next_action, next_logp = self.pi.sample(next_obs)
            target_q = torch.minimum(
                self.target_q1(next_obs, next_action),
                self.target_q2(next_obs, next_action),
            )
            q_backup = reward + (1.0 - done) * self.gamma * (target_q - self.log_alpha.exp() * next_logp)

        q1 = self.q1(obs, action)
        q1_loss = (weight * (q1 - q_backup).pow(2)).mean()
        self._step_optimizer(self.q1_optim, q1_loss, self.q1.parameters())

        q2 = self.q2(obs, action)
        q2_loss = (weight * (q2 - q_backup).pow(2)).mean()
        self._step_optimizer(self.q2_optim, q2_loss, self.q2.parameters())

        with torch.no_grad():
            target_g = torch.maximum(
                self.target_g1(next_obs, next_action),
                self.target_g2(next_obs, next_action),
            ).clamp(0.0, 1.0)
            g_backup = cost + (1.0 - done) * (1.0 - cost) * self.cost_gamma * target_g

        g1 = self.g1(obs, action)
        g1_loss = (weight * (g1 - g_backup).pow(2)).mean()
        self._step_optimizer(self.g1_optim, g1_loss, self.g1.parameters())

        g2 = self.g2(obs, action)
        g2_loss = (weight * (g2 - g_backup).pow(2)).mean()
        self._step_optimizer(self.g2_optim, g2_loss, self.g2.parameters())

        with torch.no_grad():
            target_gr = torch.minimum(
                self.target_gr1(next_obs, next_action),
                self.target_gr2(next_obs, next_action),
            ).clamp(0.0, 1.0)
            gr_backup = (1.0 - cost) + (1.0 - done) * cost * self.cost_gamma * target_gr

        gr1 = self.gr1(obs, action)
        gr1_loss = (weight * (gr1 - gr_backup).pow(2)).mean()
        self._step_optimizer(self.gr1_optim, gr1_loss, self.gr1.parameters())

        gr2 = self.gr2(obs, action)
        gr2_loss = (weight * (gr2 - gr_backup).pow(2)).mean()
        self._step_optimizer(self.gr2_optim, gr2_loss, self.gr2.parameters())

        pi_action, logp = self.pi.sample(obs)
        q_pi = torch.minimum(self.q1(obs, pi_action), self.q2(obs, pi_action))
        g_pi = torch.maximum(self.g1(obs, pi_action), self.g2(obs, pi_action))
        gr_pi = torch.minimum(self.gr1(obs, pi_action), self.gr2(obs, pi_action))
        vio = cost > 0.0
        fea = (g_pi < self.pf) & ~vio
        cri = fea & (g_pi >= self.pf - self.log_cg.detach().exp())

        lam1 = self.lam1.detach()
        lam2 = self.lam2.detach()
        loss_fea = (fea & ~cri).to(q_pi.dtype) * -q_pi
        loss_cri = cri.to(q_pi.dtype) * (-q_pi + lam1 * g_pi) / (lam1 + 1.0)
        loss_inf = (~fea & ~vio).to(q_pi.dtype) * (-q_pi + lam2 * g_pi) / (lam2 + 1.0)
        loss_vio = vio.to(q_pi.dtype) * -gr_pi
        pi_loss = (weight * (loss_fea + loss_cri + loss_inf + loss_vio + self.log_alpha.detach().exp() * logp)).mean()
        self._step_optimizer(self.pi_optim, pi_loss, self.pi.parameters())

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self._step_optimizer(self.log_alpha_optim, alpha_loss, [self.log_alpha])
        else:
            alpha_loss = torch.zeros((), device=self.device)

        with torch.no_grad():
            new_dist = self.pi.distribution(obs)
            new_z = new_dist.sample()
            new_action = torch.tanh(new_z)
            new_g = torch.maximum(self.g1(obs, new_action), self.g2(obs, new_action))
            fea_det = fea.detach()
            cri_det = cri.detach()
            g_det = g_pi.detach()
            fea_ratio = fea_det.float().mean()
            cri_ratio = cri_det.float().mean()
            vio_new = fea_det & (new_g > self.pf)
            vio_ratio = vio_new.float().mean()
            delta_cg = _masked_mean(F.leaky_relu((self.pf - g_det) - self.log_cg.detach().exp()), vio_new)
            if bool((fea_ratio > 0.0).item()) and bool((vio_ratio == 0.0).item()):
                delta_cg = delta_cg + F.leaky_relu(-self.log_cg.detach().exp())
            fea_g_vio = _masked_mean(F.leaky_relu(new_g - self.pf), cri_det)
            inf_g_inc = _masked_mean(F.leaky_relu(new_g - g_det), ~fea_det)

        log_cg_loss = -(self.log_cg * delta_cg.detach())
        self._step_optimizer(self.log_cg_optim, log_cg_loss, [self.log_cg])

        lam1_loss = -(self.lam1 * fea_g_vio.detach())
        self._step_optimizer(self.lam1_optim, lam1_loss, [self.lam1])
        self.lam1.data.clamp_(min=0.0)

        lam2_loss = -(self.lam2 * inf_g_inc.detach())
        self._step_optimizer(self.lam2_optim, lam2_loss, [self.lam2])
        self.lam2.data.clamp_(min=0.0)

        with torch.no_grad():
            dual_next_action, _ = self.dual_pi.sample(next_obs)
            dual_target_g = torch.minimum(
                self.dual_target_g1(next_obs, dual_next_action),
                self.dual_target_g2(next_obs, dual_next_action),
            ).clamp(0.0, 1.0)
            dual_g_backup = cost + (1.0 - done) * (1.0 - cost) * self.cost_gamma * dual_target_g

        dual_g1 = self.dual_g1(obs, action)
        dual_g1_loss = (dual_weight * (dual_g1 - dual_g_backup).pow(2)).mean()
        self._step_optimizer(self.dual_g1_optim, dual_g1_loss, self.dual_g1.parameters())

        dual_g2 = self.dual_g2(obs, action)
        dual_g2_loss = (dual_weight * (dual_g2 - dual_g_backup).pow(2)).mean()
        self._step_optimizer(self.dual_g2_optim, dual_g2_loss, self.dual_g2.parameters())

        dual_dist = self.dual_pi.distribution(obs)
        dual_z = dual_dist.rsample()
        dual_action = torch.tanh(dual_z)
        dual_logp = self.dual_pi.squashed_log_prob(dual_dist, dual_z)
        dual_g = torch.minimum(self.dual_g1(obs, dual_action), self.dual_g2(obs, dual_action))
        with torch.no_grad():
            main_dist = self.pi.distribution(obs)
            new_z_fixed = main_dist.sample()
            main_log_prob_on_new_z = self.pi.raw_log_prob(main_dist, new_z_fixed)
        kl = (self.dual_pi.raw_log_prob(dual_dist, dual_z) - self.pi.raw_log_prob(main_dist, dual_z)).mean()
        dual_kl = (main_log_prob_on_new_z - self.dual_pi.raw_log_prob(dual_dist, new_z_fixed)).mean()
        dual_pi_loss = (
            (dual_weight * -dual_g).mean()
            + self.lam3.detach() * kl
            + self.lam4.detach() * dual_kl
        )
        self._step_optimizer(self.dual_pi_optim, dual_pi_loss, self.dual_pi.parameters())

        lam3_loss = self.lam3 * (self.target_kl - kl.detach())
        self._step_optimizer(self.lam3_optim, lam3_loss, [self.lam3])
        self.lam3.data.clamp_(min=0.0)

        lam4_loss = self.lam4 * (self.target_kl - dual_kl.detach())
        self._step_optimizer(self.lam4_optim, lam4_loss, [self.lam4])
        self.lam4.data.clamp_(min=0.0)

        self._soft_update_all()

        info = {
            "q1_loss": _as_float(q1_loss),
            "q2_loss": _as_float(q2_loss),
            "q1": _as_float(q1),
            "q2": _as_float(q2),
            "g1_loss": _as_float(g1_loss),
            "g2_loss": _as_float(g2_loss),
            "g1": _as_float(g1),
            "g2": _as_float(g2),
            "gr1_loss": _as_float(gr1_loss),
            "gr2_loss": _as_float(gr2_loss),
            "gr1": _as_float(gr1),
            "gr2": _as_float(gr2),
            "pi_loss": _as_float(pi_loss),
            "entropy": _as_float(-logp),
            "alpha": _as_float(self.log_alpha.exp()),
            "feasible_ratio": _as_float(fea_ratio),
            "critical_ratio": _as_float(cri_ratio),
            "feasible_g_violation_ratio": _as_float(vio_ratio),
            "feasible_g_violation": _as_float(fea_g_vio),
            "infeasible_g_increment": _as_float(inf_g_inc),
            "cg": _as_float(self.log_cg.exp()),
            "lam1": _as_float(self.lam1),
            "lam2": _as_float(self.lam2),
            "dual_g1_loss": _as_float(dual_g1_loss),
            "dual_g2_loss": _as_float(dual_g2_loss),
            "dual_g1": _as_float(dual_g1),
            "dual_g2": _as_float(dual_g2),
            "dual_pi_loss": _as_float(dual_pi_loss),
            "dual_entropy": _as_float(-dual_logp),
            "kl": _as_float(kl),
            "dual_kl": _as_float(dual_kl),
            "lam3": _as_float(self.lam3),
            "lam4": _as_float(self.lam4),
            "violate_ratio": _as_float(cost),
            "IS_weight": _as_float(weight),
            "IS_weight_dual": _as_float(dual_weight),
        }
        return info

    @torch.no_grad()
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(source_param.data, alpha=self.tau)

    @torch.no_grad()
    def _soft_update_all(self) -> None:
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        self._soft_update(self.g1, self.target_g1)
        self._soft_update(self.g2, self.target_g2)
        self._soft_update(self.gr1, self.target_gr1)
        self._soft_update(self.gr2, self.target_gr2)
        self._soft_update(self.dual_g1, self.dual_target_g1)
        self._soft_update(self.dual_g2, self.dual_target_g2)

    def checkpoint(self) -> dict:
        return {
            "model": self.state_dict(),
            "optimizers": {
                "q1": self.q1_optim.state_dict(),
                "q2": self.q2_optim.state_dict(),
                "g1": self.g1_optim.state_dict(),
                "g2": self.g2_optim.state_dict(),
                "gr1": self.gr1_optim.state_dict(),
                "gr2": self.gr2_optim.state_dict(),
                "pi": self.pi_optim.state_dict(),
                "log_alpha": self.log_alpha_optim.state_dict(),
                "log_cg": self.log_cg_optim.state_dict(),
                "lam1": self.lam1_optim.state_dict(),
                "lam2": self.lam2_optim.state_dict(),
                "dual_g1": self.dual_g1_optim.state_dict(),
                "dual_g2": self.dual_g2_optim.state_dict(),
                "dual_pi": self.dual_pi_optim.state_dict(),
                "lam3": self.lam3_optim.state_dict(),
                "lam4": self.lam4_optim.state_dict(),
            },
            "config": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "gamma": self.gamma,
                "cost_gamma": self.cost_gamma,
                "tau": self.tau,
                "target_entropy": self.target_entropy,
                "pf": self.pf,
                "target_kl": self.target_kl,
            },
        }

    def load_checkpoint(self, checkpoint: dict, *, load_optimizers: bool = True) -> None:
        self.load_state_dict(checkpoint["model"])
        if load_optimizers and "optimizers" in checkpoint:
            opt = checkpoint["optimizers"]
            self.q1_optim.load_state_dict(opt["q1"])
            self.q2_optim.load_state_dict(opt["q2"])
            self.g1_optim.load_state_dict(opt["g1"])
            self.g2_optim.load_state_dict(opt["g2"])
            self.gr1_optim.load_state_dict(opt["gr1"])
            self.gr2_optim.load_state_dict(opt["gr2"])
            self.pi_optim.load_state_dict(opt["pi"])
            self.log_alpha_optim.load_state_dict(opt["log_alpha"])
            self.log_cg_optim.load_state_dict(opt["log_cg"])
            self.lam1_optim.load_state_dict(opt["lam1"])
            self.lam2_optim.load_state_dict(opt["lam2"])
            self.dual_g1_optim.load_state_dict(opt["dual_g1"])
            self.dual_g2_optim.load_state_dict(opt["dual_g2"])
            self.dual_pi_optim.load_state_dict(opt["dual_pi"])
            self.lam3_optim.load_state_dict(opt["lam3"])
            self.lam4_optim.load_state_dict(opt["lam4"])
