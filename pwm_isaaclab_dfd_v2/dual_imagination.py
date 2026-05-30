from __future__ import annotations

import torch

from .cost_utils import cfg_get, dreamer_agent_distribution, posterior_states, temporarily_disable_grads


def _flatten_state_time(states):
    feats = []
    for state in states:
        feats.append(state)
    return torch.cat(feats, dim=0)


def _rollout_dual_states(world_model, dual_policy, start_state, horizon):
    state = {key: value.flatten(0, 1) for key, value in start_state.items()}
    imagined_feats = []
    with torch.no_grad():
        for _ in range(int(horizon)):
            feat = world_model.dynamic.get_feat(state)
            action = dual_policy.sample(feat)
            state = world_model.dynamic.img_step(state, action)
            imagined_feats.append(world_model.dynamic.get_feat(state))
    if not imagined_feats:
        return world_model.dynamic.get_feat(state).detach()
    return _flatten_state_time(imagined_feats).detach()


def update_dual_in_imagination(
    batch,
    world_model,
    main_agent,
    gd_critic,
    dual_policy,
    cfg,
    *,
    logger=None,
    step=None,
):
    objective = str(cfg_get(cfg, "Objective", "max_risk")).lower()
    if objective != "max_risk":
        raise ValueError(f"DFD v2 first implementation only supports Objective='max_risk', got {objective!r}.")

    horizon = int(cfg_get(cfg, "Horizon", 5))
    kl_coeff = float(cfg_get(cfg, "KLCoeff", 1.0))
    entropy_coef = float(cfg_get(cfg, "EntropyCoef", 1.0e-4))
    grad_clip = float(cfg_get(cfg, "GradClipNorm", getattr(dual_policy, "max_grad_norm", 100.0)))

    world_model.eval()
    main_agent.eval()
    gd_critic.eval()
    dual_policy.train()

    start_state = posterior_states(
        world_model,
        batch["obs"].to(world_model.device),
        batch["action"].to(world_model.device),
        batch["is_first"].to(world_model.device),
    )
    z_train = _rollout_dual_states(world_model, dual_policy, start_state, horizon)
    if z_train.numel() == 0:
        return {}

    with torch.autocast(device_type=dual_policy.device_type, dtype=dual_policy.tensor_dtype, enabled=dual_policy.use_amp):
        dual_dist = dual_policy.distribution(z_train)
        dual_action = dual_dist.rsample()
        dual_log_prob = dual_dist.log_prob(dual_action)[..., None]
        entropy = dual_dist.entropy()[..., None]

        with temporarily_disable_grads(main_agent):
            main_dist = dreamer_agent_distribution(main_agent, z_train)
            main_log_on_dual = main_dist.log_prob(dual_action)[..., None]

        kl_to_main = (dual_log_prob - main_log_on_dual).mean()
        with temporarily_disable_grads(gd_critic):
            score = torch.minimum(
                gd_critic.gd1(z_train, dual_action),
                gd_critic.gd2(z_train, dual_action),
            ).clamp(0.0, gd_critic.risk_max)
        dual_loss = -score.mean() + kl_coeff * kl_to_main - entropy_coef * entropy.mean()

    dual_policy.optimizer.zero_grad(set_to_none=True)
    dual_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(dual_policy.actor.parameters(), grad_clip)
    with torch.no_grad():
        dual_policy.optimizer.step()

    info = {
        "loss": float(dual_loss.detach().float().item()),
        "gd_score": float(score.detach().float().mean().item()),
        "kl_to_main": float(kl_to_main.detach().float().item()),
        "entropy": float(entropy.detach().float().mean().item()),
        "grad_norm": float(torch.as_tensor(grad_norm).detach().float().item()),
        "horizon": float(horizon),
        "imagined_gd_mean": float(score.detach().float().mean().item()),
        "imagined_gd_max": float(score.detach().float().max().item()),
    }
    if logger is not None:
        for key, value in info.items():
            logger.log(f"DualImag/{key}", value, step)
    return info
