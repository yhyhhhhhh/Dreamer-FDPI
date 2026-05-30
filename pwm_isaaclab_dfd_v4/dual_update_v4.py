from __future__ import annotations

from contextlib import contextmanager

import torch

from .cost_utils import (
    cfg_get,
    dreamer_agent_distribution,
    posterior_states,
    temporarily_disable_grads,
)


@contextmanager
def _eval_mode(*modules):
    old_modes = []
    for module in modules:
        old_modes.append(getattr(module, "training", False))
        module.eval()
    try:
        yield
    finally:
        for module, was_training in zip(modules, old_modes):
            module.train(was_training)


def _normalize(value: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    flat = value.detach().float()
    return (value - flat.mean()) / flat.std(unbiased=False).clamp_min(eps)


def _predict_imagined_cost(world_model, feat, cost_cfg):
    if hasattr(world_model, "predict_cost"):
        pred_cost, _, _ = world_model.predict_cost(feat)
        return pred_cost.clamp(
            float(cfg_get(cost_cfg, "CostMin", 0.0)),
            float(cfg_get(cost_cfg, "CostMax", 1.0)),
        )
    return torch.zeros(*feat.shape[:-1], 1, dtype=feat.dtype, device=feat.device)


def _flatten_state(state):
    return {key: value.flatten(0, 1).detach() for key, value in state.items()}


def _risk_return_dual_update(
    batch,
    world_model,
    main_agent,
    gd_critic,
    dual_policy,
    cfg,
    cost_cfg,
):
    horizon = int(cfg_get(cfg, "Horizon", 5))
    gamma_cost = float(cfg_get(cfg, "GammaCost", getattr(gd_critic, "gamma_cost", 0.97)))
    kl_coeff = float(cfg_get(cfg, "KLCoeff", 1.0))
    entropy_coef = float(cfg_get(cfg, "EntropyCoef", 1.0e-4))

    start_state = posterior_states(
        world_model,
        batch["obs"].to(world_model.device),
        batch["action"].to(world_model.device),
        batch["is_first"].to(world_model.device),
    )
    state = _flatten_state(start_state)
    log_probs = []
    entropies = []
    kls = []
    costs = []

    with _eval_mode(world_model, main_agent):
        for _ in range(horizon):
            with torch.no_grad():
                feat = world_model.dynamic.get_feat(state).detach()
            dual_dist = dual_policy.distribution(feat)
            action = dual_dist.sample()
            log_prob = dual_dist.log_prob(action)[..., None]
            entropy = dual_dist.entropy()[..., None]
            with temporarily_disable_grads(main_agent):
                main_dist = dreamer_agent_distribution(main_agent, feat)
                main_log_on_dual = main_dist.log_prob(action)[..., None]
            kl = log_prob - main_log_on_dual.detach()
            with torch.no_grad():
                next_state = world_model.dynamic.img_step(state, action.detach())
                next_feat = world_model.dynamic.get_feat(next_state)
                pred_cost = _predict_imagined_cost(world_model, next_feat, cost_cfg)
            log_probs.append(log_prob)
            entropies.append(entropy)
            kls.append(kl)
            costs.append(pred_cost.detach())
            state = {key: value.detach() for key, value in next_state.items()}

    with torch.no_grad():
        terminal_feat = world_model.dynamic.get_feat(state).detach()
        terminal_action = dual_policy.sample(terminal_feat)
        terminal_risk = gd_critic.risk(terminal_feat, terminal_action).detach()
        ret = terminal_risk
        returns = []
        for cost in reversed(costs):
            ret = cost + gamma_cost * ret
            returns.append(ret)
        returns.reverse()
        risk_return = torch.stack(returns, dim=1)

    log_prob = torch.stack(log_probs, dim=1)
    entropy = torch.stack(entropies, dim=1)
    kl_to_main = torch.stack(kls, dim=1)
    risk_adv = _normalize(risk_return)
    dual_pg_loss = -(log_prob * risk_adv.detach()).mean()
    kl_loss = kl_coeff * kl_to_main.mean()
    entropy_bonus = entropy_coef * entropy.mean()
    loss = dual_pg_loss + kl_loss - entropy_bonus
    info_tensors = {
        "loss": loss.detach(),
        "policy_loss": dual_pg_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "entropy_bonus": entropy_bonus.detach(),
        "risk_return_mean": risk_return.detach().float().mean(),
        "pred_cost_mean": torch.stack(costs, dim=1).detach().float().mean(),
        "terminal_gd_mean": terminal_risk.detach().float().mean(),
        "kl_to_main": kl_to_main.detach().float().mean(),
        "entropy": entropy.detach().float().mean(),
    }
    return loss, info_tensors


def _max_risk_dual_update(batch, world_model, main_agent, gd_critic, dual_policy, cfg):
    horizon = int(cfg_get(cfg, "Horizon", 5))
    kl_coeff = float(cfg_get(cfg, "KLCoeff", 1.0))
    entropy_coef = float(cfg_get(cfg, "EntropyCoef", 1.0e-4))
    start_state = posterior_states(
        world_model,
        batch["obs"].to(world_model.device),
        batch["action"].to(world_model.device),
        batch["is_first"].to(world_model.device),
    )
    state = _flatten_state(start_state)
    feats = []
    with torch.no_grad():
        with _eval_mode(world_model, dual_policy):
            for _ in range(horizon):
                feat = world_model.dynamic.get_feat(state)
                action = dual_policy.sample(feat)
                state = world_model.dynamic.img_step(state, action)
                feats.append(world_model.dynamic.get_feat(state))
    z_train = torch.cat(feats, dim=0).detach() if feats else world_model.dynamic.get_feat(state).detach()
    dual_dist = dual_policy.distribution(z_train)
    dual_action = dual_dist.rsample()
    dual_log_prob = dual_dist.log_prob(dual_action)[..., None]
    entropy = dual_dist.entropy()[..., None]
    with temporarily_disable_grads(main_agent):
        main_dist = dreamer_agent_distribution(main_agent, z_train)
        main_log_on_dual = main_dist.log_prob(dual_action)[..., None]
    kl_to_main = dual_log_prob - main_log_on_dual
    with temporarily_disable_grads(gd_critic):
        score = gd_critic.risk(z_train, dual_action)
    risk_loss = -score.mean()
    kl_loss = kl_coeff * kl_to_main.mean()
    entropy_bonus = entropy_coef * entropy.mean()
    loss = risk_loss + kl_loss - entropy_bonus
    info_tensors = {
        "loss": loss.detach(),
        "risk_loss": risk_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "entropy_bonus": entropy_bonus.detach(),
        "risk_return_mean": score.detach().float().mean(),
        "pred_cost_mean": score.new_tensor(0.0),
        "terminal_gd_mean": score.detach().float().mean(),
        "kl_to_main": kl_to_main.detach().float().mean(),
        "entropy": entropy.detach().float().mean(),
    }
    return loss, info_tensors


def update_dual_v4(
    batch,
    world_model,
    main_agent,
    gd_critic,
    dual_policy,
    cfg,
    *,
    cost_cfg=None,
    logger=None,
    step=None,
):
    objective = str(cfg_get(cfg, "Type", "imagined_risk_return")).lower()
    grad_clip = float(cfg_get(cfg, "GradClipNorm", getattr(dual_policy, "max_grad_norm", 100.0)))

    world_model.eval()
    main_agent.eval()
    gd_critic.eval()
    dual_policy.train()

    with torch.autocast(device_type=dual_policy.device_type, dtype=dual_policy.tensor_dtype, enabled=dual_policy.use_amp):
        if objective == "imagined_risk_return":
            loss, info_tensors = _risk_return_dual_update(
                batch,
                world_model,
                main_agent,
                gd_critic,
                dual_policy,
                cfg,
                cost_cfg,
            )
        elif objective == "max_risk":
            loss, info_tensors = _max_risk_dual_update(batch, world_model, main_agent, gd_critic, dual_policy, cfg)
        else:
            raise ValueError(f"Unsupported DFD v4 DualUpdate.Type={objective!r}.")

    dual_policy.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(dual_policy.actor.parameters(), grad_clip)
    with torch.no_grad():
        dual_policy.optimizer.step()

    info = {key: float(value.detach().float().item()) for key, value in info_tensors.items()}
    info["grad_norm"] = float(torch.as_tensor(grad_norm).detach().float().item())
    info["horizon"] = float(cfg_get(cfg, "Horizon", 5))
    if logger is not None:
        for key, value in info.items():
            logger.log(f"Dual/{key}", value, step)
            logger.log(f"DualImag/{key}", value, step)
    return info
