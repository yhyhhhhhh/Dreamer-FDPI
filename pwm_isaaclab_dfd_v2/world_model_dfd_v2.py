from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod

from .cost_utils import ensure_optimizer_step_no_grad

try:
    from pwm_isaaclab.modules.world_models import ParallelWorldModel, predict_force_from_outputs
except ImportError:
    from modules.world_models import ParallelWorldModel, predict_force_from_outputs


class ContinuousCostHead(nn.Module):
    """Continuous normalized cost head with an auxiliary extreme-force logit."""

    def __init__(self, input_dim, hidden_dim, act, depth=3):
        super().__init__()
        layers = []
        in_dim = int(input_dim)
        for _ in range(max(int(depth), 1)):
            layers += [
                nn.Linear(in_dim, int(hidden_dim)),
                nn.LayerNorm(int(hidden_dim)),
                act(),
            ]
            in_dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.cost_logit = nn.Linear(in_dim, 1)
        self.extreme_logit = nn.Linear(in_dim, 1)

    def forward(self, feat):
        h = self.backbone(feat)
        return {
            "cost_logit": self.cost_logit(h),
            "extreme_logit": self.extreme_logit(h),
        }


def _weighted_mean(value, weight):
    weight = torch.as_tensor(weight, dtype=value.dtype, device=value.device).reshape_as(value)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _safe_mean(value, mask):
    if mask is None or not mask.any():
        return value.new_tensor(0.0)
    return value[mask].mean()


def _average_precision(labels, scores):
    labels = labels.detach().float().reshape(-1)
    scores = scores.detach().float().reshape(-1)
    finite = torch.isfinite(labels) & torch.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positives = labels > 0.5
    num_pos = int(positives.sum().item())
    if labels.numel() == 0 or num_pos == 0:
        return None
    order = torch.argsort(scores, descending=True)
    sorted_labels = positives[order].float()
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1.0 - sorted_labels, dim=0)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / float(num_pos)
    prev_recall = torch.cat([torch.zeros(1, device=recall.device), recall[:-1]])
    return float(((recall - prev_recall) * precision).sum().detach().cpu().item())


def _binary_prediction_metrics(prob, label, prefix):
    label = torch.as_tensor(label, dtype=prob.dtype, device=prob.device).reshape_as(prob)
    pred_label = (prob >= 0.5).to(prob.dtype)
    positive = label > 0.5
    negative = ~positive
    tp = ((pred_label > 0.5) & positive).sum().float()
    fp = ((pred_label > 0.5) & negative).sum().float()
    fn = ((pred_label <= 0.5) & positive).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-6)
    return {
        f"{prefix}/positive_ratio": float(label.mean().detach().float().item()),
        f"{prefix}/auprc": _average_precision(label, prob),
        f"{prefix}/precision@0.5": float(precision.detach().float().item()),
        f"{prefix}/recall@0.5": float(recall.detach().float().item()),
        f"{prefix}/f1@0.5": float(f1.detach().float().item()),
        f"{prefix}/prob_mean": float(prob.detach().float().mean().item()),
        f"{prefix}/prob_pos_mean": (
            float(prob[positive].detach().float().mean().item()) if positive.any() else None
        ),
        f"{prefix}/prob_neg_mean": (
            float(prob[negative].detach().float().mean().item()) if negative.any() else None
        ),
    }


def _align_sequence_tensor(value, feat, *, offset=1):
    if value is None:
        return None
    target = torch.as_tensor(value, dtype=feat.dtype, device=feat.device)
    if target.ndim == feat.ndim - 1:
        target = target[..., None]
    if target.shape[:-1] != feat.shape[:-1]:
        if target.ndim >= 3 and target.shape[1] >= offset + feat.shape[1]:
            target = target[:, offset : offset + feat.shape[1]]
        elif target.numel() == prod((*feat.shape[:-1], 1)):
            target = target.reshape(*feat.shape[:-1], 1)
    return target.reshape(*feat.shape[:-1], 1)


def _extreme_target(extreme_cost, bottom_force, normalized_cost, feat, threshold):
    if extreme_cost is not None:
        return _align_sequence_tensor(extreme_cost, feat).clamp(0.0, 1.0)
    if bottom_force is not None:
        bottom = _align_sequence_tensor(bottom_force, feat)
        return (bottom > float(threshold)).to(feat.dtype)
    return (normalized_cost > 0.9).to(feat.dtype)


class DFDV2WorldModelWithContinuousCost(ParallelWorldModel):
    """ParallelWorldModel plus a v2 continuous normalized cost head."""

    def __init__(
        self,
        video_log,
        is_proprio,
        obs_shape,
        action_dim,
        stoch,
        discrete,
        hidden,
        stem_ch,
        min_res,
        num_bin,
        max_bin,
        dyn_scale,
        rep_scale,
        val_scale,
        kl_free,
        gamma,
        lambd,
        tau,
        lr,
        eps,
        use_amp,
        act,
        device,
        force_enabled=False,
        force_hidden_dim=256,
        force_depth=4,
        force_dropout=0.1,
        force_eps=1e-3,
        force_scale=1.0,
        force_threshold=0.3,
        force_loss_weight=1.0,
        force_detach_latent=True,
        force_lambda_cls=1.0,
        force_lambda_reg=2.0,
        force_lambda_sign=0.5,
        force_focal_alpha=0.75,
        force_focal_gamma=2.0,
        force_huber_beta=0.5,
        force_reg_weight_power=0.5,
        force_reg_weight_max=10.0,
        force_signed_force=False,
        cost_head_enable=True,
        cost_hidden_dim=320,
        cost_depth=3,
        cost_loss_weight=2.0,
        cost_huber_beta=0.02,
        small_force_threshold=0.3,
        small_cost_threshold=0.05,
        small_cost_weight=2.0,
        extreme_loss_weight=0.5,
        extreme_cost_weight=4.0,
        extreme_force_threshold=5.0,
        cost_prior_loss_weight=0.5,
    ):
        super().__init__(
            video_log,
            is_proprio,
            obs_shape,
            action_dim,
            stoch,
            discrete,
            hidden,
            stem_ch,
            min_res,
            num_bin,
            max_bin,
            dyn_scale,
            rep_scale,
            val_scale,
            kl_free,
            gamma,
            lambd,
            tau,
            lr,
            eps,
            use_amp,
            act,
            device,
            force_enabled,
            force_hidden_dim,
            force_depth,
            force_dropout,
            force_eps,
            force_scale,
            force_threshold,
            force_loss_weight,
            force_detach_latent,
            force_lambda_cls,
            force_lambda_reg,
            force_lambda_sign,
            force_focal_alpha,
            force_focal_gamma,
            force_huber_beta,
            force_reg_weight_power,
            force_reg_weight_max,
            force_signed_force,
        )
        self.cost_head_enable = bool(cost_head_enable)
        self.cost_loss_weight = float(cost_loss_weight)
        self.cost_huber_beta = float(cost_huber_beta)
        self.small_force_threshold = float(small_force_threshold)
        self.small_cost_threshold = float(small_cost_threshold)
        self.small_cost_weight = float(small_cost_weight)
        self.extreme_loss_weight = float(extreme_loss_weight)
        self.extreme_cost_weight = float(extreme_cost_weight)
        self.extreme_force_threshold = float(extreme_force_threshold)
        self.cost_prior_loss_weight = float(cost_prior_loss_weight)
        if self.cost_head_enable:
            self.cost_head = ContinuousCostHead(self.feat_dim, cost_hidden_dim, act, depth=cost_depth)
            self.optimizer.add_param_group({"params": list(self.cost_head.parameters())})
        else:
            self.cost_head = None
        ensure_optimizer_step_no_grad(self.optimizer)

    def _cost_sample_weight(self, target, bottom_force=None, extreme_cost=None):
        weight = torch.ones_like(target)
        small_mask = target < self.small_cost_threshold
        if bottom_force is not None:
            bottom_force = torch.as_tensor(bottom_force, dtype=target.dtype, device=target.device).reshape_as(target)
            small_mask = small_mask | (bottom_force < self.small_force_threshold)
        weight = torch.where(small_mask, weight * self.small_cost_weight, weight)

        if extreme_cost is not None:
            extreme = torch.as_tensor(extreme_cost, dtype=target.dtype, device=target.device).reshape_as(target) > 0.5
        elif bottom_force is not None:
            extreme = bottom_force > self.extreme_force_threshold
        else:
            extreme = target > 0.9
        weight = torch.where(extreme, weight * self.extreme_cost_weight, weight)
        return weight, small_mask, extreme

    def _continuous_cost_loss(self, feat, cost, bottom_force=None, extreme_cost=None):
        if not self.cost_head_enable or cost is None:
            zero = feat.new_tensor(0.0)
            return zero, {}, None, None

        target = _align_sequence_tensor(cost, feat).clamp(0.0, 1.0)
        bottom = _align_sequence_tensor(bottom_force, feat) if bottom_force is not None else None
        extreme = _align_sequence_tensor(extreme_cost, feat) if extreme_cost is not None else None

        outputs = self.cost_head(feat)
        pred = torch.sigmoid(outputs["cost_logit"])
        reg_per = F.smooth_l1_loss(pred, target, beta=self.cost_huber_beta, reduction="none")
        weight, small_mask, extreme_mask = self._cost_sample_weight(target, bottom, extreme)
        reg_loss = _weighted_mean(reg_per, weight)

        if extreme is None:
            if bottom is not None:
                extreme = (bottom > self.extreme_force_threshold).to(target.dtype)
            else:
                extreme = (target > 0.9).to(target.dtype)
        extreme_bce = F.binary_cross_entropy_with_logits(outputs["extreme_logit"], extreme, reduction="none")
        extreme_weight = torch.where(extreme > 0.5, weight, torch.ones_like(weight))
        extreme_loss = _weighted_mean(extreme_bce, extreme_weight)
        total = self.cost_loss_weight * (reg_loss + self.extreme_loss_weight * extreme_loss)

        with torch.no_grad():
            abs_err = (pred - target).abs()
            extreme_prob = torch.sigmoid(outputs["extreme_logit"])
            metrics = {
                "loss": total.detach(),
                "reg_loss": reg_loss.detach(),
                "extreme_loss": extreme_loss.detach(),
                "loss_unweighted": (reg_loss + self.extreme_loss_weight * extreme_loss).detach(),
                "target_mean": target.mean().detach(),
                "target_max": target.max().detach(),
                "pred_mean": pred.mean().detach(),
                "pred_max": pred.max().detach(),
                "mae": abs_err.mean().detach(),
                "small_mae": _safe_mean(abs_err, small_mask).detach(),
                "extreme_mae": _safe_mean(abs_err, extreme_mask).detach(),
                "small_ratio": small_mask.float().mean().detach(),
                "extreme_ratio": extreme.float().mean().detach(),
                "extreme_prob_mean": extreme_prob.mean().detach(),
                "extreme_prob_pos_mean": _safe_mean(extreme_prob, extreme > 0.5).detach(),
            }
        return total, metrics, pred, extreme_prob

    def predict_cost(self, feat):
        if not self.cost_head_enable or self.cost_head is None:
            pred = torch.zeros(*feat.shape[:-1], 1, dtype=feat.dtype, device=feat.device)
            return pred, pred, pred
        input_dtype = feat.dtype
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            outputs = self.cost_head(feat)
            pred = torch.sigmoid(outputs["cost_logit"])
            extreme_prob = torch.sigmoid(outputs["extreme_logit"])
        return pred.to(input_dtype), extreme_prob.to(input_dtype), outputs["cost_logit"].to(input_dtype)

    def update(
        self,
        agent,
        obs,
        action,
        reward,
        done,
        is_first,
        force=None,
        cost=None,
        bottom_force=None,
        extreme_cost=None,
        logger=None,
        step=None,
    ):
        self.train()
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            post, prior, stoch, deter = self.dynamic.parallel_observe(self.encoder(obs), action, is_first)
            dyn_loss, rep_loss, real_kl, ent = self.dynamic.kl_loss(post, prior, self.kl_free)

            obs_hat = self.decoder(stoch)
            done_hat = self.done_head(deter)
            reward_hat = self.reward_head(deter)

            recon_loss = self.mse_loss(obs_hat, obs)
            done_loss = self.bce_logits_loss(done_hat, done)
            reward_loss = self.twohot_loss(reward_hat, reward)

            head_loss = done_loss + self.val_scale * reward_loss
            model_loss = self.dyn_scale * dyn_loss + head_loss
            vae_loss = recon_loss + self.rep_scale * rep_loss

            cost_feat = torch.cat((deter, stoch), dim=-1)
            cost_loss, cost_metrics, cost_pred, extreme_prob = self._continuous_cost_loss(
                cost_feat,
                cost,
                bottom_force=bottom_force,
                extreme_cost=extreme_cost,
            )
            prior_cost_loss = torch.zeros((), dtype=cost_loss.dtype, device=cost_loss.device)
            prior_cost_metrics = {}
            prior_cost_pred = None
            prior_extreme_prob = None
            if self.cost_head_enable and cost is not None and self.cost_prior_loss_weight > 0.0:
                prior_stoch = self.dynamic.get_flatten_stoch(prior)
                prior_feat = torch.cat((prior["deter"], prior_stoch), dim=-1)
                prior_cost_loss, prior_cost_metrics, prior_cost_pred, prior_extreme_prob = self._continuous_cost_loss(
                    prior_feat,
                    cost,
                    bottom_force=bottom_force,
                    extreme_cost=extreme_cost,
                )
                cost_loss = cost_loss + self.cost_prior_loss_weight * prior_cost_loss

            force_losses = None
            force_target = None
            force_pred = None
            force_nonzero_prob = None
            force_loss = torch.zeros((), dtype=vae_loss.dtype, device=self.device)
            if self.force_enabled:
                if force is None:
                    raise ValueError("ForceHead.Enable=True, but DFDV2 world_model.update got force=None.")
                force_target = torch.as_tensor(force, dtype=self.tensor_dtype, device=self.device)
                force_feat = torch.cat((deter, stoch), dim=-1)
                if self.force_detach_latent:
                    force_feat = force_feat.detach()
                force_outputs = self.force_head(force_feat.flatten(0, 1))
                force_losses = self.force_criterion(force_outputs, force_target.flatten(0, 1))
                force_loss = self.force_loss_weight * force_losses["loss"]

            total_loss = model_loss + vae_loss + force_loss + cost_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        with torch.no_grad():
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        metrics = {
            "wm_loss": float(total_loss.detach().float().item()),
            "model_loss": float(model_loss.detach().float().item()),
            "vae_loss": float(vae_loss.detach().float().item()),
            "recon_loss": float(recon_loss.detach().float().item()),
            "reward_loss": float(reward_loss.detach().float().item()),
            "reward_loss_scaled": float((self.val_scale * reward_loss).detach().float().item()),
            "discount_loss": float(done_loss.detach().float().item()),
            "done_loss": float(done_loss.detach().float().item()),
            "dyn_loss": float(dyn_loss.detach().float().item()),
            "rep_loss": float(rep_loss.detach().float().item()),
            "real_kl": float(real_kl.detach().float().item()),
            "vae_ent": float(ent.detach().float().item()),
            "cost_loss": float(cost_loss.detach().float().item()),
            "cost_loss_kind": "continuous_normalized",
            "grad_norm": float(torch.as_tensor(grad_norm).detach().float().item()),
        }
        for key, value in cost_metrics.items():
            metrics[f"cost/{key}"] = float(value.detach().float().item())
        for key, value in prior_cost_metrics.items():
            metrics[f"cost/prior_{key}"] = float(value.detach().float().item())
        metrics["cost/prior_loss"] = float(prior_cost_loss.detach().float().item())
        if cost is not None and cost_pred is not None and extreme_prob is not None:
            target = _align_sequence_tensor(cost, cost_feat).to(cost_pred.dtype)
            metrics["cost/normalized_mse"] = float(F.mse_loss(cost_pred, target).detach().float().item())
            metrics["cost/normalized_mae"] = float((cost_pred - target).abs().mean().detach().float().item())
            extreme_target = _extreme_target(extreme_cost, bottom_force, target, cost_feat, self.extreme_force_threshold)
            metrics.update(_binary_prediction_metrics(extreme_prob, extreme_target, prefix="cost/extreme"))
        if prior_cost_pred is not None and prior_extreme_prob is not None:
            prior_stoch = self.dynamic.get_flatten_stoch(prior)
            prior_feat = torch.cat((prior["deter"], prior_stoch), dim=-1)
            prior_target = _align_sequence_tensor(cost, prior_feat).to(prior_cost_pred.dtype)
            metrics["cost_prior/normalized_mse"] = float(F.mse_loss(prior_cost_pred, prior_target).detach().float().item())
            metrics["cost_prior/normalized_mae"] = float((prior_cost_pred - prior_target).abs().mean().detach().float().item())
            prior_extreme_target = _extreme_target(
                extreme_cost,
                bottom_force,
                prior_target,
                prior_feat,
                self.extreme_force_threshold,
            )
            metrics.update(_binary_prediction_metrics(prior_extreme_prob, prior_extreme_target, prefix="cost_prior/extreme"))

        if logger is not None:
            logger.log("WorldModel/recon_loss", metrics["recon_loss"], step)
            logger.log("WorldModel/reward_loss", metrics["reward_loss"], step)
            logger.log("WorldModel/reward_loss_scaled", metrics["reward_loss_scaled"], step)
            logger.log("WorldModel/dyn_loss", metrics["dyn_loss"], step)
            logger.log("WorldModel/rep_loss", metrics["rep_loss"], step)
            logger.log("WorldModel/real_kl", metrics["real_kl"], step)
            logger.log("WorldModel/vae_ent", metrics["vae_ent"], step)
            logger.log("WorldModel/grad_norm", metrics["grad_norm"], step)
            logger.log("Cost/loss", metrics["cost_loss"], step)
            for key, value in metrics.items():
                if key.startswith("cost/") and isinstance(value, (int, float)):
                    logger.log(key, value, step)

        if self.force_enabled:
            with torch.no_grad():
                force_pred, force_nonzero_prob = predict_force_from_outputs(
                    force_outputs,
                    force_scale=self.force_scale,
                    threshold=self.force_threshold,
                    signed_force=self.force_signed_force,
                )
                flat_force_target = force_target.flatten(0, 1).reshape(-1)
                target_nonzero = (flat_force_target.abs() > self.force_criterion.eps).float()
            metrics.update(
                {
                    "force_loss": float(force_losses["loss"].detach().float().item()),
                    "force_loss_weighted": float(force_loss.detach().float().item()),
                    "force_loss_cls": float(force_losses["loss_cls"].detach().float().item()),
                    "force_loss_reg": float(force_losses["loss_reg"].detach().float().item()),
                    "force_loss_sign": float(force_losses["loss_sign"].detach().float().item()),
                    "force_nonzero_rate": float(target_nonzero.mean().detach().float().item()),
                    "force_target_mean": float(flat_force_target.float().mean().detach().item()),
                    "force_pred_mean": float(force_pred.float().mean().detach().item()),
                    "force_nonzero_prob_mean": float(force_nonzero_prob.float().mean().detach().item()),
                }
            )
            if logger is not None:
                logger.log("Force/loss", metrics["force_loss"], step)
                logger.log("Force/loss_weighted", metrics["force_loss_weighted"], step)
                logger.log("Force/loss_cls", metrics["force_loss_cls"], step)
                logger.log("Force/loss_reg", metrics["force_loss_reg"], step)
                logger.log("Force/loss_sign", metrics["force_loss_sign"], step)
                logger.log("Force/nonzero_rate", metrics["force_nonzero_rate"], step)
                logger.log("Force/target_mean", metrics["force_target_mean"], step)
                logger.log("Force/pred_mean", metrics["force_pred_mean"], step)
                logger.log("Force/nonzero_prob_mean", metrics["force_nonzero_prob_mean"], step)

        return metrics
