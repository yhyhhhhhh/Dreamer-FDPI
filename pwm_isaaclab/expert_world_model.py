from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pwm_isaaclab.modules import networks as net
    from pwm_isaaclab.modules.world_models import ParallelWorldModel, predict_force_from_outputs
except ImportError:
    import modules.networks as net
    from modules.world_models import ParallelWorldModel, predict_force_from_outputs


class HurdleCostHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, act, depth=3):
        super().__init__()
        layers = []
        in_dim = int(input_dim)
        for _ in range(max(int(depth), 1)):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act(),
            ]
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.violation_head = nn.Linear(hidden_dim, 1)
        self.magnitude_head = nn.Linear(hidden_dim, 1)

    def forward(self, feat):
        h = self.backbone(feat)
        return {
            "violation_logit": self.violation_head(h),
            "magnitude_raw": self.magnitude_head(h),
        }


def predict_hurdle_cost(outputs):
    p_violate = torch.sigmoid(outputs["violation_logit"])
    magnitude = torch.relu(torch.expm1(outputs["magnitude_raw"].clamp(max=20.0)))
    return p_violate * magnitude, p_violate, magnitude


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


def cost_prediction_metrics(cost_pred, p_violate, target, prefix="cost"):
    target = torch.as_tensor(target, dtype=cost_pred.dtype, device=cost_pred.device).reshape_as(cost_pred)
    label = (target > 0).to(cost_pred.dtype)
    pred_label = (p_violate >= 0.5).to(cost_pred.dtype)
    positive = label > 0.5
    negative = ~positive
    tp = ((pred_label > 0.5) & positive).sum().float()
    fp = ((pred_label > 0.5) & negative).sum().float()
    fn = ((pred_label <= 0.5) & positive).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)
    err = (cost_pred - target).abs()
    positive_count = int(positive.sum().item())
    metrics = {
        f"{prefix}/positive_ratio": float(label.mean().detach().float().item()),
        f"{prefix}/positive_count": positive_count,
        f"{prefix}/auprc": _average_precision(label, p_violate),
        f"{prefix}/random_auprc_baseline": float(label.mean().detach().float().item()),
        f"{prefix}/precision@0.5": float(precision.detach().float().item()),
        f"{prefix}/recall@0.5": float(recall.detach().float().item()),
        f"{prefix}/f1@0.5": float(f1.detach().float().item()),
        f"{prefix}/mae_all": float(err.mean().detach().float().item()),
        f"{prefix}/mae_positive_only": None,
        f"{prefix}/no_positive_samples": positive_count == 0,
    }
    if positive_count > 0:
        metrics[f"{prefix}/mae_positive_only"] = float(err[positive].mean().detach().float().item())
    return metrics


class ExpertWorldModelWithCost(ParallelWorldModel):
    """ParallelWorldModel plus an independent scalar cost head for expert init."""

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
        cost_loss_type="auto",
        cost_loss_weight=1.0,
        cost_head_mode="hurdle",
        cost_cls_weight=1.0,
        cost_reg_weight=1.0,
        cost_prior_loss_weight=0.5,
        cost_focal_alpha=0.75,
        cost_focal_gamma=2.0,
        cost_pos_weight_max=200.0,
        cost_huber_beta=0.5,
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
        self.deter_dim = int(hidden)
        self.cost_head_mode = str(cost_head_mode).lower()
        if self.cost_head_mode == "hurdle":
            self.cost_head = HurdleCostHead(self.feat_dim, hidden, act, depth=3)
        else:
            self.cost_head = net.Head(hidden, 1, hidden, act)
        self.cost_loss_type = str(cost_loss_type).lower()
        self.cost_loss_weight = float(cost_loss_weight)
        self.cost_cls_weight = float(cost_cls_weight)
        self.cost_reg_weight = float(cost_reg_weight)
        self.cost_prior_loss_weight = float(cost_prior_loss_weight)
        self.cost_focal_alpha = float(cost_focal_alpha)
        self.cost_focal_gamma = float(cost_focal_gamma)
        self.cost_pos_weight_max = float(cost_pos_weight_max)
        self.cost_huber_beta = float(cost_huber_beta)
        self.optimizer.add_param_group({"params": list(self.cost_head.parameters())})

    def _hurdle_cost_loss(self, outputs, cost_target):
        target = torch.as_tensor(cost_target, dtype=outputs["violation_logit"].dtype, device=outputs["violation_logit"].device)
        target = target.reshape_as(outputs["violation_logit"])
        label = (target > 0).to(target.dtype)
        with torch.no_grad():
            positives = label.sum()
            negatives = label.numel() - positives
            if positives > 0:
                pos_weight = (negatives / positives.clamp_min(1.0)).clamp(max=self.cost_pos_weight_max)
            else:
                pos_weight = torch.ones((), dtype=target.dtype, device=target.device)
        bce = F.binary_cross_entropy_with_logits(
            outputs["violation_logit"],
            label,
            reduction="none",
            pos_weight=pos_weight,
        )
        prob = torch.sigmoid(outputs["violation_logit"])
        pt = prob * label + (1.0 - prob) * (1.0 - label)
        alpha_t = self.cost_focal_alpha * label + (1.0 - self.cost_focal_alpha) * (1.0 - label)
        cls_loss = (alpha_t * (1.0 - pt).pow(self.cost_focal_gamma) * bce).mean()

        positive_mask = label > 0.5
        if positive_mask.any():
            reg_target = torch.log1p(target[positive_mask])
            reg_loss = F.smooth_l1_loss(
                outputs["magnitude_raw"][positive_mask],
                reg_target,
                beta=self.cost_huber_beta,
            )
        else:
            reg_loss = torch.zeros((), dtype=target.dtype, device=target.device)
        loss = self.cost_cls_weight * cls_loss + self.cost_reg_weight * reg_loss
        pred_cost, p_violate, magnitude = predict_hurdle_cost(outputs)
        metrics = {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "positive_ratio": label.mean(),
            "pos_weight": pos_weight,
            "p_violate_mean": p_violate.mean(),
            "p_violate_pos_mean": p_violate[positive_mask].mean() if positive_mask.any() else torch.zeros((), dtype=target.dtype, device=target.device),
            "p_violate_neg_mean": p_violate[~positive_mask].mean() if (~positive_mask).any() else torch.zeros((), dtype=target.dtype, device=target.device),
            "predicted_cost_mean": pred_cost.mean(),
            "target_cost_mean": target.mean(),
            "magnitude_mean": magnitude.mean(),
        }
        return loss, metrics, pred_cost, p_violate

    def _cost_loss(self, cost_logits, cost_target):
        target = torch.as_tensor(cost_target, dtype=cost_logits.dtype, device=cost_logits.device)
        if target.ndim == 1:
            target = target[:, None]
        if target.shape != cost_logits.shape:
            target = target.reshape_as(cost_logits)

        loss_type = self.cost_loss_type
        if loss_type == "auto":
            with torch.no_grad():
                finite = target[torch.isfinite(target)]
                is_binary = (
                    finite.numel() > 0
                    and finite.min() >= 0
                    and finite.max() <= 1
                    and torch.allclose(finite, finite.round(), atol=1e-5)
                )
            loss_type = "bce" if is_binary else "mse"

        if loss_type in ("bce", "binary", "bernoulli"):
            return F.binary_cross_entropy_with_logits(cost_logits, target), "bce"
        if loss_type in ("mse", "l2"):
            return F.mse_loss(cost_logits, target), "mse"
        raise ValueError(f"Unsupported cost_loss_type={self.cost_loss_type!r}.")

    def _cost_update_loss(self, deter, stoch, prior, cost):
        if cost is None:
            zero = torch.zeros((), dtype=deter.dtype, device=deter.device)
            return zero, "none", {}, None, None

        if self.cost_head_mode == "hurdle":
            posterior_feat = torch.cat((deter, stoch), dim=-1)
            posterior_outputs = self.cost_head(posterior_feat)
            posterior_loss, posterior_metrics, pred_cost, p_violate = self._hurdle_cost_loss(posterior_outputs, cost)

            prior_stoch = self.dynamic.get_flatten_stoch(prior)
            prior_feat = torch.cat((prior["deter"], prior_stoch), dim=-1)
            prior_target = cost[:, 1 : 1 + prior_feat.shape[1]]
            prior_outputs = self.cost_head(prior_feat)
            prior_loss, prior_metrics, prior_pred_cost, prior_p_violate = self._hurdle_cost_loss(prior_outputs, prior_target)

            raw_loss = posterior_loss + self.cost_prior_loss_weight * prior_loss
            metrics = {
                "posterior_loss": posterior_loss,
                "prior_loss": prior_loss,
                "cls_loss": posterior_metrics["cls_loss"],
                "reg_loss": posterior_metrics["reg_loss"],
                "positive_ratio": posterior_metrics["positive_ratio"],
                "pos_weight": posterior_metrics["pos_weight"],
                "p_violate_mean": posterior_metrics["p_violate_mean"],
                "p_violate_pos_mean": posterior_metrics["p_violate_pos_mean"],
                "p_violate_neg_mean": posterior_metrics["p_violate_neg_mean"],
                "predicted_cost_mean": posterior_metrics["predicted_cost_mean"],
                "target_cost_mean": posterior_metrics["target_cost_mean"],
                "prior_predicted_cost_mean": prior_pred_cost.mean(),
                "prior_p_violate_mean": prior_p_violate.mean(),
            }
            return self.cost_loss_weight * raw_loss, "hurdle", metrics, pred_cost, p_violate

        cost_hat = self.cost_head(deter)
        cost_loss, cost_loss_kind = self._cost_loss(cost_hat, cost)
        if cost_loss_kind == "bce":
            pred_cost = torch.sigmoid(cost_hat)
            p_violate = pred_cost
        else:
            pred_cost = cost_hat
            p_violate = torch.sigmoid(cost_hat)
        return self.cost_loss_weight * cost_loss, cost_loss_kind, {}, pred_cost, p_violate

    def predict_cost(self, feat):
        if self.cost_head_mode == "hurdle":
            outputs = self.cost_head(feat)
            return predict_hurdle_cost(outputs)
        if feat.shape[-1] != self.deter_dim:
            feat = feat[..., : self.deter_dim]
        raw = self.cost_head(feat)
        if self.cost_loss_type in ("bce", "binary", "bernoulli", "auto"):
            pred = torch.sigmoid(raw)
            return pred, pred, pred
        return raw, torch.sigmoid(raw), raw

    def update(self, agent, obs, action, reward, done, is_first, force=None, cost=None, logger=None, step=None):
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

            cost_loss, cost_loss_kind, cost_metrics, cost_pred, cost_prob = self._cost_update_loss(
                deter,
                stoch,
                prior,
                cost,
            )

            force_losses = None
            force_target = None
            force_pred = None
            force_nonzero_prob = None
            force_loss = torch.zeros((), dtype=vae_loss.dtype, device=self.device)
            if self.force_enabled:
                if force is None:
                    raise ValueError("ForceHead.Enable=True, but ExpertWorldModelWithCost.update got force=None.")
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
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
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
            "cost_loss_kind": cost_loss_kind,
        }
        for key, value in cost_metrics.items():
            metrics[f"cost/{key}"] = float(value.detach().float().item())
        if cost is not None and cost_pred is not None and cost_prob is not None:
            metrics.update(cost_prediction_metrics(cost_pred, cost_prob, cost, prefix="cost"))

        if logger is not None:
            logger.log("WorldModel/recon_loss", metrics["recon_loss"], step)
            logger.log("WorldModel/reward_loss", metrics["reward_loss"], step)
            logger.log("WorldModel/reward_loss_scaled", metrics["reward_loss_scaled"], step)
            logger.log("WorldModel/dyn_loss", metrics["dyn_loss"], step)
            logger.log("WorldModel/rep_loss", metrics["rep_loss"], step)
            logger.log("WorldModel/real_kl", metrics["real_kl"], step)
            logger.log("WorldModel/vae_ent", metrics["vae_ent"], step)
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
