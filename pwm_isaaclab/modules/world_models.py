import torch
import torch.distributions as torchd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

try:
    from pwm_isaaclab.modules import functions_losses as func
    from pwm_isaaclab.modules import networks as net
    from pwm_isaaclab.modules import parallel_rnns as rnn
except ImportError:
    import modules.functions_losses as func
    import modules.networks as net
    import modules.parallel_rnns as rnn


params = lambda module: list(module.parameters())
swap = lambda tensor: torch.transpose(tensor, 0, 1)
ste_sample = lambda dist: dist.probs + (dist.sample() - dist.probs).detach()


def _disable_optimizer_dynamo_wrappers():
    if hasattr(torch.optim.Optimizer.add_param_group, "__wrapped__"):
        torch.optim.Optimizer.add_param_group = torch.optim.Optimizer.add_param_group.__wrapped__
    if hasattr(torch.optim.Optimizer.zero_grad, "__wrapped__"):
        torch.optim.Optimizer.zero_grad = torch.optim.Optimizer.zero_grad.__wrapped__
    if hasattr(torch.optim.Optimizer.state_dict, "__wrapped__"):
        torch.optim.Optimizer.state_dict = torch.optim.Optimizer.state_dict.__wrapped__


def _unwrap_optimizer_step(optimizer):
    optimizer_cls = optimizer.__class__
    if hasattr(optimizer_cls.step, "__wrapped__"):
        optimizer_cls.step = optimizer_cls.step.__wrapped__
    if hasattr(optimizer.step, "__wrapped__"):
        optimizer.step = optimizer.step.__wrapped__.__get__(optimizer, optimizer_cls)


class HurdleForceNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        depth=4,
        dropout=0.1,
        signed_force=False,
    ):
        super().__init__()
        self.signed_force = bool(signed_force)

        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.nonzero_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.sign_head = None
        if self.signed_force:
            self.sign_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(self, x):
        h = self.backbone(x)
        return {
            "nonzero_logit": self.nonzero_head(h).squeeze(-1),
            "mag_log_pred": self.magnitude_head(h).squeeze(-1),
            "sign_logit": self.sign_head(h).squeeze(-1) if self.signed_force else None,
        }


class HurdleForceLoss(nn.Module):
    def __init__(
        self,
        eps,
        force_scale,
        lambda_cls=1.0,
        lambda_reg=2.0,
        lambda_sign=0.5,
        focal_alpha=0.75,
        focal_gamma=2.0,
        huber_beta=0.5,
        reg_weight_power=0.5,
        reg_weight_max=10.0,
        signed_force=False,
    ):
        super().__init__()
        self.eps = float(eps)
        self.force_scale = float(force_scale)
        self.lambda_cls = float(lambda_cls)
        self.lambda_reg = float(lambda_reg)
        self.lambda_sign = float(lambda_sign)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.huber_beta = float(huber_beta)
        self.reg_weight_power = float(reg_weight_power)
        self.reg_weight_max = float(reg_weight_max)
        self.signed_force = bool(signed_force)

    def focal_bce_loss(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt).pow(self.focal_gamma) * bce).mean()

    def forward(self, outputs, force):
        force = force.reshape(-1)
        abs_force = force.abs()
        nonzero_target = (abs_force > self.eps).float()
        loss_cls = self.focal_bce_loss(outputs["nonzero_logit"], nonzero_target)

        mag_log_target = torch.log1p(abs_force / self.force_scale)
        pos_mask = nonzero_target > 0.5
        if pos_mask.any():
            reg_loss_raw = F.smooth_l1_loss(
                outputs["mag_log_pred"][pos_mask],
                mag_log_target[pos_mask],
                beta=self.huber_beta,
                reduction="none",
            )
            reg_weight = (abs_force[pos_mask] / self.eps).clamp(min=1.0).pow(self.reg_weight_power)
            reg_weight = reg_weight.clamp(max=self.reg_weight_max)
            loss_reg = (reg_weight * reg_loss_raw).mean()
        else:
            loss_reg = force.new_tensor(0.0)

        if self.signed_force and pos_mask.any():
            sign_target = (force[pos_mask] > 0).float()
            loss_sign = F.binary_cross_entropy_with_logits(
                outputs["sign_logit"][pos_mask], sign_target, reduction="mean"
            )
        else:
            loss_sign = force.new_tensor(0.0)

        total_loss = (
            self.lambda_cls * loss_cls
            + self.lambda_reg * loss_reg
            + self.lambda_sign * loss_sign
        )
        return {
            "loss": total_loss,
            "loss_cls": loss_cls.detach(),
            "loss_reg": loss_reg.detach(),
            "loss_sign": loss_sign.detach(),
        }


@torch.no_grad()
def predict_force_from_outputs(outputs, force_scale, threshold=0.3, signed_force=False):
    p_nonzero = torch.sigmoid(outputs["nonzero_logit"])
    mag = force_scale * torch.expm1(outputs["mag_log_pred"])
    mag = mag.clamp(min=0.0)
    active = p_nonzero >= threshold
    if signed_force:
        sign_prob = torch.sigmoid(outputs["sign_logit"])
        sign = torch.where(sign_prob >= 0.5, 1.0, -1.0)
        force_pred = sign * mag
    else:
        force_pred = mag
    return torch.where(active, force_pred, torch.zeros_like(force_pred)), p_nonzero


class ParallelWorldModel(nn.Module):
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
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden = hidden
        self.stoch_dim = stoch * discrete
        self.feat_dim = self.stoch_dim + hidden
        self.dyn_scale = dyn_scale
        self.rep_scale = rep_scale
        self.val_scale = val_scale
        self.kl_free = kl_free
        self.gamma = gamma
        self.lambd = lambd
        self.tau = tau
        self.device = device
        self.batch_size = -1
        self.horizon = -1
        self.video_log = video_log
        self.is_proprio = is_proprio
        self.force_enabled = bool(force_enabled)
        self.force_threshold = float(force_threshold)
        self.force_scale = float(force_scale)
        self.force_loss_weight = float(force_loss_weight)
        self.force_detach_latent = bool(force_detach_latent)
        self.force_signed_force = bool(force_signed_force)

        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = use_amp

        if is_proprio:
            num_layer, encode_dim = 3, hidden * 2
            self.encoder = net.ProprioEncoder(obs_shape, encode_dim, num_layer, act)
            self.decoder = net.ProprioDecoder(self.stoch_dim, obs_shape, encode_dim, num_layer, act)
        else:
            self.encoder = net.Encoder(obs_shape[0], obs_shape[-1], stem_ch, min_res, act)
            self.decoder = net.Decoder(self.stoch_dim, self.encoder.out_ch, obs_shape[-1], stem_ch, min_res, act)

        self.dynamic = PSSM(stoch, hidden, discrete, action_dim, self.encoder.embed, act, device)
        self.done_head = net.Head(hidden, 1, hidden, act)
        self.reward_head = net.Head(hidden, num_bin, hidden, act)
        if self.force_enabled:
            self.force_head = HurdleForceNet(
                input_dim=self.feat_dim,
                hidden_dim=force_hidden_dim,
                depth=force_depth,
                dropout=force_dropout,
                signed_force=force_signed_force,
            )
        else:
            self.force_head = None

        self.mse_loss = func.MseLoss(is_proprio)
        self.twohot_loss = func.SymLogTwoHotLoss(num_bin, -max_bin, max_bin)
        self.bce_logits_loss = F.binary_cross_entropy_with_logits
        if self.force_enabled:
            self.force_criterion = HurdleForceLoss(
                eps=force_eps,
                force_scale=force_scale,
                lambda_cls=force_lambda_cls,
                lambda_reg=force_lambda_reg,
                lambda_sign=force_lambda_sign,
                focal_alpha=force_focal_alpha,
                focal_gamma=force_focal_gamma,
                huber_beta=force_huber_beta,
                reg_weight_power=force_reg_weight_power,
                reg_weight_max=force_reg_weight_max,
                signed_force=force_signed_force,
            )
        else:
            self.force_criterion = None

        model_params = params(self.dynamic) + params(self.done_head) + params(self.reward_head)
        vae_params = params(self.encoder) + params(self.decoder)
        force_params = params(self.force_head) if self.force_enabled else []
        _disable_optimizer_dynamo_wrappers()
        self.optimizer = torch.optim.AdamW(model_params + vae_params + force_params, lr=lr, eps=eps)
        _unwrap_optimizer_step(self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @torch.no_grad()
    def preprocess(self, obs):
        if self.is_proprio:
            return torch.as_tensor(obs, dtype=self.tensor_dtype, device=self.device)

        tensor_obs = torch.as_tensor(obs, dtype=self.tensor_dtype, device=self.device) / 255
        return tensor_obs.permute(0, 3, 1, 2)[:, None]

    @torch.no_grad()
    def get_inference_feat(self, state, obs, is_first):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            embed = self.encoder(self.preprocess(obs)).squeeze(1)
            obs_stats = self.dynamic.suff_stats_layer("obs", embed)
            obs_stoch = ste_sample(self.dynamic.get_dist(obs_stats))

            is_first = torch.as_tensor(is_first, dtype=self.tensor_dtype, device=self.device)
            if is_first.sum() > 0:
                init_state = self.initial(obs_stoch.shape[0])
                for key, value in state.items():
                    num_axis = value.dim() - is_first.dim()
                    weight = is_first.unflatten(-1, [-1] + [1 for _ in range(num_axis)])
                    state[key] = value * (1 - weight) + init_state[key] * weight

            state.update({"stoch": obs_stoch, **obs_stats})
        return self.dynamic.get_feat(state), state

    @torch.no_grad()
    def update_inference_state(self, state, action):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            deter, _, para_stats, _ = self.dynamic.img_step(state, action, True)
            state.update({"deter": deter, **para_stats})
        return state

    def initial(self, batch_size):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            return self.dynamic.initial(batch_size)

    def init_imagine_buffer(self, batch_size, horizon):
        if self.batch_size != batch_size or self.horizon != horizon:
            init_zeros = lambda shape: torch.zeros(shape, dtype=self.tensor_dtype, device=self.device)
            self.batch_size, self.horizon = batch_size, horizon
            self.deter_buffer = init_zeros((batch_size, horizon + 1, self.hidden))
            self.stoch_buffer = init_zeros((batch_size, horizon + 1, self.stoch_dim))
            self.action_buffer = init_zeros((batch_size, horizon, self.action_dim))

    @torch.no_grad()
    def get_video_frame(self, prior, index):
        stoch = self.dynamic.get_flatten_stoch(prior)
        return self.decoder(stoch[index, None])

    @torch.no_grad()
    def imagine_data(self, agent, obs, action, reward, done, is_first, horizon, logger=None, step=None):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            state, _, _, _ = self.dynamic.parallel_observe(self.encoder(obs), action, is_first)
            img_state = {key: value.flatten(0, 1) for key, value in state.items()}
            batch_size = self.dynamic.get_feat(img_state).shape[0]
            self.init_imagine_buffer(batch_size, horizon)

            video_index, pred_video = torch.randint(batch_size, (1,), device=self.device), []
            for t in range(horizon):
                if logger is not None and not self.is_proprio and step % self.video_log == 0:
                    pred_video.append(self.get_video_frame(img_state, video_index))

                self.deter_buffer[:, t] = self.dynamic.get_deter(img_state)
                self.stoch_buffer[:, t] = self.dynamic.get_flatten_stoch(img_state)
                self.action_buffer[:, t] = agent.sample(
                    torch.cat((self.deter_buffer[:, t], self.stoch_buffer[:, t]), dim=-1)
                )
                img_state = self.dynamic.img_step(img_state, self.action_buffer[:, t])

            self.deter_buffer[:, -1] = self.dynamic.get_deter(img_state)
            self.stoch_buffer[:, -1] = self.dynamic.get_flatten_stoch(img_state)
            feat = torch.cat((self.deter_buffer, self.stoch_buffer), dim=-1)
            discount = (self.done_head(self.deter_buffer[:, 1:]) < 0) * self.gamma
            reward = self.twohot_loss.decode(self.reward_head(self.deter_buffer[:, 1:]))
            weight = torch.cat((torch.ones_like(reward[:, :1]), discount[:, :-1]), dim=1)

        if logger is not None and not self.is_proprio and step % self.video_log == 0:
            logger.log_video("Video/Imagination", torch.cat(pred_video, dim=1), step)
        return feat, self.action_buffer, discount, reward, weight

    def update(self, agent, obs, action, reward, done, is_first, force=None, logger=None, step=None):
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

            force_losses = None
            force_target = None
            force_pred = None
            force_nonzero_prob = None
            force_loss = torch.zeros((), dtype=vae_loss.dtype, device=self.device)
            if self.force_enabled:
                if force is None:
                    raise ValueError("ForceHead.Enable=True, but world_model.update got force=None.")
                force_target = torch.as_tensor(force, dtype=self.tensor_dtype, device=self.device)
                force_feat = torch.cat((deter, stoch), dim=-1)
                if self.force_detach_latent:
                    force_feat = force_feat.detach()
                force_outputs = self.force_head(force_feat.flatten(0, 1))
                force_losses = self.force_criterion(force_outputs, force_target.flatten(0, 1))
                force_loss = self.force_loss_weight * force_losses["loss"]

        self.scaler.scale(model_loss + vae_loss + force_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        with torch.no_grad():
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            logger.log("WorldModel/recon_loss", recon_loss.item(), step)
            logger.log("WorldModel/reward_loss", reward_loss.item(), step)
            logger.log("WorldModel/reward_loss_scaled", (self.val_scale * reward_loss).item(), step)
            logger.log("WorldModel/dyn_loss", dyn_loss.item(), step)
            logger.log("WorldModel/rep_loss", rep_loss.item(), step)
            logger.log("WorldModel/real_kl", real_kl.item(), step)
            logger.log("WorldModel/vae_ent", ent.item(), step)
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
                logger.log("Force/loss", force_losses["loss"].item(), step)
                logger.log("Force/loss_weighted", force_loss.item(), step)
                logger.log("Force/loss_cls", force_losses["loss_cls"].item(), step)
                logger.log("Force/loss_reg", force_losses["loss_reg"].item(), step)
                logger.log("Force/loss_sign", force_losses["loss_sign"].item(), step)
                logger.log("Force/nonzero_rate", target_nonzero.mean().item(), step)
                logger.log("Force/target_mean", flat_force_target.float().mean().item(), step)
                logger.log("Force/pred_mean", force_pred.float().mean().item(), step)
                logger.log("Force/nonzero_prob_mean", force_nonzero_prob.float().mean().item(), step)
            if step % self.video_log == 0 and not self.is_proprio:
                video_index = torch.randint(obs.shape[0], (1,), device=self.device)
                logger.log_video("Video/Observation", obs[video_index], step)
                logger.log_video("Video/Reconstruction", obs_hat[video_index], step)


class PSSM(nn.Module):
    def __init__(self, stoch, hidden, discrete, action_dim, embed, act, device, unimix_ratio=0.01):
        super().__init__()
        self.stoch = stoch
        self.hidden = hidden
        self.discrete = discrete
        self.action_dim = action_dim
        self.unimix_ratio = unimix_ratio
        self.embed = embed
        self.act = act
        self.device = device
        self.num_rnns = 2

        stoch_dim = stoch * discrete
        inp_dim = stoch_dim + action_dim
        self.rnn_layer = self.init_cell()
        self.inp_layer = net.InpLayer(inp_dim, hidden, hidden, act)
        self.ims_stat_layer = net.ImsStatLayer(hidden, stoch_dim, act)
        self.obs_stat_layer = net.ObsStatLayer(embed, stoch_dim, act)

        cell_ws = {}
        for idx in range(self.num_rnns):
            cell_stats = self.rnn_layer[idx].initial(1, idx)
            cell_stats = {key: value.to(device) for key, value in cell_stats.items()}
            cell_ws.update(cell_stats)
        self.cell_ws = cell_ws
        self.init_deter = torch.zeros(1, hidden, requires_grad=False).to(device)

    def init_cell(self):
        return nn.ModuleList([rnn.RNNCell(self.hidden, self.hidden, self.act) for _ in range(self.num_rnns)])

    @torch.no_grad()
    def initial(self, batch_size):
        init = {key: value.expand(batch_size, value.shape[-1]) for key, value in self.cell_ws.items()}
        init_deter = self.init_deter.expand(batch_size, self.init_deter.shape[-1])
        init_logit, init_stoch = self.get_init_stoch(init_deter)
        init.update({"logit": init_logit, "stoch": init_stoch, "deter": init_deter})
        return init

    def get_init_stoch(self, deter):
        stats = self.suff_stats_layer("ims", deter)
        dist = self.get_dist(stats)
        return stats["logit"], dist.mode

    def get_deter(self, state):
        return state["deter"]

    def get_feat(self, state):
        return torch.cat((state["deter"], state["stoch"].flatten(-2, -1)), dim=-1)

    def get_flatten_stoch(self, state):
        return state["stoch"].flatten(-2, -1)

    def get_dist(self, state):
        probs = F.softmax(state["logit"], dim=-1)
        probs = probs * (1 - self.unimix_ratio) + self.unimix_ratio / self.discrete
        return OneHotCategorical(probs=probs)

    def parallel_observe(self, embed, action, is_first):
        init = self.initial(action.shape[0])
        obs_stats = self.suff_stats_layer("obs", embed)
        oracle_stoch = ste_sample(self.get_dist(obs_stats))

        flatten_stoch = oracle_stoch.flatten(-2, -1)
        concat_input = torch.cat((flatten_stoch, action), dim=-1)
        latent, mask = self.inp_layer(concat_input), is_first
        deter, para_stats = self.cell_layers(latent, init, mask, True)

        ims_stats = self.suff_stats_layer("ims", deter[:, :-1])
        ims_stoch = ste_sample(self.get_dist(ims_stats))

        obs_stats = {key: value[:, 1:] for key, value in obs_stats.items()}
        obs_stoch = oracle_stoch[:, 1:]

        stats = {"deter": deter, **para_stats}
        stats = {key: value[:, :-1] for key, value in stats.items()}
        post = {"stoch": obs_stoch, **obs_stats, **stats}
        prior = {"stoch": ims_stoch, **ims_stats, **stats}
        return post, prior, flatten_stoch, deter

    def img_step(self, prev_state, prev_action, return_stats=False):
        prev_stoch = prev_state["stoch"].flatten(-2, -1)
        concat_input = torch.cat((prev_stoch, prev_action), dim=-1)
        deter, para_stats = self.cell_layers(self.inp_layer(concat_input), prev_state, None, False)

        ims_stats = self.suff_stats_layer("ims", deter)
        stoch = ste_sample(self.get_dist(ims_stats))
        if return_stats:
            return deter, stoch, para_stats, ims_stats

        return {"stoch": stoch, "deter": deter, **ims_stats, **para_stats}

    def suff_stats_layer(self, name, x):
        if name == "ims":
            x = self.ims_stat_layer(x)
        elif name == "obs":
            x = self.obs_stat_layer(x)
        else:
            raise NotImplementedError(name)
        return {"logit": x.unflatten(-1, (self.stoch, self.discrete))}

    def cell_layers(self, input, state, is_first, is_parallel):
        if is_parallel:
            deter, is_first = swap(input), swap(is_first)
        else:
            deter = input

        stats = {}
        for idx, layer in enumerate(self.rnn_layer):
            deter, cell_stats = layer(deter, is_first, state, is_parallel, idx)
            stats.update(cell_stats)

        if is_parallel:
            deter = swap(deter)
            stats = {key: swap(value) for key, value in stats.items()}
        return torch.tanh(deter), stats

    def kl_loss(self, post, prior, free):
        kld = torchd.kl.kl_divergence
        dist = lambda state: self.get_dist(state)
        sg = lambda state: {key: value.detach() for key, value in state.items()}

        rep_loss = kld(dist(post), dist(sg(prior)))
        dyn_loss = kld(dist(sg(post)), dist(prior))
        rep_loss = rep_loss.sum(dim=-1).mean()
        dyn_loss = dyn_loss.sum(dim=-1).mean()

        real_kl = dyn_loss
        ent = dist(post).entropy().sum(dim=-1).mean()
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        return dyn_loss, rep_loss, real_kl, ent
