# PSSM 更新原理与当前实现说明

本文档根据当前工程中的实现整理，重点解释：

- PSSM 世界模型的结构与更新原理
- `train_world_model_step()` 到 `ParallelWorldModel.update()` 的更新路径
- `train_agent_step()` 到 `ActorCriticAgent.update()` 的 actor-critic 更新路径
- PSSM/Dreamer 风格在线训练伪代码

对应源码：

- 训练入口：`pwm_isaaclab/train.py`
- 训练循环：`pwm_isaaclab/trainer.py`
- 世界模型：`pwm_isaaclab/modules/world_models.py`
- actor-critic：`pwm_isaaclab/agents.py`
- lambda-return：`pwm_isaaclab/scan.py`
- loss 工具：`pwm_isaaclab/modules/functions_losses.py`

## 1. 整体思想

当前实现是一个 Dreamer 风格的 model-based RL 训练流程：

1. 与真实 IsaacLab 环境交互，把 `(obs, action, reward, done, is_first)` 写入 replay buffer。
2. 用 replay 中的真实序列训练 PSSM 世界模型。
3. 世界模型学会：
   - 编码 observation 到 latent。
   - 用 latent + action 预测下一步 latent。
   - 从 latent 重建 observation。
   - 从 deterministic state 预测 reward。
   - 从 deterministic state 预测 done。
   - 可选地预测 force。
4. actor-critic 不直接在真实环境中更新，而是在世界模型里 rollout 想象轨迹。
5. critic 用 imagined reward/discount 计算 lambda-return。
6. actor 用 imagined advantage 和 entropy 目标更新策略。

所以整体是：

```text
real env -> replay buffer -> train world model
                         -> imagine trajectories with world model
                         -> train actor-critic on imagined trajectories
```

## 2. 主要模块

### 2.1 `ParallelWorldModel`

`ParallelWorldModel` 包含：

| 组件 | 代码名 | 作用 |
| --- | --- | --- |
| observation encoder | `encoder` | 把真实观测编码成 embedding |
| PSSM dynamics | `dynamic` | latent dynamics，类名是 `PSSM` |
| observation decoder | `decoder` | 从 stochastic latent 重建观测 |
| done head | `done_head` | 从 deterministic latent 预测 done logits |
| reward head | `reward_head` | 从 deterministic latent 预测 reward two-hot logits |
| optional force head | `force_head` | 可选，从 latent 预测接触力/force |

在当前 `PWM.yaml` 配置中，默认是 proprio observation，所以使用：

- `ProprioEncoder`
- `ProprioDecoder`

而不是图像卷积 encoder/decoder。

### 2.2 `PSSM`

`PSSM` 是 latent dynamics。每个 latent state 由两部分组成：

```text
state = {
  deter: deterministic state,
  stoch: discrete stochastic state,
  logit: stochastic distribution logits,
  rnn/internal states...
}
```

特征向量 `feat` 是：

```text
feat = concat(deter, flatten(stoch))
```

这也是 actor 和 critic 的输入维度：

```text
feat_dim = hidden + stoch * discrete
```

### 2.3 `ActorCriticAgent`

`ActorCriticAgent` 包含：

| 组件 | 代码名 | 作用 |
| --- | --- | --- |
| actor | `actor` | 输出 action distribution 的 mean/std 参数 |
| critic | `critic` | 输出 value two-hot logits |
| slow critic | `slow_critic` | 可选 target critic，用 EMA/soft update 跟随 critic |
| two-hot loss | `twohot_loss` | reward/value 的 symlog two-hot 表示 |

actor 的 action distribution 是：

```text
Normal(tanh(mean), std)
```

其中：

```text
std = std_scale * sigmoid(raw_std + 2) + std_offset
```

## 3. PSSM 的状态转移逻辑

PSSM 中有两个核心路径：

### 3.1 observation path/posterior

真实观测先经过 encoder：

```text
embed_t = encoder(obs_t)
posterior_logits_t = obs_stat_layer(embed_t)
posterior_stoch_t ~ OneHotCategorical(posterior_logits_t)
```

代码中用 straight-through estimator：

```python
ste_sample = dist.probs + (dist.sample() - dist.probs).detach()
```

这让离散 one-hot latent 在前向是采样值，在反向时近似对概率可导。

### 3.2 imagination path/prior

给定上一步 latent 和 action：

```text
input_t = concat(stoch_t, action_t)
deter_{t+1} = recurrent_transition(input_t, previous_internal_state)
prior_logits_{t+1} = ims_stat_layer(deter_{t+1})
prior_stoch_{t+1} ~ OneHotCategorical(prior_logits_{t+1})
```

`img_step()` 就是这个过程。在线执行策略时，流程是：

```text
current obs -> posterior latent -> actor action -> img_step(action) -> next prior state
```

### 3.3 parallel observe

训练世界模型时，`parallel_observe()` 一次处理整段 replay 序列。它会：

1. 编码所有 `obs`。
2. 得到所有 posterior stochastic states。
3. 把 `posterior_stoch_t` 和 `action_t` 拼起来。
4. 用 parallel RNN/scan 一次计算整段 deterministic states。
5. 构造：
   - `post`：来自真实 observation 的 posterior。
   - `prior`：来自 dynamics prediction 的 prior。

PSSM 的核心训练信号就是让 prior 接近 posterior，同时让 posterior 保留足够表达能力来重建观测、预测 reward/done。

## 4. Replay 数据与训练调度

训练循环在 `joint_train_world_model_agent()` 中。

每次环境 step：

1. 如果 replay buffer 未 warmup，使用随机动作探索。
2. 如果 replay buffer ready，使用：

```text
feat, state = world_model.get_inference_feat(state, current_obs, is_first)
action = agent.sample(feat)
state = world_model.update_inference_state(state, action)
```

3. 执行动作，得到 `next_obs, reward, done, info`。
4. 把当前步写入 replay：

```text
replay.append(current_obs, action, reward, done, is_first, force)
```

5. 如果 replay ready：
   - 每 `TrainModelEverySteps` 更新世界模型。
   - 每 `TrainAgentEverySteps` 更新 actor-critic。

replay sample 返回的张量形状大致是：

```text
obs:      [batch, horizon, obs_dim]
action:   [batch, horizon, action_dim]
reward:   [batch, horizon, 1]
done:     [batch, horizon, 1]
is_first: [batch, horizon, 1]
force:    [batch, horizon, force_dim]  # 可选
```

Replay 会避免采样跨越 episode boundary 的序列窗口。

## 5. `train_world_model_step()`

源码中这个函数很薄：

```python
def train_world_model_step(samples, world_model, agent, logger, step):
    if agent is not None:
        agent.eval()
    world_model.update(agent, *samples, logger=logger, step=step)
```

注意：

- `agent.eval()` 只是把 agent 设为 eval 模式。
- 在当前 `ParallelWorldModel.update()` 里，`agent` 参数没有实际使用。
- 真正的世界模型更新全部在 `world_model.update(...)` 中完成。

## 6. `ParallelWorldModel.update()` 详细说明

函数签名：

```python
def update(self, agent, obs, action, reward, done, is_first, force=None, logger=None, step=None):
```

### 6.1 进入训练模式与 AMP

```python
self.train()
with torch.autocast(...):
    ...
```

如果 `UseAmp=True`，前向会在 autocast 下运行，反向通过 `GradScaler` 缩放。

### 6.2 PSSM posterior/prior rollout

```python
post, prior, stoch, deter = self.dynamic.parallel_observe(
    self.encoder(obs),
    action,
    is_first,
)
```

这里做了三件事：

1. `encoder(obs)` 得到 observation embedding。
2. `parallel_observe()` 计算 posterior 和 prior。
3. 返回：
   - `post`：posterior state，来自真实 observation。
   - `prior`：prior state，来自 dynamics。
   - `stoch`：flatten 后的 stochastic latent。
   - `deter`：deterministic latent。

### 6.3 KL loss：dynamics loss 和 representation loss

```python
dyn_loss, rep_loss, real_kl, ent = self.dynamic.kl_loss(post, prior, self.kl_free)
```

`kl_loss()` 中有两个 KL：

```text
rep_loss = KL( posterior || stop_gradient(prior) )
dyn_loss = KL( stop_gradient(posterior) || prior )
```

代码：

```python
rep_loss = kld(dist(post), dist(sg(prior)))
dyn_loss = kld(dist(sg(post)), dist(prior))
```

含义：

- `dyn_loss` 只更新 prior/dynamics，让 dynamics 追上 posterior。
- `rep_loss` 主要约束 posterior representation，不让 encoder/posterior 任意漂移。
- `kl_free` 是 free nats 下限，KL 太小时不继续惩罚，避免 latent 被过度压缩。

最后：

```text
dyn_loss = max(dyn_loss, kl_free)
rep_loss = max(rep_loss, kl_free)
```

### 6.4 Observation reconstruction

```python
obs_hat = self.decoder(stoch)
recon_loss = self.mse_loss(obs_hat, obs)
```

在 proprio 模式下，`MseLoss` 就是：

```text
0.5 * ||obs_hat - obs||^2
```

对非 proprio 图像模式，代码会先做 uint8 级别比较，完全相同的像素不计残差。

### 6.5 Done prediction

```python
done_hat = self.done_head(deter)
done_loss = binary_cross_entropy_with_logits(done_hat, done)
```

`done_head` 从 deterministic latent `deter` 预测 episode 是否结束。

### 6.6 Reward prediction

```python
reward_hat = self.reward_head(deter)
reward_loss = self.twohot_loss(reward_hat, reward)
```

reward 使用 `SymLogTwoHotLoss`：

1. 先对 reward 做 symlog：

```text
symlog(x) = sign(x) * log(1 + abs(x))
```

2. 把 symlog 后的标量插值到 two-hot bins。
3. 对预测 logits 做 cross entropy。

解码时：

```text
reward = symexp(softmax(logits) @ bins)
```

这样比直接回归 reward 更稳定，尤其 reward 范围比较大时。

### 6.7 Loss 组合

世界模型 loss 分成两组：

```python
head_loss = done_loss + val_scale * reward_loss
model_loss = dyn_scale * dyn_loss + head_loss
vae_loss = recon_loss + rep_scale * rep_loss
```

整体优化目标：

```text
total_world_model_loss =
    dyn_scale * dyn_loss
  + done_loss
  + val_scale * reward_loss
  + recon_loss
  + rep_scale * rep_loss
  + optional_force_loss
```

为什么分成 `model_loss` 和 `vae_loss`？

- `model_loss` 偏 dynamics/head：学习 prior dynamics、done、reward。
- `vae_loss` 偏 representation/reconstruction：学习 posterior latent 和 observation 重建。
- 最终代码还是把它们加起来一起反向传播。

### 6.8 Optional force head

如果 `ForceHead.Enable=True`，还会训练 `force_head`。

```python
force_feat = concat(deter, stoch)
if force_detach_latent:
    force_feat = force_feat.detach()
force_outputs = self.force_head(force_feat.flatten(0, 1))
force_losses = self.force_criterion(force_outputs, force_target.flatten(0, 1))
force_loss = force_loss_weight * force_losses["loss"]
```

`HurdleForceLoss` 把 force 预测拆成：

1. 非零分类：`nonzero_logit`
2. 幅值回归：`mag_log_pred`
3. 可选符号分类：`sign_logit`

总 loss：

```text
force_loss =
    lambda_cls * focal_bce(nonzero)
  + lambda_reg * smooth_l1(log_magnitude)
  + lambda_sign * sign_bce
```

如果 `force_detach_latent=True`，force loss 不反传到 PSSM latent，只训练 force head；当前默认配置里 `DetachLatent=False`，所以 force loss 会参与塑造 latent。

### 6.9 反向传播与优化

```python
self.scaler.scale(model_loss + vae_loss + force_loss).backward()
self.scaler.unscale_(self.optimizer)
clip_grad_norm_(self.parameters(), max_norm=1000.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

世界模型优化器是 `AdamW`，参数包括：

- `dynamic`
- `done_head`
- `reward_head`
- `encoder`
- `decoder`
- 可选 `force_head`

### 6.10 日志

主要日志：

| 指标 | 含义 |
| --- | --- |
| `WorldModel/recon_loss` | observation 重建损失 |
| `WorldModel/reward_loss` | reward two-hot 损失 |
| `WorldModel/reward_loss_scaled` | 乘上 `val_scale` 后的 reward loss |
| `WorldModel/dyn_loss` | prior 追 posterior 的 KL |
| `WorldModel/rep_loss` | representation KL |
| `WorldModel/real_kl` | 未 free-nats 后的 dynamics KL |
| `WorldModel/vae_ent` | posterior distribution entropy |
| `Force/*` | force head 相关日志，可选 |

## 7. `train_agent_step()`

源码：

```python
def train_agent_step(samples, world_model, agent, imagine_horizon, logger, step):
    world_model.eval()
    imagine_outputs = world_model.imagine_data(
        agent,
        *samples[:5],
        imagine_horizon,
        logger,
        step,
    )
    agent.update(*imagine_outputs, logger, step)
```

这个函数做两件事：

1. 冻结世界模型，用 replay 序列作为 imagination 起点，生成 imagined trajectory。
2. 用 imagined trajectory 更新 actor-critic。

`imagine_data()` 被 `@torch.no_grad()` 装饰，所以 actor-critic 更新不会反向传播进世界模型。

## 8. `ParallelWorldModel.imagine_data()` 详细说明

输入：

```text
obs, action, reward, done, is_first
```

注意：这里传入的真实 `reward/done` 只是为了接口一致；想象 rollout 里的 reward/discount 来自世界模型预测。

### 8.1 从真实序列得到 posterior 起点

```python
state, _, _, _ = self.dynamic.parallel_observe(self.encoder(obs), action, is_first)
img_state = {key: value.flatten(0, 1) for key, value in state.items()}
```

这里的 `state` 是 posterior state。把 `[batch, context, ...]` flatten 成 `[batch * context, ...]` 后，每个 posterior latent 都可以作为一个 imagined rollout 的起点。

### 8.2 在 latent 空间 rollout

循环 `horizon` 步：

```python
feat_t = concat(deter_t, stoch_t)
action_t = agent.sample(feat_t)
img_state = dynamic.img_step(img_state, action_t)
```

也就是说，actor 在世界模型 latent 上选动作，PSSM dynamics 在 latent 上预测下一状态。

### 8.3 预测 imagined reward 和 discount

rollout 结束后：

```python
feat = concat(deter_buffer, stoch_buffer)
discount = (done_head(deter_buffer[:, 1:]) < 0) * gamma
reward = twohot_loss.decode(reward_head(deter_buffer[:, 1:]))
weight = concat(ones_like(reward[:, :1]), discount[:, :-1])
```

这里：

- `feat` 长度是 `horizon + 1`，包含起点和每个 imagined next state。
- `action` 长度是 `horizon`。
- `reward` 长度是 `horizon`，对应 imagined next states。
- `discount` 长度是 `horizon`，如果 predicted done logit < 0，则继续，否则 discount 为 0。
- `weight` 用于 actor/critic loss 加权。

当前代码里的 `weight` 是首步为 1，后续用前一时刻的 `discount`，不是完整累计折扣乘积。这一点是当前实现的具体行为。

返回：

```text
feat, imagined_action, discount, imagined_reward, weight
```

## 9. `ActorCriticAgent.update()` 详细说明

函数签名：

```python
def update(self, feat, action, discount, reward, weight, logger=None, step=None):
```

输入来自 `world_model.imagine_data()`：

```text
feat:     [B, H + 1, feat_dim]
action:   [B, H, action_dim]
discount: [B, H, 1]
reward:   [B, H, 1]
weight:   [B, H, 1]
```

### 9.1 Actor distribution 和 critic value

```python
means, stds, raw_value = self.get_logits_raw_value(feat)
stds = std_scale * sigmoid(stds + 2) + std_offset
dist = Normal(tanh(means[:, :-1]), stds[:, :-1])
log_prob = dist.log_prob(action)[..., None]
entropy = dist.entropy()[:, None]
```

actor 输出 `2 * action_dim`：

- 前一半是 mean。
- 后一半是 raw std。

critic 输出 value two-hot logits：

```python
raw_value = critic(feat)
value = twohot_loss.decode(raw_value)
```

### 9.2 Lambda-return target

critic target 在 `torch.no_grad()` 中计算：

```python
value = decode(raw_value)
target_value = decode(slow_critic(feat)) if use_slow_critic else value
lambda_return = parallel_lambda_return(
    reward,
    target_value[:-1],
    target_value[1:],
    discount,
    lambd,
)
```

lambda-return 公式可以写成：

```text
G_t =
  r_t
  + discount_t * (
      (1 - lambda) * V_target(s_{t+1})
      + lambda * G_{t+1}
    )
```

代码中用 parallel scan 实现，因此可以在 GPU 上并行计算整条 imagined trajectory 的 return。

### 9.3 Advantage normalization

```python
norm_adv = (lambda_return - value[:, :-1]) / self.get_scale(lambda_return)
```

`get_scale()` 使用 lambda-return 的分位数 EMA：

```text
scale = EMA(percentile(max_per)) - EMA(percentile(min_per))
scale = max(scale, 1.0)
```

默认配置：

```text
min_per = 0.05
max_per = 0.95
ema_decay = 0.99
```

作用是稳定 advantage 尺度，避免 actor gradient 随 reward 量纲剧烈变化。

### 9.4 Critic loss

```python
critic_loss = mean(
    twohot_loss(raw_value[:, :-1], lambda_return, reduce=False) * weight
)
```

critic 学习把当前 imagined state 的 value logits 对齐到 lambda-return。这里仍用 symlog two-hot loss，而不是 MSE。

### 9.5 Actor loss

代码：

```python
policy_loss = mean(log_prob * norm_adv * weight)
```

最终 total loss 是：

```python
total_loss = critic_loss - policy_loss - entropy_coef * entropy_loss
```

因为优化器最小化 `total_loss`：

- `-policy_loss` 会在 advantage 为正时提高对应 action 的 log probability。
- advantage 为负时降低对应 action 的 log probability。
- `-entropy_coef * entropy_loss` 鼓励更高 entropy，保持探索。

这里是 score-function 风格的 actor 更新：使用 imagined trajectory 中已经采样的 action，重新计算其 log probability，再乘 advantage。

### 9.6 Entropy loss

```python
entropy_loss = mean(entropy * weight)
```

它进入 total loss 的方式是：

```text
- entropy_coef * entropy_loss
```

因此最小化 total loss 会增大 entropy。

### 9.7 反向传播与 slow critic

```python
scaler.scale(total_loss).backward()
clip_grad_norm_(self.parameters(), max_norm=100.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
update_slow_critic()
```

如果 `UseSlowCritic=True`：

```text
slow_critic = (1 - tau) * slow_critic + tau * critic
```

当前 `PWM.yaml` 默认 `UseSlowCritic: False`，所以 target value 直接使用当前 critic 解码值。

## 10. PSSM 在线训练伪代码

下面是当前 `joint_train_world_model_agent()` 的高层伪代码：

```text
initialize env, replay_buffer, world_model, agent
state = world_model.initial(num_envs)
obs_dict = env.reset()
current_obs = obs_dict["policy"]
is_first = obs_dict.get("is_first", zeros)

for iter_idx in range(max_steps / num_envs):
    env_steps = iter_idx * num_envs

    if replay_buffer.ready():
        world_model.eval()
        agent.eval()

        feat, state = world_model.get_inference_feat(
            state,
            current_obs,
            is_first,
        )

        action = agent.sample(feat)
        state = world_model.update_inference_state(state, action)
    else:
        action = random_action()

    next_obs_dict, reward, done, info = env.step(action)

    if replay_buffer.include_force:
        force = extract_force_from_current_obs(current_obs_dict)
    else:
        force = None

    replay_buffer.append(
        current_obs,
        action,
        reward,
        done,
        is_first,
        force,
    )

    update rollout episode statistics

    current_obs_dict = env.reset(seed=done)
    current_obs = current_obs_dict["policy"]
    is_first = current_obs_dict.get("is_first", zeros)

    if replay_buffer.ready():
        if should_update_world_model and replay_buffer.can_sample(batch_length):
            repeat ModelUpdate times:
                samples = replay_buffer.sample(BatchSize, BatchLength)
                train_world_model_step(
                    samples,
                    world_model,
                    agent,
                    logger,
                    env_steps,
                )

        if should_update_agent and replay_buffer.can_sample(imagine_context):
            repeat AgentUpdate times:
                samples = replay_buffer.sample(ImagineBatchSize, ImagineContext)
                train_agent_step(
                    samples,
                    world_model,
                    agent,
                    ImagineHorizon,
                    logger,
                    env_steps,
                )

    if should_save:
        save world_model and agent state_dict
```

## 11. 世界模型更新伪代码

```text
function train_world_model_step(samples):
    agent.eval()
    world_model.update(samples)

function world_model.update(obs, action, reward, done, is_first, force):
    set world_model train mode

    embed = encoder(obs)
    post, prior, stoch, deter = PSSM.parallel_observe(
        embed,
        action,
        is_first,
    )

    dyn_loss, rep_loss, real_kl, entropy = PSSM.kl_loss(
        post,
        prior,
        kl_free,
    )

    obs_hat = decoder(stoch)
    reward_logits = reward_head(deter)
    done_logits = done_head(deter)

    recon_loss = mse(obs_hat, obs)
    reward_loss = symlog_twohot_cross_entropy(reward_logits, reward)
    done_loss = bce_with_logits(done_logits, done)

    total_loss =
        dyn_scale * dyn_loss
        + rep_scale * rep_loss
        + recon_loss
        + done_loss
        + val_scale * reward_loss

    if force_head_enabled:
        force_feat = concat(deter, stoch)
        if force_detach_latent:
            force_feat = stop_gradient(force_feat)

        force_outputs = force_head(force_feat)
        force_loss = hurdle_force_loss(force_outputs, force)
        total_loss += force_loss_weight * force_loss

    backprop total_loss
    clip world_model gradients to 1000
    optimizer.step()
    optimizer.zero_grad()
    log metrics
```

## 12. Imagination 伪代码

```text
function imagine_data(agent, obs, action, reward, done, is_first, horizon):
    world_model.eval()
    no_grad:
        embed = encoder(obs)
        posterior_states = PSSM.parallel_observe(embed, action, is_first).post

        img_state = flatten_batch_and_time(posterior_states)

        for t in range(horizon):
            feat_t = concat(img_state.deter, flatten(img_state.stoch))
            imagined_action_t = agent.sample(feat_t)
            save feat_t and imagined_action_t
            img_state = PSSM.img_step(img_state, imagined_action_t)

        save final feat

        reward_t = decode_twohot(reward_head(deter_{t+1}))
        discount_t = gamma if done_head(deter_{t+1}) < 0 else 0
        weight_t = 1 for first step, previous discount afterward

    return feat, imagined_action, discount, reward, weight
```

## 13. Actor-Critic 更新伪代码

```text
function train_agent_step(samples):
    world_model.eval()
    feat, action, discount, reward, weight =
        world_model.imagine_data(agent, samples, imagine_horizon)

    agent.update(feat, action, discount, reward, weight)

function agent.update(feat, action, discount, reward, weight):
    set agent train mode
    set slow_critic eval mode

    mean, raw_std = actor(feat)
    std = std_scale * sigmoid(raw_std + 2) + std_offset
    dist = Normal(tanh(mean[:, :-1]), std[:, :-1])

    log_prob = dist.log_prob(action)
    entropy = dist.entropy()

    raw_value = critic(feat)
    value = decode_twohot(raw_value)

    no_grad:
        if use_slow_critic:
            target_value = decode_twohot(slow_critic(feat))
        else:
            target_value = value

        lambda_return = parallel_lambda_return(
            reward,
            target_value[:, :-1],
            target_value[:, 1:],
            discount,
            lambda,
        )

        advantage = lambda_return - value[:, :-1]
        norm_advantage = advantage / percentile_ema_scale(lambda_return)

    critic_loss =
        mean(twohot_loss(raw_value[:, :-1], lambda_return) * weight)

    policy_loss =
        mean(log_prob * norm_advantage * weight)

    entropy_loss =
        mean(entropy * weight)

    total_loss =
        critic_loss
        - policy_loss
        - entropy_coef * entropy_loss

    backprop total_loss
    clip agent gradients to 100
    optimizer.step()
    optimizer.zero_grad()

    if use_slow_critic:
        slow_critic = (1 - tau) * slow_critic + tau * critic

    log metrics
```

## 14. 当前默认配置下的重要超参数

来自 `pwm_isaaclab/config_files/PWM.yaml`：

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `BufferWarmUp` | `51200` | replay 超过该步数后开始用策略动作和训练 |
| `BatchSize` | `64` | 世界模型训练 batch |
| `BatchLength` | `64` | 世界模型真实序列长度 |
| `ImagineBatchSize` | `64` | actor-critic imagination 起点 batch |
| `ImagineContext` | `16` | 采样多少真实上下文用于提取 posterior 起点 |
| `ImagineHorizon` | `15` | imagined rollout 长度 |
| `ModelUpdate` | `4` | 每次调度时世界模型更新次数 |
| `AgentUpdate` | `4` | 每次调度时 actor-critic 更新次数 |
| `TrainModelEverySteps` | `256` | 世界模型更新调度间隔 |
| `TrainAgentEverySteps` | `256` | actor-critic 更新调度间隔 |
| `Hidden` | `320` | deterministic latent hidden dim |
| `Stoch` | `24` | stochastic categorical 组数 |
| `Discrete` | `16` | 每组 categorical 类别数 |
| `Gamma` | `0.99` | reward/discount 折扣 |
| `Lambda` | `0.95` | lambda-return 参数 |
| `DynScale` | `0.75` | dynamics KL loss 权重 |
| `RepScale` | `0.15` | representation KL loss 权重 |
| `ValScale` | `1.25` | reward loss 权重 |
| `KLFree` | `1.0` | KL free nats 下限 |
| `EntropyCoef` | `3e-4` | actor entropy 权重 |
| `UseSlowCritic` | `False` | 默认不使用 slow critic target |
| `ForceHead.Enable` | `True` | 默认启用 force prediction head |

## 15. 读代码时容易混淆的点

1. `train_world_model_step()` 和 `train_agent_step()` 都是薄包装，核心分别在 `ParallelWorldModel.update()` 和 `ActorCriticAgent.update()`。
2. `ParallelWorldModel.update()` 的 `agent` 参数当前没有被使用。
3. 世界模型训练使用真实 replay 序列；actor-critic 训练使用世界模型想象序列。
4. `imagine_data()` 是 `no_grad`，所以 actor-critic 更新不会更新世界模型。
5. PSSM 的 posterior 来自真实观测，prior 来自动力学预测；KL loss 的两个方向用 stop-gradient 分开训练 representation 和 dynamics。
6. reward 和 value 都不是直接 MSE 回归，而是 symlog two-hot 分类式回归。
7. actor loss 在代码里叫 `policy_loss = mean(log_prob * norm_adv * weight)`，但 total loss 里是 `-policy_loss`。
8. 当前 `imagine_data()` 的 `weight` 是一步 discount 形式，不是完整 cumulative product。
9. `done_head(deter) < 0` 被当作继续信号；如果 done logit 大于等于 0，则 imagined discount 为 0。
10. 如果启用 force head，force loss 是否反向塑造 latent 取决于 `ForceHead.DetachLatent`。
