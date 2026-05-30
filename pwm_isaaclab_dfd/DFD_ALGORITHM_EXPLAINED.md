# Dual-Feasibility Dreamer 算法说明

本文档说明当前 `pwm_isaaclab_dfd/` 中新增的 Dual-Feasibility Dreamer, 简称 DFD, 分支。

DFD 的目标不是复现完整 SAC-FDPI, 而是在现有 DreamerV3/PSSM 主干上加入一个最小可运行的安全探索和风险调制层:

```text
DreamerV3/PSSM 主干
+ low-ratio dual policy exploration
+ latent Gp/Gd feasibility critics
+ risk-conditioned advantage
```

原始 `pwm_isaaclab/` 代码不被修改。DFD 只 import 和复用原始 env/world model/helper, 新算法代码全部放在 `pwm_isaaclab_dfd/`。

## 1. 总体定位

原 Dreamer/PSSM 仍然负责主要学习:

```text
real rollout -> replay
replay sequence -> PSSM world model update
world model imagination -> Dreamer actor-critic update
```

DFD 只新增三件事:

1. 在 replay 中记录安全 cost 和数据来源 source。
2. 在 Dreamer latent feature 上训练风险 critic `Gp/Gd`。
3. 可选地用 `Gp` 把 Dreamer actor 的 `norm_adv` 调制成 `safe_adv`。

第一版没有额外 main actor-only update。main actor 仍然只通过 Dreamer 原本的 actor-critic update 被更新。

## 2. 训练数据流

完整数据流如下:

```text
env obs
  |
  v
PSSM posterior/inference feat
  |
  +-- main actor action
  |
  +-- optional low-ratio dual actor action
  |
  v
env step
  |
  v
replay stores:
  obs, action, reward, done, is_first, force, cost, source
  |
  +-- world model update, same Dreamer/PSSM logic
  |
  +-- feasibility update, train Gp/Gd on posterior latent
  |
  +-- dual policy update, maximize Gd under KL constraints
  |
  v
Dreamer imagination
  |
  v
RiskConditionedActorCriticAgent.update()
  |
  +-- critic/lambda-return/entropy unchanged
  |
  +-- optional norm_adv -> safe_adv
```

## 3. 三个主开关

配置在 `config_dfd.yaml` 的 `DFD` 段:

```yaml
DFD:
  use_dual: false
  train_feasibility: false
  use_risk_conditioned_advantage: false
```

关闭全部开关时, DFD 分支应尽量等价于原始 Dreamer baseline:

* replay 只额外保存 `cost/source`, 默认 sample tuple 仍兼容原 world model。

* 不更新 `Gp/Gd`。

* 不采样 dual policy。

* 不修改 actor advantage。

* world model 和 actor-critic 主更新仍走 Dreamer 原逻辑。

推荐阶段:

```text
阶段 0: 三开关全关, 验证 baseline 跑通
阶段 1: train_feasibility=true, 只训练 Gp/Gd
阶段 2: use_dual=true, train_feasibility=true, 少量 dual exploration
阶段 3: use_risk_conditioned_advantage=true, 启用 safe_adv
```

## 4. Replay 扩展

`DFDReplayBuffer` 继承原始 `ProprioReplayBuffer`, 新增:

```text
cost_buffer
source_buffer
```

`source` 当前有三类:

```text
0 = main
1 = dual
2 = random warmup
```

默认 `sample()` 返回:

```text
obs, action, reward, done, is_first
```

这保持和原始 world model update 兼容。

当 `return_dict=True` 时返回:

```text
obs, action, reward, done, is_first, cost, source, optional force
```

world model batch 可以用 `world_model_max_dual_fraction` 限制 dual 数据比例, 避免危险样本过多污染 PSSM。

## 5. Cost 定义

第一版 cost 只使用 bottom force, 阈值默认是 `1.0`:

```python
cost = (bottom_force > 1.0).float()
```

提取顺序:

1. `info["bottom_force"]`
2. `info["diagnostics"]["bottom_force"]`
3. `info["bottom_force_peak"]`
4. `info["diagnostics"]["bottom_force_peak"]`
5. `obs["force"]` 的 bottom force 通道

默认 bottom force 通道:

```yaml
bottom_force_channels: [2, 5]
```

如果 `obs["force"]` 只有 2 维, 则取第 2 维作为 bottom force。找不到 bottom force 会直接报错, 不会 fallback 到 failure 或其他 cost。

## 6. Latent Feasibility Critics

`LatentFeasibilityModule` 包含两组 double critic:

```text
Gp:
  gp1, gp2, target_gp1, target_gp2

Gd:
  gd1, gd2, target_gd1, target_gd2
```

输入是 Dreamer latent feature 和 action:

```text
G(feat, action) -> future violation risk
```

当前实现采用 Raw Q-style 输出。网络本身不加 sigmoid, 但 target 和使用处会 clamp 到 `[0, 1]`。

## 7. Posterior Latent 对齐

训练 feasibility critic 时, 从 replay sequence 提取 posterior latent:

```python
embed = world_model.encoder(obs)
post, _, _, _ = world_model.dynamic.parallel_observe(embed, action, is_first)
feat = world_model.dynamic.get_feat(post)
```

当前实现中 `parallel_observe` 的 posterior state 从第一个 replay action 之后开始, 所以对齐为:

```text
z      = feat[:, :-1]
z_next = feat[:, 1:]
action = replay_action[:, 1 : 1 + len(z)]
cost   = replay_cost[:,   1 : 1 + len(z)]
done   = replay_done[:,   1 : 1 + len(z)]
```

语义是:

```text
z_t + action_t -> cost_t + z_{t+1}
```

## 8. Gp 更新

`Gp` 表示 main policy 下的未来违规风险。bootstrap action 来自 main actor:

```python
next_action_main = main_agent.sample(z_next)

target_gp = max(
    target_gp1(z_next, next_action_main),
    target_gp2(z_next, next_action_main),
)
target_gp = clamp(target_gp, 0, 1)
```

TD target:

```python
y_gp = cost + (1 - done) * (1 - cost) * cost_gamma * target_gp
y_gp = clamp(y_gp, 0, 1)
```

loss:

```python
loss_gp =
    mse(gp1(z, action), y_gp)
  + mse(gp2(z, action), y_gp)
```

这里用 `max(gp1, gp2)` 是保守估计, main actor 避险时宁愿高估风险。

## 9. Gd 更新

`Gd` 表示 dual policy 下的未来违规风险。bootstrap action 来自 dual actor:

```python
next_action_dual = dual_policy.sample(z_next)

target_gd = min(
    target_gd1(z_next, next_action_dual),
    target_gd2(z_next, next_action_dual),
)
target_gd = clamp(target_gd, 0, 1)
```

TD target:

```python
y_gd = cost + (1 - done) * (1 - cost) * cost_gamma * target_gd
y_gd = clamp(y_gd, 0, 1)
```

loss:

```python
loss_gd =
    mse(gd1(z, action), y_gd)
  + mse(gd2(z, action), y_gd)
```

`Gd` 服务 dual actor, 用于寻找边界和高风险样本。

## 10. Dual Policy

dual policy 结构和 Dreamer actor 风格一致:

```text
feat -> AgentLayer -> mean, std -> Normal(tanh(mean), std)
```

采样阶段不是一半环境都用 dual, 而是低比例随机启用:

```python
dual_mask = rand(num_envs) < dual_ratio
action[dual_mask] = dual_action[dual_mask]
source[dual_mask] = SOURCE_DUAL
```

默认 warmup:

```yaml
start_step: 100000
ratio_start: 0.01
ratio_final: 0.02
ratio_warmup_steps: 100000
```

如果 dual 和 main 的 KL 超过 `max_kl_for_sampling`, 暂停 dual sampling。

## 11. Dual Policy 更新

dual policy 的目标是最大化 `Gd`, 同时不能离 main policy 太远。

采样 dual action:

```python
a_dual ~ dual_policy(. | feat)
dual_g = min(gd1(feat, a_dual), gd2(feat, a_dual))
dual_g = clamp(dual_g, 0, 1)
```

双向 KL:

```python
KL(dual || main) = E_a~dual [log dual(a|z) - log main(a|z)]
KL(main || dual) = E_a~main [log main(a|z) - log dual(a|z)]
```

dual loss:

```python
dual_loss =
    - mean(dual_g)
  + lambda_dual_main * KL(dual || main)
  + lambda_main_dual * KL(main || dual)
```

更新边界:

```python
lambda_loss =
    lambda_dual_main * (target_kl - KL(dual || main))
  + lambda_main_dual * (target_kl - KL(main || dual))
```

因此当实际 KL 大于 target KL 时, 对应 lambda 会增大, 后续更强地约束 dual actor。

dual update 冻结:

```text
world model
main actor
Gp/Gd parameters
```

但 `Gd` 对 dual action 的梯度仍然可以回传到 dual actor。这个梯度只更新 dual actor, 不更新 main actor。

## 12. Risk-Conditioned Advantage

原 Dreamer actor loss 是:

```python
policy_loss = mean(log_prob * norm_adv * weight)
```

DFD 不改 critic、lambda-return、entropy, 只在开关开启后替换:

```text
norm_adv -> safe_adv
```

风险由 `Gp` 计算:

```python
with torch.no_grad():
    g = max(gp1(feat, action), gp2(feat, action))
    g = clamp(g, 0, 1)
```

三段区域:

```text
feasible:   g < pf - cg
critical:   pf - cg <= g < pf
infeasible: g >= pf
```

风险 margin:

```python
risk_margin = relu(g - (pf - cg)) / (cg + 1e-6)
risk_excess = relu(g - pf) / (1.0 - pf + 1e-6)
```

safe advantage:

```python
safe_adv = zeros_like(norm_adv)

safe_adv[feasible] =
    norm_adv[feasible]

safe_adv[critical] =
    norm_adv[critical]
  - lambda_cri * risk_margin[critical]

safe_adv[infeasible] =
    min(norm_adv[infeasible], 0)
  - lambda_inf * risk_excess[infeasible]
```

最终 actor loss 仍然是 Dreamer 风格:

```python
policy_loss = mean(log_prob * safe_adv.detach() * weight)
```

注意这里的 `Gp` 在 `torch.no_grad()` 下使用。它不会通过 action-gradient 直接更新 main actor。

## 13. 为什么不直接加 Gp penalty

第一版没有使用:

```python
actor_loss += lambda * Gp(feat, action)
```

原因是这会让 `Gp` 的 action-gradient 直接进入 main actor:

```text
dGp / da * da / dtheta_actor
```

在 Dreamer 中这很容易破坏 actor-critic coupling, 尤其当 `Gp` 早期不准或过度悲观时。当前实现只让 `Gp` 调制 score-function advantage, 更温和。

## 14. 明确没有实现的内容

第一版刻意不实现:

```text
joint_plus_fdpi
额外 main actor-only update
SAC-style reward Q
GR recovery critic
完整 importance sampling
Gp direct action-gradient to main actor
cost head 加入 world model loss
```

这能让 DFD 更像 Dreamer 的安全数据增强和 advantage 调制模块, 而不是把 FDPI 作为第二套 actor-critic 强接到 Dreamer 后面。

## 15. 关键监控指标

Replay:

```text
Replay/source_main_ratio
Replay/source_dual_ratio
Replay/source_random_ratio
Replay/cost_rate
Replay/main_cost_rate
Replay/dual_cost_rate
```

Feasibility:

```text
Feasibility/gp_loss
Feasibility/gd_loss
Feasibility/gp_mean
Feasibility/gd_mean
Feasibility/target_gp_mean
Feasibility/target_gd_mean
Feasibility/cost_rate
```

Dual:

```text
Dual/dual_loss
Dual/dual_g
Dual/kl_dual_main
Dual/kl_main_dual
Dual/lambda_dual_main
Dual/lambda_main_dual
Dual/sampling_ratio
Dual/sampling_healthy
```

Actor:

```text
ActorCritic/risk_g_mean
ActorCritic/risk_penalty
ActorCritic/feasible_ratio
ActorCritic/critical_ratio
ActorCritic/infeasible_ratio
ActorCritic/safe_adv
ActorCritic/safe_adv_delta
ActorCritic/risk_lambda_cri
ActorCritic/risk_lambda_inf
```

Rollout:

```text
Rollout/episode_cost
Rollout/episode_success_rate
Rollout/episode_failure_rate
```

## 16. 当前文件职责

```text
pwm_isaaclab_dfd/train_dfd.py
  DFD 训练入口, 构造 env/world model/agent/replay/feasibility/dual policy。

pwm_isaaclab_dfd/trainer_dfd.py
  DFD 在线训练循环, 插入 cost/source, feasibility update, dual update, risk-conditioned agent update。

pwm_isaaclab_dfd/replay_buffer_dfd.py
  带 cost/source 字段的 Dreamer replay buffer。

pwm_isaaclab_dfd/agent_dfd.py
  RiskConditionedActorCriticAgent, 复制原 ActorCritic update 并加入 optional advantage modifier。

pwm_isaaclab_dfd/feasibility.py
  LatentFeasibilityModule, 训练 Gp/Gd 和 target critics。

pwm_isaaclab_dfd/dual_policy.py
  DualPolicy, 最大化 Gd 并受双向 KL 约束。

pwm_isaaclab_dfd/utils.py
  bottom-force cost, posterior latent extraction, safe_adv, KL helper, source constants。

pwm_isaaclab_dfd/config_dfd.yaml
  DFD 独立配置, 默认三开关关闭。
```

## 17. 运行建议

先跑 baseline-disabled DFD:

```bash

PYTHONPATH=/home/yhy/surgical_robot_pro1/exts/ur3_lite:/home/yhy/PaMoRL-IsaacLab-clean:$PYTHONPATH \
/home/yhy/IsaacLab-1.4.0/isaaclab.sh -p \
  /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab_dfd/train_dfd.py \
  -n dfd-baseline-disabled \
  -seed 0 \
  -config_path /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab_dfd/config_dfd.yaml \
  -env_name Ur3Lite-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v2 \
  -device cuda:0 \
  -checkpoint_path /home/yhy/PaMoRL-main/ckpt/ur3-critic-warmup/20260524_223011_warmup/full_agent_after_critic_warmup.pt \
  --no_run_info_prompt
```

确认 baseline 正常后, 再按阶段打开:

```yaml
DFD:
  train_feasibility: true
```

然后:

```yaml
DFD:
  use_dual: true
  train_feasibility: true
```

最后:

```yaml
DFD:
  use_dual: true
  train_feasibility: true
  use_risk_conditioned_advantage: true
```
