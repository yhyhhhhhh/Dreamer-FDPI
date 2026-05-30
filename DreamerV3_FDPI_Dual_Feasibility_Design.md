# DreamerV3 / PSSM 与 FDPI 思想结合方案设计

## 0. 方案定位

本方案的目标不是严格复现 SAC-FDPI，而是把 FDPI 中最有价值、最适合 DreamerV3 的部分抽出来，作为 Dreamer 的安全探索与风险引导模块。

核心定位：

```text
DreamerV3 / PSSM 是主算法。
FDPI 提供：
1. dual policy：主动收集危险 / 边界 / 违规样本；
2. Gp / Gd：在 Dreamer latent space 中估计未来违规风险；
3. feasible / critical / infeasible 分段思想：调制 Dreamer actor 的 advantage。
```

不要把 FDPI 当成第二套 actor-critic 去强行并入 Dreamer。更合理的方式是：

```text
Dreamer 负责：
- world model
- latent imagination
- reward prediction
- value / lambda-return
- main actor-critic joint update

FDPI 负责：
- dual policy exploration
- violation / boundary sample augmentation
- latent feasibility critic
- risk-conditioned actor update
```

因此推荐方案可以命名为：

```text
Dual-Feasibility Dreamer
```

---

## 1. 当前问题与设计原则

### 1.1 之前方案容易崩塌的主要原因

之前的 Dreamer-FDPI 方案容易出现策略崩塌，核心原因通常不是“FDPI 思想不适合 Dreamer”，而是结合方式过激。

常见问题包括：

```text
1. main actor 一轮中被更新两次：
   Dreamer joint update 一次；
   FDPI actor-only update 再一次。

2. actor-only update 脱离 Dreamer critic：
   actor 使用的目标和 critic / lambda-return 不同步。

3. Gp 风险项太早、太强地作用于 actor：
   actor 可能直接学成远离目标、不靠近物体、不夹取。

4. dual policy 数据比例过高或太早进入 replay：
   world model 被失败 / 危险轨迹污染。

5. Gp 在真实 posterior latent 上训练，
   但在 imagined prior latent 上使用，
   如果不校准，可能误判 imagined state 的风险。
```

### 1.2 新方案的基本原则

新方案遵循以下原则：

```text
原则 1：Dreamer actor-critic joint update 必须保留。
原则 2：main actor 只在 ActorCriticAgent.update() 中更新。
原则 3：不再使用额外的 main_actor_update_step()。
原则 4：不再使用 joint_plus_fdpi 这种“joint 后再补一刀 actor-only”的结构。
原则 5：第一版不额外复刻 SAC Q。
原则 6：第一版不使用 GR recovery branch。
原则 7：dual policy 先作为数据增强机制，而不是直接指导 main actor。
原则 8：Gp/Gd 先训练和诊断，确认可靠后再影响 actor。
原则 9：Gp 第一版只调制 advantage，不直接通过 action-gradient 强推 actor。
```

---

## 2. 总体结构

完整数据流如下：

```text
真实环境交互
    ↓
main policy / dual policy 混合采样
    ↓
replay buffer:
    obs, action, reward, cost, done, is_first, source
    ↓
PSSM world model update
    ↓
posterior latent extraction
    ↓
latent feasibility critic update:
    Gp / Gd
    ↓
dual policy update:
    maximize Gd + KL constraint
    ↓
Dreamer imagination
    ↓
main actor-critic joint update:
    critic 正常更新
    actor 使用 risk-conditioned advantage
```

模块关系：

```text
                 ┌──────────────────────┐
                 │      Real Env         │
                 └──────────┬───────────┘
                            │
              main action / dual action
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Replay Buffer         │
                 │ obs, action, reward   │
                 │ cost, done, source    │
                 └──────────┬───────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌────────────────┐   ┌────────────────┐
│ PSSM update  │   │ Gp/Gd update    │   │ dual update     │
│ real seq     │   │ posterior z,a   │   │ maximize Gd     │
└──────┬───────┘   └────────────────┘   └────────────────┘
       │
       ▼
┌─────────────────────┐
│ Dreamer imagination │
│ feat, action, reward│
└──────────┬──────────┘
           ▼
┌───────────────────────────────┐
│ ActorCriticAgent.update()     │
│ critic: lambda-return         │
│ actor: log_prob * safe_adv    │
└───────────────────────────────┘
```

---

## 3. 需要新增或修改的模块

## 3.1 Replay Buffer 扩展

当前 PSSM replay 至少包含：

```text
obs
action
reward
done
is_first
force
```

新方案需要扩展为：

```text
obs
action
reward
cost
done
is_first
force
source
logp_main        # 可选
logp_dual        # 可选
log_weight_main  # 可选
log_weight_dual  # 可选
```

### 必须新增字段

#### 1. `cost`

用于表示当前 transition 是否违反安全约束。

```python
cost = extract_cost(info, next_obs)
```

推荐形状：

```text
[B, T, 1]
```

取值：

```text
0：未 violation
1：violation
```

如果环境提供连续 cost，也建议第一版先转成 binary violation：

```python
cost = (raw_cost > 0).float()
```

#### 2. `source`

记录样本来源：

```text
0 = main
1 = dual
2 = random
```

作用：

```text
1. 监控 dual 数据比例；
2. 控制 world model batch 中 dual 样本比例；
3. 诊断 dual 是否污染 replay；
4. 后续可用于 clipped IS 或分布修正。
```

### 可选字段

#### 1. `logp_main`, `logp_dual`

如果后续想做 KL / IS 分析，可以记录采样动作在 main policy 和 dual policy 下的 log probability。

第一版可以不强制使用 IS，只记录，方便后续诊断。

#### 2. `log_weight_main`, `log_weight_dual`

FDPI 原实现中使用 importance sampling 修正 dual/main 数据分布。Dreamer 第一版不建议直接重用完整 IS，因为容易引入高方差。

如果使用，建议：

```python
w = torch.clamp(torch.exp(log_weight), 0.1, 10.0)
```

并且只用于 Gp/Gd update，不用于 Dreamer actor update。

---

## 3.2 新增 Latent Feasibility Module

新增一个模块，例如：

```python
class LatentFeasibilityModule(nn.Module):
    def __init__(self, feat_dim, action_dim):
        self.gp1 = Critic(feat_dim + action_dim, 1)
        self.gp2 = Critic(feat_dim + action_dim, 1)
        self.target_gp1 = deepcopy(self.gp1)
        self.target_gp2 = deepcopy(self.gp2)

        self.gd1 = Critic(feat_dim + action_dim, 1)
        self.gd2 = Critic(feat_dim + action_dim, 1)
        self.target_gd1 = deepcopy(self.gd1)
        self.target_gd2 = deepcopy(self.gd2)
```

### Gp 的含义

```text
Gp(z, a)
```

表示：

```text
在 latent state z 执行动作 a，
之后继续按照 main policy 行动，
未来发生 violation 的风险。
```

### Gd 的含义

```text
Gd(z, a)
```

表示：

```text
在 latent state z 执行动作 a，
之后继续按照 dual policy 行动，
未来发生 violation 的风险。
```

### 为什么用 latent 而不是 obs

Dreamer 的 actor 和 critic 都工作在 latent feature 上：

```text
feat = concat(deter, flatten(stoch))
```

所以 Gp/Gd 也应该输入同一个 feat：

```text
FDPI 的 x  → Dreamer latent feat
FDPI 的 u  → env action
```

这样 Gp/Gd 可以直接服务于 Dreamer actor update。

---

## 3.3 新增 Dual Policy

新增一个 dual actor，结构可以和 main actor 一致：

```python
dual_actor = Actor(feat_dim, action_dim)
```

dual policy 输入：

```text
feat = concat(deter, stoch)
```

输出：

```text
action distribution
```

dual policy 的目标不是完成任务，而是：

```text
寻找高风险 / 边界 / violation 样本
```

但 dual 不能离 main 太远，所以要加 KL 约束：

```text
dual_loss =
    - Gd(z, a_dual)
    + lam_dual_main * KL(dual || main)
    + lam_main_dual * KL(main || dual)
```

---

## 3.4 修改训练循环

当前训练循环中主要有：

```text
train_world_model_step()
train_agent_step()
```

新方案增加：

```text
train_feasibility_step()
train_dual_policy_step()
```

最终训练循环变成：

```text
if train world model:
    train_world_model_step()

if train feasibility:
    train_feasibility_step()

if train dual policy:
    train_dual_policy_step()

if train main agent:
    train_agent_step()
```

其中：

```text
train_world_model_step:
    训练 PSSM，不改 main actor，不改 dual actor。

train_feasibility_step:
    用 replay posterior latent 训练 Gp/Gd。

train_dual_policy_step:
    冻结 world model、main actor、Gp/Gd；
    只更新 dual actor 和 KL multipliers。

train_agent_step:
    走原 Dreamer imagination；
    critic 正常更新；
    actor 使用 risk-conditioned advantage。
```

---

## 4. 真实环境采样设计

### 4.1 不建议直接照搬 FDPI 的半数环境 dual 采样

FDPI 当前实现中，环境数必须为偶数，前半环境使用 main policy，后半环境使用 dual policy。

这个设计在 SAC-FDPI 中可以工作，但对 Dreamer 不一定稳。因为 Dreamer 的 world model 对 replay 分布非常敏感。如果一半数据来自 dual policy，world model 很可能被危险 / 失败轨迹带偏。

所以新方案建议用低比例随机激活 dual：

```python
use_dual = (
    step >= dual_start_step
    and random.random() < dual_ratio
    and dual_is_healthy
)
```

### 4.2 推荐采样逻辑

```python
feat, state = world_model.get_inference_feat(
    state,
    current_obs,
    is_first,
)

if not replay.ready():
    action = random_action()
    source = RANDOM

else:
    if step < dual_start_step:
        action = main_actor.sample(feat)
        source = MAIN

    else:
        use_dual = (
            torch.rand(()) < dual_ratio
            and dual_is_healthy
        )

        if use_dual:
            action = dual_actor.sample(feat)
            source = DUAL
        else:
            action = main_actor.sample(feat)
            source = MAIN

next_obs, reward, done, info = env.step(action)
cost = extract_cost(info, next_obs)

replay.append(
    obs=current_obs,
    action=action,
    reward=reward,
    cost=cost,
    done=done,
    is_first=is_first,
    source=source,
)
```

### 4.3 推荐 dual 采样比例

第一版配置：

```yaml
dual_start_step: 100000
dual_ratio: 0.01
dual_ratio_max: 0.02
dual_ratio_warmup_steps: 100000
```

稳定后可以尝试：

```yaml
dual_ratio: 0.05
```

不建议第一版超过：

```yaml
dual_ratio: 0.1
```

---

## 5. World Model 更新

### 5.1 World model 主体不改

继续使用当前：

```python
world_model.update(
    agent,
    obs,
    action,
    reward,
    done,
    is_first,
    force,
)
```

PSSM 仍然学习：

```text
observation reconstruction
reward prediction
done prediction
prior / posterior KL
optional force prediction
```

### 5.2 是否把 cost 加进 world model

第一版不建议直接把 cost head 加进 PSSM loss。

理由：

```text
1. 当前 world model 已经比较复杂；
2. force head、reward head、done head 已经参与 latent 学习；
3. 直接加 cost head 可能改变 latent 表示，增加不稳定性；
4. Gp/Gd 已经承担 cost / feasibility 学习任务。
```

可以保留为第二阶段扩展：

```python
cost_logits = cost_head(deter)
cost_loss = BCEWithLogitsLoss(cost_logits, cost)
total_world_model_loss += cost_loss_weight * cost_loss
```

但第一版先不加。

### 5.3 world model batch 中控制 dual 比例

建议 replay sample 支持：

```yaml
world_model_max_dual_fraction: 0.1
```

含义：

```text
每个 world model batch 中，dual source 样本比例不超过 10%。
```

如果实现复杂，第一版也可以先不控制，但必须记录：

```text
WorldModel/source_dual_ratio
WorldModel/reward_loss
WorldModel/dyn_loss
WorldModel/rep_loss
```

如果开启 dual 后 world model loss 明显恶化，优先降低 dual_ratio。

---

## 6. Feasibility Critic 更新

### 6.1 从 replay 中提取 posterior latent

训练 Gp/Gd 时使用真实 replay sequence，但输入不是 obs，而是 posterior latent：

```python
with torch.no_grad():
    embed = world_model.encoder(obs)
    post, prior, stoch, deter = world_model.dynamic.parallel_observe(
        embed,
        action,
        is_first,
    )

    feat = concat(post["deter"], flatten(post["stoch"]))
```

需要得到：

```text
z_t
a_t
z_{t+1}
cost_t
done_t
```

如果 batch 是 `[B, T, ...]`，则：

```python
z      = feat[:, :-1]
z_next = feat[:, 1:]
a      = action[:, :-1] or action aligned with z
cost   = cost[:, :-1]
done   = done[:, :-1]
```

注意要严格对齐：

```text
z_t + action_t → cost_t + z_{t+1}
```

### 6.2 Gp 更新

Gp bootstrap 使用 main actor：

```python
with torch.no_grad():
    next_action_main = main_actor.sample(z_next)

    target_gp = torch.maximum(
        target_gp1(z_next, next_action_main),
        target_gp2(z_next, next_action_main),
    )

    target_gp = target_gp.clamp(0, 1)

    y_gp = cost + (1 - done) * (1 - cost) * cost_gamma * target_gp
    y_gp = y_gp.clamp(0, 1)
```

loss：

```python
pred_gp1 = gp1(z, action)
pred_gp2 = gp2(z, action)

loss_gp = mse(pred_gp1, y_gp) + mse(pred_gp2, y_gp)
```

这里使用 `max(target_gp1, target_gp2)`，是因为 main policy 需要保守避险，宁愿高估风险一点。

### 6.3 Gd 更新

Gd bootstrap 使用 dual actor：

```python
with torch.no_grad():
    next_action_dual = dual_actor.sample(z_next)

    target_gd = torch.minimum(
        target_gd1(z_next, next_action_dual),
        target_gd2(z_next, next_action_dual),
    )

    target_gd = target_gd.clamp(0, 1)

    y_gd = cost + (1 - done) * (1 - cost) * cost_gamma * target_gd
    y_gd = y_gd.clamp(0, 1)
```

loss：

```python
pred_gd1 = gd1(z, action)
pred_gd2 = gd2(z, action)

loss_gd = mse(pred_gd1, y_gd) + mse(pred_gd2, y_gd)
```

### 6.4 是否使用 IS 权重

第一版建议不用完整 IS。

如果要用，只用于 Gp/Gd，并且必须 clip：

```python
weight = torch.clamp(torch.exp(log_weight), 0.1, 10.0)
loss_gp = mean(weight * (pred_gp - y_gp) ** 2)
```

不要让 IS 权重进入 Dreamer actor loss。

---

## 7. Dual Policy 更新

### 7.1 dual policy 的作用

dual policy 的作用是：

```text
主动寻找高风险动作，
让 replay 中持续存在 violation / boundary samples，
从而让 Gp/Gd 学得更准。
```

它不是用于部署，也不是用于替代 main actor。

### 7.2 dual loss

从 replay 中取 posterior latent `z`：

```python
with torch.no_grad():
    z = posterior_feat(world_model, batch)
```

采样 dual 动作：

```python
a_dual, logp_dual = dual_actor.sample(z)
```

计算 dual risk：

```python
with torch.no_grad():
    gd = torch.minimum(
        gd1(z, a_dual),
        gd2(z, a_dual),
    )
```

计算 KL：

```python
kl_dual_main = E_{a ~ dual}[
    log dual(a | z) - log main(a | z)
]

kl_main_dual = E_{a ~ main}[
    log main(a | z) - log dual(a | z)
]
```

dual loss：

```python
dual_loss =
    - gd.mean()
    + lam_dual_main * kl_dual_main
    + lam_main_dual * kl_main_dual
```

### 7.3 KL multiplier 更新

如果实际 KL 超过目标，就增大 lambda：

```python
lam_dual_main_loss = - lam_dual_main * stop_gradient(kl_dual_main - target_kl)
lam_main_dual_loss = - lam_main_dual * stop_gradient(kl_main_dual - target_kl)
```

或者等价写成梯度下降形式，只要保证：

```text
KL > target_kl  → lambda 增大
KL < target_kl  → lambda 减小或不变
```

### 7.4 推荐 KL 配置

```yaml
dual_target_kl: 0.5
dual_kl_lambda_init: 1.0
dual_kl_lambda_lr: 1e-4
dual_max_kl_for_sampling: 2.0
```

如果：

```text
kl_dual_main > dual_max_kl_for_sampling
```

则暂停 dual 采样，只训练 dual 直到 KL 回到合理范围。

---

## 8. Main Actor-Critic 更新

这是新方案最关键的部分。

### 8.1 Critic 保持原 Dreamer 更新

critic loss 不改：

```python
critic_loss =
    mean(twohot_loss(raw_value[:, :-1], lambda_return) * weight)
```

lambda-return、slow critic、advantage normalization 也保持原样。

### 8.2 原始 Dreamer actor loss

当前 Dreamer actor loss 是：

```python
policy_loss = mean(log_prob * norm_adv * weight)

total_loss =
    critic_loss
    - policy_loss
    - entropy_coef * entropy_loss
```

含义：

```text
advantage > 0：提高该 action 的概率
advantage < 0：降低该 action 的概率
```

### 8.3 新方案：risk-conditioned advantage

不直接改成：

```python
actor_loss = -q_dreamer + lambda * Gp(z, action)
```

而是先把风险转成 advantage 调制项：

```python
safe_adv = norm_adv - risk_penalty
```

然后仍然使用 Dreamer 原来的策略梯度形式：

```python
policy_loss = mean(log_prob * safe_adv.detach() * weight)
```

这样 main actor 的更新语义保持一致：

```text
safe_adv 高的动作：提高概率
safe_adv 低的动作：降低概率
```

### 8.4 计算 Gp 风险

在 `ActorCriticAgent.update()` 中传入 feasibility module，或传入一个 `actor_adv_fn` 回调。

```python
with torch.no_grad():
    g = torch.maximum(
        gp1(feat[:, :-1], action),
        gp2(feat[:, :-1], action),
    )
```

注意：第一版推荐 `torch.no_grad()`，也就是 Gp 不直接通过 action-gradient 更新 actor。

### 8.5 三段式 safe advantage

定义：

```python
g_det = g.detach()

fea = g_det < pf - cg
cri = (g_det >= pf - cg) & (g_det < pf)
inf = g_det >= pf
```

风险项：

```python
risk_margin = relu(g_det - (pf - cg)) / (cg + 1e-6)
risk_excess = relu(g_det - pf) / (1.0 - pf + 1e-6)
```

safe advantage：

```python
safe_adv = torch.zeros_like(norm_adv)

safe_adv[fea] = norm_adv[fea]

safe_adv[cri] = (
    norm_adv[cri]
    - lambda_cri * risk_margin[cri]
)

safe_adv[inf] = (
    torch.minimum(norm_adv[inf], torch.zeros_like(norm_adv[inf]))
    - lambda_inf * risk_excess[inf]
)
```

解释：

```text
feasible 区域：
    完全按 Dreamer 原始 advantage 学任务。

critical 区域：
    仍允许 reward 推动策略，
    但开始扣除风险。

infeasible 区域：
    不再允许高 reward 危险动作被继续鼓励；
    即使 norm_adv > 0，也最多截断到 0；
    再额外扣除 risk_excess。
```

最终：

```python
policy_loss = mean(log_prob * safe_adv.detach() * weight)

total_loss =
    critic_loss
    - policy_loss
    - entropy_coef * entropy_loss
```

### 8.6 为什么第一版不直接让 Gp 对 action 反传

如果写成：

```python
loss += lambda * Gp(z, action)
```

那么 Gp 会通过：

```text
∂Gp/∂action * ∂action/∂actor
```

直接推 actor 改动作方向。

这不一定错，但第一版容易导致：

```text
1. Gp 太悲观时 actor 远离目标；
2. risk gradient 压过 reward gradient；
3. main policy 快速变保守；
4. Dreamer actor-critic coupling 被破坏。
```

所以第一版只让 Gp 调制 advantage，不让它直接提供 action-gradient。

---

## 9. 训练阶段安排

## 阶段 0：原始 Dreamer baseline

配置：

```yaml
use_dual: false
train_feasibility: false
use_risk_conditioned_advantage: false
```

目标：

```text
确认原始 Dreamer 可以正常靠近目标、夹取、提升 return。
```

必须记录：

```text
episode_return
success_rate
episode_cost
ActorCritic/entropy
ActorCritic/norm_adv
WorldModel/reward_loss
WorldModel/dyn_loss
WorldModel/rep_loss
```

---

## 阶段 1：只记录 cost，并训练 Gp/Gd，不影响策略

配置：

```yaml
use_dual: false
train_feasibility: true
use_risk_conditioned_advantage: false
```

目标：

```text
确认 Gp 在 main policy 数据上能否学出风险。
```

看：

```text
Feasibility/gp_loss
Feasibility/gp_mean
Feasibility/gp_pos_mean
Feasibility/gp_neg_mean
Feasibility/violation_ratio
```

如果 violation 样本太少，Gp 可能全输出低风险。这正是之后需要 dual policy 的原因。

---

## 阶段 2：开启少量 dual exploration，但仍不影响 main actor

配置：

```yaml
use_dual: true
dual_start_step: 100000
dual_ratio: 0.01
dual_ratio_max: 0.02
train_feasibility: true
use_risk_conditioned_advantage: false
```

目标：

```text
验证 dual policy 是否能增加 violation / boundary samples，
同时不导致 main policy 崩塌。
```

看：

```text
Replay/source_dual_ratio
Replay/dual_cost_ratio
Replay/main_cost_ratio
Dual/kl_dual_main
Dual/kl_main_dual
Dual/gd_mean
WorldModel/reward_loss
WorldModel/dyn_loss
episode_return
success_rate
```

如果 main return 掉崖：

```text
优先降低 dual_ratio 或推迟 dual_start_step。
```

---

## 阶段 3：启用 risk-conditioned advantage

配置：

```yaml
use_dual: true
train_feasibility: true
use_risk_conditioned_advantage: true
use_direct_gp_gradient: false
use_gr: false

risk_lambda_cri_start: 0.0
risk_lambda_cri_final: 0.02
risk_lambda_inf_start: 0.0
risk_lambda_inf_final: 0.05
risk_lambda_warmup_steps: 50000
```

目标：

```text
让 main actor 在危险区域降低动作概率，
但不破坏 Dreamer 原有学习。
```

如果出现不靠近目标：

```text
降低 lambda_cri / lambda_inf；
增大 pf；
减小 cg；
延后 risk-conditioned advantage 启用时间。
```

---

## 阶段 4：可选，加入 direct Gp penalty

只有阶段 3 稳定后再尝试。

配置：

```yaml
use_direct_gp_gradient: true
direct_gp_lambda: 1e-4
```

loss：

```python
direct_loss = mask * Gp(z.detach(), action_rsample)
total_loss += direct_gp_lambda * direct_loss.mean()
```

注意：

```text
z detach；
Gp 参数冻结；
只让梯度回到 actor；
lambda 必须非常小。
```

---

## 阶段 5：可选，加入 GR recovery branch

第一版不推荐加入 GR。

只有在以下条件满足后再考虑：

```text
1. Gp/Gd 已稳定；
2. dual policy 不污染 replay；
3. risk-conditioned advantage 不导致策略退化；
4. violation branch 定义清楚；
5. 已经有足够 violation 后恢复的数据。
```

---

## 10. 推荐配置模板

```yaml
DreamerFDPI:
  Enable: true

  Replay:
    StoreCost: true
    StoreSource: true
    WorldModelMaxDualFraction: 0.10

  Feasibility:
    Enable: true
    FeatInput: "deter_stoch"
    CostGamma: 0.97
    Pf: 0.10
    Cg: 0.03
    HiddenDim: 256
    NumLayers: 2
    DoubleCritic: true
    TargetTau: 0.005
    UseIS: false
    ClipISMin: 0.1
    ClipISMax: 10.0

  DualPolicy:
    Enable: true
    StartStep: 100000
    RatioStart: 0.01
    RatioFinal: 0.02
    RatioWarmupSteps: 100000
    TargetKL: 0.5
    MaxKLForSampling: 2.0
    LambdaKLInit: 1.0
    LambdaLR: 1.0e-4

  MainActorRisk:
    Enable: true
    StartStep: 150000
    UseRiskConditionedAdvantage: true
    UseDirectGpGradient: false
    LambdaCriFinal: 0.02
    LambdaInfFinal: 0.05
    LambdaWarmupSteps: 50000
    ClipSafeAdv: true
    SafeAdvMin: -5.0
    SafeAdvMax: 5.0

  Recovery:
    EnableGR: false
```

---

## 11. 需要改哪些代码

### 11.1 Replay Buffer

需要新增字段：

```python
cost
source
```

可选新增：

```python
logp_main
logp_dual
log_weight
log_weight_dual
```

需要修改：

```text
append()
sample()
sample_sequence()
```

确保返回：

```text
obs, action, reward, cost, done, is_first, force, source
```

---

### 11.2 Trainer

新增：

```python
train_feasibility_step()
train_dual_policy_step()
```

训练循环中加入：

```python
if replay.ready():
    if should_update_world_model:
        train_world_model_step(...)

    if should_update_feasibility:
        train_feasibility_step(...)

    if should_update_dual:
        train_dual_policy_step(...)

    if should_update_agent:
        train_agent_step(...)
```

采样阶段加入 dual action 分支：

```python
if use_dual:
    action = dual_policy.sample(feat)
else:
    action = agent.sample(feat)
```

---

### 11.3 World Model

第一版尽量不改。

只需要提供一个 helper：

```python
world_model.encode_sequence_to_feat(obs, action, is_first)
```

用于 Gp/Gd 训练时提取 posterior latent。

伪代码：

```python
@torch.no_grad()
def encode_sequence_to_feat(self, obs, action, is_first):
    embed = self.encoder(obs)
    post, prior, stoch, deter = self.dynamic.parallel_observe(
        embed,
        action,
        is_first,
    )
    feat = torch.cat([deter, stoch], dim=-1)
    return feat
```

---

### 11.4 Feasibility Module

新增文件建议：

```text
pwm_isaaclab/modules/feasibility.py
```

包含：

```python
LatentFeasibilityCritic
LatentFeasibilityModule
update_gp_gd()
soft_update_targets()
```

---

### 11.5 Dual Policy

新增文件建议：

```text
pwm_isaaclab/modules/dual_policy.py
```

或者复用 `ActorCriticAgent.actor` 的 actor 网络结构。

需要函数：

```python
dual_policy.sample(feat)
dual_policy.distribution(feat)
dual_policy.log_prob(feat, action)
```

---

### 11.6 ActorCriticAgent.update()

推荐最小侵入式改法：

```python
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
```

原逻辑：

```python
norm_adv = ...
```

新增：

```python
if advantage_modifier_fn is not None:
    norm_adv = advantage_modifier_fn(
        feat=feat[:, :-1],
        action=action,
        norm_adv=norm_adv,
        weight=weight,
        logger=logger,
        step=step,
    )
```

然后保持：

```python
policy_loss = mean(log_prob * norm_adv * weight)
```

这样可以保证：

```text
默认 Dreamer 行为完全不变。
启用 Dreamer-FDPI 时，只改 advantage。
```

---

## 12. 监控指标

### 12.1 Replay 指标

```text
Replay/source_dual_ratio
Replay/source_main_ratio
Replay/main_cost_ratio
Replay/dual_cost_ratio
Replay/violation_ratio
```

### 12.2 World Model 指标

```text
WorldModel/recon_loss
WorldModel/reward_loss
WorldModel/dyn_loss
WorldModel/rep_loss
WorldModel/real_kl
WorldModel/vae_ent
```

### 12.3 Feasibility 指标

```text
Feasibility/gp_loss
Feasibility/gd_loss
Feasibility/gp_mean
Feasibility/gd_mean
Feasibility/gp_max
Feasibility/gp_min
Feasibility/gp_pos_mean
Feasibility/gp_neg_mean
Feasibility/feasible_ratio
Feasibility/critical_ratio
Feasibility/infeasible_ratio
```

### 12.4 Dual 指标

```text
Dual/loss
Dual/gd_objective
Dual/kl_dual_main
Dual/kl_main_dual
Dual/lambda_dual_main
Dual/lambda_main_dual
Dual/action_norm
Dual/entropy
```

### 12.5 Main Actor 指标

```text
ActorCritic/norm_adv
ActorCritic/safe_adv
ActorCritic/risk_penalty
ActorCritic/entropy
ActorCritic/action_norm
ActorCritic/feasible_ratio
ActorCritic/critical_ratio
ActorCritic/infeasible_ratio
```

### 12.6 Rollout 指标

```text
episode_return
episode_cost
success_rate
distance_to_object
grasp_success
main_source_return
dual_source_return
main_source_cost
dual_source_cost
```

---

## 13. 崩塌诊断表

| 现象 | 可能原因 | 优先处理 |
| --- | --- | --- |
| main policy 不靠近目标 | risk lambda 太大 / Gp 太悲观 | 降低 lambda，延后 risk 启用 |
| success 掉崖 | dual 数据污染 world model | 降低 dual_ratio，限制 world model dual fraction |
| Gp 全接近 1 | Gp 过悲观 / violation label 太密 | 调大 pf，检查 cost 提取 |
| Gp 全接近 0 | violation 样本太少 | 开启 dual，增加少量 violation 数据 |
| dual KL 爆炸 | dual policy 跑太远 | 增大 KL lambda，暂停 dual sampling |
| entropy 快速下降 | actor 被风险项压死 | 增大 entropy_coef，降低 risk lambda |
| world model reward_loss 升高 | replay 分布变坏 | 降低 dual_ratio，检查 dual source return |
| infeasible_ratio 接近 1 | Gp 阈值太低 / cg 太大 | 调高 pf 或减小 lambda |
| feasible_ratio 长期接近 1 | Gp 没起作用 | 检查 Gp 训练和 cost label |

---

## 14. 最小可运行版本

为了尽快验证，建议第一版只实现以下内容：

```text
1. replay 增加 cost/source；
2. 新增 Gp/Gd latent critic；
3. 新增 dual actor；
4. dual 低比例采样；
5. Gp/Gd 训练；
6. ActorCriticAgent.update() 增加 advantage_modifier_fn；
7. 用 risk-conditioned advantage 替换 norm_adv；
8. 不使用 direct Gp gradient；
9. 不使用 GR；
10. 不使用完整 IS。
```

最小版 actor 修改：

```python
def risk_advantage_modifier(feat, action, norm_adv, weight):
    with torch.no_grad():
        g = torch.maximum(
            feasibility.gp1(feat, action),
            feasibility.gp2(feat, action),
        )

        fea = g < pf - cg
        cri = (g >= pf - cg) & (g < pf)
        inf = g >= pf

        risk_margin = F.relu(g - (pf - cg)) / (cg + 1e-6)
        risk_excess = F.relu(g - pf) / (1.0 - pf + 1e-6)

        safe_adv = torch.zeros_like(norm_adv)
        safe_adv[fea] = norm_adv[fea]
        safe_adv[cri] = norm_adv[cri] - lambda_cri * risk_margin[cri]
        safe_adv[inf] = torch.minimum(
            norm_adv[inf],
            torch.zeros_like(norm_adv[inf]),
        ) - lambda_inf * risk_excess[inf]

        safe_adv = safe_adv.clamp(-5.0, 5.0)

    return safe_adv
```

---

## 15. 最终推荐结论

推荐的新结合方案不是：

```text
Dreamer joint update + FDPI actor-only update
```

也不是：

```text
在 Dreamer 后面再 residual 更新一次 actor
```

也不是：

```text
完全复刻 SAC-FDPI 的 Q/G/GR/pi/dual_pi 更新
```

而是：

```text
DreamerV3 主干
+ dual policy 安全边界探索
+ latent Gp/Gd 风险估计
+ risk-conditioned advantage actor update
```

其核心优势是：

```text
1. 保留 DreamerV3 原本稳定的 actor-critic joint update；
2. 利用 dual policy 增加 violation / boundary samples；
3. 让 Gp/Gd 学到更准确的安全边界；
4. main actor 不被额外 actor-only update 干扰；
5. 第一版只调制 advantage，避免 Gp 直接把 actor 推崩；
6. 后续可以逐步扩展 direct Gp gradient、GR recovery branch、IS 修正。
```

一句话总结：

```text
把 FDPI 改造成 Dreamer 的安全数据增强与风险调制模块，而不是把 FDPI 作为第二套 actor-critic 强行接到 Dreamer 后面。
```
