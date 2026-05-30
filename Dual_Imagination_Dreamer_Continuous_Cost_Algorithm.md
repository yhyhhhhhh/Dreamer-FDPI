# Dual-Imagination Dreamer with Continuous-Cost FDPI-Style Exploration

## 0. 核心定位

本方案的目标不是完整复现 SAC-FDPI，也不是把 FDPI 的 main actor loss 原样迁移到 Dreamer 中。

本方案的核心定位是：

```text
DreamerV3 / PSSM 仍然是主算法；
main policy 仍然使用 Dreamer 的世界模型想象 + actor-critic 更新；
dual policy 作为危险 / 边界数据采集器；
Gd 作为 dual policy 的状态-动作连续风险 critic；
dual policy 在 world model imagination 中训练；
dual policy 以低比例进入真实环境采样；
真实 dual 数据进入 replay；
world model 用真实 replay 学习更准确的 continuous cost / force model；
main policy 在 cost-aware imagined reward 下学习安全夹取。
```

一句话概括：

```text
Dual policy 不是直接教 main policy 怎么夹取，
而是主动为 world model 补充 bottom-force 危险数据；
main policy 再通过更准确的 cost-aware world model 学会低冲击夹取。
```

---

## 1. 任务背景与问题定义

当前任务是接触丰富的夹取任务。主策略需要完成夹取，但夹取过程很容易与管道底部发生冲击，导致 bottom force 过大。

关键特点：

```text
1. 夹取动作是任务必要动作；
2. 接触本身不可避免；
3. 危险不是简单的“接触/不接触”，而是 bottom force 的大小；
4. 目标不是完全避免夹取接触，而是减小与底部的碰撞力；
5. 早期主策略或专家策略可能本身就会产生大量 bottom-force 风险样本；
6. 随着策略变安全，危险样本可能减少，world model 的 cost/force 边界建模可能退化。
```

因此，本方案不使用简单的：

```text
高风险动作 → 降低 main policy 采样概率
```

因为这可能压制夹取动作本身。

本方案采用：

```text
dual policy 负责采危险 / 边界数据；
world model 学连续 cost / force；
main policy 通过 cost-aware reward 学安全夹取。
```

---

## 2. 与原 FDPI 的关系

原 FDPI 的核心思想包括：

```text
1. primal/main policy 逐渐变安全；
2. violation samples 变少；
3. feasibility function 可能因为缺少危险样本而学不准；
4. dual policy 主动采集危险样本；
5. Gd 作为 dual policy continuation risk critic；
6. dual policy 最大化 Gd，并通过 KL 限制不要偏离 main policy 太远。
```

本方案继承 FDPI 的以下思想：

```text
1. dual policy 用于主动采集危险 / 边界样本；
2. Gd 是 dual policy 的状态-动作风险 critic；
3. dual policy 通过最大化 Gd 学习走向危险区域；
4. dual policy 需要 KL 约束，避免远离 main/expert 分布；
5. 当 main policy 变安全后，dual policy 继续补充危险数据，避免安全相关模型缺少危险样本。
```

但本方案与原 FDPI 不同：

```text
1. 不迁移 FDPI 的 main actor segmented loss；
2. 不使用 Gp 来直接惩罚 main policy；
3. 不做 main actor 的额外 actor-only update；
4. 不复刻 SAC-style reward Q；
5. dual policy 的主要服务对象从 feasibility function 扩展为 Dreamer world model；
6. main policy 通过 cost-aware imagined reward 学安全，而不是直接由 Gp 更新。
```

本方案可以称为：

```text
FDPI-inspired Dual-Imagination Dreamer
```

---

## 3. 总体算法闭环

完整闭环如下：

```text
main/expert policy 与少量 dual policy 在真实环境采样
        ↓
replay buffer 存储 obs, action, reward, continuous_cost, done, source
        ↓
world model 用真实 replay 学 dynamics / reward / continuous cost / force
        ↓
Gd 用真实 replay posterior latent 更新，学习 dual continuation bottom-force risk
        ↓
dual policy 从真实 posterior latent 出发，在 world model imagination 中训练
        ↓
dual policy 学会寻找 high-cost / dangerous bottom-force 动作
        ↓
dual policy 以低比例进入真实环境采样
        ↓
真实 dual 数据补充 replay 中的危险 / 边界样本
        ↓
world model cost/force prediction 更准确
        ↓
main policy 在 imagined rollout 中用 cost-aware reward 学低冲击夹取
```

其中最重要的原则是：

```text
world model 只用真实 replay 训练；
dual imagined data 不直接训练 world model。
```

---

## 4. 核心变量定义

### 4.1 Latent state

```text
z_t = Dreamer/PSSM latent feature
```

通常由 posterior state 得到：

```text
z_t = concat(deter_t, stoch_t)
```

训练 Gd 时使用真实 replay posterior latent。  
训练 dual imagination 时，从真实 posterior latent 作为起点。

---

### 4.2 Continuous cost

本方案不建议只使用 binary cost。

推荐使用连续 cost：

```text
c_t = normalized bottom-force excess
```

例如：

```text
bottom_force_t = selected bottom force magnitude
force_excess_t = relu(bottom_force_t - force_safe_threshold)
c_t = normalize_or_clip(force_excess_t)
```

可选定义：

```text
c_t ∈ [0, 1]
```

例如：

```text
c_t = clamp(force_excess_t / force_scale, 0, 1)
```

原因：

```text
binary cost 只能表示是否超阈值；
continuous cost 能区分轻微碰底和猛烈撞底；
你的目标是减小碰撞力，而不只是避免二值 violation。
```

---

### 4.3 Dual risk critic Gd

保留 FDPI-style Gd 定义：

```text
Gd(z, a):
    在 latent z 下执行动作 a 后，
    后续继续由 dual policy 控制时，
    未来 bottom-force risk 的期望累计值。
```

它是状态-动作风险 critic：

```text
Gd: (z, a) → continuous risk value
```

注意：这里的 risk 使用 continuous cost，而不是 binary violation。

---

### 4.4 Dual policy

```text
πd(a | z)
```

定义：

```text
dual policy 是危险 / 边界数据采集策略。
它不负责完成主任务；
它负责在 main/expert 分布附近寻找更高 bottom-force risk 的动作，
从而帮助 world model 学习危险区域。
```

---

### 4.5 Main policy

```text
πm(a | z)
```

定义：

```text
main policy 是实际任务策略。
它仍然使用 Dreamer 原始 actor-critic 更新形式。
本方案不使用 Gp 或 Gd 直接更新 main actor。
main policy 通过 cost-aware imagined reward 学安全夹取。
```

---

## 5. 模块设计

### 5.1 Replay Buffer

replay 至少需要存储：

```text
obs
action
reward
continuous_cost
done
is_first
force
source
```

其中：

```text
source = MAIN / DUAL / RANDOM
```

推荐额外记录：

```text
bottom_force
force_excess
binary_cost
```

原因：

```text
continuous_cost 用于学习减小 bottom force；
binary_cost 用于日志和安全事件统计；
source 用于分析 dual 是否真的增加危险/边界样本。
```

---

### 5.2 World Model

world model 继续使用 Dreamer / PSSM 主体。

需要预测：

```text
dynamics
reward
discount / done
force or continuous_cost
```

推荐优先训练 force / cost prediction：

```text
force_head(z) → predicted bottom force or force vector
cost_head(z)  → predicted continuous cost
```

如果已有 force head，则可由 force head 转换得到 continuous cost：

```text
pred_bottom_force
→ pred_force_excess
→ pred_continuous_cost
```

world model loss 中可以包含：

```text
reward loss
dynamics / representation loss
discount loss
force / cost prediction loss
```

注意：

```text
world model 不使用 imagined dual data 训练；
只使用真实 replay 数据训练。
```

---

### 5.3 Gd Module

使用 double critic：

```text
gd1(z, a)
gd2(z, a)
target_gd1(z, a)
target_gd2(z, a)
```

输出 continuous risk，可以选择：

```text
raw risk + clamp target
```

或：

```text
sigmoid risk in [0,1]
```

若 continuous cost 已归一化到 `[0,1]`，建议 Gd 输出也控制在相同范围或进行合理 clip。

---

### 5.4 Dual Policy

dual policy 可以复用 main actor 的网络结构，但独立参数。

必须支持：

```text
distribution(z)
rsample(z)
log_prob(z, action)
entropy(z)
```

注意：

```text
dual update 中 action 必须可重参数化；
否则 Gd 的 action-gradient 无法回到 dual policy。
```

---

## 6. Gd 的真实 replay 更新

Gd 应该用真实 replay 更新，而不是 imagined data。

### 6.1 数据准备

从 replay sample 一个 sequence batch：

```text
obs_seq, action_seq, continuous_cost_seq, done_seq, is_first_seq
```

用 world model posterior encode：

```text
z_t, z_{t+1}
```

时间对齐必须满足：

```text
z_t + action_t → cost_t + z_{t+1}
```

---

### 6.2 Continuous TD Target

因为 cost 是连续值，推荐 target：

```text
y_t = c_t + (1 - done_t) * gamma_c * target_Gd(z_{t+1}, a_{t+1}^d)
```

其中：

```text
a_{t+1}^d ~ πd(. | z_{t+1})
target_Gd = min(target_gd1, target_gd2)
```

伪公式：

```python
with no_grad:
    a_next_dual = dual_policy.sample(z_next)

    target_risk = min(
        target_gd1(z_next, a_next_dual),
        target_gd2(z_next, a_next_dual),
    )

    y_gd = continuous_cost + (1 - done) * gamma_c * target_risk
    y_gd = clamp(y_gd, 0, risk_max)
```

与 binary cost 版本不同，连续 cost 不建议使用：

```text
(1 - cost_t)
```

去截断未来项。因为 cost_t 不再是 absorbing violation indicator，而是 force magnitude / risk signal。

---

### 6.3 Gd Loss

```text
L_gd =
    E[ w * (gd1(z,a) - y_gd)^2 ]
  + E[ w * (gd2(z,a) - y_gd)^2 ]
```

其中 `w` 是可选的 source/risk-aware weight。

第一版不强制使用完整 FDPI importance sampling，但建议使用简单 weighting：

```text
dual source 样本权重更高；
high-cost 样本权重更高；
普通 main-safe 样本权重较低。
```

示例逻辑：

```text
w = 1
if source == DUAL: w *= w_dual_source
if continuous_cost > cost_threshold: w *= w_high_cost
```

目的：

```text
避免 Gd 被大量低 cost / 安全样本淹没。
```

---

### 6.4 Target Network Update

每次 Gd update 后 soft update：

```text
target_gd ← tau * gd + (1 - tau) * target_gd
```

---

## 7. Dual Policy 在 Imagination 中训练

### 7.1 为什么在 imagination 中训练 dual

dual policy 的目标是帮助 world model 采集危险 / 边界数据。

如果只在真实 replay posterior state 上单步更新，dual 只能在已经访问过的状态上学危险动作。

在 imagination 中训练 dual 可以让它从真实 posterior latent 出发，探索 world model 预测的附近状态分布：

```text
真实 posterior z0
→ imagined z1, z2, ..., zH
```

这样 dual 能学到：

```text
如何从任务分布附近逐步走向 bottom-force 高风险区域。
```

---

### 7.2 不使用 Monte Carlo Return

本方案明确不使用短 horizon MC cost return 作为 dual policy 的主要目标。

原因：

```text
1. dual imagination horizon 通常很短，如 H=5；
2. 5-step MC return 截断偏差大；
3. imagined cost 受 world model 误差影响；
4. dual 容易 exploit world model 的假风险。
```

本方案采用：

```text
imagined latent state augmentation + Gd-based policy improvement
```

也就是：

```text
world model imagination 只生成 dual 训练用的 imagined states；
Gd 负责评价这些 states 上的动作风险。
```

---

### 7.3 Dual Imagination Rollout

从真实 posterior latent 起点开始：

```text
z0 ~ posterior latent from real replay
```

冻结 world model，使用 dual policy rollout H 步：

```text
z0 --a0^d--> z1 --a1^d--> z2 ... zH
```

要求：

```text
1. rollout 起点必须来自真实 posterior latent；
2. 不从随机 latent 开始；
3. world model 参数冻结；
4. rollout 过程不反向传播进 world model；
5. imagined states detach 后用于 dual update。
```

---

### 7.4 Dual Update States

得到 imagined states：

```text
Z_imag = {z1, z2, ..., zH}
```

将这些 imagined states detach：

```text
z_train = stop_gradient(Z_imag)
```

然后在 `z_train` 上重新采样 dual action：

```text
a_d ~ πd(. | z_train)
```

并用 Gd 评价：

```text
g_d = min(gd1(z_train, a_d), gd2(z_train, a_d))
```

---

### 7.5 Dual Objective: Max-Risk

本方案采用 `max_risk`。

```text
score = Gd(z, a_d)
```

dual policy 目标：

```text
maximize Gd(z, a_d)
```

也就是：

```text
dual_loss_risk = - E[ Gd(z, a_d) ]
```

解释：

```text
dual policy 学习在 imagined latent states 上选择更容易导致 high bottom-force risk 的动作。
```

由于本任务 bottom-force 危险本身与夹取过程强相关，max-risk 可以让 dual 更主动地采集危险信息。

但必须加入 KL 和低采样比例，否则 dual 可能采集极端撞底数据。

---

### 7.6 KL to Main / Expert

为了让 dual 不偏离任务分布太远，需要 KL 约束。

推荐：

```text
KL(πd || πm)
```

即：

```text
a_d ~ πd(.|z)
KL ≈ log πd(a_d|z) - log πm(a_d|z)
```

dual loss：

```text
L_dual =
    - E[ Gd(z, a_d) ]
    + beta_kl * E[ KL(πd || πm) ]
    - beta_ent * E[ entropy(πd) ]
```

其中：

```text
πm = main policy 或 frozen expert policy
```

如果当前 main policy 是从专家初始化而来，也可以用 expert frozen copy 作为 reference：

```text
KL(πd || πexpert)
```

作用：

```text
1. dual 在任务分布附近探索危险；
2. 避免 dual 学成无意义乱撞；
3. 避免 dual 真实采样污染 replay。
```

---

### 7.7 Dual Update 的梯度路径

在 dual update 中：

```text
允许：
    dual_policy → action → Gd → loss

不允许：
    loss → world_model
    loss → main_policy
    loss → Gd parameters
```

也就是说：

```text
world_model frozen；
main_policy frozen；
Gd frozen；
只更新 dual_policy。
```

但 Gd 对 action 的梯度必须保留：

```text
∂Gd(z,a) / ∂a
```

否则 max-risk 目标无法给 dual policy 提供动作方向。

因此：

```text
z detach；
Gd 参数 requires_grad=False；
action 由 dual policy rsample 得到；
Gd 输出对 action 可导；
梯度只回到 dual policy。
```

---

## 8. Dual 真实采样

训练好的 dual policy 以低比例进入真实环境采样。

推荐：

```text
dual_ratio = 0.01 ~ 0.05
```

采样逻辑：

```text
if dual_enabled and random() < dual_ratio and dual_healthy:
    action = dual_policy(z)
    source = DUAL
else:
    action = main_policy(z)
    source = MAIN
```

真实 transition 写入 replay：

```text
obs, action, reward, continuous_cost, force, done, source
```

注意：

```text
只有真实 dual transition 进入 replay；
imagined dual rollout 不进入 replay；
world model 只用真实 replay 更新。
```

---

## 9. Dual 激活条件

本方案不简单照搬 FDPI 的 `feasible_ratio > 0.95`，因为当前任务早期 bottom-force 本来就容易出现。

更合理的 dual 激活条件包括：

```text
1. world model / force model 已经有基本预测能力；
2. Gd 已经能区分 high-cost 与 low-cost 样本；
3. main policy 或 expert policy 已经具备基本夹取能力；
4. replay 中危险/边界样本不足，或需要持续维护危险样本覆盖；
5. dual 与 main/expert 的 KL 没有爆炸。
```

第一版可用较简单的 gate：

```text
step > dual_start_step
and Gd_positive_mean > Gd_negative_mean
and KL(dual || main) < max_kl
```

更完整的 gate：

```text
dual_enabled =
    step > start_step
    and gd_ready
    and world_model_ready
    and kl_healthy
    and (
        main_cost_rate_recent < cost_rate_threshold
        or boundary_sample_ratio < boundary_ratio_threshold
    )
```

含义：

```text
dual 不是一开始无脑制造危险；
dual 在模型已有基础、且需要危险/边界补充时介入。
```

---

## 10. Main Policy 如何学习安全夹取

本方案不使用 `Gp` 直接更新 main policy。

main policy 仍然使用 Dreamer 原始 actor-critic 更新方式：

```text
world model imagination
→ reward / discount / value
→ lambda return
→ advantage
→ log_prob * advantage
```

但需要把 main policy 的 imagined reward 改成 cost-aware reward。

---

### 10.1 Cost-aware Imagined Reward

定义：

```text
r_safe = r_task - lambda_cost * c_pred
```

其中：

```text
r_task = world model predicted task reward
c_pred = world model predicted continuous bottom-force cost
```

例如：

```text
c_pred = normalized predicted force_excess
```

如果使用 force head：

```text
pred_bottom_force = force_head(z)
pred_force_excess = relu(pred_bottom_force - force_threshold)
c_pred = normalize_or_clip(pred_force_excess)
```

然后：

```text
safe_reward = task_reward - lambda_cost * c_pred
```

Dreamer actor/critic 使用 `safe_reward` 计算 return 和 advantage。

---

### 10.2 不改主策略更新公式

主策略 update 仍然是：

```text
policy_loss = log_prob(action) * advantage
```

区别是：

```text
advantage 来自 safe_reward 的 lambda-return
```

而不是原始 task_reward。

这叫：

```text
不改主策略更新方式；
只改主策略优化目标。
```

---

### 10.3 Lambda Cost

`lambda_cost` 可以固定，也可以自适应。

第一版建议固定小值：

```text
lambda_cost = 0.01 ~ 0.1
```

如果希望约束形式：

```text
expected_cost <= budget
```

可以使用 Lagrange multiplier：

```text
lambda_cost increases if real/imagined cost > budget
lambda_cost decreases if cost < budget
```

第一版可以先固定，避免引入额外不稳定性。

---

## 11. 是否使用 Gp

本方案第一版不需要用 Gp 更新 main policy。

可选保留 Gp 作为：

```text
1. main policy risk evaluator；
2. safety 日志；
3. dual 激活 gate 参考；
4. 后续安全 fine-tuning 备用。
```

但第一版不建议：

```text
1. Gp-as-advantage；
2. Gp direct actor loss；
3. FDPI segmented main actor loss。
```

原因：

```text
夹取动作本身与 bottom-force risk 高度耦合；
直接用 Gp 压 main actor 可能降低夹取动作概率，
破坏已有任务能力。
```

---

## 12. 训练阶段

### 阶段 0：基础模型与专家/主策略

目标：

```text
main policy 具备基本夹取能力；
replay 中存在自然 bottom-force 风险数据；
world model 开始学习 reward / dynamics / force。
```

配置：

```text
dual_real_sampling = off
dual_imagination_update = off
Gd_update = optional/on
main_cost_aware_reward = optional/off
```

---

### 阶段 1：训练 world model cost / force 与 Gd

目标：

```text
world model 能基本预测 bottom-force cost；
Gd 能区分 high-cost 和 low-cost transition。
```

开启：

```text
Gd replay update
world model force/cost loss
```

不开启：

```text
dual real sampling
dual imagination update
main policy Gp penalty
```

---

### 阶段 2：dual 在 imagination 中训练

目标：

```text
dual policy 在 imagined latent states 上学习选择 high-Gd 动作。
```

开启：

```text
dual imagination update
max-risk Gd objective
KL to main/expert
entropy
```

仍然保持：

```text
world model 不用 imagined data 训练
main policy update 不被 dual loss 影响
```

---

### 阶段 3：dual 低比例真实采样

目标：

```text
采集更多 bottom-force 危险 / 边界真实数据；
改善 world model 对危险区域的覆盖。
```

开启：

```text
dual real sampling
source=DUAL replay storage
```

控制：

```text
dual_ratio 低；
KL healthy；
真实 dual_cost_rate 监控；
world model loss 监控。
```

---

### 阶段 4：main policy cost-aware Dreamer 更新

目标：

```text
main policy 学习在完成夹取的同时减小 bottom-force impact。
```

开启：

```text
safe_reward = task_reward - lambda_cost * continuous_cost
```

保持：

```text
Dreamer actor update 公式不变；
不使用 Gp 直接更新 main policy。
```

---

## 13. 关键日志

### 13.1 Replay / Sampling

```text
Replay/source_dual_ratio
Replay/main_cost_mean
Replay/dual_cost_mean
Replay/main_cost_rate
Replay/force_excess_mean
Replay/force_excess_max
```

### 13.2 World Model

```text
WorldModel/reward_loss
WorldModel/dynamics_loss
WorldModel/force_loss
WorldModel/cost_loss
WorldModel/pred_bottom_force_mean
WorldModel/pred_cost_mean
```

### 13.3 Gd

```text
Gd/loss
Gd/mean
Gd/high_cost_mean
Gd/low_cost_mean
Gd/separation
Gd/target_mean
Gd/source_dual_loss
Gd/source_main_loss
```

### 13.4 Dual Imagination

```text
DualImag/loss
DualImag/gd_score
DualImag/kl_to_main
DualImag/entropy
DualImag/grad_norm
DualImag/horizon
DualImag/imagined_gd_mean
DualImag/imagined_gd_max
```

### 13.5 Main Policy Safety

```text
Main/task_return
Main/safe_return
Main/episode_cost_mean
Main/bottom_force_mean
Main/bottom_force_peak
Main/success_rate
Main/lambda_cost
```

---

## 14. 判断算法是否有效

算法有效不应该只看 return。至少要满足以下条件：

```text
1. dual policy 真实采样比例 > 0；
2. dual_cost_mean > main_cost_mean；
3. dual 采集的数据不是完全离任务分布的乱撞；
4. world model cost/force prediction loss 下降；
5. Gd 对 high-cost 和 low-cost 样本有区分度；
6. main policy success rate 不明显下降；
7. main policy bottom_force_mean / peak 下降；
8. cost-aware reward 开启后，main policy 能保持夹取能力并降低 force。
```

如果出现：

```text
DualImag/gd_score 很高，但 Replay/dual_cost_mean 不高
```

说明 dual 在 exploit Gd 或 world model。

如果出现：

```text
Replay/dual_cost_mean 很高，但 success_rate 掉崖、world model loss 恶化
```

说明 dual 太极端或采样比例过高。

---

## 15. 推荐默认配置

```yaml
DFD:
  ContinuousCost:
    Enable: true
    ForceThreshold: 1.0
    ForceScale: 5.0
    ClipCost: true
    CostMin: 0.0
    CostMax: 1.0

  Gd:
    Enable: true
    GammaCost: 0.97
    DoubleCritic: true
    TargetTau: 0.005
    SourceAwareWeight: true
    DualSourceWeight: 2.0
    HighCostWeight: 3.0

  DualImagination:
    Enable: true
    StartStep: 100000
    Horizon: 5
    Objective: "max_risk"
    KLCoeff: 1.0
    MaxKLForSampling: 2.0
    EntropyCoef: 1.0e-4
    GradClipNorm: 100.0

  DualSampling:
    Enable: true
    StartStep: 120000
    RatioStart: 0.01
    RatioFinal: 0.03
    RatioWarmupSteps: 100000
    RequireKLHealthy: true

  MainCostAwareReward:
    Enable: true
    StartStep: 150000
    LambdaCost: 0.03
    UsePredictedCost: true
```

---

## 16. 最终算法总结

本方案的算法逻辑是：

```text
1. 使用连续 bottom-force cost，而不是只用 binary violation。
2. 用真实 replay 训练 world model 的 force/cost prediction。
3. 用真实 replay posterior transition 训练 FDPI-style Gd。
4. Gd 估计 dual policy continuation 下的未来 bottom-force risk。
5. dual policy 在 world model imagination 中训练，但不使用短 horizon MC return。
6. imagination 只用于生成 dual 训练用 latent states。
7. dual policy 在 imagined states 上最大化 Gd，并受到 KL 约束。
8. dual policy 以低比例进入真实环境采样，采集危险/边界真实数据。
9. 真实 dual 数据进入 replay，帮助 world model 学危险区域。
10. main policy 不使用 Gp/Gd 直接更新。
11. main policy 仍然使用 Dreamer 原始 actor-critic 形式。
12. main policy 通过 cost-aware imagined reward 学习安全夹取。
```

最终闭环为：

```text
dual policy improves cost data coverage
→ world model learns better continuous cost dynamics
→ main policy optimizes task reward minus predicted force cost
→ main policy becomes safer
→ dual policy continues to maintain dangerous/boundary data coverage
```

这体现了 FDPI 的 dual-policy 思想，但适配到了 Dreamer 的 world-model learning 框架中。
