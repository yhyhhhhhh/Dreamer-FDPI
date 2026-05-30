# DFDv2 对偶策略设计方案：Dual Policy for World-Model Safety Learning

## 0. 核心结论

本方案中的对偶策略（dual policy）不是主任务策略，也不是直接用于优化主策略的安全 actor。它的核心角色是：

```text
受 KL 约束的高 bottom-force 风险数据采集器。
```

它服务于 Dreamer/PSSM 世界模型的安全建模：

```text
dual policy 主动采集主策略较少覆盖的高 cost / 非安全状态动作数据
→ replay 中增加 bottom-force 相关危险样本
→ world model 更好地学习 continuous cost / force dynamics
→ main policy 在 imagined rollout 中通过 cost-aware reward 学习安全夹取
```

因此，本方案不照搬原 FDPI 中“主策略足够安全后才激活 dual”的条件，而采用：

```text
model/Gd ready 后低比例主动采集
+ 根据 replay 中危险/边界数据覆盖动态调整采样预算
+ world model 训练时对 dual/high-cost 数据进行 source/cost-aware sampling
```

---

## 1. 对偶策略到底学出来什么

如果 dual policy 用 `Gd` 或 risk return 更新，它学到的不是安全策略，而是：

```text
在 main/expert 策略附近寻找更容易产生 bottom-force risk 的动作策略。
```

更具体地说，在夹取任务中，dual policy 可能倾向于：

```text
1. 更容易向管道底部压的动作；
2. 更激进的夹爪闭合动作；
3. 更容易产生底部接触的姿态或接近动作；
4. 在成功夹取轨迹附近制造更高 bottom-force 的动作扰动。
```

如果 KL 约束合理，dual 会成为：

```text
任务分布附近的危险扰动策略。
```

如果 KL 太弱，它会退化为：

```text
极端撞底 / 无意义危险策略。
```

所以 dual policy 的设计重点不是“越危险越好”，而是：

```text
在 main/expert 附近产生对 world model 有信息量的 high-cost / unsafe transition。
```

---

## 2. Dual policy 的总体使用逻辑

完整闭环：

```text
main/expert policy 采集正常任务数据和自然危险数据
        ↓
Gd 用真实 replay 进行 risk pretraining
        ↓
world model 学习 dynamics / reward / continuous cost / force
        ↓
dual policy 在 world model imagination 中学习高风险轨迹
        ↓
dual policy 以低比例进入真实环境采样
        ↓
真实 dual 数据进入 replay
        ↓
world model 通过 source/cost-aware sampling 学到更多 unsafe dynamics
        ↓
main policy 使用 task_reward - λ continuous_cost 学习安全夹取
```

关键原则：

```text
1. dual imagined rollout 不直接进入 replay；
2. world model 只用真实 replay 训练；
3. Gd 只用真实 replay 更新；
4. dual policy 可以在 imagination 中训练；
5. main policy 不直接由 Gd/Gp 更新；
6. main policy 通过 cost-aware imagined reward 学安全。
```

---

## 3. 对偶策略什么时候开始训练

需要区分两件事：

```text
1. dual imagination training：在模型中训练 dual；
2. dual real sampling：用 dual 真实环境采样。
```

两者不应使用完全相同的 gate。

### 3.1 Dual imagination training 的启动

dual imagination training 不直接污染 replay，因此可以比真实采样早一点启动。

推荐启动条件：

```text
step >= DualImagination.StartStep
world_model_ready = true
Gd_ready = true
```

第一版可用简单 step gate：

```yaml
DualImagination:
  StartStep: 50000
```

更稳的条件：

```text
Gd/high_cost_mean > Gd/low_cost_mean + margin
```

推荐：

```text
margin = 0.03 ~ 0.05
```

原因：

```text
如果 Gd 还不能区分 high-cost 与 low-cost，
dual 在 imagination 中最大化 risk 只是在追 critic 噪声。
```

### 3.2 Dual real sampling 的启动

真实采样更谨慎。推荐条件：

```text
step >= DualSampling.StartStep
model_ready = true
Gd_ready = true
KL(πd || πmain/expert) < MaxKLForSampling
dual_ratio_budget_available = true
```

不建议只用 KL。KL 只能说明 dual 没有偏离 main/expert 太远，不能说明 dual 是否已经有必要采样。

本方案也不要求必须等 main policy 完全安全后才启用 dual，因为当前目标是 world model 的安全建模，而不是 FDPI 原始的 off-policy actor 更新。

更准确的逻辑是：

```text
早期 main/expert 本身会产生大量 bottom-force risk：
    dual 真实采样可以关闭或极低比例；
    用自然危险数据训练 Gd 和 world model。

model/Gd ready 后：
    低比例启用 dual 真实采样。

main policy 逐渐安全后：
    如果 high-cost / boundary 数据减少，则提高 dual ratio，维护安全关键数据覆盖。
```

---

## 4. 对偶策略采样比例如何设置

不要固定一个高比例。推荐使用 **budgeted dual sampling**。

### 4.1 推荐比例

```text
阶段 0：0
阶段 1：0.5% ~ 1%
阶段 2：1% ~ 3%
最大上限：5%
```

不建议第一版超过 5%，因为 dual 数据太多会使 replay/world model 分布偏向极端危险行为。

### 4.2 推荐配置

```yaml
DualSampling:
  Enable: true
  StartStep: 100000
  RatioStart: 0.005
  RatioFinal: 0.03
  RatioWarmupSteps: 100000
  MaxRatio: 0.05
  MaxKLForSampling: 2.0
```

### 4.3 动态 ratio 逻辑

推荐：

```python
if not model_ready or not gd_ready or not kl_healthy:
    dual_ratio = 0.0

elif main_cost_rate_recent > high_cost_rate:
    # main 自己已经产生很多危险数据，不需要大量 dual
    dual_ratio = 0.005 ~ 0.01

elif high_cost_or_boundary_coverage_insufficient:
    # replay 中危险/边界数据不足
    dual_ratio = 0.02 ~ 0.03

else:
    dual_ratio = 0.01 ~ 0.02
```

其中：

```text
main_cost_rate_recent:
    最近 main-source transition 中 binary_cost 或 high continuous_cost 的比例。

boundary_coverage:
    continuous_cost 落在边界区间内的样本比例。
```

示例：

```text
boundary sample:
    BoundaryLow < continuous_cost < BoundaryHigh
```

推荐：

```yaml
BoundaryLow: 0.05
BoundaryHigh: 0.40
BoundaryRatioTarget: 0.10
```

---

## 5. Gd 的定义与阶段性语义

### 5.1 最终定义

```text
Gd(z, a):
    在 latent z 执行动作 a 后，
    后续由 dual policy 继续控制时，
    未来 bottom-force continuous risk 的期望累计值。
```

`Gd` 是状态-动作风险 critic：

```text
Gd: (z, a) → future continuous bottom-force risk
```

### 5.2 阶段性语义

由于训练早期 dual 真实采样少，Gd 的语义需要分阶段理解：

```text
早期：
    Gd 是由 main/expert 数据支持的 action-conditioned bottom-force risk critic。
    它主要学习哪些动作会导致 bottom-force risk。

中期：
    随着 dual imagination 和少量 dual real sampling，Gd 开始服务 dual policy。

后期：
    随着 dual-source 样本积累，Gd 更接近 FDPI-style dual-continuation risk critic。
```

不要在设计文档中声称早期 Gd 已经严格等价于原 FDPI 的 Gd。更准确的描述是：

```text
Gd starts as a main/expert-data-supported action-risk critic and gradually becomes a dual-continuation risk critic as dual samples accumulate.
```

---

## 6. Gd 如何更新

### 6.1 数据来源

Gd 只使用真实 replay 更新，不使用 imagined data 更新。

每个样本需要：

```text
z_t
action_t
continuous_cost_t
done_t
z_{t+1}
source_t
```

必须保证时间对齐：

```text
z_t + action_t → continuous_cost_t + z_{t+1}
```

### 6.2 Continuous cost

本方案使用连续 bottom-force cost，而不是只用 binary cost：

```text
bottom_force_t = selected bottom force magnitude
force_excess_t = relu(bottom_force_t - force_threshold)
continuous_cost_t = clamp(force_excess_t / force_scale, 0, 1)
```

binary cost 可以保留用于日志：

```text
binary_cost_t = bottom_force_t > force_threshold
```

### 6.3 TD target

因为 cost 是连续值，不应使用 binary FDPI 中的 `(1 - cost)` 截断项。

推荐 target：

```python
with torch.no_grad():
    a_next_dual = dual_policy.sample(z_next)

    target_gd = torch.minimum(
        target_gd1(z_next, a_next_dual),
        target_gd2(z_next, a_next_dual),
    )

    y_gd = continuous_cost + (1.0 - done) * gamma_cost * target_gd
    y_gd = torch.clamp(y_gd, 0.0, risk_max)
```

### 6.4 Gd loss

```python
gd_loss =
    mean(weight * (gd1(z, action) - y_gd) ** 2)
  + mean(weight * (gd2(z, action) - y_gd) ** 2)
```

每次更新后 soft update target networks：

```text
target_gd ← τ gd + (1 - τ) target_gd
```

### 6.5 是否使用 importance sampling

第一版不建议使用完整 FDPI-style importance sampling。

原因：

```text
1. 早期 dual 数据少，IS 不能解决 support 缺失；
2. main/expert 早期自然危险数据对 Gd 预训练很有价值；
3. 连续动作下 IS 方差大；
4. 本方法的目标是 world-model safety learning，不是严格复刻 FDPI 的 dual value evaluation。
```

推荐第一版使用：

```text
source/cost-aware sampling 或 weighting
```

而不是完整 IS。

---

## 7. Gd 的 source/cost-aware weighting

最低限度：

```python
weight = 1.0

if source == DUAL:
    weight *= DualSourceWeight

if continuous_cost > HighCostThreshold:
    weight *= HighCostWeight

if BoundaryLow < continuous_cost < BoundaryHigh:
    weight *= BoundaryWeight
```

推荐配置：

```yaml
Gd:
  GammaCost: 0.97
  RiskMax: 1.0
  TargetTau: 0.005
  DualSourceWeight: 2.0
  HighCostWeight: 3.0
  BoundaryWeight: 2.0
  HighCostThreshold: 0.1
```

更稳的方式是 Gd batch quota：

```text
40% uniform samples
30% high-cost samples
20% dual-source samples
10% recent samples
```

目的：

```text
避免 Gd 被不断增长的 replay 中大量 low-cost main samples 淹没。
```

### 7.1 后续 clipped IS ablation

如果后续要测试 IS，应满足：

```text
1. dual-source 样本已有足够数量；
2. replay 存储 behavior_logp；
3. KL(dual || main) 不大；
4. Gd loss 稳定；
5. 使用 clipped / normalized IS。
```

形式：

```python
rho = exp(logp_dual_current - logp_behavior)
rho = clamp(rho, 0.1, 10.0)
rho = rho / (rho.mean().detach() + 1e-6)

weight = source_cost_weight * rho.detach()
```

IS 是 ablation，不是第一版主线。

---

## 8. Dual policy 在 imagination 中如何更新

### 8.1 不推荐单步 `-Gd(z,a)` 作为最终主方案

单步最大化 Gd：

```text
dual_loss = -Gd(z, a_dual) + KL
```

虽然简单，但更像 SAC/FDPI 的单步 action-gradient，没有充分利用 world model 的多步 imagined rollout 能力。

### 8.2 推荐：Dreamer-style dual risk imagination update

从真实 posterior latent 起点开始：

```text
z0 ~ posterior latent from real replay
```

在 frozen world model 中用 dual policy rollout：

```text
z0 --a0^d--> z1 --a1^d--> ... --aH^d--> zH
```

每一步记录：

```text
predicted continuous cost c_hat_t
logπd(a_t | z_t)
KL(πd || πmain/expert)
entropy
```

末端使用 Gd bootstrap：

```text
terminal_risk = Gd(z_H, a_H^d)
```

### 8.3 Risk return

不使用纯 5-step MC return，也不只用单步 Gd。

推荐 bootstrapped risk return：

```text
R_t^risk =
    sum_{k=t}^{H-1} gamma_cost^(k-t) * c_hat_k
    + gamma_cost^(H-t) * terminal_Gd
```

更 Dreamer-style 的 lambda 版本：

```text
R_t^risk =
    c_hat_t
    + gamma_cost * [
        (1 - lambda_cost) * Gd(z_{t+1}, a_{t+1})
        + lambda_cost * R_{t+1}^risk
      ]
```

第一版可以使用简单 bootstrapped risk return，避免额外 complexity。

### 8.4 Dual actor loss

```python
risk_adv = normalize(R_risk)

dual_pg_loss = -mean(log_prob_dual * risk_adv.detach())

kl_loss = mean(KL(πd || πmain_or_expert))

entropy_loss = mean(entropy_dual)

dual_loss =
    dual_pg_loss
    + beta_kl * kl_loss
    - beta_entropy * entropy_loss
```

语义：

```text
如果某个 dual action 在 imagined trajectory 中导致更高 future bottom-force risk，
dual policy 以后更倾向于采这个动作；
但 KL 约束它不要偏离 main/expert 太远。
```

### 8.5 梯度边界

dual update 中：

```text
只更新：
    dual_policy

不更新：
    world_model
    main_policy
    Gd
```

第一版建议：

```text
world model 用 frozen/no_grad rollout；
Gd 作为 terminal/bootstrap evaluator；
dual optimizer 只 step dual_policy。
```

---

## 9. Dual real sampling

真实环境采样逻辑：

```python
if dual_enabled and random() < dual_ratio and kl_healthy:
    action = dual_policy(z)
    source = DUAL
else:
    action = main_policy(z)
    source = MAIN
```

保存到 replay：

```text
obs
action
reward
continuous_cost
binary_cost
bottom_force
force_excess
done
is_first
source
behavior_logp   # 可选，为后续 IS ablation 准备
```

---

## 10. World model 如何真正利用 dual 数据

因为 replay buffer 不断增长，dual 数据会被 uniform sampling 稀释。因此，只增加 dual 真实采样还不够。

必须让 world model update 使用 source/cost-aware sampling。

推荐：

```yaml
WorldModelSampling:
  EnableCostAwareSampling: true
  UniformRatio: 0.80
  SafetyCriticalRatio: 0.20
```

其中 safety-critical samples 包括：

```text
dual-source samples
high-cost samples
boundary-cost samples
recent samples
```

示例：

```text
80% uniform
10% high-cost
5% dual-source
5% boundary/recent
```

如果担心影响 dynamics/reward，可以分开：

```text
normal world model update:
    原始 uniform batch

extra cost/force update:
    high-cost + dual + boundary batch
```

关键是必须记录 batch composition：

```text
WorldModelBatch/source_dual_ratio
WorldModelBatch/high_cost_ratio
WorldModelBatch/boundary_ratio
```

否则 dual 采到的数据可能很少被 world model 训练到。

---

## 11. Main policy 如何学习安全

dual 不直接训练 main policy。  
main policy 必须通过 cost-aware reward 接收安全信号：

```python
safe_reward = task_reward - lambda_cost * predicted_continuous_cost
```

然后仍然走 Dreamer 原始 actor-critic 更新：

```text
safe_reward
→ lambda return
→ advantage
→ log_prob * advantage
```

也就是说：

```text
不改 main actor update 公式；
只改 main policy 的优化目标。
```

如果 main policy 的 reward 不包含 cost，那么 dual 只会让 world model 更懂危险，不会让 main policy 自动变安全。

---

## 12. 推荐训练阶段

### 阶段 A：自然危险预训练

```text
dual real sampling = 0
dual imagination training = off
Gd update = on
world model cost/force update = on
```

目标：

```text
利用 main/expert 自然产生的 bottom-force 数据预训练 cost model 和 Gd。
```

### 阶段 B：dual imagination training

```text
dual real sampling = 0 或极低
dual imagination training = on
```

条件：

```text
step >= 50k/100k
Gd 有 high/low cost separation
world model cost prediction 非全 0/全 1
```

目标：

```text
让 dual policy 在模型中学会产生 high-risk trajectory。
```

### 阶段 C：低比例 dual real sampling

```text
dual_ratio = 0.5% ~ 1%
```

条件：

```text
KL healthy
Gd ready
model ready
```

目标：

```text
验证 dual 是否真的能采到比 main 更高 cost 的真实 transition。
```

### 阶段 D：动态预算采样

```text
dual_ratio = 1% ~ 3%
cap = 5%
```

当 replay 中 high-cost/boundary 覆盖不足时提高 dual ratio；当 main 自己已经有很多危险数据时降低 dual ratio。

---

## 13. 推荐默认配置

```yaml
DFD_V2:
  Gd:
    GammaCost: 0.97
    RiskMax: 1.0
    TargetTau: 0.005
    DualSourceWeight: 2.0
    HighCostWeight: 3.0
    BoundaryWeight: 2.0
    HighCostThreshold: 0.1
    MinSeparationForDual: 0.03

  DualImagination:
    Enable: true
    StartStep: 50000
    Horizon: 5
    ReturnType: "bootstrapped_cost_gd"
    GammaCost: 0.97
    LambdaCost: 0.95
    KLCoeff: 1.0
    EntropyCoef: 1.0e-4
    GradClipNorm: 100.0

  DualSampling:
    Enable: true
    StartStep: 100000
    RatioStart: 0.005
    RatioFinal: 0.03
    RatioWarmupSteps: 100000
    MaxRatio: 0.05
    MaxKLForSampling: 2.0

  WorldModelSampling:
    EnableCostAwareSampling: true
    UniformRatio: 0.80
    SafetyCriticalRatio: 0.20
    HighCostRatio: 0.10
    DualRatio: 0.05
    BoundaryRecentRatio: 0.05

  MainCostAwareReward:
    Enable: true
    LambdaCost: 0.03
```

---

## 14. 必要日志

### Dual

```text
Dual/ratio
Dual/kl_to_main
Dual/entropy
Dual/real_cost_mean
Dual/real_cost_rate
Dual/source_count
```

### Gd

```text
Gd/loss
Gd/high_cost_mean
Gd/low_cost_mean
Gd/separation
Gd/target_mean
Gd/source_dual_loss
Gd/source_main_loss
```

### Dual imagination

```text
DualImag/risk_return_mean
DualImag/pred_cost_mean
DualImag/terminal_gd_mean
DualImag/kl
DualImag/loss
```

### Replay / world model sampling

```text
Replay/source_dual_ratio
Replay/high_cost_ratio
Replay/boundary_ratio
WorldModelBatch/source_dual_ratio
WorldModelBatch/high_cost_ratio
WorldModelBatch/boundary_ratio
```

### Main safety

```text
Main/task_return
Main/safe_return
Main/bottom_force_mean
Main/bottom_force_peak
Main/continuous_cost_mean
Main/success_rate
```

---

## 15. 有效性判断

dual policy 有效应表现为：

```text
1. Dual/real_cost_mean > Main/real_cost_mean；
2. Dual KL 不爆炸；
3. Gd 能区分 high-cost 和 low-cost；
4. world model cost/force prediction 在 high-cost/dual holdout 上变好；
5. world model batch 中能看到足够 dual/high-cost samples；
6. main policy success rate 不显著下降；
7. main bottom_force_mean / peak 逐渐下降。
```

如果出现：

```text
DualImag risk 很高，但真实 Dual/real_cost 不高
```

说明 dual exploit world model 或 Gd。

如果出现：

```text
Dual/real_cost 很高，但 world model cost loss 不下降
```

说明 dual 数据没有被 world model 有效利用，可能被 replay 稀释。

如果出现：

```text
world model cost 学好了，但 main bottom force 不下降
```

说明 main cost-aware reward 没有正确进入 actor/critic return。

---

## 16. 最终一句话总结

本方案中，dual policy 应该被设计为：

```text
一个 KL-constrained adversarial safety data collector。
```

它先依靠 main/expert 的自然危险数据预训练 Gd，再在 world model 中通过 predicted continuous cost return + terminal Gd 学习 high-risk trajectory，随后以低比例真实采样补充非安全状态动作数据。Gd 不建议第一版使用完整 IS，而应使用 source/cost-aware sampling 或 weighting。采到的 dual/high-cost 数据必须通过 world model 的 source/cost-aware batch sampling 被充分利用，最终 main policy 通过 cost-aware Dreamer reward 学习低冲击夹取。
