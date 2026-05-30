# FDPI-Regime Dreamer 技术方案

## 0. 核心定位

本方案不是完整复刻 SAC-FDPI，也不是只把 dual policy 搬进 Dreamer。更合理的结合方式是：

```text
Dreamer 负责 world model、latent imagination 和 actor-critic 学习；
FDPI 负责 Gp/Gd 风险 critic、可行域分区、dual 危险采样机制。
```

最终方案可以称为：

```text
FDPI-Regime Dreamer
```

核心思想：

```text
主策略：
    在 Gp 判定的可行域内优化任务 reward；
    在临界域内混合 reward 与 Gp；
    在不可行域内优先降低 Gp。

对偶策略：
    根据主策略的 Gp-feasible ratio 决定采样比例；
    主策略越可行，dual 比例越高；
    dual 采集危险/边界数据，帮助 world model、Gp、Gd 学到 unsafe dynamics。
```

---

## 1. 从原 FDPI 中迁移什么

原 FDPI 的关键不是简单 reward-cost 加权，而是：

```text
1. 主 reward critic Q；
2. 主 cost critic G；
3. recovery critic GR；
4. dual cost critic；
5. main policy；
6. dual policy；
7. feasible / critical / infeasible / violation 分区；
8. dual policy 主动采集高风险样本；
9. main-dual KL 约束；
10. importance sampling 修正混合 replay。
```

本方案第一版建议迁移：

```text
1. Gp：main-continuation risk critic；
2. Gd：dual-continuation risk critic；
3. Gp-based feasible / critical / infeasible 分区；
4. dual policy；
5. Gp-feasible-ratio dual activation；
6. dual-main KL 约束；
7. source/cost-aware replay sampling。
```

第一版不建议迁移：

```text
1. SAC-style reward Q；
2. GR recovery critic；
3. 完整 trajectory-level importance sampling；
4. 用 FDPI loss 完全替换 Dreamer actor update；
5. violation 区域的 GR recovery loss。
```

原因是 Dreamer 已经有 world model、imagined reward/value、lambda-return 和 actor update。第一版同时引入 Q、GR、完整 IS 会让系统过重，调参和解释都更困难。

---

## 2. Continuous Cost 定义

本方案使用连续 bottom-force cost，而不是只用 binary violation。

```text
bottom_force_t = selected bottom-force magnitude
force_excess_t = relu(bottom_force_t - force_threshold)
continuous_cost_t = clamp(force_excess_t / force_scale, 0, 1)
```

同时保留：

```text
binary_cost_t = bottom_force_t > force_threshold
```

用途：

```text
continuous_cost:
    world model cost head
    Gp/Gd TD target
    main safety objective

binary_cost:
    safety event logging
    cost rate / violation rate statistics
```

---

## 3. Gp 与 Gd

### 3.1 Gp：主策略 continuation risk

```text
Gp(z, a):
    在 latent z 下执行动作 a 后，
    后续继续由 main policy 控制时，
    未来 bottom-force continuous risk 的期望累计值。
```

Gp 用于：

```text
1. 判断 main action 是否 feasible / critical / infeasible；
2. 在不可行域内指导主策略降低风险；
3. 学习主策略附近的 safety boundary。
```

TD target：

```python
with torch.no_grad():
    a_next_main = main_policy.sample(z_next)

    target_gp = torch.maximum(
        target_gp1(z_next, a_next_main),
        target_gp2(z_next, a_next_main),
    )

    y_gp = continuous_cost + (1.0 - done) * gamma_cost * target_gp
    y_gp = torch.clamp(y_gp, 0.0, risk_max)
```

使用 `max` 的原因：

```text
Gp 服务主策略避险，风险估计应偏保守。
```

### 3.2 Gd：对偶策略 continuation risk

```text
Gd(z, a):
    在 latent z 下执行动作 a 后，
    后续继续由 dual policy 控制时，
    未来 bottom-force continuous risk 的期望累计值。
```

Gd 用于：

```text
1. 训练 dual policy；
2. 作为 dual imagined rollout 的 terminal risk bootstrap；
3. 帮助 dual 找到 high-risk / boundary behavior。
```

TD target：

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

使用 `min` 的原因：

```text
Gd 服务 high-risk dual actor。为避免 risk overestimation 被 dual exploit，target 使用较保守的 lower estimate。
```

### 3.3 不使用 `(1 - cost)` 截断

原 FDPI 的 cost 更接近 binary violation，所以可以使用：

```text
c + (1-done)(1-c)γG
```

本方案的 `continuous_cost` 是 force excess 强度，不是 absorbing violation indicator，所以不应使用：

```python
(1.0 - continuous_cost)
```

统一采用：

```python
y = continuous_cost + (1.0 - done) * gamma_cost * target_g
```

---

## 4. Gp/Gd 训练数据

Gp/Gd 都只使用真实 replay posterior latent 更新，不使用 imagined data 直接更新。

每个样本：

```text
z_t
action_t
continuous_cost_t
done_t
z_{t+1}
source_t
```

必须满足时间对齐：

```text
z_t + action_t → continuous_cost_t + z_{t+1}
```

### 4.1 Gp batch

Gp 服务 main policy，不能被 dual extreme data 主导。

推荐：

```text
60% main/recent samples
20% high-cost or boundary samples
10% dual-source samples
10% uniform samples
```

简化第一版：

```text
80% uniform
20% safety-critical
```

### 4.2 Gd batch

Gd 服务 dual policy，可更偏 high-cost / dual。

推荐：

```text
40% uniform
30% high-cost
20% dual-source
10% recent
```

---

## 5. Importance Sampling 的处理

第一版不建议完整复刻 FDPI 的 trajectory-level importance sampling。

原因：

```text
1. Dreamer actor 不是直接用 replay action 做 policy improvement；
2. world model 需要 dynamics/cost coverage，而不是严格 on-policy 分布；
3. continuous action IS 方差大；
4. early dual 数据少，IS 不能解决 support 缺失；
5. 完整 IS 会显著增加工程复杂度。
```

第一版使用：

```text
source-aware sampling
cost-aware sampling
loss weighting
```

示例：

```python
weight = 1.0

if source == DUAL:
    weight *= dual_source_weight

if continuous_cost > high_cost_threshold:
    weight *= high_cost_weight

if boundary_low < continuous_cost < boundary_high:
    weight *= boundary_weight
```

建议：

```text
Gp:
    DualSourceWeight = 1.0
    HighCostWeight = 2.0

Gd:
    DualSourceWeight = 2.0
    HighCostWeight = 3.0
```

可保留 `behavior_logp` 字段，后续做 clipped IS ablation。

---

## 6. 主策略：FDPI-Regime Actor Update

主策略用 Gp 判断 imagined action 的安全区域。

在 Dreamer imagined rollout 中：

```python
a_main, logp = main_policy.sample(z)
g = torch.maximum(gp1(z, a_main), gp2(z, a_main))
```

分区：

```python
g_mask = g.detach()

fea = g_mask < pf - cg
cri = (g_mask >= pf - cg) & (g_mask < pf)
inf = g_mask >= pf
```

其中：

```text
pf:
    可行风险阈值

cg:
    critical margin
```

### 6.1 Feasible：优化 reward

```python
loss_fea = -logp * stopgrad(A_reward)
```

语义：

```text
Gp 判定动作安全时，主策略继续按任务 reward 学习。
```

### 6.2 Critical：reward + Gp 平滑混合

```python
alpha = ((g - (pf - cg)) / (cg + eps)).clamp(0.0, 1.0)
risk_margin = F.relu(g - (pf - cg)) / (cg + eps)

loss_cri =
    -(1.0 - alpha) * logp * stopgrad(A_reward)
    + alpha * lambda_cri * risk_margin
```

语义：

```text
越接近不可行边界，reward 权重越低，Gp 安全权重越高。
```

### 6.3 Infeasible：优化 Gp

```python
risk_excess = F.relu(g - pf) / (risk_max - pf + eps)
loss_inf = lambda_inf * risk_excess
```

这里允许梯度：

```text
Gp(z,a) → action → main actor
```

但冻结 Gp 参数：

```text
actor step 只更新 main actor，不更新 Gp。
```

### 6.4 总 loss

```python
main_fdpi_loss =
    mean_mask(loss_fea, fea)
  + mean_mask(loss_cri, cri)
  + mean_mask(loss_inf, inf)
  - entropy_coef * entropy
```

注意：

```text
1. mask 用 g.detach()；
2. risk loss 中的 g 不 detach；
3. Gp 参数冻结；
4. 第一版作为独立分支，不与 Lagrangian 混合。
```

---

## 7. Dual Policy：Gp-Feasible-Ratio Driven Sampling

迁移原 FDPI 的 dual activation 思路：主策略越可行，危险样本越少，越需要 dual 补充危险/边界数据。

统计最近 main-source 数据：

```python
g_main = torch.maximum(gp1(z_main, a_main), gp2(z_main, a_main))

fea_ratio = mean(g_main < pf - cg)
cri_ratio = mean((g_main >= pf - cg) & (g_main < pf))
inf_ratio = mean(g_main >= pf)
```

关键：

```text
只统计 source == MAIN。
不要把 DUAL 样本混进去，否则 dual 会自我抑制。
```

推荐 dual ratio：

```python
if step < dual_start_step:
    dual_ratio = 0.0

elif kl_dual_main > max_kl:
    dual_ratio = 0.0

elif fea_ratio >= 0.95:
    dual_ratio = 0.50

elif fea_ratio >= 0.90:
    dual_ratio = 0.35

elif fea_ratio >= 0.80:
    dual_ratio = 0.20

elif cri_ratio >= 0.30:
    dual_ratio = 0.15

elif inf_ratio >= 0.20:
    dual_ratio = 0.05

else:
    dual_ratio = 0.10
```

保护条件：

```python
if main_real_cost_rate > high_cost_rate:
    dual_ratio = min(dual_ratio, 0.10)
```

原因：

```text
如果 main policy 真实 cost 已经很高，说明危险数据不缺，不需要 50% dual。
```

---

## 8. Dual Policy 更新

dual 目标：

```text
寻找高 bottom-force risk 的动作/轨迹；
同时不要离 main policy 太远。
```

### 8.1 简单版：Gd max-risk

```python
a_dual, logp_dual = dual_policy.sample(z)
gd = torch.minimum(gd1(z, a_dual), gd2(z, a_dual))

dual_loss =
    -gd.mean()
    + beta_kl * KL(dual || main)
    - beta_entropy * entropy
```

优点：

```text
接近 FDPI；
实现简单。
```

缺点：

```text
只看单步 Gd；
没有充分利用 world model 的 multi-step imagination。
```

### 8.2 推荐版：Imagined Risk Return + Terminal Gd

从真实 posterior latent 起点：

```text
z0 ~ replay posterior latent
```

在 frozen world model 中用 dual policy rollout H 步：

```text
z0 --a0^d--> z1 --a1^d--> ... --aH^d--> zH
```

每一步预测：

```text
predicted continuous cost c_hat_t
```

末端：

```text
terminal_gd = Gd(zH, aH)
```

risk return：

```python
R_risk =
    sum_{t=0}^{H-1} gamma_cost**t * c_hat_t
    + gamma_cost**H * terminal_gd
```

dual actor loss：

```python
dual_loss =
    -mean(logp_dual * stopgrad(normalize(R_risk)))
    + beta_kl * KL(dual || main)
    - beta_entropy * entropy
```

语义：

```text
dual 在 world model 中寻找未来会产生 high bottom-force risk 的轨迹；
Gd 用于补足 short horizon 之后的 future risk。
```

---

## 9. World Model 如何使用 Dual 数据

如果 replay 不断增长，dual/high-cost 数据会被 uniform sampling 稀释。

因此需要 safety-critical sampling：

```text
WorldModel update batch:
    80% uniform
    20% safety-critical
```

其中 safety-critical 包括：

```text
dual-source samples
high continuous cost samples
boundary cost samples
recent high-cost samples
```

如果担心影响 dynamics/reward，可拆成：

```text
normal world model update:
    uniform batch

extra cost/force head update:
    high-cost + dual + boundary batch
```

必须记录：

```text
WorldModelBatch/source_dual_ratio
WorldModelBatch/high_cost_ratio
WorldModelBatch/boundary_ratio
```

---

## 10. Replay Buffer 字段

推荐保存：

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
behavior_logp   # optional, for future IS ablation
```

source:

```text
MAIN
DUAL
RANDOM
```

---

## 11. 训练阶段

### 阶段 A：自然危险预训练

```text
dual sampling = off
main FDPI regime loss = off
world model cost/force = on
Gp/Gd update = on
```

目的：

```text
用 main/expert 自然危险数据预训练 world model、Gp、Gd。
```

### 阶段 B：启用 dual 采样

```text
dual sampling = on
ratio controlled by Gp feasible ratio
```

目的：

```text
当 main 趋于可行时，补充危险/边界数据。
```

### 阶段 C：启用 main FDPI-regime loss

```text
main actor:
    feasible → reward
    critical → reward + Gp
    infeasible → Gp
```

建议：

```text
dual sampling 先于 main FDPI loss 开启；
确保 Gp 已有基本区分能力。
```

---

## 12. 推荐配置

```yaml
FDPIRegimeDreamer:
  ContinuousCost:
    ForceThreshold: 1.0
    ForceScale: 5.0
    CostMin: 0.0
    CostMax: 1.0

  RiskCritic:
    GammaCost: 0.97
    RiskMax: 1.0
    TargetTau: 0.005
    Pf: 0.10
    Cg: 0.03

  Gp:
    Enable: true
    HighCostWeight: 2.0
    DualSourceWeight: 1.0
    BoundaryWeight: 2.0

  Gd:
    Enable: true
    HighCostWeight: 3.0
    DualSourceWeight: 2.0
    BoundaryWeight: 2.0

  MainFDPIRegime:
    Enable: true
    StartStep: 200000
    LambdaCri: 0.02
    LambdaInf: 0.05
    WarmupSteps: 100000
    EntropyCoef: 1.0e-4

  DualSampling:
    Enable: true
    StartStep: 100000
    FeasibleRatioWindow: 10000
    RatioFea95: 0.50
    RatioFea90: 0.35
    RatioFea80: 0.20
    RatioCriticalHigh: 0.15
    RatioUnsafeHigh: 0.05
    RatioDefault: 0.10
    MaxKLForSampling: 2.0
    HighMainCostRate: 0.20
    MaxRatioWhenMainCostHigh: 0.10

  DualUpdate:
    Type: "imagined_risk_return"
    Horizon: 5
    KLCoeff: 1.0
    EntropyCoef: 1.0e-4

  WorldModelSampling:
    EnableSafetyCriticalSampling: true
    UniformRatio: 0.80
    SafetyCriticalRatio: 0.20
```

---

## 13. 必要日志

### Gp/Gd

```text
Gp/loss
Gp/high_cost_mean
Gp/low_cost_mean
Gp/separation
Gp/main_action_mean
Gp/dual_action_mean

Gd/loss
Gd/high_cost_mean
Gd/low_cost_mean
Gd/separation
```

### Main regime

```text
MainFDPI/fea_ratio
MainFDPI/cri_ratio
MainFDPI/inf_ratio
MainFDPI/loss_fea
MainFDPI/loss_cri
MainFDPI/loss_inf
MainFDPI/gp_mean
MainFDPI/reward_adv_mean
```

### Dual sampling

```text
Dual/ratio
Dual/kl_to_main
Dual/real_cost_mean
Dual/source_count
Dual/active
```

### Replay / world model

```text
Replay/source_dual_ratio
Replay/main_cost_mean
Replay/dual_cost_mean
Replay/high_cost_ratio
Replay/boundary_ratio

WorldModelBatch/source_dual_ratio
WorldModelBatch/high_cost_ratio
WorldModelBatch/boundary_ratio
```

### Task safety

```text
Main/success_rate
Main/task_return
Main/bottom_force_mean
Main/bottom_force_peak
Main/continuous_cost_mean
```

---

## 14. 有效性判断

算法可行应满足：

```text
1. Gp 能区分 high-cost 与 low-cost；
2. 主策略开启 FDPI-regime 后，infeasible ratio 逐渐下降；
3. dual ratio 在 fea_ratio 高时上升；
4. dual 数据的 real cost 高于 main；
5. world model cost prediction 在 high-cost/dual holdout 上变好；
6. main success rate 不明显下降；
7. main bottom_force_mean / peak 下降。
```

失败诊断：

```text
Gp 不区分 high/low cost：
    暂缓开启 MainFDPIRegime；
    加强 high-cost/boundary sampling。

dual ratio 高但 world model cost 不改善：
    dual 数据被 replay 稀释；
    检查 WorldModelBatch/source_dual_ratio。

开启 MainFDPI 后 success 下降：
    Gp 不准；
    LambdaInf 太大；
    Pf 太低；
    MainFDPI 开启太早。

dual 50% 采样导致 replay 污染：
    限制 MaxRatioWhenMainCostHigh；
    main actor start state 不要过度采 dual 数据。
```

---

## 15. 最终建议

最推荐的结合路线：

```text
1. 不完整复刻 SAC-FDPI；
2. 迁移 FDPI 的 Gp/Gd、分区 actor loss、dual feasible-ratio activation；
3. Dreamer 保持 world model + imagination 主体；
4. 主策略用 Gp 分区：
      feasible → reward
      critical → reward + Gp
      infeasible → Gp
5. dual 用 Gd 学 high-risk trajectory；
6. dual sampling 由 main Gp feasible ratio 控制，最高可到 50%；
7. world model 通过 safety-critical sampling 使用 dual/high-cost 数据；
8. 第一版不加 GR 和完整 IS。
```

一句话：

```text
FDPI-Regime Dreamer = Dreamer 的世界模型想象学习 + FDPI 的风险分区与 dual 危险采样机制。
```
