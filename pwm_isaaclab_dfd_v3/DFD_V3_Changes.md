# DFD v3 Changes

本文档记录 `pwm_isaaclab_dfd_v3/` 相对 `pwm_isaaclab_dfd_v2/` 的主要改动，以及它如何对应
`Dual_Imagination_Dreamer_Continuous_Cost_Algorithm.md` 中的 Dual-Imagination Dreamer 方案。

## 1. 总体原则

v3 不修改 v2 原代码，而是在独立目录中新增实现：

```text
pwm_isaaclab_dfd_v3/
```

v3 继续复用 v2 中已经稳定的公共模块：

```text
world model / continuous cost head
replay buffer
Gd risk critic
main cost-aware actor-critic
cost utilities
```

v3 主要增强以下部分：

```text
dual policy wrapper
dual imagination update
dual real-sampling activation gate
v3 train entry / config / run script
```

## 2. 新增文件

```text
pwm_isaaclab_dfd_v3/__init__.py
pwm_isaaclab_dfd_v3/compat.py
pwm_isaaclab_dfd_v3/dual_policy_v3.py
pwm_isaaclab_dfd_v3/dual_imagination_v3.py
pwm_isaaclab_dfd_v3/trainer_dfd_v3.py
pwm_isaaclab_dfd_v3/train_dfd_v3.py
pwm_isaaclab_dfd_v3/train_dfd_v3.sh
pwm_isaaclab_dfd_v3/config_dfd_v3.yaml
```

其中 `compat.py` 提供 `colorama` fallback，避免 IsaacLab Python 环境缺少 `colorama` 时入口直接失败。

## 3. Dual Policy

文件：

```text
pwm_isaaclab_dfd_v3/dual_policy_v3.py
```

`DualPolicyV3` 继承 v2 的 `DualPolicyV2`，保持接口兼容：

```text
distribution(feat)
sample(feat)
rsample(feat)
log_prob(feat, action)
entropy(feat)
```

保留从 main actor 初始化的能力：

```text
InitFromMainActor: true
```

KL reference 固定为当前 main policy：

```text
KLReference: "current_main"
```

v3 没有引入 frozen expert reference。

## 4. Dual Imagination Update

文件：

```text
pwm_isaaclab_dfd_v3/dual_imagination_v3.py
```

v3 按文档实现 dual policy 在 world-model imagination 中训练：

```text
real replay posterior latent z0
    -> dual rollout imagined states z1 ... zH
    -> detach imagined states
    -> resample dual action
    -> maximize Gd under KL constraint
```

dual loss：

```text
L_dual =
    - mean(min(Gd1, Gd2))
    + KLCoeff * mean(log pi_dual(a|z) - log pi_main(a|z))
    - EntropyCoef * entropy(pi_dual)
```

梯度路径：

```text
dual_policy -> dual_action -> Gd -> dual_loss
```

冻结项：

```text
world_model 不更新
main_agent 不更新
Gd 参数不更新
```

只更新：

```text
dual_policy
```

imagined rollout 只用于训练 dual policy，不写入 replay。

## 5. Gd 与 World Model 训练

v3 继续复用 v2 的实现：

```text
pwm_isaaclab_dfd_v2/gd_risk.py
pwm_isaaclab_dfd_v2/world_model_dfd_v2.py
```

仍然满足文档要求：

```text
Gd 只用真实 replay posterior transition 更新
world model 只用真实 replay 更新
imagined dual data 不进入 replay，也不训练 world model
```

Gd target：

```text
y = continuous_cost + (1 - done) * gamma_cost * target_Gd(z_next, a_next_dual)
```

Gd 使用 double critic，并支持 source/high-cost weighting：

```text
DUAL source 样本权重更高
high-cost 样本权重更高
```

## 6. Real Dual Sampling

文件：

```text
pwm_isaaclab_dfd_v3/trainer_dfd_v3.py
```

真实环境采样仍按文档的低比例 dual 介入逻辑：

```text
if dual_enabled and random() < dual_ratio and dual_healthy:
    action = dual_policy(z)
    source = DUAL
else:
    action = main_policy(z)
    source = MAIN
```

在向量化环境中，v3 对每个 env 独立采样 dual mask：

```text
dual_mask = rand(num_envs) < dual_ratio
```

mask 命中的 env 使用 dual action，其余 env 使用 main action。

真实 transition 写入 replay：

```text
obs
action
reward
done
is_first
continuous_cost
binary_cost
extreme_cost
bottom_force
force_excess
source
```

## 7. 完整 Dual Sampling Gate

v3 已补齐文档中的更完整 gate：

```text
dual_enabled =
    step_ready
    and world_model_ready
    and main_policy_ready
    and gd_ready
    and kl_healthy
    and coverage_need
```

其中：

```text
step_ready:
    env_steps >= DualSampling.StartStep

world_model_ready:
    model_update_count >= DualSampling.MinModelUpdates

main_policy_ready:
    agent_update_count >= DualSampling.MinAgentUpdates

gd_ready:
    gd_update_count >= DualSampling.MinGdUpdates
    and last_gd_separation >= DualSampling.MinGdSeparation

kl_healthy:
    abs(last_dual_kl) <= DualImagination.MaxKLForSampling

coverage_need:
    recent_main_cost_rate < DualSampling.MainCostRateThreshold
    or recent_boundary_ratio < DualSampling.BoundaryRatioThreshold
```

`recent_main_cost_rate` 和 `recent_boundary_ratio` 由 `_RecentDualCoverage` 维护。

窗口统计包括：

```text
recent_steps
recent_main_cost_rate
recent_boundary_ratio
recent_success_rate
```

默认窗口：

```yaml
CoverageWindowSteps: 6400
```

boundary / dangerous coverage 默认定义：

```yaml
BoundaryCostMin: 0.05
BoundaryCostMax: 0.5
```

## 8. v3 默认配置

文件：

```text
pwm_isaaclab_dfd_v3/config_dfd_v3.yaml
```

关键 dual 配置：

```yaml
Replay:
  world_model_max_dual_fraction: 0.10
  cost_positive_ratio: 0.25

DualPolicy:
  LR: 8.0e-5
  Eps: 1.0e-5
  InitFromMainActor: true
  KLReference: "current_main"

DualImagination:
  Enable: true
  StartStep: 100000
  Horizon: 5
  Objective: "max_risk"
  KLCoeff: 1.0
  MaxKLForSampling: 2.0
  EntropyCoef: 1.0e-4
  GradClipNorm: 100.0
  UpdateSteps: 1

DualSampling:
  Enable: true
  StartStep: 120000
  RatioStart: 0.01
  RatioFinal: 0.03
  RatioWarmupSteps: 100000
  RequireKLHealthy: true
  MinModelUpdates: 1
  MinAgentUpdates: 1
  RequireGdReady: true
  MinGdUpdates: 1
  MinGdSeparation: 0.0
  RequireCoverageNeed: true
  MainCostRateThreshold: 0.10
  BoundaryRatioThreshold: 0.05
  CoverageWindowSteps: 6400
  BoundaryCostMin: 0.05
  BoundaryCostMax: 0.5
```

关键 main cost 优化配置：

```yaml
MainCostAwareReward:
  Enable: true
  StartStep: 100000
  LambdaStart: 0.3
  LambdaCost: 3.0
  LambdaWarmupSteps: 200000
  UsePredictedCost: true
```

## 9. 新增日志

v3 增加以下 dual sampling gate 日志：

```text
DualSampling/healthy
DualSampling/step_ready
DualSampling/world_model_ready
DualSampling/main_policy_ready
DualSampling/kl_healthy
DualSampling/gd_ready
DualSampling/coverage_need
DualSampling/low_main_cost
DualSampling/low_boundary
DualSampling/kl_to_main
DualSampling/gd_separation
DualSampling/recent_steps
DualSampling/recent_main_cost_rate
DualSampling/recent_boundary_ratio
DualSampling/recent_success_rate
```

这些日志用于判断 dual 是否按文档要求“在模型已有基础、且需要危险/边界补充时介入”。

v3 也保留 main cost 优化相关日志：

```text
Main/lambda_cost
Main/predicted_cost_mean
Main/predicted_cost_max
Main/cost_penalty_mean
Main/cost_penalty_max
Main/safe_reward_delta
Main/task_imagined_reward
Main/safe_imagined_reward
Main/task_return
Main/safe_return
Main/episode_cost_mean
Main/bottom_force_mean
Main/bottom_force_peak
Replay/cost_mean
Replay/main_cost_mean
Replay/dual_cost_mean
Replay/main_cost_rate
Replay/dual_cost_rate
Replay/extreme_cost_rate
```

这些日志用于确认 cost head、真实 cost 数据和 main policy 的 cost-aware reward 是否形成闭环。

## 10. Main Policy Cost 优化

v3 保留文档要求：

```text
main policy 不使用 Gp/Gd 直接更新
```

main policy 仍然使用 Dreamer 原始 actor-critic 更新形式，但 actor-critic 训练 reward 会在指定步数后切换为
cost-aware reward。这个部分是 v3 的安全优化闭环之一，不属于 dual policy loss。

真实环境 step 后，v3 从 force/contact 信息中提取连续 cost，并写入 replay：

```text
continuous_cost
binary_cost
extreme_cost
bottom_force
force_excess
source
```

bottom force 到 cost 的计算复用 v2 的 `cost_utils`，v3 不重新实现该逻辑。

实现位置：

```text
pwm_isaaclab_dfd_v2/cost_utils.py::extract_continuous_cost
pwm_isaaclab_dfd_v2/cost_utils.py::extract_bottom_force
pwm_isaaclab_dfd_v2/cost_utils.py::compute_continuous_cost
trainer_dfd_v3.py 中环境 step 后调用 extract_continuous_cost
```

### 10.1 Bottom Force 提取

每次真实环境 step 后，v3 先从 `info` 或 observation 中取 bottom force。

优先级：

```text
1. info["bottom_force"]
2. info["bottom_force_peak"]
3. info["diagnostics"]["bottom_force"]
4. info["diagnostics"]["bottom_force_peak"]
5. obs_dict[force_key] 或 obs_dict["force"]
```

如果从 observation 的 force 向量中提取，默认使用：

```yaml
BottomForceChannels: [2, 5]
```

计算方式：

```text
force_abs = abs(force)
bottom_force = max(force_abs[channel 2], force_abs[channel 5])
```

如果 force 维度不足以读取 `[2, 5]`，则退化为：

```text
force_dim >= 2: bottom_force = abs(force[:, 1])
force_dim == 1: bottom_force = abs(force[:, 0])
```

最后做数值保护：

```text
bottom_force = nan_to_num(bottom_force, nan=0, posinf=1e6)
bottom_force = max(bottom_force, 0)
```

### 10.2 Bottom Force 到 Continuous Cost

默认配置：

```yaml
ForceThreshold: 0.1
LowForceScale: 0.05
CostForceMax: 15.0
ExtremeForceThreshold: 5.0
ClipCost: true
CostMin: 0.0
CostMax: 1.0
```

先计算超过安全阈值的 force excess：

```text
force_excess = relu(bottom_force - ForceThreshold)
```

其中默认安全阈值是：

```text
ForceThreshold = 0.1
```

也就是说：

```text
bottom_force <= 0.1  -> force_excess = 0
bottom_force >  0.1  -> force_excess = bottom_force - 0.1
```

continuous cost 使用 log 压缩，避免大力接触时 cost 数值过大：

```text
low_scale = max(LowForceScale, 1e-6)
force_max = max(CostForceMax, low_scale)
normalizer = log1p(force_max / low_scale)

continuous_cost = log1p(force_excess / low_scale) / normalizer
```

代入 v3 默认值：

```text
low_scale = 0.05
force_max = 15.0
normalizer = log1p(15.0 / 0.05)

continuous_cost = log1p(force_excess / 0.05) / log1p(300)
```

如果开启 clipping：

```text
continuous_cost = clamp(continuous_cost, CostMin, CostMax)
```

默认就是：

```text
continuous_cost = clamp(continuous_cost, 0.0, 1.0)
```

因此：

```text
bottom_force <= 0.1        -> continuous_cost = 0
bottom_force 约等于 15.1   -> continuous_cost 约等于 1
bottom_force 更大          -> clipping 后仍为 1
```

### 10.3 Binary / Extreme Cost

除了连续 cost，v3 还同时计算二值 cost：

```text
binary_cost = 1 if bottom_force > ForceThreshold else 0
```

默认：

```text
binary_cost = 1 if bottom_force > 0.1 else 0
```

极端 cost：

```text
extreme_cost = 1 if bottom_force > ExtremeForceThreshold else 0
```

默认：

```text
extreme_cost = 1 if bottom_force > 5.0 else 0
```

最终写入 replay 的 force/cost 字段为：

```text
bottom_force
force_excess
continuous_cost
binary_cost
extreme_cost
```

world model 训练时继续学习 reward 和 cost head。main policy 做 imagined actor-critic 更新时，先由
world model 在 imagined feature 上预测 cost：

```text
predicted_continuous_cost = world_model.predict_cost(imagined_feature)
```

如果 world model 没有 `predict_cost` 接口，则回退到 force prediction head 计算 continuous cost。

当满足以下条件时启用 cost-aware reward：

```text
MainCostAwareReward.Enable == true
env_steps >= MainCostAwareReward.StartStep
```

main policy 实际用于训练的 reward 为：

```text
safe_reward = imagined_task_reward - lambda_cost * predicted_continuous_cost
```

默认：

```yaml
StartStep: 100000
LambdaStart: 0.3
LambdaCost: 3.0
LambdaWarmupSteps: 200000
```

训练中实际使用的 `lambda_cost` 会从 `LambdaStart` 线性 warmup 到 `LambdaCost`：

```text
lambda_cost = linear_warmup(
    env_steps - StartStep,
    LambdaStart,
    LambdaCost,
    LambdaWarmupSteps,
)
```

也就是说，v3 的 main actor 不是通过 `Gd` 反向传播更新，而是通过 Dreamer imagined rollout 中的
`safe_reward` 间接学习降低 force/cost，同时保留任务 reward。

实现位置：

```text
trainer_dfd_v3.py::_predict_imagined_cost
trainer_dfd_v3.py::train_agent_step_dfd_v3
```

episode 统计也同步记录 cost-aware return：

```text
episode_safe_reward += reward - lambda_cost * continuous_cost
```

对应日志：

```text
Main/safe_return
Main/episode_cost_mean
Main/bottom_force_peak
```

## 11. 启动方式

入口脚本：

```bash
bash /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab_dfd_v3/train_dfd_v3.sh
```

该脚本使用：

```text
/home/yhy/IsaacLab-1.4.0/isaaclab.sh
```

训练入口：

```text
pwm_isaaclab_dfd_v3/train_dfd_v3.py
```

配置：

```text
pwm_isaaclab_dfd_v3/config_dfd_v3.yaml
```

## 12. 已验证项目

已完成静态验证：

```text
python3 -m py_compile pwm_isaaclab_dfd_v3/*.py
bash -n pwm_isaaclab_dfd_v3/train_dfd_v3.sh
```

已验证 IsaacLab launcher 下帮助入口可用：

```text
/home/yhy/IsaacLab-1.4.0/isaaclab.sh -p \
  /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab_dfd_v3/train_dfd_v3.py -h
```

当前环境中，直接用 IsaacLab Python import：

```text
omni.isaac.lab
gymnasium
```

仍会失败。这是本地 IsaacLab/Python 环境问题，不是 v3 代码语法问题。

## 13. 与文档的对应关系

v3 当前覆盖文档中的核心闭环：

```text
真实 replay 存 continuous cost/source
world model 只用真实 replay 学 dynamics/reward/cost
Gd 只用真实 replay posterior transition 更新
dual policy 在 imagination 中最大化 Gd
dual policy 受 KL(current main) 约束
dual policy 低比例进入真实环境采样
真实 dual 数据进入 replay
main policy 通过 cost-aware imagined reward 学安全夹取
main policy 不直接使用 Gp/Gd 更新
```

v3 也补齐了文档中的完整 dual gate：

```text
world_model_ready
gd_ready
main_policy_ready
kl_healthy
dangerous / boundary coverage need
```

没有实现的文档可选项：

```text
frozen expert KL reference
adaptive lambda_cost / Lagrange multiplier
Gp module
```

这些属于文档中的可选扩展，不属于当前 v3 默认实现。
