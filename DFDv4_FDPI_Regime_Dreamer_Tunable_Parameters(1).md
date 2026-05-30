# DFDv4 / FDPI-Regime Dreamer：环境相关参数调节指南

## 0. 核心判断

DFDv4 中真正需要结合具体环境调节的参数，主要集中在四类：

```text
1. cost 的定义：什么 force 算危险，continuous cost 如何归一化；
2. Gp 的风险分区阈值：什么动作算 feasible / critical / infeasible；
3. 主策略 FDPI-regime loss 强度：不可行域内 Gp 对 actor 的影响有多强；
4. dual 采样比例：主策略多安全时开启多少 dual 数据采集。
```

其他参数第一版应尽量固定，不要一开始全部调。  
优先调安全尺度和分区阈值，而不是先调 KL、Horizon、TargetTau、完整 IS 等次要参数。

---

## 1. 参数调节优先级总览

推荐调参顺序：

```text
Step 1:
    调 ContinuousCost：
        ForceThreshold
        ForceScale

Step 2:
    训练 world model + Gp/Gd，但不开 MainFDPIRegime。
    检查 Gp/Gd 是否能区分 high-cost 与 low-cost。

Step 3:
    调 RiskCritic：
        Pf
        Cg

Step 4:
    开 DualSampling。
    调 feasible-ratio 对应的 dual ratio 档位。

Step 5:
    开 MainFDPIRegime。
    调 StartStep / LambdaCri / LambdaInf。

Step 6:
    调 WorldModelSampling 与 Gp/Gd weighting。
    确保 dual/high-cost 数据没有被大 replay 稀释。
```

最重要的环境相关参数：

```yaml
ContinuousCost:
  ForceThreshold
  ForceScale

RiskCritic:
  Pf
  Cg

MainFDPIRegime:
  StartStep
  LambdaCri
  LambdaInf

DualSampling:
  StartStep
  RatioFea95
  RatioFea90
  RatioFea80
  MaxRatioWhenMainCostHigh

WorldModelSampling:
  SafetyCriticalRatio
```

如果进一步压缩，最核心的 6 个是：

```text
ForceThreshold
ForceScale
Pf
Cg
LambdaInf
DualSampling ratio table
```

---

## 2. ContinuousCost 参数

### 2.1 参数

```yaml
ContinuousCost:
  ForceThreshold: 1.0
  ForceScale: 5.0
  CostMin: 0.0
  CostMax: 1.0
```

定义：

```text
bottom_force_t = selected bottom-force magnitude
force_excess_t = relu(bottom_force_t - ForceThreshold)
continuous_cost_t = clamp(force_excess_t / ForceScale, CostMin, CostMax)
binary_cost_t = bottom_force_t > ForceThreshold
```

### 2.2 需要结合环境调的参数

#### ForceThreshold

含义：

```text
多大的 bottom force 开始被认为是不希望出现的危险力。
```

调节依据：

```text
1. 专家策略或预训练主策略的 bottom_force_mean；
2. 专家策略或预训练主策略的 bottom_force_peak；
3. 正常夹取中不可避免的轻微接触；
4. 真实任务中可接受的底部接触力阈值。
```

判断：

```text
ForceThreshold 太低：
    正常轻微夹取接触也被认为危险；
    Gp 可能把大部分夹取动作判为 infeasible；
    主策略容易不敢夹。

ForceThreshold 太高：
    只有极端撞底才产生 cost；
    world model / Gp 难以学习安全边界；
    main bottom force 可能下降不明显。
```

#### ForceScale

含义：

```text
将 force excess 映射到 [0,1] 的尺度。
```

判断：

```text
ForceScale 太小：
    continuous_cost 很容易饱和到 1；
    Gp/Gd 只知道“都很危险”，学不到风险强弱。

ForceScale 太大：
    continuous_cost 长期接近 0；
    Gp/Gd 学不到足够安全信号。
```

### 2.3 建议先看的日志

```text
Main/bottom_force_mean
Main/bottom_force_peak
Main/force_excess_mean
Main/continuous_cost_mean
Main/binary_cost_rate

Replay/continuous_cost_mean
Replay/continuous_cost_max
Replay/high_cost_ratio
Replay/boundary_ratio
```

### 2.4 理想状态

```text
continuous_cost 不应长期全 0；
continuous_cost 不应大量饱和到 1；
binary_cost_rate 不应极端接近 0 或 1；
high-cost 和 low-cost 样本都应存在。
```

---

## 3. RiskCritic 参数：Pf / Cg / RiskMax

### 3.1 参数

```yaml
RiskCritic:
  GammaCost: 0.97
  RiskMax: 1.0
  TargetTau: 0.005
  Pf: 0.10
  Cg: 0.03
```

### 3.2 需要重点调的参数

#### Pf

含义：

```text
Gp 风险阈值。
当 Gp(z,a) >= Pf 时，当前 action 被视为 infeasible。
```

判断：

```text
Pf 太低：
    大量动作被判为 infeasible；
    MainFDPIRegime 开启后可能压制夹取；
    success rate 容易下降。

Pf 太高：
    大量危险动作仍被视为 feasible；
    主策略继续按 reward 优化，安全改善弱。
```

#### Cg

含义：

```text
critical margin。
critical 区间为：
    Pf - Cg <= Gp(z,a) < Pf
```

判断：

```text
Cg 太小：
    critical 区几乎没有样本；
    策略从 reward 优化突然切换到 Gp 优化，训练不平滑。

Cg 太大：
    过多动作进入 critical；
    安全 penalty 过早干扰任务学习。
```

### 3.3 建议先看 Gp 输出分布再设 Pf/Cg

在不开 MainFDPIRegime 时，先训练 Gp，统计：

```text
Gp/main_action_mean
Gp/main_action_p50
Gp/main_action_p75
Gp/main_action_p90
Gp/high_cost_mean
Gp/low_cost_mean
Gp/separation
```

经验设置：

```text
Pf 应该落在 low-cost Gp 和 high-cost Gp 之间；
Cg 应该让 critical ratio 有一定比例，但不应过大。
```

### 3.4 必看日志

```text
MainFDPI/fea_ratio
MainFDPI/cri_ratio
MainFDPI/inf_ratio
Gp/high_cost_mean
Gp/low_cost_mean
Gp/separation
```

### 3.5 常见问题

```text
inf_ratio 一开始接近 100%：
    Pf 太低；
    ForceScale 太小；
    Gp 过于悲观；
    Gp 还没学好就启用了 MainFDPIRegime。

fea_ratio 长期接近 100%：
    Pf 太高；
    Gp 没学到风险；
    continuous_cost 信号太弱。

cri_ratio 长期接近 0：
    Cg 太小；
    Pf 位置不合理；
    Gp 输出分布太集中。
```

---

## 4. MainFDPIRegime 参数

### 4.1 参数

```yaml
MainFDPIRegime:
  Enable: true
  StartStep: 200000
  LambdaCri: 0.02
  LambdaInf: 0.05
  WarmupSteps: 100000
  EntropyCoef: 1.0e-4
```

### 4.2 参数含义

#### StartStep

```text
什么时候开始让 Gp 分区 loss 影响 main actor。
```

建议：

```text
不要太早开启；
至少等 world model cost / Gp 有基本区分能力；
dual sampling 最好先于 MainFDPIRegime 开启一段时间。
```

#### LambdaCri

```text
critical 区域中 Gp safety penalty 的强度。
```

#### LambdaInf

```text
infeasible 区域中降低 Gp 的强度。
```

这是最敏感的主策略安全参数之一。

#### WarmupSteps

```text
让 LambdaCri / LambdaInf 从 0 逐渐增加到目标值。
```

推荐保留 warmup，避免 Gp loss 突然破坏主策略。

### 4.3 推荐初值

```yaml
MainFDPIRegime:
  StartStep: 200000
  LambdaCri: 0.02
  LambdaInf: 0.05
  WarmupSteps: 100000
```

### 4.4 调节逻辑

```text
MainFDPIRegime 开启后 success rate 明显下降：
    StartStep 太早；
    LambdaInf 太大；
    Pf 太低；
    Gp 还没学好；
    ForceThreshold / ForceScale 太激进。

bottom force 不下降：
    LambdaCri / LambdaInf 太小；
    Pf 太高；
    Gp 没有区分 high/low cost；
    world model cost prediction 不准。

critical 区样本多但 bottom force 不降：
    LambdaCri 太小；
    Cg 过大导致太多轻微风险样本进入 critical；
    Gp 输出没有校准。
```

### 4.5 建议调参方式

先固定：

```text
LambdaCri = 0.02
```

主要调：

```text
StartStep
LambdaInf
```

不要同时调太多参数。

---

## 5. DualSampling 参数

### 5.1 参数

```yaml
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
```

### 5.2 核心逻辑

dual ratio 根据 main-source 的 Gp feasible ratio 控制：

```text
main policy 越可行，dual ratio 越高；
main policy 越不安全，dual ratio 越低；
KL 爆炸时暂停 dual；
main real cost 已经很高时限制 dual ratio。
```

推荐分段：

```python
if step < StartStep:
    dual_ratio = 0.0

elif kl_dual_main > MaxKLForSampling:
    dual_ratio = 0.0

elif fea_ratio >= 0.95:
    dual_ratio = RatioFea95

elif fea_ratio >= 0.90:
    dual_ratio = RatioFea90

elif fea_ratio >= 0.80:
    dual_ratio = RatioFea80

elif cri_ratio >= 0.30:
    dual_ratio = RatioCriticalHigh

elif inf_ratio >= 0.20:
    dual_ratio = RatioUnsafeHigh

else:
    dual_ratio = RatioDefault

if main_real_cost_rate > HighMainCostRate:
    dual_ratio = min(dual_ratio, MaxRatioWhenMainCostHigh)
```

### 5.3 需要重点调的参数

```text
StartStep
RatioFea95
RatioFea90
RatioFea80
MaxRatioWhenMainCostHigh
HighMainCostRate
```

### 5.4 调节逻辑

```text
dual 数据太少，world model cost 没改善：
    提高 RatioFea80 / RatioFea90；
    提高 SafetyCriticalRatio；
    检查 WorldModelBatch/source_dual_ratio。

dual 数据太多，main success 下降：
    降低 RatioFea95；
    降低 RatioFea90；
    降低 MaxRatioWhenMainCostHigh；
    限制 main actor start states 中 dual 数据比例。

main 还很危险却开了大量 dual：
    HighMainCostRate 太高；
    main_real_cost_rate 统计错误；
    feasible_ratio 统计混入了 dual 数据。
```

### 5.5 关键注意

```text
feasible_ratio 必须只统计 source == MAIN；
不要把 source == DUAL 混进去；
否则 dual 会自我抑制。
```

---

## 6. WorldModelSampling 参数

### 6.1 参数

```yaml
WorldModelSampling:
  EnableSafetyCriticalSampling: true
  UniformRatio: 0.80
  SafetyCriticalRatio: 0.20
```

### 6.2 含义

由于 replay buffer 不断增长，dual/high-cost 数据会被 uniform sampling 稀释。  
因此 world model update batch 中需要固定比例的 safety-critical samples：

```text
source == DUAL
continuous_cost > HighCostThreshold
BoundaryLow < continuous_cost < BoundaryHigh
recent high-cost samples
```

### 6.3 推荐初值

```yaml
WorldModelSampling:
  UniformRatio: 0.80
  SafetyCriticalRatio: 0.20
```

如果 high-cost / dual holdout 上 cost prediction 不改善：

```yaml
WorldModelSampling:
  UniformRatio: 0.70
  SafetyCriticalRatio: 0.30
```

不建议一开始超过 0.30，否则 world model 可能过度偏向危险接触数据，影响正常任务 dynamics / reward 建模。

### 6.4 必看日志

```text
WorldModelBatch/source_dual_ratio
WorldModelBatch/high_cost_ratio
WorldModelBatch/boundary_ratio
WorldModel/cost_loss
WorldModel/high_cost_cost_loss
Main/success_rate
```

---

## 7. Gp/Gd Weighting 参数

### 7.1 参数

```yaml
Gp:
  HighCostWeight: 2.0
  DualSourceWeight: 1.0
  BoundaryWeight: 2.0

Gd:
  HighCostWeight: 3.0
  DualSourceWeight: 2.0
  BoundaryWeight: 2.0
```

### 7.2 为什么 Gp 和 Gd 不一样

```text
Gp 服务 main policy：
    不应被 dual extreme data 主导；
    dual 样本可以帮助它看到危险边界，但权重不能过高。

Gd 服务 dual policy：
    可以更偏 high-cost / dual-source data；
    它的目标就是帮助 dual 找到高风险轨迹。
```

### 7.3 调节逻辑

```text
Gp 过于悲观，main infeasible ratio 很高：
    降低 Gp/HighCostWeight；
    降低 Gp/DualSourceWeight；
    检查 Pf 是否太低。

Gp 分不清 high/low cost：
    提高 Gp/HighCostWeight；
    增加 boundary/high-cost sampling；
    检查 continuous_cost 是否太弱。

Gd 学不到危险：
    提高 Gd/HighCostWeight；
    提高 Gd/DualSourceWeight；
    检查 dual 数据是否真的进入 Gd batch。
```

---

## 8. DualUpdate 参数

### 8.1 参数

```yaml
DualUpdate:
  Type: "imagined_risk_return"
  Horizon: 5
  KLCoeff: 1.0
  EntropyCoef: 1.0e-4
```

### 8.2 调节优先级

这些不是第一优先级，建议先固定。

```text
Horizon:
    默认为 5。
    太短可能短视；
    太长模型误差更大。

KLCoeff:
    约束 dual 不要离 main 太远。

EntropyCoef:
    防止 dual 模式塌缩。
```

### 8.3 调节逻辑

```text
dual 采不到比 main 更高 cost：
    KLCoeff 可能太大；
    dual 被 main 绑得太紧。

dual 变成极端乱撞：
    KLCoeff 太小；
    MaxKLForSampling 太高。

dual entropy 很低：
    EntropyCoef 太小。

dual imagination risk 很高，但真实 dual cost 不高：
    dual exploit world model / Gd；
    降低 Horizon；
    增强 KL；
    检查 world model cost prediction。
```

---

## 9. 不建议第一版重点调的参数

先固定：

```yaml
RiskCritic:
  GammaCost: 0.97
  TargetTau: 0.005

DualUpdate:
  Horizon: 5
  EntropyCoef: 1.0e-4

MainFDPIRegime:
  EntropyCoef: 1.0e-4
```

只有出现明显问题时再调：

```text
Gp/Gd target 爆炸：
    降低 GammaCost；
    降低 RiskMax；
    检查 continuous_cost 是否经常饱和。

target critic 抖动：
    降低 TargetTau。

dual 明显短视：
    Horizon 从 5 提到 8，但要警惕模型误差。
```

---

## 10. 推荐默认配置

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

## 11. 实验前的参数检查清单

启动训练前确认：

```text
1. ForceThreshold 对应真实环境中“不可接受底部冲击”的力阈值；
2. ForceScale 使 continuous_cost 不长期全 0 或全 1；
3. Pf/Cg 与 Gp 输出范围匹配；
4. MainFDPIRegime 不会在 Gp 未学好时过早开启；
5. DualSampling 的 feasible_ratio 只统计 MAIN source；
6. dual ratio 最高 50% 时不会让 main actor batch 被 dual 数据主导；
7. WorldModelSampling 能保证 dual/high-cost/boundary 数据进入训练；
8. Gp 的 DualSourceWeight 不高于 Gd；
9. 没有使用完整 IS 作为第一版主线；
10. 所有关键日志都已记录。
```

---

## 12. 最终建议

第一版调参不要贪多。最合理顺序是：

```text
先调 cost 定义；
再调 Gp 阈值；
再开 dual feasible-ratio sampling；
最后开 MainFDPIRegime loss。
```

最需要结合环境调整的是：

```text
什么 force 算危险；
Gp 多大算不可行；
不可行域中 Gp loss 多强；
主策略多安全时开多少 dual；
dual/high-cost 数据在 world model batch 中占多少。
```
