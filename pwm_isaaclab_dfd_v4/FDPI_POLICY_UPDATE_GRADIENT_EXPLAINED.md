# FDPI 主策略更新梯度问题详解

本文解释 v4 中主策略为什么会在加入 GP 后出现策略崩塌，以及为什么 `DetachActionForLogProb=true` 可以修复这个问题。

重点不是调参，而是主策略 actor loss 的梯度路径写错了。

---

## 1. 先看普通 Dreamer 的 actor 更新

普通 Dreamer 的主策略更新可以简化成下面几个步骤。

世界模型先从 replay buffer 中取一段真实轨迹，得到 posterior latent：

```text
z_0, z_1, ...
```

然后在世界模型里 rollout imagination：

```text
a_t ~ pi_theta(. | z_t)
z_{t+1} = world_model(z_t, a_t)
r_t = reward_head(z_t)
```

在当前代码里，这个 imagination 过程是在 `torch.no_grad()` 下做的，所以采样出来的动作 `a_t` 对 actor update 来说是一个固定动作：

```text
a_t = stopgrad(a_t)
```

也就是说，普通 Dreamer 更新 actor 时看到的是：

```text
给定 latent z_t
给定已经采好的动作 a_t
给定这个动作后面想象出来的 return
更新策略，让好动作概率更大，坏动作概率更小
```

---

## 2. 普通 Dreamer 的公式

先定义 reward lambda-return：

```text
G_t^lambda = lambda_return(r_t, V(z_t), discount_t)
```

优势函数：

```text
A_t = (G_t^lambda - V(z_t)) / scale
```

其中：

```text
V(z_t)
```

是 critic 对当前 latent 的价值估计。

```text
scale
```

是 return 的归一化尺度，避免 advantage 太大。

普通 Dreamer actor loss 可以写成：

```text
L_actor = - E[ w_t * stopgrad(A_t) * log pi_theta(stopgrad(a_t) | z_t) ]
          - beta * E[ H(pi_theta(. | z_t)) ]
```

其中：

```text
w_t
```

是 imagination 的 discount weight。

```text
H(pi_theta)
```

是策略熵。

这个公式的含义很直观：

```text
如果 A_t > 0：
    说明这个动作比 critic 预期好
    增大 log pi(a_t | z_t)
    也就是让策略以后更容易选这个动作

如果 A_t < 0：
    说明这个动作比 critic 预期差
    减小 log pi(a_t | z_t)
    也就是让策略以后少选这个动作
```

注意最关键的一点：

```text
log pi_theta(stopgrad(a_t) | z_t)
```

这里的动作 `a_t` 是 fixed sample。它不再参与反向传播。

所以梯度是标准 policy gradient：

```text
grad_theta L_actor
= - E[ w_t * stopgrad(A_t) * grad_theta log pi_theta(a_t | z_t) ]
```

这条梯度会正确地告诉 actor：

```text
哪些动作应该更可能出现
哪些动作应该更不可能出现
```

---

## 3. FDPI 主策略原本想做什么

FDPI 想在 reward 优化之外，引入 GP 风险评估。

令：

```text
g_t = Gp(z_t, a_t)
```

其中 `g_t` 表示主策略在 latent `z_t` 下选择动作 `a_t` 的风险估计。

根据 `g_t`，把样本分为三类：

```text
feasible:
    g_t < Pf - Cg

critical:
    Pf - Cg <= g_t < Pf

infeasible:
    g_t >= Pf
```

然后对 actor loss 做 regime-aware 调整：

```text
L_fdpi
= reward 部分
  + GP risk 部分
  - entropy 部分
```

更具体一点：

```text
L_reward = - E[ w_t * q(g_t) * stopgrad(A_t) * log pi_theta(a_t | z_t) ]
```

```text
L_risk = E[ w_t * lambda(g_t) * risk_penalty(g_t) ]
```

```text
L_fdpi = L_reward + L_risk - beta * E[H]
```

其中：

```text
q(g_t)
```

是 reward 权重。风险越高，reward 可能被压低一点。

```text
risk_penalty(g_t)
```

是 GP 风险惩罚。

这个设计本身没有问题。问题出在 `a_t` 怎么放进 `log pi`。

---

## 4. 原 v4 FDPI 的关键错误

原来的 FDPI 更新里，不是用 imagination 中已经采好的固定动作，而是在 actor update 里重新采样：

```python
main_action = dist.rsample()
log_prob = dist.log_prob(main_action)
```

这里的 `rsample()` 是 reparameterized sample。

可以理解成：

```text
main_action = mu_theta(z_t) + sigma_theta(z_t) * epsilon
```

其中：

```text
epsilon ~ Normal(0, 1)
```

也就是说，`main_action` 本身依赖 actor 参数 `theta`。

然后又计算：

```text
log pi_theta(main_action | z_t)
```

这就变成了：

```text
log pi_theta( mu_theta(z_t) + sigma_theta(z_t) * epsilon | z_t )
```

也就是 `log_prob` 的输入动作也随着 `theta` 一起变。

这和普通 policy gradient 不是同一个东西。

---

## 5. 为什么这个公式会让 mean 梯度消失

用一维 Gaussian 举例。

策略是：

```text
pi(a | z) = Normal(mu, sigma)
```

Gaussian 的 log probability 是：

```text
log pi(a)
= -0.5 * ((a - mu) / sigma)^2 - log(sigma) + const
```

### 情况 A：动作是 fixed sample

如果动作 `a` 是固定的：

```text
a = stopgrad(a)
```

那么：

```text
d log pi(a) / d mu
= (a - mu) / sigma^2
```

这个值一般不是 0。

所以 actor 可以学：

```text
如果这个动作好，就把 mu 往这个动作靠近
如果这个动作差，就让 mu 远离这个动作
```

这就是正常的 reward policy gradient。

### 情况 B：动作来自 rsample，且不 detach

如果：

```text
a = mu + sigma * epsilon
```

那么：

```text
a - mu = sigma * epsilon
```

代回 log probability：

```text
log pi(mu + sigma * epsilon)
= -0.5 * epsilon^2 - log(sigma) + const
```

你会发现：

```text
mu 消失了
```

因此：

```text
d log pi(mu + sigma * epsilon) / d mu = 0
```

这就是问题的核心。

reward loss 本来应该告诉 actor：

```text
这个动作好，让 mean 更靠近它
这个动作差，让 mean 远离它
```

但是因为用了：

```text
log pi_theta(rsample_action)
```

reward 对 mean 的训练信号被抵消了。

---

## 6. 那为什么策略还会变化，甚至崩塌？

你可能会问：

```text
如果 mean 梯度都没了，那策略应该不变才对，为什么会崩？
```

原因是 actor 网络通常不是完全分开的。

它一般长这样：

```text
shared trunk
    -> mean head
    -> std head
```

虽然 reward loss 对 mean head 的直接梯度很弱，但它仍然会影响：

```text
std head
shared trunk
entropy
critic 相关的整体训练状态
```

特别是 shared trunk 一动，mean head 的输入特征也变了。

所以会出现一种很危险的情况：

```text
mean 没有被 reward 正确指导
但 shared trunk / std 还在动
结果动作分布发生了无意义漂移
```

这就是为什么你看到的现象是：

```text
一开始 reward 还能上去
过几次 update 后突然掉下去
contact / lift / approach 都接近 0
```

这不是 GP 把策略一点点压安全了，而是 reward 梯度没有正确约束住 actor，策略分布被错误梯度带偏了。

---

## 7. 为什么不是 GP 本身的问题

我做过一个关键消融：

```text
B: FDPI reward-only
   LambdaCri = 0
   LambdaInf = 0
   MinRewardWeightCri = 1
   MinRewardWeightInf = 1
```

也就是说：

```text
没有 GP risk penalty
没有 reward 降权
只保留 FDPI 分支的 reward 更新写法
```

结果它还是崩了：

```text
final total ~= 0.146
final contact = 0
final task_return ~= 49.6
```

这说明：

```text
即使不优化 GP，FDPI 分支本身的 actor reward 更新也会导致崩塌
```

所以根因不是：

```text
GP 太强
cost 太强
dual 太强
reward 权重太低
```

而是：

```text
reward log_prob 的 action 梯度路径错了
```

---

## 8. 修复后的公式

修复后的核心思想是：

```text
同一个 sampled action，
对 reward log_prob 来说要 detach，
对 GP risk 来说可以不 detach。
```

也就是使用两个视角：

```text
a_t = rsample action
```

reward 用：

```text
a_reward = stopgrad(a_t)
```

GP risk 用：

```text
a_risk = a_t
```

修复后的 reward loss：

```text
L_reward
= - E[
      w_t
      * q(stopgrad(g_t))
      * stopgrad(A_t)
      * log pi_theta(stopgrad(a_t) | z_t)
   ]
```

修复后的 risk loss：

```text
g_t = Gp(z_t, a_t)
```

```text
L_risk
= E[
      w_t
      * lambda(g_t)
      * risk_penalty(g_t)
   ]
```

最终：

```text
L_fdpi
= L_reward
   + L_risk
   - beta * E[H]
```

这样做的好处是：

```text
reward 部分：
    正确训练策略 mean，让高 reward 动作概率变大

GP risk 部分：
    仍然可以通过 action 对 actor 反传，让动作往低风险方向移动
```

这才是我们真正想要的 FDPI 更新。

---

## 9. 用一句代码解释修复

原来是：

```python
main_action = dist.rsample()
log_prob = dist.log_prob(main_action)
```

修复后是：

```python
main_action = dist.rsample()
log_prob_action = main_action.detach()
log_prob = dist.log_prob(log_prob_action)
```

注意：

```text
main_action.detach()
```

只用于 reward 的 `log_prob`。

GP 仍然用：

```python
g = gp_critic.risk(feat, main_action)
```

所以 GP 对 actor 的 action 梯度没有被关掉。

---

## 10. 梯度检查结果

我做过一个简单梯度检查。

结果是：

```text
rsample 不 detach:
    mean_grad_norm = 0.0

rsample detach:
    mean_grad_norm = 0.1236

普通 fixed action:
    mean_grad_norm = 0.3140
```

这说明：

```text
不 detach 时，reward 几乎没有办法训练动作 mean
detach 后，mean 梯度恢复
```

这和策略崩塌现象完全对应。

---

## 11. 消融实验结果

从同一个 checkpoint：

```text
full_state_v4_1499904.pth
```

恢复训练到约 1.53M。

| 实验 | 设置 | 结果 |
|---|---|---|
| 原始 v4 | 原 FDPI 写法 | 崩塌，total 约 0.142，contact 0 |
| A | Dreamer-only，不进 FDPI | 不崩，total 约 6.42 |
| B | FDPI reward-only，不加 GP penalty | 仍崩，total 约 0.146 |
| D | FDPI reward-only + detach log_prob action | 不再永久崩，total 约 7.51 |
| E | 原始 FDPI GP 设置 + detach log_prob action | 不再永久崩，total 约 9.21 |

这个对比说明：

```text
B 崩，D 不崩
```

所以问题主要来自：

```text
log_prob(rsample_action) 没有 detach
```

而不是 GP penalty 本身。

---

## 12. 为什么 action anchor 也能缓解，但不是根修复

之前 C2 加了 action anchor：

```text
L_anchor = eta * || main_action - old_imagined_action ||^2
```

它能缓解崩塌，因为它额外给 actor 一个动作约束：

```text
不要一下子偏离原来的动作太远
```

但它没有修复 reward policy gradient 的根问题。

所以 C2 会比原始稳定一些，但不如 detach 修复干净。

真正的根修复仍然是：

```text
reward log_prob 中的 sampled action 必须 detach
```

---

## 13. 当前代码中的修复状态

当前 v4 已经默认开启：

```yaml
FDPIRegimeDreamer:
  MainFDPIRegime:
    DetachActionForLogProb: true
```

也就是说，正常运行 v4 训练时，会默认使用修复后的公式。

如果需要做旧行为消融，可以在代码层面把它关掉；但正常训练不建议关闭。

---

## 14. 最后再用最简单的话总结

普通 policy gradient 应该是：

```text
我先固定一个动作 a
然后问策略：你以后要不要更容易做出这个动作？
```

原 FDPI 写成了：

```text
我让动作 a 跟着策略参数一起变
然后再问策略：你以后要不要更容易做出这个动作？
```

这样会导致：

```text
动作 mean 的 reward 学习信号被抵消
```

所以策略失去“朝高 reward 动作学习”的能力。

修复后变回：

```text
reward 评价固定动作
GP risk 仍然优化可导动作
```

这就是为什么修复后策略不再出现原来那种一开 GP/FDPI 就快速崩塌的现象。
