# FDPI 更新原理与当前实现说明

本文档根据当前工程中的 PyTorch 实现整理：

- 训练入口：`fdpi_isaaclab_minimal/train.py`
- 训练循环：`fdpi_isaaclab_minimal/fdpi/trainer.py`
- 算法更新：`fdpi_isaaclab_minimal/fdpi/sac_fpi_dual.py`
- replay buffer：`fdpi_isaaclab_minimal/fdpi/replay_buffer.py`

这里的 FDPI 不是一个单独的纯公式实现，而是和 SAC 风格 actor-critic、双策略采样、约束/安全 cost critic、importance sampling 权重结合在一起的安全强化学习实现。代码里的核心类名是 `TorchSACFPIDual`。

## 1. 整体思想

当前实现可以理解成：

1. 用 SAC 的方式学习奖励价值函数 `Q` 和主策略 `pi`。
2. 额外学习安全 cost 价值函数 `G`，用于估计未来发生约束违反的风险。
3. 再学习一个恢复/安全回归价值函数 `GR`，用于在已经违反安全约束的样本上指导策略恢复。
4. 使用一个 dual policy `dual_pi` 主动寻找高风险动作，帮助主策略看见约束边界附近或不安全区域。
5. 主策略和 dual policy 之间通过 importance sampling 权重修正数据分布。
6. dual policy 又通过双向 KL 约束限制，不让它离主策略太远。

因此它不是简单地把 reward 和 cost 加权求和，而是把状态/动作分成几种区域：

- feasible：预测 cost 风险低于阈值 `pf`，且当前样本不是 violation。
- critical：仍然 feasible，但已经接近风险边界。
- infeasible：预测风险不满足 feasible 条件，但当前样本还不是直接 violation。
- violation：当前 transition 的 cost 已经大于 0。

主策略在这些区域使用不同目标：

- feasible 区域主要最大化 reward。
- critical/infeasible 区域最大化 reward 的同时压低 cost 风险。
- violation 区域优先最大化恢复价值 `GR`。
- 所有区域仍保留 SAC entropy 项，避免策略过早塌缩。

## 2. 主要网络和参数

`TorchSACFPIDual` 中包含这些网络：

| 组件 | 代码名 | 作用 |
| --- | --- | --- |
| 主奖励 critic | `q1`, `q2` | 估计 reward soft Q-value |
| 主奖励 target critic | `target_q1`, `target_q2` | 稳定 TD target |
| 主 cost critic | `g1`, `g2` | 估计未来 violation/cost 风险 |
| 主 cost target critic | `target_g1`, `target_g2` | 稳定 cost TD target |
| 恢复 critic | `gr1`, `gr2` | 估计违反约束后的恢复/安全回归价值 |
| 恢复 target critic | `target_gr1`, `target_gr2` | 稳定恢复 TD target |
| 主策略 | `pi` | 实际希望学好的策略 |
| dual 策略 | `dual_pi` | 主动寻找高风险或边界附近行为 |
| dual cost critic | `dual_g1`, `dual_g2` | 服务 dual policy 的 cost critic |
| dual cost target critic | `dual_target_g1`, `dual_target_g2` | dual cost 的 target critic |

关键标量参数：

| 参数 | 代码名 | 含义 |
| --- | --- | --- |
| reward discount | `gamma` | 奖励 Q 的折扣因子 |
| cost discount | `cost_gamma` | cost / recovery 价值的折扣因子 |
| target update rate | `tau` | target network 软更新系数 |
| entropy temperature | `alpha = exp(log_alpha)` | SAC entropy 权重 |
| cost threshold | `pf` | 允许的风险阈值，配置中来自 `epsilon` |
| critical margin | `cg = exp(log_cg)` | feasible 与 critical 的边界宽度 |
| main cost multipliers | `lam1`, `lam2` | critical/infeasible 区域的约束乘子 |
| dual KL multipliers | `lam3`, `lam4` | dual policy 和 main policy 的 KL 约束乘子 |
| KL target | `target_kl` | dual policy 允许偏离主策略的目标 KL |

## 3. 采样阶段的 dual policy 与 IS 权重

训练循环在 `FDPIIsaacLabTrainer.train()` 中。环境数量必须是偶数，因为采样时会把并行环境分成两半：

- 前半环境执行主策略 `pi`。
- 后半环境执行 dual 策略 `dual_pi`。

当 replay 还没有进入训练阶段，即 `sample_steps < start_steps`，所有环境都执行随机动作。

进入训练阶段后：

```text
act, dual_act, log_weight, log_weight_dual = algorithm.act(obs, dual_active)
action = concat(
  act[first_half],
  dual_act[second_half],
)
```

`algorithm.act()` 会同时从主策略和 dual 策略采样，并计算两组 log importance weights：

```text
log_weight      = log dual_pi(a_main | s) - log pi(a_main | s)
log_weight_dual = log pi(a_dual | s)     - log dual_pi(a_dual | s)
```

也就是：

- 主策略采出的动作，可以被加权后当作 dual policy 分布下的样本。
- dual policy 采出的动作，可以被加权后当作主策略分布下的样本。

代码里还有一个累计权重：

```text
cumulative_log_weight      = beta * (cumulative_log_weight + log_weight)
cumulative_log_weight_dual = beta * (cumulative_log_weight_dual + log_weight_dual)
```

`beta` 会衰减长轨迹上的累计 log weight，避免重要性权重过度爆炸。最终 replay buffer 里存的是 transition：

```text
(obs, action, next_obs, reward, cost, done, log_weight, log_weight_dual)
```

需要注意一个命名细节：在 `update()` 里：

```python
weight = exp(data.log_weight_dual)
dual_weight = exp(data.log_weight)
```

也就是说：

- `weight` 用于主策略、主 Q、主 G、GR 更新，目的是把 dual 采样的数据修正回主策略分布。
- `dual_weight` 用于 dual cost critic 和 dual policy 更新，目的是把主策略采样的数据修正到 dual 分布。

## 4. `update()` 输入数据

`update(data)` 接收一个 `ExperienceBatch`：

```python
obs
action
next_obs
reward
cost
done
log_weight
log_weight_dual
```

记号如下：

- `s = obs`
- `a = action`
- `s' = next_obs`
- `r = reward`
- `c = cost`
- `d = done`
- `w = exp(log_weight_dual)`，主分支权重
- `w_dual = exp(log_weight)`，dual 分支权重

其中 `done` 在 trainer 中来自：

```python
bootstrap_done = terminated & ~truncated
```

这表示真正终止的 episode 会截断 bootstrap，而 time-limit truncated 的 episode 仍允许 bootstrap。

## 5. 第一步：更新 reward critic `Q`

代码首先用主策略在下一状态采样动作：

```python
next_action, next_logp = self.pi.sample(next_obs)
```

TD target 使用 SAC 的 clipped double Q：

```text
target_q = min(target_q1(s', a'), target_q2(s', a'))
q_backup = r + (1 - d) * gamma * (target_q - alpha * log pi(a' | s'))
```

对应代码：

```python
target_q = min(target_q1(next_obs, next_action), target_q2(next_obs, next_action))
q_backup = reward + (1 - done) * gamma * (target_q - alpha * next_logp)
```

然后分别更新 `q1` 和 `q2`：

```text
L_Qi = mean( w * (Qi(s, a) - q_backup)^2 )
```

这里的 `w` 是主分支 importance weight。因为 replay 中同时有主策略样本和 dual policy 样本，所以主 critic 更新时要把 dual policy 采样的数据修正回主策略分布。

## 6. 第二步：更新主 cost critic `G`

`G` 用来估计未来发生 cost/violation 的风险。target 使用两个 target cost critic 的较大值：

```text
target_g = max(target_g1(s', a'), target_g2(s', a'))
target_g = clamp(target_g, 0, 1)
g_backup = c + (1 - d) * (1 - c) * cost_gamma * target_g
```

解释一下这个 backup：

- 如果当前 `c = 1`，说明当前已经 violation，那么 `g_backup = 1`。
- 如果当前 `c = 0`，则继续看未来的 violation 风险。
- 使用 `max(g1, g2)` 是偏保守的估计：两个 cost critic 里谁更悲观就听谁的。
- clamp 到 `[0, 1]`，因为这里的 cost/risk 被当作概率或归一化风险来用。

loss：

```text
L_Gi = mean( w * (Gi(s, a) - g_backup)^2 )
```

## 7. 第三步：更新恢复 critic `GR`

`GR` 用于 violation 区域。主策略在当前 transition 已经违反约束时，不再优先追求 reward，而是优先让 `GR` 变大。

target：

```text
target_gr = min(target_gr1(s', a'), target_gr2(s', a'))
target_gr = clamp(target_gr, 0, 1)
gr_backup = (1 - c) + (1 - d) * c * cost_gamma * target_gr
```

直观理解：

- 当前没有 cost 时，恢复价值直接是高的：`1 - c = 1`。
- 当前有 cost 时，就看后续能否恢复。
- 使用 `min(gr1, gr2)` 是保守估计：避免过度乐观地认为自己能恢复。

loss：

```text
L_GRi = mean( w * (GRi(s, a) - gr_backup)^2 )
```

## 8. 第四步：更新主策略 `pi`

主策略从当前状态采样动作：

```python
pi_action, logp = self.pi.sample(obs)
```

然后计算三个评价量：

```text
q_pi  = min(q1(s, a_pi),  q2(s, a_pi))
g_pi  = max(g1(s, a_pi),  g2(s, a_pi))
gr_pi = min(gr1(s, a_pi), gr2(s, a_pi))
```

这里：

- `q_pi` 是保守 reward value。
- `g_pi` 是悲观 risk value。
- `gr_pi` 是保守 recovery value。

接着划分区域：

```python
vio = cost > 0
fea = (g_pi < pf) & ~vio
cri = fea & (g_pi >= pf - cg)
```

含义：

- `vio`：当前 replay transition 已经有 cost。
- `fea`：当前策略动作的风险预测低于阈值 `pf`，且当前样本不是 violation。
- `cri`：虽然 feasible，但 `g_pi` 已经接近 `pf`，即安全边界附近。
- `~fea & ~vio`：还没直接 violation，但当前动作预测风险偏高。

主策略 loss 分成四块：

```text
loss_fea = 1[fea and not cri] * (-q_pi)
loss_cri = 1[cri]             * (-q_pi + lam1 * g_pi) / (lam1 + 1)
loss_inf = 1[not fea and not vio] * (-q_pi + lam2 * g_pi) / (lam2 + 1)
loss_vio = 1[vio]             * (-gr_pi)
```

再加 entropy 项：

```text
L_pi = mean( w * (loss_fea + loss_cri + loss_inf + loss_vio + alpha * log_pi) )
```

因为优化器是在最小化 loss：

- `-q_pi` 会推动策略最大化 reward。
- `lam * g_pi` 会推动策略降低 risk。
- `-gr_pi` 会推动 violation 样本上的动作提高 recovery value。
- `alpha * log_pi` 是 SAC entropy 项，通常 `log_pi` 为负，因此最小化它会鼓励更高 entropy。

## 9. 第五步：更新 entropy temperature `alpha`

如果 `auto_alpha=True`，更新：

```text
L_alpha = - mean( log_alpha * (log_pi + target_entropy) )
```

这和 SAC 的自动温度调节一致：

- entropy 太低时，提高 `alpha`，更重视探索。
- entropy 太高时，降低 `alpha`，更重视价值目标。

## 10. 第六步：更新 `cg`, `lam1`, `lam2`

这一步负责调节主策略在约束边界附近的行为。

代码会重新从主策略采样一个 `new_action`，计算：

```text
new_g = max(g1(s, new_action), g2(s, new_action))
```

然后得到几个信号：

```text
fea_ratio = mean(fea)
cri_ratio = mean(cri)
vio_new = fea and (new_g > pf)
vio_ratio = mean(vio_new)
```

### 10.1 `cg`

`cg = exp(log_cg)` 是 critical 区域的宽度。critical 区域定义为：

```text
pf - cg <= g_pi < pf
```

代码中的更新信号：

```text
delta_cg = masked_mean(leaky_relu((pf - g_pi) - cg), vio_new)
```

如果 feasible 状态下新采样动作越过了风险阈值，`cg` 会收到调节信号，让 critical 区域对边界风险更敏感。若有 feasible 样本且没有新 violation，代码还会额外加入一个让 `cg` 收缩的信号，避免 critical 区域无限变宽。

优化目标：

```text
L_cg = - log_cg * stop_gradient(delta_cg)
```

### 10.2 `lam1`

`lam1` 控制 critical 区域里 risk 惩罚的强度。更新信号：

```text
fea_g_vio = masked_mean(leaky_relu(new_g - pf), cri)
L_lam1 = - lam1 * stop_gradient(fea_g_vio)
lam1 = max(lam1, 0)
```

如果 critical 状态的新动作越过 `pf`，`lam1` 会增大，使主策略在 critical 区域更重视降低 `g_pi`。

### 10.3 `lam2`

`lam2` 控制 infeasible 区域里 risk 惩罚的强度。更新信号：

```text
inf_g_inc = masked_mean(leaky_relu(new_g - g_pi), not feasible)
L_lam2 = - lam2 * stop_gradient(inf_g_inc)
lam2 = max(lam2, 0)
```

如果 infeasible 状态的新动作让风险继续上升，`lam2` 会增大，使策略更强地压低 `g_pi`。

## 11. 第七步：更新 dual cost critic

dual 分支使用 `dual_pi` 在下一状态采样动作：

```text
a'_dual ~ dual_pi(s')
dual_target_g = min(dual_target_g1(s', a'_dual), dual_target_g2(s', a'_dual))
dual_g_backup = c + (1 - d) * (1 - c) * cost_gamma * dual_target_g
```

这里使用 `min` 而不是主 cost critic 的 `max`。主分支要保守地避免风险，因此对 risk 用 `max`；dual 分支的目标是训练一个能寻找高风险区域的策略，但 critic target 本身使用 clipped double 形式降低过估计。

loss：

```text
L_dual_Gi = mean( w_dual * (dual_Gi(s, a) - dual_g_backup)^2 )
```

其中 `w_dual = exp(log_weight)`。

## 12. 第八步：更新 dual policy

dual policy 的目标是寻找更高 cost/risk 的动作，但不能离主策略过远。

先采样 dual 动作：

```text
a_dual ~ dual_pi(s)
dual_g = min(dual_g1(s, a_dual), dual_g2(s, a_dual))
```

dual policy loss：

```text
L_dual_pi =
  mean(w_dual * -dual_g)
  + lam3 * KL(dual_pi || pi)
  + lam4 * KL(pi || dual_pi)
```

因为优化器最小化 loss，`-dual_g` 会推动 dual policy 最大化 risk。两个 KL 项限制 dual policy 不要跑得离主策略太远。

代码中的两个 KL 近似是：

```text
kl      = E_{a ~ dual_pi}[log dual_pi(a|s) - log pi(a|s)]
dual_kl = E_{a ~ pi}     [log pi(a|s)      - log dual_pi(a|s)]
```

然后更新 `lam3`, `lam4`：

```text
L_lam3 = lam3 * (target_kl - kl)
L_lam4 = lam4 * (target_kl - dual_kl)
lam3 = max(lam3, 0)
lam4 = max(lam4, 0)
```

如果实际 KL 超过 `target_kl`，梯度下降会增大对应的 lambda，从而下一次更强地惩罚 KL。

## 13. 第九步：软更新 target networks

最后执行：

```text
target = (1 - tau) * target + tau * online
```

被更新的 target 网络包括：

- `target_q1`, `target_q2`
- `target_g1`, `target_g2`
- `target_gr1`, `target_gr2`
- `dual_target_g1`, `dual_target_g2`

## 14. `update()` 返回的监控指标

`update()` 最后返回一个 `info` 字典，trainer 会写入 TensorBoard。常用指标包括：

| 指标 | 含义 |
| --- | --- |
| `q1_loss`, `q2_loss` | reward critic 损失 |
| `g1_loss`, `g2_loss` | 主 cost critic 损失 |
| `gr1_loss`, `gr2_loss` | recovery critic 损失 |
| `pi_loss` | 主策略损失 |
| `entropy` | 主策略 entropy 估计 |
| `alpha` | SAC temperature |
| `feasible_ratio` | batch 中 feasible 样本比例 |
| `critical_ratio` | batch 中 critical 样本比例 |
| `feasible_g_violation_ratio` | feasible 状态下新动作越过 `pf` 的比例 |
| `cg` | critical margin |
| `lam1`, `lam2` | 主策略安全约束乘子 |
| `dual_g1_loss`, `dual_g2_loss` | dual cost critic 损失 |
| `dual_pi_loss` | dual 策略损失 |
| `kl`, `dual_kl` | dual/main 双向 KL 估计 |
| `lam3`, `lam4` | KL 约束乘子 |
| `violate_ratio` | batch 中 cost 的均值 |
| `IS_weight`, `IS_weight_dual` | importance weight 均值 |

trainer 会把 `feasible_ratio` 放入一个滑动窗口。窗口均值超过 `dual_thresh` 时，`dual_active=True`，后续采样阶段会实际启用 dual policy 的后半环境动作。

## 15. FDPI 训练循环伪代码

下面是当前 `FDPIIsaacLabTrainer.train()` 的高层伪代码：

```text
initialize env, algorithm, replay_buffer
obs = env.reset(seed)

sample_steps = 0
update_steps = 0
feasible_window = deque()
cumulative_log_weight = zeros(num_envs)
cumulative_log_weight_dual = zeros(num_envs)

while sample_steps < total_steps:
    dual_active = mean(feasible_window) > dual_thresh

    repeat sample_per_iteration times:
        if sample_steps < start_steps:
            action = uniform_random_action(-1, 1)
            log_weight = zeros(num_envs)
            log_weight_dual = zeros(num_envs)
        else:
            main_action, dual_action, log_weight, log_weight_dual =
                algorithm.act(obs, dual_active)

            action[first_half] = main_action[first_half]
            action[second_half] = dual_action[second_half]

        next_obs_dict, reward, terminated, truncated, extras = env.step(action)

        done_for_episode = terminated or truncated
        done_for_bootstrap = terminated and not truncated

        next_obs = policy_obs(next_obs_dict)
        cost = extract_safety_cost(extras, next_obs_dict)
        real_next_obs = terminal_next_obs(next_obs, extras)

        replay_buffer.add_batch(
            obs,
            action,
            real_next_obs,
            reward,
            cost,
            done_for_bootstrap,
            cumulative_log_weight,
            cumulative_log_weight_dual,
        )

        update episode return / cost / length statistics

        if sample_steps >= start_steps:
            cumulative_log_weight[first_half] =
                beta * (cumulative_log_weight[first_half] + log_weight[first_half])

            cumulative_log_weight_dual[second_half] =
                beta * (
                    cumulative_log_weight_dual[second_half]
                    + log_weight_dual[second_half]
                )

            reset cumulative weights for environments whose episode ended

        obs = next_obs
        sample_steps += num_envs

    if sample_steps >= start_steps and len(replay_buffer) >= batch_size:
        repeat updates_per_iteration times:
            batch = replay_buffer.sample(batch_size)
            info = algorithm.update(batch)
            feasible_window.append(info["feasible_ratio"])
            update_steps += 1

    if sample_steps >= next_log_step:
        write TensorBoard scalars
        print progress

    if sample_steps >= next_save_step:
        save algorithm checkpoint

flush logs
save final checkpoint
```

## 16. `update()` 详细伪代码

下面是 `TorchSACFPIDual.update()` 的结构化伪代码：

```text
function update(batch):
    s, a, s_next, r, c, done = batch fields
    w_main = exp(batch.log_weight_dual)
    w_dual = exp(batch.log_weight)

    # 1. reward critic
    a_next, logp_next = pi.sample(s_next)
    target_q = min(target_q1(s_next, a_next), target_q2(s_next, a_next))
    y_q = r + (1 - done) * gamma * (target_q - alpha * logp_next)

    minimize mean(w_main * (q1(s, a) - y_q)^2)
    minimize mean(w_main * (q2(s, a) - y_q)^2)

    # 2. main cost critic
    target_g = max(target_g1(s_next, a_next), target_g2(s_next, a_next))
    target_g = clamp(target_g, 0, 1)
    y_g = c + (1 - done) * (1 - c) * cost_gamma * target_g

    minimize mean(w_main * (g1(s, a) - y_g)^2)
    minimize mean(w_main * (g2(s, a) - y_g)^2)

    # 3. recovery critic
    target_gr = min(target_gr1(s_next, a_next), target_gr2(s_next, a_next))
    target_gr = clamp(target_gr, 0, 1)
    y_gr = (1 - c) + (1 - done) * c * cost_gamma * target_gr

    minimize mean(w_main * (gr1(s, a) - y_gr)^2)
    minimize mean(w_main * (gr2(s, a) - y_gr)^2)

    # 4. main policy
    a_pi, logp = pi.sample(s)
    q_pi = min(q1(s, a_pi), q2(s, a_pi))
    g_pi = max(g1(s, a_pi), g2(s, a_pi))
    gr_pi = min(gr1(s, a_pi), gr2(s, a_pi))

    vio = c > 0
    feasible = (g_pi < pf) and not vio
    critical = feasible and (g_pi >= pf - cg)
    infeasible = not feasible and not vio

    policy_objective =
        feasible_not_critical * (-q_pi)
        + critical * (-q_pi + lam1 * g_pi) / (lam1 + 1)
        + infeasible * (-q_pi + lam2 * g_pi) / (lam2 + 1)
        + violation * (-gr_pi)
        + alpha * logp

    minimize mean(w_main * policy_objective)

    # 5. entropy temperature
    if auto_alpha:
        minimize -mean(log_alpha * stop_gradient(logp + target_entropy))

    # 6. adaptive margin and main safety multipliers
    new_action ~ pi(s)
    new_g = max(g1(s, new_action), g2(s, new_action))

    vio_new = feasible and (new_g > pf)
    delta_cg = masked_mean(leaky_relu((pf - g_pi) - cg), vio_new)
    if feasible exists and no vio_new:
        delta_cg += leaky_relu(-cg)

    fea_g_vio = masked_mean(leaky_relu(new_g - pf), critical)
    inf_g_inc = masked_mean(leaky_relu(new_g - g_pi), not feasible)

    minimize -log_cg * stop_gradient(delta_cg)
    minimize -lam1 * stop_gradient(fea_g_vio)
    lam1 = max(lam1, 0)
    minimize -lam2 * stop_gradient(inf_g_inc)
    lam2 = max(lam2, 0)

    # 7. dual cost critic
    a_next_dual ~ dual_pi(s_next)
    dual_target_g =
        min(dual_target_g1(s_next, a_next_dual),
            dual_target_g2(s_next, a_next_dual))
    dual_target_g = clamp(dual_target_g, 0, 1)
    y_dual_g = c + (1 - done) * (1 - c) * cost_gamma * dual_target_g

    minimize mean(w_dual * (dual_g1(s, a) - y_dual_g)^2)
    minimize mean(w_dual * (dual_g2(s, a) - y_dual_g)^2)

    # 8. dual policy
    a_dual ~ dual_pi(s)
    dual_g = min(dual_g1(s, a_dual), dual_g2(s, a_dual))

    kl_dual_main = E_{a ~ dual_pi}[log dual_pi(a|s) - log pi(a|s)]
    kl_main_dual = E_{a ~ pi}[log pi(a|s) - log dual_pi(a|s)]

    minimize mean(w_dual * -dual_g)
             + lam3 * kl_dual_main
             + lam4 * kl_main_dual

    minimize lam3 * stop_gradient(target_kl - kl_dual_main)
    lam3 = max(lam3, 0)
    minimize lam4 * stop_gradient(target_kl - kl_main_dual)
    lam4 = max(lam4, 0)

    # 9. target networks
    soft_update(q targets, g targets, gr targets, dual_g targets)

    return logging_info
```

## 17. 运行入口伪代码

`fdpi_isaaclab_minimal/train.py` 的入口可以概括成：

```text
preparse --config
load YAML config groups as argparse defaults
parse CLI arguments
start IsaacLab AppLauncher

main:
    optionally add ur3_lite extension path
    import task package

    env_cfg = parse_env_cfg(task, device, num_envs, use_fabric)
    env = gym.make(task, cfg=env_cfg)

    obs = env.reset(seed)
    obs_dim = dim(obs["policy"])
    act_dim = flatdim(single_action_space)

    algorithm = TorchSACFPIDual(
        obs_dim,
        act_dim,
        gamma,
        cost_gamma,
        tau,
        lr,
        epsilon as pf,
        target_kl,
    )

    replay_buffer = TorchReplayBufferIS(obs_dim, act_dim, buffer_size)

    trainer = FDPIIsaacLabTrainer(
        env,
        algorithm,
        replay_buffer,
        total_steps,
        start_steps,
        beta,
        dual_thresh,
        cost extraction config,
        logging/checkpoint config,
    )

    trainer.train(seed)

finally:
    env.close()
    simulation_app.close()
```

实际运行时仍应使用 IsaacLab launcher，例如：

```bash
/home/yhy/IsaacLab-1.4.0/isaaclab.sh -p fdpi_isaaclab_minimal/train.py
```

## 18. 读代码时最容易混淆的点

1. `epsilon` 在配置里传入算法后叫 `pf`，它表示 cost/risk 阈值，不是常见 epsilon-greedy。
2. `log_weight` 和 `log_weight_dual` 在 `update()` 中看起来是反着用的；这是因为主分支要把 dual 样本修正回主策略分布，dual 分支要把主策略样本修正到 dual 分布。
3. 主 cost critic 用 `max(g1, g2)`，因为主策略要保守避险。
4. recovery critic 用 `min(gr1, gr2)`，因为违反约束后不能过度乐观。
5. dual policy 的目标是最大化 risk，但它被 `lam3/lam4` 的双向 KL 约束拉住，避免跑到离主策略太远、完全无关的数据分布。
6. trainer 中 `done_for_bootstrap = terminated & ~truncated`，所以 time-limit 截断不会让 TD target 停止 bootstrap。
