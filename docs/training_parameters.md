# PWM 在线训练参数说明

本文档说明 `pwm_isaaclab/config_files/PWM.yaml` 中各训练参数在在线联合训练路径里的含义。范围只覆盖 `pwm_isaaclab/train.py` 启动的 `JointTrainAgent` 流程。

## 训练主流程

在线训练入口是 `pwm_isaaclab/train.py`。脚本会先读取 YAML 配置和命令行参数，然后创建 IsaacLab 向量环境、世界模型、Actor-Critic agent 和 replay buffer，最后进入 `trainer.py` 里的采样-训练循环。

整体流程如下：

1. `load_config(config_path)` 读取 YAML，并检查 `Task == "JointTrainAgent"`。
2. `build_env(args, conf)` 通过 `parse_env_cfg()` 创建 IsaacLab 环境，并用 `DreamerVecEnvWrapper` 包装成 Dreamer 风格的 batched observation。
3. 从环境空间读取 `obs_dim` 和 `action_dim`，而不是使用 YAML 里的 `ObsShape`。
4. `build_world_model()` 创建 `ParallelWorldModel`，用于编码状态、学习 dynamics、预测 reward/done、生成 imagined rollout。
5. `build_agent()` 创建 `ActorCriticAgent`，在世界模型的 latent feature 上学习 actor 和 critic。
6. `ProprioReplayBuffer` 保存并行环境采样到的 `obs/action/reward/done/is_first` 序列。
7. replay buffer 未超过 `BufferWarmUp` 前，动作来自环境 action space 的随机采样；超过 warmup 后，动作来自 world model inference feature 加 actor 采样。
8. 每个 vector iteration 会向环境推进 `num_envs` 个环境步，并按配置频率训练 world model 和 agent。
9. checkpoint 保存到 `ckpt/{run_name}/world_model_{step}.pth` 和 `ckpt/{run_name}/agent_{step}.pth`。

## 顶层配置

| 参数 | 当前值 | 实际作用 |
| --- | --- | --- |
| `Task` | `"JointTrainAgent"` | 选择训练任务。当前 `train.py` 只实现了 `JointTrainAgent`，其他值会抛出 `NotImplementedError`。 |

## `BasicSettings`

| 参数 | 当前值 | 实际作用 |
| --- | --- | --- |
| `Seed` | `0` | 在当前在线训练路径中不直接使用。实际随机种子来自命令行 `-seed`，并传给 `seed_np_torch()`；环境 seed 默认也来自 `-seed`，但可被 `Env.MakeKwargs.seed` 覆盖。 |
| `ObsShape` | `0` | 在当前在线训练路径中不直接使用。`obs_dim` 由 `vec_env.single_observation_space["policy"].shape[0]` 自动读取。 |
| `UseAmp` | `True` | 是否启用自动混合精度。传入 world model 和 agent 后，会让 autocast 使用 `float16`，并启用 `GradScaler`。通常能降低显存和提升速度，但数值不稳定时可改为 `False`。 |
| `FrameSkip` | `1` | 在当前 `pwm_isaaclab/train.py` 在线训练路径中不直接使用。它在部分 eval 路径中会被读取。 |

## `JointTrainAgent`

这些参数控制在线采样、replay buffer、world model 更新、agent 更新和保存频率。

| 参数 | 当前值 | 实际作用 |
| --- | --- | --- |
| `SampleMaxSteps` | `3000000` | 总环境步数上限。代码中 `total_iters = SampleMaxSteps // num_envs`，因此每轮 vector iteration 推进 `num_envs` 步。当前 `num_envs=64` 时大约运行 `46875` 轮。 |
| `BufferMaxLength` | `2000000` | replay buffer 最多保存的环境 transition 数。内部 shape 是 `(BufferMaxLength // num_envs, num_envs, dim)`，满了以后按时间维循环覆盖旧数据。 |
| `BufferWarmUp` | `6400` | replay buffer 中 transition 数超过该值后才开始训练并使用 agent 采样动作。warmup 前使用随机动作。当前 `64` 个环境下，大约收集 `100` 轮后进入训练。 |
| `NumEnvs` | `64` | 配置期望的并行环境数量。实际环境数优先来自 `Env.MakeKwargs.num_envs`，若两者不一致，代码会打印 warning 并使用环境实际值。 |
| `BatchSize` | `64` | 从 replay buffer 采样序列的 batch size。必须 `>= num_envs` 且能被 `num_envs` 整除，因为 buffer 会为每个 env 均匀采样 `BatchSize // num_envs` 条序列。 |
| `BatchLength` | `64` | world model 每次更新使用的真实序列长度。越长越能学习长时序依赖，但显存和计算成本更高。采样窗口不会跨 episode 边界。 |
| `ImagineBatchSize` | `64` | agent 更新时，从 replay buffer 采样多少条 context 序列作为 imagination 起点。若配置为 `<= 0`，会回退为 `BatchSize`。同样需要满足 replay buffer 的 batch size 约束。 |
| `ImagineContext` | `16` | agent imagination 前用于让 world model observe 的真实上下文长度。若配置为 `<= 0`，会回退为 `BatchLength`。 |
| `ImagineHorizon` | `8` | world model 在 latent space 中向前想象的步数，agent 在这段 imagined rollout 上计算 lambda return 并更新 actor/critic。越大越看重长远收益，但模型误差也会累积。 |
| `TrainModelEverySteps` | `1` | world model 更新频率，代码会折算成 `train_model_every_iters = max(TrainModelEverySteps // num_envs, 1)`。当前值小于 `num_envs`，等价于每个 vector iteration 都尝试训练一次。 |
| `TrainAgentEverySteps` | `1` | agent 更新频率，折算方式同上。当前值等价于每个 vector iteration 都尝试训练一次。 |
| `ModelUpdate` | `1` | 每次触发 world model 训练时执行多少次 optimizer update。代码会用 `max(int(ModelUpdate), 1)` 保证至少一次。 |
| `AgentUpdate` | `1` | 每次触发 agent 训练时执行多少次 optimizer update。代码会用 `max(int(AgentUpdate), 1)` 保证至少一次。 |
| `SaveEverySteps` | `100000` | checkpoint 保存频率，同样会折算成 `save_every_iters = max(SaveEverySteps // num_envs, 1)`。注意 `iter_idx=0` 也会保存一次初始模型。 |
| `VideoLogStep` | `2000` | 传入 world model，用于控制视觉任务的 observation/reconstruction/imagination 视频日志频率。但当前 `train.py` 构建 world model 时固定传入 `is_proprio=True`，因此在线 proprio 训练中基本不会产生日志视频。 |
| `SaveOfflineEpisodes` | `False` | 是否把在线采样 episode 额外保存为 `.npz` 离线数据。命令行 `--save_offline_episodes` 或非空 `OfflineDatasetDir` 也会开启该功能。 |
| `OfflineDatasetDir` | `""` | 离线 episode 输出目录。若开启保存但未提供目录，默认写到 `ckpt/{run_name}/offline_episodes`。保存内容包含 `obs/action/reward/done/is_first`，如果 observation 中存在 `force` 也会额外保存。 |

## `Models`

这些参数定义 world model 与 agent 共用的 latent 维度、value/reward 分布表示和时序回报设置。

| 参数 | 当前值 | 实际作用 |
| --- | --- | --- |
| `Hidden` | `512` | deterministic hidden state、MLP 隐层和多处网络模块的宽度。world model feature 维度为 `Hidden + Stoch * Discrete`。当前为 `512 + 32 * 32 = 1536`。 |
| `NumBin` | `255` | reward 和 value 使用 symlog two-hot 分布时的 bin 数。world model 的 reward head 和 agent 的 critic 都输出 `NumBin` 维 logits。 |
| `MaxBin` | `20` | symlog two-hot 的范围边界，实际范围是 `[-MaxBin, MaxBin]`。训练时 target 经过 `symlog` 后必须落在这个范围内。 |
| `Act` | `"SiLU"` | 激活函数名称，通过 `getattr(torch.nn, Act)` 取得，例如当前使用 `torch.nn.SiLU`。 |
| `Stoch` | `32` | stochastic latent 变量的组数。 |
| `Discrete` | `32` | 每组 stochastic latent 的离散类别数。latent stochastic 展平维度是 `Stoch * Discrete`。 |
| `Gamma` | `0.997` | 折扣因子。world model imagination 中的 discount 为预测未 done 时的 `Gamma`，agent 计算 lambda return 时也使用它。 |
| `Lambda` | `0.95` | lambda return 的 `lambda` 系数，用于平衡短期 bootstrap 和长程 Monte Carlo 风格回报。 |
| `Tau` | `0.01` | 当前会传入 world model 和 agent 并保存为成员变量，但在线更新逻辑中没有实际使用。代码里有 `slow_critic`，但未看到基于 `Tau` 的软更新。 |

## `Models.WorldModel`

world model 负责从真实 observation/action 序列中学习 latent dynamics，并预测 observation reconstruction、reward 和 done。

| 参数 | 当前值 | 实际作用 |
| --- | --- | --- |
| `Stem` | `32` | 视觉 encoder/decoder 的初始通道数。当前在线路径固定 `is_proprio=True`，使用 `ProprioEncoder/ProprioDecoder`，因此该参数在此路径中不影响网络结构。 |
| `MinRes` | `4` | 视觉 encoder/decoder 的最小空间分辨率。当前 proprio 在线路径中不影响网络结构。 |
| `DynScale` | `0.5` | dynamics KL loss 权重。world model loss 中 `model_loss = DynScale * dyn_loss + done_loss + reward_loss`。 |
| `RepScale` | `0.1` | representation KL loss 权重。world model loss 中 `vae_loss = recon_loss + RepScale * rep_loss`。 |
| `KLFree` | `1.0` | KL free bits 下限。`dyn_loss` 和 `rep_loss` 会被 `torch.clip(..., min=KLFree)`，避免 KL 过小导致 latent 表达退化。 |
| `LR` | `1e-4` | world model 的 AdamW 学习率，优化 dynamic、done head、reward head、encoder、decoder。 |
| `Eps` | `1e-8` | world model AdamW 的 epsilon。 |
| `ValScale` | `0.0` 默认值 | `PWM.yaml` 未显式设置该字段，但 `load_config()` 的默认配置结构中存在。`train.py` 会传入 world model 构造函数，world model 内部仅保存为 `self.val_scale`，在线更新逻辑没有使用它。 |

## `Models.Agent`

agent 在 world model imagined rollout 上更新 actor 和 critic。Actor 输出连续动作分布参数，critic 输出 symlog two-hot value logits。

| 参数 | 当前值 | 实际作用 |
| --- | --- | --- |
| `EntropyCoef` | `1e-3` | actor 熵奖励权重。总损失为 `critic_loss - policy_loss - EntropyCoef * entropy_loss`，增大通常会鼓励探索。 |
| `MinStd` | `0.1` | actor 连续动作分布标准差下界。代码中 `std = (MaxStd - MinStd) * sigmoid(raw_std + 2) + MinStd`。 |
| `MaxStd` | `1.0` | actor 连续动作分布标准差上界。更大的上界会允许更强探索，但动作噪声也更大。 |
| `LR` | `3e-5` | actor 和 critic 的 AdamW 学习率。 |
| `Eps` | `1e-5` | agent AdamW 的 epsilon。 |
| `MinPer` | `0.05` | 用于优势归一化尺度估计的低分位数。代码取 lambda return 的该分位数，并用 EMA 平滑。 |
| `MaxPer` | `0.95` | 用于优势归一化尺度估计的高分位数。`scale = max(upper_ema - lower_ema, 1.0)`。 |
| `EMADecay` | `0.99` | 优势归一化上下分位数的 EMA 衰减系数。越接近 1，尺度变化越平滑但响应越慢。 |

## `Wandb`

| 参数 | 当前值 | 实际作用 |
| --- | --- | --- |
| `Project` | `"IsaacLab-PSSM"` | `wandb.init(project=...)` 使用的项目名。 |
| `Group` | 未设置 | 可选字段。未设置时默认使用命令行 `-env_name`。 |
| `Name` | 未设置 | 可选字段。未设置时默认使用 `PSSM-{env_name}-seed{seed}`。 |
| `Mode` | 未设置 | 可选字段。设置后会传给 `wandb.init(mode=...)`，例如 `"offline"`。也可以用环境变量 `WANDB_MODE=offline` 控制。 |

## `Env.MakeKwargs`

| 参数 | 当前值 | 实际作用 |
| --- | --- | --- |
| `num_envs` | `64` | 传给 IsaacLab `parse_env_cfg()` 的并行环境数。它优先于 `JointTrainAgent.NumEnvs`。 |
| `use_fabric` | `True` | 传给 `parse_env_cfg()` 的 IsaacLab Fabric 开关。 |
| `seed` | 未设置 | 可选字段。设置后会覆盖命令行 `-seed` 作为环境 seed；不设置时环境 seed 使用 `-seed`。 |

## 调参建议

- `NumEnvs` 和 `Env.MakeKwargs.num_envs`：提高并行环境数能增加采样吞吐，但也会提高仿真、replay buffer 和 batch 约束压力。修改时要同步检查 `BatchSize` 是否仍然 `>= num_envs` 且能被整除。
- `BufferWarmUp`：太小会让模型和策略过早从低质量数据开始更新；太大会增加纯随机探索时间。当前 `6400` 对 `64` 环境约等于 `100` 个 vector iteration。
- `BufferMaxLength`：越大越能保留历史经验，但显存占用按 `obs_dim/action_dim/num_envs` 增长。当前 buffer 直接建在 `args.device` 上，使用 GPU 时需要特别关注显存。
- `BatchLength` 和 `ImagineContext`：`BatchLength` 影响 world model 学习真实序列的长度，`ImagineContext` 影响 agent imagination 的起始 latent 状态质量。增大它们通常更稳但更慢、更占显存。
- `ImagineHorizon`：越大越依赖世界模型的长程预测。模型还不准时，过大的 horizon 可能把策略带偏；模型较稳后可以适当增大以强化长远规划。
- `TrainModelEverySteps`、`TrainAgentEverySteps`、`ModelUpdate`、`AgentUpdate`：这些共同决定 update-to-data ratio。当前配置在每个 vector iteration 后各更新一次，属于比较密集的在线联合训练。
- `UseAmp`：GPU 上通常建议开启；如果出现 NaN、loss 爆炸或 reward/value two-hot 断言问题，可以先关闭 AMP 排查数值问题。
- `LR`：world model 学习率过大常表现为 `recon_loss/reward_loss/KL` 不稳定；agent 学习率过大常表现为 `policy_loss`、`entropy` 或回报剧烈波动。
- `EntropyCoef`、`MinStd`、`MaxStd`：控制探索强度。探索不足时可略增 `EntropyCoef` 或动作 std 范围；动作过抖或任务接触控制不稳时，可降低探索相关参数。
- `Gamma` 和 `Lambda`：更接近 1 会更重视长远奖励，但也会增加 value 估计难度。稀疏奖励或长程任务通常需要较高值，短程稳定控制可以适当降低。

## 当前配置中需要特别注意的点

- `BasicSettings.Seed`、`ObsShape`、`FrameSkip` 在当前在线训练入口中不是主要控制项，修改它们通常不会改变训练行为。
- `VideoLogStep` 对当前 proprio 在线训练基本无可见效果，因为视频日志分支只在非 proprio world model 中记录。
- `Tau` 和 `WorldModel.ValScale` 当前没有参与在线更新公式。如果后续实现 slow critic 软更新或 value loss 加权，再调整这些参数才会有效。
- `BatchSize` 与 `num_envs` 的整除关系是硬约束，不满足时 replay buffer 采样会触发 assertion。
