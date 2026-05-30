# Expert Initialization v0.1 训练教程

本文档说明如何使用离线专家数据对当前 PaMoRL / Dreamer 代码做专家初始化预训练，并判断预训练是否有效。

当前实现走新增入口，不修改原有在线训练主入口：

- 离线专家初始化入口：`pwm_isaaclab/expert_pretrain.py`
- 在线接入占位入口：`pwm_isaaclab/train_expert_online.py`
- 专家初始化配置：`pwm_isaaclab/config_files/PWM_expert_init.yaml`
- 专家数据目录通过 `-dataset_path` 指定。
- world-model coverage 数据目录通过 `-wm_coverage_path` 指定，可选。

## 1. 环境和数据检查

进入仓库根目录：

```bash
cd /home/yhy/PaMoRL-IsaacLab-clean
```

先运行新增 smoke tests，确认 expert loader、expert replay、world model 预训练一步、actor BC 一步都能跑通：

```bash
/home/yhy/anaconda3/envs/isaaclab_14/bin/python -m unittest tests/test_expert_init.py
```

预期输出：

```text
Ran 11 tests
OK
```

如果使用系统 `/usr/bin/python3`，可能会因为没有安装 `torch` 失败；请使用上面的 `isaaclab_14` Python。

## 2. 小规模预训练试跑

正式训练前建议先做一个短 smoke run，确认 GPU、数据、日志和 checkpoint 都正常。

可以直接使用 `-max_episodes` 限制加载专家 episode 数，同时临时把 `PWM_expert_init.yaml` 中的步数改小：

```yaml
expert:
  pretrain_steps: 100
  bc_steps: 50
```

运行：

```bash
WANDB_MODE=offline /home/yhy/anaconda3/envs/isaaclab_14/bin/python pwm_isaaclab/expert_pretrain.py \
  -n ur3-expert-init-smoke \
  -seed 42 \
  -config_path /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -dataset_path /path/to/expert_episodes \
  -device cuda:0 \
  -max_episodes 64 \
  -max_wm_coverage_episodes 128 \
  --no_run_info_prompt
```

成功后会在类似下面的目录生成初始化产物：

```text
ckpt/ur3-expert-init-smoke/<timestamp>/
```

关键文件包括：

```text
replay_after_expert_load.pt
world_model_expert_pretrained.pt
actor_bc_initialized.pt
full_agent_before_online.pt
config.yaml
run_info.json
```

## 3. 正式专家初始化预训练

确认小规模试跑没问题后，将配置改回正式训练步数，例如：

```yaml
expert:
  pretrain_steps: 50000
  bc_steps: 20000
```

然后运行完整专家数据预训练：

```bash
WANDB_MODE=offline /home/yhy/anaconda3/envs/isaaclab_14/bin/python pwm_isaaclab/expert_pretrain.py \
  -n ur3-expert-init \
  -seed 42 \
  -config_path /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -device cuda:0 \
  --no_run_info_prompt
```

该命令会执行：

1. 加载 expert `.npz` shard。
2. 按 episode 切分专家轨迹。
3. 递归加载 `expert.wm_coverage_paths` 中的噪声/随机 coverage shard。
4. 构建两个 replay：world model 使用 `expert + coverage`，actor BC 只使用 `expert`。
5. 用混合 replay 预训练 world model、reward head、done head、force head 和独立 cost head。
6. 用 expert obs/action 对主 actor 做 behavior cloning 初始化。
7. 保存 world model、actor 和完整 agent 初始化 checkpoint。

默认配置已经包含：

```yaml
expert:
  path: null
  wm_coverage_paths:
    []
```

也可以在命令行额外指定 coverage 数据：

```bash
-wm_coverage_path /path/to/another/coverage_dataset
```

如果只想先小规模测试 coverage，可以加：

```bash
-max_wm_coverage_episodes 128
```

### 当前推荐重训命令：专家数据 + mixed coverage + hurdle cost head

这条命令用于重新训练 world model，同时保持 actor BC 仍只使用专家数据。当前配置会启用：

- 专家数据集：`/path/to/expert_episodes`
- world model coverage 数据集：`/path/to/wm_coverage_episodes`
- cost target：从 `force(6)` / `constraint_margin` 派生
- cost head：`hurdle` 分类 + 回归结构

```bash
cd /home/yhy/PaMoRL-IsaacLab-clean

source /home/yhy/anaconda3/etc/profile.d/conda.sh
conda activate isaaclab_14

WANDB_MODE=offline PYTHONPATH=/home/yhy/PaMoRL-IsaacLab-clean:$PYTHONPATH \
/home/yhy/anaconda3/envs/isaaclab_14/bin/python \
  pwm_isaaclab/expert_pretrain.py \
  -n ur3-expert-init-mix-wm-hurdle-cost \
  -seed 0 \
  -config_path /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -dataset_path /path/to/expert_episodes \
  -wm_coverage_path /path/to/wm_coverage_episodes \
  -device cuda:0 \
  --run_root /path/to/checkpoints \
  --run_id 20260524_hurdle_cost_mix_wm \
  --no_run_info_prompt
```

训练完成后的主要 checkpoint 在：

```text
/path/to/checkpoints/ur3-expert-init-mix-wm-hurdle-cost/20260524_hurdle_cost_mix_wm/full_agent_before_online.pt
```

重点观察这些 cost head 指标，不要只看总体 `cost_mae`：

```text
cost/positive_ratio
cost/cls_loss
cost/reg_loss
cost/auprc
cost/recall@0.5
cost/mae_positive_only
```

## 4. 预训练是否有效的判断标准

重点查看终端输出和 wandb offline 日志中的指标。

专家数据加载指标：

```text
expert/num_episodes
expert/num_steps
expert/mean_return
expert/mean_cost
expert/action_min_overall
expert/action_max_overall
expert/wm_coverage_train_steps
expert/wm_coverage_validation_steps
expert/wm_train_steps_total
expert/wm_validation_steps_total
```

world model 预训练指标：

```text
expert_init/wm_loss
expert_init/recon_loss
expert_init/reward_loss
expert_init/cost_loss
expert_init/discount_loss
expert_init/dyn_loss
expert_init/rep_loss
```

actor BC 指标：

```text
expert_init/bc_loss
expert_init/action_mse
expert_init/action_log_prob
expert_init/actor_entropy
```

离线验证指标：

```text
expert_init/validation_recon_loss
expert_init/validation_reward_mae
expert_init/validation_cost_mae
expert_init/validation_bc_loss
expert_init/validation_action_mse
```

有效的基本信号：

- `wm_loss`、`recon_loss`、`reward_loss` 整体下降，且没有 `nan` 或 `inf`。
- `cost_loss` 是 finite；当前专家数据 cost 基本为 0，因此重点是不要发散。
- `bc_loss` 下降。
- `action_mse` 下降，说明 actor 输出动作逐渐接近专家动作。
- `full_agent_before_online.pt` 成功生成。

最小验收标准是：`expert_pretrain.py` 能完整跑完，且 `action_mse` 相比初始阶段明显下降。

注意：coverage 数据只参与 world model 预训练。`bc_loss` 和 `action_mse` 仍然只来自专家数据，因此它们可以继续作为“actor 是否学到专家策略”的判断信号。

## 5. 从专家初始化 checkpoint 接在线训练

专家初始化完成后，使用 `full_agent_before_online.pt` 作为在线训练起点。

示例：

```bash
/home/yhy/anaconda3/envs/isaaclab_14/bin/python pwm_isaaclab/train_expert_online.py \
  -n ur3-expert-online \
  -seed 42 \
  -config_path /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -env_name Ur3Lite-PipeRelGoalForce-OSC-RL-Direct-v0 \
  -device cuda:0 \
  -expert_init_checkpoint /path/to/full_agent_before_online.pt \
  --no_run_info_prompt
```

在线占位入口的行为：

- 如果 `expert.enabled=true` 且专家数据可用，会加载专家 replay。
- 如果 `expert.replace_random_prefill=true`，专家 replay 会让训练跳过原来的大量 random prefill。
- 在线 replay 未成熟时，训练 batch 会从 expert replay 采样。
- 在线 replay 能采样后，默认恢复原在线 replay 采样。
- v0.1 不实现 dual policy、不实现 uncertainty/OOD、不实现 importance sampling，也不会把 BC loss 强加到在线 actor loss。

## 6. 常见问题

### `python3` 找不到 `torch`

使用项目环境：

```bash
/home/yhy/anaconda3/envs/isaaclab_14/bin/python
```

不要直接用系统 `/usr/bin/python3` 跑训练或测试。

### 显存不够

先降低配置中的 batch 或训练长度：

```yaml
JointTrainAgent:
  BatchSize: 32
  BatchLength: 32
  ImagineContext: 16

expert:
  pretrain_steps: 10000
  bc_steps: 5000
```

也可以先用 `-max_episodes` 做分批试验。

### action scale 报错

expert loader 会检查 action 是否在 actor/env 使用的归一化范围内。当前专家数据动作范围应在 `[-1, 1]`，如果换数据集后超出范围，需要先确认该数据是否是环境真实动作，而不是 actor normalized action。

### checkpoint 恢复

如果不想每次重复预训练，可以在配置中指定：

```yaml
expert:
  skip_pretrain_if_checkpoint_exists: true
  load_pretrained_world_model_path: null
  load_bc_actor_path: null
```

也可以在线时直接传：

```bash
-expert_init_checkpoint /path/to/full_agent_before_online.pt
```

## 7. 推荐工作流

1. 运行 `tests/test_expert_init.py`。
2. 用 `-max_episodes 64` 和较小 `pretrain_steps/bc_steps` 做 smoke run。
3. 查看 `action_mse`、`bc_loss`、`wm_loss` 是否正常下降。
4. 跑完整 expert pretrain。
5. 记录 `full_agent_before_online.pt` 路径。
6. 用 `train_expert_online.py` 接在线训练。

## 8. 评估 world model 和策略

新增评估脚本：

```text
pwm_isaaclab/eval_expert_policy_world_model.py
```

离线数据集评估会分别输出：

- `expert_dataset`：专家数据上的 world model 误差。
- `coverage_dataset`：噪声/随机 coverage 数据上的 world model 误差。
- `wm_mix_dataset`：按数据量混合采样的 expert + coverage 误差。

运行示例：

```bash
WANDB_MODE=offline /home/yhy/anaconda3/envs/isaaclab_14/bin/python pwm_isaaclab/eval_expert_policy_world_model.py \
  -config_path /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -checkpoint_path /path/to/full_agent_before_online.pt \
  -device cuda:0 \
  --eval_dataset \
  --num_batches 200 \
  --batch_length 64 \
  --batch_size 64 \
  -save_dir eval_expert_policy_world_model
```

快速离线 smoke：

```bash
WANDB_MODE=offline /home/yhy/anaconda3/envs/isaaclab_14/bin/python pwm_isaaclab/eval_expert_policy_world_model.py \
  -config_path /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -checkpoint_path /path/to/full_agent_before_online.pt \
  -device cuda:0 \
  --eval_dataset \
  -max_episodes 64 \
  -max_coverage_episodes 128 \
  --num_batches 20 \
  --batch_length 64 \
  --batch_size 64
```

在线环境采样评估：

```bash
WANDB_MODE=offline /home/yhy/anaconda3/envs/isaaclab_14/bin/python pwm_isaaclab/eval_expert_policy_world_model.py \
  -config_path /home/yhy/PaMoRL-IsaacLab-clean/pwm_isaaclab/config_files/PWM_expert_init.yaml \
  -checkpoint_path /path/to/full_agent_before_online.pt \
  -env_name Ur3Lite-PipeRelGoalForce-OSC-RL-Direct-v0 \
  -device cuda:0 \
  --online \
  -online_steps 8192 \
  -eval_num_envs 8 \
  -policy_mode greedy \
  --num_batches 100 \
  --batch_length 64 \
  --batch_size 64
```

核心输出指标：

```text
rollout_obs_rmse
one_step_obs_rmse
rollout_reward_mae
rollout_done_acc
rollout_cost_mae
rollout_force_mae
online_mean_episode_return
online_success_rate
```

结果会保存到：

```text
eval_expert_policy_world_model/<checkpoint_name>/
```
