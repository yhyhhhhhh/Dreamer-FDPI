import os
from collections import deque

import colorama
import numpy as np
import torch
from tqdm import tqdm

OFFLINE_EXTRA_OBS_KEYS = (
    "force",
)

FORCE_OBS_CANDIDATE_KEYS = (
    "force",
    "pipe_force_curr",
    "pipe_force",
    "contact_force",
    "pipe_contact_force",
    "ft_force",
)


def train_world_model_step(samples, world_model, agent, logger, step):
    if agent is not None:
        agent.eval()
    world_model.update(agent, *samples, logger=logger, step=step)


def train_agent_step(samples, world_model, agent, imagine_horizon, logger, step):
    world_model.eval()
    imagine_outputs = world_model.imagine_data(agent, *samples[:5], imagine_horizon, logger, step)
    agent.update(*imagine_outputs, logger, step)


def _policy_obs(obs_dict):
    return torch.as_tensor(obs_dict["policy"], dtype=torch.float32)


def _is_first(obs_dict, num_envs, device):
    is_first = obs_dict.get("is_first")
    if is_first is None:
        return torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
    is_first = torch.as_tensor(is_first, dtype=torch.float32, device=device)
    return is_first.view(num_envs, 1)


def _reset_after_step(env, done, device):
    reset_obs = env.reset(seed=done.to(torch.int32))
    current_obs = _policy_obs(reset_obs).to(device)
    is_first = _is_first(reset_obs, env.num_envs, device)
    return reset_obs, current_obs, is_first


def _extract_force_obs(obs_dict, num_envs, device, force_key=""):
    candidate_keys = (force_key,) if force_key else FORCE_OBS_CANDIDATE_KEYS
    for key in candidate_keys:
        if not key:
            continue
        value = obs_dict.get(key)
        if value is None:
            continue
        force = torch.as_tensor(value, dtype=torch.float32, device=device)
        if force.shape[0] != num_envs:
            raise ValueError(
                f"Force observation `{key}` has first dimension {force.shape[0]}, expected {num_envs}."
            )
        force = force.reshape(num_envs, -1)
        if force.shape[1] == 1:
            return force
        return torch.linalg.norm(force, dim=-1, keepdim=True)

    available = ", ".join(sorted(obs_dict.keys()))
    tried = ", ".join(key for key in candidate_keys if key)
    raise KeyError(
        "ForceHead.Enable=True but no force observation was found. "
        f"Tried keys [{tried}], available keys [{available}]."
    )


def _log_info_value(logger, tag, value, step):
    if value is None:
        return

    if isinstance(value, dict):
        for key, sub_value in value.items():
            _log_info_value(logger, f"{tag}/{key}", sub_value, step)
        return

    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() == 0:
            return
        if value.ndim == 0 or value.numel() == 1:
            logger.log(tag, value.float().item(), step)
            return
        if tag.endswith("_ids") or tag.endswith("/ids"):
            logger.log(f"{tag}_count", int(value.numel()), step)
            return
        logger.log(f"{tag}_mean", value.float().mean().item(), step)
        return

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return
        try:
            value = np.asarray(value)
        except Exception:
            return

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return
        if value.ndim == 0 or value.size == 1:
            logger.log(tag, float(value.astype(np.float32).reshape(-1)[0]), step)
            return
        if tag.endswith("_ids") or tag.endswith("/ids"):
            logger.log(f"{tag}_count", int(value.size), step)
            return
        logger.log(f"{tag}_mean", float(value.astype(np.float32).mean()), step)
        return

    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        logger.log(tag, float(value), step)


def _log_info_dict(logger, info, step):
    if not isinstance(info, dict):
        return

    skip_keys = {"terminal_observation"}
    for key, value in info.items():
        if key in skip_keys:
            continue
        _log_info_value(logger, f"Info/{key}", value, step)


class OfflineEpisodeWriter:
    def __init__(self, output_dir, num_envs):
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        self.num_envs = int(num_envs)
        self.num_saved_episodes = 0
        self.num_saved_steps = 0
        self._episodes = [self._empty_episode() for _ in range(self.num_envs)]
        self._logged_missing_keys = False

    @staticmethod
    def _empty_episode():
        episode = {"obs": [], "action": [], "reward": [], "done": [], "is_first": []}
        for key in OFFLINE_EXTRA_OBS_KEYS:
            episode[key] = []
        return episode

    def append_step(self, obs_dict, action, reward, done, is_first, env_step):
        obs_np = _policy_obs(obs_dict).detach().cpu().numpy()
        action_np = torch.as_tensor(action, dtype=torch.float32).detach().cpu().numpy()
        reward_np = torch.as_tensor(reward, dtype=torch.float32).detach().cpu().view(-1).numpy()
        done_np = torch.as_tensor(done, dtype=torch.float32).detach().cpu().view(-1).numpy()
        is_first_np = torch.as_tensor(is_first, dtype=torch.float32).detach().cpu().view(-1).numpy()
        extra_obs = {}
        missing_keys = []
        for key in OFFLINE_EXTRA_OBS_KEYS:
            value = obs_dict.get(key)
            if value is None:
                missing_keys.append(key)
                extra_obs[key] = None
                continue
            extra_obs[key] = torch.as_tensor(value, dtype=torch.float32).detach().cpu().numpy()

        if missing_keys and not self._logged_missing_keys:
            print(
                colorama.Fore.YELLOW
                + "Offline episode export is missing extra observation keys: "
                + ", ".join(missing_keys)
                + colorama.Style.RESET_ALL
            )
            self._logged_missing_keys = True

        for env_idx in range(self.num_envs):
            episode = self._episodes[env_idx]
            episode["obs"].append(obs_np[env_idx].copy())
            episode["action"].append(action_np[env_idx].copy())
            episode["reward"].append(np.float32(reward_np[env_idx]))
            episode["done"].append(np.float32(done_np[env_idx]))
            episode["is_first"].append(np.float32(is_first_np[env_idx]))
            for key, value in extra_obs.items():
                if value is not None:
                    episode[key].append(value[env_idx].copy())

            if done_np[env_idx] > 0.5:
                self._flush_env_episode(env_idx, env_step, partial=False)

    def flush_pending(self, env_step):
        for env_idx in range(self.num_envs):
            if self._episodes[env_idx]["obs"]:
                self._flush_env_episode(env_idx, env_step, partial=True)

    def _flush_env_episode(self, env_idx, env_step, partial):
        episode = self._episodes[env_idx]
        length = len(episode["obs"])
        if length == 0:
            return

        obs = np.asarray(episode["obs"], dtype=np.float32)
        action = np.asarray(episode["action"], dtype=np.float32)
        reward = np.asarray(episode["reward"], dtype=np.float32)
        done = np.asarray(episode["done"], dtype=np.float32)
        is_first = np.asarray(episode["is_first"], dtype=np.float32)
        extras = {}
        for key in OFFLINE_EXTRA_OBS_KEYS:
            if episode[key]:
                extras[key] = np.asarray(episode[key], dtype=np.float32)

        is_first[0] = 1.0
        if partial:
            done[-1] = 1.0

        suffix = "_partial" if partial else ""
        file_path = os.path.join(
            self.output_dir,
            f"episode_{self.num_saved_episodes:08d}_env{env_idx:03d}_step{int(env_step):010d}{suffix}.npz",
        )
        np.savez_compressed(
            file_path,
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            is_first=is_first,
            **extras,
        )

        self.num_saved_episodes += 1
        self.num_saved_steps += length
        self._episodes[env_idx] = self._empty_episode()


def joint_train_world_model_agent(
    env_name,
    run_name,
    vec_env,
    max_steps,
    replay_buffer,
    world_model,
    agent,
    train_model_every_steps,
    train_agent_every_steps,
    model_update,
    agent_update,
    batch_size,
    batch_length,
    imagine_batch_size,
    imagine_context,
    imagine_horizon,
    save_every_steps,
    logger,
    device,
    offline_dataset_dir=None,
    checkpoint_dir=None,
):
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir or f"ckpt/{run_name}"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(
        colorama.Fore.CYAN
        + f"Saving checkpoints to {checkpoint_dir}"
        + colorama.Style.RESET_ALL
    )
    num_envs = vec_env.num_envs
    offline_episode_writer = None
    model_update = max(int(model_update), 1)
    agent_update = max(int(agent_update), 1)
    batch_size = int(batch_size)
    batch_length = int(batch_length)
    imagine_batch_size = int(imagine_batch_size)
    imagine_context = int(imagine_context)
    if imagine_batch_size <= 0:
        imagine_batch_size = batch_size
    if imagine_context <= 0:
        imagine_context = batch_length
    model_update_count = 0
    agent_update_count = 0

    if offline_dataset_dir:
        offline_episode_writer = OfflineEpisodeWriter(offline_dataset_dir, num_envs)
        print(
            colorama.Fore.CYAN
            + f"Saving offline episodes to {offline_episode_writer.output_dir}"
            + colorama.Style.RESET_ALL
        )

    world_model.eval()
    agent.eval()
    state = world_model.initial(num_envs)
    current_obs_dict = vec_env.reset()
    current_obs = _policy_obs(current_obs_dict).to(device)
    is_first = _is_first(current_obs_dict, num_envs, device)
    sum_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episodes_completed = 0
    episode_successes = 0
    episode_failures = 0
    episode_timeouts = 0
    recent_episode_success = deque(maxlen=1024)

    logger.log(f"Rollout/IsaacLab/{env_name}_reward", 0, 0)
    logger.log("Rollout/buffer_length", 0, 0)
    # total_iters计算的是并行环境需要step的次数
    total_iters = max_steps // num_envs
    # train_model_every_iters是
    train_model_every_iters = max(train_model_every_steps // num_envs, 1)
    train_agent_every_iters = max(train_agent_every_steps // num_envs, 1)
    save_every_iters = max(save_every_steps // num_envs, 1)

    for iter_idx in tqdm(range(total_iters)):
        env_steps = iter_idx * num_envs

        if replay_buffer.ready():
            with torch.no_grad():
                world_model.eval()
                agent.eval()
                feat, state = world_model.get_inference_feat(state, current_obs, is_first)
                env_action, action = agent.sample_as_env_action(feat, greedy=False)
                state = world_model.update_inference_state(state, action)
        else:
            # 如果replay buffer中的样本数目不够启动训练，用随机动作探索填充replay buffer
            sampled = vec_env.action_space.sample()
            env_action = np.asarray(sampled, dtype=np.float32)
            action = torch.as_tensor(env_action, dtype=torch.float32, device=device)

        next_obs_dict, reward, done, info = vec_env.step(env_action)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        done = torch.as_tensor(done, dtype=torch.bool, device=device)
        _log_info_dict(logger, info, env_steps)

        terminal = torch.as_tensor(
            next_obs_dict.get("is_terminal", torch.zeros_like(done, dtype=torch.int32)),
            dtype=torch.bool,
            device=device,
        ).view(-1)
        failure = torch.as_tensor(
            next_obs_dict.get("failure", torch.zeros_like(done, dtype=torch.int32)),
            dtype=torch.bool,
            device=device,
        ).view(-1)
        episode_success = info.get("episode_success")
        if episode_success is None:
            episode_success = terminal & ~failure
        else:
            episode_success = torch.as_tensor(episode_success, dtype=torch.bool, device=device).view(-1)
        episode_failure = info.get("episode_failure")
        if episode_failure is None:
            episode_failure = terminal & failure
        else:
            episode_failure = torch.as_tensor(episode_failure, dtype=torch.bool, device=device).view(-1)
        episode_timeout = info.get("episode_timeout")
        if episode_timeout is None:
            episode_timeout = done & ~terminal
        else:
            episode_timeout = torch.as_tensor(episode_timeout, dtype=torch.bool, device=device).view(-1)

        force = None
        if getattr(replay_buffer, "include_force", False):
            force = _extract_force_obs(
                current_obs_dict,
                num_envs,
                device,
                getattr(replay_buffer, "force_key", ""),
            )

        replay_buffer.append(current_obs, action, reward, done, is_first, force=force)
        if offline_episode_writer is not None:
            offline_episode_writer.append_step(
                current_obs_dict,
                action,
                reward,
                done,
                is_first,
                env_steps + num_envs,
            )
        sum_reward += reward

        if done.any():
            done_indices = torch.nonzero(done, as_tuple=False).flatten()
            completed_now = int(done_indices.numel())
            success_now = int(episode_success[done_indices].sum().item())
            failure_now = int(episode_failure[done_indices].sum().item())
            timeout_now = int(episode_timeout[done_indices].sum().item())

            episodes_completed += completed_now
            episode_successes += success_now
            episode_failures += failure_now
            episode_timeouts += timeout_now
            recent_episode_success.extend(episode_success[done_indices].float().cpu().tolist())

            for idx in done_indices.tolist():
                if replay_buffer.ready():
                    logger.log(f"Rollout/IsaacLab/{env_name}_reward", sum_reward[idx].item(), env_steps)
                    logger.log("Rollout/buffer_length", len(replay_buffer), env_steps)
                sum_reward[idx] = 0.0

            logger.log("Rollout/episodes_completed", episodes_completed, env_steps)
            logger.log("Rollout/episode_successes", episode_successes, env_steps)
            logger.log("Rollout/episode_failures", episode_failures, env_steps)
            logger.log("Rollout/episode_timeouts", episode_timeouts, env_steps)
            logger.log("Rollout/episode_success_rate", episode_successes / max(episodes_completed, 1), env_steps)
            logger.log("Rollout/episode_failure_rate", episode_failures / max(episodes_completed, 1), env_steps)
            logger.log("Rollout/episode_timeout_rate", episode_timeouts / max(episodes_completed, 1), env_steps)
            logger.log(
                "Rollout/episode_success_rate_recent",
                float(np.mean(recent_episode_success)) if recent_episode_success else 0.0,
                env_steps,
            )

        current_obs_dict, current_obs, is_first = _reset_after_step(vec_env, done, device)

        if replay_buffer.ready():
            if iter_idx % train_model_every_iters == 0 and replay_buffer.can_sample(batch_length):
                for _ in range(model_update):
                    samples = replay_buffer.sample(batch_size, batch_length)
                    train_world_model_step(samples, world_model, agent, logger, env_steps)
                    model_update_count += 1
            if iter_idx % train_agent_every_iters == 0 and replay_buffer.can_sample(imagine_context):
                for _ in range(agent_update):
                    imagine_samples = replay_buffer.sample(imagine_batch_size, imagine_context)
                    train_agent_step(imagine_samples, world_model, agent, imagine_horizon, logger, env_steps)
                    agent_update_count += 1

            collected_steps = env_steps + num_envs
            logger.log("Train/model_updates", model_update_count, env_steps)
            logger.log("Train/agent_updates", agent_update_count, env_steps)
            logger.log("Train/model_update_ratio", model_update_count / collected_steps, env_steps)
            logger.log("Train/agent_update_ratio", agent_update_count / collected_steps, env_steps)

        if iter_idx % save_every_iters == 0:
            print(
                colorama.Fore.GREEN
                + f"Saving model at total steps {env_steps}"
                + colorama.Style.RESET_ALL
            )
            torch.save(world_model.state_dict(), os.path.join(checkpoint_dir, f"world_model_{env_steps}.pth"))
            torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"agent_{env_steps}.pth"))

    if offline_episode_writer is not None:
        offline_episode_writer.flush_pending(max_steps)
        print(
            colorama.Fore.CYAN
            + (
                f"Saved {offline_episode_writer.num_saved_episodes} offline episodes "
                f"({offline_episode_writer.num_saved_steps} steps) to "
                f"{offline_episode_writer.output_dir}"
            )
            + colorama.Style.RESET_ALL
        )


def offline_train_world_model(
    run_name,
    replay_buffer,
    world_model,
    train_steps,
    batch_size,
    batch_length,
    save_every_steps,
    logger,
    checkpoint_dir=None,
):
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir or f"ckpt/{run_name}"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(
        colorama.Fore.CYAN
        + f"Saving checkpoints to {checkpoint_dir}"
        + colorama.Style.RESET_ALL
    )

    if not replay_buffer.can_sample(batch_length):
        raise ValueError(
            f"Offline dataset cannot sample batch_length={batch_length}. "
            "Check that the dataset contains at least one full episode longer than the sample horizon."
        )

    logger.log("Offline/buffer_length", len(replay_buffer), 0)

    total_updates = int(train_steps)
    save_every_steps = max(int(save_every_steps), 1)
    for update_idx in tqdm(range(1, total_updates + 1)):
        samples = replay_buffer.sample(batch_size, batch_length)
        if replay_buffer.device != world_model.device:
            samples = tuple(sample.to(world_model.device) for sample in samples)
        train_world_model_step(samples, world_model, None, logger, update_idx)

        if update_idx % save_every_steps == 0 or update_idx == total_updates:
            print(
                colorama.Fore.GREEN
                + f"Saving world model at update {update_idx}"
                + colorama.Style.RESET_ALL
            )
            torch.save(world_model.state_dict(), os.path.join(checkpoint_dir, f"world_model_{update_idx}.pth"))
