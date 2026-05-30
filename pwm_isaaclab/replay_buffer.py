import os

import numpy as np
import torch


class ProprioReplayBuffer:
    def __init__(
        self,
        obs_dim,
        action_dim,
        num_envs,
        max_length=int(2e5),
        warmup_length=2500,
        device="cpu",
        include_force=False,
        force_dim=1,
        force_key="",
    ):
        default = (max_length // num_envs, num_envs)
        self.obs_buffer = torch.empty(*default, obs_dim, dtype=torch.float32, device=device)
        self.action_buffer = torch.empty(*default, action_dim, dtype=torch.float32, device=device)
        self.reward_buffer = torch.empty(*default, 1, dtype=torch.float32, device=device)
        self.done_buffer = torch.empty(*default, 1, dtype=torch.float32, device=device)
        self.is_first_buffer = torch.empty(*default, 1, dtype=torch.float32, device=device)
        self.force_buffer = None
        if include_force:
            self.force_buffer = torch.empty(*default, force_dim, dtype=torch.float32, device=device)

        self.num_envs = num_envs
        self.max_length = max_length
        self.include_force = bool(include_force)
        self.force_dim = int(force_dim)
        self.force_key = force_key
        self.length = -1
        self.warmup_length = warmup_length
        self.device = device

    def ready(self):
        return len(self) > self.warmup_length

    def _valid_starts(self, env_idx, horizon):
        num_rows = self.length + 1
        total_length = num_rows - horizon + 1
        if total_length <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        if horizon <= 1:
            return torch.arange(total_length, device=self.device)

        done = self.done_buffer[:num_rows, env_idx, 0] > 0
        done_windows = done.unfold(0, horizon - 1, 1)
        valid_mask = ~done_windows.any(dim=-1)
        valid_mask = valid_mask[:total_length]
        return torch.nonzero(valid_mask, as_tuple=False).flatten()

    def can_sample(self, horizon):
        if not self.ready() or self.length + 1 < horizon:
            return False

        for env_idx in range(self.num_envs):
            if self._valid_starts(env_idx, horizon).numel() == 0:
                return False
        return True

    @torch.no_grad()
    def sample(self, batch_size, horizon):
        obs, action, reward, done, is_first = [], [], [], [], []
        force = []
        assert batch_size > 0
        assert batch_size >= self.num_envs and batch_size % self.num_envs == 0, (
            f"batch_size ({batch_size}) must be >= num_envs ({self.num_envs}) "
            "and divisible by num_envs."
        )
        length = torch.arange(horizon, device=self.device)
        for env_idx in range(self.num_envs):
            valid_starts = self._valid_starts(env_idx, horizon)
            if valid_starts.numel() == 0:
                raise ValueError(
                    f"No valid replay sequences for env index {env_idx} and horizon {horizon}. "
                    "This usually means sampled windows would cross episode boundaries."
                )
            sample_ids = torch.randint(valid_starts.numel(), (batch_size // self.num_envs,), device=self.device)
            starts = valid_starts[sample_ids]
            indexes = length[None, :] + starts[:, None]

            obs.append(self.obs_buffer[indexes, env_idx])
            action.append(self.action_buffer[indexes, env_idx])
            reward.append(self.reward_buffer[indexes, env_idx])
            done.append(self.done_buffer[indexes, env_idx])
            is_first.append(self.is_first_buffer[indexes, env_idx])
            if self.include_force:
                force.append(self.force_buffer[indexes, env_idx])

        samples = (
            torch.cat(obs, dim=0),
            torch.cat(action, dim=0),
            torch.cat(reward, dim=0),
            torch.cat(done, dim=0),
            torch.cat(is_first, dim=0),
        )
        if self.include_force:
            samples = (*samples, torch.cat(force, dim=0))
        return samples

    def append(self, obs, action, reward, done, is_first, force=None):
        self.length = (self.length + 1) % (self.max_length // self.num_envs)
        self.obs_buffer[self.length] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.action_buffer[self.length] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.reward_buffer[self.length] = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1, 1)
        self.done_buffer[self.length] = torch.as_tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)
        self.is_first_buffer[self.length] = (
            torch.as_tensor(is_first, dtype=torch.float32, device=self.device).view(-1, 1)
        )
        if self.include_force:
            if force is None:
                raise ValueError("Replay buffer was created with include_force=True, but append got force=None.")
            self.force_buffer[self.length] = torch.as_tensor(
                force, dtype=torch.float32, device=self.device
            ).view(self.num_envs, self.force_dim)

    def __len__(self):
        return 0 if self.length < 0 else (self.length + 1) * self.num_envs


def _as_2d_float_array(array, name, file_path):
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2:
        raise ValueError(f"`{name}` in {file_path} must be 1D or 2D, got shape {value.shape}.")
    if value.shape[0] <= 0:
        raise ValueError(f"`{name}` in {file_path} is empty.")
    return value


def _as_1d_float_array(array, name, file_path, length):
    value = np.asarray(array, dtype=np.float32).reshape(-1)
    if value.shape[0] != length:
        raise ValueError(
            f"`{name}` in {file_path} has length {value.shape[0]}, expected {length}."
        )
    return value


def _get_required_array(data, candidate_keys, canonical_name, file_path):
    for key in candidate_keys:
        if key in data:
            return data[key], key
    available_keys = ", ".join(sorted(data.keys()))
    raise KeyError(
        f"{file_path} must contain `{canonical_name}`. "
        f"Tried keys {candidate_keys}, available keys: [{available_keys}]."
    )


def _load_episode_npz(file_path):
    with np.load(file_path) as data:
        obs_array, obs_key = _get_required_array(data, ("obs", "policy"), "obs", file_path)
        action_array, action_key = _get_required_array(data, ("action",), "action", file_path)
        reward_array, reward_key = _get_required_array(data, ("reward",), "reward", file_path)

        obs = _as_2d_float_array(obs_array, obs_key, file_path)
        action = _as_2d_float_array(action_array, action_key, file_path)
        reward = _as_1d_float_array(reward_array, reward_key, file_path, obs.shape[0])

        if action.shape[0] != obs.shape[0]:
            raise ValueError(
                f"`action` in {file_path} has length {action.shape[0]}, expected {obs.shape[0]}."
            )

        if "done" in data:
            done = _as_1d_float_array(data["done"], "done", file_path, obs.shape[0])
        elif "is_last" in data and "is_terminal" in data:
            done = np.maximum(
                _as_1d_float_array(data["is_last"], "is_last", file_path, obs.shape[0]),
                _as_1d_float_array(data["is_terminal"], "is_terminal", file_path, obs.shape[0]),
            )
        elif "is_last" in data:
            done = _as_1d_float_array(data["is_last"], "is_last", file_path, obs.shape[0])
        elif "is_terminal" in data:
            done = _as_1d_float_array(data["is_terminal"], "is_terminal", file_path, obs.shape[0])
        else:
            done = np.zeros(obs.shape[0], dtype=np.float32)

        if "is_first" in data:
            is_first = _as_1d_float_array(data["is_first"], "is_first", file_path, obs.shape[0])
        else:
            is_first = np.zeros(obs.shape[0], dtype=np.float32)

    # Each file is treated as one closed episode to prevent windows from
    # sampling across file boundaries.
    done[-1] = 1.0
    is_first[0] = 1.0

    return {
        "obs": obs,
        "action": action,
        "reward": reward,
        "done": done,
        "is_first": is_first,
        "length": int(obs.shape[0]),
        "obs_dim": int(obs.shape[1]),
        "action_dim": int(action.shape[1]),
        "file_path": file_path,
    }


def load_offline_episodes(dataset_path, device="cpu", max_episodes=None):
    dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
    if os.path.isdir(dataset_path):
        episode_files = sorted(
            os.path.join(dataset_path, name)
            for name in os.listdir(dataset_path)
            if name.endswith(".npz")
        )
    elif os.path.isfile(dataset_path) and dataset_path.endswith(".npz"):
        episode_files = [dataset_path]
    else:
        raise FileNotFoundError(
            f"Offline dataset not found: {dataset_path}. "
            "Expected a directory of `.npz` episodes or a single `.npz` file."
        )

    if max_episodes is not None and max_episodes > 0:
        episode_files = episode_files[: max_episodes]

    if not episode_files:
        raise FileNotFoundError(f"No `.npz` episode files found under {dataset_path}.")

    episodes = [_load_episode_npz(file_path) for file_path in episode_files]
    obs_dim = episodes[0]["obs_dim"]
    action_dim = episodes[0]["action_dim"]
    total_steps = 0
    episode_lengths = []

    for episode in episodes:
        if episode["obs_dim"] != obs_dim:
            raise ValueError(
                f"Inconsistent obs dim in {episode['file_path']}: "
                f"{episode['obs_dim']} != {obs_dim}."
            )
        if episode["action_dim"] != action_dim:
            raise ValueError(
                f"Inconsistent action dim in {episode['file_path']}: "
                f"{episode['action_dim']} != {action_dim}."
            )
        total_steps += episode["length"]
        episode_lengths.append(episode["length"])

    replay_buffer = ProprioReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_envs=1,
        max_length=total_steps,
        warmup_length=0,
        device=device,
    )

    cursor = 0
    for episode in episodes:
        next_cursor = cursor + episode["length"]
        replay_buffer.obs_buffer[cursor:next_cursor, 0] = torch.as_tensor(
            episode["obs"], dtype=torch.float32, device=device
        )
        replay_buffer.action_buffer[cursor:next_cursor, 0] = torch.as_tensor(
            episode["action"], dtype=torch.float32, device=device
        )
        replay_buffer.reward_buffer[cursor:next_cursor, 0] = torch.as_tensor(
            episode["reward"], dtype=torch.float32, device=device
        ).view(-1, 1)
        replay_buffer.done_buffer[cursor:next_cursor, 0] = torch.as_tensor(
            episode["done"], dtype=torch.float32, device=device
        ).view(-1, 1)
        replay_buffer.is_first_buffer[cursor:next_cursor, 0] = torch.as_tensor(
            episode["is_first"], dtype=torch.float32, device=device
        ).view(-1, 1)
        cursor = next_cursor

    replay_buffer.length = total_steps - 1
    metadata = {
        "dataset_path": dataset_path,
        "num_episodes": len(episodes),
        "total_steps": total_steps,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "min_episode_length": int(min(episode_lengths)),
        "max_episode_length": int(max(episode_lengths)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "episode_files": episode_files,
    }
    return replay_buffer, metadata


ReplayBuffer = ProprioReplayBuffer
