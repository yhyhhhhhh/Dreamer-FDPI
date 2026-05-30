from __future__ import annotations

import os
import pickle
import warnings
from typing import Any

import numpy as np


SOURCE_MAIN = 0
SOURCE_EXPERT = 1
SOURCE_DUAL_RESERVED = 2


class ExpertDataset(list):
    """List-like container with aggregate expert dataset metadata."""

    def __init__(self, episodes, metadata):
        super().__init__(episodes)
        self.metadata = metadata


def _as_array(data, key, dtype=None):
    value = np.asarray(data[key])
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    return value


def _first_existing_key(data, keys):
    for key in keys:
        if key in data:
            return key
    return None


def _as_2d_float(array, name, file_path):
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2:
        raise ValueError(f"`{name}` in {file_path} must be 1D or 2D, got {value.shape}.")
    if value.shape[0] <= 0:
        raise ValueError(f"`{name}` in {file_path} is empty.")
    return value


def _as_1d_float(array, name, file_path, length):
    value = np.asarray(array, dtype=np.float32).reshape(-1)
    if value.shape[0] != length:
        raise ValueError(f"`{name}` in {file_path} has length {value.shape[0]}, expected {length}.")
    return value


def _space_shape(space):
    if space is None:
        return None
    shape = getattr(space, "shape", None)
    if shape is not None:
        return tuple(shape)
    return None


def _env_spec_value(env_spec, *names):
    if env_spec is None:
        return None
    if isinstance(env_spec, dict):
        for name in names:
            if name in env_spec:
                return env_spec[name]
        return None
    for name in names:
        if hasattr(env_spec, name):
            return getattr(env_spec, name)
    return None


def _validate_env_spec(obs, action, env_spec, file_path):
    if env_spec is None:
        return

    obs_shape = _env_spec_value(env_spec, "obs_shape", "observation_shape")
    action_shape = _env_spec_value(env_spec, "action_shape")
    observation_space = _env_spec_value(env_spec, "observation_space", "single_observation_space")
    action_space = _env_spec_value(env_spec, "action_space", "single_action_space")

    if obs_shape is None and observation_space is not None:
        if hasattr(observation_space, "spaces") and "policy" in observation_space.spaces:
            obs_shape = _space_shape(observation_space.spaces["policy"])
        else:
            obs_shape = _space_shape(observation_space)
    if action_shape is None and action_space is not None:
        action_shape = _space_shape(action_space)

    if obs_shape is not None and int(obs_shape[-1]) != obs.shape[-1]:
        raise ValueError(
            f"Expert obs dim mismatch in {file_path}: got {obs.shape[-1]}, expected {obs_shape[-1]}."
        )
    if action_shape is not None and int(action_shape[-1]) != action.shape[-1]:
        raise ValueError(
            f"Expert action dim mismatch in {file_path}: got {action.shape[-1]}, expected {action_shape[-1]}."
        )


def _validate_action_scale(action, file_path, tolerance):
    action_min = float(np.nanmin(action))
    action_max = float(np.nanmax(action))
    if action_min < -1.0 - tolerance or action_max > 1.0 + tolerance:
        raise ValueError(
            "Expert actions appear to be in environment units rather than the normalized actor range. "
            f"{file_path} action range is [{action_min:.6f}, {action_max:.6f}], expected within "
            f"[-1, 1] +/- {tolerance}."
        )


def _done_from_data(data, length, file_path):
    if "done" in data:
        return _as_1d_float(data["done"], "done", file_path, length)
    if "terminated" in data and "truncated" in data:
        return np.maximum(
            _as_1d_float(data["terminated"], "terminated", file_path, length),
            _as_1d_float(data["truncated"], "truncated", file_path, length),
        )
    if "is_last" in data and "is_terminal" in data:
        return np.maximum(
            _as_1d_float(data["is_last"], "is_last", file_path, length),
            _as_1d_float(data["is_terminal"], "is_terminal", file_path, length),
        )
    for key in ("is_last", "is_terminal", "obs_is_last", "obs_is_terminal"):
        if key in data:
            return _as_1d_float(data[key], key, file_path, length)
    return np.zeros(length, dtype=np.float32)


def _resolve_episode_ranges(data, length):
    if "episode_starts" in data and "episode_lengths" in data:
        starts = np.asarray(data["episode_starts"], dtype=np.int64).reshape(-1)
        lengths = np.asarray(data["episode_lengths"], dtype=np.int64).reshape(-1)
        if starts.shape[0] != lengths.shape[0]:
            raise ValueError("`episode_starts` and `episode_lengths` have different lengths.")
        ranges = []
        for start, ep_len in zip(starts, lengths):
            start = int(start)
            stop = start + int(ep_len)
            if start < 0 or ep_len <= 0 or stop > length:
                raise ValueError(f"Invalid episode range [{start}, {stop}) for shard length {length}.")
            ranges.append((start, stop))
        return ranges

    first_key = _first_existing_key(data, ("is_first", "obs_is_first"))
    last_key = _first_existing_key(data, ("done", "is_last", "is_terminal", "obs_is_last", "obs_is_terminal"))
    if first_key is not None:
        first = np.asarray(data[first_key]).reshape(-1).astype(bool)
        starts = np.nonzero(first)[0].tolist()
        if not starts or starts[0] != 0:
            starts = [0] + starts
        ranges = []
        for idx, start in enumerate(starts):
            next_start = starts[idx + 1] if idx + 1 < len(starts) else length
            ranges.append((int(start), int(next_start)))
        return [(start, stop) for start, stop in ranges if stop > start]

    if last_key is not None:
        last = np.asarray(data[last_key]).reshape(-1).astype(bool)
        ends = np.nonzero(last)[0].tolist()
        ranges = []
        start = 0
        for end in ends:
            ranges.append((start, int(end) + 1))
            start = int(end) + 1
        if start < length:
            ranges.append((start, length))
        return [(start, stop) for start, stop in ranges if stop > start]

    return [(0, length)]


def _slice_optional_1d(data, keys, start, stop, file_path, length, fill_value, dtype=np.float32):
    key = _first_existing_key(data, keys)
    if key is None:
        return np.full(stop - start, fill_value, dtype=dtype), None
    value = _as_1d_float(data[key], key, file_path, length)[start:stop]
    return value.astype(dtype, copy=True), key


def _force_arrays(data, start, stop, file_path, length):
    key = _first_existing_key(data, ("force", "forceObs", "pipe_force_curr", "pipe_force", "contact_force"))
    if key is None:
        return None, None, None
    value = np.asarray(data[key], dtype=np.float32)
    if value.shape[0] != length:
        raise ValueError(f"`{key}` in {file_path} has length {value.shape[0]}, expected {length}.")
    value = value[start:stop]
    if value.ndim == 1:
        raw_force = value[:, None].astype(np.float32, copy=True)
        return raw_force, raw_force, key
    raw_force = value.reshape(value.shape[0], -1).astype(np.float32, copy=True)
    scalar_force = np.linalg.norm(raw_force, axis=-1, keepdims=True).astype(np.float32)
    return scalar_force, raw_force, key


def derive_cost_from_force_margin(
    raw_cost,
    raw_force=None,
    safety_margin=None,
    *,
    cost_target_source="raw",
    pipe_force_limit=1.0,
    bottom_force_limit=1.0,
    pipe_force_channels=(1, 4),
    bottom_force_channels=(2, 5),
):
    """Derive a cost target using v1 wall/bottom force semantics, with margin/raw fallbacks."""

    source = str(cost_target_source or "raw").lower()
    raw_cost = np.asarray(raw_cost, dtype=np.float32).reshape(-1)
    diagnostics = {
        "cost_target_source": "raw",
        "pipe_force": None,
        "bottom_force": None,
    }

    if source in ("force_margin", "force", "derived"):
        if raw_force is not None:
            force = np.asarray(raw_force, dtype=np.float32)
            if force.ndim == 1:
                force = force[:, None]
            pipe_channels = [int(idx) for idx in pipe_force_channels]
            bottom_channels = [int(idx) for idx in bottom_force_channels]
            if force.shape[-1] > max(pipe_channels + bottom_channels):
                pipe_force = np.max(force[:, pipe_channels], axis=-1)
                bottom_force = np.max(force[:, bottom_channels], axis=-1)
                pipe_limit = float(pipe_force_limit)
                bottom_limit = float(bottom_force_limit)
                pipe_cost = np.maximum(pipe_force - pipe_limit, 0.0)
                bottom_cost = np.maximum(bottom_force - bottom_limit, 0.0)
                diagnostics.update(
                    {
                        "cost_target_source": "force",
                        "pipe_force": pipe_force.astype(np.float32, copy=True),
                        "bottom_force": bottom_force.astype(np.float32, copy=True),
                    }
                )
                return np.maximum(pipe_cost, bottom_cost).astype(np.float32), diagnostics

        if source in ("force_margin", "margin", "derived") and safety_margin is not None:
            margin = np.asarray(safety_margin, dtype=np.float32).reshape(-1)
            finite_margin = np.isfinite(margin)
            if finite_margin.any():
                cost = np.zeros_like(raw_cost, dtype=np.float32)
                cost[finite_margin] = np.maximum(-margin[finite_margin], 0.0)
                diagnostics["cost_target_source"] = "margin"
                return cost, diagnostics

    return raw_cost.astype(np.float32, copy=True), diagnostics


def _load_npz_file(
    file_path,
    env_spec,
    action_tolerance,
    *,
    cost_target_source="raw",
    cost_pipe_force_limit=1.0,
    cost_bottom_force_limit=1.0,
    cost_pipe_force_channels=(1, 4),
    cost_bottom_force_channels=(2, 5),
):
    with np.load(file_path, allow_pickle=True) as data:
        obs_key = _first_existing_key(data, ("obs", "policy", "observation", "observation_policy"))
        action_key = _first_existing_key(data, ("action", "cur_actions", "actions"))
        reward_key = _first_existing_key(data, ("reward", "rewards"))
        if obs_key is None or action_key is None or reward_key is None:
            available = ", ".join(sorted(data.keys()))
            raise KeyError(
                f"{file_path} must contain obs/action/reward fields. Available keys: [{available}]."
            )

        obs = _as_2d_float(data[obs_key], obs_key, file_path)
        action = _as_2d_float(data[action_key], action_key, file_path)
        length = obs.shape[0]
        reward = _as_1d_float(data[reward_key], reward_key, file_path, length)
        if action.shape[0] != length:
            raise ValueError(f"`{action_key}` in {file_path} has length {action.shape[0]}, expected {length}.")

        _validate_env_spec(obs, action, env_spec, file_path)
        _validate_action_scale(action, file_path, action_tolerance)

        done = _done_from_data(data, length, file_path)
        if "is_first" in data:
            is_first = _as_1d_float(data["is_first"], "is_first", file_path, length)
        elif "obs_is_first" in data:
            is_first = _as_1d_float(data["obs_is_first"], "obs_is_first", file_path, length)
        else:
            is_first = np.zeros(length, dtype=np.float32)

        if "is_last" in data:
            is_last = _as_1d_float(data["is_last"], "is_last", file_path, length)
        elif "obs_is_last" in data:
            is_last = _as_1d_float(data["obs_is_last"], "obs_is_last", file_path, length)
        else:
            is_last = done.copy()

        if "is_terminal" in data:
            is_terminal = _as_1d_float(data["is_terminal"], "is_terminal", file_path, length)
        elif "terminated" in data:
            is_terminal = _as_1d_float(data["terminated"], "terminated", file_path, length)
        elif "obs_is_terminal" in data:
            is_terminal = _as_1d_float(data["obs_is_terminal"], "obs_is_terminal", file_path, length)
        else:
            is_terminal = done.copy()

        ranges = _resolve_episode_ranges(data, length)
        episodes = []
        cost_missing = "cost" not in data
        if cost_missing:
            warnings.warn(f"`cost` missing in {file_path}; filling zeros.", RuntimeWarning)

        for local_episode_id, (start, stop) in enumerate(ranges):
            ep_len = stop - start
            raw_cost, cost_key = _slice_optional_1d(data, ("cost", "costs"), start, stop, file_path, length, 0.0)
            safety_margin, margin_key = _slice_optional_1d(
                data,
                ("safety_margin", "constraint_margin", "margin"),
                start,
                stop,
                file_path,
                length,
                np.nan,
            )
            uncertainty_score, _ = _slice_optional_1d(
                data, ("uncertainty_score", "uncertainty"), start, stop, file_path, length, 0.0
            )
            ood_score, _ = _slice_optional_1d(data, ("ood_score", "ood"), start, stop, file_path, length, 0.0)
            force, raw_force, force_key = _force_arrays(data, start, stop, file_path, length)
            cost, cost_diag = derive_cost_from_force_margin(
                raw_cost,
                raw_force,
                safety_margin,
                cost_target_source=cost_target_source,
                pipe_force_limit=cost_pipe_force_limit,
                bottom_force_limit=cost_bottom_force_limit,
                pipe_force_channels=cost_pipe_force_channels,
                bottom_force_channels=cost_bottom_force_channels,
            )

            ep_done = done[start:stop].astype(np.float32, copy=True)
            ep_is_first = is_first[start:stop].astype(np.float32, copy=True)
            ep_is_last = is_last[start:stop].astype(np.float32, copy=True)
            ep_is_terminal = is_terminal[start:stop].astype(np.float32, copy=True)
            ep_is_first[0] = 1.0
            ep_done[-1] = max(ep_done[-1], ep_is_last[-1], ep_is_terminal[-1], 1.0)
            ep_is_last[-1] = 1.0

            episode = {
                "obs": obs[start:stop].astype(np.float32, copy=True),
                "action": action[start:stop].astype(np.float32, copy=True),
                "reward": reward[start:stop].astype(np.float32, copy=True),
                "cost": cost,
                "original_cost": raw_cost,
                "done": ep_done,
                "discount": (1.0 - ep_done).astype(np.float32),
                "is_first": ep_is_first,
                "is_last": ep_is_last,
                "is_terminal": ep_is_terminal,
                "source": np.full(ep_len, SOURCE_EXPERT, dtype=np.int64),
                "safety_margin": safety_margin,
                "uncertainty_score": uncertainty_score,
                "ood_score": ood_score,
                "episode_id": np.full(ep_len, local_episode_id, dtype=np.int64),
                "timestep": np.arange(ep_len, dtype=np.int64),
                "file_path": file_path,
                "source_name": "expert",
                "cost_key": cost_key,
                "cost_target_source": cost_diag["cost_target_source"],
                "safety_margin_key": margin_key,
                "force_key": force_key,
            }
            if cost_diag["pipe_force"] is not None:
                episode["pipe_force"] = cost_diag["pipe_force"]
            if cost_diag["bottom_force"] is not None:
                episode["bottom_force"] = cost_diag["bottom_force"]
            if force is not None:
                episode["force"] = force
            episodes.append(episode)

    return episodes


def _resolve_files(path):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        if path.endswith(".npz"):
            return [path]
        raise FileNotFoundError(f"Expected a .npz expert dataset file, got {path}.")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Expert dataset path not found: {path}.")
    files = []
    for root, _, names in os.walk(path):
        files.extend(os.path.join(root, name) for name in names if name.endswith(".npz"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No .npz expert shards found under {path}.")
    return files


def _load_pickle(path):
    with open(path, "rb") as fin:
        return pickle.load(fin)


def _metadata(path, episodes, files):
    num_steps = int(sum(len(ep["reward"]) for ep in episodes))
    returns = np.asarray([float(np.sum(ep["reward"])) for ep in episodes], dtype=np.float64)
    costs = np.asarray([float(np.sum(ep["cost"])) for ep in episodes], dtype=np.float64)
    all_cost = (
        np.concatenate([np.asarray(ep["cost"], dtype=np.float32).reshape(-1) for ep in episodes], axis=0)
        if episodes
        else np.zeros(0, dtype=np.float32)
    )
    all_original_cost = (
        np.concatenate(
            [
                np.asarray(ep.get("original_cost", ep["cost"]), dtype=np.float32).reshape(-1)
                for ep in episodes
            ],
            axis=0,
        )
        if episodes
        else np.zeros(0, dtype=np.float32)
    )
    pipe_force = (
        np.concatenate(
            [np.asarray(ep["pipe_force"], dtype=np.float32).reshape(-1) for ep in episodes if "pipe_force" in ep],
            axis=0,
        )
        if any("pipe_force" in ep for ep in episodes)
        else np.zeros(0, dtype=np.float32)
    )
    bottom_force = (
        np.concatenate(
            [np.asarray(ep["bottom_force"], dtype=np.float32).reshape(-1) for ep in episodes if "bottom_force" in ep],
            axis=0,
        )
        if any("bottom_force" in ep for ep in episodes)
        else np.zeros(0, dtype=np.float32)
    )
    positive_cost = all_cost[all_cost > 0]
    actions = np.concatenate([ep["action"] for ep in episodes], axis=0) if episodes else np.zeros((0, 0))
    obs_shape = list(episodes[0]["obs"].shape[1:]) if episodes else []
    action_shape = list(episodes[0]["action"].shape[1:]) if episodes else []
    return {
        "dataset_path": os.path.abspath(os.path.expanduser(path)),
        "format": "npz",
        "num_files": len(files),
        "num_episodes": int(len(episodes)),
        "num_transitions": num_steps,
        "total_steps": num_steps,
        "mean_return": float(returns.mean()) if returns.size else 0.0,
        "mean_cost": float(costs.mean()) if costs.size else 0.0,
        "cost_positive_count": int((all_cost > 0).sum()) if all_cost.size else 0,
        "cost_positive_ratio": float((all_cost > 0).mean()) if all_cost.size else 0.0,
        "derived_cost_mean": float(all_cost.mean()) if all_cost.size else 0.0,
        "derived_cost_max": float(all_cost.max()) if all_cost.size else 0.0,
        "derived_positive_cost_mean": float(positive_cost.mean()) if positive_cost.size else 0.0,
        "original_cost_positive_count": int((all_original_cost > 0).sum()) if all_original_cost.size else 0,
        "original_cost_positive_ratio": float((all_original_cost > 0).mean()) if all_original_cost.size else 0.0,
        "original_cost_mean": float(all_original_cost.mean()) if all_original_cost.size else 0.0,
        "original_cost_max": float(all_original_cost.max()) if all_original_cost.size else 0.0,
        "pipe_force_max": float(pipe_force.max()) if pipe_force.size else 0.0,
        "bottom_force_max": float(bottom_force.max()) if bottom_force.size else 0.0,
        "min_return": float(returns.min()) if returns.size else 0.0,
        "max_return": float(returns.max()) if returns.size else 0.0,
        "min_episode_length": int(min(len(ep["reward"]) for ep in episodes)) if episodes else 0,
        "max_episode_length": int(max(len(ep["reward"]) for ep in episodes)) if episodes else 0,
        "mean_episode_length": float(np.mean([len(ep["reward"]) for ep in episodes])) if episodes else 0.0,
        "obs_shape": obs_shape,
        "action_shape": action_shape,
        "obs_dim": int(obs_shape[-1]) if obs_shape else 0,
        "action_dim": int(action_shape[-1]) if action_shape else 0,
        "action_min": actions.min(axis=0).astype(float).tolist() if actions.size else [],
        "action_max": actions.max(axis=0).astype(float).tolist() if actions.size else [],
        "action_min_overall": float(actions.min()) if actions.size else 0.0,
        "action_max_overall": float(actions.max()) if actions.size else 0.0,
        "episode_files": files,
    }


def load_expert_dataset(
    path: str,
    format: str = "npz",
    env_spec: Any | None = None,
    *,
    action_tolerance: float = 1e-4,
    max_episodes: int | None = None,
    cost_target_source: str = "raw",
    cost_pipe_force_limit: float = 1.0,
    cost_bottom_force_limit: float = 1.0,
    cost_pipe_force_channels=(1, 4),
    cost_bottom_force_channels=(2, 5),
) -> ExpertDataset:
    """Load expert trajectories into a list of Dreamer-style episode dictionaries."""

    fmt = str(format).lower()
    if fmt == "pkl":
        data = _load_pickle(os.path.abspath(os.path.expanduser(path)))
        if isinstance(data, ExpertDataset):
            return data
        if not isinstance(data, list):
            raise TypeError("Pickle expert datasets must contain a list of episode dictionaries.")
        metadata = _metadata(path, data, [os.path.abspath(os.path.expanduser(path))])
        metadata["format"] = "pkl"
        return ExpertDataset(data, metadata)
    if fmt == "hdf5":
        raise NotImplementedError("Expert dataset format 'hdf5' is reserved but not implemented in v0.1.")
    if fmt != "npz":
        raise ValueError(f"Unsupported expert dataset format: {format!r}.")

    files = _resolve_files(path)
    episodes = []
    used_files = []
    for file_path in files:
        used_files.append(file_path)
        episodes.extend(
            _load_npz_file(
                file_path,
                env_spec,
                action_tolerance,
                cost_target_source=cost_target_source,
                cost_pipe_force_limit=cost_pipe_force_limit,
                cost_bottom_force_limit=cost_bottom_force_limit,
                cost_pipe_force_channels=cost_pipe_force_channels,
                cost_bottom_force_channels=cost_bottom_force_channels,
            )
        )
        if max_episodes is not None and max_episodes > 0 and len(episodes) >= max_episodes:
            episodes = episodes[:max_episodes]
            break
    metadata = _metadata(path, episodes, used_files)
    metadata["cost_target_source_requested"] = str(cost_target_source)
    metadata["cost_pipe_force_limit"] = float(cost_pipe_force_limit)
    metadata["cost_bottom_force_limit"] = float(cost_bottom_force_limit)
    metadata["cost_pipe_force_channels"] = [int(idx) for idx in cost_pipe_force_channels]
    metadata["cost_bottom_force_channels"] = [int(idx) for idx in cost_bottom_force_channels]
    return ExpertDataset(episodes, metadata)
