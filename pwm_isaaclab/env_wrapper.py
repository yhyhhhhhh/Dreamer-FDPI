from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


class DreamerVecEnvWrapper(gym.Wrapper):
    """Adapt an Isaac Lab vector env to Dreamer-style batched observations."""

    def __init__(self, env, device: str | torch.device = "cuda", ac_lim: float | None = None):
        super().__init__(env)
        self.device = torch.device(device)
        self.ac_lim = ac_lim
        self._num_envs = getattr(env.unwrapped, "num_envs", None)
        if self._num_envs is None:
            raise ValueError("Underlying env must define `unwrapped.num_envs`.")

        self._has_reset_once = False
        self._reset_obs_cache = None

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def action_space(self):
        sp = self.env.action_space
        if self.ac_lim is None or not isinstance(sp, gym.spaces.Box):
            return sp

        low = -self.ac_lim * np.ones_like(sp.low)
        high = self.ac_lim * np.ones_like(sp.high)
        return gym.spaces.Box(low=low, high=high, dtype=sp.dtype)

    @property
    def single_action_space(self):
        sp = self.action_space
        if not isinstance(sp, gym.spaces.Box):
            return sp

        low = sp.low
        high = sp.high
        if low.ndim >= 2 and low.shape[0] == self.num_envs:
            low = low[0]
            high = high[0]
        return gym.spaces.Box(low=low, high=high, dtype=sp.dtype)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def single_observation_space(self):
        base = getattr(self.env.unwrapped, "_observation_space", self.env.observation_space)
        if hasattr(base, "spaces"):
            new_spaces = {}
            for key, space in base.spaces.items():
                if isinstance(space, gym.spaces.Box):
                    low = np.asarray(space.low)
                    high = np.asarray(space.high)
                    shape = space.shape
                    if len(shape) >= 1 and shape[0] == self.num_envs:
                        low = np.asarray(low[0])
                        high = np.asarray(high[0])
                        shape = shape[1:]
                    new_spaces[key] = gym.spaces.Box(low=low, high=high, shape=shape, dtype=space.dtype)
                else:
                    new_spaces[key] = space
        else:
            new_spaces = {"policy": base}

        for flag in ("is_first", "is_last", "is_terminal", "failure"):
            if flag not in new_spaces:
                new_spaces[flag] = gym.spaces.Box(low=0, high=1, shape=(), dtype=np.bool_)
        return gym.spaces.Dict(new_spaces)

    def _to_tensor(self, value, dtype=None):
        if torch.is_tensor(value):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.to(self.device)

    def _clone_obs(self, obs):
        return {key: (value.clone() if torch.is_tensor(value) else value) for key, value in obs.items()}

    def _convert_obs(self, obs):
        converted = {}
        for key, value in obs.items():
            if torch.is_tensor(value):
                converted[key] = value.to(self.device)
            elif isinstance(value, np.ndarray):
                converted[key] = torch.as_tensor(value, device=self.device)
            else:
                converted[key] = value
        return converted

    def _obs_batch_size(self, obs) -> int:
        tensor = next(value for value in obs.values() if torch.is_tensor(value))
        return int(tensor.shape[0])

    def reset(self, seed=None, options=None):
        if not self._has_reset_once:
            reset_out = self.env.reset(seed=seed, options=options)
            if isinstance(reset_out, tuple):
                obs, _ = reset_out
            else:
                obs = reset_out
            obs = self._convert_obs(obs)
            self._reset_obs_cache = obs
            self._has_reset_once = True
            is_first = torch.ones(self._obs_batch_size(obs), dtype=torch.int32, device=self.device)
        else:
            obs = self._reset_obs_cache
            batch_size = self._obs_batch_size(obs)
            if seed is None:
                is_first = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
            else:
                is_first = self._to_tensor(seed).to(self.device).bool().to(torch.int32)

        obs_out = self._clone_obs(obs)
        batch_size = self._obs_batch_size(obs_out)
        if "failure" not in obs_out:
            obs_out["failure"] = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        else:
            obs_out["failure"] = self._to_tensor(obs_out["failure"]).view(batch_size).to(torch.int32)
        obs_out["is_first"] = is_first
        obs_out["is_last"] = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        obs_out["is_terminal"] = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        return obs_out

    def step(self, action):
        if isinstance(action, dict):
            action = action.get("action", action)

        action = self._to_tensor(action, dtype=torch.float32)
        obs_reset, reward, terminated, truncated, info = self.env.step(action)
        terminated = self._to_tensor(terminated).bool()
        truncated = self._to_tensor(truncated).bool()
        done = terminated | truncated

        obs_reset = self._convert_obs(obs_reset)
        self._reset_obs_cache = obs_reset
        self._has_reset_once = True

        obs_out = self._clone_obs(obs_reset)
        batch_size = self._obs_batch_size(obs_out)
        if "failure" not in obs_out:
            obs_out["failure"] = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        else:
            obs_out["failure"] = self._to_tensor(obs_out["failure"]).view(batch_size).to(torch.int32)
        term_ids = info.get("terminal_env_ids")
        term_obs = info.get("terminal_observation")
        if torch.is_tensor(term_ids) and isinstance(term_obs, dict) and term_ids.numel() > 0:
            term_ids = term_ids.to(self.device)
            for key, value in term_obs.items():
                if key in obs_out and torch.is_tensor(obs_out[key]) and torch.is_tensor(value):
                    obs_out[key][term_ids] = value.to(self.device)

        obs_out["is_first"] = torch.zeros_like(done, dtype=torch.int32, device=self.device)
        obs_out["is_last"] = done.to(torch.int32)
        obs_out["is_terminal"] = terminated.to(torch.int32)
        failure = obs_out["failure"].bool()
        info = dict(info)
        info["episode_success"] = terminated & ~failure
        info["episode_failure"] = terminated & failure
        info["episode_timeout"] = truncated
        reward = self._to_tensor(reward, dtype=torch.float32)
        return obs_out, reward, done, info
