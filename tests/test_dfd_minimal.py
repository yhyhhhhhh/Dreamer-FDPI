import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pwm_isaaclab_dfd.agent_dfd import RiskConditionedActorCriticAgent
from pwm_isaaclab_dfd.dual_policy import DualPolicy
from pwm_isaaclab_dfd.replay_buffer_dfd import DFDReplayBuffer
from pwm_isaaclab_dfd.trainer_dfd import _sample_policy_action
from pwm_isaaclab_dfd.utils import (
    SOURCE_DUAL,
    SOURCE_MAIN,
    extract_bottom_force_cost,
    risk_advantage_modifier,
)


class _FakeFeasibility:
    def __init__(self, risk):
        self.risk = risk

    def gp1(self, feat, action):
        return self.risk.to(feat.device).reshape_as(action[..., :1])

    def gp2(self, feat, action):
        return self.risk.to(feat.device).reshape_as(action[..., :1]) - 0.01


class _HalfMainAgent:
    def sample(self, feat, greedy=False):
        return torch.zeros(feat.shape[0], 2, dtype=torch.float16, device=feat.device)


class _FloatDualPolicy:
    def sample(self, feat, greedy=False):
        return torch.ones(feat.shape[0], 2, dtype=torch.float32, device=feat.device)


class _FakeWorldModel:
    def update_inference_state(self, state, action):
        state["action_dtype"] = action.dtype
        return state


class _DualCombineFeasibility:
    def gd1(self, feat, action):
        return torch.full(action[..., :1].shape, 0.2, device=action.device)

    def gd2(self, feat, action):
        return torch.full(action[..., :1].shape, 0.7, device=action.device)


class DFDMinimalTests(unittest.TestCase):
    def test_replay_default_and_dict_samples(self):
        replay = DFDReplayBuffer(
            obs_dim=3,
            action_dim=2,
            num_envs=2,
            max_length=20,
            warmup_length=0,
            device="cpu",
        )
        for step in range(6):
            obs = torch.full((2, 3), float(step))
            action = torch.full((2, 2), 0.1 * step)
            reward = torch.ones(2)
            done = torch.zeros(2, dtype=torch.bool)
            is_first = torch.zeros(2, 1)
            cost = torch.tensor([[0.0], [1.0 if step % 2 == 0 else 0.0]])
            source = torch.tensor([[SOURCE_MAIN], [SOURCE_DUAL if step >= 3 else SOURCE_MAIN]])
            replay.append(obs, action, reward, done, is_first, cost=cost, source=source)

        samples = replay.sample(2, 3)
        self.assertEqual(len(samples), 5)
        self.assertEqual(samples[0].shape, (2, 3, 3))
        self.assertEqual(samples[1].shape, (2, 3, 2))

        batch = replay.sample(2, 3, return_dict=True)
        self.assertEqual(batch["cost"].shape, (2, 3, 1))
        self.assertEqual(batch["source"].shape, (2, 3, 1))
        self.assertIn("dual", replay.source_stats())

    def test_bottom_force_cost_uses_bottom_channels_only(self):
        obs = {
            "force": torch.tensor(
                [
                    [0.0, 0.0, 0.9, 0.0, 0.0, 0.8],
                    [0.0, 0.0, 1.2, 0.0, 0.0, 0.5],
                    [0.0, 0.0, 0.2, 0.0, 0.0, 1.5],
                ],
                dtype=torch.float32,
            )
        }
        cost = extract_bottom_force_cost(
            {},
            obs,
            num_envs=3,
            device="cpu",
            threshold=1.0,
            bottom_force_channels=(2, 5),
        )
        self.assertTrue(torch.equal(cost.view(-1), torch.tensor([0.0, 1.0, 1.0])))

    def test_risk_advantage_modifier_segments(self):
        feat = torch.zeros(1, 3, 4)
        action = torch.zeros(1, 3, 2)
        norm_adv = torch.ones(1, 3, 1)
        risk = torch.tensor([[[0.05], [0.08], [0.12]]])
        safe_adv = risk_advantage_modifier(
            feat=feat,
            action=action,
            norm_adv=norm_adv,
            weight=torch.ones_like(norm_adv),
            feasibility=_FakeFeasibility(risk),
            pf=0.10,
            cg=0.03,
            lambda_cri=0.5,
            lambda_inf=0.25,
        )
        self.assertAlmostEqual(float(safe_adv[0, 0, 0]), 1.0)
        self.assertLess(float(safe_adv[0, 1, 0]), 1.0)
        self.assertLess(float(safe_adv[0, 2, 0]), 0.0)

    def test_agent_update_without_risk_hook_runs(self):
        torch.manual_seed(5)
        agent = RiskConditionedActorCriticAgent(
            2,
            6,
            16,
            1e-3,
            31,
            5,
            0.05,
            0.95,
            0.1,
            1.0,
            0.99,
            0.99,
            0.95,
            0.01,
            False,
            1e-3,
            1e-8,
            False,
            nn.SiLU,
            "cpu",
        )
        feat = torch.randn(2, 4, 6)
        action = torch.randn(2, 3, 2) * 0.1
        discount = torch.ones(2, 3, 1) * 0.99
        reward = torch.randn(2, 3, 1) * 0.1
        weight = torch.ones(2, 3, 1)
        agent.update(feat, action, discount, reward, weight, advantage_modifier_fn=None)

    def test_dual_action_casts_to_main_action_dtype(self):
        torch.manual_seed(7)
        feat = torch.zeros(4, 3)
        env_action, action, source, state = _sample_policy_action(
            feat=feat,
            agent=_HalfMainAgent(),
            dual_policy=_FloatDualPolicy(),
            world_model=_FakeWorldModel(),
            state={},
            use_dual_sampling=True,
            dual_ratio=1.0,
            num_envs=4,
            device="cpu",
        )
        self.assertEqual(action.dtype, torch.float16)
        self.assertEqual(state["action_dtype"], torch.float16)
        self.assertTrue(torch.equal(source.view(-1), torch.full((4,), SOURCE_DUAL, dtype=torch.int64)))
        self.assertEqual(env_action.dtype, np.float16)

    def test_dual_actor_initializes_from_main_actor(self):
        torch.manual_seed(11)
        main = RiskConditionedActorCriticAgent(
            2,
            6,
            16,
            1e-3,
            31,
            5,
            0.05,
            0.95,
            0.1,
            1.0,
            0.99,
            0.99,
            0.95,
            0.01,
            False,
            1e-3,
            1e-8,
            False,
            nn.SiLU,
            "cpu",
        )
        dual = DualPolicy(
            action_dim=2,
            feat_dim=6,
            hidden=16,
            min_std=0.1,
            max_std=1.0,
            lr=1e-3,
            eps=1e-8,
            lambda_kl_init=1.0,
            lambda_lr=1e-3,
            use_amp=False,
            act=nn.SiLU,
            device="cpu",
        )
        dual.initialize_from_main_actor(main)
        for main_param, dual_param in zip(main.actor.parameters(), dual.actor.parameters()):
            self.assertTrue(torch.equal(main_param, dual_param))

    def test_dual_gd_objective_max_is_aggressive(self):
        dual = DualPolicy(
            action_dim=2,
            feat_dim=6,
            hidden=16,
            min_std=0.1,
            max_std=1.0,
            lr=1e-3,
            eps=1e-8,
            lambda_kl_init=1.0,
            lambda_lr=1e-3,
            use_amp=False,
            act=nn.SiLU,
            device="cpu",
            gd_objective="max",
            dual_g_scale=3.0,
        )
        feat = torch.zeros(4, 6)
        action = torch.zeros(4, 2)
        gd1 = _DualCombineFeasibility().gd1(feat, action)
        gd2 = _DualCombineFeasibility().gd2(feat, action)
        self.assertTrue(torch.allclose(dual._combine_gd(gd1, gd2), torch.full((4, 1), 0.7)))
        self.assertEqual(dual.dual_g_scale, 3.0)


if __name__ == "__main__":
    unittest.main()
