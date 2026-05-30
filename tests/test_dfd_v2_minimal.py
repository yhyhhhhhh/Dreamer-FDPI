import os
import sys
import unittest

import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pwm_isaaclab_dfd_v2.agent_dfd_v2 import CostAwareActorCriticAgent
from pwm_isaaclab_dfd_v2.cost_utils import (
    SOURCE_DUAL,
    SOURCE_MAIN,
    compute_continuous_cost,
    extract_continuous_cost,
)
from pwm_isaaclab_dfd_v2.dual_imagination import update_dual_in_imagination
from pwm_isaaclab_dfd_v2.dual_policy_v2 import DualPolicyV2
from pwm_isaaclab_dfd_v2.gd_risk import GdRiskCritic
from pwm_isaaclab_dfd_v2.replay_buffer_dfd_v2 import DFDV2ReplayBuffer
from pwm_isaaclab_dfd_v2.trainer_dfd_v2 import _sample_policy_action, train_agent_step_dfd_v2
from pwm_isaaclab_dfd_v2.world_model_dfd_v2 import ContinuousCostHead, DFDV2WorldModelWithContinuousCost


class _FakeDynamic:
    def __init__(self, feat_dim, action_dim):
        self.feat_dim = int(feat_dim)
        self.action_dim = int(action_dim)

    def parallel_observe(self, embed, action, is_first):
        batch, horizon = action.shape[:2]
        base = embed[..., : self.feat_dim]
        if base.shape[1] != horizon:
            base = base[:, :horizon]
        post = {"feat": base.contiguous()}
        return post, post, None, None

    def get_feat(self, state):
        return state["feat"]

    def img_step(self, state, action):
        pad = torch.zeros(action.shape[0], self.feat_dim, dtype=action.dtype, device=action.device)
        pad[:, : self.action_dim] = action
        return {"feat": torch.tanh(state["feat"] + pad)}


class _FakeWorldModel(nn.Module):
    def __init__(self, feat_dim=4, action_dim=2):
        super().__init__()
        self.device = "cpu"
        self.device_type = "cpu"
        self.tensor_dtype = torch.float32
        self.use_amp = False
        self.dynamic = _FakeDynamic(feat_dim, action_dim)
        self.marker = nn.Parameter(torch.tensor(1.0))

    def encoder(self, obs):
        return obs

    def eval(self):
        super().eval()
        return self


class _FakeMainAgent(nn.Module):
    def __init__(self, feat_dim=4, action_dim=2):
        super().__init__()
        self.actor = nn.Linear(feat_dim, 2 * action_dim)
        self.std_scale = 0.9
        self.std_offset = 0.1
        self.device_type = "cpu"
        self.tensor_dtype = torch.float32
        self.use_amp = False

    def sample(self, feat, greedy=False):
        mean, std = self.actor(feat).chunk(2, dim=-1)
        if greedy:
            return torch.tanh(mean)
        std = self.std_scale * torch.sigmoid(std + 2) + self.std_offset
        return torch.tanh(mean) + 0.0 * std


class _ActionRisk(nn.Module):
    def forward(self, feat, action):
        return action.pow(2).sum(dim=-1, keepdim=True)


class _FakeGd(nn.Module):
    def __init__(self):
        super().__init__()
        self.gd1 = _ActionRisk()
        self.gd2 = _ActionRisk()
        self.risk_max = 10.0
        self.anchor = nn.Parameter(torch.tensor(2.0))


class _RecorderAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_reward = None

    def sample(self, feat, greedy=False):
        return torch.zeros(feat.shape[0], 2, dtype=feat.dtype, device=feat.device)

    def update(self, feat, action, discount, reward, weight, logger=None, step=None):
        self.last_reward = reward.detach().clone()


class _CostWorldModel(_FakeWorldModel):
    def imagine_data(self, agent, obs, action, reward, done, is_first, horizon, logger=None, step=None):
        batch = obs.shape[0]
        feat = torch.zeros(batch, horizon + 1, 4)
        imagined_action = torch.zeros(batch, horizon, 2)
        discount = torch.ones(batch, horizon, 1) * 0.99
        imagined_reward = torch.ones(batch, horizon, 1)
        weight = torch.ones(batch, horizon, 1)
        return feat, imagined_action, discount, imagined_reward, weight

    def predict_cost(self, feat):
        pred = torch.full((*feat.shape[:-1], 1), 0.25)
        return pred, pred, pred


class _HalfMainAgent:
    def sample(self, feat, greedy=False):
        return torch.zeros(feat.shape[0], 2, dtype=torch.float16, device=feat.device)


class _FloatDualPolicy:
    def sample(self, feat, greedy=False):
        return torch.ones(feat.shape[0], 2, dtype=torch.float32, device=feat.device)


class _StateWorldModel:
    def update_inference_state(self, state, action):
        state["action_dtype"] = action.dtype
        return state


class DFDV2MinimalTests(unittest.TestCase):
    def test_continuous_cost_and_replay_fields(self):
        parts = compute_continuous_cost(
            torch.tensor([0.05, 0.1, 0.3, 1.0, 10.0, 15.0]),
            force_threshold=0.1,
            low_force_scale=0.05,
            cost_force_max=15.0,
            extreme_force_threshold=5.0,
            clip_cost=True,
        )
        cost = parts["continuous_cost"].view(-1)
        self.assertAlmostEqual(float(cost[0]), 0.0, places=6)
        self.assertAlmostEqual(float(cost[1]), 0.0, places=6)
        self.assertTrue(torch.all(cost[2:] > cost[1:-1]))
        self.assertLess(float(cost[4]), float(cost[5]))
        self.assertTrue(torch.equal(parts["binary_cost"].view(-1), torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])))
        self.assertTrue(torch.equal(parts["extreme_cost"].view(-1), torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])))

        obs = {"force": torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 2.0]])}
        extracted = extract_continuous_cost(
            {},
            obs,
            num_envs=1,
            device="cpu",
            force_threshold=0.1,
            low_force_scale=0.05,
            cost_force_max=15.0,
            extreme_force_threshold=5.0,
            bottom_force_channels=(2, 5),
        )
        self.assertAlmostEqual(float(extracted["bottom_force"][0, 0]), 2.0)
        self.assertGreater(float(extracted["continuous_cost"][0, 0]), 0.0)
        self.assertLess(float(extracted["continuous_cost"][0, 0]), 1.0)

        replay = DFDV2ReplayBuffer(3, 2, 1, max_length=16, warmup_length=0, device="cpu")
        for step in range(6):
            source = SOURCE_DUAL if step >= 3 else SOURCE_MAIN
            cost = torch.tensor([[0.2 if source == SOURCE_DUAL else 0.0]])
            replay.append(
                torch.full((1, 3), float(step)),
                torch.zeros(1, 2),
                torch.ones(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, 1),
                continuous_cost=cost,
                binary_cost=(cost > 0).float(),
                extreme_cost=torch.tensor([[1.0 if step == 5 else 0.0]]),
                bottom_force=torch.tensor([[1.0 + float(step)]]),
                force_excess=cost * 5.0,
                source=torch.tensor([[source]]),
            )
        batch = replay.sample(1, 3, return_dict=True)
        self.assertIn("continuous_cost", batch)
        self.assertIn("binary_cost", batch)
        self.assertIn("extreme_cost", batch)
        self.assertIn("bottom_force", batch)
        self.assertTrue(torch.equal(batch["cost"], batch["continuous_cost"]))
        stats = replay.cost_stats()
        self.assertGreater(stats["dual_cost_mean"], stats["main_cost_mean"])
        self.assertGreater(stats["dual_extreme_rate"], stats["main_extreme_rate"])

    def test_continuous_cost_head_outputs_and_weighting(self):
        head = ContinuousCostHead(input_dim=4, hidden_dim=8, act=nn.SiLU, depth=1)
        feat = torch.randn(2, 3, 4)
        outputs = head(feat)
        self.assertEqual(outputs["cost_logit"].shape, (2, 3, 1))
        self.assertEqual(outputs["extreme_logit"].shape, (2, 3, 1))

        model = DFDV2WorldModelWithContinuousCost.__new__(DFDV2WorldModelWithContinuousCost)
        nn.Module.__init__(model)
        model.cost_head_enable = True
        model.cost_head = head
        model.cost_loss_weight = 1.0
        model.cost_huber_beta = 0.02
        model.small_force_threshold = 0.3
        model.small_cost_threshold = 0.05
        model.small_cost_weight = 2.0
        model.extreme_loss_weight = 0.5
        model.extreme_cost_weight = 4.0
        model.extreme_force_threshold = 5.0

        cost_target = torch.tensor([[[0.0], [0.05], [0.8]], [[0.0], [0.2], [1.0]]])
        bottom_force = torch.tensor([[[0.05], [0.2], [10.0]], [[0.1], [0.4], [15.0]]])
        extreme_cost = (bottom_force > 5.0).float()
        loss, metrics, pred, extreme_prob = model._continuous_cost_loss(
            feat,
            cost_target,
            bottom_force=bottom_force,
            extreme_cost=extreme_cost,
        )
        self.assertEqual(pred.shape, cost_target.shape)
        self.assertEqual(extreme_prob.shape, cost_target.shape)
        self.assertGreaterEqual(float(pred.detach().min()), 0.0)
        self.assertLessEqual(float(pred.detach().max()), 1.0)
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreater(float(metrics["small_ratio"]), 0.0)
        self.assertGreater(float(metrics["extreme_ratio"]), 0.0)

    def test_gd_target_has_no_one_minus_cost_truncation_and_weights(self):
        torch.manual_seed(1)
        world_model = _FakeWorldModel(feat_dim=3, action_dim=2)
        dual = _FloatDualPolicy()
        gd = GdRiskCritic(
            feat_dim=3,
            action_dim=2,
            hidden_dim=8,
            num_layers=0,
            gamma_cost=1.0,
            target_tau=0.5,
            risk_max=10.0,
            lr=0.0,
            eps=1e-8,
            use_amp=False,
            act=nn.SiLU,
            device="cpu",
            source_aware_weight=False,
        )
        for module in (gd.gd1, gd.gd2):
            module.head.weight.data.zero_()
            module.head.bias.data.zero_()
        for module in (gd.target_gd1, gd.target_gd2):
            module.head.weight.data.zero_()
            module.head.bias.data.fill_(0.5)

        batch = {
            "obs": torch.ones(1, 4, 3),
            "action": torch.zeros(1, 4, 2),
            "continuous_cost": torch.ones(1, 4, 1) * 0.5,
            "done": torch.zeros(1, 4, 1),
            "source": torch.full((1, 4, 1), SOURCE_MAIN, dtype=torch.int64),
            "extreme_cost": torch.zeros(1, 4, 1),
            "is_first": torch.zeros(1, 4, 1),
        }
        info = gd.update(batch, world_model, dual)
        self.assertAlmostEqual(info["target_mean"], 1.0, places=5)
        self.assertAlmostEqual(info["loss"], 2.0, places=5)
        self.assertAlmostEqual(float(gd.target_gd1.head.bias.detach()[0]), 0.25, places=5)

        weighted = GdRiskCritic(
            feat_dim=3,
            action_dim=2,
            hidden_dim=8,
            num_layers=0,
            gamma_cost=1.0,
            target_tau=0.5,
            risk_max=10.0,
            lr=0.0,
            eps=1e-8,
            use_amp=False,
            act=nn.SiLU,
            device="cpu",
            source_aware_weight=True,
            dual_source_weight=2.0,
            high_cost_weight=3.0,
            high_cost_threshold=0.1,
        )
        weights = weighted._weights(
            torch.tensor([[0.0], [0.2], [0.2]]),
            torch.tensor([[SOURCE_MAIN], [SOURCE_MAIN], [SOURCE_DUAL]]),
        )
        self.assertTrue(torch.allclose(weights.view(-1), torch.tensor([1.0, 3.0, 6.0])))

    def test_dual_imagination_updates_only_dual_policy(self):
        torch.manual_seed(3)
        world_model = _FakeWorldModel(feat_dim=4, action_dim=2)
        main = _FakeMainAgent(feat_dim=4, action_dim=2)
        gd = _FakeGd()
        dual = DualPolicyV2(
            action_dim=2,
            feat_dim=4,
            hidden=16,
            min_std=0.1,
            max_std=1.0,
            lr=1e-2,
            eps=1e-8,
            use_amp=False,
            act=nn.SiLU,
            device="cpu",
        )
        batch = {
            "obs": torch.randn(2, 3, 4),
            "action": torch.zeros(2, 3, 2),
            "is_first": torch.zeros(2, 3, 1),
        }
        world_before = [p.detach().clone() for p in world_model.parameters()]
        main_before = [p.detach().clone() for p in main.parameters()]
        gd_before = [p.detach().clone() for p in gd.parameters()]
        dual_before = [p.detach().clone() for p in dual.parameters()]

        info = update_dual_in_imagination(
            batch,
            world_model,
            main,
            gd,
            dual,
            {"Horizon": 3, "Objective": "max_risk", "KLCoeff": 0.0, "EntropyCoef": 0.0, "GradClipNorm": 100.0},
        )
        self.assertGreater(info["gd_score"], 0.0)
        self.assertGreater(info["grad_norm"], 0.0)
        for before, after in zip(world_before, world_model.parameters()):
            self.assertTrue(torch.equal(before, after))
        for before, after in zip(main_before, main.parameters()):
            self.assertTrue(torch.equal(before, after))
        for before, after in zip(gd_before, gd.parameters()):
            self.assertTrue(torch.equal(before, after))
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(dual_before, dual.parameters())))

    def test_main_cost_aware_reward_only_changes_reward_tensor(self):
        world_model = _CostWorldModel(feat_dim=4, action_dim=2)
        agent = _RecorderAgent()
        samples = (
            torch.zeros(2, 3, 4),
            torch.zeros(2, 3, 2),
            torch.zeros(2, 3, 1),
            torch.zeros(2, 3, 1),
            torch.zeros(2, 3, 1),
        )
        train_agent_step_dfd_v2(
            samples,
            world_model,
            agent,
            4,
            logger=None,
            step=10,
            dfd_cfg={
                "MainCostAwareReward": {"Enable": True, "StartStep": 0, "LambdaCost": 0.2},
                "ContinuousCost": {"CostMin": 0.0, "CostMax": 1.0},
            },
        )
        self.assertTrue(torch.allclose(agent.last_reward, torch.full((2, 4, 1), 0.95)))

        train_agent_step_dfd_v2(
            samples,
            world_model,
            agent,
            4,
            logger=None,
            step=10,
            dfd_cfg={
                "MainCostAwareReward": {"Enable": False, "StartStep": 0, "LambdaCost": 0.2},
                "ContinuousCost": {"CostMin": 0.0, "CostMax": 1.0},
            },
        )
        self.assertTrue(torch.allclose(agent.last_reward, torch.ones(2, 4, 1)))

    def test_dual_action_casts_to_main_action_dtype(self):
        feat = torch.zeros(4, 3)
        env_action, action, source, state = _sample_policy_action(
            feat=feat,
            agent=_HalfMainAgent(),
            dual_policy=_FloatDualPolicy(),
            world_model=_StateWorldModel(),
            state={},
            use_dual_sampling=True,
            dual_ratio=1.0,
            num_envs=4,
            device="cpu",
        )
        self.assertEqual(action.dtype, torch.float16)
        self.assertEqual(state["action_dtype"], torch.float16)
        self.assertTrue(torch.equal(source.view(-1), torch.full((4,), SOURCE_DUAL, dtype=torch.int64)))


if __name__ == "__main__":
    unittest.main()
