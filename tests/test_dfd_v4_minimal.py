import os
import sys
import unittest

import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pwm_isaaclab_dfd_v4.agent_fdpi_regime import fdpi_regime_loss_components
from pwm_isaaclab_dfd_v4.cost_utils import (
    SOURCE_DUAL,
    SOURCE_MAIN,
    compute_continuous_cost,
    extract_continuous_cost,
)
from pwm_isaaclab_dfd_v4.dual_policy_v4 import DualPolicyV4
from pwm_isaaclab_dfd_v4.dual_update_v4 import update_dual_v4
from pwm_isaaclab_dfd_v4.replay_buffer_dfd_v4 import DFDV4ReplayBuffer
from pwm_isaaclab_dfd_v4.risk_critics import GdRiskCriticV4, GpRiskCritic
from pwm_isaaclab_dfd_v4.sampling_utils import FDPIRegimeStats, dual_ratio_from_fdpi_stats


class _FakeDynamic:
    def __init__(self, feat_dim, action_dim):
        self.feat_dim = int(feat_dim)
        self.action_dim = int(action_dim)

    def parallel_observe(self, embed, action, is_first):
        del is_first
        batch, horizon = action.shape[:2]
        feat = embed[..., : self.feat_dim]
        if feat.shape[1] != horizon:
            feat = feat[:, :horizon]
        post = {"feat": feat.contiguous()}
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

    def encoder(self, obs):
        return obs

    def predict_cost(self, feat):
        pred = torch.sigmoid(feat[..., :1])
        return pred, pred, pred


class _FakePolicy(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        self.action_dim = action_dim

    def sample(self, feat, greedy=False):
        del greedy
        return torch.ones(feat.shape[0], self.action_dim, dtype=feat.dtype, device=feat.device) * 0.25


class _ConstantRisk(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, feat, action):
        return torch.full(action[..., :1].shape, self.value, dtype=feat.dtype, device=feat.device)


class _FakeGd(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_cost = 0.97
        self.risk_max = 1.0

    def risk(self, feat, action, clamp=True):
        value = action.pow(2).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        return value if clamp else value


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


class DFDV4MinimalTests(unittest.TestCase):
    def test_v3_style_continuous_cost_and_bottom_force_extraction(self):
        bottom_force = torch.tensor([0.05, 0.1, 0.3, 1.0, 10.0, 20.0])
        parts = compute_continuous_cost(
            bottom_force,
            force_threshold=0.1,
            low_force_scale=0.05,
            cost_force_max=15.0,
            force_scale=5.0,
            extreme_force_threshold=5.0,
            clip_cost=True,
        )
        expected = torch.log1p(torch.relu(bottom_force - 0.1) / 0.05) / torch.log1p(torch.tensor(15.0 / 0.05))
        expected = expected.clamp(0.0, 1.0)
        self.assertTrue(torch.allclose(parts["continuous_cost"].view(-1), expected))
        self.assertEqual(float(parts["continuous_cost"][0, 0]), 0.0)
        self.assertEqual(float(parts["continuous_cost"][1, 0]), 0.0)
        self.assertTrue(torch.all(parts["continuous_cost"].view(-1)[2:].diff() > 0.0))
        self.assertTrue(torch.equal(parts["binary_cost"].view(-1), torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])))
        self.assertTrue(torch.equal(parts["extreme_cost"].view(-1), torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])))

        obs = {"force": torch.tensor([[0.0, 0.0, 1.5, 0.0, 0.0, 3.0]])}
        extracted = extract_continuous_cost(
            {},
            obs,
            num_envs=1,
            device="cpu",
            force_threshold=0.1,
            low_force_scale=0.05,
            cost_force_max=15.0,
            bottom_force_channels=(2, 5),
        )
        self.assertAlmostEqual(float(extracted["bottom_force"][0, 0]), 3.0)
        expected_extracted = torch.log1p(torch.tensor((3.0 - 0.1) / 0.05)) / torch.log1p(torch.tensor(15.0 / 0.05))
        self.assertAlmostEqual(float(extracted["continuous_cost"][0, 0]), float(expected_extracted), places=6)

    def test_replay_fields_and_safety_critical_sampling(self):
        replay = DFDV4ReplayBuffer(3, 2, 1, max_length=32, warmup_length=0, device="cpu")
        for step in range(12):
            source = SOURCE_DUAL if step >= 8 else SOURCE_MAIN
            cost = torch.tensor([[0.3 if source == SOURCE_DUAL or step == 5 else 0.0]])
            replay.append(
                torch.full((1, 3), float(step)),
                torch.zeros(1, 2),
                torch.ones(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, 1),
                continuous_cost=cost,
                binary_cost=(cost > 0).float(),
                extreme_cost=torch.zeros(1, 1),
                bottom_force=torch.ones(1, 1) + cost,
                force_excess=cost * 5.0,
                source=torch.tensor([[source]]),
            )
        batch = replay.sample(
            4,
            3,
            return_dict=True,
            safety_critical_ratio=0.5,
            high_cost_threshold=0.1,
            boundary_low=0.05,
            boundary_high=0.4,
        )
        self.assertIn("continuous_cost", batch)
        self.assertIn("binary_cost", batch)
        self.assertIn("bottom_force", batch)
        self.assertIn("force_excess", batch)
        self.assertIn("source", batch)
        self.assertTrue(torch.equal(batch["cost"], batch["continuous_cost"]))
        self.assertGreater(float((batch["continuous_cost"] > 0.1).float().mean()), 0.0)
        self.assertGreater(replay.cost_stats()["dual_cost_mean"], replay.cost_stats()["main_cost_mean"])

    def test_gp_gd_targets_use_max_min_without_one_minus_cost(self):
        torch.manual_seed(1)
        world_model = _FakeWorldModel(feat_dim=3, action_dim=2)
        policy = _FakePolicy(action_dim=2)
        batch = {
            "obs": torch.ones(2, 3, 3),
            "action": torch.zeros(2, 3, 2),
            "continuous_cost": torch.ones(2, 3, 1) * 0.5,
            "done": torch.zeros(2, 3, 1),
            "is_first": torch.zeros(2, 3, 1),
            "source": torch.zeros(2, 3, 1, dtype=torch.long),
        }
        gp = GpRiskCritic(3, 2, 8, 0, 1.0, 0.0, 10.0, 1e-4, 1e-8, False, nn.SiLU, "cpu")
        gd = GdRiskCriticV4(3, 2, 8, 0, 1.0, 0.0, 10.0, 1e-4, 1e-8, False, nn.SiLU, "cpu")
        gp.target_critic1 = _ConstantRisk(0.2)
        gp.target_critic2 = _ConstantRisk(0.8)
        gd.target_critic1 = _ConstantRisk(0.2)
        gd.target_critic2 = _ConstantRisk(0.8)
        gp_info = gp.update(batch, world_model, policy)
        gd_info = gd.update(batch, world_model, policy)
        self.assertAlmostEqual(gp_info["target_mean"], 1.3, places=5)
        self.assertAlmostEqual(gd_info["target_mean"], 0.7, places=5)

    def test_fdpi_regime_loss_segments_are_finite(self):
        log_prob = torch.zeros(1, 3, 1, requires_grad=True)
        entropy = torch.ones(1, 3, 1)
        norm_adv = torch.ones(1, 3, 1)
        g = torch.tensor([[[0.04], [0.08], [0.15]]], requires_grad=True)
        loss, metrics = fdpi_regime_loss_components(
            log_prob=log_prob,
            entropy=entropy,
            norm_adv=norm_adv,
            g=g,
            weight=torch.ones(1, 3, 1),
            pf=0.10,
            cg=0.03,
            lambda_cri=0.02,
            lambda_inf=0.05,
            risk_max=1.0,
            entropy_coef=1.0e-4,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(metrics["fea_ratio"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(metrics["cri_ratio"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(metrics["inf_ratio"]), 1.0 / 3.0)

    def test_dual_ratio_uses_fdpi_stats_and_safety_caps(self):
        cfg = {
            "Enable": True,
            "StartStep": 10,
            "MaxKLForSampling": 2.0,
            "RatioFea95": 0.50,
            "RatioFea90": 0.35,
            "RatioFea80": 0.20,
            "RatioCriticalHigh": 0.15,
            "RatioUnsafeHigh": 0.05,
            "RatioDefault": 0.10,
            "HighMainCostRate": 0.20,
            "MaxRatioWhenMainCostHigh": 0.10,
        }
        ratio, _ = dual_ratio_from_fdpi_stats(
            step=20,
            cfg=cfg,
            stats=FDPIRegimeStats(fea_ratio=0.96, cri_ratio=0.0, inf_ratio=0.0, main_real_cost_rate=0.0, count=100),
            last_dual_kl=0.1,
        )
        self.assertAlmostEqual(ratio, 0.50)
        ratio, _ = dual_ratio_from_fdpi_stats(
            step=20,
            cfg=cfg,
            stats=FDPIRegimeStats(fea_ratio=0.96, cri_ratio=0.0, inf_ratio=0.0, main_real_cost_rate=0.5, count=100),
            last_dual_kl=0.1,
        )
        self.assertAlmostEqual(ratio, 0.10)
        ratio, _ = dual_ratio_from_fdpi_stats(
            step=20,
            cfg=cfg,
            stats=FDPIRegimeStats(fea_ratio=0.96, cri_ratio=0.0, inf_ratio=0.0, main_real_cost_rate=0.0, count=100),
            last_dual_kl=3.0,
        )
        self.assertAlmostEqual(ratio, 0.0)

    def test_dual_update_smoke_only_updates_dual_policy(self):
        torch.manual_seed(3)
        world_model = _FakeWorldModel(feat_dim=4, action_dim=2)
        main = _FakeMainAgent(feat_dim=4, action_dim=2)
        dual = DualPolicyV4(2, 4, 16, 0.1, 1.0, 1e-3, 1e-8, False, nn.SiLU, "cpu")
        gd = _FakeGd()
        batch = {
            "obs": torch.randn(2, 3, 4),
            "action": torch.zeros(2, 3, 2),
            "is_first": torch.zeros(2, 3, 1),
        }
        before = [p.detach().clone() for p in dual.parameters()]
        info = update_dual_v4(
            batch,
            world_model,
            main,
            gd,
            dual,
            {"Type": "imagined_risk_return", "Horizon": 2, "KLCoeff": 0.1, "EntropyCoef": 1e-4},
            cost_cfg={"CostMin": 0.0, "CostMax": 1.0},
        )
        self.assertIn("kl_to_main", info)
        after = list(dual.parameters())
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(after, before)))


if __name__ == "__main__":
    unittest.main()
