import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PWM_PATH = os.path.join(ROOT, "pwm_isaaclab")
if PWM_PATH not in sys.path:
    sys.path.insert(0, PWM_PATH)

from agents import ActorCriticAgent
from expert_init import pretrain_actor_bc_from_expert, pretrain_world_model_from_expert
from expert_loader import SOURCE_EXPERT, derive_cost_from_force_margin, load_expert_dataset
from expert_replay import HybridExpertReplay, SourceTaggedProprioReplayBuffer, make_expert_replay
from expert_world_model import ExpertWorldModelWithCost, cost_prediction_metrics
from critic_warmup import (
    CriticReplayMixer,
    critic_warmup_step,
    freeze_for_critic_warmup,
    sync_critic_targets,
)


def _write_synthetic_shard(path):
    rng = np.random.default_rng(7)
    lengths = np.asarray([8, 9], dtype=np.int64)
    starts = np.asarray([0, 8], dtype=np.int64)
    total = int(lengths.sum())
    obs = rng.normal(size=(total, 4)).astype(np.float32)
    action = rng.uniform(-0.75, 0.75, size=(total, 2)).astype(np.float32)
    reward = rng.normal(loc=0.1, scale=0.2, size=(total,)).astype(np.float32)
    cost = np.zeros(total, dtype=np.float32)
    cost[3] = 1.0
    done = np.zeros(total, dtype=bool)
    is_first = np.zeros(total, dtype=bool)
    is_last = np.zeros(total, dtype=bool)
    is_terminal = np.zeros(total, dtype=bool)
    for start, length in zip(starts, lengths):
        is_first[start] = True
        done[start + length - 1] = True
        is_last[start + length - 1] = True
        is_terminal[start + length - 1] = True
    force = rng.uniform(0.0, 0.2, size=(total, 6)).astype(np.float32)
    force[3, 4] = 3.0
    force[10, 2] = 2.0
    np.savez_compressed(
        path,
        obs=obs,
        action=action,
        next_obs=obs,
        reward=reward,
        cost=cost,
        force=force,
        constraint_margin=np.ones(total, dtype=np.float32),
        done=done,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        episode_starts=starts,
        episode_lengths=lengths,
        episode_returns=np.asarray([reward[:8].sum(), reward[8:].sum()], dtype=np.float32),
        episode_costs=np.asarray([cost[:8].sum(), cost[8:].sum()], dtype=np.float32),
    )


def _tiny_world_model():
    return ExpertWorldModelWithCost(
        1000,
        True,
        4,
        2,
        2,
        4,
        16,
        8,
        4,
        31,
        5,
        0.5,
        0.1,
        1.0,
        0.1,
        0.99,
        0.95,
        0.01,
        1e-3,
        1e-8,
        False,
        nn.SiLU,
        "cpu",
        False,
    )


def _tiny_agent():
    return ActorCriticAgent(
        2,
        2 * 4 + 16,
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


class ExpertInitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.tmp = tempfile.TemporaryDirectory()
        self.dataset_path = os.path.join(self.tmp.name, "shard_000000.npz")
        _write_synthetic_shard(self.dataset_path)
        self.episodes = load_expert_dataset(self.tmp.name, format="npz")

    def tearDown(self):
        self.tmp.cleanup()

    def test_derive_cost_from_force(self):
        force = np.zeros((3, 6), dtype=np.float32)
        cost, diag = derive_cost_from_force_margin(
            np.zeros(3, dtype=np.float32),
            force,
            cost_target_source="force_margin",
        )
        self.assertTrue(np.allclose(cost, 0.0))
        self.assertEqual(diag["cost_target_source"], "force")

        force[1, 4] = 3.0
        force[2, 2] = 1.5
        cost, diag = derive_cost_from_force_margin(
            np.zeros(3, dtype=np.float32),
            force,
            cost_target_source="force_margin",
        )
        self.assertAlmostEqual(float(cost[0]), 0.0)
        self.assertAlmostEqual(float(cost[1]), 2.0)
        self.assertAlmostEqual(float(cost[2]), 0.5)
        self.assertAlmostEqual(float(diag["pipe_force"][1]), 3.0)
        self.assertAlmostEqual(float(diag["bottom_force"][2]), 1.5)

    def test_load_expert_dataset(self):
        self.assertEqual(len(self.episodes), 2)
        self.assertEqual(self.episodes.metadata["num_transitions"], 17)
        self.assertEqual(self.episodes[0]["obs"].shape, (8, 4))
        self.assertEqual(self.episodes[0]["action"].shape, (8, 2))
        self.assertEqual(self.episodes[0]["reward"].shape, (8,))
        self.assertEqual(self.episodes[0]["cost"].shape, (8,))
        self.assertIn("force", self.episodes[0])
        self.assertEqual(self.episodes[0]["force"].shape, (8, 1))

    def test_load_expert_dataset_with_derived_cost(self):
        episodes = load_expert_dataset(self.tmp.name, format="npz", cost_target_source="force_margin")
        self.assertEqual(episodes.metadata["cost_positive_count"], 2)
        self.assertAlmostEqual(episodes.metadata["cost_positive_ratio"], 2 / 17)
        self.assertAlmostEqual(episodes.metadata["derived_cost_max"], 2.0)
        self.assertAlmostEqual(episodes.metadata["pipe_force_max"], 3.0)
        self.assertAlmostEqual(episodes.metadata["bottom_force_max"], 2.0)
        self.assertAlmostEqual(float(episodes[0]["cost"][3]), 2.0)
        self.assertAlmostEqual(float(episodes[1]["cost"][2]), 1.0)
        self.assertEqual(episodes[0]["cost_target_source"], "force")
        self.assertIn("original_cost", episodes[0])

    def test_replay_add_expert(self):
        replay = make_expert_replay(self.episodes, device="cpu", include_force=True)
        batch = replay.sample(2, 5, source=SOURCE_EXPERT, return_dict=True)
        self.assertTrue(torch.all(batch["source"] == SOURCE_EXPERT))
        self.assertIn("cost", batch)
        self.assertEqual(batch["cost"].shape, (2, 5, 1))
        self.assertEqual(replay.num_expert_steps(), 17)

    def test_recursive_wm_coverage_load_and_mix(self):
        nested_dir = os.path.join(self.tmp.name, "coverage", "random")
        os.makedirs(nested_dir, exist_ok=True)
        _write_synthetic_shard(os.path.join(nested_dir, "shard_000000.npz"))
        coverage = load_expert_dataset(os.path.join(self.tmp.name, "coverage"), format="npz")
        self.assertEqual(len(coverage), 2)
        expert_replay = make_expert_replay(self.episodes, device="cpu", include_force=False)
        wm_replay = make_expert_replay(list(self.episodes) + list(coverage), device="cpu", include_force=False)
        self.assertEqual(expert_replay.num_expert_steps(), 17)
        self.assertEqual(wm_replay.num_expert_steps(), 34)

    def test_hurdle_cost_loss_finite(self):
        world_model = _tiny_world_model()
        outputs = {
            "violation_logit": torch.zeros(2, 5, 1),
            "magnitude_raw": torch.zeros(2, 5, 1),
        }
        all_negative = torch.zeros(2, 5, 1)
        loss, metrics, pred_cost, p_violate = world_model._hurdle_cost_loss(outputs, all_negative)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(pred_cost).all())
        self.assertTrue(torch.isfinite(p_violate).all())
        self.assertAlmostEqual(float(metrics["reg_loss"]), 0.0)

        mixed = all_negative.clone()
        mixed[0, 2, 0] = 2.0
        loss, metrics, _, _ = world_model._hurdle_cost_loss(outputs, mixed)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(metrics["cls_loss"]))
        self.assertTrue(torch.isfinite(metrics["reg_loss"]))

    def test_balanced_cost_sequence_sampling(self):
        episodes = load_expert_dataset(self.tmp.name, format="npz", cost_target_source="force_margin")
        replay = make_expert_replay(episodes, device="cpu", include_force=False)
        batch = replay.sample(
            6,
            5,
            source=SOURCE_EXPERT,
            return_dict=True,
            cost_positive_ratio=0.3,
        )
        self.assertTrue(torch.any(batch["cost"] > 0))

    def test_world_model_pretrain_one_step(self):
        episodes = load_expert_dataset(self.tmp.name, format="npz", cost_target_source="force_margin")
        replay = make_expert_replay(episodes, device="cpu", include_force=False)
        world_model = _tiny_world_model()
        metrics = pretrain_world_model_from_expert(
            replay,
            world_model,
            num_steps=1,
            batch_size=2,
            batch_length=5,
            cost_positive_ratio=0.3,
            progress=False,
        )
        self.assertTrue(np.isfinite(metrics["wm_loss"]))
        self.assertTrue(np.isfinite(metrics["cost_loss"]))
        self.assertIn("cost/cls_loss", metrics)
        self.assertIn("cost/reg_loss", metrics)
        self.assertIn("cost/auprc", metrics)

    def test_actor_bc_one_step(self):
        replay = make_expert_replay(self.episodes, device="cpu", include_force=False)
        world_model = _tiny_world_model()
        agent = _tiny_agent()
        metrics = pretrain_actor_bc_from_expert(
            replay,
            world_model,
            agent,
            num_steps=1,
            batch_size=2,
            batch_length=5,
            progress=False,
        )
        self.assertTrue(np.isfinite(metrics["bc_loss"]))
        self.assertTrue(np.isfinite(metrics["action_mse"]))

    def test_critic_warmup_one_step_only_updates_critic(self):
        replay = make_expert_replay(self.episodes, device="cpu", include_force=False)
        mixer = CriticReplayMixer(
            [("expert", replay), ("random", replay), ("perturb", replay)],
            ratios=[1.0, 1.0, 1.0],
        )
        world_model = _tiny_world_model()
        agent = _tiny_agent()
        freeze_for_critic_warmup(world_model, agent)
        actor_before = [param.detach().clone() for param in agent.actor.parameters()]
        critic_before = [param.detach().clone() for param in agent.critic.parameters()]
        world_before = [param.detach().clone() for param in world_model.parameters()]

        batch = mixer.sample(6, 5)
        optimizer = torch.optim.AdamW(agent.critic.parameters(), lr=1e-3)
        metrics = critic_warmup_step(
            world_model,
            agent,
            batch,
            optimizer,
            imagine_horizon=3,
            bootstrap="zero",
        )
        self.assertTrue(np.isfinite(metrics["critic_loss"]))
        self.assertIn("source/expert/critic_loss", metrics)
        self.assertIn("source/random/critic_loss", metrics)
        self.assertIn("source/perturb/critic_loss", metrics)
        self.assertTrue(any(
            not torch.allclose(before, after)
            for before, after in zip(critic_before, agent.critic.parameters())
        ))
        self.assertTrue(all(
            torch.allclose(before, after)
            for before, after in zip(actor_before, agent.actor.parameters())
        ))
        self.assertTrue(all(
            torch.allclose(before, after)
            for before, after in zip(world_before, world_model.parameters())
        ))

    def test_critic_replay_mixer_balanced_sources(self):
        replay = make_expert_replay(self.episodes, device="cpu", include_force=False)
        mixer = CriticReplayMixer(
            [("expert", replay), ("random", replay), ("perturb", replay)],
            ratios=[1.0, 1.0, 1.0],
        )
        batch = mixer.sample(6, 5)
        self.assertEqual(batch["obs"].shape[0], 6)
        self.assertEqual(batch["source_counts"], {"expert": 2, "random": 2, "perturb": 2})
        self.assertEqual(batch["source_names"], ["expert", "random", "perturb"])

    def test_sync_critic_targets_copies_critic_to_slow_critic(self):
        agent = _tiny_agent()
        with torch.no_grad():
            for param in agent.critic.parameters():
                param.add_(torch.randn_like(param))
        sync_critic_targets(agent)
        self.assertTrue(all(
            torch.allclose(critic_param, slow_param)
            for critic_param, slow_param in zip(agent.critic.parameters(), agent.slow_critic.parameters())
        ))

    def test_skip_random_prefill(self):
        expert_replay = make_expert_replay(self.episodes, device="cpu", include_force=False)
        online_replay = SourceTaggedProprioReplayBuffer(
            obs_dim=4,
            action_dim=2,
            num_envs=2,
            max_length=100,
            warmup_length=50,
            device="cpu",
        )
        hybrid = HybridExpertReplay(online_replay, expert_replay, replace_random_prefill=True)
        self.assertFalse(online_replay.ready())
        self.assertTrue(hybrid.ready())
        self.assertTrue(hybrid.can_sample(5))

    def test_cost_eval_positive_metrics(self):
        target = torch.tensor([0.0, 1.0, 0.0, 2.0]).view(1, 4, 1)
        p_violate = torch.tensor([0.1, 0.8, 0.2, 0.9]).view(1, 4, 1)
        pred_cost = torch.tensor([0.0, 1.1, 0.0, 1.8]).view(1, 4, 1)
        metrics = cost_prediction_metrics(pred_cost, p_violate, target, prefix="cost")
        self.assertAlmostEqual(metrics["cost/positive_ratio"], 0.5)
        self.assertAlmostEqual(metrics["cost/precision@0.5"], 1.0)
        self.assertAlmostEqual(metrics["cost/recall@0.5"], 1.0)
        self.assertGreater(metrics["cost/auprc"], 0.99)
        self.assertFalse(metrics["cost/no_positive_samples"])

        no_pos = cost_prediction_metrics(pred_cost * 0.0, p_violate * 0.0, target * 0.0, prefix="cost")
        self.assertIsNone(no_pos["cost/auprc"])
        self.assertIsNone(no_pos["cost/mae_positive_only"])
        self.assertTrue(no_pos["cost/no_positive_samples"])


if __name__ == "__main__":
    unittest.main()
