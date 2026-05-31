import unittest

import torch

from pwm_isaaclab_dfd_v4_lite.agent_fdpi_lite import fdpi_lite_loss_components


class TestDFDV4LiteMinimal(unittest.TestCase):
    def test_lite_loss_is_reward_plus_single_gp_penalty(self):
        log_prob = torch.tensor([[[-2.0], [-1.0], [-0.5]]])
        entropy = torch.tensor([[[0.1], [0.2], [0.3]]])
        norm_adv = torch.ones_like(log_prob)
        g = torch.tensor([[[0.2], [0.5], [1.0]]])
        weight = torch.ones_like(log_prob)

        total, metrics = fdpi_lite_loss_components(
            log_prob=log_prob,
            entropy=entropy,
            norm_adv=norm_adv,
            g=g,
            weight=weight,
            pf=0.4,
            lambda_gp=0.6,
            lambda_gp_scale=0.5,
            risk_max=1.0,
            entropy_coef=0.01,
        )

        reward = (-log_prob).mean()
        risk_excess = torch.relu(g - 0.4) / 0.6
        risk = 0.6 * 0.5 * risk_excess.mean()
        entropy_bonus = 0.01 * entropy.mean()
        expected = reward + risk - entropy_bonus

        self.assertTrue(torch.allclose(total, expected))
        self.assertTrue(torch.allclose(metrics["reward_loss_total"], reward))
        self.assertTrue(torch.allclose(metrics["risk_loss_total"], risk))
        self.assertAlmostEqual(float(metrics["gp_lambda_eff"]), 0.3, places=6)
        self.assertAlmostEqual(float(metrics["gp_over_pf_ratio"]), 2.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
