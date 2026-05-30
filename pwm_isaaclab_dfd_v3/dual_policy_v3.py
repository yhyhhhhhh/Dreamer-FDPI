from __future__ import annotations

import torch

from pwm_isaaclab_dfd_v2.dual_policy_v2 import DualPolicyV2


class DualPolicyV3(DualPolicyV2):
    """DFD v3 dual actor.

    The network and distribution interface intentionally stay compatible with
    DFD v2 so the shared Gd/replay/trainer code can treat it as a drop-in
    dual policy. V3 keeps the KL reference fixed to the current main actor in
    the imagination update.
    """

    kl_reference = "current_main"

    @torch.no_grad()
    def initialize_from_main_actor(self, main_agent):
        self.actor.load_state_dict(main_agent.actor.state_dict())

    def rsample_with_log_prob(self, feat):
        action, log_prob = self.rsample(feat, return_log_prob=True)
        return action, log_prob

