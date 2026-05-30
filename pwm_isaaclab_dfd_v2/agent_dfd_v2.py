from __future__ import annotations

try:
    from pwm_isaaclab.agents import ActorCriticAgent
except ImportError:
    from agents import ActorCriticAgent

from .cost_utils import disable_optimizer_dynamo_wrappers, ensure_optimizer_step_no_grad, unwrap_optimizer_step


class CostAwareActorCriticAgent(ActorCriticAgent):
    """Original Dreamer actor-critic; v2 only changes the reward tensor passed in."""

    def __init__(self, *args, **kwargs):
        disable_optimizer_dynamo_wrappers()
        super().__init__(*args, **kwargs)
        unwrap_optimizer_step(self.optimizer)
        ensure_optimizer_step_no_grad(self.optimizer)
