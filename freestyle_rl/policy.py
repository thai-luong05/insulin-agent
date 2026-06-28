"""TinyPolicy v2: insulin rate = basal + delta(state).

The network outputs a DELTA from basal via tanh — so a zero-init network
produces basal_rate insulin, matching fasting behaviour out of the box.

This fixes the BC failure where sigmoid(0)=0.5 → action=0.3 U/min caused
catastrophic over-dosing on patients whose basal is ~0.01 U/min.

State (10-dim, same as v1).
Action: insulin rate in [0, 0.6] U/min.
"""
import torch
import torch.nn as nn

STATE_DIM  = 10
ACTION_DIM = 1
ACTION_MAX = 0.6        # U/min, simglucose pump cap
DELTA_RANGE = 0.3       # max |delta| from basal in U/min (covers any meal bolus)


class TinyPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden, ACTION_DIM)
        #zero-init the action head so the policy starts at basal
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        #stochastic exploration for PPO (very small to start)
        self.log_std    = nn.Parameter(torch.full((ACTION_DIM,), -3.0))
        self.value_head = nn.Linear(hidden, 1)

    def _basal_from_state(self, state):
        #state[...,0] is basal_rate / 0.05
        return state[..., 0] * 0.05

    def forward(self, state):
        """Deterministic action in [0, ACTION_MAX], centred at basal."""
        h = self.shared(state)
        raw = self.action_head(h).squeeze(-1)
        delta = torch.tanh(raw) * DELTA_RANGE
        basal = self._basal_from_state(state)
        return torch.clamp(basal + delta, 0.0, ACTION_MAX)

    def act_stochastic(self, state):
        """For PPO. Sample raw, log_prob, value."""
        h = self.shared(state)
        raw_mu = self.action_head(h).squeeze(-1)
        std    = self.log_std.exp().clamp(0.02, 0.5)
        dist   = torch.distributions.Normal(raw_mu, std)
        raw    = dist.rsample()
        log_p  = dist.log_prob(raw)
        delta  = torch.tanh(raw) * DELTA_RANGE
        basal  = self._basal_from_state(state)
        action = torch.clamp(basal + delta, 0.0, ACTION_MAX)
        value  = self.value_head(h).squeeze(-1)
        return action, log_p, value, raw

    def evaluate(self, state, raw):
        h = self.shared(state)
        raw_mu = self.action_head(h).squeeze(-1)
        std    = self.log_std.exp().clamp(0.02, 0.5)
        dist   = torch.distributions.Normal(raw_mu, std)
        log_p  = dist.log_prob(raw)
        value  = self.value_head(h).squeeze(-1)
        entropy = dist.entropy()
        return log_p, value, entropy
