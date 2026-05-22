import math
import torch
import torch.nn as nn
from others import MLPEncoder


LOG_SIG_MIN = -5.0
LOG_SIG_MAX = 2.0


class SACPolicy(nn.Module):
    """Squashed-Gaussian actor for SAC.
    Outputs (mu, log_sigma). For training use sample(): reparameterized
    Gaussian → tanh squash → log_prob with tanh Jacobian correction.
    For eval use deterministic(): just tanh(mu)."""

    def __init__(self, n_features, n_hidden=64, n_layers=1, init_mu_bias=0.0):
        super().__init__()
        self.encoder = MLPEncoder(n_features, n_hidden, n_layers)
        d = self.encoder.output_dim
        self.net = nn.Sequential(
            nn.Linear(d, d * 2), nn.ReLU(),
            nn.Linear(d * 2, d * 2), nn.ReLU(),
        )
        self.mu_head      = nn.Linear(d * 2, 1)
        self.log_sig_head = nn.Linear(d * 2, 1)
        #Pre-bias so initial deterministic action ≈ basal-equivalent.
        #Caller computes init_mu_bias from basal/action_scale.
        self.mu_head.bias.data.fill_(init_mu_bias)

    def forward(self, x):
        h = self.net(self.encoder(x))
        mu      = self.mu_head(h).squeeze(-1)
        log_sig = self.log_sig_head(h).squeeze(-1).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return mu, log_sig

    def sample(self, x):
        #Returns (squashed_action in [-1, 1], log_prob).
        #Reparameterized so gradients flow back through actions
        mu, log_sig = self.forward(x)
        sigma = log_sig.exp()
        eps = torch.randn_like(mu)
        raw = mu + sigma * eps
        action = torch.tanh(raw)
        #Gaussian log-prob of raw, then subtract log(1 - tanh(raw)^2) for squash.
        log_prob = -0.5 * (eps ** 2 + 2 * log_sig + math.log(2 * math.pi))
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob

    def deterministic(self, x):
        #Squashed mean — for evaluation rollouts
        mu, _ = self.forward(x)
        return torch.tanh(mu)
