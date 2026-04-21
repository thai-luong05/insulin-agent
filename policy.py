import torch
import torch.nn as nn
from others import LSTMEncoder, NormedLinear


class Policy(nn.Module):
    """Continuous action policy: outputs Normal(mu, sigma) over action in [-1, 1].
    Pump rate = action_scale * exp((action - 1) * 4)  (handled in Agent)."""
    def __init__(self, n_features, n_hidden=16, n_layers=1, init_mu_bias=0.0):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, n_hidden, n_layers)
        d = self.encoder.output_dim
        self.net = nn.Sequential(
            nn.Linear(d, d * 2), nn.ReLU(),
            nn.Linear(d * 2, d * 2), nn.ReLU(),
            nn.Linear(d * 2, d * 2), nn.ReLU(),
        )
        self.mu_head    = NormedLinear(d * 2, 1, scale=0.1)
        self.sigma_head = NormedLinear(d * 2, 1, scale=0.1)
        # Init pre-tanh bias so initial pump ≈ basal (caller computes from basal/action_scale).
        # Default 0 reproduces vanilla init; main.py overrides per patient.
        self.mu_head.bias.data.fill_(init_mu_bias)

    def forward(self, x):
        h = self.net(self.encoder(x))
        mu    = torch.tanh(self.mu_head(h)).squeeze(-1)
        sigma = torch.sigmoid(self.sigma_head(h)).squeeze(-1) + 1e-3
        return mu, sigma

    def get_dist(self, x):
        mu, sigma = self.forward(x)
        return torch.distributions.Normal(mu, sigma)
