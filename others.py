import torch
import torch.nn as nn
import torch.nn.functional as F


def NormedLinear(in_features, out_features, scale=0.1):
    layer = nn.Linear(in_features, out_features)
    layer.weight.data *= scale / layer.weight.norm(dim=1, p=2, keepdim=True)
    return layer


class LSTMEncoder(nn.Module):
    def __init__(self, n_features, n_hidden=16, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, n_hidden, n_layers, batch_first=True)
        self.output_dim = n_hidden * n_layers

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        _, (hid, _) = self.lstm(x)
        return hid.permute(1, 0, 2).contiguous().view(x.size(0), -1)


class MLPEncoder(nn.Module):
    """Drop-in replacement for LSTMEncoder. Uses the last timestep's features only."""
    def __init__(self, n_features, n_hidden=16, n_layers=1):
        super().__init__()
        self.output_dim = n_hidden * n_layers
        self.net = nn.Sequential(
            nn.Linear(n_features, self.output_dim), nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim), nn.ReLU(),
        )

    def forward(self, x):
        # x: [batch, seq_len, n_features] -> use last timestep
        return self.net(x[:, -1, :])


class Value_function(nn.Module):
    def __init__(self, n_features, n_hidden=16, n_layers=1):
        super().__init__()
        self.encoder = MLPEncoder(n_features, n_hidden, n_layers)
        d = self.encoder.output_dim
        self.net = nn.Sequential(
            nn.Linear(d, d * 2), nn.ReLU(),
            nn.Linear(d * 2, d * 2), nn.ReLU(),
            nn.Linear(d * 2, d * 2), nn.ReLU(),
        )
        self.head = NormedLinear(d * 2, 1, scale=0.1)

        self.cgm_trunk = nn.Sequential(
            nn.Linear(d + 1, d), nn.ReLU(),
        )
        self.cgm_mu_head    = NormedLinear(d, 1, scale=0.1)
        self.cgm_sigma_head = NormedLinear(d, 1, scale=0.1)

    def forward(self, x):
        return self.head(self.net(self.encoder(x)))

    def predict_cgm(self, x, action):
        e = self.encoder(x)
        a = action.detach()
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        h = self.cgm_trunk(torch.cat([e, a], dim=-1))
        cgm_mu    = torch.tanh(self.cgm_mu_head(h)).squeeze(-1)
        cgm_sigma = F.softplus(self.cgm_sigma_head(h)).squeeze(-1) + 1e-3
        return cgm_mu, cgm_sigma
