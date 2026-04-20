import torch.nn as nn


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


class Value_function(nn.Module):
    def __init__(self, n_features, n_hidden=16, n_layers=1):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, n_hidden, n_layers)
        d = self.encoder.output_dim
        self.net = nn.Sequential(
            nn.Linear(d, d * 2), nn.ReLU(),
            nn.Linear(d * 2, d * 2), nn.ReLU(),
            nn.Linear(d * 2, d * 2), nn.ReLU(),
        )
        self.head = NormedLinear(d * 2, 1, scale=0.1)

    def forward(self, x):
        return self.head(self.net(self.encoder(x)))
