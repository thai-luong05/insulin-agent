import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from policy_sac import SACPolicy
from others import MLPEncoder


class QNetwork(nn.Module):
    """Q(s, a): encoded state concatenated with scalar action -> scalar Q."""
    def __init__(self, n_features, n_hidden=64, n_layers=1):
        super().__init__()
        self.encoder = MLPEncoder(n_features, n_hidden, n_layers)
        d = self.encoder.output_dim
        self.net = nn.Sequential(
            nn.Linear(d + 1, d * 2), nn.ReLU(),
            nn.Linear(d * 2, d * 2), nn.ReLU(),
            nn.Linear(d * 2, 1),
        )

    def forward(self, x, a):
        e = self.encoder(x)
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        return self.net(torch.cat([e, a], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity, state_shape, device):
        self.capacity = capacity
        self.device   = device
        self.s   = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.a   = np.zeros((capacity,), dtype=np.float32)
        self.r   = np.zeros((capacity,), dtype=np.float32)
        self.s2  = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.d   = np.zeros((capacity,), dtype=np.float32)
        self.ptr  = 0
        self.size = 0

    def push(self, s, a, r, s2, done):
        i = self.ptr
        self.s[i]  = s
        self.a[i]  = a
        self.r[i]  = r
        self.s2[i] = s2
        self.d[i]  = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.s[idx]).to(self.device),
            torch.from_numpy(self.a[idx]).to(self.device),
            torch.from_numpy(self.r[idx]).to(self.device),
            torch.from_numpy(self.s2[idx]).to(self.device),
            torch.from_numpy(self.d[idx]).to(self.device),
        )


class SACAgent:
    def __init__(self, n_features, state_shape,
                 action_scale=1.0, device='cpu',
                 n_hidden=64, n_layers=1,
                 gamma=0.99, tau=0.005,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 buffer_size=100000, batch_size=256,
                 target_entropy=-1.0, init_mu_bias=0.0,
                 warmup_steps=5000, grad_clip=10.0):

        self.device       = torch.device(device)
        self.action_scale = action_scale
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.warmup_steps = warmup_steps
        self.grad_clip    = grad_clip
        self.steps        = 0

        self.policy = SACPolicy(n_features, n_hidden, n_layers, init_mu_bias).to(self.device)
        self.q1     = QNetwork(n_features, n_hidden, n_layers).to(self.device)
        self.q2     = QNetwork(n_features, n_hidden, n_layers).to(self.device)
        self.q1_t   = QNetwork(n_features, n_hidden, n_layers).to(self.device)
        self.q2_t   = QNetwork(n_features, n_hidden, n_layers).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        for p in self.q1_t.parameters(): p.requires_grad = False
        for p in self.q2_t.parameters(): p.requires_grad = False

        self.opt_pi = optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.opt_q1 = optim.Adam(self.q1.parameters(),     lr=critic_lr)
        self.opt_q2 = optim.Adam(self.q2.parameters(),     lr=critic_lr)

        #Auto-tuned entropy temperature: gradient drives α so E[-log_prob] -> target_entropy.
        self.target_entropy = target_entropy
        self.log_alpha      = torch.zeros(1, device=self.device, requires_grad=True)
        self.opt_alpha      = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.buffer = ReplayBuffer(buffer_size, state_shape, self.device)

    def _action_to_pump(self, action):
        #Same exponential mapping as the PPO Agent for fair comparison.
        pump = self.action_scale * math.exp((action - 1) * 4)
        return min(self.action_scale, max(0.0, pump))

    def select_action(self, state_matrix, std_basal, deterministic=False):
        x = torch.tensor(state_matrix, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                a = self.policy.deterministic(x)
            elif self.steps < self.warmup_steps:
                a = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)
            else:
                a, _ = self.policy.sample(x)
        action = float(a.item())
        action = max(-1.0, min(1.0, action))
        return self._action_to_pump(action), None, action

    def push(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)
        self.steps += 1

    def update(self):
        """One SAC gradient step. Returns stats dict, or None during warmup."""
        if self.buffer.size < self.batch_size or self.steps < self.warmup_steps:
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            a2, logp2 = self.policy.sample(s2)
            q_t = torch.min(self.q1_t(s2, a2), self.q2_t(s2, a2))
            target = r + self.gamma * (1 - d) * (q_t - alpha * logp2)

        loss_q1 = F.mse_loss(self.q1(s, a), target)
        loss_q2 = F.mse_loss(self.q2(s, a), target)

        self.opt_q1.zero_grad(); loss_q1.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.opt_q1.step()

        self.opt_q2.zero_grad(); loss_q2.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
        self.opt_q2.step()

        a_new, logp_new = self.policy.sample(s)
        q_min = torch.min(self.q1(s, a_new), self.q2(s, a_new))
        loss_pi = (alpha * logp_new - q_min).mean()

        self.opt_pi.zero_grad(); loss_pi.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.opt_pi.step()

        loss_alpha = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(); loss_alpha.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            'q1_loss': float(loss_q1.item()),
            'q2_loss': float(loss_q2.item()),
            'pi_loss': float(loss_pi.item()),
            'alpha':   float(alpha.item()),
            'entropy': float(-logp_new.mean().item()),
        }
