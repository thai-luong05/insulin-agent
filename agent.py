import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from policy import Policy
from others import Value_function


class RewardNormalizer:
    """Divides rewards by running std of discounted returns. Clips to +/-cliprew.
    Resets accumulated return at the start of each episode."""
    def __init__(self, gamma=0.99, cliprew=10.0, epsilon=1e-8):
        self.gamma, self.cliprew = gamma, cliprew
        self.ret = 0.0
        self.count = epsilon
        self.mean = 0.0
        self.M2 = 0.0

    def __call__(self, rewards):
        self.ret = 0.0
        for r in rewards:
            self.ret = self.gamma * self.ret + float(r.item())
            self.count += 1.0
            delta = self.ret - self.mean
            self.mean += delta / self.count
            self.M2 += delta * (self.ret - self.mean)
        std = (self.M2 / max(self.count - 1.0, 1.0)) ** 0.5 + 1e-8
        return torch.clamp(rewards / std, -self.cliprew, self.cliprew)


class Agent:
    def __init__(self, n_features, action_scale=1.0, device='cpu',
                 n_hidden=16, n_layers=1,
                 pi_lr=3e-4, vf_lr=3e-4,
                 gamma=0.99, lambda_=0.95,
                 entropy_coef=0.01, grad_clip=20,
                 eps_clip=0.2, n_epochs=5,
                 target_kl=0.01, normalize_reward=True,
                 init_mu_bias=0.0):

        self.device       = torch.device(device)
        self.action_scale = action_scale
        self.gamma        = gamma
        self.lambda_      = lambda_
        self.entropy_coef = entropy_coef
        self.grad_clip    = grad_clip
        self.eps_clip     = eps_clip
        self.n_epochs     = n_epochs
        self.target_kl    = target_kl
        self.normalize_reward  = normalize_reward
        self.reward_normalizer = RewardNormalizer(gamma=gamma) if normalize_reward else None

        self.policy = Policy(n_features, n_hidden, n_layers,
                             init_mu_bias=init_mu_bias).to(self.device)
        self.value  = Value_function(n_features, n_hidden, n_layers).to(self.device)

        self.opt_pi     = optim.Adam(self.policy.parameters(), lr=pi_lr)
        self.opt_vf     = optim.Adam(self.value.parameters(), lr=vf_lr)
        self.vf_loss_fn = nn.MSELoss()

    def _action_to_pump(self, action):
        """Exponential mapping: action in [-1, 1] -> pump in (0, action_scale]."""
        pump = self.action_scale * math.exp((action - 1) * 4)
        return min(self.action_scale, max(0.0, pump))

    def select_action(self, state_matrix, std_basal, deterministic=False):
        """Returns (pump_act U/min, log_prob tensor or None, raw action in [-1, 1])."""
        x = torch.tensor(state_matrix, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.policy.get_dist(x)

        if deterministic:
            action = float(dist.mean.item())
            action = max(-1.0, min(1.0, action))
            return self._action_to_pump(action), None, action

        raw    = dist.sample()
        logp   = dist.log_prob(raw)
        action = float(raw.item())
        action = max(-1.0, min(1.0, action))
        return self._action_to_pump(action), logp, action

    def _gae(self, rewards_t, values_t, last_val=0.0):
        T   = len(rewards_t)
        adv = torch.zeros(T, device=self.device)
        lam = 0.0
        for t in reversed(range(T)):
            next_v = values_t[t + 1] if t + 1 < T else last_val
            delta  = rewards_t[t] + self.gamma * next_v - values_t[t]
            lam    = delta + self.gamma * self.lambda_ * lam
            adv[t] = lam
        return adv, adv + values_t

    def update(self, episodes):
        all_states, all_actions, all_logp, all_adv, all_vtarg = [], [], [], [], []

        with torch.no_grad():
            for ep in episodes:
                states_t = torch.tensor(np.array(ep['states']),
                                        dtype=torch.float32, device=self.device)
                final_t  = torch.tensor(ep['final_state'],
                                        dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                rewards_t = torch.FloatTensor(ep['rewards']).to(self.device)
                if self.normalize_reward:
                    rewards_t = self.reward_normalizer(rewards_t)

                vals     = self.value(states_t).squeeze(-1)
                last_val = self.value(final_t).squeeze(-1).item()
                adv, vtarg = self._gae(rewards_t, vals, last_val=last_val)

                all_states.append(states_t)
                all_actions.append(torch.FloatTensor(ep['actions']).to(self.device))
                all_logp.append(torch.stack(ep['log_probs']).to(self.device).detach().squeeze(-1))
                all_adv.append(adv)
                all_vtarg.append(vtarg)

        states_t      = torch.cat(all_states, dim=0)
        actions_t     = torch.cat(all_actions, dim=0)
        log_probs_old = torch.cat(all_logp, dim=0)
        adv           = torch.cat(all_adv, dim=0)
        vtarg         = torch.cat(all_vtarg, dim=0)
        norm_adv      = (adv - adv.mean()) / (adv.std() + 1e-5)

        for _ in range(self.n_epochs):
            dist     = self.policy.get_dist(states_t)
            logp_new = dist.log_prob(actions_t)
            entropy  = dist.entropy().mean()

            log_ratio = logp_new - log_probs_old
            ratio     = torch.exp(log_ratio)

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                break

            surr1    = ratio * norm_adv
            surr2    = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * norm_adv
            pi_loss  = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.opt_pi.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            self.opt_pi.step()

            vf_loss = self.vf_loss_fn(self.value(states_t).squeeze(-1), vtarg.detach())
            self.opt_vf.zero_grad()
            vf_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_clip)
            self.opt_vf.step()
