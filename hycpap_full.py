"""HyCPAP (Wu 2024): recurrent SAC + masksembles blended with a zone-MPC prior. --patient_id N: meta warm-start + ESS fine-tune; --general: SAC from scratch; --meta: train the shared prior."""
import os
import sys
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'simulation', 'G2P2C'))

OBS_DIM   = 4          # [BG, bolus, carbs, IOB]  (paper observation)
WINDOW    = 8          # recurrent history length
ACT_MAX_X = 10.0       # action upper bound = 10 x basal (paper)
PUMP_MAX  = 0.6        # physical pump cap (U/min)
LOG_SIG_MIN, LOG_SIG_MAX = -5.0, 2.0
REWARD_SCALE = 1.0 / 50.0   # scale Eq.1 into a sane range for SAC (shape unchanged)


# paper reward (Eq.1), extended monotonically past [54,300] with a floor (flat caps let the agent farm hypo)
def paper_reward(g):
    if g < 100.0:       r = 1.0 - 0.201 * (100.0 - g) ** 1.622
    elif g <= 140.0:    r = 1.0
    else:               r = 1.0 - 0.473 * (g - 140.0) ** 0.918
    return max(r, -100.0)


def obs_vector(bg, bolus, carbs, iob):
    return np.array([bg / 200.0, bolus / 5.0, carbs / 50.0, min(iob, 30.0) / 30.0],
                    dtype=np.float32)


# masksembles (Durasov 2021): overlap-controlled mask generation
def generate_masks(num_masks, channels, scale):
    """Overlap-controlled binary masks (larger scale -> more diverse members)."""
    active = max(1, int(round(channels / (1.0 + (num_masks - 1) / scale))))
    total  = int(round(active * (1.0 + (num_masks - 1) / scale)))
    total  = max(total, channels)
    rng = np.random.default_rng(0)
    masks = np.zeros((num_masks, total), dtype=np.float32)
    for i in range(num_masks):
        ids = rng.choice(total, active, replace=False)
        masks[i, ids] = 1.0
    masks = masks[:, ~np.all(masks == 0, axis=0)]      # drop dead columns
    if masks.shape[1] >= channels:
        masks = masks[:, :channels]
    else:                                              # pad if short
        pad = np.ones((num_masks, channels - masks.shape[1]), dtype=np.float32)
        masks = np.concatenate([masks, pad], axis=1)
    keep = masks.mean()
    return masks / max(keep, 1e-3)


class Masksembles1D(nn.Module):
    def __init__(self, channels, num_masks=4, scale=4.0):
        super().__init__()
        self.register_buffer('masks', torch.from_numpy(
            generate_masks(num_masks, channels, scale)))
        self.num_masks = num_masks

    def forward(self, x, idx=None):
        if idx is None:
            idx = int(torch.randint(self.num_masks, (1,)).item())
        return x * self.masks[idx]


# gru encoder over the (obs, prev_action, prev_reward) window
class GRUEncoder(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        # per-step input: obs(4) + prev_action(1) + prev_reward(1)
        self.gru = nn.GRU(OBS_DIM + 2, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, win):
        # win: [B, WINDOW, OBS_DIM+2]
        _, h = self.gru(win)
        return h.squeeze(0)          # [B, hidden]


class RecurrentActor(nn.Module):
    def __init__(self, hidden=128, num_masks=4):
        super().__init__()
        self.enc  = GRUEncoder(hidden)
        self.mask = Masksembles1D(hidden, num_masks)
        self.l    = nn.Linear(hidden, hidden)
        self.mu_head      = nn.Linear(hidden, 1)
        self.log_sig_head = nn.Linear(hidden, 1)
        self.num_masks = num_masks

    def forward(self, win, mask_idx=None):
        h = self.enc(win)
        h = self.mask(h, mask_idx)
        h = torch.relu(self.l(h))
        mu      = self.mu_head(h).squeeze(-1)
        log_sig = self.log_sig_head(h).squeeze(-1).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return mu, log_sig

    def sample(self, win, mask_idx=None):
        mu, log_sig = self.forward(win, mask_idx)
        sigma = log_sig.exp()
        eps = torch.randn_like(mu)
        raw = mu + sigma * eps
        a = torch.tanh(raw)                       # in [-1,1]
        logp = -0.5 * (eps ** 2 + 2 * log_sig + math.log(2 * math.pi))
        logp = logp - torch.log(1 - a.pow(2) + 1e-6)
        return a, logp

    @staticmethod
    def dose(a, basal):
        # a in [-1,1] -> multiplier in [0, ACT_MAX_X] x basal, clamped to pump cap
        m = 0.5 * (a + 1.0) * ACT_MAX_X
        return torch.clamp(basal * m, 0.0, PUMP_MAX)

    def act_stochastic(self, win, basal):
        a, logp = self.sample(win)
        return self.dose(a, basal), logp, torch.zeros_like(logp), a


class RecurrentCritic(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        def q():
            return nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.enc = GRUEncoder(hidden)
        self.q1, self.q2 = q(), q()

    def forward(self, win, a):
        h = self.enc(win)
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        x = torch.cat([h, a], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class WindowReplay:
    def __init__(self, capacity=200000):
        self.cap = capacity
        self.w   = np.zeros((capacity, WINDOW, OBS_DIM + 2), dtype=np.float32)
        self.a   = np.zeros((capacity,), dtype=np.float32)
        self.r   = np.zeros((capacity,), dtype=np.float32)
        self.w2  = np.zeros((capacity, WINDOW, OBS_DIM + 2), dtype=np.float32)
        self.d   = np.zeros((capacity,), dtype=np.float32)
        self.basal = np.zeros((capacity,), dtype=np.float32)
        self.ptr = self.size = 0

    def push(self, w, a, r, w2, done, basal):
        i = self.ptr
        self.w[i], self.a[i], self.r[i], self.w2[i], self.d[i], self.basal[i] = \
            w, a, r, w2, float(done), basal
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        idx = np.random.randint(0, self.size, size=n)
        t = lambda x: torch.from_numpy(x[idx])
        return t(self.w), t(self.a), t(self.r), t(self.w2), t(self.d), t(self.basal)


class RecurrentSAC:
    def __init__(self, hidden=128, gamma=0.99, tau=0.005, lr=3e-4,
                 batch=256, warmup=2000, target_entropy=-1.0, num_masks=4):
        self.gamma, self.tau, self.batch, self.warmup = gamma, tau, batch, warmup
        self.actor    = RecurrentActor(hidden, num_masks)
        self.critic   = RecurrentCritic(hidden)
        self.critic_t = RecurrentCritic(hidden)
        self.critic_t.load_state_dict(self.critic.state_dict())
        for p in self.critic_t.parameters():
            p.requires_grad = False
        self.opt_a = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=lr)
        self.buffer = WindowReplay()
        self.steps = 0
        self.meta = None        # (arrays, resample_probs, ess) for ESS reuse

    def attach_meta(self, arrays, beta, ess):
        """ESS-reweighted reuse: resample meta txns by beta; ess = meta batch fraction."""
        self.meta = (arrays, beta / (beta.sum() + 1e-12), float(np.clip(ess, 0.0, 0.9)))

    def _sample_batch(self):
        if self.meta is None:
            return self.buffer.sample(self.batch)
        arrays, probs, ess = self.meta
        n_meta = int(self.batch * ess)
        n_new  = self.batch - n_meta
        wn, an, rn, w2n, dn, bn = self.buffer.sample(n_new)
        mw, ma, mr, mw2, md, mb = arrays
        mi = np.random.choice(len(ma), size=n_meta, p=probs)
        cat = lambda new, m: torch.cat([new, torch.from_numpy(m[mi])], 0)
        return (cat(wn, mw), cat(an, ma), cat(rn, mr),
                cat(w2n, mw2), cat(dn, md), cat(bn, mb))

    def act(self, win_np, basal, deterministic=False):
        win = torch.from_numpy(win_np).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(win); a = torch.tanh(mu)
            elif self.steps < self.warmup:
                a = torch.empty(1).uniform_(-1.0, 1.0)
            else:
                a, _ = self.actor.sample(win)
        a_val = float(a.item())
        dose = float(self.actor.dose(torch.tensor(a_val), torch.tensor(basal)).item())
        return dose, a_val

    def update(self):
        if self.buffer.size < self.batch or self.steps < self.warmup:
            return None
        w, a, r, w2, d, basal = self._sample_batch()
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor.sample(w2)
            q1t, q2t = self.critic_t(w2, a2)
            target = r + self.gamma * (1 - d) * (torch.min(q1t, q2t) - alpha * logp2)
        q1, q2 = self.critic(w, a)
        loss_c = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_c.zero_grad(); loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0); self.opt_c.step()

        an, logpn = self.actor.sample(w)
        q1n, q2n = self.critic(w, an)
        loss_a = (alpha * logpn - torch.min(q1n, q2n)).mean()
        self.opt_a.zero_grad(); loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0); self.opt_a.step()

        loss_al = -(self.log_alpha * (logpn + self.target_entropy).detach()).mean()
        self.opt_al.zero_grad(); loss_al.backward(); self.opt_al.step()
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return float(loss_c.item())


# env rollout: paper obs/action/reward + recurrent window
def run_episode(agent, patient_id, seed, train=True, days=6):
    """one continuous multi-day rollout; meal announcement read from the env's daily scenario by time-of-day"""
    from main import (PREBOLUS_STEPS, get_clinical_params, get_body_weight)
    from freestyle_rl.ppo_finetune import make_env, calibrate
    from utils.statespace import StateSpace
    from utils.pumpAction import Pump, get_basal

    env, args, name = make_env(patient_id, seed=seed)
    basal = get_basal(name)
    clin  = get_clinical_params(name)
    pump = Pump(args, patient_name=name); ss = StateSpace(args)
    _, cgm = calibrate(env, ss, args, pump)

    def now_minute(info=None):
        """minute-of-day from the env clock (0-1440)"""
        tt = info.get('time') if info is not None else None
        if tt is None:
            try:    tt = env.env.time          # sim datetime before the first step
            except Exception: return 0.0
        return tt.hour * 60 + tt.minute

    # meal carbs within the prebolus window, matched by minute-of-day
    def upcoming_carbs(now_min):
        try:
            sched = env.env.scenario.scenario.get('meal', {'time': [], 'amount': []})
        except Exception:
            return 0.0
        for t, a in zip(sched['time'], sched['amount']):
            if 0.0 <= (float(t) - now_min) <= PREBOLUS_STEPS * 5:
                return float(a)
        return 0.0

    iob = 0.0
    hist = [np.zeros(OBS_DIM + 2, dtype=np.float32) for _ in range(WINDOW)]

    def push_hist(bg, bolus, carbs, iobv, prev_a, prev_r):
        o = obs_vector(bg, bolus, carbs, iobv)
        hist.append(np.concatenate([o, [prev_a, prev_r]]).astype(np.float32))
        if len(hist) > WINDOW:
            hist.pop(0)
        return np.stack(hist)

    prev_a, prev_r = 0.0, 0.0
    c0 = upcoming_carbs(now_minute()); b0 = c0 / clin['icr'] if c0 > 0 else 0.0
    win = push_hist(cgm, b0, c0, iob, prev_a, prev_r)
    cgms = []; ep_r = 0.0
    total = 288 * int(days)
    for step in range(total):
        dose, a = agent.act(win, basal, deterministic=not train)
        s, _, done_env, info = env.step(dose)
        ng = s.CGM
        iob = 0.93 * iob + dose * 5.0
        r = paper_reward(ng)            # + terminal penalties, then uniform scale
        terminal = False
        if ng < 39.0:
            r += -20.0; terminal = True
        elif ng > 400.0:
            r += -10.0; terminal = True
        r *= REWARD_SCALE; ep_r += r
        c = upcoming_carbs(now_minute(info)); b = c / clin['icr'] if c > 0 else 0.0
        win2 = push_hist(ng, b, c, iob, a, r)
        done = terminal or (step == total - 1)
        if train:
            agent.buffer.push(win, a, r, win2, done, basal)
            agent.steps += 1
            agent.update()
        win = win2; cgms.append(ng)
        if terminal:
            break
    cgm_arr = np.array(cgms)
    return {'tir':  float(np.mean((70 <= cgm_arr) & (cgm_arr <= 180)) * 100),
            'hypo': float(np.mean(cgm_arr < 70) * 100),
            'hyper':float(np.mean(cgm_arr > 180) * 100),
            'reward': ep_r, 'steps': len(cgms)}


def plot_training_reward(rewards, out_path, title):
    """save reward-over-training curve (per-episode return + rolling mean)"""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = np.asarray(rewards, dtype=float)
    if len(r) == 0:
        return
    ep = np.arange(1, len(r) + 1)
    k = max(1, len(r) // 20)
    roll = np.convolve(r, np.ones(k) / k, mode='valid')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ep, r, color='0.75', lw=0.8, label='episode return')
    ax.plot(ep[k - 1:], roll, color='C0', lw=2, label=f'rolling mean ({k})')
    ax.set_xlabel('episode'); ax.set_ylabel('episode return (scaled reward)')
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'reward curve saved -> {out_path}', flush=True)


def train_one(patient_id, episodes=20, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agent = RecurrentSAC()
    best = -1.0; rewards = []
    for ep in range(episodes):
        m = run_episode(agent, patient_id, seed=seed + ep, train=True)
        best = max(best, m['tir']); rewards.append(m['reward'])
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'  ep {ep+1:3d}/{episodes}  tir={m["tir"]:5.1f}%  hypo={m["hypo"]:4.1f}%  '
                  f'hyper={m["hyper"]:4.1f}%  steps={m["steps"]}  buf={agent.buffer.size}  best={best:.1f}%',
                  flush=True)
    plot_training_reward(rewards, f'freestyle_rl/hycpap_train_p{patient_id}_reward.png',
                         f'HyCPAP from-scratch training reward (p{patient_id})')
    return agent


def gaussian_product(mu1, s1, mu2, s2):
    v1, v2 = s1 * s1, s2 * s2
    den = v1 + v2 + 1e-12
    return (mu1 * v2 + mu2 * v1) / den, (v1 * v2 / den) ** 0.5


def ensemble_gaussian(mus, sigs):
    """M members -> one gaussian (law of total variance); between-member var = disagreement, blend leans on MPC when they differ"""
    mus  = np.asarray(mus, dtype=float)
    sigs = np.asarray(sigs, dtype=float)
    mu   = float(mus.mean())
    var  = float((sigs ** 2).mean() + mus.var())
    return mu, var ** 0.5


def ess_weights(meta_states, new_states, iters=200, lr=0.3, l2=0.1):
    """importance reuse: logistic propensity -> beta=p/(1-p) reweights meta txns, ESS (Eq.11)=usable fraction; balanced+L2 so it can't separate perfectly (else beta~0)"""
    meta_states = np.asarray(meta_states, dtype=np.float64)
    new_states  = np.asarray(new_states,  dtype=np.float64)
    rng = np.random.default_rng(0)
    n   = min(len(meta_states), len(new_states))
    Xtr = np.vstack([meta_states[rng.choice(len(meta_states), n, replace=False)],
                     new_states [rng.choice(len(new_states),  n, replace=False)]])
    ytr = np.concatenate([np.zeros(n), np.ones(n)])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    feat = lambda S: np.hstack([(S - mu) / sd, np.ones((len(S), 1))])
    Xs = feat(Xtr); w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w)))
        grad = Xs.T @ (p - ytr) / len(ytr) + l2 * w
        w -= lr * grad
    pm = (1.0 / (1.0 + np.exp(-(feat(meta_states) @ w)))).clip(1e-3, 1 - 1e-3)
    beta = pm / (1.0 - pm)
    ess = (beta.sum() ** 2) / (np.sum(beta ** 2) + 1e-12) / len(beta)   # Eq. 11
    return beta, float(ess)


META_PREFIX = 'freestyle_rl/meta_full'


def meta_pretrain_full(episodes=800, seed=0, pool=None):
    if pool is None:
        pool = list(range(30))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agent = RecurrentSAC()
    rng = random.Random(seed)
    print(f'META (recurrent SAC) pretrain: {episodes} eps over {len(pool)} patients', flush=True)
    window = []; rewards = []
    for ep in range(episodes):
        pid = rng.choice(pool)
        m = run_episode(agent, pid, seed=seed + ep, train=True, days=1)  # 1-day eps for cross-patient variety
        window.append(m['tir']); window = window[-30:]; rewards.append(m['reward'])
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f'  ep {ep+1:4d}/{episodes}  p{pid:<2}  tir={m["tir"]:5.1f}%  '
                  f'avg30={np.mean(window):5.1f}%  buf={agent.buffer.size}', flush=True)
    torch.save({'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict()}, META_PREFIX + '.pt')
    print(f'meta saved -> {META_PREFIX}.pt', flush=True)
    plot_training_reward(rewards, META_PREFIX + '_reward.png', 'HyCPAP meta-pretrain reward')
    # buffer subset for ESS reuse (stage 2)
    n = min(agent.buffer.size, 20000)
    idx = np.random.choice(agent.buffer.size, n, replace=False)
    np.savez_compressed(META_PREFIX + '_buf.npz',
                        w=agent.buffer.w[idx], a=agent.buffer.a[idx], r=agent.buffer.r[idx],
                        w2=agent.buffer.w2[idx], d=agent.buffer.d[idx], basal=agent.buffer.basal[idx])
    print(f'meta buffer ({n}) saved -> {META_PREFIX}_buf.npz', flush=True)


def load_actor(path):
    ck = torch.load(path, map_location='cpu')
    a = RecurrentActor()
    a.load_state_dict(ck['actor'] if 'actor' in ck else ck)
    a.eval()
    return a


def collect_meta_buffer(meta_path=META_PREFIX + '.pt', n_target=20000, seed=0):
    """roll the trained meta policy across patients (no learning) to rebuild the ESS buffer without a full retrain"""
    agent = RecurrentSAC()
    agent.actor.load_state_dict(torch.load(meta_path, map_location='cpu')['actor'])
    agent.warmup = 0
    agent.update = lambda: None          # collect only
    print(f'collecting meta buffer from {meta_path} ...', flush=True)
    ep = 0
    while agent.buffer.size < n_target:
        run_episode(agent, ep % 30, seed=seed + ep, train=True)
        ep += 1
    n = min(agent.buffer.size, n_target)
    idx = np.random.choice(agent.buffer.size, n, replace=False)
    np.savez_compressed(META_PREFIX + '_buf.npz',
                        w=agent.buffer.w[idx], a=agent.buffer.a[idx], r=agent.buffer.r[idx],
                        w2=agent.buffer.w2[idx], d=agent.buffer.d[idx], basal=agent.buffer.basal[idx])
    print(f'meta buffer ({n}) saved -> {META_PREFIX}_buf.npz', flush=True)


def finetune_full(patient_id, meta_path=META_PREFIX + '.pt', episodes=80, seed=0,
                  ess_after=10):
    """stage 2: warm-start from meta, fine-tune on the new patient with ESS-reweighted buffer reuse; returns actor path"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agent = RecurrentSAC()
    ck = torch.load(meta_path, map_location='cpu')
    agent.actor.load_state_dict(ck['actor'])
    agent.critic.load_state_dict(ck['critic'])
    agent.critic_t.load_state_dict(ck['critic'])
    agent.warmup = 0          # already pretrained -> learn from step 1
    print(f'warm-started actor+critic from {meta_path}', flush=True)

    meta_arrays = None
    bufp = META_PREFIX + '_buf.npz'
    if os.path.exists(bufp):
        z = np.load(bufp)
        meta_arrays = (z['w'], z['a'], z['r'], z['w2'], z['d'], z['basal'])
        print(f'loaded meta buffer ({len(z["a"])}) for ESS reuse', flush=True)
    else:
        print('no meta buffer -> fine-tune without experience reuse', flush=True)

    best = -1.0
    for ep in range(episodes):
        m = run_episode(agent, patient_id, seed=seed + ep, train=True)
        best = max(best, m['tir'])
        if meta_arrays is not None and ep == ess_after:
            meta_obs = meta_arrays[0][:, -1, :OBS_DIM]               # current obs per meta txn
            new_obs  = agent.buffer.w[:agent.buffer.size, -1, :OBS_DIM]
            beta, ess = ess_weights(meta_obs, new_obs)
            agent.attach_meta(meta_arrays, beta, ess)
            print(f'  ESS={ess:.3f} -> reuse meta experience at that batch fraction', flush=True)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  ft ep {ep+1:3d}/{episodes} p{patient_id}  tir={m["tir"]:5.1f}%  '
                  f'hypo={m["hypo"]:4.1f}%  best={best:.1f}%', flush=True)
    out = f'freestyle_rl/hycpap_ft_p{patient_id}.pt'
    torch.save({'actor': agent.actor.state_dict()}, out)
    print(f'fine-tuned actor saved -> {out}', flush=True)
    return out


def eval_full(patient_id, actor=None, mode='blend', seed=0, hypo_floor=70.0):
    """inference: zone-MPC prior blended with the SAC ensemble (gaussian product) + bolus calc for announced meals; mode in {mpc,sac,blend}"""
    from main import (PREBOLUS_STEPS, get_clinical_params, get_body_weight)
    from freestyle_rl.ppo_finetune import make_env, calibrate
    from utils.statespace import StateSpace
    from utils.pumpAction import Pump, get_basal
    from zone_mpc import ZoneMPC, bolus_calculator

    env, args, name = make_env(patient_id, seed=seed)
    basal = get_basal(name); clin = get_clinical_params(name)
    bw = get_body_weight(name); isf = 1700.0 / clin['tdd']
    pump = Pump(args, patient_name=name); ss = StateSpace(args)
    _, cgm = calibrate(env, ss, args, pump)
    mpc = ZoneMPC(basal_rate=basal, isf=isf, basal_bg=cgm, body_weight=bw)

    try:
        sched = env.env.scenario.scenario.get('meal', {'time': [], 'amount': []})
        meal_steps   = [t / 5.0 - args.calibration for t in sched['time']]
        meal_amounts = [float(a) for a in sched['amount']]
    except Exception:
        meal_steps, meal_amounts = [], []

    X = I = 0.0; iob = 0.0; prev_cgm = cgm
    hist = [np.zeros(OBS_DIM + 2, dtype=np.float32) for _ in range(WINDOW)]
    bolus_q = 0.0; bolused = set()
    prev_a = prev_r = 0.0
    cgms, ins_tr, infos = [], [], []
    for step in range(288):
        # announced meal -> bolus calculator (queued, delivered at cap)
        for idx, (ms, ma) in enumerate(zip(meal_steps, meal_amounts)):
            if idx not in bolused and step <= ms <= step + PREBOLUS_STEPS:
                bolus_q += bolus_calculator(ma, cgm, clin['icr'], isf, iob=iob)
                bolused.add(idx)
        carbs_ann = sum(ma for ms, ma in zip(meal_steps, meal_amounts)
                        if step <= ms <= step + PREBOLUS_STEPS)
        bolus_now = min(bolus_q / 5.0, mpc.u_max) if bolus_q > 0 else 0.0
        bolus_q = max(0.0, bolus_q - bolus_now * 5.0)

        v = (cgm - prev_cgm) / 5.0
        u_mpc, sig_mpc = mpc.compute(cgm, X, I, v_now=v)

        if mode == 'mpc' or actor is None:
            corr = u_mpc
        else:
            o = obs_vector(cgm, bolus_now, carbs_ann, iob)
            hist.append(np.concatenate([o, [prev_a, prev_r]]).astype(np.float32))
            if len(hist) > WINDOW: hist.pop(0)
            win = torch.from_numpy(np.stack(hist)).unsqueeze(0)
            # Masksembles: each mask -> member dose + delta-method spread, then aggregate
            mus_m, sigs_m = [], []
            with torch.no_grad():
                for mi in range(actor.num_masks):
                    mu_pre, log_sig = actor.forward(win, mask_idx=mi)
                    mu_pre  = float(mu_pre.item())
                    sig_pre = float(log_sig.exp().item())
                    a_det   = math.tanh(mu_pre)
                    dose_m  = float(min(max(basal * 0.5 * (a_det + 1.0) * ACT_MAX_X, 0.0), PUMP_MAX))
                    dscale  = abs(basal * 0.5 * ACT_MAX_X * (1.0 - a_det * a_det))  # d(dose)/d(raw)
                    mus_m.append(dose_m)
                    sigs_m.append(max(dscale * sig_pre, 1e-4))
            mu_pi, sig_pi = ensemble_gaussian(mus_m, sigs_m)
            if mode == 'sac':
                corr = mu_pi
            else:                                   # Gaussian-product blend
                corr, _ = gaussian_product(u_mpc, sig_mpc, mu_pi, sig_pi)

        # total = meal bolus + correction, capped by the MPC hypo constraint
        pump_act = mpc.hypo_cap(bolus_now + corr, cgm, X, I, floor=hypo_floor)
        s, _, _, info = env.step(pump_act)
        prev_cgm, cgm = cgm, s.CGM
        iob = 0.93 * iob + pump_act * 5.0
        dI = -mpc.n * I + 1000.0 * (pump_act - mpc.u_basal) / mpc.V_I
        dX = -mpc.p2 * X + mpc.p3 * I
        I += mpc.dt * dI; X += mpc.dt * dX
        prev_a, prev_r = (pump_act / max(basal, 1e-6)), paper_reward(cgm) * REWARD_SCALE
        cgms.append(cgm); ins_tr.append(pump_act); infos.append(info)
    c = np.array(cgms)
    return {'tir': float(np.mean((70 <= c) & (c <= 180)) * 100),
            'hypo': float(np.mean(c < 70) * 100), 'hyper': float(np.mean(c > 180) * 100),
            'cgm_trace': cgms, 'ins_trace': ins_tr, 'infos': infos,
            'bg_target': mpc.bg_target}


def eval_and_plot(patient_id, actor, seed=0, cmd_tag='run', hypo_floor=70.0):
    """eval the blend (mode selectable), print metrics, save a plot"""
    from main import plot_results
    from datetime import datetime
    res = {}
    for md in ['blend']: #['mpc', 'sac', 'blend']:
        m = eval_full(patient_id, actor=actor, mode=md, seed=seed, hypo_floor=hypo_floor)
        res[md] = m
        print(f'{md:<6} TIR={m["tir"]:5.1f}%  hypo={m["hypo"]:4.1f}%  hyper={m["hyper"]:4.1f}%', flush=True)
    ts = os.environ.get('RUN_TS') or datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = f'bash_results/result_all/results_at_{ts}'
    os.makedirs(outdir, exist_ok=True)
    # label = {'mpc': 'ZoneMPC', 'sac': 'SAC', 'blend': 'HyCPAP-blend'}
    label = {'blend': 'HyCPAP-blend'}
    for md in ['blend']:
        m = res[md]
        fig = plot_results({'cgm': m['cgm_trace'], 'insulin': m['ins_trace'],
                            'infos': m['infos'], 'bg_target': m['bg_target']},
                           patient_id,
                           cmd=f'python hycpap_full.py {cmd_tag} --patient_id {patient_id}  [{label[md]}]')
        fname = f'{outdir}/hycpap_{md}_{ts}_p{patient_id}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'Plot saved to {fname}', flush=True)
    return res


if __name__ == '__main__':
    # one command per patient = train -> eval -> plot; `--meta` (re)trains the prior
    ap = argparse.ArgumentParser()
    ap.add_argument('--patient_id', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--meta', action='store_true',
                    help='(re)train the shared meta policy, then exit (one-time, ~4h)')
    ap.add_argument('--meta_episodes', type=int, default=2500)
    ap.add_argument('--cohort', choices=['adults', 'adolescent', 'child', 'all'],
                    default='adults',
                    help='meta-pretrain pool; paper uses adults (ids 20-29), the default')
    ap.add_argument('--general', action='store_true',
                    help='train this patient SAC from scratch instead of meta warm-start + ESS')
    a = ap.parse_args()

    if a.meta:
        _pool = {'adults': list(range(20, 30)), 'adolescent': list(range(0, 10)),
                 'child': list(range(10, 20)), 'all': list(range(30))}[a.cohort]
        print(f'meta cohort = {a.cohort}  (patient ids {_pool[0]}-{_pool[-1]})', flush=True)
        meta_pretrain_full(episodes=a.meta_episodes, seed=a.seed, pool=_pool)
        raise SystemExit

    if a.general or not os.path.exists(META_PREFIX + '.pt'):
        # general case: per-patient SAC from scratch
        if not a.general:
            print('no meta policy -- run `python hycpap_full.py --meta` once for '
                  'Meta-HyCPAP; training from scratch (general case) for now...', flush=True)
        else:
            print('general-case HyCPAP: training this patient SAC from scratch', flush=True)
        actor = train_one(a.patient_id, episodes=300, seed=a.seed).actor
    else:
        print('Meta-HyCPAP: meta warm-start + ESS fine-tune', flush=True)
        actor = load_actor(finetune_full(a.patient_id, episodes=3, ess_after=1, seed=a.seed))
    eval_and_plot(a.patient_id, actor, seed=a.seed)
