"""PPO with reward normalisation, lower exploration, and many more episodes.

Improvements over v1:
- LR 1e-4 (was 3e-4)
- Running reward normaliser (Welford)
- Episodes default 500 (was 100)
- KL early-stop guard
- Entropy bonus
- Value function bootstrapped at horizon (small constant)
"""
import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'simulation', 'G2P2C'))

from main import (
    PREBOLUS_STEPS, get_clinical_params, get_body_weight,
)
from utils.core       import get_env, get_patient_env, custom_reward
from utils.options    import Options
from utils.statespace import StateSpace
from utils.pumpAction import Pump, get_basal

from freestyle_rl.policy import TinyPolicy, ACTION_MAX
from freestyle_rl.state_builder import build_state, iob_decay

GAMMA       = 0.99
LAMBDA      = 0.95
CLIP        = 0.2
EPOCHS      = 4
LR          = 1e-4
ENT_COEF    = 0.01
KL_LIMIT    = 0.05
VAL_COEF    = 0.5


class RewardNormaliser:
    """Welford running stats. Scales rewards by their running std."""
    def __init__(self):
        self.mean = 0.0
        self.M2   = 0.0
        self.n    = 0

    def update(self, r):
        self.n += 1
        delta = r - self.mean
        self.mean += delta / self.n
        self.M2   += delta * (r - self.mean)

    def normalise(self, r):
        if self.n < 2:
            return r
        std = max((self.M2 / self.n) ** 0.5, 1e-6)
        return r / std


def step_reward(cgm):
    if 70.0 <= cgm <= 180.0:
        return 1.0
    if cgm < 70.0:
        r = -0.05 * (70.0 - cgm)
        if cgm <= 50.0:
            r -= 3.0
        return r
    r = -0.01 * (cgm - 180.0)
    if cgm > 300.0:
        r -= 1.0
    return r


def calibrate(env, ss, args, pump):
    std_basal  = pump.get_basal()
    init_state = env.reset()
    pump.calibrate(init_state)
    state_matrix, _ = ss.update(cgm=init_state.CGM, ins=0, meal=0)
    reinit = False
    for t in range(args.calibration):
        s, _, _, info = env.step(std_basal)
        state_matrix, _ = ss.update(
            cgm=s.CGM, ins=std_basal, meal=info['remaining_time'],
            hour=t, meal_type=info['meal_type'], carbs=info['future_carb'],
        )
        if info['meal_type'] != 0:
            reinit = True
    if reinit:
        return calibrate(env, ss, args, pump)
    return state_matrix, s.CGM


def make_env(patient_id, seed):
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = Options().parse()
    finally:
        sys.argv = saved_argv
    args.patient_id = patient_id
    args.seed = seed
    random.seed(seed); np.random.seed(seed)

    patients, env_ids = get_patient_env()
    name = patients[patient_id]

    if 10 <= patient_id < 20:
        bw = get_body_weight(name)
        target_total = 3.5 * bw
        proportions  = [30, 15, 45, 15, 45, 15]
        scale        = target_total / sum(proportions)
        args.meal_amount   = [max(5, round(p * scale)) for p in proportions]
        args.meal_variance = [max(1, round(a / 6)) for a in args.meal_amount]

    import gym.envs.registration as _reg
    _reg.registry.env_specs.pop(env_ids[patient_id], None)
    env = get_env(args, patient_name=name, env_id=env_ids[patient_id],
                  custom_reward=custom_reward, seed=seed)
    return env, args, name


def collect_episode(policy, env, args, patient_id, seed, rew_norm):
    name  = get_patient_env()[0][patient_id]
    basal = get_basal(name)
    bw    = get_body_weight(name)
    clin  = get_clinical_params(name)

    pump = Pump(args, patient_name=name)
    ss   = StateSpace(args)
    _, cgm_now = calibrate(env, ss, args, pump)

    try:
        sched = env.env.scenario.scenario.get('meal', {'time': [], 'amount': []})
        meal_steps   = [t / 5.0 - args.calibration for t in sched['time']]
        meal_amounts = [float(a) for a in sched['amount']]
    except Exception:
        meal_steps, meal_amounts = [], []

    states, raws, log_probs, values, raw_rewards = [], [], [], [], []
    cgm_trace = []
    iob = 0.0
    prev_cgm = cgm_now

    for step in range(288):
        announced = 0.0
        next_meal_min = 240.0
        for ms, ma in zip(meal_steps, meal_amounts):
            if ms >= step:
                if ms - step <= PREBOLUS_STEPS:
                    announced = ma
                next_meal_min = (ms - step) * 5.0
                break

        s = build_state(basal, bw, clin['isf'], clin['icr'], clin['tdd'],
                        cgm_now=cgm_now, cgm_prev=prev_cgm,
                        iob_proxy=iob, meal_carbs_announced=announced,
                        mins_to_next_meal=next_meal_min)
        s_t = torch.from_numpy(s).unsqueeze(0)
        with torch.no_grad():
            action, log_p, value, raw = policy.act_stochastic(s_t)
        u = max(0.0, min(ACTION_MAX, float(action.item())))

        env_obs, _, _, info = env.step(u)
        prev_cgm = cgm_now
        cgm_now  = env_obs.CGM
        iob      = iob_decay(iob, u * 5.0)

        r = step_reward(cgm_now)
        rew_norm.update(r)

        states.append(s)
        raws.append(float(raw.item()))
        log_probs.append(float(log_p.item()))
        values.append(float(value.item()))
        raw_rewards.append(r)
        cgm_trace.append(cgm_now)

    rewards = np.array([rew_norm.normalise(r) for r in raw_rewards], dtype=np.float32)
    cgm_arr = np.array(cgm_trace)
    tir = float(np.mean((70 <= cgm_arr) & (cgm_arr <= 180)) * 100)
    hypo = float(np.mean(cgm_arr < 70) * 100)

    return {
        'states':      np.stack(states),
        'raws':        np.array(raws, dtype=np.float32),
        'log_probs':   np.array(log_probs, dtype=np.float32),
        'values':      np.array(values, dtype=np.float32),
        'rewards':     rewards,
        'raw_return':  float(np.sum(raw_rewards)),
        'tir':         tir,
        'hypo':        hypo,
    }


def gae(rewards, values, gamma=GAMMA, lam=LAMBDA):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextvalue = 0.0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextvalue - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def ppo_update(policy, opt, batch, epochs=EPOCHS, clip=CLIP):
    S = torch.from_numpy(batch['states'])
    R = torch.from_numpy(batch['raws'])
    old_log_p = torch.from_numpy(batch['log_probs'])
    adv = torch.from_numpy(batch['adv'])
    ret = torch.from_numpy(batch['returns'])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for ep in range(epochs):
        log_p, value, entropy = policy.evaluate(S, R)
        with torch.no_grad():
            kl = (old_log_p - log_p).mean()
        if kl > KL_LIMIT:
            break
        ratio = (log_p - old_log_p).exp()
        s1 = ratio * adv
        s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        policy_loss = -torch.min(s1, s2).mean()
        value_loss  = F.mse_loss(value, ret)
        ent_loss    = -entropy.mean()
        loss = policy_loss + VAL_COEF * value_loss + ENT_COEF * ent_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()


def fine_tune(patient_id, init_policy_path=None,
              n_episodes=500, seed=0, out_path=None):
    pol = TinyPolicy()
    if init_policy_path and os.path.exists(init_policy_path):
        pol.load_state_dict(torch.load(init_policy_path, map_location='cpu'))
        print(f'loaded init from {init_policy_path}')
    else:
        print('starting from zero-init policy (outputs basal)')
    opt = torch.optim.Adam(pol.parameters(), lr=LR)
    rew_norm = RewardNormaliser()

    print(f'PPO p{patient_id} for {n_episodes} episodes')
    best_tir = -1.0
    log_interval = max(1, n_episodes // 50)
    for ep in range(n_episodes):
        env, args, _ = make_env(patient_id, seed=seed + ep)
        traj = collect_episode(pol, env, args, patient_id, seed=seed + ep,
                               rew_norm=rew_norm)
        adv, ret = gae(traj['rewards'], traj['values'])
        traj['adv'], traj['returns'] = adv, ret
        ppo_update(pol, opt, traj)

        if traj['tir'] > best_tir:
            best_tir = traj['tir']
            if out_path:
                torch.save(pol.state_dict(), out_path)

        if (ep + 1) % log_interval == 0 or ep == 0:
            print(f'  ep {ep+1:3d}/{n_episodes}: '
                  f'raw_ret={traj["raw_return"]:7.1f}  '
                  f'tir={traj["tir"]:5.1f}%  hypo={traj["hypo"]:5.1f}%  '
                  f'(best_tir={best_tir:5.1f}%)')

    print(f'\nbest TIR achieved: {best_tir:.1f}%')
    if out_path:
        print(f'saved best -> {out_path}')
    return pol


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--patient_id', type=int, default=6)
    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--init', default=None,
                    help='start from zero-init basal-output policy by default; '
                         'pass freestyle_rl/bc_policy.pt to warm-start from BC')
    ap.add_argument('--out',  default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = f'freestyle_rl/ppo_p{a.patient_id}.pt'
    fine_tune(a.patient_id, init_policy_path=a.init, n_episodes=a.episodes,
              out_path=a.out)
