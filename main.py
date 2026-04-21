"""
Glucose regulation via RL — paper: LRL2 only (no patient model, no clinician data).

Loss: L = LRL2
      LRL2 = A2C policy gradient on simglucose trajectories
           = -E[log_pi(a|s) * A(s,a)] - ent_coef*H(pi)
      Value: MSE(V(s), returns)

Architecture (adapted from G2P2C repo):
  fR  : LSTM encoder          [seq_len, n_features] -> [16]   (separate for actor & critic)
  pi  : 3-layer MLP + NormedLinear  [16] -> mu (tanh), sigma (sigmoid) -> Normal
  V   : 3-layer MLP + NormedLinear  [16] -> scalar
"""
import os, sys, random, torch
import numpy as np
import matplotlib.pyplot as plt
from decouple import config
from agent import Agent

# ── path ──────────────────────────────────────────────────────────────────────
MAIN_PATH = os.environ.get('MAIN_PATH')
if MAIN_PATH is None:
    try:    MAIN_PATH = config('MAIN_PATH')
    except: MAIN_PATH = None
if MAIN_PATH is None:
    MAIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'simulation', 'G2P2C')
if MAIN_PATH not in sys.path:
    sys.path.insert(0, MAIN_PATH)

from utils.core       import get_env, get_patient_env, custom_reward
from utils.options    import Options
from utils.statespace import StateSpace
from utils.pumpAction import Pump, get_basal
GLUCOSE_MIN, GLUCOSE_MAX = 39, 600  # normalization range from options.py


class RuleBasedAgent:
    """Zone-based heuristic policy — no training required.
    Converts normalized CGM back to mg/dL, then applies clinical rules.
    Uses same exponential pump mapping as the PPO agent.
    """
    def select_action(self, state_matrix, std_basal, deterministic=False):
        import math
        cgm      = state_matrix[-1, 0] * (GLUCOSE_MAX - GLUCOSE_MIN) + GLUCOSE_MIN
        prev_cgm = state_matrix[-2, 0] * (GLUCOSE_MAX - GLUCOSE_MIN) + GLUCOSE_MIN
        delta    = cgm - prev_cgm
        meal_ann = state_matrix[-1, 2]

        # map CGM zone to action ∈ [-1, 1]  (action=0 → basal)
        if cgm < 80:
            action = -0.5                          # below basal
        elif cgm < 130:
            action = 0.1 if delta > 2 else 0.0    # near basal
        elif cgm < 180:
            action = 0.3 if delta > 2 else 0.2
        elif cgm < 250:
            action = 0.5
        else:
            action = 0.7

        if meal_ann > 0.05:
            action = min(action + 0.3, 1.0)        # pre-bolus boost

        pump_act = std_basal * math.exp(4.0 * action)
        pump_act = max(pump_act, std_basal * 0.1)
        return pump_act, None, action

    def update(self, *args, **kwargs):
        pass  # no training


def clinical_reward(cgm, prev_cgm=None):
    """Zone-based reward aligned with T1D clinical guidelines.
    Pre-meal target: 80-130 mg/dL  (reward = 1.0)
    Post-meal goal:  <180 mg/dL
    Realistic ceil:  <220 mg/dL
    Severe hypo:     <54 mg/dL  (hard cliff)
    """
    if cgm < 54:
        return -15.0
    elif cgm < 70:
        r = -2.0 - 6.0 * (70 - cgm) / 16       # -2 at 70, -8 at 54
    elif cgm < 80:
        r = -0.5 * (80 - cgm) / 10              # -0.5 at 70, 0 at 80
    elif cgm <= 130:
        r = 1.0                                  # perfect range
    elif cgm <= 180:
        r = 1.0 - 0.5 * (cgm - 130) / 50        # 1.0→0.5 as 130→180
    elif cgm <= 220:
        r = 0.5 - 1.0 * (cgm - 180) / 40        # 0.5→-0.5 as 180→220
    elif cgm <= 300:
        r = -0.5 - 5.0 * (cgm - 220) / 80       # -0.5→-5.5 as 220→300
    else:
        r = -5.5 - 9.5 * min((cgm - 300) / 100, 1.0)  # -5.5→-15 as 300→400+

    # penalize rapid CGM decline (overdosing signal)
    if prev_cgm is not None:
        delta = cgm - prev_cgm  # negative = falling
        if delta < -4:          # dropping >4 mg/dL per 5-min step
            r += 0.05 * delta   # small penalty proportional to drop rate

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
            cgm       = s.CGM,
            ins       = std_basal,
            meal      = info['remaining_time'],
            hour      = t,
            meal_type = info['meal_type'],
            carbs     = info['future_carb'],
        )
        if info['meal_type'] != 0:
            reinit = True

    if reinit:
        return calibrate(env, ss, args, pump)

    return state_matrix


def collect_episode(agent, env, args, pump, deterministic=False):
    ss = StateSpace(args)
    state_matrix = calibrate(env, ss, args, pump)
    std_basal = pump.get_basal()

    log_probs, rewards, actions, states = [], [], [], []
    cgm_trace, ins_trace, infos = [], [], []
    prev_cgm = None

    for step in range(288):
        # store s_t BEFORE acting so states[t] aligns with actions[t]
        states.append(state_matrix.copy())

        pump_act, logp, idx = agent.select_action(
            state_matrix, std_basal, deterministic=deterministic)

        s, _, _, info = env.step(pump_act)
        reward = clinical_reward(s.CGM, prev_cgm)
        prev_cgm = s.CGM

        state_matrix, _ = ss.update(
            cgm       = s.CGM,
            ins       = pump_act,
            meal      = info['remaining_time'],
            hour      = step + 1,
            meal_type = info['meal_type'],
            carbs     = info['future_carb'],
        )

        if not deterministic:
            log_probs.append(logp)
        actions.append(idx)
        rewards.append(reward)
        cgm_trace.append(s.CGM)
        ins_trace.append(pump_act)
        infos.append(info)

    return {
        'log_probs': log_probs, 'rewards': rewards,
        'actions': actions, 'states': states,
        'final_state': state_matrix.copy(),  # s_T for GAE bootstrap
        'cgm_trace': cgm_trace, 'ins_trace': ins_trace, 'infos': infos,
    }


def train_loop(agent, env, args, n_episodes=2800, batch_size=14):
    patients, _ = get_patient_env()
    patient_name = patients[args.patient_id]
    pump = Pump(args, patient_name=patient_name)
    std_basal = pump.get_basal()
    print(f'patient={patient_name}  basal={std_basal:.5f} U/min')

    reward_history = []
    episodes = []

    for ep in range(n_episodes):
        data = collect_episode(agent, env, args, pump, deterministic=False)

        total = sum(data['rewards'])
        reward_history.append(total)

        ins_arr = np.array(data['ins_trace'])
        print(f"ep {ep+1:4d} | reward={total:8.2f} | "
              f"cgm mean={np.mean(data['cgm_trace']):6.1f} | "
              f"ins mean={ins_arr.mean():.5f} U/min")

        episodes.append(data)
        if (ep + 1) % batch_size == 0:
            agent.update(episodes)
            episodes = []

    return reward_history


def eval_loop(agent, env, args):
    patients, _ = get_patient_env()
    patient_name = patients[args.patient_id]
    pump = Pump(args, patient_name=patient_name)

    print('\nRunning 24h evaluation (deterministic)...')
    data = collect_episode(agent, env, args, pump, deterministic=True)
    cgm = data['cgm_trace']
    print(f'Eval | reward={sum(data["rewards"]):.2f} | '
          f'cgm mean={np.mean(cgm):.1f} '
          f'min={np.min(cgm):.1f} '
          f'max={np.max(cgm):.1f}')

    return {'cgm': cgm, 'insulin': data['ins_trace'], 'infos': data['infos']}


def plot_results(reward_history, eval_trace, patient_id, cmd=None):
    cgm   = eval_trace['cgm']
    ins   = eval_trace['insulin']
    infos = eval_trace['infos']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [1, 2, 1]})
    fig.subplots_adjust(hspace=0.45)

    ax = axes[0]
    if reward_history:
        ax.plot(reward_history, color='steelblue', lw=0.8, alpha=0.4)
        w  = max(1, len(reward_history) // 20)
        sm = np.convolve(reward_history, np.ones(w) / w, mode='valid')
        ax.plot(range(w - 1, len(reward_history)), sm,
                color='navy', lw=2, label=f'Smoothed (w={w})')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Rule-based — no training', ha='center',
                va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title(f'Patient {patient_id} — Training Reward (LRL2)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward')
    ax.grid(True, alpha=0.25)

    # eval CGM
    steps = np.arange(len(cgm))
    meal_steps, meal_grams = [], []
    i = 0
    while i < len(infos):
        if infos[i].get('meal_type', 0) != 0:
            start = i
            carbs = infos[i].get('future_carb', 0)
            while i < len(infos) and infos[i].get('meal_type', 0) != 0:
                i += 1
            meal_steps.append(start)
            meal_grams.append(round(carbs, 0))
        else:
            i += 1

    ax = axes[1]
    ax.axhspan(70, 180, color='limegreen', alpha=0.15, label='Target (70–180)')
    ax.axhline(70,  color='orange', lw=1, ls='--', alpha=0.7)
    ax.axhline(180, color='orange', lw=1, ls='--', alpha=0.7)
    ax.axhline(54,  color='red', lw=1, ls=':', alpha=0.6, label='Severe hypo (54)')
    ax.axhline(250, color='red', lw=1, ls=':', alpha=0.6, label='Severe hyper (250)')
    ax.plot(steps, cgm, 'b-', lw=2, label='CGM')
    for t, g in zip(meal_steps, meal_grams):
        ax.axvline(t, color='saddlebrown', lw=1.5, alpha=0.8)
        ax.text(t + 1, 390, f'{g:.0f}g CHO', fontsize=8,
                color='saddlebrown', va='top')
    ax.set_ylim(0, 430); ax.set_ylabel('CGM (mg/dL)')
    ax.set_title('Evaluation Episode — 24h Deterministic')
    ax.legend(loc='upper right', fontsize=9); ax.grid(True, alpha=0.25)

    # eval insulin
    ax = axes[2]
    ins_arr  = np.array(ins)
    ins_plot = np.where(ins_arr > 0, ins_arr, np.nan)
    ax.bar(steps, ins_plot, width=0.8, color='mediumseagreen', alpha=0.85)
    for t in meal_steps:
        ax.axvline(t, color='saddlebrown', lw=1.2, alpha=0.5)
    ax.set_yscale('log'); ax.set_ylim(bottom=1e-6)
    ax.set_ylabel('Insulin (U/min, log)'); ax.set_xlabel('Step (5 min each)')
    ax.grid(True, alpha=0.25, axis='y')

    title = cmd if cmd else f'Glucose Regulation — Patient {patient_id}'
    plt.suptitle(title, fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    args   = Options().parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    patients, env_ids = get_patient_env()
    patient  = patients[args.patient_id]
    env_id   = env_ids[args.patient_id]
    env = get_env(args, patient_name=patient, env_id=env_id,
                  custom_reward=custom_reward, seed=args.seed)
    print('meal scenario:', env.env.scenario.scenario['meal'])
    dummy_ss  = StateSpace(args)
    dummy_obs = env.reset()
    s0, _     = dummy_ss.update(cgm=float(dummy_obs.CGM), ins=0, meal=0)
    n_features = s0.shape[1]
    print(f'n_features={n_features}  seq_len={s0.shape[0]}')

    std_basal = get_basal(patient)
    args.action_scale = 5.0
    print(f'basal={std_basal:.5f} U/min  action_scale={args.action_scale}')

    agent = Agent(
        n_features   = n_features,
        action_scale = args.action_scale,
        device       = device,
        n_hidden     = 64,
        n_layers     = 1,
        pi_lr        = 3e-4,
        vf_lr        = 3e-4,
        gamma        = 0.99,
        lambda_      = 0.95,
        entropy_coef = 0.001,
        grad_clip    = 20,
        eps_clip     = 0.1,
        n_epochs     = 5,
        target_kl        = 0.01,
        normalize_reward = True,
    )

    # 2800 episodes x 288 steps ≈ 800k transitions
    reward_history = train_loop(agent, env, args, n_episodes=3000)
    eval_trace = eval_loop(agent, env, args)


    from datetime import datetime
    cmd = 'python ' + ' '.join(sys.argv)
    fig = plot_results(reward_history, eval_trace, args.patient_id, cmd=cmd)
    os.makedirs('bash_results', exist_ok=True)
    fname = f"bash_results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {fname}')
    plt.show()


def main_rule():
    """Run the rule-based baseline — no training, instant results."""
    args   = Options().parse()
    patients, env_ids = get_patient_env()
    patient  = patients[args.patient_id]
    env_id   = env_ids[args.patient_id]
    env = get_env(args, patient_name=patient, env_id=env_id,
                  custom_reward=custom_reward, seed=args.seed)

    agent = RuleBasedAgent()
    eval_trace = eval_loop(agent, env, args)

    from datetime import datetime
    cmd = 'python ' + ' '.join(sys.argv)
    fig = plot_results([], eval_trace, args.patient_id, cmd=cmd)
    os.makedirs('bash_results', exist_ok=True)
    fname = f"bash_results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {fname}')
    plt.show()


def plot_all_patients(all_results):
    """
    all_results: list of (reward_history, eval_trace, patient_name)
    One row per patient, 3 columns: reward | CGM | insulin
    """
    n = len(all_results)
    fig, axes = plt.subplots(n, 3, figsize=(18, n * 2.8),
                             gridspec_kw={'width_ratios': [1, 1.5, 1]})
    fig.suptitle('Glucose Regulation — All Patients', fontsize=14,
                 fontweight='bold')
    fig.subplots_adjust(hspace=0.6, wspace=0.35)

    for row, (reward_history, eval_trace, patient_name) in enumerate(all_results):
        cgm   = np.array(eval_trace['cgm'])
        ins   = np.array(eval_trace['insulin'])
        infos = eval_trace['infos']
        steps = np.arange(len(cgm))

        # col 0: training reward
        ax = axes[row, 0]
        rh = np.array(reward_history)
        ax.plot(rh, color='steelblue', lw=0.6, alpha=0.4)
        w  = max(len(rh) // 20, 1)
        sm = np.convolve(rh, np.ones(w) / w, mode='valid')
        ax.plot(range(w - 1, len(rh)), sm, color='navy', lw=1.2)
        ax.set_ylabel(patient_name, fontsize=6, rotation=0,
                      labelpad=58, va='center')
        ax.tick_params(labelsize=6); ax.grid(True, alpha=0.2)
        if row == 0: ax.set_title('Training Reward', fontsize=9)

        # col 1: CGM
        ax = axes[row, 1]
        ax.axhspan(70, 180, color='limegreen', alpha=0.15)
        ax.axhline(70,  color='orange', lw=0.7, ls='--', alpha=0.7)
        ax.axhline(180, color='orange', lw=0.7, ls='--', alpha=0.7)
        ax.axhline(54,  color='red',    lw=0.7, ls=':',  alpha=0.6)
        ax.plot(steps, cgm, 'b-', lw=1.0)
        i = 0
        while i < len(infos):
            if infos[i].get('meal_type', 0) != 0:
                carbs = infos[i].get('future_carb', 0)
                ax.axvline(i, color='saddlebrown', lw=0.8, alpha=0.7)
                ax.text(i + 1, cgm.max() * 0.95, f'{carbs:.0f}g',
                        fontsize=5, color='saddlebrown', va='top')
                while i < len(infos) and infos[i].get('meal_type', 0) != 0:
                    i += 1
            else:
                i += 1
        ax.set_ylim(0, max(cgm.max() * 1.15, 300))
        ax.tick_params(labelsize=6); ax.grid(True, alpha=0.2)
        if row == 0: ax.set_title('CGM (mg/dL)', fontsize=9)

        # col 2: insulin
        ax = axes[row, 2]
        ins_plot = np.where(ins > 0, ins, np.nan)
        ax.bar(steps, ins_plot, width=0.8, color='mediumseagreen', alpha=0.8)
        ax.tick_params(labelsize=6); ax.grid(True, alpha=0.2, axis='y')
        if row == 0: ax.set_title('Insulin (U/min)', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def main_all(n_episodes=400):
    """Train one agent per patient across all 30 patients, plot combined figure."""
    args   = Options().parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    patients, env_ids = get_patient_env()
    all_results = []

    for patient_id in range(30):
        patient = patients[patient_id]
        env_id  = env_ids[patient_id]
        args.patient_id = patient_id

        print(f'\n{"="*55}')
        print(f'Patient {patient_id:2d}: {patient}')
        print(f'{"="*55}')

        env = get_env(args, patient_name=patient, env_id=env_id,
                      custom_reward=custom_reward, seed=args.seed + patient_id)

        dummy_ss  = StateSpace(args)
        dummy_obs = env.reset()
        s0, _     = dummy_ss.update(cgm=float(dummy_obs.CGM), ins=0, meal=0)
        n_features = s0.shape[1]

        std_basal = get_basal(patient)
        args.action_scale = 5.0
        print(f'  basal={std_basal:.5f} U/min  action_scale={args.action_scale}')

        agent = Agent(
            n_features   = n_features,
            action_scale = args.action_scale,
            device       = device,
            n_hidden     = 64,
            n_layers     = 1,
            pi_lr        = 3e-4,
            vf_lr        = 3e-4,
            gamma        = 0.99,
            lambda_      = 0.95,
            entropy_coef = 0.001,
            grad_clip    = 20,
            eps_clip     = 0.1,
            n_epochs     = 5,
            target_kl        = 0.01,
            normalize_reward = True,
        )

        reward_history = train_loop(agent, env, args, n_episodes=n_episodes)
        eval_trace     = eval_loop(agent, env, args)
        all_results.append((reward_history, eval_trace, patient))
        print(f'  eval cgm mean={np.mean(eval_trace["cgm"]):.1f}  '
              f'min={np.min(eval_trace["cgm"]):.1f}  '
              f'max={np.max(eval_trace["cgm"]):.1f}')

    print('\nAll 30 patients done. Plotting...')
    fig = plot_all_patients(all_results)
    fig.savefig('simulation_result_all.png', dpi=120, bbox_inches='tight')
    print('Saved to simulation_result_all.png')
    plt.show()


if __name__ == '__main__':
    import sys
    if 'all' in sys.argv:
        sys.argv.remove('all')
        n_ep = 400
        remaining = [a for a in sys.argv[1:] if not a.startswith('-')]
        if remaining and remaining[0].isdigit():
            n_ep = int(remaining[0])
            sys.argv.remove(remaining[0])
        main_all(n_episodes=n_ep)
    elif 'rule' in sys.argv:
        sys.argv.remove('rule')
        main_rule()
    else:
        main()
