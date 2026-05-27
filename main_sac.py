"""Running SAC algo
"""
import os, sys, random, torch
import numpy as np
import matplotlib.pyplot as plt
from decouple import config
from agent_sac import SACAgent

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

GLUCOSE_MIN, GLUCOSE_MAX = 39, 600

#modify reward
def clinical_reward(cgm, prev_cgm=None):
    if cgm < 54:
        return -15.0
    elif cgm < 70:
        r = -2.0 - 6.0 * (70 - cgm) / 16
    elif cgm < 80:
        r = -0.5 * (80 - cgm) / 10
    elif cgm <= 130:
        r = 1.0
    elif cgm <= 180:
        r = 1.0 - 0.5 * (cgm - 130) / 50
    elif cgm <= 220:
        r = 0.5 - 1.0 * (cgm - 180) / 40
    elif cgm <= 300:
        r = -0.5 - 5.0 * (cgm - 220) / 80
    else:
        r = -5.5 - 9.5 * min((cgm - 300) / 100, 1.0)
    if prev_cgm is not None:
        delta = cgm - prev_cgm
        if delta < -4:            # rapid fall — penalise over-dosing
            r += 0.05 * delta
        if delta > 4 and cgm > 180:  # rapid rise in hyperglycemia zone — penalise
            r -= 0.02 * delta
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
    return state_matrix


def collect_episode(agent, env, args, pump, deterministic=False):
    ss = StateSpace(args)
    state_matrix = calibrate(env, ss, args, pump)
    std_basal = pump.get_basal()

    rewards, actions, cgm_trace, ins_trace, infos = [], [], [], [], []
    last_stats = None

    for step in range(288):
        s_prev = state_matrix.copy()
        pump_act, _, raw_a = agent.select_action(s_prev, std_basal, deterministic=deterministic)
        s, reward, _, info = env.step(pump_act)

        state_matrix, _ = ss.update(
            cgm=s.CGM, ins=pump_act, meal=info['remaining_time'],
            hour=step + 1, meal_type=info['meal_type'], carbs=info['future_carb'],
        )

        if not deterministic:
            done = (step == 287)
            agent.push(s_prev, raw_a, reward, state_matrix.copy(), done)
            stats = agent.update()
            if stats is not None:
                last_stats = stats

        actions.append(raw_a)
        rewards.append(reward)
        cgm_trace.append(s.CGM)
        ins_trace.append(pump_act)
        infos.append(info)

    return {
        'actions': actions, 'rewards': rewards,
        'cgm_trace': cgm_trace, 'ins_trace': ins_trace, 'infos': infos,
        'last_stats': last_stats,
    }


def train_loop(agent, env, args, n_episodes=2800):
    patients, _ = get_patient_env()
    patient_name = patients[args.patient_id]
    pump = Pump(args, patient_name=patient_name)
    std_basal = pump.get_basal()
    print(f'patient={patient_name}  basal={std_basal:.5f} U/min')

    reward_history = []
    for ep in range(n_episodes):
        data = collect_episode(agent, env, args, pump, deterministic=False)
        total = sum(data['rewards'])
        reward_history.append(total)
        ins_arr = np.array(data['ins_trace'])
        st = data['last_stats']
        if st is not None:
            print(f"ep {ep+1:4d} | r={total:8.2f} | cgm={np.mean(data['cgm_trace']):6.1f} | "
                  f"ins={ins_arr.mean():.5f} | α={st['alpha']:.3f} | "
                  f"q1={st['q1_loss']:.3f} π={st['pi_loss']:.3f} H={st['entropy']:.3f}")
        else:
            print(f"ep {ep+1:4d} | r={total:8.2f} | cgm={np.mean(data['cgm_trace']):6.1f} | "
                  f"ins={ins_arr.mean():.5f} | (warmup, no update)")
    return reward_history


def eval_loop(agent, env, args):
    patients, _ = get_patient_env()
    patient_name = patients[args.patient_id]
    pump = Pump(args, patient_name=patient_name)
    print('\nRunning 24h evaluation (deterministic)...')
    data = collect_episode(agent, env, args, pump, deterministic=True)
    cgm = data['cgm_trace']
    print(f'Eval | r={sum(data["rewards"]):.2f} | cgm mean={np.mean(cgm):.1f} '
          f'min={np.min(cgm):.1f} max={np.max(cgm):.1f}')
    return {'cgm': cgm, 'insulin': data['ins_trace'], 'infos': data['infos']}


def easy_eval_loop(agent, args, seed_offset=10000):
    import copy
    eval_args = copy.deepcopy(args)
    eval_args.meal_prob     = [1, -1, 1, -1, 1, -1]
    eval_args.meal_amount   = [40, 20, 80, 10, 60, 30]
    eval_args.meal_variance = [1e-8] * 6
    eval_args.time_variance = [1e-8] * 6

    patients, env_ids = get_patient_env()
    patient_name = patients[args.patient_id]
    env_id       = env_ids[args.patient_id]
    easy_env_id  = env_id.replace('-v0', 'sac_easy-v0')
    eval_env = get_env(eval_args, patient_name=patient_name,
                       env_id=easy_env_id, custom_reward=custom_reward,
                       seed=args.seed + seed_offset)
    pump = Pump(eval_args, patient_name=patient_name)

    print('\nRunning EASY-scenario evaluation (deterministic meals)...')
    data = collect_episode(agent, eval_env, eval_args, pump, deterministic=True)
    cgm = data['cgm_trace']
    print(f'EasyEval | r={sum(data["rewards"]):.2f} | cgm mean={np.mean(cgm):.1f} '
          f'min={np.min(cgm):.1f} max={np.max(cgm):.1f}')
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
    ax.set_title(f'Patient {patient_id} — Training Reward (SAC)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward')
    ax.grid(True, alpha=0.25)

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
    args = Options().parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f'action_scale={args.action_scale}')

    if 10 <= args.patient_id < 20:
        args.meal_amount   = [30, 15, 45, 15, 45, 15]
        args.meal_variance = [5,  3,  5,  3,  5,  3]
        print('Using child-sized meal protocol')

    patients, env_ids = get_patient_env()
    patient = patients[args.patient_id]
    env_id  = env_ids[args.patient_id]
    env = get_env(args, patient_name=patient, env_id=env_id, custom_reward=custom_reward, seed=args.seed)
    print('meal scenario:', env.env.scenario.scenario['meal'])

    dummy_ss = StateSpace(args)
    dummy_obs = env.reset()
    s0, _ = dummy_ss.update(cgm=float(dummy_obs.CGM), ins=0, meal=0)
    n_features  = s0.shape[1]
    state_shape = s0.shape
    print(f'n_features={n_features}  seq_len={s0.shape[0]}')

    std_basal = get_basal(patient)
    init_a = float(np.clip(1 + np.log(std_basal / args.action_scale) / 4, -0.95, 0.95))
    init_mu_bias = float(np.arctanh(init_a))
    print(f'basal={std_basal:.5f} U/min  action_scale={args.action_scale}  init_mu_bias={init_mu_bias:.3f}')

    agent = SACAgent(
        n_features      = n_features,
        state_shape     = state_shape,
        action_scale    = args.action_scale,
        device          = device,
        n_hidden        = 64,
        n_layers        = 1,
        gamma           = 0.99,
        tau             = 0.005,
        actor_lr        = 3e-4,
        critic_lr       = 3e-4,
        alpha_lr        = 3e-4,
        buffer_size     = 100000,
        batch_size      = 256,
        target_entropy  = -2.0,
        init_mu_bias    = init_mu_bias,
        warmup_steps    = 5000,
    )

    reward_history = train_loop(agent, env, args, n_episodes=2800)
    eval_trace  = eval_loop(agent, env, args)
    easy_trace  = easy_eval_loop(agent, args)

    from datetime import datetime
    cmd = 'python ' + ' '.join(sys.argv)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('bash_results/normal', exist_ok=True)

    fig = plot_results(reward_history, eval_trace, args.patient_id, cmd=cmd + '  [SAC]')
    fname = f'bash_results/normal/sac_results_{ts}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {fname}')

    fig_easy = plot_results(reward_history, easy_trace, args.patient_id,
                            cmd=cmd + '  [SAC EASY]')
    fname_easy = f'bash_results/normal/sac_results_{ts}_easy.png'
    fig_easy.savefig(fname_easy, dpi=150, bbox_inches='tight')
    print(f'Easy plot saved to {fname_easy}')


if __name__ == '__main__':
    main()
