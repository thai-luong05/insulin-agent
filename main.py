"""pure mpc controller for glucose regulation."""
import os, sys, random
import numpy as np
import matplotlib.pyplot as plt
from decouple import config
from mpc_controller import MPCController, IOB_GAMMA

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


def calibrate(env, ss, args, pump):
    #fill state space with 48 steps of real cgm history
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


PREBOLUS_STEPS = 3  #15 min meal pre-announce


def run_episode(mpc, env, args, pump):
    """24h mpc rollout (288 steps × 5 min)."""
    ss = StateSpace(args)
    state_matrix, current_cgm = calibrate(env, ss, args, pump)

    #iob at basal steady state
    i_eff          = mpc.egp / mpc.alpha
    prev_meal_carbs = 0.0

    #step-indexed meal schedule for pre-bolus lookahead
    try:
        sched = env.env.scenario.scenario.get('meal', {'time': [], 'amount': []})
        meal_steps  = [t / 5.0 - args.calibration
                       for t in sched['time']]
        meal_amounts = [float(a) for a in sched['amount']]
    except Exception:
        meal_steps, meal_amounts = [], []

    rewards, cgm_trace, ins_trace, infos = [], [], [], []

    for step in range(288):
        #pre-bolus if a meal starts within PREBOLUS_STEPS
        lookahead_carbs = 0.0
        for ms, ma in zip(meal_steps, meal_amounts):
            if step < ms <= step + PREBOLUS_STEPS:
                lookahead_carbs = ma
                break
        meal_carbs_for_mpc = max(prev_meal_carbs, lookahead_carbs)

        #use real cgm from env, not state_matrix (normalisation isn't reversible)
        pump_act = mpc.compute_insulin(
            G_now      = current_cgm,
            I_eff      = i_eff,
            G_target   = mpc.bg_target,
            meal_carbs = meal_carbs_for_mpc,
        )

        s, reward, _, info = env.step(pump_act)
        current_cgm = s.CGM

        state_matrix, _ = ss.update(
            cgm=s.CGM, ins=pump_act, meal=info['remaining_time'],
            hour=step + 1, meal_type=info['meal_type'], carbs=info['future_carb'],
        )

        i_eff           = IOB_GAMMA * i_eff + pump_act * 5
        prev_meal_carbs = info.get('future_carb', 0.0)

        rewards.append(reward)
        cgm_trace.append(s.CGM)
        ins_trace.append(pump_act)
        infos.append(info)

    return {
        'rewards': rewards,
        'cgm_trace': cgm_trace,
        'ins_trace': ins_trace,
        'infos': infos,
        'bg_target': mpc.bg_target,
    }


def plot_results(eval_trace, patient_id, cmd=None):
    cgm   = eval_trace['cgm']
    ins   = eval_trace['insulin']
    infos = eval_trace['infos']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(hspace=0.4)

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

    ax = axes[0]
    ax.axhspan(70, 180, color='limegreen', alpha=0.15, label='Target (70-180)')
    ax.axhline(70,  color='orange', lw=1, ls='--', alpha=0.7)
    ax.axhline(180, color='orange', lw=1, ls='--', alpha=0.7)
    ax.axhline(54,  color='red', lw=1, ls=':', alpha=0.6, label='Severe hypo (54)')
    ax.axhline(250, color='red', lw=1, ls=':', alpha=0.6, label='Severe hyper (250)')
    ax.plot(steps, cgm, 'b-', lw=2, label='CGM')
    bg_target = eval_trace.get('bg_target', 110.0)
    ax.axhline(bg_target, color='green', lw=1, ls='--', alpha=0.8, label=f'MPC target ({bg_target:.0f})')
    for t, g in zip(meal_steps, meal_grams):
        ax.axvline(t, color='saddlebrown', lw=1.5, alpha=0.8)
        ax.text(t + 1, 390, f'{g:.0f}g CHO', fontsize=8, color='saddlebrown', va='top')
    ax.set_ylim(0, 430); ax.set_ylabel('CGM (mg/dL)')
    ax.set_title(f'Patient {patient_id} — MPC 24h Evaluation')
    ax.legend(loc='upper right', fontsize=9); ax.grid(True, alpha=0.25)

    ax = axes[1]
    ins_arr  = np.array(ins)
    ins_plot = np.where(ins_arr > 0, ins_arr, np.nan)
    ax.bar(steps, ins_plot, width=0.8, color='mediumseagreen', alpha=0.85)
    for t in meal_steps:
        ax.axvline(t, color='saddlebrown', lw=1.2, alpha=0.5)
    ax.set_yscale('log'); ax.set_ylim(bottom=1e-6)
    ax.set_ylabel('Insulin (U/min, log)'); ax.set_xlabel('Step (5 min each)')
    ax.grid(True, alpha=0.25, axis='y')

    title = cmd if cmd else f'MPC Glucose Regulation — Patient {patient_id}'
    plt.suptitle(title, fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    args = Options().parse()
    random.seed(args.seed)
    np.random.seed(args.seed)

    patients, env_ids = get_patient_env()
    patient = patients[args.patient_id]
    env_id  = env_ids[args.patient_id]
    env = get_env(args, patient_name=patient, env_id=env_id,
                  custom_reward=custom_reward, seed=args.seed)
    print('meal scenario:', env.env.scenario.scenario['meal'])

    std_basal = get_basal(patient)
    print(f'patient={patient}  basal={std_basal:.5f} U/min')

    #1700 rule: ISF=1700/TDD, TDD=basal*1440, alpha=ISF/25
    tdd   = std_basal * 1440.0
    isf   = 1700.0 / tdd
    alpha = isf / 25.0
    print(f'TDD={tdd:.2f} U/day  ISF={isf:.1f} mg/dL/U  alpha={alpha:.3f}')

    #patient-type bolus ceiling
    if args.patient_id >= 20:
        max_bolus_mult = 30.0
    elif args.patient_id < 10:
        max_bolus_mult = 25.0
    else:
        max_bolus_mult = 15.0

    mpc = MPCController(basal_rate=std_basal, alpha=alpha,
                        max_bolus_multiplier=max_bolus_mult)
    mpc.basal_bg  = float(env.reset().CGM)
    #target just below resting bg to avoid fasting over-correction
    mpc.bg_target = max(110.0, mpc.basal_bg - 10.0)
    print(f'MPC ready  basal_bg={mpc.basal_bg:.1f}  bg_target={mpc.bg_target:.1f} mg/dL')
    print(f'           egp={mpc.egp:.4f}  alpha={mpc.alpha:.3f}  gamma={mpc.gamma}  beta={mpc.beta}')

    pump = Pump(args, patient_name=patient)

    #standard eval
    print('\nRunning 24h evaluation...')
    data = run_episode(mpc, env, args, pump)
    cgm  = data['cgm_trace']
    pct_tir  = np.mean([(70 <= g <= 180) for g in cgm]) * 100
    pct_hypo = np.mean([g < 70  for g in cgm]) * 100
    pct_hyper= np.mean([g > 180 for g in cgm]) * 100
    print(f'Eval  | r={sum(data["rewards"]):.2f} | '
          f'cgm mean={np.mean(cgm):.1f}  min={np.min(cgm):.1f}  max={np.max(cgm):.1f}')
    print(f'       TIR(70-180)={pct_tir:.1f}%  hypo={pct_hypo:.1f}%  hyper={pct_hyper:.1f}%')

    #easy eval (deterministic meals)
    import copy
    easy_args = copy.deepcopy(args)
    easy_args.meal_prob     = [1, -1, 1, -1, 1, -1]
    easy_args.meal_amount   = [40, 20, 80, 10, 60, 30]
    easy_args.meal_variance = [1e-8] * 6
    easy_args.time_variance = [1e-8] * 6
    easy_env_id = env_id.replace('-v0', 'mpc_easy-v0')
    easy_env = get_env(easy_args, patient_name=patient, env_id=easy_env_id,
                       custom_reward=custom_reward, seed=args.seed + 10000)
    easy_pump = Pump(easy_args, patient_name=patient)
    easy_mpc  = MPCController(basal_rate=std_basal, alpha=alpha,
                              max_bolus_multiplier=max_bolus_mult)
    easy_mpc.basal_bg  = float(easy_env.reset().CGM)
    easy_mpc.bg_target = max(110.0, easy_mpc.basal_bg - 10.0)

    print('\nRunning EASY-scenario evaluation (deterministic meals)...')
    easy_data = run_episode(easy_mpc, easy_env, easy_args, easy_pump)
    easy_cgm  = easy_data['cgm_trace']
    pct_tir_e  = np.mean([(70 <= g <= 180) for g in easy_cgm]) * 100
    pct_hypo_e = np.mean([g < 70  for g in easy_cgm]) * 100
    pct_hyper_e= np.mean([g > 180 for g in easy_cgm]) * 100
    print(f'Easy  | r={sum(easy_data["rewards"]):.2f} | '
          f'cgm mean={np.mean(easy_cgm):.1f}  min={np.min(easy_cgm):.1f}  max={np.max(easy_cgm):.1f}')
    print(f'       TIR(70-180)={pct_tir_e:.1f}%  hypo={pct_hypo_e:.1f}%  hyper={pct_hyper_e:.1f}%')

    from datetime import datetime
    import sys
    cmd = 'python ' + ' '.join(sys.argv)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('bash_results/result_all', exist_ok=True)

    fig = plot_results({'cgm': cgm, 'insulin': data['ins_trace'], 'infos': data['infos']},
                       args.patient_id, cmd=cmd + '  [MPC]')
    fname = f'bash_results/result_all/mpc_results_{ts}_p{args.patient_id}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {fname}')

    fig_easy = plot_results({'cgm': easy_cgm, 'insulin': easy_data['ins_trace'],
                             'infos': easy_data['infos']},
                            args.patient_id, cmd=cmd + '  [MPC EASY]')
    fname_easy = f'bash_results/result_all/mpc_results_{ts}_p{args.patient_id}_easy.png'
    fig_easy.savefig(fname_easy, dpi=150, bbox_inches='tight')
    print(f'Easy plot saved to {fname_easy}')


if __name__ == '__main__':
    main()
