# Pure Bergman-MPC glucose regulation using simglucose's ground-truth per-patient clinical params (CF=ISF, CR=ICR, TDI=TDD) from Quest.csv instead of the 1700/450 population rules.
import os, sys, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decouple import config
from bergman_controller import BergmanMPC
from lti_controller import LTIMPC

#controller registry: pick the prediction model with the --model flag / model= kwarg
MPC_MODELS = {'bergman': BergmanMPC, 'lti': LTIMPC}


#resolve simglucose's bundled params from the installed package (portable across venv name / OS)
import simglucose
_SIMGLUCOSE_PARAMS = os.path.join(os.path.dirname(simglucose.__file__), 'params')

#simglucose ships per-patient clinical params (CF=ISF, CR=ICR, TDI=TDD)
_QUEST_CSV = os.path.join(_SIMGLUCOSE_PARAMS, 'Quest.csv')
_quest_df = None

def get_clinical_params(patient_name):
    """Return (isf, icr, tdd, age) for a patient from simglucose's Quest.csv."""
    global _quest_df
    if _quest_df is None:
        _quest_df = pd.read_csv(_QUEST_CSV).set_index('Name')
    row = _quest_df.loc[patient_name]
    return {'isf': float(row['CF']), 'icr': float(row['CR']),
            'tdd': float(row['TDI']), 'age': int(row['Age'])}


_VPATIENT_CSV = os.path.join(_SIMGLUCOSE_PARAMS, 'vpatient_params.csv')
_vpatient_df = None

def get_body_weight(patient_name):
    """Patient body weight in kg from simglucose's vpatient_params.csv."""
    global _vpatient_df
    if _vpatient_df is None:
        _vpatient_df = pd.read_csv(_VPATIENT_CSV).set_index('Name')
    return float(_vpatient_df.loc[patient_name, 'BW'])

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


PREBOLUS_STEPS         = 3      #15 min meal pre-announce
PUMP_PHYSICAL_MAX      = 0.6    #U/min absolute pump cap (matches env)


def run_episode(mpc, env, args, pump, true_cr, prebolus_steps=PREBOLUS_STEPS):
    # 24h closed-loop rollout (288 x 5 min): meals covered open-loop by a (carbs/true_cr) bolus queued prebolus_steps ahead and delivered at the pump cap, while the Bergman MPC runs as pure correction (meal_carbs=0)
    ss = StateSpace(args)
    _, current_cgm = calibrate(env, ss, args, pump)

    #bergman insulin states, tracked externally across the rollout
    X_now = 0.0     #insulin action (1/min)
    I_now = 0.0     #plasma insulin deviation from basal (mU/L)

    #step-indexed meal schedule for the feedforward bolus
    try:
        sched = env.env.scenario.scenario.get('meal', {'time': [], 'amount': []})
        meal_steps   = [t / 5.0 - args.calibration for t in sched['time']]
        meal_amounts = [float(a) for a in sched['amount']]
    except Exception:
        meal_steps, meal_amounts = [], []

    rewards, cgm_trace, ins_trace, infos = [], [], [], []

    bolus_queue = 0.0    #units of insulin queued for delivery
    bolused     = set()

    for step in range(288):
        #queue the meal bolus once, when the meal enters the prebolus window
        for idx, (ms, ma) in enumerate(zip(meal_steps, meal_amounts)):
            if idx not in bolused and step <= ms <= step + prebolus_steps:
                bolus_queue += ma / float(true_cr)
                bolused.add(idx)

        #MPC handles correction only; meals are covered by the bolus queue
        mpc_rate = mpc.compute_insulin(
            G_now      = current_cgm,
            X_now      = X_now,
            I_now      = I_now,
            G_target   = mpc.bg_target,
            meal_carbs = 0.0,
        )

        #drain the bolus queue at the pump cap, on top of the correction rate
        if bolus_queue > 0:
            from_queue  = min(bolus_queue / 5.0, mpc.u_max)
            bolus_queue = max(0.0, bolus_queue - from_queue * 5.0)
            pump_act    = min(mpc.u_max, mpc_rate + from_queue)
        else:
            pump_act = mpc_rate

        s, reward, _, info = env.step(pump_act)
        current_cgm = s.CGM

        ss.update(
            cgm=s.CGM, ins=pump_act, meal=info['remaining_time'],
            hour=step + 1, meal_type=info['meal_type'], carbs=info['future_carb'],
        )

        #advance the external Bergman insulin states for next step's prediction
        dI = -mpc.n * I_now + 1000.0 * (pump_act - mpc.u_basal) / mpc.V_I
        dX = -mpc.p2 * X_now + mpc.p3 * I_now
        I_now = I_now + mpc.dt * dI
        X_now = X_now + mpc.dt * dX

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
    ax.axhline(50,  color='red', lw=1, ls=':', alpha=0.6, label='Severe hypo (50)')
    ax.axhline(300, color='red', lw=1, ls=':', alpha=0.6, label='Severe hyper (300)')
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


def default_max_bolus_mult(patient_id):
    if patient_id >= 20: return 30.0
    if patient_id < 10:  return 25.0
    return 15.0


def evaluate_patient(patient_id, overrides=None, seed=0, verbose=True, model='bergman'):
    # One 24h hybrid rollout (open-loop carb-ratio meal bolus + correction MPC) with optional RL overrides; returns TIR/hypo/hyper metrics + traces. Override keys: isf_mult, correction_target_offset, max_bolus_mult, R_log10, beta, p2, horizon, prebolus_steps, cr_mult. model in {'bergman','lti'} picks the prediction model.
    overrides = overrides or {}

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
    patient = patients[patient_id]

    # children: scale meals to ~3.5 g/kg/day (proportions [30,15,45,15,45,15] across meals+snacks)
    if 10 <= patient_id < 20:
        bw_kg          = get_body_weight(patient)
        target_total_g = 3.5 * bw_kg
        proportions    = [30, 15, 45, 15, 45, 15]
        scale          = target_total_g / sum(proportions)
        args.meal_amount   = [max(5, round(p * scale)) for p in proportions]
        args.meal_variance = [max(1, round(a / 6))   for a in args.meal_amount]

    import gym.envs.registration as _reg
    _reg.registry.env_specs.pop(env_ids[patient_id], None)
    env = get_env(args, patient_name=patient, env_id=env_ids[patient_id],
                  custom_reward=custom_reward, seed=seed)

    std_basal = get_basal(patient)
    clin      = get_clinical_params(patient)
    tdd       = clin['tdd']
    isf       = (1700.0 / tdd) * float(overrides.get('isf_mult', 1.0))
    cr_mult   = float(overrides.get('cr_mult', 1.0))
    true_cr   = clin['icr'] * cr_mult              #ground-truth CR for feedforward bolus
    icr       = 450.0  / tdd                        #reported for the metrics dict only
    mb_mult   = float(overrides.get('max_bolus_mult', default_max_bolus_mult(patient_id)))
    R_val     = 10.0 ** float(overrides.get('R_log10', -2.0))
    horizon   = int(overrides.get('horizon', 24))
    prebolus  = int(overrides.get('prebolus_steps', 5))
    body_kg   = get_body_weight(patient)
    basal_bg0 = float(env.reset().CGM)

    beta_val = float(overrides.get('beta', 3.0))
    p2_val   = float(overrides.get('p2', 0.025))
    MPC = MPC_MODELS[model]
    mpc = MPC(basal_rate=std_basal, isf=isf,
              basal_bg=basal_bg0, body_weight=body_kg,
              max_bolus_multiplier=mb_mult, R=R_val,
              beta=beta_val, horizon=horizon)
    mpc.p2 = p2_val  #override hardcoded X decay rate
    mpc.basal_bg  = basal_bg0
    offset        = float(overrides.get('correction_target_offset', -10.0))
    mpc.bg_target = max(110.0, mpc.basal_bg + offset)

    pump = Pump(args, patient_name=patient)
    data = run_episode(mpc, env, args, pump, true_cr=true_cr, prebolus_steps=prebolus)
    cgm  = np.array(data['cgm_trace'])

    metrics = {
        'patient_id':   patient_id,
        'patient_name': patient,
        'tir':   float(np.mean((70 <= cgm) & (cgm <= 180)) * 100),
        'hypo':  float(np.mean(cgm < 70) * 100),
        'hyper': float(np.mean(cgm > 180) * 100),
        'cgm_mean': float(np.mean(cgm)),
        'cgm_min':  float(np.min(cgm)),
        'cgm_max':  float(np.max(cgm)),
        'reward_sum': float(sum(data['rewards'])),
        'clinical': {'isf': isf, 'icr': icr, 'tdd': tdd,
                     'p3': mpc.p3, 'bg_target': mpc.bg_target,
                     'body_kg': body_kg},
        'cgm_trace': data['cgm_trace'],
        'ins_trace': data['ins_trace'],
        'infos':     data['infos'],
        'bg_target': mpc.bg_target,
    }
    if verbose:
        print(f'p{patient_id} {patient}: TIR={metrics["tir"]:.1f}%  '
              f'hypo={metrics["hypo"]:.1f}%  hyper={metrics["hyper"]:.1f}%  '
              f'(ISF={isf:.1f}  BW={body_kg:.0f}kg  p3={mpc.p3:.2e}  bgt={mpc.bg_target:.0f})')
    return metrics


def main():
    args = Options().parse()

    #train fresh every call (no caching). adjust n_samples in meta_rl.py.
    from meta_rl import tune_patient
    overrides = tune_patient(args.patient_id, seed=args.seed)

    metrics = evaluate_patient(args.patient_id, overrides=overrides,
                               seed=args.seed, verbose=True)
    cgm = metrics['cgm_trace']
    print(f'Eval  | r={metrics["reward_sum"]:.2f} | '
          f'cgm mean={metrics["cgm_mean"]:.1f}  min={metrics["cgm_min"]:.1f}  max={metrics["cgm_max"]:.1f}')

    from datetime import datetime
    cmd = 'python ' + ' '.join(sys.argv)
    ts  = os.environ.get('RUN_TS') or datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(f'bash_results/result_all/results_at_{ts}', exist_ok=True)

    fig = plot_results({'cgm': cgm, 'insulin': metrics['ins_trace'],
                        'infos': metrics['infos'], 'bg_target': metrics['bg_target']},
                       args.patient_id, cmd=cmd + '  [MPC]')
    fname = f'bash_results/result_all/results_at_{ts}/mpc_results_{ts}_p{args.patient_id}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {fname}')


if __name__ == '__main__':
    main()
