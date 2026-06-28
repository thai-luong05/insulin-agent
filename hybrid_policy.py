# HyCPAP-style hybrid: blend MPC and DRL doses via Gaussian product (Wu et al. 2024). No DRL = pure MPC.
import os
import sys
import random
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'simulation', 'G2P2C'))


def gaussian_product(mu1, sig1, mu2, sig2):
    # multiply two Gaussians -> (mu, sigma); the tighter one wins
    v1, v2 = sig1 ** 2, sig2 ** 2
    denom  = v1 + v2 + 1e-12
    mu     = (mu1 * v2 + mu2 * v1) / denom
    var    = (v1 * v2) / denom
    return mu, var ** 0.5


def ensemble_gaussian(mus, sigs):
    # collapse an ensemble of Gaussians into one; disagreement inflates sigma -> blend leans on MPC
    mus  = np.asarray(mus, dtype=float)
    sigs = np.asarray(sigs, dtype=float)
    mu   = float(mus.mean())
    var  = float((sigs ** 2).mean() + mus.var())
    return mu, var ** 0.5


def drl_dose_gaussian(policy, state_np, n_samples=16):
    # sample the policy a few times to get its dose mean and spread (U/min)
    s = torch.from_numpy(state_np).unsqueeze(0)
    doses = []
    with torch.no_grad():
        for _ in range(n_samples):
            action, _, _, _ = policy.act_stochastic(s)
            doses.append(float(action.item()))
    doses = np.asarray(doses)
    return float(doses.mean()), float(doses.std() + 1e-4)


def run_hybrid_episode(patient_id, drl_policy=None, mpc_sigma_frac=0.15,
                       overrides=None, seed=0, verbose=True, model='bergman'):
    # 24h rollout: MPC sets the dose, DRL nudges it via the blend, meal bolus is a floor; smaller mpc_sigma_frac = trust MPC more, drl_policy=None = pure MPC. model in {'bergman','lti'} picks the prediction model.
    from main import (get_clinical_params, get_body_weight, get_patient_env,
                      get_env, custom_reward, Options, Pump, get_basal,
                      MPC_MODELS, default_max_bolus_mult, calibrate, PREBOLUS_STEPS)
    from utils.statespace import StateSpace
    from freestyle_rl.state_builder import build_state, iob_decay
    import gym.envs.registration as _reg

    overrides = overrides or {}
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = Options().parse()
    finally:
        sys.argv = saved_argv
    args.patient_id = patient_id
    args.seed = seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    patients, env_ids = get_patient_env()
    patient = patients[patient_id]
    if 10 <= patient_id < 20:
        bw_kg = get_body_weight(patient)
        prop  = [30, 15, 45, 15, 45, 15]
        scale = 3.5 * bw_kg / sum(prop)
        args.meal_amount   = [max(5, round(p * scale)) for p in prop]
        args.meal_variance = [max(1, round(a / 6))   for a in args.meal_amount]

    _reg.registry.env_specs.pop(env_ids[patient_id], None)
    env = get_env(args, patient_name=patient, env_id=env_ids[patient_id],
                  custom_reward=custom_reward, seed=seed)

    basal = get_basal(patient)
    clin  = get_clinical_params(patient)
    bw    = get_body_weight(patient)
    isf   = (1700.0 / clin['tdd']) * float(overrides.get('isf_mult', 1.0))
    true_cr  = clin['icr'] * float(overrides.get('cr_mult', 1.0))
    horizon  = int(overrides.get('horizon', 24))
    prebolus = int(overrides.get('prebolus_steps', 5))
    mb_mult  = float(overrides.get('max_bolus_mult', default_max_bolus_mult(patient_id)))
    R_val    = 10.0 ** float(overrides.get('R_log10', -2.0))

    ss = StateSpace(args)
    pump = Pump(args, patient_name=patient)
    _, cgm_now = calibrate(env, ss, args, pump)
    bg0 = cgm_now

    MPC = MPC_MODELS[model]
    mpc = MPC(basal_rate=basal, isf=isf, basal_bg=bg0, body_weight=bw,
              max_bolus_multiplier=mb_mult, R=R_val,
              beta=float(overrides.get('beta', 3.0)), horizon=horizon)
    mpc.p2 = float(overrides.get('p2', 0.025))
    mpc.basal_bg  = bg0
    mpc.bg_target = max(110.0, bg0 + float(overrides.get('correction_target_offset', -10.0)))
    mpc_sigma = max(mpc_sigma_frac * mpc.u_max, 1e-3)

    try:
        sched = env.env.scenario.scenario.get('meal', {'time': [], 'amount': []})
        meal_steps   = [t / 5.0 - args.calibration for t in sched['time']]
        meal_amounts = [float(a) for a in sched['amount']]
    except Exception:
        meal_steps, meal_amounts = [], []

    #one policy or a list (ensemble) — both work
    if drl_policy is None:
        drl_members = []
    elif isinstance(drl_policy, (list, tuple)):
        drl_members = list(drl_policy)
    else:
        drl_members = [drl_policy]

    X_now = I_now = 0.0
    bolus_queue = 0.0
    bolused = set()
    iob = 0.0
    prev_cgm = cgm_now
    cgm_trace, ins_trace, infos = [], [], []

    for step in range(288):
        #queue the meal bolus ahead of the meal
        for idx, (ms, ma) in enumerate(zip(meal_steps, meal_amounts)):
            if idx not in bolused and step <= ms <= step + prebolus:
                bolus_queue += ma / float(true_cr)
                bolused.add(idx)

        mpc_corr = mpc.compute_insulin(G_now=cgm_now, X_now=X_now, I_now=I_now,
                                       G_target=mpc.bg_target, meal_carbs=0.0)
        if bolus_queue > 0:
            from_queue  = min(bolus_queue / 5.0, mpc.u_max)
            bolus_queue = max(0.0, bolus_queue - from_queue * 5.0)
        else:
            from_queue = 0.0
        mpc_full = min(mpc.u_max, mpc_corr + from_queue)

        #blend in the DRL, but never dose below the meal bolus
        if drl_members:
            s_drl = build_state(basal, bw, clin['isf'], clin['icr'], clin['tdd'],
                                cgm_now=cgm_now, cgm_prev=prev_cgm, iob_proxy=iob,
                                meal_carbs_announced=0.0, mins_to_next_meal=240.0)
            mus, sigs = zip(*(drl_dose_gaussian(m, s_drl) for m in drl_members))
            drl_mu, drl_sig = ensemble_gaussian(mus, sigs)
            blended, _ = gaussian_product(mpc_full, mpc_sigma, drl_mu, drl_sig)
            pump_act = float(min(mpc.u_max, max(from_queue, blended)))
        else:
            pump_act = mpc_full

        s, _, _, info = env.step(pump_act)
        prev_cgm = cgm_now
        cgm_now  = s.CGM
        iob = iob_decay(iob, pump_act * 5.0)

        dI = -mpc.n * I_now + 1000.0 * (pump_act - mpc.u_basal) / mpc.V_I
        dX = -mpc.p2 * X_now + mpc.p3 * I_now
        I_now += mpc.dt * dI
        X_now += mpc.dt * dX

        cgm_trace.append(cgm_now)
        ins_trace.append(pump_act)
        infos.append(info)

    cgm = np.array(cgm_trace)
    out = {
        'tir':   float(np.mean((70 <= cgm) & (cgm <= 180)) * 100),
        'hypo':  float(np.mean(cgm < 70) * 100),
        'hyper': float(np.mean(cgm > 180) * 100),
        'cgm_mean': float(cgm.mean()),
        'cgm_min': float(cgm.min()), 'cgm_max': float(cgm.max()),
        'cgm_trace': cgm_trace, 'ins_trace': ins_trace, 'infos': infos,
        'bg_target': mpc.bg_target,
    }
    if verbose:
        tag = 'MPC+DRL blend' if drl_members else 'MPC only'
        print(f'p{patient_id} [{tag}]: TIR={out["tir"]:.1f}%  hypo={out["hypo"]:.1f}%  '
              f'hyper={out["hyper"]:.1f}%  (min {out["cgm_min"]:.0f}, max {out["cgm_max"]:.0f})')
    return out


def _selftest():
    mu, sig = gaussian_product(0.10, 0.02, 0.30, 0.20)
    print(f'product(confident 0.10±0.02, vague 0.30±0.20) -> {mu:.4f}±{sig:.4f} '
          f'(should sit near 0.10)')
    mu, sig = ensemble_gaussian([0.1, 0.5, 0.9], [0.05, 0.05, 0.05])
    print(f'ensemble of disagreeing means -> {mu:.4f}±{sig:.4f} (sigma inflated by spread)')


def _load_ensemble(paths):
    # load TinyPolicy .pt files as a DRL ensemble (one file = ensemble of one)
    from freestyle_rl.policy import TinyPolicy
    members = []
    for p in paths:
        if os.path.exists(p):
            pol = TinyPolicy()
            pol.load_state_dict(torch.load(p, map_location='cpu'))
            pol.eval()
            members.append(pol)
            print(f'DRL refiner loaded: {p}')
        else:
            print(f'  (skipped missing DRL: {p})')
    return members


def main():
    import argparse
    from datetime import datetime
    from main import plot_results

    ap = argparse.ArgumentParser(description='HyCPAP MPC+DRL hybrid — same flow/output as main.py.')
    ap.add_argument('--patient_id', type=int, default=6)
    ap.add_argument('--drl', nargs='+', default=['freestyle_rl/ppo_p6.pt'],
                    help='one or more TinyPolicy .pt DRL refiners (ensemble)')
    ap.add_argument('--no-drl', action='store_true', help='pure MPC, no DRL blend')
    ap.add_argument('--model', choices=['bergman', 'lti'], default='bergman',
                    help='prediction model for the MPC (lti = linearized Bergman)')
    ap.add_argument('--mpc_sigma_frac', type=float, default=0.15,
                    help='MPC prior width as fraction of u_max (smaller = trust MPC more)')
    ap.add_argument('--no-tune', action='store_true',
                    help='skip per-patient MPC tuning (use defaults)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--selftest', action='store_true', help='just run the blend-math self-test')
    a = ap.parse_args()

    if a.selftest:
        _selftest(); return

    #tune the MPC knobs for this patient (or use defaults)
    if a.no_tune:
        from meta_rl import default_overrides
        overrides = default_overrides(a.patient_id)
    else:
        from meta_rl import tune_patient
        overrides = tune_patient(a.patient_id, seed=a.seed, model=a.model)

    drl = [] if a.no_drl else _load_ensemble(a.drl)

    #run MPC alone, then the blend
    print(f'\nRunning MPC-only baseline ({a.model})...')
    base = run_hybrid_episode(a.patient_id, drl_policy=None,
                              overrides=overrides, seed=a.seed, model=a.model)
    metrics = base
    if drl:
        print('Running HyCPAP MPC+DRL blend...')
        metrics = run_hybrid_episode(a.patient_id, drl_policy=drl,
                                     overrides=overrides,
                                     mpc_sigma_frac=a.mpc_sigma_frac, seed=a.seed,
                                     model=a.model)

    cgm = metrics['cgm_trace']
    print(f'\nEval  | cgm mean={metrics["cgm_mean"]:.1f}  '
          f'min={metrics["cgm_min"]:.1f}  max={metrics["cgm_max"]:.1f}')
    print(f'Time in range (70-180): {metrics["tir"]:.1f}%  '
          f'Hypo (<70): {metrics["hypo"]:.1f}%  Hyper (>180): {metrics["hyper"]:.1f}%')
    
    #get total insulin delivered after evaluation
    total_insulin = sum(metrics['ins_trace']) * 5.0 #U/min for 5-min steps so multiply by 5 and add all to get 1 full day
    print(f'Total insulin delivered in 1 day: {total_insulin:.2f} U/day')

    #plot + save like main.py
    tag = 'mpc' if not drl else 'hybrid'
    cmd = 'python ' + ' '.join(sys.argv)
    ts  = os.environ.get('RUN_TS') or datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = f'bash_results/result_all/results_at_{ts}'
    os.makedirs(outdir, exist_ok=True)
    fig = plot_results({'cgm': cgm, 'insulin': metrics['ins_trace'],
                        'infos': metrics['infos'], 'bg_target': metrics['bg_target']},
                       a.patient_id, cmd=cmd + f'  [{tag.upper()}]')
    fname = f'{outdir}/{tag}_results_{ts}_p{a.patient_id}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {fname}')


if __name__ == '__main__':
    main()
