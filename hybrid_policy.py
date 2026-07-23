# HyCPAP-style hybrid: blend MPC and DRL doses via Gaussian product (Wu et al. 2024). No DRL = pure MPC.
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ============ rl supervisor (lti + rl): the policy tunes the mpc knobs, not the dose ============
# rl never touches insulin. it outputs bounded mpc knobs (see SUP_KNOBS); the mpc solves the dose
# with its own hypo constraint, so the supervisor cannot command an unsafe dose. p1/p2 stay fixed
# (physio kinetics, not knobs). offline grad-clipped td3+bc.
KNOB_LO, KNOB_HI = -1.0, 1.0     # normalized action space; knob_to_phys maps to real mpc knobs
# rl tunes 5 mpc knobs (one mult/offset each); p1,p2 fixed; neutral [1,1,1,1,0] = pure mpc
SUP_KNOBS = [
    ('isf_mult',       0.5,  1.5),   # sensitivity gain (isf -> p3); <1 => doses more
    ('horizon_mult',   0.5,  2.0),   # prediction horizon
    ('R_mult',         0.3,  3.0),   # input penalty; high => conservative
    ('R_dn_mult',      0.3,  3.0),   # suspend-below-basal cost (lti only)
    ('target_offset', -15.0, 15.0),  # bg-target shift (mg/dl)
]
SUP_ADIM  = len(SUP_KNOBS)
_SUP_LO   = np.array([k[1] for k in SUP_KNOBS], dtype=np.float32)
_SUP_HI   = np.array([k[2] for k in SUP_KNOBS], dtype=np.float32)
_SUP_NEUT = np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)   # physical neutral = pure mpc


def knob_to_phys(a):
    # normalized action [-1,1] -> physical knob values
    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
    return _SUP_LO + 0.5 * (a + 1.0) * (_SUP_HI - _SUP_LO)


_SUP_NEUT_A = (2.0 * (_SUP_NEUT - _SUP_LO) / (_SUP_HI - _SUP_LO) - 1.0).astype(np.float32)   # action that gives pure mpc


def _sup_kovatchev(bg):
    bg = max(bg, 1.0)
    f = 1.509 * (np.log(bg) ** 1.084 - 5.381)
    return -10.0 * f * f


class SupActor(nn.Module):
    def __init__(self, sdim=10, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, SUP_ADIM))

    def forward(self, s):
        return torch.tanh(self.net(s))          # normalized knob vector in [-1,1]^ADIM


class SupCritic(nn.Module):
    def __init__(self, sdim=10, hid=128):
        super().__init__()
        def q():
            return nn.Sequential(nn.Linear(sdim + SUP_ADIM, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.q1, self.q2 = q(), q()

    def forward(self, s, a):
        x = torch.cat([s, a], -1); return self.q1(x), self.q2(x)

    def Q1(self, s, a):
        return self.q1(torch.cat([s, a], -1))


def _sup_setup(patient_id, overrides, seed, model):
    from main import (get_clinical_params, get_body_weight, get_patient_env, get_env,
                      custom_reward, Options, Pump, get_basal, MPC_MODELS,
                      default_max_bolus_mult, calibrate)
    from utils.statespace import StateSpace
    import gym.envs.registration as _reg
    saved = sys.argv; sys.argv = [saved[0]]
    try:    args = Options().parse()
    finally: sys.argv = saved
    args.patient_id = patient_id; args.seed = seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    patients, env_ids = get_patient_env(); patient = patients[patient_id]
    if 10 <= patient_id < 20:
        bw_kg = get_body_weight(patient); prop = [30, 15, 45, 15, 45, 15]
        scale = 3.5 * bw_kg / sum(prop)
        args.meal_amount   = [max(5, round(p * scale)) for p in prop]
        args.meal_variance = [max(1, round(a / 6))   for a in args.meal_amount]
    _reg.registry.env_specs.pop(env_ids[patient_id], None)
    env = get_env(args, patient_name=patient, env_id=env_ids[patient_id],
                  custom_reward=custom_reward, seed=seed)
    basal = get_basal(patient); clin = get_clinical_params(patient); bw = get_body_weight(patient)
    isf = (1700.0 / clin['tdd']) * overrides['isf_mult']; true_cr = clin['icr'] * overrides['cr_mult']
    horizon = int(overrides['horizon']); prebolus = int(overrides['prebolus_steps'])
    mb_mult = float(overrides.get('max_bolus_mult', default_max_bolus_mult(patient_id)))
    R_val = 10.0 ** overrides['R_log10']
    ss = StateSpace(args); pump = Pump(args, patient_name=patient)
    _, cgm0 = calibrate(env, ss, args, pump)
    mpc = MPC_MODELS[model](basal_rate=basal, isf=isf, basal_bg=cgm0, body_weight=bw,
                            max_bolus_multiplier=mb_mult, R=R_val, beta=overrides['beta'], horizon=horizon)
    mpc.p2 = overrides['p2']; mpc.basal_bg = cgm0                     # p2 FIXED here, never modulated
    mpc.bg_target = max(110.0, cgm0 + overrides['correction_target_offset'])
    try:
        sched = env.env.scenario.scenario.get('meal', {'time': [], 'amount': []})
        meal_steps = [t / 5.0 - args.calibration for t in sched['time']]
        meal_amounts = [float(a) for a in sched['amount']]
    except Exception:
        meal_steps, meal_amounts = [], []
    return env, mpc, basal, bw, clin, true_cr, prebolus, meal_steps, meal_amounts, cgm0


def run_supervisor_episode(patient_id, knob_net=None, seed=0, overrides=None, rng=None, model='lti'):
    # knob_net set -> deploy (returns metrics); knob_net None -> collect random-knob data (returns rows)
    from freestyle_rl.state_builder import build_state, iob_decay
    from meta_rl import default_overrides
    overrides = overrides or default_overrides(patient_id)
    env, mpc, basal, bw, clin, true_cr, prebolus, meal_steps, meal_amounts, cgm0 = \
        _sup_setup(patient_id, overrides, seed, model)
    base_isf     = mpc.isf                          # tuned base; knobs scale these
    base_horizon = mpc.horizon
    base_R       = mpc.R
    base_R_dn    = getattr(mpc, 'R_dn_frac', 0.1)
    base_target  = mpc.bg_target
    X = I = 0.0; bolus_q = 0.0; bolused = set(); iob = 0.0
    cgm_now = cgm0; prev_cgm = cgm0
    rows, prev, failed = [], None, False
    cgm_tr, ins_tr, infos = [], [], []
    for step in range(288):
        for idx, (ms, ma) in enumerate(zip(meal_steps, meal_amounts)):
            if idx not in bolused and step <= ms <= step + prebolus:
                bolus_q += ma / float(true_cr); bolused.add(idx)
        s = build_state(basal, bw, clin['isf'], clin['icr'], clin['tdd'],
                        cgm_now=cgm_now, cgm_prev=prev_cgm, iob_proxy=iob,
                        meal_carbs_announced=0.0, mins_to_next_meal=240.0)
        if knob_net is not None:
            with torch.no_grad():
                a = knob_net(torch.from_numpy(s).unsqueeze(0)).squeeze(0).numpy()
        else:
            a = rng.uniform(-1.0, 1.0, SUP_ADIM).astype(np.float32)
        phys = knob_to_phys(a)                      # [isf_mult, horizon_mult, r_mult, r_dn_mult, target_offset]
        # rl tunes isf/horizon/cost/target; p1,p2,n fixed
        mpc.isf     = base_isf * float(phys[0])
        mpc.p3      = mpc.isf * mpc.p2 * mpc.n * mpc.V_I / (1000.0 * mpc.Gb)
        mpc.horizon = max(4, int(round(base_horizon * float(phys[1]))))
        mpc.R       = base_R * float(phys[2])
        if hasattr(mpc, 'R_dn_frac'):
            mpc.R_dn_frac = base_R_dn * float(phys[3])
        mpc.bg_target = base_target + float(phys[4])
        mpc_corr = mpc.compute_insulin(G_now=cgm_now, X_now=X, I_now=I, G_target=mpc.bg_target)
        from_queue = min(bolus_q / 5.0, mpc.u_max) if bolus_q > 0 else 0.0
        bolus_q = max(0.0, bolus_q - from_queue * 5.0)
        delivered = float(min(mpc.u_max, mpc_corr + from_queue))
        env_s, _, _, info = env.step(delivered)
        prev_cgm = cgm_now; cgm_now = env_s.CGM
        iob = iob_decay(iob, delivered * 5.0)
        I += mpc.dt * (-mpc.n * I + 1000.0 * (delivered - mpc.u_basal) / mpc.V_I)
        X += mpc.dt * (-mpc.p2 * X + mpc.p3 * I)
        cgm_tr.append(cgm_now); ins_tr.append(delivered); infos.append(info)
        if knob_net is None:
            r = _sup_kovatchev(cgm_now)
            if prev is not None:
                rows.append((prev[0], prev[1], prev[2], s, 0.0))
            if cgm_now < 40.0 or cgm_now > 450.0:                    # medical-emergency terminal
                rows.append((s, a.astype(np.float32), np.float32(r - 100.0), s, 1.0)); prev = None; break
            prev = (s, a.astype(np.float32), np.float32(r))
        elif cgm_now <= 40.0 or cgm_now >= 600.0:      # eval terminal: mark fail, stop
            failed = True; break
    if knob_net is None:
        if prev is not None:
            rows.append((prev[0], prev[1], prev[2], s, 1.0))
        return rows
    c = np.array(cgm_tr)
    return {'tir': float(np.mean((70 <= c) & (c <= 180)) * 100), 'hypo': float(np.mean(c < 70) * 100),
            'sev_hypo': float(np.mean(c < 54) * 100), 'hyper': float(np.mean(c > 180) * 100),
            'cgm_min': float(c.min()), 'cgm_max': float(c.max()), 'cgm_mean': float(c.mean()),
            'cgm_trace': cgm_tr, 'ins_trace': ins_tr, 'infos': infos, 'bg_target': mpc.bg_target,
            'failed': failed, 'ep_len': len(cgm_tr)}


def train_supervisor(patient_id, out, overrides, seeds=40, steps=40000, model='lti'):
    import copy
    rng = np.random.default_rng(patient_id)
    S, A, R, S2, D = [], [], [], [], []
    for si in range(seeds):
        for s, a, r, s2, d in run_supervisor_episode(patient_id, knob_net=None, seed=si,
                                                      overrides=overrides, rng=rng, model=model):
            S.append(s); A.append(a); R.append(r); S2.append(s2); D.append(d)
    S = torch.tensor(np.asarray(S, np.float32)); A = torch.tensor(np.asarray(A, np.float32))   # (n, adim)
    R = torch.tensor(np.asarray(R, np.float32)).unsqueeze(1)
    S2 = torch.tensor(np.asarray(S2, np.float32)); D = torch.tensor(np.asarray(D, np.float32)).unsqueeze(1)
    mean = S.mean(0, keepdim=True); std = S.std(0, keepdim=True) + 1e-3
    Sn, S2n = (S - mean) / std, (S2 - mean) / std
    N, sdim = S.shape[0], S.shape[1]
    print(f'[supervisor p{patient_id}] {N} transitions, {SUP_ADIM} knobs (normalized [-1,1])', flush=True)
    actor = SupActor(sdim); actor_t = copy.deepcopy(actor)
    critic = SupCritic(sdim); critic_t = copy.deepcopy(critic)
    oa = torch.optim.Adam(actor.parameters(), lr=3e-4); oc = torch.optim.Adam(critic.parameters(), lr=3e-4)
    for t in range(steps):
        idx = torch.randint(0, N, (256,))
        s, a, r, s2, dn = Sn[idx], A[idx], R[idx], S2n[idx], D[idx]
        with torch.no_grad():
            noise = (torch.randn_like(a) * 0.1).clamp(-0.25, 0.25)
            a2 = (actor_t(s2) + noise).clamp(-1.0, 1.0)
            q1t, q2t = critic_t(s2, a2)
            y = r + 0.99 * (1.0 - dn) * torch.min(q1t, q2t)
        q1, q2 = critic(s, a)
        closs = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        oc.zero_grad(); closs.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 1.0); oc.step()
        if t % 2 == 0:
            pi = actor(s); q = critic.Q1(s, pi)
            lmbda = 2.5 / (q.abs().mean().detach() + 1e-6)
            aloss = -lmbda * q.mean() + F.mse_loss(pi, a)
            oa.zero_grad(); aloss.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 1.0); oa.step()
            for p, pt in zip(critic.parameters(), critic_t.parameters()):
                pt.data.mul_(0.995); pt.data.add_(0.005 * p.data)
            for p, pt in zip(actor.parameters(), actor_t.parameters()):
                pt.data.mul_(0.995); pt.data.add_(0.005 * p.data)
        if (t + 1) % 10000 == 0:
            print(f'  step {t+1}: critic={closs.item():.2f}', flush=True)
    torch.save({'actor': actor.state_dict(), 'mean': mean, 'std': std, 'sdim': sdim, 'adim': SUP_ADIM}, out)
    print(f'[supervisor] saved -> {out}', flush=True)


def _append_summary(outdir, line, header=None):
    # append one patient's line to a shared summary.txt; run.sh exports one run_ts
    # so all 30 patients land in the same file
    path = os.path.join(outdir, 'summary.txt')
    new = not os.path.exists(path)
    with open(path, 'a', encoding='utf-8') as f:
        if new and header:
            f.write(header + '\n')
        f.write(line + '\n')
    return path


def _risk_index(cgm):
    # kovatchev lbgi/hbgi/ri, same as g2p2c new_risk_index
    bg = np.asarray(cgm, dtype=float).copy(); bg[bg < 1] = 1
    f = 1.509 * (np.log(bg) ** 1.084 - 5.381)
    rl = 10 * f[f < 0] ** 2; rh = 10 * f[f > 0] ** 2
    lbgi = float(np.nan_to_num(np.mean(rl))) if rl.size else 0.0
    hbgi = float(np.nan_to_num(np.mean(rh))) if rh.size else 0.0
    return lbgi, hbgi, lbgi + hbgi


def graded_metrics(cgm):
    # graded consensus bands: <=54 S_hypo, 54-70 hypo, 70-180 normo, 180-250 hyper, >250 S_hyper
    c = np.asarray(cgm, dtype=float)
    lbgi, hbgi, ri = _risk_index(c)
    return {'normo':   float(np.mean((c > 70) & (c <= 180)) * 100),
            'hypo':    float(np.mean((c > 54) & (c <= 70)) * 100),
            'hyper':   float(np.mean((c > 180) & (c <= 250)) * 100),
            'S_hypo':  float(np.mean(c <= 54) * 100),
            'S_hyper': float(np.mean(c > 250) * 100),
            'LBGI': lbgi, 'HBGI': hbgi, 'RI': ri}


def render_table_png(csv_path, out_png, title=None, highlight=()):
    # csv -> table image
    import csv as _csv
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = list(_csv.reader(open(csv_path)))
    header, body = rows[0], rows[1:]
    ncol, nrow = len(header), len(body)
    fig, ax = plt.subplots(figsize=(min(2 + ncol * 1.15, 22), 1.1 + nrow * 0.4))
    ax.axis('off')
    tbl = ax.table(cellText=body, colLabels=header, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    for j in range(ncol):                                    # header row
        c = tbl[0, j]; c.set_facecolor('#40466e'); c.set_text_props(color='w', weight='bold')
    for i, r in enumerate(body, start=1):                    # zebra + highlight our rows
        base = '#f5f7fa' if i % 2 else '#ffffff'
        for j in range(ncol):
            tbl[i, j].set_facecolor('#fff3cd' if r[0] in highlight else base)
    if title:
        ax.set_title(title, fontweight='bold', pad=14)
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close(fig)
    return out_png


def validate_cohort(cohort, mode, model='lti', sup_seeds=40, sup_steps=40000, n_val=500, val_start=1000, tune=True):
    # validate one controller (mpc = neutral knobs; supervisor = trained rl) over a cohort.
    # tune=True uses the per-patient hill-climb base. grand-mean over all patient x seed episodes.
    import csv as _csv
    from datetime import datetime
    from meta_rl import default_overrides, tune_patient
    assert mode in ('mpc', 'supervisor')
    label = 'LTI-MPC+Sup' if mode == 'supervisor' else 'LTI-MPC'
    pools = {'adolescent': range(0, 10), 'child': range(10, 20), 'adult': range(20, 30)}
    ids = list(pools[cohort])
    if val_start < sup_seeds:
        val_start = sup_seeds                        # keep validation seeds unseen
    ts = os.environ.get('RUN_TS') or datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = f'cohort_results/{cohort}_{mode}_{ts}'; os.makedirs(outdir, exist_ok=True)
    logf = open(f'{outdir}/validation_log.txt', 'w', encoding='utf-8')
    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()
    keys = ['normo', 'hypo', 'hyper', 'S_hypo', 'S_hyper', 'LBGI', 'HBGI', 'RI', 'fail']
    meanrows = lambda rr: {k: round(float(np.mean([r[k] for r in rr])), 2) for k in keys}
    log(f'cohort validation: {cohort} | controller {label} ({mode}) | patients {ids} | model {model}')
    if mode == 'supervisor':
        log(f'collect {sup_seeds} seeds (0..{sup_seeds - 1}) | validate {n_val} unseen seeds '
            f'({val_start}..{val_start + n_val - 1})')
    else:
        log(f'validate {n_val} seeds ({val_start}..{val_start + n_val - 1}) | pure MPC, no training')
    log(f'MPC base: {"per-patient hill-climb (tune_patient)" if tune else "population defaults"}')
    log(f'started {datetime.now():%Y-%m-%d %H:%M:%S}')
    log('-' * 92)
    neutral = lambda st: torch.from_numpy(_SUP_NEUT_A).float().unsqueeze(0).expand(st.shape[0], -1)
    per_ep, pp = [], {}                              # flat episodes -> grand mean; per-patient means
    for pid in ids:
        if tune:
            log(f'p{pid}: tuning MPC (hill-climb)...')
            ov = tune_patient(pid, seed=0, model=model)
        else:
            ov = default_overrides(pid)
        if mode == 'supervisor':
            polf = f'freestyle_rl/supervisor_p{pid}.pt'
            log(f'p{pid}: training ({sup_seeds} seeds, {sup_steps} steps)...')
            train_supervisor(pid, polf, ov, seeds=sup_seeds, steps=sup_steps, model=model)
            ck = torch.load(polf, map_location='cpu')
            act = SupActor(ck['sdim']); act.load_state_dict(ck['actor']); act.eval()
            mean, std = ck['mean'], ck['std']
            net = lambda st, _a=act, _m=mean, _s=std: _a((st - _m) / _s)
        else:
            net = neutral
        p_ep = []
        for k in range(n_val):
            m = run_supervisor_episode(pid, knob_net=net, seed=val_start + k, overrides=ov, model=model)
            g = graded_metrics(m['cgm_trace']); g['fail'] = 100.0 if m.get('failed') else 0.0
            p_ep.append(g); per_ep.append(g)
        pp[pid] = meanrows(p_ep)
        log(f'  p{pid} done ({n_val} seeds): normo={pp[pid]["normo"]} S_hypo={pp[pid]["S_hypo"]} '
            f'S_hyper={pp[pid]["S_hyper"]} RI={pp[pid]["RI"]} fail={pp[pid]["fail"]}')
    grand = meanrows(per_ep)
    log('-' * 92)
    log(f'COHORT GRAND MEAN [{label}] (pooled over all patient x seed episodes):')
    log('  ' + '  '.join(f'{k}={grand[k]}' for k in keys))
    log(f'finished {datetime.now():%Y-%m-%d %H:%M:%S}')

    # per-patient tracking spreadsheet
    with open(f'{outdir}/{cohort}_per_patient.csv', 'w', newline='') as f:
        w = _csv.writer(f); w.writerow(['patient', 'controller'] + keys)
        for pid in ids:
            w.writerow([pid, label] + [pp[pid][k] for k in keys])

    # benchmark rows from {cohort}.csv + our row
    cols = ['algo', 'normo', 'hypo', 'hyper', 'S_hypo', 'S_hyper', 'LBGI', 'HBGI', 'RI', 'reward', 'fail']
    keep = []
    if os.path.exists(f'{cohort}.csv'):
        keep = [r for r in _csv.DictReader(open(f'{cohort}.csv')) if r['algo'] != label]
    with open(f'{outdir}/{cohort}_comparison.csv', 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in keep:
            w.writerow({c: r.get(c, '') for c in cols})
        row = {'algo': label, 'reward': ''}; row.update(grand)
        w.writerow({c: row.get(c, '') for c in cols})
    render_table_png(f'{outdir}/{cohort}_comparison.csv', f'{outdir}/{cohort}_comparison.png',
                     title=f'{cohort} — {label} vs G2P2C benchmark', highlight=(label,))
    render_table_png(f'{outdir}/{cohort}_per_patient.csv', f'{outdir}/{cohort}_per_patient.png',
                     title=f'{cohort} — per-patient ({label})')
    log(f'spreadsheet  -> {outdir}/{cohort}_comparison.csv  (+ .png table)')
    log(f'per-patient  -> {outdir}/{cohort}_per_patient.csv  (+ .png table)')
    logf.close()


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
    ap.add_argument('--supervisor', action='store_true',
                    help='LTI+RL: RL sets the MPC aggressiveness knob (ISF mult); the MPC doses safely')
    ap.add_argument('--sup_seeds', type=int, default=40)
    ap.add_argument('--sup_steps', type=int, default=40000)
    ap.add_argument('--cohort', choices=['adolescent', 'child', 'adult'], default=None,
                    help='validate a whole cohort: train each patient then grand-mean over unseen seeds')
    ap.add_argument('--n_val', type=int, default=500, help='validation episodes per patient (unseen seeds)')
    ap.add_argument('--val_start', type=int, default=1000, help='first validation seed (kept > sup_seeds)')
    a = ap.parse_args()

    if a.selftest:
        _selftest(); return

    if a.cohort:
        mode = 'supervisor' if a.supervisor else 'mpc'   # --supervisor picks which controller to validate
        validate_cohort(a.cohort, mode, model=a.model, sup_seeds=a.sup_seeds, sup_steps=a.sup_steps,
                        n_val=a.n_val, val_start=a.val_start, tune=not a.no_tune)
        return

    if a.supervisor:
        from meta_rl import default_overrides
        ov = default_overrides(a.patient_id)
        polf = f'freestyle_rl/supervisor_p{a.patient_id}.pt'
        print(f'Training RL supervisor (LTI+RL knob, model={a.model}) for p{a.patient_id}...')
        train_supervisor(a.patient_id, polf, ov, seeds=a.sup_seeds, steps=a.sup_steps, model=a.model)
        ck = torch.load(polf, map_location='cpu')
        act = SupActor(ck['sdim']); act.load_state_dict(ck['actor']); act.eval()
        mean, std = ck['mean'], ck['std']
        knob_net = lambda st: act((st - mean) / std)
        neutral  = lambda st: torch.from_numpy(_SUP_NEUT_A).float().unsqueeze(0).expand(st.shape[0], -1)
        print('Running MPC-only baseline (neutral knobs)...')
        base = run_supervisor_episode(a.patient_id, knob_net=neutral, seed=a.seed, overrides=ov, model=a.model)
        print('Running MPC + RL-supervisor...')
        m = run_supervisor_episode(a.patient_id, knob_net=knob_net, seed=a.seed, overrides=ov, model=a.model)
        print(f'\np{a.patient_id}: MPC-only TIR={base["tir"]:.1f} -> +RL-knobs TIR={m["tir"]:.1f}  '
              f'sevHypo={m["sev_hypo"]:.1f}  hypo={m["hypo"]:.1f}  hyper={m["hyper"]:.1f}  min={m["cgm_min"]:.0f}')
        ts = os.environ.get('RUN_TS') or datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = f'bash_results/result_all/results_at_{ts}'; os.makedirs(outdir, exist_ok=True)
        tdd = sum(m['ins_trace']) * 5.0
        foot = {'tir': m['tir'], 'tir_base': base['tir'], 'hypo': m['hypo'],
                'sev_hypo': m['sev_hypo'], 'hyper': m['hyper'], 'cgm_min': m['cgm_min'],
                'cgm_mean': m['cgm_mean'], 'cgm_max': m['cgm_max'], 'tdd': tdd}
        fig = plot_results({'cgm': m['cgm_trace'], 'insulin': m['ins_trace'],
                            'infos': m['infos'], 'bg_target': m['bg_target']}, a.patient_id,
                           cmd=f'python hybrid_policy.py --supervisor --patient_id {a.patient_id} --model {a.model}  [LTI+RL knob]',
                           metrics=foot)
        fname = f'{outdir}/supervisor_{a.model}_{ts}_p{a.patient_id}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'Plot saved to {fname}')
        sfile = _append_summary(outdir,
            f'p{a.patient_id:<2}: MPC-only TIR={base["tir"]:5.1f} -> +RL TIR={m["tir"]:5.1f} | '
            f'sevHypo={m["sev_hypo"]:4.1f} hypo={m["hypo"]:4.1f} hyper={m["hyper"]:5.1f} | '
            f'min={m["cgm_min"]:3.0f} mean={m["cgm_mean"]:3.0f} max={m["cgm_max"]:3.0f} | TDD={tdd:5.1f}U',
            header=f'# LTI+RL supervisor (model={a.model}) - one line per patient\n'
                   f'# columns: TIR(70-180) | sevHypo(<54) hypo(<70) hyper(>180) | CGM min/mean/max | TDD(U/day)')
        print(f'Summary appended to {sfile}')
        return

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
    foot = {'tir': metrics['tir'], 'hypo': metrics['hypo'], 'hyper': metrics['hyper'],
            'cgm_min': metrics['cgm_min'], 'cgm_mean': metrics['cgm_mean'],
            'cgm_max': metrics['cgm_max'], 'tdd': total_insulin}
    fig = plot_results({'cgm': cgm, 'insulin': metrics['ins_trace'],
                        'infos': metrics['infos'], 'bg_target': metrics['bg_target']},
                       a.patient_id, cmd=cmd + f'  [{tag.upper()}]', metrics=foot)
    fname = f'{outdir}/{tag}_results_{ts}_p{a.patient_id}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved to {fname}')
    sfile = _append_summary(outdir,
        f'p{a.patient_id:<2} [{tag}]: TIR={metrics["tir"]:5.1f} | hypo={metrics["hypo"]:4.1f} '
        f'hyper={metrics["hyper"]:5.1f} | min={metrics["cgm_min"]:3.0f} mean={metrics["cgm_mean"]:3.0f} '
        f'max={metrics["cgm_max"]:3.0f} | TDD={total_insulin:5.1f}U',
        header=f'# {tag.upper()} (model={a.model}) - one line per patient\n'
               f'# columns: TIR(70-180) | hypo(<70) hyper(>180) | CGM min/mean/max | TDD(U/day)')
    print(f'Summary appended to {sfile}')


if __name__ == '__main__':
    main()
