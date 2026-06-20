# Per-patient hill-climb tuning of the Bergman MPC knobs (no cache; change n_samples in tune_patient and re-run). Reward = severity-aware TIR minus weighted hypo/hyper (see _reward).
import numpy as np

PARAM_RANGES = {
    'isf_mult':                 (0.4, 2.5), # insulin sensitivity
    'correction_target_offset': (-30.0, 10.0), #glucose target setting
    'max_bolus_mult':           (10.0, 50.0),
    'R_log10':                  (-3.0, -1.0),    #R in [0.001, 0.1]
    'beta':                     (1.5, 6.0),      #meal-to-glucose gain (mg/dL per g)
    'p2':                       (0.015, 0.05),   #X decay rate (1/min); half-life 14-46 min
    'horizon':                  (12.0, 36.0),    #MPC horizon (steps); most impactful knob: hypo-prone long, hyper-prone short
    'prebolus_steps':           (3.0, 9.0),      #meal bolus lead time (steps, 15-45 min)
    'cr_mult':                  (0.7, 1.3),      #scale on ground-truth carb ratio
}

# cohort default for max_bolus_mult: high so the feedforward bolus can reach the env's 0.6 U/min cap
def cohort_max_bolus_default(patient_id):
    return 40.0


def default_overrides(patient_id):
    return {
        'isf_mult':                 1.0,
        'correction_target_offset': -10.0,
        'max_bolus_mult':           cohort_max_bolus_default(patient_id),
        'R_log10':                  -2.0,        #R = 0.01
        'beta':                     3.0,         #mg/dL per g carb
        'p2':                       0.025,       #1/min, half-life ~28 min
        'horizon':                  24.0,        #2h lookahead
        'prebolus_steps':           5.0,         #25 min meal lead
        'cr_mult':                  1.0,         #ground-truth CR
    }


def _reward(metrics):
    # severity-aware reward: separates severe hypo (<54) and severe hyper (>250) from mild excursions so the search can trade a little mild hypo for TIR
    cgm          = metrics['cgm_trace']
    n            = max(len(cgm), 1)
    severe_hypo  = sum(g < 54.0  for g in cgm) / n * 100.0   #paramedic threshold
    severe_hyper = sum(g > 250.0 for g in cgm) / n * 100.0
    mild_hypo    = metrics['hypo'] - severe_hypo             #54-70 range
    return (metrics['tir']
            - 3.0 * severe_hypo
            - 1.0 * mild_hypo
            - 0.5 * metrics['hyper']
            - 1.0 * severe_hyper)


def _perturb(base, rng, sigma=0.3):
    # gaussian perturbation around base, clipped to the search box
    out = {}
    for k, (lo, hi) in PARAM_RANGES.items():
        out[k] = float(np.clip(
            base[k] + rng.normal(0, sigma * (hi - lo)), lo, hi))
    return out


def tune_patient(patient_id, seed=0, n_samples=40):
    # hill climb over the clinical+controller knobs, starting from cohort-aware defaults each call (no cache)
    from main import evaluate_patient

    rng = np.random.default_rng(seed + patient_id)

    best = default_overrides(patient_id)
    m = evaluate_patient(patient_id, overrides=best, seed=seed, verbose=False)
    best_r = _reward(m)
    print(f'  baseline: TIR={m["tir"]:5.1f}%  hypo={m["hypo"]:4.1f}%  '
          f'hyper={m["hyper"]:4.1f}%  reward={best_r:6.2f}')

    for i in range(n_samples):
        sigma = 0.4 - 0.3 * (i / max(1, n_samples - 1))
        cand  = _perturb(best, rng, sigma=sigma)
        m     = evaluate_patient(patient_id, overrides=cand, seed=seed, verbose=False)
        r     = _reward(m)
        flag  = ''
        if r > best_r:
            best_r = r
            best   = cand
            flag   = '  <- new best'
        print(f'  s{i+1:2d}/{n_samples} sig={sigma:.2f}: TIR={m["tir"]:5.1f}%  '
              f'hypo={m["hypo"]:4.1f}%  hyper={m["hyper"]:4.1f}%  reward={r:6.2f}  '
              f'(isf*={cand["isf_mult"]:.2f} off={cand["correction_target_offset"]:+.1f} '
              f'mb*={cand["max_bolus_mult"]:.1f} R=10^{cand["R_log10"]:+.1f} '
              f'b={cand["beta"]:.2f} p2={cand["p2"]:.3f}){flag}')

    print(f'\nbest reward {best_r:.2f}  params {best}')
    return best
