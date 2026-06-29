# Closed-loop insulin regulation

Model-predictive and hybrid controllers for the FDA-accepted UVA/Padova simulator
(via `simglucose` / G2P2C). Three controllers are included:

Controller, Name, Script for running:
| Bergman MPC | Bergman 1981 nonlinear minimal model | `hybrid_policy.py --model bergman` |
| LTI MPC | Linearized Bergman, exact-discretized, UCSB zone cost | `hybrid_policy.py --model lti` |
| HyCPAP | ZoneMPC + ensemble DRL (Gaussian-product blend, Wu et al. 2024) | `hycpap_full.py` |

NOTE ON EACH GROUP OF PATIENT ID PROVIDED FROM THE SIMULATOR:
Patient ids: 0–9 adolescents, 10–19 children, 20–29 adults.
Each evaluation graph is one 24 h day (288 steps x 5 min interval).


## Setup before running script

Python **3.9 or 3.10** recommended (`gym==0.21.0` is hard to build on newer versions).

```bash
git clone https://github.com/thai-luong05/insulin-agent.git
cd insulin-agent

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

`simglucose` ships its patient parameter files (`Quest.csv`, `vpatient_params.csv`);
the code finds them automatically wherever the package is installed.

---
## Extra note on hybrid policy
`--no-drl` runs pure MPC (the dose comes straight from the tuned Bergman/LTI MPC).
Omitting it blends the MPC dose with a DRL refiner (`freestyle_rl/ppo_p6.pt`) via a
Gaussian product, running the MPC-only baseline first, then the blend.

```bash
python hybrid_policy.py --patient_id 20 --model lti --no-drl   # pure MPC
python hybrid_policy.py --patient_id 20 --model lti            # MPC + DRL blend
```

## Bergman MPC

```bash
python hybrid_policy.py --patient_id 20 --no-drl --model bergman
```

## LTI MPC

```bash
python hybrid_policy.py --patient_id 20 --no-drl --model lti
```

Both controllers personalize 9 MPC knobs per patient with a 40-sample hill-climb
before the run. To skip tuning and use population defaults (much faster):

```bash
python hybrid_policy.py --patient_id 20 --no-drl --model lti --no-tune
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--patient_id N` | 6 | patient 0–29 |
| `--model {bergman,lti}` | bergman | prediction model |
| `--no-drl` | off | pure MPC (no DRL blend) |
| `--no-tune` | off | skip hill-climb, use defaults |
| `--seed N` | 0 | RNG seed |

Dropping `--no-drl` blends a small DRL refiner (`freestyle_rl/ppo_p6.pt`) on top
of the MPC via a Gaussian product; the MPC-only baseline is always run first.

## HyCPAP

HyCPAP needs a shared meta prior first (one-time, ~hours). The trained weights are
**not** in the repo, so run this once:

```bash
python hycpap_full.py --meta --cohort adults      # -> freestyle_rl/meta_full.pt
```

Then, per patient (meta warm-start + ESS fine-tune + eval + plot):

```bash
python hycpap_full.py --patient_id 20
```

If no meta prior exists, it falls back to training that patient from scratch.

The meta prior is cohort-specific — train and evaluate within the same cohort
(the paper uses adults). Meta-train on a cohort, then only evaluate `--patient_id`
from that cohort; e.g. `--cohort adults` (ids 20–29), then `--patient_id 20–29`.
Avoid `--cohort all` (mixing cohorts dilutes the prior).

| Flag | Default | Meaning |
|---|---|---|
| `--patient_id N` | 6 | patient 0–29 |
| `--meta` | off | (re)train the shared meta prior, then exit |
| `--cohort {adults,adolescent,child,all}` | adults | meta-pretrain pool |
| `--general` | off | train this patient from scratch (skip meta + ESS) |
| `--seed N` | 0 | RNG seed |

---

## Output

Each run prints Time-in-Range / hypo / hyper / total daily insulin and saves a
24 h CGM + insulin plot to:

```
bash_results/result_all/results_at_<timestamp>/
```

The directory is created automatically on first run.
