import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation', 'G2P2C'))

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from copy import deepcopy

from agents.a2c.a2c import A2C
from agents.a2c.worker import Worker
from agents.a2c.parameters import set_args
from utils.options import Options
from utils.core import get_patient_env


def setup_experiment_dir(args):
    experiment_dir = os.path.join(os.path.dirname(__file__), 'results', args.folder_id)
    os.makedirs(experiment_dir + '/training/data', exist_ok=True)
    os.makedirs(experiment_dir + '/training/plots', exist_ok=True)
    os.makedirs(experiment_dir + '/testing/data', exist_ok=True)
    os.makedirs(experiment_dir + '/testing/plots', exist_ok=True)
    os.makedirs(experiment_dir + '/checkpoints', exist_ok=True)
    args.experiment_dir = experiment_dir
    return args


def train():
    args = Options().parse()
    args = set_args(args)  # apply G2P2C's A2C hyperparameters
    args = setup_experiment_dir(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    patients, env_ids = get_patient_env()

    args.debug = 0
    args.n_training_workers = 16
    args.n_testing_workers = 20
    args.verbose = True

    # create the G2P2C A2C agent
    agent = A2C(args, device, load=False, path1='', path2='')

    # training workers
    worker_agents = [Worker(args, 'training', patients, env_ids, i + 5, i, device)
                     for i in range(args.n_training_workers)]

    # testing workers with fixed meals
    testing_args = deepcopy(args)
    testing_args.meal_amount = [40, 20, 80, 10, 60, 30]
    testing_args.meal_variance = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
    testing_args.time_variance = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
    testing_args.meal_prob = [1, -1, 1, -1, 1, -1]
    testing_agents = [Worker(testing_args, 'testing', patients, env_ids, i + 5000, i + 5000, device)
                      for i in range(args.n_testing_workers)]

    MAX_INTERACTIONS = 4000 if args.debug else 800000
    LR_DECAY_INTERACTIONS = 2000 if args.debug else 600000
    completed_interactions = 0
    last_lr_update = 0
    reward_history = []

    for rollout in range(30000):
        # collect experience from training workers
        for i in range(args.n_training_workers):
            data, _, _ = worker_agents[i].rollout(agent.policy)
            agent.old_states[i] = data['obs']
            agent.old_actions[i] = data['act']
            agent.old_logprobs[i] = data['logp']
            agent.v_pred[i] = data['v_pred']
            agent.reward[i] = data['reward']
            agent.first_flag[i] = data['first_flag']

        # update policy
        agent.update(rollout)
        if rollout % 25 == 0:  # checkpoint every 25 rollouts to keep disk usage in check
            agent.policy.save(rollout)

        # test
        ri = 0
        for i in range(args.n_testing_workers):
            res, _, _ = testing_agents[i].rollout(agent.policy)
            ri += res[0]
        mean_ri = ri / args.n_testing_workers
        reward_history.append(mean_ri)
        print(f"Rollout {rollout}, Mean RI: {mean_ri:.2f}, Interactions: {completed_interactions}")

        # lr decay
        completed_interactions += args.n_step * args.n_training_workers
        if (completed_interactions - last_lr_update) > LR_DECAY_INTERACTIONS:
            agent.decay_lr()
            last_lr_update = completed_interactions

        if completed_interactions > MAX_INTERACTIONS:
            print("Training complete.")
            break

    return agent, reward_history, testing_args


def test(agent, reward_history, testing_args):
    patients, env_ids = get_patient_env()
    device = next(agent.policy.parameters()).device

    worker = Worker(testing_args, 'testing', patients, env_ids, seed=9999, worker_id=9999, device=device)
    worker.rollout(agent.policy)

    # read logged test data
    log_path = testing_args.experiment_dir + '/testing/data/logs_worker_9999.csv'
    df = pd.read_csv(log_path)
    df = df[pd.to_numeric(df['cgm'], errors='coerce').notna()]
    df = df.astype(float).reset_index(drop=True)

    cgm = df['cgm'].values
    insulin = df['ins'].values
    meals = df['meal'].values
    patient_id = testing_args.patient_id

    # --- combined figure with 3 subplots ---
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), gridspec_kw={'height_ratios': [1, 1.5, 0.7]})
    fig.suptitle(f'Glucose Regulation \u2014 Patient {patient_id}', fontsize=14, fontweight='bold')

    # ---- subplot 1: training reward ----
    ax1 = axes[0]
    rewards_arr = np.array(reward_history)
    ax1.plot(rewards_arr, alpha=0.3, color='steelblue', linewidth=0.5)
    window = max(len(rewards_arr) // 20, 5)
    smoothed = np.convolve(rewards_arr, np.ones(window) / window, mode='valid')
    ax1.plot(np.arange(window - 1, len(rewards_arr)), smoothed, color='darkblue', linewidth=2,
             label=f'Smoothed (w={window})')
    ax1.set_title(f'Patient {patient_id} \u2014 Training Reward (LRL2)', fontsize=11)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend(loc='upper right')

    # ---- subplot 2: CGM evaluation ----
    ax2 = axes[1]
    steps = np.arange(len(cgm))
    ax2.fill_between(steps, 70, 180, color='lightgreen', alpha=0.4, label='Target (70-180)')
    ax2.axhline(y=54, color='red', linestyle=':', alpha=0.6, label='Severe hypo (54)')
    ax2.axhline(y=250, color='red', linestyle=':', alpha=0.6, label='Severe hyper (250)')
    ax2.axhline(y=70, color='goldenrod', linestyle='--', alpha=0.6)
    ax2.axhline(y=180, color='goldenrod', linestyle='--', alpha=0.6)
    ax2.plot(steps, cgm, color='darkblue', linewidth=1.5, label='CGM')

    # annotate meals
    meal_indices = np.where(meals > 0)[0]
    for idx in meal_indices:
        cho = meals[idx]
        ax2.axvline(x=idx, color='sienna', linewidth=1.2, alpha=0.7)
        ax2.text(idx + 1, ax2.get_ylim()[1] if len(cgm) > 0 else 400, f'{cho:.0f}g CHO',
                 color='sienna', fontsize=8, va='top')

    ax2.set_title('Evaluation Episode \u2014 24h Deterministic', fontsize=11)
    ax2.set_ylabel('CGM (mg/dL)')
    ax2.set_ylim(0, max(cgm.max() * 1.15, 300))
    ax2.legend(loc='upper right', fontsize=8)

    # ---- subplot 3: insulin (log scale) ----
    ax3 = axes[2]
    ins_plot = np.clip(insulin, 1e-6, None)
    ax3.bar(steps, ins_plot, color='seagreen', alpha=0.7, width=1.0)
    ax3.set_yscale('log')
    ax3.set_ylabel('Insulin (U/min, log)')
    ax3.set_xlabel('Step (5 min each)')

    plt.tight_layout()
    plt.savefig('simulation_result.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    trained_agent, reward_history, testing_args = train()
    test(trained_agent, reward_history, testing_args)
