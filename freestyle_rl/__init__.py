"""Freestyle RL: MPC-distilled policy + PPO fine-tune.

Architecture:
    1. Collect (state, action) pairs from the working Bergman MPC across all 30
       patients using their per-patient RL-tuned hyperparameters.
    2. Train a small MLP via behavioural cloning to mimic MPC's policy.
    3. For patients where the BC policy underperforms MPC, fine-tune with PPO
       for a few hundred episodes (~30 min CPU each).

End state: a single neural policy file (`policy.pt`) plus an optional
per-patient PPO refinement, deployable as one MLP forward pass.

This is intentionally separate from `main.py` / `meta_rl.py` so the existing
27/30 MPC baseline stays intact.
"""
