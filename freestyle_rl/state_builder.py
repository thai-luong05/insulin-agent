"""Build the 10-dim state vector for the policy at each step.

Shared between BC data collection, PPO training, and inference.
"""
import numpy as np

GLUCOSE_TARGET = 110.0


def initial_state(basal, bw, isf_q, icr_q, tdd_q, init_cgm):
    """Pre-rollout state used during calibration."""
    return np.array([
        basal / 0.05,
        bw    / 100.0,
        isf_q / 200.0,
        icr_q / 30.0,
        tdd_q / 100.0,
        (init_cgm - GLUCOSE_TARGET) / 100.0,
        0.0,         #dCGM/dt
        0.0,         #IOB proxy
        0.0,         #meal announced
        1.0,         #fraction of 240 min until next meal (default: no meal soon)
    ], dtype=np.float32)


def build_state(basal, bw, isf_q, icr_q, tdd_q, cgm_now, cgm_prev,
                iob_proxy, meal_carbs_announced, mins_to_next_meal):
    """Step-wise state. mins_to_next_meal=0 if no meal within 4h horizon."""
    dcgm = (cgm_now - cgm_prev)
    return np.array([
        basal / 0.05,
        bw    / 100.0,
        isf_q / 200.0,
        icr_q / 30.0,
        tdd_q / 100.0,
        (cgm_now - GLUCOSE_TARGET) / 100.0,
        dcgm / 30.0,
        min(iob_proxy, 30.0) / 30.0,
        min(meal_carbs_announced, 150.0) / 100.0,
        min(mins_to_next_meal, 240.0) / 240.0,
    ], dtype=np.float32)


def iob_decay(prev_iob, just_delivered_U, gamma=0.93):
    """Geometric IOB tracker matching the MPC's i_eff model."""
    return gamma * prev_iob + just_delivered_U
