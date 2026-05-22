"""lti metabolic model + mpc controller.

G[k+1]     = G[k] + egp - alpha*I_eff[k] + beta*meal[k]
I_eff[k+1] = gamma*I_eff[k] + u[k]*sample_time
"""
import numpy as np

INFUSION_MINUTE_MAX = 0.6  #must match env.py
IOB_GAMMA = 0.93           #5-min decay, half-life ~48 min (lispro/aspart)


class MPCController:
    def __init__(self, basal_rate, sample_time=5, horizon=12,
                 alpha=5.0, gamma=IOB_GAMMA, beta=3.0,
                 R=0.01, G_min=90.0, max_bolus_multiplier=15.0):
        self.sample_time = int(sample_time)
        self.horizon     = int(horizon)
        self.alpha       = alpha
        self.gamma       = gamma
        self.beta        = beta
        self.R           = R
        self.G_min       = G_min
        #egp calibrated so model steady state matches patient basal equilibrium
        i_eff_ss = float(basal_rate) * self.sample_time / (1.0 - self.gamma)
        self.egp      = self.alpha * i_eff_ss
        self.basal_bg = None
        self.u_max = min(INFUSION_MINUTE_MAX,
                         float(basal_rate) * float(max_bolus_multiplier))

    def _rollout(self, u_seq, G0, I0, meal_carbs):
        G, I = float(G0), float(I0)
        Gs = []
        absorption_steps = min(6, self.horizon)
        meal_per_step = self.beta * float(meal_carbs) / absorption_steps if meal_carbs else 0.0
        for k in range(self.horizon):
            meal_k = meal_per_step if k < absorption_steps else 0.0
            G = G + self.egp - self.alpha * I + meal_k
            I = self.gamma * I + float(u_seq[k]) * self.sample_time
            Gs.append(G)
        return Gs

    def _zone_cost(self, G, G_target):
        #16x hypo penalty, std hyper penalty, gentle in-range centering
        if G < 70.0:
            return 16.0 * (G - 70.0) ** 2
        elif G > 180.0:
            return (G - 180.0) ** 2
        else:
            return 0.01 * (G - G_target) ** 2

    def _objective(self, u_seq, G0, I0, G_target, meal_carbs):
        Gs = self._rollout(u_seq, G0, I0, meal_carbs)
        track = sum(self._zone_cost(g, G_target) for g in Gs)
        effort = self.R * sum(u ** 2 for u in u_seq) * self.sample_time
        return track + effort

    def _hypo_con(self, u_seq, G0, I0, meal_carbs):
        return np.array([g - self.G_min
                         for g in self._rollout(u_seq, G0, I0, meal_carbs)])

    def compute_insulin(self, G_now, I_eff, G_target, meal_carbs=0.0):
        """grid search over 25 candidate rates, pick lowest cost respecting g_min."""
        u_max = self.u_max
        best_u = 0.0
        best_cost = float('inf')

        for u in np.linspace(0.0, u_max, 25):
            traj = self._rollout([u] * self.horizon, G_now, I_eff, meal_carbs)
            if min(traj) < self.G_min:
                continue
            cost = (sum(self._zone_cost(g, G_target) for g in traj)
                    + self.R * u ** 2 * self.horizon * self.sample_time)
            if cost < best_cost:
                best_cost = cost
                best_u = u

        return float(best_u)
