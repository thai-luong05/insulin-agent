# Bergman 1981 3-state minimal-model MPC: states G (glucose), X (insulin action), I (plasma insulin); X captures the 15-30 min insulin-action delay the 2-state LTI misses. T1D pop params: p1=0, p2=0.025/min, n=0.142/min, V_I=0.04*BW, p3 calibrated per patient from ISF.
import numpy as np

INFUSION_MINUTE_MAX = 0.6


class BergmanMPC:
    def __init__(self, basal_rate, isf, basal_bg=120.0, body_weight=55.0,
                 sample_time=5, horizon=12, R=0.01, G_min=90.0,
                 max_bolus_multiplier=15.0, beta=3.0):
        self.dt           = int(sample_time)
        self.horizon      = int(horizon)
        self.R            = float(R)
        self.G_min        = float(G_min)
        self.beta         = float(beta)
        self.absorb_steps = 6     #meal absorbs over 30 min

        #population bergman params (T1D)
        self.p1 = 0.0
        self.p2 = 0.025
        self.n  = 0.142

        #insulin distribution volume from body weight (Bergman scaling)
        self.V_I = 0.04 * float(body_weight)   #L

        #operating point
        self.Gb       = float(basal_bg)
        self.u_basal  = float(basal_rate)      #U/min, the patient's basal

        # p3 calibrated so 1 U/min steady-state delta-u drops G by ISF mg/dL: p3 = ISF*p2*n*V_I/(1000*Gb)
        self.isf = float(isf)
        self.p3  = self.isf * self.p2 * self.n * self.V_I / (1000.0 * self.Gb)

        #operating-point targets, populated by main.py before the rollout
        self.basal_bg  = None              #patient resting BG (mg/dL)
        self.bg_target = None              #correction setpoint (mg/dL)

        self.u_max = min(INFUSION_MINUTE_MAX,
                         float(basal_rate) * float(max_bolus_multiplier))

    def _step(self, G, X, I, u, meal_rate):
        """One dt step. meal_rate in mg/dL per step (already scaled)."""
        dG = -self.p1 * (G - self.Gb) - X * G + meal_rate
        dX = -self.p2 * X + self.p3 * I
        dI = -self.n * I + 1000.0 * (u - self.u_basal) / self.V_I
        return G + self.dt * dG, X + self.dt * dX, I + self.dt * dI

    def _rollout(self, u_seq, G0, X0, I0, meal_carbs):
        G, X, I = float(G0), float(X0), float(I0)
        Gs = []
        meal_per_step = (self.beta * float(meal_carbs) / self.absorb_steps) \
                        if meal_carbs > 0 else 0.0
        for k in range(self.horizon):
            mr = meal_per_step if k < self.absorb_steps else 0.0
            G, X, I = self._step(G, X, I, u_seq[k], mr)
            Gs.append(G)
        return Gs

    def _zone_cost(self, G, G_target):
        if G < 70.0:
            return 16.0 * (G - 70.0) ** 2
        elif G > 180.0:
            return (G - 180.0) ** 2
        else:
            return 0.01 * (G - G_target) ** 2

    def compute_insulin(self, G_now, X_now, I_now, G_target, meal_carbs=0.0):
        # two-pass grid search: pass 1 keeps predicted BG inside [70,180]; pass 2 falls back to soft cost with the hypo-only guard (G > G_min) if none feasible
        candidates = np.linspace(0.0, self.u_max, 25)

        #pass 1: hard clinical bounds
        best_u    = None
        best_cost = float('inf')
        G_max_clinical = 180.0
        G_min_clinical = 70.0
        for u in candidates:
            traj = self._rollout([u] * self.horizon, G_now, X_now, I_now, meal_carbs)
            if min(traj) < G_min_clinical or max(traj) > G_max_clinical:
                continue
            cost = (sum(self._zone_cost(g, G_target) for g in traj)
                    + self.R * u ** 2 * self.horizon * self.dt)
            if cost < best_cost:
                best_cost = cost
                best_u    = u
        if best_u is not None:
            return float(best_u)

        #pass 2: no feasible candidate (e.g. mid-meal). Soft-fallback with the
        #original hypo guard G_min=90 — pick the best of the rest.
        best_u    = 0.0
        best_cost = float('inf')
        for u in candidates:
            traj = self._rollout([u] * self.horizon, G_now, X_now, I_now, meal_carbs)
            if min(traj) < self.G_min:
                continue
            cost = (sum(self._zone_cost(g, G_target) for g in traj)
                    + self.R * u ** 2 * self.horizon * self.dt)
            if cost < best_cost:
                best_cost = cost
                best_u    = u
        return float(best_u)

    # ── HyCPAP-prior interface (lets this controller act as the Gaussian prior in the blend) ──
    def compute(self, G_now, X_now, I_now, v_now=0.0):
        # correction dose + a spread; sigma wider near/below the zone, tighter into hyper
        u = self.compute_insulin(G_now, X_now, I_now, self.bg_target)
        sigma = 0.10 * self.u_max * (1.0 + max(80.0 - G_now, 0.0) / 40.0)   # 80 = euglycemic zone lower bound
        return float(u), float(max(sigma, 1e-3))

    def hypo_cap(self, u_desired, G_now, X_now, I_now, floor=70.0):
        # largest dose <= u_desired whose predicted trajectory stays >= floor
        safe = 0.0
        for u in np.linspace(0.0, max(u_desired, 0.0), 13):
            if min(self._rollout([u] * self.horizon, G_now, X_now, I_now, 0.0)) >= floor:
                safe = u
        return float(min(safe, self.u_max))
