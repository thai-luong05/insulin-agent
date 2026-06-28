"""Zone-MPC with glucose- and velocity-dependent control penalties (Shi, Dassau,
Doyle, IEEE TBME 2019) — the MPC half HyCPAP actually uses.

Faithful to Shi 2019's cost design (their Eq. for J):
    J = sum_k [ zhat_k^2 + P(v_k)*z_k^2 + Dhat*v_k^2 ] + sum_k [ Rhat*u_k^2 + Rcheck*ud_k^2 ]
where
    z_k     = glucose zone excursion (0 inside the target zone [80,140], else
              distance to the nearest bound),
    v_k     = glucose velocity (rate of change),
    P(v_k)  = velocity-dependent weight on the zone term,
    Rhat    = penalty on insulin ABOVE basal, scaled DOWN when glucose is high
              and rising (dose more), UP when low/falling (the adaptive part),
    Rcheck  = penalty on insulin below basal.

Prediction uses the same Bergman 3-state model as the rest of the project (Shi
used a personalized linear model; we substitute Bergman, documented).

`compute()` returns (u*, sigma): the optimal first insulin rate and a spread,
so the controller can be used as the Gaussian prior psi(a|s) ~ N(u*+0, sigma)
in the HyCPAP blend.
"""
import numpy as np

PUMP_MAX = 0.6
ZONE_LO, ZONE_HI = 80.0, 140.0     # target euglycemic zone (Shi 2019)


class ZoneMPC:
    def __init__(self, basal_rate, isf, basal_bg=120.0, body_weight=55.0,
                 sample_time=5, horizon=24, body_target=None,
                 R_hat=0.02, R_check=0.05, D_hat=0.5, max_bolus_multiplier=40.0):
        self.dt      = int(sample_time)
        self.horizon = int(horizon)
        self.R_hat   = float(R_hat)        # base penalty, insulin above basal
        self.R_check = float(R_check)      # penalty, insulin below basal
        self.D_hat   = float(D_hat)        # velocity penalty
        # Bergman params (shared with bergman_controller)
        self.p1, self.p2, self.n = 0.0, 0.025, 0.142
        self.V_I = 0.04 * float(body_weight)
        self.Gb       = float(basal_bg)
        self.u_basal  = float(basal_rate)
        self.isf      = float(isf)
        self.p3 = self.isf * self.p2 * self.n * self.V_I / (1000.0 * self.Gb)
        self.u_max = min(PUMP_MAX, float(basal_rate) * float(max_bolus_multiplier))
        # compat attributes used by hybrid_policy's external state tracking
        self.basal_bg  = basal_bg
        self.bg_target = 0.5 * (ZONE_LO + ZONE_HI) if body_target is None else body_target

    # ── Bergman one-step + rollout (first move u, then basal) ──
    def _step(self, G, X, I, u):
        dG = -self.p1 * (G - self.Gb) - X * G
        dX = -self.p2 * X + self.p3 * I
        dI = -self.n * I + 1000.0 * (u - self.u_basal) / self.V_I
        return G + self.dt * dG, X + self.dt * dX, I + self.dt * dI

    def _rollout(self, u0, G, X, I):
        Gs = [G]
        for k in range(self.horizon):
            u = u0 if k == 0 else self.u_basal
            G, X, I = self._step(G, X, I, u)
            Gs.append(G)
        return Gs

    @staticmethod
    def _zone_excursion(g):
        if g < ZONE_LO:  return ZONE_LO - g
        if g > ZONE_HI:  return g - ZONE_HI
        return 0.0

    @staticmethod
    def _vel_zone_weight(v):
        # P(v): rising glucose (v>0) weights the (hyper) zone term more so the
        # controller acts; falling glucose softens it (avoid over-correction).
        return 1.0 + max(v, 0.0) / 5.0

    def _adaptive_R(self, g, v):
        # Shi's glucose+velocity-dependent control penalty: cheaper to dose
        # (lower R) when glucose is HIGH and RISING; dearer (higher R) when LOW
        # or FALLING -> protects against hypo.
        hi  = max(g - ZONE_HI, 0.0) / 60.0      # how far above zone
        lo  = max(ZONE_LO - g, 0.0) / 30.0      # how far below zone
        rise = max(v, 0.0) / 4.0
        fall = max(-v, 0.0) / 4.0
        scale = (1.0 + lo + fall) / (1.0 + hi + rise)
        return self.R_hat * float(np.clip(scale, 0.2, 8.0))

    def compute(self, G_now, X_now, I_now, v_now=0.0):
        best_u, best_J = 0.0, float('inf')
        for u in np.linspace(0.0, self.u_max, 41):
            traj = self._rollout(u, G_now, X_now, I_now)
            J = 0.0
            for k in range(1, len(traj)):
                g = traj[k]
                v = (traj[k] - traj[k - 1]) / self.dt
                z = self._zone_excursion(g)
                J += z * z + self._vel_zone_weight(v) * z * z + self.D_hat * v * v
            # asymmetric, adaptive control penalty on the first move
            ud_above = max(u - self.u_basal, 0.0)
            ud_below = max(self.u_basal - u, 0.0)
            Rad = self._adaptive_R(G_now, v_now)
            J += Rad * ud_above ** 2 * self.horizon + self.R_check * ud_below ** 2 * self.horizon
            # hard hypo guard (Shi keeps a safety constraint)
            if min(traj) < 70.0:
                J += 1e4 * (70.0 - min(traj)) ** 2
            if J < best_J:
                best_J, best_u = J, u
        # sigma for the blend: wider when near/below zone (less certain push),
        # tighter when confidently dosing into hyper
        sigma = 0.10 * self.u_max * (1.0 + max(ZONE_LO - G_now, 0.0) / 40.0)
        return float(best_u), float(max(sigma, 1e-3))


    def hypo_cap(self, u_desired, G, X, I, floor=70.0):
        """Largest dose <= u_desired whose predicted trajectory stays >= floor.
        This is the MPC's own hypo constraint applied to the TOTAL insulin
        (meal bolus + correction) — the paper's safety mechanism, vs a bolus
        floor that bypasses it."""
        safe = 0.0
        for u in np.linspace(0.0, max(u_desired, 0.0), 13):
            if min(self._rollout(u, G, X, I)) >= floor:
                safe = u
        return float(min(safe, self.u_max))


def bolus_calculator(carbs, glucose, icr, isf, target=110.0, iob=0.0):
    """Standard meal bolus + correction (Shi 2019 / clinical):
        bolus = carbs/ICR + max(0, (glucose - target)/ISF - IOB)."""
    meal = carbs / icr if carbs > 0 else 0.0
    corr = max(0.0, (glucose - target) / isf - iob)
    return meal + corr
