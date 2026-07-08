# LTI MPC: Bergman model with the -X*G term linearized to -Gb*X around (G=Gb, X=0), exact-discretized via matrix exponential. State x=[G-Gb, X, I], inputs [u-u_basal, meal]. UCSB zone-MPC cost (Gondhalekar & Doyle, PMC5419592).
import numpy as np
from scipy.linalg import expm

INFUSION_MINUTE_MAX = 0.6


class LTIMPC:
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
        self.Gb      = float(basal_bg)
        self.u_basal = float(basal_rate)       #U/min, the patient's basal

        # p3 calibrated so 1 U/min steady-state delta-u drops G by ISF mg/dL: p3 = ISF*p2*n*V_I/(1000*Gb)
        self.isf = float(isf)
        self.p3  = self.isf * self.p2 * self.n * self.V_I / (1000.0 * self.Gb)

        #operating-point targets, populated by main.py before the rollout
        self.basal_bg  = None              #patient resting BG (mg/dL)
        self.bg_target = None              #correction setpoint (mg/dL)

        self.u_max = min(INFUSION_MINUTE_MAX,
                         float(basal_rate) * float(max_bolus_multiplier))

        #zone-MPC: control to the band [80,140], not a setpoint
        self.zone_lo    = 80.0
        self.zone_hi    = 140.0
        self.R_dn_frac  = 0.1      #suspending insulin (below basal) is this fraction as costly as dosing above it

        #discrete-time matrices (rebuilt lazily if p2/p3/Gb are overridden after construction)
        self._disc_key = None
        self._discretize()

    def _continuous(self):
        #continuous-time (A, [B|E]) in deviation coords
        A = np.array([[-self.p1, -self.Gb,         0.0],
                      [     0.0,  -self.p2,     self.p3],
                      [     0.0,       0.0,     -self.n]])
        B = np.array([[0.0], [0.0], [1000.0 / self.V_I]])
        E = np.array([[1.0], [0.0], [0.0]])
        return A, np.hstack([B, E])

    def _discretize(self):
        #exact ZOH discretization via the van Loan augmented matrix (handles singular A when p1=0)
        A, Baug = self._continuous()
        nx, ninp = A.shape[0], Baug.shape[1]
        M = np.zeros((nx + ninp, nx + ninp))
        M[:nx, :nx] = A
        M[:nx, nx:] = Baug
        Md = expm(M * self.dt)
        self.A_d    = Md[:nx, :nx]
        self.Bd_aug = Md[:nx, nx:]          #columns: [B_d | E_d]
        self._disc_key = (self.p1, self.p2, self.p3, self.n, self.Gb, self.V_I)

    def _refresh(self):
        #main.py/hybrid_policy.py override mpc.p2 after construction; rebuild if any param changed
        key = (self.p1, self.p2, self.p3, self.n, self.Gb, self.V_I)
        if key != self._disc_key:
            self._discretize()

    def _step(self, G, X, I, u, meal_rate):
        #one dt step of the discrete LTI system (meal_rate in mg/dL per minute)
        x   = np.array([G - self.Gb, X, I])
        inp = np.array([u - self.u_basal, meal_rate])
        xn  = self.A_d @ x + self.Bd_aug @ inp
        return self.Gb + xn[0], xn[1], xn[2]

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
        #zone excursion outside [zone_lo, zone_hi]; hypo (below the band, and especially <70) hurts most
        if G < self.zone_lo:
            c = (self.zone_lo - G) ** 2
            if G < 70.0:
                c += 15.0 * (70.0 - G) ** 2     #extra-steep guard once truly low
            return c
        elif G > self.zone_hi:
            return (G - self.zone_hi) ** 2
        else:
            return 0.0                          #anywhere in the band is free

    def _input_cost(self, u):
        #asymmetric: dosing above basal is full-price, suspending below basal is cheap (R_dn_frac)
        du = u - self.u_basal
        up, dn = max(du, 0.0), max(-du, 0.0)
        return self.R * (up ** 2 + self.R_dn_frac * dn ** 2) * self.horizon * self.dt

    def compute_insulin(self, G_now, X_now, I_now, G_target, meal_carbs=0.0):
        # two-pass grid search: pass 1 keeps predicted BG inside [70,180]; pass 2 falls back to soft cost with the hypo-only guard (G > G_min) if none feasible
        self._refresh()
        candidates = np.linspace(0.0, self.u_max, 25)

        #pass 1: hard clinical bounds
        best_u    = None
        best_cost = float('inf')
        for u in candidates:
            traj = self._rollout([u] * self.horizon, G_now, X_now, I_now, meal_carbs)
            if min(traj) < 70.0 or max(traj) > 180.0:
                continue
            cost = sum(self._zone_cost(g, G_target) for g in traj) + self._input_cost(u)
            if cost < best_cost:
                best_cost = cost
                best_u    = u
        if best_u is not None:
            return float(best_u)

        #pass 2: no feasible candidate (e.g. mid-meal). Soft-fallback with the hypo guard G_min.
        best_u    = 0.0
        best_cost = float('inf')
        for u in candidates:
            traj = self._rollout([u] * self.horizon, G_now, X_now, I_now, meal_carbs)
            if min(traj) < self.G_min:
                continue
            cost = sum(self._zone_cost(g, G_target) for g in traj) + self._input_cost(u)
            if cost < best_cost:
                best_cost = cost
                best_u    = u
        return float(best_u)

    # ── HyCPAP-prior interface (lets this controller act as the Gaussian prior in the blend) ──
    def compute(self, G_now, X_now, I_now, v_now=0.0):
        # correction dose + a spread; sigma wider near/below the zone, tighter into hyper
        u = self.compute_insulin(G_now, X_now, I_now, self.bg_target)
        sigma = 0.10 * self.u_max * (1.0 + max(self.zone_lo - G_now, 0.0) / 40.0)
        return float(u), float(max(sigma, 1e-3))

    def hypo_cap(self, u_desired, G_now, X_now, I_now, floor=70.0):
        # largest dose <= u_desired whose predicted trajectory stays >= floor
        safe = 0.0
        for u in np.linspace(0.0, max(u_desired, 0.0), 13):
            if min(self._rollout([u] * self.horizon, G_now, X_now, I_now, 0.0)) >= floor:
                safe = u
        return float(min(safe, self.u_max))
