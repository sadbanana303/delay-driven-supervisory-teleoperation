#!/usr/bin/env python3
"""
Measured-threshold, delay-driven supervisory control for teleoperation
(planar two-link nonlinear manipulator).

Companion code for:
  "Measured-Threshold, Delay-Driven Supervisory Control for Teleoperation under
   Normalized Delay: A Simulation Study on a Nonlinear Two-Link Manipulator"

Controllers (all share the same saturated joint PD law with gravity compensation):
  classical   - delayed PD on the most recently received operator command
  predictive  - timestamped constant-velocity extrapolation of that command
  autonomous  - stored minimum-jerk plan plus the operator's latest retarget as an offset

Supervisors:
  supervisor         - decides on the packet-based delay estimate d_hat (Eq. 5)
  supervisor_age     - decides on d_eff = max(d_hat, packet age) (Eq. 6)
  supervisor_oracle  - decides on the true instantaneous delay
  supervisor_hand    - hand-picked thresholds
  supervisor_nohyst  - no hysteresis band, no exit margin, no dwell

Thresholds are the first persistent sign changes in the mean-RMSE curves of the
fixed controllers, measured on the plant. Hysteresis bands come from bootstrap
confidence intervals over seeds.

Delay is normalized to task duration (Td / 28 s). No claim is made about Mars
light time in seconds.

Usage:
  python delay_driven_supervisor.py            full pipeline (long)
  python delay_driven_supervisor.py --smoke    quick execution check (one seed)
  python delay_driven_supervisor.py --processes 8
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

# ---------------------------------------------------------------- parameters
DT_PLANT, DT_CTRL = 0.002, 0.010
CE = int(round(DT_CTRL / DT_PLANT))
KP, KD, TMAX = (28.0, 18.0), (3.8, 2.7), (16.0, 10.0)
N_WP, TASK_TIME, DEADLINE = 4, 28.0, 30.0
KO = 0.3
JIT = 0.10
NQ, NDQ, MODEL_UNC = 5e-4, 5e-3, 0.10
N_EV, SHIFT, SHIFT_RAMP = 2, 0.30, 4.0
MIN_DWELL_DEFAULT, EWMA_DEFAULT, TOL = 2.0, 0.10, 0.02
N_CAL, N_HOLD, SEED_CAL, SEED_HOLD = 30, 30, 0, 100
PERSIST = 2                 # further nonzero samples the new sign must hold
BAND_FLOOR = 0.05           # +/-5% floor on the T_cross band
BOOT_THR, BOOT_REG = 500, 1000
EQUIV_MARGIN = 5.0          # pre-specified relative-regret margin, percent
PRIMARY_PRIOR = 0.10        # relative error of the prior on the initial delay

# Calibration grid (21 points) and held-out grid (14 points, disjoint from it)
CAL_DELAYS = [0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0,
              9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 17.0, 20.0, 22.0]
HOLD_DELAYS = [0.7, 1.5, 2.5, 2.9, 3.3, 3.8, 4.8, 6.5, 8.5, 11.5, 14.0, 16.0,
               18.5, 21.0]
assert not set(CAL_DELAYS) & set(HOLD_DELAYS), "calibration/held-out delays overlap"

# None = no prior (est0 = 0); a float is the relative error on the initial delay
PRIOR_ERRS = [None, 0.0, PRIMARY_PRIOR, 0.50]

FIXED = ["classical", "predictive", "autonomous"]
VARIANTS = {
    "Sup": "supervisor",
    "Age": "supervisor_age",
    "Oracle": "supervisor_oracle",
    "Hand": "supervisor_hand",
    "NoHyst": "supervisor_nohyst",
}
HAND_THRESHOLDS = dict(cross=1.0, low=0.9, high=1.1, auto=10.0, auto_exit=9.5)

OUT = Path("results")
OUT.mkdir(exist_ok=True)
N_PROCS = None              # None = all cores; set with --processes

FIG_W, FIG_DPI, FONT_SIZE = 3.5, 300, 8
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# --------------------------------------------------------------------- plant
class Arm:
    l1, l2, lc1, lc2, I1, I2, g = 0.35, 0.30, 0.175, 0.15, 0.020, 0.012, 9.81

    def __init__(self, m1=1.5, m2=1.0):
        self.a = self.I1 + self.I2 + m1 * self.lc1**2 + m2 * (self.l1**2 + self.lc2**2)
        self.b = m2 * self.l1 * self.lc2
        self.c = self.I2 + m2 * self.lc2**2
        self.k1 = (m1 * self.lc1 + m2 * self.l1) * self.g
        self.k2 = m2 * self.lc2 * self.g

    def grav(self, q1, q2):
        g2 = self.k2 * math.cos(q1 + q2)
        return self.k1 * math.cos(q1) + g2, g2

    def acc(self, q1, q2, w1, w2, t1, t2):
        c2 = math.cos(q2)
        h = -self.b * math.sin(q2)
        m11 = self.a + 2 * self.b * c2
        m12 = self.c + self.b * c2
        m22 = self.c
        g1, g2 = self.grav(q1, q2)
        r1 = t1 - h * w2 * w1 - h * (w1 + w2) * w2 - g1
        r2 = t2 + h * w1 * w1 - g2
        d = m11 * m22 - m12 * m12
        return (m22 * r1 - m12 * r2) / d, (m11 * r2 - m12 * r1) / d

    def fk(self, q):
        q = np.asarray(q)
        return np.stack([
            self.l1 * np.cos(q[..., 0]) + self.l2 * np.cos(q[..., 0] + q[..., 1]),
            self.l1 * np.sin(q[..., 0]) + self.l2 * np.sin(q[..., 0] + q[..., 1]),
        ], -1)


def ik(r, x, y):
    c2 = np.clip((x * x + y * y - r.l1**2 - r.l2**2) / (2 * r.l1 * r.l2), -1.0, 1.0)
    s2 = np.sqrt(np.maximum(0.0, 1.0 - c2 * c2))
    q2 = math.atan2(s2, c2)
    return np.array([
        math.atan2(y, x) - math.atan2(r.l2 * s2, r.l1 + r.l2 * c2),
        q2,
    ])


def pd_ctrl(M, ref, dref, q1, q2, w1, w2, kp, kd):
    """Saturated joint PD law with gravity compensation (Eq. 2)."""
    g1, g2 = M.grav(q1, q2)
    t1 = kp[0] * (ref[0] - q1) + kd[0] * (dref[0] - w1) + g1
    t2 = kp[1] * (ref[1] - q2) + kd[1] * (dref[1] - w2) + g2
    return (max(-TMAX[0], min(TMAX[0], t1)),
            max(-TMAX[1], min(TMAX[1], t2)))


# ------------------------------------------------------------------ scenario
def make_delay_profile(n, profile):
    """One-way delay Td(t) per control step for the time-varying profiles."""
    t = np.arange(n) * DT_CTRL
    if profile == "ramp":                       # 0.5 s -> 15 s over the mission
        return 0.5 + (15.0 - 0.5) * (t / DEADLINE)
    if profile == "step":                       # 1 s -> 12 s at 12.6 s
        td = np.full(n, 1.0)
        td[t >= TASK_TIME * 0.45] = 12.0
        return td
    if profile == "sinusoid":                   # mean 6 s, amplitude 7 s, clipped
        return np.clip(6.0 + 7.0 * np.sin(2 * np.pi * t / DEADLINE), 0.2, 18.0)
    raise ValueError(profile)


def scenario(seed, Td, profile="constant", n_ev=None, shift=None,
             mass_scale=(1.0, 1.0), prior_err=None):
    """
    Common-random-numbers scenario. All modes see the same noise, delays,
    retargets and model error for a given seed.

    Td        : scalar (constant profile); ignored for time-varying profiles
    prior_err : None -> initial delay estimate 0; float -> relative error on the
                initial delay Td(0), i.e. est0 = Td(0) * (1 + prior_err * eps)
    """
    rng = np.random.RandomState(seed + 7919)
    R = Arm()
    xs = np.linspace(0.27, 0.43, N_WP) + 0.01 * rng.randn(N_WP)
    ys = 0.17 + 0.10 * np.sin(np.linspace(0, np.pi, N_WP)) + 0.008 * rng.randn(N_WP)
    qw = np.array([ik(R, x, y) for x, y in np.clip(np.c_[xs, ys], 0.18, 0.52)])
    n = int(round(DEADLINE / DT_CTRL))
    t = np.arange(n) * DT_CTRL

    Td_t = np.full(n, float(Td)) if profile == "constant" else make_delay_profile(n, profile)

    seg = TASK_TIME / (N_WP - 1)
    k = np.minimum((t // seg).astype(int), N_WP - 2)
    u = np.minimum((t - k * seg) / seg, 1.0)
    s = (10 * u**3 - 15 * u**4 + 6 * u**5)[:, None]
    qref = qw[k] + (qw[k + 1] - qw[k]) * s

    g = qref.copy()
    n_ev = N_EV if n_ev is None else n_ev
    shift = SHIFT if shift is None else shift
    for e in range(n_ev):
        ts = rng.uniform(0.15 + 0.40 * e, 0.45 + 0.40 * e) * TASK_TIME
        th = rng.uniform(0, 2 * np.pi)
        v = np.clip((t - ts) / SHIFT_RAMP, 0, 1)
        sm = (3 * v**2 - 2 * v**3)[:, None]
        g += sm * shift * np.array([math.cos(th), math.sin(th)])

    ul = np.maximum(0.0, Td_t * (1 + JIT * rng.randn(n)))
    dl = np.maximum(0.0, Td_t * (1 + JIT * rng.randn(n)))
    a = np.ceil((t + ul) / DT_CTRL - 1e-9).astype(int)
    best = np.full(n, -1)
    m = a < n
    np.maximum.at(best, a[m], np.arange(n)[m])
    best = np.maximum.accumulate(best)

    nq = NQ * rng.randn(n, 2)
    nw = NDQ * rng.randn(n, 2)
    mu = np.clip(MODEL_UNC * rng.randn(2), -0.25, 0.25)
    m1 = 1.5 * mass_scale[0] * (1 + mu[0])
    m2 = 1.0 * mass_scale[1] * (1 + mu[1])

    if prior_err is None:
        est0 = 0.0
    else:
        est0 = max(0.0, float(Td_t[0]) * (1.0 + prior_err * rng.randn()))

    return dict(
        R=R, M=Arm(m1, m2), n=n, g=g, qref=qref,
        dqref=np.gradient(qref, DT_CTRL, axis=0),
        ul=ul, dl=dl, best=best, nq=nq, nw=nw, est0=est0,
        q0=qw[0] + np.array([0.025, -0.015]), Td_t=Td_t,
    )


# ----------------------------------------------------------- switching logic
def choose(d, cur, th, use_hysteresis=True):
    """
    Mode selection from the decision delay d (d_hat or d_eff).
      From any mode: A when d >= T_A.
      From A: stay while d > T_A,exit; then C if d > T_high, else P.
      From P: C when d > T_high.   From C: P when d < T_low.
    The minimum dwell time is enforced by the caller.
    """
    cr, tl, thi = th["cross"], th["low"], th["high"]
    ta, tx = th["auto"], th["auto_exit"]
    if not use_hysteresis:
        if d >= ta:
            return "autonomous"
        return "classical" if d > cr else "predictive"
    if cur is None:                                   # initial mode
        return "autonomous" if d >= ta else ("classical" if d > cr else "predictive")
    if cur == "autonomous":
        if d > tx:
            return "autonomous"
        return "classical" if d > thi else "predictive"
    if d >= ta:
        return "autonomous"
    if cur == "predictive":
        return "classical" if d > thi else "predictive"
    return "predictive" if d < tl else "classical"


# --------------------------------------------------------------------- one run
def simulate(mode, Td, seed, th=None, profile="constant", n_ev=None, shift=None,
             mass_scale=(1.0, 1.0), kp_scale=1.0, prior_err=None,
             use_hysteresis=True, min_dwell=None, ewma=None, shared_auto=True,
             return_trace=False):
    """
    mode: classical | predictive | autonomous | supervisor | supervisor_age
          | supervisor_oracle | supervisor_hand | supervisor_nohyst
    kp_scale scales both Kp and Kd.
    """
    S = scenario(seed, Td, profile=profile, n_ev=n_ev, shift=shift,
                 mass_scale=mass_scale, prior_err=prior_err)
    R, M, n = S["R"], S["M"], S["n"]
    g = S["g"].tolist()
    qr = S["qref"].tolist()
    dqr = S["dqref"].tolist()
    ul = S["ul"].tolist()
    dl = S["dl"].tolist()
    best = S["best"].tolist()
    nq = S["nq"].tolist()
    nw = S["nw"].tolist()
    q0 = S["q0"].tolist()
    dg = np.gradient(S["g"], DT_CTRL, axis=0).tolist()
    Td_t = S["Td_t"]

    kp = (KP[0] * kp_scale, KP[1] * kp_scale)
    kd = (KD[0] * kp_scale, KD[1] * kp_scale)
    ewma = EWMA_DEFAULT if ewma is None else ewma
    min_dwell = MIN_DWELL_DEFAULT if min_dwell is None else min_dwell

    is_sup = mode.startswith("supervisor")
    if mode == "supervisor_hand":
        th = HAND_THRESHOLDS
    elif mode == "supervisor_nohyst":
        use_hysteresis = False
        min_dwell = 0.0

    q1, q2 = q0
    w1 = w2 = 0.0
    cur = None if is_sup else mode
    est = S["est0"]
    last_j = -1
    t_sw = 0.0
    sw = 0
    n_auto = 0
    sat = 0
    t_first_auto = None
    peak_post_sw = 0.0
    last_sw_k = -9999
    window = int(1.0 / DT_CTRL)
    qh, rh, dh = [], [], []
    mode_hist, est_hist, deff_hist = [], [], []

    for k in range(n):
        t = k * DT_CTRL
        qh.append((q1, q2))

        # operator (Earth): true goal plus correction on the stale error
        sidx = max(0, k - int(round(dl[k] / DT_CTRL)))
        rh.append((g[k][0] + KO * (g[sidx][0] - qh[sidx][0]),
                   g[k][1] + KO * (g[sidx][1] - qh[sidx][1])))
        dh.append(dg[k])

        # newest packet that has arrived, and the delay estimate (Eq. 5)
        j = best[k]
        if j != last_j and j >= 0:
            est += ewma * (ul[j] - est)
            last_j = j

        # mode decision
        if is_sup:
            if mode == "supervisor_oracle":
                d_dec = float(Td_t[k])
            elif mode == "supervisor_age":
                age = (t - j * DT_CTRL) if j >= 0 else t      # Eq. 6
                d_dec = max(est, age)
            else:
                d_dec = est
            if cur is None:
                cur = choose(d_dec, None, th, use_hysteresis)
            elif t - t_sw >= min_dwell:
                new = choose(d_dec, cur, th, use_hysteresis)
                if new != cur:
                    cur, t_sw, sw = new, t, sw + 1
                    last_sw_k = k
            if return_trace:
                est_hist.append(est)
                deff_hist.append(d_dec)

        if cur == "autonomous":
            n_auto += 1
            if t_first_auto is None:
                t_first_auto = t
        mode_hist.append(cur)

        m1 = q1 + nq[k][0]
        m2 = q2 + nq[k][1]
        v1 = w1 + nw[k][0]
        v2 = w2 + nw[k][1]

        # reference
        if cur == "autonomous":
            if shared_auto and j >= 0:
                # latest retarget applied as an offset to the stored plan (Eq. 4)
                off = (rh[j][0] - qr[j][0], rh[j][1] - qr[j][1])
                ref = (qr[k][0] + off[0], qr[k][1] + off[1])
                dref = dqr[k]
            else:
                ref, dref = qr[k], dqr[k]
        elif j < 0:
            ref, dref = q0, (0.0, 0.0)                 # no packet yet: hold initial pose
        else:
            ref, dref = rh[j], dh[j]
            if cur == "predictive":                    # Eq. 3
                age = max(0.0, t - j * DT_CTRL)
                ref = (ref[0] + age * dref[0], ref[1] + age * dref[1])

        t1, t2 = pd_ctrl(M, ref, dref, m1, m2, v1, v2, kp, kd)
        sat += abs(t1) >= 0.99 * TMAX[0] or abs(t2) >= 0.99 * TMAX[1]

        for _ in range(CE):
            a1, a2 = R.acc(q1, q2, w1, w2, t1, t2)
            w1 += a1 * DT_PLANT
            w2 += a2 * DT_PLANT
            q1 += w1 * DT_PLANT
            q2 += w2 * DT_PLANT

        # peak Cartesian error within 1 s after a switch
        if last_sw_k >= 0 and k - last_sw_k <= window:
            ee = np.linalg.norm(R.fk(np.array([q1, q2])) - R.fk(S["g"][k]))
            peak_post_sw = max(peak_post_sw, float(ee))

    e = np.linalg.norm(R.fk(np.array(qh)) - R.fk(S["g"]), axis=1)
    fe = float(np.mean(e[-int(1 / DT_CTRL):]))
    out = dict(
        mode=mode,
        Td=float(Td) if profile == "constant" else -1.0,
        profile=profile, seed=seed,
        rmse=float(np.sqrt(np.mean(e**2))),
        final_e=fe,
        success=bool(fe < TOL),
        sat=100.0 * sat / n,
        switches=sw,
        auto_frac=n_auto / n,
        t_first_auto=float(t_first_auto) if t_first_auto is not None else float("nan"),
        peak_post_sw=peak_post_sw,
        prior_err=prior_err if prior_err is not None else -1.0,
    )
    if return_trace:
        out["trace"] = dict(t=np.arange(n) * DT_CTRL, e=e, mode=mode_hist,
                            Td_t=Td_t, est=est_hist, deff=deff_hist)
    return out


def _job(a):
    mode, Td, seed, th, profile, kw = a
    return simulate(mode, Td, seed, th, profile=profile, **kw)


def run_grid(modes, delays, seeds, th=None, profile="constant", **kw):
    seeds = list(seeds)
    jobs = [(m, Td, s, th, profile, kw) for Td in delays for m in modes for s in seeds]
    with Pool(processes=N_PROCS) as p:
        return pd.DataFrame(p.map(_job, jobs, chunksize=8))


# ------------------------------------------------------------ sanity checks
def stability_index():
    R = Arm()
    worst = 0.0
    for q2 in np.linspace(0.8, 2.6, 60):
        Mq = np.array([
            [R.a + 2 * R.b * math.cos(q2), R.c + R.b * math.cos(q2)],
            [R.c + R.b * math.cos(q2), R.c],
        ])
        eig = np.linalg.eigvals(np.linalg.solve(Mq, np.diag(KD)))
        worst = max(worst, DT_CTRL * np.max(np.real(eig)))
    return worst


def sanity():
    print("\n=== Sanity ===")
    idx = stability_index()
    print(f"  damping index = {idx:.2f} (< 2 required)")
    assert idx < 1.9, "unstable gains/period"
    for s in range(3):
        c = simulate("classical", 0, s)
        p = simulate("predictive", 0, s)
        assert abs(c["rmse"] - p["rmse"]) < 1e-9, "classical != predictive at Td=0"
        assert c["sat"] < 3 and c["rmse"] < 0.015, f"zero-delay tracking poor: {c}"
    r0 = simulate("classical", 0, 0)["rmse"]
    c = simulate("classical", 2.0, 0)
    p = simulate("predictive", 2.0, 0)
    assert abs(c["rmse"] - p["rmse"]) > 1e-4, "classical == predictive at Td>0"
    print(f"  Td=0: C=P, rmse={r0:.4f} ok | Td=2: C={c['rmse']:.4f} "
          f"P={p['rmse']:.4f} differ ok")


# ------------------------------------------------------- threshold derivation
def crossing(x, d, persist=PERSIST):
    """
    First sign change of d(x) whose new sign is held for `persist` further
    nonzero samples. Exact zeros (e.g. T = 0, where C = P) are skipped. The
    crossing is linearly interpolated between the bracketing grid points.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    nz = [i for i in range(len(d)) if d[i] != 0.0]
    for i in range(len(nz) - persist - 1):
        signs = [np.sign(d[nz[i + k]]) for k in range(persist + 2)]
        if signs[0] * signs[1] < 0 and all(s == signs[1] for s in signs[1:]):
            a, b = nz[i], nz[i + 1]
            return float(x[a] + (x[b] - x[a]) * d[a] / (d[a] - d[b]))
    return None


def thresholds_from(C, P, A, delays):
    """T_cross: first persistent sign change of RMSE_P - RMSE_C.
       T_A    : first persistent sign change of RMSE_A - min(RMSE_C, RMSE_P)."""
    return (crossing(delays, P - C),
            crossing(delays, A - np.minimum(C, P)))


def _no_ta_reason(C, P, A):
    dA = A - np.minimum(C, P)
    nz = dA[dA != 0]
    if nz.size and np.all(nz < 0):
        return "autonomy always better than best continuous"
    if nz.size and np.all(nz > 0):
        return "autonomy never better than best continuous"
    return "no persistent crossing for T_A"


def _ci(vals):
    if len(vals) == 0:
        return "n/a"
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return f"[{lo:.2f}, {hi:.2f}]"


def _complete(th):
    return th.get("cross") is not None and th.get("auto") is not None


def derive_thresholds(n_ev=N_EV, shift=SHIFT, mass_scale=(1.0, 1.0),
                      kp_scale=1.0, label="nominal", require=True):
    """Calibrate T_cross, T_A and the hysteresis bands on the plant (no prior)."""
    print(f"\n=== Calibration sweep ({label}) ===")
    df = run_grid(FIXED, CAL_DELAYS, range(SEED_CAL, SEED_CAL + N_CAL),
                  n_ev=n_ev, shift=shift, mass_scale=mass_scale,
                  kp_scale=kp_scale, prior_err=None)
    df.to_csv(OUT / f"raw_calibration_{label}.csv", index=False)

    arr = {}
    for m in FIXED:
        sub = df[df["mode"] == m].sort_values(["Td", "seed"])
        arr[m] = sub.rmse.values.reshape(len(CAL_DELAYS), N_CAL)
    mean = {m: v.mean(1) for m, v in arr.items()}
    d = np.array(CAL_DELAYS, float)

    tab = pd.DataFrame({"Td": d})
    for m, short in zip(FIXED, ["C", "P", "A"]):
        tab[f"{short}_rmse"] = mean[m]
        tab[f"{short}_sd"] = arr[m].std(1, ddof=1)
    tab.to_csv(OUT / f"table_calibration_{label}.csv", index=False)
    print(tab.round(4).to_string(index=False))

    cross, ta = thresholds_from(mean["classical"], mean["predictive"],
                                mean["autonomous"], d)
    th = dict(cross=cross, low=None, high=None, auto=ta, auto_exit=None,
              status="ok")

    if cross is None or ta is None:
        reasons = []
        if cross is None:
            reasons.append("no persistent classical/predictive crossover")
        if ta is None:
            reasons.append(_no_ta_reason(mean["classical"], mean["predictive"],
                                         mean["autonomous"]))
        th["status"] = "; ".join(reasons)
        print(f"  {th['status']}")
        if require:
            raise RuntimeError(f"Threshold rule failed ({label}): {th['status']}")
    else:
        rng = np.random.RandomState(0)
        bc, bt = [], []
        for _ in range(BOOT_THR):
            ix = rng.randint(0, N_CAL, N_CAL)
            c_, t_ = thresholds_from(arr["classical"][:, ix].mean(1),
                                     arr["predictive"][:, ix].mean(1),
                                     arr["autonomous"][:, ix].mean(1), d)
            if c_ is not None:
                bc.append(c_)
            if t_ is not None:
                bt.append(t_)

        if len(bc) < 50:
            print("  WARNING: few valid bootstrap crossovers; using a 10% band")
            low, high = 0.90 * cross, 1.10 * cross
        else:
            low = min(float(np.percentile(bc, 2.5)), (1 - BAND_FLOOR) * cross)
            high = max(float(np.percentile(bc, 97.5)), (1 + BAND_FLOOR) * cross)
        if len(bt) < 50:
            print("  WARNING: few valid bootstrap T_A values; using 0.95*T_A exit")
            auto_exit = 0.95 * ta
        else:
            auto_exit = min(float(np.percentile(bt, 2.5)), ta)

        th.update(low=low, high=high, auto_exit=auto_exit)
        print(f"\n  T_cross = {cross:.2f} s  95% CI {_ci(bc)}  "
              f"(no crossing in {BOOT_THR - len(bc)}/{BOOT_THR} resamples)")
        print(f"  T_A     = {ta:.2f} s  95% CI {_ci(bt)}  "
              f"(no crossing in {BOOT_THR - len(bt)}/{BOOT_THR} resamples)")
        print(f"  T_low = {low:.2f}  T_high = {high:.2f}  T_A,exit = {auto_exit:.2f}")
        if ta < high:
            print("  T_A < T_high: the classical-only band is empty; the "
                  "supervisor moves only between P and A.")

    pd.Series({k: (np.nan if v is None else v) for k, v in th.items()}).to_csv(
        OUT / f"thresholds_{label}.csv", header=["value"])

    # calibration figure
    fig, ax = plt.subplots(figsize=(FIG_W, 2.4))
    for m, st, lab in [("classical", "o-", "C"), ("predictive", "s-", "P"),
                       ("autonomous", "^-", "A")]:
        se = arr[m].std(1, ddof=1) / np.sqrt(N_CAL) * 1.96
        ax.errorbar(d, mean[m], se, fmt=st, capsize=1.5, ms=3, lw=1, label=lab)
    for v, lb in [(th["auto"], r"$T_A$"), (th["cross"], r"$T_{cross}$")]:
        if v is not None:
            ax.axvline(v, ls=":", c="k", lw=0.8)
            ax.text(v, ax.get_ylim()[1] * 0.92, " " + lb, fontsize=FONT_SIZE - 1)
    ax.set_xlabel(r"One-way delay $T_d$ (s)")
    ax.set_ylabel("RMSE to true goal (m)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.5, 23)
    plt.tight_layout()
    plt.savefig(OUT / f"fig_calibration_{label}.pdf")
    plt.close()
    return th, arr, mean, d


# ---------------------------------------------------------------- statistics
def rel_regret_ci(sup, best, key, n_boot=BOOT_REG):
    """Relative regret (%) of `sup` against `best` and its 95% percentile
    bootstrap interval, resampling seeds in pairs. `key` seeds the resampling so
    that all variants at one delay share the same resamples."""
    rng = np.random.RandomState(key)
    n = len(sup)
    rels = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.randint(0, n, n)
        rels[i] = 100.0 * (sup[ix] - best[ix]).mean() / (best[ix].mean() + 1e-12)
    lo, hi = np.percentile(rels, [2.5, 97.5])
    rel = 100.0 * (sup - best).mean() / (best.mean() + 1e-12)
    return float(rel), float(lo), float(hi)


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (m - rank) * p[idx]))
        adj[idx] = run
    return adj


def _f(x):
    x = float(x)
    return None if math.isnan(x) else x


# ------------------------------------------------------- held-out evaluation
def heldout(th, prior_err=None, label="primary", full=True, **scen):
    """
    Compare the supervisor with the best fixed controller on held-out seeds and
    delays. With full=True all supervisor variants are run; with full=False only
    the fixed controllers and the base supervisor are run (used for the
    scenario-family and plant sweeps). `scen` passes n_ev, shift, mass_scale or
    kp_scale to the simulator.
    """
    print(f"\n=== Held-out evaluation ({label}, prior_err={prior_err}) ===")
    modes = FIXED + ["supervisor"]
    if full:
        modes += ["supervisor_age", "supervisor_oracle", "supervisor_hand",
                  "supervisor_nohyst"]
    df = run_grid(modes, HOLD_DELAYS, range(SEED_HOLD, SEED_HOLD + N_HOLD), th,
                  prior_err=prior_err, **scen)
    df.to_csv(OUT / f"raw_heldout_{label}.csv", index=False)

    rows = []
    for Td in HOLD_DELAYS:
        sub = df[df.Td == Td]

        def get(name, sub=sub):
            return sub[sub["mode"] == name].sort_values("seed")

        mu = {m: get(m).rmse.mean() for m in FIXED}
        bm = min(mu, key=mu.get)         # best fixed controller, chosen on held-out data
        b = get(bm).rmse.values
        key = int(round(Td * 1000))
        row = {"Td": Td, "Td_norm": Td / TASK_TIME,
               "C": mu["classical"], "P": mu["predictive"], "A": mu["autonomous"],
               "BestFixed": bm, "BestRMSE": mu[bm],
               "Succ_Best_%": 100 * get(bm).success.mean()}
        for vk, name in VARIANTS.items():
            if name not in modes:
                continue
            gv = get(name)
            s = gv.rmse.values
            rel, lo, hi = rel_regret_ci(s, b, key)
            row.update({
                f"{vk}_rmse": s.mean(), f"{vk}_rel": rel,
                f"{vk}_lo": lo, f"{vk}_hi": hi,
                f"{vk}_equiv": bool(hi <= EQUIV_MARGIN),
                f"{vk}_switches": gv.switches.mean(),
                f"{vk}_first_auto": gv.t_first_auto.mean(),
            })
        gs = get("supervisor")
        s = gs.rmse.values
        row["Sup_ci95"] = 1.96 * s.std(ddof=1) / np.sqrt(len(s))
        if np.all(s - b == 0):
            row["p_wilcoxon"] = 1.0
        else:
            try:
                row["p_wilcoxon"] = float(stats.wilcoxon(s, b).pvalue)
            except ValueError:
                row["p_wilcoxon"] = 1.0
        row["Succ_Sup_%"] = 100 * gs.success.mean()
        row["Sat_Sup_%"] = gs.sat.mean()
        row["AutoFrac"] = gs.auto_frac.mean()
        rows.append(row)

    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p_wilcoxon"].values)
    t.to_csv(OUT / f"table_heldout_{label}.csv", index=False)
    print(t[["Td", "C", "P", "A", "Sup_rmse", "BestRMSE", "Sup_rel", "Sup_hi",
             "Sup_equiv", "Sup_switches"]].round(4).to_string(index=False))

    sup_hi_delay = df[(df["mode"] == "supervisor") & (df.Td >= 11.0)]
    summary = {
        "label": label, "prior_err": prior_err, "n_delays": int(len(t)),
        "n_equiv": int(t["Sup_equiv"].sum()),
        "mean_rel_regret_pct": float(t["Sup_rel"].mean()),
        "max_upper_ci_pct": float(t["Sup_hi"].max()),
        "mean_switches": float(t["Sup_switches"].mean()),
        "n_holm_significant": int((t["p_holm"] < 0.05).sum()),
        "first_auto_ge11_s": _f(sup_hi_delay.t_first_auto.mean()),
    }
    if full:
        age_hi_delay = df[(df["mode"] == "supervisor_age") & (df.Td >= 11.0)]
        switched = df[(df["mode"] == "supervisor") & (df.switches > 0)]
        summary.update({
            "age_n_equiv": int(t["Age_equiv"].sum()),
            "age_mean_rel_regret_pct": float(t["Age_rel"].mean()),
            "age_max_upper_ci_pct": float(t["Age_hi"].max()),
            "age_first_auto_ge11_s": _f(age_hi_delay.t_first_auto.mean()),
            "peak_post_switch_m": _f(switched.peak_post_sw.mean()),
        })
    with open(OUT / f"summary_heldout_{label}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Equivalence (upper 95% CI <= {EQUIV_MARGIN:.0f}%): "
          f"{summary['n_equiv']}/{summary['n_delays']} delays | "
          f"mean regret {summary['mean_rel_regret_pct']:.2f}% | "
          f"max upper CI {summary['max_upper_ci_pct']:.1f}%")

    fig, ax = plt.subplots(figsize=(FIG_W, 2.6))
    for k, st in [("C", "o-"), ("P", "s-"), ("A", "^-")]:
        ax.plot(t.Td, t[k], st, ms=3, lw=1, label=k)
    ax.errorbar(t.Td, t.Sup_rmse, t.Sup_ci95, fmt="D-", c="k", ms=3, lw=1,
                capsize=1.5, label="Supervisor")
    ax.plot(t.Td, t.BestRMSE, "k--", lw=1, label="Best fixed")
    for v in [th.get("auto"), th.get("cross")]:
        if v is not None:
            ax.axvline(v, ls=":", c="gray", lw=0.8)
    ax.set_xlabel(r"One-way delay $T_d$ (s)")
    ax.set_ylabel("RMSE to true goal (m)")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"fig_heldout_{label}.pdf")
    plt.close()
    return t, summary


def variants_table(t, summary):
    """Supervisor variants averaged over the held-out delays."""
    labels = {"Sup": "Measured thresholds", "Hand": "Hand-picked thresholds",
              "Oracle": "Oracle delay", "NoHyst": "No hysteresis / dwell",
              "Age": "Age-aware"}
    rows = []
    for k, lab in labels.items():
        if f"{k}_rmse" not in t:
            continue
        rows.append({"variant": lab,
                     "mean_rmse_m": t[f"{k}_rmse"].mean(),
                     "mean_rel_regret_pct": t[f"{k}_rel"].mean(),
                     "max_upper_ci_pct": t[f"{k}_hi"].max(),
                     "switches_per_run": t[f"{k}_switches"].mean()})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table_variants.csv", index=False)
    print("\n=== Supervisor variants (Table III) ===")
    print(tab.round(4).to_string(index=False))
    if summary.get("peak_post_switch_m") is not None:
        print(f"  mean peak error within 1 s after a switch (runs that switched): "
              f"{summary['peak_post_switch_m']:.3f} m")
    return tab


def prior_sweep(th):
    """Held-out regret for each prior error on the initial delay (Table IV)."""
    print("\n=== Prior-error sweep ===")
    results, rows = {}, []
    for pe in PRIOR_ERRS:
        lab = "prior_none" if pe is None else f"prior_{pe}"
        t, s = heldout(th, prior_err=pe, label=lab, full=True)
        results[pe] = (t, s)
        rows.append({
            "prior_err": "none" if pe is None else pe,
            "mean_rel_regret_pct": s["mean_rel_regret_pct"],
            "max_upper_ci_pct": s["max_upper_ci_pct"],
            "n_equiv": s["n_equiv"],
            "first_auto_ge11_s": s["first_auto_ge11_s"],
            "age_mean_rel_regret_pct": s["age_mean_rel_regret_pct"],
            "age_n_equiv": s["age_n_equiv"],
            "age_first_auto_ge11_s": s["age_first_auto_ge11_s"],
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table_prior_sweep.csv", index=False)
    print("\n=== Prior sweep summary (Table IV) ===")
    print(tab.round(3).to_string(index=False))
    return results


# --------------------------------------------------- time-varying delay tests
def _mode_ints(modes):
    mm = {"classical": 0, "predictive": 1, "autonomous": 2}
    return np.array([mm.get(m, -1) for m in modes])


def plot_profile_trace(prof, th, trb, tra):
    t = trb["t"]
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 3.9), sharex=True)
    axes[0].plot(t, trb["Td_t"], "k-", lw=1, label=r"$T_d(t)$")
    axes[0].plot(t, trb["est"], "-", c="C1", lw=1, label=r"$\hat d$")
    axes[0].axhline(th["cross"], ls=":", c="C0", lw=0.8, label=r"$T_{cross}$")
    axes[0].axhline(th["auto"], ls=":", c="C2", lw=0.8, label=r"$T_A$")
    axes[0].set_ylabel("delay (s)")
    axes[0].legend(loc="upper left", frameon=False, ncol=4)
    axes[1].step(t, _mode_ints(trb["mode"]), where="post", c="k", lw=1,
                 label="base")
    axes[1].step(t, _mode_ints(tra["mode"]), where="post", c="C3", lw=1,
                 ls="--", label="age-aware")
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["C", "P", "A"])
    axes[1].set_ylabel("mode")
    axes[1].legend(loc="center left", frameon=False)
    axes[2].plot(t, trb["e"], "k-", lw=0.7, label="base")
    axes[2].plot(t, tra["e"], "-", c="C3", lw=0.7, label="age-aware")
    axes[2].set_ylabel("error (m)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"fig_timevarying_{prof}.pdf")
    plt.close()


def time_varying(th, prior_err=PRIMARY_PRIOR):
    """Ramp, step and sinusoid delay profiles (Table V, Fig. 4)."""
    print("\n=== Time-varying delay profiles ===")
    modes = FIXED + ["supervisor", "supervisor_age"]
    rows = []
    for prof in ["ramp", "step", "sinusoid"]:
        print(f"  profile = {prof}")
        df = run_grid(modes, [-1], range(SEED_HOLD, SEED_HOLD + N_HOLD), th,
                      profile=prof, prior_err=prior_err)

        def get(name, df=df):
            return df[df["mode"] == name].sort_values("seed")

        mu = {m: get(m).rmse.mean() for m in FIXED}
        bm = min(mu, key=mu.get)
        b = get(bm).rmse.values
        row = {"profile": prof, "C": mu["classical"], "P": mu["predictive"],
               "A": mu["autonomous"], "BestFixed": bm, "BestRMSE": mu[bm]}
        for vk, name in [("Sup", "supervisor"), ("Age", "supervisor_age")]:
            gv = get(name)
            rel, lo, hi = rel_regret_ci(gv.rmse.values, b, 12345)
            row.update({f"{vk}_rmse": gv.rmse.mean(), f"{vk}_rel": rel,
                        f"{vk}_lo": lo, f"{vk}_hi": hi,
                        f"{vk}_switches": gv.switches.mean(),
                        f"{vk}_auto_frac": gv.auto_frac.mean(),
                        f"{vk}_first_auto": gv.t_first_auto.mean()})
        rows.append(row)

        trb = simulate("supervisor", -1, SEED_HOLD, th, profile=prof,
                       prior_err=prior_err, return_trace=True)["trace"]
        tra = simulate("supervisor_age", -1, SEED_HOLD, th, profile=prof,
                       prior_err=prior_err, return_trace=True)["trace"]
        plot_profile_trace(prof, th, trb, tra)

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table_timevarying.csv", index=False)
    print(tab.round(4).to_string(index=False))
    return tab


# --------------------------------------------------------------- chattering
def chattering_test(th, prior_err=PRIMARY_PRIOR):
    """Constant delay at T_A, with and without hysteresis and dwell."""
    print("\n=== Chattering test (Td = T_A) ===")
    df = run_grid(["supervisor", "supervisor_nohyst"], [th["auto"]],
                  range(SEED_HOLD, SEED_HOLD + N_HOLD), th, prior_err=prior_err)
    rows = []
    for name in ["supervisor", "supervisor_nohyst"]:
        sub = df[df["mode"] == name]
        rows.append({"mode": name, "Td": th["auto"],
                     "mean_switches": sub.switches.mean(),
                     "mean_rmse_m": sub.rmse.mean()})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table_chattering.csv", index=False)
    print(tab.round(4).to_string(index=False))
    return tab


# ------------------------------------------- scenario-family and plant sweeps
def scenario_family_sweep(th_nom, summary_nom):
    """Recalibrate under five event families; evaluate with each family's own
    thresholds (Section VI-G)."""
    print("\n=== Scenario-family sensitivity ===")
    families = [
        dict(n_ev=0, shift=0.0, label="no_shift"),
        dict(n_ev=1, shift=0.15, label="1x0.15"),
        dict(n_ev=1, shift=0.30, label="1x0.30"),
        dict(n_ev=2, shift=0.30, label="2x0.30"),
        dict(n_ev=2, shift=0.45, label="2x0.45"),
    ]
    rows = []
    for fam in families:
        nominal = (fam["n_ev"] == N_EV and fam["shift"] == SHIFT)
        if nominal:
            th, s = th_nom, summary_nom
        else:
            th, _, _, _ = derive_thresholds(n_ev=fam["n_ev"], shift=fam["shift"],
                                            label=fam["label"], require=False)
            s = None
            if _complete(th):
                _, s = heldout(th, prior_err=PRIMARY_PRIOR,
                               label=f"family_{fam['label']}", full=False,
                               n_ev=fam["n_ev"], shift=fam["shift"])
        rows.append({
            "family": fam["label"], "n_ev": fam["n_ev"], "shift_rad": fam["shift"],
            "cross": th.get("cross"), "auto": th.get("auto"),
            "low": th.get("low"), "high": th.get("high"),
            "auto_exit": th.get("auto_exit"), "status": th.get("status", "ok"),
            "own_mean_rel_regret_pct": None if s is None else s["mean_rel_regret_pct"],
            "own_n_equiv": None if s is None else s["n_equiv"],
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table_scenario_families.csv", index=False)
    print(tab.round(3).to_string(index=False))
    return tab


def plant_sensitivity(th_nom, summary_nom):
    """Recalibrate on perturbed plants and compare own thresholds with the
    nominal thresholds (Table VI)."""
    print("\n=== Plant sensitivity ===")
    plants = [
        dict(mass_scale=(1.25, 1.25), kp_scale=1.0, label="mass+25"),
        dict(mass_scale=(0.75, 0.75), kp_scale=1.0, label="mass-25"),
        dict(mass_scale=(1.0, 1.0), kp_scale=0.8, label="gains-20"),
    ]
    rows = [{"plant": "nominal", "cross": th_nom["cross"], "auto": th_nom["auto"],
             "own_mean_rel_regret_pct": summary_nom["mean_rel_regret_pct"],
             "own_n_equiv": summary_nom["n_equiv"],
             "nom_mean_rel_regret_pct": summary_nom["mean_rel_regret_pct"],
             "nom_n_equiv": summary_nom["n_equiv"]}]
    for pl in plants:
        scen = dict(mass_scale=pl["mass_scale"], kp_scale=pl["kp_scale"])
        th, _, _, _ = derive_thresholds(label=pl["label"], require=False, **scen)
        row = {"plant": pl["label"], "cross": th.get("cross"), "auto": th.get("auto")}
        if _complete(th):
            _, s_own = heldout(th, prior_err=PRIMARY_PRIOR,
                               label=f"plant_{pl['label']}_own", full=False, **scen)
            _, s_nom = heldout(th_nom, prior_err=PRIMARY_PRIOR,
                               label=f"plant_{pl['label']}_nominal_th", full=False,
                               **scen)
            row.update({"own_mean_rel_regret_pct": s_own["mean_rel_regret_pct"],
                        "own_n_equiv": s_own["n_equiv"],
                        "nom_mean_rel_regret_pct": s_nom["mean_rel_regret_pct"],
                        "nom_n_equiv": s_nom["n_equiv"]})
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table_plant_sensitivity.csv", index=False)
    print(tab.round(3).to_string(index=False))
    return tab


# --------------------------------------------------------------- config dump
def dump_config(th):
    cfg = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "parameters": {
            "DT_PLANT": DT_PLANT, "DT_CTRL": DT_CTRL, "KP": KP, "KD": KD,
            "TMAX": TMAX, "N_WP": N_WP, "TASK_TIME": TASK_TIME,
            "DEADLINE": DEADLINE, "KO": KO, "JIT": JIT, "NQ": NQ, "NDQ": NDQ,
            "MODEL_UNC": MODEL_UNC, "N_EV": N_EV, "SHIFT": SHIFT,
            "SHIFT_RAMP": SHIFT_RAMP, "MIN_DWELL_DEFAULT": MIN_DWELL_DEFAULT,
            "EWMA_DEFAULT": EWMA_DEFAULT, "TOL": TOL, "N_CAL": N_CAL,
            "N_HOLD": N_HOLD, "SEED_CAL": SEED_CAL, "SEED_HOLD": SEED_HOLD,
            "PERSIST": PERSIST, "BAND_FLOOR": BAND_FLOOR,
            "BOOT_THR": BOOT_THR, "BOOT_REG": BOOT_REG,
            "EQUIV_MARGIN": EQUIV_MARGIN, "PRIMARY_PRIOR": PRIMARY_PRIOR,
            "CAL_DELAYS": CAL_DELAYS, "HOLD_DELAYS": HOLD_DELAYS,
            "PRIOR_ERRS": PRIOR_ERRS, "HAND_THRESHOLDS": HAND_THRESHOLDS,
        },
        "thresholds_nominal": th,
    }
    with open(OUT / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Config written to {OUT / 'config.json'}")


# ---------------------------------------------------------------- smoke test
def smoke():
    """One seed, every mode, provisional thresholds. Checks that all code paths
    execute. The thresholds below are the values reported in the manuscript and
    are used here only for this check, not by the full pipeline."""
    sanity()
    th = dict(cross=4.64, low=4.11, high=5.55, auto=3.19, auto_exit=2.98)
    print("\n=== Smoke test: Td = 12 s, seed 100, 10% prior ===")
    for m in FIXED + list(VARIANTS.values()):
        r = simulate(m, 12.0, SEED_HOLD, th, prior_err=PRIMARY_PRIOR)
        print(f"  {m:18s} rmse={r['rmse']:.4f} switches={r['switches']} "
              f"first_auto={r['t_first_auto']:.1f}")
    print("\n=== Smoke test: step profile, seed 100 ===")
    for m in ["supervisor", "supervisor_age"]:
        r = simulate(m, -1, SEED_HOLD, th, profile="step", prior_err=PRIMARY_PRIOR)
        print(f"  {m:18s} rmse={r['rmse']:.4f} switches={r['switches']} "
              f"first_auto={r['t_first_auto']:.1f}")
    print("\nSmoke test finished.")


# ---------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="run a one-seed execution check and exit")
    ap.add_argument("--processes", type=int, default=None,
                    help="number of worker processes (default: all cores)")
    args = ap.parse_args()
    N_PROCS = args.processes

    if args.smoke:
        smoke()
        sys.exit(0)

    sanity()
    th_nom, arr, mean, d = derive_thresholds(label="nominal", require=True)
    dump_config(th_nom)

    sweep = prior_sweep(th_nom)                      # Table IV, includes the primary condition
    t_primary, s_primary = sweep[PRIMARY_PRIOR]      # Table II, Fig. 3
    variants_table(t_primary, s_primary)             # Table III

    time_varying(th_nom)                             # Table V, Fig. 4
    chattering_test(th_nom)                          # Section VI-F
    scenario_family_sweep(th_nom, s_primary)         # Section VI-G
    plant_sensitivity(th_nom, s_primary)             # Table VI

    print(f"\nAll results in {OUT.resolve()}")
