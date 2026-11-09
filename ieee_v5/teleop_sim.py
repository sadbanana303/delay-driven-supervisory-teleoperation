#!/usr/bin/env python3
"""
Delay-adaptive supervisor v5.2  (2-DoF nonlinear RR arm)

Controllers: classical delayed | predictive (constant-velocity, timestamped) | autonomous (staged plan)
Supervisor : switches on measured delay, thresholds derived from plant data by a stated rule.
Task       : streaming min-jerk path, closed-loop operator, unplanned goal shift the staged plan cannot see.

Changes relative to v5.1:
- Fixed crossing() so that an exact zero at Td=0 (where classical ≡ predictive by design)
  is no longer treated as a crossover. Now skips zeros and finds the first true sign change
  between consecutive nonzero samples (linearly interpolated). This restores the correct
  crossover (~4.7 s) and T_MARS (~12.3 s).
- Docstring clarified: returns the *first* crossing only (matching the stated rule).
- Age clamp and IK numerical hardening retained (harmless / slightly safer).
- ASCII-only note text (Windows cp1252 safe).
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path
from scipy import stats

# ---------------- parameters ----------------
DT_PLANT, DT_CTRL = 0.002, 0.010
CE = int(round(DT_CTRL / DT_PLANT))
KP, KD, TMAX = (28.0, 18.0), (3.8, 2.7), (16.0, 10.0)
N_WP, TASK_TIME, DEADLINE = 4, 28.0, 30.0        # one fixed mission deadline for every run
KO = 0.3                                          # operator gain on the stale error he sees
JIT = 0.10                                        # one-way delay jitter, fraction of Td (per packet)
NQ, NDQ, MODEL_UNC = 5e-4, 5e-3, 0.10             # encoder noise [rad], velocity noise [rad/s], mass error
N_EV, SHIFT, SHIFT_RAMP = 2, 0.30, 4.0            # unplanned goal retargets per mission, size [rad], ramp [s]
MIN_DWELL, BAND, EWMA, TOL = 2.0, 0.10, 0.10, 0.02
N_CAL, N_HOLD, SEED_CAL, SEED_HOLD = 30, 30, 0, 100
CAL_DELAYS = [0, 0.5, 1, 2, 3, 4, 6, 8, 10, 13, 17, 22]
HOLD_DELAYS = [0, 0.7, 1.5, 3, 5, 7, 9, 12, 15, 22]   # off the calibration grid
OUT = Path("ieee_v5"); OUT.mkdir(exist_ok=True)

# ---------------- plant ----------------
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
            self.l1 * np.sin(q[..., 0]) + self.l2 * np.sin(q[..., 0] + q[..., 1])
        ], -1)


def ik(r, x, y):
    c2 = np.clip((x * x + y * y - r.l1**2 - r.l2**2) / (2 * r.l1 * r.l2), -1.0, 1.0)
    s2 = np.sqrt(np.maximum(0.0, 1.0 - c2 * c2))
    q2 = math.atan2(s2, c2)
    return np.array([
        math.atan2(y, x) - math.atan2(r.l2 * s2, r.l1 + r.l2 * c2),
        q2
    ])


def pd_ctrl(M, ref, dref, q1, q2, w1, w2):
    g1, g2 = M.grav(q1, q2)
    t1 = KP[0] * (ref[0] - q1) + KD[0] * (dref[0] - w1) + g1
    t2 = KP[1] * (ref[1] - q2) + KD[1] * (dref[1] - w2) + g2
    return (max(-TMAX[0], min(TMAX[0], t1)),
            max(-TMAX[1], min(TMAX[1], t2)))


# ---------------- scenario (common random numbers across modes) ----------------
def scenario(seed, Td):
    rng = np.random.RandomState(seed + 7919)
    R = Arm()
    xs = np.linspace(0.27, 0.43, N_WP) + 0.01 * rng.randn(N_WP)
    ys = 0.17 + 0.10 * np.sin(np.linspace(0, np.pi, N_WP)) + 0.008 * rng.randn(N_WP)
    qw = np.array([ik(R, x, y) for x, y in np.clip(np.c_[xs, ys], 0.18, 0.52)])
    n = int(round(DEADLINE / DT_CTRL))
    t = np.arange(n) * DT_CTRL
    seg = TASK_TIME / (N_WP - 1)
    k = np.minimum((t // seg).astype(int), N_WP - 2)
    u = np.minimum((t - k * seg) / seg, 1.0)
    s = (10 * u**3 - 15 * u**4 + 6 * u**5)[:, None]
    qref = qw[k] + (qw[k + 1] - qw[k]) * s                         # staged plan (holds last waypoint)
    g = qref.copy()                                                 # true goal = plan + unplanned retargets
    for e in range(N_EV):
        ts = rng.uniform(0.15 + 0.40 * e, 0.45 + 0.40 * e) * TASK_TIME
        th = rng.uniform(0, 2 * np.pi)
        v = np.clip((t - ts) / SHIFT_RAMP, 0, 1)
        sm = (3 * v**2 - 2 * v**3)[:, None]
        g += sm * SHIFT * np.array([math.cos(th), math.sin(th)])
    ul = np.maximum(0.0, Td * (1 + JIT * rng.randn(n)))
    dl = np.maximum(0.0, Td * (1 + JIT * rng.randn(n)))
    a = np.ceil((t + ul) / DT_CTRL - 1e-9).astype(int)
    best = np.full(n, -1)
    m = a < n
    np.maximum.at(best, a[m], np.arange(n)[m])
    best = np.maximum.accumulate(best)
    nq = NQ * rng.randn(n, 2)
    nw = NDQ * rng.randn(n, 2)
    mu = np.clip(MODEL_UNC * rng.randn(2), -0.25, 0.25)
    return dict(
        R=R,
        M=Arm(1.5 * (1 + mu[0]), 1.0 * (1 + mu[1])),
        n=n,
        g=g,
        qref=qref,
        dqref=np.gradient(qref, DT_CTRL, axis=0),
        ul=ul,
        dl=dl,
        best=best,
        nq=nq,
        nw=nw,
        est0=max(0.0, Td * (1 + 0.1 * rng.randn())),
        q0=qw[0] + np.array([0.025, -0.015])
    )


def choose(est, cur, th):
    """Hysteresis switching with dwell already enforced by caller."""
    cr, tl, thi, tm = th["cross"], th["low"], th["high"], th["mars"]
    if cur is None:
        return "autonomous" if est >= tm else ("classical" if est > cr else "predictive")
    if cur == "autonomous":
        if est > tm * (1 - BAND):
            return "autonomous"
        return "classical" if est > thi else "predictive"
    if est >= tm:
        return "autonomous"
    if cur == "predictive":
        return "classical" if est > thi else "predictive"
    # classical
    return "predictive" if est < tl else "classical"


# ---------------- one run: mode in classical|predictive|autonomous|supervisor ----------------
def simulate(mode, Td, seed, th=None):
    S = scenario(seed, Td)
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
    q1, q2 = q0
    w1 = w2 = 0.0
    cur = None if mode == "supervisor" else mode
    est = S["est0"]
    last_j = -1
    t_sw = 0.0
    sw = 0
    n_auto = 0
    sat = 0
    qh, rh, dh = [], [], []
    dg = np.gradient(S["g"], DT_CTRL, axis=0).tolist()   # operator's intended goal velocity

    for k in range(n):
        t = k * DT_CTRL
        qh.append((q1, q2))

        # operator (Earth): sees robot state ~Td-old, commands goal + correction of the stale error
        sidx = max(0, k - int(round(dl[k] / DT_CTRL)))
        r = (
            g[k][0] + KO * (g[sidx][0] - qh[sidx][0]),
            g[k][1] + KO * (g[sidx][1] - qh[sidx][1])
        )
        rh.append(r)
        dh.append(dg[k])               # packet carries command and its velocity

        j = best[k]                    # newest packet that has arrived
        if j != last_j and j >= 0:
            est += EWMA * (ul[j] - est)
            last_j = j                 # supervisor's measured one-way delay

        if mode == "supervisor":
            if cur is None:
                cur = choose(est, None, th)
            elif t - t_sw >= MIN_DWELL:
                new = choose(est, cur, th)
                if new != cur:
                    cur, t_sw, sw = new, t, sw + 1

        n_auto += (cur == "autonomous")

        m1 = q1 + nq[k][0]
        m2 = q2 + nq[k][1]
        v1 = w1 + nw[k][0]
        v2 = w2 + nw[k][1]

        if cur == "autonomous":
            ref, dref = qr[k], dqr[k]
        elif j < 0:
            ref, dref = q0, (0.0, 0.0)
        else:
            ref, dref = rh[j], dh[j]
            if cur == "predictive":
                # packet timestamp -> constant-velocity extrapolation (age clamped >= 0)
                age = max(0.0, t - j * DT_CTRL)
                ref = (ref[0] + age * dref[0], ref[1] + age * dref[1])

        t1, t2 = pd_ctrl(M, ref, dref, m1, m2, v1, v2)
        sat += abs(t1) >= 0.99 * TMAX[0] or abs(t2) >= 0.99 * TMAX[1]

        for _ in range(CE):
            a1, a2 = R.acc(q1, q2, w1, w2, t1, t2)
            w1 += a1 * DT_PLANT
            w2 += a2 * DT_PLANT
            q1 += w1 * DT_PLANT
            q2 += w2 * DT_PLANT

    e = np.linalg.norm(R.fk(np.array(qh)) - R.fk(S["g"]), axis=1)    # error to the TRUE goal
    fe = float(np.mean(e[-int(1 / DT_CTRL):]))
    return dict(
        mode=mode, Td=Td, seed=seed,
        rmse=float(np.sqrt(np.mean(e**2))),
        final_e=fe,
        success=fe < TOL,
        sat=100. * sat / n,
        switches=sw,
        auto_frac=n_auto / n
    )


def _job(a):
    return simulate(*a)


def run_grid(modes, delays, seeds, th=None):
    jobs = [(m, Td, s, th) for Td in delays for m in modes for s in seeds]
    with Pool() as p:
        return pd.DataFrame(p.map(_job, jobs, chunksize=8))


# ---------------- stability + sanity ----------------
def stability_index():
    R = Arm()
    worst = 0.0
    for q2 in np.linspace(0.8, 2.6, 60):        # joint-2 range covered by the workspace
        Mq = np.array([
            [R.a + 2 * R.b * math.cos(q2), R.c + R.b * math.cos(q2)],
            [R.c + R.b * math.cos(q2), R.c]
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
    print(f"  Td=0: C=P, rmse={r0:.4f} ok | Td=2: C={c['rmse']:.4f} P={p['rmse']:.4f} differ ok")


# ---------------- threshold derivation ----------------
def crossing(x, d):
    """
    First sign change of d(x) between consecutive nonzero samples,
    linearly interpolated. Returns None if absent.
    Exact zeros (e.g. classical == predictive at Td=0) are skipped.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    nz = [i for i in range(len(d)) if d[i] != 0.0]
    for a, b in zip(nz, nz[1:]):
        if d[a] * d[b] < 0.0:
            return float(x[a] + (x[b] - x[a]) * d[a] / (d[a] - d[b]))
    return None


def thresholds_from(C, P, A, delays):
    """
    Stated rule:
      cross  = first delay where predictive becomes worse than classical (P - C changes sign)
      tmars  = first delay where autonomous becomes better than the best continuous controller
    """
    cr = crossing(delays, P - C)                       # P-C: <0 where predictive is better
    tm = crossing(delays, A - np.minimum(C, P))        # A-best: >0 where continuous is better
    return cr, tm


def derive_thresholds():
    print("\n=== Calibration sweep ===")
    df = run_grid(
        ["classical", "predictive", "autonomous"],
        CAL_DELAYS,
        range(SEED_CAL, SEED_CAL + N_CAL)
    )
    df.to_csv(OUT / "raw_calibration.csv", index=False)

    piv = df.pivot_table(index="Td", columns="mode", values="rmse", aggfunc=list)
    arr = {
        m: np.array([piv.loc[Td, m] for Td in CAL_DELAYS])
        for m in ["classical", "predictive", "autonomous"]
    }
    mean = {m: v.mean(1) for m, v in arr.items()}
    d = np.array(CAL_DELAYS, float)

    cross, tmars = thresholds_from(
        mean["classical"], mean["predictive"], mean["autonomous"], d
    )

    tab = pd.DataFrame({
        "Td": d,
        **{m[0].upper() + "_rmse": mean[m] for m in mean},
        **{m[0].upper() + "_sd": arr[m].std(1, ddof=1) for m in arr}
    })
    tab.to_csv(OUT / "table_calibration.csv", index=False)
    print(tab.round(4).to_string(index=False))

    if cross is None:
        raise RuntimeError("No classical/predictive crossover in the calibration grid")
    if tmars is None:
        raise RuntimeError("No autonomy crossover (autonomy never/always better) - T_MARS undefined")

    # bootstrap over seeds for confidence intervals
    rng = np.random.RandomState(0)
    bc, bm = [], []
    for _ in range(500):
        ix = rng.randint(0, N_CAL, N_CAL)
        c, m_ = thresholds_from(
            arr["classical"][:, ix].mean(1),
            arr["predictive"][:, ix].mean(1),
            arr["autonomous"][:, ix].mean(1),
            d
        )
        if c is not None:
            bc.append(c)
        if m_ is not None:
            bm.append(m_)

    th = dict(
        cross=cross,
        low=(1 - BAND) * cross,
        high=(1 + BAND) * cross,
        mars=tmars
    )

    print(f"\n  crossover C/P = {cross:.2f}s  95% CI [{np.percentile(bc, 2.5):.2f}, {np.percentile(bc, 97.5):.2f}]")
    print(f"  T_MARS (A=best continuous) = {tmars:.2f}s  95% CI [{np.percentile(bm, 2.5):.2f}, {np.percentile(bm, 97.5):.2f}]")
    print(f"  T_low={th['low']:.2f}  T_high={th['high']:.2f}")

    if tmars <= th["high"]:
        print("  NOTE: T_MARS <= T_high -> classical-only band is empty; "
              "supervisor will transition predictive <-> autonomous near the Mars threshold.")

    # persist thresholds for reproducibility
    pd.Series(th).to_csv(OUT / "thresholds.csv", header=["value"])
    print(f"  Thresholds saved to {OUT / 'thresholds.csv'}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for m, st in [("classical", "o-"), ("predictive", "s-"), ("autonomous", "^-")]:
        ax.errorbar(
            d, mean[m],
            arr[m].std(1, ddof=1) / np.sqrt(N_CAL) * 1.96,
            fmt=st, capsize=2, label=m
        )
    for v, lb in [(cross, "C/P crossover"), (tmars, "T_MARS")]:
        ax.axvline(v, ls=":", c="k")
        ax.text(v, ax.get_ylim()[1] * 0.9, " " + lb)
    ax.set_xlabel("One-way delay Td (s)")
    ax.set_ylabel("RMSE to true goal (m)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_calibration.pdf")
    plt.close()
    return th


# ---------------- held-out evaluation ----------------
def heldout(th):
    print("\n=== Held-out evaluation (unseen seeds and delays) ===")
    modes = ["classical", "predictive", "autonomous", "supervisor"]
    df = run_grid(modes, HOLD_DELAYS, range(SEED_HOLD, SEED_HOLD + N_HOLD), th)
    df.to_csv(OUT / "raw_heldout.csv", index=False)

    rows = []
    for Td in HOLD_DELAYS:
        sub = df[df.Td == Td]
        get = lambda m: sub[sub["mode"] == m].sort_values("seed")
        mu = {m: get(m).rmse.mean() for m in modes[:3]}
        bm = min(mu, key=mu.get)
        b = get(bm).rmse.values
        s = get("supervisor").rmse.values
        diff = s - b
        ci = stats.t.ppf(0.975, N_HOLD - 1) * diff.std(ddof=1) / np.sqrt(N_HOLD)
        if np.all(diff == 0):
            p = 1.0
        else:
            p = stats.wilcoxon(s, b).pvalue
        rows.append({
            "Td": Td,
            "C": mu["classical"],
            "P": mu["predictive"],
            "A": mu["autonomous"],
            "Sup": s.mean(),
            "Sup_ci95": 1.96 * s.std(ddof=1) / np.sqrt(N_HOLD),
            "BestFixed": bm,
            "Oracle": mu[bm],
            "Regret": diff.mean(),
            "Regret_ci95": ci,
            "RelRegret_%": 100 * diff.mean() / mu[bm],
            "p_wilcoxon": p,
            "Succ_Sup_%": 100 * get("supervisor").success.mean(),
            "Succ_Best_%": 100 * get(bm).success.mean(),
            "Sat_Sup_%": get("supervisor").sat.mean(),
            "Switches": get("supervisor").switches.mean(),
            "AutoFrac": get("supervisor").auto_frac.mean()
        })

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table_heldout.csv", index=False)
    print(t.round(4).to_string(index=False))

    print(f"\n  max relative regret vs best fixed = {t['RelRegret_%'].max():.1f}% | "
          f"mean = {t['RelRegret_%'].mean():.1f}% | "
          f"delays with p<0.05: {(t.p_wilcoxon < 0.05).sum()}/{len(t)}")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, st in [("C", "o-"), ("P", "s-"), ("A", "^-")]:
        ax.plot(t.Td, t[k], st, ms=4, label=k)
    ax.errorbar(t.Td, t.Sup, t.Sup_ci95, fmt="D-", c="k", capsize=2, label="Supervisor")
    ax.plot(t.Td, t.Oracle, "k--", label="Best fixed")
    for v in [th["cross"], th["mars"]]:
        ax.axvline(v, ls=":", c="gray")
    ax.set_xlabel("One-way delay Td (s)")
    ax.set_ylabel("RMSE to true goal (m)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_heldout.pdf")
    plt.close()


if __name__ == "__main__":
    sanity()
    th = derive_thresholds()
    heldout(th)
    print(f"\nResults in {OUT.resolve()}")