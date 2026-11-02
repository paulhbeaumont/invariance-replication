"""Cluster-robust estimation helpers shared by the 2X diagnostics producers.

- cluster_sums: bincount over a cluster index.
- hhi_effective: inverse Herfindahl of |w|, the effective-count statistic
  producers 20, 21 and 22 all report.
- wls_fe: multi-regressor WLS on two-way-demeaned data with CRV1 and a within
  R^2. t28.fit2sls covers the one-regressor case only; the balance panels need
  several regressors at once, so the same algebra is written out here with the
  same small-sample correction (t28.ssc_k plus the equation's own regressor
  count).

Lifted verbatim from 21_rotemberg.py (hhi_effective 21:88, cluster_sums 21:98,
wls_fe 21:137) when producer 22 needed the same three. Operation order is
preserved exactly: 21 asserts two numerical identities on every run and its
tables are byte-compared across this extraction, so any reassociation here
would surface immediately.

- shock_ratio: the shock-level IV ratio and its CRV1 sandwich. BOTH of 22's
  shock-level statistics are this one object -- the exposure-robust SE is the
  ratio g'ny / g'nx, the exposure-robust first stage is g'nx / g'nz -- so it is
  written once and called twice. Lifted from 22 (exposure_robust 22:503,
  first_stage_F 22:528) when producer 28 needed the same two on the industry
  axis. It takes a CLUSTER INDEX, not a level name: `cluster_index` maps a level
  onto an index using the FSC category list, and the industry axis reaches its
  clusters a different way, so the axis-specific part stays with each producer.

Deliberately NOT lifted: 21's ar_interval and fmt_ci. WS0 runs no
identification-robust set -- BHJ's main text reports the exposure-robust
standard error and their null-imposed LM interval is a supplementary-appendix
robustness -- so those two stay in 21, which uses them for the GPSS per-category
weak-IV intervals.
"""
import sys
from pathlib import Path

import numpy as np

LIB = Path(__file__).resolve().parent
PIPELINE = LIB.parent
CODE = PIPELINE.parent
sys.path.insert(0, str(CODE / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(LIB))
import oldpath as op  # noqa: E402  (estimator only: t28/w22, as 57 loads them)


def cluster_sums(v, cl, ncl):
    return np.bincount(cl, weights=v, minlength=ncl)


def hhi_effective(w):
    """Inverse Herfindahl of |w| (the same statistic pipeline 20 reports)."""
    w = np.abs(np.asarray(w, np.float64))
    tot = w.sum()
    if tot <= 0:
        return np.nan
    p = w / tot
    return float(1.0 / (p ** 2).sum())


def shock_ratio(g, A, B, cl, coef=None):
    """The shock-level IV ratio coef = g'A / g'B and its CRV1 sandwich.

    The moment is g_n (A_n - coef B_n) and the variance is
    (g'B)^-2 sum_c (sum_{n in c} score_n)^2 with CRV1's G/(G-1) correction.
    This IS the exposure-robust variance: the correlation across counties
    sharing exposure is what the shock-level aggregation has already absorbed.

    Returns (coef, var, G); var is NaN where g'B underflows. Pass `coef` to use
    a ratio already computed elsewhere -- 22 passes fit['beta_shock'], which is
    this same quotient formed in rung_fit, so the SE is built on the number the
    table prints rather than on a re-derivation of it.

    No inversion anywhere. BHJ's main text reports this SE and a conventional
    interval; their null-imposed LM interval is a supplementary-appendix
    robustness that WS0 does not run (CONTINGENCY['not_run']).
    """
    D = float(g @ B)
    if abs(D) < 1e-300:
        return np.nan, np.nan, int(np.unique(cl).size)
    if coef is None:
        coef = float(g @ A) / D
    ncl = int(cl.max()) + 1
    sc = cluster_sums(g * (A - coef * B), cl, ncl)
    G = int(np.unique(cl).size)
    ssc = G / max(G - 1, 1)
    return coef, ssc * float(sc @ sc) / (D * D), G


def wls_fe(y, X, w, fe, nofe, cl):
    """Multi-regressor WLS on two-way-demeaned data, CRV1, within R^2.

    t28.fit2sls covers the one-regressor case; the balance panels need several
    at once, so the same algebra is written out here with the same small-sample
    correction (t28.ssc_k plus this equation's own regressor count).
    """
    M = op.w22.demean(np.column_stack([y, X]), fe, nofe, w)
    yt, Xt = M[:, 0], M[:, 1:]
    XW = Xt * w[:, None]
    A = Xt.T @ XW
    b = np.linalg.solve(A, XW.T @ yt)
    u = yt - Xt @ b
    inv = np.linalg.inv(A)
    ncl = int(cl.max()) + 1
    S = np.column_stack([cluster_sums(XW[:, q] * u, cl, ncl)
                         for q in range(Xt.shape[1])])
    n, G = len(y), int(np.unique(cl).size)
    k = op.t28.ssc_k(fe, nofe, cl) + Xt.shape[1] - 1
    ssc = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = ssc * (inv @ (S.T @ S) @ inv)
    se = np.sqrt(np.diag(V))
    sst = float(np.sum(w * yt ** 2))
    r2 = 1.0 - float(np.sum(w * u ** 2)) / sst if sst > 0 else np.nan
    return b, se, r2
