#!/usr/bin/env python3
"""Plan section 2.3: Table D1, the main disaggregation table, going DOWN.

INVARIANCE_TEST_PROTOCOL.md A.1. The county is the baseline unit, county x industry
is the cell, the treatment sits at the county and is NEVER split (A.1.1), and the
outcome decomposes exactly (A.1.2) - which is 27's assertion 1 and the only data
requirement of the whole exercise.

WHAT THE TABLE IS FOR. Panel A's log-growth coefficient moves across the columns
when nothing about the treatment has changed. Panels B and C should not. The two
bottom rows say why it moved: log growth is undefined for a cell that is empty in
either period, so a log specification silently throws those cells away, and the
share of county employment it keeps falls as the cells get finer.

    (1) County                 ccg + year FE
    (2) County x NAICS2        ccg x industry + industry x year FE
    (3) County x NAICS4        ccg x industry + industry x year FE

    Panel A    log growth,       unweighted        the benchmark everyone runs;
                                                   INADMISSIBLE as an answer (A.1.5)
    Panel A2   log growth,       MPGR weight       the weighting channel, transform held
    Panel B    midpoint growth,  MPGR weight       admissible answer 1
    Panel C    employment levels, Poisson          admissible answer 2

A -> A2 changes only the weight and A2 -> B changes only the transform, so the two
moves separate the implicit weighting from the curvature of the log rather than
confounding them. The plan's skeleton has only A and B; the middle panel is why the
pair is legible.

ccg x year APPEARS NOWHERE. It would absorb the treatment (A.1.4), and no amount of
retained variation would make it admissible.

TABLE D1b, AND WHY IT IS A COMPARISON AND NOT AN ASSERTION. Proposition 1 delivers
beta_d = beta_b when the two levels difference out the same thing, which needs a
fixed effect that is a function of (c, t) at BOTH levels - year effects only. D1's
printed columns do not satisfy that: a cell effect on one side against a county
effect on the other. So the exact test is run separately, and it is run as a
comparison, because the point is not that midpoint growth passes. THE POINT IS THAT
LOG GROWTH FAILS. Midpoint growth is asserted equal to machine precision; the log
rows are the exhibit, and the size of their gap is the finding.

THE COLLAPSE THAT MAKES D1b EXACT AND CHEAP. With a fixed effect that is a function
of (c, t) and a regressor constant within a county-year, the cell-level WLS moment
conditions depend on the cell data ONLY through sum_m w_im y_im and sum_m w_im. So
the cell regression reduces EXACTLY to a county-year regression on 65,085 rows of
collapsed sufficient statistics - same beta, same cluster sums, same z'Wx. Only the
small-sample correction differs, and it is supplied from the cell-level (N, K, G),
so the collapsed computation returns the cell-level standard error too. Asserted
against the direct 1.7M-row NAICS2 fit rather than assumed.

WHY ASSERTION -1 EXISTS, RESTATED BECAUSE IT APPLIES HERE VERBATIM. Not one identity
in this file can detect a wrong regressor. Every one of them is internal to the
constructed outcome or a linearity property of the estimator, and all of them would
pass on the flow-dG panel withdrawn on 2026-08-10. The first thing this script does
is prove the regressor is the cumulative one, at floor zero, and record the vintage
of both inputs. 22 owns that gate and this file calls it.

THE SETTINGS                                                          2026-08-14

    h = 2               the paper's horizon and step 1's headline. h=1 is never used
    z_fixed_7074        1970-1974, five years, ending the year before EFSY starts.
                        NOT in instrument_ccg_year.parquet - 14 never writes a
                        variant exposure to disk because it would overwrite the
                        production panel - so 22's build_exposure_panel rebuilds it
                        in memory and check -0b asserts it IS the parquet's, exactly
    1977-1997           post-1998 is dropped entirely, on instruction; step 1
                        section 1.9.5 is where the weak half is named
    cluster             the CCG, at every level (A.1.6)
    no floor, no winsorization, no cell dropped

Output: step_2/result/TableD1_disaggregation.tex
        step_2/result/TableD1b_invariance.tex
        step_2/result/coeff_d1.tex
        dc.LOG_DIR/d1_coefficients.csv
        dc.LOG_DIR/d1_invariance.csv
        dc.LOG_DIR/28_table_d1.log
"""

import argparse
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
STEP2_CODE = HERE.parent
STEP1_CODE = HERE.parents[3] / "step_1" / "code" / "py"
RESULT_DIR = HERE.parents[2] / "result"

sys.path.insert(0, str(STEP1_CODE))
import dcads_common as dc  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 27 owns the four construction assertions and reaches 22 through its own `w22`.
# THIS FILE MUST NOT _load 22 A SECOND TIME. 22 holds the module-level `failures`
# list and its own copy of 14_build_instrument.py; a second load would give a second
# failures list, and the exit code would then be read off a list that half the checks
# in the process never touched. Reaching 22 through 27 is the same discipline 27
# already applies to `bi` (27_cell_construction_checks.py:115-123).
c27 = _load("cellchecks", STEP2_CODE / "27_cell_construction_checks.py")
w22 = c27.w22
check, Tee, periods_for = w22.check, w22.Tee, w22.periods_for

EXPOSURE_DEFAULT = w22.EXPOSURE_DEFAULT
HORIZON = 2
RUNGS_ALL = w22.RUNGS_ALL
PANELS_ALL = ("A", "A2", "B", "C", "D1b")

N_SUPPORT = c27.N_SUPPORT          # 3,100 CCGs carry a shock

# --------------------------------------------------------------------------------
# Declared tolerances. Every one is a measured quantity plus headroom, with the
# measured value in the comment so a future drift is visible rather than absorbed.
# None of them is "machine precision" - that is not implementable.
# --------------------------------------------------------------------------------
TOL_PYFIX_B = 1e-9      # own 2SLS against pyfixest, beta;  measured 2.1e-14 rel
TOL_PYFIX_SE = 1e-8     # own 2SLS against pyfixest, se;    measured 2.8e-13 rel
TOL_POIS_B = 1e-9       # own IRLS against pf.fepois, beta; measured 6.0e-13 rel
TOL_POIS_SE = 1e-9      # own IRLS against pf.fepois, se;   measured 1.7e-12 rel
TOL_INVAR = 1e-11       # Proposition 1, |beta_cell - beta_county| at year-only FE
TOL_POIS_INVAR = 1e-9   # Proposition 1 for Poisson, once sum_m E = E_c holds. Looser
                        # than TOL_INVAR because it is two IRLS limits against each
                        # other and not one closed form against another; measured
                        # 3.8e-14 at NAICS2, which is closer to TOL_INVAR than the
                        # headroom suggests
TOL_INVAR_SE = 1e-11    # A.1.6, the clustered SEs coincide too
TOL_COLLAPSE = 1e-10    # collapsed cell fit against the direct NAICS2 cell fit
TOL_DENC = 1e-9         # cell midpoint denominator against emp_lag * (2 + g), rel
TOL_CURC = 1e-9         # cell current sum against emp_lag * (1 + g), relative

MPGR_BOUND = 2.0

# 22's own reference block, reused rather than restated: step 1 section 1.8.6 Table V.
STEP1_BETA0, STEP1_SE0 = w22.STEP1_BETA0, w22.STEP1_SE0
STEP1_F0, STEP1_N0 = w22.STEP1_F0, w22.STEP1_N0

PRETTY_RUNG = {"naics2": r"County $\times$ NAICS2",
               "naics4_alloc": r"County $\times$ NAICS4"}


def hash_index(*arrs):
    """A stable fingerprint of a row set, so 'the same rows' is checkable."""
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------------
# The estimator. pyfixest cannot be run at 19.7M rows on this machine, so the fits
# are hand-rolled on top of 22's demeaner - which 22 already validates against
# pyfixest - and the hand-rolled path is itself validated against pyfixest at every
# scale where pyfixest CAN run, on beta AND on the standard error, before a single
# NAICS4 number is read.
# --------------------------------------------------------------------------------

def drop_singletons(f1, f2):
    """Iterative singleton removal, matching pyfixest's fixef_rm='singleton'.

    A singleton contributes nothing to a within estimator - its own fixed effect
    fits it exactly - so beta does not move. N and the level counts do, and they
    enter the small-sample correction, which is why this is done rather than waved
    at. Iterative because dropping a row can orphan a group on the other factor.
    """
    keep = np.ones(len(f1), bool)
    while True:
        n_before = int(keep.sum())
        for f in (f1, f2):
            cnt = np.bincount(f[keep], minlength=int(f.max()) + 1)
            keep[keep] &= cnt[f[keep]] > 1
        if int(keep.sum()) == n_before:
            return keep


def is_nested(f, cl):
    """Does every level of f lie inside exactly one cluster?

    By the group mean rather than by a groupby-nunique: if every row of a group
    carries the same integer cluster id, that id equals the group's mean, exactly, in
    float64 for any id this panel can produce. O(n) and no 19.7M-row groupby.
    """
    n = int(f.max()) + 1
    c = np.bincount(f, minlength=n).astype(np.float64)
    s = np.bincount(f, weights=cl.astype(np.float64), minlength=n)
    mean = np.divide(s, c, out=np.zeros(n), where=c > 0)
    return bool(np.all(mean[f] == cl))


def ssc_k(f1, f2, cl):
    """pyfixest's / fixest's `fixef.K = 'nested'` parameter count.

    Verified against pyfixest 0.60 on three cluster designs: a fixed effect NESTED
    within the cluster variable contributes NO parameters, because its own degrees of
    freedom are already spent inside a cluster; a non-nested one contributes its level
    count, and when two are counted they share one collinear intercept, hence the
    -(counted - 1). Here the cluster is always the CCG and the unit effect is always
    `ccg` or `ccg x industry`, both nested in it, so exactly one factor is counted -
    but the general form is written out so a future change of cluster cannot silently
    keep the wrong correction.
    """
    k, counted = 1, 0
    for f in (f1, f2):
        if not is_nested(f, cl):
            k += int(np.unique(f).size)
            counted += 1
    return k - max(counted - 1, 0)


def crv1(zt, xt, resid, w, cl, n, k, corrected=True, correction_n=None):
    """CRV1 for a just-identified moment, with fixest's small-sample correction.

    beta - beta0 = (z~' W e) / (z~' W x~), so the cluster sums are of w * z~ * e and
    the whole variance is one scalar. ssc = G/(G-1) * (N-1)/(N-K), the same form
    27's _cluster_F uses, which is what makes every F in step 2 comparable to
    step 1's. `corrected=False` returns the same variance with ssc = 1, which is the
    part two levels of aggregation share exactly. `correction_n` supplies a common
    effective observation count when an algebraic identity represents each
    county-year as either one row or a set of weighted cell rows.
    """
    zx = float(zt @ (w * xt))
    if not np.isfinite(zx) or zx == 0.0:
        return np.nan
    g = np.bincount(cl, weights=w * zt * resid, minlength=int(cl.max()) + 1)
    G = int(np.unique(cl).size)
    n_ssc = n if correction_n is None else int(correction_n)
    if corrected and n_ssc <= k:
        raise ValueError(f"CRV1 correction requires N > K; got N={n_ssc}, K={k}")
    ssc = ((G / max(G - 1, 1))
           * ((n_ssc - 1) / (n_ssc - k))) if corrected else 1.0
    v = ssc * float(g @ g) / zx ** 2
    return float(np.sqrt(v)) if v > 0 else np.nan


def fit2sls(y, x, z, w, f1, f2, cl, singleton=True, want_F=True,
            correction_n=None):
    """One coefficient, its clustered SE and the first-stage F, on two-way FE.

    Just-identified, so beta = (z~' W y~) / (z~' W x~) exactly and no linear solve is
    needed. Pass z is x for OLS; the formula then reduces to the ordinary within
    estimator with CRV1, which is why one function covers both. `correction_n`
    affects only the CRV1 finite-sample multiplier.
    """
    keep = drop_singletons(f1, f2) if singleton else np.ones(len(y), bool)
    y, x, z, w = y[keep], x[keep], z[keep], w[keep]
    f1, f2, cl = f1[keep], f2[keep], cl[keep]
    f1 = pd.factorize(f1)[0].astype(np.int64)
    f2 = pd.factorize(f2)[0].astype(np.int64)
    cl = pd.factorize(cl)[0].astype(np.int64)
    iv = z is not x

    M = w22.demean(np.column_stack([y, x] + ([z] if iv else [])), f1, f2, w)
    yt, xt = M[:, 0], M[:, 1]
    zt = M[:, 2] if iv else xt

    zx = float(zt @ (w * xt))
    beta = float(zt @ (w * yt)) / zx if zx != 0.0 else np.nan
    n, k = len(y), ssc_k(f1, f2, cl)
    se = crv1(zt, xt, yt - beta * xt, w, cl, n, k,
              correction_n=correction_n)

    F = np.nan
    if want_F and iv:
        bfs = zx / float(zt @ (w * zt))
        sefs = crv1(zt, zt, xt - bfs * zt, w, cl, n, k,
                    correction_n=correction_n)
        F = (bfs / sefs) ** 2 if sefs and np.isfinite(sefs) and sefs > 0 else np.nan
    return dict(beta=beta, se=se, F=F, n=n, k=k,
                correction_n=n if correction_n is None else int(correction_n),
                n_dropped=int((~keep).sum()), keep=keep)


# --------------------------------------------------------------------------------
# Panel C. Poisson on levels, instrumented by control function, because pyfixest has
# no IV for Poisson and neither does anything else in this stack.
# --------------------------------------------------------------------------------

def poisson_separated(y, f1, f2):
    """Rows whose fixed effect is driven to -infinity by an all-zero group.

    A ccg x industry cell that is empty in every year contributes nothing to the
    Poisson likelihood: its own effect goes to -inf, mu goes to zero, and the score
    for the shock never sees it. Excluding those rows is the estimator's own kernel
    and not a cleaning step - but it is counted and named anyway, because it is a
    SECOND, independent way in which an admissible specification also loses the
    always-empty cells, and that belongs to the extensive-margin exhibit rather than
    being absorbed silently here.

    Iterative, for the same reason singleton dropping is: zeroing one group can empty
    another on the opposite factor.
    """
    keep = np.ones(len(y), bool)
    while True:
        n_before = int(keep.sum())
        for f in (f1, f2):
            s = np.bincount(f[keep], weights=y[keep], minlength=int(f.max()) + 1)
            keep[keep] &= s[f[keep]] > 0
        if int(keep.sum()) == n_before:
            return keep


def poisson_cf(y, x, v, f1, f2, cl, tol=1e-14, maxit=200):
    """HDFE Poisson by IRLS, each step a weighted two-way projection.

    The ppmlhdfe recursion. mu are the IRLS weights and the working response is
    eta + (y - mu)/mu; after demeaning, the within residual IS the full residual
    including the fixed effects, so the fitted eta is the working response minus it
    and no fixed effect is ever materialised.

    `v` is the control function - the residual of the shock on the instrument inside
    the SAME fixed effects. Its coefficient is not reported. THE STANDARD ERROR
    TREATS v AS DATA, so it understates by an amount this design does not correct,
    and that sentence travels with every Panel C number.

    `v=None` is the REDUCED FORM: there is no endogenous regressor, so there is no
    first stage to residualise and no control function to carry. The regressor is
    the instrument itself, the design matrix has one column, and the +1 degree of
    freedom the control function costs is not charged. The understatement caveat
    goes with it -- it was a property of treating v as data, not of Poisson. A
    zero column in v's place would be singular, which is how this path was found.
    """
    n = len(y)
    X = np.asarray(x, np.float64).reshape(-1, 1) if v is None \
        else np.column_stack([x, v])
    K = X.shape[1]
    mu = np.maximum(y, 1e-2) * 0.5 + y.mean() * 0.5
    eta = np.log(mu)
    dev_last = np.inf
    for it in range(maxit):
        ystar = eta + (y - mu) / mu
        M = w22.demean(np.column_stack([ystar] + [X[:, j] for j in range(K)]),
                       f1, f2, mu)
        A = np.empty((K, K))
        b = np.empty(K)
        for i in range(K):
            for j in range(K):
                A[i, j] = M[:, 1 + i] @ (mu * M[:, 1 + j])
            b[i] = M[:, 1 + i] @ (mu * M[:, 0])
        coef = np.linalg.solve(A, b)
        eta = ystar - (M[:, 0] - M[:, 1:] @ coef)
        eta = np.clip(eta, -300.0, 300.0)
        mu = np.exp(eta)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2.0 * float(np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
                                     - (y - mu)))
        if np.isfinite(dev) and abs(dev - dev_last) <= tol * (abs(dev) + 0.1):
            dev_last = dev
            break
        dev_last = dev

    # The sandwich, clustered on the CCG. With the fixed effects concentrated out the
    # score for beta is (y - mu) X~, where X~ is X residualised on the fixed effects
    # under the IRLS weights, and the bread is (X~' diag(mu) X~)^-1. Both are read off
    # ONE final projection at the converged mu, so they are the same object the
    # coefficient came from rather than the previous iteration's.
    #
    # X goes in with a REAL column first. w22.demean tests convergence on column 0
    # alone, so a zeros column there satisfies it on the first sweep and returns X
    # after a single pass - which is not a projection, and showed up as a 2e-5
    # discrepancy in the clustered SE against pf.fepois while beta agreed at 1e-13.
    Xt = w22.demean(np.column_stack([X[:, j] for j in range(K)]), f1, f2, mu)
    A = Xt.T @ (mu[:, None] * Xt)
    Ainv = np.linalg.inv(A)
    e = y - mu
    nb = int(cl.max()) + 1
    g = np.column_stack([np.bincount(cl, weights=e * Xt[:, j], minlength=nb)
                         for j in range(K)])
    S = g.T @ g
    G = int(np.unique(cl).size)
    # ssc_k's leading 1 is the single reported regressor; the control function is
    # the second parameter and is charged only when there is one.
    k = ssc_k(f1, f2, cl) + (K - 1)
    ssc = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = ssc * Ainv @ S @ Ainv
    return dict(beta=float(coef[0]), se=float(np.sqrt(V[0, 0])), F=np.nan,
                n=n, k=k, iters=it + 1, deviance=dev_last)


def control_function(x, z, w, f1, f2):
    """Residualise the shock on the instrument inside the same fixed effects."""
    M = w22.demean(np.column_stack([x, z]), f1, f2, w)
    xt, zt = M[:, 0], M[:, 1]
    zz = float(zt @ (w * zt))
    b = float(zt @ (w * xt)) / zz if zz != 0 else 0.0
    return xt - b * zt


# --------------------------------------------------------------------------------
# The cells, as aligned arrays. The support is identical in every year and in the
# same row order (22's load_rung asserts it), so the cell id is a row position and
# the t-h lag is an array shift, never a join.
# --------------------------------------------------------------------------------

def excluded_county_years(rung, emp, ccg_arr, frame, h):
    """The county-years where the reconstruction identity cannot hold.

    27's assert_reconstruction names them in its message and returns only the count.
    27 IS NOT EDITED to hand the set over; this recomputes it, and main() asserts the
    size against 27's a4_excluded from the same run. Two independent computations of
    the same object, cross-bound - which is worth more than one computation reused.

    den_c = 0 (no county midpoint denominator at all), or the county carries a
    SELF-CANCELLING cell, E_t = -E_t-h != 0, whose weight is zero so it drops out of
    sum_m w mpgr while still contributing -2 E_t-h to the county numerator.
    """
    ccgs, idx = np.unique(ccg_arr, return_inverse=True)
    pos_of = {c: i for i, c in enumerate(ccgs)}
    bad = []
    for y, sub in frame.groupby("year", sort=True):
        lag, cur = emp[y - h], emp[y]
        in_frame = np.zeros(len(ccgs), bool)
        p = np.array([pos_of[c] for c in sub["ccg"].to_numpy()])
        in_frame[p] = True
        ok = in_frame[idx]
        den = cur + lag
        selfc = ok & (den == 0.0) & (lag != 0.0)
        s_den_c = np.bincount(idx[ok], weights=den[ok], minlength=len(ccgs))
        s_bad = np.bincount(idx[selfc], weights=np.ones(int(selfc.sum())),
                            minlength=len(ccgs))
        hit = (s_den_c[p] == 0.0) | (s_bad[p] > 0.0)
        for c in sub["ccg"].to_numpy()[hit]:
            bad.append((int(c), int(y)))
    return set(bad)


def build_long(rung, emp, ccg_arr, code_arr, frame, h):
    """One long array set per rung, over the frame's county-years.

    Everything downstream is a bincount over dense integer indices, so 939,603 cell
    rows a year never become a join. f_cy indexes the FRAME's rows, which is what
    makes the county aggregation, the weight denominator and the D1b collapse all the
    same operation.
    """
    years = sorted(frame["year"].unique())
    yr_of = {y: i for i, y in enumerate(years)}
    ccgs, cidx = np.unique(ccg_arr, return_inverse=True)
    pos_of = {c: i for i, c in enumerate(ccgs)}
    codes, kidx = np.unique(code_arr, return_inverse=True)
    n_row = len(ccg_arr)
    cell_id = np.arange(n_row, dtype=np.int64)     # (ccg, code), fixed across years

    frow = {(int(c), int(y)): i for i, (c, y) in
            enumerate(zip(frame["ccg"].to_numpy(), frame["year"].to_numpy()))}

    cur_l, lag_l, fcell_l, fky_l, fcy_l = [], [], [], [], []
    for y, sub in frame.groupby("year", sort=True):
        lag, cur = emp[y - h], emp[y]
        in_frame = np.zeros(len(ccgs), bool)
        p = np.array([pos_of[c] for c in sub["ccg"].to_numpy()])
        in_frame[p] = True
        ok = in_frame[cidx]
        cy = np.full(len(ccgs), -1, np.int64)
        cy[p] = np.array([frow[(int(c), int(y))] for c in sub["ccg"].to_numpy()])
        cur_l.append(cur[ok])
        lag_l.append(lag[ok])
        fcell_l.append(cell_id[ok])
        fky_l.append(kidx[ok].astype(np.int64) * len(years) + yr_of[y])
        fcy_l.append(cy[cidx][ok])

    d = dict(
        cur=np.concatenate(cur_l), lag=np.concatenate(lag_l),
        f_cell=np.concatenate(fcell_l), f_ky=np.concatenate(fky_l),
        f_cy=np.concatenate(fcy_l),
        n_code=len(codes), n_year=len(years), n_cell=n_row,
    )
    del cur_l, lag_l, fcell_l, fky_l, fcy_l

    n_frame = len(frame)
    d["den_c"] = np.bincount(d["f_cy"], weights=d["cur"] + d["lag"], minlength=n_frame)
    d["lag_c"] = np.bincount(d["f_cy"], weights=d["lag"], minlength=n_frame)
    d["cur_c"] = np.bincount(d["f_cy"], weights=d["cur"], minlength=n_frame)
    return d


# --------------------------------------------------------------------------------
# The four panels.
# --------------------------------------------------------------------------------

def county_outcomes(frame):
    """Every county quantity from the instrument parquet's own emp_lag and emp_growth.

    E_ct = E_c,t-h (1 + g), so log growth is log1p(g) and midpoint growth is
    2g / (2 + g), both exactly and without a second employment series. That is what
    makes column (1) comparable to step 1's county coefficient rather than merely
    similar to it - and the cell sums are asserted against these, not the other way
    round.
    """
    g = frame["emp_growth"].to_numpy()
    lag = frame["emp_lag"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        log_c = np.where(g > -1.0, np.log1p(np.where(g > -1.0, g, 0.0)), np.nan)
    return dict(g=g, log=log_c, mpgr=2.0 * g / (2.0 + g), lev=lag * (1.0 + g),
                lag=lag, den=lag * (2.0 + g))


def cell_outcomes(d):
    """Midpoint growth, log growth and the MPGR weight, on the cell rows.

        mpgr_im = 2 (E_imt - E_im,t-h) / (E_imt + E_im,t-h)   guard on != 0
        w_im    = (E_imt + E_im,t-h) / (E_ct + E_c,t-h)       COUNTY denominator
        log_im  = log E_imt - log E_im,t-h                    both periods > 0

    THE GUARD IS != 0, NEVER > 0. A `> 0` guard silently sets midpoint growth to zero
    for EFSY's negative-employment cells while leaving their weights intact, and the
    reconstruction then fails in 82 county-years at 7.4e-4 - small enough to be
    mistaken for a tolerance and large enough to be wrong (step_2/README.md 1.1).

    w_im is normalised by the COUNTY's midpoint denominator, never by the sum over
    whichever cells survive a transformation. That is the whole reason weighted cell
    midpoint growth reconstructs county midpoint growth exactly.
    """
    cur, lag = d["cur"], d["lag"]
    den = cur + lag
    nz = den != 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        mpgr = np.where(nz, 2.0 * (cur - lag) / np.where(nz, den, 1.0), 0.0)
        denc = d["den_c"][d["f_cy"]]
        w = np.where(denc != 0.0, den / np.where(denc != 0.0, denc, 1.0), 0.0)
        pos = (cur > 0) & (lag > 0)
        log = np.where(pos, np.log(np.where(pos, cur, 1.0))
                       - np.log(np.where(pos, lag, 1.0)), 0.0)
    return mpgr, w, log, pos


def run_panels(rung, d, frame, cy, panels, tag, res, val_pf, negative="floor"):
    """Panels A, A2, B and C at one cell rung, on the frozen frame.

    Every fit takes the same county-years. The cell ROWS differ by panel, and that is
    the phenomenon rather than a defect: log growth is undefined for a cell empty in
    either period, so Panel A runs on a subset and Panel B does not. Each panel's row
    set is hashed and logged, so "the same rows" is a checkable claim.
    """
    mpgr, w, log, pos = cell_outcomes(d)
    shock = frame["shock"].to_numpy()[d["f_cy"]]
    z = frame["z"].to_numpy()[d["f_cy"]]
    cl = frame["ccg_idx"].to_numpy()[d["f_cy"]]
    f1, f2 = d["f_cell"], d["f_ky"]
    one = np.ones(len(mpgr))

    emp_tot = float(d["cur"].sum())
    n_all = len(mpgr)

    out = []
    specs = []
    if "A" in panels:
        specs.append(("A", "log growth, unweighted", log, one, pos))
    if "A2" in panels:
        specs.append(("A2", "log growth, MPGR weight", log, w, pos))
    if "B" in panels:
        specs.append(("B", "midpoint growth, MPGR weight", mpgr, w,
                      np.ones(n_all, bool)))

    for pan, label, y, wt, msk in specs:
        t0 = time.time()
        print(f"\n[D1] panel {pan} - {label} ({rung})")
        yy, xx, zz, ww = y[msk], shock[msk], z[msk], wt[msk]
        a1, a2, ac = f1[msk], f2[msk], cl[msk]
        r = fit2sls(yy, xx, zz, ww, a1, a2, ac)
        ro = fit2sls(yy, xx, xx, ww, a1, a2, ac, want_F=False)
        kept = np.flatnonzero(msk)[r["keep"]]
        share = float(d["cur"][kept].sum()) / emp_tot if emp_tot else np.nan
        wsum = np.bincount(d["f_cy"][kept], weights=wt[kept], minlength=len(frame))
        print(f"    2SLS  beta {r['beta']:+.4f} ({r['se']:.4f})  F {r['F']:.1f}  "
              f"N {r['n']:,}  K {r['k']:,}  singletons dropped {r['n_dropped']:,}")
        print(f"    OLS   beta {ro['beta']:+.4f} ({ro['se']:.4f})")
        # sum_m of THIS panel's weight within a county-year. It is 1.000 for Panel B
        # by construction, below 1 and county-varying for Panel A2 - which is the
        # implicit weighting made visible - and a retained CELL COUNT for Panel A,
        # whose weight is one.
        print(f"    rows {r['n']:,} of {n_all:,} ({100 * r['n'] / n_all:.1f}%), "
              f"county employment retained {100 * share:.2f}%, "
              f"mean sum_m weight = {wsum.mean():.4f}, "
              f"p10 = {np.quantile(wsum, 0.1):.4f}   [{time.time() - t0:.0f}s]")
        for est, rr in (("2SLS", r), ("OLS", ro)):
            out.append(dict(rung=rung, column=PRETTY_RUNG[rung], panel=pan,
                            spec=label, estimator=est, beta=rr["beta"], se=rr["se"],
                            fs_F=r["F"] if est == "2SLS" else np.nan,
                            n_fit=rr["n"], k=rr["k"], n_rows_avail=n_all,
                            emp_share=share, wsum_mean=float(wsum.mean()),
                            wsum_p10=float(np.quantile(wsum, 0.1)),
                            zero_cell_share=res[rung]["zero_cell_share"],
                            row_hash=hash_index(kept)))
        if val_pf and rung == "naics2":
            validate_2sls(yy, xx, zz, ww, a1, a2, ac, r, ro, f"{rung} panel {pan}")

    if "C" in panels:
        out += run_poisson(rung, d, frame, shock, z, cl, f1, f2, emp_tot, val_pf,
                           negative)
    return out


def poisson_sample(rung, d, f1, f2, negative):
    """Which cell-years Panel C can estimate on, and why the others cannot.

    TWO exclusions, and they are completely different animals. The whole point of
    reporting them apart is that ONE OF THEM COSTS NOTHING and the other is a choice.

    NEGATIVE EMPLOYMENT - a choice, and since 2026-08-14 it is `floor`. Poisson needs
    a weakly positive outcome. EFSY carries negative employment, confined to 1981-1997,
    never as much as one employee in absolute value, from EFSY's own desuppression and
    not from step 1's allocation ladder. These are 1,651 cell-years at NAICS4.

      floor  E_imt <- max(E_imt, 0). The cell STAYS in the panel. This is the default
             because A.3.1 is the only rule that speaks to the case and it says never
             to drop a cell - "not empty cells, not tiny ones, not extreme ones" - and
             a desuppression artifact of -0.5 employees is a tiny cell, not an event.
      drop   the row is excluded. Kept runnable, never canonical.

    The earlier justification for `drop` was that "flooring would break cells sum to
    the county". THAT ARGUMENT IS WRONG AS APPLIED HERE and is recorded so it is not
    reinstated: it is a valid reason not to floor the panel GLOBALLY, where Panels A
    and B would be corrupted and the reconstruction destroyed, but it does not
    separate the two options INSIDE Panel C. Dropping a cell carrying -0.5 removes the
    same -0.5 from the cell sum that flooring does; both leave sum_m E = E_c + |neg|.
    Measured at NAICS2 on one frame: the two differ by 102 rows and by less than 1e-4
    in beta. The choice is about A.3.1 compliance, not about the estimate.

    SEPARATED CELLS - not a choice at all, and not affected by the above. A
    ccg x industry cell empty in EVERY year has its effect at -inf, mu at zero, and
    contributes to no score; pyfixest and ppmlhdfe drop it too. `poisson_separated`
    receives max(E, 0) in BOTH modes, so the separated set is identical either way and
    the mode cannot move it.

    NOTE WHAT IS *NOT* HERE. Exits are not excluded by anything. An exit is E_t = 0
    with E_im,t-h > 0, so its cell has positive years and is never separated: all
    1,334,234 of them at NAICS4 are in Panel C at y = 0, which is a perfectly good
    Poisson outcome. Entries likewise. That is the whole difference from Panel A.
    """
    raw = d["cur"]
    n_neg = int((raw < 0).sum())
    y = np.maximum(raw, 0.0) if negative == "floor" else raw
    keep_sign = np.ones(len(raw), bool) if negative == "floor" else (raw >= 0.0)
    # max(raw, 0) in BOTH modes, so `sep` is a property of the panel and not of the
    # flag - otherwise the two modes would differ in two places at once.
    sep = poisson_separated(np.maximum(raw, 0.0), f1, f2)
    msk = keep_sign & sep
    n_sep = int((~sep).sum())
    n_drop_neg = int((~keep_sign).sum())
    # THE CHECK IS ON POSITIVE EMPLOYMENT, and the distinction is not pedantic. A
    # separated group is one whose max(E, 0) sums to zero, so its rows can still carry
    # small NEGATIVE desuppression noise - 1.4 employees at NAICS4 - and asserting
    # sum|E| == 0 fails on that while nothing is actually wrong. What has to be true is
    # that no POSITIVE employment leaves with them, which is what "Panel C estimates on
    # 100.00% of county employment" rests on.
    #
    # It is tautological against the current poisson_separated, which is fed max(E, 0),
    # and it is written anyway: if that input is ever changed to raw E, positive rows
    # would start leaking into the separated set and this is the line that would catch
    # it. 27's collapse check is kept for the same reason.
    pos_lost = float(np.maximum(raw, 0.0)[~sep].sum())
    neg_resid = float(raw[~sep].sum())
    check(pos_lost == 0.0,
          f"C  no positive employment leaves with the separated rows ({rung})",
          f"{n_sep:,} rows in ccg x industry cells with no employment in ANY year of "
          f"the window; positive employment lost {pos_lost:.1f}, negative "
          f"desuppression residue {neg_resid:.1f} employees. They are cells that never "
          f"existed inside the window, not exits")
    check(n_neg > 0,
          f"C  the negative-employment rows are {negative}ed and counted ({rung})",
          f"{n_neg:,} cell-years with E < 0, min {float(raw.min()):.3f}; EFSY "
          f"desuppression, not the allocation ladder. Mode `{negative}`: "
          + (f"set to zero and KEPT, per A.3.1" if negative == "floor"
             else f"{n_drop_neg:,} rows excluded"))
    return y, msk, dict(n_negative=n_neg, n_separated=n_sep,
                        n_dropped_negative=n_drop_neg)


def bc_decomposition(rung, d, negative, n_fit_B, n_fit_C):
    """Why Panels B and C do not run on the same number of rows.

    The question this answers is the one a reader asks first and the table cannot
    answer on its own, so it is computed from the panel, cross-checked against the
    two fits, and emitted as a note rather than left to be reconstructed.

    THE GAP IS NOT THE EXITS, and that is the point of the note. An exit is
    E_imt = 0 with E_im,t-h > 0. It is an observation in BOTH panels: midpoint growth
    gives it exactly -2, Poisson gives it y = 0, and neither has to invent anything.
    Entries are the same story with the periods reversed.

    Panel C loses exactly two things:
      the negative cells   a CHOICE, and under the canonical `floor` it loses none
      the separated cells  cells with no employment in ANY year of the estimation
                           window. Their fixed effect is at -inf and they enter no
                           score; pyfixest and ppmlhdfe drop them too. This is the
                           estimator's kernel, not a rule of ours, and it is where
                           essentially the whole gap is.

    ONE QUALIFICATION, MEASURED RATHER THAN WAVED AT. "Separated" is defined on the
    estimation window 1977-1997, but a t-h lag reaches back to 1975. So a cell with
    employment in 1975 or 1976 and none thereafter is separated AND contributes an
    exit row - it died before the window opened. Those rows ARE in the gap: 4,486 of
    1,334,234 exits at NAICS4, 0.3%. Counted here rather than asserted away, because
    the clean claim "no exit is in the gap" is false and the true one - 99.7% of exits
    are in both panels - is nearly as good and is checkable.

    Panel A is shown alongside because it is the contrast that matters: A loses every
    exit and every entry, which are real events carrying real employment, while C
    loses cells that carried nothing inside the window.
    """
    cur, lag = d["cur"], d["lag"]
    neg = (cur < 0) | (lag < 0)
    ok = ~neg
    sep = ~poisson_separated(np.maximum(cur, 0.0), d["f_cell"], d["f_ky"])
    is_exit = ok & (cur == 0) & (lag > 0)
    cat = dict(
        n_total=len(cur),
        both=int((ok & (cur > 0) & (lag > 0)).sum()),
        exit=int(is_exit.sum()),
        entry=int((ok & (cur > 0) & (lag == 0)).sum()),
        empty=int((ok & (cur == 0) & (lag == 0)).sum()),
        negative=int(neg.sum()),
        separated=int(sep.sum()),
        sep_exit=int((sep & is_exit).sum()),
        exit_emp=float(lag[is_exit].sum()),
        sep_emp=float(np.abs(cur[sep]).sum()),
    )
    n_drop = 0 if negative == "floor" else cat["negative"]

    print(f"\n[N] why Panels B and C differ ({rung})")
    print(f"    cell-years in the panel                  {cat['n_total']:>12,}")
    for k, lab in (("both", "positive in both periods"),
                   ("exit", "EXIT  E_t = 0, E_t-h > 0"),
                   ("entry", "ENTRY E_t > 0, E_t-h = 0"),
                   ("empty", "empty in both periods"),
                   ("negative", "negative employment somewhere")):
        print(f"      {lab:36s} {cat[k]:>12,}  {100 * cat[k] / cat['n_total']:5.1f}%")
    print(f"    Panel B keeps every one                  {n_fit_B:>12,}")
    print(f"      less separated cells                   {-cat['separated']:>12,}  "
          f"no employment in ANY year of the window ({cat['sep_emp']:.0f} employees)")
    print(f"      less negatives, mode `{negative:5s}`            {-n_drop:>12,}")
    print(f"    Panel C                                  {n_fit_C:>12,}")
    print(f"    EXITS AND ENTRIES ARE IN BOTH PANELS: {cat['exit'] - cat['sep_exit']:,}"
          f" of {cat['exit']:,} exits "
          f"({100 * (1 - cat['sep_exit'] / max(cat['exit'], 1)):.1f}%) and all "
          f"{cat['entry']:,} entries. Panel B takes an exit at mpgr = -2, Panel C at "
          f"y = 0.")
    print(f"    The {cat['sep_exit']:,} exits that ARE in the gap died before 1977: "
          f"their cell is empty across the whole window and its lag reaches back to "
          f"{d['n_year'] and 1975}.")
    print(f"    Panel A takes none of them: log growth is undefined on "
          f"{100 * (cat['n_total'] - cat['both']) / cat['n_total']:.1f}% of the panel, "
          f"including every exit and every entry.")

    expect = cat["n_total"] - cat["separated"] - n_drop
    check(expect == n_fit_C,
          f"N  the B/C gap is exactly separated + dropped negatives ({rung})",
          f"{cat['n_total']:,} - {cat['separated']:,} separated - {n_drop:,} negative "
          f"= {expect:,}, against Panel C's fitted {n_fit_C:,}")
    check(cat["sep_exit"] < 0.01 * max(cat["exit"], 1),
          f"N  the gap is not the exits ({rung})",
          f"{cat['sep_exit']:,} of {cat['exit']:,} exits fall in it "
          f"({100 * cat['sep_exit'] / max(cat['exit'], 1):.2f}%), all of them cells "
          f"that died before the window opened; the other "
          f"{cat['exit'] - cat['sep_exit']:,} are in both panels")
    return cat


def run_poisson(rung, d, frame, shock, z, cl, f1, f2, emp_tot, val_pf,
                negative="floor"):
    """Panel C. Levels, Poisson, instrumented by control function.

    The sample and its two exclusions are `poisson_sample`'s; this fits on it.
    """
    print(f"\n[D1] panel C - employment levels, Poisson, control function ({rung})")
    t0 = time.time()
    y, msk, cnt = poisson_sample(rung, d, f1, f2, negative)
    n_neg, n_sep = cnt["n_negative"], cnt["n_separated"]

    yy, xx, zz = y[msk], shock[msk], z[msk]
    a1 = pd.factorize(f1[msk])[0].astype(np.int64)
    a2 = pd.factorize(f2[msk])[0].astype(np.int64)
    ac = pd.factorize(cl[msk])[0].astype(np.int64)
    v = control_function(xx, zz, np.ones(len(yy)), a1, a2)
    r = poisson_cf(yy, xx, v, a1, a2, ac)
    share = float(yy.sum()) / emp_tot if emp_tot else np.nan
    print(f"    Poisson-CF  beta {r['beta']:+.4f} ({r['se']:.4f})  N {r['n']:,}  "
          f"{r['iters']} IRLS steps, deviance {r['deviance']:.6e}")
    print(f"    rows {r['n']:,} of {len(y):,}; {n_neg:,} negative ({negative}), "
          f"{n_sep:,} separated; county employment retained {100 * share:.2f}%"
          f"   [{time.time() - t0:.0f}s]")
    print("    THE SE TREATS THE CONTROL FUNCTION AS DATA and understates by an "
          "amount this design does not correct.")
    if val_pf and rung == "naics2":
        validate_poisson(yy, xx, v, a1, a2, ac, r, f"{rung} panel C")
    return [dict(rung=rung, column=PRETTY_RUNG[rung], panel="C",
                 spec="employment levels, Poisson (control function)",
                 estimator="Poisson", beta=r["beta"], se=r["se"], fs_F=np.nan,
                 n_fit=r["n"], k=r["k"], n_rows_avail=len(y), emp_share=share,
                 wsum_mean=np.nan, wsum_p10=np.nan, iters=r["iters"],
                 negative_mode=negative, **cnt,
                 row_hash=hash_index(np.flatnonzero(msk)))]


# --------------------------------------------------------------------------------
# The validation gates. Nothing at NAICS4 is read until these pass at a scale where
# pyfixest can be run.
# --------------------------------------------------------------------------------

def validate_2sls(y, x, z, w, f1, f2, cl, r_iv, r_ols, tag):
    """The hand-rolled path against pyfixest, on beta AND on the standard error.

    pyfixest refuses a negative weight, and the MPGR weight IS negative wherever a
    cell's midpoint denominator is - which EFSY's negative employment makes possible,
    in 65 cell-years at NAICS2 and 1,705 at NAICS4. Those rows are kept UNCLAMPED in
    the production fit, because clamping them would break the reconstruction that
    makes the invariance exact. So when they are present the validation is re-run on
    the non-negative subset: it is a different sample, and it is the CODE PATH that is
    being validated, not the estimate. Said out loud in the check message rather than
    silently narrowing.
    """
    import pyfixest as pf
    # pyfixest requires a STRICTLY positive weight, so the zero-weight rows go too.
    # A zero-weight row contributes nothing to either estimator, so dropping them
    # changes nothing but the row count; the negative ones are the substantive part.
    sub = w > 0
    n_neg = int((~sub).sum())
    if n_neg:
        y, x, z, w = y[sub], x[sub], z[sub], w[sub]
        f1, f2, cl = f1[sub], f2[sub], cl[sub]
        r_iv = fit2sls(y, x, z, w, f1, f2, cl)
        r_ols = fit2sls(y, x, x, w, f1, f2, cl, want_F=False)
        tag += f", on the {len(y):,} non-negative-weight rows ({n_neg:,} excluded "
        tag += "for pyfixest's sake only; the production fit keeps them)"
    df = pd.DataFrame(dict(y=y, x=x, z=z, w=w, f1=f1, f2=f2, cl=cl))
    kw = dict(vcov={"CRV1": "cl"}, fixef_rm="singleton", weights="w",
              demeaner=w22.DEMEANER)
    m = pf.feols("y ~ 1 | f1 + f2 | x ~ z", df, **kw)
    o = pf.feols("y ~ x | f1 + f2", df, **kw)
    for nm, mm, rr in (("2SLS", m, r_iv), ("OLS", o, r_ols)):
        key = "x" if nm == "OLS" else "x"
        b, se = float(mm.coef()[key]), float(mm.se()[key])
        db = abs(b - rr["beta"]) / max(abs(b), 1e-8)
        ds = abs(se - rr["se"]) / max(abs(se), 1e-12)
        check(db < TOL_PYFIX_B and ds < TOL_PYFIX_SE,
              f"V  own {nm} reproduces pyfixest, beta AND se ({tag})",
              f"beta {b:+.10f} vs {rr['beta']:+.10f} (rel {db:.2e}), "
              f"se {se:.10f} vs {rr['se']:.10f} (rel {ds:.2e}), N {int(mm._N):,} "
              f"vs {rr['n']:,}")


def validate_poisson(y, x, v, f1, f2, cl, r, tag):
    import pyfixest as pf
    # v=None is the reduced form: one regressor, so the benchmark formula has to
    # drop v too or it would be validating a different model.
    cols = dict(y=y, x=x, f1=f1, f2=f2, cl=cl)
    if v is not None:
        cols["v"] = v
    fml = "y ~ x + v | f1 + f2" if v is not None else "y ~ x | f1 + f2"
    df = pd.DataFrame(cols)
    # iwls_tol is tightened well past pyfixest's 1e-8 default. The point estimates
    # agree at 1e-14 whatever it is set to, but the clustered SE is read off the
    # cluster sums of the working residual at the LAST IRLS step, so at a loose
    # tolerance the two implementations are being compared at two different points of
    # the same iteration rather than at its limit.
    m = pf.fepois(fml, df, vcov={"CRV1": "cl"},
                  fixef_rm="none", separation_check=None,
                  iwls_tol=1e-12, iwls_maxiter=200, demeaner=w22.DEMEANER)
    b, se = float(m.coef()["x"]), float(m.se()["x"])
    db = abs(b - r["beta"]) / max(abs(b), 1e-8)
    ds = abs(se - r["se"]) / max(abs(se), 1e-12)
    check(db < TOL_POIS_B and ds < TOL_POIS_SE,
          f"V  own IRLS reproduces pf.fepois, beta AND se ({tag})",
          f"beta {b:+.10f} vs {r['beta']:+.10f} (rel {db:.2e}), "
          f"se {se:.10f} vs {r['se']:.10f} (rel {ds:.2e})")


# --------------------------------------------------------------------------------
# Table D1b. Proposition 1, at the fixed effect the proposition describes, run as a
# COMPARISON: midpoint growth is asserted invariant, log growth is not, and the size
# of the log failure is the exhibit.
# --------------------------------------------------------------------------------

def collapse(y, w, f_cy, n_frame):
    """Cell-level (y, w) to county-year sufficient statistics (Ybar, T).

    With a fixed effect that is a function of (c, t) and a regressor constant within a
    county-year, the cell moment conditions see the cell data only through these two
    numbers. sum_m w_im (y_im - a_t - beta x_ct) = T_ct (Ybar_ct - a_t - beta x_ct),
    identically, so the collapsed WLS is the SAME estimator and not an approximation
    to it.
    """
    T = np.bincount(f_cy, weights=w, minlength=n_frame)
    S = np.bincount(f_cy, weights=w * y, minlength=n_frame)
    with np.errstate(divide="ignore", invalid="ignore"):
        Ybar = np.where(T != 0.0, S / np.where(T != 0.0, T, 1.0), 0.0)
    return Ybar, T


def _within(M, w, f1, f2=None):
    """Weighted within transformation on one or two factors, in place-ish."""
    if f2 is None:
        n_g = int(f1.max()) + 1
        s = np.bincount(f1, weights=w, minlength=n_g)
        s[s == 0.0] = 1.0
        for j in range(M.shape[1]):
            M[:, j] -= (np.bincount(f1, weights=w * M[:, j], minlength=n_g) / s)[f1]
        return M
    return w22.demean(M, f1, f2, w)


def fit_collapsed(y, x, z, w, f1, f2, cl, n_cell, k_cell):
    """Within 2SLS on collapsed data, with the CELL-level small-sample correction.

    z'Wx and every cluster sum are invariant under the collapse - they are sums of the
    same terms, grouped - so the collapsed data give the cell-level beta and the
    cell-level clustered SE exactly, once (N, K) are supplied from the cell level. The
    only thing the collapse loses is the row count, and that is passed in.

    `se_raw` is the same variance WITHOUT the finite-sample correction. It is what the
    two levels share exactly: A.1.6's "identical standard errors" is a statement about
    the clustered variance, and the correction G/(G-1) (N-1)/(N-K) is the one part
    that knows how many ROWS the regression ran on, which is 65,054 at the county and
    19.7 million at the cells. Asserting the corrected SEs equal would be asserting
    that a degrees-of-freedom convention does not exist.
    """
    M = np.column_stack([y, x, z]).astype(np.float64)
    M = _within(M, w, f1, f2)
    yt, xt, zt = M[:, 0], M[:, 1], M[:, 2]
    zx = float(zt @ (w * xt))
    beta = float(zt @ (w * yt)) / zx if zx != 0.0 else np.nan
    resid = yt - beta * xt
    se = crv1(zt, xt, resid, w, cl, n_cell, k_cell)
    se_raw = crv1(zt, xt, resid, w, cl, n_cell, k_cell, corrected=False)
    return beta, se, se_raw


def boot_gap(rows, yr, cl, unit_fe, nrep, seed):
    """Paired cluster bootstrap over CCGs for a gap between two coefficients.

    A gap between two coefficients estimated on one sample is exactly the case where a
    clustered standard error is the wrong yardstick, which is why 22 bootstraps its
    wedge and why this does too. `rows` is a list of (y, w) pairs, county side first
    and cell side second, on the same county-years; the fixed effects are refitted
    inside the replicate because they move with the resample.

    The resampled CCG ids repeat, so under `unit_fe` the unit effect must be the DRAW
    and not the CCG - a county picked twice is two units, not one. 22's bootstrap
    makes the same point about the same trap.
    """
    rng = np.random.default_rng(seed)
    # One sort, then a slice per cluster. Building the index by `cl == c` inside a
    # comprehension is O(G x N) and costs more than the 200 replicates it feeds.
    order = np.argsort(cl, kind="stable")
    bounds = np.searchsorted(cl[order], np.arange(int(cl.max()) + 2))
    present = np.flatnonzero(np.diff(bounds) > 0)
    by = [order[bounds[c]:bounds[c + 1]] for c in present]
    x, z = rows[0][2], rows[0][3]
    acc = []
    for _ in range(nrep):
        pick = rng.integers(0, len(by), size=len(by))
        ix = np.concatenate([by[c] for c in pick])
        rep = np.concatenate([np.full(len(by[c]), i, np.int64)
                              for i, c in enumerate(pick)])
        gg = pd.factorize(yr[ix])[0].astype(np.int64)
        f1, f2 = ((rep, gg) if unit_fe else (gg, None))
        bs = []
        for (y, w, _, _) in rows:
            ww = w[ix]
            M = _within(np.column_stack([y[ix], x[ix], z[ix]]), ww, f1, f2)
            zx = float(M[:, 2] @ (ww * M[:, 1]))
            bs.append(float(M[:, 2] @ (ww * M[:, 0])) / zx if zx != 0 else np.nan)
        if all(np.isfinite(b) for b in bs):
            acc.append(bs[1] - bs[0])
    a = np.asarray(acc)
    if len(a) < 10:
        return np.nan, np.nan, np.nan
    return (float(np.std(a, ddof=1)), float(np.quantile(a, 0.025)),
            float(np.quantile(a, 0.975)))


FE_BLOCKS = (
    ("ccgyear", r"$ccg$ + year", True),
    ("year", r"year only", False),
)


def invariance_block(rung, d, frame, cy, nrep, seed, direct_check, negative="floor"):
    """Table D1b at one rung: where the equality holds, and where log growth breaks it.

    THE REQUIREMENT IS NOT "year effects only". Proposition 1 needs the two levels to
    difference out the SAME thing, which any fixed effect that is a function of (c, t)
    satisfies. Two of them are, both are exact, and both are run:

        ccg + year   the county side IS step 1's own specification, so the log failure
                     is measured against a coefficient a reader recognises (+0.148)
                     rather than against a cross-sectional one. The cell side carries a
                     plain ccg effect, NOT ccg x industry - a cell effect is not a
                     function of (c, t) and would break the invariance by design, which
                     is precisely what D1's printed columns do and why Panel B is not
                     expected to be flat there.
        year only    step_2/README.md section 3's wording, kept because it is the
                     minimal setting and because agreeing at two fixed effects is worth
                     more than agreeing at one.

    The county side is unweighted throughout - at the county rung the MPGR weight is
    W_c = 1 - and only the cell construction changes down the rows. That is what makes
    one county benchmark legible against four cell constructions.
    """
    mpgr, w, log, pos = cell_outcomes(d)
    n_frame = len(frame)
    yr = frame["year_idx"].to_numpy().astype(np.int64)
    cl = frame["ccg_idx"].to_numpy().astype(np.int64)
    x = frame["shock"].to_numpy()
    z = frame["z"].to_numpy()
    one_c = np.ones(n_frame)
    n_cellrows = len(mpgr)

    # The cell-level (N, K) the collapsed fit has to be told about. The cluster is the
    # CCG and the unit effect is nested in it, so only `year` is counted at BOTH rungs:
    # K = 1 + n_year either way.
    k_cell = 1 + d["n_year"]

    denlag = d["lag_c"][d["f_cy"]]
    lagsh = np.where(denlag != 0.0, d["lag"] / np.where(denlag != 0.0, denlag, 1.0), 0.0)
    base_pos = d["lag"] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        g_cell = np.where(base_pos, (d["cur"] - d["lag"])
                          / np.where(base_pos, d["lag"], 1.0), 0.0)

    SPECS = [
        ("midpoint", r"Midpoint growth, weights $\omega_{im}$", cy["mpgr"],
         mpgr, w, True),
        ("log_unw", r"Log growth, unweighted", cy["log"], log, pos.astype(float), False),
        ("log_w", r"Log growth, weights $\omega_{im}$", cy["log"], log, w * pos, False),
        ("standard", r"Standard growth, lagged-employment shares", cy["g"],
         g_cell, np.where(base_pos, lagsh, 0.0), False),
    ]

    rows = []
    for fe_key, fe_lab, unit_fe in FE_BLOCKS:
        print(f"\n[D1b] Proposition 1 at {fe_lab} fixed effects, both levels ({rung})")
        f1, f2 = ((cl, yr) if unit_fe else (yr, None))
        for key, label, y_c, y_m, w_m, exact in SPECS:
            # The county side and the cell side are read on the SAME county-years, or
            # the gap would carry a sample difference. County log growth is undefined
            # where employment reached zero, and that county-year leaves BOTH sides.
            fin = np.isfinite(y_c)
            yc = np.nan_to_num(y_c)[fin]
            Ybar, T = collapse(np.nan_to_num(y_m), w_m, d["f_cy"], n_frame)
            a1 = pd.factorize(f1[fin])[0].astype(np.int64)
            a2 = None if f2 is None else pd.factorize(f2[fin])[0].astype(np.int64)
            bc, sc, sc0 = fit_collapsed(yc, x[fin], z[fin], one_c[fin], a1, a2,
                                        cl[fin], int(fin.sum()), 1 + d["n_year"])
            bm, sm, sm0 = fit_collapsed(Ybar[fin], x[fin], z[fin], T[fin], a1, a2,
                                        cl[fin], n_cellrows, k_cell)
            gap = bm - bc
            gse, glo, ghi = boot_gap([(yc, one_c[fin], x[fin], z[fin]),
                                      (Ybar[fin], T[fin], x[fin], z[fin])],
                                     yr[fin], cl[fin], unit_fe, nrep, seed)
            rows.append(dict(rung=rung, fe=fe_key, fe_label=fe_lab, spec=key,
                             label=label, beta_county=bc, se_county=sc,
                             se_county_raw=sc0, beta_cells=bm, se_cells=sm,
                             se_cells_raw=sm0, gap=gap, gap_se=gse, gap_lo=glo,
                             gap_hi=ghi, n_county=int(fin.sum()),
                             n_cells=n_cellrows, exact=exact))
            print(f"    {label:52s} county {bc:+.6f}  cells {bm:+.6f}  "
                  f"gap {gap:+.3e}  [{'HOLDS' if exact else 'fails'}]")
            if exact:
                check(abs(gap) < TOL_INVAR,
                      f"P1 midpoint growth is invariant across the levels "
                      f"({rung}, {fe_key} FE)",
                      f"|beta_cells - beta_county| = {abs(gap):.3e} < "
                      f"{TOL_INVAR:.0e}; county {bc:+.12f}, cells {bm:+.12f}")
                # A.1.6's "identical standard errors" is a statement about the
                # clustered VARIANCE. The finite-sample correction is the one part
                # that knows how many rows the regression ran on - 65,054 against
                # 19.7 million - so it is the uncorrected SE that has to coincide,
                # and the corrected pair is reported with both row counts beside it.
                check(abs(sm0 - sc0) < TOL_INVAR_SE * max(sc0, 1e-12),
                      f"P1 and A.1.6, the clustered variance coincides too "
                      f"({rung}, {fe_key} FE)",
                      f"uncorrected |se_cells - se_county| = {abs(sm0 - sc0):.3e} "
                      f"on {sc0:.10f}; corrected {sc:.6f} vs {sm:.6f}, which differ "
                      f"only by G/(G-1)(N-1)/(N-K) at N = {int(fin.sum()):,} against "
                      f"N = {n_cellrows:,}")
            else:
                sig = ("" if not np.isfinite(glo) else
                       " and its bootstrap interval EXCLUDES zero" if glo * ghi > 0
                       else " but its bootstrap interval covers zero")
                check(True, f"P1 {key}: the equality is BROKEN ({rung}, {fe_key} FE)",
                      f"gap {gap:+.6f}, {100 * gap / bc:+.1f}% of the county "
                      f"coefficient {bc:+.6f}{sig}")

    rows += invariance_poisson(rung, d, frame, cy, yr, cl, x, z, negative)

    if direct_check:
        _assert_collapse_direct(rung, d, frame, mpgr, w, cy, yr, cl, x, z, k_cell)
    return rows


def invariance_poisson(rung, d, frame, cy, yr, cl, x, z, negative="floor"):
    """A.1.5's OTHER admissible answer, at the fixed effects the proposition names.

    Poisson is the one specification in D1b whose invariance does NOT come from the
    collapse, and whose admissible cell effect is `ccg x industry` rather than a
    function of (c, t). It is exact for a different reason, and the reason is worth
    writing down because it is what makes Panel C an answer rather than a robustness
    check.

    With cell effects a_im and year effects d_t, mu_imt = exp(a_im + d_t + b x_ct), so

        sum_m mu_imt = exp(d_t + b x_ct) sum_m exp(a_im) = exp(A_i + d_t + b x_ct)

    which is the county-level model with unit effect A_i = log sum_m exp(a_im). Summing
    the cell first-order conditions over m therefore delivers the county ones exactly:
    the cell solution (b, A_i, d_t) solves the county problem, and beta is the same
    number. That is a statement about the estimator, not about the weights, which is
    why it survives a cell effect that would break the midpoint rows.

    IT IS REPORTED AND NOT ASSERTED, for a reason that is in the data and not in the
    algebra. The derivation needs sum_m E_imt = E_ct, and Poisson needs E >= 0, and
    EFSY's negative cells make those two incompatible: dropping the 1,651 negative
    cell-years at NAICS4 removes employment the county series still carries. So the
    gap that would be zero is instead a measure of what that costs, and the amount of
    employment involved is printed beside it.
    """
    print(f"\n[D1b] Proposition 1, Poisson on levels, at A.1.5's fixed effects ({rung})")
    y_cty = cy["lev"]
    raw = d["cur"]
    f_ky_year = yr[d["f_cy"]]

    # The SAME sample rule as Panel C, through the same function, so the two Poisson
    # numbers in this file cannot be built on two different conventions.
    y_cell, msk, _cnt = poisson_sample(rung, d, d["f_cell"], f_ky_year, negative)
    nonneg = raw >= 0.0
    lost = float(raw[~nonneg].sum())

    # THE CONTROL FUNCTION IS BUILT ONCE, AT THE COUNTY, AND CARRIED DOWN. Panel C
    # residualises inside whatever fixed effects that panel uses, which is right when
    # nothing is being compared across levels. Here it would be fatal: residualising
    # the shock inside `ccg x industry` under IRLS weights gives a DIFFERENT v from
    # residualising it inside `ccg`, and the two levels would then be estimating two
    # different models, whose betas have no reason to agree. Proposition 1 is about one
    # model at two levels. v is a county-level object anyway - the shock and the
    # instrument both vary only in (c, t) - so building it once is also the more
    # defensible choice, and it is the difference between a gap of -0.1% and a gap at
    # the precision the negative cells allow.
    v_c = control_function(x, z, np.ones(len(x)), cl, yr)

    a1 = pd.factorize(d["f_cell"][msk])[0].astype(np.int64)
    a2 = pd.factorize(f_ky_year[msk])[0].astype(np.int64)
    ac = pd.factorize(cl[d["f_cy"]][msk])[0].astype(np.int64)
    xx = x[d["f_cy"]][msk]
    rm = poisson_cf(y_cell[msk], xx, v_c[d["f_cy"]][msk], a1, a2, ac)

    kc = poisson_separated(np.maximum(y_cty, 0.0), cl, yr)
    b1 = pd.factorize(cl[kc])[0].astype(np.int64)
    b2 = pd.factorize(yr[kc])[0].astype(np.int64)
    rc = poisson_cf(y_cty[kc], x[kc], v_c[kc], b1, b2, b1)

    gap = rm["beta"] - rc["beta"]
    print(f"    {'Employment levels, Poisson':52s} county {rc['beta']:+.6f}  "
          f"cells {rm['beta']:+.6f}  gap {gap:+.3e}  [measured]")

    # AND THE RESIDUAL IS SHOWN TO BE THE NEGATIVE CELLS, not asserted to be. Drop
    # every county-year carrying one, from BOTH levels by the same rule, and the two
    # betas have to coincide - because on what is left, sum_m E_imt = E_ct holds and
    # the derivation above has no remaining obstruction. If this did not close, the
    # gap would be evidence against the proposition rather than against the data, and
    # there would be no way to tell the two apart from the number above alone.
    negcy = np.zeros(len(frame), bool)
    negcy[np.unique(d["f_cy"][~nonneg])] = True
    keep_c = ~negcy
    cmsk = msk & keep_c[d["f_cy"]]
    c1 = pd.factorize(d["f_cell"][cmsk])[0].astype(np.int64)
    c2 = pd.factorize(f_ky_year[cmsk])[0].astype(np.int64)
    cc = pd.factorize(cl[d["f_cy"]][cmsk])[0].astype(np.int64)
    rm2 = poisson_cf(y_cell[cmsk], x[d["f_cy"]][cmsk], v_c[d["f_cy"]][cmsk], c1, c2, cc)
    kc2 = kc & keep_c
    e1 = pd.factorize(cl[kc2])[0].astype(np.int64)
    e2 = pd.factorize(yr[kc2])[0].astype(np.int64)
    rc2 = poisson_cf(y_cty[kc2], x[kc2], v_c[kc2], e1, e2, e1)
    gap2 = rm2["beta"] - rc2["beta"]
    check(abs(gap2) < TOL_POIS_INVAR,
          f"P1 Poisson IS invariant once sum_m E = E_c holds ({rung})",
          f"dropping the {int(negcy.sum()):,} county-years that carry a negative cell "
          f"from BOTH levels: county {rc2['beta']:+.10f}, cells {rm2['beta']:+.10f}, "
          f"gap {gap2:+.2e} < {TOL_POIS_INVAR:.0e}. On the full sample the gap is "
          f"{gap:+.2e}, so the residual there IS the {int((~nonneg).sum()):,} negative "
          f"cell-years - {abs(lost):,.0f} employees against "
          f"{float(y_cty.sum()):,.0f} - and not a failure of the proposition")
    return [dict(rung=rung, fe="poisson",
                 fe_label=r"$ccg$ (county) / $ccg\times$ind (cells), + year",
                 spec="poisson", label=r"Employment levels, Poisson",
                 beta_county=rc["beta"], se_county=rc["se"], se_county_raw=np.nan,
                 beta_cells=rm["beta"], se_cells=rm["se"], se_cells_raw=np.nan,
                 gap=gap, gap_se=np.nan, gap_lo=np.nan, gap_hi=np.nan,
                 n_county=rc["n"], n_cells=rm["n"], exact=False)]


def _assert_collapse_direct(rung, d, frame, mpgr, w, cy, yr, cl, x, z, k_cell):
    """The collapse is asserted against a direct cell-level fit, not assumed.

    Run at NAICS2 only, where 1.7M rows is affordable, and at BOTH fixed-effect
    blocks, because the collapse argument is about the fixed effect being a function
    of (c, t) and one block would not test that. If these disagree the aggregation is
    wrong and every number in D1b is void, so it is a hard check and not a diagnostic.
    """
    print(f"\n[D1b] the collapse, against a direct cell-level fit ({rung})")
    n_frame, n_cellrows = len(frame), len(mpgr)
    Ybar, T = collapse(mpgr, w, d["f_cy"], n_frame)
    cy_ix = d["f_cy"]
    cl_cell = cl[cy_ix]
    for fe_key, fe_lab, unit_fe in FE_BLOCKS:
        f1, f2 = ((cl, yr) if unit_fe else (yr, None))
        bc, sc, _ = fit_collapsed(Ybar, x, z, T, f1, f2, cl, n_cellrows, k_cell)
        g1, g2 = ((cl_cell, yr[cy_ix]) if unit_fe else (yr[cy_ix], None))
        M = _within(np.column_stack([mpgr, x[cy_ix], z[cy_ix]]), w, g1, g2)
        zx = float(M[:, 2] @ (w * M[:, 1]))
        bd = float(M[:, 2] @ (w * M[:, 0])) / zx
        sd = crv1(M[:, 2], M[:, 1], M[:, 0] - bd * M[:, 1], w, cl_cell,
                  n_cellrows, k_cell)
        check(abs(bd - bc) < TOL_COLLAPSE and abs(sd - sc) < TOL_COLLAPSE,
              f"P1 the collapse reproduces the direct {n_cellrows:,}-row cell fit "
              f"({rung}, {fe_key} FE)",
              f"beta {bd:+.12f} vs {bc:+.12f} (dev {abs(bd - bc):.2e}), "
              f"se {sd:.12f} vs {sc:.12f} (dev {abs(sd - sc):.2e})")


# --------------------------------------------------------------------------------
# LaTeX. Emitted by the run, so a table can never drift from the estimates behind it.
# --------------------------------------------------------------------------------

PANEL_LABEL = {
    "A": (r"\emph{Panel A: log growth, OLS}", r"Contract shock"),
    "A2": (r"\emph{Panel A2: log growth, MPGR weight}", r"Contract shock"),
    "B": (r"\emph{Panel B: midpoint growth, WLS}", r"Contract shock"),
    "C": (r"\emph{Panel C: employment levels, Poisson}", r"Contract shock"),
}


def bc_note(res, rungs, negative):
    r"""The note that travels with Table D1, built from the run's own counts.

    It exists because "Observations, panel B" and "Observations, panel C" differ by
    millions of rows and the table cannot say why on its own - and the first guess a
    reader makes, that the exits are missing, is the wrong one.

    Built with f-strings and never with %-formatting: every line of this note begins
    with a LaTeX comment character, and `"%   %s" % nm` reads that leading `%` as a
    conversion specifier and raises. The whole function is one long literal `%`, so the
    formatting operator is the wrong tool for it.
    """
    L = ["%", "% NOTE - why Panels B and C report different observation counts.",
         "% Panel B keeps every cell-year. Panel C loses two things, and NEITHER is the",
         "% exits or the entries: those are observations in both panels, because midpoint",
         "% growth takes an exit at -2 and Poisson takes it at y = 0.",
         "%"]
    for rg in rungs:
        c = res.get(rg, {}).get("bc")
        if not c:
            continue
        nm = PRETTY_RUNG[rg].replace(r"County $\times$ ", "county x ")
        n_drop = 0 if negative == "floor" else c["negative"]
        kept = c["exit"] - c["sep_exit"]
        pct_kept = 100 * (1 - c["sep_exit"] / max(c["exit"], 1))
        pct_nolog = 100 * (c["n_total"] - c["both"]) / c["n_total"]
        L += [
            f"%   {nm}",
            f"%     cell-years in the panel                        {c['n_total']:>12,}",
            f"%     less cells with NO employment in ANY year of   {-c['separated']:>12,}",
            f"%          the window (Poisson effect at -inf)",
            f"%     less negative-employment cells, mode `{negative}`   {-n_drop:>12,}",
            f"%     = Panel C                                      "
            f"{c['n_total'] - c['separated'] - n_drop:>12,}",
            f"%",
            f"%     of {c['exit']:,} exits, {kept:,} are in BOTH panels ({pct_kept:.1f}%);"
            f" the {c['sep_exit']:,} that are not",
            f"%       are cells that died before 1977, so the window never sees them alive",
            f"%     all {c['entry']:,} entries are in both panels",
            f"%     Panel A takes neither: log growth is undefined on {pct_nolog:.1f}% of",
            f"%       the panel, including every exit and every entry",
            "%",
        ]
    L += ["% The row that SHOULD agree does: Panels B and C both estimate on 100.00% of",
          "% county employment. Panel A does not. B and C differ by cells that carried",
          "% nothing; A differs by cells that carried something.", "%"]
    return "\n".join(L)


def tex_d1(fit, cols, path, h, exposure, y0, y1, panels, note=""):
    def cell(pan, col, what):
        m = fit[(fit["panel"] == pan) & (fit["column_key"] == col)
                & (fit["estimator"] != "OLS")]
        return None if not len(m) else m.iloc[0][what]

    L = [r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
         r"\begin{tabular}{@{\extracolsep{\fill}}l*{%d}{c}}" % len(cols), r"\toprule",
         "                " + "".join(r"&\multicolumn{1}{c}{%s}" % lab
                                      for _, lab in cols) + r"\\",
         "                " + "".join(r"&\multicolumn{1}{c}{(%d)}" % (i + 1)
                                      for i in range(len(cols))) + r"\\",
         r"\midrule"]
    for pan in [p for p in ("A", "A2", "B", "C") if p in panels]:
        head, rowlab = PANEL_LABEL[pan]
        L.append("%-46s" % head + "&" * len(cols) + r"\\")
        r1, r2 = "%-46s" % rowlab, " " * 46
        for key, _ in cols:
            b, se = cell(pan, key, "beta"), cell(pan, key, "se")
            if b is None or not np.isfinite(b):
                r1 += "&          "; r2 += "&          "
            else:
                r1 += "&%10.4f%s" % (b, w22._stars(b, se))
                r2 += "&%10s" % ("(%.4f)" % se)
        L += [r1 + r"\\", r2 + r"\\", r"\addlinespace"]
    L.append(r"\midrule")

    def plain(label, fn):
        row = "%-46s" % label
        for key, _ in cols:
            row += "&%10s" % fn(key)
        return row + r"\\"

    L.append(plain(r"Industry $\times$ year FE",
                   lambda k: "no" if k == "county" else "yes"))
    L.append(plain(r"Unit / cell FE",
                   lambda k: "ccg" if k == "county" else r"ccg$\times$ind"))
    for pan in [p for p in ("A", "A2", "B", "C") if p in panels]:
        L.append(plain(r"Observations, panel %s" % pan,
                       lambda k, p=pan: ("{:,}".format(int(cell(p, k, "n_fit")))
                                         if cell(p, k, "n_fit") is not None else "")))
    L.append(plain(r"Cells with zero employment (\%)",
                   lambda k: "n/a" if k == "county"
                   else "%.1f" % (100 * cell("B", k, "zero_cell_share"))))
    # Two decimals, not one. At NAICS2 Panel A retains 99.97% of county employment and
    # at NAICS4 it retains 99.31%; rounded to one decimal the first prints as 100.0,
    # which is the one thing this row must not say.
    for pan in [p for p in ("A", "B", "C") if p in panels]:
        L.append(plain(r"Share of county employment, panel %s (\%%)" % pan,
                       lambda k, p=pan: "100.00" if k == "county"
                       else ("%.2f" % (100 * cell(p, k, "emp_share"))
                             if cell(p, k, "emp_share") is not None else "")))
    if "A2" in panels:
        L.append(plain(r"$\sum_m \omega_{im}$ retained, panel A2",
                       lambda k: "1.000" if k == "county"
                       else "%.3f" % cell("A2", k, "wsum_mean")))
    row = "%-46s" % r"First-stage $F$"
    for key, _ in cols:
        v = cell("B", key, "fs_F")
        row += "&%10s" % ("%.1f" % v if v is not None and np.isfinite(v) else "")
    L += [row + r"\\", r"\bottomrule", r"\end{tabular}"]
    path.write_text(
        "%% Table D1, the disaggregation test. h=%d, z_%s, 2SLS, %d-%d, clustered on the CCG\n"
        "%% A -> A2 changes only the weight; A2 -> B changes only the transform. "
        "Panel C's SE treats the control function as data.\n"
        "%s%s\n" % (h, exposure, y0, y1, (note + "\n") if note else "", "\n".join(L)))
    return path


def tex_d1b(inv, path, h, exposure, y0, y1, rungs):
    """One County column, then (Cells, Gap) per rung, one block per fixed effect.

    The county side does not depend on the rung - the same county regression sits
    behind both - so printing it once is not a compression, it is the point. A check
    in the run asserts the two rungs agree on it before this ever prints.
    """
    keys = [r for r in ("naics2", "naics4_alloc") if r in rungs]
    L = [r"\begin{tabular}{@{\extracolsep{\fill}}lc%s}" % ("cc" * len(keys)),
         r"\toprule",
         "                &" + r"\multicolumn{1}{c}{County}"
         + "".join(r"&\multicolumn{2}{c}{%s}" % PRETTY_RUNG[r] for r in keys) + r"\\"
         + " ".join(r"\cmidrule(lr){%d-%d}" % (3 + 2 * i, 4 + 2 * i)
                    for i in range(len(keys))),
         "                &" + r"\multicolumn{1}{c}{$\beta_b$}"
         + "".join(r"&\multicolumn{1}{c}{$\beta_d$}&\multicolumn{1}{c}{Gap}"
                   for _ in keys) + r"\\", r"\midrule"]
    order = ["midpoint", "log_unw", "log_w", "standard", "poisson"]
    # The blocks are driven by what is IN the frame, with their labels, so adding one
    # does not mean editing this function - the Poisson block carries a different cell
    # fixed effect from the two growth blocks and would otherwise need a special case.
    for fe_key in [k for k in ("ccgyear", "year", "poisson") if k in set(inv["fe"])]:
        blk = inv[inv["fe"] == fe_key]
        fe_lab = blk.iloc[0]["fe_label"]
        L.append((r"\emph{Both levels absorb %s}" if fe_key != "poisson"
                  else r"\emph{%s}") % fe_lab + "&" * (1 + 2 * len(keys)) + r"\\")
        for key in order:
            sub = blk[blk["spec"] == key]
            if not len(sub):
                continue
            m0 = sub.iloc[0]
            r1 = "%-46s&%9.4f" % (m0["label"], m0["beta_county"])
            r2 = " " * 46 + "&%9s" % ("(%.4f)" % m0["se_county"])
            for rg in keys:
                m = sub[sub["rung"] == rg]
                if not len(m):
                    r1 += "&    &    "; r2 += "&    &    "
                    continue
                m = m.iloc[0]
                # The Poisson gap is 1.4e-05 and 3.8e-05 and would print as +0.0000 at
                # four decimals - which in a table about exactness reads as rounding
                # rather than as a result. It gets scientific notation, so the row says
                # what it actually found.
                if m["exact"]:
                    g = r"$<10^{-11}$"
                elif abs(m["gap"]) < 5e-4:
                    e = int(np.floor(np.log10(abs(m["gap"]))))
                    g = r"$%+.1f\!\times\!10^{%d}$" % (m["gap"] / 10.0 ** e, e)
                else:
                    g = "%+.4f" % m["gap"]
                r1 += "&%9.4f&%12s" % (m["beta_cells"], g)
                r2 += "&%9s&%12s" % (
                    "(%.4f)" % m["se_cells"],
                    ("" if m["exact"] or not np.isfinite(m["gap_lo"])
                     else "[%+.3f, %+.3f]" % (m["gap_lo"], m["gap_hi"])))
            L += [r1 + r"\\", r2 + r"\\"]
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text(
        "%% Table D1b. Proposition 1 at a fixed effect that is a function of (c,t) at "
        "BOTH levels. h=%d, z_%s, %d-%d, 2SLS, clustered on the CCG\n"
        "%% Midpoint growth is invariant to machine precision; log growth is NOT, and "
        "that is the exhibit. The cell side carries a plain ccg effect, not "
        "ccg x industry - a cell effect is not a function of (c,t) and breaks the "
        "invariance by design, which is what Table D1's printed columns do.\n"
        "%% Standard errors in parentheses; the county and cell clustered VARIANCES "
        "coincide on the midpoint row and their reported SEs differ only by the "
        "degrees-of-freedom correction. Brackets are 2.5/97.5 percentiles of a "
        "200-replicate paired cluster bootstrap over CCGs.\n%s\n"
        % (h, exposure, y0, y1, "\n".join(L)))
    return path


def tex_coeff(fit, inv, path, h, y0, y1):
    out = [r"% Auto-generated by 28_table_d1.py. Do not edit.",
           r"% Every number a draft quotes from plan section 2.3 comes from here."]
    tag = {"county": "County", "naics2": "Two", "naics4_alloc": "Four"}
    # A LaTeX control word is LETTERS ONLY, so the panel names cannot go into a macro
    # name verbatim: \DOneCountyA2 does not parse, and the failure would surface as an
    # error in whichever draft imported this file rather than here.
    PAN_TOKEN = {"A": "LogUnw", "A2": "LogW", "B": "Mid", "C": "Pois"}
    for key, nm in tag.items():
        for pan in ("A", "A2", "B", "C"):
            m = fit[(fit["panel"] == pan) & (fit["column_key"] == key)
                    & (fit["estimator"] != "OLS")]
            if not len(m):
                continue
            m, pt = m.iloc[0], PAN_TOKEN[pan]
            out.append(r"\newcommand{\DOne%s%s}{%.4f}" % (nm, pt, m["beta"]))
            out.append(r"\newcommand{\DOne%s%sSE}{%.4f}" % (nm, pt, m["se"]))
            out.append(r"\newcommand{\DOne%s%sN}{%s}"
                       % (nm, pt, f"{int(m['n_fit']):,}"))
            if np.isfinite(m.get("emp_share", np.nan)):
                out.append(r"\newcommand{\DOne%s%sEmpShare}{%.2f}"
                           % (nm, pt, 100 * m["emp_share"]))
    for key, nm in (("naics2", "Two"), ("naics4_alloc", "Four")):
        m = fit[(fit["panel"] == "B") & (fit["column_key"] == key)]
        if len(m):
            out.append(r"\newcommand{\DOne%sZeroCells}{%.1f}"
                       % (nm, 100 * float(m.iloc[0]["zero_cell_share"])))
        # The fixed-effect block is IN THE MACRO NAME. Both blocks are canonical, and a
        # sentence quoting a gap has to say which one it means rather than inheriting
        # whichever pandas happened to order first - the same discipline 22 applies to
        # its two cutoff rules.
        for fe_key, fesfx in (("ccgyear", ""), ("year", "YearFE"),
                              ("poisson", "")):
            for spec, snm in (("midpoint", "Midpoint"), ("log_unw", "LogUnw"),
                              ("log_w", "LogW"), ("standard", "Standard"),
                              ("poisson", "Pois")):
                s = inv[(inv["rung"] == key) & (inv["spec"] == spec)
                        & (inv["fe"] == fe_key)]
                if not len(s):
                    continue
                s = s.iloc[0]
                out.append(r"\newcommand{\POne%s%s%sCounty}{%.4f}"
                           % (nm, snm, fesfx, s["beta_county"]))
                out.append(r"\newcommand{\POne%s%s%sCells}{%.4f}"
                           % (nm, snm, fesfx, s["beta_cells"]))
                out.append(r"\newcommand{\POne%s%s%sGap}{%.4f}"
                           % (nm, snm, fesfx, s["gap"]))
                if not s["exact"] and np.isfinite(s["gap_lo"]):
                    out.append(r"\newcommand{\POne%s%s%sGapPct}{%.1f}"
                               % (nm, snm, fesfx, 100 * s["gap"] / s["beta_county"]))
    out.append(r"\newcommand{\DOneWindow}{%d--%d}" % (y0, y1))
    out.append(r"\newcommand{\DOneHorizon}{%d}" % h)
    path.write_text("\n".join(out) + "\n")
    return path


# --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", nargs="+", default=list(RUNGS_ALL),
                    choices=list(RUNGS_ALL))
    ap.add_argument("--panel", nargs="+", default=list(PANELS_ALL),
                    choices=list(PANELS_ALL))
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--exposure", default=EXPOSURE_DEFAULT)
    # EFSY's negative cells, which only Panel C has to decide about. `floor` is
    # canonical since 2026-08-14: A.3.1 is the only rule that speaks to the case and
    # it forbids dropping a cell. `drop` stays runnable so the decision is revisable
    # rather than deleted - 22's convention for a withdrawn option.
    ap.add_argument("--negative", choices=["floor", "drop"], default="floor")
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the pyfixest validation gates (they are the reason "
                         "the NAICS4 numbers are readable; use only for staging)")
    args = ap.parse_args()

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    canonical = (set(args.rung) == set(RUNGS_ALL) and set(args.panel) == set(PANELS_ALL)
                 and args.horizon == HORIZON and args.exposure == EXPOSURE_DEFAULT
                 and args.negative == "floor" and not args.no_validate)
    sfx = "" if canonical else ("__" + "-".join(sorted(args.rung)) + "_"
                                + "".join(sorted(args.panel)) + f"_h{args.horizon}"
                                + ("" if args.negative == "floor"
                                   else f"_neg{args.negative}"))

    tee = Tee(dc.LOG_DIR / f"28_table_d1{sfx}.log")
    import pyfixest as pf
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
    t0 = time.time()
    h, exposure = args.horizon, args.exposure
    eras, straddle = periods_for(h)
    y0, y1 = eras["pre"]
    est_years = list(range(y0, y1 + 1))

    print("=" * 78)
    print("Plan section 2.3 - Table D1, the disaggregation test")
    print("=" * 78)
    print(f"  h = {h}, exposure = {exposure}, instrument = z_{exposure}")
    print(f"  rungs    {args.rung}; panels {args.panel}")
    print(f"  window   pre {y0}-{y1}; straddle {straddle[0]}-{straddle[1]}, "
          f"estimated nowhere")
    print(f"  POST-{w22.NAICS_BREAK} IS DROPPED ENTIRELY, on instruction; step 1")
    print("  section 1.9.5 is where the weak half is named.")
    print("\n  DECLARED DECISIONS, recorded before any number is read.")
    print("  - the midpoint guard is `den != 0`, never `den > 0`")
    print("  - no cell is dropped and no growth rate is winsorized (A.3.1, A.3.2)")
    print("  - the treatment is the COUNTY shock at every level and is never split")
    print("    along the industry dimension (A.1.1)")
    print("  - ccg x year appears nowhere. It would absorb the treatment (A.1.4)")
    print("  - county-years where the reconstruction identity cannot hold are")
    print("    excluded from EVERY column by the same rule, as the UNION over the")
    print("    rungs run, so all three columns estimate on identical county-years")
    print("    (A.1.6). Named individually below")
    print("  - Panel A is INADMISSIBLE as an answer (A.1.5) and is reported because")
    print("    it is what a replicating paper would run")
    print("  - Panel C's SE treats the control function as data and understates")
    print("  - MPGR weights are used UNCLAMPED, negatives included, because the")
    print("    reconstruction is what makes the invariance exact; the count is logged")

    iv_disk, _ = w22.provenance_gate(h)
    iv = w22.build_exposure_panel(h, exposure, iv_disk, (y0, y1))
    frame = w22.build_frame(iv, h, exposure, est_years)
    frame = frame[frame["year"].between(y0, y1)].reset_index(drop=True)
    print(f"    the pre frame: {len(frame):,} county-years, "
          f"{frame['ccg'].nunique():,} CCGs, {y0}-{y1}")
    w22.step1_crosscheck(frame, (y0, y1))

    years_emp = list(range(y0 - h, y1 + 1))
    emps, codes, excl, res = {}, {}, {}, {}

    # ---- pass 1: 27's four assertions, and the exclusion set, per rung ----------
    for rung in args.rung:
        print(f"\n{'=' * 78}\n  rung {rung}: 27's construction assertions"
              f"\n{'=' * 78}")
        emp, ccg_arr = w22.load_rung(rung, years_emp)
        _, code_arr = c27.rung_codes(rung, years_emp[0])
        county_src = {
            "efsy": pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                                    columns=["ccg", "year", "emp"]),
            "county_panel": pd.read_parquet(dc.PANEL_OUT_DIR / "county_panel.parquet",
                                            columns=["ccg", "year", "emp"]),
        }
        c27.assert_sums(rung, emp, ccg_arr, years_emp, county_src)
        m2 = c27.support_completeness(rung, ccg_arr, code_arr)
        _, m4 = c27.assert_reconstruction(rung, emp, ccg_arr, frame, h)
        _, m3 = c27.assert_survives_fe(rung, emp, ccg_arr, code_arr, frame, h)
        s = excluded_county_years(rung, emp, ccg_arr, frame, h)
        check(len(s) == m4["a4_excluded"],
              f"E  the exclusion set matches 27's count ({rung})",
              f"{len(s)} county-years recomputed against 27's "
              f"{m4['a4_excluded']}; independent computations of the same object")
        excl[rung] = s
        emps[rung], codes[rung] = (emp, ccg_arr), code_arr
        res[rung] = {**m2, **m3, **m4}
        del county_src

    union = set().union(*excl.values()) if excl else set()
    print(f"\n[0b] the county-years excluded from EVERY column, by name")
    print(f"    {len(union)} in the union over {list(excl)}: "
          + ("; ".join(f"ccg {c} {y}" for c, y in sorted(union)[:40])
             + (" ..." if len(union) > 40 else "") if union else "none"))
    if union:
        drop = pd.MultiIndex.from_tuples(sorted(union))
        keep = ~pd.MultiIndex.from_arrays(
            [frame["ccg"], frame["year"]]).isin(drop)
        frame = frame[keep].reset_index(drop=True)
    # Both indices are dense and SORTED, not order-of-appearance, so the year index
    # here and build_long's `yr_of` are the same map and f_ky can be read against
    # frame["year_idx"] without a second lookup.
    frame["ccg_idx"] = np.searchsorted(np.unique(frame["ccg"]), frame["ccg"])
    frame["year_idx"] = np.searchsorted(np.unique(frame["year"]), frame["year"])
    fhash = hash_index(frame["ccg"].to_numpy(), frame["year"].to_numpy())
    print(f"    the estimation frame: {len(frame):,} county-years, "
          f"{frame['ccg'].nunique():,} CCGs, index hash {fhash}")
    check(frame["ccg"].nunique() == N_SUPPORT,
          "0b every column estimates on step 1's frozen 3,100 CCGs (A.1.6)",
          f"{frame['ccg'].nunique():,} CCGs on {len(frame):,} county-years after "
          f"removing {len(union)} reconstruction exclusions")

    cy = county_outcomes(frame)
    n_nolog = int((~np.isfinite(cy["log"])).sum())
    check(True, "0c county log growth is undefined where employment reached zero",
          f"{n_nolog} county-years; they leave BOTH sides of every log comparison "
          f"by the same rule")

    # ---- column (1): the county rung -------------------------------------------
    fit_rows = []
    if any(p in args.panel for p in ("A", "A2", "B", "C")):
        fit_rows += run_county(frame, cy, args.panel, not args.no_validate)

    # ---- pass 2: the cell rungs -------------------------------------------------
    inv_rows = []
    for rung in args.rung:
        print(f"\n{'=' * 78}\n  rung {rung}: Table D1 columns and Table D1b"
              f"\n{'=' * 78}")
        emp, ccg_arr = emps[rung]
        d = build_long(rung, emp, ccg_arr, codes[rung], frame, h)
        assert_cells_against_county(rung, d, frame, cy)
        res[rung]["zero_cell_share"] = float((d["cur"] == 0.0).mean())
        n_negw = int((d["cur"] + d["lag"] < 0).sum())
        print(f"    {len(d['cur']):,} cell-years, {d['n_cell']:,} ccg x industry "
              f"levels, {d['n_code'] * d['n_year']:,} industry x year levels")
        print(f"    zero-employment cell-years {100 * res[rung]['zero_cell_share']:.1f}%"
              f", negative MPGR weights {n_negw:,} "
              f"({100 * n_negw / len(d['cur']):.4f}%), carried UNCLAMPED")
        if any(p in args.panel for p in ("A", "A2", "B", "C")):
            new = run_panels(rung, d, frame, cy,
                             [p for p in args.panel if p != "D1b"],
                             rung, res, not args.no_validate, args.negative)
            fit_rows += new
            nb = [r for r in new if r["panel"] == "B" and r["estimator"] == "2SLS"]
            nc = [r for r in new if r["panel"] == "C"]
            if nb and nc:
                res[rung]["bc"] = bc_decomposition(rung, d, args.negative,
                                                   nb[0]["n_fit"], nc[0]["n_fit"])
        if "D1b" in args.panel:
            inv_rows += invariance_block(rung, d, frame, cy, args.nboot, args.seed,
                                         direct_check=(rung == "naics2"),
                                         negative=args.negative)
        del d, emp
        emps[rung] = None

    # ---- outputs ----------------------------------------------------------------
    fit = pd.DataFrame(fit_rows)
    if len(fit):
        fit["column_key"] = fit["rung"]
        # Panels A and A2 differ ONLY in the weight, so they must run on byte-identical
        # rows; if they ever stop doing so, A -> A2 stops being the weighting channel
        # and becomes a sample change wearing its name. Equal n_obs is not equal rows,
        # which is why this is a hash and not a count.
        for rg, grp in fit[fit["estimator"] == "2SLS"].groupby("rung"):
            hs = {p: grp[grp["panel"] == p]["row_hash"].iloc[0]
                  for p in ("A", "A2") if (grp["panel"] == p).any()}
            if len(hs) == 2:
                check(hs["A"] == hs["A2"],
                      f"H  panels A and A2 run on identical rows ({rg})",
                      f"row hash {hs['A']} vs {hs['A2']}; the two differ only in the "
                      f"weight, so A -> A2 is the weighting channel and nothing else")
        fit.to_csv(dc.LOG_DIR / f"d1_coefficients{sfx}.csv", index=False)
        print(f"\n  {len(fit):,} fits -> {dc.LOG_DIR / f'd1_coefficients{sfx}.csv'}")
    inv = pd.DataFrame(inv_rows)
    if len(inv):
        inv.to_csv(dc.LOG_DIR / f"d1_invariance{sfx}.csv", index=False)
        print(f"  {len(inv):,} invariance rows -> "
              f"{dc.LOG_DIR / f'd1_invariance{sfx}.csv'}")
        # D1b prints ONE county column for both rungs. That is only honest if the two
        # rungs agree on it, and they must: the county side never touches a cell file.
        # A disagreement would mean the frame moved between the rungs.
        sp = inv.groupby(["fe", "spec"])["beta_county"]
        worst = float((sp.max() - sp.min()).max())
        check(worst < 1e-14, "P1 the county side is identical at both rungs",
              f"max spread {worst:.2e} over {sp.ngroups} (fixed effect, spec) cells; "
              f"D1b prints it once")

    if canonical:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        cols = [("county", "County")] + [
            (r, PRETTY_RUNG[r]) for r in ("naics2", "naics4_alloc") if r in args.rung]
        p = tex_d1(fit, cols, RESULT_DIR / "TableD1_disaggregation.tex", h,
                   exposure, y0, y1, args.panel,
                   note=bc_note(res, [r for r in ("naics2", "naics4_alloc")
                                      if r in args.rung], args.negative))
        print(f"  table -> {p}")
        if len(inv):
            p = tex_d1b(inv, RESULT_DIR / "TableD1b_invariance.tex", h, exposure,
                        y0, y1, args.rung)
            print(f"  table -> {p}")
        p = tex_coeff(fit, inv, RESULT_DIR / "coeff_d1.tex", h, y0, y1)
        print(f"  coeff -> {p}")
    else:
        print(f"\n  PARTIAL RUN. The canonical outputs are left untouched; this run "
              f"writes to *{sfx}.* instead.")

    print("\n" + "=" * 78)
    print("  SUMMARY - Table D1, 2SLS")
    print("=" * 78)
    # Column order follows the emitted table - county, NAICS2, NAICS4 - and not
    # args.rung, so the summary and TableD1 cannot read left-to-right differently.
    order = ["county"] + [r for r in ("naics2", "naics4_alloc") if r in args.rung]
    print("  %-38s" % "" + "".join(
        "%18s" % ("County" if k == "county"
                  else PRETTY_RUNG[k].replace(r"County $\times$ ", ""))
        for k in order))
    for pan in ("A", "A2", "B", "C"):
        if pan not in args.panel:
            continue
        line = "  %-38s" % PANEL_LABEL[pan][0].replace(r"\emph{", "").replace("}", "")
        for key in order:
            m = fit[(fit["panel"] == pan) & (fit["rung"] == key)
                    & (fit["estimator"] != "OLS")]
            line += "%18s" % ("%+.4f (%.4f)" % (m.iloc[0]["beta"], m.iloc[0]["se"])
                              if len(m) else "")
        print(line)

    print(f"\n  elapsed {time.time() - t0:.0f}s")
    if w22.failures:
        print(f"\nFAIL: {len(w22.failures)} check(s) failed", file=sys.stderr)
        for n, w in w22.failures:
            print(f"  {n}: {w}", file=sys.stderr)
        tee.close()
        return 1
    print("\nall checks passed")
    tee.close()
    return 0


def run_county(frame, cy, panels, val_pf):
    """Column (1). ccg + year fixed effects, which is what step 1 reports.

    Column (1) cannot carry a county x industry effect - it has no industry dimension
    - so it carries a plain ccg effect, and the cross-check against step 1's county
    coefficient still binds. The two are separate rows and are never folded into one.

    THERE IS NO WEIGHTING CHANNEL AT THIS RUNG. W_c = 1, so Panel A and Panel A2
    coincide here by construction. That is the point rather than a defect: the
    implicit weighting is a cell-level phenomenon, and W_c = 1 against omega_im - which
    sums to one within a county-year - is exactly the pairing under which every county
    carries the same total weight at both levels.
    """
    print(f"\n{'=' * 78}\n  column (1), the county rung\n{'=' * 78}")
    x = frame["shock"].to_numpy()
    z = frame["z"].to_numpy()
    f1 = frame["ccg_idx"].to_numpy().astype(np.int64)
    f2 = frame["year_idx"].to_numpy().astype(np.int64)
    cl = f1
    one = np.ones(len(frame))
    out = []
    for pan, label, y in (("A", "log growth, unweighted", cy["log"]),
                          ("A2", "log growth, MPGR weight (W_c = 1)", cy["log"]),
                          ("B", "midpoint growth, WLS (W_c = 1)", cy["mpgr"])):
        if pan not in panels:
            continue
        msk = np.isfinite(y)
        r = fit2sls(y[msk], x[msk], z[msk], one[msk], f1[msk], f2[msk], cl[msk])
        ro = fit2sls(y[msk], x[msk], x[msk], one[msk], f1[msk], f2[msk], cl[msk],
                     want_F=False)
        print(f"\n[D1] panel {pan} - {label} (county)")
        print(f"    2SLS  beta {r['beta']:+.4f} ({r['se']:.4f})  F {r['F']:.1f}  "
              f"N {r['n']:,}  K {r['k']:,}  singletons dropped {r['n_dropped']:,}")
        print(f"    OLS   beta {ro['beta']:+.4f} ({ro['se']:.4f})")
        for est, rr in (("2SLS", r), ("OLS", ro)):
            out.append(dict(rung="county", column="County", panel=pan, spec=label,
                            estimator=est, beta=rr["beta"], se=rr["se"],
                            fs_F=r["F"] if est == "2SLS" else np.nan,
                            n_fit=rr["n"], k=rr["k"], n_rows_avail=int(msk.sum()),
                            emp_share=1.0, wsum_mean=1.0, wsum_p10=1.0,
                            zero_cell_share=np.nan,
                            row_hash=hash_index(np.flatnonzero(msk)[r["keep"]])))
        if val_pf:
            validate_2sls(y[msk], x[msk], z[msk], one[msk], f1[msk], f2[msk],
                          cl[msk], r, ro, f"county panel {pan}")

    if "C" in panels:
        print("\n[D1] panel C - employment levels, Poisson, control function (county)")
        y = cy["lev"]
        keep = poisson_separated(np.maximum(y, 0.0), f1, f2)
        a1 = pd.factorize(f1[keep])[0].astype(np.int64)
        a2 = pd.factorize(f2[keep])[0].astype(np.int64)
        ac = pd.factorize(cl[keep])[0].astype(np.int64)
        v = control_function(x[keep], z[keep], np.ones(int(keep.sum())), a1, a2)
        r = poisson_cf(y[keep], x[keep], v, a1, a2, ac)
        print(f"    Poisson-CF  beta {r['beta']:+.4f} ({r['se']:.4f})  N {r['n']:,}  "
              f"{r['iters']} IRLS steps")
        if val_pf:
            validate_poisson(y[keep], x[keep], v, a1, a2, ac, r, "county panel C")
        out.append(dict(rung="county", column="County", panel="C",
                        spec="employment levels, Poisson (control function)",
                        estimator="Poisson", beta=r["beta"], se=r["se"],
                        fs_F=np.nan, n_fit=r["n"], k=r["k"],
                        n_rows_avail=len(y), emp_share=1.0, wsum_mean=np.nan,
                        wsum_p10=np.nan, zero_cell_share=np.nan, iters=r["iters"],
                        row_hash=hash_index(np.flatnonzero(keep))))
    return out


def assert_cells_against_county(rung, d, frame, cy):
    """The cell sums ARE the county series the instrument parquet carries.

    27's assertion 1 compares the cell sums against two independent county employment
    files; this compares them against the quantities THIS table's column (1) is built
    out of - emp_lag and emp_growth from instrument_ccg_year.parquet - which is a
    different object again. Without it, column (1) and columns (2)-(3) could be
    describing two different counties and every identity below would still pass.
    """
    lag = np.abs(d["lag_c"] - cy["lag"]) / np.maximum(np.abs(cy["lag"]), 1.0)
    cur = np.abs(d["cur_c"] - cy["lev"]) / np.maximum(np.abs(cy["lev"]), 1.0)
    den = np.abs(d["den_c"] - cy["den"]) / np.maximum(np.abs(cy["den"]), 1.0)
    check(lag.max() < TOL_CURC and cur.max() < TOL_CURC and den.max() < TOL_DENC,
          f"0d the cell sums ARE column (1)'s own county series ({rung})",
          f"max rel dev: lagged {lag.max():.2e}, current {cur.max():.2e}, "
          f"midpoint denominator {den.max():.2e}")
    check(float(np.abs(cy["mpgr"]).max()) <= MPGR_BOUND + 1e-12,
          f"0e county midpoint growth is bounded on [-2, 2] ({rung})",
          f"max |mpgr_c| = {float(np.abs(cy['mpgr']).max()):.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
