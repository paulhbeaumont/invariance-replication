#!/usr/bin/env python3
"""Anderson-Rubin sets and the OLS ladder, at every rung. Part 1.5, after 1.8.6.

Two questions the first-stage F cannot answer, and both bear on whether Part 3's
comparison can be made at all.

WHAT CLAIM EACH RUNG SUPPORTS - the Anderson-Rubin set

    F >= 10 is Staiger-Stock folklore; Stock-Yogo for one instrument is 16.38;
    both are for the homoskedastic Cragg-Donald statistic and not for the
    clustered F that plan 1.5 requires. Lee, McCrary, Moreira and Porter (2022)
    put the threshold for the conventional 1.96 at F > 104.7, which nothing on
    this ladder approaches.

    Anderson-Rubin sidesteps all of it. With one instrument and one endogenous
    regressor the AR set is EXACTLY valid at any instrument strength, and it is
    the optimal weak-identification-robust test in the just-identified case. It
    can come back unbounded, and when it does that is the finding rather than a
    failure of the code.

    For H0: beta = b the AR statistic is the cluster-robust Wald test that z is
    absent from the reduced form of y - b*x, so the set is a quadratic inequality
    in b and is solved in closed form here rather than by a grid search:

        A b^2 + B b + C <= 0
        A = S_zx^2 - c * sum_g b_g^2
        B = 2 * (c * sum_g a_g b_g - S_zy * S_zx)
        C = S_zy^2 - c * sum_g a_g^2

    with a_g and b_g the within-cluster sums of z*y and z*x after the fixed
    effects are absorbed, S_zy = sum_g a_g, S_zx = sum_g b_g, and c the chi-square
    critical value. A > 0 exactly when the first-stage cluster-robust Wald exceeds
    c, which is why a rung with F below 3.84 CANNOT have a bounded set: the
    quadratic opens downward and the complement of an interval is what survives.

    A grid would have to guess a range, and a set that is two infinite rays is
    exactly the case a grid reports as "empty" or "everything" depending on where
    it was centred. The closed form has no such failure mode.

WHETHER THE LADDER IS THE LADDER - OLS at every rung

    Weak instruments bias 2SLS toward OLS, and the bias is worse where F is
    smaller. Our F varies along the ladder, so a coefficient ladder that rises
    while F rises is confounded with a weak-instrument bias that falls. If OLS is
    flat across rungs and 2SLS rises, that is the aggregation effect the project
    is about; if OLS rises too, part of what looked like aggregation is in the
    data before any instrument touches it. Running OLS costs nothing and settles
    which.

SCALE, SO THE NUMBERS SIT BESIDE 1.8.4's

    The AR variance carries the SAME finite-sample factor pyfixest applies to the
    clustered first stage, recovered by dividing its variance for the first-stage
    coefficient by the uncorrected sandwich rather than by reimplementing CRV1.
    So the F reported here reproduces Table I and the AR set is on that scale.

Output:
    dc.LOG_DIR/ar_ols_ladder.csv
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

CODE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, CODE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bi = _load("build_instrument", "14_build_instrument.py")
cf = _load("ns14_compare", "15_ns14_f_comparison.py")

RUNGS = (("county", "ccg"), ("commuting zone", "cz"),
         ("state", "state"), ("Census division", "division"))
CHI2_95 = 3.8414588206941236          # chi-square(1) at 95%, the AR critical value


def cluster_sums(vec, codes):
    """Within-cluster sums of `vec`, one entry per cluster."""
    out = np.zeros(codes.max() + 1)
    np.add.at(out, codes, vec)
    return out


def ar_set(y, x, z, codes_fe, codes_cl, factor, crit=CHI2_95):
    """The Anderson-Rubin confidence set, in closed form.

    Returns the set as a type plus at most two finite endpoints. `factor` scales
    the cluster-robust variance to whatever convention the reported F uses.
    """
    yt = cf.demean(y, codes_fe).ravel()
    xt = cf.demean(x, codes_fe).ravel()
    zt = cf.demean(z, codes_fe).ravel()

    a = cluster_sums(zt * yt, codes_cl)
    b = cluster_sums(zt * xt, codes_cl)
    q = cluster_sums(zt * zt, codes_cl)
    S_zy, S_zx, S_zz = a.sum(), b.sum(), q.sum()

    # AR(b0) is the cluster-robust WALD test that z is absent from the reduced
    # form of y - b0*x, which is what standard software reports and what makes
    # this set reproducible outside this file. The Wald form uses the FITTED
    # residual, so the cluster sums have to be residualised on z the same way:
    #   meat_g(b0) = sum_g z*[(y - b0 x) - gammahat(b0) z] = at_g - b0 * bt_g
    # Using the raw a_g and b_g instead is the score form. It is asymptotically
    # the same test and gives a visibly different finite-sample set, and it
    # breaks the identity below, which is how the substitution was caught.
    at = a - S_zy * q / S_zz
    bt = b - S_zx * q / S_zz
    c = crit * factor

    A = S_zx ** 2 - c * (bt @ bt)
    B = 2.0 * (c * (at @ bt) - S_zy * S_zx)
    C = S_zy ** 2 - c * (at @ at)

    # bt_g = sum_g z*(x - bhat z) is exactly the first-stage meat, so this IS the
    # reported first-stage Wald and A > 0 is exactly W > crit. Asserted in main().
    W = S_zx ** 2 / (factor * (bt @ bt)) if (bt @ bt) > 0 else np.nan

    if abs(A) < 1e-300:
        if abs(B) < 1e-300:
            return ("whole line" if C <= 0 else "empty"), np.nan, np.nan, W
        root = -C / B
        return ("half line", root, np.nan, W) if B > 0 else ("half line", np.nan, root, W)

    disc = B * B - 4 * A * C
    if disc < 0:
        return ("empty" if A > 0 else "whole line"), np.nan, np.nan, W
    r1 = (-B - np.sqrt(disc)) / (2 * A)
    r2 = (-B + np.sqrt(disc)) / (2 * A)
    lo, hi = min(r1, r2), max(r1, r2)
    # A > 0: the quadratic is below zero BETWEEN the roots -> a bounded interval.
    # A < 0: below zero OUTSIDE them -> two infinite rays, which is what a weak
    # instrument looks like and is reported as such rather than as an interval.
    return ("bounded", lo, hi, W) if A > 0 else ("disjoint rays", lo, hi, W)


def fit_rung(d):
    """OLS, 2SLS, the clustered first stage, and the AR set for one rung."""
    import pyfixest as pf
    vcov = {"CRV1": "unit"}
    ols = pf.feols("y ~ x | unit + year", d, vcov=vcov)
    iv = pf.feols("y ~ 1 | unit + year | x ~ z", d, vcov=vcov)
    fs = pf.feols("x ~ z | unit + year", d, vcov=vcov)
    b_fs, se_fs = float(fs.coef()["z"]), float(fs.se()["z"])

    codes_fe, _ = cf.codes_for(d, ("unit", "year"))
    codes_cl = pd.factorize(d["unit"])[0].astype(np.int64)

    # The finite-sample factor pyfixest used, recovered rather than reimplemented.
    zt = cf.demean(d[["z"]].to_numpy(), codes_fe).ravel()
    xt = cf.demean(d["x"].to_numpy(), codes_fe).ravel()
    resid = xt - b_fs * zt
    meat = cluster_sums(zt * resid, codes_cl)
    v_plain = (meat @ meat) / (zt @ zt) ** 2
    factor = se_fs ** 2 / v_plain if v_plain > 0 else 1.0

    kind, lo, hi, W = ar_set(d["y"].to_numpy(), d["x"].to_numpy(),
                             d[["z"]].to_numpy(), codes_fe, codes_cl, factor)
    return dict(
        n_units=int(d["unit"].nunique()), n_obs=int(len(d)),
        ols_beta=float(ols.coef()["x"]), ols_se=float(ols.se()["x"]),
        iv_beta=float(iv.coef()["x"]), iv_se=float(iv.se()["x"]),
        fs_F=(b_fs / se_fs) ** 2, fs_F_check=W, ss_factor=factor,
        ar_kind=kind, ar_lo=lo, ar_hi=hi,
        iv_ci_lo=float(iv.coef()["x"]) - 1.96 * float(iv.se()["x"]),
        iv_ci_hi=float(iv.coef()["x"]) + 1.96 * float(iv.se()["x"]))


def fmt_set(r):
    """The AR set as it should be read, not as three columns to reassemble."""
    if r["ar_kind"] == "bounded":
        return f"[{r['ar_lo']:+.2f}, {r['ar_hi']:+.2f}]"
    if r["ar_kind"] == "disjoint rays":
        return f"(-inf,{r['ar_lo']:+.2f}] U [{r['ar_hi']:+.2f},+inf)"
    if r["ar_kind"] == "half line":
        return (f"[{r['ar_lo']:+.2f},+inf)" if np.isfinite(r["ar_lo"])
                else f"(-inf,{r['ar_hi']:+.2f}]")
    return {"whole line": "the whole line", "empty": "empty"}[r["ar_kind"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horizon", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--exposure", nargs="+", default=list(bi.EXPOSURES),
                    choices=list(bi.EXPOSURES))
    # Stated, never inherited - see the same argument in 15_ns14_f_comparison.py.
    ap.add_argument("--period", default="1975-2006", choices=list(bi.PERIODS),
                    help="estimation window; the default is the frozen design")
    args = ap.parse_args()

    horizons = sorted(set(args.horizon))
    y0, y1 = bi.PERIODS[args.period]
    names = [e for e in bi.EXPOSURES if e in set(args.exposure)
             and e in bi.ADMISSIBLE.get(args.period, set(bi.EXPOSURES))]
    print(f"\nAnderson-Rubin sets and the OLS ladder"
          f"   h={horizons}, exposures {names}, employment weights, {args.period}")
    print(f"    AR critical value chi2(1) at 95% = {CHI2_95:.4f}; a rung whose "
          f"clustered F is below it CANNOT have a bounded set")

    county = pd.read_parquet(dc.PANEL_OUT_DIR / "instrument_ccg_year.parquet")
    county = county[county["year"].between(y0, y1)]
    _, _, universe = dc.load_analysis_ccg_map()
    rungs = bi.load_rungs(universe)

    rows = []
    for h in horizons:
        for name in names:
            base = cf.prepare(county, rungs, h, name)
            for rung, unit_col in RUNGS:
                if rung == "county":
                    d = base.rename(columns={"emp_growth": "y", "shock": "x"}).copy()
                    d["unit"] = d["ccg"]
                    d = d[["unit", "year", "y", "x", "z"]]
                else:
                    d = bi.collapse(base, unit_col, "emp_lag")[
                        ["unit", "year", "y", "x", "z"]]
                d = d[np.isfinite(d[["x", "y", "z"]].to_numpy()).all(axis=1)]
                rows.append(dict(rung=rung, h=h, exposure=name, **fit_rung(d)))

    s = pd.DataFrame(rows)
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(dc.LOG_DIR / "ar_ols_ladder.csv", index=False)
    print(f"\n[1] {len(s):,} rungs -> {dc.LOG_DIR / 'ar_ols_ladder.csv'}")

    gap = float((s["fs_F"] - s["fs_F_check"]).abs().max())
    print(f"    the AR scale reproduces the reported first-stage F: "
          f"max gap {gap:.2e}   {'PASS' if gap < 1e-6 else 'FAIL'}")
    agree = bool(((s["fs_F"] > CHI2_95) == (s["ar_kind"] == "bounded")).all())
    print(f"    bounded set <=> clustered F > {CHI2_95:.2f}: "
          f"{'PASS' if agree else 'FAIL'}")

    for h in horizons:
        for name in names:
            print(f"\n[2] h={h}, exposure {name}")
            print(f"    {'rung':<17s}{'F':>6s}{'OLS':>17s}{'2SLS':>17s}"
                  f"{'   AR 95% set'}")
            for rung, _ in RUNGS:
                r = s[(s.h == h) & (s.exposure == name) & (s.rung == rung)].iloc[0]
                print(f"    {rung:<17s}{r.fs_F:>6.1f}"
                      f"{f'{r.ols_beta:+.3f} ({r.ols_se:.3f})':>17s}"
                      f"{f'{r.iv_beta:+.3f} ({r.iv_se:.3f})':>17s}"
                      f"   {fmt_set(r)}")
    return 0 if (gap < 1e-6 and agree) else 1


if __name__ == "__main__":
    raise SystemExit(main())
