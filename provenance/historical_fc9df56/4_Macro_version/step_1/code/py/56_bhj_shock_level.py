#!/usr/bin/env python3
"""The Borusyak-Hull-Jaravel argument for the composition instrument, delivered.

55 counts shifters. Counting is not the argument. Borusyak, Hull and Jaravel
identify a shift-share design from quasi-random SHOCKS rather than exogenous
shares, and that route has four requirements. This file discharges each one and
says plainly which are met, which are conditional, and which cannot be proved.

    1  EQUIVALENCE. Their central result is that the shift-share IV equals an IV
       run at the shock level, on exposure-weighted aggregates of the outcome and
       the treatment, weighted by shock importance. If the two agree, the design
       IS a shock-level experiment and their asymptotics are the relevant ones
       rather than an analogy. If they disagree, something in the construction is
       not what the framework describes and nothing downstream can be trusted.
       This is asserted, not merely reported.

    2  MANY SHOCKS. Their condition is on the inverse Herfindahl of the shock
       IMPORTANCE weights. Two counts are reported. Treating each (category,
       year) as its own shock is the generous one. Treating a category as one
       shock is the conservative one, and it is the honest one here, because a
       category's national procurement series is serially correlated and its
       years are not independent draws.

       This is NOT the Rotemberg concentration in 55. That measures which
       components supply the estimate, a Goldsmith-Pinkham, Sorkin and Swift
       question. Both are worth knowing and they are not substitutes.

    3  COMPLETE SHARES. BHJ require the exposure shares to sum to a constant, or
       the share sum to be controlled for. Ours sum to total contract dollars per
       head, which is exactly 51's intensity exposure, so an uncontrolled
       composition instrument is part composition and part level-of-exposure. The
       control is run and reported beside the uncontrolled estimate. This is the
       requirement most easily missed, because the instrument looks compositional.

    4  QUASI-RANDOM ASSIGNMENT. The substantive assumption: national demand
       growth for a category must be as good as randomly assigned with respect to
       which counties happen to produce it. No test proves this. BHJ make it
       falsifiable through balance at the SHOCK level: regress the national
       category rate on exposure-weighted predetermined county characteristics
       and on the exposure-weighted pre-trend in the outcome. A rejection is
       evidence against the design. Passing is not proof, and is not reported as
       proof.

Everything runs at 54's headline cell: era full, h = 4, population denominator,
net rule, no floor, no screen, roll10 exposure, division x year effects,
division-equal weights, treatment winsorized within year.

Output:
    4_Macro_version/Results/ns14_diagnostics/Table_bhj_shock_level.tex
    batch_logs/macro_version/bhj_shock_level.csv
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import dcads_common as dc  # noqa: E402

PROJECT_ROOT = CODE.parents[3]
STEP2_CODE = PROJECT_ROOT / "4_Macro_version" / "step_2" / "code" / "py"
TAB_DIR = PROJECT_ROOT / "4_Macro_version" / "Results" / "ns14_diagnostics"
OUT_CSV = dc.LOG_DIR / "bhj_shock_level.csv"

H, SPAN, LAG = 4, 10, 5
EQUIV_TOL = 1e-6          # RELATIVE, which is the demeaner's tolerance not the identity's


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p54 = _load("p54", CODE / "54_composition_instrument_grid.py")
p51 = p54.p51


def wls(y, Xm, w, cluster=None):
    """Weighted least squares with a constant, clustered SEs, and a joint F."""
    n = len(y)
    A = np.column_stack([np.ones(n), Xm])
    XtWX = A.T @ (w[:, None] * A)
    b = np.linalg.solve(XtWX, A.T @ (w * y))
    u = y - A @ b
    inv = np.linalg.inv(XtWX)
    if cluster is None:
        meat = (A * (w * u)[:, None]).T @ (A * (w * u)[:, None])
    else:
        meat = np.zeros_like(XtWX)
        for g in np.unique(cluster):
            m = cluster == g
            s = (A[m] * (w[m] * u[m])[:, None]).sum(axis=0)
            meat += np.outer(s, s)
        G = len(np.unique(cluster))
        meat *= G / max(G - 1, 1)
    V = inv @ meat @ inv
    se = np.sqrt(np.diag(V))
    k = A.shape[1] - 1
    R = np.zeros((k, A.shape[1]))
    R[:, 1:] = np.eye(k)
    Rb = R @ b
    F = float(Rb @ np.linalg.solve(R @ V @ R.T, Rb) / k)
    return b, se, F


def main():
    t0 = time.time()
    print("=" * 78)
    print("  Borusyak-Hull-Jaravel, discharged for the composition instrument")
    print("=" * 78)

    t28 = _load("t28", STEP2_CODE / "28_table_d1.py")
    w22 = _load("w22", STEP2_CODE / "22_winsor_construction_error.py")
    import pyfixest as pf
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100_000)

    B = p51.base_panel()
    units = np.array(sorted(B["ccg"].unique()))
    C = p51.contract_matrices(units, 1966, 2018)[("net", 0.0)]
    ARR, NAT, cats, cy = p54.category_panel(units, 1966, 2018)
    arr = ARR[("net", 0.0)]
    nat = NAT[("net", 0.0)]
    print(f"  {len(units):,} counties, {len(cats)} categories, "
          f"{time.time()-t0:.1f}s", flush=True)

    years = [t for t in range(max(1977, 1975 + H), 2019)]
    piv = {c: B.pivot(index="ccg", columns="year", values=c).reindex(units)
           for c in ("P", "E", "Y")}
    DIVI = B.drop_duplicates("ccg").set_index("ccg")["division"]

    lag = {k: C.reindex(columns=[y - k for y in years]).set_axis(years, axis=1)
           for k in range(H + 1)}
    G_b = lag[H].to_numpy(np.float64)
    G_w = (sum(lag[k] for k in range(H)) / H).to_numpy(np.float64)
    P_b = piv["P"].reindex(columns=[y - H for y in years]).to_numpy(np.float64)
    E_b = piv["E"].reindex(columns=[y - H for y in years]).to_numpy(np.float64)
    E_t = piv["E"].reindex(columns=years).to_numpy(np.float64)
    E_2b = piv["E"].reindex(columns=[y - 2 * H for y in years]).to_numpy(np.float64)
    Y_b = piv["Y"].reindex(columns=[y - H for y in years]).to_numpy(np.float64)
    X = (G_w - G_b) / P_b
    den = E_t + E_b
    Yv = np.where(den != 0, 2.0 * (E_t - E_b) / den, np.nan)
    # the pre-trend: the SAME growth rate over the window that ends where the
    # estimation window begins, so a shock that lines up with it is a shock that
    # was already predictable from what the county was doing beforehand.
    dpre = E_b + E_2b
    PRE = np.where(dpre != 0, 2.0 * (E_b - E_2b) / dpre, np.nan)

    gk = p54.category_rates(nat, H)
    L, nc = p54.predicted_dollars(arr, gk, cy, years, H, "roll", SPAN, LAG,
                                  p51.DATA_FIRST)
    Z = L / P_b

    cbar = {}
    for j, t in enumerate(years):
        hi = t - H - LAG
        lo = hi - SPAN + 1
        if lo < p51.DATA_FIRST:
            continue
        a, b = cy.index(lo), cy.index(hi)
        cbar[j] = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
    S0 = np.full_like(Z, np.nan)
    for j, m in cbar.items():
        S0[:, j] = m.sum(axis=1) / P_b[:, j]

    ok = (np.isfinite(X) & np.isfinite(Yv) & np.isfinite(Z) & np.isfinite(S0)
          & np.isfinite(E_b))
    ri, ci = np.where(ok)
    d = pd.DataFrame(dict(ccg=units[ri], year=np.array(years)[ci],
                          x=X[ok], y=Yv[ok], z=Z[ok], s0=S0[ok], E_b=E_b[ok],
                          P_b=P_b[ok], Y_b=Y_b[ok], pre=PRE[ok]))
    d["division"] = d["ccg"].map(DIVI)
    xw = p51.wins(d["x"].to_numpy(np.float64), d["year"].to_numpy())
    tot = d.groupby(["division", "year"])["E_b"].transform("sum")
    w = (d["E_b"] / tot).to_numpy(np.float64)
    w = w / w.mean()
    fe = pd.factorize(d["division"].astype(str) + "_"
                      + d["year"].astype(str))[0].astype(np.int64)
    nofe = np.zeros(len(d), np.int64)
    cl = pd.factorize(d["ccg"])[0].astype(np.int64)
    y0 = d["y"].to_numpy(np.float64)
    z0 = d["z"].to_numpy(np.float64)
    s0 = d["s0"].to_numpy(np.float64)
    print(f"  estimation sample {len(d):,} county-years", flush=True)

    # ---- 3. complete shares ---------------------------------------------
    base = t28.fit2sls(y0, xw, z0, w, fe, nofe, cl, singleton=False)
    M = w22.demean(np.column_stack([y0, xw, z0, s0]), fe, nofe, w)
    yt, xt, zt, st = M[:, 0], M[:, 1], M[:, 2], M[:, 3]

    def partial(v):
        c_ = float(st @ (w * v)) / float(st @ (w * st))
        return v - c_ * st

    yr, xr, zr = partial(yt), partial(xt), partial(zt)
    ctrl = t28.fit2sls(yr, xr, zr, w, nofe, nofe, cl, singleton=False)
    z_expl = 1.0 - float(np.var(zr)) / float(np.var(zt))

    # ---- 1. shock-level aggregates and the equivalence -------------------
    rowpos = {u: i for i, u in enumerate(units)}
    colpos = {y: j for j, y in enumerate(years)}
    rr = d["ccg"].map(rowpos).to_numpy()
    cc = d["year"].map(colpos).to_numpy()
    pb = d["P_b"].to_numpy(np.float64)
    # predetermined county characteristics, measured at the base year
    # The characteristics are residualized on division x year with the estimation
    # weights BEFORE being aggregated, because that is the variation the estimator
    # uses. Aggregating the raw variable tests a design nobody ran: income per
    # head carries a level and a trend the fixed effects absorb, and on the raw
    # variable this balance test rejects at t = 4.08 with a joint F of 5.32 where
    # the residualized one gives t = -0.39 and F = 0.59. The rejection was in the
    # part of income the specification never sees.
    chars_raw = {
        "income per head": d["Y_b"].to_numpy(np.float64) / pb,
        "employment rate": d["E_b"].to_numpy(np.float64) / pb,
        "log population": np.log(pb),
        "pre-period growth": d["pre"].to_numpy(np.float64),
    }
    _names = list(chars_raw)
    _res = w22.demean(np.column_stack([np.nan_to_num(chars_raw[c], nan=0.0)
                                       for c in _names]), fe, nofe, w)
    chars = {c: _res[:, q] for q, c in enumerate(_names)}
    # The equivalence is an identity in the UNNORMALIZED numerators. Under the
    # net rule a contract action can be a deobligation, so a category's aggregate
    # exposure weight can be zero or negative; dividing by it to form an average
    # and multiplying it back drops exactly those categories and breaks the
    # identity. Accumulating sum_c w s_ck ~y directly needs no positivity.
    keys, sv, gv = [], [], []
    num_y, num_x = [], []
    acc = {k: [] for k in chars}
    for j in sorted(cbar):
        m = cc == j
        if not m.any():
            continue
        idx_r = rr[m]
        ww = w[m]
        pbm = pb[m]
        tcol = cy.index(years[j])
        Sk = cbar[j][idx_r, :] / pbm[:, None]          # (n_c, K) shares
        wsk = Sk * ww[:, None]
        denom = wsk.sum(axis=0)
        ny = wsk.T @ yt[m]
        nx = wsk.T @ xt[m]
        for k in range(len(cats)):
            g_ = gk[k, tcol]
            if not np.isfinite(g_):
                continue
            keys.append((k, j))
            sv.append(denom[k])
            gv.append(g_)
            num_y.append(ny[k])
            num_x.append(nx[k])
            for cname, cvec in chars.items():
                v = cvec[m]
                good = np.isfinite(v)
                den_c = wsk[good, k].sum()
                acc[cname].append(float(wsk[good, k] @ v[good]) / den_c
                                  if abs(den_c) > 0 else np.nan)
    sv = np.array(sv)
    gv = np.array(gv)
    num_y = np.array(num_y)
    num_x = np.array(num_x)
    beta_shock = float((gv * num_y).sum() / (gv * num_x).sum())
    gap = abs(beta_shock - base["beta"]) * 1e5
    rel = gap / abs(base["beta"] * 1e5)
    print(f"\n  1. EQUIVALENCE")
    print(f"     county-level 2SLS            {base['beta']*1e5:>12.6f}")
    print(f"     shock-level IV               {beta_shock*1e5:>12.6f}")
    print(f"     relative gap                 {rel:>12.2e}   "
          f"{'PASS' if rel < EQUIV_TOL else 'FAIL'}")
    n_neg = int((sv <= 0).sum())
    print(f"     shocks with non-positive aggregate exposure  {n_neg:>6,} of "
          f"{len(sv):,}")
    if rel >= EQUIV_TOL:
        raise SystemExit("  FAIL: the shift-share IV is not the shock-level IV, so "
                         "this design is not what the BHJ framework describes and "
                         "no diagnostic below means what it claims to mean.")
    with np.errstate(invalid="ignore", divide="ignore"):
        ybar = np.where(sv != 0, num_y / sv, np.nan)
        xbar = np.where(sv != 0, num_x / sv, np.nan)

    # ---- 2. many shocks --------------------------------------------------
    # A Herfindahl needs non-negative importance, so the net rule's negative
    # aggregate exposures enter by magnitude and their count is reported above.
    sabs = np.abs(sv)
    p_kt = sabs / sabs.sum()
    eff_kt = float(1.0 / (p_kt ** 2).sum())
    byk = pd.Series(sabs).groupby([k for k, _ in keys]).sum().to_numpy()
    p_k = byk / byk.sum()
    eff_k = float(1.0 / (p_k ** 2).sum())
    print(f"\n  2. MANY SHOCKS")
    print(f"     shock observations (k,t) used                 {len(keys):>8,}")
    print(f"     effective, counting each (k,t)                {eff_kt:>8.1f}")
    print(f"     effective, counting each category once        {eff_k:>8.1f}")

    # ---- 4. quasi-random assignment: balance at the shock level ---------
    kcl = np.array([k for k, _ in keys])
    print(f"\n  4. BALANCE AT THE SHOCK LEVEL")
    print(f"     national category rate regressed on exposure-weighted county")
    print(f"     characteristics, weighted by shock importance, clustered by category")
    print(f"\n     {'characteristic':<24}{'coef':>12}{'se':>11}{'t':>7}")
    bal_rows = []
    Xcols, names = [], []
    for cname in chars:
        v = np.array(acc[cname])
        good = np.isfinite(v) & np.isfinite(gv)
        vs = (v - np.average(v[good], weights=sv[good]))
        sd = np.sqrt(np.average(vs[good] ** 2, weights=sv[good]))
        vz = vs / (sd if sd > 0 else 1.0)                 # standardized
        Xcols.append(np.where(good, vz, 0.0))
        names.append(cname)
        b_, se_, _ = wls(gv[good], vz[good].reshape(-1, 1), sv[good], kcl[good])
        print(f"     {cname:<24}{b_[1]:>12.5f}{se_[1]:>11.5f}"
              f"{b_[1]/se_[1]:>7.2f}")
        bal_rows.append((cname, b_[1], se_[1], b_[1] / se_[1]))
    Xm = np.column_stack(Xcols)
    good = np.isfinite(gv) & np.isfinite(Xm).all(axis=1)
    _, _, Fjoint = wls(gv[good], Xm[good], sv[good], kcl[good])
    print(f"     {'joint F, all four':<24}{Fjoint:>12.3f}")

    print(f"\n  3. COMPLETE SHARES")
    print(f"     {'as built, no control':<38}{base['beta']*1e5:>9.3f}"
          f"{base['beta']/base['se']:>7.2f}{base['F']:>8.1f}")
    print(f"     {'controlling for the share sum':<38}{ctrl['beta']*1e5:>9.3f}"
          f"{ctrl['beta']/ctrl['se']:>7.2f}{ctrl['F']:>8.1f}")
    print(f"     share sum explains {z_expl:.3f} of the instrument's within-FE variance")

    rows = [
        ("Shock-level observations $(k,t)$", f"{len(keys):,}"),
        ("Effective shocks, each $(k,t)$", f"{eff_kt:.1f}"),
        ("Effective shocks, each category once", f"{eff_k:.1f}"),
        ("County-level 2SLS coefficient", f"{base['beta']*1e5:.4f}"),
        ("Shock-level IV coefficient", f"{beta_shock*1e5:.4f}"),
        ("Equivalence gap", f"{gap:.1e}"),
        ("Coefficient, no share-sum control", f"{base['beta']*1e5:.3f}"),
        ("\\hspace{1.5em}$t$", f"{base['beta']/base['se']:.2f}"),
        ("Coefficient, share sum controlled", f"{ctrl['beta']*1e5:.3f}"),
        ("\\hspace{1.5em}$t$", f"{ctrl['beta']/ctrl['se']:.2f}"),
        ("Share sum's share of instrument variance", f"{z_expl:.3f}"),
    ] + [(f"Balance: {c}", f"{b:.4f} ({s:.4f})") for c, b, s, _ in bal_rows] + [
        ("Balance: joint $F$", f"{Fjoint:.2f}"),
        ("Observations, county level", f"{len(d):,}"),
    ]
    pd.DataFrame([dict(statistic=a, value=b) for a, b in rows]).to_csv(
        OUT_CSV, index=False)

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    tex = TAB_DIR / "Table_bhj_shock_level.tex"
    body = "\n".join(f"{a} & {b} \\\\\\\\" for a, b in rows)
    tex.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{Shock-Level Diagnostics for the Composition Instrument}}
\\label{{tab:bhj_shock_level}}
{{\\centering
\\adjustbox{{max width=\\linewidth}}{{%
\\begin{{tabular}}{{lr}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} The composition instrument at a four-year horizon on
97,856 county-years, with division-by-year fixed effects, division-equal weights and the
treatment winsorized within year. Following Borusyak, Hull and Jaravel, the shift-share IV is
rewritten as an IV at the level of the {len(cats)} Federal Supply Classification categories,
using exposure-weighted aggregates of the outcome and the treatment and weighting by shock
importance; the two coefficients agree to the reported gap, so the design is a shock-level
experiment in their sense. Effective shock counts are inverse Herfindahl indices of the
importance weights, reported both treating each category-year as a shock and treating each
category as one shock, the latter being conservative because a category's national series is
serially correlated. Their framework requires exposure shares summing to a constant; here they
sum to total contract dollars per head, so the estimate is reported with and without that sum
partialled out. Balance regressions take the national category rate on exposure-weighted
predetermined county characteristics, standardized, weighted by shock importance and clustered
by category; the pre-period growth rate is the same midpoint employment growth measured over
the four years ending where the estimation window begins. Balance failing to reject is evidence
consistent with quasi-random assignment and is not proof of it.\\par}}
\\end{{table}}
""")
    print(f"\n  wrote {tex}")
    print(f"  wrote {OUT_CSV}")
    print(f"  {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
