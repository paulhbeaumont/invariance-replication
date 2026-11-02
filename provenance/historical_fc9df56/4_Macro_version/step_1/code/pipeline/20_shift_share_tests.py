"""Stage 6: the shift-share tests, one runner, both tables.

Ports of producers 55 and 56, row for row, sharing one input build:

  Table_many_shifters.tex     the GPSS side: shifter counts, effective
                              numbers (1/HHI), PC1 variance share,
                              common-cycle R2, first-stage concentration,
                              intensity (K=1) vs composition (K=NCAT) on a
                              shared sample.
  Table_bhj_shock_level.tex   the BHJ side: shock-level equivalence
                              (hard-failed at 1e-6 relative), effective
                              shock counts, the incomplete-shares control,
                              shock-level balance and the exposure-weighted
                              pre-trend.

Two phases:
  --phase old      characterization: the OLD constants (h=4, span=10, net,
                   unharmonized), instruments rebuilt in memory; both outputs
                   asserted string-equal to the captured reruns of producers
                   55 and 56 (_golden). Proves the ports.
  --phase locked   the superseding tables at the locked constants, consuming
                   ONLY the persisted pipeline stores, written to
                   Results/pipeline_diagnostics/. Keeps the 55/56
                   winsorized-x convention (locked.WINSOR_DIAG_GROUP).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import contracts  # noqa: E402
import instrument as inst  # noqa: E402
import panel as pnl_lib  # noqa: E402
import store  # noqa: E402
import oldpath as op  # noqa: E402  (estimator only: t28/w22, as 57 loads them)
import dcads_common as dc  # noqa: E402

TAB_DIR = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"
GOLD = store.PIPE / "_golden"
EQUIV_TOL = 1e-6

PHASES = {
    "old": dict(h=4, span=10, lag=5, rule="net", harmonize=False),
    "locked": dict(h=4, span=locked.SPAN, lag=locked.LAG, rule=locked.RULE,
                   harmonize=True),
}


# ---------------------------------------------------------------- helpers --

def two_way_demean(M):
    M = M - np.nanmean(M, axis=1, keepdims=True)
    M = M - np.nanmean(M, axis=0, keepdims=True)
    return M


def pc_profile(M):
    ok = np.isfinite(M).all(axis=1)
    A = two_way_demean(M[ok])
    s = np.linalg.svd(A, compute_uv=False)
    v = s ** 2
    v = v / v.sum()
    return float(v[0]), int(np.searchsorted(np.cumsum(v), 0.90) + 1)


def hhi_effective(w):
    w = np.abs(np.asarray(w, np.float64))
    tot = w.sum()
    if tot <= 0:
        return np.nan
    p = w / tot
    return float(1.0 / (p ** 2).sum())


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


# ------------------------------------------------------------ input build --

def build_inputs(ph, use_store):
    """Everything both tables need, either rebuilt in memory at the old
    constants or read from the persisted stores at the locked constants."""
    H, SPAN, LAG = ph["h"], ph["span"], ph["lag"]
    pnl = store.load_county_panel()
    units, piv = store.load_pivots(pnl, cols=("P", "E", "Y"))
    DIVI = pnl.drop_duplicates("ccg").set_index("ccg")["division"]

    if use_store:
        units2, cats, cy, arr, nat = store.load_tensor(ph["rule"])
        assert np.array_equal(units, units2)
        C = store.load_con(ph["rule"]).reindex(units)
    else:
        CON = contracts.contract_matrices(units, locked.FIRST_YEAR,
                                          locked.LAST_YEAR)
        C = CON[(ph["rule"], 0.0)]
        ARR, NAT, cats, cy = contracts.category_panel(
            units, locked.FIRST_YEAR, locked.LAST_YEAR,
            harmonize=ph["harmonize"])
        arr = ARR[(ph["rule"], 0.0)]
        nat = NAT[(ph["rule"], 0.0)]

    years = [t for t in range(max(1977, 1975 + H), locked.TARGET_LAST + 1)]
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
    dpre = E_b + E_2b
    PRE = np.where(dpre != 0, 2.0 * (E_b - E_2b) / dpre, np.nan)

    if use_store:
        gk = store.load_gk(H, cats, cy)
        Z_COM = store.load_instrument(H, units, "z")
        Z_INT = store.load_instrument(H, units, "z_int")
    else:
        natshift = np.nansum(G_w - G_b, axis=0) / np.nansum(P_b, axis=0)
        expo = np.full_like(X, np.nan)
        for j, t in enumerate(years):
            hi = t - H - LAG
            lo = hi - SPAN + 1
            if lo < locked.DATA_FIRST:
                continue
            yy = list(range(lo, hi + 1))
            num = C.loc[:, yy].sum(axis=1).to_numpy(np.float64)
            dpop = (piv["P"].loc[:, yy].sum(axis=1, min_count=len(yy))
                    .to_numpy(np.float64))
            expo[:, j] = num / dpop
        Z_INT = expo * natshift[None, :]
        gk = inst.category_rates(nat, H)
        L, _ = inst.predicted_dollars(arr, gk, cy, years, H, "roll", SPAN,
                                      LAG, locked.DATA_FIRST)
        Z_COM = L / P_b

    import pyfixest as pf
    op.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100_000)
    return dict(units=units, cats=cats, cy=cy, arr=arr, gk=gk, DIVI=DIVI,
                years=years, X=X, Yv=Yv, E_b=E_b, E_t=E_t, P_b=P_b,
                Y_b=Y_b, PRE=PRE, Z_INT=Z_INT, Z_COM=Z_COM)


# ------------------------------------------------- the GPSS-side table ----

def shifters_table(inp, ph):
    H, SPAN, LAG = ph["h"], ph["span"], ph["lag"]
    units, cats, cy = inp["units"], inp["cats"], inp["cy"]
    arr, gk, years = inp["arr"], inp["gk"], inp["years"]
    X, Y, E_b, P_b = inp["X"], inp["Yv"], inp["E_b"], inp["P_b"]
    Z_INT, Z_COM, DIVI = inp["Z_INT"], inp["Z_COM"], inp["DIVI"]

    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z_INT) & np.isfinite(Z_COM)
    cols = np.where(ok.any(axis=0))[0]
    rows_b = np.where(ok[:, cols].all(axis=1))[0]
    cols = cols[ok[np.ix_(rows_b, cols)].all(axis=0)]
    n_obs = int(ok.sum())
    if len(rows_b) < 100 or len(cols) < 5:
        raise SystemExit("  FAIL: the balanced block is too small to rank")

    pc_int, k90_int = pc_profile(Z_INT[np.ix_(rows_b, cols)])
    pc_com, k90_com = pc_profile(Z_COM[np.ix_(rows_b, cols)])

    G = pd.DataFrame(gk, index=cats, columns=cy).loc[:, years[0]:years[-1]]
    common = G.mean(axis=0)
    r2 = []
    for k in G.index:
        v = G.loc[k].dropna()
        if len(v) > 5 and v.std() > 0:
            r2.append(np.corrcoef(v, common.reindex(v.index))[0, 1] ** 2)
    r2 = np.array(r2)

    dollars_k = arr.sum(axis=(0, 2)).astype(np.float64)
    eff_com = hhi_effective(dollars_k)

    idx = pd.MultiIndex.from_product([units, years], names=["ccg", "year"])
    f = pd.DataFrame(index=idx).reset_index()
    f["division"] = f["ccg"].map(DIVI)
    f["x"] = X.reshape(-1)
    f["y"] = Y.reshape(-1)
    f["E_b"] = E_b.reshape(-1)
    f["P_b"] = P_b.reshape(-1)
    f["z_com"] = Z_COM.reshape(-1)
    f["z_int"] = Z_INT.reshape(-1)
    m = np.isfinite(f[["x", "y", "z_com", "z_int", "E_b"]].to_numpy()).all(axis=1)
    g = f[m]
    xw = pnl_lib.wins(g["x"].to_numpy(np.float64), g["year"].to_numpy())
    tot = g.groupby(["division", "year"])["E_b"].transform("sum")
    w = (g["E_b"] / tot).to_numpy(np.float64)
    w = w / w.mean()
    fe = pd.factorize(g["division"].astype(str) + "_"
                      + g["year"].astype(str))[0].astype(np.int64)
    nofe = np.zeros(len(g), np.int64)
    xt = op.w22.demean(xw.reshape(-1, 1), fe, nofe, w)[:, 0]

    rowpos = {u: i for i, u in enumerate(units)}
    colpos = {y: j for j, y in enumerate(years)}
    ri = g["ccg"].map(rowpos).to_numpy()
    ci = g["year"].map(colpos).to_numpy()
    pb = g["P_b"].to_numpy(np.float64)
    wx = w * xt
    alpha = np.zeros(len(cats), np.float64)
    for kk in range(len(cats)):
        cbar = np.full((len(units), len(years)), np.nan)
        for j, t in enumerate(years):
            hi = t - H - LAG
            lo = hi - SPAN + 1
            if lo < locked.DATA_FIRST:
                continue
            a, b = cy.index(lo), cy.index(hi)
            cbar[:, j] = arr[:, kk, a:b + 1].mean(axis=1)
        zk = cbar[ri, ci] / pb
        alpha[kk] = np.nansum(wx * np.nan_to_num(zk))
    a_abs = np.abs(alpha)
    order = np.argsort(-a_abs)
    top1 = a_abs[order[0]] / a_abs.sum()
    top5 = a_abs[order[:5]].sum() / a_abs.sum()
    eff_rot = hhi_effective(a_abs)

    cl = pd.factorize(g["ccg"])[0].astype(np.int64)
    est = {}
    for lab, col in (("intensity", "z_int"), ("composition", "z_com")):
        r = op.t28.fit2sls(g["y"].to_numpy(np.float64), xw,
                           g[col].to_numpy(np.float64), w, fe, nofe, cl,
                           singleton=False)
        est[lab] = (r["beta"] * 1e5, r["beta"] / r["se"], r["F"])

    T = int(len(cols))
    rows = [
        ("Number of shifters $K$", "1", f"{len(cats)}"),
        ("Shock observations ($K \\times T$)", f"{T}", f"{len(cats)*T:,}"),
        ("Effective number of shifters (1/HHI)", "1.0", f"{eff_com:.1f}"),
        ("Variance in first component", f"{pc_int:.3f}", f"{pc_com:.3f}"),
        ("Components for 90\\% of variance", f"{k90_int}", f"{k90_com}"),
        ("Median $R^2$ of shifter on common cycle", "1.000",
         f"{np.median(r2):.3f}"),
        ("Shifters with $R^2>0.5$ on common cycle", "1 of 1",
         f"{int((r2>0.5).sum())} of {len(r2)}"),
        ("Largest component's first-stage share", "1.000", f"{top1:.3f}"),
        ("Five largest components' share", "1.000", f"{top5:.3f}"),
        ("Effective components, first stage (1/HHI)", "1.0", f"{eff_rot:.1f}"),
    ]
    return rows, est, n_obs, len(cats)


# -------------------------------------------------- the BHJ-side table ----

def bhj_table(inp, ph):
    H, SPAN, LAG = ph["h"], ph["span"], ph["lag"]
    units, cats, cy = inp["units"], inp["cats"], inp["cy"]
    arr, gk, years, DIVI = inp["arr"], inp["gk"], inp["years"], inp["DIVI"]
    X, Yv, E_b, P_b = inp["X"], inp["Yv"], inp["E_b"], inp["P_b"]
    Y_b, PRE, Z = inp["Y_b"], inp["PRE"], inp["Z_COM"]

    cbar = {}
    for j, t in enumerate(years):
        hi = t - H - LAG
        lo = hi - SPAN + 1
        if lo < locked.DATA_FIRST:
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
    xw = pnl_lib.wins(d["x"].to_numpy(np.float64), d["year"].to_numpy())
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

    # ---- 3. complete shares ---------------------------------------------
    base = op.t28.fit2sls(y0, xw, z0, w, fe, nofe, cl, singleton=False)
    M = op.w22.demean(np.column_stack([y0, xw, z0, s0]), fe, nofe, w)
    yt, xt, zt, st = M[:, 0], M[:, 1], M[:, 2], M[:, 3]

    def partial(v):
        c_ = float(st @ (w * v)) / float(st @ (w * st))
        return v - c_ * st

    yr, xr, zr = partial(yt), partial(xt), partial(zt)
    ctrl = op.t28.fit2sls(yr, xr, zr, w, nofe, nofe, cl, singleton=False)
    z_expl = 1.0 - float(np.var(zr)) / float(np.var(zt))

    # ---- 1. shock-level aggregates and the equivalence -------------------
    rowpos = {u: i for i, u in enumerate(units)}
    rr = d["ccg"].map(rowpos).to_numpy()
    colpos = {y: j for j, y in enumerate(years)}
    cc = d["year"].map(colpos).to_numpy()
    pb = d["P_b"].to_numpy(np.float64)
    chars_raw = {
        "income per head": d["Y_b"].to_numpy(np.float64) / pb,
        "employment rate": d["E_b"].to_numpy(np.float64) / pb,
        "log population": np.log(pb),
        "pre-period growth": d["pre"].to_numpy(np.float64),
    }
    _names = list(chars_raw)
    _res = op.w22.demean(np.column_stack([np.nan_to_num(chars_raw[c], nan=0.0)
                                          for c in _names]), fe, nofe, w)
    chars = {c: _res[:, q] for q, c in enumerate(_names)}
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
        Sk = cbar[j][idx_r, :] / pbm[:, None]
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
    print(f"  equivalence: county {base['beta']*1e5:.6f} vs shock-level "
          f"{beta_shock*1e5:.6f}, relative gap {rel:.2e} "
          f"{'PASS' if rel < EQUIV_TOL else 'FAIL'}")
    if rel >= EQUIV_TOL:
        raise SystemExit("  FAIL: the shift-share IV is not the shock-level IV")

    # ---- 2. many shocks --------------------------------------------------
    sabs = np.abs(sv)
    p_kt = sabs / sabs.sum()
    eff_kt = float(1.0 / (p_kt ** 2).sum())
    byk = pd.Series(sabs).groupby([k for k, _ in keys]).sum().to_numpy()
    p_k = byk / byk.sum()
    eff_k = float(1.0 / (p_k ** 2).sum())

    # ---- 4. quasi-random assignment: balance at the shock level ---------
    kcl = np.array([k for k, _ in keys])
    bal_rows = []
    Xcols = []
    for cname in chars:
        v = np.array(acc[cname])
        good = np.isfinite(v) & np.isfinite(gv)
        vs = (v - np.average(v[good], weights=sv[good]))
        sd = np.sqrt(np.average(vs[good] ** 2, weights=sv[good]))
        vz = vs / (sd if sd > 0 else 1.0)
        Xcols.append(np.where(good, vz, 0.0))
        b_, se_, _ = wls(gv[good], vz[good].reshape(-1, 1), sv[good], kcl[good])
        bal_rows.append((cname, b_[1], se_[1], b_[1] / se_[1]))
    Xm = np.column_stack(Xcols)
    good = np.isfinite(gv) & np.isfinite(Xm).all(axis=1)
    _, _, Fjoint = wls(gv[good], Xm[good], sv[good], kcl[good])

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
    return rows, len(d)


# ------------------------------------------------------------- writers ----

def write_shifters(rows, est, n_obs, ncat):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    tex = TAB_DIR / "Table_many_shifters.tex"
    body = "\n".join(f"{l} & {a} & {b} \\\\" for l, a, b in rows)
    tex.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{How Many Shifters Each County Instrument Has}}
\\label{{tab:many_shifters}}
{{\\centering
\\adjustbox{{max width=\\linewidth}}{{%
\\begin{{tabular}}{{lrr}}
\\toprule
 & Intensity & Composition \\\\
 & (one national series) & ({ncat} categories) \\\\
\\midrule
{body}
\\midrule
Second-stage coefficient & {est['intensity'][0]:.2f} & {est['composition'][0]:.2f} \\\\
\\hspace{{1.5em}}$t$ & {est['intensity'][1]:.2f} & {est['composition'][1]:.2f} \\\\
\\hspace{{1.5em}}First-stage $F$ & {est['intensity'][2]:.1f} & {est['composition'][2]:.1f} \\\\
\\addlinespace Observations & {n_obs:,} & {n_obs:,} \\\\
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} Both instruments are estimated on the same county-year
sample at a four-year horizon, with division-by-year fixed effects, division-equal weights and
the treatment winsorized within year, at the locked baseline constants: the harmonized top-{ncat}
category axis, the positive contract rule, and a five-year exposure window lagged five years.
The intensity instrument multiplies a county's contract exposure over the predetermined window
by one national procurement series, so $K=1$; it is the county analogue of the
Nakamura-Steinsson alternative instrument. The composition instrument meets the county's
contract mix across {ncat} four-digit Federal Supply Classification categories with the national
change in demand for each. The effective number of shifters is the inverse Herfindahl of
national dollars by category. Variance in the first component is the share of the two-way
demeaned instrument matrix's variance in its leading singular direction; a single-factor
instrument is rank one in the cross section and sits at one. The common cycle is the
equal-weighted mean of the category series. First-stage shares follow Goldsmith-Pinkham, Sorkin
and Swift. This table supersedes Results/ns14\\_diagnostics/Table\\_many\\_shifters.tex, which was
built on the unharmonized axis with a ten-year window and the net rule.\\par}}
\\end{{table}}
""")
    csv = TAB_DIR / "many_shifters_locked.csv"
    pd.DataFrame([dict(statistic=l, intensity=a, composition=b)
                  for l, a, b in rows]).to_csv(csv, index=False)
    print(f"  wrote {tex}")
    print(f"  wrote {csv}")


def write_bhj(rows, ncat, n_obs):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    tex = TAB_DIR / "Table_bhj_shock_level.tex"
    body = "\n".join(f"{a} & {b} \\\\" for a, b in rows)
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
{n_obs:,} county-years, at the locked baseline constants (harmonized top-{ncat} category axis,
positive contract rule, five-year exposure window lagged five years), with division-by-year
fixed effects, division-equal weights and the treatment winsorized within year. Following
Borusyak, Hull and Jaravel, the shift-share IV is rewritten as an IV at the level of the {ncat}
Federal Supply Classification categories, using exposure-weighted aggregates of the outcome and
the treatment and weighting by shock importance; the two coefficients agree to the reported
gap, so the design is a shock-level experiment in their sense. Effective shock counts are
inverse Herfindahl indices of the importance weights, reported both treating each category-year
as a shock and treating each category as one shock, the latter being conservative because a
category's national series is serially correlated. Their framework requires exposure shares
summing to a constant; here they sum to total contract dollars per head, so the estimate is
reported with and without that sum partialled out. Balance regressions take the national
category rate on exposure-weighted predetermined county characteristics, standardized, weighted
by shock importance and clustered by category; the pre-period growth rate is the same midpoint
employment growth measured over the four years ending where the estimation window begins.
Balance failing to reject is evidence consistent with quasi-random assignment and is not proof
of it. This table supersedes Results/ns14\\_diagnostics/Table\\_bhj\\_shock\\_level.tex, which was
built on the unharmonized axis with a ten-year window and the net rule.\\par}}
\\end{{table}}
""")
    csv = TAB_DIR / "bhj_shock_level_locked.csv"
    pd.DataFrame([dict(statistic=a, value=b) for a, b in rows]).to_csv(
        csv, index=False)
    print(f"  wrote {tex}")
    print(f"  wrote {csv}")


# ---------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["old", "locked", "both"], default="both")
    args = ap.parse_args()
    phases = ["old", "locked"] if args.phase == "both" else [args.phase]
    for name in phases:
        t0 = time.time()
        ph = PHASES[name]
        use_store = name == "locked"
        print(f"== phase {name}: {ph} "
              f"({'persisted stores' if use_store else 'in-memory rebuild'})")
        inp = build_inputs(ph, use_store)
        rows_s, est, n_obs_s, ncat = shifters_table(inp, ph)
        rows_j, n_obs_j = bhj_table(inp, ph)
        if name == "old":
            new_s = pd.DataFrame([dict(statistic=l, intensity=a, composition=b)
                                  for l, a, b in rows_s])
            ref_s = pd.read_csv(GOLD / "rerun_many_shifters_diagnostics.csv",
                                dtype=str)
            assert new_s.astype(str).equals(ref_s), (
                "characterization FAIL: shifters table does not reproduce the "
                "captured rerun of producer 55")
            new_j = pd.DataFrame([dict(statistic=a, value=b) for a, b in rows_j])
            ref_j = pd.read_csv(GOLD / "rerun_bhj_shock_level.csv", dtype=str)
            assert new_j.astype(str).equals(ref_j), (
                "characterization FAIL: BHJ table does not reproduce the "
                "captured rerun of producer 56")
            print(f"  characterization: both tables string-equal to the "
                  f"captured reruns of producers 55 and 56 "
                  f"(n_obs {n_obs_s:,} / {n_obs_j:,}); {time.time()-t0:.1f}s")
        else:
            write_shifters(rows_s, est, n_obs_s, ncat)
            write_bhj(rows_j, ncat, n_obs_j)
            print(f"  locked tables built on {n_obs_s:,} / {n_obs_j:,} "
                  f"county-years; {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
