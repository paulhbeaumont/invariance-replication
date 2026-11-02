#!/usr/bin/env python3
"""How many shifters each county instrument actually has, and why it matters.

`document/single_factor_exposure_instruments.tex` shows that Nakamura and
Steinsson's alternative instrument is one predetermined exposure multiplied by one
national series. Two-way demeaning leaves

    ~z_it = ~s_i ~q_t,

so the instrument is RANK ONE in the cross section: the national series decides
which years count, never which units are treated more than which. A design like
that inherits neither the many-shock argument of Borusyak, Hull and Jaravel nor
the share decomposition of Goldsmith-Pinkham, Sorkin and Swift, and its validity
has to be defended as exposure-specific parallel trends plus exclusion.

Our county designs sit on the two sides of exactly that line, and this file
measures the distance rather than asserting it.

    51, the INTENSITY instrument      z_ct = (C_c,win / X_c,win) * q_t
        one national series q_t, so K = 1. The same single-factor object as
        NS, at a finer geography.

    54, the COMPOSITION instrument    z_ct = sum_k Cbar_ck,win * g_kt / X_ct-h
        K = 300 Federal Supply Classification categories, each with its own
        national series. This is Eq. (2) of the note, the multicomponent case.

WHAT IS REPORTED, AND WHY EACH LINE IS THERE

    K                        the number of shifters. The whole distinction.
    shock observations       K x T. What a shock-level law of large numbers has
                             to work with. BHJ's asymptotics run in this number,
                             not in the number of counties.
    effective number of      1 / HHI of the exposure-weighted importance of each
    shifters                 shifter. 300 categories are not 300 shocks if one
                             category carries all the dollars, and this is the
                             number BHJ's condition is actually about.
    variance in PC1          the share of the two-way demeaned instrument's
                             variance in its first cross-sectional direction.
                             The note's rank-one result says a single-factor
                             design must sit near 1. A multicomponent design
                             rotates: a county gets its shock in one year from
                             aircraft and another county in another year from
                             shipbuilding.
    PCs to reach 90%         the same fact read as a count of directions.
    median R2 on the         how much of a shifter is the common defense cycle.
    common cycle             If every category just tracks the aggregate budget,
                             K = 300 collapses back to K = 1 and nothing has been
                             gained. This is the test of whether the shifters are
                             distinct shocks or one shock wearing 300 hats.
    Rotemberg concentration  the share of the first-stage covariance contributed
                             by the largest and the five largest components,
                             following Goldsmith-Pinkham, Sorkin and Swift. If
                             one component supplies the estimate, the design is a
                             single-share design however many components it lists.

Everything is computed at the headline cell of 54: era full, h = 4, population
denominator, net rule, no floor, no screen, roll10 exposure, division x year
effects, division-equal weights, winsorized within year.

Output:
    4_Macro_version/Results/ns14_diagnostics/Table_many_shifters.tex
    batch_logs/macro_version/many_shifters_diagnostics.csv
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
OUT_CSV = dc.LOG_DIR / "many_shifters_diagnostics.csv"

H, SPAN, LAG = 4, 10, 5          # the headline cell: h = 4, roll10


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p54 = _load("p54", CODE / "54_composition_instrument_grid.py")
p51 = p54.p51


def two_way_demean(M):
    """Subtract unit and year means from a (units x years) matrix, equal weights.

    This is Eq. (4) of the note. Under it a single-factor instrument becomes
    exactly ~s_i ~q_t, whose matrix has rank one, which is what the PC1 share
    below is measuring against.
    """
    M = M - np.nanmean(M, axis=1, keepdims=True)
    M = M - np.nanmean(M, axis=0, keepdims=True)
    return M


def pc_profile(M):
    """Share of variance in the first component, and components to reach 90%."""
    ok = np.isfinite(M).all(axis=1)
    A = two_way_demean(M[ok])
    s = np.linalg.svd(A, compute_uv=False)
    v = s ** 2
    v = v / v.sum()
    return float(v[0]), int(np.searchsorted(np.cumsum(v), 0.90) + 1)


def hhi_effective(w):
    """1 / HHI of a non-negative importance vector, its effective count."""
    w = np.asarray(w, np.float64)
    w = np.abs(w)
    tot = w.sum()
    if tot <= 0:
        return np.nan
    p = w / tot
    return float(1.0 / (p ** 2).sum())


def main():
    t0 = time.time()
    print("=" * 78)
    print("  How many shifters does each county instrument have?")
    print("=" * 78)

    B = p51.base_panel()
    units = np.array(sorted(B["ccg"].unique()))
    CON = p51.contract_matrices(units, 1966, 2018)
    ARR, NAT, cats, cy = p54.category_panel(units, 1966, 2018)
    C = CON[("net", 0.0)]
    nat = NAT[("net", 0.0)]
    arr = ARR[("net", 0.0)]
    print(f"  {len(units):,} counties, {len(cats)} categories, "
          f"{time.time()-t0:.1f}s", flush=True)

    years = [t for t in range(max(1977, 1975 + H), 2019)]
    piv = {c: B.pivot(index="ccg", columns="year", values=c).reindex(units)
           for c in ("P", "E")}
    DIVI = B.drop_duplicates("ccg").set_index("ccg")["division"]

    # ---- the treatment, identical to 51's -------------------------------
    lag = {k: C.reindex(columns=[y - k for y in years]).set_axis(years, axis=1)
           for k in range(H + 1)}
    G_b = lag[H].to_numpy(np.float64)
    G_win = (sum(lag[k] for k in range(H)) / H).to_numpy(np.float64)
    P_b = piv["P"].reindex(columns=[y - H for y in years]).to_numpy(np.float64)
    E_b = piv["E"].reindex(columns=[y - H for y in years]).to_numpy(np.float64)
    E_t = piv["E"].reindex(columns=years).to_numpy(np.float64)
    X = (G_win - G_b) / P_b
    den = E_t + E_b
    Y = np.where(den != 0, 2.0 * (E_t - E_b) / den, np.nan)

    # ---- 51's instrument: ONE national series ---------------------------
    natshift = np.nansum(G_win - G_b, axis=0) / np.nansum(P_b, axis=0)   # q_t
    expo = np.full_like(X, np.nan)
    for j, t in enumerate(years):
        hi = t - H - LAG
        lo = hi - SPAN + 1
        if lo < p51.DATA_FIRST:
            continue
        yy = list(range(lo, hi + 1))
        num = C.loc[:, yy].sum(axis=1).to_numpy(np.float64)
        dpop = piv["P"].loc[:, yy].sum(axis=1, min_count=len(yy)).to_numpy(np.float64)
        expo[:, j] = num / dpop
    Z_INT = expo * natshift[None, :]

    # ---- 54's instrument: 300 national series ---------------------------
    gk = p54.category_rates(nat, H)
    L, nc = p54.predicted_dollars(arr, gk, cy, years, H, "roll", SPAN, LAG,
                                  p51.DATA_FIRST)
    Z_COM = L / P_b

    # ---- the estimation sample, shared by both --------------------------
    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z_INT) & np.isfinite(Z_COM)
    # A rolling window leaves the first years of the panel undefined for EVERY
    # county, so the dead years go before any county is asked to be complete.
    # Demanding completeness first empties the block instead of balancing it.
    cols = np.where(ok.any(axis=0))[0]
    rows = np.where(ok[:, cols].all(axis=1))[0]
    cols = cols[ok[np.ix_(rows, cols)].all(axis=0)]
    n_obs = int(ok.sum())
    print(f"  balanced block for the rank statistics: "
          f"{len(rows):,} counties x {len(cols)} years, of {n_obs:,} "
          f"estimation observations", flush=True)
    if len(rows) < 100 or len(cols) < 5:
        raise SystemExit("  FAIL: the balanced block is too small to rank")

    pc_int, k90_int = pc_profile(Z_INT[np.ix_(rows, cols)])
    pc_com, k90_com = pc_profile(Z_COM[np.ix_(rows, cols)])
    keep_cols = cols

    # ---- are the 300 shifters distinct, or one cycle in 300 hats? -------
    G = pd.DataFrame(gk, index=cats, columns=cy).loc[:, years[0]:years[-1]]
    common = G.mean(axis=0)
    r2 = []
    for k in G.index:
        v = G.loc[k].dropna()
        if len(v) > 5 and v.std() > 0:
            r2.append(np.corrcoef(v, common.reindex(v.index))[0, 1] ** 2)
    r2 = np.array(r2)

    # ---- effective number of shifters -----------------------------------
    dollars_k = arr.sum(axis=(0, 2)).astype(np.float64)      # national $ per category
    eff_com = hhi_effective(dollars_k)

    # ---- Rotemberg-style first-stage contributions ----------------------
    # alpha_k is proportional to the covariance of component k with the demeaned
    # treatment. Demeaning is a projection and self-adjoint under the estimation
    # weights, so the raw component can be used against the demeaned treatment.
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
    t28 = _load("t28", STEP2_CODE / "28_table_d1.py")
    w22 = _load("w22", STEP2_CODE / "22_winsor_construction_error.py")
    import pyfixest as pf
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100_000)
    xw = p51.wins(g["x"].to_numpy(np.float64), g["year"].to_numpy())
    tot = g.groupby(["division", "year"])["E_b"].transform("sum")
    w = (g["E_b"] / tot).to_numpy(np.float64)
    w = w / w.mean()
    fe = pd.factorize(g["division"].astype(str) + "_"
                      + g["year"].astype(str))[0].astype(np.int64)
    nofe = np.zeros(len(g), np.int64)
    xt = w22.demean(xw.reshape(-1, 1), fe, nofe, w)[:, 0]

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
            if lo < p51.DATA_FIRST:
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

    # ---- the two second-stage estimates, for the table's footer ---------
    cl = pd.factorize(g["ccg"])[0].astype(np.int64)
    est = {}
    for lab, col in (("intensity", "z_int"), ("composition", "z_com")):
        r = t28.fit2sls(g["y"].to_numpy(np.float64), xw,
                        g[col].to_numpy(np.float64), w, fe, nofe, cl,
                        singleton=False)
        est[lab] = (r["beta"] * 1e5, r["beta"] / r["se"], r["F"])

    T = int(len(keep_cols))
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
    print(f"\n  {'':<44}{'intensity':>12}{'composition':>14}")
    for lab, a, b in rows:
        clean = (lab.replace("$", "").replace("\\times", "x")
                 .replace("\\%", "%").replace("^2", "2"))
        print(f"  {clean:<44}{a:>12}{b:>14}")
    print(f"\n  {'second stage: beta, t, F':<44}"
          f"{est['intensity'][0]:>6.2f} {est['intensity'][1]:>4.2f} "
          f"{est['intensity'][2]:>5.1f}"
          f"{est['composition'][0]:>8.2f} {est['composition'][1]:>4.2f} "
          f"{est['composition'][2]:>5.1f}")
    print(f"  {'observations':<44}{n_obs:>12,}{n_obs:>14,}")

    pd.DataFrame([dict(statistic=l, intensity=a, composition=b)
                  for l, a, b in rows]).to_csv(OUT_CSV, index=False)

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    tex = TAB_DIR / "Table_many_shifters.tex"
    body = "\n".join(f"{l} & {a} & {b} \\\\\\\\" for l, a, b in rows)
    tex.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{How Many Shifters Each County Instrument Has}}
\\label{{tab:many_shifters}}
{{\\centering
\\adjustbox{{max width=\\linewidth}}{{%
\\begin{{tabular}}{{lrr}}
\\toprule
 & Intensity & Composition \\\\\\\\
 & (one national series) & (300 categories) \\\\\\\\
\\midrule
{body}
\\midrule
Second-stage coefficient & {est['intensity'][0]:.2f} & {est['composition'][0]:.2f} \\\\\\\\
\\hspace{{1.5em}}$t$ & {est['intensity'][1]:.2f} & {est['composition'][1]:.2f} \\\\\\\\
\\hspace{{1.5em}}First-stage $F$ & {est['intensity'][2]:.1f} & {est['composition'][2]:.1f} \\\\\\\\
\\addlinespace Observations & {n_obs:,} & {n_obs:,} \\\\\\\\
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} Both instruments are estimated on the same county-year
sample at a four-year horizon, with division-by-year fixed effects, division-equal weights and
the treatment winsorized within year. The intensity instrument multiplies a county's contract
exposure over a predetermined ten-year window by one national procurement series, so $K=1$; it
is the county analogue of the Nakamura-Steinsson alternative instrument. The composition
instrument meets the county's contract mix across 300 four-digit Federal Supply Classification
categories with the national change in demand for each, so $K=300$. The effective number of
shifters is the inverse Herfindahl of national dollars by category. Variance in the first
component is the share of the two-way demeaned instrument matrix's variance in its leading
singular direction; a single-factor instrument is rank one in the cross section and sits at
one. The common cycle is the equal-weighted mean of the category series. First-stage shares
follow Goldsmith-Pinkham, Sorkin and Swift, measuring each component's contribution to the
first-stage covariance.\\par}}
\\end{{table}}
""")
    print(f"\n  wrote {tex}")
    print(f"  wrote {OUT_CSV}")
    print(f"  {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
