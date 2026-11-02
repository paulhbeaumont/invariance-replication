"""Stage 52: county block local projections for the locked two-year shock.

The right-hand-side variable is the persisted h=2 composition shock spanning
[t-2,t]. The outcome windows are the direct block analogue of producer 50:

  LEAD    k = 3,4,5   log E_{c,t-2} - log E_{c,t-k}
  IMPACT  h = 0       log E_{c,t} - log E_{c,t-2}
  POST    h = 1..4    log E_{c,t+h} - log E_{c,t}

One equal-county-weighted specification runs on a common county-year rectangle.
It controls for z2_{c,t-2}, z2_{c,t-4}, z2_{c,t-6}, and S0_ct. The impact and
post regressions also control for annual log employment growth ending at t-2,
t-3, and t-4. Lead regressions omit those outcome lags because they equal or
overlap their dependent-variable windows.

The regression uses division-by-year fixed effects. Exposure-robust standard
errors are computed from the equivalent shock-level regression and clustered
on the FSC category. County-clustered standard errors are saved as a comparison.

Consumes persisted pipeline stores. Writes Results/lp_2yr/:

  Figure_lp2_irf.pdf
  Figure_lp2_irf.tex
  lp2_dynamics.csv
  lp2_summary.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import oldpath as op  # noqa: E402
import store  # noqa: E402
from clusterstats import cluster_sums, hhi_effective, wls_fe  # noqa: E402

OUT = PIPELINE.parents[2] / "Results" / "lp_2yr"
H_SHOCK = 2
LEAD_DEPTHS = (5, 4, 3)
POST_HORIZONS = (1, 2, 3, 4)
EVENT_TIMES = tuple(-k for k in LEAD_DEPTHS) + (0,) + POST_HORIZONS
PRE_TIMES = tuple(-k for k in LEAD_DEPTHS)
SHOCK_LAGS = (2, 4, 6)
OUTCOME_LAGS = ("gl1", "gl2", "gl3")
SAMPLE_LAST = locked.TARGET_LAST
Z_FIRST = locked.DATA_FIRST + locked.LAG + locked.SPAN - 1 + H_SHOCK
YEAR_LO = Z_FIRST + max(SHOCK_LAGS)
YEAR_HI = SAMPLE_LAST - max(POST_HORIZONS)
N_EXPECT = locked.N_COUNTIES * (YEAR_HI - YEAR_LO + 1)
SCALE = locked.SCALE
Z95 = 1.959963984540054
VARIANTS = {
    "three_lags": ("s0", "zlag2", "zlag4", "zlag6"),
}
LABELS = {
    "three_lags": r"Three past two-year shocks",
}


def control_names(variant, ell):
    """Return controls consistent with the timing of one outcome window."""
    controls = list(VARIANTS[variant])
    if ell >= 0:
        controls.extend(OUTCOME_LAGS)
    return tuple(controls)


def outcome_offsets(ell):
    """Return the start and end offsets of one displayed outcome window."""
    if ell < 0:
        return ell, -H_SHOCK
    if ell == 0:
        return -H_SHOCK, 0
    return 0, ell


def fwl(C, w, *vectors):
    """Residualize vectors on a weighted, already FE-demeaned control block."""
    if C.shape[1] == 0:
        return list(vectors)
    sw = np.sqrt(w)
    A = C * sw[:, None]
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    keep = s > max(A.shape) * np.finfo(np.float64).eps * s.max()
    U = U[:, keep]
    return [(v * sw - U @ (U.T @ (v * sw))) / sw for v in vectors]


def load_inputs():
    """Load stores and construct one common frame for both specifications."""
    pnl = store.load_county_panel()
    units, piv = store.load_pivots(pnl, cols=("P", "E"))
    division = pnl.drop_duplicates("ccg").set_index("ccg")["division"]
    units2, cats, cy, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")

    target = store.target_years(H_SHOCK)
    zfull = store.load_instrument(H_SHOCK, units, "z")
    zpos = {year: j for j, year in enumerate(target)}
    gk = store.load_gk(H_SHOCK, cats, cy)

    finite = np.isfinite(zfull)
    finite_years = np.array(target)[finite.any(axis=0)]
    if int(finite.sum()) != locked.N_COUNTY[H_SHOCK]:
        raise SystemExit(
            f"FAIL: h=2 instrument has {int(finite.sum()):,} finite cells, "
            f"not locked.N_COUNTY[2]={locked.N_COUNTY[H_SHOCK]:,}")
    if int(finite_years.min()) != Z_FIRST or int(finite_years.max()) != SAMPLE_LAST:
        raise SystemExit(
            f"FAIL: h=2 instrument spans {int(finite_years.min())}-"
            f"{int(finite_years.max())}, expected {Z_FIRST}-{SAMPLE_LAST}")

    years = list(range(YEAR_LO, YEAR_HI + 1))

    def ec(offset):
        cols = [t + offset for t in years]
        return piv["E"].reindex(columns=cols).to_numpy(np.float64)

    def pc(offset):
        cols = [t + offset for t in years]
        return piv["P"].reindex(columns=cols).to_numpy(np.float64)

    z2 = zfull[:, [zpos[t] for t in years]]
    zlags = {
        lag: zfull[:, [zpos[t - lag] for t in years]]
        for lag in SHOCK_LAGS
    }
    pbase = pc(-H_SHOCK)

    cbar = {}
    s0 = np.full_like(z2, np.nan)
    zcheck = np.full_like(z2, np.nan)
    for j, t in enumerate(years):
        hi = t - H_SHOCK - locked.LAG
        lo = hi - locked.SPAN + 1
        a, b = cy.index(lo), cy.index(hi)
        exposure = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        cbar[j] = exposure
        s0[:, j] = exposure.sum(axis=1) / pbase[:, j]
        gt = gk[:, cy.index(t)]
        live = np.isfinite(gt)
        zcheck[:, j] = exposure[:, live] @ gt[live] / pbase[:, j]

    gap = np.nanmax(np.abs(zcheck - z2))
    scale = max(float(np.nanmean(np.abs(z2))), 1e-300)
    if gap / scale >= locked.RTOL_COEF:
        raise SystemExit(
            f"FAIL: reconstructed h=2 instrument differs from the persisted "
            f"object, max gap relative to mean |z|={gap / scale:.2e}")

    outcomes = {}
    for ell in EVENT_TIMES:
        start, end = outcome_offsets(ell)
        outcomes[ell] = np.log(ec(end)) - np.log(ec(start))

    outcome_lags = {
        f"gl{j}": (
            np.log(ec(-H_SHOCK - (j - 1)))
            - np.log(ec(-H_SHOCK - j)))
        for j in range(1, len(OUTCOME_LAGS) + 1)
    }

    ok = np.isfinite(z2) & np.isfinite(s0)
    for value in (list(zlags.values()) + list(outcomes.values()) +
                  list(outcome_lags.values())):
        ok &= np.isfinite(value)
    if not ok.all():
        raise SystemExit(
            f"FAIL: common frame loses {int((~ok).sum()):,} county-years")
    if int(ok.sum()) != N_EXPECT:
        raise SystemExit(
            f"FAIL: common frame has {int(ok.sum()):,} rows, expected "
            f"{N_EXPECT:,}")

    ri, ci = np.where(ok)
    year_array = np.array(years)
    d = pd.DataFrame({
        "ccg": units[ri],
        "year": year_array[ci],
        "z": z2[ok],
        "s0": s0[ok],
        "pbase": pbase[ok],
    })
    for lag, value in zlags.items():
        d[f"zlag{lag}"] = value[ok]
    d["division"] = d["ccg"].map(division)
    if d["division"].isna().any():
        raise SystemExit("FAIL: division is missing on the estimation frame")

    raw = {"z": d["z"].to_numpy(), "s0": d["s0"].to_numpy()}
    for lag in SHOCK_LAGS:
        raw[f"zlag{lag}"] = d[f"zlag{lag}"].to_numpy()
    for ell, y in outcomes.items():
        raw[f"y{ell:+d}"] = y[ok]
    for name, value in outcome_lags.items():
        raw[name] = value[ok]

    w = np.ones(len(d), np.float64)
    fe = pd.factorize(d["division"].astype(str) + "_" +
                      d["year"].astype(str))[0].astype(np.int64)
    nofe = np.zeros(len(d), np.int64)
    county_cluster = pd.factorize(d["ccg"])[0].astype(np.int64)

    import pyfixest as pf
    op.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-10,
                                     fixef_maxiter=100_000)
    names = sorted(raw)
    mat = op.w22.demean(np.column_stack([raw[n] for n in names]),
                        fe, nofe, w)
    dm = {name: mat[:, j] for j, name in enumerate(names)}

    rr = ri
    cc = ci
    pb = d["pbase"].to_numpy(np.float64)
    blocks = {}
    keys, sv, gv = [], [], []
    for j, t in enumerate(years):
        m = cc == j
        exposure = cbar[j][rr[m], :] / pb[m, None]
        blocks[j] = (m, exposure)
        importance = exposure.sum(axis=0)
        gt = gk[:, cy.index(t)]
        for k in np.flatnonzero(np.isfinite(gt)):
            keys.append((k, j))
            sv.append(importance[k])
            gv.append(gt[k])
    keys = np.asarray(keys, dtype=np.int64)
    sv = np.asarray(sv, dtype=np.float64)
    gv = np.asarray(gv, dtype=np.float64)
    live = np.abs(sv) > 0
    shock = {
        "keys": keys[live], "sv": sv[live], "gv": gv[live],
        "blocks": blocks,
    }
    shock["cluster"] = pd.factorize(shock["keys"][:, 0])[0].astype(np.int64)
    shock["ncluster"] = int(shock["cluster"].max()) + 1
    category_weight = (pd.Series(shock["sv"])
                       .groupby(shock["keys"][:, 0]).sum().to_numpy())
    shock["effective_categories"] = hhi_effective(category_weight)

    return {
        "d": d, "raw": raw, "dm": dm, "w": w, "fe": fe, "nofe": nofe,
        "county_cluster": county_cluster, "shock": shock, "cats": cats,
        "years": years, "instrument_gap": gap / scale,
    }


def aggregate(shock, vector):
    """Aggregate a county-residualized vector to category-by-year cells."""
    by_year = {
        j: exposure.T @ vector[mask]
        for j, (mask, exposure) in shock["blocks"].items()
    }
    return np.array([by_year[j][k] for k, j in shock["keys"]])


def fit_horizon(inp, variant, ell):
    """Estimate one LP coefficient by county and equivalent shock routes."""
    raw, dm, w = inp["raw"], inp["dm"], inp["w"]
    controls = control_names(variant, ell)
    yname = f"y{ell:+d}"

    X = np.column_stack([raw["z"]] + [raw[c] for c in controls])
    b_county, se_county, _ = wls_fe(
        raw[yname], X, w, inp["fe"], inp["nofe"], inp["county_cluster"])

    C = np.column_stack([dm[c] for c in controls])
    yr, zr = fwl(C, w, dm[yname], dm["z"])
    denominator = float((w * zr) @ zr)
    beta = float((w * zr) @ yr) / denominator
    route_gap = abs(beta - b_county[0]) / max(abs(beta), 1e-300)
    if route_gap >= locked.RTOL_COEF:
        raise SystemExit(
            f"FAIL: county FWL and multi-regressor routes disagree for "
            f"{variant}, ell={ell}, relative gap={route_gap:.2e}")

    shock = inp["shock"]
    ny, nz = aggregate(shock, yr), aggregate(shock, zr)
    g = shock["gv"]
    D = float(g @ nz)
    beta_shock = float(g @ ny) / D
    equiv_gap = abs(beta_shock - beta) / max(abs(beta), 1e-300)
    if equiv_gap >= locked.RTOL_COEF:
        raise SystemExit(
            f"FAIL: county and shock-level routes disagree for {variant}, "
            f"ell={ell}, relative gap={equiv_gap:.2e}")

    score = g * (ny - beta * nz)
    cluster = shock["cluster"]
    sums = cluster_sums(score, cluster, shock["ncluster"])
    G = int(np.unique(cluster).size)
    variance = (G / max(G - 1, 1)) * float(sums @ sums) / (D * D)
    se_exposure = float(np.sqrt(max(variance, 0.0)))

    return {
        "variant": variant, "ell": ell, "beta": beta,
        "se_exposure": se_exposure, "se_county": float(se_county[0]),
        "t_exposure": beta / se_exposure, "n": len(yr),
        "n_controls": len(controls), "n_shock_clusters": G,
        "route_gap": route_gap, "equiv_gap": equiv_gap,
        "score": score, "D": D, "cluster": cluster,
    }


def joint_pretest(fits):
    """Joint Wald test for the three nested pre-shock outcome windows."""
    selected = sorted((f for f in fits if f["ell"] in PRE_TIMES),
                      key=lambda f: f["ell"])
    b = np.array([f["beta"] for f in selected])
    cluster = selected[0]["cluster"]
    D = selected[0]["D"]
    for fit in selected:
        if not np.array_equal(fit["cluster"], cluster) or fit["D"] != D:
            raise SystemExit("FAIL: pre-period fits do not share one design")
    ncl = int(cluster.max()) + 1
    scores = np.column_stack([
        cluster_sums(f["score"], cluster, ncl) for f in selected
    ])
    G = int(np.unique(cluster).size)
    V = (G / max(G - 1, 1)) * (scores.T @ scores) / (D * D)
    k = len(b)
    F = float(b @ np.linalg.solve(V, b) / k)
    return {"F": F, "p": float(stats.f.sf(F, k, G - 1)),
            "df_num": k, "df_den": G - 1}


def write_outputs(inp, fits, tests):
    """Write coefficient and summary CSV files."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in fits:
        start, end = outcome_offsets(f["ell"])
        rows.append({
            "variant": f["variant"], "variant_label": LABELS[f["variant"]],
            "event_time": f["ell"],
            "period": ("pre" if f["ell"] < 0 else
                       "impact" if f["ell"] == 0 else "post"),
            "outcome_start_offset": start,
            "outcome_end_offset": end,
            "outcome_window_years": end - start,
            "coef": f["beta"] * SCALE,
            "se_exposure": f["se_exposure"] * SCALE,
            "se_county": f["se_county"] * SCALE,
            "t_exposure": f["t_exposure"], "n": f["n"],
            "n_controls": f["n_controls"],
            "n_shock_clusters": f["n_shock_clusters"],
            "year_lo": YEAR_LO, "year_hi": YEAR_HI,
            "route_gap": f["route_gap"], "equiv_gap": f["equiv_gap"],
        })
    pd.DataFrame(rows).to_csv(OUT / "lp2_dynamics.csv", index=False)

    summary = []
    for variant in VARIANTS:
        test = tests[variant]
        summary.append({
            "variant": variant, "variant_label": LABELS[variant],
            "past_shock_lags": 3,
            "past_shock_offsets": "2,4,6",
            "outcome_lags_impact_post": len(OUTCOME_LAGS),
            "outcome": "log employment growth",
            "share_sum_control": True, "weight": "equal county",
            "fixed_effects": "division-by-year",
            "joint_pre_F": test["F"], "joint_pre_p": test["p"],
            "joint_pre_df_num": test["df_num"],
            "joint_pre_df_den": test["df_den"],
            "n": N_EXPECT, "n_counties": locked.N_COUNTIES,
            "year_lo": YEAR_LO, "year_hi": YEAR_HI,
            "shock_horizon": H_SHOCK,
            "effective_shifters_category":
                inp["shock"]["effective_categories"],
            "instrument_reconstruction_gap": inp["instrument_gap"],
        })
    pd.DataFrame(summary).to_csv(OUT / "lp2_summary.csv", index=False)


def write_figure(fits):
    """Write the coefficient plot and LaTeX fragment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font_path = Path("/System/Library/Fonts/Supplemental/GillSans.ttc")
    base_font = (fm.FontProperties(fname=font_path).get_name()
                 if font_path.exists() else "DejaVu Sans")
    navy = "#1A476F"
    dark, grid, reference = "#1A1A1A", "#E6E6E6", "#AAAAAA"
    plt.rcParams.update({
        "font.family": base_font, "font.size": 10,
        "figure.figsize": (8, 4.5), "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": dark,
        "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": grid, "grid.linewidth": 0.4,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots()
    ax.axvspan(-2, 0, color="#E6E6E6", alpha=0.75, zorder=0)
    ax.axhline(0, color=reference, linewidth=0.8, zorder=1)
    subset = sorted(fits, key=lambda f: f["ell"])
    x = np.array([f["ell"] for f in subset], dtype=float)
    b = np.array([f["beta"] for f in subset]) * SCALE
    se = np.array([f["se_exposure"] for f in subset]) * SCALE
    ax.errorbar(x, b, yerr=Z95 * se, fmt="o", linestyle="none",
                color=navy, ecolor=navy, capsize=3, elinewidth=1.0,
                markersize=5, zorder=3)

    ax.set_xticks(range(-5, 5))
    ax.set_xlabel("Horizon relative to the end of the two-year shock")
    ax.set_ylabel(r"$\beta \times 10^{5}$, log employment growth")
    fig.subplots_adjust(bottom=0.16)
    pdf = OUT / "Figure_lp2_irf.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    plt.close(fig)

    fragment = rf"""% Produced by step_1/code/pipeline/52_two_year_shock_lp.py
% from Results/lp_2yr/lp2_dynamics.csv. Do not edit.
\noindent\begin{{minipage}}{{\textwidth}}\centering
\textbf{{Figure. Employment dynamics around the two-year shock}}\\[0.2em]
{{\small County-level, equal-weighted local projections}}\\[0.6em]
\includegraphics[width=\textwidth]{{{pdf.name}}}
\end{{minipage}}
\par\smallskip
\noindent\begin{{minipage}}{{\textwidth}}\footnotesize\raggedright
\textit{{Notes:}} This figure plots the point estimates and 95\% confidence
intervals from separate reduced-form regressions of county log employment
growth on the locked
two-year composition shock $z^{{(2)}}_{{ct}}$, scaled by $10^5$.
The confidence intervals use exposure-robust standard errors clustered on the
FSC category. The gray strip marks the two-year shock window
$[t-2,t]$. No separate coefficient is estimated inside that window; the marker
at horizon 0 is its complete two-year impact.
\smallskip
The specification uses equal county weights, the share-sum level control
$S0_{{ct}}$, division-by-year fixed effects, and one common sample at every
horizon. It controls for the three nonoverlapping past shocks
$z^{{(2)}}_{{c,t-2}}$, $z^{{(2)}}_{{c,t-4}}$, and
$z^{{(2)}}_{{c,t-6}}$. Annual log employment growth ending at $t-2$, $t-3$,
and $t-4$ enters the impact and post regressions.
Those outcome lags are excluded from the lead regressions because they equal or
overlap the dependent-variable windows.
\smallskip
The lead markers at $-5,-4,-3$ use log employment growth over $[t-5,t-2]$,
$[t-4,t-2]$, and $[t-3,t-2]$.
The impact marker at 0 uses $[t-2,t]$. A post marker at horizon $h=1,\ldots,4$
uses $[t,t+h]$. Thus every lead window ends before the shock, the impact covers
the complete shock, and every post window begins after it.
\par
\end{{minipage}}
"""
    (OUT / "Figure_lp2_irf.tex").write_text(fragment, encoding="utf-8")


def main():
    start = time.time()
    print("== stage 52: county LP around the locked two-year shock")
    print(f"   common sample [{YEAR_LO},{YEAR_HI}], equal weights, "
          "division-by-year fixed effects")
    inp = load_inputs()
    print(f"   h=2 instrument reconstruction gap "
          f"{inp['instrument_gap']:.2e} relative to mean |z|")
    print(f"   {len(inp['d']):,} county-years, "
          f"{inp['d']['ccg'].nunique():,} counties")
    print(f"   {len(inp['shock']['gv']):,} live category-year shock cells, "
          f"{inp['shock']['effective_categories']:.1f} effective categories")

    fits, tests = [], {}
    for variant in VARIANTS:
        print(f"\n-- {LABELS[variant]}")
        vf = []
        for ell in EVENT_TIMES:
            fit = fit_horizon(inp, variant, ell)
            vf.append(fit)
            fits.append(fit)
            print(f"   ell={ell:+d}: beta={fit['beta'] * SCALE:+.4f}, "
                  f"se_exp={fit['se_exposure'] * SCALE:.4f}, "
                  f"se_county={fit['se_county'] * SCALE:.4f}, "
                  f"t={fit['t_exposure']:+.2f}")
        tests[variant] = joint_pretest(vf)
        test = tests[variant]
        print(f"   joint pre-shock test: F={test['F']:.3f}, "
              f"p={test['p']:.4f}, F({test['df_num']},{test['df_den']})")

    write_outputs(inp, fits, tests)
    write_figure(fits)
    max_route = max(f["route_gap"] for f in fits)
    max_equiv = max(f["equiv_gap"] for f in fits)
    print(f"\n   county FWL route max relative gap: {max_route:.2e}")
    print(f"   county versus shock-level max relative gap: {max_equiv:.2e}")
    print(f"   wrote {OUT / 'lp2_dynamics.csv'}")
    print(f"   wrote {OUT / 'lp2_summary.csv'}")
    print(f"   wrote {OUT / 'Figure_lp2_irf.pdf'}")
    print(f"   wrote {OUT / 'Figure_lp2_irf.tex'}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
