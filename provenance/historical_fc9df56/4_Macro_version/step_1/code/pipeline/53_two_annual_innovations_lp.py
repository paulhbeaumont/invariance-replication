"""Stage 53: two-year episode built from two annual shock innovations.

The two focal regressors are the annual composition shocks dated t and t-1.
For every outcome horizon, estimate their coefficients jointly and report their
equal-weighted average. This normalizes a one-unit total two-year innovation as
one-half unit in each episode year. Its exposure-robust standard error uses the
full covariance between the two coefficient estimates.

The horizon geometry matches the accepted annual local projection:

  LEAD    k = 3,4,5   log E_{c,t-2} - log E_{c,t-k}
  IMPACT  h = 0       log E_{c,t} - log E_{c,t-2}
  POST    h = 1..4    log E_{c,t+h} - log E_{c,t}

Every regression uses equal county weights, division-by-year fixed effects,
and the two focal share-sum controls. The baseline controls for annual shocks
dated t-2 through t-4; the expanded specification controls through t-8.
Impact and post regressions also control for annual log employment growth
ending at t-2, t-3, and t-4. Both specifications and all horizons use the
same sample.

Writes Results/lp_2yr_annual_innovations/:

  Figure_lp2_annual_innovations.pdf
  Figure_lp2_annual_innovations.tex
  lp2_annual_innovations.csv
  lp2_annual_innovations_summary.csv
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
import instrument  # noqa: E402
import locked  # noqa: E402
import oldpath as op  # noqa: E402
import store  # noqa: E402
from clusterstats import wls_fe  # noqa: E402

OUT = PIPELINE.parents[2] / "Results" / "lp_2yr_annual_innovations"
H_ANNUAL = 1
LEAD_DEPTHS = (5, 4, 3)
POST_HORIZONS = (1, 2, 3, 4)
EVENT_TIMES = tuple(-k for k in LEAD_DEPTHS) + (0,) + POST_HORIZONS
PRE_TIMES = tuple(-k for k in LEAD_DEPTHS)
ALL_PAST_SHOCK_LAGS = tuple(range(2, 9))
VARIANTS = {
    "lags_2_4": tuple(range(2, 5)),
    "lags_2_8": ALL_PAST_SHOCK_LAGS,
}
LABELS = {
    "lags_2_4": "Shock lags 2 to 4",
    "lags_2_8": "Shock lags 2 to 8",
}
OUTCOME_LAGS = ("gl1", "gl2", "gl3")
Z1_FIRST = locked.DATA_FIRST + locked.LAG + locked.SPAN
YEAR_LO = Z1_FIRST + max(ALL_PAST_SHOCK_LAGS)
YEAR_HI = locked.TARGET_LAST - max(POST_HORIZONS)
N_EXPECT = locked.N_COUNTIES * (YEAR_HI - YEAR_LO + 1)
SCALE = locked.SCALE
Z95 = 1.959963984540054


def outcome_offsets(ell):
    """Return the outcome-window endpoints relative to episode end t."""
    if ell < 0:
        return ell, -2
    if ell == 0:
        return -2, 0
    return 0, ell


def fwl(C, w, *vectors):
    """Residualize vectors on weighted, fixed-effect-demeaned controls."""
    if C.shape[1] == 0:
        return list(vectors)
    sw = np.sqrt(w)
    A = C * sw[:, None]
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    keep = s > max(A.shape) * np.finfo(np.float64).eps * s.max()
    U = U[:, keep]
    return [(v * sw - U @ (U.T @ (v * sw))) / sw for v in vectors]


def load_inputs():
    """Build the common frame and the two annual exposure maps."""
    pnl = store.load_county_panel()
    units, piv = store.load_pivots(pnl, cols=("P", "E"))
    division = pnl.drop_duplicates("ccg").set_index("ccg")["division"]
    units2, cats, cy, arr, nat = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")

    target = store.target_years(H_ANNUAL)
    g1 = instrument.category_rates(nat, H_ANNUAL)
    L1, _ = instrument.predicted_dollars(
        arr, g1, cy, target, H_ANNUAL, "roll", locked.SPAN,
        locked.LAG, locked.DATA_FIRST)
    pden = piv["P"].reindex(columns=[t - 1 for t in target]).to_numpy(
        np.float64)
    zfull = L1 / pden
    zpos = {year: j for j, year in enumerate(target)}

    finite = np.isfinite(zfull)
    finite_years = np.array(target)[finite.any(axis=0)]
    if int(finite.sum()) != locked.N_COUNTY_UNGATED[H_ANNUAL]:
        raise SystemExit(
            f"FAIL: annual instrument has {int(finite.sum()):,} finite cells, "
            f"not {locked.N_COUNTY_UNGATED[H_ANNUAL]:,}")
    if (int(finite_years.min()), int(finite_years.max())) != (
            Z1_FIRST, locked.TARGET_LAST):
        raise SystemExit(
            f"FAIL: annual instrument spans {int(finite_years.min())}-"
            f"{int(finite_years.max())}, expected "
            f"{Z1_FIRST}-{locked.TARGET_LAST}")

    years = list(range(YEAR_LO, YEAR_HI + 1))

    def ec(offset):
        return piv["E"].reindex(
            columns=[t + offset for t in years]).to_numpy(np.float64)

    def pc_for_dates(dates):
        return piv["P"].reindex(
            columns=[s - 1 for s in dates]).to_numpy(np.float64)

    focal = {
        j: zfull[:, [zpos[t - j] for t in years]]
        for j in (0, 1)
    }
    past = {
        lag: zfull[:, [zpos[t - lag] for t in years]]
        for lag in ALL_PAST_SHOCK_LAGS
    }

    exposure_maps = {0: {}, 1: {}}
    share_sums = {0: np.full_like(focal[0], np.nan),
                  1: np.full_like(focal[1], np.nan)}
    reconstruction = {0: np.full_like(focal[0], np.nan),
                      1: np.full_like(focal[1], np.nan)}
    for j in (0, 1):
        dates = [t - j for t in years]
        denom = pc_for_dates(dates)
        for q, sdate in enumerate(dates):
            hi = sdate - H_ANNUAL - locked.LAG
            lo = hi - locked.SPAN + 1
            a, b = cy.index(lo), cy.index(hi)
            exposure = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
            exposure = exposure / denom[:, q, None]
            exposure_maps[j][q] = exposure
            share_sums[j][:, q] = exposure.sum(axis=1)
            g = g1[:, cy.index(sdate)]
            live = np.isfinite(g)
            reconstruction[j][:, q] = exposure[:, live] @ g[live]

    gaps = []
    for j in (0, 1):
        gap = np.nanmax(np.abs(reconstruction[j] - focal[j]))
        scale = max(float(np.nanmean(np.abs(focal[j]))), 1e-300)
        gaps.append(gap / scale)
        if gaps[-1] >= locked.RTOL_COEF:
            raise SystemExit(
                f"FAIL: annual instrument reconstruction differs at focal "
                f"lag {j}, relative gap={gaps[-1]:.2e}")

    outcomes = {}
    for ell in EVENT_TIMES:
        start, end = outcome_offsets(ell)
        outcomes[ell] = np.log(ec(end)) - np.log(ec(start))
    outcome_lags = {
        f"gl{j}": np.log(ec(-2 - (j - 1))) - np.log(ec(-2 - j))
        for j in range(1, len(OUTCOME_LAGS) + 1)
    }

    values = (list(focal.values()) + list(past.values())
              + list(share_sums.values()) + list(outcomes.values())
              + list(outcome_lags.values()))
    ok = np.logical_and.reduce([np.isfinite(v) for v in values])
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
        "z_current": focal[0][ok],
        "z_prior": focal[1][ok],
        "s0_current": share_sums[0][ok],
        "s0_prior": share_sums[1][ok],
    })
    for lag, value in past.items():
        d[f"zlag{lag}"] = value[ok]
    d["division"] = d["ccg"].map(division)
    if d["division"].isna().any():
        raise SystemExit("FAIL: division is missing on the estimation frame")

    raw = {
        "z_current": d["z_current"].to_numpy(),
        "z_prior": d["z_prior"].to_numpy(),
        "s0_current": d["s0_current"].to_numpy(),
        "s0_prior": d["s0_prior"].to_numpy(),
    }
    for lag in ALL_PAST_SHOCK_LAGS:
        raw[f"zlag{lag}"] = d[f"zlag{lag}"].to_numpy()
    for ell, value in outcomes.items():
        raw[f"y{ell:+d}"] = value[ok]
    for name, value in outcome_lags.items():
        raw[name] = value[ok]

    w = np.ones(len(d), np.float64)
    fe = pd.factorize(
        d["division"].astype(str) + "_" + d["year"].astype(str)
    )[0].astype(np.int64)
    nofe = np.zeros(len(d), np.int64)
    county_cluster = pd.factorize(d["ccg"])[0].astype(np.int64)

    import pyfixest as pf
    op.w22.DEMEANER = pf.MapDemeaner(
        fixef_tol=1e-10, fixef_maxiter=100_000)
    names = sorted(raw)
    matrix = op.w22.demean(
        np.column_stack([raw[name] for name in names]), fe, nofe, w)
    dm = {name: matrix[:, q] for q, name in enumerate(names)}

    blocks = {0: {}, 1: {}}
    for j in (0, 1):
        for q in range(len(years)):
            mask = ci == q
            blocks[j][q] = (mask, exposure_maps[j][q][ri[mask], :])

    return {
        "d": d, "raw": raw, "dm": dm, "w": w, "fe": fe,
        "nofe": nofe, "county_cluster": county_cluster,
        "blocks": blocks, "g1": g1, "cy": cy, "cats": cats,
        "years": years, "reconstruction_gap": max(gaps),
    }


def controls_for(variant, ell):
    """Controls matching the annual LP architecture."""
    controls = ["s0_current", "s0_prior"]
    controls.extend(f"zlag{lag}" for lag in VARIANTS[variant])
    if ell >= 0:
        controls.extend(OUTCOME_LAGS)
    return tuple(controls)


def exposure_moments(inp, vector):
    """Aggregate one county vector for both focal annual instruments."""
    moments = np.zeros((2, len(inp["cats"])), np.float64)
    for j in (0, 1):
        for q, t in enumerate(inp["years"]):
            mask, exposure = inp["blocks"][j][q]
            aggregate = exposure.T @ vector[mask]
            g = inp["g1"][:, inp["cy"].index(t - j)]
            moments[j] += np.where(np.isfinite(g), g, 0.0) * aggregate
    return moments


def fit_horizon(inp, variant, ell):
    """Estimate both annual innovations and their two-year equivalent."""
    raw, dm, w = inp["raw"], inp["dm"], inp["w"]
    controls = controls_for(variant, ell)
    yname = f"y{ell:+d}"
    X = np.column_stack(
        [raw["z_current"], raw["z_prior"]]
        + [raw[name] for name in controls])
    b_county, se_county, _ = wls_fe(
        raw[yname], X, w, inp["fe"], inp["nofe"],
        inp["county_cluster"])

    C = np.column_stack([dm[name] for name in controls])
    yr, x0r, x1r = fwl(
        C, w, dm[yname], dm["z_current"], dm["z_prior"])
    Xr = np.column_stack([x0r, x1r])
    A = Xr.T @ (w[:, None] * Xr)
    rhs = Xr.T @ (w * yr)
    beta = np.linalg.solve(A, rhs)
    route_gap = np.max(
        np.abs(beta - b_county[:2]) / np.maximum(np.abs(beta), 1e-300))
    if route_gap >= locked.RTOL_COEF:
        raise SystemExit(
            f"FAIL: FWL and multi-regressor routes disagree at ell={ell}, "
            f"relative gap={route_gap:.2e}")

    rhs_shock = exposure_moments(inp, w * yr).sum(axis=1)
    A_shock = np.column_stack([
        exposure_moments(inp, w * Xr[:, q]).sum(axis=1)
        for q in range(2)
    ])
    equiv_gap = max(
        np.max(np.abs(rhs_shock - rhs) / np.maximum(np.abs(rhs), 1e-300)),
        np.max(np.abs(A_shock - A) / np.maximum(np.abs(A), 1e-300)),
    )
    if equiv_gap >= locked.RTOL_COEF:
        raise SystemExit(
            f"FAIL: county and shock-level moments disagree at ell={ell}, "
            f"relative gap={equiv_gap:.2e}")

    residual = yr - Xr @ beta
    category_scores = exposure_moments(inp, w * residual).T
    live = np.any(np.abs(category_scores) > 0, axis=1)
    category_scores = category_scores[live]
    G = category_scores.shape[0]
    invA = np.linalg.inv(A)
    V = (G / max(G - 1, 1)) * (
        invA @ (category_scores.T @ category_scores) @ invA)
    sum_direction = np.ones(2)
    equivalent_direction = np.full(2, 0.5)
    sum_beta = float(sum_direction @ beta)
    equivalent_beta = float(equivalent_direction @ beta)
    equivalent_var = float(equivalent_direction @ V @ equivalent_direction)
    equivalent_se = float(np.sqrt(max(equivalent_var, 0.0)))
    influence = category_scores @ invA @ equivalent_direction

    return {
        "variant": variant, "ell": ell,
        "beta_current": float(beta[0]),
        "beta_prior": float(beta[1]),
        "beta_sum": sum_beta,
        "beta_equivalent": equivalent_beta,
        "se_current_exposure": float(np.sqrt(max(V[0, 0], 0.0))),
        "se_prior_exposure": float(np.sqrt(max(V[1, 1], 0.0))),
        "cov_exposure": float(V[0, 1]),
        "se_equivalent_exposure": equivalent_se,
        "se_current_county": float(se_county[0]),
        "se_prior_county": float(se_county[1]),
        "t_equivalent_exposure": equivalent_beta / equivalent_se,
        "n": len(yr),
        "n_controls": len(controls),
        "n_shock_clusters": G,
        "route_gap": float(route_gap),
        "equiv_gap": float(equiv_gap),
        "joint_influence": influence,
    }


def joint_pretest(fits, variant):
    """Joint Wald test of the three two-year-equivalent lead coefficients."""
    selected = sorted(
        (fit for fit in fits
         if fit["variant"] == variant and fit["ell"] in PRE_TIMES),
        key=lambda fit: fit["ell"])
    beta = np.array([fit["beta_equivalent"] for fit in selected])
    influence = np.column_stack(
        [fit["joint_influence"] for fit in selected])
    G = influence.shape[0]
    V = (G / max(G - 1, 1)) * (influence.T @ influence)
    k = len(beta)
    F = float(beta @ np.linalg.solve(V, beta) / k)
    return {
        "F": F, "p": float(stats.f.sf(F, k, G - 1)),
        "df_num": k, "df_den": G - 1,
    }


def write_outputs(inp, fits, pretests):
    """Write component estimates, joint responses, and diagnostics."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fit in fits:
        start, end = outcome_offsets(fit["ell"])
        rows.append({
            "variant": fit["variant"],
            "variant_label": LABELS[fit["variant"]],
            "event_time": fit["ell"],
            "period": ("pre" if fit["ell"] < 0 else
                       "impact" if fit["ell"] == 0 else "post"),
            "outcome_start_offset": start,
            "outcome_end_offset": end,
            "outcome_window_years": end - start,
            "coef_current_innovation": fit["beta_current"] * SCALE,
            "coef_prior_innovation": fit["beta_prior"] * SCALE,
            "coef_sum_one_unit_each": fit["beta_sum"] * SCALE,
            "coef_two_year_equivalent": fit["beta_equivalent"] * SCALE,
            "se_current_exposure": fit["se_current_exposure"] * SCALE,
            "se_prior_exposure": fit["se_prior_exposure"] * SCALE,
            "cov_exposure_scaled": fit["cov_exposure"] * SCALE ** 2,
            "se_two_year_equivalent_exposure": (
                fit["se_equivalent_exposure"] * SCALE),
            "t_two_year_equivalent_exposure": (
                fit["t_equivalent_exposure"]),
            "se_current_county": fit["se_current_county"] * SCALE,
            "se_prior_county": fit["se_prior_county"] * SCALE,
            "n": fit["n"],
            "n_controls": fit["n_controls"],
            "n_shock_clusters": fit["n_shock_clusters"],
            "year_lo": YEAR_LO,
            "year_hi": YEAR_HI,
            "route_gap": fit["route_gap"],
            "equiv_gap": fit["equiv_gap"],
        })
    pd.DataFrame(rows).to_csv(
        OUT / "lp2_annual_innovations.csv", index=False)
    summaries = []
    for variant, lags in VARIANTS.items():
        pretest = pretests[variant]
        summaries.append({
            "variant": variant,
            "variant_label": LABELS[variant],
            "estimand": (
                "one-unit total innovation split equally across two years"),
            "two_year_equivalent_coefficient": (
                "0.5*(beta_current + beta_prior)"),
            "weight": "equal county",
            "fixed_effects": "division-by-year",
            "shock_controls": ",".join(f"z1_t-{lag}" for lag in lags),
            "share_sum_controls": "s0_t,s0_t-1",
            "outcome_lags_impact_post": 3,
            "cluster": "FSC category",
            "joint_pre_F": pretest["F"],
            "joint_pre_p": pretest["p"],
            "joint_pre_df_num": pretest["df_num"],
            "joint_pre_df_den": pretest["df_den"],
            "n": N_EXPECT,
            "n_counties": locked.N_COUNTIES,
            "year_lo": YEAR_LO,
            "year_hi": YEAR_HI,
            "instrument_reconstruction_gap": inp["reconstruction_gap"],
        })
    pd.DataFrame(summaries).to_csv(
        OUT / "lp2_annual_innovations_summary.csv", index=False)


def write_figure(fits):
    """Write the covariance-aware joint-response plot and LaTeX fragment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font_path = Path("/System/Library/Fonts/Supplemental/GillSans.ttc")
    base_font = (fm.FontProperties(fname=font_path).get_name()
                 if font_path.exists() else "DejaVu Sans")
    navy, maroon = "#1A476F", "#90353B"
    dark, grid, reference = "#1A1A1A", "#E6E6E6", "#AAAAAA"
    plt.rcParams.update({
        "font.family": base_font,
        "font.size": 10,
        "figure.figsize": (8, 4.5),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": dark,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": grid,
        "grid.linewidth": 0.4,
        "xtick.color": dark,
        "ytick.color": dark,
        "text.color": dark,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots()
    ax.axvspan(-2, 0, color=grid, alpha=0.75, zorder=0)
    ax.axhline(0, color=reference, linewidth=0.8, zorder=1)
    for variant, color, marker, offset in (
            ("lags_2_4", navy, "o", -0.07),
            ("lags_2_8", maroon, "s", 0.07)):
        ordered = sorted(
            (fit for fit in fits if fit["variant"] == variant),
            key=lambda fit: fit["ell"])
        x = np.array([fit["ell"] for fit in ordered], dtype=float)
        beta = np.array(
            [fit["beta_equivalent"] for fit in ordered]) * SCALE
        se = np.array(
            [fit["se_equivalent_exposure"] for fit in ordered]) * SCALE
        ax.errorbar(
            x + offset, beta, yerr=Z95 * se, fmt=marker,
            linestyle="none", color=color, ecolor=color, capsize=3,
            elinewidth=1.0, markersize=5, label=LABELS[variant], zorder=3)
    ax.set_xticks(range(-5, 5))
    ax.set_xlabel("Horizon relative to the end of the two-year episode")
    ax.set_ylabel(
        r"$\frac{1}{2}(\beta_t+\beta_{t-1})\times10^5$, log employment growth")
    ax.legend(
        frameon=False, bbox_to_anchor=(0.5, -0.15), loc="upper center",
        ncol=2)
    fig.subplots_adjust(bottom=0.24)
    pdf = OUT / "Figure_lp2_annual_innovations.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    plt.close(fig)

    fragment = rf"""% Produced by step_1/code/pipeline/53_two_annual_innovations_lp.py
% from Results/lp_2yr_annual_innovations/lp2_annual_innovations.csv.
% Do not edit.
\noindent\begin{{minipage}}{{\textwidth}}\centering
\textbf{{Figure. Two-year response from consecutive annual innovations}}\\[0.2em]
{{\small County-level, equal-weighted local projections}}\\[0.6em]
\includegraphics[width=\textwidth]{{{pdf.name}}}
\end{{minipage}}
\par\smallskip
\noindent\begin{{minipage}}{{\textwidth}}\footnotesize\raggedright
\textit{{Notes:}} This figure normalizes a one-unit total two-year innovation
as one-half unit in each year of the episode. Every marker is one-half the sum
of the coefficients on $z^{{(1)}}_{{c,t}}$ and $z^{{(1)}}_{{c,t-1}}$ from one
regression. The 95\% confidence intervals use
the full covariance between those coefficients and exposure-robust inference
clustered on the FSC category. The gray strip marks the episode $[t-2,t]$.
Both specifications use equal county weights, division-by-year fixed effects,
the focal share-sum controls, and the same 1987--2012 sample. The baseline
controls for annual shocks dated $t-2$ through $t-4$; the expanded
specification controls through $t-8$. Impact and post regressions also include
three annual outcome lags ending before the episode. Leads end at $t-2$,
impact covers $[t-2,t]$, and post horizons cover $[t,t+h]$. This equal-split
normalization is directly comparable in scale, but not identical, to the
official rolling $h=2$ composition shock.
\par
\end{{minipage}}
"""
    (OUT / "Figure_lp2_annual_innovations.tex").write_text(
        fragment, encoding="utf-8")


def main():
    start = time.time()
    print("== stage 53: two consecutive annual innovations")
    print(f"   common sample [{YEAR_LO},{YEAR_HI}], equal county weights")
    inp = load_inputs()
    print(f"   annual-instrument reconstruction gap "
          f"{inp['reconstruction_gap']:.2e}")
    print(f"   {len(inp['d']):,} county-years, "
          f"{inp['d']['ccg'].nunique():,} counties")

    fits, pretests = [], {}
    for variant in VARIANTS:
        print(f"   {LABELS[variant]}")
        for ell in EVENT_TIMES:
            fit = fit_horizon(inp, variant, ell)
            fits.append(fit)
            print(
                f"      ell={ell:+d}: "
                f"current={fit['beta_current'] * SCALE:+.4f}, "
                f"prior={fit['beta_prior'] * SCALE:+.4f}, "
                f"h2_equiv={fit['beta_equivalent'] * SCALE:+.4f}, "
                f"se={fit['se_equivalent_exposure'] * SCALE:.4f}, "
                f"t={fit['t_equivalent_exposure']:+.2f}")
        pretests[variant] = joint_pretest(fits, variant)
        pretest = pretests[variant]
        print(
            f"      joint pre-shock test: F={pretest['F']:.3f}, "
            f"p={pretest['p']:.4f}, "
            f"F({pretest['df_num']},{pretest['df_den']})")

    write_outputs(inp, fits, pretests)
    write_figure(fits)
    print(f"   wrote {OUT / 'lp2_annual_innovations.csv'}")
    print(f"   wrote {OUT / 'lp2_annual_innovations_summary.csv'}")
    print(f"   wrote {OUT / 'Figure_lp2_annual_innovations.pdf'}")
    print(f"   wrote {OUT / 'Figure_lp2_annual_innovations.tex'}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
