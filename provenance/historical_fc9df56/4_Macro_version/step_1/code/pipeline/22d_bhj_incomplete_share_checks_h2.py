#!/usr/bin/env python3
"""Diagnostics for rolling exposures and incomplete shares at h=2.

The county instrument remains the locked population-scaled composition object,

    z_ct = sum_k Cbar_ckt q_kt / P_c,t-2.

The diagnostic compares the uncorrected specification with BHJ period-specific
share-sum controls. It also reports fixed 1969-1973 exposures, the two parities
of non-overlapping target years, and national category shocks residualized on
four lags plus category and year effects. Equal-county and population weights
are reported separately.

Outputs in Results/pipeline_diagnostics/:
    bhj_incomplete_share_checks_h2.csv
    bhj_shock_predictability_h2.csv
    Table_bhj_incomplete_share_checks_h2.tex
    Table_bhj_shock_predictability_h2.tex
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402
import store  # noqa: E402


H = 2
LAGS = (1, 2, 3, 4)
FIXED_YEARS = tuple(range(1969, 1974))
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"


def load_bhj():
    """Load the canonical BHJ aggregation and inference functions."""
    spec = importlib.util.spec_from_file_location(
        "bhj_assumption1", PIPELINE / "22_bhj_assumption1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def county_weights(d: pd.DataFrame, scheme: str) -> np.ndarray:
    """Return equal-county or fixed-1975 population weights."""
    if scheme == "equal":
        return np.ones(len(d), np.float64)
    if scheme != "population":
        raise ValueError(scheme)
    panel = store.load_county_panel()
    p0 = (panel.loc[panel["year"].eq(locked.POP_BASE_YEAR), ["ccg", "P"]]
          .drop_duplicates("ccg").set_index("ccg")["P"])
    weights = d["ccg"].map(p0).to_numpy(np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise SystemExit("FAIL: fixed 1975 population weights are not positive")
    return weights / weights.mean()


def instrument_columns(base: dict, cbar: dict, gk: np.ndarray):
    """Build z and its population-scaled exposure sum on base county rows."""
    rr, cc, pb = base["rr"], base["cc"], base["pb"]
    z = np.full(len(rr), np.nan, np.float64)
    s0 = np.full(len(rr), np.nan, np.float64)
    for j in sorted(cbar):
        m = cc == j
        if not m.any():
            continue
        tcol = base["cy"].index(base["years"][j])
        q = gk[:, tcol]
        live = np.isfinite(q)
        if not live.any():
            continue
        exposure = cbar[j][rr[m], :][:, live].astype(np.float64)
        z[m] = exposure @ q[live] / pb[m]
        s0[m] = exposure.sum(axis=1) / pb[m]
    return z, s0


def make_input(base: dict, bhj, scheme: str, cbar: dict | None = None,
               gk: np.ndarray | None = None,
               years_keep: set[int] | None = None) -> dict:
    """Build one internally consistent county and shock-level input object."""
    cbar = base["cbar"] if cbar is None else cbar
    gk = base["gk"] if gk is None else gk
    z, s0 = instrument_columns(base, cbar, gk)
    d = base["d"].copy()
    d["z"] = z
    d["s0"] = s0
    d["y_log"] = base["y_log"]
    keep = np.isfinite(z) & np.isfinite(s0)
    if years_keep is not None:
        keep &= d["year"].isin(years_keep).to_numpy()
    if not keep.any():
        raise SystemExit(f"FAIL: {scheme} diagnostic sample is empty")
    d = d.loc[keep].reset_index(drop=True)

    units = base["units"]
    rowpos = {u: i for i, u in enumerate(units)}
    colpos = {y: j for j, y in enumerate(base["years"])}
    rr = d["ccg"].map(rowpos).to_numpy(np.int64)
    cc = d["year"].map(colpos).to_numpy(np.int64)
    pb = d["P_b"].to_numpy(np.float64)
    w = county_weights(d, scheme)
    fe = pd.factorize(
        d["division"].astype(str) + "_" + d["year"].astype(str)
    )[0].astype(np.int64)
    nofe = np.zeros(len(d), np.int64)
    cl_county = pd.factorize(d["ccg"])[0].astype(np.int64)
    xw = d["x"].to_numpy(np.float64)
    matrix = bhj.op.w22.demean(
        np.column_stack([
            d["y_log"].to_numpy(np.float64), xw,
            d["z"].to_numpy(np.float64), d["s0"].to_numpy(np.float64),
        ]),
        fe,
        nofe,
        w,
    )
    inp = dict(base)
    inp.update(
        d=d, cbar=cbar, gk=gk, w=w, fe=fe, nofe=nofe,
        cl_county=cl_county, xw=xw, yt=matrix[:, 0], xt=matrix[:, 1],
        zt=matrix[:, 2], st=matrix[:, 3], rr=rr, cc=cc, pb=pb,
    )
    return inp


def county_statistics(inp: dict, fit: dict, bhj) -> dict:
    """County-clustered IV, reduced-form, and first-stage statistics."""
    w = inp["w"]
    y, x, z = fit["yr"], fit["xr"], fit["zr"]
    cl = inp["cl_county"]
    n = len(y)
    control_rank = 0
    if fit["Cm"].shape[1]:
        control_rank = int(np.linalg.matrix_rank(
            fit["Cm"] * np.sqrt(w)[:, None]
        ))
    k = bhj.op.t28.ssc_k(inp["fe"], inp["nofe"], cl) + control_rank
    beta = fit["beta"]
    iv_se = bhj.op.t28.crv1(
        z, x, y - beta * x, w, cl, n, k, corrected=True
    )
    zz = float(z @ (w * z))
    zx = float(z @ (w * x))
    zy = float(z @ (w * y))
    xx = float(x @ (w * x))
    if min(abs(zz), abs(zx), abs(xx)) < 1e-300:
        raise SystemExit("FAIL: degenerate county IV moment")
    fs = zx / zz
    fs_se = bhj.op.t28.crv1(
        z, z, x - fs * z, w, cl, n, k, corrected=True
    )
    rf = zy / zz
    rf_se = bhj.op.t28.crv1(
        z, z, y - rf * z, w, cl, n, k, corrected=True
    )
    return {
        "county_iv_se": iv_se,
        "county_first_stage_F": (fs / fs_se) ** 2,
        "county_reduced_form": rf,
        "county_reduced_form_se": rf_se,
        "first_stage_partial_r2": (zx * zx) / (zz * xx),
        "control_rank": control_rank,
    }


def estimate(inp: dict, bhj, rung, label: str, scheme: str) -> dict:
    """Estimate one specification with county and exposure-robust inference."""
    sf = bhj.shock_frame(inp)
    fit = bhj.rung_fit(inp, sf, rung)
    if fit["gap"] > 1e-9:
        raise SystemExit(
            f"FAIL: {label}, {scheme}, county/shock coefficient gap "
            f"{fit['gap']:.3e}"
        )
    county = county_statistics(inp, fit, bhj)
    bhj_se, categories = bhj.exposure_robust(
        fit, sf, inp["cats"], "category"
    )
    bhj_F = bhj.first_stage_F(fit, sf, inp["cats"], "category")
    rf_b, rf_v, _ = bhj.shock_ratio(
        sf["gv"], fit["ny"], fit["nz"],
        bhj.cluster_index(sf, inp["cats"], "category"),
    )
    within_z_variance = float(inp["w"] @ (inp["zt"] * inp["zt"]))
    residual_z_variance = float(inp["w"] @ (fit["zr"] * fit["zr"]))
    variance_removed = (
        1.0 - residual_z_variance / within_z_variance
        if within_z_variance > 0 else np.nan
    )
    return {
        "h": H,
        "county_weight": scheme,
        "specification": label,
        "share_sum_control": "year interactions" if rung == 1 else "none",
        "beta": fit["beta"],
        "county_iv_se": county["county_iv_se"],
        "bhj_iv_se": bhj_se,
        "county_first_stage_F": county["county_first_stage_F"],
        "bhj_first_stage_F": bhj_F,
        "county_reduced_form": county["county_reduced_form"],
        "county_reduced_form_se": county["county_reduced_form_se"],
        "bhj_reduced_form": rf_b,
        "bhj_reduced_form_se": np.sqrt(rf_v),
        "first_stage_partial_r2": county["first_stage_partial_r2"],
        "instrument_variance_removed": variance_removed,
        "control_rank": county["control_rank"],
        "county_years": len(inp["d"]),
        "counties": int(inp["d"]["ccg"].nunique()),
        "first_year": int(inp["d"]["year"].min()),
        "last_year": int(inp["d"]["year"].max()),
        "shock_cells": len(sf["gv"]),
        "category_clusters": categories,
        "equivalence_gap": fit["gap"],
    }


def clustered_wls(y: np.ndarray, X: np.ndarray, w: np.ndarray,
                  category: np.ndarray, year: np.ndarray, bhj):
    """Two-way fixed-effect WLS with category-clustered CRV1 inference."""
    matrix = bhj.op.w22.demean(
        np.column_stack([y, X]), category, year, w
    )
    yd, Xd = matrix[:, 0], matrix[:, 1:]
    XW = Xd * w[:, None]
    bread0 = Xd.T @ XW
    if np.linalg.matrix_rank(bread0) != X.shape[1]:
        raise SystemExit("FAIL: shock-lag design is rank deficient")
    bread = np.linalg.inv(bread0)
    beta = bread @ (XW.T @ yd)
    residual = yd - Xd @ beta
    ncl = int(category.max()) + 1
    scores = np.column_stack([
        np.bincount(category, weights=XW[:, j] * residual, minlength=ncl)
        for j in range(X.shape[1])
    ])
    n, groups = len(y), int(np.unique(category).size)
    k = bhj.op.t28.ssc_k(category, year, category) + X.shape[1] - 1
    correction = (groups / max(groups - 1, 1)) * (
        (n - 1) / max(n - k, 1)
    )
    covariance = correction * bread @ (scores.T @ scores) @ bread
    se = np.sqrt(np.diag(covariance))
    sst = float(w @ (yd * yd))
    r2 = 1.0 - float(w @ (residual * residual)) / sst
    joint_F = float(beta @ np.linalg.solve(covariance, beta) / len(beta))
    return beta, se, covariance, residual, r2, joint_F


def predictability(base: dict, bhj, scheme: str):
    """Estimate category-shock persistence and return lag innovations."""
    inp = make_input(base, bhj, scheme)
    sf = bhj.shock_frame(inp)
    k = sf["keys"][:, 0].astype(np.int64)
    year = np.array([inp["years"][j] for j in sf["keys"][:, 1]], np.int64)
    y = sf["gv"].astype(np.float64)
    X = np.column_stack([
        np.array([
            inp["gk"][kk, inp["cy"].index(tt - lag)]
            if tt - lag in inp["cy"] else np.nan
            for kk, tt in zip(k, year)
        ], np.float64)
        for lag in LAGS
    ])
    good = np.isfinite(y) & np.isfinite(X).all(axis=1) & (sf["sv"] > 0)
    if not good.any():
        raise SystemExit(f"FAIL: no lag-complete shocks for {scheme}")
    kg = pd.factorize(k[good])[0].astype(np.int64)
    yg = pd.factorize(year[good])[0].astype(np.int64)
    beta, se, covariance, residual, r2, joint_F = clustered_wls(
        y[good], X[good], sf["sv"][good], kg, yg, bhj
    )
    df = max(int(np.unique(kg).size) - 1, 1)
    rows = []
    for pos, lag in enumerate(LAGS):
        tstat = beta[pos] / se[pos]
        rows.append({
            "h": H,
            "county_weight": scheme,
            "lag": lag,
            "coefficient": beta[pos],
            "se": se[pos],
            "t": tstat,
            "p": 2.0 * stats.t.sf(abs(tstat), df),
            "joint_F_all_lags": joint_F,
            "within_r2": r2,
            "shock_cells": int(good.sum()),
            "category_clusters": int(np.unique(kg).size),
            "first_year": int(year[good].min()),
            "last_year": int(year[good].max()),
        })

    innovations = np.full_like(inp["gk"], np.nan, np.float64)
    for kk, tt, value in zip(k[good], year[good], residual):
        innovations[kk, inp["cy"].index(int(tt))] = value
    live_years = set(int(v) for v in np.unique(year[good]))
    return pd.DataFrame(rows), innovations, live_years


def fixed_exposures(base: dict) -> dict:
    """Use the same 1969-1973 category-dollar exposure in every target year."""
    a = base["cy"].index(FIXED_YEARS[0])
    b = base["cy"].index(FIXED_YEARS[-1])
    fixed = base["cbar"][min(base["cbar"])].copy()
    _, _, _, arr, _ = store.load_tensor(locked.RULE)
    fixed[:, :] = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
    return {j: fixed for j in base["cbar"]}


def run_specs(base: dict, bhj, scheme: str,
              innovations: np.ndarray, innovation_years: set[int]):
    """Run the specification battery for one county weighting scheme."""
    rows = []
    current = make_input(base, bhj, scheme)
    rows.append(estimate(current, bhj, 0, "Current exposure", scheme))
    rows.append(estimate(
        current, bhj, 1, "Current exposure, share-sum by year", scheme
    ))

    fixed = make_input(
        base, bhj, scheme, cbar=fixed_exposures(base), gk=base["gk"]
    )
    rows.append(estimate(
        fixed, bhj, 1, "Fixed 1969-1973 exposure, share-sum by year", scheme
    ))

    all_years = set(int(v) for v in current["d"]["year"].unique())
    for parity, label in ((0, "Even target years"), (1, "Odd target years")):
        years = {year for year in all_years if year % 2 == parity}
        subset = make_input(base, bhj, scheme, years_keep=years)
        rows.append(estimate(
            subset, bhj, 1, f"{label}, share-sum by year", scheme
        ))

    reference = make_input(
        base, bhj, scheme, years_keep=innovation_years
    )
    rows.append(estimate(
        reference, bhj, 1,
        "Current shocks on lag-complete sample, share-sum by year", scheme
    ))
    innovation = make_input(
        base, bhj, scheme, gk=innovations, years_keep=innovation_years
    )
    rows.append(estimate(
        innovation, bhj, 1,
        "Lag-residualized shocks, share-sum by year", scheme
    ))
    return rows


def write_spec_table(result: pd.DataFrame) -> None:
    """Write the h=2 robustness table."""
    lines = []
    for scheme, heading in (
        ("equal", "Panel A. Equal county weights"),
        ("population", "Panel B. Population weights"),
    ):
        lines.append(
            rf"\multicolumn{{10}}{{l}}{{\textit{{{heading}}}}} \\"
        )
        for row in result[result["county_weight"] == scheme].itertuples():
            lines.append(
                f"\\hspace{{1em}}{row.specification} "
                f"& {row.county_reduced_form * 1e5:.3f} "
                f"& ({row.county_reduced_form_se * 1e5:.3f}) "
                f"& ({row.bhj_reduced_form_se * 1e5:.3f}) "
                f"& {row.beta * 1e5:.3f} "
                f"& ({row.bhj_iv_se * 1e5:.3f}) "
                f"& {row.county_first_stage_F:.2f} "
                f"& {row.bhj_first_stage_F:.2f} "
                f"& {100 * row.instrument_variance_removed:.1f}\\% "
                f"& {int(row.county_years):,} \\\\"
            )
        if scheme == "equal":
            lines.append(r"\addlinespace")
    target = OUT / "Table_bhj_incomplete_share_checks_h2.tex"
    target.write_text(
        """\\begin{table}[!htbp]
\\caption{Incomplete-Share and Repeated-Shock Checks at the Two-Year Horizon}
\\label{tab:bhj_incomplete_share_checks_h2}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrrrrrrrrr}
\\toprule
Specification & RF$\\times10^5$ & County s.e. & BHJ s.e.
& 2SLS$\\times10^5$ & BHJ s.e. & County $F$ & BHJ $F$
& $z$ var. removed & $N$ \\\\
\\midrule
"""
        + "\n".join(lines)
        + """
\\bottomrule
\\end{tabular}}
\\par}
\\medskip
{\\scriptsize \\noindent\\textit{Notes:} The dependent variable is two-year
log employment growth. The endogenous regressor is the county procurement
dollar change divided by population at $t-2$. The instrument is the locked
top-800 four-character Federal Supply Classification composition instrument,
with five-year rolling category-dollar exposures divided by population at
$t-2$ and division-by-year fixed effects. The treatment is not winsorized.
The share-sum control interacts total predetermined procurement per person with
each target-year indicator. Fixed exposures use average annual category dollars
from 1969 through 1973 while retaining the population denominator at $t-2$.
Lag-residualized national category shocks are residuals from a weighted
regression on four lags, category effects, and year effects. County standard
errors and first-stage statistics cluster by county. BHJ exposure-robust
statistics cluster by procurement category. Population weights use county
population fixed in 1975 and constant across target years.\\par}
\\end{table}
""",
        encoding="utf-8",
    )
    print(f"wrote {target}")


def write_predictability_table(result: pd.DataFrame) -> None:
    """Write the national category-shock lag regression."""
    merged = result.pivot(
        index="lag", columns="county_weight", values=["coefficient", "se", "p"]
    )
    lines = []
    for lag in LAGS:
        lines.append(
            f"$q_{{k,t-{lag}}}$ "
            f"& {merged.loc[lag, ('coefficient', 'equal')]:.4f} "
            f"& ({merged.loc[lag, ('se', 'equal')]:.4f}) "
            f"& {merged.loc[lag, ('p', 'equal')]:.3f} "
            f"& {merged.loc[lag, ('coefficient', 'population')]:.4f} "
            f"& ({merged.loc[lag, ('se', 'population')]:.4f}) "
            f"& {merged.loc[lag, ('p', 'population')]:.3f} \\\\"
        )
    first = result.groupby("county_weight", sort=False).first()
    lines.extend([
        r"\midrule",
        f"Joint $F$, four lags & {first.loc['equal', 'joint_F_all_lags']:.2f} "
        f"& & & {first.loc['population', 'joint_F_all_lags']:.2f} & & \\\\ ",
        f"Shock-level observations & {int(first.loc['equal', 'shock_cells']):,} "
        f"& & & {int(first.loc['population', 'shock_cells']):,} & & \\\\ ",
        f"Category clusters & {int(first.loc['equal', 'category_clusters']):,} "
        f"& & & {int(first.loc['population', 'category_clusters']):,} & & \\\\ ",
    ])
    target = OUT / "Table_bhj_shock_predictability_h2.tex"
    target.write_text(
        """\\begin{table}[!htbp]
\\caption{Predictability of Two-Year National Category Shocks}
\\label{tab:bhj_shock_predictability_h2}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrrrrrr}
\\toprule
& \\multicolumn{3}{c}{Equal county weights}
& \\multicolumn{3}{c}{Population weights} \\\\
\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}
Regressor & Coef. & (s.e.) & $p$ & Coef. & (s.e.) & $p$ \\\\
\\midrule
"""
        + "\n".join(lines)
        + """
\\bottomrule
\\end{tabular}}
\\par}
\\medskip
{\\scriptsize \\noindent\\textit{Notes:} The dependent variable is the
national category shock $q_{kt}$. The regression includes four lags jointly,
category effects, and year effects. Observations are category-year shocks and
weights are their BHJ importance weights under the indicated county weighting
scheme. Standard errors cluster by four-character Federal Supply Classification
category.\\par}
\\end{table}
""",
        encoding="utf-8",
    )
    print(f"wrote {target}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bhj = load_bhj()
    base = bhj.build_inputs(H)
    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("E",))
    if not np.array_equal(units, base["units"]):
        raise SystemExit("FAIL: employment and exposure county orders differ")
    employment = pivots["E"].reindex(
        index=units, columns=base["years"]
    ).to_numpy(np.float64)
    employment_t = employment[base["rr"], base["cc"]]
    employment_b = base["d"]["E_b"].to_numpy(np.float64)
    if (employment_t <= 0).any() or (employment_b <= 0).any():
        raise SystemExit("FAIL: log employment growth has a non-positive level")
    base["y_log"] = np.log(employment_t) - np.log(employment_b)

    rebuilt_z, rebuilt_s0 = instrument_columns(base, base["cbar"], base["gk"])
    if not np.allclose(
        rebuilt_z, base["d"]["z"].to_numpy(np.float64),
        rtol=locked.RTOL_VEC, atol=locked.ATOL_VEC, equal_nan=True,
    ):
        raise SystemExit("FAIL: rebuilt baseline instrument does not tie out")
    if not np.allclose(
        rebuilt_s0, base["d"]["s0"].to_numpy(np.float64),
        rtol=locked.RTOL_VEC, atol=locked.ATOL_VEC, equal_nan=True,
    ):
        raise SystemExit("FAIL: rebuilt share sum does not tie out")

    predict_rows = []
    spec_rows = []
    for scheme in ("equal", "population"):
        prediction, innovations, live_years = predictability(base, bhj, scheme)
        predict_rows.append(prediction)
        spec_rows.extend(run_specs(
            base, bhj, scheme, innovations, live_years
        ))

    predict_result = pd.concat(predict_rows, ignore_index=True)
    spec_result = pd.DataFrame(spec_rows)
    manifest_path = store.PIPE / "_golden" / "active_baseline.json"
    if not manifest_path.exists():
        raise SystemExit(f"FAIL: active baseline manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("baseline_id") != "verified_production_geography_v1":
        raise SystemExit(
            "FAIL: BHJ diagnostics require verified_production_geography_v1"
        )
    spec_csv = OUT / "bhj_incomplete_share_checks_h2.csv"
    predict_csv = OUT / "bhj_shock_predictability_h2.csv"
    spec_result.to_csv(spec_csv, index=False)
    predict_result.to_csv(predict_csv, index=False)
    write_spec_table(spec_result)
    write_predictability_table(predict_result)
    print(f"wrote {spec_csv}")
    print(f"wrote {predict_csv}")
    print(spec_result[[
        "county_weight", "specification", "beta", "county_iv_se", "bhj_iv_se",
        "county_first_stage_F", "bhj_first_stage_F", "county_years",
    ]].to_string(index=False))
    print(predict_result.to_string(index=False))


if __name__ == "__main__":
    main()
