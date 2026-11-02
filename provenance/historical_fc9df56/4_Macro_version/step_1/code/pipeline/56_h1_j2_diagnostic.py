"""Diagnose the h=1 second annual pre-period coefficient.

The target is the coefficient on z1_t for midpoint employment growth over
[t-3,t-2], the j=2 disjoint annual pre-period increment in producer 50. The
script holds p=4, rung 0', division-by-year fixed effects, and the share-sum
control fixed while varying the sample and county weights. On the matched
1987-2012 equal-weight cell, it also decomposes the coefficient by shock year
and FSC category.

Writes Results/lp_2yr_annual_innovations/:

  h1_j2_sensitivity.csv
  h1_j2_year_contributions.csv
  h1_j2_category_contributions.csv
  Figure_h1_j2_year_contributions.pdf
  Figure_h1_j2_year_contributions.tex
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
OUT = ROOT / "Results" / "lp_2yr_annual_innovations"
P_LAG = 4
SCALE = 1e5
Z95 = 1.959963984540054
SAMPLES = {
    "longest": (1983, 2012),
    "matched": (1987, 2012),
    "matched_early": (1987, 1999),
    "matched_late": (2000, 2012),
    "pre_seam": (1983, 2006),
    "post_seam": (2007, 2012),
}
WEIGHTS = ("equal", "division-EW")
np.random.seed(0)


def load_producer50():
    """Load producer 50 without relying on an importable numeric module name."""
    path = PIPELINE / "50_lp_dynamics.py"
    spec = importlib.util.spec_from_file_location("lp50", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"FAIL: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_j2(lp50, stores, label, year_lo, year_hi, weight):
    """Estimate j=2 and return exact shock-level contributions."""
    inp = lp50.build_window(
        stores, year_lo, year_hi, [("impact", 0)],
        p=P_LAG, weight=weight)
    sf = lp50.shock_frame(inp)
    controls = lp50.control_names("lead", "0'", P_LAG)
    dm, w = inp["dm"], inp["w"]
    C = np.column_stack([dm[name] for name in controls])
    yr, zr = lp50.fwl(C, w, dm["gl2"], dm["z1"])
    denominator = float((w * zr) @ zr)
    beta = float((w * zr) @ yr) / denominator

    design = np.column_stack([dm["z1"]] + [dm[name] for name in controls])
    beta_county, se_county, _ = lp50.wls_fe(
        dm["gl2"], design, w, inp["fe"], inp["nofe"], inp["cl_county"])
    route_gap = abs(beta - beta_county[0]) / max(abs(beta), 1e-300)
    if route_gap >= lp50.locked.RTOL_COEF:
        raise SystemExit(
            f"FAIL: county routes disagree for {label}/{weight}, "
            f"relative gap={route_gap:.2e}")

    ny, nz = lp50.aggregate(sf, yr), lp50.aggregate(sf, zr)
    g = sf["gv"]
    shock_denominator = float(g @ nz)
    beta_shock = float(g @ ny) / shock_denominator
    equivalence_gap = abs(beta_shock - beta) / max(abs(beta), 1e-300)
    if equivalence_gap >= lp50.locked.RTOL_COEF:
        raise SystemExit(
            f"FAIL: shock route disagrees for {label}/{weight}, "
            f"relative gap={equivalence_gap:.2e}")

    cluster = lp50.cluster_index(
        sf, inp["cats"], lp50.CONTINGENCY["cluster_level"])
    ncluster = int(cluster.max()) + 1
    score = g * (ny - beta * nz)
    sums = lp50.cluster_sums(score, cluster, ncluster)
    G = int(np.unique(cluster).size)
    se_exposure = float(
        np.sqrt((G / max(G - 1, 1)) * float(sums @ sums))
        / abs(shock_denominator))

    keys = sf["keys"]
    cell = pd.DataFrame({
        "category_index": keys[:, 0].astype(int),
        "year_index": keys[:, 1].astype(int),
        "coefficient_contribution": g * ny / shock_denominator * SCALE,
        "denominator_share": g * nz / shock_denominator,
        "score": score,
    })
    cell["year"] = cell["year_index"].map(
        dict(enumerate(inp["years"])))
    cell["category"] = cell["category_index"].map(
        dict(enumerate(inp["cats"])))
    if abs(cell["coefficient_contribution"].sum() - beta * SCALE) >= 1e-10:
        raise SystemExit(f"FAIL: contributions do not sum for {label}/{weight}")

    return {
        "sample": label,
        "year_lo": year_lo,
        "year_hi": year_hi,
        "weight": weight,
        "coefficient": beta * SCALE,
        "se_exposure": se_exposure * SCALE,
        "se_county": float(se_county[0]) * SCALE,
        "t_exposure": beta / se_exposure,
        "n": inp["n"],
        "n_counties": lp50.locked.N_COUNTIES,
        "n_shock_clusters": G,
        "route_gap": route_gap,
        "equivalence_gap": equivalence_gap,
        "cell": cell,
    }


def contribution_tables(cell):
    """Aggregate matched-cell coefficient contributions by year and category."""
    years = (cell.groupby("year", sort=True, as_index=False)
             .agg(coefficient_contribution=("coefficient_contribution", "sum"),
                  denominator_share=("denominator_share", "sum"),
                  score=("score", "sum")))
    years["absolute_coefficient_contribution"] = (
        years["coefficient_contribution"].abs())
    years["rank_by_absolute_contribution"] = (
        years["absolute_coefficient_contribution"]
        .rank(method="first", ascending=False).astype(int))

    categories = (cell.groupby("category", sort=True, as_index=False)
                  .agg(coefficient_contribution=(
                           "coefficient_contribution", "sum"),
                       denominator_share=("denominator_share", "sum"),
                       score=("score", "sum")))
    categories["absolute_coefficient_contribution"] = (
        categories["coefficient_contribution"].abs())
    categories["rank_by_absolute_contribution"] = (
        categories["absolute_coefficient_contribution"]
        .rank(method="first", ascending=False).astype(int))
    categories = categories.sort_values(
        "rank_by_absolute_contribution", kind="stable")
    return years, categories


def write_figure(years):
    """Plot the exact year contributions to the matched equal-weight estimate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font_path = Path("/System/Library/Fonts/Supplemental/GillSans.ttc")
    base_font = (fm.FontProperties(fname=font_path).get_name()
                 if font_path.exists() else "DejaVu Sans")
    navy, dark = "#1A476F", "#1A1A1A"
    grid, reference = "#E6E6E6", "#AAAAAA"
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
    ax.axhline(0, color=reference, linewidth=0.8, zorder=1)
    ax.bar(
        years["year"], years["coefficient_contribution"],
        color=navy, alpha=0.80, width=0.75, linewidth=0, zorder=2)
    ax.set_xlabel("Shock year")
    ax.set_ylabel(r"Contribution to $\hat\beta\times10^5$")
    ax.set_xticks(years["year"].iloc[::2])
    ax.tick_params(axis="x", labelrotation=45)
    fig.subplots_adjust(bottom=0.20)
    pdf = OUT / "Figure_h1_j2_year_contributions.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    plt.close(fig)

    fragment = rf"""% Produced by step_1/code/pipeline/56_h1_j2_diagnostic.py
% from Results/lp_2yr_annual_innovations/h1_j2_year_contributions.csv.
% Do not edit.
\noindent\begin{{minipage}}{{\textwidth}}\centering
\textbf{{Figure. Shock-year contributions to the second annual pre-period
coefficient}}\\[0.2em]
{{\small Equal county weights, 1987--2012}}\\[0.6em]
\includegraphics[width=\textwidth]{{{pdf.name}}}
\end{{minipage}}
\par\smallskip
\noindent\begin{{minipage}}{{\textwidth}}\footnotesize\raggedright
\textit{{Notes:}} Each bar is one shock year's exact contribution to the
coefficient on $z^{{(1)}}_{{c,t}}$ when the outcome is midpoint employment
growth over $[t-3,t-2]$. The bars sum to the estimated coefficient. The
regression uses equal county weights, division-by-year fixed effects, shock
lags one through four, and the share-sum level control.
\par
\end{{minipage}}
"""
    (OUT / "Figure_h1_j2_year_contributions.tex").write_text(
        fragment, encoding="utf-8")


def main():
    print("== stage 56: h=1 j=2 diagnostic")
    lp50 = load_producer50()
    stores = lp50.load_stores()
    fits = []
    matched_equal = None
    for label, (year_lo, year_hi) in SAMPLES.items():
        for weight in WEIGHTS:
            fit = fit_j2(lp50, stores, label, year_lo, year_hi, weight)
            fits.append(fit)
            if label == "matched" and weight == "equal":
                matched_equal = fit
            print(
                f"   {label:13s} {weight:11s}: "
                f"beta={fit['coefficient']:+.4f}, "
                f"se={fit['se_exposure']:.4f}, "
                f"t={fit['t_exposure']:+.2f}, n={fit['n']:,}")
    if matched_equal is None:
        raise SystemExit("FAIL: matched equal-weight cell was not estimated")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{key: value for key, value in fit.items() if key != "cell"}
                  for fit in fits]).to_csv(
        OUT / "h1_j2_sensitivity.csv", index=False)
    years, categories = contribution_tables(matched_equal["cell"])
    years.to_csv(OUT / "h1_j2_year_contributions.csv", index=False)
    categories.to_csv(OUT / "h1_j2_category_contributions.csv", index=False)
    write_figure(years)

    top_years = years.nsmallest(5, "rank_by_absolute_contribution")
    print("   five largest shock-year contributions, matched equal-weight cell")
    for row in top_years.itertuples(index=False):
        print(f"      {row.year}: {row.coefficient_contribution:+.4f}")
    print(f"   year contributions sum to "
          f"{years['coefficient_contribution'].sum():+.4f}")
    print(f"   wrote {OUT / 'h1_j2_sensitivity.csv'}")
    print(f"   wrote {OUT / 'h1_j2_year_contributions.csv'}")
    print(f"   wrote {OUT / 'h1_j2_category_contributions.csv'}")
    print(f"   wrote {OUT / 'Figure_h1_j2_year_contributions.pdf'}")
    print(f"   wrote {OUT / 'Figure_h1_j2_year_contributions.tex'}")


if __name__ == "__main__":
    main()
