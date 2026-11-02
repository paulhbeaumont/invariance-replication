#!/usr/bin/env python3
"""Decompose the h=2 instrument into share-sum and retained components by year.

The county specification uses two-year log employment growth, the locked
population-scaled composition instrument, division-by-year fixed effects, and
either equal county weights or county population fixed in 1975. The
share-sum-by-year component is the weighted projection of the instrument onto
the year-specific total predetermined procurement exposures. Its orthogonal
complement is the variation retained by the BHJ share-sum-by-year controls.

Outputs in Results/pipeline_diagnostics/:
    bhj_share_sum_year_contributions_h2.csv
    bhj_share_sum_leave_one_year_out_h2.csv
    bhj_share_sum_leave_five_years_out_h2.csv
    Table_bhj_share_sum_year_decomposition_h2.tex
    Figure_bhj_share_sum_year_decomposition_h2.{pdf,png,svg}
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402
import store  # noqa: E402


H = 2
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"

# Font and palette follow the project-wide graph standard.
FONT_PATH = "/System/Library/Fonts/Supplemental/GillSans.ttc"
BASE_FONT = (
    fm.FontProperties(fname=FONT_PATH).get_name()
    if Path(FONT_PATH).exists()
    else "DejaVu Sans"
)
STATA_NAVY = "#1A476F"
STATA_MAROON = "#90353B"
DARK = "#1A1A1A"
GRID_GRAY = "#E6E6E6"
REF_LINE_GRAY = "#AAAAAA"

plt.rcParams.update({
    "font.family": BASE_FONT,
    "font.size": 11,
    "figure.figsize": (10, 5.625),
    "figure.dpi": 170,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": DARK,
    "axes.linewidth": 0.6,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID_GRAY,
    "grid.linewidth": 0.4,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "text.color": DARK,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def load_module(path: Path, name: str):
    """Load one pipeline module without changing its command-line behavior."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_log_outcome(base: dict) -> None:
    """Attach the exact compact two-year log employment outcome."""
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


def ratio_statistics(z: np.ndarray, y: np.ndarray, x: np.ndarray,
                     w: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Return reduced form, first stage, and just-identified IV ratios."""
    if mask is None:
        mask = np.ones(len(z), dtype=bool)
    zm, ym, xm, wm = z[mask], y[mask], x[mask], w[mask]
    zz = float((wm * zm) @ zm)
    zy = float((wm * zm) @ ym)
    zx = float((wm * zm) @ xm)
    if min(abs(zz), abs(zx)) < 1e-300:
        raise SystemExit("FAIL: degenerate year-decomposition moment")
    return {
        "rf": zy / zz,
        "fs": zx / zz,
        "iv": zy / zx,
        "variance": zz,
        "rf_numerator": zy,
        "fs_numerator": zx,
    }


def decompose_scheme(base: dict, bhj, diagnostic, scheme: str):
    """Build annual contributions and leave-one-year-out estimates."""
    inp = diagnostic.make_input(base, bhj, scheme)
    sf = bhj.shock_frame(inp)
    fit = bhj.rung_fit(inp, sf, 1)

    w = inp["w"]
    y = inp["yt"]
    x = inp["xt"]
    z_full = inp["zt"]
    z_retained = fit["zr"]
    z_removed = z_full - z_retained
    years = inp["d"]["year"].to_numpy(np.int64)

    full = ratio_statistics(z_full, y, x, w)
    retained = ratio_statistics(z_retained, y, x, w)
    removed = ratio_statistics(z_removed, y, x, w)
    cross = float((w * z_retained) @ z_removed)
    if abs(cross) > 1e-10 * full["variance"]:
        raise SystemExit(
            f"FAIL: retained and removed components are not orthogonal for {scheme}"
        )

    component_map = {
        "retained by share-sum-by-year controls": (z_retained, retained),
        "removed by share-sum-by-year controls": (z_removed, removed),
    }
    contribution_rows = []
    for year in np.unique(years):
        mask = years == year
        for component, (z_component, totals) in component_map.items():
            variance = float((w[mask] * z_component[mask]) @ z_component[mask])
            rf_numerator = float((w[mask] * z_component[mask]) @ y[mask])
            fs_numerator = float((w[mask] * z_component[mask]) @ x[mask])
            contribution_rows.append({
                "h": H,
                "county_weight": scheme,
                "year": int(year),
                "component": component,
                "component_variance_share": variance / totals["variance"],
                "full_instrument_variance_share": variance / full["variance"],
                "rf_contribution_to_component_x1e5": (
                    rf_numerator / totals["variance"] * locked.SCALE
                ),
                "rf_contribution_to_full_x1e5": (
                    rf_numerator / full["variance"] * locked.SCALE
                ),
                "fs_contribution_to_component": (
                    fs_numerator / totals["variance"]
                ),
                "fs_contribution_to_full": fs_numerator / full["variance"],
            })

    loo_rows = []
    all_years = np.unique(years)
    for omitted_year in all_years:
        keep = years != omitted_year
        full_loo = ratio_statistics(z_full, y, x, w, keep)
        controlled_loo = ratio_statistics(
            fit["zr"], fit["yr"], fit["xr"], w, keep
        )
        loo_rows.append({
            "h": H,
            "county_weight": scheme,
            "omitted_year": int(omitted_year),
            "full_rf_x1e5": full_loo["rf"] * locked.SCALE,
            "full_fs": full_loo["fs"],
            "full_iv_x1e5": full_loo["iv"] * locked.SCALE,
            "controlled_rf_x1e5": controlled_loo["rf"] * locked.SCALE,
            "controlled_fs": controlled_loo["fs"],
            "controlled_iv_x1e5": controlled_loo["iv"] * locked.SCALE,
            "full_iv_change_x1e5": (
                full_loo["iv"] - full["iv"]
            ) * locked.SCALE,
            "controlled_iv_change_x1e5": (
                controlled_loo["iv"] - retained["iv"]
            ) * locked.SCALE,
        })

    block_rows = []
    all_years = np.unique(years)
    for first_year in range(int(all_years.min()), int(all_years.max()) - 3):
        last_year = first_year + 4
        keep = ~np.isin(years, np.arange(first_year, last_year + 1))
        full_block = ratio_statistics(z_full, y, x, w, keep)
        controlled_block = ratio_statistics(
            fit["zr"], fit["yr"], fit["xr"], w, keep
        )
        block_rows.append({
            "h": H,
            "county_weight": scheme,
            "first_omitted_year": first_year,
            "last_omitted_year": last_year,
            "full_iv_x1e5": full_block["iv"] * locked.SCALE,
            "controlled_iv_x1e5": controlled_block["iv"] * locked.SCALE,
            "full_minus_controlled_x1e5": (
                full_block["iv"] - controlled_block["iv"]
            ) * locked.SCALE,
        })

    summary = {
        "county_weight": scheme,
        "full_rf_x1e5": full["rf"] * locked.SCALE,
        "full_fs": full["fs"],
        "full_iv_x1e5": full["iv"] * locked.SCALE,
        "retained_rf_x1e5": retained["rf"] * locked.SCALE,
        "retained_fs": retained["fs"],
        "retained_iv_x1e5": retained["iv"] * locked.SCALE,
        "removed_rf_x1e5": removed["rf"] * locked.SCALE,
        "removed_fs": removed["fs"],
        "removed_iv_x1e5": removed["iv"] * locked.SCALE,
        "instrument_variance_removed": removed["variance"] / full["variance"],
        "orthogonality_gap": cross / full["variance"],
    }
    return (
        pd.DataFrame(contribution_rows), pd.DataFrame(loo_rows),
        pd.DataFrame(block_rows), summary,
    )


def verify_contributions(contributions: pd.DataFrame,
                         summaries: pd.DataFrame) -> None:
    """Assert that annual contributions reconstruct every component moment."""
    for row in summaries.itertuples():
        d = contributions.loc[contributions["county_weight"].eq(row.county_weight)]
        kept = d.loc[d["component"].str.startswith("retained")]
        removed = d.loc[d["component"].str.startswith("removed")]
        checks = {
            "retained RF": (
                kept["rf_contribution_to_component_x1e5"].sum(),
                row.retained_rf_x1e5,
            ),
            "removed RF": (
                removed["rf_contribution_to_component_x1e5"].sum(),
                row.removed_rf_x1e5,
            ),
            "retained first stage": (
                kept["fs_contribution_to_component"].sum(), row.retained_fs
            ),
            "removed first stage": (
                removed["fs_contribution_to_component"].sum(), row.removed_fs
            ),
            "full RF": (
                d["rf_contribution_to_full_x1e5"].sum(), row.full_rf_x1e5
            ),
            "full first stage": (
                d["fs_contribution_to_full"].sum(), row.full_fs
            ),
        }
        for label, (actual, expected) in checks.items():
            if not np.isclose(actual, expected, rtol=1e-11, atol=1e-11):
                raise SystemExit(
                    f"FAIL: {row.county_weight} {label} reconstruction gap "
                    f"{actual - expected:.3e}"
                )


def verify_leave_one_out(base: dict, bhj, diagnostic,
                         loo: pd.DataFrame) -> None:
    """Refit selected omissions to verify the blockwise leave-one-out shortcut."""
    all_years = sorted(int(v) for v in base["d"]["year"].unique())
    audit_years = (all_years[0], all_years[len(all_years) // 2], all_years[-1])
    for scheme in ("equal", "population"):
        for omitted_year in audit_years:
            years_keep = set(all_years) - {omitted_year}
            inp = diagnostic.make_input(
                base, bhj, scheme, years_keep=years_keep
            )
            sf = bhj.shock_frame(inp)
            fit0 = bhj.rung_fit(inp, sf, 0)
            fit1 = bhj.rung_fit(inp, sf, 1)
            full = ratio_statistics(
                fit0["zr"], fit0["yr"], fit0["xr"], inp["w"]
            )
            controlled = ratio_statistics(
                fit1["zr"], fit1["yr"], fit1["xr"], inp["w"]
            )
            expected = loo.loc[
                loo["county_weight"].eq(scheme)
                & loo["omitted_year"].eq(omitted_year)
            ].iloc[0]
            checks = {
                "full IV": (full["iv"] * locked.SCALE, expected.full_iv_x1e5),
                "controlled IV": (
                    controlled["iv"] * locked.SCALE,
                    expected.controlled_iv_x1e5,
                ),
            }
            for label, (actual, target) in checks.items():
                if not np.isclose(actual, target, rtol=1e-10, atol=1e-10):
                    raise SystemExit(
                        f"FAIL: {scheme} omit {omitted_year} {label} shortcut "
                        f"gap {actual - target:.3e}"
                    )


def write_table(summaries: pd.DataFrame, contributions: pd.DataFrame,
                loo: pd.DataFrame, blocks: pd.DataFrame) -> None:
    """Write a compact table of component estimates and influential years."""
    rows = []
    for row in summaries.itertuples():
        label = "Equal county weights" if row.county_weight == "equal" else "Fixed 1975 population weights"
        rows.append(
            f"{label} & {row.full_iv_x1e5:.3f} & {row.retained_iv_x1e5:.3f} "
            f"& {row.removed_iv_x1e5:.3f} "
            f"& {100 * row.instrument_variance_removed:.1f}\\% \\\\"
        )

    pop_removed = contributions.loc[
        contributions["county_weight"].eq("population")
        & contributions["component"].str.startswith("removed")
    ].copy()
    pop_removed["absolute_contribution"] = pop_removed[
        "rf_contribution_to_full_x1e5"
    ].abs()
    top = pop_removed.nlargest(5, "absolute_contribution")
    loo_pop = loo.loc[loo["county_weight"].eq("population")].set_index(
        "omitted_year"
    )
    top_rows = []
    for row in top.itertuples():
        leave = loo_pop.loc[row.year]
        top_rows.append(
            f"{int(row.year)} & {row.rf_contribution_to_full_x1e5:.3f} "
            f"& {row.fs_contribution_to_full:.3f} "
            f"& {leave.full_iv_x1e5:.3f} "
            f"& {leave.controlled_iv_x1e5:.3f} \\\\"
        )

    block_rows = []
    for scheme, label in (
        ("equal", "Equal county weights"),
        ("population", "Fixed 1975 population weights"),
    ):
        candidate = blocks.loc[blocks["county_weight"].eq(scheme)].copy()
        candidate["absolute_gap"] = candidate[
            "full_minus_controlled_x1e5"
        ].abs()
        row = candidate.nsmallest(1, "absolute_gap").iloc[0]
        block_rows.append(
            f"{label} & {int(row.first_omitted_year)}--{int(row.last_omitted_year)} "
            f"& {row.full_iv_x1e5:.3f} & {row.controlled_iv_x1e5:.3f} "
            f"& {row.full_minus_controlled_x1e5:.3f} \\\\"
        )

    target = OUT / "Table_bhj_share_sum_year_decomposition_h2.tex"
    target.write_text(
        """\\begin{table}[!htbp]
\\caption{Annual Decomposition of the Two-Year Composition Instrument}
\\label{tab:bhj_share_sum_year_decomposition_h2}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrrrr}
\\toprule
\\multicolumn{5}{l}{\\textit{Panel A. Orthogonal instrument components}} \\\\
County weighting & Full 2SLS & Retained 2SLS & Removed 2SLS
& $z$ variance removed \\\\
\\midrule
"""
        + "\n".join(rows)
        + """
\\addlinespace
\\multicolumn{5}{l}{\\textit{Panel B. Largest annual reduced-form contributions, fixed 1975 population weights}} \\\\
Target year & Removed RF contribution & Removed FS contribution
& Full 2SLS, omit year & Controlled 2SLS, omit year \\\\
\\midrule
"""
        + "\n".join(top_rows)
        + """
\\addlinespace
\\multicolumn{5}{l}{\\textit{Panel C. Five-year omission minimizing the absolute specification gap}} \\\\
County weighting & Omitted years & Full 2SLS & Controlled 2SLS
& Full minus controlled \\\\
\\midrule
"""
        + "\n".join(block_rows)
        + """
\\bottomrule
\\end{tabular}}
\\par}
\\medskip
{\\scriptsize \\noindent\\textit{Notes:} Coefficients and reduced-form
contributions are multiplied by $10^5$. The retained component is the residual
from projecting the composition instrument on total predetermined procurement
exposure interacted with target-year indicators. The removed component is the
orthogonal fitted value from that projection. Annual reduced-form and
first-stage contributions in Panel B use the full instrument variance as their
denominator, so contributions from both components and all years sum to the
full-sample moments. Panel B selects years by the absolute size of the removed
reduced-form contribution. Leave-one-year-out estimates are leverage
diagnostics; adjacent two-year outcome windows overlap. Panel C searches over
every contiguous five-year omission window. These are
descriptive sensitivity calculations rather than independently selected
samples. All specifications use two-year log employment growth and
division-by-year fixed effects. Population
weights use county population fixed in 1975.\\par}
\\end{table}
""",
        encoding="utf-8",
    )
    print(f"wrote {target}")


def write_figure(contributions: pd.DataFrame, loo: pd.DataFrame,
                 summaries: pd.DataFrame) -> None:
    """Plot annual contributions and leave-one-year-out population estimates."""
    pop = contributions.loc[contributions["county_weight"].eq("population")]
    retained = pop.loc[pop["component"].str.startswith("retained")].sort_values("year")
    removed = pop.loc[pop["component"].str.startswith("removed")].sort_values("year")
    leave = loo.loc[loo["county_weight"].eq("population")].sort_values(
        "omitted_year"
    )
    summary = summaries.set_index("county_weight").loc["population"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.625))
    ax = axes[0]
    width = 0.38
    ax.bar(
        retained["year"] - width / 2,
        retained["rf_contribution_to_full_x1e5"],
        width=width,
        color=STATA_NAVY,
        alpha=0.80,
        label="Retained component",
    )
    ax.bar(
        removed["year"] + width / 2,
        removed["rf_contribution_to_full_x1e5"],
        width=width,
        color=STATA_MAROON,
        alpha=0.80,
        label="Removed component",
    )
    ax.axhline(0, color=REF_LINE_GRAY, linewidth=0.8)
    ax.set_title("Annual reduced-form contributions")
    ax.set_xlabel("Target year")
    ax.set_ylabel(r"Contribution to full reduced form ($\times 10^5$)")

    ax = axes[1]
    ax.scatter(
        leave["omitted_year"], leave["full_iv_x1e5"],
        color=STATA_NAVY, s=26, marker="o", label="Full instrument",
    )
    ax.scatter(
        leave["omitted_year"], leave["controlled_iv_x1e5"],
        color=STATA_MAROON, s=26, marker="s",
        label="Share-sum-by-year controlled",
    )
    ax.axhline(
        summary.full_iv_x1e5, color=STATA_NAVY, linewidth=1.0,
        linestyle="--",
    )
    ax.axhline(
        summary.retained_iv_x1e5, color=STATA_MAROON, linewidth=1.0,
        linestyle="--",
    )
    ax.set_title("Leave-one-year-out estimates")
    ax.set_xlabel("Omitted target year")
    ax.set_ylabel(r"2SLS coefficient ($\times 10^5$)")

    ticks = np.arange(
        int(pop["year"].min()), int(pop["year"].max()) + 1, 4
    )
    for ax in axes:
        ax.set_xticks(ticks)
        ax.tick_params(axis="x", rotation=45)

    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(lab)
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(), unique.keys(), loc="upper center",
        bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    stem = OUT / "Figure_bhj_share_sum_year_decomposition_h2"
    for extension in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"))
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png/.svg")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bhj = load_module(PIPELINE / "22_bhj_assumption1.py", "bhj_assumption1")
    diagnostic = load_module(
        PIPELINE / "22d_bhj_incomplete_share_checks_h2.py",
        "bhj_incomplete_share_checks_h2",
    )
    base = bhj.build_inputs(H)
    add_log_outcome(base)

    contributions, leave_one_out, leave_five_years_out, summaries = [], [], [], []
    for scheme in ("equal", "population"):
        annual, leave, blocks, summary = decompose_scheme(
            base, bhj, diagnostic, scheme
        )
        contributions.append(annual)
        leave_one_out.append(leave)
        leave_five_years_out.append(blocks)
        summaries.append(summary)

    contribution_result = pd.concat(contributions, ignore_index=True)
    loo_result = pd.concat(leave_one_out, ignore_index=True)
    block_result = pd.concat(leave_five_years_out, ignore_index=True)
    summary_result = pd.DataFrame(summaries)
    verify_contributions(contribution_result, summary_result)
    verify_leave_one_out(base, bhj, diagnostic, loo_result)

    contribution_path = OUT / "bhj_share_sum_year_contributions_h2.csv"
    loo_path = OUT / "bhj_share_sum_leave_one_year_out_h2.csv"
    block_path = OUT / "bhj_share_sum_leave_five_years_out_h2.csv"
    summary_path = OUT / "bhj_share_sum_component_summary_h2.csv"
    contribution_result.to_csv(contribution_path, index=False)
    loo_result.to_csv(loo_path, index=False)
    block_result.to_csv(block_path, index=False)
    summary_result.to_csv(summary_path, index=False)
    write_table(summary_result, contribution_result, loo_result, block_result)
    write_figure(contribution_result, loo_result, summary_result)
    print(f"wrote {contribution_path}")
    print(f"wrote {loo_path}")
    print(f"wrote {block_path}")
    print(f"wrote {summary_path}")
    print(summary_result.to_string(index=False))


if __name__ == "__main__":
    main()
