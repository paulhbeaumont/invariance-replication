#!/usr/bin/env python3
"""Trace the population-weighted h=2 share-sum component in 1989-1993.

The decomposition uses the exact compact two-year log employment outcome,
division-by-year fixed effects, and county population fixed in 1975 as
regression weights. Category contributions use the BHJ shock-level identity.
County contributions use the corresponding county-level orthogonal component.

Outputs in Results/pipeline_diagnostics/:
    bhj_1989_1993_category_year_drivers_h2.csv
    bhj_1989_1993_category_drivers_h2.csv
    bhj_1989_1993_county_drivers_h2.csv
    bhj_1989_1993_driver_summary_h2.csv
    Table_bhj_1989_1993_drivers_h2.tex
    Figure_bhj_1989_1993_drivers_h2.{pdf,png,svg}
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
REFERENCE = PIPELINE.parent / "reference"
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402
import store  # noqa: E402


H = 2
BLOCK = tuple(range(1989, 1994))
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"

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
    "axes.grid.axis": "x",
    "axes.axisbelow": True,
    "grid.color": GRID_GRAY,
    "grid.linewidth": 0.4,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "text.color": DARK,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def load_module(path: Path, name: str):
    """Load a pipeline module without invoking its command-line entry point."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_log_outcome(base: dict) -> None:
    """Attach the compact two-year log employment outcome."""
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


def category_labels(cats: list[str]) -> dict[str, str]:
    """Resolve dollar-weighted modal FSC descriptions from the source files."""
    rotemberg = load_module(PIPELINE / "21_rotemberg.py", "rotemberg_labels")
    resolved = rotemberg.fsc_labels(set(str(cat) for cat in cats))
    return {str(cat): resolved.get(str(cat), str(cat)) for cat in cats}


def county_labels() -> dict[int, str]:
    """Return canonical county-group names by analysis county identifier."""
    path = REFERENCE / "county_ccg_crosswalk.csv"
    crosswalk = pd.read_csv(path, dtype=str)
    ids = pd.to_numeric(crosswalk["ccg_id"], errors="coerce")
    names = crosswalk["ccg_name"].fillna("").str.strip()
    d = pd.DataFrame({"ccg": ids, "name": names})
    d = d[d["ccg"].notna() & d["name"].ne("")].copy()
    d["ccg"] = d["ccg"].astype(np.int64)
    return d.drop_duplicates("ccg").set_index("ccg")["name"].to_dict()


def category_decomposition(inp: dict, sf: dict, fit0: dict, fit1: dict,
                           cats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate the removed 1989-1993 moments through the BHJ identity."""
    years = np.array([inp["years"][j] for j in sf["keys"][:, 1]], np.int64)
    category_index = sf["keys"][:, 0].astype(np.int64)
    full_variance = float(inp["w"] @ (fit0["zr"] * fit0["zr"]))
    rf_numerator = sf["gv"] * (fit0["ny"] - fit1["ny"])
    fs_numerator = sf["gv"] * (fit0["nx"] - fit1["nx"])
    d = pd.DataFrame({
        "h": H,
        "year": years,
        "category": np.asarray(cats, dtype=str)[category_index],
        "shock": sf["gv"],
        "importance_weight": sf["sv"],
        "removed_rf_contribution_to_full_x1e5": (
            rf_numerator / full_variance * locked.SCALE
        ),
        "removed_fs_contribution_to_full": fs_numerator / full_variance,
    })
    d = d[d["year"].isin(BLOCK)].copy()
    labels = category_labels(cats)
    d["description"] = d["category"].map(labels)

    category = (
        d.groupby(["category", "description"], as_index=False)
        .agg(
            removed_rf_contribution_to_full_x1e5=(
                "removed_rf_contribution_to_full_x1e5", "sum"
            ),
            removed_fs_contribution_to_full=(
                "removed_fs_contribution_to_full", "sum"
            ),
            mean_shock=("shock", "mean"),
            total_importance_weight=("importance_weight", "sum"),
        )
    )
    category["absolute_rf_contribution_x1e5"] = category[
        "removed_rf_contribution_to_full_x1e5"
    ].abs()
    category["rank_by_absolute_rf_contribution"] = category[
        "absolute_rf_contribution_x1e5"
    ].rank(method="first", ascending=False).astype(int)
    category = category.sort_values(
        "rank_by_absolute_rf_contribution"
    ).reset_index(drop=True)
    return d, category


def county_decomposition(inp: dict, fit0: dict, fit1: dict) -> pd.DataFrame:
    """Allocate the removed 1989-1993 moments across counties."""
    years = inp["d"]["year"].to_numpy(np.int64)
    block = np.isin(years, BLOCK)
    full_variance = float(inp["w"] @ (fit0["zr"] * fit0["zr"]))
    z_removed = fit0["zr"] - fit1["zr"]
    d = inp["d"].loc[block, ["ccg", "division"]].copy()
    panel = store.load_county_panel()
    state = panel.drop_duplicates("ccg").set_index("ccg")["state"]
    d["state"] = d["ccg"].map(state)
    d["removed_rf_contribution_to_full_x1e5"] = (
        inp["w"][block] * z_removed[block] * inp["yt"][block]
        / full_variance * locked.SCALE
    )
    d["removed_fs_contribution_to_full"] = (
        inp["w"][block] * z_removed[block] * inp["xt"][block]
        / full_variance
    )
    d["removed_component_variance_share"] = (
        inp["w"][block] * z_removed[block] ** 2 / full_variance
    )
    county = (
        d.groupby(["ccg", "state", "division"], as_index=False)
        .sum(numeric_only=True)
    )
    labels = county_labels()
    county["county_name"] = county["ccg"].map(labels).fillna("")
    county["county_fips"] = county["ccg"].map(lambda x: f"{int(x):05d}")
    county["display_name"] = np.where(
        county["county_name"].ne(""),
        county["county_name"] + " (" + county["county_fips"] + ")",
        county["county_fips"],
    )
    county["absolute_rf_contribution_x1e5"] = county[
        "removed_rf_contribution_to_full_x1e5"
    ].abs()
    county["rank_by_absolute_rf_contribution"] = county[
        "absolute_rf_contribution_x1e5"
    ].rank(method="first", ascending=False).astype(int)
    return county.sort_values(
        "rank_by_absolute_rf_contribution"
    ).reset_index(drop=True)


def summarize(category: pd.DataFrame, county: pd.DataFrame,
              annual_total: float) -> pd.DataFrame:
    """Summarize reconstruction and concentration of the block contribution."""
    rows = []
    for dimension, d in (("category", category), ("county", county)):
        rf = d["removed_rf_contribution_to_full_x1e5"]
        absolute = d["absolute_rf_contribution_x1e5"]
        ordered = d.sort_values("rank_by_absolute_rf_contribution")
        rows.append({
            "h": H,
            "first_year": min(BLOCK),
            "last_year": max(BLOCK),
            "dimension": dimension,
            "units": len(d),
            "signed_rf_contribution_x1e5": rf.sum(),
            "absolute_rf_contribution_x1e5": absolute.sum(),
            "top_1_absolute_share": (
                ordered.head(1)["absolute_rf_contribution_x1e5"].sum()
                / absolute.sum()
            ),
            "top_5_absolute_share": (
                ordered.head(5)["absolute_rf_contribution_x1e5"].sum()
                / absolute.sum()
            ),
            "top_10_absolute_share": (
                ordered.head(10)["absolute_rf_contribution_x1e5"].sum()
                / absolute.sum()
            ),
            "reconstruction_gap": rf.sum() - annual_total,
        })
    return pd.DataFrame(rows)


def escape_tex(value: str) -> str:
    """Escape the small subset of LaTeX characters used in source labels."""
    return str(value).replace("&", r"\&").replace("%", r"\%")


def write_table(category: pd.DataFrame, county: pd.DataFrame,
                summary: pd.DataFrame) -> None:
    """Write the leading category and county contributions."""
    category_rows = []
    for row in category.head(7).itertuples():
        label = escape_tex(f"{row.category}: {row.description}")
        category_rows.append(
            f"{label} & {row.removed_rf_contribution_to_full_x1e5:.3f} "
            f"& {row.removed_fs_contribution_to_full:.3f} \\\\"
        )
    county_rows = []
    for row in county.head(7).itertuples():
        county_rows.append(
            f"{escape_tex(row.display_name)} & "
            f"{row.removed_rf_contribution_to_full_x1e5:.3f} "
            f"& {row.removed_fs_contribution_to_full:.3f} \\\\"
        )
    concentration_rows = []
    for row in summary.itertuples():
        label = "Categories" if row.dimension == "category" else "Counties"
        concentration_rows.append(
            f"{label} & {100 * row.top_5_absolute_share:.1f}\\% "
            f"& {100 * row.top_10_absolute_share:.1f}\\% \\\\"
        )

    target = OUT / "Table_bhj_1989_1993_drivers_h2.tex"
    target.write_text(
        """\\begin{table}[!htbp]
\\caption{Sources of the 1989--1993 Share-Sum Contribution}
\\label{tab:bhj_1989_1993_drivers_h2}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrr}
\\toprule
\\multicolumn{3}{l}{\\textit{Panel A. Procurement categories}} \\\\
Category & Removed RF contribution & Removed FS contribution \\\\
\\midrule
"""
        + "\n".join(category_rows)
        + """
\\addlinespace
\\multicolumn{3}{l}{\\textit{Panel B. Counties}} \\\\
County group & Removed RF contribution & Removed FS contribution \\\\
\\midrule
"""
        + "\n".join(county_rows)
        + """
\\addlinespace
\\multicolumn{3}{l}{\\textit{Panel C. Share of absolute reduced-form contributions}} \\\\
Dimension & Top five & Top ten \\\\
\\midrule
"""
        + "\n".join(concentration_rows)
        + """
\\bottomrule
\\end{tabular}}
\\par}
\\medskip
{\\scriptsize \\noindent\\textit{Notes:} Reduced-form contributions are
multiplied by $10^5$. Contributions explain the component removed by total
predetermined procurement exposure interacted with target-year indicators for
the 1989--1993 target years. Category contributions use the BHJ shock-level
identity; county contributions use the equivalent county-level orthogonal
component. Rankings use absolute reduced-form contributions, while the table
reports their signs. The denominator is full-sample instrument variation, so
signed contributions aggregate to the 1989--1993 contribution reported in the
annual decomposition. The specification uses two-year log employment growth,
division-by-year fixed effects, and county population fixed in 1975 as
regression weights.\\par}
\\end{table}
""",
        encoding="utf-8",
    )
    print(f"wrote {target}")


def bar_panel(ax, d: pd.DataFrame, label_column: str, title: str) -> None:
    """Draw one signed horizontal contribution ranking."""
    plot = d.head(12).sort_values(
        "removed_rf_contribution_to_full_x1e5"
    )
    values = plot["removed_rf_contribution_to_full_x1e5"]
    colors = np.where(values >= 0, STATA_NAVY, STATA_MAROON)
    ax.barh(plot[label_column], values, color=colors, alpha=0.80)
    ax.axvline(0, color=REF_LINE_GRAY, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(r"Removed reduced-form contribution ($\times 10^5$)")


def write_figure(category: pd.DataFrame, county: pd.DataFrame) -> None:
    """Plot the largest category and county contributions."""
    category = category.copy()
    category["display_name"] = (
        category["category"] + ": " + category["description"]
    ).str.slice(0, 48)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.625))
    bar_panel(axes[0], category, "display_name", "Procurement categories")
    bar_panel(axes[1], county, "display_name", "County groups")
    positive = plt.Rectangle((0, 0), 1, 1, color=STATA_NAVY, alpha=0.80)
    negative = plt.Rectangle((0, 0), 1, 1, color=STATA_MAROON, alpha=0.80)
    fig.legend(
        [positive, negative], ["Positive contribution", "Negative contribution"],
        loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    stem = OUT / "Figure_bhj_1989_1993_drivers_h2"
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
    inp = diagnostic.make_input(base, bhj, "population")
    sf = bhj.shock_frame(inp)
    fit0 = bhj.rung_fit(inp, sf, 0)
    fit1 = bhj.rung_fit(inp, sf, 1)
    if max(fit0["gap"], fit1["gap"]) > 1e-9:
        raise SystemExit("FAIL: county and shock-level coefficients diverge")

    category_year, category = category_decomposition(
        inp, sf, fit0, fit1, base["cats"]
    )
    county = county_decomposition(inp, fit0, fit1)
    annual_path = OUT / "bhj_share_sum_year_contributions_h2.csv"
    annual = pd.read_csv(annual_path)
    annual_total = annual.loc[
        annual["county_weight"].eq("population")
        & annual["year"].isin(BLOCK)
        & annual["component"].str.startswith("removed"),
        "rf_contribution_to_full_x1e5",
    ].sum()
    summary = summarize(category, county, annual_total)
    if summary["reconstruction_gap"].abs().max() > 1e-10:
        raise SystemExit(
            "FAIL: category or county drivers do not reconstruct the annual total"
        )

    paths = {
        "category-year": OUT / "bhj_1989_1993_category_year_drivers_h2.csv",
        "category": OUT / "bhj_1989_1993_category_drivers_h2.csv",
        "county": OUT / "bhj_1989_1993_county_drivers_h2.csv",
        "summary": OUT / "bhj_1989_1993_driver_summary_h2.csv",
    }
    category_year.to_csv(paths["category-year"], index=False)
    category.to_csv(paths["category"], index=False)
    county.to_csv(paths["county"], index=False)
    summary.to_csv(paths["summary"], index=False)
    write_table(category, county, summary)
    write_figure(category, county)
    for path in paths.values():
        print(f"wrote {path}")
    print(summary.to_string(index=False))
    print("\nTop category contributions:")
    print(category.head(10).to_string(index=False))
    print("\nTop county contributions:")
    print(county.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
