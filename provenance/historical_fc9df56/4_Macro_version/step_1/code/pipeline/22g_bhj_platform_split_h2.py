#!/usr/bin/env python3
"""Split the population-weighted h=2 instrument by procurement content.

The specification uses the exact compact two-year log employment outcome,
division-by-year fixed effects, and county population fixed in 1975 as
regression weights. The category partitions are defined from the Federal
Supply Classification or Product and Service Code hierarchy before estimation:
numeric groups 10 through 20 identify weapons and platforms, and letter group
A identifies research and development services.

Outputs in Results/pipeline_diagnostics/:
    bhj_platform_split_h2.csv
    Table_bhj_platform_split_h2.tex
    Figure_bhj_platform_split_h2.{pdf,png,svg}
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
Z95 = 1.959963984540054

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
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "axes.axisbelow": True,
    "grid.color": GRID_GRAY,
    "grid.linewidth": 0.4,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
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


def category_masks(cats: list[str]) -> dict[str, np.ndarray]:
    """Return two exhaustive category partitions defined from PSC hierarchy."""
    codes = np.asarray([str(cat).strip().upper() for cat in cats], dtype=object)
    numeric_group = np.array([
        int(code[:2]) if len(code) >= 2 and code[:2].isdigit() else -1
        for code in codes
    ])
    platform = (numeric_group >= 10) & (numeric_group <= 20)
    research = np.array([code.startswith("A") for code in codes])
    broad = platform | research
    masks = {
        "all": np.ones(len(codes), dtype=bool),
        "platform": platform,
        "not_platform": ~platform,
        "platform_rd": broad,
        "not_platform_rd": ~broad,
    }
    if not np.array_equal(
        masks["platform"] | masks["not_platform"], masks["all"]
    ):
        raise SystemExit("FAIL: narrow category partition is not exhaustive")
    if not np.array_equal(
        masks["platform_rd"] | masks["not_platform_rd"], masks["all"]
    ):
        raise SystemExit("FAIL: broad category partition is not exhaustive")
    return masks


def subset_base(base: dict, mask: np.ndarray) -> dict:
    """Restrict the locked category tensor while retaining county observations."""
    if len(mask) != len(base["cats"]) or not mask.any():
        raise SystemExit("FAIL: invalid category subset")
    out = dict(base)
    out["cats"] = [cat for cat, keep in zip(base["cats"], mask) if keep]
    out["gk"] = base["gk"][mask, :]
    out["dollars_k"] = base["dollars_k"][mask]
    out["cbar"] = {j: values[:, mask] for j, values in base["cbar"].items()}
    return out


def raw_columns(base: dict, diagnostic) -> tuple[np.ndarray, np.ndarray]:
    """Build raw category-subset instrument and share-sum columns."""
    return diagnostic.instrument_columns(base, base["cbar"], base["gk"])


def estimate_group(base: dict, bhj, diagnostic, key: str, label: str,
                   mask: np.ndarray, total_dollars: float) -> list[dict]:
    """Estimate one category subset with and without share-sum-by-year controls."""
    group = base if mask.all() else subset_base(base, mask)
    inp = diagnostic.make_input(group, bhj, "population")
    rows = []
    for rung, control_label in (
        (0, "No share-sum controls"),
        (1, "Share-sum by year"),
    ):
        row = diagnostic.estimate(
            inp, bhj, rung, f"{label}, {control_label.lower()}", "population"
        )
        row.update({
            "group": key,
            "group_label": label,
            "control_label": control_label,
            "categories": int(mask.sum()),
            "procurement_dollar_share": (
                float(base["dollars_k"][mask].sum()) / total_dollars
            ),
        })
        rows.append(row)
    return rows


def validate_additivity(base: dict, masks: dict[str, np.ndarray], diagnostic) -> None:
    """Verify that each raw instrument partition reconstructs the full object."""
    z_all, s_all = raw_columns(base, diagnostic)
    for left, right, name in (
        ("platform", "not_platform", "narrow"),
        ("platform_rd", "not_platform_rd", "broad"),
    ):
        z_left, s_left = raw_columns(subset_base(base, masks[left]), diagnostic)
        z_right, s_right = raw_columns(subset_base(base, masks[right]), diagnostic)
        z_gap = np.nanmax(np.abs(z_all - z_left - z_right))
        s_gap = np.nanmax(np.abs(s_all - s_left - s_right))
        if z_gap > 1e-10 or s_gap > 1e-10:
            raise SystemExit(
                f"FAIL: {name} partition additivity gaps are "
                f"z={z_gap:.3e}, share_sum={s_gap:.3e}"
            )


def validate_baseline(result: pd.DataFrame) -> None:
    """Tie the all-category rows to the existing fixed-population diagnostic."""
    reference_path = OUT / "bhj_incomplete_share_checks_h2.csv"
    reference = pd.read_csv(reference_path)
    reference = reference.loc[
        reference["county_weight"].eq("population")
        & reference["specification"].isin([
            "Current exposure",
            "Current exposure, share-sum by year",
        ])
    ].copy()
    measured = result.loc[result["group"].eq("all")].copy()
    mapping = {
        "No share-sum controls": "Current exposure",
        "Share-sum by year": "Current exposure, share-sum by year",
    }
    for control_label, specification in mapping.items():
        a = measured.loc[measured["control_label"].eq(control_label)]
        b = reference.loc[reference["specification"].eq(specification)]
        if len(a) != 1 or len(b) != 1:
            raise SystemExit("FAIL: all-category tie-out row is not unique")
        for column in (
            "beta", "bhj_iv_se", "county_reduced_form",
            "bhj_reduced_form_se", "county_first_stage_F",
            "bhj_first_stage_F",
        ):
            gap = abs(float(a.iloc[0][column]) - float(b.iloc[0][column]))
            if gap > 1e-10:
                raise SystemExit(
                    f"FAIL: all-category {control_label} {column} "
                    f"tie-out gap {gap:.3e}"
                )


def write_table(result: pd.DataFrame) -> None:
    """Write reduced form, first stage, and 2SLS for each category subset."""
    lines = []
    panels = (
        ("No share-sum controls", "Panel A. No share-sum controls"),
        ("Share-sum by year", "Panel B. Share-sum interacted with target year"),
    )
    for panel_index, (control, heading) in enumerate(panels):
        lines.append(
            rf"\multicolumn{{8}}{{l}}{{\textit{{{heading}}}}} " + r"\\"
        )
        for row in result.loc[result["control_label"].eq(control)].itertuples():
            label = str(row.group_label).replace("&", r"\&")
            lines.append(
                rf"\hspace{{1em}}{label} "
                f"& {int(row.categories):d} "
                rf"& {100 * row.procurement_dollar_share:.1f}\% "
                f"& {row.county_reduced_form * locked.SCALE:.3f} "
                f"& ({row.bhj_reduced_form_se * locked.SCALE:.3f}) "
                f"& {row.bhj_first_stage_F:.2f} "
                f"& {row.beta * locked.SCALE:.3f} "
                f"& ({row.bhj_iv_se * locked.SCALE:.3f}) " + r"\\"
            )
        if panel_index == 0:
            lines.append(r"\addlinespace")
    target = OUT / "Table_bhj_platform_split_h2.tex"
    target.write_text(
        """\\begin{table}[!htbp]
\\caption{Two-Year Estimates by Department of Defense Procurement Content}
\\label{tab:bhj_platform_split_h2}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrrrrrrr}
\\toprule
Category subset & Categories & Dollar share & RF$\\times10^5$ & BHJ s.e.
& BHJ $F$ & 2SLS$\\times10^5$ & BHJ s.e. \\\\
\\midrule
"""
        + "\n".join(lines)
        + """
\\bottomrule
\\end{tabular}}
\\par}
\\medskip
{\\scriptsize \\noindent\\textit{Notes:} All rows use the compact two-year
log employment outcome, division-by-year fixed effects, and county population
fixed in 1975 as regression weights. The treatment is the county procurement
dollar change divided by population at $t-2$. The full instrument is the locked
top-800 four-character procurement composition instrument. Weapons and
platforms are numeric Federal Supply Classification groups 10 through 20.
Research and development services are Product and Service Codes beginning with
A. Each subset uses only its own national category shocks and predetermined
county exposures. Dollar shares use procurement over the locked sample.
Panel B interacts the subset's total predetermined procurement exposure per
person with target-year indicators. BHJ standard errors and first-stage
statistics cluster by four-character procurement category. The two narrow rows
form one exhaustive partition; the two broad rows form a second exhaustive
partition.\\par}
\\end{table}
""",
        encoding="utf-8",
    )
    print(f"wrote {target}")


def write_figure(result: pd.DataFrame) -> None:
    """Plot 2SLS estimates and exposure-robust confidence intervals."""
    plot = result.loc[~result["group"].eq("all")].copy()
    order = [
        "platform", "not_platform", "platform_rd", "not_platform_rd",
    ]
    labels = {
        "platform": "Weapons and platforms",
        "not_platform": "Other DoD procurement",
        "platform_rd": "Weapons, platforms, and R&D",
        "not_platform_rd": "Other DoD procurement excluding R&D",
    }
    ybase = np.arange(len(order), dtype=float)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.625))
    series = (
        ("No share-sum controls", STATA_NAVY, "o", 0.10),
        ("Share-sum by year", STATA_MAROON, "s", -0.10),
    )
    for control, color, marker, offset in series:
        d = plot.loc[plot["control_label"].eq(control)].set_index("group").loc[order]
        coefficient = d["beta"].to_numpy() * locked.SCALE
        error = Z95 * d["bhj_iv_se"].to_numpy() * locked.SCALE
        ax.errorbar(
            coefficient, ybase + offset, xerr=error, fmt=marker,
            color=color, markerfacecolor=color, markeredgecolor=color,
            markersize=6.5, capsize=3, elinewidth=1.2, linestyle="none",
            label=control,
        )
    ax.axvline(0, color=REF_LINE_GRAY, linewidth=0.8)
    ax.set_yticks(ybase)
    ax.set_yticklabels([labels[key] for key in order])
    ax.set_xlabel(r"2SLS coefficient $\times 10^5$ (95% BHJ confidence interval)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    stem = OUT / "Figure_bhj_platform_split_h2"
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
    masks = category_masks(base["cats"])
    validate_additivity(base, masks, diagnostic)

    definitions = (
        ("all", "All DoD procurement"),
        ("platform", "Weapons and platforms (FSC 10--20)"),
        ("not_platform", "Other DoD procurement"),
        ("platform_rd", "Weapons, platforms, and R&D"),
        ("not_platform_rd", "Other DoD procurement excluding R&D"),
    )
    total_dollars = float(base["dollars_k"].sum())
    rows = []
    for key, label in definitions:
        rows.extend(estimate_group(
            base, bhj, diagnostic, key, label, masks[key], total_dollars
        ))
    result = pd.DataFrame(rows)
    result["weak_bhj_first_stage"] = result["bhj_first_stage_F"] < 10
    if not np.isfinite(result.select_dtypes(include=[np.number])).all().all():
        raise SystemExit("FAIL: non-finite value in category-split results")
    validate_baseline(result)

    target = OUT / "bhj_platform_split_h2.csv"
    result.to_csv(target, index=False)
    write_table(result)
    write_figure(result)
    print(f"wrote {target}")
    print(result[[
        "group", "control_label", "categories", "procurement_dollar_share",
        "county_reduced_form", "bhj_reduced_form_se", "bhj_first_stage_F",
        "beta", "bhj_iv_se", "weak_bhj_first_stage", "equivalence_gap",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
