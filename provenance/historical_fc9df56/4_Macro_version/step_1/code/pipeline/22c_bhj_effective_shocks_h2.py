#!/usr/bin/env python3
"""BHJ shock-concentration diagnostics at h=2 under two county weights.

Outputs in Results/pipeline_diagnostics/:
    Table_bhj_effective_shocks_h2_weights.tex
    bhj_effective_shocks_h2.csv
    bhj_category_importance_h2.csv
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402
import store  # noqa: E402
from clusterstats import hhi_effective  # noqa: E402


H = 2
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"


def load_bhj():
    spec = importlib.util.spec_from_file_location(
        "bhj_assumption1", PIPELINE / "22_bhj_assumption1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def county_weights(base: dict, scheme: str) -> np.ndarray:
    """Return equal or fixed-1975 county population weights."""
    if scheme == "equal":
        return np.ones(len(base["d"]), np.float64)
    panel = store.load_county_panel()
    p0 = (panel.loc[panel["year"].eq(locked.POP_BASE_YEAR), ["ccg", "P"]]
          .drop_duplicates("ccg").set_index("ccg")["P"])
    weights = base["d"]["ccg"].map(p0).to_numpy(np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise SystemExit("FAIL: fixed 1975 population weights are not positive")
    return weights / weights.mean()


def diagnose(base: dict, bhj, scheme: str) -> tuple[dict, pd.DataFrame]:
    """Compute inverse-Herfindahl counts and category importance shares."""
    inp = dict(base)
    inp["w"] = county_weights(base, scheme)
    sf = bhj.shock_frame(inp)
    importance = sf["sv"].astype(np.float64)
    if not np.isfinite(importance).all() or (importance <= 0).any():
        raise SystemExit(f"FAIL: {scheme} shock importance is not finite and positive")

    category_index = sf["keys"][:, 0].astype(np.int64)
    category_total = np.bincount(
        category_index, weights=importance, minlength=len(inp["cats"])
    )
    live_category = category_total > 0
    if int(live_category.sum()) != len(inp["cats"]):
        raise SystemExit(
            f"FAIL: {scheme} has {int(live_category.sum()):,} positive categories, "
            f"expected {len(inp['cats']):,}"
        )
    normalized = category_total / category_total.sum()
    normalized_cells = importance / importance.sum()
    order = np.argsort(-normalized)
    detail = pd.DataFrame(
        {
            "h": H,
            "county_weight": scheme,
            "rank": np.arange(1, len(order) + 1),
            "category": np.asarray(inp["cats"], dtype=str)[order],
            "importance": category_total[order],
            "importance_share": normalized[order],
            "cumulative_importance_share": np.cumsum(normalized[order]),
        }
    )
    summary = {
        "h": H,
        "county_weight": scheme,
        "shock_cells": len(importance),
        "categories": int(live_category.sum()),
        "effective_category_year_shocks": hhi_effective(importance),
        "effective_categories": hhi_effective(category_total),
        "largest_category_year_share": float(normalized_cells.max()),
        "largest_category": str(detail.loc[0, "category"]),
        "largest_category_share": float(detail.loc[0, "importance_share"]),
        "top_5_category_share": float(detail.loc[4, "cumulative_importance_share"]),
        "top_10_category_share": float(detail.loc[9, "cumulative_importance_share"]),
        "top_20_category_share": float(detail.loc[19, "cumulative_importance_share"]),
    }
    if not (
        1 <= summary["effective_category_year_shocks"] <= summary["shock_cells"]
        and 1 <= summary["effective_categories"] <= summary["categories"]
        and 0 < summary["largest_category_year_share"]
        <= summary["largest_category_share"]
        <= summary["top_5_category_share"]
        <= summary["top_10_category_share"]
        <= summary["top_20_category_share"]
        <= 1
    ):
        raise SystemExit(f"FAIL: invalid {scheme} shock-concentration statistics")
    return summary, detail


def write_table(summary: pd.DataFrame) -> None:
    """Write the side-by-side equal- and population-weighted table."""
    values = summary.set_index("county_weight")
    equal = values.loc["equal"]
    population = values.loc["population"]
    rows = [
        ("Shock-level observations", f"{int(equal.shock_cells):,}", f"{int(population.shock_cells):,}"),
        ("Categories with positive importance", f"{int(equal.categories):,}", f"{int(population.categories):,}"),
        ("Effective category-year shocks", f"{equal.effective_category_year_shocks:.1f}", f"{population.effective_category_year_shocks:.1f}"),
        ("Effective categories", f"{equal.effective_categories:.1f}", f"{population.effective_categories:.1f}"),
        ("Largest category-year share", f"{100 * equal.largest_category_year_share:.1f}\\%", f"{100 * population.largest_category_year_share:.1f}\\%"),
        ("Largest category", str(equal.largest_category), str(population.largest_category)),
        ("Largest category share", f"{100 * equal.largest_category_share:.1f}\\%", f"{100 * population.largest_category_share:.1f}\\%"),
        ("Top 5 categories", f"{100 * equal.top_5_category_share:.1f}\\%", f"{100 * population.top_5_category_share:.1f}\\%"),
        ("Top 10 categories", f"{100 * equal.top_10_category_share:.1f}\\%", f"{100 * population.top_10_category_share:.1f}\\%"),
        ("Top 20 categories", f"{100 * equal.top_20_category_share:.1f}\\%", f"{100 * population.top_20_category_share:.1f}\\%"),
    ]
    body = "\n".join(f"\\hspace{{1em}}{label} & {ew} & {pw} \\\\" for label, ew, pw in rows)
    target = OUT / "Table_bhj_effective_shocks_h2_weights.tex"
    target.write_text(
        """\\begin{table}[!htbp]
\\caption{BHJ Shock Concentration at the Two-Year Horizon}
\\label{tab:bhj_effective_shocks_h2_weights}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrr}
\\toprule
& Equal county weights & Population weights \\\\
\\midrule
"""
        + body
        + """
\\bottomrule
\\end{tabular}}
\\par}
\\medskip
{\\scriptsize \\noindent\\textit{Notes:} Shock importance is
$s_{kt}=\\sum_c w_{ct}s_{ckt}$. Effective counts are the inverse Herfindahl
index of normalized shock-importance weights. The category calculation first
sums importance over all years within each Federal Supply Classification
category, allowing arbitrary serial dependence within category. Concentration
shares report the fraction of total shock importance assigned to the largest
categories. Population weights use county population fixed in 1975 and constant
across target years. These are descriptive diagnostics, not hypothesis tests.\\par}
\\end{table}
""",
        encoding="utf-8",
    )
    print(f"wrote {target}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bhj = load_bhj()
    base = bhj.build_inputs(H)
    summaries = []
    details = []
    for scheme in ("equal", "population"):
        summary, detail = diagnose(base, bhj, scheme)
        summaries.append(summary)
        details.append(detail)
        print(
            f"h={H}, weight={scheme}, cells={summary['shock_cells']:,}, "
            f"effective cells={summary['effective_category_year_shocks']:.1f}, "
            f"effective categories={summary['effective_categories']:.1f}, "
            f"top 10 share={100 * summary['top_10_category_share']:.2f}%"
        )

    summary_frame = pd.DataFrame(summaries)
    detail_frame = pd.concat(details, ignore_index=True)
    summary_target = OUT / "bhj_effective_shocks_h2.csv"
    detail_target = OUT / "bhj_category_importance_h2.csv"
    summary_frame.to_csv(summary_target, index=False)
    detail_frame.to_csv(detail_target, index=False)
    write_table(summary_frame)
    print(f"wrote {summary_target}")
    print(f"wrote {detail_target}")


if __name__ == "__main__":
    main()
