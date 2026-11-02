#!/usr/bin/env python3
"""Leave-one-FSC-group-out influence diagnostic for the h=2 compact IV.

The specification is the compact two-year log employment regression without
the total-exposure control. Each run removes one FSC group from the instrument,
reconstructs and residualizes the county instrument, and computes BHJ
exposure-robust inference clustered by the remaining FSC groups.

Outputs in Results/pipeline_diagnostics/:
    bhj_leave_one_fsc_group_out_h2.csv
    bhj_leave_one_fsc_group_out_summary_h2.csv
    Table_bhj_leave_one_fsc_group_out_h2.tex
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


H = 2
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"


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


def group_codes(categories) -> np.ndarray:
    """Return the FSC group corresponding to every category."""
    out = []
    for raw in categories:
        value = str(raw).strip().upper()
        out.append(value[0] if value[0].isalpha() else value[:2])
    return np.asarray(out, dtype=object)


def group_contributions(inp: dict, groups: np.ndarray,
                        group_index: dict[str, int]) -> np.ndarray:
    """Return each FSC group's raw contribution to every county instrument."""
    n_groups = len(group_index)
    membership = np.zeros((len(groups), n_groups), np.float64)
    membership[np.arange(len(groups)), [group_index[g] for g in groups]] = 1.0
    contributions = np.zeros((len(inp["d"]), n_groups), np.float64)
    for j in sorted(inp["cbar"]):
        rows = inp["cc"] == j
        if not rows.any():
            continue
        tcol = inp["cy"].index(inp["years"][j])
        shocks = inp["gk"][:, tcol]
        live = np.isfinite(shocks)
        exposure = (
            inp["cbar"][j][inp["rr"][rows], :][:, live]
            / inp["pb"][rows, None]
        )
        contributions[rows, :] = (
            exposure * shocks[live][None, :]
        ) @ membership[live, :]
    reconstructed = contributions.sum(axis=1)
    target = inp["d"]["z"].to_numpy(np.float64)
    gap = float(np.max(np.abs(reconstructed - target)))
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(target))))
    if gap > tolerance:
        raise SystemExit(
            f"FAIL: FSC contributions do not reconstruct the instrument, gap {gap:.3e}"
        )
    return contributions


def subset_shock_frame(sf: dict, keep: np.ndarray) -> dict:
    """Return the live shock frame after removing selected shock cells."""
    return {
        "keys": sf["keys"][keep],
        "sv": sf["sv"][keep],
        "gv": sf["gv"][keep],
        "blocks": sf["blocks"],
        "live": sf["live"],
    }


def run_scheme(base: dict, bhj, diagnostic, scheme: str) -> tuple[pd.DataFrame, dict]:
    """Run the baseline and every leave-one-FSC-group-out specification."""
    inp = diagnostic.make_input(base, bhj, scheme)
    sf = bhj.shock_frame(inp)
    category_groups = group_codes(inp["cats"])
    groups = np.asarray(sorted(set(category_groups.tolist())), dtype=object)
    group_index = {group: j for j, group in enumerate(groups)}
    contributions = group_contributions(inp, category_groups, group_index)
    shock_groups = category_groups[sf["keys"][:, 0]]

    baseline_fit = bhj.rung_fit(inp, sf, 0)
    baseline_se, baseline_clusters = bhj.exposure_robust(
        baseline_fit, sf, inp["cats"], "FSC group"
    )
    baseline_f = bhj.first_stage_F(
        baseline_fit, sf, inp["cats"], "FSC group"
    )
    if baseline_fit["gap"] > 1e-9:
        raise SystemExit(
            f"FAIL: {scheme} baseline county/shock gap {baseline_fit['gap']:.3e}"
        )

    rows = []
    raw_z = inp["d"]["z"].to_numpy(np.float64)
    for group in groups:
        keep = shock_groups != group
        sf_minus = subset_shock_frame(sf, keep)
        z_minus = raw_z - contributions[:, group_index[group]]
        inp_minus = dict(inp)
        inp_minus["zt"] = bhj.op.w22.demean(
            z_minus[:, None], inp["fe"], inp["nofe"], inp["w"]
        )[:, 0]
        fit = bhj.rung_fit(inp_minus, sf_minus, 0)
        if fit["gap"] > 1e-9:
            raise SystemExit(
                f"FAIL: {scheme} omit {group} county/shock gap {fit['gap']:.3e}"
            )
        se, clusters = bhj.exposure_robust(
            fit, sf_minus, inp["cats"], "FSC group"
        )
        first_stage_f = bhj.first_stage_F(
            fit, sf_minus, inp["cats"], "FSC group"
        )
        rows.append({
            "h": H,
            "county_weight": scheme,
            "omitted_fsc_group": group,
            "omitted_categories": int(np.sum(category_groups == group)),
            "omitted_shock_cells": int(np.sum(~keep)),
            "beta": fit["beta"],
            "beta_x1e5": fit["beta"] * locked.SCALE,
            "change_x1e5": (
                fit["beta"] - baseline_fit["beta"]
            ) * locked.SCALE,
            "absolute_change_x1e5": abs(
                fit["beta"] - baseline_fit["beta"]
            ) * locked.SCALE,
            "bhj_iv_se_x1e5": se * locked.SCALE,
            "bhj_first_stage_F": first_stage_f,
            "remaining_fsc_groups": clusters,
            "equivalence_gap": fit["gap"],
        })

    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include=[np.number])
    if not np.isfinite(numeric).all().all():
        raise SystemExit(f"FAIL: {scheme} leave-one-group output is non-finite")
    most = result.nlargest(1, "absolute_change_x1e5").iloc[0]
    weakest = result.nsmallest(1, "bhj_first_stage_F").iloc[0]
    summary = {
        "h": H,
        "county_weight": scheme,
        "baseline_beta_x1e5": baseline_fit["beta"] * locked.SCALE,
        "baseline_bhj_se_x1e5": baseline_se * locked.SCALE,
        "baseline_bhj_first_stage_F": baseline_f,
        "baseline_fsc_groups": baseline_clusters,
        "minimum_beta_x1e5": result["beta_x1e5"].min(),
        "maximum_beta_x1e5": result["beta_x1e5"].max(),
        "maximum_absolute_change_x1e5": most["absolute_change_x1e5"],
        "most_influential_fsc_group": most["omitted_fsc_group"],
        "beta_without_most_influential_x1e5": most["beta_x1e5"],
        "minimum_bhj_first_stage_F": weakest["bhj_first_stage_F"],
        "weakest_first_stage_omitted_fsc_group": weakest["omitted_fsc_group"],
        "maximum_equivalence_gap": result["equivalence_gap"].max(),
    }
    return result, summary


def write_table(summary: pd.DataFrame, result: pd.DataFrame) -> Path:
    """Write the leave-one-group-out summary and five largest movements."""
    summary_rows = []
    top_rows = []
    labels = {
        "equal": "Equal county weights",
        "population": "Fixed-1975 population weights",
    }
    for row in summary.itertuples():
        summary_rows.append(
            f"{labels[row.county_weight]} & {row.baseline_beta_x1e5:.3f} "
            f"& [{row.minimum_beta_x1e5:.3f}, {row.maximum_beta_x1e5:.3f}] "
            f"& {row.maximum_absolute_change_x1e5:.3f} "
            f"& {row.most_influential_fsc_group} "
            f"& {row.minimum_bhj_first_stage_F:.2f} \\\\"
        )
        top_rows.append(
            rf"\multicolumn{{6}}{{l}}{{\textit{{{labels[row.county_weight]}}}}} \\"
        )
        top = result.loc[
            result["county_weight"].eq(row.county_weight)
        ].nlargest(5, "absolute_change_x1e5")
        for item in top.itertuples():
            top_rows.append(
                f"\\hspace{{1em}}{item.omitted_fsc_group} "
                f"& {item.omitted_categories:d} "
                f"& {item.beta_x1e5:.3f} "
                f"& {item.change_x1e5:.3f} "
                f"& ({item.bhj_iv_se_x1e5:.3f}) "
                f"& {item.bhj_first_stage_F:.2f} \\\\"
            )

    target = OUT / "Table_bhj_leave_one_fsc_group_out_h2.tex"
    target.write_text(
        """\\begin{table}[!htbp]
\\caption{Influence of FSC Groups in the Two-Year Composition Instrument}
\\label{tab:bhj_leave_one_fsc_group_out_h2}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrrrrr}
\\toprule
\\multicolumn{6}{l}{\\textit{Panel A. Summary across all leave-one-group-out estimates}} \\\\
County weights & Baseline $\\hat\\beta\\times10^5$ & Leave-out range
& Largest $|\\Delta\\hat\\beta|\\times10^5$ & Omitted group & Minimum $F$ \\\\
\\midrule
"""
        + "\n".join(summary_rows)
        + """
\\addlinespace
\\multicolumn{6}{l}{\\textit{Panel B. Five omissions producing the largest coefficient movements}} \\\\
Omitted group & Categories & $\\hat\\beta\\times10^5$ &
$\\Delta\\hat\\beta\\times10^5$ & BHJ s.e. & First-stage $F$ \\\\
\\midrule
"""
        + "\n".join(top_rows)
        + """
\\bottomrule
\\end{tabular}}
\\par}
\\medskip
{\\scriptsize \\noindent\\textit{Notes:} The sample and specification match the
compact two-year log employment regression without a total-exposure control.
Each estimate removes every procurement category in one FSC group, rebuilds
the population-scaled county instrument, and residualizes it using the same
division-by-target-year fixed effects. Standard errors and first-stage
statistics use BHJ shock-level inference clustered by the remaining FSC
groups. Population weights use county population fixed in 1975.\\par}
\\end{table}
""",
        encoding="utf-8",
    )
    return target


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bhj = load_module(PIPELINE / "22_bhj_assumption1.py", "bhj_assumption1")
    diagnostic = load_module(
        PIPELINE / "22d_bhj_incomplete_share_checks_h2.py",
        "bhj_incomplete_share_checks_h2",
    )
    base = bhj.build_inputs(H)
    add_log_outcome(base)

    results = []
    summaries = []
    for scheme in ("equal", "population"):
        result, summary = run_scheme(base, bhj, diagnostic, scheme)
        results.append(result)
        summaries.append(summary)
    result = pd.concat(results, ignore_index=True)
    summary = pd.DataFrame(summaries)

    result_path = OUT / "bhj_leave_one_fsc_group_out_h2.csv"
    summary_path = OUT / "bhj_leave_one_fsc_group_out_summary_h2.csv"
    result.to_csv(result_path, index=False)
    summary.to_csv(summary_path, index=False)
    table_path = write_table(summary, result)
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {table_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
