#!/usr/bin/env python3
"""Prepare and report the h=2 BHJ hierarchical shock-correlation diagnostic.

Preparation constructs the live category-year shock panel and reports
exposure-robust inference under category, FSC-group, and year clustering. A
companion Stata producer estimates the Gaussian hierarchical random-effects
model. Finalization combines both outputs in a LaTeX table.

Outputs:
    data/intermediate/bhj_hierarchical_shocks_h2.dta
    Results/pipeline_diagnostics/bhj_hierarchical_cluster_inference_h2.csv
    Results/pipeline_diagnostics/Table_bhj_hierarchical_shock_correlation_h2.tex
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
MACRO = PIPELINE.parents[2]
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402
import store  # noqa: E402


H = 2
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"
INTERMEDIATE = MACRO / "data" / "intermediate"
SHOCK_DATA = INTERMEDIATE / "bhj_hierarchical_shocks_h2.dta"
INFERENCE_CSV = OUT / "bhj_hierarchical_cluster_inference_h2.csv"
ICC_CSV = OUT / "bhj_hierarchical_shock_icc_h2.csv"


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


def fsc_group(code: str) -> str:
    """Map a four-character category to its numeric or letter-leading group."""
    value = str(code).strip().upper()
    return value[0] if value[0].isalpha() else value[:2]


def effective_clusters(index: np.ndarray, importance: np.ndarray) -> float:
    """Return the inverse HHI of cluster-summed absolute importance."""
    totals = np.bincount(index, weights=np.abs(importance))
    totals = totals[totals > 0]
    shares = totals / totals.sum()
    return float(1.0 / np.sum(shares * shares))


def prepare() -> None:
    """Write the live shock panel and cluster-level inference comparison."""
    OUT.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    bhj = load_module(PIPELINE / "22_bhj_assumption1.py", "bhj_assumption1")
    diagnostic = load_module(
        PIPELINE / "22d_bhj_incomplete_share_checks_h2.py",
        "bhj_incomplete_share_checks_h2",
    )
    base = bhj.build_inputs(H)
    add_log_outcome(base)

    shock_frames = []
    inference_rows = []
    for scheme in ("equal", "population"):
        inp = diagnostic.make_input(base, bhj, scheme)
        sf = bhj.shock_frame(inp)
        fit = bhj.rung_fit(inp, sf, 0)
        categories = np.asarray(inp["cats"], dtype=str)[sf["keys"][:, 0]]
        years = np.asarray(inp["years"], dtype=np.int64)[sf["keys"][:, 1]]
        shocks = pd.DataFrame({
            "category": categories,
            "year": years,
            "shock": sf["gv"],
            f"importance_{scheme}": sf["sv"],
        })
        shock_frames.append(shocks)

        for level, label in (
            ("category", "Four-character category"),
            ("FSC group", "FSC group"),
            ("year", "Target year"),
        ):
            se, cluster_count = bhj.exposure_robust(
                fit, sf, inp["cats"], level
            )
            first_stage_f = bhj.first_stage_F(
                fit, sf, inp["cats"], level
            )
            cluster_index = bhj.cluster_index(sf, inp["cats"], level)
            inference_rows.append({
                "h": H,
                "county_weight": scheme,
                "cluster_level": level,
                "cluster_label": label,
                "beta": fit["beta"],
                "bhj_iv_se": se,
                "bhj_first_stage_F": first_stage_f,
                "clusters": cluster_count,
                "effective_clusters": effective_clusters(
                    cluster_index, sf["sv"]
                ),
                "equivalence_gap": fit["gap"],
            })

    shock_panel = shock_frames[0].merge(
        shock_frames[1], on=["category", "year", "shock"],
        how="outer", validate="one_to_one",
    )
    if shock_panel.isna().any().any():
        raise SystemExit("FAIL: live shock cells differ across county weights")
    shock_panel["fsc_group"] = shock_panel["category"].map(fsc_group)
    shock_panel["category_id"] = pd.factorize(shock_panel["category"])[0] + 1
    shock_panel["fsc_group_id"] = pd.factorize(shock_panel["fsc_group"])[0] + 1
    if shock_panel.groupby("category")["fsc_group"].nunique().max() != 1:
        raise SystemExit("FAIL: procurement category belongs to multiple groups")
    if len(shock_panel) != 27_446:
        raise SystemExit(
            f"FAIL: expected 27,446 live shocks, found {len(shock_panel):,}"
        )
    shock_panel.sort_values(["fsc_group", "category", "year"]).to_stata(
        SHOCK_DATA, write_index=False, version=118
    )

    inference = pd.DataFrame(inference_rows)
    if inference["equivalence_gap"].max() > 1e-9:
        raise SystemExit("FAIL: county and shock coefficients diverge")
    if not np.isfinite(inference.select_dtypes(include=[np.number])).all().all():
        raise SystemExit("FAIL: non-finite cluster-inference result")
    inference.to_csv(INFERENCE_CSV, index=False)
    print(f"wrote {SHOCK_DATA}")
    print(f"wrote {INFERENCE_CSV}")
    print(inference.to_string(index=False))


def write_table(icc: pd.DataFrame, inference: pd.DataFrame) -> None:
    """Write ICC estimates and the corresponding cluster-inference comparison."""
    icc_rows = []
    for row in icc.itertuples():
        se_text = "n/a" if pd.isna(row.model_se) else f"({row.model_se:.3f})"
        icc_rows.append(
            f"\\hspace{{1em}}{row.component_label} "
            f"& {row.icc:.3f} & {se_text} "
            f"& {row.variance_component:.5f} \\\\"
        )

    inference_rows = []
    for scheme, heading in (
        ("equal", "Equal county weights"),
        ("population", "Fixed-1975 population weights"),
    ):
        inference_rows.append(
            rf"\multicolumn{{7}}{{l}}{{\textit{{{heading}}}}} " + r"\\"
        )
        for row in inference.loc[
            inference["county_weight"].eq(scheme)
        ].itertuples():
            inference_rows.append(
                f"\\hspace{{1em}}{row.cluster_label} "
                f"& {row.beta * locked.SCALE:.3f} "
                f"& ({row.bhj_iv_se * locked.SCALE:.3f}) "
                f"& {row.bhj_first_stage_F:.2f} "
                f"& {int(row.clusters):,} "
                f"& {row.effective_clusters:.1f} & \\\\"
            )
        if scheme == "equal":
            inference_rows.append(r"\addlinespace")

    target = OUT / "Table_bhj_hierarchical_shock_correlation_h2.tex"
    target.write_text(
        """\\begin{table}[!htbp]
\\caption{Hierarchical Correlation of Procurement Shocks}
\\label{tab:bhj_hierarchical_shock_correlation_h2}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrrrrrr}
\\toprule
\\multicolumn{7}{l}{\\textit{Panel A. Shock intra-class correlations}} \\\\
Component & ICC & Model s.e. & Variance & & & \\\\
\\midrule
"""
        + "\n".join(icc_rows)
        + """
\\addlinespace
\\multicolumn{7}{l}{\\textit{Panel B. Exposure-robust inference by cluster level}} \\\\
Cluster level & $\\hat\\beta\\times10^5$ & BHJ s.e. & First-stage $F$
& Clusters & Effective clusters & \\\\
\\midrule
"""
        + "\n".join(inference_rows)
        + """
\\bottomrule
\\end{tabular}}
\\par}
\\medskip
{\\scriptsize \\noindent\\textit{Notes:} Panel A estimates the hierarchical
Gaussian random-effects model on 27,446 live category-year shocks by maximum
likelihood. Target-year fixed effects are included. FSC-group effects vary by
target year with an exchangeable covariance structure; four-character
category effects are constant across target years. Stata could not compute a
score-based variance matrix for this 37-period model, so Panel A reports point
estimates without standard errors. Panel B reports the compact two-year log
employment specification without a total-exposure control. BHJ inference is
computed from the equivalent shock-level IV regression under the indicated
cluster partition. Effective clusters are inverse Herfindahl indices of
cluster-summed absolute shock importance. Population weights use county
population fixed in 1975.\\par}
\\end{table}
""",
        encoding="utf-8",
    )
    print(f"wrote {target}")


def finalize() -> None:
    """Read the Stata ICC estimates and write the combined table."""
    if not ICC_CSV.exists():
        raise SystemExit(f"FAIL: missing Stata ICC output {ICC_CSV}")
    if not INFERENCE_CSV.exists():
        raise SystemExit(f"FAIL: missing inference output {INFERENCE_CSV}")
    icc = pd.read_csv(ICC_CSV)
    inference = pd.read_csv(INFERENCE_CSV)
    required = {
        "component_label", "icc", "model_se", "variance_component",
    }
    if not required.issubset(icc.columns):
        raise SystemExit("FAIL: Stata ICC output has an unexpected schema")
    finite_columns = ["icc", "variance_component"]
    if not np.isfinite(icc[finite_columns]).all().all():
        raise SystemExit("FAIL: non-finite ICC result")
    write_table(icc, inference)
    print(icc.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize()
    else:
        prepare()


if __name__ == "__main__":
    main()
