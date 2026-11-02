#!/usr/bin/env python3
"""Compare rolling and fixed-initial total-exposure controls at h=2.

The instrument retains its locked rolling five-year category exposures. The
fixed control is average annual procurement across 1969 through 1973, summed
over the locked categories and divided by county population in 1975. This
county-level value is interacted with every target-year indicator.

Outputs in Results/pipeline_diagnostics/:
    bhj_fixed_initial_share_control_h2.csv
    Table_bhj_fixed_initial_share_control_h2.tex
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
INITIAL_YEARS = tuple(range(1969, 1974))
POPULATION_YEAR = 1975
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


def fixed_initial_exposure(base: dict) -> np.ndarray:
    """Return 1969-1973 procurement per 1975 resident for each county."""
    units, _, years, values, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, base["units"]):
        raise SystemExit("FAIL: tensor and regression county orders differ")
    positions = [years.index(year) for year in INITIAL_YEARS]
    annual_dollars = values[:, :, positions].mean(axis=2).sum(axis=1)

    panel = store.load_county_panel()
    population = (
        panel.loc[panel["year"].eq(POPULATION_YEAR), ["ccg", "P"]]
        .drop_duplicates("ccg")
        .set_index("ccg")["P"]
        .reindex(units)
        .to_numpy(np.float64)
    )
    if not np.isfinite(population).all() or (population <= 0).any():
        raise SystemExit("FAIL: 1975 population is missing or non-positive")
    exposure = annual_dollars / population
    if not np.isfinite(exposure).all():
        raise SystemExit("FAIL: fixed initial exposure is non-finite")
    return exposure


def replace_share_control(inp: dict, fixed_by_county: np.ndarray, bhj) -> dict:
    """Retain the estimation input while replacing only its share-sum control."""
    rowpos = {county: i for i, county in enumerate(inp["units"])}
    raw = inp["d"]["ccg"].map(
        {county: fixed_by_county[i] for county, i in rowpos.items()}
    ).to_numpy(np.float64)
    check = pd.DataFrame({"ccg": inp["d"]["ccg"], "fixed": raw})
    within_county_range = (
        check.groupby("ccg")["fixed"].max()
        - check.groupby("ccg")["fixed"].min()
    )
    if within_county_range.max() > 0:
        raise SystemExit("FAIL: fixed initial control varies within county")
    st = bhj.op.w22.demean(
        raw[:, None], inp["fe"], inp["nofe"], inp["w"]
    )[:, 0]
    out = dict(inp)
    out["st"] = st
    out["fixed_initial_share_raw"] = raw
    return out


def validate_current_rows(result: pd.DataFrame) -> None:
    """Tie the no-control and rolling-control rows to their existing producer."""
    reference = pd.read_csv(OUT / "bhj_incomplete_share_checks_h2.csv")
    mapping = {
        "No total-exposure control": "Current exposure",
        "Rolling total exposure by year": (
            "Current exposure, share-sum by year"
        ),
    }
    for scheme in ("equal", "population"):
        for label, specification in mapping.items():
            measured = result.loc[
                result["county_weight"].eq(scheme)
                & result["control_definition"].eq(label)
            ]
            expected = reference.loc[
                reference["county_weight"].eq(scheme)
                & reference["specification"].eq(specification)
            ]
            if len(measured) != 1 or len(expected) != 1:
                raise SystemExit("FAIL: current-control tie-out row is not unique")
            for column in (
                "beta", "bhj_iv_se", "county_reduced_form",
                "bhj_reduced_form_se", "county_first_stage_F",
                "bhj_first_stage_F", "instrument_variance_removed",
            ):
                gap = abs(
                    float(measured.iloc[0][column])
                    - float(expected.iloc[0][column])
                )
                if gap > 1e-10:
                    raise SystemExit(
                        f"FAIL: {scheme} {label} {column} tie-out gap {gap:.3e}"
                    )


def write_table(result: pd.DataFrame) -> None:
    """Write the comparison of the three total-exposure specifications."""
    lines = []
    for panel_index, (scheme, heading) in enumerate((
        ("equal", "Panel A. Equal county weights"),
        ("population", "Panel B. Fixed-1975 population weights"),
    )):
        lines.append(
            rf"\multicolumn{{9}}{{l}}{{\textit{{{heading}}}}} " + r"\\"
        )
        for row in result.loc[result["county_weight"].eq(scheme)].itertuples():
            lines.append(
                rf"\hspace{{1em}}{row.control_definition} "
                f"& {row.county_reduced_form * locked.SCALE:.3f} "
                f"& ({row.bhj_reduced_form_se * locked.SCALE:.3f}) "
                f"& {row.beta * locked.SCALE:.3f} "
                f"& ({row.bhj_iv_se * locked.SCALE:.3f}) "
                f"& {row.county_first_stage_F:.2f} "
                f"& {row.bhj_first_stage_F:.2f} "
                rf"& {100 * row.instrument_variance_removed:.1f}\% "
                f"& {int(row.control_rank):d} " + r"\\"
            )
        if panel_index == 0:
            lines.append(r"\addlinespace")

    target = OUT / "Table_bhj_fixed_initial_share_control_h2.tex"
    target.write_text(
        """\\begin{table}[!htbp]
\\caption{Rolling and Fixed-Initial Total-Exposure Controls at the Two-Year Horizon}
\\label{tab:bhj_fixed_initial_share_control_h2}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrrrrrrrr}
\\toprule
Control & RF$\\times10^5$ & BHJ s.e. & 2SLS$\\times10^5$ & BHJ s.e.
& County $F$ & BHJ $F$ & $z$ var. removed & Rank \\\\
\\midrule
"""
        + "\n".join(lines)
        + """
\\bottomrule
\\end{tabular}}
\\par}
\\medskip
{\\scriptsize \\noindent\\textit{Notes:} The dependent variable is two-year
log employment growth. The treatment and locked instrument retain their
rolling five-year construction and population denominator at $t-2$. All rows
include division-by-year fixed effects. Rolling total exposure is the sum of
the instrument's category exposures in each county and target year. Fixed
initial total exposure is average annual procurement across 1969--1973 divided
by county population in 1975; it is constant within county before interaction
with target-year indicators. The fixed-initial control is a predetermined
exposure-intensity control, not the exact incomplete-share correction for the
time-varying instrument. County inference clusters by county. BHJ inference
clusters by four-character procurement category.\\par}
\\end{table}
""",
        encoding="utf-8",
    )
    print(f"wrote {target}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bhj = load_module(PIPELINE / "22_bhj_assumption1.py", "bhj_assumption1")
    diagnostic = load_module(
        PIPELINE / "22d_bhj_incomplete_share_checks_h2.py",
        "bhj_incomplete_share_checks_h2",
    )
    base = bhj.build_inputs(H)
    add_log_outcome(base)
    fixed_by_county = fixed_initial_exposure(base)

    rows = []
    for scheme in ("equal", "population"):
        current = diagnostic.make_input(base, bhj, scheme)
        fixed = replace_share_control(current, fixed_by_county, bhj)
        specifications = (
            (current, 0, "No total-exposure control"),
            (current, 1, "Rolling total exposure by year"),
            (fixed, 1, "Fixed initial total exposure by year"),
        )
        for inp, rung, label in specifications:
            row = diagnostic.estimate(inp, bhj, rung, label, scheme)
            row["control_definition"] = label
            row["initial_exposure_first_year"] = INITIAL_YEARS[0]
            row["initial_exposure_last_year"] = INITIAL_YEARS[-1]
            row["initial_population_year"] = POPULATION_YEAR
            rows.append(row)

    result = pd.DataFrame(rows)
    if not np.isfinite(result.select_dtypes(include=[np.number])).all().all():
        raise SystemExit("FAIL: non-finite value in fixed-control results")
    validate_current_rows(result)
    target = OUT / "bhj_fixed_initial_share_control_h2.csv"
    result.to_csv(target, index=False)
    write_table(result)
    print(f"wrote {target}")
    print(result[[
        "county_weight", "control_definition", "county_reduced_form",
        "bhj_reduced_form_se", "beta", "bhj_iv_se",
        "county_first_stage_F", "bhj_first_stage_F",
        "instrument_variance_removed", "control_rank", "equivalence_gap",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
