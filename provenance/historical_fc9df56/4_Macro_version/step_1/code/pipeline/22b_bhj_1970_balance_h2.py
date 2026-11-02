#!/usr/bin/env python3
"""BHJ shock-level balance at h=2 using fixed 1970 characteristics.

Every county characteristic is measured once in the 1970 Census cross-section
and held fixed across target years. Each characteristic is tested in a separate
BHJ falsification regression on the national category shock, with shock-year
effects, shock-importance weights, and category-clustered standard errors.

Outputs in Results/pipeline_diagnostics/:
    Table_bhj_balance_1970_h2.tex
    Table_bhj_balance_1970_h2_population_weighted.tex
    Table_bhj_balance_1970_h2_weights.tex
    bhj_balance_1970_h2.csv
    bhj_balance_1970_h2_population_weighted.csv
"""

from __future__ import annotations

import argparse
import importlib.util
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
CHAR_YEAR = 1970
CHAR_PATH = (
    dc.DROPBOX_PROJECT_ROOT
    / "4_Macro_version"
    / "data"
    / "temp"
    / "county_characteristics_1970"
    / "county_characteristics_1970_ccg.parquet"
)
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"
NAMES = [
    "income per head",
    "employment rate",
    "Unemployment rate",
    "log population",
    "Female civilian labor-force share",
    "Manufacturing share",
    "Non-tradable proxy share",
    "Government employment share",
    "Rural share",
    "Age 65+ share",
    "Black share",
    "Foreign-born share",
    "High-school completion share, age 25+",
    "College share, age 25+",
    "Family poverty rate",
    "Homeownership rate",
]


def load_bhj():
    spec = importlib.util.spec_from_file_location(
        "bhj_assumption1", PIPELINE / "22_bhj_assumption1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_population_weights(d: pd.DataFrame) -> np.ndarray:
    """Return county population fixed in the locked 1975 base year."""
    panel = store.load_county_panel()
    p0 = (panel.loc[panel["year"].eq(locked.POP_BASE_YEAR), ["ccg", "P"]]
          .drop_duplicates("ccg").set_index("ccg")["P"])
    weights = d["ccg"].map(p0).to_numpy(np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise SystemExit("FAIL: fixed 1975 population weights are not positive")
    return weights / weights.mean()


def fixed_characteristics(inp: dict) -> dict[str, np.ndarray]:
    if not CHAR_PATH.exists():
        raise SystemExit(
            f"FAIL: {CHAR_PATH} is absent; run "
            "19_ingest_1970_county_characteristics.py"
        )
    chars = pd.read_parquet(CHAR_PATH)
    chars["ccg"] = chars["ccg_id"].astype(int)
    if not chars["ccg"].is_unique:
        raise SystemExit("FAIL: the 1970 characteristic file is not unique by CCG")
    key = pd.DataFrame({"ccg": inp["d"]["ccg"].astype(int).to_numpy()})
    merged = key.merge(chars[["ccg", *NAMES]], on="ccg", how="left", validate="m:1")
    if merged[NAMES].isna().any().any():
        bad = int(merged[NAMES].isna().any(axis=1).sum())
        raise SystemExit(f"FAIL: {bad:,} county-year rows lack 1970 characteristics")
    return {name: merged[name].to_numpy(np.float64) for name in NAMES}


def estimate(inp: dict, sf: dict, values: dict[str, np.ndarray], bhj):
    w, fe, nofe = inp["w"], inp["fe"], inp["nofe"]
    g, s = sf["gv"], sf["sv"]
    cluster = bhj.cluster_index(sf, inp["cats"], "category")

    mean_g = np.average(g, weights=s)
    sd_g = np.sqrt(np.average((g - mean_g) ** 2, weights=s))
    if not np.isfinite(sd_g) or sd_g <= 0:
        raise SystemExit("FAIL: the national shock has zero variation")
    standardized_g = (g - mean_g) / sd_g
    shock_year = sf["keys"][:, 1]
    years = np.unique(shock_year)
    year_fe = (shock_year[:, None] == years[None, :]).astype(np.float64)
    design = np.column_stack([standardized_g, year_fe])

    rows = []
    for name in NAMES:
        v = values[name]
        residual = bhj.op.w22.demean(v.reshape(-1, 1), fe, nofe, w)[:, 0]
        numerator = bhj.aggregate(inp, sf, residual)
        denominator = bhj.aggregate(inp, sf, np.ones(len(v), np.float64))
        weighted = np.where(
            np.abs(denominator) > 0,
            numerator / np.where(denominator == 0, 1.0, denominator),
            np.nan,
        )
        live = np.isfinite(weighted)
        if not live.all():
            raise SystemExit(f"FAIL: {name} is undefined for some shock cells")
        beta, se, _ = bhj.wls_shock(weighted, design, s, cluster)
        n_clusters = int(np.unique(cluster).size)
        df = max(n_clusters - 1, 1)
        t = beta[0] / se[0]
        rows.append(
            {
                "h": H,
                "characteristic_year": CHAR_YEAR,
                "characteristic": name,
                "coefficient": beta[0],
                "se": se[0],
                "t": t,
                "p": 2.0 * stats.t.sf(abs(t), df),
                "county_years": len(inp["d"]),
                "counties": inp["d"]["ccg"].nunique(),
                "shock_cells": len(g),
                "category_clusters": n_clusters,
            }
        )
    return pd.DataFrame(rows)


def esc(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_")


def write_outputs(result: pd.DataFrame, weight: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if weight == "equal" else "_population_weighted"
    csv = OUT / f"bhj_balance_1970_h2{suffix}.csv"
    result.to_csv(csv, index=False)

    body = []
    for row in result.itertuples(index=False):
        body.append(
            f"\\hspace{{1em}}{esc(row.characteristic)} & {row.coefficient:.4f} "
            f"& ({row.se:.4f}) & {row.t:.2f} & {row.p:.3f} \\\\"
        )
    first = result.iloc[0]
    body.extend(
        [
            r"\midrule",
            rf"Shock-level observations & {int(first.shock_cells):,} & & & \\ ",
            rf"Category clusters & {int(first.category_clusters):,} & & & \\ ",
        ]
    )
    tex = OUT / f"Table_bhj_balance_1970_h2{suffix}.tex"
    label_suffix = "" if weight == "equal" else "_population_weighted"
    weight_note = (
        "equal county weights"
        if weight == "equal"
        else "county population fixed in 1975"
    )
    tex.write_text(
        """\\begin{table}[!htbp]
\\caption{Shock-Level Balance with 1970 County Characteristics}
\\label{tab:bhj_balance_1970_h2"""
        + label_suffix
        + """}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrrrr}
\\toprule
Characteristic & Coef. & (s.e.) & $t$ & $p$ \\\\
\\midrule
"""
        + "\n".join(body)
        + """
\\bottomrule
\\end{tabular}}
\\par}
\\medskip
{\\scriptsize \\noindent\\textit{Notes:} The horizon is two years. Every county
characteristic is fixed at the 1970 Census cross-section and repeated across
target years. Each row reports a separate BHJ falsification regression. The
dependent variable is the exposure-weighted county characteristic and the
regressor is the national procurement shock for a Federal Supply Classification
category, normalized to have unit variance. County characteristics are
residualized on division-by-year effects and aggregated to category-year cells
using the counties' exposure shares and """
        + weight_note
        + """, and standardized. Regressions
include shock-year effects, use shock-importance weights, and cluster standard
errors by category. Income is money income in 1969, as reported by the 1970
Census. The family poverty rate is the share of families below the 1969 Census
low-income threshold. The female labor-force measure is the female share of the
civilian labor force. The non-tradable proxy sums employment in wholesale and
retail trade, business, repair, and personal services, educational services, and
construction. The foreign-born share is the exact foreign-born population
divided by total population in the 1970 Census Fourth Count File 4C.
The high-school completion share is the fraction of people age 25 and older who
completed at least four years of high school. The homeownership rate is the
fraction of occupied housing units that are owner-occupied.
No characteristic is a growth rate.\\par}
\\end{table}
""",
        encoding="utf-8",
    )
    print(f"wrote {csv}")
    print(f"wrote {tex}")


def write_combined(equal: pd.DataFrame, population: pd.DataFrame) -> None:
    """Write one transparent comparison of equal and population weights."""
    keys = ["characteristic", "coefficient", "se", "p"]
    merged = equal[keys].merge(
        population[keys],
        on="characteristic",
        suffixes=("_equal", "_population"),
        validate="1:1",
    )
    body = []
    for row in merged.itertuples(index=False):
        body.append(
            f"\\hspace{{1em}}{esc(row.characteristic)} "
            f"& {row.coefficient_equal:.4f} & ({row.se_equal:.4f}) "
            f"& {row.p_equal:.3f} & {row.coefficient_population:.4f} "
            f"& ({row.se_population:.4f}) & {row.p_population:.3f} \\\\"
        )
    equal_first = equal.iloc[0]
    population_first = population.iloc[0]
    body.extend(
        [
            r"\midrule",
            rf"Shock-level observations & {int(equal_first.shock_cells):,} & & & "
            rf"{int(population_first.shock_cells):,} & & \\ ",
            rf"Category clusters & {int(equal_first.category_clusters):,} & & & "
            rf"{int(population_first.category_clusters):,} & & \\ ",
        ]
    )
    target = OUT / "Table_bhj_balance_1970_h2_weights.tex"
    target.write_text(
        """\\begin{table}[!htbp]
\\caption{BHJ Falsification Tests with 1970 County Characteristics}
\\label{tab:bhj_balance_1970_h2_weights}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrrrrrr}
\\toprule
& \\multicolumn{3}{c}{Equal county weights}
& \\multicolumn{3}{c}{Population weights} \\\\
\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}
Characteristic & Coef. & (s.e.) & $p$ & Coef. & (s.e.) & $p$ \\\\
\\midrule
"""
        + "\n".join(body)
        + """
\\bottomrule
\\end{tabular}}
\\par}
\\medskip
{\\scriptsize \\noindent\\textit{Notes:} The horizon is two years. Each row
reports a separate BHJ falsification regression of the exposure-weighted 1970
county characteristic on the national procurement shock, controlling for
shock-year effects. The shock is normalized to have unit variance. Regressions
use shock-importance weights and cluster standard errors by Federal Supply
Classification category. The two panels differ only in the county weights used
to construct the shock-level variables. Population weights use county
population fixed in 1975 and constant across target years. High-school
completion means at least
four years of high school among people age 25 and older.
Homeownership is the owner-occupied share of occupied housing units. No omnibus
joint test is reported.\\par}
\\end{table}
""",
        encoding="utf-8",
    )
    print(f"wrote {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight",
        choices=("equal", "population", "both"),
        default="both",
        help="county weight within each division-year",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bhj = load_bhj()
    base = bhj.build_inputs(H)
    values = fixed_characteristics(base)
    requested = ("equal", "population") if args.weight == "both" else (args.weight,)
    results = {}
    for weight in requested:
        inp = dict(base)
        if weight == "equal":
            inp["w"] = np.ones(len(inp["d"]), np.float64)
        else:
            inp["w"] = fixed_population_weights(inp["d"])
        sf = bhj.shock_frame(inp)
        result = estimate(inp, sf, values, bhj)
        result["county_weight"] = weight
        write_outputs(result, weight)
        results[weight] = result
        first = result.iloc[0]
        print(
            f"h={H}, weight={weight}, counties={int(first.counties):,}, "
            f"county-years={int(first.county_years):,}, "
            f"shock cells={int(first.shock_cells):,}"
        )
        print(
            result[["characteristic", "coefficient", "se", "t", "p"]]
            .to_string(index=False)
        )
    if args.weight == "both":
        write_combined(results["equal"], results["population"])


if __name__ == "__main__":
    main()
