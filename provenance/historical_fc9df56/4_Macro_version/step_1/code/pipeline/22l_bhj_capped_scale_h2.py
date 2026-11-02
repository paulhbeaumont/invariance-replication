#!/usr/bin/env python3
"""BHJ diagnostics for the house-capped procurement-intensity scale at h=2.

The locked instrument factors as

    z_ct = E_ct * sum_k s_ckt q_kt,

where E_ct is predetermined procurement dollars per resident and the category
shares s_ckt sum to one when exposure is positive. This diagnostic applies the
house within-target-year 1st/99th percentile winsorization to E_ct, leaves the
composition shares and national shocks unchanged, and estimates the capped
instrument both without and with its exact exposure-sum-by-year control.

Outputs in Results/pipeline_diagnostics/:
    bhj_capped_scale_h2.csv
    bhj_capped_scale_checks_h2.csv
    Table_bhj_capped_scale_h2.tex
"""
from __future__ import annotations

import importlib.util
import json
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
import panel as pnl_lib  # noqa: E402
import store  # noqa: E402


H = 2
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"
BASELINE_ID = "verified_production_geography_v1"
FACTOR_TOL = 1e-12


def load_module(path: Path, name: str):
    """Load one canonical pipeline producer without running its main block."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_active_baseline() -> None:
    """Refuse to run outside the verified production-geography baseline."""
    manifest_path = store.PIPE / "_golden" / "active_baseline.json"
    if not manifest_path.exists():
        raise SystemExit(f"FAIL: active baseline manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("baseline_id") != BASELINE_ID:
        raise SystemExit(f"FAIL: this diagnostic requires {BASELINE_ID}")


def population_matrix(base: dict) -> np.ndarray:
    """Return base-year population by county and target year."""
    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("P",))
    if not np.array_equal(units, base["units"]):
        raise SystemExit("FAIL: population and exposure county orders differ")
    population = pivots["P"].reindex(
        index=units,
        columns=[year - H for year in base["years"]],
    ).set_axis(base["years"], axis=1).to_numpy(np.float64)
    if not np.isfinite(population).all() or (population <= 0).any():
        raise SystemExit("FAIL: base-year population is missing or non-positive")
    return population


def capped_exposures(base: dict) -> tuple[dict, pd.DataFrame]:
    """Winsorize procurement per resident and preserve category composition."""
    population = population_matrix(base)
    capped, checks = {}, []
    for j, values in base["cbar"].items():
        raw = values.astype(np.float64)
        total = raw.sum(axis=1)
        intensity = total / population[:, j]
        capped_intensity = pnl_lib.wins(
            intensity, np.zeros(len(intensity), dtype=np.int8)
        )
        positive = intensity > 0
        factor = np.zeros_like(intensity)
        factor[positive] = capped_intensity[positive] / intensity[positive]
        transformed = raw * factor[:, None]
        capped[j] = transformed

        transformed_intensity = transformed.sum(axis=1) / population[:, j]
        target = np.where(positive, capped_intensity, 0.0)
        scale_gap = float(np.max(np.abs(transformed_intensity - target))) / max(
            float(np.max(np.abs(target))), 1e-300
        )
        composition_gap = 0.0
        if positive.any():
            original_share = raw[positive] / total[positive, None]
            transformed_total = transformed[positive].sum(axis=1)
            transformed_share = transformed[positive] / transformed_total[:, None]
            composition_gap = float(
                np.max(np.abs(original_share - transformed_share))
            )
        checks.append({
            "h": H,
            "target_year": int(base["years"][j]),
            "positive_county_exposures": int(positive.sum()),
            "zero_county_exposures": int((~positive).sum()),
            "lower_tail_adjustments": int(
                np.sum(positive & (capped_intensity > intensity))
            ),
            "upper_tail_adjustments": int(
                np.sum(capped_intensity < intensity)
            ),
            "raw_p01": float(np.percentile(intensity, 1.0)),
            "raw_p99": float(np.percentile(intensity, 99.0)),
            "maximum_relative_scale_reconstruction_gap": scale_gap,
            "maximum_composition_share_gap": composition_gap,
        })
    check_frame = pd.DataFrame(checks)
    if check_frame[
        ["maximum_relative_scale_reconstruction_gap",
         "maximum_composition_share_gap"]
    ].to_numpy().max() > FACTOR_TOL:
        raise SystemExit("FAIL: capped-scale exposure construction does not tie")
    return capped, check_frame


def leverage_statistics(inp: dict) -> tuple[float, float]:
    """Compute county concentration in within-cell instrument variation."""
    county = pd.factorize(inp["d"]["ccg"])[0].astype(np.int64)
    leverage = np.bincount(
        county,
        weights=inp["w"] * inp["zt"] * inp["zt"],
        minlength=int(county.max()) + 1,
    )
    total = float(leverage.sum())
    if total <= 0:
        raise SystemExit("FAIL: instrument has no within-cell variation")
    shares = leverage / total
    return float(1.0 / np.sum(shares**2)), float(shares.max())


def estimate_row(inp: dict, bhj, diagnostic, rung: int, label: str,
                 instrument: str, scheme: str) -> dict:
    """Estimate one specification and attach scale-concentration diagnostics."""
    row = diagnostic.estimate(inp, bhj, rung, label, scheme)
    effective, maximum = leverage_statistics(inp)
    row.update({
        "instrument_definition": instrument,
        "control_definition": (
            "exposure sum by target year" if rung == 1 else "none"
        ),
        "effective_counties": effective,
        "maximum_county_leverage": maximum,
    })
    return row


def write_table(result: pd.DataFrame) -> None:
    """Write the locked-versus-capped BHJ comparison."""
    lines = []
    for panel_index, (scheme, heading) in enumerate((
        ("equal", "Panel A. Equal county weights"),
        ("population", "Panel B. Fixed-1975 population weights"),
    )):
        lines.append(
            rf"\multicolumn{{11}}{{l}}{{\textit{{{heading}}}}} " + r"\\"
        )
        for row in result.loc[result["county_weight"].eq(scheme)].itertuples():
            lines.append(
                rf"\hspace{{1em}}{row.specification} "
                f"& {row.county_reduced_form * locked.SCALE:.3f} "
                f"& ({row.bhj_reduced_form_se * locked.SCALE:.3f}) "
                f"& {row.beta * locked.SCALE:.3f} "
                f"& ({row.bhj_iv_se * locked.SCALE:.3f}) "
                f"& {row.county_first_stage_F:.2f} "
                f"& {row.bhj_first_stage_F:.2f} "
                f"& {row.instrument_variance_removed * 100:.1f}\\% "
                f"& {row.effective_counties:.1f} "
                f"& {row.maximum_county_leverage:.3f} "
                f"& {int(row.county_years):,} " + r"\\"
            )
        if panel_index == 0:
            lines.append(r"\addlinespace")

    target = OUT / "Table_bhj_capped_scale_h2.tex"
    target.write_text(
        """\\begin{table}[!htbp]
\\caption{Capped Procurement Intensity and the Exact BHJ Control}
\\label{tab:bhj_capped_scale_h2}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrrrrrrrrrr}
\\toprule
Specification & RF$\\times10^5$ & BHJ s.e. & 2SLS$\\times10^5$ & BHJ s.e.
& County $F$ & BHJ $F$ & $z$ var. removed & Effective counties
& Maximum leverage & $N$ \\\\
\\midrule
"""
        + "\n".join(lines)
        + """
\\bottomrule
\\end{tabular}}
\\par}
\\medskip
{\\scriptsize \\noindent\\textit{Notes:} The locked instrument equals
predetermined procurement dollars per resident times the normalized-composition
shock. The capped instrument winsorizes procurement dollars per resident at
the 1st and 99th percentiles within target year using the house rule, while
leaving normalized category shares and national category shocks unchanged.
The corresponding exact BHJ control interacts each instrument's exposure sum
with target-year indicators. All rows use two-year log employment growth, the
unwinsorized two-year procurement treatment, division-by-target-year fixed
effects, and BHJ inference clustered by four-character procurement category.
Effective counties is the inverse Herfindahl index of county contributions to
within-cell instrument variation. Population weights use county population
fixed in 1975.\\par}
\\end{table}
""",
        encoding="utf-8",
    )
    print(f"wrote {target}")


def verify_locked_rows(result: pd.DataFrame) -> None:
    """Tie the locked comparison rows to the certified BHJ diagnostic output."""
    reference_path = OUT / "bhj_incomplete_share_checks_h2.csv"
    if not reference_path.exists():
        raise SystemExit(f"FAIL: certified BHJ reference is absent: {reference_path}")
    reference = pd.read_csv(reference_path)
    columns = (
        "beta", "bhj_iv_se", "county_first_stage_F", "bhj_first_stage_F",
        "county_reduced_form", "bhj_reduced_form_se",
        "instrument_variance_removed",
    )
    for scheme in ("equal", "population"):
        for controlled in (False, True):
            label = (
                "Locked scale, exposure sum by year"
                if controlled else "Locked scale, no control"
            )
            specification = (
                "Current exposure, share-sum by year"
                if controlled else "Current exposure"
            )
            actual = result.loc[
                result["county_weight"].eq(scheme)
                & result["specification"].eq(label)
            ]
            expected = reference.loc[
                reference["county_weight"].eq(scheme)
                & reference["specification"].eq(specification)
            ]
            if len(actual) != 1 or len(expected) != 1:
                raise SystemExit("FAIL: locked BHJ tie-out row is not unique")
            for column in columns:
                if not np.isclose(
                    float(actual.iloc[0][column]),
                    float(expected.iloc[0][column]),
                    rtol=1e-11,
                    atol=1e-14,
                ):
                    raise SystemExit(
                        f"FAIL: locked BHJ tie-out differs for {scheme}, "
                        f"{specification}, {column}"
                    )


def main() -> None:
    require_active_baseline()
    OUT.mkdir(parents=True, exist_ok=True)
    bhj = load_module(PIPELINE / "22_bhj_assumption1.py", "bhj_assumption1")
    diagnostic = load_module(
        PIPELINE / "22d_bhj_incomplete_share_checks_h2.py",
        "bhj_incomplete_share_checks_h2",
    )
    normalized = load_module(
        PIPELINE / "22i_bhj_normalized_composition_h2.py",
        "bhj_normalized_composition_h2",
    )
    base = bhj.build_inputs(H)
    normalized.add_log_outcome(base)
    capped_cbar, checks = capped_exposures(base)

    rows = []
    for scheme in ("equal", "population"):
        locked_input = diagnostic.make_input(base, bhj, scheme)
        capped_input = diagnostic.make_input(
            base, bhj, scheme, cbar=capped_cbar, gk=base["gk"]
        )
        for inp, rung, label, instrument in (
            (locked_input, 0, "Locked scale, no control", "locked"),
            (locked_input, 1, "Locked scale, exposure sum by year", "locked"),
            (capped_input, 0, "Capped scale, no control", "capped"),
            (capped_input, 1, "Capped scale, exposure sum by year", "capped"),
        ):
            rows.append(estimate_row(
                inp, bhj, diagnostic, rung, label, instrument, scheme
            ))

    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include=[np.number])
    if not np.isfinite(numeric).all().all():
        raise SystemExit("FAIL: non-finite value in capped-scale results")
    if result["equivalence_gap"].max() > 1e-9:
        raise SystemExit("FAIL: county and shock estimates do not agree")
    verify_locked_rows(result)

    result_path = OUT / "bhj_capped_scale_h2.csv"
    checks_path = OUT / "bhj_capped_scale_checks_h2.csv"
    result.to_csv(result_path, index=False)
    checks.to_csv(checks_path, index=False)
    write_table(result)
    print(f"wrote {result_path}")
    print(f"wrote {checks_path}")
    print(result[[
        "county_weight", "specification", "county_reduced_form",
        "bhj_reduced_form_se", "beta", "bhj_iv_se",
        "county_first_stage_F", "bhj_first_stage_F",
        "instrument_variance_removed", "effective_counties",
        "maximum_county_leverage", "equivalence_gap",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
