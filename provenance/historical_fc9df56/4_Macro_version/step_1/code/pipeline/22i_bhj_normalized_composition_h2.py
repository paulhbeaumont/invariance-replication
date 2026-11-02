#!/usr/bin/env python3
"""Compare scaled and normalized-composition instruments at h=2.

For positive-exposure county-years, the locked instrument is decomposed as

    z_ct = I_ct B_ct,
    I_ct = sum_k Cbar_ckt / P_c,t-2,
    B_ct = sum_k omega_ckt q_kt,
    omega_ckt = Cbar_ckt / sum_j Cbar_cjt.

The comparison holds the outcome, treatment, fixed effects, and estimation
weights fixed. County-years with zero predetermined procurement have undefined
normalized shares and are excluded from the common-support comparisons.

Outputs in Results/pipeline_diagnostics/:
    bhj_normalized_composition_h2.csv
    bhj_normalized_composition_checks_h2.csv
    Table_bhj_normalized_composition_h2.tex
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


def exposure_constructions(base: dict) -> tuple[dict, dict]:
    """Build common-support dollar exposures and normalized-share exposures."""
    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("P",))
    if not np.array_equal(units, base["units"]):
        raise SystemExit("FAIL: population and exposure county orders differ")
    population = pivots["P"].reindex(
        index=units,
        columns=[year - H for year in base["years"]],
    ).set_axis(base["years"], axis=1).to_numpy(np.float64)

    common, normalized = {}, {}
    for j, values in base["cbar"].items():
        total = values.sum(axis=1)
        live = np.isfinite(total) & (total > 0) & np.isfinite(population[:, j])
        dollars = values.astype(np.float64).copy()
        dollars[~live, :] = np.nan
        omega = np.full_like(dollars, np.nan, dtype=np.float64)
        omega[live, :] = values[live, :] / total[live, None]
        common[j] = dollars
        normalized[j] = omega * population[:, j, None]
    return common, normalized


def row_keys(inp: dict) -> pd.MultiIndex:
    """Return the ordered county-year key of an estimation input."""
    return pd.MultiIndex.from_frame(inp["d"][["ccg", "year"]])


def intensity_control(pure: dict, scaled: dict, bhj) -> dict:
    """Attach the scaled instrument's procurement intensity to the pure input."""
    intensity = pd.Series(
        scaled["d"]["s0"].to_numpy(np.float64), index=row_keys(scaled)
    )
    raw = intensity.reindex(row_keys(pure)).to_numpy(np.float64)
    if not np.isfinite(raw).all():
        raise SystemExit("FAIL: normalized-share intensity control is missing")
    st = bhj.op.w22.demean(
        raw[:, None], pure["fe"], pure["nofe"], pure["w"]
    )[:, 0]
    out = dict(pure)
    out["st"] = st
    out["intensity_raw"] = raw
    return out


def estimate(inp: dict, bhj, diagnostic, rung, label: str,
             scheme: str, sample: str, instrument: str) -> dict:
    """Estimate one labeled row and retain its sample and instrument metadata."""
    row = diagnostic.estimate(inp, bhj, rung, label, scheme)
    row.update({
        "control_definition": label,
        "sample_definition": sample,
        "instrument_definition": instrument,
    })
    return row


def validate_decomposition(full: dict, scaled: dict, pure: dict) -> dict:
    """Verify common support, complete shares, and z = intensity times composition."""
    if not row_keys(scaled).equals(row_keys(pure)):
        raise SystemExit("FAIL: scaled and normalized-share samples differ")
    share_gap = float(np.max(np.abs(pure["d"]["s0"].to_numpy() - 1.0)))
    intensity = scaled["d"]["s0"].to_numpy(np.float64)
    rebuilt = intensity * pure["d"]["z"].to_numpy(np.float64)
    scaled_z = scaled["d"]["z"].to_numpy(np.float64)
    identity_gap = float(np.max(np.abs(rebuilt - scaled_z)))
    dropped = len(full["d"]) - len(pure["d"])
    if share_gap > 1e-12:
        raise SystemExit(f"FAIL: normalized-share sum gap {share_gap:.3e}")
    if identity_gap > 1e-10:
        raise SystemExit(f"FAIL: scaled-instrument identity gap {identity_gap:.3e}")
    if dropped < 0:
        raise SystemExit("FAIL: normalized-share sample exceeds full sample")
    return {
        "h": H,
        "full_county_years": len(full["d"]),
        "positive_exposure_county_years": len(pure["d"]),
        "zero_exposure_county_years_dropped": dropped,
        "dropped_share": dropped / len(full["d"]),
        "maximum_normalized_share_sum_gap": share_gap,
        "maximum_scaled_instrument_identity_gap": identity_gap,
    }


def write_table(result: pd.DataFrame, checks: pd.DataFrame) -> None:
    """Write the scaled and pure-composition IV comparison."""
    lines = []
    for panel_index, (scheme, heading) in enumerate((
        ("equal", "Panel A. Equal county weights"),
        ("population", "Panel B. Fixed-1975 population weights"),
    )):
        lines.append(
            rf"\multicolumn{{9}}{{l}}{{\textit{{{heading}}}}} " + r"\\"
        )
        for row in result.loc[result["county_weight"].eq(scheme)].itertuples():
            sample = "Full" if row.sample_definition == "full" else "Positive"
            lines.append(
                rf"\hspace{{1em}}{row.control_definition} "
                f"& {sample} "
                f"& {row.county_reduced_form * locked.SCALE:.3f} "
                f"& ({row.bhj_reduced_form_se * locked.SCALE:.3f}) "
                f"& {row.beta * locked.SCALE:.3f} "
                f"& ({row.bhj_iv_se * locked.SCALE:.3f}) "
                f"& {row.county_first_stage_F:.2f} "
                f"& {row.bhj_first_stage_F:.2f} "
                f"& {int(row.county_years):,} " + r"\\"
            )
        if panel_index == 0:
            lines.append(r"\addlinespace")

    check = checks.iloc[0]
    target = OUT / "Table_bhj_normalized_composition_h2.tex"
    target.write_text(
        """\\begin{table}[!htbp]
\\caption{Scaled and Normalized-Composition Instruments at the Two-Year Horizon}
\\label{tab:bhj_normalized_composition_h2}
{\\centering
\\adjustbox{max width=\\linewidth}{%
\\begin{tabular}{lrrrrrrrr}
\\toprule
Specification & Sample & RF$\\times10^5$ & BHJ s.e. & 2SLS$\\times10^5$
& BHJ s.e. & County $F$ & BHJ $F$ & $N$ \\\\
\\midrule
"""
        + "\n".join(lines)
        + f"""
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} The scaled instrument is the
locked instrument, equal to predetermined procurement intensity times the
normalized-composition shock. Positive denotes county-years with strictly
positive predetermined procurement across the locked categories. The pure
composition instrument uses category shares that sum to one on that sample.
The intensity control interacts the scaled instrument's predetermined total
procurement per person with target-year indicators. The treatment, two-year
log employment outcome, and division-by-year fixed effects are unchanged.
Reduced-form coefficients are not comparable across the two instrument scales;
the 2SLS coefficients retain the same outcome and treatment units.
Normalized shares are undefined for
{int(check.zero_exposure_county_years_dropped):,} of
{int(check.full_county_years):,} full-sample county-years. BHJ inference
clusters by four-character procurement category.\\par}}
\\end{{table}}
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
    common_cbar, normalized_cbar = exposure_constructions(base)

    rows, check_rows = [], []
    for scheme in ("equal", "population"):
        full = diagnostic.make_input(base, bhj, scheme)
        scaled = diagnostic.make_input(
            base, bhj, scheme, cbar=common_cbar, gk=base["gk"]
        )
        pure = diagnostic.make_input(
            base, bhj, scheme, cbar=normalized_cbar, gk=base["gk"]
        )
        pure_intensity = intensity_control(pure, scaled, bhj)
        checks = validate_decomposition(full, scaled, pure)
        checks["county_weight"] = scheme
        check_rows.append(checks)

        specifications = (
            (full, 0, "Scaled instrument, no control", "full", "scaled"),
            (scaled, 0, "Scaled instrument, no control", "positive", "scaled"),
            (scaled, 1, "Scaled instrument, intensity by year", "positive", "scaled"),
            (pure, 0, "Pure composition instrument", "positive", "pure"),
            (pure_intensity, 1, "Pure composition, intensity by year", "positive", "pure"),
        )
        for inp, rung, label, sample, instrument in specifications:
            rows.append(estimate(
                inp, bhj, diagnostic, rung, label, scheme, sample, instrument
            ))

    result = pd.DataFrame(rows)
    checks = pd.DataFrame(check_rows)
    if not np.isfinite(result.select_dtypes(include=[np.number])).all().all():
        raise SystemExit("FAIL: non-finite value in normalized-share results")
    result_path = OUT / "bhj_normalized_composition_h2.csv"
    checks_path = OUT / "bhj_normalized_composition_checks_h2.csv"
    result.to_csv(result_path, index=False)
    checks.to_csv(checks_path, index=False)
    write_table(result, checks)
    print(f"wrote {result_path}")
    print(f"wrote {checks_path}")
    print(checks.to_string(index=False))
    print(result[[
        "county_weight", "control_definition", "sample_definition",
        "county_reduced_form", "bhj_reduced_form_se", "beta", "bhj_iv_se",
        "county_first_stage_F", "bhj_first_stage_F", "county_years",
        "equivalence_gap",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
