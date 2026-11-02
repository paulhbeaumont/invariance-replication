"""Test a naturally bounded rolling-five-year h=2 category shifter.

For raw normalized change r, the category shifter is 2r / (2 + abs(r)).
It preserves the local scale of r and remains strictly between -2 and 2.

Writes Results/instrument_normalization/h2_county_smooth_bounded_shifter.csv.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
STEP1_PY = PIPELINE.parent / "py"
STEP2_PY = ROOT / "step_2" / "code" / "py"
REFERENCE_OUT = (ROOT / "Results" / "instrument_normalization"
                 / "h2_county_clipping_sensitivity.csv")
OUT = (ROOT / "Results" / "instrument_normalization"
       / "h2_county_smooth_bounded_shifter.csv")

sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import store  # noqa: E402


H = 2


def load_module(name, path):
    """Load a project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """Estimate the smooth bounded shifter on the locked county sample."""
    start = time.time()
    print("== h=2 smooth bounded rolling-five-year shifter")
    p57 = load_module("p57_smooth", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module("t28_smooth", STEP2_PY / "28_table_d1.py")
    diagnostics = load_module(
        "clip_diagnostics", PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12,
                                      fixef_maxiter=100_000)

    baseline, dropped = p57.build_county(
        "composition", h=H, span=locked.SPAN, rule=locked.RULE,
        floor=locked.FLOOR, denom=locked.DENOM)
    if dropped != 0:
        raise SystemExit(f"FAIL: baseline dropped {dropped} additional counties")
    if len(baseline) != locked.N_COUNTY[H]:
        raise SystemExit(
            f"FAIL: baseline has {len(baseline):,} rows, expected "
            f"{locked.N_COUNTY[H]:,}")

    units0, _, contract_years, arr, _ = store.load_tensor(locked.RULE)
    years0 = store.target_years(H)
    raw, live_counts, tail_counts = diagnostics.raw_shifter(
        arr, contract_years, years0)
    live = np.isfinite(raw)
    filled = np.where(live, raw, 0.0)
    smooth = 2.0 * filled / (2.0 + np.abs(filled))
    cap1 = np.clip(filled, -1.0, 1.0)
    if np.any(np.abs(smooth[live]) >= 2.0):
        raise SystemExit("FAIL: smooth shifter violates its strict bounds")
    if not np.allclose(smooth[live & (raw == 0)], 0.0,
                       rtol=0.0, atol=0.0):
        raise SystemExit("FAIL: smooth shifter does not map zero to zero")

    units, years, zwide, _, _ = diagnostics.county_instruments({
        "smooth_bounded": smooth,
        "cap_1": cap1,
    })
    if not np.array_equal(units, units0) or years != years0:
        raise SystemExit("FAIL: shifter and exposure axes differ")

    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy()
    ci = baseline["year"].map(colpos).to_numpy()
    if pd.isna(ri).any() or pd.isna(ci).any():
        raise SystemExit("FAIL: baseline rows do not align to the stored panel")
    ri, ci = ri.astype(int), ci.astype(int)
    z = zwide["smooth_bounded"][ri, ci]
    zcap1 = zwide["cap_1"][ri, ci]
    if not np.isfinite(z).all() or not np.isfinite(zcap1).all():
        raise SystemExit("FAIL: a county instrument is missing")

    x, weight, fe, nofe, cluster = diagnostics.estimation_arrays(baseline)
    fit = t28.fit2sls(x, z, z, weight, fe, nofe, cluster,
                      singleton=False)
    demeaned = t28.w22.demean(
        np.column_stack([x, z]), fe, nofe, weight)
    xt, zt = demeaned[:, 0], demeaned[:, 1]
    partial_r2 = float((zt @ xt) ** 2 / ((zt @ zt) * (xt @ xt)))
    tstat = float(fit["beta"] / fit["se"])

    reference = pd.read_csv(REFERENCE_OUT)
    reference = reference.loc[reference["variant"] == "cap_1"]
    if len(reference) != 1:
        raise SystemExit("FAIL: clipping output lacks one cap-1 reference row")
    reference = reference.iloc[0]
    if int(reference["n"]) != len(baseline):
        raise SystemExit("FAIL: smooth and cap-1 samples differ")

    live_values = smooth[live]
    row = {
        "instrument": "smooth_bounded_roll5",
        "formula": "2*r/(2+abs(r))",
        "horizon": H,
        "treatment": "two-year average-window contract change per capita",
        "sample_start": int(baseline["year"].min()),
        "sample_end": int(baseline["year"].max()),
        "n": len(baseline),
        "n_counties": int(baseline["ccg"].nunique()),
        "fixed_effects": "division-by-year",
        "weight": "equal county",
        "cluster": "county",
        "first_stage": float(fit["beta"]),
        "first_stage_se": float(fit["se"]),
        "first_stage_t": tstat,
        "first_stage_F": tstat ** 2,
        "partial_r2": partial_r2,
        "instrument_sd": float(np.std(z, ddof=1)),
        "within_instrument_sd": float(np.std(zt, ddof=1)),
        "correlation_with_cap_1": float(np.corrcoef(z, zcap1)[0, 1]),
        "minimum_category_shifter": float(live_values.min()),
        "maximum_category_shifter": float(live_values.max()),
        "maximum_absolute_category_shifter":
            float(np.abs(live_values).max()),
        "live_category_years": int(sum(live_counts)),
        "raw_tail_category_years": int(sum(tail_counts)),
        "raw_tail_share": float(sum(tail_counts) / sum(live_counts)),
        "cap_1_first_stage": float(reference["first_stage"]),
        "cap_1_first_stage_F": float(reference["first_stage_F"]),
        "cap_1_partial_r2": float(reference["partial_r2"]),
        "first_stage_F_difference_from_cap_1":
            float(tstat ** 2 - reference["first_stage_F"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(OUT, index=False)

    print(f"   sample: {row['n']:,} county-years, "
          f"{row['n_counties']:,} counties")
    print(f"   realized category-shifter range: "
          f"[{row['minimum_category_shifter']:+.6f}, "
          f"{row['maximum_category_shifter']:+.6f}]")
    print(f"   smooth bounded: F={row['first_stage_F']:.3f}, "
          f"partial R2={row['partial_r2']:.6f}, "
          f"pi={row['first_stage']:+.6f}, "
          f"se={row['first_stage_se']:.6f}, "
          f"corr(cap1)={row['correlation_with_cap_1']:+.3f}")
    print(f"   cap 1 reference: F={row['cap_1_first_stage_F']:.3f}, "
          f"partial R2={row['cap_1_partial_r2']:.6f}, "
          f"pi={row['cap_1_first_stage']:+.6f}")
    print(f"   wrote {OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
