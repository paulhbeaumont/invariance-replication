"""Test a midpoint shifter using five pre-shock and two post-shock years.

The category shifter is twice the difference between the two-year post-shock
mean and the five-year pre-shock mean, divided by their sum. The construction
is naturally bounded between -2 and 2.

Writes Results/instrument_normalization/h2_county_mp5_shifter.csv.
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
       / "h2_county_mp5_shifter.csv")

sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import store  # noqa: E402


H = 2
PRE_YEARS = 5


def load_module(name, path):
    """Load a project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def category_shifters(arr, contract_years, target_years):
    """Construct MP5 and cap-1 shifters on the same category-year axis."""
    national = arr.sum(axis=0, dtype=np.float64)
    shape = (national.shape[0], len(target_years))
    mp5 = np.zeros(shape, np.float64)
    cap1 = np.zeros(shape, np.float64)
    nonzero_counts = []
    endpoint_counts = []
    for j, end_year in enumerate(target_years):
        post2 = 0.5 * (
            national[:, contract_years.index(end_year)]
            + national[:, contract_years.index(end_year - 1)])
        pre5 = np.column_stack([
            national[:, contract_years.index(end_year - H - ell)]
            for ell in range(PRE_YEARS)
        ]).mean(axis=1)

        denominator = post2 + pre5
        live = np.isfinite(denominator) & (denominator > 0)
        mp5[live, j] = 2.0 * (post2[live] - pre5[live]) / denominator[live]

        base = national[:, contract_years.index(end_year - H)]
        raw_live = np.isfinite(pre5) & (pre5 > 0)
        raw = np.zeros(len(pre5), np.float64)
        raw[raw_live] = (post2[raw_live] - base[raw_live]) / pre5[raw_live]
        cap1[:, j] = np.clip(raw, -1.0, 1.0)

        nonzero_counts.append(int(live.sum()))
        endpoint_counts.append(int((live & np.isclose(np.abs(mp5[:, j]), 2.0)).sum()))

    if np.any(np.abs(mp5) > 2.0 + 1e-12):
        raise SystemExit("FAIL: MP5 category shifter violates its bounds")
    return mp5, cap1, nonzero_counts, endpoint_counts


def main():
    """Estimate MP5 and compare it with the certified cap-1 result."""
    start = time.time()
    print("== h=2 five-year-pre versus two-year-post midpoint shifter")
    p57 = load_module("p57_mp5", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module("t28_mp5", STEP2_PY / "28_table_d1.py")
    diagnostics = load_module(
        "clip_diagnostics_mp5",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
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
    mp5, cap1, live_counts, endpoint_counts = category_shifters(
        arr, contract_years, years0)
    units, years, zwide, _, _ = diagnostics.county_instruments({
        "mp5": mp5,
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
    z = zwide["mp5"][ri, ci]
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
        raise SystemExit("FAIL: MP5 and cap-1 samples differ")

    live_values = mp5[np.isfinite(mp5)]
    row = {
        "instrument": "mp5_pre5_post2",
        "formula": "2*(post2-pre5)/(post2+pre5)",
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
        "mean_live_categories": float(np.mean(live_counts)),
        "category_years_at_endpoint": int(sum(endpoint_counts)),
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
    print(f"   category-years exactly at -2 or +2: "
          f"{row['category_years_at_endpoint']:,}")
    print(f"   MP5: F={row['first_stage_F']:.3f}, "
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
