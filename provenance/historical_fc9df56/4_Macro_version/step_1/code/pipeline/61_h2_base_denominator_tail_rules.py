"""Compare tail rules for the h=2 base-year-normalized category shifter.

The raw shifter divides the two-year post-shock average minus final pre-shock
spending by final pre-shock spending. Category-years with a zero denominator
are excluded under every tail rule.

Writes Results/instrument_normalization/h2_county_base_denominator_tail_rules.csv.
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
                 / "h2_county_denominator_comparison.csv")
OUT = (ROOT / "Results" / "instrument_normalization"
       / "h2_county_base_denominator_tail_rules.csv")

sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import store  # noqa: E402


H = 2
LOWER_Q = 0.01
UPPER_Q = 0.99


def load_module(name, path):
    """Load a project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def category_shifters(arr, contract_years, target_years):
    """Build base-year-normalized shifters with three tail rules."""
    national = arr.sum(axis=0, dtype=np.float64)
    shape = (national.shape[0], len(target_years))
    raw = np.full(shape, np.nan, np.float64)
    live = np.zeros(shape, bool)
    for j, end_year in enumerate(target_years):
        post2 = 0.5 * (
            national[:, contract_years.index(end_year)]
            + national[:, contract_years.index(end_year - 1)])
        base = national[:, contract_years.index(end_year - H)]
        live[:, j] = np.isfinite(base) & (base > 0) & np.isfinite(post2)
        raw[live[:, j], j] = (
            post2[live[:, j]] - base[live[:, j]]) / base[live[:, j]]

    pooled_low, pooled_high = np.quantile(
        raw[live], [LOWER_Q, UPPER_Q])
    values = {
        "clip_2": np.zeros(shape, np.float64),
        "winsor_pooled_1_99": np.zeros(shape, np.float64),
        "winsor_year_1_99": np.zeros(shape, np.float64),
        "clip_1_reference": np.zeros(shape, np.float64),
    }
    values["clip_2"][live] = np.clip(raw[live], -2.0, 2.0)
    values["winsor_pooled_1_99"][live] = np.clip(
        raw[live], pooled_low, pooled_high)
    values["clip_1_reference"][live] = np.clip(raw[live], -1.0, 1.0)

    year_bounds = []
    modified_counts = {name: [] for name in (
        "clip_2", "winsor_pooled_1_99", "winsor_year_1_99")}
    for j, year in enumerate(target_years):
        year_live = live[:, j]
        year_raw = raw[year_live, j]
        year_low, year_high = np.quantile(
            year_raw, [LOWER_Q, UPPER_Q])
        values["winsor_year_1_99"][year_live, j] = np.clip(
            year_raw, year_low, year_high)
        year_bounds.append({
            "year": year,
            "lower": float(year_low),
            "upper": float(year_high),
            "n_live": int(year_live.sum()),
        })
        modified_counts["clip_2"].append(
            int(((year_raw < -2.0) | (year_raw > 2.0)).sum()))
        modified_counts["winsor_pooled_1_99"].append(
            int(((year_raw < pooled_low) | (year_raw > pooled_high)).sum()))
        modified_counts["winsor_year_1_99"].append(
            int(((year_raw < year_low) | (year_raw > year_high)).sum()))

    return values, raw, live, pooled_low, pooled_high, year_bounds, modified_counts


def main():
    """Estimate the three tail rules on the locked county sample."""
    start = time.time()
    print("== h=2 base-year denominator tail-rule comparison")
    p57 = load_module("p57_tail", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module("t28_tail", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "clip_diagnostics_tail",
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
    (shifters, raw, live, pooled_low, pooled_high, year_bounds,
     modified_counts) = category_shifters(arr, contract_years, years0)
    units, years, zwide, _, _ = helper.county_instruments(shifters)
    if not np.array_equal(units, units0) or years != years0:
        raise SystemExit("FAIL: shifter and exposure axes differ")

    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy()
    ci = baseline["year"].map(colpos).to_numpy()
    if pd.isna(ri).any() or pd.isna(ci).any():
        raise SystemExit("FAIL: baseline rows do not align to the stored panel")
    ri, ci = ri.astype(int), ci.astype(int)
    zsample = {name: matrix[ri, ci] for name, matrix in zwide.items()}
    if any(not np.isfinite(z).all() for z in zsample.values()):
        raise SystemExit("FAIL: a county instrument is missing")

    reference = pd.read_csv(REFERENCE_OUT)
    reference = reference.loc[reference["variant"] == "base_drop_zero"]
    if len(reference) != 1:
        raise SystemExit("FAIL: denominator output lacks one base-year reference")
    reference = reference.iloc[0]
    if int(reference["n"]) != len(baseline):
        raise SystemExit("FAIL: tail-rule and reference samples differ")

    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    labels = {
        "clip_2": "Fixed clip at -2 and 2",
        "winsor_pooled_1_99": "Pooled category-year p1 and p99 winsorization",
        "winsor_year_1_99": "Within-year category p1 and p99 winsorization",
    }
    reference_z = zsample["clip_1_reference"]
    total_live = int(live.sum())
    total_category_years = int(live.size)
    rows = []
    year_lower = np.array([record["lower"] for record in year_bounds])
    year_upper = np.array([record["upper"] for record in year_bounds])
    for name, label in labels.items():
        z = zsample[name]
        fit = t28.fit2sls(x, z, z, weight, fe, nofe, cluster,
                          singleton=False)
        demeaned = t28.w22.demean(
            np.column_stack([x, z]), fe, nofe, weight)
        xt, zt = demeaned[:, 0], demeaned[:, 1]
        partial_r2 = float((zt @ xt) ** 2 / ((zt @ zt) * (xt @ xt)))
        tstat = float(fit["beta"] / fit["se"])
        if name == "clip_2":
            lower_min = lower_max = -2.0
            upper_min = upper_max = 2.0
        elif name == "winsor_pooled_1_99":
            lower_min = lower_max = float(pooled_low)
            upper_min = upper_max = float(pooled_high)
        else:
            lower_min, lower_max = float(year_lower.min()), float(year_lower.max())
            upper_min, upper_max = float(year_upper.min()), float(year_upper.max())
        modified = int(sum(modified_counts[name]))
        rows.append({
            "variant": name,
            "label": label,
            "horizon": H,
            "numerator": "post2 minus final pre-shock year",
            "denominator": "final pre-shock national category spending",
            "zero_denominator_rule": "exclude category-year",
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
            "correlation_with_base_clip_1":
                float(np.corrcoef(z, reference_z)[0, 1]),
            "live_category_years": total_live,
            "excluded_zero_base_category_years":
                total_category_years - total_live,
            "modified_category_years": modified,
            "modified_share_of_live": modified / total_live,
            "lower_bound_min": lower_min,
            "lower_bound_max": lower_max,
            "upper_bound_min": upper_min,
            "upper_bound_max": upper_max,
            "base_clip_1_first_stage": float(reference["first_stage"]),
            "base_clip_1_first_stage_F": float(reference["first_stage_F"]),
            "base_clip_1_partial_r2": float(reference["partial_r2"]),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"   sample: {len(baseline):,} county-years, "
          f"{baseline['ccg'].nunique():,} counties")
    print(f"   common category-year support: {total_live:,}/"
          f"{total_category_years:,}")
    print(f"   pooled p1/p99: {pooled_low:+.6f}, {pooled_high:+.6f}")
    print(f"   within-year p1 range: [{year_lower.min():+.6f}, "
          f"{year_lower.max():+.6f}]")
    print(f"   within-year p99 range: [{year_upper.min():+.6f}, "
          f"{year_upper.max():+.6f}]")
    for row in rows:
        print(f"   {row['variant']:>20}: F={row['first_stage_F']:.3f}, "
              f"partial R2={row['partial_r2']:.6f}, "
              f"pi={row['first_stage']:+.6f}, "
              f"corr(clip1)={row['correlation_with_base_clip_1']:+.3f}, "
              f"modified={100 * row['modified_share_of_live']:.2f}%")
    print(f"   base clip 1 reference: F={reference['first_stage_F']:.3f}, "
          f"partial R2={reference['partial_r2']:.6f}")
    print(f"   wrote {OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
