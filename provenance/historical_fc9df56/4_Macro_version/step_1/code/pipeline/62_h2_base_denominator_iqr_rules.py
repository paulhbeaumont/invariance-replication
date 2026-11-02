"""Test pooled and within-year IQR4/IQR5 base-normalized h=2 shifters.

The raw shifter divides the two-year post-shock average minus final pre-shock
spending by final pre-shock spending. Category-years with a zero denominator
are excluded. Cutoffs match winsor4 method(iq): median plus or minus L times
the interquartile range using Stata's percentile arithmetic.

Writes Results/instrument_normalization/h2_county_base_denominator_iqr_rules.csv.
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
DENOMINATOR_OUT = (ROOT / "Results" / "instrument_normalization"
                   / "h2_county_denominator_comparison.csv")
TAIL_OUT = (ROOT / "Results" / "instrument_normalization"
            / "h2_county_base_denominator_tail_rules.csv")
OUT = (ROOT / "Results" / "instrument_normalization"
       / "h2_county_base_denominator_iqr_rules.csv")

sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import store  # noqa: E402


H = 2
LEVELS = (4.0, 5.0)


def load_module(name, path):
    """Load a project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_shifter(arr, contract_years, target_years):
    """Return standard growth and its positive-base support."""
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
    return raw, live


def iqr_shifters(raw, live, target_years, cutoff_function):
    """Construct the four IQR shifters and record their realized cutoffs."""
    shape = raw.shape
    values = {f"iqr{level:g}_{placement}": np.zeros(shape, np.float64)
              for level in LEVELS for placement in ("pooled", "year")}
    values["clip_1_reference"] = np.zeros(shape, np.float64)
    values["clip_1_reference"][live] = np.clip(raw[live], -1.0, 1.0)
    diagnostics = {}

    for level in LEVELS:
        pooled_low, pooled_high = cutoff_function(raw[live], level)
        pooled_name = f"iqr{level:g}_pooled"
        values[pooled_name][live] = np.clip(
            raw[live], pooled_low, pooled_high)
        diagnostics[pooled_name] = {
            "lower_min": float(pooled_low),
            "lower_max": float(pooled_low),
            "upper_min": float(pooled_high),
            "upper_max": float(pooled_high),
            "n_modified": int(((raw[live] < pooled_low)
                               | (raw[live] > pooled_high)).sum()),
        }

        year_name = f"iqr{level:g}_year"
        lower, upper, modified = [], [], 0
        for j, _ in enumerate(target_years):
            year_live = live[:, j]
            year_raw = raw[year_live, j]
            year_low, year_high = cutoff_function(year_raw, level)
            values[year_name][year_live, j] = np.clip(
                year_raw, year_low, year_high)
            lower.append(year_low)
            upper.append(year_high)
            modified += int(((year_raw < year_low)
                             | (year_raw > year_high)).sum())
        diagnostics[year_name] = {
            "lower_min": float(np.min(lower)),
            "lower_max": float(np.max(lower)),
            "upper_min": float(np.min(upper)),
            "upper_max": float(np.max(upper)),
            "n_modified": int(modified),
        }
    return values, diagnostics


def main():
    """Estimate IQR4 and IQR5 under pooled and within-year placement."""
    start = time.time()
    print("== h=2 base-year denominator IQR4/IQR5 comparison")
    p57 = load_module("p57_iqr", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module("t28_iqr", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "clip_diagnostics_iqr",
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
    raw, live = raw_shifter(arr, contract_years, years0)
    shifters, diagnostics = iqr_shifters(
        raw, live, years0, t28.w22.iq_cutoffs)
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

    denominator_reference = pd.read_csv(DENOMINATOR_OUT)
    denominator_reference = denominator_reference.loc[
        denominator_reference["variant"] == "base_drop_zero"]
    if len(denominator_reference) != 1:
        raise SystemExit("FAIL: denominator output lacks one base-year reference")
    denominator_reference = denominator_reference.iloc[0]
    tail_reference = pd.read_csv(TAIL_OUT)
    if set(tail_reference["variant"]) != {
            "clip_2", "winsor_pooled_1_99", "winsor_year_1_99"}:
        raise SystemExit("FAIL: tail-rule reference rows are incomplete")
    if int(denominator_reference["n"]) != len(baseline):
        raise SystemExit("FAIL: IQR and reference samples differ")

    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    reference_z = zsample["clip_1_reference"]
    total_live = int(live.sum())
    total_category_years = int(live.size)
    labels = {
        "iqr4_pooled": "Pooled median plus or minus 4 IQR",
        "iqr4_year": "Within-year median plus or minus 4 IQR",
        "iqr5_pooled": "Pooled median plus or minus 5 IQR",
        "iqr5_year": "Within-year median plus or minus 5 IQR",
    }
    rows = []
    for name, label in labels.items():
        z = zsample[name]
        fit = t28.fit2sls(x, z, z, weight, fe, nofe, cluster,
                          singleton=False)
        demeaned = t28.w22.demean(
            np.column_stack([x, z]), fe, nofe, weight)
        xt, zt = demeaned[:, 0], demeaned[:, 1]
        partial_r2 = float((zt @ xt) ** 2 / ((zt @ zt) * (xt @ xt)))
        tstat = float(fit["beta"] / fit["se"])
        diag = diagnostics[name]
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
            "modified_category_years": diag["n_modified"],
            "modified_share_of_live": diag["n_modified"] / total_live,
            "lower_bound_min": diag["lower_min"],
            "lower_bound_max": diag["lower_max"],
            "upper_bound_min": diag["upper_min"],
            "upper_bound_max": diag["upper_max"],
            "base_clip_1_first_stage_F":
                float(denominator_reference["first_stage_F"]),
            "base_clip_1_partial_r2":
                float(denominator_reference["partial_r2"]),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"   sample: {len(baseline):,} county-years, "
          f"{baseline['ccg'].nunique():,} counties")
    print(f"   common category-year support: {total_live:,}/"
          f"{total_category_years:,}")
    for row in rows:
        print(f"   {row['variant']:>11}: F={row['first_stage_F']:.3f}, "
              f"partial R2={row['partial_r2']:.6f}, "
              f"pi={row['first_stage']:+.6f}, "
              f"corr(clip1)={row['correlation_with_base_clip_1']:+.3f}, "
              f"modified={100 * row['modified_share_of_live']:.2f}%")
        print(f"      lower [{row['lower_bound_min']:+.6f}, "
              f"{row['lower_bound_max']:+.6f}], upper "
              f"[{row['upper_bound_min']:+.6f}, "
              f"{row['upper_bound_max']:+.6f}]")
    print(f"   base clip 1 reference: "
          f"F={denominator_reference['first_stage_F']:.3f}, "
          f"partial R2={denominator_reference['partial_r2']:.6f}")
    print(f"   wrote {OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
