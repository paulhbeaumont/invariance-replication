"""Compare h=2 base-year and five-year shifter denominators under clip 1.

Every shifter uses the two-year post-shock average minus the final pre-shock
year in its numerator. The variants change only the denominator and the rule
for category-years with zero spending in the final pre-shock year.

Writes Results/instrument_normalization/h2_county_denominator_comparison.csv.
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
       / "h2_county_denominator_comparison.csv")

sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import store  # noqa: E402


H = 2
PRE_YEARS = 5
CAP = 1.0


def load_module(name, path):
    """Load a project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def category_shifters(arr, contract_years, target_years):
    """Build fixed-cap shifters with alternative scaling and zero rules."""
    national = arr.sum(axis=0, dtype=np.float64)
    shape = (national.shape[0], len(target_years))
    names = (
        "pre5_native",
        "pre5_common_support",
        "base_drop_zero",
        "base_zero_at_limit",
    )
    values = {name: np.zeros(shape, np.float64) for name in names}
    diagnostics = {name: [] for name in names}

    for j, end_year in enumerate(target_years):
        post2 = 0.5 * (
            national[:, contract_years.index(end_year)]
            + national[:, contract_years.index(end_year - 1)])
        base = national[:, contract_years.index(end_year - H)]
        pre5 = np.column_stack([
            national[:, contract_years.index(end_year - H - ell)]
            for ell in range(PRE_YEARS)
        ]).mean(axis=1)
        change = post2 - base

        pre5_live = np.isfinite(pre5) & (pre5 > 0)
        common = pre5_live & np.isfinite(base) & (base > 0)
        base_zero_entry = pre5_live & (base == 0) & (post2 > 0)

        raw_pre5 = np.zeros(len(base), np.float64)
        raw_pre5[pre5_live] = change[pre5_live] / pre5[pre5_live]
        raw_base = np.zeros(len(base), np.float64)
        raw_base[common] = change[common] / base[common]

        values["pre5_native"][:, j] = np.clip(raw_pre5, -CAP, CAP)
        values["pre5_common_support"][common, j] = np.clip(
            raw_pre5[common], -CAP, CAP)
        values["base_drop_zero"][common, j] = np.clip(
            raw_base[common], -CAP, CAP)
        values["base_zero_at_limit"][:, j] = values["base_drop_zero"][:, j]
        values["base_zero_at_limit"][base_zero_entry, j] = CAP

        rules = {
            "pre5_native": (pre5_live, raw_pre5),
            "pre5_common_support": (common, raw_pre5),
            "base_drop_zero": (common, raw_base),
            "base_zero_at_limit":
                (common | base_zero_entry,
                 np.where(base_zero_entry, np.inf, raw_base)),
        }
        for name, (live, raw) in rules.items():
            diagnostics[name].append({
                "n_live": int(live.sum()),
                "n_clipped": int((live & (np.abs(raw) > CAP)).sum()),
                "n_base_zero_entry": int(base_zero_entry.sum()),
            })
    return values, diagnostics


def main():
    """Run the fixed-sample denominator comparison."""
    start = time.time()
    print("== h=2 shifter denominator comparison under clip 1")
    p57 = load_module("p57_denominator", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module("t28_denominator", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "clip_diagnostics_denominator",
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
    shifters, diagnostics = category_shifters(arr, contract_years, years0)
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

    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    labels = {
        "pre5_native": "Five-year denominator, native support",
        "pre5_common_support": "Five-year denominator, base-positive support",
        "base_drop_zero": "Base-year denominator, drop zero base",
        "base_zero_at_limit": "Base-year denominator, zero-base entry at +1",
    }
    reference_z = zsample["pre5_native"]
    rows = []
    for name in labels:
        z = zsample[name]
        fit = t28.fit2sls(x, z, z, weight, fe, nofe, cluster,
                          singleton=False)
        demeaned = t28.w22.demean(
            np.column_stack([x, z]), fe, nofe, weight)
        xt, zt = demeaned[:, 0], demeaned[:, 1]
        partial_r2 = float((zt @ xt) ** 2 / ((zt @ zt) * (xt @ xt)))
        tstat = float(fit["beta"] / fit["se"])
        records = diagnostics[name]
        n_live = sum(record["n_live"] for record in records)
        n_clipped = sum(record["n_clipped"] for record in records)
        rows.append({
            "variant": name,
            "label": labels[name],
            "horizon": H,
            "numerator": "post2 minus final pre-shock year",
            "clip": CAP,
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
            "correlation_with_pre5_native":
                float(np.corrcoef(z, reference_z)[0, 1]),
            "live_category_years": int(n_live),
            "clipped_category_years": int(n_clipped),
            "clipped_share_of_live": float(n_clipped / n_live),
            "base_zero_entry_category_years":
                int(sum(record["n_base_zero_entry"] for record in records)),
        })

    certified = pd.read_csv(REFERENCE_OUT)
    certified = certified.loc[certified["variant"] == "cap_1"]
    if len(certified) != 1:
        raise SystemExit("FAIL: clipping output lacks one cap-1 reference row")
    certified = certified.iloc[0]
    native = next(row for row in rows if row["variant"] == "pre5_native")
    for key in ("first_stage", "first_stage_se", "first_stage_F", "partial_r2"):
        expected = float(certified[key])
        gap = abs(native[key] - expected) / max(abs(expected), 1e-300)
        if gap >= locked.RTOL_COEF:
            raise SystemExit(f"FAIL: pre5 {key} reference gap is {gap:.3e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"   sample: {len(baseline):,} county-years, "
          f"{baseline['ccg'].nunique():,} counties")
    for row in rows:
        print(f"   {row['variant']:>23}: F={row['first_stage_F']:.3f}, "
              f"partial R2={row['partial_r2']:.6f}, "
              f"pi={row['first_stage']:+.6f}, "
              f"corr(pre5)={row['correlation_with_pre5_native']:+.3f}, "
              f"clip share={100 * row['clipped_share_of_live']:.2f}%")
    print(f"   base-zero positive-entry category-years: "
          f"{rows[0]['base_zero_entry_category_years']:,}")
    print(f"   wrote {OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
