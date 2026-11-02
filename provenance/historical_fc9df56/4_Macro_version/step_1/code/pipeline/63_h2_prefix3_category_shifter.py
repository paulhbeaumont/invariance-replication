"""Test a prefix-level h=2 composition instrument under clip 1 and clip 2.

The persisted top-800 four-character procurement categories are collapsed to
the requested prefix length before national growth and county exposure are formed.
The raw shifter is standard growth from the final pre-shock year to the
two-year post-shock average. Zero-base prefix-years are excluded.

Writes Results/instrument_normalization/h2_county_prefix{N}_tail_comparison.csv.
"""
import argparse
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

sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import store  # noqa: E402


H = 2
CAPS = (1.0, 2.0)


def load_module(name, path):
    """Load a project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prefix_mapping(categories, prefix_digits):
    """Return ordered prefixes, integer assignments, and an aggregation matrix."""
    prefixes = [str(category)[:prefix_digits] for category in categories]
    unique = list(dict.fromkeys(prefixes))
    position = {prefix: j for j, prefix in enumerate(unique)}
    assignment = np.array([position[prefix] for prefix in prefixes], np.int64)
    matrix = np.zeros((len(categories), len(unique)), np.float64)
    matrix[np.arange(len(categories)), assignment] = 1.0
    return unique, assignment, matrix


def grouped_national(arr, assignment, n_groups):
    """Aggregate the national category-year panel to prefix groups."""
    national = arr.sum(axis=0, dtype=np.float64)
    grouped = np.zeros((n_groups, national.shape[1]), np.float64)
    np.add.at(grouped, assignment, national)
    if not np.allclose(grouped.sum(axis=0), national.sum(axis=0),
                       rtol=1e-12, atol=1e-5):
        raise SystemExit("FAIL: prefix aggregation does not preserve national dollars")
    return grouped


def county_instruments(arr, mapping, national, contract_years, target_years,
                       population):
    """Construct prefix-level instruments for the two cap values."""
    instruments = {
        cap: np.full((arr.shape[0], len(target_years)), np.nan, np.float64)
        for cap in CAPS
    }
    diagnostics = {cap: [] for cap in CAPS}
    for j, end_year in enumerate(target_years):
        post2 = 0.5 * (
            national[:, contract_years.index(end_year)]
            + national[:, contract_years.index(end_year - 1)])
        base = national[:, contract_years.index(end_year - H)]
        live = np.isfinite(base) & (base > 0) & np.isfinite(post2)
        raw = np.zeros(len(base), np.float64)
        raw[live] = (post2[live] - base[live]) / base[live]

        exposure_hi = end_year - H - locked.LAG
        exposure_lo = exposure_hi - locked.SPAN + 1
        if exposure_lo < locked.DATA_FIRST:
            continue
        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure4 = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        exposure3 = exposure4 @ mapping
        pbase = population[end_year - H].to_numpy(np.float64)
        for cap in CAPS:
            shifter = np.zeros(len(base), np.float64)
            shifter[live] = np.clip(raw[live], -cap, cap)
            instruments[cap][:, j] = exposure3 @ shifter / pbase
            diagnostics[cap].append({
                "n_live": int(live.sum()),
                "n_modified": int((live & (np.abs(raw) > cap)).sum()),
            })
    return instruments, diagnostics


def main(prefix_digits):
    """Estimate prefix instruments on the locked county sample."""
    start = time.time()
    if prefix_digits not in (2, 3):
        raise SystemExit("FAIL: prefix digits must be 2 or 3")
    output = (ROOT / "Results" / "instrument_normalization"
              / f"h2_county_prefix{prefix_digits}_tail_comparison.csv")
    print(f"== h=2 {prefix_digits}-character procurement-category comparison")
    p57 = load_module(
        f"p57_prefix{prefix_digits}", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        f"t28_prefix{prefix_digits}", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        f"clip_diagnostics_prefix{prefix_digits}",
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

    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("P",))
    units2, categories, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    prefixes, assignment, mapping = prefix_mapping(categories, prefix_digits)
    national = grouped_national(arr, assignment, len(prefixes))
    years = store.target_years(H)
    zwide, diagnostics = county_instruments(
        arr, mapping, national, contract_years, years, pivots["P"])

    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy()
    ci = baseline["year"].map(colpos).to_numpy()
    if pd.isna(ri).any() or pd.isna(ci).any():
        raise SystemExit("FAIL: baseline rows do not align to the stored panel")
    ri, ci = ri.astype(int), ci.astype(int)
    zsample = {cap: matrix[ri, ci] for cap, matrix in zwide.items()}
    if any(not np.isfinite(z).all() for z in zsample.values()):
        raise SystemExit("FAIL: a prefix-level county instrument is missing")

    reference1 = pd.read_csv(DENOMINATOR_OUT)
    reference1 = reference1.loc[reference1["variant"] == "base_drop_zero"]
    reference2 = pd.read_csv(TAIL_OUT)
    reference2 = reference2.loc[reference2["variant"] == "clip_2"]
    if len(reference1) != 1 or len(reference2) != 1:
        raise SystemExit("FAIL: four-character reference rows are incomplete")
    reference = {1.0: reference1.iloc[0], 2.0: reference2.iloc[0]}
    if any(int(row["n"]) != len(baseline) for row in reference.values()):
        raise SystemExit("FAIL: prefix and reference samples differ")

    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    rows = []
    for cap in CAPS:
        z = zsample[cap]
        fit = t28.fit2sls(x, z, z, weight, fe, nofe, cluster,
                          singleton=False)
        demeaned = t28.w22.demean(
            np.column_stack([x, z]), fe, nofe, weight)
        xt, zt = demeaned[:, 0], demeaned[:, 1]
        partial_r2 = float((zt @ xt) ** 2 / ((zt @ zt) * (xt @ xt)))
        tstat = float(fit["beta"] / fit["se"])
        records = diagnostics[cap]
        n_live = sum(record["n_live"] for record in records)
        n_modified = sum(record["n_modified"] for record in records)
        rows.append({
            "category_level": f"{prefix_digits}-character prefix",
            "category_universe": "persisted top-800 four-character codes",
            "n_four_character_categories": len(categories),
            "n_prefix_categories": len(prefixes),
            "tail_rule": f"clip_{cap:g}",
            "clip": cap,
            "horizon": H,
            "numerator": "post2 minus final pre-shock year",
            "denominator": "final pre-shock national prefix spending",
            "zero_denominator_rule": "exclude prefix-year",
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
            "live_prefix_years": int(n_live),
            "modified_prefix_years": int(n_modified),
            "modified_share_of_live": n_modified / n_live,
            "four_character_first_stage_F":
                float(reference[cap]["first_stage_F"]),
            "four_character_partial_r2":
                float(reference[cap]["partial_r2"]),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"   category universe: {len(categories)} four-character codes "
          f"collapsed to {len(prefixes)} {prefix_digits}-character prefixes")
    print(f"   sample: {len(baseline):,} county-years, "
          f"{baseline['ccg'].nunique():,} counties")
    for row in rows:
        print(f"   prefix{prefix_digits} {row['tail_rule']}: "
              f"F={row['first_stage_F']:.3f}, "
              f"partial R2={row['partial_r2']:.6f}, "
              f"pi={row['first_stage']:+.6f}, "
              f"modified={100 * row['modified_share_of_live']:.2f}%")
        print(f"      four-character reference: "
              f"F={row['four_character_first_stage_F']:.3f}, "
              f"partial R2={row['four_character_partial_r2']:.6f}")
    print(f"   wrote {output}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-digits", type=int, default=3,
                        choices=(2, 3))
    arguments = parser.parse_args()
    main(arguments.prefix_digits)
