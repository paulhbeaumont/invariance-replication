"""County h=2 first stages for five national-category transformations.

Every candidate predicts the locked two-year average-window contract change
per capita on the same county-year sample. The regression uses equal county
weights, division-by-year fixed effects, and county-clustered inference.

Writes Results/instrument_normalization/h2_county_first_stage_sequence.csv.
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
sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import store  # noqa: E402
import dcads_common as dc  # noqa: E402


OUT = (dc.MACRO_RESULTS_ROOT / "instrument_normalization"
       / "h2_county_first_stage_sequence.csv")
H = 2
DENOM_YEARS = 5
BASE_SCREEN_QUANTILE = 0.01


LABELS = {
    "A_current": "Endpoint numerator, full-period mean, clip 1",
    "B_matched_fullmean": "Matched numerator, full-period mean, clip 1",
    "C_matched_roll5": "Matched numerator, rolling five-year mean, clip 1",
    "D_symmetric": "Matched symmetric growth, bounded at 2",
    "E_conventional_screened":
        "Matched conventional growth, bottom 1% base screen",
}


def load_module(name, path):
    """Load one project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shifter_matrices(arr, contract_years, target_years):
    """Return the five category-by-year shifter matrices and diagnostics."""
    national = arr.sum(axis=0, dtype=np.float64)
    mu = np.nanmean(np.abs(national), axis=1)
    mu[mu == 0] = np.nan

    base_all = np.column_stack([
        national[:, contract_years.index(t - H)] for t in target_years
    ])
    positive_bases = base_all[np.isfinite(base_all) & (base_all > 0)]
    base_floor = float(np.quantile(positive_bases, BASE_SCREEN_QUANTILE))

    shape = (national.shape[0], len(target_years))
    values = {name: np.full(shape, np.nan, np.float64) for name in LABELS}
    diagnostics = {name: [] for name in LABELS}

    for j, t in enumerate(target_years):
        end = national[:, contract_years.index(t)]
        prior = national[:, contract_years.index(t - 1)]
        base = national[:, contract_years.index(t - H)]
        window = 0.5 * (end + prior)
        matched_change = window - base
        endpoint_change = end - base
        roll5 = np.column_stack([
            national[:, contract_years.index(t - H - ell)]
            for ell in range(DENOM_YEARS)
        ]).mean(axis=1)

        raw_a = endpoint_change / mu
        raw_b = matched_change / mu
        live_c = np.isfinite(roll5) & (roll5 > 0)
        raw_c = np.full(len(roll5), np.nan, np.float64)
        raw_c[live_c] = matched_change[live_c] / roll5[live_c]

        symmetric_den = window + base
        raw_d = np.zeros(len(symmetric_den), np.float64)
        live_d = symmetric_den > 0
        raw_d[live_d] = 2.0 * matched_change[live_d] / symmetric_den[live_d]

        live_e = np.isfinite(base) & (base >= base_floor)
        raw_e = np.full(len(base), np.nan, np.float64)
        raw_e[live_e] = matched_change[live_e] / base[live_e]

        values["A_current"][:, j] = np.clip(raw_a, -1.0, 1.0)
        values["B_matched_fullmean"][:, j] = np.clip(raw_b, -1.0, 1.0)
        values["C_matched_roll5"][:, j] = np.clip(raw_c, -1.0, 1.0)
        values["D_symmetric"][:, j] = raw_d
        values["E_conventional_screened"][:, j] = raw_e

        for name, raw, live, bound in (
                ("A_current", raw_a, np.isfinite(raw_a), 1.0),
                ("B_matched_fullmean", raw_b, np.isfinite(raw_b), 1.0),
                ("C_matched_roll5", raw_c, live_c, 1.0),
                ("D_symmetric", raw_d, live_d, 2.0),
                ("E_conventional_screened", raw_e, live_e, np.nan)):
            finite = live & np.isfinite(raw)
            transformed = values[name][:, j]
            at_bound = (np.isclose(np.abs(transformed), bound)
                        if np.isfinite(bound) else np.zeros(len(raw), bool))
            beyond_bound = ((np.abs(raw) > bound)
                            if np.isfinite(bound) else np.zeros(len(raw), bool))
            diagnostics[name].append({
                "n_live": int(finite.sum()),
                "n_excluded": int((~finite).sum()),
                "n_at_bound": int((finite & at_bound).sum()),
                "n_beyond_bound": int((finite & beyond_bound).sum()),
            })
    return values, diagnostics, base_floor


def instruments():
    """Construct county instruments from certified exposures and shifters."""
    pnl = store.load_county_panel()
    units, piv = store.load_pivots(pnl, cols=("P",))
    units2, cats, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")

    years = store.target_years(H)
    shifters, diagnostics, base_floor = shifter_matrices(
        arr, contract_years, years)
    stored_current = store.load_gk(H, cats, contract_years)
    shifters["A_current"] = stored_current[:, [contract_years.index(t)
                                                for t in years]]
    z = {name: np.full((len(units), len(years)), np.nan, np.float64)
         for name in LABELS}

    for j, t in enumerate(years):
        exposure_hi = t - H - locked.LAG
        exposure_lo = exposure_hi - locked.SPAN + 1
        if exposure_lo < locked.DATA_FIRST:
            continue
        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        pbase = piv["P"][t - H].to_numpy(np.float64)
        for name, gmatrix in shifters.items():
            g = gmatrix[:, j]
            live = np.isfinite(g)
            if not live.any():
                raise SystemExit(f"FAIL: {name} has no live shifter at {t}")
            z[name][:, j] = exposure[:, live] @ g[live] / pbase
    return units, years, z, diagnostics, base_floor


def first_stage(d, z, t28):
    """Run one county-clustered first stage and compute its partial R-squared."""
    x = d["x"].to_numpy(np.float64)
    w = np.ones(len(d), np.float64)
    fe = pd.factorize(d["division"].astype(str) + "_"
                      + d["year"].astype(str))[0].astype(np.int64)
    nofe = np.zeros(len(d), np.int64)
    cluster = pd.factorize(d["ccg"])[0].astype(np.int64)
    fit = t28.fit2sls(x, z, z, w, fe, nofe, cluster, singleton=False)
    demeaned = t28.w22.demean(np.column_stack([x, z]), fe, nofe, w)
    xt, zt = demeaned[:, 0], demeaned[:, 1]
    partial_r2 = float((zt @ xt) ** 2 / ((zt @ zt) * (xt @ xt)))
    tstat = float(fit["beta"] / fit["se"])
    return {
        "first_stage": float(fit["beta"]),
        "first_stage_se": float(fit["se"]),
        "first_stage_t": tstat,
        "first_stage_F": tstat ** 2,
        "partial_r2": partial_r2,
    }


def aggregate_diagnostics(records):
    """Summarize category-year support and bounding over estimation years."""
    total_live = sum(r["n_live"] for r in records)
    total_at_bound = sum(r["n_at_bound"] for r in records)
    total_beyond = sum(r["n_beyond_bound"] for r in records)
    return {
        "min_live_categories": min(r["n_live"] for r in records),
        "max_live_categories": max(r["n_live"] for r in records),
        "max_excluded_categories": max(r["n_excluded"] for r in records),
        "share_live_at_bound": total_at_bound / total_live,
        "share_live_beyond_bound_before_clip": total_beyond / total_live,
    }


def main():
    start = time.time()
    print("== h=2 county first-stage transformation sequence")
    p57 = load_module("p57_normalization", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module("t28_normalization", STEP2_PY / "28_table_d1.py")
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

    units, years, zwide, diagnostics, base_floor = instruments()
    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy()
    ci = baseline["year"].map(colpos).to_numpy()
    if pd.isna(ri).any() or pd.isna(ci).any():
        raise SystemExit("FAIL: baseline rows do not align to the stored panel")
    ri, ci = ri.astype(int), ci.astype(int)

    current = baseline["z"].to_numpy(np.float64)
    reconstructed = zwide["A_current"][ri, ci]
    gap = np.nanmax(np.abs(reconstructed - current))
    scale = max(float(np.nanmean(np.abs(current))), 1e-300)
    if gap / scale >= locked.RTOL_COEF:
        raise SystemExit(
            f"FAIL: current instrument reconstruction gap is {gap / scale:.3e}")

    rows = []
    for name, label in LABELS.items():
        z = zwide[name][ri, ci]
        if not np.isfinite(z).all():
            raise SystemExit(
                f"FAIL: {name} is missing on "
                f"{int((~np.isfinite(z)).sum()):,} baseline rows")
        fit = first_stage(baseline, z, t28)
        diag = aggregate_diagnostics(diagnostics[name])
        rows.append({
            "instrument": name,
            "label": label,
            "horizon": H,
            "treatment": "two-year average-window contract change per capita",
            "sample_start": int(baseline["year"].min()),
            "sample_end": int(baseline["year"].max()),
            "n": len(baseline),
            "n_counties": int(baseline["ccg"].nunique()),
            "fixed_effects": "division-by-year",
            "weight": "equal county",
            "cluster": "county",
            "instrument_sd": float(np.std(z, ddof=1)),
            "correlation_with_current": float(np.corrcoef(current, z)[0, 1]),
            "base_screen_quantile":
                BASE_SCREEN_QUANTILE if name == "E_conventional_screened"
                else np.nan,
            "base_screen_dollar_floor":
                base_floor if name == "E_conventional_screened" else np.nan,
            **fit,
            **diag,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    for row in rows:
        print(f"   {row['instrument']}: F={row['first_stage_F']:.3f}, "
              f"partial R2={row['partial_r2']:.6f}, "
              f"pi={row['first_stage']:+.6f}, "
              f"t={row['first_stage_t']:+.3f}, "
              f"corr(current)={row['correlation_with_current']:+.3f}")
        print(f"      live categories {row['min_live_categories']}-"
              f"{row['max_live_categories']}, "
              f"max excluded {row['max_excluded_categories']}, "
              f"at bound {100 * row['share_live_at_bound']:.2f}%, "
              f"beyond bound {100 * row['share_live_beyond_bound_before_clip']:.2f}%")
    print(f"   conventional-growth 1% base floor: ${base_floor:,.2f}")
    print(f"   current instrument reconstruction gap: {gap / scale:.3e}")
    print(f"   wrote {OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
