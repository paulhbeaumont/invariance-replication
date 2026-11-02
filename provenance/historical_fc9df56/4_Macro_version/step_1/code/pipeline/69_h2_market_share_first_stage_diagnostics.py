"""Diagnose the h=2 market-share and locked county first stages.

The diagnostic separates first-stage signal by target year and procurement
category. It also reconstructs the county-clustered standard error from cluster
scores and decomposes its meat by temporal lag within county.

Writes Results/instrument_normalization/
    h2_market_share_first_stage_year_decomposition.csv
    h2_market_share_first_stage_category_decomposition.csv
    h2_market_share_first_stage_cluster_diagnostics.csv
    h2_market_share_first_stage_cluster_score_lags.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_market_share_first_stage_cluster_diagnostics.csv
# mpgr-new-estimates: 0
# mpgr-variation-scope: h2|market-share-vs-locked|first-stage-decomposition

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
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402
import store  # noqa: E402


H = 2
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
NEW_RESULT = OUTDIR / "h2_county_exposure_window_market_share_shifter.csv"
REFERENCE = OUTDIR / "h2_county_first_stage_sequence.csv"
YEAR_OUT = OUTDIR / "h2_market_share_first_stage_year_decomposition.csv"
CATEGORY_OUT = OUTDIR / "h2_market_share_first_stage_category_decomposition.csv"
CLUSTER_OUT = OUTDIR / "h2_market_share_first_stage_cluster_diagnostics.csv"
LAG_OUT = OUTDIR / "h2_market_share_first_stage_cluster_score_lags.csv"
TOL = 1e-10


def load_module(name, path):
    """Load one project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_results():
    """Read the two first-stage rows used as numerical fixtures."""
    new = pd.read_csv(NEW_RESULT)
    if len(new) != 1:
        raise SystemExit(f"FAIL: market-share output has {len(new)} rows")
    reference = pd.read_csv(REFERENCE)
    locked_row = reference.loc[reference["instrument"].eq("A_current")]
    if len(locked_row) != 1:
        raise SystemExit(
            f"FAIL: reference output has {len(locked_row)} locked rows")
    return {
        "market_share": new.iloc[0],
        "locked": locked_row.iloc[0],
    }


def within_first_stage(x, z, weight, fe, nofe, cluster, t28):
    """Return first-stage estimates, residualized arrays, and cluster scores."""
    demeaned = t28.w22.demean(
        np.column_stack([x, z]), fe, nofe, weight)
    xt, zt = demeaned[:, 0], demeaned[:, 1]
    zz = float(zt @ (weight * zt))
    zx = float(zt @ (weight * xt))
    if not np.isfinite(zz) or zz <= 0 or not np.isfinite(zx):
        raise SystemExit("FAIL: invalid within first-stage moments")
    beta = zx / zz
    residual = xt - beta * zt
    score_obs = weight * zt * residual
    score_cluster = np.bincount(
        cluster, weights=score_obs, minlength=int(cluster.max()) + 1)
    n = len(x)
    k = t28.ssc_k(fe, nofe, cluster)
    g = int(np.unique(cluster).size)
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    meat = float(score_cluster @ score_cluster)
    se = float(np.sqrt(correction * meat / zz ** 2))
    tstat = beta / se
    partial_r2 = zx ** 2 / (zz * float(xt @ (weight * xt)))
    return {
        "beta": beta,
        "se": se,
        "t": tstat,
        "F": tstat ** 2,
        "partial_r2": partial_r2,
        "xt": xt,
        "zt": zt,
        "residual": residual,
        "score_obs": score_obs,
        "score_cluster": score_cluster,
        "zz": zz,
        "zx": zx,
        "correction": correction,
        "meat": meat,
    }


def assert_fixture(name, result, fixture):
    """Require the reconstructed first stage to match its persisted row."""
    mapping = {
        "beta": "first_stage",
        "se": "first_stage_se",
        "t": "first_stage_t",
        "F": "first_stage_F",
        "partial_r2": "partial_r2",
    }
    for calculated, saved in mapping.items():
        left = float(result[calculated])
        right = float(fixture[saved])
        gap = abs(left - right) / max(abs(right), 1e-300)
        if gap >= locked.RTOL_COEF:
            raise SystemExit(
                f"FAIL: {name} {calculated} fixture relative gap is {gap:.3e}")


def cluster_diagnostics(name, result, cluster, years, county_ids, t28):
    """Summarize county score concentration and temporal covariance."""
    unique_years = np.array(sorted(np.unique(years)))
    year_pos = pd.Series(range(len(unique_years)), index=unique_years)
    yi = pd.Series(years).map(year_pos).to_numpy(np.int64)
    n_cluster = int(cluster.max()) + 1
    score_matrix = np.zeros((n_cluster, len(unique_years)), np.float64)
    counts = np.zeros_like(score_matrix, np.int16)
    score_matrix[cluster, yi] = result["score_obs"]
    counts[cluster, yi] += 1
    if not np.all(counts == 1):
        raise SystemExit(
            f"FAIL: {name} county-year score matrix is not balanced")

    score_cluster = score_matrix.sum(axis=1)
    max_score_gap = float(np.max(np.abs(
        score_cluster - result["score_cluster"])))
    max_score_scale = float(np.max(np.abs(result["score_cluster"])))
    if max_score_gap > 1e-12 * max(max_score_scale, 1.0):
        raise SystemExit(
            f"FAIL: {name} cluster-score reconstruction relative gap is "
            f"{max_score_gap / max(max_score_scale, 1.0):.3e}")

    lag_rows = []
    contributions = []
    for lag in range(len(unique_years)):
        if lag == 0:
            contribution = float(np.square(score_matrix).sum())
        else:
            contribution = float(2.0 * (
                score_matrix[:, lag:] * score_matrix[:, :-lag]).sum())
        contributions.append(contribution)
        lag_rows.append({
            "instrument": name,
            "lag": lag,
            "score_meat_contribution": contribution,
            "share_of_cluster_meat": contribution / result["meat"],
        })
    if not np.isclose(
            sum(contributions), result["meat"], rtol=TOL, atol=1e-6):
        raise SystemExit(f"FAIL: {name} lag decomposition does not sum to meat")

    diagonal_meat = contributions[0]
    cross_year_meat = result["meat"] - diagonal_meat
    se_diagonal_only = float(np.sqrt(
        result["correction"] * diagonal_meat / result["zz"] ** 2))
    squared_scores = np.square(score_cluster)
    shares = squared_scores / squared_scores.sum()
    effective_counties = 1.0 / float(shares @ shares)
    top = int(np.argmax(shares))
    cluster_ids = np.empty(n_cluster, dtype=object)
    for code in range(n_cluster):
        values = np.unique(county_ids[cluster == code])
        if len(values) != 1:
            raise SystemExit(
                f"FAIL: {name} cluster {code} maps to {len(values)} counties")
        cluster_ids[code] = values[0]

    row = {
        "instrument": name,
        "first_stage": result["beta"],
        "first_stage_se_county_clustered": result["se"],
        "first_stage_t": result["t"],
        "first_stage_F": result["F"],
        "partial_r2": result["partial_r2"],
        "within_z_sum_squares": result["zz"],
        "within_zx_cross_product": result["zx"],
        "cluster_meat": result["meat"],
        "diagonal_year_meat": diagonal_meat,
        "cross_year_meat": cross_year_meat,
        "cross_year_share_of_cluster_meat":
            cross_year_meat / result["meat"],
        "se_diagonal_year_terms_only": se_diagonal_only,
        "cluster_to_diagonal_se_ratio": result["se"] / se_diagonal_only,
        "effective_counties_in_cluster_meat": effective_counties,
        "largest_county_meat_share": float(shares[top]),
        "largest_county_meat_ccg": cluster_ids[top],
    }
    return row, lag_rows


def category_covariance_contributions(
        arr, categories, contract_years, years, baseline, ri, population,
        xt, g_locked):
    """Accumulate category contributions to the first-stage covariance."""
    national = arr.sum(axis=0, dtype=np.float64)
    contribution = {
        "market_share": np.zeros(len(categories), np.float64),
        "locked": np.zeros(len(categories), np.float64),
    }

    baseline_year = baseline["year"].to_numpy(np.int64)
    for t in sorted(np.unique(baseline_year)):
        sample = np.flatnonzero(baseline_year == t)
        unit_rows = ri[sample]
        exposure_hi = t - H - locked.LAG
        exposure_lo = exposure_hi - locked.SPAN + 1
        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure = arr[unit_rows, :, a:b + 1].mean(axis=2).astype(np.float64)
        national_exposure = arr[:, :, a:b + 1].mean(
            axis=2).sum(axis=0, dtype=np.float64)
        live = national_exposure > 0

        post2 = 0.5 * (
            national[:, contract_years.index(t)]
            + national[:, contract_years.index(t - 1)])
        base = national[:, contract_years.index(t - H)]
        g_market = np.zeros(len(categories), np.float64)
        g_market[live] = (
            post2[live] - base[live]) / national_exposure[live]
        g_lock_t = g_locked[:, contract_years.index(t)]
        if not np.isfinite(g_lock_t).all():
            raise SystemExit(f"FAIL: locked category shifter is missing in {t}")

        pbase = population[t - H].to_numpy(np.float64)[unit_rows]
        component_scale = exposure / pbase[:, None]
        xyear = xt[sample]
        contribution["market_share"] += (
            component_scale * g_market[None, :]).T @ xyear
        contribution["locked"] += (
            component_scale * g_lock_t[None, :]).T @ xyear

    return contribution


def decomposition_rows(name, values, labels, total):
    """Create signed and absolute contribution rows."""
    if not np.isclose(values.sum(), total, rtol=TOL, atol=1e-5):
        raise SystemExit(
            f"FAIL: {name} contributions sum to {values.sum():.6e}, "
            f"expected {total:.6e}")
    abs_total = float(np.abs(values).sum())
    order = np.argsort(-np.abs(values))
    ranks = np.empty(len(values), np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return [{
        "instrument": name,
        "component": label,
        "within_zx_contribution": float(value),
        "signed_share_of_total_within_zx": float(value / total),
        "absolute_share_of_absolute_contributions":
            float(abs(value) / abs_total),
        "absolute_contribution_rank": int(rank),
    } for label, value, rank in zip(labels, values, ranks)]


def main():
    """Run the first-stage signal and cluster-score diagnostics."""
    start = time.time()
    print("== h=2 market-share first-stage diagnostics")
    p57 = load_module(
        "p57_market_diagnostics", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_market_diagnostics", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_market_diagnostics",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_market_diagnostics",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(
        fixef_tol=1e-12, fixef_maxiter=100_000)

    baseline, dropped = p57.build_county(
        "composition", h=H, span=locked.SPAN, rule=locked.RULE,
        floor=locked.FLOOR, denom=locked.DENOM)
    if dropped != 0 or len(baseline) != locked.N_COUNTY[H]:
        raise SystemExit(
            f"FAIL: baseline has {len(baseline):,} rows and {dropped} drops")

    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("P",))
    units2, categories, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    years = store.target_years(H)
    market_wide, _ = p68.build_instrument(
        arr, contract_years, years, pivots["P"])
    locked_wide = store.load_instrument(H, units, "z")

    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy()
    ci = baseline["year"].map(colpos).to_numpy()
    if pd.isna(ri).any() or pd.isna(ci).any():
        raise SystemExit("FAIL: baseline rows do not align to stored instruments")
    ri, ci = ri.astype(int), ci.astype(int)
    instruments = {
        "market_share": market_wide[ri, ci],
        "locked": locked_wide[ri, ci],
    }
    if any(not np.isfinite(z).all() for z in instruments.values()):
        raise SystemExit("FAIL: an instrument has missing sample values")

    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    fixtures = saved_results()
    results = {}
    for name, z in instruments.items():
        results[name] = within_first_stage(
            x, z, weight, fe, nofe, cluster, t28)
        assert_fixture(name, results[name], fixtures[name])

    year_rows = []
    sample_years = baseline["year"].to_numpy(np.int64)
    for name, result in results.items():
        total_zx = result["zx"]
        total_zz = result["zz"]
        values = []
        for year in sorted(np.unique(sample_years)):
            selected = sample_years == year
            zx = float(result["zt"][selected] @ result["xt"][selected])
            zz = float(result["zt"][selected] @ result["zt"][selected])
            values.append(abs(zx))
            year_rows.append({
                "instrument": name,
                "year": int(year),
                "within_zx_contribution": zx,
                "signed_share_of_total_within_zx": zx / total_zx,
                "within_z_sum_squares": zz,
                "share_of_total_within_z_sum_squares": zz / total_zz,
            })
        abs_total = sum(values)
        positions = [i for i, row in enumerate(year_rows)
                     if row["instrument"] == name]
        for pos, value in zip(positions, values):
            year_rows[pos]["absolute_share_of_absolute_zx"] = value / abs_total
        if not np.isclose(
                sum(year_rows[pos]["within_zx_contribution"]
                    for pos in positions), total_zx, rtol=TOL, atol=1e-5):
            raise SystemExit(f"FAIL: {name} annual covariance does not sum")

    cluster_rows = []
    lag_rows = []
    county_ids = baseline["ccg"].to_numpy()
    for name, result in results.items():
        row, lags = cluster_diagnostics(
            name, result, cluster, sample_years, county_ids, t28)
        cluster_rows.append(row)
        lag_rows.extend(lags)

    g_locked = store.load_gk(H, categories, contract_years)
    category_values = category_covariance_contributions(
        arr, categories, contract_years, years, baseline, ri, pivots["P"],
        results["market_share"]["xt"], g_locked)
    category_rows = []
    for name, values in category_values.items():
        category_rows.extend(decomposition_rows(
            name, values, categories, results[name]["zx"]))
        absolute_shares = np.abs(values) / np.abs(values).sum()
        row = next(item for item in cluster_rows if item["instrument"] == name)
        row["effective_categories_in_absolute_zx"] = float(
            1.0 / (absolute_shares @ absolute_shares))
        top = int(np.argmax(absolute_shares))
        row["largest_category_absolute_zx_share"] = float(
            absolute_shares[top])
        row["largest_category"] = categories[top]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(year_rows).to_csv(YEAR_OUT, index=False)
    pd.DataFrame(category_rows).to_csv(CATEGORY_OUT, index=False)
    pd.DataFrame(cluster_rows).to_csv(CLUSTER_OUT, index=False)
    pd.DataFrame(lag_rows).to_csv(LAG_OUT, index=False)

    for row in cluster_rows:
        print(f"   {row['instrument']:>12}: F={row['first_stage_F']:.3f}, "
              f"SE={row['first_stage_se_county_clustered']:.6f}, "
              f"SE(diagonal)={row['se_diagonal_year_terms_only']:.6f}, "
              f"cross-year meat share="
              f"{100 * row['cross_year_share_of_cluster_meat']:.2f}%, "
              f"effective counties="
              f"{row['effective_counties_in_cluster_meat']:.1f}, "
              f"effective categories="
              f"{row['effective_categories_in_absolute_zx']:.1f}")
    print(f"   wrote {YEAR_OUT}")
    print(f"   wrote {CATEGORY_OUT}")
    print(f"   wrote {CLUSTER_OUT}")
    print(f"   wrote {LAG_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
