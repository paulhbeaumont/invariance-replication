"""Exact delete-one-county influence audit for the h=2 first stage.

The market-share instrument is held fixed. Each specification removes one
county's full time series, recomputes division-by-year fixed effects, and
recomputes the county-clustered standard error. Balanced-panel sufficient
statistics make all county deletions algebraically exact without running 3,058
separate projection commands.

Writes Results/instrument_normalization/
    h2_market_share_delete_one_county_first_stage.csv
    h2_market_share_delete_one_county_summary.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_market_share_delete_one_county_first_stage.csv
# mpgr-new-estimates: 3058
# mpgr-variation-scope: h2|market-share|fixed-instrument|delete-one-county|exact-first-stage

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
AUSTIN_CCG = 48015
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
DETAIL_OUT = OUTDIR / "h2_market_share_delete_one_county_first_stage.csv"
SUMMARY_OUT = OUTDIR / "h2_market_share_delete_one_county_summary.csv"
BASELINE_REFERENCE = OUTDIR / "h2_county_exposure_window_market_share_shifter.csv"
AUSTIN_REFERENCE = OUTDIR / "h2_market_share_austin_county_leaveout.csv"
CHECK_RTOL = 1e-9


def load_module(name, path):
    """Load one project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relative_gap(left, right):
    """Absolute relative gap with a finite scale at zero."""
    return abs(float(left) - float(right)) / max(abs(float(right)), 1e-300)


def assert_result(label, calculated, reference):
    """Check one calculated first stage against a saved or direct result."""
    fields = {
        "first_stage": "first_stage",
        "first_stage_se": "first_stage_se",
        "first_stage_t": "first_stage_t",
        "first_stage_F": "first_stage_F",
        "partial_r2": "partial_r2",
    }
    for calculated_name, reference_name in fields.items():
        gap = relative_gap(
            calculated[calculated_name], reference[reference_name])
        if gap > CHECK_RTOL:
            raise SystemExit(
                f"FAIL: {label} {calculated_name} relative gap is {gap:.3e}")


def direct_result(baseline, z, excluded_ccg, helper, p69, t28):
    """Run the ordinary projection command for one validation deletion."""
    keep = baseline["ccg"].to_numpy() != excluded_ccg
    sample = baseline.loc[keep]
    x, weight, fe, nofe, cluster = helper.estimation_arrays(sample)
    result = p69.within_first_stage(
        x, z[keep], weight, fe, nofe, cluster, t28)
    return {
        "first_stage": result["beta"],
        "first_stage_se": result["se"],
        "first_stage_t": result["t"],
        "first_stage_F": result["F"],
        "partial_r2": result["partial_r2"],
    }


def exact_deletions(xmat, zmat, divisions, n_fe, full):
    """Return exact OLS and CRV1 results after each county deletion."""
    counties = xmat.index.to_numpy()
    x = xmat.to_numpy(np.float64)
    z = zmat.to_numpy(np.float64)
    group_codes, group_labels = pd.factorize(divisions.reindex(counties))
    if np.any(group_codes < 0):
        raise SystemExit("FAIL: a county has no division label")

    xt = np.empty_like(x)
    zt = np.empty_like(z)
    group_sizes = np.zeros(len(group_labels), np.int64)
    for group in range(len(group_labels)):
        idx = np.flatnonzero(group_codes == group)
        group_sizes[group] = len(idx)
        if len(idx) <= 1:
            raise SystemExit(
                f"FAIL: division {group_labels[group]} has only {len(idx)} county")
        xt[idx] = x[idx] - x[idx].mean(axis=0)
        zt[idx] = z[idx] - z[idx].mean(axis=0)

    county_zx = np.einsum("gt,gt->g", zt, xt)
    county_zz = np.einsum("gt,gt->g", zt, zt)
    county_xx = np.einsum("gt,gt->g", xt, xt)
    total_zx = float(county_zx.sum())
    total_zz = float(county_zz.sum())
    total_xx = float(county_xx.sum())
    beta_full = total_zx / total_zz
    score_full = county_zx - beta_full * county_zz
    n_counties, n_years = x.shape
    correction_full = (
        n_counties / (n_counties - 1)
        * ((n_counties * n_years - 1) / (n_counties * n_years - n_fe)))
    se_full = np.sqrt(
        correction_full * float(score_full @ score_full) / total_zz ** 2)
    calculated_full = {
        "first_stage": beta_full,
        "first_stage_se": se_full,
        "first_stage_t": beta_full / se_full,
        "first_stage_F": (beta_full / se_full) ** 2,
        "partial_r2": total_zx ** 2 / (total_zz * total_xx),
    }
    assert_result("full sample sufficient statistics", calculated_full, full)

    beta = np.empty(n_counties, np.float64)
    se = np.empty(n_counties, np.float64)
    partial_r2 = np.empty(n_counties, np.float64)
    meat = np.empty(n_counties, np.float64)
    remaining_counties = n_counties - 1
    remaining_n = remaining_counties * n_years
    correction = (
        remaining_counties / (remaining_counties - 1)
        * ((remaining_n - 1) / (remaining_n - n_fe)))
    sum_a2 = float(county_zx @ county_zx)
    sum_ab = float(county_zx @ county_zz)
    sum_b2 = float(county_zz @ county_zz)

    for group in range(len(group_labels)):
        idx = np.flatnonzero(group_codes == group)
        size = len(idx)
        scale = size / (size - 1)
        zx_deleted = total_zx - scale * county_zx[idx]
        zz_deleted = total_zz - scale * county_zz[idx]
        xx_deleted = total_xx - scale * county_xx[idx]
        beta_deleted = zx_deleted / zz_deleted

        zg, xg = zt[idx], xt[idx]
        cross_zx = zg @ xg.T
        cross_xz = xg @ zg.T
        cross_zz = zg @ zg.T
        denominator = size - 1
        adjusted_zx = (
            county_zx[idx, None]
            + (cross_zx + cross_xz) / denominator
            + county_zx[None, idx] / denominator ** 2)
        adjusted_zz = (
            county_zz[idx, None]
            + 2.0 * cross_zz / denominator
            + county_zz[None, idx] / denominator ** 2)
        adjusted_scores = (
            adjusted_zx - adjusted_zz * beta_deleted[None, :])
        adjusted_scores[np.arange(size), np.arange(size)] = 0.0

        all_meat = (
            sum_a2 - 2.0 * beta_deleted * sum_ab
            + beta_deleted ** 2 * sum_b2)
        unadjusted_group_scores = (
            county_zx[idx, None]
            - county_zz[idx, None] * beta_deleted[None, :])
        meat_deleted = (
            all_meat
            - np.square(unadjusted_group_scores).sum(axis=0)
            + np.square(adjusted_scores).sum(axis=0))
        if np.any(meat_deleted <= 0) or np.any(zz_deleted <= 0):
            raise SystemExit(
                f"FAIL: division {group_labels[group]} has invalid moments")

        beta[idx] = beta_deleted
        meat[idx] = meat_deleted
        se[idx] = np.sqrt(correction * meat_deleted / zz_deleted ** 2)
        partial_r2[idx] = zx_deleted ** 2 / (zz_deleted * xx_deleted)

    tstat = beta / se
    return pd.DataFrame({
        "ccg": counties,
        "division": divisions.reindex(counties).to_numpy(),
        "division_counties": group_sizes[group_codes],
        "n": remaining_n,
        "n_counties": remaining_counties,
        "first_stage": beta,
        "first_stage_se": se,
        "first_stage_t": tstat,
        "first_stage_F": np.square(tstat),
        "partial_r2": partial_r2,
        "cluster_meat": meat,
    })


def summary_rows(detail, full):
    """Build a compact distribution and extremum summary."""
    rows = [{
        "statistic": "full_sample",
        "excluded_ccg": "none",
        "excluded_county": "none",
        **full,
    }]
    extrema = (
        ("minimum_F", "first_stage_F", "min"),
        ("maximum_F", "first_stage_F", "max"),
        ("minimum_coefficient", "first_stage", "min"),
        ("maximum_coefficient", "first_stage", "max"),
        ("largest_absolute_coefficient_change", "beta_change", "absmax"),
    )
    for label, column, kind in extrema:
        if kind == "min":
            pos = detail[column].idxmin()
        elif kind == "max":
            pos = detail[column].idxmax()
        else:
            pos = detail[column].abs().idxmax()
        row = detail.loc[pos]
        rows.append({
            "statistic": label,
            "excluded_ccg": int(row["ccg"]),
            "excluded_county": row["county"],
            "first_stage": row["first_stage"],
            "first_stage_se": row["first_stage_se"],
            "first_stage_t": row["first_stage_t"],
            "first_stage_F": row["first_stage_F"],
            "partial_r2": row["partial_r2"],
        })
    for quantile in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99):
        rows.append({
            "statistic": f"deletion_quantile_{quantile:.2f}",
            "excluded_ccg": "distribution",
            "excluded_county": "distribution",
            "first_stage": detail["first_stage"].quantile(quantile),
            "first_stage_se": detail["first_stage_se"].quantile(quantile),
            "first_stage_t": detail["first_stage_t"].quantile(quantile),
            "first_stage_F": detail["first_stage_F"].quantile(quantile),
            "partial_r2": detail["partial_r2"].quantile(quantile),
        })
    return pd.DataFrame(rows)


def main():
    """Run the exact county deletion audit and its direct validations."""
    start = time.time()
    print("== h=2 market-share delete-one-county first stages")
    p57 = load_module(
        "p57_delete_county", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_delete_county", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_delete_county",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_delete_county",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_delete_county",
        PIPELINE / "69_h2_market_share_first_stage_diagnostics.py")
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
    units2, _, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    years = store.target_years(H)
    zwide, _ = p68.build_instrument(
        arr, contract_years, years, pivots["P"])
    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy(np.int64)
    ci = baseline["year"].map(colpos).to_numpy(np.int64)
    z = zwide[ri, ci]
    if not np.isfinite(z).all():
        raise SystemExit("FAIL: market-share instrument has missing sample values")

    full_reference = pd.read_csv(BASELINE_REFERENCE)
    if len(full_reference) != 1:
        raise SystemExit("FAIL: full-sample reference row is incomplete")
    full = full_reference.iloc[0][[
        "first_stage", "first_stage_se", "first_stage_t",
        "first_stage_F", "partial_r2"]].to_dict()
    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    n_fe = t28.ssc_k(fe, nofe, cluster)

    sample = baseline[["ccg", "year", "division", "x"]].copy()
    sample["z"] = z
    counties = np.sort(sample["ccg"].unique())
    sample_years = np.sort(sample["year"].unique())
    expected = len(counties) * len(sample_years)
    if len(sample) != expected:
        raise SystemExit(
            f"FAIL: sample has {len(sample):,} rows, expected {expected:,}")
    xmat = sample.pivot(index="ccg", columns="year", values="x").reindex(
        index=counties, columns=sample_years)
    zmat = sample.pivot(index="ccg", columns="year", values="z").reindex(
        index=counties, columns=sample_years)
    if xmat.isna().any().any() or zmat.isna().any().any():
        raise SystemExit("FAIL: balanced first-stage matrices contain missing data")
    divisions = sample.drop_duplicates("ccg").set_index("ccg")["division"]
    detail = exact_deletions(xmat, zmat, divisions, n_fe, full)

    names = pd.read_csv(
        STEP1_PY.parent / "reference" / "county_ccg_crosswalk.csv")
    names["ccg"] = pd.to_numeric(names["ccg_id"], errors="coerce")
    names = names.dropna(subset=["ccg"]).drop_duplicates("ccg")
    names["ccg"] = names["ccg"].astype(np.int64)
    detail["county"] = detail["ccg"].map(names.set_index("ccg")["ccg_name"])
    fallback = detail["ccg"].map(lambda value: f"CCG {int(value):05d}")
    detail["county"] = detail["county"].fillna(fallback)

    detail["beta_change"] = detail["first_stage"] - float(full["first_stage"])
    detail["se_change"] = (
        detail["first_stage_se"] - float(full["first_stage_se"]))
    detail["F_change"] = detail["first_stage_F"] - float(full["first_stage_F"])
    detail["partial_r2_change"] = (
        detail["partial_r2"] - float(full["partial_r2"]))
    detail["absolute_beta_change_rank"] = (
        detail["beta_change"].abs().rank(method="min", ascending=False)
        .astype(np.int64))
    detail["F_rank"] = (
        detail["first_stage_F"].rank(method="min", ascending=False)
        .astype(np.int64))

    austin_reference = pd.read_csv(AUSTIN_REFERENCE)
    austin_reference = austin_reference.loc[
        austin_reference["instrument"].eq("market_share")
        & austin_reference["excluded_ccg"].astype(str).eq(str(AUSTIN_CCG))]
    if len(austin_reference) != 1:
        raise SystemExit("FAIL: Austin deletion reference row is incomplete")
    austin = detail.loc[detail["ccg"].eq(AUSTIN_CCG)]
    if len(austin) != 1:
        raise SystemExit("FAIL: calculated Austin deletion row is incomplete")
    assert_result(
        "Austin deletion sufficient statistics",
        austin.iloc[0].to_dict(), austin_reference.iloc[0].to_dict())

    validation_ids = {
        AUSTIN_CCG,
        int(detail.loc[detail["first_stage_F"].idxmin(), "ccg"]),
        int(detail.loc[detail["beta_change"].abs().idxmax(), "ccg"]),
    }
    for ccg in sorted(validation_ids):
        direct = direct_result(baseline, z, ccg, helper, p69, t28)
        calculated = detail.loc[detail["ccg"].eq(ccg)].iloc[0].to_dict()
        assert_result(f"direct deletion {ccg}", calculated, direct)
        print(f"   direct projection check {ccg}: PASS")

    summary = summary_rows(detail, full)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail.sort_values("ccg").to_csv(DETAIL_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    lowest = detail.loc[detail["first_stage_F"].idxmin()]
    highest = detail.loc[detail["first_stage_F"].idxmax()]
    influential = detail.loc[detail["beta_change"].abs().idxmax()]
    print(
        f"   full sample: beta={full['first_stage']:+.6f}, "
        f"SE={full['first_stage_se']:.6f}, F={full['first_stage_F']:.3f}")
    print(
        f"   deletion F range: {lowest['first_stage_F']:.3f} "
        f"({int(lowest['ccg'])}, {lowest['county']}) to "
        f"{highest['first_stage_F']:.3f} "
        f"({int(highest['ccg'])}, {highest['county']})")
    print(
        f"   largest |beta change|: {int(influential['ccg'])}, "
        f"{influential['county']}, {influential['beta_change']:+.6f}")
    print(
        f"   deletions with F < 10: "
        f"{int((detail['first_stage_F'] < 10).sum()):,} / {len(detail):,}")
    print(f"   wrote {DETAIL_OUT}")
    print(f"   wrote {SUMMARY_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
