"""Diagnose persistence in the annual shocks used for the two-year episode.

The diagnostic keeps the equal-county weighting, division-by-year fixed
effects, and focal share-sum controls from producer 53. On one common sample,
it asks whether annual shocks dated t and t-1 remain predictable when the
existing controls z_{t-2}, z_{t-3}, and z_{t-4} are expanded through z_{t-8}.

Writes two files beside the annual-innovation LP outputs:

  annual_shock_predictability.csv
  annual_shock_autocorrelation.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import instrument  # noqa: E402
import locked  # noqa: E402
import oldpath as op  # noqa: E402
import store  # noqa: E402

OUT = PIPELINE.parents[2] / "Results" / "lp_2yr_annual_innovations"
MAX_LAG = 8
BASE_MAX_LAG = 4
Z1_FIRST = locked.DATA_FIRST + locked.LAG + locked.SPAN
YEAR_LO = Z1_FIRST + MAX_LAG
YEAR_HI = locked.TARGET_LAST - 4
N_EXPECT = locked.N_COUNTIES * (YEAR_HI - YEAR_LO + 1)
np.random.seed(0)


def residualize(y, controls):
    """Return OLS residuals and the within R-squared."""
    if controls.shape[1] == 0:
        residual = y.copy()
    else:
        beta, _, rank, _ = np.linalg.lstsq(controls, y, rcond=None)
        if rank != controls.shape[1]:
            raise SystemExit(
                f"FAIL: control matrix rank {rank}, expected "
                f"{controls.shape[1]}")
        residual = y - controls @ beta
    tss = float(y @ y)
    rss = float(residual @ residual)
    return residual, 1.0 - rss / tss


def load_frame():
    """Construct annual shocks, focal share sums, and fixed-effect codes."""
    panel = store.load_county_panel()
    units, piv = store.load_pivots(panel, cols=("P",))
    division = panel.drop_duplicates("ccg").set_index("ccg")["division"]
    units2, _, cy, arr, nat = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")

    target = store.target_years(1)
    g1 = instrument.category_rates(nat, 1)
    numerator, _ = instrument.predicted_dollars(
        arr, g1, cy, target, 1, "roll", locked.SPAN,
        locked.LAG, locked.DATA_FIRST)
    denominator = piv["P"].reindex(
        columns=[year - 1 for year in target]).to_numpy(np.float64)
    zfull = numerator / denominator
    zpos = {year: q for q, year in enumerate(target)}

    years = list(range(YEAR_LO, YEAR_HI + 1))
    shocks = {
        lag: zfull[:, [zpos[year - lag] for year in years]]
        for lag in range(MAX_LAG + 1)
    }

    share_sums = {}
    for focal_lag in (0, 1):
        values = np.full_like(shocks[focal_lag], np.nan)
        dates = [year - focal_lag for year in years]
        pbase = piv["P"].reindex(
            columns=[date - 1 for date in dates]).to_numpy(np.float64)
        for q, date in enumerate(dates):
            hi = date - 1 - locked.LAG
            lo = hi - locked.SPAN + 1
            a, b = cy.index(lo), cy.index(hi)
            exposure = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
            values[:, q] = exposure.sum(axis=1) / pbase[:, q]
        share_sums[focal_lag] = values

    values = list(shocks.values()) + list(share_sums.values())
    ok = np.logical_and.reduce([np.isfinite(value) for value in values])
    if not ok.all():
        raise SystemExit(
            f"FAIL: common frame loses {int((~ok).sum()):,} county-years")
    if int(ok.sum()) != N_EXPECT:
        raise SystemExit(
            f"FAIL: common frame has {int(ok.sum()):,} rows, expected "
            f"{N_EXPECT:,}")

    ri, ci = np.where(ok)
    year_array = np.array(years)
    frame = pd.DataFrame({
        "ccg": units[ri],
        "year": year_array[ci],
        "division": pd.Series(units[ri]).map(division).to_numpy(),
    })
    if frame["division"].isna().any():
        raise SystemExit("FAIL: division is missing on the diagnostic frame")

    raw = {f"zlag{lag}": value[ok] for lag, value in shocks.items()}
    raw["s0_current"] = share_sums[0][ok]
    raw["s0_prior"] = share_sums[1][ok]
    fe = pd.factorize(
        frame["division"].astype(str) + "_" + frame["year"].astype(str)
    )[0].astype(np.int64)
    nofe = np.zeros(len(frame), np.int64)
    weights = np.ones(len(frame), np.float64)

    import pyfixest as pf
    op.w22.DEMEANER = pf.MapDemeaner(
        fixef_tol=1e-10, fixef_maxiter=100_000)
    names = sorted(raw)
    matrix = op.w22.demean(
        np.column_stack([raw[name] for name in names]),
        fe, nofe, weights)
    demeaned = {name: matrix[:, q] for q, name in enumerate(names)}
    return frame, demeaned


def predictability_table(frame, dm):
    """Measure the incremental explanatory power of deeper shock lags."""
    share_names = ("s0_current", "s0_prior")
    rows = []
    residuals = {}
    for max_lag in range(BASE_MAX_LAG, MAX_LAG + 1):
        controls = np.column_stack(
            [dm[name] for name in share_names]
            + [dm[f"zlag{lag}"] for lag in range(2, max_lag + 1)])
        for target_lag, target_label in ((0, "z_t"), (1, "z_t-1")):
            residual, r2 = residualize(dm[f"zlag{target_lag}"], controls)
            residuals[(max_lag, target_lag)] = residual
            rows.append({
                "target": target_label,
                "max_past_shock_lag": max_lag,
                "past_shock_controls": f"z_t-2 through z_t-{max_lag}",
                "within_r2": r2,
                "residual_sd": float(np.sqrt(np.mean(residual ** 2))),
                "n": len(frame),
                "n_counties": int(frame["ccg"].nunique()),
                "year_lo": YEAR_LO,
                "year_hi": YEAR_HI,
            })

    out = pd.DataFrame(rows)
    for target in ("z_t", "z_t-1"):
        base_r2 = float(out.loc[
            (out["target"] == target)
            & (out["max_past_shock_lag"] == BASE_MAX_LAG),
            "within_r2"].iloc[0])
        mask = out["target"] == target
        out.loc[mask, "delta_r2_vs_lag4"] = (
            out.loc[mask, "within_r2"] - base_r2)
        out.loc[mask, "partial_r2_vs_lag4"] = (
            out.loc[mask, "delta_r2_vs_lag4"] / (1.0 - base_r2))

    for max_lag in range(BASE_MAX_LAG, MAX_LAG + 1):
        r0 = residuals[(max_lag, 0)]
        r1 = residuals[(max_lag, 1)]
        corr = float(np.corrcoef(r0, r1)[0, 1])
        out.loc[
            out["max_past_shock_lag"] == max_lag,
            "focal_residual_correlation"] = corr
    return out


def autocorrelation_table(frame, dm):
    """Compute annual-shock correlations after fixed effects and share sums."""
    shares = np.column_stack([dm["s0_current"], dm["s0_prior"]])
    residuals = {
        lag: residualize(dm[f"zlag{lag}"], shares)[0]
        for lag in range(MAX_LAG + 1)
    }
    current = residuals[0]
    return pd.DataFrame([{
        "lag": lag,
        "correlation_with_z_t": float(
            np.corrcoef(current, residuals[lag])[0, 1]),
        "conditioning": "division-by-year FE and focal share sums",
        "n": len(frame),
        "n_counties": int(frame["ccg"].nunique()),
        "year_lo": YEAR_LO,
        "year_hi": YEAR_HI,
    } for lag in range(1, MAX_LAG + 1)])


def main():
    print("== stage 54: annual-shock predictability diagnostic")
    frame, demeaned = load_frame()
    prediction = predictability_table(frame, demeaned)
    autocorrelation = autocorrelation_table(frame, demeaned)
    OUT.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(OUT / "annual_shock_predictability.csv", index=False)
    autocorrelation.to_csv(
        OUT / "annual_shock_autocorrelation.csv", index=False)

    wide = prediction.pivot(
        index="max_past_shock_lag", columns="target", values="within_r2")
    base, deep = wide.loc[BASE_MAX_LAG], wide.loc[MAX_LAG]
    corr_base = float(prediction.loc[
        prediction["max_past_shock_lag"] == BASE_MAX_LAG,
        "focal_residual_correlation"].iloc[0])
    corr_deep = float(prediction.loc[
        prediction["max_past_shock_lag"] == MAX_LAG,
        "focal_residual_correlation"].iloc[0])
    print(f"   sample [{YEAR_LO},{YEAR_HI}]: {len(frame):,} county-years")
    print(
        f"   z_t within R2: lag 4={base['z_t']:.6f}, "
        f"lag 8={deep['z_t']:.6f}, delta={deep['z_t']-base['z_t']:.6f}")
    print(
        f"   z_t-1 within R2: lag 4={base['z_t-1']:.6f}, "
        f"lag 8={deep['z_t-1']:.6f}, "
        f"delta={deep['z_t-1']-base['z_t-1']:.6f}")
    print(
        f"   focal residual correlation: lag 4={corr_base:.6f}, "
        f"lag 8={corr_deep:.6f}")
    print(f"   wrote {OUT / 'annual_shock_predictability.csv'}")
    print(f"   wrote {OUT / 'annual_shock_autocorrelation.csv'}")


if __name__ == "__main__":
    main()
