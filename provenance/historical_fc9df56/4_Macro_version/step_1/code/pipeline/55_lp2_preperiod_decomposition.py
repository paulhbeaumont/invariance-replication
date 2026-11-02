"""Decompose the two-year LP pre-period coefficients into annual increments.

This diagnostic uses producer 53's exact 1987-2012 frame, equal county
weights, fixed effects, two focal annual shocks, share-sum controls, and shock
lags 2 through 4. It directly estimates the three disjoint annual outcome
increments underlying the nested pre-period windows, then verifies that their
coefficients reconstruct every reported two-year pre-period coefficient.

The component estimates are also aligned with producer 50's h=1 equal-weight,
p=4, fixed-sample increment estimates.

Writes Results/lp_2yr_annual_innovations/:

  lp2_preperiod_increment_mapping.csv
  lp2_preperiod_reconstruction.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
OUT = ROOT / "Results" / "lp_2yr_annual_innovations"
H1_GRID = ROOT / "Results" / "lp" / "lp_pretrend_grid.csv"
H2_RESULTS = OUT / "lp2_annual_innovations.csv"
SCALE = 1e5
np.random.seed(0)


def load_producer53():
    """Load producer 53 without relying on an importable numeric module name."""
    path = PIPELINE / "53_two_annual_innovations_lp.py"
    spec = importlib.util.spec_from_file_location("lp53", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"FAIL: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def direct_increment_fit(lp53, inp, raw_y, demeaned_y):
    """Estimate both focal coefficients using producer 53's two routes."""
    variant = "lags_2_4"
    controls = lp53.controls_for(variant, -3)
    raw, dm, w = inp["raw"], inp["dm"], inp["w"]
    X = np.column_stack(
        [raw["z_current"], raw["z_prior"]]
        + [raw[name] for name in controls])
    county_beta, _, _ = lp53.wls_fe(
        raw_y, X, w, inp["fe"], inp["nofe"], inp["county_cluster"])

    C = np.column_stack([dm[name] for name in controls])
    yr, x0r, x1r = lp53.fwl(
        C, w, demeaned_y, dm["z_current"], dm["z_prior"])
    Xr = np.column_stack([x0r, x1r])
    beta = np.linalg.solve(Xr.T @ (w[:, None] * Xr), Xr.T @ (w * yr))
    gap = np.max(
        np.abs(beta - county_beta[:2])
        / np.maximum(np.abs(beta), 1e-300))
    if gap >= lp53.locked.RTOL_COEF:
        raise SystemExit(
            f"FAIL: direct increment routes disagree, relative gap={gap:.2e}")
    return beta, float(gap)


def h1_increment_row():
    """Read the matched h=1 equal-weight, p=4, fixed-sample increment row."""
    grid = pd.read_csv(H1_GRID)
    row = grid[
        (grid["weight"] == "equal")
        & (grid["p"] == 4)
        & (grid["sample"] == "fixed")
        & (grid["param"] == "increment")]
    if len(row) != 1:
        raise SystemExit(
            f"FAIL: expected one matched h=1 increment row, found {len(row)}")
    row = row.iloc[0]
    if (int(row["year_lo"]), int(row["year_hi"])) != (1987, 2012):
        raise SystemExit("FAIL: matched h=1 row is not on the 1987-2012 sample")
    return {j: float(row[f"b{j}"]) for j in (1, 2, 3)}


def main():
    print("== stage 55: h=2 pre-period increment decomposition")
    lp53 = load_producer53()
    inp = lp53.load_inputs()
    if (lp53.YEAR_LO, lp53.YEAR_HI) != (1987, 2012):
        raise SystemExit("FAIL: producer 53 is not on the 1987-2012 sample")

    raw, dm = inp["raw"], inp["dm"]
    cumulative = {
        1: (raw["y-3"], dm["y-3"]),
        2: (raw["y-4"], dm["y-4"]),
        3: (raw["y-5"], dm["y-5"]),
    }
    increments = {
        1: cumulative[1],
        2: (cumulative[2][0] - cumulative[1][0],
            cumulative[2][1] - cumulative[1][1]),
        3: (cumulative[3][0] - cumulative[2][0],
            cumulative[3][1] - cumulative[2][1]),
    }

    direct, route_gaps = {}, {}
    for q, (raw_y, dm_y) in increments.items():
        beta, gap = direct_increment_fit(lp53, inp, raw_y, dm_y)
        direct[q] = beta * SCALE
        route_gaps[q] = gap

    h1 = h1_increment_row()
    mapping_rows = []
    for annual_index in range(1, 5):
        prior = direct.get(annual_index, np.array([np.nan, np.nan]))[1]
        current = direct.get(annual_index - 1, np.array([np.nan, np.nan]))[0]
        h1_value = h1.get(annual_index, np.nan)
        mapping_rows.append({
            "annual_pre_increment": annual_index,
            "outcome_window_relative_to_shock": (
                f"[t-{annual_index + 1},t-{annual_index}]"),
            "h1_equal_p4_increment": h1_value,
            "h2_prior_shock_component": prior,
            "h2_current_shock_component": current,
            "prior_minus_h1": (
                prior - h1_value if np.isfinite(prior)
                and np.isfinite(h1_value) else np.nan),
            "current_minus_h1": (
                current - h1_value if np.isfinite(current)
                and np.isfinite(h1_value) else np.nan),
            "sample": "1987-2012",
            "weight": "equal county",
        })
    mapping = pd.DataFrame(mapping_rows)

    saved = pd.read_csv(H2_RESULTS)
    saved = saved[
        (saved["variant"] == "lags_2_4")
        & (saved["period"] == "pre")].set_index("event_time")
    reconstruction_rows = []
    for event_time, n_increment in ((-3, 1), (-4, 2), (-5, 3)):
        component_current = float(sum(direct[q][0]
                                      for q in range(1, n_increment + 1)))
        component_prior = float(sum(direct[q][1]
                                    for q in range(1, n_increment + 1)))
        reconstructed = 0.5 * (component_current + component_prior)
        reported = float(saved.loc[event_time, "coef_two_year_equivalent"])
        gap = reconstructed - reported
        if abs(gap) >= 1e-10:
            raise SystemExit(
                f"FAIL: event {event_time} reconstruction gap={gap:.2e}")
        reconstruction_rows.append({
            "event_time": event_time,
            "number_of_annual_increments": n_increment,
            "current_component_sum": component_current,
            "prior_component_sum": component_prior,
            "reconstructed_two_year_coefficient": reconstructed,
            "reported_two_year_coefficient": reported,
            "reconstruction_gap": gap,
            "sample": "1987-2012",
            "weight": "equal county",
        })
    reconstruction = pd.DataFrame(reconstruction_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUT / "lp2_preperiod_increment_mapping.csv", index=False)
    reconstruction.to_csv(
        OUT / "lp2_preperiod_reconstruction.csv", index=False)

    print("   annual increment mapping, coefficients x 1e5")
    for row in mapping.itertuples(index=False):
        print(
            f"      j={row.annual_pre_increment}: "
            f"h1={row.h1_equal_p4_increment:+.4f}, "
            f"prior={row.h2_prior_shock_component:+.4f}, "
            f"current={row.h2_current_shock_component:+.4f}")
    print("   exact h=2 reconstruction")
    for row in reconstruction.itertuples(index=False):
        print(
            f"      ell={row.event_time:+d}: "
            f"reported={row.reported_two_year_coefficient:+.4f}, "
            f"reconstructed={row.reconstructed_two_year_coefficient:+.4f}, "
            f"gap={row.reconstruction_gap:.2e}")
    print(f"   maximum route gap={max(route_gaps.values()):.2e}")
    print(f"   wrote {OUT / 'lp2_preperiod_increment_mapping.csv'}")
    print(f"   wrote {OUT / 'lp2_preperiod_reconstruction.csv'}")


if __name__ == "__main__":
    main()
