"""Estimate the flow-audit specification across horizons and clip values.

For category k and horizon h, the national shifter is

    clip((G_k,t - G_k,t-h) / mean_s(abs(G_k,s)), -a, a).

The regression specification uses net contracts, division-equal weights,
within-year treatment winsorization, division-by-year fixed effects, county
clustering, and the 1977-2018 full era. The h=2 reference cell uses roll-5
exposure; the h=3 and h=4 headline cells use roll-10 exposure.

Writes Results/instrument_normalization/
    flow_audit_full_sample_horizon_clip_ladder.csv
"""

# mpgr-run-class: cache_rebuild
# mpgr-rebuild-reason: treatment_changed
# mpgr-batch-approved: explicit
# mpgr-new-estimates: 7
# mpgr-variation-scope: flow-audit|full-sample-scale|h2-h4|clip1-clip4

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
import store  # noqa: E402


OUT = (dc.MACRO_RESULTS_ROOT / "instrument_normalization"
       / "flow_audit_full_sample_horizon_clip_ladder.csv")
FLOW_AUDIT = (dc.MACRO_RESULTS_ROOT / "ns14_diagnostics"
              / "Table_flow_audit.csv")
HARMONIZED_GRID = dc.LOG_DIR / "county_composition_grid_harmonized_ncat800.csv"

HORIZONS = (2, 3, 4)
CAPS = (1.0, 2.0, 3.0, 4.0)
EXPOSURE = {2: ("roll5", 5), 3: ("roll10", 10), 4: ("roll10", 10)}
SAVED_KEYS = {(2, 1.0), (2, 2.0), (3, 1.0), (4, 1.0), (4, 2.0)}
N_NEW_EXPECTED = len(HORIZONS) * len(CAPS) - len(SAVED_KEYS)


def load_module(name, path):
    """Load one project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_saved_rows():
    """Load the five coefficient rows already persisted by the flow audit."""
    saved = {}
    audit = pd.read_csv(FLOW_AUDIT)
    labels = {1.0: "current (4-digit, top-800, own-mean, clip 1)",
              2.0: "clip 2.0"}
    for h in (2, 4):
        for cap, label in labels.items():
            row = audit.loc[
                audit["horizon"].eq(h) & audit["variant"].eq(label)
            ]
            if len(row) != 1:
                raise SystemExit(
                    f"FAIL: flow audit has {len(row)} rows for h={h}, "
                    f"clip={cap:g}")
            row = row.iloc[0]
            saved[(h, cap)] = {
                "iv_beta_x1e5": float(row["beta"]),
                "iv_se_x1e5": float(row["se"]),
                "iv_t": float(row["t"]),
                "first_stage_F": float(row["F"]),
                "n": int(row["n_obs"]),
                "estimation_source": "loaded Table_flow_audit.csv",
            }

    grid = pd.read_csv(HARMONIZED_GRID)
    row = grid.loc[
        grid["era"].eq("full")
        & grid["horizon"].eq(3)
        & grid["denom"].eq("population")
        & grid["rule"].eq("net")
        & grid["floor"].eq("none")
        & grid["screen"].eq(False)
        & grid["exposure"].eq("roll10")
        & grid["lam"].eq(1.0)
        & grid["fe"].eq("divisionyear")
        & grid["weight"].eq("division-EW")
        & grid["winsor_group"].eq("year")
    ]
    if len(row) != 1:
        raise SystemExit(
            f"FAIL: harmonized grid has {len(row)} rows for h=3, clip=1")
    row = row.iloc[0]
    saved[(3, 1.0)] = {
        "iv_beta_x1e5": float(row["beta_scaled"]),
        "iv_se_x1e5": float(row["se_scaled"]),
        "iv_t": float(row["t"]),
        "first_stage_F": float(row["first_stage_F"]),
        "n": int(row["n_obs"]),
        "estimation_source": (
            "loaded county_composition_grid_harmonized_ncat800.csv"),
    }
    if set(saved) != SAVED_KEYS:
        raise SystemExit("FAIL: saved coefficient key set is incomplete")
    return saved


def run_horizon(h, span, p51, p54, t28, panel, units, pivots, contracts,
                arr, nat, contract_years, saved):
    """Build each clip variant and estimate only missing coefficient rows."""
    cfg = p51.ERAS["full"]
    years = list(range(max(cfg["lo"], cfg["first_base"] + h), cfg["hi"] + 1))
    idx = pd.MultiIndex.from_product([units, years], names=["ccg", "year"])
    frame = pd.DataFrame(index=idx).reset_index()
    state = panel.drop_duplicates("ccg").set_index("ccg")["state"]
    division = panel.drop_duplicates("ccg").set_index("ccg")["division"]
    frame["state"] = frame["ccg"].map(state)
    frame["division"] = frame["ccg"].map(division)
    for col in ("Y", "P", "E"):
        frame[col] = pivots[col].stack().reindex(idx).to_numpy()
        frame[col + "_b"] = (
            pivots[col].reindex(columns=[year - h for year in years])
            .set_axis(years, axis=1).stack().reindex(idx).to_numpy()
        )
    denominator = frame["E"] + frame["E_b"]
    frame["y"] = np.where(
        denominator != 0,
        2.0 * (frame["E"] - frame["E_b"]) / denominator,
        np.nan,
    )

    lag = {
        k: contracts.reindex(columns=[year - k for year in years])
        .set_axis(years, axis=1)
        for k in range(h + 1)
    }
    frame["G_b"] = lag[h].stack().reindex(idx).to_numpy()
    weights_h = p51.lam_weights(h, 1.0)
    frame["G_win"] = sum(weights_h[k] * lag[k] for k in range(h)) \
        .stack().reindex(idx).to_numpy()
    treatment = (frame["G_win"] - frame["G_b"]) / frame["P_b"]

    raw_shifter = p54.category_rates(nat, h, clip=np.inf)
    shifters = {
        cap: np.clip(raw_shifter, -cap, cap) for cap in CAPS
    }
    wide = {}
    contributing = {}
    for cap in CAPS:
        dollars, counts = p54.predicted_dollars(
            arr, shifters[cap], contract_years, years, h, "roll", span,
            p51.ROLL_LAG, p51.DATA_FIRST)
        wide[cap] = np.where(
            np.isnan(dollars) & (counts[None, :] > 0), 0.0, dollars)
        contributing[cap] = counts

    rows = []
    n_new = 0
    for cap in CAPS:
        instrument_values = (
            pd.DataFrame(wide[cap], index=units, columns=years)
            .stack().reindex(idx).to_numpy()
        ) / frame["P_b"].to_numpy(np.float64)
        valid = (
            np.isfinite(treatment).to_numpy()
            & np.isfinite(instrument_values)
            & np.isfinite(frame["y"]).to_numpy()
        )
        sample = frame.loc[valid]
        x = p51.wins(
            treatment.loc[valid].to_numpy(np.float64),
            sample["year"].to_numpy(),
        )
        z = instrument_values[valid].astype(np.float64)
        y = sample["y"].to_numpy(np.float64)
        county = pd.factorize(sample["ccg"])[0].astype(np.int64)
        fe = pd.factorize(
            sample["division"].astype(str) + "_" + sample["year"].astype(str)
        )[0].astype(np.int64)
        nofe = np.zeros(len(sample), np.int64)
        total = sample.groupby(["division", "year"])["E_b"].transform("sum")
        weight = (sample["E_b"] / total).to_numpy(np.float64)
        weight = weight / weight.mean()

        key = (h, cap)
        if key in saved:
            fit = saved[key]
            if len(sample) != fit["n"]:
                raise SystemExit(
                    f"FAIL: h={h}, clip={cap:g} reconstructed N={len(sample):,}, "
                    f"saved N={fit['n']:,}")
        else:
            result = t28.fit2sls(
                y, x, z, weight, fe, nofe, county, singleton=False)
            fit = {
                "iv_beta_x1e5": float(result["beta"] * 1e5),
                "iv_se_x1e5": float(result["se"] * 1e5),
                "iv_t": float(result["beta"] / result["se"]),
                "first_stage_F": float(result["F"]),
                "n": len(sample),
                "estimation_source": "estimated in horizon-by-clip ladder",
            }
            n_new += 1

        target_columns = [contract_years.index(year)
                          for year in sorted(sample["year"].unique())]
        raw_sample = raw_shifter[:, target_columns]
        live = np.isfinite(raw_sample)
        clipped = live & (np.abs(raw_sample) > cap)
        rows.append({
            "horizon": h,
            "exposure": f"roll{span}",
            "exposure_years": span,
            "clip": cap,
            "era": "1977-2018 full",
            "category_level": "four-character",
            "category_universe": "harmonized top-800 categories",
            "national_numerator": "G_k,t minus G_k,t-h",
            "national_denominator": (
                "category mean absolute spending over 1966-2018"),
            "contract_rule": "net",
            "treatment_winsorization": "within year",
            "fixed_effects": "division-by-year",
            "weight": "division-equal",
            "cluster": "county",
            "sample_start": int(sample["year"].min()),
            "sample_end": int(sample["year"].max()),
            "n_counties": int(sample["ccg"].nunique()),
            **fit,
            "live_category_years": int(live.sum()),
            "clipped_category_years": int(clipped.sum()),
            "clipped_share_of_live": float(clipped.sum() / live.sum()),
            "min_contributing_categories": int(
                contributing[cap][contributing[cap] > 0].min()),
            "max_contributing_categories": int(contributing[cap].max()),
        })
    return rows, n_new


def main():
    """Run the full-sample scale clipping ladder."""
    np.random.seed(0)
    start = time.time()
    print("== flow-audit full-sample horizon-by-clip ladder")

    p54 = load_module("p54_flow_audit_ladder",
                      STEP1_PY / "54_composition_instrument_grid.py")
    p51 = p54.p51
    t28 = load_module("t28_flow_audit_ladder",
                      STEP2_PY / "28_table_d1.py")

    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("Y", "P", "E"))
    contracts = store.load_con("net").reindex(index=units)
    tensor_units, categories, contract_years, arr, nat = store.load_tensor("net")
    if not np.array_equal(units, tensor_units):
        raise SystemExit("FAIL: county order differs between persisted stores")
    if len(categories) != 800:
        raise SystemExit(
            f"FAIL: category store has {len(categories)} categories, expected 800")

    saved = load_saved_rows()
    rows = []
    n_new = 0
    for h in HORIZONS:
        label, span = EXPOSURE[h]
        horizon_rows, horizon_new = run_horizon(
            h, span, p51, p54, t28, panel, units, pivots, contracts, arr, nat,
            contract_years, saved)
        rows.extend(horizon_rows)
        n_new += horizon_new
        n = horizon_rows[0]["n"]
        print(f"   h={h}, {label}: N={n:,}")

    if n_new != N_NEW_EXPECTED:
        raise SystemExit(
            f"FAIL: estimated {n_new} new rows, expected {N_NEW_EXPECTED}")
    result = pd.DataFrame(rows).sort_values(["horizon", "clip"])
    if len(result) != len(HORIZONS) * len(CAPS):
        raise SystemExit("FAIL: incomplete horizon-by-clip output")
    if not np.isfinite(result["first_stage_F"]).all():
        raise SystemExit("FAIL: a first-stage F-statistic is missing")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(f"   loaded {len(SAVED_KEYS)} saved rows; estimated {n_new} new rows")
    for _, row in result.iterrows():
        print(f"   h={int(row['horizon'])}, {row['exposure']}, "
              f"clip={row['clip']:g}: F={row['first_stage_F']:.3f}, "
              f"beta={row['iv_beta_x1e5']:+.3f}, "
              f"SE={row['iv_se_x1e5']:.3f}, "
              f"clipped={100 * row['clipped_share_of_live']:.2f}%")
    print(f"   wrote {OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
