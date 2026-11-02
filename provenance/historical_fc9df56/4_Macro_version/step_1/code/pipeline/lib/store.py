"""Readers for the persisted pipeline stores.

One place that knows the layout of `<MACRO_DATA_ROOT>/temp/pipeline/`, so
every consumer (stage 3 onward, the diagnostics, and future producers)
reconstructs the same objects the same way. The tensor rebuild assigns the
stored float32 cells into a zero-filled array, which reproduces the stage-1
accumulation exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parent
PIPELINE = LIB.parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402

PIPE = dc.MACRO_DATA_ROOT / "temp" / "pipeline"
RULE_TAGS = {"net": "net", "positive": "pos"}


def years_full():
    return list(range(locked.FIRST_YEAR, locked.LAST_YEAR + 1))


def target_years(h):
    return [t for t in range(max(1977, 1975 + h), locked.TARGET_LAST + 1)]


def load_county_panel():
    """county_panel_pipeline.parquet as a frame (ccg, year, Y, P, E, labels)."""
    return pd.read_parquet(PIPE / "county_panel_pipeline.parquet")


def load_pivots(pnl, cols=("P", "E")):
    units = np.array(sorted(pnl["ccg"].unique()))
    piv = {c: pnl.pivot(index="ccg", columns="year", values=c).reindex(units)
           for c in cols}
    return units, piv


def load_con(rule):
    """County x year real-dollar matrix for one rule (floor 0), wide."""
    con = pd.read_parquet(PIPE / "contracts_ccg_year.parquet")
    return con.pivot(index="ccg", columns="year",
                     values=f"dollars_{RULE_TAGS[rule]}")


def load_cats():
    return pd.read_csv(PIPE / "cats.csv")["cat"].tolist()


def load_tensor(rule):
    """(units, cats, years, arr, nat): the stage-1 category tensor for one
    rule, rebuilt by assignment from the sparse parquet, plus the national
    panel (cats x years, wide)."""
    tag = RULE_TAGS[rule]
    cats = load_cats()
    years = years_full()
    pnl = pd.read_parquet(PIPE / "county_panel_pipeline.parquet",
                          columns=["ccg"])
    units = np.array(sorted(pnl["ccg"].unique()))
    sp = pd.read_parquet(PIPE / "contracts_category_ccg_year.parquet")
    sp = sp[sp["rule"] == tag]
    arr = np.zeros((len(units), len(cats), len(years)), np.float32)
    ui = pd.Series(range(len(units)), index=units)
    ci = pd.Series(range(len(cats)), index=cats)
    yi = pd.Series(range(len(years)), index=years)
    arr[sp["ccg"].map(ui).to_numpy(),
        sp["cat"].map(ci).to_numpy(),
        sp["year"].map(yi).to_numpy()] = sp["dollars"].to_numpy(np.float32)
    nat = pd.read_parquet(PIPE / "national_category_year.parquet")
    nat = (nat[nat["rule"] == tag]
           .pivot(index="cat", columns="year", values="dollars")
           .reindex(index=cats, columns=years))
    return units, cats, years, arr, nat


def load_gk(h, cats, years):
    """The persisted category shocks for one horizon, as a (cats x years)
    float array aligned to the given orders."""
    g = pd.read_parquet(PIPE / "category_shocks_year.parquet")
    return (g[g["h"] == h].pivot(index="cat", columns="year", values="g")
            .reindex(index=cats, columns=years).to_numpy(np.float64))


def load_instrument(h, units, col):
    """One instrument column (z, z_int, or L) for one horizon as a wide
    (units x target_years) matrix, NaN where undefined."""
    ins = pd.read_parquet(PIPE / "composition_instrument_ccg_year.parquet")
    return (ins[ins["h"] == h].pivot(index="ccg", columns="year", values=col)
            .reindex(index=units, columns=target_years(h))
            .to_numpy(np.float64))


def load_analysis_panel():
    return pd.read_parquet(PIPE / "analysis_panel.parquet")
