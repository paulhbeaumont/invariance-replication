"""The industry axis: the cell tensor, the sentinels, and the FSC restatement.

One place that knows how producer 23's county x industry cells become a tensor
and how an industry-axis shock is carried back to the FSC categories it came
from, so producers 25 and 28 cannot drift on either.

- load_axis_tensor: the dense zero-filled county x cell x year array. Lifted
  from 25_industry_instrument.py:90 when 28 needed the same object; 25 now
  imports it. Rebuild is by ASSIGNMENT into a zero-filled array, exactly as
  store.load_tensor rebuilds the locked FSC tensor.
- SENTINELS: the three cells producer 23 uses to conserve dollars that reach no
  industry. They are not industries, have no interpretable national shock, and
  are excluded from every shift-share sum.
- restate_to_fsc: the industry shock carried back onto the FSC axis.

WHY THE RESTATEMENT EXISTS. WS1 asks for inference clustered at the FSC level,
and the industry cells do not partition the FSC categories: under the
probabilistic allocation one FSC spreads across many cells, so no cell has "the"
FSC parent and no partition of the 309 cells is a partition of the 800
categories. Rather than pick a parent, the shock is restated:

    industry     z_c = sum_j  cbar_cj . g_j
    concordance        cbar_cj = sum_k cbar_ck . share_kj
    substituting       z_c = sum_k cbar_ck . g~_k ,   g~_k = sum_j share_kj g_j

g~_k is the SAME national industry shocks indexed on the 800 FSC categories the
concordance is keyed on. The BHJ shock-level aggregation then runs on the locked
FSC exposure tensor with g~_k in place of g_k: 800 well-defined clusters, no
ties, no arbitrary parent, and the county/shock-level identity exact wherever
the restatement is.

WHERE IT IS EXACT. share_kj is a fixed concordance share, so the identity holds
on concordance-sourced dollars (FY1966-2006). On native-sourced dollars
(FY2007-18) the FSC->industry split is per RECORD, not a fixed share, so the
restatement is an approximation there. It is measured, never assumed:
restatement_report() reconstructs z from g~ and reports the gap per target year,
so the FY2007 seam's contribution is a number in a file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parent
PIPELINE = LIB.parent
CODE = PIPELINE.parent
sys.path.insert(0, str(CODE / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(LIB))
import locked  # noqa: E402
import store  # noqa: E402

REF_DIR = CODE / "reference"
CELLS = PIPELINE.parents[2] / "data" / "temp" / "industry_axis"

# Producer 23's sentinels. They carry the dollars that reach no industry so that
# county x year totals conserve exactly against stage 1; they are not industries
# and have no interpretable national shock, so they are held for the
# conservation assertion and then excluded from the shift-share sum.
SENTINELS = ("UNMAPPED", "NA__", "INVALID")

# The fiscal year at which producer 23's era rule switches from the concordance
# to the record's own naics_code. Above it the restatement is approximate.
CONC_LAST_YEAR = 2006


def load_axis_tensor(value_col, rule="pos"):
    """(units, cells, years, arr): the dense zero-filled county x cell x year tensor.

    Mirrors store.load_tensor: the parquet is stored long-and-nonzero, and the
    tensor is rebuilt by ASSIGNMENT into a zero-filled array, which is exactly how
    the locked FSC tensor is reconstructed (lib/store.py:69-75).
    """
    years = store.years_full()
    pnl = pd.read_parquet(store.PIPE / "county_panel_pipeline.parquet",
                          columns=["ccg"])
    units = np.array(sorted(pnl["ccg"].unique()))
    assert len(units) == locked.N_COUNTIES, len(units)

    sp = pd.read_parquet(CELLS / "industry_cells_naics4.parquet",
                         columns=["source", "rule", "ccg", "year", "axis4", value_col])
    sp = sp[sp["rule"] == rule]
    cells = sorted(sp["axis4"].unique())
    arr = np.zeros((len(units), len(cells), len(years)), np.float32)
    ui = pd.Series(range(len(units)), index=units)
    ci = pd.Series(range(len(cells)), index=cells)
    yi = pd.Series(range(len(years)), index=years)
    i = sp["ccg"].map(ui).to_numpy()
    j = sp["axis4"].map(ci).to_numpy()
    t = sp["year"].map(yi).to_numpy()
    ok = np.isfinite(i.astype(float)) & np.isfinite(t.astype(float))
    assert ok.all(), "an industry cell row is off the locked county x year grid"
    # `source` splits the era rule into two passes over disjoint years, so a
    # (ccg, axis4, year) key can appear once per pass; add rather than assign.
    np.add.at(arr, (i.astype(np.int64), j.astype(np.int64), t.astype(np.int64)),
              sp[value_col].to_numpy(np.float32))
    return units, cells, years, arr


def drop_sentinels(cells, *arrays):
    """(real_cells, *arrays restricted to them, sentinel dollar share of arrays[0])."""
    keep = [k for k, c in enumerate(cells) if c not in SENTINELS]
    sent = [k for k, c in enumerate(cells) if c in SENTINELS]
    tot = float(arrays[0].sum())
    share = float(arrays[0][:, sent, :].sum()) / tot if sent and tot else 0.0
    out = [np.ascontiguousarray(a[:, keep, :]) for a in arrays]
    return [cells[k] for k in keep], out, share


# --------------------------------------------------- the FSC restatement --

def load_share_matrix(cats, cells, allocation):
    """share_kj: (len(cats), len(cells)), rows summing to 1 where the FSC is mapped.

    `allocation` is "prob" or "modal", matching producer 23's two columns:
    prob spreads a category over its industries by the concordance shares, modal
    puts it whole on the category's modal industry (`top1_axis4`). The two share
    matrices are what makes the two instrument columns differ, so the
    restatement must use the SAME one the instrument was built from -- passing
    the prob matrix against a modal instrument would silently restate the wrong
    object.

    An FSC absent from the concordance (18 of the locked 800 -- namespace-era
    codes outside its FY2001-06 estimation window) gets an all-zero row. Its
    dollars are exactly the UNMAPPED sentinel on the concordance years, which is
    excluded from the shift-share sum, so a zero row is the right restatement
    and not a silent drop.
    """
    assert allocation in ("prob", "modal"), allocation
    t = pd.read_csv(REF_DIR / "fsc_efsy_axis_concordance.csv", comment="#",
                    dtype={"fsc": str, "axis4": str, "top1_axis4": str})
    if allocation == "modal":
        t = (t.drop_duplicates("fsc")
              .assign(axis4=lambda f: f["top1_axis4"], share=1.0))
    ki = pd.Series(range(len(cats)), index=list(cats))
    ji = pd.Series(range(len(cells)), index=list(cells))
    k = t["fsc"].map(ki)
    j = t["axis4"].map(ji)
    m = k.notna().to_numpy() & j.notna().to_numpy()
    S = np.zeros((len(cats), len(cells)), np.float64)
    np.add.at(S, (k[m].to_numpy(np.int64), j[m].to_numpy(np.int64)),
              t.loc[m, "share"].to_numpy(np.float64))
    return S


def restate_to_fsc(g_ind, S):
    """g~_k = sum_j share_kj g_j, NaN-safe over the live industry shocks.

    g_ind is (n_cells, n_years) with NaN where a cell's h-year lag is outside
    the panel. A NaN shock is UNDEFINED, not zero growth, so it is held out of
    the sum by the live mask -- the same asymmetry lib/instrument.py:34-42
    applies. A category whose whole industry set is dead in a year restates to
    NaN, which is what holds it out of the shift-share sum downstream.
    """
    live = np.isfinite(g_ind)
    num = S @ np.where(live, g_ind, 0.0)
    wgt = S @ live.astype(np.float64)
    out = np.where(wgt > 0, num, np.nan)
    return out
