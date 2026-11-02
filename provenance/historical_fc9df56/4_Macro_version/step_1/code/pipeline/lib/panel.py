"""The county base panel: income, population, employment, geography, outcomes.

- base_panel: county x year frame with Y (real income), P (population),
  E (EFSY employment), state, division, with the zero-employment screen
  applied (a county reporting zero CBP-covered employment in ANY covered
  year is dropped whole; midpoint growth at E = 0 returns the bound |y| = 2).
- rung_labels: one row per county with czone, msa (2003 metropolitan CBSA,
  else "R" + state), cea, ea labels from the frozen reference delineations.
- outcomes: y_mpgr, y_std, y_log from E_t and E_b.
- wins: within-group 1/99 winsorization (the 55/56 diagnostics convention;
  the locked estimation never winsorizes).

Extracted from producer 51 (base_panel 51:223, wins 51:273, lam_weights
51:267), producer 14 (DIVISION), and producer 57 (geography maps
57:207-227); operation order preserved so outputs are bit-identical. Run
this file directly to tie its output to the golden fixtures.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parent
PIPELINE = LIB.parent
CODE = PIPELINE.parent
REF_DIR = CODE / "reference"
sys.path.insert(0, str(CODE / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(LIB))
import dcads_common as dc  # noqa: E402

# The nine Census divisions over the fifty states plus DC (from producer 14).
DIVISION = {}
for _name, _states in (
        ("New England", "09 23 25 33 44 50"),
        ("Middle Atlantic", "34 36 42"),
        ("East North Central", "17 18 26 39 55"),
        ("West North Central", "19 20 27 29 31 38 46"),
        ("South Atlantic", "10 11 12 13 24 37 45 51 54"),
        ("East South Central", "01 21 28 47"),
        ("West South Central", "05 22 40 48"),
        ("Mountain", "04 08 16 30 32 35 49 56"),
        ("Pacific", "02 06 15 41 53")):
    for _s in _states.split():
        DIVISION[int(_s)] = _name

# The four Census regions as unions of the divisions (from 57:131).
REGION = {"New England": "Northeast", "Middle Atlantic": "Northeast",
          "East North Central": "Midwest", "West North Central": "Midwest",
          "South Atlantic": "South", "East South Central": "South",
          "West South Central": "South",
          "Mountain": "West", "Pacific": "West"}


def base_panel():
    """County x year: Y, P, E, state, division, screened and sorted."""
    xw = pd.read_csv(REF_DIR / "county_ccg_crosswalk.csv",
                     dtype={"county_fips5": str, "ccg_id": str})
    xw = xw[xw["ccg_id"] != "unplaceable"]
    st = (xw.assign(ccg=xw["ccg_id"].astype(int),
                    state=xw["county_fips5"].str[:2].astype(int))
          .drop_duplicates("ccg")[["ccg", "state"]])
    st = st[st["state"].isin(DIVISION)]
    inc = pd.read_parquet(dc.PANEL_OUT_DIR / "income_ccg_year.parquet",
                          columns=["ccg", "year", "income_real"])
    pop = pd.read_parquet(dc.PANEL_OUT_DIR / "population_ccg_year.parquet")
    efs = pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                          columns=["ccg", "year", "emp"])
    d = (inc.merge(pop, on=["ccg", "year"]).merge(efs, on=["ccg", "year"], how="left")
         .merge(st, on="ccg"))
    d = d.rename(columns={"income_real": "Y", "population": "P", "emp": "E"})
    d["division"] = d["state"].map(DIVISION)

    # Zero-employment screen: tested on E.notna() only, because income and
    # population reach back to 1969 while EFSY starts in 1975.
    n_before = d["ccg"].nunique()
    zero_ever = d.loc[d["E"].notna() & (d["E"] <= 0), "ccg"].unique()
    d = d[~d["ccg"].isin(zero_ever)]
    n_after = d["ccg"].nunique()
    if n_after < 0.95 * n_before:
        raise SystemExit(f"zero-employment screen dropped {n_before - n_after} of "
                         f"{n_before} counties; EFSY coverage is not what this "
                         f"screen assumes")
    print(f"  zero-employment screen: dropped {n_before - n_after} counties, "
          f"{n_after:,} kept", flush=True)

    return d.sort_values(["ccg", "year"]).reset_index(drop=True)


def rung_labels(units):
    """One row per county: czone, msa, cea, ea (from 57:207-227).

    msa is "M" + 2003 metropolitan CBSA code where one exists; the "R"+state
    fallback for non-metro counties is applied where state is known, i.e. at
    merge time in the runner (it needs the state column).
    """
    xw = pd.read_csv(REF_DIR / "county_ccg_crosswalk.csv",
                     dtype={"county_fips5": str, "ccg_id": str})
    xw = xw[xw["ccg_id"] != "unplaceable"].assign(
        ccg=lambda t: t["ccg_id"].astype(int))
    czmap = (xw.dropna(subset=["ccz_id"]).drop_duplicates("ccg")
               .set_index("ccg")["ccz_id"])
    cb = pd.read_csv(REF_DIR / "county_cbsa_2003.csv",
                     dtype={"county_fips5": str})
    met = cb[cb["cbsa_type"].eq("Metropolitan")][["county_fips5", "cbsa_code"]]
    msamap = (xw.merge(met, on="county_fips5", how="left")
                .dropna(subset=["cbsa_code"]).drop_duplicates("ccg")
                .set_index("ccg")["cbsa_code"])
    ba = pd.read_csv(REF_DIR / "county_bea_area_2004.csv",
                     dtype={"county_fips5": str, "cea_code": str, "ea_code": str})
    bam = xw.merge(ba, on="county_fips5", how="left")
    eamap = (bam.dropna(subset=["ea_code"]).drop_duplicates("ccg")
                .set_index("ccg")["ea_code"])
    ceamap = (bam.dropna(subset=["cea_code"]).drop_duplicates("ccg")
                 .set_index("ccg")["cea_code"])
    out = pd.DataFrame(index=pd.Index(units, name="ccg"))
    out["czone"] = out.index.map(czmap)
    out["msa_raw"] = out.index.map(msamap)
    out["cea"] = out.index.map(ceamap)
    out["ea"] = out.index.map(eamap)
    return out.reset_index()


def outcomes(E_t, E_b):
    """The three growth outcomes from endpoint and base employment."""
    y_log = np.log(E_t) - np.log(E_b)
    y_std = (E_t - E_b) / E_b
    y_mpgr = 2.0 * (E_t - E_b) / (E_t + E_b)
    return y_log, y_std, y_mpgr


def lam_weights(h, lam):
    """Window weights on years t, t-1, ... t-h+1, most recent first."""
    w = np.array([lam ** (-j) for j in range(h)], dtype=np.float64)
    return w / w.sum()


def wins(v, key):
    """Within-group winsorization at the 1st/99th percentiles (55/56
    diagnostics convention; never applied in the locked estimation)."""
    out = v.astype(np.float64).copy()
    for g in np.unique(key):
        m = key == g
        lo, hi = np.nanpercentile(out[m], [1.0, 99.0])
        out[m] = np.clip(out[m], lo, hi)
    return out


if __name__ == "__main__":
    import locked  # noqa: E402
    GOLD = dc.MACRO_DATA_ROOT / "temp" / "pipeline" / "_golden"
    ref = pd.read_parquet(GOLD / "base_panel.parquet")
    new = base_panel()
    assert list(ref.columns) == list(new.columns), (list(ref.columns),
                                                    list(new.columns))
    assert len(ref) == len(new)
    for c in ref.columns:
        a, b = ref[c].to_numpy(), new[c].to_numpy()
        if ref[c].dtype.kind in "fc":
            assert np.array_equal(np.isnan(a), np.isnan(b)), c
            assert np.allclose(a, b, rtol=locked.RTOL_VEC,
                               atol=locked.ATOL_VEC, equal_nan=True), c
        else:
            assert np.array_equal(a, b), c
    print(f"base_panel tie-out: {len(new):,} rows, all columns match")

    gold_h2 = pd.read_parquet(GOLD / "county_h2.parquet")
    units = np.array(sorted(new["ccg"].unique()))
    lab = rung_labels(units)
    stmap = new.drop_duplicates("ccg").set_index("ccg")["state"]
    lab["state"] = lab["ccg"].map(stmap)
    lab["msa"] = np.where(lab["msa_raw"].notna(),
                          "M" + lab["msa_raw"].astype(str),
                          "R" + lab["state"].astype(str))
    ref_lab = gold_h2.drop_duplicates("ccg")[["ccg", "czone", "msa", "cea", "ea"]]
    m = ref_lab.merge(lab, on="ccg", suffixes=("_ref", "_new"))
    assert len(m) == len(ref_lab)
    for c in ("czone", "msa", "cea", "ea"):
        a = m[f"{c}_ref"].astype(str).to_numpy()
        b = m[f"{c}_new"].astype(str).to_numpy()
        assert np.array_equal(a, b), c
    print(f"rung_labels tie-out: {len(m):,} counties, czone/msa/cea/ea match")
    print("panel.py: PASS")
