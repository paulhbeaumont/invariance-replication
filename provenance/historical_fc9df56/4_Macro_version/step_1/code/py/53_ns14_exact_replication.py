#!/usr/bin/env python3
"""Nakamura and Steinsson (2014) on their own inputs, not on substitutes for them.

50 runs their specification on the data this project already held: BEA personal
income standing in for state GDP, BEA county employment summed to the state
standing in for the BLS payroll survey, and a sample that starts where BEA county
income does. That is an analogue. This file is the replication.

WHAT THEY RUN, AND THE PAGE IT IS ON

    eq. (1), p. 759      (Y_it - Y_it-2)/Y_it-2 = a_i + g_t
                             + b (G_it - G_it-2)/Y_it-2 + e_it

    employment, p. 762   the same with (L_it - L_it-2)/L_it-2 on the left,
                         L = employment / population
    per capita           every variable, spending included
    effects              state and year
    cluster              the state, which is how they handle overlapping windows
    Bartik instrument    national spending scaled by "the average level of
                         spending in that state relative to state output in the
                         first five years of our sample" (p. 762)
    sample               1966-2006, so 1968-2006 is estimable at h = 2

    Table 3 col 2, the Bartik employment column, is what this file targets:

        employment   1.81 (0.41) states,  N = 1,989

THEIR INPUTS, AND WHERE EACH NOW COMES FROM

    spending     DD-350 prime military contracts, FY1966-2006. Theirs is ours:
                 "the electronic database of DD-350 military procurement forms"
                 (p. 757), which is what 01 through 08 parse.
    employment   BLS CES, the State and Area series, statewide total nonfarm,
                 all employees, not seasonally adjusted, annual average (M13).
                 51 series, every one covering 1966.
    output       BEA state GDP, all-industry total. SIC basis 1963-1997 spliced
                 to NAICS 1997-2025 at the year BEA publishes both, which is the
                 splice their footnote on p. 758 describes.
    population   BEA SAINC1 line 2, which reaches 1929.

THE SPLICE, AND WHAT IS ASSERTED ABOUT IT

NS paste growth rates rather than levels, so a level gap at 1997 is immaterial
and only a gap that differs across states can bias anything. The measured
NAICS/SIC ratio in 1997 has median 1.0142 and cross-state sd 0.0361. Windows
straddling 1997 carry a flag rather than an assumption.

Output:
    batch_logs/macro_version/ns14_exact_replication.csv
    batch_logs/macro_version/53_ns14_exact_replication.log
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import dcads_common as dc  # noqa: E402

STEP2_CODE = CODE.parents[2] / "step_2" / "code" / "py"
OUT = dc.LOG_DIR / "ns14_exact_replication.csv"
BLS_DIR = dc.SOURCE_DIR.parent / "BLS" / "SAE"

HORIZON = 2
FIRST_YEAR, LAST_YEAR = 1966, 2006
SHARE_YEARS = (1966, 1971)      # their code: year > 1965 & year < 1972
SPLICE_YEAR = 1997              # BEA publishes both bases here

# NS's 51: the states plus DC. BEA's non-state FIPS are excluded by construction.
NON_STATE = {0, 3, 7, 14, 43, 52}     # regions and the US total in BEA's numbering

# Their result, Table 3 col 2 (Bartik) and Table 2 (baseline instrument), so the
# file can say which it is and is not reproducing.
TARGET = dict(beta=1.81, se=0.41, n_obs=1989)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


failures = []


def check(ok, num, label, detail=""):
    print(f"    [{num}] {label:<54s} {detail:<34s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"{num} {label}: {detail}")
    return ok


# --------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------

def bea_state_table(table, zipname, prefix, linecode):
    z = dc.BEA_SOURCE_DIR / zipname / f"{zipname}.zip"
    with zipfile.ZipFile(z) as zf:
        member = [m for m in zf.namelist() if m.startswith(prefix + "__ALL_AREAS")][0]
        d = pd.read_csv(io.BytesIO(zf.open(member).read()), dtype=str)
    d = d[d["LineCode"].astype(str).str.strip().eq(str(linecode))].copy()
    d["fips"] = d["GeoFIPS"].str.replace('"', '', regex=False).str.strip()
    d = d[d["fips"].str.len().eq(5) & d["fips"].str.endswith("000")]
    d["state"] = d["fips"].str[:2].astype(int)
    # 1-56 are the state codes; 0 is the US total and 91-98 are BEA's regions.
    d = d[d["state"].between(1, 56) & ~d["state"].isin(NON_STATE)]
    yrs = [c for c in d.columns if c.strip().isdigit()]
    m = d.melt(id_vars=["state"], value_vars=yrs, var_name="year", value_name=table)
    m["year"] = m["year"].astype(int)
    m[table] = pd.to_numeric(m[table], errors="coerce")
    return m.dropna(subset=[table])


def state_gdp():
    """SIC 1963-1997 spliced to NAICS 1997-2025 on the growth rate, at 1997."""
    sic = bea_state_table("gdp", "SAGDP_SIC", "SAGDP2S", 1)
    nai = bea_state_table("gdp", "SAGDP", "SAGDP2", 1)
    ov = (sic[sic.year.eq(SPLICE_YEAR)].merge(nai[nai.year.eq(SPLICE_YEAR)],
                                              on="state", suffixes=("_s", "_n")))
    check(len(ov) == 51, 1, "both GDP bases publish 1997 for all 51 states",
          f"{len(ov)} states")
    ratio = (ov["gdp_n"] / ov["gdp_s"])
    check(ratio.std() < 0.10, 2, "the 1997 level gap is near-constant across states",
          f"sd {ratio.std():.4f}, median {ratio.median():.4f}")
    # Carry the NAICS level and push the SIC history back through its own growth.
    f = ov.set_index("state").eval("gdp_n / gdp_s")
    sic = sic[sic.year < SPLICE_YEAR].copy()
    sic["gdp"] = sic["gdp"] * sic["state"].map(f)
    out = pd.concat([sic, nai], ignore_index=True).sort_values(["state", "year"])
    return out[out.year.between(FIRST_YEAR, LAST_YEAR)]


def ces_employment():
    ser = pd.read_csv(BLS_DIR / "sm.series", sep="\t", dtype=str)
    ser.columns = [c.strip() for c in ser.columns]
    for c in ser.columns:
        ser[c] = ser[c].str.strip()
    m = ser[ser.area_code.eq("00000") & ser.industry_code.eq("00000000")
            & ser.data_type_code.eq("01") & ser.seasonal.eq("U")]
    m = m[~m.state_code.isin(["72", "78"])]          # PR and VI are not NS's 51
    check(len(m) == 51, 3, "51 statewide total-nonfarm CES series", f"{len(m)} found")
    ids = set(m.series_id)
    rows = []
    for ch in pd.read_csv(BLS_DIR / "sm.data.1.AllData", sep="\t", dtype=str,
                          chunksize=2_000_000):
        ch.columns = [c.strip() for c in ch.columns]
        ch["series_id"] = ch["series_id"].str.strip()
        k = ch[ch.series_id.isin(ids) & ch["period"].str.strip().eq("M13")]
        if len(k):
            rows.append(k[["series_id", "year", "value"]])
    d = pd.concat(rows, ignore_index=True)
    d["year"] = d.year.astype(int)
    d["emp"] = pd.to_numeric(d.value, errors="coerce")
    d = d.merge(m[["series_id", "state_code"]], on="series_id")
    d["state"] = d.state_code.astype(int)
    return d[["state", "year", "emp"]].dropna()


def state_population():
    """SAINC1 line 2. Reaches 1929, so the sample's first year is not a data limit."""
    return bea_state_table("pop", "SAINC", "SAINC1", 2)


def state_contracts():
    """DD-350 county actions summed to the state, deflated as everything else is."""
    p51 = _load("p51", CODE / "51_county_multiplier_grid.py")
    xw = pd.read_csv(CODE.parent / "reference" / "county_ccg_crosswalk.csv",
                     dtype={"county_fips5": str, "ccg_id": str})
    xw = xw[xw["ccg_id"] != "unplaceable"]
    st = (xw.assign(ccg=xw["ccg_id"].astype(int),
                    state=xw["county_fips5"].str[:2].astype(int))
          .drop_duplicates("ccg").set_index("ccg")["state"])
    units = np.array(sorted(st.index.unique()))
    # NS's series excludes construction; see county_actions in 51.
    C = p51.contract_matrices(units, FIRST_YEAR, LAST_YEAR,
                              drop_construction=True)[("net", 0.0)]
    g = C.groupby(st.reindex(C.index)).sum()
    g.index.name = "state"
    m = g.stack().rename("G").reset_index()
    m.columns = ["state", "year", "G"]
    return m[~m.state.isin(NON_STATE)]


def panel():
    gdp = state_gdp()
    emp = ces_employment()
    pop = state_population()
    con = state_contracts()
    d = (gdp.merge(pop, on=["state", "year"])
            .merge(emp, on=["state", "year"])
            .merge(con, on=["state", "year"], how="left"))
    d["G"] = d["G"].fillna(0.0)
    # Everything real, on the deflator the contracts already carry.
    # BEA reports state GDP in MILLIONS of current dollars; contracts are in
    # dollars, so the ratio that forms the regressor needs them on one scale.
    d["gdp"] = d["gdp"] * 1e6 * d["year"].map(dc.deflate)
    check(d.state.nunique() == 51, 4, "51 states in the merged panel",
          f"{d.state.nunique()}")
    yrs = d[d.year.between(FIRST_YEAR, LAST_YEAR)].groupby("year").size()
    check(yrs.min() == 51, 5, "every year 1966-2006 has all 51 states",
          f"min {yrs.min()}, max {yrs.max()}")
    return d.sort_values(["state", "year"]).reset_index(drop=True)


# --------------------------------------------------------------------------------
# Their specification
# --------------------------------------------------------------------------------

def build(d, outcome="employment", share_years=SHARE_YEARS):
    g = d.copy()
    g["g_pc"] = g["G"] / g["pop"]
    g["y_pc"] = g["gdp"] / g["pop"]
    g["L"] = g["emp"] / g["pop"]
    full = g.copy()                       # the national series keeps its history
    for c in ("g_pc", "y_pc", "L"):
        g[c + "_lag"] = g.groupby("state")[c].shift(HORIZON)
    g["lag_year"] = g.groupby("state")["year"].shift(HORIZON)
    g = g[g["lag_year"] == g["year"] - HORIZON]
    g = g[g["year"].between(FIRST_YEAR + HORIZON, LAST_YEAR)]

    g["x"] = (g["g_pc"] - g["g_pc_lag"]) / g["y_pc_lag"]
    g["y"] = ((g["L"] - g["L_lag"]) / g["L_lag"] if outcome == "employment"
              else (g["y_pc"] - g["y_pc_lag"]) / g["y_pc_lag"])

    # Their fracmil2: the state's mean real spending PER PERSON over the window,
    # divided by national mean real spending per person over the same window. The
    # national figure is one number common to every state, so it rescales a single
    # instrument and cannot move a just-identified beta; the share is spending per
    # head. Their p. 762 sentence says "relative to state output" and their code
    # does not, and the code is what produced the published number.
    base = full[full.year.between(*share_years)].copy()
    base["g_pc"] = base["G"] / base["pop"]
    nat_pc = (base.groupby("year")["G"].sum() / base.groupby("year")["pop"].sum()).mean()
    s = (base.groupby("state")["g_pc"].mean() / nat_pc).rename("s")
    nat = full.groupby("year")[["G", "gdp", "pop"]].sum()
    nat["g_pc"] = nat["G"] / nat["pop"]
    nat["y_pc"] = nat["gdp"] / nat["pop"]
    nat["shift"] = ((nat["g_pc"] - nat["g_pc"].shift(HORIZON))
                    / nat["y_pc"].shift(HORIZON))
    g = g.merge(s, on="state", how="left")
    g["z"] = g["s"] * g["year"].map(nat["shift"])
    g["straddles_splice"] = ((g["year"] - HORIZON) < SPLICE_YEAR) & (g["year"] >= SPLICE_YEAR)
    return g[np.isfinite(g[["x", "y", "z"]].to_numpy()).all(axis=1)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share-years", type=int, nargs=2, default=list(SHARE_YEARS))
    args = ap.parse_args()

    import pyfixest as pf
    w22 = _load("w22", STEP2_CODE / "22_winsor_construction_error.py")
    t28 = _load("t28", STEP2_CODE / "28_table_d1.py")
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)

    t0 = time.time()
    print("=" * 92)
    print("  Nakamura and Steinsson (2014), Table 3 col 2, on their own inputs")
    print("=" * 92)
    print(f"  they report  employment {TARGET['beta']} ({TARGET['se']}), "
          f"N = {TARGET['n_obs']:,}\n")
    print("  input checks")
    d = panel()
    print(f"    panel {d.year.min()}-{d.year.max()}, {len(d):,} state-years, "
          f"{time.time()-t0:.1f}s\n")

    rows = []
    print(f"  {'outcome':<12}{'share window':<15}{'years':<12}{'N':>7}"
          f"{'beta':>9}{'se':>8}{'t':>7}{'F':>8}")
    for outcome in ("employment", "output"):
        for sy in ((args.share_years[0], args.share_years[1]), (1966, 1971)):
            g = build(d, outcome, sy)
            f1 = pd.factorize(g["state"])[0].astype(np.int64)
            f2 = pd.factorize(g["year"])[0].astype(np.int64)
            r = t28.fit2sls(g["y"].to_numpy(float), g["x"].to_numpy(float),
                            g["z"].to_numpy(float), np.ones(len(g)), f1, f2,
                            f1.copy(), singleton=False)
            print(f"  {outcome:<12}{f'{sy[0]}-{sy[1]}':<15}"
                  f"{f'{g.year.min()}-{g.year.max()}':<12}{len(g):>7,}"
                  f"{r['beta']:>9.3f}{r['se']:>8.3f}{r['beta']/r['se']:>7.2f}"
                  f"{r['F']:>8.2f}")
            rows.append(dict(outcome=outcome, share_first=sy[0], share_last=sy[1],
                             first_year=int(g.year.min()), last_year=int(g.year.max()),
                             n_obs=len(g), n_states=int(g.state.nunique()),
                             n_straddling=int(g.straddles_splice.sum()),
                             beta=r["beta"], se=r["se"], t=r["beta"] / r["se"],
                             first_stage_F=r["F"]))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\n  wrote {OUT}")

    print("\n  replication checks, against Table 3 col 2")
    e = out[out.outcome.eq("employment")
            & out.share_first.eq(SHARE_YEARS[0])
            & out.share_last.eq(SHARE_YEARS[1])].iloc[0]
    check(e.n_obs == TARGET["n_obs"], 6, "observation count is theirs",
          f"{e.n_obs:,} against {TARGET['n_obs']:,}")
    check(abs(e.beta - TARGET["beta"]) < 0.20, 7, "employment coefficient is theirs",
          f"{e.beta:.3f} against {TARGET['beta']}")
    check(abs(e.se - TARGET["se"]) < 0.15, 8, "standard error is theirs",
          f"{e.se:.3f} against {TARGET['se']}")

    print(f"\n  {len(failures)} failed" if failures else "\n  all checks pass")
    for f in failures:
        print(f"    {f}")
    print(f"  {time.time()-t0:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
