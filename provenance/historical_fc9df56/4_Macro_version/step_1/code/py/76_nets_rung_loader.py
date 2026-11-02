#!/usr/bin/env python3
"""WS2 step 3: the NETS employment cells in `load_rung`'s shape.

Producer 73 writes the NETS stock cells LONG -- one row per
(universe, ccg, year, naics4, sic2), and only where an establishment was
actually placed. Every estimator in step_2 instead consumes
`22_winsor_construction_error.py::load_rung`'s contract:

    one dense float array per year, over a support that is IDENTICAL in every
    year and in the SAME row order, so the t-h lag is a plain array shift.

This module is the adapter, and the reshape it performs is not cosmetic.

THE ZEROS ARE THE POINT. A county-industry cell with no NETS establishment in
year t has to be PRESENT AS A ZERO, not absent as a missing row. Its absence is
the object of study: it is an entrant's base period or an exiter's endpoint, and
the midpoint weight (E_imt + E_im,t-h) / (E_ct + E_c,t-h) is exactly what makes
those cells reconstruct the county. Dropping them would delete the extensive
margin, which is A.3.1's error committed silently. So the support materialised
here IS the full ccg x code cross product, which is also what
`27_cell_construction_checks.py::support_completeness` measures and what
`37_table_poisson.py::check_constant_mc` requires (live cells per county
identical across every window, spread exactly zero).

THE UNIVERSE IS A UNIVERSE, NOT A FILTER. `universe` selects between the two
sets producer 73 built from the same retained establishments in one pass:
'actual' (establishments with actual, non-imputed employment on some line) and
'all'. The WS2 protocol gate makes 'actual' the headline because 57.5% of NETS
employment cells are estimates and the NETS/EFSY level ratio is 1.05 on 'actual'
against 1.50 on 'all'. Under gate rule Q2.2 the choice is made ONCE and both
rungs are rebuilt inside it -- which is why it is a parameter of the loader and
never a mask applied downstream. A `share_emp_imputed` cutoff is deliberately
NOT offered: the cell parquets carry `emp_imputed` as a sum and not the
establishment-level share, and the gate fixed the cutoff before the step-2
benchmark numbers were seen, so introducing one now would be choosing it against
visible numbers.

NETS BEGINS IN 1990. `year_floor()` is the sample rule that follows: a window
[t-h, t] is estimable only if both endpoints are inside the NETS panel, so the
baseline frame is restricted to t >= 1990 + h. That is a sample restriction, it
is reported, and it is never described as anything else.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import dcads_common as dc  # noqa: E402

NETS_DIR = dc.MACRO_DATA_ROOT / "temp" / "nets_cells"
COUNTY_PARQUET = NETS_DIR / "nets_ccg_year.parquet"
INDUSTRY_PARQUET = NETS_DIR / "nets_ccg_industry_year.parquet"

UNIVERSES = ("actual", "all")
# The first and last year producer 73's staging covers. NETS Fed2025 starts in
# 1990; the panel runs to 2023, but the composition instrument stops at target
# year 2018 (contracts end FY2018), so nothing downstream reaches past it.
NETS_FIRST_YEAR = 1990
NETS_LAST_YEAR = 2023

# The step_2 rung names, mapped onto a SQL expression over `naics4`. Both are
# cast to BIGINT so the `code` array matches EFSY's integer convention exactly
# (EFSY carries code 1131 at NAICS4 and 11 at NAICS2, as int32).
RUNG_CODE_SQL = {
    "naics4_alloc": "CAST(naics4 AS BIGINT)",
    "naics4": "CAST(naics4 AS BIGINT)",
    "naics2": "CAST(substr(naics4, 1, 2) AS BIGINT)",
}

SOURCE_UNIVERSE = {"nets_actual": "actual", "nets_all": "all"}


def universe_of(source):
    """'nets_actual' -> 'actual'. Raises on anything else, including 'efsy':
    a caller that reaches this module has already decided it wants NETS."""
    if source not in SOURCE_UNIVERSE:
        raise ValueError(f"not a NETS source: {source!r} "
                         f"(expected one of {sorted(SOURCE_UNIVERSE)})")
    return SOURCE_UNIVERSE[source]


def year_floor(h):
    """The first target year whose window [t-h, t] lies inside the NETS panel."""
    return NETS_FIRST_YEAR + h


def _connect():
    import duckdb
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    return con


def _check_inputs():
    for p in (COUNTY_PARQUET, INDUSTRY_PARQUET):
        if not p.exists():
            raise SystemExit(
                f"missing {p}. Run step_1/code/py/73_nets_cell_collapse.py "
                f"first; WS2 step 3 reads only its collapsed cell parquets, "
                f"never the establishment panel.")


def load_rung(rung, years, universe, ccg_keep=None, quiet=False):
    """`load_rung`'s contract, over the NETS cells.

    Returns (emp, ccg_arr, code_arr): `emp[y]` is a float64 array of employment
    aligned row-for-row with `ccg_arr` and `code_arr`, which are the full
    ccg x code cross product sorted by (ccg, code).

    Unlike `w22.load_rung` this returns the code array too. The EFSY path has to
    fetch it separately from `c27.rung_codes`, because there the support is a
    property of a file on disk; here it is constructed, so handing it back is
    the only way the caller can be sure it is the same object.
    """
    if rung not in RUNG_CODE_SQL:
        raise ValueError(f"unknown rung {rung!r}")
    if universe not in UNIVERSES:
        raise ValueError(f"unknown universe {universe!r}")
    _check_inputs()
    years = sorted(int(y) for y in years)
    bad = [y for y in years if not (NETS_FIRST_YEAR <= y <= NETS_LAST_YEAR)]
    if bad:
        raise SystemExit(
            f"years {bad} are outside the NETS panel "
            f"[{NETS_FIRST_YEAR}, {NETS_LAST_YEAR}]. The window rule is "
            f"year_floor(h) = {NETS_FIRST_YEAR} + h; apply it to the frame "
            f"before asking for cells.")

    code_sql = RUNG_CODE_SQL[rung]
    con = _connect()
    con.execute("CREATE TEMP TABLE yrs(year BIGINT)")
    con.executemany("INSERT INTO yrs VALUES (?)", [(y,) for y in years])

    where_ccg = ""
    if ccg_keep is not None:
        keep = np.asarray(sorted(set(int(c) for c in ccg_keep)), np.int64)
        con.execute("CREATE TEMP TABLE keep(ccg BIGINT)")
        con.executemany("INSERT INTO keep VALUES (?)", [(int(c),) for c in keep])
        where_ccg = " AND l.ccg IN (SELECT ccg FROM keep)"

    # One scan, collapsed to the requested rung. sic2 is summed away here; the
    # SIC axis is producer 73's alternative rung and is not what step_2 consumes.
    con.execute(f"""
        CREATE TEMP TABLE cells AS
        SELECT l.ccg AS ccg, {code_sql} AS code, l.year AS year,
               SUM(l.emp) AS emp
        FROM '{INDUSTRY_PARQUET}' AS l
        WHERE l.universe = '{universe}'
          AND l.year IN (SELECT year FROM yrs){where_ccg}
        GROUP BY 1, 2, 3
    """)

    # The support is the cross product of every ccg and every code that appears
    # ANYWHERE in the requested years, not of what appears in a given year. That
    # is what makes the row order year-invariant.
    ccgs = con.execute("SELECT DISTINCT ccg FROM cells ORDER BY 1").df()["ccg"]
    codes = con.execute("SELECT DISTINCT code FROM cells ORDER BY 1").df()["code"]
    ccgs = ccgs.to_numpy(np.int64)
    codes = codes.to_numpy(np.int64)
    if len(ccgs) == 0:
        raise SystemExit(f"no NETS cells for universe={universe!r}, rung={rung!r}")

    n_ccg, n_code = len(ccgs), len(codes)
    ccg_arr = np.repeat(ccgs, n_code)
    code_arr = np.tile(codes, n_ccg)
    row_of_ccg = pd.Series(np.arange(n_ccg, dtype=np.int64), index=ccgs)
    col_of_code = pd.Series(np.arange(n_code, dtype=np.int64), index=codes)

    emp = {}
    for y in years:
        g = con.execute(
            "SELECT ccg, code, emp FROM cells WHERE year = ?", [y]).df()
        v = np.zeros(n_ccg * n_code, np.float64)
        if len(g):
            ri = row_of_ccg.reindex(g["ccg"].to_numpy(np.int64)).to_numpy(np.int64)
            ci = col_of_code.reindex(g["code"].to_numpy(np.int64)).to_numpy(np.int64)
            v[ri * n_code + ci] = g["emp"].to_numpy(np.float64)
        emp[y] = v
    con.close()

    if not quiet:
        print(f"    [nets] {rung} / {universe}: {n_ccg:,} CCGs x {n_code:,} "
              f"codes = {n_ccg * n_code:,} cells/year, "
              f"{years[0]}-{years[-1]} ({len(years)} years); "
              f"support is the cross product with explicit zeros")
    return emp, ccg_arr, code_arr


def county_reference(universe, ccg_keep=None):
    """The NETS analogue of `efsy_ccg_year.parquet`: the A.1.2 check series.

    Producer 73 builds this INDEPENDENTLY of the industry cells -- both come out
    of the same establishment pass, but the county total is its own GROUP BY, so
    comparing the cell sums against it is a real check on the reshape above and
    not an identity. Columns match the EFSY file (`ccg`, `year`, `emp`) so the
    caller's comparison code is unchanged.
    """
    if universe not in UNIVERSES:
        raise ValueError(f"unknown universe {universe!r}")
    _check_inputs()
    d = pd.read_parquet(COUNTY_PARQUET,
                        columns=["universe", "ccg", "year", "emp"])
    d = d[d["universe"] == universe].drop(columns="universe")
    if ccg_keep is not None:
        d = d[d["ccg"].isin(set(int(c) for c in ccg_keep))]
    return d.reset_index(drop=True)


def ccg_support(universe):
    """Every ccg carrying NETS employment, for the support intersection."""
    _check_inputs()
    con = _connect()
    q = con.execute(
        f"SELECT DISTINCT ccg FROM '{COUNTY_PARQUET}' WHERE universe = ?",
        [universe]).df()["ccg"].to_numpy(np.int64)
    con.close()
    return np.sort(q)


def degenerate_ccgs(universe, years):
    """A.3.4: ccgs reporting ZERO NETS employment in any year of the window.

    Excluded whole, in every year, at every rung -- never capped, never
    zero-filled at the county level. Returned rather than applied so the caller
    can count them and put the count on the face of the exhibit.
    """
    _check_inputs()
    years = sorted(int(y) for y in years)
    con = _connect()
    q = con.execute(f"""
        SELECT ccg FROM '{COUNTY_PARQUET}'
        WHERE universe = ? AND year BETWEEN ? AND ?
        GROUP BY ccg
        HAVING MIN(emp) <= 0 OR COUNT(*) < ?
    """, [universe, years[0], years[-1], len(years)]).df()["ccg"]
    con.close()
    return np.sort(q.to_numpy(np.int64))


def fingerprint_paths():
    """The files a cache built on this source must be keyed to."""
    return [COUNTY_PARQUET, INDUSTRY_PARQUET]


if __name__ == "__main__":
    # A smoke test, not a producer: this module writes nothing.
    for uni in UNIVERSES:
        yrs = [2000, 2004]
        emp, ccg_arr, code_arr = load_rung("naics2", yrs, uni)
        tot = {y: emp[y].sum() for y in yrs}
        ref = county_reference(uni)
        for y in yrs:
            r = float(ref[ref["year"] == y]["emp"].sum())
            print(f"  {uni} {y}: cells {tot[y]:,.0f} vs county {r:,.0f} "
                  f"(rel gap {abs(tot[y] - r) / max(r, 1.0):.2e})")
