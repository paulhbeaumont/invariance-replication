"""WS2 step 2b input: NETS establishment births, split by primary-line EmpC.

Producer 75 collapses births carrying `n_births_actual` (the count whose
establishment reports an actual figure in its birth year) and
`emp_births_imputed`. That is a two-way split, actual against everything else.
The step-2b question is finer: NETS's estimated employment is not one thing,
it is three, and they are produced by different machinery --

    EmpC 0  actual figure reported to D&B        (42.5% of cells)
    EmpC 1  bottom of a reported range           ( 0.8%)
    EmpC 2  D&B's own estimate                   (22.7%)
    EmpC 3  Walls & Associates' estimate         (34.0%)

(shares from `NETS_FED2025.md`). A birth burst that lives in EmpC 2 is D&B
discovering an establishment and back-filling it; a burst that lives in EmpC 3
is Walls's imputation model; a burst in EmpC 0 is the only one that could be an
establishment actually opening and reporting. Producer 75's two-way split
cannot tell those apart, so this producer carries the four-way code.

This is a sibling of 75, not an edit to it, for the same reason 75 is a sibling
of 73: 75's report is the record of the birth collapse and its numbers are
cited in `ws2_step2_benchmark_report.md`. Everything that defines a birth is
IMPORTED from 75 -- the first-appearance definition, the 1990 left-censoring
exclusion, the bridge, the UNKNOWN sentinel -- so the counts here sum to 75's
by construction, and `_checks` asserts exactly that.

THE EmpC CODE IS THE PRIMARY LINE'S, not the establishment's. It comes from
producer 73's `_tmp_primary_empc.parquet`, the code of the DUNS line whose
employment is `emp_primary`, so it describes precisely the employment figure
being aggregated. An establishment-year with no matched line carries emp_code
-1 (`missing`), never folded into a real code.

Output (under the Dropbox project's `_data/temp/nets_cells/`)

  nets_births_empc_ccg_year.parquet   ccg x year x naics2 x emp_code

NAICS2, not NAICS4: the consumer's industry rung is the BDS sector, which is
NAICS2 collapsed onto 19 codes, and a NAICS4 x EmpC file would be ~5x larger
for a resolution nothing downstream can use. The county rung is a re-collapse
of this file, which is exact because every measure is a count or a sum.

Run: `.venv/bin/python step_1/code/py/77_nets_births_empc.py`
"""

import importlib.util
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc          # noqa: E402
import nets_common as nc           # noqa: E402


def _load(name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b75 = _load("nets_births", "75_nets_births.py")
c73 = b75.c73

PANEL = b75.PANEL
OUT_DIR = b75.OUT_DIR
FIRST_TMP = b75.FIRST_TMP
EMPC_TMP = c73.EMPC_TMP
COUNTY_BIRTHS = b75.COUNTY_OUT

OUT = OUT_DIR / "nets_births_empc_ccg_year.parquet"

UNKNOWN = b75.UNKNOWN
MEMORY_LIMIT = b75.MEMORY_LIMIT
BIRTH_FIRST_YEAR = b75.BIRTH_FIRST_YEAR

EMPC_LABEL = {0: "0 actual", 1: "1 range bottom", 2: "2 D&B estimate",
              3: "3 Walls estimate", -1: "missing"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def connect():
    con = duckdb.connect()
    nc.duck_settings(con, memory_limit=MEMORY_LIMIT)
    tmp = OUT_DIR / ".tmp_duck"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con


def build(con):
    """One pass over the birth rows, keyed by ccg x year x naics2 x emp_code.

    `any_actual_emp` rides along as a separate measure so the two universes of
    gate rule 1 remain reconstructible from this file without a second pass:
    `all` is the sum over every emp_code, `actual` is the sum of
    `n_births_actual`. Note that these are NOT the same cut -- `any_actual_emp`
    is true if ANY line of the establishment reports an actual figure, while
    emp_code = 0 is a statement about the primary line only -- which is exactly
    why both are carried and neither is inferred from the other.
    """
    log("registering the FIPS bridge (producer 72's classifier)")
    c73._bridge(con)
    b75._first_year(con)

    log("collapsing births by ccg x year x naics2 x EmpC")
    con.execute(f"""
        COPY (
            SELECT b.ccg                                      AS ccg,
                   CAST(p.year AS SMALLINT)                   AS year,
                   coalesce(nullif(substr(p.naics, 1, 2), ''),
                            '{UNKNOWN}')                      AS naics2,
                   CAST(coalesce(c.emp_code, -1) AS TINYINT)  AS emp_code,
                   count(*)                                   AS n_births,
                   sum(p.emp_primary)                         AS emp_births,
                   count(*) FILTER (p.any_actual_emp)         AS n_births_actual,
                   count(*) FILTER (p.small_estab)            AS n_births_small
            FROM read_parquet('{PANEL}') p
            JOIN read_parquet('{FIRST_TMP}') f
              ON f.estab_id = p.estab_id AND f.b_year = p.year
            JOIN bridge b ON b.fips_raw = coalesce(rtrim(p.fips), '')
            LEFT JOIN read_parquet('{EMPC_TMP}') c
              ON c.estab_id = p.estab_id AND c.year = p.year
            WHERE p.year >= {BIRTH_FIRST_YEAR}
            GROUP BY 1, 2, 3, 4
        ) TO '{OUT}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{OUT}')").fetchone()[0]
    log(f"wrote {OUT.name}: {n:,} rows, {OUT.stat().st_size / 1e6:.1f} MB")


def checks(con):
    """The counts must reproduce producer 75's county file exactly."""
    d = con.execute(f"""
        WITH mine AS (
            SELECT ccg, year, sum(n_births) AS n_births,
                   sum(emp_births) AS emp_births,
                   sum(n_births_actual) AS n_births_actual
            FROM read_parquet('{OUT}') GROUP BY 1, 2
        ), theirs AS (
            SELECT ccg, year, n_births, emp_births, n_births_actual
            FROM read_parquet('{COUNTY_BIRTHS}') WHERE universe = 'all'
        )
        SELECT count(*) AS n_pairs,
               max(abs(coalesce(a.n_births, 0) - coalesce(b.n_births, 0)))
                   AS d_n,
               max(abs(coalesce(a.emp_births, 0) - coalesce(b.emp_births, 0)))
                   AS d_emp,
               max(abs(coalesce(a.n_births_actual, 0)
                       - coalesce(b.n_births_actual, 0))) AS d_actual
        FROM mine a FULL OUTER JOIN theirs b USING (ccg, year)
    """).df().iloc[0]
    if not (d["d_n"] == 0 and d["d_actual"] == 0 and d["d_emp"] < 1e-6):
        raise RuntimeError(
            "births by EmpC do not re-collapse to producer 75's county file: "
            f"max |dn| = {d['d_n']}, max |demp| = {d['d_emp']}, "
            f"max |dactual| = {d['d_actual']} over {int(d['n_pairs']):,} pairs")
    log(f"additivity against producer 75: exact over {int(d['n_pairs']):,} "
        "ccg-years")

    comp = con.execute(f"""
        SELECT emp_code, sum(n_births) AS n_births, sum(emp_births) AS emp
        FROM read_parquet('{OUT}') GROUP BY 1 ORDER BY 1
    """).df()
    comp["label"] = comp.emp_code.map(EMPC_LABEL)
    comp["share"] = comp.n_births / comp.n_births.sum()
    log("EmpC composition of births:\n" + comp.to_string(index=False))
    return comp


def main():
    for path in (PANEL, EMPC_TMP):
        if not Path(path).exists():
            raise FileNotFoundError(f"missing input: {path} (run producer 73)")
    con = connect()
    try:
        build(con)
        checks(con)
    finally:
        con.close()
    log("done")


if __name__ == "__main__":
    main()
