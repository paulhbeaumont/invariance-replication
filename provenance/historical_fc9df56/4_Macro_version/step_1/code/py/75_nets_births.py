"""WS2 step 2 input: NETS establishment births, collapsed to cells.

Producer 73's collapse carries stocks (`emp`, `n_estab`) and nothing about
flows, because a birth is not a sum over an establishment-year -- it is a
statement about the establishment's whole history. The BDS entry benchmark of
WS2 step 2 needs the flow, so it is built here, once, in the same DuckDB /
parquet-in-parquet-out discipline the plan's tooling rule imposes, and every
consumer reads these cell files rather than the 765M-row panel.

This is a sibling of 73, not an edit to it: 73's report is already the record
of the stock collapse and re-running it to add a column would restate numbers
that are cited elsewhere. Its bridge, its universe rule and its UNKNOWN
sentinel are IMPORTED, so the two files are guaranteed to key identically.

DEFINITION. An establishment is born in the first year it appears in the
panel:

    b(e) = min { t : (e, t) in estab_year_panel }

and the birth is placed in the ccg and NAICS4 it carries THAT year. **1990 is
excluded**: the panel starts in 1990, so every establishment alive in 1990 has
b(e) = 1990 whether it opened that year or in 1962. That is left-censoring, not
entry, and counting it would put the entire 1990 stock into the first birth
cell. NETS's own `FirstYear` field is not used: it is D&B's belief about when
the business started, which for the 1990 cohort predates the panel and for
later cohorts is subject to the discovery lag the plan already flags. The
panel-first-appearance definition is the one that lines up with what BDS
measures (an establishment appearing in the frame), and its discovery lag is a
divergence for step 2 to document, not to correct away.

A gap in an establishment's history is NOT a second birth: the minimum is taken
over the whole panel, so a re-appearance after an absent year is a continuer.

Outputs (under the Dropbox project's `_data/temp/nets_cells/`)

  nets_births_ccg_year.parquet           universe x ccg x year
  nets_births_ccg_industry_year.parquet  universe x ccg x year x naics4

Both universes of gate rule 1 are carried: `all`, and `actual` restricted to
births whose establishment reports an actual employment figure in its birth
year. The restriction defines the universe, so both rungs come from the same
retained establishments (gate Q2 rule 2).

Report: `Results/nets_benchmark/ws2_births_report.md`.
"""

import importlib.util
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc
import nets_common as nc


def _load(name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c73 = _load("nets_cell_collapse", "73_nets_cell_collapse.py")
b72 = c73.b72

PANEL = c73.PANEL
OUT_DIR = c73.OUT_DIR
COUNTY_OUT = OUT_DIR / "nets_births_ccg_year.parquet"
INDUSTRY_OUT = OUT_DIR / "nets_births_ccg_industry_year.parquet"
FIRST_TMP = OUT_DIR / "_tmp_first_year.parquet"

RESULTS = Path(__file__).resolve().parents[3] / "Results" / "nets_benchmark"
REPORT_OUT = RESULTS / "ws2_births_report.md"

UNKNOWN = c73.UNKNOWN
MEMORY_LIMIT = c73.MEMORY_LIMIT
PANEL_FIRST_YEAR = 1990        # left-censored; never a birth cohort
BIRTH_FIRST_YEAR = 1991

MEASURE_COLS = ["n_births", "emp_births", "n_births_small",
                "n_births_actual", "emp_births_imputed"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def connect():
    con = duckdb.connect()
    nc.duck_settings(con, memory_limit=MEMORY_LIMIT)
    tmp = OUT_DIR / ".tmp_duck"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con


# ------------------------------------------------------------- first year

def _first_year(con, force=False):
    """min(year) per establishment, over the WHOLE panel.

    Materialised rather than kept as a CTE because it is scanned twice (once to
    size the censored cohort for the report, once to join the birth rows) and
    the group-by over 765M rows is the expensive part of this producer.
    """
    if FIRST_TMP.exists() and not force:
        n = con.execute(
            f"SELECT count(*) FROM read_parquet('{FIRST_TMP}')").fetchone()[0]
        log(f"first-year table: reusing {FIRST_TMP.name} ({n:,} establishments)")
        return
    log("computing first appearance year per establishment (765M-row group-by)")
    con.execute(f"""
        COPY (
            SELECT estab_id, CAST(min(year) AS SMALLINT) AS b_year
            FROM read_parquet('{PANEL}')
            GROUP BY estab_id
        ) TO '{FIRST_TMP}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{FIRST_TMP}')").fetchone()[0]
    log(f"first-year table: {n:,} establishments")


MEASURES = """
        count(*)                                              AS n_births,
        sum(p.emp_primary)                                    AS emp_births,
        count(*) FILTER (p.small_estab)                       AS n_births_small,
        count(*) FILTER (p.any_actual_emp)                    AS n_births_actual,
        sum(p.emp_primary * coalesce(p.share_emp_imputed, 0)) AS emp_births_imputed
"""


def _birth_rows(con):
    """The birth establishment-years, bridged to a ccg. One row per birth."""
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW birth_rows AS
        SELECT b.ccg                                    AS ccg,
               CAST(p.year AS SMALLINT)                 AS year,
               coalesce(nullif(substr(p.naics, 1, 4), ''), '{UNKNOWN}') AS naics4,
               p.any_actual_emp,
               p.emp_primary, p.small_estab, p.share_emp_imputed
        FROM read_parquet('{PANEL}') p
        JOIN read_parquet('{FIRST_TMP}') f
          ON f.estab_id = p.estab_id AND f.b_year = p.year
        JOIN bridge b ON b.fips_raw = coalesce(rtrim(p.fips), '')
        WHERE p.year >= {BIRTH_FIRST_YEAR}
    """)


def _universe_sql(keys):
    """`all` and `actual` from the same retained rows (gate rule 1)."""
    k = ", ".join(keys)
    return f"""
        SELECT 'all' AS universe, {k}, {MEASURES} FROM birth_rows p
        GROUP BY {k}
        UNION ALL
        SELECT 'actual' AS universe, {k}, {MEASURES} FROM birth_rows p
        WHERE p.any_actual_emp GROUP BY {k}
    """


def _collapse(con):
    log("collapsing births to ccg x naics4 x year")
    con.execute(f"""
        COPY ({_universe_sql(["ccg", "year", "naics4"])})
        TO '{INDUSTRY_OUT}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    log("collapsing births to ccg x year (independent pass)")
    con.execute(f"""
        COPY ({_universe_sql(["ccg", "year"])})
        TO '{COUNTY_OUT}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    for path in (INDUSTRY_OUT, COUNTY_OUT):
        n = con.execute(
            f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
        log(f"wrote {path.name}: {n:,} rows, "
            f"{path.stat().st_size / 1e6:.1f} MB")


# ----------------------------------------------------------------- checks

def _checks(con):
    out = {}

    # Every establishment is born exactly once, and the cohorts must exhaust
    # the establishment count: censored (1990) + placed + unplaced = all.
    out["cohorts"] = con.execute(f"""
        SELECT count(*)                                   AS n_estab,
               count(*) FILTER (b_year = {PANEL_FIRST_YEAR}) AS n_censored,
               count(*) FILTER (b_year >  {PANEL_FIRST_YEAR}) AS n_births_raw
        FROM read_parquet('{FIRST_TMP}')
    """).df().iloc[0].to_dict()

    out["placed"] = con.execute(
        "SELECT count(*) AS n FROM birth_rows").df().iloc[0].to_dict()

    diffs = ", ".join(
        f"max(abs(coalesce(a.{c}, 0) - coalesce(b.{c}, 0))) AS d_{c}"
        for c in MEASURE_COLS)
    sums = ", ".join(f"sum({c}) AS {c}" for c in MEASURE_COLS)
    out["additivity"] = con.execute(f"""
        WITH rolled AS (
            SELECT universe, ccg, year, {sums}
            FROM read_parquet('{INDUSTRY_OUT}') GROUP BY 1, 2, 3
        )
        SELECT {diffs}, count(*) AS n_pairs
        FROM read_parquet('{COUNTY_OUT}') a
        FULL OUTER JOIN rolled b USING (universe, ccg, year)
    """).df().iloc[0].to_dict()

    out["total"] = con.execute(f"""
        SELECT universe, sum(n_births) AS n_births, sum(emp_births) AS emp
        FROM read_parquet('{COUNTY_OUT}') GROUP BY 1 ORDER BY 1
    """).df()

    out["by_year"] = con.execute(f"""
        SELECT year, universe, count(DISTINCT ccg) AS n_ccg,
               sum(n_births) AS n_births, sum(emp_births) AS emp_births,
               sum(emp_births_imputed) AS emp_imputed
        FROM read_parquet('{COUNTY_OUT}') GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    return out


def write_report(chk):
    def flag(ok):
        return "PASS" if ok else "FAIL"

    coh = chk["cohorts"]
    n_estab = int(coh["n_estab"])
    n_cens = int(coh["n_censored"])
    n_raw = int(coh["n_births_raw"])
    placed = int(chk["placed"]["n"])

    L = ["# WS2 step 2 input: NETS establishment births\n\n"]
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
             "`step_1/code/py/75_nets_births.py`. A sibling of producer 73, "
             "not a re-run of it: 73 collapses stocks, this collapses the "
             "entry flow the BDS benchmark needs. The bridge, the universe "
             "rule and the `UNKNOWN` industry sentinel are imported from 73, "
             "so the four cell files key identically.\n\n")

    L.append("## Definition, and what 1990 is\n\n")
    L.append("An establishment is born in the first year it appears in the "
             "panel, `b(e) = min { t : (e,t) in estab_year_panel }`, and the "
             "birth is placed in the ccg and NAICS4 it carries that year. "
             "A gap in the history is not a second birth: the minimum is over "
             "the whole panel, so a re-appearance is a continuer.\n\n")
    L.append(f"**The 1990 cohort is left-censoring, not entry, and is "
             f"excluded.** {n_cens:,} of {n_estab:,} establishments "
             f"({n_cens / n_estab:.1%}) first appear in {PANEL_FIRST_YEAR}, "
             f"the panel's first year; almost all of them opened earlier. "
             f"Births are therefore counted from {BIRTH_FIRST_YEAR} onward, "
             f"and no exhibit may report a {PANEL_FIRST_YEAR} entry rate. "
             f"NETS's own `FirstYear` is deliberately not used: it is D&B's "
             f"belief about when the business started, which is a different "
             f"object from appearing in the frame, and is exactly what BDS "
             f"does not measure.\n\n")

    L.append("## Coverage\n\n")
    L.append(f"| stage | establishments |\n|---|---:|\n")
    L.append(f"| in the panel | {n_estab:,} |\n")
    L.append(f"| first seen in {PANEL_FIRST_YEAR} (censored, excluded) | "
             f"{n_cens:,} |\n")
    L.append(f"| born {BIRTH_FIRST_YEAR}-2023 | {n_raw:,} |\n")
    L.append(f"| of those, placed into a ccg | {placed:,} "
             f"({placed / n_raw:.4%}) |\n")
    L.append(f"\nThe {n_raw - placed:,} unplaced births are the same blank and "
             "non-state FIPS producer 73 reports as a stock coverage loss; the "
             "birth files carry the same geography as the stock files by "
             "construction, so the two are comparable cell by cell.\n")

    L.append("\n## Totals\n\n")
    L.append("| universe | births | employment at birth |\n|---|---:|---:|\n")
    for _, r in chk["total"].iterrows():
        L.append(f"| {r.universe} | {int(r.n_births):,} | {r.emp:,.0f} |\n")

    L.append("\n## Checks\n\n")
    add = chk["additivity"]
    worst, worst_col, ok = 0.0, None, True
    for key, value in add.items():
        if not key.startswith("d_"):
            continue
        gap = abs(float(value or 0.0))
        tol = 1e-6 if key == "d_emp_births_imputed" else 0.0
        if gap > tol:
            ok = False
        if gap > worst:
            worst, worst_col = gap, key[2:]
    L.append(f"- Exact additivity, `max |X_ct - sum_m X_cmt|` over every "
             f"universe x ccg x year and all {len(MEASURE_COLS)} measures: "
             f"{flag(ok)} (worst = {worst:g} on `{worst_col}`, over "
             f"{int(add['n_pairs']):,} pairs). The county file is aggregated "
             f"independently from the birth rows, not re-collapsed from the "
             f"industry file, so this compares two code paths.\n")
    L.append(f"- Cohort accounting: {flag(n_cens + n_raw == n_estab)} "
             f"({n_cens:,} censored + {n_raw:,} born = {n_estab:,}).\n")

    L.append("\n## Births by year\n\n")
    L.append("| year | ccgs | births (all) | employment at birth (all) | "
             "births (actual) | imputed share of birth employment (all) |\n")
    L.append("|---:|---:|---:|---:|---:|---:|\n")
    by = chk["by_year"].pivot(index="year", columns="universe")
    for year, r in by.iterrows():
        n_all = r[("n_births", "all")]
        e_all = r[("emp_births", "all")]
        imp = r[("emp_imputed", "all")]
        n_act = r[("n_births", "actual")]
        L.append(f"| {int(year)} | {int(r[('n_ccg', 'all')]):,} | "
                 f"{int(n_all):,} | {e_all:,.0f} | "
                 f"{0 if pd.isna(n_act) else int(n_act):,} | "
                 f"{(imp / e_all if e_all else 0):.1%} |\n")
    L.append("\nThe imputed share of birth-year employment is the reason the "
             "plan makes birth COUNTS the primary entry outcome and employment "
             "at entry the robustness: a newly discovered establishment is "
             "exactly the one D&B has no reported employment figure for.\n")

    REPORT_OUT.write_text("".join(L))
    return ok


def main():
    force = "--rebuild-first-year" in sys.argv
    report_only = "--report-only" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = connect()
    try:
        c73._bridge(con)                       # registers `bridge`; fails loud
        _first_year(con, force=force)
        _birth_rows(con)
        if report_only:
            missing = [p for p in (COUNTY_OUT, INDUSTRY_OUT) if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    "--report-only needs the birth files: "
                    + ", ".join(str(p) for p in missing))
            log("report-only: reusing the existing birth files")
        else:
            _collapse(con)
        log("running checks")
        chk = _checks(con)
        ok = write_report(chk)
        log(f"wrote {REPORT_OUT}")
        if not ok:
            raise RuntimeError(
                "births additivity failed: the county file and the industry "
                f"file disagree. See {REPORT_OUT}.")
        log("additivity PASS; births are ready for WS2 step 2")
    finally:
        con.close()


if __name__ == "__main__":
    main()
