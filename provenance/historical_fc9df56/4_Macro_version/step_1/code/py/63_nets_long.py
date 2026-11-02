"""NETS Fed2025 cleaning, producer 63: ingest checks and wide-to-long reshapes.

Reads the verified raw cache of Dropbox /_Source/NETS/Fed2025 (fetched by
nets_common.ensure_raw_cache, per-file content_hash checked) and writes to
the Dropbox project's _data/temp/nets_fed2025_staging/ (nets_common.STAGING):

  long/misc_dim.parquet     one row per DunsNumber (misc deduplicated)
  long/emp_long.parquet     duns UINTEGER, year SMALLINT, emp BIGINT, emp_code
  long/sic_long.parquet     duns, year, sic8 VARCHAR
  long/naics_long.parquet   duns, year, naics VARCHAR
  long/fips_long.parquet    duns, year, fips VARCHAR
  long/hqduns_long.parquet  duns, year, hqduns UINTEGER (sentinels dropped)
  long/move_rowid.parquet   move table plus source_row_id (file order)
  _reports/                 coverage, imputation, entry-exit-lag QA, manifest

Rules carried from the approved plan: DunsNumber must match ^[0-9]{9}$
everywhere (fail loud); HQDuns sentinels ('000000000', blank, self-pointer)
mean "no parent" and are dropped from hqduns_long with counts logged; misc's
surplus rows are deduplicated deterministically (non-null FirstYear, widest
FirstYear-LastYear span, min full-row in declared column order); the emp
zero-vs-null decision is data-driven (zeros excluded only if >50% of zero-emp
DUNS-years fall outside [FirstYear, LastYear]) and logged; every long-table
row count is reconciled against the wide table's non-null cell count.
"""

import json
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nets_common as nc

YEARS = [(1990 + i, f"{(1990 + i) % 100:02d}") for i in range(34)]
ROWGROUP = 1_000_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def raw(table):
    return f"read_parquet('{nc.RAW}/{table}.parquet')"


def write(con, select_sql, out_path, order_by):
    con.execute(
        f"COPY ({select_sql} ORDER BY {order_by}) TO '{out_path}' "
        f"(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {ROWGROUP})")
    return con.execute(
        f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]


def assert_duns_clean(con, manifest_qa):
    for t in nc.TABLES:
        bad = con.execute(
            f"SELECT count(*) FROM {raw(t)} "
            f"WHERE NOT regexp_matches(DunsNumber, '^[0-9]{{9}}$')"
        ).fetchone()[0]
        if bad:
            raise RuntimeError(f"{t}: {bad} rows with malformed DunsNumber")
    manifest_qa["duns_key_assert"] = "all 13 tables clean"
    log("duns key assert: all 13 tables clean")


def build_misc_dim(con, manifest_qa):
    n, d = con.execute(
        f"SELECT count(*), count(DISTINCT DunsNumber) FROM {raw('misc')}"
    ).fetchone()
    cols = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM {raw('misc')}").fetchall()]
    full_row = ", ".join(f'"{c}"' for c in cols)
    con.execute(f"""
        CREATE TEMP TABLE misc_ranked AS
        SELECT *, row_number() OVER (
            PARTITION BY DunsNumber
            ORDER BY (FirstYear IS NULL),
                     coalesce(LastYear, 0) - coalesce(FirstYear, 9999) DESC,
                     ({full_row})
        ) AS rk
        FROM {raw('misc')}
    """)
    con.execute("CREATE TEMP TABLE misc_dim AS "
                "SELECT * EXCLUDE (rk) FROM misc_ranked WHERE rk = 1")
    kept = con.execute("SELECT count(*) FROM misc_dim").fetchone()[0]
    if n - kept:
        con.execute(
            f"COPY (SELECT * EXCLUDE (rk) FROM misc_ranked WHERE rk > 1) TO "
            f"'{nc.REPORTS}/misc_dropped_rows.parquet' (FORMAT parquet)")
    con.execute("DROP TABLE misc_ranked")
    manifest_qa["misc"] = {"rows": n, "distinct_duns": d, "kept": kept,
                           "dropped": n - kept}
    log(f"misc: {n:,} rows, {d:,} distinct duns, kept {kept:,}, "
        f"dropped {n - kept:,}")
    rows = write(con, "SELECT * FROM misc_dim",
                 nc.LONG / "misc_dim.parquet", "DunsNumber")
    if rows != kept:
        raise RuntimeError("misc_dim row mismatch")


def _present(col, is_text):
    """Non-missing test: the ingest rtrim'd VARCHAR blanks to '' rather than
    NULL, so text columns need the empty-string filter too."""
    if is_text:
        return f"({col} IS NOT NULL AND {col} <> '')"
    return f"{col} IS NOT NULL"


def unpivot(con, table, col_fmt, value_expr, value_name, is_text, extra=""):
    """UNION ALL unpivot across the 34 years; returns the SELECT sql."""
    parts = []
    for year, yy in YEARS:
        cols = col_fmt.format(yy=yy)
        parts.append(
            f"SELECT CAST(DunsNumber AS UINTEGER) AS duns, "
            f"CAST({year} AS SMALLINT) AS year, "
            f"{value_expr.format(yy=yy)} AS {value_name}{extra.format(yy=yy)} "
            f"FROM {raw(table)} WHERE {_present(cols, is_text)}")
    return " UNION ALL ".join(parts)


def wide_nonnull_count(con, table, col_fmt, is_text):
    terms = " + ".join(
        f"count(*) FILTER (WHERE {_present(col_fmt.format(yy=yy), is_text)})"
        for _, yy in YEARS)
    return con.execute(f"SELECT {terms} FROM {raw(table)}").fetchone()[0]


def build_emp_long(con, manifest_qa):
    con.execute("CREATE TEMP TABLE emp_all AS " + unpivot(
        con, "emp", '"Emp{yy}"', '"Emp{yy}"', "emp", is_text=False,
        extra=', "EmpC{yy}" AS emp_code'))
    n_all = con.execute("SELECT count(*) FROM emp_all").fetchone()[0]
    n_wide = wide_nonnull_count(con, "emp", '"Emp{yy}"', is_text=False)
    if n_all != n_wide:
        raise RuntimeError(f"emp unpivot {n_all} != wide non-null {n_wide}")
    inside, outside = con.execute("""
        SELECT count(*) FILTER (WHERE e.year BETWEEN m.FirstYear AND m.LastYear),
               count(*) FILTER (WHERE e.year NOT BETWEEN m.FirstYear AND m.LastYear)
        FROM emp_all e JOIN misc_dim m ON e.duns = CAST(m.DunsNumber AS UINTEGER)
        WHERE e.emp = 0
    """).fetchone()
    zero_total = inside + outside
    drop_zeros = zero_total > 0 and outside > zero_total * 0.5
    manifest_qa["emp_zero_rule"] = {
        "zero_cells": zero_total, "inside_span": inside,
        "outside_span": outside, "zeros_dropped": drop_zeros}
    log(f"emp zeros: {zero_total:,} cells ({outside:,} outside span) -> "
        f"{'dropped' if drop_zeros else 'kept'}")
    where = "WHERE emp > 0" if drop_zeros else ""
    rows = write(con, f"SELECT * FROM emp_all {where}",
                 nc.LONG / "emp_long.parquet", "duns, year")
    con.execute("DROP TABLE emp_all")
    manifest_qa["emp_long_rows"] = rows
    log(f"emp_long: {rows:,} rows")


def build_simple_longs(con, manifest_qa):
    specs = [
        ("sic", '"SIC{yy}"', 'rtrim("SIC{yy}")', "sic8", "sic_long"),
        ("naics22", '"NAICS{yy}"', 'rtrim("NAICS{yy}")', "naics", "naics_long"),
        ("fips", '"FIPS{yy}"', 'rtrim("FIPS{yy}")', "fips", "fips_long"),
    ]
    for table, col, val, name, out in specs:
        sql = unpivot(con, table, col, val, name, is_text=True)
        rows = write(con, sql, nc.LONG / f"{out}.parquet", "duns, year")
        expect = wide_nonnull_count(con, table, col, is_text=True)
        if rows != expect:
            raise RuntimeError(f"{out}: {rows} != wide non-null {expect}")
        manifest_qa[f"{out}_rows"] = rows
        log(f"{out}: {rows:,} rows")


def build_hqduns_long(con, manifest_qa):
    parts = []
    for year, yy in YEARS:
        c = f'rtrim("HQDuns{yy}")'
        parts.append(
            f"SELECT CAST(DunsNumber AS UINTEGER) AS duns, "
            f"CAST({year} AS SMALLINT) AS year, CAST({c} AS UINTEGER) AS hqduns "
            f"FROM {raw('hq')} "
            f"WHERE \"HQDuns{yy}\" IS NOT NULL AND {c} <> '' "
            f"AND {c} <> '000000000' AND {c} <> DunsNumber")
        # rows failing the sentinel filter are counted below
    sql = " UNION ALL ".join(parts)
    rows = write(con, sql, nc.LONG / "hqduns_long.parquet", "duns, year")
    nonnull = wide_nonnull_count(con, "hq", '"HQDuns{yy}"', is_text=True)
    manifest_qa["hqduns_long"] = {
        "rows": rows, "wide_nonnull": nonnull,
        "sentinels_dropped": nonnull - rows}
    log(f"hqduns_long: {rows:,} rows ({nonnull - rows:,} sentinel/self "
        f"pointers dropped)")


def build_move_rowid(con, manifest_qa):
    con.execute("SET preserve_insertion_order=true")
    rows = write(
        con,
        f"SELECT row_number() OVER () AS source_row_id, * FROM {raw('move')}",
        nc.LONG / "move_rowid.parquet", "source_row_id")
    con.execute("SET preserve_insertion_order=false")
    manifest_qa["move_rowid_rows"] = rows
    log(f"move_rowid: {rows:,} rows")


def qa_coverage(con):
    parts = []
    for t in nc.TABLES:
        parts.append(f"""SELECT '{t}' AS tbl, count(*) AS rows,
            count(DISTINCT DunsNumber) AS n_duns,
            (SELECT count(*) FROM (
                SELECT DISTINCT DunsNumber FROM {raw(t)}
                ANTI JOIN misc_dim USING (DunsNumber))) AS duns_not_in_misc
            FROM {raw(t)}""")
    con.execute(f"COPY ({' UNION ALL '.join(parts)}) TO "
                f"'{nc.REPORTS}/coverage_summary.csv' (FORMAT csv, HEADER)")
    log("coverage summary written")


def qa_imputation(con):
    con.execute(f"""
        COPY (
            WITH e AS (SELECT * FROM read_parquet('{nc.LONG}/emp_long.parquet')),
            sized AS (SELECT *, CASE
                WHEN emp < 5 THEN '01_1-4' WHEN emp < 10 THEN '02_5-9'
                WHEN emp < 20 THEN '03_10-19' WHEN emp < 50 THEN '04_20-49'
                WHEN emp < 100 THEN '05_50-99' WHEN emp < 250 THEN '06_100-249'
                WHEN emp < 1000 THEN '07_250-999' ELSE '08_1000+' END AS size_class
                FROM e)
            SELECT year, size_class, emp_code, count(*) AS cells
            FROM sized GROUP BY 1, 2, 3
        ) TO '{nc.REPORTS}/emp_code_by_year_size.csv' (FORMAT csv, HEADER)
    """)
    con.execute(f"""
        COPY (
            WITH e AS (SELECT duns, year, emp,
                       year - row_number() OVER (
                           PARTITION BY duns, emp ORDER BY year) AS grp
                       FROM read_parquet('{nc.LONG}/emp_long.parquet')),
            runs AS (SELECT duns, emp, count(*) AS run_len
                     FROM e GROUP BY duns, emp, grp)
            SELECT run_len, count(*) AS n_runs
            FROM runs GROUP BY run_len ORDER BY run_len
        ) TO '{nc.REPORTS}/emp_unchanged_run_lengths.csv' (FORMAT csv, HEADER)
    """)
    log("imputation QA written (emp_code by year x size, unchanged-run lengths)")


def qa_entry_exit(con):
    con.execute(f"""
        COPY (
            WITH span AS (SELECT duns, min(year) AS first_emp,
                          max(year) AS last_emp
                          FROM read_parquet('{nc.LONG}/emp_long.parquet')
                          GROUP BY duns)
            SELECT
                count(*) AS n_duns,
                count(*) FILTER (WHERE first_emp > m.FirstYear) AS emp_starts_late,
                count(*) FILTER (WHERE first_emp < m.FirstYear) AS emp_before_firstyear,
                count(*) FILTER (WHERE last_emp > m.LastYear) AS emp_after_lastyear,
                count(*) FILTER (WHERE m.LastYear < 2023
                    AND coalesce(TRY_CAST(m.OutofBis AS INTEGER), 0) = 0)
                    AS dead_but_not_outofbis,
                count(*) FILTER (WHERE m.LastYear = 2023
                    AND TRY_CAST(m.OutofBis AS INTEGER) = 1)
                    AS alive_but_outofbis
            FROM span s JOIN misc_dim m
              ON s.duns = CAST(m.DunsNumber AS UINTEGER)
        ) TO '{nc.REPORTS}/entry_exit_lag.csv' (FORMAT csv, HEADER)
    """)
    log("entry/exit lag QA written")


def main():
    for d in (nc.LONG, nc.FIRM, nc.ESTAB, nc.REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    nc.ensure_raw_cache(log)
    con = duckdb.connect()
    nc.duck_settings(con)
    manifest_qa = {}
    assert_duns_clean(con, manifest_qa)
    build_misc_dim(con, manifest_qa)
    build_emp_long(con, manifest_qa)
    build_simple_longs(con, manifest_qa)
    build_hqduns_long(con, manifest_qa)
    build_move_rowid(con, manifest_qa)
    qa_coverage(con)
    qa_imputation(con)
    qa_entry_exit(con)
    (nc.REPORTS / "manifest_63.json").write_text(
        json.dumps(manifest_qa, indent=2, sort_keys=True))
    log("PRODUCER 63 COMPLETE")


if __name__ == "__main__":
    main()
