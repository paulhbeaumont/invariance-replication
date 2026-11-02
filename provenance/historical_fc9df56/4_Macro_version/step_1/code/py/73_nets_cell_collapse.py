"""WS2 step 1, second half: collapse the NETS establishment panel to cells.

The plan's binding tooling rule for WS2: the 765,300,955-row establishment-year
panel is collapsed once, in DuckDB, to cell-level parquets, and every later step
(benchmarking, estimation, entry margins) reads only these. The panel never
enters pandas and is never read again by a consumer.

This is a collapse, not an estimand: no growth rate, no shock, no scaling
denominator is constructed here. Every stored measure is a sum or a count, so
each one re-collapses additively from the fine cell to the county.

Governed by `Results/nets_benchmark/ws2_step0_protocol_gate.md`, whose four
binding rules are carried here as code:
  1. one universe, both rungs, rebuilt from the same retained establishments
     -> the `universe` column ('all', 'actual'), each rung grouped from the
        same source rows;
  2. the actual-employment cutoff fixed before any benchmark number is seen
     -> `any_actual_emp`, fixed here, ahead of step 2;
  3. unmatched county FIPS fail loud
     -> `_bridge()` raises on any `unmatched*` / `unplaceable` status;
  4. no cell-level tidying: zero and tiny cells are the phenomenon
     -> nothing is dropped for being small, imputed, or short-lived.

Outputs (under the Dropbox project's `_data/temp/nets_cells/`)

  nets_ccg_year.parquet           universe x ccg x year
  nets_ccg_industry_year.parquet  universe x ccg x year x naics4 x sic2

`(naics4, sic2)` is a joint partition of the county-year -- each establishment
carries exactly one `naics` and one `sic8` -- so the fine file re-collapses
exactly to the NAICS4 rung, the NAICS2 rung, the SIC2 rung, and to the county
file, and one collapse serves the whole cell-fineness ladder (gate Q1, A.1.7).

Report: `Results/nets_benchmark/ws2_step1_cells_report.md`.
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


def _load_bridge_module():
    """Import producer 72. Numbered modules need importlib (leading digit)."""
    path = Path(__file__).resolve().parent / "72_nets_ccg_bridge.py"
    spec = importlib.util.spec_from_file_location("nets_ccg_bridge", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b72 = _load_bridge_module()

PANEL = b72.PANEL
STAGING = PANEL.parents[1]
EMP_LONG = STAGING / "long" / "emp_long.parquet"
XWALK = STAGING / "estab" / "estab_xwalk.parquet"

OUT_DIR = dc.MACRO_DATA_ROOT / "temp" / "nets_cells"
COUNTY_OUT = OUT_DIR / "nets_ccg_year.parquet"
INDUSTRY_OUT = OUT_DIR / "nets_ccg_industry_year.parquet"
EMPC_TMP = OUT_DIR / "_tmp_primary_empc.parquet"

RESULTS = Path(__file__).resolve().parents[3] / "Results" / "nets_benchmark"
REPORT_OUT = RESULTS / "ws2_step1_cells_report.md"

PANEL_ROWS = b72.PANEL_ROWS
MEMORY_LIMIT = "18GB"
UNKNOWN = "UNKNOWN"          # industry sentinel: kept as a cell, never dropped
EMPC_ACTUAL_SHARE = 0.425    # NETS_FED2025.md, EmpC = 0 share of cells

# Additivity must hold exactly for every measure. `emp_imputed` is the one
# fractional sum (employment times an imputation share), so its two
# aggregation orders differ in the last bits of a double; everything else is
# an integer-valued count or headcount and must agree to the unit.
ADDITIVITY_TOL = {"emp_imputed": 1e-6}


def _tol(col):
    return ADDITIVITY_TOL.get(col, 0.0)


def _additivity_verdict(add):
    """(worst absolute gap, worst column, pass) over every measure."""
    worst, worst_col, ok = 0.0, None, True
    for key, value in add.items():
        if not key.startswith("d_"):
            continue
        col = key[2:]
        gap = abs(float(value or 0.0))
        if gap > _tol(col):
            ok = False
        if gap > worst:
            worst, worst_col = gap, col
    return worst, worst_col, ok


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def connect():
    con = duckdb.connect()
    nc.duck_settings(con, memory_limit=MEMORY_LIMIT)
    tmp = OUT_DIR / ".tmp_duck"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con


# ------------------------------------------------------------------ bridge

def _bridge(con):
    """FIPS -> ccg, reusing producer 72's classifier. Fails loud on unmatched.

    Returns the bridge frame and the inventory of everything deliberately not
    placed (blank / non-numeric FIPS, and the non-state territories), which the
    report carries as a coverage loss rather than absorbing silently.
    """
    log("scanning distinct FIPS from the panel")
    raw = con.execute(f"""
        SELECT coalesce(rtrim(fips), '') AS fips_raw,
               count(*)                      AS n_estab,
               sum(coalesce(emp_primary, 0)) AS emp
        FROM read_parquet('{PANEL}')
        GROUP BY 1
    """).df()
    log(f"{len(raw):,} distinct raw FIPS, {int(raw.n_estab.sum()):,} estab-years")

    xwalk = pd.read_csv(dc.CCG_CROSSWALK,
                        dtype={"county_fips5": str, "ccg_id": str})
    cov = b72.classify(raw, xwalk)

    bad = cov[cov.status.str.startswith(("unmatched", "unplaceable"))]
    if not bad.empty:
        rows = "\n".join(f"  {r.fips_raw!r}: {r.status} ({int(r.n_estab):,} "
                         f"establishment-years)" for _, r in bad.iterrows())
        raise RuntimeError(
            "NETS county FIPS that place into no ccg (the plan's fail-loud "
            f"merge rule; adjudicate before collapsing):\n{rows}")

    dropped = cov[cov.ccg_id.isna()].copy()
    keep = cov[cov.ccg_id.notna()][["fips_raw", "ccg_id"]].copy()
    keep["ccg"] = keep.ccg_id.astype(int)
    con.register("bridge", keep[["fips_raw", "ccg"]])
    log(f"bridge: {len(keep):,} FIPS -> {keep.ccg.nunique():,} ccgs; "
        f"{len(dropped):,} FIPS not placed "
        f"({int(dropped.n_estab.sum()):,} establishment-years)")
    return cov, dropped


# ------------------------------------------------------- primary-line EmpC

def _primary_empc(con, force=False):
    """EmpC of the line whose employment is `emp_primary`, per establishment-year.

    The establishment panel exposes only `share_emp_imputed` and
    `any_actual_emp`; the 4-way EmpC code lives at DUNS level in
    `long/emp_long.parquet`. `estab_xwalk.is_primary` marks the line the
    establishment's `emp_primary` comes from, so the join is one-to-one and the
    resulting code describes exactly the employment figure being aggregated.
    """
    if EMPC_TMP.exists() and not force:
        n = con.execute(
            f"SELECT count(*) FROM read_parquet('{EMPC_TMP}')").fetchone()[0]
        log(f"primary-line EmpC: reusing {EMPC_TMP.name} ({n:,} rows)")
        return
    log("building primary-line EmpC (emp_long x estab_xwalk); this is the "
        "heavy join")
    con.execute(f"""
        COPY (
            SELECT x.estab_id,
                   CAST(x.year AS SMALLINT)  AS year,
                   CAST(e.emp_code AS TINYINT) AS emp_code
            FROM read_parquet('{XWALK}') x
            JOIN read_parquet('{EMP_LONG}') e
              ON e.duns = CAST(x.duns AS UINTEGER) AND e.year = x.year
            WHERE x.is_primary
        ) TO '{EMPC_TMP}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{EMPC_TMP}')").fetchone()[0]
    log(f"primary-line EmpC: {n:,} establishment-years")


# --------------------------------------------------------------- measures

# Every measure is a sum or a count: all of them re-collapse additively.
MEASURES = """
        sum(p.emp_primary)                                   AS emp,
        count(*)                                             AS n_estab,
        sum(p.n_lines)                                       AS n_lines,
        sum(p.emp_primary * coalesce(p.share_emp_imputed, 0)) AS emp_imputed,
        sum(CASE WHEN p.any_actual_emp THEN p.emp_primary END) AS emp_actual,
        count(*) FILTER (p.any_actual_emp)                   AS n_actual,
        sum(p.emp_sum_exhq)                                  AS emp_sum_exhq,
        sum(p.emp_hq)                                        AS emp_hq,
        count(*) FILTER (p.small_estab)                      AS n_small,
        count(*) FILTER (p.sic_education)                    AS n_sic_education,
        count(*) FILTER (p.blacklisted)                      AS n_blacklisted,
        count(*) FILTER (p.quarantined)                      AS n_quarantined
"""

EMPC_MEASURES = "".join(f"""
        sum(CASE WHEN c.emp_code = {k} THEN p.emp_primary END) AS emp_empc{k},
        count(*) FILTER (c.emp_code = {k})                     AS n_empc{k},"""
                        for k in range(4)) + """
        count(*) FILTER (c.emp_code IS NULL)                   AS n_empc_missing
"""

MEASURE_COLS = ["emp", "n_estab", "n_lines", "emp_imputed", "emp_actual",
                "n_actual", "emp_sum_exhq", "emp_hq", "n_small",
                "n_sic_education", "n_blacklisted", "n_quarantined"]
EMPC_COLS = ([f"emp_empc{k}" for k in range(4)]
             + [f"n_empc{k}" for k in range(4)] + ["n_empc_missing"])

ALL_COLS = MEASURE_COLS + EMPC_COLS


def universe_sql(keys):
    """'all' and 'actual', the two universes of gate rule 1.

    `cells` is grouped one level finer than the file's key, by
    `any_actual_emp`, so both universes come out of the same retained
    establishments in one pass: 'all' sums over both branches, 'actual' keeps
    the true branch. The 'all' rows must be re-aggregated -- dropping the
    `any_actual_emp` column without a GROUP BY would leave two rows per cell.
    """
    k = ", ".join(keys)
    sums = ", ".join(f"sum({c}) AS {c}" for c in ALL_COLS)
    return f"""
        SELECT 'all' AS universe, {k}, {sums} FROM cells GROUP BY {k}
        UNION ALL
        SELECT 'actual' AS universe, {k}, {sums} FROM cells
        WHERE any_actual_emp GROUP BY {k}
    """


def _collapse_industry(con):
    """The fine file: universe x ccg x year x naics4 x sic2, EmpC carried."""
    log("collapsing to ccg x naics4 x sic2 x year")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW cells AS
        SELECT b.ccg                                    AS ccg,
               CAST(p.year AS SMALLINT)                 AS year,
               coalesce(nullif(substr(p.naics, 1, 4), ''), '{UNKNOWN}') AS naics4,
               coalesce(nullif(substr(p.sic8, 1, 2), ''), '{UNKNOWN}')  AS sic2,
               p.any_actual_emp                         AS any_actual_emp,
               {MEASURES},
               {EMPC_MEASURES}
        FROM read_parquet('{PANEL}') p
        JOIN bridge b ON b.fips_raw = coalesce(rtrim(p.fips), '')
        LEFT JOIN read_parquet('{EMPC_TMP}') c
               ON c.estab_id = p.estab_id AND c.year = p.year
        GROUP BY 1, 2, 3, 4, 5
    """)
    con.execute(f"""
        COPY ({universe_sql(["ccg", "year", "naics4", "sic2"])})
        TO '{INDUSTRY_OUT}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{INDUSTRY_OUT}')").fetchone()[0]
    log(f"wrote {INDUSTRY_OUT.name}: {n:,} rows, "
        f"{INDUSTRY_OUT.stat().st_size / 1e6:.0f} MB")


def _collapse_county(con):
    """The county file, aggregated INDEPENDENTLY from the panel.

    Deliberately not a re-collapse of the fine file: the gate's Q8 additivity
    check has to compare two independent aggregations, or it asserts nothing.
    The EmpC columns are the one exception -- they ride the same query, over
    the same source rows, but with no cell key.
    """
    log("collapsing to ccg x year (independent pass)")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW cells AS
        SELECT b.ccg                     AS ccg,
               CAST(p.year AS SMALLINT)  AS year,
               p.any_actual_emp          AS any_actual_emp,
               {MEASURES},
               {EMPC_MEASURES}
        FROM read_parquet('{PANEL}') p
        JOIN bridge b ON b.fips_raw = coalesce(rtrim(p.fips), '')
        LEFT JOIN read_parquet('{EMPC_TMP}') c
               ON c.estab_id = p.estab_id AND c.year = p.year
        GROUP BY 1, 2, 3
    """)
    con.execute(f"""
        COPY ({universe_sql(["ccg", "year"])}) TO '{COUNTY_OUT}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{COUNTY_OUT}')").fetchone()[0]
    log(f"wrote {COUNTY_OUT.name}: {n:,} rows, "
        f"{COUNTY_OUT.stat().st_size / 1e6:.1f} MB")


# ----------------------------------------------------------------- checks

def _checks(con, cov, dropped):
    """Blocking verification. Every entry lands in the report with PASS/FAIL."""
    out = {}
    scanned = int(cov.n_estab.sum())
    out["row_conservation"] = (scanned, PANEL_ROWS, scanned == PANEL_ROWS)

    placed = int(cov[cov.ccg_id.notna()].n_estab.sum())
    lost = int(dropped.n_estab.sum())
    out["placement"] = (placed, lost, placed + lost == PANEL_ROWS)
    out["dropped_detail"] = (
        dropped.groupby("rule")
        .agg(fips=("fips_raw", "nunique"), n_estab=("n_estab", "sum"),
             emp=("emp", "sum"))
        .sort_values("n_estab", ascending=False))

    # Employment and establishment conservation, county file vs the panel.
    out["conservation"] = con.execute(f"""
        SELECT sum(emp) AS emp, sum(n_estab) AS n_estab
        FROM read_parquet('{COUNTY_OUT}') WHERE universe = 'all'
    """).df().iloc[0].to_dict()
    out["panel_placed"] = con.execute(f"""
        SELECT sum(p.emp_primary) AS emp, count(*) AS n_estab
        FROM read_parquet('{PANEL}') p
        JOIN bridge b ON b.fips_raw = coalesce(rtrim(p.fips), '')
    """).df().iloc[0].to_dict()

    # Gate Q2: exact additivity of every measure, both universes, all rungs.
    sums = ", ".join(f"sum({c}) AS {c}" for c in MEASURE_COLS + EMPC_COLS)
    diffs = ", ".join(
        f"max(abs(coalesce(a.{c}, 0) - coalesce(b.{c}, 0))) AS d_{c}"
        for c in MEASURE_COLS + EMPC_COLS)
    out["additivity"] = con.execute(f"""
        WITH rolled AS (
            SELECT universe, ccg, year, {sums}
            FROM read_parquet('{INDUSTRY_OUT}') GROUP BY 1, 2, 3
        )
        SELECT {diffs},
               count(*) AS n_pairs
        FROM read_parquet('{COUNTY_OUT}') a
        FULL OUTER JOIN rolled b USING (universe, ccg, year)
    """).df().iloc[0].to_dict()
    out["additivity_keys"] = con.execute(f"""
        WITH r AS (SELECT DISTINCT universe, ccg, year
                   FROM read_parquet('{INDUSTRY_OUT}')),
             c AS (SELECT DISTINCT universe, ccg, year
                   FROM read_parquet('{COUNTY_OUT}'))
        SELECT (SELECT count(*) FROM c) AS n_county,
               (SELECT count(*) FROM r) AS n_industry,
               (SELECT count(*) FROM (SELECT * FROM c EXCEPT SELECT * FROM r))
                   AS county_only,
               (SELECT count(*) FROM (SELECT * FROM r EXCEPT SELECT * FROM c))
                   AS industry_only
    """).df().iloc[0].to_dict()

    # A.1.7: the coarser rungs must re-collapse from the same fine file.
    out["rungs"] = con.execute(f"""
        SELECT (SELECT count(DISTINCT naics4) FROM read_parquet('{INDUSTRY_OUT}'))
                   AS n_naics4,
               (SELECT count(DISTINCT substr(naics4, 1, 2))
                  FROM read_parquet('{INDUSTRY_OUT}')) AS n_naics2,
               (SELECT count(DISTINCT sic2) FROM read_parquet('{INDUSTRY_OUT}'))
                   AS n_sic2,
               (SELECT count(*) FROM read_parquet('{INDUSTRY_OUT}')
                 WHERE naics4 = '{UNKNOWN}') AS n_naics_unknown,
               (SELECT count(*) FROM read_parquet('{INDUSTRY_OUT}')
                 WHERE sic2 = '{UNKNOWN}') AS n_sic_unknown
    """).df().iloc[0].to_dict()

    # Gate rule 1: one universe, both rungs -- the same ccg x year set.
    out["universe_sets"] = con.execute(f"""
        WITH a AS (SELECT DISTINCT ccg, year FROM read_parquet('{COUNTY_OUT}')
                   WHERE universe = 'actual'),
             b AS (SELECT DISTINCT ccg, year FROM read_parquet('{INDUSTRY_OUT}')
                   WHERE universe = 'actual')
        SELECT (SELECT count(*) FROM (SELECT * FROM a EXCEPT SELECT * FROM b))
                   AS county_only,
               (SELECT count(*) FROM (SELECT * FROM b EXCEPT SELECT * FROM a))
                   AS industry_only
    """).df().iloc[0].to_dict()

    # EmpC composition against the documented 42.5% actual share.
    out["empc"] = con.execute(f"""
        SELECT sum(n_empc0) AS n0, sum(n_empc1) AS n1, sum(n_empc2) AS n2,
               sum(n_empc3) AS n3, sum(n_empc_missing) AS n_missing,
               sum(n_estab) AS n_estab, sum(n_actual) AS n_actual
        FROM read_parquet('{COUNTY_OUT}') WHERE universe = 'all'
    """).df().iloc[0].to_dict()

    locked = set(pd.read_parquet(b72.ANALYSIS_PANEL, columns=["ccg"])
                 .ccg.astype(int).unique())
    seen = set(con.execute(
        f"SELECT DISTINCT ccg FROM read_parquet('{COUNTY_OUT}')").df().ccg)
    out["support"] = (len(locked), len(seen), len(seen - locked),
                      sorted(locked - seen))

    out["by_year"] = con.execute(f"""
        SELECT year, universe, count(*) AS n_ccg, sum(emp) AS emp,
               sum(n_estab) AS n_estab
        FROM read_parquet('{COUNTY_OUT}') GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    out["cells_by_universe"] = con.execute(f"""
        SELECT universe, count(*) AS n_cells, sum(n_estab) AS n_estab,
               sum(emp) AS emp
        FROM read_parquet('{INDUSTRY_OUT}') GROUP BY 1 ORDER BY 1
    """).df()
    return out


def _schema(con, path):
    return con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 1").df()


def write_report(con, chk):
    def flag(ok):
        return "PASS" if ok else "FAIL"

    L = []
    L.append("# WS2 step 1, second half: the NETS cell collapse\n\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
             "`step_1/code/py/73_nets_cell_collapse.py`. Companion to "
             "`ws2_step1_bridge_report.md` (the FIPS bridge). The fail-loud "
             "gate on unmatched FIPS is armed in this producer.\n\n")

    L.append("## The two datasets\n\n")
    L.append(f"Written to `{OUT_DIR}`.\n\n")
    L.append("| File | Key | Rows | Size |\n|---|---|---:|---:|\n")
    for path, key in ((COUNTY_OUT, "universe, ccg, year"),
                      (INDUSTRY_OUT, "universe, ccg, year, naics4, sic2")):
        n = con.execute(
            f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
        L.append(f"| `{path.name}` | `{key}` | {n:,} | "
                 f"{path.stat().st_size / 1e6:,.1f} MB |\n")
    L.append("\nEvery measure is a sum or a count, so each re-collapses "
             "additively; no shares are stored. A zero cell is the absence of "
             "a row -- nothing is dropped for being small, imputed, or "
             "short-lived (gate Q2 rule 3).\n\n")
    L.append("Columns of `nets_ccg_industry_year.parquet`:\n\n")
    L.append("| column | type |\n|---|---|\n")
    for _, r in _schema(con, INDUSTRY_OUT).iterrows():
        L.append(f"| `{r.column_name}` | {r.column_type} |\n")
    L.append("\nThe county file carries the same columns without `naics4` and "
             "`sic2`.\n")

    L.append("\n## Coverage loss\n\n")
    scanned, expected, ok = chk["row_conservation"]
    placed, lost, place_ok = chk["placement"]
    L.append(f"Establishment-years scanned: {scanned:,} (manifest 66 expects "
             f"{expected:,}; {flag(ok)}). Placed into a ccg: {placed:,} "
             f"({placed / scanned:.4%}). Not placed and therefore absent from "
             f"both files: {lost:,} ({lost / scanned:.4%}).\n\n")
    L.append("| rule | distinct FIPS | establishment-years | employment |\n")
    L.append("|---|---:|---:|---:|\n")
    for rule, r in chk["dropped_detail"].iterrows():
        L.append(f"| `{rule}` | {int(r.fips):,} | {int(r.n_estab):,} | "
                 f"{r.emp:,.0f} |\n")
    L.append("\nBlank FIPS are establishment-years with no county at all; the "
             "decision recorded in `ws2_step1_bridge_report.md` was to drop "
             "them and report the loss rather than recover them from the "
             "address spells. `nonstate` is the territories, outside the "
             "county universe the ccg support is defined on. Both are reported "
             "here so the NETS-vs-EFSY coverage comparison in step 2 prices "
             "them, rather than absorbing them silently.\n")

    L.append("\n## Universes\n\n")
    L.append("`universe = 'all'` is the full placed panel; "
             "`universe = 'actual'` restricts to establishments with "
             "`any_actual_emp` (EmpC = 0 on the primary line). The restriction "
             "defines the universe, so both rungs are grouped from the same "
             "retained establishments and the county total is recomputed "
             "inside each universe -- never a cell-level-only restriction "
             "(gate Q2 rule 2). The cutoff is fixed here, before any step-2 "
             "benchmark number.\n\n")
    L.append("| universe | industry cells | establishment-years | employment |\n")
    L.append("|---|---:|---:|---:|\n")
    for _, r in chk["cells_by_universe"].iterrows():
        L.append(f"| {r.universe} | {int(r.n_cells):,} | {int(r.n_estab):,} | "
                 f"{r.emp:,.0f} |\n")

    L.append("\n## Checks\n\n")
    L.append(f"- Row conservation: {flag(ok)} ({scanned:,} vs {expected:,}).\n")
    L.append(f"- Placement accounting: {flag(place_ok)} "
             f"({placed:,} placed + {lost:,} unplaced = {scanned:,}).\n")
    L.append("- Unmatched FIPS: none reached the collapse; the producer "
             "raises on any `unmatched`/`unplaceable` status before writing.\n")

    cons, panel = chk["conservation"], chk["panel_placed"]
    emp_ok = abs(cons["emp"] - panel["emp"]) < 1e-6
    n_ok = int(cons["n_estab"]) == int(panel["n_estab"])
    L.append(f"- Employment conservation, county file vs panel: {flag(emp_ok)} "
             f"({cons['emp']:,.0f} vs {panel['emp']:,.0f}).\n")
    L.append(f"- Establishment conservation: {flag(n_ok)} "
             f"({int(cons['n_estab']):,} vs {int(panel['n_estab']):,}).\n")

    add = chk["additivity"]
    worst, worst_col, add_ok = _additivity_verdict(add)
    emp_gap = abs(float(add["d_emp"] or 0.0))
    keys = chk["additivity_keys"]
    key_ok = keys["county_only"] == 0 and keys["industry_only"] == 0
    L.append(f"- **Exact additivity (gate Q2), `max |X_ct - sum_m X_cmt|` over "
             f"every universe x ccg x year and all {len(ALL_COLS)} measures: "
             f"{flag(add_ok)} (worst = {worst:g} on `{worst_col}`, over "
             f"{int(add['n_pairs']):,} pairs). The outcome itself, `emp`: "
             f"{emp_gap:g}. Only `emp_imputed` carries a tolerance "
             f"({ADDITIVITY_TOL['emp_imputed']:g}), being the one fractional "
             f"sum; every other measure must agree exactly.**\n")
    L.append(f"- Key agreement between the two files: {flag(key_ok)} "
             f"({int(keys['n_county']):,} county keys, "
             f"{int(keys['county_only']):,} county-only, "
             f"{int(keys['industry_only']):,} industry-only).\n")

    us = chk["universe_sets"]
    us_ok = us["county_only"] == 0 and us["industry_only"] == 0
    L.append(f"- One universe, both rungs (`actual` ccg x year sets agree): "
             f"{flag(us_ok)}.\n")

    rungs = chk["rungs"]
    L.append(f"- Rungs available from the fine file: "
             f"{int(rungs['n_naics4']):,} NAICS4, {int(rungs['n_naics2']):,} "
             f"NAICS2, {int(rungs['n_sic2']):,} SIC2 "
             f"(including the `{UNKNOWN}` sentinel: "
             f"{int(rungs['n_naics_unknown']):,} rows with no NAICS, "
             f"{int(rungs['n_sic_unknown']):,} with no SIC; kept as cells, "
             f"never dropped).\n")

    e = chk["empc"]
    n_estab = e["n_estab"]
    empc_sum = sum(e[k] for k in ("n0", "n1", "n2", "n3", "n_missing"))
    share0 = e["n0"] / n_estab
    L.append(f"- EmpC buckets sum to the establishment count: "
             f"{flag(int(empc_sum) == int(n_estab))} "
             f"({int(empc_sum):,} vs {int(n_estab):,}; "
             f"{int(e['n_missing']):,} with no primary-line EmpC).\n")
    L.append(f"- EmpC = 0 share: {share0:.3%} against the "
             f"{EMPC_ACTUAL_SHARE:.1%} documented in `NETS_FED2025.md` "
             f"({flag(abs(share0 - EMPC_ACTUAL_SHARE) < 0.05)}).\n")
    L.append(f"- `n_actual` (panel's `any_actual_emp`) vs `n_empc0` "
             f"(primary-line EmpC = 0): {int(e['n_actual']):,} vs "
             f"{int(e['n0']):,}. `any_actual_emp` is true if ANY line of the "
             f"establishment reports an actual figure, so it is weakly the "
             f"larger; the `actual` universe uses it.\n")

    n_locked, n_seen, surplus, missing = chk["support"]
    L.append(f"- Support: {n_seen:,} ccgs against the locked {n_locked:,}; "
             f"{surplus:,} outside the locked support, {len(missing):,} locked "
             f"ccgs with no NETS establishment. The EFSY intersection is NOT "
             f"applied here -- the gate fixes it in step 2, and applying it "
             f"now would bake one exhibit's sample into the shared inputs.\n")

    L.append("\n## Employment by year\n\n")
    L.append("| year | ccgs (all) | employment (all) | establishments (all) | "
             "employment (actual) | establishments (actual) |\n")
    L.append("|---:|---:|---:|---:|---:|---:|\n")
    by = chk["by_year"].pivot(index="year", columns="universe")
    for year, r in by.iterrows():
        L.append(f"| {int(year)} | {int(r[('n_ccg', 'all')]):,} | "
                 f"{r[('emp', 'all')]:,.0f} | "
                 f"{int(r[('n_estab', 'all')]):,} | "
                 f"{r[('emp', 'actual')]:,.0f} | "
                 f"{int(r[('n_estab', 'actual')]):,} |\n")

    # The actual-employment universe is a coverage series in its own right, and
    # a break in it would masquerade as an employment shock in any regression
    # run inside that universe. Flagged here, in step 1, so step 2 prices it.
    piv = chk["by_year"].pivot(index="year", columns="universe", values="emp")
    share = (piv["actual"] / piv["all"]).sort_index()
    breaks = [(int(y), share.loc[y], share.loc[y] - share.shift(1).loc[y])
              for y in share.index if abs(share.shift(1).loc[y] or 0) > 0
              and abs(share.loc[y] - share.shift(1).loc[y]) > 0.10]
    L.append("\n## Coverage break in the `actual` universe\n\n")
    L.append("The `actual` universe's share of placed employment, year by "
             "year, is itself a coverage series: a jump in it is a change in "
             "who reports an actual figure, not a change in employment. Any "
             "year-pair spanning such a jump produces a growth rate driven by "
             "reporting, so step 2 must price these before the outcome swap, "
             "and step 3 must not straddle one silently.\n\n")
    if breaks:
        L.append("| year | actual share of employment | change |\n")
        L.append("|---:|---:|---:|\n")
        for y, s, d in breaks:
            L.append(f"| {y} | {s:.1%} | {d:+.1%} |\n")
        L.append(f"\nRange over 1990-2023: {share.min():.1%} "
                 f"({int(share.idxmin())}) to {share.max():.1%} "
                 f"({int(share.idxmax())}).\n")
    else:
        L.append("No year moves the share by more than 10 points. Range: "
                 f"{share.min():.1%} ({int(share.idxmin())}) to "
                 f"{share.max():.1%} ({int(share.idxmax())}).\n")

    L.append("\n## What reads these files\n\n")
    L.append("WS2 steps 2-5 read only these two parquets. The "
             "establishment-level panel is not read again: the plan's tooling "
             "rule makes this collapse the single entry point. Growth rates, "
             "the shock merge, the EFSY intersection and every A.3.4 exclusion "
             "belong to the consuming step, not here.\n")

    REPORT_OUT.write_text("".join(L))
    return chk


def main():
    force = "--rebuild-empc" in sys.argv
    # --report-only re-runs the checks and the memo against cell files that
    # already exist. It never rewrites them, so a report can be regenerated
    # without a 15-minute rebuild; the collapse itself stays the only writer.
    report_only = "--report-only" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = connect()
    try:
        cov, dropped = _bridge(con)
        if report_only:
            missing = [p for p in (COUNTY_OUT, INDUSTRY_OUT) if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    "--report-only needs the cell files: "
                    + ", ".join(str(p) for p in missing))
            log("report-only: reusing the existing cell files")
        else:
            _primary_empc(con, force=force)
            _collapse_industry(con)
            _collapse_county(con)
        log("running checks")
        chk = _checks(con, cov, dropped)
        write_report(con, chk)
        log(f"wrote {REPORT_OUT}")

        worst, worst_col, add_ok = _additivity_verdict(chk["additivity"])
        if not add_ok:
            raise RuntimeError(
                "gate Q2 additivity failed: the county file and the industry "
                f"file disagree by up to {worst:g} on `{worst_col}`. "
                f"See {REPORT_OUT}.")
        log("additivity PASS; cells are ready for WS2 steps 2-5")
    finally:
        con.close()


if __name__ == "__main__":
    main()
