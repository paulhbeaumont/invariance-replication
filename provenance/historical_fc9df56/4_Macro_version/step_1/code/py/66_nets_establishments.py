"""NETS Fed2025 cleaning, producer 66: business lines to physical establishments.

Collapses DUNS records that are business lines of the same physical plant into
establishments, following Barnatchez-Crane-Decker as refined by Crane and
Decker (2019, pp. 10-11): lines sharing a location and a firm are one
establishment; employment is summed; the industry code comes from the largest
line. Grouping is per (year, normalized address), connecting DUNS that share
an ultimate family (producer 64) or a normalized company name, with the
persistent establishment id linked across years only through shared DUNS
membership (Crane-Decker's netsnum logic), never through address continuity.

Inputs: long/{emp,sic,naics}_long, long/misc_dim, firm/ultimate_parent,
estab/address_spells + address_spell_map, raw/company.
Outputs:
  estab/estab_xwalk.parquet   duns, year, estab_id_year, estab_id,
                              provenance, is_primary, gap_bridged,
                              blacklisted, quarantined, colocated
  estab/estab_year_panel.parquet  estab_id x year with employment variants,
                              industry, location, restriction flags
  _reports/                   EstCat tabulation, HQ-employment test, suite
                              diagnostic, component audit, manifest

Pinned facts (NETS Database Description 2019, field definitions):
- EmpC: 0 = actual figure, 1 = bottom of range, 2 = D&B estimate, 3 = Walls
  estimate. share_emp_imputed weights emp with EmpC <> 0.
- EstCat is the establishment role (Single Location / Headquarters / Branch /
  Division); LevelCode is geocode precision, not hierarchy.
- Emp is "Establishment Employment at Location" (employees here), so HQ rows
  should carry site employment, not firm totals; the HQ-employment test
  checks this empirically and reports before the collapse runs.

Design decisions carried from the approved plan:
- Components come from min-label propagation over (cell, family) and
  (cell, name) partitions: the SQL-native equivalent of union-find on the
  group-label bipartite graph; no pairwise DUNS edges are materialized. Only
  multi-DUNS cells enter propagation; labels are monotone decreasing, so
  convergence is checked by an unchanged label sum.
- Guards: blank addresses and PO boxes are singletons; addresses hosting >25
  distinct families in a year are blacklisted (registered agents); cells with
  >500 DUNS are quarantined and left as singletons.
- Suite-stripping is chosen by a 3-year diagnostic (2000, 2010, 2016):
  the variant whose collapse rate is closer to the Crane-Decker-implied
  magnitude (small; target 1.05, band 1.00-1.15 DUNS per establishment)
  wins. The plan's CBP-count criterion is downgraded because the local CBP
  file carries employment, not establishment counts; recorded in the
  manifest.
- Persistent ids: components at year y link to y-1 by max shared-DUNS
  overlap, mutual-best only, ties broken toward smaller component-min-DUNS;
  a still-unlinked component then tries y-2 (gap_bridged). Non-linked
  components start new ids with the best-overlap predecessor recorded as
  parent lineage. DUNS reuse past LastYear cannot occur because the universe
  ends at LastYear.
- Primary record fixed at the persistent-id level (max mean employment, then
  earliest FirstYear, then min DUNS); each year's representative is the
  primary when present, else the largest line that year.
Gates (blocking): propagation converges; collapse rate within [1.00, 1.15];
none of the 10 largest components mixes >3 distinct names with zero shared
family; xwalk rows reconcile to the universe.
"""

import json
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nets_common as nc

XWALK_OUT = nc.ESTAB / "estab_xwalk.parquet"
PANEL_OUT = nc.ESTAB / "estab_year_panel.parquet"
DIAG_YEARS = (2000, 2010, 2016)
COLLAPSE_BAND = (1.00, 1.15)
COLLAPSE_TARGET = 1.05
FAMILY_BLACKLIST = 25
CELL_QUARANTINE = 500
PROP_ROUNDS = 12


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_base(con, qa):
    con.execute(f"""
        CREATE TEMP TABLE life AS
        SELECT CAST(DunsNumber AS UINTEGER) AS duns,
               greatest(CAST(FirstYear AS INTEGER), 1990) AS fy,
               least(CAST(LastYear AS INTEGER), 2023) AS ly,
               rtrim(EstCat) AS estcat,
               CAST(FirstYear AS INTEGER) AS first_year_raw
        FROM read_parquet('{nc.LONG}/misc_dim.parquet')
    """)
    tab = con.execute(
        "SELECT estcat, count(*) FROM life GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    qa["estcat"] = {k or "": v for k, v in tab}
    hq_vals = [k for k, _ in tab if k and "HEADQUARTER" in k.upper()]
    qa["hq_estcat_values"] = hq_vals
    con.execute(f"""
        COPY (SELECT estcat, count(*) AS n FROM life GROUP BY 1 ORDER BY 2 DESC)
        TO '{nc.REPORTS}/estcat_tabulation.csv' (FORMAT csv, HEADER)
    """)
    log(f"EstCat: {qa['estcat']}; HQ set = {hq_vals}")

    con.execute(f"""
        CREATE TEMP TABLE universe AS
        SELECT e.duns, e.year, e.emp, e.emp_code
        FROM read_parquet('{nc.LONG}/emp_long.parquet') e
        JOIN life l ON l.duns = e.duns
        WHERE e.year BETWEEN l.fy AND l.ly
    """)
    in_u = con.execute("SELECT count(*) FROM universe").fetchone()[0]
    all_e = con.execute(
        f"SELECT count(*) FROM read_parquet('{nc.LONG}/emp_long.parquet')"
    ).fetchone()[0]
    qa["universe"] = {"duns_years": in_u, "outside_life_span": all_e - in_u}
    log(f"universe: {in_u:,} duns-years ({all_e - in_u:,} outside life span)")

    con.execute(f"""
        CREATE TEMP TABLE base AS
        SELECT u.duns, u.year, u.emp, u.emp_code,
               coalesce(CAST(up.ult_duns AS UINTEGER), u.duns) AS fam,
               s.address, s.fips,
               l.estcat IN ({', '.join(repr(v) for v in hq_vals) or "''"})
                   AS is_hq_row
        FROM universe u
        JOIN life l ON l.duns = u.duns
        LEFT JOIN read_parquet('{nc.FIRM}/ultimate_parent.parquet') up
          ON CAST(up.duns AS UINTEGER) = u.duns AND up.year = u.year
        LEFT JOIN read_parquet('{nc.ESTAB}/address_spell_map.parquet') m
          ON CAST(m.duns AS UINTEGER) = u.duns AND m.year = u.year
        LEFT JOIN read_parquet('{nc.ESTAB}/address_spells.parquet') s
          ON s.duns = m.duns AND s.spell_id = m.spell_id
    """)
    con.execute(f"""
        CREATE TEMP TABLE names AS
        SELECT CAST(DunsNumber AS UINTEGER) AS duns,
               nullif({nc.sql_normalize_name('Company')}, '') AS name
        FROM read_parquet('{nc.RAW}/company.parquet')
    """)
    log("base built")


def hq_employment_test(con, qa):
    r = con.execute("""
        WITH fam_year AS (
            SELECT fam, year,
                   sum(emp) FILTER (WHERE is_hq_row) AS hq_emp,
                   sum(emp) FILTER (WHERE NOT is_hq_row) AS other_emp,
                   count(*) AS members
            FROM base WHERE year IN (2000, 2010, 2020)
            GROUP BY 1, 2
        )
        SELECT count(*),
               quantile_cont(hq_emp / nullif(other_emp, 0), 0.5),
               avg((hq_emp >= other_emp)::INT)
        FROM fam_year
        WHERE members > 1 AND hq_emp IS NOT NULL AND other_emp IS NOT NULL
    """).fetchone()
    qa["hq_employment_test"] = {
        "multi_member_family_years": r[0],
        "median_hq_to_other_ratio": r[1],
        "share_hq_geq_other": r[2]}
    log(f"HQ employment test: {qa['hq_employment_test']}")


def group_one(con, strip_suite, years_filter, tag):
    addr_expr = nc.sql_normalize_address("b.address", strip_suite=strip_suite)
    yf = f"AND b.year IN {years_filter}" if years_filter else ""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE work_{tag} AS
        WITH cells AS (
            SELECT b.duns, b.year, b.fam, n.name,
                   CASE WHEN b.address IS NULL OR rtrim(b.address) = ''
                             OR {nc.sql_is_pobox('b.address')}
                        THEN NULL ELSE {addr_expr} END AS addr
            FROM base b LEFT JOIN names n ON n.duns = b.duns
            WHERE true {yf}
        ),
        bl AS (
            SELECT year, addr FROM cells WHERE addr IS NOT NULL
            GROUP BY 1, 2 HAVING count(DISTINCT fam) > {FAMILY_BLACKLIST}
        ),
        quar AS (
            SELECT year, addr FROM cells WHERE addr IS NOT NULL
            GROUP BY 1, 2 HAVING count(*) > {CELL_QUARANTINE}
        )
        SELECT c.duns, c.year, c.fam, c.name, c.addr,
               bl.addr IS NOT NULL AS blacklisted,
               quar.addr IS NOT NULL AS quarantined
        FROM cells c
        LEFT JOIN bl ON bl.year = c.year AND bl.addr = c.addr
        LEFT JOIN quar ON quar.year = c.year AND quar.addr = c.addr
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE prop_{tag} AS
        SELECT w.duns, w.year, w.fam, w.name, w.addr, w.duns AS comp
        FROM work_{tag} w
        JOIN (SELECT year, addr FROM work_{tag}
              WHERE addr IS NOT NULL AND NOT blacklisted AND NOT quarantined
              GROUP BY 1, 2 HAVING count(*) > 1) m
          ON m.year = w.year AND m.addr = w.addr
        WHERE NOT w.blacklisted AND NOT w.quarantined
    """)
    prev_sum = None
    for rnd in range(PROP_ROUNDS):
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE prop_{tag} AS
            SELECT p.duns, p.year, p.fam, p.name, p.addr,
                   least(p.comp, f.mc, coalesce(n.mc, p.comp)) AS comp
            FROM prop_{tag} p
            JOIN (SELECT year, addr, fam, min(comp) AS mc
                  FROM prop_{tag} GROUP BY 1, 2, 3) f
              ON f.year = p.year AND f.addr = p.addr AND f.fam = p.fam
            LEFT JOIN (SELECT year, addr, name, min(comp) AS mc
                       FROM prop_{tag} WHERE name IS NOT NULL
                       GROUP BY 1, 2, 3) n
              ON n.year = p.year AND n.addr = p.addr AND n.name = p.name
        """)
        s = con.execute(
            f"SELECT sum(comp) FROM prop_{tag}").fetchone()[0]
        if s == prev_sum:
            break
        prev_sum = s
    else:
        raise RuntimeError(
            f"GATE FAILED [{tag}]: propagation not converged in "
            f"{PROP_ROUNDS} rounds")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE comp_{tag} AS
        SELECT w.duns, w.year, w.fam, w.name, w.addr, w.blacklisted,
               w.quarantined, coalesce(p.comp, w.duns) AS comp
        FROM work_{tag} w
        LEFT JOIN prop_{tag} p ON p.duns = w.duns AND p.year = w.year
    """)
    log(f"grouping [{tag}]: converged in {rnd + 1} rounds")
    return f"comp_{tag}"


def collapse_stats(con, comp_table):
    d, e = con.execute(f"""
        SELECT count(*), count(DISTINCT (year, comp)) FROM {comp_table}
    """).fetchone()
    return d, e, d / e


def suite_diagnostic(con, qa):
    res = {}
    for strip, tag in ((False, "keepsuite"), (True, "nosuite")):
        t = group_one(con, strip, DIAG_YEARS, tag)
        d, e, rate = collapse_stats(con, t)
        res[tag] = {"duns_years": d, "estabs": e, "collapse_rate": rate}
        log(f"diagnostic {tag}: collapse rate {rate:.4f}")
        for x in ("work", "prop", "comp"):
            con.execute(f"DROP TABLE IF EXISTS {x}_{tag}")
    chosen = min(res, key=lambda k: abs(res[k]["collapse_rate"]
                                        - COLLAPSE_TARGET))
    qa["suite_diagnostic"] = {**res, "chosen": chosen,
                              "criterion": f"collapse rate nearest "
                              f"{COLLAPSE_TARGET} (CBP establishment counts "
                              f"unavailable locally)"}
    log(f"suite diagnostic: chosen {chosen}")
    return chosen == "nosuite"


def persistent_ids(con, qa):
    con.execute("""
        CREATE TEMP TABLE cy AS
        SELECT duns, year, dense_rank() OVER (ORDER BY year, comp) AS cyid
        FROM comp_full
    """)
    con.execute("""
        CREATE TEMP TABLE pid_map (
            cyid BIGINT, pid BIGINT, parent_pid BIGINT, gap_bridged BOOLEAN)
    """)
    con.execute("""
        INSERT INTO pid_map
        SELECT DISTINCT cyid, cyid, NULL, false FROM cy WHERE year = 1990
    """)
    for y in range(1991, 2024):
        for gap, bridged in ((1, "false"), (2, "true")):
            if y - gap < 1990:
                continue
            con.execute(f"""
                INSERT INTO pid_map
                SELECT mut.c1, p.pid, NULL, {bridged}
                FROM (
                    WITH ov AS (
                        SELECT a.cyid AS c0, b.cyid AS c1, count(*) AS n,
                               min(a.duns) AS md
                        FROM cy a
                        JOIN cy b ON b.duns = a.duns
                        WHERE a.year = {y - gap} AND b.year = {y}
                          AND b.cyid NOT IN (SELECT cyid FROM pid_map)
                        GROUP BY 1, 2
                    ),
                    b0 AS (SELECT c0, c1 FROM (
                               SELECT c0, c1, row_number() OVER (
                                   PARTITION BY c0 ORDER BY n DESC, md) AS r
                               FROM ov) WHERE r = 1),
                    b1 AS (SELECT c1, c0 FROM (
                               SELECT c1, c0, row_number() OVER (
                                   PARTITION BY c1 ORDER BY n DESC, md) AS r
                               FROM ov) WHERE r = 1)
                    SELECT b1.c1, b1.c0
                    FROM b1 JOIN b0 ON b0.c0 = b1.c0 AND b0.c1 = b1.c1
                ) mut
                JOIN pid_map p ON p.cyid = mut.c0
            """)
        con.execute(f"""
            INSERT INTO pid_map
            SELECT u.cyid, u.cyid, best.parent_pid, false
            FROM (SELECT DISTINCT cyid FROM cy WHERE year = {y}
                  AND cyid NOT IN (SELECT cyid FROM pid_map)) u
            LEFT JOIN (
                SELECT c1 AS cyid,
                       arg_max(pid, struct_pack(n := n, p := -pid))
                           AS parent_pid FROM (
                    SELECT b.cyid AS c1, p.pid, count(*) AS n
                    FROM cy a
                    JOIN cy b ON b.duns = a.duns
                    JOIN pid_map p ON p.cyid = a.cyid
                    WHERE a.year = {y - 1} AND b.year = {y}
                    GROUP BY 1, 2)
                GROUP BY 1
            ) best ON best.cyid = u.cyid
        """)
    n_pid, n_cy = con.execute(
        "SELECT count(DISTINCT pid), count(*) FROM pid_map").fetchone()
    n_all = con.execute("SELECT count(DISTINCT cyid) FROM cy").fetchone()[0]
    if n_cy != n_all:
        raise RuntimeError(f"pid_map covers {n_cy} of {n_all} components")
    qa["persistent_ids"] = n_pid
    qa["year_components"] = n_all
    log(f"persistent ids: {n_pid:,} over {n_all:,} year-components")


def primaries_and_outputs(con, qa):
    con.execute("""
        CREATE TEMP TABLE xw AS
        SELECT c.duns, c.year, cy.cyid AS estab_id_year, p.pid AS estab_id,
               p.gap_bridged, c.blacklisted, c.quarantined, c.fam, c.name,
               c.addr, b.emp, b.emp_code, b.fips, b.is_hq_row
        FROM comp_full c
        JOIN cy ON cy.duns = c.duns AND cy.year = c.year
        JOIN pid_map p ON p.cyid = cy.cyid
        JOIN base b ON b.duns = c.duns AND b.year = c.year
    """)
    qa["universe_rows"] = con.execute(
        "SELECT count(*) FROM universe").fetchone()[0]
    for t in ("base", "universe", "comp_full", "work_full", "prop_full",
              "cy", "pid_map", "names"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.execute("""
        CREATE TEMP TABLE prim AS
        SELECT estab_id, duns AS primary_duns FROM (
            SELECT estab_id, duns, row_number() OVER (
                PARTITION BY estab_id
                ORDER BY mean_emp DESC, first_year_raw, duns) AS r
            FROM (
                SELECT x.estab_id, x.duns, avg(x.emp) AS mean_emp,
                       any_value(l.first_year_raw) AS first_year_raw
                FROM xw x JOIN life l ON l.duns = x.duns
                GROUP BY x.estab_id, x.duns)
        ) WHERE r = 1
    """)
    con.execute("""
        CREATE TEMP TABLE coloc AS
        SELECT year, addr FROM xw WHERE addr IS NOT NULL
        GROUP BY 1, 2 HAVING count(DISTINCT estab_id_year) > 1
    """)
    con.execute("""
        CREATE TEMP TABLE famlink AS
        SELECT year, estab_id_year, fam FROM xw
        GROUP BY 1, 2, 3 HAVING count(*) > 1
    """)
    con.execute("""
        CREATE TEMP TABLE namelink AS
        SELECT year, estab_id_year, name FROM xw WHERE name IS NOT NULL
        GROUP BY 1, 2, 3 HAVING count(*) > 1
    """)
    con.execute(f"""
        COPY (
            SELECT lpad(CAST(x.duns AS VARCHAR), 9, '0') AS duns, x.year,
                   x.estab_id_year, x.estab_id,
                   CASE WHEN fl.fam IS NOT NULL AND nl.name IS NOT NULL
                            THEN 'both'
                        WHEN fl.fam IS NOT NULL THEN 'family'
                        WHEN nl.name IS NOT NULL THEN 'name'
                        ELSE 'singleton' END AS provenance,
                   x.duns = pr.primary_duns AS is_primary,
                   x.gap_bridged, x.blacklisted, x.quarantined,
                   co.addr IS NOT NULL AS colocated
            FROM xw x
            JOIN prim pr ON pr.estab_id = x.estab_id
            LEFT JOIN famlink fl ON fl.year = x.year
                 AND fl.estab_id_year = x.estab_id_year AND fl.fam = x.fam
            LEFT JOIN namelink nl ON nl.year = x.year
                 AND nl.estab_id_year = x.estab_id_year AND nl.name = x.name
            LEFT JOIN coloc co ON co.year = x.year AND co.addr = x.addr
            ORDER BY duns, year
        ) TO '{XWALK_OUT}' (FORMAT parquet, COMPRESSION zstd,
                            ROW_GROUP_SIZE 1000000)
    """)
    log("xwalk written")
    for t in ("coloc", "famlink", "namelink"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    # 99.3% of establishment-years are single-line (collapse rate 1.0072):
    # those become panel rows by pure projection; only multi-line
    # establishment-years go through real aggregation.
    con.execute(f"""
        CREATE TEMP TABLE member AS
        SELECT x.*, pr.primary_duns, s.sic8, n2.naics,
               g.n_grp
        FROM xw x
        JOIN prim pr ON pr.estab_id = x.estab_id
        JOIN (SELECT estab_id, year, count(*) AS n_grp
              FROM xw GROUP BY 1, 2) g
          ON g.estab_id = x.estab_id AND g.year = x.year
        LEFT JOIN read_parquet('{nc.LONG}/sic_long.parquet') s
          ON s.duns = x.duns AND s.year = x.year
        LEFT JOIN read_parquet('{nc.LONG}/naics_long.parquet') n2
          ON n2.duns = x.duns AND n2.year = x.year
    """)
    con.execute("DROP TABLE xw")
    con.execute("""
        CREATE TEMP TABLE multi AS
        SELECT * FROM member WHERE n_grp > 1
    """)
    con.execute("""
        CREATE TEMP TABLE multi_rep AS
        SELECT estab_id, year,
               r.duns AS rep_duns, r.emp AS rep_emp, r.sic8 AS rep_sic8,
               r.naics AS rep_naics, r.addr AS rep_addr, r.fips AS rep_fips
        FROM (
            SELECT estab_id, year,
                   arg_max(struct_pack(duns := duns, emp := emp,
                           sic8 := sic8, naics := naics, addr := addr,
                           fips := fips),
                       struct_pack(a := (duns = primary_duns)::INT,
                                   b := emp,
                                   c := -CAST(duns AS BIGINT))) AS r
            FROM multi GROUP BY 1, 2
        )
    """)
    con.execute(f"""
        COPY (
            SELECT m.estab_id, m.year, 1 AS n_lines,
                   m.emp AS emp_sum_exhq, 0 AS emp_hq, m.emp AS emp_primary,
                   (m.emp_code <> 0)::DOUBLE AS share_emp_imputed,
                   m.sic8, m.naics, []::VARCHAR[] AS sic8_secondary,
                   m.addr AS address_norm, m.fips,
                   m.emp_code = 0 AS any_actual_emp,
                   m.emp < 5 AS small_estab,
                   coalesce(m.sic8 LIKE '82%', false) AS sic_education,
                   m.blacklisted, m.quarantined
            FROM member m WHERE m.n_grp = 1
            UNION ALL
            SELECT m.estab_id, m.year, count(*) AS n_lines,
                   coalesce(sum(m.emp) FILTER (WHERE NOT m.is_hq_row), 0)
                       AS emp_sum_exhq,
                   coalesce(sum(m.emp) FILTER (WHERE m.is_hq_row), 0)
                       AS emp_hq,
                   any_value(r.rep_emp) AS emp_primary,
                   sum(m.emp * (m.emp_code <> 0)::INT)
                       / nullif(sum(m.emp), 0) AS share_emp_imputed,
                   any_value(r.rep_sic8) AS sic8,
                   any_value(r.rep_naics) AS naics,
                   list_distinct(list(m.sic8)
                       FILTER (WHERE m.duns <> r.rep_duns
                               AND m.sic8 IS NOT NULL)) AS sic8_secondary,
                   any_value(r.rep_addr) AS address_norm,
                   any_value(r.rep_fips) AS fips,
                   bool_or(m.emp_code = 0) AS any_actual_emp,
                   (coalesce(sum(m.emp) FILTER (WHERE NOT m.is_hq_row), 0)
                    + coalesce(sum(m.emp) FILTER (WHERE m.is_hq_row), 0)) < 5
                       AS small_estab,
                   coalesce(any_value(r.rep_sic8) LIKE '82%', false)
                       AS sic_education,
                   bool_or(m.blacklisted) AS blacklisted,
                   bool_or(m.quarantined) AS quarantined
            FROM multi m
            JOIN multi_rep r ON r.estab_id = m.estab_id AND r.year = m.year
            GROUP BY m.estab_id, m.year
            ORDER BY estab_id, year
        ) TO '{PANEL_OUT}' (FORMAT parquet, COMPRESSION zstd,
                            ROW_GROUP_SIZE 1000000)
    """)
    xrows = con.execute(
        f"SELECT count(*) FROM read_parquet('{XWALK_OUT}')").fetchone()[0]
    urows = qa["universe_rows"]
    if xrows != urows:
        raise RuntimeError(
            f"GATE FAILED: xwalk rows {xrows} != universe {urows}")
    prows = con.execute(
        f"SELECT count(*) FROM read_parquet('{PANEL_OUT}')").fetchone()[0]
    qa["xwalk_rows"] = xrows
    qa["panel_rows"] = prows
    log(f"panel: {prows:,} estab-years; xwalk {xrows:,} rows reconciled")


def final_gates(con, qa):
    d, e, rate = collapse_stats(con, "comp_full")
    qa["collapse_rate"] = rate
    log(f"collapse rate: {rate:.4f} duns per establishment-year")
    if not (COLLAPSE_BAND[0] <= rate <= COLLAPSE_BAND[1]):
        raise RuntimeError(
            f"GATE FAILED: collapse rate {rate:.4f} outside {COLLAPSE_BAND}")
    con.execute(f"""
        COPY (
            SELECT year, comp, addr, count(*) AS n,
                   count(DISTINCT name) AS names,
                   count(*) - count(DISTINCT fam) AS fam_shared
            FROM comp_full WHERE addr IS NOT NULL
            GROUP BY 1, 2, 3 ORDER BY n DESC LIMIT 10
        ) TO '{nc.REPORTS}/largest_components.csv' (FORMAT csv, HEADER)
    """)
    bad = con.execute(f"""
        SELECT count(*) FROM read_csv('{nc.REPORTS}/largest_components.csv')
        WHERE names > 3 AND fam_shared = 0
    """).fetchone()[0]
    qa["top10_name_mix_violations"] = bad
    if bad:
        raise RuntimeError(
            f"GATE FAILED: {bad} of the 10 largest components mix >3 names "
            f"with zero shared family; see _reports/largest_components.csv")
    con.execute(f"""
        COPY (
            SELECT n AS component_size, count(*) AS n_components
            FROM (SELECT year, comp, count(*) AS n FROM comp_full
                  GROUP BY 1, 2)
            GROUP BY 1 ORDER BY 1
        ) TO '{nc.REPORTS}/component_sizes.csv' (FORMAT csv, HEADER)
    """)


def main():
    t0 = time.time()
    con = duckdb.connect()
    nc.duck_settings(con, memory_limit="100GB")
    qa = {}
    build_base(con, qa)
    hq_employment_test(con, qa)
    strip = suite_diagnostic(con, qa)
    qa["strip_suite_chosen"] = strip
    group_one(con, strip, None, "full")
    final_gates(con, qa)
    persistent_ids(con, qa)
    primaries_and_outputs(con, qa)
    (nc.REPORTS / "manifest_66.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True, default=str))
    log(f"PRODUCER 66 COMPLETE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
