"""NETS Fed2025 cleaning, producer 65: time-varying address spells.

Builds one address spell per DUNS per residence period from addressfirst
(first address), the move table (dated relocations, destination fields), and
company (current address), clamped to [max(FirstYear,1990),
min(LastYear,2023)] from misc. Raw address strings are stored; normalization
(nets_common) is used only for comparisons.

Outputs:
  estab/address_spells.parquet  duns(9-char), spell_id, year_start, year_end,
                                address, city, state, zip, fips, lat, lon,
                                source ('first'|'first_from_move'|'move'|
                                'implied_current'), firstyear_conflict,
                                possible_regeocode
  estab/address_spell_map.parquet  duns, year, spell_id (implied_current
                                   spells carry NULL years and are not mapped)
  _reports/                     boundary-convention test, county-mismatch and
                                implied-move rates, sampled movers, manifest

Rules from the approved plan:
- Boundary convention: the establishment sits at Origin through MoveYear and
  occupies Dest from MoveYear+1. Chosen empirically: over 2,824,691
  cross-county moves, fips_long matches the origin county at MoveYear in
  99.97% of cases (2,823,722 vs 483), consistent with NETS's January-snapshot
  timing. The test re-runs each execution and gates on the chosen convention
  not being the worse one.
- Same-year moves: the winner is the move whose normalized Dest matches the
  next known address (next move's Origin, else company current), then
  min(source_row_id); losers are counted and dropped.
- Missing addressfirst: spell 1 = earliest move's Origin. FirstYear > first
  MoveYear: spell 1 starts at MoveYear (the origin's last year under the
  boundary convention), flag firstyear_conflict.
- Last spell vs company current address (alive DUNS only): mismatches append
  an implied_current spell with NULL years, never overwriting history; the
  implied rate among alive DUNS gates at <= 15%.
- County validation vs fips_long after harmonizing Dade->Miami-Dade
  (12025->12086) and Shannon->Oglala Lakota (46113->46102); Broomfield-family
  counties (08014 and parents) and Connecticut from 2022 are excluded from
  the mismatch statistic. Gate: mismatch <= 5% of comparable spell-years.
- Moves under 1 mile with unchanged county: possible_regeocode flag.
- Machine gate: every DUNS's spell sequence is contiguous (year_start =
  previous year_end + 1), asserted over all spells. Five seeded sample movers
  are exported for human review, and the origin(k+1)-vs-dest(k) string
  agreement rate over all consecutive move pairs is reported (a data property
  of the move table's address renderings, not a construction check).
"""

import json
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nets_common as nc

IMPLIED_MOVE_GATE = 0.15
COUNTY_MISMATCH_GATE = 0.05
SPELLS_OUT = nc.ESTAB / "address_spells.parquet"
MAP_OUT = nc.ESTAB / "address_spell_map.parquet"

NORM = nc.sql_normalize_address("{col}", strip_suite=False)


def norm(col):
    return NORM.format(col=col)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs(con):
    con.execute(f"""
        CREATE TEMP TABLE life AS
        SELECT CAST(DunsNumber AS UINTEGER) AS duns,
               greatest(CAST(FirstYear AS INTEGER), 1990) AS fy,
               least(CAST(LastYear AS INTEGER), 2023) AS ly
        FROM read_parquet('{nc.LONG}/misc_dim.parquet')
        WHERE FirstYear IS NOT NULL AND LastYear IS NOT NULL
          AND greatest(CAST(FirstYear AS INTEGER), 1990)
              <= least(CAST(LastYear AS INTEGER), 2023)
    """)
    con.execute(f"""
        CREATE TEMP TABLE af AS
        SELECT CAST(DunsNumber AS UINTEGER) AS duns,
               Address_First AS address, City_First AS city,
               State_First AS state, ZipCode_First AS zip,
               FipsCounty_First AS fips
        FROM read_parquet('{nc.RAW}/addressfirst.parquet')
    """)
    con.execute(f"""
        CREATE TEMP TABLE cur AS
        SELECT CAST(DunsNumber AS UINTEGER) AS duns, Address AS address,
               City AS city, State AS state, ZipCode AS zip
        FROM read_parquet('{nc.RAW}/company.parquet')
    """)
    cols = {r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{nc.LONG}/move_rowid.parquet')"
    ).fetchall()}
    lat = "DestLatitude" if "DestLatitude" in cols else "NULL"
    lon = "DestLongitude" if "DestLongitude" in cols else "NULL"
    dist = "Distance" if "Distance" in cols else "NULL"
    con.execute(f"""
        CREATE TEMP TABLE mv AS
        SELECT source_row_id, CAST(DunsNumber AS UINTEGER) AS duns,
               CAST(MoveYear AS INTEGER) AS myear,
               OriginAddress, OriginCity, OriginState, OriginZIP,
               OriginFIPSCounty,
               DestAddress, DestCity, DestState, DestZIP, DestFIPSCounty,
               {lat} AS dlat, {lon} AS dlon, {dist} AS dist
        FROM read_parquet('{nc.LONG}/move_rowid.parquet') m
        JOIN life l ON CAST(m.DunsNumber AS UINTEGER) = l.duns
        WHERE CAST(MoveYear AS INTEGER) BETWEEN l.fy AND l.ly
    """)
    n = con.execute("SELECT count(*) FROM mv").fetchone()[0]
    log(f"inputs loaded: {n:,} in-life moves")
    return n


def boundary_convention_test(con, qa):
    a_right, b_right, n = con.execute(f"""
        SELECT
            count(*) FILTER (WHERE f.fips = rtrim(m.DestFIPSCounty)),
            count(*) FILTER (WHERE f.fips = rtrim(m.OriginFIPSCounty)),
            count(*)
        FROM mv m
        JOIN read_parquet('{nc.LONG}/fips_long.parquet') f
          ON f.duns = m.duns AND f.year = m.myear
        WHERE m.DestFIPSCounty IS NOT NULL AND m.DestFIPSCounty <> ''
          AND m.OriginFIPSCounty IS NOT NULL AND m.OriginFIPSCounty <> ''
          AND rtrim(m.DestFIPSCounty) <> rtrim(m.OriginFIPSCounty)
    """).fetchone()
    qa["boundary_test"] = {"cross_county_moves": n,
                           "dest_at_moveyear_right": a_right,
                           "origin_at_moveyear_right": b_right}
    log(f"boundary test on {n:,} cross-county moves: dest-from-Y right "
        f"{a_right:,}, origin-at-Y right {b_right:,}")
    if n > 1000 and b_right < a_right:
        raise RuntimeError(
            "GATE FAILED: convention 'Origin through MoveYear, Dest from "
            "MoveYear+1' scores worse than the alternative; review before "
            "building spells.")


def pick_same_year_winners(con, qa):
    con.execute(f"""
        CREATE TEMP TABLE mv1 AS
        WITH nxt AS (
            SELECT m.*,
                   lead({norm('m.OriginAddress')}) OVER w AS next_origin_norm,
                   count(*) OVER (PARTITION BY m.duns, m.myear) AS k
            FROM mv m
            WINDOW w AS (PARTITION BY m.duns ORDER BY m.myear, m.source_row_id)
        ),
        scored AS (
            SELECT *,
                   ({norm('DestAddress')} =
                    coalesce(next_origin_norm,
                             (SELECT {norm('c.address')} FROM cur c
                              WHERE c.duns = nxt.duns))) AS dest_matches_next
            FROM nxt
        ),
        ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY duns, myear
                ORDER BY (dest_matches_next IS NOT TRUE), source_row_id
            ) AS rk
            FROM scored
        )
        SELECT * EXCLUDE (rk, k, next_origin_norm, dest_matches_next)
        FROM ranked WHERE rk = 1
    """)
    total, kept = con.execute(
        "SELECT (SELECT count(*) FROM mv), count(*) FROM mv1").fetchone()
    qa["same_year_losers_dropped"] = total - kept
    log(f"same-year winners: kept {kept:,}, dropped {total - kept:,}")


def build_spells(con, qa):
    con.execute(f"""
        CREATE TEMP TABLE events AS
        WITH firstmove AS (
            SELECT duns, min(myear) AS m0 FROM mv1 GROUP BY duns
        ),
        e0 AS (
            SELECT l.duns,
                   CASE WHEN fm.m0 IS NOT NULL AND l.fy > fm.m0
                        THEN fm.m0 ELSE l.fy END AS eyear,
                   CASE WHEN af.address IS NOT NULL AND rtrim(af.address) <> ''
                        THEN af.address ELSE m1.OriginAddress END AS address,
                   CASE WHEN af.address IS NOT NULL AND rtrim(af.address) <> ''
                        THEN af.city ELSE m1.OriginCity END AS city,
                   CASE WHEN af.address IS NOT NULL AND rtrim(af.address) <> ''
                        THEN af.state ELSE m1.OriginState END AS state,
                   CASE WHEN af.address IS NOT NULL AND rtrim(af.address) <> ''
                        THEN af.zip ELSE m1.OriginZIP END AS zip,
                   CASE WHEN af.address IS NOT NULL AND rtrim(af.address) <> ''
                        THEN af.fips ELSE m1.OriginFIPSCounty END AS fips,
                   NULL::DOUBLE AS lat, NULL::DOUBLE AS lon,
                   CASE WHEN af.address IS NOT NULL AND rtrim(af.address) <> ''
                        THEN 'first' ELSE 'first_from_move' END AS source,
                   (fm.m0 IS NOT NULL AND l.fy > fm.m0) AS firstyear_conflict,
                   false AS possible_regeocode
            FROM life l
            LEFT JOIN af ON af.duns = l.duns
            LEFT JOIN firstmove fm ON fm.duns = l.duns
            LEFT JOIN mv1 m1 ON m1.duns = l.duns AND m1.myear = fm.m0
        )
        SELECT * FROM e0 WHERE address IS NOT NULL AND rtrim(address) <> ''
        UNION ALL
        SELECT duns, myear + 1 AS eyear, DestAddress, DestCity, DestState,
               DestZIP, DestFIPSCounty, dlat, dlon, 'move',
               false,
               (TRY_CAST(dist AS DOUBLE) < 1.0
                AND rtrim(OriginFIPSCounty) = rtrim(DestFIPSCounty))
        FROM mv1
    """)
    con.execute(f"""
        CREATE TEMP TABLE spells AS
        SELECT e.duns,
               row_number() OVER (PARTITION BY e.duns ORDER BY e.eyear,
                                  (e.source LIKE 'first%') DESC) AS spell_id,
               e.eyear AS year_start_raw,
               coalesce(lead(e.eyear) OVER (PARTITION BY e.duns
                        ORDER BY e.eyear, (e.source LIKE 'first%') DESC) - 1,
                        l.ly) AS year_end,
               e.* EXCLUDE (duns, eyear)
        FROM events e
        JOIN life l ON l.duns = e.duns
    """)
    dropped = con.execute("""
        SELECT count(*) FROM spells WHERE year_end < year_start_raw
    """).fetchone()[0]
    con.execute("DELETE FROM spells WHERE year_end < year_start_raw")
    qa["zero_length_spells_dropped"] = dropped
    n = con.execute("SELECT count(*) FROM spells").fetchone()[0]
    qa["spells"] = n
    log(f"spells: {n:,} rows ({dropped:,} zero-length dropped)")


def implied_current(con, qa):
    con.execute(f"""
        CREATE TEMP TABLE last_spell AS
        SELECT s.* FROM spells s
        JOIN (SELECT duns, max(spell_id) AS m FROM spells GROUP BY duns) t
          ON t.duns = s.duns AND t.m = s.spell_id
    """)
    con.execute(f"""
        CREATE TEMP TABLE implied AS
        SELECT ls.duns, ls.spell_id + 1 AS spell_id,
               NULL::INTEGER AS year_start_raw, NULL::INTEGER AS year_end,
               c.address, c.city, c.state, c.zip,
               NULL::VARCHAR AS fips, NULL::DOUBLE AS lat, NULL::DOUBLE AS lon,
               'implied_current' AS source, false AS firstyear_conflict,
               false AS possible_regeocode
        FROM last_spell ls
        JOIN life l ON l.duns = ls.duns AND l.ly = 2023
        JOIN cur c ON c.duns = ls.duns
        WHERE c.address IS NOT NULL AND rtrim(c.address) <> ''
          AND {norm('ls.address')} <> {norm('c.address')}
    """)
    alive = con.execute(
        "SELECT count(*) FROM life WHERE ly = 2023").fetchone()[0]
    n_implied = con.execute("SELECT count(*) FROM implied").fetchone()[0]
    rate = n_implied / alive if alive else 0
    qa["implied_current"] = {"alive_duns": alive, "implied": n_implied,
                             "rate": rate}
    log(f"implied current-address spells: {n_implied:,} of {alive:,} alive "
        f"({rate:.2%})")
    if rate > IMPLIED_MOVE_GATE:
        raise RuntimeError(
            f"GATE FAILED: implied_move rate {rate:.2%} exceeds "
            f"{IMPLIED_MOVE_GATE:.0%}; review address normalization before "
            f"proceeding.")


def write_outputs(con, qa):
    con.execute(f"""
        COPY (
            SELECT lpad(CAST(duns AS VARCHAR), 9, '0') AS duns, spell_id,
                   year_start_raw AS year_start, year_end,
                   rtrim(address) AS address, rtrim(city) AS city,
                   rtrim(state) AS state, rtrim(zip) AS zip,
                   rtrim(fips) AS fips, lat, lon, source,
                   firstyear_conflict, possible_regeocode
            FROM (SELECT * FROM spells UNION ALL SELECT * FROM implied)
            ORDER BY duns, spell_id
        ) TO '{SPELLS_OUT}' (FORMAT parquet, COMPRESSION zstd,
                             ROW_GROUP_SIZE 1000000)
    """)
    con.execute(f"""
        COPY (
            SELECT lpad(CAST(duns AS VARCHAR), 9, '0') AS duns,
                   CAST(y.year AS SMALLINT) AS year, spell_id
            FROM spells s,
                 LATERAL unnest(range(greatest(s.year_start_raw, 1990),
                                      s.year_end + 1)) AS y(year)
            ORDER BY duns, year
        ) TO '{MAP_OUT}' (FORMAT parquet, COMPRESSION zstd,
                          ROW_GROUP_SIZE 1000000)
    """)
    rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{MAP_OUT}')").fetchone()[0]
    qa["spell_map_rows"] = rows
    log(f"outputs written: spell map {rows:,} duns-years")


def county_validation(con, qa):
    con.execute("""
        CREATE TEMP TABLE harmon AS
        SELECT * FROM (VALUES ('12025', '12086'), ('46113', '46102'))
            AS t(old_fips, new_fips)
    """)
    con.execute("""
        CREATE TEMP TABLE exploded AS
        SELECT s.duns, y.year::SMALLINT AS year, rtrim(s.fips) AS sf
        FROM spells s,
             LATERAL unnest(range(greatest(s.year_start_raw, 1990),
                                  s.year_end + 1)) AS y(year)
        WHERE s.fips IS NOT NULL AND rtrim(s.fips) <> ''
    """)
    match, mismatch = con.execute(f"""
        WITH m AS (
            SELECT e.duns, e.year,
                   coalesce(h.new_fips, e.sf) AS spell_fips
            FROM exploded e
            LEFT JOIN harmon h ON h.old_fips = e.sf
        )
        SELECT
            count(*) FILTER (WHERE m.spell_fips =
                coalesce(h2.new_fips, f.fips)),
            count(*) FILTER (WHERE m.spell_fips <>
                coalesce(h2.new_fips, f.fips))
        FROM m
        JOIN read_parquet('{nc.LONG}/fips_long.parquet') f
          ON f.duns = CAST(m.duns AS UINTEGER) AND f.year = m.year
        LEFT JOIN harmon h2 ON h2.old_fips = f.fips
        WHERE f.fips <> '' AND substr(f.fips, 1, 2) <> '09'
          AND substr(m.spell_fips, 1, 2) <> '09'
          AND m.spell_fips NOT IN ('08014', '08013', '08001', '08059', '08123')
          AND f.fips NOT IN ('08014', '08013', '08001', '08059', '08123')
    """).fetchone()
    total = match + mismatch
    rate = mismatch / total if total else 0
    qa["county_validation"] = {
        "comparable_duns_years": total, "mismatch": mismatch, "rate": rate,
        "excluded": "CT (2022 recode) and Broomfield-family counties"}
    log(f"county validation: {rate:.3%} mismatch over {total:,} duns-years")
    if rate > COUNTY_MISMATCH_GATE:
        raise RuntimeError(
            f"GATE FAILED: county mismatch {rate:.2%} exceeds "
            f"{COUNTY_MISMATCH_GATE:.0%}.")


def sample_movers(con, qa):
    con.execute("""
        CREATE TEMP TABLE sample5 AS
        SELECT duns FROM (SELECT DISTINCT duns FROM mv1)
        USING SAMPLE reservoir(5 ROWS) REPEATABLE (42)
    """)
    con.execute(f"""
        COPY (
            SELECT s.* FROM spells s JOIN sample5 USING (duns)
            ORDER BY s.duns, s.spell_id
        ) TO '{nc.REPORTS}/sample_movers.csv' (FORMAT csv, HEADER)
    """)
    # Machine gate: gap-free spell sequences are a construction invariant and
    # must hold for every DUNS, not a sample.
    gaps = con.execute("""
        WITH seq AS (
            SELECT duns, year_start_raw,
                   lag(year_end) OVER (PARTITION BY duns ORDER BY spell_id)
                       AS prev_end
            FROM spells
        )
        SELECT count(*) FROM seq
        WHERE prev_end IS NOT NULL AND year_start_raw <> prev_end + 1
    """).fetchone()[0]
    qa["spell_gap_violations"] = gaps
    # Origin(k+1) vs dest(k) string agreement is a data property of the move
    # table, not a construction check: report the rate over all consecutive
    # move pairs (raw strings chain exactly for verified cases; variants are
    # NETS's own renderings).
    breaks, pairs = con.execute(f"""
        WITH seq AS (
            SELECT duns, {norm('OriginAddress')} AS o,
                   lag({norm('DestAddress')}) OVER (
                       PARTITION BY duns ORDER BY myear, source_row_id
                   ) AS prev_d
            FROM mv1
        )
        SELECT count(*) FILTER (WHERE o <> prev_d), count(*)
        FROM seq WHERE prev_d IS NOT NULL
    """).fetchone()
    qa["move_chain_string_breaks"] = {
        "consecutive_pairs": pairs, "breaks": breaks,
        "rate": breaks / pairs if pairs else 0}
    log(f"spell gaps: {gaps}; move-chain string breaks: {breaks:,} of "
        f"{pairs:,} pairs")
    if gaps:
        raise RuntimeError(
            f"GATE FAILED: {gaps} non-contiguous spell sequences; the spell "
            f"construction is broken.")


def main():
    t0 = time.time()
    con = duckdb.connect()
    nc.duck_settings(con)
    qa = {}
    load_inputs(con)
    boundary_convention_test(con, qa)
    pick_same_year_winners(con, qa)
    build_spells(con, qa)
    implied_current(con, qa)
    write_outputs(con, qa)
    county_validation(con, qa)
    sample_movers(con, qa)
    (nc.REPORTS / "manifest_65.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True))
    log(f"PRODUCER 65 COMPLETE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
