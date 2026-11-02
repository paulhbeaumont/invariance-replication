"""NETS Fed2025 cleaning, producer 64: ultimate-parent linkage.

Semantics verified against Crane and Decker (2019), "Business Dynamics in the
National Establishment Time Series (NETS)", pp. 11-12: HQDuns is a one-level
pointer to the headquarters establishment's DUNS, and NETS allows multiple
levels of headquarters, so establishments are united "under a single firm
identifier by 'rolling up' hqduns identifiers ... the hqduns of the highest
level headquarters, which necessarily reports itself as its own hqduns (or, in
rare cases, reports an hqduns that does not appear as a dunsnumber anywhere
else)". Cyclical linkages exist but are "extremely rare" (their footnote 6).

Input:  long/hqduns_long.parquet (duns, year, hqduns; sentinels and
        self-pointers already dropped by producer 63), long/misc_dim.parquet.
Output: firm/ultimate_parent.parquet with columns
        duns, year, ult_duns, chain_depth, converged, cycle, into_cycle,
        parent_in_panel, bridged_pointer, smoothed, ult_duns_unsmoothed.
        Only DUNS-years with a parent are emitted; an absent (duns, year)
        means ult = self.

Rules from the approved plan:
- Pointer gaps: a (duns, year) hole strictly inside the duns's own pointer
  history is bridged from t-1 (else t+1), flag bridged_pointer.
- Iterate parent-of-parent to a fixed point via active/final table splits
  (a row retires when its current ultimate has no further parent), cap 15;
  rows still active at the cap are unconverged and go to a Python chain walk:
  cycle members share ult = min(DUNS) in the cycle (cycle=true); rows
  pointing into a cycle inherit that ult (into_cycle=true).
- parent_in_panel=false when ult_duns never appears as a DunsNumber in misc.
- Smoothing acts post-iteration on the resolved ultimate, only where explicit
  rows exist at t-1, t, t+1 and ult(t-1)=ult(t+1)<>ult(t); the canonical
  column ult_duns is smoothed, ult_duns_unsmoothed is filled only on flagged
  rows. Self-blips (no explicit neighbor rows) are deliberately not smoothed.
- Blocking gate: chain_depth >= 2 must hold for at least 0.5% of emitted
  DUNS-years, else halt: a lower share would be consistent with HQDuns
  already being the ultimate pointer, which this producer cannot distinguish
  from shallow trees.
"""

import json
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nets_common as nc

CAP = 15
LOOP_MATTERED_FLOOR = 0.005
OUT = nc.FIRM / "ultimate_parent.parquet"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_bridged_parents(con, qa):
    con.execute(f"""
        CREATE TEMP TABLE p AS
        WITH p_raw AS (
            SELECT duns, year, hqduns
            FROM read_parquet('{nc.LONG}/hqduns_long.parquet')
        ),
        spans AS (
            SELECT duns, min(year) AS y0, max(year) AS y1
            FROM p_raw GROUP BY duns
        ),
        gaps AS (
            SELECT s.duns, y.year
            FROM spans s
            JOIN (SELECT DISTINCT year FROM p_raw) y
              ON y.year > s.y0 AND y.year < s.y1
            ANTI JOIN p_raw r ON r.duns = s.duns AND r.year = y.year
        )
        SELECT duns, year, hqduns, false AS bridged_pointer FROM p_raw
        UNION ALL
        SELECT g.duns, g.year,
               coalesce(prev.hqduns, next.hqduns) AS hqduns,
               true AS bridged_pointer
        FROM gaps g
        LEFT JOIN p_raw prev ON prev.duns = g.duns AND prev.year = g.year - 1
        LEFT JOIN p_raw next ON next.duns = g.duns AND next.year = g.year + 1
        WHERE coalesce(prev.hqduns, next.hqduns) IS NOT NULL
    """)
    n_raw, n_bridged = con.execute(
        "SELECT count(*) FILTER (WHERE NOT bridged_pointer), "
        "count(*) FILTER (WHERE bridged_pointer) FROM p").fetchone()
    qa["parent_rows"] = n_raw
    qa["bridged_rows"] = n_bridged
    log(f"parent map: {n_raw:,} raw rows, {n_bridged:,} bridged gap-years")


def iterate_rollup(con, qa):
    """Active/final split: a row retires when its ultimate has no parent."""
    con.execute("""
        CREATE TEMP TABLE active AS
        SELECT duns, year, hqduns AS ult, 1 AS depth, bridged_pointer FROM p
    """)
    con.execute("CREATE TEMP TABLE final AS SELECT * FROM active LIMIT 0")
    for i in range(1, CAP + 1):
        con.execute("""
            CREATE TEMP TABLE step AS
            SELECT a.duns, a.year,
                   q.hqduns AS next_ult, a.ult, a.depth, a.bridged_pointer
            FROM active a
            LEFT JOIN p q ON q.duns = a.ult AND q.year = a.year
        """)
        con.execute("""
            INSERT INTO final
            SELECT duns, year, ult, depth, bridged_pointer
            FROM step WHERE next_ult IS NULL OR next_ult = ult
        """)
        con.execute("DROP TABLE active")
        con.execute("""
            CREATE TEMP TABLE active AS
            SELECT duns, year, next_ult AS ult, depth + 1 AS depth,
                   bridged_pointer
            FROM step WHERE next_ult IS NOT NULL AND next_ult <> ult
        """)
        con.execute("DROP TABLE step")
        still = con.execute("SELECT count(*) FROM active").fetchone()[0]
        log(f"iteration {i}: {still:,} rows still walking")
        if still == 0:
            break
    unconverged = con.execute("SELECT count(*) FROM active").fetchone()[0]
    qa["iterations"] = i
    qa["unconverged_at_cap"] = unconverged
    return unconverged


def resolve_cycles(con, qa):
    """Python walk over the residual subgraph of rows still active at the cap.
    The cycle_fix table is created (empty) in main before this runs."""
    years = [r[0] for r in con.execute(
        "SELECT DISTINCT year FROM active").fetchall()]
    n_cycle = n_into = 0
    for year in years:
        seeds = {r[0]: (r[1], r[2]) for r in con.execute(
            "SELECT duns, depth, bridged_pointer FROM active WHERE year = ?",
            [year]).fetchall()}
        nodes = set(seeds)
        frontier = set(seeds)
        parent = {}
        while frontier:
            in_list = ",".join(str(d) for d in frontier)
            rows = con.execute(
                f"SELECT duns, hqduns FROM p WHERE year = {year} "
                f"AND duns IN ({in_list})").fetchall()
            frontier = set()
            for d, h in rows:
                parent[d] = h
                if h not in nodes:
                    nodes.add(h)
                    frontier.add(h)
        for seed, (depth, bridged) in seeds.items():
            path, seen = [], {}
            d = seed
            while d in parent and d not in seen:
                seen[d] = len(path)
                path.append(d)
                d = parent[d]
            if d in seen:
                cyc = set(path[seen[d]:])
                ult = min(cyc)
                is_cycle = seed in cyc
            else:
                cyc = set()
                ult = d
                is_cycle = False
            n_cycle += is_cycle
            n_into += (not is_cycle) and bool(cyc)
            con.execute(
                "INSERT INTO cycle_fix VALUES (?, ?, ?, ?, ?, ?, ?)",
                [seed, year, ult, depth, bridged, is_cycle,
                 (not is_cycle) and bool(cyc)])
    qa["cycle_members"] = n_cycle
    qa["into_cycle"] = n_into
    log(f"cycle resolution: {n_cycle} in-cycle seeds, {n_into} pointing in")


def finalize(con, qa):
    con.execute(f"""
        CREATE TEMP TABLE resolved AS
        SELECT duns, year, ult, depth, true AS converged,
               false AS cycle, false AS into_cycle, bridged_pointer
        FROM final
        UNION ALL
        SELECT duns, year, ult, depth, false AS converged,
               is_cycle AS cycle, is_into_cycle AS into_cycle, bridged_pointer
        FROM cycle_fix
    """)
    con.execute(f"""
        CREATE TEMP TABLE smoothed AS
        SELECT r.*,
               (prev.ult IS NOT NULL AND prev.ult = next.ult
                AND prev.ult <> r.ult) AS smoothed,
               CASE WHEN prev.ult IS NOT NULL AND prev.ult = next.ult
                         AND prev.ult <> r.ult THEN r.ult END
                   AS ult_unsmoothed,
               CASE WHEN prev.ult IS NOT NULL AND prev.ult = next.ult
                         AND prev.ult <> r.ult THEN prev.ult
                    ELSE r.ult END AS ult_final
        FROM resolved r
        LEFT JOIN resolved prev ON prev.duns = r.duns AND prev.year = r.year - 1
        LEFT JOIN resolved next ON next.duns = r.duns AND next.year = r.year + 1
    """)
    con.execute(f"""
        CREATE TEMP TABLE duns_in_misc AS
        SELECT DISTINCT CAST(DunsNumber AS UINTEGER) AS d
        FROM read_parquet('{nc.LONG}/misc_dim.parquet')
    """)
    con.execute(f"""
        COPY (
            SELECT lpad(CAST(s.duns AS VARCHAR), 9, '0') AS duns, s.year,
                   lpad(CAST(s.ult_final AS VARCHAR), 9, '0') AS ult_duns,
                   s.depth AS chain_depth, s.converged, s.cycle, s.into_cycle,
                   s.bridged_pointer, s.smoothed,
                   lpad(CAST(s.ult_unsmoothed AS VARCHAR), 9, '0')
                       AS ult_duns_unsmoothed,
                   m.d IS NOT NULL AS parent_in_panel
            FROM smoothed s
            LEFT JOIN duns_in_misc m ON m.d = s.ult_final
            ORDER BY s.duns, s.year
        ) TO '{OUT}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)
    """)
    rows, depth2, smoothed_n, dangling = con.execute(f"""
        SELECT count(*),
               count(*) FILTER (WHERE chain_depth >= 2),
               count(*) FILTER (WHERE smoothed),
               count(*) FILTER (WHERE NOT parent_in_panel)
        FROM read_parquet('{OUT}')
    """).fetchone()
    qa["output_rows"] = rows
    qa["loop_mattered_share"] = depth2 / rows if rows else 0
    qa["smoothed_rows"] = smoothed_n
    qa["dangling_parent_rows"] = dangling
    con.execute(f"""
        COPY (SELECT chain_depth, count(*) AS n
              FROM read_parquet('{OUT}') GROUP BY 1 ORDER BY 1)
        TO '{nc.REPORTS}/ult_chain_depth.csv' (FORMAT csv, HEADER)
    """)
    log(f"ultimate_parent: {rows:,} rows, loop-mattered "
        f"{qa['loop_mattered_share']:.4f}, {smoothed_n:,} smoothed, "
        f"{dangling:,} dangling parents")


def cross_validate(con, qa):
    # EstCat is the establishment role (Single Location/Headquarters/Branch);
    # LevelCode is geocode precision (NETS Database Description 2019).
    qa["xval"] = {}
    qa["xval"]["branch_estcat_serving_as_ultimate"] = con.execute(f"""
        SELECT count(DISTINCT h.d) FROM (
            SELECT DISTINCT CAST(ult_duns AS UINTEGER) AS d
            FROM read_parquet('{OUT}')) h
        JOIN read_parquet('{nc.LONG}/misc_dim.parquet') m
          ON CAST(m.DunsNumber AS UINTEGER) = h.d
        WHERE upper(rtrim(m.EstCat)) LIKE '%BRANCH%'
    """).fetchone()[0]
    qa["xval"]["subsidiary_without_any_parent_pointer"] = con.execute(f"""
        SELECT count(*) FROM read_parquet('{nc.LONG}/misc_dim.parquet') m
        ANTI JOIN (SELECT DISTINCT duns FROM p) q
            ON CAST(m.DunsNumber AS UINTEGER) = q.duns
        WHERE upper(rtrim(m.Subsidiary)) LIKE 'Y%'
    """).fetchone()[0]
    log(f"cross-validation: {qa['xval']}")


def main():
    t0 = time.time()
    con = duckdb.connect()
    nc.duck_settings(con)
    qa = {}
    build_bridged_parents(con, qa)
    unconverged = iterate_rollup(con, qa)
    con.execute("""
        CREATE TEMP TABLE IF NOT EXISTS cycle_fix (
            duns UINTEGER, year SMALLINT, ult UINTEGER, depth INTEGER,
            bridged_pointer BOOLEAN, is_cycle BOOLEAN, is_into_cycle BOOLEAN)
    """)
    if unconverged:
        resolve_cycles(con, qa)
    finalize(con, qa)
    cross_validate(con, qa)
    (nc.REPORTS / "manifest_64.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True))
    if qa["loop_mattered_share"] < LOOP_MATTERED_FLOOR:
        raise RuntimeError(
            f"GATE FAILED: loop moved beyond one level for only "
            f"{qa['loop_mattered_share']:.4%} of DUNS-years "
            f"(floor {LOOP_MATTERED_FLOOR:.1%}). HQDuns may already be the "
            f"ultimate pointer; halt and review before using this output.")
    log(f"PRODUCER 64 COMPLETE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
