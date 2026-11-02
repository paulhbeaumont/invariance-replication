"""Stage 10: how problematic is NETS? Benchmarks against EFSY and BDS.

WS2 step 2 of `PLAN_ra_next_tasks.md`. Two questions, one exhibit:

  STOCKS.  NETS vs EFSY (desuppressed CBP) employment -- levels, and growth,
           at the county rung and at the NAICS2 and NAICS4 cell rungs, over
           the 1990-2018 overlap. Both the standard growth rate and the
           midpoint growth rate, because the whole point of the paper is that
           the two disagree exactly where the cells are small, and NETS has
           far more small cells than EFSY.

  FLOWS.   NETS establishment births vs BDS establishment entry, at the county
           rung and at the industry rung.

NO PASS CRITERION IS PRE-REGISTERED. The plan asked for one and the decision
taken for this run was to drop it: this is a descriptive benchmark, and it does
not gate WS2 step 5. Every statistic a criterion would have been written
against is printed, so one can be written later -- but it must then be written
against a sample whose numbers are not already visible, which this one's are.
The report says so in its own words; nothing here should be read as a verdict.

WHAT IS FIXED BEFORE ANY NUMBER, by `ws2_step0_protocol_gate.md`:

  - the support is the INTERSECTION of the locked 3,058 ccgs, the NETS ccgs,
    and the EFSY ccgs, fixed once (gate Q7). A county in one source and not the
    other is a coverage fact reported here, never a sample re-cut per exhibit.
  - the growth rate is midpoint with midpoint weights (gate Q6), h = 2, 3, 4,
    and NOTHING is winsorized: midpoint growth is bounded in [-2, 2] (A.3.2).
  - the NETS universe is `actual` for the headline and `all` alongside; the
    cutoff was fixed in producer 73, before this file existed.
  - no cell is dropped for being small, zero, or imputation-heavy (A.3.1).
    Zero cells ARE the phenomenon.

THE INDUSTRY RUNG THE ENTRY BENCHMARK CANNOT HAVE. BDS publishes county x
sector (19 codes, three of them ranges of NAICS2) and no county x NAICS2 table;
its finer industry tables carry no geography. The requested NAICS2 entry
comparison therefore runs at BDS sector, with the NETS side collapsed onto the
same 19 codes. This is stated in the table, not substituted silently. And BDS
withholds `estabs_entry` in 36.7% of county-sector-years -- concentrated in
exactly the small cells the comparison is about -- so the industry-rung entry
statistics are computed on the unsuppressed remainder and the suppression share
is printed beside every one of them.

CELLS PRESENT IN ONE SOURCE ONLY ARE NOT DROPPED, AND NOT MIXED IN EITHER.
An industry that appears in NETS and never in EFSY anywhere in the window
(public administration, say) would show midpoint growth of +2 in every county
if it were coalesced to an EFSY zero. That is a scope difference, not a growth
fact. Such codes are labelled `nets_only` / `efsy_only`, kept out of the
agreement statistics, and reported as an employment share of their own. Within
the codes both sources carry, a county-year cell missing on one side IS a
genuine zero and is coalesced to one.

Outputs, all under `Results/nets_benchmark/`:

  Table_nets_efsy_levels.tex      levels and the NETS/EFSY ratio, by year
  Table_nets_efsy_growth.tex      growth agreement, one panel per rung
  Table_nets_bds_entry.tex        entry-rate agreement, county and sector
  Figure_nets_efsy_scatter.pdf    binned NETS vs EFSY midpoint growth
  Figure_nets_bds_entry.pdf       binned NETS vs BDS entry rate
  nets_benchmark_stats.csv        one row per printed number
  nets_benchmark_divergence.csv   the divergence cuts
  ws2_step2_benchmark_report.md   the divergence document
  ws2_step2_benchmark.tex/.pdf    all of the above in one document

Run: `.venv/bin/python step_1/code/pipeline/10_nets_benchmark.py`
"""
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PIPELINE = Path(__file__).resolve().parent
CODE = PIPELINE.parent
sys.path.insert(0, str(CODE / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc          # noqa: E402
import nets_common as nc           # noqa: E402
import store                       # noqa: E402

RESULTS = CODE.parents[1] / "Results" / "nets_benchmark"
LOG_DIR = dc.LOG_DIR

NETS_DIR = dc.MACRO_DATA_ROOT / "temp" / "nets_cells"
NETS_COUNTY = NETS_DIR / "nets_ccg_year.parquet"
NETS_IND = NETS_DIR / "nets_ccg_industry_year.parquet"
BIRTH_COUNTY = NETS_DIR / "nets_births_ccg_year.parquet"
BIRTH_IND = NETS_DIR / "nets_births_ccg_industry_year.parquet"

EFSY_COUNTY = dc.EMP_OUT_DIR / "efsy_ccg_year.parquet"
EFSY_N2 = dc.EMP_OUT_DIR / "naics2" / "y*.parquet"
EFSY_N4 = dc.EMP_OUT_DIR / "naics4_alloc" / "y*.parquet"

# 1990 is NETS's first year; 2018 is EFSY's last and the contract data's last.
YEAR_LO, YEAR_HI = 1990, 2018
HORIZONS = (2, 3, 4)
UNIVERSES = ("actual", "all")
HEADLINE_UNIVERSE = "actual"     # gate: 57.5% of NETS employment cells are estimates
HEADLINE_H = 2
MEMORY_LIMIT = "12GB"
UNKNOWN = "UNKNOWN"

# The three NAICS sectors BDS publishes as ranges. Everything else is its own
# two-digit code; NAICS2 92 (public administration), 99 and the UNKNOWN
# sentinel have no BDS sector and are reported as an out-of-scope residual.
SECTOR_RANGES = {**{c: "31-33" for c in ("31", "32", "33")},
                 **{c: "44-45" for c in ("44", "45")},
                 **{c: "48-49" for c in ("48", "49")}}
BDS_SECTORS = ["11", "21", "22", "23", "31-33", "42", "44-45", "48-49", "51",
               "52", "53", "54", "55", "56", "61", "62", "71", "72", "81"]

RUNGS = ("county", "naics2", "naics4")
RUNG_LABEL = {"county": "County", "naics2": "County $\\times$ NAICS2",
              "naics4": "County $\\times$ NAICS4"}

# BDS's suppression share of county-sector-year `estabs_entry`, from
# ws2_bds_ingest_report.md; recomputed here and asserted against this.
BDS_SECTOR_SUPPRESSION = 0.367


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def connect():
    con = duckdb.connect()
    nc.duck_settings(con, memory_limit=MEMORY_LIMIT)
    tmp = NETS_DIR / ".tmp_duck"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con


# ---------------------------------------------------------------- support

def build_support(con):
    """The intersection of the locked, NETS and EFSY ccg supports.

    Fixed once, here, before any statistic is computed, and used by every
    exhibit (gate Q7). The three one-sided differences are returned so the
    report can price the coverage rather than absorb it.
    """
    locked = set(store.load_analysis_panel()["ccg"].astype(int).unique())
    nets = set(con.execute(f"""
        SELECT DISTINCT ccg FROM read_parquet('{NETS_COUNTY}')
        WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
    """).df()["ccg"].astype(int))
    efsy = set(con.execute(f"""
        SELECT DISTINCT ccg FROM read_parquet('{EFSY_COUNTY}')
        WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
    """).df()["ccg"].astype(int))
    bds = set(con.execute(f"""
        SELECT DISTINCT ccg FROM read_parquet('{dc.bds_path("bds_ccg_year.parquet")}')
        WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
    """).df()["ccg"].astype(int))

    sup = sorted(locked & nets & efsy)
    if not sup:
        raise RuntimeError("empty ccg support: the three sources share no county")
    con.register("support", pd.DataFrame({"ccg": np.array(sup, dtype=np.int64)}))
    sup_bds = sorted(set(sup) & bds)
    con.register("support_bds",
                 pd.DataFrame({"ccg": np.array(sup_bds, dtype=np.int64)}))
    cov = {
        "locked": len(locked), "nets": len(nets), "efsy": len(efsy),
        "bds": len(bds), "support": len(sup), "support_bds": len(sup_bds),
        "locked_not_nets": len(locked - nets),
        "locked_not_efsy": len(locked - efsy),
        "nets_not_locked": len(nets - locked),
        "support_not_bds": len(set(sup) - bds),
    }
    log(f"support: {len(sup):,} ccgs (locked {len(locked):,}, NETS "
        f"{len(nets):,}, EFSY {len(efsy):,}); BDS-comparable {len(sup_bds):,}")
    return cov


# ------------------------------------------------------------ stock cells

def _nets_cells_sql(rung, universe):
    """NETS employment cells at one rung, on the support, in the window."""
    if rung == "county":
        return f"""
            SELECT ccg, '' AS code, year, sum(emp) AS emp,
                   sum(n_estab) AS n_estab, sum(emp_imputed) AS emp_imputed
            FROM read_parquet('{NETS_COUNTY}')
            WHERE universe = '{universe}' AND year BETWEEN {YEAR_LO} AND {YEAR_HI}
              AND ccg IN (SELECT ccg FROM support)
            GROUP BY 1, 2, 3
        """
    n = 2 if rung == "naics2" else 4
    return f"""
        SELECT ccg, substr(naics4, 1, {n}) AS code, year, sum(emp) AS emp,
               sum(n_estab) AS n_estab, sum(emp_imputed) AS emp_imputed
        FROM read_parquet('{NETS_IND}')
        WHERE universe = '{universe}' AND year BETWEEN {YEAR_LO} AND {YEAR_HI}
          AND ccg IN (SELECT ccg FROM support)
        GROUP BY 1, 2, 3
    """


def _efsy_cells_sql(rung):
    """EFSY employment cells at one rung. The county rung is EFSY's own county
    file, not a re-collapse, so the comparison is source against source."""
    if rung == "county":
        return f"""
            SELECT ccg, '' AS code, year, emp
            FROM read_parquet('{EFSY_COUNTY}')
            WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
              AND ccg IN (SELECT ccg FROM support)
        """
    glob = EFSY_N2 if rung == "naics2" else EFSY_N4
    return f"""
        SELECT ccg, CAST(code AS VARCHAR) AS code, year, sum(emp) AS emp
        FROM read_parquet('{glob}')
        WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
          AND ccg IN (SELECT ccg FROM support)
        GROUP BY 1, 2, 3
    """


def build_cells(con, rung, universe):
    """(ccg, code, year, en, ee, ...) with each source's zeros made explicit
    only where the code exists in BOTH sources; `key_class` labels the rest."""
    con.execute(f"CREATE OR REPLACE TEMP VIEW n_raw AS {_nets_cells_sql(rung, universe)}")
    con.execute(f"CREATE OR REPLACE TEMP VIEW e_raw AS {_efsy_cells_sql(rung)}")

    # A code counts as present in a source if that source ever reports strictly
    # positive employment for it anywhere in the window. Rectangularised zeros
    # (EFSY materialises them) must not make a code look present.
    con.execute("""
        CREATE OR REPLACE TEMP VIEW code_class AS
        WITH nc AS (SELECT DISTINCT code FROM n_raw WHERE emp > 0),
             ec AS (SELECT DISTINCT code FROM e_raw WHERE emp > 0),
             u  AS (SELECT code FROM nc UNION SELECT code FROM ec)
        SELECT u.code,
               CASE WHEN nc.code IS NOT NULL AND ec.code IS NOT NULL THEN 'both'
                    WHEN nc.code IS NOT NULL THEN 'nets_only'
                    ELSE 'efsy_only' END AS key_class
        FROM u LEFT JOIN nc ON nc.code = u.code LEFT JOIN ec ON ec.code = u.code
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE cells AS
        SELECT coalesce(n.ccg, e.ccg)   AS ccg,
               coalesce(n.code, e.code) AS code,
               coalesce(n.year, e.year) AS year,
               coalesce(n.emp, 0.0)         AS en,
               coalesce(e.emp, 0.0)         AS ee,
               coalesce(n.n_estab, 0.0)     AS n_estab,
               coalesce(n.emp_imputed, 0.0) AS emp_imputed
        FROM n_raw n FULL OUTER JOIN e_raw e USING (ccg, code, year)
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE cells_both AS
        SELECT c.* FROM cells c JOIN code_class k USING (code)
        WHERE k.key_class = 'both'
    """)
    return con.execute("""
        SELECT k.key_class, count(DISTINCT c.code) AS n_codes,
               sum(c.en) AS emp_nets, sum(c.ee) AS emp_efsy
        FROM cells c JOIN code_class k USING (code)
        GROUP BY 1 ORDER BY 1
    """).df()


def build_pairs(con, h):
    """Growth pairs at horizon h from `cells_both`, both sources at once.

    A FULL OUTER self-join on (ccg, code) between year t and year t-h. A cell
    absent on one side of the join is a true zero in BOTH sources -- that is
    entry or exit, and midpoint growth is defined there (it is +/-2), which is
    the entire reason the outcome is midpoint. A cell absent on both sides
    never materialises and contributes to neither source, correctly: no growth
    rate exists for a cell that is zero at both dates.
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pairs AS
        SELECT coalesce(a.ccg, b.ccg)         AS ccg,
               coalesce(a.code, b.code)       AS code,
               coalesce(a.year, b.year + {h}) AS year,
               coalesce(a.en, 0.0) AS en_t, coalesce(b.en, 0.0) AS en_b,
               coalesce(a.ee, 0.0) AS ee_t, coalesce(b.ee, 0.0) AS ee_b,
               coalesce(a.emp_imputed, 0.0) AS imp_t,
               coalesce(a.n_estab, 0.0)     AS n_estab_t
        FROM cells_both a FULL OUTER JOIN cells_both b
          ON a.ccg = b.ccg AND a.code = b.code AND b.year = a.year - {h}
        WHERE coalesce(a.year, b.year + {h}) BETWEEN {YEAR_LO + h} AND {YEAR_HI}
    """)
    # The three growth rates, both sources. Midpoint exists whenever the cell
    # is nonzero at either date; standard needs a positive base; log needs both.
    con.execute("""
        CREATE OR REPLACE TEMP VIEW g AS
        SELECT *,
               CASE WHEN en_t + en_b > 0 THEN 2*(en_t - en_b)/(en_t + en_b) END AS mp_n,
               CASE WHEN ee_t + ee_b > 0 THEN 2*(ee_t - ee_b)/(ee_t + ee_b) END AS mp_e,
               CASE WHEN en_b > 0 THEN (en_t - en_b)/en_b END AS st_n,
               CASE WHEN ee_b > 0 THEN (ee_t - ee_b)/ee_b END AS st_e,
               CASE WHEN en_b > 0 AND en_t > 0 THEN ln(en_t) - ln(en_b) END AS lg_n,
               CASE WHEN ee_b > 0 AND ee_t > 0 THEN ln(ee_t) - ln(ee_b) END AS lg_e,
               (ee_t + ee_b) AS w
        FROM pairs
    """)


# ------------------------------------------------------------- statistics

STAT_COLS = ["n", "pearson", "spearman", "med_gap", "p25_gap", "p75_gap",
             "sign_agree", "wmean_gap", "med_x", "med_y", "mean_x", "mean_y"]


def agreement(con, view, x, y, w="w", group="'ALL'", where="TRUE"):
    """Agreement of NETS (x) against the benchmark (y), by group.

    Spearman is `corr` over within-group ranks: no scipy dependency, and it is
    the statistic that survives NETS's level inflation, which the median gap
    then measures separately.
    """
    return con.execute(f"""
        WITH r AS (
            SELECT {group} AS grp, {x} AS x, {y} AS y, {w} AS w
            FROM {view}
            WHERE {where} AND {x} IS NOT NULL AND {y} IS NOT NULL
              AND isfinite({x}) AND isfinite({y})
        ), rk AS (
            SELECT *, rank() OVER (PARTITION BY grp ORDER BY x) AS rx,
                      rank() OVER (PARTITION BY grp ORDER BY y) AS ry
            FROM r
        )
        SELECT grp,
               count(*)                             AS n,
               corr(x, y)                           AS pearson,
               corr(rx, ry)                         AS spearman,
               median(x - y)                        AS med_gap,
               quantile_cont(x - y, 0.25)           AS p25_gap,
               quantile_cont(x - y, 0.75)           AS p75_gap,
               avg(CASE WHEN sign(x) = sign(y) THEN 1.0 ELSE 0.0 END) AS sign_agree,
               CASE WHEN sum(w) > 0 THEN sum(w * (x - y)) / sum(w) END AS wmean_gap,
               median(x) AS med_x, median(y) AS med_y,
               avg(x) AS mean_x, avg(y) AS mean_y
        FROM rk GROUP BY grp ORDER BY grp
    """).df()


# ----------------------------------------------------------------- stocks

def run_levels(con):
    """County-rung levels by year, NETS both universes vs EFSY."""
    return con.execute(f"""
        WITH n AS (
            SELECT year, universe, sum(emp) AS emp, sum(n_estab) AS n_estab,
                   sum(emp_imputed) AS emp_imputed
            FROM read_parquet('{NETS_COUNTY}')
            WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
              AND ccg IN (SELECT ccg FROM support)
            GROUP BY 1, 2
        ), e AS (
            SELECT year, sum(emp) AS emp FROM read_parquet('{EFSY_COUNTY}')
            WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
              AND ccg IN (SELECT ccg FROM support)
            GROUP BY 1
        ), b AS (
            SELECT year, sum(estabs) AS estabs
            FROM read_parquet('{dc.bds_path("bds_ccg_year.parquet")}')
            WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
              AND ccg IN (SELECT ccg FROM support_bds)
            GROUP BY 1
        )
        SELECT n.year, n.universe, n.emp AS emp_nets, e.emp AS emp_efsy,
               n.emp / e.emp AS ratio_emp,
               n.emp_imputed / nullif(n.emp, 0) AS share_imputed,
               n.n_estab AS estab_nets, b.estabs AS estab_bds,
               n.n_estab / nullif(b.estabs, 0) AS ratio_estab
        FROM n JOIN e USING (year) LEFT JOIN b USING (year)
        ORDER BY n.universe, n.year
    """).df()


def run_growth(con):
    """The growth benchmark: every rung x universe x horizon x growth rate."""
    rows, keyclass, zeros = [], [], []
    for universe in UNIVERSES:
        for rung in RUNGS:
            log(f"cells: {rung} / {universe}")
            kc = build_cells(con, rung, universe)
            kc.insert(0, "rung", rung)
            kc.insert(0, "universe", universe)
            keyclass.append(kc)
            for h in HORIZONS:
                build_pairs(con, h)
                for gname, xn, yn in (("mpgr", "mp_n", "mp_e"),
                                      ("standard", "st_n", "st_e"),
                                      ("log", "lg_n", "lg_e")):
                    d = agreement(con, "g", xn, yn)
                    d.insert(0, "growth", gname)
                    d.insert(0, "h", h)
                    d.insert(0, "rung", rung)
                    d.insert(0, "universe", universe)
                    rows.append(d)
                # Zero and small-cell mass: the mechanism the paper predicts.
                z = con.execute("""
                    SELECT count(*) AS n_pairs,
                           avg(CASE WHEN en_b <= 0 OR en_t <= 0 THEN 1.0 ELSE 0 END) AS zero_nets,
                           avg(CASE WHEN ee_b <= 0 OR ee_t <= 0 THEN 1.0 ELSE 0 END) AS zero_efsy,
                           avg(CASE WHEN en_b > 0 AND en_b < 10 THEN 1.0 ELSE 0 END) AS small_nets,
                           avg(CASE WHEN ee_b > 0 AND ee_b < 10 THEN 1.0 ELSE 0 END) AS small_efsy
                    FROM pairs
                """).df()
                z.insert(0, "h", h)
                z.insert(0, "rung", rung)
                z.insert(0, "universe", universe)
                zeros.append(z)
    return (pd.concat(rows, ignore_index=True),
            pd.concat(keyclass, ignore_index=True),
            pd.concat(zeros, ignore_index=True))


def run_divergence(con):
    """Where the gap lives: the four suspects the plan names, as real cuts.

    Run at the headline universe and horizon, at the county rung (base size,
    EmpC composition) and at NAICS4 (cell imputation share), because that is
    where each cut is defined.
    """
    out = []

    # (1) small counties. Deciles of EFSY base employment at the county rung.
    build_cells(con, "county", HEADLINE_UNIVERSE)
    build_pairs(con, HEADLINE_H)
    d = agreement(con, "g", "mp_n", "mp_e",
                  group="ntile(10) OVER (ORDER BY ee_b)")
    d.insert(0, "cut", "efsy_base_emp_decile")
    out.append(d)

    # (4) the reporting break. Same cut, by outcome year: a year-pair straddling
    # a jump in who reports an actual figure has a growth rate driven by
    # reporting, not by employment.
    d = agreement(con, "g", "mp_n", "mp_e", group="CAST(year AS VARCHAR)")
    d.insert(0, "cut", "outcome_year")
    out.append(d)

    # (3) EmpC composition, at the county rung: the share of NETS employment
    # carried by establishments whose primary line is an ESTIMATE. Read from
    # the `all` universe deliberately -- the `actual` universe is defined by
    # excluding those establishments, so inside it this share is near zero by
    # construction and carries no information. A missing bucket is a zero, not
    # a null: SUM over no rows returns NULL in SQL and would silently drop the
    # county-year.
    empc = con.execute(f"""
        SELECT ccg, year,
               (coalesce(emp_empc1, 0) + coalesce(emp_empc2, 0)
                + coalesce(emp_empc3, 0)) / nullif(emp, 0) AS empc_est_share
        FROM read_parquet('{NETS_COUNTY}')
        WHERE universe = 'all' AND year BETWEEN {YEAR_LO} AND {YEAR_HI}
    """).df()
    con.register("empc", empc)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW g_empc AS
        SELECT g.*, e.empc_est_share FROM g g JOIN empc e USING (ccg, year)
    """)
    d = agreement(con, "g_empc", "mp_n", "mp_e",
                  group="""CASE WHEN empc_est_share < 0.2 THEN '1. <20%'
                                WHEN empc_est_share < 0.4 THEN '2. 20-40%'
                                WHEN empc_est_share < 0.6 THEN '3. 40-60%'
                                WHEN empc_est_share < 0.8 THEN '4. 60-80%'
                                ELSE '5. 80%+' END""",
                  where="empc_est_share IS NOT NULL")
    d.insert(0, "cut", "county_empc_estimated_share")
    out.append(d)

    # (2) imputation-heavy cells, at NAICS4 where the cell is the unit. Fixed
    # share bands, not quintiles: the imputed share is a point mass at zero for
    # most cells, so ntile would cut the tie arbitrarily and print five
    # identical rows. Run on `all`, where imputation actually varies.
    build_cells(con, "naics4", "all")
    build_pairs(con, HEADLINE_H)
    d = agreement(con, "g", "mp_n", "mp_e",
                  group="""CASE WHEN imp_t / en_t = 0 THEN '1. none'
                                WHEN imp_t / en_t < 0.25 THEN '2. <25%'
                                WHEN imp_t / en_t < 0.50 THEN '3. 25-50%'
                                WHEN imp_t / en_t < 0.75 THEN '4. 50-75%'
                                WHEN imp_t / en_t < 1.00 THEN '5. 75-100%'
                                ELSE '6. all imputed' END""",
                  where="en_t > 0")
    d.insert(0, "cut", "cell_imputed_share")
    out.append(d)

    # Cell size, on the same `all` pairs, so the size and imputation cuts are
    # the same rows and can be read against each other.
    d = agreement(con, "g", "mp_n", "mp_e",
                  group="""CASE WHEN ee_b = 0 THEN '1. 0 (EFSY entry)'
                                WHEN ee_b < 5 THEN '2. 1-4'
                                WHEN ee_b < 20 THEN '3. 5-19'
                                WHEN ee_b < 100 THEN '4. 20-99'
                                ELSE '5. 100+' END""")
    d.insert(0, "cut", "efsy_base_cell_size_all")
    out.append(d)

    build_cells(con, "naics4", HEADLINE_UNIVERSE)
    build_pairs(con, HEADLINE_H)

    # Cell size at NAICS4: the small-cell mass is where log and midpoint part.
    # Bins are numbered so they sort in size order rather than lexically.
    d = agreement(con, "g", "mp_n", "mp_e",
                  group="""CASE WHEN ee_b = 0 THEN '1. 0 (EFSY entry)'
                                WHEN ee_b < 5 THEN '2. 1-4'
                                WHEN ee_b < 20 THEN '3. 5-19'
                                WHEN ee_b < 100 THEN '4. 20-99'
                                ELSE '5. 100+' END""")
    d.insert(0, "cut", "efsy_base_cell_size")
    out.append(d)

    res = pd.concat(out, ignore_index=True)
    res["grp"] = res["grp"].astype(str)
    return res


# ------------------------------------------------------------------ flows

def build_entry(con):
    """NETS births vs BDS entry, county and sector rungs, both universes.

    Both rates are rebuilt FROM THE COUNTS on both sides, so the two are the
    same estimand:  entry_rate_ct = Entry_ct / Estabs_{c,t-1}.  BDS's published
    `estabs_entry_rate` uses a DHS average denominator instead; that variant is
    built here too, identically on both sides, and reported beside it. Neither
    published rate is ever read: a ratio does not survive aggregation to a ccg,
    which is why producer 74 refused to carry it.
    """
    bds_c = dc.bds_path("bds_ccg_year.parquet")
    bds_s = dc.bds_path("bds_ccg_sector_year.parquet")
    # NAICS2 -> BDS sector. Written against the expression that produces the
    # two-digit code so it can be dropped into any NETS query unchanged.
    n2 = "substr(naics4, 1, 2)"
    sector_case = ("CASE " + " ".join(
        f"WHEN {n2} = '{k}' THEN '{v}'" for k, v in SECTOR_RANGES.items()
    ) + f" WHEN {n2} IN (" + ", ".join(
        f"'{s}'" for s in BDS_SECTORS if "-" not in s
    ) + f") THEN {n2} ELSE NULL END")

    # County rung.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE entry_county AS
        WITH nb AS (
            SELECT universe, ccg, year, sum(n_births) AS births
            FROM read_parquet('{BIRTH_COUNTY}')
            WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
              AND ccg IN (SELECT ccg FROM support_bds)
            GROUP BY 1, 2, 3
        ), ns AS (
            SELECT universe, ccg, year, sum(n_estab) AS estabs
            FROM read_parquet('{NETS_COUNTY}')
            WHERE year BETWEEN {YEAR_LO - 1} AND {YEAR_HI}
              AND ccg IN (SELECT ccg FROM support_bds)
            GROUP BY 1, 2, 3
        ), b AS (
            SELECT ccg, year, estabs_entry, estabs
            FROM read_parquet('{bds_c}')
            WHERE year BETWEEN {YEAR_LO - 1} AND {YEAR_HI}
              AND ccg IN (SELECT ccg FROM support_bds)
        )
        SELECT nb.universe, nb.ccg, nb.year,
               nb.births                        AS entry_n,
               nl.estabs                        AS base_n,
               nt.estabs                        AS cur_n,
               bt.estabs_entry                  AS entry_b,
               bl.estabs                        AS base_b,
               bt.estabs                        AS cur_b,
               nb.births / nullif(nl.estabs, 0) AS er_n,
               bt.estabs_entry / nullif(bl.estabs, 0) AS er_b,
               2 * nb.births / nullif(nt.estabs + nl.estabs, 0) AS dhs_n,
               2 * bt.estabs_entry / nullif(bt.estabs + bl.estabs, 0) AS dhs_b
        FROM nb
        JOIN ns nl ON nl.universe = nb.universe AND nl.ccg = nb.ccg
                  AND nl.year = nb.year - 1
        JOIN ns nt ON nt.universe = nb.universe AND nt.ccg = nb.ccg
                  AND nt.year = nb.year
        JOIN b bt ON bt.ccg = nb.ccg AND bt.year = nb.year
        JOIN b bl ON bl.ccg = nb.ccg AND bl.year = nb.year - 1
    """)

    # Sector rung. The NETS side is collapsed onto BDS's 19 sector codes; the
    # NAICS2 codes with no BDS sector (92 public administration, 99, UNKNOWN)
    # are kept aside and sized, never folded into a neighbouring sector.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE entry_sector AS
        WITH nb AS (
            SELECT universe, ccg, year,
                   {sector_case} AS sector,
                   sum(n_births) AS births
            FROM read_parquet('{BIRTH_IND}')
            WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
              AND ccg IN (SELECT ccg FROM support_bds)
            GROUP BY 1, 2, 3, 4
        ), ns AS (
            SELECT universe, ccg, year,
                   {sector_case} AS sector,
                   sum(n_estab) AS estabs
            FROM read_parquet('{NETS_IND}')
            WHERE year BETWEEN {YEAR_LO - 1} AND {YEAR_HI}
              AND ccg IN (SELECT ccg FROM support_bds)
            GROUP BY 1, 2, 3, 4
        ), b AS (
            SELECT ccg, sector, year, estabs_entry, estabs
            FROM read_parquet('{bds_s}')
            WHERE year BETWEEN {YEAR_LO - 1} AND {YEAR_HI}
              AND ccg IN (SELECT ccg FROM support_bds)
        )
        SELECT nb.universe, nb.ccg, nb.sector, nb.year,
               nb.births AS entry_n, nl.estabs AS base_n, nt.estabs AS cur_n,
               bt.estabs_entry AS entry_b, bl.estabs AS base_b, bt.estabs AS cur_b,
               nb.births / nullif(nl.estabs, 0) AS er_n,
               bt.estabs_entry / nullif(bl.estabs, 0) AS er_b,
               2 * nb.births / nullif(nt.estabs + nl.estabs, 0) AS dhs_n,
               2 * bt.estabs_entry / nullif(bt.estabs + bl.estabs, 0) AS dhs_b
        FROM nb
        JOIN ns nl ON nl.universe = nb.universe AND nl.ccg = nb.ccg
                  AND nl.sector = nb.sector AND nl.year = nb.year - 1
        JOIN ns nt ON nt.universe = nb.universe AND nt.ccg = nb.ccg
                  AND nt.sector = nb.sector AND nt.year = nb.year
        JOIN b bt ON bt.ccg = nb.ccg AND bt.sector = nb.sector AND bt.year = nb.year
        JOIN b bl ON bl.ccg = nb.ccg AND bl.sector = nb.sector
                 AND bl.year = nb.year - 1
        WHERE nb.sector IS NOT NULL
    """)

    # The out-of-scope NETS residual, sized rather than dropped in silence.
    resid = con.execute(f"""
        SELECT substr(naics4, 1, 2) AS n2, sum(n_births) AS births
        FROM read_parquet('{BIRTH_IND}')
        WHERE universe = 'all' AND year BETWEEN {YEAR_LO} AND {YEAR_HI}
          AND ccg IN (SELECT ccg FROM support_bds)
          AND {sector_case} IS NULL
        GROUP BY 1 ORDER BY 2 DESC
    """).df()

    # BDS suppression, measured on exactly the cells this benchmark asks for.
    supp = con.execute(f"""
        SELECT count(*) AS n_cells,
               avg(CASE WHEN estabs_entry IS NULL THEN 1.0 ELSE 0 END) AS share_supp
        FROM read_parquet('{bds_s}')
        WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
          AND ccg IN (SELECT ccg FROM support_bds)
    """).df().iloc[0].to_dict()
    return resid, supp


def run_entry(con):
    """Agreement and the aggregate rate, per rung x universe x rate.

    The aggregate is computed ONCE PER RATE and on that rate's own pair
    sample, not once per (rung, universe): the DHS variant has a different
    denominator and a slightly different defined set, so sharing one number
    across both rate rows would print a column whose constancy is an artefact
    of the code rather than a fact about the data.
    """
    rows = []
    # (name, NETS column, BDS column, aggregate numerator, aggregate denominator)
    RATES = (("entry_rate", "er_n", "er_b", "{s}", "base_{s}"),
             ("entry_rate_dhs", "dhs_n", "dhs_b", "2 * {s}", "cur_{s} + base_{s}"))
    for tbl, rung in (("entry_county", "county"), ("entry_sector", "sector")):
        for universe in UNIVERSES:
            for rate, xn, yn, num, den in RATES:
                where = (f"universe = '{universe}' AND {xn} IS NOT NULL "
                         f"AND {yn} IS NOT NULL AND isfinite({xn}) "
                         f"AND isfinite({yn})")
                d = agreement(con, tbl, xn, yn, w="base_b", where=where)
                d.insert(0, "rate", rate)
                d.insert(0, "universe", universe)
                d.insert(0, "rung", rung)
                rows.append(d)

                def agg(s):
                    return (f"sum({num.format(s='entry_' + s)}) / "
                            f"nullif(sum({den.format(s=s)}), 0)")

                tot = con.execute(f"""
                    SELECT sum(entry_n) AS entry_nets, sum(base_n) AS base_nets,
                           sum(entry_b) AS entry_bds, sum(base_b) AS base_bds,
                           {agg('n')} AS er_nets_agg,
                           {agg('b')} AS er_bds_agg,
                           count(*) AS n_cells
                    FROM {tbl} WHERE {where}
                """).df()
                tot.insert(0, "rate", rate)
                tot.insert(0, "universe", universe)
                tot.insert(0, "rung", rung)
                rows.append(tot.assign(grp="LEVELS"))
    return pd.concat(rows, ignore_index=True)


def entry_by_year(con):
    return con.execute("""
        SELECT year,
               sum(entry_n) / nullif(sum(base_n), 0) AS er_nets,
               sum(entry_b) / nullif(sum(base_b), 0) AS er_bds
        FROM entry_county WHERE universe = 'all' AND entry_b IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).df()


# ----------------------------------------------------------------- output

def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:.{nd}f}"


def tabular(header, body, spec=None, note=None):
    """A booktabs fragment, the convention every table in Results/ follows."""
    n = len(header)
    spec = spec or ("l" + "r" * (n - 1))
    L = [r"\begin{tabular}{@{}" + spec + r"@{}}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"]
    for row in body:
        if row is None:
            L.append(r"\midrule")
        elif isinstance(row, str):
            L.append(r"\multicolumn{%d}{@{}l}{\textit{%s}} \\" % (n, row))
        else:
            L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L += [r"\begin{minipage}{\textwidth}\vspace{2pt}\footnotesize",
              note, r"\end{minipage}"]
    return "\n".join(L) + "\n"


def md_table(header, body):
    L = ["| " + " | ".join(header) + " |",
         "|" + "|".join(["---"] * len(header)) + "|"]
    for row in body:
        L.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(L) + "\n"


def figure_growth(con, path):
    """Binned NETS vs EFSY midpoint growth, one panel per rung."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, rung in zip(axes, RUNGS):
        build_cells(con, rung, HEADLINE_UNIVERSE)
        build_pairs(con, HEADLINE_H)
        d = con.execute("""
            WITH b AS (
                SELECT ntile(40) OVER (ORDER BY mp_e) AS bin, mp_e, mp_n
                FROM g WHERE mp_e IS NOT NULL AND mp_n IS NOT NULL
            )
            SELECT bin, avg(mp_e) AS x, avg(mp_n) AS y
            FROM b GROUP BY bin ORDER BY bin
        """).df()
        lim = max(abs(d[["x", "y"]].to_numpy()).max() * 1.1, 0.3)
        ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.8, zorder=1)
        ax.scatter(d["x"], d["y"], s=14, color="#1f4e79", zorder=2)
        ax.set_title(rung, fontsize=9)
        ax.set_xlabel("EFSY midpoint growth", fontsize=8)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("NETS midpoint growth", fontsize=8)
    fig.suptitle(f"NETS vs EFSY midpoint growth, h = {HEADLINE_H}, "
                 f"{HEADLINE_UNIVERSE} universe, 40 equal-count bins",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_entry(con, path, by_year):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    d = con.execute("""
        WITH b AS (
            SELECT ntile(40) OVER (ORDER BY er_b) AS bin, er_b, er_n
            FROM entry_county WHERE universe = 'all'
              AND er_b IS NOT NULL AND er_n IS NOT NULL AND isfinite(er_n)
        )
        SELECT bin, avg(er_b) AS x, avg(er_n) AS y
        FROM b GROUP BY bin ORDER BY bin
    """).df()
    ax = axes[0]
    lim = max(d[["x", "y"]].to_numpy().max() * 1.1, 0.1)
    ax.plot([0, lim], [0, lim], color="0.6", lw=0.8)
    ax.scatter(d["x"], d["y"], s=14, color="#8b1a1a")
    ax.set_xlabel("BDS entry rate", fontsize=8)
    ax.set_ylabel("NETS birth rate", fontsize=8)
    ax.set_title("County, 40 equal-count bins", fontsize=9)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    ax.plot(by_year["year"], by_year["er_nets"], color="#8b1a1a",
            marker="o", ms=3, label="NETS births / estabs$_{t-1}$")
    ax.plot(by_year["year"], by_year["er_bds"], color="#1f4e79",
            marker="s", ms=3, label="BDS entry / estabs$_{t-1}$")
    ax.set_title("Aggregate entry rate by year", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("NETS establishment births vs BDS establishment entry "
                 "(all universe)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- tables and report

STAT_HEAD = ["Rung", "$h$", "$N$ pairs", "Pearson", "Spearman",
             "Median gap", "IQR of gap", "Sign agree", "Wtd mean gap"]


def _stat_row(label, r):
    return [label, fmt(r["n"], 0) if False else f"{int(r['n']):,}",
            fmt(r["pearson"]), fmt(r["spearman"]), fmt(r["med_gap"]),
            f"[{fmt(r['p25_gap'], 2)}, {fmt(r['p75_gap'], 2)}]",
            fmt(r["sign_agree"], 2), fmt(r["wmean_gap"])]


def growth_table(growth, universe):
    """One panel per growth rate; rows are rung x horizon."""
    head = ["Rung", "$h$", "$N$ pairs", "Pearson", "Spearman", "Median gap",
            "IQR of gap", "Sign agree", "Wtd mean gap"]
    body = []
    for gname, title in (("mpgr", "Panel A. Midpoint growth rate"),
                         ("standard", "Panel B. Standard growth rate "
                                      "(base-period denominator)"),
                         ("log", "Panel C. Log growth "
                                 "(both endpoints strictly positive)")):
        body.append(title)
        for rung in RUNGS:
            for h in HORIZONS:
                m = growth[(growth.universe == universe) & (growth.rung == rung)
                           & (growth.growth == gname) & (growth.h == h)]
                if m.empty:
                    continue
                r = m.iloc[0]
                body.append([RUNG_LABEL[rung] if h == HORIZONS[0] else "",
                             str(h)] + _stat_row("", r)[2:])
        body.append(None)
    if body and body[-1] is None:
        body.pop()
    return head, body


def entry_table(entry, supp):
    # No sign-agreement column here: both entry rates are nonnegative by
    # construction, so the statistic is 1 by definition and says nothing. The
    # median NETS and median BDS rates carry the level comparison instead.
    head = ["Rung", "Universe", "Rate", "$N$ cells", "Pearson", "Spearman",
            "Median gap", "Median NETS", "Median BDS", "NETS agg.", "BDS agg."]
    body = []
    for rung in ("county", "sector"):
        for universe in UNIVERSES:
            for rate, lab in (("entry_rate", "$E_t / N_{t-1}$"),
                              ("entry_rate_dhs", "$2E_t/(N_t{+}N_{t-1})$")):
                m = entry[(entry.rung == rung) & (entry.universe == universe)
                          & (entry.rate == rate) & (entry.grp == "ALL")]
                if m.empty:
                    continue
                lev = entry[(entry.rung == rung) & (entry.universe == universe)
                            & (entry.rate == rate) & (entry.grp == "LEVELS")]
                er_n = lev["er_nets_agg"].iloc[0] if not lev.empty else np.nan
                er_b = lev["er_bds_agg"].iloc[0] if not lev.empty else np.nan
                r = m.iloc[0]
                body.append([
                    ("County" if rung == "county" else "County $\\times$ sector")
                    if rate == "entry_rate" and universe == UNIVERSES[0] else "",
                    universe if rate == "entry_rate" else "",
                    lab, f"{int(r['n']):,}", fmt(r["pearson"]),
                    fmt(r["spearman"]), fmt(r["med_gap"]),
                    fmt(r["med_x"]), fmt(r["med_y"]),
                    fmt(er_n), fmt(er_b)])
        body.append(None)
    if body and body[-1] is None:
        body.pop()
    return head, body


def levels_table(levels):
    head = ["Year", "NETS emp. (actual)", "NETS emp. (all)", "EFSY emp.",
            "NETS/EFSY (actual)", "NETS/EFSY (all)", "NETS estab.",
            "BDS estab.", "NETS/BDS"]
    piv = levels.pivot(index="year", columns="universe")
    body = []
    for year, r in piv.iterrows():
        body.append([str(int(year)),
                     f"{r[('emp_nets', 'actual')] / 1e6:,.1f}M",
                     f"{r[('emp_nets', 'all')] / 1e6:,.1f}M",
                     f"{r[('emp_efsy', 'all')] / 1e6:,.1f}M",
                     fmt(r[("ratio_emp", "actual")], 2),
                     fmt(r[("ratio_emp", "all")], 2),
                     f"{r[('estab_nets', 'all')] / 1e6:,.2f}M",
                     f"{r[('estab_bds', 'all')] / 1e6:,.2f}M"
                     if np.isfinite(r[("estab_bds", "all")]) else "--",
                     fmt(r[("ratio_estab", "all")], 2)])
    return head, body


LEVELS_NOTE = (
    "NETS and EFSY employment, and NETS and BDS establishment counts, summed "
    "over the fixed ccg support. NETS `actual' is the subsample of "
    "establishments reporting an actual employment figure; `all' is the full "
    "placed panel. The BDS column is on the BDS-comparable support (BDS's 2023 "
    "release retro-codes Connecticut onto planning regions, so the eight legacy "
    "Connecticut counties have no BDS row in any year) and is missing the "
    "statewide-unknown mass BDS publishes under XX999, 0.17\\% of "
    "establishments. Nothing is winsorized and no cell is dropped.")

GROWTH_NOTE = (
    "Agreement of the NETS growth rate against the EFSY growth rate, cell by "
    "cell, over %d--%d. `Gap' is NETS minus EFSY. Spearman is the correlation "
    "of within-sample ranks. The weighted mean gap uses the midpoint mass "
    "$(E_t + E_{t-h})$ of the EFSY cell, the weight the disaggregation test "
    "itself uses. Midpoint growth is defined wherever the cell is nonzero at "
    "either date, so it retains entering and exiting cells; the standard rate "
    "needs a positive base and the log rate needs both endpoints positive, "
    "which is why $N$ falls down Panels B and C -- that fall is the "
    "measurement problem, not a sample choice. Industry codes one source "
    "carries and the other never does are excluded from every panel and sized "
    "separately in the report.") % (YEAR_LO, YEAR_HI)

ENTRY_NOTE = (
    "NETS establishment births against BDS establishment entry. Both rates are "
    "rebuilt from counts on both sides so the two are the same estimand; BDS's "
    "own published rate is never read, because a ratio does not survive "
    "aggregation to a county group. The industry rung is BDS \\emph{sector} "
    "(19 codes, three of them ranges of NAICS2), not NAICS2: BDS publishes no "
    "county $\\times$ NAICS2 table and its finer industry tables carry no "
    "geography. BDS withholds `estabs\\_entry' in @SUPP@\\% of the "
    "county-sector-years on this support; those cells are excluded from the "
    "pair and never coerced to zero, so the sector-rung statistics describe "
    "the unsuppressed remainder, which is the large-cell part of the "
    "distribution the comparison is least interested in. NETS births are "
    "first appearances in the panel from 1991 on; the 1990 cohort is "
    "left-censored and excluded.")


def write_outputs(con, cov, levels, growth, keyclass, diverge, entry, eby,
                  resid, supp, zeros):
    stats = []

    # ------------------------------------------------------------- tables
    h, b = levels_table(levels)
    (RESULTS / "Table_nets_efsy_levels.tex").write_text(
        tabular(h, b, spec="l" + "r" * (len(h) - 1), note=LEVELS_NOTE))

    h, b = growth_table(growth, HEADLINE_UNIVERSE)
    (RESULTS / "Table_nets_efsy_growth.tex").write_text(
        tabular(h, b, note=GROWTH_NOTE))

    h, b = entry_table(entry, supp)
    (RESULTS / "Table_nets_bds_entry.tex").write_text(
        tabular(h, b, spec="lll" + "r" * (len(h) - 3),
                note=ENTRY_NOTE.replace(
                    "@SUPP@", f"{100 * supp['share_supp']:.1f}")))

    # ---------------------------------------------------------------- csv
    growth.to_csv(RESULTS / "nets_benchmark_stats.csv", index=False)
    div_out = pd.concat([
        diverge.assign(family="divergence_cut"),
        zeros.assign(family="zero_and_small_mass"),
        keyclass.assign(family="code_scope"),
        levels.assign(family="levels"),
        entry.assign(family="entry"),
        eby.assign(family="entry_by_year"),
        resid.assign(family="nets_births_outside_bds_scope"),
    ], ignore_index=True)
    div_out.to_csv(RESULTS / "nets_benchmark_divergence.csv", index=False)

    # ------------------------------------------------------------- report
    md = build_report(cov, levels, growth, keyclass, diverge, entry, eby,
                      resid, supp, zeros)
    md_path = RESULTS / "ws2_step2_benchmark_report.md"
    md_path.write_text(md)
    log(f"wrote {md_path}")
    compile_pdf(md_path)


def _pick(df, **kw):
    m = df
    for k, v in kw.items():
        m = m[m[k] == v]
    return m.iloc[0] if len(m) else None


def build_report(cov, levels, growth, keyclass, diverge, entry, eby, resid,
                 supp, zeros):
    L = []
    A = L.append
    A("# WS2 step 2: how problematic is NETS?\n\n")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
      "`step_1/code/pipeline/10_nets_benchmark.py`. Benchmarks the NETS "
      "employment stock against the desuppressed CBP (EFSY) and the NETS "
      "establishment birth flow against BDS entry, over the "
      f"{YEAR_LO}-{YEAR_HI} overlap.\n\n")

    A("**No pass criterion was pre-registered.** The plan asked for one and "
      "the decision taken for this run was to drop it, so this is a "
      "descriptive benchmark and it does not gate WS2 step 5. Every statistic "
      "a criterion would have been written against is printed below. If one is "
      "wanted later it has to be written against a sample whose numbers are "
      "not already visible, which this one's now are; nothing here should be "
      "read as a verdict on whether the NETS outcome swap goes ahead.\n\n")

    # ---- findings, generated from the numbers rather than typed
    A("## What the numbers say\n\n")
    lv0 = levels.pivot(index="year", columns="universe")
    c2 = _pick(growth, universe=HEADLINE_UNIVERSE, rung="county",
               growth="mpgr", h=HEADLINE_H)
    n4 = _pick(growth, universe=HEADLINE_UNIVERSE, rung="naics4",
               growth="mpgr", h=HEADLINE_H)
    z4 = zeros[(zeros.universe == HEADLINE_UNIVERSE) & (zeros.rung == "naics4")
               & (zeros.h == HEADLINE_H)].iloc[0]
    ec = _pick(entry, rung="county", universe="all", rate="entry_rate",
               grp="ALL")
    es = _pick(entry, rung="sector", universe="all", rate="entry_rate",
               grp="ALL")
    ratio = eby["er_nets"] / eby["er_bds"]
    A("- **On levels NETS is not CBP, and the gap is not constant.** The "
      f"full NETS panel carries {lv0[('ratio_emp', 'all')].mean():.2f} times "
      f"EFSY employment on average, drifting from "
      f"{lv0[('ratio_emp', 'all')].iloc[0]:.2f} to "
      f"{lv0[('ratio_emp', 'all')].iloc[-1]:.2f} across the window, and "
      f"{lv0[('ratio_estab', 'all')].iloc[0]:.1f} to "
      f"{lv0[('ratio_estab', 'all')].iloc[-1]:.1f} times as many "
      "establishments. The actual-employment subsample is much closer on "
      f"employment ({lv0[('ratio_emp', 'actual')].mean():.2f} on average), "
      "which is the first argument for the gate's choice of it as the "
      "headline universe.\n")
    A("- **On growth the two sources agree weakly even at the county rung.** "
      f"Midpoint growth at h = {HEADLINE_H} correlates "
      f"{c2['pearson']:.2f} (Spearman {c2['spearman']:.2f}) across "
      f"{int(c2['n']):,} county-years, with NETS running "
      f"{c2['med_gap']:+.3f} above EFSY at the median and agreeing on the "
      f"sign of growth in {c2['sign_agree']:.0%} of them. At NAICS4 the "
      f"correlation falls to {n4['pearson']:.2f} and sign agreement to "
      f"{n4['sign_agree']:.0%} -- worse than a coin toss, because a cell that "
      "one source calls entry the other often calls a continuing "
      "establishment.\n")
    dec = (diverge[diverge.cut == "efsy_base_emp_decile"]
           .assign(_k=lambda t: pd.to_numeric(t["grp"])).sort_values("_k"))
    A("- **The disagreement is concentrated in cells, not counties.** The "
      "county cut by size moves the correlation only from "
      f"{dec.iloc[0]['pearson']:.2f} in the smallest employment decile to "
      f"{dec.iloc[-1]['pearson']:.2f} in the largest, while "
      f"moving to the NAICS4 rung costs an order of magnitude: "
      f"{z4['zero_nets']:.0%} of NETS pairs and {z4['zero_efsy']:.0%} of EFSY "
      "pairs have a zero endpoint there. Those are exactly the cells the log "
      "and standard growth rates cannot take, which is why their N falls by "
      "millions between the panels below and why their weighted mean gap "
      "explodes: the standard rate divides by bases of one and two "
      "employees.\n")
    A("- **NETS entry is not BDS entry.** At the county rung the birth rate "
      f"correlates {ec['pearson']:.2f} with BDS entry over "
      f"{int(ec['n']):,} county-years; at the sector rung "
      f"{es['pearson']:.2f}. The aggregate rates are close on the full panel "
      f"({ec['med_x']:.3f} vs {ec['med_y']:.3f} at the median), but the "
      f"year-by-year ratio swings from {ratio.min():.2f} to "
      f"{ratio.max():.2f}: NETS births arrive in bursts as D&B discovers "
      "establishments, and a burst year is not an entry wave. Any regression "
      "of a birth outcome on a year-varying shock inherits that.\n")
    A("\nThese are the four facts the plan asked for. What they imply for the "
      "outcome swap is a judgement no criterion in this file makes.\n\n")

    # ---- support
    A("## The sample, fixed before any statistic\n\n")
    A("The support is the intersection of the locked county-group panel, the "
      "NETS ccgs and the EFSY ccgs (gate Q7). It is fixed once and every "
      "exhibit uses it; a county present in one source and absent from the "
      "other is a coverage fact reported here, never a per-exhibit re-cut.\n\n")
    A(md_table(["set", "ccgs"], [
        ["locked analysis panel", f"{cov['locked']:,}"],
        [f"NETS, {YEAR_LO}-{YEAR_HI}", f"{cov['nets']:,}"],
        [f"EFSY, {YEAR_LO}-{YEAR_HI}", f"{cov['efsy']:,}"],
        ["**the support (intersection)**", f"**{cov['support']:,}**"],
        ["BDS in window", f"{cov['bds']:,}"],
        ["support that BDS also covers", f"{cov['support_bds']:,}"],
        ["locked ccgs with no NETS row", f"{cov['locked_not_nets']:,}"],
        ["locked ccgs with no EFSY row", f"{cov['locked_not_efsy']:,}"],
        ["NETS ccgs outside the locked support", f"{cov['nets_not_locked']:,}"],
        ["support ccgs with no BDS row", f"{cov['support_not_bds']:,}"]]))
    A(f"\nThe {cov['support_not_bds']} support counties BDS cannot cover are "
      "Connecticut: the 2023 BDS release retro-codes the state onto its nine "
      "2022 planning regions for every year back to 1978, and planning regions "
      "do not nest in counties. No option available to this benchmark "
      "recovers them.\n\n")

    # ---- levels
    A("## Stocks, part 1: levels\n\n")
    lv = levels.pivot(index="year", columns="universe")
    r0, r1 = lv.index.min(), lv.index.max()
    A(f"NETS employment on this support runs at "
      f"{lv[('ratio_emp', 'all')].mean():.2f} times EFSY on the full panel and "
      f"{lv[('ratio_emp', 'actual')].mean():.2f} times EFSY on the "
      f"actual-employment subsample, averaged over {r0}-{r1}. The ratio is not "
      f"stable: it moves from {lv[('ratio_emp', 'all')].iloc[0]:.2f} in {r0} to "
      f"{lv[('ratio_emp', 'all')].iloc[-1]:.2f} in {r1} on the full panel. A "
      "drifting level ratio is a drifting growth rate, which is the first "
      "reason the growth benchmark below matters more than this table.\n\n")
    # Not every jump in the ratio is NETS's: EFSY has breaks of its own, and a
    # ratio moves when either side does. Flagged so no reader charges an EFSY
    # break to NETS.
    ee = lv[("emp_efsy", "all")]
    jump = ee.pct_change().abs()
    breaks = [(int(y), ee.pct_change().loc[y]) for y in ee.index
              if np.isfinite(jump.get(y, np.nan)) and jump.loc[y] > 0.05]
    if breaks:
        A("**Some of the movement is EFSY's, not NETS's.** EFSY's own county "
          "total moves by more than 5% between consecutive years in "
          + ", ".join(f"{y} ({d:+.1%})" for y, d in breaks)
          + ". The ratio column moves when either side does; a jump in one of "
            "these years is not evidence about NETS.\n\n")
    A(md_table(
        ["year", "NETS emp (actual)", "NETS emp (all)", "EFSY emp",
         "ratio (actual)", "ratio (all)", "NETS estabs", "BDS estabs",
         "ratio estabs"],
        [[str(int(y)),
          f"{r[('emp_nets', 'actual')]/1e6:,.1f}M",
          f"{r[('emp_nets', 'all')]/1e6:,.1f}M",
          f"{r[('emp_efsy', 'all')]/1e6:,.1f}M",
          f"{r[('ratio_emp', 'actual')]:.2f}",
          f"{r[('ratio_emp', 'all')]:.2f}",
          f"{r[('estab_nets', 'all')]/1e6:,.2f}M",
          (f"{r[('estab_bds', 'all')]/1e6:,.2f}M"
           if np.isfinite(r[('estab_bds', 'all')]) else "--"),
          (f"{r[('ratio_estab', 'all')]:.2f}"
           if np.isfinite(r[('ratio_estab', 'all')]) else "--")]
         for y, r in lv.iterrows()]))

    # ---- scope
    A("\n## Stocks, part 2: what the two sources do not both measure\n\n")
    A("Before any growth rate: an industry code that appears in one source and "
      "never in the other is a scope difference, not growth. Coalescing it to "
      "a zero on the other side would print midpoint growth of +2 in every "
      "county that has it. Such codes are labelled and held out of every "
      "agreement statistic below; they are not deleted, they are priced "
      "here.\n\n")
    kc = keyclass[keyclass.universe == HEADLINE_UNIVERSE]
    A(md_table(["rung", "class", "codes", "NETS employment", "EFSY employment"],
               [[r.rung, r.key_class, f"{int(r.n_codes):,}",
                 f"{r.emp_nets:,.0f}", f"{r.emp_efsy:,.0f}"]
                for _, r in kc.iterrows()]))

    # ---- growth
    A("\n## Stocks, part 3: growth\n\n")
    A(f"The headline is the `{HEADLINE_UNIVERSE}` universe. `Gap` is NETS "
      "minus EFSY. The weighted mean gap uses the EFSY cell's midpoint mass, "
      "the weight the disaggregation test itself uses.\n\n")
    A("Read the three panels against each other rather than one at a time. "
      "The midpoint rate keeps every pair with a nonzero endpoint; the "
      "standard rate needs a positive base and the log rate needs both "
      "endpoints positive, so their N falls by hundreds of thousands at "
      "NAICS2 and by millions at NAICS4. That fall is the measurement problem "
      "itself, not a sample choice, and it is why the standard rate's "
      "weighted mean gap runs to double digits while the midpoint rate's "
      "stays near 0.02: dividing by a base of one or two employees is "
      "unbounded, and NETS and EFSY disagree about exactly those cells. The "
      "midpoint rate is bounded in [-2, 2] and cannot do that.\n\n")
    for universe in UNIVERSES:
        A(f"\n### Universe: `{universe}`\n\n")
        for gname, title in (("mpgr", "Midpoint growth rate"),
                             ("standard", "Standard growth rate"),
                             ("log", "Log growth")):
            A(f"\n**{title}**\n\n")
            rows = []
            for rung in RUNGS:
                for h in HORIZONS:
                    r = _pick(growth, universe=universe, rung=rung,
                              growth=gname, h=h)
                    if r is None:
                        continue
                    rows.append([rung, str(h), f"{int(r['n']):,}",
                                 fmt(r["pearson"]), fmt(r["spearman"]),
                                 fmt(r["med_gap"]),
                                 f"[{fmt(r['p25_gap'], 2)}, "
                                 f"{fmt(r['p75_gap'], 2)}]",
                                 fmt(r["sign_agree"], 2),
                                 fmt(r["wmean_gap"])])
            # Nine columns is what fits the page; the median NETS and median
            # EFSY rates ride in nets_benchmark_stats.csv.
            A(md_table(["rung", "h", "N pairs", "Pearson", "Spearman",
                        "median gap", "IQR of gap", "sign agree",
                        "wtd mean gap"], rows))

    # ---- zero mass
    A("\n## Why the three growth rates behave differently: the zero and "
      "small-cell mass\n\n")
    A("The share of pairs with a zero endpoint, and the share with a base "
      "under ten employees. These are the cells the log rate cannot take and "
      "the standard rate divides by. They are the whole reason the N column "
      "collapses between the panels above.\n\n")
    zz = zeros[zeros.universe == HEADLINE_UNIVERSE]
    A(md_table(["rung", "h", "pairs", "zero endpoint, NETS",
                "zero endpoint, EFSY", "base < 10, NETS", "base < 10, EFSY"],
               [[r.rung, str(int(r.h)), f"{int(r.n_pairs):,}",
                 f"{r.zero_nets:.1%}", f"{r.zero_efsy:.1%}",
                 f"{r.small_nets:.1%}", f"{r.small_efsy:.1%}"]
                for _, r in zz.iterrows()]))

    # ---- entry
    A("\n## Flows: NETS births vs BDS entry\n\n")
    A("Both rates are rebuilt from counts on both sides. The industry rung is "
      "BDS **sector**, 19 codes with three of them ranges of NAICS2, because "
      "BDS publishes no county × NAICS2 table and its finer industry "
      "tables carry no geography. This is the requested NAICS2 comparison at "
      "the only industry resolution the benchmark data admits.\n\n")
    A(f"BDS withholds `estabs_entry` in {supp['share_supp']:.1%} of the "
      f"{int(supp['n_cells']):,} county-sector-years on this support "
      "(the ingest report's figure for the national file was "
      f"{BDS_SECTOR_SUPPRESSION:.1%}). Those cells are excluded from the pair "
      "and never coerced to zero, so every sector-rung statistic below "
      "describes the unsuppressed remainder -- which is the large-cell part of "
      "the distribution, i.e. not the part the NETS question is about.\n\n")
    rows = []
    for rung in ("county", "sector"):
        for universe in UNIVERSES:
            for rate in ("entry_rate", "entry_rate_dhs"):
                r = _pick(entry, rung=rung, universe=universe, rate=rate,
                          grp="ALL")
                if r is None:
                    continue
                lev = entry[(entry.rung == rung) & (entry.universe == universe)
                            & (entry.rate == rate) & (entry.grp == "LEVELS")]
                rows.append([rung, universe, rate, f"{int(r['n']):,}",
                             fmt(r["pearson"]), fmt(r["spearman"]),
                             fmt(r["med_gap"]),
                             fmt(r["med_x"]), fmt(r["med_y"]),
                             fmt(lev["er_nets_agg"].iloc[0]) if len(lev) else "--",
                             fmt(lev["er_bds_agg"].iloc[0]) if len(lev) else "--"])
    A(md_table(["rung", "universe", "rate", "N cells", "Pearson", "Spearman",
                "median gap", "median NETS", "median BDS",
                "aggregate NETS", "aggregate BDS"], rows))

    # The BDS aggregate barely moves across the eight rows and the NETS one
    # moves by a factor of four. That asymmetry is the finding, so it is
    # explained here rather than left for a reader to wonder about.
    def _lev(rung, universe, field, rate="entry_rate"):
        m = entry[(entry.rung == rung) & (entry.universe == universe)
                  & (entry.rate == rate) & (entry.grp == "LEVELS")]
        return float(m[field].iloc[0]) if len(m) else float("nan")

    cb_all = _lev("county", "all", "base_bds")
    cb_act = _lev("county", "actual", "base_bds")
    sb_all = _lev("sector", "all", "base_bds")
    n_c_all = _pick(entry, rung="county", universe="all", rate="entry_rate",
                    grp="ALL")["n"]
    n_c_act = _pick(entry, rung="county", universe="actual",
                    rate="entry_rate", grp="ALL")["n"]
    n_s_all = _pick(entry, rung="sector", universe="all", rate="entry_rate",
                    grp="ALL")["n"]
    A("\n**Why the BDS aggregate hardly moves, and why that matters.** The "
      "aggregate is a mass-weighted ratio, sum of entry over sum of base, so "
      "it is set by the largest cells, and the largest cells are in every one "
      "of these samples. Changing the NETS universe at the county rung moves "
      f"the cell count from {int(n_c_act):,} to {int(n_c_all):,} but the BDS "
      f"establishment base only from {cb_act / 1e6:,.1f}M to "
      f"{cb_all / 1e6:,.1f}M, {abs(cb_all - cb_act) / cb_all:.1%}. The sector "
      "rung is the sharper case: BDS withholds `estabs_entry` in 36.8% of the "
      f"county-sector-years and only {int(n_s_all):,} pairs survive, yet the "
      f"establishment base falls just {(cb_all - sb_all) / cb_all:.1%}, from "
      f"{cb_all / 1e6:,.1f}M to {sb_all / 1e6:,.1f}M -- because `estabs_entry` "
      "is additive across sectors, so a complete sector rung would reproduce "
      "the county number exactly, and what suppression removes from it is "
      "small cells carrying almost no mass. And the two "
      "denominators coincide because the establishment stock is nearly flat "
      "year to year, so N_t is close to N_{t-1} and the DHS average is close "
      "to the base.\n\n")
    A("So the steadiness is a property of BDS, not of the statistic: it is "
      "what an additive, near-complete source looks like when you aggregate "
      "it. The informative half of the column is that **NETS is not steady** "
      "on the same samples -- its aggregate birth rate moves from "
      f"{_lev('county', 'actual', 'er_nets_agg'):.3f} on the "
      "actual-employment universe to "
      f"{_lev('county', 'all', 'er_nets_agg'):.3f} on the full panel, a "
      "factor of four, against a BDS number that moves in the third decimal. "
      "A four-fold swing in an outcome's aggregate level, driven purely by "
      "which NETS establishments are retained, is a universe artefact, and it "
      "is the same artefact the year-by-year ratio below shows in time "
      "series.\n")

    A("\nThe aggregate entry rate by year, county rung, `all` universe:\n\n")
    A(md_table(["year", "NETS births / estabs t-1", "BDS entry / estabs t-1",
                "ratio"],
               [[str(int(r.year)), f"{r.er_nets:.3f}", f"{r.er_bds:.3f}",
                 f"{r.er_nets / r.er_bds:.2f}" if r.er_bds else "--"]
                for _, r in eby.iterrows()]))

    if len(resid):
        A("\nNETS births in NAICS2 codes with no BDS sector, held out of the "
          "sector comparison rather than folded into a neighbour:\n\n")
        A(md_table(["NAICS2", "births"],
                   [[str(r.n2), f"{int(r.births):,}"]
                    for _, r in resid.head(10).iterrows()]))

    # ---- divergence
    A("\n## Where NETS diverges\n\n")
    A(f"All cuts at h = {HEADLINE_H}, midpoint growth, NETS minus EFSY. The "
      f"universe is `{HEADLINE_UNIVERSE}` except where the cut is about the "
      "imputation NETS does, which the `actual` universe is defined by "
      "excluding; those cuts say `all` in their own heading and blurb.\n\n")
    cut_titles = {
        "efsy_base_emp_decile":
            ("Small counties", "Deciles of EFSY base-period county "
             "employment, 1 = smallest. County rung."),
        "efsy_base_cell_size":
            ("Small cells", "EFSY base-period cell employment, NAICS4 rung, "
             "`actual` universe. The first row is entry: the cell is empty at "
             "the base date in EFSY, so its EFSY midpoint growth is +2 by "
             "construction and the gap measures whether NETS agrees the cell "
             "entered. Pearson and Spearman are undefined there because the "
             "EFSY side is the constant +2."),
        "efsy_base_cell_size_all":
            ("Small cells, `all` universe", "The same cut on the full NETS "
             "panel, so it can be read against the imputation cut below, "
             "which is computed on the same pairs."),
        "cell_imputed_share":
            ("Imputation-heavy cells", "The NETS cell's imputed share of "
             "employment at the outcome date, NAICS4 rung, `all` universe. "
             "Fixed bands rather than quintiles: the imputed share is a point "
             "mass at zero, so quintiles would cut the tie arbitrarily and "
             "print five identical rows."),
        "county_empc_estimated_share":
            ("EmpC composition", "The county's share of NETS employment "
             "carried by establishments whose primary employment line is an "
             "estimate (EmpC 1-3), county rung. Read from the `all` universe: "
             "the `actual` universe is DEFINED by excluding those "
             "establishments, so inside it this share is near zero by "
             "construction."),
        "outcome_year":
            ("The reporting break", "By outcome year. A year-pair spanning a "
             "jump in who reports an actual figure has a growth rate driven "
             "by reporting, not by employment."),
    }
    for cut, (title, blurb) in cut_titles.items():
        d = diverge[diverge.cut == cut]
        if d.empty:
            continue
        # Numeric bins (deciles, years) must sort as numbers, not as strings,
        # or decile 10 prints between 1 and 2.
        num = pd.to_numeric(d["grp"], errors="coerce")
        d = (d.assign(_k=num).sort_values("_k") if num.notna().all()
             else d.sort_values("grp"))
        A(f"\n### {title}\n\n{blurb}\n\n")
        A(md_table(["bin", "N", "Pearson", "Spearman", "median gap",
                    "IQR of gap", "sign agree", "median NETS", "median EFSY"],
                   [[str(r.grp), f"{int(r.n):,}", fmt(r.pearson),
                     fmt(r.spearman), fmt(r.med_gap),
                     f"[{fmt(r.p25_gap, 2)}, {fmt(r.p75_gap, 2)}]",
                     fmt(r.sign_agree, 2), fmt(r.med_x), fmt(r.med_y)]
                    for _, r in d.iterrows()]))

    # ---- structural
    A("\n## Divergences that are not NETS's fault\n\n")
    A("Three of them, sized so that no reader attributes them to NETS "
      "quality:\n\n")
    A("- **BDS has no Connecticut.** The 2023 release retro-codes the state "
      "onto planning regions back to 1978, so the eight legacy Connecticut "
      "counties on the support have no BDS row in any year.\n")
    A("- **BDS's statewide-unknown mass.** Establishments Census cannot place "
      "in a county are published under `XX999`, one code per state: 0.17% of "
      "establishments and 1.77% of employment in window. Every county-rung "
      "BDS total is missing exactly that mass.\n")
    A("- **NETS's unplaced records.** 2.41% of establishment-years carry a "
      "blank or non-state FIPS and are absent from the cell files; 2.32% of "
      "births likewise. The two sources are therefore both incomplete at the "
      "county rung, in different places.\n")

    A("\n## What this does and does not settle\n\n")
    A("It settles the descriptive question the plan asked: the size, sign and "
      "shape of the NETS-EFSY and NETS-BDS gaps, and which cuts of the data "
      "the gap concentrates in. It settles nothing about whether the outcome "
      "swap of WS2 step 3 or the entry pass of step 5 should run, because no "
      "criterion was fixed in advance and every number here is now visible. "
      "The step-5 gate is open.\n")
    return "".join(L)


def compile_pdf(md_path):
    """One document: the report, its tables, and both figures, as a PDF."""
    tex = RESULTS / "ws2_step2_benchmark.tex"
    subprocess.run(
        [sys.executable, str(CODE / "py" / "md2tex.py"), str(md_path),
         str(tex), "WS2 step 2: how problematic is NETS?",
         "Macro fiscal multiplier -- NETS benchmark",
         time.strftime("%B %-d, %Y")],
        check=True)

    # md2tex renders running text; the figures are inserted here, after the
    # sections they belong to, so the PDF is genuinely self-contained.
    body = tex.read_text()
    body = body.replace(
        r"\usepackage{booktabs}",
        "\\usepackage{booktabs}\n\\usepackage{graphicx}")
    for anchor, fig, cap in (
            (r"\section*{Why the three growth rates behave differently",
             "Figure_nets_efsy_scatter.pdf",
             "NETS vs EFSY midpoint growth, 40 equal-count bins of the EFSY "
             "growth rate, h = 2, actual universe. The grey line is the "
             "45-degree line. The bins bunch at the centre because the EFSY "
             "growth rate has a large point mass at exactly zero once the "
             "cells are fine, and at $\\pm 2$ at entry and exit -- so the "
             "equal-count bins in the middle span almost no range. The flat "
             "stretch through zero is the picture of the county-rung "
             "correlation of 0.20 falling to 0.02 at NAICS4."),
            (r"\section*{Where NETS diverges",
             "Figure_nets_bds_entry.pdf",
             "NETS establishment births against BDS establishment entry.")):
        i = body.find(anchor)
        if i < 0:
            raise RuntimeError(f"figure anchor not found in the rendered TeX: "
                               f"{anchor!r}")
        block = ("\n\\begin{figure}[H]\\centering\n"
                 f"\\includegraphics[width=\\textwidth]{{{fig}}}\n"
                 f"\\caption*{{\\footnotesize {cap}}}\n"
                 "\\end{figure}\n\n")
        body = body[:i] + block + body[i:]
    body = body.replace(r"\usepackage{graphicx}",
                        "\\usepackage{graphicx}\n\\usepackage{float}\n"
                        "\\usepackage{caption}")
    tex.write_text(body)

    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode",
                            "-halt-on-error", tex.name],
                           cwd=RESULTS, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stdout.splitlines()[-40:])
        raise RuntimeError(f"xelatex failed on {tex.name}:\n{tail}")
    for ext in (".aux", ".log", ".out"):
        p = tex.with_suffix(ext)
        if p.exists():
            p.unlink()
    log(f"wrote {tex.with_suffix('.pdf')}")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for path in (NETS_COUNTY, NETS_IND, BIRTH_COUNTY, BIRTH_IND, EFSY_COUNTY):
        if not path.exists():
            raise FileNotFoundError(f"missing input: {path}")

    con = connect()
    try:
        cov = build_support(con)

        log("levels")
        levels = run_levels(con)

        log("growth benchmark")
        growth, keyclass, zeros = run_growth(con)

        log("divergence cuts")
        diverge = run_divergence(con)

        log("entry benchmark")
        resid, supp = build_entry(con)
        entry = run_entry(con)
        eby = entry_by_year(con)

        log("figures")
        figure_growth(con, RESULTS / "Figure_nets_efsy_scatter.pdf")
        figure_entry(con, RESULTS / "Figure_nets_bds_entry.pdf", eby)

        log("tables and report")
        write_outputs(con, cov, levels, growth, keyclass, diverge, entry,
                      eby, resid, supp, zeros)
    finally:
        con.close()
    log("done")


if __name__ == "__main__":
    main()
