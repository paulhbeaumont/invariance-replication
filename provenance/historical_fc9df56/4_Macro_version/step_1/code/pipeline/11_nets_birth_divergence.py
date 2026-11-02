"""Stage 11: WS2 step 2b -- where the NETS/BDS birth gap comes from.

Producer 10 established THAT NETS births are not BDS entry: correlation 0.14 at
the county rung, an aggregate rate that swings by a factor of four with the
NETS universe, and a year-by-year ratio running from 0.55 to 3.07. This stage
asks the three follow-up questions, and nothing else:

  Q2.   WHERE does the gap live? Three orthogonal cuts of the same pairs --
        BDS sector, county size, and cell size -- each reported with BDS as the
        reference on both axes of the figure and in the sign of every gap.

  Q3c.  IS the year-to-year swing entry or imputation? The birth flow is split
        by the primary line's EmpC code, so a burst that is D&B back-filling an
        establishment it just discovered (EmpC 2) is separated from one that is
        Walls's imputation model (EmpC 3), from a range bottom (EmpC 1), and
        from an establishment that actually reported a figure (EmpC 0). The
        `actual` / `all` split of producer 10 cannot do this: it is
        `any_actual_emp`, a two-way cut, and it pools D&B with Walls.

  LINES. NETS against EFSY employment growth as a time series rather than a
        binscatter, because producer 10's Figure_nets_efsy_scatter bins ON EFSY
        and so plots a conditional mean, which collapses toward the
        unconditional mean whenever the correlation is near zero. A line chart
        of each source's own aggregate and median growth shows the marginal
        behaviour the binscatter cannot.

WHAT IS INHERITED, NOT RE-DECIDED. The support, the universes, the growth rate,
the entry-rate estimand and the agreement statistics are IMPORTED from producer
10 and producer 77; nothing here re-cuts a sample or redefines a rate. In
particular:

  - the support is producer 10's `build_support`: the intersection of the
    locked 3,058 ccgs, the NETS ccgs and the EFSY ccgs (gate Q7), further
    intersected with BDS for every entry exhibit;
  - the entry rate is  E_t / N_{t-1}, rebuilt from counts on BOTH sides, which
    is producer 10's `build_entry` verbatim;
  - the growth rate is midpoint (gate Q6), nothing winsorized, no cell dropped
    for being small or zero;
  - the EmpC split rides producer 77's birth file, which `_checks` there proves
    re-collapses to producer 75's counts exactly.

NO PASS CRITERION, for the same reason producer 10 has none: this is
descriptive, it does not gate WS2 step 5, and every number producer 10 would
have been judged against is already visible.

WHAT THE BDS SUPPRESSION DOES TO THE SECTOR CUT. BDS withholds `estabs_entry`
in 36.8% of county-sector-years, concentrated in small cells. Every
sector-rung and cell-size-rung number below is therefore computed on the
unsuppressed remainder, and the suppression share is printed BESIDE each row
rather than in a footnote, because the cut is about small cells and the missing
cells are the small ones.

Outputs, all under `Results/nets_benchmark/`:

  Table_ws2b_birth_gap_sector.tex     the gap by BDS sector
  Table_ws2b_birth_gap_size.tex       the gap by county size and by cell size
  Table_ws2b_birth_empc.tex           the birth flow by EmpC class, by year
  Figure_ws2b_birth_gap_bds.pdf       the three cuts, BDS on the x axis
  Figure_ws2b_birth_empc.pdf          the EmpC decomposition of the swing
  Figure_ws2b_growth_lines.pdf        NETS vs EFSY growth, as time series
  ws2b_birth_gap_cuts.csv             one row per printed cut
  ws2b_birth_empc.csv                 one row per year x EmpC class
  ws2b_growth_by_year.csv             one row per year x source x universe
  ws2_step2b_divergence_report.md     the document
  ws2_step2b_divergence.tex/.pdf

Run: `.venv/bin/python step_1/code/pipeline/11_nets_birth_divergence.py`
"""
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Producer 10 is the definition of the sample and of every statistic reused
# here; producer 77 is the EmpC birth file. Both are imported rather than
# re-implemented so this stage cannot drift from them.
p10 = _load("nets_benchmark", PIPELINE / "10_nets_benchmark.py")
p77 = _load("nets_births_empc", CODE / "py" / "77_nets_births_empc.py")

RESULTS = p10.RESULTS
YEAR_LO, YEAR_HI = p10.YEAR_LO, p10.YEAR_HI
UNIVERSES = p10.UNIVERSES
HEADLINE_H = p10.HEADLINE_H
SECTOR_RANGES = p10.SECTOR_RANGES
BDS_SECTORS = p10.BDS_SECTORS

BIRTHS_EMPC = p77.OUT
NETS_COUNTY = p10.NETS_COUNTY
EFSY_COUNTY = p10.EFSY_COUNTY
EMPC_LABEL = p77.EMPC_LABEL

# The EmpC codes, in the order every table and figure prints them. -1 is the
# establishment-year whose primary line matched no employment record; it is a
# class of its own and is never folded into a real code.
EMPC_CODES = (0, 1, 2, 3, -1)
EMPC_COLOR = {0: "#1f4e79", 1: "#7a9cc6", 2: "#8b1a1a", 3: "#d98c00",
              -1: "0.6"}

# Horizons for the growth line chart. h = 1 is added to the gate's h = 2
# because a line chart of annual growth is what the question asks for; both are
# plotted, neither replaces the benchmark's h.
LINE_HORIZONS = (1, 2)

SECTOR_NAME = {
    "11": "Agriculture", "21": "Mining", "22": "Utilities",
    "23": "Construction", "31-33": "Manufacturing", "42": "Wholesale",
    "44-45": "Retail", "48-49": "Transport", "51": "Information",
    "52": "Finance", "53": "Real estate", "54": "Professional",
    "55": "Management", "56": "Administrative", "61": "Education",
    "62": "Health", "71": "Arts", "72": "Accommodation", "81": "Other svcs",
}

log = p10.log
fmt = p10.fmt
tabular = p10.tabular


def sector_case(n2="naics2"):
    """NAICS2 -> BDS sector, written against a column name."""
    return ("CASE " + " ".join(
        f"WHEN {n2} = '{k}' THEN '{v}'" for k, v in SECTOR_RANGES.items()
    ) + f" WHEN {n2} IN (" + ", ".join(
        f"'{s}'" for s in BDS_SECTORS if "-" not in s
    ) + f") THEN {n2} ELSE NULL END")


# ------------------------------------------------------------------- Q2

AGG_COLS = """
    count(*)                                    AS n_cells,
    sum(entry_n)                                AS births_nets,
    sum(entry_b)                                AS entry_bds,
    sum(base_n)                                 AS base_nets,
    sum(base_b)                                 AS base_bds,
    sum(entry_n) / nullif(sum(base_n), 0)       AS er_nets,
    sum(entry_b) / nullif(sum(base_b), 0)       AS er_bds,
    median(er_n - er_b)                         AS med_gap,
    corr(er_n, er_b)                            AS pearson
"""


def _finish(d, tot_births, tot_entry, tot_base_b):
    """Decompose the aggregate rate gap over the cut's groups.

    The aggregate rate on each side is a base-weighted average of the group
    rates,  r = sum_g s_g r_g,  with s_g the group's share of that source's own
    establishment base. So the gap between the two aggregates splits exactly:

        r^N - r^B = sum_g s^B_g (r^N_g - r^B_g)   [`within`]
                  + sum_g (s^N_g - s^B_g) r^N_g   [`compos`]

    `within` is the part that comes from NETS and BDS disagreeing about the
    birth RATE inside a group weighted by how big that group is in BDS.
    `compos` is the part that comes from NETS's establishment base sitting in
    different groups than BDS's -- a scale-and-mix effect, not a rate
    disagreement. The two columns are additive across groups and sum to the
    aggregate gap, which the caller asserts. This is the decomposition to read
    for WHERE the gap originates; a raw common-denominator difference is not,
    because NETS carries 1.8 to 4.3 times BDS's establishment count and such a
    column would simply rank the groups by size.

    `share_births` / `share_entry` are the group's share of each source's own
    national birth flow. Their difference is a separate composition statement,
    about the FLOW rather than the base, and sums to zero across groups.
    """
    d = d.copy()
    d["gap_agg"] = d["er_nets"] - d["er_bds"]
    d["s_nets"] = d["base_nets"] / d["base_nets"].sum()
    d["s_bds"] = d["base_bds"] / tot_base_b
    d["within"] = d["s_bds"] * (d["er_nets"] - d["er_bds"])
    d["compos"] = (d["s_nets"] - d["s_bds"]) * d["er_nets"]
    d["contrib"] = d["within"] + d["compos"]
    d["share_births"] = d["births_nets"] / tot_births
    d["share_entry"] = d["entry_bds"] / tot_entry
    d["share_gap"] = d["share_births"] - d["share_entry"]
    return d


def cut_sector(con, universe):
    """The gap by BDS sector, with that sector's own suppression share."""
    d = con.execute(f"""
        SELECT sector AS grp, {AGG_COLS}
        FROM entry_sector
        WHERE universe = '{universe}' AND er_n IS NOT NULL
          AND er_b IS NOT NULL AND isfinite(er_n)
        GROUP BY 1 ORDER BY 1
    """).df()
    supp = con.execute(f"""
        SELECT sector AS grp,
               avg(CASE WHEN estabs_entry IS NULL THEN 1.0 ELSE 0 END)
                   AS share_supp
        FROM read_parquet('{dc.bds_path("bds_ccg_sector_year.parquet")}')
        WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
          AND ccg IN (SELECT ccg FROM support_bds)
        GROUP BY 1
    """).df()
    d = d.merge(supp, on="grp", how="left")
    d["label"] = d["grp"].map(SECTOR_NAME).fillna(d["grp"])
    return d


def cut_county_size(con, universe):
    """The gap by decile of the BDS establishment base -- 'small counties'.

    The decile is cut on the BDS base, not the NETS one, because BDS is the
    reference: a decile defined on NETS would move with the NETS universe and
    the cuts would not be comparable across the two universes.
    """
    return con.execute(f"""
        WITH d AS (
            SELECT *, ntile(10) OVER (ORDER BY base_b) AS decile
            FROM entry_county
            WHERE universe = '{universe}' AND er_n IS NOT NULL
              AND er_b IS NOT NULL AND isfinite(er_n)
        )
        SELECT CAST(decile AS VARCHAR) AS grp, {AGG_COLS}
        FROM d GROUP BY 1 ORDER BY CAST(grp AS INTEGER)
    """).df().assign(label=lambda t: t["grp"])


def cut_cell_size(con, universe):
    """The gap by county x sector cell size -- 'small cells'.

    Bins on the BDS establishment base at t-1, numbered so they sort in size
    order. This is the cut BDS suppression bites hardest, which is why the
    suppression share is carried into the table beside it.
    """
    bins = """CASE WHEN base_b < 2 THEN '1. 1'
                   WHEN base_b < 5 THEN '2. 2-4'
                   WHEN base_b < 10 THEN '3. 5-9'
                   WHEN base_b < 50 THEN '4. 10-49'
                   WHEN base_b < 250 THEN '5. 50-249'
                   ELSE '6. 250+' END"""
    return con.execute(f"""
        SELECT {bins} AS grp, {AGG_COLS}
        FROM entry_sector
        WHERE universe = '{universe}' AND er_n IS NOT NULL
          AND er_b IS NOT NULL AND isfinite(er_n)
        GROUP BY 1 ORDER BY 1
    """).df().assign(label=lambda t: t["grp"])


def run_cuts(con):
    """The three cuts, both universes, plus the additivity assertion."""
    out = []
    for universe in UNIVERSES:
        for name, fn, rung in (("bds_sector", cut_sector, "sector"),
                               ("county_size_decile", cut_county_size, "county"),
                               ("cell_size", cut_cell_size, "sector")):
            d = fn(con, universe)
            tot_b = d["births_nets"].sum()
            tot_e = d["entry_bds"].sum()
            tot_base = d["base_bds"].sum()
            d = _finish(d, tot_b, tot_e, tot_base)

            # The groups must be a PARTITION of the cut's own pair sample:
            # every total recomputed ungrouped must match the sum over groups.
            # This is the check that catches a dropped or duplicated group;
            # the additivity of `contrib` given those totals is arithmetic.
            tbl = "entry_county" if rung == "county" else "entry_sector"
            ref = con.execute(f"""
                SELECT count(*) AS n_cells, sum(entry_n) AS births_nets,
                       sum(entry_b) AS entry_bds, sum(base_n) AS base_nets,
                       sum(base_b) AS base_bds
                FROM {tbl}
                WHERE universe = '{universe}' AND er_n IS NOT NULL
                  AND er_b IS NOT NULL AND isfinite(er_n)
            """).df().iloc[0]
            for col in ("n_cells", "births_nets", "entry_bds", "base_nets",
                        "base_bds"):
                if abs(d[col].sum() - ref[col]) > 1e-6:
                    raise RuntimeError(
                        f"{name}/{universe}: the groups are not a partition -- "
                        f"{col} sums to {d[col].sum():,.6g} over groups against "
                        f"{ref[col]:,.6g} ungrouped")
            agg_gap = tot_b / d["base_nets"].sum() - tot_e / tot_base
            if abs(d["contrib"].sum() - agg_gap) > 1e-10:
                raise RuntimeError(
                    f"{name}/{universe}: within + composition sums to "
                    f"{d['contrib'].sum():.6g}, not the aggregate rate gap "
                    f"{agg_gap:.6g}")
            d.insert(0, "aggregate_gap", agg_gap)
            d.insert(0, "rung", rung)
            d.insert(0, "universe", universe)
            d.insert(0, "cut", name)
            out.append(d)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ Q3c

def run_empc(con):
    """The birth flow by primary-line EmpC class, by year, county rung.

    Two rates, and they answer different questions:

      contrib_k,t = Births_{k,t} / N_{t-1}     ADDITIVE across k, sums to the
                                               aggregate birth rate producer 10
                                               reports; this is the series whose
                                               swing is being decomposed.
      rate_k,t    = Births_{k,t} / N_{k,t-1}   the within-class birth rate: how
                                               fast the class's own stock is
                                               being added to.

    The denominators come from the NETS stock file's own EmpC counts, so
    numerator and denominator are the same classification of the same
    establishments. `all` universe throughout: the `actual` universe is defined
    by excluding EmpC-estimated establishments, so inside it three of the five
    classes are empty by construction.
    """
    codes = ", ".join(str(c) for c in EMPC_CODES if c >= 0)
    stock = con.execute(f"""
        SELECT year, ccg,
               {', '.join(f'coalesce(n_empc{k}, 0) AS n_{k}' for k in range(4))},
               coalesce(n_empc_missing, 0) AS n_m,
               n_estab
        FROM read_parquet('{NETS_COUNTY}')
        WHERE universe = 'all' AND year BETWEEN {YEAR_LO - 1} AND {YEAR_HI}
          AND ccg IN (SELECT ccg FROM support_bds)
    """).df()
    stock_long = stock.melt(id_vars=["year", "ccg", "n_estab"],
                            value_vars=[f"n_{k}" for k in range(4)] + ["n_m"],
                            var_name="k", value_name="base_k")
    stock_long["emp_code"] = stock_long["k"].map(
        {**{f"n_{k}": k for k in range(4)}, "n_m": -1})
    con.register("stock_empc", stock_long[["year", "ccg", "emp_code",
                                           "base_k", "n_estab"]])

    d = con.execute(f"""
        WITH b AS (
            SELECT year, emp_code, sum(n_births) AS births
            FROM read_parquet('{BIRTHS_EMPC}')
            WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
              AND ccg IN (SELECT ccg FROM support_bds)
            GROUP BY 1, 2
        ), s AS (
            SELECT year + 1 AS year, emp_code, sum(base_k) AS base_k
            FROM stock_empc GROUP BY 1, 2
        ), tot AS (
            SELECT year + 1 AS year, sum(n_estab) AS base_all
            FROM (SELECT DISTINCT year, ccg, n_estab FROM stock_empc)
            GROUP BY 1
        )
        SELECT s.year, s.emp_code, coalesce(b.births, 0) AS births,
               s.base_k, tot.base_all,
               coalesce(b.births, 0) / nullif(tot.base_all, 0) AS contrib,
               coalesce(b.births, 0) / nullif(s.base_k, 0)     AS rate_k
        FROM s
        JOIN tot USING (year)
        LEFT JOIN b USING (year, emp_code)
        WHERE s.year BETWEEN {YEAR_LO + 1} AND {YEAR_HI}
          AND s.emp_code IN ({codes}, -1)
        ORDER BY s.year, s.emp_code
    """).df()
    d["label"] = d["emp_code"].map(EMPC_LABEL)

    # The contributions must sum to the aggregate birth rate of the same
    # sample; otherwise the classes are not a partition of the birth flow.
    tot = con.execute(f"""
        SELECT year, sum(n_births) AS births
        FROM read_parquet('{BIRTHS_EMPC}')
        WHERE year BETWEEN {YEAR_LO} AND {YEAR_HI}
          AND ccg IN (SELECT ccg FROM support_bds)
        GROUP BY 1
    """).df()
    chk = (d.groupby("year")[["births"]].sum().join(
        tot.set_index("year"), rsuffix="_tot"))
    bad = (chk["births"] - chk["births_tot"]).abs().max()
    if bad > 0:
        raise RuntimeError(f"EmpC classes do not partition the births: "
                           f"max |diff| = {bad}")
    return d


def empc_crosslag(d):
    """Lag-1 cross-correlations of the class contribution series.

    Entry (r, c) is corr(contrib_r at t, contrib_c at t-1), so the diagonal is
    each class's own AC(1) and the off-diagonal says whether one class's big
    year is followed by another's small one. This is the matrix that explains
    the sawtooth: the AC(1) of a SUM is not a weighted average of the
    components' AC(1)s -- it carries the cross-lag covariances too, so every
    class can be individually persistent while the total alternates.
    """
    w = d.pivot(index="year", columns="emp_code", values="contrib")
    cols = list(w.columns)
    return pd.DataFrame(
        {c: [w[r].iloc[1:].reset_index(drop=True)
             .corr(w[c].iloc[:-1].reset_index(drop=True)) for r in cols]
         for c in cols}, index=cols)


def empc_variance(d, bds):
    """Which class carries the year-to-year swing.

    `var_share` is cov(contrib_k, total) / var(total), the standard additive
    variance decomposition: it sums to one across classes and says what
    fraction of the movement in the aggregate birth rate each class accounts
    for. `ac1` is the first-order autocorrelation of the demeaned series: a
    NEGATIVE value is a sawtooth (a high year followed by a low one), which is
    a file-vintage signature, not a business cycle. `corr_bds` is against the
    BDS entry rate on the same years, the only series here that is a cycle.
    """
    wide = d.pivot(index="year", columns="emp_code", values="contrib")
    total = wide.sum(axis=1)
    b = bds.set_index("year")["er_bds"].reindex(total.index)
    rows = []
    for code in list(wide.columns) + ["TOTAL"]:
        s = total if code == "TOTAL" else wide[code]
        x = s - s.mean()
        ac1 = (np.corrcoef(x.iloc[1:], x.iloc[:-1])[0, 1]
               if x.std() > 0 else np.nan)
        dx = s.diff().dropna()
        rows.append({
            "emp_code": code,
            "label": "total" if code == "TOTAL" else EMPC_LABEL[code],
            "mean": s.mean(), "sd": s.std(),
            "sd_diff": dx.std(),
            "var_share": (np.cov(s, total)[0, 1] / np.var(total, ddof=1)
                          if code != "TOTAL" else 1.0),
            "ac1": ac1,
            "corr_bds": s.corr(b),
            "corr_total": s.corr(total),
        })
    bx = b - b.mean()
    rows.append({"emp_code": "BDS", "label": "BDS entry rate",
                 "mean": b.mean(), "sd": b.std(), "sd_diff": b.diff().std(),
                 "var_share": np.nan,
                 "ac1": np.corrcoef(bx.iloc[1:], bx.iloc[:-1])[0, 1],
                 "corr_bds": 1.0, "corr_total": b.corr(total)})
    return pd.DataFrame(rows)


# -------------------------------------------------------- growth by year

def growth_by_year(con):
    """NETS and EFSY midpoint growth, county rung, as time series.

    Two statistics per source-year, because they answer different questions
    and producer 10's binscatter answers neither:

      aggregate  2 (sum E_t - sum E_{t-h}) / (sum E_t + sum E_{t-h}), the
                 growth of the national total on this support -- the number an
                 aggregation test is about;
      median     the median county's own midpoint growth, and the IQR, which
                 is the marginal DISPERSION the binscatter's conditional means
                 cannot show.

    EFSY does not depend on the NETS universe, so its series is identical in
    both universe blocks; it is emitted once per block anyway, so each block
    can be read on its own.
    """
    out = []
    for universe in UNIVERSES:
        p10.build_cells(con, "county", universe)
        for h in LINE_HORIZONS:
            p10.build_pairs(con, h)
            d = con.execute("""
                SELECT year,
                       count(*) AS n,
                       2 * (sum(en_t) - sum(en_b))
                         / nullif(sum(en_t) + sum(en_b), 0) AS agg_nets,
                       2 * (sum(ee_t) - sum(ee_b))
                         / nullif(sum(ee_t) + sum(ee_b), 0) AS agg_efsy,
                       median(mp_n) AS med_nets,
                       median(mp_e) AS med_efsy,
                       quantile_cont(mp_n, 0.25) AS p25_nets,
                       quantile_cont(mp_n, 0.75) AS p75_nets,
                       quantile_cont(mp_e, 0.25) AS p25_efsy,
                       quantile_cont(mp_e, 0.75) AS p75_efsy,
                       stddev_samp(mp_n) AS sd_nets,
                       stddev_samp(mp_e) AS sd_efsy
                FROM g WHERE mp_n IS NOT NULL AND mp_e IS NOT NULL
                GROUP BY 1 ORDER BY 1
            """).df()
            d.insert(0, "h", h)
            d.insert(0, "universe", universe)
            out.append(d)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------- figures

def figure_cuts(path, cuts, universe="all"):
    """The three cuts, BDS on the x axis in every panel.

    BDS is the reference: the 45-degree line is where NETS would sit if it
    measured what BDS measures, and vertical distance from it is the gap. Point
    area is the group's share of BDS entry, so a group that is far from the
    line but carries no mass cannot be mistaken for the explanation.
    """
    d = cuts[cuts.universe == universe]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    panels = (("bds_sector", "By BDS sector"),
              ("county_size_decile", "By county size decile\n(BDS establishment base)"),
              ("cell_size", "By cell size\n(BDS establishments at $t-1$)"))
    for ax, (cut, title) in zip(axes, panels):
        c = d[d.cut == cut]
        lim = max(float(np.nanmax(c[["er_nets", "er_bds"]].to_numpy())) * 1.12,
                  0.02)
        ax.plot([0, lim], [0, lim], color="0.6", lw=0.8, zorder=1)
        size = 18 + 320 * (c["share_entry"] / c["share_entry"].max())
        ax.scatter(c["er_bds"], c["er_nets"], s=size, alpha=0.75,
                   color="#8b1a1a", edgecolor="white", linewidth=0.5, zorder=3)
        for _, r in c.iterrows():
            ax.annotate(str(r["label"]).split(". ")[-1],
                        (r["er_bds"], r["er_nets"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=5.5,
                        color="0.25")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("BDS entry rate $E_t/N_{t-1}$", fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("NETS birth rate $B_t/N_{t-1}$", fontsize=8)
    fig.suptitle(f"Where the NETS/BDS birth gap lives ({universe} universe); "
                 "BDS is the reference on both axes, point area is the "
                 "group's share of BDS entry", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {path.name}")


def figure_empc(path, empc, bds, var):
    """The EmpC decomposition of the birth flow, against BDS."""
    wide = empc.pivot(index="year", columns="emp_code", values="contrib")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    ax = axes[0]
    ax.stackplot(wide.index,
                 *[wide[c].values for c in EMPC_CODES if c in wide],
                 labels=[EMPC_LABEL[c] for c in EMPC_CODES if c in wide],
                 colors=[EMPC_COLOR[c] for c in EMPC_CODES if c in wide],
                 alpha=0.9)
    ax.plot(bds["year"], bds["er_bds"], color="black", lw=1.4, ls="--",
            label="BDS entry rate")
    ax.set_title("Birth rate $B_{k,t}/N_{t-1}$, stacked by EmpC", fontsize=8.5)
    ax.legend(fontsize=6, frameon=False, loc="upper left")

    ax = axes[1]
    for c in EMPC_CODES:
        if c in wide:
            ax.plot(wide.index, wide[c], color=EMPC_COLOR[c], lw=1.2,
                    marker="o", ms=2.5, label=EMPC_LABEL[c])
    ax.plot(bds["year"], bds["er_bds"], color="black", lw=1.4, ls="--",
            label="BDS entry rate")
    ax.axhline(0, color="0.85", lw=0.6)
    ax.set_title("The same series, unstacked", fontsize=8.5)
    ax.legend(fontsize=6, frameon=False)

    ax = axes[2]
    v = var[~var.emp_code.isin(["TOTAL", "BDS"])].copy()
    v["_o"] = v.emp_code.map({c: i for i, c in enumerate(EMPC_CODES)})
    v = v.sort_values("_o")
    ax.bar(range(len(v)), v["var_share"],
           color=[EMPC_COLOR[c] for c in v.emp_code])
    ax.set_xticks(range(len(v)))
    ax.set_xticklabels([EMPC_LABEL[c] for c in v.emp_code], fontsize=6,
                       rotation=30, ha="right")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_title("Share of the variance of the aggregate\nbirth rate, "
                 "$\\mathrm{cov}(B_k, B)/\\mathrm{var}(B)$", fontsize=8.5)

    for ax in axes[:2]:
        ax.set_xlabel("year", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.suptitle("Q3c: is the NETS birth swing entry, or imputation? "
                 "(county rung, all universe)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {path.name}")


def figure_growth_lines(path, gby):
    """NETS vs EFSY growth as time series, the marginal the binscatter hides.

    THIS FIGURE IS THE WHOLE YEAR-BY-YEAR RECORD. It carries every series the
    section used to print as four tables -- aggregate, median with the IQR, and
    the cross-county SD, for both horizons and both NETS universes -- so the
    tables are not printed alongside it; `ws2b_growth_by_year.csv` is the
    numeric record for anyone who wants the digits. The SD row is the one that
    answers the dispersion question directly: it is the marginal width of each
    source's own growth distribution, which a binscatter of one source on the
    other cannot show at all.
    """
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.6), sharex=True)
    for j, h in enumerate(LINE_HORIZONS):
        for i, (stat, lab) in enumerate((
                ("agg", "Aggregate midpoint growth"),
                ("med", "Median county"),
                ("sd", "SD across counties"))):
            ax = axes[i][j]
            for universe, ls in (("actual", "-"), ("all", ":")):
                d = gby[(gby.h == h) & (gby.universe == universe)]
                ax.plot(d["year"], d[f"{stat}_nets"], color="#8b1a1a", ls=ls,
                        lw=1.3, marker="o", ms=2.5,
                        label=f"NETS ({universe})")
                if universe == "actual":
                    ax.plot(d["year"], d[f"{stat}_efsy"], color="#1f4e79",
                            lw=1.5, marker="s", ms=2.5, label="EFSY")
                    if stat == "med":
                        ax.fill_between(d["year"], d["p25_efsy"],
                                        d["p75_efsy"], color="#1f4e79",
                                        alpha=0.12, lw=0)
                        ax.fill_between(d["year"], d["p25_nets"],
                                        d["p75_nets"], color="#8b1a1a",
                                        alpha=0.12, lw=0)
            if stat != "sd":
                ax.axhline(0, color="0.85", lw=0.6)
            ax.set_title(f"{lab}, h = {h}", fontsize=8.5)
            ax.tick_params(labelsize=7)

            # The SD row's own scale problem is EFSY's 2017-18 break again, and
            # it is handled the same way: clip to the bulk, label what falls
            # outside. Its floor is 0, a SD being non-negative.
            if stat == "sd":
                cols = ["sd_nets", "sd_efsy"]
                inside = gby[gby.h == h][cols].to_numpy().ravel()
                hi = float(np.quantile(inside, 0.9)) * 1.8
                ax.set_ylim(0, hi)
                for _, r in gby[(gby.h == h)
                                & (gby.universe == "actual")].iterrows():
                    for col, color in (("sd_efsy", "#1f4e79"),
                                       ("sd_nets", "#8b1a1a")):
                        if r[col] > hi:
                            ax.annotate(f"{r[col]:.2f}", (r["year"], hi),
                                        textcoords="offset points",
                                        xytext=(0, -9), ha="center",
                                        fontsize=6, color=color)
                            ax.plot([r["year"]], [hi], marker="^", ms=4,
                                    color=color, clip_on=False)

            # EFSY's 2017 break (producer 10's year cut: median EFSY growth of
            # -0.66 in 2017 and -0.58 in 2018) is an order of magnitude larger
            # than everything else and flattens the whole series if it sets the
            # scale. The axis is clipped to the bulk and every point left
            # outside is labelled with its value, so nothing is hidden.
            if stat == "med":
                d = gby[(gby.h == h) & (gby.universe == "actual")]
                inside = pd.concat([
                    gby[(gby.h == h)][["med_nets"]].rename(
                        columns={"med_nets": "v"}),
                    d[["med_efsy"]].rename(columns={"med_efsy": "v"})])
                lo, hi = -0.25, max(0.25, inside["v"].max() * 1.4)
                ax.set_ylim(lo, hi)
                for _, r in d.iterrows():
                    for col, color in (("med_efsy", "#1f4e79"),
                                       ("med_nets", "#8b1a1a")):
                        if r[col] < lo:
                            ax.annotate(f"{r[col]:+.2f}", (r["year"], lo),
                                        textcoords="offset points",
                                        xytext=(0, 4), ha="center",
                                        fontsize=6, color=color)
                            ax.plot([r["year"]], [lo], marker="v", ms=4,
                                    color=color, clip_on=False)
    axes[2][0].set_xlabel("outcome year", fontsize=8)
    axes[2][1].set_xlabel("outcome year", fontsize=8)
    axes[0][0].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.suptitle("NETS vs EFSY county employment growth, midpoint rate, by "
                 "year. Bands on the middle row are the interquartile range "
                 "across counties. The middle and bottom rows are clipped to "
                 "the bulk and every off-scale point is labelled with its "
                 "value.", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {path.name}")


# ---------------------------------------------------------------- tables

def table_sector(cuts, universe="all"):
    d = cuts[(cuts.cut == "bds_sector") & (cuts.universe == universe)]
    d = d.sort_values("contrib", key=abs, ascending=False)
    body = [[r["label"], f"{int(r['n_cells']):,}",
             fmt(r["er_nets"]), fmt(r["er_bds"]), fmt(r["gap_agg"]),
             fmt(r["pearson"], 2), f"{r['share_births']:.1%}".replace("%", r"\%"),
             f"{r['share_entry']:.1%}".replace("%", r"\%"),
             fmt(r["within"], 4), fmt(r["compos"], 4),
             f"{r['share_supp']:.0%}".replace("%", r"\%")]
            for _, r in d.iterrows()]
    body.append([r"\textit{sum}", "", "", "", "", "", "", "",
                 fmt(d["within"].sum(), 4), fmt(d["compos"].sum(), 4), ""])
    return tabular(
        ["Sector", "$N$ cells", "NETS", "BDS", "gap", "$\\rho$",
         "\\% births", "\\% entry", "within", "compos.", "suppr."],
        body, spec="l" + "r" * 10)


def table_size(cuts, universe="all"):
    out = []
    for cut, head in (("county_size_decile", "Panel A: county size decile "
                       "(BDS establishment base, 1 = smallest)"),
                      ("cell_size", "Panel B: county $\\times$ sector cell "
                       "size (BDS establishments at $t-1$)")):
        d = cuts[(cuts.cut == cut) & (cuts.universe == universe)]
        body = [[str(r["label"]), f"{int(r['n_cells']):,}",
                 fmt(r["er_nets"]), fmt(r["er_bds"]), fmt(r["gap_agg"]),
                 fmt(r["pearson"], 2),
                 f"{r['share_births']:.1%}".replace("%", r"\%"),
                 f"{r['share_entry']:.1%}".replace("%", r"\%"),
                 fmt(r["within"], 4), fmt(r["compos"], 4)]
                for _, r in d.iterrows()]
        body.append([r"\textit{sum}", "", "", "", "", "", "", "",
                     fmt(d["within"].sum(), 4), fmt(d["compos"].sum(), 4)])
        out.append(head + "\n\n" + tabular(
            ["Group", "$N$ cells", "NETS", "BDS", "gap", "$\\rho$",
             "\\% births", "\\% entry", "within", "compos."],
            body, spec="l" + "r" * 9))
    return "\n\n\\vspace{1em}\n\n".join(out)


def table_empc(var):
    order = {c: i for i, c in enumerate(EMPC_CODES)}
    v = var.copy()
    v["_o"] = v.emp_code.map(lambda c: order.get(c, 99))
    v = v.sort_values("_o")
    body = [[r["label"], fmt(r["mean"]), fmt(r["sd"]), fmt(r["sd_diff"]),
             fmt(r["var_share"], 2) if np.isfinite(r["var_share"]) else "--",
             fmt(r["ac1"], 2), fmt(r["corr_bds"], 2), fmt(r["corr_total"], 2)]
            for _, r in v.iterrows()]
    return tabular(
        ["EmpC class", "mean", "SD", "SD of $\\Delta$", "var.\\ share",
         "AC(1)", "$\\rho$ BDS", "$\\rho$ total"],
        body, spec="l" + "r" * 7)


# ---------------------------------------------------------------- report

def build_report(cuts, empc, var, gby, bds, resid, supp):
    """The document.

    NO LaTeX MATH in the markdown: `md2tex.py` escapes `$`, `_` and `{`, so a
    dollar-delimited formula would set as literal text. Formulas therefore ride
    in code spans, which md2tex renders as \\code{...}, and the only symbols
    taken from unicode are the ones its table maps (x for times is written as
    the plain word).
    """
    L = []
    A = L.append
    A("# WS2 step 2b: where the NETS birth gap comes from, and what the "
      "growth series look like\n\n")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
      "`step_1/code/pipeline/11_nets_birth_divergence.py`, on producer 10's "
      "sample and with producer 10's estimands. Three questions, no new "
      "sample, no pass criterion.\n\n")
    A("Producer 10 established THAT NETS births are not BDS entry. This "
      "document asks where that gap comes from (Q2), whether the year-to-year "
      "swing in the NETS birth rate is entry or imputation (Q3c), and what "
      "the NETS and EFSY growth series look like as marginals rather than as "
      "a binscatter.\n\n")

    c_all = cuts[cuts.universe == "all"]
    sec = c_all[c_all.cut == "bds_sector"].sort_values("contrib")
    dec = c_all[c_all.cut == "county_size_decile"]
    cel = c_all[c_all.cut == "cell_size"]
    v = var.set_index("emp_code")
    tot_row = v.loc["TOTAL"]
    bds_row = v.loc["BDS"]

    def _top(d, col="contrib"):
        return d.reindex(d[col].abs().sort_values(ascending=False).index)

    # ---- headline findings, generated from the numbers
    A("## What the numbers say\n\n")

    sec_t = _top(sec)
    s1 = sec_t.iloc[0]
    A("- **By industry the gap is a composition gap, not a rate gap, and the "
      "sector terms largely cancel.** The aggregate birth-rate gap on this "
      f"sample is {sec['aggregate_gap'].iloc[0]:+.4f}, a net of sector terms "
      "an order of magnitude larger that offset each other: the largest single "
      f"contribution is {s1['label']} at {s1['contrib']:+.4f}, of which "
      f"{s1['compos']:+.4f} is composition and only {s1['within']:+.4f} is a "
      "rate disagreement inside the sector. NETS puts "
      f"{s1['share_births']:.1%} of its births in {s1['label']} against BDS's "
      f"{s1['share_entry']:.1%} of entry. Summed over all 19 sectors, "
      f"{sec['within'].sum():+.4f} of the gap is within-sector rate "
      f"disagreement and {sec['compos'].sum():+.4f} is composition.\n")

    d_lo, d_hi = dec.iloc[0], dec.iloc[-1]
    A("- **Small counties are not where the gap is.** In the smallest decile "
      f"of the BDS establishment base the NETS birth rate is "
      f"{d_lo['er_nets']:.3f} against BDS's {d_lo['er_bds']:.3f} "
      f"({d_lo['gap_agg']:+.3f}); in the largest, {d_hi['er_nets']:.3f} "
      f"against {d_hi['er_bds']:.3f} ({d_hi['gap_agg']:+.3f}). The sign even "
      "runs the wrong way for a small-county story -- NETS UNDER-counts births "
      "in small counties and over-counts them in large ones -- and the "
      f"smallest decile carries {d_lo['share_entry']:.1%} of BDS entry, so it "
      f"contributes {d_lo['contrib']:+.4f} against the largest decile's "
      f"{d_hi['contrib']:+.4f}.\n")

    c_lo, c_hi = cel.iloc[0], cel.iloc[-1]
    A("- **Small cells are where the gap is.** At the county-by-sector rung "
      f"the smallest surviving cells ({c_lo['label'].split('. ')[-1]} BDS "
      "establishments at t-1) give a NETS birth rate of "
      f"{c_lo['er_nets']:.3f} against BDS's {c_lo['er_bds']:.3f} "
      f"({c_lo['gap_agg']:+.3f}), while the largest "
      f"({c_hi['label'].split('. ')[-1]}) give "
      f"{c_hi['er_nets']:.3f} against {c_hi['er_bds']:.3f} "
      f"({c_hi['gap_agg']:+.3f}). That is the same phenomenon producer 10 "
      "found on the stock side, and it survives with BDS as the reference: "
      "the disagreement is about how fine the cell is, not about which "
      "industry or which county. It is also the cut BDS suppression bites: "
      f"{supp['share_supp']:.1%} of county-sector-years have `estabs_entry` "
      "withheld, they are the small ones, and they are absent from every row "
      "of that panel -- so the small-cell gap below is if anything "
      "understated.\n")

    dnb, walls, act = v.loc[2], v.loc[3], v.loc[0]
    A("- **The year-to-year swing is imputation, not entry.** Splitting the "
      "birth flow by the EmpC code of the establishment's primary employment "
      f"line, D&B's own estimate (EmpC 2) accounts for {dnb['var_share']:.0%} "
      "of the variance of the aggregate birth rate and Walls's estimate "
      f"(EmpC 3) for {walls['var_share']:.0%}, against {act['var_share']:.0%} "
      "for establishments that actually reported a figure (EmpC 0). Those two "
      "imputations are produced by different machinery and the `actual` / "
      "`all` split of producer 10 cannot separate them, because "
      "`any_actual_emp` pools them.\n")

    A("- **And it has the wrong autocorrelation sign to be a business "
      f"cycle.** The aggregate NETS birth rate has AC(1) = {tot_row['ac1']:+.2f} "
      "on the demeaned series -- negative, a sawtooth, a high year "
      "systematically followed by a low one -- while BDS entry over the same "
      f"counties and years has {bds_row['ac1']:+.2f}, the persistence a cycle "
      f"has. The two series correlate {tot_row['corr_bds']:+.2f}. A sawtooth "
      "is what a discovery process produces: D&B finds a batch of "
      "establishments, back-fills them with an estimate, and under the "
      "panel-first-appearance definition of a birth the whole batch lands in "
      "one year's cohort. It is not what an entry wave looks like. Any "
      "regression of a NETS birth outcome on a year-varying shock inherits "
      "the alternation directly.\n")

    gc = gby[(gby.h == 2) & (gby.universe == "actual")]
    ga = gby[(gby.h == 2) & (gby.universe == "all")]
    A("- **On growth, the flat binscatter is a correlation of near zero, not a "
      "NETS that barely moves.** At h = 2 on the headline `actual` universe "
      "the median annual cross-sectional SD of county midpoint growth is "
      f"{gc['sd_nets'].median():.3f} in NETS against "
      f"{gc['sd_efsy'].median():.3f} in EFSY -- comparable width, a ratio of "
      f"{gc['sd_nets'].median() / gc['sd_efsy'].median():.2f}. On the full "
      f"panel NETS IS narrower, {ga['sd_nets'].median():.3f} against the same "
      f"{ga['sd_efsy'].median():.3f}, a ratio of "
      f"{ga['sd_nets'].median() / ga['sd_efsy'].median():.2f}: pooling in the "
      "imputed establishments smooths the county aggregate rather than "
      "roughening it. Neither ratio is anything like the collapse the "
      "binscatter appears to show. What producer 10's "
      "`Figure_nets_efsy_scatter.pdf` shows is the near-zero correlation "
      "between them: it bins on EFSY and plots the conditional mean of NETS, "
      "and a conditional mean collapses onto the unconditional mean precisely "
      "when the correlation is near zero. That is why those points sit on a "
      "flat line at 0 instead of on the 45-degree line, and it is not "
      "evidence about dispersion. The line charts below plot each source's "
      "own marginal, and both move.\n\n")

    # ---- Q2
    A("## Q2. Where the birth gap lives\n\n")
    A("Every cut below is the same set of county (or county-by-sector) year "
      "pairs producer 10 built, partitioned three ways. BDS is the reference "
      "throughout: `gap` is NETS minus BDS, and the figure puts the BDS rate "
      "on the x axis in every panel, so the 45-degree line is agreement and "
      "vertical distance from it is the gap.\n\n")
    A("**How the gap is apportioned.** Each side's aggregate rate is a "
      "base-weighted average of its group rates, `r = sum_g s_g r_g` with "
      "`s_g` the group's share of THAT source's own establishment base. So "
      "the difference of the two aggregates splits exactly into\n\n")
    A("- `within  = sum_g s_bds_g (r_nets_g - r_bds_g)`, the part from the two "
      "sources disagreeing about the birth rate inside a group, weighted by "
      "how large the group is in BDS; and\n")
    A("- `compos. = sum_g (s_nets_g - s_bds_g) r_nets_g`, the part from "
      "NETS's establishment base sitting in different groups than BDS's -- a "
      "mix effect, not a rate disagreement.\n\n")
    A("Both columns are additive across groups and together they sum to the "
      "aggregate gap; every table prints the sum so this can be checked by "
      "eye. A raw common-denominator difference is deliberately not reported: "
      "NETS carries 1.8 to 4.3 times BDS's establishment count, so such a "
      "column would rank the groups by size and say nothing about the gap. "
      "`% births` and `% entry` are each source's own national birth flow, a "
      "separate composition statement about the FLOW rather than the base. "
      "`suppr.` is the share of that sector's county-sector-years in which "
      "BDS withholds `estabs_entry`.\n\n")

    A("### By BDS sector\n\n")
    sec_rows = _top(sec)
    A(_md_table(
        ["sector", "N cells", "NETS rate", "BDS rate", "gap", "corr",
         "% of NETS births", "% of BDS entry", "within", "compos.",
         "BDS suppressed"],
        [[r["label"], f"{int(r['n_cells']):,}", f"{r['er_nets']:.3f}",
          f"{r['er_bds']:.3f}", f"{r['gap_agg']:+.3f}", f"{r['pearson']:.2f}",
          f"{r['share_births']:.1%}", f"{r['share_entry']:.1%}",
          f"{r['within']:+.4f}", f"{r['compos']:+.4f}",
          f"{r['share_supp']:.0%}"] for _, r in sec_rows.iterrows()]
        + [["**sum**", "", "", "", "", "", "", "",
            f"{sec['within'].sum():+.4f}", f"{sec['compos'].sum():+.4f}", ""]]))
    A("\nSorted by the size of the total contribution. The sector rung is the "
      "BDS sector, 19 codes with three of them ranges of NAICS2, because BDS "
      "publishes no county-by-NAICS2 table; and every row is computed on the "
      "unsuppressed remainder of that sector's cells, which is why the "
      "suppression column belongs beside the estimates rather than in a "
      "footnote.\n\n")

    A("### By county size, decile of the BDS establishment base\n\n")
    A("The decile is cut on the BDS base, not the NETS one: a decile defined "
      "on NETS would move with the NETS universe and the two universes would "
      "not be comparable.\n\n")
    A(_md_table(
        ["decile", "N cells", "NETS rate", "BDS rate", "gap", "corr",
         "% of NETS births", "% of BDS entry", "within", "compos."],
        [[str(r["label"]), f"{int(r['n_cells']):,}", f"{r['er_nets']:.3f}",
          f"{r['er_bds']:.3f}", f"{r['gap_agg']:+.3f}", f"{r['pearson']:.2f}",
          f"{r['share_births']:.1%}", f"{r['share_entry']:.1%}",
          f"{r['within']:+.4f}", f"{r['compos']:+.4f}"]
         for _, r in dec.iterrows()]
        + [["**sum**", "", "", "", "", "", "", "",
            f"{dec['within'].sum():+.4f}", f"{dec['compos'].sum():+.4f}"]]))

    A("\n### By cell size, county by sector\n\n")
    A(_md_table(
        ["cell size", "N cells", "NETS rate", "BDS rate", "gap", "corr",
         "% of NETS births", "% of BDS entry", "within", "compos."],
        [[str(r["label"]), f"{int(r['n_cells']):,}", f"{r['er_nets']:.3f}",
          f"{r['er_bds']:.3f}", f"{r['gap_agg']:+.3f}", f"{r['pearson']:.2f}",
          f"{r['share_births']:.1%}", f"{r['share_entry']:.1%}",
          f"{r['within']:+.4f}", f"{r['compos']:+.4f}"]
         for _, r in cel.iterrows()]
        + [["**sum**", "", "", "", "", "", "", "",
            f"{cel['within'].sum():+.4f}", f"{cel['compos'].sum():+.4f}"]]))

    A("\n### The same three cuts on the `actual` universe\n\n")
    A("The `actual` universe keeps only births whose establishment reports an "
      "actual employment figure in its birth year. It is a different flow, "
      "not a subsample of the same one, and its aggregate rate is roughly a "
      "quarter of the full panel's -- so the levels below are not comparable "
      "to the tables above, only the shape of each cut is. The sector rows "
      "are in the CSV.\n\n")
    ca = cuts[cuts.universe == "actual"]
    A(_md_table(
        ["cut", "group", "NETS rate", "BDS rate", "gap", "within", "compos."],
        [[r["cut"], str(r["label"]), f"{r['er_nets']:.3f}",
          f"{r['er_bds']:.3f}", f"{r['gap_agg']:+.3f}",
          f"{r['within']:+.4f}", f"{r['compos']:+.4f}"]
         for _, r in ca[ca.cut != "bds_sector"].iterrows()]))

    A("\n**NETS births in NAICS2 codes that have no BDS sector**, held out of "
      "every sector and cell-size row above rather than folded into a "
      "neighbour. NAICS2 99 is the unclassified code and it is not small:\n\n")
    tot_births_all = c_all[c_all.cut == "county_size_decile"]["births_nets"].sum()
    A(_md_table(["NAICS2", "births", "% of NETS births on the support"],
                [[r["n2"], f"{int(r['births']):,}",
                  f"{r['births'] / tot_births_all:.1%}"]
                 for _, r in resid.iterrows()]))

    # ---- Q3c
    A("\n## Q3c. Is the year-to-year swing entry, or imputation?\n\n")
    A("The birth flow is split by the EmpC code of the DUNS line whose "
      "employment is the establishment's primary figure: 0 = an actual figure "
      "reported to D&B, 1 = the bottom of a reported range, 2 = D&B's own "
      "estimate, 3 = Walls and Associates' estimate, and `missing` for a "
      "birth year whose primary line matched no employment record. The split "
      "is producer 77's, which re-collapses to producer 75's counts "
      "exactly.\n\n")
    A("This is finer than the `actual` / `all` universes on purpose. `actual` "
      "is `any_actual_emp`: a two-way cut that pools D&B's estimate with "
      "Walls's. The two are produced by different machinery, and as the table "
      "shows they do not move together.\n\n")
    A("`B_k,t / N_t-1` uses the whole establishment stock as the denominator, "
      "so the five classes are additive and sum to the aggregate birth rate "
      "producer 10 reports. `share of variance` is "
      "`cov(B_k, B) / var(B)`, which sums to one across classes. `AC(1)` is "
      "the first-order autocorrelation of the demeaned series: negative is a "
      "sawtooth, positive is the persistence of a cycle.\n\n")
    A(_md_table(
        ["EmpC class", "mean rate", "SD", "SD of change", "share of variance",
         "AC(1)", "corr with BDS", "corr with total"],
        [[r["label"], f"{r['mean']:.4f}", f"{r['sd']:.4f}",
          f"{r['sd_diff']:.4f}",
          f"{r['var_share']:.2f}" if np.isfinite(r["var_share"]) else "--",
          f"{r['ac1']:+.2f}", f"{r['corr_bds']:+.2f}",
          f"{r['corr_total']:+.2f}"]
         for _, r in var.assign(
             _o=lambda t: t.emp_code.map(
                 lambda c: {**{c2: i for i, c2 in enumerate(EMPC_CODES)},
                            "TOTAL": 90, "BDS": 91}.get(c, 99)))
         .sort_values("_o").iterrows()]))

    cm = empc_crosslag(empc)
    A("\n**The mechanism.** Every class taken on its own has a positive "
      "AC(1), yet the total's is negative. There is no contradiction: the "
      "AC(1) of a sum is not a weighted average of its components', it also "
      "carries the CROSS-lag covariances. The table below is "
      "`corr(class in row at t, class in column at t-1)`, so the diagonal is "
      "each class's own AC(1) and the off-diagonal is one class's big year "
      "followed by another's. The decisive entry is Walls at t against D&B at "
      f"t-1, {cm.loc[3, 2]:+.2f}: a year in which D&B back-fills a large "
      "cohort of discovered establishments is followed by a year in which "
      "Walls's engine estimates a small one. That single cross term is large "
      "enough to cancel D&B's own persistence, and the aggregate alternates "
      "as the two imputation engines take turns on the same discovery "
      "flow.\n\n")
    A(_md_table(
        ["at t \\ at t-1"] + [EMPC_LABEL[c] for c in EMPC_CODES
                              if c in cm.columns],
        [[EMPC_LABEL[r]] + [f"{cm.loc[r, c]:+.2f}"
                            for c in EMPC_CODES if c in cm.columns]
         for r in EMPC_CODES if r in cm.index]))

    A("\nThe series themselves, county rung, `all` universe. Each column is "
      "that class's births over the whole establishment stock at t-1, so the "
      "row sums to `total`:\n\n")
    wide = empc.pivot(index="year", columns="emp_code", values="contrib")
    b = bds.set_index("year")["er_bds"]
    A(_md_table(
        ["year"] + [EMPC_LABEL[c] for c in EMPC_CODES if c in wide]
        + ["total", "BDS entry"],
        [[str(y)] + [f"{wide.loc[y, c]:.4f}" for c in EMPC_CODES if c in wide]
         + [f"{wide.loc[y].sum():.4f}", f"{b.get(y, float('nan')):.4f}"]
         for y in wide.index]))

    # ---- growth lines
    A("\n## NETS vs EFSY growth, as time series\n\n")
    A("Producer 10's `Figure_nets_efsy_scatter.pdf` bins on EFSY and plots "
      "the conditional mean of NETS. That is a picture of the CORRELATION, "
      "and with a correlation near zero the conditional means necessarily "
      "collapse onto a flat line at the unconditional mean -- which is what "
      "makes the points look as though NETS barely varies. It is not evidence "
      "about dispersion. Below is each source's own marginal, by year.\n\n")
    A("`aggregate` is the midpoint growth of the national total on this "
      "support, `2 (sum E_t - sum E_t-h) / (sum E_t + sum E_t-h)`, the top row "
      "of the figure; `median` (middle row, with the interquartile range as a "
      "band) and `SD` (bottom row) are across counties. h = 1 is included "
      "alongside the "
      "benchmark's h = 2 because an annual line chart is what the question "
      "asks for; neither replaces the other.\n\n")
    e1 = gby[(gby.h == 1) & (gby.universe == "actual")].set_index("year")
    e2 = gby[(gby.h == 2) & (gby.universe == "actual")].set_index("year")
    A("**Read the last two years as EFSY's break, not as NETS's.** The median "
      f"EFSY county's growth is {e1.loc[2017, 'med_efsy']:+.2f} in 2017 at "
      f"h = 1, and {e2.loc[2017, 'med_efsy']:+.2f} in 2017 and "
      f"{e2.loc[2018, 'med_efsy']:+.2f} in 2018 at h = 2 -- an order of "
      "magnitude outside anything else in the window, and producer 10's own "
      "year cut carries the same numbers. (At h = 1 the 2018 pair straddles "
      f"the break rather than containing it, so it reads "
      f"{e1.loc[2018, 'med_efsy']:+.2f}.) NETS's median in 2017 and 2018 is "
      f"{e2.loc[2017, 'med_nets']:+.2f} and {e2.loc[2018, 'med_nets']:+.2f} "
      "at h = 2. "
      "The figure's median and SD rows are clipped so the rest of the window "
      "is legible, with every off-scale point labelled; "
      "`ws2b_growth_by_year.csv` is unclipped.\n\n")
    A("**The figure above is the whole year-by-year record for this section**, "
      "and it is deliberately not repeated as a table: three rows -- "
      "aggregate, median with the interquartile range, and the cross-county "
      "SD -- by two horizons, with both NETS universes and EFSY in every "
      "panel, is four tables of numbers whose only readable content is the "
      "shape of the lines. The digits are in `ws2b_growth_by_year.csv`, one "
      "row per year x horizon x universe, carrying the same columns plus the "
      "quartiles and the pair count.\n\n")
    A("The bottom row is the one that answers the dispersion question "
      "directly. It plots the marginal width of each source's own growth "
      "distribution, which is what a binscatter of one source on the other "
      "cannot show at all. On the headline `actual` universe the two run at "
      "comparable width; on the full panel NETS is the narrower of the two by "
      "about a quarter, which is a smoothing effect of pooling in the imputed "
      "establishments and still nothing like the collapse the binscatter "
      "appears to show. A summary over years, the median of each annual "
      "cross-sectional SD:\n\n")
    A(_md_table(
        ["universe", "h", "median SD, NETS", "median SD, EFSY", "ratio"],
        [[universe, str(h),
          f"{d['sd_nets'].median():.4f}", f"{d['sd_efsy'].median():.4f}",
          f"{d['sd_nets'].median() / d['sd_efsy'].median():.2f}"]
         for universe in UNIVERSES for h in LINE_HORIZONS
         for d in [gby[(gby.h == h) & (gby.universe == universe)]]]))

    A("\n## What this does and does not settle\n\n")
    A("It settles where the birth gap sits: in small cells, and -- by "
      "industry -- in the composition of NETS's establishment base rather "
      "than in within-sector rate disagreement, with the administrative and "
      "support-services sector the single largest term. It settles that the "
      "year-to-year swing in the NETS birth rate is carried by the imputed "
      "employment classes and has the wrong autocorrelation sign to be a "
      "business cycle. And it settles that the flat binscatter of producer 10 "
      "is a near-zero correlation rather than a NETS growth rate that barely "
      "moves: the two sources' marginal dispersions are comparable on the "
      "headline universe and NETS is about a quarter narrower on the full "
      "panel.\n\n")
    A("It settles nothing about whether the WS2 step 3 outcome swap or the "
      "step 5 entry pass should run. No criterion was fixed in advance here "
      "either, and the step-5 gate remains open.\n")
    return "".join(L)


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def compile_pdf(md_path):
    tex = RESULTS / "ws2_step2b_divergence.tex"
    subprocess.run(
        [sys.executable, str(CODE / "py" / "md2tex.py"), str(md_path),
         str(tex), "WS2 step 2b: where the NETS birth gap comes from",
         "Macro fiscal multiplier -- NETS benchmark",
         time.strftime("%B %-d, %Y")],
        check=True)
    body = tex.read_text()
    body = body.replace(
        r"\usepackage{booktabs}",
        "\\usepackage{booktabs}\n\\usepackage{graphicx}\n"
        "\\usepackage{float}\n\\usepackage{caption}")
    for anchor, fig, cap in (
            (r"\section*{Q2. Where the birth gap lives",
             "Figure_ws2b_birth_gap_bds.pdf",
             "The NETS birth rate against the BDS entry rate, all universe, "
             "under three partitions of the same county-sector-year pairs. "
             "BDS is on the x axis in every panel, so the grey 45-degree line "
             "is agreement and vertical distance is the gap. Point area is "
             "the group's share of national BDS entry."),
            (r"\section*{Q3c. Is the year-to-year swing",
             "Figure_ws2b_birth_empc.pdf",
             "The NETS aggregate birth rate decomposed by the EmpC code of "
             "the primary employment line, against the BDS entry rate on the "
             "same counties and years."),
            (r"\section*{NETS vs EFSY growth, as time series",
             "Figure_ws2b_growth_lines.pdf",
             "NETS and EFSY county employment growth, midpoint rate, as "
             "marginal time series rather than a binscatter. Rows: the "
             "aggregate on this support, the median county with its "
             "interquartile range, and the cross-county standard deviation -- "
             "the last being the series that answers the dispersion question, "
             "and the one a binscatter of one source on the other cannot "
             "show. Columns are the two horizons. This figure is the whole "
             "year-by-year record for the section; the digits are in "
             "ws2b\\_growth\\_by\\_year.csv.")):
        i = body.find(anchor)
        if i < 0:
            raise RuntimeError(f"figure anchor not found: {anchor!r}")
        block = ("\n\\begin{figure}[H]\\centering\n"
                 f"\\includegraphics[width=\\textwidth]{{{fig}}}\n"
                 f"\\caption*{{\\footnotesize {cap}}}\n"
                 "\\end{figure}\n\n")
        body = body[:i] + block + body[i:]
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
    log(f"wrote {tex.with_suffix('.pdf').name}")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    for path in (BIRTHS_EMPC, NETS_COUNTY, EFSY_COUNTY):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"missing input: {path} (producer 77 must run first)")

    con = p10.connect()
    try:
        log("support (producer 10's, unchanged)")
        p10.build_support(con)

        log("entry pairs (producer 10's build_entry, unchanged)")
        resid, supp = p10.build_entry(con)

        log("Q2: the three cuts")
        cuts = run_cuts(con)

        log("Q3c: births by EmpC class")
        empc = run_empc(con)
        bds = p10.entry_by_year(con)[["year", "er_bds"]]
        var = empc_variance(empc, bds)

        log("growth by year")
        gby = growth_by_year(con)

        log("figures")
        figure_cuts(RESULTS / "Figure_ws2b_birth_gap_bds.pdf", cuts)
        figure_empc(RESULTS / "Figure_ws2b_birth_empc.pdf", empc, bds, var)
        figure_growth_lines(RESULTS / "Figure_ws2b_growth_lines.pdf", gby)

        log("tables and report")
        (RESULTS / "Table_ws2b_birth_gap_sector.tex").write_text(
            table_sector(cuts))
        (RESULTS / "Table_ws2b_birth_gap_size.tex").write_text(
            table_size(cuts))
        (RESULTS / "Table_ws2b_birth_empc.tex").write_text(table_empc(var))
        cuts.to_csv(RESULTS / "ws2b_birth_gap_cuts.csv", index=False)
        empc.to_csv(RESULTS / "ws2b_birth_empc.csv", index=False)
        var.to_csv(RESULTS / "ws2b_birth_empc_variance.csv", index=False)
        gby.to_csv(RESULTS / "ws2b_growth_by_year.csv", index=False)

        md = RESULTS / "ws2_step2b_divergence_report.md"
        md.write_text(build_report(cuts, empc, var, gby, bds, resid, supp))
        log(f"wrote {md.name}")
        compile_pdf(md)
    finally:
        con.close()
    log("done")


if __name__ == "__main__":
    main()
