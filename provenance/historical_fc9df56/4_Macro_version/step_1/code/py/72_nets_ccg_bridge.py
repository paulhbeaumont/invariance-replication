"""WS2 step 1, first half: the NETS-to-CCG bridge and its unmatched inventory.

Maps the raw county FIPS carried by the NETS Fed2025 establishment-year panel
to `ccg_id`, harmonizing county-FIPS vintages across 1990-2023, and reports
every FIPS the bridge cannot place. It does NOT aggregate: the collapse to
county x year and county x NAICS x year is the second half of step 1 and waits
on the adjudication of the unmatched list.

Report-only pass. The plan's fail-loud rule on unmatched FIPS is armed in the
follow-up, once the Connecticut 2022 and Alaska 2019 rules are decided against
the observed counts; raising here would abort before the inventory exists.

Inputs
  <staging>/estab/estab_year_panel.parquet   765,300,955 establishment-years
  step_1/code/reference/county_ccg_crosswalk.csv  (producer 10's output)
  <macro temp>/pipeline/analysis_panel.parquet    the locked 3,058-ccg support

Outputs (all under Results/nets_benchmark/)
  ws2_step1_fips_coverage.csv    raw FIPS x year, counts, ccg_id, rule, status
  ws2_step1_unmatched_fips.csv   the unplaced FIPS alone, with year span
  ws2_step1_bridge_report.md     the memo: coverage, year profiles, decisions

Vintage handling. The crosswalk already absorbs the pre-2019 vintages as group
members, so the bridge is a plain join plus a small explicit patch table:
- Broomfield CO 2001: 08014 is a group member of the 08013 group  (crosswalk)
- Dade -> Miami-Dade:  12086 is a group member of the 12025 group  (crosswalk)
- Shannon -> Oglala:   46102 is a group member of the 46113 group  (crosswalk)
- Wade Hampton -> Kusilvak: 02158 with 02270                       (crosswalk)
- Bedford city VA 2013: 51515 with the 51019 group                 (crosswalk)
- Alaska 1990s/2000s reorganizations                               (crosswalk)
- Alaska 2019: 02063 Chugach and 02066 Copper River, the split of
  Valdez-Cordova 02261, are NOT in the crosswalk (built on a 1990-2018
  window) and are patched here to the 02261 group.
- Connecticut 2022 planning regions (09110-09190) are deliberately NOT mapped:
  they do not nest inside the eight legacy counties, so any mapping is a
  modelling choice. They are reported as their own status.
"""

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc
import nets_common as nc


def _staging_root():
    """The NETS staging root, resolved, never hard-coded.

    The step-0 gate's path correction: the canonical staging lives under the
    Dropbox project root at `_data/temp/nets_fed2025_staging`. `nets_common`'s
    own STAGING is a repo-relative mirror that only exists on the machine the
    producers ran on, so it is the fallback, not the primary.
    """
    for root in (dc.MACRO_DATA_ROOT / "temp" / "nets_fed2025_staging",
                 dc.MACRO_VERSION_TEMP_DIR / "nets_fed2025_staging",
                 nc.STAGING):
        if (root / "estab" / "estab_year_panel.parquet").exists():
            return root
    raise FileNotFoundError("NETS staging not found under any known root")


PANEL = _staging_root() / "estab" / "estab_year_panel.parquet"
ANALYSIS_PANEL = dc.MACRO_DATA_ROOT / "temp" / "pipeline" / "analysis_panel.parquet"
RESULTS = Path(__file__).resolve().parents[3] / "Results" / "nets_benchmark"
COVERAGE_OUT = RESULTS / "ws2_step1_fips_coverage.csv"
UNMATCHED_OUT = RESULTS / "ws2_step1_unmatched_fips.csv"
REPORT_OUT = RESULTS / "ws2_step1_bridge_report.md"

PANEL_ROWS = 765_300_955          # manifest 66, asserted for row conservation
CT_LEGACY = {f"090{n:02d}" for n in range(1, 16, 2)}
NONSTATE_PREFIXES = {"60", "66", "68", "69", "70", "72", "74", "78"}
# Alaska 2019: Valdez-Cordova 02261 split into Chugach and Copper River.
PATCH = {"02063": "census:2019_split_02261", "02066": "census:2019_split_02261"}
PATCH_TARGET = "02261"
# Producer 65 applies exactly these two harmonizations on the NETS side; the
# bridge must land each pair in one ccg group, in whichever direction it comes.
P65_PAIRS = [("12025", "12086"), ("46113", "46102")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def inventory():
    """One pass over the 765M-row panel: raw FIPS x year counts."""
    con = duckdb.connect()
    nc.duck_settings(con, memory_limit="16GB")
    log(f"scanning {PANEL}")
    df = con.execute(f"""
        SELECT coalesce(rtrim(fips), '') AS fips_raw,
               CAST(year AS INTEGER)     AS year,
               count(*)                  AS n_estab,
               sum(coalesce(emp_primary, 0)) AS emp
        FROM read_parquet('{PANEL}')
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()
    con.close()
    log(f"inventory: {len(df):,} fips-year rows, {df.n_estab.sum():,} estab-years")
    return df


def normalize(fips_raw):
    s = fips_raw.strip()
    if s == "" or not s.isdigit():
        return None
    if len(s) > 5:
        return None
    return s.zfill(5)


def classify(df, xwalk):
    """Attach fips5, ccg_id, the rule that placed it, and a status."""
    fips_to_ccg = dict(zip(xwalk.county_fips5, xwalk.ccg_id))
    fips_to_kind = dict(zip(xwalk.county_fips5, xwalk.kind))
    fips_to_source = dict(zip(xwalk.county_fips5, xwalk.source.fillna("")))

    rows = []
    for raw in df.fips_raw.unique():
        f5 = normalize(raw)
        if f5 is None:
            rule = ("degenerate:blank" if raw.strip() == ""
                    else "degenerate:nonnumeric")
            rows.append((raw, None, None, rule, "degenerate"))
            continue
        if f5[:2] in NONSTATE_PREFIXES or int(f5[:2]) > 56 or f5[:2] == "00":
            rows.append((raw, f5, None, "nonstate", "nonstate"))
            continue
        if f5[2:] == "000" or f5 == "99999":
            rows.append((raw, f5, None, "degenerate:state_only", "degenerate"))
            continue
        if f5 in fips_to_ccg:
            ccg = fips_to_ccg[f5]
            if ccg == "unplaceable":
                rows.append((raw, f5, None, "crosswalk:unplaceable", "unplaceable"))
                continue
            kind = fips_to_kind[f5]
            src = fips_to_source[f5]
            if kind == "identity":
                rows.append((raw, f5, ccg, "crosswalk:identity", "matched_identity"))
            else:
                rule = f"crosswalk:{kind}" + (f"/{src}" if src else "")
                rows.append((raw, f5, ccg, rule, "matched_group"))
            continue
        if f5 in PATCH:
            rows.append((raw, f5, fips_to_ccg[PATCH_TARGET],
                         f"patch:{PATCH[f5]}", "matched_patch"))
            continue
        if f5[:2] == "09" and f5 not in CT_LEGACY:
            rows.append((raw, f5, None, "unmatched:ct_2022_planning_region",
                         "unmatched_ct2022"))
            continue
        rows.append((raw, f5, None, "unmatched", "unmatched_other"))

    key = pd.DataFrame(rows, columns=["fips_raw", "fips5", "ccg_id", "rule", "status"])
    return df.merge(key, on="fips_raw", how="left")


def checks(cov, xwalk):
    """Row conservation, the producer-65 pair check, and the support check."""
    out = {}
    total = int(cov.n_estab.sum())
    out["row_conservation"] = (total, PANEL_ROWS, total == PANEL_ROWS)

    fips_to_ccg = dict(zip(xwalk.county_fips5, xwalk.ccg_id))
    out["p65_pairs"] = [
        (a, b, fips_to_ccg.get(a), fips_to_ccg.get(b),
         fips_to_ccg.get(a) is not None and fips_to_ccg.get(a) == fips_to_ccg.get(b))
        for a, b in P65_PAIRS
    ]

    locked = set(
        pd.read_parquet(ANALYSIS_PANEL, columns=["ccg"]).ccg.astype(int).unique())
    seen = {int(c) for c in cov.ccg_id.dropna().unique()}
    out["support"] = (len(locked), len(seen), sorted(seen - locked),
                      len(locked - seen))
    out["locked_missing"] = sorted(locked - seen)
    return out


def year_profile(cov, mask, label):
    sub = cov[mask]
    if sub.empty:
        return f"- {label}: never observed in the panel.\n"
    g = sub.groupby("fips5").agg(first_year=("year", "min"),
                                 last_year=("year", "max"),
                                 n_estab=("n_estab", "sum"))
    lines = [f"- {label}:"]
    for f5, r in g.sort_index().iterrows():
        lines.append(f"    - `{f5}`: {int(r.first_year)}-{int(r.last_year)}, "
                     f"{int(r.n_estab):,} establishment-years")
    return "\n".join(lines) + "\n"


def write_report(cov, unmatched, chk):
    total = cov.n_estab.sum()
    by_status = (cov.groupby("status")
                 .agg(fips=("fips_raw", "nunique"), n_estab=("n_estab", "sum"),
                      emp=("emp", "sum"))
                 .sort_values("n_estab", ascending=False))
    by_status["share_estab"] = by_status.n_estab / total

    rc_seen, rc_exp, rc_ok = chk["row_conservation"]
    n_locked, n_seen, surplus, n_missing = chk["support"]

    L = []
    L.append("# WS2 step 1, first half: NETS FIPS to CCG bridge\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
             "`step_1/code/py/72_nets_ccg_bridge.py`. Report-only pass: the "
             "fail-loud gate on unmatched FIPS is not yet armed.\n")
    L.append("\n## Coverage\n\n")
    L.append(f"Establishment-years scanned: {rc_seen:,} "
             f"(manifest 66 expects {rc_exp:,}; "
             f"{'conserved' if rc_ok else 'MISMATCH'}).\n\n")
    L.append("| status | distinct FIPS | establishment-years | share | employment |\n")
    L.append("|---|---:|---:|---:|---:|\n")
    for s, r in by_status.iterrows():
        L.append(f"| {s} | {int(r.fips):,} | {int(r.n_estab):,} | "
                 f"{r.share_estab:.4%} | {r.emp:,.0f} |\n")

    by_rule = (cov.groupby("rule")
               .agg(fips=("fips_raw", "nunique"), n_estab=("n_estab", "sum"))
               .sort_values("n_estab", ascending=False))
    L.append("\n### By rule\n\n")
    L.append("| rule | distinct FIPS | establishment-years | share |\n")
    L.append("|---|---:|---:|---:|\n")
    for r, v in by_rule.iterrows():
        L.append(f"| `{r}` | {int(v.fips):,} | {int(v.n_estab):,} | "
                 f"{v.n_estab / total:.4%} |\n")

    L.append("\n## Unmatched FIPS\n\n")
    if unmatched.empty:
        L.append("None. Every FIPS in the panel places into a ccg.\n")
    else:
        L.append("| FIPS | status | years | establishment-years | share | employment |\n")
        L.append("|---|---|---|---:|---:|---:|\n")
        for _, r in unmatched.iterrows():
            L.append(f"| `{r.fips5 or r.fips_raw}` | {r.status} | "
                     f"{int(r.first_year)}-{int(r.last_year)} | "
                     f"{int(r.n_estab):,} | {r.share_estab:.4%} | {r.emp:,.0f} |\n")

    L.append("\n## Vintage year profiles\n\n")
    L.append(year_profile(cov, cov.fips5.isin(["08014", "08013"]),
                          "Broomfield CO family (08014 created 2001)"))
    L.append(year_profile(cov, cov.fips5.isin(["02063", "02066", "02261"]),
                          "Alaska 2019 (Valdez-Cordova split)"))
    L.append(year_profile(cov, cov.fips5.isin(["12025", "12086"]),
                          "Dade / Miami-Dade"))
    L.append(year_profile(cov, cov.fips5.isin(["46113", "46102"]),
                          "Shannon / Oglala Lakota"))
    L.append(year_profile(cov, cov.fips5.isin(["02270", "02158"]),
                          "Wade Hampton / Kusilvak"))
    L.append(year_profile(cov, cov.fips5.isin(["51515", "51019"]),
                          "Bedford city VA (dissolved 2013)"))
    L.append(year_profile(cov, cov.fips5.str.startswith("09").fillna(False),
                          "Connecticut (legacy counties and planning regions)"))

    L.append("\n## Checks\n\n")
    L.append(f"- Row conservation: {'PASS' if rc_ok else 'FAIL'} "
             f"({rc_seen:,} vs {rc_exp:,}).\n")
    for a, b, ca, cb, ok in chk["p65_pairs"]:
        L.append(f"- Producer-65 pair `{a}`/`{b}`: ccg {ca}/{cb} "
                 f"{'PASS' if ok else 'FAIL'}.\n")
    L.append(f"- Support: {n_seen:,} ccgs observed in NETS against the locked "
             f"{n_locked:,}; {n_missing:,} locked ccgs carry no NETS "
             f"establishment; {len(surplus):,} ccgs outside the locked "
             f"support.\n")
    if surplus:
        L.append(f"    - outside the locked support: "
                 f"{', '.join(str(c) for c in surplus[:40])}"
                 f"{' ...' if len(surplus) > 40 else ''}\n")
    if chk["locked_missing"]:
        L.append(f"    - locked ccgs with no NETS establishment: "
                 f"{', '.join(str(c) for c in chk['locked_missing'][:40])}"
                 f"{' ...' if len(chk['locked_missing']) > 40 else ''}\n")

    L.append("\n## Open decisions\n\n")
    L.append("1. Connecticut 2022 planning regions, if present above: they do "
             "not nest in the legacy counties. Options are a town-level "
             "weighted bridge, collapsing CT to one ccg, or dropping CT for "
             "2022-23. The counts above price each.\n")
    L.append("2. Any `unmatched_other` FIPS: adjudicate individually before "
             "the fail-loud gate is armed.\n")
    L.append("2b. Blank FIPS (`degenerate:blank`): establishment-years with no "
             "county at all, spread evenly across the panel's years. They are "
             "not a vintage problem and no crosswalk rule can place them; the "
             "aggregation half must decide whether to recover any of them from "
             "the address spells (producer 65) or drop them, and the drop has "
             "to be reported as a coverage loss against EFSY, not absorbed "
             "silently.\n")
    L.append("3. The Alaska 2019 patch (`02063`, `02066` to the `02261` group) "
             "lives in this producer. If it survives adjudication it belongs "
             "in `10_build_county_universe.py` instead, so every consumer of "
             "the crosswalk inherits it.\n")
    REPORT_OUT.write_text("".join(L))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    xwalk = pd.read_csv(dc.CCG_CROSSWALK,
                        dtype={"county_fips5": str, "ccg_id": str})
    log(f"crosswalk: {len(xwalk):,} FIPS rows, "
        f"{xwalk.ccg_id.nunique():,} groups")

    cov = classify(inventory(), xwalk)
    cov = cov[["fips_raw", "fips5", "year", "n_estab", "emp", "ccg_id",
               "rule", "status"]]
    cov.to_csv(COVERAGE_OUT, index=False)
    log(f"wrote {COVERAGE_OUT}")

    total = cov.n_estab.sum()
    total_emp = cov.emp.sum()
    bad = cov[cov.status.str.startswith(("unmatched", "unplaceable"))]
    unmatched = (bad.groupby(["fips_raw", "fips5", "status", "rule"], dropna=False)
                 .agg(first_year=("year", "min"), last_year=("year", "max"),
                      n_years=("year", "nunique"), n_estab=("n_estab", "sum"),
                      emp=("emp", "sum"))
                 .reset_index()
                 .sort_values("n_estab", ascending=False))
    if not unmatched.empty:
        unmatched["share_estab"] = unmatched.n_estab / total
        unmatched["share_emp"] = unmatched.emp / total_emp
    else:
        unmatched["share_estab"] = pd.Series(dtype=float)
        unmatched["share_emp"] = pd.Series(dtype=float)
    unmatched.to_csv(UNMATCHED_OUT, index=False)
    log(f"wrote {UNMATCHED_OUT}: {len(unmatched):,} unplaced FIPS, "
        f"{int(unmatched.n_estab.sum()) if not unmatched.empty else 0:,} "
        "establishment-years")

    write_report(cov, unmatched, checks(cov, xwalk))
    log(f"wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
