#!/usr/bin/env python3
"""WS2 step 2 input: the Census BDS entry series, ingested onto the CCG spine.

WHY THIS EXISTS

WS2 step 2 benchmarks NETS establishment birth rates against a public entry
series "at the county and NAICS2 level", with a pass criterion stated before
the numbers are seen. BDS is that series. This producer only INGESTS it: it
downloads the release, parses it, puts the county geography on the same
`ccg_id` spine producer 72 built for NETS, and writes parquets. It computes no
growth rate, no rate ratio and no benchmark - those are step 2's, and they are
not made here where the pass criterion has not been written yet.

WHICH FILES, AND WHAT THE INDUSTRY AXIS ACTUALLY IS                 2026-09-01

Three files out of the 2023 release, chosen by what the benchmark needs:

    bds{r}_st_cty.csv      county x year, all sectors.       The county rung.
    bds{r}_st_cty_sec.csv  county x sector x year.           The industry rung.
    bds{r}_st.csv          state x year.                     The rollup check.

The plan says "county and NAICS2". BDS's published county-industry axis is
`sector`, the 19 NAICS SECTOR codes (11, 21, ..., 81), and three of them are
RANGES of NAICS2 - 31-33 manufacturing, 44-45 retail, 48-49 transportation.
There is no county x NAICS2 BDS table; the finer national tables (vcn3, vcn4)
carry no geography at all. So the industry benchmark runs at BDS sector, and
the NETS side has to be collapsed to the same 19 codes before anything is
compared. That is a coarsening of the plan's axis, stated here rather than
papered over, and step 2's criterion has to be written against sectors.

THE RELEASE IS PINNED AND IS NOT A DATA YEAR

Every BDS release re-estimates the WHOLE 1978-2023 series, so two releases
disagree about 1995. `dc.BDS_RELEASE` pins it; the sha256 of each downloaded
file is recorded in the report so a silent re-release cannot pass unnoticed.

SUPPRESSION IS NOT ZERO, AND IT IS NOT DROPPED

BDS writes `D` for a disclosure-suppressed cell and `N` for one that is not
available (a rate with no denominator, typically). Both are parsed to NaN, and
NEITHER is ever coerced to zero: a suppressed county-sector is a cell whose
value we do not know, and zeroing it would understate entry exactly where BDS
protects small cells - which is exactly where NETS coverage is being questioned.

RATES ARE DROPPED AT THE CCG RUNG, ON PURPOSE

`estabs_entry_rate` and the seven other published rates are ratios and do not
survive aggregation, so no CCG-rung file carries one; any rate at that rung is
step 2's to construct from the counts, under the growth-rate rule the step-0
gate fixed. The raw county-level parquets keep the published rates as BDS wrote
them, so nothing is lost. The counts themselves are split by MEASUREMENT into
those that add across geography and those that do not - see ADDITIVE below -
and a non-additive column is carried only where a CCG is a single county.

THE GEOGRAPHY IS PRODUCER 72's, IMPORTED AND NOT RESTATED

The county-FIPS vintage rules - Broomfield, Miami-Dade, Shannon/Oglala, Wade
Hampton/Kusilvak, Bedford city, the Alaska reorganizations, the 2019
Valdez-Cordova patch, and the deliberate refusal to map the 2022 Connecticut
planning regions - live in `72_nets_ccg_bridge.py` and are imported from it.
Restating them here is how the NETS panel and its own benchmark end up
disagreeing about which county Chugach is. Producer 72 resolves the NETS
staging path at import, so this producer needs that mount even though it reads
no NETS data; the failure is raised with that explanation rather than silently
falling back to a second copy of the rules.

Unlike 72, which was a report-only pass, the fail-loud rule IS armed here: an
in-window county FIPS that places nowhere aborts the run.

BDS IS LOCAL-FIRST, WHICH IS AN EXCEPTION TO THE PROJECT'S DATA RULE

The project rule is that data live only under the shared Dropbox project, with
no repo-local fallback. BDS is a deliberate exception, decided 2026-09-01: the
canonical copy - source CSVs and every parquet - is in the checkout under
`4_Macro_version/data/`, and ALL downstream analysis reads that copy. It costs
little to make an exception here (about 300 MB of public, re-downloadable CSV)
and it takes a network and a Dropbox mount out of every step-2 run.

Consumers must resolve files through `dc.bds_path(name)`, which returns the
local path and raises if it is missing. It never falls back to the mirror,
because a run that silently read the mirror would be reading whatever some
other machine last wrote - the ambiguity this arrangement exists to remove.

The Dropbox copy under `_data/temp/bds/` is a MIRROR, written after the local
files and only so a co-author without the checkout can see them. A failed
mirror is reported and does not fail the run: the real output is already on
disk. `--no-mirror` skips it. `4_Macro_version/data/**` is gitignored, so none
of this is ever committed.

Outputs (canonical, in the checkout under `4_Macro_version/data/temp/bds/`)
    bds_county_year.parquet          raw county x year, counts + published rates
    bds_county_sector_year.parquet   raw county x sector x year, same
    bds_ccg_year.parquet             CCG x year, counts, no published rates
    bds_ccg_sector_year.parquet      CCG x sector x year, same
    bds_state_year.parquet           state x year, as published (rollup check)
mirrored to the Dropbox project's `_data/temp/bds/`, and
    Results/nets_benchmark/ws2_bds_ingest_report.md
    batch_logs/macro_version/74_ingest_bds.log
Source CSVs cache in `4_Macro_version/data/source/BDS/<release>/`.
"""

import argparse
import hashlib
import importlib.util
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

CODE = Path(__file__).resolve().parent
RESULTS = Path(__file__).resolve().parents[3] / "Results" / "nets_benchmark"
REPORT_OUT = RESULTS / "ws2_bds_ingest_report.md"

# The benchmark window. BDS runs 1978-2023 and is ingested whole; 1990-2018 is
# the NETS-EFSY overlap the step-2 benchmark is defined on, and it is the span
# the fail-loud coverage rule is enforced over. A FIPS that only ever appears
# outside it (the 2022 Connecticut planning regions) is a coverage fact, not a
# failure.
WINDOW = (1990, 2018)

# WHICH COUNTS ACTUALLY ADD UP ACROSS GEOGRAPHY, ESTABLISHED BY MEASUREMENT
#
# Not assumed. Every count column was summed from BDS's own county file to its
# own state file, over the 2,253 clean state-years, before this split was
# written (2026-09-01). Six columns agree to the unit in every one of them; the
# rest do not, and the pattern is not noise:
#
#   ADDITIVE       estabs, emp, estabs_entry, estabs_exit, firmdeath_estabs,
#                  firmdeath_emp                       max abs dev 0, always
#   FIRM COUNTS    firms (up to 50,075 firms, 20%), firmdeath_firms (3%)
#                  A firm operating in two counties is one firm in each county
#                  AND one firm in the state. Summing counties over-counts it.
#                  This is a definition, not a defect, and no aggregation fixes
#                  it - the firm rung is simply not available by addition.
#   CONTINUER JOB  job_creation_continuers (2.0%), job_destruction_continuers
#   FLOWS          (2.4%), their totals, net_job_creation, and denom (0.01%)
#                  An establishment that changes county between t-1 and t is a
#                  continuer for the state and a within-county employment
#                  change for neither county in full. The birth and death
#                  components (job_creation_births, job_destruction_deaths)
#                  are nearly exact, which is what pins the mechanism to
#                  continuers rather than to rounding.
#
# ADDITIVE covers everything the WS2 step-2 entry benchmark uses. The other
# columns are still carried, but only where addition is meaningful, and their
# roll-up gap is reported rather than asserted away.
ADDITIVE = ["estabs", "emp", "estabs_entry", "estabs_exit",
            "firmdeath_estabs", "firmdeath_emp"]
NONADDITIVE = ["firms", "denom",
               "job_creation", "job_creation_births", "job_creation_continuers",
               "job_destruction", "job_destruction_deaths",
               "job_destruction_continuers", "net_job_creation",
               "firmdeath_firms"]
COUNTS = ADDITIVE + NONADDITIVE

# BDS's own "statewide, county unknown" code, one per state: establishments the
# Census cannot place in a county. It is a real BDS geography category, not a
# hole in our crosswalk - the same thing DCADS's undistributed state 98 is - so
# it gets its own status and is kept OFF the CCG spine rather than dropped
# quietly or forced into some county. Its mass (about 0.17% of in-window
# establishments) is a coverage fact the step-2 benchmark has to know, because
# every county-rung BDS total is missing exactly it.
STATEWIDE_UNKNOWN_CTY = "999"

# Ratios. Kept in the raw county parquets, never summed onto the CCG spine.
RATES = ["estabs_entry_rate", "estabs_exit_rate",
         "job_creation_rate_births", "job_creation_rate",
         "job_destruction_rate_deaths", "job_destruction_rate",
         "net_job_creation_rate", "reallocation_rate"]

# BDS's own missing-value marks. D = disclosure suppressed, N = not available.
FLAGS = {"D", "N"}

failures = []


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check(ok, num, label, detail=""):
    print(f"    [{num}] {label:<54s} {detail:<38s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append((str(num), f"{label}: {detail}"))
    return ok


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, CODE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_bridge():
    """Producer 72's vintage rules, imported rather than restated."""
    try:
        return _load("nets_ccg_bridge", "72_nets_ccg_bridge.py")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "72_nets_ccg_bridge.py owns the county-FIPS vintage rules this "
            "producer needs, and it resolves the NETS staging path at import, "
            "so the NETS mount has to be present even though no NETS data is "
            f"read here. Underlying error: {exc}") from exc


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(stem, refresh=False):
    """Download one BDS csv into _data/source/BDS/, once, and hash it.

    urllib first, curl as the fallback, exactly as
    pipeline/07_naics_vintage_concordances.py and 21_ingest_cbsa.py do, and for
    the same reason: this venv carries no certifi bundle and Python does not
    read the macOS keychain, so urllib raises SSLCertVerificationError against
    census.gov while curl succeeds. Only a certificate failure falls back; a
    404 or a timeout is a real problem and must not be retried into a transport
    that would report it differently.
    """
    dest_dir = dc.BDS_SOURCE_DIR / str(dc.BDS_RELEASE)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stem}.csv"
    url = f"{dc.BDS_BASE_URL}/{stem}.csv"
    if refresh or not dest.exists():
        log(f"downloading {url}")
        tmp = dest.with_suffix(".csv.tmp")
        try:
            with urllib.request.urlopen(url, timeout=600) as r, \
                    open(tmp, "wb") as out:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
        except urllib.error.URLError as exc:
            if not isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise
            log(f"    urllib: {exc.reason.__class__.__name__}; "
                "falling back to curl")
            subprocess.run(["curl", "-sSL", "--fail", "--max-time", "600",
                            "-o", str(tmp), url], check=True)
        tmp.replace(dest)
    else:
        log(f"already here: {dest.name}")
    digest = sha256_of(dest)
    log(f"    {dest.name}  {dest.stat().st_size:,} bytes  sha256 {digest[:16]}...")
    return dest, digest


def read_bds(path):
    """Parse one BDS csv: flags to NaN, values to float, keys to strings.

    The flag columns are counted BEFORE the coercion, so the report can say how
    much of each column BDS withheld rather than reporting a NaN count that
    conflates suppression with a column that simply is not in this file.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    df["year"] = df["year"].astype(int)
    for key in ("st", "cty", "sector"):
        if key in df.columns:
            df[key] = df[key].str.strip()
    if "st" in df.columns:
        df["st"] = df["st"].str.zfill(2)
    if "cty" in df.columns:
        df["cty"] = df["cty"].str.zfill(3)

    marks = {}
    for col in COUNTS + RATES:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip()
        flagged = s.isin(FLAGS)
        marks[col] = {f: int((s == f).sum()) for f in sorted(FLAGS)}
        df[col] = pd.to_numeric(s.where(~flagged), errors="coerce")
    return df, marks


def to_ccg(df, keys, bridge, xwalk, label):
    """County rows onto the CCG spine, through producer 72's classifier.

    Suppression PROPAGATES: a CCG-year whose group contains a suppressed member
    is NaN for that column, not a partial sum. A partial sum would look like a
    smaller county and would bias the very entry comparison this file feeds.
    Most CCGs are singleton identity groups, where this is a passthrough.
    """
    df = df.copy()
    df["fips_raw"] = df["st"] + df["cty"]
    keyed = bridge.classify(df, xwalk)

    # BDS's statewide-unknown rows are claimed before the fail-loud rule runs.
    # They are a BDS category, not an unplaced county, and they carry no county
    # to place; producer 72 has never seen one because NETS has street
    # addresses. Marked, kept out of the spine, and counted in the report.
    unk = keyed["cty"] == STATEWIDE_UNKNOWN_CTY
    keyed.loc[unk, ["ccg_id", "rule", "status"]] = [
        None, "bds:statewide_unknown", "bds_statewide_unknown"]

    live = keyed[keyed["year"].between(*WINDOW)]
    stray = sorted(set(live.loc[live["status"] == "unmatched_other", "fips5"]))
    check(not stray, f"2{label}", f"[{label}] every in-window FIPS places",
          "none unmatched" if not stray else f"{len(stray)}: {stray[:6]}")

    placed = keyed[keyed["ccg_id"].notna()].copy()
    placed["ccg"] = placed["ccg_id"].astype(int)

    cols = [c for c in COUNTS if c in placed.columns]
    group = ["ccg"] + keys
    agg = placed.groupby(group, as_index=False).agg(
        **{c: (c, "sum") for c in cols},
        **{f"_n_{c}": (c, "count") for c in cols},
        n_members=("fips5", "nunique"))
    for c in cols:
        # count() ignores NaN, so a short count is a suppressed member.
        agg.loc[agg[f"_n_{c}"] < agg["n_members"], c] = np.nan
    # A non-additive column survives only where the CCG is a single county, so
    # that no sum is ever taken over a definition that does not support one. A
    # multi-county group gets NaN, not a firm count that double-counts the
    # firms operating in two of its members.
    for c in cols:
        if c in NONADDITIVE:
            agg.loc[agg["n_members"] > 1, c] = np.nan
    agg["n_suppressed_cells"] = sum(
        (agg["n_members"] - agg[f"_n_{c}"]) for c in cols)
    agg = agg.drop(columns=[f"_n_{c}" for c in cols])
    return keyed, agg.sort_values(group).reset_index(drop=True)


def state_rollup(county, state, keys_extra=()):
    """Check: county sums equal BDS's own state rows where nothing is withheld.

    The load-bearing check, and the same one the BEA panels use. BDS's counties
    partition each state, so a dropped county, a doubled group member or a
    mis-parsed flag shows up here at once and cannot show up anywhere else - a
    county quietly carrying another county's entrants leaves every county total
    plausible and the state total wrong. State-years with any suppressed member
    are EXCLUDED from the comparison and counted separately, because there the
    two sides are not summing the same cells by construction.
    """
    per_col, n_cmp, n_skip = {}, 0, 0
    for col in COUNTS:
        if col not in county.columns or col not in state.columns:
            continue
        cnt = county.groupby(["st", "year"])[col]
        got = cnt.sum(min_count=1)
        n_ok = cnt.count()
        n_all = cnt.size()
        want = state.groupby(["st", "year"])[col].sum(min_count=1)
        j = pd.concat([want.rename("bds"), got.rename("ours"),
                       (n_ok == n_all).rename("clean")], axis=1).dropna()
        n_skip += int((~j["clean"]).sum())
        j = j[j["clean"]]
        if j.empty:
            per_col[col] = (0, 0, 0.0, "")
            continue
        ad = (j["ours"] - j["bds"]).abs()
        i = ad.idxmax()
        per_col[col] = (len(j), int((ad > 1e-9).sum()), float(ad.max()),
                        f"st {i[0]} in {i[1]}")
        n_cmp += len(j)

    # Asserted on ADDITIVE only, to the unit, because these are integer counts
    # and a rounding tolerance would hide exactly the merge defect this check
    # exists to catch. The rest are measured and reported, never asserted: the
    # firm and continuer columns do not add across geography by construction,
    # and pretending otherwise would either fail the run forever or force a
    # tolerance loose enough to pass a real error too.
    bad = {c: v for c, v in per_col.items() if c in ADDITIVE and v[1]}
    n_add = sum(per_col[c][0] for c in ADDITIVE if c in per_col)
    check(not bad and n_add > 0, 3,
          "county sums equal BDS's state rows, additive columns",
          f"{n_add:,} comparisons, {n_skip:,} skipped, exact"
          if not bad else f"{len(bad)} columns disagree: {sorted(bad)[:4]}")
    return n_cmp, n_skip, per_col


def sector_rollup(county, sector):
    """Check: the sector file sums to the county file, where nothing is withheld.

    BDS's sectors partition the covered universe within a county-year, so this
    is the industry rung's version of the state check. Again only county-years
    with a complete set of unsuppressed sector cells enter the comparison, and
    the number skipped is itself a reported coverage fact: it is the share of
    the industry rung the benchmark simply cannot see.
    """
    per_col, n_cmp, n_skip = {}, 0, 0
    for col in ADDITIVE:
        g = sector.groupby(["st", "cty", "year"])[col]
        got = g.sum(min_count=1)
        clean = (g.count() == g.size())
        want = county.set_index(["st", "cty", "year"])[col]
        j = pd.concat([want.rename("cty"), got.rename("sec"),
                       clean.rename("clean")], axis=1).dropna()
        n_skip += int((~j["clean"]).sum())
        j = j[j["clean"]]
        if j.empty:
            per_col[col] = (0, 0, 0.0, "")
            continue
        ad = (j["sec"] - j["cty"]).abs()
        i = ad.idxmax()
        per_col[col] = (len(j), int((ad > 1e-9).sum()), float(ad.max()),
                        f"{i[0]}{i[1]} in {i[2]}")
        n_cmp += len(j)
    bad = {c: v for c, v in per_col.items() if v[1]}
    check(not bad and n_cmp > 0, 4,
          "sector rows roll up to the county rows, additive columns",
          f"{n_cmp:,} comparisons, {n_skip:,} withheld, exact"
          if not bad else f"{len(bad)} columns disagree: {sorted(bad)[:4]}")
    return n_cmp, n_skip, per_col


def mirror_to_dropbox(tables, skip):
    """Copy the written parquets to the Dropbox mirror, best effort.

    The local copy is canonical and is already on disk before this runs. The
    mirror exists only so a co-author with the Dropbox project and no checkout
    can see the files, so a mirror that fails - Dropbox not mounted, no space,
    a read-only share - must NOT fail a run whose real output already exists.
    It is reported instead, and the report says plainly whether it happened.
    """
    if skip:
        log("    mirror skipped (--no-mirror)")
        return "skipped by --no-mirror"
    try:
        dc.BDS_MIRROR_DIR.mkdir(parents=True, exist_ok=True)
        for path, _, _ in tables:
            shutil.copy2(path, dc.BDS_MIRROR_DIR / path.name)
    except OSError as exc:
        log(f"    MIRROR FAILED, local copy is unaffected: {exc}")
        return f"failed: {exc}"
    log(f"    mirrored {len(tables)} files -> {dc.BDS_MIRROR_DIR}")
    return "ok"


def ccg_support(ccg_c):
    """Which of the locked ccgs BDS can actually cover, and which it cannot.

    Report-only, and deliberately not a failure. The gap is Connecticut, and it
    is a consequence of a decision producer 72 already made rather than
    anything this file could fix: the 2023 BDS release RETRO-CODES Connecticut
    onto the nine 2022 planning regions for every year back to 1978, so the
    eight legacy CT counties have no BDS row in any year, and 72 refuses to map
    planning regions into counties because they do not nest. Step 2's benchmark
    therefore has no Connecticut, whatever it decides; that is a fact it has to
    state, not a check this producer should abort on.
    """
    locked = set(pd.read_parquet(
        dc.MACRO_DATA_ROOT / "temp" / "pipeline" / "analysis_panel.parquet",
        columns=["ccg"]).ccg.astype(int).unique())
    win = ccg_c[ccg_c["year"].between(*WINDOW)]
    seen = set(win["ccg"].astype(int))
    missing = sorted(locked - seen)
    ct = [c for c in missing if c // 1000 == 9]
    print(f"    locked {len(locked):,}   BDS in-window {len(seen):,}   "
          f"intersection {len(locked & seen):,}")
    print(f"    locked ccgs with no BDS row: {len(missing)} "
          f"({len(ct)} in Connecticut) {missing[:8]}")
    return locked, seen, missing, ct


def write_report(meta, cov_county, cov_sector, marks, tables, rollups,
                 support, mirrored):
    RESULTS.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("# WS2 step 2 input: the BDS ingest\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
             "`step_1/code/py/74_ingest_bds.py`. Release "
             f"**{dc.BDS_RELEASE}**, pinned. This producer ingests only; the "
             "benchmark itself, and its pass criterion, are WS2 step 2's and "
             "are not written here.\n")

    tables_rows = {path.name: n for path, _, n in tables}
    L.append("## The downloaded files\n")
    L.append("| file | bytes | sha256 | rows | year span |")
    L.append("|---|---:|---|---:|---|")
    for stem, (digest, nbytes, nrows, span) in meta.items():
        L.append(f"| `{stem}.csv` | {nbytes:,} | `{digest[:32]}` | {nrows:,} | "
                 f"{span[0]}-{span[1]} |")

    L.append("\n## The industry axis is BDS `sector`, not NAICS2\n")
    L.append("The published county-industry table is county x **sector**: the "
             "19 NAICS sector codes, three of which are ranges of NAICS2 "
             "(`31-33`, `44-45`, `48-49`). BDS publishes no county x NAICS2 "
             "table, and its finer industry tables (`vcn3`, `vcn4`) carry no "
             "geography. The step-2 benchmark therefore runs at sector, with "
             "the NETS side collapsed to the same 19 codes, and its criterion "
             "has to be written against sectors rather than NAICS2.\n")

    L.append("## What BDS withholds\n")
    L.append("`D` is disclosure suppression and `N` is not-available. Both "
             "parse to NaN and neither is ever coerced to zero.\n")
    sec = marks.get(f"bds{dc.BDS_RELEASE}_st_cty_sec", {})
    cty = marks.get(f"bds{dc.BDS_RELEASE}_st_cty", {})
    n_sec = tables_rows.get("bds_county_sector_year.parquet", 0)
    n_cty = tables_rows.get("bds_county_year.parquet", 0)
    if sec and n_sec:
        d_sec = sec["estabs_entry"]["D"]
        d_cty = cty["estabs_entry"]["D"]
        L.append(f"\n**The industry rung is a third invisible.** BDS suppresses "
                 f"`estabs_entry` in {d_sec:,} of the {n_sec:,} "
                 f"county-sector-years, **{100 * d_sec / n_sec:.1f}%**, against "
                 f"{d_cty:,} of {n_cty:,} ({100 * d_cty / n_cty:.1f}%) at the "
                 "county rung. Suppression is concentrated in exactly the small "
                 "cells the NETS comparison is about, so the county-sector "
                 "benchmark is not a benchmark on the full universe and its "
                 "pass criterion cannot be written as though it were. This is "
                 "the single most important number in this report for step 2.\n")
    for name, m in marks.items():
        rows = [(c, f["D"], f["N"]) for c, f in m.items() if f["D"] or f["N"]]
        L.append(f"\n**`{name}`**\n")
        if not rows:
            L.append("Nothing withheld in any column.\n")
            continue
        L.append("| column | D | N |")
        L.append("|---|---:|---:|")
        for c, d, n in sorted(rows, key=lambda r: -r[1]):
            L.append(f"| `{c}` | {d:,} | {n:,} |")

    L.append("\n## The CCG bridge\n")
    L.append("Producer 72's classifier and vintage rules, imported. The "
             f"fail-loud rule is armed over {WINDOW[0]}-{WINDOW[1]}.\n")
    for label, cov in (("county x year", cov_county),
                       ("county x sector x year", cov_sector)):
        g = (cov.groupby("status")["fips5"].nunique().sort_values(ascending=False))
        L.append(f"\n**{label}** — FIPS by status\n")
        L.append("| status | distinct FIPS |")
        L.append("|---|---:|")
        for status, n in g.items():
            L.append(f"| `{status}` | {n:,} |")

    win = cov_county[cov_county["year"].between(*WINDOW)]
    unk = win[win["status"] == "bds_statewide_unknown"]
    L.append("\n### BDS's statewide-unknown rows\n")
    L.append("`XX999` is BDS's own \"statewide, county unknown\" code - "
             "establishments Census cannot place in a county - and there is one "
             "per state. It is a BDS geography category, the same thing DCADS's "
             "undistributed state 98 is, not a hole in the crosswalk, so it is "
             "given its own status and kept OFF the CCG spine rather than "
             "dropped quietly or forced into a county. **Every county-rung BDS "
             "total is missing exactly this mass**, which is why it is sized "
             "here and not left implicit:\n")
    for col in ("estabs", "emp", "estabs_entry"):
        tot = float(win[col].sum())
        part = float(unk[col].sum())
        L.append(f"- `{col}`: {part:,.0f} of {tot:,.0f} in-window, "
                 f"{100 * part / tot:.3f}%")
    L.append(f"\n{unk['fips5'].nunique()} such codes appear in "
             f"{WINDOW[0]}-{WINDOW[1]}, over {len(unk):,} rows.\n")

    locked, seen, missing, ct = support
    L.append("\n### The support against the locked panel: BDS has no Connecticut\n")
    L.append(f"Locked ccgs **{len(locked):,}**, BDS in-window "
             f"**{len(seen):,}**, intersection **{len(locked & seen):,}**. "
             f"{len(missing)} locked ccgs have no BDS row in any in-window "
             f"year, and {len(ct)} of them are the eight legacy Connecticut "
             "counties.\n")
    L.append("The reason is not suppression. The 2023 BDS release **retro-codes "
             "Connecticut onto the nine 2022 planning regions for every year "
             "back to 1978**, so `09001`-`09015` appear nowhere in the file, and "
             "producer 72 refuses to map planning regions into counties because "
             "they do not nest. Step 2's NETS-vs-BDS benchmark therefore has no "
             "Connecticut available to it under any option this ingest could "
             "have chosen, and its exhibit note has to say so. Nothing here "
             "re-opens 72's decision.\n")

    L.append("\n## What does and does not add up across geography\n")
    L.append("Summing BDS's own county file to its own state file, over the "
             "state-years where nothing is withheld. Six columns agree to the "
             "unit in every comparison; the rest do not, and are carried on the "
             "CCG spine only where the group is a single county.\n")
    L.append("| column | additive? | comparisons | disagreeing | max abs dev |")
    L.append("|---|---|---:|---:|---:|")
    for col in COUNTS:
        n, nbad, mx, _ = rollups["state"][2].get(col, (0, 0, 0.0, ""))
        tag = "**yes**" if col in ADDITIVE else "no"
        L.append(f"| `{col}` | {tag} | {n:,} | {nbad:,} | {mx:,.0f} |")
    L.append("\nThe two non-additive families are definitional, not defects. A "
             "**firm** operating in two counties is one firm in each county and "
             "one firm in the state, so county firm counts cannot be summed "
             "(`firms` is off by up to 50,075 firms, 20%, in a state-year). "
             "**Continuer job flows** miss the establishment that changes "
             "county between t-1 and t, which is a continuer for the state and "
             "for neither county; the birth and death components are nearly "
             "exact, which is what pins the gap to continuers rather than to "
             "rounding. Every column the step-2 entry benchmark uses - "
             "`estabs`, `emp`, `estabs_entry`, `estabs_exit` - is in the exact "
             "set.\n")

    L.append("## The checks\n")
    L.append("| check | result |")
    L.append("|---|---|")
    n_cmp, n_skip, per = rollups["state"]
    n_add = sum(per[c][0] for c in ADDITIVE if c in per)
    L.append(f"| county sums vs BDS state rows, additive columns | {n_add:,} "
             f"clean comparisons, {n_skip:,} skipped for suppression, exact to "
             "the unit |")
    n_cmp, n_skip, per = rollups["sector"]
    L.append(f"| sector rows vs county rows, additive columns | {n_cmp:,} clean "
             f"comparisons, {n_skip:,} withheld, exact to the unit |")

    L.append("\n## What was written, and where analysis must read it\n")
    L.append("**BDS is local-first, an exception to the project's "
             "Dropbox-only data rule, decided 2026-09-01.** The canonical copy "
             f"is in the checkout at `{dc.BDS_OUT_DIR}`, and every downstream "
             "consumer must resolve it through `dc.bds_path(name)`, which "
             "returns that path and raises if the file is missing. It never "
             "falls back to the mirror: a run that silently read the mirror "
             "would be reading whatever another machine last wrote.\n")
    L.append(f"The Dropbox copy at `{dc.BDS_MIRROR_DIR}` is a **mirror** only, "
             "written after the local files so a co-author without the "
             f"checkout can see them. Mirror status this run: **{mirrored}**. "
             "A failed mirror does not fail the run, and does not affect the "
             "canonical copy. Source CSVs cache locally under "
             f"`{dc.BDS_SOURCE_DIR}`.\n")
    L.append("| file | key | rows |")
    L.append("|---|---|---:|")
    for path, key, n in tables:
        L.append(f"| `{path.name}` | {key} | {n:,} |")
    L.append("\nCCG-rung files carry the counts only. The eight published "
             "rates are ratios, do not survive aggregation, and are kept in the "
             "raw county files alone; any rate at the CCG rung is step 2's to "
             "build from the counts. The non-additive columns are present but "
             "populated only where the CCG is a single county; a multi-county "
             "group carries NaN for them rather than a sum its definition does "
             "not support.\n")
    L.append("Suppression propagates on aggregation: a multi-county CCG whose "
             "group contains a withheld member is NaN for that column, never a "
             "partial sum. `n_members` and `n_suppressed_cells` record it.\n")

    REPORT_OUT.write_text("\n".join(L) + "\n")
    log(f"wrote {REPORT_OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true",
                    help="re-download even if the csvs are already here")
    ap.add_argument("--no-mirror", action="store_true",
                    help="write the local copy only, skip the Dropbox mirror")
    args = ap.parse_args()

    tee = None
    try:
        dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
        sym = _load("first_stage_table", "19_first_stage_table.py")
        tee = sym.Tee(dc.LOG_DIR / "74_ingest_bds.log")

        r = dc.BDS_RELEASE
        print("=" * 78)
        print(f"Census BDS release {r}, onto the CCG spine")
        print("=" * 78)

        print("\n[0] the downloads")
        stems = {"county": f"bds{r}_st_cty",
                 "sector": f"bds{r}_st_cty_sec",
                 "state": f"bds{r}_st"}
        paths, meta, frames, marks = {}, {}, {}, {}
        for name, stem in stems.items():
            path, digest = fetch(stem, args.refresh)
            paths[name] = path
            df, m = read_bds(path)
            frames[name], marks[stem] = df, m
            meta[stem] = (digest, path.stat().st_size, len(df),
                          (int(df.year.min()), int(df.year.max())))

        print("\n[1] the release is the pinned one, and the span covers the window")
        span_ok = all(m[3][0] <= WINDOW[0] and m[3][1] >= WINDOW[1]
                      for m in meta.values())
        check(span_ok, 1, "every file spans the benchmark window",
              f"{WINDOW[0]}-{WINDOW[1]} inside "
              f"{meta[stems['county']][3][0]}-{meta[stems['county']][3][1]}")

        print("\n[2] the CCG bridge")
        bridge = load_bridge()
        xwalk = pd.read_csv(dc.CCG_CROSSWALK,
                            dtype={"county_fips5": str, "ccg_id": str})
        log(f"crosswalk: {len(xwalk):,} FIPS rows, "
            f"{xwalk.ccg_id.nunique():,} groups")
        cov_c, ccg_c = to_ccg(frames["county"], ["year"], bridge, xwalk, "a")
        cov_s, ccg_s = to_ccg(frames["sector"], ["sector", "year"], bridge,
                              xwalk, "b")
        log(f"county rung: {len(ccg_c):,} ccg-years, "
            f"{ccg_c.ccg.nunique():,} ccgs")
        log(f"sector rung: {len(ccg_s):,} ccg-sector-years, "
            f"{ccg_s.ccg.nunique():,} ccgs")

        print("\n[3] the support, against the locked panel")
        support = ccg_support(ccg_c)

        print("\n[4] the rollups")
        rollups = {
            "state": state_rollup(frames["county"], frames["state"]),
            "sector": sector_rollup(frames["county"], frames["sector"]),
        }

        print("\n[5] entry is present and non-negative in the window")
        win = frames["county"][frames["county"].year.between(*WINDOW)]
        e = win["estabs_entry"]
        bad = int((e < 0).sum())
        check(bad == 0 and e.notna().any(), 5,
              "in-window establishment entry is non-negative",
              f"{e.notna().sum():,} observed, {e.isna().sum():,} withheld, "
              f"{bad} negative")

        if failures:
            print(f"\n{len(failures)} CHECK(S) FAILED, nothing written")
            for num, msg in failures:
                print(f"    [{num}] {msg}")
            return 1

        print("\n[6] writing, local first")
        dc.BDS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        tables = []
        for fname, frame, key in (
                ("bds_county_year.parquet", frames["county"], "st, cty, year"),
                ("bds_county_sector_year.parquet", frames["sector"],
                 "st, cty, sector, year"),
                ("bds_state_year.parquet", frames["state"], "st, year"),
                ("bds_ccg_year.parquet", ccg_c, "ccg, year"),
                ("bds_ccg_sector_year.parquet", ccg_s, "ccg, sector, year")):
            out = dc.BDS_OUT_DIR / fname
            frame.to_parquet(out, index=False)
            tables.append((out, key, len(frame)))
            log(f"    {len(frame):,} rows -> {out}")

        mirrored = mirror_to_dropbox(tables, args.no_mirror)
        write_report(meta, cov_c, cov_s, marks, tables, rollups, support,
                     mirrored)
        print("\nall checks passed")
        return 0
    finally:
        if tee is not None:
            tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
