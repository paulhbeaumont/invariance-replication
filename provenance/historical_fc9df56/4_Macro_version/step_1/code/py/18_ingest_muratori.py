#!/usr/bin/env python3
"""The post-FY2006 contract block, and the audit that decides whether it is usable.

DCADS/MPCF ends at FY2006. That is the archival series' hard boundary, not a
choice, so every year after it has to come from somewhere else. This script reads
the county defense procurement panel Umberto Muratori shared on 2026-08-10 (with
Pedro Juarros and Daniel Valderrama), puts FY2007 onward on our CCG spine, and
then spends most of its length trying to establish that the result is the same
object as the DCADS years rather than assuming it.

WHAT IS TAKEN, AND WHAT IS NOT                                          [2], [3]

Only `year >= dc.SEAM_YEAR` is written. Two reasons, and the second is the one
that matters:

  - 1975-2006 is already built from DCADS and verified to the dollar against
    Pallante's published state-year series by 09_pallante_test.py. There is no
    reason to replace a verified series with an unverified one.
  - His FY1989-2000 block is WRONG. Its national totals run 3.0x to 3.5x ours
    ($321-401bn a year against our $106-124bn); ours matches the historical
    record for DoD prime contract awards and his does not. FY1988 is 1.84x. The
    break sits where DCADS switches its dollar unit, but a 3x gap is not a unit
    slip. Writing only the post-seam block means the bad years are absent from
    the parquet by construction, so nothing downstream needs a guard against
    them - a guard in this script could not observe a downstream consumer anyway.

THE THREE GEOGRAPHY DROPS, all reported and none silent                     [2]

  SS999          260 rows, $41,106,000 - state-undistributed, the same rows our
                 own geo_class filter excludes.
  2-char codes    34 rows, $1,807,260,620 - bare state codes, FY1966-83. Real
                 money, and much larger than the SS999 block.
  47098            1 row, $12,000, FY1980 - not a valid FIPS. Declared here by
                 name because our `unplaceable` set contains only 36173, so an
                 undeclared code would abort the run rather than be dropped.

The gate on the first two is MATERIALITY, not equality. An exact assert on
$41,106,000 would fail the moment the coauthors re-share with the extra flag
columns this script's own audit asks them for, which is an event we are trying
to cause. The file's identity is pinned by sha256 instead.

n_actions IS NULL, NEVER ZERO                                               [2]

His file carries three columns - year, county code, dollars - and no action
count. Zero would be summed as a count; null cannot be.

Output:
    dc.PANEL_OUT_DIR/contracts_muratori_ccg_year.parquet   post-seam only
    dc.LOG_DIR/muratori_ingest_report.csv                  per-year, incl. the
                                                           nominal totals 13's
                                                           check 2 reads back
    dc.LOG_DIR/muratori_audit_lambda.csv                   the per-county ratio
"""

import argparse
import collections
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

LAST_YEAR = 2018           # EFSY employment ends here, so the panel does
OVERLAP = (2001, 2006)     # where both constructions exist
BAD_BLOCK = (1988, 2000)   # his, not ours; excluded by not being written

EXPECT_COLUMNS = ("year_calendar", "country_code", "dollarsobligated")
DECLARED_BAD_FIPS = {"47098": "not a valid Tennessee FIPS; 1 row, FY1980, $12,000"}
DROP_MATERIALITY = 0.001   # dropped dollars as a share of national, per the docstring

# Contractor-HQ counties, for the place-of-performance tilt test in [5]. If his
# file redistributed foreign or subcontracted work by recipient location rather
# than place of performance, these are where it would land.
HQ_COUNTIES = {51013: "Arlington VA", 51059: "Fairfax VA", 11001: "DC",
               6037: "Los Angeles CA", 29189: "St Louis County MO",
               29510: "St Louis city MO"}

# The ours-vs-theirs overlap band, used by check 4b only. The Pallante gate
# calibrates its own band from the data; see pallante_gate.
PALLANTE_BAND = (0.93, 1.19)


def check(failures, ok, num, label, detail=""):
    """One numbered PASS/FAIL line, in 13_build_county_panel.py's format."""
    print(f"    [{num}] {label:<52s} {detail:<34s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append((str(num), f"{label} {detail}"))
    return ok


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_raw():
    """His file, with country_code kept as a STRING.

    A numeric read turns 01001 into 1001 and silently destroys every county in
    states 01-09, which is a fifth of the panel and would not raise anything.
    """
    if not dc.MURATORI_SOURCE.exists():
        raise FileNotFoundError(
            f"Dropbox-only Muratori source missing: {dc.MURATORI_SOURCE}"
        )
    df = pd.read_stata(dc.MURATORI_SOURCE)
    missing = [c for c in EXPECT_COLUMNS if c not in df.columns]
    if missing:
        raise AssertionError(f"expected columns absent: {missing}; got {list(df.columns)}")
    df["fips"] = df["country_code"].astype(str)
    df["year"] = df["year_calendar"].astype(int)
    df["dollars"] = df["dollarsobligated"].astype(float)
    return df[["fips", "year", "dollars"]]


def our_county_year():
    """Our verified DCADS collapse, CCG x year, nominal."""
    path = dc.PANEL_OUT_DIR / "contracts_ccg_year.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run 13_build_county_panel.py first")
    return pd.read_parquet(path, columns=["ccg", "year", "dollars_nominal"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--last-year", type=int, default=LAST_YEAR)
    ap.add_argument("--skip-audit", action="store_true",
                    help="ingest only; the audit is the point, so this is for debugging")
    args = ap.parse_args()
    failures = []

    print(f"\nMuratori county procurement -> CCG x year, "
          f"FY{dc.SEAM_YEAR}-{args.last_year} written, audit over "
          f"FY{OVERLAP[0]}-{OVERLAP[1]}")
    print(f"    source {dc.MURATORI_SOURCE}")

    # --- [1] the file itself ----------------------------------------------------------
    print("\n[1] the file")
    raw = load_raw()
    digest = sha256(dc.MURATORI_SOURCE)
    size = dc.MURATORI_SOURCE.stat().st_size
    print(f"    {len(raw):,} rows, FY{raw.year.min()}-{raw.year.max()}, "
          f"{raw.fips.nunique():,} distinct codes")
    print(f"    sha256 {digest}")
    print(f"    {size:,} bytes, received 2026-08-10")
    check(failures, not raw.duplicated(["year", "fips"]).any(), 1,
          "unique on (year, county code)", f"{len(raw):,} rows")

    # --- [2] geography ----------------------------------------------------------------
    print("\n[2] geography")
    national = float(raw.dollars.abs().sum())
    short = raw[raw.fips.str.len() != 5]
    undist = raw[(raw.fips.str.len() == 5) & (raw.fips.str[2:] == "999")]
    print(f"    2-char state codes  {len(short):>4,} rows  ${short.dollars.sum():>18,.0f}")
    print(f"    SS999 undistributed {len(undist):>4,} rows  ${undist.dollars.sum():>18,.0f}")
    dropped = float(short.dollars.abs().sum() + undist.dollars.abs().sum())
    check(failures, dropped / national < DROP_MATERIALITY, 2,
          "non-county geography is immaterial",
          f"{100 * dropped / national:.3f}% of ${national / 1e9:,.0f}bn")

    keep = raw[(raw.fips.str.len() == 5) & (raw.fips.str[2:] != "999")].copy()
    fips_to_ccg, unplaceable, universe = dc.load_ccg_map()
    keep["ccg"] = keep.fips.map(fips_to_ccg)
    bad = keep[keep.ccg.isna()]
    undeclared = sorted(set(bad.fips) - set(DECLARED_BAD_FIPS) - set(unplaceable))
    for code in sorted(set(bad.fips) & set(DECLARED_BAD_FIPS)):
        rows = bad[bad.fips == code]
        print(f"    declared drop {code}  {len(rows)} row(s)  "
              f"${rows.dollars.sum():,.0f}  - {DECLARED_BAD_FIPS[code]}")
    check(failures, not undeclared, 3,
          "every county code maps, or is declared",
          f"{len(undeclared)} undeclared: {undeclared[:5]}")
    if undeclared:
        print("        an unmapped code may be a territory, a foreign place, or a "
              "vintage we do not carry;\n        declare it in DECLARED_BAD_FIPS "
              "with its year, rows and dollars before rerunning.")

    keep = keep.dropna(subset=["ccg"]).copy()
    keep["ccg"] = keep.ccg.astype(np.int64)
    check(failures, bool(set(keep.ccg) <= set(universe.tolist())), 4,
          "mapped CCGs sit inside the 3,101 universe", f"{keep.ccg.nunique():,} seen")

    # --- [3] collapse and deflate ------------------------------------------------------
    print("\n[3] collapse to CCG x year")
    win = keep[keep.year.between(dc.SEAM_YEAR, args.last_year)]
    panel = win.groupby(["ccg", "year"], as_index=False)["dollars"].sum()
    panel = panel.rename(columns={"dollars": "dollars_nominal"})
    panel["dollars_real"] = panel.dollars_nominal * panel.year.map(dc.deflate)
    panel["n_actions"] = pd.array([pd.NA] * len(panel), dtype="Int64")
    neg = panel.dollars_nominal < 0
    print(f"    {len(panel):,} CCG-years, {panel.ccg.nunique():,} CCGs, "
          f"FY{panel.year.min()}-{panel.year.max()}")
    print(f"    {int(neg.sum()):,} negative CCG-years, "
          f"${panel.loc[neg, 'dollars_real'].sum() / 1e9:,.2f}bn - deobligations, kept")
    print("    n_actions is null: his file carries no action count, and a zero would "
          "be summed as one")

    # --- [4] the audit -----------------------------------------------------------------
    if not args.skip_audit:
        audit(failures, keep, args.last_year)

    # --- [5] write ---------------------------------------------------------------------
    print("\n[5] writing")
    dc.PANEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = dc.PANEL_OUT_DIR / "contracts_muratori_ccg_year.parquet"

    rep = panel.groupby("year").agg(
        n_ccg=("ccg", "nunique"),
        dollars_nominal=("dollars_nominal", "sum"),
        dollars_real_bn=("dollars_real", lambda s: s.sum() / 1e9),
        n_negative=("dollars_nominal", lambda s: int((s < 0).sum()))).reset_index()
    rep.to_csv(dc.LOG_DIR / "muratori_ingest_report.csv", index=False)

    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed - nothing written", file=sys.stderr)
        for name, why in failures:
            print(f"  {name}: {why}", file=sys.stderr)
        return 1

    panel.to_parquet(out, index=False)
    print(f"    {len(panel):,} rows -> {out}")
    print(f"    per-year report -> {dc.LOG_DIR / 'muratori_ingest_report.csv'}")
    print("\nall checks passed")
    return 0


# ---------------------------------------------------------------------------
# The audit. Everything below decides whether the block written above is the same
# object as the DCADS years, and none of it writes to the panel.
# ---------------------------------------------------------------------------

def audit(failures, keep, last_year):
    print("\n[4] the audit")
    ours = our_county_year()
    theirs = keep.groupby(["ccg", "year"], as_index=False)["dollars"].sum()

    # 4a. National totals, ours against theirs, every overlapping year. This is the
    #     finding that rules out 1989-2000, and it is recomputed rather than trusted.
    o = ours.groupby("year")["dollars_nominal"].sum()
    t = theirs.groupby("year")["dollars"].sum()
    both = sorted(set(o.index) & set(t.index))
    print("\n    national nominal, $bn         ours     theirs    theirs/ours")
    for y in both:
        flag = "  <- excluded" if BAD_BLOCK[0] <= y <= BAD_BLOCK[1] else ""
        print(f"      FY{y}                    {o[y]/1e9:>9,.1f}  {t[y]/1e9:>9,.1f}"
              f"       {t[y]/o[y]:>6.3f}{flag}")
    ratios = {y: t[y] / o[y] for y in both}
    bad = [y for y in both if BAD_BLOCK[0] <= y <= BAD_BLOCK[1]]
    check(failures, all(ratios[y] > 1.5 for y in bad), "4a",
          "the excluded block is still the anomaly it was",
          f"FY{bad[0]}-{bad[-1]} ratio {min(ratios[y] for y in bad):.2f}"
          f"-{max(ratios[y] for y in bad):.2f}")
    ov = [y for y in both if OVERLAP[0] <= y <= OVERLAP[1]]
    lo, hi = min(ratios[y] for y in ov), max(ratios[y] for y in ov)
    check(failures, PALLANTE_BAND[0] - 0.01 <= lo and hi <= PALLANTE_BAND[1] + 0.01, "4b",
          "the overlap sits in the stated band", f"{lo:.3f}-{hi:.3f}")

    # 4c. THE BLOCK WE IMPORT, against Pallante. His state-year series runs to FY2019
    #     and 09_pallante_test.py already labels 2007-2019 "out of scope, no contract
    #     file" - which is exactly the span this script fills. Validating only the
    #     2001-06 overlap and importing 2007-2018 on faith would leave the imported
    #     years checked against nothing.
    pallante_gate(failures, keep, last_year)

    # 4d. The county-specific level step. This is the threat a national comparison
    #     cannot see: a COMMON factor at the seam is absorbed by year FE, but a
    #     county-specific one is a cross-sectional discontinuity, and the exposure
    #     loading is built to exploit exactly that.
    lambda_gate(failures, ours, theirs)

    # 4e. County support either side of the seam. Reported, not gated: README section 4
    #     item 8 already records threshold-driven support changes as an accepted
    #     limitation, and the FPDS micro-purchase threshold moves again at 2015 and 2018.
    print("\n    county support across the seam (reported, not gated)")
    for y in range(OVERLAP[1] - 1, min(last_year, OVERLAP[1] + 3) + 1):
        n_ours = int((ours[(ours.year == y) & (ours.dollars_nominal != 0)]).shape[0])
        n_thm = int(theirs[theirs.year == y].ccg.nunique())
        src = "DCADS " if y <= OVERLAP[1] else "theirs"
        print(f"      FY{y}  {src}  ours {n_ours:>5,}   theirs {n_thm:>5,}")


def pallante_gate(failures, keep, last_year):
    """Muratori 2007-2018 against Pallante's published state-year series.

    THE GATE IS SELF-CALIBRATING, and it has to be. Pallante's series and his are
    two independent constructions, so their ratio is never 1 and no hand-set band
    means anything. What CAN be asked is whether the imported years agree with
    the benchmark as well as the years we have already accepted do: run the same
    comparison on FY2001-2006, where our own DCADS panel independently confirms
    his numbers, and require the imported block to sit inside that band.

    For reference, the same comparison run on our DCADS panel gives 1.000 in every
    year FY1976-2006 - Pallante's series is what 09_pallante_test.py verified us
    against, so our side of this table is exact by construction and his is not.
    """
    print("\n    the imported block against Pallante's state-year series")
    rows = dc.read_pallante_csv("data_replication/aggregates/all_dod_statename_noadj.csv")
    codes = dc.read_pallante_csv("data_replication/state_codes.txt")
    if rows is None or codes is None:
        print("      Pallante archive not present - the imported block is UNVALIDATED")
        failures.append(("4c", "Pallante archive absent, imported block unvalidated"))
        return
    fips = {r["statename"]: r["statefips"].zfill(2) for r in codes}
    his = collections.defaultdict(float)
    for r in rows:
        code = fips.get(r["statename"])
        if code is not None:
            his[int(r["fy_year"])] += float(r["tot_spending"] or 0)

    mine = keep.groupby("year")["dollars"].sum()
    ratios = {y: float(mine[y]) / his[y] for y in sorted(his)
              if y in mine.index and his[y] and y <= last_year}
    known = [r for y, r in ratios.items() if OVERLAP[0] <= y <= OVERLAP[1]]
    imported = {y: r for y, r in ratios.items() if y >= dc.SEAM_YEAR}
    if not known or not imported:
        failures.append(("4c", "no calibration or no imported years"))
        return

    print(f"      {'fy':<7}{'muratori $bn':>15}{'pallante $bn':>15}{'ratio':>9}")
    for y in sorted(imported):
        print(f"      {y:<7}{mine[y]/1e9:>15,.1f}{his[y]/1e9:>15,.1f}{ratios[y]:>9.3f}")
    klo, khi = min(known), max(known)
    ilo, ihi = min(imported.values()), max(imported.values())
    print(f"      calibration, FY{OVERLAP[0]}-{OVERLAP[1]} (independently confirmed): "
          f"{klo:.3f}-{khi:.3f}, spread {khi - klo:.3f}")
    print(f"      imported,    FY{dc.SEAM_YEAR}-{last_year}:                         "
          f"{ilo:.3f}-{ihi:.3f}, spread {ihi - ilo:.3f}")
    # Tolerance is half the calibration spread on each side: the imported band may
    # sit anywhere the known-good band does, and may be no more than half again as
    # wide, which is what distinguishes "a different construction" from "drifting".
    pad = 0.5 * (khi - klo)
    ok = (ilo >= klo - pad) and (ihi <= khi + pad) and (ihi - ilo) <= 1.5 * (khi - klo)
    check(failures, ok, "4c", "the IMPORTED block agrees as well as the known-good years",
          f"{ilo:.3f}-{ihi:.3f} vs {klo:.3f}-{khi:.3f}")


def lambda_gate(failures, ours, theirs):
    """Is the per-county ours/theirs ratio correlated with defense exposure?

    If it is, the splice does not merely add noise to the endogenous regressor -
    it shifts it systematically with the instrument's own loading, and the
    post-2006 first stage is biased rather than attenuated. That is the
    difference between a limitation and a stop.
    """
    print("\n    the county-specific level step, lambda_c")
    inst = dc.PANEL_OUT_DIR / "instrument_ccg_year.parquet"
    if not inst.exists():
        print("      instrument_ccg_year.parquet absent - run 14 first; lambda test skipped")
        failures.append(("4d", "instrument absent, lambda_c untested"))
        return
    exp = (pd.read_parquet(inst, columns=["ccg", "exposure_fixed"])
           .drop_duplicates("ccg").set_index("ccg")["exposure_fixed"])

    a = ours[ours.year.between(*OVERLAP)].groupby("ccg")["dollars_nominal"].sum()
    b = theirs[theirs.year.between(*OVERLAP)].groupby("ccg")["dollars"].sum()
    df = pd.DataFrame({"ours": a, "theirs": b}).dropna()
    df = df[(df.ours > 1e6) & (df.theirs > 1e6)]          # ratios need both sides real
    df["lam"] = np.log(df.theirs / df.ours)
    df["exposure"] = exp.reindex(df.index)
    df = df.dropna(subset=["exposure"])
    df["exp_rank"] = df.exposure.rank(pct=True)

    r = float(np.corrcoef(df.lam, df.exp_rank)[0, 1])
    n = len(df)
    tstat = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
    q = np.percentile(df.lam, [10, 50, 90])
    print(f"      {n:,} CCGs with >$1m on both sides over FY{OVERLAP[0]}-{OVERLAP[1]}")
    print(f"      log ratio   p10 {q[0]:+.3f}   median {q[1]:+.3f}   p90 {q[2]:+.3f}")
    print(f"      corr(log lambda_c, exposure percentile) = {r:+.4f}   t = {tstat:+.2f}")

    hq = df[df.index.isin(HQ_COUNTIES)]
    if len(hq):
        print("      contractor-HQ counties, for the place-of-performance tilt:")
        for ccg, row in hq.iterrows():
            print(f"        {HQ_COUNTIES[ccg]:<22} log ratio {row.lam:+.3f}"
                  f"   (panel median {q[1]:+.3f})")
        print(f"      HQ mean {hq.lam.mean():+.3f} against non-HQ "
              f"{df[~df.index.isin(HQ_COUNTIES)].lam.mean():+.3f}")

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(dc.LOG_DIR / "muratori_audit_lambda.csv")
    check(failures, abs(tstat) < 3.0, "4d",
          "lambda_c is not tilted by defense exposure", f"t = {tstat:+.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
