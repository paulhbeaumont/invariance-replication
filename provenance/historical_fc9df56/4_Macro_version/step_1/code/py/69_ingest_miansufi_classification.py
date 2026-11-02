"""Recover the Mian-Sufi (2014) NAICS4 non-tradable industry classification.

WS2 step 4 bins county x industry entry contributions by whether the industry
is non-tradable. The classification everyone cites for that is Mian and Sufi
(2014), "What Explains the 2007-2009 Drop in Employment?", Econometrica 82(6),
2197-2223, doi 10.3982/ECTA10451.

Their replication kit ships NO standalone classification file. The
classification lives as industry-invariant columns on the 458,098-row county x
industry panel, and their own do file recovers it with `egen tag(naics)` for
Appendix Table 1. This producer downloads the kit, verifies it, runs that same
collapse, and writes reference/miansufi_naics4_classification.csv with one row
per industry (294).

Source (pinned):
  https://www.econometricsociety.org/publications/econometrica/2014/11/01/
    what-explains-2007-2009-drop-employment/supp/10451_Data_and_Programs.zip
  sha256 efd632955ae01e6d69e9f30341cda9304f03d6ccfafe6ce050d3ddda5a31f245
The online appendix sits beside it as
  .../supp/Supplement-to-What-Explains-the-2007-2009-Drop-in-Employment.pdf
  sha256 40d27b9988057847554fc34923812ee4385be6ddd45de9d51ea0af7a14bb9731
and is not needed to build the table.

TWO NON-TRADABLE DEFINITIONS SHIP IN THE FILE AND THEY DIFFER. A consumer must
state which one it bins on.
  indcat == 2, 26 industries: retail (14 in NAICS 44, 8 in NAICS 45) plus 4
    food-services industries (7221, 7222, 7223, 7224). All 26 sit in the two
    least geographically concentrated quartiles, 19 in Q1 and 7 in Q2. This is
    what the authors' regressions select with `if indcat==2`.
  nontradable_strict == 1, 20 industries: the same set minus six durables
    retailers - 4411 automobile dealers, 4412 other motor vehicle dealers,
    4413 auto parts and tire stores, 4421 furniture stores, 4422 home
    furnishings stores, 4431 electronics and appliance stores. This is the
    paper's headline "restaurants and retail" definition.

The other categories: `construction` (23) is wider than NAICS 23 - it adds
construction-supplying manufacturing and wholesale (3211, 3212, 3219, 3273,
3323, 3371, 4233), the two building-materials retailers (4441, 4442), real
estate (5311, 5312, 5313), architectural and engineering services (5413), and
logging (1133). `tradable` (82): 76 industries have trade per worker at or
above $10,000 and six fall below it (3118 bakeries $3,473, 3231 printing
$9,045, 3315 foundries $5,331, 3327 machine shops $5,734, 3372 office
furniture $8,631, 5112 software publishers $2,963), consistent with the
paper's second clause admitting the top quartile of trade per worker. The file
is the authoritative assignment; do NOT re-derive bins from trade_per_worker
and a threshold.

CONCORDING TO EFSY - the one thing not to miss. EFSY's `naics12` uses the
identical CBP string coding (`1133//`, `236///`, six characters), so no
crosswalk table is needed:
  - 1998-2018 EFSY carries only full 6-digit codes, so the NAICS4 bin is a
    4-character truncation. 1975-1997 mixes 2-, 3-, 4- and 6-digit padded
    codes and is SIC-bridged. WS2 step 4 now reports both sides of the 1998
    break, so that window IS checked here: `efsy_coverage` below runs the
    assertion over the whole 1975-2016 span and is the thing this docstring
    used to say was missing.
  - PRE-1998 COVERAGE IS NOT 100% AND THE SHORTFALL IS REPORTED, NOT DROPPED.
    Between 0.36% and 1.51% of EFSY employment sits on four-digit codes outside
    the 294, worst around 1985, and a further 0.21-0.29% sits on codes producer
    11 could not push down to four digits at all. Producer 30 gives each of
    those a NAMED BIN (INVARIANCE_TEST_PROTOCOL.md A.3.1) and gates their size;
    neither is folded into "all other industries", because that row would
    otherwise mean two different things in the two column groups of the exhibit.
  - Over 1998-2016, every EFSY 4-digit code except ONE appears in the 294. The
    exception is 7225, which EFSY uses in every year and which absorbs the
    authors' 7221 (full-service restaurants) and 7222 (limited-service eating
    places). Both parents are non-tradable and both are nontradable_strict, so
    the merge is unambiguous: MAP 7225 TO NON-TRADABLE, STRICT. Naive
    truncation alone leaves 7225 unbinned - 8.3% of EFSY employment in 2016,
    and the single largest non-tradable bin.
  - With that one patch, matched coverage of EFSY employment is 100%. Without
    it, 93.5% in 1998 falling to 91.7% in 2016.
  - Six Mian-Sufi codes never appear in EFSY 1998-2016 and are simply unused:
    5161, 5173, 5175, 5181, 7221, 7222.

Usage terms: public replication material distributed by the Econometric
Society. Cite the paper.

Companion producer: 70_ingest_rajan_zingales.py, the other WS2 step 4 bin.
"""

import hashlib
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = ("https://www.econometricsociety.org/publications/econometrica/2014/11/01/"
       "what-explains-2007-2009-drop-employment/supp/10451_Data_and_Programs.zip")
SHA256 = "efd632955ae01e6d69e9f30341cda9304f03d6ccfafe6ce050d3ddda5a31f245"
MEMBER = ("MianSufiEconometrica_PublicReplicationFiles/"
          "miansufieconometrica_countyindustrylevel.dta")

REF_DIR = Path(__file__).resolve().parent.parent / "reference"
OUT = REF_DIR / "miansufi_naics4_classification.csv"

# The EFSY coverage assertion. WS2 step 4 reports both sides of the 1998 NAICS
# break, so it runs over the whole span rather than the post-break half.
COVERAGE_YEARS = (1975, 2016)
RESIDUAL3 = 100_000            # 11_build_employment_cells.py's root encoding
PATCH_7225 = 7225              # absorbs the authors' 7221 and 7222; see docstring
# Pre-stated ceilings, matching 30_entry_margin_bins.py's G3a and G3b. These cap
# how much employment the NAMED residual bins carry; they never license a drop.
MAX_ROOT_SHARE = 0.01
MAX_UNCLASS_SHARE = 0.05

COLS = ["naics", "industry", "indcat", "nontradable_strict",
        "Iemp2_2007", "Iherf", "Ihcat2", "export_worker"]
CODES = {"other": 0, "tradable": 1, "non-tradable": 2, "construction": 3}


def fetch_zip(tmpdir):
    """Download the kit and verify it. Falls back to curl where the Python
    install carries no CA bundle, which is the case on at least one of our
    machines (python.org 3.14 without Install Certificates)."""
    path = Path(tmpdir) / "10451_Data_and_Programs.zip"
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            path.write_bytes(r.read())
    except urllib.error.URLError:
        subprocess.run(["curl", "-sSLf", "--max-time", "300",
                        "-A", "Mozilla/5.0", "-o", str(path), URL], check=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == SHA256, f"replication kit hash changed: {digest}"
    return path


def collapse(zip_path):
    """The authors' Appendix Table 1 collapse: one row per tagged naics."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open(MEMBER) as fh:
            raw = pd.read_stata(fh, columns=COLS, convert_categoricals=True)

    ind = raw.drop_duplicates(subset=["naics"]).copy()
    assert len(ind) == 294, f"industry count {len(ind)} != 294"
    assert ind["Iemp2_2007"].notna().all(), "an industry has no 2007 employment"

    ind["naics4"] = ind["naics"].str.replace("/", "", regex=False)
    assert ind["naics4"].str.fullmatch(r"\d{4}").all(), "a code is not 4 digits"

    ind["indcat_label"] = ind["indcat"].astype(str)
    ind["indcat"] = ind["indcat_label"].map(CODES)
    assert ind["indcat"].notna().all(), "an industry category is unrecognised"

    ind["nontradable_strict"] = ind["nontradable_strict"].fillna(0).astype(int)
    ind["Iemp2_2007"] = ind["Iemp2_2007"].astype("float64")
    ind["emp_share_2007_pct"] = (
        100.0 * ind["Iemp2_2007"] / ind["Iemp2_2007"].sum()
    )

    return ind.rename(columns={
        "industry": "industry_name",
        "Iemp2_2007": "emp_2007",
        "Iherf": "herf_geo",
        "Ihcat2": "herf_quartile",
        "export_worker": "trade_per_worker",
    })[[
        "naics4", "industry_name", "indcat", "indcat_label",
        "nontradable_strict", "emp_2007", "emp_share_2007_pct", "herf_geo",
        "herf_quartile", "trade_per_worker",
    ]].sort_values("naics4").reset_index(drop=True)


def check(out):
    counts = out["indcat_label"].value_counts().to_dict()
    assert counts == {"other": 163, "tradable": 82, "non-tradable": 26,
                      "construction": 23}, counts
    assert out["nontradable_strict"].sum() == 20, "strict set is not 20"
    strict_sectors = set(out.loc[out.nontradable_strict == 1, "naics4"].str[:2])
    assert strict_sectors == {"44", "45", "72"}, strict_sectors
    for code, label in [("7221", "non-tradable"), ("4451", "non-tradable"),
                        ("2361", "construction"), ("3361", "tradable"),
                        ("2211", "other")]:
        got = out.loc[out.naics4 == code, "indcat_label"].iloc[0]
        assert got == label, f"{code} is {got}, expected {label}"
    assert abs(out["emp_share_2007_pct"].sum() - 100.0) < 1e-6
    return counts


def efsy_coverage(out):
    """How much EFSY employment the 294 industries reach, year by year.

    THIS IS THE ASSERTION THE DOCSTRING USED TO SAY WAS MISSING. It is a
    measurement of what the classification does NOT reach, not a filter: every
    code counted here as unclassified is reported by producer 30 in the named
    bin "Not in the Mian-Sufi 294", and every root is reported in "Unresolved
    NAICS root". A.3.1 forbids dropping either, so what is gated is their size.

    Needs producer 11's EFSY cell panel and fails loudly without it. It runs
    BEFORE the CSV is written, so a span the classification cannot honestly
    cover never leaves an artefact behind.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dcads_common as dc  # noqa: PLC0415  - optional, see docstring

    known = set(out["naics4"].astype(int)) | {PATCH_7225}
    lo, hi = COVERAGE_YEARS
    print(f"\n  EFSY COVERAGE ASSERTION, {lo}-{hi}")
    print(f"  {'year':>6}{'all emp':>14}{'matched %':>11}{'unclassified %':>16}"
          f"{'root %':>9}")
    worst_u, worst_uy, worst_r, worst_ry = 0.0, None, 0.0, None
    for y in range(lo, hi + 1):
        f = dc.EMP_OUT_DIR / "naics4_alloc" / f"y{y}.parquet"
        if not f.exists():
            raise SystemExit(f"FAIL: {f} is missing; run producer 11 first.")
        g = pd.read_parquet(f, columns=["code", "emp"]).groupby("code")["emp"].sum()
        tot = float(g.abs().sum())
        root = float(g[g.index >= RESIDUAL3].abs().sum()) / tot
        gen = g[g.index < RESIDUAL3]
        unc = float(gen[[c not in known for c in gen.index]].abs().sum()) / tot
        if unc > worst_u:
            worst_u, worst_uy = unc, y
        if root > worst_r:
            worst_r, worst_ry = root, y
        print(f"  {y:>6}{tot:>14,.0f}{100 * (1 - unc - root):>10.2f}%"
              f"{100 * unc:>15.2f}%{100 * root:>8.3f}%")

    ok = True
    for lab, sh, tol, yr, binname in (
            ("unclassified", worst_u, MAX_UNCLASS_SHARE, worst_uy,
             "Not in the Mian-Sufi 294"),
            ("unresolved root", worst_r, MAX_ROOT_SHARE, worst_ry,
             "Unresolved NAICS root")):
        good = sh <= tol
        ok = ok and good
        print(f"  [{'ok ' if good else 'FAIL'}] worst-year {lab} employment: "
              f"{yr} at {100 * sh:.3f}%, ceiling {100 * tol:.0f}%. Reported by "
              f"producer 30 in the named bin '{binname}'.")
    if not ok:
        raise SystemExit("FAIL: the classification misses too much EFSY "
                         "employment for a named residual bin to carry "
                         "honestly; it needs a real bridge, not a truncation.")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out = collapse(fetch_zip(tmp))
    counts = check(out)
    efsy_coverage(out)

    REF_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# Mian and Sufi (2014, Econometrica 82(6), 2197-2223), "
                "\"What Explains the 2007-2009 Drop in Employment?\"\n")
        f.write("# NAICS4 industry classification, one row per industry "
                "(294). Built by 69_ingest_miansufi_classification.py, whose "
                "docstring carries the caveats.\n")
        f.write(f"# Source: {URL}\n")
        f.write(f"# sha256 of replication zip: {SHA256}\n")
        f.write(f"# Member: {MEMBER}\n")
        f.write("# indcat: 0 other, 1 tradable, 2 non-tradable, 3 "
                "construction (counts 163 / 82 / 26 / 23).\n")
        f.write("# TWO non-tradable definitions: indcat==2 is 26 industries; "
                "nontradable_strict==1 is the 20 retail-and-restaurant "
                "industries (NAICS 44, 45, 722). State which one you bin on.\n")
        f.write("# EFSY: truncate naics12 to 4 characters, then map 7225 to "
                "non-tradable, strict - it absorbs 7221 and 7222 and is 8.3% "
                "of 2016 EFSY employment. Coverage is then 100%.\n")
        f.write("# emp_2007 is CBP county-industry employment summed to the "
                "industry; emp_share_2007_pct is its share of the "
                "294-industry total, as in the authors' Appendix Table 1.\n")
        f.write("# herf_geo is the industry geographical Herfindahl across "
                "counties; herf_quartile is its quartile (1 = least "
                "concentrated).\n")
        f.write("# trade_per_worker is imports plus exports per worker, USD, "
                "0 for industries with no trade measure.\n")
        f.write("# NAICS vintage is 2002/2007 (construction is 236-238, not "
                "the 1997 233-235).\n")
        out.to_csv(f, index=False)

    print(f"wrote {OUT}: {len(out)} industries, {counts}, "
          f"{int(out.nontradable_strict.sum())} strict non-tradable. "
          "All checks passed.")


if __name__ == "__main__":
    sys.exit(main())
