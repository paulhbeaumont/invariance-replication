#!/usr/bin/env python3
"""Shared paths, zoned-decimal decoding and layout handling for the DCADS parsers.

The numbered scripts cannot import one another because their names begin with a
digit, so everything reusable lives here: path resolution, the IBM zoned-decimal
table, the layout registry, and the chunked fixed-width reader.

Code lives in a local Git checkout. All raw and intermediate data live in the
shared Dropbox project. INVARIANCE_DROPBOX_ROOT can override the documented
Dropbox project path when a synced machine mounts the shared folder elsewhere.
"""

import collections
import csv
import json
import os
import re
import zipfile
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
LAYOUT_DIR = CODE_DIR.parent / "layouts"
REFERENCE_DIR = CODE_DIR.parent / "reference"
PROJECT_ROOT = CODE_DIR.parents[3]  # code -> step_1 -> 4_Macro_version -> repo root

_DEFAULT_DROPBOX_HOME = (
    Path.home() / "Library" / "CloudStorage" / "Dropbox-Matray"
)
DROPBOX_HOME = Path(
    os.environ.get("INVARIANCE_DROPBOX_HOME", _DEFAULT_DROPBOX_HOME)
).expanduser()
_DROPBOX_PROJECT_CANDIDATES = (
    DROPBOX_HOME / "_AdrienTeam" / "Chenzi_Adrien" / "Invariance_Agg_Disagg",
    DROPBOX_HOME / "Invariance_Agg_Disagg",
)


def resolve_dropbox_project_root():
    explicit = os.environ.get("INVARIANCE_DROPBOX_ROOT")
    if explicit:
        root = Path(explicit).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(
                f"INVARIANCE_DROPBOX_ROOT is not a directory: {root}"
            )
        return root
    hits = [path for path in _DROPBOX_PROJECT_CANDIDATES if path.is_dir()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(
            "Shared Dropbox project not found. Set INVARIANCE_DROPBOX_ROOT to "
            "the synced Invariance_Agg_Disagg folder."
        )
    raise RuntimeError(
        "Multiple Dropbox project mounts found. Set INVARIANCE_DROPBOX_ROOT "
        f"explicitly: {hits}"
    )


DROPBOX_PROJECT_ROOT = resolve_dropbox_project_root()
CURRENT_PAPER_DATA_ROOT = DROPBOX_PROJECT_ROOT / "1_current_paper" / "_data"
MACRO_DATA_ROOT = DROPBOX_PROJECT_ROOT / "_data"

# Tables, figures and the deck live in the shared Dropbox project, not in the
# Git checkout. The repository carries code; everything a co-author reads is on
# Dropbox, where they already have it. A producer that writes an exhibit
# resolves its directory from here rather than from its own file location,
# because a path built off __file__ lands in whichever checkout happened to run
# it and is then invisible to everyone else.
MACRO_RESULTS_ROOT = DROPBOX_PROJECT_ROOT / "4_Macro_version" / "Results"
MACRO_PRESENTATIONS_ROOT = (DROPBOX_PROJECT_ROOT / "4_Macro_version"
                            / "Presentations")

SOURCE_DIR = CURRENT_PAPER_DATA_ROOT / "source" / "DCADS"
OUT_DIR = CURRENT_PAPER_DATA_ROOT / "temp" / "dcads_year"
LOG_DIR = PROJECT_ROOT / "batch_logs" / "macro_version"

# USAspending's annual Department of Defense prime-contract archives. The raw
# zips and transaction-level parquets live in the shared Dropbox data tree.
USASPENDING_SOURCE_DIR = MACRO_DATA_ROOT / "source" / "USAspending"
USASPENDING_OUT_DIR = MACRO_DATA_ROOT / "temp" / "usaspending_year"
USASPENDING_VERIFICATION_REPORT = (
    USASPENDING_OUT_DIR / "usaspending_verification_report.csv"
)

# The cleaned employment cells built by 11_build_employment_cells.py. A sibling of
# OUT_DIR rather than a subfolder of it, because OUT_DIR is the parsed contract
# panel and the two are different objects with different row counts.
EMP_OUT_DIR = CURRENT_PAPER_DATA_ROOT / "temp" / "employment"

# The estimation sample uses one county set fixed before any period, horizon or
# aggregation choice. A CCG belongs to it only when county-total employment is
# strictly positive in every DCADS analysis year. The 1975 observation is needed
# as the lag for the first usable outcome year, and FY2006 is the final DCADS
# contract year.
POSITIVE_EMP_FIRST_YEAR = 1975
POSITIVE_EMP_LAST_YEAR = 2006

# Part 1.4's own output: contracts collapsed to CCG x year, county personal
# income, and the joined county panel that carries the shock. Built by
# 12_build_county_income.py and 13_build_county_panel.py.
PANEL_OUT_DIR = CURRENT_PAPER_DATA_ROOT / "temp" / "county_panel"
HARMONIZED_REPORT = PANEL_OUT_DIR / "contracts_harmonized_report.csv"
POST_GRID_DIR = MACRO_DATA_ROOT / "temp" / "post_first_stage_full_grid"
POST_GRID_LABELED = POST_GRID_DIR / "post_first_stage_full_grid_labeled.csv"
SYMMETRIC_FIRST_STAGE_DIR = (
    MACRO_DATA_ROOT / "temp" / "symmetric_nonoverlap_first_stage"
)
# 41 anchors the post period at 1998. 46 sweeps that anchor over 1998, 1999 and
# 2000 and writes its own panels here, so the two pipelines never share a file.
POST_ANCHOR_SWEEP_DIR = MACRO_DATA_ROOT / "temp" / "post_anchor_sweep"

# The macro application's own Dropbox scratch folder, distinct from
# MACRO_DATA_ROOT. LOG_DIR is the git checkout, so anything written only there
# is invisible to a co-author who has the Dropbox project and not the repo;
# result tables meant to be read outside the checkout are mirrored here.
MACRO_VERSION_TEMP_DIR = DROPBOX_PROJECT_ROOT / "4_Macro_version" / "data" / "temp"

# The SAME-NAMED folder inside the git checkout. The two are easy to confuse and
# are not the same place: MACRO_VERSION_TEMP_DIR is on Dropbox and visible to a
# co-author without the repo, this one is local and visible only here. It is
# gitignored (`4_Macro_version/data/**`) and already hosts the NETS staging
# mirror and the industry-axis scratch. Used by the local-first BDS block below.
MACRO_VERSION_LOCAL_ROOT = PROJECT_ROOT / "4_Macro_version"

# BEA CAINC1, county personal income. Downloaded by 12_build_county_income.py as
# a static zip - no API key - and kept under _data/source/ beside DCADS, so the
# raw archive is versioned by its sha256 and never by git.
BEA_SOURCE_DIR = CURRENT_PAPER_DATA_ROOT / "source" / "BEA"
CAINC1_URL = "https://apps.bea.gov/regional/zip/CAINC1.zip"

# The June 2003 OMB metropolitan/micropolitan delineation, downloaded by
# 21_ingest_cbsa.py. It is the FIRST CBSA vintage - the one whose definition
# Auerbach, Gorodnichenko and Murphy quote in their footnote 4, "one or more
# counties with an urban center of at least 10,000 people" - and it sits in the
# middle of our 1975-2018 panel, so the county set it produces is one fixed set
# rather than a definition that moves with t. Kept beside DCADS and BEA under
# _data/source/, versioned by sha256 and never by git; the parsed county list is
# small enough to be a tracked reference file, which is where CBSA_CROSSWALK
# points.
CBSA_SOURCE_DIR = CURRENT_PAPER_DATA_ROOT / "source" / "CBSA"
CBSA_URL = ("https://www2.census.gov/programs-surveys/metro-micro/geographies/"
            "reference-files/2003/historical-delineation-files/03mfips.txt")

# The Census NAICS-to-NAICS concordances, one workbook per revision, downloaded
# by pipeline/07_naics_vintage_concordances.py. Four revisions fall inside the
# span this project touches (1997->2002, 2002->2007, 2007->2012, 2012->2017),
# and every industry source has to be put on one vintage before a cell built
# from one can be compared with a cell built from another. Kept beside DCADS,
# BEA and CBSA under _data/source/, versioned by sha256 and never by git; the
# parsed pair tables are small enough to be tracked reference files, which is
# where the reference/naics_*_census.csv files come from.
#
# The Census file names are NOT consistently formatted across revisions - the
# 1997/2002 pair spells out "_NAICS_to_..._NAICS" and the later ones use a bare
# "_to_" - so the names are carried literally in the producer rather than built
# from a pattern. Do not "tidy" them into a format string.
CENSUS_NAICS_SOURCE_DIR = CURRENT_PAPER_DATA_ROOT / "source" / "CensusNAICS"
CENSUS_NAICS_BASE_URL = "https://www.census.gov/naics/concordances"

# Census Business Dynamics Statistics, the public establishment entry/exit
# series WS2 step 2 benchmarks the NETS birth rates against. Downloaded by
# py/74_ingest_bds.py as plain CSVs - no API key - and kept beside DCADS, BEA,
# CBSA and CensusNAICS under _data/source/, versioned by sha256 and never by
# git. BDS_RELEASE is the release vintage, not a data year: the whole 1978-2023
# series is re-estimated at each release, so mixing releases inside one exhibit
# mixes two different estimates of the same year. Pin it, and change it only
# deliberately.
#
# BDS IS DELIBERATELY LOCAL-FIRST, AND IS THE ONE EXCEPTION TO THE DROPBOX RULE
#
# The project rule is that data live only under the shared Dropbox project and
# that there is no repo-local fallback. BDS is an exception, made on purpose on
# 2026-09-01: every downstream consumer reads the LOCAL copy in the checkout
# under 4_Macro_version/data/temp/bds, and the Dropbox copy is a mirror kept so
# a co-author without the checkout can still see the files. The reason to make
# it an exception at all is that BDS is small (about 300 MB of public CSV),
# fully public, and re-downloadable, so a local copy costs little and removes a
# network-and-Dropbox dependency from every step-2 run.
#
# Read it through bds_path() below, never by joining BDS_MIRROR_DIR yourself.
# That function resolves the local copy and fails loud if it is missing, which
# is what keeps "future analysis reads local files" a property of the code
# rather than a convention someone has to remember. 4_Macro_version/data/** is
# gitignored, so nothing here is ever committed.
BDS_SOURCE_DIR = MACRO_VERSION_LOCAL_ROOT / "data" / "source" / "BDS"
BDS_RELEASE = 2023
BDS_BASE_URL = ("https://www2.census.gov/programs-surveys/bds/tables/"
                f"time-series/{BDS_RELEASE}")
BDS_OUT_DIR = MACRO_VERSION_LOCAL_ROOT / "data" / "temp" / "bds"
BDS_MIRROR_DIR = MACRO_DATA_ROOT / "temp" / "bds"


def bds_path(name):
    """The LOCAL path to one ingested BDS parquet, or a loud failure.

    Every consumer of the BDS benchmark goes through here. It never falls back
    to the Dropbox mirror: a run that silently read the mirror would be reading
    whatever another machine last wrote, which is exactly the ambiguity the
    local-first decision exists to remove. If the file is absent the fix is to
    run the ingest, and the message says so.
    """
    path = BDS_OUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. BDS is read from the LOCAL checkout copy and "
            "never from the Dropbox mirror; run "
            "`python step_1/code/py/74_ingest_bds.py` to build it.")
    return path

# The county defense procurement panel shared by Umberto Muratori (with Pedro
# Juarros and Daniel Valderrama), received 2026-08-10. It is an external source
# comparison for the raw USAspending production block. Its provenance and usage
# terms are recorded beside it in a README.md.
#
# Note the sub-path: this is 4_Macro_version/data/source/, not the
# 1_current_paper/_data/source/ tree used by the public raw inputs.
_MURATORI_REL = (Path("4_Macro_version") / "data" / "source"
                 / "data_govspending_adrien.dta")
MURATORI_SOURCE = DROPBOX_PROJECT_ROOT / _MURATORI_REL

# FY2007 is the first year that cannot come from DCADS. This is a property of the
# archive, not a tuning parameter, which is why it is a constant here rather than
# a flag on 13 and 14: both must splice at the same year or their contract panels
# stop agreeing, and 14's checks 4 and 6 assert exactly that agreement.
SEAM_YEAR = 2007

# Zip members are extracted here, parsed, and deleted again, so the uncompressed
# text never lands in the repo or in Dropbox.
SCRATCH_DIR = Path("/tmp") / "dcads_unzip"

MANIFEST = SOURCE_DIR / "manifest.csv"
TSS = REFERENCE_DIR / "dcads_tss.json"

# The ZARK Locator File, naId 1937373. Not a contract file and claimed by no
# layout, which is why the parsers skip it; it is the pre-1984 geography, read by
# 07_add_county_fips.py. See load_zark_crosswalk.
ZALF = SOURCE_DIR / "DCADS" / "series" / "RG330.DEFCON.ZALF"
ZARK_SUPPLEMENT = REFERENCE_DIR / "zark_fips_supplement.csv"

# Part 1.4's county-harmonisation inputs, read by 10_build_county_universe.py.
# county_ccg_crosswalk.csv is that script's output and every later consumer's
# input; the other two are hand-transcribed sources it is built from.
COUNTY_CHANGES = REFERENCE_DIR / "county_changes_census.csv"
VA_COMBINATIONS = REFERENCE_DIR / "va_bea_combinations.csv"
CCG_CROSSWALK = REFERENCE_DIR / "county_ccg_crosswalk.csv"

# 21_ingest_cbsa.py's output and 23_sample_screen_grid.py's input: one row per
# county FIPS that belongs to a 2003 CBSA. Tracked, unlike everything under
# _data/, because it is 1,838 rows of public delineation and a reader who cannot
# reach census.gov should still be able to reproduce the rural screen.
CBSA_CROSSWALK = REFERENCE_DIR / "county_cbsa_2003.csv"

# The county-to-commuting-zone bridge, 3,141 counties on the 1990 vintage mapping
# to 741 commuting zones. Already local and sha256-pinned by
# 2_illustration_aggregation/Immigration_Innovation_Growth/code/10_prepare_commuting_zone_bridge_duckdb.py,
# which is why the CCG set is labelled on the 1990 vintage: the CZ rung then
# needs no new reference file. Used by check 5 of 10_build_county_universe.py.
_CZ_CANDIDATES = (
    PROJECT_ROOT / "2_illustration_aggregation" / "Immigration_Innovation_Growth"
                 / "reference" / "county_to_commuting_zone.parquet",
    PROJECT_ROOT / "2_illustration_aggregation" / "fracking_controversy"
                 / "reference" / "cw_cty_czone.dta",
)
CZ_CROSSWALK = next((p for p in _CZ_CANDIDATES if p.exists()), None)

# The desuppressed County Business Patterns panel behind Part 1.3. Restricted and
# deliberately outside the project folder, like the French SCR intermediates.
EFSY_PANEL = (
    DROPBOX_HOME / "_Source" / "US" / "CBP_EFSY" / "efsy_panel_naics.parquet"
)

# EFSY files a statewide residual under fipscty=999 - employment the Census could
# not place in a county - and it must not be attributed to a county. That is right
# for fifty states and wrong for one. The District of Columbia is a single
# county-equivalent, so 11999 IS 11001, and EFSY writes DC that way in 1976-2003.
# Dropping it deleted DC's employment for 28 of the 32 panel years while DC carried
# $35.04bn of contracts.
#
# The recode is DISCOVERED, not asserted: a residual becomes the county only where
# the state has no non-999 county rows at all in that year, so the whole of the
# state's employment is sitting in the residual and no double count is possible.
# What is declared here is the answer that sweep produces, and both consumers fail
# on an undeclared state-year and on a declared one that stops occurring. Sweeping
# all 51 state equivalents over 1975-2018, state 11 in 1976-2003 is the only case;
# every other 999 row sits beside real counties and is a genuine unplaceable
# residual, which stays dropped.
#
# It lives here rather than in either consumer because 10_build_county_universe.py
# applies it to county totals and 11_build_employment_cells.py applies it to
# county-by-industry cells, and two copies of a declaration is one too many.
#
# state: (county_fips5, first_year, last_year, why)
EFSY_RESIDUAL_AS_COUNTY = {
    "11": ("11001", 1976, 2003,
           "District of Columbia. One county-equivalent, so the statewide "
           "residual is the county. EFSY has real 11001 rows in 1975 and "
           "2004-06, which is why a pooled set comparison could not see the "
           "hole - DC was present in the universe and absent from 28 of its "
           "years."),
}

# ---------------------------------------------------------------------------
# Hand-authored geography overrides for the FIPS eras, FY1984 onward.
#
# zark_fips_supplement.csv does this job for FY1966-83 and is keyed
# (fiscal_year, state_code, county_code); this is the same idea for the years
# that need no crosswalk. It lives here rather than in 07 because 08 has to know
# about it too: an override deliberately leaves a well-formed county code
# unresolved, which is otherwise exactly what 08's completeness check exists to
# catch. One definition, both readers.
#
# It is kept in code rather than in a CSV because there is one row in it and a
# file would imply a class of them. Each entry declares how many rows it expects
# to match and 07 counts what it actually matched, so an override that stops
# matching - or starts matching more than it should - fails the run instead of
# silently doing nothing. That guard is the reason this is safe to have at all.
#
# (fiscal_year, state_code, county_code) -> (county_fips5, state_fips2,
#                                            fips_source, expected rows, why)
NATIVE_OVERRIDES = {
    ("1984", "06", "36103"): (
        "06037", "06", "manual_override", 1,
        "one FY1984 row, contract F3460184C0986, whose geography block is damaged "
        "in the source: county_code 36103 names Suffolk County NEW YORK while "
        "state_code is 06 and state_name is California, city_name is blank, and "
        "city_code 0633961 and location_code 0636812 disagree with each other. "
        "The same contract has a second FY1984 action - same contract number, "
        "same DUNS 066643735, same action_date 840322, same FSC 1650, same "
        "purchasing office 3FD04 - performed in Irwindale, LOS ANGELES COUNTY, "
        "California, where city_code and location_code agree at 0636826. The "
        "county is filled from the rest of the same contract, guarded on contract "
        "number, which is the method README 1.5 already uses for 1,207 damaged "
        "FY2004 field values. The raw county_code and county_name are left as the "
        "source wrote them"),
}

DCADS_CITY_COUNTY_OVERRIDE_PATH = (
    REFERENCE_DIR / "dcads_city_county_overrides.csv"
)
DCADS_VERIFIED_CITY_COUNTY_OVERRIDE_PATH = (
    REFERENCE_DIR / "dcads_verified_city_county_overrides.csv"
)


def normalize_city_name(value):
    """Normalize a city field for an exact override key."""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def normalize_contract_id(value):
    """Normalize a contract number for an exact override key."""
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper()).strip()


def load_dcads_city_county_overrides():
    """Load verified city-specific corrections with guarded row counts."""
    required = [
        "fiscal_year", "state_code", "city_name", "reported_county_fips",
        "corrected_county_fips", "expected_rows", "evidence",
    ]
    with open(DCADS_CITY_COUNTY_OVERRIDE_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != required:
            raise ValueError(
                f"DCADS city-county override columns are {reader.fieldnames}")
        rows = list(reader)
    result = {}
    for row in rows:
        year = row["fiscal_year"].strip()
        state = row["state_code"].strip()
        city = normalize_city_name(row["city_name"])
        reported = row["reported_county_fips"].strip()
        corrected = row["corrected_county_fips"].strip()
        expected = int(row["expected_rows"])
        if not (year.isdigit() and len(state) == 2 and state.isdigit()):
            raise ValueError(f"invalid DCADS override year/state: {row}")
        if not (len(reported) == len(corrected) == 5
                and reported.isdigit() and corrected.isdigit()):
            raise ValueError(f"invalid DCADS override county FIPS: {row}")
        if reported[:2] != state or corrected[:2] != state:
            raise ValueError(f"cross-state DCADS city-county override: {row}")
        if not city or expected <= 0 or not row["evidence"].strip():
            raise ValueError(f"incomplete DCADS city-county override: {row}")
        key = (year, state, reported, city)
        if key in result:
            raise ValueError(f"duplicate DCADS city-county override: {key}")
        result[key] = (
            corrected, state, "manual_override", expected,
            row["evidence"].strip(),
        )
    return result


DCADS_CITY_COUNTY_OVERRIDES = load_dcads_city_county_overrides()


def load_verified_dcads_city_county_overrides():
    """Load systematic city and exact-contract geography corrections."""
    required = [
        "fiscal_year", "state_code", "city_name", "reported_county_fips",
        "contract_id", "corrected_county_fips", "expected_rows", "evidence",
    ]
    with open(DCADS_VERIFIED_CITY_COUNTY_OVERRIDE_PATH, newline="",
              encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != required:
            raise ValueError(
                f"verified DCADS override columns are {reader.fieldnames}")
        rows = list(reader)
    city_result = {}
    contract_result = {}
    for row in rows:
        year = row["fiscal_year"].strip()
        state = row["state_code"].strip()
        city = normalize_city_name(row["city_name"])
        reported = row["reported_county_fips"].strip()
        contract = normalize_contract_id(row["contract_id"])
        corrected = row["corrected_county_fips"].strip()
        expected = int(row["expected_rows"])
        if not (year.isdigit() and len(state) == 2 and state.isdigit()):
            raise ValueError(f"invalid verified override year/state: {row}")
        if not (len(reported) == len(corrected) == 5
                and reported.isdigit() and corrected.isdigit()):
            raise ValueError(f"invalid verified override county FIPS: {row}")
        if reported[:2] != state or corrected[:2] != state:
            raise ValueError(f"cross-state verified override: {row}")
        if not city or expected <= 0 or not row["evidence"].strip():
            raise ValueError(f"incomplete verified override: {row}")
        spec = (
            corrected, state, "manual_override", expected,
            row["evidence"].strip(),
        )
        if contract:
            key = (year, state, reported, city, contract)
            if key in contract_result:
                raise ValueError(f"duplicate verified contract override: {key}")
            contract_result[key] = spec
        else:
            key = (year, state, reported, city)
            if key in city_result:
                raise ValueError(f"duplicate verified city override: {key}")
            city_result[key] = spec
    return city_result, contract_result


(DCADS_VERIFIED_CITY_COUNTY_OVERRIDES,
 DCADS_VERIFIED_CONTRACT_COUNTY_OVERRIDES) = (
    load_verified_dcads_city_county_overrides()
)


def overrides_for(fiscal_year, era):
    """(state_code, county_code) -> override spec, for one file.

    Empty outside the FIPS eras: the ZARK era has its own supplement and a
    three-part resolver key, so nothing here applies to it.
    """
    if era == "zark":
        return {}
    return {(sr, cr): spec for (fy, sr, cr), spec in NATIVE_OVERRIDES.items()
            if fy == fiscal_year}


def city_county_overrides_for(fiscal_year, era):
    """Return city-specific override specs keyed on raw geography fields."""
    if era == "zark":
        return {}
    result = {}
    for (fy, state, reported, city), spec in DCADS_CITY_COUNTY_OVERRIDES.items():
        if fy != fiscal_year:
            continue
        raw_county = reported if era == "native" else reported[-3:]
        result[(state, raw_county, city)] = spec
    for (fy, state, reported, city), spec in (
            DCADS_VERIFIED_CITY_COUNTY_OVERRIDES.items()):
        if fy != fiscal_year:
            continue
        raw_county = reported if era == "native" else reported[-3:]
        key = (state, raw_county, city)
        if key in result:
            raise ValueError(f"overlapping DCADS city override: {key}")
        result[key] = spec
    return result


def contract_county_overrides_for(fiscal_year, era):
    """Return exact-contract override specs keyed on raw geography fields."""
    if era == "zark":
        return {}
    result = {}
    for (fy, state, reported, city, contract), spec in (
            DCADS_VERIFIED_CONTRACT_COUNTY_OVERRIDES.items()):
        if fy != fiscal_year:
            continue
        raw_county = reported if era == "native" else reported[-3:]
        result[(state, raw_county, city, contract)] = spec
    return result

CHUNK_RECORDS = 200_000

# Pallante's replication archive is the one external benchmark held in this
# project. It ships state-by-fiscal-year DoD spending in
# data_replication/aggregates/all_dod_statename_noadj.csv, which is what the
# parsed contract dollars are compared against. The Dropbox archive is read in
# place and never extracted.
_PALLANTE_REL = Path("2_illustration_aggregation") / "Pallante" / "replication" / \
    "rcodes_reassessing-cross-sect-fiscal-multiplier.zip"
PALLANTE_ARCHIVE = DROPBOX_PROJECT_ROOT / _PALLANTE_REL

# IBM signed zoned decimal. The final byte of a numeric field carries both the
# last digit and the sign, so a plain int() would drop the record. Letters A
# through I are +1 to +9, J through R are -1 to -9, and the braces are signed
# zero. Negative values are deobligations and must keep their sign.
ZONED = {}
for _i, _c in enumerate("{ABCDEFGHI"):
    ZONED[_c] = (str(_i), 1)
for _i, _c in enumerate("}JKLMNOPQR"):
    ZONED[_c] = (str(_i), -1)

# The two signed-zero code points survive the EBCDIC to ASCII conversion
# differently across transfer eras. Positive zero appears as "{" in the standard
# table and as "+" in FY1966-67. Negative zero appears as "}", as "^", and - in
# FY1966-75 and nowhere else - as a BLANK zone byte.
#
# That blank read as POSITIVE zero until 2026-08-06, and it was wrong. It is the
# whole of the FY1966-75 disagreement with Pallante's published state-year series
# (09_pallante_test.py, README 3.4), and it inverted the sign of 17,115
# deobligations worth $12.35bn. Four lines of evidence, none of which depends on
# his series:
#
#   1  the slot is otherwise empty. In FY1966-75 no negative-zero code point
#      occurs at all unless the blank is it, while -1 through -9 each occur 700
#      to 1,900 times a year.
#   2  positive zero already has its own byte in each of those years - "+" in
#      FY1966-67, plain "0" from FY1968, 22,000 to 31,000 rows a year - so the
#      blank cannot also be positive zero.
#   3  it is continuous across the layout boundary. FY1975 blank 1,701 rows ->
#      FY1976 "^" 1,665 -> FY1977 "^" 1,700, same 272-byte layout, and the ratio
#      n(-0)/n(+0) is a smooth 0.054 to 0.083 across all nineteen years.
#   4  the magnitude is settled separately by range_code, the record's own
#      dollar-magnitude bucket. The blank is a zone byte, so the field carries a
#      tenth digit and the value is head x 10. In FY1975 all 1,701 such rows fall
#      inside their declared band at head x 10, and 0 of 1,701 do at head - every
#      one falls below its own band floor. Pallante's reader strips the trailing
#      space, so his value for these rows is a tenth of the truth.
#
# Supporting, not load-bearing: 38% of the blank rows sit on a contract carrying
# a positive row of exactly the same magnitude in the same year - an action being
# backed out - against a 22-26% null for ordinary positive rows and 44-51% for
# the confirmed "^" and "}" negatives.
#
# The change is provably local. Across all 42 fiscal years the only int-typed
# field values ending in a blank with a non-blank head are these 17,115 FY1966-75
# dollar amounts; every other year, and every other numeric field, has none.
ZONED["+"] = ("0", 1)
ZONED[" "] = ("0", -1)
ZONED["^"] = ("0", -1)

# FY2001 through FY2003 encode the negative sign a third way, as the contiguous
# run p through y, which is the ASCII digits shifted up by 0x40 so that p is
# minus zero and y is minus nine. Two independent lines of evidence fix the sign
# as negative rather than positive. First, those three years contain no standard
# }JKLMNOPQR negatives at all while 5.16% of FY2001 dollar fields end in p-y,
# and deobligations run 4% to 8% of records in every year from 1966 to 2000.
# Second, reading them as negative puts the national totals at 7.0%, 7.6% and
# 9.3% above Pallante's series, inside the 7% to 12.8% band that every year from
# 1966 to 1995 occupies, where reading them as positive gives 13.8% to 16.8% and
# breaks the pattern. Adding these codes cannot disturb the earlier eras, which
# decode every dollar field without them.
for _i, _c in enumerate("pqrstuvwxy"):
    ZONED[_c] = (str(_i), -1)


def decode_zoned(raw):
    """Return the signed integer value of a zoned-decimal field, or None.

    The final byte carries the last digit and the sign together, so it is read
    before any stripping. Stripping first would turn "00000001 ", meaning 10,
    into 1.
    """
    if not raw.strip():
        return None
    last = raw[-1]
    head = raw[:-1].strip()
    if last.isdigit():
        digits, sign = head + last, 1
    elif last in ZONED:
        digit, sign = ZONED[last]
        digits = head + digit
    else:
        return None
    if not digits.isdigit():
        return None
    return sign * int(digits)


def is_blank_code(raw):
    """True when a code field carries no information.

    Lump-sum records zero-fill the fields they do not use, so a county code
    arrives as "00000" rather than blank. Treating non-blank as present would
    attach those dollars to a county that does not exist.
    """
    return not raw.strip().strip("0")


# ---------------------------------------------------------------------------
# Time keys
#
# Every era reports by a FISCAL month, never a calendar one, and the layouts
# label the field in a way that hides this: "Fiscal reporting period, yyyymm"
# reads as a calendar year-month but month 01 is October. Verified on all 23
# month_key-bearing years: fiscal month 01 carries action dates in the previous
# October, 04 in January, 12 in September, without exception.
#
# Reading month_key as a calendar month therefore shifts everything by one
# quarter, which is why the calendar key is derived here once rather than left
# to each consumer.
#
# The fiscal year itself is not one thing across the panel. Through FY1976 the
# federal fiscal year ran July to June; the July-September 1976 transition
# quarter exists precisely because it moved to October-September for FY1977.
# The 272-era files show it directly: in FY1966-76 fiscal quarter 1 is the
# calendar months G,H,I (July, August, September) and the year-end surge lands
# in F (June), while from FY1977 quarter 1 is J,K,L (October, December) and the
# surge moves to I (September).
FISCAL_YEAR_STARTS_JULY = tuple(str(y) for y in range(1966, 1977))

# 272-era report_month is a letter, A for January through L for December. It is
# a CALENDAR month, unlike month_key.
LETTER_MONTHS = "ABCDEFGHIJKL"


def fiscal_start_month(fiscal_year):
    """First calendar month of a fiscal year: 7 through FY1976, else 10.

    The transition quarter is July-September 1976 and so starts in July, like
    the July-June years it closes out rather than the October-September ones it
    opens.
    """
    label = str(fiscal_year)
    if label.endswith("tq"):
        return 7
    return 7 if label in FISCAL_YEAR_STARTS_JULY else 10


def fiscal_to_calendar(fiscal_year, fiscal_month):
    """(fiscal year label, fiscal month 1-12) -> (calendar year, calendar month).

    Returns None when the fiscal month is missing or out of range, so a
    malformed reporting period yields a null time key rather than a wrong one.
    """
    if fiscal_month is None or not 1 <= fiscal_month <= 12:
        return None
    label = str(fiscal_year)
    year = int(label[:4])
    start = fiscal_start_month(label)
    month = (start - 1 + fiscal_month - 1) % 12 + 1
    # The transition quarter is July to September 1976 and is named for the
    # calendar year it falls in, not one it ends after, so it needs no shift. A
    # full fiscal year is named for the calendar year it ends in, so the months
    # from its start month to December belong to the preceding calendar year.
    if label.endswith("tq"):
        return year, month
    return (year - 1 if month >= start else year), month


def calendar_to_fiscal_month(fiscal_year, calendar_month):
    """Inverse of the month part of fiscal_to_calendar, or None."""
    if calendar_month is None or not 1 <= calendar_month <= 12:
        return None
    return (calendar_month - fiscal_start_month(fiscal_year)) % 12 + 1


def fiscal_month_of_record(spec, cols, i):
    """Fiscal month 1-12 for record i, or None when it cannot be read.

    Two eras, two fields. From FY1984 the reporting period is month_key, whose
    last two characters are the fiscal month whether the field is four wide
    (YYMM) or six (CCYYMM). Before that there is no month_key; the 272 layout
    carries report_month, a letter naming the CALENDAR month, which has to be
    converted.

    Keyed to the field that exists rather than declared per layout, because the
    twelve fixed-width layouts would otherwise all need the same two lines and
    the choice is fully determined by the era.
    """
    names = {f["name"] for f in spec["fields"]}
    if "month_key" in names:
        raw = cols["month_key"][i]
        tail = raw[-2:] if raw else ""
        if not tail.isdigit():
            return None
        month = int(tail)
        return month if 1 <= month <= 12 else None
    if "report_month" in names:
        letter = (cols["report_month"][i] or "").strip().upper()
        if len(letter) != 1 or letter not in LETTER_MONTHS:
            return None
        return calendar_to_fiscal_month(spec.get("_fiscal_year", ""),
                                        LETTER_MONTHS.index(letter) + 1)
    return None


def derive_time_columns(spec, cols, fiscal_year, n):
    """Return {'fiscal_month': [...], 'calendar_yearmonth': [...]} for one chunk.

    fiscal_month is 1-12 with 1 meaning the first month of that fiscal year.
    calendar_yearmonth is "YYYYMM" so it sorts, and is empty where the reporting
    period is unreadable.

    month_key is the REPORTING period and lags the obligation: only 39% to 70%
    of a month's records share its modal action month. Anything that needs when
    the money was actually obligated should key on action_date, not on this.
    """
    spec = dict(spec, _fiscal_year=fiscal_year)
    months, stamps = [], []
    for i in range(n):
        fm = fiscal_month_of_record(spec, cols, i)
        months.append(fm)
        cal = fiscal_to_calendar(fiscal_year, fm)
        stamps.append(f"{cal[0]:04d}{cal[1]:02d}" if cal else "")
    return {"fiscal_month": months, "calendar_yearmonth": stamps}


# ---------------------------------------------------------------------------
# Delimited-era addressing
#
# The FY2004-06 extracts are exclamation-delimited and fixed-width at the same
# time: every record is 601 characters (603 in FY2005) and the delimiters sit at
# the same byte positions in every well-formed record. Splitting on the
# delimiter therefore throws away information, and a record that loses one
# delimiter loses a field silently. Addressing by position instead means a
# damaged record still yields every field before the damage.


def spans_from_delimiters(positions):
    """Delimiter positions -> field spans, one per field, in order."""
    spans, prev = [], 0
    for p in positions:
        spans.append((prev, p))
        prev = p + 1
    return spans


def delimited_offsets(spec, member):
    """Canonical delimiter positions for one member file of a delimited layout.

    FY2004 and FY2006 share a table; FY2005 has its own because n_actions is
    six wide there rather than four. Declared in the layout rather than inferred
    per run, so a change in the data shows up as a mismatch instead of being
    silently absorbed.
    """
    tables = spec["delimiter_offsets"]
    for key, positions in tables.items():
        if member in spec["offset_groups"][key]:
            return tuple(positions)
    raise ValueError(f"{spec['layout_id']}: no delimiter offsets declared for {member}")


def delimiter_signature(positions, canonical):
    """Identify a record's delimiter layout relative to the canonical one.

    Takes the record's delimiter positions, which the caller has already had to
    find, and returns None when they are canonical, else a (n_delims, missing,
    extra) key matching the layout's anomalies table. The four FY2004 anomalies
    each have one signature shared by every record in the group, so the key
    identifies the damage exactly rather than guessing from a delimiter count.
    """
    found = frozenset(positions)
    canon = frozenset(canonical)
    if found == canon:
        return None
    return (len(found), tuple(sorted(canon - found)), tuple(sorted(found - canon)))


def load_anomalies(spec):
    """Layout anomalies -> {signature: (tag, own->canonical map, nulled fields)}.

    The map is stored as breakpoints: [[upper_own_index, offset], ...] meaning
    own index k maps to canonical k + offset for k up to and including that
    bound. A fused entry names the one own index that holds two canonical fields
    end to end.
    """
    out = {}
    for a in spec.get("anomalies", []):
        sig = (a["n_delimiters"], tuple(a["missing"]), tuple(a["extra"]))
        out[sig] = a
    return out


def map_own_index(anomaly, k):
    """Canonical index for own index k under an anomaly's breakpoint map."""
    for bound, offset in anomaly["index_map"]:
        if k <= bound:
            return k + offset
    return None


def load_layouts(include_delimited=False):
    """Return {layout_id: spec} and a filename -> layout_id index.

    Selection is keyed to the filename and never inferred from record length,
    because FY1989 and FY1990-95 share a length of 464 while position 408 means
    different things in each.

    The FY2004-06 layouts describe delimited files, address their fields by
    position in the line rather than by byte offset, and are read by
    05_parse_dcads_delimited.py. They are excluded by default so the fixed-width
    parser never sees a spec it cannot use.
    """
    layouts, index = {}, {}
    for path in sorted(LAYOUT_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec.get("format") == "delimited" and not include_delimited:
            continue
        layouts[spec["layout_id"]] = spec
        for name in spec["applies_to"]:
            index[name] = spec["layout_id"]
    return layouts, index


def fiscal_year_of(filename):
    """Return the fiscal-year label a data file belongs to, from its name.

    Never taken from manifest.csv: fiscal_year() in 01_build_dcads_manifest.py
    reads the start year of a title's date range, which labels
    RG330.MPCF.FY66 as 1965. The transition quarter keeps its own label rather
    than folding into 1977, which is where its internal year field would put it.
    """
    m = re.search(r"\.(FY|TQ)(\d{2,4})", filename)
    if not m:
        return None
    kind, digits = m.group(1), m.group(2)
    if len(digits) == 2:
        year = 1900 + int(digits) if int(digits) >= 60 else 2000 + int(digits)
    else:
        year = int(digits)
    return f"{year}tq" if kind == "TQ" else str(year)


def iter_chunks(path, spec, chunk_records=CHUNK_RECORDS):
    """Yield {column: list} dicts, one per chunk of fixed-width records.

    Streams rather than accumulating, because the later fiscal years reach
    303 MB uncompressed and holding every chunk before concatenating them once
    exhausted memory on the largest files.
    """
    rl = spec["record_length"]
    enc = spec.get("encoding", "latin-1")
    fields = spec["fields"]
    size = path.stat().st_size
    if size % rl:
        raise ValueError(f"{path.name}: size {size:,} is not a multiple of {rl}")

    lumpsum_field = spec.get("lumpsum_field")
    lumpsum_span = None
    if lumpsum_field:
        for f in fields:
            if f["name"] == lumpsum_field:
                lumpsum_span = (f["start"], f["end"])
                break
        if lumpsum_span is None:
            raise ValueError(
                f"{spec['layout_id']}: lumpsum_field {lumpsum_field!r} is not a declared field"
            )

    with open(path, "rb") as fh:
        while True:
            block = fh.read(rl * chunk_records)
            if not block:
                break
            text = block.decode(enc)
            records = [text[i:i + rl] for i in range(0, len(text), rl)]
            cols = {}
            for f in fields:
                a, b = f["start"], f["end"]
                if f["type"] == "int":
                    cols[f["name"]] = [decode_zoned(r[a:b]) for r in records]
                else:
                    cols[f["name"]] = [r[a:b].strip() for r in records]
            if lumpsum_span:
                a, b = lumpsum_span
                cols["is_lumpsum"] = [1 if is_blank_code(r[a:b]) else 0 for r in records]
            else:
                cols["is_lumpsum"] = [0] * len(records)
            yield cols


def data_files():
    """Every candidate data file under SOURCE_DIR, documentation excluded.

    Suffix-based, which leaves the nine extensionless FY2003 files in the list
    even though the technical summary declares them to be PDFs of code lists and
    forms. They are harmless: no layout claims them, so they are reported as
    skipped rather than parsed.
    """
    return [p for p in sorted(SOURCE_DIR.rglob("*"))
            if p.is_file() and p.suffix.lower() not in
            (".pdf", ".doc", ".lay", ".txt", ".csv", ".zip", ".json", ".sha256")]


def read_pallante_csv(suffix):
    """Read one CSV out of Pallante's archive without extracting it.

    Returns a list of dict rows, or None when the Dropbox archive is unavailable.
    """
    import csv
    import io
    if not PALLANTE_ARCHIVE.exists():
        return None
    with zipfile.ZipFile(PALLANTE_ARCHIVE) as zf:
        hits = [n for n in zf.namelist()
                if n.endswith(suffix) and not n.startswith("__MACOSX")]
        if len(hits) != 1:
            return None
        with zf.open(hits[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8-sig", newline="")
            return list(csv.DictReader(text))


# ---------------------------------------------------------------------------
# Geography: ZARK to FIPS, and the classification every era shares
#
# Place of performance is coded three incompatible ways across the panel. FY1984
# through FY2003 already carry a 5-digit SSCCC FIPS county. FY2004-06 hold the
# state and the county in separate fields. FY1966-83 carry ZARK, the Pentagon's
# own pre-FIPS scheme, in which the state is an alphabetical sequence (Virginia
# is 47, not 51) and the county is a within-state alphabetical sequence that
# repeats across states.
#
# ZALF resolves the ZARK era exactly. Its layout is documented in 124.29DP.pdf
# p.4 and confirmed by a column profile of all 29,067 records: 80 characters plus
# CRLF, three record types discriminated by the last character. Type 1 is a
# state or country, type 2 a county, type 3 a city. The FIPS columns are labelled
# "GSA" in the documentation but for the 50 states and DC they are FIPS 5-2 and
# FIPS 6-4 exactly, including the gaps a formula would miss - Hawaii 12 -> 15,
# Rhode Island 40 -> 44, Washington 48 -> 53, and 245 of the 3,278 county records
# that break the naive odd-number pattern and are still right.
#
# Two things ZALF cannot do on its own, both handled by zark_fips_supplement.csv:
# it is a single undated cross-section while a few ZARK codes were redefined
# mid-panel, and its Puerto Rico block echoes the ZARK code instead of giving a
# municipio FIPS.

# The ten Alaska census divisions and three other entities that stopped existing
# before FY1984 and so cannot appear in a FIPS-era file. Reported by check 6 of
# 08_verify_county_fips.py, where this used to live, and consumed as a seed by
# 10_build_county_universe.py, which is why it moved here: two scripts read it and
# 10 makes it fail rather than merely print.
#
# The successors are this project's own research. The Census "Substantial Changes
# to Counties" list, transcribed in county_changes_census.csv, records that the
# ten Alaska divisions were deleted in the 1970s reorganisation but names no
# successor for any of them.
SUPERSEDED = {
    "02040": "Barrow/North Slope division -> North Slope Borough 02185",
    "02080": "Cordova-McCarthy division -> Valdez-Cordova 02261",
    "02120": "Kenai-Cook Inlet division -> Kenai Peninsula 02122",
    "02160": "Kuskokwim division -> Bethel 02050",
    "02190": "Outer Ketchikan -> Prince of Wales-Outer Ketchikan 02201",
    "02200": "Prince of Wales division -> Prince of Wales-Outer Ketchikan 02201",
    "02210": "Seward division -> Kenai Peninsula 02122",
    "02230": "Skagway-Yakutat -> Skagway-Yakutat-Angoon 02231",
    "02250": "Upper Yukon division -> Yukon-Koyukuk 02290",
    "02260": "Valdez-Chitina-Whittier division -> Valdez-Cordova 02261",
    "13510": "Columbus city GA, consolidated into Muscogee County 13215 in 1971",
    "51123": "Nansemond County VA, abolished 1974 -> Suffolk city 51800",
    "51695": "Nansemond city VA, 1972-74 only -> Suffolk city 51800",
}

# The successor FIPS each SUPERSEDED code folds into, parsed out of the prose
# above so the two can never drift apart. 02210 Seward and 02160 Kuskokwim are the
# two the Census list does not cover even indirectly.
SUPERSEDED_TARGET = {
    "02040": "02185", "02080": "02261", "02120": "02122", "02160": "02050",
    "02190": "02201", "02200": "02201", "02210": "02122", "02230": "02231",
    "02250": "02290", "02260": "02261", "13510": "13215", "51123": "51800",
    "51695": "51800",
}

ZALF_RECORD = 82          # 80 characters plus CRLF
ZALF_STATE, ZALF_COUNTY, ZALF_CITY = "1", "2", "3"
ZALF_COUNTS = {ZALF_STATE: 241, ZALF_COUNTY: 3278, ZALF_CITY: 25548}

# The 50 states and DC. The territories are deliberately not here: they are real
# places with real FIPS but they are not the county panel, and most employment
# sources stop at the 50 states.
STATE_FIPS = frozenset("""
01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56
""".split())

# Territories that carry real counties or municipios.
TERRITORY_FIPS = frozenset({"60", "66", "69", "72", "78"})

# FIPS-coded but not US counties: the freely associated states, sovereign since
# 1986 and 1994, and the uninhabited minor outlying islands. They must never
# enter a county panel, which is why they are classed apart from the territories
# rather than lumped in with them.
ASSOCIATED_FIPS = frozenset({"64", "68", "70", "74"})

# The source's own catch-all, "somewhere in the United States, county unknown".
# Not a state and carries no county.
UNDISTRIBUTED_STATE = "98"

# A county part of 999 is the same catch-all one level down: the state is known
# and the county is not.
UNDISTRIBUTED_COUNTY = "999"

# FY1984-88 files territories as foreign, using FIPS 10-4 alpha codes; FY1989
# onward files them as domestic with numeric FIPS. A thin tail of the alpha codes
# survives alongside the numeric ones into FY2003-06. The FY1989-2006 numeric
# convention is the target for all 42 years, so the alpha codes are recoded to
# it. Only RQ ever carries a county, and there the three digits are already the
# FIPS municipio, so RQ005 becomes 72005.
#
# WS, Western Samoa, is deliberately absent: it is a sovereign country, not the
# US territory of American Samoa, which is AQ.
ALPHA_TERRITORY = {
    "AQ": "60",   # American Samoa
    "FM": "64",   # Micronesia
    "GQ": "66",   # Guam
    "RM": "68",   # Marshall Islands
    "CQ": "69",   # Northern Mariana Islands
    "PW": "70",   # Palau
    "RQ": "72",   # Puerto Rico
    "VQ": "78",   # US Virgin Islands
    "VI": "78",   # US Virgin Islands, as FY1984-86 write it
}

# geo_class, the one column a county panel has to filter on.
#
#   county        the 50 states and DC, county known. THIS IS THE COUNTY PANEL.
#   territory     Amer Samoa, Guam, N Mariana, Puerto Rico, Virgin Is. Real
#                 places, outside the 50 states and outside most employment
#                 sources. county_fips5 may be empty where the source names no
#                 county, which is every pre-FY1989 territory row except Puerto
#                 Rico; state_fips2 still identifies the territory.
#   associated    Micronesia, Marshall Is, Palau, US Minor Outlying. FIPS-coded,
#                 not US counties. Never in a county panel.
#   undistributed state 98, the source's own "somewhere in the United States".
#   foreign       a genuinely foreign country or water area.
#   lumpsum       a DD1057 summary row, which carries no geography at all.
#   unplaceable   domestic, in the 50 states or DC, but no county resolves: a
#                 blank county field, a county part of 999, or a malformed code.
GEO_CLASSES = ("county", "territory", "associated", "undistributed",
               "foreign", "lumpsum", "unplaceable")


def zark_year(fiscal_year):
    """Fiscal-year label -> int, so "1976tq" sorts and compares as 1976."""
    return int(str(fiscal_year)[:4])


def read_zalf(path=None):
    """Parse ZALF into ({zark_state: fips_state}, {(zark_state, zark_county): (name, fips5)}).

    Two guards, both taken from the documented layout rather than invented.
    Position 3 is declared blank in all three record types, and the two records
    that violate it are junk: a second California state record carrying "YM"
    where the real one carries 06, which silently overwrote the real one when it
    was read without the guard, and a city record with the same corruption. The
    ZARK county must be numeric, which drops one further malformed record
    reading "37 OKLAHOMA 34  FORSYTH 067".

    City records are parsed only to be counted. They carry a FIPS place code and
    are the route to the four domestic rows that name a city but no county, which
    is not worth taking yet.
    """
    path = path or ZALF
    raw = path.read_bytes()
    if len(raw) % ZALF_RECORD:
        raise ValueError(f"{path.name}: {len(raw):,} bytes is not a multiple of "
                         f"{ZALF_RECORD}")
    records = [raw[i:i + ZALF_RECORD].decode("latin-1")
               for i in range(0, len(raw), ZALF_RECORD)]
    bad_eol = sum(1 for r in records if r[80:82] != "\r\n")
    if bad_eol:
        raise ValueError(f"{path.name}: {bad_eol:,} records do not end CRLF")
    records = [r[:80] for r in records]

    counts = collections.Counter(r[79] for r in records)
    if dict(counts) != ZALF_COUNTS:
        raise ValueError(f"{path.name}: record types {dict(counts)} do not match "
                         f"the expected {ZALF_COUNTS}")

    states, counties, dropped = {}, {}, []
    for r in records:
        kind = r[79]
        if r[2] != " ":
            dropped.append(r)
            continue
        if kind == ZALF_STATE:
            states[r[0:2]] = r[24:26].strip()
        elif kind == ZALF_COUNTY:
            zark_county = r[24:27]
            if not zark_county.isdigit():
                dropped.append(r)
                continue
            fips_state, fips_county = states.get(r[0:2], ""), r[49:52]
            if fips_state.isdigit() and fips_county.isdigit():
                counties[(r[0:2], zark_county)] = (r[28:47].strip(),
                                                   fips_state + fips_county)
    return states, counties, dropped


def load_zark_crosswalk(zalf_path=None, supplement_path=None):
    """Return a lookup(zark_state, zark_county, fiscal_year, county_name) -> fips5.

    ZALF supplies the base. zark_fips_supplement.csv overrides it, and is the
    only place a judgement call about pre-1984 geography lives. A supplement row
    is scoped by fiscal year, because ZALF is undated while a few ZARK codes were
    redefined mid-panel, and optionally by the recorded county name, for the one
    code that carries two different counties in the same years.

    Returns (lookup, stats) where stats reports how the table was built, and the
    lookup returns (fips5, source) with source one of "zalf" or "supplement", or
    (None, None) when nothing resolves.
    """
    import csv

    states, counties, dropped = read_zalf(zalf_path)
    path = supplement_path or ZARK_SUPPLEMENT
    supplement = collections.defaultdict(list)
    seen = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["zark_state"], row["zark_county"])
            name = row["county_name"].strip().upper()
            span = (int(row["fy_from"]), int(row["fy_to"]))
            ident = (key, name, span)
            if ident in seen:
                raise ValueError(f"{path.name}: duplicate row for {key} "
                                 f"{name or '(any name)'} {span}")
            seen.add(ident)
            supplement[key].append((span[0], span[1], name, row["fips5"].strip()))

    used = set()

    def lookup(zark_state, zark_county, fiscal_year, county_name=""):
        year = zark_year(fiscal_year)
        rows = supplement.get((zark_state, zark_county))
        if rows:
            name = (county_name or "").strip().upper()
            # A name-qualified row wins over an unqualified one covering the same
            # year, so the general case can be stated once and the exception
            # beside it.
            for want_name in (True, False):
                for i, (a, b, rname, fips) in enumerate(rows):
                    if bool(rname) != want_name or not a <= year <= b:
                        continue
                    if rname and rname != name:
                        continue
                    used.add(((zark_state, zark_county), rname, (a, b)))
                    return (fips or None), "supplement"
        hit = counties.get((zark_state, zark_county))
        return (hit[1], "zalf") if hit else (None, None)

    stats = {"zalf_states": len(states), "zalf_counties": len(counties),
             "zalf_dropped": dropped, "supplement_rows": len(seen),
             "supplement_seen": seen, "supplement_used": used,
             "state_fips": states}
    return lookup, stats


def classify_geo(state_fips2, county_fips5, is_lumpsum, state_code_raw):
    """Return the geo_class for one record. See GEO_CLASSES for what each means."""
    if is_lumpsum:
        return "lumpsum"
    if state_fips2 in STATE_FIPS:
        return "county" if county_fips5 else "unplaceable"
    if state_fips2 in TERRITORY_FIPS:
        return "territory"
    if state_fips2 in ASSOCIATED_FIPS:
        return "associated"
    if (state_code_raw or "").strip() == UNDISTRIBUTED_STATE:
        return "undistributed"
    if (state_code_raw or "").strip():
        return "foreign"
    return "unplaceable"


def extract_member(zip_path, member_name=None):
    """Extract one member of a zip to scratch and return its path.

    The later fiscal years ship only as zips, so the data has to be unpacked
    before it can be parsed. Extraction is skipped when the scratch copy is
    already present at full size, which makes a rerun cheap.
    """
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if member_name:
            infos = [i for i in infos if i.filename == member_name]
        if len(infos) != 1:
            names = [i.filename for i in infos]
            raise ValueError(f"{zip_path.name}: expected one member, found {names}")
        info = infos[0]
        dest = SCRATCH_DIR / Path(info.filename).name
        if dest.exists() and dest.stat().st_size == info.file_size:
            return dest, info.file_size
        with zf.open(info) as src, open(dest, "wb") as out:
            while True:
                buf = src.read(1 << 20)
                if not buf:
                    break
                out.write(buf)
    if dest.stat().st_size != info.file_size:
        raise ValueError(
            f"{zip_path.name}: extracted {dest.stat().st_size:,} bytes, "
            f"zip directory declares {info.file_size:,}"
        )
    return dest, info.file_size


# ---------------------------------------------------------------------------
# Part 1.4: the price index, the CCG map, and the contract collapse
#
# Everything below is shared by 12_build_county_income.py, 13_build_county_panel.py
# and 14_build_instrument.py. It lives here because two copies of a contract
# collapse is two collapses that drift apart, and the whole point of the shock is
# that the numerator and the denominator are deflated the same way.
# ---------------------------------------------------------------------------

# CPI-U, annual average, 1982-84 = 100. One index for contracts and for income,
# which is what makes their ratio a real quantity rather than two price levels
# divided by each other. 1969 onward, because BEA county personal income starts
# there; the contract panel only ever reaches back to FY1966 without a
# denominator.
#
# 2007-2018 added 2026-08-11 for the post-FY2006 extension. Source for the whole
# table is BLS CPI-U, U.S. city average, all items, annual average, series
# CUUR0000SA0. It stops at 2018 because the employment panel does, and a
# deflator no run can exercise is untested data.
CPI = {
    1966: 32.4, 1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 49.3, 1975: 53.8, 1976: 56.9, 1977: 60.6,
    1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9, 1982: 96.5, 1983: 99.6,
    1984: 103.9, 1985: 107.6, 1986: 109.6, 1987: 113.6, 1988: 118.3,
    1989: 124.0, 1990: 130.7, 1991: 136.2, 1992: 140.3, 1993: 144.5,
    1994: 148.2, 1995: 152.4, 1996: 156.9, 1997: 160.5, 1998: 163.0,
    1999: 166.6, 2000: 172.2, 2001: 177.1, 2002: 179.9, 2003: 184.0,
    2004: 188.9, 2005: 195.3, 2006: 201.6,
    2007: 207.342, 2008: 215.303, 2009: 214.537, 2010: 218.056,
    2011: 224.939, 2012: 229.594, 2013: 232.957, 2014: 236.736,
    2015: 237.017, 2016: 240.007, 2017: 245.120, 2018: 251.107,
}
CPI_BASE = 2006

# A transposed digit anywhere in the added block rescales every post-2006 dollar,
# and the symptom - a level jump at the FY2006/07 seam - is indistinguishable from
# the construction differences the splice audit is already documenting, so it
# would be misdiagnosed. Two guards: a known endpoint ratio, and monotonicity
# everywhere except 2009, which really did fall (the 2008-09 CPI reversal).
assert round(CPI[2018] / CPI[2006], 4) == 1.2456, "CPI 2018/2006 anchor moved"
assert all(CPI[y] > CPI[y - 1] for y in range(1967, 2019) if y != 2009), \
    "CPI is non-monotone somewhere other than the 2009 reversal"


def deflate(year):
    """Multiplier that puts a nominal `year` dollar into CPI_BASE dollars."""
    return CPI[CPI_BASE] / CPI[year]


def load_ccg_map():
    """(county_fips5 -> ccg_id, the unplaceable FIPS set, the sorted CCG universe).

    36173 is not a valid New York FIPS and its single contract row names no
    county, so 10_build_county_universe.py routes it to 'unplaceable' and keeps
    it out of every rung. It is returned separately rather than silently dropped,
    because a county code that fails to map for any OTHER reason is a hole in the
    crosswalk and must fail the run.
    """
    import numpy as np
    import pandas as pd
    xwalk = pd.read_csv(CCG_CROSSWALK, dtype={"county_fips5": str, "ccg_id": str})
    placeable = xwalk[xwalk["ccg_id"] != "unplaceable"]
    fips_to_ccg = dict(zip(placeable["county_fips5"], placeable["ccg_id"].astype(int)))
    unplaceable = set(xwalk.loc[xwalk["ccg_id"] == "unplaceable", "county_fips5"])
    universe = np.array(sorted(set(fips_to_ccg.values())), dtype=np.int64)
    return fips_to_ccg, unplaceable, universe


def positive_employment_profile():
    """One row per geographic CCG with its fixed-sample eligibility measures."""
    import numpy as np
    import pandas as pd

    path = EMP_OUT_DIR / "efsy_ccg_year.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run 11_build_employment_cells.py before "
            "constructing the analysis universe")

    _, _, geographic_universe = load_ccg_map()
    years = np.arange(POSITIVE_EMP_FIRST_YEAR, POSITIVE_EMP_LAST_YEAR + 1)
    emp = pd.read_parquet(path, columns=["ccg", "year", "emp"])
    emp = emp[emp["year"].between(years[0], years[-1])]
    if emp.duplicated(["ccg", "year"]).any():
        raise AssertionError("employment panel has duplicate CCG-year rows")

    expected = pd.MultiIndex.from_product(
        [geographic_universe, years], names=["ccg", "year"])
    observed = pd.MultiIndex.from_frame(emp[["ccg", "year"]])
    if not observed.equals(expected):
        missing = expected.difference(observed)
        surplus = observed.difference(expected)
        raise AssertionError(
            "employment panel is not rectangular on the geographic universe: "
            f"{len(missing)} missing and {len(surplus)} surplus CCG-years")

    emp = emp.assign(positive=np.isfinite(emp["emp"]) & (emp["emp"] > 0))
    profile = emp.groupby("ccg", sort=True).agg(
        required_years=("year", "size"),
        positive_years=("positive", "sum"),
        min_employment=("emp", "min"),
    ).reset_index()
    profile["eligible"] = profile["positive_years"] == len(years)
    nonpositive = (emp.loc[~emp["positive"], ["ccg", "year"]]
                   .groupby("ccg", sort=True)["year"]
                   .agg(lambda s: " ".join(map(str, s))))
    profile["nonpositive_years"] = profile["ccg"].map(nonpositive).fillna("")
    return profile


def load_analysis_ccg_map():
    """Full county map and the fixed strictly-positive-employment CCG universe."""
    import numpy as np

    fips_to_ccg, unplaceable, geographic_universe = load_ccg_map()
    profile = positive_employment_profile()
    if (profile["required_years"]
            != POSITIVE_EMP_LAST_YEAR - POSITIVE_EMP_FIRST_YEAR + 1).any():
        raise AssertionError("employment eligibility profile has an incomplete year span")
    universe = profile.loc[profile["eligible"], "ccg"].to_numpy(dtype=np.int64)
    if not set(universe).issubset(set(geographic_universe)):
        raise AssertionError("employment analysis universe exceeds geographic universe")
    return fips_to_ccg, unplaceable, np.sort(universe)


def build_contract_panel(ccg_universe, years, floor=0, fold_tq76=True, verbose=True):
    """Collapse contract actions to CCG x year, deflated to CPI_BASE dollars.

    Returns a DataFrame[ccg, year, dollars_real, dollars_nominal, n_actions],
    rectangular over ccg_universe x years with absent county-years filled at
    zero - a county with no contract action in a year received $0, which is a
    number, not missing data.

    `floor` is a nominal dollar threshold on the MAGNITUDE of an action. A
    deobligation of -$50,000 is a large action, not a small one, and dropping it
    while keeping the obligation it reverses would inflate the county.

    `fold_tq76` puts the July-September 1976 transition quarter into FY1976,
    which is the convention Pallante's published series uses and therefore the
    one FY1976 was verified against in 09_pallante_test.py.

    TWO SOURCES, ONE COLLAPSE                                       SEAM_YEAR

    Years before SEAM_YEAR come from the parsed DCADS parquets in OUT_DIR. From
    SEAM_YEAR the actions come from the annual USAspending transaction parquets
    in USASPENDING_OUT_DIR. The splice lives HERE, in the one function 12, 13 and
    14 all call. The USAspending branch retains action counts and applies the same
    magnitude floor to each signed action before either source is collapsed.
    """
    import numpy as np
    import pandas as pd

    fips_to_ccg, unplaceable_fips, _ = load_ccg_map()

    years = list(years)
    pre_years = [y for y in years if y < SEAM_YEAR]
    post_years = [y for y in years if y >= SEAM_YEAR]

    chunks = []
    unmapped = 0
    n_unplaceable = 0
    kept_actions = 0

    sources = [(y, f"fy{y}.parquet") for y in pre_years]
    if fold_tq76 and 1976 in pre_years and (OUT_DIR / "fy1976tq.parquet").exists():
        sources.append((1976, "fy1976tq.parquet"))

    for year, fname in sources:
        path = OUT_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run 04/05 and 07 first")
        df = pd.read_parquet(
            path, columns=["dollars", "dollars_unit", "geo_class", "county_fips5"])

        # geo_class == 'county' is the filter step_1/README.md section 1.5
        # insists on: it is the only one that excludes the FY1984-85 lump-sum
        # rows, which write a zero-filled county code rather than a blank and so
        # survive a non-blankness test.
        df = df[df["geo_class"] == "county"]
        if df.empty:
            continue

        dollars = df["dollars"].to_numpy(dtype=np.float64)
        thousands = (df["dollars_unit"] == "thousands").to_numpy()
        dollars = np.where(thousands, dollars * 1000.0, dollars)  # unit changes at FY1989

        ccg = df["county_fips5"].map(fips_to_ccg)
        bad = ccg.isna()
        drop_ok = df["county_fips5"].isin(unplaceable_fips)
        n_unplaceable += int((bad & drop_ok).sum())
        unmapped += int((bad & ~drop_ok).sum())
        ccg = ccg.fillna(-1).to_numpy(dtype=np.int64)

        keep = (~bad.to_numpy()) & (np.abs(dollars) >= floor)
        if not keep.any():
            continue
        kept_actions += int(keep.sum())
        g = pd.DataFrame({"ccg": ccg[keep], "nominal": dollars[keep]})
        agg = g.groupby("ccg", sort=False)["nominal"].agg(["sum", "size"])
        chunks.append(pd.DataFrame({
            "ccg": agg.index.to_numpy(),
            "year": year,
            "dollars_nominal": agg["sum"].to_numpy(),
            "n_actions": agg["size"].to_numpy(),
        }))

    if unmapped:
        raise AssertionError(
            f"{unmapped} contract rows have a county FIPS outside the CCG "
            f"crosswalk; 10_build_county_universe.py guarantees zero")
    if verbose and n_unplaceable:
        print(f"    {n_unplaceable} action(s) dropped as unplaceable "
              f"(FIPS {sorted(unplaceable_fips)}), as step 1 requires")

    if chunks:
        panel = pd.concat(chunks, ignore_index=True)
        # TQ76 lands on the same (ccg, 1976) key as FY1976, so the fold is a regroup
        # rather than a concat.
        panel = panel.groupby(["ccg", "year"], as_index=False).sum()
        panel["n_actions"] = panel["n_actions"].astype("Int64")
    else:
        panel = pd.DataFrame({"ccg": pd.Series(dtype=np.int64),
                              "year": pd.Series(dtype=np.int64),
                              "dollars_nominal": pd.Series(dtype=np.float64),
                              "n_actions": pd.Series(dtype="Int64")})

    if post_years:
        post, post_actions = _post_seam_block(post_years, floor)
        kept_actions += post_actions
        panel = pd.concat([panel, post], ignore_index=True)

    spine = pd.MultiIndex.from_product(
        [np.sort(np.asarray(ccg_universe)), years], names=["ccg", "year"])
    panel = panel.set_index(["ccg", "year"]).reindex(spine).reset_index()
    # Absent county-years are $0 - a county with no contract action received zero,
    # which is a number. Both raw sources carry actions, so absence means a zero
    # action count on either side of the seam.
    panel["dollars_nominal"] = panel["dollars_nominal"].fillna(0.0)
    panel["n_actions"] = panel["n_actions"].fillna(0)
    panel["n_actions"] = panel["n_actions"].astype("Int64")
    panel["dollars_real"] = panel["dollars_nominal"] * panel["year"].map(deflate)

    if verbose:
        nz = int((panel["dollars_nominal"] != 0).sum())
        label = "none" if floor == 0 else f"${floor:,}"
        span = (f"FY{min(years)}-{max(years)}"
                + (f", spliced at FY{SEAM_YEAR}" if post_years and pre_years else ""))
        print(f"    floor {label:>9}   {kept_actions:>11,} actions   "
              f"{nz:>8,} of {len(panel):,} CCG-years nonzero   "
              f"${panel['dollars_real'].sum() / 1e9:>10,.1f}bn ({CPI_BASE}$)   {span}")
    return panel[["ccg", "year", "dollars_nominal", "dollars_real", "n_actions"]]


def collapse_usaspending_contracts(years, floor=0):
    """Collapse parsed USAspending award actions to sparse CCG-year totals."""
    import numpy as np
    import pandas as pd

    chunks = []
    years = list(years)
    for year in years:
        path = USASPENDING_OUT_DIR / f"fy{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run 31_parse_usaspending.py before FY{year}")
        frame = pd.read_parquet(
            path,
            columns=[
                "federal_action_obligation",
                "award_or_idv_flag",
                "geo_class",
                "ccg",
            ],
        )
        dollars = frame["federal_action_obligation"].to_numpy(dtype=np.float64)
        keep = (
            frame["award_or_idv_flag"].eq("AWARD").to_numpy()
            & frame["geo_class"].eq("county").to_numpy()
            & frame["ccg"].notna().to_numpy()
            & (np.abs(dollars) >= floor)
        )
        if not keep.any():
            raise AssertionError(f"FY{year}: no mapped USAspending award actions")
        grouped = pd.DataFrame(
            {
                "ccg": frame.loc[keep, "ccg"].to_numpy(dtype=np.int64),
                "dollars_nominal": dollars[keep],
            }
        ).groupby("ccg", sort=False)["dollars_nominal"].agg(["sum", "size"])
        chunks.append(
            pd.DataFrame(
                {
                    "ccg": grouped.index.to_numpy(dtype=np.int64),
                    "year": year,
                    "dollars_nominal": grouped["sum"].to_numpy(dtype=np.float64),
                    "n_actions": grouped["size"].to_numpy(dtype=np.int64),
                }
            )
        )
    panel = pd.concat(chunks, ignore_index=True)
    panel["n_actions"] = panel["n_actions"].astype("Int64")
    return panel


def _post_seam_block(post_years, floor):
    """FY2007 onward, collapsed from parsed USAspending award actions."""
    post = collapse_usaspending_contracts(post_years, floor=floor)
    return post, int(post["n_actions"].sum())
