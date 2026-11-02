#!/usr/bin/env python3
"""The Pallante test: our state-year contract dollars against his published series.

Check 2 of PLAN_macro_fiscal_multiplier.md section 1.5, and the only one of the
five with an external benchmark. It could not run until 07_add_county_fips.py
wrote state_fips2 into all 42 fiscal years, because FY1966-83 carry ZARK codes
and there was no state key to join on.

This is the one check in the project whose failure would mean the numbers are
wrong rather than internally inconsistent. Every other geography check compares
the parse against itself; Pallante's series is built from his own extract of the
same archival records, by his own code, keyed on the state NAME rather than on
any numeric code, so agreement is evidence and disagreement is a finding.

It has already earned its keep twice. It found the FY1984 record whose geography
block is damaged in the source, now corrected (README 1.5). And it found the
negative-zero sign byte of the FY1966-75 dollar field, which BOTH sides had been
reading wrongly and which was the whole of the FY1966-75 disagreement. Stage B
below is what that diagnosis looks like when it is rerun.

Four stages.

  A  the test           our state-year dollars against his series exactly as he
                        ships it, reporting the maximum absolute deviation and
                        the number of state-years off by more than rounding,
                        which is what the plan asks for
  B  the diagnosis      what the difference is, each candidate as a dollar
                        amount and never a verdict asserted without one
  C  both sides fixed   his series with the negative-zero rows repaired, against
                        ours. This is the comparison that says whether the two
                        independent extracts of the archive agree
  D  the verdict        every remaining deviation named individually, in the
                        style of check 9 of 06 and check 4 of 08, so a new one
                        fails

WHY OUR SIDE AND HIS DIFFER, in one paragraph. Byte 86 of the 272-byte record is
a zoned-decimal zone byte: it carries the dollar amount's last digit and its
sign together. In FY1966-75, and in no other fiscal year, negative zero is
written there as a BLANK - 1,495 to 1,921 rows a year, becoming "^" in FY1976-77
and "}" in FY1978-83. Read as padding, as R's as.numeric() reads it, the amount
comes out a tenth of the truth and positive; that is his series. Read as
positive zero, as this project read it until 2026-08-06, the amount is right and
the sign is inverted; that was ours. The archive says negative, on four
independent grounds set out in the ZONED table in dcads_common.py. His reading
therefore overstates his series by 1.1x the negative-zero total in each of those
ten years, and stage C removes exactly that before comparing.

The benchmark is variant A: RAW, all_dod_statename_noadj.csv. It is the only one
of his eight aggregates built without a geography correction, a reporting
threshold or an FMS exclusion, so it is the only one an unadjusted parse can be
compared against at all. Comparing against B through H would be comparing our
universe with his universe minus adjustments we have not made.

WHAT THIS IS NOT. It does not verify row counts, dollar decoding, the record's
own fiscal year or the county-sums-to-state invariant - those are 06's, and 06 is
deliberately left untouched so that rerunning it unmodified still proves the
parse did not move. It does not verify the county dimension of the ZARK
crosswalk either; that is 08 check 4. It tests the STATE dimension, which is as
far up as Pallante's published geography goes.

Output: $temp/dcads_year/pallante_test_report.csv
        $temp/dcads_year/pallante_test_diagnosis.csv
        $temp/dcads_year/pallante_test_corrected.csv
"""

import collections
import csv
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc

REPORT = dc.OUT_DIR / "pallante_test_report.csv"
DIAGNOSIS = dc.OUT_DIR / "pallante_test_diagnosis.csv"
CORRECTED = dc.OUT_DIR / "pallante_test_corrected.csv"

# A state-year matches when it is within a dollar. Anything larger is a
# disagreement about which records are in the universe, not rounding: both sides
# hold whole dollars, and his series is stored as integers.
ROUNDING = 1

BENCHMARK = "data_replication/aggregates/all_dod_statename_noadj.csv"
STATE_CODES = "data_replication/state_codes.txt"

ZARK_LAYOUT = "dcads_272"

# The negative-zero sign byte of the dollar field, in its three spellings across
# the 272-byte era. The census in B1 needs all three, so that the blank can be
# shown occupying the same slot as the two nobody disputes.
NEGATIVE_ZERO_BYTES = " ^}"

# Only the BLANK spelling is corrected in stage C. His reader handles "^" and "}"
# correctly - FY1976-83 match him to the dollar in stage A, which they could not
# if he were misreading their negative zeros - and it is only the blank that
# as.numeric() silently swallows.
BLANK_ZERO_BYTE = " "

# The ten files that write negative zero as a blank. Named rather than derived so
# that a file joining or leaving the set is a visible edit.
BLANK_ZERO_YEARS = ("1966", "1967", "1968", "1969", "1970", "1971", "1972",
                    "1973", "1974", "1975")

# ---------------------------------------------------------------------------
# The exceptions, named individually so that a new one fails.
#
# What used to be here was an eleven-year MPCF-era band attributed to "a
# record-universe difference between his extract and ours". That verdict was
# wrong. It was the negative-zero sign byte, which stage B now measures and stage
# C removes, and what is left is the three items below. The old stage B had
# concluded "not dollar decoding, FY1977-83 use the same layout and the same code
# path" - true, and the hole in it was that the blank sign byte never occurs
# after FY1975, so that code path was never exercised by the control years.

# (fiscal_year, state_fips2) -> (deviation in dollars, why)
#
# Differences that survive both corrections and are about which of two fields in
# the same record to believe. Ours is defensible in each case; his is a different
# defensible reading of a record the source wrote inconsistently.
ROW_EXCEPTIONS = {
    ("2002", "13"): (
        -184_684,
        "one row, contract DACA8101D0229, whose state_code is the FIPS 10-4 "
        "country code KS for the Republic of Korea and whose state_name reads "
        "KOREA, REP. We class it foreign, on the state code, which the state name "
        "confirms. His series places it in Georgia. Ours is the defensible "
        "reading and the amount is one row"),
}

# The coded geography against the record's own state_name, which is where the two
# series genuinely part company. B3 flags 36 ZARK-era rows of 4,236,226 where the
# two disagree; these are the ones that move dollars.
#
# His series is keyed on the name and ours on the code, and the RAW variant turns
# out to resolve misspellings of US state names: NEVEDA in FY1967 and WIS.;NSIN
# in FY1973 are flagged by B3 and produce no deviation at all, so he places them
# as we do. What his key cannot place is a record whose state_name is a FOREIGN
# country while its coded geography is domestic - BERMUDA, CANADA - or whose
# state_name is blank. Those we keep and he drops, and they are all of FY1967 and
# FY1973 below.
#
# The largest, and the one worth knowing about, because it is genuinely ambiguous
# rather than simply wrong: 14 FY1967 records carry ZARK state 39 with county 046
# and county_name MONTGOMERY - Montgomery County, Pennsylvania - with a blank city,
# while their state_name reads BERMUDA. All 14 are General Electric. Genuine
# Bermuda rows elsewhere in the same file carry state_code BD and are classed
# foreign, so the two fields cannot both be right.
#
# But either could be the true place of performance. GE's Re-entry Systems plant
# was in Montgomery County PA, Bermuda hosted an Air Force downrange tracking
# station, and these are Air Force Space Systems contracts (F0469...). We keep the
# coded geography because that is the rule the parse applies in every other year,
# NOT because BERMUDA is demonstrably wrong. And unlike the FY1984 record in
# dcads_common.NATIVE_OVERRIDES, this is not three fields against one: the state
# code and the county code plus name are one geography block written together, so
# it is two facts against one from the same reporting office. Left as it is, and
# recorded here, rather than overridden.
NAME_VS_CODE_EXCEPTIONS = {
    ("1967", "42"): (
        19_329_000,
        "14 General Electric records coded to Montgomery County, Pennsylvania "
        "whose state_name reads BERMUDA, $19,133,000, and which are genuinely "
        "ambiguous - see the note above; 2 coded to Pennsylvania and named NEW "
        "YORK, $105,000; 1 named CALIFORNIA, $25,000; and the positive leg of a "
        "NEW JERSEY pair, $66,000. He cannot place a domestic record named for a "
        "foreign country, so these leave his universe"),
    ("1967", "01"): (
        -42_000, "one record coded to New York and named ALABAMA: his Alabama "
                 "holds it, ours does not"),
    ("1967", "06"): (
        -307_000, "records coded to Oregon, Virginia, Pennsylvania and Florida "
                  "whose state_name reads CALIFORNIA: his California holds them"),
    ("1967", "12"): (
        79_000, "2 records coded to Florida and named CANADA, $63,000, and 1 "
                "named CALIFORNIA, $16,000"),
    ("1967", "34"): (-66_000, "the negative leg of the NEW JERSEY pair also "
                              "counted under state 42"),
    ("1967", "36"): (
        -63_000, "loses $105,000 to records coded to Pennsylvania but named NEW "
                 "YORK, gains $42,000 from one coded to New York and named ALABAMA"),
    ("1967", "41"): (
        183_000, "7 records coded to Oregon whose state_name reads CALIFORNIA"),
    ("1967", "51"): (
        83_000, "1 record coded to Virginia whose state_name reads CALIFORNIA"),
    ("1973", "12"): (
        28_000, "one record coded to Florida whose state_name field is blank, so "
                "his name key has nothing to place it by"),
}

# FY1976 is not locked as numbers. Its deviation is verified against a quantity
# recomputed from the archive on every run: the byte-identical repeat records in
# RG330.DEFCON.FY76 and RG330.DEFCON.TQ76, which his extract does not contain and
# ours does. The declared record count in DCADS_TSS.pdf is 223,130, which is what
# we read, so the repeats are in the archival file as distributed and dropping
# them is not something we can justify. Note this is NOT a general rule of his:
# FY1978-83 carry 6,300 to 11,661 such repeats worth $120m to $382m in the 51
# states and match him to the dollar anyway. So it is a difference in what his
# FY1976 extract holds, not a deduplication step, and it is 0.05% of the year.
#
# The net is also the residue of a much larger set that mostly cancels. The 276
# repeats are $262,268,000 gross in absolute value and 90.2% of that cancels
# within the set: twelve magnitude groups are exact +/- pairs, because the file
# holds an obligation, its reversal, AND a duplicate copy of each - +/-41,215k on
# N000306C0045, Lockheed at Sunnyvale, is the largest. So the $25.8m is an
# unpaired tail, not $25.8m of duplicated spending. TQ76 contributes 9 repeats,
# none above $61k.
REPEAT_ROW_YEARS = ("1976",)

# ---------------------------------------------------------------------------
# Pallante's own state-name mapping, transcribed from adjust_state() in
# 08_StataRoutineAggregation_NS14.R, both blocks, in his order. Used only by the
# diagnosis, and only to re-key our rows the way he keys his, so that a
# geography explanation can be separated from a universe one. Duplicated keys in
# his case_when resolve to the first arm, which is what setdefault reproduces.
_ADJUST_STATE_PAIRS = [
    ("ALABAMA", "AL"), ("ALASKA", "AK"), ("ARIZONA", "AZ"), ("ARKANSAS", "AR"),
    ("CALIFORNIA", "CA"), ("COLORADO", "CO"), ("CONN", "CT"), ("DELAWARE", "DE"),
    ("DC", "DC"), ("FLORIDA", "FL"), ("IDAHO", "ID"), ("ILLINOIS", "IL"),
    ("INDIANA", "IN"), ("IOWA", "IA"), ("KANSAS", "KS"), ("HAWAII", "HI"),
    ("KENTUCKY", "KY"), ("LOUISIANA", "LA"), ("MAINE", "ME"), ("MARYLAND", "MD"),
    ("MASS", "MA"), ("MICHIGAN", "MI"), ("MINNESOTA", "MN"), ("MISS", "MS"),
    ("MISSOURI", "MO"), ("MONTANA", "MT"), ("NCAR", "NC"), ("NDAK", "ND"),
    ("NEBRASKA", "NE"), ("NEVADA", "NV"), ("NEW JERSEY", "NJ"), ("NEW YORK", "NY"),
    ("NEW MEXICO", "NM"), ("NH", "NH"), ("OHIO", "OH"), ("OKLAHOMA", "OK"),
    ("OREGON", "OR"), ("PA", "PA"), ("RI", "RI"), ("SCAR", "SC"), ("SDAK", "SD"),
    ("TENNESSEE", "TN"), ("TEXAS", "TX"), ("VERMONT", "VT"), ("WASHINGTON", "WA"),
    ("WVA", "WV"), ("WYOMING", "WY"), ("VIRGINIA", "VA"), ("WISCONSIN", "WI"),
    ("UTAH", "UT"), ("GEORGIA", "GA"),
    ("Alabama", "AL"), ("Alaska", "AK"), ("Arizona", "AZ"), ("Arkansas", "AR"),
    ("California", "CA"), ("Colorado", "CO"), ("Conn", "CT"), ("Delaware", "DE"),
    ("D.C.", "DC"), ("Florida", "FL"), ("Idaho", "ID"), ("Illinois", "IL"),
    ("Indiana", "IN"), ("Iowa", "IA"), ("Kansas", "KS"), ("Hawaii", "HI"),
    ("Kentucky", "KY"), ("Louisiana", "LA"), ("Maine", "ME"), ("Maryland", "MD"),
    ("Mass", "MA"), ("Michigan", "MI"), ("Minnesota", "MN"), ("Miss", "MS"),
    ("Missouri", "MO"), ("Montana", "MT"), ("N Carolina", "NC"), ("N Dakota", "ND"),
    ("Nebraska", "NE"), ("Nevada", "NV"), ("New Jersey", "NJ"), ("New York", "NY"),
    ("New Mexico", "NM"), ("N.H.", "NH"), ("Ohio", "OH"), ("Oklahoma", "OK"),
    ("Oregon", "OR"), ("Penn", "PA"), ("Rhode Isld", "RI"), ("R Island", "RI"),
    ("RHODE IS", "RI"), ("S Carolina", "SC"), ("S Dakota", "SD"),
    ("Tennessee", "TN"), ("Texas", "TX"), ("Vermont", "VT"), ("Washington", "WA"),
    ("W Virginia", "WV"), ("Wyoming", "WY"), ("Virginia", "VA"),
    ("Wisconsin", "WI"), ("Utah", "UT"), ("Georgia", "GA"),
]
ADJUST_STATE = {}
for _name, _code in _ADJUST_STATE_PAIRS:
    ADJUST_STATE.setdefault(_name, _code)

# His department buckets, from the same function. Everything else becomes NA in
# his pipeline and lands in a pmaNA column his row totals exclude, so this is a
# universe restriction and the diagnosis measures what it would cost us.
HIS_DEPARTMENTS = {"1", "2", "3", "4", "5"}


def year_files():
    """(label, path) per fiscal year, with the transition quarter kept separate.

    TQ76 is a file in its own right and is folded into 1976 only at the point of
    comparison, because Pallante's series has no row for it: his FY1976 covers
    fifteen months. Folding it earlier would hide which file a deviation is in.
    """
    out = []
    for path in sorted(dc.OUT_DIR.glob("fy*.parquet")):
        out.append((path.stem[2:], path))
    return out


def benchmark_year(label):
    """The fiscal year of his series that our file label belongs to."""
    return "1976" if label.endswith("tq") else label


def grouped_files(labels):
    """benchmark year -> the files that make it up, TQ76 folded into 1976."""
    out = collections.defaultdict(list)
    for label, path in labels:
        out[benchmark_year(label)].append((label, path))
    return out


def has_columns(path, names):
    """Whether a file carries every column, so an era that lacks one is skipped.

    Only FY2004-06 lack reporting_dept: the delimited files hold the awarding
    agency differently. The diagnosis says so rather than failing on it.
    """
    have = set(pq.read_schema(path).names)
    return all(n in have for n in names)


def our_state_dollars(path):
    """State dollars from one parsed file, on the same universe his series has.

    is_lumpsum rows carry no geography by construction and his series has no row
    they could belong to. Everything else is keyed on state_fips2, which is the
    column that makes the ZARK era comparable at all; the caller restricts to the
    codes his file actually carries.
    """
    table = pq.read_table(path, columns=["dollars", "state_fips2", "is_lumpsum",
                                         "dollars_unit"])
    if not table.num_rows:
        return collections.defaultdict(int)
    mult = 1000 if table.column("dollars_unit").to_pylist()[0] == "thousands" else 1
    out = collections.defaultdict(int)
    for value, state, lump in zip(table.column("dollars").to_pylist(),
                                  table.column("state_fips2").to_pylist(),
                                  table.column("is_lumpsum").to_pylist()):
        if value is None or lump or not state:
            continue
        out[state] += value * mult
    return out


def read_benchmark():
    """His state-year series, keyed (fiscal_year, state FIPS), plus the universe.

    The universe is taken from the file itself rather than from state_codes.txt,
    which carries 56 rows including the territories. His series carries 51 codes
    - the 50 states and DC - and comparing against the other five would compare
    our territory dollars against rows that do not exist.
    """
    codes = dc.read_pallante_csv(STATE_CODES)
    rows = dc.read_pallante_csv(BENCHMARK)
    if codes is None or rows is None:
        return None, None, None
    fips = {r["statename"]: r["statefips"].zfill(2) for r in codes}
    by_state = collections.defaultdict(int)
    by_state_dept = collections.defaultdict(int)
    universe = set()
    for r in rows:
        code = fips.get(r["statename"])
        if code is None:
            continue
        value = int(float(r["tot_spending"] or 0))
        by_state[(r["fy_year"], code)] += value
        by_state_dept[(r["fy_year"], code, r["awarding_sub_agency_code"])] += value
        universe.add(code)
    return by_state, by_state_dept, universe


# ---------------------------------------------------------------------------
# Reading the raw source alongside the parse.
#
# Stages B and C need two things the parquet does not carry: which rows hold the
# ambiguous sign byte, and which rows are byte-identical repeats. Both are
# properties of the source bytes, so both are read from the source, which also
# makes the diagnosis independent of the parse rather than a restatement of it.
#
# The join is positional - the nth record of the file is the nth row of the
# parquet - and rather than assume that, every reader below checks it: the value
# it decodes from the raw bytes must equal the value the parquet holds, row by
# row. A parser that reordered or dropped a record fails here instead of
# producing a quietly wrong diagnosis.


def source_path_of(path):
    """The raw file a parquet was built from, from its own source_file column."""
    table = pq.read_table(path, columns=["source_file"])
    if not table.num_rows:
        return None
    name = table.column("source_file").to_pylist()[0]
    return next((p for p in dc.data_files() if p.name == name), None)


def scan_raw(path, universe):
    """Per-state blank-zero magnitudes and repeat-row excess for one file.

    Returns None when the raw file is not reachable, so a checkout without the
    3.46 GB of source degrades to stage A rather than failing.

    zero_magnitude   state -> sum of |value| over rows whose sign byte is the
                     BLANK spelling of negative zero, and only that spelling,
                     because that is the only one his reader gets wrong. This is
                     head x 10, the magnitude both range_code and our own reader
                     agree on
    zero_rows        how many such rows, restricted to the 51 states
    sign_bytes       the full histogram of the sign byte, for the B1 census
    repeat_excess    state -> dollars in rows that are a byte-identical repeat of
                     an earlier record in the same file, counting the excess
                     copies only
    """
    raw = source_path_of(path)
    if raw is None or not raw.exists():
        return None
    spec_cols = ["dollars", "state_fips2", "is_lumpsum", "dollars_unit",
                 "layout_id"]
    table = pq.read_table(path, columns=spec_cols)
    if not table.num_rows or table.column("layout_id").to_pylist()[0] != ZARK_LAYOUT:
        return None
    mult = 1000 if table.column("dollars_unit").to_pylist()[0] == "thousands" else 1
    states = table.column("state_fips2").to_pylist()
    values = table.column("dollars").to_pylist()
    lumps = table.column("is_lumpsum").to_pylist()

    zero_mag = collections.defaultdict(int)
    repeat = collections.defaultdict(int)
    zero_rows = repeat_rows = 0
    sign_bytes = collections.Counter()
    zero_detail = []                       # (range_code, head) for the range test
    seen = set()
    misaligned = 0
    i = 0
    with open(raw, "rb") as fh:
        while True:
            buf = fh.read(272 * dc.CHUNK_RECORDS)
            if not buf:
                break
            for k in range(0, len(buf), 272):
                rec = buf[k:k + 272]
                if i >= len(values):
                    misaligned += 1
                    i += 1
                    continue
                text = rec.decode("latin-1")
                field = text[77:86]
                sign_bytes[field[-1]] += 1
                if dc.decode_zoned(field) != values[i]:
                    misaligned += 1
                state, value, lump = states[i], values[i], lumps[i]
                usable = state in universe and value is not None and not lump
                if field[-1] == BLANK_ZERO_BYTE and field[:-1].strip():
                    zero_detail.append((text[86], int(field[:-1])))
                    if usable:
                        zero_rows += 1
                        zero_mag[state] += abs(value) * mult
                if rec in seen:
                    if usable:
                        repeat_rows += 1
                        repeat[state] += value * mult
                else:
                    seen.add(rec)
                i += 1
    return {"source": raw.name, "rows": i, "misaligned": misaligned,
            "zero_magnitude": zero_mag, "zero_rows": zero_rows,
            "zero_detail": zero_detail, "sign_bytes": sign_bytes,
            "repeat_excess": repeat, "repeat_rows": repeat_rows}


def range_code_bands(path):
    """range_code -> (min, max) of |value| over rows with an unambiguous sign.

    range_code is the record's own dollar-magnitude bucket, so it adjudicates the
    magnitude of the ambiguous rows without any reference to their sign, and
    without reference to Pallante. Rows carrying the ambiguous byte are excluded
    from the bands they are then tested against.
    """
    raw = source_path_of(path)
    if raw is None or not raw.exists():
        return None
    bands = {}
    with open(raw, "rb") as fh:
        while True:
            buf = fh.read(272 * dc.CHUNK_RECORDS)
            if not buf:
                break
            for k in range(0, len(buf), 272):
                text = buf[k:k + 272].decode("latin-1")
                field, code = text[77:86], text[86]
                if field[-1] in NEGATIVE_ZERO_BYTES or not code.strip():
                    continue
                value = dc.decode_zoned(field)
                if value is None:
                    continue
                lo, hi = bands.get(code, (None, None))
                a = abs(value)
                bands[code] = (a if lo is None else min(lo, a),
                               a if hi is None else max(hi, a))
    return bands


# ---------------------------------------------------------------------------
# Stage B, the diagnosis. Each function returns rows for the diagnosis CSV and
# prints what it found. None of them changes the verdict; they exist so that a
# deviation is explained rather than tolerated.


def diagnose_zero_code_point(labels, scans):
    """Where negative zero is written, year by year, in the 272-byte layout.

    The load-bearing table of the whole diagnosis. If the blank in FY1966-75 is
    not negative zero, then negative zero is absent from ten consecutive years
    while minus one through minus nine each occur a thousand times a year, and
    the code point reappears as "^" in FY1976 with the same frequency. Reading it
    as positive zero also requires positive zero to have two representations in
    the same file, since plain "0" and "+" are already there.
    """
    print("\n  B1  where negative zero is written in the 272-byte dollar field")
    print("      A zone byte carries the last digit and the sign together. Positive zero")
    print("      always has its own byte. If the blank is not negative zero, that code")
    print("      point is missing for ten years and then returns as '^'.")
    print(f"      {'fy':<9}{'+0 byte':>9}{'n(+0)':>9}{'n(-1..-9)':>11}"
          f"{'-0 byte':>9}{'n(-0)':>8}{'n(-0)/n(+0)':>13}")
    rows = []
    for label, _ in labels:
        scan = scans.get(label)
        if scan is None:
            continue
        hist = scan["sign_bytes"]
        pos = next((c for c in "0{+" if hist.get(c)), None)
        neg = next((c for c in NEGATIVE_ZERO_BYTES if hist.get(c)), None)
        if pos is None or neg is None:
            continue
        nine = sum(hist[c] for c in "JKLMNOPQR")
        ratio = hist[neg] / hist[pos]
        rows.append({"stage": "B1_zero_code_point", "fiscal_year": label,
                     "measure": f"negative zero written as {neg!r}, positive zero as {pos!r}",
                     "rows": hist[neg], "dollars": "",
                     "note": f"n(+0)={hist[pos]}, n(-1..-9)={nine}, ratio={ratio:.3f}"})
        print(f"      fy{label:<7}{pos!r:>9}{hist[pos]:>9,}{nine:>11,}"
              f"{neg!r:>9}{hist[neg]:>8,}{ratio:>13.3f}")
    print("      The blank sits in the same slot, at the same frequency, as codes nobody")
    print("      disputes. It is negative zero.")
    return rows


def diagnose_range_code(groups, scans):
    """Does the record's own magnitude bucket put the ambiguous rows at head x 10?

    Independent of the sign question and of Pallante. If the trailing blank were
    padding, the value would be the head alone; if it is a zone byte, the field
    carries a tenth digit and the value is head x 10. range_code says which.
    """
    print("\n  B2  the magnitude, from the record's own range_code bucket")
    print("      range_code is a declared |dollar| band. Read as a zone byte the value is")
    print("      head x 10; read as padding it is the head alone. Only one fits the band.")
    print(f"      {'fy':<9}{'-0 rows':>10}{'fit at head x 10':>18}"
          f"{'fit at head':>13}{'below own floor':>17}")
    rows = []
    for year in BLANK_ZERO_YEARS:
        files = groups.get(year, [])
        if not files:
            continue
        label, path = files[0]
        scan = scans.get(label)
        bands = range_code_bands(path)
        if scan is None or not bands:
            continue
        fit10 = fit1 = below = 0
        for code, head in scan["zero_detail"]:
            band = bands.get(code)
            if band is None:
                continue
            lo, hi = band
            if lo <= head * 10 <= hi:
                fit10 += 1
            if lo <= head <= hi:
                fit1 += 1
            elif head < lo:
                below += 1
        n = len(scan["zero_detail"])
        rows.append({"stage": "B2_range_code", "fiscal_year": year,
                     "measure": "ambiguous rows inside their own range_code band",
                     "rows": n, "dollars": "",
                     "note": f"head x 10: {fit10}; head alone: {fit1}; "
                             f"below own floor at head: {below}"})
        print(f"      fy{year:<7}{n:>10,}{fit10:>18,}{fit1:>13,}{below:>17,}")
    print("      Reading the byte as padding puts the amount below its own record's")
    print("      declared floor every time. The magnitude is head x 10, so his series is")
    print("      a tenth of the truth on these rows and ours was the right magnitude.")
    return rows


def diagnose_geography(labels, universe, fips_by_code):
    """Does state_fips2 agree with the record's own state name?

    state_fips2 comes from ZALF and the supplement; state_name is the
    place-of-performance name the record carries in bytes 154-164 of the 272-byte
    layout. Two independent naming systems, exactly as in check 4 of 08, so
    agreement is evidence the crosswalk is right.

    Both directions are counted, which the earlier version of this check did not
    do: it skipped any row whose name his mapping does not cover, and so missed
    the 14 FY1967 records coded to Pennsylvania and named BERMUDA - the largest
    single geography difference in the panel. A row is counted whenever EITHER
    side places it in the 51 states.
    """
    print("\n  B3  does the ZARK state code agree with the record's own state name?")
    print("      Counted in both directions: a row is asked about whenever either the code")
    print("      or the name puts it in the 51 states. A wrong crosswalk would fail on")
    print("      whole states at a time, not on a handful of records.")
    print(f"      {'fy':<10}{'rows asked':>12}{'agree':>12}{'disagree':>10}{'$ disagreeing':>16}")
    rows = []
    total_rows = total_bad = 0
    detail = collections.Counter()
    dollars = collections.Counter()
    for label, path in labels:
        table = pq.read_table(path, columns=["dollars", "state_name", "state_fips2",
                                             "is_lumpsum", "layout_id", "dollars_unit"])
        if not table.num_rows or table.column("layout_id").to_pylist()[0] != ZARK_LAYOUT:
            continue
        mult = 1000 if table.column("dollars_unit").to_pylist()[0] == "thousands" else 1
        asked = agree = bad = 0
        bad_dollars = 0
        for value, name, state, lump in zip(table.column("dollars").to_pylist(),
                                            table.column("state_name").to_pylist(),
                                            table.column("state_fips2").to_pylist(),
                                            table.column("is_lumpsum").to_pylist()):
            if value is None or lump:
                continue
            his = fips_by_code.get(ADJUST_STATE.get((name or "").strip(), ""), "")
            ours = state or ""
            if his not in universe and ours not in universe:
                continue           # foreign or a territory on both readings
            asked += 1
            if his == ours:
                agree += 1
            else:
                bad += 1
                bad_dollars += value * mult
                detail[(label, (name or "").strip(), ours, his)] += 1
                dollars[(label, (name or "").strip(), ours, his)] += value * mult
        total_rows += asked
        total_bad += bad
        rows.append({"stage": "B3_geography", "fiscal_year": label,
                     "measure": "rows where state name and ZARK state code agree",
                     "rows": agree, "dollars": "", "note": f"{asked} rows asked"})
        rows.append({"stage": "B3_geography", "fiscal_year": label,
                     "measure": "rows where they disagree",
                     "rows": bad, "dollars": bad_dollars, "note": ""})
        print(f"      fy{label:<8}{asked:>12,}{agree:>12,}{bad:>10,}{bad_dollars:>15,}")
    print(f"\n      {total_rows - total_bad:,} of {total_rows:,} ZARK-era rows agree. "
          f"{total_bad} do not:")
    for key, n in dollars.most_common():
        label, name, ours, his = key
        print(f"        fy{label}  state_name {name!r} -> his {his or '--'}, "
              f"ours {ours or '--'}  {n:>14,}  {detail[key]} row(s)")
    print("      Every one is the source contradicting itself between its coded geography")
    print("      and its name field, not a crosswalk error.")
    return rows


def diagnose_his_key(groups, benchmark_dept, failing, fips_by_code):
    """Re-aggregate our rows on HIS key and compare at his own granularity.

    His series is keyed on the state NAME and the reporting department, not on
    any numeric code. Re-keying our rows the same way removes every geographic
    difference by construction: whatever survives is a difference about which
    records exist, or about how a record's amount is read.
    """
    print("\n  B4  our rows re-keyed the way he keys his: state name x reporting department")
    print("      This removes geography from the comparison by construction. What is left")
    print("      is a difference about record universe or about decoding.")
    print(f"      {'fy':<10}{'cells':>8}{'off':>7}{'ours':>18}{'his':>18}{'gap':>16}")
    rows, skipped = [], []
    for year in sorted(failing):
        files = groups.get(year, [])
        if not files:
            continue
        if not all(has_columns(p, ("reporting_dept",)) for _, p in files):
            skipped.append(year)
            continue
        ours = collections.defaultdict(int)
        for _, path in files:
            table = pq.read_table(path, columns=["dollars", "state_name",
                                                 "reporting_dept", "is_lumpsum",
                                                 "dollars_unit"])
            mult = 1000 if table.column("dollars_unit").to_pylist()[0] == "thousands" else 1
            for value, name, dept, lump in zip(table.column("dollars").to_pylist(),
                                               table.column("state_name").to_pylist(),
                                               table.column("reporting_dept").to_pylist(),
                                               table.column("is_lumpsum").to_pylist()):
                if value is None or lump:
                    continue
                # his adjust_state gives a two-letter code; his published file
                # keys states by FIPS, so finish through his own table
                code = fips_by_code.get(ADJUST_STATE.get((name or "").strip(), ""))
                if code:
                    ours[(code, dept)] += value * mult
        his = {(s, d): v for (y, s, d), v in benchmark_dept.items() if y == year}
        keys = set(ours) | set(his)
        off = sum(1 for k in keys if abs(ours.get(k, 0) - his.get(k, 0)) > ROUNDING)
        gap = sum(ours.values()) - sum(his.values())
        rows.append({"stage": "B4_his_key", "fiscal_year": year,
                     "measure": "cells off on his own key", "rows": off,
                     "dollars": gap, "note": f"{len(keys)} cells"})
        print(f"      fy{year:<8}{len(keys):>8}{off:>7}{sum(ours.values()):>18,}"
              f"{sum(his.values()):>18,}{gap:>16,}")
    if skipped:
        print(f"      not asked of {', '.join('fy' + y for y in skipped)}: the delimited")
        print("      FY2004-06 files hold the awarding agency in a different field, so his")
        print("      key is not reproducible there.")
    return rows


def diagnose_repeats(groups, scans):
    """Byte-identical repeat records, on the failing years and on years that match.

    The control column is the point. FY1978-83 carry thousands of these and match
    his series to the dollar, so dropping them is not a rule of his pipeline, and
    FY1976 cannot be explained by saying that it is.
    """
    print("\n  B5  byte-identical repeat records, in the 51 states")
    print("      Present in most years of the era. FY1978-83 carry thousands of them and")
    print("      match him to the dollar, so this is not a deduplication rule of his.")
    print(f"      {'fy':<10}{'repeat rows':>13}{'their dollars':>17}{'matches him?':>14}")
    rows = []
    for year in sorted(groups):
        n = amount = 0
        seen_any = False
        for label, _ in groups[year]:
            scan = scans.get(label)
            if scan is None:
                continue
            seen_any = True
            n += scan["repeat_rows"]
            amount += sum(scan["repeat_excess"].values())
        if not seen_any:
            continue
        rows.append({"stage": "B5_repeats", "fiscal_year": year,
                     "measure": "dollars in byte-identical repeat records",
                     "rows": n, "dollars": amount, "note": ""})
        print(f"      fy{year:<8}{n:>13,}{amount:>17,}"
              f"{('no' if year in REPEAT_ROW_YEARS else 'yes'):>14}")
    return rows


def diagnose_universe(groups, failing):
    """The three candidate universe mechanisms, each as a dollar amount.

    Kept from the original diagnosis. All three are measured on the failing years
    and on years that match, because a mechanism present in a year that matches
    to the dollar cannot be what breaks a year that does not.
    """
    print("\n  B6  his aggregation restrictions, measured on failing and matching years")
    print("      A restriction also present in a year matching to the dollar is not what")
    print("      breaks a year that does not.")
    print(f"      {'fy':<9}{'matches?':>10}{'$ names he drops':>19}"
          f"{'$ depts he drops':>19}")
    rows = []
    for year in sorted(groups):
        files = groups[year]
        if not all(has_columns(p, ("reporting_dept",)) for _, p in files):
            continue
        dropped_name = dropped_dept = 0
        is_zark = False
        for _, path in files:
            table = pq.read_table(path, columns=["dollars", "state_name",
                                                 "reporting_dept", "is_lumpsum",
                                                 "layout_id", "dollars_unit"])
            if not table.num_rows:
                continue
            if table.column("layout_id").to_pylist()[0] == ZARK_LAYOUT:
                is_zark = True
            mult = 1000 if table.column("dollars_unit").to_pylist()[0] == "thousands" else 1
            for value, name, dept, lump in zip(
                    table.column("dollars").to_pylist(),
                    table.column("state_name").to_pylist(),
                    table.column("reporting_dept").to_pylist(),
                    table.column("is_lumpsum").to_pylist()):
                if value is None or lump:
                    continue
                if not ADJUST_STATE.get((name or "").strip(), ""):
                    dropped_name += value * mult
                elif dept not in HIS_DEPARTMENTS:
                    dropped_dept += value * mult
        if not is_zark:
            continue
        matches = "no" if year in failing else "YES"
        rows.append({"stage": "B6_universe", "fiscal_year": year,
                     "measure": "dollars on state names his adjust_state drops",
                     "rows": "", "dollars": dropped_name, "note": "foreign and territory"})
        rows.append({"stage": "B6_universe", "fiscal_year": year,
                     "measure": "dollars in departments outside his five buckets",
                     "rows": "", "dollars": dropped_dept,
                     "note": f"benchmark matches: {matches}"})
        print(f"      fy{year:<7}{matches:>10}{dropped_name:>19,}{dropped_dept:>19,}")
    print("      The names he drops are foreign and territory work, excluded on both")
    print("      sides. Every MPCF record carries a department in his five buckets, so")
    print("      that restriction costs nothing. Neither is the mechanism.")
    return rows


# ---------------------------------------------------------------------------


def main():
    print(f"{'=' * 78}")
    print("The Pallante test: our state-year dollars against his published series")
    print(f"{'=' * 78}")

    benchmark, benchmark_dept, universe = read_benchmark()
    if benchmark is None:
        print("UNAVAILABLE: Pallante's replication archive was not found in either", file=sys.stderr)
        print("checkout, so the test could not run. Expected at:", file=sys.stderr)
        print(f"  2_illustration_aggregation/Pallante/replication/"
              f"rcodes_reassessing-cross-sect-fiscal-multiplier.zip", file=sys.stderr)
        return 2
    print(f"\nbenchmark: {BENCHMARK}")
    print(f"  variant A: RAW - no geography correction, no reporting threshold, no FMS")
    print(f"  exclusion. The only one of his eight an unadjusted parse can be compared to.")
    print(f"  {len(universe)} state codes, the 50 states and DC. Territories are not in his")
    print(f"  series and are excluded from both sides rather than counted on one.")

    codes = dc.read_pallante_csv(STATE_CODES)
    fips_by_code = {r["statecode"]: r["statefips"].zfill(2) for r in codes}

    labels = year_files()
    groups = grouped_files(labels)

    # ---- the raw scan, shared by stages B and C --------------------------
    scans = {}
    for label, path in labels:
        scan = scan_raw(path, universe)
        if scan is not None:
            scans[label] = scan
    bad_align = {k: v["misaligned"] for k, v in scans.items() if v["misaligned"]}
    if bad_align:
        print(f"\nFAIL: the raw source and the parse are not row-aligned, so the "
              f"diagnosis cannot be trusted: {bad_align}", file=sys.stderr)
        return 1
    if scans:
        aligned = sum(v["rows"] for v in scans.values())
        print(f"\nraw source re-read for the {len(scans)} files of the 272-byte era: "
              f"{aligned:,} records,")
        print(f"  every one decoding to the value the parse holds, so the positional join")
        print(f"  stages B and C use is verified rather than assumed.")
    else:
        print(f"\nnote: the raw source is not reachable from this checkout, so stages B and")
        print(f"  C are skipped and only stage A runs.")

    # ---- stage A ----------------------------------------------------------
    ours = collections.defaultdict(int)
    for label, path in labels:
        year = benchmark_year(label)
        for state, value in our_state_dollars(path).items():
            if state in universe:
                ours[(year, state)] += value

    # His series runs to 2019 and the contract archive ends at FY2006, so the
    # comparison is over the overlap. His later years are out of scope, not a
    # deviation, and saying so is the difference between a report and a number
    # that reads as a hundred per cent failure.
    our_years = sorted({y for y, _ in ours})
    his_years = sorted({y for y, _ in benchmark})
    out_of_scope = [y for y in his_years if y not in set(our_years)]

    print(f"\n{'=' * 78}\nA  the test, against his series exactly as he ships it")
    print("   Our side: is_lumpsum = 0, keyed on state_fips2, thousands rescaled where the")
    print("   era stores thousands. FY1976 is compared with TQ76 folded in, because his")
    print("   FY1976 covers the fifteen months the source reports under that label.")
    print(f"   Compared over fy{our_years[0]} to fy{our_years[-1]}, the overlap. His series")
    print(f"   runs to fy{his_years[-1]}; the {len(out_of_scope)} later years have no contract file "
          f"and are out of scope.")
    print(f"\n   {'fy':<8}{'ours':>18}{'his':>18}{'gap':>15}{'gap %':>8}"
          f"{'off':>5}{'max dev':>15}")
    rows_out = []
    for year in our_years:
        deviations = [(s, ours.get((year, s), 0) - benchmark.get((year, s), 0))
                      for s in sorted(universe)]
        off = [(s, d) for s, d in deviations if abs(d) > ROUNDING]
        our_total = sum(ours.get((year, s), 0) for s in universe)
        his_total = sum(benchmark.get((year, s), 0) for s in universe)
        pct = 100 * (our_total - his_total) / his_total if his_total else 0
        rows_out.append({
            "fiscal_year": year, "states_compared": len(universe),
            "states_off": len(off), "our_dollars": our_total,
            "his_dollars": his_total, "gap_dollars": our_total - his_total,
            "gap_pct": f"{pct:.6f}",
            "max_abs_deviation": max((abs(d) for _, d in off), default=0),
            "total_abs_deviation": sum(abs(d) for _, d in off),
            "off_detail": "; ".join(f"{s}:{d:+}" for s, d in off)})
        print(f"   {year:<8}{our_total:>18,}{his_total:>18,}"
              f"{our_total - his_total:>15,}{pct:>7.3f}%{len(off):>5}"
              f"{max((abs(d) for _, d in off), default=0):>15,}")

    exact_a = [r for r in rows_out if not r["states_off"]]
    print(f"\n   {sum(r['states_compared'] for r in rows_out):,} state-years compared "
          f"across {len(rows_out)} fiscal years.")
    print(f"   fiscal years exact in every one of the {len(universe)} states: "
          f"{len(exact_a)} of {len(rows_out)}")
    print(f"     {', '.join('fy' + r['fiscal_year'] for r in exact_a)}")
    print("   The FY1966-75 gap is negative because his series reads the negative-zero")
    print("   sign byte as padding: it is a tenth of the truth AND the wrong sign, so his")
    print("   total is too high by 1.1x the negative-zero amount. Stage B measures that")
    print("   and stage C removes it.")

    # ---- stage B ----------------------------------------------------------
    print(f"\n{'=' * 78}\nB  the diagnosis")
    diag_rows = []
    failing_a = {r["fiscal_year"] for r in rows_out if r["states_off"]}
    if not scans:
        print("\n  Skipped: the raw source is not reachable from this checkout.")
    elif not failing_a:
        print("\n  Nothing to diagnose: every state-year matched to the dollar.")
    else:
        print(f"\n  {len(failing_a)} fiscal years carry a deviation: "
              f"{', '.join('fy' + y for y in sorted(failing_a))}")
        diag_rows += diagnose_zero_code_point(labels, scans)
        diag_rows += diagnose_range_code(groups, scans)
        diag_rows += diagnose_geography(labels, universe, fips_by_code)
        diag_rows += diagnose_his_key(groups, benchmark_dept, failing_a, fips_by_code)
        diag_rows += diagnose_repeats(groups, scans)
        diag_rows += diagnose_universe(groups, failing_a)

    # ---- stage C ----------------------------------------------------------
    # His series, with the negative-zero rows repaired. On those rows he holds
    # +magnitude/10 and the archive says -magnitude, so his year-state total is
    # too high by 1.1 x magnitude. Nothing else about his series is touched.
    print(f"\n{'=' * 78}\nC  both sides corrected")
    print("   His published value on a negative-zero row is +magnitude/10; the archive")
    print("   says -magnitude. So his state-year total is high by 1.1 x magnitude, and")
    print("   that is the only thing subtracted here. Ours is the parse as it now stands.")
    zero_by_state = collections.defaultdict(int)
    for label, scan in scans.items():
        year = benchmark_year(label)
        for state, mag in scan["zero_magnitude"].items():
            zero_by_state[(year, state)] += mag
    repeat_by_state = collections.defaultdict(int)
    for label, scan in scans.items():
        year = benchmark_year(label)
        for state, amount in scan["repeat_excess"].items():
            repeat_by_state[(year, state)] += amount

    print(f"\n   {'fy':<8}{'ours':>18}{'his corrected':>18}{'gap':>14}{'off':>5}"
          f"{'-0 removed from his':>21}")
    corrected_rows, failures = [], []
    for year in our_years:
        removed = sum(zero_by_state.get((year, s), 0) for s in universe)
        his_fixed = {s: benchmark.get((year, s), 0)
                     - (11 * zero_by_state.get((year, s), 0)) // 10 for s in universe}
        deviations = [(s, ours.get((year, s), 0) - his_fixed[s]) for s in sorted(universe)]
        off = [(s, d) for s, d in deviations if abs(d) > ROUNDING]
        our_total = sum(ours.get((year, s), 0) for s in universe)
        fix_total = sum(his_fixed.values())

        unexplained = []
        for state, dev in off:
            known = (ROW_EXCEPTIONS.get((year, state))
                     or NAME_VS_CODE_EXCEPTIONS.get((year, state)))
            if known is not None and known[0] == dev:
                continue
            if year in REPEAT_ROW_YEARS and dev == repeat_by_state.get((year, state), 0):
                continue
            unexplained.append(f"state {state} off by {dev:,}, not a named exception")
        if unexplained:
            failures.append((year, unexplained))

        corrected_rows.append({
            "fiscal_year": year, "status": "FAIL" if unexplained else "ok",
            "states_compared": len(universe), "states_off": len(off),
            "our_dollars": our_total, "his_corrected_dollars": fix_total,
            "gap_dollars": our_total - fix_total,
            "negative_zero_removed_from_his": (11 * removed) // 10,
            "off_detail": "; ".join(f"{s}:{d:+}" for s, d in off),
            "unexplained": "; ".join(unexplained)})
        print(f"   {year:<8}{our_total:>18,}{fix_total:>18,}"
              f"{our_total - fix_total:>14,}{len(off):>5}{(11 * removed) // 10:>21,}")

    exact_c = [r for r in corrected_rows if not r["states_off"]]
    print(f"\n   fiscal years exact in every one of the {len(universe)} states: "
          f"{len(exact_c)} of {len(corrected_rows)}")
    print(f"     {', '.join('fy' + r['fiscal_year'] for r in exact_c)}")
    gained = sorted({r["fiscal_year"] for r in exact_c}
                    - {r["fiscal_year"] for r in exact_a})
    if gained:
        print(f"   correcting the sign byte on both sides makes {len(gained)} more years "
              f"exact: {', '.join('fy' + y for y in gained)}")

    # ---- stage D ----------------------------------------------------------
    print(f"\n{'=' * 78}\nD  the verdict")
    print("\n  Every deviation that survives stage C is named below. A deviation that is")
    print("  not on this list fails the script, in the style of check 9 of 06 and check 4")
    print("  of 08.")
    print("\n  Single records, one field of the record against another:")
    for (year, state), (dev, why) in sorted(ROW_EXCEPTIONS.items()):
        print(f"    fy{year} state {state}  {dev:>+12,}   {why}")
    print("\n  The coded geography against the record's own state name:")
    for (year, state), (dev, why) in sorted(NAME_VS_CODE_EXCEPTIONS.items()):
        print(f"    fy{year} state {state}  {dev:>+12,}   {why}")
    print("\n  Byte-identical repeat records his extract does not hold:")
    for year in REPEAT_ROW_YEARS:
        n = sum(scans[l]["repeat_rows"] for l, _ in groups.get(year, []) if l in scans)
        amount = sum(repeat_by_state.get((year, s), 0) for s in universe)
        print(f"    fy{year}  {amount:>+12,}   {n} rows, verified against the archive on")
        print(f"              every run rather than locked as a number. The declared record")
        print(f"              count in DCADS_TSS.pdf is what we read, so the repeats are in")
        print(f"              the file as distributed. Not a rule of his: FY1978-83 carry")
        print(f"              thousands of them and match to the dollar")

    with open(REPORT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    with open(CORRECTED, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(corrected_rows[0]))
        w.writeheader()
        w.writerows(corrected_rows)
    if diag_rows:
        with open(DIAGNOSIS, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["stage", "fiscal_year", "measure",
                                               "rows", "dollars", "note"])
            w.writeheader()
            w.writerows(diag_rows)

    print(f"\n{'=' * 78}")
    print(f"accounted for {len(corrected_rows) - len(failures)} of {len(corrected_rows)} "
          f"fiscal years")
    print(f"report:    {REPORT}")
    print(f"corrected: {CORRECTED}")
    if diag_rows:
        print(f"diagnosis: {DIAGNOSIS}")
    if failures:
        print(f"\nFAIL: {len(failures)} fiscal years carry a deviation nothing accounts for",
              file=sys.stderr)
        for year, why in failures:
            for w in why:
                print(f"  fy{year}: {w}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
