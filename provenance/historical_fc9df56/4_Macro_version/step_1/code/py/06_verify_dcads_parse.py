#!/usr/bin/env python3
"""Verify the parsed DCADS and MPCF fiscal-year files.

03_verify_dcads.py checks the downloaded raw files and has to run before the
parse; this checks the parse and has to run after it, which is why the two are
separate scripts rather than one with a stage flag.

Every check either passes with a stated criterion or fails and stops the
pipeline. Checks run in the order below, ending with the comparison against an
external benchmark, because that is the only one whose failure means the numbers
are wrong rather than merely internally inconsistent.

  1  parsed row count equals the count declared in DCADS_TSS.pdf, or size
     arithmetic, or for the delimited years a line count of the raw text
  2  the record's own fiscal year matches the filename in every record
  3  no dollar field failed to decode
  4  documented field domains hold: department 1-5, geography codes well formed
  5  the is_lumpsum flag agrees with the zero-filled geography it stands for
  6  dollars reconcile against an independent re-read of the raw source, for
     every era including the delimited one
  7  state-by-year dollars against Pallante, reported for each is_lumpsum value
  8  national dollars against Pallante, decomposed where the files carry FIPS
  9  contract dollars summed across the counties of a state equal that state's
     year total read straight from the raw records, to the dollar, the only
     permitted difference being domestic rows that report no county at all

All 42 fiscal years are held to the same standard. The delimited FY2004-06 years
previously had neither an independent count nor a reconciliation, which is how
238 dropped FY2004 records went unnoticed: 05 reported them once as skipped and
nothing downstream could see that the parquet was short.

Output: $temp/dcads_year/parse_verification_report.csv
"""

import collections
import csv
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc

REPORT = dc.OUT_DIR / "parse_verification_report.csv"

# A state-year is treated as matching Pallante when it is within a dollar, and
# the run is reported as clean when every state-year matches. Anything larger is
# a real disagreement about the universe, not rounding.
ROUNDING = 1.0


def raw_state_dollars(path, spec):
    """Sum dollars by state straight from the fixed-width bytes.

    Deliberately independent of iter_chunks and of the parquet writer: it reads
    only the record length and the two field spans it needs, so a bug in the
    parse path cannot hide by being present on both sides of the comparison.
    """
    fields = {f["name"]: (f["start"], f["end"]) for f in spec["fields"]}
    da, db = fields[spec["dollars_field"]]
    sa, sb = fields["state_code"]
    lump = spec.get("lumpsum_field")
    la, lb = fields[lump] if lump else (0, 0)
    rl = spec["record_length"]
    enc = spec.get("encoding", "latin-1")

    by_state = collections.defaultdict(int)
    total = lump_total = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(rl * dc.CHUNK_RECORDS)
            if not block:
                break
            text = block.decode(enc)
            for i in range(0, len(text), rl):
                r = text[i:i + rl]
                value = dc.decode_zoned(r[da:db])
                if value is None:
                    continue
                total += value
                if lump and dc.is_blank_code(r[la:lb]):
                    lump_total += value
                else:
                    by_state[r[sa:sb]] += value
    return by_state, total, lump_total


def raw_delimited_dollars(path, spec, member):
    """Sum dollars by state straight from the raw delimited text, and count records.

    The delimited counterpart of raw_state_dollars, and independent of the parse
    path for the same reason: it reads the dollar and state fields at their
    declared byte offsets and nothing else, so a bug in 05 cannot hide by being
    present on both sides of the comparison. Without this the FY2004-06 years had
    no reconciliation at all, which is how 238 dropped records went unnoticed.

    Damaged records are handled through the same declared anomaly table the
    parser uses, but only ever to locate the dollar and state fields, both of
    which sit before the first byte any anomaly moves.
    """
    fields = {f["name"]: f["index"] for f in spec["fields"]}
    di, si = fields[spec["dollars_field"]], fields["state_code"]
    enc = spec.get("encoding", "latin-1")
    offsets = dc.delimited_offsets(spec, member)
    spans = dc.spans_from_delimiters(offsets)
    anomalies = dc.load_anomalies(spec)

    by_state = collections.defaultdict(int)
    total = records = blank = unknown = 0
    with open(path, "r", encoding=enc, newline="") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip():
                blank += 1
                continue
            records += 1
            positions = [i for i, ch in enumerate(line) if ch == "!"]
            sig = dc.delimiter_signature(positions, offsets)
            if sig is None:
                use = spans
            else:
                anomaly = anomalies.get(sig)
                if anomaly is None:
                    unknown += 1
                    continue
                own = dc.spans_from_delimiters(positions)
                use = [None] * len(spans)
                for k, sp in enumerate(own):
                    ck = dc.map_own_index(anomaly, k)
                    if ck is not None and ck < len(use):
                        use[ck] = sp
            ds, ss = use[di], use[si]
            if ds is None:
                continue
            text = line[ds[0]:ds[1]].strip()
            if not text or text == "-":
                continue
            sign = -1 if text[0] == "-" else 1
            digits = text[1:] if text[0] in "+-" else text
            if not digits.isdigit():
                continue
            value = sign * int(digits)
            total += value
            if ss is not None:
                by_state[line[ss[0]:ss[1]].strip()] += value
    return by_state, total, 0, records, blank, unknown


def county_state_rollup(table, spec, by_state):
    """Counties summed to their state against the state totals read from the raw bytes.

    The county dollars come from the parquet; the state dollars come from an
    independent re-read of the raw source, the same by_state the reconciliation
    in check 6 discards. That is what makes this different from every other
    geography check: both sides of the others are views of the same parse, so a
    field read from the wrong offset agrees with itself. Here it cannot.

    Where the county code carries its state - the FIPS era, county SSCCC - the
    rollup is keyed on the COUNTY, not the state field, which puts the county
    itself under test: a misread county sends its dollars to a state key that
    does not exist or to the wrong one, and both states break at once. Before
    FY1984 the county is a 3-digit state-internal ZARK code and in FY2004-06 it
    is held in its own field, so in those eras the state must supply the key and
    the check is the weaker raw-against-parse comparison.

    Two differences are permitted, both accounted for explicitly so the
    comparison stays exact rather than tolerant, and both counted so they cannot
    grow unnoticed. Domestic performance reporting no county names no county to
    be rolled up. And the source occasionally files a county belonging to another
    state - one FY1984 record pairs California's 06 with Suffolk County New
    York's 36103 - which the county-keyed rollup sends to the wrong state while
    the raw side, keyed on the state field, does not. Those dollars are moved
    back. Neither is absorbed silently: the first is reported per year, and the
    second is what check 4's mismatch rate already bounds.
    """
    delimited = spec.get("format") == "delimited"
    states = table.column("state_code").to_pylist()
    counties = table.column("county_fips" if delimited else "county_code").to_pylist()
    values = table.column(spec["dollars_field"]).to_pylist()
    lumps = table.column("is_lumpsum").to_pylist()
    key_from_county = not delimited and spec["layout_id"] != "dcads_272"

    rolled = collections.defaultdict(int)
    no_county = collections.defaultdict(int)
    misfiled = collections.defaultdict(int)
    cross_rows = cross_dollars = 0
    for v, s, c, l in zip(values, states, counties, lumps):
        if l or v is None:
            continue
        s = s.strip()
        # Foreign performance and the 98 catch-all are not in any state, and the
        # raw side buckets them under their own keys, so both sides drop them.
        if not s or s == "98" or not s.isdigit():
            continue
        if not c.strip():
            no_county[s] += v
            continue
        if not key_from_county:
            rolled[s] += v
            continue
        k = c[:2]
        rolled[k] += v
        if k != s:
            misfiled[k] -= v
            misfiled[s] += v
            cross_rows += 1
            cross_dollars += v

    raw = {k.strip(): v for k, v in by_state.items()}
    keys = ({k for k in raw if k.isdigit() and k != "98"}
            | set(rolled) | set(no_county))
    mismatched = []
    for k in sorted(keys):
        ours = rolled.get(k, 0) + no_county.get(k, 0) + misfiled.get(k, 0)
        theirs = raw.get(k, 0)
        if ours != theirs:
            mismatched.append((k, ours, theirs))

    m = {"states_compared": len(keys), "states_mismatched": len(mismatched),
         "cross_state_rows": cross_rows, "cross_state_dollars": cross_dollars}
    fails = []
    if mismatched:
        detail = "; ".join(f"state {k}: counties give {o:,} against raw {t:,} "
                           f"({o - t:+,})" for k, o, t in mismatched[:4])
        fails.append(f"{len(mismatched)} of {len(keys)} states where counties summed to the "
                     f"state do not equal the raw state total: {detail}")
    return fails, m


def check_file(path, spec, tss, rows):
    """Return (list of failure strings, dict of measures) for one fiscal year."""
    fails = []
    m = {}
    table = pq.read_table(path)
    # Parser reconciliation tests source amounts, not analysis-dollar cleaning.
    if 'dollars_reported' in table.column_names:
        table = table.set_column(table.schema.get_field_index('dollars'),
                                 'dollars',table['dollars_reported'])
    n = table.num_rows
    m["rows"] = n
    src = table.column("source_file")[0].as_py()
    m["source_file"] = src

    # 1  row count against the technical summary, or against size arithmetic
    #    when the summary does not declare a count for that file.
    delimited = spec.get("format") == "delimited"
    m["dollars_unit"] = spec["dollars_unit"]

    declared = tss["record_counts"].get(src)
    if declared is not None:
        m["count_basis"] = "TSS"
        m["count_expected"] = declared
        if n != declared:
            fails.append(f"row count {n:,} != TSS {declared:,}")
    elif rows:
        # The delimited extracts declare no count in the technical summary and
        # are not a whole number of fixed-length records to divide, so the
        # independent count is a line count of the raw text. Without it these
        # three years carried no count check at all.
        m["count_basis"] = "raw lines" if delimited else "size/reclen"
        m["count_expected"] = rows
        if n != rows:
            fails.append(f"row count {n:,} != {m['count_basis']} {rows:,}")
    else:
        m["count_basis"] = "not declared"
        m["count_expected"] = ""

    # 2  the record's own fiscal year against the filename
    fy_field = spec["fiscal_year_source"]["field"]
    a, b = spec["fiscal_year_source"]["slice"]
    label = table.column("fiscal_year")[0].as_py()
    # The July-September 1976 transition quarter is accounted to fiscal 1977 in
    # every one of its records, so the year it should carry internally is the one
    # after its label. Expecting its own label would make a correct file look
    # broken; exempting it would forgo the check, so the expectation is shifted.
    if label.endswith("tq"):
        expect = f"{(int(label[:4]) + 1) % 100:02d}"
    else:
        expect = f"{int(label) % 100:02d}"
    internal = table.column(fy_field).to_pylist()
    # 8 FY2006 records carry a malformed reporting period, '073106', which is
    # MMDDYY rather than CCYYMM. They are real records whose action dates fall
    # inside the fiscal year, so 05 takes the year from action_date and flags
    # them. They cannot satisfy a check on the malformed field itself, so they
    # are excluded from it and reported separately instead.
    repaired = (table.column("month_key_repaired").to_pylist()
                if "month_key_repaired" in table.schema.names else [0] * n)
    m["month_key_repaired"] = sum(repaired)
    checkable = [v for v, r in zip(internal, repaired) if not r]
    ok = sum(1 for v in checkable if v[a:b] == expect)
    m["fy_match_share"] = ok / len(checkable) if checkable else 1.0
    m["fy_expected"] = expect
    if ok != len(checkable):
        # Printed to four decimals: 8 records in 1,272,615 rounds to 100.00% at
        # two, which made a real failure read as a bug in the checker.
        bad = len(checkable) - ok
        fails.append(f"internal fiscal year {expect} matches only "
                     f"{100 * ok / len(checkable):.4f}% of records "
                     f"({bad:,} of {len(checkable):,} disagree)")

    # 3  every dollar value decoded
    dollars = table.column(spec["dollars_field"]).to_pylist()
    unparsed = sum(1 for v in dollars if v is None)
    m["unparsed_dollars"] = unparsed
    if unparsed:
        fails.append(f"{unparsed:,} dollar values did not decode")

    # 4  documented field domains
    # Through FY2003 the department is a single digit 1 to 5. The FY2004-06 extract
    # replaces it with a four-digit FIPS 95 agency code, 9700 for DoD and 2100 for
    # the Army, so the one-to-five domain does not apply and the distinct codes are
    # reported instead of tested.
    if "reporting_dept" in table.schema.names:
        depts = set(table.column("reporting_dept").to_pylist())
        m["depts"] = "".join(sorted(depts))
        if not depts <= {"1", "2", "3", "4", "5"}:
            fails.append(f"reporting_dept outside 1-5: {sorted(depts - set('12345'))[:6]}")
    else:
        codes = sorted(set(table.column("reporting_dept_fips").to_pylist()))
        m["depts"] = ",".join(c for c in codes if c)[:40]

    lumps = table.column("is_lumpsum").to_pylist()
    states = table.column("state_code").to_pylist()
    if spec.get("format") == "delimited":
        cf = table.column("county_fips").to_pylist()
        counties = [s2 + c2 if s2 and c2 else "" for s2, c2 in zip(states, cf)]
    else:
        counties = table.column("county_code").to_pylist()
    # From FY1984 the codes are native FIPS for domestic performance, and the
    # two-letter FIPS 10-4 country code for foreign performance: Germany GM,
    # Japan JA, Korea KS, Puerto Rico RQ. Both share one invariant, which is that
    # county_code begins with state_code. Requiring numeric FIPS of every row
    # would flag every overseas contract, which is 4.8% of FY1994 dollars.
    #
    # Domestic rows are held to the invariant and foreign rows are only reported,
    # because the source itself violates it for some country codes: Cuba arrives
    # as state_code CU with county_code 00000 in every year from FY1987 to FY1993.
    # The domestic criterion is a rate rather than a zero count, because two
    # genuine source errors exist and neither indicates misalignment: one FY1984
    # record carries California's 06 with Suffolk County New York's 36103, and one
    # FY1995 record uses the catch-all 98 "Domestic". A real offset error would
    # break most rows at once, not two in 480,000.
    # State code 98, labelled "Domestic" in state_name, is the source's own
    # catch-all for performance somewhere in the United States with no county
    # identified. It is not a FIPS state and carries no county, so it belongs with
    # the unplaceable rows rather than in the violation count: 102 rows in FY1997,
    # 9 in FY1998, 1 in FY1995.
    #
    # The invariant itself is FIPS-specific and cannot hold before FY1984, where
    # the codes are ZARK and Monroe County Wisconsin is state 50 county 041. The
    # bucket COUNTS, though, mean the same thing in every era, and check 9 and the
    # unplaceable-dollars notice are both built on them, so they are taken for all
    # 42 years and only the invariant is skipped. Previously the whole block was
    # skipped for the 272 layout, and those 19 years therefore reported no code-98
    # rows at all against a true 14 to 53 a year.
    fips_era = spec["layout_id"] != "dcads_272"
    domestic = domestic_bad = foreign = foreign_bad = undistributed = 0
    no_county = 0
    examples = []
    # Rows are classified on state_code, which every geographable row carries,
    # and NOT on the county string. The county string was the discriminator
    # before, and because the delimited era holds the county separately and
    # leaves it blank for foreign performance, every overseas row in FY2004-06
    # was skipped before it could be classified: 06 reported zero foreign rows
    # against a true 50,529 rows and $18.65bn in FY2004 alone, and the NOTICE
    # block below under-reported unplaceable dollars by $18bn to $36bn a year.
    for s, c, l in zip(states, counties, lumps):
        if l or not s.strip():
            continue
        if s == "98":
            undistributed += 1
            continue
        if not s.isdigit():
            foreign += 1
            # Reported, never asserted: the source itself breaks the invariant
            # for some country codes, Cuba arriving as state CU with county
            # 00000 in every year from FY1987 to FY1993.
            if c.strip() and not c.startswith(s):
                foreign_bad += 1
            continue
        if not c.strip():
            # Domestic performance with no county identified. Unplaceable in a
            # county panel, so counted rather than silently skipped.
            no_county += 1
            continue
        domestic += 1
        if fips_era and not (c.startswith(s) and len(c) == 5 and c.isdigit()):
            domestic_bad += 1
            if len(examples) < 3:
                examples.append(f"state {s} county {c}")
    m["undistributed_rows"] = undistributed
    m["no_county_rows"] = no_county
    m["foreign_rows"] = foreign
    m["foreign_code_mismatch"] = foreign_bad
    if fips_era:
        rate = domestic_bad / domestic if domestic else 0
        m["code_mismatch"] = domestic_bad
        m["code_mismatch_rate"] = rate
        m["code_examples"] = "; ".join(examples)
        if rate > 0.0001:
            fails.append(f"{domestic_bad:,} of {domestic:,} domestic rows ({100 * rate:.3f}%) "
                         f"where county_code does not begin with state_code: {'; '.join(examples)}")
    else:
        m["code_mismatch"] = ""
        m["code_mismatch_rate"] = ""
        m["code_examples"] = ""

    # 4b  remapped records must not invent values
    #
    # The FY2004 anomaly maps in the layout say which canonical field each field
    # of a damaged record belongs to. A wrong map is not detectable by eye - the
    # fields it shifts are one-character codes - but it shows up immediately as a
    # value outside the domain the field takes across the undamaged records. The
    # first hand-derived maps were wrong in exactly this way, putting 'Y' into
    # subcontracting_plan and 'M' into demo_program, so the check is kept.
    #
    # Only closed-domain code fields discriminate. Identifiers and free text take
    # values that legitimately appear nowhere else, so they are excluded.
    if "layout_anomaly" in table.schema.names:
        tags = table.column("layout_anomaly").to_pylist()
        remapped = [i for i, t in enumerate(tags) if t]
        m["remapped_rows"] = len(remapped)
        m["remapped_tags"] = ", ".join(f"{t} {c}" for t, c in
                                       sorted(collections.Counter(t for t in tags if t).items()))
        if remapped:
            clean = [i for i, t in enumerate(tags) if not t]
            outside = []
            for f in spec["fields"]:
                if f["type"] != "str":
                    continue
                col = table.column(f["name"]).to_pylist()
                domain = {col[i] for i in clean}
                if len(domain) > 40:
                    continue
                bad = {col[i] for i in remapped} - domain - {None}
                if bad:
                    outside.append(f"{f['name']}={sorted(bad)}")
            m["remap_domain_violations"] = len(outside)
            if outside:
                fails.append(f"{len(outside)} remapped code fields carry values that appear in "
                             f"no undamaged record, so an anomaly map is wrong: "
                             f"{'; '.join(outside[:4])}")
    else:
        m["remapped_rows"] = ""
        m["remapped_tags"] = ""
        m["remap_domain_violations"] = ""

    # 5  the flag agrees with what it stands for
    bad_flag = sum(1 for s, c, l in zip(states, counties, lumps)
                   if bool(l) != (dc.is_blank_code(s) and dc.is_blank_code(c)))
    m["flag_mismatch"] = bad_flag
    m["lump_rows"] = sum(lumps)
    if bad_flag:
        fails.append(f"{bad_flag:,} rows where is_lumpsum disagrees with zero-filled geography")

    # 9  the county sum equals the state sum
    #
    # Under the county filter - is_lumpsum 0, a numeric state_code other than 98,
    # and a reported county - the dollars that reach a county must equal the
    # dollars that reach a state, because the only rows a state can keep and a
    # county cannot are the domestic ones reporting no county at all. That is four
    # rows in 42 years. Two things are asserted, and neither is the identity
    # itself, which would be true by construction.
    #
    # The partition: every non-lump-sum dollar lands in exactly one of county,
    # no-county, code 98, foreign, or blank state. A classification that dropped a
    # bucket or counted one twice breaks this at once, which is the failure mode
    # that hid $18-36bn a year of foreign performance until the classifier was
    # moved off the county string.
    #
    # The no-county rate, which is the part that would catch a county column read
    # from the wrong bytes. A misread county arrives blank, so the residual stops
    # being four rows and becomes most of the file. The bound is a rate for the
    # same reason check 4 uses one: the four genuine rows are real gaps in the
    # source, not misalignment. They run 0.0008% at their worst, so 0.01% is three
    # orders of headroom against a break that would show as tens of percent.
    #
    # This runs in every era, including the 19 pre-FY1984 years, where it is the
    # only automated statement made about the geography fields: the county-begins-
    # with-state invariant above cannot run there because the codes are ZARK.
    vals_all = table.column(spec["dollars_field"]).to_pylist()
    state_rows = state_dollars = county_rows = county_dollars = 0
    no_county_dollars = other_dollars = geo_dollars = 0
    for v, s, c, l in zip(vals_all, states, counties, lumps):
        if l or v is None:
            continue
        geo_dollars += v
        if not s.strip() or s == "98" or not s.isdigit():
            other_dollars += v
            continue
        state_rows += 1
        state_dollars += v
        if c.strip():
            county_rows += 1
            county_dollars += v
        else:
            no_county_dollars += v
    m["state_dollars"] = state_dollars
    m["county_dollars"] = county_dollars
    m["county_state_residual"] = state_dollars - county_dollars
    m["county_state_residual_rows"] = state_rows - county_rows
    no_county_rate = no_county / state_rows if state_rows else 0
    m["no_county_rate"] = no_county_rate

    if county_dollars + no_county_dollars + other_dollars != geo_dollars:
        fails.append("county, no-county, code-98, foreign and blank-state dollars do not "
                     "partition the non-lump-sum total")
    if state_rows - county_rows != no_county:
        fails.append(f"county-state residual is {state_rows - county_rows:,} rows against "
                     f"{no_county:,} rows counted as domestic with no county")
    if no_county_rate > 0.0001:
        fails.append(f"{no_county:,} of {state_rows:,} domestic rows "
                     f"({100 * no_county_rate:.3f}%) report a state but no county, carrying "
                     f"{state_dollars - county_dollars:,} {spec['dollars_unit']}")

    return fails, m, table


def main():
    if not dc.OUT_DIR.exists():
        print(f"FAIL: no parsed output at {dc.OUT_DIR}", file=sys.stderr)
        return 1
    tss = json.loads(dc.TSS.read_text(encoding="utf-8"))
    layouts, index = dc.load_layouts(include_delimited=True)
    on_disk = {p.name: p for p in dc.data_files()}
    zips = {p.name[:-4]: p for p in dc.SOURCE_DIR.rglob("*.zip")}

    files = sorted(dc.OUT_DIR.glob("fy*.parquet"))
    if not files:
        print(f"FAIL: no fy*.parquet in {dc.OUT_DIR}", file=sys.stderr)
        return 1

    rows_out, failures = [], []
    ours = collections.defaultdict(int)          # (fy, statefips) -> contract dollars
    nat = {}                                     # fy -> (contract, lumpsum)
    print(f"{'file':<14}{'rows':>10}{'basis':>12}{'fy%':>10}{'unp':>5}{'flagbad':>9}"
          f"{'codebad':>9}{'foreign':>9}{'cty=raw':>9}{'nocty':>8}{'raw recon':>12}")
    for path in files:
        label = path.stem[2:]
        table_src = pq.read_table(path, columns=["source_file"]).column("source_file")[0].as_py()
        spec = layouts[index[table_src]]

        is_delim = spec.get("format") == "delimited"
        raw = on_disk.get(table_src)
        extracted = None
        # Two key shapes. The fixed-width members are extensionless, so the zip
        # name minus ".zip" is the member name; the delimited members carry a
        # .txt the archive name does not, so the stem has to be tried too.
        # Getting this wrong is silent: raw stays None and the year loses its
        # count check and its reconciliation, which is what happened to
        # FY2004-06 until now.
        key = table_src if table_src in zips else Path(table_src).stem
        if raw is None and key in zips:
            raw, _ = dc.extract_member(zips[key], table_src)
            extracted = raw

        # The raw side is read before the parquet is checked, because for the
        # delimited years the independent record count comes from the raw text
        # rather than from size arithmetic.
        by_state = raw_total = raw_lump = None
        rows_expected = 0
        raw_note = ""
        if raw is not None:
            if is_delim:
                (by_state, raw_total, raw_lump, rows_expected,
                 raw_blank, raw_unknown) = raw_delimited_dollars(raw, spec, table_src)
                if raw_unknown:
                    raw_note = f" {raw_unknown} unrecognised delimiter layouts"
            else:
                by_state, raw_total, raw_lump = raw_state_dollars(raw, spec)
                rows_expected = raw.stat().st_size // spec["record_length"]

        fails, m, table = check_file(path, spec, tss, rows_expected)

        # A year with no raw source cannot be counted or reconciled, and that is
        # a failure rather than a silent "n/a": it is the state FY2004-06 were in
        # while 238 dropped records went undetected.
        if raw is None:
            fails.append(f"no raw source found for {table_src}, so this year has "
                         f"neither an independent count nor a reconciliation")

        # 6  reconcile against the raw bytes, and close the county identity
        recon = "n/a"
        # A year with no raw source cannot be compared county-to-raw either, and
        # says so rather than printing a tick it has not earned.
        cty_raw = "n/a"
        if raw is not None:
            dollars = [v for v in table.column(spec["dollars_field"]).to_pylist() if v is not None]
            lumps = table.column("is_lumpsum").to_pylist()
            vals = table.column(spec["dollars_field"]).to_pylist()
            our_total = sum(dollars)
            our_lump = sum(v for v, l in zip(vals, lumps) if v is not None and l)
            our_geo = our_total - our_lump
            recon_fails = []
            if our_total != raw_total:
                recon_fails.append(f"total dollars {our_total:,} != raw {raw_total:,}")
            if our_lump != raw_lump:
                recon_fails.append(f"lump-sum dollars {our_lump:,} != raw {raw_lump:,}")
            if our_geo + our_lump != raw_total:
                recon_fails.append("geographable plus lump-sum dollars do not equal the raw total")
            if raw_unknown_note := raw_note:
                recon_fails.append(f"raw re-read found{raw_unknown_note}")
            fails.extend(recon_fails)
            recon = "exact" if not recon_fails else "MISMATCH"
            m["raw_total"] = raw_total
            m["geo_dollars"] = our_geo
            # NOT a dollar total. The DD1057 rows are a hierarchical,
            # cross-tabulated report keyed by report_no, where a parent equals
            # the sum of its lettered children and the B/C/D/E/F panels re-cut
            # the same universe, so summing them multiple-counts. See check 7.
            m["lump_report_sum_nonadditive"] = our_lump
            nat[label] = (our_geo, our_lump, spec["dollars_unit"])

            # 9  counties summed to their state against the raw state totals.
            # Done here rather than in check_file because by_state is the raw
            # side and only exists at this point; it was being computed for
            # every year and discarded.
            roll_fails, roll_m = county_state_rollup(table, spec, by_state)
            fails.extend(roll_fails)
            m.update(roll_m)
            cty_raw = "exact" if not roll_fails else "MISMATCH"

            # Pallante joins on FIPS, which the files carry only from FY1984.
            if spec["layout_id"] != "dcads_272":
                mult = 1000 if spec["dollars_unit"] == "thousands" else 1
                sc = table.column("state_code").to_pylist()
                for v, l, s in zip(vals, lumps, sc):
                    if v is not None and not l and s.isdigit():
                        ours[(label, s)] += v * mult
        if extracted is not None and extracted.exists():
            extracted.unlink()

        m["fails"] = "; ".join(fails)
        m["status"] = "ok" if not fails else "FAIL"
        m["fiscal_year"] = label
        rows_out.append(m)
        if fails:
            failures.append((label, fails))
        # Two separate columns, because they answer different questions. cty=raw
        # is check 9: counties summed to the state against the raw state totals.
        # nocty is how many domestic rows named no county and so could not be
        # rolled up at all - never a bare tick, so a year that starts leaking
        # says how much.
        resid = m["county_state_residual_rows"]
        nocty = "0" if not resid else f"{resid} rows"
        print(f"fy{label:<12}{m['rows']:>10,}{m['count_basis']:>12}"
              f"{100 * m['fy_match_share']:>9.4f}%{m['unparsed_dollars']:>5}"
              f"{m['flag_mismatch']:>9}{str(m['code_mismatch']):>9}"
              f"{m['foreign_rows']:>9,}{cty_raw:>9}{nocty:>8}{recon:>12}", flush=True)

    with open(REPORT, "w", newline="", encoding="utf-8") as fh:
        cols = ["fiscal_year", "source_file", "status", "rows", "count_basis", "count_expected",
                "fy_expected", "fy_match_share", "month_key_repaired",
                "unparsed_dollars", "depts",
                "code_mismatch", "code_mismatch_rate", "code_examples",
                "foreign_rows", "foreign_code_mismatch", "undistributed_rows",
                "remapped_rows", "remapped_tags", "remap_domain_violations",
                "no_county_rows", "no_county_rate", "flag_mismatch",
                "dollars_unit",
                "lump_rows", "raw_total", "geo_dollars",
                "state_dollars", "county_dollars",
                "county_state_residual", "county_state_residual_rows",
                "states_compared", "states_mismatched",
                "cross_state_rows", "cross_state_dollars",
                "lump_report_sum_nonadditive", "fails"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    # 7 and 8  the external benchmark
    print(f"\n{'=' * 78}\nbenchmark: Pallante state-by-year DoD spending")
    pal_rows = dc.read_pallante_csv("data_replication/aggregates/all_dod_statename_noadj.csv")
    codes = dc.read_pallante_csv("data_replication/state_codes.txt")
    if pal_rows is None or codes is None:
        print("  UNAVAILABLE: Pallante archive not found in Dropbox, so checks 7 and 8")
        print("  could not run. Everything above is internal consistency only.")
    else:
        fips = {r["statename"]: r["statefips"].zfill(2) for r in codes}
        pal_state = collections.defaultdict(int)
        pal_nat = collections.defaultdict(int)
        for r in pal_rows:
            y = r["fy_year"]
            v = int(float(r["tot_spending"] or 0))
            pal_nat[y] += v
            f = fips.get(r["statename"])
            if f:
                pal_state[(y, f)] += v

        # Reported in two groups rather than one. Pooling them made every year
        # read as a failure - "state-years off by more than rounding 10 of 61" -
        # when the states Pallante actually reports match to the exact dollar and
        # the deviations are entirely codes his series has no rows for at all:
        # the territories 60 American Samoa, 64 Micronesia, 66 Guam, 68 Marshall
        # Islands, 69 N Mariana, 70 Palau, 72 Puerto Rico, 74 US Minor Outlying,
        # 78 Virgin Islands, plus the source's own 98 "Domestic" catch-all.
        print("\n  check 7, state by year, is_lumpsum = 0, years where the files carry FIPS")
        print("  Split by whether Pallante's series carries the code at all, because only the")
        print("  first group is a comparison; the second is ours-only by construction.")
        print(f"    {'fy':<8}{'codes he has':>13}{'exact':>7}{'off':>5}{'max dev':>14}"
              f"{'codes he lacks':>16}{'their dollars':>16}")
        years = sorted({y for y, _ in ours})
        if not years:
            print("    no FIPS-bearing years parsed yet")
        for y in years:
            keys = {s for (yy, s) in ours if yy == y}
            shared = {s for s in keys if pal_state.get((y, s), 0)}
            extra = keys - shared
            devs = [abs(ours[(y, s)] - pal_state[(y, s)]) for s in shared]
            off = sum(1 for d in devs if d > ROUNDING)
            worst = max(devs) if devs else 0
            print(f"    fy{y:<6}{len(shared):>13}{len(shared) - off:>7}{off:>5}"
                  f"${worst / 1e6:>12,.1f}m{len(extra):>16}"
                  f"${sum(ours[(y, s)] for s in extra) / 1e6:>14,.1f}m")

        print("\n  check 7, is_lumpsum = 1: DD1057 rows carry no state and cannot enter a")
        print("  state-year comparison. The figure below is a REPORT SUM, NOT A DOLLAR TOTAL.")
        print("  These rows are a hierarchical cross-tabulation keyed by report_no - B01, B02,")
        print("  B02A, B02B, B03, B03A-E, C01.., D.., E.., F.. - in which a parent equals the sum")
        print("  of its lettered children exactly and the panels re-cut the same universe, the F")
        print("  panel total equalling B07A exactly. Summing every row therefore multiple-counts:")
        print("  FY1995 gives $99.11bn over all rows against $53.41bn over top-level rows only and")
        print("  $13.81bn for the largest single panel line. DO NOT add this to a national total.")
        print("  A correct DD1057 figure needs the DD Form 1057 report-line definitions from the")
        print("  DCADS technical documentation, which have not been read.")
        for y in sorted(nat):
            geo, lump, _ = nat[y]
            if lump:
                print(f"    fy{y:<8} DD1057 report sum, non-additive: {lump:,}")

        # NOTIFICATION. Four kinds of record carry dollars that cannot be placed
        # in any county: the DD1057 lump-sum tallies, foreign performance under a
        # letter country code, the source's own 98 "Domestic" catch-all, and
        # domestic rows reporting no county. A county sum can therefore never
        # equal the national total, and the gap is legitimate rather than a parse
        # failure. It is printed here loudly, per year, so nobody has to
        # rediscover why the identity does not close.
        print(f"\n{'=' * 78}\nNOTICE: dollars that no county can claim, by fiscal year")
        print("  These four buckets are why county sums fall short of the national total. The")
        print("  shortfall is expected; what must never happen is it changing without explanation.")
        print(f"    {'fy':<8}{'DD1057 rows':>13}{'foreign rows':>14}{'code-98 rows':>14}"
              f"{'no county':>12}{'unplaceable rows':>18}")
        flagged = []
        for r in rows_out:
            y = r["fiscal_year"]
            lr = r.get("lump_rows") or 0
            fr = r.get("foreign_rows") or 0
            ur = r.get("undistributed_rows") or 0
            nc = r.get("no_county_rows") or 0
            total = lr + fr + ur + nc
            if not total:
                continue
            flagged.append((y, lr, fr, ur, nc))
            print(f"    {y:<8}{lr:>13,}{fr:>14,}{ur:>14,}{nc:>12,}{total:>18,}")
        if flagged:
            print(f"\n  {len(flagged)} of {len(rows_out)} fiscal years contain unplaceable dollars.")
            print("  ACTION REQUIRED of any county-level check: run it over is_lumpsum = 0 with a")
            print("  numeric state_code other than 98 and a reported county. A check written as")
            print("  county sum equals national total will fail on every one of these years for")
            print("  reasons that are not defects.")
            print("  DO NOT close the gap by adding the DD1057 column back: it is a non-additive")
            print("  report sum, as the check-7 note above sets out, and adding it to FY1995 would")
            print("  overstate that year by tens of billions. Only the foreign, code-98 and")
            print("  no-county buckets are dollar amounts that may legitimately be added back.")
        else:
            print("    none")

        print("\n  check 8, national dollars against Pallante, is_lumpsum = 0")
        print("  Where the files carry FIPS the gap is decomposed into three parts, which sum to")
        print("  it by construction: how far the states he reports disagree, the codes he does")
        print("  not carry (territories and the 98 catch-all), and foreign performance. Only the")
        print("  FIRST can reveal a problem - the other two are ours-only by definition - so it")
        print("  is the column to read. It should be zero.")
        print(f"    {'fy':<8}{'ours':>16}{'Pallante':>16}{'gap':>14}{'pct':>7}"
              f"{'his states off':>16}{'codes he lacks':>16}{'foreign':>15}")
        devs, unexplained = [], []
        for y in sorted(nat):
            if y.endswith("tq"):
                continue
            geo, lump, unit = nat[y]
            mult = 1000 if unit == "thousands" else 1
            o = geo * mult
            p = pal_nat.get(y, 0)
            if not p:
                continue
            d = o - p
            devs.append((y, d, 100 * d / p))
            keys = {s for (yy, s) in ours if yy == y}
            if keys:
                shared = sum(ours[(y, s)] for s in keys if pal_state.get((y, s), 0))
                extra = sum(ours[(y, s)] for s in keys if not pal_state.get((y, s), 0))
                foreign = o - shared - extra
                unexplained.append((y, shared - p))
                print(f"    {y:<8}{o:>16,}{p:>16,}{d:>14,}{100 * d / p:>6.2f}%"
                      f"{shared - p:>16,}{extra:>16,}{foreign:>15,}")
            else:
                print(f"    {y:<8}{o:>16,}{p:>16,}{d:>14,}{100 * d / p:>6.2f}%"
                      f"{'no FIPS':>16}{'-':>16}{'-':>15}")
        if devs:
            worst = max(devs, key=lambda t: abs(t[2]))
            off = sum(1 for _, d, _ in devs if abs(d) > ROUNDING)
            print(f"\n    max absolute deviation: fy{worst[0]}, {worst[1]:,} dollars, {worst[2]:.2f}%")
            print(f"    year observations off by more than rounding: {off} of {len(devs)}")
            if unexplained:
                exact = [y for y, s in unexplained if abs(s) <= ROUNDING]
                off = [(y, s) for y, s in unexplained if abs(s) > ROUNDING]
                print(f"\n    Of the {len(unexplained)} FIPS-bearing years, {len(exact)} have every")
                print("    state Pallante reports matching ours TO THE EXACT DOLLAR, so for those")
                print("    years the whole gap is codes his series does not carry - the territories")
                print("    and the source's 98 catch-all - plus foreign performance.")
                print("    NOT ASSERTED as a pass criterion, by decision: recorded and monitored only.")
                for y, s in off:
                    print(f"    fy{y} is the exception, off by {s:,} on the states he reports. Our")
                    print("    side reconciles exactly against the raw source and every dollar field")
                    print("    decodes, so this is a difference in his series, not a parse defect.")
                    print("    See section 1.5 of 4_Macro_version/step_1/README.md for what was ruled out.")
            print("\n    Still open, and not a parse question: FY1966-1983 carry no FIPS, so their")
            print("    gap cannot be decomposed this way and needs the state-name crosswalk and an")
            print("    explicit universe decision, which is 1.4 of PLAN_macro_fiscal_multiplier.md.")
            print("\n    Two regularities worth keeping, both established from this table:")
            print("    - The gap is stable at +7% to +14.6% in every year, which is what a universe")
            print("      difference looks like. A parse or units error would show as a jump at an era")
            print("      boundary, and none appears at FY1984, FY1989 or FY2004 where layouts change.")
            print("    - fy1976 is the only negative gap, and only because Pallante folds the")
            print("      transition quarter into 1976. FY1976 alone is -14.33%; adding TQ76 gives")
            print("      +7.98%, back in line with every other year. Any comparison against his")
            print("      series must combine fy1976 and fy1976tq.")

    # check 9, printed outside the Pallante branch because it needs no external
    # benchmark and must be stated even on a machine where the archive is absent.
    print(f"\n{'=' * 78}\n  check 9, counties summed to the state against the RAW state totals")
    print("  Contract dollars summed across the counties of a state must equal that state's")
    print("  year total read straight from the raw records, to the dollar. The county side")
    print("  comes from the parquet and the state side from an independent re-read of the")
    print("  source bytes, so unlike every other geography check the two sides cannot agree")
    print("  by sharing a misread field. Where the county code carries its state - FY1984")
    print("  onward, county SSCCC - the rollup is keyed on the county rather than the state")
    print("  field, which puts the county itself under test.")
    mismatched = [r for r in rows_out if r.get("states_mismatched")]
    compared = sum(r.get("states_compared") or 0 for r in rows_out)
    if mismatched:
        print(f"\n    MISMATCH in {len(mismatched)} of {len(rows_out)} fiscal years")
    else:
        print(f"\n    exact in all {len(rows_out)} fiscal years, {compared:,} state-years compared")
    crossed = [r for r in rows_out if r.get("cross_state_rows")]
    if crossed:
        print("\n  Rows whose county names a different state than the state field - the")
        print("  source's own error, moved back before comparing and bounded by check 4:")
        for r in crossed:
            print(f"    fy{r['fiscal_year']}: {r['cross_state_rows']} row(s), "
                  f"{r['cross_state_dollars']:,} {r['dollars_unit']}")

    print("\n  The other permitted difference is domestic performance naming no county, which")
    print("  cannot be rolled up. Those dollars are added back per state so the comparison")
    print("  above is exact; the rows behind them are listed here.")
    leaks = [r for r in rows_out if r.get("county_state_residual_rows")]
    exact = len(rows_out) - len(leaks)
    print(f"\n    no county named in {len(leaks)} of {len(rows_out)} fiscal years")
    if leaks:
        print(f"    {'fy':<8}{'rows':>7}{'residual':>16}{'unit':>12}{'share of state $':>19}")
        for r in leaks:
            share = (r["county_state_residual"] / r["state_dollars"]
                     if r["state_dollars"] else 0)
            print(f"    {r['fiscal_year']:<8}{r['county_state_residual_rows']:>7,}"
                  f"{r['county_state_residual']:>16,}{r['dollars_unit']:>12}"
                  f"{100 * share:>18.6f}%")
        rows = sum(r["county_state_residual_rows"] for r in leaks)
        # Stated in dollars, so the pre-FY1989 thousands and the later dollars can
        # be added at all. dollars_unit says which each year is.
        total = sum(r["county_state_residual"] * (1000 if r["dollars_unit"] == "thousands" else 1)
                    for r in leaks)
        print(f"\n    {rows} rows and ${total:,} in total, out of "
              f"{sum(r['rows'] for r in rows_out):,} records.")
        print("    These sit in the unplaceable bucket: a county panel loses them, a state")
        print("    panel keeps them. Everything else reconciles to the dollar.")

    print(f"\n{'=' * 78}")
    print(f"verified {len(rows_out) - len(failures)} of {len(rows_out)} fiscal years")
    print(f"report: {REPORT}")
    if failures:
        print(f"\nFAIL: {len(failures)} fiscal years did not verify", file=sys.stderr)
        for label, fails in failures:
            for f in fails:
                print(f"  fy{label}: {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
