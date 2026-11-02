#!/usr/bin/env python3
"""Parse the exclamation-delimited DCADS extracts for FY2004 to FY2006.

From FY2004 the files stop being pure fixed-width EBCDIC-converted records and
become a DD Form 350 extract whose fields are separated by exclamation marks.
Three things differ enough from 04_parse_dcads.py to need a separate reader:
the fields are addressed by position in the delimited line rather than by byte
offset, the dollar amounts carry an explicit leading sign instead of a trailing
zoned-decimal one, and a lone hyphen marks a field as not reported.

Layouts with "format": "delimited" are handled here and skipped by 04.

Addressing is by BYTE OFFSET, not by splitting on the delimiter. These files are
delimited and fixed-width at the same time: every record is 601 characters (603
in FY2005) and the delimiters sit at identical positions in every well-formed
record, one pattern with no variants across all 3.3 million records. Splitting
threw that away and cost 238 FY2004 records outright while silently shifting the
tail fields of 52 more. Under positional addressing a record that loses a
delimiter still yields every field before the damage, and the four known FY2004
damage patterns are declared in the layout and remapped field by field.

Output: $temp/dcads_year/fy<year>.parquet, the same per-fiscal-year files the
fixed-width parser writes, with the same derived columns.
"""

import collections
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc

CHUNK_LINES = 100_000


def parse_number(text):
    """Return the signed integer in a delimited numeric field, or None.

    Fields carry a leading sign rather than the trailing zoned-decimal sign of
    the earlier eras, and a lone hyphen means not reported rather than a minus
    sign with no digits. None arrives for a field the record does not carry at
    all, which is distinct from a field present and blank.
    """
    if text is None:
        return None
    t = text.strip()
    if not t or t == "-":
        return None
    sign = -1 if t[0] == "-" else 1
    digits = t[1:] if t[0] in "+-" else t
    if not digits.isdigit():
        return None
    return sign * int(digits)


def clean(text):
    """Strip padding, and map the not-reported hyphen to an empty string.

    None is preserved rather than emptied: a field the record does not carry is
    null, a field it carries blank is "". Collapsing the two would hide which
    fields a damaged record actually reported.
    """
    if text is None:
        return None
    t = text.strip()
    return "" if t == "-" else t


def canonical_matcher(offsets):
    """Compile a regex that matches exactly the canonical delimiter layout.

    One C-level match per record instead of 88 Python slices, which matters at
    3.3 million records. Anchoring every field width also means the match is a
    proof of canonical positions, not a heuristic: a record that has moved any
    delimiter cannot match.
    """
    widths = [b - a for a, b in dc.spans_from_delimiters(offsets)]
    pattern = "".join(f".{{{w}}}!" for w in widths) + ".*$"
    return re.compile(pattern, re.DOTALL)


def remap(line, anomaly, positions, n_fields, index_of, stats):
    """Field values by canonical index for a record with damaged delimiters.

    The record's own spans are walked and each is placed at the canonical index
    the layout's index_map gives it. Fields the record does not carry stay None.
    A fused entry splits one own field that holds two canonical fields end to
    end.

    An imputed entry then fills fields the record omits from the value the rest
    of its contract reports. These are contract and contractor attributes that do
    not vary within a contract, so the value is recoverable even though the record
    does not carry it; see the entry's own basis note for the evidence. Two
    guards: the fill applies only when the record's contract matches the one the
    evidence was established on, and only fields whose recovered value is a real
    code are listed, so nothing is filled merely to turn a null into a blank.
    """
    values = [None] * n_fields
    own = dc.spans_from_delimiters(positions)
    for k, (a, b) in enumerate(own):
        ck = dc.map_own_index(anomaly, k)
        if ck is None or ck >= n_fields:
            continue
        values[ck] = line[a:b]
    fused = anomaly.get("fused")
    if fused:
        a, b = own[fused["own_index"]]
        raw = line[a:b]
        values[index_of[fused["first"]]] = raw[:fused["split_at"]]
        values[index_of[fused["second"]]] = raw[fused["split_at"]:]
    for name in anomaly.get("nulled", []):
        values[index_of[name]] = None
    imputed = anomaly.get("imputed")
    if imputed:
        got = (values[index_of[imputed["match_field"]]] or "").strip()
        if got == imputed["match_value"]:
            for name, value in imputed["values"].items():
                values[index_of[name]] = value
            stats["imputed"][anomaly["tag"]] += len(imputed["values"])
        else:
            # The signature matched but the contract does not, so the evidence
            # for the fill does not apply to this record. Left null and reported.
            stats["imputed_skipped"][f"{anomaly['tag']}:{got}"] += 1
    return values


def iter_chunks(path, spec, member, stats):
    """Yield {column: list} dicts from the delimited file, one chunk at a time.

    Rows are read positionally. A record whose delimiters are canonical takes
    the fast path; anything else is matched against the layout's declared
    anomaly signatures. An unrecognised signature is collected and fails the
    file rather than being dropped, which is the guard that was missing when 238
    FY2004 records disappeared.
    """
    fields = spec["fields"]
    n_fields = spec["field_count"]
    index_of = {f["name"]: f["index"] for f in fields}
    enc = spec.get("encoding", "latin-1")

    offsets = dc.delimited_offsets(spec, member)
    canon_spans = dc.spans_from_delimiters(offsets)
    is_canonical = canonical_matcher(offsets)
    anomalies = dc.load_anomalies(spec)

    rows, tags = [], []
    with open(path, "r", encoding=enc, newline="") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                stats["blank_lines"] += 1
                continue
            stats["records"] += 1
            tag = ""
            if is_canonical.match(line):
                parts = line.split("!")
                if len(parts) >= n_fields + 1:
                    values = parts[:n_fields]
                    # The piece after the final delimiter is not a field, and is
                    # normally empty. One FY2005 record carries two stray
                    # characters there, which is harmless precisely because the
                    # piece is discarded, but it is counted so that a change in
                    # how often it happens does not pass unseen.
                    if len(parts) > n_fields + 1 or parts[n_fields].strip():
                        stats["trailing_chars"] += 1
                else:
                    values = [line[a:b] for a, b in canon_spans]
            else:
                positions = [i for i, ch in enumerate(line) if ch == "!"]
                sig = dc.delimiter_signature(positions, offsets)
                anomaly = anomalies.get(sig)
                if anomaly is None:
                    stats["unknown"].append((lineno, sig))
                    continue
                tag = anomaly["tag"]
                stats["tags"][tag] += 1
                values = remap(line, anomaly, positions, n_fields, index_of, stats)
            rows.append(values)
            tags.append(tag)
            if len(rows) >= CHUNK_LINES:
                yield build(rows, tags, fields)
                rows, tags = [], []
    if rows:
        yield build(rows, tags, fields)


def build(rows, tags, fields):
    cols = {}
    for f in fields:
        i = f["index"]
        if f["type"] == "int":
            cols[f["name"]] = [parse_number(r[i]) for r in rows]
        else:
            cols[f["name"]] = [clean(r[i]) for r in rows]
    cols["is_lumpsum"] = [0] * len(rows)
    cols["layout_anomaly"] = list(tags)
    return cols


def fiscal_year_from_action_date(text):
    """Fiscal year implied by an eight-digit action date, October to September."""
    if not text or len(text) < 6 or not text[:6].isdigit():
        return None
    year, month = int(text[:4]), int(text[4:6])
    return str(year + 1 if month >= 10 else year)


def month_key_repairs(cols, spec, label):
    """Flag records whose reporting period is malformed, and repair the year.

    8 FY2006 records carry '073106', which is MMDDYY for 31 July 2006 rather
    than CCYYMM. They are real records with action dates inside the fiscal year,
    so the fiscal year is taken from action_date and the fiscal month is left
    null rather than the records being discarded or mis-dated.

    The test is on the FISCAL YEAR digits, not the month digits. '073106' is six
    digits whose last two, '06', are a valid month number, so a well-formedness
    test on the month passes it; what gives it away is that characters three and
    four say '31', which is a day. Any record whose fiscal-year digits disagree
    with the file's own label is therefore treated as malformed.
    """
    field = spec["fiscal_year_source"]["field"]
    a, b = spec["fiscal_year_source"]["slice"]
    expect = label[2:4]
    flags, implied = [], collections.Counter()
    for i, v in enumerate(cols[field]):
        ok = (v is not None and len(v) == 6 and v.isdigit()
              and 1 <= int(v[4:6]) <= 12 and v[a:b] == expect)
        flags.append(0 if ok else 1)
        if not ok:
            implied[fiscal_year_from_action_date(cols["action_date"][i])] += 1
    return flags, implied


def build_table(cols, spec, fiscal_year, source_name):
    n = len(cols["is_lumpsum"])
    arrays, names = [], []
    for f in spec["fields"]:
        arrays.append(pa.array(cols[f["name"]],
                               type=pa.int64() if f["type"] == "int" else pa.string()))
        names.append(f["name"])
    time = dc.derive_time_columns(spec, cols, fiscal_year, n)
    # A repaired reporting period yields no usable month: in '073106' the last
    # two digits are a day of the month, not a fiscal month, so reading them
    # would date the record to March. Null is the honest answer; the fiscal year
    # still comes through, from action_date.
    fiscal_month = [None if r else v
                    for v, r in zip(time["fiscal_month"], cols["month_key_repaired"])]
    stamps = ["" if r else v
              for v, r in zip(time["calendar_yearmonth"], cols["month_key_repaired"])]
    arrays.append(pa.array(fiscal_month, type=pa.int8()))
    names.append("fiscal_month")
    arrays.append(pa.array(stamps, type=pa.string()))
    names.append("calendar_yearmonth")
    arrays.append(pa.array(cols["month_key_repaired"], type=pa.int8()))
    names.append("month_key_repaired")
    arrays.append(pa.array(cols["layout_anomaly"], type=pa.string()))
    names.append("layout_anomaly")
    for name, value, typ in (("is_lumpsum", 0, pa.int8()),
                             ("fiscal_year", fiscal_year, pa.string()),
                             ("dollars_unit", spec["dollars_unit"], pa.string()),
                             ("source_file", source_name, pa.string()),
                             ("layout_id", spec["layout_id"], pa.string())):
        arrays.append(pa.array([value] * n, type=typ))
        names.append(name)
    return pa.table(arrays, names=names)


def fiscal_year_of(cols, spec):
    """Read the fiscal year off the records themselves.

    These filenames carry a calendar year rather than a fiscal one, so unlike the
    fixed-width eras the label has to come from the month key inside the records.
    Malformed keys are excluded from the vote so a handful of them cannot decide
    the label of a whole file.
    """
    a, b = spec["fiscal_year_source"]["slice"]
    field = spec["fiscal_year_source"]["field"]
    counts = collections.Counter()
    for v in cols[field]:
        if v is not None and len(v) == 6 and v.isdigit() and 1 <= int(v[4:6]) <= 12:
            counts[v[a:b]] += 1
    top = max(counts, key=counts.get)
    return f"20{top}" if int(top) < 50 else f"19{top}", counts


def main():
    layouts, index = dc.load_layouts(include_delimited=True)
    delimited = {k: v for k, v in layouts.items() if v.get("format") == "delimited"}
    if not delimited:
        print("FAIL: no delimited layouts found", file=sys.stderr)
        return 1
    zips = {}
    for zp in sorted(dc.SOURCE_DIR.rglob("*.zip")):
        for spec in delimited.values():
            for member in spec["applies_to"]:
                if zp.stem == Path(member).stem:
                    zips[member] = (zp, spec)

    parsed, failed, missing = [], [], []
    for spec in delimited.values():
        for member in spec["applies_to"]:
            if member not in zips:
                missing.append(member)
                print(f"MISSING {member:<24} no zip found", flush=True)
                continue
            zp, _ = zips[member]
            extracted = None
            writer = dest = tmp = None
            stats = {"records": 0, "blank_lines": 0, "trailing_chars": 0,
                     "tags": collections.Counter(), "unknown": [],
                     "imputed": collections.Counter(),
                     "imputed_skipped": collections.Counter()}
            try:
                path, _ = dc.extract_member(zp, member)
                extracted = path
                rows = dollars = 0
                label = None
                repaired = 0
                implied_total = collections.Counter()
                for cols in iter_chunks(path, spec, member, stats):
                    if label is None:
                        label, _ = fiscal_year_of(cols, spec)
                        dest = dc.OUT_DIR / f"fy{label}.parquet"
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        tmp = dest.with_suffix(".parquet.tmp")
                    flags, implied = month_key_repairs(cols, spec, label)
                    cols["month_key_repaired"] = flags
                    repaired += sum(flags)
                    implied_total.update(implied)
                    table = build_table(cols, spec, label, member)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp, table.schema, compression="snappy")
                    writer.write_table(table)
                    rows += table.num_rows
                    dollars += sum(v for v in cols[spec["dollars_field"]] if v is not None)
                if writer is None:
                    raise ValueError("no records read")
                # A handful of malformed reporting periods is a source defect. A
                # large share of them would mean the file label is wrong, which
                # is a different and much worse problem, so it stops the run
                # rather than being quietly repaired away.
                if repaired and repaired / rows > 0.001:
                    raise ValueError(
                        f"{repaired:,} of {rows:,} records ({100 * repaired / rows:.2f}%) "
                        f"carry a fiscal year other than the file label {label}; the label "
                        f"is suspect, not the records"
                    )
                stray = {k: v for k, v in implied_total.items() if k != label}
                if stray:
                    raise ValueError(
                        f"repaired records imply fiscal years other than {label}: {stray}"
                    )
                if stats["unknown"]:
                    raise ValueError(
                        f"{len(stats['unknown'])} records carry a delimiter layout no "
                        f"anomaly in the layout declares; first at line "
                        f"{stats['unknown'][0][0]} signature {stats['unknown'][0][1]}"
                    )
                writer.close()
                tmp.rename(dest)
            except (ValueError, OSError, UnicodeDecodeError) as exc:
                failed.append((member, str(exc)))
                print(f"FAIL {member}: {exc}", flush=True)
                if writer is not None:
                    writer.close()
                continue
            finally:
                if extracted is not None and extracted.exists():
                    extracted.unlink()
            parsed.append((member, label, rows, dollars, stats, repaired))
            print(f"ok   fy{label:<7} {member:<24} {rows:>9,} rows  "
                  f"${dollars / 1e9:>7,.1f} bn", flush=True)
            if stats["blank_lines"]:
                print(f"       blank lines skipped: {stats['blank_lines']}", flush=True)
            if stats["trailing_chars"]:
                print(f"       records with stray characters after the final "
                      f"delimiter, discarded: {stats['trailing_chars']}", flush=True)
            if stats["tags"]:
                total = sum(stats["tags"].values())
                print(f"       delimiter anomalies remapped: {total} "
                      f"({', '.join(f'{t} {n}' for t, n in sorted(stats['tags'].items()))})",
                      flush=True)
            if stats["imputed"]:
                total = sum(stats["imputed"].values())
                print(f"       field values recovered from the rest of the contract: {total} "
                      f"({', '.join(f'{t} {n}' for t, n in sorted(stats['imputed'].items()))})",
                      flush=True)
            if stats["imputed_skipped"]:
                print(f"       NOT recovered, contract does not match the evidence: "
                      f"{dict(stats['imputed_skipped'])}", flush=True)
            if repaired:
                print(f"       month_key malformed and repaired from action_date: "
                      f"{repaired} (implied fiscal years "
                      f"{dict(implied_total)})", flush=True)

    print(f"\nparsed {len(parsed)} files, failed {len(failed)}, missing {len(missing)}")
    if missing:
        print(f"\nMISSING: {len(missing)} declared members absent", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
    if failed:
        print(f"\nFAIL: {len(failed)} files did not parse", file=sys.stderr)
        for m, why in failed:
            print(f"  {m}: {why}", file=sys.stderr)
    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
