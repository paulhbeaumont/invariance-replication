#!/usr/bin/env python3
"""Parse fixed-width DCADS and MPCF contract files into one parquet per fiscal year.

Layouts live in code/layouts/*.json, one per era, each declaring a record length
and the byte offsets of every field. A file is parsed only when its name appears
in some layout's applies_to list, so an unrecognised file is reported rather than
parsed against a guessed layout. Selection is keyed to the filename and never to
the record length, because FY1989 and FY1990-95 share a length of 464 while
position 408 means different things in each.

Every field the layout declares is written, plus five derived columns:
fiscal_year, is_lumpsum, dollars_unit, source_file and layout_id.
In the 272-byte era, approved report cancellations are neutralized in dollars;
dollars_reported preserves the source amount and report_correction_neutralized
marks the affected rows. Printed parse totals refer to reported source amounts.

The later fiscal years ship only as zips, so a needed member is extracted to
scratch just before it is parsed and deleted afterwards; the uncompressed text
never lands in the repo or in Dropbox.

Output: $temp/dcads_year/fy<year>.parquet
"""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc


def zipped_sources(index):
    """Map filename -> zip path, for layout-claimed files that only exist zipped.

    A zip is considered only when the member it is expected to contain is named
    in some layout, so unclaimed archives are left alone.
    """
    found = {}
    for zp in sorted(dc.SOURCE_DIR.rglob("*.zip")):
        member = zp.name[:-4]
        if member in index:
            found[member] = zp
    return found


def build_table(cols, spec, fiscal_year, source_name):
    """Turn one chunk of parsed columns into an arrow table with the derived columns."""
    n = len(cols["is_lumpsum"])
    arrays, names = [], []
    for f in spec["fields"]:
        v = cols[f["name"]]
        arrays.append(pa.array(v, type=pa.int64() if f["type"] == "int" else pa.string()))
        names.append(f["name"])
    # The reporting period is a fiscal month in every era and the fiscal year
    # itself ran July to June through FY1976, so the calendar month is derived
    # once here rather than left to each consumer to get wrong. See
    # dcads_common.derive_time_columns.
    time = dc.derive_time_columns(spec, cols, fiscal_year, n)
    arrays.append(pa.array(time["fiscal_month"], type=pa.int8()))
    names.append("fiscal_month")
    arrays.append(pa.array(time["calendar_yearmonth"], type=pa.string()))
    names.append("calendar_yearmonth")
    arrays.append(pa.array(cols["is_lumpsum"], type=pa.int8()))
    names.append("is_lumpsum")
    arrays.append(pa.array([fiscal_year] * n, type=pa.string()))
    names.append("fiscal_year")
    arrays.append(pa.array([spec["dollars_unit"]] * n, type=pa.string()))
    names.append("dollars_unit")
    arrays.append(pa.array([source_name] * n, type=pa.string()))
    names.append("source_file")
    arrays.append(pa.array([spec["layout_id"]] * n, type=pa.string()))
    names.append("layout_id")
    return pa.table(arrays, names=names)


def parse_to_parquet(path, spec, dest, fiscal_year, source_name):
    """Stream one fixed-width file into parquet, returning summary counts.

    Written chunk by chunk rather than concatenated at the end, because the later
    fiscal years reach 303 MB uncompressed and holding every chunk in memory
    before a single concat does not scale.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    dollars_field = spec["dollars_field"]
    rows = lump_rows = unparsed = 0
    dollars = lump_dollars = 0
    writer = None
    try:
        for cols in dc.iter_chunks(path, spec):
            table = build_table(cols, spec, fiscal_year, source_name)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="snappy")
            writer.write_table(table)
            rows += table.num_rows
            for value, lump in zip(cols[dollars_field], cols["is_lumpsum"]):
                lump_rows += lump
                if value is None:
                    unparsed += 1
                    continue
                dollars += value
                if lump:
                    lump_dollars += value
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"{path.name}: no records read")
    if spec['layout_id'] == 'dcads_272':
        import report_corrections
        report_corrections.clean_parquet(tmp, source_name, path)
    tmp.rename(dest)
    return {"rows": rows, "lump_rows": lump_rows, "unparsed": unparsed,
            "dollars": dollars, "lump_dollars": lump_dollars}


def main():
    layouts, index = dc.load_layouts()
    if not layouts:
        print(f"FAIL: no layouts in {dc.LAYOUT_DIR}", file=sys.stderr)
        return 1
    print(f"source  {dc.SOURCE_DIR}")
    print(f"output  {dc.OUT_DIR}")
    print(f"layouts {', '.join(sorted(layouts))}\n", flush=True)

    on_disk = {p.name: p for p in dc.data_files()}
    in_zips = zipped_sources(index)
    claimed = sorted(index, key=lambda n: (dc.fiscal_year_of(n) or "", n))

    parsed, failed, missing = [], [], []
    for name in claimed:
        spec = layouts[index[name]]
        fiscal_year = dc.fiscal_year_of(name)
        if fiscal_year is None:
            failed.append((name, "no fiscal year in filename"))
            print(f"FAIL {name}: no fiscal year in filename", flush=True)
            continue

        extracted = None
        path = on_disk.get(name)
        if path is None:
            zp = in_zips.get(name)
            if zp is None:
                missing.append(name)
                print(f"MISSING {name:<22} not on disk and not in any zip", flush=True)
                continue
            try:
                path, _ = dc.extract_member(zp, name)
                extracted = path
            except (ValueError, OSError) as exc:
                failed.append((name, f"unzip: {exc}"))
                print(f"FAIL {name}: unzip: {exc}", flush=True)
                continue

        dest = dc.OUT_DIR / f"fy{fiscal_year}.parquet"
        try:
            stats = parse_to_parquet(path, spec, dest, fiscal_year, name)
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            failed.append((name, str(exc)))
            print(f"FAIL {name}: {exc}", flush=True)
            continue
        finally:
            if extracted is not None and extracted.exists():
                extracted.unlink()

        scale = 1e6 if spec["dollars_unit"] == "thousands" else 1e9
        share = 100 * stats["lump_dollars"] / stats["dollars"] if stats["dollars"] else 0
        flag = f"  UNPARSED {stats['unparsed']:,}" if stats["unparsed"] else ""
        parsed.append((name, fiscal_year, stats))
        print(f"ok   fy{fiscal_year:<7} {name:<22} {stats['rows']:>9,} rows  "
              f"${stats['dollars'] / scale:>7,.1f} bn  "
              f"lumpsum {stats['lump_rows']:>6,} rows {share:>5.1f}% of $"
              f"{flag}", flush=True)

    unclaimed = [p for n, p in sorted(on_disk.items()) if n not in index]
    print(f"\nparsed {len(parsed)} files, failed {len(failed)}, missing {len(missing)}, "
          f"no layout {len(unclaimed)}")
    if unclaimed:
        print("\nno layout declared for these files, not parsed:")
        for p in unclaimed:
            print(f"  {p.relative_to(dc.SOURCE_DIR)}")
    if missing:
        print(f"\nMISSING: {len(missing)} layout-claimed files absent", file=sys.stderr)
        for n in missing:
            print(f"  {n}", file=sys.stderr)
    if failed:
        print(f"\nFAIL: {len(failed)} files did not parse", file=sys.stderr)
        for n, why in failed:
            print(f"  {n}: {why}", file=sys.stderr)
    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
