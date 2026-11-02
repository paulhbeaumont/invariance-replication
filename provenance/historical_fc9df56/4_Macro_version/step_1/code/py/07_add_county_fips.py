#!/usr/bin/env python3
"""Give every fiscal year a county FIPS code, and write it into the parquet.

Place of performance is coded three incompatible ways across the panel, so
nothing downstream can key on one county identifier until they are reconciled:

  FY1966-83   ZARK, the Pentagon's own pre-FIPS scheme. Resolved exactly against
              RG330.DEFCON.ZALF, whose FIPS column this is the first code to
              read, plus zark_fips_supplement.csv for what ZALF cannot say.
  FY1984-2003 already a 5-digit SSCCC FIPS county. Copied.
  FY2004-06   state and county held separately. Concatenated.

Four columns are appended to all 42 files so every year has the same schema:

  county_fips5   5-digit contemporaneous county FIPS, or empty
  state_fips2    2-digit FIPS state, or empty. Filled for a pre-FY1989 territory
                 even where the county is not, so the state identity survives
  geo_class      county | territory | associated | undistributed | foreign |
                 lumpsum | unplaceable. See dcads_common.GEO_CLASSES
  fips_source    native | concat | zalf | supplement | territory_recode |
                 manual_override | none

CONTEMPORANEOUS, not harmonised: each row gets the code that was correct in its
own fiscal year, so fiscal_year is the vintage and no separate vintage column is
needed. Ormsby stays distinct from Carson City while the source distinguishes
them, Nansemond County keeps 51123, and FY1983 carries both of Alaska's schemes
because the source does. Mapping counties onto one time-consistent set is Part
1.4's job, where the employment panel forces the choice.

TERRITORIES follow the FY1989-2006 convention, which is what 18 of the 23
post-1984 years use and the only one whose codes are genuine FIPS: numeric 60,
64, 66, 68, 69, 70, 72, 74, 78. The FY1984-88 alpha codes are recoded to it
(RQ005 -> 72005, GQ -> 66), and so is ZARK PR -> 72.

The county panel is geo_class == 'county': the 50 states and DC with a county
resolved. That replaces the compound filter of README 1.5 and closes the FY1984-85
zero-fill trap, where 3,640 and 3,744 lump-sum rows carrying about 31% of dollars
write county 00000 rather than a blank.

Rerunnable and idempotent: the four columns are dropped and rebuilt, each file is
written to .tmp and renamed, so an interrupted run leaves every file either
wholly old or wholly new.

Output: rewrites $temp/dcads_year/fy<year>.parquet in place
"""

import argparse
import collections
import csv
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc

BATCH = 200_000
NEW_COLUMNS = ("county_fips5", "state_fips2", "geo_class", "fips_source")

FIELD_DOC = {
    "county_fips5": "Contemporaneous 5-digit county FIPS (SSCCC) for the row's own "
                    "fiscal year, or empty where none resolves. Not harmonised to a "
                    "single vintage.",
    "state_fips2": "2-digit FIPS state. Equals county_fips5[:2] whenever both are "
                   "present, and is filled for a pre-FY1989 territory even where the "
                   "county is not.",
    "geo_class": "county (50 states + DC, county known - the county panel) | "
                 "territory (60, 66, 69, 72, 78; county may be empty) | "
                 "associated (64, 68, 70, 74; FIPS-coded, not US counties) | "
                 "undistributed (state 98) | foreign | lumpsum | "
                 "unplaceable (domestic, no county resolves).",
    "fips_source": "native (FY1984-2003, copied) | concat (FY2004-06) | zalf | "
                   "supplement (zark_fips_supplement.csv) | territory_recode | "
                   "manual_override (dcads_common.NATIVE_OVERRIDES or "
                   "the guarded DCADS city-county override tables) | none.",
}

ZARK_LAYOUT = "dcads_272"
DELIM_LAYOUT = "dcads_delim_2004"



def era_of(layout_id):
    """Which of the three geography encodings a layout uses."""
    if layout_id == ZARK_LAYOUT:
        return "zark"
    if layout_id == DELIM_LAYOUT:
        return "concat"
    return "native"


def well_formed_county(era, county_raw):
    """True when a county field is the shape its era declares.

    Separates the two ways a county can fail to resolve. A well-formed code that
    maps to nothing is a hole in the crosswalk and has to be fixed. A code of the
    wrong shape is the source being malformed, cannot be resolved by any
    crosswalk, and is reported rather than fixed - FY2003 has 24 rows whose
    county field holds the two-character state code and nothing else.
    """
    width = 5 if era == "native" else 3
    return (len(county_raw) == width and county_raw.isdigit()
            and county_raw[-3:] != dc.UNDISTRIBUTED_COUNTY)


def _valid_county(fips5):
    """True for a 5-digit code that names a county rather than a state's catch-all.

    A county part of 999 is the source saying "this state, county unknown", one
    level down from state 98. ZALF carries a 999 record for every state, so
    without this the catch-all would resolve to a county that does not exist -
    882 ZARK-era rows and $3.5bn of it.
    """
    return (len(fips5) == 5 and fips5.isdigit()
            and fips5[2:5] != dc.UNDISTRIBUTED_COUNTY)


def _real_state(code):
    """True for a 2-digit FIPS state, territory or associated area.

    Excludes 98, which the source uses for "somewhere in the United States" and
    writes as a state code and as the county 98999. It is not a state and must
    not reach state_fips2, or a state panel would gain a 51st entry.
    """
    return code in dc.STATE_FIPS or code in dc.TERRITORY_FIPS or code in dc.ASSOCIATED_FIPS


def resolve_zark(state_raw, county_raw, name, fiscal_year, lookup, zalf_states):
    """(county_fips5, state_fips2, fips_source) for one FY1966-83 record."""
    if state_raw == dc.UNDISTRIBUTED_STATE:
        return "", "", "none"
    fips5, source = lookup(state_raw, county_raw, fiscal_year, name)
    if fips5 and _valid_county(fips5):
        return fips5, fips5[:2], source
    # No county, but the state may still be known: Puerto Rico by rule, everything
    # else from ZALF's own state table.
    if state_raw == "PR":
        return "", dc.ALPHA_TERRITORY["RQ"], "territory_recode"
    if fips5 and _real_state(fips5[:2]):         # a 999 catch-all, state still good
        return "", fips5[:2], source
    state_fips = zalf_states.get(state_raw, "")
    if _real_state(state_fips):
        return "", state_fips, "zalf"
    return "", "", "none"


def resolve_native(state_raw, county_raw):
    """(county_fips5, state_fips2, fips_source) for one FY1984-2003 record."""
    if len(county_raw) == 5 and county_raw.isdigit():
        if not _real_state(county_raw[:2]):
            return "", "", "none"
        return (county_raw if _valid_county(county_raw) else ""), county_raw[:2], "native"
    territory = dc.ALPHA_TERRITORY.get(state_raw)
    if territory:
        # FY1984-88 writes Puerto Rico as RQ plus the true 3-digit FIPS municipio.
        if len(county_raw) == 5 and county_raw[:2] == state_raw and county_raw[2:5].isdigit():
            county = territory + county_raw[2:5]
            return (county if _valid_county(county) else ""), territory, "territory_recode"
        return "", territory, "territory_recode"
    if _real_state(state_raw):
        return "", state_raw, "native"
    return "", "", "none"


def resolve_concat(state_raw, county_raw):
    """(county_fips5, state_fips2, fips_source) for one FY2004-06 record."""
    if _real_state(state_raw):
        if len(county_raw) == 3 and county_raw.isdigit():
            county = state_raw + county_raw
            return (county if _valid_county(county) else ""), state_raw, "concat"
        return "", state_raw, "concat"
    territory = dc.ALPHA_TERRITORY.get(state_raw)
    if territory:
        return "", territory, "territory_recode"
    return "", "", "none"


def add_columns(src, lookup, zalf_states):
    """Rewrite one parquet with the four geography columns, returning a summary."""
    pf = pq.ParquetFile(src)
    names = pf.schema_arrow.names
    if "layout_id" not in names or "fiscal_year" not in names:
        raise ValueError(f"{src.name}: no layout_id/fiscal_year column")
    era = era_of(pq.read_table(src, columns=["layout_id"]).column(0)[0].as_py())
    county_field = "county_fips" if era == "concat" else "county_code"
    for need in ("state_code", county_field, "is_lumpsum", "county_name",
                 "city_name", "contract_no"):
        if need not in names:
            raise ValueError(f"{src.name}: no {need} column")

    keep = [n for n in names if n not in NEW_COLUMNS]
    tmp = src.with_suffix(".parquet.tmp")
    cache = {}
    summary = {"era": era, "rows": 0,
               "class_rows": collections.Counter(), "class_dollars": collections.Counter(),
               "source_rows": collections.Counter(), "unresolved": collections.Counter(),
               "malformed": collections.Counter(), "county_unknown": collections.Counter(),
               "fips": set(), "override_rows": collections.Counter(),
               "city_override_rows": collections.Counter(),
               "contract_override_rows": collections.Counter()}

    # The overrides for this file, keyed the way the resolver caches. The ZARK
    # era has its own supplement and a three-part cache key, so it is not served
    # from here.
    file_year = src.stem[2:]
    overrides = dc.overrides_for(file_year, era)
    city_overrides = dc.city_county_overrides_for(file_year, era)
    contract_overrides = dc.contract_county_overrides_for(file_year, era)
    writer = None
    try:
        for batch in pf.iter_batches(batch_size=BATCH, columns=keep):
            table = pa.Table.from_batches([batch])
            n = table.num_rows
            states = table.column("state_code").to_pylist()
            counties = table.column(county_field).to_pylist()
            lumps = table.column("is_lumpsum").to_pylist()
            cnames = table.column("county_name").to_pylist()
            cities = table.column("city_name").to_pylist()
            contracts = table.column("contract_no").to_pylist()
            years = table.column("fiscal_year").to_pylist()
            dollars = (table.column("dollars").to_pylist()
                       if "dollars" in keep else [0] * n)

            c5, s2, gclass, gsource = [], [], [], []
            for i in range(n):
                if lumps[i]:
                    c5.append("")
                    s2.append("")
                    gclass.append("lumpsum")
                    gsource.append("none")
                    summary["class_rows"]["lumpsum"] += 1
                    summary["class_dollars"]["lumpsum"] += dollars[i] or 0
                    summary["source_rows"]["none"] += 1
                    continue
                sr = (states[i] or "").strip()
                cr = (counties[i] or "").strip()
                nm = (cnames[i] or "").strip()
                key = (sr, cr, nm) if era == "zark" else (sr, cr)
                city_key = (sr, cr, dc.normalize_city_name(cities[i] or ""))
                contract = dc.normalize_contract_id(contracts[i] or "")
                contract_key = city_key + (contract,)
                if key in overrides:
                    summary["override_rows"][(file_year,) + key] += 1
                contract_spec = contract_overrides.get(contract_key)
                city_spec = city_overrides.get(city_key)
                if contract_spec is not None:
                    reported_fips = cr if era == "native" else sr + cr
                    guard_key = (
                        file_year, sr, reported_fips, city_key[2], contract)
                    summary["contract_override_rows"][guard_key] += 1
                    hit = contract_spec[:3]
                elif city_spec is not None:
                    reported_fips = cr if era == "native" else sr + cr
                    guard_key = (file_year, sr, reported_fips, city_key[2])
                    summary["city_override_rows"][guard_key] += 1
                    hit = city_spec[:3]
                else:
                    hit = cache.get(key)
                    if hit is None:
                        if key in overrides:
                            hit = overrides[key][:3]
                        elif era == "zark":
                            hit = resolve_zark(sr, cr, nm, years[i], lookup, zalf_states)
                        elif era == "concat":
                            hit = resolve_concat(sr, cr)
                        else:
                            hit = resolve_native(sr, cr)
                        cache[key] = hit
                county, state, source = hit
                klass = dc.classify_geo(state, county, 0, sr)
                c5.append(county)
                s2.append(state)
                gclass.append(klass)
                gsource.append(source)
                summary["class_rows"][klass] += 1
                summary["class_dollars"][klass] += dollars[i] or 0
                summary["source_rows"][source] += 1
                if county:
                    summary["fips"].add(county)
                elif (klass == "unplaceable" and cr and not dc.is_blank_code(cr)
                      and key not in overrides and city_spec is None):
                    # A 50-states row carrying a county code that produced no FIPS,
                    # in one of three kinds. 999 is the source saying the county is
                    # unknown; a well-formed code that maps to nothing is a hole in
                    # the crosswalk and has to be zero; anything else is the source
                    # being malformed and cannot be fixed by any crosswalk.
                    #
                    # An override is none of the three. Its county code is well
                    # formed and deliberately dropped, so counting it as
                    # unresolved would report a hole in the crosswalk that is
                    # really a decision, and would fail the run for it. The
                    # override guard is what holds it accountable instead.
                    if cr[-3:] == dc.UNDISTRIBUTED_COUNTY:
                        bucket = "county_unknown"
                    elif well_formed_county(era, cr):
                        bucket = "unresolved"
                    else:
                        bucket = "malformed"
                    summary[bucket][(sr, cr, nm)] += 1

            for name, values in (("county_fips5", c5), ("state_fips2", s2),
                                 ("geo_class", gclass), ("fips_source", gsource)):
                field = pa.field(name, pa.string(),
                                 metadata={"description": FIELD_DOC[name]})
                table = table.append_column(field, pa.array(values, type=pa.string()))
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="snappy")
            writer.write_table(table)
            summary["rows"] += n
    except BaseException:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(tmp, src)
    summary["resolved"] = cache
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-crosswalk", action="store_true",
                    help="also write the resolved ZARK table to $temp for inspection")
    ap.add_argument("--years", nargs="+", metavar="FY",
                    help="rewrite only these fiscal-year labels, for example 2001 2002")
    args = ap.parse_args()

    lookup, stats = dc.load_zark_crosswalk()
    zalf_states = stats["state_fips"]
    print(f"zalf        {dc.ZALF}")
    print(f"supplement  {dc.ZARK_SUPPLEMENT}")
    print(f"output      {dc.OUT_DIR}\n")
    print(f"ZALF: {stats['zalf_states']} state records, {stats['zalf_counties']} usable "
          f"county records, {len(stats['zalf_dropped'])} malformed records dropped")
    for r in stats["zalf_dropped"]:
        print(f"    dropped {r.rstrip()!r}")
    print(f"supplement: {stats['supplement_rows']} rows\n", flush=True)

    files = sorted(dc.OUT_DIR.glob("fy*.parquet"),
                   key=lambda p: (dc.zark_year(p.stem[2:]), p.stem))
    if not files:
        print(f"FAIL: no parquet in {dc.OUT_DIR}", file=sys.stderr)
        return 1
    if args.years:
        requested = set(args.years)
        available = {path.stem[2:] for path in files}
        missing = sorted(requested - available)
        if missing:
            print(f"FAIL: requested fiscal years are missing: {missing}", file=sys.stderr)
            return 1
        files = [path for path in files if path.stem[2:] in requested]

    print(f"{'fy':<9}{'era':<8}{'rows':>10}{'county':>10}{'terr':>7}{'assoc':>7}"
          f"{'undist':>8}{'foreign':>9}{'lump':>8}{'unplac':>8}{'unres':>7}")
    summaries, failed = {}, []
    for path in files:
        label = path.stem[2:]
        try:
            s = add_columns(path, lookup, zalf_states)
        except (ValueError, OSError, pa.ArrowInvalid) as exc:
            failed.append((label, str(exc)))
            print(f"FAIL fy{label}: {exc}", file=sys.stderr, flush=True)
            continue
        summaries[label] = s
        c = s["class_rows"]
        unres = sum(s["unresolved"].values())
        print(f"{label:<9}{s['era']:<8}{s['rows']:>10,}{c['county']:>10,}"
              f"{c['territory']:>7,}{c['associated']:>7,}{c['undistributed']:>8,}"
              f"{c['foreign']:>9,}{c['lumpsum']:>8,}{c['unplaceable']:>8,}"
              f"{unres:>7,}", flush=True)

    print(f"\n{'=' * 96}")
    total_unres, total_malformed = collections.Counter(), collections.Counter()
    total_unknown = collections.Counter()
    for s in summaries.values():
        total_unres.update(s["unresolved"])
        total_malformed.update(s["malformed"])
        total_unknown.update(s["county_unknown"])
    if total_unres:
        print(f"UNRESOLVED: {sum(total_unres.values()):,} rows in the 50 states or DC "
              f"carry a well-formed county code that the crosswalk does not cover")
        for (sr, cr, nm), k in total_unres.most_common(40):
            print(f"    {sr}/{cr} {nm!r}  {k:,} rows")
    else:
        print("every well-formed county code in the 50 states and DC resolved to a "
              "FIPS county")
    if total_unknown:
        print(f"\nthe source's own county-unknown code, 999, on a known state: "
              f"{sum(total_unknown.values()):,} rows across "
              f"{len({k[0] for k in total_unknown})} states. State 98 one level down; "
              f"the state is kept in state_fips2 and the row is unplaceable.")
    if total_malformed:
        print(f"\nmalformed in the source, not resolvable by any crosswalk, left "
              f"unplaceable: {sum(total_malformed.values()):,} rows")
        for (sr, cr, nm), k in total_malformed.most_common(20):
            print(f"    state {sr}, county {cr!r} {nm!r}  {k:,} rows")

    # The override guard. Each entry declares how many rows it should hit; if it
    # hits a different number the parse has moved under it and the run fails,
    # because an override that quietly matches nothing is worse than no override.
    matched = collections.Counter()
    for s in summaries.values():
        matched.update(s["override_rows"])
    active_years = set(summaries)
    native_specs = {key: spec for key, spec in dc.NATIVE_OVERRIDES.items()
                    if key[0] in active_years}
    print(f"\noverrides: {len(native_specs)} record-level hand-authored, outside the "
          "ZARK era")
    override_fail = []
    for key, spec in sorted(native_specs.items()):
        county, state, source, expected, why = spec
        got = matched.get(key, 0)
        ok = "ok" if got == expected else "FAIL"
        print(f"    FY{key[0]} state {key[1]} county {key[2]} -> county_fips5 "
              f"{county or '(none)'}, state_fips2 {state}, {source}")
        print(f"      {got} row(s) matched, {expected} expected: {ok}")
        print(f"      {why}")
        if got != expected:
            override_fail.append(
                f"override {key} matched {got} rows, expected {expected}")

    matched_city = collections.Counter()
    for s in summaries.values():
        matched_city.update(s["city_override_rows"])
    city_specs = {
        key: spec for source in (
            dc.DCADS_CITY_COUNTY_OVERRIDES,
            dc.DCADS_VERIFIED_CITY_COUNTY_OVERRIDES,
        ) for key, spec in source.items()
        if key[0] in active_years
    }
    print(f"\ncity-county overrides: {len(city_specs)} guarded year-city mappings")
    for key, spec in sorted(city_specs.items()):
        county, state, source, expected, why = spec
        got = matched_city.get(key, 0)
        ok = "ok" if got == expected else "FAIL"
        print(f"    FY{key[0]} state {key[1]} city {key[3]!r}, reported county "
              f"{key[2]} -> county_fips5 {county}, {source}")
        print(f"      {got} row(s) matched, {expected} expected: {ok}")
        print(f"      {why}")
        if got != expected:
            override_fail.append(
                f"city-county override {key} matched {got} rows, expected {expected}")

    matched_contract = collections.Counter()
    for s in summaries.values():
        matched_contract.update(s["contract_override_rows"])
    contract_specs = {
        key: spec
        for key, spec in dc.DCADS_VERIFIED_CONTRACT_COUNTY_OVERRIDES.items()
        if key[0] in active_years
    }
    print(
        f"\ncontract-county overrides: {len(contract_specs)} guarded mappings")
    for key, spec in sorted(contract_specs.items()):
        county, state, source, expected, why = spec
        got = matched_contract.get(key, 0)
        ok = "ok" if got == expected else "FAIL"
        print(
            f"    FY{key[0]} state {key[1]} city {key[3]!r}, contract "
            f"{key[4]}, reported county {key[2]} -> county_fips5 {county}, "
            f"{source}")
        print(f"      {got} row(s) matched, {expected} expected: {ok}")
        print(f"      {why}")
        if got != expected:
            override_fail.append(
                f"contract-county override {key} matched {got} rows, "
                f"expected {expected}")

    unused = stats["supplement_seen"] - stats["supplement_used"]
    print(f"\nsupplement: {len(stats['supplement_used'])} of {stats['supplement_rows']} "
          f"rows matched a record in the data")
    if unused:
        print("  rows that matched nothing - harmless, but a typo would look like this:")
        for key, name, span in sorted(unused)[:40]:
            print(f"    {key[0]}/{key[1]} {name or '(any name)':<14} FY{span[0]}-{span[1]}")

    fips = set()
    for s in summaries.values():
        fips |= s["fips"]
    print(f"\n{len(fips):,} distinct county_fips5 across selected years")

    if args.dump_crosswalk:
        # A debugging aid written to $temp, which is gitignored. The checked-in
        # artifact is the supplement, not this: everything here is reproducible
        # from ZALF in seconds.
        dump = dc.OUT_DIR / "zark_fips_resolved.csv"
        rows = set()
        for label, s in summaries.items():
            if s["era"] == "zark":
                for (sr, cr, nm), (county, state, source) in s["resolved"].items():
                    rows.add((label, sr, cr, nm, county, state, source))
        with open(dump, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["fiscal_year", "zark_state", "zark_county", "county_name",
                        "county_fips5", "state_fips2", "fips_source"])
            w.writerows(sorted(rows, key=lambda r: (r[1], r[2], r[0])))
        print(f"crosswalk dump: {dump}  ({len(rows):,} rows)")

    print(f"\nrewrote {len(summaries)} of {len(files)} fiscal years")
    if failed:
        print(f"\nFAIL: {len(failed)} files did not rewrite", file=sys.stderr)
        for label, why in failed:
            print(f"  fy{label}: {why}", file=sys.stderr)
    if override_fail:
        print(f"\nFAIL: {len(override_fail)} override(s) did not match as declared",
              file=sys.stderr)
        for why in override_fail:
            print(f"  {why}", file=sys.stderr)
    return 1 if (failed or total_unres or override_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
