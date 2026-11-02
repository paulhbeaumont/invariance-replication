"""Stage 7 (construction): the Census NAICS-to-NAICS vintage chain.

Every industry source in this project carries a different NAICS vintage, and a
cell built from one cannot be compared with a cell built from another until
they are put on a single classification. This producer builds that
classification and the maps onto it.

    the FSC x NAICS6 concordance   NAICS 2002 with a NAICS 1997 residual
                                   (estimated on DCADS 2001-06 records)
    USAspending native naics_code  the fiscal year's own vintage; the 2002,
                                   2007, 2012 and 2017 revisions all fall
                                   inside FY2007-18
    EFSY employment cells          NAICS 2012 for 1975-2016, NAICS 2017 for
                                   2017-18 (established below, and asserted)

THE TARGET IS THE VINTAGE EFSY CARRIES, because EFSY is the outcome side and
the outcome is not ours to restate. EFSY carries two vintages, not one, so the
target is not a vintage but a vintage-STABLE grouping: the 4-digit codes of
NAICS 2012, with each group of codes that the 2012->2017 revision rearranges
collapsed into a single cell. The revision touches exactly two groups and both
are closed - no mapping leaks in or out of either - so the collapse is lossless
and reversible:

    4521 + 4529  (NAICS2012)  ==  4522 + 4523  (NAICS2017)   ->  452X
    5171 + 5172  (NAICS2012)  ==  5173         (NAICS2017)   ->  517X

287 cells. Every other 4-digit code is identical across the two vintages. The
alternative - converting EFSY 2017-18 back to NAICS 2012 - would edit an
outcome variable with fractional weights to buy nothing, and is not taken. The
groups are DERIVED here from the Census file as connected components, not
hard-coded; the two above are then asserted, so a wrong file or a changed file
fails loud instead of quietly redefining the axis.

WHY THE CHAIN IS COMPOSED AT SIX DIGITS AND COLLAPSED TO FOUR ONLY AT THE END.
NAICS reuses codes. `5173` is Telecommunications Resellers in NAICS 2002 and
2007, which NAICS 2012 folds into `5179`; NAICS 2017 then reuses `5173` for
Wired and Wireless Telecommunications Carriers, a different industry. A test of
the form "is this 4-digit code in the NAICS 2012 list?" passes a 2002 `5173`
through untouched and books reseller dollars as carrier dollars. Only a chained
conversion that keeps the source vintage attached catches it. Collapsing to 4
digits before composing would destroy the information that resolves it.

Inputs (downloaded, cached under dc.CENSUS_NAICS_SOURCE_DIR, sha256 recorded):
    1997_NAICS_to_2002_NAICS.xls   sheet "Concordance 23 US NoD" - sheet 1 of
                                   that workbook is a PIVOT with subtotal rows
                                   and reversed columns, and is not the
                                   concordance
    2002_to_2007_NAICS.xls
    2007_to_2012_NAICS.xls
    2012_to_2017_NAICS.xlsx

Outputs, all tracked reference files in step_1/code/reference/:
    naics_1997_2002_census.csv   from, to, weight (one row per matched pair)
    naics_2002_2007_census.csv
    naics_2007_2012_census.csv
    naics_2012_2017_census.csv
    efsy_naics4_axis.csv         vintage, naics4, axis4  (2012 and 2017)
    naics_to_efsy_axis.csv       vintage, naics6, axis4, weight  (1997, 2002,
                                 2007, 2012, 2017)
and Results/pipeline_diagnostics/naics_vintage_drift.csv.

The Census tables are unweighted - they say that a code matched another code,
not in what proportion - so `weight` is an equal split across a source code's
matches. That is the same convention reference/sic87_naics97_census.csv already
records in its own header, and it is stated in every header written here rather
than left to be discovered. The DOLLAR exposure of that assumption cannot be
computed here, because no dollars exist at this stage; producer 08 reports it
against the FSC concordance's dollars, which is where it belongs.

Modes, following 06:
  --verify (default)  recompute every table and require it byte-identical to
                      the committed reference CSVs; writes nothing.
  --write             regenerate the reference CSVs and the drift diagnostics.
  --refresh           re-download the source workbooks instead of using the
                      cached copies (implies nothing about --write).
"""
import argparse
import csv
import hashlib
import io
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402

REF_DIR = PIPELINE.parent / "reference"
RESULTS_DIR = PIPELINE.parents[2] / "Results" / "pipeline_diagnostics"

# One entry per revision, in chain order. `sheet` is the sheet NAME where the
# workbook has more than one and the first is not the concordance, and the
# integer 0 otherwise. `col_from` and `col_to` are positional because the header
# rows differ between workbooks (row 0 in the 1997 file, row 2 in the others)
# while the column positions do not.
LINKS = [
    (1997, 2002, "1997_NAICS_to_2002_NAICS.xls", "Concordance 23 US NoD", 0, 2),
    (2002, 2007, "2002_to_2007_NAICS.xls", 0, 0, 2),
    (2007, 2012, "2007_to_2012_NAICS.xls", 0, 0, 2),
    (2012, 2017, "2012_to_2017_NAICS.xlsx", 0, 0, 2),
]
BASE_VINTAGE = 2012           # the vintage the chain composes onto
VINTAGES = (1997, 2002, 2007, 2012, 2017)

# The two groups the 2012->2017 revision rearranges. Derived from the Census
# file as connected components; asserted against this, never read from it.
EXPECTED_MERGES = {
    "452X": ({"4521", "4529"}, {"4522", "4523"}),
    "517X": ({"5171", "5172"}, {"5173"}),
}
EFSY_FIRST_YEAR, EFSY_LAST_YEAR = 1975, 2018


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name, refresh=False):
    """Download one workbook if it is not cached. Returns (path, transport).

    urllib first, curl as the fallback, exactly as 21_ingest_cbsa.py and
    22_ingest_bea_areas.py do. The fallback is not defensive padding: this venv
    carries no certifi bundle and Python does not read the macOS keychain, so
    urllib raises SSLCertVerificationError against census.gov while curl - which
    does read it - succeeds. Only a certificate failure falls back; a 404 or a
    timeout is a real problem and must not be retried into a transport that
    would report the same thing differently.
    """
    dc.CENSUS_NAICS_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    dest = dc.CENSUS_NAICS_SOURCE_DIR / name
    url = f"{dc.CENSUS_NAICS_BASE_URL}/{name}"
    transport = "cached"
    if refresh or not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        print(f"    downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as out:
                out.write(r.read())
            transport = "urllib"
        except urllib.error.URLError as exc:
            if not isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise
            print(f"    urllib: {exc.reason.__class__.__name__}; falling back to curl")
            subprocess.run(["curl", "-sSL", "--fail", "--max-time", "180",
                            "-o", str(tmp), url], check=True)
            transport = "curl"
        tmp.replace(dest)
    return dest, transport


def parse_link(path, sheet, col_from, col_to):
    """The matched 6-digit pairs in one workbook, deduplicated, sorted.

    Read with header=None so the differing header depths need no per-file
    offset: a header row cannot survive the six-digit test, and neither can the
    1997 workbook's `NNNNNN Total` subtotal rows or its `Grand Total` line.
    """
    df = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str)
    a = df[col_from].astype(str).str.strip()
    b = df[col_to].astype(str).str.strip()
    ok = a.str.fullmatch(r"\d{6}").fillna(False) & b.str.fullmatch(r"\d{6}").fillna(False)
    pairs = (pd.DataFrame({"from": a[ok], "to": b[ok]})
             .drop_duplicates().sort_values(["from", "to"]).reset_index(drop=True))
    n = pairs.groupby("from")["to"].transform("size")
    pairs["weight"] = 1.0 / n
    return pairs


def as_map(pairs):
    """{from: [(to, weight), ...]} for one link."""
    out = {}
    for f, t, w in zip(pairs["from"], pairs["to"], pairs["weight"]):
        out.setdefault(f, []).append((t, w))
    return out


def compose(first, second):
    """Chain two 6-digit maps. A code absent from `second` is a dead end and is
    dropped here rather than carried through as an identity: the Census tables
    list unchanged codes explicitly, so absence means the code did not survive
    the revision, and inventing an identity edge would put dollars on a code
    that the target vintage does not have."""
    out = {}
    for f, pairs in first.items():
        acc = {}
        for mid, w in pairs:
            for t, w2 in second.get(mid, ()):
                acc[t] = acc.get(t, 0.0) + w * w2
        if acc:
            out[f] = sorted(acc.items())
    return out


def derive_axis(pairs_2012_2017, diag):
    """The vintage-stable 4-digit axis, as connected components of the
    2012<->2017 four-digit graph. Returns (axis_of_2012, axis_of_2017)."""
    edges = {(f[:4], t[:4]) for f, t in zip(pairs_2012_2017["from"],
                                            pairs_2012_2017["to"])}
    adj = {}
    for a, b in edges:
        adj.setdefault(("2012", a), set()).add(("2017", b))
        adj.setdefault(("2017", b), set()).add(("2012", a))
    seen, comps = set(), []
    for node in sorted(adj):
        if node in seen:
            continue
        stack, comp = [node], []
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)

    axis12, axis17, merges = {}, {}, {}
    for comp in comps:
        old = {c for v, c in comp if v == "2012"}
        new = {c for v, c in comp if v == "2017"}
        codes = old | new
        if old == new and len(codes) == 1:
            label = next(iter(codes))
        else:
            prefixes = {c[:3] for c in codes}
            assert len(prefixes) == 1, (
                f"a 2012<->2017 component spans more than one 3-digit family "
                f"and has no natural label: {sorted(codes)}")
            label = next(iter(prefixes)) + "X"
            merges[label] = (old, new)
        for c in old:
            axis12[c] = label
        for c in new:
            axis17[c] = label

    assert merges == EXPECTED_MERGES, (
        f"the 2012->2017 four-digit rearrangement is not the expected one.\n"
        f"  derived:  { {k: (sorted(v[0]), sorted(v[1])) for k, v in merges.items()} }\n"
        f"  expected: { {k: (sorted(v[0]), sorted(v[1])) for k, v in EXPECTED_MERGES.items()} }")
    for label, (old, new) in sorted(merges.items()):
        diag.append(("axis_merge", label, "2012", " ".join(sorted(old))))
        diag.append(("axis_merge", label, "2017", " ".join(sorted(new))))
    diag.append(("axis_cells", "", "", len(set(axis12.values()))))
    diag.append(("naics4_codes", "2012", "", len(axis12)))
    diag.append(("naics4_codes", "2017", "", len(axis17)))
    return axis12, axis17


def efsy_live_codes(year):
    """The 4-digit codes EFSY actually reports employment on in one year.

    The naics4_alloc files are rectangular - the support is the union over all
    years, materialised as zeros - so the column of codes is the same in every
    file and says nothing about the vintage. What is reported does.
    """
    df = pd.read_parquet(dc.EMP_OUT_DIR / "naics4_alloc" / f"y{year}.parquet",
                         columns=["naics", "in_efsy", "emp"])
    live = df.loc[df["in_efsy"].to_numpy() & (df["emp"].to_numpy() > 0), "naics"]
    return set(live.astype(str))


def check_efsy(axis12, axis17, diag):
    """The vintage determination, reduced to assertions.

    Two claims are checked. That EFSY's own 2016->2017 change is exactly the
    four-digit rearrangement the Census file describes - which is what makes
    EFSY 2012-then-2017 rather than some other pair of vintages. And that every
    code EFSY reports, in every year, lands on the axis.
    """
    change_2012 = set().union(*(o for o, _ in EXPECTED_MERGES.values()))
    change_2017 = set().union(*(n for _, n in EXPECTED_MERGES.values()))
    live16, live17 = efsy_live_codes(2016), efsy_live_codes(2017)
    assert live17 - live16 == change_2017 and live16 - live17 == change_2012, (
        f"EFSY's 2016->2017 change is not the Census 2012->2017 rearrangement.\n"
        f"  EFSY added {sorted(live17 - live16)}, expected {sorted(change_2017)}\n"
        f"  EFSY dropped {sorted(live16 - live17)}, expected {sorted(change_2012)}")
    diag.append(("efsy_2016_to_2017_added", "", "", " ".join(sorted(live17 - live16))))
    diag.append(("efsy_2016_to_2017_dropped", "", "", " ".join(sorted(live16 - live17))))

    n_off = 0
    for y in range(EFSY_FIRST_YEAR, EFSY_LAST_YEAR + 1):
        live = efsy_live_codes(y)
        table = axis17 if y >= 2017 else axis12
        off = sorted(c for c in live if c not in table)
        assert not off, f"EFSY {y} reports codes that are not on the axis: {off}"
        n_off += len(off)
        diag.append(("efsy_live_codes", y, "2017" if y >= 2017 else "2012", len(live)))
    print(f"  EFSY: every live 4-digit code in {EFSY_FIRST_YEAR}-{EFSY_LAST_YEAR} "
          f"is on the axis ({n_off} exceptions)", flush=True)


def build(diag, refresh=False):
    """Every table this producer writes, plus the diagnostics, in one pass."""
    link_pairs, receipts = {}, []
    for a, b, name, sheet, cf, ct in LINKS:
        path, transport = fetch(name, refresh)
        pairs = parse_link(path, sheet, cf, ct)
        link_pairs[(a, b)] = pairs
        receipts.append((name, f"{dc.CENSUS_NAICS_BASE_URL}/{name}",
                         path.stat().st_size, sha256(path), transport))
        splits = int((pairs.groupby("from")["to"].size() > 1).sum())
        print(f"  {a}->{b}: {len(pairs):,} pairs, {pairs['from'].nunique():,} "
              f"source codes, {splits:,} of them split", flush=True)
        diag.append(("link_pairs", f"{a}_{b}", "", len(pairs)))
        diag.append(("link_source_codes", f"{a}_{b}", "", pairs["from"].nunique()))
        diag.append(("link_split_source_codes", f"{a}_{b}", "", splits))
        diag.append(("link_max_fanout", f"{a}_{b}", "",
                     int(pairs.groupby("from")["to"].size().max())))

    axis12, axis17 = derive_axis(link_pairs[(2012, 2017)], diag)

    # --- compose every vintage onto the base, at six digits -----------------
    maps = {k: as_map(v) for k, v in link_pairs.items()}
    to_base = {BASE_VINTAGE: None}          # None marks the identity
    to_base[2007] = maps[(2007, 2012)]
    to_base[2002] = compose(maps[(2002, 2007)], to_base[2007])
    to_base[1997] = compose(maps[(1997, 2002)], to_base[2002])

    # The 6-digit universe of each vintage: what the link into it produced,
    # unioned with what the link out of it consumed. A code seen on only one
    # side is real and must still be counted, which is why this is a union.
    universe = {
        1997: set(link_pairs[(1997, 2002)]["from"]),
        2002: set(link_pairs[(1997, 2002)]["to"]) | set(link_pairs[(2002, 2007)]["from"]),
        2007: set(link_pairs[(2002, 2007)]["to"]) | set(link_pairs[(2007, 2012)]["from"]),
        2012: set(link_pairs[(2007, 2012)]["to"]) | set(link_pairs[(2012, 2017)]["from"]),
        2017: set(link_pairs[(2012, 2017)]["to"]),
    }

    rows, dead = [], {}
    for v in VINTAGES:
        table = axis17 if v == 2017 else axis12
        deadends = []
        for code in sorted(universe[v]):
            if v == 2017 or v == BASE_VINTAGE:
                dest = [(code, 1.0)]
            else:
                dest = to_base[v].get(code)
                if not dest:
                    deadends.append(code)
                    continue
            acc = {}
            for c6, w in dest:
                a4 = table.get(c6[:4])
                if a4 is None:
                    deadends.append(code)
                    acc = {}
                    break
                acc[a4] = acc.get(a4, 0.0) + w
            if not acc:
                continue
            total = sum(acc.values())
            assert abs(total - 1.0) < 1e-9, (v, code, total)
            for a4, w in sorted(acc.items()):
                rows.append((v, code, a4, w))
        dead[v] = deadends
        diag.append(("vintage_codes", v, "", len(universe[v])))
        diag.append(("vintage_deadends", v, "", len(deadends)))
        if deadends:
            diag.append(("vintage_deadend_codes", v, "", " ".join(deadends[:40])))
        n_split = sum(1 for c in set(r[1] for r in rows if r[0] == v)
                      if sum(1 for r in rows if r[0] == v and r[1] == c) > 1)
        diag.append(("vintage_split_codes", v, "", n_split))
        print(f"  vintage {v}: {len(universe[v]):,} six-digit codes, "
              f"{n_split:,} split across axis cells, {len(deadends)} dead ends",
              flush=True)

    chain = pd.DataFrame(rows, columns=["vintage", "naics6", "axis4", "weight"])
    s = chain.groupby(["vintage", "naics6"])["weight"].sum()
    assert np.allclose(s.to_numpy(), 1.0, atol=1e-9), "chain weights must sum to 1"

    axis = pd.DataFrame(
        [(2012, c, a) for c, a in sorted(axis12.items())]
        + [(2017, c, a) for c, a in sorted(axis17.items())],
        columns=["vintage", "naics4", "axis4"])
    assert set(axis.loc[axis["vintage"] == 2012, "axis4"]) == \
        set(axis.loc[axis["vintage"] == 2017, "axis4"]), \
        "the axis must be identical under NAICS 2012 and NAICS 2017"

    check_efsy(axis12, axis17, diag)
    return link_pairs, axis, chain, receipts


def serialize(df, meta):
    buf = io.StringIO()
    for line in meta:
        buf.write(f"# {line}\n")
    df.to_csv(buf, index=False, float_format="%.10g")
    return buf.getvalue()


UNWEIGHTED = ("weight = equal split across a source code's matches. The Census "
              "table is unweighted: it records that a code matched another "
              "code, not in what proportion.")


def tables(link_pairs, axis, chain, receipts):
    """(path, text) for every reference file, in write order."""
    rec = {r[0]: r for r in receipts}
    out = []
    for a, b, name, *_ in LINKS:
        # The header carries only properties of the SOURCE FILE. Which
        # transport fetched it is a property of the run, not of the file, and
        # putting it here would make a cached re-run differ from the run that
        # downloaded - a --verify failure that means nothing. It is recorded in
        # download_receipt.csv beside the workbooks instead.
        _, url, nbytes, digest, _transport = rec[name]
        out.append((REF_DIR / f"naics_{a}_{b}_census.csv",
                    serialize(link_pairs[(a, b)], [
                        f"Census {a} NAICS to {b} NAICS concordance, "
                        f"6-digit matched pairs.",
                        f"Source: {url}",
                        f"sha256 of downloaded workbook: {digest}",
                        f"bytes: {nbytes}",
                        UNWEIGHTED,
                        "Built by pipeline/07_naics_vintage_concordances.py."])))
    merges = "; ".join(
        f"{lab} = {'+'.join(sorted(o))} (2012) = {'+'.join(sorted(n))} (2017)"
        for lab, (o, n) in sorted(EXPECTED_MERGES.items()))
    out.append((REF_DIR / "efsy_naics4_axis.csv", serialize(axis, [
        "The vintage-stable 4-digit industry axis: NAICS 2012 four-digit codes "
        "with each group the 2012->2017 revision rearranges collapsed.",
        "This is the classification the EFSY employment cells carry: NAICS "
        "2012 for 1975-2016, NAICS 2017 for 2017-2018.",
        f"Merged groups: {merges}",
        "Both groups are closed in the Census 2012->2017 concordance - no "
        "mapping leaks in or out - so the collapse is lossless.",
        "Built by pipeline/07_naics_vintage_concordances.py."])))
    out.append((REF_DIR / "naics_to_efsy_axis.csv", serialize(chain, [
        "NAICS 6-digit code, at its own vintage, to the vintage-stable 4-digit "
        "axis in efsy_naics4_axis.csv.",
        "Composed at SIX digits along 1997->2002->2007->2012 and collapsed to "
        "four only at the end, because NAICS reuses codes: 5173 is Telecom "
        "Resellers in 2002/2007 (folded into 5179 by 2012) and Wired and "
        "Wireless Carriers in 2017.",
        UNWEIGHTED,
        "Weights sum to 1 for every (vintage, naics6); asserted.",
        "Built by pipeline/07_naics_vintage_concordances.py."])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="regenerate the reference CSVs and the drift "
                         "diagnostics (default is --verify: recompute and "
                         "byte-compare)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download the source workbooks instead of using "
                         "the cached copies")
    args = ap.parse_args()
    t0 = time.time()
    diag = []
    link_pairs, axis, chain, receipts = build(diag, args.refresh)

    for path, text in tables(link_pairs, axis, chain, receipts):
        if args.write:
            path.write_text(text)
            print(f"  wrote {path.name}  ({len(text.splitlines()) - 6:,} rows)")
        else:
            if not path.exists():
                raise AssertionError(f"{path.name} is not committed; run --write first")
            committed = path.read_text()
            if text != committed:
                a, b = text.splitlines(), committed.splitlines()
                for i, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        raise AssertionError(
                            f"{path.name} differs from the recompute at line "
                            f"{i+1}:\n  recomputed: {x[:120]}\n  committed:  {y[:120]}")
                raise AssertionError(f"{path.name}: length differs "
                                     f"({len(a)} vs {len(b)} lines)")
            print(f"  {path.name}: recompute is byte-identical to the committed table")

    if args.write:
        dc.CENSUS_NAICS_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        with open(dc.CENSUS_NAICS_SOURCE_DIR / "download_receipt.csv", "w",
                  newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "url", "bytes", "sha256", "transport"])
            w.writerows(receipts)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dd = pd.DataFrame(diag, columns=["metric", "key1", "key2", "value"])
        dd.to_csv(RESULTS_DIR / "naics_vintage_drift.csv", index=False)
        print(f"  wrote {RESULTS_DIR / 'naics_vintage_drift.csv'}  ({len(dd)} rows)")
    print(f"  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
