#!/usr/bin/env python3
"""Parse fixed-width DCADS and MPCF contract files into per-file parquet.

Layouts live in code/layouts/*.json, one per era, each declaring a record
length and the byte offsets of every field. A file is parsed only when its name
appears in some layout's applies_to list, so an unrecognised file is reported
rather than parsed against a guessed layout.

Output: $temp/dcads_parsed/<layout_id>/<filename>.parquet
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / "1_current_paper" / "_data" / "source" / "DCADS"
LAYOUT_DIR = Path(__file__).resolve().parents[1] / "layouts"
OUT_DIR = PROJECT_ROOT / "1_current_paper" / "_data" / "temp" / "dcads_parsed"

CHUNK_RECORDS = 200_000

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
# table, as "+" in the 1960s files, and as a blank zone byte where positives are
# otherwise written as plain digits. Negative zero appears as "}" or as "^".
ZONED["+"] = ("0", 1)
ZONED[" "] = ("0", 1)
ZONED["^"] = ("0", -1)


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


def load_layouts():
    layouts, index = {}, {}
    for path in sorted(LAYOUT_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        layouts[spec["layout_id"]] = spec
        for name in spec["applies_to"]:
            index[name] = spec["layout_id"]
    return layouts, index


def parse_file(path, spec):
    """Read one fixed-width file in chunks and return a DataFrame."""
    rl = spec["record_length"]
    enc = spec.get("encoding", "latin-1")
    fields = spec["fields"]
    size = path.stat().st_size
    if size % rl:
        raise ValueError(f"{path.name}: size {size:,} is not a multiple of {rl}")

    frames = []
    with open(path, "rb") as fh:
        while True:
            block = fh.read(rl * CHUNK_RECORDS)
            if not block:
                break
            text = block.decode(enc)
            records = [text[i:i + rl] for i in range(0, len(text), rl)]
            cols = {}
            for f in fields:
                if f["type"] == "int":
                    vals = [decode_zoned(r[f["start"]:f["end"]]) for r in records]
                    cols[f["name"]] = pd.array(vals, dtype="Int64")
                else:
                    cols[f["name"]] = [r[f["start"]:f["end"]].strip() for r in records]
            frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    layouts, index = load_layouts()
    if not layouts:
        print(f"FAIL: no layouts in {LAYOUT_DIR}", file=sys.stderr)
        return 1

    data_files = [p for p in sorted(SOURCE_DIR.rglob("*"))
                  if p.is_file() and p.suffix.lower() not in
                  (".pdf", ".doc", ".lay", ".txt", ".csv", ".zip", ".json")]

    parsed, skipped, failed = [], [], []
    for path in data_files:
        layout_id = index.get(path.name)
        if layout_id is None:
            skipped.append(path)
            continue
        spec = layouts[layout_id]
        try:
            df = parse_file(path, spec)
        except (ValueError, UnicodeDecodeError) as exc:
            failed.append((path, str(exc)))
            print(f"FAIL {path.name}: {exc}", flush=True)
            continue
        dest = OUT_DIR / layout_id / f"{path.name}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        col = df.get("dollars_thousands")
        dollars = int(col.sum()) if col is not None else 0
        unparsed = int(col.isna().sum()) if col is not None else 0
        flag = f"  UNPARSED {unparsed:,}" if unparsed else ""
        parsed.append((path.name, len(df), dollars, unparsed))
        print(f"ok   {path.name:<26} {len(df):>10,} records  "
              f"${dollars / 1e6:>9,.1f} bn{flag}", flush=True)

    print(f"\nparsed {len(parsed)} files, skipped {len(skipped)}, failed {len(failed)}")
    if skipped:
        print("\nno layout declared for these files, not parsed:")
        for p in skipped:
            print(f"  {p.relative_to(SOURCE_DIR)}")
    if failed:
        print(f"\nFAIL: {len(failed)} files did not parse", file=sys.stderr)
        for p, why in failed:
            print(f"  {p.name}: {why}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
