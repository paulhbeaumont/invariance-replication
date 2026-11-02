"""Recover the official crosswalk for the FY2012 PSC renumbering.

The August 2011 PSC manual renumbered the multi-use facility service codes
(384 codes ended, 504 added, mostly Y construction, Z maintenance, C1
architect-engineer families) effective FY2012. In the raw contract panel
this shows up as 6.2% of FY2012 county dollars sitting in codes that did
not exist in FY2011, with the old codes bleeding out gradually rather than
dying at the seam, so any fixed category set spanning 2011-2012 is broken
in those families.

The manual itself contains the official conversion: a "Crosswalk from
Previous Version of Manual" table mapping every previous PSC to its
current PSC. This producer downloads the manual, extracts that table, and
writes it to reference/psc2011_crosswalk.csv with one row per (previous,
current) pair. Codes ended with no successor carry an empty current_psc.

Source (pinned):
  https://www.acquisition.gov/sites/default/files/page_file_uploads/PSC%20Manual%20-%20Final%20-%2011%20August%202011.pdf
  sha256 c06bdb39de77e9bed0b3b8a41b5daf5ccf9c3860d88904d90b8c5d458f05b411

Harmonization semantics for a consumer building a consistent category
panel: where a new code has exactly one predecessor (512 of the 552
changed codes), relabel the new code to its predecessor in post-2011
years; where several old codes merged into one new code (10 new codes),
relabel the old codes to the new code in pre-2012 years. Applied to the
county panel this cuts the 2011-to-2012 newborn dollar share from 6.2% to
0.55%, and the 2011-to-2018 comparison to 1.20% newborn / 0.51% dying,
which is background-level churn. The residual is a small set of genuinely
new codes with no predecessor (largest: D318, 0.29% of FY2012 dollars).

Requires the poppler `pdftotext` binary on PATH for the extraction step.
"""

import hashlib
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

URL = ("https://www.acquisition.gov/sites/default/files/page_file_uploads/"
       "PSC%20Manual%20-%20Final%20-%2011%20August%202011.pdf")
SHA256 = "c06bdb39de77e9bed0b3b8a41b5daf5ccf9c3860d88904d90b8c5d458f05b411"

REF_DIR = Path(__file__).resolve().parent.parent / "reference"
OUT = REF_DIR / "psc2011_crosswalk.csv"

CODE = re.compile(r"^[0-9A-Z]{4}$")
DASH = re.compile(r"^-{2,}$")


def fetch_pdf(tmpdir):
    path = Path(tmpdir) / "psc_manual_2011.pdf"
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == SHA256, f"manual hash changed: {digest}"
    return path


def parse_crosswalk(txt_path):
    lines = [l.strip() for l in open(txt_path, encoding="utf-8")]
    start = next(i for i, l in enumerate(lines)
                 if l == "Crosswalk from Previous Version of Manual")
    lines = [l for l in lines[start:] if l]
    pairs = []
    i = 0
    while i < len(lines) - 1:
        if CODE.match(lines[i]) and (CODE.match(lines[i + 1]) or DASH.match(lines[i + 1])):
            old = lines[i]
            new = "" if DASH.match(lines[i + 1]) else lines[i + 1]
            j = i + 2
            while j < len(lines) - 1 and not (
                    CODE.match(lines[j]) and (CODE.match(lines[j + 1]) or DASH.match(lines[j + 1]))):
                j += 1
            pairs.append((old, new))
            i = j
        else:
            i += 1
    return pairs


def main():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = fetch_pdf(tmp)
        txt = Path(tmp) / "manual.txt"
        subprocess.run(["pdftotext", str(pdf), str(txt)], check=True)
        pairs = parse_crosswalk(txt)

    n_current = sum(1 for _, n in pairs if n)
    changed = [(o, n) for o, n in pairs if n and n != o]
    preds = {}
    for o, n in changed:
        preds.setdefault(n, []).append(o)
    unique = sum(1 for v in preds.values() if len(v) == 1)

    assert len(pairs) == 2928, f"row count {len(pairs)} != 2928"
    assert n_current == 2913, f"rows with current code {n_current} != 2913"
    assert len(changed) == 552, f"changed rows {len(changed)} != 552"
    assert unique == 512, f"unique-predecessor codes {unique} != 512"
    assert dict(changed)["Y211"] == "Y1KA"
    assert dict(changed)["1900"] == "1905"

    REF_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# Official PSC crosswalk from the August 2011 PSC manual "
                "(FY2012 renumbering).\n")
        f.write(f"# Source: {URL}\n")
        f.write(f"# sha256 of manual PDF: {SHA256}\n")
        f.write("# current_psc empty = code ended with no successor.\n")
        f.write("previous_psc,current_psc\n")
        for o, n in pairs:
            f.write(f"{o},{n}\n")
    print(f"wrote {OUT}: {len(pairs)} rows, {len(changed)} changed, "
          f"{unique} new codes with a unique predecessor. All checks passed.")


if __name__ == "__main__":
    main()
