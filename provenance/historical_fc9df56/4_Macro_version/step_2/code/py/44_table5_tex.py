#!/usr/bin/env python3
"""Table 5 as LaTeX: what accounts for the county-to-cell log gap.

Renders from the saved decomposition file and adds no estimate.

NO STANDARD ERRORS AND NO STARS. Every line below the gap is a difference between
two coefficients, and the claim is that those differences sum to the gap exactly,
not that any one of them is distinguishable from zero. An SE printed beside a
term that exists to add up invites reading it as an estimate of something.

TWO HORIZONS, BECAUSE ONE WOULD HIDE HALF THE FAILURE. At h=4 the w^- chain is
what blows up (-2.2193 at NAICS4) while the equal-weight gap is +0.1335; at h=3
it is the other way round (+1.7476 against +0.0925). A table at a single horizon
would show one inadmissible weighting behaving itself and invite the reader to
conclude it is safe.

ENTRY AND EXIT ARE NEVER NETTED, and both panels repeat all of their terms rather
than referring back, so each panel adds up on its own page without the reader
carrying numbers between them.
"""
import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from out_paths import write_exhibit  # noqa: E402

RESULT = Path(__file__).resolve().parents[2] / "result"

PANELS = (
    ("A", r"Panel A. Disaggregated regression at $w^-$, the naive lagged share",
     "cell_naive", "gap_naive",
     (("entry", r"Entry"), ("exit", r"Exit"), ("jensen", r"Jensen"),
      ("renorm", r"Renormalisation"))),
    ("B", r"Panel B. Disaggregated regression at equal weights, $\beta^d_{OLS}$",
     "cell_equal", "gap_equal",
     (("entry", r"Entry"), ("exit", r"Exit"), ("jensen", r"Jensen"),
      ("within", r"Within-unit bridge"), ("across", r"Across-unit count"))),
)

a = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument("--horizon", type=int, nargs="+", default=[3, 4])
a.add_argument("--share", default="fixed5")
a.add_argument("--period", default="pre")
a.add_argument("--source", default="efsy",
               choices=["efsy", "nets_actual", "nets_all"],
               help="which employment series the decomposition file was\n"
                    "                     estimated on; see 43_tables_tex.py")
a.add_argument("--slides", action="store_true",
               help="Panel A only, one horizon, as a bare scaled tabular\n                     that inputs into a Beamer frame")
g = a.parse_args()

# The slide fragment is scaled geometrically rather than set smaller, so one
# factor controls both its width and its height.
SLIDE_WIDTH = "0.46"
if g.slides and len(g.horizon) != 1:
    raise SystemExit("--slides renders one horizon per file; pass one --horizon")

# The baseline frame carries no share tag in its decomposition file. The full
# table now carries its own note (ESTIMATION_BASE below); until 2026-09 only the
# slide fragment was wired for it, because the single note described the
# candidate estimation and would have misdescribed the baseline.
BASE = g.period == "baseline"

# WS2 step 3: the employment source is part of the file's identity and of what
# the note has to say. `efsy` leaves every pre-existing filename untouched.
SRC = g.source
STAG = "" if SRC == "efsy" else f"_{SRC}"
SUBDIR = None if SRC == "efsy" else "nets_baseline"
SOURCE_NOTE = {
    "efsy": "",
    "nets_actual":
        "The employment series is NETS on the actual-employment universe "
        "rather than the desuppressed CBP; the swap bundles the data source, "
        "the covered universe, the intersected county support and the "
        "degenerate-county exclusions, and NETS begins in 1990 so the "
        "estimable windows start later. ",
    "nets_all":
        "The employment series is NETS on the full panel, imputed employment "
        "included, rather than the desuppressed CBP; the swap bundles the data "
        "source, the covered universe, the intersected county support and the "
        "degenerate-county exclusions, and NETS begins in 1990 so the "
        "estimable windows start later. ",
}[SRC]

d = pd.read_csv(RESULT / (f"coeff_table_decomposition_baseline{STAG}.csv" if BASE
                          else f"coeff_table_decomposition_{g.period}{STAG}_{g.share}.csv"))
d["rung"] = d["rung"].str.replace("_alloc", "", regex=False)

# Read, never restated - see 43_tables_tex.py for why the era stopped being a
# literal here. The window differs by horizon (t-h has to reach into the era), so
# it is taken over the horizons this table actually prints rather than over the
# whole file, which also carries h=2.
_w = d[d["h"].isin(g.horizon)]
WINDOW = (f"county-years {int(_w['year_first'].min())}--"
          f"{int(_w['year_last'].max())}")

# (horizon, rung) in printed order.
KEYS = [(h, r) for h in g.horizon for r in ("naics2", "naics4")]
CELL = {(int(x["h"]), x["rung"]): x for _, x in d.iterrows()}
missing = [k for k in KEYS if k not in CELL]
if missing:
    raise SystemExit(f"the decomposition file has no rows for {missing}")
N = len(KEYS)


# The note, assembled once so the %-substitution applies to the whole paragraph
# and not merely to its last fragment. Any literal per-cent sign inside these
# strings has to be doubled: this text goes through %-formatting before it goes
# into LaTeX.
_NOTE_HEAD = (
    r"{\scriptsize \textit{Notes:} Each column decomposes the gap between the "
    r"county log-growth coefficient and the coefficient from the same regression "
    r"run on county-by-industry cells. Every entry is a coefficient or a "
    r"difference of two coefficients from county-level regressions on one sample, "
    r"and the indented terms sum to the gap above them; the largest residual "
    r"across the eight decompositions shown is %.1e. Entry and exit are reported "
    r"separately and never netted, because both are large and of opposite sign at "
    r"NAICS4 while their sum is not. The Jensen term is the "
    r"Kullback-Leibler divergence of the survivor weights from uniform. "
    r"Renormalisation is the step from weights that sum to one over survivors to "
    r"the naive lagged share on the county's full base. The within-unit and "
    r"across-unit terms are Lemma 1's split of the move to equal weights. ")
_NOTE_TAIL = (r"Point estimates only: the table is an accounting identity, not "
              r"a set of separate estimates. \par}")

ESTIMATION_CANDIDATE = (
    r"Estimation is 2SLS on the income-based instrument, %s"
    r", non-overlapping windows, county and year fixed effects, clustered on "
    r"the county. ")
ESTIMATION_BASE = (
    r"Estimation is the locked baseline: the reduced form on the harmonized "
    r"top-800 composition instrument, a five-year share window with the "
    r"positive-contract rule and a population denominator, division-by-year "
    r"effects, %s, overlapping windows, clustered on the county, and no "
    r"winsorization anywhere. Coefficients are in percentage points per "
    r"\$1{,}000 of contracts per head. ")

NOTE = (_NOTE_HEAD + SOURCE_NOTE
        + (ESTIMATION_BASE if BASE else ESTIMATION_CANDIDATE) + _NOTE_TAIL)


def row(label, values, indent=False, signed=False):
    """`signed` puts an explicit + on the gap as well as on the terms below it,
    because the reader's job on those rows is to check that they add."""
    lab = (r"\hspace{1.5em}" if indent else "") + label
    fmt = "{:+.4f}" if (indent or signed) else "{:.4f}"
    return lab + " & " + " & ".join(fmt.format(v) for v in values) + r" \\"


lines = []
worst = 0.0
for tag, title, cellk, gapk, steps in (PANELS[:1] if g.slides else PANELS):
    lines.append(r"\addlinespace" if lines else "")
    if not g.slides:
        lines.append(r"\textit{%s} \\" % title)
    if tag == "A":
        lines.append(row(r"County, $\beta^b$",
                         [float(CELL[k]["county"]) for k in KEYS]))
    lines.append(row(r"County $\times$ industry, $\beta^d$",
                     [float(CELL[k][cellk]) for k in KEYS]))
    lines.append(row(r"Gap, $\beta^d-\beta^b$",
                     [float(CELL[k][gapk]) for k in KEYS], signed=True))
    for sk, slab in steps:
        lines.append(row(slab, [float(CELL[k][sk]) for k in KEYS], indent=True))
    for k in KEYS:
        s = sum(float(CELL[k][sk]) for sk, _ in steps)
        worst = max(worst, abs(s - float(CELL[k][gapk])))
if g.slides:
    # The slide carries Panel A in full, then the two equal-weight lines that
    # the naive-share panel cannot show: the coefficient an applied person
    # actually gets, and its gap. Panel B's own decomposition stays in Table 5.
    _, _, cellk_eq, gapk_eq, _ = PANELS[1]
    lines.append(r"\addlinespace")
    lines.append(row(r"County $\times$ industry, equal weights",
                     [float(CELL[k][cellk_eq]) for k in KEYS]))
    lines.append(row(r"Gap at equal weights",
                     [float(CELL[k][gapk_eq]) for k in KEYS], signed=True))

lines = [x for x in lines if x != ""]

head = [
    r"\textit{Dependent variable} & \multicolumn{%d}{c}{Log growth of employment} \\" % N,
    r"\cmidrule(r){2-%d}" % (N + 1),
    r"\textit{Horizon} & " + " & ".join(
        r"\multicolumn{2}{c}{$h=%d$}" % h for h in g.horizon) + r" \\",
    "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i)
            for i in range(len(g.horizon))),
    r"\textit{Level of aggregation} & " + " & ".join(
        r"$\times$ NAICS%s" % ("2" if r == "naics2" else "4") for _, r in KEYS)
    + r" \\",
    *([] if g.slides else
      [r"& " + " & ".join(f"({i + 1})" for i in range(N)) + r" \\"]),
    r"\midrule",
]

tail = [
    r"\addlinespace",
    r"County-years & " + " & ".join(f"{int(CELL[k]['n_cy']):,}" for k in KEYS)
    + r" \\",
    r"\hspace{1.5em}\textit{of which log cannot use} & " + " & ".join(
        f"{int(CELL[k]['n_log_blind']):,}" for k in KEYS) + r" \\",
    r"\bottomrule",
]

if g.slides:
    tex = "\n".join([
        r"% Slide table. The county-to-cell log gap, one horizon, Panel A.",
        r"% Rendered by 44_table5_tex.py --slides. Do not edit.",
        r"\adjustbox{width=%s\linewidth}{" % SLIDE_WIDTH,
        r"\begin{tabular}{l%s}" % ("c" * N),
        r"\toprule",
        *head,
        *lines,
        *tail,
        r"\end{tabular}",
        r"}",
        "",
    ])
    out, _ = write_exhibit(
        f"TableS5_log_gap_h{g.horizon[0]}_slides_baseline{STAG}.tex" if BASE else
        f"TableS5_log_gap_h{g.horizon[0]}_slides_{g.period}{STAG}_{g.share}.tex",
        tex, quiet=True, subdir=SUBDIR)
    print(f"  wrote {out.name}, worst decomposition residual {worst:.1e}")
    raise SystemExit(0)

tex = "\n".join([
    r"% Table 5. Accounting for the county-to-cell log gap.",
    r"% Rendered by 44_table5_tex.py from the saved decomposition file. Do not edit.",
    r"\begin{table}[htbp]",
    r"\caption{What accounts for the county-to-cell log gap}",
    r"\label{tab:log_gap_decomposition}",
    r"{\centering",
    r"\adjustbox{max width=\linewidth}{",
    r"\begin{tabular}{l%s}" % ("c" * N),
    r"\toprule",
    *head,
    *lines,
    *tail,
    r"\end{tabular}",
    r"}",
    r"\par}",
    r"\medskip",
    NOTE % (worst, WINDOW),
    r"\end{table}",
    "",
])

out, _ = write_exhibit(
    f"Table5_log_gap_decomposition_baseline{STAG}.tex" if BASE else
    f"Table5_log_gap_decomposition_{g.period}{STAG}_{g.share}.tex",
    tex, quiet=True, subdir=SUBDIR)
print(f"\n  TABLE 5, h={g.horizon}, "
      + (f"locked baseline, source {SRC}" if BASE
         else f"{g.period}-1998, z_income_{g.share}, source {SRC}") + "\n")
print("  " + " " * 26 + "".join(f"{'N' + str(r)[5:] + ' h' + str(h):>12s}"
                                for h, r in KEYS))
for tag, _, cellk, gapk, steps in PANELS:
    print(f"  Panel {tag}")
    print("    " + f"{'gap':24s}" + "".join(f"{float(CELL[k][gapk]):+12.4f}"
                                            for k in KEYS))
    for sk, slab in steps:
        print("      " + f"{slab:22s}" + "".join(f"{float(CELL[k][sk]):+12.4f}"
                                                 for k in KEYS))
print(f"\n  largest residual across the {2 * N} decompositions: {worst:.1e}")
print(f"  wrote {out.relative_to(RESULT.parents[2])}")
