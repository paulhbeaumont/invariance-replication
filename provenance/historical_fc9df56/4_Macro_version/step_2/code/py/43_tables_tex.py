#!/usr/bin/env python3
"""Tables 1 to 4 as LaTeX: the disaggregation ladder, one table per growth rate.

Renders from the saved coefficient file and adds no estimate. That file carries
the coefficient, its clustered standard error and its fitted N per column, so
these tables cannot disagree with the numbers they are supposed to typeset.

THREE COLUMNS, NOT FIVE. The estimation runs an industry-by-year column at both
rungs and NEITHER is printed. What is left is one specification estimated at
three levels of aggregation, which is the comparison the ladder exists to make:
across a row, the only thing that moves is how finely the outcome is cut. An
industry-by-year column moves the fixed effects as well, so it answers a
different question -- how much of the cross-level gap is confounding industry
shocks -- and answering it inside the ladder makes the ladder harder to read
than it makes the confounding clear. Both estimates stay in the coefficient file
and can be printed by adding their keys back to COLS_FULL.

THE FOUR TABLES ARE READ DOWN, NOT ACROSS. Every one has the identical three
columns, so the reader holds a column fixed and turns the page: only Table 1 is
flat, and Tables 2 to 4 are the three ways of being wrong that the paper
enumerates.
"""
import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from out_paths import write_exhibit  # noqa: E402

RESULT = Path(__file__).resolve().parents[2] / "result"
FLAT_TOL = 1e-13

# Keys in the coefficient file, in printed order. `n2_iy` AND `n4_iy` are
# estimated and deliberately absent: the industry-by-year columns are not
# reported. The ladder's job is to show what happens as the OUTCOME is split
# more finely, holding the specification fixed, and an extra column that also
# changes the fixed effects answers a different question -- one about
# confounding industry shocks -- in the middle of that comparison. Both
# estimates stay in the coefficient file for anyone who wants them.
COLS_FULL = ("county", "n2", "n4")
COLS_SLIDES = ("county", "n2", "n4")
# The slide fragment is scaled geometrically rather than set in a smaller
# font: a frame holding one table and two bullets is sized by the table,
# and one factor controls both its width and its height.
SLIDE_WIDTH = "0.80"
COLS = COLS_FULL

SPECS = {
    "midpoint_mpgr": dict(
        n=1, dep="Midpoint growth of employment",
        caption="Midpoint growth is invariant to how finely the outcome is split",
        label="midpoint_ladder", wshort="Midpoint share",
        outcome=r"midpoint employment growth $2(S_t-S_{t-h})/(S_t+S_{t-h})$",
        weight=r"the midpoint share $\omega_{im}=(S_{imt}+S_{im,t-h})"
               r"/(S_{it}+S_{i,t-h})$",
        sample=r"Every cell enters, including those empty in one of the two "
               r"periods, because midpoint growth is defined there and its weight "
               r"is what makes the cells reconstruct the county exactly."),
    "log_naive": dict(
        n=2, dep="Log growth of employment",
        caption="Log growth at lagged-employment shares moves with the level of "
                "aggregation",
        label="log_naive_ladder", wshort="Lagged employment share",
        outcome=r"log employment growth $\ln S_t-\ln S_{t-h}$",
        weight=r"the lagged share $w^-_{im}=S_{im,t-h}/S_{i,t-h}$, computed on "
               r"the county's FULL lagged base",
        sample=r"Only cells positive in both periods enter, because the log is "
               r"undefined otherwise. Those weights therefore sum to less than "
               r"one, and a county that lost employment to exits carries less "
               r"total weight."),
    "log_equal": dict(
        n=3, dep="Log growth of employment",
        caption="Log growth at equal weights moves with the level of aggregation",
        label="log_equal_ladder", wshort="Equal weights",
        outcome=r"log employment growth $\ln S_t-\ln S_{t-h}$",
        weight=r"equal weights, $\beta^d_{OLS}$, the disaggregated regression as "
               r"it is usually run",
        sample=r"Only cells positive in both periods enter, because the log is "
               r"undefined otherwise."),
    "standard_naive": dict(
        n=4, dep="Standard growth of employment",
        caption="Standard growth at lagged-employment shares moves with the level "
                "of aggregation",
        label="standard_ladder", wshort="Lagged employment share",
        outcome=r"standard employment growth $(S_t-S_{t-h})/S_{t-h}$",
        weight=r"the lagged share $w^-_{im}=S_{im,t-h}/S_{i,t-h}$",
        sample=r"Every cell with a positive base enters, so exiters are inside "
               r"the sample at $-1$ and there is no exit term. Standard growth is "
               r"unbounded as the base goes to zero, and the weight rather than a "
               r"cap is what handles that."),
}

a = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument("--horizon", type=int, default=4)
a.add_argument("--share", default="fixed5")
a.add_argument("--period", default="pre")
a.add_argument("--source", default="efsy",
               choices=["efsy", "nets_actual", "nets_all"],
               help="which employment series the coefficient file was\n"
                    "                     estimated on; picks the file and the "
                    "note's data sentence")
a.add_argument("--year-floor", type=int, default=None,
               help="render the coefficient file 34 wrote under the same\n"
                    "                     --year-floor: an EFSY column on the "
                    "window NETS reaches")
a.add_argument("--spec", nargs="+", default=list(SPECS))
a.add_argument("--slides", action="store_true",
               help="emit a bare scaled tabular fragment that inputs into a\n                     Beamer frame, with the weights named in the header\n                     rather than only in the notes. The columns are the\n                     same three the full table prints.")
a.add_argument("--paired", action="store_true",
               help="emit the one-frame comparison fragment: midpoint and log\n"
                    "                     growth side by side, each at the three\n"
                    "                     ladder levels plus the NAICS4 column\n"
                    "                     that adds industry-by-year effects.")
a.add_argument("--paired-specs", nargs=2, default=["midpoint_mpgr", "log_naive"],
               help="the two specs the paired fragment puts side by side")
g = a.parse_args()
COLS = COLS_SLIDES if g.slides else COLS_FULL

# The baseline frame carries no share tag in its coefficient file, and its
# fixed effect is division-by-year, not county and year. The full table now
# carries its own note (NOTE_BASE below): until 2026-09 only the slide fragments
# were wired for it, because the one full-table note described the candidate
# estimation and would have misdescribed the baseline. The fix was to write the
# second note, not to let the first one print over a different design.
BASE = g.period == "baseline"

# WS2 step 3. The employment SOURCE is part of the file's identity and part of
# what the note has to say: swapping EFSY for NETS changes the data source, the
# covered universe, the ccg support and the estimable years at once, and under
# the protocol gate's Q11 that is one bundled row whose note enumerates the
# bundle. `efsy` keeps every pre-existing filename byte-identical.
SRC = g.source
STAG = "" if SRC == "efsy" else f"_{SRC}"
# A matched-window EFSY run belongs with the NETS set, not with the paper's
# Results/ root: it exists only as that comparison's control column.
SUBDIR = None if SRC == "efsy" else "nets_baseline"
SOURCE_NOTE = {
    "efsy": "",
    "nets_actual":
        "The employment series is NETS rather than the desuppressed CBP, on "
        "the actual-employment universe: establishments whose employment is "
        "reported rather than imputed by D\\&B. The swap is one bundled change "
        "-- data source, covered universe, county support intersected with the "
        "CBP one, and the degenerate-county exclusions that follow -- and no "
        "part of it is separable, because a county without a growth rate in "
        "one source has none to compare. NETS begins in 1990, so the estimable "
        "windows start later than the CBP ones and the two sets of "
        "coefficients are not cell-for-cell comparable. ",
    "nets_all":
        "The employment series is NETS rather than the desuppressed CBP, on "
        "the full panel including establishments whose employment D\\&B "
        "imputes; this is the robustness companion to the actual-employment "
        # doubled: this paragraph goes through %-formatting before LaTeX sees it
        "universe, where 57.5\\%% of employment cells are estimates. The swap "
        "is one bundled change -- data source, covered universe, county "
        "support intersected with the CBP one, and the degenerate-county "
        "exclusions that follow. NETS begins in 1990, so the estimable windows "
        "start later than the CBP ones. ",
}[SRC]

if g.year_floor is not None:
    STAG += f"_from{g.year_floor}"
    SUBDIR = "nets_baseline"
    SOURCE_NOTE += (
        "The estimation window is cut at %d so that it matches the years the "
        "NETS panel reaches; this column is the era-matched control for the "
        "NETS comparison, not the reported baseline, and it exists so that the "
        "difference between the two sources cannot be read as a difference "
        "between two eras. " % g.year_floor)

co = pd.read_csv(RESULT / (f"coeff_tables_payoff_baseline{STAG}_h{g.horizon}.csv"
                           if BASE else
                           f"coeff_tables_payoff_{g.period}{STAG}_h{g.horizon}_{g.share}.csv"))

# The estimation window is READ, never restated. Until 2026-08-18 the note below
# said "county-years before 1998" as a literal, which was correct for the only era
# then estimated and would have been silently wrong the moment --period post was
# passed: the filename and the caption would have disagreed about the sample. 34
# now writes the outcome years beside the coefficients, so a table cannot describe
# a window it was not estimated on.
WINDOW = (f"county-years {int(co['year_first'].iloc[0])}--"
          f"{int(co['year_last'].iloc[0])}")


# The two notes differ in exactly one sentence: what was estimated. Everything
# either of them says about the dependent variable, the weights and the columns
# is shared, so the two strings are kept side by side rather than assembled from
# fragments -- a reader auditing the table reads one paragraph, not a template.
_NOTE_HEAD = (
    r"{\scriptsize \textit{Notes:} The dependent variable is %s over a "
    r"%d-year window, weighted by %s. Column (1) is the county; columns (2) "
    r"and (3) split the same outcome into county-by-industry cells at two and "
    r"four NAICS digits. Every column carries the same specification, so the "
    r"only thing that moves across a row is how finely the outcome is cut. The "
    r"treatment is the county-level contract shock and is never reclassified "
    r"to the cell. %s ")
_NOTE_TAIL = (
    r"No growth rate is winsorized and no cell is dropped. Standard errors "
    r"clustered on the county in parentheses; $^{*}$, $^{**}$ and $^{***}$ "
    r"denote significance at the 10, 5 and 1 percent level. \par}")

NOTE_CANDIDATE = (_NOTE_HEAD + SOURCE_NOTE
                  + r"Estimation is 2SLS on the income-based instrument, "
                    r"%s, non-overlapping windows, with the shock and "
                    r"the instrument winsorized at five interquartile ranges "
                    r"by year. " + _NOTE_TAIL)

# The locked baseline. Reduced form, so there is no first stage and no
# winsorization to report; the fixed effect is division-by-year, and the sample
# is the full contract era rather than a pre-1998 cut with non-overlapping
# windows. Saying any of that the candidate way would misdescribe the design.
NOTE_BASE = (_NOTE_HEAD + SOURCE_NOTE
             + r"Estimation is the locked baseline: the reduced form on the "
               r"harmonized top-800 composition instrument, a five-year share "
               r"window with the positive-contract rule and a population "
               r"denominator, division-by-year effects, %s, overlapping "
               r"windows, and no winsorization anywhere. Coefficients are in "
               r"percentage points per \$1{,}000 of contracts per head. "
             + _NOTE_TAIL)

NOTE = NOTE_BASE if BASE else NOTE_CANDIDATE


def stars(b, se):
    r"""The project's convention, from 22_winsor_construction_error.py: normal
    critical values on the clustered t, and the \sym macro esttab emits."""
    if not (se > 0):
        return ""
    t = abs(b / se)
    return (r"\sym{***}" if t > 2.576 else r"\sym{**}" if t > 1.96 else
            r"\sym{*}" if t > 1.645 else "")


def build(spec):
    S = SPECS[spec]
    r = co.loc[co["spec"] == spec].iloc[0]
    # The fitted N travels with each coefficient rather than being inferred from
    # the cache. The midpoint columns happen to use every cell, but the log and
    # standard columns do not, so a cache-derived count would be right in one
    # table and silently wrong in the others.
    nobs = {c: int(r[f"n_{c}"]) for c in COLS}
    N = len(COLS)

    def gap(c):
        """Below FLAT_TOL the gap prints as a bound, because a printed 2e-14
        invites the reader to read a rounding residual as a magnitude."""
        if c == "county":
            return ""
        d = float(r[c]) - float(r["county"])
        return r"$<10^{-13}$" if abs(d) < FLAT_TOL else f"{d:+.4f}"

    if g.slides:
        head = [
            r"\textit{Weights} & \multicolumn{%d}{c}{%s} \\"
            % (N, S["wshort"]),
            r"\cmidrule(r){2-%d}" % (N + 1),
            r"\textit{Level of aggregation} & County & County $\times$ NAICS2"
            r" & County $\times$ NAICS4 \\",
            r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}",
        ]
    else:
        head = [
            r"\textit{Level of aggregation} & County & County $\times$ NAICS2"
            r" & County $\times$ NAICS4 \\",
            r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}",
        ]

    rows = [
        r"\textit{Dependent variable} & \multicolumn{%d}{c}{%s, $h=%d$} \\"
        % (N, S["dep"], g.horizon),
        r"\cmidrule(r){2-%d}" % (N + 1),
        *head,
        *([] if g.slides else
          [r"& " + " & ".join(f"({i + 1})" for i in range(N)) + r" \\"]),
        r"\midrule",
        r"Contract shock & " + " & ".join(
            f"{float(r[c]):.4f}{stars(float(r[c]), float(r['se_' + c]))}"
            for c in COLS) + r" \\",
        r"& " + " & ".join(f"({float(r['se_' + c]):.4f})" for c in COLS) + r" \\",
        *([] if g.slides else
          [r"\hspace{1.5em}\textit{gap from column (1)} & "
           + " & ".join(gap(c) for c in COLS) + r" \\"]),
        r"\\[-1.8ex]\hline\addlinespace\textit{Fixed Effects} \\",
        # EVERY PRINTED COLUMN CARRIES THE SAME EFFECTS, so every row is ticked
        # across the board. That is a consequence of dropping the
        # industry-by-year columns, not a formatting choice: what is left is one
        # specification estimated at three levels of aggregation, which is the
        # comparison the ladder exists to make.
        #
        # The names are the effects the estimator is actually given. The locked
        # baseline demeans on ONE factor, the division-by-year cell
        # (`fit_county_base` / `fit_cell_base`), so it is a single interacted
        # effect and not a county effect plus a year effect. The candidate
        # design demeans on two, `ccg` and `year`, so it gets two rows. Writing
        # "County" and "Year" under the baseline would name effects that are not
        # in it and would understate what division-by-year absorbs.
        *([r"\hspace{1.5em}Division $\times$ year & "
           + " & ".join([r"$\checkmark$"] * N) + r" \\"]
          if BASE else
          [r"\hspace{1.5em}County & " + " & ".join([r"$\checkmark$"] * N) + r" \\",
           r"\hspace{1.5em}Year & " + " & ".join([r"$\checkmark$"] * N) + r" \\"]),
        *([] if g.slides else [r"\addlinespace"]),
        r"Observations & " + " & ".join(f"{nobs[c]:,}" for c in COLS) + r" \\",
        r"\bottomrule",
    ]

    if g.slides:
        tex = "\n".join([
            r"%% Slide table %d. %s" % (S["n"], S["caption"]),
            r"% Rendered by 43_tables_tex.py --slides. Do not edit.",
            r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
            r"\adjustbox{width=%s\linewidth}{" % SLIDE_WIDTH,
            r"\begin{tabular}{l%s}" % ("c" * N),
            r"\toprule",
            *rows,
            r"\end{tabular}",
            r"}",
            "",
        ])
        name = (f"TableS{S['n']}_{S['label']}_slides_baseline{STAG}_h{g.horizon}.tex"
                if BASE else
                f"TableS{S['n']}_{S['label']}_slides_"
                f"{g.period}{STAG}_h{g.horizon}_{g.share}.tex")
        out, _ = write_exhibit(name, tex, quiet=True, subdir=SUBDIR)
        print(f"  Slide table {S['n']}  " + "  ".join(
            f"{float(r[c]):8.4f} ({float(r['se_' + c]):.4f})" for c in COLS))
        return out

    tex = "\n".join([
        r"%% Table %d. %s" % (S["n"], S["caption"]),
        r"% Rendered by 43_tables_tex.py from the saved coefficient file. Do not edit.",
        r"\begin{table}[htbp]",
        r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
        r"\caption{%s}" % S["caption"],
        r"\label{tab:%s}" % S["label"],
        r"{\centering",
        r"\adjustbox{max width=\linewidth}{",
        r"\begin{tabular}{l%s}" % ("c" * N),
        r"\toprule",
        *rows,
        r"\end{tabular}",
        r"}",
        r"\par}",
        r"\medskip",
        NOTE % (S["outcome"], g.horizon, S["weight"], S["sample"], WINDOW),
        r"\end{table}",
        "",
    ])

    name = (f"Table{S['n']}_{S['label']}_baseline{STAG}_h{g.horizon}.tex"
            if BASE else
            f"Table{S['n']}_{S['label']}_{g.period}{STAG}_h{g.horizon}_{g.share}.tex")
    out, _ = write_exhibit(name, tex, quiet=True, subdir=SUBDIR)
    print(f"  Table {S['n']}  " + "  ".join(
        f"{float(r[c]):8.4f} ({float(r['se_' + c]):.4f})" for c in COLS))
    return out


# --------------------------------------------------------------------------
# The paired slide fragment.
#
# WHY THIS ONE PRINTS THE INDUSTRY-BY-YEAR COLUMN AND TABLES 1 TO 4 DO NOT.
# The ladder tables answer one question, so they hold the specification fixed
# and move only the level of aggregation. This fragment answers two at once,
# because a seminar frame has room for one table: the three-level ladder under
# each growth rate, and then what happens when industry-by-year effects are
# added at the finest level. The two questions are kept apart by colour rather
# than by being on separate slides. County is blue, the levels that should
# reproduce it are green, and the industry-by-year column is ORANGE, because it
# changes the fixed effects as well as the level and therefore matches nothing
# by construction. Painting it green would promise an invariance the design
# does not claim.
#
# The gap row is blank wherever the difference is below FLAT_TOL, so a reader
# sees a printed gap only where there is one to read.
PAIR_HEAD = {"midpoint_mpgr": r"\textbf{\small Midpoint growth} of employment",
             "log_naive": r"\textbf{\small Log growth} of employment",
             "log_equal": r"\textbf{\small Log growth} of employment",
             "standard_naive": r"\textbf{\small Standard growth} of employment"}
PAIR_COLS = ("county", "n2", "n4", "n4_iy")
PAIR_TINT = {"county": r"\blue", "n2": r"\green", "n4": r"\green",
             "n4_iy": r"\orange"}
PAIR_LEVEL = {
    "county": "County",
    "n2": r"\begin{tabular}[c]{@{}c@{}}County\\$\times$ NAICS2\end{tabular}",
    "n4": r"\begin{tabular}[c]{@{}c@{}}County\\$\times$ NAICS4\end{tabular}",
    "n4_iy": r"\begin{tabular}[c]{@{}c@{}}County\\$\times$ NAICS4\end{tabular}",
}


def build_paired(specs):
    rows_of = {s: co.loc[co["spec"] == s].iloc[0] for s in specs}
    K = len(PAIR_COLS)
    N = K * len(specs)

    def tint(c, body):
        return "%s{%s}" % (PAIR_TINT[c], body)

    def across(fn):
        return " & ".join(fn(rows_of[s], c) for s in specs for c in PAIR_COLS)

    def coef(r, c):
        b, se = float(r[c]), float(r["se_" + c])
        return tint(c, f"{b:.4f}{stars(b, se)}")

    def stderr(r, c):
        return tint(c, f"({float(r['se_' + c]):.4f})")

    def gap(r, c):
        if c == "county":
            return ""
        d = float(r[c]) - float(r["county"])
        return "" if abs(d) < FLAT_TOL else tint(c, f"{d:+.4f}")

    def nobs(r, c):
        return f"{int(r['n_' + c]):,}"

    tex = "\n".join([
        r"% Paired slide fragment: the ladder under two growth rates, with the",
        r"% industry-by-year column at NAICS4 shown in orange.",
        r"% Rendered by 43_tables_tex.py --paired. Do not edit.",
        r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
        r"\setlength{\tabcolsep}{3.1pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l%s}" % ("c" * N),
        r"\toprule",
        r"& " + " & ".join(r"\multicolumn{%d}{c}{%s}" % (K, PAIR_HEAD[s])
                           for s in specs) + r" \\",
        "".join(r"\cmidrule(lr){%d-%d}" % (2 + i * K, 1 + (i + 1) * K)
                for i in range(len(specs))),
        r"\textit{Weights} & " + " & ".join(
            r"\multicolumn{%d}{c}{%s}" % (K, SPECS[s]["wshort"]) for s in specs)
        + r" \\",
        r"\textit{Level} & "
        + " & ".join(tint(c, PAIR_LEVEL[c]) for _ in specs for c in PAIR_COLS)
        + r" \\",
        r"& " + " & ".join(tint(c, "(%d)" % (i * K + j + 1))
                           for i, _ in enumerate(specs)
                           for j, c in enumerate(PAIR_COLS)) + r" \\",
        r"\midrule",
        r"Contract shock & " + across(coef) + r" \\",
        r"& " + across(stderr) + r" \\",
        r"\addlinespace",
        r"\hspace{0.8em}\textit{gap from county} & " + across(gap) + r" \\",
        r"\midrule",
        r"\textit{Fixed effects} \\",
        r"\hspace{0.8em}Division $\times$ year & "
        + " & ".join([r"\ding{51}"] * N) + r" \\",
        r"\hspace{0.8em}Industry $\times$ year & "
        + " & ".join(r"\orange{\ding{51}}" if c == "n4_iy" else ""
                     for _ in specs for c in PAIR_COLS) + r" \\",
        r"\addlinespace",
        r"Observations & " + across(nobs) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        "",
    ])

    # The pair is in the filename, because the frame that shows log at w^-
    # and the frame that shows log at equal weights are the same table with
    # one spec swapped, and a deck that inputs both must not have to remember
    # which one a bare TableS5 happened to be.
    tag = "_".join("".join(s.split("_")) for s in specs)
    name = (f"TableS5_ladder_paired_{tag}_slides_baseline{STAG}_h{g.horizon}.tex"
            if BASE else
            f"TableS5_ladder_paired_{tag}_slides_{g.period}{STAG}"
            f"_h{g.horizon}_{g.share}.tex")
    out, _ = write_exhibit(name, tex, quiet=True, subdir=SUBDIR)
    for s in specs:
        r = rows_of[s]
        print(f"  paired {s:<16s}" + "  ".join(
            f"{float(r[c]):8.4f} ({float(r['se_' + c]):.4f})"
            for c in PAIR_COLS))
    return out


if g.paired:
    print(f"\n  THE PAIRED LADDER, h={g.horizon}, "
          + (f"locked baseline, source {SRC}" if BASE
             else f"{g.period}-1998, z_income_{g.share}, source {SRC}") + "\n")
    print("  " + " " * 18 + "".join(f"{c:>19s}" for c in PAIR_COLS))
    o = build_paired(g.paired_specs)
    print(f"  wrote {o.relative_to(RESULT.parents[2])}")
    sys.exit(0)

print(f"\n  THE LADDER, h={g.horizon}, "
      + (f"locked baseline, source {SRC}" if BASE
         else f"{g.period}-1998, z_income_{g.share}, source {SRC}") + "\n")
print("  " + " " * 9 + "".join(f"{c:>19s}" for c in COLS))
outs = [build(s) for s in g.spec]
for o in outs:
    print(f"  wrote {o.relative_to(RESULT.parents[2])}")
