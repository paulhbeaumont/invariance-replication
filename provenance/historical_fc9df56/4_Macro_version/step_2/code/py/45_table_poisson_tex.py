#!/usr/bin/env python3
"""Table 6 as LaTeX: the disaggregation ladder under Poisson on levels.

Renders from the saved coefficient file and adds no estimate, exactly as
43_tables_tex.py does for Tables 1 to 4. That file carries the coefficient, its
clustered standard error and its fitted N per column, so this table cannot
disagree with the numbers it is supposed to typeset.

FOUR COLUMNS, AND THE SAME FOUR. Table 6 prints county, NAICS2, NAICS4 and
NAICS4 with an industry-by-year effect, which is column for column what Tables 1
to 4 print, and its FIXED-EFFECT BLOCK IS IDENTICAL TO THEIRS. That is the whole
point of absorbing `ccg` rather than A.1.5's `ccg x industry`: the reader holds a
column fixed, turns the page from Table 1 to Table 6, and the only thing that has
changed is the estimator. The NAICS2 industry-by-year estimate and the two
`cellfe` robustness fits stay in the coefficient file and are not printed.

READ IT AGAINST TABLE 1, NOT AGAINST TABLES 2 TO 4. Both are admissible answers
under A.1.5 and both are flat across columns (1), (2) and (3). They are NOT two
estimates of one number - Table 1's coefficient is on a growth rate and Table 6's
is a semi-elasticity on levels - so the two are read for their FLATNESS, which is
the claim, and never for the ratio of their magnitudes, which means nothing.
"""
import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from out_paths import write_exhibit  # noqa: E402

RESULT = Path(__file__).resolve().parents[2] / "result"
FLAT_TOL = 1e-9   # two IRLS limits against each other, not one closed form

# Keys in the coefficient file, in printed order. `n2_iy`, `n2_cellfe` and
# `n4_cellfe` are estimated and deliberately absent.
COLS = ("county", "n2", "n4", "n4_iy")

N = 6
LABEL = "poisson_ladder"
CAPTION = ("Poisson on employment levels is invariant to how finely the outcome "
           "is split")

a = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument("--horizon", type=int, default=4)
a.add_argument("--share", default="fixed5")
a.add_argument("--period", default="pre",
               help="pre / post, or 'baseline' for the locked baseline's "
                    "reduced-form mapping")
a.add_argument("--source", default="efsy",
               choices=["efsy", "nets_actual", "nets_all"],
               help="which employment series the coefficient file was\n"
                    "                     estimated on; see 43_tables_tex.py")
g = a.parse_args()

BASE = g.period == "baseline"
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

co = pd.read_csv(
    RESULT / f"coeff_table_poisson_{g.period}{STAG}_h{g.horizon}_{g.share}.csv")
r = co.iloc[0]

# The estimation window is READ, never restated, for the reason 43 records: a
# literal era in the note is right for one period and silently wrong for the other.
WINDOW = f"county-years {int(r['year_first'])}--{int(r['year_last'])}"


# The note, assembled once so the %-substitution reaches the whole paragraph.
# Any literal per-cent sign in these strings must be doubled.
_NOTE_HEAD = (
    r"{\scriptsize \textit{Notes:} The dependent variable is employment in "
    r"levels, estimated by Poisson pseudo-maximum likelihood, over a "
    r"%d-year window. This is A.1.5's second admissible answer and is the "
    r"companion to Table~\ref{tab:midpoint_ladder}: both are flat across "
    r"columns (1) to (3), which is the claim. \textbf{The two coefficients are "
    r"not on the same scale} --- a Poisson coefficient on levels is a "
    r"semi-elasticity while Table~\ref{tab:midpoint_ladder}'s is a growth-rate "
    r"coefficient --- so the tables are read for their flatness and never for "
    r"the ratio of their magnitudes. Column (1) is the county; columns (2) and "
    r"(3) split the same outcome into county-by-industry cells at two and four "
    r"NAICS digits, and column (4) adds an industry-by-year effect. The "
    r"treatment is the county-level contract shock and is never reclassified to "
    r"the cell. Every cell that exists in a county enters, exits at zero "
    r"employment and entries alike, because zero is a valid Poisson outcome; "
    r"county-industry pairs with no employment in any window are excluded by the "
    r"estimator itself and carry no employment. The proposition also holds at "
    r"the county-by-industry effect A.1.5 names in place of the county effect, "
    r"to nine digits, and that estimate is in the coefficient file. ")
_NOTE_TAIL = (r"Standard errors clustered on the county in parentheses; $^{*}$, "
              r"$^{**}$ and $^{***}$ denote significance at the 10, 5 and 1 "
              r"percent level. \par}")

ESTIMATION_CANDIDATE = (
    r"Estimation is on the income-based instrument entered as a control "
    r"function, because Poisson admits no instrumental-variables estimator "
    r"here; \textbf{the standard errors treat that control function as data "
    r"and therefore understate}. %s, non-overlapping windows, with the shock "
    r"and the instrument winsorized at five interquartile ranges by year and "
    r"no cell dropped. ")

# The locked baseline's mapping. A reduced form has no first stage, so the
# control function disappears with it -- and with it the standard-error caveat,
# which was a property of the control function and not of Poisson.
#
# The unit effect is whatever 37 was told to use and is read back from the
# coefficient file, because the two choices are not interchangeable here:
# Proposition 1 needs a free unit effect to absorb log M_c, so `ccg` keeps the
# invariance (measured gap 4e-19) and `divisionyear`, being coarser than the
# county, does not (2.5e-07 against a 1e-9 tolerance). The note therefore says
# which one was run AND what that costs, rather than naming one of them.
UNIT_FE = str(r.get("unit_fe", "ccg")) if BASE else "ccg"
UNIT_ROW = (r"Division $\times$ year" if UNIT_FE == "divisionyear"
            else "County")
UNIT_SENTENCE = {
    "ccg": r"county and year effects --- the county effect is kept rather than "
           r"the division-by-year effect of Tables 1--5, because Proposition 1 "
           r"here needs a free unit effect to absorb $\log M_c$ and an effect "
           r"coarser than the county cannot, so the ladder would not be flat "
           r"under division-by-year",
    "divisionyear":
        r"division-by-year effects, matching Tables 1--5 literally --- note "
        r"that this gives up the invariance claim, because Proposition 1 needs "
        r"a free unit effect to absorb $\log M_c$ and an effect coarser than "
        r"the county cannot provide one; the columns are therefore NOT expected "
        r"to be flat and their spread is not evidence about the estimand",
}[UNIT_FE]

ESTIMATION_BASE = (
    r"Estimation is the locked baseline's mapping: the reduced form on the "
    r"harmonized top-800 composition instrument, with no control function, a "
    r"five-year share window under the positive-contract rule and a population "
    r"denominator, " + UNIT_SENTENCE + r", %s, overlapping windows, and no "
    r"winsorization anywhere. The coefficient is therefore a reduced-form "
    r"semi-elasticity and is not comparable to the control-function estimates "
    r"of the candidate design. ")

NOTE = (_NOTE_HEAD + SOURCE_NOTE
        + (ESTIMATION_BASE if BASE else ESTIMATION_CANDIDATE) + _NOTE_TAIL)


def stars(b, se):
    r"""The project's convention, from 22_winsor_construction_error.py: normal
    critical values on the clustered t, and the \sym macro esttab emits."""
    if not (se > 0):
        return ""
    t = abs(b / se)
    return (r"\sym{***}" if t > 2.576 else r"\sym{**}" if t > 1.96 else
            r"\sym{*}" if t > 1.645 else "")


def gap(c):
    """Below FLAT_TOL the gap prints as a bound. A printed 1.3e-11 invites the
    reader to read the distance between two IRLS limits as a magnitude."""
    if c == "county":
        return ""
    d = float(r[c]) - float(r["county"])
    return r"$<10^{-9}$" if abs(d) < FLAT_TOL else f"{d:+.4f}"


nobs = {c: int(r[f"n_{c}"]) for c in COLS}
K = len(COLS)

rows = [
    r"\textit{Dependent variable} & \multicolumn{%d}{c}{Employment in levels, "
    r"$h=%d$} \\" % (K, g.horizon),
    r"\cmidrule(r){2-%d}" % (K + 1),
    r"\textit{Level of aggregation} & County & County $\times$ NAICS2"
    r" & \multicolumn{2}{c}{County $\times$ NAICS4} \\",
    r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-5}",
    r"& " + " & ".join(f"({i + 1})" for i in range(K)) + r" \\",
    r"\midrule",
    r"Contract shock & " + " & ".join(
        f"{float(r[c]):.4f}{stars(float(r[c]), float(r['se_' + c]))}"
        for c in COLS) + r" \\",
    r"& " + " & ".join(f"({float(r['se_' + c]):.4f})" for c in COLS) + r" \\",
    r"\hspace{1.5em}\textit{gap from column (1)} & "
    + " & ".join(gap(c) for c in COLS) + r" \\",
    r"\\[-1.8ex]\hline\addlinespace\textit{Fixed Effects} \\",
    # READ the unit effect, never guess it. On the baseline frame 37 takes it
    # from --unit-fe, which has no default because the two choices differ on
    # whether the invariance claim survives at all; a renderer that assumed one
    # of them would silently mislabel the other. The candidate frame has only
    # ever had one answer, so its rows are literal.
    *([r"\hspace{1.5em}" + UNIT_ROW + " & "
       + " & ".join([r"$\checkmark$"] * K) + r" \\"]
      + ([] if UNIT_FE == "divisionyear" else
         [r"\hspace{1.5em}Year & $\checkmark$ & $\checkmark$ & $\checkmark$ "
          r"& --- \\"])
      + [r"\hspace{1.5em}Industry $\times$ year & --- & --- & --- "
         r"& $\checkmark$ \\"]
      if BASE else
      [r"\hspace{1.5em}County & " + " & ".join([r"$\checkmark$"] * K) + r" \\",
       r"\hspace{1.5em}Year & $\checkmark$ & $\checkmark$ & $\checkmark$ & --- \\",
       r"\hspace{1.5em}Industry $\times$ year & --- & --- & --- & $\checkmark$ \\"]),
    r"\addlinespace",
    r"Observations & " + " & ".join(f"{nobs[c]:,}" for c in COLS) + r" \\",
    r"\bottomrule",
]

tex = "\n".join([
    r"%% Table %d. %s" % (N, CAPTION),
    r"% Rendered by 45_table_poisson_tex.py from the saved coefficient file. Do not edit.",
    r"\begin{table}[htbp]",
    r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
    r"\caption{%s}" % CAPTION,
    r"\label{tab:%s}" % LABEL,
    r"{\centering",
    r"\adjustbox{max width=\linewidth}{",
    r"\begin{tabular}{l%s}" % ("c" * K),
    r"\toprule",
    *rows,
    r"\end{tabular}",
    r"}",
    r"\par}",
    r"\medskip",
    NOTE % (g.horizon, WINDOW),
    r"\end{table}",
    "",
])

name = (f"Table{N}_{LABEL}_baseline{STAG}_h{g.horizon}.tex" if BASE else
        f"Table{N}_{LABEL}_{g.period}{STAG}_h{g.horizon}_{g.share}.tex")
out, _ = write_exhibit(name, tex, quiet=True, subdir=SUBDIR)

print(f"\n  TABLE 6, POISSON ON LEVELS, h={g.horizon}, "
      + (f"locked baseline, source {SRC}" if BASE
         else f"{g.period}-1998, z_income_{g.share}, source {SRC}") + "\n")
print("  " + " " * 9 + "".join(f"{c:>19s}" for c in COLS))
print("  Table 6  " + "  ".join(
    f"{float(r[c]):8.4f} ({float(r['se_' + c]):.4f})" for c in COLS))
print(f"\n  wrote {out.relative_to(RESULT.parents[2])}")
