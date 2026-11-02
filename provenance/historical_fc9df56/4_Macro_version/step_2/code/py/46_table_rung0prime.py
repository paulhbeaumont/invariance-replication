#!/usr/bin/env python3
"""The midpoint ladder at both shock-control rungs, both employment sources.

Table 1 is estimated at rung 0 -- no shock-level control -- which is what the
locked baseline has always carried and what makes its EFSY cells reproduce
numbers already reported. `ws0_contingency.py` records Adrien choosing rung 0'
as the reference rung for the shift-share and local-projection producers, and
rung 0' is rung 0 plus the county share-sum LEVEL S0_ct, the cross-sectional
form of BHJ's incomplete-shares control. The two answers differ by about 22% on
the county beta in producer 22's setting, so which rung Table 1 sits at is a
question this exhibit answers rather than assumes.

WHAT THIS TABLE IS FOR, and what it is not. It is NOT a claim that rung 0' is
the right rung for the disaggregation exhibits. It reports what the ladder looks
like at each rung so the choice can be made on the numbers. The claim under test
is unchanged and is the same at both rungs: midpoint growth at MPGR weights is
invariant to how finely the outcome is split. A control that is constant within
a county-year is repeated across that county's cells exactly as the instrument
is, so partialling it out cannot break the weighting identity -- and the run
asserts that rather than resting on it.

RUNG 0' HERE IS A NEW OBJECT, and the note says so. Producer 22 evaluates rung
0' on its own frame: employment-share weights within division-year, and a
winsorized treatment. This runs it on the locked baseline's frame and weights
(equal across counties, no winsorization), which is the frame Tables 1-6 use.
The two rung-0' numbers are therefore not the same estimate and neither
supersedes the other.

Reads nothing but the frame and the cells; writes one exhibit.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
_spec = importlib.util.spec_from_file_location("mpgr_disagg",
                                               HERE.parent / "mpgr_disagg.py")
md = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(md)
from out_paths import write_exhibit  # noqa: E402

check, Tee, dc = md.check, md.Tee, md.dc
RESULT_DIR, REPO_ROOT = md.RESULT_DIR, md.REPO_ROOT

SCALE = 1e5
RUNGS = ("0", "0'")
COLS = ("county", "n2", "n4", "n4_iy")
COLHEAD = {"county": "County", "n2": r"$\times$ NAICS2", "n4": r"$\times$ NAICS4",
           "n4_iy": r"$\times$ NAICS4 $+\,i.y$"}
SOURCE_LABEL = {"efsy": "CBP (EFSY)", "nets_actual": "NETS, actual",
                "nets_all": "NETS, all"}


def stars(b, se):
    if not (se > 0):
        return ""
    t = abs(b / se)
    return (r"\sym{***}" if t > 2.576 else r"\sym{**}" if t > 1.96 else
            r"\sym{*}" if t > 1.645 else "")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--source", nargs="+", default=["efsy", "nets_actual"],
                    choices=list(md.SOURCES))
    args = ap.parse_args()
    h = args.horizon

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(dc.LOG_DIR / f"46_table_rung0prime_h{h}.log")
    import pyfixest as pf
    md.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
    t0 = time.time()

    print("=" * 84)
    print(f"  THE MIDPOINT LADDER AT RUNG 0 AND RUNG 0', h = {h}")
    print("=" * 84)
    print("  rung 0  = the locked baseline, no shock-level control")
    print("  rung 0' = rung 0 + the county share-sum level S0_ct, partialled")
    print("            out of the outcome and the instrument by FWL")

    rows = []
    for src in args.source:
        panels = {r: md.Panel(h, r, frame_src="baseline", source=src)
                  for r in md.RUNGS}
        p4 = panels["naics4_alloc"]
        for rung in RUNGS:
            allc = {r: p.cy >= 0 for r, p in panels.items()}
            den = p4.cur_c + p4.lag_c
            mc = den != 0.0
            cty = p4.fit_county_base(p4.mpgr_c, mc, rung=rung)
            fits = {"county": cty}
            for key, r in (("n2", "naics2"), ("n4", "naics4_alloc")):
                p = panels[r]
                fits[key] = p.fit_cell_base(p.mpgr, p.w_mpgr[allc[r]], allc[r],
                                            indyear=False, rung=rung)
                if key == "n4":
                    fits["n4_iy"] = p.fit_cell_base(
                        p.mpgr, p.w_mpgr[allc[r]], allc[r], indyear=True,
                        rung=rung)
            row = dict(source=src, rung=rung, horizon=h,
                       year_first=int(p4.frame["year"].min()),
                       year_last=int(p4.frame["year"].max()))
            for k, f in fits.items():
                row[k] = f["beta"] * SCALE
                row[f"se_{k}"] = f["se"] * SCALE
                row[f"n_{k}"] = int(f["n"])
            rows.append(row)

            # The claim, at BOTH rungs. A county-constant control repeated
            # across cells cannot move it, and that is checked, not assumed.
            for c in ("n2", "n4"):
                gap = abs(row[c] - row["county"])
                check(gap < md.t28.TOL_INVAR,
                      f"Prop 1  midpoint/MPGR invariant at rung {rung}, "
                      f"county to {c} ({src})",
                      f"|gap| = {gap:.2e}; {row[c]:.10f} against "
                      f"{row['county']:.10f}")
            print(f"\n  {src:12s} rung {rung:2s}  " + "  ".join(
                f"{COLHEAD[c].replace(chr(92),'')[:12]:>14s}" for c in COLS))
            print(f"  {'':12s} {'':7s}  " + "  ".join(
                f"{row[c]:14.4f}" for c in COLS))

    out = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv = RESULT_DIR / f"coeff_table_rung0prime_h{h}.csv"
    out.to_csv(csv, index=False, float_format="%.15g")

    # ------------------------------------------------------------------ table
    K = len(COLS)
    body = []
    for src in args.source:
        body.append(r"\addlinespace" if body else "")
        body.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (K + 1,
                                                             SOURCE_LABEL[src]))
        for rung in RUNGS:
            r = out[(out["source"] == src) & (out["rung"] == rung)].iloc[0]
            lab = ("Rung 0, no shock control" if rung == "0"
                   else r"Rung 0$'$, $+$ share-sum level")
            body.append(r"\hspace{1.5em}" + lab + " & " + " & ".join(
                f"{float(r[c]):.4f}{stars(float(r[c]), float(r['se_' + c]))}"
                for c in COLS) + r" \\")
            body.append(r"\hspace{1.5em} & " + " & ".join(
                f"({float(r['se_' + c]):.4f})" for c in COLS) + r" \\")
    body = [x for x in body if x != ""]

    win = (f"county-years {int(out['year_first'].min())}--"
           f"{int(out['year_last'].max())}")
    tex = "\n".join([
        r"% The midpoint ladder at rung 0 and rung 0'.",
        r"% Rendered by 46_table_rung0prime.py from its own run. Do not edit.",
        r"\begin{table}[htbp]",
        r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
        r"\caption{The invariance holds at either shock-control rung}",
        r"\label{tab:rung0prime}",
        r"{\centering",
        r"\adjustbox{max width=\linewidth}{",
        r"\begin{tabular}{l%s}" % ("c" * K),
        r"\toprule",
        r"\textit{Dependent variable} & \multicolumn{%d}{c}{Midpoint growth of "
        r"employment, $h=%d$} \\" % (K, h),
        r"\cmidrule(r){2-%d}" % (K + 1),
        r"\textit{Level of aggregation} & " + " & ".join(
            COLHEAD[c] for c in COLS) + r" \\",
        r"& " + " & ".join(f"({i + 1})" for i in range(K)) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\par}",
        r"\medskip",
        r"{\scriptsize \textit{Notes:} Each block repeats the midpoint ladder of "
        r"Table~\ref{tab:midpoint_ladder} at the two shock-control rungs. Rung 0 "
        r"carries no shock-level control and is the specification Tables 1--6 "
        r"report. Rung 0$'$ adds the county share-sum level $S_{ct}$, the "
        r"cross-sectional form of the incomplete-shares control, partialled out "
        r"of both the outcome and the instrument. The control is constant within "
        r"a county-year and is repeated across that county's cells exactly as the "
        r"instrument is, so it cannot disturb the weighting identity: columns "
        r"(1) to (3) are flat at both rungs, and the run asserts it. Rung 0$'$ "
        r"is estimated here on the locked baseline's frame and weights, which is "
        r"not the frame the shift-share diagnostics evaluate it on; the two "
        r"rung-0$'$ estimates are different objects and neither supersedes the "
        r"other. Estimation is the reduced form on the harmonized top-800 "
        r"composition instrument, a five-year share window under the "
        r"positive-contract rule and a population denominator, division-by-year "
        r"effects, %s, no winsorization anywhere. Coefficients are in percentage "
        r"points per \$1{,}000 of contracts per head. Standard errors clustered "
        r"on the county in parentheses; $^{*}$, $^{**}$ and $^{***}$ denote "
        r"significance at the 10, 5 and 1 percent level. \par}" % win,
        r"\end{table}",
        "",
    ])
    path, _ = write_exhibit("Table_nets_rung0prime.tex", tex, quiet=True,
                            subdir="nets_baseline")

    n_fail = len(md.w22.failures)
    print(f"\n  wrote {csv.relative_to(REPO_ROOT)}")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
    print(f"\n{'=' * 84}\n  {n_fail} checks failed, {time.time() - t0:.1f}s"
          f"\n{'=' * 84}")
    tee.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
