#!/usr/bin/env python3
"""Table 4: what accounts for the county-to-cell log gap, by horizon and rung.

Pure coefficient accounting. Every step is a county-level regression on one
sample, the steps add to the gap exactly, and there is no inference here.

TWO CHAINS, SHARING THEIR FIRST TWO STEPS. Both start at the log coefficient a
paper reports, the county's own total, and end at a disaggregated log
coefficient. They differ only in which weight the disaggregated regression uses.

    beta^b   log, county total, all industries
      + entry               -beta( log(E_ct / cc_ct) )
      + exit                +beta( log(E_c,t-h / lc_ct) )
    ---------  log, county total, survivors only
      + Jensen              -beta( C_ct ),  C = D_KL(w-hat^c || z)   eq jensen_share
    ---------  cell log at w-hat^c
      + renormalisation                                        -> cell log at w^-
      + within-unit bridge  \\ Lemma 1, eq implicit_weight_decomp
      + across-unit count   /                                  -> cell log at EQUAL

ENTRY AND EXIT ARE NEVER NETTED. L_surv - L_full = -ENTRY + EXIT is an exact
identity with both terms nonnegative, so the two halves are separately
identified. Netting them hides the case where each is enormous and the sum is
not, which is the common case at NAICS4.

WHY THE MIDDLE OF THE CHAIN IS w-hat^c AND NOT w^-. The Jensen identity is
stated over the survivor set m(i), so its weight must sum to one there. The step
from w-hat^c to w^- is the renormalisation, reported as its own term: it is what
happens when a county that lost employment to exits carries less total weight,
which is what an applied person's regression does without meaning to.

Run class: estimate_only. PRE-1998.
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
check, Tee, dc, RESULT_DIR, REPO_ROOT = md.check, md.Tee, md.dc, md.RESULT_DIR, md.REPO_ROOT

TOL_ID = 1e-12
TOL_ADD = 1e-9

STEPS = ("entry", "exit", "jensen", "renorm", "within", "across")


def one_cell(h, rung, share, level, frame_src="candidate", source="efsy"):
    p = md.Panel(h, rung, share, level, frame_src=frame_src, source=source)
    base = frame_src == "baseline"
    ok, m, cy = p.ok, p.m, p.cy
    g = lambda v: np.where(ok, v, 1.0)

    L_full = np.log(g(p.cur_c)) - np.log(g(p.lag_c))
    L_surv = np.log(g(p.cc)) - np.log(g(p.lc))
    ENTRY = np.log(g(p.cur_c)) - np.log(g(p.cc))
    EXIT = np.log(g(p.lag_c)) - np.log(g(p.lc))
    check(bool((ENTRY[ok] >= -1e-12).all() and (EXIT[ok] >= -1e-12).all()),
          f"ENTRY and EXIT are nonnegative (h={h}, {md.PRETTY[rung]})",
          f"min {ENTRY[ok].min():.2e} and {EXIT[ok].min():.2e}")
    check(float(np.abs((L_surv - L_full + ENTRY - EXIT)[ok]).max()) < TOL_ID,
          f"L_surv - L_full = -ENTRY + EXIT (h={h}, {md.PRETTY[rung]})",
          f"max |gap| = "
          f"{float(np.abs((L_surv - L_full + ENTRY - EXIT)[ok]).max()):.2e}")

    xbar = np.bincount(cy[m], weights=p.w_hat_c * p.log[m], minlength=p.n)
    C = L_surv - xbar
    check(bool((C[ok] >= -1e-12).all()),
          f"eq jensen_share  C = D_KL >= 0 (h={h}, {md.PRETTY[rung]})",
          f"min {C[ok].min():.2e}, mean {C[ok].mean():.4f}")

    if base:
        B = lambda y: p.fit_county_base(y, ok)["beta"]
        K = lambda w: p.fit_cell_base(p.log, w, m)["beta"]
    else:
        B = lambda y: p.fit_county(y, ok)["beta"]
        K = lambda w: p.fit_cell(p.log, w, m)["beta"]

    b_full, b_surv, b_xbar = B(L_full), B(L_surv), B(xbar)
    check(abs(K(p.w_hat_c) - b_xbar) < TOL_ADD,
          f"the cell fit at w-hat^c collapses to the xbar county fit "
          f"(h={h}, {md.PRETTY[rung]})",
          f"{K(p.w_hat_c):.10f} against {b_xbar:.10f}")
    b_naive, b_1n, b_eq = K(p.w_naive), K(p.w_1n), K(p.w_equal)

    s = dict(entry=-B(ENTRY), exit=B(EXIT), jensen=b_xbar - b_surv,
             renorm=b_naive - b_xbar, within=b_1n - b_xbar,
             across=b_eq - b_1n)
    gap_naive, gap_eq = b_naive - b_full, b_eq - b_full
    check(abs(s["entry"] + s["exit"] + s["jensen"] + s["renorm"] - gap_naive)
          < TOL_ADD,
          f"the w^- chain adds to its gap (h={h}, {md.PRETTY[rung]})",
          f"|residual| = "
          f"{abs(s['entry'] + s['exit'] + s['jensen'] + s['renorm'] - gap_naive):.2e}")
    check(abs(s["entry"] + s["exit"] + s["jensen"] + s["within"] + s["across"]
              - gap_eq) < TOL_ADD,
          f"the equal-weight chain adds to its gap (h={h}, {md.PRETTY[rung]})",
          f"|residual| = {abs(s['entry'] + s['exit'] + s['jensen'] + s['within'] + s['across'] - gap_eq):.2e}")

    if base:   # baseline z is per head: report in pp per $1,000/head (x 1e5)
        sc = 1e5
        b_full, b_surv, b_xbar = b_full * sc, b_surv * sc, b_xbar * sc
        b_naive, b_1n, b_eq = b_naive * sc, b_1n * sc, b_eq * sc
        s = {k: v * sc for k, v in s.items()}
        gap_naive, gap_eq = gap_naive * sc, gap_eq * sc
    # The window travels with the row: 44 prints it in the note, and each
    # (rung, h) can carry its own if a source moves the sample.
    yrs = p.frame["year"]
    return dict(h=h, rung=rung, source=source, county=b_full, cell_naive=b_naive,
                cell_equal=b_eq, cell_what=b_xbar, cell_1n=b_1n,
                county_surv=b_surv, gap_naive=gap_naive, gap_equal=gap_eq,
                net_ext=s["entry"] + s["exit"], **s,
                year_first=int(yrs.min()), year_last=int(yrs.max()),
                n_windows=int(yrs.nunique()),
                n_cy=int(ok.sum()), n_log_blind=p.n_log_blind)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--rung", nargs="+", default=list(md.RUNGS))
    ap.add_argument("--share", default="fixed5",
                    choices=["fixed4", "fixed5", "fixed10"])
    ap.add_argument("--level", type=float, default=5.0)
    ap.add_argument("--frame", default="candidate",
                    choices=["candidate", "baseline"])
    ap.add_argument("--source", default="efsy", choices=list(md.SOURCES),
                    help="which employment series the OUTCOME is built from; "
                         "see 34_tables_payoff.py --source")
    args = ap.parse_args()
    base = args.frame == "baseline"
    src = args.source
    stag = "" if src == "efsy" else f"_{src}"

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(dc.LOG_DIR / (f"35_table_decomposition_baseline{stag}.log" if base
                            else f"35_table_decomposition_pre{stag}_{args.share}.log"))
    import pyfixest as pf
    md.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
    t0 = time.time()

    print("=" * 88)
    print("  TABLE 4: accounting for the county-to-cell log gap")
    print("=" * 88)
    if base:
        print("  the locked baseline: harmonized top-800 composition instrument,")
        print("  span 5, positive rule, population denominator, reduced form,")
        print("  division-by-year effects, full era, no winsor; units pp per")
        print("  $1,000 of contracts per head (x 1e5).")
    else:
        print(f"  pre-{md.w22.NAICS_BREAK}, z_income_{args.share}, "
              f"IQR{args.level:g} by year, ccg + year FE, clustered on county.")
    print("  Point estimates only. Entry and exit are never netted.")

    if src != "efsy":
        print(f"  OUTCOME SOURCE: {src}. NETS begins in 1990, so each horizon's "
              f"first target year is 1990 + h,")
        print("  and the ccg support is the EFSY-NETS intersection less the "
              "A.3.4 degenerates.")

    rows = [one_cell(h, r, args.share, args.level, frame_src=args.frame,
                     source=src)
            for r in args.rung for h in args.horizon]

    for weight, gapk, steps in (
            ("w^- (naive lagged share)", "gap_naive",
             ("entry", "exit", "jensen", "renorm")),
            ("EQUAL weight (beta^d_OLS)", "gap_equal",
             ("entry", "exit", "jensen", "within", "across"))):
        cellk = "cell_naive" if gapk == "gap_naive" else "cell_equal"
        print(f"\n{'=' * 88}\n  log county  ->  log county x industry at {weight}"
              f"\n{'=' * 88}")
        print(f"  {'rung':8s}{'h':>3s}{'county':>10s}{'cell':>10s}{'GAP':>10s}"
              f"{'   |':4s}" + "".join(f"{s:>11s}" for s in steps))
        for r in rows:
            print(f"  {r['rung'].replace('_alloc',''):8s}{r['h']:>3d}"
                  f"{r['county']:10.4f}{r[cellk]:10.4f}{r[gapk]:+10.4f}"
                  f"{'   |':4s}" + "".join(f"{r[s]:+11.4f}" for s in steps))
        print(f"\n  net extensive (entry + exit), never used in place of the two:")
        print(f"  {'':11s}" + "".join(
            f"{r['rung'].replace('_alloc','')}/h{r['h']}={r['net_ext']:+.4f}  "
            for r in rows))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / (f"coeff_table_decomposition_baseline{stag}.csv" if base
                        else f"coeff_table_decomposition_pre{stag}_{args.share}.csv")
    pd.DataFrame(rows).assign(period="baseline" if base else "pre",
                              share=args.share,
                              fe="divisionyear" if base else "ccg+year").to_csv(
        out, index=False, float_format="%.15g")
    print(f"\n  wrote {out.relative_to(REPO_ROOT)}")

    n_fail = len(md.w22.failures)
    print(f"\n{'=' * 88}\n  {n_fail} checks failed, {time.time() - t0:.1f}s"
          f"\n{'=' * 88}")
    tee.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
