#!/usr/bin/env python3
"""The log county-to-cell gap at w^-, over horizons and cell fineness.

Under the paper's own weight the gap between the baseline log coefficient and
the disaggregated log coefficient has exactly two sources, and this file reports
them separately over {h = 2, 3, 4} x {naics2, naics4_alloc}.

THE DECOMPOSITION, and why entry and exit can be split. Write the county's log
growth on the full total and on the survivor cells only:

    L^full_ct = log(E_ct) - log(E_c,t-h)
    L^surv_ct = log(cc_ct) - log(lc_ct)          cc, lc summed over survivors

Their difference is an exact sum of two nonnegative pieces:

    L^surv - L^full = -log(E_ct / cc_ct) + log(E_c,t-h / lc_ct)
                    = -ENTRY_ct          + EXIT_ct

ENTRY_ct is the log gross-up of CURRENT county employment contributed by cells
that were empty at t-h. EXIT_ct is the log gross-up of LAGGED employment
contributed by cells that are empty at t. Both are >= 0 by construction, and
because the regression is linear in the outcome their coefficients enter the
gap with the signs above. So the net extensive margin is reported alongside its
two halves, and a net near zero can be told apart from an extensive margin that
is genuinely small.

The second source is the paper's Jensen wedge. On the survivor set,
3_disaggregation.tex eq jensen and eq jensen_share give

    L^surv_ct = xbar_ct + C_ct,   C_ct = D_KL(w^-_c || z_ct) >= 0

with w^-_im = E_im,t-h / lc_ct normalised over the survivor set m(i), which is
the set the identity is stated over. xbar is the disaggregated coefficient at
w^-, so C enters the gap with a minus sign.

    GAP = beta(xbar) - beta(L^full)
        = [ -beta(ENTRY) + beta(EXIT) ]  +  [ -beta(C) ]
             net extensive margin              Jensen

All three identities are asserted numerically before anything is reported.

Run class: estimate_only. Frame, instrument and winsorisation are 29's.
PRE-1998 ONLY, and w^- throughout: MPGR weights appear nowhere in this file.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
STEP2_CODE = HERE.parent
STEP1_CODE = HERE.parents[3] / "step_1" / "code" / "py"
REPO_ROOT = HERE.parents[4]
RESULT_DIR = HERE.parents[2] / "result"

sys.path.insert(0, str(STEP1_CODE))
import dcads_common as dc  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t29 = _load("candidate_mpgr", STEP2_CODE / "29_candidate_iv_mpgr_check.py")
t28, c27, w22 = t29.t28, t29.c27, t29.w22
check, Tee = w22.check, w22.Tee

TOL_ID = 1e-12      # the Jensen identity, in the data
TOL_ADD = 1e-9      # additivity of the coefficients
PRETTY = {"naics2": "NAICS2", "naics4_alloc": "NAICS4"}


def one_cell(h, rung, share, level):
    frame = t29.build_frame("pre", h, share, level)
    shock, zi = frame["shock"].to_numpy(), frame["z"].to_numpy()
    ccg, yr = frame["ccg"].to_numpy(), frame["year"].to_numpy()
    n = len(frame)
    years = sorted(set(frame["year"]) | set(frame["year"] - h))

    emp, ccg_arr = w22.load_rung(rung, years)
    _, code_arr = c27.rung_codes(rung, years[0])
    keep = np.isin(ccg_arr, frame["ccg"].unique())
    if not keep.all():
        emp = {y: v[keep] for y, v in emp.items()}
        ccg_arr, code_arr = ccg_arr[keep], code_arr[keep]

    d = t28.build_long(rung, emp, ccg_arr, code_arr, frame, h)
    mpgr, _, log, pos = t28.cell_outcomes(d)
    cy, cur, lag = d["f_cy"], d["cur"], d["lag"]
    cur_c, lag_c = d["cur_c"], d["lag_c"]

    cc = np.bincount(cy[pos], weights=cur[pos], minlength=n)
    lc = np.bincount(cy[pos], weights=lag[pos], minlength=n)
    ok = (cc > 0) & (lc > 0) & (cur_c > 0) & (lag_c > 0)
    m = pos & ok[cy]

    g = lambda v: np.where(ok, v, 1.0)
    L_full = np.log(g(cur_c)) - np.log(g(lag_c))
    L_surv = np.log(g(cc)) - np.log(g(lc))
    ENTRY = np.log(g(cur_c)) - np.log(g(cc))     # >= 0
    EXIT = np.log(g(lag_c)) - np.log(g(lc))      # >= 0
    check(bool((ENTRY[ok] >= -1e-12).all() and (EXIT[ok] >= -1e-12).all()),
          f"ENTRY and EXIT are nonnegative (h={h}, {rung})",
          f"min ENTRY {ENTRY[ok].min():.2e}, min EXIT {EXIT[ok].min():.2e}")
    check(float(np.abs((L_surv - L_full + ENTRY - EXIT)[ok]).max()) < TOL_ID,
          f"L_surv - L_full = -ENTRY + EXIT (h={h}, {rung})",
          f"max |gap| = "
          f"{float(np.abs((L_surv - L_full + ENTRY - EXIT)[ok]).max()):.2e}")

    wlag = lag[m] / lc[cy[m]]
    xbar = np.bincount(cy[m], weights=wlag * log[m], minlength=n)
    C = L_surv - xbar
    check(float(np.abs(np.bincount(cy[m], weights=wlag, minlength=n)[ok]
                       - 1.0).max()) < 1e-12,
          f"w^- sums to one on the survivor set (h={h}, {rung})",
          f"max |sum w^- - 1| = "
          f"{float(np.abs(np.bincount(cy[m], weights=wlag, minlength=n)[ok] - 1.0).max()):.2e}")
    check(bool((C[ok] >= -1e-12).all()),
          f"eq jensen_share  C_it = D_KL >= 0 (h={h}, {rung})",
          f"min C = {C[ok].min():.2e}, mean {C[ok].mean():.4f}")

    S = lambda v: v[ok]
    one = np.ones(int(ok.sum()))
    F = lambda y: t28.fit2sls(S(y), S(shock), S(zi), one, S(ccg), S(yr), S(ccg),
                              singleton=False)["beta"]
    b_full, b_xbar = F(L_full), F(xbar)
    b_entry, b_exit, b_C = -F(ENTRY), F(EXIT), -F(C)
    gap = b_xbar - b_full
    check(abs(b_entry + b_exit + b_C - gap) < TOL_ADD,
          f"GAP = entry + exit + Jensen, no residual (h={h}, {rung})",
          f"|sum - gap| = {abs(b_entry + b_exit + b_C - gap):.2e}")

    return dict(h=h, rung=rung, county=b_full, cell=b_xbar, gap=gap,
                net_ext=b_entry + b_exit, entry=b_entry, exit=b_exit,
                jensen=b_C, C_wedge=-b_C, n_cy=int(ok.sum()),
                n_dropped=int((~ok).sum()), mean_C=float(C[ok].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--rung", nargs="+", default=["naics2", "naics4_alloc"],
                    choices=list(w22.RUNGS_ALL))
    ap.add_argument("--share", default="fixed5",
                    choices=["fixed4", "fixed5", "fixed10"])
    ap.add_argument("--level", type=float, default=5.0)
    args = ap.parse_args()

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(dc.LOG_DIR / f"33_log_gap_grid_pre_{args.share}.log")
    import pyfixest as pf
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
    t0 = time.time()

    print("=" * 80)
    print("  LOG GROWTH AT w^-: the county-to-cell gap and its two sources")
    print("=" * 80)
    print(f"  pre-{w22.NAICS_BREAK}, instrument z_income_{args.share}, "
          f"IQR{args.level:g} by year, ccg + year FE, clustered on county")
    print("  w^- = cell share of LAGGED county employment, normalised over the")
    print("  survivor set. MPGR weights appear nowhere in this table.")
    print("  GAP = entry + exit + Jensen, exactly. Jensen = -beta(C_it).")
    print("  C_it is nonnegative for every county-year, but its COEFFICIENT is a")
    print("  treated-control difference and can take either sign; 3_disaggregation")
    print("  .tex says so directly under eq jensen_wedge.")

    rows = [one_cell(h, r, args.share, args.level)
            for r in args.rung for h in args.horizon]

    for r in args.rung:
        sub = [x for x in rows if x["rung"] == r]
        print(f"\n{'=' * 80}\n  county x {PRETTY[r]}\n{'=' * 80}")
        print(f"  {'':28s}" + "".join(f"{'h=' + str(x['h']):>14s}" for x in sub))
        for lab, key in (("log county", "county"),
                         (f"log county x {PRETTY[r]}", "cell"),
                         ("GAP", "gap"),
                         ("  net extensive", "net_ext"),
                         ("     entry", "entry"),
                         ("     exit", "exit"),
                         ("  Jensen (-C)", "jensen")):
            rule = "  " + "-" * 70 + "\n" if lab == "GAP" else ""
            print(f"{rule}  {lab:28s}"
                  + "".join(f"{x[key]:14.4f}" for x in sub))
        print(f"  {'':28s}" + "".join(f"{'':>14s}" for x in sub))
        print(f"  {'mean C_it in the data':28s}"
              + "".join(f"{x['mean_C']:14.4f}" for x in sub))
        print(f"  {'county-years':28s}"
              + "".join(f"{x['n_cy']:14,}" for x in sub))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f"coeff_log_gap_grid_pre_{args.share}.csv"
    pd.DataFrame(rows).assign(period="pre", share=args.share, fe="ccg+year",
                              weight="w_minus_lagged_share").to_csv(
        out, index=False, float_format="%.15g")
    print(f"\n  wrote {out.relative_to(REPO_ROOT)}")

    n_fail = len(w22.failures)
    print(f"\n{'=' * 80}\n  {n_fail} checks failed, {time.time() - t0:.1f}s"
          f"\n{'=' * 80}")
    tee.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
