#!/usr/bin/env python3
"""County employment growth split into its continuer, entry and exit margins.

Midpoint growth reconstructs the county exactly, `mpgr_i = sum_m w_im mpgr_im`,
so the county outcome can be partitioned by what happened to the CELL and the
pieces still add to the county number. The partition is exhaustive and mutually
exclusive on (E_im,t-h, E_imt):

    continuers   E_im,t-h > 0 and E_imt > 0     the intensive margin
    entry        E_im,t-h <= 0 and E_imt > 0    mpgr_im = +2, so the piece is >= 0
    exit         E_im,t-h > 0 and E_imt <= 0    mpgr_im = -2, so the piece is <= 0
    always zero  neither positive               w_im = 0, contributes exactly nothing

WHY THIS AND NOT THE LOG COMPARISON. Reading the extensive margin off the gap
between county log growth and the mean of cell log growths does not work, because
that gap also carries the Jensen difference between the log of a sum and the mean
of logs ON THE CONTINUERS. Two channels, one number. Here each margin is its own
county-level variable and its own regression, and because the regression is linear
in the outcome the three betas ADD to the total. That is asserted, not hoped for.

A regression on each piece answers one question: how much of the county employment
response to the contract shock arrives through cells dying, through cells being
born, and through cells that were alive throughout.

Run class: estimate_only on the cell construction 29 rebuilt. The frame, the
instrument and the winsorization are 29's, imported rather than restated.

PRE-1998 ONLY. `w22.NAICS_BREAK` is where CBP goes native NAICS and EFSY stops
imputing from SIC, so a cell is not the same object on the two sides of it.
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

HORIZONS = (2, 3, 4)
SHARE, LEVEL, RUNG = "fixed5", 5.0, "naics4_alloc"
MARGINS = ("continuers", "entry", "exit", "always zero")
TOL_IDENTITY = 1e-12     # components against county midpoint growth, absolute
TOL_ADDITIVITY = 1e-9    # component betas against the total beta


def partition(cur, lag):
    """Exhaustive and mutually exclusive. `<= 0` on the base, not `== 0`, because
    EFSY carries desuppression artifacts at or below zero and A.3.1 forbids
    dropping them; they belong to a named margin rather than to a silent residual.
    """
    m = {"continuers": (lag > 0) & (cur > 0),
         "entry": (lag <= 0) & (cur > 0),
         "exit": (lag > 0) & (cur <= 0),
         "always zero": (lag <= 0) & (cur <= 0)}
    stacked = np.vstack([m[k] for k in MARGINS])
    check(bool(stacked.sum(axis=0).min() == 1 and stacked.sum(axis=0).max() == 1),
          "the margin partition is exhaustive and mutually exclusive",
          f"{len(cur):,} cell rows, each in exactly one of {list(MARGINS)}")
    return m


def one_horizon(h, args):
    frame = t29.build_frame("pre", h, args.share, args.level)
    shock, zinst = frame["shock"].to_numpy(), frame["z"].to_numpy()
    ccg, yr = frame["ccg"].to_numpy(), frame["year"].to_numpy()
    n = len(frame)
    years = sorted(set(frame["year"]) | set(frame["year"] - h))
    print(f"    employment files opened: {years}")

    emp, ccg_arr = w22.load_rung(args.rung, years)
    _, code_arr = c27.rung_codes(args.rung, years[0])
    keep = np.isin(ccg_arr, frame["ccg"].unique())
    if not keep.all():
        emp = {y: v[keep] for y, v in emp.items()}
        ccg_arr, code_arr = ccg_arr[keep], code_arr[keep]

    d = t28.build_long(args.rung, emp, ccg_arr, code_arr, frame, h)
    mpgr, w, _, _ = t28.cell_outcomes(d)
    cy, cur, lag = d["f_cy"], d["cur"], d["lag"]

    den_c = d["cur_c"] + d["lag_c"]
    check(bool((den_c != 0).all()),
          f"A.1.6  every county-year has a midpoint denominator (h={h})",
          f"{int((den_c == 0).sum())} county-years with den_c = 0")
    mpgr_c = 2.0 * (d["cur_c"] - d["lag_c"]) / den_c

    contrib = w * mpgr
    masks = partition(cur, lag)
    pieces = {k: np.bincount(cy[m], weights=contrib[m], minlength=n)
              for k, m in masks.items()}
    worst = float(np.abs(sum(pieces.values()) - mpgr_c).max())
    check(worst < TOL_IDENTITY,
          f"A.1.2  the margins sum to county midpoint growth (h={h})",
          f"max |sum of margins - mpgr_c| = {worst:.2e} against "
          f"{TOL_IDENTITY:.0e} over {n:,} county-years")

    fit = lambda y: t28.fit2sls(y, shock, zinst, np.ones(n), ccg, yr, ccg,
                                singleton=False)
    total = fit(mpgr_c)
    rows = []
    for k in MARGINS:
        r = fit(pieces[k])
        rows.append(dict(horizon=h, margin=k, beta=r["beta"], se=r["se"],
                         n_cells=int(masks[k].sum()),
                         weight_mass=float(w[masks[k]].sum()) / n,
                         mean_y=float(pieces[k].mean()),
                         share_of_total=r["beta"] / total["beta"]))
    gap = abs(sum(r["beta"] for r in rows if np.isfinite(r["beta"]))
              - total["beta"])
    check(gap < TOL_ADDITIVITY,
          f"the margin betas add to the total (h={h})",
          f"|sum - total| = {gap:.2e} against {TOL_ADDITIVITY:.0e}; "
          f"total {total['beta']:.6f}")
    rows.append(dict(horizon=h, margin="TOTAL", beta=total["beta"],
                     se=total["se"], n_cells=len(cur), weight_mass=1.0,
                     mean_y=float(mpgr_c.mean()), share_of_total=1.0))

    print(f"\n  {'margin':14s}{'beta':>11s}{'se':>10s}{'t':>8s}"
          f"{'share':>9s}{'cells':>13s}{'wt mass':>10s}")
    for r in rows:
        t = r["beta"] / r["se"] if r["se"] and np.isfinite(r["se"]) else np.nan
        rule = "  " + "-" * 73 + "\n" if r["margin"] == "TOTAL" else ""
        print(f"{rule}  {r['margin']:14s}{r['beta']:11.6f}{r['se']:10.6f}"
              f"{t:8.1f}{100 * r['share_of_total']:8.1f}%{r['n_cells']:13,}"
              f"{r['weight_mass']:10.4f}")
    ext = sum(r["beta"] for r in rows if r["margin"] in ("entry", "exit"))
    print(f"\n  extensive margin (entry + exit): {ext:+.6f}, "
          f"{100 * ext / total['beta']:.1f}% of the total")
    return rows, frame


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--rung", default=RUNG, choices=list(w22.RUNGS_ALL))
    ap.add_argument("--share", default=SHARE,
                    choices=["fixed4", "fixed5", "fixed10"])
    ap.add_argument("--level", type=float, default=LEVEL)
    args = ap.parse_args()

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(dc.LOG_DIR / f"31_margin_decomposition_pre_{args.share}_{args.rung}.log")
    import pyfixest as pf
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
    t0 = time.time()

    print("=" * 80)
    print("  County employment growth by margin, on the contract shock")
    print("=" * 80)
    print(f"  PRE-{w22.NAICS_BREAK} ONLY. Instrument z_income_{args.share}, "
          f"IQR{args.level:g} by year.")
    print(f"  Rung {args.rung}. ccg + year FE, clustered on the county.")
    print("  Nonoverlapping windows: the panel carries one start year every h,")
    print("  so h=2 is every other year and h=4 is every fourth.")

    allrows = []
    for h in args.horizon:
        print(f"\n{'=' * 80}\n  h = {h}\n{'=' * 80}")
        rows, _ = one_horizon(h, args)
        allrows += rows

    res = pd.DataFrame(allrows)
    print(f"\n{'=' * 80}\n  ACROSS HORIZONS: beta by margin\n{'=' * 80}")
    piv = res.pivot(index="margin", columns="horizon", values="beta")
    piv = piv.reindex([*MARGINS, "TOTAL"])
    print(f"  {'margin':14s}" + "".join(f"{'h=' + str(h):>14s}"
                                        for h in args.horizon))
    for m in piv.index:
        print(f"  {m:14s}" + "".join(f"{piv.loc[m, h]:14.6f}"
                                     for h in args.horizon))
    ext = piv.loc["entry"] + piv.loc["exit"]
    print(f"  {'-' * 76}")
    print(f"  {'entry + exit':14s}" + "".join(f"{ext[h]:14.6f}"
                                              for h in args.horizon))
    print(f"  {'as % of total':14s}"
          + "".join(f"{100 * ext[h] / piv.loc['TOTAL', h]:13.1f}%"
                    for h in args.horizon))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f"coeff_margin_decomposition_pre_{args.share}_{args.rung}.csv"
    res.assign(period="pre", share=args.share, rung=args.rung, fe="ccg+year",
               winsor=f"iq{args.level:g}_year").to_csv(
        out, index=False, float_format="%.15g")
    print(f"\n  wrote {out.relative_to(REPO_ROOT)}")

    n_fail = len(w22.failures)
    print(f"\n{'=' * 80}\n  {n_fail} checks failed, {time.time() - t0:.1f}s"
          f"\n{'=' * 80}")
    tee.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
