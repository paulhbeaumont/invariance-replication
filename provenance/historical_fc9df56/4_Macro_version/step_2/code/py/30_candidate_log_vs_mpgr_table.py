#!/usr/bin/env python3
"""Log growth against midpoint growth, at three rungs, with the fixed effects held.

28_table_d1.py moves two things at once between its columns: the rung gets finer
AND the unit effect changes from `ccg` to `ccg x industry`. 29 showed that the
second move is what flips the sign. This table removes it. Every cell here carries
`ccg + year` fixed effects, at every rung, so the only things that move across the
table are the aggregation, the transform, and the weight.

    (1) County            (2) County x NAICS2      (3) County x NAICS4

    Panel A    log growth,      unweighted     the benchmark everyone runs;
                                               INADMISSIBLE as an answer (A.1.5)
    Panel A2   log growth,      MPGR weight    the weighting channel, transform held
    Panel B    midpoint growth, MPGR weight    admissible answer 1

A -> A2 changes only the weight and A2 -> B changes only the transform, so the two
moves separate the implicit weighting from the curvature of the log rather than
confounding them.

WHAT PANEL B MUST DO HERE. `ccg` and `year` are both functions of (c, t), so
Proposition 1 binds Panel B across all three columns: the midpoint coefficient is
the SAME NUMBER at the county, at NAICS2 and at NAICS4, to machine precision. That
is asserted, not reported. Panel A has no such guarantee and is not expected to
hold still, because log growth is undefined for a cell empty in either period and a
log specification silently discards those cells. The two bottom diagnostics say how
much it discards.

Run class: estimate_only on the cell construction 29 rebuilt. The instrument, the
frame and the winsorization are 29's, imported rather than restated.
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


# 29 owns the frame, the instrument and the winsorization; 28 owns the estimator and
# the cell builder. Loading 29 reaches 28 through it, so the whole process still
# holds ONE failures list.
t29 = _load("candidate_mpgr", STEP2_CODE / "29_candidate_iv_mpgr_check.py")
t28, c27, w22 = t29.t28, t29.c27, t29.w22
check, Tee = w22.check, w22.Tee

HORIZON, PERIOD, SHARE, LEVEL = 4, "pre", "fixed5", 5.0
RUNGS = ("naics2", "naics4_alloc")
TOL_INVAR = t28.TOL_INVAR

PRETTY = {"county": "County", "naics2": "County x NAICS2",
          "naics4_alloc": "County x NAICS4"}


def fit(y, x, z, w, f1, f2, cl):
    return t28.fit2sls(y, x, z, w, f1, f2, cl, singleton=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--period", default=PERIOD, choices=["pre", "post"])
    ap.add_argument("--share", default=SHARE, choices=["fixed4", "fixed5", "fixed10"])
    ap.add_argument("--level", type=float, default=LEVEL)
    args = ap.parse_args()

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(dc.LOG_DIR / f"30_candidate_log_vs_mpgr_{args.period}"
                           f"_h{args.horizon}_{args.share}.log")
    import pyfixest as pf
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
    t0 = time.time()
    h = args.horizon

    print("=" * 78)
    print("  Log growth against midpoint growth, ccg + year fixed effects THROUGHOUT")
    print("=" * 78)
    print(f"  h = {h}, {args.period}-{w22.NAICS_BREAK}, instrument "
          f"z_income_{args.share}, IQR{args.level:g} by year")
    print("  The unit effect is the COUNTY at every rung. 28's `ccg x industry` is")
    print("  deliberately absent: 29 showed it is what flips the sign, and holding")
    print("  it out is the point of this table.")

    frame = t29.build_frame(args.period, h, args.share, args.level)
    shock, zinst = frame["shock"].to_numpy(), frame["z"].to_numpy()
    ccg, yr = frame["ccg"].to_numpy(), frame["year"].to_numpy()
    years_emp = sorted(set(frame["year"]) | set(frame["year"] - h))

    ds, cells = {}, {}
    for rung in RUNGS:
        emp, ccg_arr = w22.load_rung(rung, years_emp)
        _, code_arr = c27.rung_codes(rung, years_emp[0])
        keep = np.isin(ccg_arr, frame["ccg"].unique())
        if not keep.all():
            emp = {y: v[keep] for y, v in emp.items()}
            ccg_arr, code_arr = ccg_arr[keep], code_arr[keep]
        ds[rung] = t28.build_long(rung, emp, ccg_arr, code_arr, frame, h)
        cells[rung] = t28.cell_outcomes(ds[rung])

    # The county totals must be the same object whichever rung they are summed from.
    a, b = ds["naics2"], ds["naics4_alloc"]
    worst = float(np.max(np.abs(a["cur_c"] - b["cur_c"])
                         / np.maximum(np.abs(b["cur_c"]), 1.0)))
    check(worst < 1e-12, "A.1.2  the two rungs sum to the same county employment",
          f"max relative gap {worst:.2e} between NAICS2 and NAICS4 county totals")

    d4 = ds["naics4_alloc"]
    cur_c, lag_c = d4["cur_c"], d4["lag_c"]
    den_c = cur_c + lag_c
    nz = den_c != 0.0
    pos_c = (cur_c > 0) & (lag_c > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mpgr_c = np.where(nz, 2.0 * (cur_c - lag_c) / np.where(nz, den_c, 1.0), np.nan)
        log_c = np.where(pos_c, np.log(np.where(pos_c, cur_c, 1.0))
                         - np.log(np.where(pos_c, lag_c, 1.0)), np.nan)
    check(bool(nz.all() and pos_c.all()),
          "A.1.6  every county-year has a defined county growth rate at both "
          "transforms",
          f"{int((~nz).sum())} with no midpoint denominator, "
          f"{int((~pos_c).sum())} with a zero endpoint")

    rows, diag = [], {}
    one_c = np.ones(len(frame))

    # ---- column (1), the county ------------------------------------------------
    # One row per county-year, so the MPGR weight is identically 1 and Panel A2 IS
    # Panel A. Reported twice on purpose, because the pair is what the reader
    # compares the cell columns against.
    r_log = fit(log_c, shock, zinst, one_c, ccg, yr, ccg)
    r_mp = fit(mpgr_c, shock, zinst, one_c, ccg, yr, ccg)
    rows += [dict(panel="A", rung="county", **r_log),
             dict(panel="A2", rung="county", **r_log),
             dict(panel="B", rung="county", **r_mp)]
    diag["county"] = dict(share_A=100.0, omega_A2=1.0, zero_pct=np.nan,
                          n_cell=len(frame))

    # ---- columns (2) and (3), the cells ----------------------------------------
    emp_tot = float(cur_c.sum())
    for rung in RUNGS:
        d = ds[rung]
        mpgr, w, log, pos = cells[rung]
        cy = d["f_cy"]
        f1, f2, cl = ccg[cy], yr[cy], ccg[cy]

        rows.append(dict(panel="A", rung=rung,
                         **fit(log[pos], shock[cy][pos], zinst[cy][pos],
                               np.ones(int(pos.sum())), f1[pos], f2[pos], cl[pos])))
        rows.append(dict(panel="A2", rung=rung,
                         **fit(log[pos], shock[cy][pos], zinst[cy][pos],
                               w[pos], f1[pos], f2[pos], cl[pos])))
        rows.append(dict(panel="B", rung=rung,
                         **fit(mpgr, shock[cy], zinst[cy], w, f1, f2, cl)))

        # The two rows that say WHY Panel A moves: what the log mask throws away.
        omega = np.bincount(cy[pos], weights=w[pos], minlength=len(frame))
        diag[rung] = dict(
            share_A=100.0 * float(d["cur"][pos].sum()) / emp_tot,
            omega_A2=float(np.mean(omega)),
            zero_pct=100.0 * float(((d["cur"] == 0) & (d["lag"] == 0)).mean()),
            n_cell=len(mpgr))

    res = pd.DataFrame(rows).drop(columns=["keep"])

    def cell(panel, rung):
        r = res[(res.panel == panel) & (res.rung == rung)].iloc[0]
        return r["beta"], r["se"], r["n"]

    order = ["county"] + list(RUNGS)
    print("\n" + "=" * 78)
    print("  2SLS on the contract shock, ccg + year FE, clustered on the county")
    print("=" * 78)
    print(f"  {'':34s}" + "".join(f"{PRETTY[r]:>21s}" for r in order))
    for panel, lab in (("A", "log growth, unweighted"),
                       ("A2", "log growth, MPGR weight"),
                       ("B", "midpoint growth, MPGR weight")):
        bs = [cell(panel, r) for r in order]
        print(f"\n  Panel {panel}: {lab}")
        print(f"  {'  contract shock':34s}" + "".join(f"{b:>21.6f}" for b, _, _ in bs))
        print(f"  {'':34s}" + "".join(f"{'(' + f'{s:.6f}' + ')':>21s}"
                                      for _, s, _ in bs))
        print(f"  {'  N':34s}" + "".join(f"{n:>21,}" for _, _, n in bs))
    print(f"\n  {'cells zero in both periods (%)':34s}"
          + "".join(f"{diag[r]['zero_pct']:>21.1f}" for r in order))
    print(f"  {'share of county employment, A (%)':34s}"
          + "".join(f"{diag[r]['share_A']:>21.2f}" for r in order))
    print(f"  {'sum_m omega retained, A2':34s}"
          + "".join(f"{diag[r]['omega_A2']:>21.3f}" for r in order))

    # ---- Proposition 1 across the whole Panel B row ----------------------------
    b_ref = cell("B", "county")[0]
    for rung in RUNGS:
        b = cell("B", rung)[0]
        check(abs(b - b_ref) < TOL_INVAR,
              f"Prop 1  Panel B is invariant from the county to {PRETTY[rung]}",
              f"|beta - beta_county| = {abs(b - b_ref):.3e} against {TOL_INVAR:.0e}; "
              f"{b:.10f} against {b_ref:.10f}")
    print("\n  Panel B holds still across the row because ccg and year are both")
    print("  functions of (c, t). Panel A does not, and the bottom two rows are why:")
    print("  log growth is undefined on a cell empty in either period, so it drops")
    print("  those cells and estimates on a shrinking share of county employment.")

    a_c, a_2, a_4 = (cell("A", r)[0] for r in order)
    print(f"\n  Panel A across the row: {a_c:.6f} -> {a_2:.6f} -> {a_4:.6f}")
    print(f"  Panel A2 across the row: "
          + " -> ".join(f"{cell('A2', r)[0]:.6f}" for r in order))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / (f"coeff_candidate_log_vs_mpgr_{args.period}_h{h}"
                        f"_{args.share}.csv")
    res = res.assign(period=args.period, horizon=h, share=args.share,
                     fe="ccg+year", winsor=f"iq{args.level:g}_year",
                     zero_pct=[diag[r]["zero_pct"] for r in res["rung"]],
                     share_emp_A=[diag[r]["share_A"] for r in res["rung"]],
                     omega_A2=[diag[r]["omega_A2"] for r in res["rung"]])
    res.to_csv(out, index=False, float_format="%.15g")
    print(f"\n  wrote {out.relative_to(REPO_ROOT)}")

    n_fail = len(w22.failures)
    print(f"\n{'=' * 78}\n  {n_fail} checks failed, {time.time() - t0:.1f}s"
          f"\n{'=' * 78}")
    tee.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
