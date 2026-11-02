#!/usr/bin/env python3
"""Table 6: the disaggregation ladder under Poisson on levels, h=4.

A.1.5 names TWO admissible answers, not one. Tables 1-4 run the first (midpoint
growth at MPGR weights) and the three ways of being wrong. This runs the second.

    county   E[E_ct  | .] = exp( a_c + d_t + beta_b * shock_ct + g * v_ct )
    cells    E[E_imt | .] = exp( a_c + d_t + beta_d * shock_ct + g * v_ct )

THE OUTCOME IS EMPLOYMENT IN LEVELS, not a growth rate, so beta is a
SEMI-ELASTICITY and is NOT on the same scale as Tables 1-4. Those are not two
estimates of one number and their ratio means nothing. That sentence travels with
every number this file writes.

WHY PROPOSITION 1 HOLDS HERE, and it is not the collapse. Nothing collapses.
Summing the cell conditional means over m,

    sum_m mu_imt = M_c exp(a_c + d_t + b x_ct) = exp(a_c + log M_c + d_t + b x_ct)

which IS the county model with unit effect a_c + log M_c. The cell first-order
conditions for (beta, d_t, a_c) therefore reduce to the county ones and beta is
the same number. THE DERIVATION NEEDS M_c CONSTANT IN t, which is why this file
cannot read the cached cell panel - see `live_cells` below - and the constancy is
asserted rather than assumed.

THE FIXED EFFECT IS `ccg`, WHICH DEPARTS FROM A.1.5's LETTER. The protocol says
cell FE (`ccg x industry`) at the disaggregated level, and 28_table_d1.py derives
invariance that way from sum_m exp(a_im) = exp(A_i). Both work, and MEASURED ON
THIS PANEL THEY GIVE THE IDENTICAL COEFFICIENT to nine digits: adding cell
effects adds free parameters that satisfy their own first-order conditions
without moving sum_m mu_imt, which is the only channel through which they could
reach beta. `ccg` is taken because it makes Table 6's fixed-effect row identical
to Tables 1-4, so the reader holds a column fixed, turns the page, and the only
thing that has changed is the estimator. A.1.5's own effect is run anyway as the
`*_cellfe` columns, so "it makes no difference" is checkable in the output rather
than asserted in a comment.

A.1.4 IS NOT BENT. It forbids a fixed effect at the level where the treatment
varies. Our shock varies over (c, t), so the forbidden object is `ccg x year`,
which appears nowhere in step 2.

THE INSTRUMENT ENTERS BY CONTROL FUNCTION, because nothing in this stack does IV
for Poisson. THE STANDARD ERROR TREATS THAT RESIDUAL AS DATA and therefore
understates by an amount this design does not correct.

Run class: estimate_only on the cell construction.
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
check, Tee, dc = md.check, md.Tee, md.dc
RESULT_DIR, REPO_ROOT = md.RESULT_DIR, md.REPO_ROOT
t28, t29, c27, w22 = md.t28, md.t29, md.c27, md.w22

SPEC = "poisson_levels"
COLS = ("county", "n2", "n2_iy", "n4", "n4_iy")
EXTRA = ("n2_cellfe", "n4_cellfe")
COLHEAD = {"county": "county", "n2": "x NAICS2", "n2_iy": "x NAICS2 +i.y",
           "n4": "x NAICS4", "n4_iy": "x NAICS4 +i.y",
           "n2_cellfe": "x NAICS2 cellFE", "n4_cellfe": "x NAICS4 cellFE"}
KEY = {"naics2": ("n2", "n2_iy", "n2_cellfe"),
       "naics4_alloc": ("n4", "n4_iy", "n4_cellfe")}

F = lambda a: pd.factorize(a)[0].astype(np.int64)   # noqa: E731


def build_cells(frame, rung, h, source="efsy"):
    """The cell rows, on the support 11_build_employment_cells.py actually wrote.

    `28`'s build_long, verbatim, and then NOT `mpgr_disagg.prune`. The two differ
    in a way that decides whether Proposition 1 holds at all, so the reason is
    here rather than in a commit message.

    The cell support IS the full county x industry cross product - 27's
    `support_completeness` asserts it, because plan section 2.1 requires absent
    cells to be present as zeros rather than absent as missing. So an industry
    that never appears in a county still occupies one row per window.
    """
    years = sorted(set(frame["year"]) | set(frame["year"] - h))
    emp, ccg_arr, code_arr = md._load_cells(rung, years, source,
                                            frame["ccg"].unique())
    return t28.build_long(rung, emp, ccg_arr, code_arr, frame, h)


def live_cells(d, yr, rung):
    """Which cells Poisson estimates on, and why the rule is a CELL property.

    A cell with no employment in ANY window has its effect at -infinity, mu at
    zero, and contributes to no score; pyfixest and ppmlhdfe drop it too. That is
    the estimator's own kernel and not a cleaning step, and it is also exactly
    "this industry never appears in this county", so the two readings agree and
    the rows carry zero employment. 31.7% of the NAICS4 panel is this.

    WHAT THIS RULE IS NOT is `mpgr_disagg.prune`, which drops a row when the cell
    is empty in both periods OF THAT WINDOW. That is a different set: 368,748
    rows at NAICS4 are a county-industry that genuinely exists, empty in this
    window and populated in another. Removing those makes the per-county cell
    count vary with t, `sum_m mu_imt` stops factoring as exp(A_c + d_t + b x),
    and Proposition 1 breaks - for a SAMPLE reason that would read as a
    specification reason. Empty-in-this-window is data; never-appears was never
    there. The filter therefore has to be a property of the cell, applied to
    every window at once, which is what `poisson_separated` on (f_cell, year)
    computes.
    """
    cur = d["cur"]
    live = t28.poisson_separated(np.maximum(cur, 0.0), d["f_cell"], yr[d["f_cy"]])
    lost = float(np.abs(cur[~live]).sum())
    check(lost == 0.0,
          f"C  the dropped cells carry no employment ({md.PRETTY[rung]})",
          f"{int((~live).sum()):,} of {len(cur):,} rows dropped "
          f"({100 * (~live).mean():.1f}%), in cells with no employment in any "
          f"window; employment lost {lost:.1f}. The county totals are therefore "
          f"the same object before and after")
    return live


def check_constant_mc(d, live, frame, rung):
    """M_c constant in t, which is the condition the invariance rests on.

    Asserted rather than assumed, because it is the single thing `prune` would
    silently break and there would be nothing in the output to say so.
    """
    per_cy = np.bincount(d["f_cy"][live], minlength=len(frame))
    g = pd.DataFrame({"ccg": frame["ccg"].to_numpy(), "n": per_cy}) \
        .groupby("ccg")["n"]
    spread = int((g.max() - g.min()).max())
    check(spread == 0,
          f"P1 the cell count per county is constant across windows ({md.PRETTY[rung]})",
          f"max within-county spread in the number of live cells = {spread}; "
          f"sum_m mu_imt factors as exp(A_c + d_t + b x_ct) only if it is zero")


def fit(y, x, v, f1, f2, cl):
    """v is None on the baseline (reduced-form) path; poisson_cf then runs a
    one-column design, which is the model and not an approximation to it."""
    return t28.poisson_cf(y, x, v, F(f1), F(f2), F(cl))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--period", default="pre", choices=["pre", "post"])
    ap.add_argument("--share", default="fixed5",
                    choices=["fixed4", "fixed5", "fixed10"])
    ap.add_argument("--level", type=float, default=5.0)
    ap.add_argument("--rung", nargs="+", default=list(md.RUNGS),
                    help="a partial run prints but writes NO coefficient file")
    ap.add_argument("--frame", default="candidate",
                    choices=["candidate", "baseline"],
                    help="candidate: the IQR5 control-function design this "
                         "table was first built on. baseline: the locked "
                         "baseline's mapping -- see the FE note below")
    ap.add_argument("--source", default="efsy", choices=list(md.SOURCES),
                    help="which employment series the OUTCOME is built from; "
                         "see 34_tables_payoff.py --source")
    ap.add_argument("--unit-fe", default=None, choices=["ccg", "divisionyear"],
                    help="baseline frame only, and REQUIRED there: which unit "
                         "effect the Poisson ladder carries. See the refusal "
                         "message; there is deliberately no default.")
    args = ap.parse_args()
    h = args.horizon
    base = args.frame == "baseline"
    src = args.source

    # PROPOSITION 1 DOES NOT SURVIVE THE LOCKED BASELINE'S FIXED EFFECT, and the
    # producer refuses rather than printing a ladder whose flatness claim is
    # false. Measured on the EFSY panel at h=4, NAICS2, reduced form on z:
    #
    #     unit effect            county          cell        |gap|
    #     division x year   -0.0000723504  -0.0000720981    2.5e-07
    #     ccg + year        -0.0000046459  -0.0000046459    4.0e-19
    #
    # against TOL_POIS_INVAR = 1e-9. The reason is not numerical. Summing the
    # cell means gives sum_m mu_imt = M_c exp(eta_ct), so the county model needs
    # a free unit effect to absorb log M_c. `ccg` is exactly that effect;
    # division x year is coarser than the county and cannot absorb a county
    # quantity, so the two models genuinely differ and the ladder is not flat.
    #
    # Tables 1-5 are unaffected: their invariance is an exact weighting identity
    # that holds under ANY fixed effect coarser than the unit, which is why
    # division x year is admissible there and not here.
    #
    # The choice below is therefore an estimand decision and is left to the
    # author. `ccg` takes from the locked baseline everything transferable --
    # the composition instrument, the reduced form, the full era, no winsor --
    # and keeps A.1.5's county effect, so Table 6 remains the companion to
    # Table 1. `divisionyear` matches Tables 1-5's fixed effect literally and
    # gives up the invariance claim; it is offered so the failure can be
    # reported as a result rather than being unavailable.
    if base and args.unit_fe is None:
        raise SystemExit(
            "37 --frame baseline requires --unit-fe, and has no default.\n"
            "The locked baseline's division-by-year effect breaks Proposition 1\n"
            "for Poisson on levels: the county model needs a free unit effect to\n"
            "absorb log M_c, and an effect coarser than the county cannot. The\n"
            "measured gap is 2.5e-07 against a tolerance of 1e-9, while ccg + year\n"
            "is exact at 4e-19. Tables 1-5 are unaffected -- their invariance is a\n"
            "weighting identity that holds under any effect coarser than the unit.\n"
            "Pass --unit-fe ccg to keep the invariance claim (reduced form on the\n"
            "composition instrument, full era, no winsor, A.1.5's county effect),\n"
            "or --unit-fe divisionyear to report the failure deliberately.\n"
            "See the comment above this message before choosing.")
    unit_fe = args.unit_fe if base else "ccg"
    stag = "" if src == "efsy" else f"_{src}"
    ptag = "baseline" if base else args.period

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(dc.LOG_DIR
              / f"37_table_poisson_{ptag}{stag}_h{h}_{args.share}.log")
    import pyfixest as pf
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
    t0 = time.time()

    print("=" * 84)
    print(f"  TABLE 6, the disaggregation ladder under POISSON ON LEVELS.  h = {h}, "
          f"{ptag}" + ("" if base else f"-{w22.NAICS_BREAK}"))
    print("=" * 84)
    if base:
        # THE BASELINE MAPPING, STATED. Tables 1-5 map the locked baseline onto
        # `fit_*_base`: reduced form on z, division-by-year effects, county
        # clusters. Table 6 is mapped the same way and the mapping is written
        # down here because it is a CHOICE, not a translation:
        #   - the regressor becomes z and the control function disappears. A
        #     control function is the IV device; the locked baseline is a
        #     reduced form, and a reduced form has no first stage to residualize.
        #     beta is therefore a reduced-form semi-elasticity on levels.
        #   - the effects become division x year alone, in place of ccg + year.
        #     Proposition 1 still holds: sum_m exp(dy_{d(c)t} + b z_ct) factors
        #     as M_c exp(dy + b z) whenever M_c is constant in t, which is
        #     exactly what check_constant_mc asserts. The cell-FE column keeps
        #     A.1.5's letter by putting f_cell in place of the division-year
        #     block's role as the unit effect.
        print("  the locked baseline's mapping: REDUCED FORM on the harmonized")
        print("  top-800 composition instrument (no control function), span 5,")
        print("  positive rule, population denominator, division-by-year effects,")
        print("  full era, NO winsor anywhere. Employment in LEVELS, so beta is a")
        print("  reduced-form semi-elasticity and is NOT comparable to Tables 1-4")
        print("  nor to the candidate-frame Table 6.")
    else:
        print(f"  instrument z_income_{args.share} by control function, IQR{args.level:g}")
        print("  by year on the shock and the instrument only. Employment in LEVELS, so")
        print("  beta is a semi-elasticity and is NOT comparable to Tables 1-4.")
        print("  ccg x year appears nowhere (A.1.4).")
    if src != "efsy":
        print(f"  OUTCOME SOURCE: {src}; NETS begins in 1990 so the first target "
              f"year is {1990 + h}.")

    if base:
        frame, sample = md.baseline_frame(h, src)
    else:
        frame = t29.build_frame(args.period, h, args.share, args.level)
        frame, sample = md.nets_frame_rule(frame, h, src)
    yr, ccg = frame["year"].to_numpy(), frame["ccg"].to_numpy()
    x, z = frame["shock"].to_numpy(), frame["z"].to_numpy()
    div = (frame["division"].astype(str) + "_" + frame["year"].astype(str)
           ).to_numpy() if base else None

    if base:
        DIVFE = unit_fe == "divisionyear"
        # Reduced form: the regressor IS the instrument and there is no control
        # function at all. v is None, not a zero column -- a zero column makes
        # the 2x2 information matrix singular, and would in any case charge a
        # degree of freedom for a parameter that is not in the model.
        xreg, v = z, None
        # `divisionyear` puts the division-year block in the unit slot and
        # leaves no second factor; `ccg` keeps A.1.5's county effect and puts
        # the year in the second slot, exactly as the candidate design does.
        unit = div if DIVFE else ccg
        cellf2 = None if DIVFE else yr
    else:
        # THE CONTROL FUNCTION IS BUILT ONCE, AT THE COUNTY, AND CARRIED DOWN. The
        # shock and the instrument vary only in (c, t), so v is a county object.
        # Rebuilding it inside a column's own fixed effects under that column's IRLS
        # weights would give a different v, and Proposition 1 would then be comparing
        # two different models rather than one model at two levels.
        xreg = x
        v = t28.control_function(x, z, np.ones(len(x)), F(ccg), F(yr))
        unit, cellf2 = ccg, yr

    row = dict(spec=SPEC)

    for rung in args.rung:
        print(f"\n{'-' * 84}\n  {md.PRETTY[rung]}\n{'-' * 84}")
        d = build_cells(frame, rung, h, src)
        cur, cy = d["cur"], d["f_cy"]
        check(int((cur < 0).sum()) == 0,
              f"C  no cell carries negative employment ({md.PRETTY[rung]})",
              f"the 2026-08-17 employment rebuild dropped them; `28`'s floor "
              f"convention is a no-op here and is not applied")

        # The county series is the CELL SUM, taken before the filter, so the two
        # sides of Proposition 1 cannot drift apart the way they would against an
        # independently built series.
        cur_c = np.bincount(cy, weights=cur, minlength=len(frame))

        live = live_cells(d, yr, rung)
        check_constant_mc(d, live, frame, rung)

        k_lev, k_iy, k_cf = KEY[rung]
        ycell = np.maximum(cur, 0.0)[live]
        xcell = xreg[cy][live]
        vcell = None if v is None else v[cy][live]
        clcell = ccg[cy][live]
        unitcell = unit[cy][live]
        f2cell = (np.zeros(int(live.sum()), np.int64) if cellf2 is None
                  else cellf2[cy][live])
        u1, u2 = (("divisionyear", "") if (base and DIVFE)
                  else ("ccg", " + year"))
        for key, f2, lab in ((k_lev, f2cell, f"{u1}{u2}"),
                             (k_iy, d["f_ky"][live], f"{u1} + industry x year"),
                             # A.1.5's column swaps the UNIT effect for the cell
                             # effect and keeps the other factor exactly as it
                             # is. Under division-by-year that other factor is
                             # empty and the division-year block moves into the
                             # second slot; otherwise it stays the year.
                             (k_cf, unitcell if (base and DIVFE) else f2cell,
                              f"cellFE + {u1 if (base and DIVFE) else 'year'}"
                              f"  [A.1.5's letter]")):
            f1 = d["f_cell"][live] if key == k_cf else unitcell
            t1 = time.time()
            r = fit(ycell, xcell, vcell, f1, f2, clcell)
            row[key], row[f"se_{key}"], row[f"n_{key}"] = r["beta"], r["se"], int(r["n"])
            print(f"    {lab:34s} beta {r['beta']:+.9f} ({r['se']:.4f})  "
                  f"N {r['n']:>10,}  {r['iters']:>3d} IRLS  [{time.time() - t1:.0f}s]")

        # The gate. NAICS2 is where pyfixest can be run, and nothing at NAICS4 is
        # read until the hand-rolled IRLS has been shown to reproduce it on beta
        # AND on the clustered standard error.
        if rung == "naics2":
            t28.validate_poisson(ycell, xcell, vcell, F(unitcell), F(f2cell),
                                 F(clcell), dict(beta=row[k_lev], se=row[f"se_{k_lev}"]),
                                 f"naics2 {u1}{u2}")

        if "county" not in row:
            f2c = np.zeros(len(frame), np.int64) if (base and DIVFE) else yr
            keep = t28.poisson_separated(np.maximum(cur_c, 0.0), F(unit), F(f2c))
            rc = fit(cur_c[keep], xreg[keep],
                     None if v is None else v[keep],
                     unit[keep], f2c[keep], ccg[keep])
            row["county"] = rc["beta"]
            row["se_county"] = rc["se"]
            row["n_county"] = int(rc["n"])
            print(f"    {f'county, {u1}{u2}':34s} beta {rc['beta']:+.9f} "
                  f"({rc['se']:.4f})  N {rc['n']:>10,}  {rc['iters']:>3d} IRLS")

    # ------------------------------------------------------------- Proposition 1
    for rung in args.rung:
        k_lev, _, k_cf = KEY[rung]
        gap = row[k_lev] - row["county"]
        check(abs(gap) < t28.TOL_POIS_INVAR,
              f"Prop 1  Poisson on levels is invariant, county to {COLHEAD[k_lev]}",
              f"|gap| = {abs(gap):.2e} < {t28.TOL_POIS_INVAR:.0e}; "
              f"{row[k_lev]:.10f} against {row['county']:.10f}")
        d_cf = row[k_cf] - row[k_lev]
        check(abs(d_cf) < t28.TOL_POIS_INVAR,
              f"A.1.5  ccg and ccg x industry give the same number ({COLHEAD[k_lev]})",
              f"|gap| = {abs(d_cf):.2e}; cellFE {row[k_cf]:.10f} against "
              f"ccg {row[k_lev]:.10f}. This is what makes taking the simpler and "
              f"more comparable effect a measured choice rather than a shortcut")

    have = [c for c in COLS + EXTRA if c in row]
    W = 46
    print(f"\n{'=' * 84}\n  Table 6: employment levels, Poisson (control function)"
          f"\n{'=' * 84}")
    print("  " + " " * W + "".join(f"{COLHEAD[c]:>17s}" for c in have))
    print(f"  {'contract shock':{W}s}" + "".join(f"{row[c]:17.6f}" for c in have))
    print(f"  {'  clustered SE':{W}s}"
          + "".join(f"{'(' + format(row['se_' + c], '.4f') + ')':>17s}" for c in have))
    print(f"  {'  gap from county':{W}s}"
          + "".join(f"{'':17s}" if c == "county"
                    else f"{row[c] - row['county']:+17.6f}" for c in have))
    print(f"  {'  observations':{W}s}" + "".join(f"{row['n_' + c]:17,d}" for c in have))
    print("\n  THE SE TREATS THE CONTROL FUNCTION AS DATA and understates by an "
          "amount\n  this design does not correct. beta is a semi-elasticity on "
          "levels and is\n  NOT on the same scale as Tables 1-4.")

    n_fail = len(w22.failures)
    if sorted(args.rung) != sorted(md.RUNGS):
        print(f"\n  PARTIAL RUN ({', '.join(args.rung)}); no coefficient file written")
        print(f"\n{'=' * 84}\n  {n_fail} checks failed, {time.time() - t0:.1f}s"
              f"\n{'=' * 84}")
        tee.close()
        return 1 if n_fail else 0

    # The estimation window travels WITH the coefficients, so the rendered note is
    # the sample that was actually estimated on and cannot drift from it. These are
    # OUTCOME years: at h=4 post reads 2002-2018, not the 1998-2014 start years.
    window = dict(year_first=int(yr.min()), year_last=int(yr.max()),
                  n_windows=int(pd.unique(yr).size))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f"coeff_table_poisson_{ptag}{stag}_h{h}_{args.share}.csv"
    pd.DataFrame([row]).assign(period=ptag, horizon=h, share=args.share,
                               source=src,
                               fe=("divisionyear" if (base and DIVFE)
                                   else "ccg+year"),
                               unit_fe=unit_fe,
                               **window).to_csv(out, index=False,
                                                float_format="%.15g")
    print(f"\n  estimation window: outcome years {window['year_first']}-"
          f"{window['year_last']}, {window['n_windows']}"
          + ("" if base else " non-overlapping"))
    print(f"  wrote {out.relative_to(REPO_ROOT)}")

    print(f"\n{'=' * 84}\n  {n_fail} checks failed, {time.time() - t0:.1f}s"
          f"\n{'=' * 84}")
    tee.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
