#!/usr/bin/env python3
"""The MPGR disaggregation check on the corrected candidate instrument.

INVARIANCE_TEST_PROTOCOL.md A.1. The county is the baseline unit `i`, county x
NAICS4 is the cell `m`, and the treatment is the county contract shock, which is
never split along the industry dimension (A.1.1).

WHAT IS DIFFERENT FROM 28_table_d1.py. 28 runs the shock `z_fixed_7074` at h = 2
on step 2's own frame. This runs the corrected candidate instrument at h = 4:

    endogenous  y_income        = sum_{t+1}^{t+h} D_ct / PI_ct
    instrument  z_income_fixed5 = s_c^{5yr} * (sum_{t+1}^{t+h} D^national_t) / PI^national_t

The national denominator is the year-t sum of real personal income over the same
3,080 counties. The instrument NEVER divides by the focal county's own income,
which is the correction; see `Candidate for first stage/README.md`.

THE THREE REGRESSIONS, and why this ladder rather than 28's columns.

    (1) county            midpoint employment growth, unweighted   ccg + year FE
    (2) county x NAICS4   midpoint growth, MPGR weight             ccg + year FE
    (3) county x NAICS4   midpoint growth, MPGR weight             ccg + industry x year FE

(1) and (2) differ ONLY in the level of aggregation. Both fixed effects are
functions of (c, t), so Proposition 1 applies and the two coefficients must agree
to machine precision. That equality is the check: if it fails, the weights or the
reconstruction are wrong, and nothing downstream is readable. (3) then adds the
one fixed effect that is not a function of (c, t), and the movement from (2) to
(3) is the object of interest.

ccg x year appears nowhere. It would absorb the treatment (A.1.4).

NOTHING IS WINSORIZED EXCEPT THE SHOCK AND THE INSTRUMENT. A.3.2 forbids capping
our own growth rates, so the employment midpoint growth is used raw at both
levels. The shock and the instrument carry the candidate table's IQR5-by-year
rule, reproducing `Candidate for first stage/code/02_run_candidate_first_stage.do`
including Stata's own percentile definition.

Run class: cache_rebuild. Cell midpoint growth and MPGR weights do not exist at
h = 4 on these windows and are constructed here.
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
CANDIDATE = REPO_ROOT / "Candidate for first stage"
RESULT_DIR = HERE.parents[2] / "result"

sys.path.insert(0, str(STEP1_CODE))
import dcads_common as dc  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 28 owns the estimator, the cell builder and the outcome constructors. It reaches
# 22 through 27, which holds the single module-level `failures` list; this file
# loads 28 and inherits that same chain rather than opening a second one.
t28 = _load("table_d1", STEP2_CODE / "28_table_d1.py")
c27, w22 = t28.c27, t28.w22
check, Tee = w22.check, w22.Tee

HORIZON = 4
PERIOD = "pre"
SHARE = "fixed5"
RUNG = "naics4_alloc"
WINSOR_LEVEL = 5.0

# Measured against 28's own declared tolerances for the same objects.
TOL_INVAR = t28.TOL_INVAR          # |beta_cell - beta_county| at (c,t) fixed effects
TOL_INVAR_SE = 1e-10               # RELATIVE, and on the UNCORRECTED variance. See
                                   # 28's fit_collapsed: G/(G-1) (N-1)/(N-K) is the
                                   # one part that knows how many ROWS the regression
                                   # ran on, 12,320 against 3.7M, so asserting the
                                   # corrected SEs equal would be asserting that a
                                   # degrees-of-freedom convention does not exist
TOL_RECON = 1e-12                  # sum_m w_im mpgr_im against the county midpoint
TOL_COUNTY_EMP = 1e-9              # cell sums against EFSY's county employment, rel
MPGR_BOUND = 2.0


# --------------------------------------------------------------------------------
# Stata's percentile, because the candidate cutoffs are Stata's and a numpy
# quantile is a different estimator. `summarize, detail` sets j = n p / 100; when j
# is an integer the percentile is the midpoint of x(j) and x(j+1), otherwise it is
# x(ceil(j)). Integer arithmetic on n*p, so the integer case is exact.
# --------------------------------------------------------------------------------

def stata_pctile(xs, n, p):
    num = n * p
    if num % 100 == 0:
        j = num // 100
        return 0.5 * (xs[j - 1] + xs[j]) if j < n else xs[n - 1]
    return xs[-(-num // 100) - 1]


def winsor4_iq(v, level):
    """`winsor4 varname, method(iq) level(#)`: clip at p50 -/+ level * IQR."""
    xs = np.sort(v)
    n = len(xs)
    p25 = stata_pctile(xs, n, 25)
    p50 = stata_pctile(xs, n, 50)
    p75 = stata_pctile(xs, n, 75)
    iqr = p75 - p25
    return np.clip(v, p50 - level * iqr, p50 + level * iqr)


def winsor4_by_year(v, year, level):
    out = v.astype(np.float64).copy()
    for y in np.unique(year):
        m = year == y
        out[m] = winsor4_iq(out[m], level)
    return out


def se_uncorrected(y, x, z, w, f1, f2, cl):
    """The clustered SE with ssc = 1, which is the part two levels share exactly.

    Same demeaning and same CRV1 as 28's `fit2sls`, called with `corrected=False`.
    Every call site here passes `singleton=False`, so no row is removed and the
    arrays reaching this function are the arrays that fit2sls estimated on.
    """
    f1 = pd.factorize(f1)[0].astype(np.int64)
    f2 = pd.factorize(f2)[0].astype(np.int64)
    cl = pd.factorize(cl)[0].astype(np.int64)
    M = w22.demean(np.column_stack([y, x, z]), f1, f2, w)
    yt, xt, zt = M[:, 0], M[:, 1], M[:, 2]
    beta = float(zt @ (w * yt)) / float(zt @ (w * xt))
    return t28.crv1(zt, xt, yt - beta * xt, w, cl, len(y), 1, corrected=False)


# --------------------------------------------------------------------------------
# The frame: the candidate county panel, on reghdfe's own estimation sample.
# --------------------------------------------------------------------------------

def build_frame(period, horizon, share, level):
    """The candidate panel, keyed on the OUTCOME year so 28's cell builder lines up.

    The candidate's `year` is the start year t and its contract window is t+1
    through t+h. 28's `build_long` reads `lag, cur = emp[y - h], emp[y]`, so the
    frame's year must be the window's END. Setting year = t + h makes the
    employment window t through t+h, which is the window the shock is measured
    over, and the two are then the same object rather than two conventions that
    happen to share a name.
    """
    src = CANDIDATE / "data" / "temp" / f"{period}_h{horizon}.dta"
    d = pd.read_stata(src)
    n_raw = len(d)

    # reghdfe drops singletons before it estimates, so e(sample) is what survives
    # iterative removal on (unit, state_year). 28's function, not a reimplementation.
    f1 = pd.factorize(d["unit"].to_numpy())[0].astype(np.int64)
    f2 = pd.factorize(d["state_year"].to_numpy())[0].astype(np.int64)
    keep = t28.drop_singletons(f1, f2)
    d = d[keep].reset_index(drop=True)

    zcol = f"z_income_{share}"
    year = d["year"].to_numpy(np.int64)
    frame = pd.DataFrame({
        "ccg": d["unit"].to_numpy(np.int64),
        "year": year + horizon,
        "start_year": year,
        "state": d["state"].to_numpy(np.int64),
        "shock": winsor4_by_year(d["y_income"].to_numpy(np.float64), year, level),
        "z": winsor4_by_year(d[zcol].to_numpy(np.float64), year, level),
        "shock_raw": d["y_income"].to_numpy(np.float64),
        "z_raw": d[zcol].to_numpy(np.float64),
    }).sort_values(["year", "ccg"]).reset_index(drop=True)

    print(f"\n[frame] {src.name}")
    print(f"    {n_raw:,} rows on disk; {len(frame):,} survive reghdfe's singleton "
          f"removal on (unit, state_year)")
    print(f"    {frame['ccg'].nunique():,} counties, start years "
          f"{sorted(set(frame['start_year']))}, outcome years "
          f"{sorted(set(frame['year']))}")
    for nm, raw, win in (("shock", "shock_raw", "shock"), ("instrument", "z_raw", "z")):
        n_cap = int((frame[raw].to_numpy() != frame[win].to_numpy()).sum())
        print(f"    {nm:11s} IQR{level:g}-by-year capped {n_cap:,} of {len(frame):,} "
              f"({100 * n_cap / len(frame):.2f}%)")
    return frame


# --------------------------------------------------------------------------------
# The county outcome is the CELL SUM, never a second employment series, so
# A.1.2 holds by construction. EFSY's own county total is then an independent
# check on that sum rather than the source of it.
# --------------------------------------------------------------------------------

def county_from_cells(d):
    cur_c, lag_c = d["cur_c"], d["lag_c"]
    den = cur_c + lag_c
    nz = den != 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        mpgr = np.where(nz, 2.0 * (cur_c - lag_c) / np.where(nz, den, 1.0), np.nan)
    return mpgr, den, nz


def check_against_efsy(frame, d, horizon):
    efsy = pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                           columns=["ccg", "year", "emp"])
    e = {y: g.set_index("ccg")["emp"] for y, g in efsy.groupby("year")}
    worst_cur = worst_lag = 0.0
    for y, sub in frame.groupby("year", sort=True):
        i = sub.index.to_numpy()
        for tag, series, arr in (("cur", e[y], d["cur_c"]),
                                 ("lag", e[y - horizon], d["lag_c"])):
            ref = series.reindex(sub["ccg"].to_numpy()).to_numpy(np.float64)
            got = arr[i]
            rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)
            m = float(np.nanmax(rel))
            if tag == "cur":
                worst_cur = max(worst_cur, m)
            else:
                worst_lag = max(worst_lag, m)
    check(max(worst_cur, worst_lag) < TOL_COUNTY_EMP,
          "A.1.2  the NAICS4 cells sum to EFSY's county employment",
          f"max relative gap {max(worst_cur, worst_lag):.2e} against "
          f"{TOL_COUNTY_EMP:.0e}; current {worst_cur:.2e}, lagged {worst_lag:.2e}")


def check_reconstruction(mpgr_cell, w_cell, d, mpgr_c, nz, n_frame):
    """sum_m w_im mpgr_im = mpgr_i, unit by unit. The identity everything rests on."""
    agg = np.bincount(d["f_cy"], weights=w_cell * mpgr_cell, minlength=n_frame)
    gap = np.abs(agg - mpgr_c)[nz]
    worst = float(np.nanmax(gap)) if gap.size else 0.0
    check(worst < TOL_RECON,
          "A.1.2  weighted cell midpoint growth reconstructs the county",
          f"max |sum_m w mpgr - mpgr_c| = {worst:.2e} against {TOL_RECON:.0e} "
          f"over {int(nz.sum()):,} county-years")
    return worst


# --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--period", default=PERIOD, choices=["pre", "post"])
    ap.add_argument("--share", default=SHARE, choices=["fixed4", "fixed5", "fixed10"])
    ap.add_argument("--rung", default=RUNG, choices=list(w22.RUNGS_ALL))
    ap.add_argument("--level", type=float, default=WINSOR_LEVEL)
    args = ap.parse_args()

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(dc.LOG_DIR / f"29_candidate_iv_mpgr_{args.period}_h{args.horizon}"
                           f"_{args.share}_{args.rung}.log")
    import pyfixest as pf
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
    t0 = time.time()
    h = args.horizon

    print("=" * 78)
    print("  The MPGR disaggregation check on the corrected candidate instrument")
    print("=" * 78)
    print(f"  horizon      h = {h}")
    print(f"  period       {args.period}-{w22.NAICS_BREAK}")
    print(f"  endogenous   y_income        = cumulative contracts / county income")
    print(f"  instrument   z_income_{args.share:8s} = share x national contracts / "
          f"national income")
    print(f"  winsorized   IQR{args.level:g}, by year, on the shock and the "
          f"instrument ONLY")
    print(f"  rung         {args.rung}")
    print("\n  DECLARED DECISIONS, before any number is read.")
    print("  - the midpoint guard is `den != 0`, never `den > 0`")
    print("  - no cell is dropped and no growth rate is winsorized (A.3.1, A.3.2)")
    print("  - the treatment is the COUNTY shock at every level and is never split")
    print("    along the industry dimension (A.1.1)")
    print("  - ccg x year appears nowhere. It would absorb the treatment (A.1.4)")
    print("  - the county outcome IS the cell sum, so A.1.2 holds by construction;")
    print("    EFSY's county series is an independent check on it, not its source")
    print("  - MPGR weights are used UNCLAMPED, negatives included")

    frame = build_frame(args.period, h, args.share, args.level)
    years_emp = sorted(set(frame["year"]) | set(frame["year"] - h))

    emp, ccg_arr = w22.load_rung(args.rung, years_emp)
    _, code_arr = c27.rung_codes(args.rung, years_emp[0])
    c27.support_completeness(args.rung, ccg_arr, code_arr)

    # A.1.6: the set of baseline units must be identical at every rung. The frame
    # is the candidate's 3,080; the cell support is the panel's own. Intersect and
    # say so, rather than letting a silent inner join decide the sample.
    cells_ccg = set(np.unique(ccg_arr).tolist())
    frame_ccg = set(frame["ccg"].tolist())
    missing = sorted(frame_ccg - cells_ccg)
    print(f"\n[A.1.6] baseline units")
    print(f"    frame carries {len(frame_ccg):,} counties; the cell panel carries "
          f"{len(cells_ccg):,}")
    if missing:
        print(f"    {len(missing):,} frame counties absent from the cell panel, "
              f"dropped from EVERY column: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")
        frame = frame[frame["ccg"].isin(cells_ccg)].reset_index(drop=True)
    check(len(frame) > 0, "A.1.6  the estimation frame is non-empty",
          f"{len(frame):,} county-years, {frame['ccg'].nunique():,} counties")

    d = t28.build_long(args.rung, emp, ccg_arr, code_arr, frame, h)
    n_frame = len(frame)
    print(f"\n[cells] {len(d['cur']):,} cell rows over {n_frame:,} county-years "
          f"({d['n_code']:,} industries x {d['n_year']} years)")

    check_against_efsy(frame, d, h)

    mpgr_cell, w_cell, _, _ = t28.cell_outcomes(d)
    mpgr_c, den_c, nz = county_from_cells(d)
    check(bool(np.all(np.abs(mpgr_cell) <= MPGR_BOUND + 1e-12)),
          "A.3.1  cell midpoint growth is inside [-2, 2]",
          f"max |mpgr| = {float(np.max(np.abs(mpgr_cell))):.6f}")
    check_reconstruction(mpgr_cell, w_cell, d, mpgr_c, nz, n_frame)

    # County-years where the identity cannot hold, excluded from EVERY column by
    # the same rule so all three estimate on identical county-years (A.1.6).
    excl = t28.excluded_county_years(args.rung, emp, ccg_arr, frame, h)
    bad = np.array([(int(c), int(y)) in excl for c, y in
                    zip(frame["ccg"], frame["year"])])
    keep_cy = nz & ~bad
    print(f"\n[A.1.6] {int((~keep_cy).sum()):,} county-years excluded "
          f"({int((~nz).sum()):,} with no county midpoint denominator, "
          f"{int(bad.sum()):,} in the reconstruction exclusion set); "
          f"{int(keep_cy.sum()):,} estimated")

    shock = frame["shock"].to_numpy()
    zinst = frame["z"].to_numpy()
    ccg = frame["ccg"].to_numpy()
    yr = frame["year"].to_numpy()

    cy = d["f_cy"]
    cell_keep = keep_cy[cy]
    fits = {}

    fits["county"] = t28.fit2sls(
        mpgr_c[keep_cy], shock[keep_cy], zinst[keep_cy],
        np.ones(int(keep_cy.sum())),
        ccg[keep_cy], yr[keep_cy], ccg[keep_cy], singleton=False)

    fits["cell_cy"] = t28.fit2sls(
        mpgr_cell[cell_keep], shock[cy][cell_keep], zinst[cy][cell_keep],
        w_cell[cell_keep],
        ccg[cy][cell_keep], yr[cy][cell_keep], ccg[cy][cell_keep], singleton=False)

    fits["cell_indyear"] = t28.fit2sls(
        mpgr_cell[cell_keep], shock[cy][cell_keep], zinst[cy][cell_keep],
        w_cell[cell_keep],
        ccg[cy][cell_keep], d["f_ky"][cell_keep], ccg[cy][cell_keep], singleton=False)

    # 28's own column (3): the unit effect is the CELL, not the county. Singletons
    # are removed here because a cell effect can orphan rows, which the county
    # effect never does at four observations per cell.
    fits["cell_cellfe"] = t28.fit2sls(
        mpgr_cell[cell_keep], shock[cy][cell_keep], zinst[cy][cell_keep],
        w_cell[cell_keep],
        d["f_cell"][cell_keep], d["f_ky"][cell_keep], ccg[cy][cell_keep],
        singleton=True)

    LABEL = {"county": "(1) county, unweighted, ccg + year",
             "cell_cy": f"(2) county x {args.rung}, MPGR weight, ccg + year",
             "cell_indyear": f"(3) county x {args.rung}, MPGR weight, "
                             f"ccg + industry x year",
             "cell_cellfe": f"(4) county x {args.rung}, MPGR weight, "
                            f"ccg x industry + industry x year"}
    print("\n" + "=" * 78)
    print("  2SLS: midpoint employment growth on the contract shock")
    print("=" * 78)
    print(f"  {'specification':58s} {'beta':>12s} {'se':>10s} {'F':>9s} {'N':>12s}")
    for k in ("county", "cell_cy", "cell_indyear", "cell_cellfe"):
        r = fits[k]
        print(f"  {LABEL[k]:58s} {r['beta']:12.6f} {r['se']:10.6f} "
              f"{r['F']:9.2f} {r['n']:12,}")

    b1, b2 = fits["county"]["beta"], fits["cell_cy"]["beta"]
    s1, s2 = fits["county"]["se"], fits["cell_cy"]["se"]
    print(f"\n  positive MPGR weight on {int((w_cell[cell_keep] > 0).sum()):,} of "
          f"{int(cell_keep.sum()):,} cell rows; "
          f"{int((w_cell[cell_keep] < 0).sum()):,} negative")
    check(abs(b1 - b2) < TOL_INVAR,
          "Prop 1  (1) and (2) agree: both fixed effects are functions of (c, t)",
          f"|beta_cell - beta_county| = {abs(b1 - b2):.3e} against "
          f"{TOL_INVAR:.0e}; {b1:.10f} against {b2:.10f}")
    r1 = se_uncorrected(mpgr_c[keep_cy], shock[keep_cy], zinst[keep_cy],
                        np.ones(int(keep_cy.sum())),
                        ccg[keep_cy], yr[keep_cy], ccg[keep_cy])
    r2 = se_uncorrected(mpgr_cell[cell_keep], shock[cy][cell_keep],
                        zinst[cy][cell_keep], w_cell[cell_keep],
                        ccg[cy][cell_keep], yr[cy][cell_keep], ccg[cy][cell_keep])
    rel = abs(r1 - r2) / max(abs(r1), 1e-300)
    check(rel < TOL_INVAR_SE,
          "A.1.6   (1) and (2) carry the same UNCORRECTED clustered variance",
          f"relative gap {rel:.3e} against {TOL_INVAR_SE:.0e}; "
          f"{r1:.12f} against {r2:.12f}")
    print(f"  the corrected SEs, {s1:.6f} and {s2:.6f}, differ only through "
          f"G/(G-1) (N-1)/(N-K),")
    print(f"  which reads N = {fits['county']['n']:,} at the county and "
          f"N = {fits['cell_cy']['n']:,} at the cells.")

    b3 = fits["cell_indyear"]["beta"]
    print(f"\n  (2) -> (3), industry x year: {b2:.6f} -> {b3:.6f}, "
          f"a move of {b3 - b2:+.6f} "
          f"({100 * (b3 - b2) / b2:+.1f}% of the level in (2))")
    print("  Industry x year is not a function of (c, t), so Proposition 1 does not")
    print("  apply to it and the move is the object of interest, not a failure.")

    b4, s4 = fits["cell_cellfe"]["beta"], fits["cell_cellfe"]["se"]
    print(f"\n  (3) -> (4), the unit effect moves from the county to the CELL: "
          f"{b3:.6f} -> {b4:.6f}")
    print(f"  se {fits['cell_indyear']['se']:.6f} -> {s4:.6f}; "
          f"{fits['cell_cellfe']['n_dropped']:,} rows dropped as singletons")
    print(f"  This is 28_table_d1.py's own column (3) specification, run on the "
          f"candidate instrument.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / (f"coeff_candidate_iv_mpgr_{args.period}_h{h}"
                        f"_{args.share}_{args.rung}.csv")
    pd.DataFrame([
        dict(spec=k, label=LABEL[k], beta=fits[k]["beta"], se=fits[k]["se"],
             first_stage_F=fits[k]["F"], n=fits[k]["n"], period=args.period,
             horizon=h, share=args.share, rung=args.rung,
             winsor=f"iq{args.level:g}_year")
        for k in ("county", "cell_cy", "cell_indyear", "cell_cellfe")
    ]).to_csv(out, index=False, float_format="%.15g")
    print(f"\n  wrote {out.relative_to(REPO_ROOT)}")

    n_fail = len(w22.failures)
    print(f"\n{'=' * 78}\n  {n_fail} checks failed, {time.time() - t0:.1f}s"
          f"\n{'=' * 78}")
    tee.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
