#!/usr/bin/env python3
"""Plan section 2.2's four assertions: the gate every step-2 estimate has to pass.

    PLAN_macro_fiscal_multiplier.md section 2.2

    "The only thing that has to be settled first is that the construction is
     sound, and that comes down to four assertions:
       - cells sum to the county total exactly, in every county and every year,
         residual cells included;
       - the same counties appear at both levels;
       - the county shock survives industry-by-year fixed effects, meaning almost
         all of its variation is still there in the estimation sample;
       - weighted cell midpoint growth reconstructs county midpoint growth
         exactly, county by county and year by year."

Nothing is estimated here beyond the first stage assertion 3 needs. Part B of
INVARIANCE_TEST_PROTOCOL.md is explicit that the channel statistics come AFTER a
coefficient has moved, not before, so no channel is computed and none belongs here.

THE SETTINGS, AND WHY EACH ONE

    h = 2               the paper's horizon and step 1's headline
    z_fixed_7074        1970-1974, FIVE years, so the fixed and rolling loadings
                        differ in PLACEMENT and not in length. 19_first_stage_table
                        replaced `fixed` (1969-1974) on 2026-08-13 and its docstring
                        says "1969-1974 is discarded". That column is NOT in
                        instrument_ccg_year.parquet, so it is built in memory by
                        22's build_exposure_panel - imported, not restated, so this
                        file and the winsorization exhibit cannot end up on
                        different instruments.
    pre only, 1977-1997 post-1998 is dropped entirely, on instruction. The first
                        stage does not survive it at any rung (step 1 section 1.9.5).
    naics2, naics4_alloc  two pairs, county x NAICS2 and county x NAICS4. The county
                        is the baseline in both. naics4_raw is not a rung: pre-1998
                        its cells mix genuine industries with EFSY root buckets.

WHY THE PRE WINDOW IS 1977-1997 AND NOT 1975-1998

t-h has to reach EFSY's first year, 1975, so the first estimated year is 1977; and
1998-1999 straddle the CBP NAICS vintage break at the cell level, so they are
estimated nowhere. periods_for() in 22_winsor_construction_error.py owns those
bounds and is imported rather than restated.

fixed_7074 contaminates every year with y - h <= 1974, which at h=2 is {1975, 1976}
- both already outside the window. So contaminated years = 0 and step 1's design
window IS this window. first_stage_table.csv agrees: county, unweighted, h=2, era
pre, fixed_7074, unit+year FE gives 1977-1997, N = 65,085, contaminated_years = 0.

WHAT ASSERTION 3 ACTUALLY TESTS, AND THE GAP ITS FIRST RUN FOUND

Start from the algebra. If every county carries every industry, unweighted
industry x year fixed effects remove EXACTLY what year fixed effects remove from a
county-level regressor: the (m, t) group mean of a shock that varies only in (c, t)
is the year mean, the same for every m.

The first run of this file, on 2026-08-14, found that it did NOT hold, and the
reason was a real gap. 11_build_employment_cells.py defaulted to
--industry-support county-own, which gave a county rows only for the industries it
had EVER reported, so the support was the observed (ccg, code) pairs and not the
cross product - 79,818 a year against 3,101 x 26 at naics2, and 745,143 against
3,101 x 303 at naics4, short by 20.7%. That is a gap against plan section 2.1,
which asks for cells with no employment to be "present as zeros rather than absent
as missing, because their absence is the object of study". THE DEFAULT WAS CHANGED
TO national AND THE PANEL REBUILT. The support is now exact at both rungs.

Check 3a is nevertheless still written as the MECHANISM rather than as the bare
identity, because the mechanism is what is falsifiable: the residual difference must
be exactly zero on every (m, t) group that carries the full in-frame county set, and
may be nonzero only where a group is short. A merge error puts a difference on a
COMPLETE group; a support shortfall cannot. Asserting the identity outright would
have conflated the two, and on the county-own panel it would simply have failed
without saying why.

The rebuild is its own confirmation. Every group is now complete, the collapse is
exact - the year and industry x year residuals differ by 1.8e-14 at naics4, against
26.58% of sd(shock) before - and not one coefficient in the winsorization exhibit
moved, because the added cells are empty in both periods and carry zero weight.

The MPGR-weighted version is the binding one either way, and it is unaffected by any
of this. Under w_im the cell weights differ across m, the collapse fails for a
second and quite separate reason, and the retained share is a real number - and it
is the weighting Table D1 Panel B uses.

Output:
    batch_logs/macro_version/27_cell_construction_checks.log
    batch_logs/macro_version/cell_construction_checks.csv
    step_2/result/TableC1_cell_construction.tex
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
RESULT_DIR = HERE.parents[2] / "result"

sys.path.insert(0, str(STEP1_CODE))
import dcads_common as dc  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 22 owns the window bounds, the provenance gate, the in-memory exposure build,
# step 1's sample screen, the aligned-array cell reader and the demeaner. Importing
# them is the point: a restatement is how this file would end up gating a different
# sample from the one the winsorization exhibit and Table D1 estimate on. check()
# and failures live there too, so there is exactly one failure list in the process,
# and bi - 22's own copy of 14 - is reached through it so both files mutate the
# same module object rather than two that could drift apart.
w22 = _load("winsor", STEP2_CODE / "22_winsor_construction_error.py")
check, Tee, periods_for = w22.check, w22.Tee, w22.periods_for

EXPOSURE = w22.EXPOSURE_DEFAULT
EXPOSURE_WINDOW = w22.NEW_WINDOWS[EXPOSURE]
HORIZON = 2
RUNGS_ALL = w22.RUNGS_ALL

# Step 1's frozen universe and the estimation support. 3,101 CCGs exist; Kalawao HI
# has no BEA income denominator by declaration, so 3,100 carry a shock.
N_UNIVERSE = 3101
N_SUPPORT = 3100

# Declared tolerances. Each is a measured quantity plus headroom. The measured value
# goes in the comment so a future drift is visible rather than absorbed.
TOL_SUM_REL = 1e-12     # assertion 1, cells sum to the county, relative
TOL_RECON = 1e-12       # assertion 4, the reconstruction identity, absolute
TOL_W1 = 1e-12          # weights sum to 1 within a county-year
TOL_SHOCK = 0.0         # in-memory shock against the parquet's: EXACT, not close
TOL_COLLAPSE = 1e-12    # unweighted industry x year against year, absolute
TOL_ORTHO = 1e-9        # post-demeaning orthogonality to both fixed effects
TOL_LAG = 1e-9          # county cell lag against step 1's emp_lag, relative

MPGR_BOUND = 2.0


# --------------------------------------------------------------------------------
# The cell support. load_rung asserts it is identical in every year; the codes come
# from the first file, which is why that assertion has to come first.
# --------------------------------------------------------------------------------

def rung_codes(rung, year0):
    f = pd.read_parquet(dc.EMP_OUT_DIR / rung / f"y{year0}.parquet",
                        columns=["ccg", "code"])
    return f["ccg"].to_numpy(np.int64), f["code"].to_numpy(np.int64)


def support_completeness(rung, ccg_arr, code_arr):
    """How far the support is from the full county x industry cross product.

    Plan section 2.1 requires cells with no employment to be "present as zeros rather
    than absent as missing, because their absence is the object of study", so the
    support has to BE the cross product and `sup_short` has to be zero.

    Kept as a measurement rather than a check on purpose. It is the only line that
    would notice 11_build_employment_cells.py being rebuilt under
    --industry-support county-own, which is still a valid flag and was the default
    until 2026-08-14; a silent revert would otherwise show up nowhere except as
    check 3a's tail changing wording. If this prints a nonzero shortfall, the panel
    was built with the wrong flag - rerun 11 with no arguments.
    """
    n_ccg, n_code = int(len(np.unique(ccg_arr))), int(len(np.unique(code_arr)))
    full, have = n_ccg * n_code, len(ccg_arr)
    per_code = pd.Series(ccg_arr).groupby(pd.Series(code_arr)).nunique()
    print(f"\n[2*] support completeness ({rung})")
    print(f"    {have:,} pairs against a full cross product of {n_ccg:,} x "
          f"{n_code:,} = {full:,}: short by {full - have:,} "
          f"({100 * (full - have) / full:.3f}%)")
    print(f"    industries carried by every CCG: "
          f"{int((per_code == n_ccg).sum())} of {n_code}; "
          f"the least widely held carries {int(per_code.min()):,}")
    check(full == have, f"2e the support IS the cross product ({rung})",
          f"plan section 2.1: absent cells have to be present as zeros. If this "
          f"fails, 11_build_employment_cells.py was run with "
          f"--industry-support county-own; rerun it with no arguments")
    return dict(sup_pairs=have, sup_full=full, sup_short=full - have,
                sup_n_ccg=n_ccg, sup_n_code=n_code,
                sup_min_ccg_per_code=int(per_code.min()),
                sup_complete_codes=int((per_code == n_ccg).sum()))


# --------------------------------------------------------------------------------
# Assertion 1. Cells sum to the county total, residual cells included.
# --------------------------------------------------------------------------------

def assert_sums(rung, emp, ccg_arr, years, county_src):
    """Bincount over a dense CCG index, per year, against two independent totals.

    Nothing is filtered on `code`. EFSY's unclassifiable roots are encoded by
    11_build_employment_cells.py as RESIDUAL3 / RESIDUAL2 integers and are ordinary
    rows here, which is what "residual cells included" means. At naics2 every code
    has a sector so no residual survives; at naics4_alloc the ladder has pushed them
    down. Both are summed the same way and neither is special-cased.
    """
    print(f"\n[1] assertion 1, cells sum to the county total ({rung})")
    ccgs, idx = np.unique(ccg_arr, return_inverse=True)
    rows, worst, worst_year, worst_src = [], 0.0, None, ""
    for y in years:
        tot = np.bincount(idx, weights=emp[y], minlength=len(ccgs))
        s = pd.DataFrame({"ccg": ccgs, "cells": tot})
        for src, ref in county_src.items():
            r = ref[ref["year"] == y][["ccg", "emp"]]
            j = s.merge(r, on="ccg", how="inner")
            den = np.maximum(np.abs(j["emp"].to_numpy()), 1.0)
            dev = float((np.abs(j["cells"] - j["emp"]) / den).max())
            rows.append(dict(rung=rung, year=y, metric=f"a1_reldev_{src}",
                             value=dev, n=len(j)))
            if dev > worst:
                worst, worst_year, worst_src = dev, y, src
    n_cellyears = len(ccg_arr) * len(years)
    check(worst < TOL_SUM_REL, f"1  cells sum to the county in every year ({rung})",
          f"max relative deviation {worst:.3e} < {TOL_SUM_REL:.0e}, worst "
          f"{worst_year} against {worst_src}; {n_cellyears:,} cell-years, "
          f"{len(ccgs):,} CCGs x {len(years)} years, nothing filtered on code")
    return rows, dict(a1_worst=worst, a1_worst_year=worst_year,
                      a1_n_cellyears=n_cellyears)


# --------------------------------------------------------------------------------
# Assertion 4. Weighted cell midpoint growth reconstructs county midpoint growth.
# --------------------------------------------------------------------------------

def assert_reconstruction(rung, emp, ccg_arr, frame, h):
    """The invariance Proposition 1 delivers, asserted county by county and year by
    year, and the four supporting checks that go with it.

        mpgr_im = 2 (E_imt - E_im,t-h) / (E_imt + E_im,t-h)
        w_im    = (E_imt + E_im,t-h) / (E_ct + E_c,t-h)     COUNTY denominator
        sum_m w_im mpgr_im  ==  mpgr_ct

    THE GUARD IS != 0, NEVER > 0. A previous build guarded the growth rate on
    `den > 0`, which silently set midpoint growth to zero for EFSY's
    negative-employment cells while leaving their weights intact. Reconstruction
    then failed in 82 county-years at 7.4e-4 - small enough to be mistaken for a
    tolerance and large enough to be wrong. The only genuinely undefined case is a
    cell empty in BOTH periods.

    A zero midpoint denominator has exactly two causes, since den = 0 iff
    E_t = -E_t-h:

        E_t = E_t-h = 0   the cell is empty in both periods. Weight zero, no
                          growth rate, contributes nothing to either side, and the
                          identity is untouched.
        E_t = -E_t-h != 0 SELF-CANCELLING. Weight zero, so it drops out of the
                          reconstruction - but it contributes -2 E_t-h to the
                          county numerator. The identity cannot hold in that
                          county-year and it is excluded BY NAME, from both levels
                          by the same rule, with the count reported.
    """
    print(f"\n[4] assertion 4, the reconstruction ({rung})")
    ccgs, idx = np.unique(ccg_arr, return_inverse=True)
    pos_of = {c: i for i, c in enumerate(ccgs)}
    nb = len(ccgs)

    worst, worst_year = 0.0, None
    worst_w, n_excl, n_selfcanc, n_zero_den_c = 0.0, 0, 0, 0
    n_unbounded, n_unbounded_neg, n_zero_w_other = 0, 0, 0
    worst_c1, worst_lag = 0.0, 0.0
    rows = []

    for y, sub in frame.groupby("year", sort=True):
        lag, cur = emp[y - h], emp[y]
        in_frame = np.zeros(nb, bool)
        p = np.array([pos_of[c] for c in sub["ccg"].to_numpy()])
        in_frame[p] = True
        ok = in_frame[idx]

        den = cur + lag
        nz = den != 0.0                       # the guard. NOT den > 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            mpgr = np.where(nz, 2.0 * (cur - lag) / np.where(nz, den, 1.0), 0.0)

        # 4c. A zero-weight row is empty or self-cancelling, and nothing else.
        zw = ok & ~nz
        empty = zw & (lag == 0.0) & (cur == 0.0)
        selfc = zw & (lag != 0.0)
        n_zero_w_other += int((zw & ~empty & ~selfc).sum())
        n_selfcanc += int(selfc.sum())

        # 4b. |mpgr| <= 2 wherever employment is non-negative. Every exceedance has
        # to be a negative-employment cell, which is what makes the bound a
        # statement about EFSY rather than about the formula.
        nonneg = ok & nz & (lag >= 0) & (cur >= 0)
        over = ok & nz & (np.abs(mpgr) > MPGR_BOUND + 1e-12)
        n_unbounded += int(over.sum())
        n_unbounded_neg += int((over & ~nonneg).sum())

        s_cur = np.bincount(idx[ok], weights=cur[ok], minlength=nb)
        s_lag = np.bincount(idx[ok], weights=lag[ok], minlength=nb)
        s_num = np.bincount(idx[ok & nz], weights=(den * mpgr)[ok & nz], minlength=nb)
        s_den = np.bincount(idx[ok & nz], weights=den[ok & nz], minlength=nb)
        s_bad = np.bincount(idx[selfc], weights=np.ones(int(selfc.sum())), minlength=nb)

        den_c = s_cur + s_lag
        good = (den_c[p] != 0.0) & (s_bad[p] == 0.0)
        n_zero_den_c += int((den_c[p] == 0.0).sum())
        n_excl += int((~good).sum())

        with np.errstate(divide="ignore", invalid="ignore"):
            mpgr_c = np.where(den_c != 0.0,
                              2.0 * (s_cur - s_lag) / np.where(den_c != 0.0, den_c, 1.0),
                              np.nan)
            recon = np.where(den_c != 0.0,
                             s_num / np.where(den_c != 0.0, den_c, 1.0), np.nan)
            wsum = np.where(den_c != 0.0,
                            s_den / np.where(den_c != 0.0, den_c, 1.0), np.nan)

        dv = float(np.abs(recon[p][good] - mpgr_c[p][good]).max()) if good.any() else 0.0
        wv = float(np.abs(wsum[p][good] - 1.0).max()) if good.any() else 0.0
        # c1. County midpoint growth is bounded because county totals are
        # non-negative even where a cell is not.
        c1 = float(np.abs(mpgr_c[p][good]).max()) if good.any() else 0.0
        # c2. The county lag built from the cells IS step 1's emp_lag.
        el = sub["emp_lag"].to_numpy()
        lg = float((np.abs(s_lag[p] - el) / np.maximum(np.abs(el), 1.0)).max())

        rows.append(dict(rung=rung, year=y, metric="a4_recon_absdev", value=dv,
                         n=int(good.sum())))
        rows.append(dict(rung=rung, year=y, metric="a4_wsum_absdev", value=wv,
                         n=int(good.sum())))
        rows.append(dict(rung=rung, year=y, metric="c1_max_abs_county_mpgr", value=c1,
                         n=int(good.sum())))
        rows.append(dict(rung=rung, year=y, metric="c2_lag_reldev", value=lg,
                         n=len(el)))
        if dv > worst:
            worst, worst_year = dv, y
        worst_w = max(worst_w, wv)
        worst_c1 = max(worst_c1, c1)
        worst_lag = max(worst_lag, lg)

    check(worst < TOL_RECON,
          f"4  weighted cell midpoint growth reconstructs the county ({rung})",
          f"max |sum_m w mpgr - mpgr_c| = {worst:.3e} < {TOL_RECON:.0e}, worst "
          f"{worst_year}; {n_excl:,} county-years excluded by name "
          f"({n_zero_den_c:,} zero county denominator, "
          f"{n_excl - n_zero_den_c:,} carrying a self-cancelling cell)")
    check(worst_w < TOL_W1, f"3  MPGR weights sum to 1 within a county-year ({rung})",
          f"max |sum_m w - 1| = {worst_w:.3e} < {TOL_W1:.0e}")
    check(worst_c1 <= MPGR_BOUND + 1e-12,
          f"c1 county midpoint growth is bounded on [-2, 2] ({rung})",
          f"max |mpgr_c| = {worst_c1:.6f}")
    check(worst_lag < TOL_LAG, f"c2 the county cell lag IS step 1's emp_lag ({rung})",
          f"max relative deviation {worst_lag:.3e} < {TOL_LAG:.0e}")
    check(n_unbounded == n_unbounded_neg,
          f"4b every |mpgr| > 2 is a negative-employment cell ({rung})",
          f"{n_unbounded:,} cell-years exceed the bound, {n_unbounded_neg:,} of them "
          f"carry negative employment; the bound holds wherever E >= 0")
    check(n_zero_w_other == 0,
          f"4c every zero-weight row is empty or self-cancelling ({rung})",
          f"{n_zero_w_other} rows are neither; {n_selfcanc:,} self-cancelling "
          f"cell-years, the rest empty in both periods")
    return rows, dict(a4_worst=worst, a4_wsum=worst_w, a4_excluded=n_excl,
                      a4_zero_den_c=n_zero_den_c, a4_selfcanc=n_selfcanc,
                      a4_unbounded=n_unbounded, c1_max=worst_c1, c2_worst=worst_lag)


# --------------------------------------------------------------------------------
# Assertion 3. The county shock survives industry x year fixed effects.
# --------------------------------------------------------------------------------

def _wdemean1(x, f, w, n):
    """Weighted within-group demeaning on one factor."""
    s = np.bincount(f, weights=w, minlength=n)
    s[s == 0.0] = 1.0
    m = np.bincount(f, weights=w * x, minlength=n) / s
    return x - m[f]


def _wsd(x, w):
    mu = float(w @ x) / float(w.sum())
    return float(np.sqrt((w @ (x - mu) ** 2) / w.sum()))


def _ortho(x, f, w, n):
    """max |weighted group mean| after demeaning - did the projection converge."""
    s = np.bincount(f, weights=w, minlength=n)
    s[s == 0.0] = 1.0
    return float(np.abs(np.bincount(f, weights=w * x, minlength=n) / s).max())


def _cluster_F(y, x, w, cl, n_fe):
    """(b/se)^2 for y on x after absorption, clustered on cl.

    The instrument is single, so the first-stage t squared IS the Kleibergen-Paap F
    under the same variance estimator - the same reason 14's fit_rung computes it
    that way. CRV1's finite-sample correction, G/(G-1) * (N-1)/(N-K), is applied so
    the number is comparable to the ones step 1 reports.
    """
    xwx = float(x @ (w * x))
    if xwx <= 0:
        return np.nan
    b = float(x @ (w * y)) / xwx
    e = y - b * x
    nb = int(cl.max()) + 1
    G = int(np.unique(cl).size)          # groups PRESENT, not the width of the index
    g = np.bincount(cl, weights=w * x * e, minlength=nb)
    N, K = len(y), n_fe + 1
    ssc = (G / max(G - 1, 1)) * ((N - 1) / max(N - K, 1))
    v = ssc * float(g @ g) / xwx ** 2
    return (b / np.sqrt(v)) ** 2 if v > 0 else np.nan


def assert_survives_fe(rung, emp, ccg_arr, code_arr, frame, h):
    """Retained variation in the shock under each admissible absorption.

    ccg x year is absent on purpose. It is the one fixed effect that would absorb
    the treatment (protocol A.1.4), and no amount of retained variation would make
    it admissible.
    """
    print(f"\n[5] assertion 3, the shock survives industry x year FE ({rung})")
    ccgs, cidx = np.unique(ccg_arr, return_inverse=True)
    pos_of = {c: i for i, c in enumerate(ccgs)}
    codes, kidx = np.unique(code_arr, return_inverse=True)
    n_code = len(codes)
    cell_id = np.arange(len(ccg_arr), dtype=np.int64)   # (ccg, code), fixed across years

    sh, zz, ww, f_cell, f_ky, f_y, f_cl = [], [], [], [], [], [], []
    n_clamped = [0]
    years = sorted(frame["year"].unique())
    n_in_year = np.zeros(len(years))          # counties in the frame, by year
    for ynum, (y, sub) in enumerate(frame.groupby("year", sort=True)):
        lag, cur = emp[y - h], emp[y]
        in_frame = np.zeros(len(ccgs), bool)
        p = np.array([pos_of[c] for c in sub["ccg"].to_numpy()])
        in_frame[p] = True
        ok = in_frame[cidx]
        n_in_year[ynum] = len(p)

        den = cur + lag
        s_den = np.bincount(cidx[ok], weights=den[ok], minlength=len(ccgs))
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(s_den[cidx] != 0.0,
                         den / np.where(s_den[cidx] != 0.0, s_den[cidx], 1.0), 0.0)

        m_sh = np.full(len(ccgs), np.nan)
        m_z = np.full(len(ccgs), np.nan)
        m_sh[p] = sub["shock"].to_numpy()
        m_z[p] = sub["z"].to_numpy()

        sh.append(m_sh[cidx][ok])
        zz.append(m_z[cidx][ok])
        # A DECLARED DEVIATION, and the only one in this function. w_im is negative
        # wherever the cell's midpoint denominator is, which EFSY's negative
        # employment makes possible; a negative weight has no meaning in a weighted
        # sd and would make the variance imaginable rather than imaginary. They are
        # clamped to zero and COUNTED, not silently absorbed. This affects the
        # assertion 3 diagnostic only - assertion 4 uses the weights unclamped,
        # which is why it is the one that binds.
        n_clamped[0] += int((w[ok] < 0).sum())
        ww.append(np.maximum(w[ok], 0.0))
        f_cell.append(cell_id[ok])
        f_ky.append(kidx[ok] * len(years) + ynum)
        f_y.append(np.full(int(ok.sum()), ynum, np.int64))
        f_cl.append(cidx[ok])

    sh = np.concatenate(sh)
    zz = np.concatenate(zz)
    w_mp = np.concatenate(ww)
    f_cell = np.concatenate(f_cell)
    f_ky = np.concatenate(f_ky)
    f_y = np.concatenate(f_y)
    f_cl = np.concatenate(f_cl)
    one = np.ones(len(sh))
    n_ky, n_y, n_cell = n_code * len(years), len(years), len(ccg_arr)
    print(f"    {len(sh):,} cell-years, {n_cell:,} ccg x industry levels, "
          f"{n_ky:,} industry x year levels")
    print(f"    DECLARED: {n_clamped[0]:,} cell-years carry a NEGATIVE midpoint "
          f"weight and are clamped to zero for this diagnostic only "
          f"({100 * n_clamped[0] / max(len(sh), 1):.4f}%); assertion 4 uses them "
          f"unclamped")

    out, rows = {}, []

    # 3a. THE COLLAPSE, tested as a mechanism and not asserted as an identity.
    #
    # Unweighted industry x year fixed effects would remove exactly what year fixed
    # effects remove IF every (m, t) group carried every county in the frame. They
    # do not: the support is the observed (ccg, code) pairs and not the cross
    # product, so a group is a subset of the frame and its mean shock is not the
    # year's. TWO things follow, and only the second is testable.
    #
    # Not testable, and not true: that a COMPLETE group's mean equals the year mean.
    # The year mean weights a county by the number of industries it carries, which
    # differs across counties precisely because the support is short. So a complete
    # group differs from the year mean too, and by an amount that says nothing.
    #
    # Testable, and the real identity: EVERY COMPLETE GROUP IN A YEAR HAS THE SAME
    # MEAN, namely the plain mean of the shock over the frame's counties that year.
    # That is what "the shock varies only in (c, t)" buys, it holds however short
    # the support is, and a merge error - a shock landing on the wrong county, a
    # misaligned lag - breaks it immediately.
    r_y = _wdemean1(sh, f_y, one, n_y)
    r_ky = _wdemean1(sh, f_ky, one, n_ky)
    raw_u = _wsd(sh, one)
    grp_n = np.bincount(f_ky, minlength=n_ky)
    grp_m = np.bincount(f_ky, weights=sh, minlength=n_ky) / np.maximum(grp_n, 1)
    # n_in_year comes from the frame itself, not from the largest group, because
    # "the largest group" would define completeness in terms of the shortfall.
    complete = grp_n == n_in_year[np.arange(n_ky) % n_y]
    spread, n_comp = 0.0, 0
    for j in range(n_y):
        sel = complete & (np.arange(n_ky) % n_y == j)
        if sel.sum() > 1:
            spread = max(spread, float(grp_m[sel].max() - grp_m[sel].min()))
        n_comp += int(sel.sum())
    dev_all = float(np.abs(r_y - r_ky).max())
    n_grp = int((grp_n > 0).sum())
    # The tail of the message depends on whether the support is complete, because
    # the same sentence would otherwise claim a shortfall that is no longer there.
    # Since 2026-08-14 the panel is built on the full cross product (11's
    # --industry-support national), every group is complete, and the collapse IS
    # exact - which is the confirmation that the shortfall was the whole cause.
    tail = (f"and against year FE they differ by {dev_all:.3e} "
            f"({100 * dev_all / raw_u:.2f}% of sd(shock)), so the collapse is exact "
            f"- every industry carries every county"
            if n_comp == n_grp else
            f"Against year FE the residuals differ by {dev_all:.3e} "
            f"({100 * dev_all / raw_u:.2f}% of sd(shock)), which is the support "
            f"shortfall and nothing else - the year mean weights a county by how "
            f"many industries it carries")
    check(spread < TOL_COLLAPSE,
          f"3a every complete industry x year group shares one mean ({rung})",
          f"max spread {spread:.3e} < {TOL_COLLAPSE:.0e} over {n_comp:,} complete "
          f"groups of {n_grp:,}; the shock varies only in (c,t), so any group "
          f"carrying the whole frame must see the same mean. " + tail)
    out["fe_ky_unw"] = _wsd(r_ky, one) / raw_u
    out["fe_y_unw"] = _wsd(r_y, one) / raw_u
    out["collapse_dev_rel"] = dev_all / raw_u

    # 3b. Weighted by the MPGR weight, which is Table D1 Panel B's weighting. Here
    # the collapse fails and the number is real.
    raw_w = _wsd(sh, w_mp)
    r_ky_w = _wdemean1(sh, f_ky, w_mp, n_ky)
    z_ky_w = _wdemean1(zz, f_ky, w_mp, n_ky)
    out["fe_ky_mpgr"] = _wsd(r_ky_w, w_mp) / raw_w
    out["F_ky_mpgr"] = _cluster_F(r_ky_w, z_ky_w, w_mp, f_cl, n_ky)

    # 3c. Both admissible effects at once, which is Table D1 columns (3) and (5).
    M = w22.demean(np.column_stack([sh, zz]), f_cell, f_ky, w_mp)
    o1 = _ortho(M[:, 0], f_cell, w_mp, n_cell)
    o2 = _ortho(M[:, 0], f_ky, w_mp, n_ky)
    check(max(o1, o2) < TOL_ORTHO,
          f"3c the two-way projection converged ({rung})",
          f"max |weighted group mean| after demeaning: {o1:.2e} on ccg x industry, "
          f"{o2:.2e} on industry x year")
    out["fe_both_mpgr"] = _wsd(M[:, 0], w_mp) / raw_w
    out["F_both_mpgr"] = _cluster_F(M[:, 0], M[:, 1], w_mp, f_cl, n_cell + n_ky)

    for k, v in out.items():
        rows.append(dict(rung=rung, year=-1, metric=f"a3_{k}", value=float(v),
                         n=len(sh)))
    out["a3_n_cellyears"] = len(sh)
    out["a3_zero_cells_pct"] = 100.0 * float((w_mp == 0.0).mean())

    print(f"    sd retained, unweighted        year FE {out['fe_y_unw']:.6f}   "
          f"industry x year FE {out['fe_ky_unw']:.6f}")
    print(f"    sd retained, MPGR-weighted     industry x year FE "
          f"{out['fe_ky_mpgr']:.6f}   + ccg x industry FE {out['fe_both_mpgr']:.6f}")
    print(f"    first-stage F, MPGR-weighted   industry x year FE "
          f"{out['F_ky_mpgr']:.1f}   + ccg x industry FE {out['F_both_mpgr']:.1f}")
    check(out["fe_ky_mpgr"] > 0.5,
          f"5  the shock survives industry x year FE ({rung})",
          f"{100 * out['fe_ky_mpgr']:.1f}% of its weighted sd is still there; "
          f"the plan asks for almost all of it")
    del sh, zz, w_mp, f_cell, f_ky, f_y, f_cl, M
    return rows, out


# --------------------------------------------------------------------------------
# The table.
# --------------------------------------------------------------------------------

def tex_table(res, sup, path, h, years):
    def g(rung, k, fmt="{:.1e}"):
        v = res[rung].get(k)
        return "--" if v is None else (fmt.format(v) if isinstance(v, float)
                                       else f"{v:,}")
    L = [
        r"% Plan section 2.2, the four construction assertions",
        f"% h={h}, z_{EXPOSURE}, {years[0]}-{years[-1]}, no dollar floor",
        r"\begin{tabular}{@{\extracolsep{\fill}}lcc}",
        r"\toprule",
        r"&\multicolumn{1}{c}{county $\times$ NAICS2}"
        r"&\multicolumn{1}{c}{county $\times$ NAICS4}\\",
        r"&\multicolumn{1}{c}{(1)}&\multicolumn{1}{c}{(2)}\\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{A. Cells sum to the county total, "
        r"residual cells included}}\\",
        rf"\quad max $|\sum_m E_{{im}}-E_i|/E_i$ &{g('naics2','a1_worst')}"
        rf"&{g('naics4_alloc','a1_worst')}\\",
        rf"\quad worst year &{g('naics2','a1_worst_year')}"
        rf"&{g('naics4_alloc','a1_worst_year')}\\",
        rf"\quad cell-years &{g('naics2','a1_n_cellyears')}"
        rf"&{g('naics4_alloc','a1_n_cellyears')}\\",
        r"\addlinespace",
        r"\multicolumn{3}{l}{\textit{B. The same counties at both levels}}\\",
        rf"\quad CCGs in the rung &{g('naics2','sup_n_ccg')}"
        rf"&{g('naics4_alloc','sup_n_ccg')}\\",
        rf"\quad CCGs in the estimation frame &{sup['n_frame']:,}&{sup['n_frame']:,}\\",
        rf"\quad support identical in every year &{sup['ident']}&{sup['ident']}\\",
        rf"\quad industries &{g('naics2','sup_n_code')}"
        rf"&{g('naics4_alloc','sup_n_code')}\\",
        rf"\quad pairs short of the cross product &{g('naics2','sup_short')}"
        rf"&{g('naics4_alloc','sup_short')}\\",
        r"\addlinespace",
        r"\multicolumn{3}{l}{\textit{C. The shock survives industry $\times$ "
        r"year fixed effects}}\\",
        rf"\quad sd retained, unweighted &{g('naics2','fe_ky_unw','{:.4f}')}"
        rf"&{g('naics4_alloc','fe_ky_unw','{:.4f}')}\\",
        rf"\quad sd retained, MPGR-weighted &{g('naics2','fe_ky_mpgr','{:.4f}')}"
        rf"&{g('naics4_alloc','fe_ky_mpgr','{:.4f}')}\\",
        rf"\quad \quad and with ccg $\times$ industry FE "
        rf"&{g('naics2','fe_both_mpgr','{:.4f}')}"
        rf"&{g('naics4_alloc','fe_both_mpgr','{:.4f}')}\\",
        rf"\quad first-stage $F$, MPGR-weighted &{g('naics2','F_ky_mpgr','{:.1f}')}"
        rf"&{g('naics4_alloc','F_ky_mpgr','{:.1f}')}\\",
        r"\addlinespace",
        r"\multicolumn{3}{l}{\textit{D. Weighted cell midpoint growth "
        r"reconstructs the county}}\\",
        rf"\quad max $|\sum_m w_{{im}}g_{{im}}-g_i|$ &{g('naics2','a4_worst')}"
        rf"&{g('naics4_alloc','a4_worst')}\\",
        rf"\quad max $|\sum_m w_{{im}}-1|$ &{g('naics2','a4_wsum')}"
        rf"&{g('naics4_alloc','a4_wsum')}\\",
        rf"\quad county-years excluded by name &{g('naics2','a4_excluded')}"
        rf"&{g('naics4_alloc','a4_excluded')}\\",
        rf"\quad \quad of which self-cancelling cell &{g('naics2','a4_selfcanc')}"
        rf"&{g('naics4_alloc','a4_selfcanc')}\\",
        rf"\quad $|g_{{im}}|>2$, all negative employment &{g('naics2','a4_unbounded')}"
        rf"&{g('naics4_alloc','a4_unbounded')}\\",
        r"\addlinespace",
        rf"Cells with zero MPGR weight (\%) "
        rf"&{g('naics2','a3_zero_cells_pct','{:.2f}')}"
        rf"&{g('naics4_alloc','a3_zero_cells_pct','{:.2f}')}\\",
        rf"County-years, {years[0]}--{years[-1]} &{sup['n_countyyear']:,}"
        rf"&{sup['n_countyyear']:,}\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", nargs="+", default=list(RUNGS_ALL),
                    choices=list(RUNGS_ALL))
    ap.add_argument("--horizon", type=int, default=HORIZON)
    args = ap.parse_args()

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    canonical = set(args.rung) == set(RUNGS_ALL) and args.horizon == HORIZON
    sfx = "" if canonical else ("__" + "-".join(sorted(args.rung)) + f"_h{args.horizon}")

    tee = Tee(dc.LOG_DIR / f"27_cell_construction_checks{sfx}.log")
    import pyfixest as pf
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
    t0 = time.time()
    h = args.horizon
    eras, straddle = periods_for(h)
    y0, y1 = eras["pre"]

    print("=" * 78)
    print("Plan section 2.2 - the four construction assertions")
    print("=" * 78)
    print(f"  h = {h}, exposure = {EXPOSURE} ({EXPOSURE_WINDOW[0]}-"
          f"{EXPOSURE_WINDOW[1]}), instrument = z_{EXPOSURE}")
    print(f"  rungs    {args.rung}, as county x NAICS2 and county x NAICS4")
    print(f"  window   pre {y0}-{y1}; straddle {straddle[0]}-{straddle[1]}, "
          f"estimated nowhere")
    print(f"  POST-{w22.NAICS_BREAK} IS DROPPED ENTIRELY, on instruction. A declared")
    print("  deviation from plan check 5, which asks for the weak half to be named")
    print("  rather than dropped. Step 1 section 1.9.5 is where it is named: the")
    print("  county first stage is 6.3 over 2002-2018 with the sign flipped.")
    print("\n  DECLARED DECISIONS, recorded before any number is read.")
    print("  - the midpoint guard is `den != 0`, never `den > 0`")
    print("  - no cell is dropped and no growth rate is winsorized (A.3.1, A.3.2)")
    print("  - county-years with a zero midpoint denominator, or carrying a")
    print("    self-cancelling cell, are excluded from assertion 4 BY NAME and from")
    print("    both levels by the same rule")
    print("  - counties with lagged employment <= 0 drop from the frame: step 1's")
    print("    own screen, an A.3.1 deviation, declared below")
    print("  - ccg x year appears nowhere. It would absorb the treatment (A.1.4)")

    iv_disk, _ = w22.provenance_gate(h)
    iv = w22.build_exposure_panel(h, EXPOSURE, iv_disk, (y0, y1))
    est_years = list(range(y0, y1 + 1))
    frame = w22.build_frame(iv, h, EXPOSURE, est_years)
    frame = frame[frame["year"].between(y0, y1)].reset_index(drop=True)
    print(f"    the pre frame: {len(frame):,} county-years, "
          f"{frame['ccg'].nunique():,} CCGs, {y0}-{y1}")

    rows, res, ccg_sets = [], {}, {}
    sup = dict(n_frame=int(frame["ccg"].nunique()), n_countyyear=len(frame))
    years_emp = list(range(y0 - h, y1 + 1))

    for rung in args.rung:
        print(f"\n{'=' * 78}\n  rung {rung}, employment {years_emp[0]}-"
              f"{years_emp[-1]}\n{'=' * 78}")
        emp, ccg_arr = w22.load_rung(rung, years_emp)
        _, code_arr = rung_codes(rung, years_emp[0])
        ccg_sets[rung] = set(np.unique(ccg_arr).tolist())
        sup["n_rung"] = len(ccg_sets[rung])
        sup["ident"] = "yes"

        county_src = {
            "efsy": pd.read_parquet(dc.EMP_OUT_DIR / "efsy_ccg_year.parquet",
                                    columns=["ccg", "year", "emp"]),
            "county_panel": pd.read_parquet(
                dc.PANEL_OUT_DIR / "county_panel.parquet",
                columns=["ccg", "year", "emp"]),
        }
        r1, m1 = assert_sums(rung, emp, ccg_arr, years_emp, county_src)
        m2 = support_completeness(rung, ccg_arr, code_arr)
        r4, m4 = assert_reconstruction(rung, emp, ccg_arr, frame, h)
        r3, m3 = assert_survives_fe(rung, emp, ccg_arr, code_arr, frame, h)
        rows += r1 + r4 + r3
        res[rung] = {**m1, **m2, **m4, **m3}
        del emp

    print("\n[2] assertion 2, the same counties appear at both levels")
    if len(ccg_sets) > 1:
        a, b = (ccg_sets[r] for r in args.rung[:2])
        check(a == b, "2a the two rungs carry the identical CCG set",
              f"{len(a):,} vs {len(b):,}, "
              f"{len(a ^ b)} in one and not the other")
    else:
        print("    one rung only; the cross-rung comparison needs both")
    for rung, s in ccg_sets.items():
        check(len(s) == N_UNIVERSE, f"2b the rung is step 1's full universe ({rung})",
              f"{len(s):,} CCGs against {N_UNIVERSE:,}")
    check(sup["n_frame"] == N_SUPPORT,
          "2c the estimation frame is step 1's frozen support",
          f"{sup['n_frame']:,} CCGs against {N_SUPPORT:,}; Kalawao HI has no BEA "
          f"income denominator by declaration")
    for rung, s in ccg_sets.items():
        miss = set(frame["ccg"].unique()) - s
        check(not miss, f"2d every estimated county exists at this rung ({rung})",
              f"{len(miss)} county in the frame and not in the cells")

    out = pd.DataFrame(rows)
    out.to_csv(dc.LOG_DIR / f"cell_construction_checks{sfx}.csv", index=False)
    print(f"\n  {len(out):,} measurements -> "
          f"{dc.LOG_DIR / f'cell_construction_checks{sfx}.csv'}")
    if canonical:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        p = tex_table(res, sup, RESULT_DIR / "TableC1_cell_construction.tex",
                      h, est_years)
        print(f"  table -> {p}")
    else:
        print(f"\n  PARTIAL RUN. The canonical outputs are left untouched; this run "
              f"writes to *{sfx}.* instead.")

    print("\n" + "=" * 78)
    print("  SUMMARY - the four assertions")
    print("=" * 78)
    for rung in args.rung:
        m = res[rung]
        print(f"  {rung}")
        print(f"    1  cells sum to the county          max rel dev "
              f"{m['a1_worst']:.2e}")
        print(f"    2  identical support                {sup['n_rung']:,} CCGs, "
              f"{sup['n_frame']:,} estimated")
        print(f"    3  shock survives industry x year   "
              f"{100 * m['fe_ky_mpgr']:.1f}% of the weighted sd retained, "
              f"F = {m['F_ky_mpgr']:.1f}")
        print(f"    4  reconstruction                   max abs dev "
              f"{m['a4_worst']:.2e}, {m['a4_excluded']:,} county-years excluded")

    print(f"\n  elapsed {time.time() - t0:.0f}s")
    if w22.failures:
        print(f"\nFAIL: {len(w22.failures)} check(s) failed", file=sys.stderr)
        for n, w in w22.failures:
            print(f"  {n}: {w}", file=sys.stderr)
        tee.close()
        return 1
    print("\nall checks passed")
    tee.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
