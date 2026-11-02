#!/usr/bin/env python3
"""The three channels of INVARIANCE_TEST_PROTOCOL.md B.1 Layer 2, quantified.

The county-level log coefficient and the county x NAICS4 log coefficient differ.
The protocol names three channels that account for a cross-level gap, and this
file builds each one as the protocol specifies rather than as a residual between
two regressions the author happened to run.

CHANNEL 1, implicit weights (Lemma 1, `3_disaggregation.tex` eq
`implicit_weight_decomp`). The cell-level log regression three times, changing
ONLY the weight:

    (i)   equal weight per cell
    (ii)  1/n_i, so every county carries the same total weight
    (iii) the cell's share of the county's LAGGED employment

    (ii) - (i)  is the ACROSS-unit piece: counties with many cells respond
                differently from counties with few
    (iii) - (ii) is the WITHIN-unit piece: a county's large cells respond
                differently from its small ones

CHANNEL 2, Jensen (eq `jensen_share`). ONE variable at the county level, the
county's log growth minus the lagged-share-weighted average of its cells' log
growth. This is the `C_it` of the derivation note
`2026-07-16_153250_yachi_jensen-share-reallocation-proof.md`:

    Delta^l S_it = xbar_it + C_it,   C_it = -sum_m w^-_im Delta^l z_imt

    w^-_im = S_im,t-h / S_i,t-h      z_imt = S_imt / S_it

Its coefficient is the curvature contribution. The protocol also names a second
route to the same object: log by WLS against midpoint by WLS with the WEIGHTS
HELD FIXED, so only the nonlinearity moves. Both are reported and they are
different estimands of the same channel, not a check of one against the other.

CHANNEL 3, entry and exit (eq `decompo_growth`). The entry, exit and intensive
components of county midpoint growth, asserted to sum to the total, each
regressed on the treatment. Reported alongside is the share of total activity
that survives into the log estimation sample, which is what a log specification
cannot see.

THE BRIDGE. Because midpoint growth under MPGR weights is invariant across rungs
(Proposition 1, asserted here), the county-log to cell-log gap splits exactly:

    cell log/MPGR  -  county log
      = [county midpoint - county log]        the county-level transform residual
      + [cell midpoint/MPGR - county midpoint] EXACTLY ZERO by Proposition 1
      + [cell log/MPGR - cell midpoint/MPGR]   the curvature channel, weights held

Run class: estimate_only on the cell construction 29 rebuilt.
PRE-1998 ONLY.
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

TOL_INVAR = t28.TOL_INVAR
TOL_IDENTITY = 1e-12
TOL_BRIDGE = 1e-9


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--share", default="fixed5",
                    choices=["fixed4", "fixed5", "fixed10"])
    ap.add_argument("--rung", default="naics4_alloc", choices=list(w22.RUNGS_ALL))
    ap.add_argument("--level", type=float, default=5.0)
    args = ap.parse_args()
    h = args.horizon

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(dc.LOG_DIR / f"32_disagg_channels_pre_h{h}_{args.share}_{args.rung}.log")
    import pyfixest as pf
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
    t0 = time.time()

    print("=" * 78)
    print(f"  B.1 Layer 2, the three channels.  h = {h}, pre-{w22.NAICS_BREAK}, "
          f"{args.rung}")
    print("=" * 78)
    print(f"  instrument z_income_{args.share}, IQR{args.level:g} by year, "
          f"ccg + year FE, clustered on county")

    frame = t29.build_frame("pre", h, args.share, args.level)
    shock, zi = frame["shock"].to_numpy(), frame["z"].to_numpy()
    ccg, yr = frame["ccg"].to_numpy(), frame["year"].to_numpy()
    n = len(frame)
    years = sorted(set(frame["year"]) | set(frame["year"] - h))
    emp, ccg_arr = w22.load_rung(args.rung, years)
    _, code_arr = c27.rung_codes(args.rung, years[0])
    keep = np.isin(ccg_arr, frame["ccg"].unique())
    if not keep.all():
        emp = {y: v[keep] for y, v in emp.items()}
        ccg_arr, code_arr = ccg_arr[keep], code_arr[keep]

    d = t28.build_long(args.rung, emp, ccg_arr, code_arr, frame, h)
    mpgr, w, log, pos = t28.cell_outcomes(d)
    cy, cur, lag = d["f_cy"], d["cur"], d["lag"]
    cur_c, lag_c = d["cur_c"], d["lag_c"]

    # A common county-year sample: every quantity below has to exist on all of it.
    cc = np.bincount(cy[pos], weights=cur[pos], minlength=n)
    lc = np.bincount(cy[pos], weights=lag[pos], minlength=n)
    n_c = np.bincount(cy[pos], minlength=n).astype(float)
    ok = (cc > 0) & (lc > 0) & (n_c > 0) & (cur_c > 0) & (lag_c > 0)
    m = pos & ok[cy]
    print(f"\n  {int((~ok).sum())} of {n:,} county-years dropped "
          f"(no continuer employment); {int(ok.sum()):,} estimated")

    S = lambda v: v[ok]
    one_c = np.ones(int(ok.sum()))
    FC = lambda y, ww=None: t28.fit2sls(
        S(y), S(shock), S(zi), one_c if ww is None else ww,
        S(ccg), S(yr), S(ccg), singleton=False)
    FK = lambda y, ww, msk: t28.fit2sls(
        y[msk], shock[cy][msk], zi[cy][msk], ww,
        ccg[cy][msk], yr[cy][msk], ccg[cy][msk], singleton=False)

    log_c = np.log(cur_c) - np.log(lag_c)
    mp_c = 2.0 * (cur_c - lag_c) / (cur_c + lag_c)
    r_log_c, r_mp_c = FC(log_c), FC(mp_c)

    # ---------------- Channel 1: implicit weights ------------------------------
    print(f"\n{'=' * 78}\n  CHANNEL 1, implicit weights (Lemma 1)\n{'=' * 78}")
    w_equal = np.ones(int(m.sum()))
    w_1n = 1.0 / n_c[cy[m]]
    # w^-_im on m(i): the survivor set is the market set the Jensen identity
    # is stated over, so the denominator is the survivor lagged total, not
    # the full county total. Normalising differently would put Channel 1 and
    # Channel 2 on different benchmarks and the accounting would not close.
    w_lag = lag[m] / lc[cy[m]]
    r1 = {k: FK(log, ww, m) for k, ww in
          (("equal", w_equal), ("1/n_i", w_1n), ("lagged share", w_lag))}
    print(f"  {'weight':22s}{'beta':>12s}{'se':>10s}")
    for k in ("equal", "1/n_i", "lagged share"):
        print(f"  {k:22s}{r1[k]['beta']:12.6f}{r1[k]['se']:10.6f}")
    across = r1["1/n_i"]["beta"] - r1["equal"]["beta"]
    within = r1["lagged share"]["beta"] - r1["1/n_i"]["beta"]
    print(f"\n  across-unit piece  (1/n_i) - (equal)        {across:+12.6f}")
    print(f"  within-unit piece  (lagged) - (1/n_i)       {within:+12.6f}")
    print(f"  total implicit-weight span                  {across + within:+12.6f}")

    # ---------------- Channel 2: Jensen ----------------------------------------
    print(f"\n{'=' * 78}\n  CHANNEL 2, Jensen (eq jensen_share)\n{'=' * 78}")
    z_t = cur[m] / cc[cy[m]]
    z_l = lag[m] / lc[cy[m]]
    wl_cont = lag[m] / lc[cy[m]]
    xbar = np.bincount(cy[m], weights=wl_cont * log[m], minlength=n)
    C = -np.bincount(cy[m], weights=wl_cont * (np.log(z_t) - np.log(z_l)),
                     minlength=n)
    dlS = np.where(ok, np.log(np.where(ok, cc, 1.0))
                   - np.log(np.where(ok, lc, 1.0)), 0.0)
    check(float(np.abs((dlS - (xbar + C))[ok]).max()) < TOL_IDENTITY,
          "eq jensen_share  Delta^l S_it = xbar_it + C_it",
          f"max |gap| = {float(np.abs((dlS - (xbar + C))[ok]).max()):.2e}")
    r_dlS, r_xbar, r_C = FC(dlS), FC(xbar), FC(C)
    print(f"  {'Delta^l S_it  county log growth (continuers)':46s}"
          f"{r_dlS['beta']:11.6f}{r_dlS['se']:10.6f}")
    print(f"  {'  xbar_it     lagged-share-weighted cell logs':46s}"
          f"{r_xbar['beta']:11.6f}{r_xbar['se']:10.6f}")
    print(f"  {'  C_it        SHARE REALLOCATION, the curvature':46s}"
          f"{r_C['beta']:11.6f}{r_C['se']:10.6f}")
    check(abs(r_xbar["beta"] + r_C["beta"] - r_dlS["beta"]) < TOL_BRIDGE,
          "the Jensen decomposition is additive in the coefficients",
          f"|xbar + C - total| = "
          f"{abs(r_xbar['beta'] + r_C['beta'] - r_dlS['beta']):.2e}")

    print("\n  the protocol's second route: log/WLS against midpoint/WLS, "
          "MPGR weights held fixed")
    r_log_k = FK(log, w[m], m)
    # ALL cells of the retained county-years, not all cells in the panel: the
    # county midpoint fit above runs on `ok`, and Proposition 1 is a statement
    # about the same county-years at two rungs.
    m_all = ok[cy]
    r_mp_k = FK(mpgr, w[m_all], m_all)
    check(abs(r_mp_k["beta"] - r_mp_c["beta"]) < TOL_INVAR,
          "Prop 1  midpoint/MPGR at the cell equals the county",
          f"|gap| = {abs(r_mp_k['beta'] - r_mp_c['beta']):.2e}")
    print(f"  {'  cell log,      MPGR weight':46s}"
          f"{r_log_k['beta']:11.6f}{r_log_k['se']:10.6f}")
    print(f"  {'  cell midpoint, MPGR weight':46s}"
          f"{r_mp_k['beta']:11.6f}{r_mp_k['se']:10.6f}")
    print(f"  {'  curvature contribution, weights held':46s}"
          f"{r_log_k['beta'] - r_mp_k['beta']:+11.6f}")

    # ---------------- Channel 3: entry and exit --------------------------------
    print(f"\n{'=' * 78}\n  CHANNEL 3, entry and exit (eq decompo_growth)"
          f"\n{'=' * 78}")
    parts = {"intensive": (lag > 0) & (cur > 0),
             "entry": (lag <= 0) & (cur > 0),
             "exit": (lag > 0) & (cur <= 0)}
    ys = {k: np.bincount(cy[msk], weights=(w * mpgr)[msk], minlength=n)
          for k, msk in parts.items()}
    check(float(np.abs((sum(ys.values()) - mp_c)[ok]).max()) < TOL_IDENTITY,
          "eq decompo_growth  the components sum to county midpoint growth",
          f"max |gap| = {float(np.abs((sum(ys.values()) - mp_c)[ok]).max()):.2e}")
    r3 = {k: FC(v) for k, v in ys.items()}
    print(f"  {'component':22s}{'beta':>12s}{'se':>10s}")
    for k in ("intensive", "entry", "exit"):
        print(f"  {k:22s}{r3[k]['beta']:12.6f}{r3[k]['se']:10.6f}")
    print(f"  {'TOTAL (county midpoint)':22s}{r_mp_c['beta']:12.6f}"
          f"{r_mp_c['se']:10.6f}")
    ext = r3["entry"]["beta"] + r3["exit"]["beta"]
    print(f"\n  entry + exit, which log cannot see       {ext:+12.6f}")
    surv = 100.0 * float(cur[m].sum()) / float(cur_c[ok].sum())
    print(f"  share of activity surviving into the log sample: {surv:.2f}%")
    print(f"  MPGR weight mass retained by log:                "
          f"{w[m].sum() / int(ok.sum()):.4f}")

    # ---------------- The bridge ------------------------------------------------
    print(f"\n{'=' * 78}\n  THE BRIDGE: county log -> cell log under MPGR weights"
          f"\n{'=' * 78}")
    # The cell LOG fit runs on continuers only, because log is undefined for a
    # cell empty in either period, while the cell MIDPOINT fit runs on every
    # cell. Comparing them directly changes the transform AND the sample at once
    # and buries the extensive margin inside a term labelled curvature. The
    # midpoint fit is therefore also run on the continuer cells, so that each
    # step of the bridge moves exactly one thing.
    wsum_cont = np.bincount(cy[m], weights=w[m], minlength=n)

    # THE EXTENSIVE MARGIN IS THE ENTRY AND EXIT TERMS OF eq decompo_growth, and
    # nothing else. Dropping the entry and exit CELLS from a weighted cell fit is
    # a different operation: it also demotes counties with more churn, because
    # their retained MPGR mass is smaller. That reweighting is not part of the
    # extensive margin and is not allowed to hide in its line here. It is carried
    # in the residual term below, together with the renormalisation and the
    # curvature, all of which move when the log fit drops those cells.
    r_intensive = FC(ys["intensive"])

    t_res = r_mp_c["beta"] - r_log_c["beta"]
    prop1 = r_mp_k["beta"] - r_mp_c["beta"]
    extmar = r_intensive["beta"] - r_mp_c["beta"]
    curv = r_log_k["beta"] - r_intensive["beta"]
    gap = r_log_k["beta"] - r_log_c["beta"]
    print(f"  county log growth, full county total        {r_log_c['beta']:12.6f}")
    print(f"    + county transform residual (mid - log)   {t_res:+12.6f}")
    print(f"    + disaggregation under midpoint/MPGR      {prop1:+12.6f}"
          f"   <- Proposition 1")
    print(f"    + removing entry and exit                 {extmar:+12.6f}"
          f"   <- extensive margin, eq decompo_growth")
    print(f"    + transform, renormalisation, reweighting {curv:+12.6f}"
          f"   <- everything the log fit changes at once")
    print(f"  {'-' * 54}")
    print(f"  cell log growth, MPGR weight                {r_log_k['beta']:12.6f}")
    check(abs(t_res + prop1 + extmar + curv - gap) < TOL_BRIDGE,
          "the bridge is exact",
          f"|sum of pieces - gap| = "
          f"{abs(t_res + prop1 + extmar + curv - gap):.2e}; gap = {gap:+.6f}")
    # These two are NOT the same operation and are not expected to agree.
    # Channel 3 removes the entry and exit CONTRIBUTION from a county-level
    # outcome at weight 1. The bridge removes the entry and exit CELLS from a
    # weighted cell-level fit, which also reweights counties, since a county
    # with more churn loses more MPGR mass. The retained mass is reported so the
    # size of that reweighting is visible.
    check(abs(extmar + ext) < TOL_BRIDGE,
          "the bridge's extensive step IS Channel 3's entry + exit",
          f"{extmar:+.6f} against {-ext:+.6f}")
    print(f"\n  the residual term is a lump and is not curvature. It carries the")
    print(f"  renormalisation, the curvature, and the reweighting that drops")
    print(f"  churny counties (retained MPGR mass: mean "
          f"{wsum_cont[ok].mean():.4f}, min {wsum_cont[ok].min():.4f}).")

    # ---------------- The paper's accounting ------------------------------------
    # 3_disaggregation.tex line 55: the channels add to the total gap only when
    # they share the w^- benchmark. Assembled here in the paper's sign convention,
    # beta^b - beta^d_OLS, with entry and exit carried as the step from the full
    # county aggregate to the survivor aggregate.
    print(f"\n{'=' * 78}\n  THE PAPER'S ACCOUNTING: beta^b - beta^d_OLS"
          f"\n{'=' * 78}")
    check(abs(r1["lagged share"]["beta"] - r_xbar["beta"]) < TOL_BRIDGE,
          "the cell fit at w^- collapses to the xbar county fit",
          f"{r1['lagged share']['beta']:.10f} against {r_xbar['beta']:.10f}")
    b_full, b_surv = r_log_c["beta"], r_dlS["beta"]
    d_wm, d_1n, d_ols = (r_xbar["beta"], r1["1/n_i"]["beta"], r1["equal"]["beta"])
    ch = [("entry and exit", b_full - b_surv),
          ("Jensen, C_it (eq jensen_wedge)", b_surv - d_wm),
          ("within-unit bridge (Lemma 1)", d_wm - d_1n),
          ("across-unit cell count (Lemma 1)", d_1n - d_ols)]
    print(f"  {'level':44s}{'beta':>12s}")
    for lab, v in (("beta^b   county log, all industries", b_full),
                   ("beta^b   county log, survivors only", b_surv),
                   ("beta^d   cells at w^- (lagged share)", d_wm),
                   ("beta^d   cells at 1/n_i", d_1n),
                   ("beta^d   cells at equal weight (OLS)", d_ols)):
        print(f"  {lab:44s}{v:12.6f}")
    print(f"\n  {'channel':44s}{'beta':>12s}{'% of gap':>11s}")
    gap = b_full - d_ols
    for lab, v in ch:
        print(f"  {lab:44s}{v:12.6f}{100 * v / gap:10.1f}%")
    print(f"  {'-' * 67}")
    print(f"  {'TOTAL  beta^b - beta^d_OLS':44s}{gap:12.6f}{100.0:10.1f}%")
    check(abs(sum(v for _, v in ch) - gap) < TOL_BRIDGE,
          "the paper's four channels exhaust the gap with no residual",
          f"|sum - gap| = {abs(sum(v for _, v in ch) - gap):.2e}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = ([dict(channel="1_implicit_weight", term=k, beta=r1[k]["beta"],
                  se=r1[k]["se"]) for k in r1]
            + [dict(channel="1_implicit_weight", term="across_unit", beta=across,
                    se=np.nan),
               dict(channel="1_implicit_weight", term="within_unit", beta=within,
                    se=np.nan),
               dict(channel="2_jensen", term="county_log_continuers",
                    beta=r_dlS["beta"], se=r_dlS["se"]),
               dict(channel="2_jensen", term="xbar", beta=r_xbar["beta"],
                    se=r_xbar["se"]),
               dict(channel="2_jensen", term="C_share_reallocation",
                    beta=r_C["beta"], se=r_C["se"]),
               dict(channel="2_jensen", term="C_via_identity",
                    beta=r_C["beta"], se=r_C["se"])]
            + [dict(channel="3_entry_exit", term=k, beta=r3[k]["beta"],
                    se=r3[k]["se"]) for k in r3]
            + [dict(channel="bridge", term="county_log_full",
                    beta=r_log_c["beta"], se=r_log_c["se"]),
               dict(channel="bridge", term="county_transform_residual",
                    beta=t_res, se=np.nan),
               dict(channel="bridge", term="proposition_1", beta=prop1, se=np.nan),
               dict(channel="bridge", term="extensive_margin_decompo_growth", beta=extmar,
                    se=np.nan),
               dict(channel="bridge", term="residual_transform_renorm_reweight", beta=curv,
                    se=np.nan),
               dict(channel="bridge", term="cell_log_mpgr_weight",
                    beta=r_log_k["beta"], se=r_log_k["se"])])
    out = RESULT_DIR / f"coeff_disagg_channels_pre_h{h}_{args.share}_{args.rung}.csv"
    pd.DataFrame(rows).assign(period="pre", horizon=h, share=args.share,
                              rung=args.rung, fe="ccg+year").to_csv(
        out, index=False, float_format="%.15g")
    print(f"\n  wrote {out.relative_to(REPO_ROOT)}")

    n_fail = len(w22.failures)
    print(f"\n{'=' * 78}\n  {n_fail} checks failed, {time.time() - t0:.1f}s"
          f"\n{'=' * 78}")
    tee.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
