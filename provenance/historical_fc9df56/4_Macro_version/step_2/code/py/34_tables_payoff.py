#!/usr/bin/env python3
"""Tables 1-4: the disaggregation ladder under four specifications, h=4.

Every table has the same five columns, so the reader compares ACROSS a row:

    (1) county                              ccg + year
    (2) county x NAICS2                     ccg + year
    (3) county x NAICS2 + industry x year   ccg + industry x year
    (4) county x NAICS4                     ccg + year
    (5) county x NAICS4 + industry x year   ccg + industry x year

    Table 1  midpoint growth, MPGR weights   ADMISSIBLE (A.1.5). Flat across
                                             (1),(2),(4) by Proposition 1, and
                                             the move into (3),(5) is what the
                                             industry-by-year shock costs.
    Table 2  log growth, w^- weights         INADMISSIBLE as an answer (A.1.5).
                                             w^- is the naive lagged share on the
                                             FULL county base, which is what an
                                             applied person computes from CBP,
                                             where lagged zeros are invisible.
    Table 3  log growth, equal weights       INADMISSIBLE. beta^d_OLS, the
                                             disaggregated regression as actually
                                             run in the literature.
    Table 4  standard growth, w^- weights    INADMISSIBLE. Its cell set is cells
                                             with a positive BASE, so exiters sit
                                             inside it at -1 and the whole
                                             cross-level gap is entry.

THE SAMPLES ARE NOT HARMONISED, ON PURPOSE. Midpoint is defined on every
county-year; log is not, because a county whose every industry turned over has
no survivor aggregate to take a log of. Deleting those county-years from the
midpoint columns to match would delete the extensive margin at its most extreme,
which is A.3.1's error committed deliberately. They are counted instead, on the
face of the table, as `county-years log cannot use`.

ccg x industry APPEARS NOWHERE. See mpgr_disagg.py's header for why.

Run class: estimate_only on the cell construction. PRE-1998.
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

SPECS = ("midpoint_mpgr", "log_naive", "log_equal", "standard_naive")
LABEL = {"midpoint_mpgr":  "Table 1: midpoint growth, MPGR weights",
         "midpoint_mpgr_pop": "Table 1P: midpoint growth, MPGR weights, "
                              "population-weighted counties",
         "log_naive":      "Table 2: log growth, w^- (naive lagged share)",
         "log_equal":      "Table 3: log growth, equal weights",
         "standard_naive": "Table 4: standard growth, w^- (lagged-employment shares)"}
COLS = ("county", "n2", "n2_iy", "n4", "n4_iy")
COLHEAD = {"county": "county", "n2": "x NAICS2", "n2_iy": "x NAICS2 +i.y",
           "n4": "x NAICS4", "n4_iy": "x NAICS4 +i.y"}


def county_outcome(p, spec):
    """The county row. For log this is the FULL county total, entrants and
    exiters included, which is the number a paper reports."""
    if spec == "standard_naive":
        # Standard growth needs only a positive BASE, so its county sample is
        # lag_c > 0 rather than the log sample. A county whose every industry
        # turned over has a standard growth rate and no log one.
        ok = p.lag_c > 0
        return np.where(ok, (p.cur_c - p.lag_c) / np.where(ok, p.lag_c, 1.0),
                        0.0), ok
    if spec == "midpoint_mpgr":
        den = p.cur_c + p.lag_c
        return p.mpgr_c, den != 0.0
    g = lambda v: np.where(p.ok, v, 1.0)
    return np.log(g(p.cur_c)) - np.log(g(p.lag_c)), p.ok


def cell_terms(p, spec):
    """(outcome, weight, mask) at the cell level for one specification."""
    if spec == "midpoint_mpgr":
        allc = p.cy >= 0
        return p.mpgr, p.w_mpgr[allc], allc
    if spec == "standard_naive":
        # M^- is cells with a positive base, so EXITERS ARE INSIDE IT, sitting
        # at -1. That is why standard growth needs no exit term and its whole
        # cross-level gap is entry (eq standard_entry_decomp). The weight is the
        # lagged share on the county's full base, which is A.3.2's answer to an
        # unbounded rate: a near-zero base gets a near-zero weight.
        ok = p.lag_c > 0
        mS = (p.lag > 0) & ok[p.cy]
        g = np.where(mS, (p.cur - p.lag) / np.where(mS, p.lag, 1.0), 0.0)
        return g, p.lag[mS] / p.lag_c[p.cy[mS]], mS
    if spec == "log_naive":
        return p.log, p.w_naive, p.m
    return p.log, p.w_equal, p.m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--share", default="fixed5",
                    choices=["fixed4", "fixed5", "fixed10"])
    ap.add_argument("--level", type=float, default=5.0)
    ap.add_argument("--frame", default="candidate",
                    choices=["candidate", "baseline"],
                    help="candidate: the IQR5 2SLS design these tables were "
                         "first built on. baseline: the locked baseline "
                         "(harmonized top-800 composition instrument, span 5, "
                         "positive rule, population denominator, reduced form, "
                         "division-by-year effects, full era)")
    ap.add_argument("--source", default="efsy", choices=list(md.SOURCES),
                    help="which employment series the OUTCOME is built from. "
                         "efsy: the desuppressed CBP cells. nets_actual / "
                         "nets_all: WS2 step 3, the NETS cells at the "
                         "actual-employment universe (the headline) or the full "
                         "panel (its robustness). Changing this changes the "
                         "sample too -- NETS starts in 1990 and the ccg support "
                         "is intersected -- and the exhibit note has to "
                         "enumerate that bundle.")
    ap.add_argument("--year-floor", type=int, default=None,
                    help="baseline frame only: drop target years below this. "
                         "Its one use is an EFSY column on the window NETS can "
                         "reach, so the source comparison is not confounded "
                         "with the era. Tagged into the filename.")
    args = ap.parse_args()
    h = args.horizon
    base = args.frame == "baseline"
    src = args.source
    stag = "" if src == "efsy" else f"_{src}"
    if args.year_floor is not None:
        if not base:
            raise SystemExit("--year-floor is a baseline-frame option")
        stag += f"_from{args.year_floor}"

    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tee = Tee(dc.LOG_DIR / (f"34_tables_payoff_baseline{stag}_h{h}.log" if base
                            else f"34_tables_payoff_pre{stag}_h{h}_{args.share}.log"))
    import pyfixest as pf
    md.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
    t0 = time.time()

    print("=" * 84)
    print(f"  TABLES 1-4, the disaggregation ladder.  h = {h}, "
          f"pre-{md.w22.NAICS_BREAK}")
    print("=" * 84)
    if base:
        print("  the locked baseline: harmonized top-800 composition instrument,")
        print("  span 5, positive rule, population denominator, reduced form,")
        print("  division-by-year effects, full era, NO winsor anywhere.")
    else:
        print(f"  instrument z_income_{args.share}, IQR{args.level:g} by year on the")
        print("  shock and the instrument only. No growth rate is winsorized (A.3.2)")
        print("  and no cell is dropped (A.3.1). ccg x year appears nowhere (A.1.4).")

    if src != "efsy":
        print(f"  OUTCOME SOURCE: {src}. The bundled change is source + "
              f"universe + ccg support + A.3.4 exclusions; every")
        print("  coefficient below is on the NETS employment series and is not "
              "comparable cell-for-cell")
        print("  to an EFSY table. NETS begins in 1990, so the first target "
              f"year is {1990 + h}.")

    panels = {r: md.Panel(h, r, args.share, args.level, frame_src=args.frame,
                          source=src, year_floor=args.year_floor)
              for r in md.RUNGS}

    rows = []
    specs = SPECS + ("midpoint_mpgr_pop",) if base else SPECS
    for spec in specs:
        wspec = spec
        popw = spec == "midpoint_mpgr_pop"
        if popw:
            wspec = "midpoint_mpgr"
        p4 = panels["naics4_alloc"]
        yc, mc = county_outcome(p4, wspec)
        # The MPGR cell regressions are an exact weighted expansion of the
        # county-year regression. Their CRV1 correction therefore uses the
        # county-year count rather than the number of expanded cell rows.
        correction_n = int(mc.sum()) if wspec == "midpoint_mpgr" else None
        cells = {}
        for r, p in panels.items():
            y, w, mask = cell_terms(p, wspec)
            if popw:
                w = w * p.frame["P0"].to_numpy(np.float64)[p.cy][mask]
            fcell = p.fit_cell_base if base else p.fit_cell
            cells[r] = (fcell(y, w, mask, indyear=False,
                              correction_n=correction_n),
                        fcell(y, w, mask, indyear=True))
        wc = p4.frame["P0"].to_numpy(np.float64)[mc] if popw else None
        cty = p4.fit_county_base(yc, mc, w=wc, correction_n=correction_n) \
            if base else p4.fit_county(yc, mc, correction_n=correction_n)
        fits = dict(county=cty, n2=cells["naics2"][0], n2_iy=cells["naics2"][1],
                    n4=cells["naics4_alloc"][0], n4_iy=cells["naics4_alloc"][1])
        # The clustered SE and the fitted N travel with every coefficient. They
        # are not used by the invariance check, which is about point estimates,
        # but a table that reaches a draft needs them and recovering them later
        # would mean re-estimating.
        row = dict(spec=spec, n_county=int(mc.sum()))
        sc = 1e5 if base else 1.0   # baseline z is per head: pp per $1,000/head
        for k, f in fits.items():
            row[k] = f["beta"] * sc
            row[f"se_{k}"] = f["se"] * sc
            row[f"n_{k}"] = int(f["n"])
        rows.append(row)

    # Proposition 1: the admissible answer is flat across the no-extra-FE columns.
    for t1 in (r for r in rows if r["spec"].startswith("midpoint_mpgr")):
        tag = "population weighted" if t1["spec"].endswith("_pop") else "equal weighted"
        for c in ("n2", "n4"):
            check(abs(t1[c] - t1["county"]) < md.t28.TOL_INVAR,
                  f"Prop 1  midpoint/MPGR is invariant, county to {COLHEAD[c]} "
                  f"({tag})",
                  f"|gap| = {abs(t1[c] - t1['county']):.2e}; "
                  f"{t1[c]:.10f} against {t1['county']:.10f}")
            check(abs(t1[f"se_{c}"] - t1["se_county"])
                  < md.t28.TOL_INVAR_SE * max(t1["se_county"], 1e-12),
                  f"Prop 1  midpoint/MPGR clustered SE is invariant, county to "
                  f"{COLHEAD[c]} ({tag})",
                  f"|gap| = {abs(t1[f'se_{c}'] - t1['se_county']):.2e}; "
                  f"{t1[f'se_{c}']:.10f} against {t1['se_county']:.10f}")

    W = 46
    if base:
        print("\n  units: pp per $1,000 of contracts per head (x 1e5)")
    for r in rows:
        print(f"\n{'=' * 84}\n  {LABEL[r['spec']]}\n{'=' * 84}")
        print("  " + " " * W + "".join(f"{COLHEAD[c]:>15s}" for c in COLS))
        print(f"  {'contract shock':{W}s}"
              + "".join(f"{r[c]:15.6f}" for c in COLS))
        print(f"  {'  gap from county':{W}s}"
              + "".join(f"{'':15s}" if c == "county"
                        else f"{r[c] - r['county']:+15.6f}" for c in COLS))

    d2, d4 = panels["naics2"].diagnostics(), panels["naics4_alloc"].diagnostics()
    print(f"\n{'=' * 84}\n  DIAGNOSTICS\n{'=' * 84}")
    for lab, k, fmt in (("county-years, midpoint columns", None, None),
                        ("county-years, log columns", "n_county_years", "{:,}"),
                        ("county-years log cannot use", "n_log_blind", "{:,}"),
                        ("counties", "n_counties", "{:,}"),
                        ("cell rows, midpoint", "n_cells", "{:,}"),
                        ("cell rows, log", "n_cells_log", "{:,}"),
                        ("cells zero in both periods (%)", "pct_zero_both", "{:.1f}"),
                        ("county employment in log sample (%)", "share_emp_log", "{:.2f}"),
                        ("w^- mass retained by log", "mass_naive", "{:.4f}")):
        if k is None:
            print(f"  {lab:{W}s}{rows[0]['n_county']:>15,}{rows[0]['n_county']:>15,}")
            continue
        print(f"  {lab:{W}s}{fmt.format(d2[k]):>15s}{fmt.format(d4[k]):>15s}")
    print(f"  {'':{W}s}{'NAICS2':>15s}{'NAICS4':>15s}")

    # The estimation window travels with the coefficients. The renderers print
    # it in the note, and a note that says which county-years were used cannot
    # be reconstructed after the fact -- it has to be written down here. This is
    # not decoration under --source: the NETS floor year is a sample restriction
    # and the note is where a reader finds out about it.
    yrs = panels["naics4_alloc"].frame["year"]
    window = dict(year_first=int(yrs.min()), year_last=int(yrs.max()),
                  n_windows=int(yrs.nunique()))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / (f"coeff_tables_payoff_baseline{stag}_h{h}.csv" if base
                        else f"coeff_tables_payoff_pre{stag}_h{h}_{args.share}.csv")
    pd.DataFrame(rows).assign(period="baseline" if base else "pre", horizon=h,
                              share=args.share, source=src, **window).to_csv(
        out, index=False, float_format="%.15g")
    print(f"\n  wrote {out.relative_to(REPO_ROOT)}")

    n_fail = len(md.w22.failures)
    print(f"\n{'=' * 84}\n  {n_fail} checks failed, {time.time() - t0:.1f}s"
          f"\n{'=' * 84}")
    tee.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
