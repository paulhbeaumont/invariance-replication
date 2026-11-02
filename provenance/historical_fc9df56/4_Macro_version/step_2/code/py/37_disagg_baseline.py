"""The disaggregation test on the locked baseline, both weight schemes.

The county-level regression is the locked aggregation baseline's county row:
midpoint employment growth on the composition instrument (harmonized top-800
panel, 5-year share window, positive rule, population denominator),
division-by-year effects, reduced form, county clusters, run equal-weighted
and population-weighted. This producer splits the OUTCOME into county-by-
industry cells (NAICS2, and NAICS4 as the finer rung per A.1.7) and asks
whether the coefficient survives, which is Proposition 1's invariance claim:

    col 1  county baseline                        division-year FE
    col 2  county x NAICS2, midpoint/WLS          division-year FE
    col 3  col 2 + industry-by-year FE
    col 4  county x NAICS4, midpoint/WLS          division-year FE
    col 5  col 4 + industry-by-year FE

Cell weights are the county's regression weight times the midpoint share
w_im = (E_imt + E_im,t-h) / (E_ct + E_c,t-h), which sums to the county
weight, so columns 2 and 4 must equal column 1 EXACTLY, for the
equal-weighted scheme and the population-weighted scheme alike: the
invariance is agnostic to the across-county weighting, and the run asserts
it to 1e-9 rather than assuming it. The jump when industry-by-year effects
enter (2 to 3, 4 to 5) then measures the confounding industry shocks, which
is why one disaggregates. Log/OLS and log/WLS companions appear as the
Part-B benchmark, never as the answer.

Protocol mapping (Part 0 and A.1): baseline unit i = county, cell m =
industry; the treatment and instrument stay at (i, t) and are never split by
industry (A.1.1); S_i = sum_m S_im is asserted against EFSY's independent
county series (A.1.2); division-by-year is coarser than i and admissible,
industry-by-year is m x t and admissible (A.1.3); no county or county-year
FE anywhere (A.1.4); nothing is winsorized and no cell is dropped beyond the
estimable-support pruning shared with every step_2 exhibit (A.3).

The county anchor: column 1 on the full baseline years must reproduce the 57
grid's county cells at the locked specification (span 5, positive, no floor,
population, division-year), read from aggregation_level_grid.csv at run time.

Output: Results/ns14_diagnostics/Table_disagg_baseline.csv and .tex.
"""

from datetime import datetime
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
STEP2_CODE = HERE.parent
STEP1_CODE = HERE.parents[3] / "step_1" / "code" / "py"
RESULTS_DIR = HERE.parents[3] / "Results" / "ns14_diagnostics"
LOG_DIR = HERE.parents[4] / "batch_logs" / "macro_version"
LOG_FILE = LOG_DIR / f"disagg_baseline_{datetime.now():%Y%m%d_%H%M%S}.log"

sys.path.insert(0, str(STEP1_CODE))
import dcads_common as dc  # noqa: E402

sys.argv = [sys.argv[0]]

BASE_SPEC = dict(span=5, rule="positive", floor=0.0, denom="population")
HORIZONS = (2, 3, 4)
RUNGS = ("naics2", "naics4_alloc")
SCALE = 1e5
TOL_INV = 1e-9
GRID_CSV = dc.LOG_DIR / "aggregation_level_grid.csv"


def log(msg=""):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(str(msg) + "\n")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("== disaggregation on the locked baseline (producer 37) ==")

    import importlib.util

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    md = _load("md", STEP2_CODE / "mpgr_disagg.py")
    t28, c27, w22 = md.t28, md.c27, md.w22
    p57 = _load("p57", STEP1_CODE / "57_aggregation_invariance.py")
    import pyfixest as pf
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)

    grid = pd.read_csv(GRID_CSV)

    def rf(y, z, w, f1, f2, cl):
        w = w / w.mean()
        r = t28.fit2sls(np.asarray(y, np.float64), np.asarray(z, np.float64),
                        np.asarray(z, np.float64), w, f1, f2, cl,
                        singleton=False)
        return r["beta"] * SCALE, r["se"] * SCALE, r["beta"] / r["se"], r["F"]

    rows = []
    for h in HORIZONS:
        d, _ = p57.build_county("composition", h=h, **BASE_SPEC)
        d = d[np.isfinite(d["y_mpgr"]) & np.isfinite(d["z"])].reset_index(drop=True)

        # anchor on the FULL baseline years against the 57 grid
        dyk = (d["division"].astype(str) + "_" + d["year"].astype(str))
        f1 = pd.factorize(dyk)[0].astype(np.int64)
        ze = np.zeros(len(d), np.int64)
        cl = pd.factorize(d["ccg"])[0].astype(np.int64)
        for wlab, wvec in (("equal", np.ones(len(d))),
                           ("fixed pop", d["P0"].to_numpy(np.float64))):
            b, se, t, F = rf(d["y_mpgr"], d["z"], wvec, f1, ze, cl)
            ref = grid[(grid.instrument == "composition") & (grid.outcome == "y_mpgr")
                       & (grid.level == "county") & (grid.horizon == h)
                       & (grid.span == 5) & (grid.rule == "positive")
                       & (grid.floor == "none") & (grid.denom == "population")
                       & (grid.fe == "divisionyear") & (grid.weight == wlab)]
            rt = float(ref["rf_t"].iloc[0])
            ok = abs(t - rt) < 0.02
            log(f"h={h} {wlab:>9} county anchor vs 57 grid: t {t:.3f} vs {rt:.3f}"
                f"  {'PASS' if ok else 'FAIL'}")
            assert ok, "county column does not reproduce the locked baseline"

        # frame restricted to years with EFSY cells at t and t-h (1975-2018)
        yok = (d["year"] >= 1975 + h) & (d["year"] <= 2018)
        frame = d[yok].reset_index(drop=True)
        n_drop = len(d) - len(frame)
        log(f"h={h}: EFSY intersection drops {n_drop:,} county-years "
            f"({n_drop / len(d):.1%}); frame {len(frame):,}")

        fdyk = (frame["division"].astype(str) + "_" + frame["year"].astype(str))
        ff1 = pd.factorize(fdyk)[0].astype(np.int64)
        fze = np.zeros(len(frame), np.int64)
        fcl = pd.factorize(frame["ccg"])[0].astype(np.int64)
        WCTY = {"equal": np.ones(len(frame)),
                "fixed pop": frame["P0"].to_numpy(np.float64)}

        col1 = {}
        for wlab, wvec in WCTY.items():
            b, se, t, F = rf(frame["y_mpgr"], frame["z"], wvec, ff1, fze, fcl)
            col1[wlab] = b
            rows.append(dict(horizon=h, rung="county", col=1, fe="divisionyear",
                             weight=wlab, growth="midpoint", beta=b, se=se,
                             t=t, F=F, n=len(frame)))
            log(f"h={h} {wlab:>9} col1 county: beta {b:7.3f} t {t:5.2f}")

        for rung in RUNGS:
            years = sorted(set(frame["year"]) | set(frame["year"] - h))
            emp, ccg_arr = w22.load_rung(rung, years)
            _, code_arr = c27.rung_codes(rung, years[0])
            keep = np.isin(ccg_arr, frame["ccg"].unique())
            if not keep.all():
                emp = {y: v[keep] for y, v in emp.items()}
                ccg_arr, code_arr = ccg_arr[keep], code_arr[keep]
            cells = md.prune(t28.build_long(rung, emp, ccg_arr, code_arr,
                                            frame, h), len(frame), rung, h)
            mpgr, w_mpgr, lg, pos = t28.cell_outcomes(cells)
            fcy = cells["f_cy"]

            # A.1.2: weighted cell midpoint growth reconstructs the county
            agg = np.bincount(fcy, weights=w_mpgr * mpgr, minlength=len(frame))
            den_c = cells["cur_c"] + cells["lag_c"]
            nz = den_c != 0
            county_mpgr = np.where(nz, 2.0 * (cells["cur_c"] - cells["lag_c"])
                                   / np.where(nz, den_c, 1.0), np.nan)
            worst = float(np.nanmax(np.abs(agg - county_mpgr)[nz]))
            log(f"h={h} {rung}: midpoint reconstruction max gap {worst:.1e}")
            assert worst < 1e-9

            # county-years whose EFSY total is degenerate cannot carry a cell
            # regression row and are absent from BOTH levels' effective sample
            # only via zero weight, never by dropping rows (A.3.1)
            zc = frame["z"].to_numpy(np.float64)[fcy]
            f1c = ff1[fcy]
            fzec = np.zeros(len(fcy), np.int64)
            clc = fcl[fcy]
            iyk = cells["f_ky"].astype(np.int64)  # the m x t factor, prebuilt
            for wlab, wvec in WCTY.items():
                wcell = w_mpgr * wvec[fcy]
                b, se, t, F = rf(mpgr, zc, wcell, f1c, fzec, clc)
                gap = abs(b - col1[wlab])
                rows.append(dict(horizon=h, rung=rung, col=2 if rung == "naics2"
                                 else 4, fe="divisionyear", weight=wlab,
                                 growth="midpoint", beta=b, se=se, t=t, F=F,
                                 n=int(len(mpgr)), invariance_gap=gap))
                log(f"h={h} {wlab:>9} {rung:>12} midpoint: beta {b:7.3f} "
                    f"(county {col1[wlab]:7.3f})  |gap| {gap:.2e}  "
                    f"{'INVARIANT' if gap < TOL_INV * SCALE else 'FAIL'}")
                assert gap < TOL_INV * SCALE, "midpoint invariance failed"

                b3, se3, t3, F3 = rf(mpgr, zc, wcell, f1c, iyk, clc)
                rows.append(dict(horizon=h, rung=rung, col=3 if rung == "naics2"
                                 else 5, fe="divisionyear+industryyear",
                                 weight=wlab, growth="midpoint", beta=b3,
                                 se=se3, t=t3, F=F3, n=int(len(mpgr))))
                log(f"h={h} {wlab:>9} {rung:>12} + industry-year: "
                    f"beta {b3:7.3f} t {t3:5.2f}")

                # Part-B benchmarks: log growth, survivors only, labeled so
                for glab, gw in (("log_OLS", np.ones(int(pos.sum()))),
                                 ("log_WLS", (w_mpgr * wvec[fcy])[pos])):
                    bl, sel, tl, Fl = rf(lg[pos], zc[pos], gw, f1c[pos],
                                         fzec[pos], clc[pos])
                    rows.append(dict(horizon=h, rung=rung, col=0,
                                     fe="divisionyear", weight=wlab,
                                     growth=glab, beta=bl, se=sel, t=tl, F=Fl,
                                     n=int(pos.sum()), benchmark=True))
            log(f"  [{time.time()-t0:.0f}s]")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "Table_disagg_baseline.csv", index=False)
    with open(RESULTS_DIR / "Table_disagg_baseline.tex", "w") as fh:
        fh.write("\\begin{tabular}{llllrrr}\n\\toprule\n")
        fh.write("h & rung & FE & weight & $\\beta$ & $t$ & N \\\\\n\\midrule\n")
        for _, r in out[out["growth"] == "midpoint"].iterrows():
            fh.write(f"{r['horizon']} & {r['rung']} & {r['fe']} & {r['weight']} "
                     f"& {r['beta']:.3f} & {r['t']:.2f} & {r['n']:,} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    log(f"\nwrote {RESULTS_DIR / 'Table_disagg_baseline.csv'} and .tex "
        f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
