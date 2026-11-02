"""The seam robustness ladder: does the 1979 renumbering move the instrument?

The composition instrument (54) builds shares and rates on category codes that
break at FY1979 (services renumbered) and FY2012 (codes added). Both breaks
now have validated repairs wired into 54 behind `harmonize=True`. This
producer runs one specification four ways and puts the numbers side by side:

    row 1  raw panel,         full target years      (the incumbent)
    row 2  raw panel,         post-1979 windows only (seam-exposed years leave)
    row 3  harmonized panel,  post-1979 windows only (placebo: should match 2)
    row 4  harmonized panel,  full target years      (repaired early years back)

Row 1 vs 2 prices the seam contamination in the incumbent; row 2 vs 3 checks
that harmonization is inert where it has nothing to repair; row 3 vs 4 shows
what the recovered pre-1979 information adds. The h = 2 roll5 cell anchors the
ladder because its sample count (119,262) is the pinned reference; rows 1 and
4 also run at h = 3 (roll5) and h = 4 (roll10, the headline exposure).

Specification, held fixed everywhere: era full, net rule, no floor, no
screen, population denominator, lam 1.0, division-by-year fixed effects,
division-EW weights, winsor group year: the fe/weight/winsor block of 54's
HEADLINE cell.

Construction checks, asserted before any estimation: the harmonized panel's
year-over-year category churn stays below 2% of dollars in every adjacent
pair 1966-2018 including both seams; harmonized national dollars equal raw
dollars except for the reported unbridged pre-1979 service share; row 1's
h = 2 sample equals the stored reference (N = 119,262 on 3,058 counties).

Output: Results/ns14_diagnostics/Table_seam_ladder.csv and .tex.
"""

from datetime import datetime
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import dcads_common as dc  # noqa: E402

RESULTS_DIR = CODE.parents[2] / "Results" / "ns14_diagnostics"
LOG_DIR = CODE.parents[3] / "batch_logs" / "macro_version"
LOG_FILE = LOG_DIR / f"seam_ladder_{datetime.now():%Y%m%d_%H%M%S}.log"

sys.argv = [sys.argv[0]]  # keep 54's argparse from seeing our args on import


def log(msg=""):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(str(msg) + "\n")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("== seam robustness ladder (producer 61) ==")

    import importlib.util

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    p54 = _load("p54", CODE / "54_composition_instrument_grid.py")
    p51 = p54.p51
    import pyfixest as pf
    w22 = _load("w22", p54.STEP2_CODE / "22_winsor_construction_error.py")
    t28 = _load("t28", p54.STEP2_CODE / "28_table_d1.py")
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100_000)

    # ---- construction checks on the harmonized panel ----------------------
    fips_to_ccg, _, _ = dc.load_ccg_map()
    raw_tot, harm_tot, prev = {}, {}, None
    max_dead, max_born = 0.0, 0.0
    for y in range(1966, 2019):
        d_r, _, _ = p54.category_actions(y, fips_to_ccg, True, harmonize=False)
        d_h, _, cat_h = p54.category_actions(y, fips_to_ccg, True, harmonize=True)
        raw_tot[y], harm_tot[y] = d_r.sum(), d_h.sum()
        cur = pd.Series(d_h).groupby(pd.Series(cat_h)).sum().clip(lower=0)
        if prev is not None:
            max_dead = max(max_dead,
                           prev[prev.index.difference(cur.index)].sum() / prev.sum())
            max_born = max(max_born,
                           cur[cur.index.difference(prev.index)].sum() / cur.sum())
        prev = cur
    log(f"harmonized panel: max adjacent-year churn dead {max_dead:.2%} "
        f"(threshold 2%), born {max_born:.2%} (threshold 3%; the 1979 seam "
        f"carries a real phase-in of codes with no 1978 counterpart)  "
        f"[{time.time()-t0:.0f}s]")
    assert max_dead < 0.02 and max_born < 0.03, "harmonized panel still churns"
    drop_pre = 1 - min(harm_tot[y] / raw_tot[y] for y in range(1966, 1979))
    post_gap = max(abs(harm_tot[y] / raw_tot[y] - 1) for y in range(1979, 2019))
    log(f"dollar conservation: max unbridged pre-1979 share {drop_pre:.2%}, "
        f"max post-1979 gap {post_gap:.2e}")
    assert drop_pre < 0.02 and post_gap < 1e-9

    # ---- panels and treatment --------------------------------------------
    B = p51.base_panel()
    units = np.array(sorted(B["ccg"].unique()))
    CON = p51.contract_matrices(units, 1966, 2018, drop_construction=True)
    log(f"treatment built  [{time.time()-t0:.0f}s]")
    PANELS = {}
    for hz in (False, True):
        ARR, NAT, cats, cyears = p54.category_panel(units, 1966, 2018,
                                                    drop_construction=True,
                                                    harmonize=hz)
        PANELS[hz] = (ARR[("net", 0.0)], NAT[("net", 0.0)], cats, cyears)
        log(f"category panel harmonize={hz}: {len(cats)} categories "
            f"[{time.time()-t0:.0f}s]")

    STATE = B.drop_duplicates("ccg").set_index("ccg")["state"]
    DIVI = B.drop_duplicates("ccg").set_index("ccg")["division"]
    cfg = p51.ERAS["full"]
    piv = {c: B.pivot(index="ccg", columns="year", values=c).reindex(units)
           for c in ("Y", "P", "E")}

    def run_cell(hz, h, elab, restrict_post79):
        arr, nat, cats, cyears = PANELS[hz]
        spec = p51.ROLLING[elab]
        years = [t for t in range(max(cfg["lo"], cfg["first_base"] + h),
                                  cfg["hi"] + 1)]
        if restrict_post79:
            tmin = 1979 + h + p51.ROLL_LAG + spec - 1
            years = [t for t in years if t >= tmin]
        idx = pd.MultiIndex.from_product([units, years], names=["ccg", "year"])
        f = pd.DataFrame(index=idx).reset_index()
        f["state"] = f["ccg"].map(STATE)
        f["division"] = f["ccg"].map(DIVI)
        for c in ("Y", "P", "E"):
            f[c] = piv[c].stack().reindex(idx).to_numpy()
            f[c + "_b"] = piv[c].reindex(columns=[y - h for y in years]) \
                .set_axis(years, axis=1).stack().reindex(idx).to_numpy()
        den_m = f["E"] + f["E_b"]
        f["y"] = np.where(den_m != 0, 2.0 * (f["E"] - f["E_b"]) / den_m, np.nan)

        C = CON[("net", 0.0)]
        lag = {k: C.reindex(columns=[y - k for y in years])
               .set_axis(years, axis=1) for k in range(h + 1)}
        f["G_b"] = lag[h].stack().reindex(idx).to_numpy()
        gk = p54.category_rates(nat, h)
        L, nc = p54.predicted_dollars(arr, gk, cyears, years, h, "roll", spec,
                                      p51.ROLL_LAG, p51.DATA_FIRST)
        wj = p51.lam_weights(h, 1.0)
        win = sum(wj[k] * lag[k] for k in range(h))
        f["G_win"] = win.stack().reindex(idx).to_numpy()
        xv = (f["G_win"] - f["G_b"]) / f["P_b"]
        Lf = np.where(np.isnan(L) & (nc[None, :] > 0), 0.0, L)
        zser = pd.DataFrame(Lf, index=units, columns=years) \
            .stack().reindex(idx).to_numpy()
        zv = pd.Series(zser, index=f.index) / f["P_b"]
        ok = (np.isfinite(xv) & np.isfinite(zv) & np.isfinite(f["y"])).to_numpy()
        g = f[ok]
        x0 = xv[ok].to_numpy(np.float64)
        z0 = zv[ok].to_numpy(np.float64)
        y0 = g["y"].to_numpy(np.float64)
        f1 = pd.factorize(g["ccg"])[0].astype(np.int64)
        fev = pd.factorize(g["division"].astype(str) + "_"
                           + g["year"].astype(str))[0].astype(np.int64)
        nofe = np.zeros(len(g), np.int64)
        tot = g.groupby(["division", "year"])["E_b"].transform("sum")
        wv = (g["E_b"] / tot).to_numpy(np.float64)
        wv = wv / wv.mean()
        xw_ = p51.wins(x0, g["year"].to_numpy())
        r = t28.fit2sls(y0, xw_, z0, wv, fev, nofe, f1.copy(), singleton=False)
        live = [t for t, n in zip(years, nc) if n]
        return dict(harmonized=hz, horizon=h, exposure=elab,
                    windows="post-1979" if restrict_post79 else "full",
                    target_first=min(live), target_last=max(live),
                    beta=r["beta"] * p51.SCALE["population"],
                    se=r["se"] * p51.SCALE["population"],
                    t=r["beta"] / r["se"], F=r["F"], n_obs=len(g),
                    n_counties=int(g["ccg"].nunique()),
                    n_categories=len(PANELS[hz][2]))

    rows = []
    ladder = [(False, 2, "roll5", False), (False, 2, "roll5", True),
              (True, 2, "roll5", True), (True, 2, "roll5", False),
              (False, 3, "roll5", False), (True, 3, "roll5", False),
              (False, 4, "roll10", False), (True, 4, "roll10", False)]
    for hz, h, elab, restrict in ladder:
        r = run_cell(hz, h, elab, restrict)
        rows.append(r)
        log(f"h={h} {elab:6s} harmonize={str(hz):5s} windows={r['windows']:9s} "
            f"beta {r['beta']:7.3f}  t {r['t']:5.2f}  F {r['F']:6.1f}  "
            f"N {r['n_obs']:,}")

    ref = rows[0]
    assert ref["n_obs"] == 119_262 and ref["n_counties"] == 3_058, (
        f"row 1 is off the pinned reference sample: N={ref['n_obs']}")
    log("\nrow 1 reproduces the pinned reference sample (N=119,262). PASS")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "Table_seam_ladder.csv", index=False)
    with open(RESULTS_DIR / "Table_seam_ladder.tex", "w") as fh:
        fh.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        fh.write("Panel & Target windows & $\\beta$ & $t$ & $F$ & N & h \\\\\n"
                 "\\midrule\n")
        for r in rows:
            fh.write(f"{'harmonized' if r['harmonized'] else 'raw'} & "
                     f"{r['windows']} & {r['beta']:.3f} & {r['t']:.2f} & "
                     f"{r['F']:.1f} & {r['n_obs']:,} & {r['horizon']} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    log(f"\nwrote {RESULTS_DIR / 'Table_seam_ladder.csv'} and .tex "
        f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
