"""Re-audit the instrument's flow-construction constants on the harmonized panel.

Producer 54's construction constants were each chosen by a contest run on the
raw category panel: 4-digit codes beat 2-digit top-50 (t 3.29 vs 2.34),
top-300 beat top-800 (3.29 vs 3.23), own-mean rate scaling beat base-year
scaling (3.29 vs 1.65), clip 1.0 beat 2.0 (F 44.1-48.4 vs 26.7-37.5). Those
scores are now known to have carried counterfeit strength from the 1989
renumbering artifact, so each contest is re-staged here on the harmonized
panel, pooled 1977-2018, at the reference (h=2, roll5) and headline (h=4,
roll10) cells, under the fixed specification block of the seam ladder
(net rule, no floor, no screen, population denominator, lam 1.0,
division-by-year FE, division-EW weights, winsor group year).

Pre-committed decision rule, fixed before any number was seen: an ingredient
changes only if its rematch REVERSES the original contest at both cells;
otherwise it is kept and the row documents the re-check. This table reports;
nothing downstream switches construction without separate approval.

Digit variants truncate the harmonized 4-character code afterwards, never
before, because the bridge and relabel maps are keyed on 4-character codes.

Row 1 (the current construction) must reproduce the seam ladder's harmonized
rows, which is the regression test that this file's local panel builder
matches producer 54's.

Output: Results/ns14_diagnostics/Table_flow_audit.csv and .tex.
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
LOG_FILE = LOG_DIR / f"flow_audit_{datetime.now():%Y%m%d_%H%M%S}.log"

sys.argv = [sys.argv[0]]

VARIANTS = [
    ("current (4-digit, top-800, own-mean, clip 1)", 4, 800, "own", 1.0),
    ("2-digit, top-50", 2, 50, "own", 1.0),
    ("4-digit, top-50", 4, 50, "own", 1.0),
    ("4-digit, top-300", 4, 300, "own", 1.0),
    ("base-year rate scaling", 4, 800, "base", 1.0),
    ("clip 2.0", 4, 800, "own", 2.0),
]
CELLS = ((2, "roll5"), (4, "roll10"))
# the harmonized ncat-800 grid's reference and headline cells
LADDER_PIN = {(2, "roll5"): (2.530, 29.8), (4, "roll10"): (4.277, 36.5)}


def log(msg=""):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(str(msg) + "\n")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("== flow-construction audit (producer 62) ==")

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

    fips_to_ccg, _, _ = dc.load_ccg_map()
    B = p51.base_panel()
    units = np.array(sorted(B["ccg"].unique()))
    CON = p51.contract_matrices(units, 1966, 2018, drop_construction=True)
    STATE = B.drop_duplicates("ccg").set_index("ccg")["state"]
    DIVI = B.drop_duplicates("ccg").set_index("ccg")["division"]
    cfg = p51.ERAS["full"]
    piv = {c: B.pivot(index="ccg", columns="year", values=c).reindex(units)
           for c in ("Y", "P", "E")}
    unit_pos = {u: i for i, u in enumerate(units)}
    years_all = list(range(1966, 2019))

    # one pass over the harmonized records, cached as county x 4-char cells
    cells = {}
    for y in years_all:
        d, ccg, cat = p54.category_actions(y, fips_to_ccg, True, harmonize=True)
        g = (pd.DataFrame({"ccg": ccg, "code": cat, "d": d * dc.deflate(y)})
             .groupby(["ccg", "code"], as_index=False)["d"].sum())
        cells[y] = g
    log(f"harmonized cells cached, {sum(len(v) for v in cells.values()):,} rows "
        f"[{time.time()-t0:.0f}s]")

    def build_panel(ndig, ncat):
        nat = {}
        for y in years_all:
            g = cells[y]
            s = g.groupby(g["code"].str[:ndig])["d"].sum()
            for k, v in s.items():
                nat[k] = nat.get(k, 0.0) + v
        cats = list(pd.Series(nat).nlargest(ncat).index)
        cat_pos = {c: j for j, c in enumerate(cats)}
        arr = np.zeros((len(units), len(cats), len(years_all)), np.float32)
        for t, y in enumerate(years_all):
            g = cells[y]
            j = g["code"].str[:ndig].map(cat_pos)
            i = g["ccg"].map(unit_pos)
            m = j.notna() & i.notna()
            np.add.at(arr, (i[m].astype(int).to_numpy(),
                            j[m].astype(int).to_numpy(), t),
                      g.loc[m, "d"].to_numpy(np.float32))
        nat_df = pd.DataFrame(arr.sum(axis=0), index=cats, columns=years_all)
        return arr, nat_df, cats

    def rates(nat_df, h, mode, clip):
        v = nat_df.to_numpy(np.float64)
        out = np.full_like(v, np.nan)
        out[:, h:] = v[:, h:] - v[:, :-h]
        if mode == "own":
            scale = np.nanmean(np.abs(v), axis=1, keepdims=True)
            scale[scale == 0] = np.nan
            return np.clip(out / scale, -clip, clip)
        base = np.full_like(v, np.nan)
        base[:, h:] = v[:, :-h]
        base[base <= 0] = np.nan
        return np.clip(out / base, -clip, clip)

    def run_cell(arr, nat_df, mode, clip, h, elab):
        spec = p51.ROLLING[elab]
        years = [t for t in range(max(cfg["lo"], cfg["first_base"] + h),
                                  cfg["hi"] + 1)]
        idx = pd.MultiIndex.from_product([units, years], names=["ccg", "year"])
        f = pd.DataFrame(index=idx).reset_index()
        f["state"] = f["ccg"].map(STATE)
        f["division"] = f["ccg"].map(DIVI)
        for c in ("Y", "P", "E"):
            f[c] = piv[c].stack().reindex(idx).to_numpy()
            f[c + "_b"] = piv[c].reindex(columns=[y - h for y in years]) \
                .set_axis(years, axis=1).stack().reindex(idx).to_numpy()
        den = f["E"] + f["E_b"]
        f["y"] = np.where(den != 0, 2.0 * (f["E"] - f["E_b"]) / den, np.nan)
        C = CON[("net", 0.0)]
        lag = {k: C.reindex(columns=[y - k for y in years])
               .set_axis(years, axis=1) for k in range(h + 1)}
        f["G_b"] = lag[h].stack().reindex(idx).to_numpy()
        gk = rates(nat_df, h, mode, clip)
        L, nc = p54.predicted_dollars(arr, gk, years_all, years, h, "roll",
                                      spec, p51.ROLL_LAG, p51.DATA_FIRST)
        wj = p51.lam_weights(h, 1.0)
        f["G_win"] = sum(wj[k] * lag[k] for k in range(h)) \
            .stack().reindex(idx).to_numpy()
        xv = (f["G_win"] - f["G_b"]) / f["P_b"]
        Lf = np.where(np.isnan(L) & (nc[None, :] > 0), 0.0, L)
        zv = pd.Series(pd.DataFrame(Lf, index=units, columns=years)
                       .stack().reindex(idx).to_numpy(), index=f.index) / f["P_b"]
        ok = (np.isfinite(xv) & np.isfinite(zv) & np.isfinite(f["y"])).to_numpy()
        g = f[ok]
        x0 = p51.wins(xv[ok].to_numpy(np.float64), g["year"].to_numpy())
        z0 = zv[ok].to_numpy(np.float64)
        y0 = g["y"].to_numpy(np.float64)
        f1 = pd.factorize(g["ccg"])[0].astype(np.int64)
        fev = pd.factorize(g["division"].astype(str) + "_"
                           + g["year"].astype(str))[0].astype(np.int64)
        tot = g.groupby(["division", "year"])["E_b"].transform("sum")
        wv = (g["E_b"] / tot).to_numpy(np.float64)
        wv = wv / wv.mean()
        r = t28.fit2sls(y0, x0, z0, wv, fev, np.zeros(len(g), np.int64),
                        f1.copy(), singleton=False)
        return dict(beta=r["beta"] * 1e5, se=r["se"] * 1e5,
                    t=r["beta"] / r["se"], F=r["F"], n_obs=len(g))

    rows = []
    for label, ndig, ncat, mode, clip in VARIANTS:
        arr, nat_df, cats = build_panel(ndig, ncat)
        for h, elab in CELLS:
            r = run_cell(arr, nat_df, mode, clip, h, elab)
            rows.append(dict(variant=label, ndig=ndig, ncat=len(cats),
                             rate=mode, clip=clip, horizon=h, exposure=elab,
                             **r))
            log(f"{label:42s} h={h} {elab:6s}: beta {r['beta']:7.3f} "
                f"t {r['t']:5.2f} F {r['F']:6.1f} N {r['n_obs']:,} "
                f"[{time.time()-t0:.0f}s]")
        del arr

    out = pd.DataFrame(rows)

    # regression test: row 1 must reproduce the seam ladder's harmonized rows
    for (h, elab), (pb, pf_) in LADDER_PIN.items():
        r = out[(out["variant"].str.startswith("current")) & (out["horizon"] == h)]
        ok = (abs(r["beta"].iloc[0] - pb) < 0.05 and abs(r["F"].iloc[0] - pf_) < 1.0)
        log(f"pin check h={h}: beta {r['beta'].iloc[0]:.3f} vs {pb}, "
            f"F {r['F'].iloc[0]:.1f} vs {pf_}  {'PASS' if ok else 'FAIL'}")
        assert ok, "current construction does not reproduce the seam ladder"

    # verdicts under the pre-committed rule: reversal at both cells required
    cur = {h: out[(out["variant"].str.startswith("current"))
                  & (out["horizon"] == h)].iloc[0] for h, _ in CELLS}
    verdicts = {}
    for label in out["variant"].unique():
        if label.startswith("current"):
            continue
        wins = 0
        for h, _ in CELLS:
            r = out[(out["variant"] == label) & (out["horizon"] == h)].iloc[0]
            if abs(r["t"]) > abs(cur[h]["t"]):
                wins += 1
        verdicts[label] = "REVERSED" if wins == len(CELLS) else "kept"
    log("\nverdicts (change only on reversal at both cells):")
    for k, v in verdicts.items():
        log(f"  {k}: {v}")
    out["verdict"] = out["variant"].map(verdicts).fillna("(incumbent)")

    out.to_csv(RESULTS_DIR / "Table_flow_audit.csv", index=False)
    with open(RESULTS_DIR / "Table_flow_audit.tex", "w") as fh:
        fh.write("\\begin{tabular}{lrrrrl}\n\\toprule\n")
        fh.write("Construction & h & $\\beta$ & $t$ & $F$ & verdict \\\\\n\\midrule\n")
        for _, r in out.iterrows():
            fh.write(f"{r['variant']} & {r['horizon']} & {r['beta']:.3f} & "
                     f"{r['t']:.2f} & {r['F']:.1f} & {r['verdict']} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    log(f"\nwrote {RESULTS_DIR / 'Table_flow_audit.csv'} and .tex "
        f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
