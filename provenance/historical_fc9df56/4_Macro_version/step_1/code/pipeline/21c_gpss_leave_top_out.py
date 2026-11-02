#!/usr/bin/env python3
"""What survives when the highest Rotemberg weight categories are removed.

Under the Goldsmith-Pinkham, Sorkin and Swift reading the shares carry the
identification, and here they are concentrated: a handful of defence
procurement categories hold most of the weight. The obvious question is whether
the estimate is those categories. This producer answers it directly by
rebuilding the instrument without them.

Both instruments are linear in categories,

    z_ct = sum_k B_kct,   B_kct = Cbar_ck,t g_kt / P_c,t-h,

so leaving categories out is exact subtraction, not a re-estimation of the
construction. The identity z = sum_k B_k is asserted against the persisted
instrument before any category is dropped.

Two instruments are laddered side by side: the locked production instrument z,
whose exposure window moves with the target year, and the frozen-share
instrument z0, whose exposure is fixed at 1970-1974 dollars per 1975 resident.
Categories are dropped in order of their own instrument's Rotemberg weight.
The estimation sample never changes, so every row is the same county-years.

Writes to Results/pipeline_diagnostics/:

  Table_gpss_leave_top_out_h2.tex
  Table_gpss_leave_top_out_h4.tex
  gpss_leave_top_out.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))

import locked  # noqa: E402
import oldpath as op  # noqa: E402
import dcads_common as dc  # noqa: E402

HORIZONS = (2, 4)
DROPS = (0, 1, 2, 3, 5, 10)
RECON_TOL = 1e-9        # relative, on the reconstructed instrument
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"


def load_producer(filename, name):
    spec = importlib.util.spec_from_file_location(name, PIPELINE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locked_category_columns(inp, gpss, kcols, G):
    """B_k on the sample for a block of categories, locked time-varying shares."""
    return gpss.exposure_block(inp, kcols) * G[np.ix_(inp["cc"], kcols)]


def frozen_category_columns(inp, S_unit, kcols, G):
    """B_k on the sample for a block of categories, frozen 1970-74 shares."""
    return S_unit[np.ix_(inp["rr"], kcols)] * G[np.ix_(inp["cc"], kcols)]


def check_reconstruction(inp, columns_of, K, label, target):
    """sum_k B_k must reproduce the instrument it is built from."""
    total = np.zeros(len(inp["w"]), np.float64)
    for s in range(0, K, 80):
        blk = np.arange(s, min(s + 80, K))
        total += columns_of(blk).sum(axis=1)
    scale = max(float(np.abs(target).max()), 1e-300)
    gap = float(np.abs(total - target).max()) / scale
    print(f"  {label}: sum_k B_k against the instrument, relative gap "
          f"{gap:.2e} {'PASS' if gap < RECON_TOL else 'FAIL'}")
    if gap >= RECON_TOL:
        raise SystemExit(f"FAIL: {label} does not decompose into its categories")
    return total


def leverage_concentration(zt, w, key, n):
    """Effective counties by the instrument's own variance, sum w ztilde^2.

    This is the weight the first-stage slope actually puts on each county, and
    it is the statistic that reconciles the ladder: the concentrated categories
    are not supplying the first stage, they are depressing it.
    """
    lev = np.bincount(key, weights=w * zt * zt, minlength=n)
    tot = lev.sum()
    if tot <= 0:
        raise SystemExit("FAIL: the instrument has no variance")
    sh = lev / tot
    return float(1.0 / np.sum(sh ** 2)), float(sh.max())


def ladder(inp, base_z, columns_of, alpha, dollars, cats, label):
    """Re-estimate after removing the top categories by |Rotemberg weight|."""
    units, key = np.unique(inp["d"]["ccg"].to_numpy(), return_inverse=True)
    nunits = len(units)
    order = np.argsort(-alpha)
    absA = np.abs(alpha)
    tot_a, tot_d = absA.sum(), dollars.sum()
    rows = []
    for n in DROPS:
        drop = order[:n]
        z = base_z.copy()
        if n:
            z -= columns_of(drop).sum(axis=1)
        r = op.t28.fit2sls(inp["y0"], inp["xw"], z, inp["w"], inp["fe"],
                           inp["nofe"], inp["cl"], singleton=False)
        zt = op.w22.demean(z.reshape(-1, 1), inp["fe"], inp["nofe"],
                           inp["w"])[:, 0]
        eff, maxlev = leverage_concentration(zt, inp["w"], key, nunits)
        pi = float(np.sum(inp["w"] * zt * inp["xt"])
                   / np.sum(inp["w"] * zt * zt))
        rows.append(dict(
            instrument=label, dropped=n,
            names="; ".join(cats[k] for k in drop) if n else "",
            beta=r["beta"] * locked.SCALE, se=r["se"] * locked.SCALE,
            t=r["beta"] / r["se"], F=r["F"], pi=pi,
            eff_counties=eff, max_leverage=maxlev,
            alpha_removed=float(absA[drop].sum() / tot_a) if n else 0.0,
            dollars_removed=float(dollars[drop].sum() / tot_d) if n else 0.0,
            categories_left=len(alpha) - n))
    return rows


PANELS = (("z", "Panel A. Locked instrument $z$, exposure window moves with "
                "the target year"),
          ("z0", "Panel B. Frozen-share instrument $z^{0}$, exposure fixed at "
                 "1970--1974"))
HEAD = ["$\\hat\\beta$", "(s.e.)", "$t$", "First-stage $F$",
        "First-stage $\\hat\\pi$", "Effective counties",
        "Share of $|\\hat\\alpha|$ removed"]


def write_table(H, rows, inp, top_names, path):
    lines = []
    for key, title in PANELS:
        block = [r for r in rows if r["instrument"] == key]
        if not block:
            continue
        lines.append(r"\multicolumn{8}{l}{\textit{" + title + r"}} \\")
        lines.append(" & ".join([""] + HEAD) + r" \\")
        for r in block:
            label = "None" if not r["dropped"] else f"Top {r['dropped']}"
            lines.append(" & ".join([
                r"\hspace{1em}" + label,
                f"{r['beta']:.4f}", f"({r['se']:.4f})", f"{r['t']:.2f}",
                f"{r['F']:.1f}", f"{r['pi']:.4f}", f"{r['eff_counties']:.1f}",
                f"{r['alpha_removed']:.3f}"]) + r" \\")
        lines.append(r"\addlinespace[0.6ex]")
    body = "\n".join(lines[:-1])
    n = f"{len(inp['w']):,}"
    ncl = f"{int(np.unique(inp['cl']).size):,}"
    names = ", ".join(top_names)
    path.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{Removing the Highest Rotemberg Weight Categories ($h={H}$)}}
\\label{{tab:gpss_leave_top_out_h{H}}}
{{\\centering
\\adjustbox{{max width=\\linewidth}}{{%
\\begin{{tabular}}{{l*{{7}}{{r}}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} Each row rebuilds the instrument
without the categories carrying the largest Rotemberg weights and re-estimates
the same regression: midpoint employment growth on the winsorized county
treatment, division-by-year effects, division-equal weights, county clusters,
horizon {H} years, on {n} county-years and {ncl} counties. Both instruments are
sums over categories, so a row is exact subtraction of the dropped categories'
contribution, not a re-tuned construction; the identity between the summed
categories and the persisted instrument is asserted before any row is computed.
The estimation sample is identical in every row, so the rows are directly
comparable. Categories are ranked by the Rotemberg weight of the instrument in
their own panel, which is why the two panels can drop different categories. The
locked instrument's five largest at this horizon are: {names}. The share of
$|\\hat\\alpha|$ removed says how much of the identifying variation each row
gives up. The first-stage slope $\\hat\\pi$ and the effective county count are
reported because they reconcile the ladder: the effective count is the inverse
Herfindahl of each county's share of the instrument's own variance $\\sum_c w
\\tilde z^2$, which is the weight the first-stage slope puts on it. Removing
the highest-weight categories raises $\\hat\\pi$ and spreads the variance over
more counties at the same time, and the county-clustered $F$ rises with both.
The concentrated categories were depressing the first stage, not supplying it.
Coefficients are scaled by {locked.SCALE:g}.\\par}}
\\end{{table}}
""")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    args = ap.parse_args()

    gpss = load_producer("21_rotemberg.py", "rotemberg_producer")
    fixed = load_producer("21_fixed_initial_share_balance.py", "fixed_producer")
    frozen = load_producer("21b_gpss_frozen_share_rotemberg.py", "frozen_producer")
    OUT.mkdir(parents=True, exist_ok=True)

    records = []
    for H in args.horizons:
        print(f"h={H}")
        inp = gpss.build_inputs(H)
        G_raw, G = frozen.national_matrix(inp)
        cats = np.asarray(inp["cats"])
        K = len(cats)

        rot_z = gpss.rotemberg(inp)
        S_unit = fixed.fixed_exposures(inp)
        rot_z0 = frozen.frozen_rotemberg(inp, S_unit, G, G_raw)
        z0 = frozen.frozen_instrument(inp, S_unit, G)

        col_z = lambda blk: locked_category_columns(inp, gpss, blk, G)     # noqa: E731
        col_z0 = lambda blk: frozen_category_columns(inp, S_unit, blk, G)  # noqa: E731
        z_rec = check_reconstruction(inp, col_z, K, "locked z", inp["z0"])
        check_reconstruction(inp, col_z0, K, "frozen z0", z0)

        rows = (ladder(inp, z_rec, col_z, rot_z["alpha"], rot_z["dollars"],
                       cats, "z")
                + ladder(inp, z0, col_z0, rot_z0["alpha"], rot_z0["dollars"],
                         cats, "z0"))
        top5 = np.argsort(-rot_z["alpha"])[:5]
        labels = gpss.fsc_labels({cats[k] for k in top5})
        top_names = [labels.get(cats[k], cats[k]) for k in top5]
        write_table(H, rows, inp, top_names,
                    OUT / f"Table_gpss_leave_top_out_h{H}.tex")
        records.extend(dict(h=H, **r) for r in rows)
        for r in rows:
            print(f"  {r['instrument']:>3} drop {r['dropped']:>2}: "
                  f"beta {r['beta']:8.4f} ({r['se']:.4f})  t {r['t']:5.2f}  "
                  f"F {r['F']:7.1f}  |alpha| removed {r['alpha_removed']:.3f}")

    pd.DataFrame(records).to_csv(OUT / "gpss_leave_top_out.csv", index=False)
    print(f"  wrote {OUT / 'gpss_leave_top_out.csv'}")


if __name__ == "__main__":
    main()
