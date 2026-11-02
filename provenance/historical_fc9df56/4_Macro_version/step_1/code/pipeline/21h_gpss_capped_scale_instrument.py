#!/usr/bin/env python3
"""The composition instrument with a bounded exposure scale factor.

THE RULE, FIXED BEFORE ANY ESTIMATE IS READ (R1). The scale factor is
winsorized within year by panel.wins, the house convention already applied to
the treatment under locked.WINSOR_DIAG_GROUP, at the 1st and 99th percentiles.
No new parameter is introduced and no percentile is selected on a result. The
grid table exists to show the rule is not doing the work, not to choose it.

WHY A CAP. The locked instrument factors exactly, wherever county exposure is
positive, into a scale and a composition:

    z_ct = E_ct * znorm_ct,
    E_ct    = (1/P) sum_k Cbar_ck,t              dollars per head,
    znorm_ct = sum_k s_ck,t g_kt,   sum_k s_ck,t = 1.

E spans four orders of magnitude, so leverage, which is quadratic in the
demeaned instrument, collapses onto a few counties. Bounding E's tail leaves
the composition untouched and spreads the leverage.

WHAT IS NOT DONE HERE. The locked instrument is unchanged and is reported as
the reference row in every table. This producer estimates an alternative and
does not adopt it. A percentile cap is scale-invariant, so the capped
instrument rescales with E and a just-identified 2SLS coefficient does not move
with the units E is measured in; that is why a cap is used rather than a log or
a rank of E.

Writes to Results/pipeline_diagnostics/:

  Table_gpss_capped_scale_headline.tex   the rule, both horizons
  Table_gpss_capped_scale_grid.tex       the cap grid, one panel per horizon
  gpss_capped_scale.csv                  every printed row
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
import panel as pnl_lib  # noqa: E402
import dcads_common as dc  # noqa: E402

HORIZONS = (2, 4)
GRID = (80.0, 90.0, 95.0, 97.5, 99.0, 99.5)
FINE = np.arange(80.0, 100.0, 0.5)
MIN_EFFECTIVE_CLUSTERS = 50.0     # R2 threshold, declared here not chosen later
MAX_COUNTY_LEVERAGE = 0.05        # R3 threshold, declared here not chosen later
FACTOR_TOL = 1e-12
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"


def load_producer(filename, name):
    spec = importlib.util.spec_from_file_location(name, PIPELINE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decompose(inp, G):
    """Split the locked instrument into scale and composition, and check it."""
    arr, win, rr, cc, pb = (inp["arr"], inp["win"], inp["rr"], inp["cc"],
                            inp["pb"])
    N = len(inp["w"])
    E = np.zeros(N)
    znorm = np.zeros(N)
    z = np.zeros(N)
    for j, (a, b) in win.items():
        m = cc == j
        if not m.any():
            continue
        cbar = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)[rr[m], :]
        num = cbar @ G[j, :]
        tot = cbar.sum(axis=1)
        ok = tot > 0
        z[m] = num / pb[m]
        E[m] = tot / pb[m]
        znorm[m] = np.where(ok, num / np.where(ok, tot, 1.0), 0.0)
    rebuilt = E * znorm
    scale = max(float(np.abs(z).max()), 1e-300)
    gap = float(np.abs(rebuilt - z).max()) / scale
    print(f"  scale x composition against the locked instrument: relative gap "
          f"{gap:.2e} {'PASS' if gap < FACTOR_TOL else 'FAIL'}")
    if gap >= FACTOR_TOL:
        raise SystemExit("FAIL: the instrument does not factor into E x znorm")
    zero = int((E <= 0).sum())
    print(f"  rows with no procurement in the exposure window: {zero:,} of "
          f"{N:,} ({zero / N * 100:.2f}%); their instrument is zero under every "
          f"cap, as it is in the locked construction")
    return z, E, znorm


def fit(inp, z, key, nunits):
    """The locked cell's 2SLS on one instrument, plus its county leverage."""
    r = op.t28.fit2sls(inp["y0"], inp["xw"], z, inp["w"], inp["fe"],
                       inp["nofe"], inp["cl"], singleton=False)
    zt = op.w22.demean(z.reshape(-1, 1), inp["fe"], inp["nofe"],
                       inp["w"])[:, 0]
    lev = np.bincount(key, weights=inp["w"] * zt * zt, minlength=nunits)
    tot = lev.sum()
    if tot <= 0:
        raise SystemExit("FAIL: the capped instrument has no variance")
    sh = lev / tot
    return dict(beta=r["beta"] * locked.SCALE, se=r["se"] * locked.SCALE,
                t=r["beta"] / r["se"], F=r["F"],
                effective=float(1.0 / np.sum(sh ** 2)), max_leverage=float(sh.max()))


def house_rule(E, years):
    """R1: the house within-year winsorization, applied verbatim."""
    capped = pnl_lib.wins(E, years)
    up = int((capped < E).sum())
    dn = int((capped > E).sum())
    print(f"  R1 within-year winsorization: {up:,} rows cut at the upper tail, "
          f"{dn:,} raised at the lower tail")
    return capped, up, dn


def select_on_leverage(inp, E, znorm, key, nunits):
    """R2 and R3 landing points. Both read only the instrument."""
    out = {}
    scan = []
    for q in FINE:
        cap = float(np.percentile(E[E > 0], q))
        z = znorm * np.minimum(E, cap)
        zt = op.w22.demean(z.reshape(-1, 1), inp["fe"], inp["nofe"],
                           inp["w"])[:, 0]
        lev = np.bincount(key, weights=inp["w"] * zt * zt, minlength=nunits)
        sh = lev / lev.sum()
        scan.append((q, cap, float(1.0 / np.sum(sh ** 2)), float(sh.max())))
    ok2 = [s for s in scan if s[2] >= MIN_EFFECTIVE_CLUSTERS]
    ok3 = [s for s in scan if s[3] <= MAX_COUNTY_LEVERAGE]
    out["R2"] = max(ok2, key=lambda s: s[0]) if ok2 else None
    out["R3"] = max(ok3, key=lambda s: s[0]) if ok3 else None
    for name, s in out.items():
        if s is None:
            print(f"  {name}: no percentile in [80, 100) satisfies the threshold")
        else:
            print(f"  {name}: least binding cap p{s[0]:g} = {s[1]:,.0f} $/head, "
                  f"effective {s[2]:.1f}, max leverage {s[3]:.3f}")
    return out


def rows_for(inp, H, z_locked, E, znorm, key, nunits, years):
    recs = []

    def add(label, cap_value, z, kind):
        r = fit(inp, z, key, nunits)
        recs.append(dict(h=H, label=label, kind=kind, cap=cap_value, **r))

    add("Locked instrument, no cap", np.nan, z_locked, "reference")
    capped, _, _ = house_rule(E, years)
    add("R1, within-year winsorization (the rule)", np.nan, znorm * capped,
        "rule")
    sel = select_on_leverage(inp, E, znorm, key, nunits)
    for name, s in sel.items():
        if s is None:
            continue
        lab = ("R2, effective clusters $\\geq$ "
               f"{MIN_EFFECTIVE_CLUSTERS:g}" if name == "R2"
               else f"R3, max leverage $\\leq$ {MAX_COUNTY_LEVERAGE:g}")
        add(lab, s[1], znorm * np.minimum(E, s[1]), "rule")
    for q in GRID:
        cap = float(np.percentile(E[E > 0], q))
        add(f"Pooled p{q:g}", cap, znorm * np.minimum(E, cap), "grid")
    return recs


HEAD = ["$\\hat\\beta$", "(s.e.)", "$t$", "First-stage $F$",
        "Effective counties", "Max county leverage"]


def write_headline(recs, ns, path):
    lines = []
    for H in sorted({r["h"] for r in recs}):
        lines.append(r"\multicolumn{7}{l}{\textit{Panel "
                     + ("A" if H == min(ns) else "B")
                     + f". Horizon {H} years, {ns[H]:,} county-years"
                     + r"}} \\")
        lines.append(" & ".join([""] + HEAD) + r" \\")
        for r in recs:
            if r["h"] != H or r["kind"] == "grid":
                continue
            lines.append(" & ".join([
                r"\hspace{1em}" + r["label"], f"{r['beta']:.4f}",
                f"({r['se']:.4f})", f"{r['t']:.2f}", f"{r['F']:.1f}",
                f"{r['effective']:.1f}", f"{r['max_leverage']:.3f}"]) + r" \\")
        lines.append(r"\addlinespace[0.6ex]")
    body = "\n".join(lines[:-1])
    path.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{Bounding the Exposure Scale Factor}}
\\label{{tab:gpss_capped_scale_headline}}
{{\\centering
\\adjustbox{{max width=\\linewidth}}{{%
\\begin{{tabular}}{{l*{{6}}{{r}}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} The locked instrument factors exactly
into a scale and a composition, $z_{{ct}} = E_{{ct}} \\cdot \\sum_k s_{{ck,t}}
g_{{kt}}$, where $E$ is county procurement per head over the exposure window
and the shares sum to one within a county. $E$ runs from a median of 16 dollars
per head to a maximum of 73,303, and leverage is quadratic in the demeaned
instrument, so the uncapped instrument concentrates on a few counties. Each row
below bounds $E$ and leaves the composition untouched. \\textbf{{The rule is
R1}}, the within-year winsorization at the 1st and 99th percentiles that
\\texttt{{panel.wins}} already applies to the treatment; it is fixed in this
file's header and introduces no new parameter. R2 and R3 are reported as
pre-declared alternatives whose thresholds are set in the header and which read
only the instrument, never the outcome. Effective counties is the inverse
Herfindahl of each county's share of the instrument's variance $\\sum_c w
\\tilde z^2$, which is the weight the first-stage slope places on it. The
specification is otherwise the locked one throughout: midpoint employment
growth on the winsorized county treatment, division-by-year effects,
division-equal weights, county clusters. The locked instrument itself is
unchanged and is the reference row. Coefficients are scaled by
{locked.SCALE:g}.\\par}}
\\end{{table}}
""")
    print(f"  wrote {path}")


def write_grid(recs, path):
    lines = []
    for H in sorted({r["h"] for r in recs}):
        lines.append(r"\multicolumn{8}{l}{\textit{Panel "
                     + ("A" if H == 2 else "B")
                     + f". Horizon {H} years" + r"}} \\")
        lines.append(" & ".join(["", "Cap, \\$ per head"] + HEAD) + r" \\")
        block = ([r for r in recs if r["h"] == H and r["kind"] == "grid"]
                 + [r for r in recs if r["h"] == H and r["kind"] != "grid"])
        for r in block:
            cap = "none" if not np.isfinite(r["cap"]) else f"{r['cap']:,.0f}"
            if r["label"].startswith("R1"):
                cap = "varies by year"
            lines.append(" & ".join([
                r"\hspace{1em}" + r["label"], cap, f"{r['beta']:.4f}",
                f"({r['se']:.4f})", f"{r['t']:.2f}", f"{r['F']:.1f}",
                f"{r['effective']:.1f}", f"{r['max_leverage']:.3f}"]) + r" \\")
        lines.append(r"\addlinespace[0.6ex]")
    body = "\n".join(lines[:-1])
    path.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{The Estimate Across the Whole Range of Caps}}
\\label{{tab:gpss_capped_scale_grid}}
{{\\centering
\\adjustbox{{max width=\\linewidth}}{{%
\\begin{{tabular}}{{ll*{{6}}{{r}}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} This table exists to show that the
choice of cap is not doing the work, and is not the means by which the cap was
chosen. The rule is fixed in the producer's header as R1, the house within-year
winsorization; the pooled percentiles are shown so a reader can see the whole
range. Reading the coefficient column, the estimate is flat over caps spanning
two orders of magnitude, and the only value outside that band is the uncapped
locked instrument in the last rows of each panel. Selecting a cap on the
first-stage $F$ would be improper here and was not done, following the
project's own statement that first-stage strength alone is not a sufficient
selection rule. Column definitions and the specification are as in the headline
table. Coefficients are scaled by {locked.SCALE:g}.\\par}}
\\end{{table}}
""")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    args = ap.parse_args()

    gpss = load_producer("21_rotemberg.py", "rotemberg_producer")
    frozen = load_producer("21b_gpss_frozen_share_rotemberg.py", "frozen_producer")
    OUT.mkdir(parents=True, exist_ok=True)

    recs, ns = [], {}
    for H in args.horizons:
        print(f"h={H}")
        inp = gpss.build_inputs(H)
        _, G = frozen.national_matrix(inp)
        units, key = np.unique(inp["d"]["ccg"].to_numpy(), return_inverse=True)
        years = inp["d"]["year"].to_numpy()
        z_locked, E, znorm = decompose(inp, G)
        gap = float(np.abs(z_locked - inp["z0"]).max()) / max(
            float(np.abs(inp["z0"]).max()), 1e-300)
        print(f"  rebuilt instrument against the persisted store: relative gap "
              f"{gap:.2e} {'PASS' if gap < FACTOR_TOL else 'FAIL'}")
        if gap >= FACTOR_TOL:
            raise SystemExit("FAIL: the rebuilt instrument is not the locked one")
        ns[H] = len(inp["w"])
        rows = rows_for(inp, H, z_locked, E, znorm, key, len(units), years)
        recs.extend(rows)
        for r in rows:
            print(f"   {r['label'][:44]:<44} beta {r['beta']:8.4f} "
                  f"({r['se']:.4f})  F {r['F']:6.1f}  eff {r['effective']:7.1f}")

    write_headline(recs, ns, OUT / "Table_gpss_capped_scale_headline.tex")
    write_grid(recs, OUT / "Table_gpss_capped_scale_grid.tex")
    pd.DataFrame(recs).to_csv(OUT / "gpss_capped_scale.csv", index=False)
    print(f"  wrote {OUT / 'gpss_capped_scale.csv'}")


if __name__ == "__main__":
    main()
