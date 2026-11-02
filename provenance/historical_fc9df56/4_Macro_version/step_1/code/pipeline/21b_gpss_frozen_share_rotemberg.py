#!/usr/bin/env python3
"""GPSS Rotemberg decomposition with county exposures frozen before the sample.

The locked instrument z carries a time-varying exposure window and producer 21
decomposes it. This producer decomposes the frozen-share instrument z0 that
producer 21_fixed_initial_share_balance already constructs: county-category
exposure is mean real contract dollars over 1970-1974 divided by county
population in 1975, and only the national category shocks move. That is the
object Goldsmith-Pinkham, Sorkin and Swift decompose, and it is the version
whose shares carry the identification.

The locked design is untouched. z0 is a diagnostic instrument estimated
alongside z, and every table reports the locked z row as its anchor.

Writes to Results/pipeline_diagnostics/:

  Table_gpss_frozen_rotemberg_h2.tex   Panels A-E at h=2
  Table_gpss_frozen_rotemberg_h4.tex   Panels A-E at h=4
  gpss_frozen_rotemberg.csv            every printed row, both horizons
  gpss_frozen_rotemberg_weights.csv    all categories, both horizons
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
from clusterstats import hhi_effective  # noqa: E402

HORIZONS = (2, 4)
TOP_N = 5
FBLOCK = 80
ALPHA_TOL = 1e-10
FVEC_TOL = 1e-8
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"


def load_producer(filename, name):
    spec = importlib.util.spec_from_file_location(name, PIPELINE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------ the design --

def national_matrix(inp):
    """T x K national category shocks on the target years.

    Returns the raw matrix, which keeps NaN where a category-year shock is
    undefined, and the filled matrix used in every sum. Producer 21 keeps the
    same distinction through its `fin` mask: a missing shock and a shock of
    exactly zero are different objects, and averaging g_k over the filled
    matrix would silently count the first as the second.
    """
    G_raw = inp["gk"][:, [inp["cy"].index(y) for y in inp["years"]]].T
    G_raw = np.asarray(G_raw, np.float64)
    return G_raw, np.where(np.isfinite(G_raw), G_raw, 0.0)


def frozen_instrument(inp, S_unit, G):
    z = np.full(len(inp["w"]), np.nan)
    for j in range(len(inp["years"])):
        m = inp["cc"] == j
        z[m] = S_unit[inp["rr"][m], :] @ G[j, :]
    if not np.isfinite(z).all():
        raise SystemExit("FAIL: the frozen-share instrument has missing values")
    return z


def frozen_rotemberg(inp, S_unit, G, G_raw):
    """alpha_kt, beta_k, F_k and var(z_k) on frozen exposures.

    Mirrors 21_rotemberg.rotemberg row for row, with the time-varying window
    mean replaced by the single frozen exposure matrix. The per-category first
    stage is the same blocked, two-way-demeaned computation.
    """
    w, yt, xt = inp["w"], inp["yt"], inp["xt"]
    rr, cc, fe = inp["rr"], inp["cc"], inp["fe"]
    K, T, N = S_unit.shape[1], len(inp["years"]), len(w)

    NX = np.full((K, T), np.nan)
    NY = np.full((K, T), np.nan)
    VZ = np.full((K, T), np.nan)
    empty = 0
    for j in range(T):
        m = cc == j
        if not m.any():                 # a target year with no county in sample
            empty += 1
            NX[:, j] = 0.0
            NY[:, j] = 0.0
            continue
        Sj = S_unit[rr[m], :]
        wsk = Sj * w[m][:, None]
        NX[:, j] = wsk.T @ xt[m]
        NY[:, j] = wsk.T @ yt[m]
        VZ[:, j] = Sj.var(axis=0)
    if empty:
        print(f"  {empty} of {T} target years carry no county and contribute "
              f"nothing to the weights")
    if not (np.isfinite(NX).all() and np.isfinite(NY).all()):
        raise SystemExit("FAIL: a frozen-share moment is not finite")

    gNX = G.T * NX
    gNY = G.T * NY
    D = float(gNX.sum())
    if abs(D) < 1e-300:
        raise SystemExit("FAIL: the frozen-share first-stage covariance is zero")
    A = gNX / D
    alpha = A.sum(axis=1)
    dk = gNX.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta_k = np.where(dk != 0, gNY.sum(axis=1) / dk, np.nan)
    with np.errstate(invalid="ignore"):
        g_k = np.where(np.isfinite(G_raw.T).any(axis=1),
                       np.nanmean(G_raw.T, axis=1), np.nan)
        vz_k = np.where(np.isfinite(VZ).any(axis=1),
                        np.nanmean(VZ, axis=1), np.nan)
    dollars = inp["arr"].sum(axis=(0, 2)).astype(np.float64)

    ssck = op.t28.ssc_k(fe, inp["nofe"], inp["cl"])
    Gcl = int(np.unique(inp["cl"]).size)
    ssc = (Gcl / max(Gcl - 1, 1)) * ((N - 1) / max(N - ssck, 1))
    nfe = int(fe.max()) + 1
    sw = np.bincount(fe, weights=w, minlength=nfe)
    sw[sw == 0] = 1.0

    Fk = np.full(K, np.nan)
    for s in range(0, K, FBLOCK):
        blk = np.arange(s, min(s + FBLOCK, K))
        B = S_unit[np.ix_(rr, blk)] * G[np.ix_(cc, blk)]
        gm = np.column_stack([
            np.bincount(fe, weights=w * B[:, q], minlength=nfe) / sw
            for q in range(len(blk))
        ])
        Bt = B - gm[fe, :]
        for q, kq in enumerate(blk):
            bt = Bt[:, q]
            bb = float(bt @ (w * bt))
            if bb <= 0:
                continue
            bfs = float(bt @ (w * xt)) / bb
            se = op.t28.crv1(bt, bt, xt - bfs * bt, w, inp["cl"], N, ssck)
            if se and np.isfinite(se) and se > 0:
                Fk[kq] = (bfs / se) ** 2

    return dict(A=A, alpha=alpha, beta_k=beta_k, g_k=g_k, vz_k=vz_k, Fk=Fk,
                dollars=dollars, ssc=ssc, ssck=ssck)


def single_category(inp, S_unit, G, k):
    """The one-category frozen Bartik, undemeaned."""
    return S_unit[inp["rr"], k] * G[inp["cc"], k]


def demeaned(inp, v):
    return op.w22.demean(v.reshape(-1, 1), inp["fe"], inp["nofe"], inp["w"])[:, 0]


def verify(inp, fit, rot, S_unit, G):
    sa = float(rot["alpha"].sum())
    if abs(sa - 1.0) > ALPHA_TOL:
        raise SystemExit(f"FAIL: sum alpha = {sa:.15f}, not one")
    dec = float(np.nansum(rot["alpha"] * rot["beta_k"]))
    rel = abs(dec - fit["beta"]) / max(abs(fit["beta"]), 1e-300)
    print(f"  sum alpha = {sa:.12f}")
    print(f"  beta_2SLS {fit['beta']*locked.SCALE:.6f} vs sum alpha_k beta_k "
          f"{dec*locked.SCALE:.6f}, relative gap {rel:.2e} "
          f"{'PASS' if rel < locked.RTOL_COEF else 'FAIL'}")
    if rel >= locked.RTOL_COEF:
        raise SystemExit("FAIL: the decomposition does not reproduce the 2SLS")

    rng = np.random.default_rng(0)
    top = np.argsort(-rot["alpha"])[:TOP_N]
    probe = list(dict.fromkeys(top.tolist()
                               + rng.choice(len(rot["alpha"]), 20,
                                            replace=False).tolist()))
    worst, used, skipped = 0.0, 0, 0
    for kq in probe:
        B = single_category(inp, S_unit, G, kq)
        bt = demeaned(inp, B)
        if float(bt @ (inp["w"] * bt)) <= 0:
            skipped += 1                # no within-cell variation to instrument
            continue
        used += 1
        r = op.t28.fit2sls(inp["y0"], inp["xw"], B, inp["w"], inp["fe"],
                           inp["nofe"], inp["cl"], singleton=False)
        for a_, b_ in ((r["F"], rot["Fk"][kq]), (r["beta"], rot["beta_k"][kq])):
            if np.isfinite(a_) and np.isfinite(b_) and abs(a_) > 0:
                worst = max(worst, abs(a_ - b_) / abs(a_))
    print(f"  vectorized F_k and beta_k against fit2sls on {used} of "
          f"{len(probe)} probed categories ({skipped} degenerate, skipped): "
          f"worst relative gap {worst:.2e} "
          f"{'PASS' if worst < FVEC_TOL else 'FAIL'}")
    if used < TOP_N:
        raise SystemExit(f"FAIL: only {used} categories were checkable")
    if worst >= FVEC_TOL:
        raise SystemExit("FAIL: the vectorized first stage disagrees with the "
                         "house estimator")


# ------------------------------------------------------------- the panels --

def panel_rows(inp, fit, rot, S_unit, G, gpss, labels):
    """Every printed row, as (panel, label, v1..v6) records."""
    alpha, beta_k, Fk = rot["alpha"], rot["beta_k"], rot["Fk"]
    g_k, vz_k, dollars = rot["g_k"], rot["vz_k"], rot["dollars"]
    scale = locked.SCALE
    rows = []

    def add(panel, label, *values):
        rows.append(dict(panel=panel, label=label,
                         **{f"v{q+1}": values[q] if q < len(values) else ""
                            for q in range(6)}))

    base = inp["base"]
    add("anchor", "Locked instrument $z$",
        f"{base['beta']*scale:.4f}", f"{base['beta']/base['se']:.2f}",
        f"{base['F']:.1f}")
    add("anchor", "Frozen-share instrument $z^{0}$",
        f"{fit['beta']*scale:.4f}", f"{fit['beta']/fit['se']:.2f}",
        f"{fit['F']:.1f}")

    neg, pos = alpha < 0, alpha > 0
    tot = float(np.abs(alpha).sum())
    for name, m in (("Negative", neg), ("Positive", pos)):
        add("A", name, f"{alpha[m].sum():.3f}", f"{alpha[m].mean():.4f}",
            f"{np.abs(alpha[m]).sum()/tot:.3f}", f"{int(m.sum()):d}")

    stats_named = [("$\\hat\\alpha_k$", alpha), ("$g_k$", g_k),
                   ("$\\hat\\beta_k$", beta_k), ("$\\hat F_k$", Fk),
                   ("$\\mathrm{var}(z_k)$", vz_k)]
    M = np.column_stack([v for _, v in stats_named])
    good = np.isfinite(M).all(axis=1)
    C = np.corrcoef(M[good].T)
    for p, (name, _) in enumerate(stats_named):
        add("B", name, *[f"{C[p, q]:.3f}" for q in range(p + 1)])

    decade = np.array([(y // 10) * 10 for y in inp["years"]])
    per_year = rot["A"].sum(axis=0)
    for d in np.unique(decade):
        m = decade == d
        add("C", f"{d}s", f"{per_year[m].sum():.3f}",
            f"{per_year[m].sum()/(len(alpha)*int(m.sum())):.6f}")

    top = np.argsort(-alpha)[:TOP_N]
    yt, xt, w, cl = inp["yt"], inp["xt"], inp["w"], inp["cl"]
    unbounded = 0
    for k in top:
        B = single_category(inp, S_unit, G, k)
        r = op.t28.fit2sls(inp["y0"], inp["xw"], B, w, inp["fe"], inp["nofe"],
                           cl, singleton=False)
        lo, hi = gpss.ar_interval(demeaned(inp, B), yt, xt, w, cl, rot["ssc"])
        ar = gpss.fmt_ci(lo, hi, scale)
        unbounded += ar == "unbounded"
        conv = (f"({(r['beta']-1.96*r['se'])*scale:.2f}, "
                f"{(r['beta']+1.96*r['se'])*scale:.2f})")
        name = labels.get(inp["cats"][k], inp["cats"][k])
        add("D", gpss.tex_escape(name, width=30),
            f"{alpha[k]:.3f}", f"{beta_k[k]*scale:.2f}", conv, ar,
            f"{Fk[k]:.1f}" if np.isfinite(Fk[k]) else "n/a",
            f"{100*dollars[k]/dollars.sum():.2f}")
    print(f"  weak-instrument-robust intervals: {unbounded} of {TOP_N} "
          f"unbounded under cluster-robust Anderson-Rubin")
    sh = np.abs(alpha) / tot
    add("D", "Largest category's share of $|\\hat\\alpha|$", f"{sh.max():.3f}")
    add("D", "Five largest categories' share",
        f"{np.sort(sh)[::-1][:TOP_N].sum():.3f}")
    add("D", "Effective categories (1/HHI on $|\\hat\\alpha|$)",
        f"{hhi_effective(alpha):.1f}")

    fin = np.isfinite(beta_k)
    tot_b = float(np.nansum(alpha[fin] * beta_k[fin]))
    for name, m in (("Negative", neg & fin), ("Positive", pos & fin)):
        s = float(np.sum(alpha[m] * beta_k[m]))
        add("E", name, f"{s*scale:.3f}", f"{s/tot_b:.3f}",
            f"{np.mean(beta_k[m])*scale:.2f}")
    return rows, top


PANEL_TITLES = {
    "anchor": "Panel A. The two instruments on the same sample",
    "A": "Panel B. Negative and positive weights",
    "B": "Panel C. Correlations",
    "C": "Panel D. Variation across decades in $\\hat\\alpha_k$",
    "D": "Panel E. Top five Rotemberg weight categories",
    "E": "Panel F. $\\hat\\beta_k$ by weight sign",
}
PANEL_HEADS = {
    "anchor": ["$\\hat\\beta$", "$t$", "First-stage $F$", "", "", ""],
    "A": ["Sum", "Mean", "Share", "Categories", "", ""],
    "B": ["$\\hat\\alpha_k$", "$g_k$", "$\\hat\\beta_k$", "$\\hat F_k$",
          "$\\mathrm{var}(z_k)$", ""],
    "C": ["Sum", "Mean", "", "", "", ""],
    "D": ["$\\hat\\alpha_k$", "$\\hat\\beta_k$", "Conventional 95\\% CI",
          "AR 95\\% set", "$\\hat F_k$", "Dollar \\%"],
    "E": ["$\\hat\\alpha$-weighted sum", "Share of $\\hat\\beta$",
          "Mean $\\hat\\beta_k$", "", "", ""],
}


def write_table(H, rows, inp, path):
    lines = []
    for key in ("anchor", "A", "B", "C", "D", "E"):
        block = [r for r in rows if r["panel"] == key]
        if not block:
            continue
        if key == "E":
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\textit{" + PANEL_TITLES[key]
                     + r"}} \\")
        lines.append(" & ".join([""] + PANEL_HEADS[key]) + r" \\")
        for r in block:
            cells = [r[f"v{q+1}"] for q in range(6)]
            lines.append(" & ".join([r"\hspace{1em}" + r["label"]] + cells)
                         + r" \\")
    body = "\n".join(lines)
    n = f"{len(inp['w']):,}"
    ncl = f"{int(np.unique(inp['cl']).size):,}"
    path.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{Rotemberg Weights with County Exposures Frozen Before the Sample
($h={H}$)}}
\\label{{tab:gpss_frozen_rotemberg_h{H}}}
{{\\centering
\\adjustbox{{max width=\\linewidth}}{{%
\\begin{{tabular}}{{l*{{6}}{{r}}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} \\textbf{{This table is descriptive.}}
It decomposes an estimate already made and is not a test of the design. The
decomposed object is the frozen-share instrument $z^{{0}}$: county-category
exposure is mean real contract dollars over 1970--1974 divided by county
population in 1975, so only the national category shocks vary over time. The
locked production instrument $z$, whose exposure window moves with the target
year, is unchanged and is reported in the first panel on the same
{n} county-years and {ncl} county clusters, at a horizon of {H} years, with
division-by-year effects, division-equal weights, county clusters, and the
treatment winsorized within year. Following Goldsmith-Pinkham, Sorkin and
Swift, $\\hat\\beta=\\sum_{{k,t}}\\hat\\alpha_{{kt}}\\hat\\beta_{{kt}}$ with
weights aggregated to the category level as in their Section III.C; the
identity is asserted at run time and holds exactly here because the national
category rates are not leave-one-out. Their Section III.B normalization is
neither applied nor needed: exposure shares are dollars per head and do not sum
to one within a county. Coefficients are scaled by {locked.SCALE:g}. The
interval in the top-five panel inverts the cluster-robust Anderson-Rubin
statistic and is reported as what it is, including unbounded; the dollar share
is the category's share of national contract dollars, times 100. The mean
$\\hat\\beta_k$ by weight sign covers every category regardless of first-stage
power and is dominated by weakly identified ones, so the
$\\hat\\alpha$-weighted sums are the interpretable statistics. Two intervals
are reported for each of the top five because they disagree, and the
disagreement is the finding: the conventional interval is county-clustered and
symmetric around $\\hat\\beta_k$, while the AR set inverts the cluster-robust
Anderson-Rubin statistic and is reported as what it is, including unbounded. An
unbounded AR set alongside a large $\\hat F_k$ is not a contradiction. A single
procurement category's exposure is concentrated in a few counties, so the
cluster-level scores are heavy-tailed and the identification-robust set stays
wide even where the conventional first stage looks strong. The same pattern
holds for the locked instrument $z$.\\par}}
\\end{{table}}
""")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    args = ap.parse_args()

    gpss = load_producer("21_rotemberg.py", "rotemberg_producer")
    fixed = load_producer("21_fixed_initial_share_balance.py",
                          "fixed_share_producer")
    OUT.mkdir(parents=True, exist_ok=True)

    printed, weights = [], []
    for H in args.horizons:
        print(f"h={H}")
        inp = gpss.build_inputs(H)
        S_unit = fixed.fixed_exposures(inp)
        G_raw, G = national_matrix(inp)
        z = frozen_instrument(inp, S_unit, G)
        fit = op.t28.fit2sls(inp["y0"], inp["xw"], z, inp["w"], inp["fe"],
                             inp["nofe"], inp["cl"], singleton=False)
        rot = frozen_rotemberg(inp, S_unit, G, G_raw)
        verify(inp, fit, rot, S_unit, G)

        labels = gpss.fsc_labels(set(inp["cats"]))
        rows, top = panel_rows(inp, fit, rot, S_unit, G, gpss, labels)
        write_table(H, rows, inp,
                    OUT / f"Table_gpss_frozen_rotemberg_h{H}.tex")
        printed.extend(dict(h=H, **r) for r in rows)
        weights.append(pd.DataFrame({
            "h": H,
            "cat": inp["cats"],
            "description": [labels.get(c, "") for c in inp["cats"]],
            "alpha_k": rot["alpha"],
            "beta_k_scaled": rot["beta_k"] * locked.SCALE,
            "g_k": rot["g_k"],
            "F_k": rot["Fk"],
            "var_z_k": rot["vz_k"],
            "dollar_share": rot["dollars"] / rot["dollars"].sum(),
        }))
        a = rot["alpha"]
        sh = np.abs(a) / np.abs(a).sum()
        print(f"  beta z0 = {fit['beta']*locked.SCALE:.6f} "
              f"(se {fit['se']*locked.SCALE:.6f}), F = {fit['F']:.2f}")
        print(f"  locked z = {inp['base']['beta']*locked.SCALE:.6f} "
              f"(se {inp['base']['se']*locked.SCALE:.6f}), "
              f"F = {inp['base']['F']:.2f}")
        print(f"  top-1 share of |alpha| = {sh.max():.4f}, top-{TOP_N} = "
              f"{np.sort(sh)[::-1][:TOP_N].sum():.4f}, effective = "
              f"{hhi_effective(a):.2f}")

    pd.DataFrame(printed).to_csv(OUT / "gpss_frozen_rotemberg.csv", index=False)
    pd.concat(weights, ignore_index=True).to_csv(
        OUT / "gpss_frozen_rotemberg_weights.csv", index=False)
    print(f"  wrote {OUT / 'gpss_frozen_rotemberg.csv'}")
    print(f"  wrote {OUT / 'gpss_frozen_rotemberg_weights.csv'}")


if __name__ == "__main__":
    main()
