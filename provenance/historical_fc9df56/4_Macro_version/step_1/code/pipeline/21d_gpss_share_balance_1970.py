#!/usr/bin/env python3
"""GPSS Table 2 balance for the shares, against the full 1970 Census vector.

The shares route identifies off county procurement composition, so the question
is whether that composition is correlated with predetermined county
characteristics that also predict employment growth. Producer 19 already builds
a fixed 1970 Census cross-section of sixteen covariates, and until now they are
used only at the shock level in producer 22b. This runs them at the share level,
which is the Goldsmith-Pinkham-Sorkin-Swift Table 2 exercise.

The covariate list is imported from 22b rather than restated, so the share-level
and shock-level balance tables cannot drift apart. Producer 19 writes seventeen
covariates and 22b selects sixteen; that selection is inherited here.

Each column is one regression of a standardized left-hand variable on the
sixteen standardized covariates, with division-by-year effects, division-equal
weights and county clusters, so a coefficient is standard deviations per
standard deviation. The joint test is the CRV1 Wald that all sixteen are zero,
computed by 21_rotemberg.joint_balance_wald.

TIMING. County exposures are frozen at mean 1970-1974 procurement dollars per
1975 resident, and the characteristics are the 1970 Census. The characteristics
are therefore measured at the start of the exposure window rather than strictly
before it. That is the intended test: a county's 1970 industrial structure
plausibly determines its 1970-1974 procurement mix, and this exercise asks
whether it does. It is not a claim that the characteristics are predetermined
relative to every dollar in the window.

Writes to Results/pipeline_diagnostics/:

  Table_gpss_share_balance_1970_h2.tex, _h4.tex
  gpss_share_balance_1970.csv
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
import dcads_common as dc  # noqa: E402
from clusterstats import wls_fe  # noqa: E402

HORIZONS = (2, 4)
TOP_N = 5
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"


def load_producer(filename, name):
    spec = importlib.util.spec_from_file_location(name, PIPELINE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def characteristics(inp, bhj1970):
    """The sixteen fixed 1970 covariates on the estimation sample, standardized."""
    if not bhj1970.CHAR_PATH.exists():
        raise SystemExit(
            f"FAIL: {bhj1970.CHAR_PATH} is absent; run "
            "19_ingest_1970_county_characteristics.py")
    chars = pd.read_parquet(bhj1970.CHAR_PATH)
    chars["ccg"] = chars["ccg_id"].astype(int)
    if not chars["ccg"].is_unique:
        raise SystemExit("FAIL: the 1970 characteristic file is not unique by CCG")
    names = list(bhj1970.NAMES)
    key = pd.DataFrame({"ccg": inp["d"]["ccg"].astype(int).to_numpy()})
    merged = key.merge(chars[["ccg", *names]], on="ccg", how="left",
                       validate="m:1")
    if merged[names].isna().any().any():
        bad = int(merged[names].isna().any(axis=1).sum())
        raise SystemExit(f"FAIL: {bad:,} county-year rows lack 1970 characteristics")
    return names, np.column_stack(
        [merged[n].to_numpy(np.float64) for n in names])


def standardize(M, w, gpss, what):
    out = M.astype(np.float64).copy()
    for q in range(out.shape[1]):
        s = gpss.wsd(out[:, q], w)
        if s <= 0:
            raise SystemExit(f"FAIL: zero weighted variance in {what} column {q}")
        out[:, q] = out[:, q] / s
    return out


def left_hand_columns(inp, gpss, frozen, fixed, G_raw, G, cats):
    """The five top-weight frozen shares, the frozen instrument, the locked one."""
    S_unit = fixed.fixed_exposures(inp)
    rot = frozen.frozen_rotemberg(inp, S_unit, G, G_raw)
    top = np.argsort(-rot["alpha"])[:TOP_N]
    labels = gpss.fsc_labels({cats[k] for k in top})
    z0 = frozen.frozen_instrument(inp, S_unit, G)
    # The instrument columns are the test of the design and come first. The
    # per-category columns follow as description: a shift-share design is not
    # defended one share at a time, and these are here only because the
    # Rotemberg weights are concentrated enough that a few shares carry much of
    # the estimate.
    cols = [("Locked $z$", inp["z0"]), ("Frozen $z^{0}$", z0)]
    cols += [(gpss.tex_escape(labels.get(cats[k], cats[k]), width=15),
              S_unit[inp["rr"], k]) for k in top]
    return cols, top, rot


def effective_counties(v, inp, key, nunits):
    """Effective counties in a left-hand variable's own within-cell variance.

    A balance test on a variable whose variation sits on two counties has no
    power, so a small joint F there is not evidence of balance. This column is
    what separates the two readings.
    """
    import oldpath as op
    vt = op.w22.demean(v.reshape(-1, 1), inp["fe"], inp["nofe"],
                       inp["w"])[:, 0]
    var = np.bincount(key, weights=inp["w"] * vt * vt, minlength=nunits)
    tot = var.sum()
    if tot <= 0:
        raise SystemExit("FAIL: a left-hand column has no within-cell variance")
    sh = var / tot
    return float(1.0 / np.sum(sh ** 2))


def run(inp, gpss, cols, names, X):
    """One regression per left-hand column, plus the joint Wald."""
    w, fe, nofe, cl = inp["w"], inp["fe"], inp["nofe"], inp["cl"]
    units, key = np.unique(inp["d"]["ccg"].to_numpy(), return_inverse=True)
    Xs = standardize(X, w, gpss, "characteristics")
    out = []
    for label, raw in cols:
        s = gpss.wsd(raw, w)
        if s <= 0:
            raise SystemExit(f"FAIL: zero weighted variance in {label}")
        y = raw / s
        b, se, r2 = wls_fe(y, Xs, w, fe, nofe, cl)
        bj, _, F, pv, q, ncl = gpss.joint_balance_wald(y, Xs, w, fe, nofe, cl)
        gap = float(np.max(np.abs(b - bj)))
        if gap > 1e-9 * max(float(np.abs(b).max()), 1.0):
            raise SystemExit(
                f"FAIL: {label} coefficients disagree between wls_fe and the "
                f"joint Wald by {gap:.2e}")
        rej = int(np.sum(np.abs(b / se) > 1.96))
        eff = effective_counties(raw, inp, key, len(units))
        out.append(dict(label=label, b=b, se=se, r2=r2, F=F, p=pv,
                        rejections=rej, q=q, clusters=ncl, effective=eff))
        print(f"   {label[:24]:<24} joint F {F:6.2f} [p {pv:.3f}]  "
              f"{rej:>2} of {len(names)} significant  R2 {r2:.3f}  "
              f"effective counties {eff:8.1f}")
    return out


def write_table(H, names, res, inp, path):
    heads = [r["label"] for r in res]
    ncat = len(heads) - 2
    lines = [
        " & ".join(["", r"\multicolumn{2}{c}{\textit{The design}}",
                    rf"\multicolumn{{{ncat}}}{{c}}{{\textit{{Highest-weight "
                    rf"categories, descriptive}}}}"]) + r" \\",
        r"\cmidrule(lr){2-3}" + rf"\cmidrule(lr){{4-{len(heads) + 1}}}",
        " & ".join([""] + heads) + r" \\",
        rf"\cmidrule(lr){{2-{len(heads) + 1}}}"]
    for q, name in enumerate(names):
        lines.append(" & ".join([r"\hspace{1em}" + name]
                                + [f"{r['b'][q]:.3f}" for r in res]) + r" \\")
        lines.append(" & ".join([""]
                                + [f"({r['se'][q]:.3f})" for r in res]) + r" \\")
    lines.append(r"\midrule")
    for lab, key, fmt in (("Joint $F$, all sixteen", "F", "{:.2f}"),
                          ("$p$-value", "p", "{:.3f}"),
                          ("Individually significant", "rejections", "{:d}"),
                          ("$R^2$", "r2", "{:.3f}"),
                          ("Effective counties in the column", "effective",
                           "{:.1f}")):
        lines.append(" & ".join([r"\hspace{1em}" + lab]
                                + [fmt.format(r[key]) for r in res]) + r" \\")
    lines.append(" & ".join([r"\hspace{1em}Observations"]
                            + [f"{len(inp['w']):,}"] * len(res)) + r" \\")
    lines.append(" & ".join([r"\hspace{1em}County clusters"]
                            + [f"{res[0]['clusters']:,}"] * len(res)) + r" \\")
    body = "\n".join(lines)
    ncol = len(heads)
    path.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{Share-Level Balance Against the 1970 Census Cross-Section
($h={H}$)}}
\\label{{tab:gpss_share_balance_1970_h{H}}}
{{\\centering
\\adjustbox{{max width=\\linewidth}}{{%
\\begin{{tabular}}{{l*{{{ncol}}}{{r}}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} This is the Goldsmith-Pinkham, Sorkin
and Swift Table 2 diagnostic run at the share level. Each column is one
regression of a standardized left-hand variable on the sixteen standardized
1970 Census covariates that producer 19 builds, with division-by-year effects,
division-equal weights and county-clustered standard errors, so a coefficient
is standard deviations per standard deviation. \\textbf{{The first two columns
are the test.}} A shift-share design is not defended one share at a time; the
identifying content is the instrument, which combines 800 categories, and those
two columns ask whether it is correlated with predetermined county
characteristics. The remaining columns are the county exposures in the five
categories carrying the largest Rotemberg weights, reported because those
weights are concentrated enough that a few shares carry much of the estimate,
and they are descriptive rather than a test of the design. County exposures behind $z^{{0}}$ and
the share columns are mean real contract dollars over 1970--1974 divided by
county population in 1975. The joint test is the CRV1 Wald that all sixteen
coefficients are zero. The covariate list is imported from producer 22b, which
selects sixteen of the seventeen characteristics producer 19 writes, so the
share-level and shock-level balance tables rest on one covariate basis; BHJ
footnote 37 warns against reading either as confirming the other, since they
test different assumptions at different levels. The characteristics are the
1970 Census cross-section and the exposure window runs 1970 to 1974, so the
covariates are measured at the start of that window rather than strictly before
it. \\textbf{{Read the effective-county row before the $F$ row.}} It is the
inverse Herfindahl of each column's own within-cell variance across counties. A
balance test on a variable whose variation sits on a couple of counties has
almost no power, so a small joint $F$ there is not evidence of balance and must
not be read as a pass. The five category columns are in that position; the two
instrument columns are not. \\textbf{{Inference caveat.}} The instrument's
variance is concentrated on an effective 10.5 counties at $h=2$, so
county-clustered standard errors on {res[0]['clusters']:,} clusters are
optimistic here for the same reason they are optimistic in the second stage,
which weakens a rejection as well as a failure to reject.\\par}}
\\end{{table}}
""")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    args = ap.parse_args()

    gpss = load_producer("21_rotemberg.py", "rotemberg_producer")
    frozen = load_producer("21b_gpss_frozen_share_rotemberg.py", "frozen_producer")
    fixed = load_producer("21_fixed_initial_share_balance.py", "fixed_producer")
    bhj1970 = load_producer("22b_bhj_1970_balance_h2.py", "bhj1970_producer")
    OUT.mkdir(parents=True, exist_ok=True)

    recs = []
    for H in args.horizons:
        print(f"h={H}")
        inp = gpss.build_inputs(H)
        G_raw, G = frozen.national_matrix(inp)
        cats = np.asarray(inp["cats"])
        cols, top, rot = left_hand_columns(inp, gpss, frozen, fixed, G_raw, G,
                                           cats)
        names, X = characteristics(inp, bhj1970)
        print(f"  {len(names)} covariates, {len(cols)} left-hand columns")
        res = run(inp, gpss, cols, names, X)
        write_table(H, names, res, inp,
                    OUT / f"Table_gpss_share_balance_1970_h{H}.tex")
        for r in res:
            for q, n in enumerate(names):
                recs.append(dict(h=H, column=r["label"], characteristic=n,
                                 coef=r["b"][q], se=r["se"][q]))
            recs.append(dict(h=H, column=r["label"], characteristic="JOINT_F",
                             coef=r["F"], se=r["p"]))
            recs.append(dict(h=H, column=r["label"],
                             characteristic="EFFECTIVE_COUNTIES",
                             coef=r["effective"], se=np.nan))
    pd.DataFrame(recs).to_csv(OUT / "gpss_share_balance_1970.csv", index=False)
    print(f"  wrote {OUT / 'gpss_share_balance_1970.csv'}")


if __name__ == "__main__":
    main()
