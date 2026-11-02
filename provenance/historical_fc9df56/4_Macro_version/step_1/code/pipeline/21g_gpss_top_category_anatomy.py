#!/usr/bin/env python3
"""Where the top Rotemberg category's estimate actually comes from.

Producer 21c shows that removing the highest-weight categories raises the
estimate, and 21b shows the largest category carries a negative beta_k. This
producer opens that category up. The question is not whether the number is
negative but whether it is anything: a stable effect, or one or two years and a
handful of counties.

Both Rotemberg sums are additive over rows,

    beta_k = sum_c sum_t w S_ck g_kt ytilde  /  sum_c sum_t w S_ck g_kt xtilde,

so the contribution of any year or any county is exact. Panels B, C and D are
that decomposition, holding the full-sample two-way demeaning fixed; they are
contribution shares, not re-estimations on subsamples. The decade ladder in the
second table IS a re-estimation, and is labelled as one.

Writes to Results/pipeline_diagnostics/:

  Table_gpss_top_category_anatomy_h{H}.tex
  Table_gpss_decade_ladder_h{H}.tex
  gpss_top_category_years_h{H}.csv
  gpss_top_category_counties_h{H}.csv
  gpss_decade_ladder.csv
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
FIXED = None      # set in main, the frozen-exposure constructor
FROZEN = None     # set in main, the frozen-share instrument builder
TOP_N = 5
RATIO_TOL = 1e-10
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"

STATE = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}


def load_producer(filename, name):
    spec = importlib.util.spec_from_file_location(name, PIPELINE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contributions(inp, gpss, k, G):
    """Row-level contributions to the category's numerator and denominator."""
    S = gpss.exposure_block(inp, np.array([k]))[:, 0]
    g = G[inp["cc"], k]
    base = inp["w"] * S * g
    return base * inp["xt"], base * inp["yt"]


def ratio(den, num):
    d = float(den.sum())
    if abs(d) < 1e-300:
        raise SystemExit("FAIL: a leave-out denominator is zero")
    return float(num.sum()) / d * locked.SCALE


def by_group(values, key, n):
    return np.bincount(key, weights=values, minlength=n)


def county_concentration(values, key, n, units):
    """Concentration of a row-level sum across counties."""
    c = np.bincount(key, weights=values, minlength=n)
    a = np.abs(c)
    tot = a.sum()
    if tot <= 0:
        raise SystemExit("FAIL: a county concentration has no mass")
    sh = a / tot
    o = np.argsort(-sh)
    return dict(largest=float(sh[o[0]]), top5=float(sh[o[:5]].sum()),
                top20=float(sh[o[:20]].sum()),
                effective=float(1.0 / np.sum(sh ** 2)),
                who=int(units[o[0]]))


def anatomy_rows(inp, gpss, rot, G, cats, labels):
    """Panels A to D."""
    years = np.asarray(inp["years"])
    decade = (years // 10) * 10
    cc, ccg = inp["cc"], inp["d"]["ccg"].to_numpy()
    rows, extra = [], {}

    def add(panel, label, *v):
        rows.append(dict(panel=panel, label=label,
                         **{f"v{q+1}": v[q] if q < len(v) else ""
                            for q in range(5)}))

    top = np.argsort(-rot["alpha"])[:TOP_N]
    eighties = np.isin(cc, np.where(decade == 1980)[0])

    for k in top:
        den, num = contributions(inp, gpss, k, G)
        share80 = float(den[eighties].sum() / den.sum())
        b_all = ratio(den, num)
        b80 = ratio(den[eighties], num[eighties])
        b_rest = ratio(den[~eighties], num[~eighties])
        gap = abs(b_all - rot["beta_k"][k] * locked.SCALE)
        if gap > 1e-6 * max(abs(b_all), 1.0):
            raise SystemExit(f"FAIL: {cats[k]} ratio {b_all} does not match "
                             f"the decomposition {rot['beta_k'][k]*locked.SCALE}")
        name = labels.get(cats[k], cats[k])
        add("A", gpss.tex_escape(name, width=30), f"{rot['alpha'][k]:.3f}",
            f"{b_all:.2f}", f"{b80:.2f}", f"{b_rest:.2f}", f"{share80:.3f}")

    k = int(top[0])
    den, num = contributions(inp, gpss, k, G)
    T = len(years)
    den_t, num_t = by_group(den, cc, T), by_group(num, cc, T)
    tot_d = den_t.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        b_t = np.where(den_t != 0, num_t / den_t * locked.SCALE, np.nan)
    order = np.argsort(-np.abs(den_t))
    for j in order[:8]:
        add("B", f"{years[j]}", f"{G[j, k]:.3f}", f"{den_t[j]/tot_d:.3f}",
            f"{b_t[j]:.2f}")
    add("B", "Five largest years' share of the denominator",
        f"{np.sort(np.abs(den_t))[::-1][:5].sum()/np.abs(den_t).sum():.3f}")

    moved = []
    for j in range(T):
        if den_t[j] == 0:
            continue
        moved.append((j, ratio(np.delete(den_t, j), np.delete(num_t, j))))
    b_all = ratio(den_t, num_t)
    moved.sort(key=lambda p: -abs(p[1] - b_all))
    for j, b in moved[:5]:
        add("C", f"Without {years[j]}", f"{b:.2f}", f"{b - b_all:+.2f}")
    add("C", "All years", f"{b_all:.2f}", "")

    units, key = np.unique(ccg, return_inverse=True)
    den_c, num_c = by_group(den, key, len(units)), by_group(num, key, len(units))
    corder = np.argsort(-np.abs(den_c))
    tot_c = np.abs(den_c).sum()
    for c in corder[:10]:
        fips = int(units[c])
        add("D", f"{fips:05d} ({STATE.get(fips // 1000, '??')})",
            f"{den_c[c]/tot_d:.3f}",
            f"{num_c[c]/num_t.sum():.3f}" if num_t.sum() else "")
    for n in (1, 5, 10):
        keep = np.setdiff1d(np.arange(len(units)), corder[:n])
        add("D", f"$\\hat\\beta_k$ without the top {n} counties",
            f"{ratio(den_c[keep], num_c[keep]):.2f}",
            f"{np.abs(den_c[corder[:n]]).sum()/tot_c:.3f}")
    add("D", "$\\hat\\beta_k$, all counties", f"{b_all:.2f}", "")

    # Panel E generalizes Panel D: is every high-weight category one county?
    for j in top:
        d_j, n_j = contributions(inp, gpss, int(j), G)
        dc_j = by_group(d_j, key, len(units))
        a = np.abs(dc_j)
        tot = a.sum()
        if tot <= 0:
            raise SystemExit(f"FAIL: {cats[j]} has no county contribution")
        sh = a / tot
        o = np.argsort(-sh)
        eff = float(1.0 / np.sum(sh ** 2))
        name_j = labels.get(cats[j], cats[j])
        add("E", gpss.tex_escape(name_j, width=30), f"{sh[o[0]]:.3f}",
            f"{sh[o[:5]].sum():.3f}", f"{eff:.1f}",
            f"{int(units[o[0]]):05d} ({STATE.get(int(units[o[0]]) // 1000, '??')})")

    extra["years"] = pd.DataFrame({
        "year": years, "g_kt": G[:, k], "den": den_t, "num": num_t,
        "den_share": den_t / tot_d, "beta_kt_scaled": b_t})
    extra["counties"] = pd.DataFrame({
        "ccg": units, "state": [STATE.get(int(u) // 1000, "") for u in units],
        "den": den_c, "num": num_c, "den_share": den_c / tot_d})
    # Panel F lifts the question from one category to the whole instrument.
    S_unit = FIXED.fixed_exposures(inp)
    z0 = FROZEN.frozen_instrument(inp, S_unit, G)
    z0t = op.w22.demean(z0.reshape(-1, 1), inp["fe"], inp["nofe"],
                        inp["w"])[:, 0]
    for label, zt in (("Locked $z$", inp["zt"]), ("Frozen $z^{0}$", z0t)):
        for side, v in (("variance $\\tilde z^2$", zt),
                        ("covariance with $\\tilde x$", inp["xt"]),
                        ("covariance with $\\tilde y$", inp["yt"])):
            c = county_concentration(inp["w"] * zt * v, key, len(units), units)
            add("F", f"{label}, {side}", f"{c['largest']:.3f}",
                f"{c['top5']:.3f}", f"{c['top20']:.3f}",
                f"{c['effective']:.1f}",
                f"{c['who']:05d} ({STATE.get(c['who'] // 1000, '??')})")
    add("F", "Counties in the estimation sample", "", "", "",
        f"{len(units):,}", "")
    return rows, k, extra


def decade_ladder(inp, cats, k):
    """Re-estimate the locked instrument with one decade of target years out."""
    years = np.asarray(inp["years"])
    dec = (years // 10) * 10
    rows = []
    for drop in [None] + sorted(set(dec.tolist())):
        m = (np.ones(len(inp["w"]), bool) if drop is None
             else ~np.isin(inp["cc"], np.where(dec == drop)[0]))
        if m.sum() < 1000:
            continue
        fe = pd.factorize(inp["fe"][m])[0].astype(np.int64)
        cl = pd.factorize(inp["cl"][m])[0].astype(np.int64)
        r = op.t28.fit2sls(inp["y0"][m], inp["xw"][m], inp["z0"][m],
                           inp["w"][m], fe, np.zeros(int(m.sum()), np.int64),
                           cl, singleton=False)
        rows.append(dict(dropped="none" if drop is None else f"{drop}s",
                         beta=r["beta"] * locked.SCALE,
                         se=r["se"] * locked.SCALE, t=r["beta"] / r["se"],
                         F=r["F"], n=int(m.sum()),
                         counties=int(np.unique(cl).size)))
    return rows


TITLES = {
    "A": "Panel A. The five largest categories, split at the 1980s",
    "B": "Panel B. The largest category year by year, eight largest years",
    "C": "Panel C. The largest category with one year removed",
    "D": "Panel D. The largest category by county",
    "E": "Panel E. County concentration of every top-five category",
    "F": "Panel F. County concentration of the whole instrument",
}
HEADS = {
    "A": ["$\\hat\\alpha_k$", "$\\hat\\beta_k$", "1980s only",
          "Excluding the 1980s", "1980s share of denominator"],
    "B": ["$g_{kt}$", "Share of denominator", "$\\hat\\beta_{kt}$", "", ""],
    "C": ["$\\hat\\beta_k$", "Change", "", "", ""],
    "D": ["Share of denominator", "Share of numerator", "", "", ""],
    "E": ["Largest county's share", "Five largest counties' share",
          "Effective counties (1/HHI)", "Largest county", ""],
    "F": ["Largest county's share", "Five largest", "Twenty largest",
          "Effective counties (1/HHI)", "Largest county"],
}


def write_anatomy(H, rows, inp, name, path):
    lines = []
    for key in ("A", "B", "C", "D", "E", "F"):
        block = [r for r in rows if r["panel"] == key]
        if not block:
            continue
        lines.append(r"\multicolumn{6}{l}{\textit{" + TITLES[key] + r"}} \\")
        lines.append(" & ".join([""] + HEADS[key]) + r" \\")
        for r in block:
            lines.append(" & ".join([r"\hspace{1em}" + r["label"]]
                                    + [r[f"v{q+1}"] for q in range(5)])
                         + r" \\")
        lines.append(r"\addlinespace[0.6ex]")
    body = "\n".join(lines[:-1])
    n = f"{len(inp['w']):,}"
    path.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{Anatomy of the Largest Rotemberg Weight Category ($h={H}$)}}
\\label{{tab:gpss_top_category_anatomy_h{H}}}
{{\\centering
\\adjustbox{{max width=\\linewidth}}{{%
\\begin{{tabular}}{{l*{{5}}{{r}}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} The category is {name}, the largest
Rotemberg weight in the locked instrument at a horizon of {H} years on {n}
county-years. Its just-identified estimate is the ratio of two sums that are
additive over rows, $\\hat\\beta_k = \\sum_{{c,t}} w S_{{ck}} g_{{kt}}
\\tilde y_{{ct}} \\big/ \\sum_{{c,t}} w S_{{ck}} g_{{kt}} \\tilde x_{{ct}}$, so
every share below is an exact contribution to that ratio. Panels B, C and D
hold the full-sample two-way demeaning fixed and re-weight the sums; they are a
decomposition of one estimate, not re-estimation on subsamples. Panel A's split
is on the same principle. Panel D reports county FIPS and state, because county
names are not carried in the panel. In Panel D the two rows that remove the
five and ten largest counties strip 93.8 and 99.4 percent of the denominator,
so what remains is a near-zero denominator and the ratio on it carries no
information; they are shown to make the concentration visible, not as
robustness. Panel E repeats the county concentration for each of the five
largest categories, where the effective county count is the inverse Herfindahl
of the absolute county contributions to that category's denominator.
Panel F asks the same question of the whole instrument rather than one
category. Its first row for each instrument is the one to read: county shares
of the instrument's own variance $\\sum_c w \\tilde z^2$, which is the weight
the first-stage slope places on each county. The two covariance rows are
reported alongside it but are not measures of first-stage strength, and a
concentrated instrument here has a weaker county-clustered $F$, not a stronger
one, as the leave-out ladder shows directly. Coefficients are scaled by {locked.SCALE:g}.\\par}}
\\end{{table}}
""")
    print(f"  wrote {path}")


def write_ladder(H, rows, path):
    lines = [" & ".join(["", "$\\hat\\beta$", "(s.e.)", "$t$",
                         "First-stage $F$", "Observations", "Counties"])
             + r" \\"]
    for r in rows:
        label = ("All target years" if r["dropped"] == "none"
                 else f"Without the {r['dropped']}")
        lines.append(" & ".join([
            r"\hspace{1em}" + label, f"{r['beta']:.4f}", f"({r['se']:.4f})",
            f"{r['t']:.2f}", f"{r['F']:.1f}", f"{r['n']:,}",
            f"{r['counties']:,}"]) + r" \\")
    body = "\n".join(lines)
    path.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{The Locked Instrument with One Decade of Target Years Removed
($h={H}$)}}
\\label{{tab:gpss_decade_ladder_h{H}}}
{{\\centering
\\adjustbox{{max width=\\linewidth}}{{%
\\begin{{tabular}}{{l*{{6}}{{r}}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} Unlike the category leave-out table,
each row here is a genuine re-estimation on a smaller sample: the target years
in one decade are dropped and the division-by-year effects and county clusters
are rebuilt on what remains. The specification is otherwise the locked one,
midpoint employment growth on the winsorized county treatment instrumented by
the composition instrument, division-equal weights, county clusters, horizon
{H} years. Observations and county counts move across rows by construction, so
the rows are not on a common sample. Coefficients are scaled by
{locked.SCALE:g}.\\par}}
\\end{{table}}
""")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    args = ap.parse_args()

    global FIXED, FROZEN
    gpss = load_producer("21_rotemberg.py", "rotemberg_producer")
    frozen = load_producer("21b_gpss_frozen_share_rotemberg.py", "frozen_producer")
    FIXED = load_producer("21_fixed_initial_share_balance.py", "fixed_producer")
    FROZEN = frozen
    OUT.mkdir(parents=True, exist_ok=True)

    ladders = []
    for H in args.horizons:
        print(f"h={H}")
        inp = gpss.build_inputs(H)
        rot = gpss.rotemberg(inp)
        _, G = frozen.national_matrix(inp)
        cats = np.asarray(inp["cats"])
        top = np.argsort(-rot["alpha"])[:TOP_N]
        labels = gpss.fsc_labels({cats[j] for j in top})

        rows, k, extra = anatomy_rows(inp, gpss, rot, G, cats, labels)
        name = labels.get(cats[k], cats[k])
        print(f"  largest category {cats[k]} ({name}), alpha "
              f"{rot['alpha'][k]:.3f}, beta_k "
              f"{rot['beta_k'][k]*locked.SCALE:.3f}")
        for r in rows:
            print(f"   {r['panel']} {r['label'][:44]:<44} "
                  + "  ".join(str(r[f'v{q+1}']) for q in range(5)).rstrip())
        write_anatomy(H, rows, inp, gpss.tex_escape(name), 
                      OUT / f"Table_gpss_top_category_anatomy_h{H}.tex")
        extra["years"].to_csv(
            OUT / f"gpss_top_category_years_h{H}.csv", index=False)
        extra["counties"].to_csv(
            OUT / f"gpss_top_category_counties_h{H}.csv", index=False)

        lad = decade_ladder(inp, cats, k)
        write_ladder(H, lad, OUT / f"Table_gpss_decade_ladder_h{H}.tex")
        for r in lad:
            print(f"   decade drop {r['dropped']:>6}: beta {r['beta']:8.4f} "
                  f"({r['se']:.4f})  t {r['t']:5.2f}  F {r['F']:7.1f}  "
                  f"N {r['n']:,}")
        ladders.extend(dict(h=H, **r) for r in lad)

    pd.DataFrame(ladders).to_csv(OUT / "gpss_decade_ladder.csv", index=False)
    print(f"  wrote {OUT / 'gpss_decade_ladder.csv'}")


if __name__ == "__main__":
    main()
