"""GPSS fixed-initial-share balance diagnostic for the h=2 design.

The county-category exposures are fixed at their 1970-1974 mean real contract
dollars per 1975 resident. The national category shocks retain their observed
time variation. Rotemberg weights and the five focal categories are computed
for this fixed-share instrument, then the fixed exposures and the complete
instrument are related to county characteristics measured in 1975.

Writes to Results/pipeline_diagnostics/:

  Table_gpss_fixed_initial_shares_h2.tex
  gpss_fixed_initial_shares_h2.csv
  gpss_fixed_initial_shares_h2_weights.csv
"""
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
from clusterstats import wls_fe  # noqa: E402
from sectors import SECTORS, naics2_shares  # noqa: E402

H = 2
SHARE_FIRST = 1970
SHARE_LAST = 1974
CHAR_YEAR = 1975
TOP_N = 5
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"


def load_rotemberg_module():
    path = PIPELINE / "21_rotemberg.py"
    spec = importlib.util.spec_from_file_location("rotemberg_producer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_exposures(inp):
    """County-category exposure fixed before the outcome sample begins."""
    a = inp["cy"].index(SHARE_FIRST)
    b = inp["cy"].index(SHARE_LAST)
    P0 = inp["piv"]["P"][CHAR_YEAR].reindex(inp["units"]).to_numpy(np.float64)
    if not np.isfinite(P0).all() or np.any(P0 <= 0):
        raise SystemExit("FAIL: 1975 population is missing or nonpositive")
    dollars = inp["arr"][:, :, a:b + 1].mean(axis=2).astype(np.float64)
    return dollars / P0[:, None]


def fixed_design(inp, S_unit):
    """Construct the fixed-share instrument and its Rotemberg decomposition."""
    target_pos = [inp["cy"].index(y) for y in inp["years"]]
    G = inp["gk"][:, target_pos].T
    G = np.where(np.isfinite(G), G, 0.0)
    z = np.full(len(inp["w"]), np.nan)
    for j in range(len(inp["years"])):
        m = inp["cc"] == j
        z[m] = S_unit[inp["rr"][m], :] @ G[j, :]
    if not np.isfinite(z).all():
        raise SystemExit("FAIL: fixed-share instrument has missing values")
    fit = op.t28.fit2sls(inp["y0"], inp["xw"], z, inp["w"], inp["fe"],
                         inp["nofe"], inp["cl"], singleton=False)

    M = op.w22.demean(np.column_stack([inp["y0"], inp["xw"], z]),
                      inp["fe"], inp["nofe"], inp["w"])
    yt, xt = M[:, 0], M[:, 1]
    K, T = len(inp["cats"]), len(inp["years"])
    NX = np.full((K, T), np.nan)
    NY = np.full((K, T), np.nan)
    for j in range(T):
        m = inp["cc"] == j
        Sj = S_unit[inp["rr"][m], :]
        wsk = Sj * inp["w"][m, None]
        NX[:, j] = wsk.T @ xt[m]
        NY[:, j] = wsk.T @ yt[m]
    gnx = G.T * NX
    gny = G.T * NY
    den = float(np.nansum(gnx))
    if abs(den) < 1e-300:
        raise SystemExit("FAIL: fixed-share first-stage covariance is zero")
    A = np.where(np.isfinite(gnx), gnx / den, 0.0)
    alpha = A.sum(axis=1)
    dk = np.nansum(gnx, axis=1)
    beta_k = np.full(len(dk), np.nan)
    np.divide(np.nansum(gny, axis=1), dk, out=beta_k, where=dk != 0)
    decomposition = float(np.nansum(alpha * beta_k))
    gap = abs(decomposition - fit["beta"]) / max(abs(fit["beta"]), 1e-300)
    if abs(alpha.sum() - 1.0) > 1e-10 or gap >= locked.RTOL_COEF:
        raise SystemExit(
            f"FAIL: fixed-share Rotemberg identity, sum alpha={alpha.sum():.12f}, "
            f"relative gap={gap:.3e}"
        )
    return dict(z=z, fit=fit, alpha=alpha, beta_k=beta_k, A=A, gap=gap)


def initial_characteristics(inp):
    """County characteristics in the common initial year, 1975."""
    units = inp["units"]
    P = inp["piv"]["P"][CHAR_YEAR].reindex(units).to_numpy(np.float64)
    E = inp["piv"]["E"][CHAR_YEAR].reindex(units).to_numpy(np.float64)
    Y = inp["piv"]["Y"][CHAR_YEAR].reindex(units).to_numpy(np.float64)
    shares = naics2_shares([CHAR_YEAR], units)
    shares = (pd.DataFrame({"ccg": units, "base_year": CHAR_YEAR})
              .merge(shares, on=["ccg", "base_year"], how="left"))
    chars = {
        "Income per head": Y / P,
        "Employment rate": E / P,
        "Log population": np.log(P),
    }
    for name in SECTORS:
        chars[name] = shares[name].to_numpy(np.float64)
    names = list(chars)
    X = np.column_stack([chars[name] for name in names])
    return names, X


def balance(inp, S_unit, fixed, top, gpss):
    """GPSS Table 2 analogue on fixed 1970-1974 exposures."""
    names, X_unit = initial_characteristics(inp)
    X = X_unit[inp["rr"], :]
    good = np.isfinite(X).all(axis=1)
    if not good.all():
        raise SystemExit(f"FAIL: {int((~good).sum()):,} rows lack 1975 covariates")
    for q in range(X.shape[1]):
        scale = gpss.wsd(X[:, q], inp["w"])
        if scale <= 0:
            raise SystemExit(f"FAIL: zero variance in {names[q]}")
        X[:, q] /= scale

    cols = [(inp["cats"][k], S_unit[inp["rr"], k]) for k in top]
    cols.append(("Instrument $z^{0}$", fixed["z"]))
    out = []
    for label, lhs in cols:
        scale = gpss.wsd(lhs, inp["w"])
        if scale <= 0:
            raise SystemExit(f"FAIL: zero variance in {label}")
        b, se, r2 = wls_fe(lhs / scale, X, inp["w"], inp["fe"],
                           inp["nofe"], inp["cl"])
        out.append(dict(label=label, b=b, se=se, r2=r2))
    return names, out


def write_outputs(inp, fixed, top, names, results, labels):
    OUT.mkdir(parents=True, exist_ok=True)
    headers = [gpss_tex(labels.get(inp["cats"][k], inp["cats"][k])) for k in top]
    headers.append("$z^{0}$")

    def row(label, values):
        return " & ".join([label] + list(values)) + r" \\"

    lines = [row("", headers), rf"\cmidrule(lr){{2-{len(headers)+1}}}"]
    for p, name in enumerate(names):
        lines.append(row(r"\hspace{1em}" + name,
                         [f"{r['b'][p]:.3f}" for r in results]))
        lines.append(row("", [f"({r['se'][p]:.3f})" for r in results]))
    lines.append(row(r"\hspace{1em}$R^2$",
                     [f"{r['r2']:.3f}" for r in results]))
    lines.append(row(r"\hspace{1em}Observations",
                     [f"{len(inp['w']):,}"] * len(results)))
    lines.append(row(r"\hspace{1em}County clusters",
                     [f"{np.unique(inp['cl']).size:,}"] * len(results)))
    body = "\n".join(lines)

    tex = OUT / "Table_gpss_fixed_initial_shares_h2.tex"
    tex.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{GPSS Correlates Test with Fixed Initial County Exposures}}
\\label{{tab:gpss_fixed_initial_shares_h2}}
{{\\centering
\\adjustbox{{max width=\\linewidth}}{{%
\\begin{{tabular}}{{l*{{6}}{{r}}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} This is the Table 2 diagnostic in
Goldsmith-Pinkham, Sorkin and Swift applied to a fixed-share version of the
$h=2$ county design. County-category exposures are mean real contract dollars
over 1970--1974 divided by county population in 1975. National category shocks
retain their observed time variation. The five categories are selected by the
largest Rotemberg weights recomputed for that fixed-share instrument. County
characteristics are fixed at 1975. Each column is one regression with
division-by-year effects, division-equal weights, and county-clustered standard
errors on 113,146 county-years from 3,058 counties. Both the dependent variable
and each characteristic are standardized.
The employment data begin in 1975, so a pre-exposure employment trend ending
before 1970 cannot be constructed.\\par}}
\\end{{table}}
""")

    records = []
    for p, name in enumerate(names):
        records.append(dict(statistic=name, **{
            f"v{q+1}": f"{r['b'][p]:.6f} ({r['se'][p]:.6f})"
            for q, r in enumerate(results)
        }))
    records.append(dict(statistic="R2", **{
        f"v{q+1}": f"{r['r2']:.6f}" for q, r in enumerate(results)
    }))
    records.append(dict(statistic="County clusters", **{
        f"v{q+1}": f"{np.unique(inp['cl']).size}"
        for q, r in enumerate(results)
    }))
    records.append(dict(statistic="column", **{
        f"v{q+1}": r["label"] for q, r in enumerate(results)
    }))
    pd.DataFrame(records).to_csv(
        OUT / "gpss_fixed_initial_shares_h2.csv", index=False)

    pd.DataFrame({
        "cat": inp["cats"],
        "description": [labels.get(c, "") for c in inp["cats"]],
        "alpha_k": fixed["alpha"],
        "beta_k_scaled": fixed["beta_k"] * locked.SCALE,
    }).to_csv(OUT / "gpss_fixed_initial_shares_h2_weights.csv", index=False)
    print(f"  wrote {tex}")


def gpss_tex(value):
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def main():
    gpss = load_rotemberg_module()
    inp = gpss.build_inputs(H)
    S_unit = fixed_exposures(inp)
    fixed = fixed_design(inp, S_unit)
    top = np.argsort(-fixed["alpha"])[:TOP_N]
    labels = gpss.fsc_labels({inp["cats"][k] for k in top})
    names, results = balance(inp, S_unit, fixed, top, gpss)
    write_outputs(inp, fixed, top, names, results, labels)
    print(f"  beta fixed-share = {fixed['fit']['beta']*locked.SCALE:.6f}, "
          f"SE = {fixed['fit']['se']*locked.SCALE:.6f}")
    print(f"  Rotemberg sum = {fixed['alpha'].sum():.12f}, "
          f"relative decomposition gap = {fixed['gap']:.2e}")
    for r in results:
        tmax = float(np.max(np.abs(r["b"] / r["se"])))
        print(f"  {r['label']}: max |t| = {tmax:.3f}, R2 = {r['r2']:.3f}")


if __name__ == "__main__":
    main()
