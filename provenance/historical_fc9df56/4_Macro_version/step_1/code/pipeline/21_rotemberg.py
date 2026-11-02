"""Stage 7: the GPSS Rotemberg decomposition of the composition instrument.

Goldsmith-Pinkham, Sorkin and Swift (2020) Proposition 3 on the FSC category
axis, aggregated across years by their Section III.C, plus their Section V.A
correlates exercise for the high-weight categories. Consumes ONLY the
persisted pipeline stores. Writes to Results/pipeline_diagnostics/:

  Table_rotemberg_fsc.tex             six panels:
                                      A negative/positive weights
                                      B correlations among alpha, g, beta, F,
                                        var(z)
                                      C variation across years, by decade
                                      D top five categories, with weak-IV
                                        robust intervals
                                      E beta_k by weight sign
                                      F correlates of the high-weight shares
  rotemberg_fsc_locked.csv            every printed row
  rotemberg_weights_by_category.csv   all NCAT categories, every horizon
  rotemberg_alpha_by_year.csv         alpha_kt, the by-year Panel C

What this decomposes. The locked cell's just-identified 2SLS: midpoint
employment growth on the winsorized treatment, instrumented by the composition
instrument, division-by-year effects, division-equal weights, county clusters.
That is pipeline 20's headline, and

    beta = sum_k sum_t alpha_kt beta_kt,    sum_kt alpha_kt = 1

holds exactly here because the national category rates are NOT leave-one-out
(GPSS online Appendix I). Both identities are asserted, and the adding-up
target comes from t28.fit2sls, the same estimator that produced the frozen
numbers - not from anything this file computes.

No normalization of the growth rates (GPSS Section III.B) is applied: the
exposure shares are dollars per head and do not sum to one within a county, so
the linear dependence their normalization repairs does not arise here. This is
the BHJ Section 4.2 incomplete-shares case that pipeline 20 already reports.

NOT the same object as pipeline 20's "first-stage share" rows, which measure
|sum_l w s x~| concentration without g_kt and without normalization. The two
sets of numbers are different statistics and are expected to differ.
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import harmonize  # noqa: E402
import panel as pnl_lib  # noqa: E402
import store  # noqa: E402
import oldpath as op  # noqa: E402  (estimator only: t28/w22, as 57 loads them)
import dcads_common as dc  # noqa: E402

TAB_DIR = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"

ALPHA_TOL = 1e-10       # |sum alpha - 1|; exact by construction, not fitted
FVEC_TOL = 1e-8         # vectorized F against the house estimator, relative
TOP_N = 5               # GPSS report the top five
CHI2_95 = 3.8414588206941236
FBLOCK = 80             # categories per first-stage block

# cluster_sums, hhi_effective and wls_fe were defined here until producer 22
# needed the same three; SECTORS, OMITTED and naics2_shares likewise. Both sets
# moved to lib/ verbatim and this file's tables were byte-compared across the
# move. ar_interval and fmt_ci stay below: WS0 runs no identification-robust
# set outside this producer's per-category GPSS intervals.
from clusterstats import cluster_sums, hhi_effective, wls_fe  # noqa: E402,F401
from sectors import NAICS2_DIR, OMITTED, SECTORS, naics2_shares  # noqa: E402,F401


# ---------------------------------------------------------------- helpers --


def wsd(v, w):
    """Weighted standard deviation on the estimation sample."""
    m = np.average(v, weights=w)
    return float(np.sqrt(np.average((v - m) ** 2, weights=w)))


def joint_balance_wald(y, X, w, fe, nofe, cl):
    """CRV1 Wald test that every characteristic coefficient is zero."""
    M = op.w22.demean(np.column_stack([y, X]), fe, nofe, w)
    yt, Xt = M[:, 0], M[:, 1:]
    XW = Xt * w[:, None]
    A = Xt.T @ XW
    b = np.linalg.solve(A, XW.T @ yt)
    u = yt - Xt @ b
    inv = np.linalg.inv(A)
    ncl = int(cl.max()) + 1
    scores = np.column_stack([
        cluster_sums(XW[:, q] * u, cl, ncl)
        for q in range(Xt.shape[1])
    ])
    n, G = len(y), int(np.unique(cl).size)
    k_ssc = op.t28.ssc_k(fe, nofe, cl) + Xt.shape[1] - 1
    ssc = (G / max(G - 1, 1)) * ((n - 1) / max(n - k_ssc, 1))
    V = ssc * (inv @ (scores.T @ scores) @ inv)
    q = Xt.shape[1]
    F = float(b @ np.linalg.solve(V, b) / q)
    p = float(stats.f.sf(F, q, max(G - 1, 1)))
    return b, V, F, p, q, G


def short_name(s, width=16):
    """A readable stub of a category description, cut on a word boundary."""
    s = s.replace("/", " / ")
    if len(s) <= width:
        return s
    out = []
    for word in s.split():
        if out and len(" ".join(out + [word])) > width:
            break
        out.append(word)
    while len(out) > 1 and out[-1].lower() in ("and", "of", "the", "&", "/"):
        out.pop()                       # never end a stub on a connector
    return " ".join(out) if out else s[:width]


def tex_escape(s, width=None):
    """Category descriptions are raw contract-file text and reach the table
    verbatim; an unescaped '&' in one of them silently becomes a column break."""
    cut = bool(width) and len(s) > width
    if cut:                               # truncate BEFORE escaping, so an
        s = s[:width].rstrip(" ,;:-/")    # escape sequence can never be cut
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s + r"\ldots" if cut else s




def ar_interval(bt, yt, xt, w, cl, ssc):
    """Weak-instrument-robust interval for beta_k, GPSS's Chernozhukov-Hansen
    analogue in the just-identified case.

    Inverts the cluster-robust Anderson-Rubin statistic. The acceptance region
    {b0 : (a - b0 b)^2 <= q ssc sum_c (A_c - b0 B_c)^2} is a quadratic
    inequality, so it is solved in closed form rather than on a grid. It can be
    an interval, empty, the whole line, or the complement of an interval; each
    is reported as what it is and never silently truncated.
    """
    ncl = int(cl.max()) + 1
    a = float(bt @ (w * yt))
    b = float(bt @ (w * xt))
    Ac = cluster_sums(w * bt * yt, cl, ncl)
    Bc = cluster_sums(w * bt * xt, cl, ncl)
    q = CHI2_95 * ssc
    P = b * b - q * float(Bc @ Bc)
    Q = -2.0 * (a * b - q * float(Ac @ Bc))
    R = a * a - q * float(Ac @ Ac)
    disc = Q * Q - 4.0 * P * R
    if abs(P) < 1e-300:
        return (-np.inf, np.inf)
    if disc < 0:
        return (np.nan, np.nan) if P > 0 else (-np.inf, np.inf)
    r = np.sqrt(disc)
    lo, hi = sorted(((-Q - r) / (2 * P), (-Q + r) / (2 * P)))
    return (lo, hi) if P > 0 else (-np.inf, np.inf)


def fmt_ci(lo, hi, scale):
    if not np.isfinite(lo) and not np.isfinite(hi):
        return "unbounded"
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "unbounded"
    if np.isnan(lo):
        return "empty"
    return f"({lo*scale:.2f}, {hi*scale:.2f})"


# ------------------------------------------------------------ input build --



def build_inputs(H):
    """The locked diagnostics cell at horizon H, from the persisted stores.

    Same sample, weights, winsorization and fixed effects as pipeline 20's
    shock-level table, extended with the predetermined characteristics Panel F
    regresses on.
    """
    pnl = store.load_county_panel()
    units, piv = store.load_pivots(pnl, cols=("P", "E", "Y"))
    DIVI = pnl.drop_duplicates("ccg").set_index("ccg")["division"]
    units2, cats, cy, arr, nat = store.load_tensor(locked.RULE)
    assert np.array_equal(units, units2), "unit order drifted between stores"

    years = store.target_years(H)
    P_b = piv["P"].reindex(columns=[y - H for y in years]).to_numpy(np.float64)
    E_b = piv["E"].reindex(columns=[y - H for y in years]).to_numpy(np.float64)
    E_t = piv["E"].reindex(columns=years).to_numpy(np.float64)
    E_2b = piv["E"].reindex(columns=[y - 2 * H for y in years]).to_numpy(np.float64)
    Y_b = piv["Y"].reindex(columns=[y - H for y in years]).to_numpy(np.float64)

    C = store.load_con(locked.RULE).reindex(units)
    lag = {k: C.reindex(columns=[y - k for y in years]).set_axis(years, axis=1)
           for k in range(H + 1)}
    G_b = lag[H].to_numpy(np.float64)
    G_w = (sum(lag[k] for k in range(H)) / H).to_numpy(np.float64)
    X = (G_w - G_b) / P_b
    den = E_t + E_b
    Yv = np.where(den != 0, 2.0 * (E_t - E_b) / den, np.nan)
    dpre = E_b + E_2b
    PRE = np.where(dpre != 0, 2.0 * (E_b - E_2b) / dpre, np.nan)

    gk = store.load_gk(H, cats, cy)
    Z = store.load_instrument(H, units, "z")

    # The exposure window per target year, closing at t - H - LAG.
    win = {}
    for j, t in enumerate(years):
        hi = t - H - locked.LAG
        lo = hi - locked.SPAN + 1
        if lo >= locked.DATA_FIRST:
            win[j] = (cy.index(lo), cy.index(hi))
    S0 = np.full_like(Z, np.nan)
    for j, (a, b) in win.items():
        S0[:, j] = (arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
                    .sum(axis=1) / P_b[:, j])

    ok = (np.isfinite(X) & np.isfinite(Yv) & np.isfinite(Z) & np.isfinite(S0)
          & np.isfinite(E_b))
    ri, ci = np.where(ok)
    d = pd.DataFrame(dict(ccg=units[ri], year=np.array(years)[ci],
                          x=X[ok], y=Yv[ok], z=Z[ok], E_b=E_b[ok],
                          P_b=P_b[ok], Y_b=Y_b[ok], pre=PRE[ok]))
    d["division"] = d["ccg"].map(DIVI)
    assert len(d) == locked.N_COUNTY[H], (H, len(d), locked.N_COUNTY[H])
    assert d["ccg"].nunique() == locked.N_COUNTIES, (H, d["ccg"].nunique())

    xw = pnl_lib.wins(d["x"].to_numpy(np.float64), d["year"].to_numpy())
    tot = d.groupby(["division", "year"])["E_b"].transform("sum")
    w = (d["E_b"] / tot).to_numpy(np.float64)
    w = w / w.mean()
    fe = pd.factorize(d["division"].astype(str) + "_"
                      + d["year"].astype(str))[0].astype(np.int64)
    nofe = np.zeros(len(d), np.int64)
    cl = pd.factorize(d["ccg"])[0].astype(np.int64)

    import pyfixest as pf
    op.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100_000)

    y0 = d["y"].to_numpy(np.float64)
    z0 = d["z"].to_numpy(np.float64)
    base = op.t28.fit2sls(y0, xw, z0, w, fe, nofe, cl, singleton=False)
    M = op.w22.demean(np.column_stack([y0, xw, z0]), fe, nofe, w)

    rowpos = {u: i for i, u in enumerate(units)}
    colpos = {y: j for j, y in enumerate(years)}
    return dict(d=d, units=units, cats=cats, cy=cy, arr=arr, gk=gk, H=H,
                piv=piv,
                years=years, win=win, base=base, w=w, fe=fe, nofe=nofe, cl=cl,
                y0=y0, xw=xw, z0=z0, yt=M[:, 0], xt=M[:, 1], zt=M[:, 2],
                rr=d["ccg"].map(rowpos).to_numpy(),
                cc=d["year"].map(colpos).to_numpy(),
                pb=d["P_b"].to_numpy(np.float64))


def exposure_block(inp, kcols):
    """County-level exposure s_lkt for a block of categories, on the sample."""
    arr, win, rr, cc, pb = inp["arr"], inp["win"], inp["rr"], inp["cc"], inp["pb"]
    S = np.zeros((len(rr), len(kcols)), np.float64)
    for j, (a, b) in win.items():
        m = cc == j
        if not m.any():
            continue
        cbar = arr[:, kcols, a:b + 1].mean(axis=2).astype(np.float64)
        S[m, :] = cbar[rr[m], :] / pb[m][:, None]
    return S


# ------------------------------------------------------- the decomposition --

def rotemberg(inp):
    """alpha_kt, beta_kt and their Section III.C aggregates, plus F_k and
    var(z_k). Everything in Panels A-E is arithmetic on these arrays."""
    cats, cy, gk, arr = inp["cats"], inp["cy"], inp["gk"], inp["arr"]
    years, win, w, yt, xt = inp["years"], inp["win"], inp["w"], inp["yt"], inp["xt"]
    rr, cc, pb = inp["rr"], inp["cc"], inp["pb"]
    K, T, N = len(cats), len(years), len(w)

    NX = np.full((K, T), np.nan)
    NY = np.full((K, T), np.nan)
    GG = np.full((K, T), np.nan)
    VZ = np.full((K, T), np.nan)
    for j, (a, b) in win.items():
        m = cc == j
        if not m.any():
            continue
        cbar = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        Sk = cbar[rr[m], :] / pb[m][:, None]
        wsk = Sk * w[m][:, None]
        NX[:, j] = wsk.T @ xt[m]
        NY[:, j] = wsk.T @ yt[m]
        GG[:, j] = gk[:, cy.index(years[j])]
        VZ[:, j] = Sk.var(axis=0)

    fin = np.isfinite(GG) & np.isfinite(NX) & np.isfinite(NY)
    gNX = np.where(fin, GG * NX, 0.0)
    gNY = np.where(fin, GG * NY, 0.0)
    D = gNX.sum()
    A = gNX / D                                   # alpha_kt
    alpha = A.sum(axis=1)                         # GPSS III.C
    dk = gNX.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta_k = np.where(dk != 0, gNY.sum(axis=1) / dk, np.nan)
    g_k = np.nanmean(np.where(fin, GG, np.nan), axis=1)
    vz_k = np.nanmean(VZ, axis=1)
    dollars = arr.sum(axis=(0, 2)).astype(np.float64)

    # ---- per-category first stage, GPSS footnote 9's single-category Bartik
    ssck = op.t28.ssc_k(inp["fe"], inp["nofe"], inp["cl"])
    G = int(np.unique(inp["cl"]).size)
    ssc = (G / max(G - 1, 1)) * ((N - 1) / max(N - ssck, 1))
    fe, nfe = inp["fe"], int(inp["fe"].max()) + 1
    sw = np.bincount(fe, weights=w, minlength=nfe)
    sw[sw == 0] = 1.0
    Fk = np.full(K, np.nan)
    for s in range(0, K, FBLOCK):
        blk = np.arange(s, min(s + FBLOCK, K))
        B = exposure_block(inp, blk) * np.where(np.isfinite(GG[blk][:, cc]),
                                                GG[blk][:, cc], 0.0).T
        gm = np.column_stack([np.bincount(fe, weights=w * B[:, q],
                                          minlength=nfe) / sw
                              for q in range(len(blk))])
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


def verify(inp, rot):
    """The two identities, plus the vectorized first stage against the house
    estimator. The adding-up target is t28.fit2sls, which predates this file."""
    base = inp["base"]
    sa = float(rot["alpha"].sum())
    if abs(sa - 1.0) > ALPHA_TOL:
        raise SystemExit(f"  FAIL: sum alpha = {sa:.15f}, not one")
    dec = float(np.nansum(rot["alpha"] * rot["beta_k"]))
    rel = abs(dec - base["beta"]) / abs(base["beta"])
    print(f"  sum alpha = {sa:.12f}")
    print(f"  beta_2SLS {base['beta']*locked.SCALE:.6f} vs sum alpha_k beta_k "
          f"{dec*locked.SCALE:.6f}, relative gap {rel:.2e} "
          f"{'PASS' if rel < locked.RTOL_COEF else 'FAIL'}")
    if rel >= locked.RTOL_COEF:
        raise SystemExit("  FAIL: the decomposition does not reproduce the 2SLS")

    rng = np.random.default_rng(0)
    top = np.argsort(-rot["alpha"])[:TOP_N]
    probe = list(dict.fromkeys(top.tolist()
                               + rng.choice(len(rot["alpha"]), 20,
                                            replace=False).tolist()))
    worst = 0.0
    for kq in probe:
        S = exposure_block(inp, np.array([kq]))[:, 0]
        g_row = inp["gk"][kq, [inp["cy"].index(y) for y in inp["years"]]]
        B = S * np.where(np.isfinite(g_row[inp["cc"]]), g_row[inp["cc"]], 0.0)
        r = op.t28.fit2sls(inp["y0"], inp["xw"], B, inp["w"], inp["fe"],
                           inp["nofe"], inp["cl"], singleton=False)
        for a_, b_ in ((r["F"], rot["Fk"][kq]), (r["beta"], rot["beta_k"][kq])):
            if np.isfinite(a_) and np.isfinite(b_) and abs(a_) > 0:
                worst = max(worst, abs(a_ - b_) / abs(a_))
    print(f"  vectorized F_k and beta_k against fit2sls on {len(probe)} "
          f"categories: worst relative gap {worst:.2e} "
          f"{'PASS' if worst < FVEC_TOL else 'FAIL'}")
    if worst >= FVEC_TOL:
        raise SystemExit("  FAIL: the vectorized first stage disagrees with "
                         "the house estimator")


# ------------------------------------------------------------ the labels ---

def fsc_labels(want):
    """Dollar-weighted modal FSC description for the wanted harmonized codes.

    Descriptions exist only in the DCADS layouts (USAspending carries no
    product-or-service description), so they are read there and pushed through
    the same relabel map the category axis was harmonized with. A code that
    does not resolve is reported and falls back to itself.
    """
    _, relabel, _, _ = harmonize.load_harmonization()
    own, merged = {}, {}
    for f in sorted(glob.glob(str(dc.OUT_DIR / "fy*.parquet"))):
        stem = Path(f).stem[2:]            # "1976tq" is the transition quarter
        if not stem.isdigit() or int(stem) < 1990:   # modern description text
            continue
        t = pd.read_parquet(f, columns=["fsc", "fsc_description", "dollars"])
        raw = t["fsc"].astype(str).str.strip().str.upper().str[:4]
        t = t.assign(raw=raw, cat=raw.map(lambda c: relabel.get(c, c)))
        t = t[t["cat"].isin(want)]
        if not len(t):
            continue
        desc = t["fsc_description"].astype(str).str.strip()
        for (c, is_own, dsc), v in t.groupby(
                ["cat", t["raw"] == t["cat"], desc])["dollars"].sum().items():
            if not dsc or len(dsc) <= 2:
                continue
            tgt = own if is_own else merged
            tgt.setdefault(c, {})
            tgt[c][dsc] = tgt[c].get(dsc, 0.0) + float(v)
    # A code's OWN description always wins. The harmonized axis merges retired
    # codes into survivors (1900-1911 all land on 1905), and a merged code's
    # description would otherwise outbid the survivor's on dollars and mislabel
    # the row - "Destroyers" for what is really "Combat Ships and Landing
    # Vessels".
    # Kept verbatim: the source text is already correctly cased ("RDTE/Missile
    # and Space Systems"), and title-casing it would produce "Rdte".
    out = {}
    for c in want:
        src = own.get(c) or merged.get(c)
        if src:
            out[c] = max(src, key=src.get)
    missing = sorted(want - set(out))
    if missing:
        print(f"  NOTE: no FSC description resolved for {missing}; "
              f"falling back to the bare code")
    return out


# ------------------------------------------------------------- the panels --

def panels(inp, rot, labels):
    alpha, beta_k = rot["alpha"], rot["beta_k"]
    g_k, vz, Fk = rot["g_k"], rot["vz_k"], rot["Fk"]
    cats, years, A = inp["cats"], inp["years"], rot["A"]
    S = locked.SCALE
    NCOL = 6                      # value columns; panels use different subsets
    rows = []

    def R(label, *vals):
        rows.append((label, *(list(vals) + [""] * (NCOL - len(vals)))))

    def BANNER(text):
        rows.append((f"\\multicolumn{{{NCOL+1}}}{{l}}{{\\textit{{{text}}}}}",))

    def HEAD(*vals):
        rows.append(("", *(list(vals) + [""] * (NCOL - len(vals)))))

    pos, neg = alpha > 0, alpha < 0
    sp, sn = float(alpha[pos].sum()), float(alpha[neg].sum())
    tot = abs(sn) + sp
    BANNER("Panel A. Negative and positive weights")
    HEAD("Sum", "Mean", "Share", "Categories")
    R("\\hspace{1em}Negative", f"{sn:.3f}", f"{alpha[neg].mean():.4f}",
      f"{abs(sn)/tot:.3f}", f"{int(neg.sum())}")
    R("\\hspace{1em}Positive", f"{sp:.3f}", f"{alpha[pos].mean():.4f}",
      f"{sp/tot:.3f}", f"{int(pos.sum())}")

    ok = (np.isfinite(alpha) & np.isfinite(g_k) & np.isfinite(beta_k)
          & np.isfinite(Fk) & np.isfinite(vz))
    Cm = np.corrcoef(np.column_stack([alpha[ok], g_k[ok], beta_k[ok],
                                      Fk[ok], vz[ok]]).T)
    nm = ["$\\hat\\alpha_k$", "$g_k$", "$\\hat\\beta_k$", "$\\hat F_k$",
          "$\\mathrm{var}(z_k)$"]
    BANNER("Panel B. Correlations")
    HEAD(*nm)
    for i, n in enumerate(nm):
        R("\\hspace{1em}" + n,
          *[f"{Cm[i, j]:.3f}" if j <= i else "" for j in range(5)])

    yr = np.array(years)
    BANNER("Panel C. Variation across decades in $\\hat\\alpha_k$")
    HEAD("Sum", "Mean")
    for dd in sorted(set(((yr // 10) * 10).tolist())):
        m = (yr // 10) * 10 == dd
        s = float(A[:, m].sum())
        if abs(s) < 1e-12:
            continue
        R(f"\\hspace{{1em}}{dd}s", f"{s:.3f}",
          f"{s/(A.shape[0]*int(m.sum())):.6f}")

    top = np.argsort(-alpha)[:TOP_N]
    BANNER("Panel D. Top five Rotemberg weight categories")
    HEAD("$\\hat\\alpha_k$", "$g_k$", "$\\hat\\beta_k$", "95\\% CI",
         "$\\hat F_k$", "Dollar \\%")
    for k in top:
        nm_ = tex_escape(labels.get(cats[k], cats[k]), width=38)
        R(f"\\hspace{{1em}}{nm_}", f"{alpha[k]:.3f}",
          f"{g_k[k]:.3f}", f"{beta_k[k]*S:.2f}", rot["top"][k]["ci_conv"],
          f"{Fk[k]:.1f}", f"{rot['dshare'][k]:.2f}")
    aa = np.abs(alpha)
    R("\\hspace{1em}Largest category's share of $|\\hat\\alpha|$",
      f"{aa[top[0]]/aa.sum():.3f}")
    R("\\hspace{1em}Five largest categories' share",
      f"{aa[top].sum()/aa.sum():.3f}")
    R("\\hspace{1em}Effective categories (1/HHI on $|\\hat\\alpha|$)",
      f"{hhi_effective(aa):.1f}")

    b = inp["base"]["beta"]
    BANNER("Panel E. $\\hat\\beta_k$ by weight sign")
    HEAD("$\\hat\\alpha$-weighted sum", "Share of $\\hat\\beta$",
         "Mean $\\hat\\beta_k$")
    for lab, m in (("Negative", neg), ("Positive", pos)):
        ws = float(np.nansum(alpha[m] * beta_k[m]))
        R(f"\\hspace{{1em}}{lab}", f"{ws*S:.3f}", f"{ws/b:.3f}",
          f"{np.nanmean(beta_k[m])*S:.2f}")
    return rows, top


def correlates(inp, top):
    """Panel F: GPSS Section V.A with strictly pre-exposure covariates.

    For target year t, the treatment base year is t-H and the exposure window
    begins at t-H-LAG-SPAN+1. Every characteristic is measured one year before
    that window begins. The pre-exposure employment-growth window also ends in
    that year.
    """
    d, w, fe, nofe, cl = inp["d"], inp["w"], inp["fe"], inp["nofe"], inp["cl"]
    cats, H, piv = inp["cats"], inp["H"], inp["piv"]

    pre_year = (d["year"].to_numpy(np.int64) - H
                - locked.LAG - locked.SPAN)
    trend_start = pre_year - H
    for j, (a, _) in inp["win"].items():
        m = inp["cc"] == j
        assert np.all(pre_year[m] == int(inp["cy"][a]) - 1), \
            "Panel F characteristics do not precede the exposure window"
    panel_years = set(int(y) for y in piv["P"].columns)
    naics_years = {
        int(p.stem[1:]) for p in NAICS2_DIR.glob("y*.parquet")
        if p.stem[1:].isdigit()
    }
    timing_ok = np.array([
        int(y) in panel_years and int(ys) in panel_years and int(y) in naics_years
        for y, ys in zip(pre_year, trend_start)
    ])
    share_years = sorted(set(pre_year[timing_ok].tolist()))
    if not share_years:
        raise SystemExit("  FAIL: no pre-exposure EFSY years cover Panel F")

    sh = naics2_shares(share_years, inp["units"])
    key = pd.DataFrame(dict(ccg=d["ccg"].to_numpy(),
                            base_year=pre_year))
    # A left merge keeps the left frame's order, so the merged columns align
    # with `d` row for row - but only if the key is unique on the right. A
    # duplicated (ccg, base_year) would silently expand and shift every
    # covariate against its county, with nothing in the output to show it.
    assert not sh.duplicated(["ccg", "base_year"]).any(), \
        "NAICS2 shares are not unique by (ccg, base_year)"
    sh = key.merge(sh, on=["ccg", "base_year"], how="left")
    assert len(sh) == len(d), (len(sh), len(d))
    assert sh.loc[timing_ok, list(SECTORS)].notna().all().all(), \
        "NAICS2 merge left holes"

    year_pos = {int(y): j for j, y in enumerate(piv["P"].columns)}
    rr = inp["rr"]

    def take_at(frame, wanted):
        out = np.full(len(wanted), np.nan, np.float64)
        jj = np.array([year_pos.get(int(y), -1) for y in wanted], np.int64)
        good_j = jj >= 0
        out[good_j] = frame.to_numpy(np.float64)[rr[good_j], jj[good_j]]
        return out

    P_pre = take_at(piv["P"], pre_year)
    E_pre = take_at(piv["E"], pre_year)
    Y_pre = take_at(piv["Y"], pre_year)
    E_start = take_at(piv["E"], trend_start)
    pre_den = E_pre + E_start
    PRE = np.where(pre_den != 0,
                   2.0 * (E_pre - E_start) / pre_den, np.nan)

    chars = {
        "Income per head": Y_pre / P_pre,
        "Employment rate": E_pre / P_pre,
        "Log population": np.log(P_pre),
        "Pre-exposure employment growth": PRE,
    }
    for name in SECTORS:
        chars[name] = sh[name].to_numpy(np.float64)
    names = list(chars)
    Xm = np.column_stack([chars[c] for c in names])
    good = timing_ok & np.isfinite(Xm).all(axis=1)
    # Standardize every characteristic by its weighted standard deviation on
    # the estimation sample, as Table_NS_exposure_balance does at the state
    # level ("both procurement exposure and the listed characteristic are
    # standardized"). The exposures on the left are standardized the same way
    # below, so a coefficient reads as standard deviations per standard
    # deviation and is comparable across categories whose dollar sizes differ
    # by orders of magnitude. Demeaning is left to the fixed effects.
    for q in range(Xm.shape[1]):
        s = wsd(Xm[good, q], w[good])
        if s > 0:
            Xm[:, q] = Xm[:, q] / s
    drop_years = []
    if not good.all():
        # Rows without the full pre-exposure history are named rather than
        # zero-filled.
        for q, c in enumerate(names):
            bad_c = ~timing_ok | ~np.isfinite(Xm[:, q])
            n_c = int(bad_c.sum())
            if n_c:
                yrs = sorted(d.loc[bad_c, "year"].unique().tolist())
                drop_years += yrs
                print(f"  NOTE: '{c}' undefined for {n_c:,} rows, years {yrs}")
        print(f"  Panel F drops {int((~good).sum()):,} of {len(good):,} rows; "
              f"all characteristics precede the exposure window")

    # The coarser-share control on the instrument's own axis (2-character FSC
    # groups, GPSS Section V.A) is deliberately NOT run here: it belongs with
    # the industry-axis work in WS1.
    cols, out = [], []
    S_top = exposure_block(inp, top)
    for q, k in enumerate(top):
        cols.append((cats[k], S_top[:, q]))
    cols.append(("Instrument $z$", inp["z0"]))

    for lab, lhs in cols:
        s = wsd(lhs[good], w[good])
        y = lhs / s if s > 0 else lhs               # standardized, as for NS
        bb, se, r2 = wls_fe(y[good], Xm[good], w[good], fe[good],
                            nofe[good], cl[good])
        bj, Vj, Fj, pj, qj, Gj = joint_balance_wald(
            y[good], Xm[good], w[good], fe[good], nofe[good], cl[good])
        assert np.allclose(bb, bj, rtol=1e-10, atol=1e-12), \
            "Panel F point estimates drifted in the joint test"
        assert np.allclose(se, np.sqrt(np.diag(Vj)), rtol=1e-10, atol=1e-12), \
            "Panel F standard errors drifted in the joint test"
        out.append(dict(label=lab, b=bb, se=se, r2=r2,
                        joint_F=Fj, joint_p=pj, joint_q=qj, joint_G=Gj))
    return names, out, int(good.sum()), sorted(set(drop_years))


# ---------------------------------------------------------------- writers --

def write_table(inp, rot, rows, names, cor, n_f, labels, top, dropped):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    cats, H = inp["cats"], inp["H"]
    base, N = inp["base"], len(inp["w"])
    S = locked.SCALE
    NCOL = 6
    hdr = ([tex_escape(short_name(labels.get(cats[k], cats[k]))) for k in top]
           + ["$z$"])

    body = "\n".join((f"{r[0]} \\\\" if len(r) == 1
                      else " & ".join(list(r)) + " \\\\") for r in rows)

    def frow(lab, vals):
        pad = [""] * (NCOL - len(vals))
        return " & ".join([lab] + list(vals) + pad) + " \\\\"

    f_rows = [f"\\multicolumn{{{NCOL+1}}}{{l}}{{\\textit{{Panel F. Correlates "
              f"of the high-weight categories' exposure}}}} \\\\",
              frow("", hdr), f"\\cmidrule(lr){{2-{len(hdr)+1}}}"]
    for p, nm in enumerate(names):
        f_rows.append(frow("\\hspace{1em}" + nm,
                           [f"{c['b'][p]:.3f}" for c in cor]))
        f_rows.append(frow("", [f"({c['se'][p]:.3f})" for c in cor]))
    f_rows.append(frow("\\hspace{1em}$R^2$",
                       [f"{c['r2']:.3f}" for c in cor]))
    f_rows.append(frow("\\hspace{1em}Joint $F$, all characteristics",
                       [f"{c['joint_F']:.2f}" for c in cor]))
    f_rows.append(frow("\\hspace{1em}$p$-value",
                       [f"[{c['joint_p']:.3f}]" for c in cor]))
    f_rows.append(frow("\\hspace{1em}Observations", [f"{n_f:,}"] * len(cor)))
    fbody = "\n".join(f_rows)

    dec = float(np.nansum(rot["alpha"] * rot["beta_k"]))
    rel = abs(dec - base["beta"]) / abs(base["beta"])
    eff = hhi_effective(np.abs(rot["alpha"]))
    ndrop = N - n_f
    dropped_txt = ", ".join(str(v) for v in dropped)
    tex = TAB_DIR / "Table_rotemberg_fsc.tex"
    tex.write_text(f"""\\begin{{table}}[!htbp]
\\caption{{Which Contract Categories Drive the Estimate: Rotemberg Weights}}
\\label{{tab:rotemberg_fsc}}
{{\\centering
\\adjustbox{{max width=\\linewidth}}{{%
\\begin{{tabular}}{{l*{{{NCOL}}}{{r}}}}
\\toprule
{body}
\\midrule
{fbody}
\\bottomrule
\\end{{tabular}}}}
\\par}}
\\medskip
{{\\scriptsize \\noindent\\textit{{Notes:}} \\textbf{{This table is descriptive.}} It decomposes an
estimate already made; it is not a test of the design. The object decomposed is the county
just-identified 2SLS at a {H}-year horizon on {N:,} county-years at the locked constants:
harmonized top-{len(cats)} four-character Federal Supply Classification axis, positive contract
rule, five-year exposure window lagged five years, division-by-year effects, division-equal
weights, county clusters, treatment winsorized within year. Coefficients are scaled by {S:.0e}.
Following Goldsmith-Pinkham, Sorkin and Swift, $\\hat\\beta=\\sum_{{k,t}}\\hat\\alpha_{{kt}}
\\hat\\beta_{{kt}}$, where $\\hat\\beta_{{kt}}$ uses category $k$'s exposure in year $t$ alone; weights
are aggregated to the category level as in their Section III.C. The weights sum to one and
reproduce the estimate ({base['beta']*S:.4f}, $t$ {base['beta']/base['se']:.2f}) to a relative
gap of {rel:.0e}. Their Section III.B normalization is not applied and is not needed: exposure
shares are dollars per head and do not sum to one within a county.
\\textbf{{The category count is the effective shock count}}: {len(cats)} categories, an
effective {eff:.1f}; relabelling them onto an industry axis creates no new shocks, and the
inferential many-shock version of this exhibit belongs with that industry-axis work.
Panel D's interval is conventional and
county-clustered; the dollar share is the category's share of national contract dollars, times
100. Panel E's mean covers all categories regardless of first-stage power and is dominated by
weakly identified ones; the $\\hat\\alpha$-weighted sums are the interpretable statistics. Panel
F regresses each category's county exposure, and the instrument itself, on county
characteristics measured one year before the exposure window begins, with the same effects,
weights and clusters; exposure and each characteristic
are standardized, as in the state-level exposure balance, so coefficients are standard
deviations per standard deviation. Sector shares are from the desuppressed County Business
Patterns cells, which exclude public administration, so no government share is reported.
The pre-exposure employment-growth window ends one year before the exposure window begins and
spans {H} years. Panel F drops {ndrop:,} county-years (target years {dropped_txt}) without the
required pre-exposure history. The joint statistic tests that all {cor[0]['joint_q']}
characteristic coefficients equal zero using a county-clustered Wald test reported against
$F({cor[0]['joint_q']},{cor[0]['joint_G']-1})$. The first five columns are the GPSS share
diagnostics; the final instrument column is a descriptive summary rather than a
share-exogeneity test.\\par}}
\\end{{table}}
""")
    print(f"  wrote {tex}")

    csv = TAB_DIR / "rotemberg_fsc_locked.csv"
    recs = [dict(panel="A-E", statistic=r[0],
                 **{f"v{q+1}": (r[q + 1] if len(r) > q + 1 else "")
                    for q in range(NCOL)})
            for r in rows]
    for p, nm in enumerate(names):
        recs.append(dict(panel="F", statistic=nm,
                         **{f"v{q+1}": f"{c['b'][p]:.6f} ({c['se'][p]:.6f})"
                            for q, c in enumerate(cor)}))
    recs.append(dict(panel="F", statistic="R2",
                     **{f"v{q+1}": f"{c['r2']:.6f}"
                        for q, c in enumerate(cor)}))
    recs.append(dict(panel="F", statistic="Joint F, all characteristics",
                     **{f"v{q+1}": f"{c['joint_F']:.6f}"
                        for q, c in enumerate(cor)}))
    recs.append(dict(panel="F", statistic="Joint p-value",
                     **{f"v{q+1}": f"{c['joint_p']:.6f}"
                        for q, c in enumerate(cor)}))
    recs.append(dict(panel="F", statistic="column",
                     **{f"v{q+1}": c["label"] for q, c in enumerate(cor)}))
    pd.DataFrame(recs).to_csv(csv, index=False)
    print(f"  wrote {csv}")


def write_records(all_h, labels):
    by_cat, by_year = [], []
    for H, inp, rot in all_h:
        cats = inp["cats"]
        by_cat.append(pd.DataFrame(dict(
            h=H, cat=cats, description=[labels.get(c, "") for c in cats],
            alpha_k=rot["alpha"], beta_k=rot["beta_k"] * locked.SCALE,
            g_k=rot["g_k"], F_k=rot["Fk"], var_z_k=rot["vz_k"],
            dollar_share_pct=rot["dshare"])))
        a = pd.DataFrame(rot["A"], index=cats, columns=inp["years"]).stack()
        by_year.append(a.rename("alpha_kt").reset_index()
                       .set_axis(["cat", "year", "alpha_kt"], axis=1).assign(h=H))
    for frame, name in ((by_cat, "rotemberg_weights_by_category.csv"),
                        (by_year, "rotemberg_alpha_by_year.csv")):
        p = TAB_DIR / name
        pd.concat(frame, ignore_index=True).to_csv(p, index=False)
        print(f"  wrote {p}")


# ------------------------------------------------------------------ main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, choices=list(locked.HORIZONS), default=4,
                    help="horizon for the table; the CSVs sweep all horizons")
    ap.add_argument("--table-only", action="store_true",
                    help="skip the other horizons in the record CSVs")
    args = ap.parse_args()
    t0 = time.time()

    hs = [args.h] if args.table_only else sorted(
        set([args.h] + list(locked.HORIZONS)))
    all_h, main_pair = [], None
    for H in hs:
        print(f"== horizon h={H}")
        inp = build_inputs(H)
        rot = rotemberg(inp)
        verify(inp, rot)
        rot["dshare"] = rot["dollars"] / rot["dollars"].sum() * 100
        all_h.append((H, inp, rot))
        if H == args.h:
            main_pair = (inp, rot)

    inp, rot = main_pair
    top = np.argsort(-rot["alpha"])[:TOP_N]
    labels = fsc_labels({inp["cats"][k] for k in top}
                        | {inp["cats"][k]
                           for k in np.argsort(-np.abs(rot["alpha"]))[:10]})

    # Top-five detail: the house estimator for the point estimate and its
    # conventional clustered interval, and the cluster-robust AR inversion for
    # the weak-instrument-robust one. Both are reported, because they disagree
    # here and the disagreement is the finding.
    stats, n_unb = {}, 0
    for k in top:
        Sk = exposure_block(inp, np.array([k]))[:, 0]
        g_row = inp["gk"][k, [inp["cy"].index(y) for y in inp["years"]]]
        B = Sk * np.where(np.isfinite(g_row[inp["cc"]]), g_row[inp["cc"]], 0.0)
        r = op.t28.fit2sls(inp["y0"], inp["xw"], B, inp["w"], inp["fe"],
                           inp["nofe"], inp["cl"], singleton=False)
        Bt = op.w22.demean(B.reshape(-1, 1), inp["fe"], inp["nofe"],
                           inp["w"])[:, 0]
        lo, hi = ar_interval(Bt, inp["yt"], inp["xt"], inp["w"], inp["cl"],
                             rot["ssc"])
        ar = fmt_ci(lo, hi, locked.SCALE)
        n_unb += ar == "unbounded"
        stats[k] = dict(
            se=r["se"],
            ci_conv=f"({(r['beta']-1.96*r['se'])*locked.SCALE:.2f}, "
                    f"{(r['beta']+1.96*r['se'])*locked.SCALE:.2f})",
            ci_ar=ar)
    rot["top"] = stats
    print(f"  weak-instrument-robust intervals: {n_unb} of {len(top)} "
          f"unbounded under cluster-robust Anderson-Rubin")

    rows, top = panels(inp, rot, labels)
    names, cor, n_f, dropped = correlates(inp, top)
    write_table(inp, rot, rows, names, cor, n_f, labels, top, dropped)
    write_records(all_h, labels)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
