#!/usr/bin/env python3
"""The registered aggregation battery: one table per failure channel.

Six tables on the locked baseline (composition top-800, span 5, positive rule,
population denominator, reduced form, division-by-year effects, rung-unit
clusters), rows = the five rungs, column blocks = c_j = P_j (fixed 1975
population) and c_j = 1:

  Table 0  the main ladder: beta_b, delta, beta_a aligned and own, per rung,
           rendered from aggregation_baseline.csv (no new estimation)
  T1       transformation order: standard growth winsorized at the 1st and
           99th percentiles by year at the county versus at the rung,
           difference regressed on z
  T2       naive RHS aggregation: z summed over summed base population
           (population weights) versus the outcome's employment-share z
  T3       implicit weights: the augmented county regression under naive
           county weights versus the identity-matched weights om_c * c_j
  T4       Jensen: own log growth of the summed unit minus the share-weighted
           mean of county log growths, regressed on z; log ONLY, per the
           paper (the standard-growth wedge is an exact zero and is asserted,
           not estimated; the midpoint wedge is estimated only as the wiring
           assertion against the locked jensen column and never reported)
  T5       composition, size-by-exposure only: the reported object is the
           CHANGE in the estimated spillover delta when the two size
           interactions enter, the county's shock times county size and
           ShareTreated times unit size (size = within-year-centered log
           base employment); the interaction coefficients stay in the CSV

TWO CHANNELS ARE DELIBERATELY ABSENT. Support drift: the panel is one frozen
county frame reused at every rung, so the two regressions cannot cover
different populations; the frame's holes against the full rectangle are
counted and printed instead. Entry: no unit above the county ever enters or
exits, and the county-years excluded for a zero employment endpoint are
counted and printed per horizon.

Coefficient DIFFERENCES (T2, T3, T5) carry a stacked standard error: the two
specifications share their rung-unit clusters, so Var(b1 - b2) is computed
from the per-cluster influence functions of both fits jointly, never from the
two marginal SEs.

The midpoint outcome is the project's recorded deviation from protocol A.2.3;
duplicate y_std rows for T2, T3, T5 go to the CSV so enforcing A.2.3 later is
a re-cut of this file, not a rebuild.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import dcads_common as dc  # noqa: E402

p57 = None  # loaded in main() so --help stays fast
w22 = None
t28 = None

RESULTS = dc.MACRO_RESULTS_ROOT / "ns14_diagnostics"
CSV_OUT = dc.LOG_DIR / "aggregation_battery.csv"
BASE_CSV = dc.LOG_DIR / "aggregation_baseline.csv"
LOG = dc.LOG_DIR / "68_aggregation_battery.log"

WINSOR_PCT = (0.01, 0.99)
RUNGS = ("czone", "msa", "cea", "ea", "state")
WEIGHTS = (("P_j", "fixed pop"), ("EW", "equal"))
TABLE_H = 4


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


# ---------------------------------------------------------------------------
# Estimation: one weighted FE regression with CRV1 rung-unit clusters, plus
# per-cluster influence functions so stacked difference SEs need no second
# machinery. Single-regressor legs that must tie to aggregation_baseline.csv
# go through t28.fit2sls instead, so the SE convention is identical to the
# one that produced the locked numbers.
# ---------------------------------------------------------------------------

def cwls(y, X, w, fe, cl):
    """Multi-regressor weighted FE regression, CRV1 on `cl`.

    Returns b, V, and the per-cluster influence rows psi (G x k) with
    sum_g psi_g = 0 and Var(b) = (G/(G-1)) * sum_g psi_g psi_g'.
    """
    y = np.asarray(y, np.float64)
    Xc = [np.asarray(c, np.float64) for c in X]
    w = np.asarray(w, np.float64)
    ze = np.zeros(len(y), np.int64)
    M = w22.demean(np.column_stack([y] + Xc), fe, ze, w)
    yt, Xt = M[:, 0], M[:, 1:]
    XtWX = Xt.T @ (w[:, None] * Xt)
    inv = np.linalg.inv(XtWX)
    b = inv @ (Xt.T @ (w * yt))
    u = yt - Xt @ b
    cl = pd.factorize(cl)[0]
    G = cl.max() + 1
    S = np.zeros((G, Xt.shape[1]))
    contrib = Xt * (w * u)[:, None]
    np.add.at(S, cl, contrib)
    psi = S @ inv.T
    V = (G / max(G - 1, 1)) * (psi.T @ psi)
    return b, V, psi, G


def diff_se(psi1, j1, psi2, j2, G):
    """SE of b1[j1] - b2[j2] from the joint per-cluster influences."""
    d = psi1[:, j1] - psi2[:, j2]
    return float(np.sqrt((G / max(G - 1, 1)) * (d @ d)))


def clip_pct(v, year, p=None):
    """Percentile winsor by year; values clipped, never dropped."""
    p = WINSOR_PCT if p is None else p
    s = pd.Series(np.asarray(v, np.float64))
    g = s.groupby(np.asarray(year))
    lo = g.transform(lambda x: x.quantile(p[0]))
    hi = g.transform(lambda x: x.quantile(p[1]))
    out = s.clip(lo, hi)
    return out.to_numpy(), (s != out).to_numpy()


# ---------------------------------------------------------------------------
# The rung frame: everything every test needs at one (h, rung), built once.
# ---------------------------------------------------------------------------

def rung_frame(d, lev, fe_name="divisionyear"):
    g = d[d[lev].notna()].copy()
    g["om"] = g["E_b"] / g.groupby([lev, "year"])["E_b"].transform("sum")
    P0j = g.drop_duplicates("ccg").groupby(lev)["P0"].sum()
    _, div, straddle = p57.unit_labels(d, lev)
    di = g[lev].map(div)
    if fe_name == "regionyear":
        di = di.map(p57.REGION)
    g["fe_i"] = pd.factorize(di.astype(str) + "_" + g["year"].astype(str))[0]
    g["cl_i"] = pd.factorize(g[lev])[0]

    agg = (g.assign(wl=g["om"] * g["y_log"], wm=g["om"] * g["y_mpgr"],
                    ws=g["om"] * g["y_std"], wz=g["om"] * g["z"],
                    wp=g["P_b"] * g["z"])
             .groupby([lev, "year"], as_index=False)
             .agg(y_log=("wl", "sum"), y_mpgr=("wm", "sum"),
                  y_std=("ws", "sum"), share=("wz", "sum"),
                  zP=("wp", "sum"), Pb=("P_b", "sum"),
                  E_b=("E_b", "sum"), E_t=("E_t", "sum")))
    agg["z_naive"] = agg["zP"] / agg["Pb"]
    agg["popw"] = agg[lev].map(P0j)
    da = agg[lev].map(div)
    if fe_name == "regionyear":
        da = da.map(p57.REGION)
    agg["fe_a"] = pd.factorize(da.astype(str) + "_" + agg["year"].astype(str))[0]
    agg["cl_a"] = pd.factorize(agg[lev])[0]

    g["share"] = agg.set_index([lev, "year"])["share"].reindex(
        pd.MultiIndex.from_arrays([g[lev], g["year"]])).to_numpy()
    g["P0j"] = g[lev].map(P0j)
    return g, agg, P0j, straddle


def weights_for(g, agg, cj):
    """County-side and aggregate-side regression weights, mean one."""
    if cj == "EW":
        wc_naive = np.ones(len(g))
        wc_match = g["om"].to_numpy(np.float64)
        wa = np.ones(len(agg))
    else:
        wc_naive = g["P0"].to_numpy(np.float64)
        wc_match = (g["om"] * g["P0j"]).to_numpy(np.float64)
        wa = agg["popw"].to_numpy(np.float64)
    return (wc_naive / wc_naive.mean(), wc_match / wc_match.mean(),
            wa / wa.mean())


# ---------------------------------------------------------------------------
# The tests. Each returns tidy rows; every printed number is one of them.
# ---------------------------------------------------------------------------

def own_growth(agg, outcome):
    if outcome == "y_mpgr":
        return (2.0 * (agg["E_t"] - agg["E_b"])
                / (agg["E_t"] + agg["E_b"])).to_numpy(np.float64)
    if outcome == "y_std":
        return ((agg["E_t"] - agg["E_b"]) / agg["E_b"]).to_numpy(np.float64)
    return (np.log(agg["E_t"]) - np.log(agg["E_b"])).to_numpy(np.float64)


def t1_transformation(g, agg, lev, wa, rows, h, cj):
    ac, boundA = clip_pct(g["y_std"], g["year"])
    yA = (pd.Series(g["om"].to_numpy() * ac)
          .groupby([g[lev].to_numpy(), g["year"].to_numpy()]).sum())
    yA = yA.reindex(pd.MultiIndex.from_frame(agg[[lev, "year"]])).to_numpy()
    own = own_growth(agg, "y_std")
    gap0 = np.abs(own - agg["y_std"].to_numpy())
    assert gap0.max() < 1e-10, (
        f"T1 unwinsorized B - A is not zero at {lev}: {gap0.max():.2e}")
    yB, boundB = clip_pct(own, agg["year"])
    za = agg["share"].to_numpy(np.float64)
    fe = agg["fe_a"].to_numpy(np.int64)
    ze = np.zeros(len(agg), np.int64)
    cl = agg["cl_a"].to_numpy(np.int64)
    r = t28.fit2sls(yB - yA, za, za, wa, fe, ze, cl, singleton=False,
                    want_F=False)
    # THE PREDICTOR OF THIS CHANNEL'S SIZE, reported always: the share of the
    # raw coefficient living in the clipped tail. Winsorization order only
    # matters when the clipped mass is treatment-correlated: 59 percent of
    # the raw state coefficient in the BCHTT immigration replication, 1.6
    # percent here, and the order effect scales with it.
    r_raw = t28.fit2sls(own, za, za, wa, fe, ze, cl, singleton=False,
                        want_F=False)
    r_A = t28.fit2sls(yA, za, za, wa, fe, ze, cl, singleton=False,
                      want_F=False)
    tail = 100.0 * (r_raw["beta"] - r_A["beta"]) / r_raw["beta"]
    _, _, ps_raw, G = cwls(own, [za], wa, fe, cl)
    _, _, ps_A, _ = cwls(yA, [za], wa, fe, cl)
    sd_rm = diff_se(ps_raw, 0, ps_A, 0, G)
    shA = g.loc[boundA, "E_b"].sum() / g["E_b"].sum()
    rows += [dict(test="T1", horizon=h, level=lev, weight=cj, outcome="y_std",
                  stat="beta_construction_error", value=r["beta"] * p57.SCALE,
                  se=r["se"] * p57.SCALE, n=r["n"], n_units=agg[lev].nunique()),
             dict(test="T1", horizon=h, level=lev, weight=cj, outcome="y_std",
                  stat="beta_std_raw", value=r_raw["beta"] * p57.SCALE,
                  se=r_raw["se"] * p57.SCALE, n=r_raw["n"],
                  n_units=agg[lev].nunique()),
             dict(test="T1", horizon=h, level=lev, weight=cj, outcome="y_std",
                  stat="removed_by_winsor",
                  value=(r_raw["beta"] - r_A["beta"]) * p57.SCALE,
                  se=sd_rm * p57.SCALE, n=r_raw["n"],
                  n_units=agg[lev].nunique()),
             dict(test="T1", horizon=h, level=lev, weight=cj, outcome="y_std",
                  stat="tail_share_pct", value=tail, se=np.nan,
                  n=r_raw["n"], n_units=agg[lev].nunique()),
             dict(test="T1", horizon=h, level=lev, weight=cj, outcome="y_std",
                  stat="n_clipped_county", value=int(boundA.sum()),
                  se=np.nan, n=len(g), n_units=np.nan),
             dict(test="T1", horizon=h, level=lev, weight=cj, outcome="y_std",
                  stat="emp_share_clipped_county", value=shA, se=np.nan,
                  n=len(g), n_units=np.nan),
             dict(test="T1", horizon=h, level=lev, weight=cj, outcome="y_std",
                  stat="n_clipped_unit", value=int(boundB.sum()), se=np.nan,
                  n=len(agg), n_units=np.nan)]


def t2_naive_rhs(agg, lev, wa, rows, h, cj, outcome):
    y = agg[outcome].to_numpy(np.float64)
    fe = agg["fe_a"].to_numpy(np.int64)
    cl = agg["cl_a"].to_numpy(np.int64)
    za = agg["share"].to_numpy(np.float64)
    zn = agg["z_naive"].to_numpy(np.float64)
    ra = t28.fit2sls(y, za, za, wa, fe, np.zeros(len(agg), np.int64), cl,
                     singleton=False, want_F=False)
    rn = t28.fit2sls(y, zn, zn, wa, fe, np.zeros(len(agg), np.int64), cl,
                     singleton=False, want_F=False)
    _, _, psa, G = cwls(y, [za], wa, fe, cl)
    _, _, psn, _ = cwls(y, [zn], wa, fe, cl)
    sd = diff_se(psn, 0, psa, 0, G)
    rows += [dict(test="T2", horizon=h, level=lev, weight=cj, outcome=outcome,
                  stat=s, value=v * p57.SCALE, se=e * p57.SCALE, n=ra["n"],
                  n_units=agg[lev].nunique())
             for s, v, e in (("beta_aligned", ra["beta"], ra["se"]),
                             ("beta_naive", rn["beta"], rn["se"]),
                             ("diff", rn["beta"] - ra["beta"], sd))]
    return ra["beta"]


def t3_implicit(g, agg, lev, wc_naive, wc_match, rows, h, cj, outcome):
    y = g[outcome].to_numpy(np.float64)
    X = [g["z"].to_numpy(np.float64), g["share"].to_numpy(np.float64)]
    fe = g["fe_i"].to_numpy(np.int64)
    cl = g["cl_i"].to_numpy(np.int64)
    b1, V1, ps1, G = cwls(y, X, wc_naive, fe, cl)
    b2, V2, ps2, _ = cwls(y, X, wc_match, fe, cl)
    sd = diff_se(ps2, 0, ps1, 0, G)
    rows += [dict(test="T3", horizon=h, level=lev, weight=cj, outcome=outcome,
                  stat=s, value=v * p57.SCALE, se=e * p57.SCALE, n=len(g),
                  n_units=agg[lev].nunique())
             for s, v, e in (
                 ("beta_b_naive", b1[0], np.sqrt(V1[0, 0])),
                 ("beta_b_matched", b2[0], np.sqrt(V2[0, 0])),
                 ("delta_matched", b2[1], np.sqrt(V2[1, 1])),
                 ("diff", b2[0] - b1[0], sd))]
    return b1[0], b2[0], b2[1]


def t4_jensen(g, agg, lev, wa, rows, h, cj):
    fe = agg["fe_a"].to_numpy(np.int64)
    cl = agg["cl_a"].to_numpy(np.int64)
    za = agg["share"].to_numpy(np.float64)
    std_wedge = np.abs(own_growth(agg, "y_std") - agg["y_std"].to_numpy())
    assert std_wedge.max() < 1e-10, (
        f"T4 standard wedge is not the identity zero at {lev}: "
        f"{std_wedge.max():.2e}")
    # The paper's Jensen story is about log ONLY; the midpoint wedge is never
    # reported. It is still estimated here because its coefficient equals the
    # locked jensen column by linearity, which is the one anchor tying this
    # test's wiring to aggregation_baseline.csv.
    out = {}
    for oc in ("y_log", "y_mpgr"):
        wedge = own_growth(agg, oc) - agg[oc].to_numpy(np.float64)
        r = t28.fit2sls(wedge, za, za, wa, fe, np.zeros(len(agg), np.int64),
                        cl, singleton=False, want_F=False)
        if oc == "y_log":
            rows.append(dict(test="T4", horizon=h, level=lev, weight=cj,
                             outcome=oc, stat="beta_wedge",
                             value=r["beta"] * p57.SCALE,
                             se=r["se"] * p57.SCALE,
                             n=r["n"], n_units=agg[lev].nunique()))
        out[oc] = r["beta"]
    return out


def t5_composition(g, agg, lev, wc_match, rows, h, cj, outcome):
    """Size-by-exposure, one interaction per side: the county's own shock
    times county size, and the unit's ShareTreated times unit size. Size is
    log base employment, centered within year on its own level (counties
    across counties, units across units), so delta keeps its level reading.
    ShareTreated itself stays in, so beta_a = beta_b + delta is untouched.
    """
    y = g[outcome].to_numpy(np.float64)
    z = g["z"].to_numpy(np.float64)
    sh = g["share"].to_numpy(np.float64)
    fe = g["fe_i"].to_numpy(np.int64)
    cl = g["cl_i"].to_numpy(np.int64)
    sc = np.log(g["E_b"].to_numpy(np.float64))
    sc = sc - pd.Series(sc).groupby(g["year"].to_numpy()).transform("mean")
    sju = np.log(agg["E_b"].to_numpy(np.float64))
    sju = (pd.Series(sju, index=pd.MultiIndex.from_frame(agg[[lev, "year"]]))
           .groupby("year").transform(lambda x: x - x.mean()))
    sj = sju.reindex(pd.MultiIndex.from_arrays(
        [g[lev], g["year"]])).to_numpy(np.float64)
    b0, V0, ps0, G = cwls(y, [z, sh], wc_match, fe, cl)
    bq, Vq, psq, _ = cwls(y, [z, z * sc.to_numpy(), sh, sh * sj],
                          wc_match, fe, cl)
    sd = diff_se(psq, 2, ps0, 1, G)
    rows += [dict(test="T5", horizon=h, level=lev, weight=cj, outcome=outcome,
                  stat=s, value=v * p57.SCALE, se=e * p57.SCALE, n=len(g),
                  n_units=agg[lev].nunique())
             for s, v, e in (
                 ("delta", b0[1], np.sqrt(V0[1, 1])),
                 ("delta_Q", bq[2], np.sqrt(Vq[2, 2])),
                 ("diff", bq[2] - b0[1], sd),
                 ("theta_z_x_size", bq[1], np.sqrt(Vq[1, 1])),
                 ("lambda_share_x_size", bq[3], np.sqrt(Vq[3, 3])))]
    return b0[1], bq[2]


# ---------------------------------------------------------------------------
# Tables. Full environments in the ns14_diagnostics style; the writers consume
# the CSV file, never in-memory state.
# ---------------------------------------------------------------------------

NOTES_COMMON = (
    "Locked baseline per the \\texttt{BASELINE} dict of producer 57: "
    "composition instrument on the harmonized top-800 panel, 5-year share "
    "window lagged 5, positive-rule dollars, population denominator, reduced "
    "form, division-by-year effects, $h=%d$. Standard errors cluster on the "
    "rung unit in every regression, county-side included; the state rung has "
    "51 clusters. Weight blocks: $c_j=P_j$ fixes the unit's 1975 population, "
    "$c_j=1$ weights units equally. Differences carry a stacked standard "
    "error from the joint per-cluster influence functions of the two fits. "
    "$^{*}$, $^{**}$ and $^{***}$ denote significance at the 10, 5 and 1 "
    "percent level on the clustered t statistic. "
    "Rungs are provisional until the MSA-versus-CEA decision. " % TABLE_H)

NOTE_MPGR = (
    "The outcome is midpoint growth, the project's recorded deviation from "
    "protocol A.2.3; the share regressor is built from ex-ante employment "
    "shares, so the endogenous-weight mechanism A.2.3 names does not operate, "
    "and the residual own-versus-aligned wedge is quantified in the Jensen "
    "table. ")


def fmt(v, dec=3):
    return "" if not np.isfinite(v) else f"{v:.{dec}f}"


def stars_t(t):
    """The project's convention: normal critical values on the clustered t."""
    if not np.isfinite(t):
        return ""
    t = abs(t)
    return ("$^{***}$" if t > 2.576 else "$^{**}$" if t > 1.96 else
            "$^{*}$" if t > 1.645 else "")


def stars(v, e):
    return stars_t(v / e) if np.isfinite(e) and e > 0 else ""


def write_table(name, title, label, header, body, notes):
    tex = "\n".join([
        "%% %s" % title,
        "% Rendered by 68_aggregation_battery.py from "
        "aggregation_battery.csv. Do not edit.",
        r"\begin{table}[!htbp]",
        r"\caption{%s}" % title,
        r"\label{tab:%s}" % label,
        r"{\centering",
        r"\adjustbox{max width=\linewidth}{%",
        r"\begin{tabular}{l%s}" % ("r" * (len(header.split("&")) - 1)),
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\par}",
        r"\medskip",
        r"{\scriptsize \noindent\textit{Notes:} %s\par}" % notes,
        r"\end{table}",
        "",
    ])
    out = RESULTS / name
    out.write_text(tex)
    print(f"  wrote {out.name} -> {out.parent}")


def delta_se(o, lev, cj):
    """delta's SE, from the battery's matched augmented regression (T3),
    whose point estimate is asserted to tie to the locked delta column."""
    m = o[(o.test == "T3") & (o.stat == "delta_matched")
          & (o.outcome == "y_mpgr") & (o.level == lev)
          & (o.weight == cj) & (o.horizon == TABLE_H)]
    return float(m.iloc[0]["se"]) if len(m) else np.nan


def beta_b_se(o, lev, cj):
    """beta_b's SE, from the same T3 matched augmented regression delta_se
    reads, whose point estimate is asserted to tie to the locked beta_b.

    It cannot come from the locked file. That file stores `t_county`, the t of
    the county regression WITHOUT ShareTreated, beside `beta_b`, the county
    coefficient WITH it. Dividing one by the other returns neither regression's
    standard error, and until 2026-09-06 the ladder did exactly that: the
    equal-weighted state cell printed (0.135) for a coefficient whose clustered
    SE is 0.169, and carried three stars where the augmented fit supports two.
    """
    m = o[(o.test == "T3") & (o.stat == "beta_b_matched")
          & (o.outcome == "y_mpgr") & (o.level == lev)
          & (o.weight == cj) & (o.horizon == TABLE_H)]
    return float(m.iloc[0]["se"]) if len(m) else np.nan


def ladder_rows(o, base, cols, lab):
    """Two-line rows for the ladder: coefficient with stars, SE beneath.

    The aggregate columns are read as beta / t off the locked file, which IS
    their clustered SE because both halves come from the same fit. beta_b and
    delta take their SEs from the battery's matched augmented regression,
    because the locked file holds no SE for either.
    """
    b = base[base.horizon.eq(TABLE_H)]
    body = []
    for lev in RUNGS:
        top, bot = [lab[lev]], [""]
        for cj in ("P_j", "EW"):
            r = b[b.level.eq(lev) & b.weight.eq(cj)].iloc[0]
            dse = delta_se(o, lev, cj)
            bse = beta_b_se(o, lev, cj)
            spec = {"beta_a_aligned": r["t_agg"], "beta_a_own": r["t_agg_own"]}
            for c in cols:
                if c in ("delta", "beta_b"):
                    e = dse if c == "delta" else bse
                    top.append(fmt(r[c]) + stars(r[c], e))
                    bot.append(f"({fmt(e)})")
                else:
                    t = spec[c]
                    top.append(fmt(r[c]) + stars_t(t))
                    bot.append(f"({fmt(abs(r[c] / t))})")
        top.append(f"{int(b[b.level.eq(lev)].n_units.max()):,}")
        bot.append("")
        body.append(" & ".join(top) + r" \\")
        body.append(" & ".join(bot) + r" \\")
    return body


def cell_tailshare(o, lev, cj):
    """The tail share, masked where the raw coefficient is indistinguishable
    from zero, because a ratio to a zero-ish base reads as a huge number
    that means nothing. The components stay in the CSV either way."""
    m = (o.test.eq("T1") & o.level.eq(lev) & o.weight.eq(cj)
         & o.horizon.eq(TABLE_H))
    raw = o[m & o.stat.eq("beta_std_raw")]
    tl = o[m & o.stat.eq("tail_share_pct")]
    if not len(raw) or not len(tl):
        return ""
    v, e = raw.iloc[0]["value"], raw.iloc[0]["se"]
    if not (np.isfinite(e) and e > 0 and abs(v / e) >= 2):
        return "--"    # en dash, never the --- em-dash ligature (Hard Rule 2)
    return fmt(tl.iloc[0]["value"], 1)


def cell_pval(o, test, lev, cj, stat, outcome, dec=3):
    """Point estimate with the two-sided p-value of the stacked cluster test
    in parentheses; no SE and no stars, because the point of the exercise is
    the movement, not the levels."""
    from math import erf, sqrt
    m = (o.test.eq(test) & o.level.eq(lev) & o.weight.eq(cj)
         & o.stat.eq(stat) & o.horizon.eq(TABLE_H) & o.outcome.eq(outcome))
    r = o[m]
    if not len(r):
        return ""
    v, e = r.iloc[0]["value"], r.iloc[0]["se"]
    if not (np.isfinite(e) and e > 0):
        return fmt(v, dec)
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(v / e) / sqrt(2.0))))
    return f"{fmt(v, dec)} ({p:.3f})"


def cell(o, test, lev, cj, stat, outcome=None, dec=3, star_se=True):
    m = (o.test.eq(test) & o.level.eq(lev) & o.weight.eq(cj)
         & o.stat.eq(stat) & o.horizon.eq(TABLE_H))
    if outcome:
        m &= o.outcome.eq(outcome)
    r = o[m]
    if not len(r):
        return ""
    v, e = r.iloc[0]["value"], r.iloc[0]["se"]
    return f"{fmt(v, dec)}{stars(v, e)} ({fmt(e, dec)})" \
        if (star_se and np.isfinite(e)) else fmt(v, dec)


def render_tables(o, base):
    lab = {"czone": "Commuting zone", "msa": "MSA", "cea": "CEA", "ea": "EA",
           "state": "State"}

    body = ladder_rows(o, base,
                       ["beta_b", "delta", "beta_a_aligned", "beta_a_own"],
                       lab)
    write_table(
        "Table_agg_ladder.tex", "The aggregation ladder", "agg_ladder",
        r" & \multicolumn{4}{c}{$c_j=P_j$} & \multicolumn{4}{c}{$c_j=1$} & \\"
        "\n" + r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}" + "\n"
        r"Rung & $\beta_b$ & $\delta$ & $\beta_a^{al}$ & $\beta_a^{own}$"
        r" & $\beta_b$ & $\delta$ & $\beta_a^{al}$ & $\beta_a^{own}$ & Units",
        body,
        "Rendered from \\texttt{aggregation\\_baseline.csv}, the locked "
        "specification estimated by producer 57; this table is the object "
        "the five channel tables attribute. " + NOTES_COMMON + NOTE_MPGR)

    specs = [
        ("Table_agg_transformation_order.tex",
         "Transformation order: winsorize at the county or at the rung",
         "agg_transformation", "T1",
         [("beta_construction_error", "Construction error"),
          ("removed_by_winsor", "Removed by winsor"),
          ("tail_share_pct", "Tail share (\\%)")], "y_std",
         "Standard growth winsorized at percentiles "
         + "%g and %g" % (WINSOR_PCT[0] * 100, WINSOR_PCT[1] * 100)
         + " by year, at the county then aggregated (A) versus at the rung "
         "(B); the coefficient "
         "is (B$-$A) on the instrument. Unwinsorized, B$-$A is identically "
         "zero, asserted at 1e-10. Winsorized standard growth is a benchmark "
         "construction, never the answer; no other series is winsorized. "
         "Clipped-observation counts are in "
         "the CSV. Removed-by-winsor is beta on the unwinsorized aggregate "
         "minus beta on the county-winsorized one, with a stacked cluster "
         "SE; the tail share is that removal as a percent of the raw "
         "coefficient, the diagnostic that predicts this channel's size "
         "(1.6 percent here at the state under P_j against 59 percent in "
         "the BCHTT immigration replication, where the fence deletes the "
         "responding counties), shown only where the raw coefficient "
         "clears |t| = 2. " + NOTES_COMMON),
        ("Table_agg_naive_rhs.tex",
         "Naive aggregation of the regressor: population against employment "
         "weights", "agg_naive_rhs", "T2",
         [("beta_aligned", "$\\beta$, employment-share $z$"),
          ("beta_naive", "$\\beta$, summed-flow $z$"),
          ("diff", "Difference")], "y_mpgr",
         "Summing county flows over summed base population aggregates the "
         "regressor with population weights; the outcome aggregates with "
         "employment shares. The naive column is the benchmark, not the "
         "answer. " + NOTES_COMMON + NOTE_MPGR),
        ("Table_agg_implicit_weights.tex",
         "Implicit weights: naive against identity-matched county weights",
         "agg_implicit_weights", "T3",
         [("beta_b_naive", "$\\beta_b$, naive weights"),
          ("beta_b_matched", "$\\beta_b$, matched $\\omega_c c_j$"),
          ("diff", "Difference")], "y_mpgr",
         "Both legs are the augmented county regression (A.2.4). Naive "
         "weights are 1 (EW block) or county 1975 population ($P_j$ block); "
         "matched weights are the county's employment share in the rung unit "
         "times $c_j$, the pair under which $\\beta_a=\\beta_b+\\delta$ "
         "holds. " + NOTES_COMMON + NOTE_MPGR),
        ("Table_agg_jensen.tex",
         "Jensen: own growth of the sum against the mean of the growths",
         "agg_jensen", "T4",
         [("beta_wedge:y_log", "Log wedge")], None,
         "The wedge is the unit's own log growth from summed employment "
         "minus the employment-share mean of county log growths, regressed "
         "on the instrument. The standard-growth wedge is an exact zero by "
         "the adding-up identity and is asserted at 1e-10, not estimated. "
         + NOTES_COMMON),
        ("Table_agg_composition.tex",
         "Composition: what size-by-exposure controls do to the estimated "
         "spillover",
         "agg_composition", "T5",
         [("delta", "$\\delta$"), ("delta_Q", "$\\delta_Q$"),
          ("diff", "$\\Delta\\delta$ ($p$)")],
         "y_mpgr",
         "The exercise is the movement of the estimated spillover when the "
         "two size-by-exposure interactions enter: the county's own shock "
         "times county size, and $ShareTreated_j$ times unit size, with "
         "size the within-year-centered log of base employment at each "
         "side's own level. $ShareTreated_j$ itself stays in, so "
         "$\\beta_a=\\beta_b+\\delta$ is untouched. The two deltas print "
         "without standard errors, because their levels are not the point; "
         "the parenthesis on $\\Delta\\delta$ is the two-sided p-value of "
         "the stacked cluster test that the movement is zero, and the "
         "interaction coefficients are in the CSV. $\\delta$ is a spillover "
         "only under the maintained condition "
         "$Cov_\\omega(\\tau_i, ShareTreated_j)=0$; the controls remove "
         "only the part of composition operating through size. "
         + NOTES_COMMON + NOTE_MPGR),
    ]
    for name, title, label, test, stats, oc, notes in specs:
        body = []
        for lev in RUNGS:
            row = [lab[lev]]
            for cj in ("P_j", "EW"):
                for s, _ in stats:
                    if ":" in s:
                        st, so = s.split(":")
                        row.append(cell(o, test, lev, cj, st, so))
                    elif test == "T5":
                        # Points only; the movement carries the p-value.
                        row.append(cell_pval(o, test, lev, cj, s, oc)
                                   if s == "diff" else
                                   cell(o, test, lev, cj, s, oc,
                                        star_se=False))
                    elif s == "tail_share_pct":
                        row.append(cell_tailshare(o, lev, cj))
                    else:
                        row.append(cell(
                            o, test, lev, cj, s, oc,
                            star_se=(s != "joint_F_controls"),
                            dec=(2 if s == "joint_F_controls" else 3)))
            nu = o[o.test.eq(test) & o.level.eq(lev)
                   & o.horizon.eq(TABLE_H)]["n_units"].max()
            row.append(f"{int(nu):,}" if np.isfinite(nu) else "")
            body.append(" & ".join(row) + r" \\")
        k = len(stats)
        head = (r" & \multicolumn{%d}{c}{$c_j=P_j$} & "
                r"\multicolumn{%d}{c}{$c_j=1$} & \\" % (k, k) + "\n"
                + r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}"
                % (1 + k, 2 + k, 1 + 2 * k) + "\n"
                + "Rung & " + " & ".join(h for _, h in stats)
                + " & " + " & ".join(h for _, h in stats) + " & Units")
        write_table(name, title, label, head, body, notes)


def write_prose_numbers(o, base):
    """One file per number the slides quote in prose, per dynamic-numbers.md.

    The deck used to type these in. They went stale twice: the ladder frame
    still read 0.642 and 2.360 for a county coefficient and an aggregate that
    the locked file now puts at 0.549 and 2.235, and its note still gave a
    reconciliation chain from a vintage before the geography correction. A
    number the frame can quote is a number the frame can get wrong, so the
    producer writes it and the frame inputs it.

    Each file holds the formatted value and a literal `%` with no trailing
    newline, so TeX's \\endlinechar cannot leak a space into the sentence.
    """
    d = RESULTS / "_coeff"
    d.mkdir(parents=True, exist_ok=True)

    def put(name, text):
        (d / f"{name}.tex").write_text(text + "%")

    b = base[base.horizon.eq(TABLE_H)]
    # Every rung, both weightings: the bullets quote the state and the extremes
    # of the ratio column, and which rung is extreme is itself a result that can
    # move, so no rung is left un-inputtable.
    for lev, cj, tag in [(l, c, f"{l}_{'Pj' if c == 'P_j' else c}")
                         for l in RUNGS for c in ("P_j", "EW")]:
        r = b[b.level.eq(lev) & b.weight.eq(cj)].iloc[0]
        rc = o[o.test.eq("RECON") & o.level.eq(lev) & o.weight.eq(cj)
               & o.horizon.eq(TABLE_H)]
        put(f"agg_beta_b_{tag}", fmt(r["beta_b"]))
        put(f"agg_delta_{tag}", fmt(r["delta"]))
        put(f"agg_beta_a_own_{tag}", fmt(r["beta_a_own"]))
        put(f"agg_beta_b_noaug_{tag}", fmt(r["beta_b_noaug"]))
        put(f"agg_naive_gap_{tag}", fmt(r["beta_a_own"] - r["beta_b_noaug"]))
        put(f"agg_naive_ratio_{tag}",
            fmt((r["beta_a_own"] - r["beta_b_noaug"]) / r["delta"], 2))
        if len(rc):
            x = rc.iloc[0]
            # The chain is printed the way the note reads it, signs included,
            # so the frame concatenates snippets rather than doing arithmetic.
            for k, nm in (("gap_own_vs_naive_b", "gap"), ("ch_implicit", "implicit"),
                          ("ch_delta", "delta"), ("ch_jensen", "jensen")):
                v = float(x[k])
                put(f"agg_recon_{nm}_{tag}",
                    ("$-$" if v < 0 else "") + fmt(abs(v)))
    put("agg_n_ccg", f"{int(b['n_ccg'].max()):,}")
    print(f"  wrote {len(list(d.glob('*.tex')))} prose numbers -> {d}")


def naive_rows(o, base, lab, frag):
    """One fragment per weighting: the spillover read correctly, and the one an
    applied paper reads off the same two regressions without the augmentation.

    The naive researcher never estimates delta. They run the county regression,
    run the rung regression, and call the difference a spillover. That is
    `beta_a_own - beta_b_noaug`, where the county coefficient comes from the
    fit WITHOUT ShareTreated on the right-hand side. The correct delta comes
    from the augmented county regression, which is the only place delta is
    identified. The ratio of the two is what the slide is for.

    Both differences are read against beta_a_own, the rung's own growth rate,
    rather than the aligned aggregate. That is the regression an applied paper
    would actually run, so the naive number carries the Jensen wedge as well as
    the omitted-variable shift; the wedge is in the locked file's `jensen`
    column, and it is an order of magnitude smaller than the shift at every
    rung.

    Neither the difference nor the ratio carries a standard error. They combine
    a county fit and a rung fit that share their clusters, so an honest SE
    needs the joint influence functions the way T2, T3 and T5 do it, and a
    marginal-SE approximation dressed up with stars would be worse than the
    blank the reader can see.
    """
    b = base[base.horizon.eq(TABLE_H)]
    for cj, tag in (("P_j", "Pj"), ("EW", "EW")):
        body = []
        for lev in RUNGS:
            r = b[b.level.eq(lev) & b.weight.eq(cj)].iloc[0]
            dse, bse = delta_se(o, lev, cj), beta_b_se(o, lev, cj)
            nse, ase = r["se_b_noaug"], r["beta_a_own_se"]
            dn = r["beta_a_own"] - r["beta_b_noaug"]
            top = [lab[lev],
                   fmt(r["beta_b"]) + stars(r["beta_b"], bse),
                   fmt(r["delta"]) + stars(r["delta"], dse),
                   fmt(r["beta_a_own"]) + stars(r["beta_a_own"], ase),
                   r"\orange{%s}" % (fmt(r["beta_b_noaug"])
                                     + stars(r["beta_b_noaug"], nse)),
                   r"\orange{%s}" % fmt(dn),
                   r"\orange{%s}" % fmt(dn / r["delta"], 2),
                   f"{int(r['n_units']):,}"]
            bot = ["", f"({fmt(bse)})", f"({fmt(dse)})", f"({fmt(ase)})",
                   r"\orange{(%s)}" % fmt(nse), "", "", ""]
            body.append(" & ".join(top) + r" \\")
            body.append(" & ".join(bot) + r" \\")
        frag(f"Table_agg_naive_spillover_{tag}_slides.tex", "0.93",
             r" & \multicolumn{3}{c}{Augmented} & "
             r"\multicolumn{3}{c}{\orange{Naive}} & \\" "\n"
             + r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n"
             + r"Rung & $\beta_b$ & $\delta$ & $\beta_a^{own}$"
               r" & \orange{$\beta_b^{\,-S}$}"
               r" & \orange{$\beta_a^{own}-\beta_b^{\,-S}$}"
               r" & \orange{Ratio} & Units",
             body, 7)


def render_slides(o, base):
    """Bare adjustbox fragments for the working deck, one per table.

    No notes and no captions: the frame around them carries the message, and
    the full-table environments hold the record.
    """
    lab = {"czone": "Commuting zone", "msa": "MSA", "cea": "CEA", "ea": "EA",
           "state": "State"}

    def frag(name, width, header, body, ncol):
        # The height cap keeps a two-line-row table from overfilling a frame
        # that also carries two bullets.
        tex = "\n".join([
            "% Slide fragment. Rendered by 68_aggregation_battery.py from "
            "aggregation_battery.csv. Do not edit.",
            r"\adjustbox{max width=%s\linewidth, "
            r"max totalheight=0.58\textheight}{" % width,
            r"\begin{tabular}{l%s}" % ("r" * ncol),
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}}",
            "",
        ])
        out = RESULTS / name
        out.write_text(tex)
        print(f"  wrote {out.name} -> {out.parent}")

    write_prose_numbers(o, base)
    naive_rows(o, base, lab, frag)

    body = ladder_rows(o, base, ["beta_b", "delta", "beta_a_own"], lab)
    frag("Table_agg_ladder_slides.tex", "0.88",
         r" & \multicolumn{3}{c}{$c_j=P_j$} & \multicolumn{3}{c}{$c_j=1$} & \\"
         "\n" + r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n"
         r" & $\beta_b$ & $\delta$ & $\beta_a^{own}$"
         r" & $\beta_b$ & $\delta$ & $\beta_a^{own}$ & Units",
         body, 7)

    slide_specs = [
        ("Table_agg_transformation_order_slides.tex", "0.93", "T1",
         [("removed_by_winsor", "Removed by winsor"),
          ("beta_construction_error", "Construction error")], "y_std"),
        ("Table_agg_naive_rhs_slides.tex", "0.97", "T2",
         [("beta_aligned", r"$\beta$, share $z$"),
          ("beta_naive", r"$\beta$, summed $z$"),
          ("diff", "Difference")], "y_mpgr"),
        ("Table_agg_implicit_weights_slides.tex", "0.97", "T3",
         [("beta_b_naive", r"$\beta_b$, naive"),
          ("beta_b_matched", r"$\beta_b$, matched"),
          ("diff", "Difference")], "y_mpgr"),
        ("Table_agg_jensen_slides.tex", "0.6", "T4",
         [("beta_wedge:y_log", "Log wedge")], None),
        ("Table_agg_composition_slides.tex", "0.76", "T5",
         [("delta", r"$\delta$"), ("delta_Q", r"$\delta_Q$"),
          ("diff", r"$\Delta\delta$ ($p$)")], "y_mpgr"),
    ]
    for name, width, test, stats, oc in slide_specs:
        body = []
        for lev in RUNGS:
            row = [lab[lev]]
            for cj in ("P_j", "EW"):
                for s, _ in stats:
                    if ":" in s:
                        st, so = s.split(":")
                        row.append(cell(o, test, lev, cj, st, so))
                    elif test == "T5":
                        row.append(cell_pval(o, test, lev, cj, s, oc)
                                   if s == "diff" else
                                   cell(o, test, lev, cj, s, oc,
                                        star_se=False))
                    elif s == "tail_share_pct":
                        row.append(cell_tailshare(o, lev, cj))
                    else:
                        row.append(cell(
                            o, test, lev, cj, s, oc,
                            star_se=(s != "joint_F_controls"),
                            dec=(2 if s == "joint_F_controls" else 3)))
            row.append(f"{int(o[o.test.eq(test) & o.level.eq(lev) & o.horizon.eq(TABLE_H)]['n_units'].max()):,}")
            body.append(" & ".join(row) + r" \\")
        k = len(stats)
        head = (r" & \multicolumn{%d}{c}{$c_j=P_j$} & "
                r"\multicolumn{%d}{c}{$c_j=1$} & \\" % (k, k) + "\n"
                + r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}"
                % (1 + k, 2 + k, 1 + 2 * k) + "\n"
                + " & " + " & ".join(h for _, h in stats)
                + " & " + " & ".join(h for _, h in stats) + " & Units")
        frag(name, width, head, body, 2 * k + 1)


# ---------------------------------------------------------------------------


def main():
    global p57, w22, t28
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render-only", action="store_true",
                    help="re-render tables and slide fragments from the saved "
                         "CSVs; no estimation")
    ap.add_argument("--t1-pct", nargs=2, type=float, default=[1.0, 99.0],
                    metavar=("LO", "HI"),
                    help="T1's winsor fence, in percent (default 1 99)")
    args = ap.parse_args()
    global WINSOR_PCT
    WINSOR_PCT = (args.t1_pct[0] / 100.0, args.t1_pct[1] / 100.0)
    if args.render_only:
        o = pd.read_csv(CSV_OUT)
        base = pd.read_csv(BASE_CSV)
        render_tables(o, base)
        render_slides(o, base)
        return
    sys.stdout = Tee(LOG)
    t0 = time.time()
    p57 = __import__("importlib.util", fromlist=["x"])  # placeholder
    import importlib.util as ilu

    def _load(name, path):
        spec = ilu.spec_from_file_location(name, path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    p57 = _load("p57", CODE / "57_aggregation_invariance.py")
    import pyfixest as pf
    w22 = _load("w22", p57.STEP2_CODE / "22_winsor_construction_error.py")
    t28 = _load("t28", p57.STEP2_CODE / "28_table_d1.py")
    w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)

    if not BASE_CSV.exists():
        raise SystemExit(f"  FAIL: {BASE_CSV} missing; run 57 --baseline first")
    base = pd.read_csv(BASE_CSV)

    B = p57.BASELINE
    print("=" * 92)
    print("  THE AGGREGATION BATTERY   locked baseline, rung-unit clusters")
    print("  gate: at h=4, state, P_j the county coefficient is "
          f"{base.query('horizon==4 & level==\"state\" & weight==\"P_j\"').iloc[0].beta_b:.3f} "
          "against an own aggregate of "
          f"{base.query('horizon==4 & level==\"state\" & weight==\"P_j\"').iloc[0].beta_a_own:.3f}")
    print("=" * 92)

    rows, recon = [], []
    for h in B["horizons"]:
        d, _ = p57.build_county(B["instrument"], h=h, span=B["span"],
                                rule=B["rule"], floor=B["floor"],
                                denom=B["denom"])
        years = sorted(d["year"].unique())
        expect = 3058 * len(years)
        print(f"\n  h={h}: {len(d):,} county-years of {expect:,} possible "
              f"({expect - len(d):,} holes: unobservable z/x or a zero "
              "employment endpoint)")
        sets = {}
        for lev in RUNGS:
            sets[lev] = frozenset(d.loc[d[lev].notna(), "ccg"])
        assert len(set(sets.values())) == 1, (
            "county set differs across rungs: "
            + str({k: len(v) for k, v in sets.items()}))

        for lev in RUNGS:
            g, agg, P0j, straddle = rung_frame(d, lev, B["fe"])
            # Data identity, independent recomputation, before any estimate.
            chk = (g.groupby([lev, "year"])
                    .apply(lambda x: np.average(x["y_mpgr"],
                                                weights=x["E_b"]),
                           include_groups=False))
            gap = np.abs(chk.reindex(pd.MultiIndex.from_frame(
                agg[[lev, "year"]])).to_numpy() - agg["y_mpgr"].to_numpy())
            # 1e-9: order-of-summation noise on E_b-weighted means sits near
            # 1e-12; anything real is orders of magnitude above this.
            assert gap.max() < 1e-9, f"reconstruction {lev}: {gap.max():.2e}"
            znchk = (g.groupby([lev, "year"])
                      .apply(lambda x: np.average(x["z"], weights=x["P_b"]),
                             include_groups=False))
            zgap = np.abs(znchk.reindex(pd.MultiIndex.from_frame(
                agg[[lev, "year"]])).to_numpy() - agg["z_naive"].to_numpy())
            assert zgap.max() < 1e-9, f"T2 z_naive {lev}: {zgap.max():.2e}"

            for cj, _scheme in WEIGHTS:
                wc_naive, wc_match, wa = weights_for(g, agg, cj)
                t1_transformation(g, agg, lev, wa, rows, h, cj)
                for oc in ("y_mpgr", "y_std"):
                    ba = t2_naive_rhs(agg, lev, wa, rows, h, cj, oc)
                    bb_n, bb_m, dl = t3_implicit(g, agg, lev, wc_naive,
                                                 wc_match, rows, h, cj, oc)
                    d0, dq = t5_composition(g, agg, lev, wc_match, rows,
                                            h, cj, oc)
                    if oc == "y_mpgr":
                        # Ties to the locked CSV, in scaled units.
                        r = base[(base.horizon == h) & (base.level == lev)
                                 & (base.weight == cj)].iloc[0]
                        for got, want, nm in (
                                (ba * p57.SCALE, r["beta_a_aligned"],
                                 "beta_a_aligned"),
                                (bb_m * p57.SCALE, r["beta_b"], "beta_b"),
                                ((bb_m + dl) * p57.SCALE,
                                 r["beta_b"] + r["delta"], "b+d"),
                                (d0 * p57.SCALE, r["delta"], "delta")):
                            assert abs(got - want) < 1e-6, (
                                f"{nm} at {lev}/{cj}/h{h}: "
                                f"{got:.8f} vs locked {want:.8f}")
                wed = t4_jensen(g, agg, lev, wa, rows, h, cj)
                r = base[(base.horizon == h) & (base.level == lev)
                         & (base.weight == cj)].iloc[0]
                # The mpgr wedge coefficient IS own - aligned by linearity.
                assert abs(wed["y_mpgr"] * p57.SCALE - r["jensen"]) < 1e-6, (
                    f"jensen tie {lev}/{cj}/h{h}")
                # Reconciliation: the chain is exact; the remainder must be 0.
                rr = [x for x in rows
                      if x["horizon"] == h and x["level"] == lev
                      and x["weight"] == cj and x["outcome"] == "y_mpgr"]
                bbn = next(x["value"] for x in rr
                           if x["test"] == "T3" and x["stat"] == "beta_b_naive")
                bbm = next(x["value"] for x in rr
                           if x["test"] == "T3"
                           and x["stat"] == "beta_b_matched")
                dlm = next(x["value"] for x in rr
                           if x["test"] == "T3"
                           and x["stat"] == "delta_matched")
                t2d = next(x["value"] for x in rr
                           if x["test"] == "T2" and x["stat"] == "diff")
                gapo = r["beta_a_own"] - bbn
                rem = gapo - ((bbm - bbn) + dlm + r["jensen"])
                assert abs(rem) < 1e-6, f"reconciliation {lev}/{cj}/h{h}: {rem}"
                recon.append(dict(horizon=h, level=lev, weight=cj,
                                  gap_own_vs_naive_b=gapo,
                                  ch_implicit=bbm - bbn, ch_delta=dlm,
                                  ch_jensen=r["jensen"], remainder=rem,
                                  memo_naive_rhs_shift=t2d))
            print(f"    {lev:<7} done, straddle {straddle}, "
                  f"{time.time()-t0:6.1f}s")

    # SE-level check: my aligned t at state/h4/P_j against the locked t_agg.
    o = pd.DataFrame(rows)
    r4 = o[(o.test == "T2") & (o.level == "state") & (o.weight == "P_j")
           & (o.horizon == 4) & (o.outcome == "y_mpgr")
           & (o.stat == "beta_aligned")].iloc[0]
    want = base.query('horizon==4 & level=="state" & weight=="P_j"').iloc[0]
    t_got = r4["value"] / r4["se"]
    assert abs(t_got - want["t_agg"]) < 1e-6, (
        f"SE convention drifted: t {t_got:.6f} vs locked {want['t_agg']:.6f}")
    print(f"\n  SE check: aligned t at state/h4/P_j {t_got:.4f} "
          f"reproduces locked t_agg")

    rec = pd.DataFrame(recon)
    rec.insert(0, "test", "RECON")
    out = pd.concat([o, rec], ignore_index=True)
    out.to_csv(CSV_OUT, index=False)
    print(f"\n  wrote {CSV_OUT}")

    render_tables(pd.read_csv(CSV_OUT), base)
    render_slides(pd.read_csv(CSV_OUT), base)

    print("\n  T4/T5 headline at h=4, state, P_j:")
    for t, s, oc in (("T4", "beta_wedge", "y_log"),
                     ("T5", "diff", "y_mpgr")):
        r = o[(o.test == t) & (o.stat == s) & (o.outcome == oc)
              & (o.level == "state") & (o.weight == "P_j")
              & (o.horizon == 4)].iloc[0]
        print(f"    {t} {s} ({oc}): {r['value']:.4f} ({fmt(r['se'])})")
    print(f"\n  0 checks failed, {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
