#!/usr/bin/env python3
"""GPSS diagnostics for the locked and bounded-scale instruments.

The bounded-scale design uses the R1 rule declared in producer 21h: apply the
house 1st/99th percentile winsorization within target year to total county
procurement per resident, while leaving county procurement composition and the
national category shocks unchanged.  The resulting instrument remains linear
in category contributions conditional on the fixed scale multiplier:

    z_cap,ct = sum_k q_ct S_ckt g_kt,
    q_ct = E_cap,ct / E_ct.

This producer applies the current GPSS diagnostics to both designs on identical
estimation samples: Rotemberg weights, category removal, balance against the
sixteen fixed 1970 Census covariates, and decade removal.  It also verifies the
category adding-up and 2SLS decomposition identities before writing results.

Writes to Results/pipeline_diagnostics/:

  gpss_design_comparison.csv
  gpss_capped_rotemberg_weights.csv
  gpss_capped_leave_top_out.csv
  gpss_capped_share_balance_1970.csv
  gpss_univariate_balance_1970.csv
  gpss_capped_decade_ladder.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))

import locked  # noqa: E402
import oldpath as op  # noqa: E402
import dcads_common as dc  # noqa: E402

HORIZONS = (2, 4)
TOP_N = 5
DROPS = (0, 1, 2, 3, 5, 10)
FBLOCK = 80
IDENTITY_TOL = 1e-9
VECTOR_TOL = 1e-8
OUT = dc.MACRO_RESULTS_ROOT / "pipeline_diagnostics"


def load_producer(filename, name):
    spec = importlib.util.spec_from_file_location(name, PIPELINE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def effective_exposure_block(inp, gpss, multiplier, kcols):
    """Category exposure loadings q_ct S_ckt for selected categories."""
    return gpss.exposure_block(inp, kcols) * multiplier[:, None]


def category_block(inp, gpss, G, multiplier, kcols):
    """Category instrument contributions q_ct S_ckt g_kt."""
    return (effective_exposure_block(inp, gpss, multiplier, kcols)
            * G[np.ix_(inp["cc"], kcols)])


def instrument_from_categories(inp, gpss, G, multiplier):
    out = np.zeros(len(inp["w"]), np.float64)
    for s in range(0, len(inp["cats"]), FBLOCK):
        blk = np.arange(s, min(s + FBLOCK, len(inp["cats"])))
        out += category_block(inp, gpss, G, multiplier, blk).sum(axis=1)
    return out


def rotemberg(inp, gpss, G_raw, G, multiplier):
    """GPSS category decomposition for an observation-level scale multiplier."""
    K, T, N = len(inp["cats"]), len(inp["years"]), len(inp["w"])
    NX = np.zeros((K, T), np.float64)
    NY = np.zeros((K, T), np.float64)
    VZ = np.full((K, T), np.nan)

    for j in range(T):
        m = inp["cc"] == j
        if not m.any():
            continue
        a, b = inp["win"][j]
        cbar = inp["arr"][:, :, a:b + 1].mean(axis=2).astype(np.float64)
        S = (cbar[inp["rr"][m], :] / inp["pb"][m, None]
             * multiplier[m, None])
        wsk = S * inp["w"][m, None]
        NX[:, j] = wsk.T @ inp["xt"][m]
        NY[:, j] = wsk.T @ inp["yt"][m]
        VZ[:, j] = S.var(axis=0)

    gNX = G.T * NX
    gNY = G.T * NY
    den = float(gNX.sum())
    if abs(den) < 1e-300:
        raise SystemExit("FAIL: the first-stage covariance is zero")
    A = gNX / den
    alpha = A.sum(axis=1)
    dk = gNX.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta_k = np.where(dk != 0, gNY.sum(axis=1) / dk, np.nan)
    with np.errstate(invalid="ignore"):
        g_k = np.where(np.isfinite(G_raw.T).any(axis=1),
                       np.nanmean(G_raw.T, axis=1), np.nan)
        vz_k = np.where(np.isfinite(VZ).any(axis=1),
                        np.nanmean(VZ, axis=1), np.nan)

    ssck = op.t28.ssc_k(inp["fe"], inp["nofe"], inp["cl"])
    ncl = int(np.unique(inp["cl"]).size)
    ssc = ((ncl / max(ncl - 1, 1))
           * ((N - 1) / max(N - ssck, 1)))
    nfe = int(inp["fe"].max()) + 1
    sw = np.bincount(inp["fe"], weights=inp["w"], minlength=nfe)
    sw[sw == 0] = 1.0

    Fk = np.full(K, np.nan)
    for s in range(0, K, FBLOCK):
        blk = np.arange(s, min(s + FBLOCK, K))
        B = category_block(inp, gpss, G, multiplier, blk)
        gm = np.column_stack([
            np.bincount(inp["fe"], weights=inp["w"] * B[:, q],
                        minlength=nfe) / sw
            for q in range(len(blk))
        ])
        Bt = B - gm[inp["fe"], :]
        for q, kq in enumerate(blk):
            bt = Bt[:, q]
            bb = float(bt @ (inp["w"] * bt))
            if bb <= 0:
                continue
            pi = float(bt @ (inp["w"] * inp["xt"])) / bb
            se = op.t28.crv1(bt, bt, inp["xt"] - pi * bt, inp["w"],
                             inp["cl"], N, ssck)
            if se and np.isfinite(se) and se > 0:
                Fk[kq] = (pi / se) ** 2

    dollars = inp["arr"].sum(axis=(0, 2)).astype(np.float64)
    return dict(A=A, alpha=alpha, beta_k=beta_k, g_k=g_k, vz_k=vz_k,
                Fk=Fk, dollars=dollars, ssc=ssc, ssck=ssck)


def verify(inp, gpss, G, multiplier, z, fit, rot, label):
    rebuilt = instrument_from_categories(inp, gpss, G, multiplier)
    scale = max(float(np.abs(z).max()), 1e-300)
    gap_z = float(np.abs(rebuilt - z).max()) / scale
    alpha_sum = float(rot["alpha"].sum())
    beta_dec = float(np.nansum(rot["alpha"] * rot["beta_k"]))
    gap_b = abs(beta_dec - fit["beta"]) / max(abs(fit["beta"]), 1e-300)
    print(f"  {label}: category adding-up gap {gap_z:.2e}; "
          f"sum alpha {alpha_sum:.12f}; 2SLS gap {gap_b:.2e}")
    if gap_z >= IDENTITY_TOL or abs(alpha_sum - 1) >= IDENTITY_TOL \
            or gap_b >= IDENTITY_TOL:
        raise SystemExit(f"FAIL: {label} decomposition identity")

    rng = np.random.default_rng(0)
    top = np.argsort(-rot["alpha"])[:TOP_N]
    probe = list(dict.fromkeys(
        top.tolist() + rng.choice(len(rot["alpha"]), 20, replace=False).tolist()))
    worst, used = 0.0, 0
    for k in probe:
        B = category_block(inp, gpss, G, multiplier, np.array([k]))[:, 0]
        bt = op.w22.demean(B.reshape(-1, 1), inp["fe"], inp["nofe"],
                           inp["w"])[:, 0]
        if float(bt @ (inp["w"] * bt)) <= 0:
            continue
        r = op.t28.fit2sls(inp["y0"], inp["xw"], B, inp["w"], inp["fe"],
                           inp["nofe"], inp["cl"], singleton=False)
        used += 1
        for a, b in ((r["F"], rot["Fk"][k]),
                     (r["beta"], rot["beta_k"][k])):
            if np.isfinite(a) and np.isfinite(b) and abs(a) > 0:
                worst = max(worst, abs(a - b) / abs(a))
    print(f"  {label}: category estimator check on {used} categories, "
          f"worst relative gap {worst:.2e}")
    if used < TOP_N or worst >= VECTOR_TOL:
        raise SystemExit(f"FAIL: {label} vectorized category estimates")


def leverage(inp, z):
    units, key = np.unique(inp["d"]["ccg"].to_numpy(), return_inverse=True)
    zt = op.w22.demean(z.reshape(-1, 1), inp["fe"], inp["nofe"],
                       inp["w"])[:, 0]
    v = np.bincount(key, weights=inp["w"] * zt * zt, minlength=len(units))
    sh = v / v.sum()
    return float(1 / np.sum(sh ** 2)), float(sh.max())


def design_row(H, label, inp, z, fit, rot):
    eff, maxlev = leverage(inp, z)
    a = np.abs(rot["alpha"])
    order = np.argsort(-a)
    return dict(
        h=H, design=label, beta=fit["beta"] * locked.SCALE,
        se=fit["se"] * locked.SCALE, t=fit["beta"] / fit["se"], F=fit["F"],
        observations=len(inp["w"]), counties=int(np.unique(inp["cl"]).size),
        effective_counties=eff, max_county_leverage=maxlev,
        negative_alpha_sum=float(rot["alpha"][rot["alpha"] < 0].sum()),
        top1_abs_alpha_share=float(a[order[:1]].sum() / a.sum()),
        top5_abs_alpha_share=float(a[order[:5]].sum() / a.sum()),
        effective_categories=float(1 / np.sum((a / a.sum()) ** 2)))


def weight_rows(H, label, inp, gpss, rot):
    cats = np.asarray(inp["cats"])
    labels = gpss.fsc_labels(set(cats))
    return [dict(
        h=H, design=label, category=cats[k],
        description=labels.get(cats[k], cats[k]), alpha_k=rot["alpha"][k],
        beta_k=rot["beta_k"][k] * locked.SCALE, g_k=rot["g_k"][k],
        F_k=rot["Fk"][k], var_z_k=rot["vz_k"][k],
        dollar_share_pct=100 * rot["dollars"][k] / rot["dollars"].sum())
            for k in range(len(cats))]


def leave_top_out(inp, gpss, G, multiplier, z, rot, label):
    cats = np.asarray(inp["cats"])
    order = np.argsort(-rot["alpha"])
    aa = np.abs(rot["alpha"])
    total_a = float(aa.sum())
    rows = []
    for n in DROPS:
        drop = order[:n]
        zd = z.copy()
        if n:
            zd -= category_block(inp, gpss, G, multiplier, drop).sum(axis=1)
        r = op.t28.fit2sls(inp["y0"], inp["xw"], zd, inp["w"], inp["fe"],
                           inp["nofe"], inp["cl"], singleton=False)
        eff, maxlev = leverage(inp, zd)
        rows.append(dict(
            design=label, dropped=n,
            categories="; ".join(cats[k] for k in drop),
            beta=r["beta"] * locked.SCALE, se=r["se"] * locked.SCALE,
            t=r["beta"] / r["se"], F=r["F"], effective_counties=eff,
            max_county_leverage=maxlev,
            abs_alpha_removed=float(aa[drop].sum() / total_a) if n else 0.0))
    return rows


def balance_rows(inp, gpss, balance, bhj1970, G, multiplier, z, rot, label):
    names, X = balance.characteristics(inp, bhj1970)
    top = np.argsort(-rot["alpha"])[:TOP_N]
    labels = gpss.fsc_labels({inp["cats"][k] for k in top})
    cols = [(f"{label} instrument", z)]
    cols += [(labels.get(inp["cats"][k], inp["cats"][k]),
              effective_exposure_block(inp, gpss, multiplier,
                                       np.array([k]))[:, 0])
             for k in top]
    res = balance.run(inp, gpss, cols, names, X)
    rows = []
    for r in res:
        base = dict(design=label, column=r["label"], joint_F=r["F"],
                    joint_p=r["p"], individual_rejections=r["rejections"],
                    effective_counties=r["effective"], clusters=r["clusters"])
        for q, name in enumerate(names):
            rows.append(dict(**base, characteristic=name,
                             coefficient=r["b"][q], se=r["se"][q]))
    return rows


def univariate_balance_rows(inp, gpss, balance, bhj1970, z, label):
    """One balance regression per fixed 1970 characteristic."""
    names, X = balance.characteristics(inp, bhj1970)
    Xs = balance.standardize(X, inp["w"], gpss, "characteristics")
    sy = gpss.wsd(z, inp["w"])
    if sy <= 0:
        raise SystemExit(f"FAIL: zero weighted variance in {label}")
    y = z / sy
    units, key = np.unique(inp["d"]["ccg"].to_numpy(), return_inverse=True)
    effective = balance.effective_counties(z, inp, key, len(units))
    ncl = int(np.unique(inp["cl"]).size)
    rows = []
    for q, name in enumerate(names):
        b, se, r2 = balance.wls_fe(
            y, Xs[:, [q]], inp["w"], inp["fe"], inp["nofe"], inp["cl"])
        t = float(b[0] / se[0])
        p = float(2 * stats.t.sf(abs(t), max(ncl - 1, 1)))
        rows.append(dict(
            design=label, characteristic=name, coefficient=float(b[0]),
            se=float(se[0]), t=t, p=p, F=t * t, r2=float(r2),
            effective_counties=effective, observations=len(inp["w"]),
            clusters=ncl))
    return rows


def decade_rows(inp, z, label):
    years = np.asarray(inp["years"])
    decades = (years // 10) * 10
    rows = []
    for drop in [None] + sorted(set(decades.tolist())):
        m = (np.ones(len(inp["w"]), bool) if drop is None
             else ~np.isin(inp["cc"], np.where(decades == drop)[0]))
        if int(m.sum()) < 1000:
            continue
        fe = pd.factorize(inp["fe"][m], sort=False)[0].astype(np.int64)
        cl = pd.factorize(inp["cl"][m], sort=False)[0].astype(np.int64)
        r = op.t28.fit2sls(inp["y0"][m], inp["xw"][m], z[m], inp["w"][m],
                           fe, np.zeros(int(m.sum()), np.int64), cl,
                           singleton=False)
        rows.append(dict(
            design=label, dropped_decade="none" if drop is None else int(drop),
            beta=r["beta"] * locked.SCALE, se=r["se"] * locked.SCALE,
            t=r["beta"] / r["se"], F=r["F"], observations=int(m.sum()),
            counties=int(np.unique(cl).size)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    args = ap.parse_args()

    gpss = load_producer("21_rotemberg.py", "rotemberg_producer")
    frozen = load_producer("21b_gpss_frozen_share_rotemberg.py",
                           "frozen_producer")
    capper = load_producer("21h_gpss_capped_scale_instrument.py",
                          "capper_producer")
    balance = load_producer("21d_gpss_share_balance_1970.py",
                           "balance_producer")
    bhj1970 = load_producer("22b_bhj_1970_balance_h2.py",
                           "bhj1970_producer")
    OUT.mkdir(parents=True, exist_ok=True)

    designs, weights, leaves, balances, univariate, decades = [], [], [], [], [], []
    for H in args.horizons:
        print(f"h={H}")
        inp = gpss.build_inputs(H)
        G_raw, G = frozen.national_matrix(inp)
        z_locked, E, znorm = capper.decompose(inp, G)
        E_cap, _, _ = capper.house_rule(E, inp["d"]["year"].to_numpy())
        multiplier = np.divide(E_cap, E, out=np.zeros_like(E), where=E > 0)
        z_cap = E_cap * znorm
        gap = float(np.abs(z_locked - inp["z0"]).max()) / max(
            float(np.abs(inp["z0"]).max()), 1e-300)
        if gap >= IDENTITY_TOL:
            raise SystemExit(f"FAIL: h={H} locked instrument gap {gap:.2e}")

        specs = (
            ("uncapped", z_locked, np.ones_like(multiplier),
             gpss.rotemberg(inp)),
            ("capped_r1", z_cap, multiplier,
             rotemberg(inp, gpss, G_raw, G, multiplier)),
        )
        for label, z, q, rot in specs:
            fit = op.t28.fit2sls(inp["y0"], inp["xw"], z, inp["w"], inp["fe"],
                                 inp["nofe"], inp["cl"], singleton=False)
            verify(inp, gpss, G, q, z, fit, rot, label)
            designs.append(design_row(H, label, inp, z, fit, rot))
            weights.extend(weight_rows(H, label, inp, gpss, rot))
            leaves.extend(dict(h=H, **r) for r in
                          leave_top_out(inp, gpss, G, q, z, rot, label))
            balances.extend(dict(h=H, **r) for r in
                            balance_rows(inp, gpss, balance, bhj1970, G, q, z,
                                         rot, label))
            univariate.extend(dict(h=H, **r) for r in
                              univariate_balance_rows(
                                  inp, gpss, balance, bhj1970, z, label))
            decades.extend(dict(h=H, **r) for r in decade_rows(inp, z, label))

    files = {
        "gpss_design_comparison.csv": pd.DataFrame(designs),
        "gpss_capped_rotemberg_weights.csv": pd.DataFrame(weights),
        "gpss_capped_leave_top_out.csv": pd.DataFrame(leaves),
        "gpss_capped_share_balance_1970.csv": pd.DataFrame(balances),
        "gpss_univariate_balance_1970.csv": pd.DataFrame(univariate),
        "gpss_capped_decade_ladder.csv": pd.DataFrame(decades),
    }
    for name, frame in files.items():
        path = OUT / name
        frame.to_csv(path, index=False)
        print(f"  wrote {path} ({len(frame):,} rows)")


if __name__ == "__main__":
    main()
