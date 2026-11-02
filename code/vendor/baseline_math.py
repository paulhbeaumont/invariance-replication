"""Source-extracted arithmetic for the locked county baseline.

Extraction provenance: config/baseline_sources.json.
"""
import numpy as np
import pandas as pd


def lam_weights(h, lam):
    w = np.array([lam ** (-j) for j in range(h)], dtype=np.float64)
    return w / w.sum()


def wins(v, key):
    out = v.astype(np.float64).copy()
    for g in np.unique(key):
        m = key == g
        lo, hi = np.nanpercentile(out[m], [1.0, 99.0])
        out[m] = np.clip(out[m], lo, hi)
    return out


def demean(X, f1, f2, wt, tol=1e-12, maxit=2000):
    X = np.array(X, dtype=np.float64, copy=True)
    n1, n2 = (f1.max() + 1, f2.max() + 1)
    s1 = np.bincount(f1, weights=wt, minlength=n1)
    s2 = np.bincount(f2, weights=wt, minlength=n2)
    s1[s1 == 0], s2[s2 == 0] = (1.0, 1.0)
    scale = float(np.abs(X[:, 0]).max())
    thresh = tol * scale if scale > 0.0 else tol
    for it in range(1, maxit + 1):
        prev = X[:, 0].copy()
        for f, s, n in ((f1, s1, n1), (f2, s2, n2)):
            for j in range(X.shape[1]):
                m = np.bincount(f, weights=wt * X[:, j], minlength=n) / s
                X[:, j] -= m[f]
        moved = float(np.abs(X[:, 0] - prev).max())
        if moved < thresh:
            return X
    raise RuntimeError(f'demean did not converge in {maxit} sweeps on {len(X):,} rows: last sweep moved {moved:.3e} against a threshold of {thresh:.3e} (tol {tol:g} relative to a column scale of {scale:.3e}). The coefficient this would produce is unconverged and is not returned.')


def drop_singletons(f1, f2):
    keep = np.ones(len(f1), bool)
    while True:
        n_before = int(keep.sum())
        for f in (f1, f2):
            cnt = np.bincount(f[keep], minlength=int(f.max()) + 1)
            keep[keep] &= cnt[f[keep]] > 1
        if int(keep.sum()) == n_before:
            return keep


def is_nested(f, cl):
    n = int(f.max()) + 1
    c = np.bincount(f, minlength=n).astype(np.float64)
    s = np.bincount(f, weights=cl.astype(np.float64), minlength=n)
    mean = np.divide(s, c, out=np.zeros(n), where=c > 0)
    return bool(np.all(mean[f] == cl))


def ssc_k(f1, f2, cl):
    k, counted = (1, 0)
    for f in (f1, f2):
        if not is_nested(f, cl):
            k += int(np.unique(f).size)
            counted += 1
    return k - max(counted - 1, 0)


def crv1(zt, xt, resid, w, cl, n, k, corrected=True, correction_n=None):
    zx = float(zt @ (w * xt))
    if not np.isfinite(zx) or zx == 0.0:
        return np.nan
    g = np.bincount(cl, weights=w * zt * resid, minlength=int(cl.max()) + 1)
    G = int(np.unique(cl).size)
    n_ssc = n if correction_n is None else int(correction_n)
    if corrected and n_ssc <= k:
        raise ValueError(f'CRV1 correction requires N > K; got N={n_ssc}, K={k}')
    ssc = G / max(G - 1, 1) * ((n_ssc - 1) / (n_ssc - k)) if corrected else 1.0
    v = ssc * float(g @ g) / zx ** 2
    return float(np.sqrt(v)) if v > 0 else np.nan


def fit2sls(y, x, z, w, f1, f2, cl, singleton=True, want_F=True, correction_n=None):
    keep = drop_singletons(f1, f2) if singleton else np.ones(len(y), bool)
    y, x, z, w = (y[keep], x[keep], z[keep], w[keep])
    f1, f2, cl = (f1[keep], f2[keep], cl[keep])
    f1 = pd.factorize(f1)[0].astype(np.int64)
    f2 = pd.factorize(f2)[0].astype(np.int64)
    cl = pd.factorize(cl)[0].astype(np.int64)
    iv = z is not x
    M = demean(np.column_stack([y, x] + ([z] if iv else [])), f1, f2, w)
    yt, xt = (M[:, 0], M[:, 1])
    zt = M[:, 2] if iv else xt
    zx = float(zt @ (w * xt))
    beta = float(zt @ (w * yt)) / zx if zx != 0.0 else np.nan
    n, k = (len(y), ssc_k(f1, f2, cl))
    se = crv1(zt, xt, yt - beta * xt, w, cl, n, k, correction_n=correction_n)
    F = np.nan
    if want_F and iv:
        bfs = zx / float(zt @ (w * zt))
        sefs = crv1(zt, zt, xt - bfs * zt, w, cl, n, k, correction_n=correction_n)
        F = (bfs / sefs) ** 2 if sefs and np.isfinite(sefs) and (sefs > 0) else np.nan
    return dict(beta=beta, se=se, F=F, n=n, k=k, correction_n=n if correction_n is None else int(correction_n), n_dropped=int((~keep).sum()), keep=keep)
