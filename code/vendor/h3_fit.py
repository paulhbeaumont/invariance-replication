"""Unchanged functions extracted from the original sandbox source."""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import t
from baseline_math import demean, ssc_k, fit2sls
def require(ok, message):
    if not ok: raise RuntimeError(message)

def fit(d, ys, xs, zs, fe='ccg', cluster='unit', correction_n=None):
    """Co-author's demean/ssc_k and matrix IV sandwich; scaled for stability."""
    w = d.weight.to_numpy(float)
    f1 = pd.factorize(d[fe])[0]; f2 = pd.factorize(d.year)[0]
    cl, labels = pd.factorize(d[cluster], sort=True)
    cols = [ys] + xs + zs
    raw = d[cols].to_numpy(float)
    require(np.isfinite(raw).all() and np.isfinite(w).all() and (w > 0).all(), 'Invalid fit input')
    scales = np.maximum(abs(raw).max(axis=0), 1e-30)
    M = demean(raw / scales, f1, f2, w)
    # Each transformed column must satisfy BOTH FE normal equations.
    for ff in [f1, f2]:
        sw = np.bincount(ff, weights=w)
        for j in range(M.shape[1]):
            require(abs(np.bincount(ff, weights=w * M[:, j]) / sw).max() < 1e-9, 'Incomplete FE absorption')
    kx = len(xs); Y = M[:, 0]; X = M[:, 1:1+kx]; Z = M[:, 1+kx:]
    ZX = Z.T @ (w[:, None] * X)
    require(np.linalg.matrix_rank(ZX) == kx, 'Singular first stage')
    b = np.linalg.solve(ZX, Z.T @ (w * Y))
    zz = Z.T @ (w[:, None] * Z)
    pred = Z @ np.linalg.solve(zz, ZX)
    check = np.linalg.lstsq(pred * np.sqrt(w)[:, None], Y * np.sqrt(w), rcond=None)[0]
    require(np.allclose(b, check, atol=1e-10, rtol=1e-8), 'Projected-IV cross-check failed')
    G = len(labels); n = len(d) if correction_n is None else correction_n
    k = ssc_k(f1, f2, cl) + kx - 1
    ssc = G / (G-1) * (n-1) / (n-k)
    u = Y-X@b
    scores = np.zeros((G, kx)); np.add.at(scores, cl, Z * (w*u)[:, None])
    factor = scales[0] / scales[1:1+kx]
    psi = (scores @ np.linalg.inv(ZX).T) * np.sqrt(ssc) * factor
    ses = np.linalg.norm(psi, axis=0)
    F = []
    for j in range(kx):
        pi = np.linalg.solve(zz, Z.T@(w*X[:, j])); v = X[:, j]-Z@pi
        ss = np.zeros((G, kx)); np.add.at(ss, cl, Z*(w*v)[:, None])
        bread = np.linalg.inv(zz); V = ssc*bread@(ss.T@ss)@bread.T
        F.append(float(pi @ np.linalg.solve(V, pi) / kx))
    out = {v: dict(beta=float(b[j]*factor[j]), se=float(ses[j]), F=F[j],
                  p=float(2*student_t.sf(abs(b[j]*factor[j]/ses[j]), G-1))) for j, v in enumerate(xs)}
    if kx == 1:
        ref = fit2sls(raw[:, 0], raw[:, 1], raw[:, 2], w, f1, f2, cl.copy(), singleton=False, correction_n=correction_n)
        for key in ['beta', 'se', 'F']:
            require(np.isclose(out[xs[0]][key], ref[key], rtol=1e-7, atol=1e-12), 'Source fit2sls differs: '+key)
    return out, pd.DataFrame(psi, index=labels, columns=xs)
