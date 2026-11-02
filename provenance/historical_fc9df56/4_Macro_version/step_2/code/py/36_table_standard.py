#!/usr/bin/env python3
"""Standard growth. eq standard_entry_decomp: the wedge is entry alone.

    Delta^s S_it = sum_{m in M^-} w^-_im Delta^s S_imt + E_it/S_i,t-h

M^- is cells with positive lagged employment, so exiters are inside it at -1 and
there is no exit term. Inadmissible as an answer (A.1.5); run capped too because
it is the one rate anybody caps, each level on its own IQR5-by-year distribution
(A.3.3's trap). The capped residual is the construction error.
"""
import argparse, importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve(); sys.path.insert(0, str(H.parent))
_s = importlib.util.spec_from_file_location("md", H.parent / "mpgr_disagg.py")
md = importlib.util.module_from_spec(_s); _s.loader.exec_module(md)


def one(h, rung, share, lev, period):
    p = md.Panel(h, rung, share, lev, period=period)
    cy, cur, lag, n = p.cy, p.cur, p.lag, p.n
    ok = p.lag_c > 0
    m = (lag > 0) & ok[cy]
    g = (cur[m] - lag[m]) / lag[m]
    w = lag[m] / p.lag_c[cy[m]]
    gc = np.where(ok, (p.cur_c - p.lag_c) / np.where(ok, p.lag_c, 1.0), 0.0)
    ent = (lag <= 0) & (cur > 0)
    E = np.bincount(cy[ent], weights=cur[ent], minlength=n) / np.where(ok, p.lag_c, 1.0)

    rec = np.bincount(cy[m], weights=w * g, minlength=n) + E
    md.check(float(np.abs((rec - gc)[ok]).max()) < 1e-11,
             f"eq standard_entry_decomp holds (h={h}, {md.PRETTY[rung]})",
             f"max |gap| = {float(np.abs((rec - gc)[ok]).max()):.2e}")

    W = md.t29.winsor4_by_year
    B = lambda y: p.fit_county(y, ok)["beta"]
    K = lambda y, ww: md.t28.fit2sls(y, p.shock[cy][m], p.z[cy][m], ww,
                                     p.ccg[cy][m], p.yr[cy][m], p.ccg[cy][m],
                                     singleton=False)["beta"]
    bc, bw, be = B(gc), K(g, w), B(E)
    md.check(abs(bw + be - bc) < 1e-9,
             f"UNCAPPED gap is entry alone (h={h}, {md.PRETTY[rung]})",
             f"|residual| = {abs(bw + be - bc):.2e}")
    gcap, gkcap, Ecap = W(gc, p.yr, lev), W(g, p.yr[cy][m], lev), W(E, p.yr, lev)
    bcc, bwc, bec = B(gcap), K(gkcap, w), B(Ecap)
    return dict(h=h, rung=rung.replace("_alloc", ""), county=bc, cell_w=bw,
                cell_eq=K(g, np.ones(int(m.sum()))), entry=be, gap=bw - bc,
                county_cap=bcc, cell_cap=bwc, gap_cap=bwc - bcc,
                constr_err=bcc - bwc - bec,
                n_cap_cty=int((gc[ok] != gcap[ok]).sum()),
                n_cap_cell=int((g != gkcap).sum()))


a = argparse.ArgumentParser()
a.add_argument("--horizon", type=int, nargs="+", default=[2, 3, 4])
a.add_argument("--period", default="pre", choices=["pre", "post"])
a.add_argument("--rung", nargs="+", default=list(md.RUNGS))
a.add_argument("--share", default="fixed5")
a.add_argument("--level", type=float, default=5.0)
g = a.parse_args()
md.dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
tee = md.Tee(md.dc.LOG_DIR / f"36_table_standard_{g.period}_{g.share}.log")
import pyfixest as pf
md.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100_000)
t0 = time.time()
rows = [one(h, r, g.share, g.level, g.period) for r in g.rung for h in g.horizon]

print(f"\n  STANDARD GROWTH, {g.period}-{md.w22.NAICS_BREAK}, z_income_{g.share}, "
      f"IQR{g.level:g} by year\n")
print(f"  {'rung':8s}{'h':>3s}{'county':>8s}{'cell w-':>9s}{'gap':>8s}"
      f"{'-entry':>8s}{'cell EW':>12s}{'  ||':4s}{'cty cap':>9s}{'cell cap':>9s}"
      f"{'gap cap':>8s}{'CONSTR ERR':>12s}")
for r in rows:
    print(f"  {r['rung']:8s}{r['h']:>3d}{r['county']:8.2f}{r['cell_w']:9.2f}"
          f"{r['gap']:+8.2f}{-r['entry']:+8.2f}{r['cell_eq']:12,.0f}{'  ||':4s}"
          f"{r['county_cap']:9.2f}{r['cell_cap']:9.2f}{r['gap_cap']:+8.2f}"
          f"{r['constr_err']:+12.2f}")
out = md.RESULT_DIR / f"coeff_table_standard_{g.period}_{g.share}.csv"
pd.DataFrame(rows).assign(period=g.period).to_csv(
    out, index=False, float_format="%.15g")
print(f"\n  wrote {out.relative_to(md.REPO_ROOT)}")
print(f"  {len(md.w22.failures)} checks failed, {time.time() - t0:.1f}s")
tee.close()
