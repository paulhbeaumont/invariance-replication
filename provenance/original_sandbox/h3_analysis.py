"""Approved sandbox analysis: separate build and estimate commands; no production writes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

BOX = Path(__file__).resolve().parents[1]
ROOT = BOX.parents[1]
APP = BOX.parent / '4a_PaperApplication'
os.environ.setdefault('INVARIANCE_DROPBOX_ROOT', str(ROOT))
sys.path.insert(0, str(APP / 'code/lib'))
from baseline_io import DATA, locked
from baseline_math import demean, fit2sls, ssc_k

CACHE = BOX / 'data/h3_analysis'
OUT = BOX / 'output/h3_analysis'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + '\n')


def certified(folder, name, inputs):
    cert = read(folder / 'certificate.json')
    require(cert['status'] == 'PASS', 'Uncertified source: ' + str(folder))
    require(cert['baseline_lock'] == locked()[2], 'Source baseline lock differs')
    p = folder / name
    require(sha(p) == cert['files'][name], 'Source changed: ' + str(p))
    inputs[str(p)] = sha(p)
    inputs[str(folder / 'certificate.json')] = sha(folder / 'certificate.json')
    return pd.read_parquet(p)


def build():
    require(not (CACHE / 'certificate.json').exists(), 'Refusing to replace completed cache')
    CACHE.mkdir(parents=True, exist_ok=True)
    inputs = {}
    countyroot = DATA / 'analysis/disaggregation_h234'
    statsroot = DATA / 'analysis/naics6_log_h234'
    annualroot = DATA / 'intermediate/naics6_residual'
    for folder, producer in [(countyroot, 'disaggregation.py'), (statsroot, 'naics6_log.py'), (annualroot, 'naics6_cells.py')]:
        cert = read(folder / 'certificate.json')
        require(cert['builder_sha256'] == sha(APP / 'code/build' / producer), 'Certified builder changed')
    f = certified(countyroot, 'county_h3.parquet', inputs)
    s = certified(statsroot, 'naics6_h3.parquet', inputs)
    require(np.array_equal(f[['ccg', 'year']], s[['ccg', 'year']]), 'Statistics keys differ')
    require(len(f) == 110088 and f.ccg.nunique() == 3058, 'County support differs')
    require(f.groupby('year').size().eq(3058).all(), 'Unbalanced support')
    require((f[['E', 'E_b', 'P_b']] > 0).all().all(), 'Nonpositive endpoints')
    require(np.allclose(f.log_y, np.log(f.E / f.E_b), atol=1e-12), 'Saved log outcome differs')
    f.to_parquet(CACHE / 'county.parquet', index=False)
    s.to_parquet(CACHE / 'stats.parquet', index=False)
    cfg = read(APP / 'config/aggregation.json')
    for name, record in cfg['sources'].items():
        p = APP / 'config' / name
        require(sha(p) == record['sha256'], 'Geography hash changed')
        inputs[str(p)] = sha(p)
    x = pd.read_csv(APP / 'config/aggregation_county_ccg_crosswalk.csv', dtype=str)
    x = x[x.ccg_id.ne('unplaceable')].copy()
    x['ccg'] = x.ccg_id.astype(int)
    b = pd.read_csv(APP / 'config/aggregation_county_bea_area_2004.csv', dtype=str)
    m = x.merge(b, on='county_fips5', how='left', validate='many_to_one', sort=False)
    maps = {k: m.dropna(subset=[v]).drop_duplicates('ccg').set_index('ccg')[v]
            for k, v in [('czone', 'ccz_id'), ('cea', 'cea_code'), ('ea', 'ea_code')]}
    checks = []
    for rung in ('czone', 'cea', 'ea', 'state'):
        g = f.copy()
        g['unit'] = g.state if rung == 'state' else g.ccg.map(maps[rung])
        require(g.unit.notna().all(), 'Unmapped county')
        g['unit'] = pd.factorize(g.unit, sort=True)[0]
        g['weight'] = g.E_b / g.groupby(['unit', 'year']).E_b.transform('sum')
        g['std_y'] = (g.E - g.E_b) / g.E_b
        for v in ['x', 'z', 'std_y']:
            g['w_' + v] = g.weight * g[v]
        a = g.groupby(['unit', 'year'], as_index=False).agg(
            E=('E', 'sum'), E_b=('E_b', 'sum'), x_agg=('w_x', 'sum'), z_agg=('w_z', 'sum'),
            reconstructed=('w_std_y', 'sum'), weight_sum=('weight', 'sum'))
        a['std_y'] = (a.E - a.E_b) / a.E_b
        a['log_y'] = np.log(a.E / a.E_b)
        a['weight'] = 1.0
        error = float(abs(a.std_y - a.reconstructed).max())
        require(error < 1e-12 and np.allclose(a.weight_sum, 1, atol=1e-12), 'Aggregate reconstruction failed')
        require(np.allclose(a.groupby('year')[['E', 'E_b']].sum(), f.groupby('year')[['E', 'E_b']].sum()), 'Totals differ')
        g = g.merge(a[['unit', 'year', 'x_agg', 'z_agg']], on=['unit', 'year'], validate='many_to_one')
        require(len(g) == len(f), 'Aggregation changed county support')
        g.to_parquet(CACHE / ('county_' + rung + '.parquet'), index=False)
        a.to_parquet(CACHE / ('aggregate_' + rung + '.parquet'), index=False)
        checks.append(dict(rung=rung, n_units=int(a.unit.nunique()), reconstruction_error=error))
    # Match only continuing cells, but independently verify totals of ALL source cells.
    # Equivalent to the co-author's sparse endpoint outer join; entrants/exiters
    # are recovered from total endpoints minus continuing endpoints, not dropped.
    parts = []
    for year, g in f.groupby('year', sort=True):
        current = certified(annualroot, 'y%d.parquet' % year, inputs)[['ccg', 'code', 'emp']]
        lagged = certified(annualroot, 'y%d.parquet' % (year - 3), inputs)[['ccg', 'code', 'emp']]
        for frame, target in [(current, 'E'), (lagged, 'E_b')]:
            require(not frame.duplicated(['ccg', 'code']).any(), 'Duplicate annual cells')
            total = frame.groupby('ccg').emp.sum().reindex(g.ccg).to_numpy()
            require(np.allclose(total, g[target], rtol=1e-12, atol=1e-8), 'Cell endpoint conservation failed')
        c = current.merge(lagged, on=['ccg', 'code'], suffixes=('_cur', '_lag'), validate='one_to_one')
        c = c.loc[c.emp_cur.gt(0) & c.emp_lag.gt(0)].copy()
        c['lag_log'] = c.emp_lag * (np.log(c.emp_cur) - np.log(c.emp_lag))
        u = c.groupby('ccg').agg(E_c=('emp_cur', 'sum'), E_b_c=('emp_lag', 'sum'),
                                 lag_log=('lag_log', 'sum'), count=('code', 'size')).reindex(g.ccg).fillna(0).reset_index()
        u['year'] = year
        parts.append(u)
        print('Verified employment endpoints and continuing cells:', year, flush=True)
    d = f.merge(pd.concat(parts, ignore_index=True), on=['ccg', 'year'], validate='one_to_one')
    keep = d['count'].gt(0)
    require(int((~keep).sum()) == 11, 'Natural log sample changed')
    require(np.array_equal(d['count'], s.log_count), 'Survivor cell counts differ')
    d['lag_mass'] = d.E_b_c / d.E_b
    d['mean_log'] = d.lag_log / d.E_b_c.where(keep)
    require(np.allclose(d.lag_mass, s.lag_mass, atol=1e-12), 'Lagged survivor masses differ')
    require(np.allclose(d.loc[keep, 'mean_log'], s.loc[keep, 'lag_mean'], atol=1e-12), 'Saved weighted log outcome differs')
    d['entry'] = -np.log(d.E_c.where(keep) / d.E)
    d['exit'] = np.log(d.E_b_c.where(keep) / d.E_b)
    d['jensen'] = np.log(d.E_c.where(keep) / d.E_b_c.where(keep)) - d.mean_log
    error = float(abs(d.loc[keep, 'log_y'] - d.loc[keep, ['entry', 'exit', 'jensen', 'mean_log']].sum(axis=1)).max())
    require(error < 1e-11, 'Log identity failed')
    require(d.loc[keep, 'jensen'].min() > -1e-12, 'Negative Jensen term')
    d.to_parquet(CACHE / 'decomposition.parquet', index=False)
    files = {p.name: sha(p) for p in CACHE.glob('*.parquet')}
    write(CACHE / 'certificate.json', dict(status='PASS', inputs=inputs, files=files,
          builder_sha256=sha(__file__), checks=checks, log_identity_error=error,
          excluded_county_years=11, n_counties=3058, n_county_years=len(f)))


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


def estimate():
    cert = read(CACHE / 'certificate.json')
    require(cert['status'] == 'PASS', 'Uncertified sandbox input')
    require(cert['builder_sha256'] == sha(__file__), 'Sandbox builder changed since build')
    for name, digest in cert['files'].items():
        require(sha(CACHE/name) == digest, 'Sandbox input changed')
    for name, digest in cert['inputs'].items():
        require(sha(name) == digest, 'Original source changed')
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT/'estimates.json').exists(), 'Refusing to overwrite completed estimates')
    d = pd.read_parquet(CACHE/'decomposition.parquet').assign(weight=1., unit=lambda x: x.ccg)
    bfull, pfull = fit(d, 'log_y', ['x'], ['z'])
    common = d.loc[d['count'].gt(0)].copy()
    fits = {}; psis = {}
    for key, col in [('baseline_common', 'log_y'), ('normalized', 'mean_log'), ('entry', 'entry'), ('exit', 'exit'), ('jensen', 'jensen')]:
        r, p = fit(common, col, ['x'], ['z']); fits[key] = r['x']; psis[key] = p.x
    original = common.assign(weight=common.lag_mass)
    r, p = fit(original, 'mean_log', ['x'], ['z'], correction_n=int(common['count'].sum()))
    fits['original'] = r['x']; psis['original'] = p.x
    fits['baseline_full'] = bfull['x']; psis['baseline_full'] = pfull.x
    saved_path = APP/'results/coefficients/log_naics6.csv'
    require(read(APP/'results/manifests/log_naics6.json')['coefficients_sha256'] == sha(saved_path), 'Saved Table5 reference changed')
    saved = pd.read_csv(saved_path).query("horizon==3 and weight=='lag'").iloc[0]
    for key in ['beta', 'se']:
        require(np.isclose(saved[key], fits['original'][key], rtol=1e-8, atol=1e-12), 'Table5 column4 reproduction failed')
    require(np.isclose(saved.first_stage_F, fits['original']['F'], rtol=1e-7), 'Saved F differs')
    # Reference the actual displayed no-population Table 5 county benchmark too.
    displayed = pd.read_csv(BOX/'output/disaggregation_no_population/displayed_estimates.csv')
    refrows = displayed.loc[(displayed.horizon == 3) & (displayed.rung == 'county') & (displayed.growth == 'log')]
    require(len(refrows) > 0, 'Missing independent county reference')
    require(np.allclose(refrows.beta, fits['baseline_full']['beta'], rtol=1e-8), 'County Table5 reproduction failed')
    components = []
    for key in ['entry', 'exit', 'jensen', 'renormalization', 'sample']:
        if key in fits:
            value = fits[key]['beta']; influence = psis[key]
        else:
            a, b = ('normalized', 'original') if key == 'renormalization' else ('baseline_full', 'baseline_common')
            require(psis[a].index.equals(psis[b].index), 'Cluster labels differ')
            value = fits[a]['beta']-fits[b]['beta']; influence = psis[a]-psis[b]
        components.append(dict(component=key, beta=value, se=float(np.linalg.norm(influence))))
    gap = fits['baseline_full']['beta']-fits['original']['beta']
    error = float(sum(v['beta'] for v in components)-gap)
    require(abs(error) < 1e-12, 'Decomposition does not sum to target gap')
    pd.DataFrame(components).to_csv(OUT/'decomposition.csv', index=False)
    rows = []
    for rung in ['czone', 'cea', 'ea', 'state']:
        c = pd.read_parquet(CACHE/('county_'+rung+'.parquet'))
        a = pd.read_parquet(CACHE/('aggregate_'+rung+'.parquet'))
        for growth in ['std', 'log']:
            y = growth+'_y'
            nb, pn = fit(c, y, ['x'], ['z'])
            ab, pa = fit(a, y, ['x_agg'], ['z_agg'], fe='unit')
            ub, pu = fit(c, y, ['x', 'x_agg'], ['z', 'z_agg'])
            require(pn.index.equals(pa.index) and pn.index.equals(pu.index), 'Aggregation clusters differ')
            gap = ab['x_agg']['beta']-nb['x']['beta']
            gap_se = float(np.linalg.norm(pa.x_agg-pn.x))
            residual = ab['x_agg']['beta']-ub['x']['beta']-ub['x_agg']['beta']
            row = dict(rung=rung, growth=growth, n_counties=int(c.ccg.nunique()), n_county_years=len(c),
                       n_units=int(a.unit.nunique()), n_aggregate_years=len(a), gap=gap, gap_se=gap_se,
                       gap_p=float(2*student_t.sf(abs(gap/gap_se), a.unit.nunique()-1)), identity_residual=residual)
            for label, vals in [('county', nb['x']), ('aggregate', ab['x_agg']), ('augmented', ub['x']), ('exposure', ub['x_agg'])]:
                row.update({label+'_'+k: v for k, v in vals.items()})
            rows.append(row)
            print(rung, growth, 'county, aggregate, exposure:', [round(row[k]*1e5, 3) for k in ['county_beta', 'aggregate_beta', 'exposure_beta']], flush=True)
    pd.DataFrame(rows).to_csv(OUT/'aggregation.csv', index=False)
    write(OUT/'estimates.json', dict(status='PASS', fits=fits, components=components,
          decomposition_error=error, decomposition_gap=fits['baseline_full']['beta']-fits['original']['beta'],
          source_table5_reproduced=True, county_reference_reproduced=True,
          estimator_sha256=sha(__file__), cache_certificate_sha256=sha(CACHE/'certificate.json'),
          first_stage_F='Clustered joint Wald F, not a conditional weak-IV statistic for augmented equations',
          aggregation=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['build', 'estimate'])
    args = parser.parse_args()
    globals()[args.stage]()
