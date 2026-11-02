"""Build nonnegative NAICS6 plus parent-residual employment cells.

Scope: annual employment construction only, no regressions.
Input: frozen NAICS4 parents and hash-pinned EFSY detail.
Overfull parents scale all six-digit children proportionally to the parent.
Other parents retain detail and carry the unassigned balance as a residual.
Absent annual cells represent zero only within the frozen reported-county frame.
"""
from pathlib import Path
import sys
import json
import duckdb
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from baseline_io import APP, DATA, locked, require, sha, write_json


def split_parent(parent, six):
    """Return cells and parent-level scaling accounting; never clip raw negatives."""
    require(six.emp.notna().all() and np.isfinite(six.emp).all() and six.emp.ge(0).all(), 'Invalid six-digit employment')
    require(parent.emp.notna().all() and np.isfinite(parent.emp).all() and parent.emp.ge(0).all(), 'Invalid parent employment')
    require(not parent.duplicated(['ccg', 'parent']).any(), 'Duplicate parents')
    require(not six.duplicated(['ccg', 'code']).any(), 'Duplicate six-digit cells')
    totals = six.groupby(['ccg', 'parent'], sort=False).emp.sum().rename('six_total')
    p = parent.merge(totals, on=['ccg', 'parent'], how='left', validate='one_to_one')
    p['six_total'] = p.six_total.fillna(0)
    p['factor'] = np.divide(p.emp, p.six_total, out=np.ones(len(p)), where=p.six_total.gt(0))
    p['factor'] = np.minimum(1, p.factor)
    s = six.merge(p[['ccg', 'parent', 'factor']], on=['ccg', 'parent'], how='left', validate='many_to_one')
    require(s.factor.notna().all(), 'Six-digit cell absent from frozen parent support')
    s['raw_emp'] = s.emp
    s['emp'] = s.emp * s.factor
    sums = s.groupby(['ccg', 'parent'], sort=False).emp.sum().rename('assigned')
    p = p.merge(sums, on=['ccg', 'parent'], how='left', validate='one_to_one')
    p['assigned'] = p.assigned.fillna(0)
    p['balance'] = p.emp-p.assigned
    eps = 64*np.finfo(float).eps*np.maximum(p.emp, 1)
    require((p.balance >= -eps).all(), 'Negative residual beyond floating-point precision')
    # Roundoff-only balances go to the largest child, not a spurious tiny industry.
    tiny = p.balance.abs().le(eps) & p.assigned.gt(0)
    adj = p.loc[tiny, ['ccg', 'parent', 'balance']]
    largest = s.loc[s.groupby(['ccg', 'parent'], sort=False).emp.idxmax(), ['ccg', 'parent', 'code']]
    adj = adj.merge(largest, on=['ccg', 'parent'], validate='one_to_one').rename(columns={'balance': 'roundoff'})
    s = s.merge(adj[['ccg', 'code', 'roundoff']], on=['ccg', 'code'], how='left', validate='one_to_one')
    s['roundoff'] = s['roundoff'].fillna(0)
    s['emp'] += s['roundoff']
    p.loc[tiny, 'balance'] = 0
    require(p.balance.ge(0).all() and s.emp.ge(0).all(), 'Negative employment after reconciliation')
    residual = p[['ccg', 'parent', 'balance']].rename(columns={'balance': 'emp'})
    residual['code'] = 'R' + residual.parent.astype(str)
    residual['kind'] = 'parent_residual'
    s['kind'] = 'naics6'
    cells = pd.concat([s[['ccg','parent','code','kind','emp']], residual], ignore_index=True)
    require(not cells.duplicated(['ccg','code']).any(), 'Duplicate output cells')
    check = cells.groupby(['ccg','parent'], sort=False).emp.sum().reindex(pd.MultiIndex.from_frame(p[['ccg','parent']]))
    require(np.allclose(check.to_numpy(), p.emp, rtol=1e-13, atol=1e-10), 'Parent conservation failed')
    p['roundoff_abs'] = p[['ccg','parent']].merge(adj, on=['ccg','parent'], how='left').roundoff.fillna(0).abs().to_numpy()
    return cells, p


def main():
    _, sources, lock = locked()
    manifest = APP/'config/naics6_audit_sources.json'
    inventory = json.loads(manifest.read_text())
    for v in inventory['inputs'].values():
        require(sha(v['path']) == v['sha256'], 'Source hash mismatch')
    cw = pd.read_csv(inventory['inputs']['crosswalk']['path'], dtype=str)
    cw = cw[cw.ccg_id.ne('unplaceable')]
    crosswalk = pd.DataFrame({'fips':cw.county_fips5.astype(int), 'ccg':cw.ccg_id.astype(int)})
    require(not crosswalk.fips.duplicated().any(), 'Duplicate crosswalk')
    annual = DATA.parent/sources['snapshot_directory']/'source51_county_panel.parquet'
    require(sha(annual)==sources['inputs'][annual.name], 'County snapshot hash mismatch')
    counties = pd.read_parquet(annual, columns=['ccg','year','E'])
    units = pd.DataFrame({'ccg': sorted(counties.ccg.unique())})
    cert = json.loads((APP/'results/manifests/disaggregation_snapshot.json').read_text())
    require(cert['status']=='PASS' and cert['baseline_lock']==lock, 'Parent certificate mismatch')
    out = DATA/'intermediate/naics6_residual'
    out.mkdir(parents=True, exist_ok=False)
    files, audits, adjustments = {}, [], []
    with duckdb.connect() as con:
        con.register('crosswalk', crosswalk)
        con.register('units', units)
        for year in range(1978,2017):
            six = con.execute('''WITH src AS (
                SELECT *, CASE WHEN fipstate=11 AND fipscty=999 AND year BETWEEN 1976 AND 2003
                THEN 11001 ELSE fipstate*1000+fipscty END AS fips
                FROM read_parquet(?) WHERE year=?)
                SELECT c.ccg, naics12 AS code, CAST(substr(naics12,1,4) AS INTEGER) AS parent, sum(emp) AS emp
                FROM src JOIN crosswalk c USING(fips) JOIN units u ON c.ccg=u.ccg
                WHERE regexp_full_match(naics12,'[0-9]{6}')
                AND (fipscty<>999 OR (fipstate=11 AND year BETWEEN 1976 AND 2003))
                GROUP BY c.ccg,naics12''', [inventory['inputs']['efsy']['path'],year]).df()
            path = DATA/'intermediate/disaggregation_h234/naics4_alloc'/f'y{year}.parquet'
            require(sha(path)==cert['inputs'][f'naics4_alloc/{path.name}'], 'Parent input changed')
            p = pd.read_parquet(path, columns=['ccg','code','emp'])
            p = p[p.ccg.isin(units.ccg)].rename(columns={'code':'parent'})
            cells, accounting = split_parent(p, six)
            cells['year'] = year
            target = counties[counties.year.eq(year)].set_index('ccg').E
            summed = cells.groupby('ccg', sort=False).emp.sum().reindex(target.index)
            require(summed.notna().all() and np.allclose(summed,target,rtol=1e-13,atol=1e-9), 'County conservation failed')
            dest = out/f'y{year}.parquet'
            cells.sort_values(['ccg','code']).to_parquet(dest,index=False)
            saved = pd.read_parquet(dest)
            require(saved.equals(cells.sort_values(['ccg','code']).reset_index(drop=True)), 'Saved panel differs')
            files[dest.name] = sha(dest)
            significant = (accounting.six_total-accounting.emp) > (1e-9+1e-10*accounting.emp.abs())
            adjustments.append(accounting[accounting.factor.lt(1)].assign(year=year))
            audits.append(dict(year=year,n_counties=len(target),n_cells=len(cells),negative_cells=int(cells.emp.lt(0).sum()),
                scaled_parents=int(significant.sum()),numerical_only_scaled_parents=int((accounting.factor.lt(1)&~significant).sum()),
                employment_reduction=float(((accounting.six_total-accounting.emp).clip(lower=0)).sum()),
                residual_employment=float(cells.loc[cells.kind.eq('parent_residual'),'emp'].sum()),
                county_max_abs_gap=float((summed-target).abs().max()),roundoff_abs=float(accounting.roundoff_abs.sum())))
            print(json.dumps(audits[-1]),flush=True)
    pd.concat(adjustments,ignore_index=True).to_parquet(out/'parent_adjustments.parquet',index=False)
    pd.DataFrame(audits).to_csv(out/'annual_checks.csv',index=False)
    for name in ('parent_adjustments.parquet','annual_checks.csv'):
        files[name] = sha(out/name)
    for v in inventory['inputs'].values():
        require(sha(v['path'])==v['sha256'], 'Source changed during build')
    report = dict(status='PASS',baseline_lock=lock,files=files,builder_sha256=sha(__file__),
        sources_sha256=sha(manifest),years=[1978,2016],county_support=len(units),
        rule='Proportional reduction only in overfull parents; otherwise unchanged detail plus parent residual.',
        roundoff='Balances within 64 machine epsilons of parent scale assigned to largest child.',
        support='Annual sparse six-digit support plus all frozen parent residual cells; absent cells are zero within reported counties.',
        regressions_run=False,scaled_parent_years=sum(a['scaled_parents'] for a in audits),
        negative_cells=sum(a['negative_cells'] for a in audits),
        county_max_abs_gap=max(a['county_max_abs_gap'] for a in audits))
    write_json(out/'certificate.json',report)
    write_json(APP/'results/manifests/naics6_cells.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))


if __name__=='__main__':
    main()
