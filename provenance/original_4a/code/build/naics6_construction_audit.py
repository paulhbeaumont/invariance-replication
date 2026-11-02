"""Audit six-digit detail against frozen four-digit employment, without repairs.

Scope: construction audit only; no analysis cache or regression is produced.
Geography follows producer 11: CCG crosswalk and DC statewide residual recode.
Positive-child sums are diagnostic bounds, not a proposal to clip negatives.
"""
from pathlib import Path
import argparse
import json
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from baseline_io import APP, DATA, locked, require, sha, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    _, sources, lock = locked()
    inventory = json.loads(args.sources.read_text())
    for item in inventory['inputs'].values():
        require(sha(item['path']) == item['sha256'], f"Input hash mismatch: {item['path']}")
    raw = inventory['inputs']['efsy']['path']
    crosswalk = pd.read_csv(inventory['inputs']['crosswalk']['path'], dtype=str)
    require(not crosswalk.county_fips5.duplicated().any(), 'Duplicate crosswalk keys')
    crosswalk = crosswalk[crosswalk.ccg_id.ne('unplaceable')]
    crosswalk = pd.DataFrame({'fips': crosswalk.county_fips5.astype(int),
                              'ccg': crosswalk.ccg_id.astype(int)})
    annual = DATA.parent / sources['snapshot_directory'] / 'source51_county_panel.parquet'
    require(sha(annual) == sources['inputs'][annual.name], 'County snapshot hash mismatch')
    counties = pd.read_parquet(annual, columns=['ccg', 'year', 'E'])
    units = pd.DataFrame({'ccg': sorted(counties.ccg.unique())})
    cert = json.loads((APP / 'results/manifests/disaggregation_snapshot.json').read_text())
    require(cert['status'] == 'PASS', 'Industry snapshot not certified')
    args.output.mkdir(parents=True, exist_ok=False)
    con = duckdb.connect()
    con.register('crosswalk', crosswalk)
    con.register('units', units)
    rows, issues, negatives = [], [], []
    with con:
        for year in range(1978, 2017):
            frame = con.execute('''
                WITH source AS (
                    SELECT *, CASE WHEN fipstate=11 AND fipscty=999
                        AND year BETWEEN 1976 AND 2003 THEN 11001
                        ELSE fipstate*1000+fipscty END AS fips
                    FROM read_parquet(?) WHERE year=?
                )
                SELECT c.ccg, naics12, sum(emp) AS emp
                FROM source s JOIN crosswalk c USING(fips) JOIN units u ON c.ccg=u.ccg
                WHERE fipscty<>999 OR (fipstate=11 AND year BETWEEN 1976 AND 2003)
                GROUP BY c.ccg, naics12
            ''', [raw, year]).df()
            require(frame.emp.notna().all() and np.isfinite(frame.emp).all(), 'Invalid source employment')
            require(set(frame.ccg) == set(units.ccg), f'Missing source county in {year}')
            six = frame[frame.naics12.str.fullmatch(r'[0-9]{6}')].copy()
            six['code'] = six.naics12.str[:4].astype(int)
            neg = six[six.emp.lt(0)].assign(year=year)
            negatives.append(neg)
            six['positive_emp'] = six.emp.clip(lower=0)
            children = six.groupby(['ccg', 'code'], sort=False, as_index=False).agg(
                six_net=('emp', 'sum'), six_positive=('positive_emp', 'sum'))
            path = DATA / 'intermediate/disaggregation_h234/naics4_alloc' / f'y{year}.parquet'
            require(sha(path) == cert['inputs'][f'naics4_alloc/{path.name}'], 'Parent snapshot hash mismatch')
            parent = pd.read_parquet(path, columns=['ccg', 'code', 'emp'])
            parent = parent[parent.ccg.isin(units.ccg)]
            require(not parent.duplicated(['ccg', 'code']).any(), 'Duplicate parent keys')
            require(parent.emp.notna().all() and parent.emp.ge(0).all(), 'Invalid frozen parent')
            target = counties[counties.year.eq(year)].set_index('ccg').E
            sums = parent.groupby('ccg', sort=False).emp.sum().reindex(target.index)
            require(np.allclose(sums, target, rtol=1e-9, atol=1e-9), 'Frozen parent/county mismatch')
            comparison = parent.merge(children, on=['ccg', 'code'], how='outer', validate='one_to_one', indicator=True)
            require(not ((comparison._merge == 'right_only') & comparison.six_net.ne(0)).any(), 'Six-digit industry absent from parent support')
            comparison[['emp', 'six_net', 'six_positive']] = comparison[['emp', 'six_net', 'six_positive']].fillna(0)
            comparison['residual_net'] = comparison.emp - comparison.six_net
            comparison['residual_positive'] = comparison.emp - comparison.six_positive
            tolerance = 1e-9 + 1e-10 * comparison.emp.abs()
            bad = comparison.residual_positive.lt(-tolerance)
            issues.append(comparison[bad].assign(year=year).drop(columns='_merge'))
            total = float(parent.emp.sum())
            rows.append(dict(year=year, counties=len(units), county_employment=total,
                six_digit_cells=len(six), negative_six_cells=len(neg),
                negative_six_employment=float(neg.emp.sum()),
                overfull_parents=int(bad.sum()), overfull_counties=int(comparison.loc[bad, 'ccg'].nunique()),
                excess_employment=float(-comparison.loc[bad, 'residual_positive'].sum()),
                net_overfull_parents=int(comparison.residual_net.lt(-tolerance).sum()),
                positive_residual_employment=float(comparison.residual_positive.clip(lower=0).sum()),
                six_net_employment=float(six.emp.sum()),
                county_max_abs_gap=float((sums-target).abs().max())))
            print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(args.output / 'annual_audit.csv', index=False)
    pd.concat(issues, ignore_index=True).to_parquet(args.output / 'overfull_parents.parquet', index=False)
    pd.concat(negatives, ignore_index=True).to_parquet(args.output / 'negative_six_cells.parquet', index=False)
    report = dict(audit_status='COMPLETE', construction_status='REQUIRES_DECISION' if any(
        r['negative_six_cells'] or r['overfull_parents'] for r in rows) else 'FEASIBLE',
        years=[1978, 2016], baseline_lock=lock, source_inventory_sha256=sha(args.sources),
        script_sha256=sha(__file__), negative_six_cells=sum(r['negative_six_cells'] for r in rows),
        overfull_parent_years=sum(r['overfull_parents'] for r in rows),
        repairs_applied=False, regressions_run=False,
        outputs={f.name: sha(f) for f in args.output.iterdir()})
    for item in inventory['inputs'].values():
        require(sha(item['path']) == item['sha256'], 'Source changed during audit')
    write_json(args.output / 'summary.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
