"""Compare parsed nominal procurement with NS's independently supplied series.

Diagnostic only. No convention is selected and no regression input is written.
Compare construction inclusion and the transition-quarter assignment directly,
without adjusting amounts to fit the reference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


STATES = dict(zip(
    [1,2,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,53,54,55,56],
    'AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY'.split()))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    ns_path = args.project / '1_current_paper/_data/source/NS2014_replication/DataPrograms/coded-data/fiscal_stimulus_coded.dta'
    directory = args.project / '1_current_paper/_data/temp/dcads_year'
    paths = [directory / f'fy{year}.parquet' for year in range(1969, 2007)] + [directory / 'fy1976tq.parquet']
    hashes = {str(p): sha(p) for p in [ns_path, *paths]}
    ns = pd.read_stata(ns_path, columns=['state','year','totalpma'], convert_categoricals=False)
    ns = ns.loc[ns.state.isin(STATES.values()) & ns.year.between(1969,2006)].copy()
    ns['year'] = ns.year.astype(int)
    assert not ns.duplicated(['state','year']).any()
    assert ns.totalpma.notna().all() and len(ns) == 51*38
    with duckdb.connect() as con:
        raw = con.execute('''SELECT fiscal_year, state_fips2, geo_class,
            starts_with(trim(coalesce(fsc, '')), 'Y') AS construction,
            dollars_unit, sum(dollars) AS amount, count(*) AS actions,
            count(*) FILTER (WHERE dollars IS NULL) AS missing_amount
            FROM read_parquet(?, union_by_name=true) GROUP BY ALL''', [[str(p) for p in paths]]).df()
    assert set(raw.dollars_unit) <= {'dollars','thousands'}, set(raw.dollars_unit)
    assert raw.missing_amount.sum() == 0
    raw['nominal'] = raw.amount * np.where(raw.dollars_unit.eq('thousands'),1000.,1.)
    raw['state'] = pd.to_numeric(raw.state_fips2,errors='coerce').map(STATES)
    parts = []
    summaries = []
    for scope in ('state_mapped', 'county_only'):
        for construction in (True,False):
            for quarter in ('exclude','1976','1977'):
                d = raw.loc[raw.state.notna()].copy()
                if scope == 'county_only':
                    d = d.loc[d.geo_class.eq('county')]
                if not construction:
                    d = d.loc[~d.construction]
                if quarter == 'exclude':
                    d = d.loc[d.fiscal_year.ne('1976tq')]
                else:
                    d.loc[d.fiscal_year.eq('1976tq'),'fiscal_year'] = quarter
                d['year'] = d.fiscal_year.astype(int)
                total = d.groupby(['state','year'],as_index=False,sort=False).nominal.sum()
                comparison = ns.merge(total,on=['state','year'],how='left',validate='one_to_one')
                assert comparison.nominal.notna().all()
                comparison['scope'] = scope
                comparison['construction_included'] = construction
                comparison['transition_quarter'] = quarter
                comparison['difference'] = comparison.nominal-comparison.totalpma
                comparison['relative_difference'] = comparison.difference/comparison.totalpma
                parts.append(comparison)
                for start,end in ((1969,2006),(1976,1976),(1977,1977),(1985,2006)):
                    q = comparison.loc[comparison.year.between(start,end)]
                    summaries.append(dict(scope=scope,construction_included=construction,transition_quarter=quarter,
                        first_year=start,last_year=end,n_state_years=len(q),
                        nominal_total=float(q.nominal.sum()), ns_total=float(q.totalpma.sum()),
                        exact_within_one_dollar=int(q.difference.abs().le(1).sum()),
                        median_absolute_relative_difference=float(q.relative_difference.abs().median()),
                        weighted_absolute_relative_difference=float(q.difference.abs().sum()/q.totalpma.abs().sum())))
    assert all(sha(Path(p))==h for p,h in hashes.items()), 'input changed during audit'
    args.output.mkdir(parents=True,exist_ok=False)
    out=pd.concat(parts,ignore_index=True)
    out.to_csv(args.output/'state_year_comparison.csv',index=False)
    summary=pd.DataFrame(summaries)
    summary.to_csv(args.output/'summary.csv',index=False)
    raw.to_parquet(args.output/'parsed_source_totals.parquet',index=False)
    report=dict(status='diagnostic_not_convention_certification',source_sha256=hashes,
                script_sha256=sha(Path(__file__)),comparison_rows=len(out),
                source_amount_not_state_mapped=float(raw.loc[raw.state.isna(),'nominal'].sum()))
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    assert len(pd.read_csv(args.output/'state_year_comparison.csv'))==len(out)
    print(summary.loc[summary.first_year.eq(1976)].to_string(index=False))


if __name__ == '__main__':
    main()
