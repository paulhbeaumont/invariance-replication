"""Read-only source diagnostic; outputs are not certified estimation inputs.

Run class: cache_rebuild preparation. No analysis cache is available.
The h=2 endpoint checks describe a candidate, not an approved specification.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--efsy', type=Path, required=True)
    parser.add_argument('--crosswalk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.project / '1_current_paper/_data/temp'
    paths = {
        'population': base / 'county_panel/population_ccg_year.parquet',
        'employment': base / 'employment/efsy_ccg_year.parquet',
        'contracts': base / 'county_panel/contracts_ccg_year.parquet',
        'pipeline_contracts': args.project / '_data/temp/pipeline/contracts_ccg_year.parquet',
        'raw_efsy': args.efsy,
        'crosswalk': args.crosswalk,
    }
    before = {key: digest(path) for key, path in paths.items()}
    report = {'status': 'diagnostic_only_not_certified', 'inputs': {
        key: {'path': str(path), 'sha256': before[key]} for key, path in paths.items()
    }}
    cw = pd.read_csv(args.crosswalk, dtype=str)
    valid = cw.loc[cw.ccg_id.ne('unplaceable')].copy()
    valid['ccg'] = valid.ccg_id.astype(int)
    valid['fips'] = valid.county_fips5.astype(int)
    assert not valid.fips.duplicated().any()
    names = valid.drop_duplicates('ccg').set_index('ccg').ccg_name.fillna('Name absent from crosswalk')
    universe = set(names.index)
    report['crosswalk_units'] = len(universe)
    with duckdb.connect() as con:
        frames = {key: con.execute('SELECT * FROM read_parquet(?)', [str(paths[key])]).df()
                  for key in ('population', 'employment', 'contracts', 'pipeline_contracts')}
        raw = con.execute('''SELECT fipstate, fipscty, year, sum(emp) AS raw_emp,
            count(*) AS source_rows, count(*) FILTER (WHERE emp IS NULL) AS null_rows,
            count(*) FILTER (WHERE emp < 0) AS negative_rows,
            sum(CASE WHEN emp < 0 THEN emp ELSE 0 END) AS negative_emp
            FROM read_parquet(?) WHERE year BETWEEN 1975 AND 2018
            GROUP BY fipstate, fipscty, year''', [str(args.efsy)]).df()
    report['cache_support'] = {}
    for key, frame in frames.items():
        assert not frame.duplicated(['ccg', 'year']).any(), key
        missing = sorted(universe - set(frame.ccg))
        report['cache_support'][key] = {
            'rows': len(frame), 'units': int(frame.ccg.nunique()),
            'first_year': int(frame.year.min()), 'last_year': int(frame.year.max()),
            'missing_units': [{'ccg': int(c), 'name': names[c]} for c in missing],
            'extra_units': sorted(int(c) for c in set(frame.ccg) - universe),
            'null_counts': {c: int(n) for c, n in frame.isna().sum().items()},
        }
    emp = frames['employment']
    flags = emp.loc[emp.emp.le(0) | emp.emp.isna() | emp.filled_zero].copy()
    flags['name'] = flags.ccg.map(names)
    flags.to_csv(args.output / 'employment_flagged_years.csv', index=False)
    report['employment_flags'] = {'zero_rows': int(emp.emp.eq(0).sum()),
        'negative_rows': int(emp.emp.lt(0).sum()), 'filled_zero_rows': int(emp.filled_zero.sum())}
    endpoint = emp[['ccg', 'year', 'emp']].rename(columns={'emp': 'end_emp'})
    lag = emp[['ccg', 'year', 'emp']].rename(columns={'emp': 'base_emp'})
    lag['year'] += 2
    endpoint = endpoint.merge(lag, on=['ccg', 'year'], how='left', validate='one_to_one')
    endpoint['bad'] = (endpoint[['end_emp', 'base_emp']].isna().any(axis=1)
                       | endpoint[['end_emp', 'base_emp']].lt(0).any(axis=1)
                       | (endpoint.end_emp + endpoint.base_emp).le(0))
    report['candidate_h2_endpoints'] = {}
    for start in (1977, 1985):
        sample = endpoint.loc[endpoint.year.between(start, 2006)]
        bad = sample.loc[sample.bad].copy()
        bad['name'] = bad.ccg.map(names)
        bad.to_csv(args.output / f'undefined_mpgr_{start}_2006.csv', index=False)
        report['candidate_h2_endpoints'][str(start)] = {
            'rows': len(sample), 'undefined_rows': len(bad),
            'affected_units': int(bad.ccg.nunique()),
            'one_endpoint_zero_defined': int(((sample.end_emp.eq(0) ^ sample.base_emp.eq(0)) & ~sample.bad).sum())}
    raw['fips'] = raw.fipstate * 1000 + raw.fipscty
    residual = raw.fipscty.eq(999)
    dc = residual & raw.fipstate.eq(11) & raw.year.between(1976, 2003)
    assert not raw.loc[~residual & raw.fipstate.eq(11), 'year'].isin(raw.loc[dc, 'year']).any()
    raw.loc[dc, 'fips'] = 11001
    raw['ccg'] = raw.fips.map(valid.set_index('fips').ccg)
    mapped = raw.loc[~residual | dc]
    mapped.loc[mapped.ccg.isna()].to_csv(args.output / 'unmapped_efsy.csv', index=False)
    raw.loc[residual & ~dc].to_parquet(args.output / 'unallocated_state_residuals.parquet', index=False)
    totals = mapped.groupby(['ccg', 'year'], sort=False, as_index=False).raw_emp.sum()
    comparison = emp.merge(totals, on=['ccg', 'year'], how='outer', validate='one_to_one', indicator=True)
    comparison['difference'] = comparison.emp - comparison.raw_emp
    comparison.to_parquet(args.output / 'employment_source_comparison.parquet', index=False)
    changed = comparison.loc[comparison.difference.abs().gt(1e-6)]
    report['raw_employment_comparison'] = {
        'negative_source_rows': int(raw.negative_rows.sum()),
        'negative_source_sum': float(raw.negative_emp.sum()),
        'null_source_rows': int(raw.null_rows.sum()),
        'unmapped_county_years': int(mapped.ccg.isna().sum()),
        'cached_rows_without_source_county': int(comparison._merge.eq('left_only').sum()),
        'source_rows_without_cache': int(comparison._merge.eq('right_only').sum()),
        'changed_county_years_tolerance_1e_6': len(changed),
        'cache_minus_raw_matched_total': float(comparison.difference.sum()),
        'maximum_absolute_difference': float(comparison.difference.abs().max()),
        'state_residual_employee_years_not_allocated': float(raw.loc[residual & ~dc, 'raw_emp'].sum()),
        'dc_residual_county_years_mapped': int(dc.sum()),
    }
    presence = comparison[['ccg', 'year', '_merge']].copy()
    presence['source_absent'] = presence._merge.ne('both')
    presence = presence[['ccg', 'year', 'source_absent']]
    endpoints = endpoint.merge(presence, on=['ccg', 'year'], validate='one_to_one')
    presence['year'] += 2
    endpoints = endpoints.merge(presence.rename(columns={'source_absent': 'base_source_absent'}),
                                on=['ccg', 'year'], how='left', validate='one_to_one')
    sample = endpoints.loc[endpoints.year.between(1985, 2006)].copy()
    sample['flagged'] = sample.bad | sample.source_absent | sample.base_source_absent.fillna(True)
    flagged = sample.loc[sample.flagged].copy()
    flagged['name'] = flagged.ccg.map(names)
    flagged.to_csv(args.output / 'source_or_mpgr_flagged_1985_2006.csv', index=False)
    report['candidate_h2_source_and_mpgr_flags_1985_2006'] = {
        'flagged_rows': len(flagged), 'flagged_units': int(flagged.ccg.nunique()),
        'flagged_rows_excluding_kalawao': int(flagged.ccg.ne(15005).sum()),
        'source_absent_at_either_endpoint': int((sample.source_absent | sample.base_source_absent.fillna(True)).sum()),
    }
    assert all(digest(path) == before[key] for key, path in paths.items()), 'Input changed during audit'
    report['input_hashes_stable'] = True
    report['script_sha256'] = digest(Path(__file__))
    target = args.output / 'audit.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    assert json.loads(target.read_text()) == report
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
