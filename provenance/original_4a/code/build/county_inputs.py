"""Freeze CBP inputs and build county employment as their unchanged cell sum.

Run class: cache_rebuild. Input cache: absent; source boundary is explicitly
constructed CBP cells, not raw-to-cell replication. Snapshot and build are
separate stages. A build reads only its application-owned frozen snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import duckdb
import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[2]
REPO = APP.parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    assert json.loads(path.read_text()) == value


def divisions():
    groups = [
        ('New England', 1, '09 23 25 33 44 50'),
        ('Middle Atlantic', 1, '34 36 42'),
        ('East North Central', 2, '17 18 26 39 55'),
        ('West North Central', 2, '19 20 27 29 31 38 46'),
        ('South Atlantic', 3, '10 11 12 13 24 37 45 51 54'),
        ('East South Central', 3, '01 21 28 47'),
        ('West South Central', 3, '05 22 40 48'),
        ('Mountain', 4, '04 08 16 30 32 35 49 56'),
        ('Pacific', 4, '02 06 15 41 53'),
    ]
    return pd.DataFrame([
        (int(state), i, name, region, ['Northeast', 'Midwest', 'South', 'West'][region-1])
        for i, (name, region, states) in enumerate(groups, 1) for state in states.split()
    ], columns=['state', 'division', 'division_name', 'region', 'region_name'])


def select_counties(universe, pop, employment, presence, spec):
    """Exclude whole units; preserve a reason and year for every failure."""
    pyears = range(spec['first_population_year'], spec['last_regression_year'] + 1)
    eyears = range(spec['first_employment_year'], spec['last_regression_year'] + 1)
    for frame in (pop, employment, presence):
        if frame.duplicated(['ccg', 'year']).any():
            raise ValueError('nonunique county-year input')
    def rectangle(frame, years):
        return frame.set_index(['ccg', 'year']).reindex(
            pd.MultiIndex.from_product([universe, years], names=['ccg', 'year'])).reset_index()
    p = rectangle(pop, pyears)
    e = rectangle(employment, eyears).merge(presence, on=['ccg', 'year'], how='left', validate='one_to_one')
    failures = []
    def mark(frame, condition, reason):
        part = frame.loc[condition, ['ccg', 'year']].copy()
        part['reason'] = reason
        failures.append(part)
    mark(p, ~np.isfinite(p.population) | p.population.le(0), 'population_missing_or_nonpositive')
    mark(e, ~np.isfinite(e.employment), 'employment_cell_total_missing')
    mark(e, e.employment.lt(0), 'employment_cell_total_negative')
    mark(e, e.source_rows.isna() | e.source_rows.le(0) | e.null_rows.gt(0), 'employment_source_missing')
    lag = e[['ccg', 'year', 'employment']].rename(columns={'employment': 'base_employment'})
    lag['year'] += spec['horizon']
    pair = e.merge(lag, on=['ccg', 'year'], how='left', validate='one_to_one')
    pair = pair.loc[pair.year.between(spec['first_regression_year'], spec['last_regression_year'])]
    mark(pair, (pair.employment + pair.base_employment).le(0), 'undefined_midpoint_denominator')
    ledger = pd.concat(failures, ignore_index=True).sort_values(['ccg', 'year', 'reason'])
    retained = sorted(set(universe) - set(ledger.ccg))
    if not retained:
        raise ValueError('whole-county exclusions leave no units')
    return p.loc[p.ccg.isin(retained)], e.loc[e.ccg.isin(retained)], ledger, retained


def freeze(project, raw_efsy, snapshot, spec):
    source = project / '1_current_paper/_data/temp'
    files = {
        'population.parquet': source / 'county_panel/population_ccg_year.parquet',
        'county_reference.parquet': source / 'employment/efsy_ccg_year.parquet',
        'raw_efsy.parquet': raw_efsy,
        'crosswalk.csv': REPO / '4_Macro_version/step_1/code/reference/county_ccg_crosswalk.csv',
        'employment_producer.py': REPO / '4_Macro_version/step_1/code/py/11_build_employment_cells.py',
        'population_producer.py': REPO / '4_Macro_version/step_1/code/py/12_build_county_income.py',
        'geography_producer.py': REPO / '4_Macro_version/step_1/code/py/14_build_instrument.py',
    }
    for year in range(spec['first_employment_year'], spec['last_regression_year'] + 1):
        files[f'cells/y{year}.parquet'] = source / f'employment/naics4_alloc/y{year}.parquet'
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    snapshot.mkdir(parents=True, exist_ok=False)
    records = []
    for relative, path in files.items():
        digest = sha(path)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(path) != digest or sha(target) != digest:
            raise RuntimeError(f'source or snapshot changed during copy: {path}')
        records.append({'relative_path': relative, 'source': str(path), 'sha256': digest})
    write_json(snapshot / 'manifest.json', {
        'status': 'frozen_constructed_inputs', 'files': records,
        'specification': spec, 'raw_to_cell_replication_certified': False,
    })


def build(snapshot, output, spec):
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    if manifest['status'] != 'frozen_constructed_inputs' or manifest['specification'] != spec:
        raise ValueError('snapshot specification mismatch')
    for row in manifest['files']:
        if sha(snapshot / row['relative_path']) != row['sha256']:
            raise ValueError(f"snapshot hash mismatch: {row['relative_path']}")
    output.mkdir(parents=True, exist_ok=False)
    cw = pd.read_csv(snapshot / 'crosswalk.csv', dtype=str)
    cw = cw.loc[cw.ccg_id.ne('unplaceable')].copy()
    cw['ccg'] = cw.ccg_id.astype(np.int32)
    cw['fips'] = cw.county_fips5.astype(np.int32)
    if cw.fips.duplicated().any():
        raise ValueError('county crosswalk is not a function')
    universe = sorted(cw.ccg.unique())
    with duckdb.connect() as con:
        cells = str(snapshot / 'cells/*.parquet')
        bad = con.execute('''SELECT count(*) FROM (
            SELECT ccg, code, year FROM read_parquet(?) GROUP BY ALL HAVING count(*) <> 1
            )''', [cells]).fetchone()[0]
        missing = con.execute('''SELECT count(*) FROM read_parquet(?)
            WHERE ccg IS NULL OR code IS NULL OR year IS NULL OR emp IS NULL OR NOT isfinite(emp)''', [cells]).fetchone()[0]
        if bad or missing:
            raise ValueError(f'cell key duplicates={bad}; invalid rows={missing}')
        employment = con.execute('''SELECT ccg, year, sum(emp) AS employment, count(*) AS n_cells
            FROM read_parquet(?) GROUP BY ccg, year''', [cells]).df()
        pop = con.execute('SELECT * FROM read_parquet(?)', [str(snapshot / 'population.parquet')]).df()
        reference = con.execute('''SELECT ccg, year, emp AS reference_emp FROM read_parquet(?)
            WHERE year BETWEEN ? AND ?''', [str(snapshot / 'county_reference.parquet'), spec['first_employment_year'], spec['last_regression_year']]).df()
        presence = con.execute('''SELECT fipstate, fipscty, year, count(*) AS source_rows,
            count(*) FILTER (WHERE emp IS NULL OR NOT isfinite(emp)) AS null_rows
            FROM read_parquet(?) WHERE year BETWEEN ? AND ? GROUP BY ALL''',
            [str(snapshot / 'raw_efsy.parquet'), spec['first_employment_year'], spec['last_regression_year']]).df()
    if set(employment.ccg) - set(universe):
        raise ValueError('employment unit outside crosswalk')
    if set(employment.year) != set(range(spec['first_employment_year'], spec['last_regression_year'] + 1)):
        raise ValueError('cell-year coverage differs from approved input window')
    compare = employment.merge(reference, on=['ccg', 'year'], how='outer', validate='one_to_one')
    if compare[['employment', 'reference_emp']].isna().any().any():
        raise ValueError('county reference and cell-total coverage differ')
    difference = (compare.employment - compare.reference_emp).abs()
    if not np.allclose(compare.employment, compare.reference_emp, rtol=1e-12, atol=1e-6):
        raise ValueError(f'county-cell reconstruction failed: maximum difference {difference.max()}')
    residual = presence.fipscty.eq(999)
    dc = residual & presence.fipstate.eq(11) & presence.year.between(1976, 2003)
    if presence.loc[~residual & presence.fipstate.eq(11), 'year'].isin(presence.loc[dc, 'year']).any():
        raise ValueError('District of Columbia residual overlaps county records')
    presence['fips'] = presence.fipstate * 1000 + presence.fipscty
    presence.loc[dc, 'fips'] = 11001
    presence = presence.loc[~residual | dc].copy()
    presence['ccg'] = presence.fips.map(cw.set_index('fips').ccg)
    if presence.ccg.isna().any():
        raise ValueError('unmapped employment source county')
    presence = presence.groupby(['ccg', 'year'], sort=False, as_index=False)[['source_rows', 'null_rows']].sum()
    p, e, ledger, retained = select_counties(universe, pop, employment, presence, spec)
    dictionary = cw.drop_duplicates('ccg')[['ccg', 'ccg_name']].rename(columns={'ccg_name': 'county_name'})
    dictionary['county_name'] = dictionary.county_name.fillna('Name absent from source crosswalk')
    dictionary['state'] = dictionary.ccg // 1000
    dictionary = dictionary.merge(divisions(), on='state', how='left', validate='many_to_one')
    if dictionary[['division', 'region']].isna().any().any():
        raise ValueError('missing division or region mapping')
    dictionary['retained'] = dictionary.ccg.isin(retained)
    ledger = ledger.merge(dictionary, on='ccg', how='left', validate='many_to_one')
    panel = p.merge(e[['ccg', 'year', 'employment', 'n_cells']], on=['ccg', 'year'], how='left', validate='one_to_one')
    panel = panel.merge(dictionary.drop(columns='retained'), on='ccg', validate='many_to_one')
    panel = panel.sort_values(['ccg', 'year']).reset_index(drop=True)
    panel['employment_required'] = panel.year.ge(spec['first_employment_year'])
    if panel.loc[panel.employment_required, 'employment'].isna().any():
        raise ValueError('missing required employment survived selection')
    panel.to_parquet(output / 'county_inputs.parquet', index=False)
    dictionary.to_csv(output / 'unit_dictionary.csv', index=False)
    ledger.to_csv(output / 'county_exclusions.csv', index=False)
    compare.to_parquet(output / 'cell_county_reconstruction.parquet', index=False)
    pd.testing.assert_frame_equal(panel, pd.read_parquet(output / 'county_inputs.parquet'))
    full_pop = pop.loc[pop.year.between(spec['first_population_year'], spec['last_regression_year'])]
    share_p = full_pop.loc[~full_pop.ccg.isin(retained), 'population'].sum() / full_pop.population.sum()
    share_e = employment.loc[~employment.ccg.isin(retained), 'employment'].sum() / employment.employment.sum()
    for row in manifest['files']:
        if sha(snapshot / row['relative_path']) != row['sha256']:
            raise ValueError('frozen input changed during build')
    report = {
        'status': 'county_inputs_verified_procurement_absent',
        'input_boundary': 'frozen_existing_CBP_cells_and_population',
        'raw_to_cell_replication_certified': False,
        'population_raw_source_rebuild_certified': False,
        'specification': spec, 'snapshot_manifest': str(snapshot / 'manifest.json'),
        'snapshot_manifest_sha256': sha(snapshot / 'manifest.json'),
        'builder_sha256': sha(Path(__file__)),
        'original_counties': len(universe), 'retained_counties': len(retained),
        'excluded_counties': len(universe) - len(retained),
        'excluded_ids': sorted(int(c) for c in set(universe) - set(retained)),
        'population_employee_year_share_excluded_available_values': float(share_p),
        'employment_employee_year_share_excluded': float(share_e),
        'maximum_cell_sum_vs_saved_county_difference': float(difference.max()),
        'panel_rows': len(panel), 'regression_observations_if_procurement_complete': len(retained) * (spec['last_regression_year'] - spec['first_regression_year'] + 1),
        'panel_sha256': sha(output / 'county_inputs.parquet'),
        'exclusions_sha256': sha(output / 'county_exclusions.csv'),
        'dictionary_sha256': sha(output / 'unit_dictionary.csv'),
    }
    write_json(output / 'manifest.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['snapshot', 'build'])
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--raw-efsy', type=Path)
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        parser.error('invalid run-id')
    if args.project.resolve() == REPO.resolve():
        parser.error('data must be in shared Dropbox, not the code checkout')
    spec = json.loads((APP / 'config/county_inputs.json').read_text())
    if spec['status'] != 'approved_for_county_input_build':
        raise ValueError('county input construction not approved')
    root = args.project / '4_Macro_version/4a_PaperApplication/data/intermediate'
    snapshot = root / args.run_id / 'snapshot'
    if args.stage == 'snapshot':
        if args.raw_efsy is None:
            parser.error('--raw-efsy is required for snapshot only')
        freeze(args.project, args.raw_efsy, snapshot, spec)
    else:
        build(snapshot, root / args.run_id / 'county', spec)


if __name__ == '__main__':
    main()
