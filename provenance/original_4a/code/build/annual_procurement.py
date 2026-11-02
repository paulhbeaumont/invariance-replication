"""Build approved procurement and merge the certified county input snapshot.

MPGR run class: cache_rebuild. No certified spending cache exists.
Certification covers frozen parsed sources and approved transformations only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import duckdb
import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--run-id', required=True)
    a = ap.parse_args()
    if not a.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.run_id):
        raise ValueError('invalid run identifier')
    spec = json.loads((APP/'config/procurement.json').read_text())
    if spec['status'] != 'approved' or spec['construction'] != 'exclude_FSC_prefix_Y' or spec['transition_quarter'] != 'add_1976tq_to_1976':
        raise ValueError('unsupported procurement specification')
    root = a.project/'4_Macro_version/4a_PaperApplication'
    output = root/'data/intermediate'/a.run_id
    countyroot = root/'data/intermediate/county_inputs_20260913'
    county = countyroot/'county/county_inputs.parquet'
    county_manifest = countyroot/'county/manifest.json'
    cm = json.loads(county_manifest.read_text())
    if cm['status'] != 'county_inputs_verified_procurement_absent' or cm['panel_sha256'] != sha(county):
        raise ValueError('county input provenance mismatch')
    crosswalk = countyroot/'snapshot/crosswalk.csv'
    old = json.loads((countyroot/'snapshot/manifest.json').read_text())
    expected = next(r['sha256'] for r in old['files'] if r['relative_path']=='crosswalk.csv')
    if sha(crosswalk) != expected:
        raise ValueError('crosswalk provenance mismatch')
    paths = [a.project/f'1_current_paper/_data/temp/dcads_year/fy{y}.parquet' for y in range(1969,2007)]
    paths.append(a.project/'1_current_paper/_data/temp/dcads_year/fy1976tq.parquet')
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError('required annual procurement source absent')
    output.mkdir(parents=True,exist_ok=False)
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    records=[]
    for path in [*paths, county, county_manifest, crosswalk]:
        target=snapshot/path.name
        digest=sha(path)
        shutil.copy2(path,target)
        if sha(path)!=digest or sha(target)!=digest:
            raise ValueError('source changed during snapshot')
        records.append({'source':str(path),'snapshot':str(target.relative_to(output)),'sha256':digest})
    cw=pd.read_csv(snapshot/'crosswalk.csv',dtype=str)
    if cw.county_fips5.duplicated().any():
        raise ValueError('duplicate crosswalk key')
    with duckdb.connect() as con:
        files=[str(snapshot/p.name) for p in paths]
        amounts=con.execute('''SELECT fiscal_year, geo_class, county_fips5,
            starts_with(trim(coalesce(fsc,'')), 'Y') AS construction,
            dollars_unit, sum(dollars) AS raw_amount, count(*) AS actions,
            count(*) FILTER (WHERE dollars IS NULL) AS missing_amount
            FROM read_parquet(?, union_by_name=true) GROUP BY ALL''',[files]).df()
        annual=con.execute('SELECT * FROM read_parquet(?)',[str(snapshot/'county_inputs.parquet')]).df()
    if amounts.missing_amount.sum() or not set(amounts.dollars_unit)<= {'dollars','thousands'}:
        raise ValueError('unknown units or missing procurement amount')
    if set(amounts.fiscal_year) != {str(y) for y in range(1969,2007)}|{'1976tq'}:
        raise ValueError('procurement year labels mismatch')
    amounts['nominal']=amounts.raw_amount*np.where(amounts.dollars_unit.eq('thousands'),1000.,1.)
    amounts['year']=amounts.fiscal_year.replace({'1976tq':'1976'}).astype(int)
    amounts['ccg_label']=amounts.county_fips5.map(cw.set_index('county_fips5').ccg_id)
    county_actions=amounts.geo_class.eq('county')
    if amounts.loc[county_actions,'ccg_label'].isna().any():
        raise ValueError('unmapped county procurement; no silent loss allowed')
    amounts['reason']='retained'
    amounts.loc[amounts.construction,'reason']='construction_excluded'
    amounts.loc[~county_actions,'reason']='not_county_geocoded'
    unplaceable=county_actions & amounts.ccg_label.eq('unplaceable')
    amounts.loc[unplaceable,'reason']='declared_unplaceable'
    usable=amounts.reason.eq('retained')
    amounts['ccg']=pd.to_numeric(amounts.ccg_label,errors='coerce')
    amounts.loc[usable & ~amounts.ccg.isin(annual.ccg.unique()),'reason']='county_excluded_by_approved_sample'
    selected=amounts.loc[amounts.reason.eq('retained')]
    totals=selected.groupby(['ccg','year'],sort=False,as_index=False).agg(spending_nominal=('nominal','sum'),n_actions=('actions','sum'))
    if totals.duplicated(['ccg','year']).any():
        raise ValueError('duplicate procurement key')
    annual=annual.merge(totals,on=['ccg','year'],how='left',validate='one_to_one')
    annual['no_contract_action']=annual.n_actions.isna()
    annual[['spending_nominal','n_actions']]=annual[['spending_nominal','n_actions']].fillna(0.)
    cpi={int(y):float(v) for y,v in spec['cpi'].items()}
    annual['spending_real']=annual.spending_nominal*cpi[spec['base_year']]/annual.year.map(cpi)
    if not np.isfinite(annual.spending_real).all():
        raise ValueError('nonfinite real spending')
    if not np.isclose(annual.spending_nominal.sum(),selected.nominal.sum(),rtol=1e-12,atol=1):
        raise ValueError('procurement conservation failure')
    accounting=amounts.groupby(['fiscal_year','reason'],as_index=False,sort=False)[['nominal','actions']].sum()
    accounting.to_csv(output/'procurement_accounting.csv',index=False)
    annual.to_parquet(output/'annual_panel.parquet',index=False)
    pd.testing.assert_frame_equal(annual,pd.read_parquet(output/'annual_panel.parquet'))
    for row in records:
        if sha(output/row['snapshot'])!=row['sha256']:
            raise ValueError('snapshot changed during construction')
    manifest={'status':'certified','certification_scope':'frozen_constructed_inputs_and_approved_arithmetic_not_exact_NS_replication',
        'sources':records,'construction_specification':spec,'panel_sha256':sha(output/'annual_panel.parquet'),
        'builder_sha256':sha(Path(__file__)),'county_input_manifest_sha256':sha(snapshot/'manifest.json'),
        'rows':len(annual),'counties':int(annual.ccg.nunique()),'nominal_spending_total':float(annual.spending_nominal.sum()),
        'zero_action_county_years':int(annual.no_contract_action.sum()),'NS_reference_discrepancy_accepted':True}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert json.loads((output/'manifest.json').read_text())==manifest
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['sources','construction_specification']},indent=2))


if __name__=='__main__':
    main()
