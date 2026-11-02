"""Diagnose whether the unresolved out-of-sample audit affects paper estimates.

Uses raw-reconstructed annual inputs only after exact comparison to immutable
historical targets. Runs unchanged package estimators against independent saved
results. Does not change the production launcher's data or certify the full
application/raw build.
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd

from package_io import ROOT, logs, read, require, sha, write

sys.path.insert(0,str(ROOT/'code/vendor'))
import reproduce as original
from county_builder import county_frame

RUN = logs()/'raw_rebuild_20261027'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=int,default=1)
    attempt=parser.parse_args().attempt
    require(attempt>=1,'Attempt must be positive')
    original.verify()
    diagnostic = read(RUN/'source51_exact_value_comparison.json')
    require(all(r['status'] in ('EXACT_DATAFRAME_MATCH','EXACT_VALUES_INTEGER_WIDTH_ONLY') for r in diagnostic['files']),
            'Reconstructed annual inputs do not exactly match independent originals')
    source = ROOT/diagnostic['input_directory']
    sources = dict(snapshot_directory=diagnostic['input_directory'],
                   inputs={r['file']:r['rebuilt_sha256'] for r in diagnostic['files']})
    for filename, digest in sources['inputs'].items():
        require(sha(source/filename)==digest, 'Reconstructed diagnostic input changed')
    destination = RUN/('checkpoint' if attempt==1 else 'checkpoint_attempt'+str(attempt))
    require(not destination.exists(), 'Checkpoint already exists; preserve previous diagnostic')
    destination.mkdir()
    data = destination/'data'
    frames = {}
    storage_adapters=[]
    for h in (2,3,4):
        spec = read(ROOT/'config/county_construction.json')
        spec.update(horizon=h,first_year=1978+h,n_observations=3058*(2017-1978-h))
        f = county_frame(spec,sources)
        f['log_y']=np.log(f.E)-np.log(f.E_b)
        # Legacy pandas/NumPy expose nonmissing nullable numeric arrays as
        # object arrays. Supply the identical underlying numeric storage to the
        # unchanged estimator; do not impute, round or modify a value.
        for column in f:
            dtype=getattr(f[column].dtype,'numpy_dtype',None)
            if dtype is not None and pd.api.types.is_numeric_dtype(f[column].dtype):
                require(f[column].notna().all(),'Cannot remove nullable storage from a missing value')
                before=f[column].to_numpy(dtype=dtype)
                f[column]=before
                require(np.array_equal(f[column].to_numpy(),before),'Numeric storage conversion changed a value')
                storage_adapters.append(dict(horizon=h,column=column,numpy_dtype=str(dtype),values_changed=False))
        frames[h]=f
    county_comparisons=[]
    frozen = pd.read_parquet(ROOT/'data/disaggregation/county_h3.parquet')
    f=frames[3]
    require(np.array_equal(f[['ccg','year']],frozen[['ccg','year']]),'Reconstructed county keys differ')
    for column in f:
        require(column in frozen,'Unexpected reconstructed column: '+column)
        if pd.api.types.is_numeric_dtype(f[column]):
            original.close(f[column],frozen[column],'Raw checkpoint county '+column,rtol=1e-12,atol=1e-10)
            county_comparisons.append(dict(column=column,max_abs_difference=float(abs(f[column]-frozen[column]).max())))
        else:
            require(f[column].equals(frozen[column]),'Raw checkpoint metadata differs: '+column)
            county_comparisons.append(dict(column=column,exact_metadata_match=True))
    county_results=[]
    for col,growth in [(1,'log_y'),(6,'y')]:
        result=original.scalar_fit(f,f[growth],np.ones(len(f)),len(f))
        county_results.append(original.check_disfit(col,result,len(f),len(f)))
    original.dump_csv(destination,'county_estimates.csv',county_results)

    config=ROOT/'provenance/original_4a/config'
    declarations=read(config/'aggregation.json')
    geography_sources={}
    for name, record in declarations['sources'].items():
        path=ROOT/'provenance/historical_fc9df56'/record['source']
        require(sha(path)==record['sha256'],'Frozen geography source changed')
        geography_sources[name]=path
    x=pd.read_csv(geography_sources['aggregation_county_ccg_crosswalk.csv'],dtype=str)
    x=x[x.ccg_id.ne('unplaceable')].copy();x['ccg']=x.ccg_id.astype(int)
    b=pd.read_csv(geography_sources['aggregation_county_bea_area_2004.csv'],dtype=str)
    crosswalk=x.merge(b,on='county_fips5',how='left',validate='many_to_one',sort=False)
    cea=crosswalk.dropna(subset=['cea_code']).drop_duplicates('ccg').set_index('ccg').cea_code
    aggregation_dir=data/'aggregation';aggregation_dir.mkdir(parents=True)
    input_comparisons=[]
    for rung,h in [('state',2),('state',3),('state',4),('cea',3)]:
        g=frames[h].copy()
        g['unit']=g.state if rung=='state' else g.ccg.map(cea)
        require(g.unit.notna().all(),'Unmapped county')
        g['unit']=pd.factorize(g.unit,sort=True)[0]
        g['weight']=g.E_b/g.groupby(['unit','year']).E_b.transform('sum')
        g['std_y']=(g.E-g.E_b)/g.E_b
        for v in ('x','z','std_y'):
            g['w_'+v]=g.weight*g[v]
        a=g.groupby(['unit','year'],as_index=False).agg(
            E=('E','sum'),E_b=('E_b','sum'),x_agg=('w_x','sum'),z_agg=('w_z','sum'),
            reconstructed=('w_std_y','sum'),weight_sum=('weight','sum'))
        a['std_y']=(a.E-a.E_b)/a.E_b
        a['log_y']=np.log(a.E/a.E_b)
        a['weight']=1.0
        original.close(a.std_y,a.reconstructed,'Reconstructed aggregate growth',rtol=1e-12,atol=1e-12)
        g=g.merge(a[['unit','year','x_agg','z_agg']],on=['unit','year'],validate='many_to_one')
        for tag, frame in [('county',g),('aggregate',a)]:
            filename=f'{tag}_{rung}_h{h}.parquet'
            frame.to_parquet(aggregation_dir/filename,index=False)
            expected=pd.read_parquet(ROOT/'data/aggregation'/filename)
            columns=['unit','year','weight','E','E_b','x_agg','z_agg','std_y','log_y']
            if tag=='county':columns+=['ccg','x','z']
            for column in columns:
                original.close(frame[column],expected[column],filename+' '+column,rtol=1e-12,atol=1e-9)
            input_comparisons.append(dict(file=filename,status='MODEL_INPUTS_MATCH_INDEPENDENT_TARGET',
                                           rows=len(frame)))
    # Keep the unchanged estimator, comparisons and reference files. Only its
    # input/output directories point at the diagnostic raw reconstruction.
    original.DATA=data
    original.OUT=destination/'results'
    original.aggregation()
    actual=pd.read_csv(original.OUT/'aggregation_summary_tables/selected_results.csv').set_index('spec')
    expected=pd.read_csv(ROOT/'reference/aggregation_selected.csv').set_index('spec')
    numeric_checks=[]
    for specification in actual.index:
        for column in actual.columns.intersection(expected.columns):
            left,right=actual.loc[specification,column],expected.loc[specification,column]
            if isinstance(left,(int,float,np.number)) and isinstance(right,(int,float,np.number)):
                original.close(left,right,'Full checkpoint '+specification+' '+column)
                numeric_checks.append(dict(spec=specification,field=column,
                    absolute_difference=None if pd.isna(left) or pd.isna(right) else float(abs(left-right))))
    write(RUN/'checkpoint_all_aggregation_fields.json',dict(status='PASS',
        numeric_comparisons=len(numeric_checks),specifications=len(actual),checks=numeric_checks,
        raw_to_results_certified=False,original_population_audit_status='FAILED'))
    report=dict(status='CHECKPOINT_ESTIMATES_MATCH_UPSTREAM_AUDIT_UNRESOLVED',
                county_comparisons=county_comparisons,county_columns=[1,6],aggregation_specifications=6,
                aggregation_input_comparisons=input_comparisons,
                storage_adapters=storage_adapters,attempt=attempt,
                aggregation_numeric_comparisons=len(numeric_checks),
                original_population_audit_status='FAILED',raw_to_results_certified=False,
                independent_references_unchanged=True,
                not_yet_reconstructed=['six-digit/disaggregated regression inputs and decomposition',
                                       'balance/heterogeneity inputs and corresponding estimates'])
    write(RUN/'checkpoint_results.json',report)
    print('CHECKPOINT PASS: two county coefficients and six aggregation specifications match independent references.',flush=True)
    print('The historical population audit remains FAILED; full raw-to-results certification is false.',flush=True)


if __name__=='__main__':
    main()
