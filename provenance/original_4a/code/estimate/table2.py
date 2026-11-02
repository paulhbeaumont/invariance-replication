"""Estimate nine missing Table 2 fits; retain all saved comparison estimates.

MPGR run class: estimate_only; cache: analysis/table2_cells.
mpgr-new-estimates: 9; mpgr-batch-approved: explicit.
Three NAICS6 MPGR baseline fits and six industry-year FE fits, h=2/3/4.
MPGR CRV1 uses county-year N, matching the established invariance convention;
log CRV1 uses the naturally defined cell count. County FE are cluster-nested.
"""
from pathlib import Path
import sys,gc
import duckdb
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from baseline_io import APP,DATA,locked,read_json,require,sha,write_json
from baseline_math import fit2sls

_,_,lock=locked();root=DATA/'analysis/table2_cells';cert=read_json(root/'certificate.json')
require(cert['status']=='PASS' and cert['baseline_lock']==lock,'Cell cache mismatch')
require(sha(APP/'code/build/table2_cells.py')==cert['builder_sha256'],'Cell builder changed')
parent=DATA/'analysis/disaggregation_h234';pc=read_json(parent/'certificate.json')
require(sha(parent/'certificate.json')==cert['parent_certificate_sha256'],'County certificate changed')
oldpath=APP/'results/coefficients/disaggregation_h234_complete.csv'
require(sha(oldpath)==read_json(APP/'results/manifests/disaggregation_log_completion.json')['coefficients_sha256'],'Reference results changed')
old=pd.read_csv(oldpath)
records=[];saved=APP/'results/manifests/table2_fits';saved.mkdir(exist_ok=True)
for h in (2,3,4):
    path=parent/f'county_h{h}.parquet';require(sha(path)==pc['files'][path.name],'County cache changed')
    county=pd.read_parquet(path);n_county=len(county)
    counts=next(a for a in cert['horizon_counts'] if a['horizon']==h)
    for growth,industry_fe in [('mpgr',False),('log',True),('mpgr',True)]:
        dest=saved/f'h{h}_{growth}_iy{int(industry_fe)}.json'
        if dest.exists():
            row=read_json(dest)
            require(row['estimator_sha256']==sha(__file__) and row['certificate_sha256']==sha(root/'certificate.json'),'Stale saved fit')
            records.append(row);continue
        print(f'START h={h} {growth} industry_year={industry_fe}',flush=True)
        if not industry_fe:
            path=root/f'mpgr_h{h}.parquet';require(sha(path)==cert['files'][path.name],'MPGR statistic changed')
            s=pd.read_parquet(path);require(np.array_equal(s[['ccg','year']],county[['ccg','year']]),'Key mismatch')
            y=s.y.to_numpy();w=s.mass.to_numpy();x=county.x.to_numpy();z=county.z.to_numpy()
            c=county.fe_county.to_numpy(np.int64);fe=county.fe_year.to_numpy(np.int64)
            n_cells=counts['balanced_cells'];n_cy=n_county;groups=county.ccg.nunique()
        else:
            for name,digest in cert['files'].items():
                if name.startswith(f'h{h}/'):require(sha(root/name)==digest,'Cell file changed')
            with duckdb.connect() as con:
                d=con.execute(f'''SELECT ccg,year,fe_county,fe_industry_year,x,z,y_{growth} AS y,w_{growth} AS w
                    FROM read_parquet(?) WHERE w_{growth}>0 AND y_{growth} IS NOT NULL''',[str(root/f'h{h}/y*.parquet')]).df()
            y,x,z,w=(d[k].to_numpy() for k in ['y','x','z','w'])
            c=d.fe_county.to_numpy(np.int64);fe=d.fe_industry_year.to_numpy(np.int64)
            groups=d.ccg.nunique();n_cy=len(d[['ccg','year']].drop_duplicates());n_cells=len(d)
        n_ssc=n_cy if growth=='mpgr' else n_cells
        r=fit2sls(y,x,z,w,c,fe,c,singleton=False,correction_n=n_ssc)
        require(r['n_dropped']==0 and np.isfinite([r['beta'],r['se'],r['F']]).all(),'Invalid estimate')
        if not industry_fe:
            ref=old[old.horizon.eq(h)&old.rung.eq('county')&old.growth.eq('mpgr')].iloc[0]
            require(np.allclose([r['beta'],r['se'],r['F']],[ref.beta,ref.se,ref.first_stage_F],rtol=1e-8,atol=1e-12),'NAICS6 MPGR invariance failed')
        row=dict(horizon=h,rung='naics6',growth=growth,industry_year_fe=industry_fe,beta=r['beta'],se=r['se'],
            t=r['beta']/r['se'],first_stage_F=r['F'],n_estimation_cells=n_cells,n_county_years=n_cy,n_counties=int(groups),
            correction_n=n_ssc,ssc_k=r['k'],county_years_excluded=n_county-n_cy,first_year=int(county.year.min()),last_year=2016,status='PASS',
            estimator_sha256=sha(__file__),certificate_sha256=sha(root/'certificate.json'))
        write_json(dest,row);records.append(row);print(row,flush=True)
        del y,x,z,w,c,fe,r
        if industry_fe:del d
        gc.collect()
out=APP/'results/coefficients/table2_new.csv';pd.DataFrame(records).to_csv(out,index=False)
write_json(APP/'results/manifests/table2_new.json',dict(status='PASS',coefficients_sha256=sha(out),new_estimates=len(records),
    estimator_sha256=sha(__file__),cache_certificate_sha256=sha(root/'certificate.json')))
