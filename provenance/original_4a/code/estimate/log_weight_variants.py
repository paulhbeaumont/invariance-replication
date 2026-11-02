"""Six approved log-weight comparisons per selected variant; no baseline re-estimation.

MPGR run class: estimate_only. Batch: h=2/3/4 × NAICS2/NAICS4.
Read certified weighted-log sums. Cluster and FE: county, and county/year.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
import numpy as np
import pandas as pd
from baseline_io import APP,DATA,locked,read_json,write_json,require,sha
from baseline_math import fit2sls

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('variant',choices=['midpoint_survivors','lag_full_county'])
a=p.parse_args()
normalized=a.variant=='midpoint_survivors'
_,_,base_lock=locked()
root=DATA/('analysis/weighted_log_h234' if normalized else 'analysis/lag_weighted_log_h234')
parent=DATA/'analysis/disaggregation_h234'
cert=read_json(root/'certificate.json')
require(cert['status']=='PASS' and cert['baseline_lock']==base_lock,'Weighted-log cache certification mismatch')
require(sha(parent/'certificate.json')==cert['parent_certificate_sha256'],'County cache certificate changed')
require(sha(APP/('code/build/weighted_log.py' if normalized else 'code/build/lag_weighted_log.py'))==cert['builder_sha256'],'Weighted-log builder changed')
pcert=read_json(parent/'certificate.json')
source=APP/'results/coefficients/disaggregation_h234_complete.csv'
source_hash=sha(source)
require(source_hash==read_json(APP/'results/manifests/disaggregation_log_completion.json')['coefficients_sha256'],'Existing results changed')
old=pd.read_csv(source)
rows=[]
for h in (2,3,4):
    path=parent/f'county_h{h}.parquet'
    require(sha(path)==pcert['files'][path.name],'County frame changed')
    f=pd.read_parquet(path)
    for rung in ('naics2','naics4_alloc'):
        path=root/f'{rung}_h{h}.parquet'
        require(sha(path)==cert['files'][path.name],'Weighted-log statistic changed')
        s=pd.read_parquet(path)
        require(np.array_equal(f[['ccg','year']],s[['ccg','year']]),'County-year keys mismatch')
        keep=s.log_count.to_numpy()>0
        g=f.loc[keep]
        y=s.loc[keep,'weighted_log_mean'].to_numpy()
        w=np.ones(int(keep.sum())) if normalized else s.loc[keep,'survivor_lag_mass'].to_numpy()
        x,z=(g[k].to_numpy() for k in ('x','z'))
        c,t=(g[k].to_numpy(np.int64) for k in ('fe_county','fe_year'))
        n_cells=int(s.log_count.sum())
        r=fit2sls(y,x,z,w,c,t,c.copy(),singleton=False,correction_n=n_cells)
        require(r['n_dropped']==0 and np.isfinite([r['beta'],r['se'],r['F']]).all(),'Invalid weighted estimate')
        prior=old[old.horizon.eq(h)&old.rung.eq(rung)&old.growth.eq('log')].iloc[0]
        require(n_cells==prior.n_estimation_cells and len(g)==prior.n_county_years,'Sample differs from unweighted log')
        rows.append(dict(horizon=h,rung=rung,growth='log',weight=('midpoint_share_survivors' if normalized else 'lag_share_full_county'),beta=r['beta'],se=r['se'],
            t=r['beta']/r['se'],first_stage_F=r['F'],n_counties=int(g.ccg.nunique()),n_county_years=len(g),
            n_estimation_cells=n_cells,county_years_excluded=int((~keep).sum()),undefined_cells=int(prior.log_undefined),
            first_year=int(g.year.min()),last_year=int(g.year.max()),status='PASS'))
        print(rows[-1],flush=True)
require(sha(source)==source_hash,'Existing results were modified')
out=APP/f'results/coefficients/log_{a.variant}.csv'
require(not out.exists(),'Refusing to overwrite weighted-log results')
pd.DataFrame(rows).to_csv(out,index=False)
write_json(APP/f'results/manifests/log_{a.variant}.json',dict(status='PASS',new_estimates=6,
    coefficients_sha256=sha(out),existing_results_sha256=source_hash,cache_certificate_sha256=sha(root/'certificate.json'),
    estimator_sha256=sha(__file__),same_sample_as_unweighted_log=True,survivor_renormalization=normalized))
