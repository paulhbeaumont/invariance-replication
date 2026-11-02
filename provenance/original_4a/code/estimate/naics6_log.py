"""Estimate nine NAICS6 log comparisons from certified sufficient statistics.

Run class: estimate_only; input cache: analysis/naics6_log_h234.
County/year FE, county clusters, cell-count finite-sample correction.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from baseline_io import APP,DATA,locked,read_json,require,sha,write_json
from baseline_math import fit2sls

_,_,lock=locked()
root=DATA/'analysis/naics6_log_h234'
parent=DATA/'analysis/disaggregation_h234'
cert=read_json(root/'certificate.json')
require(cert['status']=='PASS' and cert['baseline_lock']==lock,'Cache mismatch')
require(sha(APP/'code/build/naics6_log.py')==cert['builder_sha256'],'Builder changed')
require(sha(parent/'certificate.json')==cert['parent_certificate_sha256'],'Parent certificate changed')
pc=read_json(parent/'certificate.json')
rows=[]
out=APP/'results/coefficients/log_naics6.csv'
require(not out.exists(),'Refusing to overwrite estimates')
for h in (2,3,4):
    p=parent/f'county_h{h}.parquet'
    require(sha(p)==pc['files'][p.name],'County cache changed')
    f=pd.read_parquet(p)
    p=root/f'naics6_h{h}.parquet'
    require(sha(p)==cert['files'][p.name],'Statistics changed')
    s=pd.read_parquet(p)
    require(np.array_equal(f[['ccg','year']],s[['ccg','year']]),'Key mismatch')
    keep=s.log_count.gt(0)
    g=f.loc[keep]
    n=int(s.log_count.sum())
    for variant in ('ew','midpoint','lag'):
        y=s.loc[keep,variant+'_mean'].to_numpy()
        w=s.loc[keep,'log_count' if variant=='ew' else 'lag_mass'].to_numpy(float) if variant!='midpoint' else np.ones(len(g))
        c,t=(g[k].to_numpy(np.int64) for k in ('fe_county','fe_year'))
        r=fit2sls(y,g.x.to_numpy(),g.z.to_numpy(),w,c,t,c.copy(),singleton=False,correction_n=n)
        require(r['n_dropped']==0 and np.isfinite([r['beta'],r['se'],r['F']]).all(),'Invalid fit')
        rows.append(dict(horizon=h,rung='naics6',growth='log',weight=variant,beta=r['beta'],se=r['se'],t=r['beta']/r['se'],
            first_stage_F=r['F'],n_counties=int(g.ccg.nunique()),n_county_years=len(g),n_estimation_cells=n,
            county_years_excluded=int((~keep).sum()),first_year=int(g.year.min()),last_year=int(g.year.max()),status='PASS'))
        print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(out,index=False)
write_json(APP/'results/manifests/log_naics6.json',dict(status='PASS',new_estimates=9,coefficients_sha256=sha(out),
    estimator_sha256=sha(__file__),cache_certificate_sha256=sha(root/'certificate.json'),sample='natural positive-endpoint log sample'))
