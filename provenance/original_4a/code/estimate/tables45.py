"""Six NAICS6 industry-year FE log fits, EW and full-county lag shares.

MPGR run class: estimate_only; cache: analysis/tables45_weights.
mpgr-new-estimates: 6; mpgr-batch-approved: explicit; horizons 2/3/4.
"""
from pathlib import Path
import sys,gc
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from baseline_io import APP,DATA,locked,read_json,require,sha,write_json
from baseline_math import fit2sls
_,_,lock=locked();root=DATA/'analysis/tables45_weights';cert=read_json(root/'certificate.json')
require(cert['status']=='PASS' and cert['baseline_lock']==lock,'Cache mismatch')
require(sha(APP/'code/build/tables45_weights.py')==cert['builder_sha256'],'Builder changed')
saved=APP/'results/manifests/tables45_fits';saved.mkdir(exist_ok=True);rows=[]
for h in (2,3,4):
    p=root/f'h{h}.parquet';require(sha(p)==cert['files'][p.name],'Cache hash mismatch')
    d=pd.read_parquet(p);n=len(d);n_cy=len(d[['ccg','year']].drop_duplicates())
    for weight in ('ew','lag'):
        dest=saved/f'h{h}_{weight}.json'
        if dest.exists():
            row=read_json(dest);require(row['estimator_sha256']==sha(__file__) and row['certificate_sha256']==sha(root/'certificate.json'),'Stale fit')
            rows.append(row);continue
        print(f'START h={h} log {weight} industry-year FE',flush=True)
        y,x,z=(d[k].to_numpy() for k in ('y','x','z'))
        w=np.ones(n) if weight=='ew' else d.w_lag.to_numpy()
        c=d.fe_county.to_numpy(np.int64);fe=d.fe_industry_year.to_numpy(np.int64)
        r=fit2sls(y,x,z,w,c,fe,c,singleton=False,correction_n=n)
        require(r['n_dropped']==0 and np.isfinite([r['beta'],r['se'],r['F']]).all(),'Invalid fit')
        row=dict(horizon=h,rung='naics6',growth='log',weight=weight,industry_year_fe=True,beta=r['beta'],se=r['se'],
            t=r['beta']/r['se'],first_stage_F=r['F'],n_estimation_cells=n,n_county_years=n_cy,n_counties=int(d.ccg.nunique()),
            first_year=int(d.year.min()),last_year=int(d.year.max()),correction_n=n,ssc_k=r['k'],status='PASS',
            estimator_sha256=sha(__file__),certificate_sha256=sha(root/'certificate.json'))
        write_json(dest,row);rows.append(row);print(row,flush=True)
        del y,x,z,w,c,fe,r;gc.collect()
    del d;gc.collect()
out=APP/'results/coefficients/tables45_new.csv';pd.DataFrame(rows).to_csv(out,index=False)
write_json(APP/'results/manifests/tables45_new.json',dict(status='PASS',new_estimates=len(rows),coefficients_sha256=sha(out),estimator_sha256=sha(__file__)))
