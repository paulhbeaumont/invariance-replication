"""Construct NAICS6 log sufficient statistics for three weights and horizons.

Run class: cache_augment. Input: certified NAICS6 annual employment and county
horizon caches. No existing cache or specification is rebuilt or estimated.
Sparse endpoints are outer-joined; absent industry cells are zero, not counties.
"""
from pathlib import Path
import sys
import duckdb
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from baseline_io import APP,DATA,locked,read_json,require,sha,write_json


def main():
    _,_,lock=locked()
    source=DATA/'intermediate/naics6_residual'
    cert=read_json(source/'certificate.json')
    require(cert['status']=='PASS' and cert['baseline_lock']==lock,'NAICS6 certificate mismatch')
    require(sha(APP/'code/build/naics6_cells.py')==cert['builder_sha256'],'Employment builder changed')
    for name,digest in cert['files'].items():
        require(sha(source/name)==digest,'Employment snapshot changed')
    parent=DATA/'analysis/disaggregation_h234'
    pc=read_json(parent/'certificate.json')
    require(pc['status']=='PASS' and pc['baseline_lock']==lock,'County certificate mismatch')
    out=DATA/'analysis/naics6_log_h234'
    out.mkdir(parents=True,exist_ok=False)
    files,checks={},[]
    with duckdb.connect() as con:
        for h in (2,3,4):
            county=parent/f'county_h{h}.parquet'
            require(sha(county)==pc['files'][county.name],'County frame changed')
            f=pd.read_parquet(county)
            parts=[]
            for year,g in f.groupby('year',sort=True):
                g=g.sort_values('ccg')
                s=con.execute('''WITH pairs AS (
                    SELECT coalesce(a.ccg,b.ccg) AS ccg,
                    coalesce(a.emp,0) AS cur,coalesce(b.emp,0) AS lag
                    FROM read_parquet(?) a FULL OUTER JOIN read_parquet(?) b USING(ccg,code)
                ), calc AS (SELECT *,cur>0 AND lag>0 AS pos,
                    CASE WHEN cur>0 AND lag>0 THEN ln(cur)-ln(lag) ELSE 0 END AS lg FROM pairs)
                SELECT ccg,sum(cur) AS cur,sum(lag) AS lag,
                    sum(2*(cur-lag)) AS midpoint_numerator,
                    count(*) FILTER(WHERE pos) AS log_count,
                    sum(lg) AS log_sum,sum((cur+lag)*lg) AS midpoint_log_sum,
                    sum(lag*lg) AS lag_log_sum,
                    sum(CASE WHEN pos THEN cur+lag ELSE 0 END) AS midpoint_surv,
                    sum(CASE WHEN pos THEN lag ELSE 0 END) AS lag_surv
                FROM calc GROUP BY ccg ORDER BY ccg''',[str(source/f'y{year}.parquet'),str(source/f'y{year-h}.parquet')]).df()
                require(np.array_equal(s.ccg,g.ccg),'County support mismatch')
                require(np.allclose(s.cur,g.E,rtol=1e-12,atol=1e-8) and np.allclose(s.lag,g.E_b,rtol=1e-12,atol=1e-8),'Endpoint conservation failed')
                error=float(np.max(np.abs(s.midpoint_numerator.to_numpy()/(g.E+g.E_b).to_numpy()-g.y.to_numpy())))
                require(error<1e-10,'MPGR reconstruction failed')
                s['year']=year
                s['ew_mean']=s.log_sum/s.log_count.replace(0,np.nan)
                s['midpoint_mean']=s.midpoint_log_sum/s.midpoint_surv.replace(0,np.nan)
                s['lag_mean']=s.lag_log_sum/s.lag_surv.replace(0,np.nan)
                s['lag_mass']=s.lag_surv/g.E_b.to_numpy()
                require(np.array_equal(s.log_count.gt(0),s.midpoint_surv.gt(0)) and np.array_equal(s.log_count.gt(0),s.lag_surv.gt(0)),'Weight support mismatch')
                parts.append(s[['ccg','year','log_count','ew_mean','midpoint_mean','lag_mean','lag_mass']])
                checks.append(dict(horizon=h,year=int(year),mpgr_error=error,log_empty_county_years=int(s.log_count.eq(0).sum())))
            stats=pd.concat(parts,ignore_index=True).sort_values(['ccg','year']).reset_index(drop=True)
            require(np.array_equal(stats[['ccg','year']],f[['ccg','year']]),'Cache key mismatch')
            path=out/f'naics6_h{h}.parquet'
            stats.to_parquet(path,index=False)
            pd.testing.assert_frame_equal(pd.read_parquet(path),stats)
            files[path.name]=sha(path)
            print(f'h={h}, log cells={stats.log_count.sum()}, empty county-years={stats.log_count.eq(0).sum()}',flush=True)
    write_json(out/'certificate.json',dict(status='PASS',files=files,baseline_lock=lock,
        parent_certificate_sha256=sha(parent/'certificate.json'),source_certificate_sha256=sha(source/'certificate.json'),
        builder_sha256=sha(__file__),checks=checks))


if __name__=='__main__':
    main()
