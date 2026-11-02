"""Construct only lag-share-weighted log sums from frozen industry cells.

MPGR run class: cache_rebuild, required_key_absent (weighted_log_sum).
County frames, treatment, instruments and all existing caches are read-only.
Cell weight=E_t-h/(full county E_t-h), without survivor rescaling.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
import numpy as np
import pandas as pd
from baseline_io import APP,DATA,locked,read_json,write_json,require,sha


def main():
    _,_,base_lock=locked()
    original=DATA/'analysis/disaggregation_h234'
    cert=read_json(original/'certificate.json')
    require(cert['status']=='PASS' and cert['baseline_lock']==base_lock,'Input certificate mismatch')
    snapshot=read_json(APP/'results/manifests/disaggregation_snapshot.json')
    frozen=DATA/'intermediate/disaggregation_h234'
    out=DATA/'analysis/lag_weighted_log_h234'
    out.mkdir(parents=True,exist_ok=False)
    frames={}
    for h in (2,3,4):
        path=original/f'county_h{h}.parquet'
        require(sha(path)==cert['files'][path.name],'County frame changed')
        frames[h]=pd.read_parquet(path)
    units=np.sort(frames[2].ccg.unique())
    files={}
    audits=[]
    for rung in ('naics2','naics4_alloc'):
        annual={}
        keys=None
        for year in range(1978,2017):
            name=f'{rung}/y{year}.parquet'
            require(sha(frozen/name)==snapshot['inputs'][name],'Frozen cell file changed')
            f=pd.read_parquet(frozen/name,columns=['ccg','code','emp'])
            f=f[f.ccg.isin(units)].sort_values(['ccg','code']).reset_index(drop=True)
            if keys is None:
                keys=f[['ccg','code']].to_numpy(np.int64)
            require(np.array_equal(keys,f[['ccg','code']]),'Cell support changed')
            annual[year]=f.emp.to_numpy(float)
        ci=np.searchsorted(units,keys[:,0])
        for h,frame in frames.items():
            path=original/f'{rung}_h{h}.parquet'
            require(sha(path)==cert['files'][path.name],'Source statistics changed')
            existing=pd.read_parquet(path)
            parts=[]
            for year,g in frame.groupby('year',sort=True):
                g=g.sort_values('ccg')
                cur,lag=annual[year],annual[year-h]
                pos=(cur>0)&(lag>0)
                weight=lag[pos]/g.E_b.to_numpy()[ci[pos]]
                log=np.log(cur[pos])-np.log(lag[pos])
                mass=np.bincount(ci[pos],weights=weight,minlength=len(units))
                ysum=np.bincount(ci[pos],weights=weight*log,minlength=len(units))
                count=np.bincount(ci[pos],minlength=len(units))
                require(np.isfinite(mass).all() and (mass>=0).all() and (mass<=1+1e-12).all(),'Invalid survivor lag mass')
                parts.append(pd.DataFrame({'ccg':units,'year':year,'weighted_log_sum':ysum,'survivor_lag_mass':mass,
                    'weighted_log_mean':np.divide(ysum,mass,out=np.full_like(mass,np.nan),where=mass>0),'log_count':count}))
            stats=pd.concat(parts,ignore_index=True).sort_values(['ccg','year']).reset_index(drop=True)
            require(np.array_equal(stats[['ccg','year']],frame[['ccg','year']]),'County-year key mismatch')
            require(np.array_equal(stats.log_count,existing.log_count),'Log-defined sample differs from unweighted benchmark')
            require(np.array_equal(stats.survivor_lag_mass.gt(0),stats.log_count.gt(0)),'Weighted log support mismatch')
            path=out/f'{rung}_h{h}.parquet'
            stats.to_parquet(path,index=False)
            files[path.name]=sha(path)
            audits.append(dict(horizon=h,rung=rung,log_cells=int(stats.log_count.sum()),
                county_years_excluded=int(stats.log_count.eq(0).sum()),
                min_survivor_lag_mass=float(stats.survivor_lag_mass.min()),
                mean_survivor_lag_mass=float(stats.survivor_lag_mass.mean())))
            print(audits[-1],flush=True)
        del annual
    write_json(out/'certificate.json',dict(status='PASS',files=files,baseline_lock=base_lock,
        parent_certificate_sha256=sha(original/'certificate.json'),builder_sha256=sha(__file__),
        weight='cell_lag_employment_over_full_county_lag_employment',survivor_renormalization=False,
        sample_same_as_unweighted_log=True,audits=audits))


if __name__=='__main__':
    main()
