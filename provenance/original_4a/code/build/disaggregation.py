"""Build h=2,3,4 county and cell sufficient-statistic caches, without estimation.

Run class: cache_rebuild, required_key_absent (industry outcomes).
The MPGR statistic is independently reconstructed from all cells. Log sufficient
statistics preserve the number and mean growth of surviving cells, not county
employment growth. Source cells remain unchanged in the frozen annual snapshots.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
import numpy as np
import pandas as pd
from baseline_io import APP,DATA,locked,load_cache,read_json,write_json,require,sha
from disaggregation_math import county_frame


def main():
    spec,sources,base_lock=locked()
    cfg=read_json(APP/'config/disaggregation.json')
    lock=read_json(APP/'config/disaggregation_lock.json')
    require(sha(APP/'config/disaggregation.json')==lock['config_sha256'],'Disaggregation configuration changed')
    require(sha(APP/'code/lib/disaggregation_math.py')==lock['math_sha256'],'County-frame extraction changed')
    snap=read_json(APP/'results/manifests/disaggregation_snapshot.json')
    require(snap['status']=='PASS' and snap['baseline_lock']==base_lock,'Cell snapshot not certified for baseline')
    snapshot=DATA/'intermediate/disaggregation_h234'
    for name,digest in snap['inputs'].items():
        require(sha(snapshot/name)==digest,f'Cell snapshot changed: {name}')
    out=DATA/'analysis/disaggregation_h234'
    require(not (out/'certificate.json').exists(),'Certified build already exists; do not rebuild')
    out.mkdir(parents=True,exist_ok=True)
    frames={}
    files={}
    for h in cfg['horizons']:
        path=out/f'county_h{h}.parquet'
        if path.exists():
            f=pd.read_parquet(path)
        elif h==2:
            f,_,_=load_cache()
        else:
            s=dict(spec,horizon=h,first_year=cfg['first_years'][str(h)])
            s['n_observations']=3058*(2016-s['first_year']+1)
            f=county_frame(s,sources)
        f['log_y']=np.log(f.E)-np.log(f.E_b)
        frames[h]=f
        path=out/f'county_h{h}.parquet'
        if not path.exists():
            f.to_parquet(path,index=False)
        files[path.name]=sha(path)
    units=np.sort(frames[2].ccg.unique())
    checks=[]
    log_blind=[]
    for rung in cfg['rungs']:
        annual={}
        keys=None
        for year in range(1978,2017):
            f=pd.read_parquet(snapshot/rung/f'y{year}.parquet',columns=['ccg','code','emp'])
            f=f[f.ccg.isin(units)].sort_values(['ccg','code']).reset_index(drop=True)
            k=f[['ccg','code']].to_numpy(np.int64)
            if keys is None:
                keys=k
                cell_count=np.bincount(np.searchsorted(units,keys[:,0]),minlength=len(units))
                pd.DataFrame(keys,columns=['ccg','industry']).to_parquet(out/f'{rung}_dictionary.parquet',index=False)
            require(np.array_equal(keys,k),'Source cell support changed')
            annual[year]=f.emp.to_numpy(np.float64)
        ci=np.searchsorted(units,keys[:,0])
        for h,frame in frames.items():
            saved=out/f'{rung}_h{h}.parquet'
            if saved.exists():
                stats=pd.read_parquet(saved)
                require(np.array_equal(stats[['ccg','year']],frame[['ccg','year']]),'Saved keys differ')
                for year,g in stats.groupby('year'):
                    ref=frame[frame.year.eq(year)].sort_values('ccg')
                    checks.append(dict(horizon=h,rung=rung,year=int(year),
                        reconstruction_error=float(np.max(np.abs(g.mpgr_mean.to_numpy()*g.mpgr_mass.to_numpy()-ref.y.to_numpy()))),
                        weight_error=float(np.max(np.abs(g.mpgr_mass-1))),n_cells=int(g.total_cells.sum()),
                        log_cells=int(g.log_count.sum()),log_undefined=int((g.total_cells-g.log_count).sum())))
                files[saved.name]=sha(saved)
                continue
            parts=[]
            for year,g in frame.groupby('year',sort=True):
                g=g.sort_values('ccg')
                cur,lag=annual[year],annual[year-h]
                den=cur+lag
                denc=(g.E+g.E_b).to_numpy()
                mpgr=np.divide(2*(cur-lag),den,out=np.zeros_like(den),where=den!=0)
                w=den/denc[ci]
                pos=(cur>0)&(lag>0)
                log=np.zeros_like(den)
                log[pos]=np.log(cur[pos])-np.log(lag[pos])
                mass=np.bincount(ci,weights=w,minlength=len(units))
                summed=np.bincount(ci,weights=w*mpgr,minlength=len(units))
                count=np.bincount(ci,weights=pos.astype(np.int64),minlength=len(units)).astype(np.int64)
                logsum=np.bincount(ci,weights=log,minlength=len(units))
                error=float(np.max(np.abs(summed-g.y.to_numpy())))
                weight_error=float(np.max(np.abs(mass-1)))
                require(error<cfg['reconstruction_atol'],f'MPGR reconstruction failed: h={h} {rung} {year}')
                require(weight_error<cfg['reconstruction_atol'],'Midpoint weights do not sum to one')
                for county in units[count==0]:
                    log_blind.append(dict(horizon=h,rung=rung,ccg=int(county),year=int(year),reason='no_positive_both_endpoints_industry'))
                part=pd.DataFrame({'ccg':units,'year':year,'mpgr_mean':summed/mass,'mpgr_mass':mass,
                                   'log_mean':np.divide(logsum,count,out=np.full_like(logsum,np.nan),where=count>0),'log_count':count,'total_cells':cell_count,
                                   'zero_both':np.bincount(ci,weights=((cur==0)&(lag==0)).astype(int),minlength=len(units)).astype(int),
                                   'entrant_cells':np.bincount(ci,weights=((cur>0)&(lag==0)).astype(int),minlength=len(units)).astype(int),
                                   'exit_cells':np.bincount(ci,weights=((cur==0)&(lag>0)).astype(int),minlength=len(units)).astype(int)})
                parts.append(part)
                checks.append(dict(horizon=h,rung=rung,year=int(year),reconstruction_error=error,weight_error=weight_error,
                                   n_cells=len(cur),log_cells=int(pos.sum()),log_undefined=int((~pos).sum())))
            stats=pd.concat(parts,ignore_index=True).sort_values(['ccg','year']).reset_index(drop=True)
            require(np.array_equal(stats[['ccg','year']],frame[['ccg','year']]),'Sufficient-statistic keys differ from county frame')
            path=out/f'{rung}_h{h}.parquet'
            stats.to_parquet(path,index=False)
            files[path.name]=sha(path)
            print(f'BUILT {rung} h={h}: {len(stats)} county-year sufficient-statistic rows',flush=True)
        del annual
    audit=pd.DataFrame(checks)
    audit.to_csv(APP/'results/manifests/disaggregation_reconstruction.csv',index=False)
    pd.DataFrame(log_blind,columns=['horizon','rung','ccg','year','reason']).to_csv(APP/'results/manifests/disaggregation_log_blind.csv',index=False)
    cert=dict(status='PASS',files=files,baseline_lock=base_lock,config_lock=lock,
              builder_sha256=sha(__file__),source_manifest_sha256=sha(APP/'results/manifests/disaggregation_snapshot.json'),
              reconstruction_max=float(audit.reconstruction_error.max()),weight_error_max=float(audit.weight_error.max()),
              no_county_year_drops=True,source_cells_unchanged=True,log_blind_count=len(log_blind),
              log_policy='Do not estimate a cell-log specification if any county-year has zero surviving cells')
    write_json(out/'certificate.json',cert)
    print(cert)


if __name__=='__main__':
    main()
