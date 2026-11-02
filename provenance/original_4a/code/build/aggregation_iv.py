"""Build MPGR/IV aggregation inputs from certified county and geography caches.

Run class: cache_rebuild; reason: outcome_changed. Only the small aggregation
cache is built. County MPGR, spending, and instruments are frozen saved inputs.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from baseline_io import APP,DATA,locked,read_json,require,sha,write_json


def main():
    _,_,lock=locked();geo=DATA/'analysis/aggregation_std_h24';parent=DATA/'analysis/disaggregation_h234'
    gc=read_json(geo/'certificate.json');pc=read_json(parent/'certificate.json')
    for cert in (gc,pc):require(cert['status']=='PASS' and cert['baseline_lock']==lock,'Invalid source certificate')
    require(gc['spec_sha256']==sha(APP/'config/aggregation.json'),'Geography specification changed')
    require(gc['builder_sha256']==sha(APP/'code/build/aggregation.py'),'Geography builder changed')
    require(pc['builder_sha256']==sha(APP/'code/build/disaggregation.py'),'County builder changed')
    out=DATA/'analysis/aggregation_mpgr_iv_h24';require(not out.exists(),'Certified output already exists')
    out.mkdir(parents=True);files={};checks=[]
    for h in (2,4):
        name=f'county_h{h}.parquet';require(sha(parent/name)==pc['files'][name],'County cache changed')
        f=pd.read_parquet(parent/name)[['ccg','year','x','y','z']]
        for rung in ('czone','cea','ea','state'):
            name=f'county_{rung}_h{h}.parquet';require(sha(geo/name)==gc['files'][name],'Geography cache changed')
            c=pd.read_parquet(geo/name).drop(columns='y')
            c=c.merge(f,on=['ccg','year'],how='left',validate='one_to_one',suffixes=('','_saved'))
            require(len(c)==len(f) and np.array_equal(c.z,c.z_saved),'Source support or instrument mismatch')
            c=c.drop(columns='z_saved').rename(columns={'share':'z_agg'})
            require(np.isfinite(c[['x','y','z','z_agg','weight']]).all().all(),'Missing regression input')
            c['weighted_x']=c.weight*c.x;c['weighted_y']=c.weight*c.y
            a=c.groupby(['unit','year'],as_index=False).agg(E=('E','sum'),E_b=('E_b','sum'),
                x_agg=('weighted_x','sum'),z_agg=('z_agg','first'),lag_weighted_mpgr=('weighted_y','sum'))
            a['y']=2*(a.E-a.E_b)/(a.E+a.E_b);a['weight']=1.0
            c=c.drop(columns=['weighted_x','weighted_y']).merge(a[['unit','year','x_agg']],on=['unit','year'],validate='many_to_one')
            require(c[['x','x_agg','y']].notna().all().all() and len(c)==len(f),'Cache augmentation lost observations')
            require(np.abs(c.y).le(2).all() and np.abs(a.y).le(2).all(),'Invalid MPGR bounds')
            for tag,frame in [('county',c),('aggregate',a)]:
                name=f'{tag}_{rung}_h{h}.parquet';frame.to_parquet(out/name,index=False);files[name]=sha(out/name)
            checks.append(dict(horizon=h,rung=rung,n_counties=c.ccg.nunique(),n_county_years=len(c),n_units=a.unit.nunique(),
                dropped_county_years=0,mpgr_vs_lag_weighted_mpgr_max=float(np.max(np.abs(a.y-a.lag_weighted_mpgr)))))
    write_json(out/'certificate.json',dict(status='PASS',baseline_lock=lock,builder_sha256=sha(__file__),files=files,checks=checks,
        geography_certificate_sha256=sha(geo/'certificate.json'),county_certificate_sha256=sha(parent/'certificate.json'),
        outcome='county saved MPGR; aggregate MPGR of summed employment endpoints',estimator='2SLS',
        treatment='county saved x; x_agg=sum(lag_employment_share*x)',instrument='county saved z; z_agg=sum(lag_employment_share*z)',
        weights='lagged employment shares at county level; equal aggregate units',
        protocol_note='User-requested MPGR comparison, not an exact aggregation-invariance claim.'))
    print(pd.DataFrame(checks).to_string(index=False))

if __name__=='__main__':main()
