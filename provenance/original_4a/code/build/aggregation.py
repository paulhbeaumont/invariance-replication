"""Build standard-growth aggregation inputs from certified county endpoints.

Run class: cache_augment. Input: data/analysis/disaggregation_h234 county caches.
Geography follows the source-57 first-nonmissing rule, with whole CCGs retained.
No instrument reconstruction, growth caps, estimation, or source-panel rebuild.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from baseline_io import APP,DATA,locked,read_json,require,sha,write_json


def main():
    _,_,lock=locked()
    cfg=APP/'config/aggregation.json';spec=read_json(cfg)
    parent=DATA/'analysis/disaggregation_h234';pc=read_json(parent/'certificate.json')
    require(pc['status']=='PASS' and pc['baseline_lock']==lock,'County certificate mismatch')
    require(sha(APP/'code/build/disaggregation.py')==pc['builder_sha256'],'County builder changed')
    for name,record in spec['sources'].items():
        require(sha(APP/'config'/name)==record['sha256'],'Geography source changed')
    x=pd.read_csv(APP/'config/aggregation_county_ccg_crosswalk.csv',dtype=str)
    x=x[x.ccg_id.ne('unplaceable')].copy();x['ccg']=x.ccg_id.astype(int)
    b=pd.read_csv(APP/'config/aggregation_county_bea_area_2004.csv',dtype=str)
    m=x.merge(b,on='county_fips5',how='left',validate='many_to_one',sort=False)
    maps={k:m.dropna(subset=[c]).drop_duplicates('ccg').set_index('ccg')[c]
          for k,c in [('czone','ccz_id'),('cea','cea_code'),('ea','ea_code')]}
    require(maps['cea'].loc[2231]=='761' and maps['cea'].loc[4012]=='789','Approved CEA assignments differ')
    out=DATA/'analysis/aggregation_std_h24';require(not out.exists(),'Aggregation cache already exists')
    out.mkdir(parents=True)
    files={};checks=[];inputs={}
    for h in spec['horizons']:
        path=parent/f'county_h{h}.parquet'
        require(sha(path)==pc['files'][path.name],'County cache changed')
        inputs[path.name]=sha(path)
        f=pd.read_parquet(path)
        require(not f.duplicated(['ccg','year']).any(),'Duplicate county-year')
        require(f.year.max()==2016 and f.ccg.nunique()==3058,'County support differs')
        require(f.groupby('year').ccg.nunique().eq(3058).all(),'Unbalanced county support')
        require(np.isfinite(f[['E','E_b','z','P_b']]).all().all() and (f[['E','E_b','P_b']]>0).all().all(),'Invalid county endpoints')
        for rung in spec['rungs']:
            g=f[['ccg','year','state','E','E_b','P_b','z']].copy()
            g['unit']=g.state if rung=='state' else g.ccg.map(maps[rung])
            require(g.unit.notna().all(),'Unmapped county group: '+rung)
            g['unit']=pd.factorize(g.unit,sort=True)[0].astype('int32')
            require(g.groupby('ccg').unit.nunique().eq(1).all(),'Time-varying geography')
            g['weight']=g.E_b/g.groupby(['unit','year']).E_b.transform('sum')
            g['y']=(g.E-g.E_b)/g.E_b
            g['weighted_y']=g.weight*g.y;g['weighted_z']=g.weight*g.z
            a=g.groupby(['unit','year'],as_index=False).agg(E=('E','sum'),E_b=('E_b','sum'),
                share=('weighted_z','sum'),aligned_y=('weighted_y','sum'),weight_sum=('weight','sum'),n_counties=('ccg','size'))
            a['y']=(a.E-a.E_b)/a.E_b;a['weight']=1.0
            err=float(np.max(np.abs(a.y-a.aligned_y)))
            require(err<1e-12 and np.allclose(a.weight_sum,1,rtol=0,atol=1e-12),'Aggregation reconstruction failed')
            require(np.allclose(a.groupby('year')[['E','E_b']].sum(),f.groupby('year')[['E','E_b']].sum(),rtol=1e-13,atol=1e-6),'Employment totals differ')
            g=g.merge(a[['unit','year','share']],on=['unit','year'],how='left',validate='many_to_one')
            require(len(g)==len(f) and g.share.notna().all(),'County support changed')
            g=g.drop(columns=['weighted_y','weighted_z'])
            for tag,frame in [('county',g),('aggregate',a)]:
                name=f'{tag}_{rung}_h{h}.parquet';frame.to_parquet(out/name,index=False);files[name]=sha(out/name)
            checks.append(dict(horizon=h,rung=rung,n_counties=int(g.ccg.nunique()),n_county_years=len(g),n_units=int(a.unit.nunique()),n_aggregate_years=len(a),first_year=int(g.year.min()),last_year=int(g.year.max()),growth_reconstruction_max=err,dropped_county_years=0))
    write_json(out/'certificate.json',dict(status='PASS',baseline_lock=lock,spec_sha256=sha(cfg),builder_sha256=sha(__file__),
        parent_certificate_sha256=sha(parent/'certificate.json'),inputs=inputs,files=files,checks=checks))
    print(pd.DataFrame(checks).to_string(index=False))

if __name__=='__main__':main()
