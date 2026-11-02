"""Annual p1/p99 county growth, with aggregates formed from capped county rates.

Run class: cache_augment. Input: certified aggregation_mpgr_iv_h24.
No employment, spending, instrument, geography, or support changes.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from baseline_io import APP,DATA,read_json,require,sha,write_json
from aggregation_design import specification,prepare,SPEC
OUT=DATA/'analysis/aggregation_winsorized_h24'


def annual_cap(f):
    cuts=f.groupby('year').y.quantile([.01,.99],interpolation='linear').unstack()
    y=f.y.clip(lower=f.year.map(cuts[.01]),upper=f.year.map(cuts[.99]))
    return y,cuts


def main():
    s=specification();parent=DATA/s['cache'];cert=read_json(parent/'certificate.json')
    require(cert['status']=='PASS','Uncertified parent')
    pop=pd.read_parquet(DATA/'analysis/county_population_1975.parquet').set_index('ccg').population_1975
    require(not (OUT/'certificate.json').exists(),'Certified winsorized cache exists; reuse it')
    OUT.mkdir(parents=True,exist_ok=True);files={};inputs={};counts={};cutrows=[]
    trial=pd.DataFrame({'year':[2000,2000],'y':[0.,100.]})
    capped,_=annual_cap(trial);require(np.allclose(capped,[1.,99.]),'Annual cap fixture failed')
    for h in s['horizons']:
        county_reference=None
        for rung in s['rungs']:
            frames=[]
            for level in ['county','aggregate']:
                name=f'{level}_{rung}_h{h}.parquet';require(sha(parent/name)==cert['files'][name],'Parent changed')
                inputs[name]=sha(parent/name);frames.append(pd.read_parquet(parent/name))
            c,a=prepare(*frames,pop)
            c['y_uncapped']=c.y
            c['y'],cuts=annual_cap(c)
            check=c.set_index(['ccg','year']).y.sort_index()
            if county_reference is None:
                county_reference=check
                for year,row in cuts.iterrows():cutrows.append(dict(horizon=h,year=int(year),p1=row[.01],p99=row[.99]))
                counts[str(h)]=dict(n=len(c),capped=int((c.y!=c.y_uncapped).sum()))
            else:require(check.equals(county_reference),'County caps differ by geographic partition')
            c['weight_equal']=c.aggregate_population1975/c.groupby(['unit','year']).ccg.transform('size')
            require(np.allclose(c.groupby(['unit','year']).weight_equal.sum(),c.groupby(['unit','year']).weight.sum()),'Equal weights change aggregate totals')
            a['y_uncapped']=a.y
            agg=(c.construction_share*c.y).groupby([c.unit,c.year]).sum()
            a['y']=pd.MultiIndex.from_frame(a[['unit','year']]).map(agg).to_numpy()
            require(np.isfinite(c.y).all() and np.isfinite(a.y).all(),'Invalid capped outcome')
            for level,f in [('county',c),('aggregate',a)]:
                name=f'{level}_{rung}_h{h}.parquet';f.to_parquet(OUT/name,index=False);files[name]=sha(OUT/name)
    pd.DataFrame(cutrows).to_csv(OUT/'annual_cutoffs.csv',index=False);files['annual_cutoffs.csv']=sha(OUT/'annual_cutoffs.csv')
    write_json(OUT/'certificate.json',dict(status='PASS',spec_sha256=sha(SPEC),builder_sha256=sha(__file__),inputs=inputs,files=files,
        counts=counts,quantiles='Unweighted county-year distribution, by year and horizon; linear interpolation',
        aggregate_growth='Employment-share mean of capped county standard growth; no second cap',no_drops=True))
    print(counts)

if __name__=='__main__':main()
