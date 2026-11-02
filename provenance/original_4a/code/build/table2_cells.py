"""Build NAICS6 horizon cells for the requested industry-year FE comparison.

MPGR run class: cache_rebuild; reason: required_key_absent (industry in collapsed cache).
Certified annual employment and county regressors are inputs, never raw sources.
"""
from pathlib import Path
import sys
import duckdb
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from baseline_io import APP,DATA,locked,read_json,require,sha,write_json

_,_,lock=locked()
src=DATA/'intermediate/naics6_residual'
c=read_json(src/'certificate.json')
require(c['status']=='PASS' and c['baseline_lock']==lock,'Employment certificate mismatch')
require(sha(APP/'code/build/naics6_cells.py')==c['builder_sha256'],'Employment builder changed')
for name,digest in c['files'].items():require(sha(src/name)==digest,'Employment hash mismatch')
parent=DATA/'analysis/disaggregation_h234';pc=read_json(parent/'certificate.json')
require(pc['status']=='PASS' and pc['baseline_lock']==lock,'County cache mismatch')
out=DATA/'analysis/table2_cells';out.mkdir(parents=True,exist_ok=False)
files={};checks=[];baseline_rows=[]
with duckdb.connect() as con:
    codes=con.execute('SELECT DISTINCT code,parent FROM read_parquet(?) ORDER BY code',[str(src/'y*.parquet')]).df()
    codes['industry']=np.arange(len(codes),dtype=np.int32)
    codes.to_parquet(out/'industry_dictionary.parquet',index=False)
    con.register('codes',codes)
    for h in (2,3,4):
        path=parent/f'county_h{h}.parquet';require(sha(path)==pc['files'][path.name],'County hash mismatch')
        f=pd.read_parquet(path);directory=out/f'h{h}';directory.mkdir()
        con.register('f',f)
        support=con.execute('SELECT count(*) FROM (SELECT DISTINCT ccg,code FROM read_parquet(?))',[str(src/'y*.parquet')]).fetchone()[0]
        n_cells=0;log_empty=0;stats=[]
        for year,g in f.groupby('year'):
            con.register('g',g)
            d=con.execute('''WITH pair AS (
                SELECT coalesce(a.ccg,b.ccg) AS ccg,coalesce(a.code,b.code) AS code,
                coalesce(a.emp,0) AS cur,coalesce(b.emp,0) AS lag
                FROM read_parquet(?) a FULL JOIN read_parquet(?) b USING(ccg,code)
            ), cell AS (SELECT p.*,g.year,g.fe_county,g.fe_year,g.x,g.z,g.E+g.E_b AS county_den,
                codes.industry, codes.parent FROM pair p JOIN g USING(ccg) JOIN codes USING(code)
                WHERE cur+lag>0)
            SELECT *,CAST(industry*100+year-1978 AS INTEGER) AS fe_industry_year,
                2*(cur-lag)/(cur+lag) AS y_mpgr,(cur+lag)/county_den AS w_mpgr,
                CASE WHEN cur>0 AND lag>0 THEN ln(cur)-ln(lag) ELSE NULL END AS y_log,
                CASE WHEN cur>0 AND lag>0 THEN (cur+lag)/sum(CASE WHEN cur>0 AND lag>0 THEN cur+lag ELSE 0 END) OVER(PARTITION BY ccg) ELSE 0 END AS w_log
            FROM cell''',[str(src/f'y{year}.parquet'),str(src/f'y{year-h}.parquet')]).df()
            require(not d.duplicated(['ccg','industry']).any(),'Duplicate cells')
            require(np.isfinite(d[['y_mpgr','w_mpgr','x','z']]).all().all(),'Nonfinite cell')
            d['weighted_y']=d.y_mpgr*d.w_mpgr
            ag=d.groupby('ccg',sort=True).agg(y=('weighted_y','sum'),mass=('w_mpgr','sum'))
            ref=g.sort_values('ccg')
            error=float(np.max(np.abs(ag.y.to_numpy()-ref.y.to_numpy())))
            require(error<1e-10 and np.allclose(ag.mass,1,rtol=0,atol=1e-10),'MPGR reconstruction failure')
            valid=d.y_log.notna()
            require(np.allclose(d.loc[valid].groupby('ccg').w_log.sum(),1,atol=1e-10),'Survivor weights invalid')
            checks.append(dict(horizon=h,year=int(year),mpgr_error=error,log_empty_county_years=int(len(g)-d.loc[valid].ccg.nunique())))
            stats.append(pd.DataFrame({'ccg':ag.index,'year':year,'y':ag.y.to_numpy(),'mass':ag.mass.to_numpy()}))
            n_cells+=len(d);log_empty+=checks[-1]['log_empty_county_years']
            d=d.drop(columns=['code','weighted_y'])
            path=directory/f'y{year}.parquet';d.to_parquet(path,index=False);files[str(path.relative_to(out))]=sha(path)
        s=pd.concat(stats).sort_values(['ccg','year']).reset_index(drop=True)
        require(np.array_equal(s[['ccg','year']],f[['ccg','year']]),'Collapsed key mismatch')
        path=out/f'mpgr_h{h}.parquet';s.to_parquet(path,index=False);files[path.name]=sha(path)
        baseline_rows.append(dict(horizon=h,active_cells=n_cells,balanced_cells=support*f.year.nunique(),county_years=len(f),log_empty_county_years=log_empty))
        print(baseline_rows[-1],flush=True)
files['industry_dictionary.parquet']=sha(out/'industry_dictionary.parquet')
write_json(out/'certificate.json',dict(status='PASS',files=files,baseline_lock=lock,builder_sha256=sha(__file__),
    source_certificate_sha256=sha(src/'certificate.json'),parent_certificate_sha256=sha(parent/'certificate.json'),
    schema=['ccg','industry','parent','year','fe_county','fe_year','fe_industry_year','cur','lag','county_den','y_mpgr','w_mpgr','y_log','w_log','x','z'],
    checks=checks,horizon_counts=baseline_rows))
