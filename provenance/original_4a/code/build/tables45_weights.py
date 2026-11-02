"""Add full-county lag shares to certified NAICS6 log estimation cells.

MPGR run class: cache_augment. No outcomes or samples are reconstructed.
Input: analysis/table2_cells and certified county horizon frames.
"""
from pathlib import Path
import sys
import duckdb
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from baseline_io import APP,DATA,locked,read_json,require,sha,write_json
_,_,lock=locked();src=DATA/'analysis/table2_cells';cert=read_json(src/'certificate.json')
require(cert['status']=='PASS' and cert['baseline_lock']==lock,'Cell cache mismatch')
require(sha(APP/'code/build/table2_cells.py')==cert['builder_sha256'],'Cell builder changed')
county=DATA/'analysis/disaggregation_h234';pc=read_json(county/'certificate.json')
require(sha(county/'certificate.json')==cert['parent_certificate_sha256'],'County certificate changed')
out=DATA/'analysis/tables45_weights';out.mkdir(parents=True,exist_ok=True)
require(not (out/'certificate.json').exists(),'Refusing to rebuild certified cache')
files={};counts=[]
with duckdb.connect() as con:
    for h in (2,3,4):
        for name,digest in cert['files'].items():
            if name.startswith(f'h{h}/'):require(sha(src/name)==digest,'Cell hash mismatch')
        p=county/f'county_h{h}.parquet';require(sha(p)==pc['files'][p.name],'County hash mismatch')
        f=pd.read_parquet(p,columns=['ccg','year','E_b']);require(not f.duplicated(['ccg','year']).any(),'Duplicate county keys')
        con.register('f',f)
        dest=out/f'h{h}.parquet'
        require(not dest.exists(),'Refusing to overwrite weight cache')
        con.execute(f'''COPY (SELECT d.ccg,d.year,d.industry,d.fe_county,d.fe_industry_year,d.x,d.z,d.y_log AS y,
            d.lag/f.E_b AS w_lag FROM read_parquet(?) d JOIN f USING(ccg,year)
            WHERE d.y_log IS NOT NULL AND d.w_log>0) TO '{dest}' (FORMAT PARQUET)''',[str(src/f'h{h}/y*.parquet')])
        row=con.execute('''SELECT count(*) AS n,count(DISTINCT (ccg,year,industry)) AS unique_n,
            count(*) FILTER (WHERE NOT isfinite(w_lag) OR w_lag<=0 OR w_lag>1.0000000001) AS bad,
            count(DISTINCT (ccg,year)) AS county_years FROM read_parquet(?)''',[str(dest)]).fetchone()
        require(row[0]==row[1] and row[2]==0,'Invalid augmented cache')
        reference=con.execute('SELECT count(*) FROM read_parquet(?) WHERE y_log IS NOT NULL AND w_log>0',[str(src/f'h{h}/y*.parquet')]).fetchone()[0]
        require(row[0]==reference,'Augmentation changed log support')
        files[dest.name]=sha(dest);counts.append(dict(horizon=h,log_cells=row[0],county_years=row[3]));print(counts[-1],flush=True)
write_json(out/'certificate.json',dict(status='PASS',files=files,baseline_lock=lock,builder_sha256=sha(__file__),
    source_certificate_sha256=sha(src/'certificate.json'),counts=counts,weight='lag employment / full county lag employment; no survivor renormalization'))
