"""Approved reference-year sensitivity. No growth estimates or inputs changed."""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import t
from balance_tests import BOX,ROOT,APP,HCACHE,KEYS,LABELS,fit_many,joint,holm
from h3_analysis import sha,read,write,require,fit
from baseline_math import wins,demean
from aggregation_ols import hashes

CACHE=BOX/'data/balance_reference_2000'
OUT=BOX/'output/balance_reference_2000'
OLD=BOX/'output/balance_sensitivity'

def build():
    require(not (CACHE/'certificate.json').exists(),'Refusing to overwrite completed cache')
    base=read(BOX/'data/balance_sensitivity/certificate.json')
    inventory=dict(base['inputs'])
    for folder,cert in [(BOX/'data/balance_sensitivity',base),(OLD,read(OLD/'estimates.json'))]:
        require(cert['status']=='PASS','Uncertified input')
        inventory.update({str(folder/k):v for k,v in cert['files'].items()})
    source=APP/'data/intermediate/baseline_roll5_gap5/source51_county_panel.parquet'
    cells=APP/'data/intermediate/disaggregation_h234/naics4_alloc/y2000.parquet'
    snap=APP/'results/manifests/disaggregation_snapshot.json'
    manifest=read(snap)
    require(manifest['status']=='PASS' and sha(cells)==manifest['inputs']['naics4_alloc/y2000.parquet'],'Industry source hash differs')
    for p in [source,cells,snap,BOX/'code/balance_tests.py',BOX/'code/balance_sensitivity.py',
              BOX/'output/pdf/NS_aggregation_summary_tables.pdf',BOX/'code/aggregation_summary_tables.py']:
        inventory[str(p)]=sha(p)
    hashes(inventory)
    prior=pd.read_parquet(BOX/'data/balance_sensitivity/characteristics.parquet').set_index('ccg').sort_index()
    b=pd.read_parquet(source).query('year==2000').set_index('ccg').reindex(prior.index)
    require(len(b)==3058 and np.isfinite(b[['E','P','Y']]).all().all() and b[['E','P','Y']].gt(0).all().all(),'Invalid 2000 county support')
    c=pd.read_parquet(cells);c=c.loc[c.ccg.isin(b.index)].copy()
    require(not c.duplicated(['ccg','code']).any() and np.isfinite(c.emp).all() and c.emp.ge(0).all(),'Invalid cells')
    total=c.groupby('ccg').emp.sum().reindex(b.index)
    require(np.allclose(total,b.E,rtol=1e-12,atol=1e-6),'2000 employment conservation failed')
    chars=pd.DataFrame(index=b.index)
    chars['employment']=b.E;chars['population']=b.P
    chars['jobs_per_resident']=b.E/b.P;chars['income_per_resident']=b.Y/b.P
    manu=c.naics.str[:2].isin(['31','32','33'])
    chars['manufacturing']=c.loc[manu].groupby('ccg').emp.sum().reindex(b.index,fill_value=0)/b.E
    c['share']=c.emp/c.ccg.map(b.E)
    chars['concentration']=(c.share**2).groupby(c.ccg).sum().reindex(b.index)
    require(chars[['manufacturing','concentration']].ge(0).all().all() and chars[['manufacturing','concentration']].le(1).all().all(),'Invalid shares')
    raw=chars.income_per_resident.to_numpy();lo,hi=np.percentile(raw,[1,99]);cap=wins(raw,np.zeros(len(raw),dtype=int))
    require(np.array_equal(cap,np.clip(raw,lo,hi)),'Winsorization differs')
    chars['income_original']=raw;chars['income_per_resident']=cap
    defs=[]
    for key in KEYS:
        v=chars[key];require(np.isfinite(v).all() and v.std(ddof=0)>0,'Invalid characteristic')
        chars['C_'+key]=(v-v.mean())/v.std(ddof=0)
        oldkey='C_income_winsorized' if key=='income_per_resident' else 'C_'+key
        chars['old_'+key]=prior[oldkey]
        defs.append(dict(characteristic=key,mean=float(v.mean()),sd=float(v.std(ddof=0))))
    CACHE.mkdir(parents=True,exist_ok=True);chars.reset_index().to_parquet(CACHE/'characteristics.parquet',index=False)
    hashes(inventory)
    write(CACHE/'certificate.json',dict(status='PASS',inputs=inventory,files={'characteristics.parquet':sha(CACHE/'characteristics.parquet')},
       definitions=defs,income_caps=[float(lo),float(hi)],employment_conservation_max=float(abs(total-b.E).max())))
    print('BUILD PASS: 3058 counties, 2000 employment conservation; income caps',lo,hi,flush=True)

def estimate():
    require(not (OUT/'estimates.json').exists(),'Refusing to overwrite results')
    cert=read(CACHE/'certificate.json');hashes(cert['inputs']);hashes({str(CACHE/k):v for k,v in cert['files'].items()})
    chars=pd.read_parquet(CACHE/'characteristics.parquet').set_index('ccg')
    d=pd.read_parquet(HCACHE/'county_state_h3.parquet')
    require(len(d)==110088 and d.ccg.nunique()==3058 and d.unit.nunique()==51,'Sample changed')
    require(np.allclose(d.groupby(['unit','year']).weight.sum(),1),'Weights changed')
    ys=[]
    for year,prefix in [(1978,'old_'),(2000,'C_')]:
        for key in KEYS:
            name=f'y{year}_{key}';d[name]=d.x*d.ccg.map(chars[prefix+key]);ys.append(name)
    b,se,psi,fs,meta=fit_many(d,ys)
    old=pd.read_csv(OLD/'balance.csv').query("rung=='state' and horizon==3 and panel=='A'").set_index('characteristic')
    oldfs=pd.read_csv(OLD/'first_stages.csv').query("rung=='state' and horizon==3 and panel=='A'").iloc[0]
    require(np.allclose(fs,[oldfs.F_own,oldfs.F_exposure],rtol=1e-8),'First stages changed')
    rows=[];influences=[];joints=[];diffs=[]
    for year,start in [(1978,0),(2000,6)]:
        p=2*t.sf(abs(b[1,start:start+6]/se[1,start:start+6]),50);adj=holm(p)
        for j,key in enumerate(KEYS):
            k=start+j;ci=t.ppf(.975,50)*se[1,k]
            if year==1978:require(np.allclose([b[1,k],se[1,k]],[old.loc[key,'pi'],old.loc[key,'se']],rtol=1e-7,atol=1e-10),'1978 not reproduced')
            check,_=fit(d,ys[k],['x','x_agg'],['z','z_agg'])
            require(np.allclose([check['x_agg']['beta'],check['x_agg']['se']],[b[1,k],se[1,k]],rtol=1e-7,atol=1e-10),'Single-outcome fit mismatch')
            rows.append(dict(year=year,characteristic=key,rho=b[0,k],rho_se=se[0,k],pi=b[1,k],se=se[1,k],p=p[j],holm_p=adj[j],ci_low=b[1,k]-ci,ci_high=b[1,k]+ci,F_county=fs[0],F_regional=fs[1]))
            influences.extend(dict(year=year,characteristic=key,cluster=int(label),psi=float(v)) for label,v in zip(meta['labels'],psi[:,1,k]))
        joints.append(dict(year=year,**joint(b[1,start:start+6],psi[:,1,start:start+6],50)))
    for j,key in enumerate(KEYS):
        diff=b[1,6+j]-b[1,j];ds=np.linalg.norm(psi[:,1,6+j]-psi[:,1,j])
        diffs.append(dict(characteristic=key,difference=diff,se=ds,p=2*t.sf(abs(diff/ds),50)))
    OUT.mkdir(parents=True,exist_ok=True)
    for name,values in [('balance.csv',rows),('joint.csv',joints),('paired_differences.csv',diffs),('cluster_influences.csv',influences)]:pd.DataFrame(values).to_csv(OUT/name,index=False)
    # Installed Stata check: same FE residuals and weights, separate IV implementation.
    cols=ys[6:]+['x','x_agg','z','z_agg'];raw=d[cols].to_numpy();scale=np.maximum(abs(raw).max(axis=0),1e-30)
    m=demean(raw/scale,pd.factorize(d.ccg)[0],pd.factorize(d.year)[0],d.weight.to_numpy())*scale
    export=pd.DataFrame(m,columns=['y'+str(i) for i in range(6)]+['x','x_agg','z','z_agg'])
    export['weight']=d.weight.to_numpy();export['unit']=pd.factorize(d.unit,sort=True)[0].astype(np.int32)
    export.to_stata(CACHE/'stata_inputs.dta',write_index=False,version=118)
    logs=ROOT/'batch_logs/macro_paul_balance_2000';logs.mkdir(exist_ok=True)
    lines=['clear all','set more off',f'log using "{(logs/"stata_detail.log").as_posix()}", text replace',
      f'use "{(CACHE/"stata_inputs.dta").as_posix()}", clear',f'file open r using "{(OUT/"stata.csv").as_posix()}", write replace',
      'file write r "characteristic,rho,rho_se,pi,se,F_county,F_regional" _n']
    for i,key in enumerate(KEYS):
        lines += [f'ivreg2 y{i} (x x_agg=z z_agg) [aw=weight], nocons cluster(unit) small sdofminus({meta["k"]-2}) first','matrix fs=e(first)',
          f'file write r "{key}," %21.15g (_b[x]) "," %21.15g (_se[x]) "," %21.15g (_b[x_agg]) "," %21.15g (_se[x_agg]) "," %21.15g (fs[4,1]) "," %21.15g (fs[4,2]) _n']
    lines += ['file close r','log close','exit, clear'];(CACHE/'verify.do').write_text('\n'.join(lines)+'\n')
    hashes(cert['inputs']);write(OUT/'estimates.json',dict(status='AWAITING_STATA',inputs=cert['inputs'],files={p.name:sha(p) for p in OUT.glob('*.csv')},code_sha256=sha(__file__)))
    print(pd.DataFrame(rows)[['year','characteristic','pi','se','p']].to_string(index=False),flush=True)

def verify():
    cert=read(OUT/'estimates.json');hashes(cert['inputs']);hashes({str(OUT/k):v for k,v in cert['files'].items()})
    a=pd.read_csv(OUT/'balance.csv').query('year==2000').set_index('characteristic')
    b=pd.read_csv(OUT/'stata.csv').set_index('characteristic').reindex(a.index)
    cols=['rho','rho_se','pi','se','F_county','F_regional']
    require(len(b)==6 and np.allclose(a[cols],b[cols],rtol=1e-7,atol=1e-9),'Independent Stata mismatch')
    cert.update(status='PASS',stata_verified_regressions=6,original_files_unchanged=True)
    cert['files']['stata.csv']=sha(OUT/'stata.csv');write(OUT/'estimates.json',cert)
    print('PASS: all six 2000 regressions reproduced by Stata; all six 1978 diagnostics reproduced; first stages unchanged.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','estimate','verify']);a=p.parse_args();globals()[a.stage]()
