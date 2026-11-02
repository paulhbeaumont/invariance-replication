"""Approved separate-control comparison; imports unchanged verified IV estimator."""
from heterogeneity_levels import iv
import argparse
import numpy as np
import pandas as pd
from scipy.stats import t
from h3_analysis import BOX,ROOT,APP,sha,read,write,require
from baseline_math import demean,ssc_k
from aggregation_ols import hashes

HCACHE=BOX/'data/horizon_2345'
INPUT=BOX/'data/heterogeneity_levels'
CACHE=BOX/'data/heterogeneity_separate'
OUT=BOX/'output/heterogeneity_separate'
KEYS=['employment','income_per_resident','manufacturing']

def estimate():
    require(not (OUT/'estimates.json').exists(),'Refusing to overwrite estimates')
    cert=read(INPUT/'certificate.json');require(cert['status']=='PASS','Uncertified existing cache')
    hashes(cert['inputs']);hashes({str(INPUT/k):v for k,v in cert['files'].items()})
    cert['inputs'].update({str(INPUT/k):v for k,v in cert['files'].items()})
    previous=BOX/'output/heterogeneity_levels';pc=read(previous/'estimates.json');require(pc['status']=='PASS','Original estimates not verified')
    cert['inputs'].update({str(previous/k):v for k,v in pc['files'].items()})
    for p in [INPUT/'certificate.json',previous/'estimates.json',BOX/'code/heterogeneity_levels.py',BOX/'code/render_heterogeneity_levels.py',BOX/'output/pdf/NS_spillovers_heterogeneity_levels.pdf']:
        cert['inputs'][str(p)]=sha(p)
    c=pd.read_parquet(INPUT/'county.parquet');a=pd.read_parquet(INPUT/'state.parquet')
    previous_coef=pd.read_csv(previous/'coefficients.csv')
    CACHE.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(BOX/'output/horizon_2345/aggregation.csv').query("horizon==3 and rung=='state' and growth=='std'").iloc[0]
    rows=[];summary=[];jobs=[];inf=[]
    for controls in range(4):
        ids=[] if controls==0 else [controls];exo=[f'A{i}' for i in ids]
        ce=['x','x_agg']+[f'xc{i}' for i in ids]+[f'Q{i}' for i in ids]
        cz=['z','z_agg']+[f'zc{i}' for i in ids]+[f'R{i}' for i in ids]
        ae=['x_agg']+[f'Q{i}' for i in ids];az=['z_agg']+[f'R{i}' for i in ids]
        fits={}
        for model,d,en,zz,fe in [('county',c,ce,cz,'ccg'),('state',a,ae,az,'unit')]:
            r,psi,export,meta=iv(d,en,zz,exo,fe);fits[model]=pd.DataFrame(r).set_index('variable')
            rows.extend(dict(controls=controls,model=model,**v) for v in r)
            for j,v in enumerate(en+exo):inf.extend(dict(controls=controls,model=model,variable=v,cluster=int(g),psi=float(value)) for g,value in enumerate(psi[:,j]))
            name=f'{model}_{controls}';export.to_stata(CACHE/f'{name}.dta',write_index=False,version=118)
            jobs.append(dict(name=name,controls=controls,model=model,endog=en,excluded=zz,exog=exo,**meta))
            if controls in [0,1]:
                ref=previous_coef.loc[(previous_coef.controls==controls)&(previous_coef.model==model)].set_index('variable')
                require(np.allclose(fits[model][['beta','se','p']],ref.loc[fits[model].index,['beta','se','p']],rtol=1e-7,atol=1e-12),'Previous baseline or employment fit differs')
            if controls==0:
                for variable,prefix in ([('x','augmented'),('x_agg','exposure')] if model=='county' else [('x_agg','aggregate')]):
                    for key in ['beta','se','F','p']:require(np.isclose(fits[model].loc[variable,key],old[prefix+'_'+key],rtol=1e-7,atol=1e-12),'Original baseline not reproduced')
        rr=fits['county'].loc['x_agg'];res=fits['state'].loc['x_agg','beta']-fits['county'].loc['x','beta']-rr.beta
        summary.append(dict(controls=controls,delta=rr.beta,se=rr.se,p=rr.p,identity_residual=res,n_excluded=len(cz)))
        print('CONTROLS',controls,'delta/SE (x1e5)',rr.beta*1e5,rr.se*1e5,'p',rr.p,flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,data in [('coefficients.csv',rows),('delta.csv',summary),('influences.csv',inf),('design_audit.csv',jobs)]:pd.DataFrame(data).to_csv(OUT/name,index=False)
    logs=ROOT/'batch_logs/macro_paul_heterogeneity_separate';logs.mkdir(exist_ok=True)
    lines=['clear all','set more off',f'log using "{(logs/"stata_detail.log").as_posix()}", text replace',
      f'file open r using "{(OUT/"stata.csv").as_posix()}", write replace','file write r "controls,model,variable,beta,se,F,conditional_F" _n']
    for job in jobs:
        en=job['endog'];ex=job['exog'];zz=job['excluded']
        lines += [f'use "{(CACHE/(job["name"]+".dta")).as_posix()}", clear',
         'ivreg2 y '+' '.join(ex)+' ('+' '.join(en)+' = '+' '.join(zz)+f') [aw=weight], nocons cluster(unit) small sdofminus({job["k"]-len(en)-len(ex)}) first','matrix fs=e(first)']
        for j,v in enumerate(en+ex):
            fs=f'fs[4,{j+1}]' if j<len(en) else '.';sw=f'fs[8,{j+1}]' if j<len(en) else '.'
            lines.append(f'file write r "{job["controls"]},{job["model"]},{v}," %21.15g (_b[{v}]) "," %21.15g (_se[{v}]) "," %21.15g ({fs}) "," %21.15g ({sw}) _n')
    lines+=['file close r','log close','exit, clear'];(CACHE/'verify.do').write_text('\n'.join(lines)+'\n')
    hashes(cert['inputs']);write(OUT/'estimates.json',dict(status='AWAITING_STATA',inputs=cert['inputs'],files={p.name:sha(p) for p in OUT.glob('*.csv')},estimator_sha256=sha(__file__)))

def verify():
    cert=read(OUT/'estimates.json');hashes(cert['inputs']);hashes({str(OUT/k):v for k,v in cert['files'].items()})
    keys=['controls','model','variable'];a=pd.read_csv(OUT/'coefficients.csv').set_index(keys)
    b=pd.read_csv(OUT/'stata.csv',na_values=['.'],skipinitialspace=True).set_index(keys).reindex(a.index)
    b=b.replace(r'^\s*\.\s*$',np.nan,regex=True).apply(pd.to_numeric,errors='raise')
    require(len(a)==len(b) and np.allclose(a[['beta','se']],b[['beta','se']],rtol=1e-6,atol=1e-10),'Stata coefficients or SE differ')
    mask=a.F.notna();require(np.allclose(a.loc[mask,'F'],b.loc[mask,'F'],rtol=1e-6,atol=1e-7),'Ordinary first stages differ')
    cert.update(status='PASS',stata_verified_regressions=8,stata_verified_coefficients=len(a),original_hashes_unchanged=True)
    cert['files']['stata.csv']=sha(OUT/'stata.csv');write(OUT/'estimates.json',cert)
    print('VERIFIED',len(a),'coefficients from 8 regressions; old files unchanged.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['estimate','verify']);a=p.parse_args();globals()[a.stage]()
