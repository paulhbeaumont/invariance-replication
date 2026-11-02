"""Approved eight-instrument matched-controls specification; isolated outputs."""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import t
from h3_analysis import BOX,ROOT,APP,sha,read,write,require
from baseline_math import demean,ssc_k
from aggregation_ols import hashes

HCACHE=BOX/'data/horizon_2345'
CACHE=BOX/'data/heterogeneity_levels'
OUT=BOX/'output/heterogeneity_levels'
KEYS=['employment','income_per_resident','manufacturing']

def build():
    require(not (CACHE/'certificate.json').exists(),'Refusing to overwrite cache')
    cert=read(BOX/'data/balance_sensitivity/certificate.json');inventory=dict(cert['inputs'])
    for folder,manifest in [(HCACHE,'certificate.json'),(BOX/'data/balance_sensitivity','certificate.json'),
       (BOX/'output/horizon_2345','estimates.json')]:
        c=read(folder/manifest);require(c['status']=='PASS','Uncertified source')
        inventory.update(c.get('inputs',{}));inventory.update({str(folder/k):v for k,v in c['files'].items()})
        inventory[str(folder/manifest)]=sha(folder/manifest)
    for p in [BOX/'code/h3_analysis.py',APP/'code/lib/baseline_math.py',BOX/'output/pdf/NS_aggregation_summary_2000.pdf']:
        inventory[str(p)]=sha(p)
    hashes(inventory)
    c=pd.read_parquet(HCACHE/'county_state_h3.parquet');a=pd.read_parquet(HCACHE/'aggregate_state_h3.parquet')
    chars=pd.read_parquet(BOX/'data/balance_sensitivity/characteristics.parquet').set_index('ccg')
    require(len(c)==110088 and c.ccg.nunique()==3058 and len(a)==1836 and c.unit.nunique()==51,'Sample mismatch')
    aa=a.set_index(['unit','year']).sort_index()
    require(np.allclose(c.groupby(['unit','year'])[['E','E_b']].sum().sort_index(),aa[['E','E_b']],rtol=1e-12,atol=1e-6),'Endpoint conservation')
    for v in ['std_y','x','z']:
        agg=(c.weight*c[v]).groupby([c.unit,c.year]).sum().sort_index()
        require(np.allclose(agg,aa[v if v=='std_y' else v+'_agg'],rtol=1e-12,atol=1e-10),'Frozen aggregation mismatch')
    for i,key in enumerate(KEYS,1):
        char='C_income_winsorized' if key=='income_per_resident' else 'C_'+key
        c[f'C{i}']=c.ccg.map(chars[char]);require(np.isfinite(c[f'C{i}']).all(),'Missing characteristic')
        c[f'xc{i}']=c.x*c[f'C{i}'];c[f'zc{i}']=c.z*c[f'C{i}']
        for v,target in [(f'C{i}',f'A{i}'),(f'xc{i}',f'Q{i}'),(f'zc{i}',f'R{i}')]:
            g=(c.weight*c[v]).groupby([c.unit,c.year]).sum().rename(target)
            c=c.merge(g.reset_index(),on=['unit','year'],how='left',validate='many_to_one',sort=False)
            a=a.merge(g.reset_index(),on=['unit','year'],how='left',validate='one_to_one',sort=False)
    require(np.isfinite(c.select_dtypes('number')).all().all(),'Nonfinite cache')
    CACHE.mkdir(parents=True,exist_ok=True)
    c.to_parquet(CACHE/'county.parquet',index=False);a.to_parquet(CACHE/'state.parquet',index=False)
    hashes(inventory);write(CACHE/'certificate.json',dict(status='PASS',inputs=inventory,files={p.name:sha(p) for p in CACHE.glob('*.parquet')},builder_sha256=sha(__file__)))
    print('BUILD PASS: frozen support, outcomes and weights; matched controls constructed.')

def iv(d,endog,excluded,exog,fe):
    w=d.weight.to_numpy(float);f1=pd.factorize(d[fe])[0];f2=pd.factorize(d.year)[0];cl,labels=pd.factorize(d.unit,sort=True)
    names=['std_y']+endog+excluded+exog
    raw=d[names].to_numpy(float);scale=np.maximum(abs(raw).max(axis=0),1e-30)
    m=demean(raw/scale,f1,f2,w)
    for ff in [f1,f2]:
        den=np.bincount(ff,weights=w)
        for v in m.T:require(abs(np.bincount(ff,weights=w*v)/den).max()<1e-9,'Incomplete FE absorption')
    nend=len(endog);nex=len(excluded);ne=len(exog)
    y=m[:,0];x0=m[:,1:1+nend];z0=m[:,1+nend:1+nend+nex];e=m[:,-ne:] if ne else np.empty((len(d),0))
    x=np.column_stack([x0,e]);z=np.column_stack([z0,e])
    zx=z.T@(w[:,None]*x);zz=z.T@(w[:,None]*z)
    require(np.linalg.matrix_rank(zx)==len(x[0]) and np.linalg.matrix_rank(zz)==len(z[0]),'Rank failure; stop rather than drop controls')
    b=np.linalg.solve(zx,z.T@(w*y));pred=z@np.linalg.solve(zz,zx)
    require(np.allclose(b,np.linalg.lstsq(pred*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0],rtol=1e-7,atol=1e-9),'Projected-IV mismatch')
    G=len(labels);k=ssc_k(f1,f2,cl)+x.shape[1]-1;ssc=G/(G-1)*(len(d)-1)/(len(d)-k)
    residual=y-x@b;scores=np.zeros((G,x.shape[1]));np.add.at(scores,cl,z*(w*residual)[:,None])
    xs=np.r_[scale[1:1+nend],scale[-ne:] if ne else []];factor=scale[0]/xs
    psi=scores@np.linalg.inv(zx).T*np.sqrt(ssc)*factor
    result=[]
    for j,name in enumerate(endog+exog):
        beta=b[j]*factor[j];se=np.linalg.norm(psi[:,j]);F=np.nan
        if j<nend:
            pi=np.linalg.solve(zz,z.T@(w*x0[:,j]));v=x0[:,j]-z@pi
            ss=np.zeros((G,z.shape[1]));np.add.at(ss,cl,z*(w*v)[:,None]);bread=np.linalg.inv(zz)
            V=ssc*bread@(ss.T@ss)@bread.T
            F=float(pi[:nex]@np.linalg.solve(V[:nex,:nex],pi[:nex])/nex)
        result.append(dict(variable=name,beta=beta,se=se,p=2*t.sf(abs(beta/se),G-1),F=F))
    export=pd.DataFrame(m*scale,columns=['y']+endog+excluded+exog)
    export['weight']=w;export['unit']=cl.astype(np.int32)
    norm=zx/np.sqrt((w[:,None]*z**2).sum(axis=0))[:,None]/np.sqrt((w[:,None]*x**2).sum(axis=0))[None,:]
    return result,psi,export,dict(k=int(k),n=len(d),G=G,condition=float(np.linalg.cond(norm)))

def estimate():
    require(not (OUT/'estimates.json').exists(),'Refusing to overwrite estimates')
    cert=read(CACHE/'certificate.json');hashes(cert['inputs']);hashes({str(CACHE/k):v for k,v in cert['files'].items()})
    c=pd.read_parquet(CACHE/'county.parquet');a=pd.read_parquet(CACHE/'state.parquet')
    old=pd.read_csv(BOX/'output/horizon_2345/aggregation.csv').query("horizon==3 and rung=='state' and growth=='std'").iloc[0]
    rows=[];summary=[];jobs=[];inf=[]
    for controls in range(4):
        ids=range(1,controls+1);exo=[f'A{i}' for i in ids]
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
            if controls==0:
                for variable,prefix in ([('x','augmented'),('x_agg','exposure')] if model=='county' else [('x_agg','aggregate')]):
                    for key in ['beta','se','F','p']:require(np.isclose(fits[model].loc[variable,key],old[prefix+'_'+key],rtol=1e-7,atol=1e-12),'Original baseline not reproduced')
        rr=fits['county'].loc['x_agg'];res=fits['state'].loc['x_agg','beta']-fits['county'].loc['x','beta']-rr.beta
        summary.append(dict(controls=controls,delta=rr.beta,se=rr.se,p=rr.p,identity_residual=res,n_excluded=len(cz)))
        print('CONTROLS',controls,'delta/SE (x1e5)',rr.beta*1e5,rr.se*1e5,'p',rr.p,flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,data in [('coefficients.csv',rows),('delta.csv',summary),('influences.csv',inf),('design_audit.csv',jobs)]:pd.DataFrame(data).to_csv(OUT/name,index=False)
    logs=ROOT/'batch_logs/macro_paul_heterogeneity_levels';logs.mkdir(exist_ok=True)
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
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','estimate','verify']);a=p.parse_args();globals()[a.stage]()
