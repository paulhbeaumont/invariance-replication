"""Approved isolated follow-up; no growth regressions or original outputs changed."""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, spearmanr
from balance_tests import (BOX, ROOT, APP, CACHE as OLD_CACHE, OUT as OLD_OUT,
    HCACHE, HS, RUNGS, KEYS, fit_many, joint, holm, verify as check_hashes)
from h3_analysis import sha, read, write, require
from baseline_math import wins, demean, ssc_k

CACHE = BOX/'data/balance_sensitivity'
OUT = BOX/'output/balance_sensitivity'


def build():
    require(not (CACHE/'certificate.json').exists(), 'Refusing to overwrite sensitivity cache')
    old = read(OLD_CACHE/'certificate.json'); results = read(OLD_OUT/'estimates.json')
    require(old['status']==results['status']=='PASS', 'Uncertified originals')
    inputs = dict(old['inputs'])
    for folder, cert in [(OLD_CACHE,old),(OLD_OUT,results)]:
        for name,digest in cert['files'].items():
            require(sha(folder/name)==digest,'Original file differs: '+name)
            inputs[str(folder/name)] = digest
    for p in [OLD_CACHE/'certificate.json',OLD_OUT/'estimates.json',
              BOX/'code/balance_tests.py',BOX/'output/pdf/NS_aggregation_covariate_balance.pdf']:
        inputs[str(p)] = sha(p)
    for h in HS:
        p=HCACHE/f'county_h{h}.parquet'; inputs[str(p)]=sha(p)
        require(inputs[str(p)]==read(HCACHE/'certificate.json')['files'][p.name], 'County frame differs')
    check_hashes(inputs)
    c=pd.read_parquet(OLD_CACHE/'characteristics.parquet')
    raw=c.income_per_resident.to_numpy(); lo,hi=np.percentile(raw,[1,99])
    capped=wins(raw,np.zeros(len(c),dtype=int))
    require(np.array_equal(capped,np.clip(raw,lo,hi)), 'Source winsorization rule mismatch')
    c['income_winsorized']=capped
    c['C_income_winsorized']=(capped-capped.mean())/capped.std(ddof=0)
    CACHE.mkdir(parents=True,exist_ok=True)
    c.to_parquet(CACHE/'characteristics.parquet',index=False)
    details=dict(lo=float(lo),hi=float(hi),n_low=int((raw<lo).sum()),n_high=int((raw>hi).sum()),
                 original_sd=float(raw.std()),new_sd=float(capped.std()),original_mean=float(raw.mean()),new_mean=float(capped.mean()))
    write(CACHE/'certificate.json',dict(status='PASS',inputs=inputs,files={'characteristics.parquet':sha(CACHE/'characteristics.parquet')},income=details))
    print('BUILD PASS',details,flush=True)


def residualize(d):
    w=d.weight.to_numpy(); f1=pd.factorize(d.ccg)[0]; f2=pd.factorize(d.year)[0]
    cl,labels=pd.factorize(d.unit,sort=True)
    cols=['income_original','x','x_agg','z','z_agg']
    raw=d[cols].to_numpy(); scale=np.maximum(abs(raw).max(axis=0),1e-30)
    m=demean(raw/scale,f1,f2,w)
    for ff in [f1,f2]:
        sw=np.bincount(ff,weights=w)
        require(all(abs(np.bincount(ff,weights=w*m[:,j])/sw).max()<1e-9 for j in range(5)), 'FE residuals not orthogonal')
    return m,w,cl,ssc_k(f1,f2,cl)+1


def conditional(m,w,cl,k):
    X=m[:,1:3]; Z=m[:,3:]; n=len(w); G=len(np.unique(cl))
    Q=Z.T@(w[:,None]*Z); inv=np.linalg.inv(Q)
    P=np.linalg.solve(Q,Z.T@(w[:,None]*X)); fitted=Z@P
    ssc=G/(G-1)*(n-1)/(n-k)
    ans=[]
    for j in range(2):
        other=1-j; f=fitted[:,other]
        a=float(f@(w*X[:,j])/(f@(w*f)))
        # Corrected residual uses actual other endogenous variable (not its fit).
        y=X[:,j]-a*X[:,other]
        b=np.linalg.solve(Q,Z.T@(w*y)); u=y-Z@b
        scores=np.zeros((G,2)); np.add.at(scores,cl,Z*(w*u)[:,None])
        V=ssc*inv@(scores.T@scores)@inv.T
        F=float(b@np.linalg.solve(V,b)) # excluded IVs - other endog = 1
        # Algebraically independent reduced-form residual calculation.
        require(np.allclose(b,P[:,j]-a*P[:,other],rtol=1e-9,atol=1e-12),'Conditional RF coefficients mismatch')
        require(np.allclose(u,(X-fitted)[:,j]-a*(X-fitted)[:,other],rtol=1e-9,atol=1e-12),'SW variance residual mismatch')
        ans.append(F)
    return ans


def descriptive(chars):
    rows=[]; counties=chars[['manufacturing']].copy(); first=None
    for h in HS:
        d=pd.read_parquet(HCACHE/f'county_h{h}.parquet')
        a=d.loc[d.year==1978+h].set_index('ccg').exposure.reindex(chars.index)
        if first is None: first=a
        else: require(np.array_equal(a,first),'Initial exposure differs across horizons')
        mean=d.groupby('ccg').exposure.mean().reindex(chars.index)
        counties[f'mean_exposure_h{h}']=mean
        for label,v in [('initial_1969_1973',a),('mean_rolling_exposure',mean)]:
            rows.append(dict(horizon=h,exposure=label,pearson=float(np.corrcoef(v,chars.manufacturing)[0,1]),
                             spearman=float(spearmanr(v,chars.manufacturing).correlation),n=len(v)))
    counties['initial_exposure']=first
    cut=np.quantile(first,[.2,.4,.6,.8]); require(np.all(np.diff(cut)>0),'Exposure quintile ties')
    counties['exposure_quintile']=np.searchsorted(cut,first,side='left')+1
    bins=counties.groupby('exposure_quintile').agg(n=('manufacturing','size'),manufacturing_mean=('manufacturing','mean'),
        exposure_mean=('initial_exposure','mean'),exposure_min=('initial_exposure','min'),exposure_max=('initial_exposure','max'))
    counties.reset_index().to_csv(OUT/'manufacturing_counties.csv',index=False)
    bins.reset_index().to_csv(OUT/'manufacturing_bins.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'manufacturing_correlations.csv',index=False)


def estimate():
    require(not (OUT/'estimates.json').exists(),'Refusing to overwrite sensitivity estimates')
    cert=read(CACHE/'certificate.json'); check_hashes(cert['inputs'])
    check_hashes({str(CACHE/k):v for k,v in cert['files'].items()})
    chars=pd.read_parquet(CACHE/'characteristics.parquet').set_index('ccg')
    original=pd.read_csv(OLD_OUT/'balance.csv'); fsold=pd.read_csv(OLD_OUT/'first_stages.csv')
    OUT.mkdir(parents=True,exist_ok=True); (CACHE/'stata').mkdir(exist_ok=True)
    rows=[]; joints=[]; stages=[]; swrows=[]; incomes=[]; influence=[]; jobs=[]
    for h in HS:
        for rung in RUNGS:
            d=pd.read_parquet(HCACHE/f'county_{rung}_h{h}.parquet')
            require(len(d)==3058*(2017-1978-h) and d.ccg.nunique()==3058,'Support mismatch')
            require(np.allclose(d.groupby(['unit','year']).weight.sum(),1,atol=1e-12),'Weight mismatch')
            d['quintile']=d.ccg.map(chars.quintile)
            for key in KEYS: d['b_'+key]=d.x*d.ccg.map(chars['C_'+key])
            d['income_original']=d.b_income_per_resident
            d['b_income_per_resident']=d.x*d.ccg.map(chars.C_income_winsorized)
            for panel in ['A','B']:
                keys=KEYS if panel=='A' else KEYS[2:]; f=d.copy()
                if panel=='B': f['year']=pd.factorize(list(zip(f.quintile,f.year)))[0]
                b,se,psi,fs,meta=fit_many(f,['b_'+key for key in keys]+['income_original'])
                old=original.loc[(original.horizon==h)&(original.rung==rung)&(original.panel==panel)].set_index('characteristic')
                for j,key in enumerate(keys):
                    ix=len(keys) if key=='income_per_resident' else j
                    require(np.isclose(b[1,ix],old.loc[key,'pi'],atol=1e-9,rtol=1e-7),'Original coefficient not reproduced')
                    require(np.isclose(se[1,ix],old.loc[key,'se'],atol=1e-9,rtol=1e-7),'Original SE not reproduced')
                oldfs=fsold.loc[(fsold.horizon==h)&(fsold.rung==rung)&(fsold.panel==panel)].iloc[0]
                require(np.allclose(fs,[oldfs.F_own,oldfs.F_exposure],rtol=1e-8),'First stage changed')
                p=2*student_t.sf(abs(b[1,:len(keys)]/se[1,:len(keys)]),meta['G']-1); adj=holm(p)
                crit=student_t.ppf(.975,meta['G']-1)
                for j,key in enumerate(keys):
                    row=dict(horizon=h,rung=rung,panel=panel,characteristic=key,pi=b[1,j],se=se[1,j],
                             ci_low=b[1,j]-crit*se[1,j],ci_high=b[1,j]+crit*se[1,j],p=p[j],holm_p=adj[j],n_obs=meta['n'],n_clusters=meta['G'])
                    rows.append(row)
                    influence.extend(dict(horizon=h,rung=rung,panel=panel,characteristic=key,cluster=int(label),psi=float(value)) for label,value in zip(meta['labels'],psi[:,1,j]))
                    if key=='income_per_resident':
                        delta=b[1,j]-b[1,-1]; delta_se=float(np.linalg.norm(psi[:,1,j]-psi[:,1,-1]))
                        incomes.append(dict(row,original_pi=b[1,-1],original_se=se[1,-1],original_p=old.loc[key,'p'],
                            difference=delta,difference_se=delta_se,difference_p=float(2*student_t.sf(abs(delta/delta_se),meta['G']-1)),
                            original_raw_pi=b[1,-1]/cert['income']['original_sd'],winsorized_raw_pi=b[1,j]/cert['income']['new_sd']))
                selections=[('all_six',list(range(6))),('four_other',list(range(2,6)))] if panel=='A' else [('four_other',list(range(4)))]
                for name,idx in selections:
                    joints.append(dict(horizon=h,rung=rung,panel=panel,test=name,**joint(b[1,idx],psi[:,1,idx],meta['G']-1)))
                stages.append(dict(horizon=h,rung=rung,panel=panel,F_own=fs[0],F_exposure=fs[1]))
                if rung in ['cea','state']:
                    m,w,cl,k=residualize(f); sw=conditional(m,w,cl,k)
                    swrows.append(dict(horizon=h,rung=rung,panel=panel,F_own=fs[0],F_exposure=fs[1],SW_own=sw[0],SW_exposure=sw[1],n=len(f),G=meta['G'],k=k))
                    export=pd.DataFrame(m,columns=['y','x','x_agg','z','z_agg']); export['weight']=w; export['unit']=cl.astype(np.int32)
                    name=f'{rung}_{h}_{panel}'
                    export.to_stata(CACHE/f'stata/{name}.dta',write_index=False,version=118)
                    jobs.append(dict(name=name,k=k,h=h,rung=rung,panel=panel))
                print('CHECK',h,rung,panel,'income',round(b[1,keys.index('income_per_resident')],4),'originals/F reproduced',flush=True)
    require(len(rows)==160 and len(joints)==48 and len(swrows)==16,'Incomplete grid')
    for name,data in [('balance.csv',rows),('joint_tests.csv',joints),('first_stages.csv',stages),('conditional_first_stages.csv',swrows),('income_comparison.csv',incomes),('cluster_influences.csv',influence)]:
        pd.DataFrame(data).to_csv(OUT/name,index=False)
    descriptive(chars)
    # Independent installed Stata ivreg2 comparison; paths explicit, no production writes.
    log=(ROOT/'batch_logs/macro_paul_balance/sensitivity_stata.log').as_posix()
    script=['clear all','set more off',f'log using "{log}", text replace',
            f'file open result using "{(OUT/"stata_conditional.csv").as_posix()}", write replace',
            'file write result "horizon,rung,panel,F_own,F_exposure,SW_own,SW_exposure" _n']
    for job in jobs:
        script += [f'use "{(CACHE/("stata/"+job["name"]+".dta")).as_posix()}", clear',
          f'ivreg2 y (x x_agg = z z_agg) [aw=weight], nocons cluster(unit) small sdofminus({job["k"]-2}) first',
          'matrix fs = e(first)',f'file write result "{job["h"]},{job["rung"]},{job["panel"]}," %21.15g (fs[4,1]) "," %21.15g (fs[4,2]) "," %21.15g (fs[8,1]) "," %21.15g (fs[8,2]) _n']
    script+=['file close result','log close','exit, clear']
    (CACHE/'verify_stata.do').write_text('\n'.join(script)+'\n')
    check_hashes(cert['inputs'])
    write(OUT/'estimates.json',dict(status='AWAITING_STATA_CHECK',inputs_unchanged=True,original_160_diagnostics_reproduced=True,
         original_32_first_stages_reproduced=True,estimator_sha256=sha(__file__),certificate_sha256=sha(CACHE/'certificate.json'),
         files={p.name:sha(p) for p in OUT.glob('*.csv')}))


def verify():
    r=read(OUT/'estimates.json'); check_hashes({str(OUT/k):v for k,v in r['files'].items()})
    a=pd.read_csv(OUT/'conditional_first_stages.csv').set_index(['horizon','rung','panel'])
    b=pd.read_csv(OUT/'stata_conditional.csv').set_index(['horizon','rung','panel']).reindex(a.index)
    cols=['F_own','F_exposure','SW_own','SW_exposure']
    require(len(b)==16 and np.isfinite(b[cols]).all().all(),'Incomplete Stata results')
    require(np.allclose(a[cols],b[cols],rtol=1e-7,atol=1e-8),'Stata first-stage cross-check failed')
    check_hashes(read(CACHE/'certificate.json')['inputs'])
    r.update(status='PASS',stata_ivreg2_checks=16,stata_max_absolute_difference=float(abs(a[cols]-b[cols]).max().max()))
    r['files']['stata_conditional.csv']=sha(OUT/'stata_conditional.csv'); write(OUT/'estimates.json',r)
    print('VERIFIED: 16 installed ivreg2 comparisons; all original hashes unchanged.',r['stata_max_absolute_difference'])


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['build','estimate','verify']); a=p.parse_args(); globals()[a.stage]()
