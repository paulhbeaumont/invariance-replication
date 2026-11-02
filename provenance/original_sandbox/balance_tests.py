"""Approved balance-only diagnostic. Never rebuild or overwrite growth estimates."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, f as fisher_f
from h3_analysis import BOX, ROOT, APP, sha, read, write, require, fit
from baseline_math import demean, ssc_k

CACHE = BOX/'data/balance_tests'
OUT = BOX/'output/balance_tests'
HCACHE = BOX/'data/horizon_2345'
HS = (2, 3, 4, 5)
RUNGS = ('czone', 'cea', 'ea', 'state')
KEYS = ['employment', 'population', 'jobs_per_resident', 'income_per_resident', 'manufacturing', 'concentration']
LABELS = ['Employment', 'Population', 'Employment per resident', 'Real income per resident', 'Manufacturing share', 'Industry concentration']


def verify(inputs):
    for name, digest in inputs.items():
        require(sha(name) == digest, 'Input changed: '+name)


def build():
    require(not (CACHE/'certificate.json').exists(), 'Refusing to overwrite approved diagnostic inputs')
    source = APP/'data/intermediate/baseline_roll5_gap5/source51_county_panel.parquet'
    cells = APP/'data/intermediate/disaggregation_h234/naics4_alloc/y1978.parquet'
    bsource = read(APP/'config/baseline_sources.json')
    require(sha(source) == bsource['inputs'][source.name], 'County source differs from baseline')
    snap = APP/'results/manifests/disaggregation_snapshot.json'
    manifest = read(snap)
    require(manifest['status'] == 'PASS' and sha(cells) == manifest['inputs']['naics4_alloc/y1978.parquet'], 'Cell snapshot differs')
    hcert = read(HCACHE/'certificate.json')
    require(hcert['status'] == 'PASS', 'Uncertified horizon inputs')
    inputs = dict(hcert['inputs'])
    for p in [source, cells, snap, HCACHE/'certificate.json', BOX/'code/h3_analysis.py',
              APP/'code/lib/baseline_math.py', BOX/'output/horizon_2345/aggregation.csv',
              BOX/'output/horizon_2345/estimates.json', BOX/'output/pdf/NS_aggregation_horizons_2345.pdf']:
        inputs[str(p)] = sha(p)
    for h in HS:
        for rung in RUNGS:
            for level in ['county', 'aggregate']:
                p = HCACHE/f'{level}_{rung}_h{h}.parquet'
                require(sha(p) == hcert['files'][p.name], 'Horizon cache changed')
                inputs[str(p)] = sha(p)
    verify(inputs)
    b = pd.read_parquet(source).query('year==1978').sort_values('ccg').set_index('ccg')
    require(len(b) == 3058 and b.index.is_unique, 'Initial county support differs')
    ids = pd.read_parquet(HCACHE/'county_cea_h3.parquet', columns=['ccg']).ccg.unique()
    require(np.array_equal(b.index, np.sort(ids)), 'Initial support mismatch')
    c = pd.read_parquet(cells)
    c = c.loc[c.ccg.isin(b.index)].copy()
    require(not c.duplicated(['ccg', 'code']).any() and c.emp.notna().all() and c.emp.ge(0).all(), 'Invalid industry cells')
    total = c.groupby('ccg').emp.sum().reindex(b.index)
    require(np.allclose(total, b.E, rtol=1e-12, atol=1e-6), '1978 employment conservation failed')
    require(np.isfinite(b[['E','P','Y']]).all().all() and b[['E','P','Y']].gt(0).all().all(), 'Missing/nonpositive initial characteristic')
    c['share'] = c.emp / c.ccg.map(b.E)
    manufacturing = c.naics.str[:2].isin(['31','32','33'])
    residual = ~c.naics.str.fullmatch(r'\d{4}')
    chars = pd.DataFrame(index=b.index)
    chars['employment'] = b.E
    chars['population'] = b.P
    chars['jobs_per_resident'] = b.E/b.P
    chars['income_per_resident'] = b.Y/b.P
    chars['manufacturing'] = c.loc[manufacturing].groupby('ccg').emp.sum().reindex(b.index, fill_value=0)/b.E
    chars['concentration'] = (c.share**2).groupby(c.ccg).sum().reindex(b.index)
    require(chars.manufacturing.between(0,1).all() and chars.concentration.between(0,1).all(), 'Invalid share/concentration')
    cutoffs = np.quantile(b.E, [.2,.4,.6,.8])
    require(np.all(np.diff(cutoffs)>0), 'Duplicate quintile thresholds')
    chars['quintile'] = np.searchsorted(cutoffs, b.E, side='left')+1
    definitions = []
    for key, label in zip(KEYS, LABELS):
        mean = float(chars[key].mean()); sd = float(chars[key].std(ddof=0))
        require(sd>0 and np.isfinite(chars[key]).all(), 'Invalid characteristic '+key)
        chars['C_'+key] = (chars[key]-mean)/sd
        definitions.append(dict(key=key,label=label,mean=mean,sd=sd,min=float(chars[key].min()),max=float(chars[key].max())))
    CACHE.mkdir(parents=True, exist_ok=True)
    chars.reset_index().to_parquet(CACHE/'characteristics.parquet', index=False)
    chars.reset_index().to_csv(CACHE/'characteristics.csv', index=False)
    write(CACHE/'certificate.json', dict(status='PASS', inputs=inputs, builder_sha256=sha(__file__),
        files={p.name:sha(p) for p in CACHE.glob('characteristics.*')}, definitions=definitions,
        n_counties=len(chars), initial_year=1978, quintile_cutoffs=cutoffs.tolist(),
        quintile_counts={int(k):int(v) for k,v in chars.quintile.value_counts().sort_index().items()},
        employment_conservation_max=float(abs(total-b.E).max()),
        residual_industry_codes=sorted(c.loc[residual,'naics'].unique().tolist()),
        residual_employment_share=float(c.loc[residual,'emp'].sum()/c.emp.sum())))
    verify(inputs)
    print('BUILD PASS: all 3058 counties; exact initial employment conservation.', flush=True)


def fit_many(d, ys):
    """Batch version of the unchanged sandbox IV/CRV1 operator, same two IVs."""
    w = d.weight.to_numpy(float)
    f1 = pd.factorize(d.ccg)[0]; f2 = pd.factorize(d.year)[0]
    cl, labels = pd.factorize(d.unit, sort=True)
    columns = ys+['x','x_agg','z','z_agg']
    raw = d[columns].to_numpy(float)
    require(np.isfinite(raw).all() and np.isfinite(w).all() and (w>0).all(), 'Invalid fit inputs')
    scales = np.maximum(abs(raw).max(axis=0), 1e-30)
    M = demean(raw/scales, f1, f2, w)
    for ff in [f1,f2]:
        sw = np.bincount(ff,weights=w)
        for k in range(M.shape[1]):
            require(abs(np.bincount(ff,weights=w*M[:,k])/sw).max()<1e-9, 'FE normal equation failure')
    k = len(ys); Y=M[:,:k]; X=M[:,k:k+2]; Z=M[:,k+2:]
    ZX=Z.T@(w[:,None]*X); ZZ=Z.T@(w[:,None]*Z)
    require(np.linalg.matrix_rank(ZX)==2 and np.linalg.matrix_rank(ZZ)==2, 'IV rank failure')
    bread=np.linalg.inv(ZX)
    B=bread@(Z.T@(w[:,None]*Y))
    pred=Z@np.linalg.solve(ZZ,ZX)
    cross=np.linalg.lstsq(pred*np.sqrt(w)[:,None],Y*np.sqrt(w)[:,None],rcond=None)[0]
    require(np.allclose(B,cross,rtol=1e-8,atol=1e-10), 'Projected-IV check failed')
    G=len(labels); n=len(d); dofk=ssc_k(f1,f2,cl)+1
    ssc=G/(G-1)*(n-1)/(n-dofk)
    factors=scales[:k][None,:]/scales[k:k+2,None]
    U=Y-X@B; psi=np.empty((G,2,k))
    for q in range(k):
        scores=np.zeros((G,2)); np.add.at(scores,cl,Z*(w*U[:,q])[:,None])
        psi[:,:,q]=(scores@bread.T)*np.sqrt(ssc)*factors[:,q]
    betas=B*factors; ses=np.linalg.norm(psi,axis=0)
    first=[]
    for j in range(2):
        pi=np.linalg.solve(ZZ,Z.T@(w*X[:,j])); v=X[:,j]-Z@pi
        scores=np.zeros((G,2)); np.add.at(scores,cl,Z*(w*v)[:,None])
        zz_inv=np.linalg.inv(ZZ); V=ssc*zz_inv@(scores.T@scores)@zz_inv.T
        first.append(float(pi@np.linalg.solve(V,pi)/2))
    normalized=ZX/np.sqrt((w[:,None]*Z**2).sum(axis=0))[:,None]/np.sqrt((w[:,None]*X**2).sum(axis=0))[None,:]
    return betas,ses,psi,np.array(first),dict(n=n,G=G,k=dofk,condition=float(np.linalg.cond(normalized)),labels=labels)


def joint(b, psi, df):
    V=psi.T@psi; rank=np.linalg.matrix_rank(V)
    require(rank==len(b), 'Joint test covariance is rank deficient; do not silently substitute a test')
    W=float(b@np.linalg.solve(V,b)); F=W/len(b)
    return dict(wald=W,F=F,df_num=len(b),df_denom=df,p=float(fisher_f.sf(F,len(b),df)))


def holm(p):
    p=np.asarray(p); order=np.argsort(p)
    answer=np.empty(len(p)); answer[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return answer


def estimate():
    require(not (OUT/'estimates.json').exists(), 'Refusing to overwrite completed diagnostics')
    cert=read(CACHE/'certificate.json')
    require(cert['status']=='PASS' and cert['builder_sha256']==sha(__file__), 'Diagnostic builder differs')
    verify(cert['inputs'])
    for name,digest in cert['files'].items(): require(sha(CACHE/name)==digest, 'Characteristic file changed')
    chars=pd.read_parquet(CACHE/'characteristics.parquet').set_index('ccg')
    ref=pd.read_csv(BOX/'output/horizon_2345/aggregation.csv')
    rows=[]; joints=[]; stages=[]; sourcechecks=[]; influences=[]
    for h in HS:
        for rung in RUNGS:
            d=pd.read_parquet(HCACHE/f'county_{rung}_h{h}.parquet')
            require(len(d)==3058*(2017-(1978+h)) and d.ccg.nunique()==3058, 'Sample differs')
            require(np.allclose(d.groupby(['unit','year']).weight.sum(),1,rtol=1e-12,atol=1e-12), 'Weights do not sum to one')
            require(np.array_equal(np.sort(d.ccg.unique()),chars.index), 'Characteristics support differs')
            d['quintile']=d.ccg.map(chars.quintile)
            for key in KEYS: d['balance_'+key]=d.x*d.ccg.map(chars['C_'+key])
            for panel in ['A','B']:
                keys=KEYS if panel=='A' else KEYS[2:]
                f=d.copy()
                if panel=='B':
                    # Since quintile is fixed within county, these effects span
                    # county + year + initial-quintile-by-year interactions.
                    f['year']=pd.factorize(list(zip(f.quintile,f.year)))[0]
                ys=['balance_'+key for key in keys]
                b,se,psi,fs,meta=fit_many(f,ys)
                critical=student_t.ppf(.975,meta['G']-1)
                p=2*student_t.sf(abs(b[1]/se[1]),meta['G']-1)
                adj=holm(p)
                for j,key in enumerate(keys):
                    rows.append(dict(horizon=h,rung=rung,panel=panel,characteristic=key,
                        pi=b[1,j],se=se[1,j],ci_low=b[1,j]-critical*se[1,j],ci_high=b[1,j]+critical*se[1,j],
                        p=p[j],holm_p=adj[j],rho=b[0,j],rho_se=se[0,j],n_obs=meta['n'],n_clusters=meta['G']))
                    influences.extend(dict(horizon=h,rung=rung,panel=panel,characteristic=key,cluster=int(label),psi=float(value))
                                      for label,value in zip(meta['labels'],psi[:,1,j]))
                for selection in ([('all_six',list(range(6))),('four_other',list(range(2,6)))] if panel=='A' else [('four_other',list(range(4)))]):
                    name,idx=selection
                    joints.append(dict(horizon=h,rung=rung,panel=panel,test=name,**joint(b[1,idx],psi[:,1,idx],meta['G']-1)))
                stages.append(dict(horizon=h,rung=rung,panel=panel,F_own=fs[0],F_exposure=fs[1],
                    n_obs=meta['n'],n_clusters=meta['G'],ssc_k=meta['k'],normalized_condition=meta['condition']))
                # Cross-check against pre-existing, unchanged single-outcome fit.
                existing,oldpsi=fit(f,ys[0],['x','x_agg'],['z','z_agg'])
                for j,key in enumerate(['x','x_agg']):
                    require(np.isclose(existing[key]['beta'],b[j,0],rtol=1e-7,atol=1e-10),'Existing estimator beta differs')
                    require(np.isclose(existing[key]['se'],se[j,0],rtol=1e-7,atol=1e-10),'Existing estimator SE differs')
                    require(np.isclose(existing[key]['F'],fs[j],rtol=1e-7,atol=1e-9),'Existing estimator F differs')
                    require(np.allclose(oldpsi[key],psi[:,j,0],rtol=1e-7,atol=1e-10),'Existing cluster scores differ')
                if panel=='A':
                    rr=ref.loc[(ref.horizon==h)&(ref.rung==rung)]
                    require(len(rr)==2,'Missing growth references')
                    require(np.allclose(rr.augmented_F,fs[0],rtol=1e-7) and np.allclose(rr.exposure_F,fs[1],rtol=1e-7),'Original first stage changed')
                sourcechecks.append(dict(horizon=h,rung=rung,panel=panel,existing_fit='PASS',old_first_stage='PASS' if panel=='A' else 'new_size_time_controls'))
                print('FIT',h,rung,panel,'F:',np.round(fs,3).tolist(),'pi:',np.round(b[1],3).tolist(),flush=True)
    require(len(rows)==160 and len(joints)==48 and len(stages)==32, 'Incomplete diagnostic grid')
    verify(cert['inputs'])
    OUT.mkdir(parents=True,exist_ok=True)
    for name,data in [('balance.csv',rows),('joint_tests.csv',joints),('first_stages.csv',stages),('cluster_influences.csv',influences)]:
        pd.DataFrame(data).to_csv(OUT/name,index=False)
    write(OUT/'estimates.json',dict(status='PASS',estimator_sha256=sha(__file__),certificate_sha256=sha(CACHE/'certificate.json'),
        files={p.name:sha(p) for p in OUT.glob('*.csv')},n_diagnostics=160,n_joint_tests=48,
        original_inputs_unchanged=True,source_fit_checks=sourcechecks,
        outcome='saved capped county spending times standardized 1978 characteristic',
        first_stage='Joint two-excluded-instrument clustered Wald F; not conditional weak-IV F',
        inference='CRV1 and t(G-1); joint Wald F(q,G-1); not weak-IV robust'))


def test():
    """Numerical verification against explicit dummy-variable IV, no empirical fits."""
    rng=np.random.default_rng(915)
    ncounty=40; nyear=6
    d=pd.DataFrame({'ccg':np.repeat(np.arange(ncounty),nyear),'year':np.tile(np.arange(nyear),ncounty)})
    d['unit']=d.ccg//2; d['weight']=rng.uniform(.4,1.5,len(d))
    d['quintile']=d.ccg%5
    for panel in ['A','B']:
        f=d.copy()
        if panel=='B': f['year']=pd.factorize(list(zip(f.quintile,f.year)))[0]
        f['z']=rng.normal(size=len(f)); f['z_agg']=rng.normal(size=len(f))
        f['x']=2*f.z+.3*f.z_agg+rng.normal(size=len(f))
        f['x_agg']=.2*f.z+1.4*f.z_agg+rng.normal(size=len(f))
        f['y1']=.8*f.x-.5*f.x_agg+.03*f.ccg+.01*f.year+rng.normal(size=len(f))
        f['y2']=f.x # affine-standardization sanity check: exposure coefficient zero
        b,se,psi,fs,meta=fit_many(f,['y1','y2'])
        D=np.column_stack([pd.get_dummies(f.ccg).to_numpy(),pd.get_dummies(f.year).to_numpy()])
        A=f[['y1','y2','x','x_agg','z','z_agg']].to_numpy(); w=f.weight.to_numpy()
        R=A-D@np.linalg.lstsq(D*np.sqrt(w)[:,None],A*np.sqrt(w)[:,None],rcond=None)[0]
        Y=R[:,:2]; X=R[:,2:4]; Z=R[:,4:]; bread=np.linalg.inv(Z.T@(w[:,None]*X))
        explicit=bread@(Z.T@(w[:,None]*Y))
        require(np.allclose(b,explicit,rtol=1e-9,atol=1e-9),'Explicit dummy IV differs')
        require(abs(b[1,1])<1e-10 and abs(b[0,1]-1)<1e-10,'Constant characteristic sanity check failed')
        for j in range(2):
            cl,labels=pd.factorize(f.unit,sort=True)
            scores=np.zeros((len(labels),2)); np.add.at(scores,cl,Z*(w*(Y[:,j]-X@explicit[:,j]))[:,None])
            ssc=len(labels)/(len(labels)-1)*(len(f)-1)/(len(f)-meta['k'])
            refpsi=scores@bread.T*np.sqrt(ssc)
            require(np.allclose(refpsi,psi[:,:,j],atol=1e-9,rtol=1e-8),'Explicit-dummy sandwich differs')
        print('TEST explicit-dummy coefficient and sandwich PASS:',panel,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['build','estimate','test']); args=parser.parse_args()
    globals()[args.stage]()
