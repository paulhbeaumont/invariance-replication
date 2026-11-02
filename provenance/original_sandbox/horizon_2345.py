"""Paul-approved horizon-only aggregation comparison. No production writes."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from h3_analysis import BOX, APP, DATA, locked, certified, fit, sha, require, read, write
from disaggregation_math import county_frame

CACHE = BOX/'data/horizon_2345'
OUT = BOX/'output/horizon_2345'
SRC = BOX/'source/horizon_2345'
OLD = BOX/'data/h3_analysis'
HS = (2,3,4,5)
RUNGS = ('czone','cea','ea','state')
LABELS = dict(czone='Commuting zones', cea='Component economic areas (CEA)', ea='Economic areas (EA)', state='States')


def build():
    require(not (CACHE/'certificate.json').exists(), 'Refusing to replace certified inputs')
    CACHE.mkdir(parents=True, exist_ok=True)
    spec, sources, lock = locked()
    dlock = read(APP/'config/disaggregation_lock.json')
    require(sha(APP/'code/lib/disaggregation_math.py') == dlock['math_sha256'], 'County builder changed')
    old = read(OLD/'certificate.json')
    require(old['status']=='PASS', 'Old sandbox cache not certified')
    inputs={}
    for p in [APP/'code/lib/disaggregation_math.py', APP/'code/lib/baseline_math.py',
              BOX/'code/h3_analysis.py', APP/'config/baseline.json', APP/'config/baseline_sources.json',
              OLD/'certificate.json', BOX/'output/h3_analysis/estimates.json', BOX/'output/h3_analysis/aggregation.csv']:
        inputs[str(p)] = sha(p)
    prior=read(BOX/'output/h3_analysis/estimates.json')
    require(sha(BOX/'code/h3_analysis.py')==prior['estimator_sha256'],'Original h3 estimator changed')
    maps={}
    for rung in RUNGS:
        p=OLD/('county_'+rung+'.parquet')
        require(sha(p)==old['files'][p.name], 'Old geography input changed')
        inputs[str(p)]=sha(p)
        m=pd.read_parquet(p, columns=['ccg','unit']).drop_duplicates()
        require(len(m)==3058 and not m.ccg.duplicated().any(), 'County crosses geographic units')
        maps[rung]=m.set_index('ccg').unit
    frames={h:certified(DATA/'analysis/disaggregation_h234', 'county_h%d.parquet'%h, inputs) for h in (2,3,4)}
    test=county_frame(dict(spec,horizon=3,first_year=1981,n_observations=110088), sources)
    require(np.array_equal(test[['ccg','year']],frames[3][['ccg','year']]),'Source county keys differ')
    # The frozen parquet stores identifiers in int32; this pandas version uses
    # int64. Compare values, not storage width, without relaxing numeric tolerance.
    pd.testing.assert_frame_equal(test, frames[3][test.columns], check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    print('Source county_frame reproduces frozen h=3 inputs.',flush=True)
    frames[5]=county_frame(dict(spec,horizon=5,first_year=1983,n_observations=3058*34), sources)
    frames[5]['log_y']=np.log(frames[5].E)-np.log(frames[5].E_b)
    snapshot=DATA.parent/sources['snapshot_directory']
    for name,digest in sources['inputs'].items():
        require(sha(snapshot/name)==digest,'Frozen source changed')
        inputs[str(snapshot/name)]=digest
    checks=[]
    for h,f in frames.items():
        require(f.year.min()==1978+h and f.year.max()==2016,'Calendar rule changed')
        require(len(f)==3058*(2017-(1978+h)) and f.groupby('year').ccg.nunique().eq(3058).all(),'Support changed')
        require((f[['E','E_b','P_b']]>0).all().all(),'Nonpositive endpoint')
        require(np.allclose(f.log_y,np.log(f.E/f.E_b),atol=1e-12,rtol=1e-12),'Log growth changed')
        f.to_parquet(CACHE/('county_h%d.parquet'%h),index=False)
        for rung in RUNGS:
            c=f[['ccg','year','E','E_b','x','z','log_y']].copy()
            c['unit']=c.ccg.map(maps[rung])
            require(c.unit.notna().all(),'Unmapped county')
            c['weight']=c.E_b/c.groupby(['unit','year']).E_b.transform('sum')
            c['std_y']=(c.E-c.E_b)/c.E_b
            for v in ['x','z','std_y']: c['w_'+v]=c.weight*c[v]
            a=c.groupby(['unit','year'],as_index=False).agg(E=('E','sum'), E_b=('E_b','sum'),
                x_agg=('w_x','sum'),z_agg=('w_z','sum'),reconstructed=('w_std_y','sum'),weight_sum=('weight','sum'))
            a['std_y']=(a.E-a.E_b)/a.E_b
            a['log_y']=np.log(a.E/a.E_b); a['weight']=1.
            error=float(abs(a.std_y-a.reconstructed).max())
            require(error<1e-12 and np.allclose(a.weight_sum,1,atol=1e-12,rtol=0),'Reconstruction failed')
            require(np.allclose(a.groupby('year')[['E','E_b']].sum(),f.groupby('year')[['E','E_b']].sum(),rtol=1e-12,atol=1e-6),'Totals differ')
            c=c.drop(columns=['w_x','w_z','w_std_y']).merge(a[['unit','year','x_agg','z_agg']],on=['unit','year'],validate='many_to_one')
            require(len(c)==len(f),'Dropped county-years')
            for label,frame in [('county',c),('aggregate',a)]:
                frame.to_parquet(CACHE/('%s_%s_h%d.parquet'%(label,rung,h)),index=False)
            checks.append(dict(h=h,rung=rung,n_county_years=len(c),n_units=int(a.unit.nunique()),reconstruction_error=error))
        print('Built h=%d, %d county-years, %d-2016'%(h,len(f),f.year.min()),flush=True)
    write(CACHE/'certificate.json',dict(status='PASS',inputs=inputs,files={p.name:sha(p) for p in CACHE.glob('*.parquet')},
        builder_sha256=sha(__file__),baseline_lock=lock,checks=checks,h3_source_reproduction='PASS',horizons=list(HS)))


def estimate():
    cert=read(CACHE/'certificate.json')
    require(cert['status']=='PASS' and cert['builder_sha256']==sha(__file__),'Build not certified/current')
    for p,digest in cert['inputs'].items(): require(sha(p)==digest,'Original input changed: '+p)
    for name,digest in cert['files'].items(): require(sha(CACHE/name)==digest,'Sandbox input changed')
    OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'estimates.json').exists(),'Refusing to overwrite completed estimates')
    rows=[]; contrasts=[]
    reference=pd.read_csv(BOX/'output/h3_analysis/aggregation.csv')
    for h in HS:
        for rung in RUNGS:
            c=pd.read_parquet(CACHE/('county_%s_h%d.parquet'%(rung,h)))
            a=pd.read_parquet(CACHE/('aggregate_%s_h%d.parquet'%(rung,h)))
            influences={}; pairs={}
            for growth in ['std','log']:
                y=growth+'_y'
                nb,pn=fit(c,y,['x'],['z'])
                ab,pa=fit(a,y,['x_agg'],['z_agg'],fe='unit')
                ub,pu=fit(c,y,['x','x_agg'],['z','z_agg'])
                require(pn.index.equals(pa.index) and pn.index.equals(pu.index),'Cluster alignment failed')
                gap=ab['x_agg']['beta']-nb['x']['beta']; gap_se=float(np.linalg.norm(pa.x_agg-pn.x))
                r=dict(horizon=h,rung=rung,growth=growth,n_counties=3058,n_county_years=len(c),
                    n_units=int(a.unit.nunique()),n_aggregate_years=len(a),first_year=int(c.year.min()),last_year=2016,
                    gap=gap,gap_se=gap_se,gap_p=float(2*student_t.sf(abs(gap/gap_se),a.unit.nunique()-1)),
                    identity_residual=ab['x_agg']['beta']-ub['x']['beta']-ub['x_agg']['beta'])
                for label,vals in [('county',nb['x']),('aggregate',ab['x_agg']),('augmented',ub['x']),('exposure',ub['x_agg'])]:
                    r.update({label+'_'+key:value for key,value in vals.items()})
                if h==3:
                    old=reference.query('rung==@rung and growth==@growth').iloc[0]
                    for key in reference.columns:
                        if key not in ['rung','growth']:
                            require(np.isclose(r[key],old[key],rtol=1e-8,atol=1e-12),'h3 reference differs: '+key)
                rows.append(r); pairs[growth]=r
                influences[growth]={'county':pn.x,'aggregate':pa.x_agg,'augmented':pu.x,'exposure':pu.x_agg,'gap':pa.x_agg-pn.x}
            # Paired contrasts use the SAME regressions; no additional specification.
            for target in ['county','aggregate','augmented','exposure','gap']:
                key='gap' if target=='gap' else target+'_beta'
                diff=pairs['log'][key]-pairs['std'][key]
                se=float(np.linalg.norm(influences['log'][target]-influences['std'][target]))
                contrasts.append(dict(horizon=h,rung=rung,target=target,difference=diff,se=se,
                    p=float(2*student_t.sf(abs(diff/se),a.unit.nunique()-1))))
            print('h=%d %s: exposure standard %.3f, log %.3f'%(h,rung,pairs['std']['exposure_beta']*1e5,pairs['log']['exposure_beta']*1e5),flush=True)
    for name,records in [('aggregation.csv',rows),('paired_contrasts.csv',contrasts)]:
        pd.DataFrame(records).to_csv(OUT/name,index=False)
    require(all(sha(p)==digest for p,digest in cert['inputs'].items()),'Original files changed during estimation')
    write(OUT/'estimates.json',dict(status='PASS',horizons=list(HS),h3_reproduction='PASS',
        original_inputs_unchanged=True,rows=rows,contrasts=contrasts,
        files={name:sha(OUT/name) for name in ['aggregation.csv','paired_contrasts.csv']},
        estimator_sha256=sha(__file__),base_estimator_sha256=sha(BOX/'code/h3_analysis.py'),
        certificate_sha256=sha(CACHE/'certificate.json'),new_regression_fits=96))


def render():
    r=read(OUT/'estimates.json'); require(r['status']=='PASS','Unverified results')
    for name,digest in r['files'].items(): require(sha(OUT/name)==digest,'Saved estimates changed')
    SRC.mkdir(parents=True,exist_ok=True)
    f=pd.DataFrame(r['rows']); q=pd.DataFrame(r['contrasts'])
    def star(p): return r'$^{***}$' if p<.01 else r'$^{**}$' if p<.05 else r'$^{*}$' if p<.1 else ''
    def number(b,p): return '%.3f'%(b*1e5)+star(p)
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Geography & $h$ & Standard $\hat\delta$ & Log $\hat\delta$ & Log minus std. & SE of difference \\',r'\midrule']
    for rung in RUNGS:
        for h in HS:
            s=f.query('rung==@rung and horizon==@h and growth=="std"').iloc[0]
            l=f.query('rung==@rung and horizon==@h and growth=="log"').iloc[0]
            d=q.query('rung==@rung and horizon==@h and target=="exposure"').iloc[0]
            label={'czone':'Commuting zone','cea':'CEA','ea':'EA','state':'State'}[rung] if h==2 else ''
            lines.append(label+' & '+str(h)+' & '+number(s.exposure_beta,s.exposure_p)+' & '+number(l.exposure_beta,l.exposure_p)+' & '+number(d.difference,d.p)+' & %.3f'%(d.se*1e5)+r' \\')
        lines.append(r'\addlinespace')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (SRC/'overview.tex').write_text('\n'.join(lines))
    for rung in RUNGS:
        lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Horizon & $h=2$ & $h=3$ & $h=4$ & $h=5$ \\',
            r'Years & 1980--2016 & 1981--2016 & 1982--2016 & 1983--2016 \\',
            r'County-years & 113,146 & 110,088 & 107,030 & 103,972 \\',r'\midrule']
        for growth,title in [('std','A. Standard growth'),('log','B. Log growth')]:
            d=f.query('rung==@rung and growth==@growth').set_index('horizon').loc[list(HS)]
            lines.append(r'\multicolumn{5}{l}{\textbf{'+title+r'}} \\ \addlinespace')
            for label,key in [(r'County, own spending: $\hat\beta_b^{\rm naive}$','county'),(r'Region: $\hat\beta_a$','aggregate'),
                              ('Region minus county','gap'),(r'Augmented county, own: $\hat\beta_b$','augmented'),(r'Augmented county, exposure: $\hat\delta$','exposure')]:
                b='gap' if key=='gap' else key+'_beta'
                lines.append(label+' & '+' & '.join(number(v,p) for v,p in zip(d[b],d[key+'_p']))+r' \\')
                lines.append(' & '+' & '.join('(%.3f)'%(v*1e5) for v in d[key+'_se'])+r' \\')
            lines.append(r'$\hat\beta_a-\hat\beta_b-\hat\delta$ & '+' & '.join('%.3f'%(v*1e5) for v in d.identity_residual)+r' \\')
            lines.append(r'\addlinespace')
            for label,key in [('County first-stage F','county_F'),('Region first-stage F','aggregate_F'),
                ('Augmented own first-stage F','augmented_F'),('Augmented exposure first-stage F','exposure_F')]:
                lines.append(label+' & '+' & '.join('%.2f'%v for v in d[key])+r' \\')
            lines.append(r'\midrule' if growth=='std' else r'\bottomrule')
        lines.append(r'\end{tabular}')
        (SRC/(rung+'.tex')).write_text('\n'.join(lines))
    print('Rendered all horizons and all four geographic comparisons.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['build','estimate','render'])
    args=parser.parse_args(); globals()[args.stage]()
