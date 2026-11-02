"""Re-estimate the current draft, using unchanged source-extracted IV routines.

No population multiplier. No searches over specifications. No reference rewrites.
Build, estimate, check and render are explicitly separate stages.
"""
import argparse
import ast
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import t

from package_io import ROOT,require,sha,read,write,logs
sys.path.insert(0,str(ROOT/'code/vendor'))
from baseline_math import fit2sls
from h3_fit import fit
from heterogeneity_iv import iv
from county_builder import county_frame

DATA=ROOT/'data'
OUT=ROOT/'output'
REF=ROOT/'reference'
SPECS=[('main','state',3,'std',True),('unweighted','state',3,'std',False),
       ('log','state',3,'log',True),('h2','state',2,'std',True),
       ('h4','state',4,'std',True),('cea','cea',3,'std',True)]


def dump_csv(folder,name,records):
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(folder/name,index=False)


def certify(folder,name='estimates.json',**fields):
    write(folder/name,dict(status='PASS',**fields,
                          files={p.name:sha(p) for p in folder.glob('*.csv')}))


def close(actual,expected,label,rtol=1e-6,atol=1e-11):
    require(np.allclose(actual,expected,rtol=rtol,atol=atol,equal_nan=True),
            f'{label}: independently saved reference mismatch; actual={actual}, expected={expected}')


def verify():
    manifest=read(ROOT/'provenance/input_manifest.json')
    for item in manifest:
        require(sha(ROOT/item['path'])==item['sha256'],'Input changed: '+item['path'])
    for item in read(ROOT/'provenance/function_extraction.json'):
        filename={'fit':'h3_fit.py','iv':'heterogeneity_iv.py','county_frame':'county_builder.py'}[item['function']]
        text=(ROOT/'code/vendor'/filename).read_text(encoding='utf-8')
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==item['function'])
        import hashlib
        digest=hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()
        require(digest==item['function_sha256'],'Source-extracted numerical function changed')
    print('VERIFY PASS:',len(manifest),'frozen data/source/reference files; numerical functions unchanged.',flush=True)


def build():
    """Rebuild county IV/outcome inputs and check annual employment conservation."""
    spec=read(ROOT/'config/county_construction.json')
    spec.update(horizon=3,first_year=1981,n_observations=110088)
    sources=read(REF/'upstream_source_manifest.json')
    sources['snapshot_directory']='data/upstream/baseline_roll5_gap5'
    rebuilt=county_frame(spec,sources)
    frozen=pd.read_parquet(DATA/'disaggregation/county_h3.parquet')
    for col in rebuilt:
        require(col in frozen,'Unexpected county-builder output: '+col)
        if pd.api.types.is_numeric_dtype(rebuilt[col]):
            close(rebuilt[col],frozen[col],'County builder '+col,rtol=1e-12,atol=1e-10)
        else:
            require(rebuilt[col].equals(frozen[col]),'County builder categorical '+col)
    require(rebuilt.ccg.nunique()==3058 and len(rebuilt)==110088,'County support changed')
    errors=[];industry_ids={};support=np.empty(0,dtype=np.int64)
    for year in range(1978,2017):
        cells=pd.read_parquet(DATA/f'upstream/naics6_annual/y{year}.parquet',columns=['ccg','code','emp'])
        for code in cells.code.unique():
            if code not in industry_ids:industry_ids[code]=len(industry_ids)
        require(len(industry_ids)<16384,'Expand bookkeeping key capacity explicitly')
        keys=cells.ccg.to_numpy(np.int64)*16384+cells.code.map(industry_ids).to_numpy(np.int64)
        support=np.union1d(support,np.unique(keys))
        totals=cells.groupby('ccg').emp.sum()
        if year>=1981:
            target=frozen.loc[frozen.year.eq(year)].set_index('ccg').E
        elif year<=1980:
            target=frozen.loc[frozen.year.eq(year+3)].set_index('ccg').E_b
        close(totals.reindex(target.index),target,f'Annual employment conservation {year}',rtol=1e-12,atol=1e-8)
        errors.append(float(abs(totals.reindex(target.index)-target).max()))
    # Independently check the actual aggregation data, outcome weights and IVs.
    for _,rung,h,_,_ in SPECS:
        c=pd.read_parquet(DATA/f'aggregation/county_{rung}_h{h}.parquet')
        a=pd.read_parquet(DATA/f'aggregation/aggregate_{rung}_h{h}.parquet')
        require(c.ccg.nunique()==3058 and len(c)==3058*(2017-1978-h),'Aggregation county support changed')
        g=c.groupby(['unit','year'])
        close(g.weight.sum(),np.ones(len(a)),f'{rung} h{h} share sums',atol=1e-12)
        for value,target in [('std_y','std_y'),('x','x_agg'),('z','z_agg')]:
            v=(c.weight*c[value]).groupby([c.unit,c.year]).sum()
            close(v.to_numpy(),a.sort_values(['unit','year'])[target],f'{rung} h{h} reconstruction {value}',atol=1e-10)
    path=OUT/'build';path.mkdir(parents=True,exist_ok=True)
    rebuilt.to_parquet(path/'county_h3.parquet',index=False)
    write(path/'checks.json',dict(status='PASS',county_years=110088,counties=3058,
                                annual_employment_max_error=max(errors),
                                naics6_balanced_cells=int(len(support)*frozen.year.nunique()),
                                boundary='frozen_constructed_annual_sources'))
    print('BUILD PASS: county spending/IV rebuilt; employment and regional aggregation conserved.',flush=True)


def check_disfit(column,result,ncells,ncy):
    display=pd.read_csv(REF/'displayed_estimates.csv')
    expected=display.query('table==5 and horizon==3 and column==@column').iloc[0]
    close([result['beta'],result['se'],result['F']],expected[['beta','se','F']].astype(float),f'Disaggregation col {column}')
    require(ncy==int(expected.county_years),'County-year count differs in column '+str(column))
    source=('disaggregation_h234_complete.csv' if column in [1,6,7,8] else
            'log_lag_full_county.csv' if column in [2,3] else
            'log_naics6.csv' if column==4 else 'tables45_new.csv' if column==5 else 'table2_new.csv')
    original=pd.read_csv(REF/'coefficient_files'/source)
    selected=original.horizon.eq(3)&original.rung.eq(expected.rung)&original.growth.eq(expected.growth)
    if source in ['log_naics6.csv','tables45_new.csv']:selected &= original.weight.eq('lag')
    if source=='table2_new.csv':selected &= original.industry_year_fe.eq(expected.industry_year_fe)
    counts=original.loc[selected]
    require(len(counts)==1 and ncells==int(counts.iloc[0].n_estimation_cells),
            'Cell count differs from original producer in column '+str(column))
    row=expected.to_dict()
    row.update(beta=result['beta'],se=result['se'],F=result['F'],t=result['beta']/result['se'],
               county_years=ncy,observations=ncells,pvalue=float(2*t.sf(abs(result['beta']/result['se']),3057)))
    print(f'DISAGG col {column}: {result["beta"]*1e5:.6f} ({result["se"]*1e5:.6f}) F={result["F"]:.6f}',flush=True)
    return row


def scalar_fit(f,y,w,n):
    return fit2sls(np.asarray(y),f.x.to_numpy(),f.z.to_numpy(),np.asarray(w,float),
                   f.fe_county.to_numpy(np.int64),f.fe_year.to_numpy(np.int64),
                   f.fe_county.to_numpy(np.int64),singleton=False,correction_n=n)


def small():
    require(read(OUT/'build/checks.json')['status']=='PASS','Run the build/check stage first')
    f=pd.read_parquet(DATA/'disaggregation/county_h3.parquet')
    rows=[]
    for col,growth in [(1,'log_y'),(6,'y')]:
        rows.append(check_disfit(col,scalar_fit(f,f[growth],np.ones(len(f)),len(f)),len(f),len(f)))
    for rung,lc,mc in [('naics2',2,7),('naics4_alloc',3,8)]:
        s=pd.read_parquet(DATA/f'disaggregation/{rung}_h3.parquet')
        l=pd.read_parquet(DATA/f'disaggregation/lag_{rung}_h3.parquet')
        keep=l.log_count.gt(0)
        rows.append(check_disfit(lc,scalar_fit(f.loc[keep],l.loc[keep,'weighted_log_mean'],
                   l.loc[keep,'survivor_lag_mass'],int(l.log_count.sum())),int(l.log_count.sum()),int(keep.sum())))
        rows.append(check_disfit(mc,scalar_fit(f,s.mpgr_mean,s.mpgr_mass,len(f)),int(s.total_cells.sum()),len(f)))
    d=pd.read_parquet(DATA/'disaggregation/decomposition.parquet')
    keep=d['count'].gt(0)
    rows.append(check_disfit(4,scalar_fit(d.loc[keep],d.loc[keep,'mean_log'],d.loc[keep,'lag_mass'],int(d['count'].sum())),
                             int(d['count'].sum()),int(keep.sum())))
    s=pd.read_parquet(DATA/'disaggregation/mpgr_naics6_h3.parquet')
    # The original no-industry-year-FE count includes the balanced zero-weight
    # cells; the industry-year cache contains only active cells. Retain both
    # reporting conventions, without changing any estimation moment or SSC N.
    cells=read(OUT/'build/checks.json')['naics6_balanced_cells']
    rows.append(check_disfit(9,scalar_fit(f,s.y,s.mass,len(f)),cells,len(f)))
    dump_csv(OUT/'disaggregation','small.csv',rows)

    # Original exact coefficient-gap decomposition; retain the tiny sample term.
    d=d.assign(weight=1.,unit=lambda x:x.ccg)
    full,pfull=fit(d,'log_y',['x'],['z'])
    common=d.loc[keep].copy();fits={};psis={}
    for key,col in [('baseline_common','log_y'),('normalized','mean_log'),
                    ('entry','entry'),('exit','exit'),('jensen','jensen')]:
        r,p=fit(common,col,['x'],['z']);fits[key]=r['x'];psis[key]=p.x
    r,p=fit(common.assign(weight=common.lag_mass),'mean_log',['x'],['z'],correction_n=int(common['count'].sum()))
    fits['original']=r['x'];psis['original']=p.x
    fits['baseline_full']=full['x'];psis['baseline_full']=pfull.x
    comps=[]
    for key in ['entry','exit','jensen','renormalization','sample']:
        if key in fits: value,influence=fits[key]['beta'],psis[key]
        else:
            a,b=('normalized','original') if key=='renormalization' else ('baseline_full','baseline_common')
            value,influence=fits[a]['beta']-fits[b]['beta'],psis[a]-psis[b]
        comps.append(dict(component=key,beta=value,se=float(np.linalg.norm(influence))))
    expected=pd.read_csv(REF/'decomposition.csv').set_index('component')
    actual=pd.DataFrame(comps).set_index('component')
    close(actual.loc[expected.index,['beta','se']],expected[['beta','se']],'Decomposition')
    gap=fits['baseline_full']['beta']-fits['original']['beta']
    error=float(actual.beta.sum()-gap);require(abs(error)<1e-12,'Coefficient decomposition fails')
    folder=OUT/'h3_analysis';dump_csv(folder,'decomposition.csv',comps)
    certify(folder,fits=fits,components=comps,decomposition_error=error,
            estimator_sha256=sha(ROOT/'provenance/original_sandbox/h3_analysis.py'))


def cells():
    """Two large, unchanged NAICS6 industry-year-FE regressions, run in sequence."""
    results=[]
    for col,growth in [(5,'log'),(10,'mpgr')]:
        if growth=='log':
            d=pd.read_parquet(DATA/'cells/log_iy_h3.parquet',
                             columns=['ccg','year','y','x','z','w_lag','fe_county','fe_industry_year'])
            y,x,z,w=(d[k].to_numpy() for k in ['y','x','z','w_lag'])
        else:
            columns=['ccg','year','fe_county','fe_industry_year','x','z','y_mpgr','w_mpgr']
            d=pd.read_parquet(DATA/'cells/mpgr_iy_h3',columns=columns,
                             filters=[('w_mpgr','>',0)])
            d=d.loc[d.y_mpgr.notna()]
            y,x,z,w=(d[k].to_numpy() for k in ['y_mpgr','x','z','w_mpgr'])
        ncy=len(d[['ccg','year']].drop_duplicates());n=len(d)
        print('CELL START',growth,n,'cells',flush=True)
        c=d.fe_county.to_numpy(np.int64);f=d.fe_industry_year.to_numpy(np.int64)
        r=fit2sls(y,x,z,w,c,f,c.copy(),singleton=False,correction_n=ncy if growth=='mpgr' else n)
        results.append(check_disfit(col,r,n,ncy))
        dump_csv(OUT/'disaggregation',f'column_{col}.csv',[results[-1]])
        del d,y,x,z,w,c,f,r;gc.collect()
    print('CELL PASS: both large regressions match the independent saved estimates.',flush=True)


def aggregation():
    reference=pd.read_csv(REF/'aggregation_selected.csv').set_index('spec')
    rows=[];fitstats=[]
    for name,rung,h,growth,weighted in SPECS:
        c=pd.read_parquet(DATA/f'aggregation/county_{rung}_h{h}.parquet')
        a=pd.read_parquet(DATA/f'aggregation/aggregate_{rung}_h{h}.parquet')
        if not weighted:c=c.assign(weight=1.)
        nb,pn=fit(c,growth+'_y',['x'],['z'])
        ab,pa=fit(a,growth+'_y',['x_agg'],['z_agg'],fe='unit')
        ub,pu=fit(c,growth+'_y',['x','x_agg'],['z','z_agg'])
        gap=ab['x_agg']['beta']-nb['x']['beta'];se=float(np.linalg.norm(pa.x_agg-pn.x))
        row=dict(spec=name,rung=rung,horizon=h,growth=growth,n_counties=c.ccg.nunique(),
                 n_county_years=len(c),n_units=a.unit.nunique(),n_aggregate_years=len(a),
                 gap=gap,gap_se=se,gap_p=2*t.sf(abs(gap/se),a.unit.nunique()-1),
                 identity_residual=ab['x_agg']['beta']-ub['x']['beta']-ub['x_agg']['beta'])
        for prefix,vals in [('county',nb['x']),('aggregate',ab['x_agg']),('augmented',ub['x']),('exposure',ub['x_agg'])]:
            row.update({prefix+'_'+key:value for key,value in vals.items()})
        expected=reference.loc[name]
        for key in ['augmented_beta','augmented_se','augmented_F','exposure_beta','exposure_se','exposure_F']:
            close(row[key],expected[key],name+' '+key)
        if name=='main':
            for key in ['county_beta','aggregate_beta','aggregate_se','aggregate_F','gap','gap_se','identity_residual']:
                close(row[key],expected[key],'Main '+key)
        rows.append(row)
        for model,data in [('augmented',c),('aggregate',a),('naive_county',c)]:
            fitstats.append(dict(spec=name,model=model,n=len(data),clusters=data.unit.nunique()))
        print('AGG',name,'delta',row['exposure_beta']*1e5,flush=True)
    folder=OUT/'aggregation_summary_tables'
    dump_csv(folder,'selected_results.csv',rows)
    dump_csv(folder,'fit_statistics.csv',fitstats)
    certify(folder,'certificate.json')


def heterogeneity():
    c=pd.read_parquet(DATA/'heterogeneity/county.parquet')
    a=pd.read_parquet(DATA/'heterogeneity/state.parquet')
    reference=pd.read_csv(REF/'heterogeneity_coefficients.csv').set_index(['controls','model','variable'])
    rows=[];summaries=[];design=[]
    for controls in range(4):
        ids=[] if controls==0 else [controls];exo=[f'A{i}' for i in ids]
        en=['x','x_agg']+[f'xc{i}' for i in ids]+[f'Q{i}' for i in ids]
        zz=['z','z_agg']+[f'zc{i}' for i in ids]+[f'R{i}' for i in ids]
        ae=['x_agg']+[f'Q{i}' for i in ids];az=['z_agg']+[f'R{i}' for i in ids]
        fits={}
        for model,data,xx,zs,fe in [('county',c,en,zz,'ccg'),('state',a,ae,az,'unit')]:
            rr,psi,export,meta=iv(data,xx,zs,exo,fe)
            for item in rr:
                old=reference.loc[(controls,model,item['variable'])]
                close([item['beta'],item['se'],item['F']],old[['beta','se','F']],f'Control {controls} {model} {item["variable"]}')
            rows.extend(dict(controls=controls,model=model,**item) for item in rr)
            fits[model]=pd.DataFrame(rr).set_index('variable')
            design.append(dict(name=f'{model}_{controls}',controls=controls,model=model,
                               endog=xx,excluded=zs,exog=exo,**meta))
        item=fits['county'].loc['x_agg']
        summaries.append(dict(controls=controls,delta=item.beta,se=item.se,p=item.p,
                              identity_residual=fits['state'].loc['x_agg','beta']-fits['county'].loc['x','beta']-item.beta,
                              n_excluded=len(zz)))
    folder=OUT/'heterogeneity_separate'
    dump_csv(folder,'coefficients.csv',rows);dump_csv(folder,'delta.csv',summaries)
    dump_csv(folder,'design_audit.csv',design);certify(folder)


def balance():
    d=pd.read_parquet(DATA/'aggregation/county_state_h3.parquet')
    chars=pd.read_parquet(DATA/'heterogeneity/characteristics_reference.parquet').set_index('ccg')
    reference=pd.read_csv(REF/'balance.csv').set_index(['year','characteristic'])
    keys=['employment','population','jobs_per_resident','income_per_resident','manufacturing','concentration']
    rows=[]
    for year,prefix in [(1978,'old_'),(2000,'C_')]:
        for key in keys:
            d['outcome']=d.x*d.ccg.map(chars[prefix+key])
            r,_=fit(d,'outcome',['x','x_agg'],['z','z_agg'])
            val=r['x_agg'];old=reference.loc[(year,key)]
            close([val['beta'],val['se']],old[['pi','se']],f'Balance {year} {key}',atol=1e-9)
            rows.append(dict(year=year,characteristic=key,rho=r['x']['beta'],rho_se=r['x']['se'],
                             pi=val['beta'],se=val['se'],p=val['p'],F_county=r['x']['F'],F_regional=val['F']))
    folder=OUT/'balance_reference_2000';dump_csv(folder,'balance.csv',rows);certify(folder)


def check():
    smallrows=pd.read_csv(OUT/'disaggregation/small.csv')
    combined=pd.concat([smallrows]+[pd.read_csv(OUT/f'disaggregation/column_{col}.csv') for col in [5,10]])
    combined=combined.sort_values('column');require(combined.column.tolist()==list(range(1,11)),'Incomplete disaggregation reproduction')
    for growth in ['log','mpgr']:
        mask=combined.growth.eq(growth);b=combined.loc[mask].iloc[0].beta
        combined.loc[mask,'percent_deviation']=100*(combined.loc[mask,'beta']/b-1)
    folder=OUT/'disaggregation_no_population'
    dump_csv(folder,'displayed_estimates.csv',combined.to_dict('records'))
    write(folder/'provenance.json',dict(status='PASS',files={'displayed_estimates.csv':sha(folder/'displayed_estimates.csv')},
          displayed_estimates_sha256=sha(folder/'displayed_estimates.csv'),
          inputs={p.name:sha(p) for p in (REF/'coefficient_files').glob('*.csv')}))
    for name,manifest in [('aggregation_summary_tables','certificate.json'),('h3_analysis','estimates.json'),
                          ('heterogeneity_separate','estimates.json'),('balance_reference_2000','estimates.json')]:
        require(read(OUT/name/manifest)['status']=='PASS','Incomplete stage '+name)
    write(OUT/'replication_check.json',dict(status='PASS',disaggregation_columns=10,
          aggregation_specs=6,heterogeneity_specs=4,balance_tests=12,
          references='pre-existing independent outputs; not regenerated',population_multiplier=False,
          code_sha256=sha(__file__)))
    print('CHECK PASS: all paper estimates/SEs/F-statistics and decomposition match preserved references.',flush=True)


def render(stata=None):
    require(read(OUT/'replication_check.json')['status']=='PASS','Run all numerical checks before rendering')
    subprocess.run([sys.executable,'-B',str(ROOT/'code/render_results.py')],check=True)
    if not stata:
        stata=os.environ.get('STATA_BIN') or shutil.which('stata-mp') or shutil.which('stata')
        if not stata and Path('C:/Program Files/Stata17/StataMP-64.exe').exists():
            stata='C:/Program Files/Stata17/StataMP-64.exe'
    require(stata,'Set STATA_BIN or supply --stata for the two figures')
    args=[stata,'/e' if os.name=='nt' else '-b','do',str(ROOT/'code/graph_results.do'),str(ROOT),str(logs())]
    kwargs={}
    if os.name=='nt':kwargs['creationflags']=subprocess.CREATE_NO_WINDOW
    subprocess.run(args,check=True,**kwargs)
    log=(logs()/'graphics_detail.log').read_text(errors='replace')
    require('STATA_RECAP_GRAPHICS_PASS' in log,'Stata did not complete graphics; inspect its log')
    source=ROOT/'source/results_recap_v3';build=logs()/'latex';build.mkdir(exist_ok=True)
    command=['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(build),'NS_results_recap_v3.tex']
    for _ in range(2):
        with (logs()/'pdflatex_stdout.log').open('w') as stream:
            subprocess.run(command,cwd=source,stdout=stream,stderr=subprocess.STDOUT,check=True)
    dest=OUT/'pdf';dest.mkdir(exist_ok=True)
    shutil.copy2(build/'NS_results_recap_v3.pdf',dest/'NS_results_recap.pdf')
    print('RENDER PASS:',dest/'NS_results_recap.pdf',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['verify','build','small','cells','aggregation','heterogeneity','balance','check','render','all'])
    p.add_argument('--stata',default=None)
    a=p.parse_args()
    if a.stage=='all':
        for stage in [verify,build,small,cells,aggregation,heterogeneity,balance,check]:stage()
        render(a.stata)
    elif a.stage=='render':render(a.stata)
    else:globals()[a.stage]()


if __name__=='__main__':
    main()
