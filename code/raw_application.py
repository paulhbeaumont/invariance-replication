"""Build the locked paper inputs using only a clean run's raw-derived sources.

The arithmetic below follows the preserved original builders. The NAICS6
reconciliation function and numerical estimators are executed unchanged.
Original prepared inputs are permitted ONLY inside independent comparisons
after the new input has been written. They are never construction seeds.
"""
import ast
from contextlib import contextmanager
import gc
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from package_io import ROOT,logs,read,require,sha,write
sys.path.insert(0,str(ROOT/'code/vendor'))
sys.path.insert(0,str(ROOT/'.analysis_extra'))
import numpy as np
import pandas as pd
import duckdb
import reproduce
from county_builder import county_frame
from baseline_math import wins

COMPARISONS = []
FUNCTIONS = []
_reference_read = False
_original_read = pd.read_parquet
_anchors = None
ACTIVE_RUN = None


def copy_source_tree(source,destination):
    """Preserve complete provenance even under a deeply nested Windows run."""
    def native(path):
        value=str(Path(path).resolve())
        if os.name=='nt' and not value.startswith('\\\\?\\'):
            return '\\\\?\\UNC\\'+value[2:] if value.startswith('\\\\') else '\\\\?\\'+value
        return value
    shutil.copytree(native(source),native(destination),ignore=shutil.ignore_patterns('__pycache__'))


def construction_logic_sha256():
    tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    text=Path(__file__).read_text(encoding='utf-8')
    names={'construction_read_guard','checked_original','original_function','compare','save','query',
           'growth_roundoff','resume_annual','copy_source_tree',
           'counties','smaller_industries','six_digit_annual','six_digit_cells','aggregate','characteristics','build'}
    source='\n'.join(ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
    return hashlib.sha256(source.encode()).hexdigest()


@contextmanager
def construction_read_guard():
    protected = [ROOT/'data'/name for name in ('upstream','disaggregation','cells','aggregation','heterogeneity')]
    def guarded(path,*args,**kwargs):
        if isinstance(path,(str,os.PathLike)):
            actual = Path(path).resolve()
            if any(actual==p.resolve() or p.resolve() in actual.parents for p in protected):
                require(_reference_read,'Forbidden prepared-data read during construction/estimation: '+str(path))
        return _original_read(path,*args,**kwargs)
    previous = pd.read_parquet
    pd.read_parquet = guarded
    try:
        yield
    finally:
        pd.read_parquet = previous


def checked_original(path):
    global _anchors
    if _anchors is None:
        _anchors = {r['path']:r['sha256'] for r in read(ROOT/'provenance/input_manifest.json')}
    relative = path.relative_to(ROOT).as_posix()
    require(relative in _anchors and sha(path)==_anchors[relative],
            'Independent original changed or is not anchored: '+relative)
    return _anchors[relative]


def original_function(relative,name):
    path = ROOT/relative
    digest = checked_original(path)
    text = path.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    source = ast.get_source_segment(text,node)
    scope = dict(np=np,pd=pd,require=require)
    exec(compile(source,str(path),'exec'),scope)
    FUNCTIONS.append(dict(source=relative,source_sha256=digest,function=name,
                          extracted_function_sha256=hashlib.sha256(source.encode()).hexdigest(),modified=False))
    return scope[name]


def growth_roundoff(new,old,column,relative,baseline_lag=None):
    """Audit an unchanged growth formula at numerically tiny positive cells.

    Endpoint and weight comparisons retain their original strict tolerances.
    No outcome is rounded, clipped, recoded, or replaced by a reference value.
    """
    if relative.startswith('cells/mpgr_iy_h3/') and column in ('y_mpgr','y_log'):
        for frame in (new,old):
            cur,lag=frame.cur.to_numpy(),frame.lag.to_numpy()
            if column=='y_mpgr':
                expected=2*(cur-lag)/(cur+lag)
            else:
                valid=(cur>0)&(lag>0);expected=np.full(len(frame),np.nan)
                expected[valid]=np.log(cur[valid])-np.log(lag[valid])
            require(np.allclose(frame[column],expected,rtol=1e-12,atol=1e-10,equal_nan=True),
                    'Growth formula differs from the independent endpoint definition: '+relative+' '+column)
        weight='w_mpgr' if column=='y_mpgr' else 'lag/full_county_E_b'
        if column=='y_log':
            require(baseline_lag is not None,'Log audit needs actual full-county lag weights, not survivor midpoint weights')
            county=[v for v in COMPARISONS if v['file']=='disaggregation/county_h3.parquet' and v['status']=='PASS']
            require(len(county)==1 and any(v['column']=='E_b' and v['max_abs_difference']==0 for v in county[0]['fields']),
                    'Log audit baseline employment is not independently identical')
    elif relative=='cells/log_iy_h3.parquet' and column=='y':
        checked={v['file'] for v in COMPARISONS if v['status']=='PASS'}
        require(all(f'cells/mpgr_iy_h3/y{year}.parquet' in checked for year in range(1981,2017)),
                'Combined log comparison requires all independently checked annual horizon cells')
        weight='w_lag'
    elif relative=='disaggregation/decomposition.parquet' and column=='lag_log':
        checked={v['file'] for v in COMPARISONS if v['status']=='PASS'}
        require('cells/log_iy_h3.parquet' in checked,'Decomposition audit requires the independently checked log cells')
        for frame in (new,old):
            expected=np.where(frame.E_b_c.gt(0),frame.mean_log*frame.E_b_c,0.)
            require(np.allclose(frame.lag_log,expected,rtol=1e-12,atol=1e-10,equal_nan=True),
                    'Decomposition numerator differs from the original mean-log definition')
        weight='inverse_continuing_baseline_employment'
    else:
        raise RuntimeError('Independent input values differ: '+relative+' '+column)
    a,b=new[column].to_numpy(float),old[column].to_numpy(float)
    require(np.array_equal(np.isnan(a),np.isnan(b)),'Growth missing-value support differs')
    if weight=='inverse_continuing_baseline_employment':
        w=np.divide(1.,new.E_b_c,out=np.zeros(len(new)),where=new.E_b_c.gt(0))
        v=np.divide(1.,old.E_b_c,out=np.zeros(len(old)),where=old.E_b_c.gt(0))
    elif weight=='lag/full_county_E_b':
        denominator=new.ccg.map(baseline_lag).to_numpy(float)
        require(np.isfinite(denominator).all() and (denominator>0).all(),'Invalid log audit denominator')
        w,v=new.lag.to_numpy(float)/denominator,old.lag.to_numpy(float)/denominator
    else:
        w,v=new[weight].to_numpy(float),old[weight].to_numpy(float)
    require(np.allclose(w,v,rtol=1e-12,atol=1e-10,equal_nan=True),'Growth weight differs')
    differences=[]
    for power in (1,2):
        # IV cluster scores multiply the residual by w, and the sandwich
        # squares that weighted score. w*y^2 is not that squared quantity.
        left,right=(w*a)**power,(v*b)**power
        require(np.allclose(left,right,rtol=1e-12,atol=1e-10,equal_nan=True),
                'Weighted growth moment differs beyond numerical roundoff: '+relative+' '+column)
        delta=np.abs(left-right)
        differences.append(float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.0)
    return dict(status='AMPLIFIED_ENDPOINT_ROUNDOFF',weight=weight,
                max_weighted_outcome_difference=differences[0],max_squared_weighted_outcome_difference=differences[1],
                weighted_moment_rtol=1e-12,weighted_moment_atol=1e-10,
                growth_formula_unchanged=True,data_altered=False,
                independent_coefficient_SE_F_and_support_checks_still_required=True)


def compare(frame,relative,keys,record_dir,columns=None,baseline_lag=None):
    """Independent target read, never used to construct or change the new frame."""
    global _reference_read
    path = ROOT/'data'/relative
    digest = checked_original(path)
    _reference_read = True
    try:
        old = pd.read_parquet(path,columns=columns)
    finally:
        _reference_read = False
    new = frame.sort_values(keys).reset_index(drop=True)
    old = old.sort_values(keys).reset_index(drop=True)
    require(len(old)==len(new),'Independent row count differs: '+relative)
    require(set(old.columns)<=set(new.columns),'Missing reconstructed columns: '+relative)
    fields=[]
    for column in old:
        left,right = new[column],old[column]
        roundoff=None
        if pd.api.types.is_numeric_dtype(right):
            a,b = left.to_numpy(),right.to_numpy()
            if column in keys or pd.api.types.is_integer_dtype(right) or pd.api.types.is_bool_dtype(right):
                require(np.array_equal(a,b),'Independent identifier/count differs: '+relative+' '+column)
                difference = 0.0
                changed = 0
            else:
                a,b = np.asarray(a,float),np.asarray(b,float)
                if not np.allclose(a,b,rtol=1e-12,atol=1e-10,equal_nan=True):
                    roundoff=growth_roundoff(new,old,column,relative,baseline_lag)
                delta = np.abs(a-b)
                difference = float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.0
                changed = int((~((a==b)|(np.isnan(a)&np.isnan(b)))).sum())
            fields.append(dict(column=column,max_abs_difference=difference,nonidentical_values=changed,
                               original_dtype=str(right.dtype),rebuilt_dtype=str(left.dtype)))
            if roundoff is not None:
                fields[-1]['roundoff_audit']=roundoff
        else:
            require(((left==right)|(left.isna()&right.isna())).all(),
                    'Independent categorical values differ: '+relative+' '+column)
            fields.append(dict(column=column,exact_values=True))
    COMPARISONS.append(dict(file=relative,status='PASS',rows=len(new),keys=keys,
                            original_sha256=digest,comparison_only_not_construction_input=True,fields=fields))
    write(record_dir/'input_comparisons.json',dict(status='IN_PROGRESS',files=COMPARISONS))
    print('INDEPENDENT INPUT PASS',relative,len(new),'rows',flush=True)


def save(frame,app,relative,keys,record_dir,reference=None):
    path = app/'data'/relative
    require(not path.exists(),'Refusing to replace reconstructed input: '+relative)
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(path,index=False)
    if reference is not None:
        compare(frame,reference,keys,record_dir)


def query(connection,sql,parameters):
    # Queries are only over fresh run paths or registered fresh dataframes.
    for value in parameters:
        if isinstance(value,(str,Path)) and (str(value).endswith('.parquet') or '*.parquet' in str(value)):
            path = Path(value)
            wildcard=any(char in path.name for char in '*?[')
            require(not any(char in str(path.parent) for char in '*?['),
                    'Only a filename may contain a query wildcard: '+str(value))
            # Windows cannot resolve a literal '*'. Resolve its containing
            # directory instead; DuckDB expands the unchanged filename glob.
            actual=(path.parent if wildcard else path).resolve()
            require(ACTIVE_RUN is not None and
                    (ACTIVE_RUN.resolve() in actual.parents or (wildcard and actual==ACTIVE_RUN.resolve())),
                    'Query tried to use an input outside this clean run: '+str(value))
    return connection.execute(sql,[str(v) if isinstance(v,Path) else v for v in parameters])


def counties(run,app,record_dir):
    sources = dict(snapshot_directory=(run/'source51').relative_to(ROOT).as_posix(),
                   inputs={p.name:sha(p) for p in (run/'source51').glob('*.parquet')})
    require(set(sources['inputs'])=={'source51_county_panel.parquet','source51_contracts_net.parquet'},'Annual source51 inputs incomplete')
    frames={}
    for h in (2,3,4):
        spec = read(ROOT/'config/county_construction.json')
        spec.update(horizon=h,first_year=1978+h,n_observations=3058*(2017-1978-h))
        f = county_frame(spec,sources)
        f['log_y']=np.log(f.E)-np.log(f.E_b)
        frames[h]=f
    save(frames[3],app,'disaggregation/county_h3.parquet',['ccg','year'],record_dir,'disaggregation/county_h3.parquet')
    return frames


def smaller_industries(run,app,frames,record_dir):
    units = np.sort(frames[3].ccg.unique())
    employment = run/'1_current_paper/_data/temp/employment'
    for rung in ('naics2','naics4_alloc'):
        annual={}; keys=None
        for year in range(1978,2017):
            f=pd.read_parquet(employment/rung/f'y{year}.parquet',columns=['ccg','code','emp'])
            f=f[f.ccg.isin(units)].sort_values(['ccg','code']).reset_index(drop=True)
            k=f[['ccg','code']].to_numpy(np.int64)
            if keys is None:
                keys=k
                cell_count=np.bincount(np.searchsorted(units,keys[:,0]),minlength=len(units))
            require(np.array_equal(keys,k),'Fresh annual industry support changed')
            annual[year]=f.emp.to_numpy(np.float64)
        ci=np.searchsorted(units,keys[:,0]);parts=[];lagparts=[]
        for year,g in frames[3].groupby('year',sort=True):
            g=g.sort_values('ccg');cur,lag=annual[year],annual[year-3]
            den=cur+lag;denc=(g.E+g.E_b).to_numpy()
            mpgr=np.divide(2*(cur-lag),den,out=np.zeros_like(den),where=den!=0)
            w=den/denc[ci];pos=(cur>0)&(lag>0)
            log=np.zeros_like(den);log[pos]=np.log(cur[pos])-np.log(lag[pos])
            mass=np.bincount(ci,weights=w,minlength=len(units))
            summed=np.bincount(ci,weights=w*mpgr,minlength=len(units))
            count=np.bincount(ci,weights=pos.astype(np.int64),minlength=len(units)).astype(np.int64)
            logsum=np.bincount(ci,weights=log,minlength=len(units))
            require(abs(summed-g.y.to_numpy()).max()<1e-10 and abs(mass-1).max()<1e-10,'Fresh midpoint reconstruction failed')
            parts.append(pd.DataFrame(dict(ccg=units,year=year,mpgr_mean=summed/mass,mpgr_mass=mass,
                log_mean=np.divide(logsum,count,out=np.full_like(logsum,np.nan),where=count>0),log_count=count,total_cells=cell_count,
                zero_both=np.bincount(ci,weights=((cur==0)&(lag==0)).astype(int),minlength=len(units)).astype(int),
                entrant_cells=np.bincount(ci,weights=((cur>0)&(lag==0)).astype(int),minlength=len(units)).astype(int),
                exit_cells=np.bincount(ci,weights=((cur==0)&(lag>0)).astype(int),minlength=len(units)).astype(int))))
            weight=lag[pos]/g.E_b.to_numpy()[ci[pos]]
            lagmass=np.bincount(ci[pos],weights=weight,minlength=len(units))
            ysum=np.bincount(ci[pos],weights=weight*log[pos],minlength=len(units))
            lagparts.append(pd.DataFrame(dict(ccg=units,year=year,weighted_log_sum=ysum,survivor_lag_mass=lagmass,
                weighted_log_mean=np.divide(ysum,lagmass,out=np.full_like(lagmass,np.nan),where=lagmass>0),log_count=count)))
        for name,parts_ in [(rung,parts),('lag_'+rung,lagparts)]:
            stats=pd.concat(parts_,ignore_index=True).sort_values(['ccg','year']).reset_index(drop=True)
            save(stats,app,'disaggregation/'+name+'_h3.parquet',['ccg','year'],record_dir,'disaggregation/'+name+'_h3.parquet')
        del annual;gc.collect()


def six_digit_annual(run,app,frames,record_dir):
    split_parent = original_function('provenance/original_4a/code/build/naics6_cells.py','split_parent')
    code = ROOT/'provenance/historical_fc9df56/4_Macro_version/step_1/code'
    crosswalk_path = code/'reference/county_ccg_crosswalk.csv'
    x=pd.read_csv(crosswalk_path,dtype=str);x=x[x.ccg_id.ne('unplaceable')]
    crosswalk=pd.DataFrame({'fips':x.county_fips5.astype(int),'ccg':x.ccg_id.astype(int)})
    units=pd.DataFrame({'ccg':sorted(frames[3].ccg.unique())})
    source=run/'sources/efsy_panel_naics.parquet'
    parentroot=run/'1_current_paper/_data/temp/employment/naics4_alloc'
    with duckdb.connect() as con:
        con.execute('SET threads=1')
        con.register('crosswalk',crosswalk);con.register('units',units)
        for year in range(1978,2017):
            six=query(con,'''WITH src AS (
                SELECT *, CASE WHEN fipstate=11 AND fipscty=999 AND year BETWEEN 1976 AND 2003
                THEN 11001 ELSE fipstate*1000+fipscty END AS fips
                FROM read_parquet(?) WHERE year=?)
                SELECT c.ccg, naics12 AS code, CAST(substr(naics12,1,4) AS INTEGER) AS parent, sum(emp) AS emp
                FROM src JOIN crosswalk c USING(fips) JOIN units u ON c.ccg=u.ccg
                WHERE regexp_full_match(naics12,'[0-9]{6}')
                AND (fipscty<>999 OR (fipstate=11 AND year BETWEEN 1976 AND 2003))
                GROUP BY c.ccg,naics12''',[source,year]).df()
            p=pd.read_parquet(parentroot/f'y{year}.parquet',columns=['ccg','code','emp'])
            p=p[p.ccg.isin(units.ccg)].rename(columns={'code':'parent'})
            cells,accounting=split_parent(p,six);cells['year']=year
            cells=cells.sort_values(['ccg','code']).reset_index(drop=True)
            save(cells,app,f'upstream/naics6_annual/y{year}.parquet',['ccg','code'],record_dir,f'upstream/naics6_annual/y{year}.parquet')
            print('FRESH NAICS6',year,len(cells),'cells',flush=True)


def six_digit_cells(run,app,frames,record_dir):
    source=app/'data/upstream/naics6_annual'; f=frames[3]
    parts=[];decompositions=[];active=0
    with duckdb.connect() as con:
        con.execute('SET threads=1')
        codes=query(con,'SELECT DISTINCT code,parent FROM read_parquet(?) ORDER BY code',[source/'y*.parquet']).df()
        codes['industry']=np.arange(len(codes),dtype=np.int32);con.register('codes',codes)
        support=query(con,'SELECT count(*) FROM (SELECT DISTINCT ccg,code FROM read_parquet(?))',[source/'y*.parquet']).fetchone()[0]
        for year,g in f.groupby('year',sort=True):
            con.register('g',g)
            d=query(con,'''WITH pair AS (
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
            FROM cell''',[source/f'y{year}.parquet',source/f'y{year-3}.parquet']).df()
            require(not d.duplicated(['ccg','industry']).any(),'Duplicate fresh horizon cells')
            d['weighted_y']=d.y_mpgr*d.w_mpgr
            ag=d.groupby('ccg',sort=True).agg(y=('weighted_y','sum'),mass=('w_mpgr','sum'))
            ref=g.sort_values('ccg')
            require(abs(ag.y.to_numpy()-ref.y.to_numpy()).max()<1e-10 and abs(ag.mass-1).max()<1e-10,'Fresh NAICS6 midpoint reconstruction failed')
            parts.append(pd.DataFrame({'ccg':ag.index,'year':year,'y':ag.y.to_numpy(),'mass':ag.mass.to_numpy()}))
            # The original decomposition uses NumPy logarithms of continuing
            # annual endpoints, independently of the SQL log-FE cell cache.
            current=pd.read_parquet(source/f'y{year}.parquet',columns=['ccg','code','emp'])
            lagged=pd.read_parquet(source/f'y{year-3}.parquet',columns=['ccg','code','emp'])
            c=current.merge(lagged,on=['ccg','code'],suffixes=('_cur','_lag'),validate='one_to_one')
            c=c.loc[c.emp_cur.gt(0)&c.emp_lag.gt(0)].copy()
            c['lag_log']=c.emp_lag*(np.log(c.emp_cur)-np.log(c.emp_lag))
            u=c.groupby('ccg').agg(E_c=('emp_cur','sum'),E_b_c=('emp_lag','sum'),lag_log=('lag_log','sum'),count=('code','size')).reindex(g.ccg).fillna(0).reset_index()
            u['year']=year;decompositions.append(u)
            active+=len(d);d=d.drop(columns=['code','weighted_y'])
            save(d,app,f'cells/mpgr_iy_h3/y{year}.parquet',['ccg','industry'],record_dir)
            # Compare the same independently saved yearly partition if present.
            original=ROOT/f'data/cells/mpgr_iy_h3/y{year}.parquet'
            if original.exists():
                compare(d,f'cells/mpgr_iy_h3/y{year}.parquet',['ccg','industry'],record_dir,
                        baseline_lag=g.set_index('ccg').E_b)
            print('FRESH HORIZON CELLS',year,len(d),flush=True)
        stats=pd.concat(parts,ignore_index=True).sort_values(['ccg','year']).reset_index(drop=True)
        save(stats,app,'disaggregation/mpgr_naics6_h3.parquet',['ccg','year'],record_dir,'disaggregation/mpgr_naics6_h3.parquet')
        con.register('f',f[['ccg','year','E_b']])
        dest=app/'data/cells/log_iy_h3.parquet'
        sql="COPY (SELECT d.ccg,d.year,d.industry,d.fe_county,d.fe_industry_year,d.x,d.z,d.y_log AS y,d.lag/f.E_b AS w_lag FROM read_parquet(?) d JOIN f USING(ccg,year) WHERE d.y_log IS NOT NULL AND d.w_log>0) TO '"+dest.as_posix().replace("'","''")+"' (FORMAT PARQUET)"
        query(con,sql,[app/'data/cells/mpgr_iy_h3/y*.parquet'])
        log=pd.read_parquet(dest)
        compare(log,'cells/log_iy_h3.parquet',['ccg','year','industry'],record_dir)
        require(len(log)==22506999,'Fresh large-cell log support differs')
        del log;gc.collect()
    d=f.merge(pd.concat(decompositions,ignore_index=True),on=['ccg','year'],validate='one_to_one')
    keep=d['count'].gt(0);require(int((~keep).sum())==11,'Natural log sample differs')
    d['lag_mass']=d.E_b_c/d.E_b;d['mean_log']=d.lag_log/d.E_b_c.where(keep)
    d['entry']=-np.log(d.E_c.where(keep)/d.E)
    d['exit']=np.log(d.E_b_c.where(keep)/d.E_b)
    d['jensen']=np.log(d.E_c.where(keep)/d.E_b_c.where(keep))-d.mean_log
    require(abs(d.loc[keep,'log_y']-d.loc[keep,['entry','exit','jensen','mean_log']].sum(axis=1)).max()<1e-11,'Fresh log decomposition identity fails')
    save(d,app,'disaggregation/decomposition.parquet',['ccg','year'],record_dir,'disaggregation/decomposition.parquet')
    (app/'output/build').mkdir(parents=True,exist_ok=True)
    write(app/'output/build/checks.json',dict(status='PASS',county_years=110088,counties=3058,
          naics6_balanced_cells=int(support*f.year.nunique()),naics6_active_cells=active,
          boundary='fresh_raw_sources_in_initially_empty_run',prepared_data_construction_reads=0))


def aggregate(app,frames,record_dir):
    declarations=read(ROOT/'provenance/original_4a/config/aggregation.json')
    paths={}
    for name,record in declarations['sources'].items():
        path=ROOT/'provenance/historical_fc9df56'/record['source']
        require(sha(path)==record['sha256'],'Pinned geography changed');paths[name]=path
    x=pd.read_csv(paths['aggregation_county_ccg_crosswalk.csv'],dtype=str)
    x=x[x.ccg_id.ne('unplaceable')].copy();x['ccg']=x.ccg_id.astype(int)
    b=pd.read_csv(paths['aggregation_county_bea_area_2004.csv'],dtype=str)
    m=x.merge(b,on='county_fips5',how='left',validate='many_to_one',sort=False)
    cea=m.dropna(subset=['cea_code']).drop_duplicates('ccg').set_index('ccg').cea_code
    for rung,h in [('state',2),('state',3),('state',4),('cea',3)]:
        g=frames[h].copy();g['unit']=g.state if rung=='state' else g.ccg.map(cea)
        require(g.unit.notna().all(),'Unmapped fresh county')
        g['unit']=pd.factorize(g.unit,sort=True)[0]
        g['weight']=g.E_b/g.groupby(['unit','year']).E_b.transform('sum')
        g['std_y']=(g.E-g.E_b)/g.E_b
        for v in ('x','z','std_y'):g['w_'+v]=g.weight*g[v]
        a=g.groupby(['unit','year'],as_index=False).agg(E=('E','sum'),E_b=('E_b','sum'),x_agg=('w_x','sum'),z_agg=('w_z','sum'),reconstructed=('w_std_y','sum'),weight_sum=('weight','sum'))
        a['std_y']=(a.E-a.E_b)/a.E_b;a['log_y']=np.log(a.E/a.E_b);a['weight']=1.0
        require(abs(a.std_y-a.reconstructed).max()<1e-12 and abs(a.weight_sum-1).max()<1e-12,'Fresh regional reconstruction fails')
        g=g.merge(a[['unit','year','x_agg','z_agg']],on=['unit','year'],validate='many_to_one')
        for tag,frame,keys in [('county',g,['ccg','year']),('aggregate',a,['unit','year'])]:
            rel=f'aggregation/{tag}_{rung}_h{h}.parquet'
            save(frame,app,rel,keys,record_dir,rel)


def characteristics(run,app,record_dir):
    B=pd.read_parquet(run/'source51/source51_county_panel.parquet')
    emp=run/'1_current_paper/_data/temp/employment/naics4_alloc'
    byyear={}
    for year in (1978,2000):
        b=B.loc[B.year.eq(year)].sort_values('ccg').set_index('ccg')
        c=pd.read_parquet(emp/f'y{year}.parquet')
        # The archived source snapshot covers all 3,101 CCGs, not only the
        # 3,058 paper counties. Compare like-for-like before selecting them.
        compare(c,f'upstream/naics4_characteristics/y{year}.parquet',['ccg','code'],record_dir)
        c=c.loc[c.ccg.isin(b.index)].copy()
        total=c.groupby('ccg').emp.sum().reindex(b.index)
        require(np.allclose(total,b.E,rtol=1e-12,atol=1e-6),'Fresh characteristic employment conservation fails')
        v=pd.DataFrame(index=b.index)
        v['employment']=b.E;v['population']=b.P;v['jobs_per_resident']=b.E/b.P
        v['income_per_resident']=b.Y/b.P
        manu=c.naics.str[:2].isin(['31','32','33'])
        v['manufacturing']=c.loc[manu].groupby('ccg').emp.sum().reindex(b.index,fill_value=0)/b.E
        c['share']=c.emp/c.ccg.map(b.E)
        v['concentration']=(c.share**2).groupby(c.ccg).sum().reindex(b.index)
        v['income_original']=v.income_per_resident
        v['income_per_resident']=wins(v.income_per_resident.to_numpy(),np.zeros(len(v),dtype=int))
        byyear[year]=v
    chars=byyear[2000].copy()
    for key in ['employment','population','jobs_per_resident','income_per_resident','manufacturing','concentration']:
        v=chars[key];chars['C_'+key]=(v-v.mean())/v.std(ddof=0)
        old=byyear[1978][key]
        # The preserved sensitivity builder standardizes capped 1978 income
        # with ndarray reductions, while the 2000 builder uses Series. Keep
        # that original arithmetic, including its floating-point order.
        if key=='income_per_resident':
            values=old.to_numpy();chars['old_'+key]=(values-values.mean())/values.std(ddof=0)
        else:
            chars['old_'+key]=(old-old.mean())/old.std(ddof=0)
    save(chars.reset_index(),app,'heterogeneity/characteristics_reference.parquet',['ccg'],record_dir,'heterogeneity/characteristics_reference.parquet')
    c=pd.read_parquet(app/'data/aggregation/county_state_h3.parquet')
    c=c[['ccg','year','E','E_b','x','z','log_y','unit','weight','std_y','x_agg','z_agg']].copy()
    a=pd.read_parquet(app/'data/aggregation/aggregate_state_h3.parquet')
    for i,key in enumerate(['employment','income_per_resident','manufacturing'],1):
        c[f'C{i}']=c.ccg.map(chars['old_'+key]);c[f'xc{i}']=c.x*c[f'C{i}'];c[f'zc{i}']=c.z*c[f'C{i}']
        for value,target in [(f'C{i}',f'A{i}'),(f'xc{i}',f'Q{i}'),(f'zc{i}',f'R{i}')]:
            g=(c.weight*c[value]).groupby([c.unit,c.year]).sum().rename(target)
            c=c.merge(g.reset_index(),on=['unit','year'],how='left',validate='many_to_one',sort=False)
            a=a.merge(g.reset_index(),on=['unit','year'],how='left',validate='one_to_one',sort=False)
    for tag,frame,keys in [('county',c,['ccg','year']),('state',a,['unit','year'])]:
        save(frame,app,f'heterogeneity/{tag}.parquet',keys,record_dir,f'heterogeneity/{tag}.parquet')


def resume_annual(run,app,record_dir,comparisons_dir,source_attempt):
    """Resume only fresh annual outputs of this SAME initially empty run."""
    source=run/('application' if source_attempt==1 else 'application_attempt'+str(source_attempt))
    label='application'+('_attempt'+str(source_attempt) if source_attempt>1 else '')
    require(read(record_dir/(label+'.json'))['status']=='FAILED','Only a recorded failed same-run attempt may supply completed annual files')
    prior=record_dir/('application_inputs' if source_attempt==1 else 'application_inputs_attempt'+str(source_attempt))
    records={v['file']:v for v in read(prior/'input_comparisons.json')['files']}
    current=Path(__file__).read_text(encoding='utf-8')
    previous=(source/'code/raw_application.py').read_text(encoding='utf-8')
    def function(text,name):
        node=next(v for v in ast.parse(text).body if isinstance(v,ast.FunctionDef) and v.name==name)
        return ast.get_source_segment(text,node)
    require(function(current,'six_digit_annual')==function(previous,'six_digit_annual'),
            'Annual construction arithmetic changed; cannot resume prior annual outputs')
    original_function('provenance/original_4a/code/build/naics6_cells.py','split_parent')
    reused=[]
    for year in range(1978,2017):
        relative=f'upstream/naics6_annual/y{year}.parquet'
        require(relative in records and records[relative]['status']=='PASS','Annual file was not independently verified')
        require(records[relative]['original_sha256']==checked_original(ROOT/'data'/relative),'Original annual anchor changed')
        old=source/'data'/relative;dest=app/'data'/relative
        require(not dest.exists(),'Do not replace an existing new annual input')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dest)
        digest=sha(old);require(sha(dest)==digest,'Same-run resume copy changed')
        compare(pd.read_parquet(dest),relative,['ccg','code'],comparisons_dir)
        reused.append(dict(year=year,source=old.relative_to(ROOT).as_posix(),sha256=digest))
    write(comparisons_dir/'same_run_annual_resume.json',dict(status='PASS',source_attempt=source_attempt,
          original_prepared_data_used=False,arithmetic_unchanged=True,files=reused))


def build(run,record_dir,attempt=1,resume_annual_attempt=None):
    global ACTIVE_RUN
    ACTIVE_RUN=run
    require(read(record_dir/'annual_complete.json')['status']=='PASS','Fresh annual build incomplete')
    require(duckdb.__version__=='1.4.1','Use the pinned DuckDB query engine')
    reproduce.verify()
    app=run/('application' if attempt==1 else 'application_attempt'+str(attempt))
    comparisons_dir=record_dir/('application_inputs' if attempt==1 else 'application_inputs_attempt'+str(attempt))
    require(not comparisons_dir.exists(),'Comparison attempt already exists; retain it and choose an explicit new attempt')
    require(not (record_dir/'application_selected.json').exists(),'A successful application has already been selected')
    require(not app.exists(),'Refusing to overwrite an existing fresh application')
    app.mkdir()
    comparisons_dir.mkdir()
    for directory in ('code','config','reference','provenance'):
        copy_source_tree(ROOT/directory,app/directory)
    rule_files=['provenance/original_4a/code/build/'+name+'.py' for name in (
        'disaggregation','lag_weighted_log','naics6_cells','table2_cells','tables45_weights')]
    rule_files+=['provenance/original_sandbox/'+name+'.py' for name in (
        'h3_analysis','balance_tests','balance_sensitivity','balance_reference_2000','heterogeneity_levels')]
    original_rules={name:checked_original(ROOT/name) for name in rule_files}
    with construction_read_guard():
        frames=counties(run,app,comparisons_dir)
        smaller_industries(run,app,frames,comparisons_dir)
        if resume_annual_attempt is None:
            six_digit_annual(run,app,frames,comparisons_dir)
        else:
            require(1<=resume_annual_attempt<attempt,'Resume source must be an earlier attempt in this run')
            resume_annual(run,app,record_dir,comparisons_dir,resume_annual_attempt)
        six_digit_cells(run,app,frames,comparisons_dir)
        aggregate(app,frames,comparisons_dir)
        characteristics(run,app,comparisons_dir)
    inputs={str(p.relative_to(app)):sha(p) for p in (app/'data').rglob('*.parquet')}
    write(comparisons_dir/'fresh_application_manifest.json',dict(status='PASS',files=inputs,unchanged_extracted_functions=FUNCTIONS,
          builder_sha256=sha(__file__),construction_logic_sha256=construction_logic_sha256(),
          original_construction_rules=original_rules,
          prepared_data_reads_for_construction=0,independent_reference_reads_only_after_construction=True,raw_to_results_certified=False))
    write(comparisons_dir/'input_comparisons.json',dict(status='PASS',files=COMPARISONS))
    write(record_dir/'application_selected.json',dict(status='PASS',attempt=attempt,application=app.relative_to(ROOT).as_posix(),
          records=str(comparisons_dir.resolve())))
    print('FRESH APPLICATION INPUT BUILD PASS',len(inputs),'files',flush=True)


def check_inputs(run,record_dir):
    selected=read(record_dir/'application_selected.json');require(selected['status']=='PASS','No successful application selected')
    app=ROOT/selected['application'];require(run.resolve() in app.resolve().parents,'Selected application escapes its run')
    manifest=read(ROOT/selected['records']/'fresh_application_manifest.json');require(manifest['status']=='PASS','Fresh application uncertified')
    require(manifest['construction_logic_sha256']==construction_logic_sha256(),'Application construction logic changed after successful construction')
    for name,digest in manifest['files'].items():
        require(sha(app/name)==digest,'Fresh input changed: '+name)
    return app


def estimate(run,record_dir):
    app=check_inputs(run,record_dir)
    reproduce.verify();reproduce.DATA=app/'data';reproduce.OUT=app/'output'
    with construction_read_guard():
        for stage in (reproduce.small,reproduce.cells,reproduce.aggregation,reproduce.heterogeneity,reproduce.balance,reproduce.check):
            print('FRESH ESTIMATION STAGE',stage.__name__,flush=True);stage()
    print('ALL FRESH ESTIMATES MATCH INDEPENDENT REFERENCES',flush=True)


def all_reported_numeric_fields(app,record_dir):
    cases=[('aggregation_summary_tables/selected_results.csv','aggregation_selected.csv',['spec']),
           ('heterogeneity_separate/coefficients.csv','heterogeneity_coefficients.csv',['controls','model','variable']),
           ('balance_reference_2000/balance.csv','balance.csv',['year','characteristic']),
           ('h3_analysis/decomposition.csv','decomposition.csv',['component'])]
    comparisons=[]
    # The old Table 5 display file saves t, rather than p. Apply its original
    # t_3057 reporting rule to the independently saved t values; do not derive
    # the reference category from the newly fitted coefficient or SE.
    from scipy.stats import t as student_t
    actual_dis=pd.read_csv(app/'output/disaggregation_no_population/displayed_estimates.csv').set_index('column')
    expected_dis=pd.read_csv(ROOT/'reference/displayed_estimates.csv').query('table==5 and horizon==3').set_index('column')
    require(set(actual_dis.index)==set(expected_dis.index)==set(range(1,11)),'Incomplete independent disaggregation coverage')
    for column in expected_dis.index:
        for field in ('beta','se','F','t','county_years','percent_deviation'):
            a,b=actual_dis.loc[column,field],expected_dis.loc[column,field]
            reproduce.close(a,b,'Full disaggregation comparison '+str(column)+' '+field)
            comparisons.append(dict(reference='displayed_estimates.csv',key=str(column),field=field,
                                    absolute_difference=float(abs(a-b))))
        a=actual_dis.loc[column,'pvalue']
        b=2*student_t.sf(abs(expected_dis.loc[column,'t']),3057)
        reproduce.close(a,b,'Independent disaggregation reporting p '+str(column))
        require(all((a<cut)==(b<cut) for cut in (.01,.05,.1)),
                'QUALITATIVE SIGNIFICANCE CHANGE: disaggregation '+str(column))
        comparisons.append(dict(reference='displayed_estimates.csv',key=str(column),field='pvalue_from_saved_t',
                                absolute_difference=float(abs(a-b))))
    for actual_name,reference_name,keys in cases:
        actual=pd.read_csv(app/'output'/actual_name).set_index(keys)
        expected=pd.read_csv(ROOT/'reference'/reference_name).set_index(keys)
        require(actual.index.is_unique and expected.index.is_unique,'Duplicate result keys')
        require(set(actual.index)==set(expected.index),'Independent result coverage differs: '+reference_name)
        for index in actual.index:
            for column in actual.columns.intersection(expected.columns):
                a,b=actual.loc[index,column],expected.loc[index,column]
                if isinstance(a,(int,float,np.number)) and isinstance(b,(int,float,np.number)):
                    reproduce.close(a,b,'Full result comparison '+reference_name+' '+str(index)+' '+column)
                    # In addition to numerical tolerance, independently check
                    # the reported significance category whenever p is saved.
                    if column in ('p','gap_p') or column.endswith('_p'):
                        require(all((a<cut)==(b<cut) for cut in (.01,.05,.1)),
                                'QUALITATIVE SIGNIFICANCE CHANGE: '+reference_name+' '+str(index)+' '+column)
                    comparisons.append(dict(reference=reference_name,key=str(index),field=column,
                        absolute_difference=None if pd.isna(a) or pd.isna(b) else float(abs(a-b))))
    write(record_dir/'all_reported_numeric_fields.json',dict(status='PASS',comparisons=len(comparisons),
          all_saved_common_numeric_fields_checked=True,reported_significance_categories_unchanged=True,checks=comparisons))


def render(run,record_dir):
    app=check_inputs(run,record_dir)
    require(read(app/'output/replication_check.json')['status']=='PASS','Numerical verification incomplete')
    command=[sys.executable,'-B',str(app/'code/render_results.py')]
    subprocess.run(command,check=True,cwd=app)
    stata=os.environ.get('STATA_BIN') or shutil.which('stata-mp') or shutil.which('stata')
    if not stata and Path('C:/Program Files/Stata17/StataMP-64.exe').exists():
        stata='C:/Program Files/Stata17/StataMP-64.exe'
    require(stata,'Set STATA_BIN for the two figures')
    kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
    subprocess.run([stata,'/e' if os.name=='nt' else '-b','do',str(app/'code/graph_results.do'),str(app),str(record_dir)],check=True,**kwargs)
    require('STATA_RECAP_GRAPHICS_PASS' in (record_dir/'graphics_detail.log').read_text(errors='replace'),'Fresh Stata figure rendering failed')
    latex=record_dir/'latex';latex.mkdir(exist_ok=True)
    for i in range(2):
        with (record_dir/f'pdflatex_{i+1}.log').open('w') as stream:
            subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(latex),'NS_results_recap_v3.tex'],
                           cwd=app/'source/results_recap_v3',stdout=stream,stderr=subprocess.STDOUT,check=True)
    out=app/'output/pdf';out.mkdir(exist_ok=True)
    shutil.copy2(latex/'NS_results_recap_v3.pdf',out/'NS_results_recap.pdf')


def final_check(run,record_dir):
    app=check_inputs(run,record_dir)
    selected=read(record_dir/'application_selected.json')
    application_stage='application'+('_attempt'+str(selected['attempt']) if selected['attempt']>1 else '')
    for stage in (application_stage,'estimate','render'):
        require(read(record_dir/(stage+'.json'))['status']=='PASS','Incomplete stage: '+stage)
    require(read(record_dir/'annual_complete.json')['status']=='PASS','Fresh source build incomplete')
    require(read(record_dir/'sources.json')['status']=='PASS','Raw source acquisition/verification incomplete')
    result=read(app/'output/replication_check.json');require(result['status']=='PASS','Numerical check failed')
    all_reported_numeric_fields(app,record_dir)
    reproduce.verify()
    write(record_dir/'full_verification.json',dict(status='PASS',raw_to_results_certified=True,
          initial_derived_tree_empty=read(run/'run.json')['initial_derived_tree_empty'],
          raw_boundary='Exact checksum-verified external source archives; published harmonized EFSY panel is an external source, not a rerun of its Census imputation',
          sources_and_application_rebuilt_inside_this_run=True,prepared_analysis_cache_construction_reads=0,
          numerical_summary=result,input_comparisons=len(read(ROOT/selected['records']/'input_comparisons.json')['files']),
          original_population_audit_status='FAILED',application_exception='Approved excluded Kalawao warning only',
          independent_references_unchanged=True,pdf=str((app/'output/pdf/NS_results_recap.pdf').relative_to(ROOT)),
          pdf_sha256=sha(app/'output/pdf/NS_results_recap.pdf'),public_upload_performed=False))
    print('FULL RAW-TO-RESULTS VERIFICATION PASS:',app/'output/pdf/NS_results_recap.pdf',flush=True)
