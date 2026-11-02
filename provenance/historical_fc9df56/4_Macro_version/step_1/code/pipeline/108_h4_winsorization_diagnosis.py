"""County h=4 EW second-stage comparison and clipped-county inventory.

Input: corrected annual positive nonconstruction spending, four-character categories.
Exposure: five-year county market shares over t-13,...,t-9. Shifter: mean
national category spending t-3,...,t minus t-4. Treatment uses the same spending
window divided by population t-4. Outcome: employment MPGR t-4 to t. All county
histories 1982-2016 are retained in the full-sample comparison.

Protocol scope: county is the sole unit, no industry split or aggregation test;
no cross-level identities are asserted. Treatment varies by county-year in both
fits. Division-year FE do not absorb it. Optional weights are one. Capping occurs
once on the full county-year distribution before FE. Deletions are diagnostics,
not production sample rules. Existing exhibits and h=2 results remain untouched.
"""
# mpgr-run-class: cache_rebuild (--build); estimate_only (--estimate, --single-leaveouts)
# mpgr-rebuild-reason: required_key_absent
# mpgr-cache: report_correction_rebuild annual spending/category inputs
# mpgr-output-cache: h4_winsorization_diagnosis/analysis.parquet
# mpgr-new-estimates: 2 (--estimate)
# mpgr-single-leaveouts-new-estimates: 2172 (1086 clipped counties, raw and winsorized)
# mpgr-single-leaveouts-cache: h4_winsorization_diagnosis/analysis.parquet
# mpgr-joint-leaveouts-run-class: estimate_only
# mpgr-joint-leaveouts-new-estimates: 10 (top 2,5,10,20,50; raw and winsorized)
# mpgr-joint-leaveouts-cache: h4_winsorization_diagnosis/analysis.parquet
# mpgr-selective-caps-run-class: estimate_only
# mpgr-selective-caps-new-estimates: 1091 (1086 individual caps; top 2,5,10,20,50 joint caps)
# mpgr-selective-caps-cache: h4_winsorization_diagnosis/analysis.parquet
# mpgr-cell-caps-run-class: estimate_only
# mpgr-cell-caps-new-estimates: 5597 single-cell fits plus selected ranked-prefix fits
# mpgr-cell-caps-cache: h4_winsorization_diagnosis/analysis.parquet
# mpgr-precision-caps-run-class: estimate_only
# mpgr-precision-caps-new-estimates: 5390 (5385 singles, five additional joint groups; top one reused)
# mpgr-precision-caps-cache: h4_winsorization_diagnosis/analysis.parquet
# mpgr-joint-target-run-class: estimate_only
# mpgr-joint-target-new-estimates: at most 5595 cumulative-prefix fits; saved singles and endpoints reused
# mpgr-joint-target-cache: h4_winsorization_diagnosis/analysis.parquet
# mpgr-adaptive-target-run-class: estimate_only
# mpgr-adaptive-target-new-estimates: at most 500 selected-prefix fits; exact candidate moment screening
# mpgr-adaptive-target-cache: h4_winsorization_diagnosis/analysis.parquet
# mpgr-backward-target-run-class: estimate_only
# mpgr-backward-target-cache: h4_winsorization_diagnosis/analysis.parquet
# mpgr-backward-target-scope: full winsorization; restore raw batches and singles; coefficient and SE within 5 percent
# mpgr-batch-approved: explicit
# mpgr-variation-scope: county|h4|EW|full-1982-2016|raw-pooled1-99|county-omissions
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PIPELINE=Path(__file__).resolve().parent
ROOT=PIPELINE.parents[2]
sys.path[:0]=[str(PIPELINE.parent/'py'),str(PIPELINE),str(PIPELINE/'lib')]
import dcads_common as dc
import store
SOURCE=dc.MACRO_VERSION_TEMP_DIR/'report_correction_rebuild'
CACHE=dc.MACRO_VERSION_TEMP_DIR/'h4_winsorization_diagnosis'
OUT=dc.MACRO_RESULTS_ROOT/'instrument_normalization/h4_winsorization_diagnosis'
YEARS=list(range(1982,2017))
VARIANTS={'raw':['y','x','z'],'winsorized':['yc','xc','zc']}

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build():
    assert json.loads((SOURCE/'certification.json').read_text())['status']=='PASS'
    if (CACHE/'analysis.parquet').exists():
        raise RuntimeError('Cache exists; use --estimate, not --build')
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    old=pd.read_parquet(SOURCE/'analysis_pooled.parquet')
    units=np.sort(old.ccg.unique());assert len(units)==3058
    county=store.load_county_panel()
    units_check,piv=store.load_pivots(county,cols=('E','P'))
    np.testing.assert_array_equal(units,units_check)
    spending=pd.read_parquet(SOURCE/'county_spending_positive.parquet').pivot(index='ccg',columns='year',values='dollars').reindex(units)
    p100=module('p100_h4_sparse',PIPELINE/'100_h2_full_instrument_design_grid.py')
    p100.path_for=lambda cover,rule,floor,year:SOURCE/'annual_category'/f'fy={year}.parquet'
    cats,nat,mats=p100.load_sparse('exclude_construction','positive',0,4,units)
    # h=2 reproduction checks the reused reader against the certified instrument.
    z2,_=p100.instruments((cats,nat,mats),(cats,nat,mats),piv['P'],'rolling',5)
    np.testing.assert_allclose(z2.ravel(),old.z,rtol=1e-10,atol=1e-8)
    frames=[];diagnostics=[]
    labels=old.drop_duplicates('ccg').set_index('ccg').reindex(units)
    for t in YEARS:
        base=t-4;lo=base-9;hi=base-5
        assert hi-lo+1==5 and lo>=1966
        e0=piv['E'][base].to_numpy();e1=piv['E'][t].to_numpy();pop=piv['P'][base].to_numpy()
        post=spending[list(range(base+1,t+1))].mean(axis=1).to_numpy()
        g0=spending[base].to_numpy()
        y=2*(e1-e0)/(e1+e0);x=(post-g0)/pop
        exposure=sum(mats[v] for v in range(lo,hi+1))/5
        den=nat[:,lo-1966:hi-1966+1].mean(axis=1)
        shift=nat[:,base+1-1966:t-1966+1].mean(axis=1)-nat[:,base-1966]
        scaled=np.divide(shift,den,out=np.zeros_like(shift),where=den>0)
        z=np.asarray(exposure@scaled).ravel()/pop
        assert np.isfinite(np.column_stack([x,y,z,pop])).all() and (pop>0).all()
        np.testing.assert_allclose(den[den>0]*scaled[den>0],shift[den>0],rtol=1e-12,atol=1e-6)
        # Direct county-category dot product verifies selected sparse matrix rows.
        for i in (0,len(units)//2,len(units)-1):
            shares=exposure.getrow(i).toarray().ravel()
            direct=np.sum(np.divide(shares,den,out=np.zeros_like(den),where=den>0)*shift)/pop[i]
            np.testing.assert_allclose(direct,z[i],rtol=1e-10,atol=1e-10)
        frames.append(pd.DataFrame(dict(ccg=units,year=t,x=x,y=y,z=z,
            population_base=pop,employment_base=e0,employment_end=e1,
            spending_base=g0,spending_post_mean=post,division=labels.division.to_numpy())))
        diagnostics.append(dict(year=t,base_year=base,post_start=base+1,post_end=t,
            exposure_start=lo,exposure_end=hi,n_categories=len(cats),live_categories=int((den>0).sum())))
    a=pd.concat(frames,ignore_index=True).sort_values(['ccg','year']).reset_index(drop=True)
    a['fe']=a.groupby(['division','year'],sort=False).ngroup().astype('int32')
    a['cluster']=pd.factorize(a.ccg)[0].astype('int32')
    assert len(a)==107030 and not a.duplicated(['ccg','year']).any()
    assert a.groupby('ccg').size().eq(35).all()
    cuts=[]
    for v in ('x','y','z'):
        low,high=a[v].quantile([.01,.99],interpolation='linear')
        a[v+'c']=a[v].clip(low,high);a[v+'_clipped']=a[v].ne(a[v+'c'])
        cuts.append(dict(variable=v,lower=low,upper=high,n_low=int((a[v]<low).sum()),
            n_high=int((a[v]>high).sum()),n_clipped=int(a[v+'_clipped'].sum()),
            counties_clipped=int(a.loc[a[v+'_clipped'],'ccg'].nunique())))
    a['any_clipped']=a[['x_clipped','y_clipped','z_clipped']].any(axis=1)
    names=pd.read_csv(dc.CCG_CROSSWALK)
    names['ccg']=pd.to_numeric(names.ccg_id,errors='coerce')
    names=names.dropna(subset=['ccg']).drop_duplicates('ccg').set_index('ccg')
    a['county']=a.ccg.map(names.ccg_name)
    counts=a.groupby('ccg')[['x_clipped','y_clipped','z_clipped','any_clipped']].sum().reset_index()
    counts['county']=counts.ccg.map(names.ccg_name)
    state=module('p102_h4_names',PIPELINE/'102_h2_winsorization_cell_audit.py').STATE_NAMES
    counts['state']= (counts.ccg//1000).map(state)
    counts.to_csv(OUT/'clipped_counties.csv',index=False)
    a.loc[a.any_clipped].to_csv(OUT/'clipped_cells.csv',index=False)
    pd.DataFrame(cuts).to_csv(OUT/'cutoffs.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'construction_windows.csv',index=False)
    a.to_parquet(CACHE/'analysis.parquet',index=False)
    cert=dict(status='PASS',horizon=4,n=len(a),n_counties=int(a.ccg.nunique()),
        years=YEARS,n_clipped_cells=int(a.any_clipped.sum()),
        n_clipped_counties=int(a.loc[a.any_clipped,'ccg'].nunique()),
        cache_sha256=sha(CACHE/'analysis.parquet'),source_certificate_sha256=sha(SOURCE/'certification.json'),
        source_h2_instrument_reproduced=True,scope='EW; division-year FE; county clusters; full sample; pooled unweighted 1/99')
    (CACHE/'certification.json').write_text(json.dumps(cert,indent=2)+'\n')
    print(json.dumps(cert,indent=2),flush=True);print(pd.DataFrame(cuts).to_string(index=False),flush=True)

def load():
    cert=json.loads((CACHE/'certification.json').read_text())
    assert cert['status']=='PASS' and cert['cache_sha256']==sha(CACHE/'analysis.parquet')
    a=pd.read_parquet(CACHE/'analysis.parquet')
    t28=module('t28_h4_diagnosis',ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER=pf.MapDemeaner(fixef_tol=1e-12,fixef_maxiter=100000)
    return a,t28

def fits(a,t28,excluded,label):
    p=a.loc[~a.ccg.isin(excluded)]
    assert len(p)==len(a)-35*len(excluded) and p.groupby('ccg').size().eq(35).all()
    rows=[]
    for variant,cols in VARIANTS.items():
        y,x,z=p[cols].to_numpy(np.float64).T
        fit=t28.fit2sls(y,x,z,np.ones(len(p)),p.fe.to_numpy(np.int64),np.zeros(len(p),np.int64),
            p.cluster.to_numpy(np.int64),singleton=False,want_F=True)
        assert fit['n_dropped']==0 and fit['n']==len(p)
        assert np.isfinite([fit['beta'],fit['se'],fit['F']]).all()
        rows.append(dict(sample=label,variant=variant,excluded=';'.join(map(str,excluded)),n=len(p),
            counties=int(p.ccg.nunique()),beta=fit['beta']*1e5,se=fit['se']*1e5,
            t=fit['beta']/fit['se'],F=fit['F'],horizon=4,weight='EW'))
    return rows

def estimate():
    a,t28=load();path=OUT/'baseline.csv'
    if path.exists():
        print(pd.read_csv(path).to_string(index=False));return
    rows=fits(a,t28,[],'Full sample')
    for row in rows:
        cols=VARIANTS[row['variant']]
        d=a[cols]-a.groupby('fe')[cols].transform('mean')
        y,x,z=d.to_numpy().T
        zx=np.dot(z,x);beta=np.dot(z,y)/zx
        n=len(a);g=a.cluster.nunique();k=a.fe.nunique()+1
        correction=g/(g-1)*(n-1)/(n-k)
        score=pd.Series(z*(y-beta*x)).groupby(a.cluster).sum()
        se=np.sqrt(correction*np.dot(score,score)/zx**2)
        fs=zx/np.dot(z,z)
        scorefs=pd.Series(z*(x-fs*z)).groupby(a.cluster).sum()
        sefs=np.sqrt(correction*np.dot(scorefs,scorefs)/np.dot(z,z)**2)
        np.testing.assert_allclose([beta*1e5,se*1e5,beta/se,(fs/sefs)**2],
            [row['beta'],row['se'],row['t'],row['F']],rtol=1e-10)
    pd.DataFrame(rows).to_csv(path,index=False)
    print(pd.read_csv(path).to_string(index=False),flush=True)

def single_leaveouts():
    """Rank clipped counties by actual absolute second-stage gap reduction."""
    import time
    started=time.monotonic()
    a,t28=load()
    ids=sorted(a.loc[a.any_clipped,'ccg'].unique())
    assert len(ids)==1086 and a.groupby('ccg').size().eq(35).all()
    names=pd.read_csv(OUT/'clipped_counties.csv').set_index('ccg')
    assert set(names.index[names.any_clipped.gt(0)])==set(ids)
    baseline=pd.read_csv(OUT/'baseline.csv').set_index('variant')
    full_gap=float(baseline.loc['winsorized','beta']-baseline.loc['raw','beta'])
    assert full_gap>0 and baseline.n.eq(len(a)).all()
    output=OUT/'single_county_estimates.csv'
    rows=pd.read_csv(output,keep_default_na=False).to_dict('records') if output.exists() else []
    keys={(int(r['excluded']),r['variant']) for r in rows}
    assert len(keys)==len(rows) and all(c in ids and v in VARIANTS for c,v in keys)
    # Independent EW within-FE covariance formula verifies each saved coefficient.
    fe=a.fe.to_numpy(np.int64);nf=int(fe.max())+1
    counts=np.bincount(fe,minlength=nf)
    totals={}
    for v,cols in VARIANTS.items():
        y,x,z=a[cols].to_numpy(np.float64).T
        values=np.column_stack([y,x,z,z*y,z*x])
        sums=np.array([np.bincount(fe,weights=values[:,j],minlength=nf) for j in range(5)])
        totals[v]=(values,sums)
    computed=0
    for i,ccg in enumerate(ids,1):
        present=[(int(ccg),v) in keys for v in VARIANTS]
        assert all(present) or not any(present), 'Incomplete county pair in checkpoint'
        if not all(present):
            result=fits(a,t28,[int(ccg)],str(names.loc[ccg,'county']))
            rows.extend(result)
            keys.update((int(ccg),r['variant']) for r in result)
            computed+=2
        pair=rows[-2:] if not all(present) else [r for r in rows if int(r['excluded'])==ccg]
        removed=a.ccg.eq(ccg).to_numpy()
        remain_n=counts-np.bincount(fe[removed],minlength=nf)
        assert (remain_n>0).all()
        for r in pair:
            values,sums=totals[r['variant']]
            remain=sums-np.array([np.bincount(fe[removed],weights=values[removed,j],minlength=nf) for j in range(5)])
            sy,sx,sz,szy,szx=remain
            beta=(szy.sum()-(sz*sy/remain_n).sum())/(szx.sum()-(sz*sx/remain_n).sum())
            np.testing.assert_allclose(beta*1e5,r['beta'],rtol=1e-9,atol=1e-10)
            assert r['n']==len(a)-35 and r['counties']==3057
        if i%50==0 or i==len(ids):
            pd.DataFrame(rows).to_csv(output,index=False)
            print(f'Counties completed {i}/{len(ids)}; new fits {computed}; seconds {time.monotonic()-started:.1f}',flush=True)
    result=pd.read_csv(output)
    assert len(result)==2172 and not result.duplicated(['excluded','variant']).any()
    assert np.isfinite(result[['beta','se','t','F']]).all().all()
    rank=result.pivot(index='excluded',columns='variant',values=['beta','se','t','F'])
    rank.columns=['_'.join(c) for c in rank.columns]
    rank.index.name='ccg';rank=rank.reset_index()
    rank=rank.merge(names.reset_index(),on='ccg',validate='one_to_one')
    rank['full_gap']=full_gap
    rank['gap']=rank.beta_winsorized-rank.beta_raw
    rank['gap_reduction']=abs(full_gap)-rank.gap.abs()
    rank['gap_reduction_percent']=100*rank.gap_reduction/abs(full_gap)
    rank['gap_sign_reversed']=rank.gap.lt(0)
    rank=rank.sort_values(['gap_reduction','ccg'],ascending=[False,True]).reset_index(drop=True)
    rank.insert(0,'rank',np.arange(1,len(rank)+1))
    rank['closes_gap']=rank.gap_reduction.gt(1e-10)
    rank.to_csv(OUT/'single_county_ranking.csv',index=False)
    rank.loc[rank.closes_gap].to_csv(OUT/'gap_reducing_counties.csv',index=False)
    summary=dict(status='PASS',county_candidates=len(ids),coefficient_rows=len(result),new_fits=computed,
        gap_reducing_counties=int(rank.closes_gap.sum()),full_gap=full_gap,
        cutoffs_fixed=True,retained_instruments_fixed=True,no_unintended_drops=True,
        all_coefficients_checked_by_independent_within_moments=True,joint_removals_run=False,
        analysis_sha256=sha(CACHE/'analysis.parquet'))
    (OUT/'single_county_check.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print(rank.head(10)[['rank','county','state','gap_reduction_percent','beta_raw','t_raw','beta_winsorized','t_winsorized']].to_string(index=False),flush=True)

def joint_leaveouts():
    """Remove nested groups from a frozen individual gap-reduction ranking."""
    a,t28=load()
    rankpath=OUT/'single_county_ranking.csv'
    ranking=pd.read_csv(rankpath).sort_values('rank')
    sizes=(2,5,10,20,50)
    assert ranking.head(50).closes_gap.all() and ranking.head(50).ccg.nunique()==50
    assert set(ranking.ccg)==set(a.loc[a.any_clipped,'ccg'])
    base=pd.read_csv(OUT/'baseline.csv',keep_default_na=False)
    fullgap=float(base.set_index('variant').loc['winsorized','beta']-base.set_index('variant').loc['raw','beta'])
    path=OUT/'joint_county_estimates.csv'
    rows=pd.read_csv(path,keep_default_na=False).to_dict('records') if path.exists() else []
    keys={(int(r['n_excluded']),r['variant']) for r in rows}
    assert len(keys)==len(rows)
    if not rows:
        rows=base.assign(n_excluded=0,estimate_source='saved').to_dict('records')
        keys={(0,v) for v in VARIANTS}
    membership=[];new_fits=0
    for k in sizes:
        members=ranking.head(k)
        removed=members.ccg.astype(int).tolist()
        membership.append(members[['rank','ccg','county','state']].assign(group_size=k))
        present=[(k,v) in keys for v in VARIANTS]
        assert all(present) or not any(present)
        if not all(present):
            pair=fits(a,t28,removed,f'Top {k}')
            for r in pair:r.update(n_excluded=k,estimate_source='estimated')
            rows.extend(pair);keys.update((k,v) for v in VARIANTS);new_fits+=2
        pair=[r for r in rows if int(r['n_excluded'])==k]
        p=a.loc[~a.ccg.isin(removed)].reset_index(drop=True)
        assert len(p)==107030-35*k and p.ccg.nunique()==3058-k
        for r in pair:
            assert set(map(int,str(r['excluded']).split(';')))==set(removed)
            cols=VARIANTS[r['variant']]
            y,x,z=(p[cols]-p.groupby('fe')[cols].transform('mean')).to_numpy().T
            zx=z@x;b=(z@y)/zx;n=len(p);g=p.cluster.nunique();df=p.fe.nunique()+1
            ssc=g/(g-1)*(n-1)/(n-df)
            score=pd.Series(z*(y-b*x)).groupby(p.cluster).sum()
            se=np.sqrt(ssc*(score@score)/zx**2)
            fs=zx/(z@z);scorefs=pd.Series(z*(x-fs*z)).groupby(p.cluster).sum()
            F=fs**2/(ssc*(scorefs@scorefs)/(z@z)**2)
            np.testing.assert_allclose([b*1e5,se*1e5,b/se,F],[r['beta'],r['se'],r['t'],r['F']],rtol=1e-9)
        pd.DataFrame(rows).to_csv(path,index=False)
    e=pd.read_csv(path)
    assert len(e)==12 and not e.duplicated(['n_excluded','variant']).any()
    summary=e.pivot(index='n_excluded',columns='variant',values=['beta','se','t','F'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index()
    summary['gap']=summary.beta_winsorized-summary.beta_raw
    summary['full_gap']=fullgap
    summary['gap_reduction_percent']=100*(1-summary.gap.abs()/abs(fullgap))
    summary['gap_sign_reversed']=np.sign(summary.gap)!=np.sign(fullgap)
    summary.to_csv(OUT/'joint_county_gaps.csv',index=False)
    pd.concat(membership,ignore_index=True).to_csv(OUT/'joint_county_membership.csv',index=False)
    check=dict(status='PASS',new_fits=new_fits,group_sizes=list(sizes),ranking_sha256=sha(rankpath),
        analysis_sha256=sha(CACHE/'analysis.parquet'),cutoffs_fixed=True,ranking_fixed=True,
        no_unintended_drops=True,coefficients_se_t_F_checked=True,addback_tests_run=False)
    (OUT/'joint_county_check.json').write_text(json.dumps(check,indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)
    print(json.dumps(check,indent=2),flush=True)

def selective_caps():
    """Cap selected counties only, retaining all observations and fixed cutoffs."""
    import time
    started=time.monotonic()
    a,t28=load()
    ids=sorted(a.loc[a.any_clipped,'ccg'].unique())
    assert len(ids)==1086 and len(a)==107030 and a.ccg.nunique()==3058
    cache_hash=sha(CACHE/'analysis.parquet')
    base=pd.read_csv(OUT/'baseline.csv').set_index('variant')
    b0=float(base.loc['raw','beta']);bw=float(base.loc['winsorized','beta'])
    full_change=bw-b0
    assert full_change>0
    raw=a[['y','x','z']].to_numpy(np.float64)
    capped=a[['yc','xc','zc']].to_numpy(np.float64)
    fe=a.fe.to_numpy(np.int64);cl=a.cluster.to_numpy(np.int64)
    county=a.ccg.to_numpy();nf=int(fe.max())+1
    counts=np.bincount(fe,minlength=nf)
    weights=np.ones(len(a));nofe=np.zeros(len(a),np.int64)
    names=pd.read_csv(OUT/'clipped_counties.csv').set_index('ccg')
    def fit_selected(selected):
        mask=np.isin(county,selected)
        values=raw.copy();values[mask]=capped[mask]
        assert np.array_equal(values[~mask],raw[~mask])
        assert int(mask.sum())==35*len(selected)
        y,x,z=values.T
        fit=t28.fit2sls(y,x,z,weights,fe,nofe,cl,singleton=False,want_F=True)
        assert fit['n']==len(a) and fit['n_dropped']==0
        sy,sx,sz=(np.bincount(fe,weights=v,minlength=nf) for v in (y,x,z))
        direct=((z@y)-(sz*sy/counts).sum())/((z@x)-(sz*sx/counts).sum())
        np.testing.assert_allclose(direct,fit['beta'],rtol=1e-9,atol=1e-13)
        changes=(values!=raw)
        return dict(selected=';'.join(map(str,selected)),n_selected=len(selected),n=fit['n'],counties=3058,
            beta=fit['beta']*1e5,se=fit['se']*1e5,t=fit['beta']/fit['se'],F=fit['F'],
            cells_changed=int(changes.any(axis=1).sum()),y_changed=int(changes[:,0].sum()),
            x_changed=int(changes[:,1].sum()),z_changed=int(changes[:,2].sum()),analysis_sha256=cache_hash)
    path=OUT/'selective_cap_single_estimates.csv'
    rows=pd.read_csv(path,keep_default_na=False).to_dict('records') if path.exists() else []
    keys={int(r['ccg']) for r in rows}
    assert len(keys)==len(rows) and keys.issubset(set(ids))
    assert all(r['analysis_sha256']==cache_hash and r['n']==len(a) for r in rows)
    new_fits=0
    for i,c in enumerate(ids,1):
        if int(c) not in keys:
            row=fit_selected([int(c)])
            row.update(ccg=int(c),county=names.loc[c,'county'],state=names.loc[c,'state'])
            rows.append(row);keys.add(int(c));new_fits+=1
        if i%100==0 or i==len(ids):
            pd.DataFrame(rows).to_csv(path,index=False)
            print(f'Selective caps {i}/{len(ids)}; new fits {new_fits}; seconds {time.monotonic()-started:.1f}',flush=True)
    singles=pd.read_csv(path)
    assert len(singles)==1086 and np.isfinite(singles[['beta','se','t','F']]).all().all()
    singles['raw_beta']=b0;singles['full_winsorized_beta']=bw
    singles['beta_change']=singles.beta-b0
    singles['full_change_reproduced_percent']=100*singles.beta_change/full_change
    ranking=singles.sort_values(['beta_change','ccg'],ascending=[False,True]).reset_index(drop=True)
    ranking.insert(0,'rank',np.arange(1,len(ranking)+1))
    ranking.to_csv(OUT/'selective_cap_ranking.csv',index=False)
    ranking.loc[ranking.beta_change.gt(0)].to_csv(OUT/'selective_cap_positive_counties.csv',index=False)
    sizes=(2,5,10,20,50)
    assert ranking.head(50).beta_change.gt(0).all()
    jointpath=OUT/'selective_cap_joint_estimates.csv'
    joint=pd.read_csv(jointpath,keep_default_na=False).to_dict('records') if jointpath.exists() else []
    jointkeys={int(r['n_selected']) for r in joint}
    assert len(jointkeys)==len(joint) and jointkeys.issubset(set(sizes))
    membership=[]
    for k in sizes:
        group=ranking.head(k)
        selected=group.ccg.astype(int).tolist()
        membership.append(group[['rank','ccg','county','state']].assign(group_size=k))
        if k not in jointkeys:
            row=fit_selected(selected);joint.append(row);new_fits+=1
        else:
            row=next(r for r in joint if int(r['n_selected'])==k)
            assert row['analysis_sha256']==cache_hash and row['selected']==';'.join(map(str,selected))
        pd.DataFrame(joint).to_csv(jointpath,index=False)
    result=pd.DataFrame(joint).sort_values('n_selected')
    result['beta_change']=result.beta-b0
    result['full_change_reproduced_percent']=100*result.beta_change/full_change
    result['distance_to_full_winsorized']=abs(bw-result.beta)
    result['overshoots_full_winsorized']=result.beta.gt(bw)
    result.to_csv(OUT/'selective_cap_joint_summary.csv',index=False)
    pd.concat(membership,ignore_index=True).to_csv(OUT/'selective_cap_joint_membership.csv',index=False)
    focal=a.loc[a.ccg.isin(ranking.head(50).ccg)&a.any_clipped].copy()
    focal['state']=focal.ccg.map(names.state)
    focal.to_csv(OUT/'selective_cap_top50_cells.csv',index=False)
    assert len(focal)==int(result.loc[result.n_selected.eq(50),'cells_changed'].iloc[0])
    check=dict(status='PASS',new_fits=new_fits,n=107030,n_counties=3058,n_counties_removed=0,
        single_counties=1086,joint_sizes=list(sizes),full_raw_beta=b0,full_winsorized_beta=bw,
        analysis_sha256=cache_hash,cutoffs_fixed=True,all_beta_checks_passed=True,
        unchanged_nonselected_values=True,no_unintended_drops=True)
    (OUT/'selective_cap_check.json').write_text(json.dumps(check,indent=2)+'\n')
    print('INDIVIDUAL CAPS',flush=True)
    print(ranking.head(10)[['rank','county','state','beta','t','beta_change','full_change_reproduced_percent','cells_changed']].to_string(index=False),flush=True)
    print('JOINT CAPS',flush=True)
    print(result[['n_selected','beta','se','t','F','full_change_reproduced_percent','cells_changed']].to_string(index=False),flush=True)
    print(json.dumps(check,indent=2),flush=True)

def selective_cell_caps():
    """Attribute selective capping to county-years and test fixed-ranked prefixes."""
    import time
    started=time.monotonic()
    a,t28=load()
    assert a.index.equals(pd.RangeIndex(len(a)))
    ids=np.flatnonzero(a.any_clipped.to_numpy())
    assert len(ids)==5597 and len(a)==107030
    cache_hash=sha(CACHE/'analysis.parquet')
    base=pd.read_csv(OUT/'baseline.csv').set_index('variant')
    b0=float(base.loc['raw','beta']);bw=float(base.loc['winsorized','beta']);gap=bw-b0
    raw=a[['y','x','z']].to_numpy(np.float64);cap=a[['yc','xc','zc']].to_numpy(np.float64)
    fe=a.fe.to_numpy(np.int64);cl=a.cluster.to_numpy(np.int64)
    counts=np.bincount(fe);weights=np.ones(len(a));nofe=np.zeros(len(a),np.int64)
    names=pd.read_csv(OUT/'clipped_counties.csv').set_index('ccg')
    def run(selected):
        values=raw.copy();values[selected]=cap[selected]
        changed=(values!=raw).any(axis=1)
        np.testing.assert_array_equal(np.flatnonzero(changed),np.sort(selected))
        y,x,z=values.T
        fit=t28.fit2sls(y,x,z,weights,fe,nofe,cl,singleton=False,want_F=True)
        assert fit['n']==107030 and fit['n_dropped']==0
        sy,sx,sz=(np.bincount(fe,weights=v) for v in (y,x,z))
        b=((z@y)-(sz*sy/counts).sum())/((z@x)-(sz*sx/counts).sum())
        np.testing.assert_allclose(b,fit['beta'],rtol=1e-9,atol=1e-13)
        return dict(n_selected=len(selected),counties_touched=int(a.iloc[selected].ccg.nunique()),
            n=107030,counties=3058,beta=fit['beta']*1e5,se=fit['se']*1e5,
            t=fit['beta']/fit['se'],F=fit['F'],analysis_sha256=cache_hash)
    path=OUT/'cell_cap_single_estimates.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    keys={int(r['row_id']) for r in rows}
    assert len(keys)==len(rows) and keys.issubset(set(ids))
    assert all(r['analysis_sha256']==cache_hash and r['n']==107030 for r in rows)
    computed=0
    for j,i in enumerate(ids,1):
        if int(i) not in keys:
            r=run([int(i)]);c=int(a.at[i,'ccg'])
            r.update(row_id=int(i),ccg=c,year=int(a.at[i,'year']),county=names.loc[c,'county'],state=names.loc[c,'state'])
            rows.append(r);keys.add(int(i));computed+=1
        if j%250==0 or j==len(ids):
            pd.DataFrame(rows).to_csv(path,index=False)
            print(f'Cell caps {j}/{len(ids)}; new fits {computed}; seconds {time.monotonic()-started:.1f}',flush=True)
    rank=pd.read_csv(path)
    assert len(rank)==5597 and np.isfinite(rank[['beta','se','t','F']]).all().all()
    rank['beta_change']=rank.beta-b0
    rank['individual_share_of_full_change_percent']=100*rank.beta_change/gap
    rank=rank.sort_values(['beta_change','ccg','year'],ascending=[False,True,True]).reset_index(drop=True)
    rank.insert(0,'rank',np.arange(1,len(rank)+1))
    rank.to_csv(OUT/'cell_cap_ranking.csv',index=False)
    # Exact just-identified within-FE coefficient along every fixed-rank prefix.
    # Only the affected FE group's sums and cross-products change at each step.
    y,x,z=raw.T
    sums=np.array([np.bincount(fe,weights=v) for v in (y,x,z)])
    N=(z@y)-(sums[2]*sums[0]/counts).sum()
    D=(z@x)-(sums[2]*sums[1]/counts).sum()
    np.testing.assert_allclose(N/D*1e5,b0,rtol=1e-10)
    prefixes=[]
    for k,i in enumerate(rank.row_id.astype(int),1):
        f=fe[i];before=sums[:,f].copy();after=before+cap[i]-raw[i]
        N+=cap[i,2]*cap[i,0]-raw[i,2]*raw[i,0]-(after[2]*after[0]-before[2]*before[0])/counts[f]
        D+=cap[i,2]*cap[i,1]-raw[i,2]*raw[i,1]-(after[2]*after[1]-before[2]*before[1])/counts[f]
        sums[:,f]=after
        b=N/D*1e5
        prefixes.append(dict(n_selected=k,beta=b,full_change_reproduced_percent=100*(b-b0)/gap,
            residual_gap_fraction=abs(bw-b)/abs(gap)))
    prefix=pd.DataFrame(prefixes)
    np.testing.assert_allclose(prefix.iloc[-1].beta,bw,rtol=1e-8)
    prefix.to_csv(OUT/'cell_cap_prefix_coefficients.csv',index=False)
    first=int(prefix.loc[prefix.residual_gap_fraction.le(.05),'n_selected'].iloc[0])
    # The target is the first qualifying prefix of this ranking, not a global minimum subset.
    sizes=sorted(set([50,100,first]+[k for k in (200,500,1000,2000,4000) if k<first]))
    jointpath=OUT/'cell_cap_joint_estimates.csv'
    joint=pd.read_csv(jointpath).to_dict('records') if jointpath.exists() else []
    jointkeys={int(r['n_selected']) for r in joint}
    assert len(jointkeys)==len(joint) and jointkeys.issubset(set(sizes))
    members=[]
    for k in sizes:
        selected=rank.head(k).row_id.astype(int).tolist()
        members.append(rank.head(k)[['rank','row_id','ccg','year','county','state']].assign(group_size=k))
        if k not in jointkeys:
            r=run(selected);joint.append(r);computed+=1
        else:
            r=next(r for r in joint if int(r['n_selected'])==k)
            assert r['analysis_sha256']==cache_hash
        np.testing.assert_allclose(r['beta'],prefix.loc[prefix.n_selected.eq(k),'beta'].iloc[0],rtol=1e-8)
        pd.DataFrame(joint).to_csv(jointpath,index=False)
    result=pd.DataFrame(joint).sort_values('n_selected')
    result['full_change_reproduced_percent']=100*(result.beta-b0)/gap
    result['residual_gap_fraction']=abs(bw-result.beta)/abs(gap)
    result.to_csv(OUT/'cell_cap_joint_summary.csv',index=False)
    pd.concat(members,ignore_index=True).to_csv(OUT/'cell_cap_joint_membership.csv',index=False)
    target=rank.head(first)[['rank','row_id','ccg','year','county','state','beta_change']].merge(
        a.drop(columns='county'),on=['ccg','year'],validate='one_to_one')
    assert len(target)==first
    target.to_csv(OUT/'cell_cap_target_observations.csv',index=False)
    check=dict(status='PASS',new_fits=computed,individual_cells=5597,n=107030,n_counties=3058,
        observations_removed=0,cutoffs_fixed=True,analysis_sha256=cache_hash,
        target_first_ranked_prefix=first,target_tolerance_fraction=.05,
        prefix_full_cap_matches_saved_estimate=True,joint_fits_match_prefix_moments=True,
        target_is_not_global_minimum=True)
    (OUT/'cell_cap_check.json').write_text(json.dumps(check,indent=2)+'\n')
    print(rank.head(10)[['rank','county','state','year','beta','t','beta_change']].to_string(index=False),flush=True)
    print(result[['n_selected','counties_touched','beta','se','t','F','full_change_reproduced_percent']].to_string(index=False),flush=True)
    print(json.dumps(check,indent=2),flush=True)

def precision_cell_caps():
    """Hold 212 coefficient-target caps fixed; screen remaining cells by SE reduction."""
    import time
    started=time.monotonic()
    a,t28=load();cache_hash=sha(CACHE/'analysis.parquet')
    targetpath=OUT/'cell_cap_target_observations.csv';target=pd.read_csv(targetpath)
    target_hash=sha(targetpath);anchor=target.row_id.astype(int).to_numpy()
    assert len(anchor)==212 and len(set(anchor))==212
    np.testing.assert_array_equal(a.iloc[anchor][['ccg','year']].to_numpy(),target[['ccg','year']].to_numpy())
    candidates=np.setdiff1d(np.flatnonzero(a.any_clipped),anchor)
    assert len(candidates)==5385
    saved=pd.read_csv(OUT/'cell_cap_joint_estimates.csv')
    reference=saved.loc[saved.n_selected.eq(212)].iloc[0]
    assert reference.analysis_sha256==cache_hash
    names=pd.read_csv(OUT/'clipped_counties.csv').set_index('ccg')
    raw=a[['y','x','z']].to_numpy(np.float64);cap=a[['yc','xc','zc']].to_numpy(np.float64)
    anchored=raw.copy();anchored[anchor]=cap[anchor]
    fe=a.fe.to_numpy(np.int64);cl=a.cluster.to_numpy(np.int64);counts=np.bincount(fe)
    weights=np.ones(len(a));nofe=np.zeros(len(a),np.int64)
    def run(added):
        values=anchored.copy();values[added]=cap[added]
        expected=np.sort(np.concatenate([anchor,added]))
        np.testing.assert_array_equal(np.flatnonzero((values!=raw).any(axis=1)),expected)
        np.testing.assert_array_equal(values[anchor],cap[anchor])
        y,x,z=values.T
        fit=t28.fit2sls(y,x,z,weights,fe,nofe,cl,singleton=False,want_F=True)
        assert fit['n']==107030 and fit['n_dropped']==0
        sy,sx,sz=(np.bincount(fe,weights=v) for v in (y,x,z))
        direct=((z@y)-(sz*sy/counts).sum())/((z@x)-(sz*sx/counts).sum())
        np.testing.assert_allclose(direct,fit['beta'],rtol=1e-9,atol=1e-13)
        return dict(n_added=len(added),total_capped_cells=212+len(added),n=107030,counties=3058,
            beta=fit['beta']*1e5,se=fit['se']*1e5,t=fit['beta']/fit['se'],F=fit['F'],
            analysis_sha256=cache_hash,anchor_sha256=target_hash)
    path=OUT/'precision_cap_single_estimates.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    keys={int(r['row_id']) for r in rows}
    assert len(keys)==len(rows) and keys.issubset(set(candidates))
    assert all(r['analysis_sha256']==cache_hash and r['anchor_sha256']==target_hash for r in rows)
    computed=0
    for j,i in enumerate(candidates,1):
        if int(i) not in keys:
            r=run([int(i)]);c=int(a.at[i,'ccg'])
            r.update(row_id=int(i),ccg=c,year=int(a.at[i,'year']),county=names.loc[c,'county'],state=names.loc[c,'state'])
            rows.append(r);keys.add(int(i));computed+=1
        if j%250==0 or j==len(candidates):
            pd.DataFrame(rows).to_csv(path,index=False)
            print(f'Precision caps {j}/{len(candidates)}; new fits {computed}; seconds {time.monotonic()-started:.1f}',flush=True)
    rank=pd.read_csv(path)
    assert len(rank)==5385 and np.isfinite(rank[['beta','se','t','F']]).all().all()
    rank['se_reduction']=reference.se-rank.se
    rank['se_reduction_percent']=100*rank.se_reduction/reference.se
    rank['beta_change']=rank.beta-reference.beta
    rank['beta_change_percent']=100*rank.beta_change/reference.beta
    rank=rank.sort_values(['se_reduction','ccg','year'],ascending=[False,True,True]).reset_index(drop=True)
    rank.insert(0,'rank',np.arange(1,len(rank)+1))
    rank.to_csv(OUT/'precision_cap_ranking.csv',index=False)
    rank.loc[rank.se_reduction.gt(0)].to_csv(OUT/'precision_cap_se_reducing_cells.csv',index=False)
    assert rank.head(100).se_reduction.gt(0).all()
    jointpath=OUT/'precision_cap_joint_estimates.csv'
    joint=pd.read_csv(jointpath).to_dict('records') if jointpath.exists() else []
    sizes=(1,5,10,20,50,100);jointkeys={int(r['n_added']) for r in joint}
    assert len(jointkeys)==len(joint) and jointkeys.issubset(set(sizes))
    members=[]
    for k in sizes:
        added=rank.head(k).row_id.astype(int).tolist()
        members.append(rank.head(k)[['rank','row_id','ccg','year','county','state']].assign(group_size=k))
        selected_hash=hashlib.sha256(np.array(added,dtype=np.int64).tobytes()).hexdigest()
        if k not in jointkeys:
            r=rank.iloc[0].to_dict() if k==1 else run(added)
            if k!=1:computed+=1
            r.update(selected_sha256=selected_hash,estimate_source='saved_single' if k==1 else 'estimated')
            joint.append(r)
        else:
            r=next(r for r in joint if int(r['n_added'])==k)
            assert r['selected_sha256']==selected_hash and r['anchor_sha256']==target_hash and r['analysis_sha256']==cache_hash
        pd.DataFrame(joint).to_csv(jointpath,index=False)
    result=pd.DataFrame(joint).sort_values('n_added')
    result['se_reduction']=reference.se-result.se
    result['se_reduction_percent']=100*result.se_reduction/reference.se
    result['beta_change']=result.beta-reference.beta
    result.to_csv(OUT/'precision_cap_joint_summary.csv',index=False)
    pd.concat(members,ignore_index=True).to_csv(OUT/'precision_cap_joint_membership.csv',index=False)
    rank.head(100)[['rank','row_id','ccg','year','county','state','se_reduction','beta_change']].merge(
        a.drop(columns='county'),on=['ccg','year'],validate='one_to_one').to_csv(OUT/'precision_cap_top100_observations.csv',index=False)
    check=dict(status='PASS',new_fits=computed,anchor_caps=212,candidate_cells=5385,n=107030,
        counties=3058,observations_removed=0,cutoffs_fixed=True,all_anchors_preserved=True,
        joint_sizes=list(sizes),analysis_sha256=cache_hash,anchor_sha256=target_hash,
        se_reducing_cells=int(rank.se_reduction.gt(0).sum()))
    (OUT/'precision_cap_check.json').write_text(json.dumps(check,indent=2)+'\n')
    print(rank.head(10)[['rank','county','state','year','beta','se','t','se_reduction','beta_change']].to_string(index=False),flush=True)
    print(result[['n_added','total_capped_cells','beta','se','t','F','se_reduction_percent']].to_string(index=False),flush=True)
    print(json.dumps(check,indent=2),flush=True)

def joint_target_caps():
    """Start raw and match coefficient and SE jointly along a fixed ranking."""
    import time
    started=time.monotonic();a,t28=load()
    cache_hash=sha(CACHE/'analysis.parquet')
    baseline=pd.read_csv(OUT/'baseline.csv').set_index('variant')
    bstar=float(baseline.loc['winsorized','beta']);sstar=float(baseline.loc['winsorized','se'])
    tolerance=.05
    singles=pd.read_csv(OUT/'cell_cap_single_estimates.csv')
    assert len(singles)==5597 and singles.analysis_sha256.eq(cache_hash).all()
    assert set(singles.row_id)==set(a.index[a.any_clipped])
    rank=singles.copy()
    rank['target_loss']=((rank.beta-bstar)/abs(bstar))**2+((rank.se-sstar)/sstar)**2
    rank=rank.sort_values(['target_loss','ccg','year'],ascending=[True,True,True]).reset_index(drop=True)
    rank.insert(0,'joint_rank',np.arange(1,len(rank)+1))
    rank.to_csv(OUT/'joint_target_ranking.csv',index=False)
    rank_hash=sha(OUT/'joint_target_ranking.csv')
    raw=a[['y','x','z']].to_numpy(np.float64);cap=a[['yc','xc','zc']].to_numpy(np.float64)
    values=raw.copy();fe=a.fe.to_numpy(np.int64);cl=a.cluster.to_numpy(np.int64)
    counts=np.bincount(fe);weights=np.ones(len(a));nofe=np.zeros(len(a),np.int64)
    path=OUT/'joint_target_prefix_estimates.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    assert all(r['analysis_sha256']==cache_hash and r['ranking_sha256']==rank_hash for r in rows)
    existing={int(r['n_selected']):r for r in rows}
    assert len(existing)==len(rows) and sorted(existing)==list(range(1,len(rows)+1))
    computed=0;selected=[];match=None
    fields=['beta','se','t','F','n','counties']
    for k,i in enumerate(rank.row_id.astype(int),1):
        selected.append(i);values[i]=cap[i]
        if k in existing:
            r=existing[k]
        else:
            if k==1:
                r={f:rank.iloc[0][f] for f in fields};r['estimate_source']='saved_single'
            elif k==len(rank):
                r={f:baseline.loc['winsorized',f] for f in fields};r['estimate_source']='saved_full_winsorization'
            else:
                y,x,z=values.T
                fit=t28.fit2sls(y,x,z,weights,fe,nofe,cl,singleton=False,want_F=True)
                assert fit['n']==107030 and fit['n_dropped']==0
                r=dict(beta=fit['beta']*1e5,se=fit['se']*1e5,t=fit['beta']/fit['se'],
                    F=fit['F'],n=107030,counties=3058,estimate_source='estimated');computed+=1
            r.update(n_selected=k,counties_touched=int(a.iloc[selected].ccg.nunique()),
                analysis_sha256=cache_hash,ranking_sha256=rank_hash)
            r['beta_error_percent']=100*(r['beta']-bstar)/abs(bstar)
            r['se_error_percent']=100*(r['se']-sstar)/sstar
            r['target_loss']=(r['beta_error_percent']/100)**2+(r['se_error_percent']/100)**2
            r['joint_match']=abs(r['beta_error_percent'])<=100*tolerance and abs(r['se_error_percent'])<=100*tolerance
            rows.append(r)
        assert np.isfinite([r[f] for f in ['beta','se','t','F']]).all()
        assert r['n']==107030 and r['counties']==3058
        y,x,z=values.T
        sy,sx,sz=(np.bincount(fe,weights=v) for v in (y,x,z))
        direct=((z@y)-(sz*sy/counts).sum())/((z@x)-(sz*sx/counts).sum())
        np.testing.assert_allclose(direct*1e5,r['beta'],rtol=1e-9,atol=1e-10)
        if k%100==0 or r['joint_match']:
            pd.DataFrame(rows).to_csv(path,index=False)
            print(f'Joint target prefix {k}: beta {r["beta"]:.6f}, SE {r["se"]:.6f}, t {r["t"]:.6f}; seconds {time.monotonic()-started:.1f}',flush=True)
        if r['joint_match']:
            match=r;break
    assert match is not None
    result=pd.read_csv(path)
    assert len(result)==len(selected) and not result.iloc[:-1].joint_match.any()
    np.testing.assert_array_equal(np.flatnonzero((values!=raw).any(axis=1)),np.sort(selected))
    targets=rank.head(len(selected))[['joint_rank','row_id','ccg','year','county','state','target_loss']].merge(
        a.drop(columns='county'),on=['ccg','year'],validate='one_to_one')
    targets.to_csv(OUT/'joint_target_observations.csv',index=False)
    summary=[]
    for variant,label in [('raw','Unwinsorized'),('winsorized','Full winsorization')]:
        r={f:baseline.loc[variant,f] for f in fields};r.update(sample=label,n_selected=0 if variant=='raw' else len(rank));summary.append(r)
    final=dict(match);final['sample']='Joint coefficient and SE target';summary.insert(1,final)
    pd.DataFrame(summary).to_csv(OUT/'joint_target_summary.csv',index=False)
    check=dict(status='PASS',new_fits=computed,starts_completely_raw=True,preselected_cells=0,
        candidate_cells=5597,target_coefficient=bstar,target_se=sstar,tolerance_percent=5,
        loss='sum of squared relative coefficient and SE errors',ranking_fixed=True,
        matched_prefix=len(selected),counties_touched=match['counties_touched'],
        all_observations_retained=True,cutoffs_fixed=True,not_a_global_minimum=True,
        analysis_sha256=cache_hash,ranking_sha256=rank_hash)
    (OUT/'joint_target_check.json').write_text(json.dumps(check,indent=2)+'\n')
    print(pd.DataFrame(summary)[['sample','n_selected','beta','se','t','F']].to_string(index=False),flush=True)
    print(json.dumps(check,indent=2),flush=True)

def adaptive_joint_target():
    """Greedy joint coefficient/SE matching with exact conditional cluster moments."""
    import time
    started=time.monotonic();a,t28=load();cache_hash=sha(CACHE/'analysis.parquet')
    base=pd.read_csv(OUT/'baseline.csv').set_index('variant')
    bstar=float(base.loc['winsorized','beta']);sstar=float(base.loc['winsorized','se'])
    raw=a[['y','x','z']].to_numpy(np.float64);cap=a[['yc','xc','zc']].to_numpy(np.float64)
    delta=cap-raw;values=raw.copy();allids=np.flatnonzero(a.any_clipped)
    fe=a.fe.to_numpy(np.int64);cl=a.cluster.to_numpy(np.int64)
    counts=np.bincount(fe);order=np.argsort(fe).astype(np.int64)
    starts=np.concatenate([[0],np.cumsum(counts)]).astype(np.int64)
    assert not a.duplicated(['cluster','fe']).any()
    G=a.cluster.nunique();K=a.fe.nunique()+1;Nobs=len(a)
    ssc=G/(G-1)*(Nobs-1)/(Nobs-K)
    names=pd.read_csv(OUT/'clipped_counties.csv').set_index('ccg')
    def candidate_moments(ids,yt,xt,zt,ry,rx,numer,denom,qyy,qyx,qxx):
        bs=np.full(len(ids),np.nan);ses=np.full(len(ids),np.nan);losses=np.full(len(ids),np.inf)
        position=np.empty(len(a),dtype=np.int64);position[order]=np.arange(len(a))
        for f in np.unique(fe[ids]):
            loc=np.flatnonzero(fe[ids]==f);ii=ids[loc];ng=counts[f]
            group=order[starts[f]:starts[f+1]];cc=cl[group]
            dy,dx,dz=delta[ii].T
            ni=numer+dz*yt[ii]+dy*zt[ii]+dz*dy*(1-1/ng)
            di=denom+dz*xt[ii]+dx*zt[ii]+dz*dx*(1-1/ng)
            b=np.divide(ni,di,out=np.full(len(ii),np.nan),where=di!=0)
            h=np.full((len(ii),len(group)),-1.0/ng)
            h[np.arange(len(ii)),position[ii]-starts[f]]+=1
            dny=h*(dy[:,None]*zt[group]+dz[:,None]*yt[group])+h*h*(dz*dy)[:,None]
            ddx=h*(dx[:,None]*zt[group]+dz[:,None]*xt[group])+h*h*(dz*dx)[:,None]
            ds=dny-b[:,None]*ddx
            score=ry[cc]-b[:,None]*rx[cc]
            v=qyy-2*b*qyx+b*b*qxx+np.sum(2*score*ds+ds*ds,axis=1)
            bs[loc]=b*1e5
            se=np.sqrt(np.maximum(v,0)*ssc/(di*di))*1e5
            valid=(v>0)&np.isfinite(b)&np.isfinite(se)
            ses[loc]=se
            loss=((b*1e5-bstar)/bstar)**2+((se-sstar)/sstar)**2
            losses[loc]=np.where(valid,loss,np.inf)
        return bs,ses,losses
    path=OUT/'adaptive_target_path.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    assert all(r['analysis_sha256']==cache_hash for r in rows)
    selected=[int(r['row_id']) for r in rows]
    assert len(selected)==len(set(selected)) and set(selected).issubset(set(allids))
    values[selected]=cap[selected]
    current_loss=float(rows[-1]['target_loss']) if rows else ((base.loc['raw','beta']-bstar)/bstar)**2+((base.loc['raw','se']-sstar)/sstar)**2
    status='max_500_reached';validated_initial=bool(rows)
    for step in range(len(rows)+1,501):
        if rows and rows[-1]['joint_match']:
            status='matched';break
        ids=np.setdiff1d(allids,np.array(selected,dtype=np.int64))
        yt,xt,zt=(values-np.column_stack([np.bincount(fe,weights=v)[fe]/counts[fe] for v in values.T])).T
        ry=np.bincount(cl,weights=zt*yt);rx=np.bincount(cl,weights=zt*xt)
        bpred,spred,losses=candidate_moments(ids,yt,xt,zt,ry,rx,zt@yt,zt@xt,ry@ry,ry@rx,rx@rx)
        if step==1:
            saved=pd.read_csv(OUT/'cell_cap_single_estimates.csv').set_index('row_id').loc[ids]
            np.testing.assert_allclose(bpred,saved.beta,rtol=1e-7,atol=1e-9)
            np.testing.assert_allclose(spred,saved.se,rtol=1e-7,atol=1e-9)
            validated_initial=True
        best=int(np.argmin(losses));i=int(ids[best])
        if losses[best]>=current_loss-1e-12:
            status='no_improving_single_addition';break
        selected.append(i);values[i]=cap[i]
        y,x,z=values.T
        fit=t28.fit2sls(y,x,z,np.ones(Nobs),fe,np.zeros(Nobs,np.int64),cl,singleton=False,want_F=True)
        assert fit['n']==Nobs and fit['n_dropped']==0
        b=fit['beta']*1e5;se=fit['se']*1e5
        np.testing.assert_allclose([b,se],[bpred[best],spred[best]],rtol=1e-7,atol=1e-9)
        c=int(a.at[i,'ccg'])
        current_loss=((b-bstar)/bstar)**2+((se-sstar)/sstar)**2
        r=dict(step=step,row_id=i,ccg=c,year=int(a.at[i,'year']),county=names.loc[c,'county'],state=names.loc[c,'state'],
            beta=b,se=se,t=b/se,F=fit['F'],target_loss=current_loss,beta_error_percent=100*(b-bstar)/bstar,
            se_error_percent=100*(se-sstar)/sstar,joint_match=abs(b/bstar-1)<=.05 and abs(se/sstar-1)<=.05,
            n=Nobs,counties=G,analysis_sha256=cache_hash)
        rows.append(r)
        if step%10==0 or r['joint_match']:
            pd.DataFrame(rows).to_csv(path,index=False)
            print(f'Adaptive {step}: beta {b:.6f}, SE {se:.6f}, t {b/se:.6f}, loss {current_loss:.6f}; seconds {time.monotonic()-started:.1f}',flush=True)
        if r['joint_match']:
            status='matched';break
    pd.DataFrame(rows).to_csv(path,index=False)
    result=pd.DataFrame(rows)
    np.testing.assert_array_equal(np.flatnonzero((values!=raw).any(axis=1)),np.sort(selected))
    result[['step','row_id','ccg','year','county','state']].merge(a.drop(columns='county'),on=['ccg','year'],
        validate='one_to_one').to_csv(OUT/'adaptive_target_observations.csv',index=False)
    summary=[]
    for v,label in [('raw','Unwinsorized'),('winsorized','Full winsorization')]:
        r={f:base.loc[v,f] for f in ['beta','se','t','F','n','counties']}
        r.update(sample=label,n_selected=0 if v=='raw' else len(allids));summary.append(r)
    final=rows[-1].copy();final.update(sample='Adaptive joint target',n_selected=len(rows));summary.insert(1,final)
    pd.DataFrame(summary).to_csv(OUT/'adaptive_target_summary.csv',index=False)
    check=dict(status=status,n_selected=len(selected),counties_touched=int(a.iloc[selected].ccg.nunique()),
        starts_raw=True,preselected_cells=0,maximum_additions=500,all_5597_initial_candidate_SEs_verified=validated_initial,
        selected_steps_verified_with_canonical_estimator=True,observations_removed=0,cutoffs_fixed=True,
        target_beta=bstar,target_se=sstar,tolerance_percent=5,not_a_global_minimum=True,analysis_sha256=cache_hash)
    (OUT/'adaptive_target_check.json').write_text(json.dumps(check,indent=2)+'\n')
    print(pd.DataFrame(summary)[['sample','n_selected','beta','se','t','F']].to_string(index=False),flush=True)
    print(json.dumps(check,indent=2),flush=True)

def backward_joint_target():
    """Restore raw cells from full capping, preserving coefficient and SE tolerance."""
    import time
    started=time.monotonic();a,t28=load();cache_hash=sha(CACHE/'analysis.parquet')
    base=pd.read_csv(OUT/'baseline.csv').set_index('variant')
    bstar=float(base.loc['winsorized','beta']);sstar=float(base.loc['winsorized','se'])
    raw=a[['y','x','z']].to_numpy(np.float64);cap=a[['yc','xc','zc']].to_numpy(np.float64)
    values=cap.copy();active=a.any_clipped.to_numpy().copy()
    fe=a.fe.to_numpy(np.int64);cl=a.cluster.to_numpy(np.int64)
    weights=np.ones(len(a));nofe=np.zeros(len(a),np.int64)
    def statekey(mask):return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
    fields=['beta','se','t','F','n','counties']
    memo={statekey(active):{f:base.loc['winsorized',f] for f in fields},
          statekey(np.zeros(len(a),bool)):{f:base.loc['raw',f] for f in fields}}
    path=OUT/'backward_target_trials.csv'
    trials=pd.read_csv(path,keep_default_na=False).to_dict('records') if path.exists() else []
    for r in trials:
        assert r['analysis_sha256']==cache_hash
        memo[r['candidate_state_sha256']]={f:r[f] for f in fields}
        if r['accepted']:
            ids=list(map(int,str(r['restored_row_ids']).split(';')))
            assert active[ids].all()
            active[ids]=False;values[ids]=raw[ids]
    terminal=OUT/'backward_target_check.json'
    if terminal.exists():
        check=json.loads(terminal.read_text())
        assert check['analysis_sha256']==cache_hash and check['final_state_sha256']==statekey(active)
        if check['status']=='no_further_tested_restoration':
            print(pd.read_csv(OUT/'backward_target_summary.csv').to_string(index=False));print(json.dumps(check,indent=2));return
    new_fits=0;attempts=len(trials)
    def attempt(ids,pass_id):
        nonlocal new_fits,attempts
        ids=np.asarray(ids,dtype=np.int64)
        assert len(ids)>0 and active[ids].all()
        before=int(active.sum());proposed=active.copy();proposed[ids]=False;key=statekey(proposed)
        values[ids]=raw[ids]
        if key in memo:
            fitrow=memo[key].copy();source='saved_state'
        else:
            y,x,z=values.T
            fit=t28.fit2sls(y,x,z,weights,fe,nofe,cl,singleton=False,want_F=True)
            assert fit['n']==107030 and fit['n_dropped']==0
            fitrow=dict(beta=fit['beta']*1e5,se=fit['se']*1e5,t=fit['beta']/fit['se'],F=fit['F'],n=107030,counties=3058)
            assert np.isfinite([fitrow[f] for f in fields]).all()
            memo[key]=fitrow.copy();new_fits+=1;source='estimated'
        accepted=abs(fitrow['beta']/bstar-1)<=.05 and abs(fitrow['se']/sstar-1)<=.05
        if accepted:active[ids]=False
        else:values[ids]=cap[ids]
        attempts+=1
        trials.append(dict(attempt=attempts,pass_id=pass_id,batch_size=len(ids),caps_before=before,
            caps_after=int(active.sum()),accepted=accepted,restored_row_ids=';'.join(map(str,ids)),
            candidate_state_sha256=key,analysis_sha256=cache_hash,estimate_source=source,**fitrow))
        if attempts%250==0:
            pd.DataFrame(trials).to_csv(path,index=False)
            print(f'Backward pass {pass_id}, attempts {attempts}, caps remaining {active.sum()}, new fits {new_fits}, seconds {time.monotonic()-started:.1f}',flush=True)
        return accepted
    def restore_batch(ids,pass_id):
        if attempt(ids,pass_id):return
        if len(ids)>1:
            mid=len(ids)//2
            restore_batch(ids[:mid],pass_id);restore_batch(ids[mid:],pass_id)
    status='pass_limit_reached';startpass=1+max([int(r['pass_id']) for r in trials],default=0)
    for pass_id in range(startpass,startpass+10):
        before=int(active.sum());remaining=np.flatnonzero(active)
        for start in range(0,len(remaining),256):
            restore_batch(remaining[start:start+256],pass_id)
        pd.DataFrame(trials).to_csv(path,index=False)
        print(f'Completed backward pass {pass_id}: {before} -> {active.sum()} caps',flush=True)
        if int(active.sum())==before:
            status='no_further_tested_restoration';break
    final=memo[statekey(active)].copy()
    assert abs(final['beta']/bstar-1)<=.05 and abs(final['se']/sstar-1)<=.05
    np.testing.assert_array_equal(np.flatnonzero((values!=raw).any(axis=1)),np.flatnonzero(active))
    summary=[]
    for v,label in [('raw','Unwinsorized'),('winsorized','Full winsorization')]:
        r={f:base.loc[v,f] for f in fields};r.update(sample=label,capped_cells=0 if v=='raw' else 5597);summary.append(r)
    final.update(sample='Backward-selected caps',capped_cells=int(active.sum()));summary.insert(1,final)
    pd.DataFrame(summary).to_csv(OUT/'backward_target_summary.csv',index=False)
    names=pd.read_csv(OUT/'clipped_counties.csv').set_index('ccg')
    target=a.loc[active].copy();target['row_id']=target.index;target['state']=target.ccg.map(names.state)
    target.to_csv(OUT/'backward_target_observations.csv',index=False)
    check=dict(status=status,new_fits=new_fits,total_attempts=attempts,remaining_caps=int(active.sum()),
        restored_cells=5597-int(active.sum()),counties_touched=int(target.ccg.nunique()),
        starts_fully_winsorized=True,preselected_cells=0,target_coefficient=bstar,target_se=sstar,tolerance_percent=5,
        batch_rule='ascending county/year blocks of 256, recursively halved on rejection, repeated passes',
        every_accepted_state_within_tolerance=True,single_restore_local_minimum=status=='no_further_tested_restoration',
        not_a_global_minimum=True,observations_removed=0,analysis_sha256=cache_hash,final_state_sha256=statekey(active))
    (OUT/'backward_target_check.json').write_text(json.dumps(check,indent=2)+'\n')
    print(pd.DataFrame(summary)[['sample','capped_cells','beta','se','t','F']].to_string(index=False),flush=True)
    print(json.dumps(check,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--build',action='store_true');m.add_argument('--estimate',action='store_true')
    m.add_argument('--single-leaveouts',action='store_true')
    m.add_argument('--joint-leaveouts',action='store_true')
    m.add_argument('--selective-caps',action='store_true')
    m.add_argument('--cell-caps',action='store_true')
    m.add_argument('--precision-caps',action='store_true')
    m.add_argument('--joint-target-caps',action='store_true')
    m.add_argument('--adaptive-target-caps',action='store_true')
    m.add_argument('--backward-target-caps',action='store_true')
    args=p.parse_args()
    if args.build:build()
    elif args.estimate:estimate()
    elif args.single_leaveouts:single_leaveouts()
    elif args.joint_leaveouts:joint_leaveouts()
    elif args.selective_caps:selective_caps()
    elif args.cell_caps:selective_cell_caps()
    elif args.precision_caps:precision_cell_caps()
    elif args.joint_target_caps:joint_target_caps()
    elif args.adaptive_target_caps:adaptive_joint_target()
    elif args.backward_target_caps:backward_joint_target()
