"""EW h=2 net county flows and national shifts with fixed positive exposures.

Uses the certified report-corrected positive baseline and canonical contract
readers. Adds negative obligations to observed spending and national category
spending in the shock years only. Predetermined exposure shares remain the
positive-action shares over t-11 through t-7. Two fits: raw and pooled 1/99.
"""
# mpgr-run-class: cache_rebuild (default); estimate_only (--after-nine)
# mpgr-rebuild-reason: treatment_changed
# mpgr-cache: data/temp/report_correction_rebuild/analysis_pooled.parquet
# mpgr-output-cache: data/temp/h2_net_flows_fixed_exposure_ew/analysis_pooled.parquet
# mpgr-design-keys: ccg year fe_division_year cluster
# mpgr-new-estimates: 2
# mpgr-batch-approved: explicit
# mpgr-variation-scope: county|h2|EW|full-balanced|net-x-net-shift-positive-exposure|raw-pooled1-99

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
import contracts
import dcads_common as dc
import store

SOURCE=dc.MACRO_VERSION_TEMP_DIR/'report_correction_rebuild'
CACHE=dc.MACRO_VERSION_TEMP_DIR/'h2_net_flows_fixed_exposure_ew'
OUT=dc.MACRO_RESULTS_ROOT/'instrument_normalization'
TARGETS=list(range(1982,2017))
FLOW_YEARS=list(range(1980,2017))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((SOURCE/'certification.json').read_text())['status']=='PASS'
    baseline_sha=sha(SOURCE/'analysis_pooled.parquet')
    a=pd.read_parquet(SOURCE/'analysis_pooled.parquet')
    units=np.sort(a.ccg.unique())
    assert len(a)==107030 and len(units)==3058
    np.testing.assert_array_equal(a.ccg,np.repeat(units,35))
    np.testing.assert_array_equal(a.year,np.tile(TARGETS,len(units)))
    positive=pd.read_parquet(SOURCE/'county_spending_positive.parquet').pivot(index='ccg',columns='year',values='dollars').reindex(units)
    p=pd.read_parquet(store.PIPE/'county_panel_pipeline.parquet',columns=['ccg','year','P'])
    population=p.pivot(index='ccg',columns='year',values='P').reindex(units)
    p100=module('p100_netfixed',PIPELINE/'100_h2_full_instrument_design_grid.py')
    p100.path_for=lambda cover,rule,floor,year:SOURCE/'annual_category'/f'fy={year}.parquet'
    exposure=p100.load_sparse('exclude_construction','positive',0,4,units)
    zbaseline,_=p100.instruments(exposure,exposure,population,'rolling',5)
    np.testing.assert_allclose(zbaseline.ravel(),a.z,rtol=1e-10,atol=1e-8)
    cats,natpos,matrices=exposure
    mapping,_,_=dc.load_ccg_map()
    negative=[];national=[];checks=[]
    for year in FLOW_YEARS:
        d,c=contracts.county_actions(year,mapping,drop_construction=True)
        assert np.isfinite(d).all()
        cp=pd.Series(d[d>0]*dc.deflate(year),index=c[d>0]).groupby(level=0).sum().reindex(units,fill_value=0)
        np.testing.assert_allclose(cp,positive[year],rtol=1e-10,atol=.01)
        cn=pd.Series(d[d<0]*dc.deflate(year),index=c[d<0]).groupby(level=0).sum().reindex(units,fill_value=0)
        negative.append(cn.rename(year))
        d,c,k=contracts.category_actions(year,mapping,drop_construction=True,harmonize=True)
        np0=pd.Series(d[d>0]*dc.deflate(year),index=k[d>0]).groupby(level=0).sum()
        expected=pd.Series(natpos[:,year-1966],index=cats)
        np.testing.assert_allclose(np0.reindex(cats,fill_value=0),expected,rtol=1e-10,atol=.01)
        # Positive source aggregates must not introduce a category absent from the exposure universe.
        assert np0.loc[~np0.index.isin(cats)].abs().sum()<.01
        nn=pd.Series(d[d<0]*dc.deflate(year),index=k[d<0]).groupby(level=0).sum()
        national.append(nn.rename(year))
        checks.append(dict(year=year,county_negative_real=cn.sum(),national_category_negative_real=nn.sum(),
            negative_outside_exposure_categories=nn.loc[~nn.index.isin(cats)].sum()))
        print('Checked positive baseline; aggregated negative flows:',year,flush=True)
    county_negative=pd.concat(negative,axis=1)
    natnegative=pd.concat(national,axis=1).fillna(0)
    # Outside 1980-2016 these computational placeholders are never used as flow years.
    nn=natnegative.reindex(index=cats,columns=range(1966,2019),fill_value=0).fillna(0).to_numpy()
    delta_z,_=p100.instruments(exposure,(cats,nn,{}),population,'rolling',5)
    znet,_=p100.instruments(exposure,(cats,natpos+nn,{}),population,'rolling',5)
    np.testing.assert_allclose(znet,zbaseline+delta_z,rtol=1e-10,atol=1e-8)
    # Explicit category-by-category calculation of the added negative-flow component.
    for j,t in enumerate(TARGETS):
        years=list(range(t-11,t-6))
        shareden=natpos[:,[yr-1966 for yr in years]].sum(axis=1)
        change=.5*(nn[:,t-1966]+nn[:,t-1-1966])-nn[:,t-2-1966]
        ratio=np.divide(change,shareden,out=np.zeros_like(change),where=shareden>0)
        check=sum(np.asarray(matrices[yr]@ratio).ravel() for yr in years)/population[t-2].to_numpy()
        np.testing.assert_allclose(check,delta_z[:,j],rtol=1e-10,atol=1e-8)
    pbase=population[[t-2 for t in TARGETS]].to_numpy()
    assert np.isfinite(pbase).all() and (pbase>0).all()
    def flow(wide):
        return (.5*(wide[TARGETS].to_numpy()+wide[[t-1 for t in TARGETS]].to_numpy())-
                wide[[t-2 for t in TARGETS]].to_numpy())/pbase
    np.testing.assert_allclose(flow(positive).ravel(),a.x,rtol=1e-12,atol=1e-10)
    net=positive[FLOW_YEARS]+county_negative
    xnet=flow(net)
    np.testing.assert_allclose(xnet,flow(positive)+flow(county_negative),rtol=1e-10,atol=1e-8)
    b=a.copy();b['x']=xnet.ravel();b['z']=znet.ravel()
    b['x_negative_flow_component']=flow(county_negative).ravel()
    b['z_negative_flow_component']=delta_z.ravel()
    cutoffs=[]
    for col in ('x','y','z'):
        lo,hi=np.quantile(b[col],[.01,.99],method='linear')
        b[col+'_percentile_1_99']=b[col].clip(lo,hi)
        cutoffs.append(dict(variable=col,lower=lo,upper=hi,scope='pooled',weight='unweighted'))
    for col in ('ccg','year','y','y_percentile_1_99','division','pop1975','fe_division_year','cluster'):
        np.testing.assert_array_equal(a[col],b[col])
    assert np.isfinite(b.select_dtypes(include='number').to_numpy()).all()
    calhoun=pd.read_csv(OUT/'corrected_gap_audit/calhoun_2014_signed_shock.csv').set_index('measure')
    np.testing.assert_allclose(b.loc[b.ccg.eq(5013)&b.year.eq(2014),'x'].iloc[0],calhoun.loc['net','x'],rtol=1e-10)
    CACHE.mkdir(parents=True,exist_ok=True)
    b.to_parquet(CACHE/'analysis_pooled.parquet',index=False)
    county_negative.rename_axis(index='ccg',columns='year').stack().rename('negative_real').reset_index().to_parquet(CACHE/'county_negative_flows.parquet',index=False)
    natnegative.rename_axis(index='cat',columns='year').stack().rename('negative_real').reset_index().to_parquet(CACHE/'national_category_negative_flows.parquet',index=False)
    pd.DataFrame(checks).to_csv(OUT/'h2_net_fixed_exposure_ew_source_checks.csv',index=False)
    pd.DataFrame(cutoffs).to_csv(OUT/'h2_net_fixed_exposure_ew_cutoffs.csv',index=False)
    (CACHE/'certification.json').write_text(json.dumps(dict(status='PASS',source_cache_sha256=baseline_sha,
        output_cache_sha256=sha(CACHE/'analysis_pooled.parquet'),n=107030,n_counties=3058,
        exposure_rule='positive',county_flow_rule='net',national_shift_rule='net',
        positive_sources_reproduce_baseline=True,negative_component_checks_pass=True,
        outcome_sample_weights_fe_unchanged=True,calhoun_independent_saved_audit_matches=True),indent=2)+'\n')
    t28=module('t28_netfixed',ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER=pf.MapDemeaner(fixef_tol=1e-12,fixef_maxiter=100000)
    saved=pd.read_csv(OUT/'h2_report_correction_comparison.csv')
    saved=saved.loc[saved.source_version.eq('corrected')&saved.weight.eq('equal county')].copy()
    assert len(saved)==2
    saved['flow_rule']='Positive';saved['estimate_source']='saved'
    rows=saved.to_dict('records')
    for variant in ('untreated','joint_pooled_percentile_1_99'):
        suffix='' if variant=='untreated' else '_percentile_1_99'
        y,x,z=(b[v+suffix].to_numpy(np.float64) for v in ('y','x','z'))
        fit=t28.fit2sls(y,x,z,np.ones(len(b)),b.fe_division_year.to_numpy(np.int64),
            np.zeros(len(b),np.int64),b.cluster.to_numpy(np.int64),singleton=False,want_F=True)
        assert fit['n']==len(b) and fit['n_dropped']==0
        rows.append(dict(flow_rule='Net, fixed positive exposure',variant=variant,weight='equal county',
            first_stage_F=fit['F'],second_stage=fit['beta'],second_stage_se=fit['se'],
            second_stage_t=fit['beta']/fit['se'],second_stage_x1e5=fit['beta']*1e5,
            second_stage_se_x1e5=fit['se']*1e5,n=len(b),n_counties=3058,estimate_source='estimated'))
    results=pd.DataFrame(rows)
    assert len(results)==4 and np.isfinite(results[['first_stage_F','second_stage_t','second_stage']]).all().all()
    results.to_csv(OUT/'h2_net_fixed_exposure_ew_comparison.csv',index=False)
    gaps=results.pivot(index='flow_rule',columns='variant',values='second_stage_x1e5')
    gaps['gap']=gaps.joint_pooled_percentile_1_99-gaps.untreated
    gaps['gap_closed_percent']=100*(1-gaps.gap/gaps.loc['Positive','gap'])
    gaps.to_csv(OUT/'h2_net_fixed_exposure_ew_gap.csv')
    assert sha(SOURCE/'analysis_pooled.parquet')==baseline_sha
    write_report(results,gaps)
    print(results[['flow_rule','variant','first_stage_F','second_stage_x1e5','second_stage_se_x1e5','second_stage_t']].to_string(index=False))
    print(gaps.to_string())


def write_report(results,gaps):
    lines=['# EW h=2: net obligations with fixed positive exposure shares','',
        'The full balanced sample contains 3,058 counties and 107,030 county-years, 1982 through 2016. '
        'No counties are excluded. Equal weights, division-year fixed effects, county clustering, '
        'employment and population denominators are unchanged.', '',
        '## Construction','',
        'Net spending sums positive and negative obligations, using the existing geography cleaning, '
        'report corrections, construction exclusion and real-dollar conversion. '
        'The observed treatment is [0.5(G_net,t + G_net,t-1) - G_net,t-2] / P_t-2. '
        'National category shifts use the same net-flow convention. '
        'The instrument weights those shifts with the unchanged county market shares of positive '
        'category spending over t-11 through t-7, then divides by P_t-2. '
        'Exposure shares are not recalculated from net spending, and neither shares nor the instrument are clipped.', '',
        'This is not the design grid net_matched specification, which also changes exposure to net spending. '
        'It is not net_shift_only, which retains positive observed county spending. '
        'The current test changes both the treatment and national shifts while holding exposure fixed.', '',
        'Pooled winsorization caps x, y and z separately at unweighted 1st and 99th percentiles. '
        'Thresholds are recomputed for net x and z; outcome values and thresholds are unchanged. '
        'Negative annual net totals are retained, not floored or treated as missing.', '',
        '## Table 1. EW estimates','',
        'Coefficients and SEs multiplied by 100,000.', '',
        '| Flow convention | Specification | Coefficient | SE | t | First-stage F |',
        '|---|---|---:|---:|---:|---:|']
    for r in results.itertuples():
        label='Raw' if r.variant=='untreated' else 'Joint pooled 1-99'
        lines.append(f'| {r.flow_rule} | {label} | {r.second_stage_x1e5:.6f} | {r.second_stage_se_x1e5:.6f} | {r.second_stage_t:.6f} | {r.first_stage_F:.6f} |')
    lines+=['','Source: `Results/instrument_normalization/h2_net_fixed_exposure_ew_comparison.csv`.','',
        '## Table 2. Winsorized minus raw coefficient','',
        '| Flow convention | Gap | Gap closed (%) |','|---|---:|---:|']
    for k,r in gaps.iterrows():
        lines.append(f'| {k} | {r.gap:.6f} | {r.gap_closed_percent:.6f} |')
    change=-gaps.loc['Net, fixed positive exposure','gap_closed_percent']
    lines+=['',f'The gap changes by {change:+.6f}% relative to the positive-flow baseline (Table 2). '
        'A positive change means the estimates move farther apart. '
        'This comparison evaluates signed flows while keeping positive exposure shares fixed; '
        'it does not test alternative exposure definitions or establish the accuracy of individual records.']
    lines+=['','Source: `Results/instrument_normalization/h2_net_fixed_exposure_ew_gap.csv`.','',
        '## Verification and scope','',
        'Positive county and national-category aggregates reconstructed from current source files '
        'match the corrected baseline for every required flow year, 1980 through 2016. '
        'Negative flows are added linearly; direct net instruments agree with baseline plus the negative-flow component. '
        'A category-by-category calculation checks that component, and the Calhoun 2014 treatment '
        'matches the separately saved signed-obligation audit. No fit drops observations. '
        'These checks verify construction, not independent accuracy of government contract records.', '',
        'The baseline cache hash is unchanged. Neither population-weighted estimates nor other horizons, '
        'county-exclusion diagnostics or presentation outputs are rebuilt. Two new EW fits are estimated; '
        'two positive-baseline rows are reused.', '',
        'Producer: `step_1/code/pipeline/107_h2_net_flows_fixed_exposure_ew.py`. '
        'Separate cache and certification: `data/temp/h2_net_flows_fixed_exposure_ew/`. '
        'Source checks and capping cutoffs are saved in the `h2_net_fixed_exposure_ew_*` result files.', '']
    body='\n'.join(lines);assert '\u2014' not in body
    path=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/EW_NET_FLOWS_FIXED_EXPOSURE_TEST.md'
    path.write_text(body);assert path.read_text()==body


def after_nine():
    """Compare net and positive flows on the identical saved nine-excluded sample."""
    assert json.loads((CACHE/'certification.json').read_text())['status']=='PASS'
    b=pd.read_parquet(CACHE/'analysis_pooled.parquet')
    reference=pd.read_csv(OUT/'corrected_gap_audit/h2_winsorization_leaveout_estimates_original_groups.csv')
    reference=reference.loc[reference.n_excluded.eq(9)&reference.weight.eq('equal county')].copy()
    assert len(reference)==2 and reference.excluded_ccg.nunique()==1
    excluded=list(map(int,reference.excluded_ccg.iloc[0].split(';')))
    b=b.loc[~b.ccg.isin(excluded)]
    assert len(b)==106715 and b.ccg.nunique()==3049 and b.groupby('ccg').size().eq(35).all()
    t28=module('t28_net9',ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER=pf.MapDemeaner(fixef_tol=1e-12,fixef_maxiter=100000)
    reference['flow_rule']='Positive';reference['estimate_source']='saved'
    reference['second_stage_se_x1e5']=reference.second_stage_se*1e5
    rows=reference.to_dict('records')
    for variant in ('untreated','joint_pooled_percentile_1_99'):
        suffix='' if variant=='untreated' else '_percentile_1_99'
        y,x,z=(b[v+suffix].to_numpy(np.float64) for v in ('y','x','z'))
        fit=t28.fit2sls(y,x,z,np.ones(len(b)),b.fe_division_year.to_numpy(np.int64),
            np.zeros(len(b),np.int64),b.cluster.to_numpy(np.int64),singleton=False,want_F=True)
        assert fit['n']==len(b) and fit['n_dropped']==0
        rows.append(dict(flow_rule='Net, fixed positive exposure',variant=variant,weight='equal county',
            first_stage_F=fit['F'],second_stage=fit['beta'],second_stage_se=fit['se'],
            second_stage_t=fit['beta']/fit['se'],second_stage_x1e5=fit['beta']*1e5,
            second_stage_se_x1e5=fit['se']*1e5,n=len(b),n_counties=3049,n_excluded=9,
            excluded_ccg=';'.join(map(str,excluded)),estimate_source='estimated'))
    results=pd.DataFrame(rows)
    results.to_csv(OUT/'h2_net_fixed_exposure_ew_after_nine.csv',index=False)
    gaps=results.pivot(index='flow_rule',columns='variant',values='second_stage_x1e5')
    gaps['gap']=gaps.joint_pooled_percentile_1_99-gaps.untreated
    gaps['gap_closed_percent']=100*(1-gaps.gap/gaps.loc['Positive','gap'])
    gaps.to_csv(OUT/'h2_net_fixed_exposure_ew_after_nine_gap.csv')
    print(results[['flow_rule','variant','first_stage_F','second_stage_x1e5','second_stage_t']].to_string(index=False))
    print(gaps.to_string())


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--after-nine',action='store_true')
    if parser.parse_args().after_nine:
        after_nine()
    else:
        main()
