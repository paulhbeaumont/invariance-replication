"""Compare raw and pooled-capped EW h=2 IV with meat category 8905 excluded.

Derives from producers 105 (certified corrected inputs), 98 (category
construction), and 102 (EW diagnostics). Removes the category's additive
components from both observed spending and the instrument throughout the
sample. Other categories' market shares are not renormalized. No counties
are excluded. Baseline coefficients are read, not estimated.
"""
# mpgr-run-class: cache_rebuild
# mpgr-rebuild-reason: treatment_changed
# mpgr-cache: data/temp/report_correction_rebuild/analysis_pooled.parquet
# mpgr-output-cache: data/temp/h2_exclude_meat_ew/analysis_pooled.parquet
# mpgr-design-keys: ccg year fe_division_year cluster
# mpgr-new-estimates: 2
# mpgr-batch-approved: explicit
# mpgr-variation-scope: county|h2|EW|full-sample|exclude-8905|raw-pooled1-99

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
sys.path[:0] = [str(PIPELINE.parent/'py'), str(PIPELINE), str(PIPELINE/'lib')]
import dcads_common as dc
import store

SOURCE = dc.MACRO_VERSION_TEMP_DIR/'report_correction_rebuild'
CACHE = dc.MACRO_VERSION_TEMP_DIR/'h2_exclude_meat_ew'
OUT = dc.MACRO_RESULTS_ROOT/'instrument_normalization'
YEARS = list(range(1966, 2019))
TARGETS = list(range(1982, 2017))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((SOURCE/'certification.json').read_text())['status'] == 'PASS'
    baseline_path = SOURCE/'analysis_pooled.parquet'
    baseline_sha = sha(baseline_path)
    a = pd.read_parquet(baseline_path)
    units = np.sort(a.ccg.unique())
    assert len(a) == 107030 and len(units) == 3058
    np.testing.assert_array_equal(a.ccg, np.repeat(units, len(TARGETS)))
    np.testing.assert_array_equal(a.year, np.tile(TARGETS, len(units)))
    assert not a.duplicated(['ccg','year']).any()
    # Only the unchanged population denominator is read from the county store.
    p = pd.read_parquet(store.PIPE/'county_panel_pipeline.parquet', columns=['ccg','year','P'])
    population = p.pivot(index='ccg', columns='year', values='P').reindex(units)
    assert np.isfinite(population[TARGETS].to_numpy()).all()
    frames=[]
    with duckdb.connect() as con:
        for year in YEARS:
            path=SOURCE/'annual_category'/f'fy={year}.parquet'
            d=con.execute("SELECT ccg, sum(dollars) dollars FROM read_parquet(?) WHERE cat='8905' GROUP BY ccg",
                          [str(path)]).df()
            d['year']=year
            frames.append(d)
    meat=pd.concat(frames,ignore_index=True)
    assert not meat.duplicated(['ccg','year']).any() and meat.dollars.ge(0).all()
    national=meat.groupby('year').dollars.sum().reindex(YEARS,fill_value=0)
    wide=meat.pivot(index='ccg',columns='year',values='dollars').reindex(index=units,columns=YEARS).fillna(0)
    xm=np.zeros((len(units),len(TARGETS)));zm=np.zeros_like(xm)
    diagnostics=[]
    for j,t in enumerate(TARGETS):
        pre=list(range(t-11,t-6))
        denominator=float(national[pre].mean())
        shift=float(.5*(national[t]+national[t-1])-national[t-2])
        pbase=population[t-2].to_numpy(np.float64)
        assert np.isfinite(pbase).all() and (pbase>0).all()
        local_pre=wide[pre].mean(axis=1).to_numpy()
        predicted=local_pre*shift/denominator if denominator>0 else np.zeros(len(units))
        xm[:,j]=(.5*(wide[t].to_numpy()+wide[t-1].to_numpy())-wide[t-2].to_numpy())/pbase
        zm[:,j]=predicted/pbase
        # The national denominator includes all source counties, not just the regression sample.
        all_pre=meat.loc[meat.year.isin(pre)].groupby('ccg').dollars.sum()/5
        allocated=(all_pre*shift/denominator).sum() if denominator>0 else 0
        np.testing.assert_allclose(allocated,shift if denominator>0 else 0,rtol=1e-10,atol=.001)
        diagnostics.append(dict(year=t,exposure_start=t-11,exposure_end=t-7,
            national_exposure=denominator,national_shift=shift,allocated=allocated))
    # Reuse the canonical category engine, restricted to the one removed component.
    p98=module('p98_meat_check',PIPELINE/'98_h2_all_category_market_share_design_grid.py')
    p98.cache_path=lambda rule,floor,year:SOURCE/'annual_category'/f'fy={year}.parquet'
    check,_=p98.build_instrument('positive',0,'positive',0,4,units,population,5,keep_categories={'8905'})
    np.testing.assert_allclose(zm,check,rtol=1e-10,atol=1e-8)
    spending=pd.read_parquet(SOURCE/'county_spending_positive.parquet').pivot(index='ccg',columns='year',values='dollars').reindex(units)
    pbase=population[[t-2 for t in TARGETS]].to_numpy()
    base_x=(.5*(spending[TARGETS].to_numpy()+spending[[t-1 for t in TARGETS]].to_numpy())-
            spending[[t-2 for t in TARGETS]].to_numpy())/pbase
    np.testing.assert_allclose(base_x.ravel(),a.x,rtol=1e-12,atol=1e-10)
    remaining=spending.reindex(columns=YEARS)-wide
    assert remaining.min().min()>=-.01
    new_x=(.5*(remaining[TARGETS].to_numpy()+remaining[[t-1 for t in TARGETS]].to_numpy())-
           remaining[[t-2 for t in TARGETS]].to_numpy())/pbase
    np.testing.assert_allclose(new_x.ravel(),a.x.to_numpy()-xm.ravel(),rtol=1e-10,atol=1e-8)
    b=a.copy()
    b['x_meat']=xm.ravel();b['z_meat']=zm.ravel()
    b['x']=new_x.ravel();b['z']=a.z.to_numpy()-zm.ravel()
    cutoffs=[]
    for col in ('x','y','z'):
        lo,hi=np.quantile(b[col],[.01,.99],method='linear')
        b[f'{col}_percentile_1_99']=b[col].clip(lo,hi)
        cutoffs.append(dict(variable=col,lower=lo,upper=hi,scope='pooled',weight='unweighted'))
    for col in ('ccg','year','y','y_percentile_1_99','division','pop1975','fe_division_year','cluster'):
        np.testing.assert_array_equal(b[col],a[col])
    assert np.isfinite(b.select_dtypes(include='number').to_numpy()).all()
    CACHE.mkdir(parents=True,exist_ok=True)
    b.to_parquet(CACHE/'analysis_pooled.parquet',index=False)
    meat.to_parquet(CACHE/'meat_category_county_year.parquet',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'h2_exclude_meat_ew_components.csv',index=False)
    pd.DataFrame(cutoffs).to_csv(OUT/'h2_exclude_meat_ew_cutoffs.csv',index=False)
    certificate=dict(status='PASS',run_class='cache_rebuild',reason='treatment_changed',excluded_category='8905',
        n=len(b),n_counties=len(units),n_counties_excluded=0,source_cache_sha256=baseline_sha,
        output_cache_sha256=sha(CACHE/'analysis_pooled.parquet'),
        outcome_weights_sample_fe_unchanged=True,category_component_matches_canonical_engine=True,
        spending_matches_direct_subtraction=True,category_shares_renormalized=False)
    (CACHE/'certification.json').write_text(json.dumps(certificate,indent=2)+'\n')
    print('PASS: category removal certified; starting two EW fits.',flush=True)
    t28=module('t28_meat',ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER=pf.MapDemeaner(fixef_tol=1e-12,fixef_maxiter=100000)
    saved=pd.read_csv(OUT/'h2_report_correction_comparison.csv')
    saved=saved.loc[saved.source_version.eq('corrected')&saved.weight.eq('equal county')].copy()
    assert len(saved)==2
    saved['coverage']='Baseline';saved['estimate_source']='saved'
    rows=saved.to_dict('records')
    for variant in ('untreated','joint_pooled_percentile_1_99'):
        suffix='' if variant=='untreated' else '_percentile_1_99'
        y,x,z=(b[col+suffix].to_numpy(np.float64) for col in ('y','x','z'))
        fit=t28.fit2sls(y,x,z,np.ones(len(b)),b.fe_division_year.to_numpy(np.int64),
            np.zeros(len(b),np.int64),b.cluster.to_numpy(np.int64),singleton=False,want_F=True)
        assert fit['n_dropped']==0 and fit['n']==len(b)
        rows.append(dict(coverage='Exclude 8905',source_version='corrected',weight='equal county',variant=variant,
            first_stage_F=fit['F'],second_stage=fit['beta'],second_stage_se=fit['se'],
            second_stage_t=fit['beta']/fit['se'],second_stage_x1e5=fit['beta']*1e5,
            second_stage_se_x1e5=fit['se']*1e5,n=fit['n'],n_counties=len(units),estimate_source='estimated'))
    results=pd.DataFrame(rows)
    assert len(results)==4 and np.isfinite(results[['first_stage_F','second_stage_t','second_stage']]).all().all()
    results.to_csv(OUT/'h2_exclude_meat_ew_comparison.csv',index=False)
    gaps=results.pivot(index='coverage',columns='variant',values='second_stage_x1e5')
    gaps['gap']=gaps.joint_pooled_percentile_1_99-gaps.untreated
    gaps['gap_closed_percent']=100*(1-gaps.gap/gaps.loc['Baseline','gap'])
    gaps.to_csv(OUT/'h2_exclude_meat_ew_gap.csv')
    assert sha(baseline_path)==baseline_sha
    write_memo(results,gaps)
    print(results[['coverage','variant','first_stage_F','second_stage_x1e5','second_stage_se_x1e5','second_stage_t']].to_string(index=False))
    print(gaps.to_string())


def write_memo(results,gaps):
    lines=['# EW h=2: excluding meat category 8905','',
        'This diagnostic excludes category 8905 from both observed spending and the instrument. '
        'It uses the full balanced sample, not the nine-county-removal diagnostic. '
        'All 3,058 counties and 107,030 county-years (1982 to 2016) are retained. '
        'Equal weights, division-year effects, county clustering, employment and population denominators are unchanged.', '',
        '## Construction','',
        'Let G* = G - G_8905 at every county-year. The treatment is '
        '[0.5(G* at t + G* at t-1) - G* at t-2] / population at t-2. '
        'The instrument is the baseline instrument minus the category-8905 component, '
        'using its national spending change and county market share measured over t-11 through t-7. '
        'Remaining category market shares are unchanged, with no renormalization across categories. '
        'The removal applies to every source year, 1966 to 2018, including exposure years. '
        'It excludes real meat procurement as well as records affected by reporting changes; '
        'it neither imputes missing purchases nor fixes every DeCA coverage problem.', '',
        'Pooled winsorization separately caps x, y and z at the unweighted 1st/99th percentiles. '
        'The rule is fixed; x and z thresholds are recomputed for the changed variables. '
        'The outcome and its thresholds remain exactly unchanged.', '',
        '## Table 1. First-stage F and second-stage estimates','',
        'Coefficients and standard errors multiplied by 100,000.', '',
        '| Coverage | Specification | Coefficient | SE | t | First-stage F |',
        '|---|---|---:|---:|---:|---:|']
    for r in results.itertuples():
        label='Raw' if r.variant=='untreated' else 'Pooled 1-99, x/y/z'
        lines.append(f'| {r.coverage} | {label} | {r.second_stage_x1e5:.6f} | {r.second_stage_se_x1e5:.6f} | {r.second_stage_t:.6f} | {r.first_stage_F:.6f} |')
    lines+=['','Source: `Results/instrument_normalization/h2_exclude_meat_ew_comparison.csv`.','',
        '## Table 2. Winsorized minus raw coefficient','',
        '| Coverage | Gap | Gap closed (%) |','|---|---:|---:|']
    for name,r in gaps.iterrows():
        lines.append(f'| {name} | {r.gap:.6f} | {r.gap_closed_percent:.6f} |')
    change=100*(gaps.loc['Exclude 8905','gap']/gaps.loc['Baseline','gap']-1)
    lines+=['','Source: `Results/instrument_normalization/h2_exclude_meat_ew_gap.csv`.','',
        f'The coefficient gap changes by {change:+.6f}% relative to baseline (Table 2). '
        'A positive value means the raw and winsorized estimates are farther apart. '
        'This test does not establish the effect of repairing missing DeCA purchases: '
        'it measures the effect of excluding the entire meat category.', '',
        '## Verification and reproduction','',
        'The removed instrument component matches producer 98 restricted to category 8905. '
        'The treatment matches direct reconstruction from county spending less meat spending. '
        'National category allocation includes all source counties, including counties outside the regression sample. '
        'Sample keys, outcome, weights, fixed effects and clustering match the certified input cache. '
        'Neither fit drops observations. The baseline cache hash remains unchanged. '
        'These are construction checks, not independent validation of the underlying government records.', '',
        'Run `step_1/code/pipeline/106_h2_exclude_meat_ew.py`. '
        'Inputs: `data/temp/report_correction_rebuild/`. '
        'Separate output cache and certification: `data/temp/h2_exclude_meat_ew/`. '
        'Two new EW specifications are estimated; two baseline rows are reused. '
        'No population-weighted, county-exclusion, other-horizon, or presentation outputs are rebuilt.', '']
    body='\n'.join(lines)
    assert '\u2014' not in body
    path=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/EW_EXCLUDE_MEAT_TEST.md'
    path.write_text(body)
    assert path.read_text()==body


if __name__=='__main__':
    main()
