"""Rebuild h=2 inputs for verified report cancellations, with fixed design.

County outcome: two-year employment midpoint growth. Treatment: mean real
spending in t-1,t minus t-2 spending, divided by t-2 population. Instrument:
all nonconstruction four-character categories, positive actions, no floor,
rolling exposure over t-11 through t-7, no clipping. County clusters and
division-year effects; equal county and fixed-1975 population weights.

Only the corrected source years are rebuilt. Existing comparison coefficients
are loaded, not re-estimated. The dedicated cache does not replace presentation
inputs or unrelated design-grid results.
"""
# mpgr-run-class: cache_rebuild
# mpgr-rebuild-reason: treatment_changed
# mpgr-cache: county_xy_winsorization/analysis_pooled.parquet
# mpgr-new-estimates: 4
# mpgr-batch-approved: explicit
# mpgr-variation-scope: county|h2|reported-vs-corrected|EW-P1975|raw-pooled1-99

import importlib.util
import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
sys.path[:0] = [str(PIPELINE.parent/'py'),str(PIPELINE),str(PIPELINE/'lib')]
import contracts
import harmonize
import dcads_common as dc
import report_corrections as rc
import store

CACHE = dc.MACRO_VERSION_TEMP_DIR/'report_correction_rebuild'
OUT = dc.MACRO_RESULTS_ROOT/'instrument_normalization'
BACKUP = dc.MACRO_VERSION_TEMP_DIR/'contract_report_correction_application/uncorrected_fiscal_files'
MEMO = dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/REPORT_CORRECTION_RESULTS.md'
VARIANTS = ['untreated','joint_pooled_percentile_1_99']


class UncorrectedSourceView:
    """Route preserved years to backups and unchanged years to production."""
    def __init__(self,current):
        self.current=current

    def __truediv__(self,name):
        path=BACKUP/name
        return path if path.exists() else self.current/name


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextmanager
def source_directory(path):
    """Reuse canonical readers against a preserved source snapshot."""
    current=dc.OUT_DIR
    try:
        dc.OUT_DIR=path
        yield
    finally:
        dc.OUT_DIR=current


def source_aggregates(year,mapping):
    d,c=contracts.county_actions(year,mapping,drop_construction=True)
    keep=d>0
    county=(pd.DataFrame({'ccg':c[keep],'dollars':d[keep]*dc.deflate(year)})
            .groupby('ccg',sort=False).dollars.sum())
    d,c,k=contracts.category_actions(year,mapping,drop_construction=True,harmonize=True)
    keep=d>0
    cats=(pd.DataFrame({'ccg':c[keep],'cat':k[keep],'dollars':d[keep]*dc.deflate(year)})
          .groupby(['ccg','cat'],sort=False,as_index=False).dollars.sum())
    return county,cats


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    frozen=CACHE/'uncorrected_comparison'
    frozen.mkdir(exist_ok=True)
    old_panel_path=dc.MACRO_DATA_ROOT/'temp/county_xy_winsorization/analysis_pooled.parquet'
    old_results_path=OUT/'h2_county_xyz_winsorization_pooled.csv'
    for path in (old_panel_path,old_results_path):
        dest=frozen/path.name
        if dest.exists():
            if rc.sha256(dest)!=rc.sha256(path):
                raise ValueError('Uncorrected comparison snapshot differs from current input')
        else:
            shutil.copy2(path,dest)
    old=pd.read_parquet(frozen/old_panel_path.name)
    saved=pd.read_csv(frozen/old_results_path.name)
    saved=saved.loc[saved.variant.isin(VARIANTS)].copy()
    assert len(saved)==4 and not saved.duplicated(['weight','variant']).any()
    p98=module('p98_report_correction',PIPELINE/'98_h2_all_category_market_share_design_grid.py')
    p100=module('p100_report_correction',PIPELINE/'100_h2_full_instrument_design_grid.py')
    t28=module('t28_report_correction',ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER=pf.MapDemeaner(fixef_tol=1e-12,fixef_maxiter=100000)
    county=store.load_county_panel()
    units,pivots=store.load_pivots(county,cols=('P',))
    years=list(range(1966,2019))
    spending=store.load_con('positive').reindex(index=units,columns=years)
    rawpanel=p98.make_panel(spending,county,units)
    for col in ['x','y','pop1975']:
        np.testing.assert_allclose(rawpanel[col],old[col],rtol=1e-12,atol=1e-12)
    for col in ['ccg','year','division']:
        np.testing.assert_array_equal(rawpanel[col],old[col])
    assert len(old)==107030 and old.ccg.nunique()==3058
    assert not old.duplicated(['ccg','year']).any()
    # Saved annual category files must reproduce the saved instrument before
    # any corrected source is substituted. No uncorrected regression is run.
    p100.path_for=lambda cover,rule,floor,year:p98.cache_path(rule,floor,year)
    before_sparse=p100.load_sparse('exclude_construction','positive',0,4,units)
    before_z,_=p100.instruments(before_sparse,before_sparse,pivots['P'],'rolling',5)
    np.testing.assert_allclose(before_z.ravel(),old.z,rtol=1e-10,atol=1e-8)
    print('PASS: saved spending and category inputs reproduce the uncorrected analysis cache.',flush=True)
    category_dir=CACHE/'annual_category'
    category_dir.mkdir(exist_ok=True)
    checks=[]
    mapping,_,_=dc.load_ccg_map()
    original_sources=UncorrectedSourceView(dc.OUT_DIR)
    harmonize._HARMONIZATION=None
    with source_directory(original_sources):
        original_harmonization=harmonize.load_harmonization()
    harmonize._HARMONIZATION=None
    corrected_harmonization=harmonize.load_harmonization()
    # Namespace maps depend on unchanged years; only the pre-1979 fallback
    # allocation uses affected 1979-1981 spending and must be recomputed.
    assert original_harmonization[:3]==corrected_harmonization[:3]
    (CACHE/'service_fallback_weights.json').write_text(json.dumps(dict(
        uncorrected=original_harmonization[3],corrected=corrected_harmonization[3]),indent=2)+'\n')
    applied=pd.read_csv(OUT/'correction_cleaning_applied_1966_1983.csv',dtype={'fiscal_file':str}).set_index('fiscal_file')
    for year in years:
        target=category_dir/f'fy={year}.parquet'
        prior=p98.cache_path('positive',0,year)
        if year>1983:
            if not target.exists():
                shutil.copy2(prior,target)
            if rc.sha256(target)!=rc.sha256(prior):
                raise ValueError('Unchanged annual category cache mismatch')
            continue
        current=dc.OUT_DIR/f'fy{year}.parquet'
        if rc.sha256(current)!=applied.loc[str(year),'corrected_sha256']:
            raise ValueError(f'Corrected source certification mismatch: {year}')
        harmonize._HARMONIZATION=original_harmonization
        with source_directory(original_sources):
            original_county,original_cats=source_aggregates(year,mapping)
        baseline=spending[year].to_numpy()
        np.testing.assert_allclose(original_county.reindex(units).fillna(0),baseline,rtol=1e-10,atol=.01)
        prior_cats=pd.read_parquet(prior).set_index(['ccg','cat']).dollars.sort_index()
        original_cats=original_cats.set_index(['ccg','cat']).dollars.sort_index()
        pd.testing.assert_series_equal(prior_cats,original_cats,check_exact=False,rtol=1e-10,atol=.01)
        harmonize._HARMONIZATION=corrected_harmonization
        rebuilt_county,rebuilt_cats=source_aggregates(year,mapping)
        spending[year]=rebuilt_county.reindex(units).fillna(0)
        rebuilt_cats.to_parquet(target,index=False)
        delta=spending[year].to_numpy()-baseline
        if delta.max(initial=0)>.01:
            raise ValueError('Neutralizing positive legs cannot increase positive county spending')
        checks.append(dict(year=year,changed_counties=int((abs(delta)>.01).sum()),
                           reduction_real_dollars=float(-delta.sum()),source_sha256=rc.sha256(current)))
        print(f'Rebuilt {year}: {(abs(delta)>.01).sum()} county spending cells changed.',flush=True)
    spending.stack().rename('dollars').reset_index().to_parquet(CACHE/'county_spending_positive.parquet',index=False)
    pd.DataFrame(checks).to_csv(OUT/'h2_report_correction_spending_changes.csv',index=False)
    p100.path_for=lambda cover,rule,floor,year:category_dir/f'fy={year}.parquet'
    after_sparse=p100.load_sparse('exclude_construction','positive',0,4,units)
    z,diag=p100.instruments(after_sparse,after_sparse,pivots['P'],'rolling',5)
    # Cross-check the reused sparse engine against the canonical construction.
    p98.cache_path=lambda rule,floor,year:category_dir/f'fy={year}.parquet'
    zcheck,_=p98.build_instrument('positive',0,'positive',0,4,units,pivots['P'],5)
    np.testing.assert_allclose(z,zcheck,rtol=1e-10,atol=1e-8)
    panel=p98.make_panel(spending,county,units)
    panel['z']=z.ravel()
    for col in ['ccg','year','y','division','pop1975']:
        np.testing.assert_array_equal(panel[col],old[col])
    panel['fe_division_year']=old.fe_division_year
    panel['cluster']=old.cluster
    np.testing.assert_allclose(panel.loc[panel.year>=1986,'x'],old.loc[old.year>=1986,'x'],rtol=0,atol=0)
    np.testing.assert_allclose(panel.loc[panel.year>=1995,'z'],old.loc[old.year>=1995,'z'],rtol=1e-10,atol=1e-8)
    changes=[]
    for year,idx in panel.groupby('year',sort=False).indices.items():
        changes.append(dict(year=year,n_x_changed=int((abs(panel.x.iloc[idx].to_numpy()-old.x.iloc[idx].to_numpy())>1e-8).sum()),
                            n_z_changed=int((abs(panel.z.iloc[idx].to_numpy()-old.z.iloc[idx].to_numpy())>1e-8).sum())))
    pd.DataFrame(changes).to_csv(OUT/'h2_report_correction_analysis_changes.csv',index=False)
    cutoffs=[]
    for col in ['x','y','z']:
        lo,hi=np.quantile(panel[col],[.01,.99],method='linear')
        panel[f'{col}_percentile_1_99']=panel[col].clip(lo,hi)
        cutoffs.append(dict(variable=col,lower=lo,upper=hi,scope='pooled',weight='unweighted'))
    assert np.isfinite(panel.drop(columns='division').to_numpy(dtype=float)).all()
    panel.to_parquet(CACHE/'analysis_pooled.parquet',index=False)
    pd.DataFrame(diag).to_csv(OUT/'h2_report_correction_instrument_diagnostics.csv',index=False)
    pd.DataFrame(cutoffs).to_csv(OUT/'h2_report_correction_pooled_cutoffs.csv',index=False)
    (CACHE/'certification.json').write_text(json.dumps(dict(status='PASS',n=107030,n_counties=3058,
        source_rule_sha256=rc.sha256(rc.LEDGER),outcome_weights_sample_unchanged=True,
        original_inputs_reproduced=True,instrument_engines_agree=True,source_rebuilt_through=1983),indent=2)+'\n')
    rows=saved.assign(source_version='uncorrected',estimate_source='saved').to_dict('records')
    fe=panel.fe_division_year.to_numpy(np.int64)
    cl=panel.cluster.to_numpy(np.int64)
    nofe=np.zeros(len(panel),np.int64)
    for variant in VARIANTS:
        suffix='' if variant=='untreated' else '_percentile_1_99'
        x,y,z=(panel[v+suffix].to_numpy(np.float64) for v in ['x','y','z'])
        for weight in ['equal county','fixed 1975 population']:
            w=np.ones(len(panel)) if weight=='equal county' else panel.pop1975.to_numpy(np.float64)
            first=t28.fit2sls(x,z,z,w,fe,nofe,cl,singleton=False,want_F=False)
            fit=t28.fit2sls(y,x,z,w,fe,nofe,cl,singleton=False)
            assert first['n']==fit['n']==len(panel) and first['n_dropped']==fit['n_dropped']==0
            np.testing.assert_allclose(fit['F'],(first['beta']/first['se'])**2,rtol=1e-10)
            rows.append(dict(source_version='corrected',variant=variant,weight=weight,
                first_stage=first['beta'],first_stage_se=first['se'],first_stage_F=fit['F'],
                second_stage=fit['beta'],second_stage_se=fit['se'],second_stage_t=fit['beta']/fit['se'],
                second_stage_x1e5=fit['beta']*1e5,second_stage_se_x1e5=fit['se']*1e5,
                n=len(panel),n_counties=3058,estimate_source='estimated'))
    results=pd.DataFrame(rows)
    assert len(results)==8 and not results.duplicated(['source_version','variant','weight']).any()
    assert np.isfinite(results[['first_stage_F','second_stage_x1e5','second_stage_t']]).all().all()
    path=OUT/'h2_report_correction_comparison.csv'
    results.to_csv(path,index=False)
    assert len(pd.read_csv(path))==8
    gaps=results.pivot(index=['source_version','weight'],columns='variant',values='second_stage_x1e5').reset_index()
    gaps['winsorization_gap']=gaps.joint_pooled_percentile_1_99-gaps.untreated
    baseline=gaps.loc[gaps.source_version.eq('uncorrected')].set_index('weight').winsorization_gap
    gaps['gap_reduction_percent']=100*(1-gaps.winsorization_gap/gaps.weight.map(baseline))
    gaps.to_csv(OUT/'h2_report_correction_winsorization_gap.csv',index=False)
    write_memo(results,gaps)
    print(results[['source_version','variant','weight','first_stage_F','second_stage_x1e5','second_stage_t']].to_string(index=False),flush=True)
    print(gaps.to_string(index=False),flush=True)


def write_memo(results,gaps):
    lines=['# H=2 results with verified report cancellations','',
        'The comparison changes only treatment and instrument inputs affected by the approved 1966-1983 '
        'report cancellations. The outcome, 3,058-county sample (107,030 county-years), 1982-2016 period, '
        'division-year effects and county clustering are fixed. Population weights are fixed in 1975. '
        'The instrument uses all nonconstruction four-character categories, positive actions, no action floor, '
        'rolling exposure over t-11 through t-7, and no clipping. The annual fiscal-file convention is unchanged; '
        'the separate 1976 transition-quarter file is not added to annual spending.', '',
        'The canonical pre-1979 service-code fallback allocation uses 1979-1981 spending; '
        'these data-based weights are recalculated from corrected spending. The category crosswalk files '
        'and namespace relabeling rules are held fixed.', '',
        'Pooled winsorization caps x, y and z separately at unweighted full-sample 1st and 99th percentiles, '
        'using the same algorithm for each source version. Cutoffs are recomputed from the corrected inputs. '
        'Four uncorrected estimates are loaded from saved results; four corrected specifications are estimated.', '',
        '## First and second stages','',
        'Coefficient and standard error are multiplied by 100,000.','',
        '| Source | Treatment | Weight | First-stage F | Coefficient | Standard error | t |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in results.itertuples():
        label='No winsorization' if r.variant=='untreated' else 'Pooled 1-99, x/y/z'
        lines.append(f'| {r.source_version} | {label} | {r.weight} | {r.first_stage_F:.6f} | {r.second_stage_x1e5:.6f} | {r.second_stage_se_x1e5:.6f} | {r.second_stage_t:.6f} |')
    lines+=['','Source: `Results/instrument_normalization/h2_report_correction_comparison.csv`.','',
        '## Winsorized minus unwinsorized coefficient','',
        '| Source | Weight | Gap | Gap reduction (%) |','|---|---|---:|---:|']
    for r in gaps.itertuples():
        lines.append(f'| {r.source_version} | {r.weight} | {r.winsorization_gap:.6f} | {r.gap_reduction_percent:.6f} |')
    lines+=['','Source: `h2_report_correction_winsorization_gap.csv`.','',
        '## Verification and scope','',
        'Preserved uncorrected source files reproduce the original annual spending and category caches. '
        'Those caches reproduce the saved uncorrected analysis inputs. Corrected instruments agree between '
        'the canonical annual construction and the sparse construction. The outcome, weights and sample '
        'match exactly. Treatment is unchanged from 1986 onward and the instrument from 1995 onward. '
        'No observations are dropped in estimation. Diagnostics are in the `h2_report_correction_*` CSVs.', '',
        'Corrected spending, annual category inputs and the analysis panel are in '
        '`data/temp/report_correction_rebuild/`. This is the corrected h=2 comparison cache. '
        'Presentation inputs, other horizons, full design grids, BHJ/GPSS diagnostics and LP results '
        'have not been rebuilt. Their saved results remain on their recorded input versions.', '']
    MEMO.write_text('\n'.join(lines))
    assert MEMO.read_text().startswith('# H=2')
    application=module('application104_doc',PIPELINE/'104_apply_contract_report_corrections.py')
    application.write_memo(pd.read_csv(OUT/'correction_cleaning_applied_1966_1983.csv',dtype={'fiscal_file':str}))
    state=json.loads(application.STATE.read_text())
    state['downstream_status']='Corrected h2 comparison rebuilt: data/temp/report_correction_rebuild/. Other horizons, grids, diagnostics, LPs and presentation caches not rebuilt.'
    application.STATE.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--report-only',action='store_true')
    if parser.parse_args().report_only:
        write_memo(pd.read_csv(OUT/'h2_report_correction_comparison.csv'),
                   pd.read_csv(OUT/'h2_report_correction_winsorization_gap.csv'))
    else:
        main()
