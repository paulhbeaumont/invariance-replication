"""Attribute the pooled 1/99 winsorization coefficient change to county-years.

The accounting path is raw -> cap x -> cap x,y -> cap x,y,z. This path is
descriptive and order-dependent, not a causal decomposition. County-year
contributions at each step are zero unless that variable is capped. They
sum exactly to the coefficient difference, allowing for the same weighted
division-year fixed effects used in the reported estimates. The default
accounting holds the sample fixed and does not infer standard errors.
The --leaveouts branch compares raw and pooled-capped 2SLS after removing
whole county time series, holding cached cutoffs and instruments fixed.

Let z0_tilde be the residualized raw instrument, D_x = z0_tilde' W x_cap,
D_final = zcap_tilde' W xcap, and r_xy = M(ycap-beta_xy*xcap). Contributions:
  x: -beta_raw * w*z0_tilde*(xcap-x) / D_x
  y:             w*z0_tilde*(ycap-y) / D_x
  z:             w*(zcap-z)*r_xy / D_final
The instrument-step residualization assigns the change to actual capped
instrument observations, rather than uncapped rows whose FE means change.
"""

# mpgr-run-class: estimate_only (--leaveouts, --next-group, --all-counties, --gap50, --original-groups, --after-nine-fits, --ew-focus); cache_augment (accounting and source diagnostics)
# mpgr-after-nine-new-estimates: 32 (6 individual additions and 2 frozen unions, EW/P1975, raw/capped)
# mpgr-ew-focus-new-estimates: 4 (EW-only top-three and top-five unions, raw/capped; remaining comparisons reused)
# mpgr-original-groups-new-estimates: 8 (original four and nine counties, EW/P1975, raw/capped)
# mpgr-cache: county_xy_winsorization/analysis_pooled.parquet
# mpgr-corrected-cache: macro/data/temp/report_correction_rebuild/analysis_pooled.parquet (--corrected)
# mpgr-new-estimates: 20 (--leaveouts); 4 (--next-group); 0 (accounting and source diagnostics)
# mpgr-all-counties-new-estimates: 12240 (3057 uncached county omissions x 4 fits + 3 joint omissions x 2 weights x 2 fits)
# mpgr-gap50-new-estimates: 232 (12 group endpoints, 200 single-county hybrids, 16 group hybrids, 4 full-sample hybrids)
# mpgr-batch-approved: explicit
# mpgr-variation-scope: county|h2|pooled1-99|exact-cell-accounting|EW-P1975
# mpgr-own-shift-run-class: cache_rebuild (required_key_absent; own component only)
# mpgr-own-shift-new-estimates: 0
# mpgr-own-shift-fits-run-class: estimate_only
# mpgr-own-shift-fits-new-estimates: 2
# mpgr-own-shift-fits-cache: report_correction_rebuild/ew_own_shift_components.parquet
# mpgr-own-shift-fits-scope: county|h2|EW|nine-excluded|division-year|county-cluster|raw-pooled1-99
# mpgr-first-stage-tail-run-class: cache_augment
# mpgr-first-stage-tail-new-estimates: 0 (saved raw and joint-capped specifications)

import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
sys.path[:0] = [str(PIPELINE.parent/'py'), str(PIPELINE), str(PIPELINE/'lib')]
import dcads_common as dc
import store

OUTDIR = dc.MACRO_RESULTS_ROOT/'instrument_normalization'
CACHE = dc.MACRO_DATA_ROOT/'temp/county_xy_winsorization'
CORRECTED = False
REFERENCE = OUTDIR/'h2_county_xyz_winsorization_pooled.csv'
STATE_NAMES = dict(zip(
    [1,2,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,53,54,55,56],
    ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','District of Columbia','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming']))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FOUR_COUNTIES = [48015,23023,51700,29510]


def spending_panel():
    if CORRECTED:
        p=pd.read_parquet(CACHE/'county_spending_positive.parquet')
        return p.pivot(index='ccg',columns='year',values='dollars')
    return store.load_con('positive')


def main(after_four=False, after_nine=False):
    a = pd.read_parquet(CACHE/'analysis_pooled.parquet')
    if len(a) != 107030 or a.duplicated(['ccg','year']).any():
        raise SystemExit('FAIL: invalid audit sample')
    reference = pd.read_csv(REFERENCE)
    if CORRECTED:
        reference=reference.loc[reference.source_version.eq('corrected')]
    suffix = '_after_nine' if after_nine else ('_after_four' if after_four else '')
    removed=FOUR_COUNTIES if after_four else []
    if after_nine:
        if not CORRECTED or after_four:
            raise SystemExit('FAIL: after-nine accounting requires corrected inputs and no other exclusion flag')
        reference=pd.read_csv(OUTDIR/'h2_winsorization_leaveout_estimates_original_groups.csv')
        reference=reference.loc[reference.n_excluded.eq(9)&reference.weight.eq('equal county')]
        assert len(reference)==2 and reference.excluded_ccg.nunique()==1
        removed=list(map(int,reference.excluded_ccg.iloc[0].split(';')))
    if after_four:
        reference = pd.read_csv(OUTDIR/'h2_winsorization_leaveout_estimates.csv')
        reference = reference.loc[reference.excluded.eq('All four')]
    t28 = module('t28_cellaudit', ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100000)
    names = pd.read_csv(dc.CCG_CROSSWALK)
    names['ccg'] = pd.to_numeric(names.ccg_id, errors='coerce')
    names = names.dropna(subset=['ccg']).drop_duplicates('ccg').set_index('ccg')
    a['county'] = a.ccg.map(names.ccg_name)
    a['state_name'] = (a.ccg//1000).map(STATE_NAMES)
    a['county_name_available'] = a.county.notna()
    if a.state_name.isna().any():
        raise SystemExit('FAIL: missing state label')
    a['county'] = a.county.fillna(a.ccg.map(lambda c: f'County group {c:05d}'))
    print(f'County-name lookup unavailable for {a.loc[~a.county_name_available,"ccg"].nunique()} groups; coded labels retained.')
    # Saved unit-level county data supply the actual endpoints, not fitted values.
    if after_four or after_nine:
        endpoints = pd.read_parquet(CACHE/'winsorization_cell_attribution.parquet')
        endpoints = endpoints.loc[endpoints.weight.eq('equal county')]
        cols = ['employment_before','employment_after','population_before',
                'spending_before','spending_post1','spending_post2']
        a = a.merge(endpoints[['ccg','year']+cols],on=['ccg','year'],validate='one_to_one')
        a = a.loc[~a.ccg.isin(removed)].copy()
        if len(a) != 107030-35*len(removed):
            raise SystemExit('FAIL: reduced accounting sample mismatch')
    else:
        county = store.load_county_panel().set_index(['ccg','year'])
        spending = spending_panel()
        for col, source_col, offset in [('employment_before','E',-2),('employment_after','E',0),('population_before','P',-2)]:
            idx = pd.MultiIndex.from_arrays([a.ccg, a.year+offset])
            a[col] = county[source_col].reindex(idx).to_numpy()
        for label, offset in [('spending_before',-2),('spending_post1',-1),('spending_post2',0)]:
            vals = spending.stack()
            idx = pd.MultiIndex.from_arrays([a.ccg,a.year+offset])
            a[label] = vals.reindex(idx).to_numpy()
    np.testing.assert_allclose(a.x, (.5*(a.spending_post1+a.spending_post2)-a.spending_before)/a.population_before)
    np.testing.assert_allclose(a.y, 2*(a.employment_after-a.employment_before)/(a.employment_after+a.employment_before))
    fe = a.fe_division_year.to_numpy(np.int64)
    nofe = np.zeros(len(a),np.int64)
    raw = a[['x','y','z']].to_numpy(np.float64)
    cap = a[[f'{v}_percentile_1_99' for v in ['x','y','z']]].to_numpy(np.float64)
    delta = cap-raw
    summaries, cells, counties, tail_summaries = [], [], [], []
    for weight in (['equal county'] if after_nine else ['equal county','fixed 1975 population']):
        w = np.ones(len(a)) if weight=='equal county' else a.pop1975.to_numpy(np.float64)
        m = t28.w22.demean(np.column_stack([raw,cap]),fe,nofe,w)
        x,y,z,xc,yc,zc = m.T
        d0 = z@(w*x); dx = z@(w*xc); dfinal = zc@(w*xc)
        b0 = z@(w*y)/d0; bx = z@(w*y)/dx; bxy = z@(w*yc)/dx
        bfinal = zc@(w*yc)/dfinal
        ref = reference.loc[reference.weight.eq(weight)].set_index('variant')
        np.testing.assert_allclose(b0,ref.loc['untreated','second_stage'],rtol=1e-10)
        np.testing.assert_allclose(bfinal,ref.loc['joint_pooled_percentile_1_99','second_stage'],rtol=1e-10)
        contrib_x = -b0*w*z*delta[:,0]/dx
        contrib_y = w*z*delta[:,1]/dx
        contrib_z = w*delta[:,2]*(yc-bxy*xc)/dfinal
        np.testing.assert_allclose([contrib_x.sum(),contrib_y.sum(),contrib_z.sum()],
                                   [bx-b0,bxy-bx,bfinal-bxy],rtol=1e-9,atol=1e-15)
        f = a.copy()
        f['weight'] = weight
        for label,value in [('x',contrib_x),('y',contrib_y),('z',contrib_z)]:
            f[f'{label}_contribution_x1e5'] = value*1e5
            f[f'{label}_tail'] = np.where(delta[:,{'x':0,'y':1,'z':2}[label]]<0,'upper',
                                 np.where(delta[:,{'x':0,'y':1,'z':2}[label]]>0,'lower','not_capped'))
        f['total_contribution_x1e5'] = (contrib_x+contrib_y+contrib_z)*1e5
        f['raw_instrument_spending_covariance_share'] = w*z*x/d0
        f['lost_spending_covariance_share'] = -w*z*delta[:,0]/d0
        summaries.append({'weight':weight,'beta_raw_x1e5':b0*1e5,'beta_cap_x_x1e5':bx*1e5,
            'beta_cap_xy_x1e5':bxy*1e5,'beta_cap_xyz_x1e5':bfinal*1e5,
            'x_contribution_x1e5':contrib_x.sum()*1e5,'y_contribution_x1e5':contrib_y.sum()*1e5,
            'z_contribution_x1e5':contrib_z.sum()*1e5,
            'lost_spending_covariance_share':f.lost_spending_covariance_share.sum(),
            'raw_first_stage_crossproduct':d0,'cap_x_first_stage_crossproduct':dx,
            'final_first_stage_crossproduct':dfinal,'raw_reduced_form_crossproduct':z@(w*y),
            'cap_xy_reduced_form_crossproduct':z@(w*yc),'final_reduced_form_crossproduct':zc@(w*yc)})
        totals = ['x_contribution_x1e5','y_contribution_x1e5','z_contribution_x1e5',
                  'total_contribution_x1e5','raw_instrument_spending_covariance_share','lost_spending_covariance_share']
        c = f.groupby(['ccg','county','state_name'],sort=False)[totals].sum().reset_index()
        c['weight']=weight
        c['rank_absolute_total']=c.total_contribution_x1e5.abs().rank(method='first',ascending=False).astype(int)
        c['rank_lost_covariance']=c.lost_spending_covariance_share.abs().rank(method='first',ascending=False).astype(int)
        counties.append(c)
        f['rank_absolute_total']=f.total_contribution_x1e5.abs().rank(method='first',ascending=False).astype(int)
        f['rank_lost_covariance']=f.lost_spending_covariance_share.abs().rank(method='first',ascending=False).astype(int)
        cells.append(f)
        for v in ['x','y','z']:
            tail = f.groupby(f'{v}_tail',sort=False)[f'{v}_contribution_x1e5'].agg(['sum','size']).reset_index()
            tail['weight'],tail['variable']=weight,v
            tail = tail.rename(columns={f'{v}_tail':'tail','sum':'contribution_x1e5','size':'n'})
            tail_summaries.append(tail)
    full = pd.concat(cells,ignore_index=True)
    c = pd.concat(counties,ignore_index=True)
    full.to_parquet(CACHE/f'winsorization_cell_attribution{suffix}.parquet',index=False)
    pd.DataFrame(summaries).to_csv(OUTDIR/f'h2_winsorization_attribution_summary{suffix}.csv',index=False)
    pd.concat(tail_summaries).to_csv(OUTDIR/f'h2_winsorization_attribution_tails{suffix}.csv',index=False)
    c.sort_values(['weight','rank_absolute_total']).to_csv(OUTDIR/f'h2_winsorization_attribution_counties{suffix}.csv',index=False)
    top = full.loc[(full.rank_absolute_total<=100)|(full.rank_lost_covariance<=100)]
    top.sort_values(['weight','rank_absolute_total']).to_csv(OUTDIR/f'h2_winsorization_attribution_top_cells{suffix}.csv',index=False)
    if CORRECTED:
        parts=[]
        for weight,g in full.groupby('weight',sort=False):
            for v in ['x','y','z']:
                leaders=g.assign(component=v,component_contribution_x1e5=g[f'{v}_contribution_x1e5'],
                    absolute_component=g[f'{v}_contribution_x1e5'].abs()).nlargest(10,'absolute_component')
                parts.append(leaders)
        pd.concat(parts,ignore_index=True).to_csv(OUTDIR/f'component_leading_cells{suffix}.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(c.loc[c.rank_absolute_total<=12].sort_values(['weight','rank_absolute_total']).to_string(index=False))
    print(full.loc[full.rank_absolute_total<=8,['weight','county','state_name','year','x','y','z',
        'x_contribution_x1e5','y_contribution_x1e5','z_contribution_x1e5','total_contribution_x1e5']]
        .sort_values(['weight','total_contribution_x1e5'],ascending=[True,False]).to_string(index=False))
    print('PASS: exact cell sums match the saved raw-to-pooled-winsorized coefficient change.')


def source_audit(after_four=False, case_override=None, suffix_override=None):
    """Read contracts for leading covariance or absolute gap contributors."""
    suffix = '_after_four' if after_four else ''
    rankings = pd.read_csv(OUTDIR/f'h2_winsorization_attribution_counties{suffix}.csv')
    selected = rankings.loc[rankings.rank_absolute_total<=3] if after_four else rankings.loc[rankings.rank_lost_covariance<=5]
    leaders = sorted(selected.ccg.unique())
    attribution = pd.read_parquet(CACHE/f'winsorization_cell_attribution{suffix}.parquet')
    cases = []
    for ccg in leaders:
        weights = selected.loc[selected.ccg.eq(ccg),'weight'] if after_four else ['equal county']
        for weight in weights:
            g = attribution.loc[attribution.ccg.eq(ccg)&attribution.weight.eq(weight)]
            score = g.total_contribution_x1e5.abs() if after_four else g.lost_spending_covariance_share.abs()
            g = g.assign(audit_score=score).nlargest(3,'audit_score')
            cases.extend(g[['ccg','county','state_name','year']].to_dict('records'))
    case_frame = pd.DataFrame(cases).drop_duplicates(['ccg','year'])
    if CORRECTED:
        leading=pd.read_csv(OUTDIR/'component_leading_cells.csv')
        case_frame=(leading.groupby(['weight','component'],sort=False).head(3)
                    [['ccg','county','state_name','year']].drop_duplicates(['ccg','year']))
        leaders=sorted(case_frame.ccg.unique())
    if case_override is not None:
        case_frame=case_override[['ccg','county','state_name','year']].drop_duplicates(['ccg','year'])
        leaders=sorted(case_frame.ccg.unique())
        suffix=suffix_override or '_leaders'
    cases = case_frame.to_dict('records')
    case_frame.to_csv(OUTDIR/f'h2_winsorization_source_audit_cases{suffix}.csv',index=False)
    targets = {(int(r['ccg']),int(r['year'])+j) for r in cases for j in (-2,-1,0)}
    fips_map,_,_=dc.load_ccg_map()
    mapping = pd.DataFrame({'fips':list(fips_map),'ccg':list(fips_map.values())})
    mapping = mapping.loc[mapping.ccg.isin(leaders)]
    dollars = spending_panel()
    summaries,category_rows,contract_rows,action_rows = [],[],[],[]
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        con.register('county_map',mapping)
        for year in sorted({y for _,y in targets}):
            units = sorted(c for c,y in targets if y==year)
            source = dc.OUT_DIR/f'fy{year}.parquet' if year<dc.SEAM_YEAR else dc.USASPENDING_OUT_DIR/f'fy{year}.parquet'
            schema = con.execute('SELECT * FROM read_parquet(?) LIMIT 0',[str(source)]).df().columns
            def field(name, alias):
                return f'CAST(r."{name}" AS VARCHAR) AS "{alias}"' if name in schema else f'CAST(NULL AS VARCHAR) AS "{alias}"'
            if year<dc.SEAM_YEAR:
                selections = ['m.ccg','r.dollars * CASE WHEN r.dollars_unit=\'thousands\' THEN 1000 ELSE 1 END AS dollars_nominal',
                    field('fsc','category'),field('fsc_description','category_description'),
                    field('contract_no','contract'),field('mod_no','modification'),
                    field('contractor_name','contractor'),field('city_name','city'),
                    field('county_name','reported_county'),field('county_fips5','assigned_fips'),
                    field('fips_source','geography_source'),field('action_date','action_date'),
                    field('completion_date','performance_end'),field('report_no','action_identifier'),
                    field('n_actions','n_actions'),field('source_file','source_file')]
                query = 'SELECT '+','.join(selections)+''' FROM read_parquet(?) r JOIN county_map m
                    ON lpad(CAST(r.county_fips5 AS VARCHAR),5,'0')=m.fips
                    WHERE r.geo_class='county' AND (r.fsc IS NULL OR substr(CAST(r.fsc AS VARCHAR),1,1)<>'Y')'''
            else:
                selections = ['r.ccg','r.federal_action_obligation AS dollars_nominal',
                    field('product_or_service_code','category'),field('transaction_description','category_description'),
                    field('award_id_piid','contract'),field('modification_number','modification'),
                    field('recipient_name','contractor'),field('primary_place_of_performance_city_name','city'),
                    field('primary_place_of_performance_county_name','reported_county'),field('county_fips','assigned_fips'),
                    field('county_fips_source','geography_source'),field('action_date','action_date'),
                    field('period_of_performance_current_end_date','performance_end'),
                    field('contract_transaction_unique_key','action_identifier'),field('number_of_actions','n_actions'),
                    field('source_member','source_file')]
                query = 'SELECT '+','.join(selections)+''' FROM read_parquet(?) r
                    WHERE r.geo_class='county' AND r.award_or_idv_flag='AWARD'
                    AND (r.product_or_service_code IS NULL OR substr(CAST(r.product_or_service_code AS VARCHAR),1,1)<>'Y')
                    AND r.ccg IN ('''+','.join(map(str,units))+')'
            f = con.execute(query,[str(source)]).df()
            f = f.loc[f.ccg.isin(units)].copy()
            f['year']=year
            f['source']='DCADS' if year<dc.SEAM_YEAR else 'USAspending'
            f['contractor']=f.contractor.fillna('Not available in parsed cache')
            for ccg in units:
                g = f.loc[f.ccg.eq(ccg)]
                pos = g.loc[g.dollars_nominal>0].copy()
                positive = pos.dollars_nominal.sum()
                real = positive*dc.deflate(year)
                expected = dollars.loc[ccg,year]
                np.testing.assert_allclose(real,expected,rtol=1e-10,atol=.01)
                cats = pos.groupby('category',dropna=False,sort=False).dollars_nominal.sum().sort_values(ascending=False)
                contracts = pos.groupby(['contract','contractor'],dropna=False,sort=False).dollars_nominal.sum().sort_values(ascending=False)
                label = case_frame.loc[case_frame.ccg.eq(ccg)].iloc[0]
                meta={'ccg':ccg,'county':label['county'],'state_name':label['state_name'],'year':year,
                      'source':'DCADS' if year<dc.SEAM_YEAR else 'USAspending'}
                summaries.append({**meta,'n_records':len(g),'n_positive_records':len(pos),
                    'transaction_key_check_applicable':year>=dc.SEAM_YEAR,
                    'missing_transaction_keys':int(g.action_identifier.isna().sum()) if year>=dc.SEAM_YEAR else np.nan,
                    'rows_with_repeated_transaction_key':int(g.loc[g.action_identifier.notna(),'action_identifier'].duplicated(keep=False).sum()) if year>=dc.SEAM_YEAR else np.nan,
                    'positive_nominal':positive,'net_nominal':g.dollars_nominal.sum(),
                    'positive_real':real,'cached_positive_real':expected,
                    'largest_positive_action_share':pos.dollars_nominal.max()/positive if positive>0 else np.nan,
                    'top_category':cats.index[0] if len(cats) else None,
                    'top_category_share':cats.iloc[0]/positive if positive>0 else np.nan,
                    'top_contract':contracts.index[0][0] if len(contracts) else None,
                    'top_contractor':contracts.index[0][1] if len(contracts) else None,
                    'top_contract_share':contracts.iloc[0]/positive if positive>0 else np.nan})
                for rank,(category,val) in enumerate(cats.head(5).items(),1):
                    category_rows.append({**meta,'rank':rank,'category':category,'positive_nominal':val,'share':val/positive})
                for rank,((contract,actor),val) in enumerate(contracts.head(5).items(),1):
                    contract_rows.append({**meta,'rank':rank,'contract':contract,'contractor':actor,'positive_nominal':val,'share':val/positive})
                top=pos.nlargest(5,'dollars_nominal').copy()
                top['county'],top['state_name']=label['county'],label['state_name']
                action_rows.append(top)
    if len(summaries) != len(targets):
        raise SystemExit('FAIL: source audit did not cover every target county-year')
    pd.DataFrame(summaries).to_csv(OUTDIR/f'h2_winsorization_source_audit_annual{suffix}.csv',index=False)
    pd.DataFrame(category_rows).to_csv(OUTDIR/f'h2_winsorization_source_audit_categories{suffix}.csv',index=False)
    pd.DataFrame(contract_rows).to_csv(OUTDIR/f'h2_winsorization_source_audit_contracts{suffix}.csv',index=False)
    pd.concat(action_rows,ignore_index=True).to_csv(OUTDIR/f'h2_winsorization_source_audit_top_actions{suffix}.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f'PASS: all {len(summaries)} selected source county-year totals match the analysis spending cache. No geography or contract edits.')


def leaveout_comparison(next_group=False,original_groups=False):
    """Diagnose county sensitivity without rebuilding shifters or cutoffs."""
    a = pd.read_parquet(CACHE/'analysis_pooled.parquet')
    variants = {'untreated': ['y','x','z'], 'joint_pooled_percentile_1_99':
                [f'{v}_percentile_1_99' for v in ('y','x','z')]}
    required = list(dict.fromkeys(sum(variants.values(), []) +
                    ['ccg','year','pop1975','fe_division_year','cluster']))
    if (len(a) != 107030 or a.ccg.nunique() != 3058
            or a.duplicated(['ccg','year']).any()
            or not np.isfinite(a[required].to_numpy()).all()
            or not a.groupby('ccg',sort=False).size().eq(35).all()
            or not a.groupby('ccg',sort=False).pop1975.nunique().eq(1).all()
            or not a.pop1975.gt(0).all()):
        raise SystemExit('FAIL: invalid balanced county cache')
    named = {'Austin, Texas': 48015, 'Sagadahoc, Maine': 23023,
             'Newport News city, Virginia': 51700, 'St. Louis city, Missouri': 29510}
    if CORRECTED:
        samples = {'Austin, Texas': [named['Austin, Texas']]}
    else:
        samples = {name: [ccg] for name,ccg in named.items()}
        samples['All four'] = list(named.values())
    if original_groups:
        if not CORRECTED or next_group:
            raise SystemExit('FAIL: original groups require corrected standalone mode')
        saved=pd.read_csv(OUTDIR.parent/'h2_winsorization_leaveout_estimates_next_group.csv')
        samples={}
        for label,n in [('All four',4),('Four plus next group',9)]:
            g=saved.loc[saved.excluded.eq(label)]
            assert len(g)==4 and g.excluded_ccg.nunique()==1 and g.n_excluded.eq(n).all()
            samples[label]=list(map(int,g.excluded_ccg.iloc[0].split(';')))
            assert len(set(samples[label]))==n
        assert set(samples['All four'])==set(named.values())
        assert set(samples['All four']).issubset(samples['Four plus next group'])
    if next_group:
        rankings = pd.read_csv(OUTDIR/'h2_winsorization_attribution_counties_after_four.csv')
        leaders = sorted(rankings.loc[rankings.rank_absolute_total<=3,'ccg'].unique())
        samples = {'Four plus next group': FOUR_COUNTIES+leaders}
    reference = pd.read_csv(REFERENCE)
    if CORRECTED:
        reference=reference.loc[reference.source_version.eq('corrected')]
    reference = reference.loc[reference.variant.isin(variants)].copy()
    if len(reference) != 4 or reference.duplicated(['weight','variant']).any():
        raise SystemExit('FAIL: missing saved full-sample comparison')
    fields = ['weight','variant','first_stage_F','second_stage','second_stage_se',
              'second_stage_t','n','n_counties']
    rows = reference[fields].assign(excluded='None',n_excluded=0,
        excluded_ccg='',estimate_source='saved',cutoffs_fixed=True,
        shifters_fixed=True).to_dict('records')
    if next_group:
        saved = pd.read_csv(OUTDIR/'h2_winsorization_leaveout_estimates.csv')
        rows.extend(saved.loc[saved.excluded.eq('All four')].to_dict('records'))
    t28 = module('t28_leaveout', ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12,fixef_maxiter=100000)
    for label, excluded in samples.items():
        p = a.loc[~a.ccg.isin(excluded)]
        if len(p) != len(a)-35*len(excluded):
            raise SystemExit('FAIL: county exclusion count mismatch')
        fe,cl = (p[v].to_numpy(np.int64) for v in ('fe_division_year','cluster'))
        nofe = np.zeros(len(p),np.int64)
        for weight in ('equal county','fixed 1975 population'):
            w = np.ones(len(p)) if weight=='equal county' else p.pop1975.to_numpy(np.float64)
            for variant, columns in variants.items():
                y,x,z = (p[c].to_numpy(np.float64) for c in columns)
                fit = t28.fit2sls(y,x,z,w,fe,nofe,cl,singleton=False,want_F=True)
                if fit['n_dropped'] or fit['n'] != len(p):
                    raise SystemExit('FAIL: estimation removed additional observations')
                rows.append(dict(excluded=label,excluded_ccg=';'.join(map(str,excluded)),
                    n_excluded=len(excluded),weight=weight,variant=variant,
                    first_stage_F=fit['F'],second_stage=fit['beta'],
                    second_stage_se=fit['se'],second_stage_t=fit['beta']/fit['se'],
                    n=fit['n'],n_counties=p.ccg.nunique(),estimate_source='estimated',
                    cutoffs_fixed=True,shifters_fixed=True))
    results = pd.DataFrame(rows)
    results['second_stage_x1e5'] = results.second_stage*1e5
    results['second_stage_se_x1e5'] = results.second_stage_se*1e5
    expected_rows = 12 if next_group or original_groups else (8 if CORRECTED else 24)
    if len(results) != expected_rows or not np.isfinite(results[['first_stage_F','second_stage_t','second_stage']]).all().all():
        raise SystemExit('FAIL: incomplete leaveout estimates')
    gaps = results.pivot(index=['excluded','n_excluded','weight'],columns='variant',
                         values='second_stage_x1e5').reset_index()
    gaps['gap_x1e5'] = gaps.joint_pooled_percentile_1_99-gaps.untreated
    initial = gaps.loc[gaps.excluded.eq('None')].set_index('weight').gap_x1e5
    gaps['gap_reduction_percent'] = 100*(1-gaps.gap_x1e5.abs()/gaps.weight.map(initial).abs())
    for frame,stem in [(results,'estimates'),(gaps,'gaps')]:
        suffix = '_original_groups' if original_groups else ('_next_group' if next_group else '')
        path = OUTDIR/f'h2_winsorization_leaveout_{stem}{suffix}.csv'
        frame.to_csv(path,index=False)
        check = pd.read_csv(path)
        if len(check) != len(frame):
            raise SystemExit(f'FAIL: output row count {path}')
    print(results[['excluded','weight','variant','n_counties','first_stage_F',
                   'second_stage_x1e5','second_stage_t']].to_string(index=False))
    print(gaps.to_string(index=False))
    print(f'PASS: {4*len(samples)} new estimates. Entire county series excluded; no additional drops. Cutoffs, national shifters, and retained instrument values fixed.')


def calhoun_illinois_leaveout():
    """Exclude Calhoun and St. Clair, Illinois, jointly from the nine-excluded EW sample."""
    # mpgr-run-class: estimate_only
    # mpgr-cache: report_correction_rebuild/analysis_pooled.parquet
    # mpgr-new-estimates: 2
    # mpgr-variation-scope: county|h2|EW|nine-plus-two-excluded|division-year|raw-pooled1-99
    if not CORRECTED:
        raise SystemExit('FAIL: requires --corrected')
    saved=pd.read_csv(OUTDIR/'h2_winsorization_leaveout_estimates_original_groups.csv')
    anchor=saved.loc[saved.n_excluded.eq(9)&saved.weight.eq('equal county')].copy()
    assert len(anchor)==2 and anchor.excluded_ccg.nunique()==1
    original=list(map(int,anchor.excluded_ccg.iloc[0].split(';')))
    added=[17013,17163]
    assert not set(original)&set(added)
    a=pd.read_parquet(CACHE/'analysis_pooled.parquet')
    assert len(a)==107030 and not a.duplicated(['ccg','year']).any()
    excluded=original+added
    p=a.loc[~a.ccg.isin(excluded)].copy()
    assert len(p)==107030-35*11 and p.ccg.nunique()==3047
    assert p.groupby('ccg').size().eq(35).all()
    t28=module('t28_calhoun_il_leaveout',ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER=pf.MapDemeaner(fixef_tol=1e-12,fixef_maxiter=100000)
    rows=anchor.assign(sample='Nine excluded',estimate_source='saved').to_dict('records')
    for variant,cols in [('untreated',['y','x','z']),('joint_pooled_percentile_1_99',
                         [f'{v}_percentile_1_99' for v in ('y','x','z')])]:
        y,x,z=p[cols].to_numpy(np.float64).T
        assert np.isfinite(np.column_stack([y,x,z])).all()
        fit=t28.fit2sls(y,x,z,np.ones(len(p)),p.fe_division_year.to_numpy(np.int64),
                       np.zeros(len(p),np.int64),p.cluster.to_numpy(np.int64),singleton=False,want_F=True)
        assert fit['n']==len(p) and fit['n_dropped']==0
        rows.append(dict(sample='Also exclude Calhoun and St. Clair, Illinois',weight='equal county',
            variant=variant,n=fit['n'],n_counties=p.ccg.nunique(),n_excluded=len(excluded),
            excluded_ccg=';'.join(map(str,excluded)),second_stage=fit['beta'],second_stage_se=fit['se'],
            second_stage_t=fit['beta']/fit['se'],first_stage_F=fit['F'],estimate_source='estimated'))
    result=pd.DataFrame(rows)
    result['second_stage_x1e5']=result.second_stage*1e5
    result['second_stage_se_x1e5']=result.second_stage_se*1e5
    result['cutoffs_fixed']=True;result['shifters_fixed']=True
    assert np.isfinite(result[['second_stage','second_stage_t','first_stage_F']]).all().all()
    path=OUTDIR/'calhoun_il_pair_leaveout_estimates.csv'
    result.to_csv(path,index=False)
    assert len(pd.read_csv(path))==4
    gap=result.pivot(index='sample',columns='variant',values='second_stage_x1e5')
    gap['gap']=gap.joint_pooled_percentile_1_99-gap.untreated
    gap['gap_closed_percent']=100*(1-gap.gap/gap.loc['Nine excluded','gap'])
    gap.to_csv(OUTDIR/'calhoun_il_pair_leaveout_gaps.csv')
    print(result[['sample','variant','n_counties','second_stage_x1e5','second_stage_t','first_stage_F']].to_string(index=False))
    print(gap.to_string())
    print('PASS: two new fits; 70 additional observations excluded; no other drops or data changes.')


def calhoun_illinois_report():
    """Document a Belleville/Belleview location discrepancy and its evidence limits."""
    if not CORRECTED:
        raise SystemExit('FAIL: Calhoun Illinois report requires --corrected')
    import hashlib
    import json
    d=pd.read_csv(OUTDIR/'calhoun_il_dcads_contract_records.csv',dtype=str)
    d.year=pd.to_numeric(d.year);d.dollars_nominal=pd.to_numeric(d.dollars_nominal)
    identity=(d.contractor.fillna('').str.upper().str.contains('BELLEVILLE.*SHOE',regex=True)|
        d.est_duns.eq('006277974')|d.hq_duns.eq('006277974')|d.cage.eq('90976'))
    focal=d.county_fips.eq('17013')&d.category.eq('8430')
    assert (identity&d.coded_city.fillna('').str.strip().eq('Belleview'))[focal].all()
    candidate=d.loc[focal].copy()
    candidate['proposed_county_fips']='17163'
    candidate['status']='candidate_relocation_not_applied'
    candidate.to_csv(OUTDIR/'calhoun_il_candidate_relocation_records.csv',index=False)
    annual=pd.read_csv(OUTDIR/'calhoun_il_footwear_annual.csv')
    a=pd.read_csv(OUTDIR/'calhoun_il_gap_timeline.csv')
    components=pd.read_csv(OUTDIR/'h2_winsorization_instrument_components_calhoun_il.csv',dtype={'cat':str})
    components=components.loc[components.cat.eq('8430')].sort_values('year')
    matches=pd.read_csv(OUTDIR/'calhoun_il_exact_financial_matches.csv',dtype=str)
    matches['year']=pd.to_numeric(matches.year);matches.dollars_nominal=pd.to_numeric(matches.dollars_nominal)
    assert len(matches)==127
    pp=matches.primary_place_of_performance_city_name.fillna('').str.strip()
    assert pp.eq('').all()
    overlap=pd.read_csv(OUTDIR/'calhoun_il_usaspending_geography.csv',dtype={'county_fips':str})
    source_dir=ROOT/'data/source/primary_documents/raw'
    manifest=json.loads((source_dir/'belleville_shoe_costello_2004_10_04.fetch.json').read_text())
    assert manifest['final_status']=='official-artifact'
    pdf=source_dir/'belleville_shoe_costello_2004_10_04.pdf'
    assert hashlib.sha256(pdf.read_bytes()).hexdigest()==manifest['sha256']
    text=(PIPELINE.parents[3]/'batch_logs/macro/belleville-costello-extracted.txt').read_text()
    assert '# Pages: 2' in text and 'With two plants' in text and '1986' in text
    blocked=json.loads((source_dir/'belleville_shoe_osha_1996_establishment.fetch.json').read_text())
    evidence=dict(identifier=manifest['identifier'],acquisition_status=manifest['final_status'],
        evidence_status='official-clause-inspected',page=2,printed_page='E1784',sha256=manifest['sha256'],
        source_url=manifest['source_url'],scope='Company location, 1986 Belleville facility, and two plant locations in 2004; not individual award allocation',
        relocation_status='candidate_not_applied',n_candidate_records=len(candidate),
        matched_financial_records=len(matches),matched_records_with_performance_city=int(pp.ne('').sum()),
        osha_status=blocked['final_status'])
    (OUTDIR/'calhoun_il_evidence_status.json').write_text(json.dumps(evidence,indent=2)+'\n')
    def table(head,rows):
        return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
            ['| '+' | '.join(map(str,row))+' |' for row in rows])
    years=set(components.year)
    selected=a.loc[a.year.isin(years)].sort_values('year')
    timeline=table(['Shock end','Population at t-2','Spending shock per resident','Instrument','Employment MPGR','Gap contribution'],[
        [r.year,f'{r.population_before:,.0f}',f'{r.x:,.6f}',f'{r.z:,.6f}',f'{r.y:.6f}',
         f'{r.total_contribution_x1e5:.6f}'] for r in selected.itertuples()])
    comp=table(['Shock end','Exposure years','County share of national footwear spending','National footwear change, real dollars','Footwear instrument contribution','Total instrument'],[
        [r.year,f'{r.exposure_start}-{r.exposure_end}',f'{100*r.market_share:.6f}%',
         f'{r.national_shift:,.2f}',f'{r.z_contribution:,.6f}',f'{r.z_total:,.6f}'] for r in components.itertuples()])
    at=table(['Fiscal year','Records','Positive nominal footwear obligations','Net nominal footwear obligations'],[
        [r.year,r.rows,f'${r.positive_nominal:,.0f}',f'${r.net_nominal:,.0f}'] for r in annual.itertuples()])
    addresses=candidate.loc[candidate.year.ge(2004)].groupby(['year','coded_city','coded_county','recipient_city',
        'street','recipient_zip'],dropna=False).agg(rows=('year','size')).reset_index()
    addr=table(['Year','Coded city','Coded county','Recipient city','Recipient street','Recipient ZIP','Records'],[
        list(r) for r in addresses.itertuples(index=False,name=None)])
    mt=matches.groupby('year').agg(rows=('year','size'),nominal=('dollars_nominal','sum')).reset_index()
    matchtable=table(['Year','Unique same-contract, same-amount matches','Signed nominal dollars','Performance location present'],[
        [r.year,r.rows,f'${r.nominal:,.0f}','No'] for r in mt.itertuples()])
    origin=pd.read_csv(OUTDIR/'calhoun_il_footwear_national_origins.csv')
    ot=table(['Shock end','Origin county','State','Contribution to national footwear change, real dollars'],[
        [r.year,r.county,r.state_name,f'{r.shift:,.2f}']
        for r in origin.loc[origin.rank_absolute_shift.le(3)].itertuples()])
    body=f'''# Calhoun County, Illinois: Belleville Shoe location audit

## Finding and decision

The evidence points to a Belleville/Belleview geographic coding error, not a reason
to discard Calhoun County. Footwear awards to Belleville Shoe are coded to
Belleview, Calhoun County, while the recipient address says Belleville. The
contemporaneous Congressional Record places the company's Illinois factory in
Belleville and identifies its other factory in DeWitt, Arkansas (section 4).

The affected source extract contains {len(candidate):,} records over {annual.year.nunique()}
fiscal years, {annual.year.min()} through {annual.year.max()}, with
${annual.positive_nominal.sum():,.0f} in positive nominal obligations (Table 3).
This is a candidate relocation to St. Clair County, not an applied correction.
Evidence of the company's location does not establish the physical production
location of each individual award, especially where subcontracting or another
plant is possible. No source row, instrument, outcome, or regression is changed.

## Table 1. Why this county is selected

Selection is based on the saved equal-weight second-stage gap accounting after
the same nine county exclusions, not on first-stage F or variance. The county's
total contribution is {a.total_contribution_x1e5.sum():.6f}, of which these three
county-years contribute {selected.total_contribution_x1e5.sum():.6f}
({100*selected.total_contribution_x1e5.sum()/a.total_contribution_x1e5.sum():.3f}% of the net county contribution).
These are sequential capping contributions, not predicted effects of deleting
or relocating records. The complete county timeline is saved in the source CSV.

{timeline}

Employment MPGR is a midpoint growth rate, not an ordinary percentage change.

## Table 2. The instrument comes from men's footwear, category 8430

Historical exposure uses the five years t-11 through t-7. The national change is
the average of t-1 and t minus t-2. Dollar values are real except where marked
nominal. Minor categories can offset footwear, so its instrument contribution
can slightly exceed the total instrument.

{comp}

The source contracts generating the historical footwear exposure are identified
as Belleville Shoe through contractor names, establishment/headquarters DUNS
006277974, or CAGE 90976. The full record extract retains the historical identifier
strings, including nonstandard DUNS strings; none is silently repaired.

## Table 3. Footwear awards coded to Calhoun

{at}

The local spending shock is affected by these records. The same records enter
historical category exposure and therefore affect the instrument as well.
A geography-only relocation would preserve national category spending, but would
move local spending and exposure between counties. A valid sensitivity comparison
must change both, not just the observed spending variable.

## 4. Location evidence

### Table 4a. Disagreement inside the DCADS extract

{addr}

The native city codes distinguish Belleview (04832) from Belleville (04845);
the extract also has Belleville township code 04858. Correctly coded Belleville
records elsewhere in the company extract carry St. Clair County (17163).
The proposed candidate rule is narrow: current county 17013, coded city Belleview,
category 8430, and verified Belleville Shoe identity. It does not move other
Calhoun contracts or all contracts of the company nationwide.

### Contemporaneous official plant history

Jerry Costello's October 4, 2004 Congressional Record statement identifies the
company in Belleville, describes its 1986 factory opening in Belle Valley
Industrial Park, and identifies two plants in 2004: Belleville, Illinois, and
DeWitt, Arkansas. This supports the city-name confusion interpretation, but does
not allocate every listed award across production sites.

Source: [Congressional Record, printed page E1784, PDF page 2]({manifest['source_url']}).
The downloaded official PDF has two pages; SHA-256: `{manifest['sha256']}`.
Acquisition status: `official-artifact`. Evidence status after reading page 2:
`official-clause-inspected`. The 1996 OSHA establishment page is
`{blocked['final_status']}` after its registered ladder. Its search snippet is
not treated as inspected evidence.

### Table 4b. Same-year financial matches do not verify performance location

{matchtable}

There are {len(matches)} unique matches on fiscal year, normalized contract/parent
identifier and exact nonzero signed amount. They total ${matches.dollars_nominal.sum():,.0f}
in signed nominal obligations. Every one lacks a performance city in the available
USAspending record. Thus these matches corroborate contract and financial identity,
not the proposed destination county. Repeated financial keys are excluded from
this matching exercise, not from the data or estimates; the complete extracts
remain saved. Unique financial keys alone do not guarantee the same legal action.

Some other records under these contract families have performance locations.
They include Belleville in St. Clair County in 2006 and 2007, but also a 2005
SP010005C0370 record in Peoria and GSA orders at military locations. Those
observations prevent treating the recipient address as a universal performance
address. The complete geography table is saved, including missing locations.

## Table 5. Where the national footwear changes are recorded

{ot}

These are county origins in the existing coded data, not independently verified
production locations. Their annual sums reproduce each selected national shifter.

## Next empirical check, not yet run

Test the explicitly labeled candidate relocation of the Belleview-coded Belleville
Shoe records to St. Clair. Reconstruct both local spending and historical exposure,
retain the same county sample, and compare the raw and winsorized second-stage
coefficients and t-statistics. Record first-stage F as well. The candidate
interpretation must not be promoted to a verified line-by-line performance
allocation merely because it improves the regression. No effect on the
second-stage coefficient gap is established by this audit.

## Reproduction and saved evidence

Producer 102 modes: `--corrected --calhoun-illinois`,
`--corrected --calhoun-illinois-contracts`, and
`--corrected --calhoun-illinois-report`. The first mode reconciles seven selected
county-year spending totals to the source cache and three category sums to saved
instruments. The second scans DCADS fiscal years 1966-2006 for the county and
company, and USAspending 2001-2007 for the relevant award families. No employment
series is changed or independently revalidated here.

Evidence files are in `Results/instrument_normalization/corrected_gap_audit/`:
`calhoun_il_gap_timeline.csv`, `calhoun_il_dcads_contract_records.csv`,
`calhoun_il_footwear_annual.csv`, `calhoun_il_usaspending_contract_records.csv`,
`calhoun_il_usaspending_geography.csv`, `calhoun_il_exact_financial_matches.csv`,
`calhoun_il_footwear_national_origins.csv`, `calhoun_il_candidate_relocation_records.csv`,
and `calhoun_il_evidence_status.json`. Candidate records are audit output only;
they are not connected to production geography overrides.
'''
    assert '\u2014' not in body
    report=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/CALHOUN_ILLINOIS_SHOE_AUDIT.md'
    report.write_text(body);assert report.read_text()==body
    print('PASS: saved audit, candidate records, and evidence statuses. No correction or regression run. Report:',report)


def calhoun_illinois_contracts():
    """Audit Belleville Shoe identity, coded geography and same-year award overlap."""
    if not CORRECTED:
        raise SystemExit('FAIL: Calhoun Illinois contracts require --corrected')
    frames=[]
    fields={'contract_no':'contract','mod_no':'modification','contractor_name':'contractor',
        'est_duns':'est_duns','hq_duns':'hq_duns','cage':'cage','contractor_street':'street',
        'contractor_city':'recipient_city','contractor_state':'recipient_state','contractor_zip':'recipient_zip',
        'city_name':'coded_city','county_name':'coded_county','county_fips5':'county_fips',
        'location_code':'location_code','city_code':'city_code','county_code':'county_code',
        'fips_source':'geography_source','fsc':'category','action_date':'action_date','completion_date':'completion_date',
        'report_no':'report_no','source_file':'source_file'}
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        for year in range(1966,2007):
            path=dc.OUT_DIR/f'fy{year}.parquet'
            cols=con.execute('SELECT * FROM read_parquet(?) LIMIT 0',[str(path)]).df().columns
            selections=[f'CAST("{src}" AS VARCHAR) AS "{dst}"' if src in cols else
                f'CAST(NULL AS VARCHAR) AS "{dst}"' for src,dst in fields.items()]
            predicates=["county_fips5='17013'"]
            if 'contractor_name' in cols:
                predicates.append("upper(contractor_name) LIKE '%BELLEVILLE%SHOE%'")
            for name in ['est_duns','hq_duns']:
                if name in cols:
                    predicates.append(f"CAST({name} AS VARCHAR)='006277974'")
            query='SELECT '+','.join(selections)+", dollars*CASE WHEN dollars_unit='thousands' THEN 1000 ELSE 1 END AS dollars_nominal FROM read_parquet(?) WHERE "+' OR '.join(predicates)
            d=con.execute(query,[str(path)]).df();d['year']=year;d['source_path']=str(path);frames.append(d)
    d=pd.concat(frames,ignore_index=True)
    d['contract_key']=d.contract.fillna('').str.upper().str.replace(r'[^A-Z0-9]','',regex=True)
    d.to_csv(OUTDIR/'calhoun_il_dcads_contract_records.csv',index=False)
    shoe=d.loc[d.county_fips.eq('17013')&d.category.eq('8430')].copy()
    assert len(shoe)>0
    keys=sorted(set(shoe.contract_key)-{''})
    annual=shoe.assign(positive_nominal=shoe.dollars_nominal.clip(lower=0)).groupby('year').agg(
        rows=('year','size'),positive_nominal=('positive_nominal','sum'),net_nominal=('dollars_nominal','sum'),
        contracts=('contract_key','nunique')).reset_index()
    annual.to_csv(OUTDIR/'calhoun_il_footwear_annual.csv',index=False)
    uframes=[]
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        for year in range(2001,2008):
            path=dc.USASPENDING_OUT_DIR/f'fy{year}.parquet'
            u=con.execute('''SELECT * FROM read_parquet(?) WHERE award_or_idv_flag='AWARD' AND
                (regexp_replace(upper(award_id_piid),'[^A-Z0-9]','','g') IN (SELECT unnest(?)) OR
                 regexp_replace(upper(parent_award_id_piid),'[^A-Z0-9]','','g') IN (SELECT unnest(?)))''',
                [str(path),keys,keys]).df()
            u['year']=year;u['source_path']=str(path);uframes.append(u)
    u=pd.concat(uframes,ignore_index=True)
    assert not u.contract_transaction_unique_key.isna().any()
    assert not u.contract_transaction_unique_key.duplicated().any()
    u.to_csv(OUTDIR/'calhoun_il_usaspending_contract_records.csv',index=False)
    us=[]
    for field,role in [('award_id_piid','award'),('parent_award_id_piid','parent')]:
        v=u.copy();v['contract_key']=v[field].fillna('').str.upper().str.replace(r'[^A-Z0-9]','',regex=True)
        v=v.loc[v.contract_key.isin(keys)];v['matched_id_role']=role;us.append(v)
    v=pd.concat(us,ignore_index=True)
    v=v.drop_duplicates(['contract_transaction_unique_key','contract_key'])
    overlap=v.groupby(['year','contract_key','primary_place_of_performance_city_name',
        'primary_place_of_performance_county_name','county_fips'],dropna=False).agg(
        rows=('year','size'),net_nominal=('federal_action_obligation','sum')).reset_index()
    overlap.to_csv(OUTDIR/'calhoun_il_usaspending_geography.csv',index=False)
    # Unique financial-action matches avoid Cartesian joins on repeated order amounts.
    dc0=shoe.loc[shoe.year.ge(2001)].copy()
    dc0['amount_cents']=(dc0.dollars_nominal*100).round().astype('int64')
    v['amount_cents']=(v.federal_action_obligation*100).round().astype('int64')
    matchkeys=['year','contract_key','amount_cents']
    dd=dc0.loc[~dc0.duplicated(matchkeys,keep=False)&dc0.amount_cents.ne(0)]
    uu=v.loc[~v.duplicated(matchkeys,keep=False)&v.amount_cents.ne(0)]
    matches=dd.merge(uu,on=matchkeys,how='inner',validate='one_to_one',suffixes=('_dcads','_usa'))
    matches.to_csv(OUTDIR/'calhoun_il_exact_financial_matches.csv',index=False)
    if len(matches):
        np.testing.assert_allclose(matches.dollars_nominal,matches.federal_action_obligation,rtol=0,atol=.005)
    # National origins of the category shock in the selected three episodes.
    cases=pd.read_csv(OUTDIR/'h2_winsorization_source_audit_cases_calhoun_il.csv')
    origins=[]
    for t in cases.year:
        series=[]
        for offset in [-2,-1,0]:
            frame=pd.read_parquet(CACHE/'annual_category'/f'fy={t+offset}.parquet')
            series.append(frame.loc[frame.cat.eq('8430')].groupby('ccg').dollars.sum().rename(str(offset)))
        z=pd.concat(series,axis=1).fillna(0)
        z['shift']=.5*(z['-1']+z['0'])-z['-2'];z['year']=t
        z['rank_absolute_shift']=z['shift'].abs().rank(method='first',ascending=False).astype(int)
        origins.append(z.reset_index())
    origins=pd.concat(origins,ignore_index=True)
    names=pd.read_csv(dc.CCG_CROSSWALK)
    names['ccg']=pd.to_numeric(names.ccg_id,errors='coerce')
    labels=names.dropna(subset=['ccg']).drop_duplicates('ccg').set_index('ccg').ccg_name
    origins['county']=origins.ccg.map(labels).fillna('Unmapped county group')
    origins['state_name']=(origins.ccg//1000).map(STATE_NAMES)
    components=pd.read_csv(OUTDIR/'h2_winsorization_instrument_components_calhoun_il.csv',dtype={'cat':str})
    for t,g in origins.groupby('year'):
        expected=components.loc[components.year.eq(t)&components.cat.eq('8430'),'national_shift'].iloc[0]
        np.testing.assert_allclose(g['shift'].sum(),expected,rtol=1e-10,atol=.01)
    origins.to_csv(OUTDIR/'calhoun_il_footwear_national_origins.csv',index=False)
    print('DCADS Calhoun footwear annual:');print(annual.to_string(index=False))
    print('USAspending matched-contract geography:');print(overlap.to_string(index=False))
    print('Unique exact financial-action matches:',len(matches))
    if len(matches):
        print(matches[['year','contract_key','dollars_nominal','coded_city','coded_county',
            'primary_place_of_performance_city_name','primary_place_of_performance_county_name','county_fips_usa']].head(20).to_string(index=False))
    print('Leading national category origins:');print(origins.loc[origins.rank_absolute_shift.le(3)].to_string(index=False))
    print('PASS: all selected USAspending transaction keys unique; matched amounts and national-shift totals verified. No corrections applied.')


def calhoun_illinois_sources():
    """Trace the leading Calhoun, Illinois second-stage gap episodes to records."""
    if not CORRECTED:
        raise SystemExit('FAIL: Calhoun Illinois audit requires --corrected')
    a=pd.read_parquet(CACHE/'winsorization_cell_attribution_after_nine.parquet')
    a=a.loc[a.ccg.eq(17013)].sort_values('year')
    assert len(a)==35 and a.year.is_unique
    a.to_csv(OUTDIR/'calhoun_il_gap_timeline.csv',index=False)
    cases=a.nlargest(3,'total_contribution_x1e5')[['ccg','county','state_name','year']]
    source_audit(case_override=cases,suffix_override='_calhoun_il')
    mechanism_audit(suffix_override='_calhoun_il')
    print('Selected gap contribution:',a.loc[a.year.isin(cases.year),'total_contribution_x1e5'].sum(),
        'county total:',a.total_contribution_x1e5.sum())


def ew_first_stage_tails():
    """Attribute saved first-stage fit and clustered uncertainty to fixed tail groups."""
    if not CORRECTED:
        raise SystemExit('FAIL: first-stage tail accounting requires --corrected')
    a=pd.read_parquet(CACHE/'winsorization_cell_attribution_after_nine.parquet')
    assert len(a)==106715 and a.ccg.nunique()==3049 and not a.duplicated(['ccg','year']).any()
    assert a.groupby('ccg').size().eq(35).all()
    ref=pd.read_csv(OUTDIR/'h2_winsorization_leaveout_estimates_original_groups.csv')
    ref=ref.loc[ref.n_excluded.eq(9)&ref.weight.eq('equal county')].set_index('variant')
    assert len(ref)==2
    xtail=a.x.ne(a.x_percentile_1_99).to_numpy();ztail=a.z.ne(a.z_percentile_1_99).to_numpy()
    group=np.select([xtail&ztail,xtail&~ztail,~xtail&ztail],
        ['Both extreme','Spending only extreme','Instrument only extreme'],default='Neither extreme')
    def zone(v,c):
        return np.select([v<c,v>c],['Negative tail','Positive tail'],default='Middle')
    xzone=zone(a.x,a.x_percentile_1_99);zzone=zone(a.z,a.z_percentile_1_99)
    f=pd.factorize(a.fe_division_year)[0];cl,ids=pd.factorize(a.ccg)
    counts=np.bincount(f);n=len(a);G=len(ids);k=len(counts)+1
    assert a.groupby('ccg').cluster.nunique().eq(1).all() and a.cluster.nunique()==G
    ssc=G/(G-1)*(n-1)/(n-k)
    def demean(v):
        v=np.asarray(v,np.float64);return v-np.bincount(f,weights=v)[f]/counts[f]
    summaries=[];groups=[];counties=[];cells=[]
    for variant,label in [('untreated','Raw'),('joint_pooled_percentile_1_99','Joint winsorized')]:
        s='' if variant=='untreated' else '_percentile_1_99'
        x,z,y=map(demean,[a.x if not s else a.x_percentile_1_99,
            a.z if not s else a.z_percentile_1_99,a.y if not s else a.y_percentile_1_99])
        D=np.dot(z,x);ZZ=np.dot(z,z);XX=np.dot(x,x);N=np.dot(z,y)
        pi=D/ZZ;res=x-pi*z;score=z*res;q=np.bincount(cl,weights=score);Q=np.dot(q,q)
        se=np.sqrt(ssc*Q)/ZZ;F=pi*pi/(se*se);r2=D*D/(ZZ*XX)
        b=N/D;ivq=np.bincount(cl,weights=z*(y-b*x));ivse=np.sqrt(ssc*np.dot(ivq,ivq))/abs(D)
        rr=ref.loc[variant]
        np.testing.assert_allclose([F,b,ivse,b/ivse],
            [rr.first_stage_F,rr.second_stage,rr.second_stage_se,rr.second_stage_t],rtol=1e-10)
        c=pd.DataFrame(dict(ccg=ids,variant=label,cluster_score=q,variance_share=q*q/Q,
            normalized_score=q/D,leverage_share=np.bincount(cl,weights=z*z)/ZZ,
            D=np.bincount(cl,weights=z*x)))
        c=c.merge(a[['ccg','county','state_name']].drop_duplicates('ccg'),on='ccg',validate='one_to_one')
        c=c.sort_values('variance_share',ascending=False);c['variance_rank']=np.arange(1,G+1)
        counties.append(c)
        summaries.append(dict(variant=label,n=n,n_counties=G,slope=pi,slope_se=se,F=F,
            partial_R2=r2,D=D,ZZ=ZZ,XX=XX,SSE=np.dot(res,res),cluster_score_norm=np.sqrt(Q),ssc=ssc,
            top1_variance_percent=100*c.variance_share.iloc[0],top5_variance_percent=100*c.variance_share.head(5).sum(),
            second_stage_x1e5=b*1e5,second_stage_se_x1e5=ivse*1e5,second_stage_t=b/ivse))
        # The signed allocation sum_c q_c*q_cg preserves cross-group covariance.
        for grouping,labels in [('Tail overlap',group),('Signed tails',np.char.add(np.char.add(xzone,' / '),zzone))]:
            for value in np.unique(labels):
                keep=labels==value;qg=np.bincount(cl,weights=score*keep,minlength=G)
                d=(z[keep]*x[keep]).sum();zz=(z[keep]**2).sum()
                groups.append(dict(variant=label,grouping=grouping,group=value,n=int(keep.sum()),
                    observation_percent=100*keep.mean(),D_share_percent=100*d/D,
                    leverage_share_percent=100*zz/ZZ,SSE_share_percent=100*np.dot(res[keep],res[keep])/np.dot(res,res),
                    cluster_variance_share_percent=100*np.dot(q,qg)/Q,
                    group_moment_slope=d/zz if zz>0 else np.nan,
                    demeaned_sign_agreement_percent=100*np.mean(np.sign(x[keep])==np.sign(z[keep]))))
        cell=a[['ccg','year','county','state_name','x','z']].copy()
        cell['variant']=label;cell['group']=group;cell['x_zone']=xzone;cell['z_zone']=zzone
        cell['x_demeaned']=x;cell['z_demeaned']=z;cell['fitted_x']=pi*z
        cell['residual']=res;cell['score']=score;cell['cluster_variance_contribution']=q[cl]*score/Q
        cell['leverage_share']=z*z/ZZ;cells.append(cell)
    summaries=pd.DataFrame(summaries);groups=pd.DataFrame(groups);counties=pd.concat(counties,ignore_index=True)
    cells=pd.concat(cells,ignore_index=True)
    for _,g in groups.groupby(['variant','grouping']):
        np.testing.assert_allclose(g[['D_share_percent','leverage_share_percent','SSE_share_percent',
            'cluster_variance_share_percent']].sum(),100,rtol=1e-10)
        assert g.n.sum()==n
    for _,c in cells.groupby('variant'):
        np.testing.assert_allclose(c.cluster_variance_contribution.sum(),1,rtol=1e-10)
    summaries.to_csv(OUTDIR/'ew_first_stage_tail_summary.csv',index=False)
    groups.to_csv(OUTDIR/'ew_first_stage_tail_groups.csv',index=False)
    counties.to_csv(OUTDIR/'ew_first_stage_tail_counties.csv',index=False)
    cells.to_parquet(CACHE/'ew_first_stage_tail_cells.parquet',index=False)
    leaders=cells.assign(abs_variance_contribution=cells.cluster_variance_contribution.abs()).sort_values(
        'abs_variance_contribution',ascending=False).groupby('variant',sort=False).head(30)
    leaders.to_csv(OUTDIR/'ew_first_stage_tail_leading_cells.csv',index=False)
    print(summaries.to_string(index=False));print(groups.loc[groups.grouping.eq('Tail overlap')].to_string(index=False))
    print(counties.groupby('variant',sort=False).head(5).to_string(index=False))
    print('PASS: saved F, IV coefficient and t reproduced; group covariance allocations sum exactly.')
    ew_first_stage_tail_report()


def ew_first_stage_tail_report():
    """Write saved first-stage fit and uncertainty accounting."""
    if not CORRECTED:
        raise SystemExit('FAIL: first-stage tail report requires --corrected')
    s=pd.read_csv(OUTDIR/'ew_first_stage_tail_summary.csv')
    g=pd.read_csv(OUTDIR/'ew_first_stage_tail_groups.csv')
    c=pd.read_csv(OUTDIR/'ew_first_stage_tail_counties.csv')
    leading=pd.read_csv(OUTDIR/'ew_first_stage_tail_leading_cells.csv')
    r=s.set_index('variant');raw=r.loc['Raw'];cap=r.loc['Joint winsorized']
    calhoun=c.loc[c.ccg.eq(5013)].set_index('variant')
    both=g.loc[g.variant.eq('Raw')&g.grouping.eq('Tail overlap')&g.group.eq('Both extreme')].iloc[0]
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
            ['| '+' | '.join(map(str,row))+' |' for row in rows])
    st=table(['Specification','First-stage slope','Slope SE','F','Partial R2 (%)','IV coefficient','IV t'],[
        [v.variant,f'{v.slope:.6f}',f'{v.slope_se:.6f}',f'{v.F:.6f}',f'{100*v.partial_R2:.3f}',
         f'{v.second_stage_x1e5:.6f}',f'{v.second_stage_t:.6f}'] for v in s.itertuples()])
    def group_table(grouping):
        return table(['Specification','Group','Cells','Spending covariance (%)','Instrument leverage (%)',
            'Squared residuals (%)','Cluster variance (%)','Group moment slope'],[
            [v.variant,v.group,v.n,f'{v.D_share_percent:.3f}',f'{v.leverage_share_percent:.3f}',
             f'{v.SSE_share_percent:.3f}',f'{v.cluster_variance_share_percent:.3f}',f'{v.group_moment_slope:.6f}']
            for v in g.loc[g.grouping.eq(grouping)].itertuples()])
    ct=table(['Specification','County','State','Share of first-stage variance (%)','Instrument leverage (%)'],[
        [v.variant,v.county,v.state_name,f'{100*v.variance_share:.3f}',f'{100*v.leverage_share:.3f}']
        for v in c.groupby('variant',sort=False).head(5).itertuples()])
    lt=table(['County','State','Year','Tail group','Spending after FE','Fitted spending after FE',
        'Residual','Signed variance allocation (%)'],[
        [v.county,v.state_name,v.year,v.group,f'{v.x_demeaned:,.2f}',f'{v.fitted_x:,.2f}',
         f'{v.residual:,.2f}',f'{100*v.cluster_variance_contribution:.3f}']
        for v in leading.loc[leading.variant.eq('Raw')].head(6).itertuples()])
    body=f'''# EW first stage: tail overlap, fit and clustered uncertainty

## Main finding

The first-stage F rises from {raw.F:.6f} to {cap.F:.6f}, but partial R2 changes
from {100*raw.partial_R2:.3f}% to {100*cap.partial_R2:.3f}% and the slope from
{raw.slope:.6f} to {cap.slope:.6f} (Table 1). The F increase reflects a much
smaller slope standard error, {raw.slope_se:.6f} versus {cap.slope_se:.6f},
not an increase in within-sample correlation.

Calhoun County, Arkansas, accounts for
{100*calhoun.loc['Raw','variance_share']:.3f}% of raw first-stage slope variance
and {100*calhoun.loc['Joint winsorized','variance_share']:.3f}% in the winsorized
specification (county output underlying Table 4). These are shares of the squared
standard error, not shares of procurement or spending variance.

The {int(both.n):,} cells with both spending and the instrument in their tails
account for {both.D_share_percent:.3f}% of raw spending-instrument covariance
and {both.cluster_variance_share_percent:.3f}% of the signed clustered-variance
allocation (Table 2). Thus the uncertainty is not mainly from cases with huge
spending and an ordinary instrument. Both can be large while the observed
spending change differs substantially from the common first-stage prediction;
Table 5 gives the leading examples.

This explains the F improvement, not the entire raw-to-winsorized IV coefficient
gap. The second-stage standard error actually changes from
{raw.second_stage_se_x1e5:.6f} to {cap.second_stage_se_x1e5:.6f}, while its
coefficient changes from {raw.second_stage_x1e5:.6f} to {cap.second_stage_x1e5:.6f}
(Table 1 and its source CSV). Its higher t-statistic therefore must not be
described as a smaller second-stage standard error.

## Scope

The same nine-excluded sample contains 3,049 counties and 106,715 county-years,
1982 through 2016. Equal weights, division-year fixed effects and county-clustered
standard errors are used. The instrument is the full-national positive-spending
instrument, not the other-county variant. Raw and jointly pooled 1/99-winsorized
specifications retain their saved full-sample cutoffs. No new specification,
exclusion, source correction or baseline change is made.

## Table 1. Fit is distinct from the F-statistic

IV coefficients are multiplied by 100,000. First-stage slopes are not rescaled.

{st}

The F ratio is {cap.F/raw.F:.6f}. The spending-instrument cross-product D changes
by a factor of {cap.D/raw.D:.6f}, while the norm of county first-stage residual
scores changes by a factor of {cap.cluster_score_norm/raw.cluster_score_norm:.6f}.
These reproduce the F ratio exactly through F = D^2/(ssc*sum_c q_c^2).
The variance correction ssc is the same in both specifications.

## Table 2. Are spending and the instrument extreme together?

An observation is extreme if its raw variable lies outside that variable's saved
pooled 1st and 99th percentiles. The four groups are assigned once, before fixed
effects, and kept fixed across specifications. Group moment slopes are ratios
of within-group cross-products after full-sample demeaning, not subgroup fits.

{group_table('Tail overlap')}

## Table 3. Signed overlap

The group label is spending zone / instrument zone. Middle includes every value
between the saved pooled cutoffs; it does not mean zero. The signs in these labels
refer to raw tail membership, not residualized signs.

{group_table('Signed tails')}

## Table 4. Counties contributing most to first-stage uncertainty

{ct}

The top five counties account for {raw.top5_variance_percent:.3f}% of raw
first-stage variance and {cap.top5_variance_percent:.3f}% of winsorized
first-stage variance; the lists are ranked separately (Table 4).
This is diagnostic concentration, not a proposed sample exclusion rule.

## Table 5. Leading raw first-stage residual-score observations

Spending and its fitted values are in real dollars per resident after division-year
demeaning. The prediction is the whole-sample first-stage slope times demeaned z.
Rows are ranked by the absolute value of their signed contribution to clustered
variance, not by spending size alone. A poor linear prediction is not proof that
the underlying contract record is wrong.

{lt}

## Exact variance accounting

After division-year demeaning, pi=sum(z*x)/sum(z^2) and e=x-pi*z.
The county score is q_c=sum_(i in c) z_i*e_i. County-clustered slope variance
is ssc*sum_c(q_c^2)/(sum(z^2))^2. Instrument leverage is sum(z^2) by group
divided by its full-sample total. Partial R2 is squared within-sample correlation.

For a tail group g, q_cg sums cell scores in that group. Its variance allocation
is sum_c(q_c*q_cg)/sum_c(q_c^2), preserving all cross-group covariance. Thus group
variance shares are signed and may be negative or exceed 100%; squared group
scores alone would not add up. They do not predict the effect of deleting a group.
Squared-residual shares concern first-stage fit, not the clustered variance formula.

## Verification and reproduction

Producer 102: `--corrected --first-stage-tails`; report-only:
`--corrected --first-stage-tail-report`. Both saved first-stage F-statistics,
second-stage coefficients, standard errors and t-statistics are reproduced.
All group allocations sum to their whole-sample totals, with no observations lost.
The source is the certified `winsorization_cell_attribution_after_nine.parquet`.

Outputs under `Results/instrument_normalization/corrected_gap_audit/`:
`ew_first_stage_tail_summary.csv`, `ew_first_stage_tail_groups.csv`,
`ew_first_stage_tail_counties.csv`, `ew_first_stage_tail_leading_cells.csv`.
Full cell accounting: `data/temp/report_correction_rebuild/ew_first_stage_tail_cells.parquet`.
These calculations verify arithmetic, not the accuracy of underlying contracts.
'''
    assert '\u2014' not in body
    p=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/EW_FIRST_STAGE_TAIL_DIAGNOSTIC.md'
    p.write_text(body);assert p.read_text()==body;print('Report:',p)


def ew_other_county_fits():
    """Compare raw and capped IV using other-county national changes and fixed shares."""
    if not CORRECTED:
        raise SystemExit('FAIL: other-county fits require --corrected')
    import hashlib
    import json
    cert=json.loads((CACHE/'ew_own_shift_certification.json').read_text())
    source=CACHE/'winsorization_cell_attribution_after_nine.parquet'
    assert cert['status']=='PASS' and cert['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    a=pd.read_parquet(source)
    z=pd.read_parquet(CACHE/'ew_own_shift_components.parquet')
    a=a.merge(z.rename(columns={'z':'z_check'}),on=['ccg','year'],validate='one_to_one')
    assert len(a)==106715 and a.ccg.nunique()==3049 and a.groupby('ccg').size().eq(35).all()
    np.testing.assert_allclose(a.z,a.z_check,rtol=1e-12)
    np.testing.assert_allclose(a.z_own+a.z_elsewhere,a.z,rtol=1e-10,atol=1e-8)
    lo,hi=np.quantile(a.z_elsewhere,[.01,.99],method='linear')
    a['z_elsewhere_capped']=a.z_elsewhere.clip(lo,hi)
    ref=pd.read_csv(OUTDIR/'h2_winsorization_leaveout_estimates_original_groups.csv')
    ref=ref.loc[ref.n_excluded.eq(9)&ref.weight.eq('equal county')].copy()
    assert len(ref)==2
    removed=list(map(int,ref.excluded_ccg.iloc[0].split(';')))
    assert not a.ccg.isin(removed).any()
    ref['instrument_rule']='Full national';ref['estimate_source']='saved'
    rows=ref.to_dict('records')
    t28=module('t28_other_county',ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER=pf.MapDemeaner(fixef_tol=1e-12,fixef_maxiter=100000)
    for variant in ['untreated','joint_pooled_percentile_1_99']:
        suffix='' if variant=='untreated' else '_percentile_1_99'
        y,x=(a[v+suffix].to_numpy(np.float64) for v in ['y','x'])
        z=a['z_elsewhere' if variant=='untreated' else 'z_elsewhere_capped'].to_numpy(np.float64)
        fit=t28.fit2sls(y,x,z,np.ones(len(a)),a.fe_division_year.to_numpy(np.int64),
            np.zeros(len(a),np.int64),a.cluster.to_numpy(np.int64),singleton=False,want_F=True)
        assert fit['n']==len(a) and fit['n_dropped']==0
        rows.append(dict(instrument_rule='Other counties only',variant=variant,weight='equal county',
            first_stage_F=fit['F'],second_stage=fit['beta'],second_stage_se=fit['se'],
            second_stage_t=fit['beta']/fit['se'],second_stage_x1e5=fit['beta']*1e5,
            n=len(a),n_counties=3049,n_excluded=9,excluded_ccg=ref.excluded_ccg.iloc[0],estimate_source='estimated'))
    results=pd.DataFrame(rows)
    assert np.isfinite(results[['first_stage_F','second_stage','second_stage_se','second_stage_t']]).all().all()
    gaps=results.pivot(index='instrument_rule',columns='variant',values='second_stage_x1e5')
    gaps['gap']=gaps.joint_pooled_percentile_1_99-gaps.untreated
    gaps['gap_closed_percent']=100*(1-gaps.gap/gaps.loc['Full national','gap'])
    results.to_csv(OUTDIR/'ew_other_county_iv_comparison.csv',index=False)
    gaps.to_csv(OUTDIR/'ew_other_county_iv_gaps.csv')
    pd.DataFrame([dict(variable='z_elsewhere',lower=lo,upper=hi,n=len(a),scope='pooled retained nine-excluded sample')]).to_csv(
        OUTDIR/'ew_other_county_iv_cutoffs.csv',index=False)
    table=['| Instrument | Specification | Coefficient | SE | F | t |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in results.itertuples():
        table.append(f'| {r.instrument_rule} | {"Raw" if r.variant=="untreated" else "Joint winsorized"} | '
            f'{r.second_stage_x1e5:.6f} | {r.second_stage_se*1e5:.6f} | {r.first_stage_F:.6f} | {r.second_stage_t:.6f} |')
    body=f'''# EW h=2: instrument using other counties only

## Table 1. Raw and pooled winsorized comparison

The sample contains 3,049 counties and 106,715 observations, 1982 through 2016,
with the same nine counties excluded throughout. Equal weights, division-year
fixed effects and county-clustered standard errors apply. Coefficients and their
standard errors are multiplied by 100,000.

{chr(10).join(table)}

The full-national gap is {gaps.loc['Full national','gap']:.6f}. The other-county
gap is {gaps.loc['Other counties only','gap']:.6f}, a signed gap closure of
{gaps.loc['Other counties only','gap_closed_percent']:.6f}%.

## Construction and cutoff scope

The alternative instrument is z_elsewhere = z - z_own, using the certified
components in `EW_OWN_SHIFT_ACCOUNTING.md`. Only the focal county's spending
is removed from the national category changes. Exposure shares and their
national denominators are unchanged; excluded estimation counties still enter
national spending for other counties. County spending x and employment y are
unchanged in both raw specifications.

The winsorized specifications retain the saved x and y cutoffs from the full
3,058-county sample. The full-national reference also retains its saved z cutoffs.
For z_elsewhere, pooled unweighted 1st and 99th percentiles are computed on the
available 3,049-county component cache: {lo:.9f} and {hi:.9f}. Thus the percentile
reference population for the instrument differs by the nine excluded counties;
the full-sample alternative-instrument cutoffs are not constructed in this test.
All capping precedes fixed-effect absorption. No shares are renormalized and no
baseline output is changed.

## Verification and reproduction

Producer 102 with `--corrected --own-shift-fits` estimates exactly two models and
reuses both full-national estimates. The source hash, component identity,
observation count and balanced county series are checked. Neither regression
drops observations. The comparison does not establish exogeneity of spending
in other counties.

Outputs in `Results/instrument_normalization/corrected_gap_audit/`:
`ew_other_county_iv_comparison.csv`, `ew_other_county_iv_gaps.csv`,
and `ew_other_county_iv_cutoffs.csv`.
'''
    assert '\u2014' not in body
    report=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/EW_OTHER_COUNTY_IV_TEST.md'
    report.write_text(body);assert report.read_text()==body
    print(results[['instrument_rule','variant','second_stage_x1e5','first_stage_F','second_stage_t']].to_string(index=False))
    print(gaps.to_string());print('PASS: two fits, no observation drops. Report:',report)


def ew_own_shift_accounting():
    """Split national-shift moments and capping contributions by county of origin."""
    if not CORRECTED:
        raise SystemExit('FAIL: own-shift accounting requires --corrected')
    import hashlib
    import json
    source=CACHE/'winsorization_cell_attribution_after_nine.parquet'
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    a=pd.read_parquet(source)
    assert len(a)==106715 and a.ccg.nunique()==3049
    assert not a.duplicated(['ccg','year']).any()
    units=np.sort(a.ccg.unique());years=list(range(1982,2017))
    np.testing.assert_array_equal(a.ccg,np.repeat(units,len(years)))
    np.testing.assert_array_equal(a.year,np.tile(years,len(units)))
    pbase=a.pivot(index='ccg',columns='year',values='population_before').reindex(units)
    assert np.isfinite(pbase.to_numpy()).all() and pbase.gt(0).all().all()
    p100=module('p100_ownshift',PIPELINE/'100_h2_full_instrument_design_grid.py')
    p100.path_for=lambda cover,rule,floor,year:CACHE/'annual_category'/f'fy={year}.parquet'
    # National totals include all source counties, including excluded sample counties.
    cats,national,matrices=p100.load_sparse('exclude_construction','positive',0,4,units)
    own=np.zeros((len(units),len(years)));rebuilt=np.zeros_like(own)
    for j,t in enumerate(years):
        window=list(range(t-11,t-6))
        den=national[:,[yr-1966 for yr in window]].sum(axis=1)
        inverse=np.divide(1.,den,out=np.zeros_like(den),where=den>0)
        exposure=sum(matrices[yr] for yr in window)
        shares=exposure.multiply(inverse).tocsr()
        local_change=.5*(matrices[t]+matrices[t-1])-matrices[t-2]
        change=.5*(national[:,t-1966]+national[:,t-1-1966])-national[:,t-2-1966]
        pop=pbase[t].to_numpy()
        own[:,j]=np.asarray(shares.multiply(local_change).sum(axis=1)).ravel()/pop
        rebuilt[:,j]=np.asarray(shares@change).ravel()/pop
    np.testing.assert_allclose(rebuilt.ravel(),a.z,rtol=1e-10,atol=1e-8)
    a['z_own']=own.ravel();a['z_elsewhere']=a.z-a.z_own
    assert np.isfinite(a[['z_own','z_elsewhere']]).all().all()
    # Cross-check individual category components against the saved source audit.
    cases=pd.read_csv(OUTDIR/'leading_rhs_category_comparison.csv',dtype={'cat':str})
    checked=0;unitpos={c:i for i,c in enumerate(units)};catpos={c:i for i,c in enumerate(cats)}
    for r in cases.itertuples():
        if r.ccg not in unitpos:
            continue
        i,k=unitpos[r.ccg],catpos[r.cat]
        window=list(range(r.year-11,r.year-6))
        den=national[k,[yr-1966 for yr in window]].sum()
        share=sum(matrices[yr][i,k] for yr in window)/den if den>0 else 0.
        delta=.5*(matrices[r.year][i,k]+matrices[r.year-1][i,k])-matrices[r.year-2][i,k]
        np.testing.assert_allclose(share*delta/pbase.loc[r.ccg,r.year],r.own_county_z_component,
            rtol=1e-10,atol=1e-8)
        checked+=1
    assert checked>0
    f=pd.factorize(a.fe_division_year)[0];count=np.bincount(f)
    def demean(v):
        v=np.asarray(v,np.float64)
        return v-np.bincount(f,weights=v)[f]/count[f]
    y,x,z=map(demean,[a.y,a.x,a.z])
    yc,xc,zc=map(demean,[a.y_percentile_1_99,a.x_percentile_1_99,a.z_percentile_1_99])
    parts={'Own county':demean(a.z_own),'All other counties':demean(a.z_elsewhere)}
    np.testing.assert_allclose(parts['Own county']+parts['All other counties'],z,rtol=1e-10,atol=1e-8)
    D=np.dot(z,x);N=np.dot(z,y);Dx=np.dot(z,xc)
    beta=N/D;bx=N/Dx;bxy=np.dot(z,yc)/Dx;bf=np.dot(zc,yc)/np.dot(zc,xc)
    refs=pd.read_csv(OUTDIR/'ew_tail_moment_totals.csv').query("flow_rule == 'Positive'").set_index('variant')
    np.testing.assert_allclose([beta*1e5,bx*1e5,bf*1e5],
        refs.loc[['Raw','Cap spending only','Joint capped'],'coefficient_x1e5'],rtol=1e-10)
    groups=np.select([a.x.lt(a.x_percentile_1_99),a.x.gt(a.x_percentile_1_99)],
        ['Large negative shocks','Large positive shocks'],default='Middle')
    moments=[];gaprows=[];cells=[]
    for origin,zpart in parts.items():
        dr=zpart*x;nr=zpart*y;dx=zpart*xc
        gx=beta*(dr-dx)/Dx*1e5
        gy=zpart*(yc-y)/Dx*1e5
        c=a[['ccg','year','county','state_name']].copy()
        c['origin']=origin;c['group']=groups;c['D_raw']=dr;c['N_raw']=nr;c['D_cap_x']=dx
        c['x_gap_contribution']=gx;c['y_gap_contribution']=gy
        cells.append(c)
        for group in ['All observations','Large negative shocks','Middle','Large positive shocks']:
            keep=np.ones(len(a),bool) if group=='All observations' else groups==group
            moments.append(dict(origin=origin,group=group,n=int(keep.sum()),D_raw=dr[keep].sum(),
                N_raw=nr[keep].sum(),D_cap_x=dx[keep].sum(),
                raw_D_share_percent=100*dr[keep].sum()/D,raw_N_share_percent=100*nr[keep].sum()/N,
                lost_D_share_percent=100*(dr[keep]-dx[keep]).sum()/(D-Dx),
                x_gap_contribution=gx[keep].sum(),y_gap_contribution=gy[keep].sum()))
        gaprows.extend([dict(step='Cap spending',origin=origin,contribution_x1e5=gx.sum()),
            dict(step='Cap employment',origin=origin,contribution_x1e5=gy.sum())])
    gaprows.append(dict(step='Cap instrument',origin='Not allocated by origin',contribution_x1e5=(bf-bxy)*1e5))
    moments=pd.DataFrame(moments);gaps=pd.DataFrame(gaprows)
    np.testing.assert_allclose(gaps.contribution_x1e5.sum(),(bf-beta)*1e5,rtol=1e-10)
    allrows=moments.loc[moments.group.eq('All observations')]
    np.testing.assert_allclose(allrows[['D_raw','N_raw','D_cap_x']].sum(),[D,N,Dx],rtol=1e-10)
    for origin,g in moments.groupby('origin'):
        np.testing.assert_allclose(g.loc[g.group.ne('All observations'),['D_raw','N_raw','D_cap_x']].sum(),
            g.loc[g.group.eq('All observations'),['D_raw','N_raw','D_cap_x']].iloc[0],rtol=1e-10)
    a[['ccg','year','z','z_own','z_elsewhere']].to_parquet(CACHE/'ew_own_shift_components.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(CACHE/'ew_own_shift_moment_cells.parquet',index=False)
    moments.to_csv(OUTDIR/'ew_own_shift_moments.csv',index=False)
    gaps.to_csv(OUTDIR/'ew_own_shift_gap_contributions.csv',index=False)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    certificate=dict(status='PASS',source_sha256=source_hash,n=len(a),n_counties=len(units),
        full_instrument_reproduced=True,saved_category_checks=checked,gap_identity_passed=True,
        new_regressions=0,national_universe='all source counties')
    (CACHE/'ew_own_shift_certification.json').write_text(json.dumps(certificate,indent=2)+'\n')
    print(moments.to_string(index=False));print(gaps.to_string(index=False))
    print('PASS:',certificate)
    ew_own_shift_report()


def ew_own_shift_report():
    """Report saved own-origin accounting without reconstructing category data."""
    if not CORRECTED:
        raise SystemExit('FAIL: own-shift report requires --corrected')
    import json
    certificate=json.loads((CACHE/'ew_own_shift_certification.json').read_text())
    assert certificate['status']=='PASS'
    checked=certificate['saved_category_checks']
    moments=pd.read_csv(OUTDIR/'ew_own_shift_moments.csv')
    gaps=pd.read_csv(OUTDIR/'ew_own_shift_gap_contributions.csv')
    refs=pd.read_csv(OUTDIR/'ew_tail_moment_totals.csv').query("flow_rule == 'Positive'").set_index('variant')
    beta,bx,bf=refs.loc[['Raw','Cap spending only','Joint capped'],'coefficient_x1e5'].to_numpy()/1e5
    np.testing.assert_allclose(gaps.contribution_x1e5.sum(),(bf-beta)*1e5,rtol=1e-10)
    own=moments.loc[moments.origin.eq('Own county')&moments.group.eq('All observations')].iloc[0]
    def table(head,rows):
        return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
            ['| '+' | '.join(map(str,row))+' |' for row in rows])
    mt=table(['Origin','Observations','Raw spending covariance (%)','Raw employment covariance (%)',
        'Lost spending covariance (%)','Contribution from capping spending'],[
        [r.origin,r.group,f'{r.raw_D_share_percent:.3f}',f'{r.raw_N_share_percent:.3f}',
         f'{r.lost_D_share_percent:.3f}',f'{r.x_gap_contribution:.6f}'] for r in moments.itertuples()])
    gt=table(['Step','Origin','Contribution to coefficient gap'],[
        [r.step,r.origin,f'{r.contribution_x1e5:.6f}'] for r in gaps.itertuples()])
    body=f'''# EW winsorization gap: own-county versus other-county spending

## Main finding

The own-county component supplies {own.raw_D_share_percent:.3f}% of the raw
spending-instrument covariance but {own.raw_N_share_percent:.3f}% of the raw
employment-instrument covariance (Table 1). It supplies
{own.lost_D_share_percent:.3f}% of the spending covariance removed by capping x.
In the sequential coefficient accounting, its spending-step contribution is
{own.x_gap_contribution:.6f} out of the {(bx-beta)*1e5:.6f} increase from capping
spending alone (Table 2). Local spending embedded in the national shock therefore
accounts for a substantial part of that arithmetic.

This percentage is not the gap closure from removing the own-county component.
Such removal would change both the numerator and denominator of IV, and capping
the resulting instrument would be a separate nonlinear transformation. That
regression comparison is not run here.

## Definition and scope

This accounting uses the positive-spending, four-character category instrument,
equal weights, division-year fixed effects and the same nine-excluded sample:
3,049 counties and 106,715 observations from 1982 through 2016. No regression is
estimated and no baseline variable, cutoff or sample is changed.

Let G_ckt be positive real procurement in county c, category k and year t.
The exposure share s_ckt is the sum of G_cku over u=t-11,...,t-7 divided by
the corresponding national category sum. Population is measured at t-2.

    delta G_ckt = (G_ck,t-1 + G_ckt)/2 - G_ck,t-2
    z_own,ct = sum_k s_ckt * delta G_ckt / P_c,t-2
    z_elsewhere,ct = sum_k s_ckt * sum_(d != c) delta G_dkt / P_c,t-2
    z_ct = z_own,ct + z_elsewhere,ct

National spending includes all source counties, not just the estimation sample.
Exposure denominators remain national totals including the focal county.
This is an additive decomposition of the existing instrument, not a separately
renormalized leave-one-out instrument. Other-county spending need not be exogenous.

## Table 1. Covariance and spending-capping contributions

All variables and both instrument components are demeaned on the entire
nine-excluded sample using division-year fixed effects. Raw D=sum(z*x),
raw N=sum(z*y), and capped-spending D_x=sum(z*x_capped), using demeaned values.
Covariance shares below use common whole-sample totals, including in tail rows.
They are signed contributions and need not be probabilities. Tail membership
uses the fixed full-sample pooled spending cutoffs, not component-specific cutoffs.

{mt}

## Table 2. Exact accounting for the coefficient gap

The raw coefficient is {beta*1e5:.6f}, the spending-only capped coefficient is
{bx*1e5:.6f}, and the jointly capped coefficient is {bf*1e5:.6f}.
All coefficients and contributions are multiplied by 100,000.

{gt}

The entries sum to the full gap of {(bf-beta)*1e5:.6f}. For origin j, the
spending-step contribution is beta_raw * sum(z_j*(x-x_capped))/D_x.
The employment-step contribution is sum(z_j*(y_capped-y))/D_x.
This sequential accounting is order-dependent: spending, then employment,
then the instrument. Instrument capping is nonlinear in the sum of the two
components. Its contribution is therefore left unallocated rather than
assigned to an origin through an arbitrary rule.

These are contributions to existing estimates, not coefficients estimated
using each component separately. No component-specific F or t is calculated.
Inference for the existing raw and jointly capped regressions is in Table 1
of `EW_TAIL_MOMENTS_AND_NET_CHECK.md`.

## What this can establish

The tables locate the covariance and coefficient-gap contributions by spending
origin. They do not establish data error, shock exogeneity, or a true multiplier.
The Table 1 contrast supports investigating the own-county component rather than
treating national spending as unrelated to local procurement by construction.
It does not establish that the other-county component is a valid instrument.

## Verification and reproduction

Producer 102: `--corrected --own-shift-accounting`.
Report-only execution: `--corrected --own-shift-report`.
The category construction reproduces the saved instrument at every observation;
{checked} saved category-specific own-county components match the source audit.
The component sums reproduce raw and capped-spending covariances, and the
sequential contributions reproduce the entire coefficient gap. These checks use
the same contract data and are not independent validation of source accuracy.

Tables: `Results/instrument_normalization/corrected_gap_audit/ew_own_shift_moments.csv`
and `ew_own_shift_gap_contributions.csv` in the same folder.
Component and cell caches: `data/temp/report_correction_rebuild/ew_own_shift_components.parquet`
and `ew_own_shift_moment_cells.parquet`. Certification: `ew_own_shift_certification.json`.
'''
    assert '\u2014' not in body
    report=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/EW_OWN_SHIFT_ACCOUNTING.md'
    report.write_text(body);assert report.read_text()==body
    print('Report:',report)


def ew_tail_moments():
    """Partition EW first-stage and reduced-form moments on fixed spending tails."""
    if not CORRECTED:
        raise SystemExit('FAIL: tail moments requires --corrected')
    positive=pd.read_parquet(CACHE/'winsorization_cell_attribution_after_nine.parquet')
    assert len(positive)==106715 and positive.ccg.nunique()==3049
    netdir=dc.MACRO_VERSION_TEMP_DIR/'h2_net_flows_fixed_exposure_ew'
    import json
    assert json.loads((netdir/'certification.json').read_text())['status']=='PASS'
    net=pd.read_parquet(netdir/'analysis_pooled.parquet').set_index(['ccg','year'])
    net=net.loc[pd.MultiIndex.from_frame(positive[['ccg','year']])].reset_index()
    for col in ('ccg','year','y','fe_division_year','cluster'):
        np.testing.assert_array_equal(positive[col],net[col])
    groups=np.select([positive.x.lt(positive.x_percentile_1_99),positive.x.gt(positive.x_percentile_1_99)],
        ['Large negative shocks','Large positive shocks'],default='Middle')
    inference=pd.read_csv(OUTDIR.parent/'h2_net_fixed_exposure_ew_after_nine.csv')
    f=pd.factorize(positive.fe_division_year)[0];count=np.bincount(f)
    rows=[];totals=[];cellrows=[]
    variants={'Raw':['y','x','z'],
        'Cap spending only':['y','x_percentile_1_99','z'],
        'Joint capped':[v+'_percentile_1_99' for v in ('y','x','z')]}
    for flow,a in [('Positive',positive),('Net, fixed positive exposure',net)]:
        for variant,cols in variants.items():
            y,x,z=a[cols].to_numpy(np.float64).T
            yt=y-np.bincount(f,weights=y)[f]/count[f]
            xt=x-np.bincount(f,weights=x)[f]/count[f]
            zt=z-np.bincount(f,weights=z)[f]/count[f]
            D=zt*xt;N=zt*yt;den=D.sum();num=N.sum();beta=num/den
            if variant!='Cap spending only':
                key='untreated' if variant=='Raw' else 'joint_pooled_percentile_1_99'
                ref=inference.loc[inference.flow_rule.eq(flow)&inference.variant.eq(key)].iloc[0]
                np.testing.assert_allclose(beta,ref.second_stage,rtol=1e-10)
            totals.append(dict(flow_rule=flow,variant=variant,D=den,N=num,coefficient_x1e5=beta*1e5))
            cells=a[['ccg','year']].copy()
            cells['flow_rule'],cells['variant'],cells['group']=flow,variant,groups
            cells['D'],cells['N']=D,N
            cellrows.append(cells)
            for group in ['Large negative shocks','Middle','Large positive shocks']:
                keep=groups==group
                ds=D[keep].sum();ns=N[keep].sum()
                rows.append(dict(flow_rule=flow,variant=variant,group=group,n=int(keep.sum()),
                    observation_share_percent=100*keep.mean(),D=ds,N=ns,D_share_percent=100*ds/den,
                    N_share_percent=100*ns/num,moment_ratio_x1e5=1e5*ns/ds,
                    contribution_to_beta_x1e5=1e5*ns/den))
    moments=pd.DataFrame(rows);total=pd.DataFrame(totals)
    moments.to_csv(OUTDIR/'ew_tail_moments.csv',index=False)
    total.to_csv(OUTDIR/'ew_tail_moment_totals.csv',index=False)
    pd.concat(cellrows,ignore_index=True).to_parquet(CACHE/'ew_tail_moment_cells.parquet',index=False)
    for (flow,variant),g in moments.groupby(['flow_rule','variant']):
        t=total.loc[total.flow_rule.eq(flow)&total.variant.eq(variant)].iloc[0]
        np.testing.assert_allclose([g.D.sum(),g.N.sum(),g.contribution_to_beta_x1e5.sum()],
            [t.D,t.N,t.coefficient_x1e5],rtol=1e-10)
    # Spending timelines use the same cached annual dollars, without new raw-source reads.
    spending=spending_panel().stack()
    lower=positive.loc[groups=='Large negative shocks'].copy()
    lower['rank_spending_contribution']=lower.x_contribution_x1e5.rank(method='first',ascending=False).astype(int)
    trajectories=[]
    for offset in range(-4,3):
        index=pd.MultiIndex.from_arrays([lower.ccg,lower.year+offset])
        value=spending.reindex(index).to_numpy()
        assert np.isfinite(value).all() and lower.spending_before.gt(0).all()
        d=lower[['ccg','county','state_name','year','rank_spending_contribution','x_contribution_x1e5']].copy()
        d['offset']=offset;d['spending']=value
        d['spending_relative_to_base']=value/lower.spending_before.to_numpy()
        trajectories.append(d)
    trajectory=pd.concat(trajectories,ignore_index=True)
    trajectory.to_csv(OUTDIR/'ew_negative_tail_spending_paths.csv',index=False)
    paths=[]
    for label,d in [('All negative-tail cells',trajectory),('Top 20 spending-contribution cells',trajectory.loc[trajectory.rank_spending_contribution.le(20)])]:
        for offset,g in d.groupby('offset'):
            paths.append(dict(group=label,offset=offset,n=len(g),median=g.spending_relative_to_base.median(),
                p25=g.spending_relative_to_base.quantile(.25),p75=g.spending_relative_to_base.quantile(.75)))
    paths=pd.DataFrame(paths)
    paths.to_csv(OUTDIR/'ew_negative_tail_spending_path_summary.csv',index=False)
    def table(head,values):
        return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
            ['| '+' | '.join(map(str,row))+' |' for row in values])
    inf=table(['Flows','Specification','Coefficient','F','t'],[
        [r.flow_rule,'Raw' if r.variant=='untreated' else 'Joint capped',f'{r.second_stage_x1e5:.6f}',
         f'{r.first_stage_F:.6f}',f'{r.second_stage_t:.6f}'] for r in inference.itertuples()])
    def moment_table(flow):
        return table(['Specification','Spending group','Cells','Share of first-stage cross-product (%)',
            'Share of reduced-form cross-product (%)','Group moment ratio'],[
            [r.variant,r.group,r.n,f'{r.D_share_percent:.3f}',f'{r.N_share_percent:.3f}',f'{r.moment_ratio_x1e5:.6f}']
            for r in moments.loc[moments.flow_rule.eq(flow)].itertuples()])
    timelines=table(['Group','Year relative to shock end','Cells','Median spending / spending at t-2','25th percentile','75th percentile'],[
        [r.group,r.offset,r.n,f'{r.median:.4f}',f'{r.p25:.4f}',f'{r.p75:.4f}'] for r in paths.itertuples()])
    ref=moments.loc[moments.flow_rule.eq('Positive')&moments.variant.eq('Raw')]
    tails=ref.loc[ref.group.ne('Middle')]
    gaps=pd.read_csv(OUTDIR.parent/'h2_net_fixed_exposure_ew_after_nine_gap.csv').set_index('flow_rule')
    body=f'''# EW gap: fixed-sample net test and the role of large spending shocks

## Scope and result

Every calculation uses the same nine-excluded sample: 3,049 counties, 106,715
observations, 1982 through 2016. Equal weights, division-year fixed effects and
county clustering are unchanged. The reference remains the positive-spending design,
including meat. Net flows change county spending and national shifts, not the
predetermined positive-spending market shares. Full-sample pooled cutoffs are kept
within each flow design when the nine counties are omitted.

Netting does not close the gap on this matched sample either: the gap is
{gaps.loc['Positive','gap']:.6f} under positive flows and
{gaps.loc['Net, fixed positive exposure','gap']:.6f} under net flows. The signed
gap closure is {gaps.loc['Net, fixed positive exposure','gap_closed_percent']:.6f}%.

The {int(tails.n.sum()):,} spending-tail observations are
{tails.observation_share_percent.sum():.3f}% of the sample but contribute
{tails.D_share_percent.sum():.3f}% of the raw spending-instrument cross-product
and {tails.N_share_percent.sum():.3f}% of the raw employment-instrument cross-product
(Table 2). Their importance is not confined to the first stage: it would be wrong
to describe them as producing no employment response. These are descriptive IV
moments, not identified responses for a subgroup.

## Table 1. Net and positive results with identical nine exclusions

Coefficients and group moment ratios throughout are multiplied by 100,000.

{inf}

Source: `Results/instrument_normalization/h2_net_fixed_exposure_ew_after_nine.csv`.
Only the two net specifications are estimated here; the positive endpoints are saved.
No additional observations are dropped.

## Table 2. Where the positive-flow IV moments come from

{moment_table('Positive')}

## Table 3. The same groups under net flows

{moment_table('Net, fixed positive exposure')}

Groups are defined once from the positive baseline's raw spending and its pooled
cutoffs, then held fixed across every row, including net-flow calculations.
Negative tail means raw x is below its lower cutoff, not every negative x.
Positive tail means raw x is above its upper cutoff. Middle means neither.

For each full-sample specification, demean y, x and z by division-year. A group's
first-stage cross-product is D_g = sum within group of z_tilde*x_tilde; its
reduced-form cross-product is N_g = sum of z_tilde*y_tilde. The group moment ratio
is N_g/D_g. These ratios are NOT regressions estimated within each group and
have no subgroup F or t. Group assignment uses realized spending and is not an
exogenous sample split. The full coefficient is exactly sum_g N_g / sum_g D_g.

The rows show two distinct changes under capping: the contributions of the tails
to D change, and their own N_g/D_g changes. Thus the capped estimate is not simply
the raw estimate with a few counties ignored. Even uncapped observations can have
changed moments because division-year means change and their instruments or
outcomes may be capped. No group moment ratio should be called the true multiplier.

## Table 4. Spending paths around negative-tail shocks

Each cell's annual real spending is divided by its own spending at t-2, which is
strictly positive for every negative-tail cell. A value of 1 at t-2 is therefore
an identity. The table gives unweighted medians and quartiles of those ratios;
it is not an event-study regression or a causal estimate. Top 20 refers to the
largest positive spending-capping contributions among negative-tail cells in the
fixed-nine accounting, not a new county-exclusion list. All required annual values
are observed; none are dropped or imputed.

{timelines}

The shock definition selects a large decline from t-2 to the mean of t-1 and t.
Consequently low post/base ratios are partly selected by construction. The years
t-4 and t-3 provide descriptive context for whether the base year is unusually high;
they do not independently prove recording errors or a transitory causal shock.

## What remains unresolved

The accounting establishes that extreme spending changes carry most raw IV weight,
and that capping changes both that weight and the group moment ratios. It does not
establish whether the differences reflect economic heterogeneity, award/delivery
timing, mismeasured local exposure, or remaining source errors. Neither the net-flow
test nor the meat exclusion supports treating those specific accounting issues as
the explanation of the coefficient gap. No further source correction is warranted
solely by these moment tables.

## Verification and reproduction

Producer 107 with `--after-nine` estimates the two net fits from its certified cache.
Producer 102 with `--corrected --tail-moments` produces this accounting. All four
endpoint coefficient ratios reproduce saved inference; every group's D and N sums
to the corresponding total. Spending paths use the certified annual county cache.
No raw contract source is opened. Baseline data and presentation outputs are unchanged.

Outputs under `Results/instrument_normalization/corrected_gap_audit/`:
`ew_tail_moments.csv`, `ew_tail_moment_totals.csv`,
`ew_negative_tail_spending_paths.csv`, `ew_negative_tail_spending_path_summary.csv`.
The complete cell moments are saved as `ew_tail_moment_cells.parquet` in
`data/temp/report_correction_rebuild/`.
'''
    assert '\u2014' not in body
    path=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/EW_TAIL_MOMENTS_AND_NET_CHECK.md'
    path.write_text(body);assert path.read_text()==body
    print(moments.to_string(index=False))
    print(paths.to_string(index=False))
    print('PASS: fixed-sample inference and cell moment sums verified; wrote',path)


def calhoun_contract_audit():
    """Trace Calhoun's h=2 shock to signed Hydra contract actions and dates."""
    if not CORRECTED:
        raise SystemExit('FAIL: Calhoun audit requires --corrected')
    records=[]
    with duckdb.connect() as con:
        for year in range(2010,2017):
            source=dc.USASPENDING_OUT_DIR/f'fy{year}.parquet'
            d=con.execute('''SELECT source_fiscal_year,source_member,source_row_number,
                contract_transaction_unique_key,contract_award_unique_key,award_id_piid,
                parent_award_id_piid,modification_number,transaction_number,action_date,
                federal_action_obligation,product_or_service_code,transaction_description,
                period_of_performance_current_end_date,ccg,county_fips,geo_class,
                primary_place_of_performance_city_name,
                prime_award_transaction_place_of_performance_county_fips_code raw_county,
                primary_place_of_performance_county_name,
                primary_place_of_performance_state_code,primary_place_of_performance_zip_4,
                awarding_agency_code,awarding_sub_agency_name
                FROM read_parquet(?) WHERE award_or_idv_flag='AWARD'
                AND (ccg=5013 OR award_id_piid IN ('W31P4Q10C0190','W31P4Q14C0154'))''',[str(source)]).df()
            d['year'],d['input_file']=year,str(source)
            records.append(d)
    allrows=pd.concat(records,ignore_index=True)
    allrows.to_csv(OUTDIR/'calhoun_contract_all_locations.csv',index=False)
    local=allrows.loc[allrows.ccg.eq(5013)&allrows.geo_class.eq('county')&
        ~allrows.product_or_service_code.fillna('').str.startswith('Y')].copy()
    missing=int(local.contract_transaction_unique_key.isna().sum())
    repeated=int(local.contract_transaction_unique_key.duplicated(keep=False).sum())
    if missing or repeated:
        raise SystemExit(f'FAIL: Calhoun transaction keys: {missing} missing, {repeated} duplicated rows')
    local['positive']=local.federal_action_obligation.clip(lower=0)
    local['negative']=local.federal_action_obligation.clip(upper=0)
    annual=local.groupby('year').agg(source_rows=('ccg','size'),positive=('positive','sum'),
        negative=('negative','sum'),net=('federal_action_obligation','sum')).reset_index()
    for col in ('positive','negative','net'):
        annual[col+'_real']=annual[col]*annual.year.map(dc.deflate)
    saved=spending_panel().loc[5013]
    np.testing.assert_allclose(annual.positive_real,annual.year.map(saved),rtol=1e-10,atol=.01)
    annual.to_csv(OUTDIR/'calhoun_signed_spending_annual.csv',index=False)
    actions=local.loc[local.federal_action_obligation.abs().ge(1e6)].copy()
    actions.to_csv(OUTDIR/'calhoun_large_signed_actions.csv',index=False)
    hydra=allrows.loc[allrows.award_id_piid.isin(['W31P4Q10C0190','W31P4Q14C0154'])].copy()
    hydra['positive']=hydra.federal_action_obligation.clip(lower=0)
    hydra['negative']=hydra.federal_action_obligation.clip(upper=0)
    geo=hydra.groupby(['year','award_id_piid','ccg','primary_place_of_performance_city_name',
        'primary_place_of_performance_county_name'],dropna=False).agg(rows=('year','size'),
        positive=('positive','sum'),negative=('negative','sum'),net=('federal_action_obligation','sum')).reset_index()
    geo.to_csv(OUTDIR/'calhoun_hydra_geography.csv',index=False)
    # Exact opposite amounts are only candidate matches, never automatic corrections.
    match=hydra.copy();match['absolute_cents']=(match.federal_action_obligation.abs()*100).round().astype('int64')
    pos=match.loc[match.federal_action_obligation.ge(1e6)]
    neg=match.loc[match.federal_action_obligation.le(-1e6)]
    pairs=neg.merge(pos,on=['contract_award_unique_key','absolute_cents'],suffixes=('_negative','_positive'))
    pairs.to_csv(OUTDIR/'calhoun_exact_opposite_amount_candidates.csv',index=False)
    a=pd.read_parquet(CACHE/'winsorization_cell_attribution_after_nine.parquet')
    r=a.loc[a.ccg.eq(5013)&a.year.eq(2014)].iloc[0]
    yearly=annual.set_index('year')
    rows=[]
    for measure in ('positive','net'):
        before=yearly.loc[2012,measure+'_real']
        after=.5*(yearly.loc[2013,measure+'_real']+yearly.loc[2014,measure+'_real'])
        rows.append(dict(measure=measure,before_real=before,post_average_real=after,
            change_real=after-before,population_before=r.population_before,x=(after-before)/r.population_before))
    shock=pd.DataFrame(rows).set_index('measure')
    np.testing.assert_allclose(shock.loc['positive','x'],r.x,rtol=1e-10)
    shock.to_csv(OUTDIR/'calhoun_2014_signed_shock.csv')
    components=pd.read_csv(OUTDIR/'h2_winsorization_instrument_components_after_nine_rhs.csv',dtype={'cat':str})
    components=components.loc[components.ccg.eq(5013)&components.year.eq(2014)].copy()
    np.testing.assert_allclose(components.z_contribution.sum(),r.z,rtol=1e-10)
    components.sort_values('rank_absolute_z').head(10).to_csv(OUTDIR/'calhoun_2014_instrument_components.csv',index=False)
    reduction=100*(1-abs(shock.loc['net','x'])/abs(shock.loc['positive','x']))
    def table(headers,values):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
            ['| '+' | '.join(map(str,row))+' |' for row in values])
    totals=table(['Fiscal year','Positive obligations','Negative obligations','Net obligations'],[
        [q.year,f'${q.positive:,.2f}',f'${q.negative:,.2f}',f'${q.net:,.2f}'] for q in annual.itertuples()])
    large=actions.loc[actions.year.eq(2012)].assign(size=lambda d:d.federal_action_obligation.abs()).nlargest(8,'size')
    detail=table(['Contract','Modification / transaction','Action date','Signed obligations','Recorded completion','Source description'],[
        [q.award_id_piid,f'{q.modification_number} / {q.transaction_number}',q.action_date,
         f'${q.federal_action_obligation:,.2f}',q.period_of_performance_current_end_date,q.transaction_description]
        for q in large.itertuples()])
    comparison=table(['Spending convention','2012 real dollars','Mean 2013/2014 real dollars','Change per resident'],[
        [k,f'${q.before_real:,.2f}',f'${q.post_average_real:,.2f}',f'${q.x:,.6f}'] for k,q in shock.iterrows()])
    top=components.sort_values('rank_absolute_z').head(5)
    cats=table(['Category','Historical market share (%)','National shift, real dollars','Instrument contribution'],[
        [q.cat,f'{100*q.market_share:.6f}',f'${q.national_shift:,.2f}',f'{q.z_contribution:+.6f}'] for q in top.itertuples()])
    docs=dc.MACRO_RESULTS_ROOT.parent/'data/source/primary_documents/raw'
    import json
    acquisition=[]
    for stem in ['gd_hydra_2014_12_01','gd_hydra_2013_08_16','dod_contracts_2014_09_15_hydra']:
        m=json.loads((docs/(stem+'.fetch.json')).read_text())
        acquisition.append([stem,m['final_status'],m['verification_level']])
    acquisition_table=table(['Document','Acquisition status','Evidence status'],acquisition)
    pair_table=table(['Modification','Negative transaction','Positive transaction','Date','Negative amount','Positive amount'],[
        [q.modification_number_negative,q.transaction_number_negative,q.transaction_number_positive,
         q.action_date_negative,f'${q.federal_action_obligation_negative:,.2f}',f'${q.federal_action_obligation_positive:,.2f}']
        for q in pairs.itertuples()])
    assert len(pairs)==1
    swap=pairs.iloc[0]
    assert swap.modification_number_negative==swap.modification_number_positive=='P00062'
    assert swap.ccg_negative==swap.ccg_positive==5013
    assert swap.action_date_negative==swap.action_date_positive=='2012-03-15'
    assert swap.transaction_description_negative==swap.transaction_description_positive
    body=f'''# Calhoun County, Arkansas: the 2014 procurement shock

## Finding

The 2014 shock compares a large fiscal 2012 procurement year with the average of
fiscal 2013 and 2014 (Table 1). In the parsed records, the main program is Hydra-70
rockets. Fiscal 2012 combines large option exercises with large negative price and
scope adjustments (Table 2). The baseline retains positive actions and ignores
negative ones; it is not net obligations.

Using signed net obligations for this county's three endpoint years makes the
2014 spending decline {reduction:.6f}% smaller in absolute value (Table 3).
That is an accounting diagnostic, not an estimated change in the EW coefficient
gap. No net-obligation instrument or regression is constructed in this audit.
No source correction or baseline change is made.

## Table 1. Annual spending recorded in Calhoun

Nominal dollars. All positive and negative nonconstruction AWARD actions assigned
to the county are included. Positive totals, after deflation, reconcile with the
certified analysis cache in every listed year.

{totals}

## Table 2. Largest absolute fiscal 2012 actions

Descriptions and completion dates are reported fields, not independently verified
performance or cash-payment records.

{detail}

The negative modification P00086 describes lower prices/not-to-exceed amounts and
delivery extensions due to rocket-motor problems. P00087 describes a definitization
involving government-furnished motor grain. These descriptions are consistent with
substantive contract adjustments, not sufficient evidence of duplicate reports.
Positive obligations in 2012 include option exercises with recorded completion
dates in 2014 and 2015. Award-year obligations therefore cannot be read as the
amount of contemporaneous local production. Actual delivery and payment timing
have not been verified.

## Table 3. Exact 2014 shock arithmetic

Population denominator: {r.population_before:,.0f} residents in 2012. Spending is in
the project's fixed real-dollar units. The formula is
[0.5(spending_2013 + spending_2014) - spending_2012] / population_2012.

{comparison}

The baseline spending variable is capped from {r.x:,.6f} to
{r.x_percentile_1_99:,.6f} by pooled winsorization. Its instrument is capped from
{r.z:,.6f} to {r.z_percentile_1_99:,.6f}. In the fixed-nine EW sequential accounting,
this county-year contributes {r.x_contribution_x1e5:+.6f} through spending capping,
{r.z_contribution_x1e5:+.6f} through instrument capping, and
{r.total_contribution_x1e5:+.6f} in total, in coefficient units multiplied by 100,000.
These accounting contributions are not county-deletion effects.

## Table 4. Largest category components of the 2014 instrument

{cats}

Market shares use 2003 through 2007. National shifts use the mean of 2013/2014
minus 2012. All saved category components sum to the cached instrument.
The county net-obligation calculation in Table 3 does not change these positive-action
national shifts or historical exposures.

## Duplicates, geography and limits

There are {len(local)} nonconstruction county AWARD rows across 2010 through 2016,
with {missing} missing and {repeated} repeated transaction identifiers.
The same-award search finds {len(pairs)} pairs of positive and negative actions
of at least $1 million with exactly opposite amounts. Such a match is only a review
candidate, not a rule authorizing cancellation. Different modification/transaction
numbers are not duplicates merely because they belong to the same parent contract.

### Exact same-day offset in the recorded data

{pair_table}

Both P00062 transactions describe replacing Army requirements with Foreign Military
Sales requirements. Together they add zero net obligations, while the positive-only
measure counts ${swap.federal_action_obligation_positive:,.2f}. This is a concrete
example of the positive-only measure counting an offsetting contract adjustment.
It is not a duplicated transaction identifier or a demonstrated parsing error.
The much larger P00086 reduction has no exact opposite-amount match in this audit.

The audit also saves both Hydra contract numbers across all reported locations,
without conditioning on Calhoun, in `calhoun_hydra_geography.csv`. Reported county
and city fields do not independently establish the share of work actually performed
there. Multisite production and the absence of exact delivery/payment records remain
unresolved. No geographic reassignment is justified from this audit alone.

## External verification

{acquisition_table}

The finite fetch routes returned access errors. Search snippets are not used as
verified primary-document evidence. The contract descriptions and dollar amounts
above are supported by the local government-data records, not by independently
inspected contract documents. The manifests retain the attempted URLs and statuses
in `data/source/primary_documents/raw/`.

## Reproduction

Producer: `step_1/code/pipeline/102_h2_winsorization_cell_audit.py --corrected --calhoun-contract`.
The mode reads saved USAspending fiscal-year files, the corrected spending cache,
and saved instrument and EW accounting components. It runs no regressions.
Outputs under `Results/instrument_normalization/corrected_gap_audit/`:

- `calhoun_contract_all_locations.csv`: actions and their input file, ZIP member, and source row.
- `calhoun_signed_spending_annual.csv`, `calhoun_large_signed_actions.csv`.
- `calhoun_hydra_geography.csv`, `calhoun_exact_opposite_amount_candidates.csv`.
- `calhoun_2014_signed_shock.csv`, `calhoun_2014_instrument_components.csv`.
'''
    assert '\u2014' not in body
    path=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/CALHOUN_HYDRA_2014_AUDIT.md'
    path.write_text(body);assert path.read_text()==body
    print(annual.to_string(index=False))
    print(shock.to_string())
    print(f'Absolute shock reduction using net: {reduction:.6f}%; exact opposite candidates: {len(pairs)}')
    print(geo.to_string(index=False))
    print('PASS: Calhoun source totals, shock arithmetic and cached instrument reconciled. Wrote',path)


def after_nine_gap_concentration():
    """Measure signed concentration of the fixed-sample EW capping accounting."""
    if not CORRECTED:
        raise SystemExit('FAIL: after-nine concentration requires --corrected')
    a=pd.read_parquet(CACHE/'winsorization_cell_attribution_after_nine.parquet')
    assert len(a)==106715 and a.ccg.nunique()==3049 and a.weight.eq('equal county').all()
    assert a.groupby('ccg').size().eq(35).all()
    summary=pd.read_csv(OUTDIR/'h2_winsorization_attribution_summary_after_nine.csv').iloc[0]
    gap=summary.beta_cap_xyz_x1e5-summary.beta_raw_x1e5
    a['rhs_contribution_x1e5']=a.x_contribution_x1e5+a.z_contribution_x1e5
    components=['x','y','z','rhs','total']
    cols=[v+'_contribution_x1e5' for v in components]
    c=a.groupby(['ccg','county','state_name'],as_index=False)[cols].sum()
    np.testing.assert_allclose(a.total_contribution_x1e5.sum(),gap,rtol=1e-10,atol=1e-12)
    for v in ('x','y','z'):
        assert (a.loc[a[v].eq(a[v+'_percentile_1_99']),v+'_contribution_x1e5']==0).all()
    masses=[];concentrations=[];leaders=[]
    for level,d in [('county-year',a),('county',c)]:
        for v in components:
            col=v+'_contribution_x1e5'
            positive=d.loc[d[col]>0].sort_values([col,'ccg'],ascending=[False,True])
            negative=d.loc[d[col]<0].sort_values([col,'ccg'])
            pos=positive[col].sum();neg=negative[col].sum();net=d[col].sum()
            np.testing.assert_allclose(pos+neg,net,atol=1e-12)
            masses.append(dict(level=level,component=v,positive_mass=pos,negative_mass=neg,net=net,
                percent_of_net_gap=100*net/gap,n_positive=len(positive),n_negative=len(negative),
                n_for_half_positive=int(np.searchsorted(positive[col].cumsum().to_numpy(),.5*pos)+1) if pos else 0,
                n_for_80pct_positive=int(np.searchsorted(positive[col].cumsum().to_numpy(),.8*pos)+1) if pos else 0))
            for sign,f,total in [('positive',positive,pos),('negative',negative,neg)]:
                top=f.head(20).copy()
                top['level'],top['component'],top['sign']=level,v,sign
                top['signed_contribution']=top[col]
                top['rank']=np.arange(1,len(top)+1)
                leaders.append(top)
                for n in (1,5,10,20,50,100):
                    subtotal=f.head(n)[col].sum()
                    concentrations.append(dict(level=level,component=v,sign=sign,top_n=n,
                        n_available=len(f),contribution=subtotal,percent_of_same_sign_mass=100*subtotal/total if total else 0,
                        percent_of_net_gap=100*subtotal/gap))
    mass=pd.DataFrame(masses);con=pd.DataFrame(concentrations)
    leading=pd.concat(leaders,ignore_index=True)
    mass.to_csv(OUTDIR/'after_nine_gap_signed_mass.csv',index=False)
    con.to_csv(OUTDIR/'after_nine_gap_concentration.csv',index=False)
    leading.to_csv(OUTDIR/'after_nine_gap_leaders.csv',index=False)
    c.to_csv(OUTDIR/'after_nine_gap_counties.csv',index=False)
    # This grouping retains gross positive and negative cells within counties.
    gross=a[['ccg','county','state_name','x_contribution_x1e5','z_contribution_x1e5']].copy()
    for v in ('x','z'):
        gross[v+'_positive']=gross[v+'_contribution_x1e5'].clip(lower=0)
        gross[v+'_negative']=gross[v+'_contribution_x1e5'].clip(upper=0)
    gross.groupby(['ccg','county','state_name'],as_index=False).sum().to_csv(OUTDIR/'after_nine_gap_gross_within_county.csv',index=False)
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
            ['| '+' | '.join(map(str,row))+' |' for row in rows])
    labels={'x':'Observed spending','y':'Employment outcome','z':'Instrument','rhs':'Spending + instrument','total':'All three'}
    stages=table(['Step','Coefficient','Change'],[
        ['Raw',f'{summary.beta_raw_x1e5:.6f}',''],
        ['Cap spending',f'{summary.beta_cap_x_x1e5:.6f}',f'{summary.x_contribution_x1e5:+.6f}'],
        ['Then cap outcome',f'{summary.beta_cap_xy_x1e5:.6f}',f'{summary.y_contribution_x1e5:+.6f}'],
        ['Then cap instrument',f'{summary.beta_cap_xyz_x1e5:.6f}',f'{summary.z_contribution_x1e5:+.6f}']])
    mass_table=table(['Component','Positive cell contributions','Negative cell contributions','Net','Net / gap (%)'],[
        [labels[r.component],f'{r.positive_mass:.6f}',f'{r.negative_mass:.6f}',f'{r.net:.6f}',f'{r.percent_of_net_gap:.3f}']
        for r in mass.loc[mass.level.eq('county-year')].itertuples()])
    concentration_table=table(['Level','Component','Sign','Top N','Signed contribution','Share of same-sign contributions (%)'],[
        [r.level,labels[r.component],r.sign,r.top_n,f'{r.contribution:+.6f}',f'{r.percent_of_same_sign_mass:.3f}']
        for r in con.loc[con.component.isin(['x','z'])&con.top_n.isin([5,20,50])].itertuples()])
    def leader_table(v,level,sign,n=10):
        d=leading.loc[leading.component.eq(v)&leading.level.eq(level)&leading.sign.eq(sign)].head(n)
        head=['County','State']+(['Year'] if level=='county-year' else [])+['Spending contribution','Outcome contribution','Instrument contribution','Total']
        return table(head,[[r.county.title(),r.state_name]+([int(r.year)] if level=='county-year' else [])+
            [f'{getattr(r,k):+.6f}' for k in ['x_contribution_x1e5','y_contribution_x1e5','z_contribution_x1e5','total_contribution_x1e5']]
            for r in d.itertuples()])
    inference=pd.read_csv(OUTDIR/'h2_winsorization_leaveout_estimates_original_groups.csv')
    inference=inference.loc[inference.n_excluded.eq(9)&inference.weight.eq('equal county')]
    inf=table(['Specification','Coefficient','First-stage F','Second-stage t'],[
        ['Raw' if r.variant=='untreated' else 'Joint pooled 1-99',f'{r.second_stage_x1e5:.6f}',f'{r.first_stage_F:.6f}',f'{r.second_stage_t:.6f}']
        for r in inference.itertuples()])
    xm=mass.loc[mass.level.eq('county')&mass.component.eq('x')].iloc[0]
    body=f'''# Where the EW winsorization gap comes from after nine county removals

## Scope

The original nine exclusions are fixed. The accounting uses {a.ccg.nunique():,} county groups,
{len(a):,} observations, and all 35 years from 1982 to 2016 per retained county.
It uses baseline spending, including meat category 8905, not the category-exclusion test.
Equal weights, division-year fixed effects and full-sample pooled 1st/99th-percentile
cutoffs are held fixed. Fixed effects are absorbed on this nine-excluded sample.
No further county omissions, new inference estimates, or raw-source corrections are run.

## Table 1. Exact coefficient accounting

Coefficients and contributions are multiplied by 100,000 throughout.

{stages}

The gap is {gap:.6f}. In this sequential accounting, spending capping contributes
{summary.x_contribution_x1e5:.6f}, or {100*summary.x_contribution_x1e5/gap:.3f}% of the gap.
This allocation follows the explicit order spending, outcome, instrument. It is
order-dependent, not a causal decomposition or an order-invariant attribution.
Intermediate coefficients are moment ratios used for accounting; no F or t is
claimed for those intermediate steps. Endpoint inference is reused in Table 7.

For the spending-only step, the reduced-form numerator is fixed at
{summary.raw_reduced_form_crossproduct:,.6f}. The spending-instrument cross-product
falls from {summary.raw_first_stage_crossproduct:,.6f} to
{summary.cap_x_first_stage_crossproduct:,.6f}, a reduction of
{100*summary.lost_spending_covariance_share:.3f}%. That denominator change accounts
for the move from the raw coefficient to the spending-capped coefficient in Table 1.

## Table 2. Positive and negative contributions before cancellation

{mass_table}

Positive contributions widen the positive raw-to-capped coefficient gap; negative
ones offset it. Small net contributions can conceal large opposing cell movements.
Spending + instrument is a subtotal; All three includes the outcome as well.

## Table 3. Concentration by sign

{concentration_table}

County rows first sum signed contributions across years within each county.
County-year rows do not net different years against one another. Shares use the
corresponding positive or negative contribution mass, not the much smaller net gap.
For spending, {xm.n_for_half_positive} counties account for half of positive county
contributions, and {xm.n_for_80pct_positive} account for 80%.

## Table 4. Ten largest gap-widening spending counties

{leader_table('x','county','positive')}

## Table 5. Ten largest gap-widening spending observations

{leader_table('x','county-year','positive')}

## Table 6a. Ten largest gap-widening instrument counties

{leader_table('z','county','positive')}

## Table 6b. Ten largest gap-offsetting instrument counties

{leader_table('z','county','negative')}

## Table 7. Saved endpoint inference

{inf}

## Interpretation and verification

These rankings locate the arithmetic sources of the coefficient gap. They do not
predict the effect of deleting a county: deletion also changes fixed-effect means,
the reduced form and the denominators. Nor do they establish that a contract record
is erroneous. Employment enters the instrument-step residual, so this ranking is
not an outcome-blind exclusion rule.

For residualized raw instrument z0 and denominators Dx = sum(z0*x_cap),
Df = sum(z_cap*x_cap), the cell contributions are:

- Spending: -beta_raw*z0*(x_cap-x_raw)/Dx.
- Outcome: z0*(y_cap-y_raw)/Dx.
- Instrument: (z_cap-z_raw)*(y_cap_tilde-beta_cap_xy*x_cap_tilde)/Df.

The last expression uses the raw change in z and the residualized capped outcome
and spending. This assigns contributions to actually capped observations rather
than to uncapped observations whose fixed-effect means move.
Cell sums match each coefficient step; total cell and county sums match the saved
endpoint gap. An uncapped variable has exactly zero contribution at its step.
{a.loc[~a.county_name_available,'ccg'].nunique()} county groups lack names in the canonical
lookup and retain explicit county-group identifiers; no observations are discarded.

## Reproduction

Producer: `step_1/code/pipeline/102_h2_winsorization_cell_audit.py`, using
`--corrected --after-nine-accounting`, then `--corrected --after-nine-concentration`.
Inputs: the certified `data/temp/report_correction_rebuild/analysis_pooled.parquet`
and saved endpoint data and inference. No parent or raw contract files are opened.
Cell accounting is saved in `winsorization_cell_attribution_after_nine.parquet`
in that cache directory. Machine-readable tables under
`Results/instrument_normalization/corrected_gap_audit/` are:

- `h2_winsorization_attribution_summary_after_nine.csv` and `h2_winsorization_attribution_tails_after_nine.csv`.
- `after_nine_gap_signed_mass.csv`, `after_nine_gap_concentration.csv`.
- `after_nine_gap_leaders.csv`, `after_nine_gap_counties.csv`.
- `after_nine_gap_gross_within_county.csv`, retaining positive and negative spending/instrument cell contributions within counties.
'''
    assert '\u2014' not in body
    path=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/EW_AFTER_NINE_GAP_ACCOUNTING.md'
    path.write_text(body)
    assert path.read_text()==body
    print(mass.to_string(index=False))
    print(con.loc[con.component.eq('x')&con.sign.eq('positive')].to_string(index=False))
    print('PASS: signed concentration and report saved:',path)


def after_nine_rhs_screen():
    """Select counties using only the RHS covariance changed by spending capping."""
    if not CORRECTED:
        raise SystemExit('FAIL: after-nine screen requires --corrected')
    saved=pd.read_csv(OUTDIR/'h2_winsorization_leaveout_estimates_original_groups.csv')
    anchor=saved.loc[saved.n_excluded.eq(9)]
    assert len(anchor)==4 and anchor.excluded_ccg.nunique()==1
    removed=list(map(int,anchor.excluded_ccg.iloc[0].split(';')))
    cols=['ccg','year','x','z','x_percentile_1_99','pop1975','fe_division_year']
    a=pd.read_parquet(CACHE/'analysis_pooled.parquet',columns=cols)
    a=a.loc[~a.ccg.isin(removed)]
    assert len(a)==106715 and a.ccg.nunique()==3049
    assert a.groupby('ccg').size().eq(35).all()
    names=pd.read_csv(OUTDIR/'h2_winsorization_attribution_counties.csv')
    names=names[['ccg','county','state_name']].drop_duplicates('ccg')
    f=pd.factorize(a.fe_division_year)[0]
    rankings=[];cells=[]
    for weight in ('equal county','fixed 1975 population'):
        w=np.ones(len(a)) if weight=='equal county' else a.pop1975.to_numpy(np.float64)
        W=np.bincount(f,weights=w)
        z=a.z.to_numpy(np.float64);x=a.x.to_numpy(np.float64);xc=a.x_percentile_1_99.to_numpy(np.float64)
        zt=z-np.bincount(f,weights=w*z)[f]/W[f]
        D=(w*zt*x).sum();Dc=(w*zt*xc).sum()
        assert D>0
        g=a[['ccg','year']].copy()
        g['weight']=weight
        g['covariance_loss_share']=-w*zt*(xc-x)/D
        g['covariance_loss']=g.covariance_loss_share*D
        g['raw_covariance']=D;g['cap_x_covariance']=Dc
        np.testing.assert_allclose(g.covariance_loss.sum(),D-Dc,rtol=1e-10,atol=.01)
        c=g.groupby(['ccg','weight'],as_index=False)[['covariance_loss_share','covariance_loss']].sum()
        c=c.merge(names,on='ccg',validate='one_to_one')
        c['absolute_score']=c.covariance_loss_share.abs()
        c=c.sort_values(['absolute_score','ccg'],ascending=[False,True])
        c['rank']=np.arange(1,len(c)+1)
        rankings.append(c);cells.append(g)
    ranking=pd.concat(rankings,ignore_index=True)
    ranking.to_csv(OUTDIR/'after_nine_rhs_ranking.csv',index=False)
    selected=ranking.loc[ranking['rank'].le(3)].copy()
    selected.to_csv(OUTDIR/'after_nine_rhs_selected.csv',index=False)
    allcells=pd.concat(cells,ignore_index=True)
    chosen=allcells.merge(selected[['ccg','weight','county','state_name']],on=['ccg','weight'],validate='many_to_one')
    chosen['absolute_cell_score']=chosen.covariance_loss_share.abs()
    cases=chosen.sort_values('absolute_cell_score',ascending=False).groupby(['ccg','weight'],sort=False).head(2)
    cases.to_csv(OUTDIR/'after_nine_rhs_case_years.csv',index=False)
    print(selected[['weight','rank','county','state_name','covariance_loss_share']].to_string(index=False))
    print(f'PASS: {selected.ccg.nunique()} candidate counties selected without reading y or any fitted coefficient.')


def after_nine_rhs_fits():
    """Test a frozen RHS-selected list individually and in prespecified small unions."""
    if not CORRECTED:
        raise SystemExit('FAIL: after-nine fits require --corrected')
    selected=pd.read_csv(OUTDIR/'after_nine_rhs_selected.csv')
    saved=pd.read_csv(OUTDIR/'h2_winsorization_leaveout_estimates_original_groups.csv',keep_default_na=False)
    anchor=saved.loc[saved.n_excluded.eq(9)]
    original=list(map(int,anchor.excluded_ccg.iloc[0].split(';')))
    candidates=sorted(selected.ccg.unique())
    assert not set(original)&set(candidates)
    samples={f'Add {int(c)}':[int(c)] for c in candidates}
    samples['Add top 1 per weight']=sorted(selected.loc[selected['rank'].eq(1),'ccg'].unique())
    samples['Add top 3 per weight']=candidates
    unique_samples={}
    for label,ids in samples.items():
        key=tuple(ids)
        if key not in unique_samples:
            unique_samples[key]=label
    a=pd.read_parquet(CACHE/'analysis_pooled.parquet')
    variants={'untreated':['y','x','z'],'joint_pooled_percentile_1_99':[f'{v}_percentile_1_99' for v in ('y','x','z')]}
    t28=module('t28_after_nine_rhs',ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER=pf.MapDemeaner(fixef_tol=1e-12,fixef_maxiter=100000)
    path=OUTDIR/'after_nine_rhs_estimates.csv'
    rows=pd.read_csv(path,keep_default_na=False).to_dict('records') if path.exists() else []
    keys={(r['sample'],r['weight'],r['variant']) for r in rows}
    for r in anchor.to_dict('records'):
        key=('Original nine',r['weight'],r['variant'])
        if key not in keys:
            r.update(sample='Original nine',added_counties='',n_added=0,estimate_source='saved')
            rows.append(r);keys.add(key)
    new_fits=0
    for ids,label in unique_samples.items():
        excluded=original+list(ids)
        p=a.loc[~a.ccg.isin(excluded)]
        expected=107030-35*len(excluded)
        assert len(p)==expected and p.groupby('ccg').size().eq(35).all()
        for weight in ('equal county','fixed 1975 population'):
            w=np.ones(len(p)) if weight=='equal county' else p.pop1975.to_numpy(np.float64)
            for variant,cols in variants.items():
                key=(label,weight,variant)
                if key in keys:
                    prior=next(r for r in rows if (r['sample'],r['weight'],r['variant'])==key)
                    assert set(map(int,prior['excluded_ccg'].split(';')))==set(excluded)
                    assert prior['n']==expected
                    continue
                y,x,z=p[cols].to_numpy(np.float64).T
                fit=t28.fit2sls(y,x,z,w,p.fe_division_year.to_numpy(np.int64),np.zeros(len(p),np.int64),
                               p.cluster.to_numpy(np.int64),singleton=False,want_F=True)
                assert fit['n']==expected and fit['n_dropped']==0
                rows.append(dict(sample=label,weight=weight,variant=variant,added_counties=';'.join(map(str,ids)),
                    excluded_ccg=';'.join(map(str,excluded)),n_added=len(ids),n_excluded=len(excluded),n=fit['n'],
                    n_counties=p.ccg.nunique(),first_stage_F=fit['F'],second_stage=fit['beta'],second_stage_se=fit['se'],
                    second_stage_t=fit['beta']/fit['se'],second_stage_x1e5=fit['beta']*1e5,estimate_source='estimated'))
                keys.add(key);new_fits+=1
        pd.DataFrame(rows).to_csv(path,index=False)
    e=pd.DataFrame(rows)
    assert len(e)==4*(1+len(unique_samples)) and not e.duplicated(['sample','weight','variant']).any()
    assert np.isfinite(e[['first_stage_F','second_stage','second_stage_t']]).all().all()
    d=e.pivot(index=['sample','weight','n_added','n_excluded'],columns='variant',values='second_stage_x1e5').reset_index()
    d['gap']=d.joint_pooled_percentile_1_99-d.untreated
    base=d.loc[d['sample'].eq('Original nine')].set_index('weight')
    full=saved.loc[saved.n_excluded.eq(0)].pivot(index='weight',columns='variant',values='second_stage_x1e5')
    fullgap=full.joint_pooled_percentile_1_99-full.untreated
    d['gap_closed_relative_to_nine_percent']=100*(1-d.gap.abs()/d.weight.map(base.gap.abs()))
    d['gap_closed_relative_to_full_percent']=100*(1-d.gap.abs()/d.weight.map(fullgap.abs()))
    d['raw_change']=d.untreated-d.weight.map(base.untreated)
    d['winsorized_change']=d.joint_pooled_percentile_1_99-d.weight.map(base.joint_pooled_percentile_1_99)
    d['raw_up_closure']=d.raw_change
    d['winsorized_down_closure']=-d.winsorized_change
    np.testing.assert_allclose(d.raw_up_closure+d.winsorized_down_closure,d.weight.map(base.gap)-d.gap,atol=1e-12)
    d.to_csv(OUTDIR/'after_nine_rhs_gaps.csv',index=False)
    print(d.to_string(index=False))
    print(f'PASS: {new_fits} new fits, common candidate sets across weights, no additional observation drops.')


def after_nine_source_overlap():
    """Record matched food-contract geography without imputing missing locations."""
    if not CORRECTED:
        raise SystemExit('FAIL: overlap mode requires --corrected')
    rows=[]
    with duckdb.connect() as con:
        for year,contract in [(2004,'SP030004D3083'),(2005,'SPM30005D3083'),(2006,'SPM30006D3083')]:
            g=con.execute('''SELECT primary_place_of_performance_country_name country,
                primary_place_of_performance_city_name city,county_fips,geo_class,
                sum(federal_action_obligation) nominal_obligations,count(*) source_rows
                FROM read_parquet(?) WHERE award_id_piid=? OR parent_award_id_piid=? GROUP BY ALL''',
                [str(dc.USASPENDING_OUT_DIR/f'fy{year}.parquet'),contract,contract]).df()
            assert len(g)>0
            g['year'],g['contract']=year,contract
            rows.append(g)
    result=pd.concat(rows,ignore_index=True)
    result.to_csv(OUTDIR/'after_nine_worcester_contract_overlap.csv',index=False)
    print(result.to_string(index=False))


def after_nine_rhs_report():
    """Report all frozen RHS candidates, regression movements, and source traces."""
    if not CORRECTED:
        raise SystemExit('FAIL: after-nine report requires --corrected')
    e=pd.read_csv(OUTDIR/'after_nine_rhs_estimates.csv',keep_default_na=False)
    d=pd.read_csv(OUTDIR/'after_nine_rhs_gaps.csv')
    s=pd.read_csv(OUTDIR/'after_nine_rhs_selected.csv')
    cases=pd.read_csv(OUTDIR/'after_nine_rhs_case_years.csv')
    annual=pd.read_csv(OUTDIR/'h2_winsorization_source_audit_annual_after_nine_rhs.csv')
    comp=pd.read_csv(OUTDIR/'h2_winsorization_instrument_components_after_nine_rhs.csv')
    endpoints=pd.read_csv(OUTDIR/'h2_winsorization_case_endpoints_after_nine_rhs.csv')
    overlap=pd.read_csv(OUTDIR/'after_nine_worcester_contract_overlap.csv').fillna('')
    assert len(e)==36 and s.ccg.nunique()==6 and len(cases)==12 and len(annual)==31
    assert not e.duplicated(['sample','weight','variant']).any()
    np.testing.assert_allclose(e.second_stage_t,e.second_stage/e.second_stage_se,rtol=1e-12)
    np.testing.assert_allclose(annual.positive_real,annual.cached_positive_real,rtol=1e-10,atol=1e-5)
    sums=comp.groupby(['ccg','year']).z_contribution.sum()
    np.testing.assert_allclose(sums,endpoints.set_index(['ccg','year']).z.reindex(sums.index),rtol=1e-10,atol=1e-8)
    a=pd.read_parquet(CACHE/'analysis_pooled.parquet')
    max_error=0.
    for r in e.itertuples():
        excluded=list(map(int,r.excluded_ccg.split(';')))
        p=a.loc[~a.ccg.isin(excluded)]
        assert len(p)==r.n==107030-35*r.n_excluded
        assert p.ccg.nunique()==r.n_counties and p.groupby('ccg').size().eq(35).all()
        w=np.ones(len(p)) if r.weight=='equal county' else p.pop1975.to_numpy(np.float64)
        f=pd.factorize(p.fe_division_year)[0];W=np.bincount(f,weights=w)
        cols=['y','x','z'] if r.variant=='untreated' else [f'{v}_percentile_1_99' for v in ('y','x','z')]
        y,x,z=p[cols].to_numpy(np.float64).T
        zt=z-np.bincount(f,weights=w*z)[f]/W[f]
        b=(w*zt*y).sum()/(w*zt*x).sum()
        np.testing.assert_allclose(b,r.second_stage,rtol=1e-10,atol=1e-13)
        max_error=max(max_error,abs(b-r.second_stage))
    labels={f'Add {int(r.ccg)}':f'Add {r.county}, {r.state_name}' for r in s.itertuples()}
    labels.update({'Original nine':'Original nine','Add top 1 per weight':'Nine plus Calhoun and New London',
                   'Add top 3 per weight':'Nine plus all six candidates'})
    def table(frame):
        frame=frame.copy()
        if 'sample' in frame: frame['sample']=frame['sample'].map(labels)
        if 'variant' in frame: frame['variant']=frame.variant.map({'untreated':'Raw','joint_pooled_percentile_1_99':'Winsorized'})
        return frame.rename(columns={'sample':'Removed set','weight':'Weight','n_excluded':'Total removed',
            'untreated':'Raw coefficient','joint_pooled_percentile_1_99':'Winsorized coefficient',
            'gap_closed_relative_to_nine_percent':'Remaining gap closed (%)','gap_closed_relative_to_full_percent':'Full gap closed (%)',
            'raw_change':'Change in raw','winsorized_change':'Change in winsorized','variant':'Specification',
            'second_stage_x1e5':'Coefficient x 100,000','first_stage_F':'First-stage F','second_stage_t':'Second-stage t',
            'county':'County','state_name':'State','rank':'Rank','year':'Year'}).to_markdown(index=False,floatfmt='.6f')
    group=d.loc[d['sample'].isin(['Original nine','Add top 1 per weight','Add top 3 per weight'])].sort_values(['weight','n_added'])
    individual=d.loc[d.n_added.eq(1)].sort_values(['weight','sample'])
    selected=s.copy();selected['RHS covariance loss (%)']=100*selected.covariance_loss_share
    source_rows=[]
    descriptions={5013:'Hydra-70 rockets; W31P4Q10C0190 and W31P4Q14C0154; Hampton, Arkansas',
        38005:'Camouflage/deception equipment and personal armor; Fort Totten, North Dakota',
        24047:'Food procurement; SP030004D3083, SPM30005D3083 and SPM30006D3083; Pocomoke City, Maryland',
        9011:'Submarine awards; Groton, Connecticut',18141:'HMMWV trucks; DAAE0701CS001; South Bend, Indiana',
        42133:'Bradley remanufacture/reset and combat vehicles; York, Pennsylvania'}
    for r in s.itertuples():
        years=sorted(cases.loc[cases.ccg.eq(r.ccg),'year'].unique())
        source_rows.append({'County':r.county,'State':r.state_name,'Selected shock ends':', '.join(map(str,years)),
                            'Contract-record lead':descriptions[r.ccg]})
    body=f'''# RHS-selected additions to the fixed nine-county diagnostic

## Result

The six candidates selected from the RHS alone are Calhoun, Arkansas; Benson,
North Dakota; Worcester, Maryland; New London, Connecticut; York, Pennsylvania;
and St. Joseph, Indiana (Table 1). Adding all six gives a common 15-county exclusion
set under both weighting schemes.

Under population weights, the raw coefficient increases while the winsorized
coefficient changes less; the gap narrows without estimates approaching zero
(Table 2). Both population-weighted t statistics exceed five (Table 4).
Under equal weights, the gap also narrows, but the raw t statistic is below two
for both joint additions (Table 4). The effects therefore differ across weights.
These are audit candidates, not verified errors or proposed baseline exclusions.

## Candidate selection, before fitting their regressions

Keep the exact original nine county groups excluded. For each remaining county,
sum its contribution to the change in the first-stage cross-product when observed
spending x is capped, keeping the instrument z uncapped. Let z-tilde be z after
removing weighted division-by-year means on the nine-excluded sample:

```text
D = sum_i w_i * z_tilde_i * x_i
county score = abs(sum_{{i in county}}[-w_i * z_tilde_i * (x_cap_i - x_i)] / D)
```

The screen reads only x, z, the saved x cutoff transformation, weights, fixed-effect
keys, and county/year identifiers. It does not read employment y or fit any outcome
regression. Select the top three counties separately under each weight and take
their union. Test all six individually and two unions: the top one under each
weight, and all six. No candidates are selected or discarded based on the resulting
coefficient, gap, F statistic, or t statistic.

This repeats the spending-capping covariance screen, not a screen for every possible
effect of instrument capping. The signed covariance losses in Table 1 are positive;
selection uses their absolute values. These are changes in an unnormalized
cross-product, not changes in a correlation coefficient or partial R-squared.

## Table 1. Frozen RHS candidate list

{table(selected[['weight','rank','county','state_name','RHS covariance loss (%)']])}

Source: [selected list](../../Results/instrument_normalization/corrected_gap_audit/after_nine_rhs_selected.csv).
The [full ranking](../../Results/instrument_normalization/corrected_gap_audit/after_nine_rhs_ranking.csv)
contains all 3,049 remaining county groups under each weighting scheme.

## Table 2. Joint additions to the original nine

All coefficients are multiplied by 100,000. The full gap is measured against the
unrestricted corrected sample. The remaining gap is measured against the fixed
nine-county removal. These are different denominators and should not be conflated.

{table(group[['weight','sample','n_excluded','untreated','joint_pooled_percentile_1_99',
             'gap_closed_relative_to_nine_percent','gap_closed_relative_to_full_percent','raw_change','winsorized_change']])}

Source: [gap and coefficient movements](../../Results/instrument_normalization/corrected_gap_audit/after_nine_rhs_gaps.csv).

## Table 3. Each candidate added separately

Every row retains the original nine removals and adds just the named county.
Negative gap closure means the gap widens. The change in raw minus the change in
winsorized equals the signed reduction in the gap. Individual effects are not
additive because all models are refitted on their respective samples.

{table(individual[['weight','sample','untreated','joint_pooled_percentile_1_99',
                  'gap_closed_relative_to_nine_percent','raw_change','winsorized_change']])}

Source: the same gap file as Table 2. All candidates are shown, including those
that widen the gap under equal weights.

## Table 4. Full first-stage and second-stage results

{table(e.sort_values(['weight','n_added','sample','variant'])[['weight','sample','variant','second_stage_x1e5','first_stage_F','second_stage_t','n','n_counties']])}

Source: [all 36 coefficient rows](../../Results/instrument_normalization/corrected_gap_audit/after_nine_rhs_estimates.csv).
Four rows reuse the original-nine estimates; 32 rows estimate the specified additions.

## Table 5. Source-record leads for all six candidates

{pd.DataFrame(source_rows).to_markdown(index=False)}

Two shock-end years per candidate are selected by the largest absolute contributions
to its RHS covariance score. For each shock year t, the source audit reads procurement
in t-2, t-1, and t. The [12 selected windows](../../Results/instrument_normalization/corrected_gap_audit/after_nine_rhs_case_years.csv),
[annual source totals](../../Results/instrument_normalization/corrected_gap_audit/h2_winsorization_source_audit_annual_after_nine_rhs.csv),
and [largest transaction records](../../Results/instrument_normalization/corrected_gap_audit/h2_winsorization_source_audit_top_actions_after_nine_rhs.csv)
provide the evidence behind these leads. The identified programs are not independently
validated as local production or local value added. Award timing and performance
locations require separate evidence before any correction.

### Worcester: food-contract location remains unresolved

The DCADS source assigns the named food contracts to Pocomoke City, Worcester County.
Same-year USAspending matches for those identifiers largely lack a county assignment;
the available records therefore do not independently confirm Worcester. Table 6
reports the overlap without filling missing locations. Some matched records have
other geography, but contract identifiers alone do not establish that the same
individual transaction is assigned differently. No reassignment is warranted from
this match alone.

## Table 6. Worcester food contracts in the overlapping USAspending files

{overlap.to_markdown(index=False,floatfmt='.2f')}

Source: [contract overlap](../../Results/instrument_normalization/corrected_gap_audit/after_nine_worcester_contract_overlap.csv).
Amounts are nominal net obligations, not a reconstruction of the DCADS county totals.

## Verification and scope

All 36 coefficients agree with separately calculated weighted fixed-effect IV
moments, with maximum absolute difference {max_error:.12g}. All 31 annual source
procurement totals reconcile to the corrected spending cache; all 12 instruments
reconstruct from their category components. Source consistency is not independent
proof that government records are accurate.

All removals are whole county time series. The retained sample remains balanced
over 1982 through 2016. Division-by-year effects are refitted; standard errors are
clustered by county. Population weights remain fixed in 1975. Saved full-sample
pooled 1st/99th percentile cutoffs and retained instrument values are fixed.
No additional observations are dropped, and no underlying data or baseline is changed.

## Reproduction

Producer: `step_1/code/pipeline/102_h2_winsorization_cell_audit.py`.
Run `--corrected --after-nine-screen`, `--corrected --after-nine-fits`,
`--corrected --after-nine-sources`, `--corrected --after-nine-overlap`, and
`--corrected --after-nine-report`. The source tracing and report modes do not
estimate regressions. Saved optional estimates are reused when present, with
excluded-county sets and observation counts checked before reuse.
'''
    doc=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/AFTER_NINE_RHS_CANDIDATES.md'
    assert '\u2014' not in body
    doc.write_text(body)
    assert doc.read_text()==body
    print(f'PASS: RHS selection, 36 coefficient rows, 31 spending years, and 12 instrument reconstructions verified; wrote {doc}')


def ew_focus():
    """Isolate EW-selected additions and account for their clustered uncertainty."""
    if not CORRECTED:
        raise SystemExit('FAIL: EW focus requires --corrected')
    a=pd.read_parquet(CACHE/'analysis_pooled.parquet')
    old=pd.read_csv(OUTDIR/'after_nine_rhs_estimates.csv',keep_default_na=False)
    old=old.loc[old.weight.eq('equal county')].copy()
    original=list(map(int,old.loc[old['sample'].eq('Original nine'),'excluded_ccg'].iloc[0].split(';')))
    selected=pd.read_csv(OUTDIR/'after_nine_rhs_ranking.csv')
    ew_order=selected.loc[selected.weight.eq('equal county')].sort_values('rank').ccg.astype(int).tolist()
    assert ew_order[:5]==[5013,38005,24047,9011,46127]
    t28=module('t28_ew_focus',ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER=pf.MapDemeaner(fixef_tol=1e-12,fixef_maxiter=100000)
    variants={'untreated':['y','x','z'],'joint_pooled_percentile_1_99':[f'{v}_percentile_1_99' for v in ('y','x','z')]}
    target=OUTDIR/'ew_focus_estimates.csv'
    rows=pd.read_csv(target,keep_default_na=False).to_dict('records') if target.exists() else old.to_dict('records')
    new_fits=0
    for nadd in (3,5):
        ew_ids=ew_order[:nadd];excluded=original+ew_ids
        p=a.loc[~a.ccg.isin(excluded)]
        assert len(p)==107030-35*len(excluded)
        for variant,cols in variants.items():
            existing=[r for r in rows if r['sample']==f'Add EW top {nadd}' and r['variant']==variant]
            if existing:
                assert len(existing)==1 and set(map(int,existing[0]['excluded_ccg'].split(';')))==set(excluded)
                continue
            y,x,z=p[cols].to_numpy(np.float64).T
            fit=t28.fit2sls(y,x,z,np.ones(len(p)),p.fe_division_year.to_numpy(np.int64),np.zeros(len(p),np.int64),
                           p.cluster.to_numpy(np.int64),singleton=False,want_F=True)
            assert fit['n_dropped']==0 and fit['n']==len(p)
            rows.append(dict(sample=f'Add EW top {nadd}',weight='equal county',variant=variant,added_counties=';'.join(map(str,ew_ids)),
                excluded_ccg=';'.join(map(str,excluded)),n_added=nadd,n_excluded=len(excluded),n=fit['n'],n_counties=p.ccg.nunique(),
                first_stage_F=fit['F'],second_stage=fit['beta'],second_stage_se=fit['se'],second_stage_t=fit['beta']/fit['se'],
                second_stage_x1e5=fit['beta']*1e5,estimate_source='estimated'))
            new_fits+=1
    e=pd.DataFrame(rows)
    assert len(e)==22
    e.to_csv(target,index=False)
    names=pd.read_csv(OUTDIR/'h2_winsorization_attribution_counties.csv')[['ccg','county','state_name']].drop_duplicates('ccg')
    summaries=[];county_frames=[];cell_frames=[]
    models=e.loc[e['sample'].isin(['Original nine','Add 5013','Add EW top 3','Add EW top 5','Add top 3 per weight'])]
    for r in models.itertuples():
        removed=list(map(int,r.excluded_ccg.split(';')))
        p=a.loc[~a.ccg.isin(removed)]
        assert len(p)==r.n
        f=pd.factorize(p.fe_division_year)[0];cl,county_ids=pd.factorize(p.ccg)
        assert pd.Series(p.cluster.to_numpy()).groupby(p.ccg.to_numpy()).nunique().eq(1).all()
        y,x,z=p[variants[r.variant]].to_numpy(np.float64).T
        count=np.bincount(f)
        yt=y-np.bincount(f,weights=y)[f]/count[f]
        xt=x-np.bincount(f,weights=x)[f]/count[f]
        zt=z-np.bincount(f,weights=z)[f]/count[f]
        D=(zt*xt).sum();N=(zt*yt).sum();beta=N/D
        residual=yt-beta*xt
        score=zt*residual
        q=np.bincount(cl,weights=score)
        k=t28.ssc_k(f,np.zeros(len(p),np.int64),cl)
        ssc=len(county_ids)/(len(county_ids)-1)*(len(p)-1)/(len(p)-k)
        se=np.sqrt(ssc*(q*q).sum()/D**2)
        np.testing.assert_allclose([beta,se],[r.second_stage,r.second_stage_se],rtol=1e-10,atol=1e-13)
        c=pd.DataFrame({'ccg':county_ids,'cluster_score':q,'normalized_cluster_score':q/D,
            'variance_share':q*q/(q*q).sum(),'raw_first_stage_crossproduct':np.bincount(cl,weights=zt*xt),
            'raw_reduced_form_crossproduct':np.bincount(cl,weights=zt*yt)})
        c['sample'],c['variant']=r.sample,r.variant
        c=c.merge(names,on='ccg',validate='one_to_one')
        c=c.sort_values(['variance_share','ccg'],ascending=[False,True])
        c['variance_rank']=np.arange(1,len(c)+1)
        county_frames.append(c)
        g=p[['ccg','year']].copy()
        g['normalized_score']=score/D
        g['sample'],g['variant']=r.sample,r.variant
        selected_ids=c.head(3).ccg
        g=g.loc[g.ccg.isin(selected_ids)].merge(names,on='ccg',validate='many_to_one')
        g['rank_abs_within_county']=g.normalized_score.abs().groupby(g.ccg).rank(method='first',ascending=False).astype(int)
        cell_frames.append(g)
        summaries.append(dict(sample=r.sample,variant=r.variant,coefficient_x1e5=beta*1e5,se_x1e5=se*1e5,
            t=beta/se,F=r.first_stage_F,first_stage_crossproduct=D,reduced_form_crossproduct=N,
            cluster_score_norm=np.sqrt((q*q).sum()),variance_correction=ssc,
            top1_variance_share=c.variance_share.iloc[0],top3_variance_share=c.variance_share.head(3).sum(),
            top10_variance_share=c.variance_share.head(10).sum(),top_county=c.county.iloc[0],top_state=c.state_name.iloc[0]))
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUTDIR/'ew_precision_summary.csv',index=False)
    pd.concat(county_frames,ignore_index=True).to_csv(OUTDIR/'ew_cluster_variance.csv',index=False)
    pd.concat(cell_frames,ignore_index=True).to_csv(OUTDIR/'ew_precision_leading_years.csv',index=False)
    d=e.pivot(index=['sample','n_excluded'],columns='variant',values='second_stage_x1e5').reset_index()
    d['gap']=d.joint_pooled_percentile_1_99-d.untreated
    base=d.loc[d['sample'].eq('Original nine'),'gap'].iloc[0]
    full=pd.read_csv(REFERENCE).query("source_version=='corrected' and weight=='equal county'").set_index('variant').second_stage_x1e5
    fullgap=full['joint_pooled_percentile_1_99']-full['untreated']
    d['gap_closed_relative_to_nine_percent']=100*(1-d.gap.abs()/abs(base))
    d['gap_closed_relative_to_full_percent']=100*(1-d.gap.abs()/abs(fullgap))
    d.to_csv(OUTDIR/'ew_focus_gaps.csv',index=False)
    print(summary.to_string(index=False))
    print(d.loc[d['sample'].isin(models['sample'])].to_string(index=False))
    print(pd.concat(county_frames).query('variance_rank<=3')[['sample','variant','county','state_name','variance_share']].to_string(index=False))
    print(f'PASS: {new_fits} new EW fits; variance components reproduce all {len(models)} saved SEs.')


def ew_source_continuity():
    """Track named contracts and national reporting coverage, without altering data."""
    if not CORRECTED:
        raise SystemExit('FAIL: EW continuity requires --corrected')
    geography=[];parents=[];coverage=[]
    with duckdb.connect() as con:
        for year in range(1987,1991):
            path=dc.OUT_DIR/f'fy{year}.parquet'
            d=con.execute('''SELECT contract_no,county_fips5,city_name,county_name,fsc,
                count(*) source_rows,
                sum(dollars*CASE WHEN dollars_unit='thousands' THEN 1000 ELSE 1 END) net_nominal
                FROM read_parquet(?) WHERE contract_no IN
                ('DABT1086C1853','DABT1087C1824','DABC0968DC393') GROUP BY ALL''',[str(path)]).df()
            assert not d.empty
            d['year'],d['source_file']=year,str(path)
            geography.append(d)
        for year in range(2012,2019):
            path=dc.USASPENDING_OUT_DIR/f'fy{year}.parquet'
            d=con.execute('''SELECT award_or_idv_flag,
                count(*) source_rows, sum(federal_action_obligation) net_nominal,
                sum(greatest(federal_action_obligation,0)) positive_nominal,
                count(*) FILTER (WHERE product_or_service_code='8905') meat_rows,
                coalesce(sum(federal_action_obligation) FILTER (WHERE product_or_service_code='8905'),0) meat_net_nominal,
                coalesce(sum(greatest(federal_action_obligation,0)) FILTER (WHERE product_or_service_code='8905'),0) meat_positive_nominal
                FROM read_parquet(?) WHERE starts_with(parent_award_id_piid,'HDEC')
                OR starts_with(award_id_piid,'HDEC') GROUP BY ALL''',[str(path)]).df()
            d['year'],d['source_file']=year,str(path)
            coverage.append(d)
            if year<=2016:
                for parent in ('HDEC0212D0007','HDEC0208D0005'):
                    d=con.execute('''SELECT count(*) source_rows,
                        coalesce(sum(federal_action_obligation),0) net_nominal,
                        coalesce(sum(greatest(federal_action_obligation,0)),0) positive_nominal,
                        string_agg(DISTINCT county_fips,';') assigned_counties,
                        string_agg(DISTINCT primary_place_of_performance_city_name,';') cities,
                        string_agg(DISTINCT product_or_service_code,';') categories,
                        string_agg(DISTINCT awarding_sub_agency_name,';') agencies
                        FROM read_parquet(?) WHERE award_or_idv_flag='AWARD'
                        AND parent_award_id_piid=?''',[str(path),parent]).df()
                    d['year'],d['parent'],d['source_file']=year,parent,str(path)
                    d['status']=np.where(d.source_rows.eq(0),'no records, not economic zero','records present')
                    parents.append(d)
    for filename,frames in [('ew_fort_benning_contract_geography.csv',geography),
                            ('ew_union_parent_continuity.csv',parents),('ew_hdec_reporting_coverage.csv',coverage)]:
        d=pd.concat(frames,ignore_index=True)
        d.to_csv(OUTDIR/filename,index=False)
        assert len(pd.read_csv(OUTDIR/filename))==len(d)
        print(filename, len(d), 'rows')
    print('PASS: source continuity saved; no spending, instrument, sample, or employment changes.')


def ew_focus_report():
    """Document EW inference and inspected evidence on RHS reporting continuity."""
    import hashlib,json
    if not CORRECTED:
        raise SystemExit('FAIL: EW report requires --corrected')
    e=pd.read_csv(OUTDIR/'ew_focus_estimates.csv')
    s=pd.read_csv(OUTDIR/'ew_precision_summary.csv')
    gaps=pd.read_csv(OUTDIR/'ew_focus_gaps.csv').set_index('sample')
    v=pd.read_csv(OUTDIR/'ew_cluster_variance.csv')
    geography=pd.read_csv(OUTDIR/'ew_fort_benning_contract_geography.csv')
    parents=pd.read_csv(OUTDIR/'ew_union_parent_continuity.csv')
    coverage=pd.read_csv(OUTDIR/'ew_hdec_reporting_coverage.csv')
    annual=pd.read_csv(OUTDIR/'h2_winsorization_source_audit_annual_ew_precision.csv')
    components=pd.read_csv(OUTDIR/'h2_winsorization_instrument_components_ew_precision.csv')
    np.testing.assert_allclose(annual.positive_real,annual.cached_positive_real,rtol=1e-8,atol=.01)
    totals=components.groupby(['ccg','year']).agg(rebuilt=('z_contribution','sum'),cached=('z_total','first'))
    np.testing.assert_allclose(totals.rebuilt,totals.cached,rtol=1e-8,atol=1e-6)
    docs=dc.MACRO_RESULTS_ROOT.parent/'data/source/primary_documents/raw'
    manifest=json.loads((docs/'deca_solicitation_media6557.fetch.json').read_text())
    assert manifest['final_status']=='official-artifact'
    assert hashlib.sha256(Path(manifest['artifact_path']).read_bytes()).hexdigest()==manifest['sha256']
    source=(docs/'deca_solicitation_media6557.chunked.txt').read_text()
    page8=source.split('===== PAGE 8 =====')[1].split('===== PAGE 9 =====')[0]
    assert 'Effective October 1, 2014' in page8 and 'not required to be reported' in page8
    assert len(e)==22 and len(s)==10 and len(totals)==12
    def table(head,rows):
        return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
            ['| '+' | '.join(map(str,row))+' |' for row in rows])
    labels={'Original nine':'Original nine','Add 5013':'Nine + Calhoun, Arkansas',
            'Add EW top 3':'Nine + EW top three','Add EW top 5':'Nine + EW top five',
            'Add top 3 per weight':'Nine + six from both weighting screens'}
    rows=[]
    for sample,label in labels.items():
        a=e.loc[e['sample'].eq(sample)].set_index('variant')
        raw=a.loc['untreated'];cap=a.loc['joint_pooled_percentile_1_99']
        rows.append([label,int(raw.n_excluded),f'{raw.second_stage_x1e5:.6f}',f'{raw.second_stage_se*1e5:.6f}',
            f'{raw.second_stage_t:.6f}',f'{raw.first_stage_F:.6f}',f'{cap.second_stage_x1e5:.6f}',
            f'{cap.second_stage_t:.6f}',f'{cap.first_stage_F:.6f}',f'{gaps.loc[sample,"gap_closed_relative_to_full_percent"]:.6f}%'])
    estimates=table(['Sample','Counties removed','Raw beta','Raw SE','Raw t','Raw F','Capped beta','Capped t','Capped F','Full gap closed'],rows)
    raw=s.loc[s.variant.eq('untreated')].set_index('sample')
    precision=table(['Sample','Spending-instrument cross-product D','Residual-score norm','SE x 100,000'],[
        [labels[k],f'{raw.loc[k,"first_stage_crossproduct"]:,.2f}',f'{raw.loc[k,"cluster_score_norm"]:,.6f}',
         f'{raw.loc[k,"se_x1e5"]:.6f}'] for k in labels])
    leaders=v.loc[v['sample'].eq('Add EW top 3')&v.variant.eq('untreated')].sort_values('variance_rank').head(3)
    variance=table(['County','State','Share of clustered variance'],[
        [r.county.title(),r.state_name,f'{100*r.variance_share:.6f}%'] for r in leaders.itertuples()])
    union=annual.loc[annual.ccg.eq(46127)].sort_values('year')
    uniontable=table(['Fiscal year','Union positive nominal obligations'],[[r.year,f'${r.positive_nominal:,.2f}'] for r in union.itertuples()])
    cov=coverage.loc[coverage.award_or_idv_flag.eq('AWARD')].sort_values('year')
    covtable=table(['Fiscal year','HDEC award rows, all categories','Meat award rows','Meat positive obligations'],[
        [r.year,r.source_rows,r.meat_rows,f'${r.meat_positive_nominal:,.2f}'] for r in cov.itertuples()])
    geotable=table(['Fiscal year','Contract','County','City','Category','Net nominal obligations'],[
        [r.year,r.contract_no,r.county_name,r.city_name,r.fsc,f'${r.net_nominal:,.2f}']
        for r in geography.sort_values(['contract_no','year','county_fips5']).itertuples()])
    Ddrop=100*(1-raw.loc['Add EW top 3','first_stage_crossproduct']/raw.loc['Original nine','first_stage_crossproduct'])
    qdrop=100*(1-raw.loc['Add EW top 3','cluster_score_norm']/raw.loc['Original nine','cluster_score_norm'])
    body=f'''# Equal-weight estimates: precision and RHS source audit

## Conclusion

The EW top-five RHS screen closes {gaps.loc['Add EW top 5','gap_closed_relative_to_full_percent']:.6f}% of the full raw/capped coefficient gap,
but the raw t-statistic is {raw.loc['Add EW top 5','t']:.6f} (Table 1). It is not a strong common estimate.
The immediate reason precision deteriorates is that the omitted counties contribute
more to the spending-instrument cross-product than to residual uncertainty (Table 2).

There is also a documented coverage problem: DeCA resale actions are exempt from
FPDS reporting starting October 1, 2014. The official contract document, PDF page 8,
paragraph 13, states this explicitly. Union's missing meat awards coincide with that
date (Tables 4 and 5). Missing reports must not be treated as proof of zero procurement.
The contribution of this reporting break to the coefficient gap has not been estimated.
No baseline data, instruments, employment records, or sample rules are changed.

## Design and selection

Equal county weights; h=2; division-by-year fixed effects; county-clustered standard
errors. The certified full panel has 107,030 observations, 3,058 county groups and
35 years, 1982 to 2016. Each exclusion removes all 35 observations for that county.
There are no additional observation drops. The full-sample pooled 1st/99th-percentile
cutoffs and instrument values stay fixed across samples. Capping applies jointly to
the outcome, observed spending shock, and instrument.

The original nine are Austin, Texas; Sagadahoc, Maine; Newport News city, Virginia;
St. Louis city, Missouri; St. James, Louisiana; Macomb, Michigan; Jackson, Mississippi;
Nassau, New York; and Tioga, New York.
Conditional on those removals, the EW RHS ranking selects Calhoun, Arkansas;
Benson, North Dakota; Worcester, Maryland; New London, Connecticut; and Union,
South Dakota, in that order. The ranking uses the absolute county contribution to
the change in the spending-instrument cross-product when spending is capped.
It does not use employment, the second-stage coefficient, or its t-statistic.
This is an influence diagnostic, not a defensible baseline exclusion rule by itself.

## Table 1. EW estimates

Coefficients and standard errors multiplied by 100,000. Gap closure uses the
unrestricted corrected-data EW raw/capped gap as the denominator.

{estimates}

## Table 2. Why the standard error rises

{precision}

With tildes denoting division-year residuals, D = sum(z_tilde*x_tilde),
u = y_tilde - beta*x_tilde, and q_c = sum within county c of z_tilde*u.
The county-clustered SE is sqrt(SSC)*sqrt(sum(q_c^2))/abs(D), where SSC is the
regression's degrees-of-freedom correction. The accounting reproduces all ten saved
standard errors to numerical tolerance.

From the nine-removal sample to the EW top-three addition, D falls {Ddrop:.6f}%,
but the residual-score norm falls only {qdrop:.6f}%. The coefficient is consequently
estimated less precisely, even though the first-stage F rises (Table 1).

## Table 3. Leading contributions to raw EW variance after the top-three addition

{variance}

These three counties account for {100*leaders.variance_share.sum():.6f}% of clustered variance.
Variance contributions are not the effects of deleting counties and do not establish
data errors. The source audit checks the two largest absolute-score years for each
of the three leading counties in the raw and capped specifications.

## Table 4. Union County, South Dakota: recorded spending

{uniontable}

The two meat contracts HDEC0212D0007 and HDEC0208D0005 have combined nominal
obligations of ${parents.loc[parents.year.eq(2014),'net_nominal'].sum():,.2f} in fiscal 2014.
Both are assigned to Dakota Dunes, Union County, and the awarding agency is the
Defense Commissary Agency. Neither appears in fiscal 2015 or 2016 in the parsed
award records. The national check in Table 5 uses HDEC-prefixed parent or award
identifiers, without imposing a county filter. The prefix screen is not asserted to
capture every DeCA contract or to identify all resale purchases.

## Table 5. National HDEC reporting coverage

Meat is product/service category 8905. Zero rows mean no observed reports, not zero
economic activity. Positive obligations are nominal dollars.

{covtable}

Official evidence: [DeCA contract document, PDF page 8, paragraph 13](https://prod.commissaries.com/media/6557).
The document carries solicitation HDEC02-16-R-0001 dated November 6, 2015.
Its reporting exemption takes effect at the start of fiscal 2015. This independently
supports a coverage-break interpretation of the missing resale records; it does not
supply the missing expenditure amounts or prove that actual purchases were constant.
The fiscal 2012-to-2013 discontinuity in Table 5 requires a separate historical check.

Acquisition: official-artifact, 64 PDF pages. Evidence: official-clause-inspected,
page 8, paragraph 13. Artifact SHA-256: `{manifest['sha256']}`.
Local artifact: `data/source/primary_documents/raw/deca_solicitation_media6557.pdf`.
The separate CRS R44010 version 11 acquisition ended in
`identity-mismatch-after-ladder`; it is not used as evidence here.

## Table 6. Fort Benning contract geography

{geotable}

The same Fort Benning service-contract identifiers appear under Chattahoochee and
Muscogee across 1988 and 1989. The audit also records a Rolla-to-Cusseta assignment
for DABT1087C1824. These are leads for performance-location verification, not proven
misassignments: a contract can cover more than one location. No geography correction
is justified solely by this table.

## Scope, verification, and next test

All {len(annual)} annual source totals reconcile with their county cache values and
all {len(totals)} selected instrument values reconstruct from category components.
This verifies pipeline consistency, not independent accuracy of the source records.
The study adds four EW fits, two variants for each of the top-three and top-five
sets. The other estimates are reused. No population-weighted fits are run.

The next substantive test is a consistent reporting-coverage treatment: identify
the affected resale records, apply the same coverage definition across years in
both observed spending and national shifters/exposures, then compare raw and capped
EW estimates. Dropping Union alone cannot repair a national reporting break.
The coverage treatment must be specified before using its F or t to judge results.
That rebuild and its effect on the coefficient gap remain untested.

## Reproduction and machine-readable evidence

Producer: `step_1/code/pipeline/102_h2_winsorization_cell_audit.py`.
Use `--corrected` with `--ew-focus`, `--ew-sources`, `--ew-continuity`, and
`--ew-report`. The fit mode reuses saved optional fits when available.
Results are under `Results/instrument_normalization/corrected_gap_audit/`:

- `ew_focus_estimates.csv`, `ew_focus_gaps.csv`: full inference and gap comparisons.
- `ew_precision_summary.csv`, `ew_cluster_variance.csv`, `ew_precision_leading_years.csv`: exact variance accounting.
- `ew_fort_benning_contract_geography.csv`, `ew_union_parent_continuity.csv`, `ew_hdec_reporting_coverage.csv`: source continuity with input paths.
- `h2_winsorization_source_audit_*_ew_precision.csv`, `h2_winsorization_instrument_components_ew_precision.csv`: source records and category reconstruction.
'''
    assert '\u2014' not in body
    path=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/EW_PRECISION_AND_RHS_AUDIT.md'
    path.write_text(body)
    assert path.read_text()==body
    print(f'PASS: EW fits, source reconstruction, and official clause verified; wrote {path}')


def original_groups_report():
    """Compare identical fixed county groups across archived and corrected inputs."""
    if not CORRECTED:
        raise SystemExit('FAIL: original group report requires --corrected')
    old=pd.read_csv(OUTDIR.parent/'h2_winsorization_leaveout_estimates_next_group.csv',keep_default_na=False)
    new=pd.read_csv(OUTDIR/'h2_winsorization_leaveout_estimates_original_groups.csv',keep_default_na=False)
    assert len(old)==len(new)==12
    keys=['weight','variant','n_excluded']
    comparison=old.merge(new,on=keys,suffixes=('_archived','_corrected'),validate='one_to_one')
    assert comparison.excluded_ccg_archived.eq(comparison.excluded_ccg_corrected).all()
    assert comparison.n_archived.eq(comparison.n_corrected).all()
    a=pd.read_parquet(CACHE/'analysis_pooled.parquet')
    max_error=0.
    for r in new.itertuples():
        removed=list(map(int,r.excluded_ccg.split(';'))) if r.n_excluded else []
        p=a.loc[~a.ccg.isin(removed)]
        assert len(p)==r.n==107030-35*r.n_excluded
        w=np.ones(len(p)) if r.weight=='equal county' else p.pop1975.to_numpy(np.float64)
        f=pd.factorize(p.fe_division_year)[0]; W=np.bincount(f,weights=w)
        cols=['y','x','z'] if r.variant=='untreated' else [f'{v}_percentile_1_99' for v in ('y','x','z')]
        y,x,z=p[cols].to_numpy(np.float64).T
        zt=z-np.bincount(f,weights=w*z)[f]/W[f]
        beta=(w*zt*y).sum()/(w*zt*x).sum()
        np.testing.assert_allclose(beta,r.second_stage,rtol=1e-10,atol=1e-13)
        max_error=max(max_error,abs(beta-r.second_stage))
    combined=pd.concat([old.assign(input_version='Archived inputs'),new.assign(input_version='Report-corrected inputs')],ignore_index=True)
    combined.to_csv(OUTDIR/'original_groups_input_comparison.csv',index=False)
    gap=combined.pivot(index=['input_version','weight','n_excluded'],columns='variant',values='second_stage_x1e5').reset_index()
    gap['gap']=gap.joint_pooled_percentile_1_99-gap.untreated
    full=gap.loc[gap.n_excluded.eq(0)].set_index(['input_version','weight']).gap
    gap['full_gap']=[full.loc[(r.input_version,r.weight)] for r in gap.itertuples()]
    gap['gap_closed_percent']=100*(1-gap.gap.abs()/gap.full_gap.abs())
    gap.to_csv(OUTDIR/'original_groups_input_comparison_gaps.csv',index=False)
    combined=combined.merge(gap[['input_version','weight','n_excluded','gap_closed_percent']],
                            on=['input_version','weight','n_excluded'],validate='many_to_one')
    names=pd.read_csv(dc.CCG_CROSSWALK)
    names['ccg']=pd.to_numeric(names.ccg_id,errors='coerce')
    names=names.dropna(subset=['ccg']).drop_duplicates('ccg').set_index('ccg').ccg_name
    membership=[]
    for n in (4,9):
        row=new.loc[new.n_excluded.eq(n)].iloc[0]
        for ccg in map(int,row.excluded_ccg.split(';')):
            membership.append({'Removed':n,'County':names.loc[ccg].title(),'State':STATE_NAMES[ccg//1000],'ccg':ccg})
    membership=pd.DataFrame(membership)
    membership.to_csv(OUTDIR/'original_groups_membership.csv',index=False)
    def table(frame):
        frame=frame.copy()
        if 'variant' in frame:
            frame['variant']=frame.variant.map({'untreated':'Raw','joint_pooled_percentile_1_99':'Winsorized x, y, z'})
        return frame.rename(columns={'input_version':'Inputs','weight':'Weight','n_excluded':'Removed',
            'variant':'Specification','second_stage_x1e5':'Coefficient x 100,000',
            'second_stage_t':'Second-stage t','first_stage_F':'First-stage F',
            'gap_closed_percent':'Gap closed (%)'}).to_markdown(index=False,floatfmt='.6f')
    corrected=combined.loc[combined.input_version.eq('Report-corrected inputs')].sort_values(['weight','n_excluded','variant'])
    compact=combined.loc[combined.n_excluded.gt(0)].sort_values(['weight','n_excluded','input_version','variant'])
    body=f'''# Fixed four- and nine-county removals across input versions

## Scope

The four-county set contains Austin County, Texas; Sagadahoc County, Maine;
Newport News city, Virginia; and St. Louis city, Missouri. The nine-county set
adds the five county groups in Table 3. Membership is read from the archived
estimation file and is identical across input versions and weighting schemes.
Neither set uses the top-40/50 ranking by individual gap closure.

## Table 1. Report-corrected inputs

Coefficients are multiplied by 100,000. The gap is the winsorized coefficient
minus the raw coefficient. Its percentage reduction is measured against the
full-sample gap in the same input version and weighting scheme.

{table(corrected[['weight','n_excluded','variant','second_stage_x1e5','first_stage_F','second_stage_t','gap_closed_percent']])}

Source: [corrected fits](../../Results/instrument_normalization/corrected_gap_audit/h2_winsorization_leaveout_estimates_original_groups.csv).

## Table 2. Matched comparison across input versions

This comparison holds the identities of excluded counties fixed. Each input
version supplies its own full-sample instrument values and pooled capping
cutoffs. Cutoffs and retained instruments are held fixed within each version
when counties are removed. The comparison therefore includes the consequences
of the contract corrections for spending, instruments, and capping thresholds;
it is not a cutoff-fixed decomposition of the data corrections alone.

{table(compact[['input_version','weight','n_excluded','variant','second_stage_x1e5','first_stage_F','second_stage_t','gap_closed_percent']])}

Source: [matched coefficient comparison](../../Results/instrument_normalization/corrected_gap_audit/original_groups_input_comparison.csv)
and [within-version gaps](../../Results/instrument_normalization/corrected_gap_audit/original_groups_input_comparison_gaps.csv).

## Table 3. Exact county-group membership

{membership.drop(columns='ccg').to_markdown(index=False)}

Source: [membership with identifiers](../../Results/instrument_normalization/corrected_gap_audit/original_groups_membership.csv).

## Interpretation

The fixed four- and nine-county removals retain positive estimates on the
report-corrected inputs (Table 1). They close part of the gap without the
near-zero coefficients obtained from the separately ranked large exclusion sets.
These are different diagnostics, not conflicting estimates of the same reduced
sample. Adding the five counties reduces the gap further under population
weights but increases the remaining gap under equal weights (Table 1).
Neither exercise establishes that the omitted counties contain erroneous data.

## Estimation and verification

The horizon is two years. Fixed effects are division by year; standard errors
are clustered by county. Weights are equal county weights or population fixed
in 1975. Winsorization jointly caps x, y, and z at their full-sample, pooled
1st and 99th percentiles. County removals preserve all 35 years for every
retained unit, with 3,054 county groups for the four-county omission and 3,049
for the nine-county omission.

Eight corrected-input fits are estimated; four full-sample rows and all twelve
archived-input rows are reused. There are no additional observation drops.
The twelve corrected coefficients agree with weighted fixed-effect IV moments;
maximum absolute coefficient discrepancy is {max_error:.12g}. The archived
county identifiers and sample counts match the corrected comparisons exactly.
No underlying data, instrument design, exposure window, or baseline sample is changed.

## Reproduction

Producer: `step_1/code/pipeline/102_h2_winsorization_cell_audit.py`.
Run `--corrected --original-groups`, then `--corrected --original-groups-report`.
The source for the fixed memberships and archived estimates is
`Results/instrument_normalization/h2_winsorization_leaveout_estimates_next_group.csv`.
The analysis input is the certified `report_correction_rebuild/analysis_pooled.parquet`.
'''
    doc=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/ORIGINAL_COUNTY_GROUPS_CORRECTED_CHECK.md'
    assert '\u2014' not in body
    doc.write_text(body)
    assert doc.read_text()==body
    print(f'PASS: fixed memberships, sample counts, and corrected coefficients verified; wrote {doc}')


def all_county_leaveouts():
    """Rank exact county-omission gaps and fit fixed top-5/10/20 groups."""
    if not CORRECTED:
        raise SystemExit('FAIL: --all-counties requires --corrected')
    import time
    started=time.monotonic()
    a=pd.read_parquet(CACHE/'analysis_pooled.parquet')
    variants={'untreated':['y','x','z'],
              'joint_pooled_percentile_1_99':[f'{v}_percentile_1_99' for v in ('y','x','z')]}
    cols=list(dict.fromkeys(sum(variants.values(),[])+['ccg','year','pop1975','fe_division_year','cluster']))
    assert len(a)==107030 and a.ccg.nunique()==3058
    assert not a.duplicated(['ccg','year']).any()
    assert a.groupby('ccg').size().eq(35).all()
    assert a.groupby('ccg').pop1975.nunique().eq(1).all() and a.pop1975.gt(0).all()
    assert np.isfinite(a[cols].to_numpy()).all()
    reference=pd.read_csv(REFERENCE)
    reference=reference.loc[reference.source_version.eq('corrected')&reference.variant.isin(variants)]
    assert len(reference)==4 and not reference.duplicated(['weight','variant']).any()
    names=pd.read_csv(OUTDIR/'h2_winsorization_attribution_counties.csv')
    names=names[['ccg','county','state_name']].drop_duplicates('ccg').set_index('ccg')
    ids=np.sort(a.ccg.unique())
    assert set(ids).issubset(names.index)
    values={v:a[cs].to_numpy(np.float64) for v,cs in variants.items()}
    county=a.ccg.to_numpy(np.int64)
    fe=a.fe_division_year.to_numpy(np.int64)
    cl=a.cluster.to_numpy(np.int64)
    weights={'equal county':np.ones(len(a)),'fixed 1975 population':a.pop1975.to_numpy(np.float64)}
    t28=module('t28_all_county_omissions',ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER=pf.MapDemeaner(fixef_tol=1e-12,fixef_maxiter=100000)
    def fit_subset(removed,weight,variant):
        keep=~np.isin(county,removed)
        expected=len(a)-35*len(removed)
        assert keep.sum()==expected
        y,x,z=values[variant][keep].T
        fit=t28.fit2sls(y,x,z,weights[weight][keep],fe[keep],np.zeros(expected,np.int64),
                       cl[keep],singleton=False,want_F=True)
        assert fit['n']==expected and fit['n_dropped']==0
        assert np.isfinite([fit['beta'],fit['se'],fit['F']]).all() and fit['se']>0
        return dict(weight=weight,variant=variant,second_stage=fit['beta'],second_stage_se=fit['se'],
                    second_stage_t=fit['beta']/fit['se'],first_stage_F=fit['F'],n=fit['n'],
                    n_counties=len(ids)-len(removed),n_excluded=len(removed),
                    second_stage_x1e5=fit['beta']*1e5,estimate_source='estimated',
                    cutoffs_fixed=True,retained_instruments_fixed=True)
    saved=pd.read_csv(OUTDIR/'h2_winsorization_leaveout_estimates.csv')
    saved=saved.loc[saved.n_excluded.eq(1)&saved.excluded_ccg.eq(48015)].copy()
    assert len(saved)==4 and saved.n.eq(106995).all()
    rows=[]
    for i,ccg in enumerate(ids,1):
        for weight in weights:
            for variant in variants:
                if ccg==48015:
                    row=saved.loc[saved.weight.eq(weight)&saved.variant.eq(variant)].iloc[0].to_dict()
                    row['estimate_source']='saved_austin'
                else:
                    row=fit_subset([ccg],weight,variant)
                row.update(excluded_ccg=int(ccg),county=names.loc[ccg,'county'],state_name=names.loc[ccg,'state_name'])
                rows.append(row)
        if i%100==0 or i==len(ids):
            pd.DataFrame(rows).to_csv(OUTDIR/'all_county_leaveout_estimates.csv',index=False)
            print(f'County omissions: {i}/{len(ids)}; elapsed {time.monotonic()-started:.1f}s',flush=True)
    estimates=pd.DataFrame(rows)
    assert len(estimates)==4*len(ids)
    assert not estimates.duplicated(['excluded_ccg','weight','variant']).any()
    base=reference.pivot(index='weight',columns='variant',values='second_stage_x1e5')
    base_gap=base.joint_pooled_percentile_1_99-base.untreated
    ranks=estimates.pivot(index=['excluded_ccg','county','state_name','weight'],columns='variant',
                         values='second_stage_x1e5').reset_index()
    ranks['gap_x1e5']=ranks.joint_pooled_percentile_1_99-ranks.untreated
    ranks['full_sample_gap_x1e5']=ranks.weight.map(base_gap)
    ranks['gap_closed_x1e5']=ranks.full_sample_gap_x1e5.abs()-ranks.gap_x1e5.abs()
    ranks['gap_reduction_percent']=100*ranks.gap_closed_x1e5/ranks.full_sample_gap_x1e5.abs()
    ranks=ranks.sort_values(['weight','gap_reduction_percent','excluded_ccg'],ascending=[True,False,True])
    ranks['rank']=ranks.groupby('weight').cumcount()+1
    ranks.to_csv(OUTDIR/'all_county_leaveout_ranking.csv',index=False)
    groups=[]
    memberships=[]
    for weight in weights:
        order=ranks.loc[ranks.weight.eq(weight)]
        for k in (5,10,20):
            selected=order.head(k)
            removed=selected.excluded_ccg.to_numpy(np.int64)
            for r in selected.itertuples():
                memberships.append(dict(weight=weight,n_excluded=k,rank=r.rank,ccg=r.excluded_ccg,
                                        county=r.county,state_name=r.state_name))
            for variant in variants:
                row=fit_subset(removed,weight,variant)
                row['excluded_ccg']=';'.join(map(str,removed))
                groups.append(row)
    groups=pd.DataFrame(groups)
    assert len(groups)==12
    groups.to_csv(OUTDIR/'top_county_group_estimates.csv',index=False)
    pd.DataFrame(memberships).to_csv(OUTDIR/'top_county_group_membership.csv',index=False)
    joint=groups.pivot(index=['weight','n_excluded'],columns='variant',values='second_stage_x1e5').reset_index()
    joint['gap_x1e5']=joint.joint_pooled_percentile_1_99-joint.untreated
    joint['full_sample_gap_x1e5']=joint.weight.map(base_gap)
    joint['gap_reduction_percent']=100*(1-joint.gap_x1e5.abs()/joint.full_sample_gap_x1e5.abs())
    joint.to_csv(OUTDIR/'top_county_group_gaps.csv',index=False)
    print(ranks.loc[ranks['rank'].le(5)].to_string(index=False),flush=True)
    print(joint.to_string(index=False),flush=True)
    print('PASS: all 3058 county omissions, 12 joint fits; no additional drops; full-sample cutoffs and retained instruments fixed.',flush=True)


def gap50_analysis():
    """Fit ranked top-20/30/40/50 omissions and split closure by LHS/RHS capping."""
    if not CORRECTED:
        raise SystemExit('FAIL: --gap50 requires --corrected')
    a=pd.read_parquet(CACHE/'analysis_pooled.parquet')
    assert len(a)==107030 and a.ccg.nunique()==3058
    assert not a.duplicated(['ccg','year']).any() and a.groupby('ccg').size().eq(35).all()
    assert a.groupby('ccg').pop1975.nunique().eq(1).all() and a.pop1975.gt(0).all()
    ranks=pd.read_csv(OUTDIR/'all_county_leaveout_ranking.csv').query('rank<=50')
    singles=pd.read_csv(OUTDIR/'all_county_leaveout_estimates.csv')
    groups=pd.read_csv(OUTDIR/'top_county_group_estimates.csv')
    ref=pd.read_csv(REFERENCE).query("source_version=='corrected'")
    variants={'untreated':['y','x','z'],
        'lhs_only':['y_percentile_1_99','x','z'],
        'rhs_only':['y','x_percentile_1_99','z_percentile_1_99'],
        'joint_pooled_percentile_1_99':['y_percentile_1_99','x_percentile_1_99','z_percentile_1_99']}
    t28=module('t28_gap50',ROOT/'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER=pf.MapDemeaner(fixef_tol=1e-12,fixef_maxiter=100000)
    output=OUTDIR/'gap50_estimates.csv'
    rows=pd.read_csv(output).to_dict('records') if output.exists() else []
    keys={(r['weight'],r['sample_id'],r['variant']) for r in rows}
    new_fits=0
    memberships=[]
    for weight in ('equal county','fixed 1975 population'):
        order=ranks.loc[ranks.weight.eq(weight)].sort_values('rank')
        assert len(order)==50
        samples=[('full',[])]+[(f'county_{int(r.excluded_ccg)}',[int(r.excluded_ccg)]) for r in order.itertuples()]
        for k in (20,30,40,50):
            removed=order.head(k).excluded_ccg.astype(int).tolist()
            samples.append((f'top_{k}',removed))
            memberships.extend(dict(weight=weight,k=k,ccg=int(r.excluded_ccg),rank=int(r.rank),
                county=r.county,state_name=r.state_name) for r in order.head(k).itertuples())
        for sample_id,removed in samples:
            p=a.loc[~a.ccg.isin(removed)]
            expected=107030-35*len(removed)
            assert len(p)==expected
            w=np.ones(expected) if weight=='equal county' else p.pop1975.to_numpy(np.float64)
            fe=p.fe_division_year.to_numpy(np.int64)
            for variant,cols in variants.items():
                key=(weight,sample_id,variant)
                if key in keys:
                    previous=next(r for r in rows if (r['weight'],r['sample_id'],r['variant'])==key)
                    assert previous['n']==expected
                    continue
                saved=pd.DataFrame()
                if variant in ('untreated','joint_pooled_percentile_1_99'):
                    if sample_id=='full':
                        saved=ref.loc[ref.weight.eq(weight)&ref.variant.eq(variant)]
                    elif sample_id.startswith('county_'):
                        saved=singles.loc[singles.weight.eq(weight)&singles.variant.eq(variant)&singles.excluded_ccg.eq(removed[0])]
                    elif len(removed)==20:
                        saved=groups.loc[groups.weight.eq(weight)&groups.variant.eq(variant)&groups.n_excluded.eq(20)]
                        assert set(map(int,str(saved.iloc[0].excluded_ccg).split(';')))==set(removed)
                if len(saved):
                    assert len(saved)==1
                    sr=saved.iloc[0]
                    assert sr.n==expected
                    row={field:sr[field] for field in ('second_stage','second_stage_se','second_stage_t','first_stage_F','n','n_counties')}
                    row['estimate_source']='saved'
                else:
                    y,x,z=p[cols].to_numpy(np.float64).T
                    fit=t28.fit2sls(y,x,z,w,fe,np.zeros(expected,np.int64),
                                   p.cluster.to_numpy(np.int64),singleton=False,want_F=True)
                    assert fit['n']==expected and fit['n_dropped']==0
                    row=dict(second_stage=fit['beta'],second_stage_se=fit['se'],
                             second_stage_t=fit['beta']/fit['se'],first_stage_F=fit['F'],
                             n=fit['n'],n_counties=p.ccg.nunique(),estimate_source='estimated')
                    new_fits+=1
                assert np.isfinite([row[k] for k in ('second_stage','second_stage_se','first_stage_F','second_stage_t')]).all()
                row.update(weight=weight,sample_id=sample_id,variant=variant,n_excluded=len(removed),
                    excluded_ccg=';'.join(map(str,removed)),second_stage_x1e5=row['second_stage']*1e5)
                rows.append(row);keys.add(key)
            pd.DataFrame(rows).to_csv(output,index=False)
        print(f'{weight}: finished full sample, 50 individual omissions, and 4 nested groups; {new_fits} new fits so far.',flush=True)
    estimates=pd.DataFrame(rows)
    assert len(estimates)==440 and len(keys)==440
    first_stages=estimates.pivot(index=['weight','sample_id'],columns='variant',values='first_stage_F')
    np.testing.assert_allclose(first_stages.untreated,first_stages.lhs_only,rtol=1e-10)
    np.testing.assert_allclose(first_stages.rhs_only,first_stages.joint_pooled_percentile_1_99,rtol=1e-10)
    # Verify each coefficient with independently computed weighted FE moments.
    max_error=0.
    for r in estimates.itertuples():
        removed=[] if r.sample_id=='full' else list(map(int,str(r.excluded_ccg).split(';')))
        p=a.loc[~a.ccg.isin(removed)]
        w=np.ones(len(p)) if r.weight=='equal county' else p.pop1975.to_numpy(np.float64)
        f=pd.factorize(p.fe_division_year)[0]
        W=np.bincount(f,weights=w)
        y,x,z=p[variants[r.variant]].to_numpy(np.float64).T
        yt=y-np.bincount(f,weights=w*y)[f]/W[f]
        xt=x-np.bincount(f,weights=w*x)[f]/W[f]
        zt=z-np.bincount(f,weights=w*z)[f]/W[f]
        b=(w*zt*yt).sum()/(w*zt*xt).sum()
        np.testing.assert_allclose(b,r.second_stage,rtol=1e-9,atol=1e-13)
        max_error=max(max_error,abs(b-r.second_stage))
    d=estimates.pivot(index=['weight','sample_id','n_excluded'],columns='variant',values='second_stage_x1e5').reset_index()
    d['gap']=d.joint_pooled_percentile_1_99-d.untreated
    d['lhs_gap']=.5*((d.lhs_only-d.untreated)+(d.joint_pooled_percentile_1_99-d.rhs_only))
    d['rhs_gap']=.5*((d.rhs_only-d.untreated)+(d.joint_pooled_percentile_1_99-d.lhs_only))
    np.testing.assert_allclose(d.lhs_gap+d.rhs_gap,d.gap,atol=1e-12)
    full=d.loc[d.sample_id.eq('full')].set_index('weight')
    for block in ('gap','lhs_gap','rhs_gap'):
        d[f'{block}_closed']=d.weight.map(full[block])-d[block]
    d['absolute_gap_closed_percent']=100*(1-d.gap.abs()/d.weight.map(full.gap.abs()))
    d['signed_gap_closed_percent']=100*d.gap_closed/d.weight.map(full.gap.abs())
    d['gap_crossed_zero']=np.sign(d.gap)!=np.sign(d.weight.map(full.gap))
    d['lhs_closure_pp']=100*d.lhs_gap_closed/d.weight.map(full.gap.abs())
    d['rhs_closure_pp']=100*d.rhs_gap_closed/d.weight.map(full.gap.abs())
    np.testing.assert_allclose(d.lhs_gap_closed+d.rhs_gap_closed,d.gap_closed,atol=1e-12)
    np.testing.assert_allclose(d.lhs_closure_pp+d.rhs_closure_pp,d.signed_gap_closed_percent,atol=1e-10)
    same_sign=d.loc[~d.gap_crossed_zero]
    np.testing.assert_allclose(same_sign.signed_gap_closed_percent,same_sign.absolute_gap_closed_percent,atol=1e-10)
    d.to_csv(OUTDIR/'gap50_decomposition.csv',index=False)
    county=d.loc[d.sample_id.str.startswith('county_')].copy()
    assert not county.gap_crossed_zero.any(), 'Individual classification requires same-sign gaps.'
    county['ccg']=county.sample_id.str.removeprefix('county_').astype(int)
    county=county.merge(ranks.rename(columns={'excluded_ccg':'ccg'})[['weight','ccg','county','state_name','rank','gap_reduction_percent']],
                        on=['weight','ccg'],validate='one_to_one')
    np.testing.assert_allclose(county.absolute_gap_closed_percent,county.gap_reduction_percent,atol=1e-8)
    county['dominant_block']=np.where(county.lhs_gap_closed>county.rhs_gap_closed,'LHS','RHS')
    county['pattern']=np.select([county.lhs_gap_closed.le(0),county.rhs_gap_closed.le(0)],
        ['RHS closes; LHS offsets','LHS closes; RHS offsets'],default='Both close')
    for v in ('x','y','z'):
        counts=a[v].ne(a[f'{v}_percentile_1_99']).groupby(a.ccg).sum()
        county[f'{v}_capped_years']=county.ccg.map(counts).astype(int)
    county=county.sort_values(['weight','rank'])
    county.to_csv(OUTDIR/'gap50_county_classification.csv',index=False)
    membership=pd.DataFrame(memberships)
    membership=membership.merge(county[['weight','ccg','dominant_block','pattern','lhs_closure_pp','rhs_closure_pp']],
                                on=['weight','ccg'],validate='many_to_one')
    membership.to_csv(OUTDIR/'gap50_membership.csv',index=False)
    print(d.loc[d.sample_id.str.startswith('top_'),['weight','n_excluded','untreated',
        'joint_pooled_percentile_1_99','absolute_gap_closed_percent','lhs_closure_pp','rhs_closure_pp']].to_string(index=False))
    print(membership.groupby(['weight','k','dominant_block']).size().unstack(fill_value=0).to_string())
    print(f'PASS: {new_fits} new fits; 440 saved/reused comparisons, no additional drops; weighted-moment max error {max_error:.12g}.',flush=True)


def gap50_report():
    """Report ranked county groups, full inference, and signed LHS/RHS closure."""
    if not CORRECTED:
        raise SystemExit('FAIL: --gap50-report requires --corrected')
    e=pd.read_csv(OUTDIR/'gap50_estimates.csv')
    d=pd.read_csv(OUTDIR/'gap50_decomposition.csv')
    c=pd.read_csv(OUTDIR/'gap50_county_classification.csv')
    members=pd.read_csv(OUTDIR/'gap50_membership.csv')
    assert len(e)==440 and len(c)==100 and c.ccg.nunique()==89
    assert not e.duplicated(['weight','sample_id','variant']).any()
    assert e.n.eq(107030-35*e.n_excluded).all()
    assert e.n_counties.eq(3058-e.n_excluded).all()
    np.testing.assert_allclose(e.second_stage_t,e.second_stage/e.second_stage_se,rtol=1e-12)
    np.testing.assert_allclose(c.lhs_closure_pp+c.rhs_closure_pp,c.absolute_gap_closed_percent,atol=1e-10)
    group=d.loc[d.sample_id.str.startswith('top_')].copy()
    for label in ('LHS','RHS'):
        c.loc[c.dominant_block.eq(label)].to_csv(OUTDIR/f'gap50_{label.lower()}_dominant_counties.csv',index=False)
    counts=members.groupby(['weight','k','dominant_block']).size().unstack(fill_value=0).reindex(columns=['LHS','RHS'],fill_value=0).reset_index()
    counts.to_csv(OUTDIR/'gap50_classification_counts.csv',index=False)
    patterns=c.groupby(['weight','pattern']).size().rename('Counties').reset_index()
    def table(f):
        f=f.copy()
        if 'county' in f:
            f['county']=f.county.str.title()
        return f.rename(columns={'weight':'Weight','n_excluded':'Removed','untreated':'Raw coefficient',
            'joint_pooled_percentile_1_99':'Winsorized coefficient','absolute_gap_closed_percent':'Gap closed (%)',
            'lhs_closure_pp':'LHS contribution (pp)','rhs_closure_pp':'RHS contribution (pp)',
            'gap_crossed_zero':'Gap reversed sign','rank':'Rank','county':'County','state_name':'State',
            'dominant_block':'Dominant block','pattern':'Pattern','first_stage_F':'First-stage F',
            'second_stage_t':'Second-stage t','second_stage_x1e5':'Coefficient x 100,000',
            'variant':'Specification','sample_id':'Sample'}).to_markdown(index=False,floatfmt='.6f')
    selected=d.loc[(d.weight.eq('equal county')&d.sample_id.eq('top_40'))|
                   (d.weight.eq('fixed 1975 population')&d.sample_id.eq('top_50'))].set_index('weight')
    infer=e.loc[e.sample_id.eq('full')|e.sample_id.str.startswith('top_')].copy()
    infer=infer.loc[infer.variant.isin(['untreated','joint_pooled_percentile_1_99'])]
    infer['variant']=infer.variant.map({'untreated':'Raw','joint_pooled_percentile_1_99':'Winsorized x, y, z'})
    infer=infer.sort_values(['weight','n_excluded','variant'])
    countycols=['rank','county','state_name','absolute_gap_closed_percent','lhs_closure_pp','rhs_closure_pp','dominant_block','pattern']
    blocks=[]
    for weight in ('equal county','fixed 1975 population'):
        blocks.append(f'### {weight}\n\n'+table(c.loc[c.weight.eq(weight),countycols]))
    body=f'''# Counties that close the winsorization gap: LHS versus RHS

## Conclusion

Removing the top 40 equal-weight counties closes
{selected.loc['equal county','absolute_gap_closed_percent']:.6f}% of the gap.
Removing the top 50 fixed-population-weight counties closes
{selected.loc['fixed 1975 population','absolute_gap_closed_percent']:.6f}% (Table 1).
These are separately ranked sets, not the same counties under different weights.

All 50 counties in each weighting scheme are RHS-dominant in the two-block
decomposition (Table 3). The lists contain {c.ccg.nunique()} distinct county groups.
The LHS-dominant list is empty, but some counties have positive contributions from
both blocks (Table 4). RHS-dominant does not mean that employment plays no role.

The coefficients converge toward zero, not toward a strong common estimate.
With 40 counties removed under equal weights, the raw and winsorized coefficients
are {selected.loc['equal county','untreated']:.6f} and
{selected.loc['equal county','joint_pooled_percentile_1_99']:.6f}, respectively.
With 50 removed under population weights, they are
{selected.loc['fixed 1975 population','untreated']:.6f} and
{selected.loc['fixed 1975 population','joint_pooled_percentile_1_99']:.6f}.
All coefficients here are multiplied by 100,000; inference is in Table 2.
These outcome-selected sets are diagnostic lists, not proposed baseline exclusions.

## Table 1. Actual joint removal of the ranked county sets

Rank counties by the reduction in the absolute raw-versus-winsorized gap when each
county is removed individually from the full sample. Then remove the top 20, 30,
40, or 50 jointly. The ranking is fixed, not recomputed after each deletion.
All 35 annual observations of each selected county group are removed. The sample
remains balanced over 1982 through 2016.

{table(group[['weight','n_excluded','untreated','joint_pooled_percentile_1_99','absolute_gap_closed_percent',
              'lhs_closure_pp','rhs_closure_pp','gap_crossed_zero']])}

The 50-county equal-weight group overshoots: the winsorized coefficient becomes
smaller than the raw coefficient. Its signed LHS and RHS contributions therefore
sum to the signed movement across zero, not to the percentage reduction in the
absolute gap. The 40-county set is closer to agreement than the 50-county set.
Source: [group decomposition](../../Results/instrument_normalization/corrected_gap_audit/gap50_decomposition.csv).

## Table 2. First-stage F and second-stage t statistics

Division-by-year fixed effects are refitted for every sample. Standard errors are
clustered by county. Population weights remain fixed in 1975. Pooled 1st/99th
percentile cutoffs, national shifters, and retained county instrument values remain
fixed at their full-sample values. This is sample deletion, not a rebuilt instrument.

{table(infer[['weight','n_excluded','variant','second_stage_x1e5','first_stage_F','second_stage_t','n','n_counties']])}

Source: [all estimates, including hybrid specifications](../../Results/instrument_normalization/corrected_gap_audit/gap50_estimates.csv).
There are {int(e.estimate_source.eq('estimated').sum())} estimated rows and
{int(e.estimate_source.eq('saved').sum())} reused rows. No extra observations are
dropped by estimation beyond the stated whole-county removals.

## Definition of the LHS/RHS split

For a given sample S, let beta00 be the raw coefficient, beta10 the coefficient
when only employment growth y is capped, beta01 when only spending x and the
instrument z are capped, and beta11 when all three are capped. The four fits use
the same observations, fixed effects, weights, and clustering within S.

The gap is G(S) = beta11 - beta00. Its two components are:

```text
LHS(S) = 0.5 * [(beta10 - beta00) + (beta11 - beta01)]
RHS(S) = 0.5 * [(beta01 - beta00) + (beta11 - beta10)]
G(S)   = LHS(S) + RHS(S)
```

This averages the employment-first and RHS-first paths, dividing their interaction
equally. For a removed county c, its LHS closure is LHS(full) - LHS(without c);
its RHS closure is defined the same way. The two sum exactly to its signed gap
closure. A county is RHS-dominant when its RHS closure exceeds its LHS closure;
otherwise it is LHS-dominant. No magnitude threshold determines the label.

Contributions in percentage points divide these closures by the original positive
full-sample gap. Positive means closing the gap; negative means offsetting closure.
Both sides refer to the change induced by capping, not to the accuracy of the data
or the economic source of the county's employment response.

The classification is not determined by which of the county's own values are
capped. Uncapped observations can affect the two fitted coefficients differently
because capping elsewhere changes the fitted relationship and fixed-effect means.
The saved county file also reports each county's number of capped x, y, and z years.

## Table 3. Dominant block within each nested list

{table(counts)}

Source: [classification counts](../../Results/instrument_normalization/corrected_gap_audit/gap50_classification_counts.csv).

## Table 4. Whether both blocks contribute positively, top 50 per weighting scheme

{table(patterns)}

For jointly removed groups, compute the decomposition on the jointly reduced
sample, as in Table 1. Do not sum individual county closures to obtain the joint
closure: refitting makes the deletion effects nonadditive.

## Table 5. Complete ranked county lists and individual contributions

Each row refers to removing that county alone. Rank identifies membership of each
nested group: ranks 1 through 20, 30, 40, or 50. Contributions are percentage points
of the original full-sample gap, not percentages of the row's own closure.

{chr(10).join(blocks)}

Source: [complete county classification](../../Results/instrument_normalization/corrected_gap_audit/gap50_county_classification.csv).
Separate files: [RHS-dominant counties](../../Results/instrument_normalization/corrected_gap_audit/gap50_rhs_dominant_counties.csv)
and [LHS-dominant counties, empty with headers](../../Results/instrument_normalization/corrected_gap_audit/gap50_lhs_dominant_counties.csv).
The [membership file](../../Results/instrument_normalization/corrected_gap_audit/gap50_membership.csv)
records the exact counties in each joint removal.

## Verification and scope

All 440 fitted or reused coefficients are checked against weighted, fixed-effect
residualized IV moments. The LHS and RHS components sum to each gap and their
removal contributions sum to each signed gap closure. Saved individual closures
match the full county-omission ranking. All estimates preserve the requested
county counts and balanced time series. These are numerical and sample-consistency
checks, not an independent review of the identification strategy.

No geography, employment, contract, instrument, or baseline sample is modified.
Closing the gap does not establish that these counties are miscoded. Source audits
would have to justify any data correction separately.

## Reproduction

Producer: `step_1/code/pipeline/102_h2_winsorization_cell_audit.py`.
Run `--corrected --gap50`, then `--corrected --gap50-report`.
The first mode reuses saved endpoint estimates and saved optional fits; only missing
comparisons are estimated. Inputs are the certified corrected analysis panel and
the saved all-county ranking, individual omissions, and top-20 group estimates.
'''
    path=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/COUNTY_GAP_LHS_RHS_SPLIT.md'
    assert '\u2014' not in body
    path.write_text(body)
    assert path.read_text()==body
    print(f'PASS: classification tables and complete county lists saved to {path}')


def all_county_report():
    """Verify omission coefficients by weighted moments and document the gaps."""
    if not CORRECTED:
        raise SystemExit('FAIL: --all-counties-report requires --corrected')
    estimates=pd.read_csv(OUTDIR/'all_county_leaveout_estimates.csv')
    ranks=pd.read_csv(OUTDIR/'all_county_leaveout_ranking.csv')
    groups=pd.read_csv(OUTDIR/'top_county_group_estimates.csv')
    joint=pd.read_csv(OUTDIR/'top_county_group_gaps.csv')
    members=pd.read_csv(OUTDIR/'top_county_group_membership.csv')
    ref=pd.read_csv(REFERENCE)
    ref=ref.loc[ref.source_version.eq('corrected')]
    assert len(estimates)==12232 and len(ranks)==6116 and len(groups)==12 and len(joint)==6
    assert estimates.n.eq(106995).all() and estimates.n_counties.eq(3057).all()
    assert groups.n.eq(107030-35*groups.n_excluded).all()
    assert groups.n_counties.eq(3058-groups.n_excluded).all()
    assert not estimates.duplicated(['excluded_ccg','weight','variant']).any()
    for f in (estimates,groups):
        assert np.isfinite(f[['first_stage_F','second_stage_t','second_stage','second_stage_se']]).all().all()
        np.testing.assert_allclose(f.second_stage_t,f.second_stage/f.second_stage_se,rtol=1e-12)
    a=pd.read_parquet(CACHE/'analysis_pooled.parquet')
    fi=pd.factorize(a.fe_division_year)[0]
    counties=a.ccg.to_numpy(np.int64)
    unique_ids=np.sort(np.unique(counties))
    locations={int(c):np.flatnonzero(counties==c) for c in unique_ids}
    variants={'untreated':['y','x','z'],
              'joint_pooled_percentile_1_99':[f'{v}_percentile_1_99' for v in ('y','x','z')]}
    max_error=0.
    for weight in ('equal county','fixed 1975 population'):
        w=np.ones(len(a)) if weight=='equal county' else a.pop1975.to_numpy(np.float64)
        W=np.bincount(fi,weights=w)
        for variant,cs in variants.items():
            y,x,z=a[cs].to_numpy(np.float64).T
            Y=np.bincount(fi,weights=w*y)
            X=np.bincount(fi,weights=w*x)
            Z=np.bincount(fi,weights=w*z)
            raw_n=(w*z*y).sum()
            raw_d=(w*z*x).sum()
            centered_n=raw_n-(Z*Y/W).sum()
            centered_d=raw_d-(Z*X/W).sum()
            full=ref.loc[ref.weight.eq(weight)&ref.variant.eq(variant)].iloc[0]
            np.testing.assert_allclose(centered_n/centered_d,full.second_stage,rtol=1e-9,atol=1e-14)
            chosen=estimates.loc[estimates.weight.eq(weight)&estimates.variant.eq(variant)].set_index('excluded_ccg')
            for ccg,ix in locations.items():
                f=fi[ix]
                assert len(np.unique(f))==len(ix)
                wr=w[ix]
                assert (W[f]>wr).all()
                n=centered_n-(wr*z[ix]*y[ix]).sum()+np.sum(Z[f]*Y[f]/W[f]-(Z[f]-wr*z[ix])*(Y[f]-wr*y[ix])/(W[f]-wr))
                d=centered_d-(wr*z[ix]*x[ix]).sum()+np.sum(Z[f]*X[f]/W[f]-(Z[f]-wr*z[ix])*(X[f]-wr*x[ix])/(W[f]-wr))
                computed=n/d
                actual=chosen.loc[ccg,'second_stage']
                np.testing.assert_allclose(computed,actual,rtol=1e-9,atol=1e-14)
                max_error=max(max_error,abs(computed-actual))
            for r in groups.loc[groups.weight.eq(weight)&groups.variant.eq(variant)].itertuples():
                ids=np.array([int(c) for c in r.excluded_ccg.split(';')])
                assert len(ids)==r.n_excluded and len(np.unique(ids))==len(ids)
                selected=ranks.loc[ranks.weight.eq(weight)&ranks['rank'].le(r.n_excluded)]
                assert set(ids)==set(selected.excluded_ccg)
                logged=members.loc[members.weight.eq(weight)&members.n_excluded.eq(r.n_excluded),'ccg']
                assert set(ids)==set(logged)
                keep=~np.isin(counties,ids)
                f=fi[keep]; wk=w[keep]; zk=z[keep]; yk=y[keep]; xk=x[keep]
                denom=np.bincount(f,weights=wk)
                assert (denom>0).all()
                sz=np.bincount(f,weights=wk*zk)
                n=(wk*zk*yk).sum()-(sz*np.bincount(f,weights=wk*yk)/denom).sum()
                d=(wk*zk*xk).sum()-(sz*np.bincount(f,weights=wk*xk)/denom).sum()
                np.testing.assert_allclose(n/d,r.second_stage,rtol=1e-9,atol=1e-14)
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(map(str,row))+' |' for row in rows])
    def pretty(county):
        return county.title() if county.isupper() else county
    summary=[]
    detail=[]
    for weight in ('equal county','fixed 1975 population'):
        full=ref.loc[ref.weight.eq(weight)].set_index('variant')
        raw=full.loc['untreated']; cap=full.loc['joint_pooled_percentile_1_99']
        summary.append([weight,0,f'{raw.second_stage_x1e5:.6f}',f'{cap.second_stage_x1e5:.6f}',
                        f'{cap.second_stage_x1e5-raw.second_stage_x1e5:.6f}','0.0000%'])
        for k in (0,5,10,20):
            s=ref.loc[ref.weight.eq(weight)] if k==0 else groups.loc[groups.weight.eq(weight)&groups.n_excluded.eq(k)]
            for r in s.itertuples():
                detail.append([weight,k,'Raw' if r.variant=='untreated' else 'Winsorized',
                               f'{r.second_stage_x1e5:.6f}',f'{r.first_stage_F:.6f}',f'{r.second_stage_t:.6f}',r.n_counties])
            if k:
                r=joint.loc[joint.weight.eq(weight)&joint.n_excluded.eq(k)].iloc[0]
                summary.append([weight,k,f'{r.untreated:.6f}',f'{r.joint_pooled_percentile_1_99:.6f}',
                                f'{r.gap_x1e5:.6f}',f'{r.gap_reduction_percent:.4f}%'])
    leaders=table(['Weight','Rank','County or county group','Gap without county','Gap closed'],
        [[r.weight,r.rank,f'{pretty(r.county)}, {r.state_name}',f'{r.gap_x1e5:.6f}',f'{r.gap_reduction_percent:.4f}%']
         for r in ranks.loc[ranks['rank'].le(20)].itertuples()])
    twenty=joint.loc[joint.n_excluded.eq(20)].set_index('weight')
    largest=ranks.loc[ranks['rank'].eq(1)].set_index('weight')
    body=f'''# County omissions and the winsorization gap

## Question and design

Does removing a small set of whole county histories bring the raw and winsorized
coefficients closer together? This is a diagnostic of the gap, not a proposed
baseline sample restriction or a data-error classification.

Inputs are the certified report-corrected h=2 county cache. Every omission removes
all 35 years for the specified county or fixed county group. The full-sample pooled
1st and 99th percentile cutoffs for x, y, and z stay fixed, as do the instrument
values of retained observations. National shocks and historical exposures are not
recomputed. Division-by-year fixed effects are re-estimated on each retained
sample. Inference clusters by county. Weights are equal county and fixed 1975
population, separately. No employment weights are used.

Every county is omitted individually. For each weight, the ranking is descending
in the reduction of the absolute gap between winsorized and raw coefficients.
Ties use the numeric county identifier. The top 5, 10, and 20 of that fixed
individual-omission ranking are then omitted together. Rankings are not recomputed
after each removal. The two weighting schemes use their own ranked sets.

## Table 1. How much of the gap does each group close?

Coefficients and gaps are multiplied by 100,000. Gap closed is
100 times one minus the absolute remaining gap divided by the absolute
full-sample gap. Joint results are actual re-estimates, not sums of individual
contributions.

{table(['Weight','Counties excluded','Raw coefficient','Winsorized coefficient','Gap','Gap closed'],summary)}

Source: [`top_county_group_gaps.csv`](../../Results/instrument_normalization/corrected_gap_audit/top_county_group_gaps.csv).
The full-sample rows are saved corrected estimates in
[`h2_report_correction_comparison.csv`](../../Results/instrument_normalization/h2_report_correction_comparison.csv).

Removing the top 20 closes {twenty.loc['equal county','gap_reduction_percent']:.4f}%
of the equal-weight gap and {twenty.loc['fixed 1975 population','gap_reduction_percent']:.4f}%
of the population-weighted gap (Table 1). The maximum individual-omission reductions
are {largest.loc['equal county','gap_reduction_percent']:.4f}% and
{largest.loc['fixed 1975 population','gap_reduction_percent']:.4f}%, respectively
(Table 3). The gap is therefore not explained by one county, but the ranked groups
account for a majority of its absolute size in these deletion tests.

For both weighting schemes, removing the top 20 lowers the winsorized coefficient
and also lowers the raw coefficient relative to the full sample (Table 1).
Convergence comes from the larger decline in the winsorized estimate, not from
recovering a larger raw estimate. This identifies cases worth examining; it does
not establish that their data are wrong, or that their omission belongs in the
baseline. A substantial gap remains in both top-20 samples (Table 1).

## Table 2. First-stage F and second-stage t

{table(['Weight','Counties excluded','Specification','Coefficient x 100,000','First-stage F','Second-stage t','Counties retained'],detail)}

Source: [`top_county_group_estimates.csv`](../../Results/instrument_normalization/corrected_gap_audit/top_county_group_estimates.csv)
and the saved full-sample comparison. These are conventional county-clustered
regression statistics for each diagnostic sample, not inference adjusted for
selecting counties to close the gap.

## Table 3. The individual-omission ranking

{leaders}

Source: [`all_county_leaveout_ranking.csv`](../../Results/instrument_normalization/corrected_gap_audit/all_county_leaveout_ranking.csv).
Complete coefficients, standard errors, F-statistics, t-statistics, and sample
sizes for every individual omission are in
[`all_county_leaveout_estimates.csv`](../../Results/instrument_normalization/corrected_gap_audit/all_county_leaveout_estimates.csv).
Exact joint-group membership is in
[`top_county_group_membership.csv`](../../Results/instrument_normalization/corrected_gap_audit/top_county_group_membership.csv).
Labels such as Fairfax +2 preserve the project's fixed county grouping.

## Verification and reproduction

All {len(ranks)//2:,} county omissions are covered under both weighting schemes.
There are {len(estimates):,} individual-omission fits, of which four Austin fits
are read from saved results, and {len(groups)} joint-group fits. No observations
are dropped beyond the requested whole county histories. Every individual
coefficient also matches a separate weighted group-sum calculation that explicitly
updates the fixed-effect means after deletion. Maximum absolute coefficient
discrepancy is {max_error:.12g}. Joint coefficients and exact membership are also
checked against retained-sample weighted moments.

Producer: `step_1/code/pipeline/102_h2_winsorization_cell_audit.py`.
Use `--corrected --all-counties` for estimation and
`--corrected --all-counties-report` for verification and this report. The report
mode does not re-estimate regressions. No source-data correction or production
sample restriction is implemented by either mode.
'''
    doc=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/COUNTY_LEAVEOUT_GAP_AUDIT.md'
    doc.write_text(body)
    assert doc.read_text()==body
    print(f'PASS: all omission coefficients checked, maximum absolute error {max_error:.12g}; report saved to {doc}')


def leading_county_cases():
    """Allocate each exact county-deletion gap change to that county's years."""
    if not CORRECTED:
        raise SystemExit('FAIL: leading cases require --corrected')
    ranking=pd.read_csv(OUTDIR/'all_county_leaveout_ranking.csv')
    leaders=ranking.loc[ranking['rank'].le(5)]
    saved=pd.read_csv(OUTDIR/'all_county_leaveout_estimates.csv')
    ref=pd.read_csv(REFERENCE)
    ref=ref.loc[ref.source_version.eq('corrected')]
    a=pd.read_parquet(CACHE/'analysis_pooled.parquet')
    endpoints=pd.read_parquet(CACHE/'winsorization_cell_attribution.parquet')
    endpoints=endpoints.loc[endpoints.weight.eq('equal county')].set_index(['ccg','year'])
    fi=pd.factorize(a.fe_division_year)[0]
    results=[]
    for r in leaders.itertuples():
        ix=np.flatnonzero(a.ccg.to_numpy()==r.excluded_ccg)
        w=np.ones(len(a)) if r.weight=='equal county' else a.pop1975.to_numpy(np.float64)
        W=np.bincount(fi,weights=w)
        f=fi[ix]; wr=w[ix]
        g=endpoints.loc[r.excluded_ccg].reindex(a.year.iloc[ix]).reset_index()
        g['ccg']=r.excluded_ccg
        g['ranking_weight']=r.weight
        g['county_gap_closed_x1e5']=r.gap_closed_x1e5
        for label,variant,cs in [('raw','untreated',['y','x','z']),
                               ('capped','joint_pooled_percentile_1_99',[f'{v}_percentile_1_99' for v in ('y','x','z')])]:
            y,x,z=a[cs].to_numpy(np.float64).T
            Y=np.bincount(fi,weights=w*y); X=np.bincount(fi,weights=w*x); Z=np.bincount(fi,weights=w*z)
            ztilde=z-Z[fi]/W[fi]
            denominator=(w*ztilde*x).sum()
            without=saved.loc[saved.excluded_ccg.eq(r.excluded_ccg)&saved.weight.eq(r.weight)&saved.variant.eq(variant)].iloc[0].second_stage
            full=ref.loc[ref.weight.eq(r.weight)&ref.variant.eq(variant)].iloc[0].second_stage
            minus_fe=(Y[f]-wr*y[ix]-without*(X[f]-wr*x[ix]))/(W[f]-wr)
            deleted_residual=y[ix]-without*x[ix]-minus_fe
            contribution=wr*ztilde[ix]*deleted_residual/denominator*1e5
            np.testing.assert_allclose(contribution.sum(),(full-without)*1e5,atol=1e-10)
            g[f'{label}_county_deletion_contribution_x1e5']=contribution
        g['gap_closure_contribution_x1e5']=g.capped_county_deletion_contribution_x1e5-g.raw_county_deletion_contribution_x1e5
        np.testing.assert_allclose(g.gap_closure_contribution_x1e5.sum(),r.gap_closed_x1e5,atol=1e-10)
        g['rank_within_county']=g.gap_closure_contribution_x1e5.rank(method='first',ascending=False).astype(int)
        results.append(g)
    cells=pd.concat(results,ignore_index=True)
    assert len(cells)==350
    cells.to_csv(OUTDIR/'leading_county_year_gap_contributions.csv',index=False)
    selected=cells.loc[cells.rank_within_county.le(3)].copy()
    selected.to_csv(OUTDIR/'leading_county_selected_years.csv',index=False)
    print(selected[['ranking_weight','county','state_name','year','x','y','z','x_percentile_1_99','y_percentile_1_99','z_percentile_1_99',
                    'gap_closure_contribution_x1e5','county_gap_closed_x1e5']].to_string(index=False))
    print('PASS: yearly signed contributions reproduce all ten actual county-deletion gap changes. They are not year-deletion estimates.')


def leading_rhs_audit():
    """Trace leading RHS components to current category activity and national origins."""
    if not CORRECTED:
        raise SystemExit('FAIL: RHS audit requires --corrected')
    components=pd.read_csv(OUTDIR/'h2_winsorization_instrument_components_leaders.csv',dtype={'cat':str})
    cases=pd.read_csv(OUTDIR/'leading_county_selected_years.csv')
    leading=components.loc[components.rank_absolute_z.le(3)].copy()
    targets={int(y):sorted(leading.loc[leading.year.between(y,y+2),'cat'].unique())
             for y in sorted({int(t)+j for t in leading.year for j in (-2,-1,0)})}
    annual={}
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        for year,cats in targets.items():
            annual[year]=con.execute('''SELECT ccg,cat,sum(dollars) dollars
                FROM read_parquet(?) WHERE cat IN (SELECT unnest(?)) GROUP BY ccg,cat''',
                [str(CACHE/'annual_category'/f'fy={year}.parquet'),cats]).df()
    origins=[]
    for r in leading.itertuples():
        vals=[]
        for j in (-2,-1,0):
            a=annual[r.year+j]
            vals.append(a.loc[a.cat.eq(r.cat)].set_index('ccg').dollars.rename(str(j)))
        panel=pd.concat(vals,axis=1).fillna(0)
        panel['national_shift_contribution']=.5*(panel['-1']+panel['0'])-panel['-2']
        np.testing.assert_allclose(panel.national_shift_contribution.sum(),r.national_shift,rtol=1e-10,atol=1e-5)
        panel['rank_absolute_shift']=panel.national_shift_contribution.abs().rank(method='first',ascending=False).astype(int)
        panel['origin_is_focal_county']=panel.index==r.ccg
        keep=panel.loc[panel.rank_absolute_shift.le(5)|panel.origin_is_focal_county].reset_index().rename(columns={'ccg':'origin_ccg'})
        keep['focal_ccg'],keep['year'],keep['cat']=r.ccg,r.year,r.cat
        keep['national_shift']=r.national_shift
        origins.append(keep)
    origins=pd.concat(origins,ignore_index=True)
    names=pd.read_csv(dc.CCG_CROSSWALK)
    names['ccg']=pd.to_numeric(names.ccg_id,errors='coerce')
    labels=names.dropna(subset=['ccg']).drop_duplicates('ccg').set_index('ccg').ccg_name
    origins['origin_county']=origins.origin_ccg.map(labels).fillna('Unmapped county group')
    origins['origin_state']=(origins.origin_ccg//1000).map(STATE_NAMES)
    leading['current_county_spending_average']=(leading['county_-1']+leading['county_0'])/2
    leading['current_national_spending_average']=(leading['national_-1']+leading['national_0'])/2
    leading['current_market_share']=leading.current_county_spending_average/leading.current_national_spending_average.replace(0,np.nan)
    leading['own_county_national_shift']=(leading['county_-1']+leading['county_0'])/2-leading['county_-2']
    leading=leading.merge(cases[['ccg','year','population_before']],on=['ccg','year'],validate='many_to_one')
    leading['own_county_z_component']=leading.market_share*leading.own_county_national_shift/leading.population_before
    leading['other_counties_z_component']=leading.z_contribution-leading.own_county_z_component
    leading['local_zero_all_three_years']=leading[['county_-2','county_-1','county_0']].eq(0).all(axis=1)
    contract_history=[]
    with duckdb.connect() as con:
        for year in range(1988,2000):
            f=con.execute('''SELECT contract_no contract,fsc category,fsc_description description,
                sum(CASE WHEN dollars>0 THEN dollars*CASE WHEN dollars_unit='thousands' THEN 1000 ELSE 1 END ELSE 0 END) positive_nominal,
                sum(dollars*CASE WHEN dollars_unit='thousands' THEN 1000 ELSE 1 END) net_nominal,
                count(*) source_rows FROM read_parquet(?) WHERE county_fips5='49003'
                AND contract_no IN ('F0470488C0015','F4261094C0031') GROUP BY ALL''',
                [str(dc.OUT_DIR/f'fy{year}.parquet')]).df()
            f['year']=year
            contract_history.append(f)
        mineral=[]
        for year in range(2007,2010):
            f=con.execute('''SELECT parent_award_id_piid parent_contract,award_id_piid order_id,
                product_or_service_code category,transaction_description description,
                contract_transaction_unique_key transaction_key,federal_action_obligation nominal_obligation
                FROM read_parquet(?) WHERE ccg=32021 AND award_or_idv_flag='AWARD'
                AND parent_award_id_piid='DAAA0999D0022' ''',
                [str(dc.USASPENDING_OUT_DIR/f'fy{year}.parquet')]).df()
            f['year']=year
            mineral.append(f)
    history=pd.concat(contract_history,ignore_index=True)
    history.to_csv(OUTDIR/'leading_rhs_box_elder_contract_history.csv',index=False)
    mineral=pd.concat(mineral,ignore_index=True)
    if mineral.transaction_key.isna().any() or mineral.transaction_key.duplicated().any():
        raise SystemExit('FAIL: Mineral parent contract has missing or duplicated transaction keys')
    mineral.to_csv(OUTDIR/'leading_rhs_mineral_parent_contract_actions.csv',index=False)
    origins.to_csv(OUTDIR/'leading_rhs_national_origins.csv',index=False)
    leading.to_csv(OUTDIR/'leading_rhs_category_comparison.csv',index=False)
    print(leading.loc[leading.ccg.isin([49003,32021,48273])&leading.rank_absolute_z.eq(1),
        ['county','year','cat','market_share','current_market_share','local_zero_all_three_years',
         'z_contribution','own_county_z_component','other_counties_z_component','x_total']].to_string(index=False))
    print('PASS: national origins sum to each selected category shift. No fitted coefficient or instrument is changed.')


def leading_rhs_report():
    """Write RHS evidence without treating exposure changes as proven errors."""
    if not CORRECTED:
        raise SystemExit('FAIL: RHS report requires --corrected')
    c=pd.read_csv(OUTDIR/'leading_rhs_category_comparison.csv',dtype={'cat':str})
    o=pd.read_csv(OUTDIR/'leading_rhs_national_origins.csv',dtype={'cat':str})
    h=pd.read_csv(OUTDIR/'leading_rhs_box_elder_contract_history.csv',dtype={'category':str})
    m=pd.read_csv(OUTDIR/'leading_rhs_mineral_parent_contract_actions.csv',dtype={'category':str})
    s=pd.read_csv(OUTDIR/'h2_winsorization_attribution_summary.csv')
    a=pd.read_csv(OUTDIR/'h2_winsorization_source_audit_annual_leaders.csv')
    assert len(c)==90 and c[['ccg','year']].drop_duplicates().shape[0]==30
    np.testing.assert_allclose(c.own_county_z_component+c.other_counties_z_component,c.z_contribution,atol=1e-10)
    assert not m.transaction_key.duplicated().any()
    def table(f):
        return f.to_markdown(index=False,floatfmt='.6f')
    top=c.loc[c.rank_absolute_z.eq(1)&c.ccg.isin([49003,32021,48273])].copy()
    top['Exposure window']=top.exposure_start.astype(str)+' to '+top.exposure_end.astype(str)
    top['Historical share (%)']=100*top.market_share
    top['Current share (%)']=100*top.current_market_share
    top=top.rename(columns={'county':'County','year':'Year','cat':'Category','z_contribution':'Category contribution to z',
        'own_county_z_component':'From own county','other_counties_z_component':'From other counties'})
    table2=top[['County','Year','Category','Exposure window','Historical share (%)','Current share (%)',
                'Category contribution to z','From own county','From other counties']]
    bx98=a.loc[a.ccg.eq(49003)&a.year.eq(1998)].iloc[0]
    msum=m.assign(positive=m.nominal_obligation.clip(lower=0)).groupby(['year','category'],as_index=False).agg(
        positive=('positive','sum'),net=('nominal_obligation','sum'))
    msum=msum.loc[msum.category.isin(['M151','1398','S216'])]
    selected=h.loc[(h.contract.eq('F0470488C0015')&h.year.isin([1989,1995,1996,1998]))|
                   (h.contract.eq('F4261094C0031')&h.year.isin([1994,1995,1996,1998]))]
    summary=s[['weight','beta_raw_x1e5','beta_cap_x_x1e5','beta_cap_xy_x1e5','beta_cap_xyz_x1e5']].rename(
        columns={'weight':'Weight','beta_raw_x1e5':'Raw','beta_cap_x_x1e5':'Cap x',
                 'beta_cap_xy_x1e5':'Cap x and y','beta_cap_xyz_x1e5':'Cap x, y and z'})
    origin=o.loc[((o.focal_ccg.eq(49003)&o.year.eq(1998)&o.cat.eq('1427'))|
                  (o.focal_ccg.eq(32021)&o.year.eq(2009)&o.cat.eq('M151')))&o.rank_absolute_shift.le(2)].copy()
    origin['Origin']=origin.origin_county.str.title()+', '+origin.origin_state
    origin['Contribution ($m)']=origin.national_shift_contribution/1e6
    origin['National shift ($m)']=origin.national_shift/1e6
    origin['Focal county']=origin.focal_ccg.map({49003:'Box Elder, Utah',32021:'Mineral, Nevada'})
    body=f'''# RHS investigation of the winsorization gap

## Main finding

The saved sequential decomposition points to spending and the instrument rather
than employment capping (Table 1). Source tracing identifies a separate issue:
historical exposure to a detailed category can differ sharply from the county's
recorded activity in that category during the shock window (Table 2).

This is evidence about the RHS construction, not a demonstrated data error and not
a quantified explanation of how much of the coefficient gap category coding causes.
Employment, procurement data, exposure windows, instruments, weights, and the sample
are unchanged. No regression fits are run by these diagnostic modes.

## Table 1. Saved sequential coefficient accounting

Coefficients are multiplied by 100,000. `x` is the observed procurement change per
resident; `y` is midpoint employment growth; `z` is the instrument. All caps use the
saved pooled 1st and 99th percentiles. The sequence is raw, cap x, additionally cap y,
then additionally cap z. These are order-dependent accounting comparisons, not an
order-invariant allocation of the effect of winsorization.

{table(summary)}

Source: [saved decomposition](../../Results/instrument_normalization/corrected_gap_audit/h2_winsorization_attribution_summary.csv).
These coefficient rows are reused, not newly estimated. This mode does not produce
new first-stage F statistics or second-stage t statistics.

## Table 2. Historical category exposure versus activity during the shock

The historical market share uses t-11 through t-7. The current share is the county's
combined spending in t-1 and t divided by combined national spending in those same
years. It is descriptive only and never substituted into the instrument.
The last two columns allocate the existing category contribution to the national
spending changes originating inside and outside the focal county. They are not
leave-one-out regression estimates. Monetary components are real dollars per resident.

{table(table2)}

Source: [90 leading category components across 30 windows](../../Results/instrument_normalization/corrected_gap_audit/leading_rhs_category_comparison.csv).

### Box Elder: the historical category is absent during the later shock window

For 1998 and 1999, the leading category is 1427, guided missile subsystems. County
spending in that code is zero in each of t-2, t-1, and t, despite its historical
exposure (Table 2). Its entire category contribution therefore comes from national
spending changes outside Box Elder. This is not a case in which the focal county's
own spending mechanically creates that component of its instrument.

Exact contract identifiers also appear under different category codes at different
dates (Table 3). `F0470488C0015` appears under 1427 on its positive actions and under
L014, technical representative services for guided missiles, on later negative
actions. `F4261094C0031` appears under AC23 in 1994 and 1995 and L014 in 1996 and
1998. A contract identifier can span different work, so these observations do not
prove a misclassification or justify collapsing those codes. They show why category
continuity needs a contract-level check rather than an employment-data correction.

## Table 3. Selected Box Elder contract-category observations

Nominal dollars. Positive obligations and net obligations are shown separately.

{table(selected[['contract','year','category','description','positive_nominal','net_nominal']])}

Source: [contract history, 1988 through 1999](../../Results/instrument_normalization/corrected_gap_audit/leading_rhs_box_elder_contract_history.csv).

There is also an obligations-accounting question. Box Elder's 1998 county total is
${bx98.positive_nominal:,.2f} in positive obligations and ${bx98.net_nominal:,.2f}
net of negative actions. The baseline uses the former. On `F4261094C0031`, Table 3
shows positive and net amounts with opposite signs. The negative modification
therefore matters for interpreting measured procurement, but it has not been
established as a report correction rather than a genuine deobligation. No change
to the positive-only spending rule is made here.

### Mineral: depot work appears across several category codes

For 2009, the leading instrument category is M151, operation of ammunition
facilities. Table 2 compares its historical market share with its current share.
Most of this negative instrument component originates elsewhere; the own-county
component has the opposite sign.

The USAspending parent contract `DAAA0999D0022` contains {len(m)} distinct
transaction keys in Mineral during 2007 through 2009. Its positive obligations
appear chiefly in codes other than M151 among the three shown in Table 4.
For example, order 0121 has 1398-coded descriptions mentioning ammunition storage;
order 0204 has an S216-coded description for storage of field-service ammunition.
This supports examining whether comparable depot activities are recorded under
different categories. It does not establish that their tasks or scope are identical.

## Table 4. Selected categories within Mineral's parent contract

Nominal dollars. This is a subset of categories, not the entire county total.
Full parent-contract records are saved, including zero and negative actions.

{table(msum)}

Source: [all parent-contract actions](../../Results/instrument_normalization/corrected_gap_audit/leading_rhs_mineral_parent_contract_actions.csv).

### Kleberg: changes in fuel-category exposure and an unresolved location

The historical share of category 9140 for the 2002 instrument differs sharply from
the current share (Table 2). National spending in that category increases while
Kleberg's category spending decreases: the own-county and other-county instrument
components have opposite signs. Thus its positive predicted component is not
simply its observed local fuel expansion embedded in the national total.
The contract-specific geography question remains open as documented in
[the county investigation, Table 4](LEADING_COUNTY_GAP_INVESTIGATION.md).

## Table 5. Leading geographic origins of two national category shifts

This identifies where to inspect the underlying national spending change. Contributions
are signed and can offset; the displayed top two need not sum to the national total.

{table(origin[['Focal county','year','cat','Origin','Contribution ($m)','National shift ($m)']])}

Source: [national origins](../../Results/instrument_normalization/corrected_gap_audit/leading_rhs_national_origins.csv).

## Interpretation and remaining evidence

1. The RHS has concrete issues to investigate even if employment is measured correctly:
   differences between obligations and production timing, positive versus net actions,
   and changes in detailed category activity or classification.
2. A historical share need not equal a current share. That difference is not itself
   a failure of shift-share design. The unresolved question is whether the difference
   reflects economic specialization changing or inconsistent category measurement.
3. It is not established that harmonizing categories or changing obligation accounting
   would close the raw-versus-winsorized coefficient gap. That requires a defensible
   source-based correction and a matched comparison of both estimates.
4. Box Elder's named contracts and Mineral's parent contract are the direct RHS leads.
   No employment corrections or current-share substitutions are proposed.

## Reproduction and verification

Run producer `step_1/code/pipeline/102_h2_winsorization_cell_audit.py` with
`--corrected --leading-rhs`, then `--corrected --leading-rhs-report`.
The existing corrected cache and leading-window source diagnostics are prerequisites.
All 90 national shifts are checked against the sum of their geographic components;
all own-county and other-county contributions sum to the saved category contribution.
The Mineral parent-contract keys are nonmissing and unique.
These checks establish consistency with the parsed inputs, not independent accuracy
of the underlying government records. Employment verification is not extended by
these modes.
'''
    path=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/LEADING_COUNTY_RHS_AUDIT.md'
    assert '\u2014' not in body
    path.write_text(body)
    assert path.read_text()==body
    print(f'PASS: saved RHS evidence verified; wrote {path}')


def leading_county_validation():
    """Save focused source checks without changing geography or employment."""
    if not CORRECTED:
        raise SystemExit('FAIL: leading validation requires --corrected')
    pieces=[]
    with duckdb.connect() as con:
        for year,contract in [(2004,'SP060004D0454'),(2005,'SP060005D0453')]:
            path=dc.USASPENDING_OUT_DIR/f'fy{year}.parquet'
            rows=con.execute('''SELECT primary_place_of_performance_city_name city,
                prime_award_transaction_place_of_performance_county_fips_code raw_county,
                county_fips assigned_county, sum(federal_action_obligation) nominal_obligations,
                count(*) source_rows FROM read_parquet(?)
                WHERE award_id_piid=? OR parent_award_id_piid=? GROUP BY ALL''',
                [str(path),contract,contract]).df()
            if rows.empty:
                raise SystemExit(f'FAIL: no overlap records for {contract}')
            rows['year'],rows['contract']=year,contract
            rows['source_file']=str(path)
            pieces.append(rows)
        timeline=con.execute('''SELECT cast(fipstate as integer)*1000+cast(fipscty as integer) ccg,
            year, substr(naics12,1,2) sector, sum(emp) employment
            FROM read_parquet(?) WHERE
            (cast(fipstate as integer)=49 AND cast(fipscty as integer)=3 AND year BETWEEN 1994 AND 2001)
            OR (cast(fipstate as integer)=32 AND cast(fipscty as integer)=21 AND year BETWEEN 2005 AND 2011)
            GROUP BY ALL ORDER BY ccg,year,sector''',[str(dc.EFSY_PANEL)]).df()
    pd.concat(pieces,ignore_index=True).to_csv(OUTDIR/'leading_kleberg_overlap_geography.csv',index=False)
    timeline.to_csv(OUTDIR/'leading_employment_sector_timelines.csv',index=False)
    totals=timeline.groupby(['ccg','year'],as_index=False).employment.sum()
    totals.to_csv(OUTDIR/'leading_employment_total_timelines.csv',index=False)
    print(pd.concat(pieces,ignore_index=True).to_string(index=False))
    print(totals.to_string(index=False))
    print('Source checks saved. These do not authorize a geography or employment correction.')


def leading_county_report():
    """Document the leading county omissions and the limits of source validation."""
    if not CORRECTED:
        raise SystemExit('FAIL: leading report requires --corrected')
    import json
    cells=pd.read_csv(OUTDIR/'leading_county_year_gap_contributions.csv')
    selected=pd.read_csv(OUTDIR/'leading_county_selected_years.csv')
    ranking=pd.read_csv(OUTDIR/'all_county_leaveout_ranking.csv').query('rank<=5')
    annual=pd.read_csv(OUTDIR/'h2_winsorization_source_audit_annual_leaders.csv')
    employment=pd.read_csv(OUTDIR/'h2_winsorization_employment_source_totals_leaders.csv')
    components=pd.read_csv(OUTDIR/'h2_winsorization_instrument_components_leaders.csv')
    sectors=pd.read_csv(OUTDIR/'h2_winsorization_employment_sector_changes_leaders.csv')
    joint=pd.read_csv(OUTDIR/'top_county_group_gaps.csv')
    overlap=pd.read_csv(OUTDIR/'leading_kleberg_overlap_geography.csv').fillna('')
    assert len(cells)==350 and len(selected)==30 and len(ranking)==10
    assert not cells.duplicated(['ccg','year']).any()
    np.testing.assert_allclose(annual.positive_real,annual.cached_positive_real,rtol=1e-10,atol=1e-5)
    np.testing.assert_allclose(employment.cached_employment,employment.source_employment,rtol=1e-12,atol=1e-8)
    endpoint=selected.set_index(['ccg','year'])
    source_sums=components.groupby(['ccg','year']).z_contribution.sum()
    np.testing.assert_allclose(source_sums,endpoint.z.reindex(source_sums.index),rtol=1e-10,atol=1e-8)
    manifest=dc.MACRO_RESULTS_ROOT.parent/'data/source/primary_documents/raw/bls_mlr_june_2007_katrina.fetch.json'
    evidence=json.loads(manifest.read_text())
    assert 'official-artifact' in json.dumps(evidence)
    summaries=[]
    for r in ranking.itertuples():
        g=cells.loc[cells.ccg.eq(r.excluded_ccg)]
        np.testing.assert_allclose(g.gap_closure_contribution_x1e5.sum(),r.gap_closed_x1e5,atol=1e-10)
        summaries.append({'weight':r.weight,'county':r.county,'state_name':r.state_name,
            'ccg':r.excluded_ccg,'gap_closed_percent':r.gap_reduction_percent,
            **{f'{v}_capped_years':int(g[v].ne(g[f'{v}_percentile_1_99']).sum()) for v in ('x','y','z')}})
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUTDIR/'leading_county_investigation_summary.csv',index=False)
    assert summary.loc[summary.ccg.eq(6059),['x_capped_years','y_capped_years','z_capped_years']].eq(0).all().all()
    def md(frame):
        frame=frame.copy()
        if 'county' in frame:
            frame['county']=frame.county.str.title()
        frame=frame.rename(columns={'state_name':'State','county':'County','weight':'Weight',
            'gap_closed_percent':'Gap closed (%)','x_capped_years':'Years x capped',
            'y_capped_years':'Years y capped','z_capped_years':'Years z capped',
            'gap_closure_contribution_x1e5':'Contribution x 100,000',
            'employment_before':'Employment t-2','employment_after':'Employment t',
            'capped_variables':'Variables capped','year':'Year','cat':'Category',
            'market_share':'Historical market share','z_contribution':'Component of z','z_total':'Total z'})
        return frame.to_markdown(index=False,floatfmt='.6f')
    def case(ccg,year):
        return endpoint.loc[(ccg,year)]
    def jobs(ccg,year,sector):
        return sectors.loc[sectors.ccg.eq(ccg)&sectors.year.eq(year)&sectors.sector.astype(str).eq(str(sector)),'employment_change'].iloc[0]
    brief=selected[['county','state_name','year','gap_closure_contribution_x1e5','employment_before','employment_after']].copy()
    brief['capped_variables']=[' '.join(v for v in ('x','y','z') if getattr(r,v)!=getattr(r,f'{v}_percentile_1_99')) or 'none' for r in selected.itertuples()]
    instrument=components.loc[components.rank_absolute_z.eq(1),['county','year','cat','market_share','z_contribution','z_total']]
    notes=[]
    for r in overlap.itertuples():
        notes.append(f'| {r.year} | `{r.contract}` | {r.source_rows} | ${r.nominal_obligations:,.2f} | {r.raw_county or "blank"} | {r.assigned_county or "blank"} |')
    joint_five=joint.loc[joint.n_excluded.eq(5)].set_index('weight').gap_reduction_percent
    body=f'''# Leading counties in the raw versus winsorized gap

## Conclusion

No additional data correction is established by this investigation. The ten counties
are the top five actual county-deletion gap reducers under each weighting scheme,
not a list of proven data errors. Removing them jointly closes {joint_five['equal county']:.6f}% of the
equal-weight gap and {joint_five['fixed 1975 population']:.6f}% of the fixed-population gap
([county-omission audit, Table 1](COUNTY_LEAVEOUT_GAP_AUDIT.md)).

Three findings matter. Orange County is untouched by winsorization in every sample
year, despite its effect on the gap. Orleans has an independently documented disaster
in its leading employment window. Box Elder and Mineral require employment-source
checks; Kleberg requires contract-specific geography evidence. None of those open
checks justifies dropping a county from the baseline.

## Table 1. Gap closure and number of capped years

Each county has 35 observations, 1982 through 2016. `x` is the two-year procurement
change per resident; `y` is two-year midpoint employment growth; `z` is the instrument.
The gap is the winsorized coefficient minus the raw coefficient. Gap closure comes
from refitting both specifications after deleting the entire county time series.
Cutoffs and retained instruments are held fixed; division-by-year effects are refitted.
Population weights are fixed in 1975.

{md(summary.drop(columns='ccg'))}

Source: [county summary](../../Results/instrument_normalization/corrected_gap_audit/leading_county_investigation_summary.csv).

## What the individual cases show

### Austin County, Texas: contract timing, not an established duplicate

The relevant years are 2008 through 2010 (Table 2). The existing
[FMTV contract audit](AUSTIN_FMTV_CONTRACT_AUDIT.md), Tables 1 and 2, documents
211 distinct transaction keys and compares the initial contract actions with the
Department of Justice complaint. Large obligations booked in one year do not mean
the corresponding production or payroll occurred in that year. This is a plausible
timing mismatch, not evidence that the obligations should be erased. That audit's
unresolved contract-total reconciliation remains open.

### Box Elder, Utah: employment around the industry-classification transition

Employment is {case(49003,1998).employment_before:,.0f} in 1996 and
{case(49003,1998).employment_after:,.0f} in 1998. Manufacturing sector 33 contributes
{jobs(49003,1998,33):,.3f} jobs to that change, while professional services sector 54
contributes {jobs(49003,1998,54):,.3f}. The 1997-to-1999 comparison also spans this
period (Table 2). These sector movements make the SIC-to-NAICS transition and
source imputation worth checking, but do not establish that either caused the
county-total change. Industry reclassification alone should not change total jobs.
The instrument's largest category component in both 1998 and 1999 is 1427 (Table 3).

Status: cached county totals match the EFSY panel. A comparison with original CBP
county-total records is still needed; no independent validation of these totals is
claimed. The saved annual timeline covers 1994 through 2001.

### Sagadahoc, Maine: recurring ship-procurement exposure

Spending and the instrument are each capped in 30 of 35 years (Table 1).
The leading instrument category is 1905 in all three selected windows (Table 3).
The contract records identify Bath Iron Works, including `N0002485C2036`,
`N0002487C2221`, and `N0002413C2305`. The source records therefore point to repeated
concentrated ship procurement rather than one isolated outlier. This is not an
independent validation of contract amounts or their local production timing.

### Kleberg, Texas: a contract-specific location question remains open

The leading instrument categories are fuel codes 9140 and 9130 (Table 3).
The DCADS actions assign fuel suppliers Coastal and Valero to Corpus Christi,
Kleberg County. Corpus Christi is a multicounty place, so its city name alone
cannot establish an error. The historical geography audit classifies the city
as a legitimate multicounty case
([geography review](REMAINING_REPEATED_CANDIDATES.md)).

Two exact contract identifiers occur in the same fiscal years in USAspending, but
both raw and assigned county fields are blank there (Table 4). Those matches
cannot resolve the county. The needed evidence is the contract's performance or
delivery address, not merely the supplier's corporate or refinery address.
No reassignment to Nueces County is warranted on the evidence available here.

### Mineral, Nevada: an employment decline requiring original-source validation

Employment falls from {case(32021,2009).employment_before:,.0f} in 2007 to
{case(32021,2009).employment_after:,.0f} in 2009. Sector 49 accounts for
{jobs(32021,2009,49):,.0f} jobs of the change. The uncapped midpoint rate is
{case(32021,2009).y:.6f}; both the outcome and the instrument are capped (Table 2).
The contract records identify Hawthorne depot operations, and category M151
accounts for the largest instrument component in 1998 and 2009 (Table 3).
No source inspected here establishes whether the employment fall is a genuine
contraction, a coverage change, or a reporting problem. EFSY reconciliation alone
cannot answer that question. The saved annual timeline covers 2005 through 2011.

### Nassau, New York: instrument capping rather than outcome capping

Only `z` is capped in the three selected years; neither `x` nor `y` is capped
there (Table 2). Category 1510 is the largest instrument component in 1983 and
1984, with historical county market shares shown in Table 3. The contract records
identify Grumman aircraft procurement. Thus the immediate mechanism in these
windows is compression of the instrument, not trimming employment growth.
No additional geography or amount error has been verified.

### Orange, California: none of its observations is winsorized

All three cap counts are zero (Table 1). Nevertheless, omitting Orange closes
{summary.loc[summary.ccg.eq(6059),'gap_closed_percent'].iloc[0]:.6f}% of the
population-weighted gap. Its employment changes, spending and instrument are
identical in the raw and capped input panels.

This is possible because capping other counties changes the regression slopes,
the fixed-effect means, and the instrument-spending covariance used by 2SLS.
Orange's unchanged observations therefore affect the two fitted coefficients
differently. Its deletion effect is not evidence of an extreme Orange observation
that should be cleaned. Its leading years are 1985, 1986, and 1992 (Table 2).

### Orleans, Louisiana: the outcome cap intersects Hurricane Katrina

Employment falls from {case(22071,2006).employment_before:,.0f} in 2004 to
{case(22071,2006).employment_after:,.0f} in 2006. Midpoint growth of
{case(22071,2006).y:.6f} is capped at {case(22071,2006).y_percentile_1_99:.6f},
while spending and the instrument are untouched in that observation (Table 2).
Accommodation and food services accounts for {jobs(22071,2006,72):,.0f} jobs and
health care for {jobs(22071,2006,62):,.0f} jobs of the change.

The BLS article in the June 2007 *Monthly Labor Review*, printed page 7 and Chart 1,
documents a June 2006 year-over-year New Orleans employment loss of 92,900 following
Katrina. The article identifies its geographic unit as Orleans Parish. It supports
the disaster interpretation, but does not validate the exact EFSY totals: its
employment coverage and time comparison differ.
[Official BLS issue held by FRASER](https://fraser.stlouisfed.org/files/docs/publications/bls_mlr/bls_mlr_200706.pdf).

Evidence: official-artifact; official-clause-inspected. Direct BLS retrieval returned
HTTP 403; the official FRASER copy was retrieved and its extracted text inspected.
The 2006 contribution in Table 2 is an allocation of the county deletion result,
not an estimate of the effect of deleting the Katrina year alone.

### Baltimore City, Maryland: mixed instrument and uncapped observations

Only the instrument is capped in 1989; none of the variables is capped in the
selected 1988 and 1992 observations (Table 2). The leading instrument categories
are 5865 and 5841 (Table 3). Source actions identify Westinghouse and Martin
Marietta. The employment endpoints reconcile to EFSY, but a contractor name does
not prove a performance location. No county reassignment or employment correction
has been established.

### Fairfax county group, Virginia: leading years are uncapped

The three selected years, 1984, 1999, and 2000, are uncapped even though other
years in this county group are capped (Tables 1 and 2). Employment rises from
{case(51059,1999).employment_before:,.0f} in 1997 to
{case(51059,1999).employment_after:,.0f} in 1999. Professional services accounts
for {jobs(51059,1999,54):,.0f} of that increase. Category D399 is the largest
instrument component in 1999 and 2000 (Table 3). The saved source checks do not
identify a data error. The analysis unit is the fixed Fairfax county group,
not Fairfax County alone.

## Table 2. Three largest signed year contributions per county

Contributions are scaled by 100,000. Across all 35 years they sum exactly to the
change in the gap from deleting the whole county. They are not individual-year
deletion estimates. The three displayed contributions need not sum to the county
total because other years can offset them.

{md(brief)}

Source: [all 350 contributions](../../Results/instrument_normalization/corrected_gap_audit/leading_county_year_gap_contributions.csv).
Employment-sector numbers in the case descriptions come from
[sector changes](../../Results/instrument_normalization/corrected_gap_audit/h2_winsorization_employment_sector_changes_leaders.csv).

## Table 3. Largest absolute instrument component in each selected window

Market share is a fraction. The exposure window is five years, from t-11 through
t-7; the national shift is average spending in t-1 and t minus spending in t-2.
The instrument is divided by county population in t-2. No clipping is applied to
these underlying instrument components. Components can offset one another, so the
largest absolute component can exceed the total instrument.

{md(instrument)}

Source: [all category components](../../Results/instrument_normalization/corrected_gap_audit/h2_winsorization_instrument_components_leaders.csv).

## Table 4. Kleberg fuel-contract overlap, USAspending records

| Fiscal year | Contract identifier | Rows | Nominal obligations | Raw county | Assigned county |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(notes)}

These are contract-identifier matches, not a claim that all individual transaction
amounts match across sources. Source:
[overlap records](../../Results/instrument_normalization/corrected_gap_audit/leading_kleberg_overlap_geography.csv).

## Verification and remaining work

- All {len(annual)} selected annual procurement totals reconcile to the corrected
  source records; maximum absolute difference is
  ${abs(annual.positive_real-annual.cached_positive_real).max():.12f}.
- All {len(employment)} distinct employment endpoints reconcile to EFSY; maximum
  absolute difference is {employment.cache_minus_source.abs().max():.12g} jobs.
- All 30 selected instruments reconstruct from saved category-year dollars.
- These tests rule out discrepancies between the analysis cache and the inspected
  inputs. They do not independently establish input accuracy.
- Original county-total CBP validation remains open for Box Elder and Mineral.
  The locally available EFSY raw archive covers 1989, not the needed endpoint years.
- Kleberg requires historical contract performance or delivery addresses.
- No data, instrument, sample, weight, or regression estimate is changed by these
  diagnostic modes. No additional regression fits are run.

## Reproduction and files

Producer: `step_1/code/pipeline/102_h2_winsorization_cell_audit.py`.
Run `--corrected --leading-cases`, `--corrected --leading-sources`,
`--corrected --leading-validation`, then `--corrected --leading-report`.
The required county-deletion fits are documented in `COUNTY_LEAVEOUT_GAP_AUDIT.md`.
Source actions are in
[transaction evidence](../../Results/instrument_normalization/corrected_gap_audit/h2_winsorization_source_audit_top_actions_leaders.csv).
The [annual spending reconciliation](../../Results/instrument_normalization/corrected_gap_audit/h2_winsorization_source_audit_annual_leaders.csv),
[employment reconciliation](../../Results/instrument_normalization/corrected_gap_audit/h2_winsorization_employment_source_totals_leaders.csv),
and [employment timelines](../../Results/instrument_normalization/corrected_gap_audit/leading_employment_total_timelines.csv)
are saved alongside it.
'''
    doc=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/LEADING_COUNTY_GAP_INVESTIGATION.md'
    assert '\u2014' not in body
    doc.write_text(body)
    assert doc.read_text()==body
    print(f'PASS: 10 county allocations, 74 spending years, 57 employment endpoints, and 30 instruments verified. Wrote {doc}')


def mechanism_audit(suffix_override=None):
    """Trace each selected instrument to saved national and local category dollars."""
    p98 = module('p98_mechanisms',PIPELINE/'98_h2_all_category_market_share_design_grid.py')
    suffix=suffix_override if suffix_override is not None else ('' if CORRECTED else '_after_four')
    cases = pd.read_csv(OUTDIR/f'h2_winsorization_source_audit_cases{suffix}.csv')
    a = pd.read_parquet(CACHE/('winsorization_cell_attribution.parquet' if suffix_override is not None else f'winsorization_cell_attribution{suffix}.parquet'))
    a = a.loc[a.weight.eq('equal county')].merge(cases[['ccg','year']],on=['ccg','year'],validate='one_to_one')
    years = sorted({y for t in cases.year for y in list(range(t-11,t-6))+[t-2,t-1,t]})
    county_ids = sorted(cases.ccg.unique())
    national,local = {},{}
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        for year in years:
            path = str(CACHE/'annual_category'/f'fy={year}.parquet') if CORRECTED else str(p98.cache_path('positive',0,year))
            national[year] = con.execute('SELECT cat, sum(dollars) dollars FROM read_parquet(?) GROUP BY cat',[path]).df().set_index('cat').dollars
            local[year] = con.execute('SELECT ccg, cat, sum(dollars) dollars FROM read_parquet(?) WHERE ccg IN ('+
                ','.join(map(str,county_ids))+') GROUP BY ccg,cat',[path]).df()
    components=[]
    for r in a.itertuples():
        exposure_years=range(r.year-11,r.year-6)
        natpre=pd.concat([national[y] for y in exposure_years],axis=1).fillna(0).sum(axis=1)/5
        localpre=pd.concat([local[y].loc[local[y].ccg.eq(r.ccg)] for y in exposure_years])
        localpre=localpre.groupby('cat',sort=False).dollars.sum()/5
        work=pd.DataFrame({'county_pre':localpre,'national_pre':natpre}).fillna(0)
        work=work.loc[work.national_pre.gt(0)]
        work['market_share']=work.county_pre/work.national_pre
        for j in (-2,-1,0):
            work[f'national_{j}']=national[r.year+j].reindex(work.index).fillna(0)
            county=local[r.year+j].loc[local[r.year+j].ccg.eq(r.ccg)].set_index('cat').dollars
            work[f'county_{j}']=county.reindex(work.index).fillna(0)
        work['national_shift']=.5*(work['national_-1']+work['national_0'])-work['national_-2']
        work['z_contribution']=work.market_share*work.national_shift/r.population_before
        work['x_category_contribution']=(.5*(work['county_-1']+work['county_0'])-work['county_-2'])/r.population_before
        np.testing.assert_allclose(work.z_contribution.sum(),r.z,rtol=1e-10,atol=1e-9)
        work=work.loc[work.county_pre.gt(0)].copy()
        work['ccg'],work['county'],work['state_name'],work['year']=r.ccg,r.county,r.state_name,r.year
        work['exposure_start'],work['exposure_end']=r.year-11,r.year-7
        work['z_total'],work['x_total']=r.z,r.x
        work['rank_absolute_z']=work.z_contribution.abs().rank(method='first',ascending=False).astype(int)
        components.append(work.reset_index())
    result=pd.concat(components,ignore_index=True)
    result.to_csv(OUTDIR/f'h2_winsorization_instrument_components{suffix}.csv',index=False)
    a.to_csv(OUTDIR/f'h2_winsorization_case_endpoints{suffix}.csv',index=False)
    print(result.loc[result.rank_absolute_z.le(2),['county','year','cat','exposure_start','exposure_end',
        'market_share','national_shift','z_contribution','z_total','x_category_contribution','x_total']].to_string(index=False))
    print(f'PASS: category sums reproduce all {len(a)} selected cached instruments. No instrument or exposure changes.')


def employment_audit(suffix_override=None):
    """Compare selected county employment endpoints with source EFSY sectors."""
    suffix=suffix_override if suffix_override is not None else ('' if CORRECTED else '_after_four')
    a=pd.read_csv(OUTDIR/f'h2_winsorization_case_endpoints{suffix}.csv')
    target=pd.concat([a[['ccg','year','employment_after']].rename(columns={'employment_after':'cached_employment'}),
        a[['ccg','year','employment_before']].assign(year=a.year-2).rename(columns={'employment_before':'cached_employment'})])
    target=target.drop_duplicates(['ccg','year'])
    fips_map,_,_=dc.load_ccg_map()
    mapping=pd.DataFrame({'fips':list(fips_map),'ccg':list(fips_map.values())})
    mapping=mapping.loc[mapping.ccg.isin(a.ccg)]
    mapping['fips']=mapping.fips.astype(int)
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        con.register('mapping',mapping)
        con.register('targets',target[['ccg','year']])
        sectors=con.execute('''SELECT m.ccg, r.year, substr(r.naics12,1,2) sector,
            sum(r.emp) source_employment, count(*) source_rows,
            sum(CASE WHEN r.emp<0 THEN r.emp ELSE 0 END) negative_employment
            FROM read_parquet(?) r JOIN mapping m
            ON CAST(r.fipstate AS INTEGER)*1000+CAST(r.fipscty AS INTEGER)=m.fips
            JOIN targets t ON m.ccg=t.ccg AND r.year=t.year
            GROUP BY m.ccg,r.year,sector''',[str(dc.EFSY_PANEL)]).df()
    totals=sectors.groupby(['ccg','year'],sort=False)[['source_employment','source_rows','negative_employment']].sum().reset_index()
    totals=target.merge(totals,on=['ccg','year'],how='left',validate='one_to_one')
    if totals.source_employment.isna().any():
        raise SystemExit('FAIL: missing source employment endpoints')
    totals['cache_minus_source']=totals.cached_employment-totals.source_employment
    labels=a[['ccg','county','state_name']].drop_duplicates('ccg')
    totals=totals.merge(labels,on='ccg',validate='many_to_one')
    changes=[]
    for r in a.itertuples():
        s=sectors.loc[sectors.ccg.eq(r.ccg)].pivot(index='sector',columns='year',values='source_employment').fillna(0)
        d=(s[r.year]-s[r.year-2]).rename('employment_change').reset_index()
        d['ccg'],d['county'],d['state_name'],d['year']=r.ccg,r.county,r.state_name,r.year
        d['total_employment_change']=r.employment_after-r.employment_before
        d['rank_absolute_change']=d.employment_change.abs().rank(method='first',ascending=False).astype(int)
        changes.append(d)
    changes=pd.concat(changes,ignore_index=True)
    output_suffix=suffix_override or ''
    totals.to_csv(OUTDIR/f'h2_winsorization_employment_source_totals{output_suffix}.csv',index=False)
    changes.to_csv(OUTDIR/f'h2_winsorization_employment_sector_changes{output_suffix}.csv',index=False)
    print(totals.to_string(index=False))
    print(changes.loc[changes.rank_absolute_change.le(2)].to_string(index=False))
    print(f'Checked {len(totals)} source endpoints; maximum absolute cache-source difference = {totals.cache_minus_source.abs().max():.12g}. Source accuracy is not independently established.')


def corrected_report():
    """Write the corrected-cache accounting and source-verification evidence."""
    if not CORRECTED:
        raise SystemExit('FAIL: report requires --corrected')
    summary=pd.read_csv(OUTDIR/'h2_winsorization_attribution_summary.csv')
    leading=pd.read_csv(OUTDIR/'component_leading_cells.csv')
    annual=pd.read_csv(OUTDIR/'h2_winsorization_source_audit_annual.csv')
    actions=pd.read_csv(OUTDIR/'h2_winsorization_source_audit_top_actions.csv')
    endpoints=pd.read_csv(OUTDIR/'h2_winsorization_case_endpoints.csv')
    components=pd.read_csv(OUTDIR/'h2_winsorization_instrument_components.csv')
    employment=pd.read_csv(OUTDIR/'h2_winsorization_employment_source_totals.csv')
    leaveout=pd.read_csv(OUTDIR/'h2_winsorization_leaveout_gaps.csv')
    np.testing.assert_allclose(annual.positive_real,annual.cached_positive_real,rtol=1e-10,atol=.01)
    np.testing.assert_allclose(employment.cached_employment,employment.source_employment,rtol=1e-10,atol=1e-8)
    by_case=components.groupby(['ccg','year']).z_contribution.sum()
    np.testing.assert_allclose(by_case,endpoints.set_index(['ccg','year']).z.reindex(by_case.index),rtol=1e-10,atol=1e-9)
    for r in summary.itertuples():
        np.testing.assert_allclose(r.beta_cap_xyz_x1e5-r.beta_raw_x1e5,
            r.x_contribution_x1e5+r.y_contribution_x1e5+r.z_contribution_x1e5,rtol=1e-10)
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(map(str,row))+' |' for row in rows])
    def link(stem):
        return f'[`{stem}.csv`](../../Results/instrument_normalization/corrected_gap_audit/{stem}.csv)'
    stages=table(['Weight','Raw','Cap spending only','Also cap employment','Also cap instrument','Spending covariance removed'],
        [[r.weight,f'{r.beta_raw_x1e5:.6f}',f'{r.beta_cap_x_x1e5:.6f}',f'{r.beta_cap_xy_x1e5:.6f}',
          f'{r.beta_cap_xyz_x1e5:.6f}',f'{100*r.lost_spending_covariance_share:.4f}%'] for r in summary.itertuples()])
    leaders=leading.groupby(['weight','component'],sort=False).head(3)
    leaders_table=table(['Weight','Capped variable','County','Year','Contribution to coefficient change'],
        [[r.weight,r.component,f'{r.county}, {r.state_name}',r.year,f'{r.component_contribution_x1e5:+.6f}'] for r in leaders.itertuples()])
    instrument_table=table(['County','Year','Largest category','Exposure years','National-category market share','Category instrument','Total instrument'],
        [[f'{r.county}, {r.state_name}',r.year,r.cat,f'{r.exposure_start} to {r.exposure_end}',
          f'{100*r.market_share:.4f}%',f'{r.z_contribution:,.2f}',f'{r.z_total:,.2f}']
          for r in components.loc[components.rank_absolute_z.eq(1)].itertuples()])
    austin=endpoints.loc[endpoints.ccg.eq(48015)&endpoints.year.eq(2010)].iloc[0]
    timing=actions.loc[(actions.ccg.eq(48015)&actions.year.eq(2008)&actions.contract.eq('W56HZV08C0460'))|
                      (actions.ccg.eq(23023)&actions.year.eq(2011))].copy()
    timing=timing.sort_values(['ccg','year','dollars_nominal'],ascending=[True,True,False]).groupby('ccg',sort=False).head(3)
    timing_table=table(['County','Contract','Modification','Action date','Nominal obligation','Recorded performance end'],
        [[f'{r.county}, {r.state_name}',r.contract,r.modification,r.action_date,f'${r.dollars_nominal:,.2f}',r.performance_end]
         for r in timing.itertuples()])
    emp_table=table(['County','Outcome year','Employment two years before','Employment at outcome year'],
        [[f'{r.county}, {r.state_name}',r.year,f'{r.employment_before:,.6f}',f'{r.employment_after:,.6f}']
         for r in endpoints.loc[endpoints.ccg.isin([13053,32021,22071,2013])].itertuples()])
    leaveout_table=table(['Weight','Full-sample gap','Gap without Austin County','Gap reduction'],
        [[r.weight,
          f'{leaveout.loc[(leaveout.excluded.isna())&(leaveout.weight.eq(r.weight)),"gap_x1e5"].iloc[0]:.6f}',
          f'{r.gap_x1e5:.6f}',f'{r.gap_reduction_percent:.4f}%']
         for r in leaveout.loc[leaveout.excluded.eq('Austin, Texas')].itertuples()])
    us=annual.loc[annual.transaction_key_check_applicable]
    body=f'''# What drives the corrected-data winsorization gap?

## Scope

The audit holds the balanced sample, four-character instrument, five-year exposure
window, division-by-year fixed effects, and weights fixed. It uses the certified
`data/temp/report_correction_rebuild/analysis_pooled.parquet` cache. Capping uses
the pooled, unweighted 1st and 99th percentiles of each variable, before removing
fixed effects. Weights are equal county or fixed 1975 population. No observations
are excluded, no source values are edited, and no alternative grid is estimated.

## Table 1. Where the coefficient changes

Coefficients are multiplied by 100,000. The sequence is raw, cap spending x,
then employment growth y, then instrument z.

{stages}

Source: {link('h2_winsorization_attribution_summary')}.

The main upward step comes from capping spending. Capping employment has a small
net effect along this path. Capping the instrument offsets part of the spending
step under equal weighting but adds to it under population weighting (Table 1).
Large positive and negative cell contributions can cancel; a small net component
does not mean every underlying observation is innocuous.

With one instrument, the coefficient is the instrument-outcome crossproduct
divided by the instrument-spending crossproduct, after applying the specified
weights and fixed effects. Capping x alone leaves the numerator unchanged and
changes the denominator. The last column of Table 1 is one minus the capped-x
denominator divided by the raw denominator. It is not a share of procurement
dollars, an F-statistic, or a sample-exclusion rule.

This is exact, order-dependent accounting, not a causal decomposition. Intermediate
stages have no standard errors calculated here. A county's contribution is not
the coefficient change that would result from deleting that county.

## Table 2. Does Austin County explain the gap?

The regression removes Austin County's entire 35-year series while retaining the
full-sample cutoffs and the instrument values of all remaining counties.

{leaveout_table}

Source: {link('h2_winsorization_leaveout_gaps')}. Removing Austin County closes
10.6949% of the equal-weight gap and 2.3405% of the population-weighted gap
(Table 2). Austin County therefore does not explain the difference between the
raw and jointly capped estimates.

## A concrete spending observation

For Austin County, Texas, in 2010, x is {austin.x:,.2f} real dollars per resident;
capping makes it {austin.x_percentile_1_99:,.2f}. Employment midpoint growth is
{austin.y:.6f}, unchanged by its cap. The spending endpoints are
${austin.spending_before:,.2f} in 2008, ${austin.spending_post1:,.2f} in 2009, and
${austin.spending_post2:,.2f} in 2010, in the cache's real-dollar units. The formula is
the average of the last two amounts minus the first, divided by the 2008 population
of {austin.population_before:,.0f}. Source: {link('h2_winsorization_case_endpoints')}.

Thus the transformation compresses the recorded spending movement without
compressing this employment response. A much larger coefficient per capped dollar
need not reflect a larger employment response. This establishes the arithmetic,
not which version measures the economically relevant spending shock.

## Table 3. Leading cells in each component

The three largest absolute contributions for each variable and weight determine
the source-audit cases. Signed contributions are in coefficient units times
100,000. They need not have the same sign as the observation's total contribution.

{leaders_table}

Source: {link('component_leading_cells')}. The full county rankings are in
{link('h2_winsorization_attribution_counties')}.

## Table 4. What makes the instruments large

Each row shows the largest absolute category contribution for one audited case.
Instrument values are in real dollars per resident. Market shares use the displayed
historical exposure window, not current employment or regression residuals.

{instrument_table}

Source: {link('h2_winsorization_instrument_components')}. Summing all categories
reproduces each of the {len(endpoints)} audited instrument observations. The
historical category share can generate a large instrument even when contemporaneous
local spending in that category is small; this is distinct from a data-merge error.

## Table 5. Obligation dates versus performance dates

These are transaction records, not independently verified deliveries or outlays.

{timing_table}

Source: {link('h2_winsorization_source_audit_top_actions')}. Austin's May 2008
record describes 10,000 vehicles; its September option record describes 9,300
vehicles. Their recorded performance ends in 2010. The Sagadahoc records likewise
book obligations against contracts with later performance end dates (Table 5).
This makes obligation timing versus production timing a concrete hypothesis to
investigate. A performance end date does not reveal annual delivery shares and
does not justify spreading dollars uniformly across years.

## What the source checks establish

- All {len(annual)} selected county-year spending totals reproduce the parsed
  source after the approved report-correction and geography rules.
- All {len(employment)} employment endpoints reproduce the EFSY input;
  the largest absolute numerical difference is {employment.cache_minus_source.abs().max():.12g}.
- Across {len(us)} audited USAspending county-years, there are
  {int(us.rows_with_repeated_transaction_key.sum())} rows with a repeated nonmissing
  transaction key within county-year and {int(us.missing_transaction_keys.sum())}
  missing keys. This check does not detect the same economic action recorded under
  different keys, or prove that obligation amounts are correct. DCADS report
  numbers are not treated as globally unique transaction keys.

Sources: {link('h2_winsorization_source_audit_annual')} and
{link('h2_winsorization_employment_source_totals')}.

Austin here is Austin County, containing Sealy, not the city of Austin. The
guarded Sealy assignment is supported by the geographic and same-year contract
evidence in [CITY_COUNTY_GEOGRAPHY_AUDIT.md](CITY_COUNTY_GEOGRAPHY_AUDIT.md).
The 1982 St. James cancellation is outside the 1977 to 1981 exposure window
for its 1988 instrument; see [ST_JAMES_ACTION_CORRECTION_AUDIT.md](ST_JAMES_ACTION_CORRECTION_AUDIT.md).
Neither observation should be declared erroneous simply because it is influential.

## Table 6. Employment cases requiring independent source checks

{emp_table}

Source: {link('h2_winsorization_case_endpoints')}; sector contributions are in
{link('h2_winsorization_employment_sector_changes')}. Matching the EFSY input
rules out an aggregation discrepancy at these endpoints, not errors in EFSY,
coverage changes, or establishment-level reporting problems. These employment
cases remain unresolved. Table 1 nevertheless shows that employment capping is
not the principal net source of the coefficient increase along this path.

## Conclusion and remaining verification

The coefficient gap is traceable to the compression of large spending movements,
not an unexplained change in the regression sample. The source audit does not
establish that all influential amounts are genuine, nor does it justify deleting
counties or choosing cutoffs to improve significance.

The Austin action history is in
[AUSTIN_FMTV_CONTRACT_AUDIT.md](AUSTIN_FMTV_CONTRACT_AUDIT.md). Its initial
action plus definitization matches the amount in an official Department of
Justice complaint to $12.82, and the complaint places vehicle claims for payment
between December 2008 and December 2011. The remaining amount checks are the
large Sagadahoc ship contracts in Table 5 and N0002488C2055 in Newport News.
Distinguish incremental obligations from contract ceilings, definitizations,
and reversals. Keep procurement timing separate from delivery and local
production. There is no additional approved data correction from these checks.

An attempted independent acquisition of the Defense Department's December 2009
FMTV Selected Acquisition Report ended `blocked-after-ladder` (HTTP 403).
Evidence status is `metadata-only`; indexed snippets are not amount verification.
The acquisition manifest is
`data/source/primary_documents/raw/dod_fmtv_sar_december_2009.fetch.json`.

## Reproduction

Producer: `step_1/code/pipeline/102_h2_winsorization_cell_audit.py`.
Run `--corrected`, then `--corrected --sources`, `--corrected --mechanisms`,
`--corrected --employment-audit`, and `--corrected --report`.
The report mode reads saved evidence and verifies the reconciliations; it runs no
regressions. Detailed outputs are in
`Results/instrument_normalization/corrected_gap_audit/`.
'''
    doc=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/CORRECTED_WINSORIZATION_GAP_AUDIT.md'
    doc.write_text(body)
    assert doc.read_text()==body
    print(f'PASS: report reconciliations and saved text verified. Wrote {doc}')


def austin_contract_audit():
    """Reconstruct the complete parsed history of Austin's leading FMTV contract."""
    contract='W56HZV08C0460'
    files=[str(dc.USASPENDING_OUT_DIR/f'fy{year}.parquet') for year in range(2007,2019)]
    with duckdb.connect() as con:
        actions=con.execute('''SELECT source_fiscal_year AS fiscal_year, action_date,
            modification_number, federal_action_obligation, TRY_CAST(total_dollars_obligated AS DOUBLE) AS reported_total_obligated,
            transaction_description, period_of_performance_start_date,
            period_of_performance_current_end_date, period_of_performance_potential_end_date,
            contract_transaction_unique_key, product_or_service_code, county_fips, ccg,
            primary_place_of_performance_city_name, primary_place_of_performance_county_name,
            primary_place_of_performance_state_code
            FROM read_parquet(?) WHERE award_id_piid=?
            ORDER BY action_date, modification_number, contract_transaction_unique_key''',[files,contract]).df()
    if len(actions)!=211 or sorted(actions.fiscal_year.unique())!=list(range(2008,2018)):
        raise SystemExit('FAIL: unexpected FMTV contract history')
    if actions.contract_transaction_unique_key.isna().any() or actions.contract_transaction_unique_key.duplicated().any():
        raise SystemExit('FAIL: missing or repeated transaction key')
    if not actions.ccg.eq(48015).all() or not actions.county_fips.astype(str).eq('48015').all():
        raise SystemExit('FAIL: contract does not map entirely to Austin County')
    actions['cumulative_obligation']=actions.federal_action_obligation.cumsum()
    annual=(actions.groupby('fiscal_year',sort=True).federal_action_obligation
            .agg(n_actions='size',positive_obligations=lambda s:s[s.gt(0)].sum(),
                 deobligations=lambda s:s[s.lt(0)].sum(),net_obligations='sum').reset_index())
    annual['cumulative_net_obligations']=annual.net_obligations.cumsum()
    base=actions.loc[actions.modification_number.isin(['0','PZ0005']),'federal_action_obligation'].sum()
    official_base=2_099_328_517
    official_option=1_666_884_022
    checks=pd.DataFrame([
        {'comparison':'Base award and definitization','parsed_obligations':base,
         'official_complaint_amount':official_base,'difference':base-official_base,
         'official_location':'paragraph 21, PDF page 6'},
        {'comparison':'All fiscal 2008 actions','parsed_obligations':annual.loc[annual.fiscal_year.eq(2008),'net_obligations'].iloc[0],
         'official_complaint_amount':official_base+official_option,
         'difference':annual.loc[annual.fiscal_year.eq(2008),'net_obligations'].iloc[0]-official_base-official_option,
         'official_location':'paragraph 21, PDF page 6; option is authorized amount'}])
    if abs(checks.iloc[0].difference)>20:
        raise SystemExit('FAIL: parsed base award does not match official complaint')
    source=dc.MACRO_RESULTS_ROOT.parent/'data/source/primary_documents/raw'
    complaint=source/'doj_bae_fmtv_complaint_2015.txt'
    manifest=source/'doj_bae_fmtv_complaint_2015.fetch.json'
    import json
    fetched=json.loads(manifest.read_text())
    if fetched['final_status']!='official-artifact' or fetched['signature']!='pdf':
        raise SystemExit('FAIL: official complaint acquisition is not certified')
    text=complaint.read_text()
    for phrase in ['$2,099,328,517','May 30, 2008','26,789 FMTVs','more than $3.6 billion']:
        if phrase not in text:
            raise SystemExit(f'FAIL: official complaint phrase missing: {phrase}')
    OUTDIR.mkdir(parents=True,exist_ok=True)
    actions.to_csv(OUTDIR/'austin_fmtv_contract_actions.csv',index=False)
    annual.to_csv(OUTDIR/'austin_fmtv_contract_annual.csv',index=False)
    checks.to_csv(OUTDIR/'austin_fmtv_official_amount_checks.csv',index=False)
    def mdtable(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(map(str,row))+' |' for row in rows])
    annual_table=mdtable(['Fiscal year','Positive obligations','Deobligations','Net obligations','Cumulative net'],
        [[r.fiscal_year,f'${r.positive_obligations:,.2f}',f'${r.deobligations:,.2f}',
          f'${r.net_obligations:,.2f}',f'${r.cumulative_net_obligations:,.2f}'] for r in annual.itertuples()])
    initial=actions.loc[actions.fiscal_year.eq(2008)&actions.federal_action_obligation.ne(0)]
    initial_table=mdtable(['Date','Modification','Obligation','Description'],
        [[r.action_date,r.modification_number,f'${r.federal_action_obligation:,.2f}',
          str(r.transaction_description).replace('|','/')] for r in initial.itertuples()])
    report=f'''# Austin County FMTV contract audit

## Question

Contract `W56HZV08C0460` accounts for 76.4319% of positive nominal procurement
assigned to Austin County, Texas, in fiscal 2008. Austin's 2010 two-year spending
change is -$141,489.81 per resident before capping and -$1,224.43 after capping.
This audit asks whether the leading spending observation is a duplicate, a coding
error, or a large contract whose execution spans several years.

## Table 1. Complete parsed obligation history

{annual_table}

Source: [`austin_fmtv_contract_annual.csv`](../../Results/instrument_normalization/corrected_gap_audit/austin_fmtv_contract_annual.csv).
The action-level file contains {len(actions)} records, {actions.federal_action_obligation.gt(0).sum()}
positive obligations, {actions.federal_action_obligation.lt(0).sum()} deobligations,
and {actions.federal_action_obligation.eq(0).sum()} zero-dollar actions. Every
record has a distinct transaction key. The complete file is
[`austin_fmtv_contract_actions.csv`](../../Results/instrument_normalization/corrected_gap_audit/austin_fmtv_contract_actions.csv).

## Table 2. Fiscal 2008 actions with nonzero obligations

{initial_table}

The initial undefinitized action and modification `PZ0005` sum to
${base:,.2f}. The Department of Justice complaint states a base amount of
${official_base:,.2f}; the difference is ${base-official_base:,.2f}. It also
states that the contract included an option of ${official_option:,.2f} for an
additional 10,000 units. These figures are in paragraph 21, PDF page 6.

The same complaint states that the Army issued the undefinitized action on
May 30, 2008 (paragraph 27, PDF page 8). It states that BAE submitted claims for
payment for 26,789 vehicles totaling more than $3.6 billion from December 17,
2008, through December 19, 2011 (paragraph 83, PDF page 25). These are allegations
in a government complaint, not findings after trial. The dates and amounts cited
here are used only to validate the scale and timing of the procurement record.

Source: [official Department of Justice complaint](../../data/source/primary_documents/raw/doj_bae_fmtv_complaint_2015.pdf),
acquisition status `official-artifact`; evidence status `official-clause-inspected`.
The acquisition manifest is
[`doj_bae_fmtv_complaint_2015.fetch.json`](../../data/source/primary_documents/raw/doj_bae_fmtv_complaint_2015.fetch.json).

## Conclusion

The leading Austin observation is not explained by a repeated transaction key,
an incorrect county assignment, or double counting the initial award and its
definitization. The parsed initial action plus definitization matches the amount
in the official complaint to $12.82 (Table 2). The obligation series is genuinely
lumpy: it records $3.740 billion in fiscal 2008, while the complaint places vehicle
claims for payment between December 2008 and December 2011 (Table 1 and paragraph
83 of the complaint).

This evidence supports a timing interpretation. The 2010 regressor subtracts
fiscal 2008 obligations from the average of fiscal 2009 and 2010 obligations.
For this contract, those annual net amounts are $3.740 billion, $82.071 million,
and $201.138 million, respectively (Table 1), although contract execution and
claims continued across those years. Thus the extreme negative regressor is a
real feature of obligation timing, not evidence that local production stopped
between 2008 and 2010.

The audit does not establish the correct economic transformation. It does not
show how vehicle production, deliveries, payroll, or subcontracting were allocated
across years or counties. It therefore does not justify winsorization, deletion,
or spreading obligations mechanically across the performance period.

The Defense Department Selected Acquisition Report, GAO report 09-968R, and an
Armed Services Board document ended `blocked-after-ladder` with HTTP 403. Their
indexed text is not treated as verified amount or delivery evidence.

## Reproduction

Producer: `step_1/code/pipeline/102_h2_winsorization_cell_audit.py --corrected --austin-contract`.
The mode reads the parsed USAspending fiscal files and the certified official
complaint. It runs no regressions and makes no data changes.
'''
    doc=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/AUSTIN_FMTV_CONTRACT_AUDIT.md'
    doc.write_text(report)
    assert doc.read_text()==report
    print(annual.to_string(index=False))
    print(checks.to_string(index=False))
    print(f'PASS: {len(actions)} unique-key contract actions; official base amount matches within $20. Wrote {doc}')


def st_james_audit():
    """Trace St. James procurement geography, dollar encoding, and fuel exposure."""
    import hashlib
    annual,actions,fuelgeo=[],[],[]
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        for year in range(1966,1990):
            path=str(dc.OUT_DIR/f'fy{year}.parquet')
            d=con.execute("SELECT * FROM read_parquet(?) WHERE county_fips5='22093'",[path]).df()
            d['year']=year
            source_amount='dollars_reported' if 'dollars_reported' in d.columns else 'dollars'
            d['dollars_nominal']=d[source_amount]*np.where(d.dollars_unit.eq('thousands'),1000,1)
            d=d.loc[d.fsc.isna()|~d.fsc.astype(str).str.startswith('Y')]
            actions.append(d)
            for category in ('all','9130','9140'):
                g=d if category=='all' else d.loc[d.fsc.eq(category)]
                positive=g.loc[g.dollars_nominal.gt(0),'dollars_nominal'].sum()
                annual.append(dict(year=year,category=category,rows=len(g),
                    positive_nominal=positive,net_nominal=g.dollars_nominal.sum(),
                    positive_real=positive*dc.deflate(year),n_contracts=g.contract_no.nunique()))
            if year>=1977:
                g=con.execute(f'''SELECT fsc,geo_class,county_fips5,county_name,state_name,
                    sum(CASE WHEN {source_amount}>0 THEN {source_amount} ELSE 0 END
                        *CASE WHEN dollars_unit='thousands' THEN 1000 ELSE 1 END) positive_nominal,
                    sum({source_amount}*CASE WHEN dollars_unit='thousands' THEN 1000 ELSE 1 END) net_nominal,count(*) n_rows
                    FROM read_parquet(?) WHERE fsc IN ('9130','9140')
                    GROUP BY fsc,geo_class,county_fips5,county_name,state_name''',[path]).df()
                g['year']=year
                fuelgeo.append(g)
    a=pd.concat(actions,ignore_index=True)
    annual=pd.DataFrame(annual)
    fuelgeo=pd.concat(fuelgeo,ignore_index=True)
    a.to_csv(OUTDIR/'st_james_contract_actions.csv',index=False)
    annual.to_csv(OUTDIR/'st_james_spending_annual.csv',index=False)
    fuelgeo.to_csv(OUTDIR/'st_james_national_fuel_geography.csv',index=False)
    # Byte-level checks use declared field offsets and retain each source record.
    layouts,index=dc.load_layouts()
    files={p.name:p for p in dc.data_files()}
    rawrows=[]
    for year in (1981,1982):
        name=f'RG330.DEFCON.FY{str(year)[-2:]}'
        spec=layouts[index[name]]
        path=files[name]
        fields={f['name']:(f['start'],f['end']) for f in spec['fields']}
        record_length=spec['record_length']
        selected=a.loc[a.year.eq(year)]
        wanted=set(zip(selected.state_code,selected.city_code))
        with path.open('rb') as fh:
            record_number=0
            while block:=fh.read(record_length*10000):
                if len(block)%record_length:
                    raise SystemExit('FAIL: partial raw record')
                for pos in range(0,len(block),record_length):
                    record_number+=1
                    raw=block[pos:pos+record_length]
                    s=raw.decode(spec.get('encoding','latin-1'))
                    def field(k):
                        lo,hi=fields[k]
                        return s[lo:hi].strip()
                    if (field('state_code'),field('city_code')) not in wanted:
                        continue
                    lo,hi=fields['dollars']
                    amount=dc.decode_zoned(s[lo:hi])
                    rawrows.append(dict(year=year,record_number=record_number,
                        contract_no=field('contract_no'),report_no=field('report_no'),
                        contractor_code=field('contractor_code'),contractor_name=field('contractor_name'),
                        state_code=field('state_code'),city_code=field('city_code'),
                        county_name=field('county_name'),fsc=field('fsc'),
                        dollars_encoded=s[lo:hi],dollars_nominal=amount*1000,
                        record_sha256=hashlib.sha256(raw).hexdigest(),record_text=s,
                        source_path=str(path)))
    raw=pd.DataFrame(rawrows)
    for year in (1981,1982):
        r=raw.loc[raw.year.eq(year)]
        p=a.loc[a.year.eq(year)]
        # All selected source locations in these years must resolve to St. James.
        keys=['contract_no','report_no','contractor_code','dollars_nominal','fsc']
        if len(r)!=len(p):
            raise SystemExit('FAIL: source-location row count differs from assigned county')
        left=r.groupby(keys,dropna=False).size().sort_index()
        right=p.groupby(keys,dropna=False).size().sort_index()
        pd.testing.assert_series_equal(left,right,check_names=False)
    raw.to_csv(OUTDIR/'st_james_raw_record_verification.csv',index=False)
    print(annual.loc[(annual.year>=1977)&annual.category.isin(['all','9140'])].to_string(index=False))
    print(a.loc[a.year.isin([1981,1982])].groupby(['year','contract_no','contractor_name'],dropna=False).dollars_nominal.sum().sort_values(ascending=False).head(18).to_string())
    print(f'PASS: {len(raw)} raw source records match parsed records. Exact duplicate raw rows: {raw.duplicated(["year","record_sha256"]).sum()}. No corrections applied.')
    st_james_exposure()


def st_james_exposure():
    """Read only the six cached category-years entering the two exposure windows."""
    p98=module('p98_stjames',PIPELINE/'98_h2_all_category_market_share_design_grid.py')
    # Use the exact sparse category cache that produced the baseline instrument.
    exposure=[]
    with duckdb.connect() as con:
        for year in range(1977,1983):
            row=con.execute('''SELECT sum(dollars) national_dollars,
                sum(CASE WHEN ccg=22093 THEN dollars ELSE 0 END) county_dollars
                FROM read_parquet(?) WHERE cat='9140' ''',[str(p98.cache_path('positive',0,year))]).df().iloc[0]
            exposure.append(dict(year=year,national_real=row['national_dollars'],county_real=row['county_dollars']))
    exposure=pd.DataFrame(exposure)
    exposure.to_csv(OUTDIR/'st_james_fuel_exposure_annual.csv',index=False)
    print(exposure.to_string(index=False))


def st_james_report():
    """Document a source-coded correction without changing any analysis input."""
    strings={k:str for k in ['fsc','contract_no','report_no','contractor_code','correction_code']}
    a=pd.read_csv(OUTDIR/'st_james_contract_actions.csv',dtype=strings)
    raw=pd.read_csv(OUTDIR/'st_james_raw_record_verification.csv',dtype=strings)
    raw['correction_code_from_bytes']=raw.record_text.str[30:31].str.strip()
    trio=a.loc[a.year.eq(1982)&a.contract_no.eq('LA60082C5010')&a.report_no.eq('3082')].copy()
    trio['correction_code']=trio.correction_code.fillna('')
    labels={'':'Initial entry','1':'Corrected entry','2':'Reversal of initial entry'}
    trio['code_meaning']=trio.correction_code.map(labels)
    amounts=trio.set_index('correction_code').dollars_nominal
    assert len(trio)==3 and amounts['']==573747000 and amounts['1']==57375000 and amounts['2']==-573747000
    keys=['year','contract_no','report_no','contractor_code','dollars_nominal','fsc']
    verified=trio.merge(raw[keys+['record_sha256','record_number','correction_code_from_bytes']],
                        on=keys,validate='one_to_one')
    assert len(verified)==3 and verified.correction_code.eq(verified.correction_code_from_bytes).all()
    verified.to_csv(OUTDIR/'st_james_verified_correction_trio.csv',index=False)
    gross=trio.dollars_nominal.clip(lower=0).sum()
    revised=amounts['1']
    spending=pd.read_csv(OUTDIR/'st_james_spending_annual.csv',dtype={'category':str})
    exposure=pd.read_csv(OUTDIR/'st_james_fuel_exposure_annual.csv')
    e=exposure.loc[exposure.year.between(1977,1981)]
    market_share=e.county_real.sum()/e.national_real.sum()
    cmp=pd.read_csv(OUTDIR/'h2_winsorization_instrument_components_after_four.csv',dtype={'cat':str})
    z=cmp.loc[cmp.ccg.eq(22093)&cmp.year.eq(1988)&cmp.cat.eq('9140')].iloc[0]
    np.testing.assert_allclose(market_share,z.market_share,rtol=1e-10)
    assert exposure.loc[exposure.year.between(1977,1980),'county_real'].eq(0).all()
    # All raw-to-cache records, not only the three-row correction, must agree.
    assert len(raw)==111 and not raw.duplicated(['year','record_sha256']).any()
    for year in [1981,1982]:
        r=raw.loc[raw.year.eq(year)]
        p=a.loc[a.year.eq(year)]
        pd.testing.assert_series_equal(r.groupby(keys,dropna=False).size().sort_index(),
            p.groupby(keys,dropna=False).size().sort_index(),check_names=False)
    def table(frame,columns):
        lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
        for vals in frame[columns].itertuples(index=False,name=None):
            lines.append('| '+' | '.join(str(v) for v in vals)+' |')
        return '\n'.join(lines)
    display=verified[['calendar_yearmonth','correction_code','code_meaning','dollars_nominal']].copy()
    display.columns=['Reporting month','Code','Meaning','Nominal dollars']
    display['Nominal dollars']=display['Nominal dollars'].map(lambda v:f'{v:,.0f}')
    period=spending.loc[spending.category.eq('9140')&spending.year.between(1980,1986),['year','positive_nominal','net_nominal']].copy()
    period.columns=['Fiscal year','Positive dollars','Net dollars']
    for col in ['Positive dollars','Net dollars']:
        period[col]=period[col].map(lambda v:f'{v:,.0f}')
    doc=dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/ST_JAMES_ACTION_CORRECTION_AUDIT.md'
    body=f'''# St. James: source corrections and fuel-procurement exposure

## Finding and implementation status

One reporting correction is verified for fiscal year 1982, contract LA60082C5010,
report 3082, Derby & Co. Applying a positive-only filter to the uncorrected source
amounts retains a reversed initial report. The approved source cleaning and its
application status are documented in `REPORT_CORRECTION_CLEANING.md`. This audit
uses reported source amounts and does not estimate a counterfactual F or t.

## Table 1. The verified reporting correction

{table(display,list(display.columns))}

The FY1981/1982 Code Translation Manual explicitly defines blank as initial entry,
1 as corrected entry, and 2 as reversal of initial entry. Source: NARA file
`1_current_paper/_data/source/DCADS/DCADS/1982/124.7DP.pdf`, PDF page 170,
printed manual page 2 (NARA reference page 149). The page has been inspected as
both extracted text and a rendered image. The three amounts and correction codes
also match the original fixed-width records. Evidence:
[`st_james_verified_correction_trio.csv`](../../Results/instrument_normalization/st_james_verified_correction_trio.csv).

For these three rows, the positive filter counts ${gross:,.0f}; the corrected
positive action is ${revised:,.0f}. The excess is ${gross-revised:,.0f}.
This is an administrative correction, not a discretionary outlier cap and not an
argument to switch the entire spending series to net obligations. Genuine later
negative modifications remain a separate spending-definition choice.

The county-action and category-action readers in `pipeline/lib/contracts.py` do
not load correction codes. Positive aggregators then retain amounts greater than
zero. The raw parser preserves the codes; the loss occurs when the positive
spending series is constructed. All {len(raw)} St. James source records for
1981 and 1982 match the parsed records; there are no identical raw-row duplicates.

## Table 2. Fuel-oil procurement, nominal dollars

{table(period,list(period.columns))}

Source: [`st_james_spending_annual.csv`](../../Results/instrument_normalization/st_james_spending_annual.csv).
Positive and net dollars are displayed as observed, without applying Table 1's
correction. The 1981 episode remains large even under net accounting.

## Why the 1988 instrument is large

The exposure window is 1977 through 1981. Local fuel-oil spending is zero in
1977 through 1980; 1981 supplies the entire local numerator. Its share of pooled
national fuel-oil spending is {100*market_share:.6f}%. Multiplying that historical
share by the 1988 national two-year shock produces ${z.z_contribution:,.6f} per
resident, although local fuel-oil spending is zero in 1986, 1987, and 1988.
Sources: [`st_james_fuel_exposure_annual.csv`](../../Results/instrument_normalization/st_james_fuel_exposure_annual.csv)
and [`h2_winsorization_instrument_components_after_four.csv`](../../Results/instrument_normalization/h2_winsorization_instrument_components_after_four.csv).

The verified correction is in 1982, outside this exposure window. It therefore
does not explain the 1988 fuel-oil instrument component. An accounting error and
a potentially unrepresentative historical exposure coexist here.

## What the external documents establish

DOE's 1983 SPR Annual Report, included in its official FOIA release
[HQ-2019-01130-F](https://www.energy.gov/sites/default/files/2021-07/HQ-2019-01130-F%20-%201st%20Partial%20Response.pdf),
PDF page 16, describes the St. James facility as a government-owned marine
terminal. Page 18 records continuing SPR procurement in 1983 through
both DFSC and direct PEMEX contracts. Evidence status: official-clause-inspected.
The fall in local DCADS dollars cannot by itself establish that the terminal
closed or that procurement for the SPR ceased.

GAO report RCED-84-61, printed pages 1 through 4 (PDF pages 15 through 18),
describes oil purchases from suppliers and traders, including delivery at SPR
receiving terminals and direct DOE procurement from PEMEX. The report is stored
from the [Justia mirror](https://gao.justia.com/department-of-defense/1984/9/defense-fuel-supply-center-procedures-for-purchasing-strategic-petroleum-reserve-oil-rced-84-61/RCED-84-61-full-report.pdf).
Evidence status: mirror-text-only; direct GAO retrieval was blocked with HTTP 403.

These documents support a delivery-location interpretation of the fuel episode,
not a claim that the full contract value was local value added. They do not match
every St. James action to an individual SPR delivery. The precise reason local
DCADS fuel purchases disappear after 1982 is not established. The separate
court-case retrieval and GAO EMD-82-19 retrieval were blocked; their indexed
snippets are not used as verified document evidence.

## Correction rule and scope

The cross-file audit is in `REPORT_CORRECTION_AUDIT.md`. Approved initial/reversal
pairs are neutralized before applying the positive-only filter, with reported
amounts preserved. Genuine negative contract modifications are a separate concept.
County influence and regression significance do not enter this correction rule.
Application and downstream rebuild status are in `REPORT_CORRECTION_CLEANING.md`.

## Reproduction

Producer: `step_1/code/pipeline/102_h2_winsorization_cell_audit.py`.
`--st-james` reads the historical source records and the six cached exposure
years. `--st-james-report` verifies the saved evidence and writes this note.
Acquisition manifests and source documents are in `data/source/primary_documents/raw/`.
No regressions are run by either mode.
'''
    doc.parent.mkdir(parents=True,exist_ok=True)
    doc.write_text(body)
    assert doc.read_text()==body
    print(f'PASS: verified source-coded correction and byte-level record matches. Wrote {doc}')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corrected',action='store_true',help='Use certified report-corrected h2 inputs; save separate audit outputs.')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--sources',action='store_true')
    mode.add_argument('--leaveouts',action='store_true')
    mode.add_argument('--original-groups',action='store_true',help='Fit the exact archived four- and nine-county groups on corrected data.')
    mode.add_argument('--original-groups-report',action='store_true',help='Compare fixed county groups across archived and corrected inputs.')
    mode.add_argument('--after-nine-screen',action='store_true',help='Freeze a RHS-only candidate list conditional on the original nine removals.')
    mode.add_argument('--after-nine-accounting',action='store_true',help='Account for the EW raw/capped gap with the original nine removals fixed.')
    mode.add_argument('--after-nine-concentration',action='store_true',help='Report signed cell and county concentration of the fixed-sample EW gap.')
    mode.add_argument('--after-nine-fits',action='store_true',help='Test the frozen RHS candidates individually and in small groups.')
    mode.add_argument('--after-nine-sources',action='store_true',help='Trace RHS-selected county-years to contracts and instrument categories.')
    mode.add_argument('--after-nine-overlap',action='store_true',help='Check food-contract geography in overlapping source years.')
    mode.add_argument('--after-nine-report',action='store_true',help='Write all RHS candidate results and source-record evidence.')
    mode.add_argument('--ew-focus',action='store_true',help='Separate EW-selected additions and diagnose clustered uncertainty.')
    mode.add_argument('--ew-sources',action='store_true',help='Trace leading EW precision cases to RHS source records.')
    mode.add_argument('--ew-continuity',action='store_true',help='Save EW contract geography and food-contract reporting continuity.')
    mode.add_argument('--ew-report',action='store_true',help='Write verified EW inference and source-continuity evidence.')
    mode.add_argument('--all-counties',action='store_true',help='Rank all corrected county omissions; estimate top-5/10/20 groups.')
    mode.add_argument('--all-counties-report',action='store_true',help='Verify saved full county scan and write the gap report.')
    mode.add_argument('--gap50',action='store_true',help='Extend county omissions through 50 and split closure between LHS and RHS.')
    mode.add_argument('--gap50-report',action='store_true',help='Write the county-gap LHS/RHS split and full inference tables.')
    mode.add_argument('--leading-cases',action='store_true',help='Identify the years behind the leading ten county omissions.')
    mode.add_argument('--leading-sources',action='store_true',help='Trace saved leading years to spending, categories, and employment sources.')
    mode.add_argument('--leading-validation',action='store_true',help='Save focused employment timelines and Kleberg cross-source geography checks.')
    mode.add_argument('--leading-rhs',action='store_true',help='Trace RHS category components to current local activity and national origins.')
    mode.add_argument('--leading-rhs-report',action='store_true',help='Document RHS exposure, category, and obligations evidence.')
    mode.add_argument('--leading-report',action='store_true',help='Verify and document the leading county source investigation.')
    mode.add_argument('--after-four',action='store_true')
    mode.add_argument('--sources-after-four',action='store_true')
    mode.add_argument('--next-group',action='store_true')
    mode.add_argument('--mechanisms',action='store_true')
    mode.add_argument('--employment-audit',action='store_true')
    mode.add_argument('--report',action='store_true',help='Write the corrected-data accounting and source-evidence memo.')
    mode.add_argument('--austin-contract',action='store_true',help='Audit Austin County FMTV contract actions and official evidence.')
    mode.add_argument('--calhoun-contract',action='store_true',help='Audit Calhoun Hydra signed obligations and the 2014 shock.')
    mode.add_argument('--tail-moments',action='store_true',help='Account for first-stage and reduced-form moments on fixed EW spending tails.')
    mode.add_argument('--own-shift-accounting',action='store_true',help='Split the fixed-nine EW instrument moments by own versus other county spending.')
    mode.add_argument('--own-shift-report',action='store_true',help='Write the saved own-shift accounting report without reconstructing components.')
    mode.add_argument('--own-shift-fits',action='store_true',help='Estimate raw and capped IV using the cached other-county component.')
    mode.add_argument('--first-stage-tails',action='store_true',help='Account for saved first-stage fit and cluster variance by spending/instrument tail overlap.')
    mode.add_argument('--first-stage-tail-report',action='store_true',help='Write saved first-stage tail diagnostics without recalculating moments.')
    mode.add_argument('--calhoun-illinois',action='store_true',help='Trace Calhoun Illinois leading second-stage gap episodes to contracts and categories.')
    mode.add_argument('--calhoun-illinois-contracts',action='store_true',help='Trace Belleville Shoe historical identities and same-year USAspending performance geography.')
    mode.add_argument('--calhoun-illinois-report',action='store_true',help='Document Belleville/Belleview evidence and bounded candidate relocation records.')
    mode.add_argument('--calhoun-illinois-leaveout',action='store_true',help='Exclude Calhoun and St. Clair, Illinois, jointly; EW, raw versus capped.')
    mode.add_argument('--st-james',action='store_true')
    mode.add_argument('--st-james-report',action='store_true')
    args=parser.parse_args()
    if args.corrected:
        if args.after_four or args.sources_after_four or args.next_group or args.st_james or args.st_james_report:
            parser.error('--corrected does not support after-four, next-group, or St. James modes')
        import json
        CORRECTED=True
        CACHE=dc.MACRO_VERSION_TEMP_DIR/'report_correction_rebuild'
        certificate=json.loads((CACHE/'certification.json').read_text())
        if certificate['status']!='PASS':
            raise SystemExit('FAIL: corrected cache certification')
        REFERENCE=OUTDIR/'h2_report_correction_comparison.csv'
        OUTDIR=OUTDIR/'corrected_gap_audit'
        OUTDIR.mkdir(parents=True,exist_ok=True)
    if args.leading_cases:
        leading_county_cases()
    elif args.leading_validation:
        leading_county_validation()
    elif args.leading_rhs:
        leading_rhs_audit()
    elif args.leading_rhs_report:
        leading_rhs_report()
    elif args.leading_report:
        leading_county_report()
    elif args.leading_sources:
        if not CORRECTED:
            parser.error('--leading-sources requires --corrected')
        source_audit(case_override=pd.read_csv(OUTDIR/'leading_county_selected_years.csv'))
        mechanism_audit(suffix_override='_leaders')
        employment_audit(suffix_override='_leaders')
    elif args.gap50:
        gap50_analysis()
    elif args.gap50_report:
        gap50_report()
    elif args.all_counties_report:
        all_county_report()
    elif args.all_counties:
        all_county_leaveouts()
    elif args.austin_contract:
        austin_contract_audit()
    elif args.calhoun_contract:
        calhoun_contract_audit()
    elif args.tail_moments:
        ew_tail_moments()
    elif args.own_shift_accounting:
        ew_own_shift_accounting()
    elif args.own_shift_report:
        ew_own_shift_report()
    elif args.own_shift_fits:
        ew_other_county_fits()
    elif args.first_stage_tails:
        ew_first_stage_tails()
    elif args.first_stage_tail_report:
        ew_first_stage_tail_report()
    elif args.calhoun_illinois:
        calhoun_illinois_sources()
    elif args.calhoun_illinois_contracts:
        calhoun_illinois_contracts()
    elif args.calhoun_illinois_leaveout:
        calhoun_illinois_leaveout()
    elif args.calhoun_illinois_report:
        calhoun_illinois_report()
    elif args.report:
        corrected_report()
    elif args.st_james_report:
        st_james_report()
    elif args.st_james:
        st_james_audit()
    elif args.employment_audit:
        employment_audit()
    elif args.mechanisms:
        mechanism_audit()
    elif args.original_groups:
        leaveout_comparison(original_groups=True)
    elif args.original_groups_report:
        original_groups_report()
    elif args.after_nine_screen:
        after_nine_rhs_screen()
    elif args.after_nine_accounting:
        main(after_nine=True)
    elif args.after_nine_concentration:
        after_nine_gap_concentration()
    elif args.after_nine_fits:
        after_nine_rhs_fits()
    elif args.after_nine_sources:
        if not CORRECTED:
            parser.error('--after-nine-sources requires --corrected')
        source_audit(case_override=pd.read_csv(OUTDIR/'after_nine_rhs_case_years.csv'),suffix_override='_after_nine_rhs')
        mechanism_audit(suffix_override='_after_nine_rhs')
    elif args.after_nine_overlap:
        after_nine_source_overlap()
    elif args.after_nine_report:
        after_nine_rhs_report()
    elif args.ew_focus:
        ew_focus()
    elif args.ew_continuity:
        ew_source_continuity()
    elif args.ew_report:
        ew_focus_report()
    elif args.ew_sources:
        if not CORRECTED:
            parser.error('--ew-sources requires --corrected')
        cases=pd.read_csv(OUTDIR/'ew_precision_leading_years.csv')
        cases=cases.loc[cases['sample'].eq('Add EW top 3')&cases.rank_abs_within_county.le(2)]
        source_audit(case_override=cases,suffix_override='_ew_precision')
        mechanism_audit(suffix_override='_ew_precision')
    elif args.leaveouts or args.next_group:
        leaveout_comparison(next_group=args.next_group)
    elif args.sources or args.sources_after_four:
        source_audit(after_four=args.sources_after_four)
    else:
        main(after_four=args.after_four)
