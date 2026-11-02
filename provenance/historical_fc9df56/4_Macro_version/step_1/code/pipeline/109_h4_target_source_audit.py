"""Trace the fixed h=4 backward-restoration target to annual inputs and actions.

Read-only with respect to source data, analysis caches, and specifications.
Outputs are a source investigation, not a sample selection or an error ledger.
County treatment includes unreadable category codes; the composition instrument
requires harmonized readable categories. Canonical harmonize_arrays carries
source row IDs through one-to-many bridges to preserve contract attribution.
"""
# mpgr-run-class: cache_augment
# mpgr-cache: h4_winsorization_diagnosis/analysis.parquet
# mpgr-new-estimates: 0
# mpgr-variation-scope: none; fixed 226-cell source audit
import argparse
import hashlib
import json
import sys
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path[:0] = [str(PIPELINE.parent/'py'), str(PIPELINE/'lib')]
import dcads_common as dc
import harmonize

BASE = dc.MACRO_RESULTS_ROOT/'instrument_normalization/h4_winsorization_diagnosis'
OUT = BASE/'source_audit'
SOURCE = dc.MACRO_VERSION_TEMP_DIR/'report_correction_rebuild'
ANALYSIS = dc.MACRO_VERSION_TEMP_DIR/'h4_winsorization_diagnosis/analysis.parquet'
TARGET = BASE/'backward_target_observations.csv'
MEMO = dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/H4_TARGET_AUDIT.md'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def load_target():
    a = pd.read_csv(TARGET)
    full = pd.read_parquet(ANALYSIS)
    check = json.loads((BASE/'backward_target_check.json').read_text())
    cert = json.loads((ANALYSIS.parent/'certification.json').read_text())
    assert sha(ANALYSIS) == cert['cache_sha256']
    assert len(a) == 226 and a.ccg.nunique() == 55
    assert not a.duplicated(['ccg','year']).any()
    pd.testing.assert_frame_equal(a[full.columns].reset_index(drop=True),
        full.iloc[a.row_id][full.columns].reset_index(drop=True), check_dtype=False,
        check_exact=False, rtol=1e-12, atol=1e-12)
    mask = np.zeros(len(full), bool); mask[a.row_id] = True
    assert hashlib.sha256(np.packbits(mask).tobytes()).hexdigest() == check['final_state_sha256']
    OUT.mkdir(parents=True, exist_ok=True)
    return a


def trace():
    a = load_target()
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        annual = con.execute('''SELECT ccg,cat,
            CAST(regexp_extract(filename,'fy=([0-9]+)',1) AS INTEGER) AS year,
            sum(dollars) AS dollars FROM read_parquet(?,filename=true,hive_partitioning=false)
            GROUP BY ccg,cat,year''', [str(SOURCE/'annual_category/fy=*.parquet')]).df()
    nat = annual.groupby(['cat','year']).dollars.sum().unstack('year').fillna(0)
    local = annual.loc[annual.ccg.isin(a.ccg)].pivot(index=['ccg','cat'],columns='year',values='dollars').fillna(0)
    spending = pd.read_parquet(SOURCE/'county_spending_positive.parquet').pivot(index='ccg',columns='year',values='dollars')
    components, links, targets = [], [], []
    for r in a.itertuples():
        b = r.year-4; pre = list(range(r.year-13,r.year-8)); post = list(range(b+1,r.year+1))
        loc = local.loc[r.ccg].reindex(nat.index,fill_value=0)
        den = nat[pre].mean(axis=1); ex = loc[pre].mean(axis=1)
        shift = nat[post].mean(axis=1)-nat[b]
        shares = (ex/den.where(den>0)).fillna(0)
        z = shares*shift/r.population_base
        np.testing.assert_allclose(z.sum(),r.z,rtol=1e-10,atol=1e-7)
        xx = (spending.loc[r.ccg,post].mean()-spending.loc[r.ccg,b])/r.population_base
        np.testing.assert_allclose(xx,r.x,rtol=1e-10,atol=1e-7)
        yy = 2*(r.employment_end-r.employment_base)/(r.employment_end+r.employment_base)
        np.testing.assert_allclose(yy,r.y,rtol=1e-12,atol=1e-12)
        f = pd.DataFrame(dict(cat=nat.index,z_component=z.to_numpy(),
            exposure_local=ex.to_numpy(),exposure_national=den.to_numpy(),market_share=shares.to_numpy(),
            national_base=nat[b].to_numpy(),national_post=nat[post].mean(axis=1).to_numpy(),
            local_base=loc[b].to_numpy(),local_post=loc[post].mean(axis=1).to_numpy()))
        f = f.loc[(f.exposure_local!=0)|(f.local_base!=0)|(f.local_post!=0)].copy()
        f['row_id']=r.row_id; f['ccg']=r.ccg; f['year']=r.year
        f['abs_z_share']=f.z_component.abs()/z.abs().sum()
        components.append(f)
        lead = f.loc[f.z_component.abs().idxmax()]
        targets.append(dict(row_id=r.row_id,top_category=lead['cat'],top_category_abs_z_share=lead.abs_z_share,
            reconstructed_x=xx,reconstructed_z=z.sum(),opposite_x_z=np.sign(r.x)!=np.sign(r.z)))
        for role, years in [('exposure',pre),('baseline',[b]),('post',post)]:
            for sy in years:
                links.append(dict(row_id=r.row_id,ccg=r.ccg,year=r.year,source_year=sy,role=role,
                    annual_spending=spending.loc[r.ccg,sy]))
    c = pd.concat(components,ignore_index=True)
    c.to_parquet(OUT/'instrument_components.parquet',index=False)
    l = pd.DataFrame(links);l.to_csv(OUT/'source_year_links.csv',index=False)
    t = a.merge(pd.DataFrame(targets),on='row_id',validate='one_to_one')
    # Connected outcome windows organize repeated years, not distinct award identities.
    episodes=[]
    for ccg,g in t.groupby('ccg',sort=True):
        end=-1;ep=0
        for r in g.sort_values('year').itertuples():
            if r.year-4>end: ep+=1
            end=max(end,r.year)
            episodes.append(dict(row_id=r.row_id,window_group=f'{ccg:05d}-{ep:02d}'))
    t=t.merge(pd.DataFrame(episodes),on='row_id',validate='one_to_one')
    t.to_csv(OUT/'target_inventory.csv',index=False)
    t.groupby(['ccg','county','state','window_group']).agg(target_cells=('row_id','size'),
        first_target=('year','min'),last_target=('year','max')).reset_index().to_csv(OUT/'window_groups.csv',index=False)
    # Leading national origins of the largest instrument component, including
    # counties outside the target set; retain signed contributions, not only levels.
    origins=[]
    for r in t.itertuples():
        v=annual.loc[annual.cat.eq(r.top_category)].pivot(index='ccg',columns='year',values='dollars').fillna(0)
        v=v.reindex(columns=range(1966,2019),fill_value=0)
        b=r.year-4; pre=list(range(r.year-13,r.year-8)); post=list(range(b+1,r.year+1))
        d=v[post].mean(axis=1)-v[b]
        for origin in d.abs().nlargest(5).index:
            origins.append(dict(row_id=r.row_id,target_ccg=r.ccg,year=r.year,cat=r.top_category,
                origin_ccg=origin,origin_base=v.loc[origin,b],origin_post=v.loc[origin,post].mean(),
                origin_exposure=v.loc[origin,pre].mean(),origin_shift=d[origin],
                national_shift=d.sum(),share_absolute_origins=abs(d[origin])/d.abs().sum()))
    pd.DataFrame(origins).to_csv(OUT/'national_shift_origins.csv',index=False)
    manifest=dict(status='PASS',analysis_sha256=sha(ANALYSIS),target_sha256=sha(TARGET),
        n_targets=len(a),n_counties=int(a.ccg.nunique()),n_window_groups=t.window_group.nunique(),
        n_source_county_years=len(l[['ccg','source_year']].drop_duplicates()),
        source_category_files={p.name:sha(p) for p in sorted((SOURCE/'annual_category').glob('*.parquet'))},
        construction_reconciled=True,source_records_verified=False,new_regressions=0)
    (OUT/'trace_check.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='source_category_files'},indent=2),flush=True)


def read_actions(con,year,units,mapping):
    path=(dc.OUT_DIR if year<dc.SEAM_YEAR else dc.USASPENDING_OUT_DIR)/f'fy{year}.parquet'
    schema=con.execute('SELECT * FROM read_parquet(?) LIMIT 0',[str(path)]).df().columns
    def field(names,alias):
        name=next((n for n in names.split('|') if n in schema),None)
        return (f'CAST(r."{name}" AS VARCHAR)' if name else 'CAST(NULL AS VARCHAR)')+f' AS "{alias}"'
    common=[('action_date','action_date'),('completion_date|period_of_performance_current_end_date','performance_end'),
        ('contractor_name|recipient_name','contractor'),('fsc|product_or_service_code','category'),
        ('fsc_description|transaction_description','description'),('contract_no|award_id_piid','contract'),
        ('mod_no|modification_number','modification'),('report_no|contract_transaction_unique_key','action_key'),
        ('reporting_dept|reporting_dept_fips|awarding_agency_code','agency'),('parent_award_id_piid','parent_award'),
        ('city_name|primary_place_of_performance_city_name','city'),('county_name|primary_place_of_performance_county_name','reported_county'),
        ('county_fips5|county_fips','assigned_fips'),('fips_source|county_fips_source','geography_source'),
        ('contractor_city','recipient_city'),('contractor_state|recipient_state_code','recipient_state'),
        ('est_duns','duns'),('source_file|source_member','source_file'),('source_row_number','source_row_number'),
        ('dollars_unit','dollars_unit'),('dollars_reported','dollars_reported'),
        ('report_correction_neutralized','neutralized'),('n_actions|number_of_actions','n_actions')]
    sels=[field(n,k) for n,k in common]
    if year<dc.SEAM_YEAR:
        con.register('county_map',mapping.loc[mapping.ccg.isin(units)])
        query='SELECT m.ccg,r.dollars * CASE WHEN r.dollars_unit=\'thousands\' THEN 1000 ELSE 1 END AS dollars,'+','.join(sels)+''' FROM read_parquet(?) r JOIN county_map m ON r.county_fips5=m.fips
            WHERE r.geo_class='county' AND (r.fsc IS NULL OR substr(CAST(r.fsc AS VARCHAR),1,1)<>'Y')'''
    else:
        query='SELECT r.ccg,r.federal_action_obligation AS dollars,'+','.join(sels)+''' FROM read_parquet(?) r WHERE r.geo_class='county' AND r.award_or_idv_flag='AWARD'
            AND (r.product_or_service_code IS NULL OR substr(CAST(r.product_or_service_code AS VARCHAR),1,1)<>'Y') AND r.ccg IN ('''+','.join(map(str,units))+')'
    f=con.execute(query,[str(path)]).df();f['year']=year;f['source']='DCADS' if year<dc.SEAM_YEAR else 'USAspending'
    return f,path


def records():
    a=load_target(); trace_check=json.loads((OUT/'trace_check.json').read_text())
    assert trace_check['target_sha256']==sha(TARGET)
    links=pd.read_csv(OUT/'source_year_links.csv')
    need=links[['ccg','source_year']].drop_duplicates()
    spend=pd.read_parquet(SOURCE/'county_spending_positive.parquet').set_index(['ccg','year']).dollars
    fm,_,_=dc.load_ccg_map();mapping=pd.DataFrame({'fips':list(fm),'ccg':list(fm.values())})
    assert not mapping.fips.duplicated().any()
    labels=a.drop_duplicates('ccg').set_index('ccg')[['county','state']]
    annual=[];tops=[];contract_parts=[];cat_parts=[];man=[]
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        for year,g in need.groupby('source_year',sort=True):
            year=int(year); units=list(map(int,g.ccg));f,path=read_actions(con,year,units,mapping)
            assert f.dollars.notna().all()
            valid=f.category.notna() & f.category.str[:4].str.len().eq(4)
            dv,ids,cats=harmonize.harmonize_arrays(year,f.loc[valid,'dollars'].to_numpy(np.float64),
                f.index[valid].to_numpy(np.int64),f.loc[valid,'category'].str[:4].to_numpy())
            h=pd.DataFrame({'source_id':ids,'harmonized_category':cats,'harmonized_dollars':dv})
            h=h.loc[~h.harmonized_category.str.startswith('Y')]
            h=h.join(f[['ccg','contract','agency','parent_award','contractor']],on='source_id',validate='many_to_one')
            pos=h.loc[h.harmonized_dollars>0]
            observed=pos.groupby(['ccg','harmonized_category']).harmonized_dollars.sum()*dc.deflate(year)
            expected=pd.read_parquet(SOURCE/'annual_category'/f'fy={year}.parquet')
            expected=expected.loc[expected.ccg.isin(units)].set_index(['ccg','cat']).dollars
            observed.index.names=expected.index.names
            pair=pd.concat([observed.rename('observed'),expected.rename('expected')],axis=1).fillna(0)
            np.testing.assert_allclose(pair.observed,pair.expected,rtol=1e-9,atol=.02)
            for ccg in units:
                v=f.loc[f.ccg.eq(ccg)];p=v.loc[v.dollars>0];neg=v.loc[v.dollars<0]
                positive=p.dollars.sum();real=positive*dc.deflate(year);target=spend.loc[ccg,year]
                np.testing.assert_allclose(real,target,rtol=1e-10,atol=.02)
                nkey=v.action_key.dropna();nkey=nkey.loc[nkey.ne('')]
                cg=pos.loc[pos.ccg.eq(ccg)].groupby('harmonized_category').harmonized_dollars.sum().sort_values(ascending=False)
                for cat,dol in cg.items():
                    cat_parts.append(dict(ccg=ccg,year=year,cat=cat,positive_nominal=dol,share_county_positive=dol/positive if positive else np.nan))
                annual.append(dict(ccg=ccg,year=year,county=labels.loc[ccg,'county'],state=labels.loc[ccg,'state'],
                    n_records=len(v),n_positive=len(p),n_negative=len(neg),positive_nominal=positive,negative_nominal=neg.dollars.sum(),
                    net_nominal=v.dollars.sum(),positive_real=real,cached_real=target,
                    top_action_share=p.dollars.max()/positive if positive else np.nan,
                    negative_to_positive=-neg.dollars.sum()/positive if positive else np.nan,
                    top_category=cg.index[0] if len(cg) else '',
                    top_category_share=cg.iloc[0]/positive if positive and len(cg) else np.nan,
                    rows_repeated_action_key=int(nkey.duplicated(keep=False).sum()) if year>=dc.SEAM_YEAR else np.nan,
                    missing_action_keys=int(v.action_key.isna().sum()+v.action_key.eq('').sum()),
                    neutralized_records=int(v.neutralized.eq('true').sum()),
                    missing_contractor_names=int(v.contractor.isna().sum()),
                    evidence_status='parsed_source_reconciled; award_amount_and_location_unverified'))
                for side,subset in [('positive',p.nlargest(5,'dollars')),('negative',neg.nsmallest(5,'dollars'))]:
                    top=subset.copy();top['tail']=side;tops.append(top)
            f['positive']=f.dollars.clip(lower=0);f['negative']=f.dollars.clip(upper=0)
            keys=['ccg','year','agency','parent_award','contract','contractor']
            agg=f.groupby(keys,dropna=False).agg(positive_nominal=('positive','sum'),negative_nominal=('negative','sum'),
                records=('dollars','size'),largest_action=('dollars','max')).reset_index()
            contract_parts.append(agg)
            man.append(dict(year=year,path=str(path),sha256=sha(path),records_read=len(f),county_years=len(units)))
            print(f'PASS source {year}: {len(units)} county-years, {len(f):,} records; county and harmonized categories reconcile',flush=True)
    d=pd.DataFrame(annual);assert len(d)==len(need) and not d.duplicated(['ccg','year']).any()
    d.to_csv(OUT/'source_annual.csv',index=False)
    pd.concat(tops,ignore_index=True).to_csv(OUT/'source_top_actions.csv',index=False)
    pd.concat(contract_parts,ignore_index=True).to_parquet(OUT/'contract_annual.parquet',index=False)
    pd.DataFrame(cat_parts).to_parquet(OUT/'source_category_annual.parquet',index=False)
    pd.DataFrame(man).to_csv(OUT/'source_manifest.csv',index=False)
    assert sha(ANALYSIS)==trace_check['analysis_sha256'] and sha(TARGET)==trace_check['target_sha256']
    result=dict(status='PASS',source_county_years=len(d),records_read=sum(v['records_read'] for v in man),
        county_totals_reconciled=True,harmonized_category_totals_reconciled=True,
        external_award_verification='not_performed',data_corrections_applied=0,new_regressions=0)
    (OUT/'records_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def early_reports():
    """Inspect unresolved report families in the fixed target source years.

    A candidate is an equal-and-opposite amount within a reporting family,
    or an exact repeated source record with a positive current amount.
    Contract/action-date differences are retained rather than forced to match.
    Every candidate is checked against the immutable 272-byte source.
    No candidate is interpreted as an approved cancellation pair.
    """
    load_target()
    links=pd.read_csv(OUT/'source_year_links.csv')
    need=links.loc[links.source_year.le(1983),['ccg','source_year']].drop_duplicates()
    mapping,_,_=dc.load_ccg_map()
    spec=json.loads((PIPELINE.parent/'layouts/dcads_272.json').read_text())
    keys=['ccg','reporting_dept','purchasing_office','report_no','contractor_code','fsc']
    families=[];records=[]
    for year,g in need.groupby('source_year',sort=True):
        year=int(year);f=pd.read_parquet(dc.OUT_DIR/f'fy{year}.parquet')
        f['ccg']=f.county_fips5.map(mapping);f['record_number']=f.index+1
        f=f.loc[f.ccg.isin(g.ccg)&f.geo_class.eq('county')&~f.fsc.fillna('').str.startswith('Y')].copy()
        f['ccg']=f.ccg.astype(int)
        groups=[]
        for key,v in f.groupby(keys,dropna=False):
            if len(v)<2: continue
            if any(pd.isna(x) or str(x).strip()=='' for x in key): continue
            pos=set(v.loc[v.dollars.gt(0),'dollars']);neg=set(-v.loc[v.dollars.lt(0),'dollars'])
            opposite=bool(pos & neg)
            fields=[z['name'] for z in spec['fields']]
            # The reported amount is used in the raw signature; current dollars
            # determine whether an unneutralized positive entry remains.
            payload=v[fields].copy();payload['dollars']=v.dollars_reported
            duplicate=payload.duplicated(keep=False)&v.dollars.gt(0)
            if opposite or duplicate.any(): groups.append((key,v,opposite,bool(duplicate.any())))
        if not groups: continue
        source_names={v.source_file.iloc[0] for _,v,_,_ in groups}
        paths={name:list(dc.SOURCE_DIR.rglob(name)) for name in source_names}
        assert all(len(v)==1 for v in paths.values())
        for key,v,opposite,duplicate in groups:
            family=f'{year}:'+':'.join(map(str,key))
            hashes=[]
            path=paths[v.source_file.iloc[0]][0]
            with path.open('rb') as raw:
                for r in v.itertuples():
                    raw.seek((r.record_number-1)*272);b=raw.read(272);assert len(b)==272
                    text=b.decode('latin-1')
                    for field in spec['fields']:
                        value=text[field['start']:field['end']]
                        value=dc.decode_zoned(value) if field['type']=='int' else value.strip()
                        expected=r.dollars_reported if field['name']=='dollars' else getattr(r,field['name'])
                        assert value==expected,(year,r.record_number,field['name'])
                    digest=hashlib.sha256(b).hexdigest();hashes.append(digest)
                    row=r._asdict();row.pop('Index',None)
                    row.update(family_id=family,source_year=year,raw_sha256=digest,raw_hex=b.hex(),
                        raw_path=str(path),evidence_status='raw_fields_verified; correction_unresolved')
                    records.append(row)
            pos=v.dollars.clip(lower=0).sum()*1000;net=v.dollars.sum()*1000
            families.append(dict(family_id=family,source_year=year,ccg=key[0],contractor=v.contractor_name.iloc[0],
                report_no=key[3],category=key[-1],records=len(v),unique_raw_records=len(set(hashes)),
                equal_opposite_amount=opposite,positive_exact_duplicate=bool((pd.Series(hashes).duplicated(keep=False).to_numpy() & v.dollars.gt(0).to_numpy()).any()),
                positive_nominal=pos,net_nominal=net,n_contract_spellings=v.contract_no.nunique(),
                correction_codes=';'.join(sorted(v.correction_code.unique())),
                evidence_status='candidate_only; no data change'))
    f=pd.DataFrame(families).sort_values('positive_nominal',ascending=False)
    r=pd.DataFrame(records)
    f.to_csv(OUT/'early_report_candidates.csv',index=False)
    r.to_csv(OUT/'early_report_candidate_records.csv',index=False)
    berks=r.loc[r.ccg.eq(42011)&r.source_year.eq(1979)&r.report_no.eq('0937')]
    berks.to_csv(OUT/'berks_1979_report_0937.csv',index=False)
    result=dict(status='PASS',candidate_families=len(f),candidate_records=len(r),
        candidate_counties=int(f.ccg.nunique()),all_candidate_raw_fields_verified=True,
        approved_corrections=0,source_data_changed=False)
    (OUT/'early_reports_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f.head(15).to_string(index=False),flush=True);print(json.dumps(result,indent=2),flush=True)


def report():
    a=load_target();t=pd.read_csv(OUT/'target_inventory.csv',dtype={'top_category':str})
    d=pd.read_csv(OUT/'source_annual.csv');check=json.loads((OUT/'records_check.json').read_text())
    assert check['status']=='PASS'
    rows=t.groupby(['ccg','county','state']).agg(cells=('year','size'),first=('year','min'),last=('year','max'),
        x_caps=('x_clipped','sum'),y_caps=('y_clipped','sum'),z_caps=('z_clipped','sum')).reset_index().sort_values(['cells','ccg'],ascending=[False,True])
    rows.to_csv(OUT/'county_inventory.csv',index=False)
    text=['# H=4 target source audit','',
        '## Scope','',
        'The target is the fixed backward-restoration set. This audit does not search for exclusions, rerun regressions, or change data. '
        'Capping this set is a sensitivity exercise, not a justified baseline cleaning rule. The set is path-dependent and is not proven minimal.','',
        f'Table 1 covers {len(t)} county-years in {t.ccg.nunique()} counties. '
        f'{int((~t.y_clipped).sum())} target rows have only spending/instrument caps; {int(t.y_clipped.sum())} also have an employment cap. '
        f'The audit reconciles {len(d)} distinct source county-years and {check["records_read"]:,} parsed records (source_annual.csv; records_check.json).','',
        '## What is verified','',
        'Each target employment outcome, spending treatment, and instrument reproduces the saved analysis value. '
        'Every audited source county-year reproduces both county positive spending and its harmonized category totals. '
        'These checks connect the regression cache to the parsed action files. They do not independently establish that a reported obligation, performance location, or source parsing is correct.','',
        '## What still requires investigation','',
        'The action files identify large awards and negative adjustments; their sizes alone are not errors. '
        'A negative adjustment is not automatically a cancellation of a positive record. A repeated contract number can represent legitimate modifications. '
        'Performance years need not coincide with the year an obligation is recorded. Recipient addresses alone do not establish performance locations. '
        'External award records have not been checked by this producer. No additional correction is established.','',
        'The source audit covers local spending and historical exposure years. National category shifts are traced to their largest county origins in national_shift_origins.csv, '
        'but records for origins outside the target county-year union are not yet audited. '
        'Employment endpoints reproduce the cache formula but are not independently verified against employment source records.','',
        '## Table 1. Target counties','',
        '| County | State | Target cells | First target year | Last target year | Spending caps | Employment caps | Instrument caps |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows.itertuples():
        text.append(f'| {r.county} | {r.state} | {r.cells} | {r.first} | {r.last} | {r.x_caps} | {r.y_caps} | {r.z_caps} |')
    text += ['', 'Source: county_inventory.csv and target_inventory.csv in Results/instrument_normalization/h4_winsorization_diagnosis/source_audit/.','',
        '## Files and interpretation','',
        '- target_inventory.csv: exact county-year IDs, raw/capped values, and largest instrument category.',
        '- source_year_links.csv: baseline, four post years, and five historical exposure years for every target.',
        '- instrument_components.parquet: category-level instrument contributions, market shares, national and local amounts.',
        '- window_groups.csv: connected overlapping outcome windows, not claims of a single award episode.',
        '- national_shift_origins.csv: five largest absolute county contributions to the leading category national shift per target.',
        '- source_annual.csv: positive and negative obligations, concentration, key availability, and reconciliation.',
        '- source_top_actions.csv: five largest positive and five largest negative actions per source county-year, with available dates and locations.',
        '- contract_annual.parquet: annual contract totals; missing agency, parent, or contractor fields are retained, so groups are audit leads rather than globally unique award identities.',
        '- source_category_annual.parquet: harmonized positive category totals, including one-to-many service bridges.',
        '- source_manifest.csv, trace_check.json, records_check.json: input identities, coverage and verification status.','',
        'The county spending series includes unreadable category codes. The instrument cannot allocate those records to categories. '
        'The resulting difference in coverage is part of the existing construction and is not treated here as an error.','']
    candidate_path=OUT/'early_report_candidates.csv'
    if candidate_path.exists():
        cf=pd.read_csv(candidate_path,dtype={'category':str,'correction_codes':str})
        cr=pd.read_csv(OUT/'early_report_candidate_records.csv',dtype={'contract_no':str,'report_no':str,'correction_code':str})
        affected=cf.merge(pd.read_csv(OUT/'source_year_links.csv'),on=['ccg','source_year'],validate='many_to_many')
        assert not affected.duplicated(['family_id','row_id','role']).any()
        affected[['family_id','row_id','ccg','year','source_year','role']].to_csv(OUT/'family_target_links.csv',index=False)
        text += ['## Table 2. Unresolved early report families','',
            f'The fixed source-year union contains {len(cf)} candidate families in {cf.ccg.nunique()} counties, linked to {affected.row_id.nunique()} target rows. '
            f'All {len(cr)} candidate records match the original 272-byte fields. '
            'These are investigation leads, not accepted errors or a proposed blanket netting rule. '
            'Selection here uses repeated source reports or equal-and-opposite amounts within reporting families, not regression performance.','',
            '| County | State | Source year | Contractor | Positive nominal dollars | Net nominal dollars | Source records | Unique raw records |',
            '|---|---|---:|---|---:|---:|---:|---:|']
        labels=t.drop_duplicates('ccg').set_index('ccg')
        for r in cf.head(10).itertuples():
            text.append(f'| {labels.loc[r.ccg,"county"]} | {labels.loc[r.ccg,"state"]} | {r.source_year} | {r.contractor} | {r.positive_nominal:,.0f} | {r.net_nominal:,.0f} | {r.records} | {r.unique_raw_records} |')
        text += ['', 'Source: early_report_candidates.csv, early_report_candidate_records.csv, and family_target_links.csv. '
            'The table is ordered by family positive dollars, not by coefficient influence.','']
        b=cr.loc[cr.ccg.eq(42011)&cr.source_year.eq(1979)&cr.report_no.eq('0937')].sort_values('record_number')
        if len(b):
            big=b.loc[b.dollars.gt(0),'dollars'].max()*1000
            bp=b.dollars.clip(lower=0).sum()*1000;bn=b.dollars.sum()*1000
            text += ['## Table 3. Berks County, Pennsylvania: report 0937 in 1979','',
                f'This family contains {len(b)} records, including positive entries of ${big:,.0f}. '
                f'The positive-only total is ${bp:,.0f}; the signed total is ${bn:,.0f}. '
                'Exact SHA256 repetitions establish repeated original source records, not duplication introduced by the county merge. '
                'All entries carry correction code 1. The approved cancellation matcher requires code 2 on the reversing leg and identical contract/action fields; '
                'this family therefore falls outside that matcher. Different contract spellings remain visible below. '
                'The economically correct retained amount has not been established.','',
                '| Original record | Contract as recorded | Action date code | Correction code | Nominal dollars | Raw SHA256 prefix |',
                '|---:|---|---|---|---:|---|']
            for r in b.itertuples():
                text.append(f'| {r.record_number} | `{r.contract_no}` | {r.action_date} | {r.correction_code} | {r.dollars*1000:,.0f} | `{r.raw_sha256[:16]}` |')
            br=t.loc[t.ccg.eq(42011)&t.year.eq(1983)].iloc[0]
            text += ['',f'In target year {int(br.year)}, the source year 1979 is the baseline of the spending difference. '
                f'The cached raw spending treatment is {br.x:,.6f} real dollars per person; its capped value is {br.xc:,.6f}. '
                'The same records enter national category spending, so correcting them would require rebuilding the affected instrument values as well as local spending. '
                'No correction or second-stage effect of correcting this family is estimated here.','',
                'Sources: berks_1979_report_0937.csv (full raw record fingerprints and bytes), target_inventory.csv, '
                'and producer 103_audit_contract_report_corrections.py, match() cancellation conditions.','']
        prior=pd.read_csv(BASE.parent/'correction_audit_reversals.csv',dtype={'fiscal_file':str,'report_no':str})
        nl=cr.loc[cr.ccg.eq(9011)&((cr.source_year.eq(1982)&cr.report_no.eq('2399'))|(cr.source_year.eq(1983)&cr.report_no.eq('1820')))].copy()
        if len(nl):
            nl['fiscal_file']=nl.source_year.astype(str)
            nl=nl.merge(prior[['fiscal_file','record_number','n_candidates','match_status']],
                on=['fiscal_file','record_number'],how='left',validate='one_to_one')
            nl.to_csv(OUT/'new_london_report_triplets.csv',index=False)
            text += ['## Table 4. New London County, Connecticut: unresolved report triplets','',
                'The saved cancellation audit marks the code-2 entries below ambiguous_original, with two candidate positive entries each. '
                'The positive-only construction retains both positives. Pairing the negative with either equal positive produces the same family dollar total; '
                'the matcher requires unique record identity even when the aggregate dollar result is invariant to that choice. '
                'This is a concrete administrative-report lead, not evidence that large ship awards in general should be capped.','',
                '| Source year | Report | Record | Correction code | Nominal dollars | Saved reversal status | Candidates |',
                '|---:|---|---:|---|---:|---|---:|']
            for r in nl.sort_values(['source_year','record_number']).itertuples():
                code=r.correction_code if pd.notna(r.correction_code) else '(blank)'
                status=r.match_status if pd.notna(r.match_status) else 'not a reversing entry'
                count=str(int(r.n_candidates)) if pd.notna(r.n_candidates) else ''
                text.append(f'| {r.source_year} | {r.report_no} | {r.record_number} | {code} | {r.dollars*1000:,.0f} | {status} | {count} |')
            text += ['', 'Sources: new_london_report_triplets.csv and Results/instrument_normalization/correction_audit_reversals.csv. '
                'The raw fields are verified in early_report_candidate_records.csv. No cleaning or regression has been applied to these families by this audit.','']
    text += ['## Reproduction','',
        'Run from the code checkout using 4_Macro_version/.venv/bin/python and '
        '4_Macro_version/step_1/code/pipeline/109_h4_target_source_audit.py. '
        'Use --trace, then --records, then --early-reports, then --report. '
        'For a memo-only edit use --report; the other steps need not run. '
        'Shell logs belong in batch_logs/macro/.','']
    MEMO.write_text('\n'.join(text))
    assert MEMO.read_text()=='\n'.join(text)
    for path in OUT.glob('*.csv'):
        pd.read_csv(path,low_memory=False)
    for path in OUT.glob('*.parquet'):
        pd.read_parquet(path)
    assert sha(ANALYSIS)==json.loads((OUT/'trace_check.json').read_text())['analysis_sha256']
    print(MEMO,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trace',action='store_true');p.add_argument('--records',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--early-reports',action='store_true')
    args=p.parse_args()
    if args.trace: trace()
    if args.records: records()
    if args.early_reports: early_reports()
    if args.report: report()
    if not any(vars(args).values()): p.error('Select --trace, --records, --early-reports, or --report')
