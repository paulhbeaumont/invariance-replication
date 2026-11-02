"""Executable aggregation specification, weight construction and publication gate."""
import numpy as np
import pandas as pd
from baseline_io import APP,DATA,read_json,require,sha

SPEC=APP/'config/paper_aggregation.json'
FIXTURE=APP/'tests/fixtures/aggregation_weights.json'
MAN=APP/'results/manifests/aggregation_aligned_h24.json'
CHECK=APP/'results/manifests/aggregation_aligned_check.json'
BOUND=['code/lib/aggregation_design.py','code/estimate/aggregation_standard_population.py',
       'code/estimate/aggregation_iv.py','code/estimate/aggregation_delta_inference.py',
       'code/tests/test_aggregation_design.py','code/tests/verify_aggregation_design.py']


def specification():
    s=read_json(SPEC)
    expected=dict(id='standard_employment_share_aggregate_population1975',status='approved',outcome='standard',winsorized=False,
        horizons=[2,4],rungs=['czone','cea','ea','state'],last_year=2016,n_counties=3058,population_year=1975,
        county_weight='lagged_employment_share * aggregate_population1975',aggregate_weight='aggregate_population1975',
        construction_share='E_b / aggregate_E_b',county_fe=['ccg','year'],aggregate_fe=['unit','year'],cluster='unit',
        county_equation={'endogenous':['x'],'instruments':['z']},
        augmented_equation={'endogenous':['x','x_agg'],'instruments':['z','z_agg']},
        aggregate_equation={'endogenous':['x_agg'],'instruments':['z_agg']})
    for k,v in expected.items():require(s.get(k)==v,'Unapproved aggregation design: '+k)
    require(sha(FIXTURE)==s['fixture_sha256'],'Approved fixture changed')
    require(sha(DATA/'analysis/county_population_1975.parquet')==s['population_sha256'],'Population input changed')
    return s


def weights(c,pop):
    require(not c.duplicated(['ccg','year']).any(),'Duplicate county-year')
    mapping=c[['ccg','unit']].drop_duplicates()
    require(not mapping.ccg.duplicated().any(),'County crosses units')
    require(np.isfinite(c.E_b).all() and c.E_b.gt(0).all(),'Invalid lagged employment')
    mapping['population']=mapping.ccg.map(pop)
    require(np.isfinite(mapping.population).all() and mapping.population.gt(0).all(),'Missing or invalid population')
    P=mapping.groupby('unit').population.sum()
    d=c.copy()
    d['construction_share']=d.E_b/d.groupby(['unit','year']).E_b.transform('sum')
    d['aggregate_population1975']=d.unit.map(P)
    d['county_regression_weight']=d.construction_share*d.aggregate_population1975
    validate_weights(d,pop)
    return d,P


def validate_weights(d,pop):
    mapping=d[['ccg','unit']].drop_duplicates()
    P=mapping.assign(population=mapping.ccg.map(pop)).groupby('unit').population.sum()
    # The within-unit ratio check rejects county population even when sums agree.
    expected_share=d.E_b/d.groupby(['unit','year']).E_b.transform('sum')
    require(np.allclose(d.construction_share,expected_share,rtol=1e-12,atol=1e-14),'Wrong construction share')
    require(np.allclose(d.aggregate_population1975,d.unit.map(P),rtol=1e-12),'Wrong aggregate population')
    require(np.allclose(d.county_regression_weight/d.unit.map(P),expected_share,rtol=1e-12,atol=1e-14),'Wrong county weight ratio')
    totals=d.groupby(['unit','year']).county_regression_weight.sum()
    require(np.allclose(totals,totals.index.get_level_values('unit').map(P),rtol=1e-12),'County weights do not sum to aggregate weight')


def prepare(c,a,pop):
    d,P=weights(c,pop);a=a.copy()
    require(not a.duplicated(['unit','year']).any(),'Duplicate aggregate-year')
    totals=d.groupby(['unit','year'])[['E','E_b']].sum();keys=pd.MultiIndex.from_frame(a[['unit','year']])
    require(set(keys)==set(totals.index),'Aggregate support mismatch')
    require(np.allclose(a[['E','E_b']],totals.loc[keys],rtol=1e-10,atol=1e-8),'Employment endpoints do not add up')
    for col,source in [('x_agg','x'),('z_agg','z')]:
        v=(d[source]*d.construction_share).groupby([d.unit,d.year]).sum()
        require(np.allclose(a[col],v.loc[keys],rtol=1e-10,atol=1e-10),'Wrong aggregate '+col)
        require(np.allclose(d[col],pd.MultiIndex.from_frame(d[['unit','year']]).map(v),rtol=1e-10,atol=1e-10),'Wrong county aggregate '+col)
    d['y']=(d.E-d.E_b)/d.E_b;a['y']=(a.E-a.E_b)/a.E_b
    d['weight']=d.county_regression_weight
    a['aggregate_regression_weight']=a.unit.map(P);a['weight']=a.aggregate_regression_weight
    require(np.isfinite(d[['y','weight']]).all().all() and np.isfinite(a[['y','weight']]).all().all(),'Invalid regression input')
    return d,a


def binding():
    return {name:sha(APP/name) for name in BOUND}


def load_results():
    s=specification();path=APP/s['coefficients'];m=read_json(MAN);v=read_json(CHECK)
    require(m['status']=='PASS' and v['status']=='PASS','Aggregation not verified')
    require(m['spec_sha256']==sha(SPEC) and m['code_hashes']==binding(),'Stale aggregation specification or code')
    require(m['coefficients_sha256']==sha(path) and v['manifest_sha256']==sha(MAN),'Stale aggregation results or verification')
    require(v['specification_checks']=='PASS' and v['numerical_checks']=='PASS','Missing specification or numerical checks')
    return pd.read_csv(path)
