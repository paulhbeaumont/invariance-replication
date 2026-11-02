"""Winsorize observed county treatment, outcome, or both in h=2 2SLS.

The reference uses every four-character category surviving the construction
screen, rolling five-year exposure ending at t-7, positive spending, no action
floor, no instrument clipping, and the common 1982--2016 county sample.
Cutoffs are unweighted within-year 1st and 99th percentiles (linear
interpolation), computed before fixed-effect absorption. The same cutoffs and
unchanged instrument are used under equal and fixed-1975 population weights.
No observations or counties are removed. Raw first stages and untreated
coefficient rows are read from the full design grid.

The --joint branch loads the saved county analysis cache and caps x, y, and z
separately within year using either percentiles or median +/- k times IQR.
IQR cutoffs follow the project's winsor4 percentile convention. Weights,
fixed effects, identifiers, and the estimation sample are never capped.
The --pooled branch uses unconditional 1st/99th-percentile cutoffs for each
of x, y, and z and includes the saved within-year comparison rows.
"""

# mpgr-run-class: cache_augment
# mpgr-cache: county_panel_pipeline.parquet, contracts_ccg_year.parquet, all_category_market_share_grid
# mpgr-output-cache: county_xy_winsorization/analysis.parquet
# mpgr-new-estimates: 2 (--pooled); 8 (--joint); 6 (default)
# mpgr-batch-approved: explicit
# mpgr-variation-scope: county|h2|4-character|span5|positive|exclude-construction|winsor-x-y-both

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
sys.path[:0] = [str(PIPELINE.parent / 'py'), str(PIPELINE), str(PIPELINE / 'lib')]
import dcads_common as dc
import store

OUTDIR = dc.MACRO_RESULTS_ROOT / 'instrument_normalization'
OUT = OUTDIR / 'h2_county_xy_winsorization.csv'
CUTOFFS = OUTDIR / 'h2_county_xy_winsorization_cutoffs.csv'
CACHE = dc.MACRO_DATA_ROOT / 'temp' / 'county_xy_winsorization'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p98 = module('p98_xywinsor', PIPELINE / '98_h2_all_category_market_share_design_grid.py')
    t28 = module('t28_xywinsor', ROOT / 'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100000)

    saved = pd.read_csv(OUTDIR / 'h2_full_instrument_design_grid.csv')
    reference = saved.loc[saved.category_digits.eq(4) & saved.span.eq(5)
        & saved.exposure_mode.eq('rolling') & saved.nominal_floor.eq(0)
        & saved.coverage.eq('exclude_construction') & saved.action_design.eq('positive')]
    if len(reference) != 2 or reference.weight.nunique() != 2:
        raise SystemExit('FAIL: expected one untreated reference per weight')
    county = store.load_county_panel()
    units, pivots = store.load_pivots(county, cols=('P',))
    spending = store.load_con('positive').reindex(index=units, columns=range(1966, 2019))
    if spending.isna().any().any():
        raise SystemExit('FAIL: county spending cache is incomplete')
    panel = p98.make_panel(spending, county, units)
    zwide, _ = p98.build_instrument('positive', 0, 'positive', 0, 4, units, pivots['P'], 5)
    assert np.array_equal(panel.ccg.to_numpy(), np.repeat(units, 35))
    assert np.array_equal(panel.year.to_numpy(), np.tile(p98.TARGET_YEARS, len(units)))
    panel['z'] = zwide.ravel()
    if (len(panel) != 107030 or panel.ccg.nunique() != 3058
            or not reference.n.eq(len(panel)).all()):
        raise SystemExit('FAIL: sample does not match the untreated grid')
    panel['fe_division_year'] = panel.groupby(['division', 'year'], sort=False).ngroup().astype('int32')
    panel['cluster'] = pd.factorize(panel.ccg)[0].astype('int32')
    cutoff_rows = []
    for var in ('x', 'y'):
        cut = panel.groupby('year', sort=False)[var].quantile([.01, .99], interpolation='linear').unstack()
        lo, hi = panel.year.map(cut[.01]), panel.year.map(cut[.99])
        panel[f'{var}_winsor'] = panel[var].clip(lower=lo, upper=hi)
        for year, group in panel.groupby('year', sort=False):
            a, b = cut.loc[year, .01], cut.loc[year, .99]
            cutoff_rows.append({'variable': var, 'year': year, 'p01': a, 'p99': b,
                'n_below': int((group[var] < a).sum()), 'n_above': int((group[var] > b).sum()),
                'n': len(group), 'cutoff_weight': 'unweighted', 'quantile_method': 'linear'})
    if panel[['x', 'y', 'z', 'x_winsor', 'y_winsor', 'pop1975']].isna().any().any():
        raise SystemExit('FAIL: missing analysis values')
    CACHE.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CACHE / 'analysis.parquet', index=False)
    pd.DataFrame(cutoff_rows).to_csv(CUTOFFS, index=False)

    z = panel.z.to_numpy(np.float64)
    fe = panel.fe_division_year.to_numpy(np.int64)
    cluster = panel.cluster.to_numpy(np.int64)
    nofe = np.zeros(len(panel), np.int64)
    fs_fields = ['first_stage', 'first_stage_se', 'first_stage_t', 'first_stage_F']
    rows = []
    for weight in ('equal county', 'fixed 1975 population'):
        ref = reference.loc[reference.weight.eq(weight)].iloc[0]
        w = np.ones(len(panel)) if weight == 'equal county' else panel.pop1975.to_numpy(np.float64)
        meta = {'weight': weight, 'horizon': 2, 'category_digits': 4, 'span': 5,
            'action_design': 'positive', 'coverage': 'exclude_construction',
            'fixed_effects': 'division-year', 'cluster': 'county',
            'n': len(panel), 'n_counties': len(units), 'sample_start': 1982, 'sample_end': 2016,
            'instrument_tail_rule': 'none', 'cutoff_weight': 'unweighted', 'cutoff_scope': 'within year'}
        rows.append({**meta, 'variant': 'untreated', **{f: ref[f] for f in fs_fields},
            'second_stage': ref.second_stage, 'second_stage_se': ref.second_stage_se,
            'second_stage_t': ref.second_stage_t, 'n_x_changed': 0, 'n_y_changed': 0,
            'estimate_source': 'saved_full_grid'})
        xwin = panel.x_winsor.to_numpy(np.float64)
        fs = t28.fit2sls(xwin, z, z, w, fe, nofe, cluster, singleton=False, want_F=False)
        fs_win = dict(first_stage=fs['beta'], first_stage_se=fs['se'],
                      first_stage_t=fs['beta']/fs['se'], first_stage_F=(fs['beta']/fs['se'])**2)
        for name, xv, yv in [('x_only', 'x_winsor', 'y'), ('y_only', 'x', 'y_winsor'),
                             ('both', 'x_winsor', 'y_winsor')]:
            fit = t28.fit2sls(panel[yv].to_numpy(np.float64), panel[xv].to_numpy(np.float64),
                z, w, fe, nofe, cluster, singleton=False, want_F=False)
            if fit['n_dropped'] != 0 or fit['n'] != len(panel):
                raise SystemExit('FAIL: estimation removed observations')
            rows.append({**meta, 'variant': name,
                **(fs_win if xv == 'x_winsor' else {f: ref[f] for f in fs_fields}),
                'second_stage': fit['beta'], 'second_stage_se': fit['se'],
                'second_stage_t': fit['beta']/fit['se'],
                'n_x_changed': int((panel[xv] != panel.x).sum()),
                'n_y_changed': int((panel[yv] != panel.y).sum()),
                'estimate_source': 'estimated'})
    results = pd.DataFrame(rows)
    for field in ('second_stage', 'second_stage_se'):
        results[f'{field}_x1e5'] = results[field]*1e5
    if len(results) != 8 or not np.isfinite(results[['first_stage_F', 'second_stage_t']]).all().all():
        raise SystemExit('FAIL: incomplete estimates')
    results.to_csv(OUT, index=False)
    print(results[['variant', 'weight', 'first_stage_F', 'second_stage_x1e5',
        'second_stage_se_x1e5', 'second_stage_t', 'n_x_changed', 'n_y_changed']].to_string(index=False))
    print(f'Wrote {OUT}; six new 2SLS estimates and two reused untreated rows.')


def joint_main(pooled=False):
    """Estimate the requested joint winsorization rules from saved county data."""
    panel = pd.read_parquet(CACHE / 'analysis.parquet')
    required = ['x', 'y', 'z', 'pop1975', 'fe_division_year', 'cluster', 'ccg', 'year']
    if (not set(required).issubset(panel.columns) or len(panel) != 107030
            or panel.ccg.nunique() != 3058 or panel.duplicated(['ccg', 'year']).any()
            or not np.isfinite(panel[required].to_numpy()).all()):
        raise SystemExit('FAIL: invalid county-level analysis cache')
    reference = pd.read_csv(OUT)
    reference = reference.loc[reference.variant.eq('untreated')].copy()
    if len(reference) != 2:
        raise SystemExit('FAIL: missing untreated coefficient rows')
    t28 = module('t28_jointwinsor', ROOT / 'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100000)
    fe, cluster = (panel[v].to_numpy(np.int64) for v in ('fe_division_year', 'cluster'))
    nofe = np.zeros(len(panel), np.int64)
    rows, cutoff_rows = reference.to_dict('records'), []
    for r in rows:
        r.update(n_z_changed=0, joint_rule='none', winsorized_variables='none')
    if pooled:
        comparison = pd.read_csv(OUTDIR / 'h2_county_xyz_winsorization.csv')
        comparison = comparison.loc[comparison.variant.eq('joint_percentile_1_99')]
        if len(comparison) != 2:
            raise SystemExit('FAIL: missing within-year comparison')
        rows.extend(comparison.to_dict('records'))
    groups = {'all_years': np.arange(len(panel))} if pooled else panel.groupby('year', sort=False).indices
    rules = [('percentile_1_99', None)] if pooled else [('percentile_1_99', None), ('iqr4', 4), ('iqr5', 5), ('iqr10', 10)]
    for rule, k in rules:
        arrays = {}
        counts = {}
        for var in ('x', 'y', 'z'):
            raw = panel[var].to_numpy(np.float64)
            capped = raw.copy()
            for year, idx in groups.items():
                values = raw[idx]
                if k is None:
                    lo, hi = np.quantile(values, [.01, .99], method='linear')
                else:
                    lo, hi = t28.w22.iq_cutoffs(values, k)
                if not np.isfinite([lo, hi]).all() or hi <= lo:
                    raise SystemExit(f'FAIL: degenerate {rule} cutoffs for {var}, {year}')
                capped[idx] = np.clip(values, lo, hi)
                cutoff_rows.append({'rule': rule, 'variable': var, 'year': year,
                    'lower': lo, 'upper': hi, 'n': len(idx),
                    'n_below': int((values < lo).sum()), 'n_above': int((values > hi).sum()),
                    'definition': 'p01,p99' if k is None else f'median +/- {k}*(p75-p25)',
                    'quantile_method': 'linear' if k is None else 'Stata_pctile',
                    'cutoff_weight': 'unweighted', 'cutoff_scope': 'pooled' if pooled else 'within year'})
            arrays[var] = capped
            counts[f'n_{var}_changed'] = int((raw != capped).sum())
            panel[f'{var}_{rule}'] = capped
            if not pooled and k is None and var in ('x', 'y'):
                np.testing.assert_array_equal(capped, panel[f'{var}_winsor'].to_numpy())
        x, y, z = (arrays[v] for v in ('x', 'y', 'z'))
        for weight in ('equal county', 'fixed 1975 population'):
            w = np.ones(len(panel)) if weight == 'equal county' else panel.pop1975.to_numpy(np.float64)
            ref = reference.loc[reference.weight.eq(weight)].iloc[0].to_dict()
            first = t28.fit2sls(x, z, z, w, fe, nofe, cluster, singleton=False, want_F=False)
            fit = t28.fit2sls(y, x, z, w, fe, nofe, cluster, singleton=False)
            if fit['n_dropped'] or first['n_dropped'] or fit['n'] != len(panel):
                raise SystemExit('FAIL: estimation changed sample')
            fst = first['beta']/first['se']
            np.testing.assert_allclose(fit['F'], fst**2, rtol=1e-10)
            rows.append({**ref, 'variant': f'joint_pooled_{rule}' if pooled else f'joint_{rule}', 'joint_rule': rule,
                'cutoff_scope': 'pooled' if pooled else 'within year',
                'winsorized_variables': 'x,y,z', 'instrument_tail_rule': rule,
                **counts, 'first_stage': first['beta'], 'first_stage_se': first['se'],
                'first_stage_t': fst, 'first_stage_F': fst**2,
                'second_stage': fit['beta'], 'second_stage_se': fit['se'],
                'second_stage_t': fit['beta']/fit['se'],
                'second_stage_x1e5': fit['beta']*1e5, 'second_stage_se_x1e5': fit['se']*1e5,
                'estimate_source': 'estimated_joint'})
    results = pd.DataFrame(rows)
    if len(results) != (6 if pooled else 10) or not np.isfinite(results[['first_stage_F', 'second_stage_t']]).all().all():
        raise SystemExit('FAIL: incomplete joint winsorization estimates')
    cutoff_frame = pd.DataFrame(cutoff_rows)
    counts_frame = cutoff_frame.groupby(['rule', 'variable'], sort=False)[['n_below', 'n_above', 'n']].sum().reset_index()
    counts_frame['pct_changed'] = 100*(counts_frame.n_below+counts_frame.n_above)/counts_frame.n
    stem = 'h2_county_xyz_winsorization_pooled' if pooled else 'h2_county_xyz_winsorization'
    results.to_csv(OUTDIR / f'{stem}.csv', index=False)
    cutoff_frame.to_csv(OUTDIR / f'{stem}_cutoffs.csv', index=False)
    counts_frame.to_csv(OUTDIR / f'{stem}_counts.csv', index=False)
    panel.to_parquet(CACHE / ('analysis_pooled.parquet' if pooled else 'analysis_joint.parquet'), index=False)
    print(results[['variant', 'weight', 'first_stage_F', 'second_stage_x1e5',
        'second_stage_se_x1e5', 'second_stage_t']].to_string(index=False))
    print(counts_frame.to_string(index=False))
    print(f'{2*len(rules)} new 2SLS estimates; {len(results)-2*len(rules)} saved rows reused. No observations removed.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint', action='store_true')
    parser.add_argument('--pooled', action='store_true')
    args = parser.parse_args()
    joint_main(pooled=args.pooled) if args.joint or args.pooled else main()
