"""Cross exposure, category detail, action floors, action rules, and coverage.

All 324 instrument designs use the same 3,058 counties and 1982--2016 years.
Each has equal-county and fixed-1975 population-weighted estimates. The outcome
is county h=2 midpoint employment growth. County treatment and national shift
are mean spending in t-1,t minus spending in t-2, divided by t-2 population
for the county treatment. Division-year effects and county clusters are fixed.

Floors refer to absolute nominal source-action dollars, before category-code
harmonization or deflation. Positive and net rules apply consistently except
the explicitly named net-shift-only diagnostic, which keeps positive exposure
and treatment. Categories with nonpositive national exposure cannot be used;
their count is reported. Fixed exposure uses 1966--1974; rolling exposure ends
at t-7. Construction means the source/category Y-prefix screen, not military
personnel compensation. No clipping or category-rank threshold is imposed.

Uses the source readers and estimators of producers 98 and 99. Saved compatible
coefficient cells are reused. Outputs and checkpoints remain separate from the
paper baseline and the window-by-category grid.
"""

# mpgr-run-class: cache_rebuild
# mpgr-rebuild-reason: required_key_absent
# mpgr-cache: parsed actions plus county_panel_pipeline.parquet
# mpgr-new-estimates: 608
# mpgr-batch-approved: explicit
# mpgr-variation-scope: county|h2|full-factorial|same-sample|no-clip

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

PIPELINE = Path(__file__).resolve().parent
sys.path[:0] = [str(PIPELINE.parent / 'py'), str(PIPELINE), str(PIPELINE / 'lib')]
import contracts
import dcads_common as dc
import harmonize
import locked
import store

ROOT = PIPELINE.parents[2]
CACHE = dc.MACRO_DATA_ROOT / 'temp' / 'full_instrument_design_grid'
OUTDIR = dc.MACRO_RESULTS_ROOT / 'instrument_normalization'
OUT = OUTDIR / 'h2_full_instrument_design_grid.csv'
DIAG = OUTDIR / 'h2_full_instrument_design_grid_diagnostics.csv'
CHECKPOINT = CACHE / 'coefficient_checkpoint.csv'
YEARS = list(range(1966, 2019))
TARGETS = list(range(1982, 2017))
FLOORS = (0, 10000, 25000)
COVERAGES = ('exclude_construction', 'include_construction')
DESIGNS = [('rolling', s) for s in (3, 5, 7, 9, 10)] + [('fixed_1966_1974', 9)]
ACTIONS = {'positive': ('positive', 'positive', 'positive'),
           'net_shift_only': ('positive', 'net', 'positive'),
           'net_matched': ('net', 'net', 'net')}
KEYS = ['coverage', 'nominal_floor', 'category_digits', 'exposure_mode',
        'span', 'action_design', 'weight']


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def path_for(coverage, rule, floor, year):
    return CACHE / coverage / f'rule={rule}_floor={floor}' / f'fy={year}.parquet'


def county_path(coverage, rule, floor):
    return CACHE / coverage / f'county_{rule}_{floor}.parquet'


def source_signature(p98):
    paths = [dc.OUT_DIR / f'fy{y}.parquet' if y < dc.SEAM_YEAR
             else dc.USASPENDING_OUT_DIR / f'fy{y}.parquet' for y in YEARS]
    paths += sorted((PIPELINE.parent / 'reference').glob('*.csv'))
    paths += [Path(contracts.__file__), Path(harmonize.__file__), Path(__file__)]
    records = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def build_cache(units, p98):
    """Construct missing source-floor and coverage cells; preserve floor-zero inputs."""
    signature = source_signature(p98)
    manifest_path = CACHE / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['source_signature'] != signature:
            raise SystemExit('FAIL: full-grid cache source signature changed')
        for cover in COVERAGES:
            for rule in ('positive', 'net'):
                for floor in FLOORS:
                    if not county_path(cover, rule, floor).exists():
                        raise SystemExit('FAIL: incomplete county cache')
                    if any(not path_for(cover, rule, floor, y).exists() for y in YEARS):
                        raise SystemExit('FAIL: incomplete category cache')
        return signature
    CACHE.mkdir(parents=True, exist_ok=True)
    fips_map, _, _ = dc.load_ccg_map()
    parts = {(c, r, f): [] for c in COVERAGES for r in ('positive', 'net') for f in FLOORS}
    for year in YEARS:
        dcat, ccat, cat = contracts.category_actions(
            year, fips_map, drop_construction=False, harmonize=False)
        for cover in COVERAGES:
            drop = cover == 'exclude_construction'
            dall, call = contracts.county_actions(year, fips_map, drop_construction=drop)
            for rule in ('positive', 'net'):
                for floor in FLOORS:
                    dest = path_for(cover, rule, floor, year)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if drop and floor == 0:
                        # These source-compatible category aggregates need no rebuild.
                        pd.read_parquet(p98.cache_path(rule, 0, year)).to_parquet(dest, index=False)
                        continue
                    keep = np.abs(dall) >= floor
                    if rule == 'positive':
                        keep &= dall > 0
                    county = pd.DataFrame({'ccg': call[keep], 'dollars': dall[keep] * dc.deflate(year)})
                    parts[(cover, rule, floor)].append(
                        county.groupby('ccg', as_index=False, sort=False).dollars.sum().assign(year=year))
                    keep = np.abs(dcat) >= floor
                    if rule == 'positive':
                        keep &= dcat > 0
                    if drop:
                        keep &= ~pd.Series(cat).str.startswith('Y').to_numpy()
                    d, c, k = harmonize.harmonize_arrays(year, dcat[keep], ccat[keep], cat[keep])
                    if drop:
                        keep_y = ~pd.Series(k).str.startswith('Y').to_numpy()
                        d, c, k = d[keep_y], c[keep_y], k[keep_y]
                    frame = pd.DataFrame({'ccg': c, 'cat': k, 'dollars': d * dc.deflate(year)})
                    frame.groupby(['ccg', 'cat'], as_index=False, sort=False).dollars.sum().to_parquet(dest, index=False)
        if year % 10 == 0:
            print(f'   source-floor cache through {year}', flush=True)
    for (cover, rule, floor), frames in parts.items():
        if cover == 'exclude_construction' and floor == 0:
            wide = store.load_con(rule).reindex(index=units, columns=YEARS)
            frame = wide.rename_axis(index='ccg', columns='year').stack().rename('dollars').reset_index()
        else:
            frame = pd.concat(frames, ignore_index=True)
        frame.to_parquet(county_path(cover, rule, floor), index=False)
    manifest_path.write_text(json.dumps({'source_signature': signature,
        'floor_order': 'absolute nominal source action before harmonization and deflation',
        'first_year': 1966, 'last_year': 2018}, indent=2) + '\n')
    return signature


def load_sparse(cover, rule, floor, digits, units):
    """Load a grouped annual sparse panel without constructing a dense tensor."""
    pattern = str(path_for(cover, rule, floor, '*'))
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        frame = con.execute('''SELECT ccg, substr(cat,1,?) AS category,
            CAST(regexp_extract(filename, 'fy=([0-9]+)', 1) AS INTEGER) AS year,
            sum(dollars) AS dollars FROM read_parquet(?, filename=true, hive_partitioning=false)
            GROUP BY ccg, category, year''', [digits, pattern]).df()
    categories = sorted(frame.category.unique())
    group_position = {g: j for j, g in enumerate(categories)}
    national = (frame.groupby(['category', 'year'], sort=False).dollars.sum()
                .unstack('year').reindex(index=categories, columns=YEARS).fillna(0).to_numpy())
    unit_position = {c: i for i, c in enumerate(units)}
    frame['i'] = frame.ccg.map(unit_position)
    frame['j'] = frame.category.map(group_position)
    matrices = {}
    for year, f in frame.loc[frame.i.notna()].groupby('year', sort=False):
        matrices[int(year)] = sparse.csr_matrix((f.dollars.to_numpy(),
            (f.i.to_numpy(np.int64), f.j.to_numpy(np.int64))), shape=(len(units), len(categories)))
    return categories, national, matrices


def instruments(exposure, national_shift, population, mode, span):
    """Allocate matched national shifts through predetermined county market shares."""
    cats, national, matrices = exposure
    shift_cats, shift_nat, _ = national_shift
    shift_nat = pd.DataFrame(shift_nat, index=shift_cats).reindex(cats).fillna(0).to_numpy()
    result = np.zeros((population.shape[0], len(TARGETS)))
    diagnostics = []
    for j, t in enumerate(TARGETS):
        lo, hi = (1966, 1974) if mode.startswith('fixed') else (t - 7 - span + 1, t - 7)
        if lo < 1966 or hi > t - 7 or hi-lo+1 != span:
            raise SystemExit('FAIL: exposure-window or gap mismatch')
        indices = [y-1966 for y in range(lo, hi+1)]
        denominator = national[:, indices].mean(axis=1)
        live = denominator > 0
        change = .5 * (shift_nat[:, t-1966] + shift_nat[:, t-1-1966]) - shift_nat[:, t-2-1966]
        exposure_matrix = sum((matrices[y] for y in range(lo, hi+1))) / span
        scaled = np.divide(change, denominator, out=np.zeros_like(change), where=live)
        values = np.asarray(exposure_matrix @ scaled).ravel()
        pbase = population[t-2].to_numpy(np.float64)
        result[:, j] = values / pbase
        identity = denominator[live] * scaled[live] - change[live]
        relative_gap = np.max(np.abs(identity), initial=0) / max(np.max(np.abs(change[live]), initial=0), 1)
        if relative_gap > 1e-12:
            raise SystemExit('FAIL: national-dollar allocation identity')
        diagnostics.append({'year': t, 'exposure_start': lo, 'exposure_end': hi,
            'actual_gap_before_base': t-2-hi, 'live_groups': int(live.sum()),
            'nonpositive_exposure_groups': int((~live).sum()),
            'negative_county_exposure_cells': int((exposure_matrix.data < 0).sum()),
            'allocation_identity_relative_gap': relative_gap})
    if not np.isfinite(result).all():
        raise SystemExit('FAIL: nonfinite instrument')
    return result, diagnostics


def import_saved_rows(p98):
    """Read compatible no-floor cells without re-estimating them."""
    d = pd.read_csv(OUTDIR / 'h2_all_category_window_by_category_grid.csv')
    d['action_design'] = 'positive'
    n = pd.read_csv(p98.OUT)
    n = n[n.variant.isin(['allcat_net_national_shift', 'allcat_net_matched'])].copy()
    n['action_design'] = n.variant.map({'allcat_net_national_shift': 'net_shift_only', 'allcat_net_matched': 'net_matched'})
    n['category_digits'], n['exposure_mode'] = 4, 'rolling'
    d = pd.concat([d, n], ignore_index=True)
    d['coverage'] = 'exclude_construction'
    d['estimate_source'] = 'saved_compatible_no_floor'
    if len(d) != 40 or d.duplicated(KEYS).any():
        raise SystemExit('FAIL: expected 40 unique compatible saved estimates')
    if not d.n.eq(3058*35).all():
        raise SystemExit('FAIL: saved estimates use a different sample')
    return d


def main():
    started = time.time()
    p98 = module('p98_full', PIPELINE / '98_h2_all_category_market_share_design_grid.py')
    t28 = module('t28_full', ROOT / 'step_2/code/py/28_table_d1.py')
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100000)
    county_panel = store.load_county_panel()
    units, pivots = store.load_pivots(county_panel, cols=('P',))
    if len(units) != 3058:
        raise SystemExit('FAIL: county universe differs from requested grid')
    signature = build_cache(units, p98)
    saved = pd.read_csv(CHECKPOINT) if CHECKPOINT.exists() else import_saved_rows(p98)
    if 'source_signature' in saved and not saved.source_signature.eq(signature).all():
        raise SystemExit('FAIL: coefficient checkpoint source mismatch')
    saved['source_signature'] = signature
    rows = saved.to_dict('records')
    estimates = {tuple(r[k] for k in KEYS): r for r in rows}
    diagnostics, computed = [], 0
    for cover in COVERAGES:
        for floor in FLOORS:
            panels = {}
            for rule in ('positive', 'net'):
                f = pd.read_parquet(county_path(cover, rule, floor))
                wide = f.pivot(index='ccg', columns='year', values='dollars').reindex(index=units, columns=YEARS).fillna(0)
                panels[rule] = p98.make_panel(wide, county_panel, units)
            for digits in (4, 3, 2):
                groups = {rule: load_sparse(cover, rule, floor, digits, units)
                          for rule in ('positive', 'net')}
                for mode, span in DESIGNS:
                    for action, (erule, srule, trule) in ACTIONS.items():
                        meta = {'coverage': cover, 'nominal_floor': floor,
                            'category_digits': digits, 'exposure_mode': mode, 'span': span,
                            'action_design': action, 'exposure_action_rule': erule,
                            'national_shift_action_rule': srule, 'treatment_action_rule': trule,
                            'tail_rule': 'none', 'horizon': 2, 'fixed_effects': 'division-year',
                            'cluster': 'county', 'minimum_gap_before_base': 5,
                            'category_universe': 'all harmonized categories in selected coverage',
                            'threshold_order': 'source_action_before_harmonization',
                            'source_signature': signature}
                        prefix = tuple(meta[k] for k in KEYS[:-1])
                        weights = ('equal county', 'fixed 1975 population')
                        present = [prefix + (w,) in estimates for w in weights]
                        if any(present) and not all(present):
                            raise SystemExit('FAIL: partial weight pair in checkpoint')
                        # The first input pair also verifies the sparse implementation
                        # directly against producer 98 without repeating a regression.
                        tieout = (cover == 'exclude_construction' and floor == 0
                                  and digits == 4 and mode == 'rolling' and span == 5
                                  and action == 'positive')
                        if all(present) and not tieout:
                            continue
                        z, diag = instruments(groups[erule], groups[srule], pivots['P'], mode, span)
                        if tieout:
                            ref, _ = p98.build_instrument('positive', 0, 'positive', 0,
                                4, units, pivots['P'], 5)
                            np.testing.assert_allclose(z, ref, rtol=1e-10, atol=1e-8)
                            print('   sparse instrument matches producer 98', flush=True)
                            if all(present):
                                continue
                        for row in p98.estimate(panels[trule], z, units, t28):
                            r = {**meta, **row, 'estimate_source': 'estimated_full_grid'}
                            estimates[prefix + (r['weight'],)] = r
                            rows.append(r)
                            computed += 1
                        diagnostics.extend([{**meta, **r} for r in diag])
                pd.DataFrame(rows).to_csv(CHECKPOINT, index=False)
                print(f'   {cover} floor={floor} digits={digits}: {len(rows)}/648 rows', flush=True)
                del groups
    results = pd.DataFrame(rows)
    if len(results) != 648 or results.duplicated(KEYS).any():
        raise SystemExit('FAIL: incomplete or duplicate full factorial grid')
    fields = ['first_stage', 'first_stage_se', 'first_stage_F', 'second_stage', 'second_stage_se', 'second_stage_t']
    if not np.isfinite(results[fields].to_numpy()).all():
        raise SystemExit('FAIL: nonfinite regression estimates')
    if not results.n.eq(3058*35).all() or not results.n_counties.eq(3058).all():
        raise SystemExit('FAIL: grid sample differs across designs')
    results['second_stage_x1e5'] = results.second_stage * 1e5
    results = results.sort_values(KEYS).reset_index(drop=True)
    results.to_csv(OUT, index=False)
    if diagnostics:
        pd.DataFrame(diagnostics).to_csv(DIAG, index=False)
    print(f'   complete: {len(results)} estimates, {computed} newly estimated; {time.time()-started:.1f}s', flush=True)
    print(f'   output: {OUT}', flush=True)


if __name__ == '__main__':
    main()
