# Disaggregation, horizons 2, 3 and 4

## Approved scope

The same 3,058 county groups appear at county, county × NAICS2 and county × allocated NAICS4 levels. Horizon-specific estimation windows are 1980–2016 for h=2, 1981–2016 for h=3, and 1982–2016 for h=4. The h=2 county MPGR coefficient is read from the verified baseline, not overwritten or re-estimated as a new specification.

For each h, treatment is `[mean(D_t,...,D_t-h+1) − D_t-h]/P_t-h`, with pooled p1/p99 winsorization at county level. Five-year exposure ends at `t-h-5`. National shift, county support, equal county-year weighting at the baseline, and the source procurement conventions match the locked county design. Treatment and instrument are repeated across industry cells without modification.

## Protocol answers

1. Baseline unit: consistent county group. Cell: industry, at the existing two-digit and allocated four-digit rungs.
2. All cells sum to the independently frozen county employment totals. The audit covers 1978–2016, the endpoint years needed by the three horizons, and is saved in `results/manifests/disaggregation_cell_source_audit.csv`.
3. The endogenous regressor and excluded instrument vary at county-year level, with the same h used for spending, population date, outcome and exposure timing.
4. Their county-year subscript is identical in every disaggregation regression. Neither variable is allocated to industry.
5. County and year fixed effects are used at every rung. No county-year or industry-year effects are added.
6. MPGR cells receive midpoint employment shares. The log-growth rows are explicitly unweighted-cell benchmarks, not the invariance answer. Growth rates are never winsorized.
7. The MPGR sample is identical across rungs within each horizon. Source cells are not removed or recleaned. Zero-both-endpoint cells have zero midpoint weight. Log growth is computed only when both endpoints are positive; county-years lacking any such cell have no observation in the log benchmark, and those losses are explicitly reported. The county MPGR sample is not restricted to accommodate log growth.
8. Source additivity, midpoint reconstruction, weight sums and preserved county-year keys are checked before estimation. No channel decomposition is used to select specifications.
9. The unweighted log-growth comparison deliberately illustrates departure from the MPGR weighting/transformation. Undefined log observations are counted in the table.
10. The payoff exhibit is `results/tables/disaggregation_h234_complete.md`, which compares coefficients and t-statistics across the three rungs for each horizon.

## Architecture and execution

```text
config/disaggregation.json                    Approved dimensions and conventions
config/disaggregation_lock.json               Specification and arithmetic hashes
code/build/disaggregation_snapshot.py         One-time unchanged cell acquisition and audit
code/build/disaggregation.py                  County and cell sufficient-statistic caches
code/lib/disaggregation_math.py               County-frame arithmetic extracted from baseline
code/estimate/disaggregation.py               Estimates from certified caches only
code/tests/disaggregation_direct_check.py     Direct h=2 NAICS2 verification
code/outputs/disaggregation.py                Saved-result report
```

Data live under shared `data/intermediate/disaggregation_h234/` and `data/analysis/disaggregation_h234/`. Code lives in the Git checkout. Logs belong to repository-root `batch_logs/macro_paper_application/`. The source boundary is unchanged constructed CBP/EFSY cells, not raw-data reconstruction.

The arithmetic reference is the existing `step_2/code/py/mpgr_disagg.py` machinery and its source estimator. The application does not import that producer, its candidate instrument, its sample defaults, its pruning routine, or its source-selection machinery. The county-frame arithmetic is extracted from the independently verified application baseline. The numerical estimator remains the pinned application-owned `baseline_math.py`.

## Exact sufficient-statistic implementation

With county/year effects and regressors constant across industry cells of a county-year, the relevant moments depend only on cell weight sums and weighted outcome sums. Store each county-year's weighted cell MPGR mean and total midpoint weight; for the log benchmark store the mean surviving-cell log growth and the number of surviving cells. Estimate on those means with their corresponding total weights. This preserves both the 2SLS coefficient and county-cluster score sums.

MPGR uses the same county-year observation count for its finite-sample correction at every rung. The log benchmark uses the number of surviving cell observations. This convention is explicit because using the number of collapsed rows for a cell-log regression would change its standard error.

The direct h=2 NAICS2 regressions use the actual cell rows and verify coefficient, standard error and first-stage F against the sufficient-statistic estimates. Direct NAICS4 and h=3/4 cell regressions are not separately repeated. Their annual-cell reconstruction checks and the same reduction apply; this distinction is recorded rather than presented as independent direct validation at every rung.

## Log-sample authority and results

All 18 requested rows have estimates in `results/coefficients/disaggregation_h234_complete.csv`. `config/log_sample_policy.json` authorizes the natural log-defined sample. `code/estimate/disaggregation_log_completion.py` runs only the four h=3/4 cell-log fits; existing estimates are reused without modification. `code/outputs/disaggregation_complete.py` renders the completed table and county-versus-cell gaps.

`results/manifests/disaggregation_log_blind.csv` lists county-years that contribute no cell-log observation. The completed table reports those losses and all undefined cell observations. This is the consequence of the log transformation, not a reason to exclude counties from MPGR. The source-cache certificate describes constructed data; its historical blocking policy does not override the explicit log-sample authority. The partial estimator/report preserve their own output identities and are not the completed comparison.

This run is disaggregation only. It does not authorize the separate standard-growth aggregation design or resolve the historical 2018 result discrepancy.


## Approved log-weight comparisons

`config/log_weight_variants.json` records two additional weight definitions for h=2/3/4 at NAICS2 and NAICS4: midpoint shares normalized over surviving industries and lagged employment shares divided by full county lagged employment. The natural log-defined sample, county/year fixed effects, clustering and treatment remain unchanged. County references are reused, not re-estimated.

`code/estimate/log_weight_variants.py midpoint_survivors` uses the saved weighted-log mean with county-year weight one. `code/estimate/log_weight_variants.py lag_full_county` uses the lag-weighted mean with the surviving lag-share mass. The latter sums are built by `code/build/lag_weighted_log.py`; other county inputs and caches remain untouched. Results and gaps are in `results/tables/log_weight_variants.md`.

The six full-county midpoint-share fits in `results/tables/disaggregation_weighted_log.md` use the total county midpoint denominator without survivor renormalization. They are separately labeled and do not define the researcher-style survivor-normalized comparison. Weighting comparisons are not, by themselves, a decomposition of curvature and lost entry/exit observations.
