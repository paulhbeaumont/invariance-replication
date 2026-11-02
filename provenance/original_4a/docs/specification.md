# Paper-application baseline

Authority: `config/baseline.json`, its hash lock `config/baseline_lock.json`, and the input inventory `config/baseline_sources.json`. The selected exposure is rolling five years with a five-year gap. The baseline includes county and year fixed effects and ends in 2016.

The files `approved_spec.yaml`, `county_inputs.json`, `procurement.json`, and `exposure_grid.json` describe the diagnostic 1985–2006 implementation. They are not authority for the selected paper baseline. `code/run.py` refuses diagnostic execution while the baseline lock is present.

## Regression weights

The active paper baseline is selected by `config/paper_weights.json`: fixed county population in 1975. County-industry regressions multiply the existing industry-specific weight by county 1975 population. Aggregation is governed by `config/paper_aggregation.json`: aggregate regressions use summed 1975 population; both county equations use that aggregate population times the lagged county employment share within the aggregate. Aggregate spending and instruments use those same employment shares.

The equal-weight numerical fixture and its hash lock certify the constructed inputs and estimator extraction, not the active population-weighted results. `code/estimate/population_tables.py` writes active disaggregation estimates under `results/coefficients/population1975/`. No fixture is regenerated when weights change.

## Equations

Let D be annual real net procurement, P county population, and E county employment from the established EFSY/CBP county aggregation. For county c and regression endpoint t:

```text
h = 2
Gbar_ct = (D_ct + D_c,t-1)/2
x_raw_ct = (Gbar_ct - D_c,t-2)/P_c,t-2
x_ct = pooled p1/p99 winsorization of x_raw_ct
 y_ct = 2*(E_ct - E_c,t-2)/(E_ct + E_c,t-2)
W_t = {t-11, t-10, t-9, t-8, t-7}
a_ct = sum_{s in W_t} D_cs / sum_{s in W_t} P_cs
n_t = [sum_c Gbar_ct - sum_c D_c,t-2] / sum_c P_c,t-2
z_ct = a_ct*n_t
```

The window average excludes the baseline year. Exposure is a ratio of sums, not a mean of annual per-person ratios. National sums include the focal county and use the frozen retained-county support. The instrument and employment growth are not winsorized. No composition instrument, leave-one-out correction, nominal action floor or additional transition-quarter adjustment is applied.

The spending snapshot comes from producer 51: net obligations, construction excluded, real dollars using its deflator. Its annual series is an input, not reconstructed by the estimator. The coefficient is an employment-growth response per real procurement dollar per baseline resident, not a dollar-output multiplier.

## Sample and estimation

The estimation frame contains 3,058 county groups and 113,146 observations, 1980–2016, as recorded in `tests/fixtures/baseline_roll5_gap5_reference.csv`. Whole-county support is frozen from the source producer's employment screen through 2018. The estimation endpoint and the historical screening endpoint are distinct. The build checks positive, nonmissing endpoint employment and population and does not delete individual rows.

The paper county baseline estimates 2SLS with fixed 1975 population weights, county and year fixed effects, and county-clustered CRV1 standard errors. The frozen equal-weight reference is a separate input-certification test. The first-stage F is the one-instrument clustered Wald statistic. The finite-sample convention is the source-extracted `ssc_k` and `crv1` implementation in `code/lib/baseline_math.py`.

In the equal-weight reference stage only, the frozen grid row is independent of the self-contained builder. Its coefficient, standard error, t, F and counts must match under relative tolerance 1e-10 and absolute tolerance 1e-12. Population-weighted paper estimates are validated separately and are not required to match that equal-weight fixture. A failure stops publication; the fixture must not be regenerated to accommodate the builder.

## Stages and guards

- `code/build/snapshot_baseline.py`: one-time acquisition of the declared constructed inputs, source functions and reference fixture. It imports no sibling code and refuses to overwrite the snapshot or fixture.
- `code/build/baseline.py`: reads application-owned snapshots and writes the county analysis cache plus a certificate. It runs no regressions.
- `code/estimate/baseline.py`: reads the certified county cache only. It cannot rebuild data. It checks the frozen reference before writing coefficients.
- `code/outputs/baseline.py`: reads verified coefficients only. It cannot run estimation.
- `code/run_baseline.py`: runs exactly one named stage and saves its log at the repository-root `batch_logs/macro_paper_application/`.

Configuration, source inventory, reference, numerical routines, cached data, and cache-builder/I/O hashes are checked at their respective boundaries. An intentional specification change requires explicit approval and a distinct specification, cache and result identity, not a patched reference fixture.

## Protocol pre-code answers and test boundaries

1. The baseline unit is the consistent county group. For disaggregation the cell is industry, at the two-digit and allocated four-digit rungs used by the existing producer. No geographic aggregate rung is selected here.
2. County employment must equal the sum of its unchanged industry cells. The county-cache reference match is verified. Separate annual-cell reconstruction and h=2/3/4 midpoint reconstruction checks are recorded in `docs/disaggregation.md`; the county-cache reference match alone is not their evidence.
3. Both the endogenous treatment `x_ct` and excluded instrument `z_ct` vary by county-year and are defined above.
4. Disaggregation repeats both variables unchanged across industries. It does not split procurement or reconstruct an industry-specific instrument.
5. The selected baseline has county and year effects. An exact disaggregation comparison can repeat these effects at the cell level. County-by-year effects are prohibited because they absorb treatment. Industry-by-year effects are a separate specification, not an exact-equality fixture.
6. The selected county outcome is midpoint growth with fixed 1975 population weights. Its disaggregated counterpart requires county 1975 population times `(E_cmt + E_cm,t-2)/(E_ct + E_c,t-2)` weights. The active aggregation exhibit uses standard growth with employment-share times aggregate-population weights, as defined in `config/paper_aggregation.json`.
7. Every test must retain the same county support and apply exclusions to whole counties at all rungs. Industry zeros must not be discarded to make a growth transformation run. The selected baseline cache has no additional row deletions.
8. The county-cache and frozen-reference checks pass in `results/manifests/baseline_roll5_gap5_verification.json`. Disaggregation checks and execution boundaries are in `docs/disaggregation.md`. Channel statistics do not precede estimation.
9. No log-growth or standard-growth comparison is produced by the baseline stage. Such comparisons must be explicitly labeled in the test exhibit rather than substituted for the baseline.
10. The equal-weight reference exhibit is the reproduced baseline, `results/tables/baseline_roll5_gap5.md`. The separate disaggregation exhibit is `results/tables/disaggregation_h234_complete.md`; natural log-sample losses are explicit and do not restrict the MPGR sample.

## Verification boundary

Normal build, estimate and report stages have no runtime dependency on sibling macro code. The clean-room guard test runs the estimator from an isolated application copy and reproduces the coefficient file byte for byte. Inputs are application-owned constructed snapshots, not a certified raw-to-result acquisition pipeline. The historical 2018 saved-result discrepancy remains unresolved and is not erased by matching the selected 2016 reference.
