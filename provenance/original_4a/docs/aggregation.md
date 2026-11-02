# Aggregation specification and execution

The sole active specification is `config/paper_aggregation.json`. It selects unwinsorized standard employment growth, horizons 2 and 4, CZ/CEA/EA/state, the fixed retained county support and the 2016 endpoint. Source employment, spending and instrument inputs come only from the certified application cache.

## Equations and weights

Let omega_cjt = E_c,t-h / sum_c E_c,t-h and P_j1975 = sum_c P_c1975 within aggregate j. Aggregate spending X_jt and instrument Z_jt are omega-weighted county inputs.

- Both county regressions use omega_cjt * P_j1975 as their regression weight, county and year fixed effects, and aggregate-unit clustering.
- The unaugmented county equation instruments x with z.
- The augmented county equation instruments x and X with z and Z; delta_spill is the coefficient on X.
- The aggregate equation uses P_j1975 as its regression weight, aggregate-unit and year fixed effects, and aggregate-unit clustering. It instruments X with Z.
- County and aggregate outcomes are (E_t-E_t-h)/E_t-h, constructed from their respective employment endpoints. Neither outcome is winsorized.
- Delta_naive is beta_aggregate minus beta_county, with joint cluster-level IV covariance and equation-specific CRV1 adjustments. Tests use G-1 degrees of freedom.

The employment shares and population weights are separate named columns. Each county regression weight divided by aggregate population must equal the county's lagged employment share. Summing weights alone is not sufficient. Fixed population weights and level-specific FE do not impose an exact coefficient-addition identity.

## Stages

`code/run_aggregation.py estimate` runs only the 16 county-side fits and reuses the eight hash-pinned aggregate fits. The immutable aggregate reference has superseded county-side columns, which are not consumed. `code/lib/aggregation_design.py` supplies the shared design and validates county weights, aggregate inputs, geography and endpoint additivity.

`code/run_aggregation.py check` runs the fixed hand-calculated fixture and deliberate-failure tests, checks sample and aggregate-result preservation, and binds the check receipt to the specification, code, inputs and output.

`code/run_aggregation.py report` renders the paper collection. Direct renderer calls also require a matching specification and numerical check receipt. The PDF compiler separately enforces layout and visual review. Rendering never runs estimation.

The fixture `tests/fixtures/aggregation_weights.json` fixes employment 25/75, population 600/400 and expected county weights 250/750. These expected values require user approval to change; they must never be regenerated from the constructor. No equality of county coefficients across geographic partitions is required.

## Scope and historical material

Tables 1-5 retain their disaggregation specifications. `config/paper_weights.json` owns their population weighting, not aggregation weights. Historical reduced-form and MPGR configurations and coefficients are not active aggregation specifications. `results/tables/tables_for_paper_superseded_county_population/` is a preserved, explicitly superseded collection and must not be used for the paper.

The input boundary is certified constructed county data, not raw-data replication certification. Specification tests and numerical agreement are separate checks. No local hash mechanism can protect against intentionally rewriting both the specification and its approved fixtures; changing those requires explicit approval.
