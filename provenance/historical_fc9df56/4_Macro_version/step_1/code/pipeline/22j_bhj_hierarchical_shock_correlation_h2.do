version 16
clear all
set more off

local project "/Users/f1azmm00/Research/aggregation-disaggregation-invariance"
local macro "`project'/4_Macro_version"
local results "/Users/f1azmm00/Library/CloudStorage/Dropbox-Matray/_AdrienTeam/Chenzi_Adrien/Invariance_Agg_Disagg/4_Macro_version/Results/pipeline_diagnostics"

capture log close bhjicc
log using "`project'/batch_logs/macro/22j_bhj_hierarchical_shock_correlation_h2.log", text replace name(bhjicc)

use "`macro'/data/intermediate/bhj_hierarchical_shocks_h2.dta", clear
assert _N == 27446
assert !missing(shock, year, category_id, fsc_group_id)
sort category_id year
by category_id: assert fsc_group_id == fsc_group_id[1]

quietly tabulate year, generate(target_year_)
ds target_year_*
local target_year_indicators `r(varlist)'

mixed shock i.year || fsc_group_id: `target_year_indicators', nocons cov(exchangeable) || category_id:, mle vce(oim) nolog

local observations = e(N)
quietly levelsof fsc_group_id, local(group_levels)
local groups : word count `group_levels'
quietly levelsof category_id, local(category_levels)
local categories : word count `category_levels'

local variance_group = exp(_b[lns1_1_1:_cons])^2
local variance_category = exp(_b[lns2_1_1:_cons])^2
local variance_residual = exp(_b[lnsig_e:_cons])^2
local variance_total = `variance_group' + `variance_category' + `variance_residual'

nlcom ///
    (fsc_group_year_icc: exp(_b[lns1_1_1:_cons])^2 / (`variance_total')) ///
    (category_icc: exp(_b[lns2_1_1:_cons])^2 / (`variance_total')), post

local icc_group = _b[fsc_group_year_icc]
local icc_category = _b[category_icc]

clear
set obs 2
generate str32 component = ""
generate str48 component_label = ""
generate double icc = .
generate double model_se = .
generate double variance_component = .
generate long mixed_observations = `observations'
generate int fsc_groups = `groups'
generate int categories = `categories'

replace component = "fsc_group_year" in 1
replace component_label = "FSC group by target year" in 1
replace icc = `icc_group' in 1
replace variance_component = `variance_group' in 1

replace component = "category" in 2
replace component_label = "Four-character category" in 2
replace icc = `icc_category' in 2
replace variance_component = `variance_category' in 2

export delimited using "`results'/bhj_hierarchical_shock_icc_h2.csv", replace
list, noobs clean

log close bhjicc
