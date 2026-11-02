version 18.0
clear all
set more off
set varabbrev off

local repo "/Users/f1azmm00/Research/aggregation-disaggregation-invariance"
local dropbox "/Users/f1azmm00/Library/CloudStorage/Dropbox-Matray/_AdrienTeam/Chenzi_Adrien/Invariance_Agg_Disagg"
local source "`dropbox'/1_current_paper/_data/source/NS2014_replication/DataPrograms/for-regressions"
local results "`repo'/4_Macro_version/Results/ns14_diagnostics"
local temp "`dropbox'/4_Macro_version/data/temp/ns14_diagnostics"
local logs "`repo'/batch_logs/macro_version"

capture mkdir "`results'"
capture mkdir "`temp'"
capture mkdir "`logs'"

capture log close
log using "`logs'/51_ns14_exposure_diagnostics.log", text replace

capture which grstyle
if _rc ssc install grstyle, replace
capture which colorpalette
if _rc ssc install palettes, replace
capture which colrspace
if _rc ssc install colrspace, replace

set scheme s2color
grstyle clear
grstyle init
grstyle set plain, horizontal grid noextend
grstyle color background white
grstyle color plotregion_line white
grstyle color matrix_plotregion white
grstyle color major_grid "230 230 230"
grstyle color minor_grid "230 230 230"
grstyle linewidth major_grid vthin
grstyle linewidth minor_grid vthin
grstyle color tick "26 26 26"
grstyle color tick_label "26 26 26"
grstyle color axisline "26 26 26"
grstyle linewidth axisline thin
grstyle color heading black
grstyle color subheading "50 50 50"
grstyle color body "26 26 26"
grstyle set legend 6, nobox
grstyle linestyle legend none
grstyle set color "26 71 111" "144 53 59" "110 142 132" "108 70 117" "204 102 0" "217 194 99", n(6)
grstyle set symbolsize medium
grstyle linewidth p medium
grstyle graphsize x 8
grstyle graphsize y 4.5

tempfile baseline_characteristics channels main_data shifter slopes lead_results robustness_results balance_results channel_results

* Baseline state characteristics measured in the first sample year.
use "`source'/statefiscal.dta", clear
keep if year == 1966
isid statenum
assert state != ""
generate base_manuf = Manuf_GSP_SIC / Total_GSP_SIC
generate base_mining = Mining_GSP_SIC / Total_GSP_SIC
generate base_fedmil = fed_mil_GSP_SIC / Total_GSP_SIC
generate base_const = Const_GSP_SIC / Total_GSP_SIC
generate base_lnpop = ln(population)
generate base_emprate = employ_CES * 1000 / population
generate base_iur = iur / 100
keep statenum base_*
assert !missing(statenum, base_manuf, base_mining, base_fedmil, base_const, base_lnpop, base_emprate, base_iur)
save `baseline_characteristics', replace

* Candidate channels distinct from recorded state procurement.
use "`source'/statefiscal.dta", clear
keep statenum year Dfed_mil_emp Dfed_civi_emp Dfed_mil_GSP Dfed_civi_GSP Dstate_local_GSP
isid statenum year
save `channels', replace

* Exact Nakamura-Steinsson state panel and predetermined exposure instrument.
use "`source'/statefiscal_bi.dta", clear
isid statenum year
merge m:1 statenum using `baseline_characteristics', assert(match) nogen
merge 1:1 statenum year using `channels', assert(match) nogen
tsset statenum year

assert reldif(Dpredict_spend2, fracmil2 * Drcapspend_nat) < 1e-6 if !missing(Dpredict_spend2)
generate realint2 = fedfunds - cpi
generate z_lead2 = F2.Dpredict_spend2
generate z_lead4 = F4.Dpredict_spend2

preserve
keep year Drcapspend_nat
duplicates drop
isid year
save `shifter', replace
restore
save `main_data', replace

* Table III employment replication.
ivregress 2sls Demp (Drcapspend = Dpredict_spend2) i.year i.statenum, vce(cluster statenum)
local ns_b = _b[Drcapspend]
local ns_se = _se[Drcapspend]
local ns_N = e(N)
local ns_states = e(N_clust)
assert `ns_N' == 1989
assert `ns_states' == 51
assert abs(`ns_b' - 1.81) < 0.01
assert abs(`ns_se' - 0.41) < 0.01

preserve
clear
set obs 1
generate double beta = `ns_b'
generate double se = `ns_se'
generate long N = `ns_N'
generate int states = `ns_states'
generate double published_beta = 1.810
generate double published_se = 0.410
save "`temp'/ns14_replication.dta", replace
export delimited using "`temp'/ns14_replication.csv", replace
restore

* Reduced-form lead placebos.
tempname lead_handle
postfile `lead_handle' byte column str20 specification double b0 se0 b2 se2 b4 se4 p_joint N states using `lead_results', replace

regress Demp Dpredict_spend2 i.year i.statenum, vce(cluster statenum)
post `lead_handle' (1) ("Current") (_b[Dpredict_spend2]) (_se[Dpredict_spend2]) (.) (.) (.) (.) (.) (e(N)) (e(N_clust))

regress Demp Dpredict_spend2 z_lead2 i.year i.statenum, vce(cluster statenum)
test z_lead2
post `lead_handle' (2) ("Lead 2") (_b[Dpredict_spend2]) (_se[Dpredict_spend2]) (_b[z_lead2]) (_se[z_lead2]) (.) (.) (r(p)) (e(N)) (e(N_clust))

regress Demp Dpredict_spend2 z_lead2 z_lead4 i.year i.statenum, vce(cluster statenum)
test z_lead2 z_lead4
post `lead_handle' (3) ("Leads 2 and 4") (_b[Dpredict_spend2]) (_se[Dpredict_spend2]) (_b[z_lead2]) (_se[z_lead2]) (_b[z_lead4]) (_se[z_lead4]) (r(p)) (e(N)) (e(N_clust))

regress Demp Dpredict_spend2 z_lead2 z_lead4 i.year i.statenum if mod(year, 2) == 0, vce(cluster statenum)
test z_lead2 z_lead4
post `lead_handle' (4) ("Even years") (_b[Dpredict_spend2]) (_se[Dpredict_spend2]) (_b[z_lead2]) (_se[z_lead2]) (_b[z_lead4]) (_se[z_lead4]) (r(p)) (e(N)) (e(N_clust))

regress Demp Dpredict_spend2 z_lead2 z_lead4 i.year i.statenum if mod(year, 2) == 1, vce(cluster statenum)
test z_lead2 z_lead4
post `lead_handle' (5) ("Odd years") (_b[Dpredict_spend2]) (_se[Dpredict_spend2]) (_b[z_lead2]) (_se[z_lead2]) (_b[z_lead4]) (_se[z_lead4]) (r(p)) (e(N)) (e(N_clust))
postclose `lead_handle'

use `lead_results', clear
sort column
save "`temp'/ns14_lead_placebos.dta", replace
export delimited using "`temp'/ns14_lead_placebos.csv", replace

* Year-specific exposure gradients and national shifter.
use `main_data', clear
tempname slope_handle
postfile `slope_handle' int year double theta se N using `slopes', replace
quietly levelsof year if !missing(Demp, fracmil2), local(years)
foreach yy of local years {
    quietly regress Demp c.fracmil2 if year == `yy', vce(robust)
    post `slope_handle' (`yy') (_b[fracmil2]) (_se[fracmil2]) (e(N))
}
postclose `slope_handle'

use `slopes', clear
merge 1:1 year using `shifter', assert(match using) keep(match) nogen
generate lo = theta - invnormal(0.975) * se
generate hi = theta + invnormal(0.975) * se
save "`temp'/ns14_exposure_gradient_by_year.dta", replace
export delimited using "`temp'/ns14_exposure_gradient_by_year.csv", replace

twoway (rcap hi lo year, lcolor(navy) lwidth(thin)) ///
       (scatter theta year, mcolor(navy) msymbol(O) msize(small)), ///
       yline(0, lcolor("26 26 26") lwidth(vthin)) ///
       ytitle("Employment-growth exposure gradient") xtitle("") ///
       xlabel(1968(4)2006, labsize(small)) legend(off) ///
       name(g_gradient, replace)

twoway (line Drcapspend_nat year, lcolor(maroon) lwidth(medthick)), ///
       yline(0, lcolor("26 26 26") lwidth(vthin)) ///
       ytitle("National procurement change") xtitle("") ///
       xlabel(1968(4)2006, labsize(small)) legend(off) ///
       name(g_shift, replace)

graph combine g_gradient g_shift, cols(1) xcommon imargin(tiny) graphregion(color(white))
graph export "`results'/Figure_NS_exposure_gradient.pdf", replace

* Exposure balance using 1966 state characteristics.
use `main_data', clear
keep if year == 1966
isid statenum
egen z_exposure = std(fracmil2)
tempname balance_handle
postfile `balance_handle' byte order str36 characteristic double beta se p N using `balance_results', replace
local balance_vars base_manuf base_mining base_fedmil base_const base_lnpop base_emprate base_iur
local balance_labels `""Manufacturing output share" "Mining output share" "Federal military output share" "Construction output share" "Log population" "Employment-to-population ratio" "Insured unemployment rate""'
local k = 0
foreach var of local balance_vars {
    local ++k
    local lab : word `k' of `balance_labels'
    egen z_`var' = std(`var')
    quietly regress z_`var' z_exposure, vce(robust)
    local p = 2 * ttail(e(df_r), abs(_b[z_exposure] / _se[z_exposure]))
    post `balance_handle' (`k') ("`lab'") (_b[z_exposure]) (_se[z_exposure]) (`p') (e(N))
}
postclose `balance_handle'

use `balance_results', clear
sort order
assert N == 51
save "`temp'/ns14_exposure_balance.dta", replace
export delimited using "`temp'/ns14_exposure_balance.csv", replace

* IV sensitivity to exposure-specific baseline and macro controls.
use `main_data', clear
local base_interactions c.base_manuf#i.year c.base_mining#i.year c.base_fedmil#i.year c.base_const#i.year c.base_lnpop#i.year c.base_emprate#i.year c.base_iur#i.year
tempname robustness_handle
postfile `robustness_handle' byte column str42 specification double beta se fs_F N states using `robustness_results', replace

ivregress 2sls Demp (Drcapspend = Dpredict_spend2) i.year i.statenum, vce(cluster statenum)
local b = _b[Drcapspend]
local se = _se[Drcapspend]
local N = e(N)
local states = e(N_clust)
tempvar sample1
generate byte `sample1' = e(sample)
quietly regress Drcapspend Dpredict_spend2 i.year i.statenum if `sample1', vce(cluster statenum)
local fs_F = (_b[Dpredict_spend2] / _se[Dpredict_spend2])^2
post `robustness_handle' (1) ("Baseline") (`b') (`se') (`fs_F') (`N') (`states')

ivregress 2sls Demp `base_interactions' (Drcapspend = Dpredict_spend2) i.year i.statenum, vce(cluster statenum)
local b = _b[Drcapspend]
local se = _se[Drcapspend]
local N = e(N)
local states = e(N_clust)
tempvar sample2
generate byte `sample2' = e(sample)
quietly regress Drcapspend Dpredict_spend2 `base_interactions' i.year i.statenum if `sample2', vce(cluster statenum)
local fs_F = (_b[Dpredict_spend2] / _se[Dpredict_spend2])^2
post `robustness_handle' (2) ("Baseline covariates by year") (`b') (`se') (`fs_F') (`N') (`states')

ivregress 2sls Demp i.statenum#c.Doilprice (Drcapspend = Dpredict_spend2) i.year i.statenum, vce(cluster statenum)
local b = _b[Drcapspend]
local se = _se[Drcapspend]
local N = e(N)
local states = e(N_clust)
tempvar sample3
generate byte `sample3' = e(sample)
quietly regress Drcapspend Dpredict_spend2 i.statenum#c.Doilprice i.year i.statenum if `sample3', vce(cluster statenum)
local fs_F = (_b[Dpredict_spend2] / _se[Dpredict_spend2])^2
post `robustness_handle' (3) ("State oil sensitivities") (`b') (`se') (`fs_F') (`N') (`states')

ivregress 2sls Demp i.statenum#c.realint2 (Drcapspend = Dpredict_spend2) i.year i.statenum, vce(cluster statenum)
local b = _b[Drcapspend]
local se = _se[Drcapspend]
local N = e(N)
local states = e(N_clust)
tempvar sample4
generate byte `sample4' = e(sample)
quietly regress Drcapspend Dpredict_spend2 i.statenum#c.realint2 i.year i.statenum if `sample4', vce(cluster statenum)
local fs_F = (_b[Dpredict_spend2] / _se[Dpredict_spend2])^2
post `robustness_handle' (4) ("State interest-rate sensitivities") (`b') (`se') (`fs_F') (`N') (`states')

ivregress 2sls Demp i.statenum#c.Doilprice i.statenum#c.realint2 (Drcapspend = Dpredict_spend2) i.year i.statenum, vce(cluster statenum)
local b = _b[Drcapspend]
local se = _se[Drcapspend]
local N = e(N)
local states = e(N_clust)
tempvar sample5
generate byte `sample5' = e(sample)
quietly regress Drcapspend Dpredict_spend2 i.statenum#c.Doilprice i.statenum#c.realint2 i.year i.statenum if `sample5', vce(cluster statenum)
local fs_F = (_b[Dpredict_spend2] / _se[Dpredict_spend2])^2
post `robustness_handle' (5) ("Oil and interest-rate sensitivities") (`b') (`se') (`fs_F') (`N') (`states')
postclose `robustness_handle'

use `robustness_results', clear
sort column
assert states == 51
save "`temp'/ns14_iv_robustness.dta", replace
export delimited using "`temp'/ns14_iv_robustness.csv", replace

* Standardized reduced forms for candidate alternative channels.
use `main_data', clear
tempname channel_handle
postfile `channel_handle' byte order str42 outcome double beta se p N states using `channel_results', replace
local channel_vars Dfed_mil_emp Dfed_civi_emp Dfed_mil_GSP Dfed_civi_GSP Dstate_local_GSP Dpop Drcappurchases
local channel_labels `""Federal military employment" "Federal civilian employment" "Federal military output" "Federal civilian output" "State and local government output" "Population" "Government purchases measure""'
local k = 0
foreach var of local channel_vars {
    local ++k
    local lab : word `k' of `channel_labels'
    preserve
    keep if !missing(`var', Dpredict_spend2)
    egen y_std = std(`var')
    egen z_std = std(Dpredict_spend2)
    quietly regress y_std z_std i.year i.statenum, vce(cluster statenum)
    local p = 2 * ttail(e(df_r), abs(_b[z_std] / _se[z_std]))
    post `channel_handle' (`k') ("`lab'") (_b[z_std]) (_se[z_std]) (`p') (e(N)) (e(N_clust))
    restore
}
postclose `channel_handle'

use `channel_results', clear
sort order
assert states == 51
save "`temp'/ns14_alternative_channels.dta", replace
export delimited using "`temp'/ns14_alternative_channels.csv", replace

do "`repo'/4_Macro_version/step_1/code/stata/52_ns14_diagnostic_tables.do" parent

display as result "NS baseline employment multiplier: " %9.3f `ns_b' " (" %9.3f `ns_se' ")"
display as result "Outputs: `results'"
log close
exit 0
