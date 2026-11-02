// mpgr-run-class: estimate_only
// mpgr-cache: _data/temp/post_first_stage_full_grid/*_h2.dta, *_h3.dta, *_h4.dta
// mpgr-new-estimates: 17280
// mpgr-variation-scope: county+cz|award+duration|all-treatments|h2+h3+h4|std+log+mpgr|all-shares|all-winsor
// mpgr-batch-approved: explicit

version 18.0
clear all
set more off
set varabbrev off

local root "/Users/f1azmm00/Research/aggregation-disaggregation-invariance"
local input "/Users/f1azmm00/Library/CloudStorage/Dropbox-Matray/_AdrienTeam/Chenzi_Adrien/Invariance_Agg_Disagg/_data/temp/post_first_stage_full_grid"
local logs "`root'/batch_logs/macro_version"

capture log close
log using "`logs'/45_endpoint_growth_first_stage_grid.log", text replace

capture which reghdfe
assert _rc == 0
capture which winsor2
assert _rc == 0

capture program drop winsor4
program define winsor4
    syntax varlist(numeric min=1 max=1) [if], METhod(string) Level(string) [NEWvar(string)]
    version 13.0
    marksample touse
    tokenize `level'
    local threshold `1'

    if "`method'" == "tail" & "`newvar'" == "" {
        local uptail = 100 - `threshold'
        local lowtail = `threshold'
        quietly winsor2 `varlist' if `touse', cuts(`lowtail' `uptail') replace
    }
    if "`method'" == "tail" & "`newvar'" != "" {
        quietly generate `newvar' = `varlist'
        local uptail = 100 - `threshold'
        local lowtail = `threshold'
        quietly winsor2 `newvar' if `touse', cuts(`lowtail' `uptail') replace
    }
    if "`method'" == "iq" & "`newvar'" == "" {
        quietly summarize `varlist' if `touse', detail
        replace `varlist' = r(p50) - `threshold' * (r(p75) - r(p25)) if `touse' & `varlist' < r(p50) - `threshold' * (r(p75) - r(p25)) & `varlist' < .
        replace `varlist' = r(p50) + `threshold' * (r(p75) - r(p25)) if `touse' & `varlist' > r(p50) + `threshold' * (r(p75) - r(p25)) & `varlist' < .
    }
    if "`method'" == "iq" & "`newvar'" != "" {
        quietly generate `newvar' = `varlist'
        quietly summarize `newvar' if `touse', detail
        replace `newvar' = r(p50) - `threshold' * (r(p75) - r(p25)) if `touse' & `newvar' < r(p50) - `threshold' * (r(p75) - r(p25)) & `newvar' < .
        replace `newvar' = r(p50) + `threshold' * (r(p75) - r(p25)) if `touse' & `newvar' > r(p50) + `threshold' * (r(p75) - r(p25)) & `newvar' < .
    }
end

capture program drop winsor4_by
program define winsor4_by
    syntax varname(numeric), METhod(string) Level(string) BY(varname) GENerate(name)
    clonevar `generate' = `varlist'
    quietly levelsof `by', local(groups)
    foreach value of local groups {
        winsor4 `generate' if `by' == `value', method(`method') level(`level')
    }
end

tempname handle
tempfile results
postfile `handle' str8 rung str8 flow str24 treatment byte horizon str8 growth str3 share_code str14 winsor_rule str6 winsor_group double n_obs double n_units double n_states double df_absorbed double fs_coef double fs_se double fs_F str40 status using `results', replace

capture program drop run_grid_file
program define run_grid_file
    syntax, Rung(string) Flow(string) Treatment(string) Horizon(integer) Input(string) Handle(name)
    local growths std log mpgr
    local sharecodes s01 s02 s03 s04 s05 s06 s07 s08 s09 s10 s11 s12
    local winvars none iq4_pool iq5_pool tail1_pool iq4_year iq5_year tail1_year iq4_state iq5_state tail1_state

    use "`input'/`rung'_`flow'_`treatment'_h`horizon'.dta", clear
    sort unit year

    quietly reghdfe y_std, absorb(unit state_year) residuals(_sample_resid) vce(cluster unit) tolerance(1e-12)
    keep if e(sample)
    drop _sample_resid
    local df_a = e(df_a)
    quietly count
    local N = r(N)
    quietly levelsof unit, local(unit_values)
    local G : word count `unit_values'
    quietly levelsof state, local(state_values)
    local S : word count `state_values'
    local se_factor = sqrt((`N' - 1) / (`N' - `df_a' - 2))

    foreach g of local growths {
        clonevar y_`g'_none = y_`g'

        clonevar y_`g'_iq4_pool = y_`g'
        winsor4 y_`g'_iq4_pool, method(iq) level(4)
        clonevar y_`g'_iq5_pool = y_`g'
        winsor4 y_`g'_iq5_pool, method(iq) level(5)
        clonevar y_`g'_tail1_pool = y_`g'
        winsor4 y_`g'_tail1_pool, method(tail) level(1)

        winsor4_by y_`g', method(iq) level(4) by(year) generate(y_`g'_iq4_year)
        winsor4_by y_`g', method(iq) level(5) by(year) generate(y_`g'_iq5_year)
        winsor4_by y_`g', method(tail) level(1) by(year) generate(y_`g'_tail1_year)

        winsor4_by y_`g', method(iq) level(4) by(state) generate(y_`g'_iq4_state)
        winsor4_by y_`g', method(iq) level(5) by(state) generate(y_`g'_iq5_state)
        winsor4_by y_`g', method(tail) level(1) by(state) generate(y_`g'_tail1_state)
    }

    foreach g of local growths {
        foreach w of local winvars {
            quietly reghdfe y_`g'_`w', absorb(unit state_year) residuals(r_y_`g'_`w') tolerance(1e-12)
        }
        foreach s of local sharecodes {
            local idx = substr("`s'", 2, 2)
            quietly reghdfe z`g'`idx', absorb(unit state_year) residuals(r_z_`g'_`s') tolerance(1e-12)
        }
    }

    foreach g of local growths {
        foreach s of local sharecodes {
            foreach w of local winvars {
                local group = "none"
                if strpos("`w'", "pool") local group = "pooled"
                if strpos("`w'", "year") local group = "year"
                if strpos("`w'", "state") local group = "state"
                local rule = "none"
                if strpos("`w'", "iq4") local rule = "iq4"
                if strpos("`w'", "iq5") local rule = "iq5"
                if strpos("`w'", "tail1") local rule = "tail1"
                capture quietly regress r_y_`g'_`w' r_z_`g'_`s', noconstant vce(cluster unit)
                if _rc == 0 & e(N) > 0 {
                    local b = _b[r_z_`g'_`s']
                    local se = _se[r_z_`g'_`s'] * `se_factor'
                    local F = (`b' / `se')^2
                    post `handle' ("`rung'") ("`flow'") ("`treatment'") (`horizon') ("`g'") ("`s'") ("`rule'") ("`group'") (e(N)) (`G') (`S') (`df_a') (`b') (`se') (`F') ("ok")
                }
                else {
                    local rc = _rc
                    post `handle' ("`rung'") ("`flow'") ("`treatment'") (`horizon') ("`g'") ("`s'") ("`rule'") ("`group'") (.) (`G') (`S') (`df_a') (.) (.) (.) ("regression_failed_rc`rc'")
                }
            }
        }
    }
    noisily display "complete: `rung' `flow' `treatment' h=`horizon', N=`N'"
end

foreach rung in county cz {
    foreach flow in award duration {
        foreach treatment in net_nofloor net_floor25000 positive_nofloor positive_floor25000 {
            foreach horizon in 2 3 4 {
                run_grid_file, rung(`rung') flow(`flow') treatment(`treatment') horizon(`horizon') input("`input'") handle(`handle')
            }
        }
    }
}

postclose `handle'
use `results', clear
sort rung flow treatment horizon growth share_code winsor_rule winsor_group
format fs_coef fs_se fs_F %21.15g
export delimited using "`logs'/endpoint_growth_first_stage_grid.csv", replace datafmt
save "`input'/endpoint_growth_first_stage_grid.dta", replace

quietly count
assert r(N) == 17280
quietly count if status != "ok"
assert r(N) == 0
quietly count if missing(fs_coef) | missing(fs_se) | missing(fs_F)
assert r(N) == 0

log close
exit 0
