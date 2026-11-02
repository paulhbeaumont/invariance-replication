version 18.0
clear all
set more off
set varabbrev off

local root "`c(pwd)'"
local data_root : environment INVARIANCE_DROPBOX_ROOT
if "`data_root'" == "" {
    local home : environment HOME
    local data_root "`home'/Library/CloudStorage/Dropbox-Matray/_AdrienTeam/Chenzi_Adrien/Invariance_Agg_Disagg"
    capture confirm file "`data_root'/_data/temp/symmetric_nonoverlap_first_stage/pre_h2.dta"
    if _rc != 0 local data_root "`home'/Library/CloudStorage/Dropbox-Matray/Invariance_Agg_Disagg"
}
local input "`data_root'/_data/temp/symmetric_nonoverlap_first_stage"
local logs "`root'/batch_logs/macro_version"

capture confirm file "`input'/pre_h2.dta"
assert _rc == 0

capture log close
log using "`logs'/42_symmetric_nonoverlap_first_stages.log", text replace

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

capture program drop winsor4_by_inplace
program define winsor4_by_inplace
    syntax varname(numeric), METhod(string) Level(string) BY(varname)
    quietly levelsof `by', local(groups)
    foreach value of local groups {
        winsor4 `varlist' if `by' == `value', method(`method') level(`level')
    }
end

tempname handle
tempfile results
postfile `handle' str4 period byte horizon str10 scaling str8 share_label str8 winsor_rule str6 winsor_group double n_obs double n_units double n_states double df_absorbed double fs_coef double fs_se double fs_F str40 status using `results', replace

capture program drop run_cell
program define run_cell
    syntax, Period(string) Horizon(integer) Input(string) Handle(name)

    use "`input'/`period'_h`horizon'.dta", clear
    sort unit year

    quietly reghdfe y_income, absorb(unit state_year) residuals(_sample_resid) vce(cluster unit) tolerance(1e-12)
    keep if e(sample)
    drop _sample_resid
    assert !missing(y_income, y_employment)
    foreach scaling in income employment {
        foreach share in fixed4 fixed5 fixed10 {
            assert !missing(z_`scaling'_`share')
        }
    }
    local df_a = e(df_a)
    quietly count
    local N = r(N)
    quietly levelsof unit, local(unit_values)
    local G : word count `unit_values'
    quietly levelsof state, local(state_values)
    local S : word count `state_values'
    local se_factor = sqrt((`N' - 1) / (`N' - `df_a' - 2))

    local winvars n i4p i4y i5p i5y t1p t1y
    foreach scaling in inc emp {
        local scaling_label "income"
        if "`scaling'" == "emp" local scaling_label "employment"

        foreach w of local winvars {
            clonevar y_`scaling'_`w' = y_`scaling_label'
            if "`w'" == "i4p" winsor4 y_`scaling'_`w', method(iq) level(4)
            if "`w'" == "i4y" winsor4_by_inplace y_`scaling'_`w', method(iq) level(4) by(year)
            if "`w'" == "i5p" winsor4 y_`scaling'_`w', method(iq) level(5)
            if "`w'" == "i5y" winsor4_by_inplace y_`scaling'_`w', method(iq) level(5) by(year)
            if "`w'" == "t1p" winsor4 y_`scaling'_`w', method(tail) level(1)
            if "`w'" == "t1y" winsor4_by_inplace y_`scaling'_`w', method(tail) level(1) by(year)
            quietly reghdfe y_`scaling'_`w', absorb(unit state_year) residuals(r_y_`scaling'_`w') tolerance(1e-12)
        }

        foreach share in fixed4 fixed5 fixed10 {
            foreach w of local winvars {
                clonevar z_`scaling'_`share'_`w' = z_`scaling_label'_`share'
                if "`w'" == "i4p" winsor4 z_`scaling'_`share'_`w', method(iq) level(4)
                if "`w'" == "i4y" winsor4_by_inplace z_`scaling'_`share'_`w', method(iq) level(4) by(year)
                if "`w'" == "i5p" winsor4 z_`scaling'_`share'_`w', method(iq) level(5)
                if "`w'" == "i5y" winsor4_by_inplace z_`scaling'_`share'_`w', method(iq) level(5) by(year)
                if "`w'" == "t1p" winsor4 z_`scaling'_`share'_`w', method(tail) level(1)
                if "`w'" == "t1y" winsor4_by_inplace z_`scaling'_`share'_`w', method(tail) level(1) by(year)
                quietly reghdfe z_`scaling'_`share'_`w', absorb(unit state_year) residuals(r_z_`scaling'_`share'_`w') tolerance(1e-12)
            }
        }

        foreach share in fixed4 fixed5 fixed10 {
            foreach w of local winvars {
                local rule "none"
                local group "none"
                if substr("`w'", 1, 2) == "i4" local rule "iq4"
                if substr("`w'", 1, 2) == "i5" local rule "iq5"
                if substr("`w'", 1, 2) == "t1" local rule "tail1"
                if inlist("`w'", "i4p", "i5p", "t1p") local group "pooled"
                if inlist("`w'", "i4y", "i5y", "t1y") local group "year"

                capture quietly regress r_y_`scaling'_`w' r_z_`scaling'_`share'_`w', noconstant vce(cluster unit)
                if _rc == 0 & e(N) > 0 {
                    local b = _b[r_z_`scaling'_`share'_`w']
                    local se = _se[r_z_`scaling'_`share'_`w'] * `se_factor'
                    local F = (`b' / `se')^2
                    post `handle' ("`period'") (`horizon') ("`scaling_label'") ("`share'") ("`rule'") ("`group'") (e(N)) (`G') (`S') (`df_a') (`b') (`se') (`F') ("ok")
                }
                else {
                    local rc = _rc
                    post `handle' ("`period'") (`horizon') ("`scaling_label'") ("`share'") ("`rule'") ("`group'") (.) (`G') (`S') (`df_a') (.) (.) (.) ("regression_failed_rc`rc'")
                }
            }
        }
    }
    noisily display "complete: `period' h=`horizon', N=`N'"
end

foreach period in pre post {
    foreach horizon in 2 3 4 {
        run_cell, period(`period') horizon(`horizon') input("`input'") handle(`handle')
    }
}

postclose `handle'
use `results', clear
sort period horizon scaling share_label winsor_rule winsor_group
format fs_coef fs_se fs_F %21.15g
export delimited using "`logs'/symmetric_nonoverlap_first_stage.csv", replace datafmt
save "`input'/symmetric_nonoverlap_first_stage.dta", replace

quietly count
assert r(N) == 252
quietly count if status != "ok"
assert r(N) == 0
quietly count if missing(fs_coef) | missing(fs_se) | missing(fs_F)
assert r(N) == 0

log close
exit 0
