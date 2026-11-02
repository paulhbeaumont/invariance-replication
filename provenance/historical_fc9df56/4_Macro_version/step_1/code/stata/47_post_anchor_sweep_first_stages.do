version 18.0
clear all
set more off
set varabbrev off

// Post-1998 anchor-year sweep, county first stages.
//
// Structural clone of 42's estimator: Frisch-Waugh-Lovell in two steps, with
// e(sample) fixed once on the unwinsorized LHS, a manual small-sample factor
// because the second-step regress does not know the absorbed degrees of
// freedom, LHS-only winsorization within start year, county and state-by-year
// fixed effects, and county clustering. Cloning rather than generalizing 42 is
// deliberate: any drift in this shared code path is global, so the
// anchor-1998 pin in 48 catches it with certainty.
//
// Each panel carries nine instruments, three exposure anchors by three window
// lengths. The diagonal, exposure anchor == schedule anchor, is the coupled
// headline; the off-diagonal separates the exposure window from the start-year
// phase, which a coupled-only grid cannot do.
//
// Two things 42 does not do.
//
// fs_rho, the partial correlation of the residualized outcome with the
// residualized instrument, is posted alongside fs_F. n_obs is not constant
// across anchors - 10/9/9 windows at h=2 and 5/4/4 at h=4 - and F scales with
// N at a fixed partial correlation, so F alone cannot say whether the anchor
// changed the instrument or only the sample size.
//
// The second postfile is a leave-one-window-out band at anchor 1998, dropping
// each start year in turn. It is the null distribution the across-anchor
// spread has to beat: if moving the anchor shifts F no more than deleting one
// arbitrary window does, the anchor did not do anything.
//
// Output:
//     batch_logs/macro_version/post_anchor_sweep_first_stage.csv   162 rows
//     batch_logs/macro_version/post_anchor_sweep_lowo.csv          126 rows

local root "`c(pwd)'"
local data_root : environment INVARIANCE_DROPBOX_ROOT
if "`data_root'" == "" {
    local home : environment HOME
    local data_root "`home'/Library/CloudStorage/Dropbox-Matray/_AdrienTeam/Chenzi_Adrien/Invariance_Agg_Disagg"
    capture confirm file "`data_root'/_data/temp/post_anchor_sweep/post_s1998_h2.dta"
    if _rc != 0 local data_root "`home'/Library/CloudStorage/Dropbox-Matray/Invariance_Agg_Disagg"
}
local input "`data_root'/_data/temp/post_anchor_sweep"
local logs "`root'/batch_logs/macro_version"

capture confirm file "`input'/post_s1998_h2.dta"
assert _rc == 0

capture log close
log using "`logs'/47_post_anchor_sweep_first_stages.log", text replace

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

// Fixes e(sample) on the unwinsorized outcome, then builds both LHS variants
// and every residual the cell needs. Shared by the main grid and the
// leave-one-window-out band so the two can never drift apart.
capture program drop prepare_cell
program define prepare_cell, rclass
    syntax, Anchors(string) Lengths(string)

    quietly reghdfe y_cum, absorb(unit state_year) residuals(_sample_resid) vce(cluster unit) tolerance(1e-12)
    keep if e(sample)
    drop _sample_resid
    local df_a = e(df_a)
    quietly count
    local N = r(N)
    quietly levelsof unit, local(unit_values)
    local G : word count `unit_values'
    quietly levelsof state, local(state_values)
    local S : word count `state_values'
    quietly levelsof year, local(year_values)
    local W : word count `year_values'

    // 41 and 46 build a balanced county set, so absorption must not drop a
    // single cell. 42 never checked this.
    assert `G' == 3080
    assert `S' == 48
    assert `N' == `G' * `W'
    assert `N' - `df_a' - 2 > 0

    local se_factor = sqrt((`N' - 1) / (`N' - `df_a' - 2))

    clonevar y_none = y_cum
    winsor4_by y_cum, method(tail) level(1) by(year) generate(y_tail1_year)
    quietly count if y_tail1_year != y_none
    local n_winsorized = r(N)

    quietly summarize year, meanonly
    local first_start = r(min)
    local last_start = r(max)

    quietly reghdfe y_none, absorb(unit state_year) residuals(r_y_none) tolerance(1e-12)
    quietly reghdfe y_tail1_year, absorb(unit state_year) residuals(r_y_tail1_year) tolerance(1e-12)
    foreach anchor of local anchors {
        foreach length of local lengths {
            quietly reghdfe z`anchor'_`length', absorb(unit state_year) residuals(r_z`anchor'_`length') tolerance(1e-12)
        }
    }

    return scalar first_start = `first_start'
    return scalar last_start = `last_start'
    return scalar N = `N'
    return scalar G = `G'
    return scalar S = `S'
    return scalar W = `W'
    return scalar df_a = `df_a'
    return scalar se_factor = `se_factor'
    return scalar n_winsorized = `n_winsorized'
end

tempname handle
tempfile results
postfile `handle' int sched_anchor int exposure_anchor byte horizon byte share_len ///
    str8 winsor_rule str6 winsor_group double n_windows double n_obs double n_units ///
    double n_states double df_absorbed double first_outcome_year double last_outcome_year ///
    double fs_coef double fs_se double fs_F double fs_rho double sd_r_y double sd_r_z ///
    double n_winsorized byte is_diagonal str40 status using `results', replace

tempname lowo_handle
tempfile lowo_results
postfile `lowo_handle' int sched_anchor int exposure_anchor byte horizon byte share_len ///
    str8 winsor_rule str6 winsor_group double dropped_start double n_windows double n_obs ///
    double df_absorbed double fs_coef double fs_se double fs_F double fs_rho ///
    str40 status using `lowo_results', replace

capture program drop run_cell
program define run_cell
    syntax, Sched(integer) Horizon(integer) Input(string) Handle(name) ///
        Anchors(string) Lengths(string)

    use "`input'/post_s`sched'_h`horizon'.dta", clear
    sort unit year

    prepare_cell, anchors("`anchors'") lengths("`lengths'")
    // Snapshot every return before any command below clears r().
    local N = r(N)
    local G = r(G)
    local S = r(S)
    local W = r(W)
    local df_a = r(df_a)
    local se_factor = r(se_factor)
    local n_winsorized = r(n_winsorized)
    local first_outcome = r(first_start) + 1
    local last_outcome = r(last_start) + `horizon'

    foreach outcome in none tail1_year {
        local rule "none"
        local group "none"
        if "`outcome'" == "tail1_year" {
            local rule "tail1"
            local group "year"
        }
        foreach anchor of local anchors {
            local diagonal = (`anchor' == `sched')
            foreach length of local lengths {
                capture quietly regress r_y_`outcome' r_z`anchor'_`length', noconstant vce(cluster unit)
                if _rc == 0 & e(N) > 0 {
                    local n_used = e(N)
                    local b = _b[r_z`anchor'_`length']
                    local se = _se[r_z`anchor'_`length'] * `se_factor'
                    local F = (`b' / `se')^2
                    quietly correlate r_y_`outcome' r_z`anchor'_`length'
                    local rho = r(rho)
                    quietly summarize r_y_`outcome'
                    local sd_y = r(sd)
                    quietly summarize r_z`anchor'_`length'
                    local sd_z = r(sd)
                    post `handle' (`sched') (`anchor') (`horizon') (`length') ("`rule'") ("`group'") ///
                        (`W') (`n_used') (`G') (`S') (`df_a') (`first_outcome') (`last_outcome') ///
                        (`b') (`se') (`F') (`rho') (`sd_y') (`sd_z') ///
                        (`n_winsorized') (`diagonal') ("ok")
                }
                else {
                    local rc = _rc
                    post `handle' (`sched') (`anchor') (`horizon') (`length') ("`rule'") ("`group'") ///
                        (`W') (.) (`G') (`S') (`df_a') (`first_outcome') (`last_outcome') ///
                        (.) (.) (.) (.) (.) (.) ///
                        (`n_winsorized') (`diagonal') ("regression_failed_rc`rc'")
                }
            }
        }
    }
    noisily display "complete: anchor `sched' h=`horizon', N=`N', windows=`W', outcome `first_outcome'-`last_outcome'"
end

// The null band. Anchor 1998 with its own exposure, dropping one start year at
// a time, so the sample shrinks by exactly one window at constant exposure.
capture program drop run_lowo
program define run_lowo
    syntax, Sched(integer) Horizon(integer) Input(string) Handle(name) ///
        Anchors(string) Lengths(string)

    use "`input'/post_s`sched'_h`horizon'.dta", clear
    quietly levelsof year, local(all_starts)

    foreach dropped of local all_starts {
        preserve
        quietly drop if year == `dropped'
        sort unit year
        prepare_cell, anchors("`anchors'") lengths("`lengths'")
        local W = r(W)
        local df_a = r(df_a)
        local se_factor = r(se_factor)

        foreach outcome in none tail1_year {
            local rule "none"
            local group "none"
            if "`outcome'" == "tail1_year" {
                local rule "tail1"
                local group "year"
            }
            foreach anchor of local anchors {
                foreach length of local lengths {
                    capture quietly regress r_y_`outcome' r_z`anchor'_`length', noconstant vce(cluster unit)
                    if _rc == 0 & e(N) > 0 {
                        local n_used = e(N)
                        local b = _b[r_z`anchor'_`length']
                        local se = _se[r_z`anchor'_`length'] * `se_factor'
                        local F = (`b' / `se')^2
                        quietly correlate r_y_`outcome' r_z`anchor'_`length'
                        local rho = r(rho)
                        post `handle' (`sched') (`anchor') (`horizon') (`length') ("`rule'") ("`group'") ///
                            (`dropped') (`W') (`n_used') (`df_a') (`b') (`se') (`F') (`rho') ("ok")
                    }
                    else {
                        local rc = _rc
                        post `handle' (`sched') (`anchor') (`horizon') (`length') ("`rule'") ("`group'") ///
                            (`dropped') (`W') (.) (`df_a') (.) (.) (.) (.) ("regression_failed_rc`rc'")
                    }
                }
            }
        }
        restore
    }
    noisily display "leave-one-window-out complete: anchor `sched' h=`horizon'"
end

local all_anchors "1998 1999 2000"
local all_lengths "4 5 10"

foreach sched in 1998 1999 2000 {
    foreach horizon in 2 3 4 {
        run_cell, sched(`sched') horizon(`horizon') input("`input'") ///
            handle(`handle') anchors("`all_anchors'") lengths("`all_lengths'")
    }
}

foreach horizon in 2 3 4 {
    run_lowo, sched(1998) horizon(`horizon') input("`input'") ///
        handle(`lowo_handle') anchors("1998") lengths("`all_lengths'")
}

postclose `handle'
postclose `lowo_handle'

use `results', clear
sort sched_anchor horizon exposure_anchor share_len winsor_rule
format fs_coef fs_se fs_F fs_rho sd_r_y sd_r_z %21.15g
export delimited using "`logs'/post_anchor_sweep_first_stage.csv", replace datafmt
save "`input'/post_anchor_sweep_first_stage.dta", replace

quietly count
assert r(N) == 162
quietly count if status != "ok"
assert r(N) == 0
quietly count if missing(fs_coef) | missing(fs_se) | missing(fs_F) | missing(fs_rho)
assert r(N) == 0
quietly count if is_diagonal == 1
assert r(N) == 54
quietly count if n_obs != 3080 * n_windows
assert r(N) == 0

use `lowo_results', clear
sort horizon share_len winsor_rule dropped_start
format fs_coef fs_se fs_F fs_rho %21.15g
export delimited using "`logs'/post_anchor_sweep_lowo.csv", replace datafmt
save "`input'/post_anchor_sweep_lowo.dta", replace

quietly count
assert r(N) == 126
quietly count if status != "ok"
assert r(N) == 0

log close
exit 0
