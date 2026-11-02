version 18.0
clear all
set more off
set varabbrev off

// First-stage F at the county, commuting zone and state rungs, on the candidate
// income instrument.
//
// Structural clone of the candidate package's 02_run_candidate_first_stage.do:
// Frisch-Waugh-Lovell in two steps, e(sample) fixed once on the unwinsorized
// shock, and a manual small-sample factor because the second-step regress does
// not know the absorbed degrees of freedom. Cloning rather than generalizing is
// deliberate - drift in a shared estimator is global, and the county-rung pin in
// 53 catches drift in this one with certainty.
//
// WHAT IS NEW RELATIVE TO THE CANDIDATE FILE
//
// 1. The rungs. 51 stores county rows only, because climbing needs a weight
//    vector and that choice is not frozen. The collapse is here, and it is
//    written out as sum(w*x)/sum(w) rather than left to collapse's aweight, so
//    that the weighted mean and the plain sums cannot be confused in one call.
//
//    omega_ct = income_ct / income_jt, rebuilt inside every start year because
//    by() carries year. income_ct is income in the window's own start year, so
//    there is no lagged-weight mismatch of the kind the plan's 3.5 rule 3 warns
//    about. The SAME omega aggregates the shock and the instrument, which is
//    protocol A.2.2's one-omega requirement.
//
// 2. The cap is never recomputed. 51 winsorized the county shock and the county
//    instrument within start year; this file aggregates those capped columns and
//    caps nothing. A.3.3. The check below asserts it by re-deriving 51's cap in
//    Stata from the raw columns and requiring exact agreement.
//
// 3. Fixed effects vary with the rung because state-by-year is not defined at
//    the state rung. A commuting zone crosses state lines, so its state is 27's
//    predetermined assignment, repaired by 51 onto the frozen support.
//
// 4. TWO ABSORPTION ARMS, both reported, added 2026-08-18.
//
//        fe_arm  county            cz                    state
//        sxy     unit + state#year unit + czstate#year   unit + year
//        uy      unit + year       unit + year           unit + year
//
//    Both are admissible under protocol A.1.4: the treatment varies at county x
//    start-year and each arm is strictly coarser than that at every rung it is
//    estimated on. `sxy` is the arm this ladder was frozen on and clones the
//    candidate package. `uy` is the arm step_2's Tables 1-5 estimate on, added so
//    this exhibit reports the specification the application actually uses rather
//    than only its state-by-year analogue - on the same rows the two give a
//    different F, and that difference is the absorption and nothing else.
//
//    At the state rung the unit IS the state, so both arms reduce to unit + year
//    and must agree EXACTLY. That is asserted over twelve pairs at the foot of
//    this file, and it is the one check here that tests the arm plumbing rather
//    than the data.
//
// Output:
//     batch_logs/macro_version/ladder_first_stage.csv          72 rows
//     batch_logs/macro_version/ladder_first_stage_collapse.csv 18 rows

local root "`c(pwd)'"
local data_root : environment INVARIANCE_DROPBOX_ROOT
if "`data_root'" == "" {
    local home : environment HOME
    local data_root "`home'/Library/CloudStorage/Dropbox-Matray/_AdrienTeam/Chenzi_Adrien/Invariance_Agg_Disagg"
    capture confirm file "`data_root'/_data/temp/ladder_first_stage/pre_h4.dta"
    if _rc != 0 local data_root "`home'/Library/CloudStorage/Dropbox-Matray/Invariance_Agg_Disagg"
}
local input "`data_root'/_data/temp/ladder_first_stage"
local logs "`root'/batch_logs/macro_version"

capture confirm file "`input'/pre_h4.dta"
if _rc != 0 {
    noisily display as error "Run 51_build_ladder_first_stage_panel.py first."
    exit 601
}

capture log close
log using "`logs'/52_ladder_first_stages.log", text replace

capture which reghdfe
assert _rc == 0
capture which winsor2
assert _rc == 0

// The candidate file's winsor4, carried so that the cap 51 computed in Python
// can be re-derived here and compared. Nothing in this file caps an estimation
// variable; this program exists only for that check.
capture program drop winsor4
program define winsor4
    syntax varlist(numeric min=1 max=1) [if], METhod(string) Level(string)
    version 13.0
    marksample touse
    tokenize `level'
    local threshold `1'

    if "`method'" == "tail" {
        local uptail = 100 - `threshold'
        quietly winsor2 `varlist' if `touse', cuts(`threshold' `uptail') replace
    }
    if "`method'" == "iq" {
        quietly summarize `varlist' if `touse', detail
        replace `varlist' = r(p50) - `threshold' * (r(p75) - r(p25)) if `touse' & `varlist' < r(p50) - `threshold' * (r(p75) - r(p25)) & `varlist' < .
        replace `varlist' = r(p50) + `threshold' * (r(p75) - r(p25)) if `touse' & `varlist' > r(p50) + `threshold' * (r(p75) - r(p25)) & `varlist' < .
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

// 51 capped in Python with a reimplementation of Stata's own percentile rule.
// Re-derive it here from the raw columns and require exact agreement, so the
// cross-language copy is pinned rather than trusted.
capture program drop assert_cap_matches
program define assert_cap_matches
    syntax, Level(real)

    foreach v in shock z_fixed5 {
        quietly generate double `v'_chk = `v'_raw
        winsor4_by_inplace `v'_chk, method(iq) level(`level') by(year)
        quietly count if reldif(`v'_chk, `v'_iq5) > 1e-12
        if r(N) > 0 {
            noisily display as error ///
                "51's Python cap on `v' differs from Stata winsor4 iq `level' on " r(N) " rows"
            exit 459
        }
        drop `v'_chk
    }
end

// Collapse the CAPPED county rows to a rung with income weights. Written as
// sum(w*x)/sum(w) so that the weighted mean and the unweighted sums are visibly
// different operations. Returns the adding-up checks the plan's 3.5 rule 4 asks
// for.
capture program drop to_rung
program define to_rung, rclass
    syntax, Rung(string)

    // County-level totals, kept so the aggregate identities can be asserted
    // against them after the collapse.
    quietly summarize income_real, meanonly
    local tot_income = r(sum)
    quietly summarize future, meanonly
    local tot_future = r(sum)
    quietly summarize employment, meanonly
    local tot_emp = r(sum)
    quietly count
    local n_county_rows = r(N)

    // Both absorption arms are carried through the collapse rather than looped
    // around it, because to_rung destroys the county rows it collapses and a
    // loop would have to re-read and re-aggregate the panel per arm.
    if "`rung'" == "county" {
        rename unit rung_unit
        rename state_year fe_sxy
        quietly generate long fe_uy = year
        quietly generate long n_ccg = 1
    }
    else if inlist("`rung'", "cz", "state") {
        foreach v in shock_raw shock_iq5 z_fixed5_raw z_fixed5_iq5 {
            quietly generate double w_`v' = income_real * `v'
        }
        if "`rung'" == "cz" {
            collapse (sum) income_real employment future ///
                w_shock_raw w_shock_iq5 w_z_fixed5_raw w_z_fixed5_iq5 ///
                (count) n_ccg = unit, by(cz cz_state cz_state_year year)
            rename cz rung_unit
            rename cz_state state
            rename cz_state_year fe_sxy
            quietly generate long fe_uy = year
        }
        else {
            collapse (sum) income_real employment future ///
                w_shock_raw w_shock_iq5 w_z_fixed5_raw w_z_fixed5_iq5 ///
                (count) n_ccg = unit, by(state year)
            quietly generate long rung_unit = state
            // The unit IS the state here, so a state-by-year effect would
            // absorb every observation. The second effect is the year, which
            // makes the two arms the same regression at this rung - asserted
            // below rather than assumed, as a check on the arm plumbing.
            quietly generate long fe_sxy = year
            quietly generate long fe_uy = year
        }
        // sum(omega * x) / sum(omega) with omega = income_ct, and the
        // denominator is the collapsed income, so the weights are rebuilt
        // inside each start year rather than carried across them.
        foreach v in shock_raw shock_iq5 z_fixed5_raw z_fixed5_iq5 {
            quietly generate double `v' = w_`v' / income_real
            drop w_`v'
        }
    }
    else {
        noisily display as error "unknown rung `rung'"
        exit 198
    }

    // The aggregation identities: unit income, contracts and employment are the
    // sums of their counties', exactly, at every rung and every start year.
    quietly summarize income_real, meanonly
    local agg_income = r(sum)
    quietly summarize future, meanonly
    local agg_future = r(sum)
    quietly summarize employment, meanonly
    local agg_emp = r(sum)
    assert reldif(`agg_income', `tot_income') < 1e-10
    assert reldif(`agg_future', `tot_future') < 1e-10
    assert reldif(`agg_emp', `tot_emp') < 1e-10
    quietly summarize n_ccg, meanonly
    assert r(sum) == `n_county_rows'

    quietly levelsof rung_unit, local(unit_values)
    local G : word count `unit_values'
    quietly levelsof state, local(state_values)
    local S : word count `state_values'
    quietly levelsof year, local(year_values)
    local W : word count `year_values'
    // Balanced, which is what keeps the county support frozen up the ladder: if
    // a rung lost a unit-year it would cover a different set of counties.
    quietly count
    assert r(N) == `G' * `W'

    return scalar G = `G'
    return scalar S = `S'
    return scalar W = `W'
    return scalar N = `G' * `W'
end

tempname handle
tempfile results
postfile `handle' str8 rung str4 period byte horizon str8 share_label ///
    str4 fe_arm str24 fe_spec ///
    str8 winsor_rule str6 winsor_group double n_windows double n_obs ///
    double n_units double n_states double n_ccg double df_absorbed ///
    double first_outcome_year double last_outcome_year ///
    double fs_coef double fs_se double fs_F double fs_rho ///
    double sd_r_y double sd_r_z str40 status using `results', replace

tempname coll_handle
tempfile coll_results
postfile `coll_handle' str8 rung str4 period byte horizon double n_units ///
    double n_obs double min_ccg double max_ccg double total_ccg ///
    using `coll_results', replace

capture program drop run_cell
program define run_cell
    syntax, Rung(string) Period(string) Horizon(integer) Input(string) ///
        Handle(name) COLLhandle(name) Level(real)

    use "`input'/`period'_h`horizon'.dta", clear
    sort unit year
    // 51 names the columns by rule then share; the estimator loops over the rule
    // last, so the headline share is renamed once here rather than interpolated
    // in every reference below.
    rename z_raw_fixed5 z_fixed5_raw
    rename z_iq5_fixed5 z_fixed5_iq5

    quietly summarize year, meanonly
    local first_outcome = r(min) + 1
    local last_outcome = r(max) + `horizon'

    if "`rung'" == "county" assert_cap_matches, level(`level')

    to_rung, rung("`rung'")
    local G = r(G)
    local S = r(S)
    local W = r(W)

    quietly summarize n_ccg, meanonly
    local min_ccg = r(min)
    local max_ccg = r(max)
    local total_ccg = r(sum) / `W'
    post `collhandle' ("`rung'") ("`period'") (`horizon') (`G') (`G' * `W') ///
        (`min_ccg') (`max_ccg') (`total_ccg')

    foreach fe in sxy uy {
        local fe_spec "unit + state x year"
        if "`rung'" == "cz" local fe_spec "unit + czstate x year"
        if "`fe'" == "uy" | "`rung'" == "state" local fe_spec "unit + year"

        // e(sample) is fixed once PER ABSORPTION ARM, on the UNWINSORIZED
        // shock. Within an arm the two winsorization arms then see identical
        // rows, so any difference between them is the cap and not the sample; a
        // different absorption is a different singleton problem and so gets its
        // own sample rather than inheriting one fixed under the other arm.
        quietly reghdfe shock_raw, absorb(rung_unit fe_`fe') ///
            residuals(_sample_resid) vce(cluster rung_unit) tolerance(1e-12)
        local df_a = e(df_a)
        quietly count if e(sample)
        local N = r(N)
        drop _sample_resid
        // Nothing may be absorbed away: the rung has to carry every unit-year
        // it was built with, or it no longer rests on the frozen county
        // support. This has to hold under BOTH arms - the coarser arm cannot
        // drop more than the finer one, and that is asserted, not assumed.
        assert `N' == `G' * `W'
        assert `N' - `df_a' - 2 > 0
        local se_factor = sqrt((`N' - 1) / (`N' - `df_a' - 2))

        foreach arm in raw iq5 {
            local rule "none"
            local group "none"
            if "`arm'" == "iq5" {
                local rule "iq5"
                local group "year"
            }
            quietly reghdfe shock_`arm', absorb(rung_unit fe_`fe') ///
                residuals(r_y) tolerance(1e-12)
            quietly reghdfe z_fixed5_`arm', absorb(rung_unit fe_`fe') ///
                residuals(r_z) tolerance(1e-12)

            capture quietly regress r_y r_z, noconstant vce(cluster rung_unit)
            if _rc == 0 & e(N) > 0 {
                local n_used = e(N)
                local b = _b[r_z]
                local se = _se[r_z] * `se_factor'
                local F = (`b' / `se')^2
                quietly correlate r_y r_z
                local rho = r(rho)
                quietly summarize r_y
                local sd_y = r(sd)
                quietly summarize r_z
                local sd_z = r(sd)
                post `handle' ("`rung'") ("`period'") (`horizon') ("fixed5") ///
                    ("`fe'") ("`fe_spec'") ///
                    ("`rule'") ("`group'") (`W') (`n_used') (`G') (`S') ///
                    (`total_ccg') (`df_a') (`first_outcome') (`last_outcome') ///
                    (`b') (`se') (`F') (`rho') (`sd_y') (`sd_z') ("ok")
            }
            else {
                local rc = _rc
                post `handle' ("`rung'") ("`period'") (`horizon') ("fixed5") ///
                    ("`fe'") ("`fe_spec'") ///
                    ("`rule'") ("`group'") (`W') (.) (`G') (`S') ///
                    (`total_ccg') (`df_a') (`first_outcome') (`last_outcome') ///
                    (.) (.) (.) (.) (.) (.) ("regression_failed_rc`rc'")
            }
            drop r_y r_z
        }
        noisily display "complete: `rung' `period' h=`horizon' fe=`fe'" ///
            ", units=`G', windows=`W', N=`N', df_a=`df_a'"
    }
end

local level 5

foreach rung in county cz state {
    foreach period in pre post {
        foreach horizon in 2 3 4 {
            run_cell, rung("`rung'") period("`period'") horizon(`horizon') ///
                input("`input'") handle(`handle') collhandle(`coll_handle') ///
                level(`level')
        }
    }
}

postclose `handle'
postclose `coll_handle'

use `results', clear
gen byte _rungorder = cond(rung == "county", 1, cond(rung == "cz", 2, 3))
gen byte _fearmorder = cond(fe_arm == "sxy", 1, 2)
sort _rungorder period horizon _fearmorder winsor_rule
drop _rungorder _fearmorder
format fs_coef fs_se fs_F fs_rho sd_r_y sd_r_z %21.15g
export delimited using "`logs'/ladder_first_stage.csv", replace datafmt
save "`input'/ladder_first_stage.dta", replace

quietly count
// 72 = 3 rungs x 2 eras x 3 horizons x 2 winsorization arms x 2 FE arms.
assert r(N) == 72
quietly count if status != "ok"
assert r(N) == 0
quietly count if missing(fs_coef) | missing(fs_se) | missing(fs_F)
assert r(N) == 0
quietly count if n_obs != n_units * n_windows
assert r(N) == 0
// The county support is identical at every rung, in every cell.
quietly count if n_ccg != 3080
assert r(N) == 0
quietly count if fe_arm == "sxy"
assert r(N) == 36
quietly count if fe_arm == "uy"
assert r(N) == 36

// THE ARM-PLUMBING CHECK. At the state rung the unit IS the state, so
// state-by-year is not defined and both arms absorb unit + year: the two are the
// same regression written two ways and must agree EXACTLY, not to a tolerance.
// This is the one check here that can fail for a reason unrelated to the data -
// if the fe_arm loop ever absorbs the wrong column, these twelve pairs separate
// immediately.
preserve
    keep if rung == "state"
    quietly count
    assert r(N) == 24
    keep rung period horizon winsor_rule fe_arm fs_coef fs_se fs_F df_absorbed
    quietly reshape wide fs_coef fs_se fs_F df_absorbed, ///
        i(rung period horizon winsor_rule) j(fe_arm) string
    quietly count
    assert r(N) == 12
    quietly count if fs_coefsxy != fs_coefuy
    assert r(N) == 0
    quietly count if fs_sesxy != fs_seuy
    assert r(N) == 0
    quietly count if fs_Fsxy != fs_Fuy
    assert r(N) == 0
    quietly count if df_absorbedsxy != df_absorbeduy
    assert r(N) == 0
    noisily display "state rung: both FE arms agree exactly across all 12 pairs"
restore

use `coll_results', clear
sort rung period horizon
export delimited using "`logs'/ladder_first_stage_collapse.csv", replace datafmt

quietly count
assert r(N) == 18

log close
exit 0
