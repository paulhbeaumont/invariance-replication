version 18.0
clear all
set more off
set varabbrev off

local root "`c(pwd)'"
local data_root : environment INVARIANCE_DROPBOX_ROOT
if "`data_root'" == "" {
    local home : environment HOME
    local data_root "`home'/Library/CloudStorage/Dropbox-Matray/_AdrienTeam/Chenzi_Adrien/Invariance_Agg_Disagg"
    capture confirm file "`data_root'/_data/temp/post_first_stage_full_grid/county_award_net_nofloor_h2_overlapping.dta"
    if _rc != 0 local data_root "`home'/Library/CloudStorage/Dropbox-Matray/Invariance_Agg_Disagg"
}
local input "`data_root'/_data/temp/post_first_stage_full_grid"
local logs "`root'/batch_logs/macro_version"

capture confirm file "`input'/county_award_net_nofloor_h2_overlapping.dta"
assert _rc == 0

capture log close
log using "`logs'/37_verify_post_first_stage_estimator.log", text replace

use "`input'/county_award_net_nofloor_h2_overlapping.dta", clear
quietly reghdfe y_cum zcum02, absorb(unit state_year) vce(cluster unit) tolerance(1e-12)
local full_b = _b[zcum02]
local full_se = _se[zcum02]
local full_N = e(N)

quietly reghdfe y_cum, absorb(unit state_year) residuals(ry) vce(cluster unit) tolerance(1e-12)
keep if e(sample)
local df_a = e(df_a)
quietly reghdfe zcum02, absorb(unit state_year) residuals(rz) tolerance(1e-12)
quietly regress ry rz, noconstant vce(cluster unit)
local fast_b = _b[rz]
local fast_se = _se[rz] * sqrt((e(N) - 1) / (e(N) - `df_a' - 2))

display "df absorbed: " `df_a'
display "full coefficient: " %21.15g `full_b'
display "fast coefficient: " %21.15g `fast_b'
display "full standard error: " %21.15g `full_se'
display "fast standard error: " %21.15g `fast_se'
assert abs(`full_b' - `fast_b') < 1e-8
assert abs(`full_se' - `fast_se') < 1e-8
assert `full_N' == e(N)

use "`input'/post_first_stage_full_grid.dta", clear
keep if rung == "county" & flow == "award" & window_mode == "overlapping" & treatment == "net_nofloor" & horizon == 2 & growth == "cum" & share_code == "s02" & winsor_rule == "none" & winsor_group == "none"
assert _N == 1
assert abs(fs_coef - `full_b') < 1e-8
assert abs(fs_se - `full_se') < 1e-8
assert n_obs == `full_N'

display "full coefficient: " %21.15g `full_b'
display "fast coefficient: " %21.15g `fast_b'
display "full standard error: " %21.15g `full_se'
display "fast standard error: " %21.15g `fast_se'
display "estimator cross-check passed"

log close
exit 0
