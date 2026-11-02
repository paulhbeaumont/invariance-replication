version 17.0
clear all
set more off
args projectroot
if `"`projectroot'"' == "" local projectroot "C:/Users/pbeaum/Beaumont Dropbox/Paul Beaumont/Trade_midpoint"
local box "`projectroot'/4_Macro_version/sandbox_Paul"
local result "`box'/output/results_recap_v3"
capture log close
log using "`projectroot'/batch_logs/macro_paul_results_recap_v3/graphics_detail.log", text replace

* Plot only: all numbers come from verified, saved regressions.
* Figure 1 palette in Paul's 301_Graphs.do: navy%80 and maroon%90.
* White environment reproduced from its 00b_own_programs.do.
set scheme s2color
import delimited using "`result'/waterfall.csv", clear varnames(1) stringcols(2 6) asdouble
assert _N == 6
assert abs(change[1]-6.123090808742639) < 1e-9
assert abs(change[6]-3.983928961174704) < 1e-9
twoway ///
    (rbar low high position if change >= 0, barwidth(.62) color(maroon%90)) ///
    (rbar low high position if change < 0, barwidth(.62) color(navy%80)) ///
    (pcspike connector connector_left connector connector_right if position < 5, lcolor(gs10) lwidth(thin)) ///
    (scatter label_y position, msymbol(none) mlabel(label) mlabposition(0) mlabcolor(black) mlabsize(vlarge)), ///
    xlabel(0 "{&beta}{subscript:b}" 1 "Jensen" 2 "Weights" 3 "Entry" 4 "Exit" 5 "{&beta}{subscript:d}", nogrid labsize(vlarge)) ///
    ylabel(0(2)8, nogrid angle(0) labsize(large)) ///
    yscale(range(0 8)) xscale(range(-.6 5.6)) ///
    xtitle("") ytitle("Coefficient", size(large)) legend(off) ///
    graphregion(color(white)) bgcolor(white) plotregion(margin(medium)) ///
    xsize(9) ysize(2.2) name(disaggregation_gap, replace)
graph export "`result'/figures/disaggregation_gap.pdf", as(pdf) replace
graph export "`result'/figures/disaggregation_gap.png", as(png) width(2000) replace
graph save "`result'/figures/disaggregation_gap.gph", replace

* Adapt the coefplot settings from 301_Graphs.do, lines 69-89 and 175-182.
* The dividend-paper regressions are not rerun on this unrelated dataset.
* Posting the saved coefficient/variance matrices is for plotting only.
import delimited using "`result'/balance_1978.csv", clear varnames(1) asdouble
assert _N == 5
matrix balance_b = J(1,5,0)
matrix balance_V = J(5,5,0)
forvalues i=1/5 {
    matrix balance_b[1,`i'] = beta[`i']
    matrix balance_V[`i',`i'] = se[`i']^2
}
matrix colnames balance_b = employment jobs_per_resident income_per_resident manufacturing concentration
matrix colnames balance_V = employment jobs_per_resident income_per_resident manufacturing concentration
matrix rownames balance_V = employment jobs_per_resident income_per_resident manufacturing concentration
capture program drop post_saved_balance
program define post_saved_balance, eclass
    ereturn post balance_b balance_V, obs(110088) dof(50)
    ereturn local cmd "post_saved_balance"
end
post_saved_balance
estimates store Balance1978
assert e(df_r) == 50

local bar levels(99 95) pstyle(p1) graphregion(color(white)) plotregion(margin(medium))
local symbolS msymbol(S) msize(medsmall)
local line xline(0) mfcolor(white) drop(_cons)
local Navy mcolor(navy) ciopts(lwidth(1 ..) lcolor(navy*.4 navy*1))
coefplot ///
    (Balance1978, `line' `Navy' `symbolS' `bar'), ///
    coeflabels(employment="Employment (size)" jobs_per_resident="Employment per resident" ///
        income_per_resident="Income per resident" manufacturing="Manufacturing share" ///
        concentration="Industry concentration") ///
    graphregion(color(white)) bgcolor(white) legend(off) ///
    xsize(5.5) ysize(3.2) xlabel(-2(1)4, nogrid) grid(none) ///
    xtitle("Coefficient on regional exposure") name(covariate_balance, replace)
graph export "`result'/figures/covariate_balance.pdf", as(pdf) replace
graph export "`result'/figures/covariate_balance.png", as(png) width(2000) replace
graph save "`result'/figures/covariate_balance.gph", replace
display "STATA_RECAP_GRAPHICS_PASS"
log close
