version 17.0
clear all
set more off
args projectroot
if `"`projectroot'"' == "" local projectroot "C:/Users/pbeaum/Beaumont Dropbox/Paul Beaumont/Trade_midpoint"
local box "`projectroot'/1_current_paper/paper/_paper/paul_rewrite_sandbox"
local result "`box'/assets/fiscal_application"
capture log close
log using "`projectroot'/batch_logs/paper_fiscal_disaggregation_revision/graphics_detail.log", text replace

* Presentation only: saved components, original endpoint = Panel A column 4.
* Figure 1 palette from Paul's 301_Graphs.do; no regression is rerun.
set scheme s2color
import delimited using "`result'/disaggregation_waterfall.csv", clear varnames(1) stringcols(2 6) asdouble
assert _N == 6
assert abs(change[1]-6.123090808742639) < 1e-9
assert abs(change[6]-3.983928961174704) < 1e-9
twoway ///
    (rbar low high position if change >= 0, barwidth(.62) color(maroon%90)) ///
    (rbar low high position if change < 0, barwidth(.62) color(navy%80)) ///
    (pcspike connector connector_left connector connector_right if position < 5, lcolor(gs10) lwidth(thin)) ///
    (scatter label_y position, msymbol(none) mlabel(label) mlabposition(0) mlabcolor(black) mlabsize(medlarge)), ///
    xlabel(0 `""{&beta}{subscript:b}" "(col. 1, Panel A)""' 1 "Jensen" 2 "Weights" 3 "Entry" 4 "Exit" ///
           5 `""{&beta}{subscript:d}" "(col. 4, Panel A)""', nogrid labsize(medium)) ///
    ylabel(0(2)8, nogrid angle(0) labsize(medium)) ///
    yscale(range(0 8.3)) xscale(range(-.7 5.7)) ///
    xtitle("") ytitle("Coefficient", size(medium)) legend(off) ///
    graphregion(color(white)) bgcolor(white) plotregion(margin(medium)) ///
    xsize(6.5) ysize(3.5) name(disaggregation_gap, replace)
graph export "`result'/disaggregation_gap_portrait.pdf", as(pdf) replace
graph export "`result'/disaggregation_gap_portrait.png", as(png) width(2200) replace
graph save "`result'/disaggregation_gap_portrait.gph", replace
display "STATA_DISAGGREGATION_REVISION_PASS"
log close
