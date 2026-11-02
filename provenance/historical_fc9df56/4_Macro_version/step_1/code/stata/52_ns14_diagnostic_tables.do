version 18.0
clear all
set more off
set varabbrev off

args mode
local standalone = ("`mode'" != "parent")
local repo "/Users/f1azmm00/Research/aggregation-disaggregation-invariance"
local dropbox "/Users/f1azmm00/Library/CloudStorage/Dropbox-Matray/_AdrienTeam/Chenzi_Adrien/Invariance_Agg_Disagg"
local results "`repo'/4_Macro_version/Results/ns14_diagnostics"
local temp "`dropbox'/4_Macro_version/data/temp/ns14_diagnostics"
local logs "`repo'/batch_logs/macro_version"
local rowend "\\"

capture mkdir "`results'"
capture mkdir "`logs'"

if `standalone' {
    capture log close
    log using "`logs'/52_ns14_diagnostic_tables.log", text replace
}

* Published comparison and replication result.
use "`temp'/ns14_replication.dta", clear
assert _N == 1
assert N == 1989
assert states == 51
local published_beta = strtrim(strofreal(published_beta[1], "%9.3f"))
local published_se = strtrim(strofreal(published_se[1], "%9.3f"))
local beta = strtrim(strofreal(beta[1], "%9.3f"))
local se = strtrim(strofreal(se[1], "%9.3f"))
local N = strtrim(strofreal(N[1], "%12.0fc"))
local states = strtrim(strofreal(states[1], "%12.0fc"))

file open frep using "`results'/Table_NS_replication.tex", write replace
file write frep "\begin{table}[!htbp]" _n
file write frep "\caption{Nakamura-Steinsson Employment Replication}" _n
file write frep "\label{tab:ns_replication}" _n
file write frep "{\centering" _n
file write frep "\adjustbox{max width=\linewidth}{%" _n
file write frep "\begin{tabular}{lrr}" _n
file write frep "\toprule" _n
file write frep " & Published & Replication `rowend'" _n
file write frep "\midrule" _n
file write frep "Procurement multiplier & `published_beta' & `beta' `rowend'" _n
file write frep " & (`published_se') & (`se') `rowend'" _n
file write frep "\midrule\addlinespace" _n
file write frep "\textit{Fixed Effects} & & `rowend'" _n
file write frep "\hspace{1.5em}State & " (char(36)) "\checkmark" (char(36)) " & " (char(36)) "\checkmark" (char(36)) " `rowend'" _n
file write frep "\hspace{1.5em}Year & " (char(36)) "\checkmark" (char(36)) " & " (char(36)) "\checkmark" (char(36)) " `rowend'" _n
file write frep "\addlinespace Observations & 1,989 & `N' `rowend'" _n
file write frep "Number of states & 51 & `states' `rowend'" _n
file write frep "\bottomrule" _n
file write frep "\end{tabular}}" _n
file write frep "\par}" _n
file write frep "\medskip" _n
file write frep "{\scriptsize \noindent\textit{Notes:} The dependent variable is the two-year growth rate of the state employment-to-population ratio. State procurement growth is instrumented with the predetermined 1966--1971 procurement exposure interacted with the national procurement change. Standard errors are clustered by state. The published estimate is from Nakamura and Steinsson (2014), Table III, employment row.\par}" _n
file write frep "\end{table}" _n
file close frep

* Reduced-form lead tests.
use "`temp'/ns14_lead_placebos.dta", clear
sort column
assert _N == 5
file open flead using "`results'/Table_NS_parallel_trends.tex", write replace
file write flead "\begin{table}[!htbp]" _n
file write flead "\caption{Lead Tests for the Single-Factor Instrument}" _n
file write flead "\label{tab:ns_parallel_trends}" _n
file write flead "{\centering" _n
file write flead "\adjustbox{max width=\linewidth}{%" _n
file write flead "\begin{tabular}{lrrrrr}" _n
file write flead "\toprule" _n
file write flead " & Current & Lead 2 & Leads 2 and 4 & Even years & Odd years `rowend'" _n
file write flead "\midrule" _n
file write flead "Current instrument"
forvalues j = 1/5 {
    local value = strtrim(strofreal(b0[`j'], "%9.3f"))
    file write flead " & `value'"
}
file write flead " `rowend'" _n
file write flead ""
forvalues j = 1/5 {
    local value = strtrim(strofreal(se0[`j'], "%9.3f"))
    file write flead " & (`value')"
}
file write flead " `rowend'[2pt]" _n
file write flead "Instrument lead, two years"
forvalues j = 1/5 {
    if missing(b2[`j']) file write flead " & "
    else {
        local value = strtrim(strofreal(b2[`j'], "%9.3f"))
        file write flead " & `value'"
    }
}
file write flead " `rowend'" _n
file write flead ""
forvalues j = 1/5 {
    if missing(se2[`j']) file write flead " & "
    else {
        local value = strtrim(strofreal(se2[`j'], "%9.3f"))
        file write flead " & (`value')"
    }
}
file write flead " `rowend'[2pt]" _n
file write flead "Instrument lead, four years"
forvalues j = 1/5 {
    if missing(b4[`j']) file write flead " & "
    else {
        local value = strtrim(strofreal(b4[`j'], "%9.3f"))
        file write flead " & `value'"
    }
}
file write flead " `rowend'" _n
file write flead ""
forvalues j = 1/5 {
    if missing(se4[`j']) file write flead " & "
    else {
        local value = strtrim(strofreal(se4[`j'], "%9.3f"))
        file write flead " & (`value')"
    }
}
file write flead " `rowend'" _n
file write flead "\midrule" _n
file write flead "Joint lead-test " (char(36)) "p" (char(36)) "-value"
forvalues j = 1/5 {
    if missing(p_joint[`j']) file write flead " & "
    else if p_joint[`j'] < 0.001 file write flead " & " (char(36)) "<0.001" (char(36))
    else {
        local value = strtrim(strofreal(p_joint[`j'], "%9.3f"))
        file write flead " & `value'"
    }
}
file write flead " `rowend'" _n
file write flead "\midrule\addlinespace" _n
file write flead "\textit{Fixed Effects} & & & & & `rowend'" _n
file write flead "\hspace{1.5em}State"
forvalues j = 1/5 {
    file write flead " & " (char(36)) "\checkmark" (char(36))
}
file write flead " `rowend'" _n
file write flead "\hspace{1.5em}Year"
forvalues j = 1/5 {
    file write flead " & " (char(36)) "\checkmark" (char(36))
}
file write flead " `rowend'" _n
file write flead "\addlinespace Observations"
forvalues j = 1/5 {
    local value = strtrim(strofreal(N[`j'], "%12.0fc"))
    file write flead " & `value'"
}
file write flead " `rowend'" _n
file write flead "Number of states"
forvalues j = 1/5 {
    local value = strtrim(strofreal(states[`j'], "%12.0fc"))
    file write flead " & `value'"
}
file write flead " `rowend'" _n
file write flead "\bottomrule" _n
file write flead "\end{tabular}}" _n
file write flead "\par}" _n
file write flead "\medskip" _n
file write flead "{\scriptsize \noindent\textit{Notes:} The dependent variable is the two-year growth rate of the state employment-to-population ratio. All columns are reduced-form regressions. The instrument is predetermined state procurement exposure interacted with the national procurement change. Columns 4 and 5 retain alternating end years so the two-year outcome windows do not overlap within each column. Standard errors are clustered by state. Lead coefficients can reflect anticipation or persistence as well as a failure of parallel trends.\par}" _n
file write flead "\end{table}" _n
file close flead

* Figure wrapper.
confirm file "`results'/Figure_NS_exposure_gradient.pdf"
file open ffig using "`results'/Figure_NS_exposure_gradient.tex", write replace
file write ffig "\begin{figure}[!htbp]" _n
file write ffig "\caption{Employment Exposure Gradient and National Procurement}" _n
file write ffig "\label{fig:ns_exposure_gradient}" _n
file write ffig "\begin{center}" _n
file write ffig "\includegraphics[width=\linewidth]{Figure_NS_exposure_gradient.pdf}" _n
file write ffig "\end{center}" _n
file write ffig "\vspace{-.5cm}" _n
file write ffig "{\scriptsize \noindent\textit{Notes:} The upper panel reports the coefficient from a separate cross-state regression of two-year employment growth on predetermined procurement exposure in each year. Vertical bars are 95 percent confidence intervals based on heteroskedasticity-robust standard errors. The lower panel reports the two-year national procurement change over lagged national output used as the common shifter.\par}" _n
file write ffig "\end{figure}" _n
file close ffig

* Exposure balance.
use "`temp'/ns14_exposure_balance.dta", clear
sort order
assert _N == 7
assert N == 51
local balance_N = strtrim(strofreal(N[1], "%12.0fc"))
file open fbal using "`results'/Table_NS_exposure_balance.tex", write replace
file write fbal "\begin{table}[!htbp]" _n
file write fbal "\caption{Predetermined Procurement Exposure and State Characteristics}" _n
file write fbal "\label{tab:ns_exposure_balance}" _n
file write fbal "{\centering" _n
file write fbal "\adjustbox{max width=\linewidth}{%" _n
file write fbal "\begin{tabular}{lrrr}" _n
file write fbal "\toprule" _n
file write fbal "Characteristic & Standardized coefficient & Standard error & " (char(36)) "p" (char(36)) "-value `rowend'" _n
file write fbal "\midrule" _n
forvalues i = 1/`=_N' {
    local beta = strtrim(strofreal(beta[`i'], "%9.3f"))
    local se = strtrim(strofreal(se[`i'], "%9.3f"))
    file write fbal (characteristic[`i']) " & `beta' & `se' & "
    if p[`i'] < 0.001 file write fbal (char(36)) "<0.001" (char(36))
    else {
        local pvalue = strtrim(strofreal(p[`i'], "%9.3f"))
        file write fbal "`pvalue'"
    }
    file write fbal " `rowend'" _n
}
file write fbal "\addlinespace \multicolumn{4}{l}{Observations per regression: `balance_N'} `rowend'" _n
file write fbal "\bottomrule" _n
file write fbal "\end{tabular}}" _n
file write fbal "\par}" _n
file write fbal "\medskip" _n
file write fbal "{\scriptsize \noindent\textit{Notes:} Each row is a separate cross-state regression. Both procurement exposure and the listed characteristic are standardized. Characteristics are measured in 1966, the first year of the 1966--1971 exposure window. Heteroskedasticity-robust standard errors are reported. These are balance diagnostics, not tests of the exclusion restriction.\par}" _n
file write fbal "\end{table}" _n
file close fbal

* IV sensitivity.
use "`temp'/ns14_iv_robustness.dta", clear
sort column
assert _N == 5
assert states == 51
local robustness_states = strtrim(strofreal(states[1], "%12.0fc"))
file open frob using "`results'/Table_NS_iv_robustness.tex", write replace
file write frob "\begin{table}[!htbp]" _n
file write frob "\caption{Sensitivity of the Employment Multiplier}" _n
file write frob "\label{tab:ns_iv_robustness}" _n
file write frob "{\centering" _n
file write frob "\adjustbox{max width=\linewidth}{%" _n
file write frob "\begin{tabular}{lrrrr}" _n
file write frob "\toprule" _n
file write frob "Specification & Multiplier & Standard error & First-stage " (char(36)) "F" (char(36)) " & Observations `rowend'" _n
file write frob "\midrule" _n
forvalues i = 1/`=_N' {
    local beta = strtrim(strofreal(beta[`i'], "%9.3f"))
    local se = strtrim(strofreal(se[`i'], "%9.3f"))
    local fs = strtrim(strofreal(fs_F[`i'], "%9.2f"))
    local N = strtrim(strofreal(N[`i'], "%12.0fc"))
    file write frob (specification[`i']) " & `beta' & `se' & `fs' & `N' `rowend'" _n
}
file write frob "\addlinespace \multicolumn{5}{l}{Number of states: `robustness_states'} `rowend'" _n
file write frob "\bottomrule" _n
file write frob "\end{tabular}}" _n
file write frob "\par}" _n
file write frob "\medskip" _n
file write frob "{\scriptsize \noindent\textit{Notes:} The dependent variable is the two-year growth rate of the state employment-to-population ratio. State procurement growth is instrumented with predetermined procurement exposure interacted with the national procurement change. All specifications include state and year fixed effects. Baseline characteristics are measured in 1966 and interacted with year indicators. The oil and interest-rate specifications allow each state to have its own sensitivity to the corresponding macroeconomic series. Standard errors are clustered by state.\par}" _n
file write frob "\end{table}" _n
file close frob

* Alternative-channel reduced forms.
use "`temp'/ns14_alternative_channels.dta", clear
sort order
assert _N == 7
assert states == 51
local channel_states = strtrim(strofreal(states[1], "%12.0fc"))
file open fchan using "`results'/Table_NS_exclusion_stress.tex", write replace
file write fchan "\begin{table}[H]" _n
file write fchan "\caption{Reduced Forms for Alternative Channels}" _n
file write fchan "\label{tab:ns_exclusion_stress}" _n
file write fchan "{\centering" _n
file write fchan "\adjustbox{max width=\linewidth}{%" _n
file write fchan "\begin{tabular}{lrrrr}" _n
file write fchan "\toprule" _n
file write fchan "Outcome & Standardized coefficient & Standard error & " (char(36)) "p" (char(36)) "-value & Observations `rowend'" _n
file write fchan "\midrule" _n
forvalues i = 1/`=_N' {
    local beta = strtrim(strofreal(beta[`i'], "%9.3f"))
    local se = strtrim(strofreal(se[`i'], "%9.3f"))
    local N = strtrim(strofreal(N[`i'], "%12.0fc"))
    file write fchan (outcome[`i']) " & `beta' & `se' & "
    if p[`i'] < 0.001 file write fchan (char(36)) "<0.001" (char(36))
    else {
        local pvalue = strtrim(strofreal(p[`i'], "%9.3f"))
        file write fchan "`pvalue'"
    }
    file write fchan " & `N' `rowend'" _n
}
file write fchan "\addlinespace \multicolumn{5}{l}{Number of states: `channel_states'} `rowend'" _n
file write fchan "\bottomrule" _n
file write fchan "\end{tabular}}" _n
file write fchan "\par}" _n
file write fchan "\medskip" _n
file write fchan "{\scriptsize \noindent\textit{Notes:} Each row is a separate reduced-form regression of the listed outcome on the single-factor instrument. Both the outcome and instrument are standardized within the available estimation sample. All regressions include state and year fixed effects. Standard errors are clustered by state. A nonzero coefficient does not by itself reject exclusion because procurement may cause the listed outcome. The table identifies observable channels through which the instrument may represent a broader bundle of defense and federal activity than recorded procurement alone.\par}" _n
file write fchan "\end{table}" _n
file close fchan

display as result "NS diagnostic tables: `results'"
if `standalone' log close
