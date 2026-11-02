"""Paul's consolidated report. Read saved estimates; never build or estimate."""
import csv
import hashlib
import json
from pathlib import Path

from scipy.stats import t as student_t

BOX = Path(__file__).resolve().parents[1]
SRC = BOX / 'source/results_recap_v3'
OUT = BOX / 'output/results_recap_v3'
SCALE = 100000


def sha(p):
    with Path(p).open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def rows(p):
    with Path(p).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def certified(folder, name='estimates.json'):
    d = read(folder / name)
    assert d['status'] == 'PASS', folder
    for file, digest in d.get('files', {}).items():
        assert sha(folder / file) == digest, file
    return d


def row(label, values):
    return label + ' & ' + ' & '.join(values) + r' \\'


def coef(b, p):
    stars = '***' if float(p) < .01 else '**' if float(p) < .05 else '*' if float(p) < .1 else ''
    return f'{float(b) * SCALE:.3f}' + (r'$^{' + stars + '}$' if stars else '')


def number(x):
    return f'{int(float(x)):,}'


def emit(name, lines):
    (SRC / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def table_start(n):
    if n == 10:
        return [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l' + 'c' * n + r'@{}}', r'\toprule']
    return [r'\begin{tabular}{@{}l' + 'c' * n + r'@{}}', r'\toprule']


def table_end(n):
    return r'\bottomrule\end{tabular*}' if n == 10 else r'\bottomrule\end{tabular}'


def emit_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    # Preserve every previous report, source and result used by this renderer.
    previous = {}
    for sub in ['output', 'source']:
        for p in (BOX / sub).rglob('*'):
            if p.is_file() and 'results_recap_v3' not in p.parts and p.name != 'NS_results_recap_v3.pdf':
                previous[str(p)] = sha(p)
    SRC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)

    aggdir = BOX / 'output/aggregation_summary_tables'
    certified(aggdir, 'certificate.json')
    sepdir = BOX / 'output/heterogeneity_separate'
    certified(sepdir)
    baldir = BOX / 'output/balance_reference_2000'
    certified(baldir)
    h3dir = BOX / 'output/h3_analysis'
    h3 = certified(h3dir)
    assert sha(BOX / 'provenance/original_sandbox/h3_analysis.py') == h3['estimator_sha256']
    assert abs(h3['decomposition_error']) < 1e-12
    disdir = BOX / 'output/disaggregation_no_population'
    provenance = certified(disdir, 'provenance.json')
    assert sha(disdir / 'displayed_estimates.csv') == provenance['displayed_estimates_sha256']
    dis = sorted([r for r in rows(disdir / 'displayed_estimates.csv')
                  if int(r['table']) == 5 and int(r['horizon']) == 3], key=lambda r: int(r['column']))
    assert [int(r['column']) for r in dis] == list(range(1, 11))
    # The display export omits p and cell counts for Table 5. Recover counts
    # from its hash-pinned saved coefficient files and p from the existing t rule.
    appcoefs = BOX / 'reference/coefficient_files'
    counts_sources = {}
    for name in ['disaggregation_h234_complete.csv', 'log_lag_full_county.csv',
                 'log_naics6.csv', 'tables45_new.csv', 'table2_new.csv']:
        assert sha(appcoefs / name) == provenance['inputs'][name]
        counts_sources[name] = rows(appcoefs / name)
    for r in dis:
        col = int(r['column'])
        source = ('disaggregation_h234_complete.csv' if col in [1, 6, 7, 8] else
                  'log_lag_full_county.csv' if col in [2, 3] else
                  'log_naics6.csv' if col == 4 else
                  'tables45_new.csv' if col == 5 else 'table2_new.csv')
        candidates = [s for s in counts_sources[source] if int(s['horizon']) == 3
                      and s['rung'] == r['rung'] and s['growth'] == r['growth']
                      and (source not in ['log_naics6.csv', 'tables45_new.csv'] or s['weight'] == 'lag')
                      and (source != 'table2_new.csv' or s['industry_year_fe'] == r['industry_year_fe'])]
        assert len(candidates) == 1, (col, candidates)
        s = candidates[0]
        assert abs(float(s['beta'])-float(r['beta'])) < 1e-12
        r['observations'] = s['n_estimation_cells']
        r['pvalue'] = 2*student_t.sf(abs(float(r['t'])), 3057)
    assert abs(float(dis[0]['beta']) - h3['fits']['baseline_full']['beta']) < 1e-12
    assert abs(float(dis[3]['beta']) - h3['fits']['original']['beta']) < 1e-12
    agg = {r['spec']: r for r in rows(aggdir / 'selected_results.csv')}
    fits = {(r['spec'], r['model']): r for r in rows(aggdir / 'fit_statistics.csv')}
    delta = {int(r['controls']): r for r in rows(sepdir / 'delta.csv')}
    control_coefs = {(int(r['controls']), r['model'], r['variable']): r
                     for r in rows(sepdir / 'coefficients.csv')}
    design = {(int(r['controls']), r['model']): r for r in rows(sepdir / 'design_audit.csv')}
    assert abs(float(delta[0]['delta']) - float(agg['main']['exposure_beta'])) < 1e-12

    # All ten original columns retained; panel B alone.
    lines = table_start(10) + [
        row('Growth rate', [r'\multicolumn{5}{c}{Log}', r'\multicolumn{5}{c}{MPGR}']),
        r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}',
        row('', ['', r'\multicolumn{4}{c}{County $\times$}', '', r'\multicolumn{4}{c}{County $\times$}']),
        row('Level', ['County', 'NAICS2', 'NAICS4', 'NAICS6', 'NAICS6'] * 2),
        row('', [f'({i})' for i in range(1, 11)]), r'\midrule',
        row('County spending', [coef(r['beta'], r['pvalue']) for r in dis]),
        row('', [f'({float(r["se"]) * SCALE:.3f})' for r in dis]),
        row(r'\% deviation', [f'{float(r["percent_deviation"]):.1f}' + r'\%' for r in dis]),
        r'\addlinespace[4pt]',
        row('County FE', ['Yes'] * 10), row('Year FE', ['Yes'] * 10),
        row(r'Industry $\times$ year FE', ['--'] * 4 + ['Yes'] + ['--'] * 4 + ['Yes']),
        r'\addlinespace[4pt]', row(r'\textit{First-stage F}', [''] * 10),
        row(r'\hspace{1em}County spending', [f'{float(r["F"]):.2f}' for r in dis]),
        r'\addlinespace[7pt]',
        row('Observations', [number(r['observations']) for r in dis]),
        row('County-years', [number(r['county_years']) for r in dis]),
        row('Clusters', ['3,058'] * 10), r'\bottomrule\end{tabular*}']
    emit('disaggregation.tex', lines)

    a = agg['main']
    lines = table_start(4) + [
        row('', ['Baseline', 'Aggregate', 'Spillover', 'Naive gap']),
        row('', [r'$\beta_b$', r'$\beta_a$', r'$\delta$', r'$\delta^{\mathrm{naive}}$']),
        row('', [f'({i})' for i in range(1, 5)]), r'\midrule',
        row('', [coef(a[k + '_beta'], a[k + '_p']) for k in ['augmented', 'aggregate', 'exposure']]
            + [coef(a['gap'], a['gap_p'])]),
        row('', [f'({float(a[k + "_se"]) * SCALE:.3f})' for k in ['augmented', 'aggregate', 'exposure']]
            + [f'({float(a["gap_se"]) * SCALE:.3f})']), r'\addlinespace[12pt]',
        row(r'\textit{First-stage F}', [''] * 4),
        row(r'\hspace{1em}County spending', [f'{float(a["augmented_F"]):.2f}', '--', f'{float(a["augmented_F"]):.2f}', '--']),
        row(r'\hspace{1em}Regional exposure', [f'{float(a["exposure_F"]):.2f}', f'{float(a["aggregate_F"]):.2f}', f'{float(a["exposure_F"]):.2f}', '--']),
        r'\addlinespace[12pt]',
        row('Observations', [number(fits[('main', 'augmented')]['n']), number(fits[('main', 'aggregate')]['n']), number(fits[('main', 'augmented')]['n']), '--']),
        row('Clusters', ['51'] * 4), table_end(4)]
    emit('aggregation_main.tex', lines)

    # Paul requested exchanging the previous second and fifth columns.
    order = ['unweighted', 'log', 'h4', 'cea', 'h2']
    alternatives = [agg[k] for k in order]
    lines = table_start(5) + [
        row('', [r'\multicolumn{5}{c}{Spillover ($\delta$)}']),
        r'\cmidrule(lr){2-6}',
        row('', [f'({i})' for i in range(1, 6)]), r'\midrule',
        row('', [coef(r['exposure_beta'], r['exposure_p']) for r in alternatives]),
        row('', [f'({float(r["exposure_se"]) * SCALE:.3f})' for r in alternatives]),
        r'\addlinespace[8pt]',
        row('Growth rate', ['Standard', 'Log', r'\multicolumn{3}{c}{Standard}']),
        r'\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-6}',
        row('Horizon $h$', ['3', '3', '4', '3', '2']),
        row('Aggregation', ['State', 'State', 'State', 'CEA', 'State']),
        row('County weights', ['No', 'Yes', 'Yes', 'Yes', 'Yes']),
        r'\addlinespace[12pt]', row(r'\textit{First-stage F}', [''] * 5),
        row(r'\hspace{1em}County spending', [f'{float(r["augmented_F"]):.2f}' for r in alternatives]),
        row(r'\hspace{1em}Regional exposure', [f'{float(r["exposure_F"]):.2f}' for r in alternatives]),
        r'\addlinespace[12pt]',
        row('Observations', [number(fits[(k, 'augmented')]['n']) for k in order]),
        row('Clusters', [number(fits[(k, 'augmented')]['clusters']) for k in order]),
        table_end(5)]
    emit('aggregation_alternatives.tex', lines)

    control_order = [0, 1, 3, 2]
    lines = table_start(4) + [
        row('', [r'\shortstack{No controls\\\strut}', r'\shortstack{Employment\\(size)}', r'\shortstack{Manufacturing\\share}', r'\shortstack{Income per\\resident}']),
        row('', [r'$\delta$'] * 4), row('', [f'({i})' for i in range(1, 5)]), r'\midrule',
        row('', [coef(delta[i]['delta'], delta[i]['p']) for i in control_order]),
        row('', [f'({float(delta[i]["se"]) * SCALE:.3f})' for i in control_order]),
        row('$p$-value', [f'{float(delta[i]["p"]):.3f}' for i in control_order]),
        r'\addlinespace[8pt]', row(r'\textit{First-stage F}', [''] * 4),
        row(r'\hspace{1em}County spending', [f'{float(control_coefs[(i, "county", "x")]["F"]):.2f}' for i in control_order]),
        row(r'\hspace{1em}Regional exposure', [f'{float(control_coefs[(i, "county", "x_agg")]["F"]):.2f}' for i in control_order]),
        row(r'\hspace{1em}\shortstack[l]{Characteristic $\times$\\county spending}', ['--'] + [f'{float(control_coefs[(i, "county", "xc" + str(i))]["F"]):.2f}' for i in control_order[1:]]),
        row(r'\hspace{1em}\shortstack[l]{Regional characteristic\\interaction}', ['--'] + [f'{float(control_coefs[(i, "county", "Q" + str(i))]["F"]):.2f}' for i in control_order[1:]]),
        r'\addlinespace[12pt]',
        row('Observations', [number(design[(i, 'county')]['n']) for i in control_order]),
        row('Clusters', [number(design[(i, 'county')]['G']) for i in control_order]),
        table_end(4)]
    emit('aggregation_controls.tex', lines)

    parts = {r['component']: float(r['beta']) * SCALE for r in h3['components']}
    baseline = h3['fits']['baseline_full']['beta'] * SCALE
    disagg = h3['fits']['original']['beta'] * SCALE
    assert abs(sum(parts.values()) - (baseline - disagg)) < 1e-8
    # Export certified arithmetic for Stata graphics. Moving b -> d reverses
    # the signs of the earlier d -> b accounting; no fitted coefficient changes.
    bars = []
    def bar(position, name, before, after, value, endpoint=False):
        bars.append(dict(position=position, component=name, low=min(before, after),
                         high=max(before, after), change=value,
                         label=f'{value:.3f}' if endpoint else f'{value:+.3f}',
                         label_y=max(before, after)+.24, connector=after,
                         connector_left=position+.31, connector_right=position+.69))
    bar(0, 'baseline', 0, baseline, baseline, True)
    level = baseline
    for i, key in enumerate(['jensen', 'renormalization', 'entry', 'exit'], 1):
        change = -parts[key]
        bar(i, key, level, level+change, change)
        level += change
    bar(5, 'disaggregated', 0, disagg, disagg, True)
    assert abs((level-disagg)-parts['sample']) < 1e-8
    emit_csv(OUT/'waterfall.csv', bars)

    # Uniform presentation, initial characteristics only, and no population.
    b = {(int(r['year']), r['characteristic']): r for r in rows(baldir / 'balance.csv')}
    keys = ['employment', 'jobs_per_resident', 'income_per_resident', 'manufacturing', 'concentration']
    labels = ['Employment (size)', 'Employment per resident',
              'Income per resident', 'Manufacturing share', 'Industry concentration']
    emit_csv(OUT/'balance_1978.csv', [dict(characteristic=k, label=label,
             beta=b[(1978,k)]['pi'], se=b[(1978,k)]['se'], df=50)
             for k, label in zip(keys, labels)])

    doc = [r'\pdfminorversion=6', r'\documentclass[11pt,letterpaper]{article}',
        r'\usepackage[margin=.7in]{geometry}',
        r'\usepackage[T1]{fontenc}\usepackage{lmodern}',
        r'\usepackage{amsmath,booktabs,array,graphicx,pdflscape,fancyhdr}',
        r'\usepackage[hidelinks]{hyperref}',
        r'\setlength{\parindent}{0pt}\setlength{\parskip}{8pt}',
        r'\renewcommand{\arraystretch}{1.10}',
        r'\pagestyle{fancy}\fancyhf{}\fancyfoot[L]{\footnotesize Paul\textquotesingle s sandbox | NS instrument}\fancyfoot[R]{\footnotesize\thepage}\renewcommand{\headrulewidth}{0pt}',
        r'\newcommand{\heading}[2]{{\Large #1}\par{\small #2}\par\vspace{8pt}}',
        r'\newcommand{\starsnote}{$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}',
        r'\begin{document}\begin{landscape}\setlength{\parskip}{4pt}',
        r'\heading{Disaggregation: $h=3$}{Table D1. Original Table 5, Panel B | Lag-share log weights | No population multiplier | IV}',
        r'{\fontsize{9}{10.5}\selectfont\renewcommand{\arraystretch}{1.08}\setlength{\tabcolsep}{2pt}\input{disaggregation.tex}}',
        r'{\fontsize{8.5}{10}\selectfont\textit{Notes.} Employment growth is the outcome. Log weights are $E_{cm,t-3}/E_{c,t-3}$, without survivor renormalization; MPGR weights are $(E_{cmt}+E_{cm,t-3})/(E_{ct}+E_{c,t-3})$. County spending and its NS-style instrument repeat across industries. Parentheses contain county-clustered SEs; coefficients and SEs use the existing $10^5$ display multiplier. \starsnote\ Observations count county-industry rows in cell regressions; county-years count distinct county-year pairs. Industry-by-year FE subsume year FE.}\par',
        r'{\small Figure D1. From the county coefficient $\beta_b$ to the disaggregated coefficient $\beta_d$: columns (1) and (4)}\par',
        r'\begin{center}\includegraphics[width=.86\linewidth,height=2.05in,keepaspectratio]{../../output/results_recap_v3/figures/disaggregation_gap.pdf}\end{center}',
        r'{\fontsize{8.5}{10}\selectfont\textit{Notes.} Bars show successive changes from $\widehat\beta_b$ to $\widehat\beta_d$. Positive changes are red; negative changes are blue. Weights captures survivor-share renormalization. Jensen, entry and exit use the same common-sample county IV operator. A small remainder of ' + f'{parts["sample"]:.3f}' + r' from 11 county-years without surviving NAICS6 cells is omitted from the graph; both endpoint coefficients retain their original values.}\par',
        r'\end{landscape}\setlength{\parskip}{8pt}\clearpage',
        r'\heading{Aggregation}{Table A1. Main results | Standard growth | $h=3$ | States | Weighted IV}',
        r'\begin{center}\small\setlength{\tabcolsep}{7pt}\input{aggregation_main.tex}\end{center}',
        r'\vspace{5pt}{\small\textit{Notes.} Parentheses contain state-clustered SEs. Coefficients and SEs use the $10^5$ display multiplier. \starsnote\ The sample contains 3,058 counties and 51 state groups, 1981--2016.}',
        r'Baseline and spillover coefficients come from the same augmented county regression:',
        r'\[g_{ct}=\alpha_c+\lambda_t+\beta_b x_{ct}+\delta X_{st}+u_{ct}.\]',
        r'The aggregate coefficient comes from a separately estimated state regression:',
        r'\[g_{st}=\alpha_s+\lambda_t+\beta_a X_{st}+v_{st}.\]',
        r'County spending is $x_{ct}$; regional exposure is $X_{st}$. Both are instrumented with the county and regional NS-style instruments. Standard growth is $g_{ct}=(E_{ct}-E_{c,t-3})/E_{c,t-3}$. County regression weights are $\omega_{cst}=E_{c,t-3}/E_{s,t-3}$; regional exposure and its instrument use the same shares. Each state-year receives total weight one, and state regressions are equally weighted.',
        r'The naive gap is $\widehat\delta^{\mathrm{naive}}=\widehat\beta_a-\widehat\beta_b^{\mathrm{naive}}$, where the naive county regression omits regional exposure. Its SE accounts for covariance between the two coefficients. It combines 110,088 county-years and 1,836 state-years, so there is no separate observation count or first stage for that difference.',
        r'{\small Both first-stage rows repeat under Baseline and Spillover because these columns share one regression. Each ordinary clustered F tests both excluded instruments jointly for the indicated endogenous variable. The state equation has one excluded instrument. The remaining coefficient-identity residual is ' + f'{float(a["identity_residual"])*SCALE:.3f}' + r' on the display scale. The spillover interpretation requires the draft\textquotesingle s condition excluding exposure-related treatment heterogeneity.}',
        r'\clearpage',
        r'\heading{Aggregation}{Table A2. Alternative specifications | IV}',
        r'\begin{center}\small\setlength{\tabcolsep}{7pt}\input{aggregation_alternatives.tex}\end{center}',
        r'\vspace{5pt}{\small\textit{Notes.} Parentheses contain SEs clustered by state, except column (4), which clusters by component economic area (CEA). Coefficients and SEs use the $10^5$ display multiplier. \starsnote}',
        r'Each column reports the regional-exposure coefficient from the augmented county regression with county and year FE. Relative to Table A1, column (1) removes county regression weights and column (2) uses log growth. These two columns illustrate the weighting and Jensen channels. Column (3) changes the horizon to four years; column (4) changes the geography to component economic areas; and column (5) changes the horizon to two years.',
        r'County weights are lagged employment shares when marked Yes, and equal county-year weights when marked No. Regional exposure and its instrument retain employment-share aggregation in every column.',
        r'All columns retain the same 3,058 counties. Samples cover 1980--2016 for $h=2$, 1981--2016 for $h=3$, and 1982--2016 for $h=4$. The two first-stage rows are ordinary clustered tests of the two excluded instruments for county spending and regional exposure. Differences in significance do not themselves test differences between coefficients.',
        r'\clearpage',
        r'\heading{Aggregation}{Table A3. One heterogeneity control at a time | Standard growth | $h=3$ | States | Weighted IV}',
        r'\begin{center}\small\setlength{\tabcolsep}{6pt}\input{aggregation_controls.tex}\end{center}',
        r'\begingroup\small',
        r'\vspace{5pt}{\small\textit{Notes.} Only the spillover coefficient $\delta$ is reported. Parentheses contain state-clustered SEs; p-values use $t_{50}$. Coefficients and SEs use the $10^5$ display multiplier. \starsnote}',
        r'Each control enters separately. Employment enters linearly in levels; income means real income per resident; manufacturing is its employment share. All three characteristics use 1978 values and are standardized across the same counties. Income is winsorized at the national 1st and 99th county percentiles before standardization.',
        r'For each characteristic $C_c$, the county regression is',
        r'\[g_{ct}=\alpha_c+\lambda_t+\beta_bx_{ct}+\delta X_{st}+\theta x_{ct}C_c+\eta Q_{st}+\kappa A_{st}+u_{ct},\]',
        r'\[Q_{st}=\sum_{c\in s}\omega_{cst}x_{ct}C_c,\qquad A_{st}=\sum_{c\in s}\omega_{cst}C_c.\]',
        r'The four endogenous variables $(x_{ct},X_{st},x_{ct}C_c,Q_{st})$ are instrumented by',
        r'\[z_{ct},\quad Z_{st},\quad z_{ct}C_c,\quad \sum_{c\in s}\omega_{cst}z_{ct}C_c.\]',
        r'$A_{st}$ is included as an exogenous control; the fixed county characteristic is absorbed by county FE. The uncontrolled column uses only the two original instruments.',
        r'{\small Each ordinary first-stage F tests all excluded instruments jointly for the indicated endogenous variable. The county interaction is $x_{ct}C_c$. The regional characteristic interaction is $Q_{st}=\sum_c\omega_{cst}x_{ct}C_c$; it is not generally equal to the product of regional exposure and the regional mean characteristic. Their interpretation relies on the conditional exclusion assumptions for the interaction instruments.}',
        r'Income alone leaves the point estimate near the uncontrolled result but reduces precision. Manufacturing alone raises the point estimate. These comparisons use the same 110,088 observations and 51 state clusters.',
        r'\endgroup',
        r'\clearpage',
        r'\heading{Covariate balance}{Figure A1. Regional exposure and 1978 county characteristics | States | $h=3$}',
        r'\begin{center}\includegraphics[width=.88\linewidth]{../../output/results_recap_v3/figures/covariate_balance.pdf}\end{center}',
        r'{\small\textit{Notes.} Square markers show $\widehat\pi_C$; pale and dark intervals show 99\% and 95\% confidence intervals, respectively. Characteristics use 1978 values standardized across 3,058 counties; income is winsorized before standardization. No $10^5$ multiplier applies to these diagnostic coefficients.}',
        r'For each fixed county characteristic $C_c^{(r)}$, the balance diagnostic estimates',
        r'\[x_{ct}C_c^{(r)}=\alpha_c+\lambda_t+\rho_C^{(r)}x_{ct}+\pi_C^{(r)}X_{st}+u_{ct}^{(r)}.\]',
        r'County spending and regional exposure are instrumented with $z_{ct}$ and $Z_{st}$. The test is $H_0:\pi_C^{(r)}=0$. Weights are lagged employment shares; FE are county and year; SEs are clustered by state. The sample remains 1981--2016, with 110,088 observations and 51 clusters. Although $C_c^{(r)}$ is fixed, its product with county spending varies over time.',
        r'Non-rejection of balance does not rule out unobserved treatment heterogeneity. The figure motivates the illustrative control comparisons without treating them as a complete adjustment for heterogeneity.',
        r'\end{document}']
    doc = [line + r'\par' if (not line.startswith('\\') and not line.endswith(r'\par')) else line for line in doc]
    emit('NS_results_recap_v3.tex', doc)

    for path, digest in previous.items():
        assert sha(path) == digest, f'Previous artifact changed: {path}'
    audit = {'status': 'PASS', 'operation': 'report_only', 'new_regression_fits': 0,
             'disaggregation_table': 5, 'disaggregation_panel': 'B', 'horizon': 3,
             'decomposition_gap': (baseline-disagg)/SCALE,
             'decomposition_error': h3['decomposition_error'],
             'aggregation_alternative_order': order, 'control_reference_year': 1978,
             'heterogeneity_control_order': control_order,
             'balance_reference_years': [1978],
             'balance_characteristics': keys,
             'omitted_graph_remainder': parts['sample']/SCALE,
             'graph_direction': 'baseline_to_disaggregated',
             'graph_engine': 'Stata; run graph_results.do after this renderer',
             'previous_artifacts_preserved': len(previous),
             'inputs': {str(p): sha(p) for p in [
                 disdir/'displayed_estimates.csv', h3dir/'estimates.json',
                 aggdir/'selected_results.csv', aggdir/'fit_statistics.csv',
                 sepdir/'delta.csv', sepdir/'coefficients.csv', baldir/'balance.csv',
                 BOX/'code/graph_results.do']},
             'files': {str(p.relative_to(BOX)): sha(p) for root in [SRC, OUT]
                       for p in root.rglob('*') if p.is_file() and p.name != 'provenance.json'}}
    (OUT/'provenance.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(f'Report source ready. {len(previous)} previous artifacts unchanged. No new regressions.')


if __name__ == '__main__':
    main()
