"""Prepare, apply and verify the approved cancellation pairs through FY1983.

Every fiscal file has a preserved uncorrected backup. All corrected files are
staged and checked before publication. Reruns are idempotent. Neither raw source
files nor downstream regression caches are rewritten by this program.
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0,str(PIPELINE.parent/'py'))
import dcads_common as dc
import report_corrections as rc

OUT = dc.MACRO_RESULTS_ROOT/'instrument_normalization'
WORK = dc.MACRO_VERSION_TEMP_DIR/'contract_report_correction_application'
BACKUP = WORK/'uncorrected_fiscal_files'
STAGE = WORK/'staged'
STATE = WORK/'application_state.json'
MEMO = dc.MACRO_RESULTS_ROOT.parent/'Instrument_analysis/Cleaning_spendingData/REPORT_CORRECTION_CLEANING.md'


def prepare():
    ledger = pd.read_csv(OUT/'correction_audit_canceled_report_ledger.csv',dtype={'fiscal_file':str})
    proof = pd.read_csv(OUT/'correction_audit_raw_verification.csv')
    a = ledger.merge(proof[['row_id','sha256']],on='row_id',validate='one_to_one')
    if len(a)!=25780 or a.reversal_id.nunique()!=12890 or (a.amount%1000).any():
        raise ValueError('Ledger does not match the approved audit')
    if a.amount.clip(lower=0).sum()!=30011877000:
        raise ValueError('Positive cancellation amount does not match approved audit')
    if not a.fiscal_file.str[:4].astype(int).between(1966,1983).all():
        raise ValueError('Out-of-scope fiscal file')
    a['dollars_reported'] = a.amount//1000
    a = a.rename(columns={'reversal_id':'pair_id','pair_role':'role'})
    a = a[['source_file','record_number','dollars_reported','sha256','pair_id','role']]
    content = a.sort_values(['source_file','record_number']).to_csv(index=False)
    if rc.LEDGER.exists() and rc.LEDGER.read_text()!=content:
        raise ValueError('Refusing to overwrite a different approved ledger')
    rc.LEDGER.write_text(content)
    manifest = dict(n_canceled_records=len(a),n_pairs=a.pair_id.nunique(),
                    first_year=1966,last_year=1983,positive_nominal=30011877000,
                    ledger_sha256=rc.sha256(rc.LEDGER),audit_producer_commit='d271423')
    rc.MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Approved ledger: {len(a):,} records',flush=True)


def apply():
    for path in [BACKUP,STAGE]:
        path.mkdir(parents=True,exist_ok=True)
    ledger = rc.rules()
    source_map = {p.name:p for p in dc.data_files() if p.name in set(ledger.source_file)}
    summaries = []
    paths = []
    for source in sorted(ledger.source_file.unique()):
        label = dc.fiscal_year_of(source)
        path = dc.OUT_DIR/f'fy{label}.parquet'
        backup = BACKUP/path.name
        table = pq.read_table(path)
        if not backup.exists():
            if 'dollars_reported' in table.column_names:
                raise ValueError(f'Missing uncorrected backup for already-cleaned {path}')
            shutil.copy2(path,backup)
            if rc.sha256(path)!=rc.sha256(backup):
                raise ValueError('Backup checksum mismatch')
        frozen = pq.read_table(backup)
        if not rc.reported_table(table).drop([c for c in rc.EXTRA if c in table.column_names]).equals(frozen):
            raise ValueError(f'Current source differs from frozen file outside approved corrections: {label}')
        result = rc.clean_table(table,source,source_map[source])
        staged = STAGE/path.name
        pq.write_table(result,staged,compression='snappy')
        if not pq.read_table(staged).equals(result):
            raise ValueError('Staging roundtrip failed')
        original = frozen['dollars'].to_numpy()
        clean = result['dollars'].to_numpy()
        flags = result['report_correction_neutralized'].to_numpy()
        expected = ledger.loc[ledger.source_file.eq(source)]
        loss = int(np.maximum(original,0).sum()-np.maximum(clean,0).sum())*1000
        if int(flags.sum())!=len(expected) or loss!=int(expected.dollars_reported.clip(lower=0).sum())*1000:
            raise ValueError('Per-file count/amount does not reconcile to frozen ledger')
        summaries.append(dict(fiscal_file=label,n_rows=len(original),n_neutralized=int(flags.sum()),
                              reported_positive_nominal=int(np.maximum(original,0).sum())*1000,
                              clean_positive_nominal=int(np.maximum(clean,0).sum())*1000,
                              excess_removed_nominal=loss,net_change_nominal=int((clean-original).sum())*1000,
                              uncorrected_sha256=rc.sha256(backup),corrected_sha256=rc.sha256(staged)))
        paths.append((path,staged))
        print(f'Staged {label}: {flags.sum():,} records; positive reduction ${loss:,}',flush=True)
    summary = pd.DataFrame(summaries).sort_values('fiscal_file').reset_index(drop=True)
    if len(summary)!=19 or summary.n_neutralized.sum()!=25780 or summary.excess_removed_nominal.sum()!=30011877000:
        raise ValueError('Full-period reconciliation failed')
    if summary.net_change_nominal.sum()!=0:
        raise ValueError('Canceled source dollars must sum to zero across fiscal files')
    state = dict(status='publishing',completed_files=[],ledger_sha256=rc.sha256(rc.LEDGER),
                 downstream_status='NOT_REBUILT: spending panels, instruments, regression caches and presentation results')
    STATE.write_text(json.dumps(state,indent=2)+'\n')
    for path,staged in paths:
        staged.replace(path)
        expected_sha = summary.loc[summary.fiscal_file.eq(path.stem[2:]),'corrected_sha256'].iloc[0]
        if rc.sha256(path)!=expected_sha:
            raise ValueError('Published-file hash mismatch')
        state['completed_files'].append(path.name)
        STATE.write_text(json.dumps(state,indent=2)+'\n')
    state['status']='complete'
    STATE.write_text(json.dumps(state,indent=2)+'\n')
    summary.to_csv(OUT/'correction_cleaning_applied_1966_1983.csv',index=False)
    write_memo(summary)
    print('PASS: 19 files, 25,780 records, exact accounting, backups, all other fields preserved.',flush=True)


def verify_source():
    """Use the parser verifier's independent raw-source reconciliation."""
    spec = importlib.util.spec_from_file_location('verify06',PIPELINE.parent/'py/06_verify_dcads_parse.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    tss = json.loads(dc.TSS.read_text())
    layouts,index = dc.load_layouts(include_delimited=False)
    sources = {p.name:p for p in dc.data_files() if p.name in set(rc.rules().source_file)}
    output = []
    for source,raw in sorted(sources.items()):
        label = dc.fiscal_year_of(source)
        layout = layouts[index[source]]
        failures,metrics,reported = verifier.check_file(dc.OUT_DIR/f'fy{label}.parquet',layout,tss,raw.stat().st_size//272)
        raw_states,raw_total,raw_lump = verifier.raw_state_dollars(raw,layout)
        state_failures,_ = verifier.county_state_rollup(reported,layout,raw_states)
        total = sum(reported['dollars'].to_pylist())
        if failures or state_failures or total!=raw_total or raw_lump!=0:
            raise ValueError((label,failures,state_failures,total,raw_total,raw_lump))
        current = pq.read_table(dc.OUT_DIR/f'fy{label}.parquet')
        expected = rc.clean_table(pq.read_table(BACKUP/f'fy{label}.parquet'),source,raw)
        if not current.equals(expected) or not rc.clean_table(current,source,raw).equals(current):
            raise ValueError('Published cleaning or idempotency failure')
        output.append(dict(fiscal_file=label,n_rows=metrics['rows'],reported_dollars=total,
                           raw_dollars=raw_total,parser_checks='PASS',raw_state_reconciliation='PASS',
                           approved_cleaning='PASS',idempotent='PASS'))
        print(f'Verified raw source and cleaned file: {label}',flush=True)
    pd.DataFrame(output).sort_values('fiscal_file').to_csv(OUT/'correction_cleaning_source_verification.csv',index=False)
    print('PASS: all 19 source reconciliations and published-file idempotency checks.',flush=True)


def write_memo(summary):
    summary = summary.sort_values('fiscal_file')
    if (OUT/'h2_report_correction_comparison.csv').exists():
        downstream = ('The corrected h=2 spending, instrument and analysis inputs are in '
            '`data/temp/report_correction_rebuild/`. The equal-weight and fixed-1975 population-weight '
            'comparisons, without winsorization and with pooled 1-99 winsorization of x/y/z, are in '
            '`REPORT_CORRECTION_RESULTS.md`. Presentation inputs, other horizons, design grids, '
            'BHJ/GPSS diagnostics and LP results have not been rebuilt. Do not mix those cached '
            'uncorrected instruments with corrected source inputs.')
    else:
        downstream = ('**Spending panels, instruments, regression caches and presentation results have not been rebuilt.** '
            'Existing results therefore do not incorporate this correction. Rebuild those inputs before '
            'estimating effects of the cleaning; do not mix corrected source files with cached uncorrected instruments.')
    lines = ['# Contract reporting correction rule, 1966-1983','',
        '## Production source files','',
        f'The 19 fiscal files contain {summary.n_rows.sum():,} rows. The approved rule neutralizes '
        '25,780 source records belonging to 12,890 verified initial/reversal pairs. No source row is deleted. '
        'Source: `Results/instrument_normalization/correction_cleaning_applied_1966_1983.csv`.', '',
        '- `dollars`: analysis amount in the original source unit (thousands), zero on approved canceled reports.',
        '- `dollars_reported`: amount exactly as reported in the original source.',
        '- `report_correction_neutralized`: indicator for an approved canceled report.', '',
        'Corrected replacements remain unless they are themselves reversed by another approved pair. '
        'Ambiguous cases and unknown correction codes remain untouched. The rule is independent of '
        'county influence, first-stage strength and second-stage significance. Raw source bytes are immutable.', '',
        'The positive-only spending sum excludes $30,011,877,000 from canceled reports. Across all covered '
        'fiscal files, the net-dollar change is exactly zero. Individual fiscal-year net totals can differ '
        'because some original and reversing reports appear in different fiscal files.', '',
        '## Annual reconciliation','',
        '| Fiscal file | Neutralized records | Removed positive dollars | Net-dollar change |',
        '|---|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.fiscal_file} | {r.n_neutralized:,} | {r.excess_removed_nominal:,.0f} | {r.net_change_nominal:,.0f} |')
    lines += ['', 'Source: `correction_cleaning_applied_1966_1983.csv` in `Results/instrument_normalization/`.', '',
        '## Reproducibility and verification','',
        'The approved ledger and its digest are in `step_1/code/reference/dcads_report_cancellations.csv` '
        'and `.json`. Producer 04 applies the rule before publishing each 272-byte-era parquet. '
        'Producer 104 stages and applies the rule to existing parsed files. It is idempotent and checks '
        'each approved record against its raw-byte SHA256 and all original parsed fields. Every non-dollar '
        'column, row position and reported amount must match the preserved source table exactly. '
        'All 19 corrected files are staged and verified before publication. Publication progress is recorded '
        'in `data/temp/contract_report_correction_application/application_state.json`.', '',
        'Uncorrected parsed files are preserved under '
        '`data/temp/contract_report_correction_application/uncorrected_fiscal_files/`. '
        'Parser verification 06 and source audit 103 use `dollars_reported` when present so that their '
        'raw-source checks remain distinct from the analysis-dollar cleaning checks.', '',
        'Source-level verification results are in `correction_cleaning_source_verification.csv` '
        'under `Results/instrument_normalization/`. The `--verify-source` option invokes the parser '
        'verifier on the 19 affected files, reconciles source dollars by state to independent raw-file reads, '
        'and checks the published cleaning and its idempotency.', '',
        '## Downstream status','',
        downstream, '',
        'Fiscal files from 1984 onward are untouched. Their administrative-correction handling remains unresolved.', '']
    MEMO.write_text('\n'.join(lines))
    assert MEMO.read_text().startswith('# Contract reporting')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepare-ledger',action='store_true')
    p.add_argument('--apply',action='store_true')
    p.add_argument('--verify-source',action='store_true')
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args()
    if not (args.prepare_ledger or args.apply or args.verify_source or args.report_only):
        p.error('Choose --prepare-ledger, --apply, --verify-source or --report-only')
    if args.prepare_ledger:
        prepare()
    if args.apply:
        apply()
    if args.verify_source:
        verify_source()
    if args.report_only:
        write_memo(pd.read_csv(OUT/'correction_cleaning_applied_1966_1983.csv',dtype={'fiscal_file':str}))
