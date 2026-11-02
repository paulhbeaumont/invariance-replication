"""Audit administrative reversals, without changing production spending.

Scope: canonical fiscal files, 1966-2018, including the 1976 transition quarter.
Exact matching is deliberately conservative. Missing or ambiguous matches are
unresolved, not evidence that the source is correct. Monetary totals are nominal.
This source-data audit estimates no regressions and creates no production cache.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / 'py'))
import dcads_common as dc

OUT = dc.MACRO_RESULTS_ROOT / 'instrument_normalization'
TEMP = dc.MACRO_VERSION_TEMP_DIR / 'contract_report_correction_audit'
MEMO = dc.MACRO_RESULTS_ROOT.parent / 'Instrument_analysis/Cleaning_spendingData/REPORT_CORRECTION_AUDIT.md'
IDENTITY = ['reporting_dept', 'purchasing_office', 'report_no', 'contract_no']
PAYLOAD = ['contractor_code', 'action_date', 'fsc', 'state_code', 'city_code', 'county_code']
TEXT = IDENTITY + PAYLOAD + ['correction_code', 'calendar_yearmonth', 'source_file',
                            'county_fips5', 'geo_class', 'contractor_name', 'county_name', 'state_name']


def export(con, query, name):
    path = OUT / f'correction_audit_{name}.csv'
    con.execute(f"COPY ({query}) TO '{path}' (HEADER, DELIMITER ',')")
    return path


def inventory(con):
    files = [(str(y), dc.OUT_DIR / f'fy{y}.parquet') for y in range(1966, 2007)]
    files += [('1976tq', dc.OUT_DIR / 'fy1976tq.parquet')]
    files += [(str(y), dc.USASPENDING_OUT_DIR / f'fy{y}.parquet') for y in range(2007, 2019)]
    rows, codes = [], []
    for label, path in sorted(files):
        if not path.exists():
            raise FileNotFoundError(path)
        con.execute(f"CREATE OR REPLACE TEMP VIEW input_file AS SELECT * FROM read_parquet('{path}', file_row_number=true)")
        columns = set(con.sql('DESCRIBE input_file').df().column_name)
        old = 'correction_code' in columns
        us = 'federal_action_obligation' in columns
        if not us:
            units = con.sql('SELECT DISTINCT dollars_unit FROM input_file').fetchall()
            if any(x[0] not in ('thousands', 'dollars') for x in units):
                raise ValueError((label, units))
        source_amount = 'dollars_reported' if 'dollars_reported' in columns else 'dollars'
        amount = ('federal_action_obligation' if us else
                  f"{source_amount} * CASE WHEN dollars_unit='thousands' THEN 1000 ELSE 1 END")
        n, positive, net, missing = con.sql(f'''SELECT count(*), sum(greatest({amount},0)),
            sum({amount}), count(*) FILTER (WHERE {amount} IS NULL) FROM input_file''').fetchone()
        metadata = sorted(c for c in columns if any(s in c for s in ('correct', 'modif', 'change_month', 'id_key')))
        rows.append(dict(fiscal_file=label, path=str(path), n_rows=n, positive_nominal=positive,
                         net_nominal=net, missing_amount_rows=missing, explicit_correction_field=old,
                         audit_status='explicit_reversal_matching' if old else 'no_explicit_correction_field',
                         other_metadata=';'.join(metadata), size_bytes=path.stat().st_size))
        if old:
            code = "coalesce(trim(correction_code),'')"
            frame = con.sql(f'''SELECT '{label}' fiscal_file, {code} correction_code, count(*) n_rows,
                sum(greatest({amount},0)) positive_nominal, sum({amount}) net_nominal,
                count(*) FILTER(WHERE {amount}<0) negative_rows
                FROM input_file GROUP BY 1,2''').df()
            codes.append(frame)
            projection = ','.join(f"coalesce(trim(cast({c} AS VARCHAR)),'') AS {c}" for c in TEXT)
            sql = f'''SELECT '{label}:' || file_row_number::VARCHAR row_id,
                '{label}' fiscal_file, file_row_number+1 record_number, {amount} amount,
                {projection} FROM input_file'''
            con.execute(('CREATE TABLE reports AS ' if len(codes) == 1 else 'INSERT INTO reports ') + sql)
        print(f'Inventory {label}: {n:,} records, correction field={old}', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'correction_audit_inventory.csv', index=False)
    pd.concat(codes, ignore_index=True).to_csv(OUT / 'correction_audit_codes.csv', index=False)
    if len(rows) != 54:
        raise AssertionError('Canonical file coverage')


def match(con, save=True):
    exact = ' AND '.join(f'r.{c}=o.{c}' for c in IDENTITY + PAYLOAD)
    required = ' AND '.join(f"r.{c}<>''" for c in IDENTITY)
    # Reporting month, rather than obligation date, orders administrative reports.
    con.execute(f'''CREATE TABLE candidates AS SELECT r.row_id reversal_id, o.row_id original_id
        FROM reports r JOIN reports o ON {exact} AND r.amount=-o.amount
        WHERE r.correction_code='2' AND o.correction_code IN ('','1') AND r.amount<>0
        AND {required} AND regexp_full_match(r.calendar_yearmonth,'[0-9]{{6}}')
        AND regexp_full_match(o.calendar_yearmonth,'[0-9]{{6}}')
        AND o.calendar_yearmonth<=r.calendar_yearmonth''')
    con.execute('''CREATE TABLE candidate_counts AS SELECT *,
        count(*) OVER(PARTITION BY reversal_id) n_originals,
        count(*) OVER(PARTITION BY original_id) n_reversals FROM candidates''')
    con.execute('''CREATE TABLE accepted_pairs AS SELECT reversal_id,original_id FROM candidate_counts
        WHERE n_originals=1 AND n_reversals=1''')
    replacement_key = ' AND '.join(f'r.{c}=c.{c}' for c in IDENTITY)
    con.execute(f'''CREATE TABLE replacement_candidates AS
        SELECT a.reversal_id,c.row_id corrected_id FROM accepted_pairs a
        JOIN reports r ON r.row_id=a.reversal_id JOIN reports c ON {replacement_key}
        AND c.calendar_yearmonth=r.calendar_yearmonth AND c.correction_code='1'
        AND c.row_id<>a.original_id''')
    con.execute('''CREATE TABLE replacement_counts AS SELECT *,
        count(*) OVER(PARTITION BY reversal_id) n_corrected,
        count(*) OVER(PARTITION BY corrected_id) n_pairs FROM replacement_candidates''')
    con.execute('''CREATE TABLE matched AS SELECT a.*,c.corrected_id FROM accepted_pairs a
        LEFT JOIN replacement_counts c ON a.reversal_id=c.reversal_id AND c.n_corrected=1 AND c.n_pairs=1''')
    con.execute('''CREATE TABLE status AS SELECT r.*,coalesce(n.n_candidates,0) n_candidates,
        a.original_id,a.corrected_id, CASE WHEN r.amount=0 THEN 'zero_reversal'
        WHEN n.n_candidates IS NULL THEN 'no_exact_original'
        WHEN a.original_id IS NULL THEN 'ambiguous_original'
        WHEN a.corrected_id IS NULL THEN 'unique_pair_replacement_unresolved'
        ELSE 'unique_pair_and_replacement' END match_status
        FROM reports r LEFT JOIN (SELECT reversal_id,count(*) n_candidates FROM candidates GROUP BY 1) n
        ON r.row_id=n.reversal_id LEFT JOIN matched a ON r.row_id=a.reversal_id
        WHERE r.correction_code='2' ''')
    if not save:
        return
    export(con, 'SELECT * FROM status ORDER BY fiscal_file,row_id', 'reversals')
    export(con, '''SELECT fiscal_file,match_status,count(*) n_reversals,
        sum(abs(amount)) absolute_reversal_nominal FROM status GROUP BY 1,2 ORDER BY 1,2''', 'match_summary')
    con.execute('''CREATE TABLE canceled AS SELECT r.*,a.reversal_id,'initial_or_corrected' pair_role
        FROM matched a JOIN reports r ON a.original_id=r.row_id UNION ALL
        SELECT r.*,a.reversal_id,'reversal' pair_role FROM matched a JOIN reports r ON a.reversal_id=r.row_id''')
    export(con, 'SELECT * FROM canceled ORDER BY reversal_id,pair_role', 'canceled_report_ledger')
    export(con, '''SELECT m.*,r.amount reversal_amount,o.amount original_amount,c.amount corrected_amount,
        r.contractor_name,r.contract_no,r.fsc,r.county_name,r.state_name,
        o.fiscal_file original_file,r.fiscal_file reversal_file,c.fiscal_file corrected_file,
        o.calendar_yearmonth original_month,r.calendar_yearmonth correction_month
        FROM matched m JOIN reports r ON r.row_id=m.reversal_id JOIN reports o ON o.row_id=m.original_id
        LEFT JOIN reports c ON c.row_id=m.corrected_id ORDER BY abs(r.amount) DESC,m.reversal_id''', 'pairs')
    # Each cancelled pair sums to zero. Positive-only excess is the positive leg,
    # which can be the reversal itself when a negative initial report is reversed.
    export(con, '''SELECT fiscal_file,count(*) n_canceled_records,
        sum(greatest(amount,0)) excess_positive_nominal,
        sum(CASE WHEN geo_class='county' AND NOT starts_with(fsc,'Y') THEN greatest(amount,0) ELSE 0 END)
        county_nonconstruction_excess_nominal FROM canceled GROUP BY 1 ORDER BY 1''', 'annual_excess')
    export(con, '''SELECT fiscal_file,county_fips5,county_name,state_name,fsc,count(*) n_records,
        sum(greatest(amount,0)) excess_positive_nominal FROM canceled
        WHERE geo_class='county' AND NOT starts_with(fsc,'Y') GROUP BY 1,2,3,4,5
        HAVING sum(greatest(amount,0))>0 ORDER BY excess_positive_nominal DESC''', 'county_category_excess')
    export(con, "SELECT * FROM reports WHERE correction_code NOT IN ('','1','2') ORDER BY abs(amount) DESC,row_id", 'unknown_codes')


def verify_raw(con):
    # Verify every cancellation against original bytes, not just the largest case.
    # Parquet row positions correspond to fixed-width record positions; field
    # assertions make this ordering assumption testable for every selected row.
    selected = con.sql('''SELECT DISTINCT r.* FROM reports r WHERE row_id IN
        (SELECT row_id FROM canceled UNION SELECT corrected_id FROM matched WHERE corrected_id IS NOT NULL)
        OR correction_code NOT IN ('','1','2') ORDER BY source_file,record_number''').df()
    spec = json.loads((PIPELINE.parent / 'layouts/dcads_272.json').read_text())
    fields = {f['name']: f for f in spec['fields']}
    sources = {p.name: p for p in dc.data_files() if p.name in set(selected.source_file)}
    proof = []
    for source, frame in selected.groupby('source_file', sort=False):
        with sources[source].open('rb') as f:
            for row in frame.itertuples(index=False):
                f.seek((row.record_number-1)*272)
                raw = f.read(272)
                text = raw.decode('latin-1')
                assert len(raw) == 272
                for name in IDENTITY + PAYLOAD + ['correction_code']:
                    field = fields[name]
                    assert text[field['start']:field['end']].strip() == getattr(row,name), (row.row_id,name)
                field = fields['dollars']
                assert dc.decode_zoned(text[field['start']:field['end']])*1000 == row.amount, row.row_id
                proof.append(dict(row_id=row.row_id,source_file=source,record_number=row.record_number,
                                  sha256=hashlib.sha256(raw).hexdigest(),raw_fields_match=True))
        print(f'Raw verification {source}: {len(frame):,} records', flush=True)
    pd.DataFrame(proof).to_csv(OUT / 'correction_audit_raw_verification.csv',index=False)


def verify(con):
    assert con.sql('SELECT count(*)-count(DISTINCT row_id) FROM reports').fetchone()[0] == 0
    assert con.sql('SELECT count(*)-count(DISTINCT row_id) FROM canceled').fetchone()[0] == 0
    assert con.sql('SELECT max(abs(s)) FROM (SELECT sum(amount) s FROM canceled GROUP BY reversal_id)').fetchone()[0] == 0
    assert con.sql('SELECT count(*) FROM reports WHERE amount IS NULL').fetchone()[0] == 0
    # Source-confirmed St. James fixture, independent of the ranking of cases.
    fixture = con.sql("""SELECT o.amount,r.amount,c.amount FROM matched m
        JOIN reports r ON r.row_id=m.reversal_id JOIN reports o ON o.row_id=m.original_id
        JOIN reports c ON c.row_id=m.corrected_id
        WHERE r.fiscal_file='1982' AND r.contract_no='LA60082C5010' AND r.report_no='3082'""").fetchall()
    assert fixture == [(573747000,-573747000,57375000)], fixture
    assert con.sql("SELECT count(*) FROM reports WHERE fiscal_file='1976tq'").fetchone()[0] > 0
    print('PASS: unique source IDs, one-to-one cancellation, zero-sum pairs, nonmissing amounts, transition quarter, St. James fixture.',flush=True)


def report(con):
    inv = pd.read_csv(OUT / 'correction_audit_inventory.csv')
    annual = pd.read_csv(OUT / 'correction_audit_annual_excess.csv')
    pairs = pd.read_csv(OUT / 'correction_audit_pairs.csv')
    statuses = con.sql('SELECT match_status,count(*) n FROM status GROUP BY 1 ORDER BY 1').fetchall()
    unknown = con.sql("SELECT count(*) FROM reports WHERE correction_code NOT IN ('','1','2')").fetchone()[0]
    county_count = con.sql("""SELECT count(DISTINCT county_fips5) FROM canceled
        WHERE amount>0 AND geo_class='county' AND NOT starts_with(fsc,'Y')""").fetchone()[0]
    positive_reversals = con.sql('''SELECT count(*) FROM matched m JOIN reports r
        ON r.row_id=m.reversal_id WHERE r.amount>0''').fetchone()[0]
    cross_year = con.sql('''SELECT count(*) FROM matched m JOIN reports r ON r.row_id=m.reversal_id
        JOIN reports o ON o.row_id=m.original_id WHERE r.fiscal_file<>o.fiscal_file''').fetchone()[0]
    raw_checked = len(pd.read_csv(OUT / 'correction_audit_raw_verification.csv'))
    n_reports = con.sql('SELECT count(*) FROM reports').fetchone()[0]
    full = inv.loc[inv.explicit_correction_field,'positive_nominal'].sum()
    extra = annual.excess_positive_nominal.sum()
    text = ['# Administrative reporting corrections in contract spending', '',
        '## Scope and status', '',
        'This producer audits reported source amounts, using `dollars_reported` when that column is present. '
        'It does not modify production data. Applied cleaning and downstream rebuild status are documented '
        'in `REPORT_CORRECTION_CLEANING.md`. Amounts below are nominal dollars, not regression coefficients.', '',
        f'Table 1 inventories {len(inv):,} canonical fiscal files ({inv.n_rows.sum():,} records) from 1966 through 2018, including the 1976 '
        f'transition quarter. The {int(inv.explicit_correction_field.sum())} files covering 1966-1983 contain '
        f'an explicit correction field ({n_reports:,} reports). The other '
        f'{int((~inv.explicit_correction_field).sum())} files, 1984-2018, do not expose that field. '
        'Their modification/date/identifier metadata are listed but are not treated as equivalent reversal codes. '
        'No absence-of-error claim is made for these later years. USAspending overlap files before the canonical '
        '2007 source seam are outside this inventory.', '',
        '## What constitutes a correction', '',
        'NARA 125.2DP.pdf, PDF page 17 (DD350 form), labels code 1 as a corrected entry and code 2 as a '
        'reversing entry. NARA 124.7DP.pdf, PDF page 170 (FY1981/1982 translation manual), explicitly labels '
        'blank as initial entry, 1 as corrected entry, and 2 as reversal of initial entry. '
        'These are administrative report corrections, not ordinary negative contract modifications.', '',
        'The FY1984 documentation illustrates why code meanings must not be carried across eras blindly. '
        'In 124.9DP.pdf, PDF page 173, the listing manual defines Type of Report as 0 original, 1 canceling, '
        'and 2 correcting. Those meanings differ from the 272-byte correction field. The master-file layout '
        'on PDF page 22 contains a last-change month but no Type of Report field. This audit cannot infer '
        'from that layout whether every correction is already incorporated in the archived master file.', '',
        'A candidate pair has equal and opposite dollars, identical department, purchasing office, report '
        'number, contract number, contractor code, action date, category, and raw state/city/county codes. '
        'The initial report must be blank or code 1 and its reporting month cannot follow the code-2 report. '
        'All four report-identity fields must be nonempty. Matching spans fiscal files and the transition quarter. '
        'A pair is accepted only if both reports have exactly one candidate. Repeated ambiguous payloads are not '
        'arbitrarily assigned. Same-month ordering cannot be recovered from a monthly file.', '',
        'A corrected replacement is linked separately using the department, purchasing office, report and '
        'contract numbers, code 1, and the reversal reporting month. The link must be unique in both directions. '
        'The replacement may change amount, contractor, category or geography. Report/contract-number changes '
        'can prevent a link. An unmatched reversal or replacement remains unresolved.', '',
        f'There are {unknown:,} reports carrying other correction-field values (`correction_audit_codes.csv` and the unknown-code '
        'ledger). Their bytes are verified, but their interpretation is unresolved; none is automatically recoded.', '',
        '## Table 1. Coverage', '',
        '| Period | Files | Reversal matching |', '|---|---:|---|',
        '| 1966-1983, including transition quarter | 19 | Explicit codes, conservative matching |',
        '| 1984-2006 | 23 | No explicit correction field |',
        '| 2007-2018 | 12 | No explicit correction field |', '',
        'Source: `correction_audit_inventory.csv`.', '',
        '## Table 2. Code-2 report classification', '',
        '| Classification | Reports |','|---|---:|']
    text += [f'| {s} | {n:,} |' for s,n in statuses]
    text += ['', 'Source: `correction_audit_match_summary.csv`; reversal-level classifications are in `correction_audit_reversals.csv`.', '',
        '## Size of the positive-only accounting error', '',
        f'The {len(pairs):,} uniquely matched pairs contain ${extra:,.0f} in positive legs, '
        f'{100*extra/full:.4f}% of the ${full:,.0f} positive spending in the explicit-code files. '
        'This is the amount that remains in a positive-only sum even though the matched pair cancels exactly. '
        f'The county, nonconstruction subset is ${annual.county_nonconstruction_excess_nominal.sum():,.0f}. '
        f'The positive legs in that subset span {county_count:,} distinct source county FIPS codes. '
        'These totals cover all source counties, not just the balanced regression sample. '
        'They are accounting amounts for accepted matches, not an estimate of all data errors.', '',
        f'Among accepted pairs, {positive_reversals:,} reverse a negative initial report and therefore have '
        f'a positive reversal. There are {cross_year:,} cross-fiscal-file pairs. The annual table attributes '
        'the excess to the fiscal file containing the positive leg; pairwise net cancellation need not '
        'occur within a single fiscal year.', '',
        'Source: `correction_audit_annual_excess.csv` and `correction_audit_inventory.csv`.', '',
        '## Table 3. Largest uniquely matched reversals', '',
        '| Fiscal file | Contractor | Contract | Initial dollars | Reversal dollars | Replacement dollars |',
        '|---|---|---|---:|---:|---:|']
    for r in pairs.head(15).itertuples():
        replacement = 'Unresolved' if pd.isna(r.corrected_amount) else f'{r.corrected_amount:,.0f}'
        text.append(f'| {r.reversal_file} | {r.contractor_name} | {r.contract_no} | {r.original_amount:,.0f} | {r.reversal_amount:,.0f} | {replacement} |')
    text += ['', 'Source: `correction_audit_pairs.csv`.', '',
        '## Verification and limits', '',
        f'The raw-byte check covers {raw_checked:,} distinct reports. Every accepted original, reversal and uniquely linked replacement, plus every unknown-code report, '
        'is checked against its 272-byte original record. The check covers amount, correction code, all matching '
        'fields, record position, and a SHA256 record fingerprint. Source-row uniqueness and zero-sum cancellation '
        'are asserted. The St. James fixture must reproduce +573,747,000, -573,747,000, and +57,375,000 '
        '(contract LA60082C5010, report 3082, fiscal 1982). '
        'Raw evidence: `correction_audit_raw_verification.csv`.', '',
        'The audit does not establish how 1984-2018 snapshots process administrative corrections. '
        'It does not recode unknown correction values, resolve ambiguous identities, or prove that every '
        'unmatched negative report is an ordinary deobligation. It does not assess effects on either regression stage.', '',
        '## Production decision', '',
        'A defensible mechanical rule is to neutralize verified original/reversal pairs before applying a '
        'positive-action filter. Retain the corrected replacement unless it is itself reversed by another '
        'verified pair. A positive reversal undoing a negative initial entry must also be neutralized. '
        'This is not a switch to net spending for every contract. Ambiguous cases require separate evidence. '
        'The approved production rule and its application are documented in `REPORT_CORRECTION_CLEANING.md`. '
        'First- and second-stage comparisons require rebuilt analysis inputs.', '',
        '## Reproduction', '',
        '`step_1/code/pipeline/103_audit_contract_report_corrections.py` builds the inventory, matching ledger, '
        'raw-record checks and this memo. `--report-only` refreshes prose from the saved audit database. '
        'The audit database is in `data/temp/contract_report_correction_audit/`; all CSVs are in '
        '`Results/instrument_normalization/`.', '']
    MEMO.parent.mkdir(parents=True,exist_ok=True)
    MEMO.write_text('\n'.join(text))
    assert MEMO.read_text().startswith('# Administrative')
    print(f'Memo: {MEMO}',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-only',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    TEMP.mkdir(parents=True,exist_ok=True)
    with duckdb.connect(str(TEMP / 'audit.duckdb')) as con:
        con.execute("SET memory_limit='4GB'")
        con.execute('SET threads=4')
        if not args.report_only:
            for name in ['reports','candidates','candidate_counts','accepted_pairs','replacement_candidates',
                         'replacement_counts','matched','status','canceled']:
                con.execute(f'DROP TABLE IF EXISTS {name}')
            inventory(con)
            match(con)
            verify(con)
            verify_raw(con)
        report(con)


if __name__ == '__main__':
    main()
