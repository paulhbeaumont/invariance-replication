"""Apply the approved 1966-1983 administrative-cancellation ledger.

Source bytes and row counts are immutable. The analysis dollars column is zero
on verified initial/reversal pairs; dollars_reported preserves source units.
The ledger is pinned by SHA256 and every selected source record is rechecked.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import dcads_common as dc

REFERENCE = Path(__file__).resolve().parent.parent / 'reference'
LEDGER = REFERENCE / 'dcads_report_cancellations.csv'
MANIFEST = REFERENCE / 'dcads_report_cancellations.json'
EXTRA = ['dollars_reported', 'report_correction_neutralized']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def rules():
    manifest = json.loads(MANIFEST.read_text())
    if sha256(LEDGER) != manifest['ledger_sha256']:
        raise ValueError('Correction ledger digest mismatch')
    frame = pd.read_csv(LEDGER, dtype={'source_file': str, 'pair_id': str})
    if len(frame) != manifest['n_canceled_records'] or frame.duplicated(['source_file','record_number']).any():
        raise ValueError('Correction ledger row count or key mismatch')
    grouped = frame.groupby('pair_id').dollars_reported.agg(['size','sum'])
    if not (grouped['size'].eq(2) & grouped['sum'].eq(0)).all():
        raise ValueError('Cancellation pairs must have two equal-and-opposite legs')
    return frame


def reported_table(table):
    """Source-dollar view for parser reconciliation, not for analysis."""
    if 'dollars_reported' in table.column_names:
        table = table.set_column(table.schema.get_field_index('dollars'), 'dollars', table['dollars_reported'])
    return table


def clean_table(table, source_name, source_path):
    selected = rules().loc[lambda x: x.source_file.eq(source_name)]
    if selected.empty:
        raise ValueError(f'No approved cancellation ledger for {source_name}')
    if set(table['source_file'].unique().to_pylist()) != {source_name}:
        raise ValueError('Mixed or mismatched source file')
    if set(table['layout_id'].unique().to_pylist()) != {'dcads_272'}:
        raise ValueError('Correction rule is restricted to the 272-byte era')
    if set(table['dollars_unit'].unique().to_pylist()) != {'thousands'}:
        raise ValueError('Expected source dollars in thousands')
    if source_path.stat().st_size != table.num_rows*272:
        raise ValueError('Source/parquet row count mismatch')
    indices = selected.record_number.to_numpy(dtype=np.int64)-1
    if indices.min()<0 or indices.max()>=table.num_rows:
        raise ValueError('Ledger position outside source file')
    raw_view = reported_table(table)
    sample = raw_view.take(pa.array(indices)).to_pydict()
    spec = json.loads((REFERENCE.parent/'layouts/dcads_272.json').read_text())
    with source_path.open('rb') as f:
        for j, row in enumerate(selected.itertuples(index=False)):
            f.seek((row.record_number-1)*272)
            raw = f.read(272)
            if hashlib.sha256(raw).hexdigest()!=row.sha256:
                raise ValueError(f'Raw byte fingerprint mismatch: {source_name}:{row.record_number}')
            text = raw.decode('latin-1')
            for field in spec['fields']:
                value = text[field['start']:field['end']]
                value = dc.decode_zoned(value) if field['type']=='int' else value.strip()
                if sample[field['name']][j] != value:
                    raise ValueError(f"Parsed/source mismatch: {source_name}:{row.record_number}:{field['name']}")
            if sample['dollars'][j] != row.dollars_reported:
                raise ValueError('Approved amount differs from source amount')
    original = raw_view['dollars']
    dollars = original.to_numpy().copy()
    if np.isnan(dollars).any():
        raise ValueError('Missing dollar amount in explicit-code era')
    neutralized = np.zeros(table.num_rows,dtype=bool)
    neutralized[indices] = True
    dollars[neutralized] = 0
    result = table.drop([c for c in EXTRA if c in table.column_names])
    result = result.set_column(result.schema.get_field_index('dollars'),'dollars',pa.array(dollars,type=pa.int64()))
    result = result.append_column('dollars_reported',original)
    result = result.append_column('report_correction_neutralized',pa.array(neutralized))
    # Every non-dollar field and every source amount survives exactly, in order.
    compare = reported_table(result).drop(EXTRA)
    expected = raw_view.drop([c for c in EXTRA if c in raw_view.column_names])
    if not compare.equals(expected):
        raise ValueError('Unexpected change outside the analysis-dollar column')
    return result


def clean_parquet(path, source_name, source_path):
    table = pq.read_table(path)
    result = clean_table(table,source_name,source_path)
    staged = path.with_name(path.name+'.correction-staged')
    pq.write_table(result,staged,compression='snappy')
    if not pq.read_table(staged).equals(result):
        raise ValueError('Correction parquet roundtrip failed')
    staged.replace(path)
    return int(np.count_nonzero(result['report_correction_neutralized'].to_numpy()))
