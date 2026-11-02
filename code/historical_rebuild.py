"""Run hash-verified historical builders with outputs isolated from shared data.

Paths and explicitly recorded historical command-line defaults are set here.
Paul's approved excluded-Kalawao audit exception is recorded separately from
the original audit failure. Original source bytes and numerical construction
rules are unchanged. Prepared targets serve only as independent comparisons.
"""
import csv
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import threading
import weakref
from contextlib import contextmanager, redirect_stderr

from package_io import ROOT, logs, read, require, sha, write

COMMIT = 'fc9df562037324ecfeb1cae1d34de8b13d408ce5'
VENDOR = ROOT / 'provenance/historical_fc9df56'
MANIFEST = ROOT / 'provenance/historical_code_manifest_fc9df56.json'
STORAGE = ROOT / 'data/raw_rebuild'
WORKSPACE = STORAGE / 'built/historical_fc9df56'
RUNLOG = 'raw_rebuild_20261027'
EFSY_SOURCE = None
EFSY_CERTIFICATE = None
EXCLUDED_UNIT_WARNING = 'KALAWAO_15005_OUTSIDE_FIXED_PAPER_SAMPLE'


def freeze(checkout):
    checkout = checkout.resolve()
    git = lambda *args: subprocess.check_output(['git','-C',str(checkout),*args], text=True).strip()
    require(git('rev-parse','HEAD') == COMMIT, 'Not the pinned historical commit')
    require(not git('status','--porcelain','--untracked-files=no'), 'Historical checkout has tracked modifications')
    snapshot = read(ROOT / 'provenance/original_4a/config/source_snapshot.json')
    require(snapshot['git_commit'] == COMMIT, 'Source snapshot commit changed')
    for name, expected in snapshot['files'].items():
        require(sha(checkout / '4_Macro_version' / name) == expected, 'Historical source hash differs: ' + name)
    tree = git('ls-tree','-r',COMMIT,'--','4_Macro_version/step_1/code',
               '4_Macro_version/step_1/README.md','4_Macro_version/step_2/code/py')
    files = []
    for line in tree.splitlines():
        metadata, relative = line.split('\t',1)
        mode, kind, oid = metadata.split()
        require(kind == 'blob' and mode in ('100644','100755'), 'Unexpected non-file historical source')
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe historical source path')
        source, target = checkout / path, VENDOR / path
        digest = sha(source)
        if target.exists():
            require(sha(target) == digest, 'Preserved historical source differs; refusing overwrite')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source,target)
        require(sha(target) == digest, 'Historical source changed during preservation')
        files.append(dict(path=relative, sha256=digest, bytes=source.stat().st_size, git_blob=oid))
    require(not git('status','--porcelain','--untracked-files=no'), 'Historical checkout changed during preservation')
    record = dict(commit=COMMIT, source_repository='thetfplab/aggregation-disaggregation-invariance',
                  files=files, independent_snapshot_anchors=snapshot['files'],
                  status='PINNED_CODE_RECOVERED_HASH_VERIFIED_NOT_FULL_RAW_REPLICATION')
    if MANIFEST.exists():
        require(read(MANIFEST) == record, 'Historical code manifest differs; refusing overwrite')
    else:
        write(MANIFEST, record)
    write(logs()/RUNLOG/'code_recovery.json',dict(status='PASS',commit=COMMIT,
          checked_utc=datetime.now(timezone.utc).isoformat(),files=len(files),
          independently_pinned_files=len(snapshot['files']),manifest_sha256=sha(MANIFEST),
          raw_to_results_certified=False))
    print('HISTORICAL CODE PASS:',len(files),'files;',len(snapshot['files']),'independent hash anchors',flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verified_code():
    manifest = read(MANIFEST)
    require(manifest['commit'] == COMMIT, 'Historical commit changed')
    for item in manifest['files']:
        require(sha(VENDOR/item['path']) == item['sha256'], 'Preserved historical code changed: '+item['path'])
    return VENDOR/'4_Macro_version/step_1/code'


def setup(stage):
    code = verified_code()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    # The historical common module explicitly supports this process-local path
    # override. Never point its writable defaults at the real shared project.
    os.environ['INVARIANCE_DROPBOX_ROOT'] = str(WORKSPACE.resolve())
    sys.path.insert(0,str(code/'py'))
    dc = load('dcads_common',code/'py/dcads_common.py')
    dc.LOG_DIR = logs()/RUNLOG/stage
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    dc.SCRATCH_DIR = WORKSPACE/'scratch'
    dc.SOURCE_DIR = STORAGE/'raw/dcads'
    catalog = read(ROOT/'provenance/raw_source_catalog.json')
    from raw_sources import raw_path
    zalf = [a for a in catalog['assets'] if a['source']=='dcads' and Path(a['relative_path']).name=='RG330.DEFCON.ZALF']
    require(len(zalf)==1, 'Ambiguous raw ZALF source')
    dc.ZALF = raw_path(zalf[0])
    dc.EFSY_PANEL = EFSY_SOURCE if EFSY_SOURCE is not None else STORAGE/'built/efsy_panel_naics.parquet'
    for path in [dc.OUT_DIR, dc.EMP_OUT_DIR, dc.PANEL_OUT_DIR, dc.SCRATCH_DIR, dc.BEA_SOURCE_DIR]:
        require(WORKSPACE.resolve() in path.resolve().parents, 'Historical output path escapes isolated workspace')
    require(logs().resolve() in dc.LOG_DIR.resolve().parents, 'Historical logs escape root log tree')
    return code, dc


@contextmanager
def output_io_retries():
    """Retry only Windows sharing violations on this build's own file moves.

    Dropbox/antivirus can briefly lock a newly closed file. No bytes, paths or
    construction rules are changed; existing failed-attempt records are kept.
    """
    originals = {(owner,name):getattr(owner,name) for owner in (os,Path) for name in ('rename','replace')}
    depth = threading.local()
    readers = weakref.WeakSet()
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None
    original_reader = pq.ParquetFile if pq is not None else None
    if original_reader is not None:
        class TrackedParquetFile(original_reader):
            def __init__(self, source, *args, **kwargs):
                super().__init__(source,*args,**kwargs)
                self._ns_path = Path(source).resolve() if isinstance(source,(str,os.PathLike)) else None
                if self._ns_path is not None and WORKSPACE.resolve() in self._ns_path.parents:
                    readers.add(self)
    def release_completed_readers(source,target):
        paths = {Path(source).resolve(),Path(target).resolve()}
        for reader in list(readers):
            if reader._ns_path in paths:
                # A requested file move occurs after the original batch loop and
                # writer.close(). Windows requires releasing that completed
                # reader before replacing its source; Linux does not.
                reader.close(force=True)
                readers.discard(reader)
    def operation(original):
        def moved(source, target, *args, **kwargs):
            if getattr(depth,'active',False):
                return original(source,target,*args,**kwargs)
            depth.active = True
            try:
                for attempt in range(21):
                    try:
                        return original(source,target,*args,**kwargs)
                    except PermissionError as exc:
                        scoped = all(WORKSPACE.resolve() in Path(p).resolve().parents for p in (source,target))
                        if not scoped or getattr(exc,'winerror',None) not in (5,32,33) or attempt==20:
                            raise
                        if attempt==0:
                            print('Transient Windows file lock; retrying output move:',Path(target).name,flush=True)
                        release_completed_readers(source,target)
                        time.sleep(0.25)
            finally:
                depth.active = False
        return moved
    try:
        if pq is not None:
            pq.ParquetFile = TrackedParquetFile
        for (owner,name), original in originals.items():
            setattr(owner,name,operation(original))
        yield
    finally:
        if pq is not None:
            pq.ParquetFile = original_reader
        for (owner,name), original in originals.items():
            setattr(owner,name,original)


def call_original(module, arguments):
    previous = sys.argv
    try:
        sys.argv = [module.__file__,*arguments]
        with output_io_retries():
            return module.main()
    finally:
        sys.argv = previous


def original_main(module, arguments):
    result = call_original(module,arguments)
    require(result in (None,0), 'Historical builder reported failure: '+Path(module.__file__).name)


def excluded_population_failure(result, stderr, rebuilt_units, reference_units):
    """Accept only Paul's approved, single out-of-sample Kalawao assertion.

    The pinned producer's audit still FAILED. This decision allows the paper
    application to proceed; it does not repair data or certify the whole source.
    """
    require(result == 1, 'Unexpected historical population return code')
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    require(len(lines)==2 and lines[0]=='FAIL: 1 check(s) failed',
            'Only the single approved Kalawao audit failure may be a warning')
    match = re.fullmatch(r'income gap: CCG 15005 has no income denominator but \$([\d,]+) of contracts',lines[1])
    require(match is not None and int(match.group(1).replace(',',''))>0,
            'Population failure is not the approved excluded-Kalawao assertion')
    rebuilt = sorted(int(c) for c in rebuilt_units)
    reference = sorted(int(c) for c in reference_units)
    require(len(rebuilt)==len(set(rebuilt))==3058 and rebuilt==reference,
            'Retained county identifiers differ from the independent fixed paper sample')
    require(15005 not in rebuilt and 15005 not in reference,
            'Kalawao cannot be a warning if it belongs to the retained sample')
    return dict(code=EXCLUDED_UNIT_WARNING,county=15005,county_name='Kalawao, HI',
                original_audit_status='FAILED',original_return_code=result,
                original_stderr=stderr,retained_counties=len(rebuilt),
                retained_county_ids=rebuilt,reference_county_ids=reference,
                retained_identifiers_match_reference=True,excluded_from_both_samples=True,
                reason='BEA reports no separate income/population denominator for Kalawao.',
                approved_by='Paul Beaumont',approval_date='2026-10-27',
                data_or_specification_changed=False)


def population_main(code, dc, record):
    import pandas as pd
    capture = io.StringIO()
    previous_stderr = sys.stderr
    class Tee:
        def write(self, value):
            previous_stderr.write(value)
            return capture.write(value)
        def flush(self):
            previous_stderr.flush()
    with redirect_stderr(Tee()):
        result = call_original(load('historical_income',code/'py/12_build_county_income.py'),
                               ['--first-year','1969','--last-year','2018'])
    record.update(original_audit_return_code=result,
                  original_audit_status='PASS' if result in (None,0) else 'FAILED',
                  original_audit_stderr=capture.getvalue())
    if result not in (None,0):
        # Fail closed on unknown errors before loading any supporting evidence.
        lines = [line.strip() for line in capture.getvalue().splitlines() if line.strip()]
        require(result==1 and len(lines)==2 and lines[0]=='FAIL: 1 check(s) failed',
                'Historical population audit has an unapproved failure')
        producer = load('population_sample_source51',code/'py/51_county_multiplier_grid.py')
        panel = producer.base_panel()
        reference = ROOT/'data/upstream/baseline_roll5_gap5/source51_county_panel.parquet'
        expected = read(ROOT/'reference/upstream_source_manifest.json')['inputs'][reference.name]
        require(sha(reference)==expected, 'Independent county-support reference changed')
        original_units = pd.read_parquet(reference,columns=['ccg']).ccg.unique()
        record['warning'] = excluded_population_failure(result,capture.getvalue(),panel.ccg.unique(),original_units)
        evidence = [dc.PANEL_OUT_DIR/'income_ccg_year.parquet',
                    dc.PANEL_OUT_DIR/'population_ccg_year.parquet',
                    dc.EMP_OUT_DIR/'efsy_ccg_year.parquet',reference]
        record['warning']['evidence_files'] = [dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in evidence]
        record['status']='READY_WITH_EXCLUDED_UNIT_WARNING'
        print('APPLICATION WARNING: Kalawao (15005) remains excluded; original population audit FAILED. '
              'The 3,058 retained county identifiers are unchanged.',flush=True)


def checked_efsy():
    if EFSY_SOURCE is not None:
        from raw_sources import verified_asset
        report = read(EFSY_CERTIFICATE)
        source = verified_asset('efsy','efsy_panel_naics.csv.zip')
        require(report['status']=='SOURCE_CONVERSION_PASS' and report['rows']==44262620
                and report['source_sha256']==sha(source)
                and not any(report[k] for k in ('filtering','recoding','allocation')),
                'Fresh published-source conversion is not certified')
        require(sha(EFSY_SOURCE)==report['output_sha256'], 'Fresh converted EFSY source changed')
        return
    report = read(logs()/'raw_rebuild_20261022/efsy_reference_comparison.json')
    require(report['status']=='PASS' and report['employment_values_not_bitwise_equal']==0,
            'Public EFSY comparison has not passed')
    require(sha(STORAGE/'built/efsy_panel_naics.parquet')==report['rebuilt_sha256'],
            'Converted public EFSY source changed')


def verify_dcads_sources():
    from raw_sources import raw_path
    assets = [a for a in read(ROOT/'provenance/raw_source_catalog.json')['assets'] if a['source']=='dcads']
    seen = set()
    for asset in assets:
        path = raw_path(asset)
        if path in seen:
            continue
        require(path.is_file() and path.stat().st_size==asset['expected_bytes'] and sha(path)==asset['expected_sha256'],
                'Raw DCADS input differs: '+asset['relative_path'])
        seen.add(path)
    print('Verified raw DCADS files:',len(seen),flush=True)


def passed_stage(name):
    """Application readiness, including the narrowly recorded population warning."""
    paths = [logs()/RUNLOG/(name+'_stage.json')]
    paths += list((logs()/RUNLOG).glob(name+'_attempt*_stage.json'))
    for path in paths:
        if not path.exists():
            continue
        record = read(path)
        if record.get('commit')!=COMMIT:
            continue
        if record.get('status')=='PASS':
            return True
        warning = record.get('warning',{})
        if (name=='population' and record.get('stage')=='population'
                and record.get('status')=='READY_WITH_EXCLUDED_UNIT_WARNING'
                and record.get('original_audit_status')=='FAILED'
                and warning.get('code')==EXCLUDED_UNIT_WARNING):
            excluded_population_failure(warning['original_return_code'],warning['original_stderr'],
                                        warning['retained_county_ids'],warning['reference_county_ids'])
            # The decision was made from the actual rebuilt and independent
            # reference support. It is valid only for those unchanged inputs.
            evidence = warning.get('evidence_files',[])
            expected = {p.relative_to(ROOT).as_posix() for p in (
                WORKSPACE/'1_current_paper/_data/temp/county_panel/income_ccg_year.parquet',
                WORKSPACE/'1_current_paper/_data/temp/county_panel/population_ccg_year.parquet',
                WORKSPACE/'1_current_paper/_data/temp/employment/efsy_ccg_year.parquet',
                ROOT/'data/upstream/baseline_roll5_gap5/source51_county_panel.parquet')}
            require(len(evidence)==4 and {item['path'] for item in evidence}==expected,
                    'Excluded-unit warning lacks its exact input evidence')
            for item in evidence:
                relative = Path(item['path'])
                require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe audit evidence path')
                require(sha(ROOT/relative)==item['sha256'], 'Population-warning evidence changed')
            return True
    return False


def stage(name, years=None, usaspending_root=None, attempt=1, resume_missing=False):
    code, dc = setup(name)
    label = name + ('_'+str(years) if years else '')
    require(attempt>=1, 'Attempt must be positive')
    if attempt>1:
        label += '_attempt'+str(attempt)
    certificate = logs()/RUNLOG/(label+'_stage.json')
    require(not certificate.exists(), 'A stage record already exists; preserve it rather than overwrite/retry silently')
    record = dict(status='RUNNING',stage=name,years=years,commit=COMMIT,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  interpreter=sys.version, resume_missing=resume_missing,
                  isolated_workspace=WORKSPACE.relative_to(ROOT).as_posix(),raw_to_results_certified=False)
    write(certificate,record)
    try:
        if name=='population':
            from raw_sources import verified_asset
            # The historical reader audits its declared geography gap against
            # both employment and procurement; retain those checks in full.
            require(passed_stage('employment') and passed_stage('dcads-geography'),
                    'Population audit requires completed employment and contract geography')
            require(all((dc.USASPENDING_OUT_DIR/('fy%d.parquet'%y)).exists() for y in range(2007,2019)),
                    'Population audit requires the full USAspending production block')
            require(not (dc.PANEL_OUT_DIR/'population_ccg_year.parquet').exists(), 'Population output already exists')
            source = verified_asset('bea_cainc1','CAINC1.zip')
            destination = dc.BEA_SOURCE_DIR/'CAINC1/CAINC1.zip'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                require(sha(destination)==sha(source), 'Staged BEA source changed')
            else:
                shutil.copy2(source,destination)
            population_main(code,dc,record)
        elif name=='employment':
            checked_efsy()
            require(not (dc.EMP_OUT_DIR/'efsy_ccg_year.parquet').exists(), 'Employment output already exists')
            original_main(load('historical_employment',code/'py/11_build_employment_cells.py'),
                          ['--industry-support','national','--negative-cells','drop'])
        elif name=='dcads-fixed':
            require(resume_missing or not list(dc.OUT_DIR.glob('fy*.parquet')), 'DCADS parse outputs already exist')
            verify_dcads_sources()
            module = load('historical_fixed',code/'py/04_parse_dcads.py')
            original_layouts = dc.load_layouts
            layouts, index = original_layouts()
            if resume_missing:
                import pyarrow.parquet as pq
                completed = []
                for filename, layout in index.items():
                    output = dc.OUT_DIR/('fy'+dc.fiscal_year_of(filename)+'.parquet')
                    if output.exists():
                        saved = pq.ParquetFile(output)
                        require({'dollars','source_file','layout_id'} <= set(saved.schema.names),
                                'Incomplete existing parsed output: '+output.name)
                        completed.append(filename)
                record['reused_completed_files']=completed
                pending = {k:v for k,v in index.items() if k not in completed}
                dc.load_layouts = lambda:(layouts,pending)
                print('Explicit resume: existing completed fiscal files:',len(completed),'; pending:',len(pending),flush=True)
            try:
                original_main(module,[])
            finally:
                dc.load_layouts = original_layouts
            require(all((dc.OUT_DIR/('fy'+dc.fiscal_year_of(f)+'.parquet')).is_file() for f in index),
                    'The complete fixed-width source list is not parsed')
        elif name=='dcads-delimited':
            require(not any((dc.OUT_DIR/('fy%d.parquet'%y)).exists() for y in (2004,2005,2006)),
                    'Delimited DCADS outputs already exist')
            verify_dcads_sources()
            original_main(load('historical_delimited',code/'py/05_parse_dcads_delimited.py'),[])
        elif name=='dcads-geography':
            for prerequisite in ('dcads-fixed','dcads-delimited'):
                require(passed_stage(prerequisite), 'Incomplete DCADS parse')
            original_main(load('historical_geography',code/'py/07_add_county_fips.py'),[])
        elif name=='usaspending':
            from usaspending_bundle import production_archives
            require(years, 'Provide explicit --years')
            parser = load('historical_usaspending',code/'py/31_parse_usaspending.py')
            selected = parser.parse_years(years)
            archives = production_archives(selected,override=usaspending_root)
            record['usaspending_archive_source'] = 'package_bundle' if usaspending_root is None else 'explicit_verified_directory'
            fips_to_ccg, _, _ = dc.load_ccg_map()
            names = parser.county_name_lookup(fips_to_ccg)
            overrides = parser.load_city_county_overrides(fips_to_ccg)
            reports = []
            for asset, source in archives:
                year = asset['fiscal_year']
                require(not (dc.USASPENDING_OUT_DIR/('fy%d.parquet'%year)).exists(), 'USAspending output already exists')
                print('Parsing verified USAspending archive:',year,flush=True)
                with output_io_retries():
                    _, report = parser.parse_archive(year,source,fips_to_ccg,names,overrides)
                reports.append(report)
            # Per-selection reports avoid collisions between independent year jobs.
            write(dc.LOG_DIR/('parse_'+years+'.json'),reports)
        elif name=='county-inputs':
            import numpy as np
            import pandas as pd
            for prerequisite in ('employment','population','dcads-geography'):
                require(passed_stage(prerequisite), 'Incomplete annual source build')
            require(all((dc.USASPENDING_OUT_DIR/('fy%d.parquet'%y)).exists() for y in range(2007,2019)),
                    'USAspending annual sources incomplete')
            destination = WORKSPACE/'source51'
            require(not destination.exists(), 'Reconstructed source51 inputs already exist')
            destination.mkdir()
            producer = load('historical_source51',code/'py/51_county_multiplier_grid.py')
            panel = producer.base_panel()
            units = np.array(sorted(panel.ccg.unique()))
            require(len(units)==3058, 'Historical county support differs; do not change the screen')
            matrices = producer.contract_matrices(units,1966,2018,drop_construction=True)
            outputs = {'source51_county_panel.parquet':panel,
                       'source51_contracts_net.parquet':matrices[('net',0.0)]}
            reference = ROOT/'data/upstream/baseline_roll5_gap5'
            anchors = read(ROOT/'reference/upstream_source_manifest.json')
            comparisons = []
            for filename, frame in outputs.items():
                frame.to_parquet(destination/filename,index=(filename=='source51_contracts_net.parquet'))
                require(sha(reference/filename)==anchors['inputs'][filename], 'Independent target changed')
                old = pd.read_parquet(reference/filename)
                try:
                    pd.testing.assert_frame_equal(frame,old,check_exact=True)
                    comparison = dict(file=filename,status='EXACT_DATAFRAME_MATCH',rows=len(frame))
                except AssertionError as exc:
                    comparison = dict(file=filename,status='DIFFERENT',reason=str(exc)[:6000],rows=len(frame))
                    # Windows stores the unchanged state code as int32 rather
                    # than the historical macOS int64. Set aside ONLY that
                    # lossless storage width, retaining the strict diagnostic.
                    if filename=='source51_county_panel.parquet' and frame.state.dtype!=old.state.dtype:
                        require(np.issubdtype(frame.state.dtype,np.integer) and np.issubdtype(old.state.dtype,np.integer)
                                and np.array_equal(frame.state,old.state), 'State values or non-integer storage differ')
                        width_only = frame.astype({'state':old.state.dtype})
                        try:
                            pd.testing.assert_frame_equal(width_only,old,check_exact=True)
                            comparison.update(status='EXACT_VALUES_STATE_INTEGER_WIDTH_ONLY',
                                              state_storage_original=str(old.state.dtype),state_storage_rebuilt=str(frame.state.dtype))
                        except AssertionError:
                            pass
                comparison.update(rebuilt_sha256=sha(destination/filename),reference_sha256=anchors['inputs'][filename])
                comparisons.append(comparison)
            write(logs()/RUNLOG/'source51_input_comparison.json',dict(files=comparisons,raw_to_results_certified=False))
            require(all(c['status'] in ('EXACT_DATAFRAME_MATCH','EXACT_VALUES_STATE_INTEGER_WIDTH_ONLY') for c in comparisons),
                    'Reconstructed source51 input differs; preserve and investigate, never rewrite references')
        else:
            raise ValueError('Unknown historical stage')
        if record['status']=='RUNNING':
            record['status']='PASS'
    except Exception as exc:
        record.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        record['finished_utc']=datetime.now(timezone.utc).isoformat()
        write(certificate,record)
    print('HISTORICAL STAGE',record['status']+':',label,'; full raw-to-results certification remains false',flush=True)
