"""Run unchanged historical source producers, then the locked paper application.

Each run begins in a new directory. Existing raw archives are allowed only as
checksum-verified external sources. Prepared analysis caches are never seeds.
Comparison targets remain in the original package and are not regenerated.
"""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import re
import subprocess
import sys

from package_io import ROOT,logs,read,require,sha,write
import historical_rebuild as historical


def utc():
    return datetime.now(timezone.utc).isoformat()


def configure(run_id):
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,64}',run_id) is not None,'Unsafe run identifier')
    run = ROOT/'data/raw_rebuild/runs'/run_id
    historical.WORKSPACE = run
    historical.RUNLOG = 'full_raw_'+run_id
    historical.EFSY_SOURCE = run/'sources/efsy_panel_naics.parquet'
    historical.EFSY_CERTIFICATE = logs()/historical.RUNLOG/'efsy_conversion.json'
    return run,logs()/historical.RUNLOG


def initialize(run_id):
    run,record_dir = configure(run_id)
    require(not run.exists(),'Clean verification requires a NEW run-id; do not overwrite a prior run')
    historical.verified_code()
    run.mkdir(parents=True)
    record = dict(run_id=run_id,created_utc=utc(),initial_derived_tree_empty=True,
                  historical_commit=historical.COMMIT,specification_sha256=sha(ROOT/'docs/SPECIFICATION.md'),
                  reference_manifest_sha256=sha(ROOT/'provenance/input_manifest.json'),
                  historical_code_manifest_sha256=sha(historical.MANIFEST),
                  raw_source_catalog_sha256=sha(ROOT/'provenance/raw_source_catalog.json'),
                  usaspending_catalog_sha256=sha(ROOT/'provenance/usaspending_source_catalog_20261023.json'),
                  raw_to_results_certified=False)
    write(run/'run.json',record)
    write(record_dir/'initial_state.json',record)
    print('NEW CLEAN RUN:',run,flush=True)


def checked_run(run_id):
    run,record_dir = configure(run_id)
    control = read(run/'run.json')
    require(control['run_id']==run_id and control['initial_derived_tree_empty'],'Uncertified initial state')
    for key,path in [('specification_sha256',ROOT/'docs/SPECIFICATION.md'),
                     ('reference_manifest_sha256',ROOT/'provenance/input_manifest.json'),
                     ('historical_code_manifest_sha256',historical.MANIFEST),
                     ('raw_source_catalog_sha256',ROOT/'provenance/raw_source_catalog.json'),
                     ('usaspending_catalog_sha256',ROOT/'provenance/usaspending_source_catalog_20261023.json')]:
        require(sha(path)==control[key],'Run definition or independent reference manifest changed: '+key)
    return run,record_dir


def annual(run_id):
    run,record_dir = checked_run(run_id)
    from raw_sources import build_efsy
    if historical.EFSY_SOURCE.exists():
        historical.checked_efsy()
        print('Resume this run only: verified completed EFSY source conversion.',flush=True)
    else:
        with historical.output_io_retries():
            build_efsy(historical.EFSY_SOURCE,historical.EFSY_CERTIFICATE)
    for name in ('employment','dcads-fixed','dcads-delimited','dcads-geography','usaspending','population','county-inputs'):
        label = 'usaspending_2007-2018' if name=='usaspending' else name
        if historical.passed_stage(label):
            print('Resume this run only: completed',name,flush=True)
            continue
        previous = list(record_dir.glob(name+'*_stage.json'))
        attempt = 1 if not previous else 1+max([1]+[int(re.search(r'_attempt(\d+)_stage',p.name).group(1)) for p in previous if '_attempt' in p.name])
        historical.stage(name,years='2007-2018' if name=='usaspending' else None,
                         attempt=attempt,resume_missing=(name=='dcads-fixed' and bool(previous)))
    write(record_dir/'annual_complete.json',dict(status='PASS',finished_utc=utc(),
          workspace=run.relative_to(ROOT).as_posix(),raw_to_results_certified=False,
          all_annual_data_generated_within_this_run=True,
          population_original_audit_status='FAILED',population_application_warning='KALAWAO_15005_OUTSIDE_FIXED_PAPER_SAMPLE'))
    print('FRESH ANNUAL SOURCE BUILD PASS',flush=True)


def sources(run_id):
    _,record_dir=checked_run(run_id)
    from raw_sources import download
    import usaspending_bundle as bundle
    download(['efsy','bea_cainc1','dcads','naics_concordances','bea_areas_2004'],
             certificate=record_dir/'source_acquisition.json')
    catalog=bundle.checked_catalog();bundle.check_metadata(catalog)
    archives=[]
    for asset in catalog['assets']:
        archives.append(bundle.verify_file(bundle.bundle_path(bundle.archive_relative(asset).as_posix()),asset))
        print('RAW SOURCE ARCHIVE PASS',asset['id'],flush=True)
    write(record_dir/'sources.json',dict(status='PASS',archives=archives,public_source_records=220,
          exact_preserved_usaspending_vintage=True,raw_to_results_certified=False))


def recorded_stage(run_id,name,callback,attempt=1):
    run,record_dir = checked_run(run_id)
    require(attempt>=1,'Attempt must be positive')
    label=name+('_attempt'+str(attempt) if attempt>1 else '')
    path = record_dir/(label+'.json')
    require(not path.exists(),'Stage record already exists; preserve the prior attempt')
    record = dict(status='RUNNING',started_utc=utc(),run_id=run_id,attempt=attempt,raw_to_results_certified=False,
                  interpreter=sys.version,controller_sha256=sha(__file__),
                  packages={name:getattr(sys.modules[name],'__version__',None) for name in ('numpy','pandas','pyarrow','duckdb','scipy') if name in sys.modules})
    write(path,record)
    try:
        callback(run,record_dir)
        record['status']='PASS'
    except Exception as exc:
        record.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        record['finished_utc']=utc()
        write(path,record)


def application(run_id,attempt=1,resume_annual_attempt=None):
    from raw_application import build
    recorded_stage(run_id,'application',lambda run,records:build(run,records,attempt,resume_annual_attempt),attempt)


def estimate(run_id):
    from raw_application import estimate as run_estimates
    recorded_stage(run_id,'estimate',run_estimates)


def render(run_id):
    from raw_application import render as run_render
    recorded_stage(run_id,'render',run_render)


def check(run_id):
    from raw_application import final_check
    final_check(*checked_run(run_id))


def analysis_environment(python):
    executable = Path(python).resolve()
    require(executable.is_file(),'Analysis interpreter does not exist: '+str(executable))
    env = dict(os.environ)
    if os.name=='nt':
        parent = executable.parent
        env['PATH']=os.pathsep.join([str(parent),str(parent/'Library/bin'),str(parent/'Scripts'),env.get('PATH','')])
    return executable,env


def analysis_preflight(python):
    executable,env=analysis_environment(python)
    code="import sys;sys.path.insert(0,"+repr(str(ROOT/'code'))+");import raw_application;print('ANALYSIS ENVIRONMENT READY')"
    subprocess.run([str(executable),'-B','-c',code],check=True,env=env,cwd=ROOT)


def analysis_subprocess(stage,run_id,python):
    executable,env=analysis_environment(python)
    command = [str(executable),'-B',str(ROOT/'raw_to_results.py'),stage,'--run-id',run_id]
    subprocess.run(command,check=True,env=env,cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['init','sources','annual','application','estimate','render','check','all'])
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--attempt',type=int,default=1,help='Explicit new application-build attempt; earlier failed outputs and records are preserved')
    parser.add_argument('--resume-annual-from-attempt',type=int,default=None,
                        help='Explicitly reuse independently verified annual outputs from an earlier failed application attempt in this same clean run')
    parser.add_argument('--analysis-python',default=None,
                        help='Interpreter with the pinned analysis requirements; raw producers use the calling environment')
    args = parser.parse_args()
    if args.stage=='all':
        require(args.analysis_python,'For all, provide --analysis-python pointing to the pinned analysis environment; raw and analysis requirements use different pandas versions')
        analysis_preflight(args.analysis_python)
        initialize(args.run_id)
        sources(args.run_id)
        annual(args.run_id)
        for stage in ('application','estimate','render','check'):
            analysis_subprocess(stage,args.run_id,args.analysis_python)
    elif args.stage=='init':
        initialize(args.run_id)
    elif args.stage=='application':
        application(args.run_id,args.attempt,args.resume_annual_from_attempt)
    else:
        globals()[args.stage](args.run_id)


if __name__=='__main__':
    main()
