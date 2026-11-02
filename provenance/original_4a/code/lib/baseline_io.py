"""Paths and fail-closed input checks for the selected baseline."""
import hashlib
import json
import os
from pathlib import Path

APP = Path(__file__).resolve().parents[2]
ROOT = APP.parents[1]
DEFAULT_SHARED = Path.home()/'Library/CloudStorage/Dropbox-Matray/_AdrienTeam/Chenzi_Adrien/Invariance_Agg_Disagg'
SHARED = Path(os.environ.get('INVARIANCE_DROPBOX_ROOT', DEFAULT_SHARED))
DATA = SHARED/'4_Macro_version/4a_PaperApplication/data'
CACHE = DATA/'analysis/baseline_roll5_gap5'
LOG = ROOT/'batch_logs/macro_paper_application'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def locked():
    lock = read_json(APP/'config/baseline_lock.json')
    require(sha(APP/'config/baseline.json')==lock['config_sha256'],'Baseline configuration differs from its approved lock')
    require(sha(APP/'config/baseline_sources.json')==lock['sources_sha256'],'Source inventory differs from its approved lock')
    spec = read_json(APP/'config/baseline.json')
    sources = read_json(APP/'config/baseline_sources.json')
    require(sha(APP/'code/lib/baseline_math.py')==sources['math_sha256'],'Extracted estimator differs from source snapshot')
    require(sha(APP/'tests/fixtures/baseline_roll5_gap5_reference.csv')==sources['reference_sha256'],'Reference fixture differs from saved grid row')
    return spec,sources,lock


def load_cache():
    import pandas as pd
    spec,sources,lock = locked()
    cert = read_json(CACHE/'certificate.json')
    require(cert['status']=='PASS','Analysis cache is not certified')
    require(cert['lock']==lock,'Cache belongs to a different specification')
    require(sha(CACHE/'county.parquet')==cert['cache_sha256'],'Analysis cache hash mismatch')
    require(sha(APP/'code/build/baseline.py')==cert['builder_sha256'],'Cache builder differs from certificate')
    require(sha(APP/'code/lib/baseline_io.py')==cert['io_sha256'],'Cache I/O rules differ from certificate')
    frame=pd.read_parquet(CACHE/'county.parquet')
    require(len(frame)==spec['n_observations'],'Unexpected cached observation count')
    require(not frame.duplicated(['ccg','year']).any(),'Duplicate county-year cache keys')
    return frame,spec,cert
