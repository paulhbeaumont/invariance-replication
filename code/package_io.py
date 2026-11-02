"""Portable, fail-closed package paths. Never consult the original project."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=True)+'\n',encoding='utf-8')


def logs():
    # One batch_logs folder at the enclosing research-project root. When this
    # package becomes its own standalone repository, its own root takes over.
    parent=next((p for p in ROOT.parents if (p/'INVARIANCE_TEST_PROTOCOL.md').exists()),ROOT)
    folder=parent/'batch_logs/ns_replication_Paul'
    folder.mkdir(parents=True,exist_ok=True)
    return folder
