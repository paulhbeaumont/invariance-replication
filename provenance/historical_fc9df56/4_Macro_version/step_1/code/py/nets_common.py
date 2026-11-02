"""Shared helpers for the NETS Fed2025 cleaning producers (63-66).

Provides the staging paths, the verified raw-parquet cache fetch from Dropbox
/_Source/NETS/Fed2025 (per-file content_hash checked against the store's
_manifest.json), and the one canonical pair of normalizers, emitted as DuckDB
SQL expressions so producers 65 and 66 share literally the same normalization.

Normalization contract:
- normalize_name: uppercase, punctuation to spaces, trailing legal-suffix
  tokens stripped (INC, LLC, CORP, CO, LTD and their long forms), whitespace
  collapsed.
- normalize_address: uppercase, punctuation to spaces, USPS-style token
  standardization (STREET->ST, NORTH->N, SUITE->STE, ...), optional stripping
  of a trailing unit designator (STE/APT/FL/... plus its token), whitespace
  collapsed. PO-box detection covers PO BOX, P O BOX, POB, GENERAL DELIVERY.
"""

import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dcads_common import MACRO_DATA_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[4]
STAGING = MACRO_DATA_ROOT / "temp" / "nets_fed2025_staging"
RAW = STAGING / "raw"
LONG = STAGING / "long"
FIRM = STAGING / "firm"
ESTAB = STAGING / "estab"
REPORTS = STAGING / "_reports"
# DuckDB spill is transient scratch, kept off the synced Dropbox tree so the
# sync client never sees the churn.
DUCK_TMP = PROJECT_ROOT / "data" / "temp" / ".tmp_duck_nets"
REMOTE_DIR = "/_Source/NETS/Fed2025"
OAUTH = Path.home() / ".claude" / "tools" / "dropbox_sweep" / ".oauth.json"
BLOCK = 4 * 1024 * 1024

TABLES = [
    "addressfirst", "company", "emp", "fips", "hq", "hqcompany",
    "misc", "move", "movesummary", "naics22", "ratings", "sales", "sic",
]


def duck_settings(con, memory_limit="64GB"):
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET memory_limit='{memory_limit}'")
    DUCK_TMP.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{DUCK_TMP}'")


# ---------------------------------------------------------------- Dropbox

_ROOT_HEADER = {}


def _mint_token():
    c = json.loads(OAUTH.read_text())
    data = urllib.parse.urlencode({
        "grant_type": "refresh_token", "refresh_token": c["refresh_token"],
        "client_id": c["app_key"], "client_secret": c["app_secret"],
    }).encode()
    req = urllib.request.Request(
        "https://api.dropboxapi.com/oauth2/token", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        token = json.loads(r.read())["access_token"]
    if not _ROOT_HEADER:
        req = urllib.request.Request(
            "https://api.dropboxapi.com/2/users/get_current_account",
            data=b"null", method="POST",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            root = json.loads(r.read())["root_info"]["root_namespace_id"]
        _ROOT_HEADER["Dropbox-API-Path-Root"] = json.dumps(
            {".tag": "root", "root": root})
    return token


def _download(token, remote_path, local_path):
    """Stream one file down, returning its Dropbox content_hash."""
    req = urllib.request.Request(
        "https://content.dropboxapi.com/2/files/download", method="POST",
        headers={"Authorization": f"Bearer {token}",
                 "Dropbox-API-Arg": json.dumps({"path": remote_path}),
                 **_ROOT_HEADER})
    hashes = []
    with urllib.request.urlopen(req, timeout=600) as r, open(local_path, "wb") as out:
        while True:
            chunk = r.read(BLOCK)
            if not chunk:
                break
            out.write(chunk)
            hashes.append(hashlib.sha256(chunk).digest())
    return hashlib.sha256(b"".join(hashes)).hexdigest()


def ensure_raw_cache(log=print):
    """Fetch the 13 parquet tables and manifest; verify each content_hash.
    Idempotent: a file whose hash already matches the manifest is kept."""
    RAW.mkdir(parents=True, exist_ok=True)
    token = _mint_token()
    man_path = RAW / "_manifest.json"
    _download(token, f"{REMOTE_DIR}/_manifest.json", man_path)
    manifest = json.loads(man_path.read_text())
    missing = [t for t in TABLES if t not in manifest]
    if missing:
        raise RuntimeError(f"store manifest lacks tables: {missing}")
    for t in TABLES:
        local = RAW / f"{t}.parquet"
        want = manifest[t]["uploaded_hash"]
        if local.exists():
            got = _hash_file(local)
            if got == want:
                continue
            log(f"[raw] {t}: stale/partial cache, refetching")
        for attempt in range(3):
            try:
                got = _download(token, f"{REMOTE_DIR}/{t}.parquet", local)
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(15 * (attempt + 1))
                token = _mint_token()
        else:
            raise RuntimeError(f"download kept failing: {t}")
        if got != want:
            raise RuntimeError(
                f"content_hash mismatch on {t}: got {got}, manifest {want}")
        log(f"[raw] {t}: fetched and verified")
    return manifest


def _hash_file(path):
    hashes = []
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(BLOCK)
            if not chunk:
                break
            hashes.append(hashlib.sha256(chunk).digest())
    return hashlib.sha256(b"".join(hashes)).hexdigest()


# ---------------------------------------------------------------- normalizers

_LEGAL_SUFFIXES = ("INCORPORATED", "CORPORATION", "COMPANY", "LIMITED",
                   "CORP", "INC", "LLC", "LTD", "CO")

_USPS_TOKENS = {
    "STREET": "ST", "AVENUE": "AVE", "BOULEVARD": "BLVD", "DRIVE": "DR",
    "ROAD": "RD", "LANE": "LN", "COURT": "CT", "PLACE": "PL",
    "HIGHWAY": "HWY", "PARKWAY": "PKWY", "SQUARE": "SQ", "TERRACE": "TER",
    "CIRCLE": "CIR", "TRAIL": "TRL", "EXPRESSWAY": "EXPY", "FREEWAY": "FWY",
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
    "NORTHEAST": "NE", "NORTHWEST": "NW", "SOUTHEAST": "SE",
    "SOUTHWEST": "SW", "SUITE": "STE", "FLOOR": "FL", "BUILDING": "BLDG",
    "APARTMENT": "APT", "ROOM": "RM", "DEPARTMENT": "DEPT",
}

_UNIT_TOKENS = ("STE", "APT", "FL", "BLDG", "RM", "UNIT", "DEPT", "#")


def _sql_clean(col):
    """Uppercase, punctuation to spaces, collapse whitespace, trim."""
    return (f"trim(regexp_replace(regexp_replace(upper({col}), "
            f"'[^A-Z0-9#]+', ' ', 'g'), ' +', ' ', 'g'))")


def sql_normalize_name(col):
    expr = _sql_clean(col)
    suffixes = "|".join(_LEGAL_SUFFIXES)
    # strip up to two trailing legal-suffix tokens (e.g. "X CO INC")
    for _ in range(2):
        expr = f"trim(regexp_replace({expr}, ' ({suffixes})$', ''))"
    return expr


def sql_normalize_address(col, strip_suite):
    expr = _sql_clean(col)
    for long, short in _USPS_TOKENS.items():
        expr = f"regexp_replace({expr}, '\\b{long}\\b', '{short}', 'g')"
    if strip_suite:
        units = "|".join(t for t in _UNIT_TOKENS if t != "#")
        expr = f"trim(regexp_replace({expr}, ' ({units}) ?[A-Z0-9]*$', ''))"
        expr = f"trim(regexp_replace({expr}, ' #[A-Z0-9]*$', ''))"
    return f"trim(regexp_replace({expr}, ' +', ' ', 'g'))"


def sql_is_pobox(col):
    n = _sql_clean(col)
    return (f"(regexp_matches({n}, '^P ?O B(OX)?\\b') OR "
            f"regexp_matches({n}, '^POB\\b') OR "
            f"regexp_matches({n}, 'GENERAL DELIVERY'))")
