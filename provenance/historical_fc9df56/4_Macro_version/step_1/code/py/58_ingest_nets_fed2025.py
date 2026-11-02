"""Convert the Fed 2025 NETS delivery from CSV to parquet in Dropbox.

Source: <Dropbox home>/Adrien Matray/panjiva/data/nets/*.csv, the 13-table
NETS establishment suite (DunsNumber-keyed). The CSVs live in Dropbox as
online-only placeholders; reading them hydrates each file in full, so the
pipeline processes one file at a time, smallest first, and evicts the local
copy (fileproviderctl evict) once its parquet is safely uploaded.

Destination: Dropbox /_Source/NETS/Fed2025/<stem>.parquet (zstd). That folder
is excluded from this machine's selective sync, so the parquet is written to a
local staging directory and uploaded through the Dropbox API (chunked upload
session, verified against Dropbox's content_hash), then the staging copy is
deleted. Nothing heavy remains local when a file completes.

Typing rules: DuckDB sniffs types from the first 100k rows, except that any
column whose name contains duns, zip, fips, sic, naics, cbsa, msa, or phone is
forced to VARCHAR. These are identifier codes with meaningful leading zeros
(DUNS 015588676, FIPS 01xxx, SIC 0111); an integer type would corrupt them.
VARCHAR columns are rtrim'd because the CSVs pad strings with trailing spaces.

Idempotent: a file whose manifest entry records an uploaded content_hash is
skipped, so the pipeline can be re-run after an interruption. The manifest
lives in <project root>/batch_logs/nets_fed2025/manifest.json.
"""

import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dcads_common import DROPBOX_HOME, MACRO_DATA_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC = DROPBOX_HOME / "Adrien Matray" / "panjiva" / "data" / "nets"
STAGING = MACRO_DATA_ROOT / "temp" / "nets_fed2025_staging"
MANIFEST = PROJECT_ROOT / "batch_logs" / "nets_fed2025" / "manifest.json"
REMOTE_DIR = "/_Source/NETS/Fed2025"
OAUTH = Path.home() / ".claude" / "tools" / "dropbox_sweep" / ".oauth.json"
FORCE_VARCHAR = re.compile(r"duns|zip|fips|sic|naics|cbsa|msa|phone", re.I)
SNIFF_ROWS = 100_000
BLOCK = 4 * 1024 * 1024          # Dropbox content_hash block size
CHUNK = 36 * BLOCK               # 144 MB upload chunks, block-aligned


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(manifest):
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def allocated_bytes(path):
    out = subprocess.run(
        ["stat", "-f", "%b", str(path)], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip()) * 512


# ---------------------------------------------------------------- Dropbox API

_ROOT_HEADER = {}


def mint_token():
    c = json.loads(OAUTH.read_text())
    data = urllib.parse.urlencode({
        "grant_type": "refresh_token", "refresh_token": c["refresh_token"],
        "client_id": c["app_key"], "client_secret": c["app_secret"],
    }).encode()
    req = urllib.request.Request(
        "https://api.dropboxapi.com/oauth2/token", data=data, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        token = json.loads(r.read())["access_token"]
    if not _ROOT_HEADER:
        # Team-rooted account: unheadered calls address the member home
        # namespace, but /_Source lives at the team root the local mount shows.
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


def api_call(token, url, arg, body=b"", timeout=600):
    for attempt in range(5):
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {token}",
            "Dropbox-API-Arg": json.dumps(arg),
            "Content-Type": "application/octet-stream",
            **_ROOT_HEADER,
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}"), token
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            if e.code == 401:
                token = mint_token()
            elif e.code == 429 or e.code >= 500:
                time.sleep(15 * (attempt + 1))
            else:
                raise RuntimeError(f"Dropbox API {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError):
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"Dropbox API kept failing: {url}")


def upload(path, remote_path):
    """Chunked upload; returns Dropbox content_hash after verifying it matches
    the locally computed hash of what was sent."""
    token = mint_token()
    block_hashes = []
    offset = 0
    session = None
    content = "https://content.dropboxapi.com/2/files/upload_session"
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if session is None:
                out, token = api_call(token, f"{content}/start",
                                      {"close": False}, chunk)
                session = out["session_id"]
            elif chunk:
                _, token = api_call(
                    token, f"{content}/append_v2",
                    {"cursor": {"session_id": session, "offset": offset},
                     "close": False}, chunk)
            for i in range(0, len(chunk), BLOCK):
                block_hashes.append(hashlib.sha256(chunk[i:i + BLOCK]).digest())
            offset += len(chunk)
            if len(chunk) < CHUNK:
                break
    out, token = api_call(
        token, f"{content}/finish",
        {"cursor": {"session_id": session, "offset": offset},
         "commit": {"path": remote_path, "mode": "overwrite",
                    "autorename": False, "mute": True}})
    local_hash = hashlib.sha256(b"".join(block_hashes)).hexdigest()
    if out["content_hash"] != local_hash:
        raise RuntimeError(
            f"content_hash mismatch on {remote_path}: "
            f"local {local_hash} vs remote {out['content_hash']}")
    if out["size"] != path.stat().st_size:
        raise RuntimeError(f"size mismatch on {remote_path}")
    return out["content_hash"]


# ---------------------------------------------------------------- conversion

def sniff_schema(con, csv_path):
    rows = con.execute(
        "DESCRIBE SELECT * FROM read_csv(?, auto_detect=true, sample_size=?)",
        [str(csv_path), SNIFF_ROWS],
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def convert(con, csv_path, out_path):
    """Read every column as VARCHAR, then cast to the sniffed type with
    blank-as-NULL handling: NETS pads fields with spaces and writes missing
    numerics as blanks, which breaks typed CSV reads mid-file. CAST (not
    TRY_CAST) keeps the conversion fail-loud on genuinely malformed values."""
    sniffed = sniff_schema(con, csv_path)
    schema = [
        (name, "VARCHAR" if FORCE_VARCHAR.search(name) else dtype)
        for name, dtype in sniffed
    ]
    select = ", ".join(
        f'rtrim("{name}") AS "{name}"' if dtype == "VARCHAR"
        else f'CAST(NULLIF(trim("{name}"), \'\') AS {dtype}) AS "{name}"'
        for name, dtype in schema
    )
    # Every tier applies SELECT DISTINCT and a numeric-DunsNumber filter: the
    # delivery concatenates overlapping data pulls, so company, hqcompany, and
    # addressfirst ship 1.7x-4.8x exact duplicate rows plus stray mid-file
    # header lines (all three collapse to the same 96,116,441 DUNS records the
    # other tables carry). Exact duplicates carry no information and would
    # multiply matches in any DUNS merge.
    #
    # Parse tiers: (1) strict without quoting, the format most of the files
    # use; (2) strict with quote='"', for files where fields containing commas
    # are quoted (company, hqcompany) so those rows parse intact; (3) lenient,
    # for corrupt stretches (sales ships a run of NUL bytes).
    first = schema[0][0]
    tiers = [
        ("strict", ""),
        ("requote", ", quote='\"', escape='\"'"),
        ("salvage", ", quote='\"', escape='\"', strict_mode=false, null_padding=true"),
    ]
    source_rows = con.execute(
        "SELECT count(*) FROM read_csv(?, auto_detect=true, sample_size=?, "
        "all_varchar=true, quote='\"', escape='\"', strict_mode=false, "
        "null_padding=true)",
        [str(csv_path), SNIFF_ROWS],
    ).fetchone()[0]
    copied = tier_used = None
    for tier, opts in tiers:
        try:
            copied = con.execute(
                f"""
                COPY (
                    SELECT DISTINCT {select}
                    FROM read_csv(?, auto_detect=true, sample_size=?,
                                  all_varchar=true{opts})
                    WHERE rtrim("{first}") SIMILAR TO '[0-9]+'
                ) TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)
                """,
                [str(csv_path), SNIFF_ROWS],
            ).fetchone()[0]
            tier_used = tier
            break
        except duckdb.Error as exc:
            if "CSV Error" not in str(exc) or tier == "salvage":
                raise
            log(f"[{tier}-failed] {csv_path.stem}: {str(exc).splitlines()[0]}")
    if tier_used != "strict":
        log(f"[{tier_used}] {csv_path.stem}: kept {copied:,} distinct rows "
            f"of {source_rows:,} source rows")
    check = con.execute(
        "SELECT count(*) FROM read_parquet(?)", [str(out_path)]
    ).fetchone()[0]
    if check != copied:
        raise RuntimeError(f"row mismatch: copied {copied}, parquet has {check}")
    return copied, schema, source_rows, tier_used


def evict(csv_path):
    """Programmatic eviction is unavailable: this macOS's fileproviderctl has
    no evict subcommand, and NSFileProviderManager.evictItem is entitlement-
    gated to Dropbox's own app. The hydrated CSVs must be released by hand in
    Finder (select the folder, Remove Download). Returns local bytes so the
    caller can report what is still on disk."""
    return allocated_bytes(csv_path)


def main():
    STAGING.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    files = sorted(SRC.glob("*.csv"), key=lambda p: p.stat().st_size)
    if not files:
        log(f"no CSV files under {SRC}")
        sys.exit(1)
    log(f"{len(files)} CSVs, {sum(f.stat().st_size for f in files)/1e9:.0f} GB logical")

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='16GB'")

    failures = 0
    for f in files:
        stem = f.stem
        staged = STAGING / f"{stem}.parquet"
        entry = manifest.get(stem, {})
        if entry.get("uploaded_hash"):
            log(f"[skip] {stem}: already uploaded ({entry['rows']:,} rows)")
        else:
            log(f"[start] {stem}: {f.stat().st_size/1e9:.1f} GB CSV")
            t0 = time.time()
            try:
                rows, schema, source_rows, tier = convert(con, f, staged)
                log(f"[converted] {stem}: {rows:,} rows "
                    f"(of {source_rows:,} source), "
                    f"{staged.stat().st_size/1e9:.1f} GB parquet")
                content_hash = upload(staged, f"{REMOTE_DIR}/{stem}.parquet")
            except Exception as exc:
                failures += 1
                log(f"[fail] {stem}: {exc}")
                staged.unlink(missing_ok=True)
                continue
            manifest[stem] = {
                "rows": rows,
                "rows_source": source_rows,
                "parse_tier": tier,
                "csv_bytes": f.stat().st_size,
                "parquet_bytes": staged.stat().st_size,
                "uploaded_hash": content_hash,
                "remote": f"{REMOTE_DIR}/{stem}.parquet",
                "schema": [list(c) for c in schema],
                "minutes": round((time.time() - t0) / 60, 1),
            }
            save_manifest(manifest)
            staged.unlink()
            log(f"[ok] {stem}: uploaded and verified, "
                f"{manifest[stem]['minutes']} min")
        alloc = allocated_bytes(f)
        if alloc > 1_000_000:
            alloc = evict(f)
            state = "evicted" if alloc < 1_000_000 else f"STILL LOCAL ({alloc/1e9:.1f} GB)"
            manifest.setdefault(stem, {})["evict"] = state
            save_manifest(manifest)
            log(f"[evict] {stem}: {state}")

    if failures:
        log(f"PIPELINE FAILED: {failures} file(s) failed")
        sys.exit(1)
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
