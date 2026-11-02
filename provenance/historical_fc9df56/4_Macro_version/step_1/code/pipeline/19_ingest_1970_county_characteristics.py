#!/usr/bin/env python3
"""Build fixed 1970 county characteristics from official Census files.

The sources are the U.S. Census Bureau's County and City Data Book, 1972, and
1970 Fourth Count File 4C, Population, distributed by the National Archives,
plus the Census Bureau's county age-sex-race file. The fixed-width county
records provide a common 1970 cross-section for the BHJ shock-level balance
diagnostic.

Output:
    4_Macro_version/data/temp/county_characteristics_1970/
        county_characteristics_1970_ccg.parquet
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
import dcads_common as dc  # noqa: E402


URL = (
    "https://nara-media.s3.amazonaws.com/electronic-records/rg-029/CCDB/"
    "RG029.CCDB72"
)
EXPECTED_SHA256 = "c5c0aaf5a93131d0ce08419e9ce02cbaa7c7bc5819e1426a476472fbac934111"
EXPECTED_BYTES = 12_259_856
RACE_URL = (
    "https://www2.census.gov/programs-surveys/popest/tables/1900-1980/"
    "counties/asrh/co-asr-1970.xls"
)
RACE_SHA256 = "e90411a687bb3901b31cf6ea3fb77d5748e1726825c45a31aafd09f4a1592d3b"
RACE_BYTES = 6_750_720
EXPECTED_RECORDS = 4_658
EXPECTED_COUNTIES = 3_141
FOURTH_COUNT_URL = (
    "https://catalog.archives.gov/medialz/electronic-records/rg-029/"
    "CEN70SSF/CEN70.SSF.4CP.zip"
)
FOURTH_COUNT_SHA256 = "5a13faa1d4e2a4207afb2fc1b4309c9f4e05572dd7f6061e733a2a3e5308a825"
FOURTH_COUNT_BYTES = 164_364_620
FOURTH_COUNT_MEMBER = "CEN70.CT4CP.USCO.TXT"
FOURTH_COUNT_MEMBER_BYTES = 311_911_920
FOURTH_COUNT_PHYSICAL_BYTES = 2_040
EXPECTED_NATIONAL_POPULATION = 203_212_877
EXPECTED_NATIONAL_FOREIGN_BORN = 9_619_302

ROOT = dc.DROPBOX_PROJECT_ROOT / "4_Macro_version"
SOURCE = ROOT / "data" / "source" / "1970_county_characteristics" / "raw"
OUT = ROOT / "data" / "temp" / "county_characteristics_1970"
RAW = SOURCE / "RG029.CCDB72"
RECEIPT = SOURCE / "RG029.CCDB72.download.json"
RACE_RAW = SOURCE / "co-asr-1970.xls"
RACE_RECEIPT = SOURCE / "co-asr-1970.download.json"
DOCUMENTATION = SOURCE / "nara_ccdb_1972_documentation.pdf"
DOCUMENTATION_SHA256 = "bb7a39f82fa85ea981ebdc3eed174f0f9fdc5eb9ac7c32096aaf91a0c30bb5ae"
FOURTH_COUNT_RAW = SOURCE / "CEN70.SSF.4CP.zip"
FOURTH_COUNT_RECEIPT = SOURCE / "CEN70.SSF.4CP.download.json"
FOURTH_COUNT_DOCUMENTATION = (
    SOURCE / "nara_census_1970_fourth_count_population_documentation.pdf"
)
FOURTH_COUNT_DOCUMENTATION_SHA256 = (
    "9441cabc7a5ed5d8b53d50896421015d36a6b5a95f483f39d729e34bf154f6fc"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def acquire_file(url: str, target: Path, receipt: Path, expected_bytes: int,
                 expected_hash: str, extra: dict | None = None) -> None:
    """Download one pinned official file and write its identity receipt."""
    downloaded = False
    if not target.exists():
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as fh:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                fh.write(block)
        tmp.replace(target)
        downloaded = True
    digest = sha256(target)
    size = target.stat().st_size
    if size != expected_bytes or digest != expected_hash:
        raise SystemExit(
            f"FAIL: source identity mismatch for {target.name}, "
            f"bytes={size}, sha256={digest}"
        )
    record = {
        "url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "downloaded_this_run": downloaded,
        "bytes": size,
        "sha256": digest,
    }
    record.update(extra or {})
    receipt.write_text(
        json.dumps(
            record,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def acquire() -> None:
    """Acquire and verify the official Census source files."""
    SOURCE.mkdir(parents=True, exist_ok=True)
    if not DOCUMENTATION.exists():
        raise SystemExit(
            "FAIL: the CCDB documentation is absent. Retrieve it through "
            "FETCH_PROTOCOL_CONTRACT v1 before running this producer."
        )
    doc_hash = sha256(DOCUMENTATION)
    if doc_hash != DOCUMENTATION_SHA256:
        raise SystemExit(f"FAIL: CCDB documentation hash drifted: {doc_hash}")
    if not FOURTH_COUNT_DOCUMENTATION.exists():
        raise SystemExit(
            "FAIL: the 1970 Fourth Count documentation is absent. Retrieve it "
            "through FETCH_PROTOCOL_CONTRACT v1 before running this producer."
        )
    fourth_count_doc_hash = sha256(FOURTH_COUNT_DOCUMENTATION)
    if fourth_count_doc_hash != FOURTH_COUNT_DOCUMENTATION_SHA256:
        raise SystemExit(
            "FAIL: 1970 Fourth Count documentation hash drifted: "
            f"{fourth_count_doc_hash}"
        )
    acquire_file(
        URL, RAW, RECEIPT, EXPECTED_BYTES, EXPECTED_SHA256,
        {"documentation_sha256": doc_hash},
    )
    acquire_file(RACE_URL, RACE_RAW, RACE_RECEIPT, RACE_BYTES, RACE_SHA256)
    acquire_file(
        FOURTH_COUNT_URL,
        FOURTH_COUNT_RAW,
        FOURTH_COUNT_RECEIPT,
        FOURTH_COUNT_BYTES,
        FOURTH_COUNT_SHA256,
        {"documentation_sha256": fourth_count_doc_hash},
    )


def field(record: str, number: int, decimals: int = 0) -> float:
    """Read one CCDB flag-plus-tally field using the documented positions."""
    flag_pos = number * 10 + 60
    flag = record[flag_pos]
    tally = record[flag_pos + 1 : flag_pos + 10]
    if flag == "0":
        value = int(tally.replace("_", "-").strip() or "0")
        return value / (10**decimals)
    if flag == "4":
        return 0.0
    if flag in "123567":
        return np.nan
    raise ValueError(f"unrecognized flag {flag!r} in field {number}")


def parse_counties() -> pd.DataFrame:
    """Parse the 3,141 table-2 county and county-equivalent records."""
    raw = RAW.read_bytes()
    records = raw.splitlines()
    if len(records) != EXPECTED_RECORDS:
        raise SystemExit(f"FAIL: expected {EXPECTED_RECORDS:,} records, found {len(records):,}")
    if any(len(record) != 2_630 for record in records):
        lengths = sorted(set(map(len, records)))
        raise SystemExit(f"FAIL: CCDB record lengths are {lengths}, expected only 2630")

    rows = []
    for blob in records:
        record = blob.decode("ascii")
        is_dc_total = record[0] == "2" and record[1:3] == "11" and not record[3:6].strip()
        if record[0] != "2" or (not record[3:6].strip() and not is_dc_total):
            continue
        fips = "11001" if is_dc_total else record[1:3] + record[3:6]
        pop = field(record, 3)
        civilian_labor_force = field(record, 38)
        employed = field(record, 42)
        age25 = field(record, 26)
        families = field(record, 54)
        occupied_housing_units = field(record, 99)
        rural_nonfarm = field(record, 218)
        farm = field(record, 219)
        rows.append(
            {
                "county_fips5": fips,
                "county_name_1970": record[20:70].strip(),
                "population_1970": pop,
                "population_age65plus_1970": pop
                * field(record, 14, decimals=1)
                / 100.0,
                "rural_population_1970": rural_nonfarm + farm,
                "population_age25plus_1970": age25,
                "high_school_population_age25plus_1970": age25
                * field(record, 29, decimals=1)
                / 100.0,
                "college_population_age25plus_1970": age25
                * field(record, 30, decimals=1)
                / 100.0,
                "employed_civilian_1970": employed,
                "civilian_labor_force_1970": civilian_labor_force,
                "female_civilian_labor_force_1970": field(record, 39),
                "unemployed_civilian_1970": civilian_labor_force
                * field(record, 41, decimals=1)
                / 100.0,
                "manufacturing_employment_1970": employed
                * field(record, 43, decimals=1)
                / 100.0,
                "government_employment_1970": employed
                * field(record, 48, decimals=1)
                / 100.0,
                "professional_managerial_employment_1970": employed
                * field(record, 49, decimals=1)
                / 100.0,
                "nontradable_proxy_employment_1970": employed
                * sum(field(record, f, decimals=1) for f in (44, 45, 46, 47))
                / 100.0,
                "aggregate_money_income_1969": pop * field(record, 71),
                "families_1970": families,
                "families_below_low_income_1969": families
                * field(record, 73, decimals=1)
                / 100.0,
                "occupied_housing_units_1970": occupied_housing_units,
                "owner_occupied_housing_units_1970": occupied_housing_units
                * field(record, 101, decimals=1)
                / 100.0,
            }
        )
    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_COUNTIES or not out["county_fips5"].is_unique:
        raise SystemExit(
            f"FAIL: parsed {len(out):,} county rows; expected {EXPECTED_COUNTIES:,} unique rows"
        )
    return out


def race_counts() -> pd.DataFrame:
    """Read the Census county age-sex-race estimates for July 1, 1970."""
    data = pd.read_excel(RACE_RAW, sheet_name="1970", header=5)
    data = data.dropna(subset=["FIPS State and County Codes", "Race/Sex Indicator"])
    if len(data) != 18_846:
        raise SystemExit(f"FAIL: expected 18,846 county-race-sex rows, found {len(data):,}")
    age_cols = list(data.columns[3:])
    data["count"] = data[age_cols].sum(axis=1)
    data["county_fips5"] = (
        data["FIPS State and County Codes"].astype(int).astype(str).str.zfill(5)
    )
    groups = set(data["Race/Sex Indicator"].unique())
    expected = {
        "White male", "White female", "Black male", "Black female",
        "Other races male", "Other races female",
    }
    if groups != expected:
        raise SystemExit(f"FAIL: unexpected race-sex groups: {sorted(groups)}")
    total = data.groupby("county_fips5", sort=True)["count"].sum()
    black = (
        data[data["Race/Sex Indicator"].str.startswith("Black")]
        .groupby("county_fips5", sort=True)["count"]
        .sum()
        .reindex(total.index)
    )
    out = pd.DataFrame(
        {
            "county_fips5": total.index,
            "race_population_1970": total.to_numpy(np.float64),
            "black_population_1970": black.to_numpy(np.float64),
        }
    )
    if len(out) != EXPECTED_COUNTIES or out.isna().any().any():
        raise SystemExit(f"FAIL: race file produced {len(out):,} incomplete county rows")
    return out


def fourth_count_tallies(blocks: list[bytes]) -> list[int]:
    """Read the 1,118 population tallies from one five-block data record."""
    if len(blocks) != 5 or any(
        len(block) != FOURTH_COUNT_PHYSICAL_BYTES for block in blocks
    ):
        raise ValueError("Fourth Count data record is not five complete physical blocks")
    raw = (
        blocks[0][1_144:2_040]
        + blocks[1]
        + blocks[2]
        + blocks[3]
        + blocks[4][:1_928]
    )
    if len(raw) != 1_118 * 8:
        raise ValueError(f"Fourth Count population segment has {len(raw):,} bytes")
    values = []
    for index in range(1_118):
        token = raw[index * 8 : (index + 1) * 8].decode("ascii").strip()
        try:
            values.append(int(token or "0"))
        except ValueError as exc:
            raise ValueError(
                f"invalid Fourth Count tally at population field {index + 1}: {token!r}"
            ) from exc
    return values


def foreign_born_counts() -> pd.DataFrame:
    """Read exact county foreign-born counts from 1970 Fourth Count File 4C."""
    rows: dict[str, dict[str, int]] = {}
    physical_records = 0
    data_record_types = {f"{value:02d}" for value in range(1, 13)}
    with zipfile.ZipFile(FOURTH_COUNT_RAW) as archive:
        info = archive.getinfo(FOURTH_COUNT_MEMBER)
        if info.file_size != FOURTH_COUNT_MEMBER_BYTES:
            raise SystemExit(
                f"FAIL: {FOURTH_COUNT_MEMBER} has {info.file_size:,} bytes, "
                f"expected {FOURTH_COUNT_MEMBER_BYTES:,}"
            )
        with archive.open(FOURTH_COUNT_MEMBER) as source:
            while True:
                first = source.read(FOURTH_COUNT_PHYSICAL_BYTES)
                if not first:
                    break
                if len(first) != FOURTH_COUNT_PHYSICAL_BYTES or first[119:120] != b"$":
                    raise SystemExit(
                        "FAIL: malformed Fourth Count logical-record boundary at "
                        f"physical record {physical_records + 1:,}"
                    )
                record_type = first[117:119].decode("ascii")
                if record_type in data_record_types:
                    block_count = 5
                elif record_type in {"13", "14"}:
                    block_count = 2
                else:
                    raise SystemExit(f"FAIL: unknown Fourth Count record type {record_type!r}")
                blocks = [first]
                for _ in range(block_count - 1):
                    block = source.read(FOURTH_COUNT_PHYSICAL_BYTES)
                    if len(block) != FOURTH_COUNT_PHYSICAL_BYTES:
                        raise SystemExit("FAIL: truncated Fourth Count logical record")
                    blocks.append(block)
                physical_records += block_count

                # Types 01, 05, and 09 partition the total population into
                # urban, rural nonfarm, and rural farm components.
                if record_type not in {"01", "05", "09"}:
                    continue
                fips = (first[1:3] + first[69:72]).decode("ascii")
                tallies = fourth_count_tallies(blocks)
                population = sum(tallies[:54])  # Table 17: age by sex.
                foreign_stock = tallies[88]  # Table 21: total foreign stock.
                native_foreign_parentage = sum(tallies[89:122])
                foreign_born = sum(tallies[122:155])  # Table 22: foreign born.
                if native_foreign_parentage + foreign_born != foreign_stock:
                    raise SystemExit(
                        "FAIL: Fourth Count Tables 21 and 22 do not reconcile "
                        f"for county {fips}, record type {record_type}"
                    )
                row = rows.setdefault(
                    fips,
                    {
                        "fourth_count_population_1970": 0,
                        "foreign_born_population_1970": 0,
                    },
                )
                row["fourth_count_population_1970"] += population
                row["foreign_born_population_1970"] += foreign_born

    expected_physical_records = FOURTH_COUNT_MEMBER_BYTES // FOURTH_COUNT_PHYSICAL_BYTES
    if physical_records != expected_physical_records:
        raise SystemExit(
            f"FAIL: parsed {physical_records:,} physical records, "
            f"expected {expected_physical_records:,}"
        )
    out = (
        pd.DataFrame.from_dict(rows, orient="index")
        .rename_axis("county_fips5")
        .reset_index()
    )
    if len(out) != EXPECTED_COUNTIES or not out["county_fips5"].is_unique:
        raise SystemExit(
            f"FAIL: Fourth Count produced {len(out):,} county rows; "
            f"expected {EXPECTED_COUNTIES:,} unique rows"
        )
    if int(out["fourth_count_population_1970"].sum()) != EXPECTED_NATIONAL_POPULATION:
        raise SystemExit("FAIL: Fourth Count county population does not match the national total")
    if int(out["foreign_born_population_1970"].sum()) != EXPECTED_NATIONAL_FOREIGN_BORN:
        raise SystemExit("FAIL: Fourth Count foreign-born count does not match the national total")
    if (
        (out["fourth_count_population_1970"] <= 0).any()
        or (out["foreign_born_population_1970"] < 0).any()
        or (
            out["foreign_born_population_1970"]
            > out["fourth_count_population_1970"]
        ).any()
    ):
        raise SystemExit("FAIL: invalid Fourth Count county population counts")
    return out


def collapse_to_ccg(
    counties: pd.DataFrame, race: pd.DataFrame, foreign_born: pd.DataFrame
) -> pd.DataFrame:
    """Map 1970 counties to the project's fixed county-combination groups."""
    crosswalk = pd.read_csv(dc.CCG_CROSSWALK, dtype={"county_fips5": str, "ccg_id": str})
    crosswalk["county_fips5"] = crosswalk["county_fips5"].str.zfill(5)
    merged = counties.merge(race, on="county_fips5", how="left", validate="1:1")
    if merged[["race_population_1970", "black_population_1970"]].isna().any().any():
        raise SystemExit("FAIL: the CCDB and Census race county universes differ")
    merged = merged.merge(foreign_born, on="county_fips5", how="left", validate="1:1")
    if merged[
        ["fourth_count_population_1970", "foreign_born_population_1970"]
    ].isna().any().any():
        raise SystemExit("FAIL: the CCDB and Fourth Count county universes differ")
    if not np.array_equal(
        merged["population_1970"].to_numpy(np.int64),
        merged["fourth_count_population_1970"].to_numpy(np.int64),
    ):
        raise SystemExit("FAIL: CCDB and Fourth Count county populations differ")
    merged = merged.merge(
        crosswalk[["county_fips5", "ccg_id"]],
        on="county_fips5",
        how="left",
        validate="1:1",
    )
    missing = merged.loc[merged["ccg_id"].isna(), "county_fips5"].tolist()
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} 1970 counties lack a CCG mapping: {missing[:20]}")

    sums = [c for c in merged.columns if c.endswith("_1970") or c.endswith("_1969")]
    sums.remove("county_name_1970") if "county_name_1970" in sums else None
    out = merged.groupby("ccg_id", sort=True)[sums].sum(min_count=1).reset_index()
    pop = out["population_1970"]
    emp = out["employed_civilian_1970"]
    labor_force = out["civilian_labor_force_1970"]
    age25 = out["population_age25plus_1970"]
    out["log population"] = np.log(pop)
    out["income per head"] = out["aggregate_money_income_1969"] / pop
    out["employment rate"] = emp / pop
    out["Unemployment rate"] = out["unemployed_civilian_1970"] / labor_force
    out["Female civilian labor-force share"] = (
        out["female_civilian_labor_force_1970"] / labor_force
    )
    out["Manufacturing share"] = out["manufacturing_employment_1970"] / emp
    out["Prof./managerial occupation share"] = (
        out["professional_managerial_employment_1970"] / emp
    )
    out["Non-tradable proxy share"] = out["nontradable_proxy_employment_1970"] / emp
    out["Government employment share"] = out["government_employment_1970"] / emp
    out["Rural share"] = out["rural_population_1970"] / pop
    out["Age 65+ share"] = out["population_age65plus_1970"] / pop
    out["Black share"] = out["black_population_1970"] / out["race_population_1970"]
    out["Foreign-born share"] = (
        out["foreign_born_population_1970"] / out["fourth_count_population_1970"]
    )
    out["High-school completion share, age 25+"] = (
        out["high_school_population_age25plus_1970"] / age25
    )
    out["College share, age 25+"] = out["college_population_age25plus_1970"] / age25
    out["Family poverty rate"] = (
        out["families_below_low_income_1969"] / out["families_1970"]
    )
    out["Homeownership rate"] = (
        out["owner_occupied_housing_units_1970"]
        / out["occupied_housing_units_1970"]
    )

    named = [
        "log population",
        "income per head",
        "employment rate",
        "Unemployment rate",
        "Female civilian labor-force share",
        "Manufacturing share",
        "Prof./managerial occupation share",
        "Non-tradable proxy share",
        "Government employment share",
        "Rural share",
        "Age 65+ share",
        "Black share",
        "Foreign-born share",
        "High-school completion share, age 25+",
        "College share, age 25+",
        "Family poverty rate",
        "Homeownership rate",
    ]
    bad = ~np.isfinite(out[named]).all(axis=1)
    if bad.any():
        raise SystemExit(f"FAIL: {int(bad.sum())} CCGs have incomplete 1970 characteristics")
    for c in named[2:]:
        if not out[c].between(0, 1).all():
            raise SystemExit(f"FAIL: {c} lies outside [0,1]")
    return out[["ccg_id", *named]]


def main() -> None:
    acquire()
    counties = parse_counties()
    race = race_counts()
    foreign_born = foreign_born_counts()
    ccg = collapse_to_ccg(counties, race, foreign_born)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "county_characteristics_1970_ccg.parquet"
    ccg.to_parquet(target, index=False)
    check = pd.read_parquet(target)
    if not check.equals(ccg):
        raise SystemExit("FAIL: persisted 1970 characteristics do not round-trip exactly")
    print(f"parsed counties: {len(counties):,}")
    print(
        "foreign-born population: "
        f"{int(foreign_born['foreign_born_population_1970'].sum()):,}"
    )
    print(f"collapsed CCGs: {len(ccg):,}")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
