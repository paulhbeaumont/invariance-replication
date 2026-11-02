"""Triage city-county conflicts that appear in both source-audit windows.

The screen combines three inputs: the raw-source comparisons from producer 75,
the time-consistent county-group crosswalk, and fixed-vintage 2020 Census
TIGERweb place boundaries. It also ranks unresolved locations by the combined
county contribution to the current h=2 market-share first-stage cluster meat.

The Census boundary check is a screen, not historical proof for 2001-2009.
No production geography is altered by this producer.

Writes Results/instrument_normalization/
    usaspending_city_county_repeated_candidate_triage.csv
    usaspending_city_county_repeated_candidate_summary.csv
and Instrument_analysis/Cleaning_spendingData/
    REPEATED_CANDIDATE_TRIAGE.md
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/usaspending_city_county_repeated_candidate_triage.csv
# mpgr-new-estimates: 0
# mpgr-variation-scope: contracts|repeated-city-county-conflicts|census-2020-screen|h2-first-stage-rank

import argparse
import importlib.util
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
STEP1_PY = PIPELINE.parent / "py"
STEP2_PY = ROOT / "step_2" / "code" / "py"
sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402
import store  # noqa: E402


H = 2
EXPECTED_REPEATED = 33
PLACE_SERVICE = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
    "Places_CouSub_ConCity_SubMCD/MapServer"
)
COUNTY_SERVICE = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
    "State_County/MapServer"
)
PLACE_LAYERS = {25: "incorporated_place", 26: "census_designated_place"}
COUNTY_LAYER = 55
TIGER_VINTAGE = "2020 Census, January 1 2020 vintage"
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
CANDIDATE_IN = OUTDIR / "usaspending_city_county_override_candidates.csv"
FIRST_STAGE_IN = OUTDIR / "h2_county_exposure_window_market_share_shifter.csv"
TRIAGE_OUT = OUTDIR / "usaspending_city_county_repeated_candidate_triage.csv"
SUMMARY_OUT = OUTDIR / "usaspending_city_county_repeated_candidate_summary.csv"
CACHE_DIR = dc.MACRO_DATA_ROOT / "temp" / "usaspending_city_county_audit"
CENSUS_CACHE = CACHE_DIR / "tigerweb_place_counties_2020.csv"
MEMO_OUT = (dc.DROPBOX_PROJECT_ROOT / "4_Macro_version" / "Instrument_analysis"
            / "Cleaning_spendingData" / "REPEATED_CANDIDATE_TRIAGE.md")
OVERRIDE_PATH = (
    PIPELINE.parent / "reference" / "usaspending_city_county_overrides.csv"
)
CROSSWALK_PATH = PIPELINE.parent / "reference" / "county_ccg_crosswalk.csv"
MILITARY_PATTERN = re.compile(r"(?:^| )(?:FORT|AFB)(?: |$)")


def load_module(name, path):
    """Load a project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_name(value):
    """Normalize a place name to the source-audit convention."""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def arcgis_request(url, parameters, post=False):
    """Read one JSON response from the Census ArcGIS service."""
    encoded = urllib.parse.urlencode(parameters).encode()
    if post:
        request = urllib.request.Request(
            url, data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent": "aggregation-disaggregation-invariance"})
    else:
        request = urllib.request.Request(
            f"{url}?{encoded.decode()}",
            headers={"User-Agent": "aggregation-disaggregation-invariance"})
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(f"Census TIGERweb error: {result['error']}")
    return result


def state_place_index(state):
    """Return 2020 incorporated places and CDPs for one state."""
    frames = []
    for layer, place_type in PLACE_LAYERS.items():
        offset = 0
        rows = []
        while True:
            result = arcgis_request(
                f"{PLACE_SERVICE}/{layer}/query",
                {"where": f"STATE='{state}'", "outFields":
                 "NAME,BASENAME,STATE,PLACE,GEOID,AREALAND",
                 "returnGeometry": "false", "resultOffset": offset,
                 "resultRecordCount": 2000, "orderByFields": "GEOID",
                 "f": "json"})
            features = result.get("features", [])
            rows.extend(item["attributes"] for item in features)
            if len(features) < 2000:
                break
            offset += len(features)
        frame = pd.DataFrame(rows)
        if len(frame):
            frame["place_type"] = place_type
            frame["normalized_basename"] = frame["BASENAME"].map(normalize_name)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[
            "NAME", "BASENAME", "STATE", "PLACE", "GEOID", "AREALAND",
            "place_type", "normalized_basename"])
    return pd.concat(frames, ignore_index=True)


def place_counties(place_geoid, layer):
    """Intersect one fixed-vintage Census place polygon with counties."""
    result = arcgis_request(
        f"{PLACE_SERVICE}/{layer}/query",
        {"where": f"GEOID='{place_geoid}'", "outFields": "GEOID",
         "returnGeometry": "true", "outSR": 4326,
         "geometryPrecision": 5, "f": "json"})
    features = result.get("features", [])
    if len(features) != 1:
        raise RuntimeError(
            f"Census place {place_geoid} has {len(features)} geometries")
    geometry = json.dumps(features[0]["geometry"], separators=(",", ":"))
    counties = arcgis_request(
        f"{COUNTY_SERVICE}/{COUNTY_LAYER}/query",
        {"where": "1=1", "geometry": geometry,
         "geometryType": "esriGeometryPolygon", "inSR": 4326,
         "spatialRel": "esriSpatialRelIntersects",
         "outFields": "STATE,COUNTY,BASENAME,GEOID",
         "returnGeometry": "false", "f": "json"}, post=True)
    return sorted({
        item["attributes"]["GEOID"]
        for item in counties.get("features", [])
    })


def census_screen(candidates, refresh=False):
    """Return fixed-vintage Census place-to-county matches for candidates."""
    keys = candidates[["state", "city"]].drop_duplicates()
    if CENSUS_CACHE.exists() and not refresh:
        cached = pd.read_csv(CENSUS_CACHE, dtype="string", keep_default_na=False)
        cached["census_place_found"] = cached["census_place_found"].eq("True")
        check = keys.merge(cached[["state", "city"]].drop_duplicates(),
                           on=["state", "city"], how="left", indicator=True)
        if check["_merge"].eq("both").all():
            return cached

    indexes = {state: state_place_index(state)
               for state in sorted(keys["state"].unique())}
    rows = []
    for item in keys.itertuples(index=False):
        index = indexes[item.state]
        matches = index.loc[index["normalized_basename"].eq(item.city)]
        place_geoids = []
        place_types = []
        counties = set()
        for match in matches.itertuples(index=False):
            layer = next(key for key, value in PLACE_LAYERS.items()
                         if value == match.place_type)
            place_geoids.append(str(match.GEOID))
            place_types.append(str(match.place_type))
            counties.update(place_counties(str(match.GEOID), layer))
        rows.append({
            "state": item.state,
            "city": item.city,
            "census_place_found": bool(len(matches)),
            "census_place_geoids": ";".join(sorted(place_geoids)),
            "census_place_types": ";".join(sorted(set(place_types))),
            "census_counties_2020": ";".join(sorted(counties)),
            "census_vintage": TIGER_VINTAGE,
        })
    result = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(CENSUS_CACHE, index=False)
    return result


def repeated_candidates():
    """Reshape the 33 conflicts observed in both audit comparisons."""
    frame = pd.read_csv(
        CANDIDATE_IN,
        dtype={"state": "string", "dcads_county": "string",
               "usaspending_county": "string"})
    counts = frame.groupby(["state", "city"])["comparison"].nunique()
    keys = counts[counts.eq(2)].reset_index()[["state", "city"]]
    if len(keys) != EXPECTED_REPEATED:
        raise SystemExit(
            f"FAIL: repeated candidate count is {len(keys)}, expected "
            f"{EXPECTED_REPEATED}")
    selected = frame.merge(keys, on=["state", "city"], validate="many_to_one")
    constant = ["dcads_county", "usaspending_county"]
    for column in constant:
        varying = selected.groupby(["state", "city"])[column].nunique()
        if not varying.eq(1).all():
            raise SystemExit(f"FAIL: {column} differs across audit windows")
    base = selected.drop_duplicates(["state", "city"])[
        ["state", "city", *constant]]
    keep = [
        "candidate_rank", "dcads_dominant_share",
        "usaspending_dominant_share", "dcads_total_dollars_real",
        "usaspending_total_dollars_real", "minimum_source_dollars_real",
    ]
    wide = selected.pivot(index=["state", "city"], columns="comparison",
                          values=keep)
    wide.columns = [f"{value}__{comparison}"
                    for value, comparison in wide.columns]
    return base.merge(wide.reset_index(), on=["state", "city"],
                      validate="one_to_one")


def first_stage_county_importance():
    """Return current h=2 market-share first-stage importance by CCG."""
    p57 = load_module(
        "p57_repeated_triage", STEP1_PY / "57_aggregation_invariance.py")
    p68 = load_module(
        "p68_repeated_triage",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_repeated_triage",
        PIPELINE / "69_h2_market_share_first_stage_diagnostics.py")
    helper = load_module(
        "helper_repeated_triage",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    t28 = load_module(
        "t28_repeated_triage", STEP2_PY / "28_table_d1.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(
        fixef_tol=1e-12, fixef_maxiter=100_000)

    baseline, dropped = p57.build_county(
        "composition", h=H, span=locked.SPAN, rule=locked.RULE,
        floor=locked.FLOOR, denom=locked.DENOM)
    if dropped != 0 or len(baseline) != locked.N_COUNTY[H]:
        raise SystemExit(
            f"FAIL: baseline has {len(baseline):,} rows and {dropped} drops")
    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("P",))
    units2, _, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    years = store.target_years(H)
    zwide, _ = p68.build_instrument(arr, contract_years, years, pivots["P"])
    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy(np.int64)
    ci = baseline["year"].map(colpos).to_numpy(np.int64)
    z = zwide[ri, ci]
    x, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    result = p69.within_first_stage(x, z, weight, fe, nofe, cluster, t28)
    fixture = pd.read_csv(FIRST_STAGE_IN).iloc[0]
    p69.assert_fixture("market_share", result, fixture)

    ccgs = pd.unique(baseline["ccg"])
    signal = np.bincount(
        cluster, weights=result["zt"] * result["xt"], minlength=len(ccgs))
    variance = np.bincount(
        cluster, weights=result["zt"] ** 2, minlength=len(ccgs))
    return pd.DataFrame({
        "ccg": pd.Series(ccgs).astype(str).str.zfill(5),
        "county_cluster_meat_share": result["score_cluster"] ** 2 / result["meat"],
        "county_within_instrument_variance_share": variance / result["zz"],
        "county_signed_first_stage_signal_share": signal / result["zx"],
    })


def classify(row, verified_keys):
    """Assign a screening status without changing production geography."""
    key = (row.state, row.city, row.usaspending_county, row.dcads_county)
    if key in verified_keys:
        return "verified_error"
    if row.same_analysis_ccg:
        return "same_analysis_ccg"
    if MILITARY_PATTERN.search(row.city):
        return "installation_boundary_review"
    counties = set(filter(None, str(row.census_counties_2020).split(";")))
    dcads_in = row.dcads_county in counties
    usa_in = row.usaspending_county in counties
    if dcads_in and usa_in:
        return "census_multicounty_screen"
    if usa_in and not dcads_in:
        return "probable_dcads_misassignment"
    if dcads_in and not usa_in:
        return "probable_usaspending_misassignment"
    if row.census_place_found:
        return "census_place_nonmatching_counties"
    return "no_exact_census_place"


def build_triage(refresh_census=False):
    """Combine raw-source, county-group, Census, and first-stage evidence."""
    candidates = repeated_candidates()
    census = census_screen(candidates, refresh=refresh_census)
    result = candidates.merge(
        census, on=["state", "city"], validate="one_to_one")

    crosswalk = pd.read_csv(
        CROSSWALK_PATH, dtype="string").drop_duplicates("county_fips5")
    lookup_ccg = crosswalk.set_index("county_fips5")["ccg_id"]
    lookup_name = crosswalk.set_index("county_fips5")["ccg_name"]
    for prefix, county_column in (("dcads", "dcads_county"),
                                  ("usaspending", "usaspending_county")):
        result[f"{prefix}_ccg"] = result[county_column].map(lookup_ccg)
        result[f"{prefix}_analysis_group"] = result[county_column].map(
            lookup_name)
    if result[["dcads_ccg", "usaspending_ccg"]].isna().any().any():
        raise SystemExit("FAIL: a disputed county is absent from the CCG map")
    result["same_analysis_ccg"] = result["dcads_ccg"].eq(
        result["usaspending_ccg"])

    importance = first_stage_county_importance().set_index("ccg")
    metrics = [
        "county_cluster_meat_share",
        "county_within_instrument_variance_share",
        "county_signed_first_stage_signal_share",
    ]
    for metric in metrics:
        result[f"dcads_{metric}"] = result["dcads_ccg"].map(importance[metric])
        result[f"usaspending_{metric}"] = result["usaspending_ccg"].map(
            importance[metric])
        result[f"pair_{metric}"] = np.where(
            result["same_analysis_ccg"], result[f"dcads_{metric}"],
            result[f"dcads_{metric}"] + result[f"usaspending_{metric}"])

    overrides = pd.read_csv(OVERRIDE_PATH, dtype="string")
    verified_keys = set(zip(
        overrides["state_code"].map({"TX": "48"}).fillna(""),
        overrides["city_name"].map(normalize_name),
        overrides["reported_county_fips"],
        overrides["corrected_county_fips"],
    ))
    result["screening_status"] = [
        classify(row, verified_keys) for row in result.itertuples(index=False)]
    result["historical_verification_required"] = ~result[
        "screening_status"].isin(["verified_error", "same_analysis_ccg"])
    review = result["historical_verification_required"]
    result["review_priority_rank"] = pd.NA
    order = result.loc[review].sort_values(
        ["pair_county_cluster_meat_share",
         "minimum_source_dollars_real__overlap_2001_2006"],
        ascending=[False, False]).index
    result.loc[order, "review_priority_rank"] = np.arange(1, len(order) + 1)
    result["review_priority_rank"] = result[
        "review_priority_rank"].astype("Int64")
    return result.sort_values(
        ["historical_verification_required", "review_priority_rank",
         "state", "city"], ascending=[False, True, True, True])


def percent(value):
    """Format one share for the audit memo."""
    percentage = 100 * float(value)
    if abs(percentage) < 0.0001:
        return f"{percentage:.3e}%"
    return f"{percentage:.4f}%"


def unique_ccg_summary(triage):
    """Summarize importance without double-counting shared county groups."""
    definitions = {
        "all_repeated_candidates": pd.Series(True, index=triage.index),
        "all_except_verified_error": triage["screening_status"].ne(
            "verified_error"),
        "historical_verification_required": triage[
            "historical_verification_required"],
        "nonroutine_review": triage["screening_status"].isin([
            "installation_boundary_review", "probable_dcads_misassignment",
            "probable_usaspending_misassignment", "no_exact_census_place",
            "census_place_nonmatching_counties"]),
    }
    rows = []
    for sample, selected in definitions.items():
        parts = []
        for prefix in ("dcads", "usaspending"):
            parts.append(triage.loc[selected, [
                f"{prefix}_ccg", f"{prefix}_county_cluster_meat_share",
                f"{prefix}_county_within_instrument_variance_share",
                f"{prefix}_county_signed_first_stage_signal_share",
            ]].rename(columns={
                f"{prefix}_ccg": "ccg",
                f"{prefix}_county_cluster_meat_share": "cluster_meat_share",
                f"{prefix}_county_within_instrument_variance_share":
                    "within_instrument_variance_share",
                f"{prefix}_county_signed_first_stage_signal_share":
                    "signed_first_stage_signal_share",
            }))
        counties = pd.concat(parts, ignore_index=True).drop_duplicates("ccg")
        rows.append({
            "sample": sample,
            "n_locations": int(selected.sum()),
            "n_unique_ccgs": len(counties),
            "union_cluster_meat_share": counties["cluster_meat_share"].sum(),
            "union_within_instrument_variance_share":
                counties["within_instrument_variance_share"].sum(),
            "union_signed_first_stage_signal_share":
                counties["signed_first_stage_signal_share"].sum(),
        })
    return pd.DataFrame(rows)


def write_memo(triage, summary):
    """Write the screening results and the decision boundary."""
    counts = triage["screening_status"].value_counts().sort_index()
    review = triage.loc[triage["historical_verification_required"]].sort_values(
        "review_priority_rank")
    lines = [
        "# Repeated city-county candidate triage",
        "",
        "## Scope",
        "",
        (f"The source audit identifies {len(triage)} state-city locations in "
         "both comparison windows. Table 1 screens these locations using the "
         "time-consistent county groups, fixed-vintage 2020 Census place "
         "boundaries, and the current h=2 market-share first-stage cluster "
         "scores."),
        "",
        ("The Census screen uses the official TIGERweb 2020 Census layers for "
         "incorporated places, census designated places, and counties. The "
         "boundary vintage is January 1, 2020. It is a triage device rather "
         "than proof of county boundaries during 2001 to 2009."),
        "",
        "## Screening results",
        "",
        "**Table 1. Status counts among repeated candidates**",
        "",
        "| Screening status | Locations |",
        "|---|---:|",
    ]
    for status, count in counts.items():
        lines.append(f"| `{status}` | {int(count)} |")
    lines += [
        "",
        (f"Exactly {int(triage['same_analysis_ccg'].sum())} locations map to "
         "the same time-consistent county group under both reported county "
         "codes (Table 1). Exactly "
         f"{int(triage['historical_verification_required'].sum())} locations "
         "require historical review after the Sealy decision and the "
         "same-group cases are separated (Table 1)."),
        "",
        "**Table 2. Joint first-stage importance using unique county groups**",
        "",
        ("| Candidate set | Locations | Unique county groups | Cluster meat "
         "share | Within-instrument variance share |"),
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.sample}` | {row.n_locations} | {row.n_unique_ccgs} | "
            f"{percent(row.union_cluster_meat_share)} | "
            f"{percent(row.union_within_instrument_variance_share)} |")
    remaining = summary.loc[
        summary["sample"].eq("all_except_verified_error")].iloc[0]
    nonroutine = summary.loc[summary["sample"].eq("nonroutine_review")].iloc[0]
    lines += [
        "",
        (f"The {int(remaining['n_locations'])} repeated candidates other than "
         f"Sealy touch {int(remaining['n_unique_ccgs'])} unique county groups "
         f"and jointly account for {percent(remaining['union_cluster_meat_share'])} "
         "of the current first-stage cluster meat and "
         f"{percent(remaining['union_within_instrument_variance_share'])} of "
         "within-instrument variance (Table 2). The "
         f"{int(nonroutine['n_locations'])} nonroutine cases account for "
         f"{percent(nonroutine['union_cluster_meat_share'])} of cluster meat "
         "and "
         f"{percent(nonroutine['union_within_instrument_variance_share'])} of "
         "within-instrument variance (Table 2)."),
        "",
        "## Priority cases",
        "",
        ("Table 3 orders the historically unresolved locations by the sum of "
         "the disputed counties' shares of the current first-stage cluster "
         "meat. This is a priority measure, not the counterfactual effect of a "
         "county correction."),
        "",
        "**Table 3. First 15 locations for historical review**",
        "",
        ("| Rank | Location | DCADS county | USAspending county | 2020 Census "
         "counties | Status | Pair meat share |"),
        "|---:|---|---:|---:|---|---|---:|",
    ]
    for row in review.head(15).itertuples(index=False):
        location = f"{row.city.title()}, state FIPS {row.state}"
        census_counties = row.census_counties_2020 or "none"
        lines.append(
            f"| {row.review_priority_rank} | {location} | "
            f"{row.dcads_county} | {row.usaspending_county} | "
            f"{census_counties} | `{row.screening_status}` | "
            f"{percent(row.pair_county_cluster_meat_share)} |")
    lines += [
        "",
        "## Decision rule",
        "",
        ("A `census_multicounty_screen` result means both disputed counties "
         "intersect the named 2020 Census place. It supports leaving the raw "
         "records unchanged unless the 2001 to 2009 evidence identifies a "
         "source-specific coding break."),
        "",
        ("A `probable_dcads_misassignment` or "
         "`probable_usaspending_misassignment` result identifies which raw "
         "county lies outside the 2020 place boundary. It remains provisional "
         "until the historical boundary and the transactions driving the "
         "conflict are checked."),
        "",
        ("Military installation names remain `installation_boundary_review` "
         "because Census place boundaries need not equal the installation "
         "boundary. Locations without an exact Census place match require "
         "manual place-name resolution."),
        "",
        "No additional production override is authorized by this screen.",
        "",
        "## Files",
        "",
        ("- Full triage table: "
         "`Results/instrument_normalization/"
         "usaspending_city_county_repeated_candidate_triage.csv`."),
        ("- Joint-importance summary: "
         "`Results/instrument_normalization/"
         "usaspending_city_county_repeated_candidate_summary.csv`."),
        ("- Raw candidate table: "
         "`Results/instrument_normalization/"
         "usaspending_city_county_override_candidates.csv`."),
        ("- Census screen cache: "
         "`_data/temp/usaspending_city_county_audit/"
         "tigerweb_place_counties_2020.csv`."),
        ("- Census service metadata: "
         f"`{PLACE_SERVICE}/25`, `{PLACE_SERVICE}/26`, and "
         f"`{COUNTY_SERVICE}/{COUNTY_LAYER}`."),
        "",
    ]
    MEMO_OUT.parent.mkdir(parents=True, exist_ok=True)
    MEMO_OUT.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run the repeated-candidate triage and write its audit artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-census", action="store_true")
    args = parser.parse_args()
    start = time.time()
    print("== repeated USAspending city-county candidate triage")
    triage = build_triage(refresh_census=args.refresh_census)
    summary = unique_ccg_summary(triage)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    triage.to_csv(TRIAGE_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    write_memo(triage, summary)
    print(f"   repeated candidates: {len(triage)}")
    print(f"   same analysis CCG: {int(triage['same_analysis_ccg'].sum())}")
    print("   statuses:")
    for status, count in triage["screening_status"].value_counts().items():
        print(f"      {status}: {count}")
    print(f"   wrote {TRIAGE_OUT}")
    print(f"   wrote {SUMMARY_OUT}")
    print(f"   wrote {MEMO_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
