"""Audit category 2320 and decompose its h=2 first-stage influence.

The audit traces positive category-2320 obligations by place of performance,
recipient county, award, year, and county. The first-stage block independently
includes or excludes category 2320 from realized county procurement and from
the market-share instrument on the unchanged 1980-2016 county sample.

Writes Results/instrument_normalization/
    h2_market_share_2320_four_cell_first_stage.csv
    h2_market_share_2320_location_year.csv
    h2_market_share_2320_austin_transactions.csv
    h2_market_share_2320_sealy_transactions.csv
    h2_market_share_2320_sealy_recipients.csv
    h2_market_share_2320_austin_contractors.csv
    h2_market_share_2320_top_county_changes.csv
"""

# mpgr-run-class: cache_augment
# mpgr-cache: _data/temp/pipeline/contracts_category_ccg_year.parquet
# mpgr-output-cache: Results/instrument_normalization/h2_market_share_2320_four_cell_first_stage.csv
# mpgr-new-estimates: 4
# mpgr-variation-scope: h2|market-share|category-2320-four-cell|location-audit

import csv
import importlib.util
import io
import sys
import time
import zipfile
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
FOCUS = "2320"
TARGET_CCG = 48015
SCORE_YEARS = (2007, 2008)
OUTDIR = dc.MACRO_RESULTS_ROOT / "instrument_normalization"
FIRST_STAGE_OUT = OUTDIR / "h2_market_share_2320_four_cell_first_stage.csv"
LOCATION_YEAR_OUT = OUTDIR / "h2_market_share_2320_location_year.csv"
TRANSACTION_OUT = OUTDIR / "h2_market_share_2320_austin_transactions.csv"
SEALY_TRANSACTION_OUT = OUTDIR / "h2_market_share_2320_sealy_transactions.csv"
RECIPIENT_OUT = OUTDIR / "h2_market_share_2320_sealy_recipients.csv"
CONTRACTOR_OUT = OUTDIR / "h2_market_share_2320_austin_contractors.csv"
TOP_COUNTY_OUT = OUTDIR / "h2_market_share_2320_top_county_changes.csv"
TOL = 1e-10


def load_module(name, path):
    """Load one project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_fips(series):
    """Return five-character digit-only county codes."""
    text = series.fillna("").astype(str).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True).str.replace(
        r"\D", "", regex=True)
    return text.str.zfill(5).where(text.ne(""), "")


def processed_transactions():
    """Read positive category-2320 award actions from parsed USAspending."""
    import duckdb

    paths = [
        str(dc.USASPENDING_OUT_DIR / f"fy{year}.parquet")
        for year in range(2001, 2019)
    ]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise SystemExit(
            f"FAIL: {len(missing)} parsed USAspending files are missing")
    connection = duckdb.connect()
    frame = connection.execute("""
        SELECT
            CAST(source_fiscal_year AS INTEGER) AS fiscal_year,
            source_member,
            source_row_number,
            contract_transaction_unique_key,
            contract_award_unique_key,
            award_id_piid,
            modification_number,
            federal_action_obligation,
            action_date,
            period_of_performance_start_date,
            period_of_performance_current_end_date,
            primary_place_of_performance_city_name,
            primary_place_of_performance_county_name,
            prime_award_transaction_place_of_performance_county_fips_code,
            primary_place_of_performance_state_code,
            prime_award_transaction_recipient_county_fips_code,
            recipient_state_code,
            action_type_code,
            action_type,
            transaction_description,
            ccg
        FROM read_parquet(?, union_by_name=true)
        WHERE award_or_idv_flag = 'AWARD'
          AND geo_class = 'county'
          AND federal_action_obligation > 0
          AND SUBSTR(UPPER(TRIM(CAST(product_or_service_code AS VARCHAR))),
                     1, 4) = ?
    """, [paths, FOCUS]).df()
    connection.close()
    if frame.empty:
        raise SystemExit("FAIL: no positive USAspending category-2320 actions")
    frame["place_county_fips"] = normalize_fips(
        frame["prime_award_transaction_place_of_performance_county_fips_code"])
    frame["recipient_county_fips"] = normalize_fips(
        frame["prime_award_transaction_recipient_county_fips_code"])
    return frame


def dcads_contractors():
    """Read Austin category-2320 contractor and performance locations."""
    import duckdb

    paths = [str(path) for path in sorted(dc.OUT_DIR.glob("fy*.parquet"))]
    if not paths:
        raise SystemExit("FAIL: no DCADS fiscal-year files are available")
    connection = duckdb.connect()
    frame = connection.execute("""
        SELECT
            fiscal_year,
            action_date,
            completion_date,
            contract_no,
            contractor_name,
            contractor_street,
            contractor_city,
            contractor_state,
            contractor_zip,
            city_name AS place_city_name,
            county_name AS place_county_name,
            state_name AS place_state_name,
            fsc_description,
            CASE WHEN dollars_unit = 'thousands'
                 THEN dollars * 1000.0 ELSE dollars END AS positive_nominal_dollars
        FROM read_parquet(?, union_by_name=true)
        WHERE CAST(county_fips5 AS VARCHAR) = ?
          AND SUBSTR(UPPER(TRIM(CAST(fsc AS VARCHAR))), 1, 4) = ?
          AND CASE WHEN dollars_unit = 'thousands'
                   THEN dollars * 1000.0 ELSE dollars END > 0
        ORDER BY fiscal_year, positive_nominal_dollars DESC
    """, [paths, str(TARGET_CCG), FOCUS]).df()
    connection.close()
    if frame.empty:
        raise SystemExit("FAIL: DCADS has no Austin category-2320 contracts")
    return frame


def raw_recipient_names(years):
    """Extract recipient identities for audited Sealy transactions."""
    import pyarrow.csv as pacsv

    columns = (
        "contract_transaction_unique_key",
        "contract_award_unique_key",
        "award_id_piid",
        "modification_number",
        "federal_action_obligation",
        "action_date",
        "recipient_name",
        "recipient_parent_name",
        "recipient_city_name",
        "recipient_county_name",
        "prime_award_transaction_recipient_county_fips_code",
        "recipient_state_code",
        "primary_place_of_performance_city_name",
        "primary_place_of_performance_county_name",
        "prime_award_transaction_place_of_performance_county_fips_code",
        "primary_place_of_performance_state_code",
        "award_or_idv_flag",
        "product_or_service_code",
        "transaction_description",
    )
    parts = []
    for year in years:
        matches = list(dc.USASPENDING_SOURCE_DIR.glob(f"FY{year}_*.zip"))
        if len(matches) != 1:
            raise SystemExit(
                f"FAIL: FY{year} has {len(matches)} USAspending archives")
        with zipfile.ZipFile(matches[0]) as archive:
            members = [
                member for member in archive.infolist()
                if not member.is_dir() and member.filename.endswith(".csv")
            ]
            if not members:
                raise SystemExit(f"FAIL: FY{year} archive has no CSV members")
            for member in members:
                with archive.open(member) as source:
                    reader = pacsv.open_csv(
                        source,
                        read_options=pacsv.ReadOptions(block_size=32 << 20),
                        convert_options=pacsv.ConvertOptions(
                            include_columns=list(columns),
                            strings_can_be_null=False),
                    )
                    row_offset = 0
                    for batch in reader:
                        frame = batch.to_pandas()
                        focus = (
                            frame["award_or_idv_flag"].eq("AWARD")
                            & (pd.to_numeric(
                                frame["federal_action_obligation"],
                                errors="coerce") > 0)
                            & frame["product_or_service_code"].astype(
                                str).str.strip().str.upper().str[:4].eq(FOCUS)
                            & frame[
                                "primary_place_of_performance_city_name"
                            ].astype(str).str.strip().str.upper().eq("SEALY")
                        )
                        if focus.any():
                            selected = frame.loc[focus].copy()
                            selected.insert(0, "fiscal_year", year)
                            selected.insert(1, "source_member", member.filename)
                            selected.insert(
                                2, "source_batch_row",
                                row_offset + np.flatnonzero(focus.to_numpy()))
                            parts.append(selected)
                        row_offset += len(frame)
    if not parts:
        raise SystemExit("FAIL: raw archives have no audited Sealy actions")
    return pd.concat(parts, ignore_index=True)


def focus_components(arr, focus, contract_years, years, population):
    """Construct category-2320 treatment and instrument components."""
    national = arr[:, focus, :].sum(axis=0, dtype=np.float64)
    z = np.full((arr.shape[0], len(years)), np.nan, np.float64)
    x = np.full_like(z, np.nan)
    for j, year in enumerate(years):
        exposure_hi = year - H - locked.LAG
        exposure_lo = exposure_hi - locked.SPAN + 1
        if exposure_lo < locked.DATA_FIRST:
            continue
        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure = arr[:, focus, a:b + 1].mean(axis=1).astype(np.float64)
        national_exposure = float(exposure.sum())
        if not np.isfinite(national_exposure) or national_exposure <= 0:
            raise SystemExit(
                f"FAIL: category 2320 has invalid exposure in {year}")
        current = arr[:, focus, contract_years.index(year)].astype(np.float64)
        prior = arr[:, focus, contract_years.index(year - 1)].astype(np.float64)
        base = arr[:, focus, contract_years.index(year - H)].astype(np.float64)
        national_shift = float(
            0.5 * (national[contract_years.index(year)]
                   + national[contract_years.index(year - 1)])
            - national[contract_years.index(year - H)])
        pbase = population[year - H].to_numpy(np.float64)
        z[:, j] = exposure / national_exposure * national_shift / pbase
        x[:, j] = (0.5 * (current + prior) - base) / pbase
    return x, z


def four_cell_first_stage(
        baseline, full_z, x_focus, z_focus, helper, p69, t28):
    """Estimate the four combinations of full and 2320-excluded objects."""
    x_full = baseline["x"].to_numpy(np.float64)
    variants = (
        ("full treatment", "full instrument", x_full, full_z),
        ("full treatment", "instrument excluding 2320", x_full,
         full_z - z_focus),
        ("treatment excluding 2320", "full instrument", x_full - x_focus,
         full_z),
        ("treatment excluding 2320", "instrument excluding 2320",
         x_full - x_focus, full_z - z_focus),
    )
    _, weight, fe, nofe, cluster = helper.estimation_arrays(baseline)
    target = baseline["ccg"].eq(TARGET_CCG).to_numpy()
    target_code = np.unique(cluster[target])
    if len(target_code) != 1:
        raise SystemExit("FAIL: Austin County maps to multiple clusters")
    rows = []
    for treatment_label, instrument_label, x, z in variants:
        if not np.isfinite(x).all() or not np.isfinite(z).all():
            raise SystemExit(
                f"FAIL: nonfinite values for {treatment_label}, "
                f"{instrument_label}")
        result = p69.within_first_stage(
            x, z, weight, fe, nofe, cluster, t28)
        target_score = float(result["score_cluster"][int(target_code[0])])
        rows.append({
            "treatment": treatment_label,
            "instrument": instrument_label,
            "category_2320_in_treatment": treatment_label == "full treatment",
            "category_2320_in_instrument": instrument_label == "full instrument",
            "sample_start": int(baseline["year"].min()),
            "sample_end": int(baseline["year"].max()),
            "n": len(baseline),
            "n_counties": int(baseline["ccg"].nunique()),
            "fixed_effects": "division-by-year",
            "weight": "equal county",
            "cluster": "county",
            "first_stage": result["beta"],
            "first_stage_se": result["se"],
            "first_stage_t": result["t"],
            "first_stage_F": result["F"],
            "partial_r2": result["partial_r2"],
            "austin_share_of_within_instrument_variance": float(
                result["zt"][target] @ result["zt"][target] / result["zz"]),
            "austin_share_of_cluster_meat":
                target_score ** 2 / result["meat"],
        })
    return pd.DataFrame(rows)


def location_outputs(arr, focus, units, contract_years, processed):
    """Summarize category-2320 geography, timing, and county reallocations."""
    values = arr[:, focus, :].astype(np.float64)
    labels = pd.read_csv(
        STEP1_PY.parent / "reference" / "county_ccg_crosswalk.csv")
    labels["ccg_id"] = pd.to_numeric(labels["ccg_id"], errors="coerce")
    labels = labels.dropna(subset=["ccg_id"])
    labels["ccg_id"] = labels["ccg_id"].astype(np.int64)
    name_map = labels.drop_duplicates("ccg_id").set_index("ccg_id")["ccg_name"]
    target = int(np.flatnonzero(units == TARGET_CCG)[0])
    rows = []
    for year in range(2001, 2019):
        yi = contract_years.index(year)
        level = values[:, yi]
        national = float(level.sum())
        austin = float(level[target])
        rows.append({
            "year": year,
            "national_positive_real_dollars": national,
            "austin_positive_real_dollars": austin,
            "austin_national_share": austin / national if national > 0 else np.nan,
            "austin_absolute_rank": int(np.sum(level > austin) + 1),
            "largest_ccg": int(units[int(np.argmax(level))]),
            "largest_county": str(name_map.get(units[int(np.argmax(level))], "")),
            "largest_county_positive_real_dollars": float(level.max()),
        })
    annual = pd.DataFrame(rows)

    processed = processed.copy()
    processed["is_austin_place"] = processed["ccg"].eq(TARGET_CCG)
    processed["is_austin_recipient"] = processed[
        "recipient_county_fips"].eq(str(TARGET_CCG))
    grouped = processed.groupby("fiscal_year", sort=False)
    raw_year = grouped.agg(
        positive_nominal_dollars=("federal_action_obligation", "sum"),
        positive_actions=("federal_action_obligation", "size"),
        unique_awards=("contract_award_unique_key", "nunique"),
    ).reset_index()
    austin = processed.loc[processed["is_austin_place"]]
    austin_grouped = austin.groupby("fiscal_year", sort=False)
    austin_year = austin_grouped.agg(
        austin_positive_nominal_dollars=("federal_action_obligation", "sum"),
        austin_positive_actions=("federal_action_obligation", "size"),
        austin_unique_awards=("contract_award_unique_key", "nunique"),
        austin_unique_place_cities=("primary_place_of_performance_city_name",
                                    "nunique"),
    ).reset_index()
    same = (austin.assign(
        same_recipient_dollars=np.where(
            austin["is_austin_recipient"],
            austin["federal_action_obligation"], 0.0))
        .groupby("fiscal_year", sort=False)["same_recipient_dollars"].sum()
        .rename("austin_recipient_same_county_positive_nominal_dollars")
        .reset_index())
    annual = (annual.merge(raw_year, left_on="year", right_on="fiscal_year",
                           how="left")
              .drop(columns="fiscal_year")
              .merge(austin_year, left_on="year", right_on="fiscal_year",
                     how="left")
              .drop(columns="fiscal_year")
              .merge(same, left_on="year", right_on="fiscal_year", how="left")
              .drop(columns="fiscal_year"))
    zero_columns = [
        "austin_positive_nominal_dollars",
        "austin_positive_actions",
        "austin_unique_awards",
        "austin_unique_place_cities",
        "austin_recipient_same_county_positive_nominal_dollars",
    ]
    annual[zero_columns] = annual[zero_columns].fillna(0.0)
    annual["austin_place_share_from_nominal_actions"] = (
        annual["austin_positive_nominal_dollars"]
        / annual["positive_nominal_dollars"])
    annual["austin_recipient_same_county_dollar_share"] = (
        annual["austin_recipient_same_county_positive_nominal_dollars"]
        / annual["austin_positive_nominal_dollars"])
    check = annual.loc[
        annual["year"].ge(locked.SEAM_YEAR)
        & annual["austin_place_share_from_nominal_actions"].notna()]
    if not np.allclose(
            check["austin_national_share"],
            check["austin_place_share_from_nominal_actions"],
            rtol=1e-5, atol=1e-6):
        gap = float(np.max(np.abs(
            check["austin_national_share"]
            - check["austin_place_share_from_nominal_actions"])))
        raise SystemExit(
            f"FAIL: processed and tensor Austin shares differ by {gap:.3e}")

    top_rows = []
    for year in range(2005, 2013):
        current = values[:, contract_years.index(year)]
        prior = values[:, contract_years.index(year - 1)]
        base = values[:, contract_years.index(year - H)]
        change = 0.5 * (current + prior) - base
        order = np.argsort(-np.abs(change))[:20]
        for rank, position in enumerate(order, 1):
            top_rows.append({
                "year": year,
                "absolute_change_rank": rank,
                "ccg": int(units[position]),
                "county": str(name_map.get(units[position], "")),
                "category_2320_two_year_change_real_dollars":
                    float(change[position]),
                "category_2320_level_t_minus_2_real_dollars":
                    float(base[position]),
                "category_2320_mean_t_minus_1_t_real_dollars":
                    float(0.5 * (current[position] + prior[position])),
            })
    return annual, pd.DataFrame(top_rows), austin


def main():
    """Run the category-2320 audit and four-cell first-stage diagnostic."""
    start = time.time()
    print("== h=2 market-share category 2320 audit")
    p57 = load_module(
        "p57_2320_audit", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module(
        "t28_2320_audit", STEP2_PY / "28_table_d1.py")
    helper = load_module(
        "helper_2320_audit",
        PIPELINE / "57_h2_shifter_clipping_diagnostics.py")
    p68 = load_module(
        "p68_2320_audit",
        PIPELINE / "68_h2_exposure_window_market_share_shifter.py")
    p69 = load_module(
        "p69_2320_audit",
        PIPELINE / "69_h2_market_share_first_stage_diagnostics.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(
        fixef_tol=1e-12, fixef_maxiter=100_000)

    baseline, dropped = p57.build_county(
        "composition", h=H, span=locked.SPAN, rule=locked.RULE,
        floor=locked.FLOOR, denom=locked.DENOM)
    if dropped != 0 or len(baseline) != locked.N_COUNTY[H]:
        raise SystemExit(
            f"FAIL: baseline has {len(baseline):,} rows and {dropped} drops")
    county_panel = store.load_county_panel()
    units, pivots = store.load_pivots(county_panel, cols=("P",))
    units2, categories, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    if FOCUS not in categories:
        raise SystemExit("FAIL: category 2320 is absent from the tensor")
    focus = categories.index(FOCUS)
    years = store.target_years(H)
    full_wide, _ = p68.build_instrument(
        arr, contract_years, years, pivots["P"])
    x_focus_wide, z_focus_wide = focus_components(
        arr, focus, contract_years, years, pivots["P"])
    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy(np.int64)
    ci = baseline["year"].map(colpos).to_numpy(np.int64)
    full_z = full_wide[ri, ci]
    x_focus = x_focus_wide[ri, ci]
    z_focus = z_focus_wide[ri, ci]

    full_fit = p69.within_first_stage(
        baseline["x"].to_numpy(np.float64), full_z,
        *helper.estimation_arrays(baseline)[1:], t28)
    p69.assert_fixture(
        "market_share", full_fit, p69.saved_results()["market_share"])
    first_stage = four_cell_first_stage(
        baseline, full_z, x_focus, z_focus, helper, p69, t28)

    processed = processed_transactions()
    annual, top_counties, austin = location_outputs(
        arr, focus, units, contract_years, processed)
    sealy = processed.loc[
        processed["primary_place_of_performance_city_name"].astype(
            str).str.strip().str.upper().eq("SEALY")].copy()
    raw_recipients = raw_recipient_names(SCORE_YEARS)
    raw_recipients["federal_action_obligation"] = pd.to_numeric(
        raw_recipients["federal_action_obligation"], errors="raise")
    raw_counts = raw_recipients.groupby("fiscal_year").agg(
        actions=("federal_action_obligation", "size"),
        dollars=("federal_action_obligation", "sum"))
    processed_counts = (sealy.loc[sealy["fiscal_year"].isin(SCORE_YEARS)]
                        .groupby("fiscal_year")
                        .agg(actions=("federal_action_obligation", "size"),
                             dollars=("federal_action_obligation", "sum")))
    if not raw_counts.index.equals(processed_counts.index):
        raise SystemExit("FAIL: raw and processed audit years differ")
    if not np.array_equal(raw_counts["actions"], processed_counts["actions"]):
        raise SystemExit("FAIL: raw and processed Sealy action counts differ")
    if not np.allclose(
            raw_counts["dollars"], processed_counts["dollars"],
            rtol=1e-12, atol=0.01):
        raise SystemExit("FAIL: raw and processed Sealy dollars differ")

    recipient_summary = (raw_recipients.groupby([
        "fiscal_year", "recipient_name", "recipient_parent_name",
        "recipient_city_name", "recipient_county_name",
        "prime_award_transaction_recipient_county_fips_code",
        "recipient_state_code",
    ], dropna=False, sort=False)
        .agg(positive_actions=("federal_action_obligation", "size"),
             positive_nominal_dollars=("federal_action_obligation", "sum"),
             unique_awards=("contract_award_unique_key", "nunique"))
        .reset_index()
        .sort_values(["fiscal_year", "positive_nominal_dollars"],
                     ascending=[True, False]))
    contractors = dcads_contractors()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    first_stage.to_csv(FIRST_STAGE_OUT, index=False)
    annual.to_csv(LOCATION_YEAR_OUT, index=False)
    austin.to_csv(TRANSACTION_OUT, index=False)
    sealy.to_csv(SEALY_TRANSACTION_OUT, index=False)
    recipient_summary.to_csv(RECIPIENT_OUT, index=False)
    contractors.to_csv(CONTRACTOR_OUT, index=False)
    top_counties.to_csv(TOP_COUNTY_OUT, index=False)

    for _, row in first_stage.iterrows():
        print(
            f"   {row['treatment']}; {row['instrument']}: "
            f"pi={row['first_stage']:+.6f}, "
            f"SE={row['first_stage_se']:.6f}, "
            f"F={row['first_stage_F']:.3f}, "
            f"Austin meat={100 * row['austin_share_of_cluster_meat']:.2f}%")
    for year in SCORE_YEARS:
        row = annual.loc[annual["year"].eq(year)].iloc[0]
        recipient_share = row["austin_recipient_same_county_dollar_share"]
        recipient_text = (
            f"{100 * recipient_share:.2f}%"
            if np.isfinite(recipient_share) else "not defined (zero actions)")
        print(
            f"   FY{year} category 2320: Austin place share="
            f"{100 * row['austin_national_share']:.2f}%, "
            f"positive actions={int(row['austin_positive_actions'])}, "
            f"same-county recipient share={recipient_text}")
        sealy_year = sealy.loc[sealy["fiscal_year"].eq(year)]
        place = (sealy_year.groupby(
            ["place_county_fips", "primary_place_of_performance_county_name"],
            dropna=False)["federal_action_obligation"].sum()
            .sort_values(ascending=False))
        print(
            f"      Sealy positive actions: "
            f"{len(sealy_year)}, place county allocation={place.to_dict()}")
    print(f"   wrote {FIRST_STAGE_OUT}")
    print(f"   wrote {LOCATION_YEAR_OUT}")
    print(f"   wrote {TRANSACTION_OUT}")
    print(f"   wrote {SEALY_TRANSACTION_OUT}")
    print(f"   wrote {RECIPIENT_OUT}")
    print(f"   wrote {CONTRACTOR_OUT}")
    print(f"   wrote {TOP_COUNTY_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
