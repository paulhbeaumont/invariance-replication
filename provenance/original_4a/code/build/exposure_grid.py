"""Seven exposure constructions on a fixed, explicit annual county panel.

Annual input columns: ccg, year, spending_real, population, employment.
No geography correction, missing-to-zero conversion, trimming or winsorization
occurs here. The annual input must be certified by its own source builder.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


VARIANTS = (
    {"id": "fixed_1969_1974", "start": 1969, "end": 1974},
    *({"id": f"rolling_L{length}_gap{gap}", "length": length, "gap": gap}
      for gap in (1, 5) for length in (4, 5, 10)),
)
EXPOSURE_FORMULA = "mean_annual_per_capita_relative_national"


def sample_digest(frame: pd.DataFrame) -> str:
    keys = frame[["ccg", "year"]].sort_values(["ccg", "year"])
    return hashlib.sha256(keys.to_csv(index=False).encode()).hexdigest()


def window(variant: dict, year: int, horizon: int) -> list[int]:
    if "start" in variant:
        return list(range(variant["start"], variant["end"] + 1))
    end = year - horizon - variant["gap"]
    return list(range(end - variant["length"] + 1, end + 1))


def build_grid(annual: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    """Construct all variants; only absent calendar history removes years.

    Missing/nonfinite values inside supplied calendar coverage are errors, not
    permission to change the county support. Candidate outcome failures also
    fail rather than silently changing the sample.
    """
    required = {"ccg", "year", "spending_real", "population", "employment"}
    if not required.issubset(annual):
        raise ValueError(f"annual panel missing columns: {sorted(required - set(annual))}")
    if annual.empty or annual[["ccg", "year"]].isna().any().any():
        raise ValueError("empty panel or missing identifiers")
    if annual.duplicated(["ccg", "year"]).any():
        raise ValueError("annual panel keys are not unique")
    for col in ("ccg", "year"):
        values = annual[col].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{col} must contain finite integers")
    units = np.sort(annual.ccg.unique())
    years = np.sort(annual.year.unique())
    expected = pd.MultiIndex.from_product([units, years], names=["ccg", "year"])
    indexed = annual.set_index(["ccg", "year"])
    if len(indexed) != len(expected):
        raise ValueError("annual panel is not rectangular; source builder must account for missing keys")
    h = spec["horizon"]
    if type(h) is not int or h < 1:
        raise ValueError("horizon must be a positive integer")
    growth = spec["outcome_growth"]
    if growth not in {"standard", "midpoint"}:
        raise ValueError("outcome_growth requires an explicit standard or midpoint selection")
    if spec["exposure_formula"] != EXPOSURE_FORMULA:
        raise ValueError("unapproved exposure formula")
    if spec["national_shift"] != "window_change_over_baseline_population":
        raise ValueError("unapproved national-shift formula")
    if spec["variants"] != list(VARIANTS):
        raise ValueError("variant list differs from the approved seven-way design")
    if spec["national_support"] != "all_input_counties" or spec["leave_one_out"] is not False:
        raise ValueError("national-support or leave-out specification differs")
    pivot = {c: indexed[c].unstack("year").reindex(index=units, columns=years)
             for c in ("spending_real", "population", "employment")}
    spending, pop, emp = (pivot[c] for c in ("spending_real", "population", "employment"))
    if not np.isfinite(spending.to_numpy()).all():
        raise ValueError("annual spending has missing/nonfinite values; no zero filling allowed")
    if not np.isfinite(pop.to_numpy()).all() or (pop.to_numpy() <= 0).any():
        raise ValueError("annual population must be finite and strictly positive")
    national_d = spending.sum(axis=0)
    national_p = pop.sum(axis=0)
    national_rate = national_d / national_p
    annual_rate = spending / pop
    candidate_years = list(range(spec["first_regression_year"], spec["last_regression_year"] + 1))
    if not candidate_years:
        raise ValueError("empty requested regression period")
    missing = {}
    variants = list(VARIANTS)
    data_years = set(int(y) for y in years)
    parts = []
    for t in candidate_years:
        baseline = t - h
        outcome_and_treatment_years = set(range(baseline, t + 1))
        if not outcome_and_treatment_years.issubset(data_years):
            raise ValueError(f"treatment/outcome history absent for regression year {t}")
        e0, e1 = emp[baseline].to_numpy(), emp[t].to_numpy()
        if not np.isfinite(e0).all() or not np.isfinite(e1).all() or (e0 < 0).any() or (e1 < 0).any():
            raise ValueError(f"missing or negative county employment endpoints in {t}; sample decision required")
        denominator = e0 if growth == "standard" else (e0 + e1) / 2
        if (denominator <= 0).any():
            raise ValueError(f"undefined {growth} growth in {t}; zero-denominator sample decision required")
        y = (e1 - e0) / denominator
        window_years = list(range(baseline + 1, t + 1))
        dbar = spending[window_years].mean(axis=1)
        x = (dbar - spending[baseline]) / pop[baseline]
        shift = (national_d[window_years].mean() - national_d[baseline]) / national_p[baseline]
        block = pd.DataFrame({"ccg": units.astype(np.int32), "year": np.int32(t),
                              "x": x.to_numpy(), "y": y, "national_shift": shift})
        block['employment_base'] = e0
        block['employment_end'] = e1
        block['growth_denominator'] = denominator
        block['zero_mass'] = denominator == 0
        block['analysis_weight_equal'] = 1.0
        for geography in ('state', 'division', 'region'):
            if geography in annual:
                values = indexed[geography].unstack('year').reindex(index=units, columns=years)
                if values.isna().any().any() or values.nunique(axis=1).ne(1).any():
                    raise ValueError(f'{geography} must be complete and constant within county')
                block[geography] = values.iloc[:, 0].to_numpy(dtype=np.int32)
        for variant in variants:
            label = variant["id"]
            w = window(variant, t, h)
            if max(w) >= baseline:
                raise ValueError(f"exposure {label} touches the treatment baseline in {t}")
            absent = sorted(set(w) - data_years)
            if absent:
                block[f"exposure__{label}"] = np.nan
                block[f"z__{label}"] = np.nan
                missing.setdefault(label, {})[str(t)] = absent
            else:
                scale = float(national_rate[w].mean())
                if not np.isfinite(scale) or scale == 0:
                    raise ValueError(f"zero/nonfinite national exposure normalization for {label} in {t}")
                exposure = annual_rate[w].mean(axis=1).to_numpy() / scale
                block[f"exposure__{label}"] = exposure
                block[f"z__{label}"] = exposure * shift
        parts.append(block)
    full = pd.concat(parts, ignore_index=True).sort_values(["ccg", "year"]).reset_index(drop=True)
    for geography in ('division', 'region'):
        if geography in full:
            full[f'{geography}_year'] = full.groupby([geography, 'year'], sort=True).ngroup().astype(np.int32)
    zcols = [f"z__{v['id']}" for v in variants]
    mask = np.isfinite(full[zcols].to_numpy()).all(axis=1)
    full["common_sample"] = mask
    common = full.loc[mask]
    if common.empty or common.ccg.nunique() != len(units):
        raise ValueError("empty common sample or a county lost; no automatic county exclusion")
    if common.groupby("year").size().nunique() != 1:
        raise ValueError("common support is not constant across years")
    report = {"candidate_observations": len(full), "common_observations": len(common),
              "counties": len(units), "common_first_year": int(common.year.min()),
              "common_last_year": int(common.year.max()), "sample_sha256": sample_digest(common),
              "history_unavailable": missing, "variants": {}}
    for v in variants:
        name = v["id"]
        valid = np.isfinite(full[f"z__{name}"])
        report["variants"][name] = {"available_observations": int(valid.sum()),
                                      "available_first_year": int(full.loc[valid, "year"].min()),
                                      "available_last_year": int(full.loc[valid, "year"].max())}
    # JSON serialization is part of the saved report's contract.
    json.dumps(report, allow_nan=False)
    return full, report
