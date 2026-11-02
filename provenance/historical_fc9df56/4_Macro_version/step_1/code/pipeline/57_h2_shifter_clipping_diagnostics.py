"""County h=2 first-stage diagnostics for rolling-five-year shifter clipping.

The national category shifter uses the matched two-year numerator and the
five-year pre-shock denominator. The script holds the county sample, exposure,
weights, fixed effects, and county clustering fixed while varying only the
treatment of the shifter tails.

Writes Results/instrument_normalization/h2_county_clipping_sensitivity.csv and
Results/instrument_normalization/h2_county_clipping_decomposition.csv.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parents[2]
STEP1_PY = PIPELINE.parent / "py"
STEP2_PY = ROOT / "step_2" / "code" / "py"
SEQUENCE_OUT = (ROOT / "Results" / "instrument_normalization"
                / "h2_county_first_stage_sequence.csv")
SENSITIVITY_OUT = (ROOT / "Results" / "instrument_normalization"
                   / "h2_county_clipping_sensitivity.csv")
DECOMPOSITION_OUT = (ROOT / "Results" / "instrument_normalization"
                     / "h2_county_clipping_decomposition.csv")

sys.path.insert(0, str(STEP1_PY))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import store  # noqa: E402


H = 2
DENOM_YEARS = 5
CAPS = (0.5, 1.0, 1.5, 2.0, 5.0)


def load_module(name, path):
    """Load a project producer without changing its command-line interface."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_shifter(arr, contract_years, target_years):
    """Build the matched rolling-five-year category shifter."""
    national = arr.sum(axis=0, dtype=np.float64)
    raw = np.full((national.shape[0], len(target_years)), np.nan, np.float64)
    live_counts = []
    tail_counts = []
    for j, t in enumerate(target_years):
        end = national[:, contract_years.index(t)]
        prior = national[:, contract_years.index(t - 1)]
        base = national[:, contract_years.index(t - H)]
        window = 0.5 * (end + prior)
        denominator = np.column_stack([
            national[:, contract_years.index(t - H - ell)]
            for ell in range(DENOM_YEARS)
        ]).mean(axis=1)
        live = np.isfinite(denominator) & (denominator > 0)
        raw[live, j] = (window[live] - base[live]) / denominator[live]
        live_counts.append(int(live.sum()))
        tail_counts.append(int((live & (np.abs(raw[:, j]) > 1.0)).sum()))
    return raw, live_counts, tail_counts


def shifter_variants(raw):
    """Return cap-grid, exclusion, and additive component shifters."""
    live = np.isfinite(raw)
    filled = np.where(live, raw, 0.0)
    cap1 = np.clip(filled, -1.0, 1.0)
    tail = live & (np.abs(raw) > 1.0)
    inside = np.where(live & ~tail, raw, 0.0)
    tail_cap = np.where(tail, np.sign(raw), 0.0)
    excess = filled - cap1

    variants = {
        f"cap_{cap:g}": np.clip(filled, -cap, cap) for cap in CAPS
    }
    variants["exclude_tail"] = inside
    variants["unclipped"] = filled
    components = {
        "inside": inside,
        "tail_cap": tail_cap,
        "clip_1": cap1,
        "excess": excess,
        "unclipped": filled,
    }
    if not np.allclose(components["clip_1"], inside + tail_cap,
                       rtol=0.0, atol=1e-14):
        raise SystemExit("FAIL: clip-1 component identity does not hold")
    if not np.allclose(components["unclipped"], cap1 + excess,
                       rtol=0.0, atol=1e-14):
        raise SystemExit("FAIL: unclipped component identity does not hold")
    return variants, components


def county_instruments(shifters):
    """Aggregate category shifters with the locked county exposures."""
    panel = store.load_county_panel()
    units, pivots = store.load_pivots(panel, cols=("P",))
    units2, _, contract_years, arr, _ = store.load_tensor(locked.RULE)
    if not np.array_equal(units, units2):
        raise SystemExit("FAIL: county order differs between persisted stores")
    years = store.target_years(H)
    result = {
        name: np.full((len(units), len(years)), np.nan, np.float64)
        for name in shifters
    }
    names = list(shifters)
    for j, t in enumerate(years):
        exposure_hi = t - H - locked.LAG
        exposure_lo = exposure_hi - locked.SPAN + 1
        if exposure_lo < locked.DATA_FIRST:
            continue
        a = contract_years.index(exposure_lo)
        b = contract_years.index(exposure_hi)
        exposure = arr[:, :, a:b + 1].mean(axis=2).astype(np.float64)
        pbase = pivots["P"][t - H].to_numpy(np.float64)
        matrix = np.column_stack([shifters[name][:, j] for name in names])
        values = exposure @ matrix / pbase[:, None]
        for col, name in enumerate(names):
            result[name][:, j] = values[:, col]
    return units, years, result, arr, contract_years


def estimation_arrays(baseline):
    """Build fixed-effect and county-cluster arrays for the locked sample."""
    x = baseline["x"].to_numpy(np.float64)
    weight = np.ones(len(baseline), np.float64)
    fe = pd.factorize(baseline["division"].astype(str) + "_"
                      + baseline["year"].astype(str))[0].astype(np.int64)
    nofe = np.zeros(len(baseline), np.int64)
    cluster = pd.factorize(baseline["ccg"])[0].astype(np.int64)
    return x, weight, fe, nofe, cluster


def joint_first_stage(x, regressors, weight, fe, nofe, cluster, t28):
    """Estimate a clustered multivariate within regression."""
    regressors = np.asarray(regressors, np.float64)
    if regressors.ndim == 1:
        regressors = regressors[:, None]
    demeaned = t28.w22.demean(
        np.column_stack([x, regressors]), fe, nofe, weight)
    xt = demeaned[:, 0]
    zt = demeaned[:, 1:]
    scales = np.std(zt, axis=0, ddof=1)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise SystemExit("FAIL: a joint-regression component has zero variance")
    zs = zt / scales
    gram = zs.T @ (weight[:, None] * zs)
    rhs = zs.T @ (weight * xt)
    beta_scaled = np.linalg.solve(gram, rhs)
    residual = xt - zs @ beta_scaled

    groups = int(cluster.max()) + 1
    scores = np.zeros((groups, zs.shape[1]), np.float64)
    np.add.at(scores, cluster, zs * (weight * residual)[:, None])
    bread = np.linalg.inv(gram)
    n = len(x)
    k = t28.ssc_k(fe, nofe, cluster) + regressors.shape[1] - 1
    g = int(np.unique(cluster).size)
    correction = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    vcov_scaled = correction * bread @ (scores.T @ scores) @ bread
    transform = np.diag(1.0 / scales)
    beta = beta_scaled / scales
    vcov = transform @ vcov_scaled @ transform
    se = np.sqrt(np.diag(vcov))
    return beta, se, vcov, xt, zt


def main():
    """Run the fixed-sample clipping sensitivity and component decomposition."""
    start = time.time()
    print("== h=2 rolling-five-year shifter clipping diagnostics")
    p57 = load_module("p57_clip", STEP1_PY / "57_aggregation_invariance.py")
    t28 = load_module("t28_clip", STEP2_PY / "28_table_d1.py")
    import pyfixest as pf
    t28.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-12,
                                      fixef_maxiter=100_000)

    baseline, dropped = p57.build_county(
        "composition", h=H, span=locked.SPAN, rule=locked.RULE,
        floor=locked.FLOOR, denom=locked.DENOM)
    if dropped != 0:
        raise SystemExit(f"FAIL: baseline dropped {dropped} additional counties")
    if len(baseline) != locked.N_COUNTY[H]:
        raise SystemExit(
            f"FAIL: baseline has {len(baseline):,} rows, expected "
            f"{locked.N_COUNTY[H]:,}")

    units0, _, contract_years0, arr0, _ = store.load_tensor(locked.RULE)
    years0 = store.target_years(H)
    raw, live_counts, tail_counts = raw_shifter(
        arr0, contract_years0, years0)
    variants, components = shifter_variants(raw)
    all_shifters = {**variants, **{
        f"component_{name}": value for name, value in components.items()
    }}
    units, years, zwide, _, _ = county_instruments(all_shifters)
    if not np.array_equal(units, units0) or years != years0:
        raise SystemExit("FAIL: shifter and exposure axes differ")

    rowpos = {unit: i for i, unit in enumerate(units)}
    colpos = {year: j for j, year in enumerate(years)}
    ri = baseline["ccg"].map(rowpos).to_numpy()
    ci = baseline["year"].map(colpos).to_numpy()
    if pd.isna(ri).any() or pd.isna(ci).any():
        raise SystemExit("FAIL: baseline rows do not align to the stored panel")
    ri, ci = ri.astype(int), ci.astype(int)

    zsample = {name: matrix[ri, ci] for name, matrix in zwide.items()}
    missing = {name: int((~np.isfinite(z)).sum())
               for name, z in zsample.items() if not np.isfinite(z).all()}
    if missing:
        raise SystemExit(f"FAIL: missing county instruments: {missing}")
    if not np.allclose(zsample["cap_1"],
                       zsample["component_inside"]
                       + zsample["component_tail_cap"],
                       rtol=0.0, atol=1e-10):
        raise SystemExit("FAIL: county clip-1 identity does not hold")
    if not np.allclose(zsample["unclipped"],
                       zsample["cap_1"] + zsample["component_excess"],
                       rtol=0.0, atol=1e-8):
        raise SystemExit("FAIL: county unclipped identity does not hold")

    x, weight, fe, nofe, cluster = estimation_arrays(baseline)
    cap1 = zsample["cap_1"]
    sensitivity_rows = []
    labels = {
        **{f"cap_{cap:g}": f"Clip at {cap:g}" for cap in CAPS},
        "exclude_tail": "Set raw absolute shifter above 1 to zero",
        "unclipped": "No clipping",
    }
    total_live = int(sum(live_counts))
    total_tail = int(sum(tail_counts))
    for name in [*(f"cap_{cap:g}" for cap in CAPS),
                 "exclude_tail", "unclipped"]:
        z = zsample[name]
        fit = t28.fit2sls(x, z, z, weight, fe, nofe, cluster,
                          singleton=False)
        demeaned = t28.w22.demean(
            np.column_stack([x, z]), fe, nofe, weight)
        xt, zt = demeaned[:, 0], demeaned[:, 1]
        partial_r2 = float((zt @ xt) ** 2 / ((zt @ zt) * (xt @ xt)))
        tstat = float(fit["beta"] / fit["se"])
        sensitivity_rows.append({
            "variant": name,
            "label": labels[name],
            "horizon": H,
            "n": len(baseline),
            "n_counties": int(baseline["ccg"].nunique()),
            "fixed_effects": "division-by-year",
            "weight": "equal county",
            "cluster": "county",
            "first_stage": float(fit["beta"]),
            "first_stage_se": float(fit["se"]),
            "first_stage_t": tstat,
            "first_stage_F": tstat ** 2,
            "partial_r2": partial_r2,
            "instrument_sd": float(np.std(z, ddof=1)),
            "within_instrument_sd": float(np.std(zt, ddof=1)),
            "correlation_with_clip_1": float(np.corrcoef(z, cap1)[0, 1]),
            "live_category_years": total_live,
            "tail_category_years": total_tail,
            "tail_share_of_live_category_years": total_tail / total_live,
        })

    sequence = pd.read_csv(SEQUENCE_OUT)
    reference = sequence.loc[sequence["instrument"] == "C_matched_roll5"]
    if len(reference) != 1:
        raise SystemExit("FAIL: sequence output lacks one rolling-five-year row")
    cap1_row = next(row for row in sensitivity_rows if row["variant"] == "cap_1")
    for key in ("first_stage", "first_stage_se", "first_stage_F", "partial_r2"):
        expected = float(reference.iloc[0][key])
        gap = abs(cap1_row[key] - expected) / max(abs(expected), 1e-300)
        if gap >= locked.RTOL_COEF:
            raise SystemExit(f"FAIL: cap-1 {key} reference gap is {gap:.3e}")

    component_names = ("inside", "tail_cap", "clip_1", "excess", "unclipped")
    component_z = {
        name: zsample[f"component_{name}"] for name in component_names
    }
    component_matrix = np.column_stack([component_z[name]
                                        for name in component_names])
    demeaned = t28.w22.demean(
        np.column_stack([x, component_matrix]), fe, nofe, weight)
    xt = demeaned[:, 0]
    zt = {name: demeaned[:, j + 1]
          for j, name in enumerate(component_names)}
    covariance = {name: float(zt[name] @ (weight * xt))
                  for name in component_names}
    if not np.isclose(covariance["clip_1"],
                      covariance["inside"] + covariance["tail_cap"],
                      rtol=1e-12, atol=1e-8):
        raise SystemExit("FAIL: clip-1 first-stage covariance does not decompose")
    if not np.isclose(covariance["unclipped"],
                      covariance["clip_1"] + covariance["excess"],
                      rtol=1e-12, atol=1e-8):
        raise SystemExit("FAIL: raw first-stage covariance does not decompose")

    joint_beta, joint_se, _, _, _ = joint_first_stage(
        x, np.column_stack([component_z["clip_1"], component_z["excess"]]),
        weight, fe, nofe, cluster, t28)
    one_beta, one_se, _, _, _ = joint_first_stage(
        x, component_z["clip_1"], weight, fe, nofe, cluster, t28)
    if not np.isclose(one_beta[0], cap1_row["first_stage"], rtol=1e-11):
        raise SystemExit("FAIL: multivariate estimator beta cross-check failed")
    if not np.isclose(one_se[0], cap1_row["first_stage_se"], rtol=1e-10):
        raise SystemExit("FAIL: multivariate estimator SE cross-check failed")

    component_rows = []
    for name in component_names:
        component_rows.append({
            "component": name,
            "horizon": H,
            "n": len(baseline),
            "within_instrument_sd": float(np.std(zt[name], ddof=1)),
            "within_covariance_with_treatment": covariance[name] / len(x),
            "share_of_clip_1_covariance":
                covariance[name] / covariance["clip_1"]
                if name in ("inside", "tail_cap", "clip_1") else np.nan,
            "share_of_unclipped_covariance":
                covariance[name] / covariance["unclipped"]
                if name in ("clip_1", "excess", "unclipped") else np.nan,
            "joint_first_stage":
                float(joint_beta[0]) if name == "clip_1"
                else float(joint_beta[1]) if name == "excess" else np.nan,
            "joint_first_stage_se":
                float(joint_se[0]) if name == "clip_1"
                else float(joint_se[1]) if name == "excess" else np.nan,
            "joint_first_stage_t":
                float(joint_beta[0] / joint_se[0]) if name == "clip_1"
                else float(joint_beta[1] / joint_se[1])
                if name == "excess" else np.nan,
            "joint_first_stage_F":
                float((joint_beta[0] / joint_se[0]) ** 2) if name == "clip_1"
                else float((joint_beta[1] / joint_se[1]) ** 2)
                if name == "excess" else np.nan,
        })

    SENSITIVITY_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sensitivity_rows).to_csv(SENSITIVITY_OUT, index=False)
    pd.DataFrame(component_rows).to_csv(DECOMPOSITION_OUT, index=False)

    print(f"   sample: {len(baseline):,} county-years, "
          f"{baseline['ccg'].nunique():,} counties")
    print(f"   tail category-years: {total_tail:,}/{total_live:,} "
          f"({100 * total_tail / total_live:.2f}%)")
    for row in sensitivity_rows:
        print(f"   {row['variant']:>12}: F={row['first_stage_F']:.3f}, "
              f"partial R2={row['partial_r2']:.6f}, "
              f"pi={row['first_stage']:+.6f}, "
              f"corr(clip1)={row['correlation_with_clip_1']:+.3f}")
    print("   covariance shares:")
    print(f"      inside / clip1 = "
          f"{100 * covariance['inside'] / covariance['clip_1']:+.2f}%")
    print(f"      capped tail / clip1 = "
          f"{100 * covariance['tail_cap'] / covariance['clip_1']:+.2f}%")
    print(f"      clip1 / raw = "
          f"{100 * covariance['clip_1'] / covariance['unclipped']:+.2f}%")
    print(f"      excess / raw = "
          f"{100 * covariance['excess'] / covariance['unclipped']:+.2f}%")
    print("   joint clip1 + excess first stage:")
    print(f"      clip1: pi={joint_beta[0]:+.6f}, se={joint_se[0]:.6f}, "
          f"t={joint_beta[0] / joint_se[0]:+.3f}")
    print(f"      excess: pi={joint_beta[1]:+.6f}, se={joint_se[1]:.6f}, "
          f"t={joint_beta[1] / joint_se[1]:+.3f}")
    print(f"   wrote {SENSITIVITY_OUT}")
    print(f"   wrote {DECOMPOSITION_OUT}")
    print(f"   done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
