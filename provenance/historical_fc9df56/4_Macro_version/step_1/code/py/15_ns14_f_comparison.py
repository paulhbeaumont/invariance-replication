#!/usr/bin/env python3
"""Our instrument against Nakamura and Steinsson's, on their own diagnostic.

14_build_instrument.py already reports a first-stage F at every rung, and that F
is cluster-robust because plan section 1.5 wants inference clustered on the unit.
Nakamura and Steinsson report something else. Footnote 30 of the 2014 AER gives a
CRAGG-DONALD statistic, which is homoskedastic and unclustered, so 14's numbers
cannot be laid beside theirs without recomputing ours their way. That is all this
file does. It builds no data and changes nothing 14 wrote.

WHICH NS14 INSTRUMENT IS OURS

Footnote 30 reports two, and they are twenty times apart:

  baseline      one instrument per unit, the interaction of unit dummies with the
                national buildup. The first stage carries a unit-specific slope
                gamma_u, estimated from the panel.
                    Cragg-Donald F ~ 5 (state), ~ 8 (region); partial R2 12%, 18%
  Bartik-type   ONE instrument, a pre-set exposure scalar times the national
                buildup. Footnote 32 says the scaling factor is "the average level
                of military spending in each state relative to state output".
                    Cragg-Donald F = 106 (state), 53 (region)

Ours is the second. Z_ct = s_c x buildup_t is a single excluded instrument with an
exposure imposed rather than estimated, so the row of their footnote we belong
next to is the 106/53 one, not the 5/8 one. The `ns` exposure of 14 is their s_i
transcribed - the mean over the sample of the annual ratio - which makes the
correspondence exact rather than approximate for that column.

To show that this is a claim about our instrument and not about our data, the
NS14 BASELINE is also run here, on our panel, at the rungs where it is feasible:
x on i(unit, buildup) with unit and year fixed effects. If the mapping is right,
that specification should reproduce their weak-instrument problem in our data.

WHAT IS COMPUTED

For one endogenous regressor the Cragg-Donald statistic IS the homoskedastic F on
the excluded instruments, so both the single-instrument and many-instrument cases
come from the same three numbers:

    RSS_r   from x on the fixed effects alone
    RSS_u   from x on the fixed effects and the instruments
    K       the RANK of the instrument matrix after the fixed effects are absorbed

    CD F           = ((RSS_r - RSS_u) / K) / (RSS_u / (N - dfFE - K))
    partial R2     = (RSS_r - RSS_u) / RSS_r

K is the rank and not the column count on purpose. Under year fixed effects the
unit-by-buildup interactions sum to something within rounding of the year effect
itself, so one column of the baseline design is very nearly redundant; taking the
column count would divide by a K that is one too large and quietly flatter the F.

Fixed effects are absorbed by alternating projections rather than by dummies, and
the single-instrument F is checked against pyfixest's iid t-squared, which is the
same number by a different route.

Output:
    dc.LOG_DIR/ns14_f_comparison.csv          our instrument, every rung
    dc.LOG_DIR/ns14_baseline_analogue.csv     their baseline, run on our panel
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcads_common as dc  # noqa: E402

# --------------------------------------------------------------------------------
# 14 owns the rung map and the collapse. Import it rather than restating it, so a
# later change there cannot leave this file quietly describing a different ladder.
# The leading digit makes it unimportable by name, hence the explicit spec.
# --------------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "build_instrument", Path(__file__).resolve().parent / "14_build_instrument.py")
bi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bi)

# Nakamura and Steinsson (2014), AER 104(3), footnote 30, p. 14. Transcribed so
# the comparison sits in the output file and not only in a reader's memory.
NS14 = {
    ("baseline", "state"): dict(F=5.0, partial_R2=0.12, approx=True),
    ("baseline", "region"): dict(F=8.0, partial_R2=0.18, approx=True),
    ("bartik", "state"): dict(F=106.0, partial_R2=np.nan, approx=False),
    ("bartik", "region"): dict(F=53.0, partial_R2=np.nan, approx=False),
}

RUNGS = (("county", "ccg"), ("commuting zone", "cz"),
         ("state", "state"), ("Census division", "division"))
FE_SPECS = (("none", ()), ("year", ("year",)), ("unit+year", ("unit", "year")))

# The baseline design is N x n_units dense once the fixed effects are absorbed.
# At the county rung it exceeds the declared cap and is not a first stage anyone
# would report. The cap is stated rather than discovered.
BASELINE_MAX_UNITS = 1_000


def demean(mat, codes, tol=1e-12, maxiter=200):
    """Absorb fixed effects by alternating projections.

    `codes` is a list of integer group vectors, one per fixed effect. Returns the
    residual of `mat` on the union of those effects. One group is projected at a
    time and the sweep repeats until nothing moves, which is exact in the limit
    and converges in a handful of passes for a well-behaved two-way panel.
    """
    x = np.asarray(mat, dtype=np.float64).copy()
    if x.ndim == 1:
        x = x[:, None]
    if not codes:
        return x - x.mean(axis=0, keepdims=True)
    counts = [np.bincount(c) for c in codes]
    for _ in range(maxiter):
        before = x.copy()
        for c, n in zip(codes, counts):
            means = np.zeros((len(n), x.shape[1]))
            np.add.at(means, c, x)
            means /= np.maximum(n, 1)[:, None]
            x -= means[c]
        if np.abs(x - before).max() < tol:
            break
    return x


def f_stat(x, Z, codes, dffe):
    """Cragg-Donald F and partial R2 of Z for the endogenous regressor x.

    x and Z are residualised on the fixed effects first, which is Frisch-Waugh and
    makes the two sums of squares comparable by construction.
    """
    n = len(x)
    xt = demean(x, codes).ravel()
    zt = demean(Z, codes)
    rss_r = float(xt @ xt)
    k = int(np.linalg.matrix_rank(zt, tol=1e-10))
    if k == 0 or rss_r <= 0:
        return dict(cd_F=np.nan, partial_R2=np.nan, k_rank=k,
                    k_cols=zt.shape[1], n_obs=n, df_resid=np.nan)
    beta, *_ = np.linalg.lstsq(zt, xt, rcond=None)
    resid = xt - zt @ beta
    rss_u = float(resid @ resid)
    df = n - dffe - k
    explained = max(rss_r - rss_u, 0.0)
    return dict(cd_F=(explained / k) / (rss_u / df) if df > 0 and rss_u > 0 else np.nan,
                partial_R2=explained / rss_r,
                k_rank=k, k_cols=zt.shape[1], n_obs=n, df_resid=df)


def codes_for(frame, fe):
    """Integer group vectors and the fixed effects' degrees of freedom.

    The intercept is always one of them. Two-way effects share it, hence the -1:
    a unit-and-year design absorbs n_units + n_years - 1 parameters, not the sum.
    """
    if not fe:
        return [], 1
    out, dffe = [], 1
    for name in fe:
        c = pd.factorize(frame[name])[0]
        out.append(c.astype(np.int64))
        dffe += int(c.max()) + 1 - 1
    return out, dffe


def cross_check(frame, fe):
    """The single-instrument F again, from pyfixest under an iid variance.

    A rank-one Cragg-Donald statistic is the squared homoskedastic t, so this and
    f_stat must agree. They are computed by entirely different routes, which is
    the only reason the agreement is worth asserting.
    """
    import pyfixest as pf
    f = "x ~ z" if not fe else "x ~ z | " + " + ".join(fe)
    fit = pf.feols(f, frame, vcov="iid")
    b, se = float(fit.coef()["z"]), float(fit.se()["z"])
    return (b / se) ** 2 if se > 0 else np.nan


def prepare(county_all, rungs, h, exposure):
    """The estimation sample of 14, rebuilt here so the two cannot drift apart.

    14's collapse wants three columns the saved parquet does not carry. `own_dD`
    is recovered exactly, since the shock is that change over lagged income and
    nothing was rounded on the way out. `pre_D` and `pre_Y` are the 1969-74 window
    sums, which feed only 14's alternative-construction diagnostic and nothing
    here; rebuilding them would mean re-reading the whole contract panel, so they
    are set to NaN and dropped in unit_frame before anything can read them. If a
    later edit makes this file need them, rebuild them - do not trust the NaN.
    """
    base = county_all[county_all["h"] == h].copy()
    base["z"] = base[f"z_{exposure}"]
    base["own_dD"] = base["shock"] * base["income_real_lag"]
    base["pre_D"] = np.nan
    base["pre_Y"] = np.nan
    base = base.merge(rungs, on="ccg", how="left")
    keep = np.isfinite(base[["shock", "emp_growth", "z", "income_real_lag",
                             "emp_lag"]].to_numpy()).all(axis=1)
    return base[keep & (base["emp_lag"] > 0) & (base["income_real_lag"] > 0)]


def unit_frame(base, rung, unit_col, weight_col):
    """One rung's collapsed data, carrying BOTH candidate shifters.

    `buildup_nat` is the common national series, D^nat over lagged Y^nat, with no
    subtraction. `buildup_loo` nets out the unit's own dollars. The distinction is
    immaterial at the county rung and decisive at the top of the ladder, so both
    are carried and the baseline analogue is run on each - see baseline_design.
    """
    if rung == "county":
        d = base.rename(columns={"emp_growth": "y", "shock": "x"}).copy()
        d["unit"] = d["ccg"]
        with np.errstate(divide="ignore", invalid="ignore"):
            d["buildup_nat"] = d["nat_dD"] / d["nat_Y_lag"]
        d["buildup_loo"] = d["buildup"]
        return d[["unit", "year", "y", "x", "z", "buildup_nat", "buildup_loo"]]
    d = bi.collapse(base, unit_col, weight_col)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["buildup_nat"] = d["nat_dD"] / d["nat_Y_lag"]
        d["buildup_loo"] = ((d["nat_dD"] - d["unit_dD"])
                            / (d["nat_Y_lag"] - d["unit_Y_lag"]))
    return d[["unit", "year", "y", "x", "z", "buildup_nat", "buildup_loo"]]


# The two shifters the baseline analogue is run on, and why both are needed.
#
#   national   the common series, which is what NS14 interact with state dummies.
#              Faithful to them. Under year fixed effects the K interactions sum
#              to the year effect, so the design loses a rank and K comes back as
#              n_units - 1. That rank loss is theirs too and is not a defect.
#   unit LOO   our default shifter. At the county rung it is the national series
#              to four decimals. At the division rung, where one unit is a ninth
#              of national procurement, the subtraction leaves the unit's OWN
#              dollars in the shifter with a negative sign, so the interaction is
#              mechanically related to the regressor and its partial R2 is not
#              comparable to anything NS14 report. Run and reported so the
#              inflation is visible rather than inherited.
SHIFTERS = (("national", "buildup_nat"), ("unit LOO", "buildup_loo"))


def baseline_design(d, shifter_col):
    """The NS14 baseline instrument set: unit dummies interacted with the shifter.

    Column u holds the shifter on unit u's rows and zero elsewhere, which is what
    i(unit, shifter) expands to and what gives the first stage a slope per unit.
    """
    u, levels = pd.factorize(d["unit"])
    Z = np.zeros((len(d), len(levels)))
    Z[np.arange(len(d)), u] = d[shifter_col].to_numpy()
    return Z


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horizon", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--exposure", nargs="+", default=list(bi.EXPOSURES),
                    choices=list(bi.EXPOSURES))
    ap.add_argument("--weights", default="employment",
                    choices=("employment", "income"),
                    help="the omega that collapses the shock and the instrument")
    # The window is stated, never inherited. instrument_ccg_year.parquet spans
    # 1975-2018 since the FY2007 splice, and this file used to read all of it, so
    # its numbers would silently have changed sample the moment the panel grew.
    ap.add_argument("--period", default="1975-2006", choices=list(bi.PERIODS),
                    help="estimation window; the default is the frozen design")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    wcol = "emp_lag" if args.weights == "employment" else "income_real_lag"
    horizons = sorted(set(args.horizon))
    y0, y1 = bi.PERIODS[args.period]
    names = [e for e in bi.EXPOSURES if e in set(args.exposure)
             and e in bi.ADMISSIBLE.get(args.period, set(bi.EXPOSURES))]

    print("\nour instrument on Nakamura and Steinsson's diagnostic")
    print(f"    horizons {horizons}, exposures {names}, {args.weights} weights, "
          f"{args.period}")
    print("\n    NS14 footnote 30, for reference")
    print("        baseline    one instrument per unit    F ~ 5 state, ~ 8 region"
          "    partial R2 12%, 18%")
    print("        Bartik      one instrument             F = 106 state, 53 region")

    county_all = pd.read_parquet(dc.PANEL_OUT_DIR / "instrument_ccg_year.parquet")
    county_all = county_all[county_all["year"].between(y0, y1)]
    _, _, universe = dc.load_analysis_ccg_map()
    rungs = bi.load_rungs(universe)

    rows, base_rows = [], []
    worst_check = 0.0
    for h in horizons:
        for exposure in names:
            base = prepare(county_all, rungs, h, exposure)
            for rung, unit_col in RUNGS:
                d = unit_frame(base, rung, unit_col, wcol)
                d = d[np.isfinite(d[["x", "z"]].to_numpy()).all(axis=1)]
                Z = d[["z"]].to_numpy()
                for fe_name, fe in FE_SPECS:
                    codes, dffe = codes_for(d, fe)
                    r = f_stat(d["x"].to_numpy(), Z, codes, dffe)
                    ref = cross_check(d, fe)
                    if np.isfinite(ref) and np.isfinite(r["cd_F"]):
                        worst_check = max(
                            worst_check, abs(r["cd_F"] - ref) / max(ref, 1.0))
                    rows.append(dict(rung=rung, n_units=int(d["unit"].nunique()),
                                     h=h, exposure=exposure, fe=fe_name,
                                     instrument="bartik (ours)", **r,
                                     pyfixest_iid_F=ref))

                # The baseline instrument set is unit identity interacted with a
                # shifter and never touches s_c, so it is the same regression for
                # every exposure. Run it once, on the first, rather than three
                # times to three identical rows.
                if (args.skip_baseline or exposure != names[0]
                        or d["unit"].nunique() > BASELINE_MAX_UNITS):
                    continue
                codes, dffe = codes_for(d, ("unit", "year"))
                for shifter, scol in SHIFTERS:
                    r = f_stat(d["x"].to_numpy(), baseline_design(d, scol),
                               codes, dffe)
                    base_rows.append(dict(rung=rung,
                                          n_units=int(d["unit"].nunique()),
                                          h=h, shifter=shifter, fe="unit+year",
                                          instrument="NS14 baseline", **r))

    sweep = pd.DataFrame(rows)
    dc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(dc.LOG_DIR / "ns14_f_comparison.csv", index=False)
    print(f"\n[1] {len(sweep):,} fits -> {dc.LOG_DIR / 'ns14_f_comparison.csv'}")
    print(f"    rank-one Cragg-Donald equals pyfixest's iid t-squared, "
          f"max relative gap {worst_check:.2e}   "
          f"{'PASS' if worst_check < 1e-8 else 'FAIL'}")

    head = sweep[(sweep["fe"] == "unit+year")].copy()
    for h in horizons:
        print(f"\n[2] h={h}, unit and year fixed effects, our Bartik instrument")
        print(f"    {'rung':<17s} {'units':>6s} {'obs':>8s}  " +
              "  ".join(f"{e:>16s}" for e in names))
        print(f"    {'':<17s} {'':>6s} {'':>8s}  " +
              "  ".join(f"{'CD F   part.R2':>16s}" for _ in names))
        for rung, _ in RUNGS:
            sel = head[(head["h"] == h) & (head["rung"] == rung)]
            if sel.empty:
                continue
            first = sel.iloc[0]
            cells = []
            for e in names:
                s = sel[sel["exposure"] == e]
                if s.empty:
                    cells.append(f"{'--':>16s}")
                else:
                    cells.append(f"{s.iloc[0]['cd_F']:>8.1f} "
                                 f"{100 * s.iloc[0]['partial_R2']:>6.1f}%")
            print(f"    {rung:<17s} {int(first['n_units']):>6,} "
                  f"{int(first['n_obs']):>8,}  " + "  ".join(cells))

    if base_rows:
        bframe = pd.DataFrame(base_rows)
        bframe.to_csv(dc.LOG_DIR / "ns14_baseline_analogue.csv", index=False)
        print(f"\n[3] NS14's own baseline, run on our panel -> "
              f"{dc.LOG_DIR / 'ns14_baseline_analogue.csv'}")
        print(f"    {'rung':<17s} {'h':>2s} {'shifter':<9s} {'units':>6s} "
              f"{'K':>5s} {'obs':>8s} {'CD F':>8s} {'part.R2':>8s}")
        for _, r in bframe.sort_values(["h", "shifter", "n_units"]).iterrows():
            print(f"    {r['rung']:<17s} {int(r['h']):>2d} {r['shifter']:<9s} "
                  f"{int(r['n_units']):>6,} {int(r['k_rank']):>5,} "
                  f"{int(r['n_obs']):>8,} {r['cd_F']:>8.1f} "
                  f"{100 * r['partial_R2']:>7.1f}%")
        county_ref = head[(head["rung"] == "county")
                          & (head["h"] == max(horizons))].iloc[0]
        k_county = int(county_ref["n_units"])
        n_county = int(county_ref["n_obs"])
        print(f"\n    counties not attempted: {BASELINE_MAX_UNITS:,}-unit cap, "
              f"{k_county:,} instruments on {n_county:,} observations is "
              f"N/K = {n_county / k_county:.0f}")

    return 0 if worst_check < 1e-8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
