"""Stage 5: the standing correctness gate. Fail loud, rerun any time.

Three layers:
  0. locked.py is asserted against the old modules field by field
     (lib/oldpath.check_lock), so constant drift in either copy fails here.
  a. Vector layer: analysis_panel.parquet against the golden county frames
     (the old build_county output), merged on (ccg, year) per horizon, NaN
     masks equal, values at RTOL_VEC/ATOL_VEC. A diagnostic block reports
     the max relative gap by year (seam years flagged) even on pass.
  b. Coefficient layer: every row of the frozen aggregation_baseline.csv
     (5 levels x 3 horizons x EW/P_j, all coefficient columns) re-estimated
     from analysis_panel.parquet with the ported baseline loop (57:566-647),
     using step_2's fit2sls/demean via the oldpath harness, asserted at
     RTOL_COEF.

The chain this closes: step 0b proved the old code reproduces the frozen CSV
(gap 0); layer (a) proves the persisted panel equals the old in-memory
frame; layer (b) proves the estimation from the persisted panel lands on the
frozen numbers.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import panel as pnl_lib  # noqa: E402
import oldpath as op  # noqa: E402

PIPE = op.dc.MACRO_DATA_ROOT / "temp" / "pipeline"
GOLD = PIPE / "_golden"
SEAM_FLAG = {1979, 1989, 2007, 2011, 2012}

VEC_COLS = [("z", "z"), ("x", "x_pos"), ("y_mpgr", "y_mpgr"),
            ("y_std", "y_std"), ("y_log", "y_log"), ("E_b", "E_b"),
            ("E_t", "E_t"), ("P_b", "P_b"), ("P0", "popw")]
LABEL_COLS = ["state", "division", "czone", "msa", "cea", "ea"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def baseline_manifest_gate():
    """Verify the declared active geography baseline and its persisted assets."""
    path = GOLD / "active_baseline.json"
    if not path.exists():
        raise AssertionError(f"active baseline manifest is absent: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["baseline_id"] == "verified_production_geography_v1"
    for relative, expected in manifest["assets_sha256"].items():
        asset = PIPE / relative
        assert asset.exists(), asset
        observed = sha256(asset)
        assert observed == expected, (relative, expected, observed)
    print(f"--- active baseline: {manifest['baseline_id']}; "
          f"{len(manifest['assets_sha256'])} asset hashes tie")


def layer_a(apanel):
    print("--- layer (a): vectors vs golden county frames")
    for h in locked.HORIZONS:
        ref = pd.read_parquet(GOLD / f"county_h{h}.parquet")
        new = apanel[apanel["h"] == h]
        m = ref.merge(new, on=["ccg", "year"], suffixes=("_ref", "_new"))
        # The golden frames were captured at TARGET_LAST = 2018. Since
        # 2026-09-02 the sample ends at 2016, so the panel is a strict SUBSET of
        # golden by whole target years. What this layer verifies is the
        # CONSTRUCTION, not the sample, so the test is that every row the panel
        # has is present in golden and agrees exactly. A shorter panel passes; a
        # panel with a row golden does not have, or a row that disagrees, does
        # not. If the sample is ever extended past golden's, this assert fires
        # and the golden set has to be recaptured rather than quietly outgrown.
        assert len(m) == len(new), (h, len(ref), len(new), len(m))
        assert new["year"].max() <= ref["year"].max(), (h, "panel outruns golden")
        worst = 0.0
        by_year = {}
        for rc, nc in VEC_COLS:
            a = m[rc + ("_ref" if rc + "_ref" in m.columns else "")].to_numpy(float) \
                if rc + "_ref" in m.columns else m[rc].to_numpy(float)
            b = m[nc + ("_new" if nc + "_new" in m.columns else "")].to_numpy(float) \
                if nc + "_new" in m.columns else m[nc].to_numpy(float)
            assert np.array_equal(np.isnan(a), np.isnan(b)), (h, rc)
            ok = ~np.isnan(a)
            rel = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(a[ok]), 1e-300)
            rel[a[ok] == b[ok]] = 0.0
            if len(rel):
                worst = max(worst, rel.max())
                yr = m.loc[ok, "year"].to_numpy()
                for y in np.unique(yr):
                    g = rel[yr == y].max()
                    by_year[y] = max(by_year.get(y, 0.0), g)
            if not np.allclose(a, b, rtol=locked.RTOL_VEC,
                               atol=locked.ATOL_VEC, equal_nan=True):
                raise AssertionError(
                    f"h={h} column {rc}: max rel gap {rel.max():.3e} exceeds "
                    f"RTOL_VEC={locked.RTOL_VEC}")
        for rc in LABEL_COLS:
            a = m[f"{rc}_ref"] if f"{rc}_ref" in m.columns else m[rc]
            b = m[f"{rc}_new"] if f"{rc}_new" in m.columns else m[rc]
            assert np.array_equal(a.astype(str).to_numpy(),
                                  b.astype(str).to_numpy()), (h, rc)
        top = sorted(by_year.items(), key=lambda kv: -kv[1])[:3]
        seam = {y: g for y, g in by_year.items() if y in SEAM_FLAG}
        zrows = m["z"] == 0 if "z" in m.columns else m["z_ref"] == 0
        print(f"  h={h}: {len(m):,} rows, max rel gap {worst:.3e}; "
              f"worst years {[(int(y), f'{g:.1e}') for y, g in top]}; "
              f"seam years {[(int(y), f'{g:.1e}') for y, g in sorted(seam.items())]}; "
              f"z==0 rows: {int(zrows.sum()):,}")


def estimate_baseline(apanel):
    """Return the complete 30-row locked aggregation baseline."""
    state_division = (apanel.drop_duplicates("state")
                      .set_index("state")["division"])
    SCALE = locked.SCALE
    rows = []
    for h in locked.HORIZONS:
        d = apanel[apanel["h"] == h].rename(columns={"popw": "P0"}).copy()
        for lev in locked.LEVELS:
            g = d[d[lev].notna()].copy()
            g["om"] = g["E_b"] / g.groupby([lev, "year"])["E_b"].transform("sum")
            key = pd.MultiIndex.from_arrays([g[lev], g["year"]])
            shr = (g.assign(wy=g["om"] * g[locked.OUTCOME], wz=g["om"] * g["z"])
                     .groupby([lev, "year"])[["wy", "wz"]].sum())
            g["share"] = shr["wz"].reindex(key).to_numpy()
            agg = shr.reset_index().rename(columns={"wy": "y", "wz": "share"})
            P0j = g.drop_duplicates("ccg").groupby(lev)["P0"].sum()
            # unit_labels (57:424-450): plurality-of-base-population state,
            # with the state level special-cased (a state is its own label)
            u = d.drop_duplicates("ccg")
            if lev == "state":
                v = pd.unique(u["state"])
                ust, straddle = pd.Series(v, index=v), 0
            else:
                pop = u.groupby([lev, "state"])["P0"].sum().reset_index()
                ust = (pop.sort_values("P0", ascending=False)
                          .drop_duplicates(lev).set_index(lev)["state"])
                straddle = int((u.groupby(lev)["state"].nunique() > 1).sum())
            div = ust.map(state_division)
            di, da = g[lev].map(div), agg[lev].map(div)
            rk_i = di.astype(str) + "_" + g["year"].astype(str)
            rk_a = da.astype(str) + "_" + agg["year"].astype(str)
            for cj in locked.SCALARS:
                wc = (g["om"] if cj == "EW"
                      else g["om"] * g[lev].map(P0j)).to_numpy(np.float64)
                wa = (np.ones(len(agg)) if cj == "EW"
                      else agg[lev].map(P0j).to_numpy(np.float64))
                wc, wa = wc / wc.mean(), wa / wa.mean()
                f1 = pd.factorize(rk_i)[0].astype(np.int64)
                f1a = pd.factorize(rk_a)[0].astype(np.int64)
                ze = np.zeros(len(g), np.int64)
                zea = np.zeros(len(agg), np.int64)
                clc = pd.factorize(g[lev])[0].astype(np.int64)
                cla = pd.factorize(agg[lev])[0].astype(np.int64)

                y = g[locked.OUTCOME].to_numpy(np.float64)
                z = g["z"].to_numpy(np.float64)
                sh = g["share"].to_numpy(np.float64)
                M = op.w22.demean(np.column_stack([y, z, sh]), f1, ze, wc)
                A = np.column_stack([M[:, 1], M[:, 2]])
                sol = np.linalg.solve(A.T @ (wc[:, None] * A),
                                      A.T @ (wc * M[:, 0]))
                bb, dl = float(sol[0]), float(sol[1])
                Ma = op.w22.demean(np.column_stack([agg["y"], agg["share"]]),
                                   f1a, zea, wa)
                ba = float(Ma[:, 1] @ (wa * Ma[:, 0])) / float(
                    Ma[:, 1] @ (wa * Ma[:, 1]))
                gap = abs(ba - bb - dl) * SCALE

                Ej = g.groupby([lev, "year"])[["E_b", "E_t"]].sum()
                own = 2.0 * (Ej["E_t"] - Ej["E_b"]) / (Ej["E_t"] + Ej["E_b"])
                own = own.reindex(pd.MultiIndex.from_frame(
                    agg[[lev, "year"]])).to_numpy()
                Mo = op.w22.demean(np.column_stack([own, agg["share"]]),
                                   f1a, zea, wa)
                bo = float(Mo[:, 1] @ (wa * Mo[:, 0])) / float(
                    Mo[:, 1] @ (wa * Mo[:, 1]))
                ro = op.t28.fit2sls(own, agg["share"].to_numpy(np.float64),
                                    agg["share"].to_numpy(np.float64),
                                    wa, f1a, zea, cla, singleton=False)
                rc = op.t28.fit2sls(y, z, z, wc, f1, ze, clc, singleton=False)
                ra = op.t28.fit2sls(agg["y"].to_numpy(np.float64),
                                    agg["share"].to_numpy(np.float64),
                                    agg["share"].to_numpy(np.float64),
                                    wa, f1a, zea, cla, singleton=False)
                rows.append(dict(
                    horizon=h, level=lev, weight=cj,
                    n_units=int(g[lev].nunique()), n_county=len(g),
                    n_agg=len(agg), straddle=straddle,
                    beta_b=bb * SCALE, delta=dl * SCALE,
                    beta_a_aligned=ba * SCALE, beta_a_own=bo * SCALE,
                    jensen=(bo - ba) * SCALE, construction_gap=gap,
                    t_agg_own=ro["beta"] / ro["se"],
                    t_county=rc["beta"] / rc["se"],
                    t_agg=ra["beta"] / ra["se"]))
    return pd.DataFrame(rows)


def layer_b(apanel):
    print("--- layer (b): every frozen aggregation_baseline.csv row re-estimated")
    frozen = pd.read_csv(GOLD / "frozen_aggregation_baseline.csv")
    new = estimate_baseline(apanel)
    key = ["horizon", "level", "weight"]
    m = frozen.merge(new, on=key, suffixes=("_ref", "_new"))
    assert len(m) == len(frozen) == 30, (len(frozen), len(new), len(m))
    worst, worst_cell = 0.0, None
    for c in [c for c in frozen.columns if c not in key]:
        a = m[f"{c}_ref"].to_numpy(float)
        b = m[f"{c}_new"].to_numpy(float)
        # construction_gap is a machine-zero identity residual; compare in
        # absolute terms at SCALE like 57 reports it, not relatively.
        if c == "construction_gap":
            assert np.abs(a - b).max() < 1e-8, c
            continue
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-12)
        if rel.max() > worst:
            worst = rel.max()
            worst_cell = (c, m.loc[rel.argmax(), key].tolist())
        if not np.allclose(a, b, rtol=locked.RTOL_COEF, atol=0):
            raise AssertionError(
                f"coefficient column {c}: max rel gap {rel.max():.3e} at "
                f"{m.loc[rel.argmax(), key].tolist()} exceeds "
                f"RTOL_COEF={locked.RTOL_COEF}")
    print(f"  30/30 rows tie; max rel gap {worst:.3e} at {worst_cell}")


def main():
    t0 = time.time()
    baseline_manifest_gate()
    n = op.check_lock(locked)
    print(f"--- lock check: {n} constants match the old modules")
    apanel = pd.read_parquet(PIPE / "analysis_panel.parquet")
    for h in locked.HORIZONS:
        nh = (apanel["h"] == h).sum()
        assert nh == locked.N_COUNTY[h], (h, nh)
    assert apanel["ccg"].nunique() == locked.N_COUNTIES
    layer_a(apanel)
    layer_b(apanel)
    print(f"05_baseline_tieout: PASS in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
