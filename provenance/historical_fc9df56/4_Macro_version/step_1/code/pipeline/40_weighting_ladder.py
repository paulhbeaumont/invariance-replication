"""Producer 40 (4X band): the county regression under the weight family.

ONE regression, run under twelve weights. The basic specification of the locked
baseline, log employment growth on the composition instrument, estimated at the
county level throughout. Nothing about the estimating equation changes across
the table. The only thing that moves is which counties carry how much weight.

THE WEIGHT FAMILY, AND WHY IT IS A FAMILY

A county's weight is its employment share inside a region j, times one constant
per region:

    w_c  =  om_c  x  c_j          om_c = E_{c,t-h} / sum_{i in j} E_{i,t-h}

    c_j = 1     EW,  every region counts once
    c_j = P_j   Pop, every person counts once (P_j fixed at POP_BASE_YEAR)

That is the shape the aggregation identity requires: shares inside, one
constant per region outside. Indexing j by the rung turns the two baseline arms
into a ladder, because at j = county the group is the county itself, om_c = 1,
and the two columns collapse to w_c = 1 and w_c = P_c. Those are exactly the
`wc = None` and `wc = P0` arms of step_2/34_tables_payoff.py:193, so the county
row is not a new number and is the table's own correctness check.

om and P0j are built the same way producer 68 builds them (68:141-142,
68:172-183), so a cell here and a cell there mean the same object.

WHAT THIS TABLE IS NOT

68 reports beta_b, delta and beta_a^own: the county coefficient from the
AUGMENTED regression (own z plus the rung-level share of z), the cross-level
term, and the aggregate's own-growth coefficient. This table reports the basic
specification only. There is no share control and no aggregate-side regression,
so `beta` here is not 68's `beta_b` and the two do not tie out. The rung enters
through the weight and nowhere else.

THE FIXED EFFECT DOES NOT MOVE WITH THE RUNG

Division-by-year on the COUNTY's own division, at every rung, which is the
locked baseline's effect. 57 instead maps a rung unit to the division of its
plurality state and puts the effect there, which is right for an aggregate-side
regression and wrong here: it would make the rung change the controls as well
as the weight, and then a movement down a column would have two causes.

Outcome is y_log, the county's own log growth over [t-h, t]. Coefficients are
scaled by locked.SCALE, so a unit is percentage points per $1,000 of contracts
per head. Standard errors cluster on the county.

    .venv/bin/python step_1/code/pipeline/40_weighting_ladder.py
    .venv/bin/python step_1/code/pipeline/40_weighting_ladder.py --horizons 2 3 4
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import locked  # noqa: E402
import store  # noqa: E402
import dcads_common as dc  # noqa: E402

PIPE = store.PIPE
RESULTS = dc.MACRO_RESULTS_ROOT / "weighting_ladder"
CSV_OUT = RESULTS / "weighting_ladder.csv"
TEX_OUT = RESULTS / "Table_weighting_ladder.tex"
LOG = dc.LOG_DIR / "40_weighting_ladder.log"

OUTCOME = "y_log"
# County first: it is the rung at which om_c = 1 and the two columns are the
# locked baseline's own two arms. Then up the ladder.
#
# Division is on the end and is NOT in locked.LEVELS, which stops at the state
# because that is the ladder producer 57 climbs. It belongs here anyway: it is
# the rung the WS4 local projection weights on, so leaving it out would mean
# the one weight already in use on this project had no row in the table of
# weights. At this rung the weight groups and the absorbed division-by-year
# effect are the same partition, which is worth seeing rather than assuming.
RUNGS = ("county",) + tuple(locked.LEVELS) + ("division",)
PRETTY = {"county": "County", "czone": "Commuting zone", "msa": "MSA",
          "cea": "CEA", "ea": "EA", "state": "State",
          "division": "Census division"}
FAILURES: list[str] = []


class Tee:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, v):
        self.file.write(v)
        self.stdout.write(v)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  {detail}" if detail
                                                      else ""))
    if not ok:
        FAILURES.append(label)


def weight_vectors(g, rung):
    """(om, P0j-mapped) for one rung, on one horizon's frame.

    At rung 'county' the group is the county-year itself, so om is identically
    one and P0j is the county's own POP_BASE_YEAR population. Written as the
    same groupby rather than special-cased, so the collapse is a property of
    the arithmetic and not of a branch.
    """
    key = "ccg" if rung == "county" else rung
    om = g["E_b"] / g.groupby([key, "year"])["E_b"].transform("sum")
    P0j = g.drop_duplicates("ccg").groupby(key)["popw"].sum()
    return om.to_numpy(np.float64), g[key].map(P0j).to_numpy(np.float64)


def fit(g, w):
    """The basic specification: reduced form, division-by-year, county clusters."""
    import pyfixest as pf
    d = g.assign(_w=w / np.mean(w))
    m = pf.feols(f"{OUTCOME} ~ z | divyear", d, weights="_w",
                 vcov={"CRV1": "ccg"}, fixef_rm="singleton")
    return (float(m.coef()["z"]) * locked.SCALE,
            float(m.se()["z"]) * locked.SCALE, int(m._N))


def stars(t):
    """House markup for the slide tables: $^{***}$, and nothing when |t| is
    below 1.645. `\\sym` is an estout macro and is not defined in this deck."""
    a = abs(t)
    s = "***" if a >= 2.576 else "**" if a >= 1.96 else "*" if a >= 1.645 else ""
    return f"$^{{{s}}}$" if s else ""


def render(out, hs, ungated):
    """One slide fragment, house pattern: adjustbox, provenance line, two
    column blocks with $c_j=P_j$ first, numeric columns right-aligned.

    Column blocks are the two constants, rows are the rungs, and a row reads as
    one sentence: this is what the coefficient becomes when regions of this size
    are the thing counted once.
    """
    w = out.set_index(["rung", "weight", "horizon"])
    nh = len(hs)
    L = ["% Slide fragment. Rendered by 40_weighting_ladder.py from "
         "weighting_ladder.csv. Do not edit.",
         r"\adjustbox{max width=0.88\linewidth, max totalheight=0.58\textheight}{",
         r"\begin{tabular}{l" + "r" * (2 * nh) + "r}", r"\toprule",
         (" & \\multicolumn{%d}{c}{$c_j=P_j$} & \\multicolumn{%d}{c}{$c_j=1$}"
          " & \\\\" % (nh, nh)),
         (r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}"
          % (nh + 1, nh + 2, 2 * nh + 1)),
         (" & " + " & ".join(f"$h={h}$" for h in hs) + " & "
          + " & ".join(f"$h={h}$" for h in hs) + r" & Units \\"),
         r"\midrule"]
    for rung in RUNGS:
        cells, ses = [], []
        for cj in ("P_j", "EW"):
            for h in hs:
                r = w.loc[(rung, cj, h)]
                cells.append(f"{r['beta']:.3f}{stars(r['t'])}")
                ses.append(f"({r['se']:.3f})")
        n_units = int(w.loc[(rung, "EW", hs[0])]["n_units"])
        L.append(f"{PRETTY[rung]} & " + " & ".join(cells)
                 + f" & {n_units:,} \\\\")
        L.append(" & " + " & ".join(ses) + r" & \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    TEX_OUT.write_text("\n".join(L) + "\n")
    print(f"tex -> {TEX_OUT}")


def horizon_list(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=None,
                    help="default: every horizon present in analysis_panel")
    return ap.parse_args(argv).horizons


def main(horizons=None):
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    tee = Tee(LOG)
    try:
        panel = pd.read_parquet(PIPE / "analysis_panel.parquet")
        have = sorted(panel["h"].unique())
        hs = tuple(have) if horizons is None else tuple(sorted(set(horizons)))
        missing = [h for h in hs if h not in have]
        if missing:
            raise SystemExit(
                f"  FAIL: analysis_panel has no h={missing}. Build it first: "
                f"03 and 04 with --horizons {' '.join(map(str, sorted(set(have) | set(hs))))}")
        ungated = [h for h in hs if h not in locked.HORIZONS]
        print(f"40_weighting_ladder: outcome {OUTCOME}, horizons {hs}, "
              f"rungs {RUNGS}")
        if ungated:
            print(f"  NOTE: h={ungated} is outside locked.HORIZONS. Those rows "
                  f"carry no golden fixture and no frozen coefficient; they are "
                  f"reported as ungated in the table note.")

        rows = []
        for h in hs:
            d = panel[panel["h"] == h].copy()
            d["divyear"] = (d["division"].astype(str) + "_"
                            + d["year"].astype(str))
            for rung in RUNGS:
                g = d if rung == "county" else d[d[rung].notna()]
                g = g.copy()
                om, P0j = weight_vectors(g, rung)
                if rung == "county":
                    check(float(np.abs(om - 1.0).max()) < 1e-12,
                          f"om == 1 at the county rung (h={h})",
                          f"max |om - 1| = {np.abs(om - 1.0).max():.2e}")
                for cj, w in (("EW", om), ("P_j", om * P0j)):
                    b, se, n = fit(g, w)
                    rows.append(dict(horizon=h, rung=rung, weight=cj,
                                     beta=b, se=se, t=b / se if se else np.nan,
                                     n_fit=n, n_frame=len(g),
                                     n_units=int(g["ccg"].nunique() if
                                                 rung == "county"
                                                 else g[rung].nunique()),
                                     gated=h in locked.HORIZONS))
                    print(f"  h={h} {rung:<6} {cj:<4} beta={b:8.4f} "
                          f"({se:.4f})  t={b / se:5.2f}  n={n:,}")

        out = pd.DataFrame(rows)
        out.to_csv(CSV_OUT, index=False)
        print(f"\n{len(out)} cells -> {CSV_OUT}")
        render(out, hs, ungated)

        # The county row under EW is the unweighted county regression and under
        # P_j the popw-weighted one, which is the whole of the collapse claim in
        # the docstring. Assert the weights themselves rather than a coefficient
        # from another producer built on another frame.
        d4 = panel[panel["h"] == max(hs)].copy()
        om, P0j = weight_vectors(d4, "county")
        check(np.allclose(om * P0j, d4["popw"].to_numpy(np.float64), rtol=0,
                          atol=0),
              "at the county rung the P_j weight IS popw, exactly")

        print(f"\ndone in {time.time() - t0:.1f}s"
              + (f"; {len(FAILURES)} FAILURES: {FAILURES}" if FAILURES
                 else "; all checks pass"))
    finally:
        tee.close()
    if FAILURES:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    main(horizon_list())
