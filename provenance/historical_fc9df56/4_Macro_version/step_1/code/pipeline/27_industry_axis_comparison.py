"""WS1 step 5: the FSC-axis against the industry-axis instrument, one table.

Estimation cell is IDENTICAL to 20/21/22, so the only thing that moves between rows
is which instrument column enters: x winsorized 1/99 within year (the diagnostics
convention, locked.WINSOR_DIAG_GROUP), division-by-year fixed effects, weights E_b
normalised within division-year, clustered by ccg, outcome y_mpgr. The estimator is
28_table_d1.fit2sls, reached through lib/oldpath exactly as 21 and 22 reach it.

FOUR INSTRUMENTS, and the middle two are what make the comparison readable:

  z             the locked FSC-axis instrument, pooled national shock
  z_ind_noloo   the industry axis, pooled national shock   <- ONE change from z
  z_ind         the industry axis, leave-one-out shock     <- the WS1 headline
  z_ind_modal   the industry axis, LOO, modal allocation   <- allocation robustness

WS1 step 2 asks for the leave-one-out instrument as the headline, and the locked FSC
build is not leave-one-out. So the headline row differs from the locked row in TWO
things at once, which B.3.1 of INVARIANCE_TEST_PROTOCOL.md forbids attributing to
either. z_ind_noloo is the row that splits the bundle: z -> z_ind_noloo is the axis
change alone, and z_ind_noloo -> z_ind is the LOO change alone. The table note says
so, per B.3.2.

This is a robustness and interpretation exercise, not a fresh identification claim -
the concordance relabels FSC shocks across industries and creates no new ones, so
the effective shock count remains the FSC count.

Outputs: Results/industry_axis/Table_industry_axis.tex
         Results/industry_axis/coeff_industry_axis.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE.parent / "py"))
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "lib"))
import dcads_common as dc  # noqa: E402
import locked  # noqa: E402
import panel as pnl_lib  # noqa: E402
import oldpath as op  # noqa: E402  (estimator only: t28/w22, as 21 and 22 load them)

IN = dc.MACRO_DATA_ROOT / "temp" / "pipeline_industry"
RESULTS = PIPELINE.parents[2] / "Results" / "industry_axis"

INSTRUMENTS = [
    ("z", "FSC axis (locked)", "pooled"),
    ("z_ind_noloo", "Industry axis, pooled shock", "pooled"),
    ("z_ind", "Industry axis, leave-one-out", "LOO"),
    ("z_ind_modal", "Industry axis, LOO, modal", "LOO"),
]


def stars(b, se):
    if not (np.isfinite(b) and np.isfinite(se) and se > 0):
        return ""
    t = abs(b / se)
    return "***" if t > 2.576 else "**" if t > 1.96 else "*" if t > 1.645 else ""


def estimate(d, zcol):
    """One 2SLS fit in the 20/21/22 estimation cell."""
    xw = pnl_lib.wins(d["x_pos"].to_numpy(np.float64), d["year"].to_numpy())
    tot = d.groupby(["division", "year"])["E_b"].transform("sum")
    w = (d["E_b"] / tot).to_numpy(np.float64)
    w = w / w.mean()
    fe = pd.factorize(d["division"].astype(str) + "_"
                      + d["year"].astype(str))[0].astype(np.int64)
    nofe = np.zeros(len(d), np.int64)
    cl = pd.factorize(d["ccg"])[0].astype(np.int64)
    y0 = d[locked.OUTCOME].to_numpy(np.float64)
    z0 = d[zcol].to_numpy(np.float64)
    iv = op.t28.fit2sls(y0, xw, z0, w, fe, nofe, cl, singleton=False)
    # the reduced form: the outcome on the instrument directly, same cell
    rf = op.t28.fit2sls(y0, z0, z0, w, fe, nofe, cl, singleton=False)
    return iv, rf


def main():
    t0 = time.time()
    import pyfixest as pf
    op.w22.DEMEANER = pf.MapDemeaner(fixef_tol=1e-10, fixef_maxiter=100_000)

    ap = pd.read_parquet(IN / "analysis_panel_industry.parquet")
    rows, corr_rows = [], []

    for h in locked.HORIZONS:
        d = ap[ap["h"] == h].reset_index(drop=True)
        assert len(d) == locked.N_COUNTY[h], (h, len(d))
        for zcol, label, shock in INSTRUMENTS:
            iv, rf = estimate(d, zcol)
            rows.append(dict(h=h, instrument=zcol, label=label, shock=shock,
                             beta=iv["beta"], se=iv["se"], F=iv["F"], n=iv["n"],
                             rf_beta=rf["beta"], rf_se=rf["se"]))
            print(f"h={h} {zcol:>12}: beta={iv['beta']:+.4f} ({iv['se']:.4f}) "
                  f"F={iv['F']:.1f}  RF={rf['beta']:+.6f} ({rf['se']:.6f})",
                  flush=True)
        cols = [c for c, _, _ in INSTRUMENTS]
        C = d[cols].corr()
        for a in cols:
            for b in cols:
                corr_rows.append(dict(h=h, a=a, b=b, corr=float(C.loc[a, b])))

    res = pd.DataFrame(rows)
    # Coefficients are reported at locked.SCALE, the convention 21 and 57 print in
    # (21:336). The CSV carries both, so nothing downstream has to guess.
    S = locked.SCALE
    res["beta_scaled"] = res["beta"] * S
    res["se_scaled"] = res["se"] * S
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "coeff_industry_axis.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(RESULTS / "corr_industry_axis.csv", index=False)

    # ---- LaTeX
    H = list(locked.HORIZONS)
    L = []
    L.append(r"\begin{tabular}{l" + "c" * len(H) + "}")
    L.append(r"\toprule")
    L.append(" & " + " & ".join(rf"$h={h}$" for h in H) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\textit{Panel A. 2SLS, $y^{\mathrm{mpgr}}$ on "
             r"$x$ (coefficients $\times 10^{5}$)}} \\" % (len(H) + 1))
    for zcol, label, _ in INSTRUMENTS:
        b, s = [], []
        for h in H:
            r = res[(res.h == h) & (res.instrument == zcol)].iloc[0]
            b.append(f"{r.beta_scaled:.4f}{stars(r.beta, r.se)}")
            s.append(f"({r.se_scaled:.4f})")
        L.append(f"{label} & " + " & ".join(b) + r" \\")
        L.append(" & " + " & ".join(s) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\textit{Panel B. First-stage $F$}} \\"
             % (len(H) + 1))
    for zcol, label, _ in INSTRUMENTS:
        v = [f"{res[(res.h == h) & (res.instrument == zcol)].iloc[0].F:.1f}"
             for h in H]
        L.append(f"{label} & " + " & ".join(v) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\textit{Panel C. Reduced form, "
             r"$y^{\mathrm{mpgr}}$ on the instrument (coefficients $\times 10^{5}$)}}"
             r" \\" % (len(H) + 1))
    for zcol, label, _ in INSTRUMENTS:
        b, s = [], []
        for h in H:
            r = res[(res.h == h) & (res.instrument == zcol)].iloc[0]
            b.append(f"{r.rf_beta * S:.4f}{stars(r.rf_beta, r.rf_se)}")
            s.append(f"({r.rf_se * S:.4f})")
        L.append(f"{label} & " + " & ".join(b) + r" \\")
        L.append(" & " + " & ".join(s) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\textit{Panel D. Correlation with the locked "
             r"FSC instrument}} \\" % (len(H) + 1))
    cdf = pd.DataFrame(corr_rows)
    for zcol, label, _ in INSTRUMENTS[1:]:
        v = [f"{cdf[(cdf.h == h) & (cdf.a == 'z') & (cdf.b == zcol)].iloc[0]['corr']:.4f}"
             for h in H]
        L.append(f"{label} & " + " & ".join(v) + r" \\")
    L.append(r"\midrule")
    L.append("Observations & "
             + " & ".join(f"{locked.N_COUNTY[h]:,}" for h in H) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")

    # The note enumerates the bundle, per INVARIANCE_TEST_PROTOCOL.md B.3.2.
    note = [
        r"\begin{minipage}{\textwidth}\footnotesize",
        r"\emph{Notes.} County-level 2SLS of midpoint employment growth on contract "
        r"growth over lagged population, instrumented by the composition "
        r"instrument. Estimation cell identical across all rows and identical to the "
        r"shift-share diagnostics: $x$ winsorised at 1/99 within year, "
        r"division-by-year fixed effects, weighted by base employment normalised "
        r"within division-year, standard errors clustered by county. Exposure shares "
        r"are five-year means closing five years before the treatment base, so they "
        r"are predetermined. The FSC-axis row reproduces the locked pipeline "
        rf"exactly ({res[(res.h == 4) & (res.instrument == 'z')].iloc[0].beta_scaled:.6f} "
        r"at $h=4$).",
        "",
        r"\emph{Reading the rows.} Row 2 differs from row 1 in \emph{one} thing, the "
        r"axis: contract dollars are relabelled from FSC/PSC categories onto "
        r"309 four-digit industry cells through the concordance, with the national "
        r"shock pooled in both. Row 3 differs from row 2 in \emph{one} thing, the "
        r"leave-one-out national shock. Row 3 is the headline and therefore differs "
        r"from row 1 in \emph{two} things at once; rows 2 and 3 exist so that the "
        r"axis change and the leave-one-out change can be attributed separately "
        r"rather than blamed on whichever the reader already suspected. Row 4 "
        r"replaces probabilistic allocation through the concordance with "
        r"modal-industry assignment.",
        "",
        r"\emph{What this is and is not.} The concordance \emph{relabels} FSC shocks "
        r"across industries; it does not create new ones. The effective number of "
        r"shocks remains the FSC count, and industries loading on the same FSC "
        r"category carry mechanically correlated shifters. The industry axis is a "
        r"robustness and interpretation exercise, not a fresh identification claim.",
        r"\end{minipage}",
    ]
    (RESULTS / "Table_industry_axis.tex").write_text("\n".join(L) + "\n")
    (RESULTS / "Table_industry_axis_notes.tex").write_text("\n".join(note) + "\n")
    print(f"\nwrote {RESULTS.name}/Table_industry_axis.tex and 2 csv")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
