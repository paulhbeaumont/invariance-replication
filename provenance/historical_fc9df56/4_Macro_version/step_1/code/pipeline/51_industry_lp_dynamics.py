"""WS4's lead-lag local projection, re-run on WS1's NAICS-4 industry axis.

Approved in Results/lp/lp_formula_memo.md section 9, written before this file
existed. Producer 50 is the WS4 battery on the locked FSC composition
instrument; producer 28 already carries 22's STATIC battery onto the industry
axis. What has never run there is the DYNAMIC battery -- the lead families, the
joint lead Wald, the increment cross-check, the effective-df reading. 22's
Panel E pre-trend and 50's joint lead test are different objects (Panel E
regresses pre-period growth on z at h = 2/3/4; 50 regresses forward-dated lead
growth on z1 at h = 1 with p = 3 augmentation) and are nested nowhere.

THE QUESTION: does the lead behaviour survive relabelling the shock from FSC
categories onto industries, or is it a property of the FSC categorisation?

THIS IS A WRAPPER, NOT A FORK. Producer 50 is already shock-agnostic below
load_stores(): build_window, run_cell, horizon_fit, shock_frame, joint_test,
increment_lead_test, effective_df, validate_house and first_stage all consume
one S dict and never rebuild an instrument. So this file swaps TWO keys of that
dict and calls 50's own functions unchanged, which is the relationship 28 has to
22. Nothing in 50 is edited, nothing in Results/lp/ is rewritten, and locked.py
is not touched.

    S["gk"]       (cats x cy) the national shock on the LOCKED FSC axis
    S["Z1_full"]  (units x ty) the county instrument built from it

Everything else in S -- units, cats, cy, arr, piv, DIVI, sh, C -- stays the
locked object. That is the whole point: the sample, the weights, the fixed
effects and the cluster levels are literally 50's, so only the shock moves.

WHY THE SHOCK IS RESTATED ONTO THE FSC AXIS (memo section 9.3). Industry cells do
not partition FSC categories, so there is no parent lookup:

    z_c = sum_j cbar_cj g_j = sum_k cbar_ck g~_k ,   g~_k = sum_j share_kj g_j

The diagnostics run on the RESTATED instrument, not the natively built one.
This is load-bearing, not convenient. horizon_fit asserts that the county route
and the BHJ shock-level route agree to locked.RTOL_COEF, and that assert is the
check that the aggregation is right; it can only hold for the instrument the
(exposure, shock) decomposition actually describes. Feeding the native z_ind
against the locked exposure tensor would break it wherever the restatement is
inexact. Running on z^r makes it exact by construction and keeps 800
well-defined clusters.

Because the restated instrument is built by handing g~_k to the SAME
lib/instrument.predicted_dollars call 50 makes, with the SAME locked tensor and
the SAME locked SPAN/LAG/CLIP/DATA_FIRST, the window arithmetic cannot drift
from 50's. Only gk differs.

THE FY2007 SEAM, AND WHY IT NEARLY VANISHES AT h = 1 (memo section 9.4).
Producer 23's era rule uses the concordance for FY1966-2006 and the record's own
naics_code for FY2007-18, so the restatement is exact only where the exposure
window closes by CONC_LAST_YEAR = 2006. At h = 1 that window closes at t - 6,
three years earlier than h = 4's t - 9, so it is exact for t <= 2012: 31 of the
33 target years on the headline window, against 8.1% of cells inexact at h = 4.
NO SAMPLE IS RESTRICTED -- dropping the seam years would be a sample move on an
approved estimand -- and the gap is measured per target year into
lp_industry_restatement.csv rather than papered over.

SCOPE, as approved (memo section 9.5). Two industry columns, z_ind_prob and
z_ind_modal_pooled, with the locked z retained as a reference that is
RE-ESTIMATED here rather than quoted, because a reference that came off disk
would prove nothing about this wrapper. Rungs 0' and 0. All five horizon
families. Rung 5' is NOT run. One figure.

HOW TO READ THE RESULT, pre-committed (memo section 9.6). Section 7 of the memo
already establishes that the locked p = 0.0088 is the sharpest cell of its own
24-cell grid and fails under a change to the weight, the front year or the
depth. So an industry column that does not reject is NOT "the axis fixes the
pre-trend", and one that does is not confirmation. The legitimate comparison is
like-for-like across the grid, which is why the grid runs for all three columns.

Outputs, all to Results/lp_industry/ (a sibling; Results/lp/ is never touched):
    lp_industry_dynamics.csv        one row per column x rung x kind x horizon
    lp_industry_summary.csv         the joint tests, per column x rung
    lp_industry_restatement.csv     the per-target-year seam audit
    lp_industry_pretrend_grid.csv   the 24-cell grid, per column
    Figure_lp_irf_industry.{pdf,tex}
    Table_lp_industry_leads.tex
    Table_lp_industry_pretrend.tex
"""
import argparse
import importlib
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
import instrument as inst  # noqa: E402
import industry as ind  # noqa: E402

# 50 is the battery. Its module name leads with a digit, so it is reached
# through importlib rather than a bare import -- 28's route to 22.
p50 = importlib.import_module("50_lp_dynamics")

RESULTS = PIPELINE.parents[2] / "Results" / "lp_industry"
LOCKED_LP = PIPELINE.parents[2] / "Results" / "lp"

H = p50.H                      # 1, the shock's own differencing horizon
SCALE = locked.SCALE
Z95 = p50.Z95

# 28's tolerance, on the concordance-era target years only, as a share of the
# mean |z|. The seam years are measured and reported, never asserted away.
EXACT_RTOL = 1e-3

# The reference column is re-estimated, not read. The two industry columns are
# 28's, and carry 28's keys so the two producers' CSVs join on `column`.
COLUMNS = (
    ("z",                  "FSC axis (locked)",           None),
    ("z_ind_prob",         "Industry axis, probabilistic", "prob"),
    ("z_ind_modal_pooled", "Industry axis, modal",         "modal"),
)
ALLOC_COL = {"prob": "dollars_prob", "modal": "dollars_modal"}

RUNGS = ("0'", "0")            # rung 5' is out of scope (memo section 9.5)


def log(m):
    print(m, flush=True)


# ------------------------------------------------- the industry shock at h=1 --

def industry_gk(allocation, cats):
    """(g~_k on the locked FSC axis, the native g_j, the cells) at h = 1.

    25_industry_instrument.py restricts --h to locked.HORIZONS = (2,3,4) and
    persists no h = 1 object, so this is built in memory exactly as 50 builds
    z1 -- and producer 25 is neither re-run nor modified. Every constant is
    inherited from the lock; the only thing that moves relative to 50 is the
    axis of k.

    The share matrix must match the allocation the shock was built from. Passing
    the prob matrix against a modal shock would silently restate the wrong
    object, so the two are drawn from one argument and never from two.
    """
    units, cells, cy, arr = ind.load_axis_tensor(ALLOC_COL[allocation])
    cells, (arr,), sent = ind.drop_sentinels(cells, arr)
    nat = pd.DataFrame(arr.sum(axis=0).astype(np.float64), index=cells,
                       columns=cy)
    g_ind = inst.category_rates(nat, H)          # locked CLIP
    S = ind.load_share_matrix(cats, cells, allocation)
    gt = ind.restate_to_fsc(g_ind, S)
    log(f"    {allocation}: {len(cells)} cells, sentinel share "
        f"{sent * 100:.4f}%, restated onto {gt.shape[0]} FSC categories")
    return gt, g_ind, cells, cy, arr, units


def column_store(S, gk_new, label):
    """A copy of 50's S with the shock swapped and NOTHING else.

    Z1_full is rebuilt by handing the new gk to the SAME predicted_dollars call
    50 makes, against the SAME locked tensor -- so the window rule, the roll
    lag, the span and the DATA_FIRST floor cannot drift from 50's by
    construction, and z_c = sum_k cbar_ck g~_k / P_{c,t-1} holds exactly. That
    identity is what horizon_fit's equiv_gap assert tests.
    """
    ty = sorted(S["zpos"], key=S["zpos"].get)
    L, _ = inst.predicted_dollars(S["arr"], gk_new, S["cy"], ty, H, "roll",
                                  locked.SPAN, locked.LAG, locked.DATA_FIRST)
    P_lag = S["piv"]["P"].reindex(columns=[t - H for t in ty]).to_numpy(np.float64)
    Z1 = L / P_lag

    fin = np.isfinite(Z1)
    ref = np.isfinite(S["Z1_full"])
    fy = np.array(ty)[fin.any(0)]
    z1_first = int(fy.min())
    log(f"    {label}: z1 finite {int(fin.sum()):,} cells, "
        f"years {z1_first}-{int(fy.max())}")

    # MEASURED, not inherited (memo section 9.5). 50's Z1_FINITE and its
    # YEAR_LO = z1_first + P_LAG arithmetic are properties of the locked z1.
    # A different finite support would put the industry column on a different
    # window, and a p-value compared across different windows is meaningless --
    # so this stops rather than silently shifting.
    if not np.array_equal(fin, ref):
        raise SystemExit(
            f"  FAIL: {label} is finite on a different set of (county, year) "
            f"cells than the locked z1 ({int(fin.sum()):,} vs "
            f"{int(ref.sum()):,}). Every column must run on one rectangle for "
            "the comparison to mean anything; fix the axis coverage before "
            "reading a coefficient.")
    if z1_first != S["z1_first"]:
        raise SystemExit(
            f"  FAIL: {label}'s first finite target year is {z1_first} against "
            f"the locked instrument's {S['z1_first']}, so the common window "
            "would differ between columns. Do NOT shift it silently.")

    out = dict(S)
    out["gk"] = gk_new
    out["Z1_full"] = Z1
    return out, Z1


# --------------------------------------------- the restatement, measured -----

def restatement_report(column, S, gt, g_ind, arr_ind, cy_ind, Zr):
    """Per target year: does sum_k cbar_ck g~_k reproduce the native industry z?

    The file that makes the FY2007 seam auditable, in the shape of 28's
    industry_fsc_restatement.csv. Zr is the restated instrument the diagnostics
    actually run on; Z0 is the instrument the industry axis builds natively at
    the same constants. Where producer 23's era rule sources the exposure window
    from the concordance the two are the SAME NUMBER.
    """
    ty = sorted(S["zpos"], key=S["zpos"].get)
    L0, _ = inst.predicted_dollars(arr_ind, g_ind, cy_ind, ty, H, "roll",
                                   locked.SPAN, locked.LAG, locked.DATA_FIRST)
    P_lag = S["piv"]["P"].reindex(columns=[t - H for t in ty]).to_numpy(np.float64)
    Z0 = L0 / P_lag

    ok = np.isfinite(Zr) & np.isfinite(Z0)
    sc = float(np.nanmean(np.abs(Z0[ok])))
    rel = np.abs(Zr - Z0) / sc

    rows, exact = [], np.zeros(len(ty), bool)
    for j, t in enumerate(ty):
        m = ok[:, j]
        if not m.any():
            continue
        hi = t - H - locked.LAG
        lo = hi - locked.SPAN + 1
        era = ("concordance" if hi <= ind.CONC_LAST_YEAR
               else "native" if lo > ind.CONC_LAST_YEAR else "seam")
        exact[j] = era == "concordance"
        rows.append(dict(column=column, h=H, target_year=t, window_lo=lo,
                         window_hi=hi, era=era, n=int(m.sum()),
                         corr=float(np.corrcoef(Zr[m, j], Z0[m, j])[0, 1]),
                         max_rel_gap=float(rel[m, j].max()),
                         mean_rel_gap=float(rel[m, j].mean())))
    me = ok & exact[None, :]
    worst = float(rel[me].max()) if me.any() else np.nan
    if not (worst < EXACT_RTOL):
        raise SystemExit(
            f"  FAIL: {column}: the FSC restatement is not exact on the "
            f"concordance-era target years (worst {worst:.2e} of mean |z|); "
            "the shock-level machinery is not running on the instrument it "
            "claims to")
    n_seam = int((ok & ~exact[None, :]).sum())
    log(f"    restatement exact on {int(me.sum()):,} of {int(ok.sum()):,} cells "
        f"(worst {worst:.1e} of mean |z|); {n_seam:,} cells on seam/native "
        f"years ({sum(1 for r in rows if r['era'] != 'concordance')} of "
        f"{len(rows)} target years)")
    return rows


# ----------------------------------------------------------------- anchors ---

def anchor_locked(cell):
    """The re-estimated locked column against the frozen Results/lp/lp_summary.csv.

    THIS IS THE CHECK THAT PROVES THE WRAPPER INTRODUCED NOTHING. The locked
    column runs through exactly the code path the industry columns run through,
    so if it reproduces 50's published joint lead test then every difference the
    industry columns show is the shock and not this file. Compared to the file
    on disk, never to a constant here: a constant would be this producer
    agreeing with itself.
    """
    csv = LOCKED_LP / "lp_summary.csv"
    if not csv.exists():
        log(f"  ANCHOR SKIPPED: no {csv}. The locked column has not been tied "
            "to the published exhibit; no industry number should be read.")
        return
    s = pd.read_csv(csv)
    s = s[(s["sample"] == "common") & (s["rung"] == cell["rung"])]
    if s.empty:
        log(f"  ANCHOR SKIPPED: no common/rung {cell['rung']} row in {csv.name}.")
        return
    r = s.iloc[0]
    gap = abs(cell["jl"]["F"] - float(r["joint_lead_F"]))
    rel = gap / max(abs(float(r["joint_lead_F"])), 1e-300)
    log(f"  ANCHOR rung {cell['rung']}: joint lead F {cell['jl']['F']:.6f} vs "
        f"the frozen {float(r['joint_lead_F']):.6f}  (rel gap {rel:.1e})")
    if rel >= locked.RTOL_COEF:
        raise SystemExit(
            "  FAIL: the re-estimated locked column does not reproduce "
            f"{csv.name}. This wrapper is not faithful to producer 50 and no "
            "industry column may be read from this run.")


def anchor_industry(grid_cells, headline, column):
    """Each industry column's (division-EW, p = P_LAG, own) grid cell against
    that column's own rung-0' headline from this run.

    50's _anchor_check ties the same cell to the frozen lp_summary.csv; no such
    frozen anchor exists for an industry column, so the in-run headline is the
    reference. Absent, the check is SKIPPED loudly rather than passed silently.
    """
    a = next((c for c in grid_cells
              if c["weight"] == "division-EW" and c["p"] == p50.P_LAG
              and c["sample"] == "own"), None)
    if a is None or headline is None:
        log(f"  ANCHOR SKIPPED for {column}: no (division-EW, p = "
            f"{p50.P_LAG}, own) cell or no headline to tie it to.")
        return
    rel = abs(a["jl"]["F"] - headline["jl"]["F"]) / \
        max(abs(headline["jl"]["F"]), 1e-300)
    log(f"  ANCHOR {column}: grid (division-EW, p = {p50.P_LAG}, own) F "
        f"{a['jl']['F']:.6f} vs headline {headline['jl']['F']:.6f} "
        f"(rel gap {rel:.1e})")
    if rel >= locked.RTOL_COEF:
        raise SystemExit(
            f"  FAIL: {column}'s grid does not reproduce its own headline; the "
            "grid is measuring something other than what the table reports.")


# ------------------------------------------------------------------ writers --

def dynamics_rows(column, label, cell):
    return [dict(column=column, column_label=label, **r)
            for r in p50.dynamics_rows(cell)]


def summary_rows(column, label, cell, scope):
    r = p50.summary_row(cell, scope)
    F_eff, p_eff, df_eff = p50.effective_df(
        cell["jl"]["F"], cell["jl"]["k"], cell["jl"]["G"],
        cell["eff_all"]["category"])
    return dict(column=column, column_label=label,
                F_eff=F_eff, p_value_eff=p_eff, df_eff=df_eff, **r)


def grid_rows(column, label, cells):
    """The 24-cell grid, in 50's grid schema plus `column`."""
    out = []
    for c in cells:
        specs = [("cumulative", "leads k = 2,3,4, windows [t-k, t-1]", c["jl"]),
                 ("increment", "disjoint one-year windows [t-j-1, t-j]", c["il"]),
                 ("literal", "overlapping leads, windows [t-k, t]", c["jlit"])]
        for param, note, t in specs:
            if t is None:
                continue
            F_eff, p_eff, df_eff = p50.effective_df(
                t["F"], t["k"], t["G"], c["eff_all"]["category"])
            row = dict(column=column, column_label=label, weight=c["weight"],
                       p=c["p"], sample=c["sample"], rung=c["rung"],
                       param=param, param_note=note, year_lo=c["year_lo"],
                       year_hi=c["year_hi"], n=c["n"], k=t["k"], F=t["F"],
                       p_value=t["p"], G=t["G"],
                       G_eff=c["eff_all"]["category"], df_eff=df_eff,
                       F_eff=F_eff, p_value_eff=p_eff, maxcorr=t["maxcorr"],
                       condV=t["cond"], first_stage_F=c["fsF"])
            for i in range(3):
                row[f"b{i + 1}"] = t["betas"][i] * SCALE
                row[f"se_exp{i + 1}"] = t["ses"][i] * SCALE
            row["n_controls"] = c["ncontrol"]
            out.append(row)
    return out


def labels_short(key):
    """The column label, trimmed to what a note sentence can carry."""
    return {"z": "FSC", "z_ind_prob": "industry/prob",
            "z_ind_modal_pooled": "industry/modal"}.get(key, key)


def stars(b, se):
    if not np.isfinite(se) or se <= 0:
        return ""
    t = abs(b / se)
    return "***" if t > 2.576 else "**" if t > 1.96 else "*" if t > 1.645 else ""


def _p(v):
    return "$<0.001$" if v < 0.001 else f"{v:.4f}"


def write_leads_table(cells, out):
    """Table_lp_industry_leads.tex: the joint lead verdict per column x rung.

    Notes are generated from the run, never typed, so a stale figure is
    impossible (30's convention).
    """
    cols = [c for c in COLUMNS if any(x["column"] == c[0] for x in cells)]
    keys = [c[0] for c in cols]
    n = len(keys)
    by = {(x["column"], x["rung"]): x for x in cells}

    L = [r"% Produced by step_1/code/pipeline/51_industry_lp_dynamics.py.",
         r"% Do not edit.",
         r"\begin{tabular}{@{}l" + "c" * n + r"@{}}", r"\toprule",
         " & " + " & ".join(rf"\textbf{{{c[1]}}}" for c in cols) + r" \\",
         r"\midrule"]

    for rung in RUNGS:
        present = [k for k in keys if (k, rung) in by]
        if not present:
            continue
        tag = ("reported baseline" if rung == "0'"
               else "diagnostic, NOT admissible")
        L.append(rf"\multicolumn{{{n + 1}}}{{@{{}}l}}{{\textit{{Rung {rung} "
                 rf"({tag})}}}} \\")
        for k in (2, 3, 4):
            cells_txt = []
            for key in keys:
                c = by.get((key, rung))
                if c is None:
                    cells_txt.append("")
                    continue
                f = next(x for x in c["cell"]["fits"]
                         if x["kind"] == "lead" and x["hz"] == k)
                b, se = f["beta"] * SCALE, f["se_exposure"] * SCALE
                cells_txt.append(rf"{b:+.4f}{stars(b, se)}")
            L.append(rf"\quad lead $k={k}$ & " + " & ".join(cells_txt) + r" \\")
            se_txt = []
            for key in keys:
                c = by.get((key, rung))
                if c is None:
                    se_txt.append("")
                    continue
                f = next(x for x in c["cell"]["fits"]
                         if x["kind"] == "lead" and x["hz"] == k)
                se_txt.append(rf"({f['se_exposure'] * SCALE:.4f})")
            L.append(" & " + " & ".join(se_txt) + r" \\")

        def row(lbl, fn):
            v = []
            for key in keys:
                c = by.get((key, rung))
                v.append("" if c is None else fn(c))
            L.append(rf"\quad {lbl} & " + " & ".join(v) + r" \\")

        row("Joint lead $F$ (3, $G-1$)", lambda c: f"{c['cell']['jl']['F']:.3f}")
        row("Joint lead $p$", lambda c: _p(c["cell"]["jl"]["p"]))
        row("$p$ at effective df", lambda c: _p(c["p_eff"])
            if np.isfinite(c["p_eff"]) else "")
        row("Increment-parameterised $p$",
            lambda c: _p(c["cell"]["il"]["p"]))
        row("Overlapping-lead $p$", lambda c: _p(c["cell"]["jlit"]["p"]))
        row("First-stage $F$ ($x_1$ on $z1$)",
            lambda c: f"{c['cell']['fsF']:.2f}")
        row("Effective shifters, category",
            lambda c: f"{c['cell']['eff_all']['category']:.2f}")
        # G is NOT the same across columns and must not be left to the note:
        # the 18 locked FSCs absent from the concordance restate to an all-zero
        # row, hence to a NaN shock, so the industry columns run on 782 live
        # categories rather than 800. Printed so the df of the F above is
        # readable off the table.
        row("Shock clusters $G$", lambda c: f"{c['cell']['jl']['G']:,}")
        row("$n$", lambda c: f"{c['cell']['n']:,}")
        L.append(r"\addlinespace")

    L += [r"\bottomrule", r"\end{tabular}"]
    out.with_suffix(".tex").write_text("\n".join(L) + "\n")
    print(f"  wrote {out.with_suffix('.tex')}")

    ns = sorted({x["cell"]["n"] for x in cells})
    gs = ", ".join(
        f"{labels_short(k)} $G = {by[(k, RUNGS[0])]['cell']['jl']['G']:,}$"
        for k in keys if (k, RUNGS[0]) in by)
    note = rf"""% Produced by step_1/code/pipeline/51_industry_lp_dynamics.py.
% Do not edit.
\footnotesize
\textbf{{What this is.}} WS4's county lead-lag local projection
(\texttt{{50\_lp\_dynamics.py}}), re-run with the shock relabelled onto WS1's
NAICS-4 industry axis. Each coefficient is $\beta$ on the one-year composition
instrument $z1_{{ct}}$ from a separate reduced-form regression of county midpoint
employment growth over one window, scaled by $10^{{5}}$; leads are the
non-overlapping windows $g_{{c,t-k \to t-1}}$. Lag augmentation is $p =
{p50.P_LAG}$. Fixed effects are division$\times$year, weights are $E_{{c,t-1}}$
normalised within division-year. Standard errors in parentheses are
exposure-robust, computed on the equivalent shock-level regression (Borusyak,
Hull and Jaravel 2022, Section 5.1) and clustered on the FSC category.
Stars at 1\%, 5\%, 10\%.
\smallskip
\textbf{{Only the shock moves.}} The industry columns restate the shock back onto
the locked FSC exposure tensor, $\tilde{{g}}_k = \sum_j \text{{share}}_{{kj}}
g_j$, so the sample, the weights, the fixed effects and the cluster levels are
the locked objects. All columns run on $n = {ns[0]:,}$ county-years; that row is
a check, not a result. The concordance relabels existing FSC dollars and creates
no new shocks, so inference stays clustered at the FSC category. The cluster
count nonetheless differs by column ({gs}): the 18 locked categories absent from
the concordance -- namespace-era codes outside its FY2001--06 estimation window
-- restate to an all-zero row and hence to an undefined shock, so they leave the
shock-level sum. Their dollars are exactly the excluded sentinel, so this is the
right restatement and not a silent drop, but it is why the reference column's $F$
is read against 799 denominator degrees of freedom and the industry columns'
against 781.
\smallskip
\textbf{{Effective df.}} $p$ at effective df reads the same statistic against
$1/\text{{HHI}}$ effective shifters rather than the literal category count. It is
reported alongside the literal figure, never substituted; the gap between them is
the size of the question, not a correction.
\smallskip
\textbf{{Read this against the grid, not on its own.}} On the locked instrument
the headline $p$ is the sharpest cell of the 24-cell pre-trend grid
(Table~\ref{{tab:lp_industry_pretrend}}) and fails under a change to the weight,
the front year or the lag depth. A column that does not reject here is therefore
not evidence that the axis removes the pre-trend, and one that does is not
confirmation. Rung 0 drops the share-sum control and is not admissible under
\texttt{{ws0\_contingency.py}}; it is shown as a diagnostic. Note the small-sample
bias of short-panel local projections (Herbst and Johannsen).
"""
    out.with_name(out.name + "_notes").with_suffix(".tex").write_text(note)
    print(f"  wrote {out.with_name(out.name + '_notes').with_suffix('.tex')}")


def write_pretrend_table(rows, out):
    """Table_lp_industry_pretrend.tex: the grid's joint lead F and p per column.

    Panel A is the fixed window (identical rows at every depth, so a movement in
    F is the lag order and nothing else -- the panel to read); Panel B is each
    depth on its own longest window. Within each, the nested leads then the
    disjoint increments. Columns are the instruments; the weight splits the
    table in two, since it is now the third dimension.
    """
    df = pd.DataFrame(rows)
    keys = [c[0] for c in COLUMNS if c[0] in set(df["column"])]
    labels = {c[0]: c[1] for c in COLUMNS}
    n = len(keys)

    L = [r"% Produced by step_1/code/pipeline/51_industry_lp_dynamics.py",
         r"% --pretrend-grid. Do not edit.",
         r"\begin{tabular}{@{}l" + "cc" * n + r"@{}}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{labels[k]}}}"
                            for k in keys) + r" \\"]
    L.append("".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                     for i in range(n)))
    L.append(" & " + " & ".join([r"$F$ & $p$"] * n) + r" \\")
    L.append(r"\midrule")

    for weight in p50.WEIGHTS:
        L.append(rf"\multicolumn{{{2 * n + 1}}}{{@{{}}l}}{{\textbf{{Weight: "
                 rf"{weight}}}}} \\")
        for mode, ptitle in (("fixed", "One window at every depth"),
                             ("own", "Each depth on its own longest window")):
            sub = df[(df["weight"] == weight) & (df["sample"] == mode)]
            if sub.empty:
                continue
            L.append(rf"\addlinespace")
            L.append(rf"\multicolumn{{{2 * n + 1}}}{{@{{}}l}}{{\textit{{"
                     rf"{ptitle}}}}} \\")
            for param, plabel in (("cumulative", r"nested leads $k=2,3,4$"),
                                  ("increment",
                                   r"disjoint increments $j=1,2,3$")):
                s2 = sub[sub["param"] == param]
                if s2.empty:
                    continue
                L.append(rf"\multicolumn{{{2 * n + 1}}}{{@{{}}l}}{{\quad "
                         rf"\textit{{{plabel}}}}} \\")
                for p in p50.P_GRID:
                    vals = []
                    lo = hi = nn = None
                    for k in keys:
                        r = s2[(s2["column"] == k) & (s2["p"] == p)]
                        if r.empty:
                            vals += ["", ""]
                            continue
                        r = r.iloc[0]
                        lo, hi, nn = int(r["year_lo"]), int(r["year_hi"]), \
                            int(r["n"])
                        vals += [f"{r['F']:.3f}", _p(float(r["p_value"]))]
                    lab = (rf"\quad $p={p}$" if mode == "fixed"
                           else rf"\quad $p={p}$, $t \geq {lo}$ ({nn:,})")
                    L.append(lab + " & " + " & ".join(vals) + r" \\")
        L.append(r"\addlinespace")

    L += [r"\bottomrule", r"\end{tabular}"]
    out.with_suffix(".tex").write_text("\n".join(L) + "\n")
    print(f"  wrote {out.with_suffix('.tex')}")


def write_figure(by_column, out):
    """Figure_lp_irf_industry.pdf plus its .tex fragment.

    Rung 0' only, common window. Three series on one panel: the locked reference
    and the two industry columns, points with 95% exposure-robust intervals,
    x-jittered so the intervals do not overplot. Rung 0 is estimated and
    reported in the tables but gets no figure (memo section 9.5).
    """
    style = {"z": ("#7a7a7a", "o", "-", 0.0),
             "z_ind_prob": ("#1f3b73", "o", "-", -0.08),
             "z_ind_modal_pooled": ("#b4472f", "o", "--", +0.08)}
    labels = {c[0]: c[1] for c in COLUMNS}

    plt, fig, ax = p50._fig_open()
    for key, _, _ in COLUMNS:
        if key not in by_column:
            continue
        fits = by_column[key]["cell"]["fits"]
        colr, mk, ls, dx = style[key]
        pts = sorted(((p50.XPOS[(f["kind"], f["hz"])], f) for f in fits
                      if (f["kind"], f["hz"]) in p50.XPOS), key=lambda t: t[0])
        x = np.array([p for p, _ in pts]) + dx
        b = np.array([f["beta"] for _, f in pts]) * SCALE
        se = np.array([f["se_exposure"] for _, f in pts]) * SCALE
        # The gap at x = -1 is a missing horizon, not a value: the
        # non-overlapping lead at k = 1 is a zero-length window. Draw the two
        # segments separately rather than joining across it.
        for seg in (x < 0, x >= 0):
            ax.plot(x[seg], b[seg], color=colr, ls=ls, lw=1.0, zorder=3)
        ax.errorbar(x, b, yerr=Z95 * se, fmt="none", ecolor=colr,
                    elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
        ax.scatter(x, b, s=16, color=colr, marker=mk, zorder=4,
                   label=labels[key])
    pdf = out.with_suffix(".pdf")
    p50._fig_close(plt, fig, ax, pdf)

    lines = []
    for key, lab, _ in COLUMNS:
        if key not in by_column:
            continue
        c = by_column[key]["cell"]
        lines.append(rf"{lab}: joint lead $F = {c['jl']['F']:.3f}$, "
                     rf"$p = {c['jl']['p']:.4f}$")
    seam = by_column.get("z_ind_prob", {}).get("n_seam_years")
    n_ty = by_column.get("z_ind_prob", {}).get("n_target_years")
    seam_txt = ("" if seam is None else
                rf" The FSC restatement is exact on {n_ty - seam} of {n_ty} "
                rf"target years on this window; the {seam} seam years are "
                r"measured in \texttt{lp\_industry\_restatement.csv} and no "
                r"sample is restricted.")
    frag = rf"""% Produced by step_1/code/pipeline/51_industry_lp_dynamics.py from
% Results/lp_industry/lp_industry_dynamics.csv. Do not edit.
\begin{{minipage}}{{\textwidth}}\centering
\textbf{{Figure. Lead and lag dynamics, FSC axis against the industry axis}}\\[0.2em]
{{\small County midpoint employment growth on the one-year composition
instrument, by horizon and by the axis the shock is categorised on}}\\[0.6em]
\includegraphics[width=\textwidth]{{{pdf.name}}}
\end{{minipage}}
\begin{{minipage}}{{\textwidth}}\vspace{{0.4em}}\footnotesize
\textbf{{What is plotted.}} Each point is $\beta$ from a separate reduced-form
regression of county midpoint employment growth over one window on the one-year
composition instrument $z1_{{ct}}$, scaled by $10^{{5}}$, at rung 0' on the common
window $[{p50.YEAR_LO}, {p50.YEAR_HI}]$. Bars are 95\% intervals from the
exposure-robust standard error, clustered on the FSC category. The shaded strip
is the shock year $[t-1,t]$. Negative horizons are the non-overlapping lead
placebos $g_{{c,t-k \to t-1}}$, $k = 2,3,4$; horizon 0 is $g_{{c,t-1 \to t}}$;
positive horizons are $g_{{c,t \to t+h}}$. There is no horizon $-1$: the
non-overlapping lead at $k=1$ is a zero-length window. Series are nudged
horizontally so their intervals do not overplot.
\smallskip
\textbf{{Only the shock moves.}} The industry series restate the shock onto the
locked FSC exposure tensor, so the sample ($n = {by_column['z']['cell']['n']:,}$
county-years), the weights, the fixed effects and the cluster levels are
identical across the three series.{seam_txt}
\smallskip
\textbf{{Joint lead tests.}} {"; ".join(lines)}. Lag augmentation is
$p = {p50.P_LAG}$, so the leads test pre-trends rather than shock persistence;
note the small-sample bias of short-panel local projections (Herbst and
Johannsen). These are single cells of a grid: on the locked instrument the
headline $p$ is the sharpest of the 24 cells in
\texttt{{Table\_lp\_industry\_pretrend.tex}} and does not survive a change to the
weight, the front year or the depth, so no series here should be read as passing
or failing a pre-trend test on its own.
\end{{minipage}}
"""
    out.with_suffix(".tex").write_text(frag)
    print(f"  wrote {out.with_suffix('.tex')}")


# -------------------------------------------------------------------- main --

def run_grid(S_col, column, label, headline):
    """50's 24-cell pre-trend grid on one column, at GRID_RUNG.

    The cell loop is 50's pretrend_grid verbatim in structure; what cannot be
    reused is its anchor, which ties the (division-EW, p, own) cell to the
    frozen lp_summary.csv. See anchor_industry.
    """
    lo_fixed = S_col["z1_first"] + max(p50.P_GRID)
    cells = []
    for weight in p50.WEIGHTS:
        for mode in ("fixed", "own"):
            for p in p50.P_GRID:
                lo = lo_fixed if mode == "fixed" else S_col["z1_first"] + p
                cells.append(p50.run_cell(
                    S_col, lo, p50.YEAR_HI, p50.LEAD_KINDS, p50.GRID_RUNG,
                    mode, f"[{column}] pre-trend grid, weight {weight}, "
                    f"p = {p}, {mode} window", p=p, weight=weight))
    if column == "z":
        # The locked column's grid has a frozen anchor of its own; use 50's.
        p50._anchor_check(cells)
    else:
        anchor_industry(cells, headline, column)
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--column", action="append",
                    choices=[c[0] for c in COLUMNS],
                    help="restrict to one or more instrument columns "
                         "(default: all three, the locked reference included -- "
                         "it is the check that this wrapper is faithful)")
    ap.add_argument("--rung", choices=("0'", "0", "both"), default="both",
                    help="0' is the reported baseline; 0 drops the share-sum "
                         "control and is NOT admissible under "
                         "ws0_contingency.py -- a diagnostic. Rung 5' is out "
                         "of scope here (lp_formula_memo.md section 9.5)")
    ap.add_argument("--pretrend-grid", action="store_true",
                    help="run ONLY the 24-cell grid (weight x lag depth x "
                         "window) per column and write "
                         "lp_industry_pretrend_grid.csv and "
                         "Table_lp_industry_pretrend.tex, and NOTHING else")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the shocks, run every construction check and "
                         "the restatement audit, then stop before estimating")
    a = ap.parse_args()
    t0 = time.time()

    want = tuple(c for c in COLUMNS
                 if a.column is None or c[0] in set(a.column))
    rungs = RUNGS if a.rung == "both" else (a.rung,)

    log(f"== stage 51: WS4 lead-lag battery on the industry axis, h = {H}, "
        f"p = {p50.P_LAG}")
    log(f"   columns {[c[0] for c in want]}, rungs {list(rungs)}, "
        f"common window [{p50.YEAR_LO}, {p50.YEAR_HI}]")
    log("   producer 50 is imported and called unchanged; only S['gk'] and "
        "S['Z1_full'] are swapped")

    # ONE store load for every column. This also runs 50's own locked-instrument
    # asserts (Z1_FINITE, SAMPLE_LAST, the front constraint) before anything
    # here touches the industry axis.
    log("\n-- stores (locked; 50's own asserts run here)")
    S = p50.load_stores()

    log("\n-- the industry shocks at h = 1, restated onto the FSC axis")
    stores, restate = {}, []
    for key, label, alloc in want:
        if alloc is None:
            stores[key] = (S, None)
            continue
        gt, g_ind, cells, cy_ind, arr_ind, units_ind = industry_gk(
            alloc, S["cats"])
        assert np.array_equal(units_ind, S["units"]), \
            "the industry tensor's county order differs from the locked one"
        S_col, Zr = column_store(S, gt, label)
        rows = restatement_report(key, S, gt, g_ind, arr_ind, cy_ind, Zr)
        restate += rows
        stores[key] = (S_col, rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    if restate:
        csv = RESULTS / "lp_industry_restatement.csv"
        pd.DataFrame(restate).to_csv(csv, index=False)
        print(f"  wrote {csv}  ({len(restate)} rows)")

    if a.dry_run:
        log(f"\n  dry run: construction and restatement checks passed in "
            f"{time.time() - t0:.1f}s; nothing estimated, no exhibit written")
        return

    # ---- the grid is a separate run, and writes only its own two files ------
    if a.pretrend_grid:
        log("\n== PRE-TREND GRID only: the headline exhibits are NOT rewritten")
        rows = []
        for key, label, _ in want:
            S_col, _ = stores[key]
            log(f"\n-- {label}: headline cell first, as the grid's anchor")
            head = p50.run_cell(S_col, p50.YEAR_LO, p50.YEAR_HI,
                                p50.LEAD_KINDS, p50.GRID_RUNG, "common",
                                f"[{key}] anchor, rung {p50.GRID_RUNG}")
            cells = run_grid(S_col, key, label, head)
            rows += grid_rows(key, label, cells)
        csv = RESULTS / "lp_industry_pretrend_grid.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"\n  wrote {csv}  ({len(rows)} rows)")
        write_pretrend_table(rows, RESULTS / "Table_lp_industry_pretrend")
        log(f"  done in {time.time() - t0:.1f}s")
        return

    # ---- the headline battery ----------------------------------------------
    dyn, summ, table_rows, by_column = [], [], [], {}
    for key, label, _ in want:
        S_col, rrows = stores[key]
        for rung in rungs:
            log(f"\n-- {label}, rung {rung}")
            cell = p50.run_cell(S_col, p50.YEAR_LO, p50.YEAR_HI, p50.ALL_KINDS,
                                rung, "common",
                                f"[{key}] common sample, rung {rung}")
            F_eff, p_eff, df_eff = p50.effective_df(
                cell["jl"]["F"], cell["jl"]["k"], cell["jl"]["G"],
                cell["eff_all"]["category"])
            rec = dict(column=key, label=label, rung=rung, cell=cell,
                       p_eff=p_eff)
            table_rows.append(rec)
            dyn += dynamics_rows(key, label, cell)
            summ.append(summary_rows(key, label, cell,
                                     "all horizons on one common sample"))
            if key == "z":
                anchor_locked(cell)
            if rung == "0'":
                by_column[key] = dict(
                    cell=cell,
                    n_seam_years=(None if rrows is None else
                                  sum(1 for r in rrows
                                      if r["era"] != "concordance"
                                      and p50.YEAR_LO <= r["target_year"]
                                      <= p50.YEAR_HI)),
                    n_target_years=(None if rrows is None else
                                    sum(1 for r in rrows
                                        if p50.YEAR_LO <= r["target_year"]
                                        <= p50.YEAR_HI)))

    # Same shock, same rectangle: a differing n means the sample moved.
    ns = {r["cell"]["n"] for r in table_rows}
    if len(ns) != 1:
        raise SystemExit(
            f"  FAIL: columns ran on different samples {sorted(ns)}. Only the "
            "shock is supposed to move; a differing n voids every comparison.")

    log("")
    pd.DataFrame(dyn).to_csv(RESULTS / "lp_industry_dynamics.csv", index=False)
    print(f"  wrote {RESULTS / 'lp_industry_dynamics.csv'}  ({len(dyn)} rows)")
    pd.DataFrame(summ).to_csv(RESULTS / "lp_industry_summary.csv", index=False)
    print(f"  wrote {RESULTS / 'lp_industry_summary.csv'}  ({len(summ)} rows)")
    write_leads_table(table_rows, RESULTS / "Table_lp_industry_leads")
    if by_column:
        write_figure(by_column, RESULTS / "Figure_lp_irf_industry")

    log(f"\n  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
