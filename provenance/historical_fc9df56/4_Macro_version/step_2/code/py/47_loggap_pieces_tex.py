#!/usr/bin/env python3
"""The pieces of the county-to-cell log gap at h=4, each drawn from zero.

One TikZ fragment for a Beamer frame. For each level (NAICS2, NAICS4) there is
one row per piece of the gap, entry, exit, Jensen, renormalisation, and a final
row for their sum. Every bar is anchored at a common zero line, so its sign is
read off the side it sits on and no path has to be followed. Entry and exit sit
on adjacent rows, so an offsetting pair reads as two mirror-image bars, and a
bracket prints their sum.

Signs are county minus cell: a bar to the right of zero is a piece by which the
cell log coefficient falls short of the county's. 35_table_decomposition.py
stores cell minus county, so this script negates every piece.

Run class: graph_only.
Input:  step_2/result/coeff_table_decomposition_baseline.csv (rows h=4, efsy).
Output: Results/Figure_loggap_pieces_baseline_h4.tex, mirrored to Dropbox.

    4_Macro_version/.venv/bin/python 4_Macro_version/step_2/code/py/47_loggap_pieces_tex.py
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from out_paths import write_exhibit  # noqa: E402

RESULT = Path(__file__).resolve().parents[2] / "result"

a = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument("--horizon", type=int, default=4)
a.add_argument("--source", default="efsy", choices=["efsy", "nets_actual", "nets_all"])
g = a.parse_args()

STAG = "" if g.source == "efsy" else f"_{g.source}"
CSV = RESULT / f"coeff_table_decomposition_baseline{STAG}.csv"

# Row order and the colours the deck's legend already uses for these pieces.
STEPS = (("entry", "Entry", "entrybar", "green!60!black"),
         ("exit", "Exit", "exitbar", "red!85!black"),
         ("jensen", "Jensen's", "jbar", "purple"),
         ("renorm", "Renorm", "renormbar", "lightgrey"))
PANELS = (("naics2", r"County $\times$ NAICS2"),
          ("naics4_alloc", r"County $\times$ NAICS4"))

# Geometry, in cm. The x axis maps a piece to v * SC, zero at the origin.
XMIN, XMAX, SC = -0.16, 0.20, 26.0
ROW, BAR = 0.38, 0.22          # row pitch and bar thickness
PANEL_GAP = 0.55
LABEL_W = 0.80                 # width reserved for a \tiny signed value label
LEFT = 1.1                     # room for row labels left of the axis start


def X(v):
    return v * SC


def signed(v):
    return f"{v:+.3f}"


def bar(L, yc, v, style, color, label=None):
    """One bar from zero to v, its value printed just past its far end.
    Returns the bar's right extent including any label to its right."""
    lo, hi = min(0.0, X(v)), max(0.0, X(v))
    L.append(rf"\draw[{style}] ({lo:.3f},{yc - BAR / 2:.3f}) rectangle "
             rf"({hi:.3f},{yc + BAR / 2:.3f});")
    txt = label if label is not None else f"${signed(v)}$"
    if v >= 0:
        L.append(rf"\node[tinylabel, anchor=west, {color}] at ({hi + 0.06:.3f},{yc:.3f}) {{{txt}}};")
        return hi + 0.06 + LABEL_W
    L.append(rf"\node[tinylabel, anchor=east, {color}] at ({lo - 0.06:.3f},{yc:.3f}) {{{txt}}};")
    return hi


def panel(r, y_top):
    """TikZ for one level. Returns (lines, y_bottom)."""
    b_cell = float(r["cell_naive"])
    b_cty = float(r["county"])
    pieces = {k: -float(r[k]) for k, _, _, _ in STEPS}   # county minus cell
    gap = b_cty - b_cell
    L = []
    L.append(rf"\node[lab, anchor=west] at ({X(XMIN) - LEFT:.3f},{y_top:.3f}) "
             rf"{{\textbf{{{r['_label']}}}}};")
    L.append(rf"\node[lab, anchor=east] at ({X(XMAX) + 2.2:.3f},{y_top:.3f}) "
             rf"{{\green{{$\hat\beta^{{c\times j}}={b_cell:.3f}$}}, "
             rf"\blue{{$\hat\beta^{{county}}={b_cty:.3f}$}}}};")
    right_ext = []
    for k, (key, name, style, color) in enumerate(STEPS):
        yc = y_top - ROW * (k + 1)
        L.append(rf"\node[lab, anchor=east, {color}] at ({X(XMIN) - 0.15:.3f},{yc:.3f}) {{{name}}};")
        right_ext.append(bar(L, yc, pieces[key], style, color))
    # The sum, the gap the Results table reports.
    yc = y_top - ROW * (len(STEPS) + 1) - 0.08
    L.append(rf"\node[lab, anchor=east] at ({X(XMIN) - 0.15:.3f},{yc:.3f}) {{\textbf{{Gap}}}};")
    bar(L, yc, gap, "gapbar", "black",
        label=rf"$\hat\beta^{{county}}-\hat\beta^{{c\times j}}={signed(gap)}$")
    # Bracket over the entry and exit rows, with their sum.
    net = pieces["entry"] + pieces["exit"]
    xb = max(right_ext[:2]) + 0.12
    y0 = y_top - ROW * 1 + BAR / 2
    y1 = y_top - ROW * 2 - BAR / 2
    L.append(rf"\draw[decorate, decoration={{brace, amplitude=3pt, mirror}}, grey, thin] "
             rf"({xb:.3f},{y1:.3f}) -- ({xb:.3f},{y0:.3f});")
    L.append(rf"\node[tinylabel, anchor=west, grey] at ({xb + 0.18:.3f},{(y0 + y1) / 2:.3f}) "
             rf"{{entry + exit $= {signed(net)}$}};")
    return L, yc - ROW * 0.6


def main():
    co = pd.read_csv(CSV)
    co = co[(co["h"] == g.horizon) & (co["source"] == g.source)]
    rows = {}
    for rung, label in PANELS:
        sub = co[co["rung"] == rung]
        if len(sub) != 1:
            sys.exit(f"expected one row for h={g.horizon}, rung={rung}, source={g.source}; "
                     f"got {len(sub)}")
        r = sub.iloc[0].to_dict()
        r["_label"] = label
        rows[rung] = r

    stamp = dt.datetime.fromtimestamp(CSV.stat().st_mtime).strftime("%Y-%m-%d")
    out = [
        "% Slide fragment: the pieces of the county-to-cell log gap, each drawn from "
        f"zero, h={g.horizon}, locked baseline, source {g.source}.",
        f"% Rendered by 47_loggap_pieces_tex.py from {CSV.name} (file dated {stamp}).",
        "% Point estimates only: coefficient accounting, no inference. Do not edit.",
        r"\begin{tikzpicture}[",
        r"    axis/.style={-{Stealth[length=2mm]}, semithick, grey},",
        r"    tick/.style={grey, thin},",
        r"    jbar/.style={draw=purple!80, fill=purple!65},",
        r"    entrybar/.style={draw=green!80!black, fill=green!60},",
        r"    exitbar/.style={draw=red!85!black, fill=red!60},",
        r"    renormbar/.style={draw=lightgrey, fill=lightgrey!65},",
        r"    gapbar/.style={draw=black, fill=ultralightgrey!70, line width=0.4pt},",
        r"    zero/.style={grey, thin},",
        r"    lab/.style={font=\scriptsize},",
        r"    tinylabel/.style={font=\tiny, align=center}",
        r"]",
    ]
    y_top = 0.0
    for rung, _ in PANELS:
        L, y_end = panel(rows[rung], y_top)
        out += L
        y_top = y_end - PANEL_GAP
    y_axis = y_top + PANEL_GAP - 0.45
    # One zero line through both panels, then the axis.
    out.append(rf"\draw[zero] (0,{0.0 - ROW * 0.55:.3f}) -- (0,{y_axis:.3f});")
    out.append(rf"\draw[axis] ({X(XMIN):.3f},{y_axis:.3f}) -- ({X(XMAX) + 0.3:.3f},{y_axis:.3f});")
    v = -0.15
    while v <= XMAX + 1e-9:
        out.append(rf"\draw[tick] ({X(v):.3f},{y_axis - 0.07:.3f}) -- ({X(v):.3f},{y_axis + 0.07:.3f});")
        out.append(rf"\node[tinylabel, anchor=north] at ({X(v):.3f},{y_axis - 0.1:.3f}) {{{v:+.2f}}};")
        v += 0.05
    out.append(rf"\node[tinylabel, anchor=north east, grey] at ({X(XMAX) + 0.3:.3f},{y_axis - 0.38:.3f}) "
               r"{contribution to $\hat\beta^{county}-\hat\beta^{c\times j}$};")
    out.append(rf"\path[use as bounding box] ({X(XMIN) - LEFT - 0.1:.3f},{y_axis - 0.75:.3f}) "
               rf"rectangle ({X(XMAX) + 2.3:.3f},0.3);")
    out.append(r"\end{tikzpicture}")

    name = f"Figure_loggap_pieces_baseline{STAG}_h{g.horizon}.tex"
    write_exhibit(name, "\n".join(out) + "\n")
    for rung, _ in PANELS:
        r = rows[rung]
        print(f"  {rung:<12s} gap {float(r['county']) - float(r['cell_naive']):+.4f}"
              + "".join(f"  {k} {-float(r[k]):+.4f}" for k, _, _, _ in STEPS))


if __name__ == "__main__":
    main()
