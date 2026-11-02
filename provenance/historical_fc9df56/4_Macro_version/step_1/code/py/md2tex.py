#!/usr/bin/env python3
"""Typeset a formula memo (.md) as LaTeX, matching PLAN_ra_next_tasks.tex.

The markdown is the document of record; this only renders it. Handles the
subset the memos actually use: ATX headings, booktabs-style pipe tables,
fenced code blocks, bullets, bold, inline code, rules, and the 30 non-ASCII
glyphs the memos carry.

Usage: md2tex.py IN.md OUT.tex "Title" "Subtitle" "Date"
"""
import re
import sys

# Unicode -> LaTeX, applied to running text only (never inside verbatim).
UNI = {
    "\u2212": "$-$", "\u2014": "---", "\u2013": "--", "\u2192": "$\\rightarrow$",
    "\u00a7": "\\S", "\u00d7": "$\\times$", "\u2032": "$'$",
    "\u2265": "$\\geq$", "\u2264": "$\\leq$", "\u2248": "$\\approx$",
    "\u2208": "$\\in$", "\u00b1": "$\\pm$", "\u00b7": "$\\cdot$",
    "\u221a": "$\\sqrt{\\,}$", "\u2026": "\\ldots",
    "\u03a3": "$\\Sigma$", "\u03b2": "$\\beta$", "\u03c1": "$\\rho$",
    "\u03b3": "$\\gamma$", "\u03bb": "$\\lambda$", "\u03b5": "$\\varepsilon$",
    "\u03b4": "$\\delta$", "\u03c6": "$\\varphi$",
    "\u2081": "$_1$", "\u2082": "$_2$", "\u2083": "$_3$", "\u2075": "$^5$",
    "\u00f8": "\\o{}", "\u00fc": "\\\"u",
}
# Combining marks: attach to the preceding letter.
COMB = {"\u0304": "\\bar", "\u0303": "\\tilde"}

SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s):
    """Escape LaTeX specials, then map unicode. Backslash first."""
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in SPECIAL.items():
        s = s.replace(a, b)
    for mark, cmd in COMB.items():
        s = re.sub(r"(\w)" + mark, lambda m, c=cmd: f"${c}{{{m.group(1)}}}$", s)
    for a, b in UNI.items():
        s = s.replace(a, b)
    return s


def inline(s):
    """Inline code, bold, italic. Code is escaped and shielded from bold."""
    slots = []

    def stash(m):
        slots.append(m.group(1))
        return f"\x00{len(slots) - 1}\x00"

    s = re.sub(r"`([^`]+)`", stash, s)
    s = esc(s)
    # DOTALL: emphasis in the memos routinely spans the source line break, and
    # paragraphs reach here already joined.
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s, flags=re.S)
    s = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"\\textit{\1}", s, flags=re.S)
    # Straight quotes both render as closing quotes in TeX; pair them up.
    s = re.sub(r'"([^"]*)"', r"``\1''", s, flags=re.S)
    for i, raw in enumerate(slots):
        s = s.replace(f"\x00{i}\x00", r"\code{" + code_esc(raw) + "}")
    return s


def code_esc(s):
    """Escape a code span and let it break: identifiers here are long enough
    (paths, `g_{c,t-k->t-1}`, `37_disagg_baseline.py:115`) that an unbreakable
    \\texttt box is what pushes lines past the margin."""
    s = esc(s)
    return re.sub(r"([_/.:\-])", r"\1\\allowbreak{}", s)


def table(rows):
    """Pipe table -> booktabs tabular. Row 2 is the alignment separator."""
    def cells(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]

    head, body = cells(rows[0]), [cells(r) for r in rows[2:]]
    n = len(head)
    body = [(b + [""] * n)[:n] for b in body]

    # Column widths. A table of short cells sets naturally on 'l' columns; one
    # with prose cells overruns the margin, so every column becomes a p{} whose
    # width is its share of the widest content. 88 chars is the measured point
    # at \small where a natural table stops fitting \textwidth.
    span = [max(len(r[c]) for r in [head] + body) for c in range(n)]
    if sum(span) > 88:
        tot = float(sum(span))
        spec = "".join(f"p{{{0.96 * s / tot:.3f}\\textwidth}}" for s in span)
    else:
        spec = "l" * n

    out = [r"\begin{center}", r"\small",
           r"\begin{tabular}{@{}" + spec + r"@{}}", r"\toprule",
           " & ".join(inline(c) for c in head) + r" \\", r"\midrule"]
    for b in body:
        out.append(" & ".join(inline(c) for c in b) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return out


def convert(md):
    lines = md.split("\n")
    out, i, in_list = [], 0, False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(r"\end{itemize}")
            in_list = False

    while i < len(lines):
        ln = lines[i]

        if ln.startswith("```"):
            close_list()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            # No breaklines= here: this TinyTeX's fancyvrb predates that key,
            # and no code block in the memos exceeds the measure anyway.
            out += [r"\begin{Verbatim}[fontsize=\small, xleftmargin=1em]",
                    *buf, r"\end{Verbatim}"]
            continue

        if ln.strip().startswith("|"):
            close_list()
            buf = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                buf.append(lines[i])
                i += 1
            if len(buf) >= 2:
                out += table(buf)
            continue

        if ln.startswith("# "):
            close_list()
            i += 1
            continue  # the H1 becomes \title
        if ln.startswith("### "):
            close_list()
            out.append(r"\subsection*{" + inline(ln[4:]) + "}")
            i += 1
            continue
        if ln.startswith("## "):
            close_list()
            out.append(r"\section*{" + inline(ln[3:]) + "}")
            i += 1
            continue

        if ln.strip() == "---":
            close_list()
            out.append(r"\medskip\hrule\medskip")
            i += 1
            continue

        m = re.match(r"^(\s*)[-*] (.*)$", ln)
        if m:
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            # Gather the item's continuation lines so emphasis spanning them
            # is still one span by the time inline() sees it.
            buf, i = [m.group(2)], i + 1
            while (i < len(lines) and lines[i].strip()
                   and not re.match(r"^\s*[-*] ", lines[i])
                   and not lines[i].startswith(("#", "```", "|"))
                   and lines[i].strip() != "---"):
                buf.append(lines[i].strip())
                i += 1
            out.append(r"\item " + inline(" ".join(buf)))
            continue

        if not ln.strip():
            close_list()
            out.append("")
            i += 1
            continue

        # A plain paragraph: join its wrapped lines before converting.
        buf = []
        while (i < len(lines) and lines[i].strip()
               and not re.match(r"^\s*[-*] ", lines[i])
               and not lines[i].startswith(("#", "```", "|"))
               and lines[i].strip() != "---"):
            buf.append(lines[i].strip())
            i += 1
        out.append(inline(" ".join(buf)))

    close_list()
    return "\n".join(out)


PRE = r"""%% Typeset rendering of %(src)s (the markdown file is the
%% document of record; regenerate this PDF after editing it).
%% Generated by md2tex.py -- do not hand-edit.
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{fontspec}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{fancyvrb}
\usepackage[colorlinks=true, linkcolor=blue!50!black, urlcolor=blue!50!black]{hyperref}
\setmonofont{Menlo}[Scale=0.85]
%% parskip, enumitem and microtype are not in this TinyTeX tree; the two
%% settings this memo actually used from them are inlined instead.
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55\baselineskip}
\setlength{\itemsep}{2pt}
\emergencystretch=3em
\newcommand{\code}[1]{\texttt{\small #1}}

\title{\vspace{-1.2cm}%(title)s}
\author{%(author)s}
\date{%(date)s}

\begin{document}
\maketitle
\vspace{-0.8cm}

"""


def main():
    src, dst, title, author, date = sys.argv[1:6]
    md = open(src, encoding="utf-8").read()
    body = convert(md)
    head = PRE % dict(src=src.split("/")[-1], title=title,
                      author=author, date=date)
    open(dst, "w", encoding="utf-8").write(head + body + "\n\n\\end{document}\n")
    print(f"wrote {dst}: {len(body.splitlines())} body lines")


if __name__ == "__main__":
    main()
