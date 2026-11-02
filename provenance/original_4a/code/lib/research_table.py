"""Complete regression tabular renderer for saved estimates in house style.

Labels are trusted LaTeX. This module formats results; it never estimates them.
"""
import json
import math


def row(label, values):
    return label + " & " + " & ".join(values) + r" \\" + "\n"


def render_table(dependent_variable, columns, panels, fixed_effects, no_fe_reason="", column_groups=None, group_label="Level", column_label=None):
    """Render panels with coefficients and statistics, with explicit FE metadata.

    coefficients: [{label, estimates, standard_errors, pvalues}]
    statistics: [{label, values, digits}] with digits=0 for integer counts.
    fixed_effects: {LaTeX label: [bool per column]}.
    """
    n = len(columns)
    if not n or not panels or (not fixed_effects and not no_fe_reason.strip()):
        raise ValueError("Columns, panels and explicit FE metadata are required")
    contract = {"kind": "regression", "fixed_effects": list(fixed_effects)}
    if not fixed_effects:
        contract["no_fe_reason"] = no_fe_reason
    out = [r"\begin{tabular}{l" + "c" * n + "}\n",
           "% qa-table: " + json.dumps(contract) + "\n", r"\toprule" + "\n",
           row(r"\textit{Dependent variable}", [r"\multicolumn{" + str(n) + "}{c}{" + dependent_variable + "}"]),
           r"\cmidrule(r){2-" + str(n + 1) + "}\n"]
    if column_groups:
        if any(type(span) is not int or span < 1 for _, span in column_groups) or sum(span for _, span in column_groups) != n:
            raise ValueError("Column groups must span exactly the model columns")
        out.append(row(r"\textit{" + group_label + "}", [r"\multicolumn{" + str(span) + "}{c}{" + label + "}" for label, span in column_groups]))
        start = 2
        for _, span in column_groups:
            out.append(r"\cmidrule(lr){" + str(start) + "-" + str(start + span - 1) + "}")
            start += span
        out.append("\n")
    out.extend([row(r"\textit{" + (column_label or ("Weights" if column_groups else "Specification")) + "}", columns),
                "".join(r"\cmidrule(lr){" + str(j) + "-" + str(j) + "}" for j in range(2, n + 2)) + "\n",
                row("", [f"({i + 1})" for i in range(n)])])
    for panel in panels:
        out.append(r"\midrule\addlinespace" + "\n")
        if panel.get("title"):
            out.append(r"\multicolumn{" + str(n + 1) + r"}{l}{" + (r"\textbf{" if panel.get("bold_title") else r"\textit{") + panel["title"] + r"}} \\" + "\n")
        if panel.get("space_after_title"):
            out.append(r"\addlinespace" + "\n")
        for coef in panel["coefficients"]:
            b, se, p = (coef[k] for k in ("estimates", "standard_errors", "pvalues"))
            if any(len(v) != n for v in (b, se, p)):
                raise ValueError("Coefficient width differs from column count")
            if not all(math.isfinite(x) for v in (b, se, p) for x in v) or any(x < 0 for x in se) or any(not 0 <= x <= 1 for x in p):
                raise ValueError("Invalid coefficient, standard error or p-value")
            stars = ["***" if x < .01 else "**" if x < .05 else "*" if x < .1 else "" for x in p]
            out.append(row(coef["label"], [f"{x:.3f}" + ("$^{" + s + "}$" if s else "") for x, s in zip(b, stars)]))
            out.append(row("", [f"({x:.3f})" for x in se]))
            if "percent_deviations" in coef:
                deviations = coef["percent_deviations"]
                if len(deviations) != n or not all(math.isfinite(x) for x in deviations):
                    raise ValueError("Percentage deviations must be finite and match the model columns")
                out.append(row(r"\textit{\% deviation}", ["[" + (f"{x:+.1f}" if abs(x) >= .05 else "0.0") + r"\%]" for x in deviations]))
        if fixed_effects:
            out.extend([r"\midrule\addlinespace" + "\n", r"\textit{Fixed Effects} \\" + "\n"])
            for label, flags in fixed_effects.items():
                if len(flags) != n or any(type(x) is not bool for x in flags) or not any(flags):
                    raise ValueError("FE indicators must match columns and include at least one fitted FE")
                out.append(row(r"\hspace{1.5em}" + label, [r"$\checkmark$" if flag else "--" for flag in flags]))
        out.append(r"\addlinespace" + "\n")
        for stat in panel["statistics"]:
            values, digits = stat["values"], stat["digits"]
            if len(values) != n or not isinstance(digits, int) or digits < 0 or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid statistic width, precision or values")
            if digits == 0 and any(v != int(v) for v in values):
                raise ValueError("Integer statistics cannot be silently rounded")
            out.append(row(stat["label"], [f"{v:,.{digits}f}" for v in values]))
    out.extend([r"\bottomrule" + "\n", r"\end{tabular}" + "\n"])
    return "".join(out)
