# -*- coding: utf-8 -*-
"""
Builds the main-text tables, assembles the manuscript markdown with the six main-text
display items embedded, and converts it to .docx via pandoc using the house template
for styling. A separate supplementary .docx is produced for everything else.

Main text (6 display items):
  Figure 1  study overview            Table 1  cohort characteristics
  Figure 2  discrimination            Table 2  performance across the four phases
  Figure 3  comparison vs previous
  Figure 4  calibration + SHAP
"""
import os
import re
import subprocess

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "data", "04_results")
PNG = os.path.join(BASE, "figures", "PNG_300DPI")
PAPER = os.path.join(BASE, "paper")
PANDOC = r"C:\Users\ditop\AppData\Local\Pandoc\pandoc.exe"
TEMPLATE = os.path.join(PAPER, "template", "empty draft with captions.docx")


def md_table(df, align=None):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join(align or ["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Table 1 — cohort characteristics
# ---------------------------------------------------------------------------
t1 = pd.read_excel(os.path.join(RES, "table1_cohort.xlsx"), sheet_name="table1")
t1 = t1.rename(columns={"Variable": "Characteristic"})
t1_md = md_table(t1)

# ---------------------------------------------------------------------------
# Table 2 — performance across the four phases
# ---------------------------------------------------------------------------
ci = pd.read_excel(os.path.join(RES, "ci_and_fairness.xlsx"), sheet_name="ci_overall")
h2h = pd.read_excel(os.path.join(RES, "old_model_comparison.xlsx"), sheet_name="head_to_head")
w1 = pd.read_excel(os.path.join(RES, "phase2_wave1.xlsx"), sheet_name="performance")
w1p = pd.read_excel(os.path.join(RES, "phase2_wave1.xlsx"), sheet_name="vs_previous")

def g(df, col, row=0):
    return df.iloc[row][col] if col in df.columns else ""

rows = [
    {"Phase / analysis": "**Phase 1** — hold-out (development)", "Population": "Retrospective",
     "n": int(ci.iloc[0]["N"]), "ROC AUC": g(ci, "ROC AUC"), "Sensitivity": g(ci, "Sensitivity"),
     "Specificity": g(ci, "Specificity"), "MCC": g(ci, "MCC")},
    {"Phase / analysis": "Phase 1 — model, concordant subset", "Population": "Retrospective",
     "n": int(h2h.iloc[0]["N"]), "ROC AUC": f'{h2h.iloc[0]["ROC AUC"]:.3f}',
     "Sensitivity": f'{h2h.iloc[0]["Sensitivity"]:.3f}', "Specificity": f'{h2h.iloc[0]["Specificity"]:.3f}',
     "MCC": f'{h2h.iloc[0]["MCC"]:.3f}'},
    {"Phase / analysis": "Phase 1 — *previous model* (combined decision)", "Population": "Retrospective",
     "n": int(h2h.iloc[1]["N"]), "ROC AUC": "—",
     "Sensitivity": f'{h2h.iloc[1]["Sensitivity"]:.3f}', "Specificity": f'{h2h.iloc[1]["Specificity"]:.3f}',
     "MCC": f'{h2h.iloc[1]["MCC"]:.3f}'},
    {"Phase / analysis": "**Phase 2** — pre-implementation batch", "Population": "Prospective, 5–15 Jun 2026",
     "n": int(w1.iloc[0]["N"]), "ROC AUC": g(w1, "ROC AUC"), "Sensitivity": g(w1, "Sensitivity"),
     "Specificity": g(w1, "Specificity"), "MCC": g(w1, "MCC")},
    {"Phase / analysis": "Phase 2 — model, subset with previous decision",
     "Population": "Prospective", "n": int(w1p.iloc[0]["N"]), "ROC AUC": g(w1p, "ROC AUC"),
     "Sensitivity": g(w1p, "Sensitivity"), "Specificity": g(w1p, "Specificity"), "MCC": g(w1p, "MCC")},
    {"Phase / analysis": "Phase 2 — *previous model*", "Population": "Prospective",
     "n": int(w1p.iloc[1]["N"]), "ROC AUC": "—", "Sensitivity": g(w1p, "Sensitivity", 1),
     "Specificity": g(w1p, "Specificity", 1), "MCC": g(w1p, "MCC", 1)},
    {"Phase / analysis": "**Phase 4** — post-implementation (routine use)",
     "Population": "Prospective, 22 Jun–6 Jul 2026", "n": 72, "ROC AUC": "0.821 (0.71–0.92)",
     "Sensitivity": "0.909", "Specificity": "0.429", "MCC": "0.396"},
]
t2_md = md_table(pd.DataFrame(rows))

with open(os.path.join(PAPER, "tables_main.md"), "w", encoding="utf8") as f:
    f.write("## Table 1\n\n" + t1_md + "\n\n## Table 2\n\n" + t2_md + "\n")
print("tables_main.md written")

# ---------------------------------------------------------------------------
# Assemble the manuscript markdown with the six display items
# ---------------------------------------------------------------------------
src = open(os.path.join(PAPER, "MANUSCRIPT_DRAFT.md"), encoding="utf8").read()
src = re.sub(r"^> .*$", "", src, flags=re.M)
src = src.split("## Figure / table mapping")[0]
src = re.sub(r"^# Manuscript draft.*$", "", src, flags=re.M)

def fig_block(name, number, caption):
    return (f"\n\n![]({PNG}/{name}.png)\n\n"
            f"**Figure {number}.** {caption}\n\n")

# insert display items at their first mention
src = src.replace("# Results", "# Results" + fig_block(
    "figure_12_study_overview", 1,
    "Study overview. The four sequential phases: retrospective model development, "
    "pre-implementation evaluation on new data, integration into the laboratory information "
    "system through the clinical decision-support rule engine, and post-implementation "
    "evaluation in routine use."), 1)

src = src.replace("## 3.2 Phase 1", "### Table 1\n\n" + t1_md +
                  "\n\n**Table 1.** Characteristics of the development cohort, by urine-culture "
                  "result. Continuous variables are median [interquartile range]; categorical "
                  "variables are n (%).\n\n## 3.2 Phase 1", 1)

src = src.replace("### 3.2.2 Probability calibration", fig_block(
    "figure_M2_discrimination", 2,
    "Discrimination on the hold-out set (n=2248). (A) Receiver-operating-characteristic curves and "
    "(B) precision–recall curves for the final model and the automated machine-learning benchmark."
) + "### 3.2.2 Probability calibration", 1)

src = src.replace("## 3.3 Phase 2", fig_block(
    "figure_M3_vs_previous", 3,
    "Comparison with the previously deployed model on the hold-out set. (A) Proportion of samples "
    "receiving a decision. (B) Classification metrics in the concordant subset (n=1248) in which the "
    "previous combined decision was issued. (C) Receiver-operating-characteristic curves, with the "
    "previously deployed operating point shown as a marker."
) + fig_block(
    "figure_M4_calibration_shap", 4,
    "Model behaviour on the hold-out set. (A) Reliability curve of predicted probabilities. "
    "(B) Mean absolute SHAP values, indicating the contribution of each feature."
) + "## 3.3 Phase 2", 1)

src = src.replace("## 3.6 Subgroup", "### Table 2\n\n" + t2_md +
                  "\n\n**Table 2.** Model performance across the four study phases, with the "
                  "previously deployed model shown for the subsets in which it issued a decision. "
                  "Values are point estimates with 95% confidence intervals where computed.\n\n"
                  "## 3.6 Subgroup", 1)

head = """---
title: 'PLACEHOLDER — Title (to be written last)'
---

**Authors:** Deniz Ilhan Topcu, Lucia Puig Chacon, Emilio Flores Pardo

*Affiliations: PLACEHOLDER · Corresponding author: PLACEHOLDER*

# Introduction

PLACEHOLDER — to be written after the Discussion, per the drafting order.

"""
out_md = os.path.join(PAPER, "_manuscript_for_word.md")
open(out_md, "w", encoding="utf8").write(head + src)

subprocess.run([PANDOC, out_md, "-o", os.path.join(PAPER, "UTI_manuscript_draft.docx"),
                f"--reference-doc={TEMPLATE}"], check=True)
print("UTI_manuscript_draft.docx written (4 figures + 2 tables embedded)")

# ---------------------------------------------------------------------------
# Supplementary document
# ---------------------------------------------------------------------------
SUPP = [
    ("figure_03_confusion_matrices", "Confusion matrices on the hold-out set for the final model and the automated benchmark."),
    ("figure_06_correctness_quadrant", "Correctness of the model and the previously deployed decision, relative to culture, in the concordant subset."),
    ("figure_15_phase2_wave1", "Phase 2. (A) Performance against culture in the pre-implementation batch. (B) Locally computed versus web-service probabilities."),
    ("figure_14_implementation_fidelity", "Probabilities reported by the laboratory information system versus probabilities recomputed offline (post-implementation urine samples)."),
    ("figure_11_atb_subgroup_roc", "Discrimination by antibiotic exposure in the hold-out set."),
    ("figure_13_sex_subgroup_roc", "Discrimination by sex in the hold-out set."),
    ("figure_08_shap_beeswarm", "SHAP summary (beeswarm) for the final model on the hold-out set."),
]
s = ["# Supplementary material\n"]
for i, (name, cap) in enumerate(SUPP, 1):
    s.append(f"\n![]({PNG}/{name}.png)\n\n**Supplementary Figure S{i}.** {cap}\n")
s.append("\n# Supplementary tables\n")
for i, (f_, sheet, cap) in enumerate([
        ("table1_cohort.xlsx", "cohort_flow", "Cohort flow from the source export to the analytic cohort."),
        ("model_performance_holdout.xlsx", 0, "Hold-out performance of all developed models."),
        ("calibration.xlsx", 0, "Calibration metrics before and after post-hoc recalibration."),
        ("atb_subgroup.xlsx", 0, "Performance by antibiotic exposure."),
        ("ci_and_fairness.xlsx", "by_sex", "Performance by sex, with 95% confidence intervals."),
        ("implementation_fidelity.xlsx", "summary", "Agreement between reported and recomputed probabilities (post-implementation)."),
        ("retrospective_pipeline_check.xlsx", "summary", "Agreement between the deployment path and the development-time encoding (retrospective)."),
], 1):
    df = pd.read_excel(os.path.join(RES, f_), sheet_name=sheet)
    s.append(f"\n**Supplementary Table S{i}.** {cap}\n\n" + md_table(df) + "\n")
supp_md = os.path.join(PAPER, "_supplement_for_word.md")
open(supp_md, "w", encoding="utf8").write("\n".join(s))
subprocess.run([PANDOC, supp_md, "-o", os.path.join(PAPER, "UTI_supplement.docx"),
                f"--reference-doc={TEMPLATE}"], check=True)
print("UTI_supplement.docx written (7 figures + 7 tables)")
