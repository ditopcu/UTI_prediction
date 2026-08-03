# -*- coding: utf-8 -*-
"""Table 1 as a standalone .docx.
- 3 significant figures, median (IQR) in parentheses; categorical n (%).
- p reported as <0.05.
- Boruta-retained predictors marked with * (footnote).
Recomputed from raw values via RAW_INDEX.
"""
import os
import subprocess
from math import floor, log10

import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANDOC = r"C:\Users\ditop\AppData\Local\Pandoc\pandoc.exe"
TEMPLATE = os.path.join(BASE, "paper", "template", "empty draft with captions.docx")
TARGET = "CULTIVO_PATOLOGICO"

# original variables retained by Boruta (→ marked with *)
RETAINED = {"Age, years", "Sex", "Leukocyte esterase", "Nitrite", "Protein",
            "Blood (Hb peroxidase)", "Specific gravity", "Bacteria, /µL",
            "White blood cells, /µL", "Red blood cells, /µL", "Epithelial cells, /µL",
            "Sysmex Gram flag"}


def sig(x, n=3):
    if pd.isna(x):
        return ""
    if x == 0:
        return "0"
    d = n - 1 - floor(log10(abs(x)))
    return f"{round(x, d):.0f}" if d <= 0 else f"{x:.{d}f}"


raw = pd.read_excel(os.path.join(BASE, "data", "01_raw", "uti_raw.xlsx")); raw["RAW_INDEX"] = raw.index + 1
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
keep = ["EDAD", "SEXO", "LEUT", "NITT", "PROTT", "HEMATT", "GLUT", "PHT", "DENST",
        "RBO", "WBCO", "EC", "BACTS", "CASTS", "YLC", "BACT_INFO"]
c = ml[["RAW_INDEX", TARGET]].merge(raw[["RAW_INDEX"] + keep], on="RAW_INDEX", how="left")
c["SEXO"] = c["SEXO"].map({"H": "Male", "M": "Female"})
for col in ["LEUT", "NITT", "PROTT", "HEMATT", "GLUT"]:
    c[col] = c[col].astype(str).replace({"Negativo": "Negative", "Positivo": "Positive"})
def gram(x):
    s = str(x).lower()
    if "sin informaci" in s: return "No information"
    if "negativo" in s and ("positivo" in s or "gran positivo" in s): return "Mixed"
    if "positivo" in s: return "Gram-positive"
    if "negativo" in s: return "Gram-negative"
    return "No information"
c["GRAM"] = c["BACT_INFO"].map(gram)
neg, pos = c[c[TARGET] == 0], c[c[TARGET] == 1]
COLS = ["Characteristic", f"Culture-negative (n={len(neg)})",
        f"Culture-positive (n={len(pos)})", "p-value"]
rows = []

def star(label):
    return label + " *" if label in RETAINED else label

def cont(label, col):
    p = stats.mannwhitneyu(neg[col].dropna(), pos[col].dropna()).pvalue
    def mi(s):
        return f"{sig(s.median())} ({sig(s.quantile(.25))}-{sig(s.quantile(.75))})"
    rows.append([star(label), mi(neg[col]), mi(pos[col]), "<0.05" if p < 0.05 else f"{p:.2f}"])

def cat(label, col, order):
    ct = pd.crosstab(c[col], c[TARGET]).reindex(order).fillna(0)
    p = stats.chi2_contingency(ct.values)[1]
    rows.append([star(label), "", "", "<0.05" if p < 0.05 else f"{p:.2f}"])
    for k in order:
        nn, pp = (neg[col] == k).sum(), (pos[col] == k).sum()
        rows.append([f" {k}",
                     f"{nn} ({nn/len(neg)*100:.1f}%)", f"{pp} ({pp/len(pos)*100:.1f}%)", ""])

# Demographics
cont("Age, years", "EDAD")
cat("Sex", "SEXO", ["Male", "Female"])
# Dipstick
cat("Leukocyte esterase", "LEUT", ["Negative", "25", "75", "500"])
cat("Nitrite", "NITT", ["Negative", "Positive"])
cat("Protein", "PROTT", ["Negative", "15", "30", "100", "300", "1000"])
cat("Blood (Hb peroxidase)", "HEMATT", ["Negative", "10", "20", "50", "250"])
cat("Glucose", "GLUT", ["Negative", "Positive"])
cont("pH", "PHT")
cont("Specific gravity", "DENST")
# Flow cytometry
cont("Bacteria, /µL", "BACTS")
cont("White blood cells, /µL", "WBCO")
cont("Red blood cells, /µL", "RBO")
cont("Epithelial cells, /µL", "EC")
cont("Hyaline casts, /µL", "CASTS")
cont("Yeasts, /µL", "YLC")
cat("Sysmex Gram flag", "GRAM", ["Gram-negative", "Gram-positive", "Mixed", "No information"])

df = pd.DataFrame(rows, columns=COLS)
md = ["# Table 1", "", "| " + " | ".join(COLS) + " |", "|" + "|".join(["---"] * len(COLS)) + "|"]
for _, r in df.iterrows():
    md.append("| " + " | ".join(str(x) for x in r) + " |")
md.append(
    "\n\n**Table 1.** Characteristics of the study population, by urine-culture result. "
    "Continuous variables are reported as median (interquartile range); categorical variables as "
    "n (%). p-values are from the Mann-Whitney U test (continuous variables) or the chi-squared "
    "test (categorical variables). All comparisons were significant at p<0.05 except urine "
    "glucose (p=0.09). "
    "\\* Variable retained by Boruta feature selection and used in the final model.\n")
open(os.path.join(BASE, "paper", "_table1.md"), "w", encoding="utf8").write("\n".join(md))
subprocess.run([PANDOC, os.path.join(BASE, "paper", "_table1.md"), "-o",
                os.path.join(BASE, "paper", "UTI_Table1.docx"), f"--reference-doc={TEMPLATE}"], check=True)
print(df.to_string(index=False))
print("\nUTI_Table1.docx written  (rows:", len(df), ")")
