# -*- coding: utf-8 -*-
"""
§2 Study population — cohort flow + Table 1 (baseline characteristics).

Descriptive statistics only (no modelling). The exact modelled cohort (n=14,985)
is recovered by joining the ML dataset's RAW_INDEX back to the raw export, so
Table 1 uses original-scale, interpretable values (not the encoded/scaled matrix).
Stratified by culture outcome, with p-values (Mann-Whitney U for continuous,
chi-square for categorical).

Outputs: data/04_results/table1_cohort.xlsx  (sheets: cohort_flow, table1)
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "CULTIVO_PATOLOGICO"

# ---------------------------------------------------------------------------
# 1. Cohort flow (reproduce v2 -> ML exclusion steps for the counts)
# ---------------------------------------------------------------------------
raw = pd.read_excel(os.path.join(BASE, "data", "01_raw", "uti_raw.xlsx"))
v2 = pd.read_excel(os.path.join(BASE, "data", "02_interim", "uti_cleaned_v2.xlsx"))

flow = []
flow.append(("Raw records", len(raw)))
flow.append(("Manually cleaned (v2)", len(v2)))
d = v2.copy()
d["NITT"] = d["NITT"].replace("Positivo", 1)
d = d.drop(columns=["FILTER"])
n0 = len(d)
d = d.dropna()
flow.append(("After dropna (remove missing)", len(d)))
d = d.drop_duplicates()
flow.append(("After drop_duplicates", len(d)))
d = d.drop(columns=["FECHA", "XTAL", "UROT", "BILT", "CETOT"])
d = d[d["EDAD"] >= 18]
flow.append(("After age >= 18", len(d)))
d = d[d["RBO"] != 99999.0]
flow.append(("After RBO 99999 (equipment error) removal", len(d)))
flow_df = pd.DataFrame(flow, columns=["Step", "N"])
flow_df["Excluded"] = flow_df["N"].shift(1) - flow_df["N"]
print("=== Cohort flow ===")
print(flow_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. Exact modelled cohort with original values (RAW_INDEX join)
# ---------------------------------------------------------------------------
raw["RAW_INDEX"] = raw.index + 1
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
cols = ["EDAD", "SEXO", "LEUT", "NITT", "DENST", "PHT", "PROTT", "HEMATT",
        "RBO", "WBCO", "EC", "BACTS", "CASTS", "YLC", "BACT_INFO"]
coh = ml[["RAW_INDEX", TARGET]].merge(raw[["RAW_INDEX"] + cols], on="RAW_INDEX", how="left")
assert coh[cols].notna().all(axis=1).mean() > 0.99, "unexpected NaNs after join"
print(f"\nModelled cohort: n={len(coh)}  "
      f"(culture+ {int((coh[TARGET]==1).sum())}, culture- {int((coh[TARGET]==0).sum())}, "
      f"prevalence {(coh[TARGET]==1).mean()*100:.1f}%)")

neg = coh[coh[TARGET] == 0]
pos = coh[coh[TARGET] == 1]

# Gram flag: map long text -> short category
def gram_cat(x):
    s = str(x).lower()
    if "sin informaci" in s: return "No info"
    if ("negativo" in s) and ("positivo" in s or "gran positivo" in s): return "Mixed"
    if "positivo" in s: return "Gram+"
    if "negativo" in s: return "Gram-"
    return "No info"
coh["GRAM_FLAG"] = coh["BACT_INFO"].map(gram_cat)
neg = coh[coh[TARGET] == 0]; pos = coh[coh[TARGET] == 1]

# ---------------------------------------------------------------------------
# 3. Table 1
# ---------------------------------------------------------------------------
def med_iqr(s):
    return f"{s.median():.2f} [{s.quantile(.25):.2f}-{s.quantile(.75):.2f}]"

def cont_row(label, col):
    p = stats.mannwhitneyu(neg[col].dropna(), pos[col].dropna()).pvalue
    return {"Variable": label, "Overall": med_iqr(coh[col]),
            "Culture- (n=%d)" % len(neg): med_iqr(neg[col]),
            "Culture+ (n=%d)" % len(pos): med_iqr(pos[col]),
            "p": f"{p:.3g}"}

def cat_rows(label, col, order=None):
    cats = order or sorted(coh[col].dropna().unique().tolist())
    ct = pd.crosstab(coh[col], coh[TARGET])
    ct = ct.reindex(cats).fillna(0)
    p = stats.chi2_contingency(ct.values)[1]
    rows = [{"Variable": label, "Overall": "", "Culture- (n=%d)" % len(neg): "",
             "Culture+ (n=%d)" % len(pos): "", "p": f"{p:.3g}"}]
    for c in cats:
        ov = (coh[col] == c).sum(); nn = (neg[col] == c).sum(); pp = (pos[col] == c).sum()
        rows.append({"Variable": f"  {c}",
                     "Overall": f"{ov} ({ov/len(coh)*100:.1f}%)",
                     "Culture- (n=%d)" % len(neg): f"{nn} ({nn/len(neg)*100:.1f}%)",
                     "Culture+ (n=%d)" % len(pos): f"{pp} ({pp/len(pos)*100:.1f}%)",
                     "p": ""})
    return rows

coh["SEXO"] = coh["SEXO"].map({"H": "Male", "M": "Female"}).fillna(coh["SEXO"])
neg = coh[coh[TARGET] == 0]; pos = coh[coh[TARGET] == 1]

rows = []
rows.append(cont_row("Age, years", "EDAD"))
rows += cat_rows("Sex", "SEXO", order=["Male", "Female"])
rows += cat_rows("Leukocyte esterase", "LEUT", order=["Negativo", "25", "75", "500"])
rows += cat_rows("Nitrite", "NITT", order=["Negativo", "Positivo"])
rows.append(cont_row("Specific gravity", "DENST"))
rows.append(cont_row("pH", "PHT"))
rows.append(cont_row("Bacteria (/uL)", "BACTS"))
rows.append(cont_row("White blood cells (/uL)", "WBCO"))
rows.append(cont_row("Red blood cells (/uL)", "RBO"))
rows.append(cont_row("Epithelial cells (/uL)", "EC"))
rows.append(cont_row("Hyaline casts (/uL)", "CASTS"))
rows.append(cont_row("Yeasts (/uL)", "YLC"))
rows += cat_rows("Sysmex Gram flag", "GRAM_FLAG", order=["Gram-", "Gram+", "Mixed", "No info"])

table1 = pd.DataFrame(rows)
print("\n=== Table 1 ===")
print(table1.to_string(index=False))

out = os.path.join(BASE, "data", "04_results", "table1_cohort.xlsx")
with pd.ExcelWriter(out) as xw:
    flow_df.to_excel(xw, sheet_name="cohort_flow", index=False)
    table1.to_excel(xw, sheet_name="table1", index=False)
print(f"\nSaved: {out}")
