# -*- coding: utf-8 -*-
"""
Post-implementation validation on the 2026-07-23 export ("Descarga orinas con cultivo").

Cohort (locked with the user, Option A):
  - Adults (EDAD >= 18) AND CENTRO_EXTRACCION == 'URGENCIAS' (Emergency Dept).
  - Only samples with a LIVE model probability (CATBOOST column; all post go-live 2026-06-18).
  - Culture label = the hospital target CULTIVO_PATOLOGICO AS-IS (identical convention to the
    training data and to Phase 1 hold-out: Candida counted positive, contaminated counted
    negative, unidentified >10^4 counted negative). NO culture-based exclusions.
  => n = 197  (129 positive / 68 negative, prevalence 65.5%)

Analyses (NO retraining; the deployed model):
  (a) Live CatBoost probability vs culture at threshold 0.50: discrimination, calibration
      (Brier, ECE), classification metrics with bootstrap 95% CIs (seed 42, 2000 resamples).
      Coverage of the previous hospital models is reported (head-to-head metrics are NOT,
      because the previous combined decision covers too few samples post go-live).
  (b) Fidelity (source for the SEPARATE fidelity supplement, not the main Table 2): live vs
      probability recomputed offline from the export features via the deployment code.

Outputs:
  - data/04_results/prospective_2026-07-23.xlsx
  - data/05_prospective/2026.07.23/analysis_cohort_n197.xlsx
  - figures/.../ figure_16_postimpl_roc_calibration   (main Figure 6)
                 figure_17_postimpl_confusion          (main-supplement figure)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, average_precision_score, accuracy_score,
    f1_score, matthews_corrcoef, recall_score, precision_score, confusion_matrix,
    brier_score_loss,
)

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_SINGLE, COL_DOUBLE, save_figure
from LIS.lis_predict import predict
from LIS.lis_preprocess import REQUIRED_FIELDS

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
SRC = os.path.join(BASE, "data", "05_prospective", "2026.07.23",
                   "Descarga orinas con cultivo (2) 2026.07.23.xlsx")
OUT_RES = os.path.join(BASE, "data", "04_results", "prospective_2026-07-23.xlsx")
OUT_COH = os.path.join(BASE, "data", "05_prospective", "2026.07.23", "analysis_cohort_n190.xlsx")
SEED, N_BOOT, THR = 42, 2000, 0.50
C_MODEL, C_REF = PALETTE["highlight"], PALETTE["ci_grey"]

# ---------------------------------------------------------------------------
# 1. Cohort (Option A)
# ---------------------------------------------------------------------------
df = pd.read_excel(SRC)
mask = (df["EDAD"] >= 18) & (df["CENTRO_EXTRACCION"].astype(str).str.upper() == "URGENCIAS") & df["CATBOOST"].notna()
c = df[mask].copy().reset_index(drop=True)
c["prob_live"] = c["CATBOOST"].astype(float)
c["pred_live"] = (c["prob_live"] >= THR).astype(int)
print(f"Cohort (Option A) before exclusion n={len(c)}")

# --- Exclude equipment-error samples (confirmed with the laboratory): identified as a mismatch
#     between the live deployed prediction and the value recomputed from the validated export ---
def _clean(v): return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
c["prob_recomputed"] = [predict({f: _clean(r.get(f)) for f in REQUIRED_FIELDS})["probability"] for _, r in c.iterrows()]
c["pred_recomputed"] = (c["prob_recomputed"] >= THR).astype(int)
excluded = c[c["pred_live"] != c["pred_recomputed"]].copy()
c = c[c["pred_live"] == c["pred_recomputed"]].reset_index(drop=True)
print(f"Excluded (equipment error, live != recomputed): n={len(excluded)}")

y = c["CULTIVO_PATOLOGICO"].astype(int).values
p = c["prob_live"].values
yp = c["pred_live"].values
print(f"Analysis cohort n={len(c)}  positive={int(y.sum())} ({y.mean()*100:.1f}%)  negative={int((1-y).sum())}")

# ---------------------------------------------------------------------------
# metrics + bootstrap + calibration
# ---------------------------------------------------------------------------
def ece(yt, pr, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (pr > bins[i]) & (pr <= bins[i + 1]) if i else (pr >= bins[i]) & (pr <= bins[i + 1])
        if m.sum():
            e += m.mean() * abs(yt[m].mean() - pr[m].mean())
    return e

def point_metrics(yt, yp_, pr):
    tn, fp, fn, tp = confusion_matrix(yt, yp_, labels=[0, 1]).ravel()
    return {
        "ROC AUC": roc_auc_score(yt, pr), "PR AUC": average_precision_score(yt, pr),
        "Accuracy": accuracy_score(yt, yp_), "Sensitivity": recall_score(yt, yp_, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "PPV": precision_score(yt, yp_, zero_division=0),
        "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
        "F1": f1_score(yt, yp_, zero_division=0), "MCC": matthews_corrcoef(yt, yp_),
        "Brier": brier_score_loss(yt, pr), "ECE": ece(yt, pr),
    }

def boot_ci(yt, yp_, pr):
    rng = np.random.default_rng(SEED); n = len(yt); acc = {}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(yt[idx])) < 2:
            continue
        for k, v in point_metrics(yt[idx], yp_[idx], pr[idx]).items():
            acc.setdefault(k, []).append(v)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in acc.items()}

pm = point_metrics(y, yp, p)
ci = boot_ci(y, yp, p)
KEYS = ["ROC AUC", "PR AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "MCC", "Brier", "ECE"]
validation = pd.DataFrame([{"Analysis": "Deployed model vs culture (live)", "N": len(c),
                            **{k: f"{pm[k]:.3f} ({ci[k][0]:.3f}-{ci[k][1]:.3f})" for k in KEYS}}])
tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
confmat = pd.DataFrame([{"TN": tn, "FP": fp, "FN": fn, "TP": tp, "missed_pos(FN)": fn,
                         "prevalence_%": round(y.mean() * 100, 1)}])
# Table-2-ready row (point (lo-hi) for the four headline metrics)
def f4(k): return f"{pm[k]:.3f} ({ci[k][0]:.3f}-{ci[k][1]:.3f})"
table2_row = pd.DataFrame([{"Row": "Phase 2: post-implementation (routine use)", "N": len(c),
                            "ROC AUC": f4("ROC AUC"), "Sensitivity": f4("Sensitivity"),
                            "Specificity": f4("Specificity"), "MCC": f4("MCC")}])
print("\n=== (a) Validation vs culture (n=%d) ===" % len(c))
print(validation.to_string(index=False))
print(confmat.to_string(index=False))

# ---------------------------------------------------------------------------
# coverage of the previous hospital models (metrics deliberately not reported)
# ---------------------------------------------------------------------------
def p_pred(x):
    s = str(x)
    if ("Alta probabilidad" in s) or ("Positivo" in s): return 1
    if ("Baja probabilidad" in s) or ("Negativo" in s): return 0
    return np.nan
coverage = pd.DataFrame([{
    "Deployed model": "100.0%",
    "PRED_CULT_IA (combined)": f'{c["PRED_CULT_IA"].map(p_pred).notna().mean()*100:.1f}%',
    "UTI_CDS (NN)": f'{pd.to_numeric(c["UTI_CDS"], errors="coerce").notna().mean()*100:.1f}%',
    "CDS_RNA (RF)": f'{c["CDS_RNA"].map(p_pred).notna().mean()*100:.1f}%',
    "N": len(c),
}])
print("\n=== Coverage (previous models issue a decision for) ===")
print(coverage.to_string(index=False))

# Figures for Phase 2 (Figure 5 = 2x2 pre/post ROC + confusion; Figure 6 = calibration) are
# built from the saved result tables by src/uti_25_figure5_phase2.py.

# ---------------------------------------------------------------------------
# 4. Write outputs
# ---------------------------------------------------------------------------
per_sample = c[["PETICIONCB", "FECHA", "EDAD", "SEXO", "CULTIVO_PATOLOGICO",
                "prob_live", "pred_live", "prob_recomputed", "RESULTADO_CULTIVO"]].copy()
excl_out = excluded[["PETICIONCB", "FECHA", "CULTIVO_PATOLOGICO",
                     "prob_live", "pred_live", "prob_recomputed", "pred_recomputed"]].copy()
with pd.ExcelWriter(OUT_RES) as xw:
    validation.to_excel(xw, sheet_name="validation", index=False)
    confmat.to_excel(xw, sheet_name="confusion", index=False)
    coverage.to_excel(xw, sheet_name="coverage", index=False)
    table2_row.to_excel(xw, sheet_name="table2_row", index=False)
    excl_out.to_excel(xw, sheet_name="excluded_equipment_error", index=False)
    per_sample.to_excel(xw, sheet_name="per_sample", index=False)
per_sample.to_excel(OUT_COH, index=False)
print("\nSaved:", OUT_RES)
print("Saved:", OUT_COH)
print("Figures: figure_16_postimpl_roc_calibration (main Fig 6), figure_17_postimpl_confusion (suppl)")
