# -*- coding: utf-8 -*-
"""
§8 ROC / PR: new model vs the previous (published) hospital model — NO retraining.

On the concordant subset where the previous deployed decision (PRED_CULT_IA) exists
(same patients, culture as ground truth):
  - New model (CatBoost+Optuna): full ROC/PR curve (probability output)
  - Previous model NN score (UTI_CDS, 0-100): full ROC/PR curve (continuous score)
  - PRED_CULT_IA (deployed decision) and CDS_RNA (RF): binary -> single operating points

Outputs:
  - data/04_results/roc_pr_vs_previous.xlsx
  - figures/.../ figure_10_roc_pr_vs_previous
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_DOUBLE, save_figure

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
TARGET = "CULTIVO_PATOLOGICO"

C_NEW = PALETTE["highlight"]    # ruby   — new model
C_NN = PALETTE["accent2"]       # orange — previous NN (UTI_CDS) score curve
C_PRED = PALETTE["base1"]       # steel  — PRED_CULT_IA operating point
C_CDS = PALETTE["accent3"]      # amethyst — CDS_RNA operating point
C_REF = PALETTE["ci_grey"]

# ---------------------------------------------------------------------------
# Reconstruct holdout, load model, join previous-model outputs
# ---------------------------------------------------------------------------
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1); y = df[TARGET]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_test, X_other, y_test, y_other = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)

model = CatBoostClassifier(); model.load_model(os.path.join(BASE, "models", "model_optuna.cbm"))
proba_new = model.predict_proba(X_other[model.feature_names_])[:, 1]

raw = pd.read_excel(os.path.join(BASE, "data", "01_raw", "uti_raw.xlsx"))
raw["RAW_INDEX"] = raw.index + 1
def p_cds_rna(x):
    s = str(x); return 1 if s.startswith("Alta") else (0 if s.startswith("Baja") else np.nan)
def p_pred(x):
    s = str(x)
    if ("Alta probabilidad" in s) or ("Positivo" in s): return 1
    if ("Baja probabilidad" in s) or ("Negativo" in s): return 0
    return np.nan
raw["CDS_RNA_b"] = raw["CDS_RNA"].map(p_cds_rna)
raw["PRED_b"] = raw["PRED_CULT_IA"].map(p_pred)
raw["UTI_CDS_score"] = pd.to_numeric(raw["UTI_CDS"], errors="coerce")

hold = pd.DataFrame({"RAW_INDEX": ml.loc[X_other.index, "RAW_INDEX"].values,
                     "y": y_other.values, "new": proba_new})
hold = hold.merge(raw[["RAW_INDEX", "UTI_CDS_score", "CDS_RNA_b", "PRED_b"]], on="RAW_INDEX", how="left")

# Concordant subset: previous deployed decision available
sub = hold[hold["PRED_b"].notna()].copy()
yt = sub["y"].astype(int).values
n = len(sub)
print(f"Concordant subset n={n}  (culture+ {int(yt.sum())})")

# ---------------------------------------------------------------------------
# Curves + operating points
# ---------------------------------------------------------------------------
fpr_new, tpr_new, _ = roc_curve(yt, sub["new"].values); auc_new = auc(fpr_new, tpr_new)
fpr_nn, tpr_nn, _ = roc_curve(yt, sub["UTI_CDS_score"].values); auc_nn = auc(fpr_nn, tpr_nn)
prec_new, rec_new, _ = precision_recall_curve(yt, sub["new"].values); ap_new = average_precision_score(yt, sub["new"].values)
prec_nn, rec_nn, _ = precision_recall_curve(yt, sub["UTI_CDS_score"].values); ap_nn = average_precision_score(yt, sub["UTI_CDS_score"].values)

def op_point(pred):
    pred = pred.astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
    tpr = tp / (tp + fn); fpr = fp / (fp + tn)
    prec = tp / (tp + fp) if (tp + fp) else np.nan; rec = tpr
    return fpr, tpr, rec, prec

pred_fpr, pred_tpr, pred_rec, pred_prec = op_point(sub["PRED_b"].values)
cds_fpr, cds_tpr, cds_rec, cds_prec = op_point(sub["CDS_RNA_b"].values)
prev = yt.mean()

summary = pd.DataFrame([
    {"Model": "New (CatBoost+Optuna)", "ROC AUC": round(auc_new, 4), "PR AUC": round(ap_new, 4), "Type": "curve"},
    {"Model": "Previous NN score (UTI_CDS)", "ROC AUC": round(auc_nn, 4), "PR AUC": round(ap_nn, 4), "Type": "curve"},
    {"Model": "PRED_CULT_IA (deployed)", "ROC AUC": np.nan, "PR AUC": np.nan, "Type": f"point sens={pred_tpr:.3f} spec={1-pred_fpr:.3f}"},
    {"Model": "CDS_RNA (RF)", "ROC AUC": np.nan, "PR AUC": np.nan, "Type": f"point sens={cds_tpr:.3f} spec={1-cds_fpr:.3f}"},
])
print(summary.to_string(index=False))
out_xlsx = os.path.join(BASE, "data", "04_results", "roc_pr_vs_previous.xlsx")
summary.to_excel(out_xlsx, index=False)
print(f"Saved: {out_xlsx}")

# ---------------------------------------------------------------------------
# Figure 10 — ROC (A) + PR (B)
# ---------------------------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(COL_DOUBLE, COL_DOUBLE * 0.46))

# ROC
axA.plot(fpr_new, tpr_new, lw=2, color=C_NEW, label=f"New model (AUC {auc_new:.3f})")
axA.plot(fpr_nn, tpr_nn, lw=2, color=C_NN, label=f"Previous NN score (AUC {auc_nn:.3f})")
axA.plot(pred_fpr, pred_tpr, "D", ms=8, color=C_PRED, label="PRED_CULT_IA (deployed)")
axA.plot(cds_fpr, cds_tpr, "^", ms=9, color=C_CDS, label="CDS_RNA (RF)")
axA.plot([0, 1], [0, 1], ls="--", lw=0.8, color=C_REF)
axA.set_xlabel("1 - Specificity"); axA.set_ylabel("Sensitivity")
axA.set_xlim(-0.01, 1.01); axA.set_ylim(-0.01, 1.01)
axA.legend(loc="lower right", fontsize=7.5)
axA.set_title(f"A. ROC (n={n})")

# PR
axB.plot(rec_new, prec_new, lw=2, color=C_NEW, label=f"New model (AP {ap_new:.3f})")
axB.plot(rec_nn, prec_nn, lw=2, color=C_NN, label=f"Previous NN score (AP {ap_nn:.3f})")
axB.plot(pred_rec, pred_prec, "D", ms=8, color=C_PRED, label="PRED_CULT_IA (deployed)")
axB.plot(cds_rec, cds_prec, "^", ms=9, color=C_CDS, label="CDS_RNA (RF)")
axB.axhline(prev, ls="--", lw=0.8, color=C_REF, label=f"Prevalence {prev:.2f}")
axB.set_xlabel("Recall"); axB.set_ylabel("Precision")
axB.set_xlim(-0.01, 1.01); axB.set_ylim(0, 1.02)
axB.legend(loc="lower left", fontsize=7.5)
axB.set_title("B. Precision-Recall")

save_figure(fig, "figure_10_roc_pr_vs_previous", outdir=FIG_DIR)
print("Done.")
