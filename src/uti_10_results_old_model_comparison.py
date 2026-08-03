# -*- coding: utf-8 -*-
"""
§8 Comparison against the existing hospital system — NO retraining.

Hospital system:
  - UTI_CDS  (neural network, 0-100 score, threshold 50)   -> base model
  - CDS_RNA  (random forest, Alta/Baja)                     -> base model
  - PRED_CULT_IA (combined decision of the two)  == the DEPLOYED hospital decision
    the hospital actually operates on; only issued when the two base models concur,
    covering ~52% of samples.

Compares the deployed CatBoost+Optuna model (loaded, not retrained) against the
hospital system on the holdout (Other) set, joined to the raw algorithm columns
via RAW_INDEX. Head-to-head metrics are computed on the subset where the deployed
hospital decision (PRED_CULT_IA) exists, for a fair same-patient comparison.

Outputs:
  - data/04_results/old_model_comparison.xlsx  (coverage + metrics sheets)
  - figures/.../ figure_04_decision_coverage
                 figure_05_headtohead_metrics
                 figure_06_correctness_quadrant
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    matthews_corrcoef, f1_score, accuracy_score, recall_score, precision_score,
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_SINGLE, COL_DOUBLE, save_figure

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
TARGET = "CULTIVO_PATOLOGICO"

C_OPTUNA = PALETTE["highlight"]   # ruby   — proposed model
C_PRED = PALETTE["base1"]         # steel  — PRED_CULT_IA (deployed hospital decision)
C_UTICDS = PALETTE["accent2"]     # orange — UTI_CDS (NN)
C_CDSRNA = PALETTE["accent3"]     # amethyst — CDS_RNA (RF)
C_REF = PALETTE["ci_grey"]

# ---------------------------------------------------------------------------
# 1. Reconstruct encoded holdout (keep index → RAW_INDEX) and load final model
# ---------------------------------------------------------------------------
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1)
y = df[TARGET]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_test, X_other, y_test, y_other = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)

model_opt = CatBoostClassifier(); model_opt.load_model(os.path.join(BASE, "models", "model_optuna.cbm"))
feats = model_opt.feature_names_
proba_opt = model_opt.predict_proba(X_other[feats])[:, 1]
pred_opt = model_opt.predict(X_other[feats]).astype(int)

hold = pd.DataFrame({
    "RAW_INDEX": ml.loc[X_other.index, "RAW_INDEX"].values,
    "y": y_other.values,
    "opt_pred": pred_opt,
    "opt_proba": proba_opt,
})

# ---------------------------------------------------------------------------
# 2. Parse hospital algorithm outputs from raw, join on RAW_INDEX
# ---------------------------------------------------------------------------
raw = pd.read_excel(os.path.join(BASE, "data", "01_raw", "uti_raw.xlsx"))
raw["RAW_INDEX"] = raw.index + 1

def p_cds_rna(x):
    s = str(x); return 1 if s.startswith("Alta") else (0 if s.startswith("Baja") else np.nan)
def p_pred(x):
    s = str(x)
    if ("Alta probabilidad" in s) or ("Positivo" in s): return 1
    if ("Baja probabilidad" in s) or ("Negativo" in s): return 0
    return np.nan
def p_uti_cds(x):
    return np.nan if pd.isna(x) else (1 if x >= 50 else 0)

raw["UTI_CDS_b"] = raw["UTI_CDS"].map(p_uti_cds)
raw["CDS_RNA_b"] = raw["CDS_RNA"].map(p_cds_rna)
raw["PRED_b"] = raw["PRED_CULT_IA"].map(p_pred)

hold = hold.merge(raw[["RAW_INDEX", "UTI_CDS_b", "CDS_RNA_b", "PRED_b"]], on="RAW_INDEX", how="left")
N = len(hold)

# ---------------------------------------------------------------------------
# 3. Decision coverage within the holdout
# ---------------------------------------------------------------------------
cov = {
    "CatBoost+Optuna": 100.0,
    "PRED_CULT_IA": hold["PRED_b"].notna().mean() * 100,
    "UTI_CDS": hold["UTI_CDS_b"].notna().mean() * 100,
    "CDS_RNA": hold["CDS_RNA_b"].notna().mean() * 100,
}
cov_df = pd.DataFrame({"Model": list(cov), "Coverage_%": [round(v, 1) for v in cov.values()],
                       "N_holdout": [N] * 4})
print("Holdout N =", N)
print(cov_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Head-to-head on the PRED_CULT_IA-present subset (same patients)
# ---------------------------------------------------------------------------
sub = hold[hold["PRED_b"].notna()].copy()
n_sub = len(sub)
yt = sub["y"].astype(int).values

def bmetrics(name, y_pred, y_proba=None):
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, y_pred, labels=[0, 1]).ravel()
    row = {
        "Model": name, "N": len(yt),
        "Accuracy": accuracy_score(yt, y_pred),
        "Sensitivity": recall_score(yt, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "PPV": precision_score(yt, y_pred, zero_division=0),
        "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
        "F1": f1_score(yt, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(yt, y_pred),
        "ROC AUC": roc_auc_score(yt, y_proba) if y_proba is not None else np.nan,
        "PR AUC": average_precision_score(yt, y_proba) if y_proba is not None else np.nan,
    }
    return row

h2h = pd.DataFrame([
    bmetrics("CatBoost+Optuna", sub["opt_pred"].values, sub["opt_proba"].values),
    bmetrics("PRED_CULT_IA (hospital)", sub["PRED_b"].values),
    bmetrics("UTI_CDS (NN)", sub["UTI_CDS_b"].values),
    bmetrics("CDS_RNA (RF)", sub["CDS_RNA_b"].values),
])
numc = [c for c in h2h.columns if c not in ("Model", "N")]
h2h[numc] = h2h[numc].round(4)
print(f"\nHead-to-head on PRED_CULT_IA-present subset (n={n_sub}):")
print(h2h.to_string(index=False))

# Also: CatBoost+Optuna on the FULL holdout (its natural coverage), for context
tn, fp, fn, tp = confusion_matrix(hold["y"], hold["opt_pred"], labels=[0, 1]).ravel()

out_xlsx = os.path.join(BASE, "data", "04_results", "old_model_comparison.xlsx")
with pd.ExcelWriter(out_xlsx) as xw:
    cov_df.to_excel(xw, sheet_name="coverage", index=False)
    h2h.to_excel(xw, sheet_name="head_to_head", index=False)
print(f"\nSaved: {out_xlsx}")

# ---------------------------------------------------------------------------
# 5. Figure 4 — decision coverage
# ---------------------------------------------------------------------------
order = ["CatBoost+Optuna", "PRED_CULT_IA", "UTI_CDS", "CDS_RNA"]
colors = [C_OPTUNA, C_PRED, C_UTICDS, C_CDSRNA]
vals = [cov[m] for m in order]

fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.85))
bars = ax.bar(range(len(order)), vals, color=colors, width=0.65)
for i, v in enumerate(vals):
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=10)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(["CatBoost\n+Optuna", "PRED_\nCULT_IA", "UTI_CDS", "CDS_RNA"], fontsize=9)
ax.set_ylabel("Samples with a decision (%)")
ax.set_ylim(0, 108)
ax.set_yticks([0, 25, 50, 75, 100])
save_figure(fig, "figure_04_decision_coverage", outdir=FIG_DIR)

# ---------------------------------------------------------------------------
# 6. Figure 5 — head-to-head metrics (proposed vs deployed hospital decision)
# ---------------------------------------------------------------------------
metric_names = ["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "MCC"]
opt_vals = h2h.loc[h2h["Model"] == "CatBoost+Optuna", metric_names].values.ravel()
pred_vals = h2h.loc[h2h["Model"] == "PRED_CULT_IA (hospital)", metric_names].values.ravel()

xpos = np.arange(len(metric_names)); w = 0.38
fig, ax = plt.subplots(figsize=(COL_DOUBLE, COL_DOUBLE * 0.42))
ax.bar(xpos - w/2, opt_vals, w, color=C_OPTUNA, label="CatBoost+Optuna")
ax.bar(xpos + w/2, pred_vals, w, color=C_PRED, label="PRED_CULT_IA (hospital)")
for i in range(len(metric_names)):
    ax.text(xpos[i] - w/2, opt_vals[i] + 0.01, f"{opt_vals[i]:.2f}", ha="center", va="bottom", fontsize=8)
    ax.text(xpos[i] + w/2, pred_vals[i] + 0.01, f"{pred_vals[i]:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xpos); ax.set_xticklabels(metric_names, fontsize=9)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", ncol=2)
ax.set_title(f"Concordant subset (n={n_sub})")
save_figure(fig, "figure_05_headtohead_metrics", outdir=FIG_DIR)

# ---------------------------------------------------------------------------
# 7. Figure 6 — correctness quadrant (new model vs hospital decision)
# ---------------------------------------------------------------------------
opt_ok = (sub["opt_pred"].values == yt)
pred_ok = (sub["PRED_b"].values.astype(int) == yt)
q = np.array([[(opt_ok & pred_ok).sum(),  (opt_ok & ~pred_ok).sum()],
              [(~opt_ok & pred_ok).sum(), (~opt_ok & ~pred_ok).sum()]])

fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.9))
ax.imshow(q, cmap="Blues")
thr = q.max() / 2
lbls = [["Both correct", "Only CatBoost\ncorrect"],
        ["Only hospital\ncorrect", "Both wrong"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{lbls[i][j]}\n{q[i,j]}  ({q[i,j]/n_sub*100:.1f}%)",
                ha="center", va="center", fontsize=9,
                color="white" if q[i, j] > thr else PALETTE["text"])
ax.set_xticks([0, 1], ["Hospital correct", "Hospital wrong"], fontsize=9)
ax.set_yticks([0, 1], ["CatBoost correct", "CatBoost wrong"], fontsize=9)
for s in ax.spines.values():
    s.set_visible(False)
save_figure(fig, "figure_06_correctness_quadrant", outdir=FIG_DIR)

print("\nDone.")
