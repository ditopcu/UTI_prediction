# -*- coding: utf-8 -*-
"""
§7 Model performance on the holdout (Other) set — NO retraining.

Loads the three saved models and scores the reconstructed holdout:
  - models/model_baseline.cbm      (CatBoost baseline)
  - models/model_optuna.cbm        (CatBoost + Optuna, final)
  - models/autogluon_uti_ec/       (AutoGluon best_quality ensemble)

Produces:
  - data/04_results/model_performance_holdout.xlsx   (metrics table)
  - figures/{PNG_300DPI,TIFF_600DPI,PDF_VECTOR}/
        figure_01_roc_holdout
        figure_02_pr_holdout
        figure_03_confusion_matrices

Manuscript figure style: plot_styles/figure_style.py (tufte + fixed palette).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef, f1_score,
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_SINGLE, COL_DOUBLE, save_figure

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
TARGET = "CULTIVO_PATOLOGICO"

# Fixed semantic colours (see CLAUDE.md → Figure Standards)
C_OPTUNA = PALETTE["highlight"]   # ruby  — proposed final model
C_AG = PALETTE["base1"]           # steel — AutoGluon benchmark
C_BASE = PALETTE["base2"]         # grey  — CatBoost baseline
C_REF = PALETTE["ci_grey"]        # grey  — reference line

# ---------------------------------------------------------------------------
# 1. Reconstruct the ENCODED holdout (CatBoost models) — deterministic split
# ---------------------------------------------------------------------------
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1)
y = df[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_test, X_other, y_test, y_other = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)
y_other = y_other.reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Load CatBoost models (no retraining) and score
# ---------------------------------------------------------------------------
model_opt = CatBoostClassifier(); model_opt.load_model(os.path.join(BASE, "models", "model_optuna.cbm"))
model_bas = CatBoostClassifier(); model_bas.load_model(os.path.join(BASE, "models", "model_baseline.cbm"))

feats = model_opt.feature_names_                      # the 16 Boruta features
X_other_b = X_other[feats]

proba_opt = model_opt.predict_proba(X_other_b)[:, 1]
pred_opt = model_opt.predict(X_other_b).astype(int)
proba_bas = model_bas.predict_proba(X_other_b)[:, 1]
pred_bas = model_bas.predict(X_other_b).astype(int)

# ---------------------------------------------------------------------------
# 3. Reconstruct the UN-ENCODED holdout (AutoGluon) and score the saved predictor
# ---------------------------------------------------------------------------
from autogluon.tabular import TabularPredictor
ag = TabularPredictor.load(os.path.join(BASE, "models", "autogluon_uti_ec"), verbosity=0)

v2 = pd.read_excel(os.path.join(BASE, "data", "02_interim", "uti_cleaned_v2.xlsx"))
raw = pd.read_excel(os.path.join(BASE, "data", "01_raw", "uti_raw.xlsx"))
safe_cols = ["FECHA", "EDAD", "SEXO", "WBCO", "EC", "BACTS", "RBO", "PHT", "YLC", "CASTS"]
raw_dedup = raw.drop_duplicates(subset=safe_cols, keep="first").copy()
raw_dedup["RAW_INDEX"] = raw_dedup.index + 1
v2 = v2.merge(raw_dedup[safe_cols + ["RAW_INDEX"]], on=safe_cols, how="left")

dfv = v2.copy()
dfv["NITT"] = dfv["NITT"].replace("Positivo", 1)
dfv = dfv.drop(columns=["FILTER"])
dfv.dropna(inplace=True)
orig_cols_v = [c for c in dfv.columns if c != "RAW_INDEX"]
dfv = dfv.drop_duplicates(subset=orig_cols_v, keep="first")
dfv = dfv.drop(columns=["FECHA", "XTAL", "UROT", "BILT", "CETOT"])
dfv = dfv[dfv["EDAD"] >= 18]
dfv["RBO"] = dfv["RBO"].replace(99999.0, np.nan)
dfv.dropna(subset=["RBO"], inplace=True)
bins_s = list(range(18, 90, 10))
bins = bins_s + [90, int(dfv["EDAD"].max()) + 1]
labels = [f"{i}-{i+9}" for i in bins_s] + [">=90"]
dfv["EDAD_CATEGORICA"] = pd.cut(dfv["EDAD"], bins=bins, labels=labels, right=False)
dfv["DENST"] = dfv["DENST"] / 1000
for col in ["PROTT", "CASTS", "YLC"]:
    dfv[col] = dfv[col].apply(lambda x: 0 if x == 0 else 1)

boruta_orig = ["DENST", "HEMATT", "RBO", "WBCO", "EC", "BACTS",
               "SEXO", "LEUT", "NITT", "PROTT", "BACT_INFO_baja", "EDAD_CATEGORICA"]
df_ag = dfv[boruta_orig + [TARGET, "RAW_INDEX"]].copy()
for col in ["SEXO", "LEUT", "NITT", "PROTT", "BACT_INFO_baja", "EDAD_CATEGORICA"]:
    df_ag[col] = df_ag[col].astype(str)

X_ag = df_ag.drop(columns=[TARGET])
y_ag = df_ag[TARGET]
_, X_temp_ag, _, y_temp_ag = train_test_split(X_ag, y_ag, test_size=0.40, random_state=42)
_, X_other_ag, _, y_other_ag = train_test_split(X_temp_ag, y_temp_ag, test_size=0.375, random_state=42)
other_ag = pd.concat([X_other_ag, y_other_ag], axis=1).drop(columns=["RAW_INDEX"])

proba_ag = ag.predict_proba(other_ag.drop(columns=[TARGET]))[1].values
pred_ag = ag.predict(other_ag.drop(columns=[TARGET])).astype(int).values
y_other_ag_vals = y_other_ag.reset_index(drop=True).values

# ---------------------------------------------------------------------------
# 4. Metrics table
# ---------------------------------------------------------------------------
def metrics(name, y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Model": name, "N": len(y_true),
        "ROC AUC": roc_auc_score(y_true, y_proba),
        "PR AUC": average_precision_score(y_true, y_proba),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "PPV": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

table = pd.DataFrame([
    metrics("CatBoost baseline", y_other, pred_bas, proba_bas),
    metrics("CatBoost+Optuna", y_other, pred_opt, proba_opt),
    metrics("AutoGluon ensemble", y_other_ag_vals, pred_ag, proba_ag),
])
num_cols = [c for c in table.columns if c not in ("Model", "N")]
table[num_cols] = table[num_cols].round(4)

out_xlsx = os.path.join(BASE, "data", "04_results", "model_performance_holdout.xlsx")
table.to_excel(out_xlsx, index=False)
print(table.to_string(index=False))
print(f"\nSaved metrics: {out_xlsx}")

# ---------------------------------------------------------------------------
# 5. Figure 1 — ROC (CatBoost+Optuna vs AutoGluon)
# ---------------------------------------------------------------------------
fpr_o, tpr_o, _ = roc_curve(y_other, proba_opt)
fpr_a, tpr_a, _ = roc_curve(y_other_ag_vals, proba_ag)

fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.9))
ax.plot(fpr_o, tpr_o, lw=2, color=C_OPTUNA, label=f"CatBoost+Optuna (AUC {auc(fpr_o, tpr_o):.3f})")
ax.plot(fpr_a, tpr_a, lw=2, color=C_AG, label=f"AutoGluon (AUC {auc(fpr_a, tpr_a):.3f})")
ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=C_REF)
ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
ax.legend(loc="lower right")
save_figure(fig, "figure_01_roc_holdout", outdir=FIG_DIR)

# ---------------------------------------------------------------------------
# 6. Figure 2 — Precision–Recall
# ---------------------------------------------------------------------------
prec_o, rec_o, _ = precision_recall_curve(y_other, proba_opt)
prec_a, rec_a, _ = precision_recall_curve(y_other_ag_vals, proba_ag)
prev = y_other.mean()

fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.9))
ax.plot(rec_o, prec_o, lw=2, color=C_OPTUNA,
        label=f"CatBoost+Optuna (AP {average_precision_score(y_other, proba_opt):.3f})")
ax.plot(rec_a, prec_a, lw=2, color=C_AG,
        label=f"AutoGluon (AP {average_precision_score(y_other_ag_vals, proba_ag):.3f})")
ax.axhline(prev, ls="--", lw=0.8, color=C_REF, label=f"Prevalence {prev:.2f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1.02)
ax.legend(loc="lower left")
save_figure(fig, "figure_02_pr_holdout", outdir=FIG_DIR)

# ---------------------------------------------------------------------------
# 7. Figure 3 — Confusion matrices (final model + AutoGluon)
# ---------------------------------------------------------------------------
def draw_cm(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    thr = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thr else PALETTE["text"], fontsize=13)
    ax.set_xticks([0, 1], ["Neg", "Pos"]); ax.set_yticks([0, 1], ["Neg", "Pos"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)

fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, COL_DOUBLE * 0.45))
draw_cm(axes[0], y_other, pred_opt, "A. CatBoost+Optuna")
draw_cm(axes[1], y_other_ag_vals, pred_ag, "B. AutoGluon")
save_figure(fig, "figure_03_confusion_matrices", outdir=FIG_DIR)

print("\nDone.")
