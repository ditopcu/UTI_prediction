# -*- coding: utf-8 -*-
"""
§10 Antibiotic (ATB) subgroup analysis — NO retraining.

Final CatBoost+Optuna model (loaded) scored on the holdout (Other) set, split into
patients on antibiotic treatment vs not, identified by keyword matching over the raw
medication field TTO_ATB_GAYA (joined via RAW_INDEX). Tests whether performance is
consistent across subgroups.

Outputs:
  - data/04_results/atb_subgroup.xlsx
  - figures/.../ figure_11_atb_subgroup_roc
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, average_precision_score, roc_auc_score,
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    recall_score, confusion_matrix,
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_SINGLE, save_figure

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
TARGET = "CULTIVO_PATOLOGICO"

C_ATB = PALETTE["accent2"]      # orange  — ATB-treated
C_NO = PALETTE["base1"]         # steel   — no ATB
C_ALL = PALETTE["highlight"]    # ruby    — overall
C_REF = PALETTE["ci_grey"]

ATB_KEYWORDS = [
    "CEFALOSPORINAS", "FLUORQUINOLONAS", "FLUOROQUINOLONAS", "PENICILINAS",
    "ASOCIACIONES DE PENICILINAS", "MACROLIDOS", "SULFONAMIDAS Y TRIMETOPRIM",
    "NITROFURANO", "POLIMIXINAS", "LINCOSAMIDAS", "ANTIBACTERIANOS GLICOPEPTIDOS",
    "CARBAPENEMS", "TETRACICLINAS", "AMINOGLUCOSIDOS",
    "ANTIBIOTICOS (ANTIINFECC. INTESTINALES)", "ANTIBIOTICOS (ANTIMICOTICOS",
    "OTROS ANTIBACTERIANOS", "PENICILINAS DE AMPLIO ESPECTRO", "PENICILINAS BETALACTAMASA",
]

# ---------------------------------------------------------------------------
# Reconstruct holdout, load model, join ATB flag
# ---------------------------------------------------------------------------
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1); y = df[TARGET]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_test, X_other, y_test, y_other = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)

model = CatBoostClassifier(); model.load_model(os.path.join(BASE, "models", "model_optuna.cbm"))
feats = model.feature_names_

raw = pd.read_excel(os.path.join(BASE, "data", "01_raw", "uti_raw.xlsx")); raw["RAW_INDEX"] = raw.index + 1
def has_atb(x):
    s = str(x).upper()
    return int(any(k.upper() in s for k in ATB_KEYWORDS))
raw["has_atb"] = raw["TTO_ATB_GAYA"].map(has_atb)

hold = pd.DataFrame({
    "RAW_INDEX": ml.loc[X_other.index, "RAW_INDEX"].values,
    "y": y_other.values,
    "pred": model.predict(X_other[feats]).astype(int),
    "proba": model.predict_proba(X_other[feats])[:, 1],
}).merge(raw[["RAW_INDEX", "has_atb"]], on="RAW_INDEX", how="left")

# ---------------------------------------------------------------------------
# Metrics per subgroup
# ---------------------------------------------------------------------------
def metrics(name, d):
    yt = d["y"].astype(int).values; yp = d["pred"].values; pr = d["proba"].values
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {"Subgroup": name, "N": len(d), "Prevalence": round(yt.mean(), 3),
            "ROC AUC": round(roc_auc_score(yt, pr), 4),
            "PR AUC": round(average_precision_score(yt, pr), 4),
            "Accuracy": round(accuracy_score(yt, yp), 4),
            "Balanced Acc": round(balanced_accuracy_score(yt, yp), 4),
            "Sensitivity": round(recall_score(yt, yp), 4),
            "Specificity": round(tn / (tn + fp), 4),
            "F1": round(f1_score(yt, yp), 4),
            "MCC": round(matthews_corrcoef(yt, yp), 4)}

atb = hold[hold["has_atb"] == 1]; noatb = hold[hold["has_atb"] == 0]
tab = pd.DataFrame([metrics("All (holdout)", hold),
                    metrics("ATB-treated", atb),
                    metrics("No ATB", noatb)])
print(tab.to_string(index=False))
out = os.path.join(BASE, "data", "04_results", "atb_subgroup.xlsx")
tab.to_excel(out, index=False)
print(f"\nSaved: {out}")

# ---------------------------------------------------------------------------
# Figure 11 — ROC by subgroup (consistency)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.95))
for d, c, lbl in [(hold, C_ALL, "All"), (noatb, C_NO, "No ATB"), (atb, C_ATB, "ATB-treated")]:
    yt = d["y"].astype(int).values; pr = d["proba"].values
    fpr, tpr, _ = roc_curve(yt, pr)
    ax.plot(fpr, tpr, lw=2, color=c, label=f"{lbl} (AUC {auc(fpr, tpr):.3f}, n={len(d)})")
ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=C_REF)
ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
ax.legend(loc="lower right", fontsize=7.5)
save_figure(fig, "figure_11_atb_subgroup_roc", outdir=FIG_DIR)
print("Done.")
