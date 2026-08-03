# -*- coding: utf-8 -*-
"""
§11/§10 Uncertainty (bootstrap 95% CI) + fairness by sex — NO retraining.

Final CatBoost+Optuna model (loaded) scored on the holdout (Other) set:
  - Bootstrap 95% CIs for headline metrics (overall).
  - Performance by sex (Male/Female) with 95% CIs (fairness check).

Outputs:
  - data/04_results/ci_and_fairness.xlsx  (sheets: ci_overall, by_sex)
  - figures/.../ figure_13_sex_subgroup_roc
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef, recall_score, confusion_matrix,
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_SINGLE, save_figure

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
TARGET = "CULTIVO_PATOLOGICO"
SEED = 42
N_BOOT = 2000

C_M = PALETTE["base1"]      # steel  — male
C_F = PALETTE["accent3"]    # amethyst — female
C_ALL = PALETTE["highlight"]
C_REF = PALETTE["ci_grey"]

# ---------------------------------------------------------------------------
# Reconstruct holdout, load model, join sex
# ---------------------------------------------------------------------------
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1); y = df[TARGET]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_test, X_other, y_test, y_other = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)

model = CatBoostClassifier(); model.load_model(os.path.join(BASE, "models", "model_optuna.cbm"))
feats = model.feature_names_

raw = pd.read_excel(os.path.join(BASE, "data", "01_raw", "uti_raw.xlsx")); raw["RAW_INDEX"] = raw.index + 1
raw["SEX"] = raw["SEXO"].map({"H": "Male", "M": "Female"})

hold = pd.DataFrame({
    "RAW_INDEX": ml.loc[X_other.index, "RAW_INDEX"].values,
    "y": y_other.values.astype(int),
    "pred": model.predict(X_other[feats]).astype(int),
    "proba": model.predict_proba(X_other[feats])[:, 1],
}).merge(raw[["RAW_INDEX", "SEX"]], on="RAW_INDEX", how="left")

# ---------------------------------------------------------------------------
# Metrics + bootstrap CIs
# ---------------------------------------------------------------------------
def point_metrics(yt, yp, pr):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {
        "ROC AUC": roc_auc_score(yt, pr),
        "PR AUC": average_precision_score(yt, pr),
        "Accuracy": accuracy_score(yt, yp),
        "Sensitivity": recall_score(yt, yp),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "F1": f1_score(yt, yp),
        "MCC": matthews_corrcoef(yt, yp),
    }

def boot_ci(yt, yp, pr, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(yt)
    keys = ["ROC AUC", "PR AUC", "Accuracy", "Sensitivity", "Specificity", "F1", "MCC"]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(yt[idx])) < 2:
            continue
        m = point_metrics(yt[idx], yp[idx], pr[idx])
        for k in keys:
            acc[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in acc.items()}

def row(name, d):
    yt = d["y"].values; yp = d["pred"].values; pr = d["proba"].values
    pm = point_metrics(yt, yp, pr); ci = boot_ci(yt, yp, pr)
    out = {"Group": name, "N": len(d), "Prevalence": round(yt.mean(), 3)}
    for k in ["ROC AUC", "PR AUC", "Accuracy", "Sensitivity", "Specificity", "F1", "MCC"]:
        out[k] = f"{pm[k]:.3f} [{ci[k][0]:.3f}-{ci[k][1]:.3f}]"
    return out

overall = pd.DataFrame([row("All (holdout)", hold)])
male = hold[hold["SEX"] == "Male"]; female = hold[hold["SEX"] == "Female"]
by_sex = pd.DataFrame([row("Male", male), row("Female", female)])

print("=== Overall (95% CI) ===")
print(overall.to_string(index=False))
print("\n=== By sex (95% CI) ===")
print(by_sex.to_string(index=False))

out_xlsx = os.path.join(BASE, "data", "04_results", "ci_and_fairness.xlsx")
with pd.ExcelWriter(out_xlsx) as xw:
    overall.to_excel(xw, sheet_name="ci_overall", index=False)
    by_sex.to_excel(xw, sheet_name="by_sex", index=False)
print(f"\nSaved: {out_xlsx}")

# ---------------------------------------------------------------------------
# Figure 13 — ROC by sex
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.95))
for d, c, lbl in [(hold, C_ALL, "All"), (male, C_M, "Male"), (female, C_F, "Female")]:
    yt = d["y"].values; pr = d["proba"].values
    fpr, tpr, _ = roc_curve(yt, pr)
    ax.plot(fpr, tpr, lw=2, color=c, label=f"{lbl} (AUC {auc(fpr, tpr):.3f}, n={len(d)})")
ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=C_REF)
ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
ax.legend(loc="lower right", fontsize=7.5)
save_figure(fig, "figure_13_sex_subgroup_roc", outdir=FIG_DIR)
print("Done.")
