# -*- coding: utf-8 -*-
"""
§9 Interpretability — TreeSHAP on the final CatBoost+Optuna model (NO retraining).

Loads models/model_optuna.cbm, computes exact/deterministic TreeSHAP values on the
holdout (Other) set, and renders beeswarm + mean|SHAP| bar summaries with readable
feature labels.

Outputs:
  - figures/.../ figure_08_shap_beeswarm
                 figure_09_shap_bar
  - data/04_results/shap_importance.xlsx   (mean|SHAP| ranking)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_SINGLE, COL_DOUBLE, save_figure

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
TARGET = "CULTIVO_PATOLOGICO"

# Readable labels for the 16 encoded features
LABELS = {
    "DENST": "Specific gravity",
    "HEMATT": "Blood (Hb peroxidase)",
    "RBO": "Red blood cells",
    "WBCO": "White blood cells",
    "EC": "Epithelial cells",
    "BACTS": "Bacteria",
    "SEXO_M": "Sex (male)",
    "LEUT_25": "Leukocyte esterase 25",
    "LEUT_75": "Leukocyte esterase 75",
    "LEUT_500": "Leukocyte esterase 500",
    "NITT_1": "Nitrite positive",
    "PROTT_1": "Protein positive",
    "BACT_INFO_baja_1": "Sysmex Gram+ flag",
    "BACT_INFO_baja_2": "Sysmex Gram± (mixed) flag",
    "BACT_INFO_baja_3": "Sysmex no-Gram-info flag",
    "EDAD_CATEGORICA_28-37": "Age 28-37",
}

# ---------------------------------------------------------------------------
# Reconstruct holdout, load model, compute SHAP
# ---------------------------------------------------------------------------
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1); y = df[TARGET]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_test, X_other, y_test, y_other = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)

model = CatBoostClassifier(); model.load_model(os.path.join(BASE, "models", "model_optuna.cbm"))
feats = model.feature_names_
X_orig = X_other[feats]                       # original names (required by CatBoost SHAP)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_orig)   # exact, deterministic for tree models

X_other_b = X_orig.rename(columns=LABELS)     # readable labels, same column order → for plots

# mean|SHAP| ranking
imp = pd.DataFrame({
    "Feature": X_other_b.columns,
    "mean_abs_SHAP": np.abs(shap_values).mean(axis=0),
}).sort_values("mean_abs_SHAP", ascending=False).round(4)
out_xlsx = os.path.join(BASE, "data", "04_results", "shap_importance.xlsx")
imp.to_excel(out_xlsx, index=False)
print(imp.to_string(index=False))
print(f"\nSaved: {out_xlsx}")

# ---------------------------------------------------------------------------
# Figure 8 — beeswarm
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(COL_DOUBLE * 0.62, COL_DOUBLE * 0.6))
shap.summary_plot(shap_values, X_other_b, show=False, max_display=16, plot_size=None)
plt.gca().set_xlabel("SHAP value")
save_figure(fig, "figure_08_shap_beeswarm", outdir=FIG_DIR)

# ---------------------------------------------------------------------------
# Figure 9 — mean|SHAP| bar
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(COL_DOUBLE * 0.55, COL_DOUBLE * 0.5))
shap.summary_plot(shap_values, X_other_b, plot_type="bar", show=False,
                  max_display=16, color=PALETTE["highlight"], plot_size=None)
plt.gca().set_xlabel("mean |SHAP|")
save_figure(fig, "figure_09_shap_bar", outdir=FIG_DIR)

print("\nDone.")
