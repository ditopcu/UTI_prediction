# -*- coding: utf-8 -*-
"""
Reproduce the Boruta feature-selection result for the supplement (figure + table).

Feature selection (not model training) is re-run deterministically (seed 42) on the
training partition, exactly as in the development pipeline: a balanced random forest
(max_depth 5) base learner, BorutaPy(n_estimators='auto'). The confirmed set matches
the 16 features of the deployed model.

Outputs:
  - data/04_results/boruta_selection.xlsx
  - figures/.../ figure_S_boruta_selection
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

warnings.filterwarnings("ignore")
np.int = int; np.float = float; np.bool = bool   # BorutaPy compat with modern numpy

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_SINGLE, save_figure

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
TARGET = "CULTIVO_PATOLOGICO"

_LAB = {"DENST": "Specific gravity", "HEMATT": "Blood (Hb peroxidase)", "RBO": "Red blood cells",
        "WBCO": "White blood cells", "EC": "Epithelial cells", "BACTS": "Bacteria",
        "SEXO_M": "Sex (male)", "LEUT_25": "Leukocyte esterase 25", "LEUT_75": "Leukocyte esterase 75",
        "LEUT_500": "Leukocyte esterase 500", "NITT_1": "Nitrite positive", "PROTT_1": "Protein positive",
        "GLUT": "Glucose (positive)", "YLC_1": "Yeasts (present)", "CASTS_1": "Hyaline casts (present)",
        "BACT_INFO_baja_1": "Sysmex Gram+ flag", "BACT_INFO_baja_2": "Sysmex Gram± flag",
        "BACT_INFO_baja_3": "Sysmex no-Gram-info flag"}

def LABEL(f):
    if f in _LAB:
        return _LAB[f]
    if f.startswith("EDAD_CATEGORICA_"):
        return "Age " + f.split("_")[-1]
    if f.startswith("PHT_"):
        return "pH " + f.split("_")[-1]
    return f
LABELS = type("D", (), {"get": staticmethod(lambda f, d=None: LABEL(f))})()

# ---------------------------------------------------------------------------
# Training partition (encoded)
# ---------------------------------------------------------------------------
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1); y = df[TARGET]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)

# ---------------------------------------------------------------------------
# Boruta (deterministic, seed 42) — feature selection only
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5, random_state=42)
sel = BorutaPy(rf, n_estimators="auto", verbose=0, random_state=42)
sel.fit(X_train.values, y_train.values)
rf.fit(X_train.values, y_train.values)

bor = pd.DataFrame({
    "Feature": [LABELS.get(f, f) for f in X_train.columns],
    "Encoded name": X_train.columns,
    "Boruta decision": np.where(sel.support_, "Confirmed",
                        np.where(sel.support_weak_, "Tentative", "Rejected")),
    "Boruta ranking": sel.ranking_,
    "RF importance": np.round(rf.feature_importances_, 5),
}).sort_values(["Boruta ranking", "RF importance"], ascending=[True, False]).reset_index(drop=True)

n_conf = int((bor["Boruta decision"] == "Confirmed").sum())
print(f"Boruta confirmed: {n_conf} / {len(bor)} encoded features")
print(bor.to_string(index=False))
out = os.path.join(BASE, "data", "04_results", "boruta_selection.xlsx")
bor.to_excel(out, index=False)
print("Saved:", out)

# ---------------------------------------------------------------------------
# Figure — importance bar, coloured by Boruta decision
# ---------------------------------------------------------------------------
b = bor.sort_values("RF importance")
colors = {"Confirmed": PALETTE["highlight"], "Tentative": PALETTE["accent2"], "Rejected": PALETTE["base2"]}
fig, ax = plt.subplots(figsize=(COL_SINGLE * 1.5, COL_SINGLE * 1.5))
ax.barh(range(len(b)), b["RF importance"], color=[colors[s] for s in b["Boruta decision"]], height=.74)
ax.set_yticks(range(len(b))); ax.set_yticklabels(b["Feature"], fontsize=8)
ax.set_xlabel("Random-forest importance")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ["Confirmed", "Rejected"]]
ax.legend(handles, ["Confirmed (retained)", "Rejected"], loc="lower right", fontsize=9)
save_figure(fig, "figure_S_boruta_selection", outdir=FIG_DIR)
print("Done.")
