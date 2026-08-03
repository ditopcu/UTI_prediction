# -*- coding: utf-8 -*-
"""
§11 Calibration of the final CatBoost+Optuna model — NO retraining.

Loads models/model_optuna.cbm, scores the holdout (Other) set, and assesses
probability calibration:
  - reliability (calibration) curve
  - Brier score
  - Expected Calibration Error (ECE, 10 equal-width bins)

Post-hoc recalibration (model unchanged): a 1-D calibrator (isotonic and Platt/
sigmoid) is FIT ON THE TEST SET probabilities and APPLIED to the Other set, so the
holdout stays untouched. Before vs after is compared.

Outputs:
  - data/04_results/calibration.xlsx
  - figures/.../ figure_07_calibration
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_SINGLE, COL_DOUBLE, save_figure

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
TARGET = "CULTIVO_PATOLOGICO"

C_RAW = PALETTE["highlight"]    # ruby   — raw model
C_ISO = PALETTE["base1"]        # steel  — isotonic
C_SIG = PALETTE["accent2"]      # orange — sigmoid/Platt
C_REF = PALETTE["ci_grey"]

# ---------------------------------------------------------------------------
# Reconstruct Test + Other splits (deterministic) and load final model
# ---------------------------------------------------------------------------
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1); y = df[TARGET]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_test, X_other, y_test, y_other = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)

model = CatBoostClassifier(); model.load_model(os.path.join(BASE, "models", "model_optuna.cbm"))
feats = model.feature_names_
p_test = model.predict_proba(X_test[feats])[:, 1]
p_other = model.predict_proba(X_other[feats])[:, 1]
yt = y_other.values.astype(int)
ytest = y_test.values.astype(int)


def ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error, equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        e += (m.sum() / len(y_true)) * abs(acc - conf)
    return e


# ---------------------------------------------------------------------------
# Post-hoc recalibration: fit on TEST probs, apply to OTHER probs
# ---------------------------------------------------------------------------
iso = IsotonicRegression(out_of_bounds="clip").fit(p_test, ytest)
p_iso = iso.predict(p_other)

platt = LogisticRegression().fit(p_test.reshape(-1, 1), ytest)
p_sig = platt.predict_proba(p_other.reshape(-1, 1))[:, 1]

variants = [("Raw (CatBoost+Optuna)", p_other, C_RAW),
            ("Isotonic (post-hoc)", p_iso, C_ISO),
            ("Sigmoid/Platt (post-hoc)", p_sig, C_SIG)]

rows = []
for name, p, _ in variants:
    rows.append({"Method": name, "Brier": round(brier_score_loss(yt, p), 4),
                 "ECE": round(ece(yt, p), 4)})
cal_tab = pd.DataFrame(rows)
print(cal_tab.to_string(index=False))

out_xlsx = os.path.join(BASE, "data", "04_results", "calibration.xlsx")
cal_tab.to_excel(out_xlsx, index=False)
print(f"\nSaved: {out_xlsx}")

# ---------------------------------------------------------------------------
# Figure 7 — reliability curve (A) + predicted-probability histogram (B)
# ---------------------------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(COL_DOUBLE, COL_DOUBLE * 0.42))

# Panel A: reliability curves
axA.plot([0, 1], [0, 1], ls="--", lw=0.9, color=C_REF, label="Perfect")
for name, p, c in variants:
    frac, mean_pred = calibration_curve(yt, p, n_bins=10, strategy="quantile")
    b = brier_score_loss(yt, p); e = ece(yt, p)
    axA.plot(mean_pred, frac, "o-", lw=1.8, ms=4, color=c,
             label=f"{name.split(' (')[0]} (Brier {b:.3f}, ECE {e:.3f})")
axA.set_xlabel("Mean predicted probability"); axA.set_ylabel("Observed frequency")
axA.set_xlim(0, 1); axA.set_ylim(0, 1)
axA.legend(loc="upper left", fontsize=7)
axA.set_title("A. Reliability curve")

# Panel B: histogram of raw predicted probabilities
axB.hist(p_other, bins=20, color=C_RAW, alpha=0.85)
axB.set_xlabel("Predicted probability (raw)"); axB.set_ylabel("Count")
axB.set_xlim(0, 1)
axB.set_title("B. Predicted probability distribution")
for s in ("top", "right"):
    axB.spines[s].set_visible(False)

save_figure(fig, "figure_07_calibration", outdir=FIG_DIR)
print("\nDone.")
