# -*- coding: utf-8 -*-
"""
Phase 2 main figures, built from the saved result tables (no recomputation):
  Figure 5 (2x2): A pre-implementation ROC (Wave 1, n=91) with the previously deployed
                  operating point; B post-implementation ROC (n=190);
                  C pre-implementation confusion matrix; D post-implementation confusion matrix.
  Figure 6:       post-implementation calibration (reliability curve, Brier, ECE).

Sources:
  data/04_results/phase2_wave1.xlsx        (sheet per_episode: PROB_UTI, PRED_UTI, CULTIVO_PATOLOGICO, PREV_PRED)
  data/04_results/prospective_2026-07-23.xlsx (sheet per_sample: prob_live, pred_live, CULTIVO_PATOLOGICO)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, brier_score_loss

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_SINGLE, COL_DOUBLE, save_figure

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
RES = os.path.join(BASE, "data", "04_results")
C_MODEL, C_PREV, C_REF, C_TXT = PALETTE["highlight"], PALETTE["base1"], PALETTE["ci_grey"], PALETTE["text"]
THR = 0.50

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
w = pd.read_excel(os.path.join(RES, "phase2_wave1.xlsx"), sheet_name="per_episode")
y1 = w["CULTIVO_PATOLOGICO"].astype(int).values
p1 = w["PROB_UTI"].astype(float).values
prev = w.dropna(subset=["PREV_PRED"])
yp1, prevb = prev["CULTIVO_PATOLOGICO"].astype(int).values, prev["PREV_PRED"].astype(int).values

d = pd.read_excel(os.path.join(RES, "prospective_2026-07-23.xlsx"), sheet_name="per_sample")
y2 = d["CULTIVO_PATOLOGICO"].astype(int).values
p2 = d["prob_live"].astype(float).values
yp2 = d["pred_live"].astype(int).values


def draw_cm(ax, yt, yp, title):
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    ax.imshow(cm, cmap="Reds")
    t = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > t else C_TXT)
    ax.set_xticks([0, 1], ["Pred.\nnegative", "Pred.\npositive"], fontsize=8)
    ax.set_yticks([0, 1], ["Culture\nnegative", "Culture\npositive"], fontsize=8)
    ax.set_title(title, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------------------
# Figure 5 (2x2)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(COL_DOUBLE, COL_DOUBLE * 0.98), constrained_layout=True)

# A: pre ROC + previous operating point
f1, t1, _ = roc_curve(y1, p1)
ax[0, 0].plot(f1, t1, lw=2, color=C_MODEL, label=f"Model (AUC {auc(f1, t1):.3f}, n={len(y1)})")
tn, fp, fn, tp = confusion_matrix(yp1, prevb, labels=[0, 1]).ravel()
ax[0, 0].plot(fp / (fp + tn), tp / (tp + fn), "D", ms=8, color=C_PREV,
              label=f"Previous model (n={len(yp1)})")
ax[0, 0].plot([0, 1], [0, 1], ls="--", lw=0.8, color=C_REF)
ax[0, 0].set_xlabel("1 - Specificity"); ax[0, 0].set_ylabel("Sensitivity")
ax[0, 0].set_xlim(-.01, 1.01); ax[0, 0].set_ylim(-.01, 1.01)
ax[0, 0].legend(loc="lower right", fontsize=7.5)
ax[0, 0].set_title("A. Pre-implementation ROC", fontsize=10)

# B: post ROC
f2, t2, _ = roc_curve(y2, p2)
ax[0, 1].plot(f2, t2, lw=2, color=C_MODEL, label=f"Model (AUC {auc(f2, t2):.3f}, n={len(y2)})")
ax[0, 1].plot([0, 1], [0, 1], ls="--", lw=0.8, color=C_REF)
ax[0, 1].set_xlabel("1 - Specificity"); ax[0, 1].set_ylabel("Sensitivity")
ax[0, 1].set_xlim(-.01, 1.01); ax[0, 1].set_ylim(-.01, 1.01)
ax[0, 1].legend(loc="lower right", fontsize=7.5)
ax[0, 1].set_title("B. Post-implementation ROC", fontsize=10)

# C / D: confusion matrices
draw_cm(ax[1, 0], y1, w["PRED_UTI"].astype(int).values, f"C. Pre-implementation (n={len(y1)})")
draw_cm(ax[1, 1], y2, yp2, f"D. Post-implementation (n={len(y2)})")

save_figure(fig, "figure_15_phase2_evaluation", outdir=FIG_DIR)

# ---------------------------------------------------------------------------
# Figure 6: post-implementation calibration
# ---------------------------------------------------------------------------
nb = 10
bins = np.linspace(0, 1, nb + 1)
xs, ys = [], []
for i in range(nb):
    m = (p2 > bins[i]) & (p2 <= bins[i + 1]) if i else (p2 >= bins[i]) & (p2 <= bins[i + 1])
    if m.sum():
        xs.append(p2[m].mean()); ys.append(y2[m].mean())
brier = brier_score_loss(y2, p2)
ece = sum((((p2 > bins[i]) & (p2 <= bins[i + 1]) if i else (p2 >= bins[i]) & (p2 <= bins[i + 1])).mean()
           * abs(y2[m2].mean() - p2[m2].mean()))
          for i in range(nb)
          for m2 in [((p2 > bins[i]) & (p2 <= bins[i + 1]) if i else (p2 >= bins[i]) & (p2 <= bins[i + 1]))]
          if m2.sum())

fig, axc = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.95))
axc.plot([0, 1], [0, 1], ls="--", lw=0.9, color=C_REF, label="Perfect")
axc.plot(xs, ys, "o-", lw=1.8, ms=5, color=C_MODEL,
         label=f"Model (Brier {brier:.3f}, ECE {ece:.3f})")
axc.set_xlabel("Predicted probability"); axc.set_ylabel("Observed frequency")
axc.set_xlim(0, 1); axc.set_ylim(0, 1)
axc.legend(loc="upper left", fontsize=8)
save_figure(fig, "figure_16_postimpl_calibration", outdir=FIG_DIR)

print("Saved figure_15_phase2_evaluation (Fig 5, 2x2) and figure_16_postimpl_calibration (Fig 6).")
print(f"post-impl calibration: Brier {brier:.3f}, ECE {ece:.3f}")
