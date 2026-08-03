# -*- coding: utf-8 -*-
"""
Composite MAIN-TEXT figures for the manuscript (Figures 2-4) and the two main tables.

Figure 1 is the study-overview schematic (built separately from HTML).
Everything else produced by uti_09-19 becomes supplementary material.

Outputs:
  figures/.../ figure_M2_discrimination
               figure_M3_vs_previous
               figure_M4_calibration_shap
  paper/tables_main.md   (Table 1 and Table 2, markdown for the Word build)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score,
    confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,
    matthews_corrcoef, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_DOUBLE, save_figure

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
TARGET = "CULTIVO_PATOLOGICO"

C_NEW = PALETTE["highlight"]; C_AG = PALETTE["base2"]; C_PREV = PALETTE["base1"]
C_NN = PALETTE["accent2"]; C_RF = PALETTE["accent3"]; C_REF = PALETTE["ci_grey"]

LABELS = {"DENST": "Specific gravity", "HEMATT": "Blood (Hb)", "RBO": "Red blood cells",
          "WBCO": "White blood cells", "EC": "Epithelial cells", "BACTS": "Bacteria",
          "SEXO_M": "Sex (male)", "LEUT_25": "Leuk. esterase 25", "LEUT_75": "Leuk. esterase 75",
          "LEUT_500": "Leuk. esterase 500", "NITT_1": "Nitrite positive",
          "PROTT_1": "Protein positive", "BACT_INFO_baja_1": "Sysmex Gram+ flag",
          "BACT_INFO_baja_2": "Sysmex Gram± flag", "BACT_INFO_baja_3": "Sysmex no-Gram-info",
          "EDAD_CATEGORICA_28-37": "Age 28-37"}

# ---------------------------------------------------------------------------
# Data + models
# ---------------------------------------------------------------------------
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1); y = df[TARGET]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, random_state=42)
X_te, X_ot, y_te, y_ot = train_test_split(X_tmp, y_tmp, test_size=0.375, random_state=42)

model = CatBoostClassifier(); model.load_model(os.path.join(BASE, "models", "model_optuna.cbm"))
feats = model.feature_names_
p_new = model.predict_proba(X_ot[feats])[:, 1]
pred_new = model.predict(X_ot[feats]).astype(int)
yt = y_ot.values.astype(int)

from autogluon.tabular import TabularPredictor  # noqa: E402
ag = TabularPredictor.load(os.path.join(BASE, "models", "autogluon_uti_ec"), verbosity=0)
v2 = pd.read_excel(os.path.join(BASE, "data", "02_interim", "uti_cleaned_v2.xlsx"))
raw = pd.read_excel(os.path.join(BASE, "data", "01_raw", "uti_raw.xlsx"))
safe = ["FECHA", "EDAD", "SEXO", "WBCO", "EC", "BACTS", "RBO", "PHT", "YLC", "CASTS"]
rd = raw.drop_duplicates(subset=safe, keep="first").copy(); rd["RAW_INDEX"] = rd.index + 1
v2 = v2.merge(rd[safe + ["RAW_INDEX"]], on=safe, how="left")
d = v2.copy(); d["NITT"] = d["NITT"].replace("Positivo", 1); d = d.drop(columns=["FILTER"]); d.dropna(inplace=True)
d = d.drop_duplicates(subset=[c for c in d.columns if c != "RAW_INDEX"], keep="first")
d = d.drop(columns=["FECHA", "XTAL", "UROT", "BILT", "CETOT"]); d = d[d["EDAD"] >= 18]
d["RBO"] = d["RBO"].replace(99999.0, np.nan); d.dropna(subset=["RBO"], inplace=True)
bs = list(range(18, 90, 10)); bins = bs + [90, int(d["EDAD"].max()) + 1]
d["EDAD_CATEGORICA"] = pd.cut(d["EDAD"], bins=bins, labels=[f"{i}-{i+9}" for i in bs] + [">=90"], right=False)
d["DENST"] = d["DENST"] / 1000
for c in ["PROTT", "CASTS", "YLC"]:
    d[c] = d[c].apply(lambda x: 0 if x == 0 else 1)
bo = ["DENST", "HEMATT", "RBO", "WBCO", "EC", "BACTS", "SEXO", "LEUT", "NITT", "PROTT",
      "BACT_INFO_baja", "EDAD_CATEGORICA"]
dag = d[bo + [TARGET, "RAW_INDEX"]].copy()
for c in ["SEXO", "LEUT", "NITT", "PROTT", "BACT_INFO_baja", "EDAD_CATEGORICA"]:
    dag[c] = dag[c].astype(str)
_, Xt_ag, _, yt_ag = train_test_split(dag.drop(columns=[TARGET]), dag[TARGET], test_size=0.40, random_state=42)
_, Xo_ag, _, yo_ag = train_test_split(Xt_ag, yt_ag, test_size=0.375, random_state=42)
other_ag = pd.concat([Xo_ag, yo_ag], axis=1).drop(columns=["RAW_INDEX"])
p_ag = ag.predict_proba(other_ag.drop(columns=[TARGET]))[1].values
y_ag = yo_ag.values.astype(int)

# previous model outputs on the hold-out
raw["RAW_INDEX"] = raw.index + 1
def p_rna(x):
    s = str(x); return 1 if s.startswith("Alta") else (0 if s.startswith("Baja") else np.nan)
def p_pr(x):
    s = str(x)
    if ("Alta probabilidad" in s) or ("Positivo" in s): return 1
    if ("Baja probabilidad" in s) or ("Negativo" in s): return 0
    return np.nan
raw["PREV"] = raw["PRED_CULT_IA"].map(p_pr); raw["RNA"] = raw["CDS_RNA"].map(p_rna)
raw["NN"] = pd.to_numeric(raw["UTI_CDS"], errors="coerce")
hold = pd.DataFrame({"RAW_INDEX": ml.loc[X_ot.index, "RAW_INDEX"].values, "y": yt,
                     "p": p_new, "pred": pred_new}).merge(
    raw[["RAW_INDEX", "PREV", "RNA", "NN"]], on="RAW_INDEX", how="left")
sub = hold.dropna(subset=["PREV"]).copy(); ys = sub["y"].values.astype(int)

# ---------------------------------------------------------------------------
# Figure M2 — discrimination (ROC + PR)
# ---------------------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(COL_DOUBLE, COL_DOUBLE * 0.44), constrained_layout=True)
f1_, t1_, _ = roc_curve(yt, p_new); f2_, t2_, _ = roc_curve(y_ag, p_ag)
a.plot(f1_, t1_, lw=2, color=C_NEW, label=f"CatBoost+Optuna (AUC {auc(f1_,t1_):.3f})")
a.plot(f2_, t2_, lw=1.8, color=C_AG, label=f"AutoGluon (AUC {auc(f2_,t2_):.3f})")
a.plot([0, 1], [0, 1], ls="--", lw=.8, color=C_REF)
a.set_xlabel("1 - Specificity"); a.set_ylabel("Sensitivity"); a.set_title("A. ROC", fontsize=10)
a.legend(loc="lower right", fontsize=7.5); a.set_xlim(-.01, 1.01); a.set_ylim(-.01, 1.01)
pr1, rc1, _ = precision_recall_curve(yt, p_new); pr2, rc2, _ = precision_recall_curve(y_ag, p_ag)
b.plot(rc1, pr1, lw=2, color=C_NEW, label=f"CatBoost+Optuna (AP {average_precision_score(yt,p_new):.3f})")
b.plot(rc2, pr2, lw=1.8, color=C_AG, label=f"AutoGluon (AP {average_precision_score(y_ag,p_ag):.3f})")
b.axhline(yt.mean(), ls="--", lw=.8, color=C_REF, label=f"Prevalence {yt.mean():.2f}")
b.set_xlabel("Recall"); b.set_ylabel("Precision"); b.set_title("B. Precision-Recall", fontsize=10)
b.legend(loc="lower left", fontsize=7.5); b.set_xlim(-.01, 1.01); b.set_ylim(0, 1.02)
save_figure(fig, "figure_M2_discrimination", outdir=FIG_DIR)

# ---------------------------------------------------------------------------
# Figure M3 — comparison with the previous model
# ---------------------------------------------------------------------------
fig, (a, b, c) = plt.subplots(1, 3, figsize=(COL_DOUBLE, COL_DOUBLE * 0.40),
                              constrained_layout=True)
cov = [100.0, hold["PREV"].notna().mean() * 100, hold["NN"].notna().mean() * 100,
       hold["RNA"].notna().mean() * 100]
a.bar(range(4), cov, color=[C_NEW, C_PREV, C_NN, C_RF], width=.65)
for i, v in enumerate(cov):
    a.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
a.set_xticks(range(4)); a.set_xticklabels(["CatBoost", "Combined", "NN", "RF"], fontsize=6.5)
a.set_ylabel("Episodes with a decision (%)"); a.set_ylim(0, 112); a.set_title("A. Coverage", fontsize=10)

names = ["Acc", "Sens", "Spec", "MCC"]
def mv(yp):
    tn, fp, fn, tp = confusion_matrix(ys, yp, labels=[0, 1]).ravel()
    return [accuracy_score(ys, yp), recall_score(ys, yp), tn / (tn + fp), matthews_corrcoef(ys, yp)]
m_new = mv(sub["pred"].values.astype(int)); m_prev = mv(sub["PREV"].values.astype(int))
xp = np.arange(4); w = .38
b.bar(xp - w/2, m_new, w, color=C_NEW, label="CatBoost+Optuna")
b.bar(xp + w/2, m_prev, w, color=C_PREV, label="Previous")
for i in range(4):
    b.text(xp[i]-w/2, m_new[i]+.015, f"{m_new[i]:.2f}", ha="center", fontsize=6.5)
    b.text(xp[i]+w/2, m_prev[i]+.015, f"{m_prev[i]:.2f}", ha="center", fontsize=6.5)
b.set_xticks(xp); b.set_xticklabels(names, fontsize=8); b.set_ylim(0, 1.05)
b.set_ylabel("Score"); b.legend(fontsize=6.5, loc="upper right", ncol=2,
                                columnspacing=.8, handlelength=1.2)
b.set_title(f"B. Head-to-head (n={len(sub)})", fontsize=10)

fr, tr, _ = roc_curve(ys, sub["p"].values); fn_, tn_, _ = roc_curve(ys, sub["NN"].values)
c.plot(fr, tr, lw=2, color=C_NEW, label=f"CatBoost+Optuna (AUC {auc(fr,tr):.3f})")
c.plot(fn_, tn_, lw=1.8, color=C_NN, label=f"Previous NN (AUC {auc(fn_,tn_):.3f})")
tn0, fp0, fn0, tp0 = confusion_matrix(ys, sub["PREV"].astype(int), labels=[0, 1]).ravel()
c.plot(fp0/(fp0+tn0), tp0/(tp0+fn0), "D", ms=7, color=C_PREV, label="Combined decision")
c.plot([0, 1], [0, 1], ls="--", lw=.8, color=C_REF)
c.set_xlabel("1 - Specificity"); c.set_ylabel("Sensitivity"); c.set_title("C. ROC", fontsize=10)
c.legend(loc="lower right", fontsize=6.8); c.set_xlim(-.01, 1.01); c.set_ylim(-.01, 1.01)
save_figure(fig, "figure_M3_vs_previous", outdir=FIG_DIR)

# ---------------------------------------------------------------------------
# Figure M4 — calibration + SHAP
# ---------------------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(COL_DOUBLE, COL_DOUBLE * 0.44), constrained_layout=True)
fr_, mp_ = calibration_curve(yt, p_new, n_bins=10, strategy="quantile")
a.plot([0, 1], [0, 1], ls="--", lw=.9, color=C_REF, label="Perfect")
a.plot(mp_, fr_, "o-", lw=1.8, ms=4, color=C_NEW,
       label=f"Model (Brier {brier_score_loss(yt,p_new):.3f})")
a.set_xlabel("Mean predicted probability"); a.set_ylabel("Observed frequency")
a.set_xlim(0, 1); a.set_ylim(0, 1); a.legend(loc="upper left", fontsize=7.5)
a.set_title("A. Calibration", fontsize=10)

sv = shap.TreeExplainer(model).shap_values(X_ot[feats])
imp = pd.Series(np.abs(sv).mean(0), index=[LABELS.get(f, f) for f in feats]).sort_values()
b.barh(range(len(imp)), imp.values, color=C_NEW, height=.72)
b.set_yticks(range(len(imp))); b.set_yticklabels(imp.index, fontsize=7)
b.set_xlabel("mean |SHAP|"); b.set_title("B. Feature contributions", fontsize=10)
save_figure(fig, "figure_M4_calibration_shap", outdir=FIG_DIR)

print("Main figures written.")
