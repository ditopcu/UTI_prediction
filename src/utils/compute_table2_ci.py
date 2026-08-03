# -*- coding: utf-8 -*-
"""Bootstrap 95% CIs for the Table 2 rows that were reported as point estimates only:
  - Phase 1: model, concordant subset (n=1248)          -> ROC AUC, Sens, Spec, MCC
  - Phase 1: previous model (combined decision, n=1248)  -> Sens, Spec, MCC (no proba -> no ROC AUC)
  - Phase 2: post-implementation (routine use, n=72)     -> ROC AUC, Sens, Spec, MCC

Methodology is IDENTICAL to uti_16 (SEED=42, N_BOOT=2000, percentile 2.5/97.5).
Reconstructs the concordant subset exactly as uti_10; post-implementation from matched_cohort.xlsx.
Writes data/04_results/table2_ci.xlsx (consumed by build_all_tables_xlsx.py).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, recall_score, confusion_matrix,
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET = "CULTIVO_PATOLOGICO"
SEED = 42
N_BOOT = 2000


def point(yt, yp, pr=None):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {
        "ROC AUC": roc_auc_score(yt, pr) if pr is not None else np.nan,
        "Sensitivity": recall_score(yt, yp, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "MCC": matthews_corrcoef(yt, yp),
    }


def boot(yt, yp, pr=None, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(yt)
    keys = ["ROC AUC", "Sensitivity", "Specificity", "MCC"]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(yt[idx])) < 2:
            continue
        m = point(yt[idx], yp[idx], pr[idx] if pr is not None else None)
        for k in keys:
            acc[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            if len(v) else (np.nan, np.nan) for k, v in acc.items()}


def fmt(pm, ci, keys):
    out = {}
    for k in keys:
        if pr_na(pm, k):
            out[k] = "NA"
        else:
            out[k] = f"{pm[k]:.3f} ({ci[k][0]:.3f}-{ci[k][1]:.3f})"
    return out


def pr_na(pm, k):
    return k == "ROC AUC" and (pm[k] is None or (isinstance(pm[k], float) and np.isnan(pm[k])))


# ---------------------------------------------------------------------------
# Rows 3-4: reconstruct concordant subset exactly as uti_10
# ---------------------------------------------------------------------------
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1); y = df[TARGET]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_test, X_other, y_test, y_other = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)

model = CatBoostClassifier(); model.load_model(os.path.join(BASE, "models", "model_optuna.cbm"))
feats = model.feature_names_
hold = pd.DataFrame({
    "RAW_INDEX": ml.loc[X_other.index, "RAW_INDEX"].values,
    "y": y_other.values.astype(int),
    "opt_pred": model.predict(X_other[feats]).astype(int),
    "opt_proba": model.predict_proba(X_other[feats])[:, 1],
})
raw = pd.read_excel(os.path.join(BASE, "data", "01_raw", "uti_raw.xlsx")); raw["RAW_INDEX"] = raw.index + 1


def p_pred(x):
    s = str(x)
    if ("Alta probabilidad" in s) or ("Positivo" in s): return 1
    if ("Baja probabilidad" in s) or ("Negativo" in s): return 0
    return np.nan


raw["PRED_b"] = raw["PRED_CULT_IA"].map(p_pred)
hold = hold.merge(raw[["RAW_INDEX", "PRED_b"]], on="RAW_INDEX", how="left")
sub = hold[hold["PRED_b"].notna()].copy()
yt = sub["y"].astype(int).values

KEYS = ["ROC AUC", "Sensitivity", "Specificity", "MCC"]
rows = []

# Phase 1: model, concordant subset (has proba)
pm = point(yt, sub["opt_pred"].values, sub["opt_proba"].values)
ci = boot(yt, sub["opt_pred"].values, sub["opt_proba"].values)
rows.append(dict(Row="Phase 1: model, concordant subset", N=len(yt), **fmt(pm, ci, KEYS)))

# Phase 1: previous model (binary decision -> no proba -> ROC AUC = NA)
pm2 = point(yt, sub["PRED_b"].astype(int).values, None)
ci2 = boot(yt, sub["PRED_b"].astype(int).values, None)
rows.append(dict(Row="Phase 1: previous model (combined decision)", N=len(yt), **fmt(pm2, ci2, KEYS)))

# ---------------------------------------------------------------------------
# Row 8: post-implementation (matched_cohort.xlsx)
# ---------------------------------------------------------------------------
mc = pd.read_excel(os.path.join(BASE, "data", "05_prospective", "2026-06_postimpl", "matched_cohort.xlsx"))
ytp = mc[TARGET].astype(int).values
pmp = point(ytp, mc["pred"].astype(int).values, mc["prob"].astype(float).values)
cip = boot(ytp, mc["pred"].astype(int).values, mc["prob"].astype(float).values)
rows.append(dict(Row="Phase 2: post-implementation (routine use)", N=len(ytp), **fmt(pmp, cip, KEYS)))

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.to_string(index=False))

# sanity: point estimates vs the numbers already in Table 2
print("\n--- point-estimate check (should match existing Table 2) ---")
print(f"concordant model:  ROC {pm['ROC AUC']:.3f}  Sens {pm['Sensitivity']:.3f}  "
      f"Spec {pm['Specificity']:.3f}  MCC {pm['MCC']:.3f}   (table: 0.875/0.854/0.778/0.633)")
print(f"previous model:                Sens {pm2['Sensitivity']:.3f}  "
      f"Spec {pm2['Specificity']:.3f}  MCC {pm2['MCC']:.3f}   (table: 0.806/0.810/0.608)")
print(f"post-implementation: ROC {pmp['ROC AUC']:.3f}  Sens {pmp['Sensitivity']:.3f}  "
      f"Spec {pmp['Specificity']:.3f}  MCC {pmp['MCC']:.3f}   (table: 0.821/0.909/0.429/0.396)")

OUT = os.path.join(BASE, "data", "04_results", "table2_ci.xlsx")
res.to_excel(OUT, index=False)
print("\nSaved:", OUT)
