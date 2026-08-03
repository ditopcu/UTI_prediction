# -*- coding: utf-8 -*-
"""
Phase 2 — Pre-implementation evaluation on new data (Wave 1, 5–15 June 2026).

Three analyses on the 91 consecutive episodes of the Wave-1 batch (all with a urine
culture):
  1. Model performance against culture on new, unseen data (bootstrap 95% CIs).
  2. Deployment verification — locally computed probabilities vs the probabilities
     returned by the deployed web service (ITU_WS), on the episodes matched through
     the bridge file.
  3. Comparison against the previously deployed hospital model (UTI_CDS / CDS_RNA /
     PRED_CULT_IA), which is available for a subset of the batch.

NOTE: `ITU_WS` is the web-service deployment of THIS model (its `Prediccion` field is
"<probability>-<binary>"), not the previous hospital model.

Outputs:
  - data/04_results/phase2_wave1.xlsx
  - figures/.../ figure_15_phase2_wave1
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, average_precision_score, confusion_matrix,
    accuracy_score, f1_score, matthews_corrcoef, recall_score, precision_score,
)

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_DOUBLE, save_figure

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
ND = os.path.join(BASE, "LIS", "new_data")
SEED, N_BOOT = 42, 2000

C_MODEL = PALETTE["highlight"]
C_WS = PALETTE["base1"]
C_PREV = PALETTE["accent2"]
C_REF = PALETTE["ci_grey"]

# ---------------------------------------------------------------------------
# Load: local predictions + web-service predictions + previous-model outputs
# ---------------------------------------------------------------------------
cmp_ = pd.read_excel(os.path.join(ND, "comparison_mymodel_vs_ITU_WS.xlsx"))
src = pd.read_excel(os.path.join(ND, "20260506_1506.xlsx"))

def p_cds_rna(x):
    s = str(x); return 1 if s.startswith("Alta") else (0 if s.startswith("Baja") else np.nan)
def p_pred(x):
    s = str(x)
    if ("Alta probabilidad" in s) or ("Positivo" in s): return 1
    if ("Baja probabilidad" in s) or ("Negativo" in s): return 0
    return np.nan
src["PREV_PRED"] = src["PRED_CULT_IA"].map(p_pred)
src["CDS_RNA_b"] = src["CDS_RNA"].map(p_cds_rna)
src["UTI_CDS_score"] = pd.to_numeric(src["UTI_CDS"], errors="coerce")

d = cmp_.merge(src[["PETICIONCB", "PREV_PRED", "CDS_RNA_b", "UTI_CDS_score"]],
               on="PETICIONCB", how="left")
d = d.dropna(subset=["CULTIVO_PATOLOGICO", "PROB_UTI"]).reset_index(drop=True)
y = d["CULTIVO_PATOLOGICO"].astype(int).values
print(f"Wave 1 batch: n={len(d)}  culture+ {int(y.sum())} ({y.mean()*100:.1f}%)")

# ---------------------------------------------------------------------------
# 1. Model performance vs culture (all episodes) with bootstrap CIs
# ---------------------------------------------------------------------------
def pmetrics(yt, yp, pr=None):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    m = {"Accuracy": accuracy_score(yt, yp), "Sensitivity": recall_score(yt, yp),
         "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
         "PPV": precision_score(yt, yp, zero_division=0),
         "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
         "F1": f1_score(yt, yp), "MCC": matthews_corrcoef(yt, yp)}
    if pr is not None:
        m["ROC AUC"] = roc_auc_score(yt, pr); m["PR AUC"] = average_precision_score(yt, pr)
    return m

def boot(yt, yp, pr=None):
    rng = np.random.default_rng(SEED); n = len(yt); acc = {}
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(yt[i])) < 2: continue
        for k, v in pmetrics(yt[i], yp[i], None if pr is None else pr[i]).items():
            acc.setdefault(k, []).append(v)
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in acc.items()}

def fmt(name, n, yt, yp, pr=None):
    pm = pmetrics(yt, yp, pr); ci = boot(yt, yp, pr)
    row = {"Analysis": name, "N": n}
    for k, v in pm.items():
        row[k] = f"{v:.3f} [{ci[k][0]:.3f}-{ci[k][1]:.3f}]"
    return row

perf = pd.DataFrame([fmt("Model vs culture (all Wave-1)", len(d), y,
                         d["PRED_UTI"].astype(int).values, d["PROB_UTI"].values)])
print("\n=== 1. Performance vs culture ===")
print(perf.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. Deployment verification: local vs deployed web service
# ---------------------------------------------------------------------------
mm = d.dropna(subset=["WS_PRED"]).copy()
mm["abs_diff"] = (mm["PROB_UTI"] - mm["WS_PROB"]).abs()
dep = pd.DataFrame([{
    "Episodes matched to the web service": len(mm),
    "Binary agreement, n": int((mm["PRED_UTI"] == mm["WS_PRED"]).sum()),
    "Binary agreement, %": round((mm["PRED_UTI"] == mm["WS_PRED"]).mean() * 100, 1),
    "Probability correlation (r)": round(mm["PROB_UTI"].corr(mm["WS_PROB"]), 3),
    "Mean |difference|": round(mm["abs_diff"].mean(), 4),
    "Max |difference|": round(mm["abs_diff"].max(), 4),
    "Within 0.01, %": round((mm["abs_diff"] <= 0.01).mean() * 100, 1),
}])
print("\n=== 2. Deployment verification (local vs web service) ===")
print(dep.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Comparison with the previously deployed model
# ---------------------------------------------------------------------------
cov = pd.DataFrame([{
    "Model (this study)": "100.0%",
    "PRED_CULT_IA (previous, deployed)": f"{d['PREV_PRED'].notna().mean()*100:.1f}%",
    "UTI_CDS (previous, NN)": f"{d['UTI_CDS_score'].notna().mean()*100:.1f}%",
    "CDS_RNA (previous, RF)": f"{d['CDS_RNA_b'].notna().mean()*100:.1f}%",
    "N": len(d),
}])
print("\n=== 3a. Decision coverage in the Wave-1 batch ===")
print(cov.to_string(index=False))

sub = d.dropna(subset=["PREV_PRED"]).copy()
ys = sub["CULTIVO_PATOLOGICO"].astype(int).values
h2h = pd.DataFrame([
    fmt("Model (concordant subset)", len(sub), ys, sub["PRED_UTI"].astype(int).values, sub["PROB_UTI"].values),
    fmt("PRED_CULT_IA (previous)", len(sub), ys, sub["PREV_PRED"].astype(int).values),
])
print("\n=== 3b. Head-to-head vs previous model (subset where it issues a decision) ===")
print(h2h.to_string(index=False))

out = os.path.join(BASE, "data", "04_results", "phase2_wave1.xlsx")
with pd.ExcelWriter(out) as xw:
    perf.to_excel(xw, sheet_name="performance", index=False)
    dep.to_excel(xw, sheet_name="deployment_check", index=False)
    cov.to_excel(xw, sheet_name="coverage", index=False)
    h2h.to_excel(xw, sheet_name="vs_previous", index=False)
    d.to_excel(xw, sheet_name="per_episode", index=False)
print(f"\nSaved: {out}")
# Phase 2 figures (Figure 5 = 2x2 pre/post ROC + confusion) are built from the saved result
# tables by src/uti_25_figure5_phase2.py; this script only produces the Wave-1 metrics.
print("Done.")
