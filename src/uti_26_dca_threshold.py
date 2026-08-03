# -*- coding: utf-8 -*-
"""
Decision-curve (net benefit) and threshold analysis for the deployed model, to support the
Discussion's rule-out / actionability claims. NO retraining.

Cohorts:
  - Hold-out (development, n=2248), reconstructed from the saved model.
  - Post-implementation routine use (n=190), from prospective_2026-07-23.xlsx (per_sample).

Outputs:
  data/04_results/dca_threshold.xlsx  (threshold sweeps + net-benefit curves + key thresholds)
"""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "CULTIVO_PATOLOGICO"

# --- hold-out (reconstruct) ---
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1); y = df[TARGET]
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, random_state=42)
X_te, X_oth, y_te, y_oth = train_test_split(X_tmp, y_tmp, test_size=0.375, random_state=42)
model = CatBoostClassifier(); model.load_model(os.path.join(BASE, "models", "model_optuna.cbm"))
feats = model.feature_names_
hold = pd.DataFrame({"y": y_oth.values.astype(int),
                     "p": model.predict_proba(X_oth[feats])[:, 1]})

# --- post-implementation ---
post = pd.read_excel(os.path.join(BASE, "data", "04_results", "prospective_2026-07-23.xlsx"),
                     sheet_name="per_sample")[["CULTIVO_PATOLOGICO", "prob_live"]]
post = post.rename(columns={"CULTIVO_PATOLOGICO": "y", "prob_live": "p"})
post["y"] = post["y"].astype(int)

def sweep(d, name):
    y = d["y"].values; p = d["p"].values; n = len(y); prev = y.mean()
    rows = []
    for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
        yp = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        neg_rate = (yp == 0).mean()   # proportion the model would "clear" (rule-out candidates)
        rows.append({"cohort": name, "threshold": thr, "n": n, "prevalence": round(prev, 3),
                     "sensitivity": round(sens, 3), "specificity": round(spec, 3),
                     "PPV": round(ppv, 3), "NPV": round(npv, 3),
                     "predicted_negative_%": round(neg_rate * 100, 1), "FN": int(fn), "TN": int(tn)})
    return pd.DataFrame(rows)

def net_benefit(d, name):
    y = d["y"].values; p = d["p"].values; n = len(y); prev = y.mean()
    rows = []
    for pt in np.round(np.arange(0.05, 0.96, 0.05), 2):
        yp = (p >= pt).astype(int)
        tp = int(((yp == 1) & (y == 1)).sum()); fp = int(((yp == 1) & (y == 0)).sum())
        w = pt / (1 - pt)
        nb_model = tp / n - (fp / n) * w
        nb_all = prev - (1 - prev) * w
        rows.append({"cohort": name, "threshold_prob": pt,
                     "NB_model": round(nb_model, 4), "NB_treat_all": round(nb_all, 4),
                     "NB_treat_none": 0.0,
                     "model_beats_all": nb_model > nb_all, "model_beats_none": nb_model > 0})
    return pd.DataFrame(rows)

def key_thresholds(d, name):
    y = d["y"].values; p = d["p"].values
    out = {}
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    best_npv95 = None; best_sens95 = None
    for thr in grid:
        yp = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        npv = tn / (tn + fn) if (tn + fn) else 0
        neg = (yp == 0).mean() * 100
        # lowest threshold still giving NPV>=0.95 (max cleared while safe)
        if npv >= 0.95 and (tn + fn) > 0:
            if best_npv95 is None or neg > best_npv95["cleared_%"]:
                best_npv95 = {"threshold": thr, "NPV": round(npv, 3), "sensitivity": round(sens, 3),
                              "specificity": round(spec, 3), "cleared_%": round(neg, 1)}
        if sens >= 0.95:
            best_sens95 = {"threshold": thr, "sensitivity": round(sens, 3), "specificity": round(spec, 3),
                           "NPV": round(npv, 3), "cleared_%": round(neg, 1)}
    out["NPV>=0.95 (best culture-reduction)"] = best_npv95 or "not achievable at any threshold"
    out["sens>=0.95 (highest threshold)"] = best_sens95 or "not achievable"
    return {name: out}

sw = pd.concat([sweep(hold, "hold-out"), sweep(post, "post-impl")], ignore_index=True)
nb = pd.concat([net_benefit(hold, "hold-out"), net_benefit(post, "post-impl")], ignore_index=True)

print("=== PREVALENCE === hold-out %.3f | post-impl %.3f" % (hold.y.mean(), post.y.mean()))
print("\n=== KEY THRESHOLDS (post-impl) ==="); print(key_thresholds(post, "post-impl"))
print("\n=== KEY THRESHOLDS (hold-out) ==="); print(key_thresholds(hold, "hold-out"))
print("\n=== POST-IMPL threshold sweep ==="); print(sweep(post, "post-impl").to_string(index=False))
print("\n=== POST-IMPL net benefit (does model beat treat-all / treat-none?) ===")
print(net_benefit(post, "post-impl").to_string(index=False))

out = os.path.join(BASE, "data", "04_results", "dca_threshold.xlsx")
with pd.ExcelWriter(out) as xw:
    sw.to_excel(xw, sheet_name="threshold_sweep", index=False)
    nb.to_excel(xw, sheet_name="net_benefit", index=False)
print("\nSaved:", out)

# ---------------------------------------------------------------------------
# Supplement figure: decision curve (net benefit) for the post-implementation cohort
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_SINGLE, save_figure
set_style("tufte")
nbp = net_benefit(post, "post-impl")
x = nbp["threshold_prob"].values
fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.85))
ax.plot(x, nbp["NB_treat_all"].values, lw=1.5, ls="--", color=PALETTE["base1"], label="Treat all")
ax.axhline(0, lw=1.2, ls=":", color=PALETTE["ci_grey"], label="Treat none")
ax.plot(x, nbp["NB_model"].values, lw=2.2, color=PALETTE["highlight"], label="Deployed model")
ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
ax.set_xlim(0.05, 0.80); ax.set_ylim(-0.10, float(nbp["NB_model"].max()) * 1.12)
ax.legend(loc="upper right", fontsize=8)
ax.set_title(f"Post-implementation (n={len(post)})", fontsize=10)
save_figure(fig, "figure_18_postimpl_dca", outdir=os.path.join(BASE, "figures"))
print("Figure: figure_18_postimpl_dca (supplement)")
