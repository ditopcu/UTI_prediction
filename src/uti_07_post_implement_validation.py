# -*- coding: utf-8 -*-
"""
Post-implementation prospective validation of the deployed CatBoost UTI model.

Joins the live model output log (`Catboost results`) with the culture-result
download (`Descarga orinas con cultivo`) on the lab petition number, then
evaluates the deployed CatBoost predictions against the culture ground truth
(CULTIVO_PATOLOGICO) on the matched cohort.

Outputs a single self-contained HTML report with embedded figures.
"""
import base64
import io
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    confusion_matrix, f1_score, matthews_corrcoef, brier_score_loss,
)
from sklearn.calibration import calibration_curve

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "05_prospective", "2026-06_postimpl")
CB_FILE = os.path.join(DATA, "catboost_live_predictions.xlsx")
DE_FILE = os.path.join(DATA, "culture_results.xlsx")
OUT_HTML = os.path.join(DATA, "validation_report.html")
OUT_XLSX = os.path.join(DATA, "matched_cohort.xlsx")
THRESHOLD = 0.50
SEED = 42

# Original UTI CatBoost+Optuna results on the Other/holdout set (from CLAUDE.md)
ORIGINAL = {"ROC AUC": 0.875, "F1": 0.826, "MCC": 0.625}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - half, center + half)


def bootstrap_ci(metric_fn, y, s, n_boot=2000, seed=SEED):
    """Percentile bootstrap 95% CI for a probability-based metric."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:  # need both classes
            continue
        vals.append(metric_fn(y[idx], s[idx]))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.1f}%"


# ---------------------------------------------------------------------------
# Load + join
# ---------------------------------------------------------------------------
cb = pd.read_excel(CB_FILE)
de = pd.read_excel(DE_FILE)

prob_col = [c for c in cb.columns if c.startswith("Probabilidad")][0]
pred_col = [c for c in cb.columns if c.startswith("Predicci") and "Resultado" not in c][0]
pet_col = [c for c in cb.columns if "eticion" in c][0]

cb_slim = cb[[pet_col, "Fecha", "Paciente", prob_col, pred_col]].rename(
    columns={pet_col: "PETICION", prob_col: "prob", pred_col: "pred"}
)

m = cb_slim.merge(de, left_on="PETICION", right_on="PETICIONCB", how="inner")
m = m.dropna(subset=["prob", "pred", "CULTIVO_PATOLOGICO"]).reset_index(drop=True)

y = m["CULTIVO_PATOLOGICO"].astype(int).values
p = m["prob"].astype(float).values
yhat = m["pred"].astype(int).values

n = len(m)
n_pos = int(y.sum())
n_neg = n - n_pos
prevalence = n_pos / n

m.to_excel(OUT_XLSX, index=False)

# ---------------------------------------------------------------------------
# Binary metrics @ 0.50
# ---------------------------------------------------------------------------
tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
sens = tp / (tp + fn) if (tp + fn) else np.nan          # recall / sensitivity
spec = tn / (tn + fp) if (tn + fp) else np.nan
ppv = tp / (tp + fp) if (tp + fp) else np.nan
npv = tn / (tn + fn) if (tn + fn) else np.nan
acc = (tp + tn) / n
bal_acc = (sens + spec) / 2
f1 = f1_score(y, yhat)
mcc = matthews_corrcoef(y, yhat)

binary_rows = [
    ("Sensitivity (Recall)", sens, wilson_ci(tp, tp + fn)),
    ("Specificity", spec, wilson_ci(tn, tn + fp)),
    ("PPV (Precision)", ppv, wilson_ci(tp, tp + fp)),
    ("NPV", npv, wilson_ci(tn, tn + fn)),
    ("Accuracy", acc, wilson_ci(tp + tn, n)),
    ("Balanced accuracy", bal_acc, None),
    ("F1 score", f1, None),
    ("MCC", mcc, None),
]

# ---------------------------------------------------------------------------
# Probability metrics
# ---------------------------------------------------------------------------
roc_auc = roc_auc_score(y, p)
pr_auc = average_precision_score(y, p)
roc_ci = bootstrap_ci(roc_auc_score, y, p)
pr_ci = bootstrap_ci(average_precision_score, y, p)
brier = brier_score_loss(y, p)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})

# ROC
fpr, tpr, _ = roc_curve(y, p)
fig = plt.figure(figsize=(5, 4.2))
plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"CatBoost (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1)
plt.xlabel("1 - Specificity (FPR)"); plt.ylabel("Sensitivity (TPR)")
plt.title("ROC Curve — Prospective cohort"); plt.legend(loc="lower right")
plt.grid(alpha=0.25)
roc_img = fig_to_b64(fig)

# PR
prec, rec, _ = precision_recall_curve(y, p)
fig = plt.figure(figsize=(5, 4.2))
plt.plot(rec, prec, color="#059669", lw=2, label=f"CatBoost (AP = {pr_auc:.3f})")
plt.axhline(prevalence, ls="--", color="#9ca3af", lw=1, label=f"Baseline = {prevalence:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curve"); plt.legend(loc="lower left")
plt.grid(alpha=0.25)
pr_img = fig_to_b64(fig)

# Confusion matrix
cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(4.2, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual (culture)")
ax.set_title(f"Confusion Matrix @ {THRESHOLD:.2f}")
cm_img = fig_to_b64(fig)

# Calibration
frac_pos, mean_pred = calibration_curve(y, p, n_bins=5, strategy="quantile")
fig = plt.figure(figsize=(5, 4.2))
plt.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1, label="Perfect calibration")
plt.plot(mean_pred, frac_pos, "o-", color="#dc2626", lw=2, label=f"CatBoost (Brier = {brier:.3f})")
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
plt.title("Calibration (reliability) curve"); plt.legend(loc="upper left")
plt.grid(alpha=0.25)
cal_img = fig_to_b64(fig)

# Probability distribution by class
fig = plt.figure(figsize=(5, 4.2))
bins = np.linspace(0, 1, 21)
plt.hist(p[y == 0], bins=bins, alpha=0.6, color="#3b82f6", label="Culture negative")
plt.hist(p[y == 1], bins=bins, alpha=0.6, color="#ef4444", label="Culture positive")
plt.axvline(THRESHOLD, ls="--", color="#111827", lw=1, label=f"Threshold {THRESHOLD:.2f}")
plt.xlabel("Predicted probability"); plt.ylabel("Count")
plt.title("Predicted probability by true class"); plt.legend()
dist_img = fig_to_b64(fig)

# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def metric_row(name, val, ci):
    ci_txt = "" if ci is None else f"<span class='ci'>[{pct(ci[0])} – {pct(ci[1])}]</span>"
    return f"<tr><td>{name}</td><td class='val'>{pct(val)}</td><td>{ci_txt}</td></tr>"

binary_html = "\n".join(metric_row(*r) for r in binary_rows)

def cmp_row(name, prosp, orig):
    delta = prosp - orig
    cls = "pos" if delta >= 0 else "neg"
    arrow = "▲" if delta >= 0 else "▼"
    return (f"<tr><td>{name}</td><td class='val'>{prosp:.3f}</td>"
            f"<td>{orig:.3f}</td><td class='{cls}'>{arrow} {delta:+.3f}</td></tr>")

cmp_html = "\n".join([
    cmp_row("ROC AUC", roc_auc, ORIGINAL["ROC AUC"]),
    cmp_row("F1 score", f1, ORIGINAL["F1"]),
    cmp_row("MCC", mcc, ORIGINAL["MCC"]),
])

html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Post-Implementation Validation — CatBoost UTI</title>
<style>
  :root {{ --bg:#f8fafc; --card:#ffffff; --border:#e2e8f0; --ink:#0f172a; --muted:#64748b; --accent:#2563eb; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--ink); margin:0; padding:32px; line-height:1.5; }}
  .wrap {{ max-width:1080px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  h2 {{ font-size:19px; margin:34px 0 14px; border-bottom:2px solid var(--border); padding-bottom:6px; }}
  .sub {{ color:var(--muted); margin:0 0 24px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:18px 0; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px; text-align:center; }}
  .card .big {{ font-size:28px; font-weight:700; color:var(--accent); }}
  .card .lbl {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card);
           border:1px solid var(--border); border-radius:12px; overflow:hidden; }}
  th,td {{ padding:10px 14px; text-align:left; border-bottom:1px solid var(--border); }}
  th {{ background:#f1f5f9; font-size:13px; text-transform:uppercase; letter-spacing:.03em; color:var(--muted); }}
  td.val {{ font-weight:700; }}
  .ci {{ color:var(--muted); font-size:13px; }}
  .pos {{ color:#059669; font-weight:600; }}
  .neg {{ color:#dc2626; font-weight:600; }}
  .figs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; margin-top:16px; }}
  .fig {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px; }}
  .fig img {{ width:100%; height:auto; display:block; }}
  .note {{ background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:14px 16px;
           font-size:14px; color:#713f12; margin-top:16px; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:40px; text-align:center; }}
</style></head>
<body><div class="wrap">
  <h1>Post-Implementation Prospective Validation</h1>
  <p class="sub">Deployed CatBoost UTI model vs. urine culture (CULTIVO_PATOLOGICO) &middot;
     operating threshold {THRESHOLD:.2f}</p>

  <div class="cards">
    <div class="card"><div class="big">{n}</div><div class="lbl">Matched patients</div></div>
    <div class="card"><div class="big">{n_pos}</div><div class="lbl">Culture positive</div></div>
    <div class="card"><div class="big">{n_neg}</div><div class="lbl">Culture negative</div></div>
    <div class="card"><div class="big">{prevalence*100:.0f}%</div><div class="lbl">Prevalence</div></div>
    <div class="card"><div class="big">{roc_auc:.3f}</div><div class="lbl">ROC AUC</div></div>
  </div>

  <h2>1. Discrimination (probability-based)</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>95% CI (bootstrap)</th></tr>
    <tr><td>ROC AUC</td><td class="val">{roc_auc:.3f}</td><td class="ci">[{roc_ci[0]:.3f} – {roc_ci[1]:.3f}]</td></tr>
    <tr><td>PR AUC (Average Precision)</td><td class="val">{pr_auc:.3f}</td><td class="ci">[{pr_ci[0]:.3f} – {pr_ci[1]:.3f}]</td></tr>
  </table>
  <div class="figs">
    <div class="fig"><img src="data:image/png;base64,{roc_img}"></div>
    <div class="fig"><img src="data:image/png;base64,{pr_img}"></div>
  </div>

  <h2>2. Classification @ threshold {THRESHOLD:.2f}</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>95% CI (Wilson)</th></tr>
    {binary_html}
  </table>
  <div class="figs">
    <div class="fig"><img src="data:image/png;base64,{cm_img}"></div>
    <div class="fig"><img src="data:image/png;base64,{dist_img}"></div>
  </div>

  <h2>3. Prospective vs. original holdout</h2>
  <table>
    <tr><th>Metric</th><th>Prospective (n={n})</th><th>Original holdout</th><th>Δ</th></tr>
    {cmp_html}
  </table>
  <div class="note"><b>Interpretation.</b> The original holdout metrics come from the development
     "Other" set (CatBoost+Optuna). Prospective prevalence is {prevalence*100:.0f}%
     ({n_pos}/{n}) vs. the development set's ~50%, which shifts threshold-dependent metrics.
     With n={n}, confidence intervals are wide — read deltas as directional, not definitive.</div>

  <h2>4. Calibration</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Brier score (lower is better)</td><td class="val">{brier:.3f}</td></tr>
  </table>
  <div class="figs">
    <div class="fig"><img src="data:image/png;base64,{cal_img}"></div>
  </div>

  <footer>Generated from <code>catboost_live_predictions.xlsx</code> &amp;
     <code>culture_results.xlsx</code> &middot;
     joined on lab petition number &middot; matched cohort n={n}.</footer>
</div></body></html>"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"Matched cohort: n={n}  (pos={n_pos}, neg={n_neg}, prevalence={prevalence:.1%})")
print(f"ROC AUC = {roc_auc:.3f}  95%CI [{roc_ci[0]:.3f}, {roc_ci[1]:.3f}]")
print(f"PR  AUC = {pr_auc:.3f}  95%CI [{pr_ci[0]:.3f}, {pr_ci[1]:.3f}]")
print(f"Sens={sens:.3f}  Spec={spec:.3f}  PPV={ppv:.3f}  NPV={npv:.3f}")
print(f"Acc={acc:.3f}  BalAcc={bal_acc:.3f}  F1={f1:.3f}  MCC={mcc:.3f}  Brier={brier:.3f}")
print(f"Confusion [tn fp / fn tp] = [{tn} {fp} / {fn} {tp}]")
print(f"Report : {OUT_HTML}")
print(f"Cohort : {OUT_XLSX}")
