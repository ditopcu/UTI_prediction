# -*- coding: utf-8 -*-
"""
Phase 3 — Implementation fidelity of the deployed CDS pipeline.

Verifies that the model running in production (Python service -> CDSS rule engine ->
LIS test results) reproduces the offline model. For the post-implementation episodes
that have both a live LIS result and the underlying urinalysis, the model probability
is recomputed offline with the deployment preprocessor and compared with the value
returned by the LIS.

Run from the project root:  python -m src.uti_17_implementation_fidelity
                       or:  python src/uti_17_implementation_fidelity.py

Outputs:
  - data/04_results/implementation_fidelity.xlsx
  - figures/.../ figure_14_implementation_fidelity
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)                      # so `LIS` package is importable
sys.path.insert(0, os.path.join(BASE, "plot_styles"))
from figure_style import set_style, PALETTE, COL_SINGLE, save_figure
from LIS.lis_predict import predict

set_style("tufte")
FIG_DIR = os.path.join(BASE, "figures")
DATA = os.path.join(BASE, "data", "05_prospective", "2026-06_postimpl")

C_PT = PALETTE["highlight"]
C_REF = PALETTE["ci_grey"]

REQUIRED = ["EDAD", "SEXO", "DENST", "HEMATT", "RBO", "WBCO", "EC", "BACTS",
            "LEUT", "NITT", "PROTT", "BACT_INFO"]

# ---------------------------------------------------------------------------
# Load deployed output + underlying urinalysis, join on petition number
# ---------------------------------------------------------------------------
cb = pd.read_excel(os.path.join(DATA, "catboost_live_predictions.xlsx"))
de = pd.read_excel(os.path.join(DATA, "culture_results.xlsx"))

prob_col = [c for c in cb.columns if c.startswith("Probabilidad")][0]
pred_col = [c for c in cb.columns if c.startswith("Predicci") and "Resultado" not in c][0]
pet_col = [c for c in cb.columns if "eticion" in c][0]

cb = cb[[pet_col, prob_col, pred_col]].rename(
    columns={pet_col: "PETICION", prob_col: "deployed_proba", pred_col: "deployed_pred"})
m = cb.merge(de.rename(columns={"PETICIONCB": "PETICION"}), on="PETICION", how="inner")
m = m.dropna(subset=["deployed_proba"]).reset_index(drop=True)
print(f"Episodes with both a live LIS result and the underlying urinalysis: n={len(m)}")

# ---------------------------------------------------------------------------
# Recompute offline with the deployment preprocessor + saved model
# ---------------------------------------------------------------------------
rows = []
for _, r in m.iterrows():
    patient = {f: r[f] for f in REQUIRED if f in m.columns}
    res = predict(patient)
    rows.append({
        "PETICION": r["PETICION"],
        "deployed_proba": float(r["deployed_proba"]),
        "deployed_pred": int(r["deployed_pred"]) if pd.notna(r["deployed_pred"]) else np.nan,
        "offline_proba": res["probability"],
        "offline_pred": res["prediction"],
        "error": res["error"],
    })
f = pd.DataFrame(rows)

ok = f[f["error"].isna()].copy()
n_err = int(f["error"].notna().sum())
ok["abs_diff"] = (ok["offline_proba"] - ok["deployed_proba"]).abs()
# the LIS reports the probability rounded to 2 decimals
ok["match_2dp"] = (ok["offline_proba"].round(2) == ok["deployed_proba"].round(2))
ok["pred_match"] = (ok["offline_pred"] == ok["deployed_pred"])

summary = pd.DataFrame([{
    "Episodes compared": len(ok),
    "Preprocessing errors": n_err,
    "Probability match (2 dp), n": int(ok["match_2dp"].sum()),
    "Probability match (2 dp), %": round(ok["match_2dp"].mean() * 100, 2),
    "Mean |difference|": round(ok["abs_diff"].mean(), 5),
    "Max |difference|": round(ok["abs_diff"].max(), 5),
    "Binary prediction concordance, %": round(ok["pred_match"].mean() * 100, 2),
}])
print(summary.to_string(index=False))
if n_err:
    print("\nErrors:")
    print(f[f["error"].notna()][["PETICION", "error"]].to_string(index=False))

out = os.path.join(BASE, "data", "04_results", "implementation_fidelity.xlsx")
with pd.ExcelWriter(out) as xw:
    summary.to_excel(xw, sheet_name="summary", index=False)
    f.to_excel(xw, sheet_name="per_episode", index=False)
print(f"\nSaved: {out}")

# ---------------------------------------------------------------------------
# Figure — deployed vs offline probability
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.95))
ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color=C_REF, label="Identity")
ax.scatter(ok["deployed_proba"], ok["offline_proba"], s=26, color=C_PT,
           alpha=0.8, edgecolor="white", linewidth=0.5,
           label=f"Episodes (n={len(ok)})")
ax.set_xlabel("Probability reported by the LIS")
ax.set_ylabel("Probability recomputed offline")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.legend(loc="upper left", fontsize=8)
save_figure(fig, "figure_14_implementation_fidelity", outdir=FIG_DIR)
print("Done.")
