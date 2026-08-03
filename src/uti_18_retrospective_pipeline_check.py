# -*- coding: utf-8 -*-
"""
Deployment-pipeline verification on RETROSPECTIVE data.

Isolates the question raised by the post-implementation fidelity check: is the
deployment preprocessor (LIS/lis_preprocess.py -> lis_predict.py) numerically
equivalent to the training-time encoding?

For the hold-out episodes, the model probability is computed twice:
  Path A (training encoding) : encoded matrix from data/03_processed -> model
  Path B (deployment path)   : raw LIS field values -> LIS.lis_predict.predict()

Here the inputs are guaranteed identical (same episodes, same source export), so any
disagreement is a preprocessing defect, not a data-timing effect.

Outputs:
  - data/04_results/retrospective_pipeline_check.xlsx
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from LIS.lis_predict import predict

TARGET = "CULTIVO_PATOLOGICO"
REQUIRED = ["EDAD", "SEXO", "DENST", "HEMATT", "RBO", "WBCO", "EC", "BACTS",
            "LEUT", "NITT", "PROTT", "BACT_INFO"]

# ---------------------------------------------------------------------------
# Path A — training encoding
# ---------------------------------------------------------------------------
ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1); y = df[TARGET]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_test, X_other, y_test, y_other = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)

model = CatBoostClassifier(); model.load_model(os.path.join(BASE, "models", "model_optuna.cbm"))
proba_A = model.predict_proba(X_other[model.feature_names_])[:, 1]
raw_index = ml.loc[X_other.index, "RAW_INDEX"].values

# ---------------------------------------------------------------------------
# Path B — deployment preprocessor on the raw LIS values
# ---------------------------------------------------------------------------
raw = pd.read_excel(os.path.join(BASE, "data", "01_raw", "uti_raw.xlsx"))
raw["RAW_INDEX"] = raw.index + 1
raw_idx = raw.set_index("RAW_INDEX")

rows = []
for ri, pa in zip(raw_index, proba_A):
    r = raw_idx.loc[ri]
    patient = {f: r[f] for f in REQUIRED}
    res = predict(patient)
    rows.append({"RAW_INDEX": ri, "proba_training_encoding": float(pa),
                 "proba_deployment_path": res["probability"], "error": res["error"]})

f = pd.DataFrame(rows)
ok = f[f["error"].isna()].copy()
n_err = int(f["error"].notna().sum())
ok["abs_diff"] = (ok["proba_deployment_path"] - ok["proba_training_encoding"]).abs()

summary = pd.DataFrame([{
    "Episodes compared": len(ok),
    "Preprocessing errors": n_err,
    "Identical to 4 dp, n": int((ok["abs_diff"] < 5e-5).sum()),
    "Identical to 4 dp, %": round((ok["abs_diff"] < 5e-5).mean() * 100, 2),
    "Within 0.01, %": round((ok["abs_diff"] <= 0.01).mean() * 100, 2),
    "Mean |difference|": round(ok["abs_diff"].mean(), 6),
    "Max |difference|": round(ok["abs_diff"].max(), 6),
}])
print(summary.to_string(index=False))
if n_err:
    print("\nError types:")
    print(f[f["error"].notna()]["error"].value_counts().head(10).to_string())
if (ok["abs_diff"] >= 5e-5).any():
    print("\nLargest disagreements:")
    print(ok.nlargest(5, "abs_diff").to_string(index=False))

out = os.path.join(BASE, "data", "04_results", "retrospective_pipeline_check.xlsx")
with pd.ExcelWriter(out) as xw:
    summary.to_excel(xw, sheet_name="summary", index=False)
    f.to_excel(xw, sheet_name="per_episode", index=False)
print(f"\nSaved: {out}")
