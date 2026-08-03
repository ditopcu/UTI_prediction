# -*- coding: utf-8 -*-
"""
Gray-zone stratified analysis (NO retraining): how the deployed CatBoost+Optuna model performs
on the hold-out set stratified by the PREVIOUS system's decision status.

Strata:
  - Overall hold-out
  - Concordant     : previous combined decision (PRED_CULT_IA) was issued
  - Non-concordant : previous system abstained (no combined decision) == the "gray zone"
       * Discordant : both component models (UTI_CDS, CDS_RNA) ran but DISAGREED  <-- emphasis
       * Component missing : >=1 component result not issued (not a hard case, just uncovered)

Point: extending full coverage to the gray zone costs only ~2 accuracy points overall, and that
cost is concentrated in the genuinely discordant cases (n=344).

Output: data/04_results/gray_zone_stratified.xlsx  (sheet: strata)
"""
import os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, accuracy_score
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "CULTIVO_PATOLOGICO"

ml = pd.read_excel(os.path.join(BASE, "data", "03_processed", "uti_ml_final.xlsx"))
df = ml.drop(columns=[c for c in ["ID", "EDAD", "RAW_INDEX"] if c in ml.columns])
X = df.drop(TARGET, axis=1); y = df[TARGET]
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.40, random_state=42)
Xte, Xoth, yte, yoth = train_test_split(Xtmp, ytmp, test_size=0.375, random_state=42)

m = CatBoostClassifier(); m.load_model(os.path.join(BASE, "models", "model_optuna.cbm"))
feats = m.feature_names_
hold = pd.DataFrame({
    "RAW_INDEX": ml.loc[Xoth.index, "RAW_INDEX"].values,
    "y": yoth.values.astype(int),
    "pred": m.predict(Xoth[feats]).astype(int),
    "proba": m.predict_proba(Xoth[feats])[:, 1],
})

raw = pd.read_excel(os.path.join(BASE, "data", "01_raw", "uti_raw.xlsx")); raw["RAW_INDEX"] = raw.index + 1
def p_rna(x): s = str(x); return 1 if s.startswith("Alta") else (0 if s.startswith("Baja") else np.nan)
def p_pred(x):
    s = str(x)
    if ("Alta probabilidad" in s) or ("Positivo" in s): return 1
    if ("Baja probabilidad" in s) or ("Negativo" in s): return 0
    return np.nan
def p_uti(x): return np.nan if pd.isna(x) else (1 if x >= 50 else 0)
raw["UTI_b"] = raw["UTI_CDS"].map(p_uti); raw["RNA_b"] = raw["CDS_RNA"].map(p_rna); raw["PRED_b"] = raw["PRED_CULT_IA"].map(p_pred)
hold = hold.merge(raw[["RAW_INDEX", "UTI_b", "RNA_b", "PRED_b"]], on="RAW_INDEX", how="left")
N = len(hold)

def stats(name, d):
    yt = d["y"].values; yp = d["pred"].values; pr = d["proba"].values
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {"Stratum": name, "n": len(d), "Prevalence (%)": round(100 * yt.mean(), 1),
            "Accuracy": round(accuracy_score(yt, yp), 3),
            "Sensitivity": round(tp / (tp + fn), 3) if (tp + fn) else np.nan,
            "Specificity": round(tn / (tn + fp), 3) if (tn + fp) else np.nan,
            "MCC": round(matthews_corrcoef(yt, yp), 3),
            "ROC AUC": round(roc_auc_score(yt, pr), 3)}

matched = hold[hold["PRED_b"].notna()]
nonc = hold[hold["PRED_b"].isna()]
discordant = nonc[nonc["UTI_b"].notna() & nonc["RNA_b"].notna()]
missing = nonc[nonc["UTI_b"].isna() | nonc["RNA_b"].isna()]

res = pd.DataFrame([
    stats("Overall hold-out", hold),
    stats("Concordant (previous decision issued)", matched),
    stats("Non-concordant (previous system abstained)", nonc),
    stats("Discordant (both components ran but disagreed)", discordant),
    stats("Component result missing (>=1 not issued)", missing),
])
print("Hold-out N =", N, "| coverage %.1f%% | non-concordant %.1f%%" % (100*len(matched)/N, 100*len(nonc)/N))
print(res.to_string(index=False))

out = os.path.join(BASE, "data", "04_results", "gray_zone_stratified.xlsx")
with pd.ExcelWriter(out) as xw:
    res.to_excel(xw, sheet_name="strata", index=False)
print("\nSaved:", out)
