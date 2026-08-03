"""
Model performance on the 48 MATCHED patients (those with an ITU_WS prediction),
using CULTIVO_PATOLOGICO as ground truth. Evaluates BOTH my CatBoost+Optuna model
and the hospital ITU_WS web service.

Run from project root:
    venv\\Scripts\\python.exe -m LIS.perf_matched
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, precision_score, recall_score, balanced_accuracy_score,
)

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ND = os.path.join(_BASE, "LIS", "new_data")
PRED = os.path.join(ND, "20260506_1506_predicted.xlsx")
DESC = os.path.join(ND, "descarga_new_ITU_0506_1506.xlsx")
MATCH = os.path.join(ND, "UTI_matchpatients.xlsx")
OUT = os.path.join(ND, "performance_matched_vs_culture.xlsx")

PETICION_COL = "Peticion n\xba"


def _parse_ws(s):
    m = re.match(r"(\d*\.?\d+)\-(\d)", str(s))
    return (float(m.group(1)), int(m.group(2))) if m else (None, None)


def metrics(y_true, y_pred, y_prob, label):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan      # recall / sensitivity
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan        # precision
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    return {
        "model": label,
        "n": len(y_true),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "sensitivity(recall)": sens,
        "specificity": spec,
        "PPV(precision)": ppv,
        "NPV": npv,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC_AUC": auc,
    }


def main():
    p = pd.read_excel(PRED)
    d = pd.read_excel(DESC)
    m = pd.read_excel(MATCH)

    d[["WS_PROB", "WS_PRED"]] = d["Prediccion"].apply(lambda s: pd.Series(_parse_ws(s)))
    bridge = (
        m[[PETICION_COL, "Id."]].dropna()
        .astype({PETICION_COL: "int64", "Id.": "int64"})
        .drop_duplicates(subset=PETICION_COL, keep="first")
        .rename(columns={PETICION_COL: "PETICIONCB", "Id.": "OrderID"})
    )
    d_sorted = d.sort_values("fecha").drop_duplicates(subset="OrderID", keep="last")
    ws = d_sorted[["OrderID", "WS_PROB", "WS_PRED"]].astype({"OrderID": "int64"})

    merged = p.merge(bridge, on="PETICIONCB", how="left").merge(ws, on="OrderID", how="left")
    matched = merged[merged["WS_PRED"].notna()].copy()
    matched["WS_PRED"] = matched["WS_PRED"].astype(int)

    print(f"Matched patients (have ITU_WS): {len(matched)}")

    # ground truth available?
    gt = matched.dropna(subset=["CULTIVO_PATOLOGICO"]).copy()
    gt["CULTIVO_PATOLOGICO"] = gt["CULTIVO_PATOLOGICO"].astype(int)
    print(f"  with culture result (CULTIVO_PATOLOGICO): {len(gt)}")
    print(f"  positive cultures: {int(gt['CULTIVO_PATOLOGICO'].sum())}, negative: {int((gt['CULTIVO_PATOLOGICO']==0).sum())}")
    print()

    y = gt["CULTIVO_PATOLOGICO"].values
    rows = [
        metrics(y, gt["PRED_UTI"].astype(int).values, gt["PROB_UTI"].values, "My CatBoost+Optuna"),
        metrics(y, gt["WS_PRED"].values, gt["WS_PROB"].values, "Hospital ITU_WS"),
    ]
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(res.to_string(index=False))
    print()

    with pd.ExcelWriter(OUT) as xl:
        res.to_excel(xl, sheet_name="metrics", index=False)
        show = ["FECHA", "PETICIONCB", "PACIENTE", "EDAD", "SEXO",
                "PROB_UTI", "PRED_UTI", "WS_PROB", "WS_PRED", "CULTIVO_PATOLOGICO"]
        show = [c for c in show if c in gt.columns]
        det = gt[show].copy()
        det["MINE_correct"] = det["PRED_UTI"].astype(int) == det["CULTIVO_PATOLOGICO"]
        det["WS_correct"] = det["WS_PRED"].astype(int) == det["CULTIVO_PATOLOGICO"]
        det.to_excel(xl, sheet_name="patient_detail", index=False)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
