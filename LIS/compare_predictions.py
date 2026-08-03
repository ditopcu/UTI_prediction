"""
Compare my CatBoost+Optuna predictions against the hospital ITU_WS web-service
predictions, matching patients through the UTI_matchpatients bridge file.

Join chain:
    predicted.PETICIONCB  ==  matchpatients['Peticion nº']
    matchpatients['Id.']  ==  descarga.OrderID

Run from project root:
    venv\\Scripts\\python.exe -m LIS.compare_predictions
"""

import os
import re
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ND = os.path.join(_BASE, "LIS", "new_data")
PRED = os.path.join(ND, "20260506_1506_predicted.xlsx")
DESC = os.path.join(ND, "descarga_new_ITU_0506_1506.xlsx")
MATCH = os.path.join(ND, "UTI_matchpatients.xlsx")
OUT = os.path.join(ND, "comparison_mymodel_vs_ITU_WS.xlsx")

PETICION_COL = "Peticion n\xba"   # 'Peticion nº'


def _parse_ws(s):
    """'0.9591-1' -> (0.9591, 1)."""
    m = re.match(r"(\d*\.?\d+)\-(\d)", str(s))
    return (float(m.group(1)), int(m.group(2))) if m else (None, None)


def main():
    p = pd.read_excel(PRED)
    d = pd.read_excel(DESC)
    m = pd.read_excel(MATCH)

    # --- parse ITU_WS prediction string ---
    d[["WS_PROB", "WS_PRED"]] = d["Prediccion"].apply(lambda s: pd.Series(_parse_ws(s)))

    # --- bridge: Peticion nº -> Id. (dedupe, keep first) ---
    bridge = (
        m[[PETICION_COL, "Id."]]
        .dropna()
        .astype({PETICION_COL: "int64", "Id.": "int64"})
        .drop_duplicates(subset=PETICION_COL, keep="first")
        .rename(columns={PETICION_COL: "PETICIONCB", "Id.": "OrderID"})
    )

    # --- descarga: one row per OrderID (latest prediction by fecha) ---
    d_sorted = d.sort_values("fecha").drop_duplicates(subset="OrderID", keep="last")
    ws = d_sorted[["OrderID", "WS_PROB", "WS_PRED", "fecha"]].astype({"OrderID": "int64"})

    # --- merge chain ---
    merged = p.merge(bridge, on="PETICIONCB", how="left")
    merged = merged.merge(ws, on="OrderID", how="left")

    matched = merged[merged["WS_PRED"].notna()].copy()
    matched["WS_PRED"] = matched["WS_PRED"].astype(int)

    n_total = len(p)
    n_bridge = merged["OrderID"].notna().sum()
    n_ws = len(matched)
    print(f"My patients: {n_total}")
    print(f"  -> matched to bridge (Id.): {n_bridge}")
    print(f"  -> matched to ITU_WS prediction: {n_ws}")
    print()

    # --- agreement between the two models ---
    agree = (matched["PRED_UTI"] == matched["WS_PRED"]).mean()
    print(f"Model agreement (my PRED_UTI vs ITU_WS): {agree:.3f} ({(matched['PRED_UTI']==matched['WS_PRED']).sum()}/{n_ws})")
    corr = matched["PROB_UTI"].corr(matched["WS_PROB"])
    print(f"Probability correlation (Pearson): {corr:.3f}")
    print()
    print("Cross-tab (rows=my pred, cols=ITU_WS pred):")
    print(pd.crosstab(matched["PRED_UTI"], matched["WS_PRED"], rownames=["MINE"], colnames=["ITU_WS"]))
    print()

    # --- vs ground truth where culture exists ---
    gt = matched.dropna(subset=["CULTIVO_PATOLOGICO"])
    if len(gt):
        my_acc = (gt["PRED_UTI"] == gt["CULTIVO_PATOLOGICO"]).mean()
        ws_acc = (gt["WS_PRED"] == gt["CULTIVO_PATOLOGICO"]).mean()
        print(f"Accuracy vs real culture (n={len(gt)}):")
        print(f"  My model : {my_acc:.3f}")
        print(f"  ITU_WS   : {ws_acc:.3f}")
    print()

    # --- save side-by-side ---
    cols = ["FECHA", "PETICIONCB", "PACIENTE", "EDAD", "SEXO",
            "PROB_UTI", "PRED_UTI", "WS_PROB", "WS_PRED", "CULTIVO_PATOLOGICO"]
    cols = [c for c in cols if c in matched.columns]
    out_df = merged[[c for c in cols if c in merged.columns]].copy()
    out_df["AGREE"] = out_df["PRED_UTI"] == out_df["WS_PRED"]
    out_df.to_excel(OUT, index=False)
    print(f"Saved side-by-side -> {OUT}")


if __name__ == "__main__":
    main()
