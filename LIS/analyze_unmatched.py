"""
Detailed analysis of patients that matched the bridge (got an Id./OrderID)
but have NO ITU_WS prediction in the descarga file.

Run from project root:
    venv\\Scripts\\python.exe -m LIS.analyze_unmatched
"""

import os
import re
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ND = os.path.join(_BASE, "LIS", "new_data")
PRED = os.path.join(ND, "20260506_1506_predicted.xlsx")
DESC = os.path.join(ND, "descarga_new_ITU_0506_1506.xlsx")
MATCH = os.path.join(ND, "UTI_matchpatients.xlsx")
OUT = os.path.join(ND, "unmatched_patients_analysis.xlsx")

PETICION_COL = "Peticion n\xba"


def _parse_ws(s):
    m = re.match(r"(\d*\.?\d+)\-(\d)", str(s))
    return (float(m.group(1)), int(m.group(2))) if m else (None, None)


def main():
    p = pd.read_excel(PRED)
    d = pd.read_excel(DESC)
    m = pd.read_excel(MATCH)

    d[["WS_PROB", "WS_PRED"]] = d["Prediccion"].apply(lambda s: pd.Series(_parse_ws(s)))

    bridge = (
        m[[PETICION_COL, "Id."]]
        .dropna()
        .astype({PETICION_COL: "int64", "Id.": "int64"})
        .drop_duplicates(subset=PETICION_COL, keep="first")
        .rename(columns={PETICION_COL: "PETICIONCB", "Id.": "OrderID"})
    )

    d_sorted = d.sort_values("fecha").drop_duplicates(subset="OrderID", keep="last")
    ws = d_sorted[["OrderID", "WS_PROB", "WS_PRED", "fecha"]].astype({"OrderID": "int64"})

    merged = p.merge(bridge, on="PETICIONCB", how="left").merge(ws, on="OrderID", how="left")

    unmatched = merged[merged["WS_PRED"].isna()].copy()
    matched = merged[merged["WS_PRED"].notna()].copy()

    print(f"Total my patients: {len(merged)}")
    print(f"  matched to ITU_WS  : {len(matched)}")
    print(f"  UNMATCHED (no WS)  : {len(unmatched)}")
    print()

    # Did the unmatched even get an OrderID from the bridge?
    no_order = unmatched["OrderID"].isna().sum()
    has_order = unmatched["OrderID"].notna().sum()
    print(f"Of the {len(unmatched)} unmatched:")
    print(f"  got an OrderID via bridge but NOT in descarga: {has_order}")
    print(f"  no OrderID at all (bridge gap)               : {no_order}")
    print()

    # Date distribution of unmatched vs descarga coverage
    unmatched["day"] = pd.to_datetime(unmatched["FECHA"]).dt.date
    d["day"] = pd.to_datetime(d["fecha"]).dt.date
    print("Unmatched patients per day:")
    print(unmatched["day"].value_counts().sort_index().to_string())
    print()

    # Is the OrderID range of unmatched outside descarga's coverage?
    if has_order:
        uo = unmatched["OrderID"].dropna()
        print(f"Unmatched OrderID range : {int(uo.min())} - {int(uo.max())}")
        print(f"descarga OrderID range  : {int(d['OrderID'].min())} - {int(d['OrderID'].max())}")
        in_range = ((uo >= d['OrderID'].min()) & (uo <= d['OrderID'].max())).sum()
        print(f"  unmatched OrderIDs within descarga range but absent: {in_range}")
    print()

    # Service / procedencia breakdown (clinical context)
    if "SERVICIO" in unmatched.columns:
        print("Unmatched by SERVICIO:")
        print(unmatched["SERVICIO"].value_counts().to_string())
    print()

    # ----- write Excel with two sheets -----
    cols = ["FECHA", "PETICIONCB", "OrderID", "PACIENTE", "EDAD", "SEXO",
            "SERVICIO", "PROB_UTI", "PRED_UTI", "CULTIVO_PATOLOGICO", "PRED_ERROR"]
    cols = [c for c in cols if c in unmatched.columns]

    summary = pd.DataFrame({
        "metric": [
            "my_total", "matched_to_ITU_WS", "unmatched",
            "unmatched_with_orderid_absent_in_descarga",
            "unmatched_no_orderid",
            "descarga_total_rows", "descarga_orderids_not_in_my_91",
        ],
        "value": [
            len(merged), len(matched), len(unmatched),
            int(has_order), int(no_order),
            len(d), int((~d["OrderID"].astype("int64").isin(bridge["OrderID"])).sum()),
        ],
    })

    with pd.ExcelWriter(OUT) as xl:
        unmatched[cols].to_excel(xl, sheet_name="unmatched_patients", index=False)
        unmatched["day"].value_counts().sort_index().rename_axis("day").reset_index(name="n").to_excel(
            xl, sheet_name="per_day", index=False)
        summary.to_excel(xl, sheet_name="summary", index=False)

    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
