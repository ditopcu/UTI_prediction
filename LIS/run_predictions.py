"""
Batch UTI prediction over an Excel file in LIS/new_data/.
Reads each row, builds the raw patient dict, calls LIS.lis_predict.predict(),
and writes results (prediction, probability, error) back to a new Excel file.

Run from the project root (UTI_Alicante):
    venv\\Scripts\\python.exe -m LIS.run_predictions
"""

import os
import math
import pandas as pd

from LIS.lis_predict import predict
from LIS.lis_preprocess import REQUIRED_FIELDS

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_DATA_DIR = os.path.join(_BASE_DIR, "LIS", "new_data")
INPUT_FILE = os.path.join(NEW_DATA_DIR, "20260506_1506.xlsx")
OUTPUT_FILE = os.path.join(NEW_DATA_DIR, "20260506_1506_predicted.xlsx")


def _clean(value):
    """Turn pandas NaN into None so validation reports it as a null field."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main():
    df = pd.read_excel(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {os.path.basename(INPUT_FILE)}")

    preds, probas, errors = [], [], []
    for _, row in df.iterrows():
        patient = {f: _clean(row.get(f)) for f in REQUIRED_FIELDS}
        result = predict(patient)
        preds.append(result["prediction"])
        probas.append(result["probability"])
        errors.append(result["error"])

    df["PRED_UTI"] = preds
    df["PROB_UTI"] = probas
    df["PRED_ERROR"] = errors

    df.to_excel(OUTPUT_FILE, index=False)

    n_ok = sum(e is None for e in errors)
    n_err = len(errors) - n_ok
    n_pos = sum(p == 1 for p in preds if p is not None)
    n_neg = sum(p == 0 for p in preds if p is not None)
    print(f"Done. Predicted: {n_ok} (pos={n_pos}, neg={n_neg}), errors: {n_err}")
    if n_err:
        from collections import Counter
        for msg, cnt in Counter(e for e in errors if e).most_common():
            print(f"  [{cnt}x] {msg}")
    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
