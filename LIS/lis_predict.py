"""
UTI prediction for a single patient from LIS data.
Loads the CatBoost+Optuna model once, exposes predict().
"""

import os
from catboost import CatBoostClassifier
from LIS.lis_preprocess import preprocess, PreprocessError

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_BASE_DIR, "models", "model_optuna.cbm")

# Load model at import time (once)
_model = CatBoostClassifier()
_model.load_model(MODEL_PATH)


def predict(patient: dict) -> dict:
    """
    Predict UTI for a single patient.

    Parameters
    ----------
    patient : dict
        Raw LIS values. Required keys:
        EDAD, SEXO, DENST, HEMATT, RBO, WBCO, EC, BACTS,
        LEUT, NITT, PROTT, BACT_INFO

    Returns
    -------
    dict with:
        prediction  : int   — 0 (negative) or 1 (positive)
        probability : float — P(positive culture), 0.0–1.0
        error       : str   — None if success, error message if failed
    """
    try:
        df = preprocess(patient)
        proba = _model.predict_proba(df)[0, 1]
        pred = int(proba >= 0.5)
        return {
            'prediction': pred,
            'probability': round(float(proba), 4),
            'error': None,
        }
    except PreprocessError as e:
        return {
            'prediction': None,
            'probability': None,
            'error': str(e),
        }
