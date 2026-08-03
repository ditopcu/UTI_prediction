"""
Reproduce the exact CatBoost+Optuna training pipeline from UTI_ML_Pipeline_v2
and save the model as .cbm for LIS deployment.
"""

import os
import pandas as pd
import numpy as np
import json
import optuna
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from catboost import CatBoostClassifier
from boruta import BorutaPy

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
TARGET = 'CULTIVO_PATOLOGICO'

# Repository root, resolved from this file (src/utils/ -> repo root).
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get('UTI_DATA_DIR', os.path.join(BASE, 'data'))
MODEL_DIR = os.environ.get('UTI_MODEL_DIR', os.path.join(BASE, 'models'))

# ── 1. Load pre-encoded ML dataset ──────────────────────────────────
ml = pd.read_excel(f"{DATA_DIR}/03_processed/uti_ml_final.xlsx")
df_enc = ml.drop(columns=[c for c in ['ID', 'EDAD', 'RAW_INDEX'] if c in ml.columns])

X_enc = df_enc.drop(TARGET, axis=1)
y_enc = df_enc[TARGET]

print(f"Encoded dataset: {X_enc.shape}")
print(f"Features: {list(X_enc.columns)}")

# ── 2. Same split as notebook: 60/25/15 ─────────────────────────────
X_train_enc, X_temp_enc, y_train_enc, y_temp_enc = train_test_split(
    X_enc, y_enc, test_size=0.40, random_state=RANDOM_STATE)
X_test_enc, X_other_enc, y_test_enc, y_other_enc = train_test_split(
    X_temp_enc, y_temp_enc, test_size=0.375, random_state=RANDOM_STATE)

print(f"Train: {X_train_enc.shape}, Test: {X_test_enc.shape}, Other: {X_other_enc.shape}")

# ── 3. Boruta feature selection ─────────────────────────────────────
print("\n--- Boruta Feature Selection ---")
rf = RandomForestClassifier(
    n_jobs=-1, class_weight='balanced', max_depth=5, random_state=RANDOM_STATE)

feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=RANDOM_STATE)
feat_selector.fit(X_train_enc.values, y_train_enc.values)

selected_encoded = X_train_enc.columns[feat_selector.support_].tolist()
print(f"Boruta selected {len(selected_encoded)} features: {selected_encoded}")

# Apply selection
X_train_b = X_train_enc[selected_encoded]
X_test_b = X_test_enc[selected_encoded]
X_other_b = X_other_enc[selected_encoded]

# ── 4. Optuna hyperparameter optimization (30 trials) ───────────────
print("\n--- Optuna Optimization (30 trials) ---")

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'od_type': 'Iter', 'od_wait': 50,
        'random_seed': RANDOM_STATE, 'verbose': False
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train_b, y_train_enc,
              eval_set=(X_test_b, y_test_enc),
              early_stopping_rounds=50, verbose=False)
    return roc_auc_score(y_test_enc, model.predict_proba(X_test_b)[:, 1])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print(f"Best trial ROC AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# ── 5. Train final model with best params ───────────────────────────
print("\n--- Training Final Model ---")
best_params = study.best_params
best_params['random_seed'] = RANDOM_STATE
best_params['verbose'] = 100

model_optuna = CatBoostClassifier(**best_params)
model_optuna.fit(X_train_b, y_train_enc,
                 eval_set=(X_test_b, y_test_enc),
                 early_stopping_rounds=50)

# ── 6. Evaluate on Other (holdout) set ──────────────────────────────
y_pred_opt = model_optuna.predict(X_other_b)
y_proba_opt = model_optuna.predict_proba(X_other_b)[:, 1]

print("\n--- CatBoost + Optuna — Other Set ---")
print(classification_report(y_other_enc, y_pred_opt))
print(f"ROC AUC: {roc_auc_score(y_other_enc, y_proba_opt):.4f}")

# ── 7. Save model and metadata ──────────────────────────────────────
model_path = f"{MODEL_DIR}/model_optuna.cbm"
model_optuna.save_model(model_path)
print(f"\nSaved model to {model_path}")

meta = {
    'selected_encoded': selected_encoded,
    'best_params_optuna': {k: v for k, v in best_params.items()
                           if k not in ['verbose', 'random_seed']},
    'best_auc_optuna': study.best_value,
}
meta_path = f"{MODEL_DIR}/uti_models_meta.json"
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"Saved metadata to {meta_path}")

print(f"\nDone. Model ready for LIS deployment.")
