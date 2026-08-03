# ============================================================
# UTI Model Evaluation: SHAP, ROC, PR, Confusion Matrix
# CatBoost+Optuna vs AutoGluon (Other/Holdout set)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,
    classification_report
)
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 11,
})
FIG_DIR = 'figures'

# ============================================================
# 1. DATA PREP & MODEL TRAINING (CatBoost)
# ============================================================
print("=" * 60)
print("1. Veri hazırlık ve CatBoost eğitimi")
print("=" * 60)

ml = pd.read_excel('data/03_processed/uti_ml_final.xlsx')
df = ml.drop(columns=[c for c in ['ID', 'EDAD', 'RAW_INDEX'] if c in ml.columns])

target = 'CULTIVO_PATOLOGICO'
X = df.drop(target, axis=1)
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_test, X_other, y_test, y_other = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)

# Boruta
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
feat_selector.fit(X_train.values, y_train.values)
selected = X_train.columns[feat_selector.support_].tolist()
print(f"Boruta selected: {len(selected)} features")

X_train_b, X_test_b, X_other_b = X_train[selected], X_test[selected], X_other[selected]

# Optuna
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'od_type': 'Iter', 'od_wait': 50, 'random_seed': 42, 'verbose': False
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train_b, y_train, eval_set=(X_test_b, y_test), early_stopping_rounds=50, verbose=False)
    return roc_auc_score(y_test, model.predict_proba(X_test_b)[:, 1])

print("Running Optuna (30 trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

best_params = study.best_params
best_params['random_seed'] = 42
best_params['verbose'] = 0
model_cb = CatBoostClassifier(**best_params)
model_cb.fit(X_train_b, y_train, eval_set=(X_test_b, y_test), early_stopping_rounds=50, verbose=0)
print(f"CatBoost+Optuna trained. Best ROC AUC (test): {study.best_value:.4f}")

# CatBoost predictions on Other set
y_pred_cb = model_cb.predict(X_other_b)
y_proba_cb = model_cb.predict_proba(X_other_b)[:, 1]

# ============================================================
# 2. AUTOGLUON PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("2. AutoGluon yükleme ve tahmin")
print("=" * 60)

from autogluon.tabular import TabularPredictor
ag = TabularPredictor.load('models/autogluon_uti_ec', verbosity=0)

# AutoGluon Other seti hazırla (unscaled)
v2 = pd.read_excel('data/02_interim/uti_cleaned_v2.xlsx')
raw = pd.read_excel('data/01_raw/uti_raw.xlsx')
safe_cols = ['FECHA', 'EDAD', 'SEXO', 'WBCO', 'EC', 'BACTS', 'RBO', 'PHT', 'YLC', 'CASTS']
raw_dedup = raw.drop_duplicates(subset=safe_cols, keep='first').copy()
raw_dedup['RAW_INDEX'] = raw_dedup.index + 1
v2 = v2.merge(raw_dedup[safe_cols + ['RAW_INDEX']], on=safe_cols, how='left')
dfv = v2.copy()
dfv['NITT'] = dfv['NITT'].replace('Positivo', 1)
dfv = dfv.drop(columns=['FILTER'])
dfv.dropna(inplace=True)
orig_cols_v = [c for c in dfv.columns if c != 'RAW_INDEX']
dfv = dfv.drop_duplicates(subset=orig_cols_v, keep='first')
dfv = dfv.drop(columns=['FECHA', 'XTAL', 'UROT', 'BILT', 'CETOT'])
dfv = dfv[dfv['EDAD'] >= 18]
dfv['RBO'] = dfv['RBO'].replace(99999.0, np.nan)
dfv.dropna(subset=['RBO'], inplace=True)
bins_s = list(range(18, 90, 10))
bins = bins_s + [90, int(dfv['EDAD'].max()) + 1]
labels = [f'{i}-{i+9}' for i in bins_s] + ['>=90']
dfv['EDAD_CATEGORICA'] = pd.cut(dfv['EDAD'], bins=bins, labels=labels, right=False)
dfv['DENST'] = dfv['DENST'] / 1000
for col in ['PROTT', 'CASTS', 'YLC']:
    dfv[col] = dfv[col].apply(lambda x: 0 if x == 0 else 1)

boruta_orig = ['DENST', 'HEMATT', 'RBO', 'WBCO', 'EC', 'BACTS',
               'SEXO', 'LEUT', 'NITT', 'PROTT', 'BACT_INFO_baja', 'EDAD_CATEGORICA']
df_ag = dfv[boruta_orig + ['CULTIVO_PATOLOGICO', 'RAW_INDEX']].copy()
for col in ['SEXO', 'LEUT', 'NITT', 'PROTT', 'BACT_INFO_baja', 'EDAD_CATEGORICA']:
    df_ag[col] = df_ag[col].astype(str)

X_ag = df_ag.drop(columns=['CULTIVO_PATOLOGICO'])
y_ag = df_ag['CULTIVO_PATOLOGICO']
_, X_temp_ag, _, y_temp_ag = train_test_split(X_ag, y_ag, test_size=0.40, random_state=42)
_, X_other_ag, _, y_other_ag = train_test_split(X_temp_ag, y_temp_ag, test_size=0.375, random_state=42)

other_ag = pd.concat([X_other_ag, y_other_ag], axis=1).drop(columns=['RAW_INDEX'])

y_pred_ag = ag.predict(other_ag.drop(columns=[target]))
y_proba_ag = ag.predict_proba(other_ag.drop(columns=[target]))[1]
y_other_ag_vals = y_other_ag.values

print(f"AutoGluon best model: {ag.model_best}")
print(f"Predictions ready. Other set size: {len(y_other_ag_vals)}")

# ============================================================
# 3. ROC CURVES (Combined)
# ============================================================
print("\n" + "=" * 60)
print("3. ROC Curves")
print("=" * 60)

fig, ax = plt.subplots(1, 1, figsize=(8, 7))

# CatBoost
fpr_cb, tpr_cb, _ = roc_curve(y_other, y_proba_cb)
auc_cb = auc(fpr_cb, tpr_cb)
ax.plot(fpr_cb, tpr_cb, lw=2.5, label=f'CatBoost+Optuna (AUC={auc_cb:.4f})', color='#2196F3')

# AutoGluon
fpr_ag, tpr_ag, _ = roc_curve(y_other_ag_vals, y_proba_ag)
auc_ag = auc(fpr_ag, tpr_ag)
ax.plot(fpr_ag, tpr_ag, lw=2.5, label=f'AutoGluon Ensemble (AUC={auc_ag:.4f})', color='#FF5722')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve — Other Set (Holdout)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
plt.savefig(f'{FIG_DIR}/01_roc_curve_comparison.png')
plt.close()
print(f"  Saved: {FIG_DIR}/01_roc_curve_comparison.png")

# ============================================================
# 4. PR CURVES (Combined)
# ============================================================
print("4. Precision-Recall Curves")

fig, ax = plt.subplots(1, 1, figsize=(8, 7))

prec_cb, rec_cb, _ = precision_recall_curve(y_other, y_proba_cb)
ap_cb = average_precision_score(y_other, y_proba_cb)
ax.plot(rec_cb, prec_cb, lw=2.5, label=f'CatBoost+Optuna (AP={ap_cb:.4f})', color='#2196F3')

prec_ag, rec_ag, _ = precision_recall_curve(y_other_ag_vals, y_proba_ag)
ap_ag = average_precision_score(y_other_ag_vals, y_proba_ag)
ax.plot(rec_ag, prec_ag, lw=2.5, label=f'AutoGluon Ensemble (AP={ap_ag:.4f})', color='#FF5722')

prevalence = y_other.mean()
ax.axhline(y=prevalence, color='gray', linestyle='--', lw=1, alpha=0.5, label=f'Baseline (prevalence={prevalence:.2f})')

ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision-Recall Curve — Other Set (Holdout)', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([0.0, 1.05])
plt.savefig(f'{FIG_DIR}/02_pr_curve_comparison.png')
plt.close()
print(f"  Saved: {FIG_DIR}/02_pr_curve_comparison.png")

# ============================================================
# 5. CONFUSION MATRICES (Side by Side)
# ============================================================
print("5. Confusion Matrices")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, y_true, y_pred, title in [
    (axes[0], y_other, y_pred_cb, 'CatBoost+Optuna'),
    (axes[1], y_other_ag_vals, y_pred_ag, 'AutoGluon Ensemble'),
]:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negatif (0)', 'Pozitif (1)'],
                yticklabels=['Negatif (0)', 'Pozitif (1)'],
                annot_kws={'size': 14})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'{title}\n(Other Set)', fontsize=13, fontweight='bold')

plt.suptitle('Confusion Matrix Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/03_confusion_matrix_comparison.png')
plt.close()
print(f"  Saved: {FIG_DIR}/03_confusion_matrix_comparison.png")

# ============================================================
# 6. SHAP - CatBoost+Optuna
# ============================================================
print("\n" + "=" * 60)
print("6. SHAP Analysis — CatBoost+Optuna")
print("=" * 60)

explainer_cb = shap.TreeExplainer(model_cb)
shap_values_cb = explainer_cb.shap_values(X_other_b)

# Summary plot (beeswarm)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values_cb, X_other_b, show=False, max_display=16)
plt.title('SHAP Summary — CatBoost+Optuna (Other Set)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/04_shap_catboost_summary.png')
plt.close()
print(f"  Saved: {FIG_DIR}/04_shap_catboost_summary.png")

# Bar plot (mean |SHAP|)
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values_cb, X_other_b, plot_type='bar', show=False, max_display=16)
plt.title('SHAP Feature Importance — CatBoost+Optuna', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/05_shap_catboost_bar.png')
plt.close()
print(f"  Saved: {FIG_DIR}/05_shap_catboost_bar.png")

# ============================================================
# 7. SHAP - AutoGluon (best underlying model)
# ============================================================
print("\n" + "=" * 60)
print("7. SHAP Analysis — AutoGluon")
print("=" * 60)

# AutoGluon'un en iyi tek modeli üzerinden SHAP
# WeightedEnsemble içindeki modelleri bul
info = ag.info()
best_single = None
lb = ag.leaderboard(silent=True)
# L1'deki en iyi tree model
l1_models = lb[lb['stack_level'] == 1].sort_values('score_val', ascending=False)
for model_name in l1_models['model'].values:
    if any(t in model_name for t in ['CatBoost', 'LightGBM', 'XGBoost']):
        best_single = model_name
        break

print(f"  Best single tree model for SHAP: {best_single}")

# AutoGluon'un transformer'ından geçmiş veriyi al
ag_other_X = other_ag.drop(columns=[target])

# AutoGluon internal model'den SHAP için feature importance kullan
# Permutation-based feature importance zaten var, SHAP için model extract edelim
from autogluon.tabular import TabularPredictor

# Predict with specific model
y_proba_best_single = ag.predict_proba(ag_other_X, model=best_single)[1].values

# AutoGluon modelden SHAP almak zor olabilir, permutation importance kullan
# Bunun yerine AG'nin kendi feature_importance'ını görselleştirelim
importance_ag = ag.feature_importance(other_ag, silent=True)

fig, ax = plt.subplots(figsize=(10, 6))
importance_sorted = importance_ag.sort_values('importance', ascending=True)
colors = ['#FF5722' if p < 0.05 else '#BDBDBD' for p in importance_sorted['p_value']]
ax.barh(importance_sorted.index, importance_sorted['importance'], color=colors,
        xerr=importance_sorted['stddev'], capsize=3)
ax.set_xlabel('Permutation Importance (decrease in ROC AUC)', fontsize=12)
ax.set_title(f'AutoGluon Feature Importance — {ag.model_best}\n(Other Set, permutation-based)',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/06_autogluon_feature_importance.png')
plt.close()
print(f"  Saved: {FIG_DIR}/06_autogluon_feature_importance.png")

# SHAP for AutoGluon's best single tree model
# Extract the actual CatBoost/LightGBM model from AG
try:
    ag_model_path = f'models/autogluon_uti_ec/models/{best_single}'
    inner_model = ag._trainer.load_model(best_single)

    # AG'nin feature-transformed verisini al
    ag_other_transformed = ag._learner.transform_features(ag_other_X)
    # Inner model'in beklediği feature'lara filtrele
    model_features = inner_model._features
    if model_features:
        ag_other_transformed = ag_other_transformed[model_features]

    if 'CatBoost' in best_single:
        inner_cb = inner_model.model
        explainer_ag = shap.TreeExplainer(inner_cb)
        shap_values_ag = explainer_ag.shap_values(ag_other_transformed)
    elif 'LightGBM' in best_single:
        inner_lgb = inner_model.model
        explainer_ag = shap.TreeExplainer(inner_lgb)
        shap_values_ag = explainer_ag.shap_values(ag_other_transformed)
        if isinstance(shap_values_ag, list):
            shap_values_ag = shap_values_ag[1]  # positive class
    else:
        raise ValueError(f"SHAP not supported for {best_single}")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values_ag, ag_other_transformed, show=False, max_display=16)
    plt.title(f'SHAP Summary — AutoGluon ({best_single})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/07_shap_autogluon_summary.png')
    plt.close()
    print(f"  Saved: {FIG_DIR}/07_shap_autogluon_summary.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values_ag, ag_other_transformed, plot_type='bar', show=False, max_display=16)
    plt.title(f'SHAP Bar — AutoGluon ({best_single})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/08_shap_autogluon_bar.png')
    plt.close()
    print(f"  Saved: {FIG_DIR}/08_shap_autogluon_bar.png")

except Exception as e:
    print(f"  AutoGluon SHAP extraction failed: {e}")
    print("  Permutation importance plot already saved as fallback.")

# ============================================================
# 8. COMBINED SUMMARY FIGURE
# ============================================================
print("\n" + "=" * 60)
print("8. Combined Summary Figure")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# ROC
ax = axes[0, 0]
ax.plot(fpr_cb, tpr_cb, lw=2.5, label=f'CatBoost+Optuna (AUC={auc_cb:.4f})', color='#2196F3')
ax.plot(fpr_ag, tpr_ag, lw=2.5, label=f'AutoGluon (AUC={auc_ag:.4f})', color='#FF5722')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC Curve', fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

# PR
ax = axes[0, 1]
ax.plot(rec_cb, prec_cb, lw=2.5, label=f'CatBoost+Optuna (AP={ap_cb:.4f})', color='#2196F3')
ax.plot(rec_ag, prec_ag, lw=2.5, label=f'AutoGluon (AP={ap_ag:.4f})', color='#FF5722')
ax.axhline(y=prevalence, color='gray', linestyle='--', lw=1, alpha=0.5)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve', fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, alpha=0.3)

# Confusion Matrix CatBoost
ax = axes[1, 0]
cm_cb = confusion_matrix(y_other, y_pred_cb)
sns.heatmap(cm_cb, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
            annot_kws={'size': 14})
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('CatBoost+Optuna — Confusion Matrix', fontweight='bold')

# Confusion Matrix AutoGluon
ax = axes[1, 1]
cm_ag = confusion_matrix(y_other_ag_vals, y_pred_ag)
sns.heatmap(cm_ag, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
            annot_kws={'size': 14})
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('AutoGluon Ensemble — Confusion Matrix', fontweight='bold')

plt.suptitle('Model Comparison — Other Set (Holdout)', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/09_combined_summary.png')
plt.close()
print(f"  Saved: {FIG_DIR}/09_combined_summary.png")

print("\n" + "=" * 60)
print("TAMAMLANDI! Tüm figürler figures/ klasörüne kaydedildi.")
print("=" * 60)
