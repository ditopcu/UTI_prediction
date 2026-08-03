# ============================================================
# Gram-negative vs Gram-positive Prediction
# Among culture-positive UTI patients
# Two models: with and without BACT_INFO_baja
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc,
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix, classification_report
)
from catboost import CatBoostClassifier
from boruta import BorutaPy

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.size': 11,
})

RANDOM_STATE = 42
FIG_DIR = 'figures'

# ============================================================
# 1. DATA PREPARATION
# ============================================================
print("=" * 60)
print("1. Data Preparation")
print("=" * 60)

raw = pd.read_excel('data/01_raw/uti_raw.xlsx')
v2 = pd.read_excel('data/02_interim/uti_cleaned_v2.xlsx')

# RAW_INDEX matching
safe_cols = ['FECHA', 'EDAD', 'SEXO', 'WBCO', 'EC', 'BACTS', 'RBO', 'PHT', 'YLC', 'CASTS']
raw_dedup = raw.drop_duplicates(subset=safe_cols, keep='first').copy()
raw_dedup['RAW_INDEX'] = raw_dedup.index + 1
v2 = v2.merge(raw_dedup[safe_cols + ['RAW_INDEX']], on=safe_cols, how='left')

# Preprocessing (same as uti_01)
df = v2.copy()
df['NITT'] = df['NITT'].replace('Positivo', 1)
df = df.drop(columns=['FILTER'])
df.dropna(inplace=True)
orig_cols = [c for c in df.columns if c != 'RAW_INDEX']
df = df.drop_duplicates(subset=orig_cols, keep='first')
df = df.drop(columns=['FECHA', 'XTAL', 'UROT', 'BILT', 'CETOT'])
df = df[df['EDAD'] >= 18]
df['RBO'] = df['RBO'].replace(99999.0, np.nan)
df.dropna(subset=['RBO'], inplace=True)

bins_s = list(range(18, 90, 10))
bins = bins_s + [90, int(df['EDAD'].max()) + 1]
labels = [f'{i}-{i+9}' for i in bins_s] + ['>=90']
df['EDAD_CATEGORICA'] = pd.cut(df['EDAD'], bins=bins, labels=labels, right=False)
df['DENST'] = df['DENST'] / 1000
for col in ['PROTT', 'CASTS', 'YLC']:
    df[col] = df[col].apply(lambda x: 0 if x == 0 else 1)

print(f"Preprocessed: {df.shape}")

# ============================================================
# 2. GRAM TARGET VARIABLE
# ============================================================
print("\n" + "=" * 60)
print("2. Gram Classification Target")
print("=" * 60)

# Extract bacteria from raw and classify gram
raw['RAW_INDEX'] = raw.index + 1

def extract_bacteria(x):
    if pd.isna(x): return np.nan
    lines = str(x).split('\n')
    b = lines[0].strip()
    if b in ['0', 'Negativo', '']: return np.nan
    return b

gram_neg = [
    'Escherichia coli', 'Klebsiella pneumoniae', 'Proteus mirabilis',
    'Pseudomonas aeruginosa', 'Citrobacter koseri', 'Citrobacter freundii',
    'Klebsiella oxytoca', 'Morganella morganii', 'Serratia marcescens',
    'Enterobacter cloacae', 'Providencia stuartii', 'Klebsiella aerogenes',
    'Proteus vulgaris', 'Raoultella ornithinolytica', 'Enterobacter aerogenes',
    'Kluyvera ascorbata',
]
gram_pos = [
    'Enterococcus faecalis', 'Staphylococcus aureus', 'Streptococcus agalactiae',
    'Staphylococcus saprophyticus', 'Staphylococcus epidermidis',
    'Enterococcus faecium', 'Staphylococcus hominis', 'Aerococcus urinae',
]

def classify_gram(b):
    if pd.isna(b): return np.nan
    if b in gram_neg: return 0  # Gram-negative
    if b in gram_pos: return 1  # Gram-positive
    return np.nan  # fungi, mixed, rare → exclude

raw['bacteria'] = raw['RESULTADO_CULTIVO'].apply(extract_bacteria)
raw['GRAM'] = raw['bacteria'].apply(classify_gram)

# Merge gram target into preprocessed data
df_gram = df.merge(raw[['RAW_INDEX', 'GRAM']], on='RAW_INDEX', how='left')

# Filter: only culture-positive AND gram classifiable
df_gram = df_gram[df_gram['CULTIVO_PATOLOGICO'] == 1]
df_gram = df_gram.dropna(subset=['GRAM'])
df_gram['GRAM'] = df_gram['GRAM'].astype(int)

print(f"Culture-positive with gram info: {len(df_gram)}")
print(f"Gram-negative (0): {(df_gram['GRAM'] == 0).sum()}")
print(f"Gram-positive (1): {(df_gram['GRAM'] == 1).sum()}")
print(f"Imbalance ratio: {(df_gram['GRAM'] == 0).sum() / (df_gram['GRAM'] == 1).sum():.1f}:1")

# ============================================================
# 3. FEATURE SETS
# ============================================================
print("\n" + "=" * 60)
print("3. Feature Sets")
print("=" * 60)

features_with_bact = [
    'DENST', 'HEMATT', 'RBO', 'WBCO', 'EC', 'BACTS',
    'SEXO', 'LEUT', 'NITT', 'PROTT', 'BACT_INFO_baja', 'EDAD_CATEGORICA'
]

features_no_bact = [f for f in features_with_bact if f != 'BACT_INFO_baja']

cat_cols_with = ['SEXO', 'LEUT', 'NITT', 'PROTT', 'BACT_INFO_baja', 'EDAD_CATEGORICA']
cat_cols_no = [c for c in cat_cols_with if c != 'BACT_INFO_baja']

TARGET = 'GRAM'

print(f"With BACT_INFO:    {len(features_with_bact)} features")
print(f"Without BACT_INFO: {len(features_no_bact)} features")

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def prepare_data(df_in, features, cat_cols):
    """Prepare model-ready dataframe."""
    df_out = df_in[features + [TARGET, 'RAW_INDEX']].copy()
    for col in cat_cols:
        df_out[col] = df_out[col].astype(str)
    return df_out

def split_data(df_in, features, cat_cols):
    """Split into train/test/other with stratification."""
    df_ready = prepare_data(df_in, features, cat_cols)
    X = df_ready.drop(columns=[TARGET])
    y = df_ready[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y)
    X_test, X_other, y_test, y_other = train_test_split(
        X_temp, y_temp, test_size=0.375, random_state=RANDOM_STATE, stratify=y_temp)

    return X_train, X_test, X_other, y_train, y_test, y_other

def run_optuna(X_train, y_train, X_test, y_test, cat_features, n_trials=30):
    """Optuna hyperparameter search."""
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'auto_class_weights': 'Balanced',
            'od_type': 'Iter', 'od_wait': 50,
            'random_seed': RANDOM_STATE, 'verbose': False,
            'cat_features': cat_features,
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study

def train_final(X_train, y_train, X_test, y_test, best_params, cat_features):
    """Train final model with best params."""
    best_params['random_seed'] = RANDOM_STATE
    best_params['verbose'] = 0
    best_params['auto_class_weights'] = 'Balanced'
    best_params['cat_features'] = cat_features
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test),
              early_stopping_rounds=50, verbose=0)
    return model

def calc_metrics(name, y_true, y_pred, y_proba):
    return {
        'Model': name,
        'N': len(y_true),
        'ROC AUC': round(roc_auc_score(y_true, y_proba), 4),
        'PR AUC': round(average_precision_score(y_true, y_proba), 4),
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Bal. Acc': round(balanced_accuracy_score(y_true, y_pred), 4),
        'F1': round(f1_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall': round(recall_score(y_true, y_pred), 4),
        'MCC': round(matthews_corrcoef(y_true, y_pred), 4),
    }

# ============================================================
# 5. MODEL A: WITH BACT_INFO_baja
# ============================================================
print("\n" + "=" * 60)
print("5. Model A: WITH BACT_INFO_baja")
print("=" * 60)

X_train_A, X_test_A, X_other_A, y_train_A, y_test_A, y_other_A = \
    split_data(df_gram, features_with_bact, cat_cols_with)

# Drop RAW_INDEX for training
ri_train_A = X_train_A['RAW_INDEX']
ri_other_A = X_other_A['RAW_INDEX']
X_train_A = X_train_A.drop(columns=['RAW_INDEX'])
X_test_A = X_test_A.drop(columns=['RAW_INDEX'])
X_other_A = X_other_A.drop(columns=['RAW_INDEX'])

cat_idx_A = [i for i, c in enumerate(X_train_A.columns) if c in cat_cols_with]

print(f"Train: {X_train_A.shape}, Test: {X_test_A.shape}, Other: {X_other_A.shape}")
print(f"Cat feature indices: {cat_idx_A}")

print("Running Optuna (30 trials)...")
study_A = run_optuna(X_train_A, y_train_A, X_test_A, y_test_A, cat_idx_A)
print(f"Best ROC AUC (test): {study_A.best_value:.4f}")

model_A = train_final(X_train_A, y_train_A, X_test_A, y_test_A, study_A.best_params, cat_idx_A)

y_pred_A = model_A.predict(X_other_A).astype(int)
y_proba_A = model_A.predict_proba(X_other_A)[:, 1]

print("\n--- Model A: Other Set ---")
print(classification_report(y_other_A, y_pred_A, target_names=['Gram-', 'Gram+']))

# ============================================================
# 6. MODEL B: WITHOUT BACT_INFO_baja
# ============================================================
print("\n" + "=" * 60)
print("6. Model B: WITHOUT BACT_INFO_baja")
print("=" * 60)

X_train_B, X_test_B, X_other_B, y_train_B, y_test_B, y_other_B = \
    split_data(df_gram, features_no_bact, cat_cols_no)

ri_other_B = X_other_B['RAW_INDEX']
X_train_B = X_train_B.drop(columns=['RAW_INDEX'])
X_test_B = X_test_B.drop(columns=['RAW_INDEX'])
X_other_B = X_other_B.drop(columns=['RAW_INDEX'])

cat_idx_B = [i for i, c in enumerate(X_train_B.columns) if c in cat_cols_no]

print(f"Train: {X_train_B.shape}, Test: {X_test_B.shape}, Other: {X_other_B.shape}")

print("Running Optuna (30 trials)...")
study_B = run_optuna(X_train_B, y_train_B, X_test_B, y_test_B, cat_idx_B)
print(f"Best ROC AUC (test): {study_B.best_value:.4f}")

model_B = train_final(X_train_B, y_train_B, X_test_B, y_test_B, study_B.best_params, cat_idx_B)

y_pred_B = model_B.predict(X_other_B).astype(int)
y_proba_B = model_B.predict_proba(X_other_B)[:, 1]

print("\n--- Model B: Other Set ---")
print(classification_report(y_other_B, y_pred_B, target_names=['Gram-', 'Gram+']))

# ============================================================
# 7. COMPARISON TABLE
# ============================================================
print("\n" + "=" * 60)
print("7. Comparison")
print("=" * 60)

results = []
for name, yt, yp, ypp in [
    ('A: With BACT_INFO (Other)', y_other_A, y_pred_A, y_proba_A),
    ('B: Without BACT_INFO (Other)', y_other_B, y_pred_B, y_proba_B),
]:
    results.append(calc_metrics(name, yt, yp, ypp))

comp = pd.DataFrame(results)
pd.set_option('display.width', 180)
print(comp.to_string(index=False))

# ============================================================
# 8. PLOTS: ROC, PR, CONFUSION MATRICES
# ============================================================
print("\n" + "=" * 60)
print("8. Plots")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# ROC
ax = axes[0, 0]
for name, yt, ypp, color in [
    ('With BACT_INFO', y_other_A, y_proba_A, '#2196F3'),
    ('Without BACT_INFO', y_other_B, y_proba_B, '#FF5722'),
]:
    fpr, tpr, _ = roc_curve(yt, ypp)
    ax.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC={auc(fpr, tpr):.4f})', color=color)
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC Curve — Gram Prediction', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# PR
ax = axes[0, 1]
for name, yt, ypp, color in [
    ('With BACT_INFO', y_other_A, y_proba_A, '#2196F3'),
    ('Without BACT_INFO', y_other_B, y_proba_B, '#FF5722'),
]:
    prec, rec, _ = precision_recall_curve(yt, ypp)
    ap = average_precision_score(yt, ypp)
    ax.plot(rec, prec, lw=2.5, label=f'{name} (AP={ap:.4f})', color=color)
prevalence = y_other_A.mean()
ax.axhline(y=prevalence, color='gray', linestyle='--', lw=1, alpha=0.5,
           label=f'Baseline (prev={prevalence:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve — Gram Prediction', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# CM - Model A
ax = axes[1, 0]
cm_A = confusion_matrix(y_other_A, y_pred_A)
sns.heatmap(cm_A, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Gram-', 'Gram+'], yticklabels=['Gram-', 'Gram+'],
            annot_kws={'size': 16})
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('With BACT_INFO — Confusion Matrix', fontweight='bold')

# CM - Model B
ax = axes[1, 1]
cm_B = confusion_matrix(y_other_B, y_pred_B)
sns.heatmap(cm_B, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=['Gram-', 'Gram+'], yticklabels=['Gram-', 'Gram+'],
            annot_kws={'size': 16})
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Without BACT_INFO — Confusion Matrix', fontweight='bold')

plt.suptitle('Gram Prediction — Model Comparison (Other Set)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/gram_model_comparison.png')
plt.close()
print(f"Saved: {FIG_DIR}/gram_model_comparison.png")

# ============================================================
# 9. SHAP — Both Models
# ============================================================
print("\n" + "=" * 60)
print("9. SHAP Analysis")
print("=" * 60)

for label, model, X_other_shap, suffix in [
    ('Model A (With BACT_INFO)', model_A, X_other_A, 'gram_with_bact'),
    ('Model B (Without BACT_INFO)', model_B, X_other_B, 'gram_no_bact'),
]:
    print(f"\n  Computing SHAP for {label}...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_other_shap)

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_other_shap, show=False, max_display=12)
    plt.title(f'SHAP Summary — {label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/shap_{suffix}_summary.png')
    plt.close()
    print(f"  Saved: {FIG_DIR}/shap_{suffix}_summary.png")

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_other_shap, plot_type='bar', show=False, max_display=12)
    plt.title(f'SHAP Bar — {label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/shap_{suffix}_bar.png')
    plt.close()
    print(f"  Saved: {FIG_DIR}/shap_{suffix}_bar.png")

# ============================================================
# 10. EXPORT
# ============================================================
print("\n" + "=" * 60)
print("10. Export")
print("=" * 60)

comp.to_excel('data/04_results/gram_prediction_comparison.xlsx', index=False)
print("Saved: data/04_results/gram_prediction_comparison.xlsx")

print("\nDone!")
