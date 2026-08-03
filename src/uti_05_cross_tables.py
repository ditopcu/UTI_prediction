# ============================================================
# Cross Tables: Our Models vs Existing Hospital Algorithms
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.size': 11
})

# ============================================================
# 1. Raw veri + mevcut algoritma predictions
# ============================================================
print("1. Veri yükleme...")
raw = pd.read_excel('data/01_raw/uti_raw.xlsx')

raw['CDS_RNA_pred'] = raw['CDS_RNA'].apply(
    lambda x: 1 if str(x).startswith('Alta') else (0 if str(x).startswith('Baja') else np.nan)
)

def pred_binary(x):
    s = str(x).strip()
    if 'Alta probabilidad' in s or 'Positivo' in s:
        return 1
    elif 'Baja probabilidad' in s or 'Negativo' in s:
        return 0
    return np.nan

raw['PRED_binary'] = raw['PRED_CULT_IA'].apply(pred_binary)
raw['UTI_CDS_pred'] = np.where(
    raw['UTI_CDS'] >= 50, 1,
    np.where(raw['UTI_CDS'].isna(), np.nan, 0)
)
raw['RAW_INDEX'] = raw.index + 1

# ============================================================
# 2. CatBoost+Optuna predictions (Other set)
# ============================================================
print("2. CatBoost+Optuna eğitimi...")
ml = pd.read_excel('data/03_processed/uti_ml_final.xlsx')
df_cb = ml.drop(columns=[c for c in ['ID', 'EDAD', 'RAW_INDEX'] if c in ml.columns])
target = 'CULTIVO_PATOLOGICO'
X = df_cb.drop(target, axis=1)
y = df_cb[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_test, X_other, y_test, y_other = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
feat_selector.fit(X_train.values, y_train.values)
selected = X_train.columns[feat_selector.support_].tolist()
X_train_b, X_test_b, X_other_b = X_train[selected], X_test[selected], X_other[selected]

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
    model.fit(X_train_b, y_train, eval_set=(X_test_b, y_test),
              early_stopping_rounds=50, verbose=False)
    return roc_auc_score(y_test, model.predict_proba(X_test_b)[:, 1])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
best_params = study.best_params
best_params['random_seed'] = 42
best_params['verbose'] = 0
model_cb = CatBoostClassifier(**best_params)
model_cb.fit(X_train_b, y_train, eval_set=(X_test_b, y_test),
             early_stopping_rounds=50, verbose=0)
cb_preds = model_cb.predict(X_other_b).astype(int)

# ============================================================
# 3. AutoGluon predictions (Other set)
# ============================================================
print("3. AutoGluon yükleme...")
from autogluon.tabular import TabularPredictor
ag = TabularPredictor.load('models/autogluon_uti_ec', verbosity=0)

v2 = pd.read_excel('data/02_interim/uti_cleaned_v2.xlsx')
safe_cols = ['FECHA', 'EDAD', 'SEXO', 'WBCO', 'EC', 'BACTS', 'RBO', 'PHT', 'YLC', 'CASTS']
raw_dedup = raw.drop_duplicates(subset=safe_cols, keep='first').copy()
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
labels = [f"{i}-{i+9}" for i in bins_s] + ['>=90']
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

ag_other_no_raw = X_other_ag.drop(columns=['RAW_INDEX'])
ag_preds = ag.predict(ag_other_no_raw).values.astype(int)

# ============================================================
# 4. RAW_INDEX ile birleştir
# ============================================================
print("4. Birleştirme...")

other_raw_idx = ml.loc[X_other.index, 'RAW_INDEX'].values
df_cb_result = pd.DataFrame({
    'RAW_INDEX': other_raw_idx,
    'CatBoost_pred': cb_preds,
    'Actual': y_other.values
})

df_ag_result = pd.DataFrame({
    'RAW_INDEX': X_other_ag['RAW_INDEX'].values,
    'AG_pred': ag_preds,
})

df_cb_merged = df_cb_result.merge(
    raw[['RAW_INDEX', 'UTI_CDS_pred', 'CDS_RNA_pred', 'PRED_binary']],
    on='RAW_INDEX', how='left'
)
df_all = df_cb_merged.merge(df_ag_result, on='RAW_INDEX', how='inner')

print(f"Ortak satır: {len(df_all)}")
print(f"UTI_CDS mevcut: {df_all['UTI_CDS_pred'].notna().sum()}")
print(f"CDS_RNA mevcut: {df_all['CDS_RNA_pred'].notna().sum()}")
print(f"PRED_CULT_IA mevcut: {df_all['PRED_binary'].notna().sum()}")

# ============================================================
# 5. CROSS TABLES - Prediction Agreement
# ============================================================
print("\n5. Cross table figürleri oluşturuluyor...")

label_map = {0: 'Negatif', 1: 'Pozitif'}
algo_cols = [
    ('UTI_CDS_pred', 'UTI_CDS (NN)'),
    ('CDS_RNA_pred', 'CDS_RNA (RF)'),
    ('PRED_binary', 'PRED_CULT_IA'),
]
our_cols = [
    ('CatBoost_pred', 'CatBoost+Optuna'),
    ('AG_pred', 'AutoGluon'),
]

# --- Figure 10: Prediction cross tables ---
fig, axes = plt.subplots(2, 3, figsize=(20, 13))

for row, (our_col, our_name) in enumerate(our_cols):
    for col_idx, (algo_col, algo_name) in enumerate(algo_cols):
        ax = axes[row, col_idx]
        mask = df_all[algo_col].notna()
        sub = df_all[mask].copy()

        if len(sub) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        sub[algo_col] = sub[algo_col].astype(int)

        ct = pd.crosstab(
            sub[our_col].map(label_map),
            sub[algo_col].map(label_map),
        )
        # Sıralama: Negatif, Pozitif
        for idx_name in ['Negatif', 'Pozitif']:
            if idx_name not in ct.index:
                ct.loc[idx_name] = 0
            if idx_name not in ct.columns:
                ct[idx_name] = 0
        ct = ct.loc[['Negatif', 'Pozitif'], ['Negatif', 'Pozitif']]

        agreement = (sub[our_col] == sub[algo_col].astype(int)).mean() * 100

        sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                    annot_kws={'size': 18}, cbar=False, linewidths=2,
                    linecolor='white')
        ax.set_ylabel(our_name, fontsize=12, fontweight='bold')
        ax.set_xlabel(algo_name, fontsize=12, fontweight='bold')
        ax.set_title(f"n={len(sub)}, Agreement={agreement:.1f}%",
                     fontsize=11, fontweight='bold')

plt.suptitle(
    'Prediction Cross-Tabulation: Our Models vs Existing Algorithms\n(Other/Holdout Set)',
    fontsize=16, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('figures/10_cross_tables_comparison.png')
plt.close()
print("  Saved: figures/10_cross_tables_comparison.png")

# --- Figure 11: Correctness cross tables ---
fig, axes = plt.subplots(2, 3, figsize=(22, 14))

for row, (our_col, our_name) in enumerate(our_cols):
    for col_idx, (algo_col, algo_name) in enumerate(algo_cols):
        ax = axes[row, col_idx]
        mask = df_all[algo_col].notna()
        sub = df_all[mask].copy()

        if len(sub) == 0:
            continue

        sub[algo_col] = sub[algo_col].astype(int)
        our_ok = (sub[our_col] == sub['Actual'])
        algo_ok = (sub[algo_col] == sub['Actual'])

        both_ok = (our_ok & algo_ok).sum()
        our_only = (our_ok & ~algo_ok).sum()
        algo_only = (~our_ok & algo_ok).sum()
        both_wrong = (~our_ok & ~algo_ok).sum()

        ct_data = np.array([[both_ok, algo_only],
                            [our_only, both_wrong]])

        ct_df = pd.DataFrame(
            ct_data,
            index=[f'{our_name} Correct', f'{our_name} Wrong'],
            columns=[f'{algo_name} Correct', f'{algo_name} Wrong']
        )

        colors = np.array([['#4CAF50', '#FF9800'],
                           ['#2196F3', '#F44336']])
        # Custom heatmap
        ax.imshow([[0, 1], [2, 3]], cmap='RdYlGn_r', alpha=0)
        for i in range(2):
            for j in range(2):
                val = ct_data[i, j]
                pct = val / len(sub) * 100
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=True, color=colors[i, j], alpha=0.7))
                ax.text(j, i, f"{val}\n({pct:.1f}%)",
                        ha='center', va='center', fontsize=15, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f'{algo_name}\nCorrect', f'{algo_name}\nWrong'], fontsize=10)
        ax.set_yticklabels([f'{our_name}\nCorrect', f'{our_name}\nWrong'], fontsize=10)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        ax.set_title(
            f"n={len(sub)} | Both correct: {both_ok} | Both wrong: {both_wrong}",
            fontsize=10, fontweight='bold'
        )

plt.suptitle(
    'Correctness Cross-Tab: Our Models vs Existing Algorithms\n'
    '(Green=Both Correct, Red=Both Wrong, Blue=Ours Only, Orange=Theirs Only)',
    fontsize=14, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('figures/11_cross_tables_correctness.png')
plt.close()
print("  Saved: figures/11_cross_tables_correctness.png")

print("\nTamamlandı!")
