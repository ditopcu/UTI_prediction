# ============================================================
# UTI AutoGluon - best_quality (EC dahil, Boruta features)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ---- 1. Veri Hazırlık (v2'den, scale/encode yok) ----

v2 = pd.read_excel("data/02_interim/uti_cleaned_v2.xlsx")
raw = pd.read_excel("data/01_raw/uti_raw.xlsx")

# RAW_INDEX eşleştirmesi (önceki analizle aynı)
safe_cols = ['FECHA', 'EDAD', 'SEXO', 'WBCO', 'EC', 'BACTS', 'RBO', 'PHT', 'YLC', 'CASTS']
raw_dedup = raw.drop_duplicates(subset=safe_cols, keep='first').copy()
raw_dedup['RAW_INDEX'] = raw_dedup.index + 1
v2 = v2.merge(raw_dedup[safe_cols + ['RAW_INDEX']], on=safe_cols, how='left')

# Script adımlarını tekrarla (uti_01 ile aynı)
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

# EDAD_CATEGORICA oluştur (script ile aynı)
bins_start = list(range(18, 90, 10))
bins = bins_start + [90, df['EDAD'].max() + 1]
labels = [f'{i}-{i+9}' for i in bins_start] + ['>=90']
df['EDAD_CATEGORICA'] = pd.cut(df['EDAD'], bins=bins, labels=labels, right=False)

# DENST / 1000 (script ile aynı)
df['DENST'] = df['DENST'] / 1000

# PROTT, CASTS, YLC binary recode (script ile aynı)
for col in ['PROTT', 'CASTS', 'YLC']:
    df[col] = df[col].apply(lambda x: 0 if x == 0 else 1)

print(f"Filtreleme sonrası: {df.shape[0]} satır (beklenen: 14985)")

# ---- 2. Boruta seçilen orijinal feature'lar ----

boruta_features = [
    'DENST', 'HEMATT', 'RBO', 'WBCO', 'EC', 'BACTS',  # sayısal
    'SEXO', 'LEUT', 'NITT', 'PROTT', 'BACT_INFO_baja', 'EDAD_CATEGORICA'  # kategorik
]
target = 'CULTIVO_PATOLOGICO'

df_model = df[boruta_features + [target, 'RAW_INDEX']].copy()

# Kategorikleri string'e çevir (AutoGluon category olarak tanısın)
cat_cols = ['SEXO', 'LEUT', 'NITT', 'PROTT', 'BACT_INFO_baja', 'EDAD_CATEGORICA']
for col in cat_cols:
    df_model[col] = df_model[col].astype(str)

print(f"Model veri seti: {df_model.shape}")
print(f"Feature'lar: {boruta_features}")
print(f"Kategorikler: {cat_cols}")

# ---- 3. Train/Test/Other split (aynı random_state) ----

X = df_model.drop(columns=[target])
y = df_model[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)
X_test, X_other, y_test, y_other = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
other_data = pd.concat([X_other, y_other], axis=1)

print(f"\nTrain: {len(train_data)}, Test: {len(test_data)}, Other: {len(other_data)}")

# ---- 4. AutoGluon ----

from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label=target,
    eval_metric='roc_auc',
    path='models/autogluon_uti_ec',
    verbosity=3,
)

predictor.fit(
    train_data=train_data.drop(columns=['RAW_INDEX']),
    tuning_data=test_data.drop(columns=['RAW_INDEX']),
    presets='best_quality',
    time_limit=1800,
    use_bag_holdout=True,
)

# ---- 5. Sonuçlar ----

print("\n" + "="*60)
print("LEADERBOARD")
print("="*60)
leaderboard = predictor.leaderboard(test_data.drop(columns=['RAW_INDEX']), silent=False)
print(leaderboard.to_string())

print("\n" + "="*60)
print("OTHER SET (holdout) DEĞERLENDİRME")
print("="*60)
perf = predictor.evaluate(other_data.drop(columns=['RAW_INDEX']), silent=False)
print(perf)

print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)
importance = predictor.feature_importance(test_data.drop(columns=['RAW_INDEX']))
print(importance)
