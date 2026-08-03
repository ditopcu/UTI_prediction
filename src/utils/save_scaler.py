"""
Reproduce the exact preprocessing from uti_01_dataset_clean.py
and save the fitted StandardScaler for LIS deployment.
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler

# Repository root, resolved from this file (src/utils/ -> repo root).
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get('UTI_DATA_DIR', os.path.join(BASE, 'data'))
MODEL_DIR = os.environ.get('UTI_MODEL_DIR', os.path.join(BASE, 'models'))

# ── 1. Load v2 dataset ──────────────────────────────────────────────
df = pd.read_excel(os.path.join(DATA_DIR, '02_interim', 'uti_cleaned_v2.xlsx'))
print(f"Loaded v2: {df.shape}")

# ── 2. Reproduce cleaning (same order as uti_01_dataset_clean.py) ──

# NITT: 'Positivo' → 1, then categorical
df['NITT'] = df['NITT'].replace('Positivo', 1)
df['NITT'] = pd.Categorical(df['NITT'])

# SEXO, BACT_INFO_baja → categorical
df['SEXO'] = pd.Categorical(df['SEXO'])
df['BACT_INFO_baja'] = pd.Categorical(df['BACT_INFO_baja'])

# Drop FILTER
df = df.drop(columns=['FILTER'])

# Drop NaN rows
df.dropna(inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop columns
df = df.drop(columns=['FECHA', 'XTAL', 'UROT', 'BILT', 'CETOT'])

# PHT → categorical with >=8 grouping
df['PHT'] = pd.Categorical(df['PHT'])
df['PHT_temp'] = df['PHT'].astype(float)
df['PHT'] = df['PHT_temp'].apply(lambda x: '>=8' if x >= 8 else str(x))
df['PHT'] = pd.Categorical(df['PHT'])
df = df.drop(columns=['PHT_temp'])

# Binary recode: PROTT, CASTS, YLC
for col in ['PROTT', 'CASTS', 'YLC']:
    df[col] = df[col].apply(lambda x: 0 if x == 0 else 1)
    df[col] = pd.Categorical(df[col])

# Age filter
df = df[df['EDAD'] >= 18]

# EDAD_CATEGORICA
bins_start_points = list(range(18, 90, 10))
bins = bins_start_points + [90, df['EDAD'].max() + 1]
labels = [f'{i}-{i+9}' for i in bins_start_points]
labels.append('>=90')
df['EDAD_CATEGORICA'] = pd.cut(df['EDAD'], bins=bins, labels=labels, right=False)
df['EDAD_CATEGORICA'] = df['EDAD_CATEGORICA'].astype('category')

# LEUT → categorical
df['LEUT'] = pd.Categorical(df['LEUT'])

# RBO: remove 99999
df['RBO'] = df['RBO'].replace(99999.0, np.nan)
df.dropna(subset=['RBO'], inplace=True)

# DENST / 1000
df['DENST'] = df['DENST'] / 1000

# Drop RESULTADO_CULTIVO (and parsed columns if they existed)
if 'RESULTADO_CULTIVO' in df.columns:
    df = df.drop(columns=['RESULTADO_CULTIVO'])

# Drop leakage columns
for c in ['Bacteria', 'Colony_Count', 'Sensible_Antibiotics', 'Resistant_Antibiotics']:
    if c in df.columns:
        df = df.drop(columns=[c])

print(f"After cleaning: {df.shape}")

# ── 3. Identify numerical columns (same logic as original) ──────────
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'CULTIVO_PATOLOGICO' in num_cols:
    num_cols.remove('CULTIVO_PATOLOGICO')

print(f"Numerical columns to scale: {num_cols}")

# ── 4. Fit scaler ───────────────────────────────────────────────────
scaler = StandardScaler()
scaler.fit(df[num_cols])

print("\nScaler parameters:")
for col, mean, std in zip(num_cols, scaler.mean_, scaler.scale_):
    print(f"  {col:10s}  mean={mean:.6f}  std={std:.6f}")

# ── 5. Save scaler as joblib ────────────────────────────────────────
out_path = os.path.join(MODEL_DIR, 'standard_scaler.joblib')
joblib.dump({'scaler': scaler, 'columns': num_cols}, out_path)
print(f"\nSaved scaler to {out_path}")

# ── 6. Also save as plain JSON (for config embedding) ───────────────
scaler_dict = {
    col: {'mean': float(m), 'std': float(s)}
    for col, m, s in zip(num_cols, scaler.mean_, scaler.scale_)
}
json_path = os.path.join(MODEL_DIR, 'scaler_params.json')
with open(json_path, 'w') as f:
    json.dump(scaler_dict, f, indent=2)
print(f"Saved scaler params JSON to {json_path}")

# ── 7. Verification: compare with ML-ready dataset ──────────────────
df_ml = pd.read_excel(os.path.join(DATA_DIR, '03_processed', 'uti_ml_final.xlsx'))
print(f"\nVerification against ML dataset ({df_ml.shape[0]} rows):")
for col in num_cols:
    if col in df_ml.columns:
        ml_mean = df_ml[col].mean()
        ml_std = df_ml[col].std()
        print(f"  {col:10s}  ML_data mean={ml_mean:.4f} std={ml_std:.4f}  (should be ~0 and ~1)")
