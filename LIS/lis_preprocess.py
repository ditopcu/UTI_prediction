"""
Transforms a single raw LIS patient record into the 16-feature
model-ready format expected by the CatBoost+Optuna UTI model.

Input: dict with keys listed in REQUIRED_FIELDS below.
Output: pandas DataFrame (1×16) ready for model.predict_proba().
"""

import pandas as pd

# ── 16 Boruta-selected features (exact order the model expects) ──────
MODEL_FEATURES = [
    'DENST', 'HEMATT', 'RBO', 'WBCO', 'EC', 'BACTS',
    'SEXO_M', 'LEUT_25', 'LEUT_75', 'LEUT_500',
    'NITT_1', 'PROTT_1',
    'BACT_INFO_baja_1', 'BACT_INFO_baja_2', 'BACT_INFO_baja_3',
    'EDAD_CATEGORICA_28-37',
]

# ── Required raw input fields from LIS ───────────────────────────────
REQUIRED_FIELDS = [
    'EDAD', 'SEXO', 'DENST', 'HEMATT', 'RBO', 'WBCO', 'EC', 'BACTS',
    'LEUT', 'NITT', 'PROTT', 'BACT_INFO',
]

# ── StandardScaler parameters (fit on full training data) ────────────
SCALER_PARAMS = {
    'DENST':  {'mean': 1.0170046046046044,  'std': 0.007688568976846775},
    'HEMATT': {'mean': 101.39406072739406,   'std': 111.76485697517653},
    'RBO':    {'mean': 1586.1292425759093,   'std': 6980.803059995904},
    'WBCO':   {'mean': 2636.917257257257,    'std': 7524.9127452222565},
    'EC':     {'mean': 28.580013346680012,   'std': 83.73992469896548},
    'BACTS':  {'mean': 18487.615115115113,   'std': 30677.177086134197},
}

# ── BACT_INFO text → BACT_INFO_baja mapping ─────────────────────────
# Order matters: longer/more specific patterns FIRST to avoid partial matches.
BACT_INFO_MAP = [
    ('gram negativo y gram positivo', 2),  # mixed — must be before 'gram negativo'
    ('gram negativo y gran positivo', 2),  # typo variant in LIS
    ('gram negativo',                 0),  # only gram-negative
    ('gram positivo',                 1),  # only gram-positive
    ('sin informacion gram',          3),  # no gram info (also: sin información)
]

# ── HEMATT string → numeric ──────────────────────────────────────────
HEMATT_MAP = {'Negativo': 0, '10': 10, '20': 20, '50': 50, '250': 250}

# ── PROTT string → numeric (then binarized: 0→0, else→1) ────────────
PROTT_MAP = {'Negativo': 0, '15': 15, '30': 30, '100': 100, '300': 300, '1000': 1000}

# ── Valid value sets (for input validation) ──────────────────────────
VALID_SEXO = {'M', 'H'}
VALID_LEUT = {'Negativo', '25', '75', '500'}
VALID_NITT = {'Negativo', 'Positivo'}


class PreprocessError(Exception):
    """Raised when input validation or transformation fails."""
    pass


def _validate(patient: dict) -> None:
    """Check required fields, types, and value ranges."""
    missing = [f for f in REQUIRED_FIELDS if f not in patient]
    if missing:
        raise PreprocessError(f"Missing fields: {missing}")

    nulls = [f for f in REQUIRED_FIELDS if patient[f] is None]
    if nulls:
        raise PreprocessError(f"Null values: {nulls}")

    if patient['EDAD'] < 18:
        raise PreprocessError(f"EDAD={patient['EDAD']} < 18, model requires adults")

    if float(patient['RBO']) == 99999.0:
        raise PreprocessError("RBO=99999 indicates equipment error")

    sexo = str(patient['SEXO']).strip()
    if sexo not in VALID_SEXO:
        raise PreprocessError(f"SEXO='{sexo}' not in {VALID_SEXO}")

    leut = str(patient['LEUT']).strip()
    if leut not in VALID_LEUT:
        raise PreprocessError(f"LEUT='{leut}' not in {VALID_LEUT}")

    nitt = str(patient['NITT']).strip()
    if nitt not in VALID_NITT:
        raise PreprocessError(f"NITT='{nitt}' not in {VALID_NITT}")


def _map_bact_info(raw_text: str) -> int:
    """Map BACT_INFO full text → BACT_INFO_baja (0/1/2/3)."""
    text = raw_text.strip().lower()
    text = text.replace('ó', 'o').replace('á', 'a').replace('é', 'e')
    for keyword, code in BACT_INFO_MAP:
        if keyword in text:
            return code
    raise PreprocessError(f"Unknown BACT_INFO: '{raw_text}'")


def _scale(value: float, col: str) -> float:
    """Apply StandardScaler: (x - mean) / std."""
    p = SCALER_PARAMS[col]
    return (value - p['mean']) / p['std']


def preprocess(patient: dict) -> pd.DataFrame:
    """
    Transform raw LIS dict → model-ready DataFrame (1×16).

    Expected input dict example:
    {
        'EDAD': 72,
        'SEXO': 'M',
        'DENST': 1.015,         # specific gravity as-is from LIS
        'HEMATT': '250',        # string from LIS ('Negativo','10','20','50','250')
        'RBO': 1234.5,
        'WBCO': 5678.9,
        'EC': 12.3,
        'BACTS': 9876.5,
        'LEUT': 'Negativo',     # 'Negativo','25','75','500'
        'NITT': 'Negativo',     # 'Negativo','Positivo'
        'PROTT': '30',          # string from LIS ('Negativo','15','30','100','300','1000')
        'BACT_INFO': 'El escategrama sugiere la presencia de gram negativo',
    }
    """
    _validate(patient)

    # ── 1. Numeric conversions ───────────────────────────────────────
    denst = float(patient['DENST'])
    hematt_str = str(patient['HEMATT']).strip()
    hematt = HEMATT_MAP.get(hematt_str)
    if hematt is None:
        raise PreprocessError(f"Unknown HEMATT: '{hematt_str}'")
    hematt = float(hematt)
    rbo = float(patient['RBO'])
    wbco = float(patient['WBCO'])
    ec = float(patient['EC'])
    bacts = float(patient['BACTS'])

    # ── 2. Scale numerics ────────────────────────────────────────────
    denst_s = _scale(denst, 'DENST')
    hematt_s = _scale(hematt, 'HEMATT')
    rbo_s = _scale(rbo, 'RBO')
    wbco_s = _scale(wbco, 'WBCO')
    ec_s = _scale(ec, 'EC')
    bacts_s = _scale(bacts, 'BACTS')

    # ── 3. One-hot: SEXO → SEXO_M ───────────────────────────────────
    sexo = str(patient['SEXO']).strip()
    sexo_m = 1 if sexo == 'M' else 0

    # ── 4. One-hot: LEUT → LEUT_25, LEUT_75, LEUT_500 ───────────────
    leut = str(patient['LEUT']).strip()
    leut_25  = 1 if leut == '25'  else 0
    leut_75  = 1 if leut == '75'  else 0
    leut_500 = 1 if leut == '500' else 0

    # ── 5. Binary: NITT → NITT_1 ─────────────────────────────────────
    nitt = str(patient['NITT']).strip()
    nitt_1 = 1 if nitt == 'Positivo' else 0

    # ── 6. Binary: PROTT → PROTT_1 (0 stays 0, any other → 1) ───────
    prott_str = str(patient['PROTT']).strip()
    prott_num = PROTT_MAP.get(prott_str)
    if prott_num is None:
        raise PreprocessError(f"Unknown PROTT: '{prott_str}'")
    prott_1 = 0 if prott_num == 0 else 1

    # ── 7. One-hot: BACT_INFO → BACT_INFO_baja_1/2/3 ────────────────
    baja = _map_bact_info(str(patient['BACT_INFO']))
    bact_info_1 = 1 if baja == 1 else 0
    bact_info_2 = 1 if baja == 2 else 0
    bact_info_3 = 1 if baja == 3 else 0

    # ── 8. One-hot: EDAD → EDAD_CATEGORICA_28-37 ─────────────────────
    edad = int(patient['EDAD'])
    edad_28_37 = 1 if 28 <= edad < 38 else 0

    # ── 9. Assemble row in model feature order ───────────────────────
    row = {
        'DENST': denst_s,
        'HEMATT': hematt_s,
        'RBO': rbo_s,
        'WBCO': wbco_s,
        'EC': ec_s,
        'BACTS': bacts_s,
        'SEXO_M': sexo_m,
        'LEUT_25': leut_25,
        'LEUT_75': leut_75,
        'LEUT_500': leut_500,
        'NITT_1': nitt_1,
        'PROTT_1': prott_1,
        'BACT_INFO_baja_1': bact_info_1,
        'BACT_INFO_baja_2': bact_info_2,
        'BACT_INFO_baja_3': bact_info_3,
        'EDAD_CATEGORICA_28-37': edad_28_37,
    }

    return pd.DataFrame([row], columns=MODEL_FEATURES)
