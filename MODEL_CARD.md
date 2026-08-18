# Model Card — UTI urinalysis-only culture-positivity predictor

## Overview
- **Model:** CatBoost gradient-boosting classifier, hyperparameters tuned with Optuna.
- **Task:** binary prediction of a positive (pathological) urine culture from routine automated urinalysis and demographics.
- **Output:** probability of a positive culture; binary decision at a fixed 0.50 threshold.
- **File:** `models/model_optuna.cbm` (deployed model). Baseline: `models/model_baseline.cbm`.
- **Scaler:** `models/standard_scaler.joblib` / `models/scaler_params.json` (for the deployment pipeline).
- **Metadata:** `models/uti_models_meta.json` (selected features, tuned parameters, CV AUC).

## Inputs (16 encoded features -> 12 original variables)
DENST (specific gravity), HEMATT (blood/Hb peroxidase), RBO (erythrocytes), WBCO (leukocytes),
EC (epithelial cells), BACTS (bacteria), SEXO (sex), LEUT (dipstick leukocyte esterase),
NITT (nitrite), PROTT (protein), BACT_INFO (scattergram Gram-type flag), EDAD_CATEGORICA (age band).
Glucose, pH, hyaline casts and yeasts were evaluated but not retained by Boruta selection.
**No clinical variable is used** (no body temperature, symptoms, or vital signs).

## Tuned hyperparameters (Optuna, 30 trials)
iterations 801, learning_rate 0.119, depth 4, l2_leaf_reg 1.42, border_count 73.
Best trial score during the search: 0.882 ROC AUC on the test partition (the search objective;
no cross-validation was used for model selection).

The search space also sampled `bagging_temperature` (0.846, recorded in
`models/uti_models_meta.json`), but the fitted model uses CatBoost's default `bootstrap_type=MVS`
with `subsample=0.8`, which ignores `bagging_temperature`. That value therefore had no effect on
the released model and is not reported as part of its configuration.

## Training data
Retrospective single-center cohort, Emergency Department, adults (>=18 y), Hospital Universitario San
Juan de Alicante. Source export 2020-07-30 to 2026-04-27; analytic cohort n=14,985 (complete-case).
Reference standard: urine culture >=10^4 CFU/mL. Instruments: Sysmex UC-3500 (test strip) and UF-5000
(flow cytometry). **The clinical data are not publicly available (patient-privacy / institutional
restrictions).**

## Performance
- Internal hold-out (n=2248): ROC AUC 0.874; well calibrated (Brier 0.139, ECE 0.025).
- Prospective post-implementation (n=511, routine use, 22 June to 9 August 2026): ROC AUC 0.871
  (95% CI 0.837 to 0.904), sensitivity 0.872, specificity 0.743, MCC 0.623; calibration Brier 0.136,
  ECE 0.039. Decision-curve analysis showed positive net benefit.

## Intended use and limitations
Intended as a **calibrated risk estimate reported alongside the urinalysis** to support (not replace)
clinical judgement. It is **not** a validated rule-out test and **not** a diagnostic model for
symptomatic UTI. Discrimination and calibration in routine use held at the internal hold-out level,
but that was measured in the same population, on the same analyzer platform and against the same
culture-reporting convention; a different setting can shift both case mix and calibration. Local
recalibration, a locally chosen threshold, and ongoing monitoring are required before any
culture-reduction use. External validation on other populations and analyzer platforms is required
before transport.

## Reproduce
`python src/utils/train_and_save_model.py` (retrains and saves the model from the analytic dataset;
requires the restricted data). Deployment preprocessing: see `LIS/lis_preprocess.py`,
`LIS/lis_predict.py`.

## Ethics
Approved by the institutional review board (approval number to be inserted); conducted per the
Declaration of Helsinki.
