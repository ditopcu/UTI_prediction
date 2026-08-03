# Urinalysis-only machine-learning prediction of urine-culture positivity

Code and deployed model for a laboratory-embedded machine-learning system that predicts a positive
urine culture from routine automated urinalysis and demographics, and is integrated into routine
laboratory reporting through a clinical decision-support rule engine.

This repository accompanies the manuscript *"A urinalysis-only machine-learning model for urinary
tract infection prediction: integration into routine laboratory reporting"* (in submission).

## What it does
- Predicts the probability of a positive (pathological) urine culture (>=10^4 CFU/mL) using only
  automated urinalysis parameters and age/sex; no clinical variables.
- Final model: CatBoost tuned with Optuna; features selected with Boruta; explained with TreeSHAP.
- Deployed via a Python service behind an AlinIQ CDS rule engine into the Gestlab laboratory
  information system, reporting a probability, a binary call (threshold 0.50), and an interpretive
  comment with every eligible urinalysis.

## Key results
- Internal hold-out (n=2248): ROC AUC 0.874, well calibrated (Brier 0.139, ECE 0.025).
- Matched or exceeded the previously deployed two-model hospital system while extending decision
  coverage from 55.5% to 100% of samples with a complete urinalysis.
- Prospective post-implementation (n=190): ROC AUC 0.761, high sensitivity; positive net benefit by
  decision-curve analysis. Reported per TRIPOD+AI.

See `MODEL_CARD.md` for full model details, intended use, and limitations.

## Repository layout
```
src/uti_01_dataset_clean.py        preprocessing record (Colab export; see note in the file)
src/utils/save_scaler.py           locally runnable cleaning + StandardScaler fitting
src/utils/train_and_save_model.py  Boruta selection, Optuna search, final training, model export
LIS/                               deployment package (preprocess / predict / batch runner)
models/                            deployed model (model_optuna.cbm), baseline, scaler, metadata
requirements.txt / MODEL_CARD.md / LICENSE / README.md
```
The repository deliberately ships only the path from the raw data to a trained, deployed model.
The scripts that generated the manuscript's tables, figures and secondary analyses are not
included: they operate on the restricted clinical data and therefore cannot be run by anyone
outside the institution. The reported results are documented in the manuscript and its
supplement, and the corresponding code can be requested from the corresponding author.

Also not included: clinical data (`data/`), the manuscript itself, the AutoGluon benchmark
artifact, and third-party copyrighted writing references. See `.gitignore`.

## Setup
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
```
Python 3.12.10.

## Reproduce
With the analysis dataset in place under `data/` (or `UTI_DATA_DIR` pointing at it):
- Fit and export the scaler used by the deployment path: `python src/utils/save_scaler.py`
- Retrain and save the model end to end: `python src/utils/train_and_save_model.py`
  (Boruta selection, 30-trial Optuna search, final CatBoost fit, writes `models/model_optuna.cbm`
  and `models/uti_models_meta.json`).

To score new samples with the released model without retraining, use the deployment path:
`LIS/lis_preprocess.py` then `LIS/lis_predict.py` (`LIS/run_predictions.py` runs a batch).

## Data availability
The clinical data cannot be shared publicly owing to patient-privacy and institutional restrictions.
The analysis code and the trained model are provided here; the model is derived from restricted data
and is released for research transparency.

## Citation
To be completed on acceptance (DOI). This repository will be archived on Zenodo.

## License
MIT (see `LICENSE`). This covers the code and the released model files.
