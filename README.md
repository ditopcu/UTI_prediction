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
src/            analysis and results pipeline (uti_01..uti_27); utils/ helpers; colab/ reference
plot_styles/    shared manuscript figure style module
LIS/            deployment package (preprocess / predict / run); *.py only
models/         deployed model (model_optuna.cbm), baseline, scaler, metadata
requirements.txt / MODEL_CARD.md / LICENSE / README.md
```
The Colab notebook versions of the pipeline are kept out of the repository; the equivalent
scripts are `src/colab/uti_ml_pipeline_v2_last.py` (development) and `src/uti_06_gram_prediction.py`.

Not included (restricted, third-party or regenerable): clinical data (`data/`), the manuscript
(`paper/`), the AutoGluon benchmark artifact, figures, and copyrighted writing-style references.
See `.gitignore`.

## Setup
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
```
Python 3.12.10.

## Reproduce
- Train and save the model: `python src/utils/train_and_save_model.py` (requires the restricted data).
- Result tables/figures: `src/uti_09` .. `src/uti_27` (each loads the saved model; no retraining).

## Data availability
The clinical data cannot be shared publicly owing to patient-privacy and institutional restrictions.
The analysis code and the trained model are provided here; the model is derived from restricted data
and is released for research transparency.

## Citation
To be completed on acceptance (DOI). This repository will be archived on Zenodo.

## License
MIT (see `LICENSE`). This covers the code and the released model files.
