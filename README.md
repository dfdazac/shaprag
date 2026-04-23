Analysis of the data presented in [Jaspers, Yorrick RJ, et al. "Lipidomic biomarkers in plasma correlate with disease severity in adrenoleukodystrophy." Communications Medicine 4.1 (2024): 175.](https://www.nature.com/articles/s43856-024-00605-9).

## Overview

This repository provides code to predict the presence of adrenal insufficiency from lipidomics data and to interpret the resulting models:
- **Single experiments**: nested cross-validation with feature selection and SHAP for one model configuration (`src/predict.py`).
- **Experiment sweeps**: run many configurations in parallel (`run_experiments.py`).
- **Result summaries**: pick the best configuration per model and aggregate SHAP importances (`src/print_best_per_model.py`, `src/aggregate_importances.py`).
- **Interactive web app**: explore SHAP results and enrich them with online databases (`src/app_shap.py`).

Models supported: Random Forest, LightGBM, CatBoost, XGBoost, and TabPFN.

## Installation

- **Python**: recommended Python ≥ 3.10.
- **Install dependencies**:

```bash
pip install -r requirements.txt
pip install scikit-learn imbalanced-learn optuna shap matplotlib catboost lightgbm xgboost tabpfn tqdm
```

## Data

- Place the main Excel file as:
  - `data/SupplementaryData1-with-age.xlsx` (sheet `lipidomics_data_males` is used).
- VLCFA-only analyses additionally use:
  - `data/vlcfas.csv` (list of VLCFA lipid columns).

## 1. Single experiment (`src/predict.py`)

This script runs nested cross-validation, performs feature selection, fits the chosen model, and computes SHAP values.

```bash
python src/predict.py \
  --model_type {rf,lightgbm,catboost,xgboost,tabpfn} \
  --k 100 \
  --num_trials 30 \
  --imputer {knn,min5} \
  [--normalize] \
  [--exclude_controls] \
  [--vlcfas_only]
```

- **`--model_type`**: classifier to use (default: `lightgbm`).
- **`--k`**: number of top features selected by mutual information (default: `100`).
- **`--num_trials`**: Optuna trials for hyperparameter tuning (ignored for TabPFN).
- **`--imputer`**: missing-value imputer: `knn` (default) or `min5`.
- **`--normalize`**: apply z-score normalization after imputation.
- **`--exclude_controls`**: drop rows where `Presence of adrenal insufficiency` is `Control`.
- **`--vlcfas_only`**: restrict features to VLCFA lipids listed in `data/vlcfas.csv` plus age.

**Outputs** (per run, in `experiments/YYYY-MM-DD-HHMMSS-<hash>/`):
- `log.json`: configuration, per-fold metrics, and 95% CIs.
- `{model_type}_{fold}_shap_summary.png`: SHAP summary plot for each outer fold (non-TabPFN models).
- `{model_type}_{fold}_shap_feature_importance.csv`: mean |SHAP| per feature.
- `instance_shap_table.csv`: per-sample SHAP values for all lipids and folds (used by the web app).

## 2. Multiple experiments (`run_experiments.py`)

This script sweeps over many model/feature/imputer settings by repeatedly calling `src/predict.py` in parallel.

```bash
python run_experiments.py
```

- The grid is defined at the top of `run_experiments.py` (`k_values`, `model_types`, `normalize_options`, `imputer_options`).
- At most `MAX_PARALLEL` jobs run concurrently (default: 4).
- Each run creates its own subfolder under `experiments/` with the same structure as above.

## 3. Analysing experiment results

### 3.1 Best-performing configuration per model (`src/print_best_per_model.py`)

Given a directory containing many experiment folders (e.g. `experiments/v4`), this script finds, for each model type, the run with the highest mean ROC AUC.

```bash
# From the repository root
python src/print_best_per_model.py experiments/v4 --k 100
```

- **Positional `base_dir`**: folder containing experiment subdirectories (default: `.`).
- **`--k`** (optional): only include runs with this `k` value.

It prints, per model:
- Mean ROC AUC and 95% CI, and mean PR AUC and 95% CI.
- The hyperparameters (`k`, `normalize`, `imputer`, `exclude_controls`, `vlcfas_only`).
- The folder name of the best run.

### 3.2 Aggregating SHAP importances (`src/aggregate_importances.py`)

Aggregate and visualise SHAP feature importances from one or more experiment folders.

```bash
# Example with several experiment folders
python src/aggregate_importances.py \
  experiments/v4/2025-08-31-235806-1e9e1f \
  [more folders ...] \
  --top-k 50
```

- **Positional `folders`**: one or more experiment directories containing `*_shap_feature_importance.csv` files.
- **`--top-k`**: number of most important features to include in the heatmaps (optional).

The script:
- Aggregates SHAP importances across folds and folders.
- Annotates features with missing-value statistics from `data/SupplementaryData1.xlsx`.
- Plots heatmaps of SHAP importance and missing counts, coloured by lipid class.
- Saves `output/YYYY-MM-DD-HHMMSS/aggregated_importances.csv` with overall importance per feature.

## 4. Web app for SHAP-based enrichment (`src/app_shap.py`)

The Streamlit app lets you interactively explore per-lipid SHAP results and enrich them with online databases (RefMet, Metabolomics Workbench, KEGG) and an optional language-model summary.

### 4.1 Prerequisites

- At least one completed experiment folder with `instance_shap_table.csv` (from `src/predict.py`).
- In `src/app_shap.py`, set the `exp_dir` variable to point to that folder, for example:

```python
exp_dir = "experiments/v4/2025-08-31-235806-1e9e1f"
```

- Environment variables (optional but recommended):

```bash
# Required for language-model summaries
export OPENAI_API_KEY="your-openai-key"

# Optional for OpenAI-compatible providers; must be the API base URL
# Example:
export LLM_API_URL="https://your-llm-provider.example.com/v1"
```

### 4.2 Running the app

From the repository root:

```bash
streamlit run src/app_shap.py
```

The app will:
- Show the top lipids by mean |SHAP| across folds for the selected experiment.
- Allow selection of a lipid and fold to inspect per-sample SHAP values and correlations with other lipids.
- Query RefMet, Metabolomics Workbench, and KEGG for functional context.
- Optionally generate a concise, text-based interpretation using the OpenAI API.
