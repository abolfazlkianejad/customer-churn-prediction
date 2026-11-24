# SmartPredict – Customer Churn Prediction System

SmartPredict is a production-oriented machine learning project for estimating the
probability that a customer will churn, using the **IBM Telco Customer Churn**
dataset as the reference dataset.

The codebase follows a layered architecture with centralized configuration,
explicit data contracts, reproducible experiments, versioned model artifacts,
a reusable SHAP explainability workflow, Optuna optimization, decision-threshold
analysis, and a FastAPI inference interface. The **foundation, data, model,
explainability, optimization, and serving layers are implemented**:
configuration, ingestion, validation, stratified splitting, reusable
preprocessing, model training/evaluation, comparison, serialization, validation
report persistence, leakage-safe hyperparameter search, threshold selection,
CLI orchestration, SHAP explanations, REST serving, container packaging, and
deployment validation. SHAP, the API, the optimization workflow, and the Docker
container were verified against the real saved artifacts; the deployed model and
its isolated final test report remain unchanged.

> The real IBM Telco CSV is not committed to source control. Place it at
> `data/raw/telco_customer_churn.csv` before running the data or training CLIs.
> This workspace contains the verified real dataset (SHA-256
> `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`).
> Automated tests use synthetic fixtures only. Every metric labelled
> **real data** below was produced from the 7,043-row IBM Telco CSV; no
> synthetic score is presented as real-world performance.

## Current status

| Layer | Status |
| --- | --- |
| Configuration, paths, logging, seeds, exceptions | Implemented |
| Data loading, validation, and stratified splitting | Implemented |
| Reusable sklearn preprocessing pipeline | Implemented |
| Model registry, factory, and training service | Implemented |
| Evaluation metrics, comparison, and selection | Implemented |
| Model artifacts, metadata, and plots | Implemented |
| Training CLI (`python -m smartpredict.cli.train`) | Implemented |
| Real-data preparation, training, evaluation, artifacts | Validated with real data |
| Artifact-based inference smoke test | Implemented and verified |
| Validation reports for every candidate | Implemented and verified with real data |
| FastAPI serving (`/health`, `/predict`, `/predict/batch`, `/model/info`) | Implemented and verified with real data |
| SHAP explainability (global and local) | Implemented and verified with real data |
| Optuna study, training-only CV, validation comparison | Implemented and verified with real data |
| Decision-threshold analysis and saved cutoff | Implemented and verified with real validation data |
| Threshold-aware inference and API responses | Implemented and verified |
| Optimized candidate promotion | Not performed (would require a new isolated test run) |
| Docker image and Compose deployment | Implemented and validated with the real saved artifacts |
| Optimized production image (<1.5 GB) | Implemented and validated; predictions unchanged |
| Deployment tests (synthetic fixtures/mocks) | 19 tests added and passing |

## Architecture

```
config (Pydantic Settings + YAML)
        |
core (paths, logging, seeds, exceptions)
        |
data -> features -> models -> evaluation
        |              |          |
        |       explainability    +--> optimization (Optuna + threshold)
        |                                      |
        +------------- artifact contract ------+
                            |
                           api
```

- **config** – single source of truth for runtime settings and declarative
  experiment configuration.
- **core** – cross-cutting infrastructure with no machine-learning logic.
- **data** – load, validate, and split raw records into typed datasets.
- **features** – build one preprocessing pipeline reused by training and serving.
- **models** – training, serialization, and a filesystem-backed version registry.
- **evaluation** – metrics, comparison, selection, and plots.
- **optimization** – seeded Optuna studies, training-only stratified CV,
  validation-only comparison, and business-constrained threshold selection.
- **explainability** – SHAP reports derived from saved artifacts and validation
  rows only.
- **api** – FastAPI application with health, prediction (saved threshold
  applied), and model-info routes.
- **cli** – thin command-line entry points, including `optimize`.

### Folder structure

```
.
├── src/smartpredict/
│   ├── config/          # settings.py (Pydantic Settings), schema.py, __init__.py
│   ├── core/            # paths.py, logging.py, seed.py, exceptions.py
│   ├── data/            # loader, schema, validator, splitter, pipeline
│   ├── features/        # reusable sklearn preprocessor
│   ├── models/          # registry, factory, training, artifacts, pipeline
│   ├── evaluation/      # metrics, comparison, plots, evaluation service
│   ├── optimization/    # study, CV, search space, threshold, artifacts, pipeline
│   ├── explainability/  # SHAP service using saved artifacts only
│   ├── api/             # app, schemas, dependencies, and routes
│   ├── deployment/      # container health probe
│   └── cli/             # prepare_data, train, optimize, explain, serve
├── tests/               # pytest suite mirroring src/smartpredict
├── configs/             # data.yaml and train.yaml
├── artifacts/           # models, plots, reports, explanations (git-ignored)
├── data/                # raw and processed datasets (git-ignored)
├── notebooks/           # exploration only
├── scripts/             # one-off utility scripts
├── pyproject.toml       # package metadata and tool configuration
├── Dockerfile           # multi-stage production API image
├── docker-compose.yml   # local API deployment with read-only model mount
├── .dockerignore        # build-context exclusions
├── requirements.txt     # pinned project runtime dependencies
├── requirements-runtime.txt # explicit pinned CPU-only image dependencies
└── requirements-dev.txt # pinned test and lint dependencies
```

## Technology stack

Python 3.11+ · NumPy · Pandas · Scikit-learn · XGBoost · Optuna · SQLite ·
Matplotlib · Seaborn · SHAP · FastAPI · Pydantic · Pydantic Settings · PyYAML ·
Uvicorn · Pytest · Ruff · mypy · Docker · Docker Compose

`pyproject.toml` declares compatible version ranges and tool settings;
`requirements*.txt` pin the exact versions used for validation.

## Getting started

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Use `requirements.txt` instead of `requirements-dev.txt` for a runtime-only
installation.

### 3. Configure the environment (optional)

```bash
copy .env.example .env     # Windows
cp .env.example .env       # macOS / Linux
```

All environment variables use the `SMARTPREDICT_` prefix, for example
`SMARTPREDICT_LOG_LEVEL=DEBUG` or `SMARTPREDICT_RANDOM_SEED=7`.

### 4. Run the validation suite

```bash
python -m pytest
python -m ruff check src tests
python -m mypy
```

## Configuration

- `src/smartpredict/config/settings.py` loads and validates environment-based
  settings (paths, logging, seeds, API host/port, model identity).
- `src/smartpredict/config/schema.py` defines immutable, typed configuration
  objects (`DatasetSpec`, `DataConfig`, `FeatureConfig`, `ModelConfig`,
  `TrainingConfig`).
- `src/smartpredict/config/loader.py` reads the YAML files, validates their
  structure, and resolves paths relative to the project root. Column names are
  **never hardcoded in the pipeline code**; they come from `configs/data.yaml`.
- `configs/data.yaml` describes the IBM Telco dataset: 21 expected columns,
  column types, numeric ranges, category domains, and stratified split ratios
  (70/15/15).
- `configs/train.yaml` holds experiment naming, all four model candidates and
  their hyperparameters, preprocessing choices, cross-validation, the primary
  selection metric/direction, secondary metrics, artifact names, and SHAP
  settings (`method`, `sample_size`, `local_examples`, `output_directory`).
  Selection remains validation-only; test data is never used for tuning or
  explanations.
- The `optimization:` block declares the model family, trial/timeout/seed
  settings, metric and direction, `StratifiedKFold` count, study name and SQLite
  storage path, train/validation CSV locations, explicit XGBoost/Random Forest
  search spaces, and the threshold grid/criterion/constraints. Only train and
  validation paths are accepted here; no test file is configured.

### Dataset requirements

### Dataset source and installation

Download the public **IBM Telco Customer Churn** sample (7,043 rows, 21
columns) from the IBM Community dataset page or the Kaggle mirror, then place
the CSV at `data/raw/telco_customer_churn.csv` (or set `dataset.raw_path` in a
custom `data.yaml`).

```bash
# PowerShell verification using the checked-in run fingerprint
Get-FileHash data/raw/telco_customer_churn.csv -Algorithm SHA256
# Expected: 16320C9C1EC72448DB59AA0A26A0B95401046BEF5D02FD3AEB906448E3055E91
```

The file must:

- be a comma-separated CSV with a header containing the 21 configured columns;
- use one of the probed encodings (`utf-8-sig`, `utf-8`, `cp1252`, `latin-1`);
- use the exact target labels `Yes` / `No` and unique values in `customerID`;
- contain at least two rows per target class.

All `Yes` / `No` category values in `configs/data.yaml` are explicitly quoted:
PyYAML would otherwise parse them as booleans. The loader now rejects
YAML booleans instead of silently converting them to `"True"` / `"False"`,
while `SeniorCitizen: [0, 1]` remains an intentional integer-to-string domain.

No dataset is committed to this repository. Tests use synthetic fixtures that
match the declared schema; they are never written into `data/`.

## Data ingestion workflow

1. `config.loader.load_dataset_spec()` validates `configs/data.yaml` and builds
   a `DatasetSpec` (columns, declared types, ranges, categories, split config).
2. `data.loader.load_csv()` checks file existence/extension/size, detects the
   encoding, reads every field as a raw string, normalizes header whitespace,
   and records metadata (path, SHA-256, encoding, row/column counts, raw and
   normalized headers, inferred dtypes, load timestamp).
3. `data.schema.DatasetSchema.from_spec()` turns the specification into a
   runtime schema used by validation and preprocessing.
4. `data.validation.validate_dataset()` coerces declared columns, runs all
   checks, and returns both the coerced frame and a detailed report. It never
   drops or silently repairs rows.
5. `data.splitter.stratified_split()` separates train/validation/test using the
   configured ratios and seed, before any preprocessing statistics are fitted.
6. `data.pipeline.run_data_pipeline()` orchestrates the steps and writes:

   - `data/processed/{train,validation,test}.csv`
   - `artifacts/reports/validation_report.json`
   - `artifacts/reports/split_metadata.json`
   - `artifacts/reports/dataset_metadata.json`

### Validation rules

| Rule | Code | Severity |
| --- | --- | --- |
| Required columns missing | `MISSING_COLUMNS` | error |
| Empty dataset | `EMPTY_DATASET` | error |
| Missing categorical / ID / target values | `MISSING_VALUES` | error |
| Duplicate customer IDs | `DUPLICATE_ID` | error |
| Non-numeric or infinite numeric values | `INVALID_NUMERIC` | error |
| Numeric value outside configured range | `NUMERIC_OUT_OF_RANGE` | error |
| Category outside configured domain | `UNEXPECTED_CATEGORY` | error |
| Target other than `Yes` / `No` | `INVALID_TARGET` | error |
| Column dtype inconsistent with schema | `INCONSISTENT_DTYPE` | error |
| Fewer than two rows per class | `INSUFFICIENT_CLASS_SAMPLES` | error |
| Blank numeric strings | `BLANK_NUMERIC` | warning |
| Undeclared extra columns | `UNEXPECTED_COLUMNS` | warning |

`TotalCharges` is special-cased: blank strings in the IBM Telco export are
converted to missing values and reported as `BLANK_NUMERIC` warnings with row
examples. They are imputed later by the training-only preprocessing pipeline.
Non-numeric tokens such as `abc` remain blocking errors. Reports are written
even when validation fails; with `strict=True` (default) the pipeline stops
before splitting via `DataValidationError`.

## Running the data pipeline

```bash
python -m smartpredict.cli.prepare_data
python -m smartpredict.cli.prepare_data \
  --data-config configs/data.yaml \
  --train-config configs/train.yaml \
  --output-dir data/processed \
  --reports-dir artifacts/reports
```

Use `--allow-invalid` only to inspect a failing dataset; processed splits are
still written in that mode. Split metadata records the seed, ratios, sizes,
class counts, and the exact row index assigned to each split, so runs are
auditable and test isolation can be verified.

### Real-data preparation result (Phase 3)

Running `python -m smartpredict.cli.prepare_data` on the real CSV produced:

| Check | Real-data result |
| --- | --- |
| Rows / columns loaded | 7,043 / 21 |
| Blocking validation errors | 0 |
| Warnings | 1 (`BLANK_NUMERIC`, `TotalCharges`, 11 rows) |
| Rows preserved | 7,043 |
| Split sizes | train 4,930 / validation 1,056 / test 1,057 |
| Split ratios | 0.699986 / 0.149936 / 0.150078 (seed 42, stratified) |
| Class counts (train) | 3,622 `No` / 1,308 `Yes` |
| Class counts (validation) | 776 `No` / 280 `Yes` |
| Class counts (test) | 776 `No` / 281 `Yes` |

All 11 blank `TotalCharges` rows contain a single space, have `tenure = 0`,
and remain in the dataset as missing numeric values for training-only median
imputation. Re-running the split with seed 42 reproduces identical disjoint
row indices covering all 7,043 rows.

## Preprocessing architecture

`features.preprocessor` exposes three functions:

- `build_preprocessor(spec, feature_config)` returns an **unfitted**
  `ColumnTransformer` built from `configs/train.yaml`:

  - numeric columns: `SimpleImputer` (median) → optional `StandardScaler`;
  - categorical columns: `SimpleImputer` (most frequent) → `OneHotEncoder`
    with `handle_unknown="ignore"` (`OrdinalEncoder` is also supported);

  the ID and target columns are dropped by `remainder="drop"`.
- `prepare_features_frame(frame, spec)` converts validated columns into
  sklearn-compatible dtypes.
- `fit_transform_features(train, validation, test, ...)` calls `fit_transform`
  **only on training rows**, then `transform` on validation and test, returning
  aligned arrays, feature names, and the fitted transformer. The fitted object
  is picklable so the exact same pipeline can be reused at inference time.

The preprocessor is fitted once and passed unchanged into training.

## Model training and evaluation

### Supported models

`models.registry.ModelRegistry` registers four classifiers, all configurable in
`configs/train.yaml`:

| Config name | Estimator |
| --- | --- |
| `logistic_regression` | `sklearn.linear_model.LogisticRegression` |
| `decision_tree` | `sklearn.tree.DecisionTreeClassifier` |
| `random_forest` | `sklearn.ensemble.RandomForestClassifier` |
| `xgboost` | `xgboost.XGBClassifier` (optional dependency) |

Missing XGBoost produces a clear `ModelError` and the remaining candidates still
train. Duplicate registrations are rejected unless explicitly replaced.

### Training workflow

`python -m smartpredict.cli.train` performs these steps:

1. Load `data.yaml` and `train.yaml`; validate model names, parameters,
   primary metric, and optimization direction.
2. Check that the real raw CSV exists; load, validate, and stratify it with the
   Phase 2 components.
3. Fit preprocessing **only on the training split** and transform validation
   and test once. The fitted transformer is reused by every candidate.
4. Train each enabled candidate, generate validation predictions, and compute
   accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and a
   classification report. Probability-based metrics are skipped with a warning
   when a model has no `predict_proba` or only one class is present.
5. Rank successful models by `evaluation.primary_metric` and
   `evaluation.optimization_direction`. Failures are reported without losing
   successful results. **Validation data alone selects the best model.**
6. Evaluate the selected model **exactly once** on the isolated test split.
   Test metrics never influence selection or hyperparameters.
7. Persist one validation report per candidate under
   `artifacts/reports/validation_reports/`, then save the selected model
   artifacts and generate the confusion matrix, ROC, and precision-recall
   plots. The isolated final test evaluation remains a separate
   `artifacts/reports/final_test_report.json`.

All four candidates used the same 4,930 × 45 preprocessed training matrix.
The 45 columns are the 4 declared numeric features plus 41 one-hot columns;
`customerID` is dropped by the transformer and never reaches a model.
Imputation statistics, scaling statistics, and encoder categories were fitted
on training rows only; validation and test rows were transformed without
fitting.

### Real-data results (Phase 3)

Validation metrics below are **real data** and were the sole basis for model
selection (`roc_auc`, maximize). Test metrics were computed exactly once after
selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg precision | Brier | Log loss | Fit (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost **(selected)** | 0.7983 | 0.6414 | 0.5429 | 0.5880 | **0.8459** | 0.6431 | 0.1359 | 0.4175 | 0.2022 |
| logistic_regression | 0.8059 | 0.6459 | 0.5929 | 0.6182 | 0.8453 | 0.6310 | 0.1367 | 0.4188 | 0.0236 |
| decision_tree | 0.7926 | 0.6063 | 0.6214 | 0.6138 | 0.8387 | 0.6207 | 0.1395 | 0.4244 | 0.0127 |
| random_forest | 0.7945 | 0.6352 | 0.5286 | 0.5770 | 0.8383 | 0.6312 | 0.1383 | 0.4264 | 0.3220 |

The selected `xgboost` model's **real-data** final test report (1,057 rows):

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7985 |
| Precision | 0.6619 |
| Recall | 0.4947 |
| F1 | 0.5662 |
| ROC-AUC | 0.8376 |
| Average precision | 0.6637 |
| Brier score | 0.1375 |
| Log loss | 0.4238 |

Test confusion matrix `[[TN, FP], [FN, TP]] = [[705, 71], [142, 139]]`.
The full classification report, validation comparison, and plots are stored
in `artifacts/reports/` and `artifacts/plots/`.

> **Threshold-policy caveat:** `artifacts/reports/final_test_report.json` was
> generated before the decision threshold changed to `0.46`. It evaluates the
> original `0.5` decision policy and does **not** evaluate the current API
> labeling policy. It was deliberately left unchanged in Phases 5 and 6; no new
> isolated test evaluation was performed for the 0.46 cutoff or the unpromoted
> optimized candidate.

### Artifacts and overwrite policy

```text
artifacts/
├── models/
│   ├── best_model.joblib
│   ├── preprocessing_pipeline.joblib
│   └── model_metadata.json
├── reports/
│   ├── validation_reports/
│   │   ├── decision_tree_validation_report.json
│   │   ├── logistic_regression_validation_report.json
│   │   ├── random_forest_validation_report.json
│   │   └── xgboost_validation_report.json
│   ├── model_comparison.json
│   ├── validation_report.json
│   ├── split_metadata.json
│   ├── dataset_metadata.json
│   └── final_test_report.json
├── explainability/
│   ├── global_feature_importance.json
│   ├── global_summary_plot.png
│   ├── global_bar_plot.png
│   ├── local_explanations.json
│   ├── local_explanation_00.png
│   ├── local_explanation_01.png
│   └── local_explanation_02.png
└── plots/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── precision_recall_curve.png
```

Existing artifact names are never overwritten by default. Re-run the relevant
CLI with `--overwrite` to intentionally replace them. Each file in
`validation_reports/` keeps the original evaluation JSON fields at the top level
for backward compatibility and adds metadata with the model name, split, model
configuration, experiment/split settings, and creation timestamp. The four real
reports contain 8 metrics, a confusion matrix, and a classification report.
`final_test_report.json` remains the only isolated-test report; it labels the
validation and test sections explicitly and records that selection used
validation data only. Metadata includes the model name/parameters, seed, raw
dataset SHA-256, sizes of all three splits, selected metric direction, selected
model, validation and test metrics, transformed feature names, timestamp, and
software versions.

### Commands

```bash
python -m smartpredict.cli.train
python -m smartpredict.cli.train \
  --data-config configs/data.yaml \
  --train-config configs/train.yaml \
  --models-dir artifacts/models \
  --reports-dir artifacts/reports \
  --plots-dir artifacts/plots
```

The CLI returns a non-zero exit code when the dataset is missing, configuration
is invalid, no model succeeds, or artifacts would be overwritten without
permission.

### Example inference workflow

The smoke test loads the saved artifacts once and never fits anything:

```bash
python scripts/inference_smoke.py --limit 3
```

The same flow is available from Python:

```python
import joblib
import pandas as pd

from smartpredict.config.loader import load_dataset_spec
from smartpredict.models.artifacts import build_artifact_paths
from smartpredict.config.loader import load_training_config
from smartpredict.models.inference import predict_from_raw

spec = load_dataset_spec()
paths = build_artifact_paths(load_training_config().artifacts)
model = joblib.load(paths["best_model"])
preprocessor = joblib.load(paths["preprocessor"])

raw = pd.read_csv(spec.raw_path, dtype=str, keep_default_na=False).head(3)
result = predict_from_raw(raw, spec, preprocessor, model)
print(result.labels, result.probabilities, result.threshold, len(result.feature_names))
```

`predict_from_raw()` validates feature values, applies the saved transformer,
and calls `predict` / `predict_proba`. It skips target-specific rules because
prediction batches need not contain a balanced (or any) `Churn` column; feature
domains, numeric ranges, and missing-value rules still apply. The verified
smoke run produced labels `[1, 0, 0]`, probabilities
`[0.698650, 0.022951, 0.469401]`, and 45 feature names with no `customerID`.
Phase 5 adds an optional `threshold` argument, for example
`predict_from_raw(..., threshold=0.46)`. Positive-class probabilities are
threshold-independent; only labels are recomputed as `probability >= threshold`.
The API reads `decision_threshold.selected_threshold` from
`model_metadata.json` and falls back to `0.5` when no saved value exists.

## SHAP explainability (Phase 4)

`python -m smartpredict.cli.explain` loads `best_model.joblib`,
`preprocessing_pipeline.joblib`, and `model_metadata.json`; it does **not**
retrain or refit anything. It samples raw rows from
`data/processed/validation.csv`, transforms them with the saved pipeline, and
verifies that the transformed feature count matches the saved model's
`n_features_in_` and the saved feature names. The isolated test split is never
used for explanations.

```bash
python -m smartpredict.cli.explain
python -m smartpredict.cli.explain \
  --models-dir artifacts/models \
  --reports-dir artifacts/reports \
  --output-dir artifacts/explainability \
  --sample-size 100 \
  --local-examples 3 \
  --overwrite
```

`configs/train.yaml` supplies the defaults (`method: shap_tree`,
`sample_size: 100`, `local_examples: 3`,
`output_directory: artifacts/explainability`); the CLI flags override them.
`--overwrite` is required once any expected output exists. The global report is
the mean absolute SHAP value across the sampled validation rows, while the local
report contains one contribution per transformed feature for each explained
customer. Global and local outputs are separate JSON files with summary, bar,
and per-customer waterfall plots. SHAP/backend failures are wrapped in
`ExplainabilityError` rather than leaking raw exceptions.

The real-data run explained 100 validation rows with the saved `xgboost` model
and 45 transformed features. The five largest global drivers were
`Contract_Month-to-month` (0.68898), `tenure` (0.44147), `MonthlyCharges`
(0.30284), `OnlineSecurity_No` (0.26503), and `TotalCharges` (0.18329). The
local report contained 3 customers, for example `9945-PSVIP` (label 0,
probability 0.0060), `2325-NBPZG` (label 0, probability 0.4658), and
`4223-BKEOR` (label 0, probability 0.0567), each with 45 feature contributions.
These values explain model behaviour on validation rows; they are not causal
claims and are not final-test metrics.

## FastAPI serving (Phase 4)

`python -m smartpredict.cli.serve` starts Uvicorn with `create_app()`. Routes are
mounted below the configured API prefix (`SMARTPREDICT_API_PREFIX`, default
`/api/v1`). The application lifespan warms and validates the saved artifact
bundle at startup, caches it once per process, and logs explicit startup and
shutdown messages. Startup fails fast when the bundle is missing or incompatible.

| Method | Path | Request body | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/v1/health` | — | Report artifact readiness and loaded model name. |
| `POST` | `/api/v1/predict` | One raw customer record object | Predict one customer. |
| `POST` | `/api/v1/predict/batch` | `{"records": [...]}` | Predict up to 1,000 raw records. |
| `GET` | `/api/v1/model/info` | — | Report selected model, metadata, and transformed feature names. |

```bash
python -m smartpredict.cli.serve --host 0.0.0.0 --port 8000
# Development only:
python -m smartpredict.cli.serve --reload

# Example requests:
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/model/info
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @one_raw_customer.json
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"records": [{...}, {...}]}'
```

Requests are validated by `predict_from_raw()`, so the API contract follows the
declared dataset schema instead of a duplicated 21-column Pydantic model.
Prediction responses include a `threshold` field; Phase 5 loads the saved
validation-selected cutoff from `model_metadata.json` and passes it into
`predict_from_raw()`. Probabilities are unchanged by the cutoff. Invalid raw
values return HTTP 400; missing or unreadable artifacts return HTTP 503. An
in-process smoke test against the real saved artifacts returned HTTP 200 for
all four endpoints, `/model/info` reported `xgboost` with 45 features, one
`/predict` call returned label `[0]` with probability
`[0.006016685627400875]`, and `/predict/batch` returned 2 predictions. A Phase 5
recheck reported `threshold: 0.46` in prediction responses while probabilities
remained identical to the 0.5 baseline. Phase 6 additionally exposes
`decision_threshold` on `/model/info`, sanitizes metadata paths in responses,
and verifies the same 0.46 threshold inside the production container.

## Docker deployment and production readiness (Phase 6)

### Architecture and image contents

The production image is a multi-stage build:

1. **Builder** – installs build tooling only, compiles `smartpredict` into a
   wheel from `pyproject.toml` and `src/`, and never installs development extras.
2. **Runtime** – installs only the pinned serving dependencies listed in
   `requirements-runtime.txt`, installs the wheel with `--no-deps`, copies only
   `configs/`, and runs the API as the non-root `app` user (UID/GID 10001).

The current base image is a pinned production Python slim image through the
Microsoft Container Registry mirror:
`mcr.microsoft.com/mirror/docker/library/python@sha256:2c5bc243b1cc47985ee4fb768bb0bbd4490481c5d0897a62da31b7f30b7304a7`
(`3.11-slim-bookworm`, Python 3.11.14). Direct Docker Hub access to
`python:3.11-slim-bookworm` still fails with registry authorization denial in
this environment; the MCR mirror of the official Python image is accessible and
is **not** a devcontainer image. Alpine was deliberately not used because the
compiled NumPy/SciPy/scikit-learn/XGBoost wheels are glibc/manylinux builds.
The former `mcr.microsoft.com/devcontainers/python:3.11` base remains documented
below only as the pre-optimization Phase 6 baseline.

The image contains no raw datasets, processed splits, tests, notebooks, virtual
environments, caches, `.env` files, credentials, or
training/plotting/optimization libraries. Build tooling exists only in the
builder stage and is uninstalled from the final runtime. Training is never run
during image build or container startup.

Model artifacts stay outside the image. Compose bind-mounts the host
`artifacts/models/` directory read-only at `/app/artifacts/models`, and the API
loads `best_model.joblib`, `preprocessing_pipeline.joblib`, and
`model_metadata.json` together. The loader verifies the selected model name,
feature names/count, preprocessor transform contract, estimator
`predict`/`predict_proba` contract, estimator feature count, and saved threshold
before serving. Startup fails fast with a clear, non-secret error if any check
fails.

### Docker installation

Install Docker Engine or Docker Desktop with the Compose v2 plugin, then verify:

```bash
docker --version
docker compose version
```

### Build and run

```bash
docker compose build api
docker compose up -d api
docker compose ps
docker compose logs -f api
docker compose down
```

The service name is `api`, the container name is `smartpredict-api`, and the
current image tag is `smartpredict-churn-api:optimized`. `restart: unless-stopped`
is configured and no database, queue, or other auxiliary service is introduced.

### Runtime configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `SMARTPREDICT_ENVIRONMENT` | `production` | Deployment environment label. |
| `SMARTPREDICT_API_HOST` | `0.0.0.0` | Uvicorn bind address. |
| `SMARTPREDICT_API_PORT` | `8000` | Container and host API port. |
| `SMARTPREDICT_API_PREFIX` | `/api/v1` | Route prefix. |
| `SMARTPREDICT_CONFIGS_DIR` | `/app/configs` | Runtime YAML directory. |
| `SMARTPREDICT_ARTIFACTS_DIR` | `/app/artifacts` | Artifact root. |
| `SMARTPREDICT_MODELS_DIR` | `/app/artifacts/models` | Explicit model bundle override. |
| `SMARTPREDICT_HOST_MODELS_DIR` | `./artifacts/models` | Compose host-side bind-mount source (not passed into the container). |
| `SMARTPREDICT_LOG_LEVEL` | `INFO` | Application log level. |
| `SMARTPREDICT_LOG_FORMAT` | `json` | Structured logs in the image. |
| `SMARTPREDICT_LOG_TO_FILE` | `false` | Keeps logs on stdout for the container runtime. |
| `SMARTPREDICT_HEALTHCHECK_URL` | derived | Optional probe URL override. |
| `SMARTPREDICT_HEALTHCHECK_TIMEOUT_SECONDS` | `3.0` | Probe timeout. |

Local development is unchanged: copy `.env.example` to `.env` and use the
existing `python -m smartpredict.cli.serve` workflow.

### Model artifact requirements

The host directory mounted at `/app/artifacts/models` must contain:

```text
best_model.joblib
preprocessing_pipeline.joblib
model_metadata.json
```

`model_metadata.json` must retain `model.selected`, `features.count`,
`features.names`, and `decision_threshold.selected_threshold` (`0.46` in the
deployed baseline). The API never refits the estimator or preprocessor, and
probabilities remain independent of the threshold. `/model/info` returns a
sanitized copy of metadata; filesystem paths are reduced to basenames.

### Health check

Both the Dockerfile and Compose define a health check using the dependency-free
probe:

```bash
python -m smartpredict.deployment.healthcheck
```

The probe calls `GET ${SMARTPREDICT_API_PREFIX}/health` on loopback and requires
`status == "ok"` and `artifacts_ready == true`. Manual verification:

```bash
curl http://localhost:8000/api/v1/health
docker inspect --format '{{.State.Health.Status}}' smartpredict-api
```

### Prediction examples

The examples below use synthetic records only; they are not real customer rows.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @one_raw_customer.json

curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d @batch_raw_customers.json
```

Both responses include the applied `threshold` of `0.46`. Probabilities remain
unchanged when the threshold changes; only labels are recomputed as
`probability >= threshold`.

### Automated deployment tests

`tests/test_deployment/` adds 19 tests using synthetic fixtures and mocks only.
They cover application import, artifact-contract loading, health, single and
batch prediction, threshold respect and threshold-independent probabilities,
metadata path sanitization, missing artifacts, invalid feature metadata,
`fit`/`fit_transform` never being called during inference, startup/shutdown
logging, startup failure logging, and the container health probe. No test-set
data is used.

### Actual Docker validation in this workspace

These are observed results, not estimates:

- `docker compose build api` succeeded after starting the Docker Desktop engine.
- Docker Hub returned HTTP 403 for the official Python base image; the
  `mcr.microsoft.com/devcontainers/python:3.11` base was pulled and used.
- `docker compose up -d api` started `smartpredict-api` with `status=running`,
  `health=healthy`, `restarts=0`, and a read-only bind mount of the real
  `artifacts/models` directory.
- `docker inspect` reported the process user as `app` (UID 10001). A fresh
  container file listing showed only `/app/configs/data.yaml`,
  `/app/configs/train.yaml`, and `/app/requirements-runtime.txt` under `/app`,
  with no model artifacts, datasets, tests, or notebooks copied into the image.
- `GET /api/v1/health` returned `status=ok`, `artifacts_ready=true`,
  `model_loaded=true`, `preprocessor_loaded=true`, `model_name=xgboost`.
- `GET /api/v1/model/info` reported `xgboost`, 45 features,
  `decision_threshold=0.46`, and only the basename
  `telco_customer_churn.csv` for `dataset.raw_path`.
- A synthetic single prediction returned label `0` and probability
  `0.2727810740470886`. A synthetic batch of two records returned labels
  `[0, 0]` and probabilities `[0.2727810740470886, 0.10454940050840378]`.
  Both responses reported threshold `0.46`.
- `docker stop -t 10 smartpredict-api` produced the application log
  `SmartPredict API shutdown complete`, Uvicorn's clean shutdown messages, and
  container exit code `0` with `OOMKilled=false`.

These smoke predictions validate the deployed baseline and saved threshold;
they are not model-performance results and do not use the test split.

### Production image optimization (current)

The Phase 6 baseline image was measured at **4.11 GB** with
`docker image ls` (`docker image inspect .Size` reported `1,217,257,697`
bytes). The optimized image is **1.05 GB** (`279,313,976` bytes by
`docker image inspect`), a **74.45%** reduction by CLI-reported size. The
`<1.5 GB` target is met; the preferred `<1 GB` target is not, because the
CPU ML wheels alone occupy roughly 561 MiB of site-packages (XGBoost 208.4
MiB, SciPy 106.6 MiB, pandas 70.0 MiB, scikit-learn 45.8 MiB, NumPy 41.9
MiB). Correctness was not traded for a smaller number.

Optimization changes:

- Both stages use the digest-pinned official Python 3.11 slim image from the
  accessible MCR Docker Hub mirror instead of a devcontainer image.
- `requirements-runtime.txt` now pins every direct and required transitive
  package and is installed with `--no-deps`, so XGBoost's Linux-only
  `nvidia-nccl-cu12` requirement cannot enter a CPU serving image. CPU import,
  artifact loading, and inference were validated without it.
- Removed from the runtime: `nvidia-nccl-cu12`, `GitPython`, `gitdb`, `smmap`,
  `pip`, `setuptools`, and `wheel`. `setuptools`/`wheel` remain only in the
  isolated builder stage with pinned versions.
- SHAP, Matplotlib, Seaborn, Optuna, notebooks, and test/lint packages remain
  absent from the serving image; they were already absent in Phase 6 and are
  not used by the API contract.
- The temporary wheel is deleted after installation, no pip cache is retained,
  and `.dockerignore` excludes `artifacts/` and `*.whl` from the build context.
- The non-root `app` user (UID/GID 10001), read-only model mount,
  configurable `SMARTPREDICT_*` variables, `EXPOSE 8000`, and health check are
  unchanged.

Fresh-container validation after the optimization:

- `smartpredict-churn-api:optimized` built successfully (image ID
  `9b692921e15d`) and a fresh container became `healthy` with exit code `0`.
- `/api/v1/health` reported `model_name=xgboost`, all artifact flags `true`.
- `/api/v1/model/info` reported `xgboost`, 45 features, threshold `0.46`, and
  dataset basename `telco_customer_churn.csv`.
- The same synthetic smoke records produced the exact Phase 6 results:
  single `[0]` / `[0.2727810740470886]`, batch `[0,0]` /
  `[0.2727810740470886, 0.10454940050840378]`. No tolerance was needed.
- Security checks confirmed UID/GID `10001`, mount `Mode:"ro"` / `RW:false`,
  no project tests/notebooks/raw data/credentials in the image, no GPU or
  NVIDIA libraries, and only port `8000/tcp` exposed by the image.
- Host regression suite: `188 passed`; Ruff, Mypy, and `compileall` all
  exited `0`.
- All four protected artifact hashes remained byte-identical; no model,
  preprocessing artifact, threshold, or `final_test_report.json` was modified.

## Optuna optimization and threshold analysis (Phase 5)

`python -m smartpredict.cli.optimize` adds a validation-only tuning phase on top
of the saved Phase 3 model. It never loads `data/processed/test.csv`, never
writes `best_model.joblib`, and does not replace the deployed baseline or
`artifacts/reports/final_test_report.json`. The optimized XGBoost configuration
is persisted as a comparison candidate because promotion would require a fresh
isolated test evaluation, which this phase deliberately avoids.

### Workflow and data isolation

1. Load `configs/data.yaml`, `configs/train.yaml`, the saved baseline model,
   preprocessor, and metadata. Read only `data/processed/train.csv` and
   `data/processed/validation.csv`.
2. Validate the configured search space and create or resume a seeded SQLite
   Optuna study (`TPESampler`, seed 42; `MedianPruner`, 3 startup trials and 2
   warmup steps). A second run without `--resume` fails explicitly when storage
   already exists; failed trials are retained and an all-failed study raises a
   domain error.
3. Each trial suggests only allow-listed parameters. `StratifiedKFold` (5 folds)
   partitions training rows; a new preprocessor is built and fitted inside each
   fold on fold-train rows only, then applied to fold-validation rows. Validation
   and test splits never enter the objective.
4. The configured metric is reported raw. Although direction is YAML-configurable,
   the study objective is always internally maximized; configured minimization
   metrics are negated for trial scores only.
5. The best trial's parameters are merged with fixed baseline parameters, the
   candidate is refit on all training rows, and it is evaluated on validation
   rows beside the unchanged saved baseline.
6. Threshold analysis uses baseline validation probabilities over a configured
   grid that always includes 0.5. The selected cutoff is stored in
   `model_metadata.json` and used by inference/API label generation;
   probabilities remain threshold-independent.
7. Study plots, JSON reports, and the additive metadata update are written under
   `artifacts/optimization/`.

### Configuration and search spaces

| Optimization key | Value used |
| --- | --- |
| `model_type` | `xgboost` (`random_forest` space is also configured but inactive) |
| `n_trials` / `timeout_seconds` | `12` / `900` seconds |
| `seed` | `42` (TPE sampler and estimator fallback) |
| `metric` / `direction` | `roc_auc` / `maximize` |
| `cv_folds` | `5` stratified folds |
| `study_name` / `storage_file` | `churn_xgboost_optuna` / `optuna_study.db` |
| `output_directory` | `artifacts/optimization` |
| `resume` | `false`; CLI `--resume` is required to reuse storage |
| `data_directory` | `data/processed` |
| `train_file` / `validation_file` | `train.csv` / `validation.csv` |

| Threshold key | Value used |
| --- | --- |
| `minimum` / `maximum` / `step` | `0.05` / `0.95` / `0.01` |
| `criterion` | `weighted_score` |
| `weights` | precision `0.25`, recall `0.50`, F1 `0.25` |
| `min_precision` | `0.60` |
| `min_recall` / `min_specificity` | `0.0` / `0.0` |

Active XGBoost search space:

| Parameter | Distribution |
| --- | --- |
| `n_estimators` | integer 100–600, step 50 |
| `max_depth` | integer 3–8 |
| `learning_rate` | float 0.01–0.20, log scale |
| `subsample` | float 0.60–1.00, step 0.05 |
| `colsample_bytree` | float 0.60–1.00, step 0.05 |
| `min_child_weight` | integer 1–10 |
| `gamma` | float 0.00–1.00, step 0.05 |
| `reg_alpha` | float 0.00–1.00, step 0.05 |
| `reg_lambda` | float 0.50–5.00, step 0.50 |

The optional Random Forest space covers `n_estimators` 200–800 (step 50),
`max_depth` 5–20, `min_samples_leaf` 1–10, and `max_features` choices
`sqrt` / `log2`. Only model parameters declared in YAML are suggested; all other
baseline settings remain fixed.

### Commands

```bash
python -m smartpredict.cli.optimize --help
python -m smartpredict.cli.optimize
python -m smartpredict.cli.optimize --output-dir artifacts/optimization_review
python -m smartpredict.cli.optimize \
  --n-trials 20 --timeout-seconds 300 --seed 7 \
  --metric roc_auc --direction maximize --model-type xgboost --cv-folds 5
python -m smartpredict.cli.optimize --resume
```

### Real-data results (Phase 5)

The real run used 4,930 training and 1,056 validation rows from the verified IBM
Telco dataset.

| Completed | Failed | Pruned | Duration | Best raw CV ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 0 | 0 | 11.36 s | 0.8483459498 |

Best searched parameters (merged with the fixed baseline settings):

| Parameter | Value |
| --- | ---: |
| `colsample_bytree` | 0.8 |
| `gamma` | 0.35 |
| `learning_rate` | 0.0198524734 |
| `max_depth` | 3 |
| `min_child_weight` | 8 |
| `n_estimators` | 400 |
| `reg_alpha` | 0.55 |
| `reg_lambda` | 4.5 |
| `subsample` | 0.8 |

Validation-only comparison (optimized candidate minus baseline):

| Metric | Baseline | Optimized candidate | Difference |
| --- | ---: | ---: | ---: |
| Accuracy | 0.7983 | 0.8011 | +0.0028 |
| Precision | 0.6414 | 0.6522 | +0.0108 |
| Recall | 0.5429 | 0.5357 | -0.0071 |
| F1 | 0.5880 | 0.5882 | +0.0002 |
| ROC-AUC | 0.8459 | 0.8514 | +0.0055 |
| Average precision | 0.6431 | 0.6522 | +0.0091 |
| Brier score (lower better) | 0.1359 | 0.1337 | -0.0022 |
| Log loss (lower better) | 0.4175 | 0.4111 | -0.0064 |

These are mixed validation results: ROC-AUC, average precision, accuracy, F1,
Brier score, and log loss moved slightly in a favourable direction, but recall
decreased. The optimized candidate is **not** the deployed model, and no test
metric was computed for it.

### Decision-threshold analysis

The configured `weighted_score` balances precision 0.25, recall 0.50, and F1
0.25 among thresholds satisfying `min_precision: 0.60`. The 0.05–0.95 grid
produced 91 rows including the preserved 0.5 baseline, of which 49 were
eligible; no fallback was needed.

| Metric | 0.5 baseline | Selected 0.46 |
| --- | ---: | ---: |
| Precision | 0.6414 | 0.6092 |
| Recall | 0.5429 | 0.5679 |
| F1 | 0.5880 | 0.5878 |
| Accuracy | 0.7983 | 0.7888 |
| Specificity | 0.8905 | 0.8686 |
| Selection score | 0.5788 | 0.5832 |

Compared with 0.5, the selected cutoff raises recall by 0.0250 but lowers
precision by 0.0322, accuracy by 0.0095, specificity by 0.0219, and F1 by
0.0002. It was chosen solely by the configured validation criterion. The saved
value changes labels through `probability >= threshold`; probabilities
themselves are unchanged.

### Optimization artifacts

```text
artifacts/optimization/
├── optuna_study.db                  # resumable SQLite study
├── optuna_study.json                # trial snapshots, best parameters, CV result
├── best_parameters.json             # searched and merged parameters
├── optimization_history.png
├── parameter_importance.png
├── threshold_analysis.json          # full grid, metrics, selection outcome
├── threshold_precision_recall.png
├── threshold_f1.png
└── model_comparison.json            # validation-only baseline vs candidate
```

`artifacts/models/model_metadata.json` is updated **additively**: the previous
`final_test` section, model identity, features, and split sizes are retained,
while new `optimization` and `decision_threshold` sections record the study,
candidate, CV score, validation metrics, and selected cutoff. A second seeded
run in a separate output directory was machine-compared with the original: all
12 trial parameters/values, the best parameters, the best CV score, and the
selected threshold matched. The `final_test_report.json` SHA-256 remained
`C9FC11B83BB251A6BF1594E67A3E4F9A7C04190B8DBCD6EDCB43F788F57B5713` before and
after Phase 5.

## Reproducibility and model versioning

- A single `random_seed` setting (default `42`) drives Python and NumPy RNGs;
  estimator-level `random_state` values come from `configs/train.yaml`, with the
  experiment seed used as the fallback.
- The raw dataset fingerprint and split metadata make each run auditable.
- The selected model and the exact fitted preprocessor are serialized together
  as separate joblib files and described by `model_metadata.json`.
- Optuna uses the same seed for the TPE sampler, estimator fallback, and
  shuffled `StratifiedKFold`; the SQLite study and full trial snapshots make a
  run resumable and auditable.
- The threshold grid, criterion, constraints, and selected cutoff are versioned
  in `model_metadata.json`; inference defaults to 0.5 if the saved field is
  absent.

## Roadmap

1. ~~Data schema validation, splitting, and reusable preprocessing.~~ Done.
2. ~~Model registry, training, evaluation, selection, and artifacts.~~ Done.
3. ~~Provide the real IBM Telco CSV and run the training CLI for real metrics.~~ Done.
4. ~~SHAP explainability with global, local, and plot outputs.~~ Done.
5. ~~FastAPI endpoints and HTTP integration tests.~~ Done.
6. ~~Optuna optimization, validation comparison, and threshold analysis.~~ Done.
7. ~~Docker image and container-level deployment checks.~~ Done.

## Verification commands used for the Phase 4 and Phase 5 runs

```bash
python -m smartpredict.cli.prepare_data
python -m smartpredict.cli.train --overwrite
python -m smartpredict.cli.explain --overwrite
python -m smartpredict.cli.optimize --output-dir artifacts/optimization_review
python -m smartpredict.cli.optimize --resume
python scripts/inference_smoke.py --limit 3
python -m pytest
python -m ruff check src tests
python -m mypy
python -m compileall -q src tests
python -m smartpredict.cli.prepare_data --help
python -m smartpredict.cli.train --help
python -m smartpredict.cli.optimize --help
python -m smartpredict.cli.explain --help
python -m smartpredict.cli.serve --help
docker compose build api
docker compose up -d api
python -m smartpredict.deployment.healthcheck
docker compose down
```

The real-data validation still shows the same split sizes and selection
(`xgboost` by validation `roc_auc`). `train --overwrite` regenerated the
candidate validation reports under `artifacts/reports/validation_reports/` and
the final test report separately. `explain --overwrite` regenerated SHAP
artifacts under `artifacts/explainability/` from the saved model and
preprocessor using validation rows only. A real in-process API smoke test
returned 200 responses for `/api/v1/health`, `/api/v1/model/info`,
`POST /api/v1/predict`, and `POST /api/v1/predict/batch`. `--overwrite` flags
were used only because artifacts already existed; overwrite guards remain active
by default.

The Phase 5 run completed 12 trials with no failures and wrote every documented
optimization artifact. A second seeded run in a separate output directory was
machine-compared for identical trial parameters/values, best parameters, best CV
score, and selected threshold. The saved 0.46 cutoff changed 4 of 100 real
validation labels while leaving probabilities unchanged, and the API returned
HTTP 200 for all four endpoints with the saved threshold in prediction
responses. SHAP regeneration and the inference smoke test both succeeded after
metadata integration. The 169-test suite, Ruff, mypy, `compileall`, and all CLI
`--help` checks passed; the isolated final test report hash was unchanged.

### Phase 6 verification summary

The final Phase 6 checks were run after all deployment changes:

- `python -m pytest`: **188 passed** (169 prior tests plus 19 synthetic
  deployment tests), with only pre-existing dependency/experimental warnings.
- `python -m ruff check src tests`: passed.
- `python -m mypy`: passed (55 source files).
- `python -m compileall -q src tests`: passed with no output.
- All five CLI `--help` commands (`prepare_data`, `train`, `optimize`, `explain`,
  `serve`) exited successfully.
- The final Docker rebuild and container health recheck succeeded with
  `health=healthy`, `model=xgboost`, 45 features, and threshold `0.46`.

Artifact integrity was verified by comparing SHA-256 values recorded before and
after Phase 6:

| Artifact | SHA-256 (same before and after) |
| --- | --- |
| `best_model.joblib` | `92B3F6D6975A1582226A3DE16C0C29D1FFF92BD11035D9F54A61B638474000C2` |
| `preprocessing_pipeline.joblib` | `56166A0A2AC2B0131872AF697A91A8428CBB0803E4CE7734B1C7A08ACDEA448F` |
| `model_metadata.json` | `7FC7CEFC65F6C906EB65D8E3FC5162D92E71C0453E2E7C4C0245062652BC008F` |
| `final_test_report.json` | `C9FC11B83BB251A6BF1594E67A3E4F9A7C04190B8DBCD6EDCB43F788F57B5713` |

Local re-evaluation of the saved artifacts on the same synthetic smoke records
reproduced the container labels and probabilities exactly, confirming that
Phase 6 did not alter model weights, preprocessing, probabilities, or the saved
0.46 decision policy. All nine optimization artifacts under
`artifacts/optimization/` remain present, and `model_metadata.json` still records
`decision_threshold.selected_threshold = 0.46` with
`optimized_candidate_deployed = false`.

